"""Run native MMSU references and paired temporal conditions in each worker."""
from __future__ import annotations

import inspect
import json
import os
from pathlib import Path
from time import perf_counter
import runpy


def condition_execution_order(conditions, doc_index):
    """Cyclically balance condition timing positions without changing membership."""
    conditions = list(conditions)
    if not conditions:
        return []
    offset = int(doc_index) % len(conditions)
    return conditions[offset:] + conditions[:offset]


def main():
    import numpy as np
    import torch
    import transformers
    from lmms_eval.tasks.mmsufinal.utils import mmsufinal_process_results
    from mllm_comp.integrations.lmms_eval import plugin

    from .controller import TemporalController, compact_audio_request
    from .metrics import jsonable
    from .observer import TemporalObserver
    from .ltbm_observer import LTBMProvenanceObserver

    settings = json.loads(Path(os.environ["AUDIO_TEMPORAL_CONFIG"]).read_text())
    output_dir = Path(settings["output_dir"])
    rank = int(os.environ.get("RANK", "0"))
    original_bridge = plugin.LMMSEvalRequestBridge
    sessions = []
    if transformers.__version__ != "4.57.3":
        raise RuntimeError(
            f"Temporal hooks target Transformers 4.57.3; found {transformers.__version__}")

    class TemporalBridge(original_bridge):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            tower = self.runtime.model.thinker.audio_tower
            extractor = self.runtime.processor.feature_extractor
            if len(tower.layers) != settings["encoder_layers"]:
                raise ValueError("Runtime audio encoder differs from the frozen analysis plan")
            observer_type = (LTBMProvenanceObserver if settings["experiment"] == "provenance"
                             else TemporalObserver)
            self.observer = observer_type(
                tower,
                sample_rate=extractor.sampling_rate,
                hop_length=extractor.hop_length,
                layers=settings["layers"],
                lags_ms=settings["lags_ms"],
                region_lag_ms=settings["region_lag_ms"],
                prediction_grid=[condition for condition in settings["conditions"]
                                 if condition["family"] == "predictability"])
            self.controller = (None if settings["experiment"] == "provenance" else
                               TemporalController(
                                   tower,
                                   sample_rate=extractor.sampling_rate,
                                   hop_length=extractor.hop_length))
            self.observer.install()
            if self.controller:
                self.controller.install()
            self.completed = 0
            self.samples_file = (output_dir / f"samples.rank{rank}.jsonl").open(
                "x", encoding="utf-8")
            self.conditions_file = ((output_dir / f"conditions.rank{rank}.jsonl").open(
                "x", encoding="utf-8") if settings["conditions"] else None)
            sessions.append(self)
            (output_dir / f"runtime.rank{rank}.json").write_text(json.dumps({
                "torch": torch.__version__,
                "transformers": transformers.__version__,
                "audio_source": inspect.getfile(type(tower)),
                "audio_layers": len(tower.layers),
                "n_window": tower.n_window,
                "audio_backend": tower.config._attn_implementation,
                "dtype": str(tower.dtype),
                "sampling_rate": extractor.sampling_rate,
                "hop_length": extractor.hop_length,
                "experiment": settings["experiment"],
                "compression_mode": ("native_ltbm" if self.controller is None
                                     else "none_plus_request_local_analysis"),
            }, indent=2), encoding="utf-8")

        def _timed(self, callback, kwargs):
            device = next(self.runtime.model.thinker.audio_tower.parameters()).device
            if device.type == "cuda":
                torch.cuda.synchronize(device)
                torch.cuda.reset_peak_memory_stats(device)
            started = perf_counter()
            prediction = callback(kwargs)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
                peak = int(torch.cuda.max_memory_allocated(device))
            else:
                peak = None
            return prediction, perf_counter() - started, peak

        def _generate_sample(self, sample, **kwargs):
            doc = self._document(kwargs["task"], kwargs["split"], kwargs["doc_id"])
            doc_index = int(kwargs["doc_id"])
            seed = settings["seed"] + doc_index
            example = doc_index in settings["example_doc_ids"]
            actual_generation, _ = self._generation_kwargs(kwargs["request_generation"])
            generation_config = self.runtime.model.thinker.generation_config
            if (actual_generation.get("thinker_do_sample", generation_config.do_sample)
                    or actual_generation.get("thinker_num_beams", generation_config.num_beams) != 1):
                raise ValueError("Paired temporal experiments require deterministic generation")

            callback = lambda call_kwargs: super(TemporalBridge, self)._generate_sample(
                sample, **call_kwargs)
            if settings["experiment"] == "provenance":
                self.observer.begin(example=example, seed=seed)
                prediction, seconds, peak_memory = self._timed(callback, kwargs)
                session = self.runtime.metadata.get("_compression_session")
                records = tuple(getattr(session, "active_records", ()))
                if len(records) != 1:
                    raise RuntimeError("Expected exactly one active native LTBM stage")
                report = records[0].activation.last_report
                measurements, arrays = self.observer.finish(report)
                accuracy = mmsufinal_process_results(doc, [prediction])["accuracy"]["score"]
                metadata = {
                    key: doc.get(key) for key in (
                        "id", "audio_path", "question_type", "sub_category",
                        "task_name", "dataset_name")}
                metadata.update(
                    doc_index=doc_index, task=kwargs["task"], split=kwargs["split"], rank=rank)
                reference = {
                    **metadata,
                    "experiment": "provenance",
                    "prediction": prediction,
                    "accuracy": accuracy,
                    "diagnostic_generation_seconds": seconds,
                    "peak_gpu_memory_bytes": peak_memory,
                    **measurements,
                }
                if example:
                    name = f"example.rank{rank}.doc{doc_index}.npz"
                    np.savez_compressed(output_dir / name, **arrays)
                    reference["example_file"] = name
                self.samples_file.write(json.dumps(
                    jsonable(reference), ensure_ascii=False, allow_nan=False) + "\n")
                self.samples_file.flush()
                self.completed += 1
                return prediction
            self.controller.begin(seed=seed)
            self.observer.begin(example=example, seed=seed)
            prediction, seconds, peak_memory = self._timed(callback, kwargs)
            measurements, arrays = self.observer.finish()
            controller_reference = self.controller.finish()
            accuracy = mmsufinal_process_results(doc, [prediction])["accuracy"]["score"]
            metadata = {
                key: doc.get(key) for key in (
                    "id", "audio_path", "question_type", "sub_category",
                    "task_name", "dataset_name")}
            metadata.update(
                doc_index=doc_index, task=kwargs["task"], split=kwargs["split"], rank=rank)
            reference = {
                **metadata,
                "experiment": settings["experiment"],
                "prediction": prediction,
                "accuracy": accuracy,
                "diagnostic_generation_seconds": seconds,
                "peak_gpu_memory_bytes": peak_memory,
                "controller_reference": controller_reference,
                **measurements,
            }
            if example:
                name = f"example.rank{rank}.doc{doc_index}.npz"
                np.savez_compressed(output_dir / name, **arrays)
                reference["example_file"] = name
            self.samples_file.write(json.dumps(
                jsonable(reference), ensure_ascii=False, allow_nan=False) + "\n")
            self.samples_file.flush()

            # Rotate a frozen condition list by document so timing is not
            # systematically confounded with warm-up/execution position.
            ordered_conditions = condition_execution_order(
                settings["conditions"], doc_index)
            for evaluation_order, condition in enumerate(ordered_conditions):
                device = next(self.controller.tower.parameters()).device
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                    torch.cuda.reset_peak_memory_stats(device)
                preparation_started = perf_counter()
                self.controller.begin(condition, seed=seed, retain_trace=example)
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                    preparation_peak = int(torch.cuda.max_memory_allocated(device))
                else:
                    preparation_peak = None
                preparation_seconds = perf_counter() - preparation_started
                call_kwargs = {**kwargs, "cache_hook": None}
                with compact_audio_request(self, self.controller):
                    answer, condition_seconds, condition_peak = self._timed(callback, call_kwargs)
                details = self.controller.finish()
                score = mmsufinal_process_results(doc, [answer])["accuracy"]["score"]
                row = {
                    **metadata,
                    "experiment": settings["experiment"],
                    "condition_id": condition["id"],
                    "condition": condition,
                    "evaluation_order": evaluation_order,
                    "prediction": answer,
                    "accuracy": score,
                    "baseline_accuracy": accuracy,
                    "accuracy_delta": score - accuracy,
                    "analysis_preparation_seconds": preparation_seconds,
                    "diagnostic_generation_seconds": condition_seconds,
                    "diagnostic_total_seconds": preparation_seconds + condition_seconds,
                    "peak_gpu_memory_bytes": (max(preparation_peak, condition_peak)
                                              if preparation_peak is not None else condition_peak),
                    "intervention": details,
                }
                self.conditions_file.write(json.dumps(
                    jsonable(row), ensure_ascii=False, allow_nan=False) + "\n")
                self.conditions_file.flush()
            self.completed += 1
            # Only the untouched answer enters lmms-eval's native outputs and cache.
            return prediction

        def close_temporal(self):
            if self.controller:
                self.controller.close()
            self.observer.close()
            self.samples_file.close()
            if self.conditions_file:
                self.conditions_file.close()

    plugin.LMMSEvalRequestBridge = TemporalBridge
    try:
        runpy.run_module("lmms_eval", run_name="__main__")
    finally:
        plugin.LMMSEvalRequestBridge = original_bridge
        for session in sessions:
            session.close_temporal()


if __name__ == "__main__":
    main()
