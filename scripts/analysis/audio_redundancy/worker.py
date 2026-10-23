"""Native lmms-eval process with temporary task path and request hooks."""
from contextlib import contextmanager
from dataclasses import replace
import inspect
import json
import os
from pathlib import Path
import runpy
from time import perf_counter

from .common import write_json, write_row


class _MutableRunner:
    """Expose a writable generate method while retaining the native runner."""

    def __init__(self, runner):
        self._runner = runner
        self.generate = runner.generate

    def __getattr__(self, name):
        return getattr(self._runner, name)


@contextmanager
def mutable_compact_audio_request(bridge, controller, compact_audio_request):
    """Allow B/C/E token compaction through the existing frozen runner facade."""
    if controller.keep_indices is None:
        with compact_audio_request(bridge, controller):
            yield
        return
    facade = bridge.facade
    bridge.facade = replace(facade, runner=_MutableRunner(facade.runner))
    try:
        with compact_audio_request(bridge, controller):
            yield
    finally:
        bridge.facade = facade


def condition_order(conditions, index):
    if not conditions:
        return []
    offset = index % len(conditions)
    return conditions[offset:] + conditions[:offset]


def main():
    import numpy as np
    import torch
    import transformers
    from lmms_eval.tasks import TaskManager
    from lmms_eval.tasks.mmsufinal.utils import mmsufinal_process_results
    from mllm_comp.integrations.lmms_eval import plugin
    from scripts.analysis.audio_temporal.controller import compact_audio_request
    from .calibration import ModelBank
    from .capture import RepresentationCapture
    from .controller import RedundancyController
    from .measurements import observe

    settings = json.loads(Path(os.environ["AUDIO_REDUNDANCY_CONFIG"]).read_text())
    phase = os.environ["AUDIO_REDUNDANCY_PHASE"]
    root = Path(settings["output_dir"])
    output = root / phase
    rank = int(os.environ.get("RANK", "0"))
    if transformers.__version__ != "4.57.3":
        raise RuntimeError(f"These native Qwen hooks target Transformers 4.57.3; found {transformers.__version__}")
    torch.set_num_threads(settings["worker_threads"])
    original_bridge, original_task_config = plugin.LMMSEvalRequestBridge, TaskManager._get_config
    sessions = []

    def task_config(manager, name):
        raw = original_task_config(manager, name)
        if name == "mmsufinal":
            raw = dict(raw)
            raw["dataset_kwargs"] = {**raw.get("dataset_kwargs", {}), "data_files": {"test": settings["mmsu_jsonl"]}}
        return raw

    class RedundancyBridge(original_bridge):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            tower = self.runtime.model.thinker.audio_tower
            extractor = self.runtime.processor.feature_extractor
            if len(tower.layers) != settings["encoder_layers"]:
                raise ValueError("Audio tower depth differs from frozen plan")
            if extractor.sampling_rate != 16000 or extractor.hop_length != 160:
                raise ValueError("Expected native 16 kHz / 10 ms frontend")
            self.capture = RepresentationCapture(tower, sample_rate=extractor.sampling_rate,
                                                 hop_length=extractor.hop_length, layers=settings["layers"])
            self.capture.install()
            self.controller = RedundancyController(tower, self.capture)
            self.controller.install()
            self.bank = ModelBank(root / "models") if phase == "evaluation" and settings["needs_calibration"] else None
            self.samples = (output / f"samples.rank{rank}.jsonl").open("x", encoding="utf-8")
            self.conditions = (output / f"conditions.rank{rank}.jsonl").open("x", encoding="utf-8")
            sessions.append(self)
            write_json(output / f"runtime.rank{rank}.json", dict(
                torch=torch.__version__, transformers=transformers.__version__,
                audio_source=inspect.getfile(type(tower)), encoder_layers=len(tower.layers),
                n_window=tower.n_window, dtype=str(tower.dtype), backend=tower.config._attn_implementation,
                phase=phase, rank=rank, mmsu_jsonl=settings["mmsu_jsonl"],
                mmsu_dataset_dir=os.environ.get("MMSUFINAL_DATASET_DIR")))

        def sync(self):
            device = next(self.capture.tower.parameters()).device
            if device.type == "cuda":
                torch.cuda.synchronize(device)

        def _generate_sample(self, sample, **kwargs):
            doc = self._document(kwargs["task"], kwargs["split"], kwargs["doc_id"])
            index = int(kwargs["doc_id"])
            if doc["id"] != settings["document_ids"][index] or doc["audio_path"] != settings["audio_paths"][index]:
                raise ValueError("Runtime task row differs from frozen fold/coverage mapping")
            actual, _ = self._generation_kwargs(kwargs["request_generation"])
            defaults = self.runtime.model.thinker.generation_config
            if actual.get("thinker_do_sample", defaults.do_sample) or actual.get("thinker_num_beams", defaults.num_beams) != 1:
                raise ValueError("Paired analysis requires deterministic greedy generation")
            seed = settings["seed"] + index
            example = phase == "evaluation" and index in settings["example_doc_ids"]
            metadata = {key: doc.get(key) for key in (
                "id", "audio_path", "question_type", "sub_category", "task_name", "dataset_name")}
            metadata.update(doc_index=index, rank=rank, task="mmsufinal")
            self.controller.reset()
            self.capture.begin(example=example, seed=seed)
            self.sync()
            started = perf_counter()
            prediction = super()._generate_sample(sample, **kwargs)
            self.sync()
            seconds = perf_counter() - started
            layout, _ = self.capture.finish()
            accuracy = mmsufinal_process_results(doc, [prediction])["accuracy"]["score"]
            reference = dict(**metadata, prediction=prediction, accuracy=accuracy,
                             diagnostic_generation_seconds=seconds,
                             encoder_tokens=layout["encoder_tokens"], projector_tokens=layout["projector_tokens"],
                             native_pool_crossings=layout["native_pool_crossings"])
            if phase == "calibration":
                np.savez_compressed(output / "features" / f"doc{index}.npz",
                    **self.capture.calibration_arrays(settings["calibration_tokens"], seed=seed))
                write_row(self.samples, reference)
                return prediction
            self.sync()
            started = perf_counter()
            observations, discrete, bases, arrays = observe(self.capture, settings, self.bank,
                settings["folds"][index] if self.bank else None, example=example, seed=seed)
            self.sync()
            reference.update(observations, observation_seconds=perf_counter() - started)
            self.controller.bases = bases
            if example:
                filename = f"example.doc{index}.npz"
                np.savez_compressed(output / filename, **arrays)
                reference["example_file"] = filename
            write_row(self.samples, reference)
            for order, condition in enumerate(condition_order(settings["conditions"], index)):
                self.sync()
                started = perf_counter()
                self.controller.prepare(condition, discrete, seed=seed, example=example)
                self.sync()
                preparation_seconds = perf_counter() - started
                started = perf_counter()
                with mutable_compact_audio_request(self, self.controller, compact_audio_request):
                    answer = super()._generate_sample(sample, **{**kwargs, "cache_hook": None})
                self.sync()
                generation_seconds = perf_counter() - started
                details = self.controller.finish()
                score = mmsufinal_process_results(doc, [answer])["accuracy"]["score"]
                record = dict(**metadata, condition_id=condition["id"], condition=condition,
                    evaluation_order=order, prediction=answer, accuracy=score,
                    baseline_accuracy=accuracy, accuracy_delta=score - accuracy,
                    analysis_preparation_seconds=preparation_seconds,
                    diagnostic_generation_seconds=generation_seconds,
                    diagnostic_total_seconds=preparation_seconds + generation_seconds,
                    intervention=details)
                if example and self.controller.arrays:
                    filename = f"example.doc{index}.{condition['id']}.npz"
                    np.savez_compressed(output / filename, **self.controller.arrays)
                    record["example_file"] = filename
                write_row(self.conditions, record)
            self.controller.reset()
            # Only the native reference answer enters lmms-eval's own scorer.
            return prediction

        def close(self):
            self.controller.close()
            self.capture.close()
            self.samples.close()
            self.conditions.close()

    plugin.LMMSEvalRequestBridge = RedundancyBridge
    TaskManager._get_config = task_config
    try:
        runpy.run_module("lmms_eval", run_name="__main__")
    finally:
        plugin.LMMSEvalRequestBridge = original_bridge
        TaskManager._get_config = original_task_config
        try:
            for session in sessions:
                session.close()
        finally:
            if torch.distributed.is_available() and torch.distributed.is_initialized():
                torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
