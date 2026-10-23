"""Native lmms-eval worker with isolated acoustic hooks and paired answers."""
import inspect
import json
import os
from pathlib import Path
import runpy
from time import perf_counter


def condition_order(conditions, doc_index):
    if not conditions:
        return []
    offset = doc_index % len(conditions)
    return conditions[offset:] + conditions[:offset]


def main():
    import numpy as np
    import torch
    import transformers
    from lmms_eval.tasks import TaskManager
    from lmms_eval.tasks.mmsufinal.utils import mmsufinal_process_results
    from mllm_comp.integrations.lmms_eval import plugin
    from scripts.analysis.audio_temporal.metrics import jsonable
    from .observer import AcousticObserver
    from .signal import acoustic_targets, acoustic_profile, audio_intervention, mel_intervention, spectrum

    if transformers.__version__ != "4.57.3":
        raise RuntimeError(f"Acoustic hooks require Transformers 4.57.3, found {transformers.__version__}")
    settings = json.loads(Path(os.environ["AUDIO_ACOUSTIC_CONFIG"]).read_text())
    output = Path(settings["output_dir"])
    rank = int(os.environ.get("RANK", "0"))
    original_bridge = plugin.LMMSEvalRequestBridge
    original_task_config = TaskManager._get_config
    sessions = []

    def task_config(manager, name):
        raw = original_task_config(manager, name)
        if name == "mmsufinal" and settings.get("mmsu_jsonl"):
            raw = dict(raw)
            raw["dataset_kwargs"] = {**raw.get("dataset_kwargs", {}),
                                     "data_files": {"test": settings["mmsu_jsonl"]}}
        return raw

    class AcousticBridge(original_bridge):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            tower = self.runtime.model.thinker.audio_tower
            self.extractor = self.runtime.processor.feature_extractor
            if len(tower.layers) != settings["encoder_layers"]:
                raise ValueError("Audio encoder depth differs from the frozen plan")
            if self.extractor.sampling_rate != 16000 or self.extractor.hop_length != 160:
                raise ValueError("This suite requires the native 16 kHz / 10 ms Whisper frontend")
            self.observer = AcousticObserver(tower, sample_rate=16000,
                hop_length=self.extractor.hop_length, layers=settings["layers"])
            self.observer.install()
            self.samples_file = (output / f"samples.rank{rank}.jsonl").open("x", encoding="utf-8")
            self.conditions_file = (output / f"conditions.rank{rank}.jsonl").open("x", encoding="utf-8")
            sessions.append(self)
            (output / f"runtime.rank{rank}.json").write_text(json.dumps(dict(
                torch=torch.__version__, transformers=transformers.__version__,
                audio_source=inspect.getfile(type(tower)), encoder_layers=len(tower.layers),
                n_window=tower.n_window, dtype=str(tower.dtype),
                backend=tower.config._attn_implementation, sampling_rate=16000,
                hop_length=self.extractor.hop_length,
                mmsu_jsonl=settings.get("mmsu_jsonl"),
                mmsu_dataset_dir=os.environ.get("MMSUFINAL_DATASET_DIR")), indent=2), encoding="utf-8")

        def _generate_sample(self, sample, **kwargs):
            doc = self._document(kwargs["task"], kwargs["split"], kwargs["doc_id"])
            doc_index = int(kwargs["doc_id"])
            actual, _ = self._generation_kwargs(kwargs["request_generation"])
            defaults = self.runtime.model.thinker.generation_config
            if actual.get("thinker_do_sample", defaults.do_sample) or actual.get("thinker_num_beams", defaults.num_beams) != 1:
                raise ValueError("Paired acoustic analysis requires greedy deterministic generation")
            decoded_paths = [item["audio"] for message in sample.conversation
                             if isinstance(message.get("content"), list)
                             for item in message["content"]
                             if item.get("type") == "audio" and isinstance(item.get("audio"), str)]
            metadata = {key: doc.get(key) for key in (
                "id", "audio_path", "question_type", "sub_category", "task_name", "dataset_name")}
            metadata.update(doc_index=doc_index, rank=rank, task=kwargs["task"], split=kwargs["split"])
            metadata["decoded_audio_paths"] = decoded_paths
            reference_targets = None
            reference_wave = None
            reference_spectrum = None
            reference_prediction = None
            reference_accuracy = None
            device = next(self.observer.tower.parameters()).device
            for order, condition in enumerate([None] + condition_order(settings["conditions"], doc_index)):
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                    torch.cuda.reset_peak_memory_stats(device)
                started = perf_counter()
                self.observer.begin(condition=condition,
                    example=condition is None and doc_index in settings["example_doc_ids"])
                capture = {}

                def on_wave(original, changed):
                    nonlocal reference_targets, reference_wave
                    if condition is None:
                        reference_wave = original.copy()
                        reference_targets = acoustic_targets(original)
                    elif not np.array_equal(original, reference_wave):
                        raise RuntimeError("The native decoder returned a different waveform for a paired condition")

                with audio_intervention(self.facade.processor_adapter, condition, capture, on_wave), \
                        mel_intervention(self.extractor, condition):
                    call_kwargs = kwargs if condition is None else {**kwargs, "cache_hook": None}
                    prediction = super()._generate_sample(sample, **call_kwargs)
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                elapsed = perf_counter() - started
                measurements, arrays = self.observer.finish()
                score = mmsufinal_process_results(doc, [prediction])["accuracy"]["score"]
                if condition is None:
                    reference_prediction, reference_accuracy = prediction, score
                    reference_spectrum = spectrum(capture["original_wave"])
                row = dict(**metadata, **measurements, prediction=prediction, accuracy=score,
                    diagnostic_seconds=elapsed,
                    peak_gpu_memory_bytes=int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else None,
                    waveform_filter=capture.get("filter"),
                    spectrum=spectrum(capture["input_wave"]) if condition and condition["family"] == "wave" else reference_spectrum)
                if condition is None:
                    row["acoustic_proxy_summary"] = acoustic_profile(reference_wave, reference_targets)
                    if doc_index in settings["example_doc_ids"]:
                        name = f"example.rank{rank}.doc{doc_index}.npz"
                        arrays.update({f"target_{key}": value for key, value in reference_targets.items()
                                       if isinstance(value, np.ndarray)})
                        np.savez_compressed(output / name, **arrays)
                        row["example_file"] = name
                    handle = self.samples_file
                else:
                    row.update(condition_id=condition["id"], condition=condition, evaluation_order=order - 1,
                               baseline_accuracy=reference_accuracy, damage=reference_accuracy - score)
                    handle = self.conditions_file
                handle.write(json.dumps(jsonable(row), ensure_ascii=False, allow_nan=False) + "\n")
                handle.flush()
            return reference_prediction

        def close_acoustic(self):
            self.observer.close()
            self.samples_file.close()
            self.conditions_file.close()

    plugin.LMMSEvalRequestBridge = AcousticBridge
    TaskManager._get_config = task_config
    try:
        runpy.run_module("lmms_eval", run_name="__main__")
    finally:
        TaskManager._get_config = original_task_config
        plugin.LMMSEvalRequestBridge = original_bridge
        for session in sessions:
            session.close_acoustic()


if __name__ == "__main__":
    main()
