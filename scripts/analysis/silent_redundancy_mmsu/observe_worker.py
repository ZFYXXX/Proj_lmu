"""Native lmms-eval worker for one uncompressed observational pass."""
from __future__ import annotations

import inspect
import json
import os
from pathlib import Path
import runpy


def main():
    import numpy as np
    import torch
    import transformers
    from lmms_eval.tasks import TaskManager
    from lmms_eval.tasks.mmsufinal.utils import mmsufinal_process_results
    from mllm_comp.integrations.lmms_eval import plugin

    from .metrics import jsonable
    from .observer import SilentRedundancyObserver

    if transformers.__version__ != "4.57.3":
        raise RuntimeError(f"Analysis targets Transformers 4.57.3; found {transformers.__version__}")
    settings = json.loads(Path(os.environ["SILENT_REDUNDANCY_CONFIG"]).read_text())
    output = Path(settings["output_dir"])
    rank = int(os.environ.get("RANK", "0"))
    original_bridge = plugin.LMMSEvalRequestBridge
    original_task_config = TaskManager._get_config
    sessions = []

    def task_config(manager, name):
        raw = original_task_config(manager, name)
        if name == "mmsufinal" and settings.get("mmsu_jsonl"):
            raw = dict(raw)
            raw["dataset_kwargs"] = {
                **raw.get("dataset_kwargs", {}),
                "data_files": {"test": settings["mmsu_jsonl"]},
            }
        return raw

    class ObservationBridge(original_bridge):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            tower = self.runtime.model.thinker.audio_tower
            extractor = self.runtime.processor.feature_extractor
            self.observer = SilentRedundancyObserver(
                tower,
                sample_rate=extractor.sampling_rate,
                hop_length=extractor.hop_length,
                layers=settings["layers"],
                query_block=settings["query_block"],
            )
            self.observer.install()
            self.records = (output / f"samples.rank{rank}.jsonl").open("x", encoding="utf-8")
            sessions.append(self)
            (output / f"runtime.rank{rank}.json").write_text(json.dumps({
                "torch": torch.__version__,
                "transformers": transformers.__version__,
                "audio_source": inspect.getfile(type(tower)),
                "encoder_layers": len(tower.layers),
                "heads": tower.layers[0].self_attn.num_heads,
                "n_window": tower.n_window,
                "dtype": str(tower.dtype),
                "backend": tower.config._attn_implementation,
                "observed_layers": settings["layers"],
                "compression": False,
            }, indent=2), encoding="utf-8")

        def _generate_sample(self, sample, **kwargs):
            doc = self._document(kwargs["task"], kwargs["split"], kwargs["doc_id"])
            doc_index = int(kwargs["doc_id"])
            actual, _ = self._generation_kwargs(kwargs["request_generation"])
            defaults = self.runtime.model.thinker.generation_config
            if actual.get("thinker_do_sample", defaults.do_sample) or \
                    actual.get("thinker_num_beams", defaults.num_beams) != 1:
                raise ValueError("Paired analysis requires greedy deterministic generation")
            example = doc_index in settings["example_doc_ids"]
            self.observer.begin(example=example)
            prediction = super()._generate_sample(sample, **kwargs)
            measurements, arrays = self.observer.finish()
            row = {
                "id": doc["id"],
                "doc_index": doc_index,
                "audio_path": doc["audio_path"],
                "question_type": doc["question_type"],
                "sub_category": doc.get("sub_category"),
                "task_name": doc.get("task_name"),
                "prediction": prediction,
                "accuracy": mmsufinal_process_results(doc, [prediction])["accuracy"]["score"],
                **measurements,
            }
            if example:
                filename = f"example.rank{rank}.doc{doc_index}.npz"
                np.savez_compressed(output / filename, **arrays)
                row["example_file"] = filename
            self.records.write(json.dumps(jsonable(row), ensure_ascii=False, allow_nan=False) + "\n")
            self.records.flush()
            return prediction

        def close_analysis(self):
            self.observer.close()
            self.records.close()

    plugin.LMMSEvalRequestBridge = ObservationBridge
    TaskManager._get_config = task_config
    try:
        runpy.run_module("lmms_eval", run_name="__main__")
    finally:
        TaskManager._get_config = original_task_config
        plugin.LMMSEvalRequestBridge = original_bridge
        for session in sessions:
            session.close_analysis()


if __name__ == "__main__":
    main()
