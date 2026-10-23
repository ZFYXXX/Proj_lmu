"""Inspect installed framework entry points without loading models or datasets."""

from __future__ import annotations

import importlib
import json

from mllm_comp.integrations.lmms_eval import LMMSEvalAdapter
from mllm_comp.integrations.ms_swift import MSSwiftTrainAdapter


def main() -> int:
    lmms_report = LMMSEvalAdapter.dependency_report()
    swift_report = MSSwiftTrainAdapter.dependency_report()
    assert lmms_report["available"], lmms_report
    assert swift_report["available"], swift_report

    plugin = importlib.import_module("mllm_comp.integrations.lmms_eval.plugin")
    lmms_model = importlib.import_module("lmms_eval.api.model")
    manifests = plugin.get_model_manifests()
    assert manifests, "No platform lmms-eval model entry points"
    for manifest in manifests:
        for path in (manifest.simple_class_path, manifest.chat_class_path):
            module, name = path.rsplit(".", 1)
            assert issubclass(
                getattr(importlib.import_module(module), name), lmms_model.lmms
            )

    swift_model = importlib.import_module("swift.model")
    sft = importlib.import_module("swift.pipelines.train.sft")
    assert callable(sft.sft_main)
    loaders = {
        name: swift_model.MODEL_MAPPING[name].loader.__name__
        for name in ("qwen2_5_omni", "qwen2_5_vl", "qwen3_vl", "llava_onevision_qwen")
    }
    report = {
        "lmms_eval": {
            **lmms_report,
            "module": importlib.import_module("lmms_eval").__file__,
            "models": {
                entry.model_id: {
                    "simple": entry.simple_class_path,
                    "chat": entry.chat_class_path,
                }
                for entry in manifests
            },
        },
        "ms_swift": {
            **swift_report,
            "module": importlib.import_module("swift").__file__,
            "loaders": loaders,
        },
        "model_load_attempted": False,
        "status": "passed",
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
