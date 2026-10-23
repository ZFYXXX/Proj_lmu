"""Public definition, dispatcher and training-transport coverage."""
import base64
import json
from types import SimpleNamespace

import pytest

from mllm_comp.compressors.multimodal_registry import builtin_multimodal_compression_contracts
from mllm_comp.config.schema import experiment_config_from_dict
from mllm_comp.pipeline.definition import resolve_compression_definition
from mllm_comp.core.support import SupportStatus

OPERATIONS = {"dop", "shortv", "redundancy_lens", "yopo", "structural_pruning",
              "short_lvlm", "adallava", "adallava_l", "adallava_h", "skip_vision"}
PROJECTORS = {"cdpruner", "dppruner", "eadp", "hiprune"}
DECODERS = {"vlm_pruner", "core_matching", "twigvlm"}
BACKBONES = ("llava_onevision", "qwen25_vl", "qwen3_vl")


def experiment(backbone, method, parameters=None):
    operation = method in OPERATIONS
    return experiment_config_from_dict({"experiment": {"name": "comparison"},
        "backbone": {"name": backbone, "model_path": "/native-fixture"},
        "pipeline": {"stages": [{"name": method, "plugin": method, "algorithm": method,
            "variant": f"{method}.image", "binding": f"{method}.image.{backbone}",
            "placement": "projector_level" if method in PROJECTORS else "llm_level",
            "method": "operation_compression" if operation else "token_compression",
            "object_family": "module_unit" if operation else "tensor_slot",
            "execution_phase": "prepare" if operation else "forward",
            "modalities": ["image"], "target_modalities": ["image"], "config": parameters or {}}]},
        "train": {"backend": "ms_swift", "compression_mode": "compressed",
                  "args": {"packing": False, "padding_free": False}},
        "eval": {"backend": "lmms_eval", "runtime_batch_size": 1},
        "output": {"run_dir": "/native-fixture/output"}})


def test_all_17_entries_have_three_explicit_declared_bindings():
    registry = builtin_multimodal_compression_contracts()
    for method in OPERATIONS | PROJECTORS | DECODERS:
        for backbone in BACKBONES:
            binding = registry.binding(f"{method}.image.{backbone}")
            assert callable(binding.executor_factory)
            units = registry.support.find(algorithm=method, backbone=backbone)
            assert units and all(unit.status is SupportStatus.DECLARED for unit in units)


@pytest.mark.parametrize("backbone", BACKBONES)
@pytest.mark.parametrize("method", sorted(OPERATIONS | PROJECTORS | DECODERS))
def test_public_definition_and_swift_payload(backbone, method):
    from mllm_comp.integrations.ms_swift.adapter import _training_compression_spec
    from mllm_comp.modality_split.pipeline import compile_pipeline_split
    config = experiment(backbone, method)
    definition = resolve_compression_definition(config, compression_mode="compressed", interface="train")
    assert len(definition.stages) == 1
    assert compile_pipeline_split(config, definition=definition) is None
    encoded, metadata = _training_compression_spec(config)
    payload = json.loads(base64.urlsafe_b64decode(encoded))
    assert payload["stage"]["algorithm"] == method
    assert metadata["enabled"]


@pytest.mark.parametrize("backbone", BACKBONES)
def test_training_plugin_installs_comparison_operation(backbone, tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")
    if backbone == "llava_onevision":
        pytest.importorskip("llava")
    from tests.fixtures import native_image
    from tests.unit.test_ms_swift_training_lifecycle import _plugin
    plugin = _plugin(monkeypatch)
    from mllm_comp.core.vocabulary import RuntimeMode
    records = []
    recorder = SimpleNamespace(record=lambda event, **fields: records.append({"event": event, **fields}),
        next_forward_index=lambda: 1, claim_default_forward_evidence=lambda: True)
    model = native_image.model(backbone).train()
    config = experiment(backbone, "dop", {"mha_depth": 1, "mlp_depth": 1, "token_reducer": "none"})
    plugin._install_image_model_bridge(model, backbone_name=backbone, stage=config.pipeline.stages[0],
        runtime_mode=RuntimeMode.DEFAULT, recorder=recorder)
    try:
        inputs = native_image.inputs(torch, backbone)
        labels = torch.full_like(inputs["input_ids"], -100)
        labels[:, -1] = 3
        output = model(**inputs, labels=labels)
        output.loss.backward()
        forward = next(item for item in records if item["event"] == "forward")
        assert any(item.get("comparison_operation_executed") for item in forward["layout_records"])
        assert any(item["event"] == "backward" for item in records)
    finally:
        plugin._close_model_bridge(model)
