"""Configured factories use the existing prepare, parameter and rollback path."""

from types import SimpleNamespace

import pytest

from mllm_comp.backbones.qwen3_vl import create_qwen3_vl_backbone
from mllm_comp.core.errors import BackendIntegrationError
from mllm_comp.compressors.extensions import compression_contracts
from mllm_comp.config.schema import experiment_config_from_dict
from mllm_comp.pipeline.definition import resolve_compression_definition
from mllm_comp.pipeline.runtime_session import RuntimeCompressionDispatcher
from tests.unit.test_qwen3_vl_attention_replacement import _torch_fixture, _fake_mlp


CALLS = []


def scaled_factory(request, *, factor=2.0):
    import torch

    CALLS.append((request["strategy"], request["module_path"], request["original_module"]))

    class Scaled(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.base = request["default_factory"]()
            reference = next(request["original_module"].parameters())
            self.scale = torch.nn.Parameter(reference.new_tensor(factor))

        def forward(self, *args, **kwargs):
            result = self.base(*args, **kwargs)
            if isinstance(result, tuple):
                return (result[0] * self.scale, *result[1:])
            return result * self.scale

    return Scaled()


def invalid_factory(request):
    return request["strategy"]


def original_factory(request):
    return request["original_module"]


def _stage(strategy, *, factor=2.0, function="scaled_factory", targets=(0,)):
    attention = strategy.startswith("attn_")
    path = "attn" if attention else "ffn"
    return experiment_config_from_dict({
        "backbone": {"name": "qwen3_vl"}, "pipeline": {"stages": [{
        "name": "custom-module", "placement": "llm_level", "execution_phase": "prepare",
        "algorithm": "qwen3_vl_attention_replacement" if attention else "qwen3_vl_ffn_replacement",
        "method": "operation_compression", "object_family": "module_unit",
        "operation": {"type": "replace", "rules": {"factory": {
            "callable": __name__ + ":" + function,
            "params": {"factor": factor} if function == "scaled_factory" else {},
        }}},
        "schedule": {"targets": [{"site": f"llm.layer[{i}].{path}"} for i in targets]},
        "config": {"strategy": strategy, "rank": 4, "bottleneck_dim": 8,
                   "window_size": 3, "decode_window_size": 3, "feature_dim": 4, "num_slots": 2},
    }]}}).pipeline.stages[0]


def _setup(strategy, **kwargs):
    torch, attention_type = _torch_fixture()
    layer = torch.nn.Module()
    layer.self_attn = attention_type()
    layer.mlp = _fake_mlp(torch)
    model = torch.nn.Module()
    model.model = torch.nn.Module()
    model.model.language_model = torch.nn.Module()
    model.model.language_model.layers = torch.nn.ModuleList([layer])
    model.model.visual = None
    model.lm_head = None
    stage = _stage(strategy, **kwargs)
    runtime = SimpleNamespace(model=model, metadata={})
    config = SimpleNamespace(backbone=SimpleNamespace(name="qwen3_vl"),
                             pipeline=SimpleNamespace(stages=[stage], extensions=[]))
    facade = create_qwen3_vl_backbone()
    definition = resolve_compression_definition(config, compression_mode="compressed")
    session = RuntimeCompressionDispatcher(contract_registry=definition.registry).resolve(
        definition=definition, config=config, facade=facade, runtime=runtime,
        compression_mode="compressed", execution_phase="generate",
    )
    return torch, layer, model, stage, runtime, session


@pytest.mark.parametrize("strategy", [
    "attn_linear_attention", "attn_cnn", "attn_identity", "attn_lolcats_linear_window",
    "attn_nha", "attn_sparse_window", "ffn_low_rank", "ffn_identity", "ffn_two_layer_mlp",
])
def test_all_nine_strategies_execute_configured_factory_with_gradients_and_restore(strategy):
    CALLS.clear()
    torch, layer, model, stage, runtime, session = _setup(strategy)
    attribute = "self_attn" if strategy.startswith("attn_") else "mlp"
    original = getattr(layer, attribute)
    original.eval()
    assert CALLS == []
    session.install()
    replacement = getattr(layer, attribute)
    assert CALLS == [(strategy, "llm.layer[0]." + ("attn" if attribute == "self_attn" else "ffn"), original)]
    assert replacement.training is False
    assert replacement.scale is dict(model.named_parameters())["model.language_model.layers.0." + attribute + ".scale"]
    inputs = torch.randn(2, 5, 16, requires_grad=True)
    kwargs = {"attention_mask": torch.ones(2, 5), "position_embeddings": None} if attribute == "self_attn" else {}
    baseline = replacement.base(inputs, **kwargs)
    actual = replacement(inputs, **kwargs)
    if isinstance(actual, tuple):
        assert actual[1] is None
        baseline, actual = baseline[0], actual[0]
    torch.testing.assert_close(actual, baseline * 2)
    actual.square().mean().backward()
    assert inputs.grad is not None and torch.isfinite(inputs.grad).all()
    assert replacement.scale.grad is not None and replacement.scale.grad.abs() > 0
    optimizer = torch.optim.SGD(model.parameters(), lr=.1)
    optimizer.step()
    assert replacement.scale.item() != 2.0
    session.install()
    assert len(CALLS) == 1
    stage.operation["rules"]["factory"]["params"]["factor"] = 7
    metadata = session.runtime_metadata()["activations"][0]
    assert metadata["rules"]["factory"]["params"]["factor"] == 2
    assert metadata["rules_changed"] and metadata["development_runtime"]
    assert metadata["canonical_evidence_applicable"] is False
    assert metadata["support_status"] == "declared"
    session.uninstall()
    assert getattr(layer, attribute) is original
    assert all(parameter is not replacement.scale for parameter in model.parameters())


def test_invalid_factory_return_rolls_back_original_module():
    _, layer, _, _, _, session = _setup("ffn_identity", function="invalid_factory")
    original = layer.mlp
    with pytest.raises(BackendIntegrationError, match="torch.nn.Module"):
        session.install()
    assert layer.mlp is original


def test_new_factory_parameters_cannot_silently_reuse_an_installed_replacement():
    _, layer, model, stage, runtime, first = _setup("ffn_identity")
    original = layer.mlp
    first.install()
    installed = layer.mlp
    second = RuntimeCompressionDispatcher(contract_registry=compression_contracts()).resolve(
        config=SimpleNamespace(pipeline=SimpleNamespace(stages=[_stage("ffn_identity", factor=3)])),
        facade=create_qwen3_vl_backbone(), runtime=runtime,
        compression_mode="compressed", execution_phase="generate",
    )
    with pytest.raises(BackendIntegrationError, match="different.*factory"):
        second.install()
    assert layer.mlp is installed
    first.uninstall()
    assert layer.mlp is original


def test_returning_original_is_rejected_without_poisoning_later_install():
    _, layer, _, _, runtime, session = _setup("ffn_identity", function="original_factory")
    original = layer.mlp
    with pytest.raises(BackendIntegrationError, match="new module"):
        session.install()
    assert layer.mlp is original
    assert not hasattr(original, "mllm_comp_factory_spec")
    assert not hasattr(original, "mllm_comp_replacement_strategy")
    stage = _stage("ffn_identity")
    stage.operation.pop("rules")
    later = RuntimeCompressionDispatcher(contract_registry=compression_contracts()).resolve(
        config=SimpleNamespace(pipeline=SimpleNamespace(stages=[stage])),
        facade=create_qwen3_vl_backbone(), runtime=runtime,
        compression_mode="compressed", execution_phase="generate",
    )
    later.install()
    assert layer.mlp is not original
    later.uninstall()
    assert layer.mlp is original
