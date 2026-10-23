"""Rule config, lazy loading and exact-recipe evidence share one contract."""

from types import SimpleNamespace

import pytest

from mllm_comp.compressors.extensions import compression_contracts
from mllm_comp.compressors.rules import load_rules, recipe_changed, resolve_rule_binding, rule_specs
from mllm_comp.core.errors import ConfigError
from mllm_comp.pipeline.stage import CompressionStageConfig


def scale(request, *, factor=2):
    return request["value"] * factor


def _stage(**fields):
    return SimpleNamespace(variant="fastv.image", operation={}, criterion={}, **fields)


def _supported(stage):
    if not stage.variant:
        return frozenset()
    return compression_contracts().variant(stage.variant).supported_rules


def _spec(path="tests.unit.test_compression_rules:scale", **params):
    return {"callable": path, "params": params}


def test_loading_binds_params_but_parsing_never_imports(monkeypatch):
    import mllm_comp.compressors.rules as module

    stage = _stage()
    stage.criterion["scorer"] = _spec(factor=3)
    calls = []
    original = module.import_module
    monkeypatch.setattr(module, "import_module", lambda name: (calls.append(name), original(name))[1])
    assert rule_specs(stage, supported_rules=_supported(stage)) == {"scorer": stage.criterion["scorer"]}
    assert calls == []
    rules = load_rules(stage, supported_rules=_supported(stage))
    assert rules["scorer"]({"value": 4}) == 12
    assert rules["scorer"]({"value": 5}) == 15
    assert calls == ["tests.unit.test_compression_rules"]
    assert load_rules(_stage(), supported_rules={"scorer", "selector"}) == {}
    assert calls == ["tests.unit.test_compression_rules"]


def test_named_rules_share_the_original_loading_path_and_reject_duplicates():
    stage = _stage()
    stage.operation = {"rules": {"scorer": _spec(factor=4)}}
    assert load_rules(stage, supported_rules=_supported(stage))["scorer"]({"value": 3}) == 12
    stage.criterion = {"scorer": _spec(factor=4)}
    with pytest.raises(ConfigError, match="both"):
        rule_specs(stage, supported_rules=_supported(stage))


def test_video_loader_parses_once_and_records_the_loaded_parameters(monkeypatch):
    from mllm_comp.compressors import rules as module
    from mllm_comp.compressors.video._rules import load_video_rules

    assert load_video_rules(None) == ({}, {})
    stage = _stage()
    stage.variant = "visionzip.video"
    stage.operation = {"rules": {"scorer": _spec(factor=3)}}
    parsed = []
    original = module.rule_specs

    def parse(value, *, supported_rules):
        parsed.append(value)
        return original(value, supported_rules=supported_rules)

    monkeypatch.setattr(module, "rule_specs", parse)
    functions, specs = load_video_rules(stage)
    stage.operation["rules"]["scorer"]["params"]["factor"] = 99
    assert parsed == [stage]
    assert functions["scorer"]({"value": 4}) == 12
    assert specs["scorer"]["params"] == {"factor": 3}


@pytest.mark.parametrize("rules", [None, [], "rule", {"unknown": _spec()}])
def test_named_rules_reject_invalid_or_unknown_entries(rules):
    stage = _stage()
    stage.operation["rules"] = rules
    with pytest.raises(ConfigError):
        rule_specs(stage, supported_rules=_supported(stage))


@pytest.mark.parametrize("plugin", ["qwen3_vl_attention_replacement", "qwen3_vl_ffn_replacement"])
def test_module_factory_is_declared_research_in_definition_and_manifest(plugin, monkeypatch):
    from mllm_comp.pipeline.definition import _validate_stage
    from mllm_comp.run_manifest import build_run_manifest
    from tests.unit.test_manifest_v2_compression_records import _config
    import mllm_comp.compressors.rules as module

    monkeypatch.setattr(module, "import_module", lambda name: pytest.fail("Imported at config time"))
    attention = "attention" in plugin
    raw = {"name": "module-rule", "algorithm": plugin, "placement": "llm_level",
           "execution_phase": "prepare", "method": "operation_compression",
           "object_family": "module_unit", "operation": {"type": "replace"},
           "schedule": {"targets": [{"site": "llm.layer[0]." + ("attn" if attention else "ffn")}]},
           "config": {"strategy": "attn_identity" if attention else "ffn_identity"}}
    baseline = build_run_manifest(_config(raw)).compression_stages[0]
    raw["operation"]["rules"] = {"factory": _spec("external_rules:factory")}
    config = _config(raw)
    _validate_stage(config.pipeline.stages[0], backbone_name="qwen3_vl", interface="eval", registry=compression_contracts())
    record = build_run_manifest(config).compression_stages[0]
    assert record.support_status == "declared"
    assert baseline.support_status == "declared"
    assert compression_contracts().algorithm(plugin).metadata["capability_status"] == "active"
    assert record.metadata["rules"] == raw["operation"]["rules"]
    assert record.metadata["development_runtime"] is True
    assert record.metadata["canonical_evidence_applicable"] is False
    assert record.metadata["legacy_descriptor"] is False
    assert record.metadata["support_claim_is_exact"] is False


@pytest.mark.parametrize("spec", [None, "pkg:fn", {}, {"callable": "bad"},
    {"callable": "pkg:Class.method"}, {"callable": "pkg:fn", "param": {}},
    {"callable": "pkg:fn", "params": []}, {"callable": "pkg:fn", "params": {1: 2}},
    {"callable": "pkg:fn", "params": {"x": float("nan")}},
    {"callable": "pkg:fn", "params": {"x": object()}},
])
def test_invalid_descriptions_fail_before_activation(spec):
    stage = _stage()
    stage.criterion["scorer"] = spec
    with pytest.raises(ConfigError):
        rule_specs(stage, supported_rules=_supported(stage))


@pytest.mark.parametrize("path,params", [
    ("compression_rule_module_does_not_exist:rule", {}),
    ("tests.unit.test_compression_rules:missing", {}),
    ("types:SimpleNamespace", {}),
    ("tests.unit.test_compression_rules:scale", {"unknown": 1}),
])
def test_import_or_signature_errors_are_config_errors(path, params):
    stage = _stage()
    stage.operation["selector"] = {"callable": path, "params": params}
    with pytest.raises(ConfigError):
        load_rules(stage, supported_rules=_supported(stage))


@pytest.mark.parametrize("variant,location,name", [
    ("fastv.image", "operation", "grouper"),
    ("fastv.image", "operation", "scorer"),
    ("local_temporal_bipartite_merging_audio.audio", "criterion", "scorer"),
    ("tome.image.encoder_dense_boundary", "operation", "aggregator"),
    (None, "operation", "selector"),
])
def test_unsupported_or_misplaced_rules_are_not_ignored(variant, location, name):
    stage = _stage()
    stage.variant = variant
    getattr(stage, location)[name] = _spec()
    with pytest.raises(ConfigError):
        rule_specs(stage, supported_rules=_supported(stage))


def _stage_dict(backbone):
    audio = backbone == "qwen25_omni"
    algorithm = "local_temporal_bipartite_merging_audio" if audio else "fastv"
    variant = algorithm + (".audio" if audio else ".image")
    return {
        "name": "custom-rules", "plugin": algorithm, "algorithm": algorithm,
        "variant": variant, "binding": f"{variant}.{backbone}",
        "placement": "projector_level" if audio else "llm_level", "execution_phase": "forward",
        "method": "token_compression", "object_family": "tensor_slot",
        "target_modalities": ["audio" if audio else "image"],
        "modalities": ["audio" if audio else "image"],
        "conditioning_modalities": [] if audio else ["text"],
        "preserved_modalities": ["text", "image", "video"] if audio else ["text", "video", "audio"],
        "media_alignment": "temporal" if audio else "token", "operation": {},
    }


@pytest.mark.parametrize("backbone", ["qwen25_omni", "qwen25_vl", "qwen3_vl", "llava_onevision"])
def test_rule_projection_preserves_binding_positions_and_historical_evidence(backbone, monkeypatch):
    from mllm_comp.compressors.multimodal_registry import builtin_multimodal_compression_contracts
    from mllm_comp.pipeline.v2_validation import validate_v2_stage_contract
    from mllm_comp.run_manifest import build_run_manifest
    from tests.unit.test_manifest_v2_compression_records import _config
    import mllm_comp.compressors.rules as module

    monkeypatch.setattr(module, "import_module", lambda name: pytest.fail("Config/Manifest imported user code"))
    raw = _stage_dict(backbone)
    baseline = build_run_manifest(_config(raw, backbone_name=backbone)).compression_stages[0]
    slot = "aggregator" if backbone == "qwen25_omni" else "selector"
    raw["operation"][slot] = _spec("not_installed_yet:my_rule", factor=3)
    stage = CompressionStageConfig.from_dict(raw)
    registry = builtin_multimodal_compression_contracts()
    original = registry.binding(raw["binding"])
    projection = validate_v2_stage_contract(stage, registry=registry, backbone_name=backbone)
    binding = projection.bindings[0]
    assert binding.targets == original.targets
    assert not binding.metadata.get("position_changed")
    assert recipe_changed(binding) and not recipe_changed(original)
    assert "rules" not in original.metadata
    record = build_run_manifest(_config(raw, backbone_name=backbone)).compression_stages[0]
    assert record.metadata["rules"] == {slot: raw["operation"][slot]}
    assert record.metadata["rules_changed"] is True
    assert record.metadata["legacy_descriptor"] is False
    assert record.metadata["canonical_evidence_applicable"] is False
    assert record.metadata["development_runtime"] is True
    assert record.support_status == "declared"
    assert record.target_sites == baseline.target_sites
    assert [{k: v for k, v in item.items() if k != "canonical_evidence_applicable"}
        for item in record.evidence] == baseline.evidence
    assert all(item["canonical_evidence_applicable"] is False for item in record.evidence)
    assert resolve_rule_binding(binding=original, rules={}) is original


def test_legacy_definition_rejects_rule_instead_of_ignoring_it():
    from mllm_comp.pipeline.definition import _validate_stage

    stage = CompressionStageConfig.from_dict({
        "name": "legacy", "plugin": "operation_replace", "placement": "llm_level",
        "execution_phase": "prepare", "method": "operation_compression", "object_family": "module_unit",
        "operation": {"selector": _spec()},
    })
    with pytest.raises(ConfigError, match="does not support"):
        _validate_stage(stage, backbone_name="qwen3_vl", interface="eval", registry=compression_contracts())


def test_dispatcher_records_rules_before_install_and_rejects_legacy_overrides():
    from dataclasses import replace
    from mllm_comp.compressors.multimodal_registry import builtin_multimodal_compression_contracts
    from mllm_comp.core.errors import BackendIntegrationError
    from mllm_comp.pipeline.runtime_session import RuntimeCompressionDispatcher

    registry = builtin_multimodal_compression_contracts()
    raw = _stage_dict("qwen25_vl")
    raw["operation"]["selector"] = _spec()
    stage = CompressionStageConfig.from_dict(raw)
    events = []
    activation = SimpleNamespace(install=lambda: events.append("installed"),
        uninstall=lambda: None, runtime_metadata=lambda: {})
    contracts = SimpleNamespace(algorithm=registry.algorithm, variant=registry.variant,
        support=registry.support, binding=lambda name: replace(registry.binding(name), executor_factory=lambda **kwargs: activation))
    runtime = SimpleNamespace(model=object(), metadata={})
    facade = SimpleNamespace(capabilities=lambda: SimpleNamespace(name="qwen25_vl"))
    config = SimpleNamespace(pipeline=SimpleNamespace(stages=[stage]))
    dispatcher = RuntimeCompressionDispatcher(contract_registry=contracts)
    session = dispatcher.resolve(config=config, facade=facade, runtime=runtime,
        compression_mode="compressed", execution_phase="forward")
    assert events == []
    metadata = session.runtime_metadata()["activations"][0]
    assert metadata["rules"] == {"selector": _spec()}
    assert metadata["rules_changed"] and metadata["development_runtime"]
    assert metadata["canonical_evidence_applicable"] is False
    assert metadata["support_status"] == "declared"
    config.pipeline.stages = [SimpleNamespace(enabled=True, variant=None,
        execution_phase="generate", operation={"selector": _spec()}, criterion={})]
    with pytest.raises(BackendIntegrationError, match="does not support"):
        dispatcher.resolve(config=config, facade=facade, runtime=runtime, compression_mode="compressed")


@pytest.mark.parametrize("filename", ["fastv_feature_norm.yaml", "ltbm_weighted_merge.yaml"])
def test_example_configs_resolve_through_the_formal_definition(filename):
    from pathlib import Path
    import yaml
    from mllm_comp.config.schema import experiment_config_from_dict
    from mllm_comp.pipeline.definition import resolve_compression_definition

    path = Path(__file__).resolve().parents[2] / "examples/compression_rules" / filename
    config = experiment_config_from_dict(yaml.safe_load(path.read_text(encoding="utf-8")))
    definition = resolve_compression_definition(config, compression_mode="compressed", interface="analysis", runtime_batch_size=1)
    assert len(definition.stages) == 1
    assert load_rules(definition.stages[0], supported_rules=definition.projections[definition.stages[0].name].variant.supported_rules)


def test_example_functions_execute_the_documented_pruning_and_merging_rules():
    torch = pytest.importorskip("torch")
    from mllm_comp.compressors.image.fastv import FastVSpec, fastv_image_tokens
    from mllm_comp.compressors.projector_level.local_temporal_bipartite_merging import local_temporal_bipartite_merge

    module = "mllm_comp.compressors.rule_examples:"
    stage = _stage()
    stage.criterion["scorer"] = _spec(module + "fastv_feature_norm", power=2.0)
    values = torch.arange(6, dtype=torch.float32)[:, None]
    kwargs = dict(image_indices=torch.tensor([1, 2, 3, 4]), scorer_query=None,
        attention_keys=None, spec=FastVSpec(prune_layer=0))
    output = fastv_image_tokens(values, **kwargs, **load_rules(stage, supported_rules=_supported(stage)))
    assert output.kept_indices.tolist() == [0, 3, 4, 5]
    stage.operation["selector"] = _spec(module + "fastv_uniform_select")
    output = fastv_image_tokens(values, **kwargs, **load_rules(stage, supported_rules=_supported(stage)))
    assert output.kept_indices.tolist() == [0, 1, 3, 5]
    stage.variant = "local_temporal_bipartite_merging_audio.audio"
    stage.criterion = {}
    stage.operation = {"grouper": _spec(module + "ltbm_nearest_pairs"),
        "aggregator": _spec(module + "ltbm_destination_weighted_mean", destination_weight=3.0)}
    audio = torch.tensor([[0.], [10.], [20.], [30.]], requires_grad=True)
    merged, details = local_temporal_bipartite_merge(audio, keep_count=2, window_size=1, **load_rules(stage, supported_rules=_supported(stage)))
    torch.testing.assert_close(merged, torch.tensor([[7.5], [27.5]]))
    assert details["merge_pairs"] == [(0, 1), (2, 3)]
    merged.sum().backward()
    torch.testing.assert_close(audio.grad, torch.tensor([[.25], [.75], [.25], [.75]]))
