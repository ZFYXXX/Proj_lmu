from types import SimpleNamespace

import pytest

from mllm_comp.compressors.image.positions import tome_encoder_operation, tome_output_site
from mllm_comp.compressors.multimodal_registry import builtin_multimodal_compression_contracts


@pytest.mark.parametrize("backbone", ["qwen3_vl", "qwen25_vl", "llava_onevision"])
def test_tome_output_position_is_independent_and_preserves_structural_steps(backbone):
    binding = builtin_multimodal_compression_contracts().binding(f"tome.image.{backbone}")
    defaults = {target.name: target for target in binding.targets}
    for site in ("projector_output", defaults["permanent_output_merge"].target_site):
        stage = SimpleNamespace(operation={"targets": {"permanent_output_merge": {"target_site": site}}})
        assert binding.position_resolver(stage=stage, binding=binding) is binding
    stage = SimpleNamespace(operation={"targets": {"permanent_output_merge": {"target_site": "projector_input"}}})
    resolved = binding.position_resolver(stage=stage, binding=binding)
    assert [(t.name, t.order, t.placement, t.required_handles) for t in resolved.targets] == [
        (t.name, t.order, t.placement, t.required_handles) for t in binding.targets]
    assert resolved.dependencies == binding.dependencies and resolved.supported_phases == binding.supported_phases
    assert resolved.metadata["position_changed"] is True
    assert resolved.targets[-1].target_site == tome_output_site(stage, backbone)
    assert resolved.targets[-1].target_site.startswith("pre_")
    for target in resolved.targets:
        if "restore" in target.name:
            assert target is defaults[target.name]
    # The existing top-level shorthand still controls Encoder, alongside output.
    stage.operation.update(layers=[0], layer_edge="before", target_site=(
        "vision_block_output" if backbone == "llava_onevision" else "visual_block_output"))
    resolved = binding.position_resolver(stage=stage, binding=binding)
    assert resolved.targets[0].metadata["application_positions"] == [{"boundary": "inter_layer", "index": 0}]
    assert resolved.targets[-1].target_site.startswith("pre_")
    assert tome_encoder_operation(stage, backbone)["layers"] == [0]
    for target in binding.targets:
        if "restore" not in target.name:
            continue
        stage.operation = {"targets": {target.name: {"target_site": target.target_site}}}
        assert binding.position_resolver(stage=stage, binding=binding) is binding
        stage.operation["targets"][target.name]["target_site"] = "projector_input"
        with pytest.raises(ValueError, match="structural consumer"):
            binding.position_resolver(stage=stage, binding=binding)
    for values in ({"layers": [0]}, {"layer_edge": "after"}, {"target_site": "llm_decoder_layer_output"}):
        with pytest.raises(ValueError):
            binding.position_resolver(stage=SimpleNamespace(operation={"targets": {"permanent_output_merge": values}}), binding=binding)


@pytest.mark.parametrize("backbone", ["qwen3_vl", "qwen25_vl", "llava_onevision"])
def test_tome_output_manifest_keeps_default_history_and_both_positions(backbone):
    from mllm_comp.run_manifest import build_run_manifest
    from tests.unit.test_manifest_v2_compression_records import _config

    stage = {"name": "tome-output", "placement": "encoder_level", "execution_phase": "forward",
        "plugin": "tome", "algorithm": "tome", "variant": "tome.image.encoder_dense_boundary",
        "binding": f"tome.image.{backbone}", "target_modalities": ["image"], "config": {"r": 1, "output_r": 1}}
    baseline = build_run_manifest(_config(stage, backbone_name=backbone)).compression_stages[0]
    stage["operation"] = {"targets": {"permanent_output_merge": {"target_site": "projector_output"}}}
    alias = build_run_manifest(_config(stage, backbone_name=backbone)).compression_stages[0]
    assert alias.target_sites == baseline.target_sites
    assert alias.evidence == baseline.evidence and alias.support_status == baseline.support_status
    stage["operation"]["targets"]["permanent_output_merge"]["target_site"] = "projector_input"
    stage["operation"]["targets"]["encoder_merge"] = {"layers": [1]}
    moved = build_run_manifest(_config(stage, backbone_name=backbone)).compression_stages[0]
    assert len(moved.target_sites) == len(baseline.target_sites)
    assert moved.target_sites[:-1] == baseline.target_sites[:-1]
    assert moved.target_sites[-1].startswith("pre_")
    assert moved.metadata["canonical_evidence_applicable"] is False
    assert moved.metadata["application_positions"][f"tome.image.{backbone}:encoder_merge"] == [
        {"boundary": "post_attention_pre_ffn", "index": 1}]
    assert [{key: value for key, value in item.items() if key != "canonical_evidence_applicable"}
        for item in moved.evidence] == baseline.evidence


@pytest.mark.parametrize("config,selection", [
    ({"r": 1, "layers": [1]}, None),
    ({"r_by_layer": {"1": 1}}, None),
    ({"r": 1, "layers": [0]}, {"layers": [1]}),
])
def test_llava_tome_explicit_legacy_layers_keep_exact_evidence(config, selection):
    from mllm_comp.run_manifest import build_run_manifest
    from mllm_comp.pipeline.v2_validation import validate_v2_stage_contract
    from tests.unit.test_manifest_v2_compression_records import _config

    registry = builtin_multimodal_compression_contracts()
    binding = registry.binding("tome.image.llava_onevision")
    stage = {"name": "tome-alias", "placement": "encoder_level", "execution_phase": "forward",
        "plugin": "tome", "algorithm": "tome", "variant": "tome.image.encoder_dense_boundary",
        "target_modalities": ["image"], "preserved_modalities": ["text", "video", "audio"],
        "media_alignment": "spatial", "config": config}
    if selection is None:
        stage["binding"] = binding.name
    else:
        stage["bindings"] = [{"binding": binding.name, "config": selection}]
    baseline = build_run_manifest(_config(stage, backbone_name="llava_onevision")).compression_stages[0]
    values = {"layers": [1]}
    stage["operation"] = {"targets": {"encoder_merge": values}}
    resolved_config = _config(stage, backbone_name="llava_onevision")
    resolved = validate_v2_stage_contract(resolved_config.pipeline.stages[0], registry=registry,
        backbone_name="llava_onevision").bindings[0]
    alias = build_run_manifest(resolved_config).compression_stages[0]
    assert resolved.metadata["position_changed"] is False
    assert alias.evidence == baseline.evidence and alias.support_status == baseline.support_status
    assert alias.metadata["support_claim_is_exact"] is True
    assert alias.metadata["application_positions"][f"{binding.name}:encoder_merge"] == [
        {"boundary": "post_attention_pre_ffn", "index": 1}]
    if selection is not None:
        # An explicit factory selection overrides stage.bindings, including None.
        direct = binding.position_resolver(stage=resolved_config.pipeline.stages[0], binding=binding, binding_config=None)
        assert direct.metadata["position_changed"] is True
    values["layers"] = [2]
    moved = build_run_manifest(_config(stage, backbone_name="llava_onevision")).compression_stages[0]
    assert moved.metadata["canonical_evidence_applicable"] is False
