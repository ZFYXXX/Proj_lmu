"""Image position resolution retains compound steps and legacy recipes."""

from pathlib import Path
from types import SimpleNamespace

import pytest

from mllm_comp.compressors.image.positions import tome_encoder_positions
from mllm_comp.compressors.multimodal_registry import builtin_multimodal_compression_contracts


@pytest.mark.parametrize("backbone", ["llava_onevision", "qwen25_vl", "qwen3_vl"])
@pytest.mark.parametrize("algorithm", ["visionzip", "prumerge", "gap", "divprune", "aim", "vispruner"])
def test_projector_sites_preserve_steps_and_default_evidence(backbone, algorithm):
    from mllm_comp.compressors.image.positions import image_projector_sites
    binding = builtin_multimodal_compression_contracts().binding(f"{algorithm}.image.{backbone}")
    stage = SimpleNamespace(operation={})
    assert binding.position_resolver(stage=stage, binding=binding) is binding
    stage.operation = {"target_site": "projector_output"}
    assert binding.position_resolver(stage=stage, binding=binding) is binding
    stage.operation = {"target_site": "projector_input"}
    sites = image_projector_sites(stage, algorithm, backbone)
    resolved = binding.position_resolver(stage=stage, binding=binding)
    assert all(site.startswith("pre_") for site in sites)
    assert tuple(target.name for target in resolved.targets) == tuple(target.name for target in binding.targets)
    assert tuple(target.target_site for target in resolved.targets) == sites
    assert resolved.metadata["position_changed"] is True
    assert resolved.supported_phases == binding.supported_phases
    if backbone == "qwen3_vl":
        stage.operation = {"targets": {binding.targets[0].name: {"target_site": "projector_input"}}}
        assert image_projector_sites(stage, algorithm, backbone) == (sites[0], binding.targets[1].target_site)
        stage.operation = {"targets": {binding.targets[1].name: {"target_site": "projector_input"}}}
        assert image_projector_sites(stage, algorithm, backbone) == (binding.targets[0].target_site, sites[1])
    for invalid in ({"layers": [0]}, {"layer_edge": "before"}, {"target_site": "llm_decoder_layer_output_image_tokens"},
            {"targets": {"unknown": {}}}, {"targets": {binding.targets[0].name: {"layers": [0]}}}):
        stage.operation = invalid
        with pytest.raises(ValueError):
            image_projector_sites(stage, algorithm, backbone)


@pytest.mark.parametrize("backbone", ["llava_onevision", "qwen25_vl", "qwen3_vl"])
def test_tome_compound_steps_and_equivalent_boundaries(backbone):
    binding = builtin_multimodal_compression_contracts().binding(f"tome.image.{backbone}")
    prefix = "vision" if backbone == "llava_onevision" else "visual"
    values = {"target_site": f"{prefix}_block_output", "layers": [1]}
    stage = SimpleNamespace(operation={"targets": {"encoder_merge": values}})
    resolved = binding.position_resolver(stage=stage, binding=binding)
    assert resolved.targets[1:] == binding.targets[1:]
    assert resolved.targets[0].name == "encoder_merge"
    assert resolved.metadata["position_changed"] is True
    positions = tome_encoder_positions(stage, backbone, 3)
    values.update(layers=[2], layer_edge="before")
    assert tome_encoder_positions(stage, backbone, 3) == positions
    assert len(positions) == 1 and positions[0].index == 2
    assert binding.position_resolver(stage=SimpleNamespace(operation={}), binding=binding) is binding


@pytest.mark.parametrize("relative", [
    "configs/train/imagellm/qwen25_vl_tome_train_step.yaml",
    "configs/eval/lmms_eval/qwen25_vl_imagellm_tome_forward_probe.yaml",
])
def test_existing_qwen_tome_yaml_keeps_default_position(relative):
    import yaml
    raw = yaml.safe_load((Path(__file__).resolve().parents[2] / relative).read_text(encoding="utf-8"))
    stage = next(item for item in raw["pipeline"]["stages"] if item.get("algorithm") == "tome")
    binding = builtin_multimodal_compression_contracts().binding(stage["binding"])
    assert binding.position_resolver(stage=SimpleNamespace(**stage), binding=binding) is binding


@pytest.mark.parametrize("values", [
    {"layers": [3]}, {"layers": []}, {"layers": [1, 1]}, {"layers": [True]},
    {"target_site": "unknown"}, {"layer_edge": "before"},
    {"targets": {"dense_restore": {"layers": [1]}}},
    {"layers": [1], "targets": {"encoder_merge": {"layers": [2]}}},
])
def test_invalid_tome_position_is_rejected(values):
    with pytest.raises(ValueError):
        tome_encoder_positions(SimpleNamespace(operation=values), "qwen3_vl", 3)


@pytest.mark.parametrize("source,layers,edge", [(True, [1], None), (-1, [1], None), (1, [0], "before"), (2, [0], None), (3, [2], None)])
def test_tome_fixed_score_source_rejects_invalid_or_future_layer(source, layers, edge):
    from mllm_comp.compressors.image.positions import tome_encoder_score_layer
    values = {"target_site": "visual_block_output", "layers": layers}
    if edge:
        values["layer_edge"] = edge
    with pytest.raises(ValueError):
        tome_encoder_score_layer(SimpleNamespace(operation=values, criterion={"score_layer": source}), "qwen3_vl", 3)


def test_tome_fixed_source_is_independent_of_application_and_marks_evidence():
    from mllm_comp.compressors.image.positions import tome_encoder_score_layer
    binding = builtin_multimodal_compression_contracts().binding("tome.image.qwen3_vl")
    stage = SimpleNamespace(operation={"layers": [1, 2]}, criterion={"score_layer": 0})
    assert tome_encoder_score_layer(stage, "qwen3_vl", 3) == 0
    resolved = binding.position_resolver(stage=stage, binding=binding)
    assert resolved.targets[0].metadata["score_layer"] == 0
    assert resolved.targets[1:] == binding.targets[1:]
    assert resolved.metadata["position_changed"] is True


@pytest.mark.parametrize("backbone", ["llava_onevision", "qwen25_vl", "qwen3_vl"])
def test_same_inter_layer_boundary_can_compute_its_own_input_keys(backbone):
    from mllm_comp.compressors.image.positions import tome_encoder_score_layer
    prefix = "vision" if backbone == "llava_onevision" else "visual"
    stage = SimpleNamespace(operation={"target_site": f"{prefix}_block_output", "layers": [1], "layer_edge": "before"}, criterion={"score_layer": 1}, config={"r": 1})
    assert tome_encoder_score_layer(stage, backbone, 3) == 1


@pytest.mark.parametrize("backbone", ["llava_onevision", "qwen25_vl", "qwen3_vl"])
def test_fastv_layer_entry_and_previous_output_share_one_boundary(backbone):
    from mllm_comp.compressors.image.positions import decoder_token_position
    binding = builtin_multimodal_compression_contracts().binding(f"fastv.image.{backbone}")
    stage = SimpleNamespace(operation={}, config={"prune_layer": 2})
    assert binding.position_resolver(stage=stage, binding=binding) is binding
    canonical = decoder_token_position(stage, 2, 4)
    stage.operation = {"target_site": "llm_decoder_layer_output_image_tokens", "layers": [1]}
    assert decoder_token_position(stage, 2, 4) == canonical
    assert binding.position_resolver(stage=stage, binding=binding).metadata["position_changed"] is False
    stage.operation.update(layers=[2], layer_edge="before")
    assert decoder_token_position(stage, 2, 4) == canonical
    stage.operation.update(layers=[0])
    assert decoder_token_position(stage, 1, 4).index == 0
    stage.operation.update(layers=[4])
    assert decoder_token_position(stage, 2, 4).index == 4


@pytest.mark.parametrize("values,source", [
    ({"layers": []}, 2), ({"layers": [1, 2]}, 2), ({"layers": [True]}, 2),
    ({"layers": [5]}, 2), ({"layers": [0]}, 2), ({"layers": [4]}, 5),
    ({"layer_edge": "before"}, 2), ({"layers": [1], "layer_edge": "after"}, 2),
    ({"targets": {"other_step": {"layers": [2]}}}, 2),
])
def test_fastv_invalid_or_unavailable_boundaries_are_rejected(values, source):
    from mllm_comp.compressors.image.positions import decoder_token_position
    with pytest.raises(ValueError):
        decoder_token_position(SimpleNamespace(operation=values), source, 4)


def test_fastv_resolver_uses_existing_config_precedence():
    binding = builtin_multimodal_compression_contracts().binding("fastv.image.qwen3_vl")
    stage = SimpleNamespace(operation={"layers": [1], "prune_layer": 0}, criterion={"prune_layer": 1})
    resolved = binding.position_resolver(stage=stage, binding=binding)
    assert resolved.metadata["position_changed"] is False
    assert resolved.targets[0].metadata["score_layer"] == 0


def test_fastv_mid_is_a_separate_residual_boundary_and_rejects_future_sources():
    from mllm_comp.compressors.image.positions import decoder_token_position
    stage = SimpleNamespace(operation={"target_site": "llm_decoder_layer_post_attention_image_tokens", "layers": [0]})
    assert decoder_token_position(stage, 1, 4).as_dict() == {"boundary": "post_attention_pre_ffn", "index": 0}
    with pytest.raises(ValueError, match="unavailable"):
        decoder_token_position(stage, 2, 4)
    stage.operation["layers"] = [4]
    with pytest.raises(ValueError, match="outside"):
        decoder_token_position(stage, 2, 4)
    stage.operation.update(layers=[1], layer_edge="before")
    with pytest.raises(ValueError, match="full-layer"):
        decoder_token_position(stage, 2, 4)


def test_dart_requires_completed_source_and_accepts_terminal_source():
    from mllm_comp.compressors.image.positions import decoder_token_position
    def stage(index, mid=False):
        return SimpleNamespace(operation={"target_site": "llm_decoder_layer_post_attention_image_tokens" if mid else "llm_decoder_layer_output_image_tokens",
            "layers": [index], **({} if mid else {"layer_edge": "before"})})
    assert decoder_token_position(stage(4), 4, 4, algorithm="dart").index == 4
    assert decoder_token_position(stage(1, True), 1, 4, algorithm="dart").boundary == "post_attention_pre_ffn"
    for value, source in [(stage(0), 1), (stage(0, True), 1), (stage(1), 2), (stage(1, True), 2)]:
        with pytest.raises(ValueError, match="completed scoring layer"):
            decoder_token_position(value, source, 4, algorithm="dart")


@pytest.mark.parametrize("backbone", ["qwen3_vl", "qwen25_vl", "llava_onevision"])
def test_dart_public_factory_preserves_invalid_config_exception(backbone):
    from mllm_comp.compressors.multimodal_registry import builtin_multimodal_compression_contracts
    from mllm_comp.core.errors import BackendIntegrationError
    from mllm_comp.pipeline.stage import CompressionStageConfig
    stage = CompressionStageConfig.from_dict({"name": "dart-invalid", "placement": "llm_level",
        "execution_phase": "forward", "plugin": "dart", "algorithm": "dart", "variant": "dart.image",
        "binding": f"dart.image.{backbone}", "target_modalities": ["image"], "config": {"retain_rato": 0.5}})
    binding = builtin_multimodal_compression_contracts().binding(stage.binding)
    # Config is rejected before a provider or model is inspected or installed.
    with pytest.raises(BackendIntegrationError, match="retain_rato"):
        binding.executor_factory(model=SimpleNamespace(), runtime=SimpleNamespace(metadata={}), facade=None,
            stage=stage, binding=binding, binding_config=None, execution_phase=stage.execution_phase)


def test_sgl_fractional_default_does_not_limit_external_guide_positions():
    from mllm_comp.compressors.image.positions import decoder_token_position
    for index, mid in [(0, False), (0, True), (4, False)]:
        stage = SimpleNamespace(operation={"target_site": "llm_decoder_layer_post_attention_image_tokens" if mid else "llm_decoder_layer_output_image_tokens",
            "layers": [index], **({} if mid else {"layer_edge": "before"})})
        assert decoder_token_position(stage, 2, 4, algorithm="sgl").index == index
    with pytest.raises(ValueError):
        decoder_token_position(SimpleNamespace(operation={"layers": [1, 2]}), 2, 4, algorithm="sgl")


@pytest.mark.parametrize("backbone", ["qwen3_vl", "qwen25_vl", "llava_onevision"])
def test_ivc_position_keeps_completed_source_and_final_binding_config(backbone):
    from mllm_comp.compressors.image.positions import decoder_token_position
    binding = builtin_multimodal_compression_contracts().binding(f"ivc_prune.image.{backbone}")
    stage = SimpleNamespace(operation={}, config={"prune_layer": 2, "single_image_only": True})
    assert binding.position_resolver(stage=stage, binding=binding) is binding
    assert "prune_after_deepstack" not in binding.metadata
    if backbone == "qwen3_vl":
        assert binding.legalizer == "qwen3_vl.ivc_native_mrope_compact_cache"
    target = binding.targets[0].name
    values = {"target_site": "llm_decoder_layer_output_image_tokens", "layers": [1]}
    stage.operation = {"targets": {target: values}}
    assert decoder_token_position(stage, 2, 4, algorithm="ivc_prune", backbone=backbone).index == 2
    assert binding.position_resolver(stage=stage, binding=binding).metadata["position_changed"] is False
    selected = SimpleNamespace(binding=binding.name, config={"prune_layer": 1})
    stage.bindings = [selected]
    resolved = binding.position_resolver(stage=stage, binding=binding)
    assert resolved.metadata["position_changed"] is True
    assert resolved.targets[0].metadata["score_layer"] == 0
    assert binding.position_resolver(stage=stage, binding=binding, binding_config=None).metadata["position_changed"] is False
    values.update(layers=[0], layer_edge="before")
    with pytest.raises(ValueError, match="completed scoring layer"):
        decoder_token_position(stage, 1, 4, algorithm="ivc_prune", backbone=backbone)
    values.update(target_site="llm_decoder_layer_post_attention_image_tokens", layers=[1])
    values.pop("layer_edge")
    assert decoder_token_position(stage, 1, 4, algorithm="ivc_prune", backbone=backbone).boundary == "post_attention_pre_ffn"
    with pytest.raises(ValueError, match="completed scoring layer"):
        decoder_token_position(stage, 2, 4, algorithm="ivc_prune", backbone=backbone)
    values["layers"] = [4]
    with pytest.raises(ValueError, match="outside"):
        decoder_token_position(stage, 2, 4, algorithm="ivc_prune", backbone=backbone)
    values.update(target_site="llm_decoder_layer_output_image_tokens", layer_edge="before")
    assert decoder_token_position(stage, 4, 4, algorithm="ivc_prune", backbone=backbone).index == 4


@pytest.mark.parametrize("source", [True, 1.5, "2"])
def test_ivc_non_integer_source_is_rejected_even_with_an_explicit_valid_application(source):
    binding = builtin_multimodal_compression_contracts().binding("ivc_prune.image.qwen3_vl")
    stage = SimpleNamespace(operation={"layers": [3]}, config={"prune_layer": source})
    with pytest.raises(TypeError, match="integer"):
        binding.position_resolver(stage=stage, binding=binding)


@pytest.mark.parametrize("backbone", ["qwen3_vl", "qwen25_vl", "llava_onevision"])
def test_feather_two_stages_keep_order_sources_and_binding_precedence(backbone):
    from mllm_comp.compressors.image.positions import feather_decoder_positions
    binding = builtin_multimodal_compression_contracts().binding(f"feather.image.{backbone}")
    stage = SimpleNamespace(operation={}, config={"stage_layers": [1, 2], "single_image_only": True})
    assert binding.position_resolver(stage=stage, binding=binding) is binding
    values = {"target_site": "llm_decoder_layer_output_image_tokens", "layers": [0, 1]}
    stage.operation = {"targets": {"two_stage_no_rope_prune": values}}
    assert binding.position_resolver(stage=stage, binding=binding).metadata["position_changed"] is False
    values.update(layers=[2, 3], layer_edge="before")
    resolved = binding.position_resolver(stage=stage, binding=binding)
    assert resolved.targets[0].metadata["score_layers"] == [0, 1]
    assert resolved.metadata["position_changed"] is True
    selected = SimpleNamespace(binding=binding.name, config={"stage_layers": [2, 3]})
    stage.bindings = [selected]
    assert binding.position_resolver(stage=stage, binding=binding).metadata["position_changed"] is False
    assert binding.position_resolver(stage=stage, binding=binding, binding_config=None).metadata["position_changed"] is True
    values.update(layers=[2, 4])
    assert [p.index for p in feather_decoder_positions(stage, (1, 4), 4, backbone=backbone)] == [2, 4]
    for invalid in ([0, 2], [1], [2, 2], [3, 2], [1, 5]):
        values["layers"] = invalid
        with pytest.raises(ValueError):
            feather_decoder_positions(stage, (1, 2), 4, backbone=backbone)
    values.update(target_site="llm_decoder_layer_post_attention_image_tokens", layers=[1, 2])
    values.pop("layer_edge")
    assert all(p.boundary == "post_attention_pre_ffn" for p in feather_decoder_positions(stage, (1, 2), 4, backbone=backbone))
    values["layers"] = [1, 4]
    with pytest.raises(ValueError, match="outside"):
        feather_decoder_positions(stage, (1, 2), 4, backbone=backbone)


@pytest.mark.parametrize("backbone", ["qwen3_vl", "qwen25_vl", "llava_onevision"])
def test_nuwa_stage2_positions_preserve_composite_targets_and_source_precedence(backbone):
    from dataclasses import replace
    from mllm_comp.compressors.image.nuwa import NuwaSpec
    from mllm_comp.compressors.image.positions import nuwa_decoder_position
    binding = builtin_multimodal_compression_contracts().binding(f"nuwa.image.{backbone}")
    stage = SimpleNamespace(operation={}, config={"stage2_after_layer": 2})
    assert binding.position_resolver(stage=stage, binding=binding) is binding
    values = {"target_site": "llm_decoder_layer_output_image_tokens", "layers": [1]}
    stage.operation = {"targets": {"stage2_last_prompt_attention_selection": values}}
    resolved = binding.position_resolver(stage=stage, binding=binding)
    assert resolved.metadata["position_changed"] is False
    assert resolved.targets[:-1] == tuple(
        replace(target, metadata={**target.metadata, "projector_position": "after"})
        for target in binding.targets[:-1]
    )
    assert resolved.targets[-1].metadata["score_layer"] == 1
    selected = SimpleNamespace(binding=binding.name, config={"stage2_after_layer": 1})
    stage.bindings = [selected]
    assert binding.position_resolver(stage=stage, binding=binding).metadata["position_changed"] is True
    assert binding.position_resolver(stage=stage, binding=binding, binding_config=None).metadata["position_changed"] is False
    values.update(layers=[0], layer_edge="before")
    with pytest.raises(ValueError, match="unavailable"):
        nuwa_decoder_position(stage, NuwaSpec(stage2_after_layer=2), 4, backbone=backbone)
    assert nuwa_decoder_position(stage, NuwaSpec(stage2_after_layer=1), 4, backbone=backbone).index == 0
    for spec in (NuwaSpec(random_stage2=True), NuwaSpec(stage2_score="paper_text_cosine")):
        assert nuwa_decoder_position(stage, spec, 4, backbone=backbone).index == 0
        assert spec.stage2_score_layer is None
    values.update(layers=[4])
    assert nuwa_decoder_position(stage, NuwaSpec(stage2_after_layer=4), 4, backbone=backbone).index == 4
    with pytest.raises(ValueError, match="unavailable"):
        nuwa_decoder_position(stage, NuwaSpec(stage2_after_layer=5), 4, backbone=backbone)
    values.update(target_site="llm_decoder_layer_post_attention_image_tokens")
    values.pop("layer_edge")
    with pytest.raises(ValueError, match="outside"):
        nuwa_decoder_position(stage, NuwaSpec(stage2_after_layer=4), 4, backbone=backbone)
    values["layers"] = [0, 1]
    with pytest.raises(ValueError, match="exactly one"):
        nuwa_decoder_position(stage, NuwaSpec(stage2_after_layer=1), 4, backbone=backbone)


@pytest.mark.parametrize("backbone", ["qwen3_vl", "qwen25_vl", "llava_onevision"])
def test_btp_positions_preserve_stage_order_sources_and_binding_precedence(backbone):
    from mllm_comp.compressors.image.btp import BalancedTokenPruningSpec
    from mllm_comp.compressors.image.positions import btp_decoder_positions
    binding = builtin_multimodal_compression_contracts().binding(f"btp.image.{backbone}")
    config = dict(prune_before_layers=[2, 3, 4], retain_ratios=[0.5, 0.5, 0.0],
        attention_fractions=[0.4, 1.0, 0.0], attention_oversample=[1.2, 1.2, 1.0], rebalance_stages=[False, True, False])
    stage = SimpleNamespace(operation={}, config=config)
    spec = BalancedTokenPruningSpec(**config)
    assert binding.position_resolver(stage=stage, binding=binding) is binding
    assert "first_stage_after_deepstack" not in binding.metadata
    values = {"layers": [1, 2, 3], "layer_edge": "before"}
    stage.operation = {"targets": {"decoder_balanced_pruning_boundaries": values}}
    assert binding.position_resolver(stage=stage, binding=binding).metadata["position_changed"] is False
    values.update(layers=[2, 3, 4])
    resolved = binding.position_resolver(stage=stage, binding=binding)
    assert resolved.metadata["position_changed"] is True
    assert resolved.targets[0].metadata["score_layers"] == [0, 1]
    stage.bindings = [SimpleNamespace(binding=binding.name, config={"prune_before_layers": [3, 4, 5]})]
    assert binding.position_resolver(stage=stage, binding=binding).metadata["position_changed"] is False
    assert binding.position_resolver(stage=stage, binding=binding, binding_config=None).metadata["position_changed"] is True
    assert [point.index for point in btp_decoder_positions(stage, spec, 4, backbone=backbone)] == [2, 3, 4]
    for invalid in ([0, 1], [0, 1, 1], [2, 1, 3], [0, 1, 5]):
        values["layers"] = invalid
        with pytest.raises(ValueError):
            btp_decoder_positions(stage, spec, 4, backbone=backbone)
    values.update(target_site="llm_decoder_layer_post_attention_image_tokens", layers=[0, 1, 2])
    values.pop("layer_edge")
    assert all(point.boundary == "post_attention_pre_ffn" for point in btp_decoder_positions(stage, spec, 4, backbone=backbone))
    future = BalancedTokenPruningSpec(**{**config, "prune_before_layers": [3, 4, 5]})
    with pytest.raises(ValueError, match="unavailable"):
        btp_decoder_positions(stage, future, 4, backbone=backbone)


@pytest.mark.parametrize("backbone", ["qwen3_vl", "qwen25_vl", "llava_onevision"])
def test_halfv_positions_preserve_sources_and_binding_precedence(backbone):
    from mllm_comp.compressors.image.halfv import HalfVSpec
    from mllm_comp.compressors.image.positions import halfv_decoder_positions
    binding = builtin_multimodal_compression_contracts().binding(f"halfv.image.{backbone}")
    config = dict(ivr_layer_index=1, ssr_layer_index=2)
    stage = SimpleNamespace(operation={}, config=config)
    spec = HalfVSpec(**config)
    assert binding.position_resolver(stage=stage, binding=binding) is binding
    default_positions = halfv_decoder_positions(stage, spec, 4, backbone=backbone)
    values = {"layers": [1, 2], "layer_edge": "before"}
    stage.operation = {"targets": {"decoder_ivr_ssr_boundaries": values}}
    before = binding.position_resolver(stage=stage, binding=binding)
    assert before.metadata["position_changed"] is False
    assert halfv_decoder_positions(stage, spec, 4, backbone=backbone) == default_positions
    values.update(layers=[0, 1], layer_edge="after")
    assert binding.position_resolver(stage=stage, binding=binding) == before
    assert halfv_decoder_positions(stage, spec, 4, backbone=backbone) == default_positions
    values.update(layers=[2, 4], layer_edge="before")
    resolved = binding.position_resolver(stage=stage, binding=binding)
    assert resolved.metadata["position_changed"] is True
    assert resolved.targets[0].metadata["score_layers"] == [0, 1]
    stage.bindings = [SimpleNamespace(binding=binding.name, config={"ivr_layer_index": 2, "ssr_layer_index": 4})]
    assert binding.position_resolver(stage=stage, binding=binding).metadata["position_changed"] is False
    assert binding.position_resolver(stage=stage, binding=binding, binding_config=None).metadata["position_changed"] is True
    assert [p.index for p in halfv_decoder_positions(stage, spec, 4, backbone=backbone)] == [2, 4]
    for invalid in ([0, 2], [1], [2, 1], [1, 1], [1, 5]):
        values["layers"] = invalid
        with pytest.raises(ValueError):
            halfv_decoder_positions(stage, spec, 4, backbone=backbone)
    values.update(target_site="llm_decoder_layer_post_attention_image_tokens", layers=[1, 3])
    values.pop("layer_edge")
    assert all(p.boundary == "post_attention_pre_ffn" for p in halfv_decoder_positions(stage, spec, 4, backbone=backbone))
    with pytest.raises(ValueError, match="scoring"):
        halfv_decoder_positions(stage, HalfVSpec(ivr_layer_index=2, ssr_layer_index=3), 4, backbone=backbone)


@pytest.mark.parametrize("backbone", ["qwen3_vl", "qwen25_vl", "llava_onevision"])
def test_pyramid_fixed_sources_and_application_positions(backbone):
    from mllm_comp.compressors.image.positions import pyramid_decoder_positions
    binding = builtin_multimodal_compression_contracts().binding(f"pyramid_drop.image.{backbone}")
    stage = SimpleNamespace(operation={}, config={"layer_indices": [0, 1, 2], "num_stages": 4})
    assert binding.position_resolver(stage=stage, binding=binding) is binding
    stage.operation = {"layers": [1, 2, 3], "layer_edge": "before"}
    resolved = binding.position_resolver(stage=stage, binding=binding)
    assert resolved.metadata["position_changed"] is (backbone == "llava_onevision")
    stage.operation["layers"] = [2, 3, 4]
    assert [p.index for p in pyramid_decoder_positions(stage, (0, 1, 2), 4, backbone=backbone)] == [2, 3, 4]
    stage.bindings = [SimpleNamespace(binding=binding.name, config={"layer_indices": [1, 2, 3]})]
    assert binding.position_resolver(stage=stage, binding=binding).metadata["position_changed"] is (backbone == "llava_onevision")
    assert binding.position_resolver(stage=stage, binding=binding, binding_config=None).metadata["position_changed"] is True
    for invalid in ([0, 2, 3], [1, 2], [3, 2, 1], [1, 1, 3], [1, 2, 5]):
        stage.operation["layers"] = invalid
        with pytest.raises(ValueError):
            pyramid_decoder_positions(stage, (0, 1, 2), 4, backbone=backbone)


@pytest.mark.parametrize("backbone", ["qwen3_vl", "qwen25_vl", "llava_onevision"])
def test_finepruner_iac_features_and_arp_source_availability(backbone):
    from mllm_comp.compressors.image.finepruner import FinePrunerSpec
    from mllm_comp.compressors.image.positions import finepruner_decoder_positions
    binding = builtin_multimodal_compression_contracts().binding(f"finepruner.image.{backbone}")
    config = dict(iac_after_layers=[1, 2], iac_hash_bits=[2, 1], arp_center_after_layer=3, arp_layer_offsets=[0, 1])
    stage = SimpleNamespace(operation={}, config=config)
    spec = FinePrunerSpec(**config)
    assert binding.position_resolver(stage=stage, binding=binding) is binding
    assert "first_stage_after_deepstack" not in binding.metadata
    values = {"layers": [1, 2, 4], "layer_edge": "before"}
    stage.operation = {"targets": {"decoder_iac_arp_boundaries": values}}
    assert binding.position_resolver(stage=stage, binding=binding).metadata["position_changed"] is False
    values["layers"] = [0, 1, 3]
    resolved = binding.position_resolver(stage=stage, binding=binding)
    assert resolved.metadata["position_changed"] is True
    assert resolved.targets[0].metadata["score_layers"] == [2, 3]
    assert [p.index for p in finepruner_decoder_positions(stage, spec, 4, backbone=backbone)] == [0, 1, 3]
    for invalid in ([0, 1, 2], [0, 1], [2, 1, 3], [0, 0, 3], [1, 2, 5]):
        values["layers"] = invalid
        with pytest.raises(ValueError):
            finepruner_decoder_positions(stage, spec, 4, backbone=backbone)
    values.update(layers=[1, 2, 3])
    stage.bindings = [SimpleNamespace(binding=binding.name, config={"arp_layer_offsets": [0]})]
    assert binding.position_resolver(stage=stage, binding=binding).metadata["position_changed"] is False
    assert binding.position_resolver(stage=stage, binding=binding, binding_config=None).metadata["position_changed"] is True
    values.update(target_site="llm_decoder_layer_post_attention_image_tokens", layers=[0, 1, 3])
    values.pop("layer_edge")
    assert all(p.boundary == "post_attention_pre_ffn" for p in finepruner_decoder_positions(stage, spec, 4, backbone=backbone))
    values["layers"] = [0, 1, 2]
    with pytest.raises(ValueError, match="scoring"):
        finepruner_decoder_positions(stage, spec, 3, backbone=backbone)


@pytest.mark.parametrize("backbone", ["qwen3_vl", "qwen25_vl", "llava_onevision"])
def test_transprune_completed_sources_stage_positions_and_precedence(backbone):
    from mllm_comp.compressors.image.transprune import TransPruneSpec
    from mllm_comp.compressors.image.positions import transprune_decoder_positions
    binding = builtin_multimodal_compression_contracts().binding(f"transprune.image.{backbone}")
    config = dict(accumulating_layers=[1, 2, 3, 4], prune_after_layers=[1, 2, 3], retain_ratios=[0.75, 0.5, 0.25])
    stage = SimpleNamespace(operation={}, config=config)
    spec = TransPruneSpec(**config)
    assert binding.position_resolver(stage=stage, binding=binding) is binding
    default_positions = transprune_decoder_positions(stage, spec, 4, backbone=backbone)
    values = {"layers": [1, 2, 3], "layer_edge": "before"}
    stage.operation = {"targets": {"decoder_ttv_iga_boundaries": values}}
    before = binding.position_resolver(stage=stage, binding=binding)
    assert before.metadata["position_changed"] is False
    assert transprune_decoder_positions(stage, spec, 4, backbone=backbone) == default_positions
    values.update(layers=[0, 1, 2], layer_edge="after")
    assert binding.position_resolver(stage=stage, binding=binding) == before
    assert transprune_decoder_positions(stage, spec, 4, backbone=backbone) == default_positions
    values.update(layers=[2, 3, 4], layer_edge="before")
    resolved = binding.position_resolver(stage=stage, binding=binding)
    assert resolved.metadata["position_changed"] is True
    assert resolved.targets[0].metadata["score_layers"] == [0, 1, 2]
    stage.bindings = [SimpleNamespace(binding=binding.name, config={"prune_after_layers": [2, 3, 4]})]
    assert binding.position_resolver(stage=stage, binding=binding).metadata["position_changed"] is False
    assert binding.position_resolver(stage=stage, binding=binding, binding_config=None).metadata["position_changed"] is True
    assert [p.index for p in transprune_decoder_positions(stage, spec, 4, backbone=backbone)] == [2, 3, 4]
    for invalid in ([0, 2, 3], [1, 2], [2, 1, 3], [1, 1, 3], [1, 2, 5]):
        values["layers"] = invalid
        with pytest.raises(ValueError):
            transprune_decoder_positions(stage, spec, 4, backbone=backbone)
    values.update(target_site="llm_decoder_layer_post_attention_image_tokens", layers=[1, 2, 3])
    values.pop("layer_edge")
    assert all(p.boundary == "post_attention_pre_ffn" for p in transprune_decoder_positions(stage, spec, 4, backbone=backbone))
    with pytest.raises(ValueError):
        transprune_decoder_positions(stage, spec, 3, backbone=backbone)


@pytest.mark.parametrize("backbone", ["qwen3_vl", "qwen25_vl", "llava_onevision"])
def test_pact_fixed_source_and_actual_application(backbone):
    from mllm_comp.compressors.image.pact import PACTSpec
    from mllm_comp.compressors.image.positions import pact_decoder_position
    binding = builtin_multimodal_compression_contracts().binding(f"pact.image.{backbone}")
    spec = PACTSpec(reduction_layer=1)
    stage = SimpleNamespace(operation={}, config={"reduction_layer": 1})
    assert binding.position_resolver(stage=stage, binding=binding) is binding
    values = {"layers": [1], "layer_edge": "before"}
    stage.operation = {"targets": {"decoder_pact_boundary": values}}
    assert binding.position_resolver(stage=stage, binding=binding).metadata["position_changed"] is False
    values["layers"] = [4]
    assert pact_decoder_position(stage, spec, 4).index == 4
    assert binding.position_resolver(stage=stage, binding=binding).targets[0].metadata["score_layers"] == [0]
    stage.bindings = [SimpleNamespace(binding=binding.name, config={"reduction_layer": 4})]
    assert binding.position_resolver(stage=stage, binding=binding).metadata["position_changed"] is False
    assert binding.position_resolver(stage=stage, binding=binding, binding_config=None).metadata["position_changed"] is True
    for invalid in ([0], [1, 2], [5], [-1], [True]):
        values["layers"] = invalid
        with pytest.raises(ValueError):
            pact_decoder_position(stage, spec, 4)
    values.update(target_site="llm_decoder_layer_post_attention_image_tokens", layers=[1])
    values.pop("layer_edge")
    assert pact_decoder_position(stage, spec, 4).boundary == "post_attention_pre_ffn"
    with pytest.raises(ValueError, match="scoring"):
        pact_decoder_position(stage, PACTSpec(reduction_layer=4), 4)


@pytest.mark.parametrize("backbone", ["qwen3_vl", "qwen25_vl", "llava_onevision"])
def test_sparsevlm_sources_initial_terminal_and_precedence(backbone):
    from mllm_comp.compressors.image.sparsevlm import SparseVLMSpec
    from mllm_comp.compressors.image.positions import sparsevlm_decoder_positions

    binding = builtin_multimodal_compression_contracts().binding(f"sparsevlm.image.{backbone}")
    config = {"prune_layers": [0, 2], "schedule_keep_tokens": [2, 1]}
    stage = SimpleNamespace(operation={}, config=config)
    spec = SparseVLMSpec(**config)
    assert binding.position_resolver(stage=stage, binding=binding) is binding
    values = {"layers": [1, 3], "layer_edge": "before"}
    stage.operation = {"targets": {"decoder_merge_boundaries": values}}
    assert binding.position_resolver(stage=stage, binding=binding).metadata["position_changed"] is False
    values["layers"] = [0, 4]
    assert [p.index for p in sparsevlm_decoder_positions(stage, spec, 4, backbone=backbone)] == [0, 4]
    resolved = binding.position_resolver(stage=stage, binding=binding)
    assert resolved.targets[0].metadata["score_layers"] == [0, 2]
    values["layers"] = [2, 4]
    stage.bindings = [SimpleNamespace(binding=binding.name, config={"prune_layers": [1, 3]})]
    assert binding.position_resolver(stage=stage, binding=binding).metadata["position_changed"] is False
    assert binding.position_resolver(stage=stage, binding=binding, binding_config=None).metadata["position_changed"] is True
    for invalid in ([0, 1], [1], [1, 1], [2, 1], [0, 5]):
        values["layers"] = invalid
        with pytest.raises(ValueError):
            sparsevlm_decoder_positions(stage, spec, 4, backbone=backbone)
    values.update(target_site="llm_decoder_layer_post_attention_image_tokens", layers=[0, 2])
    values.pop("layer_edge")
    assert all(p.boundary == "post_attention_pre_ffn" for p in sparsevlm_decoder_positions(stage, spec, 4, backbone=backbone))
    with pytest.raises(ValueError):
        sparsevlm_decoder_positions(stage, spec, 2, backbone=backbone)


@pytest.mark.parametrize("backbone", ["qwen3_vl", "llava_onevision"])
def test_saint_packed_projector_steps_and_default_evidence(backbone):
    test_projector_sites_preserve_steps_and_default_evidence(backbone, "saint")


@pytest.mark.parametrize("backbone", ["qwen3_vl", "qwen25_vl", "llava_onevision"])
@pytest.mark.parametrize("algorithm", ["mmtok", "zoo_prune"])
def test_mmtok_zoo_projector_steps_and_default_evidence(backbone, algorithm):
    test_projector_sites_preserve_steps_and_default_evidence(backbone, algorithm)


def test_saint_encoder_module_identity_and_initial_mid_terminal_positions():
    from mllm_comp.compressors.image.positions import saint_encoder_positions
    binding = builtin_multimodal_compression_contracts().binding("saint.image.qwen25_vl")
    sources = (0, 2)
    stage = SimpleNamespace(config={"layer_indices": list(sources)}, operation={})
    assert binding.position_resolver(stage=stage, binding=binding) is binding
    for site, layers, edge, expected in (
        ("visual_block_output", [0, 2], "after", [("inter_layer", 1), ("inter_layer", 3)]),
        ("visual_block_output", [0, 4], "before", [("inter_layer", 0), ("inter_layer", 4)]),
        ("visual_block_post_attention_residual_pre_mlp", [0, 3], None, [("post_attention_pre_ffn", 0), ("post_attention_pre_ffn", 3)]),
    ):
        values = {"target_site": site, "layers": layers, **({"layer_edge": edge} if edge else {})}
        stage.operation = {"targets": {"encoder_progressive_prune": values}}
        points = saint_encoder_positions(stage, sources, 4)
        assert [(point.boundary, point.index) for point in points] == expected
        resolved = binding.position_resolver(stage=stage, binding=binding)
        assert resolved.targets[0].metadata["key_projection_layers"] == list(sources)
        assert resolved.metadata["position_changed"] is (layers != [0, 2])
    for invalid in ({"layers": [0]}, {"layers": [0, 4], "target_site": "visual_block_post_attention_residual_pre_mlp"},
            {"layers": [0, 0]}, {"layers": [False, 2]}, {"target_site": "projector_input"}):
        stage.operation = invalid
        with pytest.raises(ValueError):
            saint_encoder_positions(stage, sources, 4)
    stage.operation = {"layers": [0, 1], "layer_edge": "before"}
    stage.bindings = [SimpleNamespace(binding=binding.name, config={"layer_indices": [2, 3]})]
    assert binding.position_resolver(stage=stage, binding=binding).targets[0].metadata["key_projection_layers"] == [2, 3]
    assert binding.position_resolver(stage=stage, binding=binding, binding_config=None).targets[0].metadata["key_projection_layers"] == [0, 2]
    with pytest.raises(ValueError, match="projection layer"):
        saint_encoder_positions(stage, (0, 4), 4)


@pytest.mark.parametrize("backbone", ["qwen3_vl", "qwen25_vl", "llava_onevision"])
@pytest.mark.parametrize("algorithm", ["nuwa", "restore"])
def test_compound_projector_positions_preserve_default_and_later_steps(backbone, algorithm):
    from mllm_comp.compressors.image.positions import image_projector_sites, nuwa_decoder_position
    from mllm_comp.compressors.image.nuwa import NuwaSpec
    binding = builtin_multimodal_compression_contracts().binding(f"{algorithm}.image.{backbone}")
    stage = SimpleNamespace(operation={}, config={"stage2_after_layer": 1} if algorithm == "nuwa" else {})
    assert binding.position_resolver(stage=stage, binding=binding) is binding
    count = 2 if backbone == "qwen3_vl" else 1
    stage.operation = {"target_site": binding.targets[0].target_site}
    assert binding.position_resolver(stage=stage, binding=binding) is binding
    stage.operation = {"target_site": "projector_output"}
    same = binding.position_resolver(stage=stage, binding=binding)
    if backbone == "llava_onevision":
        assert same.targets[0].target_site == "post_vision_projector"
        assert same.metadata["position_changed"] is True
    else:
        assert same is binding
    stage.operation = {"target_site": "projector_input"}
    moved = binding.position_resolver(stage=stage, binding=binding)
    assert tuple(target.name for target in moved.targets) == tuple(target.name for target in binding.targets)
    assert all(target.target_site.startswith("pre_") for target in moved.targets[:count])
    assert moved.targets[-1].target_site == binding.targets[-1].target_site
    assert moved.supported_phases == binding.supported_phases
    if algorithm == "nuwa":
        assert nuwa_decoder_position(stage, NuwaSpec(stage2_after_layer=1), 4, backbone=backbone).index == 1
        stage.operation = {"targets": {
            binding.targets[0].name: {"target_site": "projector_input"},
            binding.targets[-1].name: {"target_site": "llm_decoder_layer_post_attention_image_tokens", "layers": [2]},
        }}
        mixed = binding.position_resolver(stage=stage, binding=binding)
        assert mixed.targets[0].target_site.startswith("pre_")
        assert mixed.targets[-1].metadata["application_positions"] == [{"boundary": "post_attention_pre_ffn", "index": 2}]
        if count == 2:
            assert mixed.targets[1].target_site == binding.targets[1].target_site
        stage.operation = {"layers": [2], "layer_edge": "before"}
        assert image_projector_sites(stage, algorithm, backbone) == tuple(target.target_site for target in binding.targets[:count])
    else:
        stage.operation = {"targets": {binding.targets[-1].name: {"target_site": "llm_decoder_attention_calibration"}}}
        assert binding.position_resolver(stage=stage, binding=binding) is binding
        stage.operation["targets"][binding.targets[-1].name]["layers"] = [0]
        with pytest.raises(ValueError, match="attention logits"):
            image_projector_sites(stage, algorithm, backbone)
    for invalid in ({"target_site": "projector_input", "layers": [0]}, {"targets": {"unknown": {}}}):
        stage.operation = invalid
        with pytest.raises(ValueError):
            binding.position_resolver(stage=stage, binding=binding)


@pytest.mark.parametrize("algorithm,alias", [
    ("pyramid_drop", "llm_decoder_layer_loop"),
    ("visionzip", "post_encoder_pre_llm"),
])
def test_existing_qwen25_probe_aliases_preserve_default_targets_and_metadata(algorithm, alias):
    import yaml
    from mllm_comp.compressors.image.positions import image_projector_sites, pyramid_decoder_positions
    path = Path(__file__).resolve().parents[2] / f"configs/eval/lmms_eval/qwen25_vl_imagellm_{algorithm}_forward_probe.yaml"
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    original = next(item for item in raw["pipeline"]["stages"] if item.get("algorithm") == algorithm)
    stage = SimpleNamespace(**original)
    assert stage.operation["target_site"] == alias
    binding = builtin_multimodal_compression_contracts().binding(stage.binding)
    default = SimpleNamespace(**{**original, "operation": {k: v for k, v in stage.operation.items() if k != "target_site"}})
    expected = binding.position_resolver(stage=default, binding=binding)
    actual = binding.position_resolver(stage=stage, binding=binding)
    assert actual is expected is binding
    assert actual.targets == expected.targets and actual.metadata == expected.metadata
    assert actual.supported_phases == expected.supported_phases
    if algorithm == "pyramid_drop":
        assert pyramid_decoder_positions(stage, (0, 2), 4, backbone="qwen25_vl") == pyramid_decoder_positions(
            default, (0, 2), 4, backbone="qwen25_vl")
    else:
        assert image_projector_sites(stage, algorithm, "qwen25_vl") == image_projector_sites(default, algorithm, "qwen25_vl")


@pytest.mark.parametrize("backbone", ["llava_onevision", "qwen3_vl"])
@pytest.mark.parametrize("algorithm,alias", [
    ("pyramid_drop", "llm_decoder_layer_loop"),
    ("visionzip", "post_encoder_pre_llm"),
])
def test_qwen25_probe_aliases_are_not_accepted_by_other_backbones(backbone, algorithm, alias):
    binding = builtin_multimodal_compression_contracts().binding(f"{algorithm}.image.{backbone}")
    with pytest.raises(ValueError, match="not implemented|cannot execute"):
        binding.position_resolver(stage=SimpleNamespace(operation={"target_site": alias}), binding=binding)


@pytest.mark.parametrize("algorithm,alias", [
    ("fastv", "llm_decoder_layer_loop"),
    ("sparsevlm", "llm_decoder_layer_loop"),
    ("aim", "post_encoder_pre_llm"),
    ("prumerge", "post_encoder_pre_llm"),
])
def test_qwen25_probe_aliases_are_not_accepted_by_other_methods(algorithm, alias):
    binding = builtin_multimodal_compression_contracts().binding(f"{algorithm}.image.qwen25_vl")
    with pytest.raises(ValueError, match="not implemented|cannot execute"):
        binding.position_resolver(stage=SimpleNamespace(operation={"target_site": alias}), binding=binding)
