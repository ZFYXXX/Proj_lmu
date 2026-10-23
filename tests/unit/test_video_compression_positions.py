from types import SimpleNamespace

import pytest

from mllm_comp.compressors.multimodal_registry import builtin_multimodal_compression_contracts
from mllm_comp.compressors.video.positions import video_projector_position, video_projector_site


@pytest.mark.parametrize("backbone", ["llava_onevision", "qwen25_vl", "qwen3_vl"])
@pytest.mark.parametrize("algorithm,target", [("vqtoken", "fixed_k_cluster"), ("visionzip", "dominant_contextual_reduce"), ("fastvid", "dyseg_stprune_dtm")])
def test_video_projector_config_preserves_recipe_and_resolves_actual_input(backbone, algorithm, target):
    binding = builtin_multimodal_compression_contracts().binding(f"{algorithm}.video.{backbone}")
    stage = SimpleNamespace(operation={})
    assert binding.position_resolver(stage=stage, binding=binding) is binding
    for site in ("projector_output", binding.targets[0].target_site):
        stage.operation = {"target_site": site}
        assert binding.position_resolver(stage=stage, binding=binding) is binding
    for operation in (
        {"target_site": "projector_input"},
        {"targets": {target: {"target_site": "projector_input"}}},
        {"target_site": video_projector_site(backbone, "before")},
    ):
        stage.operation = operation
        resolved = binding.position_resolver(stage=stage, binding=binding)
        assert video_projector_position(stage, backbone, target_name=target) == "before"
        assert resolved.targets[0].target_site == video_projector_site(backbone, "before")
        assert resolved.targets[0].name == target
        assert resolved.targets[0].required_handles == binding.targets[0].required_handles
        assert resolved.supported_phases == binding.supported_phases
        assert resolved.metadata["position_changed"] is True
        assert resolved.metadata["verified_phases"] == binding.metadata["verified_phases"]
    for operation in (
        {"layers": [0]}, {"layer_edge": "before"}, {"target_site": "llm_decoder_layer_output"},
        {"targets": {"missing": {}}}, {"targets": {target: {"layers": [0]}}},
        {"target_site": "projector_input", "targets": {target: {"target_site": "projector_output"}}},
    ):
        stage.operation = operation
        with pytest.raises(ValueError):
            binding.position_resolver(stage=stage, binding=binding)


@pytest.mark.parametrize("backbone", ["llava_onevision", "qwen25_vl", "qwen3_vl"])
def test_vqtoken_input_runtime_does_not_claim_original_checkpoint_evidence(backbone):
    from mllm_comp.backbones.vqtoken_video import VQTokenVideoExecutor

    binding = builtin_multimodal_compression_contracts().binding(f"vqtoken.video.{backbone}")
    stage = SimpleNamespace(algorithm="vqtoken", variant="vqtoken.video", operation={"target_site": "projector_input"})
    executor = VQTokenVideoExecutor(backbone=backbone, model=object(), stage=stage, binding=binding)
    metadata = executor.runtime_metadata()
    assert metadata["projector_position"] == "before"
    assert metadata["target_site"] == video_projector_site(backbone, "before")
    assert metadata["canonical_evidence_applicable"] is False
    assert metadata["real_checkpoint_verified"] is False
    assert metadata["support_activation_claimed"] is False
    assert metadata["evidence_artifact"] is None


@pytest.mark.parametrize("backbone", ["llava_onevision", "qwen25_vl", "qwen3_vl"])
@pytest.mark.parametrize("algorithm,projector,other,defaults,moved", [
    ("dytok", "dynamic_visionzip_reduce", "detection_budget_provider", list(range(16, 24)), [0, 2]),
    ("dycoke", "ttm_visual_merge", "decode_kv_view_prune", [3], [1]),
])
def test_video_compound_positions_preserve_each_step(backbone, algorithm, projector, other, defaults, moved):
    binding = builtin_multimodal_compression_contracts().binding(f"{algorithm}.video.{backbone}")
    original = {target.name: target for target in binding.targets}
    for operation in ({}, {"targets": {
        projector: {"target_site": "projector_output"},
        other: {"target_site": original[other].target_site, "layers": defaults},
    }}):
        assert binding.position_resolver(stage=SimpleNamespace(operation=operation), binding=binding) is binding
    for change_projector, change_other in ((True, False), (False, True), (True, True)):
        stage = SimpleNamespace(operation={"targets": {
            projector: {"target_site": "projector_input" if change_projector else "projector_output"},
            other: {"layers": moved if change_other else defaults},
        }})
        resolved = binding.position_resolver(stage=stage, binding=binding)
        assert [(t.name, t.order, t.placement, t.required_handles) for t in resolved.targets] == [
            (t.name, t.order, t.placement, t.required_handles) for t in binding.targets]
        assert resolved.dependencies == binding.dependencies
        assert resolved.supported_phases == binding.supported_phases
        assert resolved.metadata["position_changed"] is True
        targets = {target.name: target for target in resolved.targets}
        assert targets[projector].target_site == video_projector_site(backbone, "before" if change_projector else "after")
        assert targets[other].target_site == original[other].target_site
        assert targets[other].metadata["component_role"] == original[other].metadata["component_role"]
        if not change_other:
            assert targets[other] is original[other]
        elif algorithm == "dytok":
            assert targets[other].metadata["score_layers"] == moved
            assert targets[other].metadata["paper_compression_stage"] is False
        else:
            assert targets[other].metadata["score_layer"] == 0
            assert targets[other].metadata["application_positions"] == [{"boundary": "decode_kv_before_attention", "index": 1}]
    for values in ({"layers": []}, {"layers": [True]}, {"layers": [-1]},
                   {"layers": [2, 1]}, {"layers": [1, 1]},
                   {"target_site": "llm_decoder_layer_post_attention_image_tokens"},
                   {"layer_edge": "after"}):
        with pytest.raises(ValueError):
            binding.position_resolver(stage=SimpleNamespace(operation={"targets": {other: values}}), binding=binding)


def test_video_auxiliary_and_kv_layers_validate_real_depth_and_lifecycle():
    from mllm_comp.compressors.video.positions import video_detection_layers, video_decode_prune_layer

    detector = SimpleNamespace(operation={"targets": {"detection_budget_provider": {"layers": [0, 3]}}})
    assert video_detection_layers(detector, tuple(range(16, 24)), 4) == (0, 3)
    with pytest.raises(ValueError, match="depth"):
        video_detection_layers(detector, tuple(range(16, 24)), 3)
    for layer in (1, 3):
        stage = SimpleNamespace(operation={"targets": {"decode_kv_view_prune": {"layers": [layer], "target_site": "decode_kv_before_attention"}}})
        assert video_decode_prune_layer(stage, 3, 4, backbone="qwen3_vl") == layer
    for layers in ([0], [4], [1, 2]):
        with pytest.raises(ValueError):
            video_decode_prune_layer(SimpleNamespace(operation={"targets": {"decode_kv_view_prune": {"layers": layers}}}), 3, 4, backbone="qwen3_vl")
