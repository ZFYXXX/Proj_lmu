"""Learned video modules resolve their own projection boundaries and evidence."""

from types import SimpleNamespace

import pytest

from mllm_comp.compressors.multimodal_registry import builtin_multimodal_compression_contracts
from mllm_comp.compressors.video.positions import (
    video_learned_projector_position,
    video_longvu_frame_selection_site,
)
from mllm_comp.run_manifest import build_run_manifest
from tests.unit.test_manifest_v2_compression_records import _config


@pytest.mark.parametrize("backbone", ["llava_onevision", "qwen25_vl", "qwen3_vl"])
@pytest.mark.parametrize("algorithm,target", [
    ("llamavid", "qformer_context_content_pool"),
    ("longvu", "dino_frame_select_sva_fusion"),
])
def test_learned_module_positions_keep_default_recipe_and_existing_target(backbone, algorithm, target):
    binding = builtin_multimodal_compression_contracts().binding(f"{algorithm}.video.{backbone}")
    for operation in ({}, {"target_site": binding.targets[0].target_site},
                      {"target_site": "projector_input"},
                      {"targets": {target: {"target_site": "projector_input"}}}):
        assert binding.position_resolver(stage=SimpleNamespace(operation=operation), binding=binding) is binding
    moves = ({"target_site": "projector_output"},) if algorithm == "llamavid" else (
        {"frame_selection_site": "pre_output_projector"}, {"frame_selection_site": "post_output_projector"})
    for values in moves:
        stage = SimpleNamespace(operation={"targets": {target: values}})
        resolved = binding.position_resolver(stage=stage, binding=binding)
        assert resolved.metadata["position_changed"] is True
        assert resolved.supported_phases == binding.supported_phases
        assert resolved.dependencies == binding.dependencies
        assert len(resolved.targets) == 1
        assert resolved.targets[0].name == target
        assert resolved.targets[0].required_handles == binding.targets[0].required_handles
        if algorithm == "llamavid":
            assert video_learned_projector_position(stage, backbone, target_name=target) == "after"
            assert resolved.targets[0].target_site == "llamavid_post_output_projectors"
        else:
            assert video_longvu_frame_selection_site(stage, backbone) == values["frame_selection_site"]
            assert resolved.targets[0].target_site == binding.targets[0].target_site
            assert resolved.targets[0].metadata["application_positions"] == [
                {"operation": "dino_frame_selection", "boundary": values["frame_selection_site"]},
                {"operation": "sva_fusion", "boundary": "longvu_pre_output_projector"}]
    invalid = [{"layers": [0]}, {"layer_edge": "after"}, {"target_site": "llm_decoder_layer_output"}]
    if algorithm == "longvu":
        invalid += [{"target_site": "projector_output"}, {"frame_selection_site": "post_sva"}]
    for values in invalid:
        with pytest.raises(ValueError):
            binding.position_resolver(stage=SimpleNamespace(operation={"targets": {target: values}}), binding=binding)


@pytest.mark.parametrize("backbone", ["llava_onevision", "qwen25_vl", "qwen3_vl"])
@pytest.mark.parametrize("algorithm,target", [
    ("llamavid", "qformer_context_content_pool"),
    ("longvu", "dino_frame_select_sva_fusion"),
])
def test_learned_projection_manifest_records_inner_steps_and_evidence(backbone, algorithm, target):
    stage = {"name": "learned-position", "placement": "projector_level", "execution_phase": "forward",
        "plugin": algorithm, "algorithm": algorithm, "variant": f"{algorithm}.video",
        "binding": f"{algorithm}.video.{backbone}", "target_modalities": ["video"]}
    baseline = build_run_manifest(_config(stage, backbone_name=backbone)).compression_stages[0]
    alias = build_run_manifest(_config({**stage, "operation": {"target_site": "projector_input"}}, backbone_name=backbone)).compression_stages[0]
    assert alias.target_sites == baseline.target_sites
    assert alias.evidence == baseline.evidence and alias.support_status == baseline.support_status
    assert alias.metadata == baseline.metadata
    key = f"{algorithm}.video.{backbone}:{target}"
    if algorithm == "llamavid":
        assert baseline.metadata["application_positions"][key][0]["boundary"] == "llamavid_pre_output_projectors"
        values = {"target_site": "projector_output"}
    else:
        assert baseline.metadata["application_positions"][key] == [
            {"operation": "dino_frame_selection", "boundary": "pre_sva"},
            {"operation": "sva_fusion", "boundary": "longvu_pre_output_projector"}]
        values = {"frame_selection_site": "post_output_projector"}
    moved = build_run_manifest(_config({**stage, "operation": {"targets": {target: values}}}, backbone_name=backbone)).compression_stages[0]
    assert moved.metadata["canonical_evidence_applicable"] is False
    assert moved.support_status == "declared"
    assert [{k: v for k, v in item.items() if k != "canonical_evidence_applicable"}
        for item in moved.evidence] == baseline.evidence
    assert moved.metadata["application_positions"][key][0]["boundary"] == (
        "llamavid_post_output_projectors" if algorithm == "llamavid" else "post_output_projector")
