from __future__ import annotations

import pytest

from mllm_comp.config.schema import experiment_config_from_dict
from mllm_comp.run_manifest import build_run_manifest


@pytest.mark.parametrize("algorithm", ["fastv", "dart", "d2pruner"])
@pytest.mark.parametrize("backbone", ["llava_onevision", "qwen25_vl", "qwen3_vl"])
@pytest.mark.parametrize("phase", ["forward", "generate"])
@pytest.mark.parametrize("mid", [False, True])
def test_manifest_decoder_records_application_separately_from_historical_recipe(backbone, phase, mid, algorithm):
    source_field = "prune_layer" if algorithm == "fastv" else "prune_after_layer"
    site = "llm_decoder_layer_post_attention_image_tokens" if mid else "llm_decoder_layer_output_image_tokens"
    stage = {
        "name": "fastv-position", "placement": "llm_level", "execution_phase": phase,
        "plugin": algorithm, "algorithm": algorithm, "variant": f"{algorithm}.image",
        "binding": f"{algorithm}.image.{backbone}", "target_modalities": ["image"],
        "config": {source_field: 2},
        "operation": {"targets": {"decoder_prune_boundary": {"target_site": site, "layers": [3]}}},
    }
    record = build_run_manifest(_config(stage, backbone_name=backbone)).compression_stages[0]
    assert record.target_sites == [site]
    assert record.metadata["application_positions"] == {f"{algorithm}.image.{backbone}:decoder_prune_boundary": [{"boundary": "post_attention_pre_ffn" if mid else "inter_layer", "index": 3 if mid else 4}]}
    assert record.metadata["score_layers"] == {f"{algorithm}.image.{backbone}:decoder_prune_boundary": 1}
    assert record.metadata["canonical_evidence_applicable"] is False
    assert record.support_status == "declared"
    baseline = build_run_manifest(_config({**stage, "operation": {}}, backbone_name=backbone)).compression_stages[0]
    assert [{key: value for key, value in item.items() if key != "canonical_evidence_applicable"} for item in record.evidence] == baseline.evidence
    assert all(item["canonical_evidence_applicable"] is False for item in record.evidence)


@pytest.mark.parametrize("algorithm", ["fastv", "dart", "d2pruner"])
@pytest.mark.parametrize("backbone", ["llava_onevision", "qwen25_vl", "qwen3_vl"])
@pytest.mark.parametrize("boundary,changed", [(1, False), (2, True)])
def test_decoder_binding_override_has_the_same_position_projection_in_validation_and_manifest(backbone, boundary, changed, algorithm):
    source_field = "prune_layer" if algorithm == "fastv" else "prune_after_layer"
    from mllm_comp.compressors.multimodal_registry import builtin_multimodal_compression_contracts
    from mllm_comp.pipeline.v2_validation import validate_v2_stage_contract
    config = _config({
        "name": "fastv-override", "placement": "llm_level", "execution_phase": "generate",
        "plugin": algorithm, "algorithm": algorithm, "variant": f"{algorithm}.image", "target_modalities": ["image"],
        "conditioning_modalities": ["text"], "preserved_modalities": ["text", "video", "audio"], "media_alignment": "token",
        "bindings": [{"binding": f"{algorithm}.image.{backbone}", "config": {source_field: 1}}],
        "config": {source_field: 2}, "operation": {"layers": [boundary]},
    }, backbone_name=backbone)
    stage = config.pipeline.stages[0]
    projection = validate_v2_stage_contract(stage, registry=builtin_multimodal_compression_contracts(), backbone_name=backbone)
    binding = projection.bindings[0]
    assert binding.metadata["position_changed"] is changed
    assert binding.targets[0].metadata["score_layer"] == 0
    record = build_run_manifest(config).compression_stages[0]
    assert record.metadata["support_claim_is_exact"] is not changed
    assert record.metadata["score_layers"] == {f"{algorithm}.image.{backbone}:decoder_prune_boundary": 0}


def _config(
    stage: dict[str, object],
    *,
    backbone_name: str = "qwen3_vl",
) -> object:
    return experiment_config_from_dict(
        {
            "experiment": {"name": "manifest-v2", "run_id": "fixture"},
            "backbone": {"name": backbone_name, "model_id": "fixture"},
            "pipeline": {"stages": [stage]},
            "analysis": {},
            "train": {},
            "eval": {},
            "data": {},
            "output": {"run_dir": "runs/fixture", "manifest_path": "manifest.json"},
        }
    )


@pytest.mark.parametrize("backbone", ["llava_onevision", "qwen25_vl", "qwen3_vl"])
@pytest.mark.parametrize("initial", [False, True])
def test_manifest_tome_position_preserves_compound_steps_and_historical_refs(backbone, initial):
    prefix = "vision" if backbone == "llava_onevision" else "visual"
    site = f"{prefix}_block_output"
    record = build_run_manifest(_config({
        "name": "tome-position", "placement": "encoder_level", "execution_phase": "forward",
        "plugin": "tome", "algorithm": "tome", "variant": "tome.image.encoder_dense_boundary",
        "binding": f"tome.image.{backbone}", "target_modalities": ["image"],
        "operation": {"targets": {"encoder_merge": {"target_site": site,
            "layers": [0] if initial else [1], **({"layer_edge": "before"} if initial else {})}}},
    }, backbone_name=backbone)).compression_stages[0]
    assert record.target_sites[0] == site
    assert len(record.target_sites) == (4 if backbone == "qwen3_vl" else 3)
    assert record.metadata["application_positions"] == {f"tome.image.{backbone}:encoder_merge": [{"boundary": "inter_layer", "index": 0 if initial else 2}]}
    assert record.support_status == "declared"
    assert record.metadata["canonical_evidence_applicable"] is False
    assert record.evidence and all(item["canonical_evidence_applicable"] is False for item in record.evidence)


def test_manifest_keeps_legacy_projection_and_adds_v2_stage_identity() -> None:
    config = _config(
        {
            "name": "visionzip-image",
            "placement": "projector_level",
            "execution_phase": "generate",
            "plugin": "visionzip",
            "algorithm": "visionzip",
            "variant": "visionzip.image",
            "binding": "visionzip.image.qwen3_vl",
            "target_modalities": ["image"],
            "conditioning_modalities": ["text"],
            "preserved_modalities": ["text", "video"],
            "coupling_policy": "independent",
            "media_alignment": "spatial",
            "operation": {"target_site": "post_merger_pre_scatter"},
        }
    )

    manifest = build_run_manifest(config)

    assert manifest.schema_version == 2
    assert manifest.compressor_name == "visionzip"
    assert len(manifest.compression_stages) == 1
    record = manifest.compression_stages[0]
    assert record.algorithm == "visionzip"
    assert record.variant == "visionzip.image"
    assert record.bindings == ["visionzip.image.qwen3_vl"]
    assert record.target_modalities == ["image"]
    assert record.preserved_modalities == ["text", "video"]
    assert record.algorithm_version == "cvpr25-reference-v1"
    assert record.target_sites == [
        "post_final_merger_pre_llm_scatter",
        "post_deepstack_mergers_pre_deepstack_injection",
    ]
    assert record.support_status == "declared"
    assert record.metadata["support_unit_keys"] == [
        "visionzip:visionzip.image:qwen3_vl:image:generate"
    ]
    assert record.metadata["support_claim_is_exact"] is True


def test_manifest_projects_legacy_modalities_without_faking_v2_variant() -> None:
    config = _config(
        {
            "name": "legacy-audio",
            "placement": "projector_level",
            "execution_phase": "generate",
            "plugin": "segmentwise_topk_audio",
            "modalities": ["audio"],
        }
    )

    record = build_run_manifest(config).compression_stages[0]

    assert record.algorithm == "segmentwise_topk_audio"
    assert record.variant is None
    assert record.target_modalities == ["audio"]
    assert record.support_status == "active"
    assert record.metadata["legacy_descriptor"] is True
    assert record.metadata["capability_status"] == "active"


@pytest.mark.parametrize("changed", [False, True])
def test_manifest_resolves_audio_position_and_historical_evidence(changed: bool) -> None:
    site = (
        "llm_decoder_layer_post_attention_audio_tokens" if changed
        else "llm_decoder_layer_output_audio_tokens"
    )
    record = build_run_manifest(_config(
        {
            "name": "audio-pyramid",
            "placement": "llm_level",
            "execution_phase": "generate",
            "plugin": "audio_pyramid_attention_pruning",
            "algorithm": "audio_pyramid_attention_pruning",
            "variant": "audio_pyramid_attention_pruning.audio",
            "binding": "audio_pyramid_attention_pruning.audio.qwen25_omni",
            "target_modalities": ["audio"],
            "operation": {"target_site": site},
        },
        backbone_name="qwen25_omni",
    )).compression_stages[0]
    assert record.target_sites == [site]
    assert record.support_status == ("declared" if changed else "active")
    assert record.metadata["support_claim_is_exact"] is not changed
    assert record.evidence
    if changed:
        assert record.metadata["canonical_evidence_applicable"] is False
        assert all(item["canonical_evidence_applicable"] is False for item in record.evidence)


def test_manifest_marks_descriptor_backed_candidate_as_target() -> None:
    config = _config(
        {
            "name": "uniform-average-candidate",
            "placement": "projector_level",
            "execution_phase": "generate",
            "plugin": "uniform_average_pooling_audio",
            "modalities": ["audio"],
        },
        backbone_name="qwen25_omni",
    )

    record = build_run_manifest(config).compression_stages[0]

    assert record.support_status == "target"
    assert record.metadata["capability_status"] == "target"


def test_manifest_projects_runtime_layout_evidence_into_exact_stage() -> None:
    config = _config(
        {
            "name": "visionzip-image",
            "placement": "projector_level",
            "execution_phase": "generate",
            "plugin": "visionzip",
            "algorithm": "visionzip",
            "variant": "visionzip.image",
            "binding": "visionzip.image.qwen3_vl",
            "target_modalities": ["image"],
        }
    )
    runtime = {
        "layout_delta_summary": {
            "record_count": 1,
            "records": [
                {
                    "stage_name": "visionzip-image",
                    "binding": "visionzip.image.qwen3_vl",
                    "source_count": 64,
                    "result_count": 32,
                    "operation": "merge",
                    "mapping_sha256": "fixture-hash",
                }
            ],
        }
    }

    manifest = build_run_manifest(
        config,
        compression_runtime_metadata=runtime,
    )

    record = manifest.compression_stages[0]
    assert record.layout_delta_summary["record_count"] == 1
    assert record.layout_delta_summary["records"][0]["mapping_sha256"] == (
        "fixture-hash"
    )
    assert manifest.per_stage_stats["visionzip-image"] == (
        record.layout_delta_summary
    )


@pytest.mark.parametrize("edge, layer, index", [("before", 0, 0), ("before", 2, 2), ("after", 1, 2)])
def test_manifest_records_application_coordinates_and_keeps_evidence(edge, layer, index):
    stage = {
        "name": "application-boundary", "placement": "llm_level", "execution_phase": "generate",
        "plugin": "audio_pyramid_attention_pruning", "algorithm": "audio_pyramid_attention_pruning",
        "variant": "audio_pyramid_attention_pruning.audio",
        "binding": "audio_pyramid_attention_pruning.audio.qwen25_omni",
        "target_modalities": ["audio"],
        "operation": {"layers": [layer], "layer_edge": edge},
        "activation_scope": {"layers": [0]},
    }
    record = build_run_manifest(_config(stage, backbone_name="qwen25_omni")).compression_stages[0]
    assert list(record.metadata["application_positions"].values()) == [[{
        "boundary": "inter_layer", "index": index,
    }]]
    assert record.support_status == "declared"
    assert record.metadata["canonical_evidence_applicable"] is False
    assert record.evidence and all(item["canonical_evidence_applicable"] is False for item in record.evidence)


@pytest.mark.parametrize("operation", [
    {}, {"layers": [8, 17, 26]}, {"layers": [9, 18, 27], "layer_edge": "before"},
])
def test_equal_layer_boundary_expressions_keep_canonical_support(operation):
    config = _config({
        "name": "equal-boundaries", "placement": "llm_level", "execution_phase": "generate",
        "plugin": "audio_pyramid_attention_pruning", "algorithm": "audio_pyramid_attention_pruning",
        "variant": "audio_pyramid_attention_pruning.audio",
        "binding": "audio_pyramid_attention_pruning.audio.qwen25_omni",
        "target_modalities": ["audio"], "operation": operation,
    }, backbone_name="qwen25_omni")
    record = build_run_manifest(config).compression_stages[0]
    assert record.support_status == "active"
    assert record.metadata["support_claim_is_exact"] is True
    assert record.evidence and all("canonical_evidence_applicable" not in item for item in record.evidence)


@pytest.mark.parametrize("algorithm", [
    "adjacent_token_merging_audio", "encoder_attention_topk_audio", "progressive_attention_pruning_audio",
])
@pytest.mark.parametrize("initial", [False, True])
def test_encoder_application_position_evidence(algorithm, initial):
    full_site = "audio_encoder_layer_output_post_block"
    canonical_site = full_site if algorithm == "progressive_attention_pruning_audio" else "audio_encoder_layer_post_mhsa_pre_ffn"
    operation = {"layers": [0, 2]}
    if initial:
        operation.update({"target_site": full_site, "layer_edge": "before"})
    record = build_run_manifest(_config({
        "name": "encoder-boundaries", "placement": "encoder_level", "execution_phase": "generate",
        "plugin": algorithm, "algorithm": algorithm, "variant": f"{algorithm}.audio",
        "binding": f"{algorithm}.audio.qwen25_omni", "target_modalities": ["audio"],
        "activation_scope": {"layers": [0, 2]}, "operation": operation,
    }, backbone_name="qwen25_omni")).compression_stages[0]
    assert record.target_sites == [full_site if initial else canonical_site]
    assert record.support_status == ("declared" if initial else "active")
    assert record.metadata["support_claim_is_exact"] is not initial
    inter_layer = initial or canonical_site == full_site
    indices = [1, 3] if not initial and inter_layer else [0, 2]
    assert list(record.metadata["application_positions"].values()) == [[
        {"boundary": "inter_layer" if inter_layer else "post_attention_pre_ffn", "index": index}
        for index in indices
    ]]
    assert record.evidence
    assert all(item.get("canonical_evidence_applicable", True) is not initial for item in record.evidence)


def test_encoder_cadence_before_alias_keeps_canonical_support():
    algorithm = "progressive_attention_pruning_audio"
    record = build_run_manifest(_config({
        "name": "encoder-cadence", "placement": "encoder_level", "execution_phase": "generate",
        "plugin": algorithm, "algorithm": algorithm, "variant": f"{algorithm}.audio",
        "binding": f"{algorithm}.audio.qwen25_omni", "target_modalities": ["audio"],
        "operation": {"layers": [7, 8], "layer_edge": "before"},
    }, backbone_name="qwen25_omni")).compression_stages[0]
    assert record.support_status == "active"
    assert record.metadata["support_claim_is_exact"] is True


@pytest.mark.parametrize("algorithm,score_source", [
    ("uniform_sampling_audio", None), ("uniform_average_pooling_audio", None),
    ("local_temporal_bipartite_merging_audio", None), ("fastslm_hta_audio", None),
    ("segmentwise_topk_audio", "uniform"), ("segmentwise_topk_audio", "random"),
])
@pytest.mark.parametrize("placement,site", [
    ("llm_level", "llm_decoder_layer_post_attention_audio_tokens"),
    ("encoder_level", "audio_encoder_layer_post_mhsa_pre_ffn"),
])
def test_manifest_reports_temporal_rule_layer_position(algorithm, score_source, placement, site):
    record = build_run_manifest(_config({
        "name": "temporal-layer", "placement": placement, "execution_phase": "generate",
        "criterion": {"score_source": score_source} if score_source else {},
        "plugin": algorithm, "algorithm": algorithm, "variant": f"{algorithm}.audio",
        "binding": f"{algorithm}.audio.qwen25_omni", "target_modalities": ["audio"],
        "operation": {"target_site": site, "layers": [1, 2]},
    }, backbone_name="qwen25_omni")).compression_stages[0]
    assert record.target_sites == [site]
    assert list(record.metadata["application_positions"].values()) == [[
        {"boundary": "post_attention_pre_ffn", "index": i} for i in [1, 2]
    ]]
    assert record.metadata["canonical_evidence_applicable"] is False
    assert record.support_status == "declared"



def test_manifest_reports_segmentwise_attention_encoder_position():
    record = build_run_manifest(_config({
        "name": "segmentwise-encoder", "placement": "encoder_level", "execution_phase": "forward",
        "plugin": "segmentwise_topk_audio", "algorithm": "segmentwise_topk_audio", "variant": "segmentwise_topk_audio.audio",
        "binding": "segmentwise_topk_audio.audio.qwen25_omni", "target_modalities": ["audio"],
        "operation": {"type": "keep", "target_site": "audio_encoder_layer_output_post_block", "layers": [2, 3]},
    }, backbone_name="qwen25_omni")).compression_stages[0]
    assert record.target_sites == ["audio_encoder_layer_output_post_block"]
    assert list(record.metadata["application_positions"].values()) == [[
        {"boundary": "inter_layer", "index": i} for i in [3, 4]
    ]]
    assert record.metadata["canonical_evidence_applicable"] is False
    assert record.support_status == "declared"



def test_manifest_reports_segmentwise_external_attention_llm_position():
    record = build_run_manifest(_config({
        "name": "segmentwise-llm", "placement": "llm_level", "execution_phase": "forward",
        "plugin": "segmentwise_topk_audio", "algorithm": "segmentwise_topk_audio", "variant": "segmentwise_topk_audio.audio",
        "binding": "segmentwise_topk_audio.audio.qwen25_omni", "target_modalities": ["audio"],
        "operation": {"type": "keep", "target_site": "llm_decoder_layer_output_audio_tokens", "layers": [0], "layer_edge": "before"},
    }, backbone_name="qwen25_omni")).compression_stages[0]
    assert record.target_sites == ["llm_decoder_layer_output_audio_tokens"]
    assert list(record.metadata["application_positions"].values()) == [[{"boundary": "inter_layer", "index": 0}]]
    assert record.metadata["canonical_evidence_applicable"] is False
    assert record.support_status == "declared"



@pytest.mark.parametrize("family", ["encoder", "llm"])
@pytest.mark.parametrize("algorithm", ["unsupervised_segmentation_audio", "similarity_attention_merge_prune_audio"])
def test_manifest_reports_layer_exploration_position(family, algorithm):
    site = "audio_encoder_layer_post_mhsa_pre_ffn" if family == "encoder" else "llm_decoder_layer_post_attention_audio_tokens"
    record = build_run_manifest(_config({
        "name": f"{algorithm}-{family}", "placement": f"{family}_level", "execution_phase": "forward",
        "plugin": algorithm, "algorithm": algorithm,
        "variant": f"{algorithm}.audio", "binding": f"{algorithm}.audio.qwen25_omni",
        "target_modalities": ["audio"],
        "operation": {"type": "merge_prune" if algorithm == "similarity_attention_merge_prune_audio" else "merge", "target_site": site, "layers": [1]},
    }, backbone_name="qwen25_omni")).compression_stages[0]
    assert record.target_sites == [site]
    assert list(record.metadata["application_positions"].values()) == [[{"boundary": "post_attention_pre_ffn", "index": 1}]]
    assert record.metadata["canonical_evidence_applicable"] is False
    assert record.support_status == "declared"


@pytest.mark.parametrize("backbone", ["llava_onevision", "qwen25_vl", "qwen3_vl"])
def test_manifest_tome_fixed_source_preserves_historical_refs_without_claiming_them(backbone):
    record = build_run_manifest(_config({
        "name": "fixed-source", "placement": "encoder_level", "execution_phase": "forward",
        "plugin": "tome", "algorithm": "tome", "variant": "tome.image.encoder_dense_boundary",
        "binding": f"tome.image.{backbone}", "target_modalities": ["image"],
        "criterion": {"score_layer": 0},
    }, backbone_name=backbone)).compression_stages[0]
    assert record.metadata["score_layers"] == {f"tome.image.{backbone}:encoder_merge": 0}
    assert record.support_status == "declared"
    assert record.metadata["canonical_evidence_applicable"] is False
    assert record.evidence and all(item["canonical_evidence_applicable"] is False for item in record.evidence)


@pytest.mark.parametrize("algorithm", ["encoder_attention_topk_audio", "progressive_attention_pruning_audio", "adjacent_token_merging_audio"])
def test_manifest_encoder_rule_llm_position_preserves_history_scope(algorithm):
    record = build_run_manifest(_config({
        "name": "encoder-rule-llm", "placement": "llm_level", "execution_phase": "forward",
        "plugin": algorithm, "algorithm": algorithm, "variant": f"{algorithm}.audio",
        "binding": f"{algorithm}.audio.qwen25_omni", "target_modalities": ["audio"],
        "activation_scope": {"layers": [0, 2]},
        "operation": {"target_site": "llm_decoder_layer_output_audio_tokens", "layers": [0, 2], "layer_edge": "before"},
    }, backbone_name="qwen25_omni")).compression_stages[0]
    assert record.target_sites == ["llm_decoder_layer_output_audio_tokens"]
    assert record.metadata["canonical_evidence_applicable"] is False
    assert record.support_status == "declared"
    assert list(record.metadata["application_positions"].values()) == [[{"boundary": "inter_layer", "index": i} for i in [0, 2]]]
    assert record.evidence and all(item["canonical_evidence_applicable"] is False for item in record.evidence)


@pytest.mark.parametrize("algorithm", ["encoder_attention_topk_audio", "progressive_attention_pruning_audio", "adjacent_token_merging_audio"])
@pytest.mark.parametrize("site", ["audio_encoder_output_pre_projector", "audio_projector_output_post_projector"])
def test_manifest_encoder_rule_projector_position_preserves_history_scope(algorithm, site):
    record = build_run_manifest(_config({
        "name": "encoder-rule-projector", "placement": "projector_level", "execution_phase": "forward",
        "plugin": algorithm, "algorithm": algorithm, "variant": f"{algorithm}.audio",
        "binding": f"{algorithm}.audio.qwen25_omni", "target_modalities": ["audio"],
        "activation_scope": {"layers": [0, 2]}, "operation": {"type": "merge" if algorithm == "adjacent_token_merging_audio" else "keep", "target_site": site},
    }, backbone_name="qwen25_omni")).compression_stages[0]
    assert record.target_sites == [site]
    assert record.support_status == "declared"
    assert record.metadata["canonical_evidence_applicable"] is False
    assert record.evidence and all(item["canonical_evidence_applicable"] is False for item in record.evidence)


@pytest.mark.parametrize("backbone", ["qwen3_vl", "qwen25_vl", "llava_onevision"])
@pytest.mark.parametrize("mid", [False, True])
def test_manifest_sgl_keeps_fractional_recipe_and_records_application(backbone, mid):
    site = "llm_decoder_layer_post_attention_image_tokens" if mid else "llm_decoder_layer_output_image_tokens"
    stage = {"name": "sgl-position", "placement": "llm_level", "execution_phase": "generate",
        "plugin": "sgl", "algorithm": "sgl", "variant": "sgl.image", "binding": f"sgl.image.{backbone}",
        "target_modalities": ["image"], "config": {"guide_model_path": "fixture-guide", "prune_layer_fraction": 0.5},
        "operation": {"target_site": site, "layers": [0], **({} if mid else {"layer_edge": "before"})}}
    record = build_run_manifest(_config(stage, backbone_name=backbone)).compression_stages[0]
    assert record.target_sites == [site]
    assert record.metadata["application_positions"] == {f"sgl.image.{backbone}:decoder_guide_prune_boundary": [
        {"boundary": "post_attention_pre_ffn" if mid else "inter_layer", "index": 0}]}
    assert record.metadata["canonical_evidence_applicable"] is False
    baseline = build_run_manifest(_config({**stage, "operation": {}}, backbone_name=backbone)).compression_stages[0]
    assert [{key: value for key, value in item.items() if key != "canonical_evidence_applicable"}
        for item in record.evidence] == baseline.evidence


@pytest.mark.parametrize("backbone", ["qwen3_vl", "qwen25_vl", "llava_onevision"])
@pytest.mark.parametrize("phase", ["forward", "generate"])
@pytest.mark.parametrize("mid", [False, True])
def test_manifest_ivc_preserves_history_and_records_fixed_source_and_application(backbone, phase, mid):
    site = "llm_decoder_layer_post_attention_image_tokens" if mid else "llm_decoder_layer_output_image_tokens"
    target = "native_rope_coordinate_prune" if backbone == "llava_onevision" else "native_mrope_coordinate_prune"
    stage = {"name": "ivc-position", "placement": "llm_level", "execution_phase": phase,
        "plugin": "ivc_prune", "algorithm": "ivc_prune", "variant": "ivc_prune.image", "binding": f"ivc_prune.image.{backbone}",
        "target_modalities": ["image"], "config": {"prune_layer": 1},
        "operation": {"targets": {target: {"target_site": site, "layers": [2], **({} if mid else {"layer_edge": "before"})}}}}
    record = build_run_manifest(_config(stage, backbone_name=backbone)).compression_stages[0]
    assert record.target_sites == [site]
    assert record.metadata["application_positions"] == {f"ivc_prune.image.{backbone}:{target}": [
        {"boundary": "post_attention_pre_ffn" if mid else "inter_layer", "index": 2}]}
    assert record.metadata["score_layers"] == {f"ivc_prune.image.{backbone}:{target}": 0}
    assert record.metadata["canonical_evidence_applicable"] is False
    baseline = build_run_manifest(_config({**stage, "operation": {}}, backbone_name=backbone)).compression_stages[0]
    assert [{key: value for key, value in item.items() if key != "canonical_evidence_applicable"}
        for item in record.evidence] == baseline.evidence


@pytest.mark.parametrize("backbone", ["qwen3_vl", "qwen25_vl", "llava_onevision"])
@pytest.mark.parametrize("phase", ["forward", "generate"])
@pytest.mark.parametrize("mid", [False, True])
def test_manifest_feather_keeps_both_sources_and_actual_application_boundaries(backbone, phase, mid):
    site = "llm_decoder_layer_post_attention_image_tokens" if mid else "llm_decoder_layer_output_image_tokens"
    stage = {"name": "feather-position", "placement": "llm_level", "execution_phase": phase,
        "plugin": "feather", "algorithm": "feather", "variant": "feather.image", "binding": f"feather.image.{backbone}",
        "target_modalities": ["image"], "config": {"stage_layers": [1, 2]},
        "operation": {"target_site": site, "layers": [2, 3], **({} if mid else {"layer_edge": "before"})}}
    record = build_run_manifest(_config(stage, backbone_name=backbone)).compression_stages[0]
    assert record.target_sites == [site]
    assert record.metadata["application_positions"] == {f"feather.image.{backbone}:two_stage_no_rope_prune": [
        {"boundary": "post_attention_pre_ffn" if mid else "inter_layer", "index": i} for i in (2, 3)]}
    assert record.metadata["score_layers"] == {f"feather.image.{backbone}:two_stage_no_rope_prune": [0, 1]}
    assert record.metadata["canonical_evidence_applicable"] is False
    baseline = build_run_manifest(_config({**stage, "operation": {}}, backbone_name=backbone)).compression_stages[0]
    assert [{key: value for key, value in item.items() if key != "canonical_evidence_applicable"}
        for item in record.evidence] == baseline.evidence


@pytest.mark.parametrize("backbone", ["qwen3_vl", "qwen25_vl", "llava_onevision"])
@pytest.mark.parametrize("phase", ["forward", "generate"])
@pytest.mark.parametrize("mid", [False, True])
def test_manifest_nuwa_keeps_stage1_targets_history_and_actual_stage2_position(backbone, phase, mid):
    site = "llm_decoder_layer_post_attention_image_tokens" if mid else "llm_decoder_layer_output_image_tokens"
    target = "stage2_last_prompt_attention_selection"
    stage = {"name": "nuwa-position", "placement": "projector_level", "execution_phase": phase,
        "plugin": "nuwa", "algorithm": "nuwa", "variant": "nuwa.image", "binding": f"nuwa.image.{backbone}",
        "target_modalities": ["image"], "config": {"stage2_after_layer": 1},
        "operation": {"targets": {target: {"target_site": site, "layers": [2], **({} if mid else {"layer_edge": "before"})}}}}
    record = build_run_manifest(_config(stage, backbone_name=backbone)).compression_stages[0]
    baseline = build_run_manifest(_config({**stage, "operation": {}}, backbone_name=backbone)).compression_stages[0]
    assert record.target_sites == [*baseline.target_sites[:-1], site]
    assert record.metadata["application_positions"] == {f"nuwa.image.{backbone}:{target}": [
        {"boundary": "post_attention_pre_ffn" if mid else "inter_layer", "index": 2}]}
    assert record.metadata["score_layers"] == {f"nuwa.image.{backbone}:{target}": 0}
    assert record.metadata["canonical_evidence_applicable"] is False
    assert [{key: value for key, value in item.items() if key != "canonical_evidence_applicable"}
        for item in record.evidence] == baseline.evidence


@pytest.mark.parametrize("backbone", ["qwen3_vl", "qwen25_vl", "llava_onevision"])
@pytest.mark.parametrize("phase", ["forward", "generate"])
@pytest.mark.parametrize("mid", [False, True])
@pytest.mark.parametrize("algorithm", ["btp", "halfv", "pyramid_drop", "finepruner", "transprune", "pact", "sparsevlm"])
def test_manifest_multistage_pruning_keeps_history_and_positions(backbone, phase, mid, algorithm):
    site = "llm_decoder_layer_post_attention_image_tokens" if mid else "llm_decoder_layer_output_image_tokens"
    target = "decoder_balanced_pruning_boundaries"
    stage = {"name": "btp-position", "placement": "llm_level", "execution_phase": phase,
        "plugin": "btp", "algorithm": "btp", "variant": "btp.image", "binding": f"btp.image.{backbone}",
        "target_modalities": ["image"], "config": {"prune_before_layers": [2, 3, 4],
            "retain_ratios": [0.5, 0.5, 0.0], "attention_fractions": [0.4, 1.0, 0.0],
            "attention_oversample": [1.2, 1.2, 1.0], "rebalance_stages": [False, True, False]},
        "operation": {"targets": {target: {"target_site": site, "layers": [0, 1, 2] if mid else [2, 3, 4],
            **({} if mid else {"layer_edge": "before"})}}}}
    if algorithm == "halfv":
        target = "decoder_ivr_ssr_boundaries"
        stage.update(name="halfv-position", plugin=algorithm, algorithm=algorithm, variant=f"{algorithm}.image",
            binding=f"{algorithm}.image.{backbone}", config={"ivr_layer_index": 1, "ssr_layer_index": 2},
            operation={"targets": {target: {"target_site": site, "layers": [2, 3], **({} if mid else {"layer_edge": "before"})}}})
    elif algorithm == "pyramid_drop":
        target = "decoder_stage_boundary"
        stage.update(name="pyramid-position", plugin=algorithm, algorithm=algorithm, variant=f"{algorithm}.image",
            binding=f"{algorithm}.image.{backbone}", config={"num_stages": 3, "layer_indices": [0, 1]},
            operation={"targets": {target: {"target_site": site, "layers": [2, 3], **({} if mid else {"layer_edge": "before"})}}})
    elif algorithm == "finepruner":
        target = "decoder_iac_arp_boundaries"
        stage.update(name="finepruner-position", plugin=algorithm, algorithm=algorithm, variant=f"{algorithm}.image",
            binding=f"{algorithm}.image.{backbone}", config={"iac_after_layers": [1, 2], "iac_hash_bits": [2, 1],
                "arp_center_after_layer": 3, "arp_layer_offsets": [0, 1]},
            operation={"targets": {target: {"target_site": site, "layers": [0, 1, 3], **({} if mid else {"layer_edge": "before"})}}})
    elif algorithm == "transprune":
        target = "decoder_ttv_iga_boundaries"
        stage.update(name="transprune-position", plugin=algorithm, algorithm=algorithm, variant=f"{algorithm}.image",
            binding=f"{algorithm}.image.{backbone}", config={"accumulating_layers": [1, 2], "prune_after_layers": [1, 2],
                "retain_ratios": [0.75, 0.25]},
            operation={"targets": {target: {"target_site": site, "layers": [2, 3], **({} if mid else {"layer_edge": "before"})}}})
    elif algorithm == "pact":
        target = "decoder_pact_boundary"
        stage.update(name="pact-position", plugin=algorithm, algorithm=algorithm, variant=f"{algorithm}.image",
            binding=f"{algorithm}.image.{backbone}", config={"reduction_layer": 1},
            operation={"targets": {target: {"target_site": site, "layers": [2], **({} if mid else {"layer_edge": "before"})}}})
    elif algorithm == "sparsevlm":
        target = "decoder_merge_boundaries"
        stage.update(name="sparsevlm-position", plugin=algorithm, algorithm=algorithm, variant=f"{algorithm}.image",
            binding=f"{algorithm}.image.{backbone}", config={"prune_layers": [0, 1], "schedule_keep_tokens": [2, 1]},
            operation={"targets": {target: {"target_site": site, "layers": [2, 3], **({} if mid else {"layer_edge": "before"})}}})
    indices = stage["operation"]["targets"][target]["layers"]
    record = build_run_manifest(_config(stage, backbone_name=backbone)).compression_stages[0]
    baseline = build_run_manifest(_config({**stage, "operation": {}}, backbone_name=backbone)).compression_stages[0]
    assert record.target_sites == [site]
    assert record.metadata["application_positions"] == {f"{algorithm}.image.{backbone}:{target}": [
        {"boundary": "post_attention_pre_ffn" if mid else "inter_layer", "index": index}
        for index in indices]}
    expected_sources = {} if algorithm == "pyramid_drop" and backbone == "llava_onevision" else {f"{algorithm}.image.{backbone}:{target}": [2, 3] if algorithm == "finepruner" else [0] if algorithm == "pact" else [0, 1]}
    assert record.metadata.get("score_layers", {}) == expected_sources
    assert record.metadata["canonical_evidence_applicable"] is False
    assert [{key: value for key, value in item.items() if key != "canonical_evidence_applicable"}
        for item in record.evidence] == baseline.evidence


@pytest.mark.parametrize("backbone", ["qwen3_vl", "qwen25_vl"])
@pytest.mark.parametrize("phase", ["forward", "generate"])
@pytest.mark.parametrize("edge,layers", [("before", [1, 2]), ("after", [0, 1])])
def test_manifest_pyramid_explicit_source_equivalence_keeps_history_applicable(backbone, phase, edge, layers):
    stage = {"name": "pyramid-equivalent", "placement": "llm_level", "execution_phase": phase,
        "plugin": "pyramid_drop", "algorithm": "pyramid_drop", "variant": "pyramid_drop.image",
        "binding": f"pyramid_drop.image.{backbone}", "target_modalities": ["image"],
        "config": {"num_stages": 3, "layer_indices": [0, 1]},
        "operation": {"layers": layers, "layer_edge": edge}}
    record = build_run_manifest(_config(stage, backbone_name=backbone)).compression_stages[0]
    baseline = build_run_manifest(_config({**stage, "operation": {}}, backbone_name=backbone)).compression_stages[0]
    assert record.metadata.get("canonical_evidence_applicable") == baseline.metadata.get("canonical_evidence_applicable")
    assert record.evidence == baseline.evidence


@pytest.mark.parametrize("backbone", ["qwen3_vl", "qwen25_vl", "llava_onevision"])
@pytest.mark.parametrize("algorithm", ["visionzip", "prumerge", "gap", "divprune", "aim", "vispruner", "mmtok", "zoo_prune"])
def test_manifest_projector_sites_preserve_default_and_historical_evidence(backbone, algorithm):
    stage = {"name": "projector-position", "placement": "projector_level", "execution_phase": "forward",
        "plugin": algorithm, "algorithm": algorithm, "variant": f"{algorithm}.image",
        "binding": f"{algorithm}.image.{backbone}", "target_modalities": ["image"]}
    baseline = build_run_manifest(_config(stage, backbone_name=backbone)).compression_stages[0]
    alias = build_run_manifest(_config({**stage, "operation": {"target_site": "projector_output"}}, backbone_name=backbone)).compression_stages[0]
    assert alias.target_sites == baseline.target_sites
    assert alias.evidence == baseline.evidence
    assert alias.support_status == baseline.support_status
    assert alias.metadata.get("canonical_evidence_applicable") == baseline.metadata.get("canonical_evidence_applicable")
    moved = build_run_manifest(_config({**stage, "operation": {"target_site": "projector_input"}}, backbone_name=backbone)).compression_stages[0]
    assert len(moved.target_sites) == len(baseline.target_sites)
    assert all(site.startswith("pre_") for site in moved.target_sites)
    assert moved.metadata["canonical_evidence_applicable"] is False
    assert [{key: value for key, value in item.items() if key != "canonical_evidence_applicable"}
        for item in moved.evidence] == baseline.evidence


@pytest.mark.parametrize("backbone", ["qwen3_vl", "qwen25_vl", "llava_onevision"])
def test_manifest_saint_original_stage_and_key_module_positions(backbone):
    encoder = backbone == "qwen25_vl"
    stage = {"name": "saint-position", "placement": "encoder_level" if encoder else "projector_level",
        "execution_phase": "forward", "plugin": "saint", "algorithm": "saint",
        "variant": "saint.image.encoder_progressive_prune" if encoder else "saint.image.packed_feature_prune",
        "binding": f"saint.image.{backbone}", "target_modalities": ["image"],
        "config": {"layer_indices": [0, 2], "keep_ratios": [0.5, 0.5]} if encoder else {}}
    baseline = build_run_manifest(_config(stage, backbone_name=backbone)).compression_stages[0]
    alias = {"layers": [0, 2]} if encoder else {"target_site": "projector_output"}
    same = build_run_manifest(_config({**stage, "operation": alias}, backbone_name=backbone)).compression_stages[0]
    assert same.evidence == baseline.evidence
    assert same.metadata.get("canonical_evidence_applicable") == baseline.metadata.get("canonical_evidence_applicable")
    operation = {"target_site": "visual_block_post_attention_residual_pre_mlp", "layers": [0, 3]} if encoder else {"target_site": "projector_input"}
    moved = build_run_manifest(_config({**stage, "operation": operation}, backbone_name=backbone)).compression_stages[0]
    assert moved.metadata["canonical_evidence_applicable"] is False
    assert [{key: value for key, value in item.items() if key != "canonical_evidence_applicable"}
        for item in moved.evidence] == baseline.evidence
    if encoder:
        key = "saint.image.qwen25_vl:encoder_progressive_prune"
        assert moved.metadata["application_positions"][key] == [{"boundary": "post_attention_pre_ffn", "index": i} for i in (0, 3)]
        assert moved.metadata["key_projection_layers"][key] == [0, 2]
        assert "score_layers" not in moved.metadata
    else:
        assert all(site.startswith("pre_") for site in moved.target_sites)


@pytest.mark.parametrize("backbone", ["qwen3_vl", "qwen25_vl", "llava_onevision"])
@pytest.mark.parametrize("algorithm", ["nuwa", "restore"])
def test_manifest_compound_projector_and_later_position_evidence(backbone, algorithm):
    stage = {"name": "compound-position", "placement": "projector_level", "execution_phase": "forward",
        "plugin": algorithm, "algorithm": algorithm, "variant": f"{algorithm}.image",
        "binding": f"{algorithm}.image.{backbone}", "target_modalities": ["image"],
        "config": {"stage2_after_layer": 1} if algorithm == "nuwa" else {}}
    baseline = build_run_manifest(_config(stage, backbone_name=backbone)).compression_stages[0]
    alias = {"target_site": baseline.target_sites[0]}
    same = build_run_manifest(_config({**stage, "operation": alias}, backbone_name=backbone)).compression_stages[0]
    assert same.evidence == baseline.evidence and same.support_status == baseline.support_status
    operation = {"target_site": "projector_input"}
    if algorithm == "nuwa":
        operation = {"targets": {
            "stage1_regional_visual_aggregation": {"target_site": "projector_input"},
            "stage2_last_prompt_attention_selection": {"target_site": "llm_decoder_layer_post_attention_image_tokens", "layers": [2]},
        }}
    moved = build_run_manifest(_config({**stage, "operation": operation}, backbone_name=backbone)).compression_stages[0]
    assert len(moved.target_sites) == len(baseline.target_sites)
    assert moved.target_sites[0].startswith("pre_")
    assert moved.metadata["canonical_evidence_applicable"] is False
    assert [{key: value for key, value in item.items() if key != "canonical_evidence_applicable"}
        for item in moved.evidence] == baseline.evidence
    if algorithm == "nuwa":
        assert moved.metadata["application_positions"][f"nuwa.image.{backbone}:stage2_last_prompt_attention_selection"] == [{"boundary": "post_attention_pre_ffn", "index": 2}]
    else:
        assert moved.target_sites[-1] == baseline.target_sites[-1]


@pytest.mark.parametrize("backbone", ["qwen3_vl", "qwen25_vl", "llava_onevision"])
@pytest.mark.parametrize("algorithm,target", [("vqtoken", "fixed_k_cluster"), ("visionzip", "dominant_contextual_reduce"), ("fastvid", "dyseg_stprune_dtm")])
def test_manifest_video_projector_position_preserves_history(backbone, algorithm, target):
    from mllm_comp.compressors.video.positions import video_projector_site
    stage = {"name": "video-projector-position", "placement": "projector_level", "execution_phase": "forward",
        "plugin": algorithm, "algorithm": algorithm, "variant": f"{algorithm}.video",
        "binding": f"{algorithm}.video.{backbone}", "target_modalities": ["video"]}
    baseline = build_run_manifest(_config(stage, backbone_name=backbone)).compression_stages[0]
    alias = build_run_manifest(_config({**stage, "operation": {"target_site": "projector_output"}}, backbone_name=backbone)).compression_stages[0]
    assert alias.target_sites == baseline.target_sites
    assert alias.evidence == baseline.evidence and alias.support_status == baseline.support_status
    moved = build_run_manifest(_config({**stage, "operation": {"targets": {target: {"target_site": "projector_input"}}}}, backbone_name=backbone)).compression_stages[0]
    assert moved.target_sites == [video_projector_site(backbone, "before")]
    assert moved.metadata["canonical_evidence_applicable"] is False
    assert moved.support_status == "declared"
    assert [{key: value for key, value in item.items() if key != "canonical_evidence_applicable"}
        for item in moved.evidence] == baseline.evidence


@pytest.mark.parametrize("backbone", ["qwen3_vl", "qwen25_vl", "llava_onevision"])
@pytest.mark.parametrize("algorithm,projector,other,defaults,layers", [
    ("dytok", "dynamic_visionzip_reduce", "detection_budget_provider", list(range(16, 24)), [0, 2]),
    ("dycoke", "ttm_visual_merge", "decode_kv_view_prune", [3], [1]),
])
@pytest.mark.parametrize("phase", ["forward", "generate"])
def test_manifest_video_compound_positions_preserve_history(backbone, algorithm, projector, other, defaults, layers, phase):
    from mllm_comp.compressors.video.positions import video_projector_site

    stage = {"name": "video-compound-position", "placement": "projector_level", "execution_phase": phase,
        "plugin": algorithm, "algorithm": algorithm, "variant": f"{algorithm}.video",
        "binding": f"{algorithm}.video.{backbone}", "target_modalities": ["video"]}
    baseline = build_run_manifest(_config(stage, backbone_name=backbone)).compression_stages[0]
    alias = build_run_manifest(_config({**stage, "operation": {"targets": {
        projector: {"target_site": "projector_output"}, other: {"layers": defaults},
    }}}, backbone_name=backbone)).compression_stages[0]
    assert alias.target_sites == baseline.target_sites
    assert alias.evidence == baseline.evidence and alias.support_status == baseline.support_status
    # A change to either step is exploratory, even when the Projector stays at its default.
    for move_projector in (False, True):
        moved = build_run_manifest(_config({**stage, "operation": {"targets": {
            projector: {"target_site": "projector_input" if move_projector else "projector_output"},
            other: {"layers": layers},
        }}}, backbone_name=backbone)).compression_stages[0]
        assert len(moved.target_sites) == len(baseline.target_sites) == 2
        assert video_projector_site(backbone, "before" if move_projector else "after") in moved.target_sites
        key = f"{algorithm}.video.{backbone}:{other}"
        assert moved.metadata["score_layers"][key] == (layers if algorithm == "dytok" else 0)
        if algorithm == "dycoke":
            assert moved.metadata["application_positions"][key] == [{"boundary": "decode_kv_before_attention", "index": 1}]
        assert moved.metadata["canonical_evidence_applicable"] is False
        assert moved.support_status == "declared"
        assert [{key: value for key, value in item.items() if key != "canonical_evidence_applicable"}
            for item in moved.evidence] == baseline.evidence
