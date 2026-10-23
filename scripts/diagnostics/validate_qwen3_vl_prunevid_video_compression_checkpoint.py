"""Validate canonical PruneVid video inference on Qwen3-VL.

The offline batch-one diagnostic reuses the shared Qwen3 real-checkpoint
harness and the fixed four ``nextqa_mc_test`` documents.  It requires both
canonical PruneVid stages: deterministic temporal/static-dynamic projector
clustering and the layer-10 VTP/cache-compaction boundary.  Main visual and
all DeepStack projector streams must replay the same source groups before VTP;
the post-VTP sequence must preserve native M-RoPE anchors and exact
``visual_pos_masks`` while physically compacting DynamicCache layers 0--10.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Mapping, Sequence

from mllm_comp.compressors.video.config import VIDEO_REFERENCE_REVISION
from mllm_comp.pipeline.stage import CompressionStageConfig
from scripts.diagnostics.video_method_checks import strict_timing_matches as _timing_matches

if __package__:
    from scripts.diagnostics import (
        validate_qwen25_vl_prunevid_video_compression_checkpoint as _qwen25_prunevid,
    )
    from scripts.diagnostics import (
        validate_qwen3_vl_video_compression_checkpoint as _qwen3,
    )
else:
    import validate_qwen25_vl_prunevid_video_compression_checkpoint as _qwen25_prunevid
    import validate_qwen3_vl_video_compression_checkpoint as _qwen3


FIXED_NEXTQA_DOC_IDS = _qwen3.FIXED_NEXTQA_DOC_IDS
CANONICAL_PRUNEVID_CONFIG = _qwen25_prunevid.CANONICAL_PRUNEVID_CONFIG
QWEN3_PRUNEVID_VIDEO_FPS = _qwen3.QWEN3_VIDEO_FPS
QWEN3_PRUNEVID_VIDEO_MAX_PIXELS = _qwen3.QWEN3_VIDEO_MAX_PIXELS


_conversation_for_args = _qwen3._conversation_for_args


def _conversation(
    *,
    prompt: str,
    video: Path,
    fps: float = QWEN3_PRUNEVID_VIDEO_FPS,
    max_pixels: int = QWEN3_PRUNEVID_VIDEO_MAX_PIXELS,
) -> list[dict[str, Any]]:
    return _qwen3._conversation(
        prompt=prompt,
        video=video,
        fps=fps,
        max_pixels=max_pixels,
    )


def _stage() -> CompressionStageConfig:
    return CompressionStageConfig.from_dict(
        {
            "name": "prunevid-video-qwen3-vl-real-checkpoint",
            "placement": "projector_level",
            "execution_phase": "generate",
            "plugin": "prunevid",
            "algorithm": "prunevid",
            "variant": "prunevid.video",
            "target_modalities": ["video"],
            "conditioning_modalities": ["text"],
            "preserved_modalities": ["text", "image", "audio"],
            "coupling_policy": "independent",
            "media_alignment": "spatiotemporal",
            "binding": "prunevid.video.qwen3_vl",
            "config": {},
        }
    )


def _preprocessing_record(args: argparse.Namespace) -> Mapping[str, Any]:
    return {
        "fps": float(args.video_fps),
        "max_pixels": int(args.video_max_pixels),
        "image_patch_size": 16,
        "enable_thinking": False,
        "provider": "qwen_vl_utils.process_vision_info",
        "constraint": (
            "uniform post-merger temporal rasters satisfy canonical PruneVid "
            "DPC-KNN and minimum clustering cardinality"
        ),
    }


def _validate_prunevid_sample(
    *,
    baseline: Mapping[str, Any],
    compressed: Mapping[str, Any],
    activation_records: Sequence[Mapping[str, Any]],
    algorithm_records: Sequence[Mapping[str, Any]],
    vtp_records: Sequence[Mapping[str, Any]],
    cache_records: Sequence[Mapping[str, Any]],
    latest_splice: Mapping[str, Any],
    latest_generation: Mapping[str, Any],
    latest_vtp: Mapping[str, Any] | None,
    source_grid_thw: Sequence[int],
    source_grid_token_count: int,
    spatial_merge_size: int,
    second_per_grid_seconds: float | None,
    labels_absent: bool,
    source_placeholder_spans: Sequence[Sequence[int]],
    native_video_count: int,
    expected_deepstack_branches: int,
    **_unused: Any,
) -> dict[str, bool]:
    if latest_vtp is None:
        raise RuntimeError("PruneVid Qwen3-VL bridge did not publish VTP evidence")
    if not (
        len(activation_records)
        == len(algorithm_records)
        == len(vtp_records)
        == len(cache_records)
        == 1
    ):
        raise RuntimeError(
            "PruneVid must publish one projector, algorithm, VTP, and cache "
            "record per Qwen3-VL sample"
        )
    if len(source_grid_thw) != 3:
        raise RuntimeError(f"Invalid Qwen3-VL source grid: {source_grid_thw!r}")
    temporal, height, width = (int(item) for item in source_grid_thw)
    if (
        spatial_merge_size < 1
        or height % spatial_merge_size
        or width % spatial_merge_size
    ):
        raise RuntimeError("Qwen3-VL source grid is incompatible with its merger")
    grid_height = height // spatial_merge_size
    grid_width = width // spatial_merge_size
    tokens_per_frame = grid_height * grid_width
    expected_source_count = temporal * tokens_per_frame

    projector = activation_records[0]
    algorithm = algorithm_records[0]
    vtp = vtp_records[0]
    cache = cache_records[0]
    sequence_delta = vtp["sequence_layout_delta"]
    splice_layout = latest_splice["layout_delta"]
    source_count = int(projector["source_count"])
    projector_count = int(projector["result_count"])
    final_count = int(vtp["result_count"])
    baseline_first = int(baseline["decoder"]["prefill"]["query_length"])
    compressed_first = int(compressed["decoder"]["prefill"]["query_length"])
    baseline_after = int(
        baseline["decoder_after_vtp_boundary"]["prefill"]["query_length"]
    )
    compressed_after = int(
        compressed["decoder_after_vtp_boundary"]["prefill"]["query_length"]
    )
    sequence_source = int(sequence_delta["source_count"])
    sequence_result = int(sequence_delta["result_count"])
    representatives = tuple(int(item) for item in algorithm["representative_indices"])
    kept_visual = tuple(int(item) for item in vtp["kept_visual_indices"])
    dropped_visual = tuple(int(item) for item in vtp["dropped_visual_indices"])
    source_mrope = tuple(
        tuple(int(value) for value in row)
        for row in latest_splice["source_visual_mrope"]
    )
    projector_mrope = tuple(
        tuple(int(value) for value in row)
        for row in latest_splice["result_visual_mrope"]
    )
    vtp_source_mrope = tuple(
        tuple(int(value) for value in row) for row in latest_vtp["source_visual_mrope"]
    )
    final_mrope = tuple(
        tuple(int(value) for value in row) for row in latest_vtp["result_visual_mrope"]
    )
    expected_projector_mrope = tuple(source_mrope[index] for index in representatives)
    expected_final_mrope = tuple(projector_mrope[index] for index in kept_visual)
    retained_positions = tuple(
        int(item) for item in latest_splice["retained_placeholder_positions"]
    )
    projector_removed = tuple(
        int(item) for item in latest_splice["removed_placeholder_positions"]
    )
    expected_vtp_removed = tuple(retained_positions[index] for index in dropped_visual)
    vtp_removed = tuple(
        int(item) for item in latest_vtp["dropped_source_placeholder_positions"]
    )
    combined_removed = tuple(sorted((*projector_removed, *vtp_removed)))
    source_visual_indices = tuple(
        int(item) for item in latest_vtp["source_visual_sequence_indices"]
    )
    result_visual_indices = tuple(
        int(item) for item in latest_vtp["result_visual_sequence_indices"]
    )
    source_mask_indices = tuple(
        int(item) for item in latest_vtp["source_visual_pos_mask_indices"]
    )
    result_mask_indices = tuple(
        int(item) for item in latest_vtp["result_visual_pos_mask_indices"]
    )
    deepstack_source_shapes = tuple(
        tuple(int(item) for item in shape)
        for shape in algorithm["deepstack_source_shapes"]
    )
    deepstack_result_shapes = tuple(
        tuple(int(item) for item in shape)
        for shape in algorithm["deepstack_result_shapes"]
    )
    splice_deepstack_shapes = tuple(
        tuple(int(item) for item in shape)
        for shape in latest_splice["deepstack_result_shapes"]
    )
    vtp_deepstack_shapes = tuple(
        tuple(int(item) for item in shape)
        for shape in latest_vtp["deepstack_projector_result_shapes"]
    )
    rope_values = tuple(int(item) for item in latest_vtp["rope_deltas"])
    expected_rope_delta = (
        int(latest_vtp["max_active_rotary_position"])
        + 1
        - int(latest_vtp["active_position_count"])
    )
    combined_removed_count = source_count - final_count
    generated_token_count = int(compressed["generated_token_count"])
    decode_query_lengths = tuple(
        int(item) for item in compressed["decoder"]["decode_query_lengths"]
    )
    post_vtp_decode_query_lengths = tuple(
        int(item)
        for item in compressed["decoder_after_vtp_boundary"]["decode_query_lengths"]
    )
    native_spans = _qwen3._spans(source_placeholder_spans)
    recorded_spans = _qwen3._spans(latest_splice["source_placeholder_spans"])

    checks = {
        "compression_applied": bool(latest_splice["compression_applied"]),
        "projector_non_identity_token_reduction": source_count > projector_count,
        "vtp_non_identity_token_reduction": projector_count > final_count > 0,
        "one_native_video": native_video_count == 1,
        "native_temporal_spans_preserved": (
            recorded_spans == native_spans
            and len(recorded_spans) == temporal
            and all(end - start == tokens_per_frame for start, end in recorded_spans)
            and _qwen3._span_count(recorded_spans) == source_count
        ),
        "grid_and_source_counts_agree": (
            source_count == expected_source_count == source_grid_token_count
            and int(latest_splice["source_placeholder_count"]) == source_count
            and int(algorithm["frame_count"]) == temporal
            and int(algorithm["grid_height"]) == grid_height
            and int(algorithm["grid_width"]) == grid_width
            and int(algorithm["tokens_per_frame"]) == tokens_per_frame
            and tokens_per_frame >= CANONICAL_PRUNEVID_CONFIG["min_tokens_for_cluster"]
        ),
        "projector_layout_records_agree": (
            int(splice_layout["source_count"]) == source_count
            and int(splice_layout["result_count"]) == projector_count
            and int(latest_splice["result_placeholder_count"]) == projector_count
            and int(algorithm["source_count"]) == source_count
            and int(algorithm["result_count"]) == projector_count
        ),
        "projector_decoder_length_reduction_exact": (
            baseline_first - compressed_first == source_count - projector_count
        ),
        "baseline_layer_lengths_stable": baseline_first == baseline_after,
        "vtp_boundary_lengths_exact": (
            compressed_first == sequence_source
            and compressed_after == sequence_result
            and sequence_source - sequence_result == projector_count - final_count
        ),
        "combined_decoder_length_reduction_exact": (
            baseline_after - compressed_after == combined_removed_count
        ),
        "source_video_grid_preserved": (
            bool(latest_splice["source_video_grid_preserved"])
            and tuple(int(item) for item in latest_splice["source_video_grid_thw"])
            == (temporal, height, width)
        ),
        "source_video_timing_preserved": _timing_matches(
            second_per_grid_seconds,
            latest_splice.get("second_per_grid_seconds"),
        ),
        "sparse_grid_policy": (
            latest_splice["result_grid_policy"] == "irregular_sparse_source_anchors"
        ),
        "inference_labels_absent": labels_absent,
        "canonical_config_exact": (
            dict(algorithm["canonical_config"]) == CANONICAL_PRUNEVID_CONFIG
        ),
        "projected_signals_exact": (
            tuple(algorithm["projected_features_shape"])[0] == source_count
            and tuple(algorithm["result_features_shape"])[0] == projector_count
            and bool(algorithm["projected_features_finite"])
            and bool(algorithm["result_features_finite"])
        ),
        "projector_mrope_sparse_anchor_gather_exact": (
            len(source_mrope) == source_count
            and len(projector_mrope) == projector_count
            and all(len(row) == 3 for row in (*source_mrope, *projector_mrope))
            and projector_mrope == expected_projector_mrope
        ),
        "projector_deepstack_layout_exact": (
            int(algorithm["deepstack_branch_count"])
            == expected_deepstack_branches
            == len(deepstack_source_shapes)
            == len(deepstack_result_shapes)
            == len(splice_deepstack_shapes)
            and all(shape[0] == source_count for shape in deepstack_source_shapes)
            and all(shape[0] == projector_count for shape in deepstack_result_shapes)
            and deepstack_result_shapes == splice_deepstack_shapes
            and bool(latest_splice["shared_main_deepstack_layout"])
        ),
        "vtp_config_and_layout_exact": (
            int(vtp["selected_layer"])
            == CANONICAL_PRUNEVID_CONFIG["vtp_selected_layer"]
            and float(vtp["alpha"]) == CANONICAL_PRUNEVID_CONFIG["vtp_alpha"]
            and int(vtp["source_count"]) == projector_count
            and len(kept_visual) == final_count
            and len(dropped_visual) == projector_count - final_count
            and tuple(sorted((*kept_visual, *dropped_visual)))
            == tuple(range(projector_count))
        ),
        "vtp_sequence_delta_visual_only": (
            int(sequence_delta["dropped_count"]) == projector_count - final_count
            and sequence_source - sequence_result == projector_count - final_count
        ),
        "layer10_cache_compaction_exact": (
            dict(vtp["cache"]) == dict(cache)
            and _qwen25_prunevid._cache_layers_exact(
                cache,
                source_length=sequence_source,
                result_length=sequence_result,
            )
        ),
        "vtp_mrope_sparse_anchor_gather_exact": (
            vtp_source_mrope == projector_mrope
            and final_mrope == expected_final_mrope
            and len(final_mrope) == final_count
            and all(len(row) == 3 for row in final_mrope)
            and tuple(int(item) for item in latest_vtp["kept_visual_indices"])
            == kept_visual
            and len(source_visual_indices) == projector_count
            and len(result_visual_indices) == final_count
            and result_visual_indices == tuple(sorted(set(result_visual_indices)))
        ),
        "vtp_visual_pos_masks_exact": (
            bool(latest_vtp["visual_pos_masks_compacted"])
            and bool(latest_vtp["visual_pos_masks_compacted_exact"])
            and source_mask_indices == source_visual_indices
            and result_mask_indices == result_visual_indices
        ),
        "deepstack_completed_before_vtp": (
            bool(latest_vtp["deepstack_completed_before_vtp"])
            and int(latest_vtp["deepstack_branch_count"])
            == int(latest_vtp["deepstack_consumed_branch_count"])
            == expected_deepstack_branches
            and vtp_deepstack_shapes == deepstack_result_shapes
        ),
        "vtp_removed_placeholder_lineage_exact": (
            len(retained_positions) == projector_count
            and vtp_removed == expected_vtp_removed
            and len(projector_removed) == source_count - projector_count
            and len(combined_removed) == combined_removed_count
            and len(set(combined_removed)) == combined_removed_count
        ),
        "bridge_vtp_layout_exact": (
            int(latest_vtp["selected_layer"])
            == CANONICAL_PRUNEVID_CONFIG["vtp_selected_layer"]
            and int(latest_vtp["source_visual_count"]) == projector_count
            and int(latest_vtp["result_visual_count"]) == final_count
            and int(latest_vtp["source_sequence_length"]) == sequence_source
            and int(latest_vtp["result_sequence_length"]) == sequence_result
            and latest_vtp["position_policy"] == "gather_native_mrope"
        ),
        "post_vtp_rope_delta_exact": (
            bool(latest_vtp["rope_delta_exact"])
            and int(latest_vtp["active_position_count"]) == sequence_result
            and int(latest_vtp["expected_rope_delta"]) == expected_rope_delta
            and rope_values == (expected_rope_delta,)
            and tuple(int(item) for item in latest_splice["rope_deltas"]) == rope_values
        ),
        "generation_prefill_complete": bool(latest_generation["prefill_complete"]),
        "generation_lengths_and_removal_exact": (
            int(latest_generation["source_sequence_length"]) == baseline_first
            and int(latest_generation["result_sequence_length"]) == sequence_result
            and int(latest_generation["removed_count"]) == combined_removed_count
        ),
        "baseline_logits_finite": bool(baseline["logits_finite"]),
        "compressed_logits_finite": bool(compressed["logits_finite"]),
        "bounded_cached_generation_exact": (
            int(baseline["generated_token_count"]) == 4
            and generated_token_count == 4
            and tuple(baseline["decoder"]["decode_query_lengths"]) == (1, 1, 1)
            and tuple(baseline["decoder_after_vtp_boundary"]["decode_query_lengths"])
            == (1, 1, 1)
            and decode_query_lengths == (1, 1, 1)
            and post_vtp_decode_query_lengths == (1, 1, 1)
            and int(latest_generation["decode_repairs"]) == 3
            and _qwen25_prunevid._decode_record_coherent(
                latest_generation.get("last_decode_record"),
                removed_count=combined_removed_count,
                result_sequence_length=sequence_result,
            )
        ),
        "baseline_decoder_alignment": all(baseline["decoder"]["invariants"].values())
        and all(baseline["decoder_after_vtp_boundary"]["invariants"].values()),
        "compressed_decoder_alignment": all(
            compressed["decoder"]["invariants"].values()
        )
        and all(compressed["decoder_after_vtp_boundary"]["invariants"].values()),
    }
    if not all(checks.values()):
        raise RuntimeError(f"PruneVid Qwen3-VL sample evidence failed: {checks!r}")
    return checks


def _profile() -> _qwen3.Qwen3VideoDiagnosticProfile:
    return _qwen3.Qwen3VideoDiagnosticProfile(
        algorithm="prunevid",
        variant="prunevid.video",
        binding="prunevid.video.qwen3_vl",
        canonical_config=CANONICAL_PRUNEVID_CONFIG,
        stage_factory=_stage,
        conversation_factory=_conversation_for_args,
        sample_validator=_validate_prunevid_sample,
        preprocessing_record=_preprocessing_record,
        projector_label="PruneVid",
        reference_revision=VIDEO_REFERENCE_REVISION,
        post_boundary_layer=CANONICAL_PRUNEVID_CONFIG["vtp_selected_layer"] + 1,
    )


def _parser() -> argparse.ArgumentParser:
    parser = _qwen3._parser()
    parser.set_defaults(
        video_fps=QWEN3_PRUNEVID_VIDEO_FPS,
        video_max_pixels=QWEN3_PRUNEVID_VIDEO_MAX_PIXELS,
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    return _qwen3._execute(_parser().parse_args(argv), profile=_profile())


if __name__ == "__main__":
    raise SystemExit(main())
