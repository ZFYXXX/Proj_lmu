"""Validate canonical DyCoKe video inference on Qwen3-VL.

The offline batch-one diagnostic reuses the shared Qwen3 real-checkpoint
harness and the same four fixed ``nextqa_mc_test`` documents.  It freezes the
two-pass temporal token merge, identical replay on all DeepStack branches,
native sparse M-RoPE and ``visual_pos_masks`` legalization, and the
non-destructive decode-time KV views used from decoder layer 3 onward.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

from mllm_comp.compressors.video.config import VIDEO_REFERENCE_REVISION
from mllm_comp.pipeline.stage import CompressionStageConfig

if __package__:
    from scripts.diagnostics import (
        validate_qwen25_vl_dycoke_video_compression_checkpoint as _dycoke,
    )
    from scripts.diagnostics import (
        validate_qwen3_vl_video_compression_checkpoint as _qwen3,
    )
else:
    import validate_qwen25_vl_dycoke_video_compression_checkpoint as _dycoke
    import validate_qwen3_vl_video_compression_checkpoint as _qwen3


FIXED_NEXTQA_DOC_IDS = _qwen3.FIXED_NEXTQA_DOC_IDS
CANONICAL_DYCOKE_CONFIG = _dycoke.CANONICAL_DYCOKE_CONFIG
QWEN3_DYCOKE_VIDEO_FPS = _qwen3.QWEN3_VIDEO_FPS
QWEN3_DYCOKE_VIDEO_MAX_PIXELS = _qwen3.QWEN3_VIDEO_MAX_PIXELS
_TIMING_TOLERANCE = 1e-6


_conversation_for_args = _qwen3._conversation_for_args
_preprocessing_record = _qwen3._preprocessing_record


def _conversation(
    *,
    prompt: str,
    video: Path,
    fps: float = QWEN3_DYCOKE_VIDEO_FPS,
    max_pixels: int = QWEN3_DYCOKE_VIDEO_MAX_PIXELS,
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
            "name": "dycoke-video-qwen3-vl-real-checkpoint",
            "placement": "projector_level",
            "execution_phase": "generate",
            "plugin": "dycoke",
            "algorithm": "dycoke",
            "variant": "dycoke.video",
            "target_modalities": ["video"],
            "conditioning_modalities": ["text"],
            "preserved_modalities": ["text", "image", "audio"],
            "coupling_policy": "independent",
            "media_alignment": "spatiotemporal",
            "binding": "dycoke.video.qwen3_vl",
            "config": {},
        }
    )


def _timing_matches(source: float | None, recorded: Any) -> bool:
    if source is None:
        return recorded is None
    try:
        source_value = float(source)
        recorded_value = float(recorded)
    except (TypeError, ValueError):
        return False
    return (
        math.isfinite(source_value)
        and source_value > 0
        and math.isfinite(recorded_value)
        and recorded_value > 0
        and math.isclose(
            source_value,
            recorded_value,
            rel_tol=_TIMING_TOLERANCE,
            abs_tol=_TIMING_TOLERANCE,
        )
    )


def _span_positions(spans: Sequence[Sequence[int]]) -> tuple[int, ...]:
    return tuple(
        position
        for start, end in _qwen3._spans(spans)
        for position in range(start, end)
    )


def _profile() -> _qwen3.Qwen3VideoDiagnosticProfile:
    return _qwen3.Qwen3VideoDiagnosticProfile(
        algorithm="dycoke",
        variant="dycoke.video",
        binding="dycoke.video.qwen3_vl",
        canonical_config=CANONICAL_DYCOKE_CONFIG,
        stage_factory=_stage,
        conversation_factory=_conversation_for_args,
        sample_validator=_validate_dycoke_sample,
        preprocessing_record=_preprocessing_record,
        projector_label="DyCoKe",
        reference_revision=VIDEO_REFERENCE_REVISION,
        secondary_layout_key="decode_kv_views",
        min_new_tokens=4,
    )


def _validate_dycoke_sample(
    *,
    baseline: Mapping[str, Any],
    compressed: Mapping[str, Any],
    activation_records: Sequence[Mapping[str, Any]],
    algorithm_records: Sequence[Mapping[str, Any]],
    secondary_records: Sequence[Mapping[str, Any]],
    latest_splice: Mapping[str, Any],
    latest_generation: Mapping[str, Any],
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
    if len(activation_records) != 1 or len(algorithm_records) != 1:
        raise RuntimeError(
            "DyCoKe must publish one layout and one algorithm record per sample"
        )
    if len(source_grid_thw) != 3 or spatial_merge_size < 1:
        raise RuntimeError("Qwen3-VL source grid metadata is invalid")
    temporal, height, width = (int(item) for item in source_grid_thw)
    spatial_area = height * width
    merge_area = spatial_merge_size**2
    if spatial_area % merge_area:
        raise RuntimeError("Qwen3-VL grid is not divisible by spatial merge size")
    tokens_per_frame = spatial_area // merge_area
    expected_frame_counts = _dycoke._expected_frame_result_counts(
        temporal,
        tokens_per_frame,
    )
    expected_source = temporal * tokens_per_frame
    expected_result = sum(expected_frame_counts)

    layout = activation_records[0]
    algorithm = algorithm_records[0]
    source_count = int(layout["source_count"])
    result_count = int(layout["result_count"])
    baseline_length = int(baseline["decoder"]["prefill"]["query_length"])
    compressed_length = int(compressed["decoder"]["prefill"]["query_length"])
    decode_queries = tuple(
        int(item) for item in compressed["decoder"]["decode_query_lengths"]
    )
    frame_records_by_video = tuple(algorithm["frame_records_by_video"])
    frame_records = (
        () if len(frame_records_by_video) != 1 else tuple(frame_records_by_video[0])
    )
    selected_sources = tuple(
        int(item)
        for frame in frame_records
        for item in frame["selected_source_indices"]
    )
    representatives = tuple(
        int(item) for item in latest_splice["representative_indices"]
    )
    source_mrope = tuple(
        tuple(int(value) for value in row)
        for row in latest_splice["source_visual_mrope"]
    )
    result_mrope = tuple(
        tuple(int(value) for value in row)
        for row in latest_splice["result_visual_mrope"]
    )
    native_spans = _qwen3._spans(source_placeholder_spans)
    recorded_source_spans = _qwen3._spans(latest_splice["source_placeholder_spans"])
    result_spans = _qwen3._spans(latest_splice["result_placeholder_spans"])
    source_positions = _span_positions(recorded_source_spans)
    result_positions = _span_positions(result_spans)
    retained_source_positions = tuple(
        int(item) for item in latest_splice["retained_placeholder_positions"]
    )
    algorithm_result_positions = tuple(
        int(item) for item in algorithm["result_visual_positions"]
    )
    splice_layout = latest_splice["layout_delta"]
    rope_deltas = tuple(int(item) for item in latest_splice["rope_deltas"])
    source_deepstack_shapes = tuple(
        tuple(int(item) for item in shape)
        for shape in algorithm["deepstack_source_shapes"]
    )
    algorithm_deepstack_shapes = tuple(
        tuple(int(item) for item in shape)
        for shape in algorithm["deepstack_result_shapes"]
    )
    splice_deepstack_shapes = tuple(
        tuple(int(item) for item in shape)
        for shape in latest_splice["deepstack_result_shapes"]
    )

    checks = {
        "compression_applied": bool(latest_splice["compression_applied"]),
        "non_identity_token_reduction": source_count > result_count,
        "one_native_video": native_video_count == int(algorithm["video_count"]) == 1,
        "native_temporal_spans_preserved": (
            recorded_source_spans == native_spans
            and _qwen3._span_count(recorded_source_spans) == source_count
        ),
        "native_grid_source_count_exact": (
            source_grid_token_count == source_count == expected_source
            and int(latest_splice["source_placeholder_count"]) == source_count
        ),
        "canonical_ttm_result_exact": result_count == expected_result,
        "layout_splice_algorithm_counts_agree": (
            int(splice_layout["source_count"]) == source_count
            and int(splice_layout["result_count"]) == result_count
            and int(latest_splice["result_placeholder_count"]) == result_count
            and int(algorithm["source_count"]) == source_count
            and int(algorithm["result_count"]) == result_count
            and tuple(int(item) for item in algorithm["exact_budgets"])
            == (result_count,)
        ),
        "decoder_length_reduction_exact": (
            baseline_length - compressed_length == source_count - result_count
        ),
        "source_video_grid_preserved": (
            latest_splice["source_video_grid_preserved"] is True
            and tuple(int(item) for item in latest_splice["source_video_grid_thw"])
            == (temporal, height, width)
        ),
        "source_video_timing_preserved": _timing_matches(
            second_per_grid_seconds,
            latest_splice.get("second_per_grid_seconds"),
        ),
        "sparse_grid_policy_exact": (
            latest_splice["result_grid_policy"] == "irregular_sparse_source_anchors"
        ),
        "representatives_source_ordered": (
            representatives == selected_sources
            and len(representatives) == result_count
            and representatives == tuple(sorted(set(representatives)))
            and algorithm["representatives_source_ordered"] is True
        ),
        "retained_source_positions_exact": retained_source_positions
        == tuple(source_positions[index] for index in representatives),
        "result_placeholder_positions_exact": (
            len(result_positions) == result_count
            and result_positions == algorithm_result_positions
        ),
        "mrope_source_count_aligned": len(source_mrope) == source_count,
        "mrope_result_count_aligned": len(result_mrope) == result_count,
        "mrope_sparse_anchor_gather_exact": result_mrope
        == tuple(source_mrope[index] for index in representatives),
        "rope_delta_scalar_recorded": len(rope_deltas) == 1,
        "visual_pos_masks_compacted": bool(latest_splice["visual_pos_masks_compacted"]),
        "deepstack_branch_count_exact": (
            int(algorithm["deepstack_branch_count"])
            == int(latest_splice["deepstack_branch_count"])
            == expected_deepstack_branches
            == len(source_deepstack_shapes)
            == len(algorithm_deepstack_shapes)
            == len(splice_deepstack_shapes)
        ),
        "deepstack_source_layout_exact": all(
            len(shape) == 2 and shape[0] == source_count and shape[1] > 0
            for shape in source_deepstack_shapes
        ),
        "deepstack_result_layout_exact": (
            algorithm_deepstack_shapes == splice_deepstack_shapes
            and all(
                len(shape) == 2 and shape[0] == result_count and shape[1] > 0
                for shape in algorithm_deepstack_shapes
            )
        ),
        "main_deepstack_layout_shared": (
            latest_splice["shared_main_deepstack_layout"] is True
        ),
        "canonical_config_exact": dict(algorithm["canonical_config"])
        == CANONICAL_DYCOKE_CONFIG,
        "decode_stage_exact": (
            algorithm["decode_stage"] == "full_cache_sparse_view_from_layer_3"
            and algorithm["physical_cache_retained"] is True
        ),
        "decode_queries_exact": len(decode_queries) == 3
        and all(item == 1 for item in decode_queries),
        "decode_kv_views_exact": _dycoke._validate_decode_records(
            secondary_records,
            compressed_prefill_length=compressed_length,
            result_visual_positions=result_positions,
            decode_query_count=len(decode_queries),
        ),
        "generation_prefill_complete": latest_generation["prefill_complete"] is True,
        "generation_lengths_exact": (
            int(latest_generation["source_sequence_length"]) == baseline_length
            and int(latest_generation["result_sequence_length"]) == compressed_length
            and int(latest_generation["removed_count"]) == source_count - result_count
        ),
        "decode_repair_exact": _dycoke._decode_repair_exact(
            latest_generation,
            baseline_length=baseline_length,
            compressed_length=compressed_length,
            decode_query_count=len(decode_queries),
            removed_count=source_count - result_count,
        ),
        "inference_labels_absent": labels_absent,
        "baseline_logits_finite": baseline["logits_finite"] is True,
        "compressed_logits_finite": compressed["logits_finite"] is True,
        "baseline_decoder_alignment": all(baseline["decoder"]["invariants"].values()),
        "compressed_decoder_alignment": all(
            compressed["decoder"]["invariants"].values()
        ),
        "baseline_bounded_generation": int(baseline["generated_token_count"]) == 4,
        "compressed_bounded_generation": int(compressed["generated_token_count"]) == 4,
    }
    if not all(checks.values()):
        raise RuntimeError(f"DyCoKe Qwen3-VL sample evidence failed: {checks!r}")
    return checks


def _parser() -> argparse.ArgumentParser:
    parser = _qwen3._parser()
    parser.set_defaults(
        video_fps=QWEN3_DYCOKE_VIDEO_FPS,
        video_max_pixels=QWEN3_DYCOKE_VIDEO_MAX_PIXELS,
        max_new_tokens=4,
        estimated_peak_mib=32768,
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    return _qwen3._execute(_parser().parse_args(argv), profile=_profile())


if __name__ == "__main__":
    raise SystemExit(main())
