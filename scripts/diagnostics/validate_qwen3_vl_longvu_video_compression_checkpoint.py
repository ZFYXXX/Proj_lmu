"""Validate canonical LongVU video inference on Qwen3-VL.

The offline batch-one diagnostic reuses the shared Qwen3 real-checkpoint
harness and the same four fixed ``nextqa_mc_test`` documents.  It freezes the
local frozen DINOv2 selector, trainable SVA fusion, authoritative source grid,
sparse M-RoPE gather, compact ``visual_pos_masks``, synchronized DeepStack
branches, and bounded cached generation.
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
        validate_qwen25_vl_longvu_video_compression_checkpoint as _qwen25_longvu,
    )
    from scripts.diagnostics import (
        validate_qwen3_vl_video_compression_checkpoint as _qwen3,
    )
else:
    import validate_qwen25_vl_longvu_video_compression_checkpoint as _qwen25_longvu
    import validate_qwen3_vl_video_compression_checkpoint as _qwen3


FIXED_NEXTQA_DOC_IDS = _qwen3.FIXED_NEXTQA_DOC_IDS
CANONICAL_LONGVU_CONFIG = _qwen25_longvu.CANONICAL_LONGVU_CONFIG
QWEN3_LONGVU_VIDEO_FPS = 1.0
# LongVU emits a 12x12 query raster for each selected frame.  These bounds keep
# every fixed Qwen3 post-merge frame at least 12x12 at its widest aspect ratio.
QWEN3_LONGVU_VIDEO_MIN_PIXELS = 256 * 32 * 32
QWEN3_LONGVU_VIDEO_MAX_PIXELS = 320 * 32 * 32
_QWEN3_PATCH_SIZE = 16
_QWEN3_TIMING_TOLERANCE = 1e-6


def _conversation(
    *,
    prompt: str,
    video: Path,
    fps: float = QWEN3_LONGVU_VIDEO_FPS,
    min_pixels: int = QWEN3_LONGVU_VIDEO_MIN_PIXELS,
    max_pixels: int = QWEN3_LONGVU_VIDEO_MAX_PIXELS,
) -> list[dict[str, Any]]:
    if min_pixels <= 0:
        raise ValueError("Qwen3-VL LongVU min_pixels must be positive")
    if min_pixels > max_pixels:
        raise ValueError("Qwen3-VL LongVU min_pixels cannot exceed max_pixels")
    conversation = _qwen3._conversation(
        prompt=prompt,
        video=video,
        fps=fps,
        max_pixels=max_pixels,
    )
    conversation[1]["content"][0]["min_pixels"] = int(min_pixels)
    return conversation


def _conversation_for_args(
    *,
    prompt: str,
    video: Path,
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    return _conversation(
        prompt=prompt,
        video=video,
        fps=args.video_fps,
        min_pixels=args.video_min_pixels,
        max_pixels=args.video_max_pixels,
    )


def _stage() -> CompressionStageConfig:
    return CompressionStageConfig.from_dict(
        {
            "name": "longvu-video-qwen3-vl-real-checkpoint",
            "placement": "projector_level",
            "execution_phase": "generate",
            "plugin": "longvu",
            "algorithm": "longvu",
            "variant": "longvu.video",
            "target_modalities": ["video"],
            "conditioning_modalities": [],
            "preserved_modalities": ["text", "image", "audio"],
            "coupling_policy": "independent",
            "media_alignment": "spatiotemporal",
            "binding": "longvu.video.qwen3_vl",
            "config": {},
        }
    )


def _preprocessing_record(args: argparse.Namespace) -> Mapping[str, Any]:
    return {
        "fps": float(args.video_fps),
        "min_pixels": int(args.video_min_pixels),
        "max_pixels": int(args.video_max_pixels),
        "image_patch_size": _QWEN3_PATCH_SIZE,
        "enable_thinking": False,
        "provider": "qwen_vl_utils.process_vision_info",
        "constraint": (
            "native source grid remains authoritative while LongVU uses local "
            "DINOv2 selection and trainable SVA fusion"
        ),
    }


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
        and source_value > 0.0
        and math.isfinite(recorded_value)
        and math.isclose(
            source_value,
            recorded_value,
            rel_tol=0.0,
            abs_tol=_QWEN3_TIMING_TOLERANCE,
        )
    )


def _validate_longvu_sample(
    *,
    baseline: Mapping[str, Any],
    compressed: Mapping[str, Any],
    activation_records: Sequence[Mapping[str, Any]],
    algorithm_records: Sequence[Mapping[str, Any]],
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
    provider_capture_delta: int,
    provider_metadata: Mapping[str, Any],
    activation_metadata: Mapping[str, Any],
    auxiliary_checkpoint_paths: Mapping[str, Path],
    **_unused: Any,
) -> dict[str, bool]:
    if len(activation_records) != 1 or len(algorithm_records) != 1:
        raise RuntimeError(
            "LongVU must publish one layout and one algorithm record per sample"
        )
    if len(source_grid_thw) != 3:
        raise RuntimeError(f"Invalid Qwen3-VL source grid: {source_grid_thw!r}")
    temporal, raw_height, raw_width = tuple(int(value) for value in source_grid_thw)
    if (
        spatial_merge_size < 1
        or raw_height % spatial_merge_size
        or raw_width % spatial_merge_size
    ):
        raise RuntimeError("Qwen3-VL source grid is incompatible with its merger")

    layout = activation_records[0]
    algorithm = algorithm_records[0]
    splice_layout = latest_splice["layout_delta"]
    source_count = int(layout["source_count"])
    result_count = int(layout["result_count"])
    projected_height = raw_height // spatial_merge_size
    projected_width = raw_width // spatial_merge_size
    projected_per_frame = projected_height * projected_width
    expected_source = temporal * projected_per_frame
    expected_raw = temporal * raw_height * raw_width
    baseline_length = int(baseline["decoder"]["prefill"]["query_length"])
    compressed_length = int(compressed["decoder"]["prefill"]["query_length"])

    selected_by_video = tuple(algorithm["selected_frame_indices_by_video"])
    representatives_by_video = tuple(algorithm["representative_indices"])
    selected = () if len(selected_by_video) != 1 else tuple(selected_by_video[0])
    representatives = (
        () if len(representatives_by_video) != 1 else tuple(representatives_by_video[0])
    )
    selected_count = len(selected)

    native_spans = _qwen3._spans(source_placeholder_spans)
    recorded_source_spans = _qwen3._spans(latest_splice["source_placeholder_spans"])
    result_spans = _qwen3._spans(latest_splice["result_placeholder_spans"])
    splice_representatives = tuple(
        int(value) for value in latest_splice["representative_indices"]
    )
    source_mrope = tuple(
        tuple(int(value) for value in row)
        for row in latest_splice["source_visual_mrope"]
    )
    result_mrope = tuple(
        tuple(int(value) for value in row)
        for row in latest_splice["result_visual_mrope"]
    )
    expected_result_mrope = tuple(
        source_mrope[index] for index in splice_representatives
    )
    deepstack_source_shapes = tuple(
        tuple(int(value) for value in shape)
        for shape in algorithm["deepstack_source_shapes"]
    )
    algorithm_deepstack_result_shapes = tuple(
        tuple(int(value) for value in shape)
        for shape in algorithm["deepstack_result_shapes"]
    )
    splice_deepstack_result_shapes = tuple(
        tuple(int(value) for value in shape)
        for shape in latest_splice["deepstack_result_shapes"]
    )
    projected_shape = tuple(
        int(value) for value in algorithm["projected_features_shape"]
    )
    raw_shape = tuple(int(value) for value in algorithm["raw_features_shape"])
    pixels_shape = tuple(int(value) for value in algorithm["pixels_shape"])
    dino_shape = tuple(int(value) for value in algorithm["dinov2_features_shape"])
    result_shape = tuple(int(value) for value in algorithm["result_features_shape"])
    auxiliary_checkpoint = auxiliary_checkpoint_paths.get("dino_checkpoint")
    decode_record = latest_generation.get("last_decode_record")
    generated_token_count = int(compressed["generated_token_count"])
    decode_query_lengths = tuple(
        int(value) for value in compressed["decoder"]["decode_query_lengths"]
    )

    checks = {
        "compression_applied": bool(latest_splice["compression_applied"]),
        "non_identity_token_reduction": source_count > result_count > 0,
        "one_native_video": native_video_count == 1,
        "projected_frame_raster_supports_12x12_queries": (
            projected_height >= 12 and projected_width >= 12
        ),
        "native_temporal_spans_preserved": (
            recorded_source_spans == native_spans
            and len(native_spans) == temporal
            and all(end - start == projected_per_frame for start, end in native_spans)
        ),
        "selected_result_spans_exact": (
            len(result_spans) == selected_count
            and all(end - start == 144 for start, end in result_spans)
        ),
        "grid_and_source_counts_agree": (
            source_grid_token_count == expected_source == source_count
            and int(latest_splice["source_placeholder_count"]) == source_count
            and _qwen3._span_count(recorded_source_spans) == source_count
        ),
        "result_placeholder_count_agrees": (
            int(latest_splice["result_placeholder_count"]) == result_count
            and _qwen3._span_count(result_spans) == result_count
        ),
        "layout_records_agree": (
            int(splice_layout["source_count"])
            == int(algorithm["source_count"])
            == source_count
            and int(splice_layout["result_count"])
            == int(algorithm["result_count"])
            == result_count
        ),
        "decoder_length_reduction_exact": (
            baseline_length - compressed_length == source_count - result_count
        ),
        "source_video_grid_preserved": (
            bool(latest_splice["source_video_grid_preserved"])
            and tuple(int(value) for value in latest_splice["source_video_grid_thw"])
            == (temporal, raw_height, raw_width)
        ),
        "source_video_timing_preserved": _timing_matches(
            second_per_grid_seconds,
            latest_splice.get("second_per_grid_seconds"),
        ),
        "sparse_grid_policy": (
            latest_splice["result_grid_policy"] == "irregular_sparse_source_anchors"
        ),
        "representatives_exact_and_source_ordered": (
            len(representatives) == result_count
            and tuple(int(value) for value in representatives) == splice_representatives
            and splice_representatives == tuple(sorted(set(splice_representatives)))
        ),
        "mrope_source_count_aligned": (
            len(source_mrope) == source_count
            and all(len(row) == 3 for row in source_mrope)
        ),
        "mrope_result_count_aligned": (
            len(result_mrope) == result_count
            and all(len(row) == 3 for row in result_mrope)
        ),
        "mrope_sparse_anchor_gather_exact": result_mrope == expected_result_mrope,
        "visual_pos_masks_compacted": bool(latest_splice["visual_pos_masks_compacted"]),
        "deepstack_branch_count_exact": (
            int(latest_splice["deepstack_branch_count"])
            == int(algorithm["deepstack_branch_count"])
            == expected_deepstack_branches
            == len(deepstack_source_shapes)
            == len(algorithm_deepstack_result_shapes)
            == len(splice_deepstack_result_shapes)
        ),
        "deepstack_source_and_result_shapes_exact": (
            algorithm_deepstack_result_shapes == splice_deepstack_result_shapes
            and all(
                len(shape) == 2 and shape[0] == source_count
                for shape in deepstack_source_shapes
            )
            and all(
                len(shape) == 2 and shape[0] == result_count
                for shape in algorithm_deepstack_result_shapes
            )
        ),
        "main_deepstack_layout_shared": bool(
            latest_splice["shared_main_deepstack_layout"]
        ),
        "generation_prefill_complete": bool(latest_generation["prefill_complete"]),
        "generation_lengths_aligned": (
            int(latest_generation["source_sequence_length"]) == baseline_length
            and int(latest_generation["result_sequence_length"]) == compressed_length
            and int(latest_generation["removed_count"]) == source_count - result_count
        ),
        "cached_decode_exact": (
            generated_token_count >= 2
            and int(latest_generation["decode_repairs"]) == generated_token_count - 1
            and decode_query_lengths == (1,) * (generated_token_count - 1)
            and isinstance(decode_record, Mapping)
        ),
        "inference_labels_absent": labels_absent,
        "provider_capture_once": provider_capture_delta == 1,
        "provider_identity_exact": (
            provider_metadata.get("provider") == "qwen3_vl_longvu_video_signals"
        ),
        "provider_video_only": provider_metadata.get("video_only") is True,
        "provider_raw_stream_attached": (
            provider_metadata.get("raw_stream_attached") is True
        ),
        "provider_dino_frozen_and_loaded": (
            provider_metadata.get("dino_frozen") is True
            and provider_metadata.get("dino_loaded") is True
        ),
        "provider_dino_path_exact": (
            auxiliary_checkpoint is not None
            and Path(str(provider_metadata.get("dino_path"))) == auxiliary_checkpoint
        ),
        "fusion_registered_and_trainable": (
            activation_metadata.get("fusion_registered_pre_optimizer") is True
            and int(activation_metadata.get("fusion_parameter_count", 0)) > 0
            and int(activation_metadata.get("fusion_trainable_parameter_count", -1))
            == int(activation_metadata.get("fusion_parameter_count", 0))
        ),
        "canonical_config_exact": (
            dict(algorithm["canonical_config"]) == CANONICAL_LONGVU_CONFIG
        ),
        "canonical_result_count_exact": result_count == selected_count * 144,
        "single_video_counts_exact": (
            int(algorithm["video_count"]) == 1
            and tuple(algorithm["per_video_source_counts"]) == (source_count,)
            and tuple(algorithm["per_video_result_counts"]) == (result_count,)
            and tuple(algorithm["exact_budgets"]) == (result_count,)
        ),
        "projected_signal_shape_exact": (
            len(projected_shape) == 2 and projected_shape[0] == source_count
        ),
        "raw_signal_shape_exact": len(raw_shape) == 2 and raw_shape[0] == expected_raw,
        "pixel_signal_shape_exact": pixels_shape
        == (
            temporal,
            3,
            raw_height * _QWEN3_PATCH_SIZE,
            raw_width * _QWEN3_PATCH_SIZE,
        ),
        "dino_signal_shape_exact": dino_shape == (temporal, 576, 1536),
        "result_signal_shape_exact": (
            len(result_shape) == 2
            and result_shape[0] == result_count
            and result_shape[1] == projected_shape[1]
        ),
        "all_runtime_signals_finite": all(
            algorithm.get(name) is True
            for name in (
                "projected_features_finite",
                "raw_features_finite",
                "pixels_finite",
                "dinov2_features_finite",
                "result_features_finite",
            )
        ),
        "baseline_logits_finite": bool(baseline["logits_finite"]),
        "compressed_logits_finite": bool(compressed["logits_finite"]),
        "baseline_decoder_alignment": all(baseline["decoder"]["invariants"].values()),
        "compressed_decoder_alignment": all(
            compressed["decoder"]["invariants"].values()
        ),
    }
    if not all(checks.values()):
        raise RuntimeError(f"LongVU Qwen3-VL sample evidence failed: {checks!r}")
    return checks


def _profile() -> _qwen3.Qwen3VideoDiagnosticProfile:
    return _qwen3.Qwen3VideoDiagnosticProfile(
        algorithm="longvu",
        variant="longvu.video",
        binding="longvu.video.qwen3_vl",
        canonical_config=CANONICAL_LONGVU_CONFIG,
        stage_factory=_stage,
        conversation_factory=_conversation_for_args,
        sample_validator=_validate_longvu_sample,
        preprocessing_record=_preprocessing_record,
        projector_label="LongVU",
        reference_revision=VIDEO_REFERENCE_REVISION,
        auxiliary_checkpoints=(("dino_checkpoint", "longvu_dino_path"),),
    )


def _parser() -> argparse.ArgumentParser:
    parser = _qwen3._parser()
    parser.add_argument("--dino_checkpoint", type=Path, required=True)
    parser.add_argument(
        "--video_min_pixels",
        type=int,
        default=QWEN3_LONGVU_VIDEO_MIN_PIXELS,
    )
    parser.set_defaults(
        video_fps=QWEN3_LONGVU_VIDEO_FPS,
        video_max_pixels=QWEN3_LONGVU_VIDEO_MAX_PIXELS,
        estimated_peak_mib=57344,
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.max_new_tokens != 4:
        raise RuntimeError("LongVU fixed evidence requires max_new_tokens=4")
    return _qwen3._execute(args, profile=_profile())


if __name__ == "__main__":
    raise SystemExit(main())
