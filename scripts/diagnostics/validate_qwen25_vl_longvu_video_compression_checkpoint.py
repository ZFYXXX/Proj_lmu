"""Validate canonical LongVU inference on Qwen2.5-VL.

The offline batch-one diagnostic compares the fixed four NExT-QA samples with
their native baselines and freezes LongVU's local DINOv2 frame selection,
trainable SVA fusion, authoritative source ``video_grid_thw``, sparse native
M-RoPE gather, cached decode repair, and finite generation behavior.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

from mllm_comp.compressors.video.config import CANONICAL_VIDEO_CONFIGS
from mllm_comp.pipeline.stage import CompressionStageConfig
from scripts.diagnostics import qwen25_provider_video_checkpoint_common as _common


FIXED_NEXTQA_DOC_IDS = _common.FIXED_NEXTQA_DOC_IDS
CANONICAL_LONGVU_CONFIG = dict(CANONICAL_VIDEO_CONFIGS["longvu"])
QWEN25_LONGVU_VIDEO_FPS = 1.0
# LongVU emits a 12x12 structural query raster for every selected frame.  The
# fixed Qwen videos reach a worst native aspect ratio of 1.875, so this lower
# bound preserves at least 24x24 raw patches (12x12 after spatial merge).
QWEN25_LONGVU_VIDEO_MIN_PIXELS = 256 * 28 * 28
QWEN25_LONGVU_VIDEO_MAX_PIXELS = 320 * 28 * 28


def _stage() -> CompressionStageConfig:
    return CompressionStageConfig.from_dict(
        {
            "name": "longvu-video-qwen25-vl-real-checkpoint",
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
            "binding": "longvu.video.qwen25_vl",
            "config": {},
        }
    )


def _selected_frames_exact(
    selected: Sequence[int],
    scores: Sequence[Sequence[float]],
    *,
    frame_count: int,
) -> bool:
    selected_tuple = tuple(int(value) for value in selected)
    if (
        not selected_tuple
        or selected_tuple != tuple(sorted(set(selected_tuple)))
        or selected_tuple[0] < 0
        or selected_tuple[-1] >= frame_count
    ):
        return False
    window_size = int(CANONICAL_LONGVU_CONFIG["window_size"])
    expected_lengths = tuple(
        min(window_size, frame_count - start)
        for start in range(0, frame_count, window_size)
    )
    score_rows = tuple(tuple(float(value) for value in row) for row in scores)
    if tuple(len(row) for row in score_rows) != expected_lengths:
        return False
    if any(not math.isfinite(value) for row in score_rows for value in row):
        return False
    forced_midpoints = tuple(
        start + (stop - start) // 2
        for start in range(0, frame_count, window_size)
        for stop in (min(start + window_size, frame_count),)
    )
    midpoint_scores = tuple(row[len(row) // 2] for row in score_rows)
    return all(index in selected_tuple for index in forced_midpoints) and all(
        value == 0.0 for value in midpoint_scores
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
    merge_size: int,
    labels_absent: bool,
    video_span_count: int,
    provider_capture_delta: int,
    provider_metadata: Mapping[str, Any],
    activation_metadata: Mapping[str, Any],
    auxiliary_checkpoint: Path,
    **_unused: Any,
) -> dict[str, bool]:
    if len(activation_records) != 1 or len(algorithm_records) != 1:
        raise RuntimeError(
            "LongVU must publish one layout and one algorithm record per sample"
        )
    layout = activation_records[0]
    algorithm = algorithm_records[0]
    source_count = int(layout["source_count"])
    result_count = int(layout["result_count"])
    temporal, raw_height, raw_width = tuple(int(value) for value in source_grid_thw)
    projected_height = raw_height // merge_size
    projected_width = raw_width // merge_size
    baseline_length = int(baseline["decoder"]["prefill"]["query_length"])
    compressed_length = int(compressed["decoder"]["prefill"]["query_length"])

    selected_by_video = tuple(algorithm["selected_frame_indices_by_video"])
    selected = () if len(selected_by_video) != 1 else tuple(selected_by_video[0])
    selected_count = len(selected)
    representatives_by_video = tuple(algorithm["representative_indices"])
    representatives = (
        () if len(representatives_by_video) != 1 else tuple(representatives_by_video[0])
    )

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
    splice_layout = latest_splice["layout_delta"]
    decode_record = latest_generation.get("last_decode_record")
    expected_source = temporal * projected_height * projected_width
    expected_raw = temporal * raw_height * raw_width
    projected_shape = tuple(
        int(value) for value in algorithm["projected_features_shape"]
    )
    raw_shape = tuple(int(value) for value in algorithm["raw_features_shape"])
    pixels_shape = tuple(int(value) for value in algorithm["pixels_shape"])
    dino_shape = tuple(int(value) for value in algorithm["dinov2_features_shape"])
    result_shape = tuple(int(value) for value in algorithm["result_features_shape"])

    checks = {
        "compression_applied": bool(latest_splice["compression_applied"]),
        "non_identity_token_reduction": source_count > result_count > 0,
        "one_contiguous_video_span": video_span_count == 1,
        "projected_frame_raster_supports_12x12_queries": (
            raw_height % merge_size == 0
            and raw_width % merge_size == 0
            and projected_height >= 12
            and projected_width >= 12
        ),
        "grid_and_source_counts_agree": (
            source_grid_token_count == expected_source == source_count
            and int(latest_splice["source_placeholder_count"]) == source_count
        ),
        "result_placeholder_count_agrees": (
            int(latest_splice["result_placeholder_count"]) == result_count
        ),
        "layout_records_agree": (
            int(splice_layout["source_count"]) == source_count
            and int(splice_layout["result_count"]) == result_count
            and int(algorithm["source_count"]) == source_count
            and int(algorithm["result_count"]) == result_count
        ),
        "decoder_length_reduction_exact": (
            baseline_length - compressed_length == source_count - result_count
        ),
        "source_video_grid_preserved": (
            bool(latest_splice["source_video_grid_preserved"])
            and tuple(int(value) for value in latest_splice["source_video_grid_thw"])
            == (temporal, raw_height, raw_width)
        ),
        "native_timing_preserved": (
            math.isfinite(float(latest_splice["second_per_grid_seconds"]))
            and float(latest_splice["second_per_grid_seconds"]) > 0.0
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
        "rope_delta_published": bool(latest_splice["rope_deltas"]),
        "generation_prefill_complete": bool(latest_generation["prefill_complete"]),
        "generation_source_length_aligned": (
            int(latest_generation["source_sequence_length"]) == baseline_length
        ),
        "generation_result_length_aligned": (
            int(latest_generation["result_sequence_length"]) == compressed_length
        ),
        "generation_removed_count_aligned": (
            int(latest_generation["removed_count"]) == source_count - result_count
        ),
        "decode_repair_exercised": (
            int(latest_generation["decode_repairs"]) >= 1
            and isinstance(decode_record, Mapping)
        ),
        "inference_labels_absent": labels_absent,
        "provider_capture_once": provider_capture_delta == 1,
        "provider_video_only": provider_metadata.get("video_only") is True,
        "provider_raw_stream_attached": (
            provider_metadata.get("raw_stream_attached") is True
        ),
        "provider_dino_frozen_and_loaded": (
            provider_metadata.get("dino_frozen") is True
            and provider_metadata.get("dino_loaded") is True
        ),
        "provider_dino_path_exact": (
            Path(str(provider_metadata.get("dino_path"))) == auxiliary_checkpoint
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
        == (temporal, 3, raw_height * 14, raw_width * 14),
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
        raise RuntimeError(f"LongVU Qwen2.5-VL sample evidence failed: {checks!r}")
    return checks


def _profile() -> _common.Qwen25ProviderVideoDiagnosticProfile:
    return _common.Qwen25ProviderVideoDiagnosticProfile(
        algorithm="longvu",
        variant="longvu.video",
        binding="longvu.video.qwen25_vl",
        canonical_config=CANONICAL_LONGVU_CONFIG,
        auxiliary_argument="dino_checkpoint",
        auxiliary_runtime_key="longvu_dino_path",
        auxiliary_report_key="dinov2_checkpoint",
        auxiliary_label="DINOv2 Giant",
        stage_factory=_stage,
        sample_validator=_validate_longvu_sample,
        defining_stages=(
            "local_frozen_dinov2_frame_selection",
            "trainable_sva_native_dino_fusion",
            "qwen25_native_grid_sparse_mrope_splice",
            "qwen25_cached_decode_repair",
        ),
        estimate_basis=(
            "56 GiB includes the Qwen2.5-VL 3B checkpoint, full fp32 DINOv2 "
            "Giant, fixed 1-fps videos at the LongVU-safe spatial floor, "
            "trainable SVA fusion, native baseline plus compressed cached "
            "generation, and allocator variance"
        ),
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    _common.add_common_arguments(
        parser,
        video_min_pixels=QWEN25_LONGVU_VIDEO_MIN_PIXELS,
        video_max_pixels=QWEN25_LONGVU_VIDEO_MAX_PIXELS,
        estimated_peak_mib=57344,
    )
    parser.set_defaults(video_fps=QWEN25_LONGVU_VIDEO_FPS)
    parser.add_argument("--dino_checkpoint", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    return _common.main(_parser().parse_args(argv), profile=_profile())


if __name__ == "__main__":
    raise SystemExit(main())
