"""Validate canonical LongVU inference on LLaVA-OneVision.

The offline batch-one diagnostic compares the same four fixed NExT-QA samples
against their native baseline and freezes LongVU's local DINOv2 frame
selection, frozen feature provider, trainable SVA fusion, exact source groups,
LLaVA splice, decoder alignment, and finite generation behavior.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

from mllm_comp.pipeline.stage import CompressionStageConfig
from scripts.diagnostics import llava_provider_video_checkpoint_common as _common


FIXED_NEXTQA_DOC_IDS = _common.FIXED_NEXTQA_DOC_IDS
CANONICAL_LONGVU_CONFIG = {
    "window_size": 16,
    "threshold": 0.83,
    "max_frames": 400,
    "min_frames": 1,
    "dino_interp_size": 576,
    "vision_hidden_size": 1024,
    "query_num": 144,
    "connector_depth": 3,
}


def _stage() -> CompressionStageConfig:
    return CompressionStageConfig.from_dict(
        {
            "name": "longvu-video-llava-onevision-real-checkpoint",
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
            "binding": "longvu.video.llava_onevision",
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
    window_size = CANONICAL_LONGVU_CONFIG["window_size"]
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
    labels_absent: bool,
    placeholder_count: int,
    frame_count: int,
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
    baseline_length = int(baseline["decoder"]["prefill"]["query_length"])
    compressed_length = int(compressed["decoder"]["prefill"]["query_length"])
    selected_by_video = tuple(algorithm["selected_frame_indices_by_video"])
    selected = () if len(selected_by_video) != 1 else tuple(selected_by_video[0])
    selected_count = len(selected)
    representatives_by_video = tuple(algorithm["representative_indices"])
    representatives = (
        () if len(representatives_by_video) != 1 else tuple(representatives_by_video[0])
    )
    structural_source = tuple(latest_splice["structural_source_positions"])
    structural_result = tuple(latest_splice["structural_result_positions"])
    expected_source = frame_count * 196
    checks = {
        "compression_applied": bool(latest_splice["compression_applied"]),
        "non_identity_token_reduction": source_count > result_count > 0,
        "native_projected_frame_raster_exact": source_count == expected_source,
        "canonical_frame_count_exact": frame_count == 32,
        "layout_and_splice_counts_agree": (
            int(latest_splice["source_patch_count"]) == source_count
            and int(latest_splice["result_patch_count"]) == result_count
            and int(algorithm["source_count"]) == source_count
            and int(algorithm["result_count"]) == result_count
        ),
        "decoder_length_reduction_exact": (
            baseline_length - compressed_length == source_count - result_count
        ),
        "structural_tokens_preserved": len(structural_source) == len(structural_result),
        "one_video_placeholder": placeholder_count == 1,
        "inference_labels_absent": labels_absent,
        "provider_capture_once": provider_capture_delta == 1,
        "provider_video_only": provider_metadata.get("video_only") is True,
        "provider_raw_stream_attached": provider_metadata.get("raw_stream_attached")
        is True,
        "provider_dino_frozen_and_loaded": (
            provider_metadata.get("dino_frozen") is True
            and provider_metadata.get("dino_loaded") is True
        ),
        "provider_dino_path_exact": Path(str(provider_metadata.get("dino_path")))
        == auxiliary_checkpoint,
        "fusion_registered_and_trainable": (
            activation_metadata.get("fusion_registered_pre_optimizer") is True
            and int(activation_metadata.get("fusion_parameter_count", 0)) > 0
            and int(activation_metadata.get("fusion_trainable_parameter_count", -1))
            == int(activation_metadata.get("fusion_parameter_count", 0))
        ),
        "canonical_config_exact": dict(algorithm["canonical_config"])
        == CANONICAL_LONGVU_CONFIG,
        "canonical_result_count_exact": result_count == selected_count * 144,
        "single_video_counts_exact": (
            int(algorithm["video_count"]) == 1
            and tuple(algorithm["per_video_source_counts"]) == (source_count,)
            and tuple(algorithm["per_video_result_counts"]) == (result_count,)
            and tuple(algorithm["exact_budgets"]) == (result_count,)
        ),
        "projected_signal_shape_exact": tuple(algorithm["projected_features_shape"])[0]
        == source_count,
        "raw_signal_shape_exact": tuple(algorithm["raw_features_shape"])[0]
        == frame_count * 729,
        "pixel_signal_shape_exact": (
            tuple(algorithm["pixels_shape"])[0] == frame_count
            and tuple(algorithm["pixels_shape"])[1] == 3
        ),
        "dino_signal_shape_exact": tuple(algorithm["dinov2_features_shape"])
        == (frame_count, 576, 1536),
        "result_signal_shape_exact": tuple(algorithm["result_features_shape"])[0]
        == result_count,
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
        "representatives_source_ordered": representatives
        == tuple(sorted(set(int(value) for value in representatives))),
        "baseline_logits_finite": bool(baseline["logits_finite"]),
        "compressed_logits_finite": bool(compressed["logits_finite"]),
        "baseline_decoder_alignment": all(baseline["decoder"]["invariants"].values()),
        "compressed_decoder_alignment": all(
            compressed["decoder"]["invariants"].values()
        ),
    }
    if not all(checks.values()):
        raise RuntimeError(f"LongVU LLaVA sample evidence failed: {checks!r}")
    return checks


def _profile() -> _common.LlavaProviderVideoDiagnosticProfile:
    return _common.LlavaProviderVideoDiagnosticProfile(
        algorithm="longvu",
        variant="longvu.video",
        binding="longvu.video.llava_onevision",
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
            "llava_projector_to_llm_splice",
        ),
        estimate_basis=(
            "32 GiB includes the 0.5B LLaVA checkpoint, full DINOv2 Giant, "
            "32-frame vision features, trainable SVA fusion, baseline plus "
            "compressed generation, and allocator variance"
        ),
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    _common.add_common_arguments(parser)
    parser.add_argument("--dino_checkpoint", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    return _common.main(_parser().parse_args(argv), profile=_profile())


if __name__ == "__main__":
    raise SystemExit(main())
