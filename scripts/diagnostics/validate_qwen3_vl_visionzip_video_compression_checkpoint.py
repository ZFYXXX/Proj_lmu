"""Validate canonical VisionZip video inference on Qwen3-VL.

The offline, batch-one diagnostic reuses the shared Qwen3 real-checkpoint
harness and the same four fixed ``nextqa_mc_test`` documents.  It additionally
requires VisionZip's native penultimate attention/hidden signals, independent
``126 dominant + 21 contextual`` aggregation in every temporal slice, and
exact source-group/merge-target replay on all native DeepStack branches.

Qwen3's interleaved temporal placeholder spans, original grid, sparse M-RoPE,
``visual_pos_masks``, masks, positions, cache inputs, and bounded decode remain
owned by the shared Qwen3 legalizer rather than by a fabricated dense result
grid.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Mapping, Sequence

from mllm_comp.compressors.video.config import VIDEO_REFERENCE_REVISION
from mllm_comp.pipeline.stage import CompressionStageConfig

if __package__:
    from scripts.diagnostics import (
        validate_qwen3_vl_video_compression_checkpoint as _qwen3,
    )
else:
    import validate_qwen3_vl_video_compression_checkpoint as _qwen3


FIXED_NEXTQA_DOC_IDS = _qwen3.FIXED_NEXTQA_DOC_IDS
CANONICAL_VISIONZIP_CONFIG = {
    "dominant_num": 126,
    "contextual_num": 21,
    "attn_layer": -2,
    "hidden_state_layer": -2,
}
QWEN3_VISIONZIP_VIDEO_FPS = 1.0
QWEN3_VISIONZIP_VIDEO_MIN_PIXELS = 192 * 32 * 32
QWEN3_VISIONZIP_VIDEO_MAX_PIXELS = 224 * 32 * 32


def _conversation(
    *,
    prompt: str,
    video: Path,
    fps: float = QWEN3_VISIONZIP_VIDEO_FPS,
    min_pixels: int = QWEN3_VISIONZIP_VIDEO_MIN_PIXELS,
    max_pixels: int = QWEN3_VISIONZIP_VIDEO_MAX_PIXELS,
) -> list[dict[str, Any]]:
    if min_pixels <= 0:
        raise ValueError("Qwen3-VL VisionZip min_pixels must be positive")
    if min_pixels > max_pixels:
        raise ValueError("Qwen3-VL VisionZip min_pixels cannot exceed max_pixels")
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
            "name": "visionzip-video-qwen3-vl-real-checkpoint",
            "placement": "projector_level",
            "execution_phase": "generate",
            "plugin": "visionzip",
            "algorithm": "visionzip",
            "variant": "visionzip.video",
            "target_modalities": ["video"],
            "conditioning_modalities": [],
            "preserved_modalities": ["text", "image", "audio"],
            "coupling_policy": "independent",
            "media_alignment": "spatiotemporal",
            "binding": "visionzip.video.qwen3_vl",
            "config": {},
        }
    )


def _preprocessing_record(args: argparse.Namespace) -> Mapping[str, Any]:
    return {
        "fps": float(args.video_fps),
        "min_pixels": int(args.video_min_pixels),
        "max_pixels": int(args.video_max_pixels),
        "image_patch_size": 16,
        "enable_thinking": False,
        "provider": "qwen_vl_utils.process_vision_info",
        "constraint": "each post-merger temporal slice exceeds 147 tokens",
    }


def _profile() -> _qwen3.Qwen3VideoDiagnosticProfile:
    return _qwen3.Qwen3VideoDiagnosticProfile(
        algorithm="visionzip",
        variant="visionzip.video",
        binding="visionzip.video.qwen3_vl",
        canonical_config=CANONICAL_VISIONZIP_CONFIG,
        stage_factory=_stage,
        conversation_factory=_conversation_for_args,
        sample_validator=_validate_visionzip_sample,
        preprocessing_record=_preprocessing_record,
        projector_label="VisionZip",
        reference_revision=VIDEO_REFERENCE_REVISION,
    )


def _validate_visionzip_sample(
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
    labels_absent: bool,
    source_placeholder_spans: Sequence[Sequence[int]],
    native_video_count: int,
    expected_deepstack_branches: int,
    provider_capture_delta: int,
    **_unused: Any,
) -> dict[str, bool]:
    if len(activation_records) != 1 or len(algorithm_records) != 1:
        raise RuntimeError(
            "VisionZip must publish one layout and one algorithm record per sample"
        )
    if len(source_grid_thw) != 3:
        raise RuntimeError(f"Invalid Qwen3-VL source grid: {source_grid_thw!r}")

    layout = activation_records[0]
    algorithm = algorithm_records[0]
    splice_layout = latest_splice["layout_delta"]
    source_count = int(layout["source_count"])
    result_count = int(layout["result_count"])
    temporal, height, width = (int(item) for item in source_grid_thw)
    per_slice_source = height * width // (spatial_merge_size**2)
    expected_result = temporal * 147
    baseline_length = int(baseline["decoder"]["prefill"]["query_length"])
    compressed_length = int(compressed["decoder"]["prefill"]["query_length"])
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
    expected_result_mrope = tuple(source_mrope[index] for index in representatives)
    frame_summaries = tuple(algorithm["frame_summaries"])
    decode_record = latest_generation.get("last_decode_record")
    native_spans = _qwen3._spans(source_placeholder_spans)
    recorded_source_spans = _qwen3._spans(latest_splice["source_placeholder_spans"])
    result_spans = _qwen3._spans(latest_splice["result_placeholder_spans"])
    deepstack_shapes = tuple(
        tuple(int(item) for item in shape)
        for shape in latest_splice["deepstack_result_shapes"]
    )
    algorithm_deepstack_shapes = tuple(
        tuple(int(item) for item in shape)
        for shape in algorithm["deepstack_result_shapes"]
    )

    checks = {
        "compression_applied": bool(latest_splice["compression_applied"]),
        "non_identity_token_reduction": source_count > result_count,
        "one_native_video": native_video_count == 1,
        "native_temporal_spans_preserved": (
            recorded_source_spans == native_spans
            and len(native_spans) == temporal
            and all(end - start == per_slice_source for start, end in native_spans)
        ),
        "result_temporal_spans_exact": (
            len(result_spans) == temporal
            and all(end - start == 147 for start, end in result_spans)
        ),
        "grid_and_source_counts_agree": (
            source_grid_token_count == source_count
            and int(latest_splice["source_placeholder_count"]) == source_count
            and _qwen3._span_count(recorded_source_spans) == source_count
        ),
        "source_per_slice_strictly_exceeds_result": per_slice_source > 147,
        "canonical_per_slice_result_exact": result_count == expected_result,
        "result_placeholder_count_agrees": (
            int(latest_splice["result_placeholder_count"]) == result_count
            and _qwen3._span_count(result_spans) == result_count
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
            and tuple(int(item) for item in latest_splice["source_video_grid_thw"])
            == (temporal, height, width)
        ),
        "sparse_grid_policy": (
            latest_splice["result_grid_policy"] == "irregular_sparse_source_anchors"
        ),
        "representatives_source_ordered": (
            len(representatives) == result_count
            and representatives == tuple(sorted(representatives))
            and len(set(representatives)) == result_count
        ),
        "mrope_source_count_aligned": len(source_mrope) == source_count,
        "mrope_result_count_aligned": len(result_mrope) == result_count,
        "mrope_sparse_anchor_gather_exact": result_mrope == expected_result_mrope,
        "visual_pos_masks_compacted": bool(latest_splice["visual_pos_masks_compacted"]),
        "deepstack_branch_count_exact": (
            int(latest_splice["deepstack_branch_count"])
            == int(algorithm["deepstack_branch_count"])
            == expected_deepstack_branches
            == len(deepstack_shapes)
            == len(algorithm_deepstack_shapes)
        ),
        "deepstack_result_counts_exact": (
            deepstack_shapes == algorithm_deepstack_shapes
            and all(
                len(shape) == 2 and shape[0] == result_count
                for shape in deepstack_shapes
            )
        ),
        "main_deepstack_layout_shared": bool(
            latest_splice["shared_main_deepstack_layout"]
        ),
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
        "canonical_config_exact": (
            dict(algorithm["canonical_config"]) == CANONICAL_VISIONZIP_CONFIG
        ),
        "penultimate_signal_layers_exact": (
            int(algorithm["attention_layer"]) == -2
            and int(algorithm["hidden_state_layer"]) == -2
        ),
        "penultimate_signal_shapes_aligned": (
            tuple(algorithm["attention_shape"]) == (source_count,)
            and len(tuple(algorithm["hidden_state_shape"])) == 2
            and int(tuple(algorithm["hidden_state_shape"])[0]) == source_count
            and int(tuple(algorithm["hidden_state_shape"])[1]) > 0
        ),
        "penultimate_signals_finite": (
            bool(algorithm["attention_finite"])
            and bool(algorithm["hidden_state_finite"])
        ),
        "temporal_slice_count_exact": (
            int(algorithm["temporal_slice_count"]) == temporal
            and len(frame_summaries) == temporal
        ),
        "algorithm_representatives_source_ordered": bool(
            algorithm["representatives_source_ordered"]
        ),
        "baseline_logits_finite": bool(baseline["logits_finite"]),
        "compressed_logits_finite": bool(compressed["logits_finite"]),
        "baseline_decoder_alignment": all(baseline["decoder"]["invariants"].values()),
        "compressed_decoder_alignment": all(
            compressed["decoder"]["invariants"].values()
        ),
    }
    if not all(checks.values()):
        raise RuntimeError(f"VisionZip Qwen3-VL sample evidence failed: {checks!r}")
    return checks


def _parser() -> argparse.ArgumentParser:
    parser = _qwen3._parser()
    parser.set_defaults(
        video_fps=QWEN3_VISIONZIP_VIDEO_FPS,
        video_max_pixels=QWEN3_VISIONZIP_VIDEO_MAX_PIXELS,
    )
    parser.add_argument(
        "--video_min_pixels",
        type=int,
        default=QWEN3_VISIONZIP_VIDEO_MIN_PIXELS,
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    return _qwen3._execute(_parser().parse_args(argv), profile=_profile())


if __name__ == "__main__":
    raise SystemExit(main())
