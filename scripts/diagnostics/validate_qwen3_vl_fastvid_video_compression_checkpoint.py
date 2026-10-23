"""Validate canonical FastVID video inference on Qwen3-VL.

The offline, batch-one diagnostic reuses the shared Qwen3 real-checkpoint
harness and the same four fixed ``nextqa_mc_test`` documents.  It freezes the
native final-block attention/raw stream, projected frame globals, exact
DySeg/STPrune/DTM decisions, and potentially overlapping value lineage.  That
lineage must be replayed on every native DeepStack branch.

Qwen3's authoritative grid/timing, interleaved temporal placeholders, sparse
M-RoPE, ``visual_pos_masks``, masks, positions, cache inputs, and bounded
decode remain owned by the shared legalizer rather than by a fabricated dense
result grid.
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
        validate_qwen25_vl_fastvid_video_compression_checkpoint as _qwen25_fastvid,
    )
    from scripts.diagnostics import (
        validate_qwen3_vl_video_compression_checkpoint as _qwen3,
    )
else:
    import validate_qwen25_vl_fastvid_video_compression_checkpoint as _qwen25_fastvid
    import validate_qwen3_vl_video_compression_checkpoint as _qwen3


FIXED_NEXTQA_DOC_IDS = _qwen3.FIXED_NEXTQA_DOC_IDS
CANONICAL_FASTVID_CONFIG = _qwen25_fastvid.CANONICAL_FASTVID_CONFIG
QWEN3_FASTVID_VIDEO_FPS = _qwen3.QWEN3_VIDEO_FPS
QWEN3_FASTVID_VIDEO_MAX_PIXELS = _qwen3.QWEN3_VIDEO_MAX_PIXELS


_conversation_for_args = _qwen3._conversation_for_args


def _conversation(
    *,
    prompt: str,
    video: Path,
    fps: float = QWEN3_FASTVID_VIDEO_FPS,
    max_pixels: int = QWEN3_FASTVID_VIDEO_MAX_PIXELS,
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
            "name": "fastvid-video-qwen3-vl-real-checkpoint",
            "placement": "projector_level",
            "execution_phase": "generate",
            "plugin": "fastvid",
            "algorithm": "fastvid",
            "variant": "fastvid.video",
            "target_modalities": ["video"],
            "conditioning_modalities": [],
            "preserved_modalities": ["text", "image", "audio"],
            "coupling_policy": "independent",
            "media_alignment": "spatiotemporal",
            "binding": "fastvid.video.qwen3_vl",
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
            "uniform post-merger temporal rasters satisfy canonical FastVID "
            "retention and DTM"
        ),
    }


def _profile() -> _qwen3.Qwen3VideoDiagnosticProfile:
    return _qwen3.Qwen3VideoDiagnosticProfile(
        algorithm="fastvid",
        variant="fastvid.video",
        binding="fastvid.video.qwen3_vl",
        canonical_config=CANONICAL_FASTVID_CONFIG,
        stage_factory=_stage,
        conversation_factory=_conversation_for_args,
        sample_validator=_validate_fastvid_sample,
        preprocessing_record=_preprocessing_record,
        projector_label="FastVID",
        reference_revision=VIDEO_REFERENCE_REVISION,
    )


def _validate_fastvid_sample(
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
    **_unused: Any,
) -> dict[str, bool]:
    if len(activation_records) != 1 or len(algorithm_records) != 1:
        raise RuntimeError(
            "FastVID must publish one layout and one algorithm record per sample"
        )
    if len(source_grid_thw) != 3:
        raise RuntimeError(f"Invalid Qwen3-VL source grid: {source_grid_thw!r}")

    temporal, height, width = (int(item) for item in source_grid_thw)
    if (
        spatial_merge_size < 1
        or height % spatial_merge_size
        or width % spatial_merge_size
    ):
        raise RuntimeError(
            "Qwen3-VL source grid is incompatible with its spatial merger"
        )
    projected_tokens_per_frame = height * width // (spatial_merge_size**2)
    raw_tokens_per_frame = height * width
    expected_source_count = temporal * projected_tokens_per_frame

    layout = activation_records[0]
    algorithm = algorithm_records[0]
    splice_layout = latest_splice["layout_delta"]
    source_count = int(layout["source_count"])
    result_count = int(layout["result_count"])
    baseline_length = int(baseline["decoder"]["prefill"]["query_length"])
    compressed_length = int(compressed["decoder"]["prefill"]["query_length"])
    frame_output_counts = tuple(int(item) for item in algorithm["frame_output_counts"])
    algorithm_representatives = tuple(
        int(item) for item in algorithm["representative_indices"]
    )
    splice_representatives = tuple(
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
    expected_result_mrope = tuple(
        source_mrope[index] for index in splice_representatives
    )
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
    decode_record = latest_generation.get("last_decode_record")
    generated_token_count = int(compressed["generated_token_count"])
    decode_query_lengths = tuple(
        int(item) for item in compressed["decoder"]["decode_query_lengths"]
    )
    removed_count = source_count - result_count

    checks = {
        "compression_applied": bool(latest_splice["compression_applied"]),
        "non_identity_token_reduction": source_count > result_count,
        "one_native_video": native_video_count == 1,
        "native_temporal_spans_preserved": (
            recorded_source_spans == native_spans
            and len(native_spans) == temporal
            and all(
                end - start == projected_tokens_per_frame for start, end in native_spans
            )
        ),
        "result_temporal_spans_exact": (
            len(result_spans) == temporal
            and tuple(end - start for start, end in result_spans) == frame_output_counts
        ),
        "grid_and_source_counts_agree": (
            source_count == expected_source_count
            and source_grid_token_count == source_count
            and int(latest_splice["source_placeholder_count"]) == source_count
            and _qwen3._span_count(recorded_source_spans) == source_count
        ),
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
            baseline_length - compressed_length == removed_count
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
        "representatives_source_ordered": (
            len(splice_representatives) == result_count
            and splice_representatives == tuple(sorted(splice_representatives))
            and len(set(splice_representatives)) == result_count
        ),
        "algorithm_and_splice_representatives_agree": (
            algorithm_representatives == splice_representatives
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
            int(latest_generation["removed_count"]) == removed_count
        ),
        "cached_decode_exact": (
            generated_token_count >= 2
            and int(latest_generation["decode_repairs"]) == generated_token_count - 1
            and decode_query_lengths == (1,) * (generated_token_count - 1)
            and _qwen25_fastvid._decode_record_coherent(
                decode_record,
                removed_count=removed_count,
                result_sequence_length=compressed_length,
            )
        ),
        "inference_labels_absent": labels_absent,
        "provider_capture_once": provider_capture_delta == 1,
        "provider_identity_exact": (
            provider_metadata["provider"] == "qwen3_vl_fastvid_video_signals"
        ),
        "provider_native_qwen_path": (
            bool(provider_metadata["video_only"])
            and bool(provider_metadata["raw_stream_attached"])
            and not bool(provider_metadata["siglip_pooling_head_loaded"])
            and provider_metadata["siglip_path"] is None
        ),
        "canonical_config_exact": (
            dict(algorithm["canonical_config"]) == CANONICAL_FASTVID_CONFIG
        ),
        "projected_signal_shape_exact": (
            tuple(algorithm["projected_features_shape"])[0] == source_count
            and len(tuple(algorithm["projected_features_shape"])) == 2
            and int(tuple(algorithm["projected_features_shape"])[1]) > 0
        ),
        "raw_signal_shape_exact": (
            tuple(algorithm["raw_features_shape"])[0] == temporal * raw_tokens_per_frame
            and int(algorithm["raw_tokens_per_frame"]) == raw_tokens_per_frame
            and len(tuple(algorithm["raw_features_shape"])) == 2
            and int(tuple(algorithm["raw_features_shape"])[1]) > 0
        ),
        "frame_global_signal_shape_exact": (
            tuple(algorithm["frame_global_features_shape"])[0] == temporal
            and len(tuple(algorithm["frame_global_features_shape"])) == 2
            and int(tuple(algorithm["frame_global_features_shape"])[1]) > 0
        ),
        "saliency_signal_shape_exact": tuple(
            algorithm["token_saliency_attention_shape"]
        )
        == (source_count,),
        "all_runtime_signals_finite": all(
            bool(algorithm[name])
            for name in (
                "projected_features_finite",
                "raw_features_finite",
                "frame_global_features_finite",
                "token_saliency_attention_finite",
            )
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
        raise RuntimeError(f"FastVID Qwen3-VL sample evidence failed: {checks!r}")
    return checks


def _parser() -> argparse.ArgumentParser:
    parser = _qwen3._parser()
    parser.set_defaults(
        video_fps=QWEN3_FASTVID_VIDEO_FPS,
        video_max_pixels=QWEN3_FASTVID_VIDEO_MAX_PIXELS,
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    return _qwen3._execute(_parser().parse_args(argv), profile=_profile())


if __name__ == "__main__":
    raise SystemExit(main())
