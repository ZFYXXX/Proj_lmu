"""Validate canonical METok video inference on Qwen3-VL.

The offline batch-one diagnostic reuses the shared Qwen3 real-checkpoint
harness and the same four fixed ``nextqa_mc_test`` documents. It requires the
local frozen SigLIP double tower, event/keyframe projector pooling, and decoder
pruning before layers 3, 10, and 19. Main and all native DeepStack projector
streams must share the projector layout; every DeepStack branch must be
consumed before decoder pruning begins. Native M-RoPE, ``visual_pos_masks``,
cache prefixes, final rope delta, and bounded cached decode remain exact.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Mapping, Sequence

from mllm_comp.compressors.video.config import VIDEO_REFERENCE_REVISION
from mllm_comp.pipeline.stage import CompressionStageConfig

if __package__:
    from scripts.diagnostics import (
        validate_qwen25_vl_metok_video_compression_checkpoint as _qwen25_metok,
    )
    from scripts.diagnostics import (
        validate_qwen3_vl_video_compression_checkpoint as _qwen3,
    )
else:
    import validate_qwen25_vl_metok_video_compression_checkpoint as _qwen25_metok
    import validate_qwen3_vl_video_compression_checkpoint as _qwen3


FIXED_NEXTQA_DOC_IDS = _qwen3.FIXED_NEXTQA_DOC_IDS
METOK_BOUNDARY_LAYERS = _qwen25_metok.METOK_BOUNDARY_LAYERS
CANONICAL_METOK_CONFIG = _qwen25_metok.CANONICAL_METOK_CONFIG
QWEN3_METOK_VIDEO_FPS = _qwen3.QWEN3_VIDEO_FPS
QWEN3_METOK_VIDEO_MAX_PIXELS = _qwen3.QWEN3_VIDEO_MAX_PIXELS


_conversation_for_args = _qwen3._conversation_for_args


def _conversation(
    *,
    prompt: str,
    video: Path,
    fps: float = QWEN3_METOK_VIDEO_FPS,
    max_pixels: int = QWEN3_METOK_VIDEO_MAX_PIXELS,
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
            "name": "metok-video-qwen3-vl-real-checkpoint",
            "placement": "projector_level",
            "execution_phase": "generate",
            "plugin": "metok",
            "algorithm": "metok",
            "variant": "metok.video",
            "target_modalities": ["video"],
            "conditioning_modalities": ["text"],
            "preserved_modalities": ["text", "image", "audio"],
            "coupling_policy": "independent",
            "media_alignment": "spatiotemporal",
            "binding": "metok.video.qwen3_vl",
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
            "native source grid remains authoritative while METok uses its "
            "post-merger raster and frozen local SigLIP-reconstructed frames"
        ),
    }


def _validate_metok_sample(
    *,
    baseline: Mapping[str, Any],
    compressed: Mapping[str, Any],
    activation_records: Sequence[Mapping[str, Any]],
    algorithm_records: Sequence[Mapping[str, Any]],
    secondary_records: Sequence[Mapping[str, Any]],
    cache_records: Sequence[Mapping[str, Any]],
    latest_splice: Mapping[str, Any],
    latest_metok: Sequence[Mapping[str, Any]] | None,
    latest_generation: Mapping[str, Any],
    provider_metadata: Mapping[str, Any],
    provider_capture_delta: int,
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
    if latest_metok is None:
        raise RuntimeError("METok Qwen3-VL bridge did not publish staged evidence")
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
    tokens_per_frame = height * width // (spatial_merge_size**2)
    native_spans = _qwen3._spans(source_placeholder_spans)
    native_temporal_spans_preserved = (
        native_video_count == 1
        and len(native_spans) == temporal
        and all(end - start == tokens_per_frame for start, end in native_spans)
        and _qwen3._span_count(native_spans) == source_grid_token_count
    )
    checks = _qwen25_metok._validate_metok_sample(
        baseline=baseline,
        compressed=compressed,
        projector_records=activation_records,
        algorithm_records=algorithm_records,
        pruning_records=secondary_records,
        cache_records=cache_records,
        latest_splice=latest_splice,
        latest_metok=latest_metok,
        latest_generation=latest_generation,
        provider_metadata=provider_metadata,
        provider_capture_delta=provider_capture_delta,
        source_grid_thw=source_grid_thw,
        source_grid_token_count=source_grid_token_count,
        spatial_merge_size=spatial_merge_size,
        second_per_grid_seconds=second_per_grid_seconds,
        labels_absent=labels_absent,
        video_span_count=len(native_spans),
        provider_name="qwen3_vl_metok_video_signals",
        backbone_label="Qwen3-VL",
        video_spans_valid=native_temporal_spans_preserved,
    )

    algorithm = algorithm_records[0]
    projector_count = int(activation_records[0]["result_count"])
    splice_deepstack_shapes = tuple(
        tuple(int(item) for item in shape)
        for shape in latest_splice["deepstack_result_shapes"]
    )
    algorithm_deepstack_shapes = tuple(
        tuple(int(item) for item in shape)
        for shape in algorithm["deepstack_result_shapes"]
    )
    bridge_deepstack_shapes = tuple(
        tuple(
            tuple(int(item) for item in shape)
            for shape in record["deepstack_projector_result_shapes"]
        )
        for record in latest_metok
    )
    staged_visual_masks_exact = all(
        bool(record["visual_pos_masks_compacted"])
        and bool(record["visual_pos_masks_compacted_exact"])
        and tuple(int(item) for item in record["source_visual_pos_mask_indices"])
        == tuple(int(item) for item in record["source_visual_sequence_indices"])
        and tuple(int(item) for item in record["result_visual_pos_mask_indices"])
        == tuple(int(item) for item in record["result_visual_sequence_indices"])
        for record in latest_metok
    )
    staged_deepstack_completion_exact = all(
        int(record["deepstack_branch_count"]) == expected_deepstack_branches
        and int(record["deepstack_consumed_branch_count"])
        == expected_deepstack_branches
        and bool(record["deepstack_completed_before_pruning"])
        and bool(record["deepstack_keep_synchronized"])
        and shapes == splice_deepstack_shapes
        for record, shapes in zip(latest_metok, bridge_deepstack_shapes)
    )
    qwen3_checks = {
        "native_temporal_spans_preserved": native_temporal_spans_preserved,
        "projector_visual_pos_masks_compacted": bool(
            latest_splice["visual_pos_masks_compacted"]
        ),
        "projector_deepstack_branch_count_exact": (
            int(latest_splice["deepstack_branch_count"])
            == int(algorithm["deepstack_branch_count"])
            == expected_deepstack_branches
            == len(splice_deepstack_shapes)
            == len(algorithm_deepstack_shapes)
        ),
        "projector_deepstack_shapes_exact": (
            splice_deepstack_shapes == algorithm_deepstack_shapes
            and all(
                len(shape) == 2 and shape[0] == projector_count
                for shape in splice_deepstack_shapes
            )
        ),
        "projector_deepstack_layout_shared": bool(
            latest_splice["shared_main_deepstack_layout"]
        ),
        "staged_visual_pos_masks_exact": staged_visual_masks_exact,
        "staged_deepstack_completion_exact": (
            len(bridge_deepstack_shapes) == len(METOK_BOUNDARY_LAYERS)
            and staged_deepstack_completion_exact
        ),
    }
    checks.update(qwen3_checks)
    if not all(checks.values()):
        raise RuntimeError(f"METok Qwen3-VL sample evidence failed: {checks!r}")
    return checks


def _profile() -> _qwen3.Qwen3VideoDiagnosticProfile:
    return _qwen3.Qwen3VideoDiagnosticProfile(
        algorithm="metok",
        variant="metok.video",
        binding="metok.video.qwen3_vl",
        canonical_config=CANONICAL_METOK_CONFIG,
        stage_factory=_stage,
        conversation_factory=_conversation_for_args,
        sample_validator=_validate_metok_sample,
        preprocessing_record=_preprocessing_record,
        projector_label="METok",
        reference_revision=VIDEO_REFERENCE_REVISION,
        post_boundary_layers=METOK_BOUNDARY_LAYERS,
        auxiliary_checkpoints=(("siglip_checkpoint", "metok_siglip_path"),),
        secondary_layout_key="pruning_layout_deltas",
    )


def _parser() -> argparse.ArgumentParser:
    parser = _qwen3._parser()
    parser.add_argument("--siglip_checkpoint", type=Path, required=True)
    parser.set_defaults(
        video_fps=QWEN3_METOK_VIDEO_FPS,
        video_max_pixels=QWEN3_METOK_VIDEO_MAX_PIXELS,
        estimated_peak_mib=49152,
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.max_new_tokens != 4:
        raise RuntimeError("METok fixed evidence requires max_new_tokens=4")
    return _qwen3._execute(args, profile=_profile())


if __name__ == "__main__":
    raise SystemExit(main())
