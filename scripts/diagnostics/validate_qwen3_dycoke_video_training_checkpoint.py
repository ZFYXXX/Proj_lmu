"""Validate the exact DyCoKe/Qwen3-VL three-step training cell.

This profile reuses the shared DyCoKe batch-one lifecycle while owning the
Qwen3-VL legalization checks: sparse M-RoPE, compact ``visual_pos_masks``,
and identical source-group replay on all three DeepStack branches.  Training
forwards never install the generation-only sparse visual-KV controller.
"""

from __future__ import annotations

import argparse
import math
from functools import partial
from pathlib import Path
from typing import Any, Mapping, Sequence

from mllm_comp.backbones.base import BackboneRuntime
from mllm_comp.backbones.qwen3_vl import Qwen3VLSample
from mllm_comp.pipeline.stage import CompressionStageConfig
from scripts.diagnostics import qwen_video_checkpoint_common as qwen_training
from scripts.diagnostics import validate_qwen3_vl_dycoke_video_compression_checkpoint as inference
from scripts.diagnostics import validate_qwen25_dycoke_video_training_checkpoint as shared
from scripts.diagnostics.qwen_video_checkpoint_common import valid_spans as _spans
from scripts.diagnostics.video_checkpoint_common import _gpu_snapshot, _tensor_rows
from scripts.diagnostics.video_training_checkpoint_common import (
    training_activation,
    training_main,
    training_parser,
    training_state,
    video_training_fixture,
)

CANONICAL_DYCOKE_CONFIG = inference.CANONICAL_DYCOKE_CONFIG
FIXED_NEXTQA_DOC_IDS = inference.FIXED_NEXTQA_DOC_IDS
SCOPE = {
    "algorithm": "dycoke",
    "variant": "dycoke.video",
    "backbone": "qwen3_vl",
    "binding": "dycoke.video.qwen3_vl",
}


def _stage() -> CompressionStageConfig:
    return CompressionStageConfig.from_dict(
        {
            "name": "dycoke-video-qwen3-vl-three-step-train",
            "placement": "projector_level",
            "execution_phase": "forward",
            "plugin": "dycoke",
            "algorithm": SCOPE["algorithm"],
            "variant": SCOPE["variant"],
            "binding": SCOPE["binding"],
            "target_modalities": ["video"],
            "conditioning_modalities": ["text"],
            "preserved_modalities": ["text", "image", "audio"],
            "coupling_policy": "independent",
            "media_alignment": "spatiotemporal",
            "config": {},
        }
    )


def _load_runtime(
    checkpoint: Path, *, torch: Any, train: bool
) -> tuple[Any, Any, Any, BackboneRuntime]:
    return qwen_training.load_qwen3_runtime(checkpoint, torch=torch, train=train)


_activation = partial(training_activation, canonical_config=CANONICAL_DYCOKE_CONFIG)


def _grid_record(state: Any, batch: Any) -> dict[str, Any]:
    rows = _tensor_rows(state.video_grid_thw)
    if len(rows) != 1 or len(rows[0]) != 3:
        raise RuntimeError(f"Expected one video_grid_thw row, got {rows!r}")
    merge_size = int(batch.metadata["spatial_merge_size"])
    temporal, height, width = rows[0]
    merge_area = merge_size**2
    if merge_size < 1 or (height * width) % merge_area:
        raise RuntimeError("Qwen3-VL grid is not divisible by spatial merge size")
    per_slice = height * width // merge_area
    if temporal < 1 or per_slice < 1:
        raise RuntimeError("Qwen3-VL video grid is empty")
    timing = state.native_model_inputs.get("second_per_grid_ts")
    timing_values = (
        [] if timing is None else qwen_training.float_values(timing)
    )
    if any(not math.isfinite(value) or value <= 0 for value in timing_values):
        raise RuntimeError(f"Invalid optional Qwen3-VL timing: {timing_values!r}")
    return {
        "video_grid_thw": rows[0],
        "spatial_merge_size": merge_size,
        "temporal_slice_count": temporal,
        "per_temporal_slice_source_count": per_slice,
        "source_grid_token_count": temporal * per_slice,
        "second_per_grid_ts": timing_values,
    }


def _fixture(
    *, document: Mapping[str, Any], video: Path, output_dir: Path
) -> tuple[dict[str, Any], Path]:
    return video_training_fixture(
        document=document, video=video, output_dir=output_dir,
        preprocessing={
            'fps': inference.QWEN3_DYCOKE_VIDEO_FPS,
            'max_pixels': inference.QWEN3_DYCOKE_VIDEO_MAX_PIXELS,
            'enable_thinking': False,
        },
    )


def _timing_preserved(
    splice: Mapping[str, Any] | None,
    timing: Sequence[float],
) -> bool:
    if splice is None:
        return False
    if not timing:
        return splice.get("second_per_grid_seconds") is None
    return len(timing) == 1 and inference._timing_matches(
        float(timing[0]), splice.get("second_per_grid_seconds")
    )


def _layout_checks(
    *,
    logits: Any,
    state: Any,
    grid: Mapping[str, Any],
    records: Sequence[Mapping[str, Any]],
    algorithm_records: Sequence[Mapping[str, Any]],
    splice: Mapping[str, Any] | None,
    metadata: Mapping[str, Any],
    training_labels: bool,
    torch: Any,
) -> dict[str, bool]:
    source_count = int(records[0].get("source_count", 0)) if len(records) == 1 else 0
    result_count = int(records[0].get("result_count", 0)) if len(records) == 1 else 0
    algorithm = algorithm_records[0] if len(algorithm_records) == 1 else None
    algorithm_map = algorithm if isinstance(algorithm, Mapping) else {}
    temporal = int(grid["temporal_slice_count"])
    per_slice = int(grid["per_temporal_slice_source_count"])
    expected_frame_counts = shared.inference._expected_frame_result_counts(
        temporal, per_slice
    )
    source_spans = _spans(
        () if splice is None else splice.get("source_placeholder_spans", ())
    )
    result_spans = _spans(
        () if splice is None else splice.get("result_placeholder_spans", ())
    )
    native_spans = tuple(
        (int(span.start), int(span.end))
        for span in state.modality_token_spans
        if span.modality == "video"
    )
    representatives = tuple(
        int(value)
        for value in (
            () if splice is None else splice.get("representative_indices", ())
        )
    )
    source_mrope = tuple(
        () if splice is None else splice.get("source_visual_mrope", ())
    )
    result_mrope = tuple(
        () if splice is None else splice.get("result_visual_mrope", ())
    )
    frame_records_by_video = tuple(algorithm_map.get("frame_records_by_video", ()))
    frame_records = (
        () if len(frame_records_by_video) != 1 else tuple(frame_records_by_video[0])
    )
    selected_sources = tuple(
        int(value)
        for frame in frame_records
        for value in frame.get("selected_source_indices", ())
    )
    source_deepstack_shapes = tuple(
        tuple(int(item) for item in shape)
        for shape in algorithm_map.get("deepstack_source_shapes", ())
    )
    algorithm_deepstack_shapes = tuple(
        tuple(int(item) for item in shape)
        for shape in algorithm_map.get("deepstack_result_shapes", ())
    )
    splice_deepstack_shapes = tuple(
        tuple(int(item) for item in shape)
        for shape in (
            () if splice is None else splice.get("deepstack_result_shapes", ())
        )
    )
    result_positions = tuple(
        position for start, end in result_spans for position in range(start, end)
    )
    layout_positions = (
        ()
        if len(records) != 1
        else tuple(
            int(value) for value in records[0].get("result_visual_positions", ())
        )
    )
    algorithm_positions = tuple(
        int(value) for value in algorithm_map.get("result_visual_positions", ())
    )
    labels = state.labels
    labels_absent = labels is None and (
        "labels" not in state.native_model_inputs
        or state.native_model_inputs["labels"] is None
    )
    source_length = int(state.input_ids.shape[1])
    expected_result_length = source_length - source_count + result_count
    mask = state.attention_mask
    splice_layout = {} if splice is None else splice.get("layout_delta", {})
    return {
        "one_compression_invocation": (
            len(records) == 1 and len(algorithm_records) == 1
        ),
        "compression_applied": bool(
            splice is not None and splice.get("compression_applied")
        ),
        "non_identity_token_reduction": source_count > result_count > 0,
        "native_temporal_spans_preserved": (
            source_spans == native_spans
            and len(native_spans) == temporal
            and all(end - start == per_slice for start, end in native_spans)
        ),
        "result_temporal_spans_exact": (
            len(result_spans) == temporal
            and tuple(end - start for start, end in result_spans)
            == expected_frame_counts
            and sum(expected_frame_counts) == result_count
        ),
        "placeholder_counts_exact": bool(
            splice is not None
            and int(splice.get("source_placeholder_count", 0)) == source_count
            and int(splice.get("result_placeholder_count", 0)) == result_count
            and qwen_training.span_count(source_spans) == source_count
            and qwen_training.span_count(result_spans) == result_count
        ),
        "layout_records_agree": bool(
            splice is not None
            and int(splice_layout.get("source_count", 0)) == source_count
            and int(splice_layout.get("result_count", 0)) == result_count
            and int(algorithm_map.get("source_count", 0)) == source_count
            and int(algorithm_map.get("result_count", 0)) == result_count
        ),
        "source_grid_count_exact": source_count == int(grid["source_grid_token_count"]),
        "source_video_grid_preserved": bool(
            splice is not None
            and splice.get("source_video_grid_preserved")
            and tuple(int(item) for item in splice.get("source_video_grid_thw", ()))
            == tuple(int(item) for item in grid["video_grid_thw"])
        ),
        "source_video_timing_preserved": _timing_preserved(
            splice, grid["second_per_grid_ts"]
        ),
        "sparse_grid_policy": bool(
            splice is not None
            and splice.get("result_grid_policy") == "irregular_sparse_source_anchors"
        ),
        "representatives_match_ttm_source_selection": (
            len(representatives) == result_count
            and representatives == selected_sources
            and representatives == tuple(sorted(set(representatives)))
        ),
        "mrope_source_count_aligned": len(source_mrope) == source_count,
        "mrope_result_count_aligned": len(result_mrope) == result_count,
        "native_mrope_gather_exact": bool(
            splice is not None and qwen_training.mrope_gather_exact(splice)
        ),
        "native_position_plane_count_exact": bool(
            splice is not None
            and int(splice.get("source_position_plane_count", 0)) in (3, 4)
            and int(splice.get("result_position_plane_count", 0))
            == int(splice.get("source_position_plane_count", -1))
        ),
        "rope_delta_present": bool(splice is not None and splice.get("rope_deltas")),
        "visual_pos_masks_compacted": bool(
            splice is not None
            and splice.get("visual_pos_masks_compacted")
            and int(splice.get("visual_pos_masks_result_count", 0)) == result_count
        ),
        "main_and_three_deepstack_layouts_exact": bool(
            splice is not None
            and int(splice.get("deepstack_branch_count", 0)) == 3
            and int(algorithm_map.get("deepstack_branch_count", 0)) == 3
            and len(source_deepstack_shapes) == 3
            and all(
                len(shape) == 2 and shape[0] == source_count
                for shape in source_deepstack_shapes
            )
            and algorithm_deepstack_shapes == splice_deepstack_shapes
            and len(splice_deepstack_shapes) == 3
            and all(
                len(shape) == 2 and shape[0] == result_count
                for shape in splice_deepstack_shapes
            )
            and splice.get("shared_main_deepstack_layout")
            and int(splice.get("deepstack_group_count", 0)) == result_count
        ),
        "attention_position_cache_compacted": bool(
            splice is not None
            and splice.get("attention_mask_compacted")
            and splice.get("position_ids_compacted")
            and splice.get("cache_position_dense")
        ),
        "result_visual_positions_exact": (
            len(result_positions) == result_count
            and layout_positions == result_positions
            and algorithm_positions == result_positions
        ),
        "source_attention_mask_aligned": bool(
            mask is not None and tuple(mask.shape) == tuple(state.input_ids.shape)
        ),
        "training_completion_labels_compacted": bool(
            not training_labels
            or (
                splice is not None
                and splice.get("labels_present")
                and splice.get("training_labels_compacted")
                and splice.get("visual_labels_ignored")
                and qwen_training.completion_only_labels_exact(state, splice)
            )
        ),
        "eval_labels_absent": bool(training_labels or labels_absent),
        "training_use_cache_false": state.native_model_inputs.get("use_cache") is False,
        "finite_logits": bool(
            logits is not None and torch.isfinite(logits).all().item()
        ),
        "label_logit_layout_aligned": bool(
            logits is not None
            and int(logits.shape[0]) == 1
            and int(logits.shape[1]) == expected_result_length
        ),
        **shared._dycoke_algorithm_checks(
            algorithm,
            source_count=source_count,
            result_count=result_count,
            grid=grid,
        ),
        **shared._forward_runtime_checks(metadata),
    }


def _training_conversation(*, prompt: str, video: Path) -> Any:
    return inference._conversation(
        prompt=prompt,
        video=video,
        fps=inference.QWEN3_DYCOKE_VIDEO_FPS,
        max_pixels=inference.QWEN3_DYCOKE_VIDEO_MAX_PIXELS,
    )


def _training_sample(*, sample_id: str, conversation: Any) -> Qwen3VLSample:
    return Qwen3VLSample(
        sample_id=sample_id,
        conversation=conversation,
        enable_thinking=False,
    )


def _training_profile(runtime_loader: Any) -> shared.DyCoKeTrainingProfile:
    return shared.DyCoKeTrainingProfile(
        scope=SCOPE,
        canonical_config=CANONICAL_DYCOKE_CONFIG,
        fixed_nextqa_doc_ids=FIXED_NEXTQA_DOC_IDS,
        framework_label="Qwen3-VL",
        stage_factory=_stage,
        conversation_factory=_training_conversation,
        sample_factory=_training_sample,
        runtime_loader=runtime_loader,
        training_state_factory=training_state,
        grid_record_factory=_grid_record,
        activation_factory=_activation,
        fixture_factory=_fixture,
        layout_checker=_layout_checks,
        estimate_basis=(
            "DyCoKe Qwen3-VL generation peaked at 5.109/6.012 GiB "
            "allocated/reserved; the 32 GiB bound plus a separate 16 GiB "
            "margin covers final-norm autograd, three DeepStack branches, "
            "fresh-checkpoint reload, and allocator variance"
        ),
    )


def _deepstack_handles(facade: Any, model: Any) -> tuple[str, ...]:
    handles = facade.model_adapter.structural_handles(model)
    return tuple(
        sorted(
            name for name in handles if name.startswith("vision_deepstack_projector[")
        )
    )


def _run(args: argparse.Namespace) -> dict[str, Any]:
    handle_snapshots: list[tuple[str, ...]] = []

    def tracked_runtime_loader(
        checkpoint: Path, *, torch: Any, train: bool
    ) -> tuple[Any, Any, Any, BackboneRuntime]:
        result = _load_runtime(checkpoint, torch=torch, train=train)
        handles = _deepstack_handles(result[0], result[2])
        if len(handles) != 3:
            raise RuntimeError(
                "The canonical Qwen3-VL checkpoint must expose exactly three "
                f"DeepStack projectors, got {handles!r}"
            )
        handle_snapshots.append(handles)
        return result

    report = shared._run(
        args,
        profile=_training_profile(tracked_runtime_loader),
    )
    if len(handle_snapshots) != 2 or handle_snapshots[0] != handle_snapshots[1]:
        raise RuntimeError(
            "Post-reload Qwen3-VL DeepStack structural handles changed: "
            f"{handle_snapshots!r}"
        )
    report["post_reload_compressed_inference"]["checks"][
        "deepstack_handles_exact"
    ] = True
    report["deepstack_projector_handles"] = list(handle_snapshots[0])
    report["training"].update(
        {
            "deepstack_branch_count": 3,
            "defining_stages": [
                "differentiable_two_pass_ttm_gather",
                "three_branch_deepstack_source_group_replay",
                "generation_sparse_visual_kv_owned_by_generate_unit",
            ],
        }
    )
    return report


def _parser() -> argparse.ArgumentParser:
    parser = training_parser(estimated_peak_mib=32768)
    return parser


def main(argv: list[str] | None = None) -> int:
    return training_main(argv, parser=_parser(), run=_run, scope=SCOPE, gpu_snapshot=_gpu_snapshot)


if __name__ == "__main__":
    raise SystemExit(main())
