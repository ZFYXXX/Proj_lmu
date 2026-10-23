"""Validate the exact METok/Qwen3-VL five-step training cell.

This offline, batch-one evidence harness trains only the facade-owned final
normalization weight on the shared doc-442 test-derived fixture. Every forward
must execute event-aware projector pooling and progressive visual pruning
before decoder layers 3, 10, and 19. Training deliberately uses
``use_cache=False``; its three boundary records must therefore prove that no
DynamicCache was created or compacted. The local SigLIP vision/text capability
stays frozen and outside both optimizer and lightweight-checkpoint ownership.
The native source grid and timing remain unchanged while projector and decoder
layouts gather their exact three-axis M-RoPE anchors.
"""

from __future__ import annotations

import argparse
import gc
import math
import time
from functools import partial
from pathlib import Path
from typing import Any, Mapping, Sequence

from mllm_comp.backbones.base import BackboneRuntime
from mllm_comp.backbones.qwen3_vl import Qwen3VLSample, create_qwen3_vl_backbone
from mllm_comp.pipeline.stage import CompressionStageConfig
from scripts.diagnostics import qwen_video_checkpoint_common as qwen_training
from scripts.diagnostics import validate_qwen3_vl_metok_video_compression_checkpoint as inference
from scripts.diagnostics import (
    validate_qwen25_vl_metok_video_compression_checkpoint as qwen25_inference,
)
from scripts.diagnostics.qwen_video_checkpoint_common import valid_spans as _spans
from scripts.diagnostics.video_checkpoint_common import (
    prepare_checkpoint_environment,
    checkpoint_gpu_report,
    _checkpoint_record,
    _gpu_snapshot,
    _tensor_rows,
)
from scripts.diagnostics.video_training_checkpoint_common import (
    training_report,
    FIXTURE_DOC_ID,
    OPTIMIZER_STEPS,
    append_jsonl,
    canonical,
    checkpoint_saved,
    final_norm_optimizer,
    inference_forward,
    inference_state,
    load_lightweight_checkpoint,
    load_training_runtime,
    prepare_training_device,
    print_training_step,
    record_post_reload,
    record_training_step,
    save_training_checkpoint,
    target_parameter_checks,
    training_activation,
    training_cache_exact as _training_cache_exact,
    training_document,
    training_environment_report,
    training_forward_step,
    training_main,
    training_parameter_report,
    training_parser,
    training_state as _training_state,
    video_training_fixture,
)

CANONICAL_METOK_CONFIG = inference.CANONICAL_METOK_CONFIG
METOK_BOUNDARY_LAYERS = inference.METOK_BOUNDARY_LAYERS
FIXED_NEXTQA_DOC_IDS = inference.FIXED_NEXTQA_DOC_IDS
SCOPE = {
    "algorithm": "metok",
    "variant": "metok.video",
    "backbone": "qwen3_vl",
    "binding": "metok.video.qwen3_vl",
}


def _stage() -> CompressionStageConfig:
    return CompressionStageConfig.from_dict(
        {
            "name": "metok-video-qwen3-vl-five-step-train",
            "placement": "projector_level",
            "execution_phase": "forward",
            "plugin": "metok",
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
    checkpoint: Path,
    siglip_checkpoint: Path,
    *,
    torch: Any,
    train: bool,
) -> tuple[Any, Any, Any, BackboneRuntime]:
    facade, processor, model, runtime = load_training_runtime(
        checkpoint,
        facade=create_qwen3_vl_backbone(),
        torch=torch,
        train=train,
        device_map={"": 0},
    )
    runtime.metadata["metok_siglip_path"] = str(siglip_checkpoint)
    return facade, processor, model, runtime


_activation = partial(training_activation, canonical_config=CANONICAL_METOK_CONFIG)


def _grid_record(state: Any, batch: Any) -> dict[str, Any]:
    rows = _tensor_rows(state.video_grid_thw)
    if len(rows) != 1 or len(rows[0]) != 3:
        raise RuntimeError(f"Expected one video_grid_thw row, got {rows!r}")
    merge_size = int(batch.metadata["spatial_merge_size"])
    temporal, height, width = rows[0]
    if merge_size < 1 or height % merge_size or width % merge_size:
        raise RuntimeError("Qwen3-VL source grid is incompatible with its merger")
    grid_height = height // merge_size
    grid_width = width // merge_size
    per_slice = grid_height * grid_width
    if temporal < CANONICAL_METOK_CONFIG["num_segments"] or per_slice < 4:
        raise RuntimeError(
            "Frozen Qwen3-VL raster cannot exercise canonical METok: "
            f"grid={rows[0]!r}, per_slice={per_slice}"
        )
    timing_value = state.native_model_inputs.get("second_per_grid_ts")
    timing = (
        []
        if timing_value is None
        else qwen_training.float_values(timing_value)
    )
    if any(not math.isfinite(value) or value <= 0 for value in timing):
        raise RuntimeError(f"Invalid optional Qwen3-VL timing values: {timing!r}")
    return {
        "video_grid_thw": rows[0],
        "spatial_merge_size": merge_size,
        "temporal_slice_count": temporal,
        "grid_height": grid_height,
        "grid_width": grid_width,
        "per_temporal_slice_source_count": per_slice,
        "raw_tokens_per_temporal_slice": height * width,
        "source_grid_token_count": temporal * per_slice,
        "second_per_grid_ts": timing,
    }


def _fixture(
    *, document: Mapping[str, Any], video: Path, output_dir: Path
) -> tuple[dict[str, Any], Path]:
    return video_training_fixture(
        document=document, video=video, output_dir=output_dir,
        preprocessing={
            'fps': inference.QWEN3_METOK_VIDEO_FPS,
            'max_pixels': inference.QWEN3_METOK_VIDEO_MAX_PIXELS,
            'enable_thinking': False,
        },
    )


def _training_pruning_chain_exact(
    pruning_records: Sequence[Mapping[str, Any]],
    cache_records: Sequence[Mapping[str, Any]],
    latest_metok: Sequence[Mapping[str, Any]],
    *,
    projector_count: int,
    expected_deepstack_shapes: Sequence[Sequence[int]],
) -> bool:
    if not (
        len(pruning_records)
        == len(cache_records)
        == len(latest_metok)
        == len(METOK_BOUNDARY_LAYERS)
    ):
        return False
    prior_visual = projector_count

    prior_sequence_result: int | None = None
    prior_result_visual_mrope: tuple[tuple[int, ...], ...] | None = None
    deepstack_shapes = tuple(
        tuple(int(item) for item in shape) for shape in expected_deepstack_shapes
    )
    if len(deepstack_shapes) != 3:
        return False
    for stage_index, boundary_layer in enumerate(METOK_BOUNDARY_LAYERS):
        record = pruning_records[stage_index]
        cache = cache_records[stage_index]
        bridge = latest_metok[stage_index]
        sequence = record.get("sequence_layout_delta", {})
        source_visual = int(record.get("source_count", -1))
        result_visual = int(record.get("result_count", -1))
        source_sequence = int(sequence.get("source_count", -1))
        result_sequence = int(sequence.get("result_count", -1))
        kept = tuple(int(item) for item in record.get("kept_visual_indices", ()))
        dropped = tuple(int(item) for item in record.get("dropped_visual_indices", ()))
        retained = tuple(
            int(item)
            for item in record.get("cumulative_retained_projector_indices", ())
        )
        cumulative_prompt = tuple(
            int(item) for item in record.get("cumulative_prompt_keep_indices", ())
        )
        source_visual_positions = tuple(
            int(item) for item in bridge.get("source_visual_sequence_indices", ())
        )
        result_visual_positions = tuple(
            int(item) for item in bridge.get("result_visual_sequence_indices", ())
        )
        source_mask_positions = tuple(
            int(item) for item in bridge.get("source_visual_pos_mask_indices", ())
        )
        result_mask_positions = tuple(
            int(item) for item in bridge.get("result_visual_pos_mask_indices", ())
        )
        bridge_deepstack_shapes = tuple(
            tuple(int(item) for item in shape)
            for shape in bridge.get("deepstack_projector_result_shapes", ())
        )
        source_mrope = tuple(
            tuple(int(value) for value in row)
            for row in bridge.get("source_visual_mrope", ())
        )
        result_mrope = tuple(
            tuple(int(value) for value in row)
            for row in bridge.get("result_visual_mrope", ())
        )
        expected_result_mrope = tuple(source_mrope[index] for index in kept)
        plane_count = len(source_mrope[0]) if source_mrope else 0
        expected_rope_delta = int(bridge.get("expected_rope_delta", 2**31))
        rope_deltas = tuple(int(item) for item in bridge.get("rope_deltas", ()))
        exact = (
            int(record.get("boundary_layer", -1)) == boundary_layer
            and int(record.get("stage_index", -1)) == stage_index
            and source_visual == prior_visual
            and result_visual == len(kept)
            and tuple(sorted((*kept, *dropped))) == tuple(range(source_visual))
            and len(retained) == result_visual
            and retained == tuple(sorted(set(retained)))
            and source_sequence - result_sequence == source_visual - result_visual
            and len(tuple(record.get("kept_sequence_indices", ()))) == result_sequence
            and len(cumulative_prompt) == result_sequence
            and bool(record.get("final_visual_removal"))
            is (stage_index == len(METOK_BOUNDARY_LAYERS) - 1)
            and dict(record.get("cache", {})) == dict(cache)
            and _training_cache_exact(
                cache,
                boundary_layer=boundary_layer,
                source_length=source_sequence,
                result_length=result_sequence,
            )
            and int(bridge.get("boundary_layer", -1)) == boundary_layer
            and int(bridge.get("stage_index", -1)) == stage_index
            and int(bridge.get("source_visual_count", -1)) == source_visual
            and int(bridge.get("result_visual_count", -1)) == result_visual
            and int(bridge.get("source_sequence_length", -1)) == source_sequence
            and int(bridge.get("result_sequence_length", -1)) == result_sequence
            and tuple(
                int(item) for item in bridge.get("cumulative_prompt_keep_indices", ())
            )
            == cumulative_prompt
            and len(source_visual_positions) == source_visual
            and len(result_visual_positions) == result_visual
            and source_visual_positions == tuple(sorted(set(source_visual_positions)))
            and result_visual_positions == tuple(sorted(set(result_visual_positions)))
            and len(source_mrope) == source_visual
            and len(result_mrope) == result_visual
            and plane_count in (3, 4)
            and all(len(row) == plane_count for row in (*source_mrope, *result_mrope))
            and result_mrope == expected_result_mrope
            and source_mask_positions == source_visual_positions
            and result_mask_positions == result_visual_positions
            and bool(bridge.get("visual_pos_masks_compacted"))
            and bool(bridge.get("visual_pos_masks_compacted_exact"))
            and int(bridge.get("deepstack_branch_count", -1)) == 3
            and int(bridge.get("deepstack_consumed_branch_count", -1)) == 3
            and bool(bridge.get("deepstack_completed_before_pruning"))
            and bool(bridge.get("deepstack_keep_synchronized"))
            and bridge_deepstack_shapes == deepstack_shapes
            and bool(bridge.get("rope_delta_exact"))
            and int(bridge.get("active_position_count", -1)) == result_sequence
            and expected_rope_delta
            == int(bridge.get("max_active_rotary_position", -1)) + 1 - result_sequence
            and rope_deltas == (expected_rope_delta,)
        )
        if not exact:
            return False
        if (
            prior_sequence_result is not None
            and source_sequence != prior_sequence_result
        ):
            return False
        if (
            prior_result_visual_mrope is not None
            and source_mrope != prior_result_visual_mrope
        ):
            return False
        prior_visual = result_visual

        prior_sequence_result = result_sequence
        prior_result_visual_mrope = result_mrope
    return prior_visual == 0


def _metok_algorithm_checks(
    *,
    projector: Mapping[str, Any] | None,
    algorithm: Mapping[str, Any] | None,
    pruning_records: Sequence[Mapping[str, Any]],
    cache_records: Sequence[Mapping[str, Any]],
    splice: Mapping[str, Any] | None,
    latest_metok: Sequence[Mapping[str, Any]],
    provider: Mapping[str, Any] | None,
    provider_capture_delta: int,
    grid: Mapping[str, Any],
) -> dict[str, bool]:
    projector = {} if projector is None else projector
    algorithm = {} if algorithm is None else algorithm
    splice = {} if splice is None else splice
    provider = {} if provider is None else provider
    source_count = int(projector.get("source_count", 0))
    projector_count = int(projector.get("result_count", 0))
    temporal = int(grid["temporal_slice_count"])
    grid_height = int(grid["grid_height"])
    grid_width = int(grid["grid_width"])
    tokens_per_frame = int(grid["per_temporal_slice_source_count"])
    raw_tokens_per_frame = int(grid["raw_tokens_per_temporal_slice"])
    expected_source = int(grid["source_grid_token_count"])
    expected_grid = tuple(int(item) for item in grid["video_grid_thw"])
    expected_timing = tuple(float(item) for item in grid["second_per_grid_ts"])
    splice_representatives = tuple(
        int(item) for item in splice.get("representative_indices", ())
    )
    source_mrope = tuple(
        tuple(int(value) for value in row)
        for row in splice.get("source_visual_mrope", ())
    )
    result_mrope = tuple(
        tuple(int(value) for value in row)
        for row in splice.get("result_visual_mrope", ())
    )
    expected_projector_mrope = (
        tuple(source_mrope[index] for index in splice_representatives)
        if source_mrope and splice_representatives
        else ()
    )
    plane_count = int(splice.get("source_position_plane_count", -1))
    algorithm_deepstack_source_shapes = tuple(
        tuple(int(item) for item in shape)
        for shape in algorithm.get("deepstack_source_shapes", ())
    )
    algorithm_deepstack_result_shapes = tuple(
        tuple(int(item) for item in shape)
        for shape in algorithm.get("deepstack_result_shapes", ())
    )
    splice_deepstack_result_shapes = tuple(
        tuple(int(item) for item in shape)
        for shape in splice.get("deepstack_result_shapes", ())
    )
    timing_preserved = splice.get("second_per_grid_seconds") is None
    if expected_timing:
        try:
            observed_timing = float(splice.get("second_per_grid_seconds"))
        except (TypeError, ValueError):
            observed_timing = float("nan")
        timing_preserved = (
            len(expected_timing) == 1
            and math.isfinite(observed_timing)
            and math.isclose(
                observed_timing,
                expected_timing[0],
                rel_tol=0.0,
                abs_tol=1e-9,
            )
        )
    pruning_exact = _training_pruning_chain_exact(
        pruning_records,
        cache_records,
        latest_metok,
        projector_count=projector_count,
        expected_deepstack_shapes=algorithm_deepstack_result_shapes,
    )
    final_rope_deltas = (
        tuple(int(item) for item in latest_metok[-1].get("rope_deltas", ()))
        if latest_metok
        else ()
    )
    staged_visual_masks_exact = bool(latest_metok) and all(
        bool(record.get("visual_pos_masks_compacted"))
        and bool(record.get("visual_pos_masks_compacted_exact"))
        and tuple(
            int(item) for item in record.get("source_visual_pos_mask_indices", ())
        )
        == tuple(int(item) for item in record.get("source_visual_sequence_indices", ()))
        and tuple(
            int(item) for item in record.get("result_visual_pos_mask_indices", ())
        )
        == tuple(int(item) for item in record.get("result_visual_sequence_indices", ()))
        for record in latest_metok
    )
    staged_deepstack_completion_exact = bool(latest_metok) and all(
        int(record.get("deepstack_branch_count", -1)) == 3
        and int(record.get("deepstack_consumed_branch_count", -1)) == 3
        and bool(record.get("deepstack_completed_before_pruning"))
        and bool(record.get("deepstack_keep_synchronized"))
        and tuple(
            tuple(int(item) for item in shape)
            for shape in record.get("deepstack_projector_result_shapes", ())
        )
        == algorithm_deepstack_result_shapes
        for record in latest_metok
    )
    return {
        "canonical_source_raster_exact": (
            source_count == expected_source
            and int(algorithm.get("frame_count", 0)) == temporal
            and int(algorithm.get("grid_height", 0)) == grid_height
            and int(algorithm.get("grid_width", 0)) == grid_width
            and int(algorithm.get("tokens_per_frame", 0)) == tokens_per_frame
        ),
        "projector_non_identity_reduction": source_count > projector_count > 0,
        "projector_layout_and_splice_counts_agree": (
            int(algorithm.get("source_count", 0)) == source_count
            and int(algorithm.get("result_count", 0)) == projector_count
            and int(splice.get("source_placeholder_count", 0)) == source_count
            and int(splice.get("result_placeholder_count", 0)) == projector_count
            and int(splice.get("layout_delta", {}).get("source_count", 0))
            == source_count
            and int(splice.get("layout_delta", {}).get("result_count", 0))
            == projector_count
        ),
        "source_video_grid_preserved": (
            bool(splice.get("source_video_grid_preserved"))
            and tuple(int(item) for item in splice.get("source_video_grid_thw", ()))
            == expected_grid
        ),
        "source_video_timing_preserved": timing_preserved,
        "sparse_grid_policy": (
            splice.get("result_grid_policy") == "irregular_sparse_source_anchors"
        ),
        "canonical_config_exact": (
            qwen25_inference._metok_common._canonical_config_exact(
                algorithm.get("canonical_config", {})
            )
        ),
        "projected_and_raw_signals_exact": (
            tuple(algorithm.get("projected_features_shape", ()))[:1] == (source_count,)
            and tuple(algorithm.get("result_features_shape", ()))[:1]
            == (projector_count,)
            and tuple(algorithm.get("raw_features_shape", ()))[:1]
            == (temporal * raw_tokens_per_frame,)
            and int(algorithm.get("raw_tokens_per_frame", 0)) == raw_tokens_per_frame
            and bool(algorithm.get("projected_features_finite"))
            and bool(algorithm.get("result_features_finite"))
            and bool(algorithm.get("raw_features_finite"))
            and bool(algorithm.get("frame_text_similarity_finite"))
            and len(tuple(algorithm.get("frame_text_similarities", ()))) == temporal
        ),
        "projector_mrope_sparse_anchor_gather_exact": (
            len(source_mrope) == source_count
            and len(result_mrope) == projector_count
            and plane_count in (3, 4)
            and int(splice.get("result_position_plane_count", -1)) == plane_count
            and all(len(row) == plane_count for row in (*source_mrope, *result_mrope))
            and result_mrope == expected_projector_mrope
        ),
        "projector_visual_pos_masks_exact": (
            bool(splice.get("visual_pos_masks_compacted"))
            and int(splice.get("visual_pos_masks_result_count", 0)) == projector_count
        ),
        "projector_deepstack_layout_exact": (
            int(algorithm.get("deepstack_branch_count", 0))
            == len(algorithm_deepstack_source_shapes)
            == len(algorithm_deepstack_result_shapes)
            == len(splice_deepstack_result_shapes)
            == 3
            and all(
                len(shape) == 2 and shape[0] == source_count
                for shape in algorithm_deepstack_source_shapes
            )
            and all(
                len(shape) == 2 and shape[0] == projector_count
                for shape in algorithm_deepstack_result_shapes
            )
            and algorithm_deepstack_result_shapes == splice_deepstack_result_shapes
            and bool(splice.get("shared_main_deepstack_layout"))
            and int(splice.get("deepstack_group_count", 0)) == projector_count
        ),
        "staged_visual_pos_masks_exact": staged_visual_masks_exact,
        "staged_deepstack_completion_exact": staged_deepstack_completion_exact,
        "local_frozen_siglip_provider_exact": (
            algorithm.get("similarity_source")
            == "qwen_patchified_video_inverse_siglip_384"
            and bool(algorithm.get("assistant_answer_excluded"))
            and bool(algorithm.get("siglip_local_only"))
            and provider_capture_delta == 1
            and provider.get("provider") == "qwen3_vl_metok_video_signals"
            and bool(provider.get("installed"))
            and bool(provider.get("video_only"))
            and bool(provider.get("raw_stream_attached"))
            and bool(provider.get("siglip_loaded"))
            and provider.get("siglip_vision_enabled") is True
            and bool(provider.get("siglip_frozen"))
            and bool(provider.get("siglip_local_only"))
            and bool(provider.get("assistant_answer_excluded"))
            and bool(provider.get("siglip_path"))
        ),
        "three_stage_no_cache_pruning_and_mrope_exact": pruning_exact,
        "final_visual_removal_exact": bool(
            pruning_exact
            and int(pruning_records[-1].get("result_count", -1)) == 0
            and int(latest_metok[-1].get("result_visual_count", -1)) == 0
        ),
        "final_rope_delta_published": (
            bool(final_rope_deltas)
            and tuple(int(item) for item in splice.get("rope_deltas", ()))
            == final_rope_deltas
        ),
    }


def _step_checks(
    *,
    loss: Any,
    logits: Any,
    state: Any,
    grid: Mapping[str, Any],
    projector_records: Sequence[Mapping[str, Any]],
    algorithm_records: Sequence[Mapping[str, Any]],
    pruning_records: Sequence[Mapping[str, Any]],
    cache_records: Sequence[Mapping[str, Any]],
    bridge_metadata: Mapping[str, Any],
    splice: Mapping[str, Any] | None,
    latest_metok: Sequence[Mapping[str, Any]],
    provider: Mapping[str, Any] | None,
    provider_capture_delta: int,
    parameters: Sequence[Mapping[str, Any]],
    stable_layout: bool,
    torch: Any,
) -> dict[str, bool]:
    exact_record_counts = len(projector_records) == len(algorithm_records) == 1 and len(
        pruning_records
    ) == len(cache_records) == len(latest_metok) == len(METOK_BOUNDARY_LAYERS)
    final_sequence = (
        int(latest_metok[-1].get("result_sequence_length", 0)) if latest_metok else 0
    )
    native_spans = tuple(
        (int(span.start), int(span.end))
        for span in state.modality_token_spans
        if span.modality == "video"
    )
    source_spans = _spans(
        () if splice is None else splice.get("source_placeholder_spans", ())
    )
    labels = state.labels
    source_labels_ignored = False
    completion_labels_present = False
    if labels is not None and native_spans:
        source_labels_ignored = all(
            bool((labels[0, start:end] == -100).all().item())
            for start, end in native_spans
        )
        completion_labels_present = bool((labels != -100).any().item())
    capabilities = bridge_metadata.get("capabilities")
    if not isinstance(capabilities, Mapping):
        capabilities = {}
    return {
        "one_projector_and_three_pruning_invocations": exact_record_counts,
        "compression_applied": bool(
            splice is not None and splice.get("compression_applied")
        ),
        "finite_loss": bool(loss is not None and torch.isfinite(loss).all().item()),
        "finite_logits": bool(
            logits is not None and torch.isfinite(logits).all().item()
        ),
        "post_layer19_label_logit_layout_aligned": bool(
            logits is not None
            and final_sequence > 0
            and int(logits.shape[1]) == final_sequence
        ),
        "native_temporal_spans_preserved": bool(
            source_spans == native_spans
            and len(native_spans) == int(grid["temporal_slice_count"])
            and all(
                end - start == int(grid["per_temporal_slice_source_count"])
                for start, end in native_spans
            )
            and qwen_training.span_count(native_spans)
            == int(grid["source_grid_token_count"])
        ),
        "training_video_labels_ignored": source_labels_ignored,
        "training_completion_labels_present": completion_labels_present,
        "projector_training_layout_compacted": bool(
            splice is not None
            and splice.get("attention_mask_compacted")
            and splice.get("position_ids_compacted")
            and splice.get("cache_position_dense")
            and splice.get("labels_present")
            and splice.get("training_labels_compacted")
            and splice.get("visual_labels_ignored")
            and capabilities.get("training_label_repair")
            and capabilities.get("deepstack_shared_layout")
            and qwen_training.completion_only_labels_exact(state, splice)
        ),
        "training_cache_disabled": state.native_model_inputs.get("use_cache") is False,
        "layout_stable_across_steps": stable_layout,
        **_metok_algorithm_checks(
            projector=(projector_records[0] if len(projector_records) == 1 else None),
            algorithm=(algorithm_records[0] if len(algorithm_records) == 1 else None),
            pruning_records=pruning_records,
            cache_records=cache_records,
            splice=splice,
            latest_metok=latest_metok,
            provider=provider,
            provider_capture_delta=provider_capture_delta,
            grid=grid,
        ),
        **target_parameter_checks(parameters),
    }


def _run(args: argparse.Namespace) -> dict[str, Any]:
    visible = prepare_checkpoint_environment(args.physical_gpu)

    checkpoint = args.checkpoint.resolve()
    siglip_checkpoint = args.siglip_checkpoint.resolve()
    parquet_path = args.nextqa_parquet.resolve()
    video_root = args.video_root.resolve()
    output_dir = args.output_dir.resolve()
    progress_path = output_dir / "progress.jsonl"
    if not parquet_path.is_file():
        raise RuntimeError(f"NExT-QA Parquet is missing: {parquet_path}")
    if not video_root.is_dir():
        raise RuntimeError(f"NExT-QA video root is missing: {video_root}")
    checkpoint_info = _checkpoint_record(checkpoint)
    siglip_checkpoint_info = _checkpoint_record(siglip_checkpoint)
    document, video = training_document(parquet_path, video_root, FIXED_NEXTQA_DOC_IDS)
    fixture, fixture_path = _fixture(
        document=document,
        video=video,
        output_dir=output_dir,
    )
    append_jsonl(
        progress_path,
        {
            "event": "start",
            "scope": SCOPE,
            "code_revision": args.code_revision,
            "fixture_doc_id": FIXTURE_DOC_ID,
            "intended_trainable_handle": "llm.final_norm.weight",
            "defining_stages": [
                "event_segment_keyframe_adaptive_pool",
                "decoder_prune_layers_3_10_19",
            ],
        },
    )

    torch, transformers, gpu_before, admission = prepare_training_device(args, framework_label='Qwen3-VL')
    torch.use_deterministic_algorithms(True, warn_only=True)
    started = time.time()
    stage = _stage()
    conversation = inference._conversation(
        prompt=str(document["prompt"]),
        video=video,
        fps=inference.QWEN3_METOK_VIDEO_FPS,
        max_pixels=inference.QWEN3_METOK_VIDEO_MAX_PIXELS,
    )
    sample = Qwen3VLSample(
        sample_id=f"nextqa_mc_test:{FIXTURE_DOC_ID}:sft",
        conversation=conversation,
        enable_thinking=False,
    )

    facade, processor, model, runtime = _load_runtime(
        checkpoint,
        siglip_checkpoint,
        torch=torch,
        train=True,
    )
    batch, state = _training_state(
        facade=facade,
        processor=processor,
        sample=sample,
        target=str(document["target"]),
    )
    grid = _grid_record(state, batch)
    handles = facade.model_adapter.structural_handles(model)
    deepstack_handles = tuple(
        sorted(
            name for name in handles if name.startswith("vision_deepstack_projector[")
        )
    )
    if len(deepstack_handles) != 3:
        raise RuntimeError(
            "The canonical Qwen3-VL checkpoint must expose exactly three "
            f"DeepStack projectors, got {deepstack_handles!r}"
        )
    selected, optimizer, optimizer_config = final_norm_optimizer(
        facade, model, learning_rate=args.learning_rate, torch=torch,
    )
    selected_names = [name for name, _ in selected]
    activation = _activation(
        facade=facade,
        model=model,
        runtime=runtime,
        stage=stage,
    )
    activation.install()
    step_records: list[dict[str, Any]] = []
    first_layout: Any | None = None
    try:
        model.train()
        for step in range(1, OPTIMIZER_STEPS + 1):
            metadata_before = activation.runtime_metadata()
            projector_before = int(metadata_before["projector_invocations"])
            pruning_before = int(metadata_before["pruning_invocations"])
            provider_before = metadata_before.get("signal_provider") or {}
            provider_capture_before = int(provider_before.get("capture_count", 0))

            loss, logits, parameter_evidence = training_forward_step(
                facade=facade, runtime=runtime, state=state, optimizer=optimizer,
                selected=selected, torch=torch,
            )

            metadata = activation.runtime_metadata()
            projector_records = metadata["projector_layout_deltas"]
            algorithm_records = metadata["algorithm_records"]
            pruning_records = metadata["pruning_layout_deltas"]
            cache_records = metadata["cache_repairs"]
            provider = metadata.get("signal_provider") or {}
            provider_capture_delta = (
                int(provider.get("capture_count", 0)) - provider_capture_before
            )
            bridge = metadata.get("bridge") or {}
            splice = bridge.get("latest_splice")
            latest_metok = tuple(bridge.get("latest_metok") or ())
            current_layout = canonical(
                {
                    "projector": (
                        projector_records[0] if len(projector_records) == 1 else None
                    ),
                    "algorithm": (
                        algorithm_records[0] if len(algorithm_records) == 1 else None
                    ),
                    "pruning": pruning_records,
                    "cache": cache_records,
                    "splice": splice,
                    "latest_metok": latest_metok,
                    "provider": {
                        key: value
                        for key, value in provider.items()
                        if key != "capture_count"
                    },
                }
            )
            if first_layout is None:
                first_layout = current_layout
            checks = _step_checks(
                loss=loss,
                logits=logits,
                state=state,
                grid=grid,
                projector_records=projector_records,
                algorithm_records=algorithm_records,
                pruning_records=pruning_records,
                cache_records=cache_records,
                bridge_metadata=bridge,
                splice=splice,
                latest_metok=latest_metok,
                provider=provider,
                provider_capture_delta=provider_capture_delta,
                parameters=parameter_evidence,
                stable_layout=current_layout == first_layout,
                torch=torch,
            )
            checks["one_projector_and_three_pruning_invocations"] = bool(
                checks["one_projector_and_three_pruning_invocations"]
                and int(metadata["projector_invocations"]) == projector_before + 1
                and int(metadata["pruning_invocations"]) == pruning_before + 3
            )
            record = {
                "event": "optimizer_step",
                "step": step,
                "loss": float(loss.detach().float().item()),
                "logits_shape": [int(item) for item in logits.shape],
                "trainable_parameters": parameter_evidence,
                "checks": checks,
            }
            if step == 1:
                record["native_layout"] = current_layout
            record_training_step(progress_path, step_records, record)
            print_training_step(
                record,
                source_visual_tokens=projector_records[0]['source_count'],
                projector_visual_tokens=projector_records[0]['result_count'],
                layer3_visual_tokens=pruning_records[0]['result_count'],
                layer10_visual_tokens=pruning_records[1]['result_count'],
                layer19_visual_tokens=pruning_records[2]['result_count'],
                gradient_norm=parameter_evidence[0]['gradient_norm'],
                parameter_delta=parameter_evidence[0]['max_parameter_delta'],
            )
        lightweight_checkpoint = save_training_checkpoint(
            args, records=step_records, selected=selected,
            scope=SCOPE, canonical_config=CANONICAL_METOK_CONFIG,
            optimizer=optimizer_config, optimizer_steps=OPTIMIZER_STEPS, torch=torch,
        )
        checkpoint_saved(progress_path, lightweight_checkpoint, optimizer_steps=OPTIMIZER_STEPS)
    finally:
        if activation.installed:
            activation.uninstall()

    del optimizer, selected, state, batch, runtime, model, processor, facade, activation
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()

    reload_facade, reload_processor, reload_model, reload_runtime = _load_runtime(
        checkpoint,
        siglip_checkpoint,
        torch=torch,
        train=False,
    )
    reload_result = load_lightweight_checkpoint(
        reload_model,
        lightweight_checkpoint,
        torch=torch,
    )
    reload_sample = Qwen3VLSample(
        sample_id=f"nextqa_mc_test:{FIXTURE_DOC_ID}:reload",
        conversation=conversation,
        enable_thinking=False,
    )
    reload_batch, reload_state = inference_state(
        facade=reload_facade, processor=reload_processor, sample=reload_sample,
    )
    reload_grid = _grid_record(reload_state, reload_batch)
    if reload_grid != grid:
        raise RuntimeError("Post-reload native Qwen3-VL grid/timing changed")
    reload_handles = reload_facade.model_adapter.structural_handles(reload_model)
    reload_deepstack_handles = tuple(
        sorted(
            name
            for name in reload_handles
            if name.startswith("vision_deepstack_projector[")
        )
    )
    if reload_deepstack_handles != deepstack_handles:
        raise RuntimeError("Post-reload Qwen3-VL DeepStack handles changed")
    reload_activation = _activation(
        facade=reload_facade,
        model=reload_model,
        runtime=reload_runtime,
        stage=stage,
    )
    reload_activation.install()
    try:
        reload_logits = inference_forward(
            facade=reload_facade, runtime=reload_runtime, state=reload_state, torch=torch,
        )
        reload_metadata = reload_activation.runtime_metadata()
        reload_projectors = reload_metadata["projector_layout_deltas"]
        reload_algorithms = reload_metadata["algorithm_records"]
        reload_pruning = reload_metadata["pruning_layout_deltas"]
        reload_caches = reload_metadata["cache_repairs"]
        reload_provider = reload_metadata.get("signal_provider") or {}
        reload_bridge = reload_metadata.get("bridge") or {}
        reload_splice = reload_bridge.get("latest_splice")
        reload_latest_metok = tuple(reload_bridge.get("latest_metok") or ())
        reload_algorithm_checks = _metok_algorithm_checks(
            projector=(reload_projectors[0] if len(reload_projectors) == 1 else None),
            algorithm=(reload_algorithms[0] if len(reload_algorithms) == 1 else None),
            pruning_records=reload_pruning,
            cache_records=reload_caches,
            splice=reload_splice,
            latest_metok=reload_latest_metok,
            provider=reload_provider,
            provider_capture_delta=int(reload_provider.get("capture_count", 0)),
            grid=reload_grid,
        )
        final_sequence_length = (
            int(reload_latest_metok[-1].get("result_sequence_length", 0))
            if reload_latest_metok
            else 0
        )
        post_reload_checks = {
            "checkpoint_tensors_exact": bool(reload_result["all_exact"]),
            "native_grid_and_timing_exact": reload_grid == grid,
            "deepstack_handles_exact": reload_deepstack_handles == deepstack_handles,
            "one_projector_and_three_pruning_invocations": bool(
                int(reload_metadata["projector_invocations"]) == 1
                and int(reload_metadata["pruning_invocations"]) == 3
                and len(reload_projectors) == len(reload_algorithms) == 1
                and len(reload_pruning)
                == len(reload_caches)
                == len(reload_latest_metok)
                == len(METOK_BOUNDARY_LAYERS)
            ),
            "compression_applied": bool(
                reload_splice is not None and reload_splice.get("compression_applied")
            ),
            "labels_absent": reload_state.labels is None,
            "finite_logits": bool(
                reload_logits is not None and torch.isfinite(reload_logits).all().item()
            ),
            "post_layer19_layout_aligned": bool(
                reload_logits is not None
                and final_sequence_length > 0
                and int(reload_logits.shape[1]) == final_sequence_length
            ),
            **reload_algorithm_checks,
        }
        post_reload = {
            "checkpoint_reload": reload_result,
            "grid": reload_grid,
            "logits_shape": (
                [int(item) for item in reload_logits.shape]
                if reload_logits is not None
                else None
            ),
            "projector_compression": (
                canonical(reload_projectors[0]) if len(reload_projectors) == 1 else None
            ),
            "algorithm": (
                canonical(reload_algorithms[0]) if len(reload_algorithms) == 1 else None
            ),
            "pruning_compression": canonical(reload_pruning),
            "cache": canonical(reload_caches),
            "splice": canonical(reload_splice),
            "bridge_latest_metok": canonical(reload_latest_metok),
            "provider": canonical(reload_provider),
            "checks": post_reload_checks,
        }
        record_post_reload(progress_path, post_reload)
    finally:
        if reload_activation.installed:
            reload_activation.uninstall()

    torch.cuda.synchronize()
    gpu_after = _gpu_snapshot(args.physical_gpu)
    return training_report(
        args,
        scope=SCOPE,
        canonical_config=CANONICAL_METOK_CONFIG,
        checkpoint=checkpoint_info,
        fixture_path=fixture_path,
        started=started,
        provenance={
            "siglip_checkpoint": siglip_checkpoint_info,
        },
        boundaries={
            "single_sample_multiple_videos": False,
        },
        environment={
            **training_environment_report(visible=visible, torch=torch, transformers=transformers),
            "deterministic_algorithms": True,
        },
        fixture=fixture,
        native_grid=grid,
        deepstack_projector_handles=list(deepstack_handles),
        training={
            **training_parameter_report(
                optimizer_config,
                selected_names,
                step_records,
                optimizer_steps=OPTIMIZER_STEPS,
            ),
            "siglip_trainable_parameters": 0,
            "frame_count": int(grid["temporal_slice_count"]),
            "defining_stages": ["event_segment_keyframe_adaptive_pool", "decoder_prune_layers_3_10_19"],
            "use_cache": False,
            "training_cache_policy": "none_training_forward",
            "dynamic_cache_compaction_claimed": False,
        },
        lightweight_checkpoint=lightweight_checkpoint,
        post_reload_compressed_inference=post_reload,
        gpu=checkpoint_gpu_report(
            torch=torch,
            before=gpu_before,
            admission=admission,
            after=gpu_after,
            estimate_basis="8.430 GiB evidenced METok/Qwen3 generation reservation plus "
            "three DeepStack branches and final-norm-only autograd; the "
            f"configured {args.estimated_peak_mib} MiB task estimate plus "
            "the separate safety margin covers fresh-checkpoint reload and "
            "allocator variance",
        ),
    )


def _parser() -> argparse.ArgumentParser:
    parser = training_parser(estimated_peak_mib=32768)
    parser.add_argument("--siglip_checkpoint", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    return training_main(argv, parser=_parser(), run=_run, scope=SCOPE, gpu_snapshot=_gpu_snapshot)


if __name__ == "__main__":
    raise SystemExit(main())
