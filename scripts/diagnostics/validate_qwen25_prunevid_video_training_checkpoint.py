"""Validate the exact PruneVid/Qwen2.5-VL five-step training cell.

The harness is offline, batch-one, and internal to evidence generation.  It
trains only the facade-owned ``llm.final_norm.weight`` on the shared doc-442
test-derived fixture, durably records every optimizer step, saves that one
tensor, reloads it into a fresh real checkpoint, and performs one compressed
inference forward.  Every forward must execute both canonical PruneVid stages:
projector temporal/static-dynamic reduction and layer-10 VTP.  Training uses
``use_cache=False`` and therefore requires the explicit no-cache training
record instead of claiming DynamicCache compaction.  The native source grid
and timing remain unchanged while both sparse projector and VTP layouts gather
their exact three-axis M-RoPE anchors.
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
from mllm_comp.backbones.qwen25_vl import Qwen25VLSample, create_qwen25_vl_backbone
from mllm_comp.pipeline.stage import CompressionStageConfig
from scripts.diagnostics import (
    validate_qwen25_vl_prunevid_video_compression_checkpoint as inference,
)
from scripts.diagnostics.qwen_video_checkpoint_common import qwen25_training_grid_record
from scripts.diagnostics.qwen_video_checkpoint_common import (
    mrope_gather_exact as _mrope_gather_exact,
)
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

CANONICAL_PRUNEVID_CONFIG = inference.CANONICAL_PRUNEVID_CONFIG
FIXED_NEXTQA_DOC_IDS = inference.FIXED_NEXTQA_DOC_IDS
SCOPE = {
    "algorithm": "prunevid",
    "variant": "prunevid.video",
    "backbone": "qwen25_vl",
    "binding": "prunevid.video.qwen25_vl",
}


def _stage() -> CompressionStageConfig:
    return CompressionStageConfig.from_dict(
        {
            "name": "prunevid-video-qwen25-vl-five-step-train",
            "placement": "projector_level",
            "execution_phase": "forward",
            "plugin": "prunevid",
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


def _load_runtime(checkpoint: Path, *, torch: Any, train: bool) -> tuple[Any, Any, Any, BackboneRuntime]:
    return load_training_runtime(
        checkpoint, facade=create_qwen25_vl_backbone(), torch=torch, train=train, device_map={"": 0},
    )


_activation = partial(training_activation, canonical_config=CANONICAL_PRUNEVID_CONFIG)


def _grid_record(state: Any, batch: Any) -> dict[str, Any]:
    rows = _tensor_rows(state.video_grid_thw)
    if len(rows) != 1 or len(rows[0]) != 3:
        raise RuntimeError(f"Expected one video_grid_thw row, got {rows!r}")
    merge_size = int(batch.metadata["spatial_merge_size"])
    temporal, height, width = rows[0]
    if merge_size < 1 or height % merge_size or width % merge_size:
        raise RuntimeError("Qwen2.5-VL source grid is incompatible with its merger")
    grid_height = height // merge_size
    grid_width = width // merge_size
    per_slice = grid_height * grid_width
    if per_slice < CANONICAL_PRUNEVID_CONFIG["min_tokens_for_cluster"]:
        raise RuntimeError(
            "Frozen Qwen2.5-VL PruneVid raster is below the canonical clustering "
            f"minimum: {rows[0]!r}"
        )
    return {
        **qwen25_training_grid_record(
            rows[0], merge_size=merge_size, timing=state.video_second_per_grid,
        ),
        "grid_height": grid_height,
        "grid_width": grid_width,
    }


def _fixture(
    *, document: Mapping[str, Any], video: Path, output_dir: Path
) -> tuple[dict[str, Any], Path]:
    return video_training_fixture(
        document=document, video=video, output_dir=output_dir,
        preprocessing={
            'fps': inference.QWEN25_PRUNEVID_VIDEO_FPS,
            'max_pixels': inference.QWEN25_PRUNEVID_VIDEO_MAX_PIXELS,
        },
    )


def _timing_preserved(splice: Mapping[str, Any], expected: float) -> bool:
    try:
        observed = float(splice.get("second_per_grid_seconds"))
    except (TypeError, ValueError):
        return False
    return math.isfinite(observed) and math.isclose(
        observed, expected, rel_tol=0.0, abs_tol=1e-9
    )


def _prunevid_algorithm_checks(
    *,
    projector: Mapping[str, Any] | None,
    algorithm: Mapping[str, Any] | None,
    vtp: Mapping[str, Any] | None,
    cache: Mapping[str, Any] | None,
    splice: Mapping[str, Any] | None,
    latest_vtp: Mapping[str, Any] | None,
    grid: Mapping[str, Any],
    source_sequence_length: int,
) -> dict[str, bool]:
    projector = {} if projector is None else projector
    algorithm = {} if algorithm is None else algorithm
    vtp = {} if vtp is None else vtp
    cache = {} if cache is None else cache
    splice = {} if splice is None else splice
    latest_vtp = {} if latest_vtp is None else latest_vtp
    representatives = tuple(
        int(item) for item in algorithm.get("representative_indices", ())
    )
    kept_visual = tuple(int(item) for item in vtp.get("kept_visual_indices", ()))
    dropped_visual = tuple(int(item) for item in vtp.get("dropped_visual_indices", ()))
    sequence_delta = vtp.get("sequence_layout_delta", {})
    splice_layout = splice.get("layout_delta", {})
    source_count = int(projector.get("source_count", 0))
    projector_count = int(projector.get("result_count", 0))
    final_count = int(vtp.get("result_count", 0))
    sequence_source = int(sequence_delta.get("source_count", 0))
    sequence_result = int(sequence_delta.get("result_count", 0))
    temporal = int(grid["temporal_slice_count"])
    grid_height = int(grid["grid_height"])
    grid_width = int(grid["grid_width"])
    tokens_per_frame = int(grid["per_temporal_slice_source_count"])
    expected_source = int(grid["source_grid_token_count"])
    expected_grid = tuple(int(item) for item in grid["video_grid_thw"])
    expected_timing = float(grid["second_per_grid_ts"][0])
    source_mrope = tuple(
        tuple(int(value) for value in row)
        for row in splice.get("source_visual_mrope", ())
    )
    projector_mrope = tuple(
        tuple(int(value) for value in row)
        for row in splice.get("result_visual_mrope", ())
    )
    vtp_source_mrope = tuple(
        tuple(int(value) for value in row)
        for row in latest_vtp.get("source_visual_mrope", ())
    )
    final_mrope = tuple(
        tuple(int(value) for value in row)
        for row in latest_vtp.get("result_visual_mrope", ())
    )
    expected_projector_mrope = (
        tuple(source_mrope[index] for index in representatives)
        if source_mrope and representatives
        else ()
    )
    expected_final_mrope = (
        tuple(projector_mrope[index] for index in kept_visual)
        if projector_mrope and kept_visual
        else ()
    )
    retained_positions = tuple(
        int(item) for item in splice.get("retained_placeholder_positions", ())
    )
    projector_removed = tuple(
        int(item) for item in splice.get("removed_placeholder_positions", ())
    )
    vtp_removed = tuple(
        int(item) for item in latest_vtp.get("dropped_source_placeholder_positions", ())
    )
    expected_vtp_removed = (
        tuple(retained_positions[index] for index in dropped_visual)
        if retained_positions and dropped_visual
        else ()
    )
    combined_removed = tuple(sorted((*projector_removed, *vtp_removed)))
    source_visual_sequence_indices = tuple(
        int(item) for item in latest_vtp.get("source_visual_sequence_indices", ())
    )
    result_visual_sequence_indices = tuple(
        int(item) for item in latest_vtp.get("result_visual_sequence_indices", ())
    )
    rope_values = tuple(int(item) for item in latest_vtp.get("rope_deltas", ()))
    expected_rope_delta = (
        int(latest_vtp.get("max_active_rotary_position", -1))
        + 1
        - int(latest_vtp.get("active_position_count", 0))
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
        "vtp_non_identity_reduction": projector_count > final_count > 0,
        "projector_layout_counts_exact": (
            int(algorithm.get("source_count", 0)) == source_count
            and int(algorithm.get("result_count", 0)) == projector_count
            and int(splice.get("source_placeholder_count", 0)) == source_count
            and int(splice.get("result_placeholder_count", 0)) == projector_count
            and int(splice_layout.get("source_count", 0)) == source_count
            and int(splice_layout.get("result_count", 0)) == projector_count
        ),
        "source_video_grid_preserved": (
            bool(splice.get("source_video_grid_preserved"))
            and tuple(int(item) for item in splice.get("source_video_grid_thw", ()))
            == expected_grid
        ),
        "source_video_timing_preserved": _timing_preserved(splice, expected_timing),
        "sparse_grid_policy": (
            splice.get("result_grid_policy") == "irregular_sparse_source_anchors"
        ),
        "canonical_config_exact": dict(algorithm.get("canonical_config", {}))
        == CANONICAL_PRUNEVID_CONFIG,
        "projected_signal_shapes_exact": (
            tuple(algorithm.get("projected_features_shape", ()))[:1] == (source_count,)
            and tuple(algorithm.get("result_features_shape", ()))[:1]
            == (projector_count,)
            and len(tuple(algorithm.get("projected_features_shape", ()))) == 2
            and len(tuple(algorithm.get("result_features_shape", ()))) == 2
        ),
        "projected_signals_finite": bool(
            algorithm.get("projected_features_finite")
            and algorithm.get("result_features_finite")
        ),
        "projector_mrope_sparse_anchor_gather_exact": (
            len(source_mrope) == source_count
            and len(projector_mrope) == projector_count
            and all(len(row) == 3 for row in source_mrope)
            and all(len(row) == 3 for row in projector_mrope)
            and projector_mrope == expected_projector_mrope
            and _mrope_gather_exact(splice)
        ),
        "vtp_config_and_layout_exact": (
            int(vtp.get("selected_layer", -1))
            == CANONICAL_PRUNEVID_CONFIG["vtp_selected_layer"]
            and float(vtp.get("alpha", 0.0)) == CANONICAL_PRUNEVID_CONFIG["vtp_alpha"]
            and int(vtp.get("source_count", 0)) == projector_count
            and len(kept_visual) == final_count
            and len(dropped_visual) == projector_count - final_count
            and tuple(sorted((*kept_visual, *dropped_visual)))
            == tuple(range(projector_count))
        ),
        "vtp_sequence_delta_visual_only": (
            int(sequence_delta.get("dropped_count", -1))
            == projector_count - final_count
            and sequence_source - sequence_result == projector_count - final_count
            and sequence_source
            == source_sequence_length - source_count + projector_count
            and sequence_result == source_sequence_length - source_count + final_count
        ),
        "training_cache_absent_by_contract": (
            dict(vtp.get("cache", {})) == dict(cache)
            and _training_cache_exact(
                cache,
                source_length=sequence_source,
                result_length=sequence_result,
            )
        ),
        "vtp_mrope_sparse_anchor_gather_exact": (
            vtp_source_mrope == projector_mrope
            and len(final_mrope) == final_count
            and all(len(row) == 3 for row in final_mrope)
            and final_mrope == expected_final_mrope
            and tuple(int(item) for item in latest_vtp.get("kept_visual_indices", ()))
            == kept_visual
            and len(source_visual_sequence_indices) == projector_count
            and len(result_visual_sequence_indices) == final_count
            and result_visual_sequence_indices
            == tuple(sorted(set(result_visual_sequence_indices)))
        ),
        "vtp_removed_placeholder_lineage_exact": (
            len(retained_positions) == projector_count
            and vtp_removed == expected_vtp_removed
            and len(projector_removed) == source_count - projector_count
            and len(combined_removed) == source_count - final_count
            and len(set(combined_removed)) == source_count - final_count
        ),
        "bridge_vtp_layout_exact": (
            int(latest_vtp.get("selected_layer", -1))
            == CANONICAL_PRUNEVID_CONFIG["vtp_selected_layer"]
            and int(latest_vtp.get("source_visual_count", 0)) == projector_count
            and int(latest_vtp.get("result_visual_count", 0)) == final_count
            and int(latest_vtp.get("source_sequence_length", 0)) == sequence_source
            and int(latest_vtp.get("result_sequence_length", 0)) == sequence_result
            and latest_vtp.get("position_policy") == "gather_native_mrope"
        ),
        "post_vtp_rope_delta_exact": (
            bool(latest_vtp.get("rope_delta_exact"))
            and int(latest_vtp.get("active_position_count", 0)) == sequence_result
            and int(latest_vtp.get("expected_rope_delta", -1)) == expected_rope_delta
            and rope_values == (expected_rope_delta,)
            and tuple(int(item) for item in splice.get("rope_deltas", ()))
            == rope_values
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
    vtp_records: Sequence[Mapping[str, Any]],
    cache_records: Sequence[Mapping[str, Any]],
    splice: Mapping[str, Any] | None,
    latest_vtp: Mapping[str, Any] | None,
    parameters: Sequence[Mapping[str, Any]],
    stable_layout: bool,
    torch: Any,
) -> dict[str, bool]:
    projector = projector_records[0] if len(projector_records) == 1 else None
    algorithm = algorithm_records[0] if len(algorithm_records) == 1 else None
    vtp = vtp_records[0] if len(vtp_records) == 1 else None
    cache = cache_records[0] if len(cache_records) == 1 else None
    final_sequence_length = (
        int(latest_vtp.get("result_sequence_length", 0))
        if latest_vtp is not None
        else 0
    )
    source_sequence_length = int(state.input_ids.shape[1])
    video_spans = [
        span for span in state.modality_token_spans if span.modality == "video"
    ]
    one_video_span = len(video_spans) == 1
    video_span_count = (
        int(video_spans[0].end - video_spans[0].start) if one_video_span else 0
    )
    labels = state.labels
    source_labels_ignored = False
    completion_labels_present = False
    if labels is not None and one_video_span:
        video_labels = labels[0, video_spans[0].start : video_spans[0].end]
        source_labels_ignored = bool((video_labels == -100).all().item())
        completion_labels_present = bool((labels != -100).any().item())
    checks = {
        "one_projector_and_vtp_invocation": (
            len(projector_records)
            == len(algorithm_records)
            == len(vtp_records)
            == len(cache_records)
            == 1
        ),
        "compression_applied": bool(
            splice is not None and splice.get("compression_applied")
        ),
        "finite_loss": bool(loss is not None and torch.isfinite(loss).all().item()),
        "finite_logits": bool(
            logits is not None and torch.isfinite(logits).all().item()
        ),
        "post_vtp_label_logit_layout_aligned": bool(
            logits is not None
            and final_sequence_length > 0
            and int(logits.shape[0]) == 1
            and int(logits.shape[1]) == final_sequence_length
        ),
        "one_contiguous_video_span": bool(
            one_video_span and video_span_count == int(grid["source_grid_token_count"])
        ),
        "source_attention_mask_aligned": bool(
            state.attention_mask is not None
            and tuple(state.attention_mask.shape) == tuple(state.input_ids.shape)
        ),
        "training_video_labels_ignored": source_labels_ignored,
        "completion_labels_present": completion_labels_present,
        "training_cache_disabled": bool(
            state.native_model_inputs.get("use_cache") is False
        ),
        "layout_stable_across_steps": stable_layout,
        **_prunevid_algorithm_checks(
            projector=projector,
            algorithm=algorithm,
            vtp=vtp,
            cache=cache,
            splice=splice,
            latest_vtp=latest_vtp,
            grid=grid,
            source_sequence_length=source_sequence_length,
        ),
        **target_parameter_checks(parameters),
    }
    return checks


def _run(args: argparse.Namespace) -> dict[str, Any]:
    visible = prepare_checkpoint_environment(args.physical_gpu)

    checkpoint = args.checkpoint.resolve()
    parquet_path = args.nextqa_parquet.resolve()
    video_root = args.video_root.resolve()
    output_dir = args.output_dir.resolve()
    progress_path = output_dir / "progress.jsonl"
    checkpoint_info = _checkpoint_record(checkpoint)
    document, video = training_document(parquet_path, video_root, FIXED_NEXTQA_DOC_IDS)
    fixture, fixture_path = _fixture(
        document=document, video=video, output_dir=output_dir
    )
    append_jsonl(
        progress_path,
        {
            "event": "start",
            "scope": SCOPE,
            "code_revision": args.code_revision,
            "fixture_doc_id": FIXTURE_DOC_ID,
            "intended_trainable_handle": "llm.final_norm.weight",
            "defining_stages": ["temporal_spatial_reduce", "vtp_prefill_prune"],
        },
    )

    torch, transformers, gpu_before, admission = prepare_training_device(args, framework_label='Qwen2.5-VL')
    torch.use_deterministic_algorithms(True, warn_only=True)
    started = time.time()
    stage = _stage()
    conversation = inference._conversation(
        prompt=str(document["prompt"]),
        video=video,
        fps=inference.QWEN25_PRUNEVID_VIDEO_FPS,
        max_pixels=inference.QWEN25_PRUNEVID_VIDEO_MAX_PIXELS,
    )
    sample = Qwen25VLSample(
        sample_id=f"nextqa_mc_test:{FIXTURE_DOC_ID}:sft",
        conversation=conversation,
    )

    facade, processor, model, runtime = _load_runtime(
        checkpoint, torch=torch, train=True
    )
    batch, state = _training_state(
        facade=facade,
        processor=processor,
        sample=sample,
        target=str(document["target"]),
    )
    grid = _grid_record(state, batch)
    selected, optimizer, optimizer_config = final_norm_optimizer(
        facade, model, learning_rate=args.learning_rate, torch=torch,
    )
    selected_names = [name for name, _ in selected]
    activation = _activation(facade=facade, model=model, runtime=runtime, stage=stage)
    activation.install()
    step_records: list[dict[str, Any]] = []
    first_layout: Any | None = None
    try:
        model.train()
        for step in range(1, OPTIMIZER_STEPS + 1):
            metadata_before = activation.runtime_metadata()
            projector_before = int(metadata_before["projector_invocations"])
            vtp_before = int(metadata_before["vtp_invocations"])
            loss, logits, parameter_evidence = training_forward_step(
                facade=facade, runtime=runtime, state=state, optimizer=optimizer,
                selected=selected, torch=torch,
            )
            metadata = activation.runtime_metadata()
            projector_records = metadata["projector_layout_deltas"]
            algorithm_records = metadata["algorithm_records"]
            vtp_records = metadata["vtp_layout_deltas"]
            cache_records = metadata["cache_repairs"]
            bridge = metadata.get("bridge") or {}
            splice = bridge.get("latest_splice")
            latest_vtp = bridge.get("latest_vtp")
            current_layout = canonical(
                {
                    "projector": (
                        projector_records[0] if len(projector_records) == 1 else None
                    ),
                    "algorithm": (
                        algorithm_records[0] if len(algorithm_records) == 1 else None
                    ),
                    "vtp": vtp_records[0] if len(vtp_records) == 1 else None,
                    "cache": cache_records[0] if len(cache_records) == 1 else None,
                    "splice": splice,
                    "latest_vtp": latest_vtp,
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
                vtp_records=vtp_records,
                cache_records=cache_records,
                splice=splice,
                latest_vtp=latest_vtp,
                parameters=parameter_evidence,
                stable_layout=current_layout == first_layout,
                torch=torch,
            )
            checks["one_projector_and_vtp_invocation"] = bool(
                checks["one_projector_and_vtp_invocation"]
                and int(metadata["projector_invocations"]) == projector_before + 1
                and int(metadata["vtp_invocations"]) == vtp_before + 1
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
                final_vtp_visual_tokens=vtp_records[0]['result_count'],
                gradient_norm=parameter_evidence[0]['gradient_norm'],
                parameter_delta=parameter_evidence[0]['max_parameter_delta'],
            )
        lightweight_checkpoint = save_training_checkpoint(
            args, records=step_records, selected=selected,
            scope=SCOPE, canonical_config=CANONICAL_PRUNEVID_CONFIG,
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
        checkpoint, torch=torch, train=False
    )
    reload_result = load_lightweight_checkpoint(
        reload_model, lightweight_checkpoint, torch=torch
    )
    reload_sample = Qwen25VLSample(
        sample_id=f"nextqa_mc_test:{FIXTURE_DOC_ID}:reload",
        conversation=conversation,
    )
    reload_batch, reload_state = inference_state(
        facade=reload_facade, processor=reload_processor, sample=reload_sample,
    )
    reload_grid = _grid_record(reload_state, reload_batch)
    if reload_grid != grid:
        raise RuntimeError("Post-reload native Qwen2.5-VL grid/timing changed")
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
        reload_vtps = reload_metadata["vtp_layout_deltas"]
        reload_caches = reload_metadata["cache_repairs"]
        reload_bridge = reload_metadata.get("bridge") or {}
        reload_splice = reload_bridge.get("latest_splice")
        reload_latest_vtp = reload_bridge.get("latest_vtp")
        reload_algorithm_checks = _prunevid_algorithm_checks(
            projector=(reload_projectors[0] if len(reload_projectors) == 1 else None),
            algorithm=(reload_algorithms[0] if len(reload_algorithms) == 1 else None),
            vtp=(reload_vtps[0] if len(reload_vtps) == 1 else None),
            cache=(reload_caches[0] if len(reload_caches) == 1 else None),
            splice=reload_splice,
            latest_vtp=reload_latest_vtp,
            grid=reload_grid,
            source_sequence_length=int(reload_state.input_ids.shape[1]),
        )
        final_sequence_length = (
            int(reload_latest_vtp.get("result_sequence_length", 0))
            if reload_latest_vtp is not None
            else 0
        )
        post_reload_checks = {
            "checkpoint_tensors_exact": bool(reload_result["all_exact"]),
            "native_grid_and_timing_exact": reload_grid == grid,
            "one_projector_and_vtp_invocation": bool(
                int(reload_metadata["projector_invocations"]) == 1
                and int(reload_metadata["vtp_invocations"]) == 1
                and len(reload_projectors)
                == len(reload_algorithms)
                == len(reload_vtps)
                == len(reload_caches)
                == 1
            ),
            "compression_applied": bool(
                reload_splice is not None and reload_splice.get("compression_applied")
            ),
            "labels_absent": reload_state.labels is None,
            "finite_logits": bool(
                reload_logits is not None and torch.isfinite(reload_logits).all().item()
            ),
            "post_vtp_layout_aligned": bool(
                reload_logits is not None
                and final_sequence_length > 0
                and int(reload_logits.shape[1]) == final_sequence_length
            ),
            **reload_algorithm_checks,
        }
        post_reload = {
            "checkpoint_reload": reload_result,
            "logits_shape": [int(item) for item in reload_logits.shape],
            "grid": reload_grid,
            "projector_compression": canonical(reload_projectors[0]),
            "algorithm": canonical(reload_algorithms[0]),
            "vtp_compression": canonical(reload_vtps[0]),
            "cache": canonical(reload_caches[0]),
            "splice": canonical(reload_splice),
            "bridge_latest_vtp": canonical(reload_latest_vtp),
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
        canonical_config=CANONICAL_PRUNEVID_CONFIG,
        checkpoint=checkpoint_info,
        fixture_path=fixture_path,
        started=started,
        boundaries={
            "single_sample_multiple_videos": False,
        },
        environment={
            **training_environment_report(visible=visible, torch=torch, transformers=transformers),
            "deterministic_algorithms": True,
        },
        fixture=fixture,
        native_grid=grid,
        training={
            **training_parameter_report(
                optimizer_config,
                selected_names,
                step_records,
                optimizer_steps=OPTIMIZER_STEPS,
            ),
            "defining_stages": ["temporal_spatial_reduce", "vtp_prefill_prune"],
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
            estimate_basis="8.114 GiB evidenced PruneVid/Qwen2.5 inference allocation and "
            "sub-9 GiB observed Qwen2.5 final-norm-only training peers; the "
            "32 GiB bound plus 16 GiB margin remains conservative",
        ),
    )


def _parser() -> argparse.ArgumentParser:
    parser = training_parser(estimated_peak_mib=32768)
    return parser


def main(argv: list[str] | None = None) -> int:
    return training_main(argv, parser=_parser(), run=_run, scope=SCOPE, gpu_snapshot=_gpu_snapshot)


if __name__ == "__main__":
    raise SystemExit(main())
