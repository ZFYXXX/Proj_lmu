"""Validate the exact FastVID/Qwen3-VL five-step training cell.

The harness is offline, batch-one, and internal to evidence generation.  It
trains only the facade-owned ``llm.final_norm.weight`` on the shared doc-442
test-derived fixture, durably records every optimizer step before applying
gates, saves that one tensor, reloads it into a fresh real checkpoint, and
performs one compressed inference forward.  Every forward must preserve the
native Qwen3-VL grid/optional timing, sparse M-RoPE anchors,
``visual_pos_masks``, and identical value-lineage replay on all three
DeepStack branches while FastVID executes canonical DySeg, STPrune, and both
single-/multi-frame DTM from the native Qwen visual signals.
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
from scripts.diagnostics import validate_qwen3_vl_fastvid_video_compression_checkpoint as inference
from scripts.diagnostics.fastvid_video_checkpoint_common import (
    fastvid_algorithm_checks as _fastvid_algorithm_checks,
)
from scripts.diagnostics.qwen_video_checkpoint_common import (
    mrope_gather_exact as _mrope_gather_exact,
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
    projector_layout_record,
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
    training_activation,
    training_document,
    training_environment_report,
    training_forward_step,
    training_main,
    training_parameter_report,
    training_parser,
    training_state as _training_state,
    training_step_checks,
    video_training_fixture,
)

CANONICAL_FASTVID_CONFIG = inference.CANONICAL_FASTVID_CONFIG
FIXED_NEXTQA_DOC_IDS = inference.FIXED_NEXTQA_DOC_IDS
SCOPE = {
    "algorithm": "fastvid",
    "variant": "fastvid.video",
    "backbone": "qwen3_vl",
    "binding": "fastvid.video.qwen3_vl",
}


def _stage() -> CompressionStageConfig:
    return CompressionStageConfig.from_dict(
        {
            "name": "fastvid-video-qwen3-vl-five-step-train",
            "placement": "projector_level",
            "execution_phase": "forward",
            "plugin": "fastvid",
            "algorithm": SCOPE["algorithm"],
            "variant": SCOPE["variant"],
            "binding": SCOPE["binding"],
            "target_modalities": ["video"],
            "conditioning_modalities": [],
            "preserved_modalities": ["text", "image", "audio"],
            "coupling_policy": "independent",
            "media_alignment": "spatiotemporal",
            "config": {},
        }
    )


def _load_runtime(checkpoint: Path, *, torch: Any, train: bool) -> tuple[Any, Any, Any, BackboneRuntime]:
    return load_training_runtime(
        checkpoint, facade=create_qwen3_vl_backbone(), torch=torch, train=train, device_map={"": 0},
    )


_activation = partial(training_activation, canonical_config=CANONICAL_FASTVID_CONFIG)


def _grid_record(state: Any, batch: Any) -> dict[str, Any]:
    rows = _tensor_rows(state.video_grid_thw)
    if len(rows) != 1 or len(rows[0]) != 3:
        raise RuntimeError(f"Expected one video_grid_thw row, got {rows!r}")
    merge_size = int(batch.metadata["spatial_merge_size"])
    temporal, height, width = rows[0]
    if merge_size < 1 or height % merge_size or width % merge_size:
        raise RuntimeError(
            "Qwen3-VL FastVID source grid is incompatible with its spatial merger"
        )
    per_slice = height * width // (merge_size**2)
    retain_num = int(per_slice * float(CANONICAL_FASTVID_CONFIG["retention_ratio"]))
    context_num = int(retain_num * float(CANONICAL_FASTVID_CONFIG["stprune_d"]))
    salient_num = retain_num - context_num
    if retain_num < 2 or context_num < 1 or salient_num < 1:
        raise RuntimeError(
            "Frozen Qwen3-VL FastVID preprocessing must leave at least one "
            f"salient and one contextual token per slice, got {rows[0]!r}"
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
        "per_temporal_slice_source_count": per_slice,
        "source_grid_token_count": temporal * per_slice,
        "raw_tokens_per_frame": height * width,
        "retain_count_per_frame": retain_num,
        "salient_count_per_frame": salient_num,
        "context_count_per_frame": context_num,
        "second_per_grid_ts": timing,
    }


def _fixture(
    *, document: Mapping[str, Any], video: Path, output_dir: Path
) -> tuple[dict[str, Any], Path]:
    return video_training_fixture(
        document=document, video=video, output_dir=output_dir,
        preprocessing={
            'fps': inference.QWEN3_FASTVID_VIDEO_FPS,
            'max_pixels': inference.QWEN3_FASTVID_VIDEO_MAX_PIXELS,
            'enable_thinking': False,
        },
    )


def _timing_preserved(
    splice: Mapping[str, Any] | None, expected: Sequence[float]
) -> bool:
    if splice is None:
        return False
    if not expected:
        return splice.get("second_per_grid_seconds") is None
    if len(expected) != 1:
        return False
    try:
        observed = float(splice.get("second_per_grid_seconds"))
    except (TypeError, ValueError):
        return False
    return math.isfinite(observed) and math.isclose(
        observed, float(expected[0]), rel_tol=0.0, abs_tol=1e-9
    )


def _layout_checks(
    *,
    logits: Any,
    state: Any,
    grid: Mapping[str, Any],
    records: Sequence[Mapping[str, Any]],
    algorithm_records: Sequence[Mapping[str, Any]],
    bridge_metadata: Mapping[str, Any],
    splice: Mapping[str, Any] | None,
    provider_capture_delta: int,
    provider_metadata: Mapping[str, Any],
    training_labels: bool,
    torch: Any,
) -> dict[str, bool]:
    source_count = int(records[0].get("source_count", 0)) if len(records) == 1 else 0
    result_count = int(records[0].get("result_count", 0)) if len(records) == 1 else 0
    algorithm = algorithm_records[0] if len(algorithm_records) == 1 else None
    source_length = int(state.input_ids.shape[1])
    expected_result_length = source_length - source_count + result_count
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
    frame_output_counts = tuple(
        int(value)
        for value in (
            () if algorithm is None else algorithm.get("frame_output_counts", ())
        )
    )
    deepstack_shapes = tuple(
        tuple(int(item) for item in shape)
        for shape in (
            () if splice is None else splice.get("deepstack_result_shapes", ())
        )
    )
    algorithm_deepstack_shapes = tuple(
        tuple(int(item) for item in shape)
        for shape in (
            () if algorithm is None else algorithm.get("deepstack_result_shapes", ())
        )
    )
    capabilities = bridge_metadata.get("capabilities")
    if not isinstance(capabilities, Mapping):
        capabilities = {}
    labels = state.labels
    labels_absent = labels is None and (
        "labels" not in state.native_model_inputs
        or state.native_model_inputs["labels"] is None
    )
    mask = state.attention_mask
    splice_layout = {} if splice is None else splice.get("layout_delta", {})
    expected_grid = tuple(int(value) for value in grid["video_grid_thw"])
    expected_timing = tuple(float(value) for value in grid["second_per_grid_ts"])
    plane_count = int(
        -1 if splice is None else splice.get("source_position_plane_count", -1)
    )
    checks = {
        "one_compression_invocation": (
            len(records) == 1 and len(algorithm_records) == 1
        ),
        "compression_applied": bool(
            splice is not None and splice.get("compression_applied")
        ),
        "non_identity_token_reduction": source_count > result_count,
        "native_temporal_spans_preserved": (
            bool(native_spans)
            and source_spans == native_spans
            and len(native_spans) == int(grid["temporal_slice_count"])
            and all(
                end - start == int(grid["per_temporal_slice_source_count"])
                for start, end in native_spans
            )
        ),
        "result_temporal_spans_exact": (
            len(result_spans) == int(grid["temporal_slice_count"])
            and len(frame_output_counts) == int(grid["temporal_slice_count"])
            and tuple(end - start for start, end in result_spans) == frame_output_counts
        ),
        "source_grid_count_exact": source_count == int(grid["source_grid_token_count"]),
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
            and algorithm is not None
            and int(algorithm.get("source_count", 0)) == source_count
            and int(algorithm.get("result_count", 0)) == result_count
        ),
        "source_video_grid_preserved": bool(
            splice is not None
            and splice.get("source_video_grid_preserved")
            and tuple(int(value) for value in splice.get("source_video_grid_thw", ()))
            == expected_grid
        ),
        "source_video_timing_preserved": _timing_preserved(splice, expected_timing),
        "sparse_grid_policy": bool(
            splice is not None
            and splice.get("result_grid_policy") == "irregular_sparse_source_anchors"
        ),
        "representatives_source_ordered": (
            len(representatives) == result_count
            and representatives == tuple(sorted(representatives))
            and len(set(representatives)) == result_count
        ),
        "algorithm_and_splice_representatives_agree": bool(
            algorithm is not None
            and tuple(
                int(value) for value in algorithm.get("representative_indices", ())
            )
            == representatives
        ),
        "mrope_source_count_aligned": len(source_mrope) == source_count
        and plane_count in (3, 4)
        and all(len(row) == plane_count for row in source_mrope),
        "mrope_result_count_aligned": len(result_mrope) == result_count
        and all(len(row) == plane_count for row in result_mrope),
        "native_mrope_gather_exact": bool(
            splice is not None and _mrope_gather_exact(splice)
        ),
        "native_position_plane_count_exact": bool(
            splice is not None
            and plane_count in (3, 4)
            and int(splice.get("result_position_plane_count", -1)) == plane_count
        ),
        "rope_delta_present": bool(splice is not None and splice.get("rope_deltas")),
        "visual_pos_masks_compacted": bool(
            splice is not None
            and splice.get("visual_pos_masks_compacted")
            and int(splice.get("visual_pos_masks_result_count", 0)) == result_count
        ),
        "main_and_three_deepstack_layouts_exact": bool(
            splice is not None
            and algorithm is not None
            and int(splice.get("deepstack_branch_count", 0)) == 3
            and int(algorithm.get("deepstack_branch_count", 0)) == 3
            and len(deepstack_shapes) == 3
            and deepstack_shapes == algorithm_deepstack_shapes
            and all(
                len(shape) == 2 and shape[0] == result_count
                for shape in deepstack_shapes
            )
            and splice.get("shared_main_deepstack_layout")
            and int(splice.get("deepstack_group_count", 0)) == result_count
            and capabilities.get("deepstack_shared_layout")
        ),
        "attention_position_cache_compacted": bool(
            splice is not None
            and splice.get("attention_mask_compacted")
            and splice.get("position_ids_compacted")
            and splice.get("cache_position_dense")
        ),
        "provider_capture_once": provider_capture_delta == 1,
        "provider_identity_exact": provider_metadata.get("provider")
        == "qwen3_vl_fastvid_video_signals",
        "provider_native_qwen_path": bool(
            provider_metadata.get("video_only")
            and provider_metadata.get("raw_stream_attached")
            and not provider_metadata.get("siglip_pooling_head_loaded")
            and provider_metadata.get("siglip_path") is None
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
                and capabilities.get("training_label_repair")
                and qwen_training.completion_only_labels_exact(state, splice)
            )
        ),
        "eval_labels_absent": bool(training_labels or labels_absent),
        "finite_logits": bool(
            logits is not None and torch.isfinite(logits).all().item()
        ),
        "label_logit_layout_aligned": bool(
            logits is not None
            and int(logits.shape[0]) == 1
            and int(logits.shape[1]) == expected_result_length
        ),
        **_fastvid_algorithm_checks(
            algorithm,
            source_count=source_count,
            grid=grid,
        ),
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
        },
    )

    torch, transformers, gpu_before, admission = prepare_training_device(args, framework_label='Qwen3-VL')
    torch.use_deterministic_algorithms(True, warn_only=True)
    started = time.time()
    stage = _stage()
    conversation = inference._conversation(
        prompt=str(document["prompt"]),
        video=video,
        fps=inference.QWEN3_FASTVID_VIDEO_FPS,
        max_pixels=inference.QWEN3_FASTVID_VIDEO_MAX_PIXELS,
    )
    sample = Qwen3VLSample(
        sample_id=f"nextqa_mc_test:{FIXTURE_DOC_ID}:sft",
        conversation=conversation,
        enable_thinking=False,
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
    activation = _activation(facade=facade, model=model, runtime=runtime, stage=stage)
    activation.install()
    step_records: list[dict[str, Any]] = []
    first_layout: Any | None = None
    try:
        model.train()
        for step in range(1, OPTIMIZER_STEPS + 1):
            metadata_before = activation.runtime_metadata()
            invocation_before = int(metadata_before["projector_invocations"])
            capture_before = int(metadata_before["signal_provider"]["capture_count"])
            loss, logits, parameter_evidence = training_forward_step(
                facade=facade, runtime=runtime, state=state, optimizer=optimizer,
                selected=selected, torch=torch,
            )
            metadata = activation.runtime_metadata()
            records = metadata["projector_layout_deltas"]
            algorithm_records = metadata["algorithm_records"]
            bridge_metadata = metadata.get("bridge") or {}
            splice = bridge_metadata.get("latest_splice")
            provider_metadata = dict(metadata["signal_provider"])
            capture_delta = int(provider_metadata["capture_count"]) - capture_before
            current_layout = projector_layout_record(records, algorithm_records, splice)
            if first_layout is None:
                first_layout = current_layout
            checks = _layout_checks(
                logits=logits,
                state=state,
                grid=grid,
                records=records,
                algorithm_records=algorithm_records,
                bridge_metadata=bridge_metadata,
                splice=splice,
                provider_capture_delta=capture_delta,
                provider_metadata=provider_metadata,
                torch=torch,
                training_labels=True,
            )
            checks.update(training_step_checks(
                loss=loss,
                parameters=parameter_evidence,
                stable_layout=current_layout == first_layout,
                torch=torch,
            ))
            checks["one_compression_invocation"] = bool(
                checks["one_compression_invocation"]
                and int(metadata["projector_invocations"]) == invocation_before + 1
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
                source_visual_tokens=records[0]['source_count'],
                result_visual_tokens=records[0]['result_count'],
                gradient_norm=parameter_evidence[0]['gradient_norm'],
                parameter_delta=parameter_evidence[0]['max_parameter_delta'],
            )
        lightweight_checkpoint = save_training_checkpoint(
            args, records=step_records, selected=selected,
            scope=SCOPE, canonical_config=CANONICAL_FASTVID_CONFIG,
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
        reload_records = reload_metadata["projector_layout_deltas"]
        reload_algorithm_records = reload_metadata["algorithm_records"]
        reload_provider = dict(reload_metadata["signal_provider"])
        reload_bridge = reload_metadata.get("bridge") or {}
        reload_splice = reload_bridge.get("latest_splice")
        post_reload_checks = {
            "checkpoint_tensors_exact": bool(reload_result["all_exact"]),
            "native_grid_and_timing_exact": reload_grid == grid,
            "deepstack_handles_exact": reload_deepstack_handles == deepstack_handles,
            **_layout_checks(
                logits=reload_logits,
                state=reload_state,
                grid=reload_grid,
                records=reload_records,
                algorithm_records=reload_algorithm_records,
                bridge_metadata=reload_bridge,
                splice=reload_splice,
                provider_capture_delta=int(reload_provider["capture_count"]),
                provider_metadata=reload_provider,
                training_labels=False,
                torch=torch,
            ),
        }
        post_reload_checks["one_compression_invocation"] = bool(
            post_reload_checks["one_compression_invocation"]
            and int(reload_metadata["projector_invocations"]) == 1
        )
        post_reload = {
            "checkpoint_reload": reload_result,
            "logits_shape": [int(item) for item in reload_logits.shape],
            "grid": reload_grid,
            **projector_layout_record(reload_records, reload_algorithm_records, reload_splice),
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
        canonical_config=CANONICAL_FASTVID_CONFIG,
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
        deepstack_projector_handles=list(deepstack_handles),
        training=training_parameter_report(
            optimizer_config,
            selected_names,
            step_records,
            optimizer_steps=OPTIMIZER_STEPS,
        ),
        lightweight_checkpoint=lightweight_checkpoint,
        post_reload_compressed_inference=post_reload,
        gpu=checkpoint_gpu_report(
            torch=torch,
            before=gpu_before,
            admission=admission,
            after=gpu_after,
            estimate_basis="5.108 GiB evidenced FastVID/Qwen3 inference allocation plus "
            "native visual-signal capture and final-norm-only autograd; the "
            f"configured {args.estimated_peak_mib} MiB task bound plus the "
            "separate safety margin covers a fresh checkpoint reload and "
            "allocator variance",
        ),
    )


def _parser() -> argparse.ArgumentParser:
    parser = training_parser(estimated_peak_mib=49152)
    return parser


def main(argv: list[str] | None = None) -> int:
    return training_main(argv, parser=_parser(), run=_run, scope=SCOPE, gpu_snapshot=_gpu_snapshot)


if __name__ == "__main__":
    raise SystemExit(main())
