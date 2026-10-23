"""Validate the exact VisionZip/Qwen2.5-VL five-step training cell.

The harness is offline, batch-one, and internal to evidence generation.  It
trains only the facade-owned ``llm.final_norm.weight`` on the shared doc-442
test-derived fixture, durably records every optimizer step before applying
gates, saves that one tensor, reloads it into a fresh real checkpoint, and
performs one compressed inference forward.  Every forward must preserve the
native Qwen2.5-VL grid/timing and sparse M-RoPE anchors while VisionZip executes
its exact per-temporal-slice ``126 dominant + 21 contextual`` path.
"""

from __future__ import annotations

import argparse
import gc
import time
from functools import partial
from pathlib import Path
from typing import Any, Mapping, Sequence

from mllm_comp.backbones.base import BackboneRuntime
from mllm_comp.backbones.qwen25_vl import Qwen25VLSample, create_qwen25_vl_backbone
from mllm_comp.pipeline.stage import CompressionStageConfig
from scripts.diagnostics import (
    validate_qwen25_vl_visionzip_video_compression_checkpoint as inference,
)
from scripts.diagnostics.qwen_video_checkpoint_common import qwen25_training_grid_record
from scripts.diagnostics.qwen_video_checkpoint_common import (
    mrope_gather_exact as _mrope_gather_exact,
)
from scripts.diagnostics.qwen_video_checkpoint_common import (
    training_timing_preserved as _timing_preserved,
)
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

CANONICAL_VISIONZIP_CONFIG = inference.CANONICAL_VISIONZIP_CONFIG
FIXED_NEXTQA_DOC_IDS = inference.FIXED_NEXTQA_DOC_IDS
DOMINANT_TOKENS_PER_SLICE = 126
CONTEXTUAL_TOKENS_PER_SLICE = 21
RESULT_TOKENS_PER_SLICE = 147
SCOPE = {
    "algorithm": "visionzip",
    "variant": "visionzip.video",
    "backbone": "qwen25_vl",
    "binding": "visionzip.video.qwen25_vl",
}


def _stage() -> CompressionStageConfig:
    return CompressionStageConfig.from_dict(
        {
            "name": "visionzip-video-qwen25-vl-five-step-train",
            "placement": "projector_level",
            "execution_phase": "forward",
            "plugin": "visionzip",
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
        checkpoint, facade=create_qwen25_vl_backbone(), torch=torch, train=train, device_map={"": 0},
    )


_activation = partial(training_activation, canonical_config=CANONICAL_VISIONZIP_CONFIG)


def _grid_record(state: Any, batch: Any) -> dict[str, Any]:
    rows = _tensor_rows(state.video_grid_thw)
    if len(rows) != 1 or len(rows[0]) != 3:
        raise RuntimeError(f"Expected one video_grid_thw row, got {rows!r}")
    merge_size = int(batch.metadata["spatial_merge_size"])
    temporal, height, width = rows[0]
    per_slice = height * width // (merge_size**2)
    if per_slice <= RESULT_TOKENS_PER_SLICE:
        raise RuntimeError(
            "Frozen Qwen2.5-VL VisionZip preprocessing must keep every source "
            f"slice above {RESULT_TOKENS_PER_SLICE} tokens, got {rows[0]!r}"
        )
    return qwen25_training_grid_record(
        rows[0], merge_size=merge_size, timing=state.video_second_per_grid,
    )


def _fixture(
    *, document: Mapping[str, Any], video: Path, output_dir: Path
) -> tuple[dict[str, Any], Path]:
    return video_training_fixture(
        document=document, video=video, output_dir=output_dir,
        preprocessing={
            'fps': inference.QWEN25_VISIONZIP_VIDEO_FPS,
            'min_pixels': inference.QWEN25_VISIONZIP_VIDEO_MIN_PIXELS,
            'max_pixels': inference.QWEN25_VISIONZIP_VIDEO_MAX_PIXELS,
        },
    )


def _visionzip_algorithm_checks(
    record: Mapping[str, Any] | None,
    *,
    source_count: int,
    result_count: int,
    grid: Mapping[str, Any],
) -> dict[str, bool]:
    record = {} if record is None else record
    frames = tuple(record.get("frame_summaries", ()))
    attention_shape = tuple(record.get("attention_shape", ()))
    hidden_shape = tuple(record.get("hidden_state_shape", ()))
    temporal = int(grid["temporal_slice_count"])
    per_slice = int(grid["per_temporal_slice_source_count"])
    expected_source = int(grid["source_grid_token_count"])
    expected_result = temporal * RESULT_TOKENS_PER_SLICE
    return {
        "canonical_source_raster_exact": (
            source_count == expected_source
            and int(record.get("source_count", 0)) == expected_source
            and per_slice > RESULT_TOKENS_PER_SLICE
        ),
        "canonical_result_budget_exact": (
            result_count == expected_result
            and int(record.get("result_count", 0)) == expected_result
            and int(record.get("per_temporal_slice_budget", 0))
            == RESULT_TOKENS_PER_SLICE
        ),
        "canonical_config_exact": dict(record.get("canonical_config", {}))
        == CANONICAL_VISIONZIP_CONFIG,
        "penultimate_signal_layers_exact": (
            int(record.get("attention_layer", 0)) == -2
            and int(record.get("hidden_state_layer", 0)) == -2
        ),
        "penultimate_signal_shapes_aligned": (
            attention_shape == (source_count,)
            and len(hidden_shape) == 2
            and int(hidden_shape[0]) == source_count
            and int(hidden_shape[1]) > 0
        ),
        "penultimate_signals_finite": bool(
            record.get("attention_finite") and record.get("hidden_state_finite")
        ),
        "temporal_slice_count_exact": (
            int(record.get("temporal_slice_count", 0)) == temporal
            and len(frames) == temporal
        ),
        "algorithm_representatives_source_ordered": bool(
            record.get("representatives_source_ordered")
        ),
    }


def _layout_checks(
    *,
    logits: Any,
    state: Any,
    grid: Mapping[str, Any],
    records: Sequence[Mapping[str, Any]],
    algorithm_records: Sequence[Mapping[str, Any]],
    splice: Mapping[str, Any] | None,
    provider_capture_delta: int,
    training_labels: bool,
    torch: Any,
) -> dict[str, bool]:
    source_count = int(records[0].get("source_count", 0)) if len(records) == 1 else 0
    result_count = int(records[0].get("result_count", 0)) if len(records) == 1 else 0
    algorithm = algorithm_records[0] if len(algorithm_records) == 1 else None
    representatives = (
        ()
        if splice is None
        else tuple(int(value) for value in splice.get("representative_indices", ()))
    )
    source_mrope = (
        () if splice is None else tuple(splice.get("source_visual_mrope", ()))
    )
    result_mrope = (
        () if splice is None else tuple(splice.get("result_visual_mrope", ()))
    )
    source_length = int(state.input_ids.shape[1])
    expected_result_length = source_length - source_count + result_count
    video_spans = [
        span for span in state.modality_token_spans if span.modality == "video"
    ]
    one_video_span = len(video_spans) == 1
    span_count = int(video_spans[0].end - video_spans[0].start) if one_video_span else 0
    labels = state.labels
    source_labels_ignored = False
    completion_labels_present = False
    if training_labels and labels is not None and one_video_span:
        video_labels = labels[0, video_spans[0].start : video_spans[0].end]
        source_labels_ignored = bool((video_labels == -100).all().item())
        completion_labels_present = bool((labels != -100).any().item())
    labels_absent = labels is None and (
        "labels" not in state.native_model_inputs
        or state.native_model_inputs["labels"] is None
    )
    mask = state.attention_mask
    splice_layout = {} if splice is None else splice.get("layout_delta", {})
    expected_grid = tuple(int(value) for value in grid["video_grid_thw"])
    expected_timing = float(grid["second_per_grid_ts"][0])
    checks = {
        "one_compression_invocation": (
            len(records) == 1 and len(algorithm_records) == 1
        ),
        "compression_applied": bool(
            splice is not None and splice.get("compression_applied")
        ),
        "non_identity_token_reduction": source_count > result_count,
        "one_contiguous_video_span": one_video_span and span_count == source_count,
        "source_grid_count_exact": source_count == int(grid["source_grid_token_count"]),
        "placeholder_counts_exact": bool(
            splice is not None
            and int(splice.get("source_placeholder_count", 0)) == source_count
            and int(splice.get("result_placeholder_count", 0)) == result_count
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
        "mrope_source_count_aligned": len(source_mrope) == source_count,
        "mrope_result_count_aligned": len(result_mrope) == result_count,
        "native_mrope_gather_exact": bool(
            splice is not None and _mrope_gather_exact(splice)
        ),
        "rope_delta_present": bool(splice is not None and splice.get("rope_deltas")),
        "provider_capture_once": provider_capture_delta == 1,
        "source_attention_mask_aligned": bool(
            mask is not None and tuple(mask.shape) == tuple(state.input_ids.shape)
        ),
        "training_video_labels_ignored": (
            source_labels_ignored if training_labels else labels_absent
        ),
        "completion_labels_present_or_eval_absent": (
            completion_labels_present if training_labels else labels_absent
        ),
        "finite_logits": bool(
            logits is not None and torch.isfinite(logits).all().item()
        ),
        "label_logit_layout_aligned": bool(
            logits is not None
            and int(logits.shape[0]) == 1
            and int(logits.shape[1]) == expected_result_length
        ),
        **_visionzip_algorithm_checks(
            algorithm,
            source_count=source_count,
            result_count=result_count,
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

    torch, transformers, gpu_before, admission = prepare_training_device(args, framework_label='Qwen2.5-VL')
    torch.use_deterministic_algorithms(True, warn_only=True)
    started = time.time()
    stage = _stage()
    conversation = inference._conversation(
        prompt=str(document["prompt"]),
        video=video,
        fps=inference.QWEN25_VISIONZIP_VIDEO_FPS,
        min_pixels=inference.QWEN25_VISIONZIP_VIDEO_MIN_PIXELS,
        max_pixels=inference.QWEN25_VISIONZIP_VIDEO_MAX_PIXELS,
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
            invocation_before = int(metadata_before["projector_invocations"])
            capture_before = int(metadata_before["signal_provider"]["capture_count"])
            loss, logits, parameter_evidence = training_forward_step(
                facade=facade, runtime=runtime, state=state, optimizer=optimizer,
                selected=selected, torch=torch,
            )
            metadata = activation.runtime_metadata()
            records = metadata["projector_layout_deltas"]
            algorithm_records = metadata["algorithm_records"]
            splice = (metadata.get("bridge") or {}).get("latest_splice")
            capture_delta = (
                int(metadata["signal_provider"]["capture_count"]) - capture_before
            )
            current_layout = projector_layout_record(records, algorithm_records, splice)
            if first_layout is None:
                first_layout = current_layout
            checks = _layout_checks(
                logits=logits,
                state=state,
                grid=grid,
                records=records,
                algorithm_records=algorithm_records,
                splice=splice,
                provider_capture_delta=capture_delta,
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
            scope=SCOPE, canonical_config=CANONICAL_VISIONZIP_CONFIG,
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
        reload_records = reload_metadata["projector_layout_deltas"]
        reload_algorithm_records = reload_metadata["algorithm_records"]
        reload_splice = (reload_metadata.get("bridge") or {}).get("latest_splice")
        post_reload_checks = {
            "checkpoint_tensors_exact": bool(reload_result["all_exact"]),
            "native_grid_and_timing_exact": reload_grid == grid,
            **_layout_checks(
                logits=reload_logits,
                state=reload_state,
                grid=reload_grid,
                records=reload_records,
                algorithm_records=reload_algorithm_records,
                splice=reload_splice,
                provider_capture_delta=int(
                    reload_metadata["signal_provider"]["capture_count"]
                ),
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
        canonical_config=CANONICAL_VISIONZIP_CONFIG,
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
            estimate_basis="8.478 GiB evidenced VisionZip/Qwen2.5 inference allocation plus "
            "signal capture and final-norm-only autograd; the 48 GiB bound "
            "retains more than 5.5x the inference allocation",
        ),
    )


def _parser() -> argparse.ArgumentParser:
    parser = training_parser(estimated_peak_mib=49152)
    return parser


def main(argv: list[str] | None = None) -> int:
    return training_main(argv, parser=_parser(), run=_run, scope=SCOPE, gpu_snapshot=_gpu_snapshot)


if __name__ == "__main__":
    raise SystemExit(main())
