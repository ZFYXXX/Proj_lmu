"""Validate the exact DyCoKe/Qwen2.5-VL three-step training cell.

The offline batch-one harness repeats the shared doc-442 test-derived SFT
fixture for exactly three optimizer steps.  Every forward must execute the
canonical differentiable two-pass TTM gather while preserving Qwen2.5-VL's
native video grid, timing, sparse MRoPE anchors, placeholders, masks, and
labels.  The generation-only sparse visual-KV controller must remain absent
from ``FORWARD``.  One facade-owned final-norm tensor is trained, saved at
step three, reloaded into a fresh real checkpoint, and exercised again.
"""

from __future__ import annotations

import argparse
import gc
import time
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from mllm_comp.backbones.base import BackboneRuntime
from mllm_comp.backbones.qwen25_vl import Qwen25VLSample, create_qwen25_vl_backbone
from mllm_comp.pipeline.stage import CompressionStageConfig
from scripts.diagnostics import validate_qwen25_vl_dycoke_video_compression_checkpoint as inference
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
from scripts.diagnostics.video_method_checks import (
    dycoke_forward_runtime_checks as _forward_runtime_checks,
)
from scripts.diagnostics.video_training_checkpoint_common import (
    projector_layout_record,
    training_report,
    FIXTURE_DOC_ID,
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

CANONICAL_DYCOKE_CONFIG = inference.CANONICAL_DYCOKE_CONFIG
FIXED_NEXTQA_DOC_IDS = inference.FIXED_NEXTQA_DOC_IDS
SCOPE = {
    "algorithm": "dycoke",
    "variant": "dycoke.video",
    "backbone": "qwen25_vl",
    "binding": "dycoke.video.qwen25_vl",
}
OPTIMIZER_STEPS = 3


@dataclass(frozen=True)
class DyCoKeTrainingProfile:
    """Backbone-specific hooks for the shared DyCoKe training lifecycle."""

    scope: Mapping[str, str]
    canonical_config: Mapping[str, Any]
    fixed_nextqa_doc_ids: tuple[int, ...]
    framework_label: str
    stage_factory: Callable[[], CompressionStageConfig]
    conversation_factory: Callable[..., Any]
    sample_factory: Callable[..., Any]
    runtime_loader: Callable[..., tuple[Any, Any, Any, BackboneRuntime]]
    training_state_factory: Callable[..., tuple[Any, Any]]
    grid_record_factory: Callable[[Any, Any], dict[str, Any]]
    activation_factory: Callable[..., Any]
    fixture_factory: Callable[..., tuple[dict[str, Any], Path]]
    layout_checker: Callable[..., dict[str, bool]]
    estimate_basis: str


def _stage() -> CompressionStageConfig:
    return CompressionStageConfig.from_dict(
        {
            "name": "dycoke-video-qwen25-vl-three-step-train",
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


def _load_runtime(checkpoint: Path, *, torch: Any, train: bool) -> tuple[Any, Any, Any, BackboneRuntime]:
    return load_training_runtime(
        checkpoint, facade=create_qwen25_vl_backbone(), torch=torch, train=train, device_map={"": 0},
    )


_activation = partial(training_activation, canonical_config=CANONICAL_DYCOKE_CONFIG)


def _grid_record(state: Any, batch: Any) -> dict[str, Any]:
    rows = _tensor_rows(state.video_grid_thw)
    if len(rows) != 1 or len(rows[0]) != 3:
        raise RuntimeError(f"Expected one video_grid_thw row, got {rows!r}")
    merge_size = int(batch.metadata["spatial_merge_size"])
    temporal, height, width = rows[0]
    merge_area = merge_size**2
    if merge_size < 1 or (height * width) % merge_area:
        raise RuntimeError("Qwen2.5-VL grid is not divisible by spatial merge size")
    per_slice = height * width // merge_area
    if temporal < 1 or per_slice < 1:
        raise RuntimeError("Qwen2.5-VL video grid is empty")
    return qwen25_training_grid_record(
        rows[0], merge_size=merge_size, timing=state.video_second_per_grid,
    )


def _fixture(
    *, document: Mapping[str, Any], video: Path, output_dir: Path
) -> tuple[dict[str, Any], Path]:
    return video_training_fixture(
        document=document, video=video, output_dir=output_dir,
        preprocessing={
            'fps': inference.QWEN25_VIDEO_FPS,
            'max_pixels': inference.QWEN25_VIDEO_MAX_PIXELS,
        },
    )


def _dycoke_algorithm_checks(
    record: Mapping[str, Any] | None,
    *,
    source_count: int,
    result_count: int,
    grid: Mapping[str, Any],
) -> dict[str, bool]:
    record = {} if record is None else record
    temporal = int(grid["temporal_slice_count"])
    per_slice = int(grid["per_temporal_slice_source_count"])
    expected_source = int(grid["source_grid_token_count"])
    expected_counts = inference._expected_frame_result_counts(temporal, per_slice)
    expected_result = sum(expected_counts)
    result_positions = tuple(record.get("result_visual_positions", ()))
    return {
        "native_grid_source_count_exact": (
            source_count == expected_source
            and int(record.get("source_count", 0)) == expected_source
            and tuple(record.get("frame_source_counts", ())) == (per_slice,) * temporal
        ),
        "canonical_ttm_result_exact": (
            result_count == expected_result
            and int(record.get("result_count", 0)) == expected_result
            and tuple(record.get("frame_result_counts", ())) == expected_counts
        ),
        "single_video_counts_exact": (
            int(record.get("video_count", 0)) == 1
            and tuple(record.get("per_video_source_counts", ())) == (expected_source,)
            and tuple(record.get("per_video_result_counts", ())) == (expected_result,)
            and tuple(record.get("exact_budgets", ())) == (expected_result,)
        ),
        "canonical_config_exact": dict(record.get("canonical_config", {}))
        == CANONICAL_DYCOKE_CONFIG,
        "algorithm_representatives_source_ordered": record.get(
            "representatives_source_ordered"
        )
        is True,
        "result_visual_positions_exact": (
            len(result_positions) == expected_result
            and result_positions == tuple(sorted(set(result_positions)))
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
    metadata: Mapping[str, Any],
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
    frame_records_by_video = (
        () if algorithm is None else tuple(algorithm.get("frame_records_by_video", ()))
    )
    frame_records = (
        () if len(frame_records_by_video) != 1 else tuple(frame_records_by_video[0])
    )
    selected_sources = tuple(
        int(value)
        for frame in frame_records
        for value in frame.get("selected_source_indices", ())
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
    result_span = (
        ()
        if splice is None
        else tuple(int(value) for value in splice.get("result_visual_span", ()))
    )
    expected_result_positions = (
        () if len(result_span) != 2 else tuple(range(result_span[0], result_span[1]))
    )
    layout_positions = (
        ()
        if len(records) != 1
        else tuple(records[0].get("result_visual_positions", ()))
    )
    algorithm_positions = (
        () if algorithm is None else tuple(algorithm.get("result_visual_positions", ()))
    )
    return {
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
        "source_video_timing_preserved": _timing_preserved(
            splice,
            expected_timing,
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
            splice is not None and _mrope_gather_exact(splice)
        ),
        "rope_delta_present": bool(splice is not None and splice.get("rope_deltas")),
        "result_visual_positions_exact": (
            len(layout_positions) == result_count
            and layout_positions == expected_result_positions
            and algorithm_positions == layout_positions
        ),
        "source_attention_mask_aligned": bool(
            mask is not None and tuple(mask.shape) == tuple(state.input_ids.shape)
        ),
        "training_video_labels_ignored": (
            source_labels_ignored if training_labels else labels_absent
        ),
        "completion_labels_present_or_eval_absent": (
            completion_labels_present if training_labels else labels_absent
        ),
        "training_use_cache_false": state.native_model_inputs.get("use_cache") is False,
        "finite_logits": bool(
            logits is not None and torch.isfinite(logits).all().item()
        ),
        "label_logit_layout_aligned": bool(
            logits is not None
            and int(logits.shape[0]) == 1
            and int(logits.shape[1]) == expected_result_length
        ),
        **_dycoke_algorithm_checks(
            algorithm,
            source_count=source_count,
            result_count=result_count,
            grid=grid,
        ),
        **_forward_runtime_checks(metadata),
    }


def _training_conversation(*, prompt: str, video: Path) -> Any:
    return inference._conversation(
        prompt=prompt,
        video=video,
        fps=inference.QWEN25_VIDEO_FPS,
        max_pixels=inference.QWEN25_VIDEO_MAX_PIXELS,
    )


def _training_sample(*, sample_id: str, conversation: Any) -> Qwen25VLSample:
    return Qwen25VLSample(sample_id=sample_id, conversation=conversation)


def _training_profile() -> DyCoKeTrainingProfile:
    return DyCoKeTrainingProfile(
        scope=SCOPE,
        canonical_config=CANONICAL_DYCOKE_CONFIG,
        fixed_nextqa_doc_ids=FIXED_NEXTQA_DOC_IDS,
        framework_label="Qwen2.5-VL",
        stage_factory=_stage,
        conversation_factory=_training_conversation,
        sample_factory=_training_sample,
        runtime_loader=_load_runtime,
        training_state_factory=_training_state,
        grid_record_factory=_grid_record,
        activation_factory=_activation,
        fixture_factory=_fixture,
        layout_checker=_layout_checks,
        estimate_basis=(
            "DyCoKe Qwen2.5-VL generation peaked at 8.114/9.522 GiB "
            "allocated/reserved; the 32 GiB bound plus a separate 16 GiB "
            "margin conservatively covers final-norm-only autograd"
        ),
    )


def _run(
    args: argparse.Namespace,
    *,
    profile: DyCoKeTrainingProfile | None = None,
) -> dict[str, Any]:
    profile = _training_profile() if profile is None else profile
    visible = prepare_checkpoint_environment(args.physical_gpu)

    checkpoint = args.checkpoint.resolve()
    parquet_path = args.nextqa_parquet.resolve()
    video_root = args.video_root.resolve()
    output_dir = args.output_dir.resolve()
    progress_path = output_dir / "progress.jsonl"
    checkpoint_info = _checkpoint_record(checkpoint)
    document, video = training_document(parquet_path, video_root, profile.fixed_nextqa_doc_ids)
    fixture, fixture_path = profile.fixture_factory(
        document=document,
        video=video,
        output_dir=output_dir,
    )
    append_jsonl(
        progress_path,
        {
            "event": "start",
            "scope": profile.scope,
            "code_revision": args.code_revision,
            "fixture_doc_id": FIXTURE_DOC_ID,
            "intended_trainable_handle": "llm.final_norm.weight",
            "decoder_training_cache_policy": "none_training_forward",
        },
    )

    torch, transformers, gpu_before, admission = prepare_training_device(
        args, framework_label=profile.framework_label
    )
    torch.use_deterministic_algorithms(True, warn_only=True)
    started = time.time()
    stage = profile.stage_factory()
    conversation = profile.conversation_factory(
        prompt=str(document["prompt"]),
        video=video,
    )
    sample = profile.sample_factory(
        sample_id=f"nextqa_mc_test:{FIXTURE_DOC_ID}:sft",
        conversation=conversation,
    )

    facade, processor, model, runtime = profile.runtime_loader(
        checkpoint,
        torch=torch,
        train=True,
    )
    batch, state = profile.training_state_factory(
        facade=facade,
        processor=processor,
        sample=sample,
        target=str(document["target"]),
    )
    grid = profile.grid_record_factory(state, batch)
    selected, optimizer, optimizer_config = final_norm_optimizer(
        facade, model, learning_rate=args.learning_rate, torch=torch,
    )
    selected_names = [name for name, _ in selected]
    activation = profile.activation_factory(
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
            invocation_before = int(metadata_before["projector_invocations"])
            loss, logits, parameter_evidence = training_forward_step(
                facade=facade, runtime=runtime, state=state, optimizer=optimizer,
                selected=selected, torch=torch,
            )
            metadata = activation.runtime_metadata()
            records = metadata["projector_layout_deltas"]
            algorithm_records = metadata["algorithm_records"]
            splice = (metadata.get("bridge") or {}).get("latest_splice")
            current_layout = projector_layout_record(records, algorithm_records, splice)
            if first_layout is None:
                first_layout = current_layout
            checks = profile.layout_checker(
                logits=logits,
                state=state,
                grid=grid,
                records=records,
                algorithm_records=algorithm_records,
                splice=splice,
                metadata=metadata,
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
            scope=profile.scope, canonical_config=profile.canonical_config,
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

    reload_facade, reload_processor, reload_model, reload_runtime = (
        profile.runtime_loader(
            checkpoint,
            torch=torch,
            train=False,
        )
    )
    reload_result = load_lightweight_checkpoint(
        reload_model,
        lightweight_checkpoint,
        torch=torch,
    )
    reload_sample = profile.sample_factory(
        sample_id=f"nextqa_mc_test:{FIXTURE_DOC_ID}:reload",
        conversation=conversation,
    )
    reload_batch, reload_state = inference_state(
        facade=reload_facade, processor=reload_processor, sample=reload_sample,
    )
    reload_grid = profile.grid_record_factory(reload_state, reload_batch)
    if reload_grid != grid:
        raise RuntimeError(
            f"Post-reload native {profile.framework_label} grid/timing changed"
        )
    reload_activation = profile.activation_factory(
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
            **profile.layout_checker(
                logits=reload_logits,
                state=reload_state,
                grid=reload_grid,
                records=reload_records,
                algorithm_records=reload_algorithm_records,
                splice=reload_splice,
                metadata=reload_metadata,
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
            "decoder_training_cache_policy": "none_training_forward",
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
        scope=profile.scope,
        canonical_config=profile.canonical_config,
        checkpoint=checkpoint_info,
        fixture_path=fixture_path,
        started=started,
        boundaries={
            "generation_kv_behavior_owned_by_generate_unit": True,
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
            "decoder_training_cache_policy": "none_training_forward",
            "generation_sparse_kv_controller_installed": False,
        },
        lightweight_checkpoint=lightweight_checkpoint,
        post_reload_compressed_inference=post_reload,
        gpu=checkpoint_gpu_report(
            torch=torch,
            before=gpu_before,
            admission=admission,
            after=gpu_after,
            estimate_basis=profile.estimate_basis,
        ),
    )


def _parser() -> argparse.ArgumentParser:
    parser = training_parser(estimated_peak_mib=32768)
    return parser


def main(argv: list[str] | None = None) -> int:
    return training_main(argv, parser=_parser(), run=_run, scope=SCOPE, gpu_snapshot=_gpu_snapshot)


if __name__ == "__main__":
    raise SystemExit(main())
