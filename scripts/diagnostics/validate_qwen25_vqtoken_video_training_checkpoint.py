"""Validate the exact VQToken/Qwen2.5-VL five-step training cell.

The harness is offline, batch-one, and internal to evidence generation.  It
trains only the facade-owned ``llm.final_norm.weight`` on the shared doc-442
test-derived fixture, durably records each step before applying gates, saves
that one tensor, reloads it into a fresh real checkpoint, and performs one
compressed inference forward.
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
from scripts.diagnostics import qwen_video_checkpoint_common as inference
from scripts.diagnostics import video_checkpoint_common as _video_checkpoint_common
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
    training_document,
    training_environment_report,
    training_forward_step,
    training_main,
    training_parameter_report,
    training_parser,
    training_state as _training_state,
    video_training_fixture,
)

CANONICAL_VQTOKEN_CONFIG = _video_checkpoint_common.CANONICAL_VQTOKEN_CONFIG
FIXED_NEXTQA_DOC_IDS = _video_checkpoint_common.FIXED_NEXTQA_DOC_IDS
SCOPE = {
    "algorithm": "vqtoken",
    "variant": "vqtoken.video",
    "backbone": "qwen25_vl",
    "binding": "vqtoken.video.qwen25_vl",
}


def _stage() -> CompressionStageConfig:
    return CompressionStageConfig.from_dict(
        {
            "name": "vqtoken-video-qwen25-vl-five-step-train",
            "placement": "projector_level",
            "execution_phase": "forward",
            "plugin": "vqtoken",
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


_activation = partial(training_activation, canonical_config=CANONICAL_VQTOKEN_CONFIG)


def _step_checks(
    *,
    loss: Any,
    logits: Any,
    state: Any,
    grid_token_count: int,
    records: Sequence[Mapping[str, Any]],
    splice: Mapping[str, Any] | None,
    parameters: Sequence[Mapping[str, Any]],
    stable_layout: bool,
    torch: Any,
) -> dict[str, bool]:
    source_count = int(records[0].get("source_count", 0)) if len(records) == 1 else 0
    result_count = int(records[0].get("result_count", 0)) if len(records) == 1 else 0
    source_length = int(state.input_ids.shape[1])
    expected_result_length = source_length - source_count + result_count
    checks = {
        "one_compression_invocation": len(records) == 1,
        "compression_applied": bool(
            splice is not None and splice.get("compression_applied")
        ),
        "non_identity_token_reduction": source_count > result_count,
        "canonical_fixed_k_result": result_count == 32,
        "source_grid_count_exact": source_count == grid_token_count,
        "placeholder_counts_exact": bool(
            splice is not None
            and int(splice.get("source_placeholder_count", 0)) == source_count
            and int(splice.get("result_placeholder_count", 0)) == result_count
        ),
        "source_video_grid_preserved": bool(
            splice is not None and splice.get("source_video_grid_preserved")
        ),
        "native_mrope_gather_exact": bool(
            splice is not None and _mrope_gather_exact(splice)
        ),
        "rope_delta_present": bool(splice is not None and splice.get("rope_deltas")),
        "finite_loss": bool(loss is not None and torch.isfinite(loss).all().item()),
        "finite_logits": bool(
            logits is not None and torch.isfinite(logits).all().item()
        ),
        "label_logit_layout_aligned": bool(
            logits is not None and int(logits.shape[1]) == expected_result_length
        ),
        "layout_stable_across_steps": stable_layout,
        **target_parameter_checks(parameters),
    }
    return checks


def _grid_record(state: Any, batch: Any) -> dict[str, Any]:
    rows = _tensor_rows(state.video_grid_thw)
    if len(rows) != 1 or len(rows[0]) != 3:
        raise RuntimeError(f"Expected one video_grid_thw row, got {rows!r}")
    merge_size = int(batch.metadata["spatial_merge_size"])
    t, height, width = rows[0]
    token_count = t * height * width // (merge_size**2)
    return {
        "video_grid_thw": rows[0],
        "spatial_merge_size": merge_size,
        "source_grid_token_count": token_count,
        "second_per_grid_ts": inference.float_values(state.video_second_per_grid),
    }


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

    torch, transformers, gpu_before, admission = prepare_training_device(args, framework_label="Qwen2.5-VL")
    started = time.time()
    stage = _stage()
    conversation = inference._conversation(
        prompt=str(document["prompt"]),
        video=video,
        fps=inference.QWEN25_VIDEO_FPS,
        max_pixels=inference.QWEN25_VIDEO_MAX_PIXELS,
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
            loss, logits, parameter_evidence = training_forward_step(
                facade=facade, runtime=runtime, state=state, optimizer=optimizer,
                selected=selected, torch=torch,
            )
            metadata = activation.runtime_metadata()
            records = metadata["projector_layout_deltas"]
            splice = (metadata.get("bridge") or {}).get("latest_splice")
            current_layout = canonical(records[0]) if len(records) == 1 else None
            if first_layout is None:
                first_layout = current_layout
            checks = _step_checks(
                loss=loss,
                logits=logits,
                state=state,
                grid_token_count=int(grid["source_grid_token_count"]),
                records=records,
                splice=splice,
                parameters=parameter_evidence,
                stable_layout=current_layout == first_layout,
                torch=torch,
            )
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
            scope=SCOPE, canonical_config=CANONICAL_VQTOKEN_CONFIG,
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
        reload_splice = (reload_metadata.get("bridge") or {}).get("latest_splice")
        source_count = (
            int(reload_records[0]["source_count"]) if len(reload_records) == 1 else 0
        )
        result_count = (
            int(reload_records[0]["result_count"]) if len(reload_records) == 1 else 0
        )
        expected_length = (
            int(reload_state.input_ids.shape[1]) - source_count + result_count
        )
        post_reload_checks = {
            "checkpoint_tensors_exact": bool(reload_result["all_exact"]),
            "one_compression_invocation": bool(
                int(reload_metadata["projector_invocations"]) == 1
                and len(reload_records) == 1
            ),
            "compression_applied": bool(
                reload_splice is not None and reload_splice.get("compression_applied")
            ),
            "non_identity_token_reduction": source_count > result_count,
            "canonical_fixed_k_result": result_count == 32,
            "source_grid_count_exact": (
                source_count == int(reload_grid["source_grid_token_count"])
            ),
            "native_mrope_gather_exact": bool(
                reload_splice is not None and _mrope_gather_exact(reload_splice)
            ),
            "labels_absent": reload_state.labels is None,
            "finite_logits": bool(
                reload_logits is not None and torch.isfinite(reload_logits).all().item()
            ),
            "layout_aligned": bool(
                reload_logits is not None
                and int(reload_logits.shape[1]) == expected_length
            ),
        }
        post_reload = {
            "checkpoint_reload": reload_result,
            "logits_shape": [int(item) for item in reload_logits.shape],
            "grid": reload_grid,
            "compression": canonical(reload_records[0]),
            "splice": canonical(reload_splice),
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
        canonical_config=CANONICAL_VQTOKEN_CONFIG,
        checkpoint=checkpoint_info,
        fixture_path=fixture_path,
        started=started,
        environment=training_environment_report(
            visible=visible, torch=torch, transformers=transformers
        ),
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
            estimate_basis="8.114 GiB evidenced inference peak plus final-norm-only autograd; "
            "32 GiB bound retains more than 3x the inference allocation",
        ),
    )


def _parser() -> argparse.ArgumentParser:
    parser = training_parser(estimated_peak_mib=32768)
    return parser


def main(argv: list[str] | None = None) -> int:
    return training_main(argv, parser=_parser(), run=_run, scope=SCOPE, gpu_snapshot=_gpu_snapshot)


if __name__ == "__main__":
    raise SystemExit(main())
