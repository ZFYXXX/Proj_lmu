"""Validate the exact VQToken/LLaVA-OneVision five-step training cell.

This is an internal evidence harness, not a public support-gate bypass.  It
uses the repository's facade-owned ``llm.final_norm.weight`` convention, repeats
the single test-derived NExT-QA fixture for exactly five optimizer steps,
saves only that selected trainable tensor, reloads it into a fresh copy of
the real base checkpoint, and runs one compressed inference forward.
"""

from __future__ import annotations

import argparse
import gc
import math
import time
from functools import partial
from pathlib import Path
from typing import Any, Mapping, Sequence

from mllm_comp.backbones.llava_onevision import (
    LlavaOneVisionSample,
    create_llava_onevision_backbone,
)
from mllm_comp.pipeline.stage import CompressionStageConfig
from scripts.diagnostics.video_checkpoint_common import (
    prepare_checkpoint_environment,
    checkpoint_gpu_report,
    CANONICAL_VQTOKEN_CONFIG,
    FIXED_NEXTQA_DOC_IDS,
    _checkpoint_record,
    _gpu_snapshot,
)
from scripts.diagnostics.video_training_checkpoint_common import (
    FIXTURE_DOC_ID,
    OPTIMIZER_STEPS,
    append_jsonl,
    canonical as _canonical,
    checkpoint_saved,
    final_norm_optimizer,
    inference_forward,
    inference_state,
    load_lightweight_checkpoint,
    load_training_runtime,
    llava_training_fixture as _fixture,
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
    training_report,
    training_state as _training_state,
)

INTENDED_TRAINABLE_HANDLE = "llm.final_norm.weight"
SCOPE = {
    "algorithm": "vqtoken",
    "variant": "vqtoken.video",
    "backbone": "llava_onevision",
    "binding": "vqtoken.video.llava_onevision",
}


def _stage() -> CompressionStageConfig:
    return CompressionStageConfig.from_dict(
        {
            "name": "vqtoken-video-llava-onevision-five-step-train",
            "placement": "projector_level",
            "execution_phase": "forward",
            "plugin": "vqtoken",
            "algorithm": SCOPE["algorithm"],
            "variant": SCOPE["variant"],
            "target_modalities": ["video"],
            "conditioning_modalities": [],
            "preserved_modalities": ["text", "image", "audio"],
            "coupling_policy": "independent",
            "media_alignment": "spatiotemporal",
            "binding": SCOPE["binding"],
            "config": {},
        }
    )


def _training_checks(
    *,
    loss: Any,
    logits: Any,
    records: Sequence[Mapping[str, Any]],
    latest_splice: Mapping[str, Any] | None,
    parameters: Sequence[Mapping[str, Any]],
    torch: Any,
) -> dict[str, bool]:
    source_count = int(records[0].get("source_count", 0)) if len(records) == 1 else 0
    result_count = int(records[0].get("result_count", 0)) if len(records) == 1 else 0
    sequence_representatives = (
        ()
        if latest_splice is None
        else latest_splice.get("sequence_representative_indices", ())
    )
    checks = {
        "one_compression_invocation": len(records) == 1,
        "compression_applied": bool(
            latest_splice is not None and latest_splice.get("compression_applied")
        ),
        "non_identity_token_reduction": source_count > result_count,
        "canonical_fixed_k_result": result_count == 32,
        "splice_counts_match_layout": bool(
            latest_splice is not None
            and int(latest_splice.get("source_patch_count", 0)) == source_count
            and int(latest_splice.get("result_patch_count", 0)) == result_count
        ),
        "structural_tokens_preserved": bool(
            latest_splice is not None
            and len(latest_splice.get("structural_source_positions", ()))
            == len(latest_splice.get("structural_result_positions", ()))
        ),
        "finite_loss": bool(loss is not None and torch.isfinite(loss).all().item()),
        "finite_logits": bool(
            logits is not None and torch.isfinite(logits).all().item()
        ),
        "label_logit_layout_aligned": bool(
            logits is not None
            and sequence_representatives
            and int(logits.shape[1]) == len(sequence_representatives)
        ),
        "exact_single_intended_target": bool(
            len(parameters) == 1
            and parameters[0].get("facade_handle") == INTENDED_TRAINABLE_HANDLE
        ),
        "target_gradient_present": bool(
            len(parameters) == 1 and parameters[0].get("gradient_present")
        ),
        "target_gradient_finite": bool(
            len(parameters) == 1 and parameters[0].get("gradient_finite")
        ),
        "target_gradient_nonzero": bool(
            len(parameters) == 1 and parameters[0].get("gradient_nonzero")
        ),
        "target_parameter_updated": bool(
            len(parameters) == 1
            and math.isfinite(float(parameters[0].get("max_parameter_delta", 0.0)))
            and float(parameters[0].get("max_parameter_delta", 0.0)) > 0.0
        ),
    }
    return checks


_activation = partial(training_activation, canonical_config=CANONICAL_VQTOKEN_CONFIG)


def _load_runtime(
    checkpoint: Path, *, torch: Any, train: bool
) -> tuple[Any, Any, Any, Any]:
    return load_training_runtime(
        checkpoint, facade=create_llava_onevision_backbone(), torch=torch,
        train=train, device_map="cuda:0",
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
        document=document, video=video, output_dir=output_dir,
    )
    append_jsonl(
        progress_path,
        {
            "event": "start",
            "code_revision": args.code_revision,
            "fixture_doc_id": FIXTURE_DOC_ID,
            "intended_trainable_handle": INTENDED_TRAINABLE_HANDLE,
        },
    )

    torch, transformers, gpu_before, admission = prepare_training_device(args, framework_label="LLaVA-OneVision")
    started = time.time()
    stage = _stage()
    sample = LlavaOneVisionSample(
        sample_id=f"nextqa_mc_test:{FIXTURE_DOC_ID}:sft",
        prompt=str(document["prompt"]),
        video=str(video),
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
    selected, optimizer, optimizer_config = final_norm_optimizer(
        facade, model, learning_rate=args.learning_rate, torch=torch,
    )
    selected_names = [name for name, _ in selected]
    activation = _activation(model=model, runtime=runtime, facade=facade, stage=stage)
    activation.install()
    step_records: list[dict[str, Any]] = []
    try:
        model.train()
        for step in range(1, OPTIMIZER_STEPS + 1):
            metadata_before = activation.runtime_metadata()
            invocation_before = int(metadata_before["projector_invocations"])
            loss, logits, parameter_records = training_forward_step(
                facade=facade, runtime=runtime, state=state, optimizer=optimizer,
                selected=selected, torch=torch,
            )
            metadata = activation.runtime_metadata()
            if int(metadata["projector_invocations"]) != invocation_before + 1:
                raise RuntimeError("VQToken did not run exactly once in one train step")
            records = metadata["projector_layout_deltas"]
            latest_splice = (metadata.get("bridge") or {}).get("latest_splice")
            checks = _training_checks(
                loss=loss,
                logits=logits,
                records=records,
                latest_splice=latest_splice,
                parameters=parameter_records,
                torch=torch,
            )
            record = {
                "step": step,
                "loss": float(loss.detach().float().item()),
                "logits_shape": [int(item) for item in logits.shape],
                "compression": _canonical(records[0]) if len(records) == 1 else None,
                "splice": _canonical(latest_splice),
                "trainable_parameters": parameter_records,
                "checks": checks,
            }
            record_training_step(progress_path, step_records, record)
            print_training_step(
                record,
                source_visual_tokens=records[0]['source_count'],
                result_visual_tokens=records[0]['result_count'],
                checks_passed=True,
            )
        lightweight_checkpoint = save_training_checkpoint(
            args, records=step_records, selected=selected,
            scope=SCOPE, canonical_config=CANONICAL_VQTOKEN_CONFIG,
            optimizer=optimizer_config, optimizer_steps=OPTIMIZER_STEPS, torch=torch,
        )
        checkpoint_saved(
            progress_path, lightweight_checkpoint, optimizer_steps=OPTIMIZER_STEPS,
        )
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
    reload_sample = LlavaOneVisionSample(
        sample_id=f"nextqa_mc_test:{FIXTURE_DOC_ID}:reload",
        prompt=str(document["prompt"]),
        video=str(video),
    )
    reload_batch, reload_state = inference_state(
        facade=reload_facade, processor=reload_processor, sample=reload_sample,
    )
    reload_activation = _activation(
        model=reload_model,
        runtime=reload_runtime,
        facade=reload_facade,
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
        post_reload_checks = {
            "checkpoint_tensors_exact": bool(reload_result["all_exact"]),
            "one_compression_invocation": (
                int(reload_metadata["projector_invocations"]) == 1
                and len(reload_records) == 1
            ),
            "compression_applied": bool(
                reload_splice is not None and reload_splice.get("compression_applied")
            ),
            "non_identity_token_reduction": bool(
                len(reload_records) == 1
                and int(reload_records[0]["source_count"])
                > int(reload_records[0]["result_count"])
            ),
            "canonical_fixed_k_result": bool(
                len(reload_records) == 1
                and int(reload_records[0]["result_count"]) == 32
            ),
            "labels_absent": reload_state.labels is None,
            "finite_logits": bool(
                reload_logits is not None and torch.isfinite(reload_logits).all().item()
            ),
            "layout_aligned": bool(
                reload_logits is not None
                and reload_splice is not None
                and int(reload_logits.shape[1])
                == len(reload_splice.get("sequence_representative_indices", ()))
            ),
        }
        post_reload = {
            "checkpoint_reload": reload_result,
            "logits_shape": (
                None if reload_logits is None
                else [int(item) for item in reload_logits.shape]
            ),
            "compression": _canonical(reload_records[0]) if len(reload_records) == 1 else None,
            "splice": _canonical(reload_splice),
            "checks": post_reload_checks,
        }
        record_post_reload(progress_path, post_reload)
    finally:
        if reload_activation.installed:
            reload_activation.uninstall()

    torch.cuda.synchronize()
    gpu_after = _gpu_snapshot(args.physical_gpu)
    report = training_report(
        args,
        scope=SCOPE,
        canonical_config=CANONICAL_VQTOKEN_CONFIG,
        checkpoint=checkpoint_info,
        fixture_path=fixture_path,
        started=started,
        environment={
            **training_environment_report(visible=visible, torch=torch, transformers=transformers),
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        },
        fixture=fixture,
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
        ),
    )
    return report


def _parser() -> argparse.ArgumentParser:
    return training_parser(estimated_peak_mib=49152)


def main(argv: list[str] | None = None) -> int:
    return training_main(argv, parser=_parser(), run=_run, scope=SCOPE, gpu_snapshot=_gpu_snapshot)


if __name__ == "__main__":
    raise SystemExit(main())
