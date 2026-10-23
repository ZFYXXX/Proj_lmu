"""Validate the exact DyCoKe/LLaVA-OneVision three-step training cell.

The offline batch-one harness repeats the shared doc-442 test-derived SFT
fixture for exactly three optimizer steps.  Every forward must execute the
canonical differentiable two-pass TTM gather and the native LLaVA legalizer,
while installing no generation-only sparse visual-KV controller.  The harness
trains only the facade-owned ``llm.final_norm.weight``, durably records each
step, saves that one tensor, reloads it into a fresh real checkpoint, and runs
one post-reload compressed forward.  This is code-correctness evidence only.
"""

from __future__ import annotations

import argparse
import gc
import time
from functools import partial
from pathlib import Path
from typing import Any, Mapping, Sequence

from mllm_comp.backbones.base import BackboneRuntime
from mllm_comp.backbones.llava_onevision import (
    LlavaOneVisionSample,
    create_llava_onevision_backbone,
)
from mllm_comp.pipeline.stage import CompressionStageConfig
from scripts.diagnostics import validate_llava_dycoke_video_compression_checkpoint as inference
from scripts.diagnostics.video_checkpoint_common import (
    prepare_checkpoint_environment,
    checkpoint_gpu_report,
    _checkpoint_record,
    _gpu_snapshot,
)
from scripts.diagnostics.video_method_checks import (
    dycoke_forward_runtime_checks as _forward_runtime_checks,
)
from scripts.diagnostics.video_training_checkpoint_common import (
    projector_layout_record,
    training_report,
    FIXTURE_DOC_ID,
    append_jsonl,
    canonical,
    checkpoint_saved,
    final_norm_optimizer,
    inference_forward,
    inference_state,
    llava_training_fixture as _fixture,
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
)

CANONICAL_DYCOKE_CONFIG = inference.CANONICAL_DYCOKE_CONFIG
FIXED_NEXTQA_DOC_IDS = inference.FIXED_NEXTQA_DOC_IDS
SOURCE_TOKENS_PER_FRAME = inference.LLAVA_PATCHES_PER_FRAME
SCOPE = {
    "algorithm": "dycoke",
    "variant": "dycoke.video",
    "backbone": "llava_onevision",
    "binding": "dycoke.video.llava_onevision",
}
OPTIMIZER_STEPS = 3


def _stage() -> CompressionStageConfig:
    return CompressionStageConfig.from_dict(
        {
            "name": "dycoke-video-llava-onevision-three-step-train",
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
        checkpoint, facade=create_llava_onevision_backbone(), torch=torch, train=train, device_map="cuda:0",
    )


_activation = partial(training_activation, canonical_config=CANONICAL_DYCOKE_CONFIG)


def _dycoke_algorithm_checks(
    record: Mapping[str, Any] | None,
    *,
    source_count: int,
    result_count: int,
    frame_count: int,
) -> dict[str, bool]:
    record = {} if record is None else record
    expected_counts = inference._expected_frame_result_counts(frame_count)
    expected_source = frame_count * SOURCE_TOKENS_PER_FRAME
    expected_result = sum(expected_counts)
    result_positions = tuple(record.get("result_visual_positions", ()))
    return {
        "native_projected_frame_raster_exact": (
            source_count == expected_source
            and int(record.get("source_count", 0)) == expected_source
            and tuple(record.get("frame_source_counts", ()))
            == (SOURCE_TOKENS_PER_FRAME,) * frame_count
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
        "representatives_source_ordered": record.get("representatives_source_ordered")
        is True,
        "result_visual_positions_exact": (
            len(result_positions) == expected_result
            and result_positions == tuple(sorted(set(result_positions)))
        ),
    }


def _step_checks(
    *,
    loss: Any,
    logits: Any,
    records: Sequence[Mapping[str, Any]],
    algorithm_records: Sequence[Mapping[str, Any]],
    splice: Mapping[str, Any] | None,
    metadata: Mapping[str, Any],
    parameters: Sequence[Mapping[str, Any]],
    stable_layout: bool,
    frame_count: int,
    labels_present: bool,
    use_cache_disabled: bool,
    torch: Any,
) -> dict[str, bool]:
    source_count = int(records[0].get("source_count", 0)) if len(records) == 1 else 0
    result_count = int(records[0].get("result_count", 0)) if len(records) == 1 else 0
    algorithm = algorithm_records[0] if len(algorithm_records) == 1 else None
    sequence_representatives = (
        () if splice is None else splice.get("sequence_representative_indices", ())
    )
    algorithm_positions = (
        () if algorithm is None else algorithm.get("result_visual_positions", ())
    )
    layout_positions = (
        () if len(records) != 1 else records[0].get("result_visual_positions", ())
    )
    return {
        "one_compression_invocation": (
            len(records) == 1 and len(algorithm_records) == 1
        ),
        "compression_applied": bool(
            splice is not None and splice.get("compression_applied")
        ),
        "non_identity_token_reduction": source_count > result_count,
        "splice_counts_match_layout": bool(
            splice is not None
            and int(splice.get("source_patch_count", 0)) == source_count
            and int(splice.get("result_patch_count", 0)) == result_count
        ),
        "structural_tokens_preserved": bool(
            splice is not None
            and len(splice.get("structural_source_positions", ()))
            == len(splice.get("structural_result_positions", ()))
        ),
        "visual_positions_match_layout": tuple(algorithm_positions)
        == tuple(layout_positions),
        "finite_loss": bool(loss is not None and torch.isfinite(loss).all().item()),
        "finite_logits": bool(
            logits is not None and torch.isfinite(logits).all().item()
        ),
        "training_labels_present": labels_present,
        "label_logit_position_layout_aligned": bool(
            logits is not None
            and sequence_representatives
            and int(logits.shape[1]) == len(sequence_representatives)
        ),
        "training_use_cache_false": use_cache_disabled,
        "layout_stable_across_steps": stable_layout,
        **_dycoke_algorithm_checks(
            algorithm,
            source_count=source_count,
            result_count=result_count,
            frame_count=frame_count,
        ),
        **_forward_runtime_checks(metadata),
        **target_parameter_checks(parameters),
    }


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
            "decoder_training_cache_policy": "none_training_forward",
        },
    )

    torch, transformers, gpu_before, admission = prepare_training_device(args, framework_label='LLaVA')
    torch.use_deterministic_algorithms(True, warn_only=True)
    started = time.time()
    stage = _stage()
    sample = LlavaOneVisionSample(
        sample_id=f"nextqa_mc_test:{FIXTURE_DOC_ID}:sft",
        prompt=str(document["prompt"]),
        video=str(video),
    )

    facade, processor, model, runtime = _load_runtime(
        checkpoint,
        torch=torch,
        train=True,
    )
    batch, state = _training_state(
        facade=facade,
        processor=processor,
        sample=sample,
        target=str(document["target"]),
    )
    frame_count = int(batch.metadata["video"]["frame_count"])
    if frame_count != inference.LLAVA_FRAME_COUNT:
        raise RuntimeError(
            f"DyCoKe LLaVA fixture must decode 32 frames, got {frame_count}"
        )
    labels_present = bool(
        state.labels is not None
        and state.labels.ndim == 2
        and int(state.labels.shape[0]) == 1
        and bool((state.labels != -100).any().item())
    )
    use_cache_disabled = state.native_model_inputs.get("use_cache") is False
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
            checks = _step_checks(
                loss=loss,
                logits=logits,
                records=records,
                algorithm_records=algorithm_records,
                splice=splice,
                metadata=metadata,
                parameters=parameter_evidence,
                stable_layout=current_layout == first_layout,
                frame_count=frame_count,
                labels_present=labels_present,
                use_cache_disabled=use_cache_disabled,
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
            scope=SCOPE, canonical_config=CANONICAL_DYCOKE_CONFIG,
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
        torch=torch,
        train=False,
    )
    reload_result = load_lightweight_checkpoint(
        reload_model,
        lightweight_checkpoint,
        torch=torch,
    )
    reload_sample = LlavaOneVisionSample(
        sample_id=f"nextqa_mc_test:{FIXTURE_DOC_ID}:reload",
        prompt=str(document["prompt"]),
        video=str(video),
    )
    reload_batch, reload_state = inference_state(
        facade=reload_facade, processor=reload_processor, sample=reload_sample,
    )
    reload_frame_count = int(reload_batch.metadata["video"]["frame_count"])
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
        source_count = (
            int(reload_records[0]["source_count"]) if len(reload_records) == 1 else 0
        )
        result_count = (
            int(reload_records[0]["result_count"]) if len(reload_records) == 1 else 0
        )
        reload_algorithm = (
            reload_algorithm_records[0] if len(reload_algorithm_records) == 1 else None
        )
        sequence_representatives = (
            ()
            if reload_splice is None
            else reload_splice.get("sequence_representative_indices", ())
        )
        post_reload_checks = {
            "checkpoint_tensors_exact": bool(reload_result["all_exact"]),
            "one_compression_invocation": bool(
                int(reload_metadata["projector_invocations"]) == 1
                and len(reload_records) == 1
                and len(reload_algorithm_records) == 1
            ),
            "compression_applied": bool(
                reload_splice is not None and reload_splice.get("compression_applied")
            ),
            "non_identity_token_reduction": source_count > result_count,
            "splice_counts_match_layout": bool(
                reload_splice is not None
                and int(reload_splice.get("source_patch_count", 0)) == source_count
                and int(reload_splice.get("result_patch_count", 0)) == result_count
            ),
            "structural_tokens_preserved": bool(
                reload_splice is not None
                and len(reload_splice.get("structural_source_positions", ()))
                == len(reload_splice.get("structural_result_positions", ()))
            ),
            "labels_absent": reload_state.labels is None,
            "post_reload_use_cache_false": reload_state.native_model_inputs.get(
                "use_cache"
            )
            is False,
            "finite_logits": bool(
                reload_logits is not None and torch.isfinite(reload_logits).all().item()
            ),
            "layout_aligned": bool(
                reload_logits is not None
                and sequence_representatives
                and int(reload_logits.shape[1]) == len(sequence_representatives)
            ),
            "visual_positions_match_layout": bool(
                reload_algorithm is not None
                and len(reload_records) == 1
                and tuple(reload_algorithm.get("result_visual_positions", ()))
                == tuple(reload_records[0].get("result_visual_positions", ()))
            ),
            **_dycoke_algorithm_checks(
                reload_algorithm,
                source_count=source_count,
                result_count=result_count,
                frame_count=reload_frame_count,
            ),
            **_forward_runtime_checks(reload_metadata),
        }
        post_reload = {
            "checkpoint_reload": reload_result,
            "logits_shape": [int(item) for item in reload_logits.shape],
            "compression": canonical(reload_records[0]),
            "algorithm": canonical(reload_algorithm),
            "splice": canonical(reload_splice),
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
        scope=SCOPE,
        canonical_config=CANONICAL_DYCOKE_CONFIG,
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
        training={
            **training_parameter_report(
                optimizer_config,
                selected_names,
                step_records,
                optimizer_steps=OPTIMIZER_STEPS,
            ),
            "frame_count": frame_count,
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
            estimate_basis="DyCoKe LLaVA generation peaked at 5.420/10.120 GiB "
            "allocated/reserved; the 32 GiB bound plus a separate 16 GiB "
            "margin conservatively covers final-norm-only autograd",
        ),
    )


def _parser() -> argparse.ArgumentParser:
    parser = training_parser(estimated_peak_mib=32768)
    return parser


def main(argv: list[str] | None = None) -> int:
    return training_main(argv, parser=_parser(), run=_run, scope=SCOPE, gpu_snapshot=_gpu_snapshot)


if __name__ == "__main__":
    raise SystemExit(main())
