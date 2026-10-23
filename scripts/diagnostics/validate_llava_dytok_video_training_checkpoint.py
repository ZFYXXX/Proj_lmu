"""Validate the exact DyToK/LLaVA-OneVision five-step training cell.

The harness is offline, batch-one, and internal to evidence generation.  It
trains only the facade-owned ``llm.final_norm.weight`` on the shared doc-442
test-derived fixture, records all five optimizer steps, saves that one tensor,
reloads it into a fresh real checkpoint, and performs one compressed forward.
Every forward must execute one frozen local detector pass over layers 16--23,
derive non-uniform frame budgets, and run the canonical dynamic VisionZip
reduction.  The separate detector and SigLIP tower remain frozen external
capability state and are excluded from the lightweight checkpoint.
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
from scripts.diagnostics import validate_llava_dytok_video_compression_checkpoint as inference
from scripts.diagnostics.video_checkpoint_common import (
    prepare_checkpoint_environment,
    checkpoint_gpu_report,
    _checkpoint_record,
    _gpu_snapshot,
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
    llava_training_fixture as _fixture,
    load_lightweight_checkpoint,
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

CANONICAL_DYTOK_CONFIG = inference.CANONICAL_DYTOK_CONFIG
FIXED_NEXTQA_DOC_IDS = inference.FIXED_NEXTQA_DOC_IDS
PROJECTED_TOKENS_PER_FRAME = 196
RESULT_TOKENS_PER_FRAME = 147
SCOPE = {
    "algorithm": "dytok",
    "variant": "dytok.video",
    "backbone": "llava_onevision",
    "binding": "dytok.video.llava_onevision",
}


def _stage() -> CompressionStageConfig:
    return CompressionStageConfig.from_dict(
        {
            "name": "dytok-video-llava-onevision-five-step-train",
            "placement": "projector_level",
            "execution_phase": "forward",
            "plugin": "dytok",
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
    detector_checkpoint: Path,
    detector_siglip: Path,
    *,
    torch: Any,
    train: bool,
) -> tuple[Any, Any, Any, BackboneRuntime]:
    facade = create_llava_onevision_backbone()
    processor = facade.processor_adapter.load_processor(
        str(checkpoint), local_files_only=True
    )
    model = facade.model_adapter.load_model(
        str(checkpoint),
        train=train,
        local_files_only=True,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
        device_map="cuda:0",
    )
    disable_checkpointing = getattr(model, "gradient_checkpointing_disable", None)
    if callable(disable_checkpointing):
        disable_checkpointing()
    runtime = BackboneRuntime(
        model=model,
        processor=processor,
        model_path_or_id=str(checkpoint),
        device=torch.device("cuda:0"),
        dtype=torch.bfloat16,
        metadata={
            "enable_talker": False,
            "speech_sample_rate": None,
            "runtime_mode": "default",
            "compression_mode": "compressed",
            "dytok_detection_model_path": str(detector_checkpoint),
            "dytok_detection_siglip_path": str(detector_siglip),
            "dytok_detection_device_map": "cuda:0",
            "dytok_detection_dtype": torch.float16,
        },
    )
    return facade, processor, model, runtime


_activation = partial(training_activation, canonical_config=CANONICAL_DYTOK_CONFIG)


def _detector_calls(metadata: Mapping[str, Any]) -> int:
    detector = metadata.get("detector")
    return int(detector.get("calls", 0)) if isinstance(detector, Mapping) else 0


def _visionzip_captures(metadata: Mapping[str, Any]) -> int:
    visionzip = metadata.get("visionzip_signals")
    return (
        int(visionzip.get("capture_count", 0)) if isinstance(visionzip, Mapping) else 0
    )


def _dytok_algorithm_checks(
    record: Mapping[str, Any] | None,
    *,
    source_count: int,
    result_count: int,
    frame_count: int,
    detector_call_delta: int,
    detector_metadata: Mapping[str, Any],
    detector_registered_as_trainable_child: bool,
) -> dict[str, bool]:
    record = {} if record is None else record
    frames = tuple(record.get("frame_summaries", ()))
    importance = tuple(float(item) for item in record.get("frame_importance", ()))
    provider_importance = tuple(
        float(item) for item in record.get("provider_frame_importance", ())
    )
    budgets = tuple(int(item) for item in record.get("total_budgets", ()))
    expected_source = frame_count * PROJECTED_TOKENS_PER_FRAME
    expected_result = min(frame_count * RESULT_TOKENS_PER_FRAME, expected_source)
    attention_shape = tuple(record.get("attention_shape", ()))
    hidden_shape = tuple(record.get("hidden_state_shape", ()))
    detection_attention = record.get("detection_attention")
    detection_attention = (
        detection_attention if isinstance(detection_attention, Mapping) else {}
    )
    prompt_layout = record.get("prompt_layout")
    prompt_layout = prompt_layout if isinstance(prompt_layout, Mapping) else {}
    return {
        "canonical_frame_count_exact": frame_count == 32,
        "native_projected_frame_raster_exact": (
            source_count == expected_source
            and int(record.get("source_count", 0)) == expected_source
        ),
        "canonical_fixed_total_exact": (
            result_count == expected_result
            and int(record.get("result_count", 0)) == expected_result
            and int(record.get("budget_total", -1)) == expected_result
            and sum(budgets) == expected_result
        ),
        "canonical_config_exact": dict(record.get("canonical_config", {}))
        == CANONICAL_DYTOK_CONFIG,
        "target_layers_exact": tuple(record.get("target_layers", ()))
        == tuple(range(16, 24)),
        "visionzip_signal_shapes_aligned": (
            attention_shape == (source_count,)
            and len(hidden_shape) == 2
            and int(hidden_shape[0]) == source_count
            and int(hidden_shape[1]) > 0
        ),
        "visionzip_signals_finite": bool(record.get("visionzip_signals_finite")),
        "temporal_slice_count_exact": (
            int(record.get("temporal_slice_count", 0)) == frame_count
            and len(frames) == frame_count
            and len(importance) == frame_count
            and len(provider_importance) == frame_count
            and len(budgets) == frame_count
        ),
        "provider_importance_preserved": (
            abs(float(record.get("provider_frame_importance_sum", 0.0)) - 1.0) <= 1e-3
            and float(record.get("provider_kernel_importance_max_abs_delta", 1.0))
            <= 1e-3
        ),
        "detector_frame_source_exact": record.get("detection_frame_source")
        == "native_llava_siglip_pixels",
        "detector_payload_boundary_exact": (
            record.get("detection_model_frozen") is True
            and record.get("detection_model_local_only") is True
            and record.get("conditioning_excludes_assistant_answer") is True
            and tuple(detection_attention.get("target_layers", ()))
            == tuple(range(16, 24))
            and detection_attention.get("query") == "last_prompt_token"
            and detection_attention.get("keys") == "visual_patch_tokens"
            and prompt_layout.get("single_visual_placeholder") is True
            and prompt_layout.get("assistant_answer_excluded") is True
        ),
        "detector_runtime_boundary_exact": (
            detector_call_delta == 1
            and detector_metadata.get("provider")
            == "frozen_llava_onevision_dytok_detector"
            and detector_metadata.get("prepared") is True
            and detector_metadata.get("frozen") is True
            and detector_metadata.get("local_files_only") is True
            and detector_metadata.get("assistant_answer_excluded") is True
            and detector_metadata.get("attention_query") == "last_prompt_token"
            and detector_metadata.get("attention_key_normalization")
            == "visual_keys_only"
            and tuple(detector_metadata.get("target_layers", ()))
            == tuple(range(16, 24))
            and detector_registered_as_trainable_child is False
        ),
    }


def _step_checks(
    *,
    loss: Any,
    logits: Any,
    records: Sequence[Mapping[str, Any]],
    algorithm_records: Sequence[Mapping[str, Any]],
    splice: Mapping[str, Any] | None,
    provider_capture_delta: int,
    visionzip_capture_delta: int,
    detector_call_delta: int,
    provider_metadata: Mapping[str, Any],
    detector_registered_as_trainable_child: bool,
    parameters: Sequence[Mapping[str, Any]],
    stable_layout: bool,
    frame_count: int,
    torch: Any,
) -> dict[str, bool]:
    source_count = int(records[0].get("source_count", 0)) if len(records) == 1 else 0
    result_count = int(records[0].get("result_count", 0)) if len(records) == 1 else 0
    algorithm = algorithm_records[0] if len(algorithm_records) == 1 else None
    sequence_representatives = (
        () if splice is None else splice.get("sequence_representative_indices", ())
    )
    checks = {
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
        "provider_capture_once": provider_capture_delta == 1,
        "visionzip_capture_once": visionzip_capture_delta == 1,
        "provider_video_only": bool(provider_metadata.get("video_only")),
        "provider_detector_frozen": bool(provider_metadata.get("detector_frozen")),
        "provider_detector_local_only": bool(
            provider_metadata.get("detector_local_only")
        ),
        "provider_answer_excluded": bool(
            provider_metadata.get("assistant_answer_excluded")
        ),
        "finite_loss": bool(loss is not None and torch.isfinite(loss).all().item()),
        "finite_logits": bool(
            logits is not None and torch.isfinite(logits).all().item()
        ),
        "label_logit_position_layout_aligned": bool(
            logits is not None
            and sequence_representatives
            and int(logits.shape[1]) == len(sequence_representatives)
        ),
        "layout_stable_across_steps": stable_layout,
        **_dytok_algorithm_checks(
            algorithm,
            source_count=source_count,
            result_count=result_count,
            frame_count=frame_count,
            detector_call_delta=detector_call_delta,
            detector_metadata=(
                provider_metadata.get("detector")
                if isinstance(provider_metadata.get("detector"), Mapping)
                else {}
            ),
            detector_registered_as_trainable_child=(
                detector_registered_as_trainable_child
            ),
        ),
        **target_parameter_checks(parameters),
    }
    return checks


def _run(args: argparse.Namespace) -> dict[str, Any]:
    visible = prepare_checkpoint_environment(args.physical_gpu)

    checkpoint = args.checkpoint.resolve()
    detector_checkpoint = args.detector_checkpoint.resolve()
    detector_siglip = args.detector_siglip.resolve()
    parquet_path = args.nextqa_parquet.resolve()
    video_root = args.video_root.resolve()
    output_dir = args.output_dir.resolve()
    progress_path = output_dir / "progress.jsonl"
    checkpoint_info = _checkpoint_record(checkpoint)
    detector_info = _checkpoint_record(detector_checkpoint)
    detector_siglip_info = _checkpoint_record(detector_siglip)
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
        detector_checkpoint,
        detector_siglip,
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
            provider_before = metadata_before["signal_provider"]
            capture_before = int(provider_before["capture_count"])
            detector_before = _detector_calls(provider_before)
            visionzip_before = _visionzip_captures(provider_before)
            loss, logits, parameter_evidence = training_forward_step(
                facade=facade, runtime=runtime, state=state, optimizer=optimizer,
                selected=selected, torch=torch,
            )
            metadata = activation.runtime_metadata()
            records = metadata["projector_layout_deltas"]
            algorithm_records = metadata["algorithm_records"]
            provider_metadata = dict(metadata["signal_provider"])
            splice = (metadata.get("bridge") or {}).get("latest_splice")
            capture_delta = int(provider_metadata["capture_count"]) - capture_before
            detector_call_delta = _detector_calls(provider_metadata) - detector_before
            visionzip_capture_delta = (
                _visionzip_captures(provider_metadata) - visionzip_before
            )
            current_layout = projector_layout_record(records, algorithm_records, splice)
            if first_layout is None:
                first_layout = current_layout
            checks = _step_checks(
                loss=loss,
                logits=logits,
                records=records,
                algorithm_records=algorithm_records,
                splice=splice,
                provider_capture_delta=capture_delta,
                visionzip_capture_delta=visionzip_capture_delta,
                detector_call_delta=detector_call_delta,
                provider_metadata=provider_metadata,
                detector_registered_as_trainable_child=bool(
                    metadata["detector_registered_as_trainable_child"]
                ),
                parameters=parameter_evidence,
                stable_layout=current_layout == first_layout,
                frame_count=frame_count,
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
            scope=SCOPE, canonical_config=CANONICAL_DYTOK_CONFIG,
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
        detector_checkpoint,
        detector_siglip,
        torch=torch,
        train=False,
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
        reload_provider = dict(reload_metadata["signal_provider"])
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
            "provider_capture_once": int(reload_provider["capture_count"]) == 1,
            "visionzip_capture_once": _visionzip_captures(reload_provider) == 1,
            "provider_video_only": bool(reload_provider.get("video_only")),
            "provider_detector_frozen": bool(reload_provider.get("detector_frozen")),
            "provider_detector_local_only": bool(
                reload_provider.get("detector_local_only")
            ),
            "provider_answer_excluded": bool(
                reload_provider.get("assistant_answer_excluded")
            ),
            "labels_absent": reload_state.labels is None,
            "finite_logits": bool(
                reload_logits is not None and torch.isfinite(reload_logits).all().item()
            ),
            "label_logit_position_layout_aligned": bool(
                reload_logits is not None
                and sequence_representatives
                and int(reload_logits.shape[1]) == len(sequence_representatives)
            ),
            **_dytok_algorithm_checks(
                reload_algorithm,
                source_count=source_count,
                result_count=result_count,
                frame_count=reload_frame_count,
                detector_call_delta=_detector_calls(reload_provider),
                detector_metadata=(
                    reload_provider.get("detector")
                    if isinstance(reload_provider.get("detector"), Mapping)
                    else {}
                ),
                detector_registered_as_trainable_child=bool(
                    reload_metadata["detector_registered_as_trainable_child"]
                ),
            ),
        }
        post_reload = {
            "checkpoint_reload": reload_result,
            "logits_shape": [int(item) for item in reload_logits.shape],
            "compression": canonical(reload_records[0]),
            "algorithm": canonical(reload_algorithm),
            "splice": canonical(reload_splice),
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
        canonical_config=CANONICAL_DYTOK_CONFIG,
        checkpoint=checkpoint_info,
        fixture_path=fixture_path,
        started=started,
        provenance={
            "detector_checkpoint": detector_info,
            "detector_siglip": detector_siglip_info,
        },
        boundaries={
            "detector_checkpoint_training_claim": False,
            "single_sample_multiple_videos": False,
        },
        environment={
            **training_environment_report(visible=visible, torch=torch, transformers=transformers),
            "detector_dtype": "float16",
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
            "detector_trainable": False,
            "detector_registered_as_trainable_child": False,
            "detector_in_lightweight_checkpoint": False,
            "auxiliary_siglip_trainable": False,
            "frame_count": frame_count,
        },
        lightweight_checkpoint=lightweight_checkpoint,
        post_reload_compressed_inference=post_reload,
        gpu=checkpoint_gpu_report(
            torch=torch,
            before=gpu_before,
            admission=admission,
            after=gpu_after,
            estimate_basis="DyToK LLaVA inference used the base checkpoint plus a separate "
            "frozen detector; the 48 GiB bound covers base-model autograd, "
            "detector attention, dynamic VisionZip, fresh reload, and variance",
        ),
    )


def _parser() -> argparse.ArgumentParser:
    parser = training_parser(estimated_peak_mib=49152)
    parser.add_argument("--detector_checkpoint", type=Path, required=True)
    parser.add_argument("--detector_siglip", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    return training_main(argv, parser=_parser(), run=_run, scope=SCOPE, gpu_snapshot=_gpu_snapshot)


if __name__ == "__main__":
    raise SystemExit(main())
