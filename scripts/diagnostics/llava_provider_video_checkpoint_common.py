"""Shared real-checkpoint runner for provider-owned LLaVA video compressors.

LongVU and LLaMA-VID both attach an auxiliary local model and a trainable
compressor before the native LLaVA video splice.  This module owns only their
shared four-sample baseline/compressed execution, GPU admission, artifact, and
failure contracts.  Each algorithm supplies exact defining-stage validators.
"""

from __future__ import annotations

import argparse
import json
import shlex
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping

from mllm_comp.backbones.base import BackboneRuntime, TraceRequest
from mllm_comp.backbones.llava_onevision import (
    LlavaOneVisionSample,
    create_llava_onevision_backbone,
)
from mllm_comp.compressors.video.config import VIDEO_REFERENCE_REVISION
from mllm_comp.pipeline.stage import CompressionStageConfig
from scripts.diagnostics import video_checkpoint_common as _common
from scripts.diagnostics.video_checkpoint_common import (
    _DecoderInputAudit,
    _decoder_summary,
    _parse_choice,
    _shape,
    _tensor_rows,
    add_checkpoint_arguments,
    admit_generation_device,
    compare_generation,
    generation_activation,
    generation_case,
    generation_documents,
    generation_environment,
    generation_sample,
    load_generation_runtime,
    prepare_checkpoint_environment,
    prepare_generation_state,
)

FIXED_NEXTQA_DOC_IDS = _common.FIXED_NEXTQA_DOC_IDS


@dataclass(frozen=True)
class LlavaProviderVideoDiagnosticProfile:
    algorithm: str
    variant: str
    binding: str
    canonical_config: Mapping[str, Any]
    auxiliary_argument: str
    auxiliary_runtime_key: str
    auxiliary_report_key: str
    auxiliary_label: str
    stage_factory: Callable[[], CompressionStageConfig]
    sample_validator: Callable[..., dict[str, bool]]
    defining_stages: tuple[str, ...]
    estimate_basis: str


def run(
    args: argparse.Namespace,
    *,
    profile: LlavaProviderVideoDiagnosticProfile,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    visible = prepare_checkpoint_environment(args.physical_gpu)

    checkpoint = args.checkpoint.resolve()
    auxiliary_checkpoint = Path(getattr(args, profile.auxiliary_argument)).resolve()
    parquet_path = args.nextqa_parquet.resolve()
    video_root = args.video_root.resolve()
    if not auxiliary_checkpoint.is_dir():
        raise RuntimeError(
            f"{profile.auxiliary_label} directory is missing: {auxiliary_checkpoint}"
        )

    checkpoint_info = _common._checkpoint_record(checkpoint)
    auxiliary_info = _common._checkpoint_record(auxiliary_checkpoint)
    documents, videos = generation_documents(parquet_path, video_root)

    torch, transformers, gpu_before, admission = admit_generation_device(
        args,
        bridge="LLaVA",
    )
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.cuda.reset_peak_memory_stats()
    started = time.time()

    facade = create_llava_onevision_backbone()
    processor, model, runtime = load_generation_runtime(
        checkpoint,
        facade=facade,
        torch=torch,
        dtype=torch.float16,
        device_map="cuda:0",
        metadata={
            "enable_talker": False,
            "speech_sample_rate": None,
            profile.auxiliary_runtime_key: str(auxiliary_checkpoint),
        },
    )
    handles = facade.model_adapter.structural_handles(model)
    first_layer = handles.get("llm.layer[0]")
    if first_layer is None:
        raise RuntimeError("LLaVA structural handle llm.layer[0] is missing")
    audit = _common._DecoderInputAudit(first_layer)
    activation = generation_activation(
        facade=facade,
        runtime=runtime,
        binding_name=profile.binding,
        stage=profile.stage_factory(),
    )
    if dict(activation.runtime_metadata()["canonical_config"]) != dict(
        profile.canonical_config
    ):
        raise RuntimeError(
            f"Resolved {profile.auxiliary_label} configuration differs from canonical"
        )

    cases: dict[str, Any] = {}
    sample_rows: list[dict[str, Any]] = []
    try:
        for document in documents:
            doc_id = int(document["doc_id"])
            video = videos[document["video"]]
            sample = LlavaOneVisionSample(
                sample_id=f"nextqa_mc_test:{doc_id}",
                prompt=document["prompt"],
                video=str(video),
            )
            batch, state = prepare_generation_state(
                facade=facade,
                runtime=runtime,
                processor=processor,
                sample=sample,
                bridge="LLaVA",
            )
            labels_absent = state.labels is None and (
                "labels" not in state.native_model_inputs
                or state.native_model_inputs["labels"] is None
            )
            placeholder_count = len(batch.metadata["video_placeholder_positions"])
            video_metadata = dict(batch.metadata["video"])
            frame_count = int(video_metadata["frame_count"])

            with compare_generation(
                activation=activation,
                generate=generation_report,
                facade=facade,
                runtime=runtime,
                state=state,
                batch=batch,
                audit=audit,
                torch=torch,
                max_new_tokens=args.max_new_tokens,
            ) as (baseline, compressed, before_metadata, metadata):
                before_invocations = int(before_metadata["projector_invocations"])
                before_captures = int(
                    before_metadata["signal_provider"]["capture_count"]
                )
                if int(metadata["projector_invocations"]) != before_invocations + 1:
                    raise RuntimeError(
                        f"{profile.algorithm} projector invocation did not increment once"
                    )
                layout_records = metadata["projector_layout_deltas"]
                algorithm_records = metadata["algorithm_records"]
                provider_metadata = dict(metadata["signal_provider"])
                provider_capture_delta = (
                    int(provider_metadata["capture_count"]) - before_captures
                )
                bridge_metadata = metadata["bridge"]
                latest_splice = bridge_metadata["latest_splice"]
                if latest_splice is None:
                    raise RuntimeError("LLaVA bridge published no splice evidence")
                checks = profile.sample_validator(
                    baseline=baseline,
                    compressed=compressed,
                    activation_records=layout_records,
                    algorithm_records=algorithm_records,
                    latest_splice=latest_splice,
                    labels_absent=labels_absent,
                    placeholder_count=placeholder_count,
                    frame_count=frame_count,
                    provider_capture_delta=provider_capture_delta,
                    provider_metadata=provider_metadata,
                    activation_metadata=metadata,
                    auxiliary_checkpoint=auxiliary_checkpoint,
                    document=document,
                    batch=batch,
                    state=state,
                )
                case_activation = {
                    **metadata,
                    "projector_layout_deltas": layout_records,
                    "algorithm_records": algorithm_records,
                }

            case = {
                **generation_case(document, video),
                "native_video": {
                    "frame_count": frame_count,
                    "total_frames": int(video_metadata["total_frames"]),
                    "fps": float(video_metadata["fps"]),
                    "frame_indices": list(video_metadata["frame_indices"]),
                    "timestamps_seconds": list(video_metadata["timestamps_seconds"]),
                    "frame_size": list(video_metadata["frame_size"]),
                    "decode_policy": video_metadata["decode_policy"],
                    "modalities": list(batch.metadata["modalities"]),
                    "placeholder_positions": list(
                        batch.metadata["video_placeholder_positions"]
                    ),
                    "labels_absent_for_inference": labels_absent,
                },
                "baseline": baseline,
                "compressed": compressed,
                "activation": case_activation,
                "bridge_latest_splice": latest_splice,
                "checks": checks,
            }
            cases[str(doc_id)] = case
            sample_rows.append(
                {
                    **generation_sample(document, baseline, compressed),
                    "compression_applied": checks["compression_applied"],
                    "source_visual_tokens": layout_records[0]["source_count"],
                    "result_visual_tokens": layout_records[0]["result_count"],
                }
            )
    finally:
        if activation.installed:
            activation.uninstall()
        audit.close()

    torch.cuda.synchronize()
    gpu_after = _common._gpu_snapshot(args.physical_gpu)
    report = {
        "status": "passed",
        "scope": {
            "algorithm": profile.algorithm,
            "variant": profile.variant,
            "backbone": "llava_onevision",
            "binding": profile.binding,
            "execution_phase": "generate",
            "canonical_config": dict(profile.canonical_config),
            "quality_claim": "functional_migration_only",
        },
        "provenance": {
            "code_revision": args.code_revision,
            "reference_revision": VIDEO_REFERENCE_REVISION,
            "command": shlex.join(sys.argv),
            "checkpoint": checkpoint_info,
            profile.auxiliary_report_key: auxiliary_info,
            "nextqa_parquet": str(parquet_path),
            "nextqa_parquet_size_bytes": parquet_path.stat().st_size,
            "fixed_doc_ids": list(FIXED_NEXTQA_DOC_IDS),
        },
        "environment": {
            **generation_environment(
                visible=visible,
                torch=torch,
                transformers=transformers,
                dtype="float16",
            ),
            "attention_implementation": "sdpa",
        },
        "gpu": {
            "before": gpu_before,
            "admission": admission,
            "estimate_basis": profile.estimate_basis,
            "observed_peak_allocated_mib": round(
                torch.cuda.max_memory_allocated(0) / (1024**2), 3
            ),
            "observed_peak_reserved_mib": round(
                torch.cuda.max_memory_reserved(0) / (1024**2), 3
            ),
            "after": gpu_after,
            "other_processes_modified": False,
        },
        "structural_handles": sorted(handles),
        "defining_stages": list(profile.defining_stages),
        "samples": cases,
        "sample_count": len(cases),
        "elapsed_seconds": round(time.time() - started, 3),
        "support_activation_claimed": False,
        "boundaries": {
            "full_nextqa_benchmark": False,
            "paper_quality_or_efficiency_reproduction": False,
            "convergence_claim": False,
        },
    }
    if tuple(int(item) for item in cases) != FIXED_NEXTQA_DOC_IDS:
        raise RuntimeError("The completed sample set differs from the fixed manifest")
    return report, sample_rows


def main(
    args: argparse.Namespace,
    *,
    profile: LlavaProviderVideoDiagnosticProfile,
) -> int:
    if args.max_new_tokens < 2:
        raise RuntimeError("max_new_tokens must be at least 2")
    output_dir = args.output_dir.resolve()
    report_path = output_dir / "report.json"
    samples_path = output_dir / "samples.jsonl"
    if report_path.exists() or samples_path.exists():
        raise RuntimeError(
            f"Refusing to overwrite existing evidence below {output_dir}"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()
    try:
        report, sample_rows = run(args, profile=profile)
    except BaseException as exc:
        failure = {
            "status": "failed",
            "scope": {
                "algorithm": profile.algorithm,
                "variant": profile.variant,
                "backbone": "llava_onevision",
                "binding": profile.binding,
                "execution_phase": "generate",
            },
            "error_type": type(exc).__name__,
            "error": str(exc),
            "command": shlex.join(sys.argv),
            "code_revision": args.code_revision,
            "physical_gpu": args.physical_gpu,
            "elapsed_seconds": round(time.time() - started, 3),
        }
        try:
            import torch

            if torch.cuda.is_available() and torch.cuda.device_count() == 1:
                failure["observed_peak_allocated_mib"] = round(
                    torch.cuda.max_memory_allocated() / (1024**2), 3
                )
                failure["observed_peak_reserved_mib"] = round(
                    torch.cuda.max_memory_reserved() / (1024**2), 3
                )
        except Exception:
            pass
        try:
            failure["gpu_after"] = _common._gpu_snapshot(args.physical_gpu)
        except Exception:
            pass
        _common._atomic_write_json(report_path, failure)
        raise
    _common._atomic_write_jsonl(samples_path, sample_rows)
    report["artifacts"] = {
        "report": str(report_path),
        "lmms_eval_samples": str(samples_path),
    }
    _common._atomic_write_json(report_path, report)
    print(json.dumps({"status": report["status"], "report": str(report_path)}))
    return 0


def add_common_arguments(parser: argparse.ArgumentParser) -> None:
    add_checkpoint_arguments(parser, estimated_peak_mib=32768)
    parser.add_argument("--max_new_tokens", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)


__all__ = [
    "FIXED_NEXTQA_DOC_IDS",
    "LlavaProviderVideoDiagnosticProfile",
    "add_common_arguments",
    "main",
    "run",
]


def generation_report(
    *,
    facade: Any,
    runtime: BackboneRuntime,
    state: Any,
    batch: Any,
    audit: _DecoderInputAudit,
    torch: Any,
    max_new_tokens: int,
) -> dict[str, Any]:
    mark = audit.mark()
    started = time.time()
    result = facade.runner.generate(
        runtime,
        state,
        processed_batch=batch,
        trace_request=TraceRequest(text_logits=True),
        max_new_tokens=max_new_tokens,
        do_sample=False,
    )
    torch.cuda.synchronize()
    if result.text is None:
        raise RuntimeError("LLaVA generation returned no text branch")
    token_rows = _tensor_rows(result.text.token_ids)
    if len(token_rows) != 1 or not 1 <= len(token_rows[0]) <= max_new_tokens:
        raise RuntimeError(
            "Generated token count is outside the finite request bound: "
            f"rows={token_rows!r}, max_new_tokens={max_new_tokens}"
        )
    traces = result.traces
    logits = None if traces is None else traces.text_logits
    if logits is None or not bool(torch.isfinite(logits).all().item()):
        raise RuntimeError("Generated text logits are missing or non-finite")
    text_rows = list(result.text.text)
    if len(text_rows) != 1:
        raise RuntimeError(f"Expected one decoded result, got {text_rows!r}")
    return {
        "generated_token_ids": token_rows[0],
        "generated_token_count": len(token_rows[0]),
        "text": text_rows[0],
        "parsed_choice": _parse_choice(text_rows[0]),
        "logits_shape": _shape(logits),
        "logits_finite": True,
        "decoder": _decoder_summary(audit.since(mark)),
        "elapsed_seconds": round(time.time() - started, 3),
    }
