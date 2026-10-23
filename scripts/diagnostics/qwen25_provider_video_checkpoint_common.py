"""Shared real-checkpoint runner for provider-owned Qwen2.5-VL compressors.

LongVU and LLaMA-VID attach a local auxiliary model plus a trainable
compressor before the native Qwen2.5-VL video splice.  This module owns their
shared fixed-four baseline/compressed execution, GPU admission, native grid and
M-RoPE evidence, artifact contract, and fail-closed lifecycle.  Each algorithm
supplies only its exact defining-stage assertions.
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

from mllm_comp.backbones.qwen25_vl import Qwen25VLSample, create_qwen25_vl_backbone
from mllm_comp.compressors.video.config import VIDEO_REFERENCE_REVISION
from mllm_comp.pipeline.stage import CompressionStageConfig
from scripts.diagnostics import qwen_video_checkpoint_common as _qwen_common
from scripts.diagnostics import video_checkpoint_common as _video_checkpoint_common
from scripts.diagnostics.qwen_video_checkpoint_common import qwen25_video_input_record
from scripts.diagnostics.video_checkpoint_common import (
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

FIXED_NEXTQA_DOC_IDS = _video_checkpoint_common.FIXED_NEXTQA_DOC_IDS


@dataclass(frozen=True)
class Qwen25ProviderVideoDiagnosticProfile:
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


def _conversation(
    *,
    prompt: str,
    video: Path,
    fps: float,
    min_pixels: int,
    max_pixels: int,
) -> list[dict[str, Any]]:
    if min_pixels <= 0 or max_pixels <= 0:
        raise ValueError("Qwen2.5-VL video pixel bounds must be positive")
    if min_pixels > max_pixels:
        raise ValueError("Qwen2.5-VL video min_pixels cannot exceed max_pixels")
    conversation = _qwen_common._conversation(
        prompt=prompt,
        video=video,
        fps=fps,
        max_pixels=max_pixels,
    )
    conversation[1]["content"][0]["min_pixels"] = int(min_pixels)
    return conversation


def run(
    args: argparse.Namespace,
    *,
    profile: Qwen25ProviderVideoDiagnosticProfile,
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

    checkpoint_info = _video_checkpoint_common._checkpoint_record(checkpoint)
    auxiliary_info = _video_checkpoint_common._checkpoint_record(auxiliary_checkpoint)
    documents, videos = generation_documents(parquet_path, video_root)

    torch, transformers, gpu_before, admission = admit_generation_device(
        args,
        bridge="Qwen2.5-VL",
    )
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.cuda.reset_peak_memory_stats()
    started = time.time()

    facade = create_qwen25_vl_backbone()
    processor, model, runtime = load_generation_runtime(
        checkpoint,
        facade=facade,
        torch=torch,
        dtype=torch.bfloat16,
        device_map={"": 0},
        metadata={
            "enable_talker": False,
            "speech_sample_rate": None,
            profile.auxiliary_runtime_key: str(auxiliary_checkpoint),
        },
    )
    handles = facade.model_adapter.structural_handles(model)
    first_layer = handles.get("llm.layer[0]")
    if first_layer is None:
        raise RuntimeError("Qwen2.5-VL structural handle llm.layer[0] is missing")
    audit = _video_checkpoint_common._DecoderInputAudit(first_layer)
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
            conversation = _conversation(
                prompt=document["prompt"],
                video=video,
                fps=args.video_fps,
                min_pixels=args.video_min_pixels,
                max_pixels=args.video_max_pixels,
            )
            sample = Qwen25VLSample(
                sample_id=f"nextqa_mc_test:{doc_id}",
                conversation=conversation,
            )
            batch, state = prepare_generation_state(
                facade=facade,
                runtime=runtime,
                processor=processor,
                sample=sample,
                bridge="Qwen2.5-VL",
            )
            native_video = qwen25_video_input_record(state, batch)

            with compare_generation(
                activation=activation,
                generate=_qwen_common.generation_report,
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
                latest_generation = bridge_metadata["latest_generation"]
                if latest_splice is None or latest_generation is None:
                    raise RuntimeError(
                        "Qwen2.5-VL bridge published incomplete splice/generation evidence"
                    )
                checks = profile.sample_validator(
                    baseline=baseline,
                    compressed=compressed,
                    activation_records=layout_records,
                    algorithm_records=algorithm_records,
                    latest_splice=latest_splice,
                    latest_generation=latest_generation,
                    source_grid_thw=native_video["video_grid_thw"],
                    source_grid_token_count=native_video["source_grid_token_count"],
                    merge_size=native_video["spatial_merge_size"],
                    labels_absent=native_video["labels_absent_for_inference"],
                    video_span_count=1,
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
                    **native_video,
                    "conversation": conversation,
                    "requested_fps": float(args.video_fps),
                    "requested_min_pixels": int(args.video_min_pixels),
                    "requested_max_pixels": int(args.video_max_pixels),
                },
                "baseline": baseline,
                "compressed": compressed,
                "activation": case_activation,
                "bridge_latest_splice": latest_splice,
                "bridge_latest_generation": latest_generation,
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
    gpu_after = _video_checkpoint_common._gpu_snapshot(args.physical_gpu)
    report = {
        "status": "passed",
        "scope": {
            "algorithm": profile.algorithm,
            "variant": profile.variant,
            "backbone": "qwen25_vl",
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
            "video_preprocessing": {
                "fps": float(args.video_fps),
                "min_pixels": int(args.video_min_pixels),
                "max_pixels": int(args.video_max_pixels),
                "provider": "qwen_vl_utils.process_vision_info",
            },
        },
        "environment": {
            **generation_environment(
                visible=visible,
                torch=torch,
                transformers=transformers,
                dtype="bfloat16",
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
    profile: Qwen25ProviderVideoDiagnosticProfile,
) -> int:
    if args.max_new_tokens < 2:
        raise RuntimeError("max_new_tokens must be at least 2")
    if args.video_fps <= 0:
        raise RuntimeError("video_fps must be positive")
    if args.video_min_pixels <= 0 or args.video_max_pixels <= 0:
        raise RuntimeError("video pixel bounds must be positive")
    if args.video_min_pixels > args.video_max_pixels:
        raise RuntimeError("video_min_pixels cannot exceed video_max_pixels")
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
                "backbone": "qwen25_vl",
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
            failure["gpu_after"] = _video_checkpoint_common._gpu_snapshot(args.physical_gpu)
        except Exception:
            pass
        _video_checkpoint_common._atomic_write_json(report_path, failure)
        raise
    _video_checkpoint_common._atomic_write_jsonl(samples_path, sample_rows)
    report["artifacts"] = {
        "report": str(report_path),
        "lmms_eval_samples": str(samples_path),
    }
    _video_checkpoint_common._atomic_write_json(report_path, report)
    print(json.dumps({"status": report["status"], "report": str(report_path)}))
    return 0


def add_common_arguments(
    parser: argparse.ArgumentParser,
    *,
    video_min_pixels: int,
    video_max_pixels: int,
    estimated_peak_mib: int,
) -> None:
    add_checkpoint_arguments(parser, estimated_peak_mib=estimated_peak_mib)
    parser.add_argument("--max_new_tokens", type=int, default=4)
    parser.add_argument("--video_fps", type=float, default=1.0)
    parser.add_argument("--video_min_pixels", type=int, default=video_min_pixels)
    parser.add_argument("--video_max_pixels", type=int, default=video_max_pixels)
    parser.add_argument("--seed", type=int, default=42)


__all__ = [
    "FIXED_NEXTQA_DOC_IDS",
    "Qwen25ProviderVideoDiagnosticProfile",
    "add_common_arguments",
    "main",
    "run",
]
