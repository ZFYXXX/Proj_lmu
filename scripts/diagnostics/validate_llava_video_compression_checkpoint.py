"""Validate one canonical LLaVA-OneVision video-compression inference cell.

The diagnostic is intentionally offline and single-GPU.  It runs the same four
fixed ``nextqa_mc_test`` documents once through the native baseline and once
through an explicitly installed video-compression binding, then writes both a
machine-readable evidence report and lmms-eval-shaped per-sample JSONL.

Only the canonical VQToken cell is admitted in this first end-to-end slice.
Additional algorithms must add their own exact resource and defining-stage
checks before being accepted here.
"""

from __future__ import annotations

import argparse
import shlex
import sys
import time
from typing import Any, Mapping, Sequence

from mllm_comp.backbones.llava_onevision import (
    LlavaOneVisionSample,
    create_llava_onevision_backbone,
)
from mllm_comp.pipeline.stage import CompressionStageConfig
from scripts.diagnostics.llava_provider_video_checkpoint_common import (
    generation_report as _generation_report,
)
from scripts.diagnostics.video_checkpoint_common import (
    CANONICAL_VQTOKEN_CONFIG,
    FIXED_NEXTQA_DOC_IDS,
    _DecoderInputAudit,
    _checkpoint_record,
    _gpu_snapshot,
    add_checkpoint_arguments,
    admit_generation_device,
    compare_generation,
    generation_activation,
    generation_case,
    generation_documents,
    generation_environment,
    checkpoint_gpu_report,
    generation_sample,
    inference_main,
    load_generation_runtime,
    prepare_checkpoint_environment,
    prepare_generation_state,
)


def _stage() -> CompressionStageConfig:
    return CompressionStageConfig.from_dict(
        {
            "name": "vqtoken-video-llava-onevision-real-checkpoint",
            "placement": "projector_level",
            "execution_phase": "generate",
            "plugin": "vqtoken",
            "algorithm": "vqtoken",
            "variant": "vqtoken.video",
            "target_modalities": ["video"],
            "conditioning_modalities": [],
            "preserved_modalities": ["text", "image", "audio"],
            "coupling_policy": "independent",
            "media_alignment": "spatiotemporal",
            "binding": "vqtoken.video.llava_onevision",
            "config": {},
        }
    )


def _validate_vqtoken_sample(
    *,
    baseline: Mapping[str, Any],
    compressed: Mapping[str, Any],
    activation_records: Sequence[Mapping[str, Any]],
    latest_splice: Mapping[str, Any],
    labels_absent: bool,
    placeholder_count: int,
) -> dict[str, bool]:
    if len(activation_records) != 1:
        raise RuntimeError(
            f"VQToken must publish one projector LayoutDelta per sample, got "
            f"{len(activation_records)}"
        )
    layout = activation_records[0]
    source_count = int(layout["source_count"])
    result_count = int(layout["result_count"])
    baseline_length = int(baseline["decoder"]["prefill"]["query_length"])
    compressed_length = int(compressed["decoder"]["prefill"]["query_length"])
    structural_source = tuple(latest_splice["structural_source_positions"])
    structural_result = tuple(latest_splice["structural_result_positions"])
    checks = {
        "compression_applied": bool(latest_splice["compression_applied"]),
        "non_identity_token_reduction": source_count > result_count,
        "canonical_fixed_k_result": result_count == 32,
        "layout_and_splice_counts_agree": (
            int(latest_splice["source_patch_count"]) == source_count
            and int(latest_splice["result_patch_count"]) == result_count
        ),
        "decoder_length_reduction_exact": (
            baseline_length - compressed_length == source_count - result_count
        ),
        "structural_tokens_preserved": len(structural_source) == len(structural_result),
        "one_video_placeholder": placeholder_count == 1,
        "inference_labels_absent": labels_absent,
        "baseline_logits_finite": bool(baseline["logits_finite"]),
        "compressed_logits_finite": bool(compressed["logits_finite"]),
        "baseline_decoder_alignment": all(baseline["decoder"]["invariants"].values()),
        "compressed_decoder_alignment": all(
            compressed["decoder"]["invariants"].values()
        ),
    }
    if not all(checks.values()):
        raise RuntimeError(f"VQToken sample evidence failed: {checks!r}")
    return checks


def _run(args: argparse.Namespace) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    visible = prepare_checkpoint_environment(args.physical_gpu)

    checkpoint = args.checkpoint.resolve()
    parquet_path = args.nextqa_parquet.resolve()
    video_root = args.video_root.resolve()

    checkpoint_info = _checkpoint_record(checkpoint)
    documents, videos = generation_documents(parquet_path, video_root)

    torch, transformers, gpu_before, admission = admit_generation_device(
        args,
        bridge="LLaVA",
    )
    started = time.time()
    facade = create_llava_onevision_backbone()
    processor, model, runtime = load_generation_runtime(
        checkpoint,
        facade=facade,
        torch=torch,
        dtype=torch.float16,
        device_map="cuda:0",
        metadata={"enable_talker": False, "speech_sample_rate": None},
    )
    handles = facade.model_adapter.structural_handles(model)
    first_layer = handles.get("llm.layer[0]")
    if first_layer is None:
        raise RuntimeError("LLaVA structural handle llm.layer[0] is missing")
    audit = _DecoderInputAudit(first_layer)
    activation = generation_activation(
        facade=facade,
        runtime=runtime,
        binding_name="vqtoken.video.llava_onevision",
        stage=_stage(),
    )
    if (
        dict(activation.runtime_metadata()["canonical_config"])
        != CANONICAL_VQTOKEN_CONFIG
    ):
        raise RuntimeError(
            "Resolved VQToken configuration differs from the canonical profile"
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

            with compare_generation(
                activation=activation,
                generate=_generation_report,
                facade=facade,
                runtime=runtime,
                state=state,
                batch=batch,
                audit=audit,
                torch=torch,
                max_new_tokens=args.max_new_tokens,
            ) as (baseline, compressed, before_metadata, metadata):
                before_invocations = int(before_metadata["projector_invocations"])
                if int(metadata["projector_invocations"]) != before_invocations + 1:
                    raise RuntimeError(
                        "VQToken projector invocation count did not increment once"
                    )
                records = metadata["projector_layout_deltas"]
                bridge_metadata = metadata["bridge"]
                latest_splice = bridge_metadata["latest_splice"]
                if latest_splice is None:
                    raise RuntimeError("LLaVA bridge published no splice evidence")
                checks = _validate_vqtoken_sample(
                    baseline=baseline,
                    compressed=compressed,
                    activation_records=records,
                    latest_splice=latest_splice,
                    labels_absent=labels_absent,
                    placeholder_count=placeholder_count,
                )
                case_activation = {
                    **metadata,
                    "projector_layout_deltas": records,
                }

            video_metadata = dict(batch.metadata["video"])
            case = {
                **generation_case(document, video),
                "native_video": {
                    "frame_count": int(video_metadata["frame_count"]),
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
                    "source_visual_tokens": records[0]["source_count"],
                    "result_visual_tokens": records[0]["result_count"],
                }
            )
    finally:
        if activation.installed:
            activation.uninstall()
        audit.close()

    torch.cuda.synchronize()
    gpu_after = _gpu_snapshot(args.physical_gpu)
    report = {
        "status": "passed",
        "scope": {
            "algorithm": "vqtoken",
            "variant": "vqtoken.video",
            "backbone": "llava_onevision",
            "binding": "vqtoken.video.llava_onevision",
            "execution_phase": "generate",
            "canonical_config": CANONICAL_VQTOKEN_CONFIG,
            "quality_claim": "functional_migration_only",
        },
        "provenance": {
            "code_revision": args.code_revision,
            "command": shlex.join(sys.argv),
            "checkpoint": checkpoint_info,
            "nextqa_parquet": str(parquet_path),
            "nextqa_parquet_size_bytes": parquet_path.stat().st_size,
            "fixed_doc_ids": list(FIXED_NEXTQA_DOC_IDS),
        },
        "environment": generation_environment(
            visible=visible,
            torch=torch,
            transformers=transformers,
            dtype="float16",
        ),
        "gpu": checkpoint_gpu_report(
            before=gpu_before,
            admission=admission,
            after=gpu_after,
            torch=torch,
        ),
        "structural_handles": sorted(handles),
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


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    add_checkpoint_arguments(parser, estimated_peak_mib=24576)
    parser.add_argument("--max_new_tokens", type=int, default=4)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.max_new_tokens < 2:
        raise RuntimeError("max_new_tokens must be at least 2")
    return inference_main(args, _run)


if __name__ == "__main__":
    raise SystemExit(main())
