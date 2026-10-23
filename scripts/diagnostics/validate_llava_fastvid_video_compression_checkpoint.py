"""Validate canonical FastVID video inference on LLaVA-OneVision.

The diagnostic is offline, batch-one, and single-GPU. It reuses the same four
fixed ``nextqa_mc_test`` documents and native baseline policy as the existing
LLaVA slices, while freezing FastVID's projected/raw/global/saliency signals
and exact DySeg, STPrune, and single-/multi-frame DTM decisions.
"""

from __future__ import annotations

import argparse
import shlex
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

from mllm_comp.backbones.llava_onevision import (
    LlavaOneVisionSample,
    create_llava_onevision_backbone,
)
from mllm_comp.compressors.video.config import VIDEO_REFERENCE_REVISION
from mllm_comp.pipeline.stage import CompressionStageConfig
from scripts.diagnostics import (
    llava_provider_video_checkpoint_common as _llava_provider_video_checkpoint_common,
)
from scripts.diagnostics.video_checkpoint_common import (
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

if __package__:
    from scripts.diagnostics import video_checkpoint_common as _common
else:
    import video_checkpoint_common as _common


FIXED_NEXTQA_DOC_IDS = _common.FIXED_NEXTQA_DOC_IDS
CANONICAL_FASTVID_CONFIG = {
    "retention_ratio": 0.10,
    "dyseg_c": 8,
    "dyseg_tau": 0.9,
    "stprune_d": 0.4,
    "dtm_p": 4,
    "dtm_beta": 0.6,
}


def _stage() -> CompressionStageConfig:
    return CompressionStageConfig.from_dict(
        {
            "name": "fastvid-video-llava-onevision-real-checkpoint",
            "placement": "projector_level",
            "execution_phase": "generate",
            "plugin": "fastvid",
            "algorithm": "fastvid",
            "variant": "fastvid.video",
            "target_modalities": ["video"],
            "conditioning_modalities": [],
            "preserved_modalities": ["text", "image", "audio"],
            "coupling_policy": "independent",
            "media_alignment": "spatiotemporal",
            "binding": "fastvid.video.llava_onevision",
            "config": {},
        }
    )


def _validate_fastvid_sample(
    *,
    baseline: Mapping[str, Any],
    compressed: Mapping[str, Any],
    activation_records: Sequence[Mapping[str, Any]],
    algorithm_records: Sequence[Mapping[str, Any]],
    latest_splice: Mapping[str, Any],
    labels_absent: bool,
    placeholder_count: int,
    frame_count: int,
    provider_capture_delta: int,
    provider_metadata: Mapping[str, Any],
) -> dict[str, bool]:
    if len(activation_records) != 1 or len(algorithm_records) != 1:
        raise RuntimeError(
            "FastVID must publish one layout and one algorithm record per sample"
        )
    layout = activation_records[0]
    algorithm = algorithm_records[0]
    source_count = int(layout["source_count"])
    result_count = int(layout["result_count"])
    baseline_length = int(baseline["decoder"]["prefill"]["query_length"])
    compressed_length = int(compressed["decoder"]["prefill"]["query_length"])
    structural_source = tuple(latest_splice["structural_source_positions"])
    structural_result = tuple(latest_splice["structural_result_positions"])
    expected_source = frame_count * 196
    checks = {
        "compression_applied": bool(latest_splice["compression_applied"]),
        "non_identity_token_reduction": source_count > result_count,
        "native_projected_frame_raster_exact": source_count == expected_source,
        "canonical_frame_count_exact": frame_count == 32,
        "layout_and_splice_counts_agree": (
            int(latest_splice["source_patch_count"]) == source_count
            and int(latest_splice["result_patch_count"]) == result_count
            and int(algorithm["source_count"]) == source_count
            and int(algorithm["result_count"]) == result_count
        ),
        "decoder_length_reduction_exact": (
            baseline_length - compressed_length == source_count - result_count
        ),
        "structural_tokens_preserved": len(structural_source) == len(structural_result),
        "one_video_placeholder": placeholder_count == 1,
        "inference_labels_absent": labels_absent,
        "provider_capture_once": provider_capture_delta == 1,
        "provider_pooling_head_loaded": bool(
            provider_metadata["siglip_pooling_head_loaded"]
        ),
        "provider_video_only": bool(provider_metadata["video_only"]),
        "provider_raw_stream_attached": bool(provider_metadata["raw_stream_attached"]),
        "canonical_config_exact": dict(algorithm["canonical_config"])
        == CANONICAL_FASTVID_CONFIG,
        "projected_signal_shape_exact": (
            tuple(algorithm["projected_features_shape"])[0] == source_count
            and len(tuple(algorithm["projected_features_shape"])) == 2
            and tuple(algorithm["projected_features_shape"])[1] > 0
        ),
        "raw_signal_shape_exact": (
            tuple(algorithm["raw_features_shape"])[0]
            == frame_count * int(algorithm["raw_tokens_per_frame"])
            and int(algorithm["raw_tokens_per_frame"]) == 729
            and len(tuple(algorithm["raw_features_shape"])) == 2
            and tuple(algorithm["raw_features_shape"])[1] > 0
        ),
        "frame_global_signal_shape_exact": (
            tuple(algorithm["frame_global_features_shape"])[0] == frame_count
            and len(tuple(algorithm["frame_global_features_shape"])) == 2
            and tuple(algorithm["frame_global_features_shape"])[1] > 0
        ),
        "saliency_signal_shape_exact": tuple(
            algorithm["token_saliency_attention_shape"]
        )
        == (source_count,),
        "all_runtime_signals_finite": all(
            bool(algorithm[name])
            for name in (
                "projected_features_finite",
                "raw_features_finite",
                "frame_global_features_finite",
                "token_saliency_attention_finite",
            )
        ),
        "representatives_source_ordered": bool(
            algorithm["representatives_source_ordered"]
        ),
        "baseline_logits_finite": bool(baseline["logits_finite"]),
        "compressed_logits_finite": bool(compressed["logits_finite"]),
        "baseline_decoder_alignment": all(baseline["decoder"]["invariants"].values()),
        "compressed_decoder_alignment": all(
            compressed["decoder"]["invariants"].values()
        ),
    }
    if not all(checks.values()):
        raise RuntimeError(f"FastVID LLaVA sample evidence failed: {checks!r}")
    return checks


def _run(args: argparse.Namespace) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    visible = prepare_checkpoint_environment(args.physical_gpu)

    checkpoint = args.checkpoint.resolve()
    siglip_checkpoint = args.siglip_checkpoint.resolve()
    parquet_path = args.nextqa_parquet.resolve()
    video_root = args.video_root.resolve()

    checkpoint_info = _common._checkpoint_record(checkpoint)
    siglip_info = _common._checkpoint_record(siglip_checkpoint)
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
        metadata={
            "enable_talker": False,
            "speech_sample_rate": None,
            "fastvid_siglip_path": str(siglip_checkpoint),
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
        binding_name="fastvid.video.llava_onevision",
        stage=_stage(),
    )
    if (
        dict(activation.runtime_metadata()["canonical_config"])
        != CANONICAL_FASTVID_CONFIG
    ):
        raise RuntimeError(
            "Resolved FastVID configuration differs from the canonical profile"
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
                generate=_llava_provider_video_checkpoint_common.generation_report,
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
                        "FastVID projector invocation count did not increment once"
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
                checks = _validate_fastvid_sample(
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
            "algorithm": "fastvid",
            "variant": "fastvid.video",
            "backbone": "llava_onevision",
            "binding": "fastvid.video.llava_onevision",
            "execution_phase": "generate",
            "canonical_config": CANONICAL_FASTVID_CONFIG,
            "quality_claim": "functional_migration_only",
        },
        "provenance": {
            "code_revision": args.code_revision,
            "reference_revision": VIDEO_REFERENCE_REVISION,
            "command": shlex.join(sys.argv),
            "checkpoint": checkpoint_info,
            "siglip_checkpoint": siglip_info,
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
    add_checkpoint_arguments(parser, estimated_peak_mib=32768)
    parser.add_argument("--siglip_checkpoint", type=Path, required=True)
    parser.add_argument("--max_new_tokens", type=int, default=4)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.max_new_tokens < 2:
        raise RuntimeError("max_new_tokens must be at least 2")
    return inference_main(args, _run)


if __name__ == "__main__":
    raise SystemExit(main())
