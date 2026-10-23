"""Validate canonical DyToK video inference on LLaVA-OneVision.

The diagnostic is offline, batch-one, and single-GPU.  It loads a separate
frozen local LLaVA-OneVision detector, obtains layers-16-through-23 visual-only
attention for each of the same four fixed NExT-QA samples, and requires the
resulting dynamic VisionZip budgets and native LLaVA splice to be exact.
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
CANONICAL_DYTOK_CONFIG = {
    "mode": "dynamic",
    "base_compressor": "visionzip",
    "base_config": {
        "dominant_num": 126,
        "contextual_num": 21,
        "attn_layer": -2,
        "hidden_state_layer": -2,
    },
    "llm_config": {"target_layers": tuple(range(16, 24))},
    "upper_limit": 196,
}


def _stage() -> CompressionStageConfig:
    return CompressionStageConfig.from_dict(
        {
            "name": "dytok-video-llava-onevision-real-checkpoint",
            "placement": "projector_level",
            "execution_phase": "generate",
            "plugin": "dytok",
            "algorithm": "dytok",
            "variant": "dytok.video",
            "target_modalities": ["video"],
            "conditioning_modalities": ["text"],
            "preserved_modalities": ["text", "image", "audio"],
            "coupling_policy": "independent",
            "media_alignment": "spatiotemporal",
            "binding": "dytok.video.llava_onevision",
            "config": {},
        }
    )


def _detector_calls(metadata: Mapping[str, Any]) -> int:
    provider = metadata.get("signal_provider")
    detector = provider.get("detector") if isinstance(provider, Mapping) else None
    return int(detector.get("calls", 0)) if isinstance(detector, Mapping) else 0


def _validate_dytok_sample(
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
    detector_call_delta: int,
    detector_metadata: Mapping[str, Any],
    detector_registered_as_trainable_child: bool,
) -> dict[str, bool]:
    if len(activation_records) != 1 or len(algorithm_records) != 1:
        raise RuntimeError(
            "DyToK must publish one layout and one algorithm record per sample"
        )
    layout = activation_records[0]
    algorithm = algorithm_records[0]
    source_count = int(layout["source_count"])
    result_count = int(layout["result_count"])
    baseline_length = int(baseline["decoder"]["prefill"]["query_length"])
    compressed_length = int(compressed["decoder"]["prefill"]["query_length"])
    structural_source = tuple(latest_splice["structural_source_positions"])
    structural_result = tuple(latest_splice["structural_result_positions"])
    frame_summaries = tuple(algorithm["frame_summaries"])
    importance = tuple(float(value) for value in algorithm["frame_importance"])
    provider_importance = tuple(
        float(value) for value in algorithm["provider_frame_importance"]
    )
    budgets = tuple(int(value) for value in algorithm["total_budgets"])
    expected_source = frame_count * 196
    expected_result = min(frame_count * 147, expected_source)
    checks = {
        "compression_applied": bool(latest_splice["compression_applied"]),
        "non_identity_token_reduction": source_count > result_count,
        "native_frame_raster_count_exact": source_count == expected_source,
        "canonical_fixed_total_exact": (
            result_count == expected_result
            and int(algorithm["budget_total"]) == expected_result
            and sum(budgets) == expected_result
        ),
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
        "detector_called_once": detector_call_delta == 1,
        "canonical_config_exact": dict(algorithm["canonical_config"])
        == CANONICAL_DYTOK_CONFIG,
        "target_layers_exact": tuple(algorithm["target_layers"])
        == tuple(range(16, 24)),
        "visionzip_signal_shapes_aligned": (
            tuple(algorithm["attention_shape"]) == (source_count,)
            and len(tuple(algorithm["hidden_state_shape"])) == 2
            and int(tuple(algorithm["hidden_state_shape"])[0]) == source_count
            and int(tuple(algorithm["hidden_state_shape"])[1]) > 0
        ),
        "visionzip_signals_finite": bool(algorithm["visionzip_signals_finite"]),
        "temporal_slice_count_exact": (
            int(algorithm["temporal_slice_count"]) == frame_count
            and len(frame_summaries) == frame_count
            and len(importance) == frame_count
            and len(budgets) == frame_count
        ),
        "provider_importance_preserved": (
            len(provider_importance) == frame_count
            and abs(float(algorithm["provider_frame_importance_sum"]) - 1.0) <= 1e-3
            and float(algorithm["provider_kernel_importance_max_abs_delta"]) <= 1e-3
        ),
        "detector_frame_source_exact": algorithm["detection_frame_source"]
        == "native_llava_siglip_pixels",
        "detector_payload_boundary_exact": (
            algorithm["detection_model_frozen"] is True
            and algorithm["detection_model_local_only"] is True
            and algorithm["conditioning_excludes_assistant_answer"] is True
            and algorithm["detection_attention"]["target_layers"]
            == tuple(range(16, 24))
            and algorithm["detection_attention"]["query"] == "last_prompt_token"
            and algorithm["detection_attention"]["keys"] == "visual_patch_tokens"
            and algorithm["prompt_layout"]["single_visual_placeholder"] is True
            and algorithm["prompt_layout"]["assistant_answer_excluded"] is True
        ),
        "detector_runtime_boundary_exact": (
            detector_metadata.get("provider") == "frozen_llava_onevision_dytok_detector"
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
        "baseline_logits_finite": bool(baseline["logits_finite"]),
        "compressed_logits_finite": bool(compressed["logits_finite"]),
        "baseline_decoder_alignment": all(baseline["decoder"]["invariants"].values()),
        "compressed_decoder_alignment": all(
            compressed["decoder"]["invariants"].values()
        ),
        "baseline_bounded_decode": len(baseline["decoder"]["decode_query_lengths"])
        >= 1,
        "compressed_bounded_decode": len(compressed["decoder"]["decode_query_lengths"])
        >= 1,
    }
    if not all(checks.values()):
        raise RuntimeError(f"DyToK LLaVA sample evidence failed: {checks!r}")
    return checks


def _run(args: argparse.Namespace) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    visible = prepare_checkpoint_environment(args.physical_gpu)

    checkpoint = args.checkpoint.resolve()
    detector_checkpoint = args.detector_checkpoint.resolve()
    detector_siglip = args.detector_siglip.resolve()
    parquet_path = args.nextqa_parquet.resolve()
    video_root = args.video_root.resolve()

    checkpoint_info = _common._checkpoint_record(checkpoint)
    detector_info = _common._checkpoint_record(detector_checkpoint)
    siglip_info = _common._checkpoint_record(detector_siglip)
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
            "dytok_detection_model_path": str(detector_checkpoint),
            "dytok_detection_siglip_path": str(detector_siglip),
            "dytok_detection_device_map": "cuda:0",
            "dytok_detection_dtype": torch.float16,
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
        binding_name="dytok.video.llava_onevision",
        stage=_stage(),
    )
    if (
        dict(activation.runtime_metadata()["canonical_config"])
        != CANONICAL_DYTOK_CONFIG
    ):
        raise RuntimeError(
            "Resolved DyToK configuration differs from the canonical profile"
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
                before_detector_calls = _detector_calls(before_metadata)
                if int(metadata["projector_invocations"]) != before_invocations + 1:
                    raise RuntimeError(
                        "DyToK projector invocation count did not increment once"
                    )
                layout_records = metadata["projector_layout_deltas"]
                algorithm_records = metadata["algorithm_records"]
                provider = metadata["signal_provider"]
                provider_capture_delta = (
                    int(provider["capture_count"]) - before_captures
                )
                detector_call_delta = _detector_calls(metadata) - before_detector_calls
                detector_metadata = provider["detector"]
                if not isinstance(detector_metadata, Mapping):
                    raise RuntimeError("DyToK detector published no runtime evidence")
                bridge_metadata = metadata["bridge"]
                latest_splice = bridge_metadata["latest_splice"]
                if latest_splice is None:
                    raise RuntimeError("LLaVA bridge published no splice evidence")
                checks = _validate_dytok_sample(
                    baseline=baseline,
                    compressed=compressed,
                    activation_records=layout_records,
                    algorithm_records=algorithm_records,
                    latest_splice=latest_splice,
                    labels_absent=labels_absent,
                    placeholder_count=placeholder_count,
                    frame_count=frame_count,
                    provider_capture_delta=provider_capture_delta,
                    detector_call_delta=detector_call_delta,
                    detector_metadata=detector_metadata,
                    detector_registered_as_trainable_child=bool(
                        metadata["detector_registered_as_trainable_child"]
                    ),
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
                    "frame_budgets": algorithm_records[0]["total_budgets"],
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
            "algorithm": "dytok",
            "variant": "dytok.video",
            "backbone": "llava_onevision",
            "binding": "dytok.video.llava_onevision",
            "execution_phase": "generate",
            "canonical_config": CANONICAL_DYTOK_CONFIG,
            "quality_claim": "functional_migration_only",
        },
        "provenance": {
            "code_revision": args.code_revision,
            "reference_revision": VIDEO_REFERENCE_REVISION,
            "command": shlex.join(sys.argv),
            "checkpoint": checkpoint_info,
            "detector_checkpoint": detector_info,
            "detector_siglip": siglip_info,
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
            "detector_registered_in_training_checkpoint": False,
        },
    }
    if tuple(int(item) for item in cases) != FIXED_NEXTQA_DOC_IDS:
        raise RuntimeError("The completed sample set differs from the fixed manifest")
    return report, sample_rows


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    add_checkpoint_arguments(parser, estimated_peak_mib=32768)
    parser.add_argument("--detector_checkpoint", type=Path, required=True)
    parser.add_argument("--detector_siglip", type=Path, required=True)
    parser.add_argument("--max_new_tokens", type=int, default=4)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.max_new_tokens < 2:
        raise RuntimeError("max_new_tokens must be at least 2")
    return inference_main(args, _run)


if __name__ == "__main__":
    raise SystemExit(main())
