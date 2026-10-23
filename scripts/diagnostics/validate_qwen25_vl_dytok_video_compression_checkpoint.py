"""Validate canonical DyToK video inference on Qwen2.5-VL.

The diagnostic is offline, batch-one, and single-GPU.  It uses the same four
fixed ``nextqa_mc_test`` documents as the other exact inference cells.  A
separate frozen local LLaVA-OneVision detector consumes frames reconstructed
from Qwen2.5-VL's native patchified video tensor, then canonical dynamic
VisionZip reduction is legalized through the source ``video_grid_thw`` and
ordered sparse native M-RoPE anchors.
"""

from __future__ import annotations

import argparse
import math
import shlex
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

from mllm_comp.backbones.qwen25_vl import Qwen25VLSample, create_qwen25_vl_backbone
from mllm_comp.compressors.video.config import VIDEO_REFERENCE_REVISION
from mllm_comp.pipeline.stage import CompressionStageConfig
from scripts.diagnostics import video_checkpoint_common as _video_checkpoint_common
from scripts.diagnostics.qwen_video_checkpoint_common import qwen25_video_input_record
from scripts.diagnostics.video_checkpoint_common import (
    add_checkpoint_arguments,
    admit_generation_device,
    compare_generation,
    generation_activation,
    generation_case,
    generation_documents,
    checkpoint_gpu_report,
    generation_sample,
    inference_main,
    load_generation_runtime,
    prepare_checkpoint_environment,
    prepare_generation_state,
)

if __package__:
    from scripts.diagnostics import qwen_video_checkpoint_common as _qwen_common
else:
    import qwen_video_checkpoint_common as _qwen_common


FIXED_NEXTQA_DOC_IDS = _video_checkpoint_common.FIXED_NEXTQA_DOC_IDS
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
QWEN25_DYTOK_VIDEO_FPS = 1.0
QWEN25_DYTOK_VIDEO_MIN_PIXELS = 160 * 28 * 28
QWEN25_DYTOK_VIDEO_MAX_PIXELS = 128 * 32 * 32


def _conversation(
    *,
    prompt: str,
    video: Path,
    fps: float = QWEN25_DYTOK_VIDEO_FPS,
    min_pixels: int = QWEN25_DYTOK_VIDEO_MIN_PIXELS,
    max_pixels: int = QWEN25_DYTOK_VIDEO_MAX_PIXELS,
) -> list[dict[str, Any]]:
    if min_pixels <= 0:
        raise ValueError("Qwen2.5-VL video min_pixels must be positive")
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


def _stage() -> CompressionStageConfig:
    return CompressionStageConfig.from_dict(
        {
            "name": "dytok-video-qwen25-vl-real-checkpoint",
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
            "binding": "dytok.video.qwen25_vl",
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
    latest_generation: Mapping[str, Any],
    source_grid_thw: Sequence[int],
    source_grid_token_count: int,
    spatial_merge_size: int,
    second_per_grid_seconds: float,
    labels_absent: bool,
    video_span_count: int,
    provider_capture_delta: int,
    detector_call_delta: int,
    detector_metadata: Mapping[str, Any],
    detector_registered_as_trainable_child: bool,
) -> dict[str, bool]:
    if len(activation_records) != 1 or len(algorithm_records) != 1:
        raise RuntimeError(
            "DyToK must publish one layout and one algorithm record per sample"
        )
    if len(source_grid_thw) != 3:
        raise RuntimeError(f"Invalid Qwen2.5-VL source grid: {source_grid_thw!r}")

    layout = activation_records[0]
    algorithm = algorithm_records[0]
    splice_layout = latest_splice["layout_delta"]
    source_count = int(layout["source_count"])
    result_count = int(layout["result_count"])
    temporal, height, width = (int(item) for item in source_grid_thw)
    per_slice_source = height * width // (spatial_merge_size**2)
    expected_result = min(temporal * 147, source_count)
    baseline_length = int(baseline["decoder"]["prefill"]["query_length"])
    compressed_length = int(compressed["decoder"]["prefill"]["query_length"])
    representatives = tuple(
        int(item) for item in latest_splice["representative_indices"]
    )
    source_mrope = tuple(
        tuple(int(value) for value in row)
        for row in latest_splice["source_visual_mrope"]
    )
    result_mrope = tuple(
        tuple(int(value) for value in row)
        for row in latest_splice["result_visual_mrope"]
    )
    expected_result_mrope = tuple(source_mrope[index] for index in representatives)
    frame_summaries = tuple(algorithm["frame_summaries"])
    importance = tuple(float(value) for value in algorithm["frame_importance"])
    provider_importance = tuple(
        float(value) for value in algorithm["provider_frame_importance"]
    )
    budgets = tuple(int(value) for value in algorithm["total_budgets"])
    decode_record = latest_generation.get("last_decode_record")

    checks = {
        "compression_applied": bool(latest_splice["compression_applied"]),
        "non_identity_token_reduction": source_count > result_count,
        "one_contiguous_video_span": video_span_count == 1,
        "grid_and_source_counts_agree": (
            source_grid_token_count == source_count
            and int(latest_splice["source_placeholder_count"]) == source_count
        ),
        "source_capacity_supports_fixed_total": (
            per_slice_source > 147 and source_count >= temporal * 147
        ),
        "canonical_fixed_total_exact": (
            result_count == expected_result
            and int(algorithm["budget_total"]) == expected_result
            and sum(budgets) == expected_result
        ),
        "result_placeholder_count_agrees": (
            int(latest_splice["result_placeholder_count"]) == result_count
        ),
        "layout_records_agree": (
            int(splice_layout["source_count"]) == source_count
            and int(splice_layout["result_count"]) == result_count
            and int(algorithm["source_count"]) == source_count
            and int(algorithm["result_count"]) == result_count
        ),
        "decoder_length_reduction_exact": (
            baseline_length - compressed_length == source_count - result_count
        ),
        "source_video_grid_preserved": (
            bool(latest_splice["source_video_grid_preserved"])
            and tuple(int(item) for item in latest_splice["source_video_grid_thw"])
            == (temporal, height, width)
        ),
        "source_video_timing_preserved": (
            math.isfinite(second_per_grid_seconds)
            and second_per_grid_seconds > 0
            and math.isclose(
                float(latest_splice["second_per_grid_seconds"]),
                second_per_grid_seconds,
                rel_tol=0.0,
                abs_tol=1e-9,
            )
        ),
        "sparse_grid_policy": (
            latest_splice["result_grid_policy"] == "irregular_sparse_source_anchors"
        ),
        "representatives_source_ordered": (
            len(representatives) == result_count
            and representatives == tuple(sorted(representatives))
            and len(set(representatives)) == result_count
        ),
        "mrope_source_count_aligned": len(source_mrope) == source_count,
        "mrope_result_count_aligned": len(result_mrope) == result_count,
        "mrope_sparse_anchor_gather_exact": result_mrope == expected_result_mrope,
        "generation_prefill_complete": bool(latest_generation["prefill_complete"]),
        "generation_source_length_aligned": (
            int(latest_generation["source_sequence_length"]) == baseline_length
        ),
        "generation_result_length_aligned": (
            int(latest_generation["result_sequence_length"]) == compressed_length
        ),
        "generation_removed_count_aligned": (
            int(latest_generation["removed_count"]) == source_count - result_count
        ),
        "decode_repair_exercised": (
            int(latest_generation["decode_repairs"]) >= 1
            and isinstance(decode_record, Mapping)
        ),
        "inference_labels_absent": labels_absent,
        "provider_capture_once": provider_capture_delta == 1,
        "detector_called_once": detector_call_delta == 1,
        "canonical_config_exact": (
            dict(algorithm["canonical_config"]) == CANONICAL_DYTOK_CONFIG
        ),
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
            int(algorithm["temporal_slice_count"]) == temporal
            and len(frame_summaries) == temporal
            and len(importance) == temporal
            and len(provider_importance) == temporal
            and len(budgets) == temporal
        ),
        "provider_importance_preserved": (
            abs(float(algorithm["provider_frame_importance_sum"]) - 1.0) <= 1e-3
            and float(algorithm["provider_kernel_importance_max_abs_delta"]) <= 1e-3
        ),
        "detector_frame_source_exact": algorithm["detection_frame_source"]
        == "qwen_patchified_video_inverse_siglip_384",
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
        raise RuntimeError(f"DyToK Qwen2.5-VL sample evidence failed: {checks!r}")
    return checks


def _run(args: argparse.Namespace) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    visible = prepare_checkpoint_environment(args.physical_gpu)

    checkpoint = args.checkpoint.resolve()
    detector_checkpoint = args.detector_checkpoint.resolve()
    detector_siglip = args.detector_siglip.resolve()
    parquet_path = args.nextqa_parquet.resolve()
    video_root = args.video_root.resolve()

    checkpoint_info = _video_checkpoint_common._checkpoint_record(checkpoint)
    detector_info = _video_checkpoint_common._checkpoint_record(detector_checkpoint)
    siglip_info = _video_checkpoint_common._checkpoint_record(detector_siglip)
    documents, videos = generation_documents(parquet_path, video_root)

    torch, transformers, gpu_before, admission = admit_generation_device(
        args,
        bridge="Qwen2.5-VL",
    )

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
            "dytok_detection_model_path": str(detector_checkpoint),
            "dytok_detection_siglip_path": str(detector_siglip),
            "dytok_detection_device_map": "cuda:0",
            "dytok_detection_dtype": torch.float16,
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
        binding_name="dytok.video.qwen25_vl",
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
            source_grid_thw = native_video["video_grid_thw"]
            merge_size = native_video["spatial_merge_size"]
            per_slice_source_count = (
                source_grid_thw[1] * source_grid_thw[2] // (merge_size**2)
            )
            if per_slice_source_count <= 147:
                raise RuntimeError(
                    "Frozen Qwen2.5-VL DyToK preprocessing did not keep the "
                    f"source slice above 147 tokens: grid={source_grid_thw!r}"
                )

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
                latest_generation = bridge_metadata["latest_generation"]
                if latest_splice is None or latest_generation is None:
                    raise RuntimeError(
                        "Qwen2.5-VL bridge published incomplete splice/generation "
                        "evidence"
                    )
                checks = _validate_dytok_sample(
                    baseline=baseline,
                    compressed=compressed,
                    activation_records=layout_records,
                    algorithm_records=algorithm_records,
                    latest_splice=latest_splice,
                    latest_generation=latest_generation,
                    source_grid_thw=native_video["video_grid_thw"],
                    source_grid_token_count=native_video["source_grid_token_count"],
                    spatial_merge_size=native_video["spatial_merge_size"],
                    second_per_grid_seconds=native_video["second_per_grid_ts"][0],
                    labels_absent=native_video["labels_absent_for_inference"],
                    video_span_count=1,
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
                    **native_video,
                    "conversation": conversation,
                    "requested_fps": float(args.video_fps),
                    "requested_min_pixels": int(args.video_min_pixels),
                    "requested_max_pixels": int(args.video_max_pixels),
                    "per_temporal_slice_source_count": per_slice_source_count,
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
                    "frame_budgets": algorithm_records[0]["total_budgets"],
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
            "algorithm": "dytok",
            "variant": "dytok.video",
            "backbone": "qwen25_vl",
            "binding": "dytok.video.qwen25_vl",
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
            "video_preprocessing": {
                "fps": float(args.video_fps),
                "min_pixels": int(args.video_min_pixels),
                "max_pixels": int(args.video_max_pixels),
                "provider": "qwen_vl_utils.process_vision_info",
                "detector_frame_transform": (
                    "inverse_qwen25_patch_permutation_then_tubelet_mean_"
                    "resize_384_siglip_normalization"
                ),
            },
        },
        "environment": {
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "cuda_visible_devices": visible,
            "logical_device": "cuda:0",
            "logical_device_count": torch.cuda.device_count(),
            "gpu_name": torch.cuda.get_device_name(0),
            "backbone_dtype": "bfloat16",
            "detector_dtype": "float16",
            "attention_implementation": "sdpa",
            "offline": True,
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
    add_checkpoint_arguments(parser, estimated_peak_mib=57344)
    parser.add_argument("--detector_checkpoint", type=Path, required=True)
    parser.add_argument("--detector_siglip", type=Path, required=True)
    parser.add_argument("--max_new_tokens", type=int, default=4)
    parser.add_argument("--video_fps", type=float, default=QWEN25_DYTOK_VIDEO_FPS)
    parser.add_argument(
        "--video_min_pixels",
        type=int,
        default=QWEN25_DYTOK_VIDEO_MIN_PIXELS,
    )
    parser.add_argument(
        "--video_max_pixels",
        type=int,
        default=QWEN25_DYTOK_VIDEO_MAX_PIXELS,
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.max_new_tokens < 2:
        raise RuntimeError("max_new_tokens must be at least 2")
    if args.video_min_pixels > args.video_max_pixels:
        raise RuntimeError("video_min_pixels cannot exceed video_max_pixels")
    return inference_main(args, _run)


if __name__ == "__main__":
    raise SystemExit(main())
