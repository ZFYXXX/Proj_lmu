"""Validate canonical FastVID video inference on Qwen2.5-VL.

The diagnostic is offline, batch-one, and single-GPU. It reuses the same four
fixed ``nextqa_mc_test`` documents and native baseline policy as the existing
Qwen2.5-VL slices, while freezing FastVID's projected, native final-block,
frame-global, and mean-received-saliency signals plus exact DySeg, STPrune,
and single-/multi-frame DTM decisions.

The source ``video_grid_thw`` and timing remain authoritative. Compressed
embeddings are legalized through ordered sparse source anchors and gathered
native M-RoPE positions rather than a fabricated dense result grid.
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
from scripts.diagnostics.qwen_video_checkpoint_common import (
    decode_record_coherent as _decode_record_coherent,
    qwen25_video_input_record,
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
    from scripts.diagnostics import qwen_video_checkpoint_common as _qwen_common
else:
    import qwen_video_checkpoint_common as _qwen_common


FIXED_NEXTQA_DOC_IDS = _video_checkpoint_common.FIXED_NEXTQA_DOC_IDS
CANONICAL_FASTVID_CONFIG = {
    "retention_ratio": 0.10,
    "dyseg_c": 8,
    "dyseg_tau": 0.9,
    "stprune_d": 0.4,
    "dtm_p": 4,
    "dtm_beta": 0.6,
}
QWEN25_FASTVID_VIDEO_FPS = _qwen_common.QWEN25_VIDEO_FPS
QWEN25_FASTVID_VIDEO_MAX_PIXELS = _qwen_common.QWEN25_VIDEO_MAX_PIXELS


def _conversation(
    *,
    prompt: str,
    video: Path,
    fps: float = QWEN25_FASTVID_VIDEO_FPS,
    max_pixels: int = QWEN25_FASTVID_VIDEO_MAX_PIXELS,
) -> list[dict[str, Any]]:
    return _qwen_common._conversation(
        prompt=prompt,
        video=video,
        fps=fps,
        max_pixels=max_pixels,
    )


def _stage() -> CompressionStageConfig:
    return CompressionStageConfig.from_dict(
        {
            "name": "fastvid-video-qwen25-vl-real-checkpoint",
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
            "binding": "fastvid.video.qwen25_vl",
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
    latest_generation: Mapping[str, Any],
    source_grid_thw: Sequence[int],
    source_grid_token_count: int,
    spatial_merge_size: int,
    second_per_grid_seconds: float,
    labels_absent: bool,
    video_span_count: int,
    provider_capture_delta: int,
    provider_metadata: Mapping[str, Any],
) -> dict[str, bool]:
    if len(activation_records) != 1 or len(algorithm_records) != 1:
        raise RuntimeError(
            "FastVID must publish one layout and one algorithm record per sample"
        )
    if len(source_grid_thw) != 3:
        raise RuntimeError(f"Invalid Qwen2.5-VL source grid: {source_grid_thw!r}")

    temporal, height, width = (int(item) for item in source_grid_thw)
    if (
        spatial_merge_size < 1
        or height % spatial_merge_size
        or width % spatial_merge_size
    ):
        raise RuntimeError(
            "Qwen2.5-VL source grid is incompatible with its spatial merger"
        )
    projected_tokens_per_frame = height * width // (int(spatial_merge_size) ** 2)
    raw_tokens_per_frame = height * width
    expected_source_count = temporal * projected_tokens_per_frame

    layout = activation_records[0]
    algorithm = algorithm_records[0]
    splice_layout = latest_splice["layout_delta"]
    source_count = int(layout["source_count"])
    result_count = int(layout["result_count"])
    baseline_length = int(baseline["decoder"]["prefill"]["query_length"])
    compressed_length = int(compressed["decoder"]["prefill"]["query_length"])
    algorithm_representatives = tuple(
        int(item) for item in algorithm["representative_indices"]
    )
    splice_representatives = tuple(
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
    expected_result_mrope = tuple(
        source_mrope[index] for index in splice_representatives
    )
    decode_record = latest_generation.get("last_decode_record")
    generated_token_count = int(compressed["generated_token_count"])
    decode_query_lengths = tuple(
        int(item) for item in compressed["decoder"]["decode_query_lengths"]
    )
    removed_count = source_count - result_count

    checks = {
        "compression_applied": bool(latest_splice["compression_applied"]),
        "non_identity_token_reduction": source_count > result_count,
        "one_contiguous_video_span": video_span_count == 1,
        "grid_and_source_counts_agree": (
            source_count == expected_source_count
            and source_grid_token_count == source_count
            and int(latest_splice["source_placeholder_count"]) == source_count
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
            baseline_length - compressed_length == removed_count
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
            len(splice_representatives) == result_count
            and splice_representatives == tuple(sorted(splice_representatives))
            and len(set(splice_representatives)) == result_count
        ),
        "algorithm_and_splice_representatives_agree": (
            algorithm_representatives == splice_representatives
        ),
        "mrope_source_count_aligned": (
            len(source_mrope) == source_count
            and all(len(row) == 3 for row in source_mrope)
        ),
        "mrope_result_count_aligned": (
            len(result_mrope) == result_count
            and all(len(row) == 3 for row in result_mrope)
        ),
        "mrope_sparse_anchor_gather_exact": result_mrope == expected_result_mrope,
        "generation_prefill_complete": bool(latest_generation["prefill_complete"]),
        "generation_source_length_aligned": (
            int(latest_generation["source_sequence_length"]) == baseline_length
        ),
        "generation_result_length_aligned": (
            int(latest_generation["result_sequence_length"]) == compressed_length
        ),
        "generation_removed_count_aligned": (
            int(latest_generation["removed_count"]) == removed_count
        ),
        "cached_decode_exact": (
            generated_token_count >= 2
            and int(latest_generation["decode_repairs"]) == generated_token_count - 1
            and decode_query_lengths == (1,) * (generated_token_count - 1)
            and _decode_record_coherent(
                decode_record,
                removed_count=removed_count,
                result_sequence_length=compressed_length,
            )
        ),
        "inference_labels_absent": labels_absent,
        "provider_capture_once": provider_capture_delta == 1,
        "provider_identity_exact": (
            provider_metadata["provider"] == "qwen25_vl_fastvid_video_signals"
        ),
        "provider_native_qwen_path": (
            bool(provider_metadata["video_only"])
            and bool(provider_metadata["raw_stream_attached"])
            and not bool(provider_metadata["siglip_pooling_head_loaded"])
            and provider_metadata["siglip_path"] is None
        ),
        "canonical_config_exact": (
            dict(algorithm["canonical_config"]) == CANONICAL_FASTVID_CONFIG
        ),
        "projected_signal_shape_exact": (
            tuple(algorithm["projected_features_shape"])[0] == source_count
            and len(tuple(algorithm["projected_features_shape"])) == 2
            and int(tuple(algorithm["projected_features_shape"])[1]) > 0
        ),
        "raw_signal_shape_exact": (
            tuple(algorithm["raw_features_shape"])[0] == temporal * raw_tokens_per_frame
            and int(algorithm["raw_tokens_per_frame"]) == raw_tokens_per_frame
            and len(tuple(algorithm["raw_features_shape"])) == 2
            and int(tuple(algorithm["raw_features_shape"])[1]) > 0
        ),
        "frame_global_signal_shape_exact": (
            tuple(algorithm["frame_global_features_shape"])[0] == temporal
            and len(tuple(algorithm["frame_global_features_shape"])) == 2
            and int(tuple(algorithm["frame_global_features_shape"])[1]) > 0
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
        "algorithm_representatives_source_ordered": bool(
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
        raise RuntimeError(f"FastVID Qwen2.5-VL sample evidence failed: {checks!r}")
    return checks


def _run(args: argparse.Namespace) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    visible = prepare_checkpoint_environment(args.physical_gpu)

    checkpoint = args.checkpoint.resolve()
    parquet_path = args.nextqa_parquet.resolve()
    video_root = args.video_root.resolve()

    checkpoint_info = _video_checkpoint_common._checkpoint_record(checkpoint)
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
        metadata={"enable_talker": False, "speech_sample_rate": None},
    )
    handles = facade.model_adapter.structural_handles(model)
    first_layer = handles.get("llm.layer[0]")
    if first_layer is None:
        raise RuntimeError("Qwen2.5-VL structural handle llm.layer[0] is missing")
    audit = _video_checkpoint_common._DecoderInputAudit(first_layer)
    activation = generation_activation(
        facade=facade,
        runtime=runtime,
        binding_name="fastvid.video.qwen25_vl",
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
            conversation = _conversation(
                prompt=document["prompt"],
                video=video,
                fps=args.video_fps,
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
            retain_num = int(
                per_slice_source_count * CANONICAL_FASTVID_CONFIG["retention_ratio"]
            )
            context_num = int(retain_num * CANONICAL_FASTVID_CONFIG["stprune_d"])
            salient_num = retain_num - context_num
            filtered_num = per_slice_source_count - salient_num
            if (
                retain_num < 2
                or salient_num < 1
                or context_num < 1
                or filtered_num < 4
                or retain_num >= per_slice_source_count
            ):
                raise RuntimeError(
                    "Frozen Qwen2.5-VL preprocessing cannot exercise canonical "
                    f"FastVID: grid={source_grid_thw!r}, "
                    f"per_slice={per_slice_source_count}"
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
                latest_generation = bridge_metadata["latest_generation"]
                if latest_splice is None or latest_generation is None:
                    raise RuntimeError(
                        "Qwen2.5-VL bridge published incomplete splice/generation "
                        "evidence"
                    )
                checks = _validate_fastvid_sample(
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
                    **native_video,
                    "conversation": conversation,
                    "requested_fps": float(args.video_fps),
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
            "algorithm": "fastvid",
            "variant": "fastvid.video",
            "backbone": "qwen25_vl",
            "binding": "fastvid.video.qwen25_vl",
            "execution_phase": "generate",
            "canonical_config": CANONICAL_FASTVID_CONFIG,
            "quality_claim": "functional_migration_only",
        },
        "provenance": {
            "code_revision": args.code_revision,
            "reference_revision": VIDEO_REFERENCE_REVISION,
            "command": shlex.join(sys.argv),
            "checkpoint": checkpoint_info,
            "nextqa_parquet": str(parquet_path),
            "nextqa_parquet_size_bytes": parquet_path.stat().st_size,
            "fixed_doc_ids": list(FIXED_NEXTQA_DOC_IDS),
            "video_preprocessing": {
                "fps": float(args.video_fps),
                "max_pixels": int(args.video_max_pixels),
                "provider": "qwen_vl_utils.process_vision_info",
                "constraint": (
                    "uniform post-merger temporal rasters satisfy canonical "
                    "FastVID retention and DTM"
                ),
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
    add_checkpoint_arguments(parser, estimated_peak_mib=49152)
    parser.add_argument("--max_new_tokens", type=int, default=4)
    parser.add_argument("--video_fps", type=float, default=QWEN25_FASTVID_VIDEO_FPS)
    parser.add_argument(
        "--video_max_pixels",
        type=int,
        default=QWEN25_FASTVID_VIDEO_MAX_PIXELS,
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.max_new_tokens < 2:
        raise RuntimeError("max_new_tokens must be at least 2")
    return inference_main(args, _run)


if __name__ == "__main__":
    raise SystemExit(main())
