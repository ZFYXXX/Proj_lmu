"""Validate canonical PruneVid video inference on Qwen2.5-VL.

The diagnostic is offline, batch-one, and single-GPU. It reuses the same four
fixed ``nextqa_mc_test`` documents and native baseline policy as the existing
Qwen2.5-VL slices, while requiring both canonical PruneVid stages: deterministic
temporal/static-dynamic projector clustering and the additional layer-10
VTP/cache-compaction boundary.

The source ``video_grid_thw`` and timing remain authoritative. Projector and
VTP reductions retain ordered sparse source anchors, gathered native M-RoPE
positions, exact rope deltas, and physically compact DynamicCache prefixes.
"""

from __future__ import annotations

import argparse
import math
import shlex
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

from mllm_comp.backbones.base import BackboneRuntime
from mllm_comp.backbones.qwen25_vl import Qwen25VLSample, create_qwen25_vl_backbone
from mllm_comp.compressors.video.config import VIDEO_REFERENCE_REVISION
from mllm_comp.pipeline.stage import CompressionStageConfig
from scripts.diagnostics.video_method_checks import CANONICAL_PRUNEVID_CONFIG, prunevid_cache_layers_exact as _cache_layers_exact
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
QWEN25_PRUNEVID_VIDEO_FPS = _qwen_common.QWEN25_VIDEO_FPS
QWEN25_PRUNEVID_VIDEO_MAX_PIXELS = _qwen_common.QWEN25_VIDEO_MAX_PIXELS


def _conversation(
    *,
    prompt: str,
    video: Path,
    fps: float = QWEN25_PRUNEVID_VIDEO_FPS,
    max_pixels: int = QWEN25_PRUNEVID_VIDEO_MAX_PIXELS,
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
            "name": "prunevid-video-qwen25-vl-real-checkpoint",
            "placement": "projector_level",
            "execution_phase": "generate",
            "plugin": "prunevid",
            "algorithm": "prunevid",
            "variant": "prunevid.video",
            "target_modalities": ["video"],
            "conditioning_modalities": ["text"],
            "preserved_modalities": ["text", "image", "audio"],
            "coupling_policy": "independent",
            "media_alignment": "spatiotemporal",
            "binding": "prunevid.video.qwen25_vl",
            "config": {},
        }
    )


def _generation_report(
    *,
    facade: Any,
    runtime: BackboneRuntime,
    state: Any,
    batch: Any,
    first_layer_audit: Any,
    post_vtp_layer_audit: Any,
    torch: Any,
    max_new_tokens: int,
) -> dict[str, Any]:
    post_vtp_mark = post_vtp_layer_audit.mark()
    report = _qwen_common.generation_report(
        facade=facade,
        runtime=runtime,
        state=state,
        batch=batch,
        audit=first_layer_audit,
        torch=torch,
        max_new_tokens=max_new_tokens,
    )
    report["decoder_after_vtp_boundary"] = _video_checkpoint_common._decoder_summary(
        post_vtp_layer_audit.since(post_vtp_mark)
    )
    return report


def _validate_prunevid_sample(
    *,
    baseline: Mapping[str, Any],
    compressed: Mapping[str, Any],
    projector_records: Sequence[Mapping[str, Any]],
    algorithm_records: Sequence[Mapping[str, Any]],
    vtp_records: Sequence[Mapping[str, Any]],
    cache_records: Sequence[Mapping[str, Any]],
    latest_splice: Mapping[str, Any],
    latest_generation: Mapping[str, Any],
    latest_vtp: Mapping[str, Any],
    source_grid_thw: Sequence[int],
    source_grid_token_count: int,
    spatial_merge_size: int,
    second_per_grid_seconds: float,
    labels_absent: bool,
    video_span_count: int,
) -> dict[str, bool]:
    if not (
        len(projector_records)
        == len(algorithm_records)
        == len(vtp_records)
        == len(cache_records)
        == 1
    ):
        raise RuntimeError(
            "PruneVid must publish one projector, algorithm, VTP, and cache "
            "record per sample"
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
    grid_height = height // spatial_merge_size
    grid_width = width // spatial_merge_size
    tokens_per_frame = grid_height * grid_width
    expected_source_count = temporal * tokens_per_frame

    projector = projector_records[0]
    algorithm = algorithm_records[0]
    vtp = vtp_records[0]
    cache = cache_records[0]
    sequence_delta = vtp["sequence_layout_delta"]
    splice_layout = latest_splice["layout_delta"]
    source_count = int(projector["source_count"])
    projector_count = int(projector["result_count"])
    final_count = int(vtp["result_count"])
    baseline_first = int(baseline["decoder"]["prefill"]["query_length"])
    compressed_first = int(compressed["decoder"]["prefill"]["query_length"])
    baseline_after = int(
        baseline["decoder_after_vtp_boundary"]["prefill"]["query_length"]
    )
    compressed_after = int(
        compressed["decoder_after_vtp_boundary"]["prefill"]["query_length"]
    )
    sequence_source = int(sequence_delta["source_count"])
    sequence_result = int(sequence_delta["result_count"])
    representatives = tuple(int(item) for item in algorithm["representative_indices"])
    kept_visual = tuple(int(item) for item in vtp["kept_visual_indices"])
    dropped_visual = tuple(int(item) for item in vtp["dropped_visual_indices"])
    source_mrope = tuple(
        tuple(int(value) for value in row)
        for row in latest_splice["source_visual_mrope"]
    )
    projector_mrope = tuple(
        tuple(int(value) for value in row)
        for row in latest_splice["result_visual_mrope"]
    )
    vtp_source_mrope = tuple(
        tuple(int(value) for value in row) for row in latest_vtp["source_visual_mrope"]
    )
    final_mrope = tuple(
        tuple(int(value) for value in row) for row in latest_vtp["result_visual_mrope"]
    )
    expected_projector_mrope = tuple(source_mrope[index] for index in representatives)
    expected_final_mrope = tuple(projector_mrope[index] for index in kept_visual)
    retained_positions = tuple(
        int(item) for item in latest_splice["retained_placeholder_positions"]
    )
    expected_vtp_removed = tuple(retained_positions[index] for index in dropped_visual)
    vtp_removed = tuple(
        int(item) for item in latest_vtp["dropped_source_placeholder_positions"]
    )
    projector_removed = tuple(
        int(item) for item in latest_splice["removed_placeholder_positions"]
    )
    combined_removed = tuple(sorted((*projector_removed, *vtp_removed)))
    source_visual_sequence_indices = tuple(
        int(item) for item in latest_vtp["source_visual_sequence_indices"]
    )
    result_visual_sequence_indices = tuple(
        int(item) for item in latest_vtp["result_visual_sequence_indices"]
    )
    rope_values = tuple(int(item) for item in latest_vtp["rope_deltas"])
    expected_rope_delta = (
        int(latest_vtp["max_active_rotary_position"])
        + 1
        - int(latest_vtp["active_position_count"])
    )
    combined_removed_count = source_count - final_count
    generated_token_count = int(compressed["generated_token_count"])
    decode_query_lengths = tuple(
        int(item) for item in compressed["decoder"]["decode_query_lengths"]
    )
    post_vtp_decode_query_lengths = tuple(
        int(item)
        for item in compressed["decoder_after_vtp_boundary"]["decode_query_lengths"]
    )

    checks = {
        "compression_applied": bool(latest_splice["compression_applied"]),
        "projector_non_identity_token_reduction": source_count > projector_count,
        "vtp_non_identity_token_reduction": projector_count > final_count > 0,
        "one_contiguous_video_span": video_span_count == 1,
        "grid_and_source_counts_agree": (
            source_count == expected_source_count
            and source_grid_token_count == source_count
            and int(latest_splice["source_placeholder_count"]) == source_count
            and int(algorithm["frame_count"]) == temporal
            and int(algorithm["grid_height"]) == grid_height
            and int(algorithm["grid_width"]) == grid_width
            and int(algorithm["tokens_per_frame"]) == tokens_per_frame
        ),
        "projector_layout_records_agree": (
            int(splice_layout["source_count"]) == source_count
            and int(splice_layout["result_count"]) == projector_count
            and int(latest_splice["result_placeholder_count"]) == projector_count
            and int(algorithm["source_count"]) == source_count
            and int(algorithm["result_count"]) == projector_count
        ),
        "projector_decoder_length_reduction_exact": (
            baseline_first - compressed_first == source_count - projector_count
        ),
        "baseline_layer_lengths_stable": baseline_first == baseline_after,
        "vtp_boundary_lengths_exact": (
            compressed_first == sequence_source
            and compressed_after == sequence_result
            and sequence_source - sequence_result == projector_count - final_count
        ),
        "combined_decoder_length_reduction_exact": (
            baseline_after - compressed_after == combined_removed_count
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
        "inference_labels_absent": labels_absent,
        "canonical_config_exact": (
            dict(algorithm["canonical_config"]) == CANONICAL_PRUNEVID_CONFIG
        ),
        "projected_signal_shapes_exact": (
            tuple(algorithm["projected_features_shape"])[0] == source_count
            and tuple(algorithm["result_features_shape"])[0] == projector_count
            and len(tuple(algorithm["projected_features_shape"])) == 2
            and len(tuple(algorithm["result_features_shape"])) == 2
        ),
        "projected_signals_finite": (
            bool(algorithm["projected_features_finite"])
            and bool(algorithm["result_features_finite"])
        ),
        "projector_mrope_sparse_anchor_gather_exact": (
            len(source_mrope) == source_count
            and len(projector_mrope) == projector_count
            and all(len(row) == 3 for row in source_mrope)
            and all(len(row) == 3 for row in projector_mrope)
            and projector_mrope == expected_projector_mrope
        ),
        "vtp_config_and_layout_exact": (
            int(vtp["selected_layer"])
            == CANONICAL_PRUNEVID_CONFIG["vtp_selected_layer"]
            and float(vtp["alpha"]) == CANONICAL_PRUNEVID_CONFIG["vtp_alpha"]
            and int(vtp["source_count"]) == projector_count
            and len(kept_visual) == final_count
            and len(dropped_visual) == projector_count - final_count
            and tuple(sorted((*kept_visual, *dropped_visual)))
            == tuple(range(projector_count))
        ),
        "vtp_sequence_delta_visual_only": (
            int(sequence_delta["dropped_count"]) == projector_count - final_count
            and sequence_source - sequence_result == projector_count - final_count
        ),
        "layer10_cache_compaction_exact": (
            dict(vtp["cache"]) == dict(cache)
            and _cache_layers_exact(
                cache,
                source_length=sequence_source,
                result_length=sequence_result,
            )
        ),
        "vtp_mrope_sparse_anchor_gather_exact": (
            vtp_source_mrope == projector_mrope
            and len(final_mrope) == final_count
            and all(len(row) == 3 for row in final_mrope)
            and final_mrope == expected_final_mrope
            and tuple(int(item) for item in latest_vtp["kept_visual_indices"])
            == kept_visual
            and len(source_visual_sequence_indices) == projector_count
            and len(result_visual_sequence_indices) == final_count
            and result_visual_sequence_indices
            == tuple(sorted(set(result_visual_sequence_indices)))
        ),
        "vtp_removed_placeholder_lineage_exact": (
            len(retained_positions) == projector_count
            and vtp_removed == expected_vtp_removed
            and len(projector_removed) == source_count - projector_count
            and len(combined_removed) == combined_removed_count
            and len(set(combined_removed)) == combined_removed_count
        ),
        "bridge_vtp_layout_exact": (
            int(latest_vtp["selected_layer"])
            == CANONICAL_PRUNEVID_CONFIG["vtp_selected_layer"]
            and int(latest_vtp["source_visual_count"]) == projector_count
            and int(latest_vtp["result_visual_count"]) == final_count
            and int(latest_vtp["source_sequence_length"]) == sequence_source
            and int(latest_vtp["result_sequence_length"]) == sequence_result
            and latest_vtp["position_policy"] == "gather_native_mrope"
        ),
        "post_vtp_rope_delta_exact": (
            bool(latest_vtp["rope_delta_exact"])
            and int(latest_vtp["active_position_count"]) == sequence_result
            and int(latest_vtp["expected_rope_delta"]) == expected_rope_delta
            and rope_values == (expected_rope_delta,)
            and tuple(int(item) for item in latest_splice["rope_deltas"]) == rope_values
        ),
        "generation_prefill_complete": bool(latest_generation["prefill_complete"]),
        "generation_lengths_and_removal_exact": (
            int(latest_generation["source_sequence_length"]) == baseline_first
            and int(latest_generation["result_sequence_length"]) == sequence_result
            and int(latest_generation["removed_count"]) == combined_removed_count
        ),
        "baseline_logits_finite": bool(baseline["logits_finite"]),
        "compressed_logits_finite": bool(compressed["logits_finite"]),
        "bounded_cached_generation_exact": (
            int(baseline["generated_token_count"]) == 4
            and generated_token_count == 4
            and tuple(baseline["decoder"]["decode_query_lengths"]) == (1, 1, 1)
            and tuple(baseline["decoder_after_vtp_boundary"]["decode_query_lengths"])
            == (1, 1, 1)
            and decode_query_lengths == (1, 1, 1)
            and post_vtp_decode_query_lengths == (1, 1, 1)
            and int(latest_generation["decode_repairs"]) == 3
            and _decode_record_coherent(
                latest_generation.get("last_decode_record"),
                removed_count=combined_removed_count,
                result_sequence_length=sequence_result,
            )
        ),
        "baseline_decoder_alignment": all(baseline["decoder"]["invariants"].values())
        and all(baseline["decoder_after_vtp_boundary"]["invariants"].values()),
        "compressed_decoder_alignment": all(
            compressed["decoder"]["invariants"].values()
        )
        and all(compressed["decoder_after_vtp_boundary"]["invariants"].values()),
    }
    if not all(checks.values()):
        raise RuntimeError(f"PruneVid Qwen2.5-VL sample evidence failed: {checks!r}")
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
    post_vtp_layer = handles.get(
        f"llm.layer[{CANONICAL_PRUNEVID_CONFIG['vtp_selected_layer'] + 1}]"
    )
    if first_layer is None or post_vtp_layer is None:
        raise RuntimeError(
            "Qwen2.5-VL structural handles do not cover the PruneVid VTP boundary"
        )
    first_layer_audit = _video_checkpoint_common._DecoderInputAudit(first_layer)
    post_vtp_layer_audit = _video_checkpoint_common._DecoderInputAudit(post_vtp_layer)
    activation = generation_activation(
        facade=facade,
        runtime=runtime,
        binding_name="prunevid.video.qwen25_vl",
        stage=_stage(),
    )
    if (
        dict(activation.runtime_metadata()["canonical_config"])
        != CANONICAL_PRUNEVID_CONFIG
    ):
        raise RuntimeError(
            "Resolved PruneVid configuration differs from the canonical profile"
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
            if (
                source_grid_thw[0] < 1
                or per_slice_source_count
                < CANONICAL_PRUNEVID_CONFIG["min_tokens_for_cluster"]
                or int(
                    per_slice_source_count * CANONICAL_PRUNEVID_CONFIG["cluster_ratio"]
                )
                < 1
            ):
                raise RuntimeError(
                    "Frozen Qwen2.5-VL preprocessing cannot exercise canonical "
                    f"PruneVid: grid={source_grid_thw!r}, "
                    f"per_slice={per_slice_source_count}"
                )

            with compare_generation(
                activation=activation,
                generate=_generation_report,
                facade=facade,
                runtime=runtime,
                state=state,
                batch=batch,
                first_layer_audit=first_layer_audit,
                post_vtp_layer_audit=post_vtp_layer_audit,
                torch=torch,
                max_new_tokens=args.max_new_tokens,
            ) as (baseline, compressed, before, metadata):
                before_projector = int(before["projector_invocations"])
                before_vtp = int(before["vtp_invocations"])
                if (
                    int(metadata["projector_invocations"]) != before_projector + 1
                    or int(metadata["vtp_invocations"]) != before_vtp + 1
                ):
                    raise RuntimeError(
                        "PruneVid projector/VTP invocation counts did not increment once"
                    )
                projector_records = metadata["projector_layout_deltas"]
                algorithm_records = metadata["algorithm_records"]
                vtp_records = metadata["vtp_layout_deltas"]
                cache_records = metadata["cache_repairs"]
                bridge_metadata = metadata["bridge"]
                latest_splice = bridge_metadata["latest_splice"]
                latest_generation = bridge_metadata["latest_generation"]
                latest_vtp = bridge_metadata["latest_vtp"]
                if (
                    latest_splice is None
                    or latest_generation is None
                    or latest_vtp is None
                ):
                    raise RuntimeError(
                        "Qwen2.5-VL bridge published incomplete splice/VTP/generation "
                        "evidence"
                    )
                checks = _validate_prunevid_sample(
                    baseline=baseline,
                    compressed=compressed,
                    projector_records=projector_records,
                    algorithm_records=algorithm_records,
                    vtp_records=vtp_records,
                    cache_records=cache_records,
                    latest_splice=latest_splice,
                    latest_generation=latest_generation,
                    latest_vtp=latest_vtp,
                    source_grid_thw=native_video["video_grid_thw"],
                    source_grid_token_count=native_video["source_grid_token_count"],
                    spatial_merge_size=native_video["spatial_merge_size"],
                    second_per_grid_seconds=native_video["second_per_grid_ts"][0],
                    labels_absent=native_video["labels_absent_for_inference"],
                    video_span_count=1,
                )
                case_activation = {
                    **metadata,
                    "projector_layout_deltas": projector_records,
                    "algorithm_records": algorithm_records,
                    "vtp_layout_deltas": vtp_records,
                    "cache_repairs": cache_records,
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
                "bridge_latest_vtp": latest_vtp,
                "bridge_latest_generation": latest_generation,
                "checks": checks,
            }
            cases[str(doc_id)] = case
            sample_rows.append(
                {
                    **generation_sample(document, baseline, compressed),
                    "compression_applied": checks["compression_applied"],
                    "source_visual_tokens": projector_records[0]["source_count"],
                    "projector_visual_tokens": projector_records[0]["result_count"],
                    "final_vtp_visual_tokens": vtp_records[0]["result_count"],
                }
            )
    finally:
        if activation.installed:
            activation.uninstall()
        post_vtp_layer_audit.close()
        first_layer_audit.close()

    torch.cuda.synchronize()
    gpu_after = _video_checkpoint_common._gpu_snapshot(args.physical_gpu)
    report = {
        "status": "passed",
        "scope": {
            "algorithm": "prunevid",
            "variant": "prunevid.video",
            "backbone": "qwen25_vl",
            "binding": "prunevid.video.qwen25_vl",
            "execution_phase": "generate",
            "canonical_config": CANONICAL_PRUNEVID_CONFIG,
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
                    "uniform dense post-merger temporal rasters satisfy canonical "
                    "PruneVid DPC-KNN and minimum clustering cardinality"
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
    parser.add_argument("--video_fps", type=float, default=QWEN25_PRUNEVID_VIDEO_FPS)
    parser.add_argument(
        "--video_max_pixels",
        type=int,
        default=QWEN25_PRUNEVID_VIDEO_MAX_PIXELS,
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.max_new_tokens != 4:
        raise RuntimeError("PruneVid fixed evidence requires max_new_tokens=4")
    return inference_main(args, _run)


if __name__ == "__main__":
    raise SystemExit(main())
