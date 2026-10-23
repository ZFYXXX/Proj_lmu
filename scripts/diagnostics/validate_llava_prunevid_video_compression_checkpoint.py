"""Validate canonical PruneVid video inference on LLaVA-OneVision.

The diagnostic is offline, batch-one, and single-GPU. It reuses the same four
fixed ``nextqa_mc_test`` documents and native baseline policy as the existing
LLaVA slices, while requiring both canonical stages: deterministic projector
clustering and the additional layer-10 VTP/cache compaction boundary.
"""

from __future__ import annotations

import argparse
import shlex
import sys
import time
from typing import Any, Mapping, Sequence

from mllm_comp.backbones.base import BackboneRuntime
from mllm_comp.backbones.llava_onevision import (
    LlavaOneVisionSample,
    create_llava_onevision_backbone,
)
from mllm_comp.compressors.video.config import VIDEO_REFERENCE_REVISION
from mllm_comp.pipeline.stage import CompressionStageConfig
from scripts.diagnostics.video_method_checks import CANONICAL_PRUNEVID_CONFIG, prunevid_cache_layers_exact as _cache_layers_exact
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
def _stage() -> CompressionStageConfig:
    return CompressionStageConfig.from_dict(
        {
            "name": "prunevid-video-llava-onevision-real-checkpoint",
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
            "binding": "prunevid.video.llava_onevision",
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
    report = _llava_provider_video_checkpoint_common.generation_report(
        facade=facade,
        runtime=runtime,
        state=state,
        batch=batch,
        audit=first_layer_audit,
        torch=torch,
        max_new_tokens=max_new_tokens,
    )
    report["decoder_after_vtp_boundary"] = _common._decoder_summary(
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
    latest_vtp: Mapping[str, Any],
    labels_absent: bool,
    placeholder_count: int,
    frame_count: int,
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
    projector = projector_records[0]
    algorithm = algorithm_records[0]
    vtp = vtp_records[0]
    cache = cache_records[0]
    sequence_delta = vtp["sequence_layout_delta"]

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
    structural_source = tuple(latest_splice["structural_source_positions"])
    structural_result = tuple(latest_splice["structural_result_positions"])
    source_visual_span = tuple(
        int(item) for item in latest_splice["source_visual_span"]
    )
    result_visual_span = tuple(
        int(item) for item in latest_splice["result_visual_span"]
    )
    kept_visual = tuple(int(item) for item in vtp["kept_visual_indices"])
    dropped_visual = tuple(int(item) for item in vtp["dropped_visual_indices"])
    expected_source = frame_count * 196

    checks = {
        "compression_applied": bool(latest_splice["compression_applied"]),
        "projector_non_identity_token_reduction": source_count > projector_count,
        "vtp_non_identity_token_reduction": projector_count > final_count > 0,
        "native_32_by_14_by_14_raster_exact": (
            frame_count == 32
            and source_count == expected_source
            and int(algorithm["frame_count"]) == frame_count
            and int(algorithm["grid_height"]) == 14
            and int(algorithm["grid_width"]) == 14
            and int(algorithm["tokens_per_frame"]) == 196
        ),
        "projector_layout_and_splice_counts_agree": (
            int(latest_splice["source_patch_count"]) == source_count
            and int(latest_splice["result_patch_count"]) == projector_count
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
            baseline_after - compressed_after == source_count - final_count
        ),
        "native_structural_tokens_preserved": (
            len(structural_source) == len(structural_result)
            and source_visual_span[1] - source_visual_span[0]
            == source_count + len(structural_source)
            and result_visual_span[1] - result_visual_span[0]
            == projector_count + len(structural_result)
        ),
        "one_video_placeholder": placeholder_count == 1,
        "inference_labels_absent": labels_absent,
        "canonical_config_exact": dict(algorithm["canonical_config"])
        == CANONICAL_PRUNEVID_CONFIG,
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
            and int(sequence_delta["source_count"])
            - int(sequence_delta["result_count"])
            == projector_count - final_count
        ),
        "layer10_cache_compaction_exact": (
            dict(vtp["cache"]) == dict(cache)
            and _cache_layers_exact(
                cache,
                source_length=sequence_source,
                result_length=sequence_result,
            )
        ),
        "bridge_vtp_layout_exact": (
            int(latest_vtp["selected_layer"])
            == CANONICAL_PRUNEVID_CONFIG["vtp_selected_layer"]
            and int(latest_vtp["source_visual_count"]) == projector_count
            and int(latest_vtp["result_visual_count"]) == final_count
            and int(latest_vtp["source_sequence_length"]) == sequence_source
            and int(latest_vtp["result_sequence_length"]) == sequence_result
            and bool(latest_vtp["structural_tokens_preserved"])
            and latest_vtp["position_policy"] == "dense_one_dimensional_rebuild"
        ),
        "baseline_logits_finite": bool(baseline["logits_finite"]),
        "compressed_logits_finite": bool(compressed["logits_finite"]),
        "bounded_generation_exact": (
            int(baseline["generated_token_count"]) == 4
            and int(compressed["generated_token_count"]) == 4
            and tuple(baseline["decoder"]["decode_query_lengths"]) == (1, 1, 1)
            and tuple(compressed["decoder"]["decode_query_lengths"]) == (1, 1, 1)
            and tuple(baseline["decoder_after_vtp_boundary"]["decode_query_lengths"])
            == (1, 1, 1)
            and tuple(compressed["decoder_after_vtp_boundary"]["decode_query_lengths"])
            == (1, 1, 1)
        ),
        "baseline_decoder_alignment": all(baseline["decoder"]["invariants"].values())
        and all(baseline["decoder_after_vtp_boundary"]["invariants"].values()),
        "compressed_decoder_alignment": all(
            compressed["decoder"]["invariants"].values()
        )
        and all(compressed["decoder_after_vtp_boundary"]["invariants"].values()),
    }
    if not all(checks.values()):
        raise RuntimeError(f"PruneVid LLaVA sample evidence failed: {checks!r}")
    return checks


def _run(args: argparse.Namespace) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    visible = prepare_checkpoint_environment(args.physical_gpu)

    checkpoint = args.checkpoint.resolve()
    parquet_path = args.nextqa_parquet.resolve()
    video_root = args.video_root.resolve()

    checkpoint_info = _common._checkpoint_record(checkpoint)
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
    post_vtp_layer = handles.get(
        f"llm.layer[{CANONICAL_PRUNEVID_CONFIG['vtp_selected_layer'] + 1}]"
    )
    if first_layer is None or post_vtp_layer is None:
        raise RuntimeError(
            "LLaVA structural handles do not cover the PruneVid VTP boundary"
        )
    first_layer_audit = _common._DecoderInputAudit(first_layer)
    post_vtp_layer_audit = _common._DecoderInputAudit(post_vtp_layer)
    activation = generation_activation(
        facade=facade,
        runtime=runtime,
        binding_name="prunevid.video.llava_onevision",
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
                latest_vtp = bridge_metadata["latest_vtp"]
                if latest_splice is None or latest_vtp is None:
                    raise RuntimeError(
                        "LLaVA bridge published incomplete PruneVid splice/VTP evidence"
                    )
                checks = _validate_prunevid_sample(
                    baseline=baseline,
                    compressed=compressed,
                    projector_records=projector_records,
                    algorithm_records=algorithm_records,
                    vtp_records=vtp_records,
                    cache_records=cache_records,
                    latest_splice=latest_splice,
                    latest_vtp=latest_vtp,
                    labels_absent=labels_absent,
                    placeholder_count=placeholder_count,
                    frame_count=frame_count,
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
                "bridge_latest_vtp": latest_vtp,
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
        first_layer_audit.close()
        post_vtp_layer_audit.close()

    torch.cuda.synchronize()
    gpu_after = _common._gpu_snapshot(args.physical_gpu)
    report = {
        "status": "passed",
        "scope": {
            "algorithm": "prunevid",
            "variant": "prunevid.video",
            "backbone": "llava_onevision",
            "binding": "prunevid.video.llava_onevision",
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
    add_checkpoint_arguments(parser, estimated_peak_mib=49152)
    parser.add_argument("--max_new_tokens", type=int, default=4)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.max_new_tokens != 4:
        raise RuntimeError("PruneVid fixed evidence requires max_new_tokens=4")
    return inference_main(args, _run)


if __name__ == "__main__":
    raise SystemExit(main())
