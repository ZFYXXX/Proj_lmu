"""Validate canonical METok video inference on LLaVA-OneVision.

The diagnostic is offline, batch-one, and single-GPU.  It runs the same four
fixed ``nextqa_mc_test`` documents as every other video cell and requires the
complete METok path: local frozen SigLIP text conditioning, four-event
projector pooling, and decoder pruning before layers 3, 10, and 19.
"""

from __future__ import annotations

import argparse
import shlex
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

from mllm_comp.backbones.base import BackboneRuntime
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
METOK_BOUNDARY_LAYERS = (3, 10, 19)
CANONICAL_METOK_CONFIG = {
    "segmentation_method": "similarity",
    "num_segments": 4,
    "keyframe_method": "similarity",
    "keyframe_ratio": 0.4,
    "top_k_events": 3,
    "distinguish_important_events": True,
    "s1_important": 2,
    "s2_important": 4,
    "s1_normal": 4,
    "s2_normal": 8,
    "pool_mode": "bilinear",
    "non_keyframe_method": "adaptive_pool",
    "keyframe_pool_enable": False,
    "inside_compression": True,
    "pruning_layers": METOK_BOUNDARY_LAYERS,
    "ratio": 0.76,
}


def _stage() -> CompressionStageConfig:
    return CompressionStageConfig.from_dict(
        {
            "name": "metok-video-llava-onevision-real-checkpoint",
            "placement": "projector_level",
            "execution_phase": "generate",
            "plugin": "metok",
            "algorithm": "metok",
            "variant": "metok.video",
            "target_modalities": ["video"],
            "conditioning_modalities": ["text"],
            "preserved_modalities": ["text", "image", "audio"],
            "coupling_policy": "independent",
            "media_alignment": "spatiotemporal",
            "binding": "metok.video.llava_onevision",
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
    boundary_audits: Mapping[int, Any],
    torch: Any,
    max_new_tokens: int,
) -> dict[str, Any]:
    marks = {layer: audit.mark() for layer, audit in boundary_audits.items()}
    report = _llava_provider_video_checkpoint_common.generation_report(
        facade=facade,
        runtime=runtime,
        state=state,
        batch=batch,
        audit=first_layer_audit,
        torch=torch,
        max_new_tokens=max_new_tokens,
    )
    report["decoder_boundaries"] = {
        str(layer): _common._decoder_summary(audit.since(marks[layer]))
        for layer, audit in boundary_audits.items()
    }
    return report


def _cache_layers_exact(
    cache: Mapping[str, Any],
    *,
    boundary_layer: int,
    source_length: int,
    result_length: int,
) -> bool:
    expected_layers = tuple(range(boundary_layer))
    touched = tuple(int(item) for item in cache.get("touched_layers", ()))
    before = {int(item["layer_index"]): item for item in cache.get("layers_before", ())}
    after = {int(item["layer_index"]): item for item in cache.get("layers_after", ())}
    untouched = tuple(sorted(set(before).difference(expected_layers)))
    return (
        cache.get("policy") == "compact_prefix"
        and cache.get("boundary") == "before_layer"
        and int(cache.get("boundary_layer", -1)) == boundary_layer
        and int(cache.get("source_length", -1)) == source_length
        and int(cache.get("result_length", -1)) == result_length
        and cache.get("cache_type") == "DynamicCache"
        and touched == expected_layers
        and tuple(sorted(before)) == tuple(sorted(after))
        and all(index in before and index in after for index in expected_layers)
        and all(
            int(before[index]["logical_length"]) == source_length
            and int(before[index]["physical_length"]) == source_length
            and int(after[index]["logical_length"]) == result_length
            and int(after[index]["physical_length"]) == result_length
            for index in expected_layers
        )
        and all(before[index] == after[index] for index in untouched)
    )


def _canonical_config_exact(value: Mapping[str, Any]) -> bool:
    observed = dict(value)
    pruning_layers = observed.get("pruning_layers")
    if isinstance(pruning_layers, Sequence) and not isinstance(
        pruning_layers, (str, bytes)
    ):
        observed["pruning_layers"] = tuple(int(item) for item in pruning_layers)
    return observed == CANONICAL_METOK_CONFIG


def _pruning_chain_exact(
    pruning_records: Sequence[Mapping[str, Any]],
    cache_records: Sequence[Mapping[str, Any]],
    latest_metok: Sequence[Mapping[str, Any]],
    *,
    projector_count: int,
) -> bool:
    if not (
        len(pruning_records)
        == len(cache_records)
        == len(latest_metok)
        == len(METOK_BOUNDARY_LAYERS)
    ):
        return False
    prior_visual = projector_count

    prior_sequence_result: int | None = None
    for stage_index, boundary_layer in enumerate(METOK_BOUNDARY_LAYERS):
        record = pruning_records[stage_index]
        cache = cache_records[stage_index]
        bridge = latest_metok[stage_index]
        sequence = record.get("sequence_layout_delta", {})
        source_visual = int(record.get("source_count", -1))
        result_visual = int(record.get("result_count", -1))
        source_sequence = int(sequence.get("source_count", -1))
        result_sequence = int(sequence.get("result_count", -1))
        kept = tuple(int(item) for item in record.get("kept_visual_indices", ()))
        dropped = tuple(int(item) for item in record.get("dropped_visual_indices", ()))
        retained_projector = tuple(
            int(item)
            for item in record.get("cumulative_retained_projector_indices", ())
        )
        if not (
            int(record.get("boundary_layer", -1)) == boundary_layer
            and int(record.get("stage_index", -1)) == stage_index
            and source_visual == prior_visual
            and result_visual == len(kept)
            and tuple(sorted((*kept, *dropped))) == tuple(range(source_visual))
            and len(retained_projector) == result_visual
            and retained_projector == tuple(sorted(set(retained_projector)))
            and source_sequence - result_sequence == source_visual - result_visual
            and len(tuple(record.get("kept_sequence_indices", ()))) == result_sequence
            and len(tuple(record.get("cumulative_prompt_keep_indices", ())))
            == result_sequence
            and bool(record.get("final_visual_removal"))
            is (stage_index == len(METOK_BOUNDARY_LAYERS) - 1)
            and dict(record.get("cache", {})) == dict(cache)
            and _cache_layers_exact(
                cache,
                boundary_layer=boundary_layer,
                source_length=source_sequence,
                result_length=result_sequence,
            )
            and int(bridge.get("boundary_layer", -1)) == boundary_layer
            and int(bridge.get("stage_index", -1)) == stage_index
            and int(bridge.get("source_visual_count", -1)) == source_visual
            and int(bridge.get("result_visual_count", -1)) == result_visual
            and int(bridge.get("source_sequence_length", -1)) == source_sequence
            and int(bridge.get("result_sequence_length", -1)) == result_sequence
            and bool(bridge.get("structural_tokens_preserved"))

        ):
            return False
        if (
            prior_sequence_result is not None
            and source_sequence != prior_sequence_result
        ):
            return False
        prior_visual = result_visual

        prior_sequence_result = result_sequence
    return prior_visual == 0


def _all_decoder_summaries(report: Mapping[str, Any]) -> tuple[Mapping[str, Any], ...]:
    boundaries = report.get("decoder_boundaries", {})
    return (
        report["decoder"],
        *(boundaries[str(layer)] for layer in METOK_BOUNDARY_LAYERS),
    )


def _validate_metok_sample(
    *,
    baseline: Mapping[str, Any],
    compressed: Mapping[str, Any],
    projector_records: Sequence[Mapping[str, Any]],
    algorithm_records: Sequence[Mapping[str, Any]],
    pruning_records: Sequence[Mapping[str, Any]],
    cache_records: Sequence[Mapping[str, Any]],
    latest_splice: Mapping[str, Any],
    latest_metok: Sequence[Mapping[str, Any]],
    provider_metadata: Mapping[str, Any],
    provider_capture_delta: int,
    labels_absent: bool,
    placeholder_count: int,
    frame_count: int,
) -> dict[str, bool]:
    if not (
        len(projector_records) == len(algorithm_records) == 1
        and len(pruning_records)
        == len(cache_records)
        == len(latest_metok)
        == len(METOK_BOUNDARY_LAYERS)
    ):
        raise RuntimeError(
            "METok must publish one projector/algorithm record and three "
            "pruning/cache/bridge records per sample"
        )
    projector = projector_records[0]
    algorithm = algorithm_records[0]
    source_count = int(projector["source_count"])
    projector_count = int(projector["result_count"])
    baseline_first = int(baseline["decoder"]["prefill"]["query_length"])
    compressed_first = int(compressed["decoder"]["prefill"]["query_length"])
    baseline_boundaries = {
        layer: int(
            baseline["decoder_boundaries"][str(layer)]["prefill"]["query_length"]
        )
        for layer in METOK_BOUNDARY_LAYERS
    }
    compressed_boundaries = {
        layer: int(
            compressed["decoder_boundaries"][str(layer)]["prefill"]["query_length"]
        )
        for layer in METOK_BOUNDARY_LAYERS
    }
    frame_records = tuple(algorithm["frame_records"])
    category_counts = {
        f"important_{str(keyframe).lower()}": sum(
            bool(item["important"]) and bool(item["keyframe"]) is keyframe
            for item in frame_records
        )
        for keyframe in (True, False)
    }
    category_counts.update(
        {
            f"normal_{str(keyframe).lower()}": sum(
                not bool(item["important"]) and bool(item["keyframe"]) is keyframe
                for item in frame_records
            )
            for keyframe in (True, False)
        }
    )
    expected_source = frame_count * 196
    final_sequence = int(pruning_records[-1]["sequence_layout_delta"]["result_count"])
    structural_source = tuple(latest_splice["structural_source_positions"])
    structural_result = tuple(latest_splice["structural_result_positions"])
    source_visual_span = tuple(
        int(item) for item in latest_splice["source_visual_span"]
    )
    result_visual_span = tuple(
        int(item) for item in latest_splice["result_visual_span"]
    )
    checks = {
        "compression_applied": bool(latest_splice["compression_applied"]),
        "projector_non_identity_token_reduction": source_count > projector_count > 0,
        "native_32_by_14_by_14_raster_exact": (
            frame_count == 32
            and source_count == expected_source
            and int(algorithm["frame_count"]) == frame_count
            and int(algorithm["grid_height"]) == 14
            and int(algorithm["grid_width"]) == 14
            and int(algorithm["tokens_per_frame"]) == 196
        ),
        "native_siglip_27_by_27_raw_raster_exact": (
            int(algorithm["raw_tokens_per_frame"]) == 729
            and tuple(algorithm["raw_features_shape"])[0] == frame_count * 729
            and bool(algorithm["raw_features_finite"])
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
        "baseline_layer_lengths_stable": (
            all(value == baseline_first for value in baseline_boundaries.values())
        ),
        "pruning_boundary_lengths_exact": (
            int(pruning_records[0]["sequence_layout_delta"]["source_count"])
            == compressed_first
            and all(
                compressed_boundaries[layer]
                == int(pruning_records[index]["sequence_layout_delta"]["result_count"])
                for index, layer in enumerate(METOK_BOUNDARY_LAYERS)
            )
        ),
        "final_visual_removal_exact": (
            int(pruning_records[-1]["result_count"]) == 0
            and baseline_boundaries[19] - final_sequence == source_count
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
        "canonical_config_exact": _canonical_config_exact(
            algorithm["canonical_config"]
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
            and bool(algorithm["frame_text_similarity_finite"])
            and len(tuple(algorithm["frame_text_similarities"])) == frame_count
        ),
        "local_siglip_provider_exact": (
            algorithm["similarity_source"] == "native_siglip_visual_features"
            and bool(algorithm["assistant_answer_excluded"])
            and bool(algorithm["siglip_local_only"])
            and provider_capture_delta == 1
            and bool(provider_metadata["installed"])
            and bool(provider_metadata["siglip_loaded"])
            and provider_metadata["siglip_vision_enabled"] is False
            and bool(provider_metadata["siglip_frozen"])
            and bool(provider_metadata["siglip_local_only"])
            and bool(provider_metadata["assistant_answer_excluded"])
            and bool(provider_metadata["siglip_path"])
        ),
        "three_stage_pruning_and_cache_exact": _pruning_chain_exact(
            pruning_records,
            cache_records,
            latest_metok,
            projector_count=projector_count,
        ),
        "baseline_logits_finite": bool(baseline["logits_finite"]),
        "compressed_logits_finite": bool(compressed["logits_finite"]),
        "bounded_generation_exact": (
            int(baseline["generated_token_count"]) == 4
            and int(compressed["generated_token_count"]) == 4
            and all(
                tuple(summary["decode_query_lengths"]) == (1, 1, 1)
                for summary in (
                    *_all_decoder_summaries(baseline),
                    *_all_decoder_summaries(compressed),
                )
            )
        ),
        "baseline_decoder_alignment": all(
            all(summary["invariants"].values())
            for summary in _all_decoder_summaries(baseline)
        ),
        "compressed_decoder_alignment": all(
            all(summary["invariants"].values())
            for summary in _all_decoder_summaries(compressed)
        ),
    }
    if not all(checks.values()):
        raise RuntimeError(f"METok LLaVA sample evidence failed: {checks!r}")
    return checks


def _run(args: argparse.Namespace) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    visible = prepare_checkpoint_environment(args.physical_gpu)

    checkpoint = args.checkpoint.resolve()
    siglip_checkpoint = args.siglip_checkpoint.resolve()
    parquet_path = args.nextqa_parquet.resolve()
    video_root = args.video_root.resolve()

    checkpoint_info = _common._checkpoint_record(checkpoint)
    siglip_checkpoint_info = _common._checkpoint_record(siglip_checkpoint)
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
            "metok_siglip_path": str(siglip_checkpoint),
        },
    )
    handles = facade.model_adapter.structural_handles(model)
    first_layer = handles.get("llm.layer[0]")
    boundary_layers = {
        layer: handles.get(f"llm.layer[{layer}]") for layer in METOK_BOUNDARY_LAYERS
    }
    if first_layer is None or any(value is None for value in boundary_layers.values()):
        raise RuntimeError("LLaVA structural handles do not cover METok boundaries")
    first_layer_audit = _common._DecoderInputAudit(first_layer)
    boundary_audits = {
        layer: _common._DecoderInputAudit(owner)
        for layer, owner in boundary_layers.items()
    }
    activation = generation_activation(
        facade=facade,
        runtime=runtime,
        binding_name="metok.video.llava_onevision",
        stage=_stage(),
    )
    if (
        dict(activation.runtime_metadata()["canonical_config"])
        != CANONICAL_METOK_CONFIG
    ):
        raise RuntimeError(
            "Resolved METok configuration differs from the canonical profile"
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
                boundary_audits=boundary_audits,
                torch=torch,
                max_new_tokens=args.max_new_tokens,
            ) as (baseline, compressed, before, metadata):
                before_projector = int(before["projector_invocations"])
                before_pruning = int(before["pruning_invocations"])
                provider_before = before["signal_provider"] or {}
                provider_before_capture = int(provider_before.get("capture_count", 0))
                if (
                    int(metadata["projector_invocations"]) != before_projector + 1
                    or int(metadata["pruning_invocations"]) != before_pruning + 3
                ):
                    raise RuntimeError(
                        "METok projector/pruning invocation counts did not increment once/three"
                    )
                projector_records = metadata["projector_layout_deltas"]
                algorithm_records = metadata["algorithm_records"]
                pruning_records = metadata["pruning_layout_deltas"]
                cache_records = metadata["cache_repairs"]
                provider_metadata = metadata["signal_provider"]
                provider_capture_delta = (
                    int(provider_metadata["capture_count"]) - provider_before_capture
                )
                bridge_metadata = metadata["bridge"]
                latest_splice = bridge_metadata["latest_splice"]
                latest_metok = bridge_metadata["latest_metok"]
                if latest_splice is None or latest_metok is None:
                    raise RuntimeError(
                        "LLaVA bridge published incomplete METok splice/pruning evidence"
                    )
                checks = _validate_metok_sample(
                    baseline=baseline,
                    compressed=compressed,
                    projector_records=projector_records,
                    algorithm_records=algorithm_records,
                    pruning_records=pruning_records,
                    cache_records=cache_records,
                    latest_splice=latest_splice,
                    latest_metok=latest_metok,
                    provider_metadata=provider_metadata,
                    provider_capture_delta=provider_capture_delta,
                    labels_absent=labels_absent,
                    placeholder_count=placeholder_count,
                    frame_count=frame_count,
                )
                case_activation = {
                    **metadata,
                    "projector_layout_deltas": projector_records,
                    "algorithm_records": algorithm_records,
                    "pruning_layout_deltas": pruning_records,
                    "cache_repairs": cache_records,
                    "provider_capture_delta": provider_capture_delta,
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
                "bridge_latest_metok": latest_metok,
                "checks": checks,
            }
            cases[str(doc_id)] = case
            sample_rows.append(
                {
                    **generation_sample(document, baseline, compressed),
                    "compression_applied": checks["compression_applied"],
                    "source_visual_tokens": projector_records[0]["source_count"],
                    "projector_visual_tokens": projector_records[0]["result_count"],
                    "layer3_visual_tokens": pruning_records[0]["result_count"],
                    "layer10_visual_tokens": pruning_records[1]["result_count"],
                    "layer19_visual_tokens": pruning_records[2]["result_count"],
                }
            )
    finally:
        if activation.installed:
            activation.uninstall()
        first_layer_audit.close()
        for audit in boundary_audits.values():
            audit.close()

    torch.cuda.synchronize()
    gpu_after = _common._gpu_snapshot(args.physical_gpu)
    report = {
        "status": "passed",
        "scope": {
            "algorithm": "metok",
            "variant": "metok.video",
            "backbone": "llava_onevision",
            "binding": "metok.video.llava_onevision",
            "execution_phase": "generate",
            "canonical_config": CANONICAL_METOK_CONFIG,
            "quality_claim": "functional_migration_only",
        },
        "provenance": {
            "code_revision": args.code_revision,
            "reference_revision": VIDEO_REFERENCE_REVISION,
            "command": shlex.join(sys.argv),
            "checkpoint": checkpoint_info,
            "siglip_checkpoint": siglip_checkpoint_info,
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
    if args.max_new_tokens != 4:
        raise RuntimeError("METok fixed evidence requires max_new_tokens=4")
    return inference_main(args, _run)


if __name__ == "__main__":
    raise SystemExit(main())
