"""Validate canonical METok video inference on Qwen2.5-VL.

The diagnostic is offline, batch-one, and single-GPU. It runs the same four
fixed nextqa_mc_test documents as the other video cells and requires the
complete METok path: the existing local frozen SigLIP double tower, four-event
projector pooling, and decoder pruning before layers 3, 10, and 19.

The source video_grid_thw and timing remain authoritative. Projector and
decoder compaction gather native M-RoPE coordinates through their exact keep
vectors, publish the final rope delta, and repair bounded cached decode.
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
    from scripts.diagnostics import (
        validate_llava_metok_video_compression_checkpoint as _metok_common,
    )
else:
    import qwen_video_checkpoint_common as _qwen_common
    import validate_llava_metok_video_compression_checkpoint as _metok_common


FIXED_NEXTQA_DOC_IDS = _video_checkpoint_common.FIXED_NEXTQA_DOC_IDS
METOK_BOUNDARY_LAYERS = _metok_common.METOK_BOUNDARY_LAYERS
CANONICAL_METOK_CONFIG = _metok_common.CANONICAL_METOK_CONFIG
QWEN25_METOK_VIDEO_FPS = _qwen_common.QWEN25_VIDEO_FPS
QWEN25_METOK_VIDEO_MAX_PIXELS = _qwen_common.QWEN25_VIDEO_MAX_PIXELS


def _conversation(
    *,
    prompt: str,
    video: Path,
    fps: float = QWEN25_METOK_VIDEO_FPS,
    max_pixels: int = QWEN25_METOK_VIDEO_MAX_PIXELS,
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
            "name": "metok-video-qwen25-vl-real-checkpoint",
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
            "binding": "metok.video.qwen25_vl",
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
    report = _qwen_common.generation_report(
        facade=facade,
        runtime=runtime,
        state=state,
        batch=batch,
        audit=first_layer_audit,
        torch=torch,
        max_new_tokens=max_new_tokens,
    )
    report["decoder_boundaries"] = {
        str(layer): _video_checkpoint_common._decoder_summary(audit.since(marks[layer]))
        for layer, audit in boundary_audits.items()
    }
    return report


def _all_decoder_summaries(
    report: Mapping[str, Any],
) -> tuple[Mapping[str, Any], ...]:
    boundaries = report.get("decoder_boundaries", {})
    return (
        report["decoder"],
        *(boundaries[str(layer)] for layer in METOK_BOUNDARY_LAYERS),
    )


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
    prior_result_visual_mrope: tuple[tuple[int, ...], ...] | None = None
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
        cumulative_prompt = tuple(
            int(item) for item in record.get("cumulative_prompt_keep_indices", ())
        )
        source_visual_positions = tuple(
            int(item) for item in bridge.get("source_visual_sequence_indices", ())
        )
        result_visual_positions = tuple(
            int(item) for item in bridge.get("result_visual_sequence_indices", ())
        )
        source_mrope = tuple(
            tuple(int(value) for value in row)
            for row in bridge.get("source_visual_mrope", ())
        )
        result_mrope = tuple(
            tuple(int(value) for value in row)
            for row in bridge.get("result_visual_mrope", ())
        )
        expected_result_mrope = tuple(source_mrope[index] for index in kept)
        expected_rope_delta = int(bridge.get("expected_rope_delta", 2**31))
        rope_deltas = tuple(int(item) for item in bridge.get("rope_deltas", ()))
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
            and len(cumulative_prompt) == result_sequence
            and bool(record.get("final_visual_removal"))
            is (stage_index == len(METOK_BOUNDARY_LAYERS) - 1)
            and dict(record.get("cache", {})) == dict(cache)
            and _metok_common._cache_layers_exact(
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
            and tuple(
                int(item) for item in bridge.get("cumulative_prompt_keep_indices", ())
            )
            == cumulative_prompt
            and len(source_visual_positions) == source_visual
            and len(result_visual_positions) == result_visual
            and source_visual_positions == tuple(sorted(set(source_visual_positions)))
            and result_visual_positions == tuple(sorted(set(result_visual_positions)))
            and len(source_mrope) == source_visual
            and len(result_mrope) == result_visual
            and all(len(row) == 3 for row in (*source_mrope, *result_mrope))
            and result_mrope == expected_result_mrope
            and bool(bridge.get("rope_delta_exact"))
            and int(bridge.get("active_position_count", -1)) == result_sequence
            and expected_rope_delta
            == int(bridge.get("max_active_rotary_position", -1)) + 1 - result_sequence
            and rope_deltas == (expected_rope_delta,)
        ):
            return False
        if (
            prior_sequence_result is not None
            and source_sequence != prior_sequence_result
        ):
            return False
        if (
            prior_result_visual_mrope is not None
            and source_mrope != prior_result_visual_mrope
        ):
            return False
        prior_visual = result_visual

        prior_sequence_result = result_sequence
        prior_result_visual_mrope = result_mrope
    return prior_visual == 0


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
    latest_generation: Mapping[str, Any],
    provider_metadata: Mapping[str, Any],
    provider_capture_delta: int,
    source_grid_thw: Sequence[int],
    source_grid_token_count: int,
    spatial_merge_size: int,
    second_per_grid_seconds: float | None,
    labels_absent: bool,
    video_span_count: int,
    provider_name: str = "qwen25_vl_metok_video_signals",
    backbone_label: str = "Qwen2.5-VL",
    video_spans_valid: bool | None = None,
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
    if len(source_grid_thw) != 3:
        raise RuntimeError(f"Invalid {backbone_label} source grid: {source_grid_thw!r}")
    temporal, height, width = (int(item) for item in source_grid_thw)
    if (
        spatial_merge_size < 1
        or height % spatial_merge_size
        or width % spatial_merge_size
    ):
        raise RuntimeError(
            f"{backbone_label} source grid is incompatible with its spatial merger"
        )
    grid_height = height // spatial_merge_size
    grid_width = width // spatial_merge_size
    tokens_per_frame = grid_height * grid_width
    raw_tokens_per_frame = height * width
    expected_source_count = temporal * tokens_per_frame

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
    final_sequence = int(pruning_records[-1]["sequence_layout_delta"]["result_count"])
    representatives = tuple(int(item) for item in algorithm["representative_indices"])
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
    expected_projector_mrope = tuple(
        source_mrope[index] for index in splice_representatives
    )
    generated_token_count = int(compressed["generated_token_count"])
    removed_count = source_count
    decode_query_lengths = tuple(
        int(item) for item in compressed["decoder"]["decode_query_lengths"]
    )
    final_rope_deltas = tuple(int(item) for item in latest_metok[-1]["rope_deltas"])
    checks = {
        "compression_applied": bool(latest_splice["compression_applied"]),
        "projector_non_identity_token_reduction": source_count > projector_count > 0,
        "one_contiguous_video_span": (
            video_span_count == 1
            if video_spans_valid is None
            else bool(video_spans_valid)
        ),
        "native_grid_and_source_counts_agree": (
            source_count == expected_source_count
            and source_grid_token_count == source_count
            and int(latest_splice["source_placeholder_count"]) == source_count
            and int(algorithm["frame_count"]) == temporal
            and int(algorithm["grid_height"]) == grid_height
            and int(algorithm["grid_width"]) == grid_width
            and int(algorithm["tokens_per_frame"]) == tokens_per_frame
        ),
        "native_raw_raster_exact": (
            int(algorithm["raw_tokens_per_frame"]) == raw_tokens_per_frame
            and tuple(algorithm["raw_features_shape"])[0]
            == temporal * raw_tokens_per_frame
            and bool(algorithm["raw_features_finite"])
        ),
        "projector_layout_and_splice_counts_agree": (
            int(latest_splice["result_placeholder_count"]) == projector_count
            and int(latest_splice["layout_delta"]["source_count"]) == source_count
            and int(latest_splice["layout_delta"]["result_count"]) == projector_count
            and int(algorithm["source_count"]) == source_count
            and int(algorithm["result_count"]) == projector_count
        ),
        "projector_decoder_length_reduction_exact": (
            baseline_first - compressed_first == source_count - projector_count
        ),
        "baseline_layer_lengths_stable": all(
            value == baseline_first for value in baseline_boundaries.values()
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
        "source_video_grid_preserved": (
            bool(latest_splice["source_video_grid_preserved"])
            and tuple(int(item) for item in latest_splice["source_video_grid_thw"])
            == (temporal, height, width)
        ),
        "source_video_timing_preserved": (
            latest_splice.get("second_per_grid_seconds") is None
            if second_per_grid_seconds is None
            else (
                math.isfinite(second_per_grid_seconds)
                and second_per_grid_seconds > 0
                and math.isclose(
                    float(latest_splice["second_per_grid_seconds"]),
                    second_per_grid_seconds,
                    rel_tol=0.0,
                    abs_tol=1e-9,
                )
            )
        ),
        "sparse_grid_policy": (
            latest_splice["result_grid_policy"] == "irregular_sparse_source_anchors"
        ),
        "projector_representatives_exact": (
            representatives == splice_representatives
            and len(representatives) == projector_count
            and representatives == tuple(sorted(set(representatives)))
        ),
        "projector_mrope_sparse_anchor_gather_exact": (
            len(source_mrope) == source_count
            and len(result_mrope) == projector_count
            and all(len(row) == 3 for row in (*source_mrope, *result_mrope))
            and result_mrope == expected_projector_mrope
        ),
        "inference_labels_absent": labels_absent,
        "canonical_config_exact": _metok_common._canonical_config_exact(
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
            and len(tuple(algorithm["frame_text_similarities"])) == temporal
        ),
        "local_siglip_provider_exact": (
            algorithm["similarity_source"] == "qwen_patchified_video_inverse_siglip_384"
            and bool(algorithm["assistant_answer_excluded"])
            and bool(algorithm["siglip_local_only"])
            and provider_capture_delta == 1
            and provider_metadata["provider"] == provider_name
            and bool(provider_metadata["installed"])
            and bool(provider_metadata["video_only"])
            and bool(provider_metadata["raw_stream_attached"])
            and bool(provider_metadata["siglip_loaded"])
            and provider_metadata["siglip_vision_enabled"] is True
            and bool(provider_metadata["siglip_frozen"])
            and bool(provider_metadata["siglip_local_only"])
            and bool(provider_metadata["assistant_answer_excluded"])
            and bool(provider_metadata["siglip_path"])
        ),
        "three_stage_pruning_cache_and_mrope_exact": _pruning_chain_exact(
            pruning_records,
            cache_records,
            latest_metok,
            projector_count=projector_count,
        ),
        "final_rope_delta_published": (
            tuple(int(item) for item in latest_splice["rope_deltas"])
            == final_rope_deltas
        ),
        "generation_prefill_complete": bool(latest_generation["prefill_complete"]),
        "generation_source_length_aligned": (
            int(latest_generation["source_sequence_length"]) == baseline_first
        ),
        "generation_result_length_aligned": (
            int(latest_generation["result_sequence_length"]) == final_sequence
        ),
        "generation_removed_count_aligned": (
            int(latest_generation["removed_count"]) == removed_count
        ),
        "cached_decode_exact": (
            generated_token_count == 4
            and int(latest_generation["decode_repairs"]) == 3
            and decode_query_lengths == (1, 1, 1)
            and _decode_record_coherent(
                latest_generation.get("last_decode_record"),
                removed_count=removed_count,
                result_sequence_length=final_sequence,
            )
        ),
        "baseline_logits_finite": bool(baseline["logits_finite"]),
        "compressed_logits_finite": bool(compressed["logits_finite"]),
        "bounded_generation_exact": (
            int(baseline["generated_token_count"]) == 4
            and generated_token_count == 4
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
        raise RuntimeError(f"METok {backbone_label} sample evidence failed: {checks!r}")
    return checks


def _run(args: argparse.Namespace) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    visible = prepare_checkpoint_environment(args.physical_gpu)

    checkpoint = args.checkpoint.resolve()
    siglip_checkpoint = args.siglip_checkpoint.resolve()
    parquet_path = args.nextqa_parquet.resolve()
    video_root = args.video_root.resolve()

    checkpoint_info = _video_checkpoint_common._checkpoint_record(checkpoint)
    siglip_checkpoint_info = _video_checkpoint_common._checkpoint_record(
        siglip_checkpoint
    )
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
            "metok_siglip_path": str(siglip_checkpoint),
        },
    )
    handles = facade.model_adapter.structural_handles(model)
    first_layer = handles.get("llm.layer[0]")
    boundary_layers = {
        layer: handles.get(f"llm.layer[{layer}]") for layer in METOK_BOUNDARY_LAYERS
    }
    if first_layer is None or any(value is None for value in boundary_layers.values()):
        raise RuntimeError(
            "Qwen2.5-VL structural handles do not cover METok boundaries"
        )
    first_layer_audit = _video_checkpoint_common._DecoderInputAudit(first_layer)
    boundary_audits = {
        layer: _video_checkpoint_common._DecoderInputAudit(owner)
        for layer, owner in boundary_layers.items()
    }
    activation = generation_activation(
        facade=facade,
        runtime=runtime,
        binding_name="metok.video.qwen25_vl",
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
                source_grid_thw[0] < 4
                or source_grid_thw[1] % merge_size
                or source_grid_thw[2] % merge_size
                or per_slice_source_count < 4
            ):
                raise RuntimeError(
                    "Frozen Qwen2.5-VL preprocessing cannot exercise canonical "
                    f"METok: grid={source_grid_thw!r}, "
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
                        "METok projector/pruning invocation counts did not increment "
                        "once/three"
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
                latest_generation = bridge_metadata["latest_generation"]
                if (
                    latest_splice is None
                    or latest_metok is None
                    or latest_generation is None
                ):
                    raise RuntimeError(
                        "Qwen2.5-VL bridge published incomplete "
                        "splice/METok/generation evidence"
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
                    latest_generation=latest_generation,
                    provider_metadata=provider_metadata,
                    provider_capture_delta=provider_capture_delta,
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
                    "pruning_layout_deltas": pruning_records,
                    "cache_repairs": cache_records,
                    "provider_capture_delta": provider_capture_delta,
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
                "bridge_latest_metok": latest_metok,
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
    gpu_after = _video_checkpoint_common._gpu_snapshot(args.physical_gpu)
    report = {
        "status": "passed",
        "scope": {
            "algorithm": "metok",
            "variant": "metok.video",
            "backbone": "qwen25_vl",
            "binding": "metok.video.qwen25_vl",
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
            "video_preprocessing": {
                "fps": float(args.video_fps),
                "max_pixels": int(args.video_max_pixels),
                "provider": "qwen_vl_utils.process_vision_info",
                "constraint": (
                    "native source grid remains authoritative while METok uses "
                    "its post-merger raster and local SigLIP-reconstructed frames"
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
    parser.add_argument("--siglip_checkpoint", type=Path, required=True)
    parser.add_argument("--max_new_tokens", type=int, default=4)
    parser.add_argument("--video_fps", type=float, default=QWEN25_METOK_VIDEO_FPS)
    parser.add_argument(
        "--video_max_pixels",
        type=int,
        default=QWEN25_METOK_VIDEO_MAX_PIXELS,
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.max_new_tokens != 4:
        raise RuntimeError("METok fixed evidence requires max_new_tokens=4")
    return inference_main(args, _run)


if __name__ == "__main__":
    raise SystemExit(main())
