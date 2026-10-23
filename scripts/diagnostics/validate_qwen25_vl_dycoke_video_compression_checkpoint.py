"""Validate canonical DyCoKe video inference on Qwen2.5-VL.

The offline batch-one diagnostic runs the shared four ``nextqa_mc_test``
documents.  It freezes DyCoKe's two-pass temporal token merging, Qwen2.5-VL's
native video grid and sparse MRoPE legalization, and the non-destructive
decode-time KV views used from decoder layer 3 onward.
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
from scripts.diagnostics.qwen_video_checkpoint_common import qwen25_video_input_record
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
    from scripts.diagnostics import qwen_video_checkpoint_common as _common
else:
    import qwen_video_checkpoint_common as _common


FIXED_NEXTQA_DOC_IDS = _video_checkpoint_common.FIXED_NEXTQA_DOC_IDS
QWEN25_VIDEO_FPS = _common.QWEN25_VIDEO_FPS
QWEN25_VIDEO_MAX_PIXELS = _common.QWEN25_VIDEO_MAX_PIXELS
CANONICAL_DYCOKE_CONFIG = {
    "merging_ratio": 0.7,
    "decoder_prune_layer": 3,
    "kv_drop_ratio": 0.8,
    "attention_similarity_threshold": 0.9,
}


def _conversation(
    *,
    prompt: str,
    video: Path,
    fps: float = QWEN25_VIDEO_FPS,
    max_pixels: int = QWEN25_VIDEO_MAX_PIXELS,
) -> list[dict[str, Any]]:
    return _common._conversation(
        prompt=prompt,
        video=video,
        fps=fps,
        max_pixels=max_pixels,
    )


def _stage() -> CompressionStageConfig:
    return CompressionStageConfig.from_dict(
        {
            "name": "dycoke-video-qwen25-vl-real-checkpoint",
            "placement": "projector_level",
            "execution_phase": "generate",
            "plugin": "dycoke",
            "algorithm": "dycoke",
            "variant": "dycoke.video",
            "target_modalities": ["video"],
            "conditioning_modalities": ["text"],
            "preserved_modalities": ["text", "image", "audio"],
            "coupling_policy": "independent",
            "media_alignment": "spatiotemporal",
            "binding": "dycoke.video.qwen25_vl",
            "config": {},
        }
    )


def _generation_report(
    *,
    facade: Any,
    runtime: BackboneRuntime,
    state: Any,
    batch: Any,
    audit: Any,
    torch: Any,
    max_new_tokens: int,
) -> dict[str, Any]:
    report = _common.generation_report(
        facade=facade,
        runtime=runtime,
        state=state,
        batch=batch,
        audit=audit,
        torch=torch,
        max_new_tokens=max_new_tokens,
        min_new_tokens=max_new_tokens,
    )
    if report["generated_token_count"] != max_new_tokens:
        raise RuntimeError(
            "DyCoKe requires the exact bounded decode length: "
            f"count={report['generated_token_count']}, max_new_tokens={max_new_tokens}"
        )
    return report


def _expected_frame_result_counts(
    frame_count: int,
    tokens_per_frame: int,
) -> tuple[int, ...]:
    if frame_count < 1 or tokens_per_frame < 1:
        raise ValueError("DyCoKe frame dimensions must be positive")
    keep = max(
        1,
        int((1.0 - float(CANONICAL_DYCOKE_CONFIG["merging_ratio"])) * tokens_per_frame),
    )
    counts = []
    for frame_index in range(frame_count):
        if frame_index % 4 == 0 or (
            frame_index % 2 == 0 and frame_index + 1 >= frame_count
        ):
            counts.append(tokens_per_frame)
        else:
            counts.append(keep)
    return tuple(counts)


def _validate_decode_records(
    records: Sequence[Mapping[str, Any]],
    *,
    compressed_prefill_length: int,
    result_visual_positions: Sequence[int],
    decode_query_count: int,
) -> bool:
    visual = tuple(int(item) for item in result_visual_positions)
    expected_keep = max(
        1,
        int(len(visual) * (1.0 - float(CANONICAL_DYCOKE_CONFIG["kv_drop_ratio"]))),
    )
    if len(records) != decode_query_count or not records:
        return False
    for index, record in enumerate(records):
        sequence_length = int(record["sequence_length"])
        kept = tuple(int(item) for item in record["kept_visual_positions"])
        view = tuple(int(item) for item in record["view_indices"])
        record_visual = tuple(int(item) for item in record["visual_positions"])
        expected_view = tuple(
            position
            for position in range(sequence_length)
            if position not in set(visual) or position in set(kept)
        )
        if not (
            sequence_length == compressed_prefill_length + index + 1
            and record_visual == visual
            and int(record["visual_count"]) == len(visual)
            and int(record["kept_visual_count"]) == expected_keep
            and int(record["dropped_visual_count"]) == len(visual) - expected_keep
            and len(kept) == expected_keep
            and kept == tuple(sorted(set(kept)))
            and set(kept).issubset(visual)
            and int(record["view_length"]) == len(expected_view)
            and view == expected_view
            and view[-1] == sequence_length - 1
            and math.isfinite(float(record["attention_similarity"]))
            and -1.0 <= float(record["attention_similarity"]) <= 1.0
            and isinstance(record["reused_visual_selection"], bool)
            and record["physical_cache_retained"] is True
            and int(record["attention_source_layer"]) == 2
            and int(record["first_sparse_layer"]) == 3
        ):
            return False
    return records[0]["reused_visual_selection"] is False


def _decode_repair_exact(
    latest_generation: Mapping[str, Any],
    *,
    baseline_length: int,
    compressed_length: int,
    decode_query_count: int,
    removed_count: int,
) -> bool:
    if int(latest_generation["decode_repairs"]) != decode_query_count:
        return False
    record = latest_generation.get("last_decode_record")
    if not isinstance(record, Mapping):
        return False
    already_compact = record.get("already_compact")
    input_length = record.get("input_attention_length")
    if already_compact is True:
        return input_length in (None, compressed_length + decode_query_count)
    return bool(
        already_compact is False
        and int(input_length) == baseline_length + decode_query_count
        and int(record["output_attention_length"])
        == compressed_length + decode_query_count
        and int(record["removed_count"]) == removed_count
    )


def _validate_dycoke_sample(
    *,
    baseline: Mapping[str, Any],
    compressed: Mapping[str, Any],
    activation_records: Sequence[Mapping[str, Any]],
    algorithm_records: Sequence[Mapping[str, Any]],
    decode_records: Sequence[Mapping[str, Any]],
    latest_splice: Mapping[str, Any],
    latest_generation: Mapping[str, Any],
    source_grid_thw: Sequence[int],
    source_grid_token_count: int,
    spatial_merge_size: int,
    second_per_grid_seconds: float,
    labels_absent: bool,
    video_span_count: int,
) -> dict[str, bool]:
    if len(activation_records) != 1 or len(algorithm_records) != 1:
        raise RuntimeError(
            "DyCoKe must publish one layout and one algorithm record per sample"
        )
    if len(source_grid_thw) != 3 or spatial_merge_size < 1:
        raise RuntimeError("Qwen2.5-VL source grid metadata is invalid")
    temporal, height, width = (int(item) for item in source_grid_thw)
    spatial_area = height * width
    merge_area = spatial_merge_size**2
    if spatial_area % merge_area:
        raise RuntimeError("Qwen2.5-VL grid is not divisible by spatial merge size")
    tokens_per_frame = spatial_area // merge_area
    expected_frame_counts = _expected_frame_result_counts(
        temporal,
        tokens_per_frame,
    )
    expected_source = temporal * tokens_per_frame
    expected_result = sum(expected_frame_counts)

    layout = activation_records[0]
    algorithm = algorithm_records[0]
    source_count = int(layout["source_count"])
    result_count = int(layout["result_count"])
    baseline_length = int(baseline["decoder"]["prefill"]["query_length"])
    compressed_length = int(compressed["decoder"]["prefill"]["query_length"])
    decode_queries = tuple(
        int(item) for item in compressed["decoder"]["decode_query_lengths"]
    )
    frame_records_by_video = tuple(algorithm["frame_records_by_video"])
    frame_records = (
        () if len(frame_records_by_video) != 1 else tuple(frame_records_by_video[0])
    )
    selected_sources = tuple(
        int(item)
        for frame in frame_records
        for item in frame["selected_source_indices"]
    )
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
    result_span = tuple(int(item) for item in latest_splice["result_visual_span"])
    expected_result_positions = tuple(range(result_span[0], result_span[1]))
    result_positions = tuple(int(item) for item in layout["result_visual_positions"])
    splice_layout = latest_splice["layout_delta"]
    rope_deltas = tuple(int(item) for item in latest_splice["rope_deltas"])

    checks = {
        "compression_applied": bool(latest_splice["compression_applied"]),
        "non_identity_token_reduction": source_count > result_count,
        "one_contiguous_video_span": video_span_count == 1,
        "native_grid_source_count_exact": (
            source_grid_token_count == source_count == expected_source
        ),
        "canonical_ttm_result_exact": result_count == expected_result,
        "layout_splice_algorithm_counts_agree": (
            int(splice_layout["source_count"]) == source_count
            and int(splice_layout["result_count"]) == result_count
            and int(latest_splice["source_placeholder_count"]) == source_count
            and int(latest_splice["result_placeholder_count"]) == result_count
            and int(algorithm["source_count"]) == source_count
            and int(algorithm["result_count"]) == result_count
            and tuple(int(item) for item in algorithm["exact_budgets"])
            == (result_count,)
        ),
        "decoder_length_reduction_exact": (
            baseline_length - compressed_length == source_count - result_count
        ),
        "source_video_grid_preserved": (
            latest_splice["source_video_grid_preserved"] is True
            and tuple(int(item) for item in latest_splice["source_video_grid_thw"])
            == (temporal, height, width)
        ),
        "video_timing_preserved": math.isclose(
            float(latest_splice["second_per_grid_seconds"]),
            second_per_grid_seconds,
            rel_tol=0.0,
            abs_tol=1e-9,
        ),
        "sparse_grid_policy_exact": (
            latest_splice["result_grid_policy"] == "irregular_sparse_source_anchors"
        ),
        "representatives_source_ordered": (
            representatives == selected_sources
            and len(representatives) == result_count
            and representatives == tuple(sorted(set(representatives)))
            and algorithm["representatives_source_ordered"] is True
        ),
        "mrope_source_count_aligned": len(source_mrope) == source_count,
        "mrope_result_count_aligned": len(result_mrope) == result_count,
        "mrope_sparse_anchor_gather_exact": result_mrope
        == tuple(source_mrope[index] for index in representatives),
        "rope_delta_scalar_recorded": len(rope_deltas) == 1,
        "result_visual_positions_exact": (
            result_positions == expected_result_positions
            and tuple(int(item) for item in algorithm["result_visual_positions"])
            == result_positions
            and len(result_positions) == result_count
        ),
        "canonical_config_exact": dict(algorithm["canonical_config"])
        == CANONICAL_DYCOKE_CONFIG,
        "decode_stage_exact": (
            algorithm["decode_stage"] == "full_cache_sparse_view_from_layer_3"
            and algorithm["physical_cache_retained"] is True
        ),
        "decode_queries_exact": len(decode_queries) == 3
        and all(item == 1 for item in decode_queries),
        "decode_kv_views_exact": _validate_decode_records(
            decode_records,
            compressed_prefill_length=compressed_length,
            result_visual_positions=result_positions,
            decode_query_count=len(decode_queries),
        ),
        "generation_prefill_complete": latest_generation["prefill_complete"] is True,
        "generation_lengths_exact": (
            int(latest_generation["source_sequence_length"]) == baseline_length
            and int(latest_generation["result_sequence_length"]) == compressed_length
            and int(latest_generation["removed_count"]) == source_count - result_count
        ),
        "decode_repair_exact": _decode_repair_exact(
            latest_generation,
            baseline_length=baseline_length,
            compressed_length=compressed_length,
            decode_query_count=len(decode_queries),
            removed_count=source_count - result_count,
        ),
        "inference_labels_absent": labels_absent,
        "baseline_logits_finite": baseline["logits_finite"] is True,
        "compressed_logits_finite": compressed["logits_finite"] is True,
        "baseline_decoder_alignment": all(baseline["decoder"]["invariants"].values()),
        "compressed_decoder_alignment": all(
            compressed["decoder"]["invariants"].values()
        ),
        "baseline_bounded_generation": int(baseline["generated_token_count"]) == 4,
        "compressed_bounded_generation": int(compressed["generated_token_count"]) == 4,
    }
    if not all(checks.values()):
        raise RuntimeError(f"DyCoKe Qwen2.5-VL sample evidence failed: {checks!r}")
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
        binding_name="dycoke.video.qwen25_vl",
        stage=_stage(),
    )
    if (
        dict(activation.runtime_metadata()["canonical_config"])
        != CANONICAL_DYCOKE_CONFIG
    ):
        raise RuntimeError("Resolved DyCoKe configuration differs from canonical")

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
            ) as (baseline, compressed, before, metadata):
                before_projector = int(before["projector_invocations"])
                if int(metadata["projector_invocations"]) != before_projector + 1:
                    raise RuntimeError(
                        "DyCoKe projector invocation count did not increment once"
                    )
                layout_records = metadata["projector_layout_deltas"]
                algorithm_records = metadata["algorithm_records"]
                decode_records = metadata["decode_kv_views"]
                bridge_metadata = metadata["bridge"]
                latest_splice = bridge_metadata["latest_splice"]
                latest_generation = bridge_metadata["latest_generation"]
                if latest_splice is None or latest_generation is None:
                    raise RuntimeError(
                        "Qwen2.5-VL bridge published incomplete splice/generation evidence"
                    )
                checks = _validate_dycoke_sample(
                    baseline=baseline,
                    compressed=compressed,
                    activation_records=layout_records,
                    algorithm_records=algorithm_records,
                    decode_records=decode_records,
                    latest_splice=latest_splice,
                    latest_generation=latest_generation,
                    source_grid_thw=native_video["video_grid_thw"],
                    source_grid_token_count=native_video["source_grid_token_count"],
                    spatial_merge_size=native_video["spatial_merge_size"],
                    second_per_grid_seconds=native_video["second_per_grid_ts"][0],
                    labels_absent=native_video["labels_absent_for_inference"],
                    video_span_count=1,
                )
                case_activation = {
                    **metadata,
                    "projector_layout_deltas": layout_records,
                    "algorithm_records": algorithm_records,
                    "decode_kv_views": decode_records,
                }

            case = {
                **generation_case(document, video),
                "native_video": {
                    **native_video,
                    "conversation": conversation,
                    "requested_fps": float(args.video_fps),
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
                    "decode_kv_view_count": len(decode_records),
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
            "algorithm": "dycoke",
            "variant": "dycoke.video",
            "backbone": "qwen25_vl",
            "binding": "dycoke.video.qwen25_vl",
            "execution_phase": "generate",
            "canonical_config": CANONICAL_DYCOKE_CONFIG,
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
            "physical_cache_compaction": False,
        },
    }
    return report, sample_rows


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    add_checkpoint_arguments(parser, estimated_peak_mib=49152)
    parser.add_argument("--max_new_tokens", type=int, default=4)
    parser.add_argument("--video_fps", type=float, default=QWEN25_VIDEO_FPS)
    parser.add_argument("--video_max_pixels", type=int, default=QWEN25_VIDEO_MAX_PIXELS)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.max_new_tokens != 4:
        raise RuntimeError("DyCoKe evidence freezes max_new_tokens=4")
    if args.video_fps <= 0 or args.video_max_pixels <= 0:
        raise RuntimeError("video_fps and video_max_pixels must be positive")
    return inference_main(args, _run)


if __name__ == "__main__":
    raise SystemExit(main())
