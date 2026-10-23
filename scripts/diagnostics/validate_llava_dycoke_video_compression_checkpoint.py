"""Validate canonical DyCoKe video inference on LLaVA-OneVision.

The offline, batch-one diagnostic runs the same four fixed
``nextqa_mc_test`` documents as every other inference cell.  It freezes both
defining DyCoKe stages: two-pass temporal token merging over native LLaVA
frame rasters and decode-time sparse KV views derived from layer-2 attention
for layers 3 and above.  The physical Transformers 4.57.3 ``DynamicCache``
must remain complete throughout generation.
"""

from __future__ import annotations

import argparse
import math
import shlex
import sys
import time
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
CANONICAL_DYCOKE_CONFIG = {
    "merging_ratio": 0.7,
    "decoder_prune_layer": 3,
    "kv_drop_ratio": 0.8,
    "attention_similarity_threshold": 0.9,
}
LLAVA_FRAME_COUNT = 32
LLAVA_PATCHES_PER_FRAME = 196


def _stage() -> CompressionStageConfig:
    return CompressionStageConfig.from_dict(
        {
            "name": "dycoke-video-llava-onevision-real-checkpoint",
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
            "binding": "dycoke.video.llava_onevision",
            "config": {},
        }
    )


def _expected_frame_result_counts(frame_count: int) -> tuple[int, ...]:
    keep = max(
        1,
        int(
            (1.0 - float(CANONICAL_DYCOKE_CONFIG["merging_ratio"]))
            * LLAVA_PATCHES_PER_FRAME
        ),
    )
    counts = []
    for frame_index in range(frame_count):
        if frame_index % 4 == 0 or (
            frame_index % 2 == 0 and frame_index + 1 >= frame_count
        ):
            counts.append(LLAVA_PATCHES_PER_FRAME)
        else:
            counts.append(keep)
    return tuple(counts)


def _validate_decode_records(
    records: Sequence[Mapping[str, Any]],
    *,
    compressed_prefill_length: int,
    result_visual_count: int,
    decode_query_count: int,
) -> bool:
    expected_keep = max(
        1,
        int(
            result_visual_count
            * (1.0 - float(CANONICAL_DYCOKE_CONFIG["kv_drop_ratio"]))
        ),
    )
    if len(records) != decode_query_count or not records:
        return False
    for index, record in enumerate(records):
        sequence_length = int(record["sequence_length"])
        visual_count = int(record["visual_count"])
        kept_count = int(record["kept_visual_count"])
        dropped_count = int(record["dropped_visual_count"])
        view = tuple(int(item) for item in record["view_indices"])
        kept = tuple(int(item) for item in record["kept_visual_positions"])
        visual = tuple(int(item) for item in record["visual_positions"])
        if not (
            sequence_length == compressed_prefill_length + index + 1
            and visual_count == result_visual_count
            and kept_count == expected_keep
            and dropped_count == visual_count - kept_count
            and int(record["view_length"]) == sequence_length - dropped_count
            and len(view) == int(record["view_length"])
            and view == tuple(sorted(set(view)))
            and view[-1] == sequence_length - 1
            and kept == tuple(sorted(set(kept)))
            and set(kept).issubset(visual)
            and math.isfinite(float(record["attention_similarity"]))
            and isinstance(record["reused_visual_selection"], bool)
            and record["physical_cache_retained"] is True
            and int(record["attention_source_layer"]) == 2
            and int(record["first_sparse_layer"]) == 3
        ):
            return False
    return records[0]["reused_visual_selection"] is False


def _validate_dycoke_sample(
    *,
    baseline: Mapping[str, Any],
    compressed: Mapping[str, Any],
    activation_records: Sequence[Mapping[str, Any]],
    algorithm_records: Sequence[Mapping[str, Any]],
    decode_records: Sequence[Mapping[str, Any]],
    latest_splice: Mapping[str, Any],
    labels_absent: bool,
    placeholder_count: int,
    frame_count: int,
) -> dict[str, bool]:
    if len(activation_records) != 1 or len(algorithm_records) != 1:
        raise RuntimeError(
            "DyCoKe must publish one layout and one algorithm record per sample"
        )
    layout = activation_records[0]
    algorithm = algorithm_records[0]
    source_count = int(layout["source_count"])
    result_count = int(layout["result_count"])
    baseline_length = int(baseline["decoder"]["prefill"]["query_length"])
    compressed_length = int(compressed["decoder"]["prefill"]["query_length"])
    structural_source = tuple(latest_splice["structural_source_positions"])
    structural_result = tuple(latest_splice["structural_result_positions"])
    expected_frame_counts = _expected_frame_result_counts(frame_count)
    expected_source = frame_count * LLAVA_PATCHES_PER_FRAME
    expected_result = sum(expected_frame_counts)
    compressed_decode_queries = tuple(
        int(item) for item in compressed["decoder"]["decode_query_lengths"]
    )
    checks = {
        "compression_applied": bool(latest_splice["compression_applied"]),
        "non_identity_token_reduction": source_count > result_count,
        "native_32_by_196_raster_exact": (
            frame_count == LLAVA_FRAME_COUNT and source_count == expected_source == 6272
        ),
        "canonical_ttm_result_exact": result_count == expected_result == 2960,
        "layout_and_splice_counts_agree": (
            int(latest_splice["source_patch_count"]) == source_count
            and int(latest_splice["result_patch_count"]) == result_count
            and int(algorithm["source_count"]) == source_count
            and int(algorithm["result_count"]) == result_count
            and tuple(int(item) for item in algorithm["exact_budgets"])
            == (result_count,)
        ),
        "decoder_length_reduction_exact": (
            baseline_length - compressed_length == source_count - result_count
        ),
        "structural_tokens_preserved": len(structural_source) == len(structural_result),
        "one_video_placeholder": placeholder_count == 1,
        "inference_labels_absent": labels_absent,
        "canonical_config_exact": dict(algorithm["canonical_config"])
        == CANONICAL_DYCOKE_CONFIG,
        "representatives_source_ordered": bool(
            algorithm["representatives_source_ordered"]
        ),
        "result_visual_positions_exact": (
            len(tuple(algorithm["result_visual_positions"])) == result_count
            and tuple(algorithm["result_visual_positions"])
            == tuple(layout["result_visual_positions"])
        ),
        "decode_stage_exact": (
            algorithm["decode_stage"] == "full_cache_sparse_view_from_layer_3"
            and algorithm["physical_cache_retained"] is True
        ),
        "decode_queries_bounded": (
            len(compressed_decode_queries) >= 1
            and all(item == 1 for item in compressed_decode_queries)
        ),
        "decode_kv_views_exact": _validate_decode_records(
            decode_records,
            compressed_prefill_length=compressed_length,
            result_visual_count=result_count,
            decode_query_count=len(compressed_decode_queries),
        ),
        "baseline_logits_finite": bool(baseline["logits_finite"]),
        "compressed_logits_finite": bool(compressed["logits_finite"]),
        "baseline_decoder_alignment": all(baseline["decoder"]["invariants"].values()),
        "compressed_decoder_alignment": all(
            compressed["decoder"]["invariants"].values()
        ),
        "baseline_bounded_generation": (
            1 <= int(baseline["generated_token_count"]) <= 4
        ),
        "compressed_bounded_generation": (
            1 <= int(compressed["generated_token_count"]) <= 4
        ),
    }
    if not all(checks.values()):
        raise RuntimeError(f"DyCoKe LLaVA sample evidence failed: {checks!r}")
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
    if first_layer is None:
        raise RuntimeError("LLaVA structural handle llm.layer[0] is missing")
    audit = _common._DecoderInputAudit(first_layer)
    activation = generation_activation(
        facade=facade,
        runtime=runtime,
        binding_name="dycoke.video.llava_onevision",
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
                if latest_splice is None:
                    raise RuntimeError("LLaVA bridge published no splice evidence")
                checks = _validate_dycoke_sample(
                    baseline=baseline,
                    compressed=compressed,
                    activation_records=layout_records,
                    algorithm_records=algorithm_records,
                    decode_records=decode_records,
                    latest_splice=latest_splice,
                    labels_absent=labels_absent,
                    placeholder_count=placeholder_count,
                    frame_count=frame_count,
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
                    "decode_kv_view_count": len(decode_records),
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
            "algorithm": "dycoke",
            "variant": "dycoke.video",
            "backbone": "llava_onevision",
            "binding": "dycoke.video.llava_onevision",
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
            "physical_cache_compaction": False,
        },
    }
    return report, sample_rows


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    add_checkpoint_arguments(parser, estimated_peak_mib=32768)
    parser.add_argument("--max_new_tokens", type=int, default=4)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.max_new_tokens < 2:
        raise RuntimeError("max_new_tokens must be at least 2")
    return inference_main(args, _run)


if __name__ == "__main__":
    raise SystemExit(main())
