"""Offline fixed-four LLaMA-VID evidence on the native Qwen video layouts."""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Mapping

from mllm_comp.pipeline.stage import CompressionStageConfig
from scripts.diagnostics import qwen25_provider_video_checkpoint_common as common
from scripts.diagnostics import (
    validate_llava_llamavid_video_compression_checkpoint as llama,
)

CANONICAL_LLAMAVID_CONFIG = llama.CANONICAL_LLAMAVID_CONFIG
FIXED_NEXTQA_DOC_IDS = common.FIXED_NEXTQA_DOC_IDS


def stage(backbone: str) -> CompressionStageConfig:
    return CompressionStageConfig.from_dict(
        {
            "name": f"llamavid-video-{backbone}-real-checkpoint",
            "placement": "projector_level",
            "execution_phase": "generate",
            "plugin": "llamavid",
            "algorithm": "llamavid",
            "variant": "llamavid.video",
            "target_modalities": ["video"],
            "conditioning_modalities": ["text"],
            "preserved_modalities": ["text", "image", "audio"],
            "coupling_policy": "independent",
            "media_alignment": "spatiotemporal",
            "binding": f"llamavid.video.{backbone}",
            "config": {},
        }
    )


def value_lineage_exact(records, temporal: int, per_frame: int) -> bool:
    """Qwen's equal-size spatial merge gives uniform projected content weights."""
    if len(records) != 2 * temporal:
        return False
    for index, record in enumerate(records):
        frame = index // 2
        kind = "context" if index % 2 == 0 else "content"
        weights = tuple(float(v) for v in record["weights"])
        if (
            record["kind"] != kind
            or int(record["temporal_index"]) != frame
            or tuple(record["source_indices"])
            != tuple(range(frame * per_frame, (frame + 1) * per_frame))
            or len(weights) != per_frame
            or any(not math.isfinite(v) or v < 0 for v in weights)
            or not math.isclose(sum(weights), 1.0, abs_tol=1e-5)
        ):
            return False
        if kind == "content" and any(
            not math.isclose(v, 1 / per_frame, rel_tol=0.01, abs_tol=1e-6)
            for v in weights
        ):
            return False
    return True


def validate_sample(
    *,
    baseline,
    compressed,
    activation_records,
    algorithm_records,
    latest_splice,
    latest_generation,
    source_grid_thw,
    source_grid_token_count,
    labels_absent,
    provider_capture_delta,
    provider_metadata,
    activation_metadata,
    merge_size=None,
    spatial_merge_size=None,
    auxiliary_checkpoint=None,
    auxiliary_checkpoint_paths=None,
    expected_deepstack_branches=None,
    video_span_count=None,
    native_video_count=None,
    source_placeholder_spans=None,
    **unused,
) -> dict[str, bool]:
    if len(activation_records) != 1 or len(algorithm_records) != 1:
        raise RuntimeError("LLaMA-VID requires exactly one invocation per video")
    layout, record = activation_records[0], algorithm_records[0]
    splice, generation = latest_splice, latest_generation
    temporal, height, width = map(int, source_grid_thw)
    merge = int(merge_size if merge_size is not None else spatial_merge_size)
    per_frame = height * width // merge**2
    source, result = int(layout["source_count"]), int(layout["result_count"])
    anchors = tuple(int(v) for v in record["representative_indices"][0])
    expected_anchors = tuple(
        t * per_frame + offset for t in range(temporal) for offset in (0, 1)
    )
    source_mrope = tuple(map(tuple, splice["source_visual_mrope"]))
    result_mrope = tuple(map(tuple, splice["result_visual_mrope"]))
    base_length = int(baseline["decoder"]["prefill"]["query_length"])
    compressed_length = int(compressed["decoder"]["prefill"]["query_length"])
    bert = Path(auxiliary_checkpoint or auxiliary_checkpoint_paths["bert_checkpoint"])
    checks = {
        "compression_applied": bool(splice["compression_applied"]),
        "non_identity_token_reduction": source > result > 0,
        "source_grid_count_exact": source
        == source_grid_token_count
        == temporal * per_frame,
        "raw_raster_exact": int(record["raw_features_shape"][0])
        == temporal * height * width,
        "canonical_two_tokens_per_unit": result == temporal * 2,
        "canonical_config_exact": dict(record["canonical_config"])
        == CANONICAL_LLAMAVID_CONFIG,
        "source_ordered_anchors_exact": anchors
        == expected_anchors
        == tuple(splice["representative_indices"]),
        "structural_singletons_exact": tuple(map(tuple, record["source_groups"][0]))
        == tuple((a,) for a in anchors),
        "exact_overlapping_value_lineage": value_lineage_exact(
            record["value_records_by_video"][0], temporal, per_frame
        ),
        "placeholder_counts_exact": int(splice["source_placeholder_count"]) == source
        and int(splice["result_placeholder_count"]) == result,
        "layout_counts_exact": all(
            int(r["source_count"]) == source and int(r["result_count"]) == result
            for r in (record, splice["layout_delta"])
        ),
        "grid_preserved": bool(splice["source_video_grid_preserved"])
        and tuple(splice["source_video_grid_thw"]) == tuple(source_grid_thw),
        "sparse_grid_policy": splice["result_grid_policy"]
        == "irregular_sparse_source_anchors",
        "mrope_exact": len(source_mrope) == source
        and len(result_mrope) == result
        and all(len(row) == 3 for row in source_mrope)
        and result_mrope == tuple(source_mrope[a] for a in anchors),
        "sequence_reduction_exact": base_length - compressed_length == source - result,
        "generation_lengths_exact": int(generation["source_sequence_length"])
        == base_length
        and int(generation["result_sequence_length"]) == compressed_length
        and int(generation["removed_count"]) == source - result,
        "cached_decode_repaired": bool(generation["prefill_complete"])
        and int(generation["decode_repairs"]) >= 1
        and isinstance(generation.get("last_decode_record"), Mapping),
        "rope_delta_published": bool(splice["rope_deltas"]),
        "labels_absent": labels_absent,
        "provider_capture_once": provider_capture_delta == 1,
        "local_bert_bound": provider_metadata["bert_tokenizer_loaded"] is True
        and provider_metadata["qformer_bound"] is True
        and Path(provider_metadata["bert_path"]) == bert
        and Path(activation_metadata["bert_path"]) == bert,
        "prompt_conditioning_exact": record["text_context_present"] is True
        and record["text_policy"] == "user_text_without_teacher_forced_answer"
        and llama._bert_token_shapes_exact(record["bert_token_shapes"]),
        "bert_token_policy": provider_metadata["text_max_length"] == 256
        and provider_metadata["text_truncation_side"] == "left"
        and record["bert_local_only"] is True,
        "raw_provider_exact": record["raw_stream"]
        == "qwen_final_visual_block_pre_merger_raster"
        and provider_metadata["raw_stream_attached"] is True
        and provider_metadata["video_only"] is True,
        "qformer_registered_trainable": activation_metadata[
            "qformer_registered_pre_optimizer"
        ]
        is True
        and int(activation_metadata["qformer_parameter_count"]) > 0
        and activation_metadata["qformer_parameter_count"]
        == activation_metadata["qformer_trainable_parameter_count"],
        "compression_state_api": tuple(
            activation_metadata["compression_checkpoint_api"]
        )
        == ("compression_state_dict", "load_compression_state_dict")
        and activation_metadata["method_checkpoint_required"] is False,
        "finite_signals": all(
            record[name] is True
            for name in (
                "projected_features_finite",
                "raw_features_finite",
                "result_features_finite",
            )
        ),
        "finite_generation": baseline["logits_finite"] is True
        and compressed["logits_finite"] is True
        and 1 < int(compressed["generated_token_count"]) <= 4,
        "decoder_alignment": all(baseline["decoder"]["invariants"].values())
        and all(compressed["decoder"]["invariants"].values()),
    }
    if expected_deepstack_branches is None:
        checks["one_video_span"] = video_span_count == 1
        checks["native_timing"] = (
            math.isfinite(float(splice["second_per_grid_seconds"]))
            and float(splice["second_per_grid_seconds"]) > 0
        )
    else:
        checks.update(
            {
                "one_native_video": native_video_count == 1,
                "temporal_placeholder_spans": tuple(
                    map(tuple, splice["source_placeholder_spans"])
                )
                == tuple(map(tuple, source_placeholder_spans))
                and len(splice["result_placeholder_spans"]) == temporal
                and all(
                    end - start == 2
                    for start, end in splice["result_placeholder_spans"]
                ),
                "visual_masks_compacted": splice["visual_pos_masks_compacted"] is True,
                "deepstack_count": int(splice["deepstack_branch_count"])
                == int(record["deepstack_branch_count"])
                == expected_deepstack_branches
                == 3,
                "deepstack_layout_shared": splice["shared_main_deepstack_layout"]
                is True,
                "deepstack_shapes": tuple(map(tuple, record["deepstack_result_shapes"]))
                == tuple(map(tuple, splice["deepstack_result_shapes"]))
                and all(
                    shape[0] == result for shape in record["deepstack_result_shapes"]
                )
                and all(
                    shape[0] == source for shape in record["deepstack_source_shapes"]
                ),
            }
        )
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise RuntimeError(f"LLaMA-VID Qwen evidence failed: {failed}")
    return checks


def _profile():
    return common.Qwen25ProviderVideoDiagnosticProfile(
        algorithm="llamavid",
        variant="llamavid.video",
        binding="llamavid.video.qwen25_vl",
        canonical_config=CANONICAL_LLAMAVID_CONFIG,
        auxiliary_argument="bert_checkpoint",
        auxiliary_runtime_key="llamavid_bert_path",
        auxiliary_report_key="bert_checkpoint",
        auxiliary_label="BERT Base Uncased",
        stage_factory=lambda: stage("qwen25_vl"),
        sample_validator=validate_sample,
        defining_stages=(
            "local_bert_qformer_cross_attention",
            "context_and_mean_content_pool",
            "qwen_native_sparse_mrope",
        ),
        estimate_basis="32 GiB covers the 3B base, BERT/QFormer, native video and four-token generation; 16 GiB margin.",
    )


def _parser():
    parser = argparse.ArgumentParser()
    common.add_common_arguments(
        parser,
        video_min_pixels=4 * 28 * 28,
        video_max_pixels=128 * 28 * 28,
        estimated_peak_mib=32768,
    )
    parser.add_argument("--bert_checkpoint", type=Path, required=True)
    return parser


def main(argv=None):
    return common.main(_parser().parse_args(argv), profile=_profile())


if __name__ == "__main__":
    raise SystemExit(main())
