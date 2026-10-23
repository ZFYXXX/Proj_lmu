"""Validate canonical LLaMA-VID inference on LLaVA-OneVision.

The offline batch-one diagnostic compares the same four fixed NExT-QA samples
against their native baseline and freezes the local BERT/QFormer context path,
mean-content path, exact value lineage, LLaVA splice, decoder alignment, and
finite generation behavior.  No method-specific checkpoint is claimed.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

from mllm_comp.pipeline.stage import CompressionStageConfig
from scripts.diagnostics import llava_provider_video_checkpoint_common as _common


FIXED_NEXTQA_DOC_IDS = _common.FIXED_NEXTQA_DOC_IDS
CANONICAL_LLAMAVID_CONFIG = {
    "num_query": 32,
    "bert_type": "qformer",
    "compress_type": "mean",
}


def _stage() -> CompressionStageConfig:
    return CompressionStageConfig.from_dict(
        {
            "name": "llamavid-video-llava-onevision-real-checkpoint",
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
            "binding": "llamavid.video.llava_onevision",
            "config": {},
        }
    )


def _bert_token_shapes_exact(records: Sequence[Mapping[str, Any]]) -> bool:
    if len(records) != 1 or set(records[0]) != {"input_ids", "attention_mask"}:
        return False
    input_shape = tuple(int(value) for value in records[0]["input_ids"])
    mask_shape = tuple(int(value) for value in records[0]["attention_mask"])
    return (
        input_shape == mask_shape
        and len(input_shape) == 2
        and (input_shape[0] == 1 and 0 < input_shape[1] <= 256)
    )


def _value_lineage_exact(
    records: Sequence[Mapping[str, Any]],
    *,
    frame_count: int,
    tokens_per_frame: int,
    raw_tokens_per_frame: int,
) -> bool:
    if len(records) != frame_count * 2:
        return False
    projected_side = math.isqrt(tokens_per_frame)
    raw_side = math.isqrt(raw_tokens_per_frame)
    if (
        projected_side * projected_side != tokens_per_frame
        or raw_side * raw_side != raw_tokens_per_frame
    ):
        return False
    projected_mean_counts = [0] * tokens_per_frame
    for raw_h in range(raw_side):
        for raw_w in range(raw_side):
            target_h = min(raw_h * projected_side // raw_side, projected_side - 1)
            target_w = min(raw_w * projected_side // raw_side, projected_side - 1)
            projected_mean_counts[target_h * projected_side + target_w] += 1
    projected_mean_weights = tuple(
        count / raw_tokens_per_frame for count in projected_mean_counts
    )
    for index, record in enumerate(records):
        temporal = index // 2
        kind = "context" if index % 2 == 0 else "content"
        source_indices = tuple(int(value) for value in record.get("source_indices", ()))
        weights = tuple(float(value) for value in record.get("weights", ()))
        expected_indices = tuple(
            range(temporal * tokens_per_frame, (temporal + 1) * tokens_per_frame)
        )
        if (
            int(record.get("temporal_index", -1)) != temporal
            or record.get("kind") != kind
            or source_indices != expected_indices
            or len(weights) != tokens_per_frame
            or any(not math.isfinite(value) or value < 0.0 for value in weights)
            or not math.isclose(sum(weights), 1.0, rel_tol=0.0, abs_tol=1e-5)
        ):
            return False
        if kind == "content" and any(
            not math.isclose(
                value,
                expected,
                rel_tol=1e-2,
                abs_tol=1e-6,
            )
            for value, expected in zip(weights, projected_mean_weights)
        ):
            return False
    return True


def _validate_llamavid_sample(
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
    provider_metadata: Mapping[str, Any],
    activation_metadata: Mapping[str, Any],
    auxiliary_checkpoint: Path,
    **_unused: Any,
) -> dict[str, bool]:
    if len(activation_records) != 1 or len(algorithm_records) != 1:
        raise RuntimeError(
            "LLaMA-VID must publish one layout and one algorithm record per sample"
        )
    layout = activation_records[0]
    algorithm = algorithm_records[0]
    source_count = int(layout["source_count"])
    result_count = int(layout["result_count"])
    baseline_length = int(baseline["decoder"]["prefill"]["query_length"])
    compressed_length = int(compressed["decoder"]["prefill"]["query_length"])
    representatives_by_video = tuple(algorithm["representative_indices"])
    representatives = (
        ()
        if len(representatives_by_video) != 1
        else tuple(int(value) for value in representatives_by_video[0])
    )
    structural_source = tuple(latest_splice["structural_source_positions"])
    structural_result = tuple(latest_splice["structural_result_positions"])
    projected_tokens_per_frame = 196
    raw_tokens_per_frame = 729
    expected_source = frame_count * projected_tokens_per_frame
    expected_result = frame_count * 2
    parameter_count = int(activation_metadata.get("qformer_parameter_count", 0))
    trainable_count = int(
        activation_metadata.get("qformer_trainable_parameter_count", -1)
    )
    checks = {
        "compression_applied": bool(latest_splice["compression_applied"]),
        "non_identity_token_reduction": source_count > result_count > 0,
        "canonical_frame_count_exact": frame_count == 32,
        "native_projected_frame_raster_exact": source_count == expected_source,
        "canonical_two_tokens_per_frame_exact": result_count == expected_result,
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
        "provider_video_only": provider_metadata.get("video_only") is True,
        "provider_raw_stream_attached": provider_metadata.get("raw_stream_attached")
        is True,
        "provider_bert_loaded_and_qformer_bound": (
            provider_metadata.get("bert_tokenizer_loaded") is True
            and provider_metadata.get("qformer_bound") is True
        ),
        "provider_bert_path_exact": Path(str(provider_metadata.get("bert_path")))
        == auxiliary_checkpoint,
        "provider_text_policy_exact": (
            int(provider_metadata.get("text_max_length", 0)) == 256
            and provider_metadata.get("text_truncation_side") == "left"
        ),
        "qformer_registered_and_trainable": (
            activation_metadata.get("qformer_registered_pre_optimizer") is True
            and parameter_count > 0
            and trainable_count == parameter_count
        ),
        "activation_bert_path_exact": Path(str(activation_metadata.get("bert_path")))
        == auxiliary_checkpoint,
        "canonical_config_exact": dict(algorithm["canonical_config"])
        == CANONICAL_LLAMAVID_CONFIG,
        "single_video_counts_exact": (
            int(algorithm["video_count"]) == 1
            and tuple(algorithm["per_video_source_counts"]) == (source_count,)
            and tuple(algorithm["per_video_result_counts"]) == (result_count,)
            and tuple(algorithm["exact_budgets"]) == (result_count,)
        ),
        "projected_signal_shape_exact": tuple(algorithm["projected_features_shape"])[0]
        == expected_source,
        "raw_signal_shape_exact": tuple(algorithm["raw_features_shape"])[0]
        == frame_count * raw_tokens_per_frame,
        "result_signal_shape_exact": tuple(algorithm["result_features_shape"])[0]
        == expected_result,
        "all_runtime_signals_finite": all(
            algorithm.get(name) is True
            for name in (
                "projected_features_finite",
                "raw_features_finite",
                "result_features_finite",
            )
        ),
        "text_context_and_bert_tokens_exact": (
            algorithm.get("text_context_present") is True
            and _bert_token_shapes_exact(tuple(algorithm["bert_token_shapes"]))
        ),
        "representatives_source_ordered": representatives
        == tuple(sorted(set(representatives))),
        "baseline_logits_finite": bool(baseline["logits_finite"]),
        "compressed_logits_finite": bool(compressed["logits_finite"]),
        "baseline_decoder_alignment": all(baseline["decoder"]["invariants"].values()),
        "compressed_decoder_alignment": all(
            compressed["decoder"]["invariants"].values()
        ),
    }
    if not all(checks.values()):
        raise RuntimeError(f"LLaMA-VID LLaVA sample evidence failed: {checks!r}")
    return checks


def _profile() -> _common.LlavaProviderVideoDiagnosticProfile:
    return _common.LlavaProviderVideoDiagnosticProfile(
        algorithm="llamavid",
        variant="llamavid.video",
        binding="llamavid.video.llava_onevision",
        canonical_config=CANONICAL_LLAMAVID_CONFIG,
        auxiliary_argument="bert_checkpoint",
        auxiliary_runtime_key="llamavid_bert_path",
        auxiliary_report_key="bert_checkpoint",
        auxiliary_label="BERT Base Uncased",
        stage_factory=_stage,
        sample_validator=_validate_llamavid_sample,
        defining_stages=(
            "local_bert_qformer_cross_attention",
            "context_attention_and_mean_content_pool",
            "llava_projector_to_llm_splice",
        ),
        estimate_basis=(
            "32 GiB includes the 0.5B LLaVA checkpoint, local BERT/QFormer, "
            "32-frame raw and projected features, baseline plus compressed "
            "generation, and allocator variance"
        ),
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    _common.add_common_arguments(parser)
    parser.add_argument("--bert_checkpoint", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    return _common.main(_parser().parse_args(argv), profile=_profile())


if __name__ == "__main__":
    raise SystemExit(main())
