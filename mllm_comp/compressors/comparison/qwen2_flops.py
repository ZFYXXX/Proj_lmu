"""Shared Qwen2/OneVision prefill-FLOPs accounting helpers.

The formal OP comparison uses one user-specified decoder-entry geometry for
every method: 3,215 visual tokens, 37 non-visual tokens and 28 Qwen2 decoder
layers.  Calibration examples still determine rankings, masks and importance
scores, but their own AnyRes/text lengths must not change the theoretical FLOPs
budget.  Within the fixed geometry the retained ratio is::

    sum_layer(F_method(layer)) / sum_layer(F_baseline(layer))

``aggregate_samplewise_flops`` remains useful for paired diagnostic inputs and
uses the same ratio-of-sums rule, but generated formal configs additionally
pin the reference geometry below.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Iterable


FLOPS_AGGREGATION = "sum_method_flops_div_sum_baseline_flops"
FLOPS_REFERENCE_MODE = "fixed_decoder_entry_geometry"
REFERENCE_VISUAL_TOKENS = 3215
REFERENCE_NONVISUAL_TOKENS = 37
REFERENCE_DECODER_LAYERS = 28


def reference_geometry_metadata() -> Dict[str, Any]:
    """Return the immutable formal OP FLOPs geometry."""

    return {
        "flops_reference_mode": FLOPS_REFERENCE_MODE,
        "flops_reference_visual_tokens": REFERENCE_VISUAL_TOKENS,
        "flops_reference_nonvisual_tokens": REFERENCE_NONVISUAL_TOKENS,
        "flops_reference_decoder_layers": REFERENCE_DECODER_LAYERS,
    }


def fixed_reference_token_shapes(count: int = 1) -> list[list[int]]:
    """Return ``[nonvisual, visual]`` pairs for formal policy selectors."""

    if int(count) < 1:
        raise ValueError("Reference-shape count must be positive")
    return [
        [REFERENCE_NONVISUAL_TOKENS, REFERENCE_VISUAL_TOKENS]
        for _ in range(int(count))
    ]


def aggregate_samplewise_flops(
    method_flops: Iterable[float], baseline_flops: Iterable[float]
) -> Dict[str, Any]:
    """Aggregate paired sample FLOPs with the project's canonical formula.

    ``mean_flops_ratio`` is retained as a compatibility key for existing
    configs and validators, but its value is the dataset-level weighted ratio.
    ``unweighted_mean_sample_flops_ratio`` is diagnostic only and must not be
    used for policy selection.
    """

    selected = [float(value) for value in method_flops]
    baseline = [float(value) for value in baseline_flops]
    if not selected or len(selected) != len(baseline):
        raise ValueError("FLOP aggregation requires equally sized non-empty inputs")
    if any(not math.isfinite(value) or value < 0.0 for value in selected):
        raise ValueError("Method FLOPs must be finite and non-negative")
    if any(not math.isfinite(value) or value <= 0.0 for value in baseline):
        raise ValueError("Baseline FLOPs must be finite and positive")

    selected_sum = math.fsum(selected)
    baseline_sum = math.fsum(baseline)
    sample_ratios = [value / full for value, full in zip(selected, baseline)]
    dataset_ratio = selected_sum / baseline_sum
    return {
        # Backward-compatible name used by generated OP configs.
        "mean_flops_ratio": float(dataset_ratio),
        "dataset_flops_ratio": float(dataset_ratio),
        "unweighted_mean_sample_flops_ratio": float(
            math.fsum(sample_ratios) / len(sample_ratios)
        ),
        "minimum_flops_ratio": float(min(sample_ratios)),
        "maximum_flops_ratio": float(max(sample_ratios)),
        "method_flops_sum": float(selected_sum),
        "baseline_flops_sum": float(baseline_sum),
        "sample_count": len(selected),
        "flops_aggregation": FLOPS_AGGREGATION,
    }
