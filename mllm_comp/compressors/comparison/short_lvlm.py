"""Short-LVLM layer-pruning primitives for OneVision/Qwen2.

The authors' public repository currently contains the paper README and vanilla
model trees but no released TIS/SCP implementation.  This module therefore
implements the equations explicitly and records every otherwise ambiguous
choice in the checkpoint-specific calibration asset.  It never copies a
LLaVA-1.5 layer list into the 28-layer OV/Qwen2 decoder.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Iterable, Optional, Sequence

import torch
import torch.nn.functional as F


SHORT_LVLM_DEFAULTS: Dict[str, Any] = {
    "method": "short_lvlm",
    "pruned_layers": [],
    "projection_pairs": [],
    "candidate_layers": [],
    "target_flops_ratio": 0.67,
    # Whole-layer pruning is discrete.  With 28 equal-cost Qwen2 blocks the
    # nearest paper-legal point is 19/28=0.67857, so 0.5% is unattainable.
    "target_flops_tolerance": 0.01,
    "calibrated_flops_ratio": None,
    "calibrated_absolute_target_error": None,
    "flops_aggregation": "sum_method_flops_div_sum_baseline_flops",
    "flops_reference_mode": "fixed_decoder_entry_geometry",
    "flops_reference_visual_tokens": 3215,
    "flops_reference_nonvisual_tokens": 37,
    "flops_reference_decoder_layers": 28,
    "calibration_dataset": "llava_665k",
    "calibration_samples": 1000,
    "calibration_seed": 0,
    "calibration_asset_path": "",
    "calibration_asset_sha256": "",
    "token_importance_top_fraction": 0.10,
    "subspace_rank": 64,
    "candidate_scope": "last_half",
    "tis_attention_interpretation": (
        "intra_received_columns_visual_cross_columns_text_cross_rows"
    ),
    "scp_feature_pooling": "mean_all_valid_vl_tokens_per_boundary",
    "scp_projection_targets": "qkv_gate_up_input_weights",
    "layer_pairing": "ascending_pruned_nearest_unique_retained_lower_tie",
    "structural_decode_pruning": True,
    "image_only": True,
    "log_first_n": 1,
}


def normalize_short_lvlm_config(raw: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    cfg = dict(SHORT_LVLM_DEFAULTS)
    if raw:
        unknown = sorted(set(raw) - set(cfg))
        if unknown:
            raise ValueError(
                f"Unknown Short-LVLM config keys: {', '.join(unknown)}"
            )
        cfg.update(raw)
    cfg["method"] = "short_lvlm"
    cfg["pruned_layers"] = sorted({int(item) for item in cfg["pruned_layers"]})
    cfg["candidate_layers"] = sorted(
        {int(item) for item in cfg["candidate_layers"]}
    )
    pairs = []
    for raw_pair in cfg["projection_pairs"]:
        if not isinstance(raw_pair, dict) or set(raw_pair) != {
            "pruned_layer",
            "retained_layer",
        }:
            raise ValueError(
                "Each Short-LVLM projection pair must contain only "
                "pruned_layer and retained_layer"
            )
        pairs.append(
            {
                "pruned_layer": int(raw_pair["pruned_layer"]),
                "retained_layer": int(raw_pair["retained_layer"]),
            }
        )
    cfg["projection_pairs"] = pairs
    for key in (
        "target_flops_ratio",
        "target_flops_tolerance",
        "token_importance_top_fraction",
    ):
        cfg[key] = float(cfg[key])
    for key in (
        "calibrated_flops_ratio",
        "calibrated_absolute_target_error",
    ):
        cfg[key] = None if cfg[key] is None else float(cfg[key])
    for key in (
        "calibration_samples",
        "calibration_seed",
        "subspace_rank",
        "log_first_n",
        "flops_reference_visual_tokens",
        "flops_reference_nonvisual_tokens",
        "flops_reference_decoder_layers",
    ):
        cfg[key] = int(cfg[key])
    for key in (
        "calibration_dataset",
        "calibration_asset_path",
        "calibration_asset_sha256",
        "candidate_scope",
        "tis_attention_interpretation",
        "scp_feature_pooling",
        "scp_projection_targets",
        "layer_pairing",
        "flops_aggregation",
        "flops_reference_mode",
    ):
        cfg[key] = (
            str(cfg[key]).strip().lower()
            if key != "calibration_asset_path"
            else str(cfg[key]).strip()
        )
    for key in ("structural_decode_pruning", "image_only"):
        cfg[key] = bool(cfg[key])

    if not cfg["pruned_layers"]:
        raise ValueError("Short-LVLM requires a non-empty pruned_layers policy")
    if any(item < 0 for item in cfg["pruned_layers"] + cfg["candidate_layers"]):
        raise ValueError("Short-LVLM layer indices must be non-negative")
    if not set(cfg["pruned_layers"]).issubset(set(cfg["candidate_layers"])):
        raise ValueError("Short-LVLM pruned layers must lie in candidate_layers")
    if len(pairs) != len(cfg["pruned_layers"]):
        raise ValueError("Short-LVLM needs one SCP pair per pruned layer")
    if sorted(pair["pruned_layer"] for pair in pairs) != cfg["pruned_layers"]:
        raise ValueError("Short-LVLM SCP pairs do not cover every pruned layer")
    retained = [pair["retained_layer"] for pair in pairs]
    if len(set(retained)) != len(retained):
        raise ValueError("Short-LVLM retained SCP layers must be unique")
    if set(retained) & set(cfg["pruned_layers"]):
        raise ValueError("A Short-LVLM SCP target cannot itself be pruned")
    if not 0.0 < cfg["target_flops_ratio"] <= 1.0:
        raise ValueError("Short-LVLM target_flops_ratio must lie in (0, 1]")
    if not 0.0 <= cfg["target_flops_tolerance"] < 1.0:
        raise ValueError("Short-LVLM target_flops_tolerance is invalid")
    if cfg["calibrated_flops_ratio"] is None:
        raise ValueError("Short-LVLM requires a calibrated_flops_ratio")
    expected_error = abs(
        cfg["calibrated_flops_ratio"] - cfg["target_flops_ratio"]
    )
    if cfg["calibrated_absolute_target_error"] is None or not math.isclose(
        cfg["calibrated_absolute_target_error"], expected_error, abs_tol=1e-12
    ):
        raise ValueError("Short-LVLM calibrated target error is inconsistent")
    if expected_error > cfg["target_flops_tolerance"]:
        raise ValueError("Short-LVLM whole-layer policy misses its tolerance")
    if (
        cfg["flops_aggregation"] != "sum_method_flops_div_sum_baseline_flops"
        or cfg["flops_reference_mode"] != "fixed_decoder_entry_geometry"
        or cfg["flops_reference_visual_tokens"] < 1
        or cfg["flops_reference_nonvisual_tokens"] < 1
        or cfg["flops_reference_decoder_layers"] < 2
    ):
        raise ValueError(
            "Short-LVLM requires canonical aggregation and positive calibrated geometry"
        )
    if not 0.0 < cfg["token_importance_top_fraction"] <= 1.0:
        raise ValueError("Short-LVLM TIS fraction must lie in (0, 1]")
    if cfg["subspace_rank"] < 1:
        raise ValueError("Short-LVLM subspace_rank must be positive")
    if cfg["calibration_samples"] < 1 or not cfg["calibration_dataset"]:
        raise ValueError("Short-LVLM requires calibration provenance")
    if not cfg["calibration_asset_path"]:
        raise ValueError("Short-LVLM requires an SCP calibration asset")
    digest = cfg["calibration_asset_sha256"]
    if digest and (len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest)):
        raise ValueError("Short-LVLM asset SHA-256 is invalid")
    expected_literals = {
        "candidate_scope": "last_half",
        "tis_attention_interpretation": (
            "intra_received_columns_visual_cross_columns_text_cross_rows"
        ),
        "scp_feature_pooling": "mean_all_valid_vl_tokens_per_boundary",
        "scp_projection_targets": "qkv_gate_up_input_weights",
        "layer_pairing": (
            "ascending_pruned_nearest_unique_retained_lower_tie"
        ),
    }
    for key, expected in expected_literals.items():
        if cfg[key] != expected:
            raise ValueError(f"Short-LVLM requires {key}={expected!r}")
    if not cfg["structural_decode_pruning"]:
        raise ValueError("Paper-faithful Short-LVLM must prune decode layers too")
    if cfg["log_first_n"] < 0:
        raise ValueError("Short-LVLM log_first_n must be non-negative")
    return cfg


def nearest_whole_layer_budget(
    *, num_layers: int, target_flops_ratio: float, maximum_prune_fraction: float = 0.5
) -> Dict[str, Any]:
    layers = int(num_layers)
    target = float(target_flops_ratio)
    if layers < 2 or not 0.0 < target <= 1.0:
        raise ValueError("Short-LVLM layer budget inputs are invalid")
    maximum_pruned = min(layers - 1, int(math.floor(layers * maximum_prune_fraction)))
    candidates = []
    for pruned in range(maximum_pruned + 1):
        retained = layers - pruned
        ratio = retained / layers
        candidates.append(
            {
                "pruned_layer_count": pruned,
                "retained_layer_count": retained,
                "flops_ratio": ratio,
                "absolute_target_error": abs(ratio - target),
            }
        )
    return min(
        candidates,
        key=lambda row: (
            row["absolute_target_error"],
            row["flops_ratio"] > target,
            -row["retained_layer_count"],
        ),
    )


def select_pruned_layers(
    layer_scores: Sequence[float],
    *,
    candidate_layers: Iterable[int],
    prune_count: int,
) -> list[int]:
    candidates = sorted({int(item) for item in candidate_layers})
    count = int(prune_count)
    if count < 1 or count > len(candidates):
        raise ValueError("Short-LVLM prune_count is outside candidate scope")
    if any(not 0 <= layer < len(layer_scores) for layer in candidates):
        raise ValueError("Short-LVLM candidate layer is outside the score vector")
    if any(not math.isfinite(float(layer_scores[layer])) for layer in candidates):
        raise ValueError("Short-LVLM layer scores must be finite")
    selected = sorted(
        candidates,
        key=lambda layer: (-float(layer_scores[layer]), layer),
    )[:count]
    return sorted(selected)


def pair_pruned_with_nearest_unique_retained(
    *, pruned_layers: Iterable[int], num_layers: int
) -> list[dict[str, int]]:
    pruned = sorted({int(item) for item in pruned_layers})
    layers = int(num_layers)
    if not pruned or any(not 0 <= item < layers for item in pruned):
        raise ValueError("Short-LVLM pruned layer list is invalid")
    available = set(range(layers)) - set(pruned)
    if len(available) < len(pruned):
        raise ValueError("Short-LVLM lacks unique retained SCP partners")
    pairs = []
    for pruned_layer in pruned:
        retained_layer = min(
            available, key=lambda item: (abs(item - pruned_layer), item)
        )
        available.remove(retained_layer)
        pairs.append(
            {
                "pruned_layer": pruned_layer,
                "retained_layer": retained_layer,
            }
        )
    return pairs


@torch.no_grad()
def tis_layer_statistics(
    hidden_before: torch.Tensor,
    hidden_after: torch.Tensor,
    attention: torch.Tensor,
    *,
    visual_spans: Sequence[Sequence[int]],
    top_fraction: float,
) -> Dict[str, Any]:
    """Compute the paper's intra/cross-modal TIS and selected-token cosine.

    The unreleased implementation leaves the attention-axis convention
    unspecified.  The only causal interpretation that gives both modalities a
    cross signal is recorded in the config: visual importance uses columns of
    text-query -> visual-key attention, while text importance uses rows of that
    same matrix.  Intra-modal importance is received attention (columns).
    """

    if hidden_before.shape != hidden_after.shape or hidden_before.ndim != 3:
        raise ValueError("Short-LVLM TIS expects matching [1,L,D] states")
    if hidden_before.shape[0] != 1 or attention.ndim != 4 or attention.shape[0] != 1:
        raise ValueError("Short-LVLM TIS calibration requires batch_size=1")
    sequence_length = int(hidden_before.shape[1])
    if attention.shape[-2:] != (sequence_length, sequence_length):
        raise ValueError("Short-LVLM attention/state sequence lengths disagree")
    fraction = float(top_fraction)
    if not 0.0 < fraction <= 1.0:
        raise ValueError("Short-LVLM top_fraction must lie in (0,1]")

    visual_mask = torch.zeros(
        sequence_length, dtype=torch.bool, device=hidden_before.device
    )
    previous_end = 0
    for raw_span in visual_spans:
        start, end = map(int, raw_span)
        if not 0 <= start < end <= sequence_length or start < previous_end:
            raise ValueError(f"Invalid Short-LVLM visual span [{start}, {end})")
        visual_mask[start:end] = True
        previous_end = end
    visual_indices = visual_mask.nonzero(as_tuple=True)[0]
    text_indices = (~visual_mask).nonzero(as_tuple=True)[0]
    if visual_indices.numel() < 1 or text_indices.numel() < 1:
        raise ValueError("Short-LVLM TIS needs visual and nonvisual tokens")

    mean_attention = attention.float().mean(dim=1)[0]
    visual_self = mean_attention.index_select(0, visual_indices).index_select(
        1, visual_indices
    )
    text_self = mean_attention.index_select(0, text_indices).index_select(
        1, text_indices
    )
    text_to_visual = mean_attention.index_select(0, text_indices).index_select(
        1, visual_indices
    )
    visual_intra = visual_self.mean(dim=0)
    text_intra = text_self.mean(dim=0)
    visual_cross = text_to_visual.mean(dim=0)
    text_cross = text_to_visual.mean(dim=1)
    importance = torch.empty(
        sequence_length, dtype=torch.float32, device=hidden_before.device
    )
    importance[visual_indices] = 0.5 * (visual_intra + visual_cross)
    importance[text_indices] = 0.5 * (text_intra + text_cross)
    if not torch.isfinite(importance).all():
        raise RuntimeError("Short-LVLM TIS produced non-finite importance")
    selected_count = max(1, int(sequence_length * fraction))
    selected = importance.topk(
        min(sequence_length, selected_count), largest=True, sorted=True
    ).indices.sort().values
    before = hidden_before[0].index_select(0, selected).float()
    after = hidden_after[0].index_select(0, selected).float()
    token_cosines = F.cosine_similarity(before, after, dim=-1, eps=1e-8)
    layer_cosine = token_cosines.mean()
    if not torch.isfinite(layer_cosine):
        raise RuntimeError("Short-LVLM layer cosine is non-finite")
    return {
        "layer_cosine": float(layer_cosine.item()),
        "selected_count": int(selected.numel()),
        "visual_token_count": int(visual_indices.numel()),
        "text_token_count": int(text_indices.numel()),
        "selected_indices": selected,
        "importance": importance,
    }
