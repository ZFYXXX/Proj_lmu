"""Layer-wise Structural-Pruning-LVLM policy for OV-Qwen2.

The released ShortGPT variant measures block influence as
``1 - cosine(h_before, h_after)`` over calibration tokens, removes the
lowest-scoring decoder blocks, and recovery-finetunes the shortened model.
This port keeps the original Qwen2 layer indices in the checkpoint and skips
the selected blocks at runtime.  That is numerically equivalent to deleting
the blocks while preserving compatibility with the unpruned base checkpoint.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Iterable, Optional, Sequence


DEFAULT_CONFIG: Dict[str, Any] = {
    "method": "structural_pruning",
    "pruned_layers": [],
    "calibration_examples": 50,
    "calibration_metric": "block_influence_1_minus_cosine",
    "selection_rule": "lowest_individual_block_influence",
    "calibration_source": "vlm_instruction_data",
    "original_layer_count": 28,
    "target_flops_retain_ratio": 19.0 / 28.0,
    "image_only": False,
    "structural_decode_pruning": True,
}


def normalize_structural_pruning_config(
    raw: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    cfg = dict(DEFAULT_CONFIG)
    if raw:
        cfg.update(raw)
    cfg["method"] = "structural_pruning"
    cfg["pruned_layers"] = sorted(
        {int(item) for item in cfg.get("pruned_layers", [])}
    )
    cfg["calibration_examples"] = int(cfg["calibration_examples"])
    cfg["original_layer_count"] = int(cfg["original_layer_count"])
    cfg["target_flops_retain_ratio"] = float(
        cfg["target_flops_retain_ratio"]
    )
    cfg["image_only"] = bool(cfg["image_only"])
    cfg["structural_decode_pruning"] = bool(
        cfg["structural_decode_pruning"]
    )

    allowed = set(DEFAULT_CONFIG)
    unknown = sorted(set(cfg) - allowed)
    if unknown:
        raise ValueError(
            "Unknown Structural-Pruning-LVLM config keys: "
            + ", ".join(unknown)
        )
    if cfg["calibration_examples"] < 1:
        raise ValueError("Structural pruning needs at least one calibration example")
    if cfg["original_layer_count"] < 2:
        raise ValueError("Structural pruning needs at least two decoder layers")
    if not cfg["pruned_layers"]:
        raise ValueError("Structural pruning requires calibrated pruned_layers")
    if any(layer < 0 for layer in cfg["pruned_layers"]):
        raise ValueError("Structural pruning layer indices must be non-negative")
    if cfg["pruned_layers"][-1] >= cfg["original_layer_count"]:
        raise ValueError("Structural pruning layer index exceeds original depth")
    measured = (
        cfg["original_layer_count"] - len(cfg["pruned_layers"])
    ) / float(cfg["original_layer_count"])
    if not math.isclose(
        measured, cfg["target_flops_retain_ratio"], rel_tol=0.0, abs_tol=1e-12
    ):
        raise ValueError(
            "Structural pruning target FLOPs ratio must equal its retained "
            f"whole-layer fraction: measured={measured:.12f}"
        )
    if cfg["calibration_metric"] != "block_influence_1_minus_cosine":
        raise ValueError("Unsupported Structural-Pruning-LVLM calibration metric")
    if cfg["selection_rule"] != "lowest_individual_block_influence":
        raise ValueError("Unsupported Structural-Pruning-LVLM selection rule")
    if not cfg["structural_decode_pruning"]:
        raise ValueError(
            "A physically pruned model must skip the same blocks during decode"
        )
    return cfg


def block_influence_sum(before: Any, after: Any) -> Any:
    """Return the official per-sample sum of ``1 - cosine`` over tokens."""

    import torch
    import torch.nn.functional as F

    if not isinstance(before, torch.Tensor) or not isinstance(after, torch.Tensor):
        raise TypeError("Structural block influence expects tensors")
    if before.shape != after.shape or before.ndim != 3:
        raise ValueError("Structural block influence expects matching [B,S,D] states")
    similarity = F.cosine_similarity(before.float(), after.float(), dim=-1)
    similarity = torch.nan_to_num(similarity, nan=0.5)
    return (1.0 - similarity).sum(dim=-1)


def select_structural_pruning_layers(
    layer_scores: Sequence[float], *, prune_count: int
) -> list[int]:
    """Select the individually least influential layers, as released."""

    scores = [float(item) for item in layer_scores]
    count = int(prune_count)
    if count < 1 or count >= len(scores):
        raise ValueError("Structural prune_count must be in [1, layer_count)")
    if any(not math.isfinite(item) for item in scores):
        raise ValueError("Structural calibration scores must be finite")
    return sorted(range(len(scores)), key=lambda idx: (scores[idx], idx))[:count]


def retained_layer_fraction(
    layer_count: int, pruned_layers: Iterable[int]
) -> float:
    pruned = {int(item) for item in pruned_layers}
    if layer_count < 1 or any(not 0 <= item < layer_count for item in pruned):
        raise ValueError("Invalid structural-pruning layer set")
    return (layer_count - len(pruned)) / float(layer_count)
