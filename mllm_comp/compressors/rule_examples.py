"""Small stateless rules used by the research examples, not standard recipes."""

from __future__ import annotations

import math
from typing import Any


def fastv_feature_norm(request: dict[str, Any], *, power: float = 2.0) -> Any:
    """Rank visual tokens by their feature norm at the application boundary."""
    if not math.isfinite(power) or power <= 0:
        raise ValueError("power must be finite and positive")
    values = request["hidden_states"]
    indices = request["image_indices"].to(device=values.device)
    return values.index_select(0, indices).detach().float().abs().pow(power).sum(-1)


def fastv_uniform_select(request: dict[str, Any]) -> Any:
    """Keep protected tokens, then sample the remaining visual positions evenly."""
    import torch

    protected = set(request["protected_image_positions"])
    count = int(request["image_indices"].numel())
    candidates = [index for index in range(count) if index not in protected]
    needed = request["keep_count"] - len(protected)
    chosen = sorted(protected)
    if needed:
        chosen.extend(candidates[index * len(candidates) // needed] for index in range(needed))
    return torch.tensor(chosen, dtype=torch.long, device=request["hidden_states"].device)


def ltbm_nearest_pairs(request: dict[str, Any]) -> list[tuple[int, int]]:
    """Merge the earliest sources into their nearest destination partition slot."""
    destinations = request["destination_indices"]
    return [(source, destinations[min(slot, len(destinations) - 1)])
            for slot, source in enumerate(request["source_indices"][:request["merge_count"]])]


def ltbm_destination_weighted_mean(request: dict[str, Any], *, destination_weight: float = 2.0) -> Any:
    """Weight the destination separately; every source retains unit weight."""
    if not math.isfinite(destination_weight) or destination_weight <= 0:
        raise ValueError("destination_weight must be finite and positive")
    values = request["values"]  # [destination, source_0, ...], differentiable
    return (values[0] * destination_weight + values[1:].sum(0)) / (destination_weight + values.shape[0] - 1)
