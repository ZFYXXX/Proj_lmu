"""Layer application boundaries, independent of a rule's scoring schedule."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class LayerPosition:
    """An inter-layer index counts completed layers; zero is initial input.

    At post_attention_pre_ffn, index identifies the current zero-based layer.
    Providers with inter-layer feature injection must preserve that ordering
    when choosing the physical hook for an inter-layer position.
    """

    boundary: str
    index: int

    def as_dict(self) -> dict[str, Any]:
        return {"boundary": self.boundary, "index": self.index}


def resolve_layer_positions(
    operation: Mapping[str, Any],
    default_layers: Sequence[int],
    *,
    boundary: str,
    num_layers: int | None = None,
) -> tuple[LayerPosition, ...]:
    """Keep legacy rule layers intact while resolving optional application layers.

    operation.layers overrides application layers only. For a full-layer site,
    layer_edge=before is an alias for the previous layer's output, including
    the initial input at layer zero. The default edge is after.
    """

    if "layer_edge" in operation and "layers" not in operation:
        raise ValueError("operation.layer_edge requires explicit operation.layers.")
    layers = operation.get("layers", default_layers)
    if (
        not isinstance(layers, (list, tuple))
        or not layers
        or any(isinstance(i, bool) or not isinstance(i, int) or i < 0 for i in layers)
        or list(layers) != sorted(set(layers))
    ):
        raise ValueError("operation.layers must be non-empty, increasing non-negative integers.")
    edge = operation.get("layer_edge", "after")
    if not isinstance(edge, str) or edge not in {"before", "after"}:
        raise ValueError("operation.layer_edge must be 'before' or 'after'.")
    if boundary == "post_attention_pre_ffn":
        if "layer_edge" in operation:
            raise ValueError("operation.layer_edge applies only to full-layer sites.")
        positions = tuple(LayerPosition(boundary, i) for i in layers)
    elif boundary == "post_layer":
        positions = tuple(LayerPosition("inter_layer", i + (edge == "after")) for i in layers)
    else:
        raise ValueError(f"{boundary!r} is not a layer compression boundary.")
    upper = num_layers + (boundary == "post_layer" and edge == "before") if num_layers is not None else None
    if upper is not None and any(i >= upper for i in layers):
        raise ValueError(f"operation.layers is outside the {num_layers}-layer stack.")
    return positions
