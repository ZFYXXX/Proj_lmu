"""Backbone-neutral DivPrune kernel for one image-feature sequence.

The released DivPrune algorithm chooses an isolated first centre and then uses
greedy farthest-first cosine selection.  This module implements that shared
math once.  Backbone bindings own image/media splitting, structural-token
protection, prompt legalization, and provider lifecycle.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Iterable, Mapping

from mllm_comp.core.layout import LayoutDelta, LayoutTokenUnit
from ._rules import rule_metadata, score_rule, selection_rule


@dataclass(frozen=True)
class DivPruneSpec:
    """Pinned layer-0 defaults and cardinality controls for DivPrune."""

    retain_ratio: float = 0.098
    target_count: int | None = None
    prune_layer: int = 0
    distance: str = "cosine"
    selection: str = "official_greedy_maxmin"
    min_visual_tokens: int = 1
    distance_chunk_size: int = 512

    def __post_init__(self) -> None:
        if isinstance(self.retain_ratio, bool) or not isinstance(
            self.retain_ratio, (int, float)
        ):
            raise TypeError("retain_ratio must be a real number.")
        if not 0.0 < float(self.retain_ratio) <= 1.0:
            raise ValueError("retain_ratio must be in (0, 1].")
        for name in (
            "prune_layer",
            "min_visual_tokens",
            "distance_chunk_size",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be an integer.")
        if self.target_count is not None and (
            isinstance(self.target_count, bool)
            or not isinstance(self.target_count, int)
        ):
            raise TypeError("target_count must be an integer or None.")
        if self.prune_layer != 0:
            raise ValueError("DivPrune currently supports only prune_layer=0.")
        if self.distance != "cosine":
            raise ValueError("DivPrune requires distance='cosine'.")
        if self.selection != "official_greedy_maxmin":
            raise ValueError(
                "DivPrune requires selection='official_greedy_maxmin'."
            )
        if self.min_visual_tokens < 1:
            raise ValueError("min_visual_tokens must be at least one.")
        if self.distance_chunk_size < 1:
            raise ValueError("distance_chunk_size must be at least one.")
        if self.target_count is not None and self.target_count < self.min_visual_tokens:
            raise ValueError("target_count must be at least min_visual_tokens.")

    def resolve_target_count(self, source_count: int) -> int:
        """Resolve the exact non-empty output count for one source sequence."""

        _require_positive_integer("source_count", source_count)
        requested = (
            self.target_count
            if self.target_count is not None
            else int(round(float(self.retain_ratio) * source_count))
        )
        return min(source_count, max(self.min_visual_tokens, requested))


@dataclass(frozen=True)
class DivPruneOutput:
    """Source-ordered selected features and their complete drop mapping."""

    tokens: Any
    indices: Any
    selection_order: Any
    layout_delta: LayoutDelta
    metadata: Mapping[str, Any]

    @property
    def selected_indices(self) -> Any:
        return self.indices

    @property
    def delta(self) -> LayoutDelta:
        return self.layout_delta


def select_divprune_indices(
    tokens: Any,
    target_count: int,
    *,
    distance_chunk_size: int = 512,
    protected_indices: Iterable[int] = (),
    rules: Mapping[str, Any] | None = None,
) -> Any:
    """Return greedy max-min indices in algorithm selection order.

    With no protected indices this matches the pinned fuxian implementation:
    the first centre maximizes nearest-neighbour cosine distance, and every
    later centre maximizes its minimum distance to the selected set.  Structural
    indices supplied by a binding seed the selected set before farthest-first
    selection and still count toward ``target_count``.
    """

    torch = _torch()
    _validate_tokens(tokens, torch=torch)
    _require_positive_integer("target_count", target_count)
    _require_positive_integer("distance_chunk_size", distance_chunk_size)
    source_count = int(tokens.shape[0])
    if target_count > source_count:
        raise ValueError("target_count cannot exceed the source token count.")
    protected = _normalise_indices(
        protected_indices,
        source_count=source_count,
        name="protected_indices",
    )
    if len(protected) > target_count:
        raise ValueError("target_count cannot retain all protected indices.")

    work = tokens.detach()
    if not bool(torch.isfinite(work).all().item()):
        raise ValueError("tokens must contain only finite values.")
    eps = max(1e-12, float(torch.finfo(work.dtype).tiny))
    normalized = torch.nn.functional.normalize(work, dim=-1, eps=eps)
    request = {"tokens": tokens, "normalized_features": normalized,
               "keep_count": target_count, "protected_indices": protected}
    custom_distances = None
    if rules and "scorer" in rules:
        # Match the replaced cosine matmul's precision without evaluating it.
        distance_dtype = normalized.dtype
        if distance_dtype != torch.float64 and torch.is_autocast_enabled(normalized.device.type):
            distance_dtype = torch.get_autocast_dtype(normalized.device.type)
        custom_distances = score_rule(rules, "scorer", request,
            normalized, expected_shape=(source_count, source_count), dtype=distance_dtype)
    custom = selection_rule(rules, "selector", {**request, "scores": custom_distances}, count=source_count,
        keep_count=target_count, device=tokens.device, protected=protected)
    if custom is not None:
        return custom

    selected = list(protected)
    chosen = torch.zeros(source_count, dtype=torch.bool, device=tokens.device)
    if selected:
        protected_tensor = torch.tensor(
            selected,
            dtype=torch.long,
            device=tokens.device,
        )
        chosen[protected_tensor] = True
        if custom_distances is None:
            selected_vectors = normalized.index_select(0, protected_tensor)
            distances = 1.0 - torch.matmul(normalized, selected_vectors.transpose(0, 1))
        else:
            distances = custom_distances.index_select(1, protected_tensor)
        min_distance = distances.min(dim=1).values
        min_distance[chosen] = float("-inf")
    else:
        if custom_distances is None:
            first_scores = _second_nearest_neighbor_distance(
                normalized,
                chunk_size=distance_chunk_size,
                torch=torch,
            )
        else:
            first_scores = custom_distances.topk(min(2, source_count), dim=1, largest=False).values[:, -1]
        first = int(torch.argmax(first_scores).item())
        selected.append(first)
        chosen[first] = True
        if custom_distances is None:
            min_distance = 1.0 - torch.mv(normalized, normalized[first])
        else:
            min_distance = custom_distances[:, first].clone()
        min_distance[chosen] = float("-inf")

    while len(selected) < target_count:
        next_index = int(torch.argmax(min_distance).item())
        selected.append(next_index)
        chosen[next_index] = True
        if custom_distances is None:
            distance_to_next = 1.0 - torch.mv(normalized, normalized[next_index])
        else:
            distance_to_next = custom_distances[:, next_index]
        min_distance = torch.minimum(min_distance, distance_to_next)
        min_distance[chosen] = float("-inf")

    return torch.tensor(selected, dtype=torch.long, device=tokens.device)


def divprune_image(
    tokens: Any,
    *,
    spec: DivPruneSpec | None = None,
    protected_indices: Iterable[int] = (),
    coordinates: Iterable[Any] = (),
    media_unit: str | LayoutTokenUnit = "image_feature",
    source_layout: Mapping[str, Any] | None = None,
    result_layout: Mapping[str, Any] | None = None,
    rules: Mapping[str, Any] | None = None,
) -> DivPruneOutput:
    """Select one image sequence and restore selected features to source order."""

    torch = _torch()
    _validate_tokens(tokens, torch=torch)
    active_spec = spec or DivPruneSpec()
    source_count = int(tokens.shape[0])
    target_count = active_spec.resolve_target_count(source_count)
    protected = _normalise_indices(
        protected_indices,
        source_count=source_count,
        name="protected_indices",
    )
    if len(protected) > target_count:
        raise ValueError(
            "DivPrune target count is smaller than the protected structural set."
        )

    if target_count == source_count:
        selection_order = torch.arange(source_count, device=tokens.device)
        selected = selection_order
        output_tokens = tokens
    else:
        selection_order = select_divprune_indices(
            tokens,
            target_count,
            distance_chunk_size=active_spec.distance_chunk_size,
            protected_indices=protected,
            rules=rules,
        )
        selected = selection_order.sort().values
        output_tokens = tokens.index_select(0, selected)

    selected_tuple = tuple(int(index) for index in selected.detach().cpu().tolist())
    selected_set = set(selected_tuple)
    dropped = tuple(index for index in range(source_count) if index not in selected_set)
    coordinate_tuple = tuple(coordinates)
    delta = LayoutDelta.drop(
        source_count,
        dropped,
        media_unit=media_unit,
        coordinates=coordinate_tuple,
        protected_indices=protected,
        source_layout=source_layout,
        result_layout=result_layout,
    )
    selection_tuple = tuple(
        int(index) for index in selection_order.detach().cpu().tolist()
    )
    metadata = MappingProxyType(
        {
            "algorithm": "divprune",
            "variant": "divprune.image",
            "sequence_contract": "single_image_sequence",
            "distance": "custom" if rules and "scorer" in rules else active_spec.distance,
            "selection": "custom" if rules and "selector" in rules else active_spec.selection,
            **rule_metadata(rules),
            "prune_layer": active_spec.prune_layer,
            "retain_ratio": float(active_spec.retain_ratio),
            "requested_target_count": active_spec.target_count,
            "effective_target_count": target_count,
            "min_visual_tokens": active_spec.min_visual_tokens,
            "distance_chunk_size": active_spec.distance_chunk_size,
            "source_count": source_count,
            "result_count": target_count,
            "selected_indices": selected_tuple,
            "selection_order": selection_tuple,
            "protected_indices": protected,
            "token_order": "source_index",
            "compression_applied": target_count < source_count,
        }
    )
    return DivPruneOutput(
        tokens=output_tokens,
        indices=selected,
        selection_order=selection_order,
        layout_delta=delta,
        metadata=metadata,
    )


def _second_nearest_neighbor_distance(
    normalized: Any,
    *,
    chunk_size: int,
    torch: Any,
) -> Any:
    source_count = int(normalized.shape[0])
    if source_count == 1:
        return normalized.new_ones(1)
    result = normalized.new_empty(source_count)
    keys = normalized.transpose(0, 1)
    for start in range(0, source_count, chunk_size):
        stop = min(start + chunk_size, source_count)
        distances = 1.0 - torch.matmul(normalized[start:stop], keys)
        result[start:stop] = distances.topk(2, dim=1, largest=False).values[:, 1]
    return result


def _validate_tokens(tokens: Any, *, torch: Any) -> None:
    if not isinstance(tokens, torch.Tensor):
        raise TypeError("tokens must be a torch.Tensor.")
    if tokens.ndim != 2:
        raise ValueError(
            "DivPrune uses a single-sequence contract; tokens must have shape "
            "[tokens, hidden]."
        )
    if not tokens.is_floating_point():
        raise TypeError("tokens must use a floating dtype.")
    if int(tokens.shape[0]) < 1 or int(tokens.shape[1]) < 1:
        raise ValueError("DivPrune requires a non-empty token and hidden dimension.")


def _normalise_indices(
    values: Iterable[int],
    *,
    source_count: int,
    name: str,
) -> tuple[int, ...]:
    result: list[int] = []
    for value in values:
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(f"{name} must contain integers.")
        result.append(value)
    normalized = tuple(sorted(set(result)))
    if any(index < 0 or index >= source_count for index in normalized):
        raise ValueError(f"{name} contains an index outside the source sequence.")
    return normalized


def _require_positive_integer(name: str, value: Any) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer.")
    if value < 1:
        raise ValueError(f"{name} must be at least one.")


def _torch() -> Any:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - project runtime supplies torch
        raise RuntimeError("DivPrune execution requires PyTorch.") from exc
    return torch


__all__ = [
    "DivPruneOutput",
    "DivPruneSpec",
    "divprune_image",
    "select_divprune_indices",
]
