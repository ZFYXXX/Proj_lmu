"""Backbone-neutral PruMerge/PruMerge+ image-token kernel.

The released method chooses adaptive attention outliers, folds their nearest
key-space neighbours into the chosen centres without normalising the
attention weights, and emits one residual global token.  Backbones own the
attention/key capture and structural-token policy.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Iterable, Mapping

from mllm_comp.core.layout import LayoutDelta, LayoutTokenUnit, OrderingPolicy
from ._rules import aggregate_rule, label_rule, rule_metadata, score_rule, selection_rule


@dataclass(frozen=True)
class PruMergeSpec:
    variant: str = "plus"
    adaptive: bool = True
    reduction_ratio: float = 0.125
    target_count: int | None = None
    iqr_multiplier: float = 1.5
    min_centers: int = 1
    cluster_size: int = 32

    def __post_init__(self) -> None:
        if self.variant not in {"base", "plus"}:
            raise ValueError("variant must be 'base' or 'plus'")
        if not 0.0 < self.reduction_ratio < 1.0:
            raise ValueError("reduction_ratio must be in (0, 1)")
        if self.target_count is not None and self.target_count < 2:
            raise ValueError("target_count must be at least two")
        if self.iqr_multiplier <= 0.0:
            raise ValueError("iqr_multiplier must be positive")
        if self.min_centers < 1 or self.cluster_size < 1:
            raise ValueError("min_centers and cluster_size must be positive")


@dataclass(frozen=True)
class PruMergeOutput:
    tokens: Any
    indices: Any
    layout_delta: LayoutDelta
    metadata: Mapping[str, Any]
    aggregation_plan: tuple[Any, ...] = ()

    @property
    def kept_indices(self) -> Any:
        return self.indices


def prumerge_image_tokens(
    tokens: Any,
    attention: Any,
    keys: Any,
    *,
    spec: PruMergeSpec = PruMergeSpec(),
    protected_indices: Iterable[int] = (),
    coordinates: Iterable[Any] = (),
    media_unit: str | LayoutTokenUnit = LayoutTokenUnit.IMAGE_PATCH,
    source_layout: Mapping[str, Any] | None = None,
    result_layout: Mapping[str, Any] | None = None,
    rules: Mapping[str, Any] | None = None,
) -> PruMergeOutput:
    """Apply released PruMerge selection and value equations to one image."""

    import torch
    import torch.nn.functional as F

    if not isinstance(tokens, torch.Tensor) or tokens.ndim != 2:
        raise ValueError("tokens must have shape [tokens, hidden]")
    if not tokens.is_floating_point() or int(tokens.shape[0]) < 2:
        raise ValueError("PruMerge needs at least two floating image tokens")
    source_count = int(tokens.shape[0])
    protected = _indices(protected_indices, source_count, "protected_indices")
    candidates = tuple(index for index in range(source_count) if index not in protected)
    if len(candidates) < 2:
        raise ValueError("PruMerge needs at least two non-structural image tokens")

    importance = _importance(attention, source_count, torch).to(device=tokens.device)
    metric = _keys(keys, source_count, torch).to(device=tokens.device)
    request = {"tokens": tokens, "attention": attention, "keys": keys,
        "candidate_indices": candidates, "protected_indices": protected, "spec": spec}
    importance = score_rule(rules, "scorer", request, importance)
    candidate_tensor = torch.tensor(candidates, dtype=torch.long, device=tokens.device)
    candidate_importance = importance.index_select(0, candidate_tensor)
    center_count, adaptive_ratio = _center_count(
        candidate_importance,
        candidate_count=len(candidates),
        protected_count=len(protected),
        spec=spec,
        torch=torch,
    )
    ranking = torch.argsort(candidate_importance, descending=True, stable=True)
    attention_local = ranking[:center_count]
    selected_local = attention_local
    if spec.variant == "plus":
        selected_local = _plus_centers(
            selected_local,
            candidate_count=len(candidates),
            ratio=adaptive_ratio,
            target_count=(
                None
                if spec.target_count is None
                else max(1, spec.target_count - len(protected) - 1)
            ),
            torch=torch,
        )
    # A residual global token requires at least one unselected candidate.
    if int(selected_local.numel()) >= len(candidates):
        selected_local = selected_local[: len(candidates) - 1]
    centers = candidate_tensor.index_select(0, selected_local)
    custom = selection_rule(rules, "selector",
        {**request, "scores": importance, "keep_count": int(centers.numel())},
        count=source_count, keep_count=int(centers.numel()), device=tokens.device)
    if custom is not None:
        if not set(custom.tolist()).issubset(candidates):
            raise ValueError("PruMerge selector must return eligible center indices.")
        centers = custom

    normalised = F.normalize(metric.float(), dim=-1, eps=1e-6)
    centre_metric = normalised.index_select(0, centers)
    similarity = centre_metric @ normalised.index_select(0, candidate_tensor).T
    center_local_lookup = {value: index for index, value in enumerate(candidates)}
    for row, source_index in enumerate(centers.detach().cpu().tolist()):
        similarity[row, center_local_lookup[int(source_index)]] = float("-inf")
    neighbour_count = min(spec.cluster_size, len(candidates) - 1)
    neighbour_local = similarity.topk(neighbour_count, dim=-1).indices
    neighbour_indices = candidate_tensor.index_select(0, neighbour_local.reshape(-1)).view(
        int(centers.numel()), neighbour_count
    )
    neighbour_indices = label_rule(rules, "grouper",
        {**request, "center_indices": centers, "similarities": similarity,
         "neighbour_count": neighbour_count}, neighbour_indices)
    if rules and "grouper" in rules:
        for center, row in zip(centers.tolist(), neighbour_indices.tolist()):
            if len(set(row)) != neighbour_count or not set(row).issubset(set(candidates) - {center}):
                raise ValueError("PruMerge grouper must return unique eligible non-center neighbours per row.")
    centre_values = tokens.index_select(0, centers)
    neighbour_values = tokens.index_select(0, neighbour_indices.reshape(-1)).view(
        int(centers.numel()), neighbour_count, int(tokens.shape[1])
    )
    neighbour_weights = importance.index_select(
        0, neighbour_indices.reshape(-1)
    ).view(int(centers.numel()), neighbour_count, 1)
    merged_centres = centre_values + (
        neighbour_values.float() * neighbour_weights.float()
    ).sum(dim=1).to(centre_values.dtype)

    center_set = set(int(value) for value in centers.detach().cpu().tolist())
    residual = tuple(index for index in candidates if index not in center_set)
    residual_tensor = torch.tensor(residual, dtype=torch.long, device=tokens.device)
    global_value = (
        tokens.index_select(0, residual_tensor).float()
        * importance.index_select(0, residual_tensor).float().unsqueeze(-1)
    ).sum(dim=0).to(tokens.dtype)

    # Serialize by source representative so placeholder/M-RoPE repair remains
    # deterministic.  The released centre values and final residual-global
    # value are unchanged by this ordering legalization.
    records: list[tuple[int, Any, tuple[int, ...], str]] = []
    for row, source_index in enumerate(centers.detach().cpu().tolist()):
        records.append((int(source_index), merged_centres[row], (int(source_index),), "center"))
    for index in protected:
        records.append((index, tokens[index], (index,), "protected"))
    records.append((residual[0], global_value, residual, "global"))
    records.sort(key=lambda item: item[0])
    output = torch.stack([item[1] for item in records])
    mapping: dict[int, int | None] = {index: None for index in range(source_count)}
    representatives: list[int] = []
    for result_index, (representative, _value, sources, _kind) in enumerate(records):
        representatives.append(representative)
        for source_index in sources:
            mapping[source_index] = result_index
    delta = LayoutDelta.from_mapping(
        source_count=source_count,
        old_to_new=mapping,
        result_count=len(records),
        media_unit=media_unit,
        coordinates=coordinates,
        protected_indices=protected,
        source_layout=source_layout,
        result_layout={
            **dict(result_layout or {}),
            "representative_indices": tuple(representatives),
        },
        ordering_policy=OrderingPolicy.PRESERVE_SOURCE,
    )
    metadata = MappingProxyType(
        {
            "algorithm": "prumerge",
            **rule_metadata(rules),
            "variant": f"prumerge.{spec.variant}",
            "source_count": source_count,
            "result_count": len(records),
            "center_count": int(centers.numel()),
            "global_count": 1,
            "protected_count": len(protected),
            "adaptive": spec.adaptive and spec.target_count is None,
            "adaptive_ratio": adaptive_ratio,
            "cluster_size": neighbour_count,
            "importance_source": "custom" if rules and "scorer" in rules else "mean_received_attention",
            "merge_mode": "custom" if rules and "aggregator" in rules else "official_unnormalized_attention_sum",
            "uniform_policy": "custom" if rules and "selector" in rules else "official_flat" if spec.variant == "plus" else "none",
            "center_indices": tuple(int(value) for value in centers.detach().cpu().tolist()),
            "neighbour_indices": tuple(
                tuple(int(value) for value in row)
                for row in neighbour_indices.detach().cpu().tolist()
            ),
            "token_order": "source_representative",
        }
    )
    plans = ()
    if rules:
        center_rows = {int(center): row for row, center in enumerate(centers.tolist())}
        plans = tuple((kind,
            (representative, *neighbour_indices[center_rows[representative]].tolist()) if kind == "center" else sources,
            neighbour_weights[center_rows[representative], :, 0].detach() if kind == "center"
            else importance.index_select(0, residual_tensor).detach() if kind == "global" else None)
            for representative, _, sources, kind in records)
    result = PruMergeOutput(output, torch.tensor(representatives, dtype=torch.long, device=tokens.device), delta, metadata, plans)
    if rules and "aggregator" in rules:
        from dataclasses import replace
        result = replace(result, tokens=replay_prumerge_values(tokens, result, rules=rules))
    return result


def replay_prumerge_values(tokens: Any, result: PruMergeOutput, *, rules: Mapping[str, Any] | None = None) -> Any:
    """Replay saved overlapping neighborhoods/weights, without rescoring branches."""
    import torch
    if not result.aggregation_plan:
        raise ValueError("PruMerge value replay requires a saved rule decision plan.")
    output = []
    for kind, sources, weights in result.aggregation_plan:
        indices = torch.tensor(sources, dtype=torch.long, device=tokens.device)
        values = tokens.index_select(0, indices)
        if kind == "protected":
            output.append(values[0])
            continue
        weights = weights.to(device=tokens.device)
        if kind == "center":
            default = values[0] + (values[1:].float() * weights.float()[:, None]).sum(0).to(tokens.dtype)
        else:
            default = (values.float() * weights.float()[:, None]).sum(0).to(tokens.dtype)
        output.append(aggregate_rule(rules, "aggregator",
            {"values": values, "source_indices": sources, "weights": weights, "kind": kind}, default))
    return torch.stack(output)


def _center_count(importance: Any, *, candidate_count: int, protected_count: int, spec: PruMergeSpec, torch: Any) -> tuple[int, float]:
    if spec.target_count is not None:
        count = max(1, min(candidate_count - 1, spec.target_count - protected_count - 1))
        return count, float(count) / float(candidate_count)
    if spec.adaptive:
        values = importance.detach().float()
        q1 = torch.quantile(values, 0.25)
        q3 = torch.quantile(values, 0.75)
        ratio = float((values > q3 + spec.iqr_multiplier * (q3 - q1)).float().mean().item())
    else:
        ratio = spec.reduction_ratio
    count = min(candidate_count - 1, max(spec.min_centers, int(candidate_count * ratio)))
    return count, max(ratio, 1.0 / float(candidate_count))


def _plus_centers(selected: Any, *, candidate_count: int, ratio: float, target_count: int | None, torch: Any) -> Any:
    if target_count is None:
        step = max(1, int(1.0 / max(ratio, 1.0 / candidate_count)) // 3)
        uniform = torch.arange(0, candidate_count, step, device=selected.device)
        limit = candidate_count - 1
        # The released PruMerge+ adaptive path uses the union of attention
        # outliers and flat uniform samples.  It does not fill that union to
        # the maximum possible centre count.
        return torch.unique(torch.cat((selected, uniform)), sorted=False)[:limit]
    else:
        limit = min(candidate_count - 1, target_count)
        uniform_needed = max(0, limit - int(selected.numel()))
        if not uniform_needed:
            return selected[:limit]
        step = max(1, candidate_count // uniform_needed)
        uniform = torch.arange(0, candidate_count, step, device=selected.device)
    combined = torch.unique(torch.cat((selected, uniform)), sorted=False)
    if int(combined.numel()) < limit:
        all_indices = torch.arange(candidate_count, device=selected.device)
        remaining = all_indices[~torch.isin(all_indices, combined)]
        combined = torch.cat((combined, remaining))
    return combined[:limit]


def _importance(attention: Any, count: int, torch: Any) -> Any:
    if not isinstance(attention, torch.Tensor):
        raise TypeError("attention must be a tensor")
    value = attention
    if value.ndim == 4:
        if int(value.shape[0]) != 1:
            raise ValueError("attention batch must be one")
        value = value[0]
    if value.ndim == 3:
        value = value.detach().float().mean(dim=0)
    if value.ndim != 2 or tuple(value.shape) != (count, count):
        raise ValueError("attention must end in [tokens, tokens]")
    return value.detach().float().mean(dim=-2)


def _keys(keys: Any, count: int, torch: Any) -> Any:
    if not isinstance(keys, torch.Tensor):
        raise TypeError("keys must be a tensor")
    value = keys
    if value.ndim == 4:
        if int(value.shape[0]) != 1:
            raise ValueError("key batch must be one")
        value = value[0]
    if value.ndim == 3:
        if int(value.shape[1]) != count:
            raise ValueError("key token dimension does not match tokens")
        value = value.transpose(0, 1).reshape(count, -1)
    if value.ndim != 2 or int(value.shape[0]) != count:
        raise ValueError("keys must have shape [tokens, dim] or [heads, tokens, dim]")
    return value.detach()


def _indices(values: Iterable[int], upper: int, name: str) -> tuple[int, ...]:
    result = tuple(sorted(int(value) for value in values))
    if len(set(result)) != len(result) or any(value < 0 or value >= upper for value in result):
        raise ValueError(f"{name} contains an invalid index")
    return result


__all__ = ["PruMergeOutput", "PruMergeSpec", "prumerge_image_tokens"]
