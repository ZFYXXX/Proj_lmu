"""Backbone-neutral AIM bipartite visual-token merging."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping

from mllm_comp.core.layout import LayoutDelta, OrderingPolicy
from ._rules import score_rule, aggregate_rule, rule_metadata


@dataclass(frozen=True)
class AIMSpec:
    retain_ratio: float = 0.2
    target_count: int | None = None
    min_visual_tokens: int = 1

    def __post_init__(self) -> None:
        if isinstance(self.retain_ratio, bool) or not isinstance(
            self.retain_ratio, (int, float)
        ):
            raise TypeError("retain_ratio must be a real number.")
        if not 0.0 < float(self.retain_ratio) <= 1.0:
            raise ValueError("retain_ratio must be in (0, 1].")
        for name in ("target_count", "min_visual_tokens"):
            value = getattr(self, name)
            if value is not None and (
                isinstance(value, bool) or not isinstance(value, int) or value < 1
            ):
                raise ValueError(f"{name} must be a positive integer or None.")
        object.__setattr__(self, "retain_ratio", float(self.retain_ratio))

    def result_count(self, source_count: int) -> int:
        requested = (
            round(source_count * self.retain_ratio)
            if self.target_count is None
            else self.target_count
        )
        return min(source_count, max(self.min_visual_tokens, requested))


@dataclass(frozen=True)
class AIMOutput:
    tokens: Any
    representative_indices: Any
    token_sizes: Any
    layout_delta: LayoutDelta
    metadata: Mapping[str, Any] = field(default_factory=dict)
    merge_rounds: tuple[tuple[tuple[int, int], ...], ...] = ()
    eligible_indices: tuple[int, ...] = ()
    output_order: tuple[int, ...] = ()


def aim_image_tokens(
    tokens: Any,
    *,
    spec: AIMSpec | None = None,
    media_unit: str = "image_feature",
    source_layout: Mapping[str, Any] | None = None,
    result_layout: Mapping[str, Any] | None = None,
    protected_indices: Iterable[int] = (),
    rules: Mapping[str, Any] | None = None,
) -> AIMOutput:
    """Iteratively merge alternating source/destination tokens to an exact K."""

    torch = _torch()
    if (
        not isinstance(tokens, torch.Tensor)
        or tokens.ndim != 2
        or not tokens.is_floating_point()
    ):
        raise ValueError("tokens must be a floating rank-2 tensor.")
    count = int(tokens.shape[0])
    if count < 1:
        raise ValueError("AIM requires at least one visual token.")
    active = spec or AIMSpec()
    protected = tuple(sorted(set(int(value) for value in protected_indices)))
    if any(value < 0 or value >= count for value in protected):
        raise ValueError("protected_indices contains an out-of-range index.")
    if protected:
        protected_set = set(protected)
        eligible = torch.tensor(
            [index for index in range(count) if index not in protected_set],
            dtype=torch.long,
            device=tokens.device,
        )
        if int(eligible.numel()) == 0:
            indices = torch.arange(count, dtype=torch.long, device=tokens.device)
            delta = LayoutDelta.identity(
                count,
                media_unit=media_unit,
                protected_indices=protected,
                layout=source_layout or {},
            )
            return AIMOutput(
                tokens=tokens,
                representative_indices=indices,
                token_sizes=torch.ones(count, device=tokens.device),
                layout_delta=delta,
                metadata={"algorithm": "aim", "compression_applied": False, **rule_metadata(rules)},
            )
        total_target = active.result_count(count)
        eligible_target = min(
            int(eligible.numel()), max(1, total_target - len(protected))
        )
        nested = aim_image_tokens(
            tokens.index_select(0, eligible),
            spec=AIMSpec(target_count=eligible_target),
            media_unit=media_unit, rules=rules,
        )
        items: list[tuple[int, Any, tuple[int, ...], float]] = []
        for nested_result in range(nested.layout_delta.result_count):
            local_group = tuple(
                source
                for source, target in nested.layout_delta.old_to_new.items()
                if target == nested_result
            )
            source_group = tuple(int(eligible[index].item()) for index in local_group)
            representative = int(
                eligible[int(nested.representative_indices[nested_result].item())].item()
            )
            items.append(
                (
                    representative,
                    nested.tokens[nested_result],
                    source_group,
                    float(nested.token_sizes[nested_result].item()),
                )
            )
        for source in protected:
            items.append((source, tokens[source], (source,), 1.0))
        replay_representatives = [item[0] for item in items]
        output_order = tuple(sorted(range(len(items)), key=lambda i: replay_representatives[i]))
        items.sort(key=lambda item: item[0])
        output = torch.stack([item[1] for item in items], dim=0)
        mapping: list[int | None] = [None] * count
        for result_index, item in enumerate(items):
            for source in item[2]:
                mapping[source] = result_index
        representatives = torch.tensor(
            [item[0] for item in items], dtype=torch.long, device=tokens.device
        )
        sizes = torch.tensor(
            [item[3] for item in items], dtype=torch.float32, device=tokens.device
        )
        resolved_layout = dict(result_layout or {})
        resolved_layout.setdefault(
            "representative_indices", tuple(int(value) for value in representatives.tolist())
        )
        delta = LayoutDelta.from_mapping(
            source_count=count,
            result_count=len(items),
            old_to_new=mapping,
            media_unit=media_unit,
            protected_indices=protected,
            source_layout=source_layout or {},
            result_layout=resolved_layout,
            ordering_policy=OrderingPolicy.EXPLICIT,
        )
        return AIMOutput(
            tokens=output,
            representative_indices=representatives,
            token_sizes=sizes,
            layout_delta=delta,
            metadata={
                "algorithm": "aim",
                "source_tokens": count,
                "result_tokens": len(items),
                "protected_indices": protected,
                "compression_applied": not delta.is_identity,
                **rule_metadata(rules),
                **({"merge_value": "custom"} if rules and "aggregator" in rules else {}),
            },
            merge_rounds=nested.merge_rounds,
            eligible_indices=tuple(int(i) for i in eligible.tolist()),
            output_order=output_order,
        )
    target = active.result_count(count)
    current = tokens.float().unsqueeze(0)
    representatives = torch.arange(
        count, dtype=torch.long, device=tokens.device
    ).unsqueeze(0)
    groups: list[tuple[int, ...]] = [(index,) for index in range(count)]
    rounds: list[tuple[tuple[int, int], ...]] = []
    while int(current.shape[1]) > target:
        removed = min(int(current.shape[1]) - target, int(current.shape[1]) // 2)
        if removed < 1:
            break
        current, representatives, groups = _aim_merge_step(
            current, representatives, groups, removed, rules=rules,
            round_index=len(rounds), decisions=rounds,
        )
    output = current.squeeze(0).to(tokens.dtype)
    representative_values = tuple(int(value) for value in representatives[0].tolist())
    order = torch.argsort(representatives[0], stable=True)
    if not torch.equal(order, torch.arange(target, device=order.device)):
        output = output.index_select(0, order)
        representatives = representatives.index_select(1, order)
        groups = [groups[int(index)] for index in order.tolist()]
        representative_values = tuple(
            int(value) for value in representatives[0].tolist()
        )
    old_to_new = [None] * count
    for result_index, group in enumerate(groups):
        for source_index in group:
            old_to_new[source_index] = result_index
    resolved_result_layout = dict(result_layout or {})
    resolved_result_layout.setdefault(
        "representative_indices", representative_values
    )
    delta = LayoutDelta.from_mapping(
        source_count=count,
        result_count=target,
        old_to_new=old_to_new,
        media_unit=media_unit,
        source_layout=source_layout or {},
        result_layout=resolved_result_layout,
        ordering_policy=OrderingPolicy.EXPLICIT,
    )
    sizes = torch.tensor(
        [len(group) for group in groups],
        dtype=torch.float32,
        device=tokens.device,
    )
    return AIMOutput(
        tokens=output,
        representative_indices=representatives.squeeze(0),
        token_sizes=sizes,
        layout_delta=delta,
        metadata={
            "algorithm": "aim",
            "source_tokens": count,
            "result_tokens": target,
            "representative_indices": representative_values,
            "cluster_sizes": tuple(int(value) for value in sizes.tolist()),
            "compression_applied": not delta.is_identity,
            **rule_metadata(rules),
            **({"merge_value": "custom"} if rules and "aggregator" in rules else {}),
        },
        merge_rounds=tuple(rounds), eligible_indices=tuple(range(count)),
        output_order=tuple(int(i) for i in order.tolist()),
    )


def _aim_merge_step(
    tokens: Any,
    representatives: Any,
    groups: list[tuple[int, ...]],
    removed: int,
    *, rules: Mapping[str, Any] | None = None, round_index: int = 0,
    decisions: list[Any] | None = None, pairs: tuple[tuple[int, int], ...] | None = None,
) -> tuple[Any, Any, list[tuple[int, ...]]]:
    torch = _torch()
    with torch.no_grad():
        metric = tokens / tokens.norm(dim=-1, keepdim=True).clamp(min=1e-12)
        source_metric = metric[..., ::2, :]
        destination_metric = metric[..., 1::2, :]
        if pairs is None:
            scores = source_metric @ destination_metric.transpose(-1, -2)
            request = {"values": tokens[0].detach(), "metric": metric[0],
                "source_indices": tuple(range(0, len(groups), 2)),
                "destination_indices": tuple(range(1, len(groups), 2)),
                "merge_count": removed, "round_index": round_index,
                "scores": scores[0]}
            scores = score_rule(rules, "scorer", request, scores[0]).unsqueeze(0)
            node_max, node_index = scores.max(dim=-1)
            edge_order = node_max.argsort(dim=-1, descending=True, stable=True)[..., None]
            unmerged_index = edge_order[..., removed:, :]
            source_index = edge_order[..., :removed, :]
            destination_index = node_index[..., None].gather(-2, source_index)
            if rules and "grouper" in rules:
                pairs = _validate_pairs(rules["grouper"]({**request, "scores": scores[0]}),
                    count=len(groups), removed=removed)
        if pairs is not None:
            source_index = torch.tensor([source // 2 for source, _ in pairs], device=tokens.device).reshape(1, removed, 1)
            destination_index = torch.tensor([destination // 2 for _, destination in pairs], device=tokens.device).reshape(1, removed, 1)
            selected = {source for source, _ in pairs}
            unmerged_index = torch.tensor([i // 2 for i in range(0, len(groups), 2) if i not in selected],
                device=tokens.device, dtype=torch.long).reshape(1, -1, 1)
        actual_pairs = tuple((int(source) * 2, int(destination) * 2 + 1)
            for source, destination in zip(source_index.flatten().tolist(), destination_index.flatten().tolist()))
        if decisions is not None:
            decisions.append(actual_pairs)
        destination_count = int(destination_metric.shape[-2])
        unmerged_positions = unmerged_index.squeeze(-1) * 2
        destination_positions = (
            torch.arange(destination_count, device=tokens.device).unsqueeze(0) * 2
            + 1
        )
        all_positions = torch.cat((unmerged_positions, destination_positions), dim=1)
        sort_index = all_positions.argsort(dim=1, stable=True)

    source = tokens[..., ::2, :]
    destination = tokens[..., 1::2, :]
    batch, source_count, hidden = source.shape
    unmerged = source.gather(
        -2, unmerged_index.expand(batch, source_count - removed, hidden)
    )
    selected_source = source.gather(
        -2, source_index.expand(batch, removed, hidden)
    )
    destination = destination.scatter_reduce(
        -2,
        destination_index.expand(batch, removed, hidden),
        selected_source,
        reduce="mean",
    )
    if rules and "aggregator" in rules:
        rows = []
        for destination_local in range(destination_count):
            destination_position = destination_local * 2 + 1
            assigned = tuple(source for source, target in actual_pairs if target == destination_position)
            if not assigned:
                rows.append(tokens[0, destination_position])
                continue
            indices = (destination_position, *assigned)
            group_values = tokens[0, list(indices)]
            value = aggregate_rule(rules, "aggregator", {"values": group_values,
                "indices": indices, "destination": destination_position, "round_index": round_index,
                "token_sizes": group_values.new_tensor([len(groups[i]) for i in indices]),
                "default_values": destination[0, destination_local]}, destination[0, destination_local])
            if torch.is_grad_enabled() and group_values.requires_grad and not value.requires_grad:
                raise ValueError("AIM aggregator detached the value gradient.")
            rows.append(value)
        destination = torch.stack(rows).unsqueeze(0)
    merged = torch.cat((unmerged, destination), dim=1)
    merged = merged.gather(1, sort_index.unsqueeze(-1).expand_as(merged))

    rep_source = representatives[..., ::2]
    rep_destination = representatives[..., 1::2]
    unmerged_rep = rep_source.gather(-1, unmerged_index.squeeze(-1))
    new_rep = torch.cat((unmerged_rep, rep_destination), dim=-1)
    new_rep = new_rep.gather(1, sort_index)

    source_slots = list(range(0, len(groups), 2))
    destination_slots = list(range(1, len(groups), 2))
    source_values = [source // 2 for source, _ in actual_pairs]
    destination_values = [destination // 2 for _, destination in actual_pairs]
    destination_groups = [groups[index] for index in destination_slots]
    for source_local, destination_local in zip(source_values, destination_values):
        destination_groups[destination_local] = (
            destination_groups[destination_local] + groups[source_slots[source_local]]
        )
    unmerged_values = [int(value) for value in unmerged_index[0, :, 0].tolist()]
    new_groups = [groups[source_slots[index]] for index in unmerged_values]
    new_groups.extend(destination_groups)
    sort_values = [int(value) for value in sort_index[0].tolist()]
    new_groups = [tuple(sorted(new_groups[index])) for index in sort_values]
    return merged, new_rep, new_groups


def _validate_pairs(value: Any, *, count: int, removed: int) -> tuple[tuple[int, int], ...]:
    torch = _torch()
    if isinstance(value, torch.Tensor):
        if value.dtype not in (torch.int32, torch.int64) or value.shape != (removed, 2):
            raise ValueError("AIM grouper must return integer [merge_count, 2] pairs.")
        value = value.detach().cpu().tolist()
    if not isinstance(value, (list, tuple)) or len(value) != removed:
        raise ValueError("AIM grouper must preserve the exact round merge budget.")
    pairs = []
    for pair in value:
        if (not isinstance(pair, (list, tuple)) or len(pair) != 2
                or any(isinstance(i, bool) or not isinstance(i, int) for i in pair)):
            raise ValueError("AIM grouper must return integer source/destination pairs.")
        source, destination = pair
        if not (0 <= source < count and source % 2 == 0 and 0 <= destination < count and destination % 2 == 1):
            raise ValueError("AIM grouper must retain the current even-source/odd-destination partition.")
        pairs.append((source, destination))
    if len({source for source, _ in pairs}) != removed:
        raise ValueError("AIM grouper may select each source only once.")
    return tuple(pairs)


def replay_aim_image_tokens(tokens: Any, result: AIMOutput, *, rules: Mapping[str, Any] | None = None) -> Any:
    """Replay saved per-round decisions on actual values without rematching.

    The optional aggregator receives the current nodes each round. Token sizes
    remain lineage counts, not implicit weights in the original mean formula.
    """
    torch = _torch()
    if tokens.ndim != 2 or int(tokens.shape[0]) != result.layout_delta.source_count:
        raise ValueError("AIM replay values must match the original source length.")
    if result.layout_delta.is_identity:
        return tokens
    eligible = result.eligible_indices
    if not eligible or not result.merge_rounds:
        raise ValueError("AIM replay requires the saved eligible layout and merge rounds.")
    indices = torch.tensor(eligible, device=tokens.device, dtype=torch.long)
    current = tokens.index_select(0, indices).float().unsqueeze(0)
    representatives = torch.arange(len(eligible), device=tokens.device).unsqueeze(0)
    groups = [(i,) for i in range(len(eligible))]
    for round_index, pairs in enumerate(result.merge_rounds):
        current, representatives, groups = _aim_merge_step(current, representatives, groups, len(pairs),
            rules=rules, round_index=round_index, pairs=pairs)
    values = current[0].to(tokens.dtype)
    protected = result.layout_delta.protected_indices
    if protected:
        values = torch.cat((values, tokens[list(protected)]))
    order = torch.tensor(result.output_order, device=tokens.device, dtype=torch.long)
    return values.index_select(0, order)


def _torch() -> Any:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("AIM execution requires PyTorch.") from exc
    return torch


__all__ = ["AIMOutput", "AIMSpec", "aim_image_tokens", "replay_aim_image_tokens"]
