"""Backbone-neutral SAINT redundancy pruning for visual patch tokens.

The kernel owns only SAINT's alternating-set score and deterministic token
selection.  A backbone adapter is responsible for invoking it at configured
encoder boundaries and for updating attention segments, rotary positions and
the final image-placeholder layout.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping

from mllm_comp.core.layout import LayoutDelta, LayoutTokenUnit
from ._rules import rule_metadata, score_rule, selection_rule


@dataclass(frozen=True)
class SAINTSpec:
    """Cardinality and scoring controls from the supplied Qwen2.5-VL port."""

    keep_ratio: float = 0.5
    target_count: int | None = None
    prune_level: str = "token"
    group_size: int = 4
    similarity_threshold: float = 0.7
    neighbour_count: int = 5
    gamma: float = 10.0
    max_iterations: int = 32

    def __post_init__(self) -> None:
        if isinstance(self.keep_ratio, bool) or not isinstance(
            self.keep_ratio, (int, float)
        ):
            raise TypeError("keep_ratio must be a real number.")
        if not 0.0 < float(self.keep_ratio) <= 1.0:
            raise ValueError("keep_ratio must be in (0, 1].")
        if self.target_count is not None and (
            isinstance(self.target_count, bool)
            or not isinstance(self.target_count, int)
            or self.target_count < 1
        ):
            raise ValueError("target_count must be a positive integer or None.")
        if self.prune_level not in {"token", "group"}:
            raise ValueError("prune_level must be 'token' or 'group'.")
        for name in ("group_size", "neighbour_count", "max_iterations"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer.")
        for name in ("similarity_threshold", "gamma"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(f"{name} must be a real number.")
        if not -1.0 <= float(self.similarity_threshold) <= 1.0:
            raise ValueError("similarity_threshold must be in [-1, 1].")
        if float(self.gamma) < 0.0:
            raise ValueError("gamma must be non-negative.")
        object.__setattr__(self, "keep_ratio", float(self.keep_ratio))
        object.__setattr__(
            self, "similarity_threshold", float(self.similarity_threshold)
        )
        object.__setattr__(self, "gamma", float(self.gamma))

    def result_count(self, source_count: int) -> int:
        if isinstance(source_count, bool) or not isinstance(source_count, int):
            raise TypeError("source_count must be an integer.")
        if source_count < 1:
            raise ValueError("source_count must be positive.")
        if self.prune_level == "group":
            groups = source_count // self.group_size
            if groups < 1:
                return source_count
            requested = (
                max(1, round(groups * self.keep_ratio))
                if self.target_count is None
                else max(1, self.target_count // self.group_size)
            )
            return min(groups, requested) * self.group_size
        aligned = (source_count // self.group_size) * self.group_size
        if aligned < 1:
            return source_count
        requested = (
            max(self.group_size, round(source_count * self.keep_ratio))
            if self.target_count is None
            else max(self.group_size, self.target_count)
        )
        requested = (requested // self.group_size) * self.group_size
        return min(source_count, aligned, requested)


@dataclass(frozen=True)
class SAINTOutput:
    tokens: Any
    selected_indices: Any
    layout_delta: LayoutDelta
    metadata: Mapping[str, Any] = field(default_factory=dict)


def saint_image_tokens(
    tokens: Any,
    keys: Any,
    *,
    spec: SAINTSpec | None = None,
    media_unit: str = LayoutTokenUnit.IMAGE_PATCH.value,
    source_layout: Mapping[str, Any] | None = None,
    result_layout: Mapping[str, Any] | None = None,
    protected_indices: Iterable[int] = (),
    rules: Mapping[str, Any] | None = None,
) -> SAINTOutput:
    """Select one SAINT-pruned visual patch sequence in source order."""

    torch = _torch()
    functional = _functional()
    active = spec or SAINTSpec()
    if (
        not isinstance(tokens, torch.Tensor)
        or tokens.ndim != 2
        or not tokens.is_floating_point()
    ):
        raise ValueError("tokens must be a floating rank-2 tensor.")
    if (
        not isinstance(keys, torch.Tensor)
        or keys.ndim not in {2, 3}
        or not keys.is_floating_point()
    ):
        raise ValueError("keys must be a floating rank-2 or rank-3 tensor.")
    count = int(tokens.shape[0])
    if count < 1 or int(keys.shape[0]) != count:
        raise ValueError("tokens and keys must have the same positive token count.")
    keys = keys.to(device=tokens.device)
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
            return _identity_output(
                tokens,
                media_unit=media_unit,
                protected=protected,
                source_layout=source_layout or {},
            )
        total_target = active.result_count(count)
        eligible_target = max(1, total_target - len(protected))
        eligible_target = min(int(eligible.numel()), eligible_target)
        nested = saint_image_tokens(
            tokens.index_select(0, eligible),
            keys.index_select(0, eligible),
            spec=SAINTSpec(
                keep_ratio=1.0,
                target_count=eligible_target,
                prune_level=active.prune_level,
                group_size=min(active.group_size, eligible_target),
                similarity_threshold=active.similarity_threshold,
                neighbour_count=active.neighbour_count,
                gamma=active.gamma,
                max_iterations=active.max_iterations,
            ),
            media_unit=media_unit,
            rules=rules,
        )
        selected = torch.cat(
            (eligible.index_select(0, nested.selected_indices),
             torch.tensor(protected, dtype=torch.long, device=tokens.device))
        ).sort().values
        output = tokens.index_select(0, selected)
        selected_values = tuple(int(value) for value in selected.tolist())
        mapping = [None] * count
        for result_index, source_index in enumerate(selected_values):
            mapping[source_index] = result_index
        resolved_layout = dict(result_layout or {})
        resolved_layout.setdefault("representative_indices", selected_values)
        delta = LayoutDelta.from_mapping(
            source_count=count,
            result_count=len(selected_values),
            old_to_new=mapping,
            media_unit=media_unit,
            protected_indices=protected,
            source_layout=source_layout or {},
            result_layout=resolved_layout,
        )
        return SAINTOutput(
            tokens=output,
            selected_indices=selected,
            layout_delta=delta,
            metadata={
                **dict(nested.metadata),
                "source_tokens": count,
                "result_tokens": len(selected_values),
                "protected_indices": protected,
                "compression_applied": not delta.is_identity,
            },
        )
    flat_keys = keys.detach().float().reshape(count, -1)
    target = active.result_count(count)
    if target >= count:
        selected = torch.arange(count, dtype=torch.long, device=tokens.device)
    elif active.prune_level == "group":
        groups = count // active.group_size
        aligned = groups * active.group_size
        group_keys = flat_keys[:aligned].reshape(groups, active.group_size, -1)
        metric = functional.normalize(group_keys.mean(dim=1), p=2, dim=-1)
        kept_groups = _iterative_saint_indices(
            metric, target // active.group_size, active, rules=rules
        )
        offsets = torch.arange(
            active.group_size, dtype=torch.long, device=tokens.device
        )
        selected = (
            kept_groups.unsqueeze(1) * active.group_size + offsets
        ).reshape(-1)
    else:
        metric = functional.normalize(flat_keys, p=2, dim=-1)
        selected = _iterative_saint_indices(metric, target, active, rules=rules)
    selected = selected.sort().values
    output = tokens.index_select(0, selected)
    selected_values = tuple(int(value) for value in selected.tolist())
    selected_set = set(selected_values)
    mapping = [None] * count
    for result_index, source_index in enumerate(selected_values):
        mapping[source_index] = result_index
    resolved_result_layout = dict(result_layout or {})
    resolved_result_layout.setdefault("representative_indices", selected_values)
    delta = LayoutDelta.from_mapping(
        source_count=count,
        result_count=len(selected_values),
        old_to_new=mapping,
        media_unit=media_unit,
        source_layout=source_layout or {},
        result_layout=resolved_result_layout,
    )
    return SAINTOutput(
        tokens=output,
        selected_indices=selected,
        layout_delta=delta,
        metadata={
            "algorithm": "saint",
            **rule_metadata(rules),
            "prune_level": active.prune_level,
            "source_tokens": count,
            "result_tokens": len(selected_values),
            "target_tokens": target,
            "dropped_indices": tuple(
                index for index in range(count) if index not in selected_set
            ),
            "compression_applied": not delta.is_identity,
        },
    )


def _identity_output(
    tokens: Any,
    *,
    media_unit: str,
    protected: tuple[int, ...],
    source_layout: Mapping[str, Any],
) -> SAINTOutput:
    torch = _torch()
    count = int(tokens.shape[0])
    indices = torch.arange(count, dtype=torch.long, device=tokens.device)
    delta = LayoutDelta.identity(
        count,
        media_unit=media_unit,
        protected_indices=protected,
        layout=source_layout,
    )
    return SAINTOutput(
        tokens=tokens,
        selected_indices=indices,
        layout_delta=delta,
        metadata={
            "algorithm": "saint",
            "source_tokens": count,
            "result_tokens": count,
            "compression_applied": False,
        },
    )


def saint_redundancy_indices(
    normalised_keys: Any,
    target_count: int,
    *,
    similarity_threshold: float,
    neighbour_count: int,
    gamma: float,
    rules: Mapping[str, Any] | None = None,
) -> Any:
    """Run one alternating-set SAINT selection pass.

    ``neighbour_count`` is retained as a first-class reference parameter.  The
    supplied implementation, like the aligned SAINT port, scores every pair
    above the threshold and does not use K as an additional eligibility gate.
    """

    del neighbour_count
    torch = _torch()
    count = int(normalised_keys.shape[0])
    if target_count >= count:
        return torch.arange(count, dtype=torch.long, device=normalised_keys.device)
    a = normalised_keys[::2]
    b = normalised_keys[1::2]
    if int(b.shape[0]) == 0:
        return torch.arange(count, dtype=torch.long, device=normalised_keys.device)
    with torch.no_grad():
        scores = a @ b.transpose(0, 1)
        valid = scores >= similarity_threshold
        valid_counts = valid.sum(dim=-1)
        average = (scores * valid).sum(dim=-1) / valid_counts.clamp(min=1)
        primary = valid_counts.float() * torch.exp(
            gamma * (average - similarity_threshold)
        )
        fallback = scores.mean(dim=-1)
        redundancy = torch.where(valid_counts > 0, primary, fallback)
        redundancy = score_rule(rules, "scorer",
            {"normalized_keys": normalised_keys, "similarities": scores,
             "keep_count": target_count, "similarity_threshold": similarity_threshold,
             "gamma": gamma}, redundancy)
        remove = min(count - target_count, int(a.shape[0]))
        custom = selection_rule(rules, "selector",
            {"normalized_keys": normalised_keys, "scores": redundancy,
             "keep_count": count - remove, "target_count": target_count},
            count=count, keep_count=count - remove, device=normalised_keys.device)
        if custom is not None:
            return custom.sort().values
        order = torch.argsort(redundancy, descending=True, stable=True)
        kept_a = order[remove:]
        a_positions = torch.arange(
            0, count, 2, dtype=torch.long, device=normalised_keys.device
        )
        b_positions = torch.arange(
            1, count, 2, dtype=torch.long, device=normalised_keys.device
        )
        return torch.cat((a_positions[kept_a], b_positions)).sort().values


def _iterative_saint_indices(keys: Any, target: int, spec: SAINTSpec,
                             *, rules: Mapping[str, Any] | None = None) -> Any:
    torch = _torch()
    selected = torch.arange(
        int(keys.shape[0]), dtype=torch.long, device=keys.device
    )
    current = keys
    for _ in range(spec.max_iterations):
        if int(selected.numel()) <= target:
            break
        kept = saint_redundancy_indices(
            current,
            target,
            similarity_threshold=spec.similarity_threshold,
            neighbour_count=spec.neighbour_count,
            gamma=spec.gamma,
            rules=rules,
        )
        if int(kept.numel()) >= int(selected.numel()):
            break
        selected = selected.index_select(0, kept)
        current = current.index_select(0, kept)
    if int(selected.numel()) != target:
        raise RuntimeError(
            "SAINT selection failed to reach the configured aligned target count."
        )
    return selected


def _torch() -> Any:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("SAINT execution requires PyTorch.") from exc
    return torch


def _functional() -> Any:
    try:
        import torch.nn.functional as functional
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("SAINT execution requires PyTorch.") from exc
    return functional


__all__ = [
    "SAINTOutput",
    "SAINTSpec",
    "saint_image_tokens",
    "saint_redundancy_indices",
]
