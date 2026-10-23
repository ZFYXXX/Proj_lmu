"""Validation at image algorithm rule boundaries; no model or layout execution."""

from __future__ import annotations

from types import MappingProxyType
from typing import Any, Mapping


def score_rule(rules: Mapping[str, Any] | None, name: str, request: Mapping[str, Any], default: Any,
               *, expected_shape: tuple[int, ...] | None = None, dtype: Any = None) -> Any:
    """Evaluate scores using a default or explicit shape/dtype specification.

    An explicit shape lets callers use an existing dtype/device reference
    without computing the default scores that the custom rule replaces.
    """
    import torch

    if not rules or name not in rules:
        return default
    with torch.no_grad():
        value = rules[name](MappingProxyType(dict(request)))
    if not isinstance(value, torch.Tensor) or not value.is_floating_point():
        raise ValueError(f"{name} must return a floating tensor.")
    shape = tuple(default.shape) if expected_shape is None else expected_shape
    if tuple(value.shape) != shape or not bool(torch.isfinite(value).all()):
        raise ValueError(f"{name} must return finite scores with shape {shape}.")
    return value.detach().to(device=default.device, dtype=default.dtype if dtype is None else dtype)


def selection_rule(rules: Mapping[str, Any] | None, name: str, request: Mapping[str, Any],
                   *, count: int, keep_count: int, device: Any,
                   protected: Any = ()) -> Any | None:
    """Validate local indices; the enclosing algorithm owns ordering and gather."""
    import torch

    if not rules or name not in rules:
        return None
    with torch.no_grad():
        selected = rules[name](MappingProxyType(dict(request)))
    if (not isinstance(selected, torch.Tensor) or selected.ndim != 1
            or selected.dtype not in (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8)):
        raise ValueError(f"{name} must return a one-dimensional integer tensor.")
    selected = selected.detach().to(device=device, dtype=torch.long)
    if (selected.numel() != keep_count or selected.unique().numel() != keep_count
            or bool(((selected < 0) | (selected >= count)).any())):
        raise ValueError(f"{name} must return {keep_count} unique indices in [0, {count}).")
    if not set(int(i) for i in protected).issubset(set(selected.tolist())):
        raise ValueError(f"{name} must retain protected indices.")
    return selected


def rule_metadata(rules: Mapping[str, Any] | None) -> dict[str, Any]:
    return ({"rule_hooks": tuple(rules), "rules_changed": True,
             "canonical_evidence_applicable": False, "development_runtime": True}
            if rules else {})


def label_rule(rules: Mapping[str, Any] | None, name: str, request: Mapping[str, Any], default: Any) -> Any:
    """Replace per-token bucket labels without taking ownership of layout construction."""
    import torch

    if not rules or name not in rules:
        return default
    with torch.no_grad():
        value = rules[name](MappingProxyType(dict(request)))
    if (not isinstance(value, torch.Tensor) or tuple(value.shape) != tuple(default.shape)
            or value.dtype not in (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8)):
        raise ValueError(f"{name} must return integer labels with shape {tuple(default.shape)}.")
    return value.detach().to(device=default.device, dtype=torch.long)


def aggregate_rule(rules: Mapping[str, Any] | None, name: str, request: Mapping[str, Any], default: Any) -> Any:
    """Apply a stateless value rule in the caller's gradient/replay scope."""
    import torch

    if not rules or name not in rules:
        return default
    value = rules[name](MappingProxyType(dict(request)))
    if (not isinstance(value, torch.Tensor) or not value.is_floating_point()
            or tuple(value.shape) != tuple(default.shape) or not bool(torch.isfinite(value).all())):
        raise ValueError(f"{name} must return finite values with shape {tuple(default.shape)}.")
    if torch.is_grad_enabled() and default.requires_grad and not value.requires_grad:
        raise ValueError(f"{name} must preserve the value gradient.")
    return value.to(device=default.device, dtype=default.dtype)


def replay_rule_groups(values: Any, groups: Any, *, rules: Mapping[str, Any] | None = None,
                       source_groups: Any = None) -> Any:
    """Replay a fixed merge membership; only value aggregation is re-evaluated."""
    from .decoder_lifecycle import replay_mean_groups
    import torch

    if not rules or "aggregator" not in rules:
        return replay_mean_groups(values, groups)
    outputs = []
    for index, group in enumerate(groups):
        members = values.index_select(0, torch.tensor(group, device=values.device, dtype=torch.long))
        default = members.mean(dim=0)
        if len(group) > 1:
            default = aggregate_rule(rules, "aggregator", {
                "values": members,
                "source_indices": tuple(group if source_groups is None else source_groups[index]),
                "default_values": default,
            }, default)
        outputs.append(default)
    return torch.stack(outputs)
