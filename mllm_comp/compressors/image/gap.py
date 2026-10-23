"""Grounding-Aware Pruning (GAP) positional pruning protocol."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Iterable, Mapping

from mllm_comp.core.layout import LayoutDelta, LayoutTokenUnit
from ._rules import rule_metadata, score_rule, selection_rule


@dataclass(frozen=True)
class GAPSpec:
    retain_ratio: float = 0.5
    target_count: int | None = None

    def __post_init__(self) -> None:
        if not 0.0 < self.retain_ratio <= 1.0:
            raise ValueError("retain_ratio must be in (0, 1]")
        if self.target_count is not None and self.target_count < 1:
            raise ValueError("target_count must be positive")


@dataclass(frozen=True)
class GAPOutput:
    tokens: Any
    indices: Any
    layout_delta: LayoutDelta
    metadata: Mapping[str, Any]

    @property
    def kept_indices(self) -> Any:
        return self.indices


def gap_image_tokens(
    tokens: Any,
    attention: Any,
    *,
    spec: GAPSpec = GAPSpec(),
    protected_indices: Iterable[int] = (),
    coordinates: Iterable[Any] = (),
    media_unit: str | LayoutTokenUnit = LayoutTokenUnit.IMAGE_PATCH,
    source_layout: Mapping[str, Any] | None = None,
    result_layout: Mapping[str, Any] | None = None,
    rules: Mapping[str, Any] | None = None,
) -> GAPOutput:
    """Top-k by received attention, then restore source order exactly."""

    import torch

    if not isinstance(tokens, torch.Tensor) or tokens.ndim != 2:
        raise ValueError("tokens must have shape [tokens, hidden]")
    source_count = int(tokens.shape[0])
    if source_count < 1:
        raise ValueError("GAP cannot prune an empty image")
    protected = _indices(protected_indices, source_count)
    candidates = tuple(index for index in range(source_count) if index not in protected)
    if spec.target_count is None:
        regular_target = max(1, int(round(len(candidates) * spec.retain_ratio)))
    else:
        regular_target = max(0, int(spec.target_count) - len(protected))
    regular_target = min(len(candidates), regular_target)
    importance = _importance(attention, source_count, torch).to(device=tokens.device)
    request = {"tokens": tokens, "attention": attention, "keep_count": regular_target + len(protected),
               "protected_indices": protected, "spec": spec}
    if request["keep_count"] < source_count:
        importance = score_rule(rules, "scorer", request, importance)
    custom = None
    if request["keep_count"] < source_count:
        custom = selection_rule(rules, "selector", {**request, "scores": importance},
            count=source_count, keep_count=request["keep_count"], device=tokens.device, protected=protected)
    if custom is not None:
        kept = custom.sort().values
    else:
        if regular_target:
            candidate_tensor = torch.tensor(candidates, dtype=torch.long, device=tokens.device)
            ranked = importance.index_select(0, candidate_tensor).topk(
                regular_target, sorted=False
            ).indices
            selected = candidate_tensor.index_select(0, ranked)
        else:
            selected = torch.empty(0, dtype=torch.long, device=tokens.device)
        kept = torch.tensor(
            sorted((*protected, *(int(value) for value in selected.detach().cpu().tolist()))),
            dtype=torch.long,
            device=tokens.device,
        )
    kept_set = set(int(value) for value in kept.detach().cpu().tolist())
    delta = LayoutDelta.drop(
        source_count,
        (index for index in range(source_count) if index not in kept_set),
        media_unit=media_unit,
        coordinates=coordinates,
        protected_indices=protected,
        source_layout=source_layout,
        result_layout={
            **dict(result_layout or {}),
            "position_policy": "preserve_original",
        },
    )
    return GAPOutput(
        tokens=tokens.index_select(0, kept),
        indices=kept,
        layout_delta=delta,
        metadata=MappingProxyType(
            {
                "algorithm": "gap",
                "variant": "gap.image",
                "source_count": source_count,
                "result_count": int(kept.numel()),
                "retain_ratio": spec.retain_ratio,
                "requested_target_count": spec.target_count,
                "importance_source": "custom" if rules and "scorer" in rules else "mean_received_attention",
                "selection": "custom" if rules and "selector" in rules else "cls_visual_topk_proxy",
                **rule_metadata(rules),
                "token_order": "original",
                "position_policy": "preserve_original",
                "protected_count": len(protected),
            }
        ),
    )


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


def _indices(values: Iterable[int], upper: int) -> tuple[int, ...]:
    result = tuple(sorted(int(value) for value in values))
    if len(set(result)) != len(result) or any(value < 0 or value >= upper for value in result):
        raise ValueError("protected_indices contains an invalid index")
    return result


__all__ = ["GAPOutput", "GAPSpec", "gap_image_tokens"]
