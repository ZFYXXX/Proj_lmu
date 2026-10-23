"""Backbone-neutral VisPruner visual-cue token selection.

The released method splits one output budget between tokens with high visual
encoder attention and tokens that remain diverse after iterative bipartite
duplicate removal.  This module owns that math once.  Backbone bindings own
attention capture, crop/image partitioning, projector/merger boundaries, and
prompt-layout legalization.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from types import MappingProxyType
from typing import Any, Mapping

from mllm_comp.core.layout import LayoutDelta, LayoutTokenUnit
from ._rules import rule_metadata, score_rule, selection_rule


VISPRUNER_CONFIG_KEYS = frozenset(
    {"target_count", "important_ratio", "removal_step", "min_visual_tokens"}
)


def merge_vispruner_executor_config(
    *,
    stage: Any,
    binding_config: Any,
    extra_keys: frozenset[str] = frozenset(),
) -> dict[str, Any]:
    """Merge method knobs and reject misspelled explicit configuration."""

    allowed = VISPRUNER_CONFIG_KEYS | extra_keys
    result: dict[str, Any] = {}
    if stage is not None:
        for name in ("operation", "criterion", "schedule"):
            value = getattr(stage, name, None)
            if isinstance(value, Mapping):
                result.update(
                    {key: item for key, item in value.items() if key in allowed}
                )
    for source, value in (
        ("stage.config", getattr(stage, "config", None)),
        ("binding_config.config", getattr(binding_config, "config", None)),
    ):
        if not isinstance(value, Mapping):
            continue
        unknown = sorted(set(value) - allowed)
        if unknown:
            raise ValueError(f"Unknown VisPruner {source} keys: {unknown}.")
        result.update(dict(value))
    return result


@dataclass(frozen=True)
class VisPrunerSpec:
    """Released selection defaults for one image or crop sequence."""

    target_count: int = 32
    important_ratio: float = 0.5
    removal_step: int = 8
    min_visual_tokens: int = 1

    def __post_init__(self) -> None:
        if isinstance(self.target_count, bool) or not isinstance(
            self.target_count, int
        ):
            raise TypeError("target_count must be an integer.")
        if isinstance(self.important_ratio, bool) or not isinstance(
            self.important_ratio, (int, float)
        ):
            raise TypeError("important_ratio must be a real number.")
        if not math.isfinite(float(self.important_ratio)):
            raise ValueError("important_ratio must be finite.")
        if not 0.0 <= float(self.important_ratio) <= 1.0:
            raise ValueError("important_ratio must be in [0, 1].")
        for name in ("removal_step", "min_visual_tokens"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be an integer.")
            if value < 1:
                raise ValueError(f"{name} must be at least one.")
        if self.target_count < self.min_visual_tokens:
            raise ValueError("target_count must be at least min_visual_tokens.")

    def resolve_target_count(
        self,
        source_count: int,
        *,
        target_count: int | None = None,
    ) -> int:
        _require_positive_int("source_count", source_count)
        requested = self.target_count if target_count is None else target_count
        _require_positive_int("target_count", requested)
        return min(source_count, max(self.min_visual_tokens, requested))


@dataclass(frozen=True)
class VisPrunerOutput:
    """Source-ordered selected output tokens and exact drop mapping."""

    tokens: Any
    indices: Any
    selection_order: Any
    layout_delta: LayoutDelta
    metadata: Mapping[str, Any]

    @property
    def selected_indices(self) -> Any:
        return self.indices


def select_vispruner_indices(
    selection_features: Any,
    attention: Any,
    target_count: int,
    *,
    important_ratio: float = 0.5,
    removal_step: int = 8,
    rules: Mapping[str, Any] | None = None,
) -> Any:
    """Return released important-plus-diverse indices in selection order."""

    torch = _torch()
    _validate_features(selection_features, "selection_features", torch=torch)
    _require_positive_int("target_count", target_count)
    _require_positive_int("removal_step", removal_step)
    if isinstance(important_ratio, bool) or not isinstance(
        important_ratio, (int, float)
    ):
        raise TypeError("important_ratio must be a real number.")
    if not math.isfinite(float(important_ratio)) or not 0.0 <= float(
        important_ratio
    ) <= 1.0:
        raise ValueError("important_ratio must be finite and in [0, 1].")
    source_count = int(selection_features.shape[0])
    if target_count > source_count:
        raise ValueError("target_count cannot exceed the source token count.")

    importance = _mean_received_attention(
        attention,
        token_count=source_count,
        torch=torch,
    )
    request = {"selection_features": selection_features, "attention": attention,
               "keep_count": target_count, "important_ratio": important_ratio,
               "removal_step": removal_step}
    if target_count < source_count:
        importance = score_rule(rules, "scorer", request, importance)
        custom = selection_rule(rules, "selector", {**request, "scores": importance},
            count=source_count, keep_count=target_count, device=selection_features.device)
        if custom is not None:
            return custom
    ranking = torch.argsort(importance, descending=True, stable=True)
    important_count = min(target_count, int(target_count * float(important_ratio)))
    diverse_count = target_count - important_count
    important = ranking[:important_count]
    residual = ranking[important_count:]

    if diverse_count:
        features = torch.nn.functional.normalize(
            selection_features.detach().float(), dim=-1, eps=1e-6
        )
        while int(residual.numel()) > diverse_count:
            remove_count = min(removal_step, int(residual.numel()) - diverse_count)
            even = residual[::2]
            odd = residual[1::2]
            if int(odd.numel()) == 0:
                break
            duplicate = (
                features.index_select(0, even)
                @ features.index_select(0, odd).transpose(0, 1)
            ).max(dim=-1).values
            kept_even_rank = torch.argsort(
                duplicate, descending=True, stable=True
            )[remove_count:]
            residual = torch.cat((even.index_select(0, kept_even_rank), odd), dim=0)
        selected = torch.cat((important, residual), dim=0)
    else:
        selected = important
    if int(selected.numel()) != target_count:
        raise RuntimeError(
            f"VisPruner selected {int(selected.numel())} tokens, expected "
            f"{target_count}."
        )
    return selected


def vispruner_image_tokens(
    output_tokens: Any,
    selection_features: Any,
    attention: Any,
    *,
    spec: VisPrunerSpec | None = None,
    target_count: int | None = None,
    media_unit: str | LayoutTokenUnit = "image_feature",
    source_layout: Mapping[str, Any] | None = None,
    result_layout: Mapping[str, Any] | None = None,
    rules: Mapping[str, Any] | None = None,
) -> VisPrunerOutput:
    """Select one image sequence and restore outputs to source order."""

    torch = _torch()
    _validate_features(output_tokens, "output_tokens", torch=torch)
    _validate_features(selection_features, "selection_features", torch=torch)
    if int(output_tokens.shape[0]) != int(selection_features.shape[0]):
        raise ValueError("output and selection feature counts must match.")
    if output_tokens.device != selection_features.device:
        raise ValueError("output and selection features must share one device.")
    active = spec or VisPrunerSpec()
    source_count = int(output_tokens.shape[0])
    resolved_target = active.resolve_target_count(
        source_count,
        target_count=target_count,
    )
    selection_order = select_vispruner_indices(
        selection_features,
        attention,
        resolved_target,
        important_ratio=float(active.important_ratio),
        removal_step=active.removal_step,
        rules=rules,
    )
    indices = selection_order.sort().values
    tokens = (
        output_tokens
        if resolved_target == source_count
        else output_tokens.index_select(0, indices)
    )
    kept = tuple(int(value) for value in indices.detach().cpu().tolist())
    kept_set = set(kept)
    dropped = tuple(index for index in range(source_count) if index not in kept_set)
    delta = LayoutDelta.drop(
        source_count,
        dropped,
        media_unit=media_unit,
        source_layout=source_layout,
        result_layout=result_layout,
    )
    order = tuple(int(value) for value in selection_order.detach().cpu().tolist())
    important_count = min(
        resolved_target,
        int(resolved_target * float(active.important_ratio)),
    )
    metadata = MappingProxyType(
        {
            "algorithm": "vispruner",
            "variant": "vispruner.image",
            "attention_source": "custom" if rules and "scorer" in rules else "visual_encoder_mean_received",
            "diversity": "custom" if rules and "selector" in rules else "iterative_odd_even_bipartite_cosine_removal",
            **rule_metadata(rules),
            "target_count": active.target_count if target_count is None else target_count,
            "effective_target_count": resolved_target,
            "important_ratio": float(active.important_ratio),
            "important_count": important_count,
            "diverse_count": resolved_target - important_count,
            "removal_step": active.removal_step,
            "source_count": source_count,
            "result_count": resolved_target,
            "selected_indices": kept,
            "selection_order": order,
            "token_order": "source_index",
            "compression_applied": resolved_target < source_count,
        }
    )
    return VisPrunerOutput(tokens, indices, selection_order, delta, metadata)


def _mean_received_attention(attention: Any, *, token_count: int, torch: Any) -> Any:
    if not isinstance(attention, torch.Tensor):
        raise TypeError("attention must be a torch.Tensor.")
    value = attention
    if value.ndim == 1:
        if int(value.shape[0]) != token_count:
            raise ValueError("attention token dimension must match features.")
        value = value.detach().float()
    elif value.ndim == 4:
        if int(value.shape[0]) != 1:
            raise ValueError("attention accepts only a singleton batch dimension.")
        value = value[0]
    if value.ndim == 3:
        if tuple(value.shape[-2:]) != (token_count, token_count):
            if int(value.shape[0]) != 1 or int(value.shape[-1]) != token_count:
                raise ValueError("attention token dimensions must match features.")
            value = value[0].detach().float().mean(dim=0)
        else:
            value = value.detach().float().mean(dim=0).mean(dim=-2)
    elif value.ndim == 2:
        if int(value.shape[-1]) != token_count:
            raise ValueError("attention token dimensions must match features.")
        value = value.detach().float()
        value = value.mean(dim=-2)
    else:
        if value.ndim != 1:
            raise ValueError(
                "attention must contain per-token received scores or a square "
                "attention matrix."
            )
    if not bool(torch.isfinite(value).all().item()):
        raise ValueError("attention must contain only finite values.")
    return value


def _validate_features(value: Any, name: str, *, torch: Any) -> None:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor.")
    if value.ndim != 2 or min(int(item) for item in value.shape) < 1:
        raise ValueError(f"{name} must have shape [tokens, hidden].")
    if not value.is_floating_point():
        raise TypeError(f"{name} must use a floating dtype.")
    if not bool(torch.isfinite(value).all().item()):
        raise ValueError(f"{name} must contain only finite values.")


def _require_positive_int(name: str, value: Any) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer.")
    if value < 1:
        raise ValueError(f"{name} must be at least one.")


def _torch() -> Any:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - runtime dependency
        raise RuntimeError("VisPruner execution requires PyTorch.") from exc
    return torch


__all__ = [
    "VISPRUNER_CONFIG_KEYS",
    "VisPrunerOutput",
    "VisPrunerSpec",
    "merge_vispruner_executor_config",
    "select_vispruner_indices",
    "vispruner_image_tokens",
]
