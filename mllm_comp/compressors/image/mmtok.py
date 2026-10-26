"""Backbone-neutral MMTok multimodal-coverage selection.

MMTok selects projected image features that jointly cover prompt embeddings
and the complete visual feature set.  This module owns that shared math only;
backbone bindings own crop/image partitioning, feature capture, prompt layout
repair, AnyRes legalization, and Qwen3-VL DeepStack synchronization.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from types import MappingProxyType
from typing import Any, Mapping

from mllm_comp.core.layout import LayoutDelta, LayoutTokenUnit
from ._rules import rule_metadata, score_rule, selection_rule


MMTOK_CONFIG_KEYS = frozenset(
    {
        "retain_ratio",
        "target_count",
        "alpha",
        "text_temperature",
        "vision_temperature",
        "min_visual_tokens",
    }
)


def merge_mmtok_executor_config(
    *,
    stage: Any,
    binding_config: Any,
    extra_keys: frozenset[str] = frozenset(),
) -> dict[str, Any]:
    """Merge MMTok knobs while keeping control-plane fields out of the kernel."""

    allowed = MMTOK_CONFIG_KEYS | extra_keys
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
            raise ValueError(f"Unknown MMTok {source} keys: {unknown}.")
        result.update(dict(value))
    if "retain_ratio" in result and "target_count" in result:
        raise ValueError(
            "MMTok accepts only one explicit shared budget: retain_ratio or "
            "target_count."
        )
    return result


@dataclass(frozen=True)
class MMTokSpec:
    """Paper defaults plus one dynamic-resolution output budget."""

    retain_ratio: float = 0.2
    target_count: int | None = None
    alpha: float = 0.5
    text_temperature: float = 0.02
    vision_temperature: float = 0.2
    min_visual_tokens: int = 1

    def __post_init__(self) -> None:
        for name in (
            "retain_ratio",
            "alpha",
            "text_temperature",
            "vision_temperature",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(f"{name} must be a real number.")
            if not math.isfinite(float(value)):
                raise ValueError(f"{name} must be finite.")
        if not 0.0 < float(self.retain_ratio) <= 1.0:
            raise ValueError("retain_ratio must be in (0, 1].")
        if float(self.alpha) < 0.0:
            raise ValueError("alpha must be non-negative.")
        if float(self.text_temperature) <= 0.0:
            raise ValueError("text_temperature must be positive.")
        if float(self.vision_temperature) <= 0.0:
            raise ValueError("vision_temperature must be positive.")
        if isinstance(self.min_visual_tokens, bool) or not isinstance(
            self.min_visual_tokens, int
        ):
            raise TypeError("min_visual_tokens must be an integer.")
        if self.min_visual_tokens < 1:
            raise ValueError("min_visual_tokens must be at least one.")
        if self.target_count is not None and (
            isinstance(self.target_count, bool)
            or not isinstance(self.target_count, int)
        ):
            raise TypeError("target_count must be an integer or None.")
        if self.target_count is not None and self.target_count < self.min_visual_tokens:
            raise ValueError("target_count must be at least min_visual_tokens.")

    def resolve_target_count(
        self,
        source_count: int,
        *,
        target_count: int | None = None,
    ) -> int:
        _require_positive_int("source_count", source_count)
        requested = self.target_count if target_count is None else target_count
        if requested is not None:
            _require_positive_int("target_count", requested)
        else:
            requested = int(float(self.retain_ratio) * source_count)
        return min(source_count, max(self.min_visual_tokens, requested))


@dataclass(frozen=True)
class MMTokOutput:
    """Source-ordered selected projected features and their exact mapping."""

    tokens: Any
    indices: Any
    selection_order: Any
    layout_delta: LayoutDelta
    metadata: Mapping[str, Any]

    @property
    def selected_indices(self) -> Any:
        return self.indices


def greedy_mmtok_coverage_indices(coverage: Any, target_count: int) -> Any:
    """Greedily maximize marginal row coverage in algorithm selection order."""

    torch = _torch()
    if not isinstance(coverage, torch.Tensor):
        raise TypeError("coverage must be a torch.Tensor.")
    if coverage.ndim != 2 or min(int(value) for value in coverage.shape) < 1:
        raise ValueError("coverage must have shape [targets, image_tokens].")
    if not coverage.is_floating_point():
        raise TypeError("coverage must use a floating dtype.")
    _require_positive_int("target_count", target_count)
    source_count = int(coverage.shape[1])
    if target_count > source_count:
        raise ValueError("target_count cannot exceed the image token count.")
    work = coverage.float()
    if not bool(torch.isfinite(work).all().item()):
        raise ValueError("coverage must contain only finite values.")
    best = torch.zeros(
        int(work.shape[0]), dtype=work.dtype, device=work.device
    )
    blocked = torch.zeros(source_count, dtype=work.dtype, device=work.device)
    selected = torch.empty(target_count, dtype=torch.long, device=work.device)
    for step in range(target_count):
        gain = (work - best.unsqueeze(1)).clamp_min_(0).sum(dim=0)
        gain.add_(blocked)
        index = torch.argmax(gain)
        selected[step] = index
        best = torch.maximum(best, work[:, index])
        blocked[index] = float("-inf")
    return selected


def mmtok_image_tokens(
    projected_tokens: Any,
    vision_tokens: Any,
    text_embeddings: Any,
    *,
    spec: MMTokSpec | None = None,
    target_count: int | None = None,
    media_unit: str | LayoutTokenUnit = "image_feature",
    source_layout: Mapping[str, Any] | None = None,
    result_layout: Mapping[str, Any] | None = None,
    rules: Mapping[str, Any] | None = None,
) -> MMTokOutput:
    """Select one image sequence using released multimodal coverage math."""

    torch = _torch()
    _validate_features(
        projected_tokens,
        vision_tokens,
        text_embeddings,
        torch=torch,
    )
    active = spec or MMTokSpec()
    source_count = int(projected_tokens.shape[0])
    resolved_target = active.resolve_target_count(
        source_count, target_count=target_count
    )

    if resolved_target == source_count:
        selection_order = torch.arange(source_count, device=projected_tokens.device)
    else:
        # Match the release: normalize in the input dtype, then compute FP32
        # coverage. Early promotion changed many BF16 greedy selections.
        def released_normalize(value):
            return (value / value.norm(dim=-1, keepdim=True).clamp(min=1e-8)).float()
        projected = released_normalize(projected_tokens)
        vision = released_normalize(vision_tokens)
        text = released_normalize(text_embeddings)
        text_vision = torch.nn.functional.softmax(
            (text @ projected.transpose(0, 1)) / float(active.text_temperature),
            dim=1,
        ) / float(text.shape[0])
        vision_vision = torch.nn.functional.softmax(
            (vision @ vision.transpose(0, 1)) / float(active.vision_temperature),
            dim=1,
        ) / float(source_count)
        combined = torch.cat(
            (text_vision, vision_vision * float(active.alpha)), dim=0
        )
        request = {"projected_tokens": projected_tokens, "vision_tokens": vision_tokens,
                   "text_embeddings": text_embeddings, "keep_count": resolved_target, "spec": active}
        combined = score_rule(rules, "scorer", request, combined)
        selection_order = selection_rule(rules, "selector", {**request, "scores": combined},
            count=source_count, keep_count=resolved_target, device=projected_tokens.device)
        if selection_order is None:
            selection_order = greedy_mmtok_coverage_indices(combined, resolved_target)
    indices = selection_order.sort().values
    tokens = (
        projected_tokens
        if resolved_target == source_count
        else projected_tokens.index_select(0, indices)
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
    order = tuple(
        int(value) for value in selection_order.detach().cpu().tolist()
    )
    metadata = MappingProxyType(
        {
            "algorithm": "mmtok",
            "variant": "mmtok.image",
            "coverage_objective": "custom" if rules and "scorer" in rules else "text_vision_plus_alpha_vision_vision",
            **rule_metadata(rules),
            "coverage_compute_dtype": "float32",
            "normalization_policy": "official_input_dtype_then_float32",
            "retain_ratio": float(active.retain_ratio),
            "requested_target_count": (
                active.target_count if target_count is None else target_count
            ),
            "effective_target_count": resolved_target,
            "alpha": float(active.alpha),
            "text_temperature": float(active.text_temperature),
            "vision_temperature": float(active.vision_temperature),
            "text_token_count": int(text_embeddings.shape[0]),
            "source_count": source_count,
            "result_count": resolved_target,
            "selected_indices": kept,
            "selection_order": order,
            "token_order": "source_index",
            "compression_applied": resolved_target < source_count,
        }
    )
    return MMTokOutput(tokens, indices, selection_order, delta, metadata)


def _validate_features(
    projected: Any,
    vision: Any,
    text: Any,
    *,
    torch: Any,
) -> None:
    for name, value in (
        ("projected_tokens", projected),
        ("vision_tokens", vision),
        ("text_embeddings", text),
    ):
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"{name} must be a torch.Tensor.")
        if value.ndim != 2 or min(int(item) for item in value.shape) < 1:
            raise ValueError(f"{name} must have shape [tokens, hidden].")
        if not value.is_floating_point():
            raise TypeError(f"{name} must use a floating dtype.")
        if not bool(torch.isfinite(value).all().item()):
            raise ValueError(f"{name} must contain only finite values.")
    if int(projected.shape[0]) != int(vision.shape[0]):
        raise ValueError("projected and visual token counts must match.")
    if int(projected.shape[1]) != int(text.shape[1]):
        raise ValueError("projected image and text embedding dimensions must match.")
    devices = {projected.device, vision.device, text.device}
    if len(devices) != 1:
        raise ValueError("MMTok inputs must be on the same device.")


def _require_positive_int(name: str, value: Any) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer.")
    if value < 1:
        raise ValueError(f"{name} must be at least one.")


def _torch() -> Any:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - runtime dependency
        raise RuntimeError("MMTok execution requires PyTorch.") from exc
    return torch


__all__ = [
    "MMTOK_CONFIG_KEYS",
    "MMTokOutput",
    "MMTokSpec",
    "greedy_mmtok_coverage_indices",
    "merge_mmtok_executor_config",
    "mmtok_image_tokens",
]
