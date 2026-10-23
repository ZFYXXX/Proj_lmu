"""Backbone-neutral DART duplication-aware image-token selection.

DART chooses a small set of image and post-image text pivots from rotary key
states, then lets each pivot claim the least-similar remaining image tokens in
the decoder-normalized hidden space.  Providers own key reconstruction,
decoder normalization, prompt spans, positions, masks, and cache repair.
"""

from __future__ import annotations

from ._rules import rule_metadata, score_rule, selection_rule

from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping

from mllm_comp.core.layout import LayoutDelta

from .decoder_config import require_finite_real


@dataclass(frozen=True)
class DARTSpec:
    """Train-free DART defaults from the pinned fuxian reproduction."""

    prune_after_layer: int = 2
    retain_ratio: float = 1.0 / 9.0
    target_visual_tokens: int | None = None
    pivot_image_tokens: int = 4
    pivot_text_tokens: int = 4
    pivot_source: str = "max_key_l1"
    similarity_source: str = "normalized_hidden"
    allocation: str = "paper_exact"
    position_mode: str = "preserve"
    min_visual_tokens: int = 1

    def __post_init__(self) -> None:
        for name in (
            "prune_after_layer",
            "pivot_image_tokens",
            "pivot_text_tokens",
            "min_visual_tokens",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be an integer.")
        if self.prune_after_layer < 1:
            raise ValueError("prune_after_layer must be at least one.")
        retain_ratio = require_finite_real(self.retain_ratio, "retain_ratio")
        if not 0.0 < retain_ratio <= 1.0:
            raise ValueError("retain_ratio must be in (0, 1].")
        if self.target_visual_tokens is not None and (
            isinstance(self.target_visual_tokens, bool)
            or not isinstance(self.target_visual_tokens, int)
            or self.target_visual_tokens < 1
        ):
            raise ValueError("target_visual_tokens must be None or a positive integer.")
        if self.pivot_image_tokens < 1:
            raise ValueError("pivot_image_tokens must be at least one.")
        if self.pivot_text_tokens < 0:
            raise ValueError("pivot_text_tokens must be non-negative.")
        if self.min_visual_tokens < 1:
            raise ValueError("min_visual_tokens must be at least one.")
        if self.pivot_source != "max_key_l1":
            raise ValueError("pivot_source must be 'max_key_l1'.")
        if self.similarity_source != "normalized_hidden":
            raise ValueError("similarity_source must be 'normalized_hidden'.")
        if self.allocation not in {"paper_exact", "official_code"}:
            raise ValueError("allocation must be 'paper_exact' or 'official_code'.")
        if self.position_mode != "preserve":
            raise ValueError("position_mode must be 'preserve'.")

    def resolve_visual_count(self, source_visual_tokens: int) -> int:
        if (
            isinstance(source_visual_tokens, bool)
            or not isinstance(source_visual_tokens, int)
            or source_visual_tokens < 1
        ):
            raise ValueError("source_visual_tokens must be a positive integer.")
        requested = self.target_visual_tokens
        if requested is None:
            requested = round(source_visual_tokens * float(self.retain_ratio))
        return min(
            source_visual_tokens,
            max(self.min_visual_tokens, int(requested)),
        )


@dataclass(frozen=True)
class DARTResult:
    """One DART selection and its source-ordered sequence mapping."""

    hidden_states: Any
    kept_indices: Any
    dropped_indices: Any
    kept_image_indices: Any
    image_pivot_indices: Any
    text_pivot_indices: Any
    layout_delta: LayoutDelta
    metadata: Mapping[str, Any] = field(default_factory=dict)


def dart_post_image_text_indices(
    image_mask: Any,
    *,
    attention_mask: Any | None = None,
    labels: Any | None = None,
    ignore_index: int = -100,
) -> Any:
    """Return valid post-image prompt tokens eligible to become text pivots."""

    torch = _torch()
    if not isinstance(image_mask, torch.Tensor) or image_mask.ndim != 1:
        raise ValueError("image_mask must be a rank-one torch.Tensor.")
    image_mask = image_mask.bool()
    token_count = int(image_mask.numel())
    image_indices = image_mask.nonzero(as_tuple=False).flatten()
    if int(image_indices.numel()) < 1:
        raise ValueError("DART requires at least one image token.")
    expected = torch.arange(
        int(image_indices[0].item()),
        int(image_indices[-1].item()) + 1,
        device=image_indices.device,
    )
    if not torch.equal(image_indices, expected):
        raise ValueError("DART requires one contiguous image span.")

    valid = torch.ones(token_count, dtype=torch.bool, device=image_mask.device)
    if attention_mask is not None:
        if not isinstance(attention_mask, torch.Tensor):
            raise TypeError("attention_mask must be a torch.Tensor.")
        row = attention_mask[0] if attention_mask.ndim == 2 else attention_mask
        if row.ndim != 1 or int(row.numel()) != token_count:
            raise ValueError("attention_mask does not match image_mask.")
        valid = row.to(device=image_mask.device).bool()

    prompt_end = token_count
    if labels is not None:
        if not isinstance(labels, torch.Tensor):
            raise TypeError("labels must be a torch.Tensor.")
        row = labels[0] if labels.ndim == 2 else labels
        if row.ndim != 1 or int(row.numel()) != token_count:
            raise ValueError("labels do not match image_mask.")
        supervised = row.to(image_mask.device).ne(ignore_index) & valid
        supervised_indices = supervised.nonzero(as_tuple=False).flatten()
        if int(supervised_indices.numel()):
            prompt_end = int(supervised_indices[0].item())

    start = int(image_indices[-1].item()) + 1
    if start >= prompt_end:
        return torch.empty(0, dtype=torch.long, device=image_mask.device)
    candidates = torch.arange(start, prompt_end, device=image_mask.device)
    return candidates[valid.index_select(0, candidates)]


def dart_image_tokens(
    hidden_states: Any,
    normalized_hidden_states: Any,
    key_states: Any,
    *,
    image_indices: Any,
    text_candidate_indices: Any,
    spec: DARTSpec | None = None,
    protected_indices: Iterable[int] = (),
    media_unit: str = "llm_token",
    source_layout: Mapping[str, Any] | None = None,
    result_layout: Mapping[str, Any] | None = None,
    rules: Mapping[str, Any] | None = None,
) -> DARTResult:
    """Select non-duplicate image tokens at DART's single decoder boundary."""

    torch = _torch()
    functional = _functional()
    active_spec = spec or DARTSpec()
    hidden = hidden_states
    normalized_hidden = normalized_hidden_states
    if not isinstance(hidden, torch.Tensor) or hidden.ndim != 2:
        raise ValueError("hidden_states must be a rank-2 torch.Tensor.")
    if not hidden.is_floating_point():
        raise TypeError("hidden_states must use a floating dtype.")
    if not isinstance(normalized_hidden, torch.Tensor) or normalized_hidden.ndim != 2:
        raise ValueError("normalized_hidden_states must be a rank-2 torch.Tensor.")
    if not normalized_hidden.is_floating_point():
        raise TypeError("normalized_hidden_states must use a floating dtype.")
    if not isinstance(key_states, torch.Tensor) or key_states.ndim < 2:
        raise ValueError("key_states must have a sequence axis followed by key features.")
    sequence_length = int(hidden.shape[0])
    if tuple(normalized_hidden.shape) != tuple(hidden.shape):
        raise ValueError("normalized_hidden_states must match hidden_states shape.")
    if int(key_states.shape[0]) != sequence_length:
        raise ValueError("key_states sequence axis must match normalized hidden states.")
    # Fixed scoring context can come from an earlier device than application.
    normalized_hidden = normalized_hidden.to(hidden.device)
    flattened_keys = key_states.reshape(sequence_length, -1).to(hidden.device)

    images = _index_tensor(image_indices, device=hidden.device, name="image_indices")
    text_candidates = _index_tensor(
        text_candidate_indices,
        device=hidden.device,
        name="text_candidate_indices",
    )
    _validate_unique_source_order(images, sequence_length, "image_indices")
    _validate_unique_source_order(
        text_candidates, sequence_length, "text_candidate_indices"
    )
    if int(images.numel()) < 1:
        raise ValueError("DART requires at least one image token.")
    if set(images.tolist()).intersection(text_candidates.tolist()):
        raise ValueError("DART image and text pivot candidates must be disjoint.")
    protected = tuple(sorted(set(int(value) for value in protected_indices)))
    if any(value < 0 or value >= sequence_length for value in protected):
        raise ValueError("protected_indices contains an out-of-range index.")

    image_count = int(images.numel())
    requested_target = active_spec.resolve_visual_count(image_count)
    image_position = {value: index for index, value in enumerate(images.tolist())}
    protected_image_positions = tuple(
        image_position[value] for value in protected if value in image_position
    )
    target = min(image_count, max(requested_target, len(protected_image_positions)))
    request = {"hidden_states": hidden, "normalized_hidden_states": normalized_hidden,
        "key_states": key_states, "image_indices": images, "text_candidate_indices": text_candidates,
        "keep_count": target, "protected_indices": protected_image_positions, "spec": active_spec}
    pivot_scores = None
    if target < image_count and rules and "scorer" in rules:
        pivot_scores = score_rule(rules, "scorer", request, flattened_keys.abs().sum(dim=-1))

    if target >= image_count:
        selected_positions = torch.arange(image_count, device=hidden.device)
        image_pivots = torch.empty(0, dtype=torch.long, device=hidden.device)
        text_pivots = torch.empty(0, dtype=torch.long, device=hidden.device)
    else:
        protected_position_set = set(protected_image_positions)
        eligible_positions = torch.tensor(
            [
                position
                for position in range(image_count)
                if position not in protected_position_set
            ],
            dtype=torch.long,
            device=hidden.device,
        )
        image_pivot_count = min(
            active_spec.pivot_image_tokens,
            max(0, target - len(protected_image_positions)),
            int(eligible_positions.numel()),
        )
        if image_pivot_count:
            image_key_norms = flattened_keys.index_select(0, images).abs().sum(dim=-1)
            if pivot_scores is not None:
                image_key_norms = pivot_scores.index_select(0, images)
            eligible_norms = image_key_norms.index_select(0, eligible_positions)
            ranked = _stable_descending_indices(eligible_norms)
            image_pivots = eligible_positions.index_select(
                0, ranked[:image_pivot_count]
            )
        else:
            image_pivots = torch.empty(0, dtype=torch.long, device=hidden.device)

        text_pivot_count = min(
            active_spec.pivot_text_tokens, int(text_candidates.numel())
        )
        if text_pivot_count:
            text_norms = flattened_keys.index_select(0, text_candidates).abs().sum(dim=-1)
            if pivot_scores is not None:
                text_norms = pivot_scores.index_select(0, text_candidates)
            text_ranked = _stable_descending_indices(text_norms)
            text_pivots = text_candidates.index_select(
                0, text_ranked[:text_pivot_count]
            )
        else:
            text_pivots = torch.empty(0, dtype=torch.long, device=hidden.device)

        selected_mask = torch.zeros(image_count, dtype=torch.bool, device=hidden.device)
        if protected_image_positions:
            selected_mask[
                torch.tensor(
                    protected_image_positions,
                    dtype=torch.long,
                    device=hidden.device,
                )
            ] = True
        if int(image_pivots.numel()):
            selected_mask[image_pivots] = True
        pivot_sequence_indices = torch.cat(
            (images.index_select(0, image_pivots), text_pivots)
        )
        remaining_budget = target - int(selected_mask.sum().item())
        if remaining_budget > 0 and int(pivot_sequence_indices.numel()) < 1:
            raise ValueError("DART could not form an image or text pivot.")
        if remaining_budget == 0:
            quotas = []
        elif active_spec.allocation == "paper_exact":
            quotas = _quota_per_pivot(
                remaining_budget,
                int(pivot_sequence_indices.numel()),
            )
        else:
            quotas = [
                int(requested_target // int(pivot_sequence_indices.numel()))
            ] * int(pivot_sequence_indices.numel())

        work = normalized_hidden
        eps = max(1e-12, float(torch.finfo(work.dtype).tiny))
        normalized = functional.normalize(work, dim=-1, eps=eps)
        image_features = normalized.index_select(0, images)
        for pivot, quota in zip(pivot_sequence_indices.tolist(), quotas):
            remaining = (~selected_mask).nonzero(as_tuple=False).flatten()
            count = min(int(quota), int(remaining.numel()))
            if count <= 0:
                continue
            similarities = torch.mv(
                image_features.index_select(0, remaining),
                normalized[int(pivot)],
            )
            similarities = score_rule(rules, "diversity_scorer",
                {**request, "pivot_index": pivot, "candidate_indices": remaining,
                 "normalized_features": normalized, "keep_count": count}, similarities)
            least_similar = _stable_ascending_indices(similarities)[:count]
            selected_mask[remaining.index_select(0, least_similar)] = True

        selected_positions = selected_mask.nonzero(as_tuple=False).flatten()
        if (
            active_spec.allocation == "paper_exact"
            and int(selected_positions.numel()) != target
        ):
            raise RuntimeError(
                "DART exact allocation did not produce the requested visual count."
            )

    if target < image_count:
        custom = selection_rule(rules, "selector",
            {**request, "scores": pivot_scores, "image_pivots": image_pivots,
             "text_pivots": text_pivots, "keep_count": int(selected_positions.numel())},
            count=image_count, keep_count=int(selected_positions.numel()), device=hidden.device,
            protected=protected_image_positions)
        if custom is not None:
            selected_positions = custom.sort().values
    kept_images = images.index_select(0, selected_positions)
    image_set = set(images.tolist())
    kept_image_set = set(kept_images.tolist())
    dropped = tuple(
        index
        for index in range(sequence_length)
        if index in image_set and index not in kept_image_set
    )
    dropped_set = set(dropped)
    kept = torch.tensor(
        [index for index in range(sequence_length) if index not in dropped_set],
        dtype=torch.long,
        device=hidden.device,
    )
    dropped_tensor = torch.tensor(dropped, dtype=torch.long, device=hidden.device)
    non_image = tuple(index for index in range(sequence_length) if index not in image_set)
    layout_protected = tuple(sorted(set(non_image) | set(protected)))
    delta = LayoutDelta.drop(
        sequence_length,
        dropped,
        media_unit=media_unit,
        protected_indices=layout_protected,
        source_layout=source_layout,
        result_layout=result_layout,
    )
    output = hidden.index_select(0, kept)
    return DARTResult(
        hidden_states=output,
        kept_indices=kept,
        dropped_indices=dropped_tensor,
        kept_image_indices=kept_images,
        image_pivot_indices=images.index_select(0, image_pivots),
        text_pivot_indices=text_pivots,
        layout_delta=delta,
        metadata={
            "algorithm": "dart",
            **rule_metadata(rules),
            "variant": "dart.image",
            "allocation": active_spec.allocation,
            "prune_after_layer": active_spec.prune_after_layer,
            "retain_ratio": float(active_spec.retain_ratio),
            "requested_visual_tokens": requested_target,
            "source_tokens": sequence_length,
            "result_tokens": int(output.shape[0]),
            "source_image_tokens": image_count,
            "result_image_tokens": int(kept_images.numel()),
            "image_pivot_count": int(image_pivots.numel()),
            "text_pivot_count": int(text_pivots.numel()),
            "protected_image_tokens": len(protected_image_positions),
            "selected_indices": tuple(int(value) for value in kept.tolist()),
            "compression_applied": bool(dropped),
        },
    )


def _quota_per_pivot(total: int, pivot_count: int) -> list[int]:
    if pivot_count < 1:
        return []
    quotient, remainder = divmod(max(0, int(total)), int(pivot_count))
    return [quotient + int(index < remainder) for index in range(pivot_count)]


def _stable_descending_indices(values: Any) -> Any:
    torch = _torch()
    return torch.argsort(values, descending=True, stable=True)


def _stable_ascending_indices(values: Any) -> Any:
    torch = _torch()
    return torch.argsort(values, stable=True)


def _index_tensor(value: Any, *, device: Any, name: str) -> Any:
    torch = _torch()
    if isinstance(value, torch.Tensor):
        result = value.to(device=device, dtype=torch.long)
    else:
        result = torch.tensor(tuple(value), dtype=torch.long, device=device)
    if result.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional.")
    return result


def _validate_unique_source_order(value: Any, upper: int, name: str) -> None:
    items = value.tolist()
    if items != sorted(set(items)):
        raise ValueError(f"{name} must be unique and in source order.")
    if any(index < 0 or index >= upper for index in items):
        raise ValueError(f"{name} contains an index outside the sequence.")


def _torch() -> Any:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - optional local dependency
        raise RuntimeError("DART execution requires PyTorch.") from exc
    return torch


def _functional() -> Any:
    try:
        from torch.nn import functional
    except ImportError as exc:  # pragma: no cover - optional local dependency
        raise RuntimeError("DART execution requires PyTorch.") from exc
    return functional


__all__ = [
    "DARTResult",
    "DARTSpec",
    "dart_image_tokens",
    "dart_post_image_text_indices",
]
