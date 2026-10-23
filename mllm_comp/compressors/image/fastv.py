"""Backbone-neutral FastV decoder token selection.

FastV keeps every text token and ranks image tokens once before a configured
decoder layer.  Providers own the exact visual positions, RoPE projections,
causal visibility, sparse position legalization, and DynamicCache repair.
This module owns the released ranking; shared selection materialization builds
the differentiable gather and source-ordered drop mapping.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Iterable, Mapping

from .token_selection import ImageTokenSelection, ImageTokenSelectionResult
from .pyramid_drop import pyramid_attention_scores, pyramid_instruction_indices
from .decoder_config import require_finite_real


@dataclass(frozen=True)
class FastVSpec:
    """Released defaults for the single FastV decoder boundary."""

    prune_layer: int = 2
    prune_ratio: float = 0.5
    score_source: str = "last_token"
    min_visual_tokens: int = 1
    random_seed: int = 0

    def __post_init__(self) -> None:
        for name in ("prune_layer", "min_visual_tokens", "random_seed"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be an integer.")
        if self.prune_layer < 0:
            raise ValueError("prune_layer must be non-negative.")
        prune_ratio = require_finite_real(self.prune_ratio, "prune_ratio")
        if not 0.0 <= prune_ratio < 1.0:
            raise ValueError("prune_ratio must be in [0, 1).")
        if self.min_visual_tokens < 1:
            raise ValueError("min_visual_tokens must be at least one.")
        if self.score_source not in {"last_token", "mean_query"}:
            raise ValueError("score_source must be 'last_token' or 'mean_query'.")

    def resolve_visual_count(self, source_visual_tokens: int) -> int:
        if (
            isinstance(source_visual_tokens, bool)
            or not isinstance(source_visual_tokens, int)
            or source_visual_tokens < 1
        ):
            raise ValueError("source_visual_tokens must be a positive integer.")
        requested = round(source_visual_tokens * (1.0 - float(self.prune_ratio)))
        return min(
            source_visual_tokens,
            max(self.min_visual_tokens, int(requested)),
        )


@dataclass(frozen=True)
class FastVResult(ImageTokenSelectionResult):
    """One FastV selection and its complete serialized-sequence mapping."""


def fastv_scorer_query_indices(
    image_mask: Any,
    *,
    score_source: str,
    attention_mask: Any | None = None,
    labels: Any | None = None,
) -> Any:
    """Resolve released prompt queries without consuming teacher-forced answers."""

    torch = _torch()
    if score_source not in {"last_token", "mean_query"}:
        raise ValueError("score_source must be 'last_token' or 'mean_query'.")
    boundaries = pyramid_instruction_indices(
        image_mask,
        attention_mask=attention_mask,
        labels=labels,
    )
    prompt_end = int(boundaries[0].item())
    if score_source == "last_token":
        return boundaries[:1]
    indices = torch.arange(prompt_end + 1, device=boundaries.device)
    if attention_mask is None:
        return indices
    mask = attention_mask[0] if attention_mask.ndim == 2 else attention_mask
    return indices[mask.index_select(0, indices).bool()]


def fastv_image_tokens(
    hidden_states: Any,
    *,
    image_indices: Any,
    scorer_query: Any | None,
    attention_keys: Any | None,
    attention_key_mask: Any | None = None,
    spec: FastVSpec | None = None,
    protected_indices: Iterable[int] = (),
    media_unit: str = "llm_token",
    source_layout: Mapping[str, Any] | None = None,
    result_layout: Mapping[str, Any] | None = None,
    scorer: Callable[..., Any] | None = None,
    selector: Callable[..., Any] | None = None,
) -> FastVResult:
    """Apply the released one-shot FastV image-token ranking.

    ``scorer_query`` contains RoPE-applied query vectors from the final shallow
    layer and ``attention_keys`` contains RoPE-applied keys for the complete
    serialized prompt.  Providers pass one query for ``last_token`` and all
    eligible prefix queries for ``mean_query``.  At ``prune_layer=0`` both are
    ``None`` and selection follows the released seeded-random branch.

    Optional stateless rules receive read-only mappings of the actual values,
    fixed-source Q/K, visual indices, budget and protected visual positions.
    The selector returns visual-local indices. Rules run only for non-identity
    decisions; the platform retains source ordering and differentiable gather.
    """

    torch = _torch()
    active_spec = spec or FastVSpec()
    if not isinstance(hidden_states, torch.Tensor) or hidden_states.ndim != 2:
        raise ValueError("hidden_states must be a rank-2 torch.Tensor.")
    if not hidden_states.is_floating_point():
        raise TypeError("hidden_states must use a floating dtype.")
    sequence_length = int(hidden_states.shape[0])
    if sequence_length < 1:
        raise ValueError("FastV requires a non-empty sequence.")
    images = _index_tensor(
        image_indices,
        device=hidden_states.device,
        name="image_indices",
    )
    _validate_unique_source_order(images, sequence_length, "image_indices")
    if int(images.numel()) < 1:
        raise ValueError("FastV requires at least one image token.")
    protected = tuple(sorted(set(int(value) for value in protected_indices)))
    if any(value < 0 or value >= sequence_length for value in protected):
        raise ValueError("protected_indices contains an out-of-range index.")

    image_count = int(images.numel())
    keep_count = active_spec.resolve_visual_count(image_count)
    image_positions = {value: index for index, value in enumerate(images.tolist())}
    protected_positions = tuple(
        image_positions[value] for value in protected if value in image_positions
    )
    keep_count = min(image_count, max(keep_count, len(protected_positions)))
    request = {
        "hidden_states": hidden_states,
        "image_indices": images,
        "scorer_query": scorer_query,
        "attention_keys": attention_keys,
        "attention_key_mask": attention_key_mask,
        "keep_count": keep_count,
        "protected_image_positions": protected_positions,
        "prune_layer": active_spec.prune_layer,
        "score_source": active_spec.score_source,
        "source_layout": source_layout,
        "result_layout": result_layout,
    }

    if keep_count == image_count:
        scores = hidden_states.new_zeros((image_count,), dtype=torch.float32)
        ranked = torch.arange(image_count, device=hidden_states.device)
        ranking = "identity"
    elif scorer is not None:
        with torch.no_grad():
            scores = scorer(request)
        if (not isinstance(scores, torch.Tensor) or not scores.is_floating_point()
                or tuple(scores.shape) != (image_count,) or not bool(torch.isfinite(scores).all())):
            raise ValueError("FastV scorer must return finite floating scores with shape [image_count].")
        scores = scores.detach().to(device=hidden_states.device)
        ranked = _stable_descending_indices(scores)
        ranking = "custom_scorer"
    elif active_spec.prune_layer == 0:
        if scorer_query is not None or attention_keys is not None:
            raise ValueError("FastV layer-zero selection must not receive attention Q/K.")
        generator = torch.Generator(device="cpu")
        generator.manual_seed(active_spec.random_seed)
        ranked = torch.randperm(image_count, generator=generator).to(
            device=hidden_states.device
        )
        scores = hidden_states.new_zeros((image_count,), dtype=torch.float32)
        ranking = "seeded_random"
    else:
        if scorer_query is None or attention_keys is None:
            raise ValueError("FastV attention ranking requires scorer_query and keys.")
        query = scorer_query
        if not isinstance(query, torch.Tensor):
            raise TypeError("scorer_query must be a torch.Tensor.")
        if not isinstance(attention_keys, torch.Tensor):
            raise TypeError("attention_keys must be a torch.Tensor.")
        if query.ndim == 2:
            query_count = 1
        elif query.ndim == 3:
            query_count = int(query.shape[0])
        else:
            raise ValueError("scorer_query must have shape [H,D] or [Q,H,D].")
        if active_spec.score_source == "last_token" and query_count != 1:
            raise ValueError("last_token FastV requires exactly one scorer query.")
        image_keys = attention_keys.index_select(
            0, images.to(device=attention_keys.device)
        )
        scores = pyramid_attention_scores(
            instruction_query=query,
            image_keys=image_keys,
            image_count=image_count,
            device=hidden_states.device,
            attention_keys=attention_keys,
            attention_key_mask=attention_key_mask,
            image_key_indices=images,
        )
        ranked = _stable_descending_indices(scores)
        ranking = active_spec.score_source

    if selector is not None and keep_count != image_count:
        with torch.no_grad():
            selected = selector({**request, "scores": scores})
        selected = torch.as_tensor(selected, device=hidden_states.device)
        if selected.ndim != 1 or selected.dtype not in {
            torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8,
        }:
            raise ValueError("FastV selector must return one-dimensional integer visual-local indices.")
        selected_positions = selected.tolist()
        if (len(selected_positions) != keep_count or len(set(selected_positions)) != keep_count
                or any(i < 0 or i >= image_count for i in selected_positions)
                or not set(protected_positions).issubset(selected_positions)):
            raise ValueError("FastV selector must return exactly keep_count unique in-range indices including protected image positions.")
        ranking = "custom_scorer_custom_selector" if scorer is not None else "custom_selector"
    else:
        protected_set = set(protected_positions)
        selected_positions = list(protected_positions)
        selected_positions.extend(
            value
            for value in ranked.tolist()
            if value not in protected_set
        )
        selected_positions = selected_positions[:keep_count]
    selected_positions.sort()
    selected_position_tensor = torch.tensor(
        selected_positions,
        dtype=torch.long,
        device=hidden_states.device,
    )
    kept_images = images.index_select(0, selected_position_tensor)
    decision = ImageTokenSelection(
        kept_image_indices=kept_images,
        image_scores=scores,
        metadata={
            "algorithm": "fastv",
            "variant": "fastv.image",
            "ranking": ranking,
            "prune_layer": active_spec.prune_layer,
            "prune_ratio": float(active_spec.prune_ratio),
            "score_source": "custom" if scorer is not None and keep_count != image_count else active_spec.score_source,
            **({"source_query_policy": active_spec.score_source} if scorer is not None and keep_count != image_count else {}),
        },
    )
    return FastVResult.from_selection(decision, hidden_states, image_indices=images,
        protected_indices=protected, media_unit=media_unit,
        source_layout=source_layout, result_layout=result_layout)


def _stable_descending_indices(scores: Any) -> Any:
    torch = _torch()
    return torch.argsort(scores, descending=True, stable=True)


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
        raise RuntimeError("FastV execution requires PyTorch.") from exc
    return torch


__all__ = [
    "FastVResult",
    "FastVSpec",
    "fastv_image_tokens",
    "fastv_scorer_query_indices",
]
