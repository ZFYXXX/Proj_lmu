"""Backbone-neutral D²Pruner debiased graph selection.

Providers reconstruct the scorer-layer last-text attention and own decoder
layout/cache repair.  This module loads the train-free positional prior and
implements only the released relative-importance, hybrid-graph, pivot/MIS
selection mathematics.
"""

from __future__ import annotations

from ._rules import rule_metadata, score_rule, selection_rule

from dataclasses import dataclass, field
from pathlib import Path
import posixpath
from typing import Any, Iterable, Mapping

from mllm_comp.core.layout import LayoutDelta

from .decoder_config import require_finite_real


@dataclass(frozen=True)
class D2PrunerSpec:
    prune_after_layer: int = 2
    retain_ratio: float = 0.25
    pivot_ratio: float = 0.7
    semantic_weight: float = 0.5
    similarity_threshold: float = 0.8
    epsilon: float = 1e-7
    bias_prior_path: str | None = None
    bias_resize_mode: str = "linear"
    spatial_layout: str = "near_square_row_major"
    score_source: str = "last_text_token"
    min_visual_tokens: int = 1

    def __post_init__(self) -> None:
        if (
            isinstance(self.prune_after_layer, bool)
            or not isinstance(self.prune_after_layer, int)
            or self.prune_after_layer < 1
        ):
            raise ValueError("prune_after_layer must be an integer of at least one.")
        retain_ratio = require_finite_real(self.retain_ratio, "retain_ratio")
        pivot_ratio = require_finite_real(self.pivot_ratio, "pivot_ratio")
        semantic_weight = require_finite_real(self.semantic_weight, "semantic_weight")
        similarity_threshold = require_finite_real(
            self.similarity_threshold, "similarity_threshold"
        )
        epsilon = require_finite_real(self.epsilon, "epsilon")
        if not 0.0 < retain_ratio <= 1.0:
            raise ValueError("retain_ratio must be in (0, 1].")
        if not 0.0 <= pivot_ratio <= 1.0:
            raise ValueError("pivot_ratio must be in [0, 1].")
        if not 0.0 <= semantic_weight <= 1.0:
            raise ValueError("semantic_weight must be in [0, 1].")
        if not 0.0 <= similarity_threshold <= 1.0:
            raise ValueError("similarity_threshold must be in [0, 1].")
        if epsilon <= 0.0:
            raise ValueError("epsilon must be positive.")
        if self.bias_prior_path is not None and (
            not isinstance(self.bias_prior_path, str)
            or not self.bias_prior_path.strip()
        ):
            raise ValueError("bias_prior_path must be None or a non-empty path.")
        if self.bias_resize_mode != "linear":
            raise ValueError("bias_resize_mode must be 'linear'.")
        if self.spatial_layout != "near_square_row_major":
            raise ValueError("spatial_layout must be 'near_square_row_major'.")
        if self.score_source != "last_text_token":
            raise ValueError("score_source must be 'last_text_token'.")
        if (
            isinstance(self.min_visual_tokens, bool)
            or not isinstance(self.min_visual_tokens, int)
            or self.min_visual_tokens < 1
        ):
            raise ValueError("min_visual_tokens must be a positive integer.")

    def resolve_visual_count(self, source_visual_tokens: int) -> int:
        if (
            isinstance(source_visual_tokens, bool)
            or not isinstance(source_visual_tokens, int)
            or source_visual_tokens < 1
        ):
            raise ValueError("source_visual_tokens must be a positive integer.")
        requested = int(source_visual_tokens * float(self.retain_ratio))
        return min(
            source_visual_tokens,
            max(self.min_visual_tokens, requested),
        )


@dataclass(frozen=True)
class D2PrunerResult:
    hidden_states: Any
    kept_indices: Any
    dropped_indices: Any
    kept_image_indices: Any
    relative_scores: Any
    adjacency_edges: int
    layout_delta: LayoutDelta
    metadata: Mapping[str, Any] = field(default_factory=dict)


def load_d2pruner_bias_prior(
    path: str | Path,
    *,
    expected_model_id: str | Path | None = None,
    expected_scorer_layer: int | None = None,
) -> tuple[Any, Mapping[str, Any]]:
    """Load and validate one calibrated positional-attention prior on CPU."""

    torch = _torch()
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(
            "D²Pruner requires its calibrated positional-bias prior at "
            f"{source}. Run the pinned fuxian calibration workflow or provide "
            "bias_prior_path explicitly."
        )
    payload = torch.load(source, map_location="cpu", weights_only=True)
    metadata: dict[str, Any] = {}
    if isinstance(payload, Mapping):
        if "bias" not in payload:
            raise ValueError("D²Pruner prior mapping must contain a 'bias' tensor.")
        prior = payload["bias"]
        metadata = {key: value for key, value in payload.items() if key != "bias"}
    else:
        prior = payload
    if (
        not isinstance(prior, torch.Tensor)
        or prior.ndim != 1
        or int(prior.numel()) < 2
    ):
        raise ValueError("D²Pruner prior must be a rank-one tensor with at least two bins.")
    if not bool(torch.isfinite(prior).all().item()) or bool((prior < 0).any().item()):
        raise ValueError("D²Pruner prior must contain finite non-negative values.")
    if expected_model_id is not None:
        expected_identity = _checkpoint_identity(expected_model_id)
        calibrated_model = metadata.get("model_id", metadata.get("model_path"))
        if calibrated_model is None:
            raise ValueError(
                "D²Pruner prior metadata must identify its calibrated model."
            )
        calibrated_identity = _checkpoint_identity(calibrated_model)
        if calibrated_identity != expected_identity:
            raise ValueError(
                "D²Pruner prior checkpoint mismatch: calibrated for "
                f"{calibrated_identity!r}, executing {expected_identity!r}."
            )
        metadata["validated_model_identity"] = expected_identity
    if expected_scorer_layer is not None:
        calibrated_layer = metadata.get("scorer_layer")
        if (
            isinstance(calibrated_layer, bool)
            or not isinstance(calibrated_layer, int)
            or calibrated_layer != expected_scorer_layer
        ):
            raise ValueError(
                "D²Pruner prior scorer-layer mismatch: calibrated for "
                f"{calibrated_layer!r}, expected {expected_scorer_layer}."
            )
        metadata["validated_scorer_layer"] = expected_scorer_layer
    return prior.float().contiguous(), metadata


def _checkpoint_identity(value: str | Path) -> str:
    text = str(value).strip().rstrip("/\\").replace("\\", "/")
    if not text:
        raise ValueError("D²Pruner expected model identity must be non-empty.")
    identity = posixpath.normpath(text)
    if identity in {"", "."}:
        raise ValueError("D²Pruner expected model identity must be non-empty.")
    return identity


def d2pruner_image_tokens(
    hidden_states: Any,
    *,
    image_indices: Any,
    image_attention_scores: Any,
    bias_prior: Any,
    bias_metadata: Mapping[str, Any] | None = None,
    spec: D2PrunerSpec | None = None,
    protected_indices: Iterable[int] = (),
    media_unit: str = "llm_token",
    source_layout: Mapping[str, Any] | None = None,
    result_layout: Mapping[str, Any] | None = None,
    rules: Mapping[str, Any] | None = None,
) -> D2PrunerResult:
    """Apply D²Pruner to one serialized image prompt."""

    torch = _torch()
    active_spec = spec or D2PrunerSpec()
    if not isinstance(hidden_states, torch.Tensor) or hidden_states.ndim != 2:
        raise ValueError("hidden_states must be a rank-2 torch.Tensor.")
    if not hidden_states.is_floating_point():
        raise TypeError("hidden_states must use a floating dtype.")
    sequence_length = int(hidden_states.shape[0])
    images = _index_tensor(image_indices, device=hidden_states.device, name="image_indices")
    _validate_unique_source_order(images, sequence_length, "image_indices")
    image_count = int(images.numel())
    if image_count < 1:
        raise ValueError("D²Pruner requires at least one image token.")
    scores = _score_tensor(
        image_attention_scores,
        count=image_count,
        device=hidden_states.device,
    )
    prior = _prior_tensor(bias_prior)
    resized_prior = _resize_bias_prior(
        prior,
        image_count,
        device=hidden_states.device,
    )
    relative_scores = scores.float() / (resized_prior + float(active_spec.epsilon))
    adjacency = _hybrid_adjacency(
        hidden_states.index_select(0, images),
        semantic_weight=float(active_spec.semantic_weight),
        similarity_threshold=float(active_spec.similarity_threshold),
    )

    protected = tuple(sorted(set(int(value) for value in protected_indices)))
    if any(value < 0 or value >= sequence_length for value in protected):
        raise ValueError("protected_indices contains an out-of-range index.")
    image_position = {value: index for index, value in enumerate(images.tolist())}
    protected_positions = tuple(
        image_position[value] for value in protected if value in image_position
    )
    target = min(
        image_count,
        max(active_spec.resolve_visual_count(image_count), len(protected_positions)),
    )
    request = {"hidden_states": hidden_states, "image_indices": images,
        "image_attention_scores": scores, "bias_prior": resized_prior, "adjacency": adjacency,
        "keep_count": target, "protected_indices": protected_positions, "spec": active_spec}
    if target < image_count:
        relative_scores = score_rule(rules, "scorer", request, relative_scores)
    selected_positions = _select_with_pivots(
        relative_scores,
        adjacency,
        target_count=target,
        pivot_ratio=float(active_spec.pivot_ratio),
        protected_positions=protected_positions,
    )
    if target < image_count:
        custom = selection_rule(rules, "selector", {**request, "scores": relative_scores},
            count=image_count, keep_count=target, device=hidden_states.device, protected=protected_positions)
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
        device=hidden_states.device,
    )
    dropped_tensor = torch.tensor(
        dropped,
        dtype=torch.long,
        device=hidden_states.device,
    )
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
    output = hidden_states.index_select(0, kept)
    prior_info = dict(bias_metadata or {})
    return D2PrunerResult(
        hidden_states=output,
        kept_indices=kept,
        dropped_indices=dropped_tensor,
        kept_image_indices=kept_images,
        relative_scores=relative_scores,
        adjacency_edges=int(adjacency.sum().item()),
        layout_delta=delta,
        metadata={
            "algorithm": "d2pruner",
            "variant": "d2pruner.image",
            **rule_metadata(rules),
            "prune_after_layer": active_spec.prune_after_layer,
            "retain_ratio": float(active_spec.retain_ratio),
            "pivot_ratio": float(active_spec.pivot_ratio),
            "source_tokens": sequence_length,
            "result_tokens": int(output.shape[0]),
            "source_image_tokens": image_count,
            "result_image_tokens": int(kept_images.numel()),
            "adjacency_edges": int(adjacency.sum().item()),
            "bias_prior_bins": int(prior.numel()),
            "bias_calibration_samples": int(prior_info.get("sample_count", 0)),
            "selected_indices": tuple(int(value) for value in kept.tolist()),
            "compression_applied": bool(dropped),
        },
    )


def _resize_bias_prior(prior: Any, count: int, *, device: Any) -> Any:
    functional = _functional()
    if int(prior.numel()) == count:
        return prior.to(device=device, dtype=_torch().float32)
    return functional.interpolate(
        prior.float().view(1, 1, -1),
        size=count,
        mode="linear",
        align_corners=False,
    ).view(-1).to(device=device)


def _hybrid_adjacency(
    visual_hidden: Any,
    *,
    semantic_weight: float,
    similarity_threshold: float,
) -> Any:
    torch = _torch()
    functional = _functional()
    normalized = functional.normalize(visual_hidden.float(), dim=-1, eps=1e-8)
    semantic = normalized @ normalized.transpose(0, 1)
    semantic = (semantic - semantic.amin()) / (
        semantic.amax() - semantic.amin() + 1e-7
    )
    count = int(visual_hidden.shape[0])
    width = max(1, int(round(float(count) ** 0.5)))
    indices = torch.arange(count, device=visual_hidden.device)
    rows = torch.div(indices, width, rounding_mode="floor")
    columns = indices.remainder(width)
    spatial = (
        ((rows[:, None] - rows[None, :]).abs() <= 1)
        & ((columns[:, None] - columns[None, :]).abs() <= 1)
    ).to(dtype=semantic.dtype)
    spatial.fill_diagonal_(False)
    fused = semantic_weight * semantic + (1.0 - semantic_weight) * spatial
    adjacency = fused > similarity_threshold
    adjacency.fill_diagonal_(False)
    return adjacency


def _select_with_pivots(
    relative_scores: Any,
    adjacency: Any,
    *,
    target_count: int,
    pivot_ratio: float,
    protected_positions: tuple[int, ...],
) -> Any:
    torch = _torch()
    count = int(relative_scores.numel())
    target = min(count, max(1, int(target_count)))
    selected = torch.zeros(count, dtype=torch.bool, device=relative_scores.device)
    if protected_positions:
        selected[
            torch.tensor(
                protected_positions,
                dtype=torch.long,
                device=relative_scores.device,
            )
        ] = True
    remaining_budget = target - int(selected.sum().item())
    if remaining_budget > 0:
        pivot_count = min(
            remaining_budget,
            max(1, int(target * pivot_ratio)),
        )
        eligible_scores = relative_scores.float().masked_fill(selected, float("-inf"))
        pivots = _stable_descending_indices(eligible_scores)[:pivot_count]
        selected[pivots] = True

    candidates = relative_scores.float().clone()
    candidates[selected] = float("-inf")
    selected_indices = selected.nonzero(as_tuple=False).flatten()
    if int(selected_indices.numel()):
        candidates[adjacency.index_select(0, selected_indices).any(dim=0)] = float(
            "-inf"
        )
    while int(selected.sum().item()) < target:
        ranked = _stable_descending_indices(candidates)
        index = ranked[0]
        if not bool(torch.isfinite(candidates[index]).item()):
            break
        selected[index] = True
        candidates[index] = float("-inf")
        candidates[adjacency[index]] = float("-inf")
    if int(selected.sum().item()) < target:
        fallback = relative_scores.float().masked_fill(selected, float("-inf"))
        needed = target - int(selected.sum().item())
        selected[_stable_descending_indices(fallback)[:needed]] = True
    result = selected.nonzero(as_tuple=False).flatten()
    if int(result.numel()) != target:
        raise RuntimeError("D²Pruner failed to produce its exact visual budget.")
    return result


def _stable_descending_indices(values: Any) -> Any:
    torch = _torch()
    return torch.argsort(values, descending=True, stable=True)


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


def _score_tensor(value: Any, *, count: int, device: Any) -> Any:
    torch = _torch()
    if not isinstance(value, torch.Tensor):
        raise TypeError("image_attention_scores must be a torch.Tensor.")
    result = value.to(device=device)
    if result.ndim != 1 or int(result.numel()) != count:
        raise ValueError("image_attention_scores must contain one value per image token.")
    if not bool(torch.isfinite(result).all().item()) or bool((result < 0).any().item()):
        raise ValueError("image_attention_scores must be finite and non-negative.")
    return result


def _prior_tensor(value: Any) -> Any:
    torch = _torch()
    if (
        not isinstance(value, torch.Tensor)
        or value.ndim != 1
        or int(value.numel()) < 2
    ):
        raise ValueError("bias_prior must be a rank-one tensor with at least two bins.")
    if not bool(torch.isfinite(value).all().item()) or bool((value < 0).any().item()):
        raise ValueError("bias_prior must contain finite non-negative values.")
    return value.float()


def _torch() -> Any:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - optional local dependency
        raise RuntimeError("D²Pruner execution requires PyTorch.") from exc
    return torch


def _functional() -> Any:
    try:
        from torch.nn import functional
    except ImportError as exc:  # pragma: no cover - optional local dependency
        raise RuntimeError("D²Pruner execution requires PyTorch.") from exc
    return functional


__all__ = [
    "D2PrunerResult",
    "D2PrunerSpec",
    "d2pruner_image_tokens",
    "load_d2pruner_bias_prior",
]
