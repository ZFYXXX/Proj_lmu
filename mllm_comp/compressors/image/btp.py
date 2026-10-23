"""Backbone-neutral Balanced Token Pruning for image decoder tokens.

The released Qwen2.5-VL recipe prunes immediately before decoder layers
4/7/16/23.  The first two stages seed cosine max-min diversity with
last-text-query attention, the third rebalances an over-selected attention
pool across the two source halves, and the final stage removes the remaining
visual span.  Providers own attention extraction, masks, positions, cache
repair, and backbone-specific loop integration.
"""

from __future__ import annotations

from ._rules import rule_metadata, score_rule, selection_rule

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from mllm_comp.core.layout import LayoutDelta

from .decoder_config import require_finite_real


@dataclass(frozen=True)
class BalancedTokenPruningSpec:
    """Published Qwen2.5-VL BTP schedule expressed as native configuration."""

    prune_before_layers: tuple[int, ...] = (4, 7, 16, 23)
    retain_ratios: tuple[float, ...] = (0.5, 0.5, 0.5, 0.0)
    attention_fractions: tuple[float, ...] = (0.1, 0.4, 1.0, 0.0)
    attention_oversample: tuple[float, ...] = (1.2, 1.0, 1.2, 1.0)
    rebalance_stages: tuple[bool, ...] = (False, False, True, False)
    diversity_metric: str = "cosine"
    diversity_initialization: str = "released_semantic_nearest_neighbor"
    score_query: str = "last_text_token"
    position_policy: str = "preserve_original"
    release_variant: str = "qwen2_5_vl"
    single_image_only: bool = True

    def __post_init__(self) -> None:
        layers = _integer_tuple(
            self.prune_before_layers,
            "prune_before_layers",
            minimum=2,
        )
        if not layers or layers != tuple(sorted(set(layers))):
            raise ValueError(
                "prune_before_layers must be non-empty, sorted, and unique."
            )
        stage_count = len(layers)
        ratios = _real_tuple(self.retain_ratios, "retain_ratios")
        fractions = _real_tuple(
            self.attention_fractions,
            "attention_fractions",
        )
        oversample = _real_tuple(
            self.attention_oversample,
            "attention_oversample",
        )
        rebalance = _bool_tuple(self.rebalance_stages, "rebalance_stages")
        for name, values in (
            ("retain_ratios", ratios),
            ("attention_fractions", fractions),
            ("attention_oversample", oversample),
            ("rebalance_stages", rebalance),
        ):
            if len(values) != stage_count:
                raise ValueError(
                    f"{name} must contain exactly {stage_count} stage values."
                )
        if any(not 0.0 <= value <= 1.0 for value in ratios):
            raise ValueError("retain_ratios values must lie in [0, 1].")
        if any(not 0.0 <= value <= 1.0 for value in fractions):
            raise ValueError("attention_fractions values must lie in [0, 1].")
        if any(value < 1.0 for value in oversample):
            raise ValueError("attention_oversample values must be at least 1.")
        zero_stages = tuple(index for index, value in enumerate(ratios) if value == 0)
        if zero_stages != (stage_count - 1,):
            raise ValueError(
                "The released BTP variant requires exactly one zero-retention "
                "FinalWipe stage at the end."
            )
        if any(
            enabled and ratios[index] != 0.5
            for index, enabled in enumerate(rebalance)
        ):
            raise ValueError(
                "The released BTP rebalance stage requires retain_ratio=0.5."
            )
        if self.diversity_metric != "cosine":
            raise ValueError("diversity_metric must be 'cosine'.")
        if self.diversity_initialization != "released_semantic_nearest_neighbor":
            raise ValueError(
                "diversity_initialization must be "
                "'released_semantic_nearest_neighbor'."
            )
        if self.score_query != "last_text_token":
            raise ValueError("score_query must be 'last_text_token'.")
        if self.position_policy != "preserve_original":
            raise ValueError("position_policy must be 'preserve_original'.")
        if self.release_variant != "qwen2_5_vl":
            raise ValueError("release_variant must be 'qwen2_5_vl'.")
        if self.single_image_only is not True:
            raise ValueError("Balanced Token Pruning is single-image only.")
        object.__setattr__(self, "prune_before_layers", layers)
        object.__setattr__(self, "retain_ratios", ratios)
        object.__setattr__(self, "attention_fractions", fractions)
        object.__setattr__(self, "attention_oversample", oversample)
        object.__setattr__(self, "rebalance_stages", rebalance)

    @property
    def boundary_after_layer_indices(self) -> tuple[int, ...]:
        """Zero-based completed layer after which each pre-layer stage runs."""

        return tuple(layer - 2 for layer in self.prune_before_layers)

    def stage_index(self, boundary_after_layer: int) -> int:
        if isinstance(boundary_after_layer, bool) or not isinstance(
            boundary_after_layer, int
        ):
            raise TypeError("boundary_after_layer must be an integer.")
        try:
            return self.boundary_after_layer_indices.index(boundary_after_layer)
        except ValueError as exc:
            raise ValueError(
                f"decoder layer {boundary_after_layer} is not a BTP boundary."
            ) from exc

    def validate_decoder_depth(self, layer_count: int) -> None:
        if (
            isinstance(layer_count, bool)
            or not isinstance(layer_count, int)
            or layer_count < self.prune_before_layers[-1]
        ):
            raise ValueError(
                "BTP decoder depth must reach prune-before layer "
                f"{self.prune_before_layers[-1]}."
            )


@dataclass(frozen=True)
class BalancedTokenPruningResult:
    """One source-ordered BTP stage and its exact sequence layout change."""

    hidden_states: Any
    kept_indices: Any
    dropped_indices: Any
    kept_image_indices: Any
    selected_visual_indices: Any
    layout_delta: LayoutDelta
    metadata: Mapping[str, Any] = field(default_factory=dict)
    attention_scores: Any | None = None


def btp_rebalanced_attention_topk(
    scores: Any,
    *,
    target_count: int,
    oversample: float,
) -> Any:
    """Apply the released pre/post-half rebalance to an attention top-k pool."""

    torch = _torch()
    values = _attention_scores(scores)
    target = min(max(1, _integer(target_count, "target_count", minimum=1)), int(values.numel()))
    factor = require_finite_real(oversample, "oversample")
    if factor < 1.0:
        raise ValueError("oversample must be at least 1.")
    pool_count = min(
        int(values.numel()),
        max(target, int(target * factor)),
    )
    ranked = values.topk(pool_count, largest=True, sorted=True).indices
    earlier = ranked[ranked < target]
    later = ranked[ranked >= target]
    selected = torch.cat(
        (earlier[:target], later[: max(0, target - int(earlier.numel()))])
    )
    if int(selected.numel()) != target:
        raise RuntimeError("BTP rebalance could not satisfy the visual budget.")
    return selected.to(dtype=torch.long)


def btp_maxmin_diversity_fill(
    visual_hidden: Any,
    *,
    target_count: int,
    seed_indices: Sequence[int] | Any = (),
) -> Any:
    """Exact running-min form of the released cosine max-min selector."""

    torch = _torch()
    if (
        not isinstance(visual_hidden, torch.Tensor)
        or visual_hidden.ndim != 2
        or int(visual_hidden.shape[0]) < 1
        or not visual_hidden.is_floating_point()
    ):
        raise ValueError("visual_hidden must be a non-empty floating [N,D] tensor.")
    token_count = int(visual_hidden.shape[0])
    target = min(
        token_count,
        max(1, _integer(target_count, "target_count", minimum=1)),
    )
    seeds = torch.as_tensor(
        seed_indices,
        dtype=torch.long,
        device=visual_hidden.device,
    ).reshape(-1)
    ordered: list[int] = []
    seen: set[int] = set()
    for raw in seeds.detach().cpu().tolist():
        index = int(raw)
        if not 0 <= index < token_count:
            raise ValueError("BTP diversity seed lies outside the visual span.")
        if index not in seen:
            ordered.append(index)
            seen.add(index)
    ordered = ordered[:target]
    normalized = torch.nn.functional.normalize(
        visual_hidden.float(),
        p=2,
        dim=-1,
        eps=1e-8,
    )
    distance = 1.0 - normalized @ normalized.transpose(0, 1)
    distance.fill_diagonal_(float("inf"))
    selected_mask = torch.zeros(
        token_count,
        dtype=torch.bool,
        device=visual_hidden.device,
    )
    if ordered:
        selected_tensor = torch.tensor(
            ordered,
            dtype=torch.long,
            device=visual_hidden.device,
        )
        selected_mask[selected_tensor] = True
        min_distance = distance.index_select(0, selected_tensor).amin(dim=0)
    else:
        nearest = distance.amin(dim=1)
        first = int(nearest.argmax().item())
        ordered.append(first)
        selected_mask[first] = True
        min_distance = distance[first].clone()
    min_distance[selected_mask] = float("-inf")
    while len(ordered) < target:
        next_index = int(min_distance.argmax().item())
        if bool(selected_mask[next_index].item()):
            raise RuntimeError("BTP diversity selector attempted a duplicate token.")
        ordered.append(next_index)
        selected_mask[next_index] = True
        min_distance = torch.minimum(min_distance, distance[next_index])
        min_distance[selected_mask] = float("-inf")
    return torch.tensor(ordered, dtype=torch.long, device=visual_hidden.device)


def balanced_token_pruning_image_tokens(
    hidden_states: Any,
    *,
    image_indices: Any,
    boundary_after_layer: int,
    image_attention_scores: Any | None,
    spec: BalancedTokenPruningSpec | None = None,
    media_unit: str = "llm_token",
    source_layout: Mapping[str, Any] | None = None,
    result_layout: Mapping[str, Any] | None = None,
    rules: Mapping[str, Any] | None = None,
) -> BalancedTokenPruningResult:
    """Run one BTP stage after the scorer layer and before the next layer."""

    torch = _torch()
    active_spec = spec or BalancedTokenPruningSpec()
    hidden, images = _validate_inputs(hidden_states, image_indices=image_indices)
    stage_index = active_spec.stage_index(boundary_after_layer)
    visual_count = int(images.numel())
    retain_ratio = active_spec.retain_ratios[stage_index]
    target = (
        0
        if retain_ratio == 0.0
        else min(visual_count, max(1, int(visual_count * retain_ratio)))
    )
    scores = None
    attention_seed_count = 0
    if target == visual_count:
        selected_local = torch.arange(
            visual_count,
            dtype=torch.long,
            device=hidden.device,
        )
        stage_kind = "identity"
    elif target == 0:
        selected_local = torch.empty(0, dtype=torch.long, device=hidden.device)
        stage_kind = "final_wipe"
    else:
        if image_attention_scores is None:
            raise ValueError("A non-final BTP stage requires image attention scores.")
        scores = _attention_scores(image_attention_scores).to(hidden.device)
        if int(scores.numel()) != visual_count:
            raise ValueError("BTP attention scores do not match the image span.")
        request = {"hidden_states": hidden, "image_indices": images,
            "image_attention_scores": image_attention_scores, "stage_index": stage_index,
            "keep_count": target, "spec": active_spec}
        scores = score_rule(rules, "scorer", request, scores)
        if active_spec.rebalance_stages[stage_index]:
            selected_local = btp_rebalanced_attention_topk(
                scores,
                target_count=target,
                oversample=active_spec.attention_oversample[stage_index],
            )
            attention_seed_count = int(selected_local.numel())
            stage_kind = "rebalanced_attention"
        else:
            nominal = int(target * active_spec.attention_fractions[stage_index])
            seed_count = min(
                visual_count,
                max(
                    0,
                    int(nominal * active_spec.attention_oversample[stage_index]),
                ),
            )
            seeds = (
                scores.topk(seed_count, largest=True, sorted=True).indices
                if seed_count
                else torch.empty(0, dtype=torch.long, device=hidden.device)
            )
            attention_seed_count = int(seeds.numel())
            selected_local = btp_maxmin_diversity_fill(
                hidden.index_select(0, images),
                target_count=target,
                seed_indices=seeds,
            )
            stage_kind = "attention_seeded_diversity"
        custom = selection_rule(rules, "selector", {**request, "scores": scores},
            count=visual_count, keep_count=target, device=hidden.device)
        if custom is not None:
            selected_local = custom
            stage_kind = "custom_selection"
            attention_seed_count = 0
        selected_local = selected_local.sort().values
    kept_images = images.index_select(0, selected_local)
    image_set = set(int(value) for value in images.tolist())
    kept_image_set = set(int(value) for value in kept_images.tolist())
    dropped = tuple(sorted(image_set - kept_image_set))
    dropped_set = set(dropped)
    kept = torch.tensor(
        [index for index in range(int(hidden.shape[0])) if index not in dropped_set],
        dtype=torch.long,
        device=hidden.device,
    )
    non_image = tuple(
        index for index in range(int(hidden.shape[0])) if index not in image_set
    )
    delta = LayoutDelta.drop(
        int(hidden.shape[0]),
        dropped,
        media_unit=media_unit,
        protected_indices=non_image,
        source_layout=source_layout,
        result_layout=result_layout,
    )
    output = hidden.index_select(0, kept)
    metadata = {
        "algorithm": "btp",
        "variant": "btp.image",
        **rule_metadata(rules),
        "stage_index": stage_index,
        "stage_kind": stage_kind,
        "prune_before_layer": active_spec.prune_before_layers[stage_index],
        "boundary_after_layer": boundary_after_layer,
        "retain_ratio": retain_ratio,
        "attention_fraction": active_spec.attention_fractions[stage_index],
        "attention_oversample": active_spec.attention_oversample[stage_index],
        "rebalance": active_spec.rebalance_stages[stage_index],
        "attention_seed_count": attention_seed_count,
        "diversity_fill_count": max(0, target - attention_seed_count),
        "source_tokens": int(hidden.shape[0]),
        "result_tokens": int(output.shape[0]),
        "source_image_tokens": visual_count,
        "result_image_tokens": int(kept_images.numel()),
        "selected_indices": tuple(int(value) for value in kept.tolist()),
        "selected_image_indices": tuple(int(value) for value in kept_images.tolist()),
        "position_policy": active_spec.position_policy,
        "compression_applied": bool(dropped),
    }
    return BalancedTokenPruningResult(
        hidden_states=output,
        kept_indices=kept,
        dropped_indices=torch.tensor(
            dropped,
            dtype=torch.long,
            device=hidden.device,
        ),
        kept_image_indices=kept_images,
        selected_visual_indices=selected_local,
        layout_delta=delta,
        metadata=metadata,
        attention_scores=scores,
    )


def _validate_inputs(hidden_states: Any, *, image_indices: Any) -> tuple[Any, Any]:
    torch = _torch()
    if (
        not isinstance(hidden_states, torch.Tensor)
        or hidden_states.ndim != 2
        or not hidden_states.is_floating_point()
    ):
        raise ValueError("hidden_states must be a floating rank-2 tensor.")
    images = torch.as_tensor(
        image_indices,
        dtype=torch.long,
        device=hidden_states.device,
    )
    if images.ndim != 1 or int(images.numel()) < 1:
        raise ValueError("image_indices must be a non-empty rank-one sequence.")
    values = tuple(int(value) for value in images.tolist())
    if values != tuple(range(values[0], values[-1] + 1)):
        raise ValueError("BTP requires one contiguous serialized image span.")
    if values[0] < 0 or values[-1] >= int(hidden_states.shape[0]):
        raise ValueError("image_indices contains an out-of-range position.")
    return hidden_states, images


def _attention_scores(value: Any) -> Any:
    torch = _torch()
    if (
        not isinstance(value, torch.Tensor)
        or value.ndim != 1
        or int(value.numel()) < 1
        or not value.is_floating_point()
    ):
        raise ValueError("BTP attention scores must be a non-empty floating vector.")
    if not bool(torch.isfinite(value).all().item()):
        raise ValueError("BTP attention scores must be finite.")
    return value.float()


def _integer(value: Any, name: str, *, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer.")
    if value < minimum:
        raise ValueError(f"{name} must be at least {minimum}.")
    return value


def _integer_tuple(value: Any, name: str, *, minimum: int) -> tuple[int, ...]:
    if isinstance(value, (str, bytes)):
        raise TypeError(f"{name} must be a sequence of integers.")
    try:
        result = tuple(value)
    except TypeError as exc:
        raise TypeError(f"{name} must be a sequence of integers.") from exc
    for item in result:
        _integer(item, name, minimum=minimum)
    return result


def _real_tuple(value: Any, name: str) -> tuple[float, ...]:
    if isinstance(value, (str, bytes)):
        raise TypeError(f"{name} must be a sequence of real numbers.")
    try:
        return tuple(require_finite_real(item, name) for item in value)
    except TypeError as exc:
        raise TypeError(f"{name} must be a sequence of real numbers.") from exc


def _bool_tuple(value: Any, name: str) -> tuple[bool, ...]:
    if isinstance(value, (str, bytes)):
        raise TypeError(f"{name} must be a sequence of booleans.")
    try:
        result = tuple(value)
    except TypeError as exc:
        raise TypeError(f"{name} must be a sequence of booleans.") from exc
    if any(not isinstance(item, bool) for item in result):
        raise TypeError(f"{name} must contain only booleans.")
    return result


def _torch() -> Any:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - optional local dependency
        raise RuntimeError("Balanced Token Pruning requires PyTorch.") from exc
    return torch


__all__ = [
    "BalancedTokenPruningResult",
    "BalancedTokenPruningSpec",
    "balanced_token_pruning_image_tokens",
    "btp_maxmin_diversity_fill",
    "btp_rebalanced_attention_topk",
]
