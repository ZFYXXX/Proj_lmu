"""Backbone-neutral TransPrune scoring and image-token selection.

TransPrune accumulates Token Transition Variation (TTV) across selected
decoder layers and combines it with Instruction-Guided Attention (IGA) from
the following layer.  This module owns that mathematics once.  Backbone
providers only capture native module boundaries, project their own Q/K states,
and legalize masks, positions, caches, and serialized image layout.
"""

from __future__ import annotations

from ._rules import rule_metadata, score_rule, selection_rule

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from mllm_comp.core.layout import LayoutDelta

from .decoder_config import require_finite_real
from .pyramid_drop import pyramid_attention_scores


@dataclass(frozen=True)
class TransPruneSpec:
    """Fixed public-release recipe expressed as a native framework contract."""

    accumulating_layers: tuple[int, ...] = (7, 8, 9, 10, 11, 12)
    prune_after_layers: tuple[int, ...] = (7, 9, 12)
    retain_ratios: tuple[float, ...] = (0.875, 0.625, 0.125)
    alpha: float = 0.5
    direction_rule: str = "release_1_minus_cosine"
    ttv_softmax_scope: str = "release_full_sequence"
    iga_source: str = "next_layer_post_image_text"
    position_policy: str = "contiguous"
    score_accumulation: str = "fp32"
    single_image_only: bool = True

    def __post_init__(self) -> None:
        accumulating = _integer_tuple(
            self.accumulating_layers,
            "accumulating_layers",
            minimum=1,
        )
        pruning = _integer_tuple(
            self.prune_after_layers,
            "prune_after_layers",
            minimum=1,
        )
        ratios = tuple(
            require_finite_real(value, "retain_ratios")
            for value in self.retain_ratios
        )
        alpha = require_finite_real(self.alpha, "alpha")
        if not accumulating or accumulating != tuple(sorted(set(accumulating))):
            raise ValueError("accumulating_layers must be non-empty, sorted, and unique.")
        if not pruning or pruning != tuple(sorted(set(pruning))):
            raise ValueError("prune_after_layers must be non-empty, sorted, and unique.")
        if any(layer not in accumulating for layer in pruning):
            raise ValueError("Every pruning layer must also accumulate TTV.")
        if len(ratios) != len(pruning):
            raise ValueError("retain_ratios must match prune_after_layers.")
        if any(not 0.0 <= ratio <= 1.0 for ratio in ratios):
            raise ValueError("retain_ratios must lie in [0, 1].")
        if any(left <= right for left, right in zip(ratios, ratios[1:])):
            raise ValueError("retain_ratios must be strictly decreasing.")
        if not 0.0 <= alpha <= 1.0:
            raise ValueError("alpha must lie in [0, 1].")
        if self.direction_rule not in {
            "release_1_minus_cosine",
            "paper_1_minus_abs_cosine",
        }:
            raise ValueError("direction_rule must select the release or paper equation.")
        if self.ttv_softmax_scope not in {
            "release_full_sequence",
            "paper_visual_tokens",
        }:
            raise ValueError(
                "ttv_softmax_scope must be release_full_sequence or paper_visual_tokens."
            )
        if self.iga_source != "next_layer_post_image_text":
            raise ValueError("iga_source must be 'next_layer_post_image_text'.")
        if self.position_policy != "contiguous":
            raise ValueError("position_policy must be 'contiguous'.")
        if self.score_accumulation != "fp32":
            raise ValueError("score_accumulation must be 'fp32'.")
        if self.single_image_only is not True:
            raise ValueError("The native TransPrune adaptation is single-image only.")
        object.__setattr__(self, "accumulating_layers", accumulating)
        object.__setattr__(self, "prune_after_layers", pruning)
        object.__setattr__(self, "retain_ratios", ratios)
        object.__setattr__(self, "alpha", alpha)

    @property
    def accumulating_layer_indices(self) -> tuple[int, ...]:
        """Zero-based decoder indices whose module transitions are captured."""

        return tuple(layer - 1 for layer in self.accumulating_layers)

    @property
    def boundary_after_layer_indices(self) -> tuple[int, ...]:
        """Zero-based completed layers after which selection runs."""

        return tuple(layer - 1 for layer in self.prune_after_layers)

    def stage_index(self, boundary_after_layer: int) -> int:
        boundary = _integer(boundary_after_layer, "boundary_after_layer", minimum=0)
        try:
            return self.boundary_after_layer_indices.index(boundary)
        except ValueError as exc:
            raise ValueError(
                f"decoder layer {boundary} is not a TransPrune boundary."
            ) from exc

    def validate_decoder_depth(self, layer_count: int) -> None:
        count = _integer(layer_count, "layer_count", minimum=1)
        if self.accumulating_layers[-1] > count:
            raise ValueError("TransPrune accumulation exceeds decoder depth.")
        if self.prune_after_layers[-1] >= count:
            raise ValueError(
                "Every TransPrune stage requires a following decoder layer for IGA."
            )


@dataclass(frozen=True)
class TransPruneResult:
    """One progressive TransPrune selection and its exact layout delta."""

    hidden_states: Any
    kept_indices: Any
    dropped_indices: Any
    kept_image_indices: Any
    selected_visual_indices: Any
    layout_delta: LayoutDelta
    normalized_ttv: Any
    normalized_iga: Any
    combined_scores: Any
    metadata: Mapping[str, Any] = field(default_factory=dict)


def transprune_module_transition_variation(
    before: Any,
    after: Any,
    *,
    image_indices: Any,
    direction_rule: str,
    softmax_scope: str,
    rules: Mapping[str, Any] | None = None,
    module_name: str | None = None,
) -> Any:
    """Compute one module's released magnitude-times-direction TTV vector."""

    torch = _torch()
    if (
        not isinstance(before, torch.Tensor)
        or not isinstance(after, torch.Tensor)
        or before.ndim != 3
        or tuple(before.shape) != tuple(after.shape)
        or int(before.shape[0]) != 1
        or not before.is_floating_point()
        or not after.is_floating_point()
    ):
        raise ValueError("TransPrune module states must match floating [1,T,D] tensors.")
    images = _image_indices(image_indices, int(before.shape[1]), before.device)
    before_fp32 = before.detach().float()
    after_fp32 = after.detach().float()
    magnitude = torch.linalg.vector_norm(after_fp32, dim=-1) / torch.linalg.vector_norm(
        before_fp32,
        dim=-1,
    ).clamp_min(1e-8)
    cosine = torch.nn.functional.cosine_similarity(
        before_fp32,
        after_fp32,
        dim=-1,
        eps=1e-8,
    )
    if direction_rule == "release_1_minus_cosine":
        direction = 1.0 - cosine
    elif direction_rule == "paper_1_minus_abs_cosine":
        direction = 1.0 - cosine.abs()
    else:
        raise ValueError(f"Unsupported TransPrune direction rule {direction_rule!r}.")
    if softmax_scope == "release_full_sequence":
        weights = torch.softmax(direction, dim=-1, dtype=torch.float32)
        result = weights[0].index_select(0, images) * magnitude[0].index_select(
            0, images
        )
    elif softmax_scope == "paper_visual_tokens":
        visual_direction = direction[0].index_select(0, images)
        result = torch.softmax(
            visual_direction,
            dim=-1,
            dtype=torch.float32,
        ) * magnitude[0].index_select(0, images)
    else:
        raise ValueError(f"Unsupported TransPrune softmax scope {softmax_scope!r}.")
    if not bool(torch.isfinite(result).all().item()):
        raise RuntimeError("TransPrune module TTV contains non-finite values.")
    return score_rule(rules, "transition_scorer",
        {"before": before, "after": after, "image_indices": images,
         "module": module_name, "direction_rule": direction_rule, "softmax_scope": softmax_scope}, result)


def transprune_layer_transition_variation(
    attention_before: Any,
    attention_after: Any,
    ffn_before: Any,
    ffn_after: Any,
    *,
    image_indices: Any,
    direction_rule: str,
    softmax_scope: str,
    rules: Mapping[str, Any] | None = None,
    attention_image_indices: Any | None = None,
    selected_visual_indices: Any | None = None,
) -> Any:
    """Sum module TTV while retaining the attention's pre-selection softmax scope."""

    attention = transprune_module_transition_variation(
        attention_before, attention_after,
        image_indices=attention_image_indices if selected_visual_indices is not None else image_indices,
        direction_rule=direction_rule, softmax_scope=softmax_scope,
        rules=rules, module_name="attention",
    )
    if selected_visual_indices is not None:
        attention = attention.index_select(0, selected_visual_indices.to(attention.device))
    ffn = transprune_module_transition_variation(
        ffn_before, ffn_after, image_indices=image_indices,
        direction_rule=direction_rule, softmax_scope=softmax_scope,
        rules=rules, module_name="ffn",
    )
    return attention.to(ffn.device) + ffn


def transprune_instruction_guided_attention(
    *,
    instruction_queries: Any,
    image_keys: Any,
    attention_keys: Any,
    image_indices: Any,
    attention_key_mask: Any | None,
) -> Any:
    """Score image keys after a full-key causal softmax, then average Q/H."""

    torch = _torch()
    with torch.autocast(device_type=image_keys.device.type, enabled=False):
        return pyramid_attention_scores(
            instruction_query=instruction_queries.detach().float(),
            image_keys=image_keys.detach().float(),
            attention_keys=attention_keys.detach().float(),
            attention_key_mask=attention_key_mask,
            image_key_indices=image_indices,
            image_count=int(image_keys.shape[0]),
            device=image_keys.device,
        ).float()


def transprune_image_tokens(
    hidden_states: Any,
    *,
    image_indices: Any,
    boundary_after_layer: int,
    ttv_history: Sequence[Any],
    image_attention_scores: Any,
    initial_visual_tokens: int,
    spec: TransPruneSpec | None = None,
    protected_indices: Sequence[int] = (),
    media_unit: str = "llm_token",
    source_layout: Mapping[str, Any] | None = None,
    result_layout: Mapping[str, Any] | None = None,
    rules: Mapping[str, Any] | None = None,
) -> TransPruneResult:
    """Combine accumulated TTV and next-layer IGA, then drop in source order."""

    torch = _torch()
    active_spec = spec or TransPruneSpec()
    if (
        not isinstance(hidden_states, torch.Tensor)
        or hidden_states.ndim != 2
        or not hidden_states.is_floating_point()
    ):
        raise ValueError("hidden_states must be a floating rank-two tensor.")
    images = _image_indices(
        image_indices,
        int(hidden_states.shape[0]),
        hidden_states.device,
    )
    initial = _integer(initial_visual_tokens, "initial_visual_tokens", minimum=1)
    visual_count = int(images.numel())
    if initial < visual_count:
        raise ValueError("initial_visual_tokens cannot be smaller than the current span.")
    stage_index = active_spec.stage_index(boundary_after_layer)
    if not ttv_history:
        raise ValueError("TransPrune requires at least one accumulated TTV vector.")
    vectors = tuple(
        _score_vector(value, visual_count, hidden_states.device, "TTV")
        for value in ttv_history
    )
    iga = _score_vector(
        image_attention_scores,
        visual_count,
        hidden_states.device,
        "IGA",
    )
    accumulated = torch.stack(vectors).sum(dim=0)
    normalized_ttv = accumulated / accumulated.sum().clamp_min(1e-8)
    normalized_iga = iga / iga.sum().clamp_min(1e-8)
    combined = (
        active_spec.alpha * normalized_ttv
        + (1.0 - active_spec.alpha) * normalized_iga
    )
    target = min(
        visual_count,
        max(0, int(initial * active_spec.retain_ratios[stage_index])),
    )
    if target == 0 and stage_index + 1 < len(active_spec.prune_after_layers):
        raise ValueError(
            "TransPrune configuration exhausts the image span before its final stage."
        )
    request = {"hidden_states": hidden_states, "image_indices": images, "ttv_history": vectors,
        "image_attention_scores": iga, "stage_index": stage_index,
        "keep_count": target, "spec": active_spec}
    if 0 < target < visual_count:
        combined = score_rule(rules, "scorer", request, combined)
    selected_local = (
        combined.topk(target, largest=True, sorted=True).indices.sort().values
        if target
        else torch.empty(0, dtype=torch.long, device=hidden_states.device)
    )
    if 0 < target < visual_count:
        custom = selection_rule(rules, "selector", {**request, "scores": combined},
            count=visual_count, keep_count=target, device=hidden_states.device,
            protected=tuple(i for i, value in enumerate(images.tolist()) if value in protected_indices))
        if custom is not None:
            selected_local = custom.sort().values
    kept_images = images.index_select(0, selected_local)
    image_set = set(int(value) for value in images.tolist())
    kept_image_set = set(int(value) for value in kept_images.tolist())
    dropped_values = tuple(sorted(image_set - kept_image_set))
    dropped_set = set(dropped_values)
    kept = torch.tensor(
        [
            index
            for index in range(int(hidden_states.shape[0]))
            if index not in dropped_set
        ],
        dtype=torch.long,
        device=hidden_states.device,
    )
    protected = tuple(
        sorted(
            set(int(value) for value in protected_indices)
            | {
                index
                for index in range(int(hidden_states.shape[0]))
                if index not in image_set
            }
        )
    )
    delta = LayoutDelta.drop(
        int(hidden_states.shape[0]),
        dropped_values,
        media_unit=media_unit,
        protected_indices=protected,
        source_layout=source_layout,
        result_layout=result_layout,
    )
    output = hidden_states.index_select(0, kept)
    metadata = {
        "algorithm": "transprune",
        "variant": "transprune.image",
        **rule_metadata(rules),
        "stage_index": stage_index,
        "boundary_after_layer": boundary_after_layer,
        "iga_layer_index": boundary_after_layer + 1,
        "retain_ratio_of_original": active_spec.retain_ratios[stage_index],
        "alpha": active_spec.alpha,
        "accumulated_layer_count": len(vectors),
        "source_tokens": int(hidden_states.shape[0]),
        "result_tokens": int(output.shape[0]),
        "initial_image_tokens": initial,
        "source_image_tokens": visual_count,
        "result_image_tokens": int(kept_images.numel()),
        "selected_indices": tuple(int(value) for value in kept.tolist()),
        "selected_image_indices": tuple(int(value) for value in kept_images.tolist()),
        "position_policy": active_spec.position_policy,
        "cache_policy": "layer_local",
        "compression_applied": bool(dropped_values),
    }
    return TransPruneResult(
        hidden_states=output,
        kept_indices=kept,
        dropped_indices=torch.tensor(
            dropped_values,
            dtype=torch.long,
            device=hidden_states.device,
        ),
        kept_image_indices=kept_images,
        selected_visual_indices=selected_local,
        layout_delta=delta,
        normalized_ttv=normalized_ttv,
        normalized_iga=normalized_iga,
        combined_scores=combined,
        metadata=metadata,
    )


def select_transprune_ttv_history(
    ttv_history: Sequence[Any],
    selected_visual_indices: Any,
) -> list[Any]:
    """Keep every accumulated TTV vector aligned with current survivors."""

    torch = _torch()
    selected = torch.as_tensor(selected_visual_indices, dtype=torch.long)
    if selected.ndim != 1:
        raise ValueError("selected_visual_indices must be rank one.")
    return [
        value.index_select(0, selected.to(value.device))
        for value in ttv_history
    ]


def _score_vector(value: Any, count: int, device: Any, name: str) -> Any:
    torch = _torch()
    if not isinstance(value, torch.Tensor) or not value.is_floating_point():
        raise TypeError(f"TransPrune {name} must be a floating tensor.")
    vector = value.detach().reshape(-1).float().to(device)
    if int(vector.numel()) != count:
        raise ValueError(f"TransPrune {name} does not match current image tokens.")
    if not bool(torch.isfinite(vector).all().item()):
        raise ValueError(f"TransPrune {name} must be finite.")
    return vector


def _image_indices(value: Any, token_count: int, device: Any) -> Any:
    torch = _torch()
    images = torch.as_tensor(value, dtype=torch.long, device=device)
    if images.ndim != 1 or int(images.numel()) < 1:
        raise ValueError("image_indices must be a non-empty rank-one sequence.")
    values = tuple(int(item) for item in images.tolist())
    if values != tuple(range(values[0], values[-1] + 1)):
        raise ValueError("TransPrune requires one contiguous serialized image span.")
    if values[0] < 0 or values[-1] >= token_count:
        raise ValueError("image_indices contains an out-of-range position.")
    return images


def _integer_tuple(value: Any, name: str, *, minimum: int) -> tuple[int, ...]:
    if not isinstance(value, (tuple, list)):
        raise TypeError(f"{name} must be a tuple or list of integers.")
    return tuple(_integer(item, name, minimum=minimum) for item in value)


def _integer(value: Any, name: str, *, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must contain integers.")
    if value < minimum:
        raise ValueError(f"{name} values must be at least {minimum}.")
    return value


def _torch() -> Any:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - optional local dependency
        raise RuntimeError("TransPrune requires PyTorch.") from exc
    return torch


__all__ = [
    "TransPruneResult",
    "TransPruneSpec",
    "select_transprune_ttv_history",
    "transprune_image_tokens",
    "transprune_instruction_guided_attention",
    "transprune_layer_transition_variation",
    "transprune_module_transition_variation",
]
