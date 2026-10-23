"""Backbone-neutral HalfV selection for image decoder tokens.

The released Qwen2.5-VL path applies two prefill-only stages.  IVR runs
AnchorCover before zero-based decoder layer 2: last-text attention seeds a
greedy cosine max-min cover.  SSR runs before layer 21 and keeps the strongest
last-text-attended tokens.  Providers own Q/K projection, optional RoPE,
sequence repair, and completed-prefix cache compaction.
"""

from __future__ import annotations

from ._rules import rule_metadata, score_rule, selection_rule

from dataclasses import dataclass, field
from typing import Any, Mapping

from mllm_comp.core.layout import LayoutDelta

from .decoder_config import require_finite_real


@dataclass(frozen=True)
class HalfVSpec:
    """Released Qwen2.5-VL HalfV schedule expressed as native config."""

    ivr_layer_index: int = 2
    ivr_retention_ratio: float = 0.25
    anchor_ratio: float = 0.1
    ssr_layer_index: int = 21
    ssr_retention_ratio: float = 0.05
    score_query: str = "last_text_token"
    ivr_use_rope: bool = False
    ssr_use_rope: bool = True
    diversity_metric: str = "cosine"
    position_policy: str = "preserve_original"
    backbone_strategy: str = "qwen_extreme_token_sparsity"
    release_variant: str = "qwen2_5_vl"
    score_accumulation: str = "fp32"
    single_image_only: bool = True

    def __post_init__(self) -> None:
        ivr_layer = _integer(self.ivr_layer_index, "ivr_layer_index", minimum=1)
        ssr_layer = _integer(self.ssr_layer_index, "ssr_layer_index", minimum=2)
        ivr_ratio = require_finite_real(
            self.ivr_retention_ratio,
            "ivr_retention_ratio",
        )
        anchor_ratio = require_finite_real(self.anchor_ratio, "anchor_ratio")
        ssr_ratio = require_finite_real(
            self.ssr_retention_ratio,
            "ssr_retention_ratio",
        )
        if ssr_layer <= ivr_layer:
            raise ValueError("ssr_layer_index must follow ivr_layer_index.")
        if not 0.0 < ivr_ratio <= 1.0:
            raise ValueError("ivr_retention_ratio must lie in (0, 1].")
        if not 0.0 <= anchor_ratio <= 1.0:
            raise ValueError("anchor_ratio must lie in [0, 1].")
        if not 0.0 < ssr_ratio <= ivr_ratio:
            raise ValueError(
                "ssr_retention_ratio must be positive and no larger than "
                "ivr_retention_ratio."
            )
        if self.score_query != "last_text_token":
            raise ValueError("score_query must be 'last_text_token'.")
        if self.ivr_use_rope is not False:
            raise ValueError("The released HalfV IVR scorer disables RoPE.")
        if self.ssr_use_rope is not True:
            raise ValueError("The released Qwen HalfV SSR scorer requires RoPE.")
        if self.diversity_metric != "cosine":
            raise ValueError("diversity_metric must be 'cosine'.")
        if self.position_policy != "preserve_original":
            raise ValueError("position_policy must be 'preserve_original'.")
        if self.backbone_strategy != "qwen_extreme_token_sparsity":
            raise ValueError(
                "backbone_strategy must be 'qwen_extreme_token_sparsity'."
            )
        if self.release_variant != "qwen2_5_vl":
            raise ValueError("release_variant must be 'qwen2_5_vl'.")
        if self.score_accumulation != "fp32":
            raise ValueError("score_accumulation must be 'fp32'.")
        if self.single_image_only is not True:
            raise ValueError("HalfV is single-image only.")
        object.__setattr__(self, "ivr_layer_index", ivr_layer)
        object.__setattr__(self, "ssr_layer_index", ssr_layer)
        object.__setattr__(self, "ivr_retention_ratio", ivr_ratio)
        object.__setattr__(self, "anchor_ratio", anchor_ratio)
        object.__setattr__(self, "ssr_retention_ratio", ssr_ratio)

    @property
    def boundary_after_layer_indices(self) -> tuple[int, int]:
        """Completed zero-based layers after which IVR and SSR run."""

        return (self.ivr_layer_index - 1, self.ssr_layer_index - 1)

    def stage_index(self, boundary_after_layer: int) -> int:
        boundary = _integer(
            boundary_after_layer,
            "boundary_after_layer",
            minimum=0,
        )
        try:
            return self.boundary_after_layer_indices.index(boundary)
        except ValueError as exc:
            raise ValueError(
                f"decoder layer {boundary} is not a HalfV boundary."
            ) from exc

    def validate_decoder_depth(self, layer_count: int) -> None:
        count = _integer(layer_count, "layer_count", minimum=1)
        if self.ssr_layer_index >= count:
            raise ValueError(
                "HalfV decoder depth must include zero-based SSR layer "
                f"{self.ssr_layer_index}."
            )


@dataclass(frozen=True)
class HalfVResult:
    """One source-ordered HalfV stage and its exact sequence layout."""

    hidden_states: Any
    kept_indices: Any
    dropped_indices: Any
    kept_image_indices: Any
    selected_visual_indices: Any
    layout_delta: LayoutDelta
    metadata: Mapping[str, Any] = field(default_factory=dict)
    attention_scores: Any | None = None


def halfv_last_text_attention_scores(query: Any, image_keys: Any,
                                      *, rules: Mapping[str, Any] | None = None) -> Any:
    """Released unscaled, per-head-softmax then head-mean scorer."""

    torch = _torch()
    if (
        not isinstance(query, torch.Tensor)
        or query.ndim != 3
        or int(query.shape[0]) != 1
        or not query.is_floating_point()
    ):
        raise ValueError("HalfV query must be a floating [1,H,D] tensor.")
    if (
        not isinstance(image_keys, torch.Tensor)
        or image_keys.ndim != 3
        or int(image_keys.shape[0]) < 1
        or not image_keys.is_floating_point()
    ):
        raise ValueError("HalfV image_keys must be a non-empty floating [N,H,D].")
    if tuple(query.shape[1:]) != tuple(image_keys.shape[1:]):
        raise ValueError("HalfV query and image key head shapes must match.")
    if not bool(torch.isfinite(query).all().item()) or not bool(
        torch.isfinite(image_keys).all().item()
    ):
        raise ValueError("HalfV scorer inputs must be finite.")
    logits = torch.einsum(
        "qhd,khd->hqk",
        query.float(),
        image_keys.float(),
    )
    scores = torch.softmax(logits, dim=-1, dtype=torch.float32).mean(dim=0)[0]
    if not bool(torch.isfinite(scores).all().item()):
        raise RuntimeError("HalfV attention scores contain non-finite values.")
    return score_rule(rules, "source_scorer", {"query": query, "image_keys": image_keys}, scores)


def halfv_anchor_cover_select(
    visual_hidden: Any,
    attention_scores: Any,
    *,
    target_count: int,
    anchor_ratio: float,
) -> tuple[Any, int]:
    """Attention anchors followed by released greedy cosine max-min fill."""

    torch = _torch()
    if (
        not isinstance(visual_hidden, torch.Tensor)
        or visual_hidden.ndim != 2
        or int(visual_hidden.shape[0]) < 1
        or not visual_hidden.is_floating_point()
    ):
        raise ValueError("visual_hidden must be a non-empty floating [N,D] tensor.")
    scores = _attention_scores(attention_scores).to(visual_hidden.device)
    if int(scores.numel()) != int(visual_hidden.shape[0]):
        raise ValueError("HalfV attention scores must match visual tokens.")
    target = min(
        int(visual_hidden.shape[0]),
        _integer(target_count, "target_count", minimum=0),
    )
    ratio = require_finite_real(anchor_ratio, "anchor_ratio")
    if not 0.0 <= ratio <= 1.0:
        raise ValueError("anchor_ratio must lie in [0, 1].")
    if target == 0:
        return torch.empty(0, dtype=torch.long, device=visual_hidden.device), 0
    anchor_count = min(target, max(0, int(target * ratio)))
    if anchor_count:
        anchors = scores.topk(anchor_count, largest=True, sorted=True).indices
        selected = [int(value) for value in anchors.detach().cpu().tolist()]
    else:
        anchors = torch.empty(0, dtype=torch.long, device=visual_hidden.device)
        selected = []
    normalized = torch.nn.functional.normalize(
        visual_hidden.float(),
        p=2,
        dim=-1,
        eps=1e-8,
    )
    selected_mask = torch.zeros(
        int(visual_hidden.shape[0]),
        dtype=torch.bool,
        device=visual_hidden.device,
    )
    if selected:
        selected_mask[anchors] = True
        similarities = normalized @ normalized.index_select(0, anchors).transpose(0, 1)
        min_distance = (1.0 - similarities).amin(dim=1)
    else:
        min_distance = torch.full(
            (int(visual_hidden.shape[0]),),
            2.0,
            dtype=torch.float32,
            device=visual_hidden.device,
        )
    min_distance[selected_mask] = -1.0
    while len(selected) < target:
        next_index = int(min_distance.argmax().item())
        if bool(selected_mask[next_index].item()):
            raise RuntimeError("HalfV AnchorCover attempted a duplicate token.")
        selected.append(next_index)
        selected_mask[next_index] = True
        new_distance = 1.0 - normalized @ normalized[next_index]
        min_distance = torch.minimum(min_distance, new_distance)
        min_distance[selected_mask] = -1.0
    return (
        torch.tensor(
            selected,
            dtype=torch.long,
            device=visual_hidden.device,
        ).sort().values,
        anchor_count,
    )


def halfv_image_tokens(
    hidden_states: Any,
    *,
    image_indices: Any,
    boundary_after_layer: int,
    image_attention_scores: Any,
    initial_visual_tokens: int,
    selection_hidden_states: Any | None = None,
    spec: HalfVSpec | None = None,
    media_unit: str = "llm_token",
    source_layout: Mapping[str, Any] | None = None,
    result_layout: Mapping[str, Any] | None = None,
    rules: Mapping[str, Any] | None = None,
) -> HalfVResult:
    """Run one IVR or SSR stage after its scorer layer."""

    torch = _torch()
    active_spec = spec or HalfVSpec()
    hidden, images = _validate_inputs(hidden_states, image_indices=image_indices)
    initial = _integer(initial_visual_tokens, "initial_visual_tokens", minimum=1)
    visual_count = int(images.numel())
    if initial < visual_count:
        raise ValueError("initial_visual_tokens cannot be smaller than the current span.")
    stage_index = active_spec.stage_index(boundary_after_layer)
    scores = _attention_scores(image_attention_scores).to(hidden.device)
    if int(scores.numel()) != visual_count:
        raise ValueError("HalfV attention scores do not match the image span.")
    target = min(visual_count, max(0, int((visual_count * active_spec.ivr_retention_ratio)
        if stage_index == 0 else (initial * active_spec.ssr_retention_ratio))))
    request = {"hidden_states": hidden, "image_indices": images,
        "selection_hidden_states": selection_hidden_states, "image_attention_scores": image_attention_scores,
        "stage_index": stage_index, "keep_count": target, "spec": active_spec}
    if 0 < target < visual_count:
        scores = score_rule(rules, "scorer", request, scores)
    if stage_index == 0:
        if initial != visual_count:
            raise ValueError("HalfV IVR must receive the original visual span.")
        if (
            not isinstance(selection_hidden_states, torch.Tensor)
            or tuple(selection_hidden_states.shape) != tuple(hidden.shape)
            or not selection_hidden_states.is_floating_point()
        ):
            raise ValueError(
                "HalfV IVR selection_hidden_states must match hidden_states."
            )
        target = min(
            visual_count,
            max(0, int(visual_count * active_spec.ivr_retention_ratio)),
        )
        selected_local, anchor_count = halfv_anchor_cover_select(
            selection_hidden_states.to(hidden.device).index_select(0, images),
            scores,
            target_count=target,
            anchor_ratio=active_spec.anchor_ratio,
        )
        stage = "ivr"
        score_rope = active_spec.ivr_use_rope
    else:
        target = min(
            visual_count,
            max(0, int(initial * active_spec.ssr_retention_ratio)),
        )
        selected_local = (
            scores.topk(target, largest=True, sorted=True).indices.sort().values
            if target
            else torch.empty(0, dtype=torch.long, device=hidden.device)
        )
        anchor_count = 0
        stage = "ssr"
        score_rope = active_spec.ssr_use_rope
    if 0 < target < visual_count:
        custom = selection_rule(rules, "selector", {**request, "scores": scores},
            count=visual_count, keep_count=target, device=hidden.device)
        if custom is not None:
            selected_local = custom.sort().values
            anchor_count = 0
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
        "algorithm": "halfv",
        "variant": "halfv.image",
        **rule_metadata(rules),
        **({"selection": "custom"} if rules and "selector" in rules else {}),
        "stage_index": stage_index,
        "stage": stage,
        "boundary_after_layer": boundary_after_layer,
        "prune_before_layer_index": boundary_after_layer + 1,
        "retention_ratio": (
            active_spec.ivr_retention_ratio
            if stage == "ivr"
            else active_spec.ssr_retention_ratio
        ),
        "target_relative_to": "current_visual" if stage == "ivr" else "original_visual",
        "attention_anchor_count": anchor_count,
        "coverage_fill_count": max(0, target - anchor_count),
        "score_rope": score_rope,
        "score_accumulation": active_spec.score_accumulation,
        "source_tokens": int(hidden.shape[0]),
        "result_tokens": int(output.shape[0]),
        "initial_image_tokens": initial,
        "source_image_tokens": visual_count,
        "result_image_tokens": int(kept_images.numel()),
        "selected_indices": tuple(int(value) for value in kept.tolist()),
        "selected_image_indices": tuple(int(value) for value in kept_images.tolist()),
        "position_policy": active_spec.position_policy,
        "compression_applied": bool(dropped),
    }
    return HalfVResult(
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
        raise ValueError("HalfV requires one contiguous serialized image span.")
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
        raise ValueError("HalfV attention scores must be a non-empty floating vector.")
    if not bool(torch.isfinite(value).all().item()):
        raise ValueError("HalfV attention scores must be finite.")
    return value.float()


def _integer(value: Any, name: str, *, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer.")
    if value < minimum:
        raise ValueError(f"{name} must be at least {minimum}.")
    return value


def _torch() -> Any:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - optional local dependency
        raise RuntimeError("HalfV requires PyTorch.") from exc
    return torch


__all__ = [
    "HalfVResult",
    "HalfVSpec",
    "halfv_anchor_cover_select",
    "halfv_image_tokens",
    "halfv_last_text_attention_scores",
]
