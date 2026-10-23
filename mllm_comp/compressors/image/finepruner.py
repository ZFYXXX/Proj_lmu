"""Backbone-neutral FinePruner IAC and ARP image-token selection.

FinePruner first removes redundant visual tokens with deterministic random-
hyperplane LSH clustering (IAC), then scores the remaining span with attention
averaged across several decoder layers and the least-biased heads (ARP).
Backbone bindings own serialized image spans, decoder-layer scheduling,
position rebuilding, masks, and layer-local cache semantics.
"""

from __future__ import annotations

from ._rules import label_rule, rule_metadata, score_rule, selection_rule

from dataclasses import dataclass, field
import math
from typing import Any, Iterable, Mapping, Sequence

from mllm_comp.core.layout import LayoutDelta

from .decoder_config import require_finite_real


@dataclass(frozen=True)
class FinePrunerSpec:
    """Paper-faithful train-free FinePruner configuration."""

    iac_after_layers: tuple[int, ...] = (7, 14)
    iac_hash_bits: tuple[int, ...] = (48, 41)
    arp_center_after_layer: int = 21
    arp_layer_offsets: tuple[int, ...] = (-1, 0, 1)
    arp_retain_ratio: float = 0.6
    target_visual_tokens: int | None = None
    head_ratio: float = 0.5
    random_seed: int = 126
    representative: str = "first"
    score_query: str = "last_text_token"
    position_policy: str = "contiguous"
    single_image_only: bool = True

    def __post_init__(self) -> None:
        iac_layers = _integer_tuple(
            self.iac_after_layers,
            "iac_after_layers",
            minimum=1,
        )
        hash_bits = _integer_tuple(
            self.iac_hash_bits,
            "iac_hash_bits",
            minimum=1,
        )
        offsets = _integer_tuple(
            self.arp_layer_offsets,
            "arp_layer_offsets",
            minimum=None,
        )
        if not iac_layers:
            raise ValueError("iac_after_layers must not be empty.")
        if iac_layers != tuple(sorted(set(iac_layers))):
            raise ValueError("iac_after_layers must be sorted and unique.")
        if len(iac_layers) != len(hash_bits):
            raise ValueError(
                "iac_after_layers and iac_hash_bits must have equal length."
            )
        if any(bits > 63 for bits in hash_bits):
            raise ValueError("iac_hash_bits values must be in [1, 63].")
        if (
            isinstance(self.arp_center_after_layer, bool)
            or not isinstance(self.arp_center_after_layer, int)
            or self.arp_center_after_layer < 1
        ):
            raise ValueError("arp_center_after_layer must be a positive integer.")
        if not offsets:
            raise ValueError("arp_layer_offsets must not be empty.")
        targets = tuple(self.arp_center_after_layer + offset for offset in offsets)
        if any(layer < 1 for layer in targets):
            raise ValueError("ARP target completed-layer counts must be positive.")
        if targets != tuple(sorted(set(targets))):
            raise ValueError(
                "ARP target completed-layer counts must be sorted and unique."
            )
        if iac_layers[-1] >= targets[0]:
            raise ValueError("IAC stages must precede every ARP target layer.")
        retain_ratio = require_finite_real(
            self.arp_retain_ratio,
            "arp_retain_ratio",
        )
        if not 0.0 < retain_ratio <= 1.0:
            raise ValueError("arp_retain_ratio must be in (0, 1].")
        head_ratio = require_finite_real(self.head_ratio, "head_ratio")
        if not 0.0 < head_ratio <= 1.0:
            raise ValueError("head_ratio must be in (0, 1].")
        if self.target_visual_tokens is not None and (
            isinstance(self.target_visual_tokens, bool)
            or not isinstance(self.target_visual_tokens, int)
            or self.target_visual_tokens < 1
        ):
            raise ValueError(
                "target_visual_tokens must be None or a positive integer."
            )
        if isinstance(self.random_seed, bool) or not isinstance(self.random_seed, int):
            raise TypeError("random_seed must be an integer.")
        if self.representative != "first":
            raise ValueError("representative must be 'first'.")
        if self.score_query != "last_text_token":
            raise ValueError("score_query must be 'last_text_token'.")
        if self.position_policy != "contiguous":
            raise ValueError("position_policy must be 'contiguous'.")
        if self.single_image_only is not True:
            raise ValueError("FinePruner native adaptation is single-image only.")
        object.__setattr__(self, "iac_after_layers", iac_layers)
        object.__setattr__(self, "iac_hash_bits", hash_bits)
        object.__setattr__(self, "arp_layer_offsets", offsets)
        object.__setattr__(self, "arp_retain_ratio", retain_ratio)
        object.__setattr__(self, "head_ratio", head_ratio)

    @property
    def arp_target_after_layers(self) -> tuple[int, ...]:
        return tuple(
            self.arp_center_after_layer + offset
            for offset in self.arp_layer_offsets
        )

    @property
    def arp_apply_after_layer(self) -> int:
        return self.arp_target_after_layers[-1]

    def iac_stage_index(self, completed_layers: int) -> int:
        if isinstance(completed_layers, bool) or not isinstance(completed_layers, int):
            raise TypeError("completed_layers must be an integer.")
        try:
            return self.iac_after_layers.index(completed_layers)
        except ValueError as exc:
            raise ValueError(
                f"completed layer {completed_layers} is not an IAC boundary."
            ) from exc

    def iac_bits(self, completed_layers: int) -> int:
        return self.iac_hash_bits[self.iac_stage_index(completed_layers)]

    def resolve_final_visual_count(self, current_visual_tokens: int) -> int:
        if (
            isinstance(current_visual_tokens, bool)
            or not isinstance(current_visual_tokens, int)
            or current_visual_tokens < 1
        ):
            raise ValueError("current_visual_tokens must be a positive integer.")
        requested = self.target_visual_tokens
        if requested is None:
            requested = int(current_visual_tokens * self.arp_retain_ratio)
        return min(current_visual_tokens, max(1, int(requested)))

    def validate_decoder_depth(self, layer_count: int) -> None:
        if (
            isinstance(layer_count, bool)
            or not isinstance(layer_count, int)
            or layer_count < self.arp_apply_after_layer
        ):
            raise ValueError(
                "FinePruner decoder depth must reach completed layer "
                f"{self.arp_apply_after_layer}."
            )


@dataclass(frozen=True)
class FinePrunerResult:
    """One IAC or ARP selection and its source-ordered layout change."""

    hidden_states: Any
    kept_indices: Any
    dropped_indices: Any
    kept_image_indices: Any
    layout_delta: LayoutDelta
    metadata: Mapping[str, Any] = field(default_factory=dict)
    hash_codes: Any | None = None
    selected_heads: Any | None = None
    head_kl: Any | None = None
    token_scores: Any | None = None


def finepruner_lsh_codes(
    visual_hidden: Any,
    *,
    hash_bits: int,
    random_seed: int,
) -> Any:
    """Return deterministic random-hyperplane LSH codes without global RNG use."""

    torch = _torch()
    if (
        not isinstance(visual_hidden, torch.Tensor)
        or visual_hidden.ndim != 2
        or int(visual_hidden.shape[0]) < 1
        or not visual_hidden.is_floating_point()
    ):
        raise ValueError("visual_hidden must be a non-empty floating [N,D] tensor.")
    if isinstance(hash_bits, bool) or not isinstance(hash_bits, int):
        raise TypeError("hash_bits must be an integer.")
    if not 1 <= hash_bits <= 63:
        raise ValueError("hash_bits must be in [1, 63].")
    if isinstance(random_seed, bool) or not isinstance(random_seed, int):
        raise TypeError("random_seed must be an integer.")
    generator = torch.Generator(device="cpu")
    generator.manual_seed(random_seed)
    hyperplanes = torch.randn(
        hash_bits,
        int(visual_hidden.shape[-1]),
        generator=generator,
        dtype=torch.float32,
        device="cpu",
    ).to(device=visual_hidden.device)
    positive = visual_hidden.float() @ hyperplanes.transpose(0, 1) >= 0
    powers = torch.bitwise_left_shift(
        torch.ones(hash_bits, dtype=torch.long, device=visual_hidden.device),
        torch.arange(
            hash_bits - 1,
            -1,
            -1,
            dtype=torch.long,
            device=visual_hidden.device,
        ),
    )
    return (positive.to(torch.long) * powers).sum(dim=-1)


def finepruner_first_representatives(hash_codes: Any) -> Any:
    """Return the minimum source index from every hash bucket."""

    torch = _torch()
    if (
        not isinstance(hash_codes, torch.Tensor)
        or hash_codes.ndim != 1
        or int(hash_codes.numel()) < 1
    ):
        raise ValueError("hash_codes must be a non-empty rank-one tensor.")
    representatives: dict[int, int] = {}
    for index, code in enumerate(hash_codes.detach().cpu().tolist()):
        representatives.setdefault(int(code), index)
    return torch.tensor(
        sorted(representatives.values()),
        dtype=torch.long,
        device=hash_codes.device,
    )


def finepruner_iac_image_tokens(
    hidden_states: Any,
    *,
    image_indices: Any,
    completed_layers: int,
    initial_visual_tokens: int,
    spec: FinePrunerSpec | None = None,
    protected_indices: Iterable[int] = (),
    media_unit: str = "llm_token",
    source_layout: Mapping[str, Any] | None = None,
    result_layout: Mapping[str, Any] | None = None,
    rules: Mapping[str, Any] | None = None,
) -> FinePrunerResult:
    """Apply one IAC boundary and retain the first token in every LSH bucket."""

    torch = _torch()
    active_spec = spec or FinePrunerSpec()
    hidden, images, protected = _validate_inputs(
        hidden_states,
        image_indices=image_indices,
        protected_indices=protected_indices,
    )
    image_count = int(images.numel())
    if initial_visual_tokens < image_count:
        raise ValueError("initial_visual_tokens cannot be smaller than the current span.")
    stage_index = active_spec.iac_stage_index(completed_layers)
    codes = finepruner_lsh_codes(
        hidden.index_select(0, images),
        hash_bits=active_spec.iac_hash_bits[stage_index],
        random_seed=active_spec.random_seed + stage_index,
    )
    request = {"hidden_states": hidden, "image_indices": images,
        "stage_index": stage_index, "spec": active_spec}
    codes = label_rule(rules, "grouper", request, codes)
    selected_local = finepruner_first_representatives(codes)
    bucket_count = int(selected_local.numel())
    custom = selection_rule(rules, "iac_selector", {**request, "labels": codes, "keep_count": bucket_count},
        count=image_count, keep_count=bucket_count, device=hidden.device)
    if custom is not None:
        if codes.index_select(0, custom).unique().numel() != bucket_count:
            raise ValueError("iac_selector must retain exactly one representative per bucket.")
        selected_local = custom.sort().values
    image_start = int(images[0].item())
    protected_local = torch.tensor(
        sorted(
            value - image_start
            for value in protected
            if image_start <= value <= int(images[-1].item())
        ),
        dtype=torch.long,
        device=hidden.device,
    )
    if int(protected_local.numel()):
        selected_local = torch.cat((selected_local, protected_local)).unique().sort().values
    final_budget = active_spec.target_visual_tokens
    budget_guarded = final_budget is not None and int(
        selected_local.numel()
    ) < min(image_count, final_budget)
    if budget_guarded:
        selected_local = torch.arange(
            image_count,
            dtype=torch.long,
            device=hidden.device,
        )
    return _selection_result(
        hidden,
        images=images,
        selected_local=selected_local,
        protected=protected,
        stage="iac",
        completed_layers=completed_layers,
        media_unit=media_unit,
        source_layout=source_layout,
        result_layout=result_layout,
        extra_metadata={
            **rule_metadata(rules),
            "stage_index": stage_index,
            "hash_bits": active_spec.iac_hash_bits[stage_index],
            "random_seed": active_spec.random_seed + stage_index,
            "bucket_count": bucket_count,
            "protected_image_tokens": int(protected_local.numel()),
            "final_visual_budget": final_budget,
            "budget_guarded": budget_guarded,
        },
        hash_codes=codes,
    )


def finepruner_layer_averaged_head_scores(
    scorer_attentions: Sequence[Any],
    *,
    visual_count: int,
) -> Any:
    """Average last-text-query visual attention across ARP target layers."""

    torch = _torch()
    if not scorer_attentions:
        raise ValueError("FinePruner ARP requires target-layer attentions.")
    vectors: list[Any] = []
    reference_shape: tuple[int, int] | None = None
    for attention in scorer_attentions:
        if not isinstance(attention, torch.Tensor):
            raise TypeError("FinePruner scorer attention must be a torch tensor.")
        if attention.ndim == 2:
            vector = attention.float()
        elif attention.ndim == 4 and int(attention.shape[0]) == 1:
            if int(attention.shape[-1]) != visual_count:
                raise ValueError(
                    "Pre-sliced FinePruner attention must contain visual keys only."
                )
            vector = attention.float()[0, :, -1, :]
        else:
            raise ValueError("FinePruner attention must be [H,N] or [1,H,Q,N].")
        shape = (int(vector.shape[0]), int(vector.shape[1]))
        if shape[0] < 1 or shape[1] != visual_count:
            raise ValueError("FinePruner attention does not match the visual span.")
        if not bool(torch.isfinite(vector).all().item()):
            raise ValueError("FinePruner attention must be finite.")
        if reference_shape is not None and shape != reference_shape:
            raise ValueError("FinePruner target-layer attention shapes differ.")
        reference_shape = shape
        vectors.append(vector)
    return torch.stack([vector.to(vectors[0].device) for vector in vectors], dim=0).mean(dim=0)


def finepruner_attention_head_slice(
    *,
    query: Any,
    attention_keys: Any,
    image_key_indices: Any,
    attention_key_mask: Any | None = None,
) -> Any:
    """Reconstruct one layer's actual last-query image attention as ``[H,N]``."""

    torch = _torch()
    if not isinstance(query, torch.Tensor) or not isinstance(
        attention_keys, torch.Tensor
    ):
        raise TypeError("query and attention_keys must be torch tensors.")
    if query.ndim == 2:
        query = query.unsqueeze(0)
    if query.ndim != 3 or int(query.shape[0]) != 1:
        raise ValueError("FinePruner query must have shape [H,D] or [1,H,D].")
    if attention_keys.ndim != 3:
        raise ValueError("attention_keys must have shape [tokens,heads,head_dim].")
    if tuple(query.shape[1:]) != tuple(attention_keys.shape[1:]):
        raise ValueError("FinePruner query and key head shapes differ.")
    indices = torch.as_tensor(
        image_key_indices,
        dtype=torch.long,
        device=attention_keys.device,
    )
    if indices.ndim != 1 or int(indices.numel()) < 1:
        raise ValueError("image_key_indices must be a non-empty rank-one sequence.")
    values = [int(value) for value in indices.tolist()]
    if values != sorted(set(values)) or any(
        value < 0 or value >= int(attention_keys.shape[0]) for value in values
    ):
        raise ValueError("image_key_indices must be unique source-ordered key indices.")
    logits = torch.einsum("qhd,thd->qht", query, attention_keys)
    logits = logits * float(query.shape[-1]) ** -0.5
    if attention_key_mask is not None:
        mask = torch.as_tensor(
            attention_key_mask,
            dtype=torch.bool,
            device=attention_keys.device,
        )
        if mask.ndim == 1:
            mask = mask.unsqueeze(0)
        if tuple(mask.shape) != (1, int(attention_keys.shape[0])):
            raise ValueError("attention_key_mask must have shape [keys] or [1,keys].")
        if not bool(mask.any().item()):
            raise ValueError("attention_key_mask must expose at least one key.")
        logits = logits.masked_fill(~mask.unsqueeze(1), float("-inf"))
    probabilities = torch.softmax(logits, dim=-1, dtype=torch.float32)
    return probabilities[0].index_select(1, indices)


def finepruner_select_unbiased_heads(
    layer_averaged_attention: Any,
    *,
    head_ratio: float,
) -> tuple[Any, Any]:
    """Select the heads with smallest KL divergence from a uniform prior."""

    torch = _torch()
    if (
        not isinstance(layer_averaged_attention, torch.Tensor)
        or layer_averaged_attention.ndim != 2
    ):
        raise ValueError("layer_averaged_attention must be [H,N].")
    head_count, visual_count = (int(value) for value in layer_averaged_attention.shape)
    if head_count < 1 or visual_count < 1:
        raise ValueError("FinePruner ARP attention cannot be empty.")
    ratio = require_finite_real(head_ratio, "head_ratio")
    if not 0.0 < ratio <= 1.0:
        raise ValueError("head_ratio must be in (0, 1].")
    distribution = torch.softmax(layer_averaged_attention.float(), dim=-1)
    kl = (
        distribution
        * (
            torch.log(distribution.clamp_min(1e-12))
            + math.log(float(visual_count))
        )
    ).sum(dim=-1)
    selected_count = min(head_count, max(1, round(head_count * ratio)))
    selected = _stable_ascending_indices(kl)[:selected_count]
    return selected.to(torch.long), kl


def finepruner_arp_image_tokens(
    hidden_states: Any,
    scorer_attentions: Sequence[Any],
    *,
    image_indices: Any,
    completed_layers: int,
    initial_visual_tokens: int,
    spec: FinePrunerSpec | None = None,
    protected_indices: Iterable[int] = (),
    media_unit: str = "llm_token",
    source_layout: Mapping[str, Any] | None = None,
    result_layout: Mapping[str, Any] | None = None,
    rules: Mapping[str, Any] | None = None,
) -> FinePrunerResult:
    """Apply ARP after averaging layers and selecting least-biased heads."""

    torch = _torch()
    active_spec = spec or FinePrunerSpec()
    if completed_layers != active_spec.arp_apply_after_layer:
        raise ValueError(
            "FinePruner ARP must run after completed layer "
            f"{active_spec.arp_apply_after_layer}."
        )
    hidden, images, protected = _validate_inputs(
        hidden_states,
        image_indices=image_indices,
        protected_indices=protected_indices,
    )
    if initial_visual_tokens < int(images.numel()):
        raise ValueError("initial_visual_tokens cannot be smaller than the current span.")
    averaged = finepruner_layer_averaged_head_scores(
        scorer_attentions,
        visual_count=int(images.numel()),
    ).to(hidden.device)
    request = {"hidden_states": hidden, "image_indices": images,
        "scorer_attentions": scorer_attentions, "spec": active_spec}
    averaged = score_rule(rules, "scorer", request, averaged)
    selected_heads, head_kl = finepruner_select_unbiased_heads(
        averaged,
        head_ratio=active_spec.head_ratio,
    )
    custom_heads = selection_rule(rules, "head_selector",
        {**request, "scores": averaged, "head_kl": head_kl, "keep_count": int(selected_heads.numel())},
        count=int(averaged.shape[0]), keep_count=int(selected_heads.numel()), device=hidden.device)
    if custom_heads is not None:
        selected_heads = custom_heads
    token_scores = averaged.index_select(0, selected_heads).mean(dim=0)
    image_values = set(int(value) for value in images.tolist())
    target = min(
        int(images.numel()),
        max(
            active_spec.resolve_final_visual_count(int(images.numel())),
            sum(value in image_values for value in protected),
        ),
    )
    image_start = int(images[0].item())
    protected_local = torch.tensor(
        sorted(
            value - image_start
            for value in protected
            if image_start <= value <= int(images[-1].item())
        ),
        dtype=torch.long,
        device=hidden.device,
    )
    selected_mask = torch.zeros(
        int(images.numel()),
        dtype=torch.bool,
        device=hidden.device,
    )
    if int(protected_local.numel()):
        selected_mask[protected_local] = True
    remaining = target - int(selected_mask.sum().item())
    if remaining:
        ranked = _stable_descending_indices(
            token_scores.masked_fill(selected_mask, float("-inf"))
        )
        selected_mask[ranked[:remaining]] = True
    selected_local = selected_mask.nonzero(as_tuple=False).flatten()
    if target < int(images.numel()):
        custom = selection_rule(rules, "selector",
            {**request, "scores": token_scores, "selected_heads": selected_heads,
             "keep_count": target, "protected_indices": tuple(protected_local.tolist())},
            count=int(images.numel()), keep_count=target, device=hidden.device, protected=protected_local)
        if custom is not None:
            selected_local = custom.sort().values
    return _selection_result(
        hidden,
        images=images,
        selected_local=selected_local,
        protected=protected,
        stage="arp",
        completed_layers=completed_layers,
        media_unit=media_unit,
        source_layout=source_layout,
        result_layout=result_layout,
        extra_metadata={
            **rule_metadata(rules),
            "target_after_layers": active_spec.arp_target_after_layers,
            "target_visual_tokens": target,
            "selected_head_count": int(selected_heads.numel()),
            "protected_image_tokens": int(protected_local.numel()),
            "head_ratio": active_spec.head_ratio,
            "score_query": active_spec.score_query,
        },
        selected_heads=selected_heads,
        head_kl=head_kl,
        token_scores=token_scores,
    )


def _selection_result(
    hidden: Any,
    *,
    images: Any,
    selected_local: Any,
    protected: tuple[int, ...],
    stage: str,
    completed_layers: int,
    media_unit: str,
    source_layout: Mapping[str, Any] | None,
    result_layout: Mapping[str, Any] | None,
    extra_metadata: Mapping[str, Any],
    hash_codes: Any | None = None,
    selected_heads: Any | None = None,
    head_kl: Any | None = None,
    token_scores: Any | None = None,
) -> FinePrunerResult:
    torch = _torch()
    sequence_length = int(hidden.shape[0])
    kept_images = images.index_select(0, selected_local)
    image_set = set(int(value) for value in images.tolist())
    kept_image_set = set(int(value) for value in kept_images.tolist())
    protected_image = image_set.intersection(protected)
    if not protected_image.issubset(kept_image_set):
        raise ValueError("FinePruner may not drop a protected image token.")
    dropped = tuple(sorted(image_set - kept_image_set))
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
    metadata = {
        "algorithm": "finepruner",
        "variant": "finepruner.image",
        "stage": stage,
        "completed_layers": completed_layers,
        "source_tokens": sequence_length,
        "result_tokens": int(output.shape[0]),
        "source_image_tokens": int(images.numel()),
        "result_image_tokens": int(kept_images.numel()),
        "selected_indices": tuple(int(value) for value in kept.tolist()),
        "selected_image_indices": tuple(int(value) for value in kept_images.tolist()),
        "compression_applied": bool(dropped),
        **dict(extra_metadata),
    }
    return FinePrunerResult(
        hidden_states=output,
        kept_indices=kept,
        dropped_indices=dropped_tensor,
        kept_image_indices=kept_images,
        layout_delta=delta,
        metadata=metadata,
        hash_codes=hash_codes,
        selected_heads=selected_heads,
        head_kl=head_kl,
        token_scores=token_scores,
    )


def _validate_inputs(
    hidden_states: Any,
    *,
    image_indices: Any,
    protected_indices: Iterable[int],
) -> tuple[Any, Any, tuple[int, ...]]:
    torch = _torch()
    if (
        not isinstance(hidden_states, torch.Tensor)
        or hidden_states.ndim != 2
        or not hidden_states.is_floating_point()
    ):
        raise ValueError("hidden_states must be a floating rank-2 tensor.")
    if isinstance(image_indices, torch.Tensor):
        images = image_indices.to(device=hidden_states.device, dtype=torch.long)
    else:
        images = torch.tensor(
            tuple(image_indices),
            dtype=torch.long,
            device=hidden_states.device,
        )
    if images.ndim != 1 or int(images.numel()) < 1:
        raise ValueError("image_indices must be a non-empty rank-one sequence.")
    values = [int(value) for value in images.tolist()]
    if values != list(range(values[0], values[-1] + 1)):
        raise ValueError("FinePruner requires one contiguous serialized image span.")
    if values[0] < 0 or values[-1] >= int(hidden_states.shape[0]):
        raise ValueError("image_indices contains an index outside the sequence.")
    protected = tuple(sorted(set(int(value) for value in protected_indices)))
    if any(value < 0 or value >= int(hidden_states.shape[0]) for value in protected):
        raise ValueError("protected_indices contains an out-of-range index.")
    return hidden_states, images, protected


def _integer_tuple(
    value: Any,
    name: str,
    *,
    minimum: int | None,
) -> tuple[int, ...]:
    if isinstance(value, (str, bytes)):
        raise TypeError(f"{name} must be a sequence of integers.")
    try:
        result = tuple(value)
    except TypeError as exc:
        raise TypeError(f"{name} must be a sequence of integers.") from exc
    if any(isinstance(item, bool) or not isinstance(item, int) for item in result):
        raise TypeError(f"{name} must contain only integers.")
    if minimum is not None and any(item < minimum for item in result):
        raise ValueError(f"{name} values must be at least {minimum}.")
    return result


def _stable_descending_indices(values: Any) -> Any:
    torch = _torch()
    return torch.argsort(values, descending=True, stable=True)


def _stable_ascending_indices(values: Any) -> Any:
    torch = _torch()
    return torch.argsort(values, stable=True)


def _torch() -> Any:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - optional local dependency
        raise RuntimeError("FinePruner execution requires PyTorch.") from exc
    return torch


__all__ = [
    "FinePrunerResult",
    "FinePrunerSpec",
    "finepruner_arp_image_tokens",
    "finepruner_attention_head_slice",
    "finepruner_first_representatives",
    "finepruner_iac_image_tokens",
    "finepruner_layer_averaged_head_scores",
    "finepruner_lsh_codes",
    "finepruner_select_unbiased_heads",
]
