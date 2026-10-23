"""Backbone-neutral Nüwa two-stage image-token compression.

Stage 1 owns the released regional benchmark selection, semantic/spatial
aggregation, and exact source representatives.  Stage 2 owns the released
last-prompt-attention selection at one decoder boundary.  Backbone adapters
capture native visual metrics and legalize AnyRes, M-RoPE, DeepStack, masks,
and DynamicCache; no provider model class is replaced here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from types import MappingProxyType
from typing import Any, Mapping, Sequence

from mllm_comp.core.layout import LayoutDelta, LayoutTokenUnit

from .decoder_config import require_finite_real
from ._rules import rule_metadata, score_rule, selection_rule


NUWA_CONFIG_KEYS = frozenset(
    {
        "stage1_total_tokens",
        "stage2_after_layer",
        "stage2_retain_ratio",
        "stage2_score",
        "region_size",
        "top_n_per_region",
        "pillar_quantile",
        "distance_squared_ratio",
        "aggregation_self_mode",
        "semantic_hidden_start",
        "semantic_hidden_end",
        "random_stage2",
        "random_seed",
        "position_mode",
        "single_image_only",
    }
)


def merge_nuwa_executor_config(
    *,
    stage: Any,
    binding_config: Any,
) -> dict[str, Any]:
    """Merge Nüwa controls and fail closed on misspelled explicit keys."""

    result: dict[str, Any] = {}
    if stage is not None:
        for name in ("operation", "criterion", "schedule"):
            value = getattr(stage, name, None)
            if isinstance(value, Mapping):
                result.update(
                    {key: item for key, item in value.items() if key in NUWA_CONFIG_KEYS}
                )
    for source, value in (
        ("stage.config", getattr(stage, "config", None)),
        ("binding_config.config", getattr(binding_config, "config", None)),
    ):
        if not isinstance(value, Mapping):
            continue
        unknown = tuple(sorted(key for key in value if key not in NUWA_CONFIG_KEYS))
        if unknown:
            raise ValueError(f"Unknown Nüwa {source} keys: {list(unknown)!r}.")
        result.update(dict(value))
    return result


@dataclass(frozen=True)
class NuwaSpec:
    """Fixed fuxian/public-release recipe represented as a native contract."""

    stage1_total_tokens: int = 582
    stage2_after_layer: int = 16
    stage2_retain_ratio: float = 0.125
    stage2_score: str = "released_last_text_attention"
    region_size: int = 2
    top_n_per_region: int = 3
    pillar_quantile: float = 0.75
    distance_squared_ratio: float = 0.26
    aggregation_self_mode: str = "released_add"
    semantic_hidden_start: int = 16
    semantic_hidden_end: int = 25
    random_stage2: bool = False
    random_seed: int = 44
    position_mode: str = "rpme"
    single_image_only: bool = True

    def __post_init__(self) -> None:
        for name, minimum in (
            ("stage1_total_tokens", 1),
            ("stage2_after_layer", 1),
            ("region_size", 1),
            ("top_n_per_region", 1),
            ("semantic_hidden_start", 0),
            ("semantic_hidden_end", 1),
            ("random_seed", 0),
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be an integer.")
            if value < minimum:
                raise ValueError(f"{name} must be >= {minimum}.")
        retain = require_finite_real(
            self.stage2_retain_ratio, "stage2_retain_ratio"
        )
        pillar = require_finite_real(self.pillar_quantile, "pillar_quantile")
        distance = require_finite_real(
            self.distance_squared_ratio, "distance_squared_ratio"
        )
        if not 0.0 < retain <= 1.0:
            raise ValueError("stage2_retain_ratio must lie in (0, 1].")
        if not 0.0 <= pillar <= 1.0:
            raise ValueError("pillar_quantile must lie in [0, 1].")
        if not 0.0 < distance <= 1.0:
            raise ValueError("distance_squared_ratio must lie in (0, 1].")
        if self.semantic_hidden_end <= self.semantic_hidden_start:
            raise ValueError("Nüwa semantic hidden-state range is empty.")
        if self.stage2_score not in {
            "released_last_text_attention",
            "paper_text_cosine",
        }:
            raise ValueError("Unsupported Nüwa stage2_score.")
        if self.aggregation_self_mode not in {
            "released_add",
            "paper_normalized",
        }:
            raise ValueError("Unsupported Nüwa aggregation_self_mode.")
        if not isinstance(self.random_stage2, bool):
            raise TypeError("random_stage2 must be a bool.")
        if self.position_mode != "rpme":
            raise ValueError("Nüwa position_mode must be 'rpme'.")
        if self.single_image_only is not True:
            raise ValueError("The native Nüwa adaptation is single-image only.")
        object.__setattr__(self, "stage2_retain_ratio", retain)
        object.__setattr__(self, "pillar_quantile", pillar)
        object.__setattr__(self, "distance_squared_ratio", distance)

    def stage1_target(self, source_count: int, *, crop_count: int = 1) -> int:
        source = _positive_integer(source_count, "source_count")
        crops = _positive_integer(crop_count, "crop_count")
        return min(source, max(1, self.stage1_total_tokens // crops))

    @property
    def stage2_score_layer(self) -> int | None:
        """Only the released attention rule consumes a fixed decoder input."""
        return self.stage2_after_layer - 1 if not self.random_stage2 and self.stage2_score == "released_last_text_attention" else None

    def validate_decoder_depth(self, layer_count: int) -> None:
        count = _positive_integer(layer_count, "layer_count")
        if self.stage2_after_layer >= count:
            raise ValueError(
                "stage2_after_layer must follow completed layers and precede "
                "an existing decoder layer."
            )


@dataclass(frozen=True)
class NuwaStage1Result:
    tokens: Any
    selected_indices: Any
    selection_order: Any
    aggregation_weights: Any
    salience: Any
    key_norms: Any
    layout_delta: LayoutDelta
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class NuwaStage2Result:
    hidden_states: Any
    kept_indices: Any
    dropped_indices: Any
    kept_image_indices: Any
    selected_visual_indices: Any
    scores: Any
    layout_delta: LayoutDelta
    metadata: Mapping[str, Any] = field(default_factory=dict)


def nuwa_visual_salience(attention: Any, keys: Any) -> tuple[Any, Any]:
    """Return released no-CLS attention-received × key-norm salience."""

    torch = _torch()
    if not isinstance(attention, torch.Tensor) or not attention.is_floating_point():
        raise TypeError("Nüwa visual attention must be a floating tensor.")
    if not isinstance(keys, torch.Tensor) or not keys.is_floating_point():
        raise TypeError("Nüwa visual keys must be a floating tensor.")
    value = attention.detach().float()
    if value.ndim == 4:
        if int(value.shape[0]) != 1:
            raise ValueError("Nüwa visual attention batch must be one.")
        value = value[0]
    if value.ndim == 3:
        if int(value.shape[-2]) != int(value.shape[-1]):
            raise ValueError("Nüwa visual attention must be square.")
        received = value.mean(dim=(0, 1))
    elif value.ndim == 2:
        if int(value.shape[0]) != int(value.shape[1]):
            raise ValueError("Nüwa visual attention must be square.")
        received = value.mean(dim=0)
    elif value.ndim == 1:
        received = value
    else:
        raise ValueError("Nüwa visual attention must be rank one to four.")

    key_value = keys.detach().float()
    if key_value.ndim == 4:
        if int(key_value.shape[0]) != 1:
            raise ValueError("Nüwa visual key batch must be one.")
        key_value = key_value[0]
    if key_value.ndim == 3:
        # Native providers expose [heads, tokens, dim].
        if int(key_value.shape[1]) != int(received.numel()):
            raise ValueError("Nüwa visual key token count does not match attention.")
        token_keys = key_value.permute(1, 0, 2).reshape(
            int(key_value.shape[1]), -1
        )
        key_norms = token_keys.norm(p=2, dim=-1)
    elif key_value.ndim == 2:
        if int(key_value.shape[0]) != int(received.numel()):
            raise ValueError("Nüwa visual key token count does not match attention.")
        key_norms = key_value.norm(p=2, dim=-1)
    else:
        raise ValueError("Nüwa visual keys must be rank two to four.")
    return received * key_norms, key_norms


def regional_candidate_indices(
    salience: Any,
    *,
    grid_height: int,
    grid_width: int,
    region_size: int,
    top_n_per_region: int,
    rules: Mapping[str, Any] | None = None,
) -> Any:
    """Return the regional top-N union, including partial border regions."""

    torch = _torch()
    height = _positive_integer(grid_height, "grid_height")
    width = _positive_integer(grid_width, "grid_width")
    region = _positive_integer(region_size, "region_size")
    top_n = _positive_integer(top_n_per_region, "top_n_per_region")
    if not isinstance(salience, torch.Tensor) or salience.ndim != 1:
        raise ValueError("Nüwa salience must be a rank-one tensor.")
    if int(salience.numel()) != height * width:
        raise ValueError("Nüwa salience does not match its visual grid.")
    candidates: list[Any] = []
    for row in range(0, height, region):
        for col in range(0, width, region):
            rows = torch.arange(row, min(row + region, height), device=salience.device)
            cols = torch.arange(col, min(col + region, width), device=salience.device)
            local = (rows[:, None] * width + cols[None, :]).reshape(-1)
            count = min(top_n, int(local.numel()))
            order = torch.argsort(
                salience.index_select(0, local), descending=True, stable=True
            )[:count]
            selected = local.index_select(0, order)
            custom = selection_rule(rules, "regional_selector",
                {"scores": salience, "candidates": local, "keep_count": count,
                 "grid_height": height, "grid_width": width, "region_origin": (row, col)},
                count=height * width, keep_count=count, device=salience.device)
            if custom is not None:
                if not set(custom.tolist()).issubset(set(local.tolist())):
                    raise ValueError("Nüwa regional_selector must select within the current region.")
                selected = custom
            candidates.append(selected)
    if not candidates:
        raise ValueError("Nüwa produced no spatial regions.")
    return torch.cat(candidates).unique(sorted=True)


def nuwa_stage1_image_tokens(
    aggregation_features: Any,
    semantic_features: Any,
    attention: Any,
    keys: Any,
    *,
    grid_height: int,
    grid_width: int,
    spec: NuwaSpec | None = None,
    target_count: int | None = None,
    crop_count: int = 1,
    media_unit: str | LayoutTokenUnit = "image_feature",
    source_layout: Mapping[str, Any] | None = None,
    result_layout: Mapping[str, Any] | None = None,
    rules: Mapping[str, Any] | None = None,
) -> NuwaStage1Result:
    """Select regional benchmarks and aggregate compatible nearby tokens."""

    torch = _torch()
    _floating_matrix(aggregation_features, "aggregation_features", torch=torch)
    _floating_matrix(semantic_features, "semantic_features", torch=torch)
    source_count = int(aggregation_features.shape[0])
    if int(semantic_features.shape[0]) != source_count:
        raise ValueError("Nüwa aggregation and semantic token counts must match.")
    semantic_features = semantic_features.to(device=aggregation_features.device)
    height = _positive_integer(grid_height, "grid_height")
    width = _positive_integer(grid_width, "grid_width")
    if height * width != source_count:
        raise ValueError("Nüwa visual grid does not match feature tokens.")
    active = spec or NuwaSpec()
    resolved_target = (
        active.stage1_target(source_count, crop_count=crop_count)
        if target_count is None
        else min(source_count, _positive_integer(target_count, "target_count"))
    )
    salience, key_norms = nuwa_visual_salience(attention, keys)
    salience = salience.to(device=aggregation_features.device)
    key_norms = key_norms.to(device=aggregation_features.device)
    request = {"values": aggregation_features, "semantic_features": semantic_features,
               "attention": attention, "keys": keys, "key_norms": key_norms,
               "grid_height": height, "grid_width": width, "spec": active}
    salience = score_rule(rules, "stage1_scorer", request, salience)
    candidates = regional_candidate_indices(
        salience,
        grid_height=height,
        grid_width=width,
        region_size=active.region_size,
        top_n_per_region=active.top_n_per_region,
        rules=rules,
    )
    selected_count = min(resolved_target, int(candidates.numel()))
    candidate_scores = salience.index_select(0, candidates)
    local_order = torch.argsort(
        candidate_scores, descending=True, stable=True
    )[:selected_count]
    selection_order = candidates.index_select(0, local_order)
    custom = selection_rule(rules, "stage1_selector",
        {**request, "scores": salience, "candidates": candidates, "keep_count": selected_count},
        count=source_count, keep_count=selected_count, device=aggregation_features.device)
    if custom is not None:
        if not set(custom.tolist()).issubset(set(candidates.tolist())):
            raise ValueError("Nüwa stage1_selector must select regional candidates.")
        selection_order = custom
    selected = selection_order.sort().values

    semantic = torch.nn.functional.normalize(
        semantic_features.detach().float(), dim=-1, eps=1e-8
    )
    similarity = torch.relu(
        semantic.index_select(0, selected) @ semantic.transpose(0, 1)
    )
    rows = torch.arange(height, device=semantic.device).repeat_interleave(width)
    cols = torch.arange(width, device=semantic.device).repeat(height)
    coordinates = torch.stack((rows, cols), dim=-1).float()
    center_coordinates = coordinates.index_select(0, selected)
    distance = torch.cdist(center_coordinates, coordinates)
    max_distance = math.sqrt(max(1, height - 1) ** 2 + max(1, width - 1) ** 2)
    threshold = max_distance * math.sqrt(active.distance_squared_ratio)
    proximity = (1.0 - distance / max(threshold, 1e-8)).clamp(0.0, 1.0)
    weights = similarity * proximity

    benchmark_norms = key_norms.index_select(0, selected)
    pillar_threshold = torch.quantile(
        benchmark_norms.float(), active.pillar_quantile
    )
    pillars = benchmark_norms >= pillar_threshold
    if bool(pillars.any().item()):
        weights[pillars] = 0
    self_columns = selected.unsqueeze(-1)
    if active.aggregation_self_mode == "released_add":
        weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        weights.scatter_(1, self_columns, 1.0)
    else:
        weights.scatter_(1, self_columns, 1.0)
        weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-8)
    weights = score_rule(rules, "aggregation_weights",
        {**request, "selected_indices": selected, "scores": salience,
         "default_weights": weights, "similarity": similarity, "proximity": proximity}, weights)
    tokens = weights.to(aggregation_features.dtype) @ aggregation_features

    kept = tuple(int(value) for value in selected.detach().cpu().tolist())
    kept_set = set(kept)
    dropped = tuple(index for index in range(source_count) if index not in kept_set)
    delta = LayoutDelta.drop(
        source_count,
        dropped,
        media_unit=media_unit,
        source_layout=source_layout,
        result_layout=result_layout,
    )
    metadata = MappingProxyType(
        {
            "algorithm": "nuwa",
            "variant": "nuwa.image",
            "stage": "stage1",
            **rule_metadata(rules),
            "source_tokens": source_count,
            "regional_candidate_tokens": int(candidates.numel()),
            "requested_tokens": resolved_target,
            "result_tokens": int(tokens.shape[0]),
            "selected_indices": kept,
            "selection_order": tuple(
                int(value) for value in selection_order.detach().cpu().tolist()
            ),
            "pillar_tokens": None if rules and "aggregation_weights" in rules else int(pillars.sum().item()),
            "aggregation_self_mode": "custom_weights" if rules and "aggregation_weights" in rules else active.aggregation_self_mode,
            "aggregation_support": "overlapping_weighted_collectors",
            "layout_representatives": "selected_benchmark_source_indices",
            "position_mode": active.position_mode,
        }
    )
    return NuwaStage1Result(
        tokens=tokens,
        selected_indices=selected,
        selection_order=selection_order,
        aggregation_weights=weights,
        salience=salience,
        key_norms=key_norms,
        layout_delta=delta,
        metadata=metadata,
    )


def nuwa_stage2_image_tokens(
    hidden_states: Any,
    *,
    image_indices: Any,
    image_attention_scores: Any | None,
    spec: NuwaSpec | None = None,
    protected_indices: Sequence[int] = (),
    media_unit: str | LayoutTokenUnit = "llm_token",
    source_layout: Mapping[str, Any] | None = None,
    result_layout: Mapping[str, Any] | None = None,
    rules: Mapping[str, Any] | None = None,
) -> NuwaStage2Result:
    """Apply Nüwa Stage 2 at the configured completed-layer boundary."""

    torch = _torch()
    _floating_matrix(hidden_states, "hidden_states", torch=torch)
    active = spec or NuwaSpec()
    images = torch.as_tensor(
        image_indices, dtype=torch.long, device=hidden_states.device
    )
    if images.ndim != 1 or int(images.numel()) < 1:
        raise ValueError("Nüwa Stage 2 requires non-empty rank-one image indices.")
    if bool((images[1:] <= images[:-1]).any().item()) or int(images[0]) < 0 or int(
        images[-1]
    ) >= int(hidden_states.shape[0]):
        raise ValueError("Nüwa Stage 2 image indices must be sorted and in range.")
    if int(images[-1]) - int(images[0]) + 1 != int(images.numel()):
        raise ValueError("Nüwa Stage 2 requires one contiguous image span.")
    protected = tuple(sorted(set(int(value) for value in protected_indices)))
    if any(value < 0 or value >= int(hidden_states.shape[0]) for value in protected):
        raise ValueError("Nüwa protected index lies outside the sequence.")
    protected_set = set(protected)
    selectable_mask = torch.tensor(
        [int(value) not in protected_set for value in images.detach().cpu().tolist()],
        dtype=torch.bool,
        device=images.device,
    )
    selectable_images = images[selectable_mask]
    visual_count = int(selectable_images.numel())
    if visual_count < 1:
        raise ValueError("Nüwa Stage 2 found no selectable image tokens.")
    target = min(
        visual_count,
        max(1, int(math.ceil(visual_count * active.stage2_retain_ratio))),
    )
    if active.random_stage2:
        generator = torch.Generator(device=hidden_states.device)
        generator.manual_seed(active.random_seed)
        scores = torch.rand(
            visual_count,
            device=hidden_states.device,
            generator=generator,
        )
    elif active.stage2_score == "released_last_text_attention":
        if not isinstance(image_attention_scores, torch.Tensor) or not (
            image_attention_scores.is_floating_point()
        ):
            raise TypeError("Nüwa released Stage 2 requires floating attention scores.")
        scores = image_attention_scores.detach().reshape(-1).float().to(
            hidden_states.device
        )
        if int(scores.numel()) == int(images.numel()):
            scores = scores[selectable_mask]
        elif int(scores.numel()) != visual_count:
            raise ValueError("Nüwa Stage 2 scores do not match the image span.")
    else:
        image_set = set(int(value) for value in images.detach().cpu().tolist())
        text_indices = torch.tensor(
            [
                index
                for index in range(int(hidden_states.shape[0]))
                if index not in image_set
            ],
            dtype=torch.long,
            device=hidden_states.device,
        )
        if int(text_indices.numel()) < 1:
            raise ValueError("Nüwa paper Stage 2 found no text tokens.")
        query = torch.nn.functional.normalize(
            hidden_states.index_select(0, text_indices).float().mean(dim=0),
            dim=0,
            eps=1e-8,
        )
        scores = torch.nn.functional.normalize(
            hidden_states.index_select(0, selectable_images).float(),
            dim=-1,
            eps=1e-8,
        ) @ query
    request = {"hidden_states": hidden_states, "image_indices": images,
               "selectable_indices": selectable_images, "image_attention_scores": image_attention_scores,
               "keep_count": target, "spec": active}
    if target < visual_count:
        scores = score_rule(rules, "stage2_scorer", request, scores)
    selected_local = torch.argsort(scores, descending=True, stable=True)[:target]
    if target < visual_count:
        custom = selection_rule(rules, "stage2_selector", {**request, "scores": scores},
            count=visual_count, keep_count=target, device=hidden_states.device)
        if custom is not None:
            selected_local = custom
    selected_local = selected_local.sort().values
    selected_images = selectable_images.index_select(0, selected_local)
    kept_images = torch.tensor(
        sorted(
            set(int(value) for value in selected_images.detach().cpu().tolist())
            | (set(int(value) for value in images.detach().cpu().tolist()) & protected_set)
        ),
        dtype=torch.long,
        device=images.device,
    )
    image_set = set(int(value) for value in images.detach().cpu().tolist())
    kept_image_set = set(
        int(value) for value in kept_images.detach().cpu().tolist()
    )
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
            protected_set
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
    metadata = MappingProxyType(
        {
            "algorithm": "nuwa",
            "variant": "nuwa.image",
            "stage": "stage2",
            **rule_metadata(rules),
            "stage2_after_layer": active.stage2_after_layer,
            "score_mode": "custom" if rules and "stage2_scorer" in rules else active.stage2_score,
            "retain_ratio": active.stage2_retain_ratio,
            "source_tokens": int(hidden_states.shape[0]),
            "result_tokens": int(output.shape[0]),
            "source_image_tokens": visual_count,
            "source_image_span_tokens": int(images.numel()),
            "result_image_tokens": int(kept_images.numel()),
            "selected_indices": tuple(int(value) for value in kept.tolist()),
            "selected_image_indices": tuple(
                int(value) for value in kept_images.tolist()
            ),
            "position_mode": active.position_mode,
            "position_policy": "preserve_stage1_rpme",
            "cache_policy": "compact_prefix",
        }
    )
    return NuwaStage2Result(
        hidden_states=output,
        kept_indices=kept,
        dropped_indices=torch.tensor(
            dropped_values, dtype=torch.long, device=hidden_states.device
        ),
        kept_image_indices=kept_images,
        selected_visual_indices=selected_local,
        scores=scores,
        layout_delta=delta,
        metadata=metadata,
    )


def relative_position_mapping(
    selected_indices: Any,
    *,
    original_length: int,
    visual_start: int,
) -> Any:
    """Nüwa RPME: extend selected source positions over the original range."""

    torch = _torch()
    selected = torch.as_tensor(selected_indices, dtype=torch.long)
    if selected.ndim != 1 or int(selected.numel()) < 1:
        raise ValueError("Nüwa RPME requires non-empty rank-one indices.")
    if bool((selected[1:] < selected[:-1]).any().item()):
        raise ValueError("Nüwa RPME indices must be sorted.")
    original = _positive_integer(original_length, "original_length")
    start = _non_negative_integer(visual_start, "visual_start")
    if int(selected[0]) < 0 or int(selected[-1]) >= original:
        raise ValueError("Nüwa RPME index lies outside the original visual range.")
    result = torch.empty_like(selected)
    result[0] = start
    if int(selected.numel()) > 1:
        denominator = max(1, int(selected[-1]))
        span = max(0, original - 1)
        result[1:] = (
            torch.ceil(selected[1:].float() * float(span) / float(denominator))
            .long()
            .add(start)
        )
    return result


def build_nuwa_prefill_position_ids(
    sequence_length: int,
    *,
    visual_span: tuple[int, int],
    selected_indices: Any,
    original_visual_length: int,
    device: Any,
) -> Any:
    """Build one-dimensional RPME positions and shift the text suffix."""

    torch = _torch()
    length = _positive_integer(sequence_length, "sequence_length")
    start, end = (int(value) for value in visual_span)
    selected = torch.as_tensor(selected_indices, dtype=torch.long, device=device)
    if not 0 <= start < end <= length or end - start != int(selected.numel()):
        raise ValueError("Nüwa RPME visual span does not match selected indices.")
    original = _positive_integer(original_visual_length, "original_visual_length")
    positions = torch.arange(length, dtype=torch.long, device=device)
    positions[start:end] = relative_position_mapping(
        selected,
        original_length=original,
        visual_start=start,
    ).to(device)
    suffix_length = length - end
    if suffix_length:
        positions[end:] = torch.arange(
            start + original,
            start + original + suffix_length,
            dtype=torch.long,
            device=device,
        )
    return positions


def _floating_matrix(value: Any, name: str, *, torch: Any) -> None:
    if (
        not isinstance(value, torch.Tensor)
        or value.ndim != 2
        or not value.is_floating_point()
    ):
        raise ValueError(f"{name} must be a floating rank-two tensor.")


def _positive_integer(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer.")
    if value < 1:
        raise ValueError(f"{name} must be positive.")
    return value


def _non_negative_integer(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer.")
    if value < 0:
        raise ValueError(f"{name} must be non-negative.")
    return value


def _torch() -> Any:
    try:
        import torch
    except ModuleNotFoundError as exc:  # pragma: no cover - optional runtime
        raise RuntimeError("Nüwa requires PyTorch at execution time.") from exc
    return torch


__all__ = [
    "NUWA_CONFIG_KEYS",
    "NuwaSpec",
    "NuwaStage1Result",
    "NuwaStage2Result",
    "build_nuwa_prefill_position_ids",
    "merge_nuwa_executor_config",
    "nuwa_stage1_image_tokens",
    "nuwa_stage2_image_tokens",
    "nuwa_visual_salience",
    "regional_candidate_indices",
    "relative_position_mapping",
]
