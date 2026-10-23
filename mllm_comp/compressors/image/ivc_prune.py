"""IVC-Prune coordinate-anchor and foreground token selection."""

from __future__ import annotations

from ._rules import rule_metadata, score_rule, selection_rule

from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Iterable, Mapping

from mllm_comp.core.layout import LayoutDelta, LayoutTokenUnit


@dataclass(frozen=True)
class IVCPruneSpec:
    prune_layer: int = 16
    total_retain_ratio: float = 0.5
    axis_ratio_each: float = 0.1
    semantic_seed_ratio: float = 0.01
    post_image_text_skip: int = 2
    strict_total_budget: bool = False
    min_visual_tokens: int = 1

    def __post_init__(self) -> None:
        if isinstance(self.prune_layer, bool) or not isinstance(self.prune_layer, int):
            raise TypeError("prune_layer must be an integer")
        if self.prune_layer < 1:
            raise ValueError("prune_layer must have a preceding value-capture layer")
        if not 0.0 < self.total_retain_ratio <= 1.0:
            raise ValueError("total_retain_ratio must be in (0, 1]")
        if not 0.0 <= self.axis_ratio_each <= 0.5:
            raise ValueError("axis_ratio_each must be in [0, 0.5]")
        if not 0.0 < self.semantic_seed_ratio <= 1.0:
            raise ValueError("semantic_seed_ratio must be in (0, 1]")
        if self.post_image_text_skip < 0 or self.min_visual_tokens < 1:
            raise ValueError("skip/minimum values are invalid")


@dataclass(frozen=True)
class IVCPruneOutput:
    hidden_states: Any
    kept_indices: Any
    kept_image_indices: Any
    selected_visual_indices: Any
    layout_delta: LayoutDelta
    metadata: Mapping[str, Any]


def project_ivc_value_states(hidden_states: Any, decoder_layer: Any) -> Any:
    """Project one decoder input to a flattened multi-head V space [L, H*D]."""

    import torch

    if not isinstance(hidden_states, torch.Tensor) or hidden_states.ndim != 2:
        raise ValueError("hidden_states must have shape [sequence, hidden]")
    normalised = decoder_layer.input_layernorm(hidden_states.unsqueeze(0))
    attention = decoder_layer.self_attn
    config = getattr(attention, "config", None)
    num_key_value_heads = int(
        getattr(
            attention,
            "num_key_value_heads",
            getattr(config, "num_key_value_heads", 0),
        )
    )
    num_attention_heads = int(
        getattr(
            attention,
            "num_heads",
            getattr(config, "num_attention_heads", 0),
        )
    )
    head_dim = int(
        getattr(
            attention,
            "head_dim",
            int(hidden_states.shape[-1]) // max(1, num_attention_heads),
        )
    )
    if num_key_value_heads < 1 or head_dim < 1:
        raise ValueError("decoder attention does not expose a valid value-head layout")
    values = attention.v_proj(normalised).view(
        1,
        int(hidden_states.shape[0]),
        num_key_value_heads,
        head_dim,
    )
    return values[0].reshape(int(hidden_states.shape[0]), -1)


def ivc_coordinate_scores(
    position_embeddings: tuple[Any, Any],
    *,
    image_indices: Any,
    sequence_length: int,
    rules: Mapping[str, Any] | None = None,
) -> tuple[Any, Any]:
    """Reduce native 1-D or M-RoPE cos/sin tensors to per-image-token scores."""

    import torch

    cosine, sine = position_embeddings
    image_indices = torch.as_tensor(image_indices, dtype=torch.long, device=cosine.device)
    result = (
        _coordinate_plane(cosine, image_indices, sequence_length, torch),
        _coordinate_plane(sine, image_indices, sequence_length, torch),
    )
    if rules and "coordinate_scorer" in rules:
        scores = score_rule(rules, "coordinate_scorer",
            {"position_embeddings": position_embeddings, "image_indices": image_indices,
             "sequence_length": sequence_length}, torch.stack(result))
        return scores[0], scores[1]
    return result


def ivc_prune_image_tokens(
    hidden_states: Any,
    *,
    image_indices: Any,
    cosine_scores: Any,
    sine_scores: Any,
    value_states: Any,
    text_query_indices: Any,
    before_layer: int,
    spec: IVCPruneSpec = IVCPruneSpec(),
    protected_indices: Iterable[int] = (),
    media_unit: str | LayoutTokenUnit = "llm_token",
    source_layout: Mapping[str, Any] | None = None,
    result_layout: Mapping[str, Any] | None = None,
    rules: Mapping[str, Any] | None = None,
) -> IVCPruneOutput:
    import math
    import torch

    if before_layer < spec.prune_layer:
        raise ValueError("IVC-Prune requires its completed value-capture layer before application")
    if not isinstance(hidden_states, torch.Tensor) or hidden_states.ndim != 2:
        raise ValueError("hidden_states must have shape [sequence, hidden]")
    sequence_length = int(hidden_states.shape[0])
    image_indices = torch.as_tensor(image_indices, dtype=torch.long, device=hidden_states.device)
    cosine_scores = torch.as_tensor(cosine_scores, device=hidden_states.device).float()
    sine_scores = torch.as_tensor(sine_scores, device=hidden_states.device).float()
    values = torch.as_tensor(value_states, device=hidden_states.device).float()
    text_queries = torch.as_tensor(text_query_indices, dtype=torch.long, device=hidden_states.device)
    visual_count = int(image_indices.numel())
    if image_indices.ndim != 1 or visual_count < 1:
        raise ValueError("image_indices must be a non-empty vector")
    if cosine_scores.shape != image_indices.shape or sine_scores.shape != image_indices.shape:
        raise ValueError("coordinate scores must match image_indices")
    if values.ndim != 2 or int(values.shape[0]) != sequence_length:
        raise ValueError("value_states must have shape [sequence, value_dim]")
    if text_queries.ndim != 1 or int(text_queries.numel()) < 1:
        raise ValueError("text_query_indices must contain prompt text")
    protected = _indices(protected_indices, sequence_length)
    protected_local = torch.nonzero(
        torch.isin(image_indices, torch.tensor(protected, dtype=torch.long, device=hidden_states.device)),
        as_tuple=False,
    ).flatten()

    visual_values = values.index_select(0, image_indices)
    request = {"hidden_states": hidden_states, "image_indices": image_indices,
        "value_states": values, "text_query_indices": text_queries,
        "cosine_scores": cosine_scores, "sine_scores": sine_scores, "spec": spec}
    seed_scores = _value_scores(values.index_select(0, text_queries), visual_values, torch)
    seed_scores = score_rule(rules, "semantic_scorer", {**request, "phase": "seed"}, seed_scores)
    seed_count = min(visual_count, max(1, int(round(visual_count * spec.semantic_seed_ratio))))
    seed_local = seed_scores.topk(seed_count, sorted=False).indices
    foreground_queries = torch.cat(
        (values.index_select(0, text_queries), visual_values.index_select(0, seed_local)),
        dim=0,
    )
    foreground_scores = _value_scores(foreground_queries, visual_values, torch)
    foreground_scores = score_rule(rules, "semantic_scorer",
        {**request, "phase": "foreground", "foreground_queries": foreground_queries, "seed_indices": seed_local}, foreground_scores)
    axis_count = min(visual_count, max(0, int(round(visual_count * spec.axis_ratio_each))))
    if axis_count:
        cosine_local = cosine_scores.topk(axis_count, sorted=False).indices
        sine_local = sine_scores.topk(axis_count, sorted=False).indices
        axis_local = torch.unique(torch.cat((cosine_local, sine_local)), sorted=False)
    else:
        axis_local = torch.empty(0, dtype=torch.long, device=hidden_states.device)
    target = min(
        visual_count,
        max(spec.min_visual_tokens, len(protected), int(round(visual_count * spec.total_retain_ratio))),
    )
    priority = torch.unique(torch.cat((axis_local, seed_local, protected_local)), sorted=False)
    if int(priority.numel()) > target and spec.strict_total_budget:
        strength = torch.maximum(cosine_scores, sine_scores) + foreground_scores
        protected_set = set(int(value) for value in protected_local.detach().cpu().tolist())
        optional = [int(value) for value in priority.detach().cpu().tolist() if int(value) not in protected_set]
        room = max(0, target - len(protected_set))
        optional_tensor = torch.tensor(optional, dtype=torch.long, device=hidden_states.device)
        chosen_optional = (
            optional_tensor.index_select(0, strength.index_select(0, optional_tensor).topk(room, sorted=False).indices)
            if room and int(optional_tensor.numel())
            else torch.empty(0, dtype=torch.long, device=hidden_states.device)
        )
        priority = torch.cat((protected_local, chosen_optional))
    remaining = max(0, target - int(priority.numel()))
    if remaining:
        ranked = foreground_scores.argsort(descending=True)
        candidates = ranked[~torch.isin(ranked, priority)]
        selected_local = torch.cat((priority, candidates[:remaining]))
    else:
        selected_local = priority
    selected_local = torch.unique(selected_local, sorted=True)
    if int(selected_local.numel()) < visual_count:
        custom = selection_rule(rules, "selector",
            {**request, "scores": foreground_scores, "seed_scores": seed_scores,
             "axis_indices": axis_local, "seed_indices": seed_local,
             "keep_count": int(selected_local.numel()), "protected_indices": tuple(protected_local.tolist())},
            count=visual_count, keep_count=int(selected_local.numel()), device=hidden_states.device, protected=protected_local)
        if custom is not None:
            selected_local = custom.sort().values
    kept_image = image_indices.index_select(0, selected_local)
    source_image_set = set(int(value) for value in image_indices.detach().cpu().tolist())
    kept_image_set = set(int(value) for value in kept_image.detach().cpu().tolist())
    kept = torch.tensor(
        [index for index in range(sequence_length) if index not in source_image_set or index in kept_image_set],
        dtype=torch.long,
        device=hidden_states.device,
    )
    kept_set = set(int(value) for value in kept.detach().cpu().tolist())
    delta = LayoutDelta.drop(
        sequence_length,
        (index for index in range(sequence_length) if index not in kept_set),
        media_unit=media_unit,
        protected_indices=protected,
        source_layout=source_layout,
        result_layout={**dict(result_layout or {}), "position_policy": "preserve_original"},
    )
    return IVCPruneOutput(
        hidden_states=hidden_states.index_select(0, kept),
        kept_indices=kept,
        kept_image_indices=kept_image,
        selected_visual_indices=selected_local,
        layout_delta=delta,
        metadata=MappingProxyType(
            {
                "algorithm": "ivc_prune",
                "variant": "ivc_prune.image",
                **rule_metadata(rules),
                **({"application_position": result_layout["application_position"]}
                   if result_layout and "application_position" in result_layout else {"before_layer": before_layer}),
                "visual_tokens_before": visual_count,
                "visual_tokens_after": int(selected_local.numel()),
                "axis_tokens": int(axis_local.numel()),
                "semantic_seed_tokens": int(seed_local.numel()),
                "foreground_fill_tokens": max(0, int(selected_local.numel()) - int(priority.numel())),
                "text_query_tokens": int(text_queries.numel()),
                "coordinate_source": "native_rope_cos_sin",
                "position_policy": "preserve_original",
                "strict_total_budget": spec.strict_total_budget,
            }
        ),
    )


def ivc_post_image_text_indices(
    image_mask: Any,
    *,
    attention_mask: Any | None = None,
    skip: int = 2,
) -> Any:
    """Return active post-image prompt positions with the released boundary skip."""

    import torch

    row = torch.as_tensor(image_mask).bool()
    image = row.nonzero(as_tuple=False).flatten()
    if not int(image.numel()):
        raise ValueError("image_mask contains no visual tokens")
    active = torch.ones_like(row)
    if attention_mask is not None:
        active = torch.as_tensor(attention_mask, device=row.device).reshape(-1).bool()
        if active.shape != row.shape:
            raise ValueError("attention_mask does not match image_mask")
    start = int(image[-1].item()) + 1 + skip
    candidates = torch.arange(start, int(row.numel()), device=row.device)
    candidates = candidates[active.index_select(0, candidates)] if int(candidates.numel()) else candidates
    if not int(candidates.numel()):
        start = int(image[-1].item()) + 1
        candidates = torch.arange(start, int(row.numel()), device=row.device)
        candidates = candidates[active.index_select(0, candidates)] if int(candidates.numel()) else candidates
    if not int(candidates.numel()):
        raise ValueError("IVC-Prune found no active post-image prompt text")
    return candidates.long()


def _coordinate_plane(value: Any, image_indices: Any, sequence_length: int, torch: Any) -> Any:
    if not isinstance(value, torch.Tensor) or value.ndim < 2:
        raise ValueError("native RoPE values must be tensors")
    sequence_axes = [axis for axis, size in enumerate(value.shape) if int(size) == sequence_length]
    if not sequence_axes:
        raise ValueError("could not locate the native RoPE sequence axis")
    axis = sequence_axes[-1]
    selected = value.detach().float().index_select(axis, image_indices.to(value.device))
    reduce_axes = tuple(index for index in range(selected.ndim) if index != axis)
    return selected.mean(dim=reduce_axes)


def _value_scores(query: Any, visual: Any, torch: Any) -> Any:
    import math

    similarity = query @ visual.T / math.sqrt(int(visual.shape[-1]))
    return torch.softmax(similarity, dim=-1).mean(dim=0)


def _indices(values: Iterable[int], upper: int) -> tuple[int, ...]:
    result = tuple(sorted(int(value) for value in values))
    if len(set(result)) != len(result) or any(value < 0 or value >= upper for value in result):
        raise ValueError("protected_indices contains an invalid index")
    return result


__all__ = [
    "IVCPruneOutput",
    "IVCPruneSpec",
    "ivc_coordinate_scores",
    "ivc_post_image_text_indices",
    "ivc_prune_image_tokens",
    "project_ivc_value_states",
]
