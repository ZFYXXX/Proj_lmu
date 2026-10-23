"""FEATHER two-stage no-RoPE decoder token pruning."""

from __future__ import annotations

from ._rules import rule_metadata, score_rule, selection_rule

from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Iterable, Mapping

from mllm_comp.core.layout import LayoutDelta, LayoutTokenUnit


@dataclass(frozen=True)
class FEATHERSpec:
    stage_layers: tuple[int, int] = (8, 16)
    reduction_ratio: float = 0.7
    uniform_stride: int = 3
    target_final_retain_ratio: float | None = None
    min_visual_tokens: int = 1

    def __post_init__(self) -> None:
        if len(self.stage_layers) != 2 or tuple(sorted(set(self.stage_layers))) != self.stage_layers:
            raise ValueError("stage_layers must contain two increasing layer indices")
        if self.stage_layers[0] < 1:
            raise ValueError("FEATHER stages must have a preceding scorer layer")
        if not 0.0 < self.reduction_ratio < 1.0:
            raise ValueError("reduction_ratio must be in (0, 1)")
        if self.uniform_stride < 1 or self.min_visual_tokens < 1:
            raise ValueError("uniform_stride and min_visual_tokens must be positive")
        if self.target_final_retain_ratio is not None and not 0.0 < self.target_final_retain_ratio <= 1.0:
            raise ValueError("target_final_retain_ratio must be in (0, 1]")

    def stage_index(self, before_layer: int) -> int:
        try:
            return self.stage_layers.index(before_layer)
        except ValueError as exc:
            raise ValueError("before_layer is not a FEATHER stage") from exc


@dataclass(frozen=True)
class FEATHEROutput:
    hidden_states: Any
    kept_indices: Any
    kept_image_indices: Any
    selected_visual_indices: Any
    layout_delta: LayoutDelta
    metadata: Mapping[str, Any]


def feather_no_rope_scores(
    hidden_states: Any,
    decoder_layer: Any,
    *,
    image_indices: Any,
    query_indices: Any,
    rules: Mapping[str, Any] | None = None,
) -> Any:
    """Last-prompt Q-to-visual-K attention with rotary embedding omitted."""

    import math
    import torch

    if not isinstance(hidden_states, torch.Tensor) or hidden_states.ndim != 2:
        raise ValueError("hidden_states must have shape [sequence, hidden]")
    image_indices = torch.as_tensor(image_indices, dtype=torch.long, device=hidden_states.device)
    query_indices = torch.as_tensor(query_indices, dtype=torch.long, device=hidden_states.device)
    if image_indices.ndim != 1 or not int(image_indices.numel()):
        raise ValueError("image_indices must be a non-empty vector")
    if query_indices.ndim != 1 or not int(query_indices.numel()):
        raise ValueError("query_indices must be a non-empty vector")
    normalised = decoder_layer.input_layernorm(hidden_states.unsqueeze(0))
    attention = decoder_layer.self_attn
    length = int(hidden_states.shape[0])
    config = getattr(attention, "config", None)
    num_heads = int(
        getattr(
            attention,
            "num_heads",
            getattr(config, "num_attention_heads", 0),
        )
    )
    num_key_value_heads = int(
        getattr(
            attention,
            "num_key_value_heads",
            getattr(config, "num_key_value_heads", 0),
        )
    )
    head_dim = int(
        getattr(
            attention,
            "head_dim",
            int(hidden_states.shape[-1]) // max(1, num_heads),
        )
    )
    if num_heads < 1 or num_key_value_heads < 1 or head_dim < 1:
        raise ValueError("decoder attention does not expose a valid head layout")
    query = attention.q_proj(normalised).view(
        1, length, num_heads, head_dim
    ).transpose(1, 2)
    keys = attention.k_proj(normalised).view(
        1, length, num_key_value_heads, head_dim
    ).transpose(1, 2)
    if callable(getattr(attention, "q_norm", None)):
        query = attention.q_norm(query)
    if callable(getattr(attention, "k_norm", None)):
        keys = attention.k_norm(keys)
    groups = int(
        getattr(
            attention,
            "num_key_value_groups",
            num_heads // num_key_value_heads,
        )
    )
    if groups > 1:
        keys = torch.repeat_interleave(keys, groups, dim=1)
    # The release uses the final prompt token.  Accepting a query vector keeps
    # adapter-side masking exact; the final active entry is selected here.
    query_index = int(query_indices[-1].item())
    scores = torch.matmul(
        query[:, :, query_index : query_index + 1],
        keys.index_select(2, image_indices).transpose(2, 3),
    ) * float(getattr(attention, "scaling", 1.0 / math.sqrt(head_dim)))
    result = torch.softmax(scores.float(), dim=-1).mean(dim=1)[0, 0]
    return score_rule(rules, "scorer",
        {"hidden_states": hidden_states, "query": query, "keys": keys,
         "image_indices": image_indices, "query_indices": query_indices,
         "logits": scores, "decoder_layer": decoder_layer}, result)


def feather_image_tokens(
    hidden_states: Any,
    *,
    image_indices: Any,
    image_attention_scores: Any,
    before_layer: int,
    initial_visual_tokens: int,
    spec: FEATHERSpec = FEATHERSpec(),
    protected_indices: Iterable[int] = (),
    media_unit: str | LayoutTokenUnit = "llm_token",
    source_layout: Mapping[str, Any] | None = None,
    result_layout: Mapping[str, Any] | None = None,
    rules: Mapping[str, Any] | None = None,
) -> FEATHEROutput:
    import math
    import torch

    if not isinstance(hidden_states, torch.Tensor) or hidden_states.ndim != 2:
        raise ValueError("hidden_states must have shape [sequence, hidden]")
    sequence_length = int(hidden_states.shape[0])
    image_indices = torch.as_tensor(image_indices, dtype=torch.long, device=hidden_states.device)
    scores = torch.as_tensor(image_attention_scores, device=hidden_states.device).float()
    if image_indices.ndim != 1 or scores.shape != image_indices.shape:
        raise ValueError("FEATHER image indices and scores must be matching vectors")
    if initial_visual_tokens < int(image_indices.numel()):
        raise ValueError("initial_visual_tokens cannot be smaller than the current image span")
    stage = spec.stage_index(before_layer)
    current = int(image_indices.numel())
    protected = _indices(protected_indices, sequence_length)
    protected_local = torch.nonzero(
        torch.isin(image_indices, torch.tensor(protected, dtype=torch.long, device=hidden_states.device)),
        as_tuple=False,
    ).flatten()
    retained = 1.0 - spec.reduction_ratio
    if stage == 0:
        final_floor = (
            0
            if spec.target_final_retain_ratio is None
            else int(round(initial_visual_tokens * spec.target_final_retain_ratio))
        )
        attention_target = min(
            current,
            max(spec.min_visual_tokens, int(current * retained), final_floor),
        )
        ranked = scores.topk(attention_target, sorted=False).indices
        uniform = _uniform_indices(current, spec.uniform_stride, hidden_states.device, torch)
        selected_local = torch.unique(torch.cat((ranked, uniform, protected_local)), sorted=True)
        uniform_added = int(selected_local.numel()) - int(torch.unique(torch.cat((ranked, protected_local))).numel())
        target_mode = "released_attention_union_uniform"
    else:
        if spec.target_final_retain_ratio is None:
            target = int(current * retained * retained)
            target_mode = "released_squared_remaining_ratio"
        else:
            target = int(round(initial_visual_tokens * spec.target_final_retain_ratio))
            target_mode = "original_span_target"
        target = min(current, max(spec.min_visual_tokens, target, int(protected_local.numel())))
        # Protected items can consume part of this budget, so the optional
        # prefix below must still contain the highest-scoring candidates.
        ranked = scores.topk(target, sorted=True).indices
        selected_local = torch.unique(torch.cat((ranked, protected_local)), sorted=True)
        if int(selected_local.numel()) > target:
            protected_set = set(int(value) for value in protected_local.detach().cpu().tolist())
            optional = [int(value) for value in ranked.detach().cpu().tolist() if int(value) not in protected_set]
            chosen = sorted((*protected_set, *optional[: max(0, target - len(protected_set))]))
            selected_local = torch.tensor(chosen, dtype=torch.long, device=hidden_states.device)
        attention_target = target
        uniform_added = 0
    if int(selected_local.numel()) < current:
        custom = selection_rule(rules, "selector",
            {"hidden_states": hidden_states, "image_indices": image_indices, "scores": scores,
             "stage_index": stage, "keep_count": int(selected_local.numel()),
             "attention_target": attention_target, "protected_indices": tuple(protected_local.tolist()), "spec": spec},
            count=current, keep_count=int(selected_local.numel()), device=hidden_states.device, protected=protected_local)
        if custom is not None:
            selected_local = custom.sort().values
            target_mode = "custom_selection"
            uniform_added = 0
    kept_image = image_indices.index_select(0, selected_local)
    selected_set = set(int(value) for value in kept_image.detach().cpu().tolist())
    kept = torch.tensor(
        [index for index in range(sequence_length) if index not in set(image_indices.detach().cpu().tolist()) or index in selected_set],
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
    return FEATHEROutput(
        hidden_states=hidden_states.index_select(0, kept),
        kept_indices=kept,
        kept_image_indices=kept_image,
        selected_visual_indices=selected_local,
        layout_delta=delta,
        metadata=MappingProxyType(
            {
                "algorithm": "feather",
                "variant": "feather.image",
                **rule_metadata(rules),
                "stage": stage + 1,
                **({"application_position": result_layout["application_position"]}
                   if result_layout and "application_position" in result_layout else {"before_layer": before_layer}),
                "visual_tokens_before": current,
                "visual_tokens_after": int(selected_local.numel()),
                "attention_tokens": attention_target,
                "uniform_tokens_added": uniform_added,
                "target_mode": target_mode,
                "score_source": "custom" if rules and "scorer" in rules else "last_prompt_no_rope",
                "position_policy": "preserve_original",
            }
        ),
    )


def _uniform_indices(count: int, stride: int, device: Any, torch: Any) -> Any:
    import math

    width = max(1, int(math.ceil(math.sqrt(count))))
    height = max(1, int(math.ceil(float(count) / float(width))))
    rows = torch.arange(0, height, stride, device=device)
    columns = torch.arange(0, width, stride, device=device)
    values = (rows[:, None] * width + columns[None, :]).reshape(-1)
    return values[values < count].long()


def _indices(values: Iterable[int], upper: int) -> tuple[int, ...]:
    result = tuple(sorted(int(value) for value in values))
    if len(set(result)) != len(result) or any(value < 0 or value >= upper for value in result):
        raise ValueError("protected_indices contains an invalid index")
    return result


__all__ = ["FEATHEROutput", "FEATHERSpec", "feather_image_tokens", "feather_no_rope_scores"]
