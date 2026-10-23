"""RedundancyLens operation reductions for the OneVision Qwen2 decoder.

The public RedundancyLens repository releases Qwen2-VL model-side changes but
does not release its search-result JSON files or a LLaVA-OneVision module.  The
helpers below reproduce the released module semantics without inventing a
layer ranking:

* Hollow Attention replaces global visual-to-visual causal attention with a
  preceding local window while preserving every other causal-mask entry.
* Probe-Activated Dynamic FFN samples 10% of visual-token intermediate states,
  ranks channels by mean absolute activation, and keeps the top 20% only for
  visual tokens.  Text-token channels remain dense.

Layer lists are explicit configuration inputs and must be produced by a
separate target-checkpoint calibration.  This keeps the 28-layer OV-Qwen2
adaptation auditable and avoids silently copying Qwen2-VL search rankings.
"""

from __future__ import annotations

import math
import os
from typing import Any, Dict, Mapping, Optional, Sequence

import torch
from torch import nn
from torch.nn import functional as F

from .qwen2_flops import aggregate_samplewise_flops


DEFAULT_CONFIG: Dict[str, Any] = {
    "method": "redundancy_lens",
    "attention_reduction_layers": [],
    "ffn_reduction_layers": [],
    "attention_range": 256,
    "probe_ratio": 0.1,
    "ffn_keep_ratio": 0.2,
    "prefill_only": True,
    "layer_ranking_source": "target_checkpoint_calibration_required",
    "operation_backend": "released_dense_mask_semantics",
    "image_only": True,
    "log_first_n": 1,
}


def build_hybrid_layer_priority(
    *,
    layer_count: int,
    preassigned_count: int,
    searched_priority: Sequence[int],
) -> list[int]:
    """Build the paper's hybrid ranking without fabricating missing stages.

    The last ``Lp`` layers are pre-ranked from the final layer backwards.  The
    remaining prefix is ordered by Algorithm 1 on the target checkpoint.  A
    partial ``searched_priority`` is valid (formal 0.67 policies need only the
    prefix they consume), but every supplied layer must belong to the searched
    prefix and must be unique.
    """

    layers = int(layer_count)
    preassigned = int(preassigned_count)
    if layers < 1 or not 0 <= preassigned <= layers:
        raise ValueError("Invalid RedundancyLens hybrid layer geometry")
    fixed = list(range(layers - 1, layers - preassigned - 1, -1))
    searched = [int(layer) for layer in searched_priority]
    if len(set(searched)) != len(searched):
        raise ValueError("RedundancyLens searched priority contains duplicates")
    search_limit = layers - preassigned
    if any(layer < 0 or layer >= search_limit for layer in searched):
        raise ValueError(
            "RedundancyLens searched priority must stay outside preassigned layers"
        )
    return fixed + searched


def penalized_ranking_score(
    candidate_performance: float,
    baseline_performance: float,
    *,
    penalty_alpha: float = 2.0,
) -> tuple[float, float]:
    """Apply the paper's penalty to a validation-performance difference."""

    alpha = float(penalty_alpha)
    if alpha <= 1.0:
        raise ValueError("RedundancyLens penalty alpha must be greater than one")
    delta = float(candidate_performance) - float(baseline_performance)
    return (delta * alpha if delta < 0.0 else delta), delta


def _layer_list(value: Any, name: str) -> list[int]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError(f"{name} must be a sequence of layer indices")
    result = [int(item) for item in value]
    if any(item < 0 for item in result):
        raise ValueError(f"{name} layer indices must be non-negative")
    if len(set(result)) != len(result):
        raise ValueError(f"{name} contains duplicate layer indices")
    return sorted(result)


def normalize_redundancy_lens_config(
    raw: Optional[Mapping[str, Any]],
) -> Dict[str, Any]:
    cfg = dict(DEFAULT_CONFIG)
    if raw:
        cfg.update(dict(raw))
    if str(cfg.get("method", "")).lower() != "redundancy_lens":
        raise ValueError("RedundancyLens config must declare method=redundancy_lens")
    cfg["method"] = "redundancy_lens"
    cfg["attention_reduction_layers"] = _layer_list(
        cfg.get("attention_reduction_layers"), "attention_reduction_layers"
    )
    cfg["ffn_reduction_layers"] = _layer_list(
        cfg.get("ffn_reduction_layers"), "ffn_reduction_layers"
    )
    if not cfg["attention_reduction_layers"] and not cfg["ffn_reduction_layers"]:
        raise ValueError(
            "RedundancyLens requires calibrated attention and/or FFN layers"
        )
    cfg["attention_range"] = int(cfg["attention_range"])
    if cfg["attention_range"] < 1:
        raise ValueError("attention_range must be positive")
    for key in ("probe_ratio", "ffn_keep_ratio"):
        cfg[key] = float(cfg[key])
        if not 0.0 < cfg[key] <= 1.0:
            raise ValueError(f"{key} must satisfy 0 < ratio <= 1")
    if not bool(cfg.get("prefill_only", True)):
        raise ValueError(
            "RedundancyLens visual operations are prefill-only in OneVision"
        )
    cfg["prefill_only"] = True
    if str(cfg.get("operation_backend")) not in {
        "released_dense_mask_semantics",
        "indexed_structural",
    }:
        raise ValueError("Unsupported RedundancyLens operation_backend")
    cfg["image_only"] = bool(cfg.get("image_only", True))
    cfg["log_first_n"] = int(cfg.get("log_first_n", 1))
    if cfg["log_first_n"] < 0:
        raise ValueError("log_first_n must be non-negative")
    return cfg


def visual_token_mask(
    runtime: Optional[Mapping[str, Any]],
    sequence_length: int,
    *,
    device: Optional[torch.device] = None,
) -> Optional[torch.Tensor]:
    """Return the exact expanded visual-token mask for a batch-one prefill."""

    if not runtime or not runtime.get("has_image") or int(sequence_length) <= 1:
        return None
    spans = runtime.get("expanded_visual_spans")
    if not isinstance(spans, list) or not spans:
        return None
    mask = torch.zeros(int(sequence_length), dtype=torch.bool, device=device)
    for raw_span in spans:
        if not isinstance(raw_span, (list, tuple)) or len(raw_span) != 2:
            raise ValueError("RedundancyLens visual span must be a start/end pair")
        start, end = int(raw_span[0]), int(raw_span[1])
        if not 0 <= start < end <= int(sequence_length):
            raise ValueError(
                f"Invalid RedundancyLens visual span {(start, end)} for length "
                f"{sequence_length}"
            )
        mask[start:end] = True
    return mask if bool(mask.any()) else None


def prepare_hollow_attention_mask(
    attention_mask: torch.Tensor,
    *,
    runtime: Optional[Mapping[str, Any]],
    attention_range: int,
) -> torch.Tensor:
    """Apply the released Hollow-Attention pattern to a 4-D additive mask.

    For visual query/key pairs, only self and the preceding ``RA-1`` sequence
    positions remain available.  Text-to-visual, visual-to-text and text-to-
    text entries are copied byte-for-byte from the original causal mask.
    """

    if attention_mask is None or attention_mask.ndim != 4:
        raise ValueError("Hollow Attention requires a 4-D additive mask")
    if int(attention_mask.shape[0]) != 1:
        raise ValueError("RedundancyLens currently requires batch_size=1")
    q_len, kv_len = map(int, attention_mask.shape[-2:])
    if q_len != kv_len or q_len <= 1:
        return attention_mask
    visual = visual_token_mask(runtime, q_len, device=attention_mask.device)
    if visual is None:
        return attention_mask
    radius = int(attention_range)
    if radius < 1:
        raise ValueError("attention_range must be positive")

    hollow = attention_mask.clone()
    minimum = torch.finfo(hollow.dtype).min
    visual_indices = visual.nonzero(as_tuple=True)[0]
    # Building query/key distances for the entire decoder sequence is
    # needlessly quadratic in all (mostly textual) tokens.  Only visual-to-
    # visual entries can change, so update that submatrix in bounded row
    # chunks.  The two comparisons are exactly equivalent to
    # ``key <= query and query - key < radius`` and avoid an int64 VxV delta.
    query_chunk_size = max(
        1,
        int(os.environ.get("OV_REDUNDANCY_LENS_MASK_QUERY_CHUNK_SIZE", "128")),
    )
    visual_keys = visual_indices[None, :]
    target = hollow[0, 0]
    for start in range(0, int(visual_indices.numel()), query_chunk_size):
        query_indices = visual_indices[start : start + query_chunk_size]
        visual_queries = query_indices[:, None]
        blocked = visual_keys > visual_queries
        blocked |= visual_keys <= (visual_queries - radius)
        target_rows = target.index_select(0, query_indices)
        visual_block = target_rows.index_select(1, visual_indices)
        visual_block.masked_fill_(blocked, minimum)
        target_rows.index_copy_(1, visual_indices, visual_block)
        target.index_copy_(0, query_indices, target_rows)
    if isinstance(runtime, dict):
        runtime["redundancy_lens_attention_mask_backend"] = (
            "visual_block_chunked"
        )
        runtime["redundancy_lens_attention_mask_query_chunk_size"] = (
            query_chunk_size
        )
    return hollow


def apply_probe_activated_ffn_mask(
    intermediate_states: torch.Tensor,
    *,
    cfg: Optional[Mapping[str, Any]],
    runtime: Optional[Dict[str, Any]],
) -> torch.Tensor:
    """Mask visual FFN channels using the released activation probe rule."""

    if cfg is None or intermediate_states.ndim != 3:
        return intermediate_states
    if int(intermediate_states.shape[0]) != 1:
        raise ValueError("RedundancyLens currently requires batch_size=1")
    visual = visual_token_mask(
        runtime,
        int(intermediate_states.shape[1]),
        device=intermediate_states.device,
    )
    if visual is None:
        return intermediate_states
    visual_states = intermediate_states[0, visual]
    visual_count, channel_count = map(int, visual_states.shape)
    sample_count = max(1, int(math.floor(visual_count * float(cfg["probe_ratio"]))))
    sample_count = min(sample_count, visual_count)
    keep_count = max(1, int(math.floor(channel_count * float(cfg["ffn_keep_ratio"]))))
    keep_count = min(keep_count, channel_count)

    sampled_rows = torch.randperm(
        visual_count, device=intermediate_states.device
    )[:sample_count]
    importance = visual_states.index_select(0, sampled_rows).abs().mean(dim=0)
    keep_channels = torch.topk(
        importance, k=keep_count, largest=True, sorted=False
    ).indices

    masked_visual = torch.zeros_like(visual_states)
    masked_visual[:, keep_channels] = visual_states[:, keep_channels]
    result = intermediate_states.clone()
    result[0, visual] = masked_visual
    if isinstance(runtime, dict):
        runtime["redundancy_lens_probe_visual_tokens"] = sample_count
        runtime["redundancy_lens_ffn_channels_before"] = channel_count
        runtime["redundancy_lens_ffn_channels_after"] = keep_count
        runtime["redundancy_lens_ffn_actual_keep_ratio"] = (
            keep_count / float(channel_count)
        )
        runtime["redundancy_lens_ffn_mask_calls"] = int(
            runtime.get("redundancy_lens_ffn_mask_calls", 0)
        ) + 1
        runtime[
            "redundancy_lens_ffn_operation_backend_executed"
        ] = "released_dense_mask_semantics"
    return result


def _structural_linear_kind(module: nn.Module) -> Optional[str]:
    """Return the exact structural-projection backend supported by ``module``.

    RedundancyLens selects output channels after an activation probe.  A plain
    ``nn.Linear`` can be indexed directly.  During continuation training PEFT
    wraps that layer with LoRA; indexing only ``base_layer.weight`` would then
    silently discard the adapter.  PEFT 0.10's linear wrapper is deliberately
    recognized by attributes instead of importing PEFT, so inference keeps no
    optional dependency and the selected projection remains differentiable.
    """

    if type(module) is nn.Linear:
        return "plain"
    base = getattr(module, "base_layer", None)
    if type(base) is not nn.Linear:
        return None
    if not all(
        hasattr(module, name)
        for name in (
            "active_adapters",
            "lora_A",
            "lora_B",
            "lora_dropout",
            "scaling",
            "use_dora",
        )
    ):
        return None
    active = tuple(getattr(module, "active_adapters", ()))
    use_dora = getattr(module, "use_dora")
    if any(bool(use_dora.get(name, False)) for name in active):
        # DoRA's magnitude branch is not separable by a simple row/column
        # gather.  Fail closed rather than silently changing its computation.
        return None
    return "peft_lora"


def _selected_bias(
    bias: Optional[torch.Tensor], indices: torch.Tensor
) -> Optional[torch.Tensor]:
    return None if bias is None else bias.index_select(0, indices)


def _active_lora_adapters(module: nn.Module) -> tuple[str, ...]:
    if bool(getattr(module, "disable_adapters", False)) or bool(
        getattr(module, "merged", False)
    ):
        return ()
    lora_a = getattr(module, "lora_A")
    return tuple(
        name
        for name in getattr(module, "active_adapters", ())
        if name in lora_a
    )


def _linear_selected_outputs(
    module: nn.Module,
    inputs: torch.Tensor,
    output_indices: torch.Tensor,
) -> torch.Tensor:
    """Evaluate only selected output rows, including exact LoRA gradients."""

    kind = _structural_linear_kind(module)
    if kind == "plain":
        return F.linear(
            inputs,
            module.weight.index_select(0, output_indices),
            _selected_bias(module.bias, output_indices),
        )
    if kind != "peft_lora":
        raise TypeError("unsupported structural linear projection")
    base = module.base_layer
    result = F.linear(
        inputs,
        base.weight.index_select(0, output_indices),
        _selected_bias(base.bias, output_indices),
    )
    result_dtype = result.dtype
    for name in _active_lora_adapters(module):
        lora_a = module.lora_A[name]
        lora_b = module.lora_B[name]
        adapter_inputs = inputs.to(lora_a.weight.dtype)
        low_rank = lora_a(module.lora_dropout[name](adapter_inputs))
        delta = F.linear(
            low_rank,
            lora_b.weight.index_select(0, output_indices),
            _selected_bias(lora_b.bias, output_indices),
        )
        result = result + delta * module.scaling[name]
    return result.to(result_dtype)


def _linear_selected_inputs(
    module: nn.Module,
    inputs: torch.Tensor,
    input_indices: torch.Tensor,
) -> torch.Tensor:
    """Evaluate a linear layer whose omitted input columns are exactly zero."""

    kind = _structural_linear_kind(module)
    if kind == "plain":
        return F.linear(
            inputs,
            module.weight.index_select(1, input_indices),
            module.bias,
        )
    if kind != "peft_lora":
        raise TypeError("unsupported structural linear projection")
    base = module.base_layer
    result = F.linear(
        inputs,
        base.weight.index_select(1, input_indices),
        base.bias,
    )
    result_dtype = result.dtype
    for name in _active_lora_adapters(module):
        lora_a = module.lora_A[name]
        lora_b = module.lora_B[name]
        adapter_inputs = inputs.to(lora_a.weight.dtype)
        dropped = module.lora_dropout[name](adapter_inputs)
        low_rank = F.linear(
            dropped,
            lora_a.weight.index_select(1, input_indices),
            lora_a.bias,
        )
        result = result + lora_b(low_rank) * module.scaling[name]
    return result.to(result_dtype)


def probe_activated_structural_ffn(
    hidden_states: torch.Tensor,
    *,
    gate_proj: nn.Module,
    up_proj: nn.Module,
    down_proj: nn.Module,
    act_fn: Any,
    cfg: Optional[Mapping[str, Any]],
    runtime: Optional[Dict[str, Any]],
) -> Optional[torch.Tensor]:
    """Execute the paper's indexed visual FFN when plain weights are available.

    Returning ``None`` requests the released dense-mask fallback.  Plain linear
    and PEFT/LoRA-wrapped projections are both supported: the latter evaluates
    selected base rows/columns and the matching low-rank adapter terms, keeping
    the forward result and LoRA gradients equivalent to the dense masked form.
    """

    if cfg is None or str(cfg.get("operation_backend")) != "indexed_structural":
        return None
    if hidden_states.ndim != 3 or int(hidden_states.shape[0]) != 1:
        return None
    projection_kinds = tuple(
        _structural_linear_kind(module)
        for module in (gate_proj, up_proj, down_proj)
    )
    if any(kind is None for kind in projection_kinds):
        if isinstance(runtime, dict):
            runtime[
                "redundancy_lens_ffn_operation_backend_fallback"
            ] = "unsupported_projection_preserves_dense_semantics"
        return None
    visual = visual_token_mask(
        runtime,
        int(hidden_states.shape[1]),
        device=hidden_states.device,
    )
    if visual is None:
        return None

    visual_indices = visual.nonzero(as_tuple=True)[0]
    text_indices = (~visual).nonzero(as_tuple=True)[0]
    visual_inputs = hidden_states.index_select(1, visual_indices)[0]
    visual_count = int(visual_inputs.shape[0])
    channel_count = int(gate_proj.out_features)
    sample_count = max(
        1, int(math.floor(visual_count * float(cfg["probe_ratio"])))
    )
    sample_count = min(sample_count, visual_count)
    keep_count = max(
        1, int(math.floor(channel_count * float(cfg["ffn_keep_ratio"])))
    )
    keep_count = min(keep_count, channel_count)
    sampled_rows = torch.randperm(
        visual_count, device=hidden_states.device
    )[:sample_count]

    # The router is training-free: selection itself does not carry gradients.
    with torch.no_grad():
        sample_inputs = visual_inputs.index_select(0, sampled_rows)
        sample_middle = act_fn(gate_proj(sample_inputs)) * up_proj(sample_inputs)
        importance = sample_middle.abs().mean(dim=0)
        keep_channels = torch.topk(
            importance, k=keep_count, largest=True, sorted=False
        ).indices

    visual_middle = act_fn(
        _linear_selected_outputs(gate_proj, visual_inputs, keep_channels)
    ) * _linear_selected_outputs(up_proj, visual_inputs, keep_channels)
    visual_output = _linear_selected_inputs(
        down_proj,
        visual_middle,
        keep_channels,
    )

    output = hidden_states.new_zeros(
        hidden_states.shape[0], hidden_states.shape[1], down_proj.out_features
    )
    output[0].index_copy_(0, visual_indices, visual_output)
    if int(text_indices.numel()):
        text_inputs = hidden_states.index_select(1, text_indices)[0]
        text_output = down_proj(act_fn(gate_proj(text_inputs)) * up_proj(text_inputs))
        output[0].index_copy_(0, text_indices, text_output)

    if isinstance(runtime, dict):
        runtime["redundancy_lens_probe_visual_tokens"] = sample_count
        runtime["redundancy_lens_ffn_channels_before"] = channel_count
        runtime["redundancy_lens_ffn_channels_after"] = keep_count
        runtime["redundancy_lens_ffn_actual_keep_ratio"] = (
            keep_count / float(channel_count)
        )
        runtime["redundancy_lens_ffn_mask_calls"] = int(
            runtime.get("redundancy_lens_ffn_mask_calls", 0)
        ) + 1
        runtime[
            "redundancy_lens_ffn_operation_backend_executed"
        ] = "indexed_structural"
        runtime["redundancy_lens_ffn_projection_kinds"] = list(
            projection_kinds
        )
    return output


def qwen2_redundancy_lens_layer_flops(
    *,
    visual_tokens: int,
    nonvisual_tokens: int,
    hidden_size: int,
    intermediate_size: int,
    num_attention_heads: int,
    num_key_value_heads: int,
    hollow_attention: bool,
    dynamic_ffn: bool,
    attention_range: int = 256,
    probe_ratio: float = 0.1,
    ffn_keep_ratio: float = 0.2,
) -> float:
    """Count one theoretical sparse RedundancyLens Qwen2 prefill layer.

    This uses the same convention as the other OP ports: one multiply-add is
    two FLOPs and the dense attention baseline is a square ``N x N`` matrix.
    Projection cost is unchanged.  Hollow Attention replaces only the visual
    pair block by a preceding local range, while Dynamic FFN reduces only the
    visual rows of the SwiGLU intermediate projection.
    """

    v = int(visual_tokens)
    t = int(nonvisual_tokens)
    d = int(hidden_size)
    m = int(intermediate_size)
    qh = int(num_attention_heads)
    kvh = int(num_key_value_heads)
    radius = int(attention_range)
    probe = float(probe_ratio)
    keep = float(ffn_keep_ratio)
    if min(v, t) < 0 or min(d, m, qh, kvh, radius) < 1:
        raise ValueError("RedundancyLens FLOP dimensions are invalid")
    if d % qh or qh % kvh:
        raise ValueError("RedundancyLens requires valid Qwen2 GQA geometry")
    if not 0.0 < probe <= 1.0:
        raise ValueError("RedundancyLens probe_ratio must lie in (0, 1]")
    if not 0.0 < keep <= 1.0:
        raise ValueError("RedundancyLens ffn_keep_ratio must lie in (0, 1]")

    n = v + t
    kv_width = kvh * (d // qh)
    projections = 4.0 * n * d * d + 4.0 * n * d * kv_width
    attention_pairs = n * n
    if hollow_attention:
        attention_pairs = n * n - v * v + v * min(v, radius)
    attention = 4.0 * d * attention_pairs
    ffn_rows = t + v * (keep if dynamic_ffn else 1.0)
    ffn = 6.0 * d * m * ffn_rows
    if dynamic_ffn and v:
        # Probe-Activated Dynamic FFN first evaluates both full-width SwiGLU
        # input projections on floor(probe_ratio * V) sampled visual tokens,
        # then evaluates all visual rows with the selected intermediate
        # channels.  The released code uses V // 10 for the default 10%; our
        # max(1, floor(...)) mirrors the guarded OneVision runtime for tiny
        # synthetic visual spans.  Omitting this probe made the theoretical
        # retained-FLOP ratio about 2--3 percentage points too optimistic.
        probe_tokens = max(1, int(math.floor(v * probe)))
        ffn += 4.0 * d * m * probe_tokens
    return float(projections + attention + ffn)


def qwen2_redundancy_lens_decoder_flops(
    *,
    visual_tokens: int,
    nonvisual_tokens: int,
    layer_count: int,
    attention_reduction_layers: Sequence[int],
    ffn_reduction_layers: Sequence[int],
    hidden_size: int,
    intermediate_size: int,
    num_attention_heads: int,
    num_key_value_heads: int,
    attention_range: int = 256,
    probe_ratio: float = 0.1,
    ffn_keep_ratio: float = 0.2,
) -> float:
    """Return total decoder-prefill FLOPs for one sample shape."""

    layers = int(layer_count)
    if layers < 1:
        raise ValueError("RedundancyLens layer_count must be positive")
    attention_layers = set(
        _layer_list(attention_reduction_layers, "attention_reduction_layers")
    )
    ffn_layers = set(_layer_list(ffn_reduction_layers, "ffn_reduction_layers"))
    if any(layer >= layers for layer in attention_layers | ffn_layers):
        raise ValueError("RedundancyLens reduction layer is outside the decoder")
    common = dict(
        visual_tokens=int(visual_tokens),
        nonvisual_tokens=int(nonvisual_tokens),
        hidden_size=int(hidden_size),
        intermediate_size=int(intermediate_size),
        num_attention_heads=int(num_attention_heads),
        num_key_value_heads=int(num_key_value_heads),
        attention_range=int(attention_range),
        probe_ratio=float(probe_ratio),
        ffn_keep_ratio=float(ffn_keep_ratio),
    )
    return float(
        sum(
            qwen2_redundancy_lens_layer_flops(
                **common,
                hollow_attention=layer in attention_layers,
                dynamic_ffn=layer in ffn_layers,
            )
            for layer in range(layers)
        )
    )


def select_redundancy_lens_flops_counts(
    *,
    sample_shapes: Sequence[Mapping[str, Any]],
    target_flops_ratio: float,
    layer_count: int,
    hidden_size: int,
    intermediate_size: int,
    num_attention_heads: int,
    num_key_value_heads: int,
    attention_range: int = 256,
    probe_ratio: float = 0.1,
    ffn_keep_ratio: float = 0.2,
    max_attention_reduction_layers: Optional[int] = None,
    max_ffn_reduction_layers: Optional[int] = None,
) -> Dict[str, Any]:
    """Select reduction counts using summed method / summed baseline FLOPs.

    FLOPs depend on the number, but not the identity, of reduced layers.  The
    identity is supplied separately by the paper's layer-ranking strategy.
    Optional maxima keep the count search inside the prefix actually covered
    by a finite target-checkpoint ranking asset; missing layer identities must
    never be synthesized just to hit a FLOP target.
    Ties prefer balanced attention/FFN counts and then fewer total reductions.
    """

    count = int(layer_count)
    target = float(target_flops_ratio)
    if count < 1 or not 0.0 < target <= 1.0:
        raise ValueError("Invalid RedundancyLens FLOP search target")
    attention_limit = (
        count
        if max_attention_reduction_layers is None
        else int(max_attention_reduction_layers)
    )
    ffn_limit = (
        count
        if max_ffn_reduction_layers is None
        else int(max_ffn_reduction_layers)
    )
    if not 0 <= attention_limit <= count:
        raise ValueError("Invalid RedundancyLens attention search cap")
    if not 0 <= ffn_limit <= count:
        raise ValueError("Invalid RedundancyLens FFN search cap")
    shapes = [
        (int(item["visual_tokens"]), int(item["nonvisual_tokens"]))
        for item in sample_shapes
    ]
    if not shapes or any(v < 1 or t < 1 for v, t in shapes):
        raise ValueError("RedundancyLens needs positive calibration shapes")

    dense_layers: list[int] = []
    full_costs = [
        qwen2_redundancy_lens_decoder_flops(
            visual_tokens=v,
            nonvisual_tokens=t,
            layer_count=count,
            attention_reduction_layers=dense_layers,
            ffn_reduction_layers=dense_layers,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            num_attention_heads=num_attention_heads,
            num_key_value_heads=num_key_value_heads,
            attention_range=attention_range,
            probe_ratio=probe_ratio,
            ffn_keep_ratio=ffn_keep_ratio,
        )
        for v, t in shapes
    ]
    best_key = None
    best = None
    for attention_count in range(attention_limit + 1):
        attention_layers = list(range(count - attention_count, count))
        for ffn_count in range(ffn_limit + 1):
            ffn_layers = list(range(count - ffn_count, count))
            method_costs = []
            for (v, t), full in zip(shapes, full_costs):
                selected = qwen2_redundancy_lens_decoder_flops(
                    visual_tokens=v,
                    nonvisual_tokens=t,
                    layer_count=count,
                    attention_reduction_layers=attention_layers,
                    ffn_reduction_layers=ffn_layers,
                    hidden_size=hidden_size,
                    intermediate_size=intermediate_size,
                    num_attention_heads=num_attention_heads,
                    num_key_value_heads=num_key_value_heads,
                    attention_range=attention_range,
                    probe_ratio=probe_ratio,
                    ffn_keep_ratio=ffn_keep_ratio,
                )
                method_costs.append(float(selected))
            aggregate = aggregate_samplewise_flops(
                method_costs, full_costs
            )
            mean_ratio = float(aggregate["mean_flops_ratio"])
            key = (
                abs(mean_ratio - target),
                abs(attention_count - ffn_count),
                attention_count + ffn_count,
                attention_count,
            )
            if best_key is None or key < best_key:
                best_key = key
                best = {
                    "attention_reduction_layer_count": attention_count,
                    "ffn_reduction_layer_count": ffn_count,
                    "max_attention_reduction_layers": attention_limit,
                    "max_ffn_reduction_layers": ffn_limit,
                    **aggregate,
                    "absolute_target_error": abs(mean_ratio - target),
                }
    if best is None:
        raise AssertionError("RedundancyLens FLOP search produced no candidate")
    return best
