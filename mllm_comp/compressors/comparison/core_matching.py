"""Paper-guided CoreMatching for the OneVision Qwen2 decoder.

The public CoreMatching repository is tied to LLaVA-1.5: its active path
hard-codes the visual interval ``[5, 581)``, allocates tensors on ``cuda:0``
and zeros cached K/V rows.  This module instead implements Algorithm 1 from
the paper against the exact post-AnyRes visual spans recorded by OneVision.

CoreMatching has two coupled components:

* During prefill, each FFN records the sentence-wise core neurons obtained by
  counting every token's top positive activations.  At the configured pruning
  layer, visual tokens are ranked by the number of positive activations that
  intersect that layer's sentence-wise core neurons and the paper's maximum
  geometric-distance knee chooses the core-token set.
* During decode, each layer evaluates only the request-specific core FFN
  channels captured for that layer during prefill.

The paper-default configuration leaves the optional visual-token budget band
disabled (minimum zero, maximum one).  A non-default bound is an explicit
compute-budget adaptation: it still uses the paper's core-neuron scores and
maximum-distance knee, but restores or removes the lowest-ranked boundary
tokens when the dynamic knee misses a requested budget.  It never adds encoder
work, and the unbounded paper configuration remains available separately.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Mapping, Optional, Sequence

import torch
from torch import nn
from torch.nn import functional as F

from .qwen2_flops import aggregate_samplewise_flops


DEFAULT_CONFIG: Dict[str, Any] = {
    "method": "core_matching",
    "token_core_ratio": 0.2,
    "sentence_core_ratio": 0.4,
    # Paper l=2, expressed as the number of completed decoder blocks.
    "prune_after_layer": 2,
    "threshold_method": "maximum_geometric_distance",
    "threshold_tie_rule": "greater_equal",
    "minimum_visual_retain_ratio": 0.0,
    "maximum_visual_retain_ratio": 1.0,
    "operation_backend": "indexed_structural",
    "prefill_token_pruning": True,
    "decode_neuron_pruning": True,
    "paper_default": True,
    "adaptation": "paper_direct",
    "image_only": True,
    "log_first_n": 1,
}


def normalize_core_matching_config(
    raw: Optional[Mapping[str, Any]],
) -> Dict[str, Any]:
    cfg = dict(DEFAULT_CONFIG)
    if raw:
        cfg.update(dict(raw))
    if str(cfg.get("method", "")).lower() != "core_matching":
        raise ValueError("CoreMatching config must declare method=core_matching")
    cfg["method"] = "core_matching"
    for key in ("token_core_ratio", "sentence_core_ratio"):
        cfg[key] = float(cfg[key])
        if not 0.0 < cfg[key] <= 1.0:
            raise ValueError(f"{key} must satisfy 0 < ratio <= 1")
    cfg["prune_after_layer"] = int(cfg["prune_after_layer"])
    if cfg["prune_after_layer"] < 1:
        raise ValueError("CoreMatching prune_after_layer must be positive")
    if str(cfg.get("threshold_method")) != "maximum_geometric_distance":
        raise ValueError("CoreMatching requires maximum_geometric_distance")
    if str(cfg.get("threshold_tie_rule")) != "greater_equal":
        raise ValueError(
            "CoreMatching paper Algorithm 1 requires threshold_tie_rule="
            "greater_equal (score >= threshold)"
        )
    for key in (
        "minimum_visual_retain_ratio",
        "maximum_visual_retain_ratio",
    ):
        cfg[key] = float(cfg[key])
        if not 0.0 <= cfg[key] <= 1.0:
            raise ValueError(f"{key} must lie in [0, 1]")
    if cfg["maximum_visual_retain_ratio"] <= 0.0:
        raise ValueError("maximum_visual_retain_ratio must be positive")
    if (
        cfg["minimum_visual_retain_ratio"]
        > cfg["maximum_visual_retain_ratio"]
    ):
        raise ValueError(
            "minimum_visual_retain_ratio cannot exceed "
            "maximum_visual_retain_ratio"
        )
    if str(cfg.get("operation_backend")) != "indexed_structural":
        raise ValueError("CoreMatching requires indexed_structural operations")
    for key in (
        "prefill_token_pruning",
        "decode_neuron_pruning",
        "image_only",
    ):
        if not bool(cfg.get(key, False)):
            raise ValueError(f"CoreMatching requires {key}=true")
        cfg[key] = True
    has_floor = cfg["minimum_visual_retain_ratio"] > 0.0
    has_cap = cfg["maximum_visual_retain_ratio"] < 1.0
    cfg["paper_default"] = bool(cfg.get("paper_default", True))
    if has_floor or has_cap:
        cfg["paper_default"] = False
        if has_floor and has_cap:
            cfg["adaptation"] = "target_budget_visual_retain_band"
        elif has_floor:
            cfg["adaptation"] = "target_budget_minimum_visual_floor"
        else:
            cfg["adaptation"] = "target_budget_maximum_visual_cap"
    elif str(cfg.get("adaptation")) != "paper_direct":
        raise ValueError(
            "CoreMatching without a visual budget bound must declare "
            "adaptation=paper_direct"
        )
    cfg["log_first_n"] = int(cfg.get("log_first_n", 1))
    if cfg["log_first_n"] < 0:
        raise ValueError("log_first_n must be non-negative")
    return cfg


def visual_indices_from_runtime(
    runtime: Optional[Mapping[str, Any]],
    sequence_length: int,
    *,
    device: Optional[torch.device] = None,
) -> Optional[torch.LongTensor]:
    """Resolve every expanded OneVision visual position without fixed offsets."""

    if not runtime or not runtime.get("has_image") or int(sequence_length) <= 1:
        return None
    spans = runtime.get("expanded_visual_spans")
    if not isinstance(spans, list) or not spans:
        return None
    parts = []
    previous_end = 0
    for raw in spans:
        if not isinstance(raw, (list, tuple)) or len(raw) != 2:
            raise ValueError("CoreMatching visual span must be a start/end pair")
        start, end = int(raw[0]), int(raw[1])
        if not 0 <= start < end <= int(sequence_length) or start < previous_end:
            raise ValueError(
                f"Invalid CoreMatching visual span {(start, end)} for length "
                f"{sequence_length}"
            )
        parts.append(torch.arange(start, end, device=device, dtype=torch.long))
        previous_end = end
    return torch.cat(parts) if parts else None


def sentence_core_neurons(
    intermediate_states: torch.Tensor,
    *,
    token_core_ratio: float,
    sentence_core_ratio: float,
) -> tuple[torch.LongTensor, torch.Tensor]:
    """Return Algorithm-1 sentence core neurons and their token frequencies.

    The token-wise set contains the largest ``rho`` fraction of *positive*
    activations for that token, matching equations (2)--(4).  A single batched
    ``topk`` is used with a per-row validity mask, avoiding one GPU launch per
    token while preserving the positive-count-dependent set size.
    """

    if intermediate_states.ndim != 2 or int(intermediate_states.shape[0]) < 1:
        raise ValueError("CoreMatching expects [tokens, intermediate] states")
    token_count, channel_count = map(int, intermediate_states.shape)
    if channel_count < 1:
        raise ValueError("CoreMatching FFN has no intermediate channels")
    rho = float(token_core_ratio)
    beta = float(sentence_core_ratio)
    if not 0.0 < rho <= 1.0 or not 0.0 < beta <= 1.0:
        raise ValueError("CoreMatching sparsity ratios must lie in (0, 1]")

    detached = intermediate_states.detach()
    positive_counts = (detached > 0).sum(dim=-1, dtype=torch.long)
    per_token_k = torch.ceil(positive_counts.to(torch.float32) * rho).long()
    per_token_k = torch.minimum(per_token_k, positive_counts)
    maximum_k = int(per_token_k.max().item())
    frequency = torch.zeros(
        channel_count, dtype=torch.int32, device=detached.device
    )
    if maximum_k > 0:
        top_values, top_indices = torch.topk(
            detached, k=maximum_k, dim=-1, largest=True, sorted=True
        )
        ranks = torch.arange(maximum_k, device=detached.device).view(1, -1)
        valid = (ranks < per_token_k.view(-1, 1)) & (top_values > 0)
        frequency.scatter_add_(
            0,
            top_indices.reshape(-1),
            valid.reshape(-1).to(dtype=frequency.dtype),
        )

    sentence_k = max(1, int(math.ceil(channel_count * beta)))
    sentence_k = min(sentence_k, channel_count)
    core_indices = torch.topk(
        frequency, k=sentence_k, largest=True, sorted=False
    ).indices.long()
    return core_indices, frequency


def capture_core_matching_prefill(
    intermediate_states: torch.Tensor,
    *,
    cfg: Optional[Mapping[str, Any]],
    runtime: Optional[Dict[str, Any]],
    layer_idx: int,
) -> None:
    """Capture one layer's request-specific core neurons during prefill."""

    if cfg is None or runtime is None or not runtime.get("has_image"):
        return
    if intermediate_states.ndim != 3 or int(intermediate_states.shape[0]) != 1:
        return
    if int(intermediate_states.shape[1]) <= 1:
        return
    with torch.no_grad():
        states = intermediate_states[0]
        core, frequency = sentence_core_neurons(
            states,
            token_core_ratio=float(cfg["token_core_ratio"]),
            sentence_core_ratio=float(cfg["sentence_core_ratio"]),
        )
        private = runtime.setdefault("_core_matching_core_neuron_tensors", {})
        private[int(layer_idx)] = core.detach()
        counts = runtime.setdefault("core_matching_core_neuron_counts", {})
        counts[str(int(layer_idx))] = int(core.numel())
        runtime["core_matching_core_neuron_layers_captured"] = len(private)
        runtime["core_matching_frequency_max"] = int(frequency.max().item())

        if int(layer_idx) + 1 == int(cfg["prune_after_layer"]):
            visual_indices = visual_indices_from_runtime(
                runtime,
                int(states.shape[0]),
                device=states.device,
            )
            if visual_indices is None:
                raise RuntimeError(
                    "CoreMatching cannot score tokens without expanded visual spans"
                )
            visual_states = states.index_select(0, visual_indices)
            token_scores = (visual_states.index_select(1, core) > 0).sum(
                dim=-1, dtype=torch.long
            )
            runtime["_core_matching_visual_indices"] = visual_indices.detach()
            runtime["_core_matching_visual_scores"] = token_scores.detach()


def maximum_geometric_distance_index(sorted_scores: torch.Tensor) -> int:
    """Paper knee: point with maximum distance from endpoint chord."""

    if sorted_scores.ndim != 1 or int(sorted_scores.numel()) < 1:
        raise ValueError("CoreMatching knee requires a non-empty score vector")
    count = int(sorted_scores.numel())
    if count == 1:
        return 0
    scores = sorted_scores.to(dtype=torch.float64)
    x = torch.arange(count, dtype=torch.float64, device=scores.device)
    start = torch.stack((x[0], scores[0]))
    end = torch.stack((x[-1], scores[-1]))
    line = end - start
    norm = torch.linalg.vector_norm(line)
    if float(norm.item()) == 0.0:
        return count - 1
    unit = line / norm
    points = torch.stack((x, scores), dim=1)
    offsets = points - start
    projections = (offsets @ unit).unsqueeze(1) * unit.unsqueeze(0)
    distances = torch.linalg.vector_norm(offsets - projections, dim=1)
    return int(torch.argmax(distances).item())


def build_core_token_plan(
    *,
    sequence_length: int,
    visual_indices: torch.LongTensor,
    visual_scores: torch.Tensor,
    minimum_visual_retain_ratio: float = 0.0,
    maximum_visual_retain_ratio: float = 1.0,
) -> Dict[str, Any]:
    """Select paper core tokens, optionally constrained by a budget band."""

    length = int(sequence_length)
    if visual_indices.ndim != 1 or visual_scores.ndim != 1:
        raise ValueError("CoreMatching visual indices/scores must be vectors")
    if int(visual_indices.numel()) != int(visual_scores.numel()):
        raise ValueError("CoreMatching visual index/score lengths differ")
    visual_count = int(visual_indices.numel())
    if visual_count < 1:
        raise ValueError("CoreMatching has no visual tokens to select")
    if int(visual_indices.min().item()) < 0 or int(visual_indices.max().item()) >= length:
        raise ValueError("CoreMatching visual index is outside the sequence")

    sorted_scores, order = torch.sort(
        visual_scores, descending=True, stable=True
    )
    knee_index = maximum_geometric_distance_index(sorted_scores)
    threshold = sorted_scores[knee_index]
    dynamic_mask = visual_scores >= threshold
    dynamic_count = int(dynamic_mask.sum().item())

    floor_ratio = float(minimum_visual_retain_ratio)
    if not 0.0 <= floor_ratio <= 1.0:
        raise ValueError("CoreMatching visual floor must lie in [0, 1]")
    cap_ratio = float(maximum_visual_retain_ratio)
    if not 0.0 < cap_ratio <= 1.0:
        raise ValueError("CoreMatching visual cap must lie in (0, 1]")
    if floor_ratio > cap_ratio:
        raise ValueError("CoreMatching visual floor cannot exceed its cap")
    floor_count = (
        max(1, int(math.ceil(visual_count * floor_ratio)))
        if floor_ratio > 0.0
        else 0
    )
    cap_count = max(1, int(math.ceil(visual_count * cap_ratio)))
    selected_count = min(max(dynamic_count, floor_count), cap_count)
    floor_applied = selected_count > dynamic_count
    cap_applied = selected_count < dynamic_count
    if floor_applied or cap_applied:
        selected_local = order[:selected_count]
    else:
        selected_local = dynamic_mask.nonzero(as_tuple=True)[0]
    selected_visual = torch.sort(
        visual_indices.index_select(0, selected_local)
    ).values

    keep_mask = torch.ones(length, dtype=torch.bool, device=visual_indices.device)
    keep_mask[visual_indices] = False
    keep_mask[selected_visual] = True
    keep_indices = keep_mask.nonzero(as_tuple=True)[0].long()
    return {
        "keep_indices": keep_indices,
        "selected_visual_indices": selected_visual,
        "visual_tokens_before": visual_count,
        "visual_tokens_after": int(selected_visual.numel()),
        "dynamic_visual_tokens": dynamic_count,
        "minimum_visual_tokens": floor_count,
        "maximum_visual_tokens": cap_count,
        "minimum_floor_applied": bool(floor_applied),
        "maximum_cap_applied": bool(cap_applied),
        "knee_index": int(knee_index),
        "threshold": int(threshold.item()),
    }


def qwen2_prefill_layer_flops(
    *,
    sequence_tokens: int,
    hidden_size: int,
    intermediate_size: int,
    num_attention_heads: int,
    num_key_value_heads: int,
) -> int:
    """Count one dense Qwen2 prefill layer (multiply-add = two FLOPs)."""

    tokens = int(sequence_tokens)
    hidden = int(hidden_size)
    intermediate = int(intermediate_size)
    heads = int(num_attention_heads)
    kv_heads = int(num_key_value_heads)
    if min(tokens, hidden, intermediate, heads, kv_heads) < 1:
        raise ValueError("CoreMatching FLOP dimensions must be positive")
    if hidden % heads:
        raise ValueError("hidden_size must be divisible by num_attention_heads")
    head_dim = hidden // heads
    kv_width = kv_heads * head_dim
    flops = 4 * tokens * hidden * hidden  # Q + O projections.
    flops += 4 * tokens * hidden * kv_width  # K + V projections.
    flops += 4 * tokens * tokens * hidden  # QK + AV attention.
    flops += 6 * tokens * hidden * intermediate  # SwiGLU FFN.
    return int(flops)


def qwen2_core_matching_prefill_flops(
    *,
    num_layers: int,
    prune_after_layer: int,
    nonvisual_tokens: int,
    visual_tokens: int,
    kept_visual_tokens: int,
    hidden_size: int,
    intermediate_size: int,
    num_attention_heads: int,
    num_key_value_heads: int,
) -> Dict[str, Any]:
    """Return dense/CoreMatching prefill FLOPs for one OneVision sample.

    CoreMatching computes ``prune_after_layer`` complete dense blocks, then
    physically removes non-core visual tokens for every remaining block.  Its
    sentence-core FFN width reduction is decode-only and is intentionally not
    charged to this prefill target.
    """

    layers = int(num_layers)
    dense_layers = int(prune_after_layer)
    text = int(nonvisual_tokens)
    visual = int(visual_tokens)
    kept = int(kept_visual_tokens)
    if layers < 1 or not 1 <= dense_layers <= layers:
        raise ValueError("CoreMatching prune_after_layer must lie in [1, layers]")
    if text < 1 or visual < 1 or not 1 <= kept <= visual:
        raise ValueError("Invalid CoreMatching token counts")
    dimensions = dict(
        hidden_size=int(hidden_size),
        intermediate_size=int(intermediate_size),
        num_attention_heads=int(num_attention_heads),
        num_key_value_heads=int(num_key_value_heads),
    )
    dense_layer = qwen2_prefill_layer_flops(
        sequence_tokens=text + visual, **dimensions
    )
    compact_layer = qwen2_prefill_layer_flops(
        sequence_tokens=text + kept, **dimensions
    )
    dense_total = layers * dense_layer
    selected_total = (
        dense_layers * dense_layer
        + (layers - dense_layers) * compact_layer
    )
    return {
        "dense_flops": int(dense_total),
        "core_matching_flops": int(selected_total),
        "flops_ratio": float(selected_total / dense_total),
        "dense_layer_flops": int(dense_layer),
        "compact_layer_flops": int(compact_layer),
    }


def select_core_matching_visual_budget(
    *,
    samples: Sequence[Mapping[str, Any]],
    num_layers: int,
    prune_after_layer: int,
    hidden_size: int,
    intermediate_size: int,
    num_attention_heads: int,
    num_key_value_heads: int,
    target_flops_ratio: float,
    budget_grid_step: float = 0.0001,
) -> Dict[str, Any]:
    """Fit a visual-token cap/floor to a dataset prefill-FLOPs target.

    The paper's maximum-geometric-distance selection remains authoritative.
    A floor restores top-ranked tokens when its knee is too aggressive; a cap
    removes the lowest-ranked selected tokens when the knee is not aggressive
    enough.  Each sample uses its real OneVision lengths, then method and
    baseline FLOPs are separately summed before division.  The paper-default
    knee is always reported separately.
    """

    if not samples:
        raise ValueError("CoreMatching floor calibration requires samples")
    target = float(target_flops_ratio)
    step = float(budget_grid_step)
    if not 0.0 < target <= 1.0:
        raise ValueError("target_flops_ratio must lie in (0, 1]")
    if not 0.0 < step <= 1.0:
        raise ValueError("budget_grid_step must lie in (0, 1]")

    candidate_count = int(round(1.0 / step))
    if not math.isclose(candidate_count * step, 1.0, abs_tol=1e-9):
        raise ValueError("budget_grid_step must evenly divide one")

    dimensions = dict(
        num_layers=int(num_layers),
        prune_after_layer=int(prune_after_layer),
        hidden_size=int(hidden_size),
        intermediate_size=int(intermediate_size),
        num_attention_heads=int(num_attention_heads),
        num_key_value_heads=int(num_key_value_heads),
    )
    best = None
    paper_default = None
    for candidate_index in range(candidate_count + 1):
        budget_ratio = float(round(candidate_index * step, 10))
        # Candidate zero is the unbounded paper knee.  Positive candidates are
        # interpreted as a cap when the paper mean is above target, otherwise
        # as a floor.  We determine the direction after measuring candidate 0.
        floor_ratio = 0.0
        cap_ratio = 1.0
        if candidate_index > 0 and paper_default is not None:
            if float(paper_default["mean_flops_ratio"]) > target:
                cap_ratio = budget_ratio
            else:
                floor_ratio = budget_ratio
        method_costs = []
        baseline_costs = []
        kept_counts = []
        for sample in samples:
            visual = int(sample["visual_tokens"])
            dynamic = int(sample["dynamic_visual_tokens"])
            if visual < 1 or not 1 <= dynamic <= visual:
                raise ValueError("Invalid CoreMatching calibration sample")
            floor_count = (
                max(1, int(math.ceil(visual * floor_ratio)))
                if floor_ratio > 0.0
                else 0
            )
            cap_count = max(1, int(math.ceil(visual * cap_ratio)))
            kept = min(max(dynamic, floor_count), cap_count)
            result = qwen2_core_matching_prefill_flops(
                nonvisual_tokens=int(sample["nonvisual_tokens"]),
                visual_tokens=visual,
                kept_visual_tokens=kept,
                **dimensions,
            )
            method_costs.append(float(result["core_matching_flops"]))
            baseline_costs.append(float(result["dense_flops"]))
            kept_counts.append(kept)
        aggregate = aggregate_samplewise_flops(
            method_costs, baseline_costs
        )
        row = {
            "minimum_visual_retain_ratio": floor_ratio,
            "maximum_visual_retain_ratio": cap_ratio,
            **aggregate,
            "mean_kept_visual_tokens": float(
                sum(kept_counts) / len(kept_counts)
            ),
            "minimum_kept_visual_tokens": int(min(kept_counts)),
            "maximum_kept_visual_tokens": int(max(kept_counts)),
        }
        if candidate_index == 0:
            paper_default = dict(row)
        key = (
            abs(float(row["mean_flops_ratio"]) - target),
            float(row["mean_flops_ratio"]) > target,
            floor_ratio,
            -cap_ratio,
        )
        if best is None or key < best[0]:
            best = (key, row)

    assert best is not None and paper_default is not None
    selected = dict(best[1])
    selected["target_flops_ratio"] = target
    selected["absolute_target_error"] = abs(
        float(selected["mean_flops_ratio"]) - target
    )
    selected["budget_grid_step"] = step
    selected["paper_default"] = paper_default
    return selected


def select_core_matching_visual_floor(**kwargs: Any) -> Dict[str, Any]:
    """Backward-compatible wrapper for the earlier calibration API."""

    if "floor_grid_step" in kwargs and "budget_grid_step" not in kwargs:
        kwargs["budget_grid_step"] = kwargs.pop("floor_grid_step")
    return select_core_matching_visual_budget(**kwargs)


def _is_plain_linear(module: nn.Module) -> bool:
    return type(module) is nn.Linear


def core_matching_decode_ffn(
    hidden_states: torch.Tensor,
    *,
    gate_proj: nn.Module,
    up_proj: nn.Module,
    down_proj: nn.Module,
    act_fn: Any,
    cfg: Optional[Mapping[str, Any]],
    runtime: Optional[Dict[str, Any]],
    layer_idx: int,
) -> Optional[torch.Tensor]:
    """Execute only the prefill-captured sentence core channels on decode."""

    if cfg is None or runtime is None or not runtime.get("has_image"):
        return None
    if hidden_states.ndim != 3 or tuple(hidden_states.shape[:2]) != (1, 1):
        return None
    private = runtime.get("_core_matching_core_neuron_tensors")
    if not isinstance(private, dict) or int(layer_idx) not in private:
        raise RuntimeError(
            f"CoreMatching decode has no prefill core neurons for layer {layer_idx}"
        )
    if not all(_is_plain_linear(module) for module in (gate_proj, up_proj, down_proj)):
        runtime[
            "core_matching_decode_backend_fallback"
        ] = "adapter_wrapped_projection_dense_decode"
        return None
    core = private[int(layer_idx)].to(device=hidden_states.device, dtype=torch.long)
    gate = F.linear(
        hidden_states,
        gate_proj.weight.index_select(0, core),
        gate_proj.bias.index_select(0, core) if gate_proj.bias is not None else None,
    )
    up = F.linear(
        hidden_states,
        up_proj.weight.index_select(0, core),
        up_proj.bias.index_select(0, core) if up_proj.bias is not None else None,
    )
    output = F.linear(
        act_fn(gate) * up,
        down_proj.weight.index_select(1, core),
        down_proj.bias,
    )
    runtime["core_matching_decode_ffn_calls"] = int(
        runtime.get("core_matching_decode_ffn_calls", 0)
    ) + 1
    runtime["core_matching_decode_channels_before"] = int(gate_proj.out_features)
    runtime["core_matching_decode_channels_after"] = int(core.numel())
    runtime["core_matching_decode_actual_keep_ratio"] = int(core.numel()) / float(
        gate_proj.out_features
    )
    runtime["core_matching_decode_operation_backend"] = "indexed_structural"
    return output
