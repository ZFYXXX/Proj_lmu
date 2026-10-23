"""Depth-wise Operation Pruning (DOP) for LLaVA-OneVision.

The final DOP paper parameterizes an image-token policy by ``(d_A, d_P,
n_v)``: image tokens participate in MHA through depth ``d_A`` and in the MLP
through depth ``d_P``.  Deeper operations bypass the image group while the
residual stream keeps the corresponding token slots.  ``n_v`` is produced by
an existing token reducer; token_reducer=vispruner selects DOP-V; token_reducer=none keeps all visual
slots and only prunes operation depth (the standalone depth-only experiment).

The helpers in this module are deliberately model-agnostic.  Indexed Q/K/V and
residual execution lives in the local Qwen2 fork, where it can also maintain a
shorter per-layer KV cache during autoregressive decoding.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple

import torch


DOP_DEFAULTS: Dict[str, Any] = {
    "method": "dop",
    # This is only a runnable smoke profile.  It must not be used as the
    # project's formal 0.67-FLOP policy: Table 16 is a Qwen2.5-VL *resizing*
    # result, whereas our model is LLaVA-OneVision/Next with AnyRes SigLIP and
    # a Qwen2 decoder.  Formal jobs load a target-checkpoint-specific policy
    # produced from the paper's TextVQA-100 additive-divergence calibration.
    "profile": "smoke_qwen2_5_vl_table16_depth_with_ov_vispruner",
    "mha_depth": 28,
    "mlp_depth": 20,
    "source_visual_tokens": 1296,
    "source_policy_visual_tokens": 640,
    "target_tflops_ratio": 0.43,
    "policy_source": (
        "smoke_only_table16_qwen2_5_vl_depth_not_a_formal_ov_policy_"
        "textvqa100_calibration_required"
    ),
    # Qwen2.5-VL's paper point uses input resizing. OneVision instead needs a
    # spatially valid token reducer, so retain the implemented DOP-V/VisPruner
    # path and map 640/1296 onto each 729-patch SigLIP crop: 360 tokens/crop.
    "token_reducer": "vispruner",
    "tokens_per_crop": 360,
    "important_ratio": 0.5,
    "vision_layer": -1,
    "attention_source": "siglip_mean_received",
    "similarity": "cosine",
    "removal_step": 8,
    "spatial_restore": True,
    "operation_backend": "indexed_exact",
    "single_image_only": True,
    "image_only": True,
    "log_first_n": 1,
}


def normalize_dop_config(raw: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Validate a fixed DOP policy and its internal DOP-V selector settings."""

    cfg = dict(DOP_DEFAULTS)
    if raw:
        unknown = sorted(set(raw) - set(cfg))
        if unknown:
            raise ValueError(f"Unknown DOP config keys: {', '.join(unknown)}")
        cfg.update(raw)

    cfg["method"] = "dop"
    cfg["profile"] = str(cfg["profile"])
    cfg["mha_depth"] = int(cfg["mha_depth"])
    cfg["mlp_depth"] = int(cfg["mlp_depth"])
    cfg["source_visual_tokens"] = int(cfg["source_visual_tokens"])
    cfg["source_policy_visual_tokens"] = int(
        cfg["source_policy_visual_tokens"]
    )
    cfg["target_tflops_ratio"] = float(cfg["target_tflops_ratio"])
    cfg["policy_source"] = str(cfg["policy_source"])
    cfg["token_reducer"] = str(cfg["token_reducer"]).strip().lower()
    cfg["tokens_per_crop"] = int(cfg["tokens_per_crop"])
    cfg["important_ratio"] = float(cfg["important_ratio"])
    cfg["vision_layer"] = int(cfg["vision_layer"])
    cfg["attention_source"] = str(cfg["attention_source"]).strip().lower()
    cfg["similarity"] = str(cfg["similarity"]).strip().lower()
    cfg["removal_step"] = int(cfg["removal_step"])
    cfg["spatial_restore"] = bool(cfg["spatial_restore"])
    cfg["operation_backend"] = str(cfg["operation_backend"]).strip().lower()
    cfg["single_image_only"] = bool(cfg["single_image_only"])
    cfg["image_only"] = bool(cfg["image_only"])
    cfg["log_first_n"] = int(cfg["log_first_n"])

    if cfg["mha_depth"] < 0 or cfg["mlp_depth"] < 0:
        raise ValueError("DOP mha_depth and mlp_depth must be >= 0")
    if cfg["source_visual_tokens"] < 1:
        raise ValueError("DOP source_visual_tokens must be >= 1")
    if not 0 <= cfg["source_policy_visual_tokens"] <= cfg["source_visual_tokens"]:
        raise ValueError(
            "DOP source_policy_visual_tokens must lie in [0, source_visual_tokens]"
        )
    if not 0.0 < cfg["target_tflops_ratio"] <= 1.0:
        raise ValueError("DOP target_tflops_ratio must satisfy 0 < ratio <= 1")
    if cfg["token_reducer"] not in {"vispruner", "none"}:
        raise ValueError("DOP token_reducer must be vispruner (DOP_V) or none (depth-only DOP)")
    if cfg["tokens_per_crop"] < 1:
        raise ValueError("DOP tokens_per_crop must be >= 1")
    if not 0.0 <= cfg["important_ratio"] <= 1.0:
        raise ValueError("DOP important_ratio must satisfy 0 <= ratio <= 1")
    if cfg["attention_source"] != "siglip_mean_received":
        raise ValueError(
            "OneVision DOP-V requires attention_source='siglip_mean_received'"
        )
    if cfg["similarity"] != "cosine":
        raise ValueError("OneVision DOP-V requires cosine similarity")
    if cfg["removal_step"] < 1:
        raise ValueError("DOP removal_step must be >= 1")
    if not cfg["spatial_restore"]:
        raise ValueError("OneVision AnyRes DOP-V requires spatial_restore=true")
    if cfg["operation_backend"] != "indexed_exact":
        raise ValueError("DOP currently supports operation_backend='indexed_exact'")
    if not cfg["single_image_only"]:
        raise ValueError("DOP's indexed OneVision path currently requires single_image_only=true")
    if cfg["log_first_n"] < 0:
        raise ValueError("DOP log_first_n must be >= 0")
    return cfg


def resolve_dop_visual_span(
    runtime: Optional[dict], sequence_length: int
) -> Optional[Tuple[int, int]]:
    """Resolve the exact expanded image span for a single OneVision image."""

    if not runtime or not runtime.get("has_image"):
        return None
    if int(runtime.get("batch_size", 0)) != 1:
        runtime["skip_reason"] = "DOP currently requires batch_size=1"
        return None
    placeholder_positions = runtime.get("placeholder_positions") or []
    if len(placeholder_positions) != 1:
        runtime["skip_reason"] = "DOP requires exactly one image placeholder"
        return None

    raw_length = int(runtime["orig_seq_len"])
    visual_length = int(sequence_length) - raw_length + 1
    visual_start = int(placeholder_positions[0])
    visual_end = visual_start + visual_length
    if visual_length <= 0 or visual_start < 0 or visual_end > int(sequence_length):
        runtime["skip_reason"] = (
            f"Invalid DOP visual span [{visual_start}, {visual_end}) for "
            f"sequence length {sequence_length}"
        )
        return None

    runtime.pop("skip_reason", None)
    if runtime.get("visual_tokens") is not None:
        runtime.setdefault(
            "uncompressed_visual_tokens", int(runtime["visual_tokens"])
        )
    runtime["visual_start"] = visual_start
    runtime["visual_end"] = visual_end
    runtime["visual_tokens"] = visual_length
    return visual_start, visual_end


def nonvisual_indices(
    sequence_length: int,
    visual_span: Tuple[int, int],
    *,
    device: Optional[torch.device] = None,
) -> torch.LongTensor:
    """Indices retained by an operation after the visual group is bypassed."""

    start, end = (int(visual_span[0]), int(visual_span[1]))
    if start < 0 or end < start or end > int(sequence_length):
        raise ValueError(
            f"Invalid visual span [{start}, {end}) for sequence length {sequence_length}"
        )
    return torch.cat(
        (
            torch.arange(start, device=device, dtype=torch.long),
            torch.arange(end, int(sequence_length), device=device, dtype=torch.long),
        )
    )


def topk_output_kl(
    original_logits: torch.Tensor,
    pruned_logits: torch.Tensor,
    *,
    top_k: int = 10,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Equation (10): KL on the original distribution's top-k vocabulary."""

    if original_logits.shape != pruned_logits.shape:
        raise ValueError("DOP KL logits must have identical shapes")
    if original_logits.ndim < 1 or original_logits.shape[-1] < 1:
        raise ValueError("DOP KL logits require a non-empty vocabulary axis")
    k = min(int(top_k), int(original_logits.shape[-1]))
    if k < 1 or eps <= 0:
        raise ValueError("DOP KL requires top_k >= 1 and eps > 0")

    original_prob = original_logits.float().softmax(dim=-1)
    pruned_prob = pruned_logits.float().softmax(dim=-1)
    indices = original_prob.topk(k, dim=-1).indices
    p = original_prob.gather(-1, indices) + float(eps)
    q = pruned_prob.gather(-1, indices) + float(eps)
    p = p / p.sum(dim=-1, keepdim=True)
    q = q / q.sum(dim=-1, keepdim=True)
    return (p * (p.log() - q.log())).sum(dim=-1)


def qwen2_decoder_flops(
    *,
    num_layers: int,
    nonvisual_tokens: int,
    visual_tokens: int,
    mha_depth: int,
    mlp_depth: int,
    hidden_size: int,
    intermediate_size: int,
    num_attention_heads: int,
    num_key_value_heads: int,
) -> int:
    """Decoder FLOPs adjusted for Qwen2 grouped-query attention.

    For each MHA layer, Q and O cost ``4*n*h^2``, K and V cost
    ``4*n*h*kv_dim``, and QK/AV cost ``4*n^2*h``.  Qwen2's SwiGLU MLP costs
    ``6*n*h*m``.  This is the Qwen-specific counterpart of paper Eq. (11).
    """

    values = (
        num_layers,
        nonvisual_tokens,
        visual_tokens,
        hidden_size,
        intermediate_size,
        num_attention_heads,
        num_key_value_heads,
    )
    if any(int(value) < 0 for value in values):
        raise ValueError("DOP FLOP dimensions must be non-negative")
    if num_attention_heads < 1 or num_key_value_heads < 1:
        raise ValueError("DOP FLOPs require positive attention-head counts")
    if hidden_size % num_attention_heads:
        raise ValueError("hidden_size must be divisible by num_attention_heads")
    mha_depth = min(max(int(mha_depth), 0), int(num_layers))
    mlp_depth = min(max(int(mlp_depth), 0), int(num_layers))
    head_dim = int(hidden_size) // int(num_attention_heads)
    kv_dim = int(num_key_value_heads) * head_dim
    total = 0
    for layer in range(int(num_layers)):
        n_a = int(nonvisual_tokens) + (int(visual_tokens) if layer < mha_depth else 0)
        n_p = int(nonvisual_tokens) + (int(visual_tokens) if layer < mlp_depth else 0)
        total += 4 * n_a * int(hidden_size) * int(hidden_size)
        total += 4 * n_a * int(hidden_size) * kv_dim
        total += 4 * n_a * n_a * int(hidden_size)
        total += 6 * n_p * int(hidden_size) * int(intermediate_size)
    return int(total)


def interpolate_divergence_curve(
    sampled: Mapping[int, float], maximum: int
) -> Dict[int, float]:
    """Linearly interpolate sparse depth/token samples as in paper Eq. (8)."""

    if maximum < 0 or not sampled:
        raise ValueError("DOP interpolation requires samples and maximum >= 0")
    points = sorted((int(x), float(y)) for x, y in sampled.items())
    if points[0][0] < 0 or points[-1][0] > int(maximum):
        raise ValueError("DOP interpolation samples fall outside [0, maximum]")
    # Endpoint extension makes custom sparse files robust.  The paper's own
    # six-point sets already include the low endpoint and the unpruned anchor.
    if points[0][0] != 0:
        points.insert(0, (0, points[0][1]))
    if points[-1][0] != int(maximum):
        points.append((int(maximum), points[-1][1]))

    result: Dict[int, float] = {}
    segment = 0
    for value in range(int(maximum) + 1):
        while segment + 1 < len(points) - 1 and value > points[segment + 1][0]:
            segment += 1
        x0, y0 = points[segment]
        x1, y1 = points[segment + 1]
        ratio = 0.0 if x1 == x0 else (value - x0) / (x1 - x0)
        result[value] = float(y0 + ratio * (y1 - y0))
    return result


def optimize_dop_policy(
    *,
    mha_divergence: Mapping[int, float],
    mlp_divergence: Mapping[int, float],
    token_divergence: Mapping[int, float],
    num_layers: int,
    max_visual_tokens: int,
    nonvisual_tokens: int,
    hidden_size: int,
    intermediate_size: int,
    num_attention_heads: int,
    num_key_value_heads: int,
    target_flops_ratio: float,
    min_visual_alpha: float = 1.5,
) -> Dict[str, Any]:
    """Enumerate Eq. (7) after interpolating the three additive curves."""

    if not 0 < float(target_flops_ratio) <= 1:
        raise ValueError("target_flops_ratio must satisfy 0 < ratio <= 1")
    if min_visual_alpha < 0:
        raise ValueError("min_visual_alpha must be >= 0")
    da = interpolate_divergence_curve(mha_divergence, int(num_layers))
    dp = interpolate_divergence_curve(mlp_divergence, int(num_layers))
    dv = interpolate_divergence_curve(token_divergence, int(max_visual_tokens))

    flop_args = dict(
        num_layers=int(num_layers),
        nonvisual_tokens=int(nonvisual_tokens),
        hidden_size=int(hidden_size),
        intermediate_size=int(intermediate_size),
        num_attention_heads=int(num_attention_heads),
        num_key_value_heads=int(num_key_value_heads),
    )
    full_flops = qwen2_decoder_flops(
        visual_tokens=int(max_visual_tokens),
        mha_depth=int(num_layers),
        mlp_depth=int(num_layers),
        **flop_args,
    )
    budget = float(target_flops_ratio) * full_flops
    token_only_costs = [
        qwen2_decoder_flops(
            visual_tokens=n,
            mha_depth=int(num_layers),
            mlp_depth=int(num_layers),
            **flop_args,
        )
        for n in range(int(max_visual_tokens) + 1)
    ]
    n_equiv = min(
        range(int(max_visual_tokens) + 1),
        key=lambda n: abs(token_only_costs[n] - budget),
    )
    n_min = min(
        int(max_visual_tokens), int(math.ceil(float(min_visual_alpha) * n_equiv))
    )

    best: Optional[Tuple[float, int, int, int, int]] = None
    for depth_a in range(int(num_layers) + 1):
        for depth_p in range(int(num_layers) + 1):
            for n_visual in range(n_min, int(max_visual_tokens) + 1):
                flops = qwen2_decoder_flops(
                    visual_tokens=n_visual,
                    mha_depth=depth_a,
                    mlp_depth=depth_p,
                    **flop_args,
                )
                if flops > budget:
                    continue
                divergence = da[depth_a] + dp[depth_p] + dv[n_visual]
                candidate = (float(divergence), -flops, depth_a, depth_p, n_visual)
                if best is None or candidate < best:
                    best = candidate
    if best is None:
        raise ValueError(
            "No DOP policy satisfies the FLOP and minimum-visual-token constraints"
        )

    divergence, neg_flops, depth_a, depth_p, n_visual = best
    selected_flops = -neg_flops
    return {
        "mha_depth": int(depth_a),
        "mlp_depth": int(depth_p),
        "visual_tokens": int(n_visual),
        "estimated_divergence": float(divergence),
        "flops": int(selected_flops),
        "full_flops": int(full_flops),
        "flops_ratio": float(selected_flops / full_flops),
        "target_flops_ratio": float(target_flops_ratio),
        "token_only_equivalent": int(n_equiv),
        "minimum_visual_tokens": int(n_min),
    }


def optimize_dop_policy_reference_geometry(
    *,
    mha_divergence: Mapping[int, float],
    mlp_divergence: Mapping[int, float],
    token_divergence: Mapping[int, float],
    num_layers: int,
    max_visual_tokens: int,
    reference_visual_tokens: int,
    reference_nonvisual_tokens: int,
    hidden_size: int,
    intermediate_size: int,
    num_attention_heads: int,
    num_key_value_heads: int,
    target_flops_ratio: float,
    min_visual_alpha: float = 1.5,
) -> Dict[str, Any]:
    """Solve DOP on the fixed formal decoder-entry geometry.

    ``max_visual_tokens`` remains the VisPruner control range (729 patches per
    crop in the calibration curves).  A candidate control coordinate is mapped
    proportionally to the user's common 3,215-token decoder entry before MHA
    and MLP FLOPs are counted.  Thus TextVQA selects the divergence curves only;
    its sample-specific AnyRes/text lengths cannot alter the formal budget.
    """

    layers = int(num_layers)
    control_max = int(max_visual_tokens)
    visual_full = int(reference_visual_tokens)
    text = int(reference_nonvisual_tokens)
    if layers != 28:
        raise ValueError("Formal DOP FLOPs reference requires 28 layers")
    if control_max < 1 or visual_full < 1 or text < 1:
        raise ValueError("DOP fixed-reference dimensions must be positive")
    if not 0.0 < float(target_flops_ratio) <= 1.0:
        raise ValueError("target_flops_ratio must satisfy 0 < ratio <= 1")
    if float(min_visual_alpha) < 0.0:
        raise ValueError("min_visual_alpha must be >= 0")

    da = interpolate_divergence_curve(mha_divergence, layers)
    dp = interpolate_divergence_curve(mlp_divergence, layers)
    dv = interpolate_divergence_curve(token_divergence, control_max)
    minimum_control = max(1, min(int(value) for value in token_divergence))

    def retained_visual(control: int) -> int:
        # Round half up rather than Python's ties-to-even so the mapping is
        # explicit and stable across every generator/validator.
        return max(
            1,
            min(
                visual_full,
                int(math.floor(visual_full * int(control) / control_max + 0.5)),
            ),
        )

    flop_args = dict(
        num_layers=layers,
        nonvisual_tokens=text,
        hidden_size=int(hidden_size),
        intermediate_size=int(intermediate_size),
        num_attention_heads=int(num_attention_heads),
        num_key_value_heads=int(num_key_value_heads),
    )
    full_flops = qwen2_decoder_flops(
        visual_tokens=visual_full,
        mha_depth=layers,
        mlp_depth=layers,
        **flop_args,
    )
    budget = float(target_flops_ratio) * full_flops
    token_only_costs = {
        control: qwen2_decoder_flops(
            visual_tokens=retained_visual(control),
            mha_depth=layers,
            mlp_depth=layers,
            **flop_args,
        )
        for control in range(minimum_control, control_max + 1)
    }
    token_only_equivalent = min(
        token_only_costs,
        key=lambda control: abs(token_only_costs[control] - budget),
    )
    minimum_tokens_per_crop = min(
        control_max,
        max(
            minimum_control,
            int(math.ceil(float(min_visual_alpha) * token_only_equivalent)),
        ),
    )

    best: Optional[Tuple[float, int, int, int, int]] = None
    for depth_a in range(layers + 1):
        for depth_p in range(layers + 1):
            for control in range(minimum_tokens_per_crop, control_max + 1):
                selected_visual = retained_visual(control)
                flops = qwen2_decoder_flops(
                    visual_tokens=selected_visual,
                    mha_depth=depth_a,
                    mlp_depth=depth_p,
                    **flop_args,
                )
                if flops > budget:
                    continue
                divergence = da[depth_a] + dp[depth_p] + dv[control]
                candidate = (
                    float(divergence),
                    -int(flops),
                    depth_a,
                    depth_p,
                    control,
                )
                if best is None or candidate < best:
                    best = candidate
    if best is None:
        raise ValueError(
            "No fixed-reference DOP policy satisfies the FLOP and minimum-token constraints"
        )

    divergence, neg_flops, depth_a, depth_p, control = best
    selected_flops = int(-neg_flops)
    selected_visual = retained_visual(control)
    ratio = float(selected_flops / full_flops)
    return {
        "mha_depth": int(depth_a),
        "mlp_depth": int(depth_p),
        "tokens_per_crop": int(control),
        "visual_tokens": int(selected_visual),
        "estimated_divergence": float(divergence),
        "flops": selected_flops,
        "full_flops": int(full_flops),
        "flops_ratio": ratio,
        "mean_flops_ratio": ratio,
        "dataset_flops_ratio": ratio,
        "unweighted_mean_sample_flops_ratio": ratio,
        "minimum_flops_ratio": ratio,
        "maximum_flops_ratio": ratio,
        "method_flops_sum": float(selected_flops),
        "baseline_flops_sum": float(full_flops),
        "sample_count": 1,
        "flops_aggregation": "sum_method_flops_div_sum_baseline_flops",
        "flops_reference_mode": "fixed_decoder_entry_geometry",
        "flops_reference_visual_tokens": visual_full,
        "flops_reference_nonvisual_tokens": text,
        "flops_reference_decoder_layers": layers,
        "target_flops_ratio": float(target_flops_ratio),
        "token_only_equivalent_per_crop": int(token_only_equivalent),
        "minimum_tokens_per_crop": int(minimum_tokens_per_crop),
        "visual_token_mean_retain_ratio": float(selected_visual / visual_full),
        "control_to_reference_visual_mapping": (
            "round_half_up(reference_visual_tokens*control/max_control)"
        ),
    }


def interpolate_sample_visual_tokens(
    sampled: Mapping[int, int], coordinate: int
) -> int:
    """Interpolate an AnyRes sample's measured visual-token count.

    OneVision's spatial restore unpads a sparse stitched grid and adds image
    newline tokens, so ``tokens_per_crop * crop_count`` is not the number of
    tokens entering the LLM.  DOP calibration measures that expanded count at
    every sparse token point.  This helper applies the same piecewise-linear
    interpolation as paper Eq. (8), but rounds to a feasible integer count.
    """

    if not sampled:
        raise ValueError("DOP sample visual-token mapping must not be empty")
    points = sorted((int(x), int(y)) for x, y in sampled.items())
    if any(x < 0 or y < 0 for x, y in points):
        raise ValueError("DOP sample visual-token coordinates/counts must be >= 0")
    if len({x for x, _ in points}) != len(points):
        raise ValueError("DOP sample visual-token coordinates must be unique")
    target = int(coordinate)
    if target <= points[0][0]:
        return int(points[0][1])
    if target >= points[-1][0]:
        return int(points[-1][1])
    for (x0, y0), (x1, y1) in zip(points, points[1:]):
        if x0 <= target <= x1:
            ratio = (target - x0) / float(x1 - x0)
            value = y0 + ratio * (y1 - y0)
            return max(0, int(round(value)))
    raise AssertionError("DOP visual-token interpolation did not find a segment")


def optimize_dop_policy_samplewise(
    *,
    mha_divergence: Mapping[int, float],
    mlp_divergence: Mapping[int, float],
    token_divergence: Mapping[int, float],
    sample_shapes: Sequence[Mapping[str, Any]],
    num_layers: int,
    max_visual_tokens: int,
    hidden_size: int,
    intermediate_size: int,
    num_attention_heads: int,
    num_key_value_heads: int,
    target_flops_ratio: float,
    min_visual_alpha: float = 1.5,
) -> Dict[str, Any]:
    """Solve DOP using summed exact per-sample OneVision FLOPs.

    ``max_visual_tokens`` is the maximum *control coordinate* (729 SigLIP
    patches per crop), while each sample supplies the measured mapping from
    that coordinate to its actual post-AnyRes visual-token count.  This avoids
    the mathematically incorrect shortcut ``F(E[V], E[T])``: attention FLOPs
    contain a quadratic sequence-length term, so the formal objective uses
    ``sum(F_method) / sum(F_baseline)`` (equivalently ``E[F_method] /
    E[F_baseline]``), not an unweighted mean of per-sample ratios.
    """

    if not 0 < float(target_flops_ratio) <= 1:
        raise ValueError("target_flops_ratio must satisfy 0 < ratio <= 1")
    if min_visual_alpha < 0:
        raise ValueError("min_visual_alpha must be >= 0")
    if not sample_shapes:
        raise ValueError("sample_shapes must contain at least one sample")
    if int(num_layers) < 1 or int(max_visual_tokens) < 1:
        raise ValueError("DOP layer/token maxima must be positive")
    if int(num_attention_heads) < 1 or int(num_key_value_heads) < 1:
        raise ValueError("DOP FLOPs require positive attention-head counts")
    if int(hidden_size) % int(num_attention_heads):
        raise ValueError("hidden_size must be divisible by num_attention_heads")

    da = interpolate_divergence_curve(mha_divergence, int(num_layers))
    dp = interpolate_divergence_curve(mlp_divergence, int(num_layers))
    dv = interpolate_divergence_curve(token_divergence, int(max_visual_tokens))
    minimum_coordinate = min(int(value) for value in token_divergence)
    if minimum_coordinate < 1:
        minimum_coordinate = 1

    parsed_samples = []
    for index, raw in enumerate(sample_shapes):
        nonvisual = int(raw.get("nonvisual_tokens", -1))
        baseline_visual = int(raw.get("baseline_visual_tokens", -1))
        mapping_raw = raw.get("visual_tokens_by_per_crop")
        if nonvisual < 1 or baseline_visual < 1 or not isinstance(
            mapping_raw, Mapping
        ):
            raise ValueError(f"Invalid DOP sample shape at index {index}")
        mapping = {int(key): int(value) for key, value in mapping_raw.items()}
        if mapping.get(int(max_visual_tokens)) != baseline_visual:
            raise ValueError(
                "DOP sample maximum-coordinate visual count must equal baseline"
            )
        parsed_samples.append((nonvisual, baseline_visual, mapping))

    head_dim = int(hidden_size) // int(num_attention_heads)
    kv_dim = int(num_key_value_heads) * head_dim

    def mha_cost(tokens: int) -> int:
        n = int(tokens)
        return int(
            4 * n * int(hidden_size) * int(hidden_size)
            + 4 * n * int(hidden_size) * kv_dim
            + 4 * n * n * int(hidden_size)
        )

    def mlp_cost(tokens: int) -> int:
        return int(6 * int(tokens) * int(hidden_size) * int(intermediate_size))

    # Cache the four mean per-layer components for every feasible crop-token
    # coordinate.  Policy enumeration is then linear in the number of policy
    # triples rather than repeatedly looping over samples and decoder layers.
    coordinate_costs: Dict[int, Tuple[float, float, float, float]] = {}
    coordinate_visual_counts: Dict[int, Tuple[int, ...]] = {}
    for coordinate in range(minimum_coordinate, int(max_visual_tokens) + 1):
        visual_counts = tuple(
            interpolate_sample_visual_tokens(mapping, coordinate)
            for _, _, mapping in parsed_samples
        )
        coordinate_visual_counts[coordinate] = visual_counts
        mha_full = 0
        mha_text = 0
        mlp_full = 0
        mlp_text = 0
        for (nonvisual, _, _), visual in zip(parsed_samples, visual_counts):
            mha_full += mha_cost(nonvisual + visual)
            mha_text += mha_cost(nonvisual)
            mlp_full += mlp_cost(nonvisual + visual)
            mlp_text += mlp_cost(nonvisual)
        count = float(len(parsed_samples))
        coordinate_costs[coordinate] = (
            mha_full / count,
            mha_text / count,
            mlp_full / count,
            mlp_text / count,
        )

    def policy_flops(coordinate: int, depth_a: int, depth_p: int) -> float:
        mha_full, mha_text, mlp_full, mlp_text = coordinate_costs[coordinate]
        return float(
            int(depth_a) * mha_full
            + (int(num_layers) - int(depth_a)) * mha_text
            + int(depth_p) * mlp_full
            + (int(num_layers) - int(depth_p)) * mlp_text
        )

    full_flops = policy_flops(
        int(max_visual_tokens), int(num_layers), int(num_layers)
    )
    budget = float(target_flops_ratio) * full_flops
    token_only_costs = {
        coordinate: policy_flops(coordinate, int(num_layers), int(num_layers))
        for coordinate in coordinate_costs
    }
    token_only_equivalent = min(
        token_only_costs,
        key=lambda coordinate: abs(token_only_costs[coordinate] - budget),
    )
    minimum_tokens_per_crop = min(
        int(max_visual_tokens),
        max(
            minimum_coordinate,
            int(math.ceil(float(min_visual_alpha) * token_only_equivalent)),
        ),
    )

    best: Optional[Tuple[float, float, int, int, int]] = None
    for coordinate in range(
        minimum_tokens_per_crop, int(max_visual_tokens) + 1
    ):
        for depth_a in range(int(num_layers) + 1):
            for depth_p in range(int(num_layers) + 1):
                flops = policy_flops(coordinate, depth_a, depth_p)
                if flops > budget:
                    continue
                divergence = da[depth_a] + dp[depth_p] + dv[coordinate]
                candidate = (
                    float(divergence),
                    -float(flops),
                    depth_a,
                    depth_p,
                    coordinate,
                )
                if best is None or candidate < best:
                    best = candidate
    if best is None:
        raise ValueError(
            "No samplewise DOP policy satisfies the FLOP and minimum-token constraints"
        )

    divergence, neg_flops, depth_a, depth_p, coordinate = best
    selected_flops = -neg_flops
    selected_visual = coordinate_visual_counts[coordinate]
    baseline_visual = [sample[1] for sample in parsed_samples]
    selected_sample_flops = []
    baseline_sample_flops = []
    for (nonvisual, baseline_count, _), selected_count in zip(
        parsed_samples, selected_visual
    ):
        selected_sample_flops.append(
            int(depth_a) * mha_cost(nonvisual + selected_count)
            + (int(num_layers) - int(depth_a)) * mha_cost(nonvisual)
            + int(depth_p) * mlp_cost(nonvisual + selected_count)
            + (int(num_layers) - int(depth_p)) * mlp_cost(nonvisual)
        )
        baseline_sample_flops.append(
            int(num_layers)
            * (
                mha_cost(nonvisual + baseline_count)
                + mlp_cost(nonvisual + baseline_count)
            )
        )
    selected_sum = math.fsum(selected_sample_flops)
    baseline_sum = math.fsum(baseline_sample_flops)
    sample_ratios = [
        selected / baseline
        for selected, baseline in zip(
            selected_sample_flops, baseline_sample_flops
        )
    ]
    dataset_ratio = selected_sum / baseline_sum
    return {
        "mha_depth": int(depth_a),
        "mlp_depth": int(depth_p),
        "tokens_per_crop": int(coordinate),
        "estimated_divergence": float(divergence),
        "mean_flops": float(selected_flops),
        "mean_full_flops": float(full_flops),
        "flops_ratio": float(dataset_ratio),
        "mean_flops_ratio": float(dataset_ratio),
        "dataset_flops_ratio": float(dataset_ratio),
        "unweighted_mean_sample_flops_ratio": float(
            math.fsum(sample_ratios) / len(sample_ratios)
        ),
        "minimum_flops_ratio": float(min(sample_ratios)),
        "maximum_flops_ratio": float(max(sample_ratios)),
        "method_flops_sum": float(selected_sum),
        "baseline_flops_sum": float(baseline_sum),
        "flops_aggregation": "sum_method_flops_div_sum_baseline_flops",
        "target_flops_ratio": float(target_flops_ratio),
        "token_only_equivalent_per_crop": int(token_only_equivalent),
        "minimum_tokens_per_crop": int(minimum_tokens_per_crop),
        "sample_count": len(parsed_samples),
        "selected_visual_tokens": {
            "minimum": min(selected_visual),
            "maximum": max(selected_visual),
            "mean": float(sum(selected_visual) / len(selected_visual)),
        },
        "baseline_visual_tokens": {
            "minimum": min(baseline_visual),
            "maximum": max(baseline_visual),
            "mean": float(sum(baseline_visual) / len(baseline_visual)),
        },
        "visual_token_mean_retain_ratio": float(
            sum(selected_visual) / sum(baseline_visual)
        ),
        "visual_count_interpolation": (
            "piecewise_linear_per_sample_post_anyres_counts"
        ),
    }
