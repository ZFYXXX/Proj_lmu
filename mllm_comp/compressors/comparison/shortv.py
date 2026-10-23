"""ShortV operation pruning for LLaVA-OneVision/Qwen2.

ShortV freezes visual residual rows in low-contribution decoder layers.  In a
replaced layer, non-visual queries still attend the *complete* visual/text K/V
sequence, while visual Q/O rows and visual MLP rows are not evaluated.  This
is intentionally different from DOP's deeper MHA bypass, where visual rows are
removed from both the query and K/V sides.

The original release hard-codes a 35-token prefix and a contiguous LLaVA-1.5
visual block.  This port consumes the exact post-projector OneVision spans
recorded by ``llava_arch`` so AnyRes lengths and multiple image blocks do not
need architecture-specific constants.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, Optional, Sequence, Tuple

import torch

from .qwen2_flops import aggregate_samplewise_flops


SHORTV_DEFAULTS: Dict[str, Any] = {
    "method": "shortv",
    "profile": "ov_qwen2_gqa256_calibrated",
    "replaced_layers": [],
    "target_flops_ratio": 0.67,
    "target_flops_tolerance": 0.005,
    "calibration_dataset": "gqa",
    "calibration_samples": 256,
    "calibration_seed": 0,
    "calibration_asset_path": "",
    "calibration_asset_sha256": "",
    "calibrated_mean_flops_ratio": None,
    "calibrated_minimum_flops_ratio": None,
    "calibrated_maximum_flops_ratio": None,
    "calibrated_absolute_target_error": None,
    "layer_ranking_source": "shortv_last_token_output_kl",
    "operation_backend": "indexed_nonvisual_q_full_kv",
    "prefill_only": True,
    "image_only": True,
    "allow_multiple_images": True,
    "log_first_n": 1,
}


def _layer_list(value: Any) -> list[int]:
    if isinstance(value, str):
        values: Iterable[Any] = [part for part in value.split(",") if part.strip()]
    elif isinstance(value, Sequence):
        values = value
    else:
        raise TypeError("ShortV replaced_layers must be a sequence or comma-separated string")
    layers = [int(item) for item in values]
    if len(set(layers)) != len(layers):
        raise ValueError("ShortV replaced_layers contains duplicates")
    return layers


def normalize_shortv_config(raw: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Validate a checkpoint-specific ShortV layer policy."""

    cfg = dict(SHORTV_DEFAULTS)
    if raw:
        unknown = sorted(set(raw) - set(cfg))
        if unknown:
            raise ValueError(f"Unknown ShortV config keys: {', '.join(unknown)}")
        cfg.update(raw)

    cfg["method"] = "shortv"
    cfg["profile"] = str(cfg["profile"])
    cfg["replaced_layers"] = _layer_list(cfg["replaced_layers"])
    cfg["target_flops_ratio"] = float(cfg["target_flops_ratio"])
    cfg["target_flops_tolerance"] = float(cfg["target_flops_tolerance"])
    cfg["calibration_dataset"] = str(cfg["calibration_dataset"]).strip().lower()
    cfg["calibration_samples"] = int(cfg["calibration_samples"])
    cfg["calibration_seed"] = int(cfg["calibration_seed"])
    cfg["calibration_asset_path"] = str(cfg["calibration_asset_path"])
    cfg["calibration_asset_sha256"] = str(
        cfg["calibration_asset_sha256"]
    ).strip().lower()
    for key in (
        "calibrated_mean_flops_ratio",
        "calibrated_minimum_flops_ratio",
        "calibrated_maximum_flops_ratio",
        "calibrated_absolute_target_error",
    ):
        cfg[key] = None if cfg[key] is None else float(cfg[key])
    cfg["layer_ranking_source"] = str(cfg["layer_ranking_source"])
    cfg["operation_backend"] = str(cfg["operation_backend"]).strip().lower()
    cfg["prefill_only"] = bool(cfg["prefill_only"])
    cfg["image_only"] = bool(cfg["image_only"])
    cfg["allow_multiple_images"] = bool(cfg["allow_multiple_images"])
    cfg["log_first_n"] = int(cfg["log_first_n"])

    if not cfg["replaced_layers"]:
        raise ValueError(
            "ShortV requires non-empty checkpoint-specific replaced_layers; "
            "run the GQA layer-contribution calibration first"
        )
    if min(cfg["replaced_layers"]) < 0:
        raise ValueError("ShortV replaced_layers must be non-negative")
    if not 0.0 < cfg["target_flops_ratio"] <= 1.0:
        raise ValueError("ShortV target_flops_ratio must satisfy 0 < ratio <= 1")
    if not 0.0 <= cfg["target_flops_tolerance"] < 1.0:
        raise ValueError("ShortV target_flops_tolerance must satisfy 0 <= value < 1")
    if not cfg["calibration_dataset"]:
        raise ValueError("ShortV calibration_dataset must be non-empty")
    if cfg["calibration_samples"] < 1:
        raise ValueError("ShortV calibration_samples must be positive")
    if cfg["calibration_asset_sha256"] and (
        len(cfg["calibration_asset_sha256"]) != 64
        or any(
            character not in "0123456789abcdef"
            for character in cfg["calibration_asset_sha256"]
        )
    ):
        raise ValueError("ShortV calibration_asset_sha256 must be a SHA-256 hex digest")
    for key in (
        "calibrated_mean_flops_ratio",
        "calibrated_minimum_flops_ratio",
        "calibrated_maximum_flops_ratio",
    ):
        if cfg[key] is not None and not 0.0 < cfg[key] <= 1.0:
            raise ValueError(f"ShortV {key} must satisfy 0 < ratio <= 1")
    if (
        cfg["calibrated_absolute_target_error"] is not None
        and cfg["calibrated_absolute_target_error"] < 0.0
    ):
        raise ValueError(
            "ShortV calibrated_absolute_target_error must be non-negative"
        )
    if cfg["operation_backend"] != "indexed_nonvisual_q_full_kv":
        raise ValueError(
            "ShortV requires operation_backend='indexed_nonvisual_q_full_kv'"
        )
    if not cfg["prefill_only"]:
        raise ValueError("ShortV must be prefill_only to preserve normal decoding")
    if cfg["log_first_n"] < 0:
        raise ValueError("ShortV log_first_n must be non-negative")
    return cfg


def _validate_visual_spans(
    spans: Iterable[Sequence[int]], sequence_length: int
) -> Tuple[Tuple[int, int], ...]:
    normalized = tuple((int(item[0]), int(item[1])) for item in spans)
    previous_end = 0
    for start, end in normalized:
        if not 0 <= start < end <= int(sequence_length):
            raise ValueError(
                f"Invalid ShortV visual span [{start}, {end}) for sequence "
                f"length {sequence_length}"
            )
        if start < previous_end:
            raise ValueError("ShortV visual spans must be sorted and non-overlapping")
        previous_end = end
    return normalized


def resolve_shortv_visual_spans(
    runtime: Optional[dict], sequence_length: int
) -> Optional[Tuple[Tuple[int, int], ...]]:
    """Resolve exact post-projector visual blocks for one prefill sample."""

    if not runtime or not runtime.get("has_image"):
        return None
    if int(runtime.get("batch_size", 0)) != 1:
        runtime["skip_reason"] = "ShortV requires batch_size=1"
        return None

    exact = runtime.get("expanded_visual_spans")
    exact_length = runtime.get("expanded_prefill_seq_len")
    if exact and int(exact_length or -1) == int(sequence_length):
        spans = _validate_visual_spans(exact, sequence_length)
    else:
        placeholders = runtime.get("placeholder_positions") or []
        if len(placeholders) != 1:
            runtime["skip_reason"] = (
                "ShortV needs exact expanded spans when a prompt contains "
                "multiple image placeholders"
            )
            return None
        raw_length = int(runtime.get("orig_seq_len", sequence_length))
        visual_length = int(sequence_length) - raw_length + 1
        start = int(placeholders[0])
        spans = _validate_visual_spans(
            ((start, start + visual_length),), sequence_length
        )

    if not runtime.get("allow_multiple_images", True) and len(spans) != 1:
        runtime["skip_reason"] = "ShortV config requires exactly one image"
        return None
    runtime.pop("skip_reason", None)
    runtime["shortv_visual_spans"] = [list(span) for span in spans]
    runtime["visual_tokens"] = sum(end - start for start, end in spans)
    return spans


def complement_indices(
    sequence_length: int,
    visual_spans: Iterable[Sequence[int]],
    *,
    device: Optional[torch.device] = None,
) -> torch.LongTensor:
    """Return every non-visual sequence index in absolute causal order."""

    spans = _validate_visual_spans(visual_spans, int(sequence_length))
    pieces = []
    cursor = 0
    for start, end in spans:
        if cursor < start:
            pieces.append(torch.arange(cursor, start, device=device, dtype=torch.long))
        cursor = end
    if cursor < int(sequence_length):
        pieces.append(
            torch.arange(cursor, int(sequence_length), device=device, dtype=torch.long)
        )
    if not pieces:
        return torch.empty(0, device=device, dtype=torch.long)
    return torch.cat(pieces)


def shortv_operation_indices(
    runtime: Optional[dict],
    sequence_length: int,
    *,
    device: Optional[torch.device] = None,
) -> Optional[Tuple[torch.LongTensor, torch.LongTensor]]:
    """Build ShortV's non-visual Q/MLP and complete K/V index vectors."""

    spans = resolve_shortv_visual_spans(runtime, sequence_length)
    if spans is None:
        return None
    nonvisual = complement_indices(sequence_length, spans, device=device)
    if nonvisual.numel() == 0:
        if runtime is not None:
            runtime["skip_reason"] = "ShortV cannot run without a non-visual query"
        return None
    complete = torch.arange(sequence_length, device=device, dtype=torch.long)
    return nonvisual, complete


def qwen2_shortv_layer_flops(
    *,
    nonvisual_tokens: int,
    visual_tokens: int,
    hidden_size: int,
    intermediate_size: int,
    num_attention_heads: int,
    num_key_value_heads: int,
    replaced: bool,
) -> int:
    """Count one Qwen2 layer with the paper's ShortV operator semantics.

    Multiply-adds count as two FLOPs.  A replaced layer evaluates Q/O and the
    SwiGLU MLP only for non-visual rows, but K/V for all rows and attention for
    ``nonvisual_queries x complete_keys``.  The grouped-query K/V width is
    derived from the target Qwen2 configuration rather than copied from the
    original LLaMA/MHA experiments.
    """

    values = (
        nonvisual_tokens,
        visual_tokens,
        hidden_size,
        intermediate_size,
        num_attention_heads,
        num_key_value_heads,
    )
    if any(int(value) < 0 for value in values):
        raise ValueError("ShortV FLOP dimensions must be non-negative")
    if num_attention_heads < 1 or num_key_value_heads < 1:
        raise ValueError("ShortV FLOPs require positive attention-head counts")
    if hidden_size % num_attention_heads:
        raise ValueError("hidden_size must be divisible by num_attention_heads")

    text = int(nonvisual_tokens)
    total = text + int(visual_tokens)
    query_rows = text if replaced else total
    mlp_rows = text if replaced else total
    key_rows = total
    head_dim = int(hidden_size) // int(num_attention_heads)
    kv_width = int(num_key_value_heads) * head_dim

    flops = 4 * query_rows * int(hidden_size) * int(hidden_size)  # Q + O
    flops += 4 * key_rows * int(hidden_size) * kv_width  # K + V
    flops += 4 * query_rows * key_rows * int(hidden_size)  # QK + AV
    flops += 6 * mlp_rows * int(hidden_size) * int(intermediate_size)  # SwiGLU
    return int(flops)


def qwen2_shortv_decoder_flops(
    *,
    num_layers: int,
    replaced_layer_count: int,
    nonvisual_tokens: int,
    visual_tokens: int,
    hidden_size: int,
    intermediate_size: int,
    num_attention_heads: int,
    num_key_value_heads: int,
) -> Dict[str, Any]:
    """Return full/ShortV decoder FLOPs and retention for one sample."""

    layers = int(num_layers)
    replaced = int(replaced_layer_count)
    if layers < 1 or not 0 <= replaced <= layers:
        raise ValueError("ShortV replaced_layer_count must lie in [0, num_layers]")
    common = dict(
        nonvisual_tokens=int(nonvisual_tokens),
        visual_tokens=int(visual_tokens),
        hidden_size=int(hidden_size),
        intermediate_size=int(intermediate_size),
        num_attention_heads=int(num_attention_heads),
        num_key_value_heads=int(num_key_value_heads),
    )
    full_layer = qwen2_shortv_layer_flops(replaced=False, **common)
    short_layer = qwen2_shortv_layer_flops(replaced=True, **common)
    full = layers * full_layer
    selected = (layers - replaced) * full_layer + replaced * short_layer
    return {
        "full_flops": int(full),
        "shortv_flops": int(selected),
        "flops_ratio": float(selected / full),
        "full_layer_flops": int(full_layer),
        "replaced_layer_flops": int(short_layer),
    }


def select_shortv_policy(
    *,
    layer_order: Sequence[int],
    token_shapes: Sequence[Sequence[int]],
    hidden_size: int,
    intermediate_size: int,
    num_attention_heads: int,
    num_key_value_heads: int,
    target_flops_ratio: float,
) -> Dict[str, Any]:
    """Choose the closest whole-layer ShortV budget on calibration samples.

    ``token_shapes`` contains ``(nonvisual_tokens, visual_tokens)`` for each
    sample.  Method and baseline FLOPs are separately summed across those real
    OneVision shapes before division.
    """

    order = [int(layer) for layer in layer_order]
    num_layers = len(order)
    if sorted(order) != list(range(num_layers)):
        raise ValueError("ShortV layer_order must be a permutation of decoder layers")
    if not token_shapes:
        raise ValueError("ShortV policy selection requires calibration token shapes")
    target = float(target_flops_ratio)
    if not 0.0 < target <= 1.0:
        raise ValueError("target_flops_ratio must satisfy 0 < ratio <= 1")

    curve = []
    for count in range(num_layers + 1):
        method_costs = []
        baseline_costs = []
        for shape in token_shapes:
            if len(shape) != 2:
                raise ValueError("Each ShortV token shape must be [nonvisual, visual]")
            result = qwen2_shortv_decoder_flops(
                num_layers=num_layers,
                replaced_layer_count=count,
                nonvisual_tokens=int(shape[0]),
                visual_tokens=int(shape[1]),
                hidden_size=hidden_size,
                intermediate_size=intermediate_size,
                num_attention_heads=num_attention_heads,
                num_key_value_heads=num_key_value_heads,
            )
            method_costs.append(float(result["shortv_flops"]))
            baseline_costs.append(float(result["full_flops"]))
        aggregate = aggregate_samplewise_flops(
            method_costs, baseline_costs
        )
        mean_ratio = float(aggregate["mean_flops_ratio"])
        curve.append(
            {
                "replaced_layer_count": count,
                **aggregate,
            }
        )

    selected = min(
        curve,
        key=lambda row: (
            abs(float(row["mean_flops_ratio"]) - target),
            # Prefer the non-over-budget neighbour when equally close.
            float(row["mean_flops_ratio"]) > target,
            int(row["replaced_layer_count"]),
        ),
    )
    count = int(selected["replaced_layer_count"])
    return {
        **selected,
        "replaced_layers": order[:count],
        "replaced_layer_count": count,
        "target_flops_ratio": target,
        "absolute_target_error": abs(float(selected["mean_flops_ratio"]) - target),
        "budget_curve": curve,
    }
