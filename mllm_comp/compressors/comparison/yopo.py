"""YOPO operation-pruning primitives for the OneVision Qwen2 decoder.

YOPO does not remove visual tokens.  It prunes *visual-related operations* by
masking visual attention columns for inactive heads, optionally replacing the
visual-to-visual causal block by a local neighbourhood, masking a suffix of
visual FFN channels, and disabling text-to-visual attention in deep layers.

The released Qwen2-VL train-free code enables a checkpoint-calibrated head mask
and the deep-layer cut, but leaves neighbour attention and FFN pruning disabled.
The public Qwen2-VL mask is retained only as an explicit reproduction option;
OneVision checkpoints should load a mask calibrated on their own weights.
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch

from .qwen2_flops import aggregate_samplewise_flops


YOPO_DEFAULTS: Dict[str, Any] = {
    "method": "yopo",
    "profile": "official_qwen2vl_train_free",
    "head_mask_source": "official_qwen2vl_mask14",
    "head_mask_path": None,
    "head_prune_start_layer": 2,
    "layer_drop_start_layer": 24,
    "layer_drop_query_scope": "post_visual_text",
    # The released Qwen2-VL train-free branch applies the deep-layer cut to
    # logits (and therefore renormalizes).  The released LLaVA training branch
    # instead multiplies visual-key probabilities after softmax.  Keep the two
    # source implementations explicit rather than silently mixing them.
    "layer_drop_backend": "qwen2vl_pre_softmax",
    # The public repository simulates the cut with masks.  Formal policies can
    # request the mathematically intended text-only indexed execution, which
    # also creates a correspondingly compact deep-layer KV cache.
    "layer_drop_operation_backend": "masked_simulation",
    # Only the released LLaVA branch additionally masks per-head outputs for
    # visual query rows before O projection.
    "visual_query_head_mask": False,
    # The public simulation also masks decode queries.  Set this for the
    # experiment contract in which all pruning is restricted to prefill.
    "prefill_only": False,
    "neighbor_window_size": None,
    "neighbor_grid_width": None,
    "ffn_keep_ratio": 1.0,
    "mask_backend": "released_simulation",
    "renormalize_after_head_mask": False,
    "single_image_only": True,
    "image_only": True,
    "log_first_n": 1,
}


def normalize_yopo_config(raw: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    cfg = dict(YOPO_DEFAULTS)
    if raw:
        cfg.update(raw)

    cfg["method"] = "yopo"
    cfg["profile"] = str(cfg["profile"])
    cfg["head_mask_source"] = str(cfg["head_mask_source"]).lower()
    cfg["head_prune_start_layer"] = int(cfg["head_prune_start_layer"])
    cfg["layer_drop_query_scope"] = str(cfg["layer_drop_query_scope"]).lower()
    cfg["layer_drop_backend"] = str(cfg["layer_drop_backend"]).lower()
    cfg["layer_drop_operation_backend"] = str(
        cfg["layer_drop_operation_backend"]
    ).lower()
    cfg["visual_query_head_mask"] = bool(cfg["visual_query_head_mask"])
    cfg["prefill_only"] = bool(cfg["prefill_only"])
    cfg["ffn_keep_ratio"] = float(cfg["ffn_keep_ratio"])
    cfg["mask_backend"] = str(cfg["mask_backend"]).lower()
    cfg["renormalize_after_head_mask"] = bool(
        cfg["renormalize_after_head_mask"]
    )
    cfg["single_image_only"] = bool(cfg["single_image_only"])
    cfg["image_only"] = bool(cfg["image_only"])
    cfg["log_first_n"] = int(cfg["log_first_n"])

    if cfg["layer_drop_start_layer"] is not None:
        cfg["layer_drop_start_layer"] = int(cfg["layer_drop_start_layer"])
    if cfg["neighbor_window_size"] is not None:
        cfg["neighbor_window_size"] = int(cfg["neighbor_window_size"])
    if cfg["neighbor_grid_width"] is not None:
        cfg["neighbor_grid_width"] = int(cfg["neighbor_grid_width"])

    if cfg["head_mask_source"] not in {
        "official_qwen2vl_mask14",
        "file",
        "none",
    }:
        raise ValueError(
            "YOPO head_mask_source must be official_qwen2vl_mask14, file, or none"
        )
    if cfg["head_mask_source"] == "file" and not cfg.get("head_mask_path"):
        raise ValueError("YOPO head_mask_source=file requires head_mask_path")
    if cfg["head_prune_start_layer"] < 0:
        raise ValueError("YOPO head_prune_start_layer must be >= 0")
    if (
        cfg["layer_drop_start_layer"] is not None
        and cfg["layer_drop_start_layer"] < 0
    ):
        raise ValueError("YOPO layer_drop_start_layer must be >= 0 or null")
    if cfg["layer_drop_query_scope"] != "post_visual_text":
        raise ValueError(
            "YOPO currently supports layer_drop_query_scope=post_visual_text"
        )
    if cfg["layer_drop_backend"] not in {
        "qwen2vl_pre_softmax",
        "llava_post_softmax",
    }:
        raise ValueError(
            "YOPO layer_drop_backend must be qwen2vl_pre_softmax or "
            "llava_post_softmax"
        )
    if cfg["layer_drop_operation_backend"] not in {
        "masked_simulation",
        "indexed_deep_text_only",
    }:
        raise ValueError(
            "YOPO layer_drop_operation_backend must be masked_simulation or "
            "indexed_deep_text_only"
        )
    if (
        cfg["layer_drop_operation_backend"] == "indexed_deep_text_only"
        and cfg["prefill_only"]
    ):
        raise ValueError(
            "YOPO indexed deep-layer pruning must also preserve its compact "
            "cache during decode; prefill_only is unsupported for this backend"
        )
    if cfg["neighbor_window_size"] is not None:
        if cfg["neighbor_window_size"] < 1 or cfg["neighbor_window_size"] % 2 != 1:
            raise ValueError("YOPO neighbor_window_size must be a positive odd integer")
    if cfg["neighbor_grid_width"] is not None and cfg["neighbor_grid_width"] < 1:
        raise ValueError("YOPO neighbor_grid_width must be >= 1")
    if not 0.0 < cfg["ffn_keep_ratio"] <= 1.0:
        raise ValueError("YOPO ffn_keep_ratio must satisfy 0 < ratio <= 1")
    if cfg["mask_backend"] != "released_simulation":
        raise ValueError("YOPO currently reproduces mask_backend=released_simulation")
    if cfg["renormalize_after_head_mask"]:
        raise ValueError(
            "The released YOPO code does not renormalize attention after head masking"
        )
    return cfg


def yopo_uses_indexed_text_only(
    cfg: Optional[Dict[str, Any]], layer_idx: Optional[int] = None
) -> bool:
    if not cfg or str(cfg.get("layer_drop_operation_backend")) != (
        "indexed_deep_text_only"
    ):
        return False
    boundary = cfg.get("layer_drop_start_layer")
    if boundary is None:
        return False
    return layer_idx is None or int(layer_idx) >= int(boundary)


def resolve_yopo_visual_span(
    runtime: Optional[dict], sequence_length: int
) -> Optional[Tuple[int, int]]:
    """Infer the exact dynamic OneVision visual span from its placeholder."""

    if not runtime or not runtime.get("has_image"):
        return None
    if int(runtime.get("batch_size", 0)) != 1:
        runtime["skip_reason"] = "YOPO currently requires batch_size=1"
        return None
    placeholder_positions = runtime.get("placeholder_positions") or []
    if len(placeholder_positions) != 1:
        runtime["skip_reason"] = "YOPO requires exactly one image placeholder"
        return None

    raw_length = int(runtime["orig_seq_len"])
    visual_length = int(sequence_length) - raw_length + 1
    visual_start = int(placeholder_positions[0])
    visual_end = visual_start + visual_length
    if visual_length <= 0 or visual_start < 0 or visual_end > int(sequence_length):
        runtime["skip_reason"] = (
            f"Invalid YOPO visual span [{visual_start}, {visual_end}) for "
            f"sequence length {sequence_length}"
        )
        return None

    runtime.pop("skip_reason", None)
    runtime["visual_start"] = visual_start
    runtime["visual_end"] = visual_end
    runtime["visual_tokens"] = visual_length
    relative_segments = runtime.get("yopo_neighbor_segments_relative")
    if relative_segments is not None:
        if not isinstance(relative_segments, list) or not relative_segments:
            raise ValueError("YOPO AnyRes neighbor layout must be a non-empty list")
        absolute_segments = []
        coverage = torch.zeros(visual_length, dtype=torch.bool)
        for raw_segment in relative_segments:
            if not isinstance(raw_segment, dict):
                raise ValueError("YOPO neighbor segment must be a mapping")
            start = int(raw_segment["start"])
            end = int(raw_segment["end"])
            width = int(raw_segment["grid_width"])
            if not 0 <= start < end <= visual_length or width < 1:
                raise ValueError("YOPO neighbor segment is outside the visual span")
            if (end - start) % width:
                raise ValueError("YOPO neighbor segment is not a rectangular grid")
            if bool(coverage[start:end].any()):
                raise ValueError("YOPO neighbor segments overlap")
            coverage[start:end] = True
            absolute_segments.append(
                {
                    "start": visual_start + start,
                    "end": visual_start + end,
                    "grid_width": width,
                }
            )
        if not bool(coverage.all()):
            raise ValueError("YOPO neighbor segments must cover every visual token")
        runtime["yopo_neighbor_segments"] = absolute_segments
    return visual_start, visual_end


def _official_mask_asset_path() -> Path:
    return Path(__file__).resolve().parent / "assets" / "yopo_qwen2vl_mask14.json"


def _mask_from_active_heads(
    active_heads: Any, *, layer_count: int, head_count: int
) -> torch.BoolTensor:
    if not isinstance(active_heads, list) or len(active_heads) != layer_count:
        raise ValueError(
            f"YOPO active-head list must contain {layer_count} layers, got "
            f"{len(active_heads) if isinstance(active_heads, list) else type(active_heads)}"
        )
    mask = torch.zeros(layer_count, head_count, dtype=torch.bool)
    for layer_idx, heads in enumerate(active_heads):
        if not isinstance(heads, list):
            raise ValueError(f"YOPO active heads for layer {layer_idx} must be a list")
        if len(set(int(head) for head in heads)) != len(heads):
            raise ValueError(f"YOPO active heads for layer {layer_idx} contain duplicates")
        for head in heads:
            head = int(head)
            if head < 0 or head >= head_count:
                raise ValueError(
                    f"YOPO head {head} in layer {layer_idx} is outside [0, {head_count})"
                )
            mask[layer_idx, head] = True
    return mask


def _load_mask_file(path: str) -> torch.Tensor:
    resolved = os.path.abspath(os.path.expanduser(path))
    if not os.path.isfile(resolved):
        raise FileNotFoundError(f"YOPO head mask does not exist: {resolved}")
    suffix = Path(resolved).suffix.lower()
    if suffix == ".json":
        with open(resolved, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if isinstance(payload, dict) and "active_heads" in payload:
            return payload["active_heads"]  # type: ignore[return-value]
        return torch.as_tensor(payload)
    if suffix == ".npy":
        import numpy as np

        return torch.from_numpy(np.load(resolved))
    try:
        payload = torch.load(resolved, map_location="cpu", weights_only=True)
    except TypeError:  # torch < 2.0 compatibility
        payload = torch.load(resolved, map_location="cpu")
    if isinstance(payload, dict):
        for key in ("mask", "head_mask", "active_mask"):
            if key in payload:
                payload = payload[key]
                break
    return torch.as_tensor(payload)


def load_yopo_head_mask(
    cfg: Dict[str, Any], *, layer_count: int, head_count: int
) -> torch.BoolTensor:
    """Load and validate the static sample-agnostic active-head mask."""

    source = cfg["head_mask_source"]
    if source == "none":
        return torch.ones(layer_count, head_count, dtype=torch.bool)
    if source == "official_qwen2vl_mask14":
        with open(_official_mask_asset_path(), "r", encoding="utf-8") as handle:
            payload = json.load(handle)
        expected = payload.get("shape")
        if expected != [layer_count, head_count]:
            raise ValueError(
                "The official YOPO Qwen2-VL mask is 28x28 and cannot be silently "
                f"applied to a {layer_count}x{head_count} decoder"
            )
        return _mask_from_active_heads(
            payload["active_heads"],
            layer_count=layer_count,
            head_count=head_count,
        )

    payload = _load_mask_file(str(cfg["head_mask_path"]))
    if isinstance(payload, list):
        return _mask_from_active_heads(
            payload, layer_count=layer_count, head_count=head_count
        )
    mask = torch.as_tensor(payload, dtype=torch.bool, device="cpu")
    if tuple(mask.shape) != (layer_count, head_count):
        raise ValueError(
            f"YOPO head mask must be [{layer_count}, {head_count}], got {tuple(mask.shape)}"
        )
    return mask


def build_yopo_neighbor_allow_mask(
    visual_tokens: int,
    *,
    window_size: int,
    grid_width: Optional[int] = None,
    device: Optional[torch.device] = None,
) -> torch.BoolTensor:
    """Return the public-code Chebyshev local window for a rectangular grid.

    The released LLaVA code names the setting ``window`` and uses
    ``(window - 1) / 2`` as coordinate radius.  It also requires a complete
    rectangular visual grid.  AnyRes sequences with base/newline tokens must
    therefore provide an explicit compatible grid or leave this optional
    component disabled, as in the released Qwen2-VL train-free profile.
    """

    visual_tokens = int(visual_tokens)
    if visual_tokens < 1:
        raise ValueError("YOPO visual_tokens must be positive")
    if grid_width is None:
        root = int(math.isqrt(visual_tokens))
        if root * root != visual_tokens:
            raise ValueError(
                "YOPO neighbor attention needs neighbor_grid_width for a "
                "non-square visual sequence"
            )
        grid_width = root
    grid_width = int(grid_width)
    if visual_tokens % grid_width != 0:
        raise ValueError(
            f"YOPO visual length {visual_tokens} is not divisible by grid width {grid_width}"
        )
    rows = torch.arange(visual_tokens, device=device) // grid_width
    cols = torch.arange(visual_tokens, device=device) % grid_width
    radius = (int(window_size) - 1) // 2
    return (rows[:, None] - rows[None, :]).abs().le(radius) & (
        cols[:, None] - cols[None, :]
    ).abs().le(radius)


def build_yopo_runtime_neighbor_allow_mask(
    visual_tokens: int,
    *,
    window_size: int,
    grid_width: Optional[int],
    runtime: Optional[dict],
    device: Optional[torch.device] = None,
) -> torch.BoolTensor:
    """Build an AnyRes-aware block-local visual-neighbour mask.

    OneVision concatenates a 27x27 base view with a variable unpadded spatial
    grid (whose row-ending newline tokens form an extra column).  Treat each
    view as its own rectangular grid and forbid invented neighbours across the
    view boundary.  A fixed square/rectangular grid remains available for the
    original LLaVA-1.5 layout and tiny tests.
    """

    visual_tokens = int(visual_tokens)
    segments = runtime.get("yopo_neighbor_segments_relative") if runtime else None
    if not segments:
        return build_yopo_neighbor_allow_mask(
            visual_tokens,
            window_size=window_size,
            grid_width=grid_width,
            device=device,
        )
    allow = torch.zeros(
        visual_tokens, visual_tokens, dtype=torch.bool, device=device
    )
    coverage = torch.zeros(visual_tokens, dtype=torch.bool, device=device)
    for segment in segments:
        start = int(segment["start"])
        end = int(segment["end"])
        width = int(segment["grid_width"])
        if not 0 <= start < end <= visual_tokens or (end - start) % width:
            raise ValueError("Invalid YOPO AnyRes neighbor segment")
        if bool(coverage[start:end].any()):
            raise ValueError("YOPO AnyRes neighbor segments overlap")
        allow[start:end, start:end] = build_yopo_neighbor_allow_mask(
            end - start,
            window_size=window_size,
            grid_width=width,
            device=device,
        )
        coverage[start:end] = True
    if not bool(coverage.all()):
        raise ValueError("YOPO AnyRes neighbor layout does not cover visual tokens")
    return allow


def prepare_yopo_attention_mask(
    attention_mask: Optional[torch.Tensor],
    *,
    attn_logits: torch.Tensor,
    cfg: Optional[Dict[str, Any]],
    runtime: Optional[dict],
    past_len: int,
    layer_idx: Optional[int] = None,
) -> Optional[torch.Tensor]:
    """Replace the prefill V2V causal block with YOPO's local window."""

    if (
        not cfg
        or cfg.get("neighbor_window_size") is None
        or not runtime
        or yopo_uses_indexed_text_only(cfg, layer_idx)
    ):
        return attention_mask
    if past_len != 0 or attn_logits.shape[-2] != attn_logits.shape[-1]:
        return attention_mask
    visual_start = runtime.get("visual_start")
    visual_end = runtime.get("visual_end")
    if visual_start is None or visual_end is None:
        return attention_mask
    visual_start, visual_end = int(visual_start), int(visual_end)
    visual_tokens = visual_end - visual_start

    if attention_mask is None:
        raise ValueError(
            "YOPO neighbor attention requires the decoder's explicit 4-D causal mask"
        )
    if attention_mask.ndim != 4:
        raise ValueError("YOPO neighbor attention requires a 4-D additive mask")
    allow = build_yopo_runtime_neighbor_allow_mask(
        visual_tokens,
        window_size=int(cfg["neighbor_window_size"]),
        grid_width=cfg.get("neighbor_grid_width"),
        runtime=runtime,
        device=attn_logits.device,
    )
    local = torch.zeros(
        visual_tokens,
        visual_tokens,
        dtype=attention_mask.dtype,
        device=attention_mask.device,
    )
    local.masked_fill_(~allow, float("-inf"))
    result = attention_mask.clone()
    result[..., visual_start:visual_end, visual_start:visual_end] = local
    return result


def apply_yopo_layer_drop_to_logits(
    attn_logits: torch.Tensor,
    *,
    cfg: Optional[Dict[str, Any]],
    runtime: Optional[dict],
    layer_idx: int,
    past_len: int,
) -> torch.Tensor:
    """Disable post-image text queries from reading visual keys in deep layers."""

    if (
        not cfg
        or not runtime
        or cfg.get("layer_drop_start_layer") is None
        or yopo_uses_indexed_text_only(cfg, layer_idx)
    ):
        return attn_logits
    if str(cfg.get("layer_drop_backend")) != "qwen2vl_pre_softmax":
        return attn_logits
    if bool(cfg.get("prefill_only", False)) and (
        int(past_len) != 0 or int(attn_logits.shape[-2]) <= 1
    ):
        return attn_logits
    if int(layer_idx) < int(cfg["layer_drop_start_layer"]):
        return attn_logits
    visual_start = runtime.get("visual_start")
    visual_end = runtime.get("visual_end")
    if visual_start is None or visual_end is None:
        return attn_logits
    visual_start, visual_end = int(visual_start), int(visual_end)
    query_positions = torch.arange(
        past_len,
        past_len + attn_logits.shape[-2],
        device=attn_logits.device,
    )
    text_queries = query_positions >= visual_end
    if text_queries.any():
        attn_logits[:, :, text_queries, visual_start:visual_end] = float("-inf")
    return attn_logits


def apply_yopo_head_mask_to_probs(
    attention_probs: torch.Tensor,
    *,
    cfg: Optional[Dict[str, Any]],
    runtime: Optional[dict],
    active_head_mask: Optional[torch.Tensor],
    layer_idx: int,
    past_len: int = 0,
) -> torch.Tensor:
    """Multiply visual-key probabilities by the released static head mask.

    This intentionally happens *after* softmax and is not renormalized, exactly
    as in the released LLaVA and Qwen2-VL mask-simulation code.
    """

    if (
        not cfg
        or not runtime
        or active_head_mask is None
        or yopo_uses_indexed_text_only(cfg, layer_idx)
    ):
        return attention_probs
    if bool(cfg.get("prefill_only", False)) and (
        int(past_len) != 0 or int(attention_probs.shape[-2]) <= 1
    ):
        return attention_probs
    if int(layer_idx) < int(cfg["head_prune_start_layer"]):
        return attention_probs
    visual_start = runtime.get("visual_start")
    visual_end = runtime.get("visual_end")
    if visual_start is None or visual_end is None:
        return attention_probs
    visual_start, visual_end = int(visual_start), int(visual_end)
    row = active_head_mask[int(layer_idx)].to(
        device=attention_probs.device, dtype=attention_probs.dtype
    )
    if (
        str(cfg.get("layer_drop_backend")) == "llava_post_softmax"
        and cfg.get("layer_drop_start_layer") is not None
        and int(layer_idx) >= int(cfg["layer_drop_start_layer"])
    ):
        # The released LLaVA simulation multiplies the visual-key probability
        # block by zero after softmax.  It intentionally does not renormalize.
        row = torch.zeros_like(row)
    attention_probs[..., visual_start:visual_end].mul_(row.view(1, -1, 1, 1))
    return attention_probs


def prepare_yopo_sdpa_attention_mask(
    attention_mask: Optional[torch.Tensor],
    *,
    cfg: Optional[Dict[str, Any]],
    runtime: Optional[dict],
    layer_idx: int,
    past_len: int,
    query_length: int,
    key_length: int,
    batch_size: int,
    device: torch.device,
    dtype: torch.dtype,
) -> Optional[torch.Tensor]:
    """Build the exact pre-softmax YOPO mask without materialising QK.

    Post-softmax head/layer pruning is handled independently by masking the
    value tensor.  Because matrix multiplication is linear in values, this is
    algebraically identical to multiplying the corresponding probabilities,
    while allowing PyTorch's memory-efficient SDPA kernel on long prefills.
    """

    if not cfg or not runtime or yopo_uses_indexed_text_only(cfg, layer_idx):
        return attention_mask
    query_length = int(query_length)
    key_length = int(key_length)
    batch_size = int(batch_size)
    if attention_mask is not None:
        if (
            attention_mask.ndim != 4
            or int(attention_mask.shape[0]) != batch_size
            or int(attention_mask.shape[-2]) != query_length
            or int(attention_mask.shape[-1]) != key_length
        ):
            raise ValueError("YOPO SDPA mask does not match the current Q/K shape")
        result = attention_mask
    else:
        result = None

    visual_start = runtime.get("visual_start")
    visual_end = runtime.get("visual_end")
    if visual_start is None or visual_end is None:
        return result
    visual_start, visual_end = int(visual_start), int(visual_end)
    if not 0 <= visual_start < visual_end <= key_length:
        raise ValueError("YOPO visual span is outside the current KV sequence")

    neighbor_window = cfg.get("neighbor_window_size")
    neighbor_active = (
        neighbor_window is not None
        and int(past_len) == 0
        and query_length == key_length
    )
    pre_softmax_drop = (
        str(cfg.get("layer_drop_backend")) == "qwen2vl_pre_softmax"
        and cfg.get("layer_drop_start_layer") is not None
        and int(layer_idx) >= int(cfg["layer_drop_start_layer"])
        and not (
            bool(cfg.get("prefill_only", False))
            and (int(past_len) != 0 or query_length <= 1)
        )
    )
    if not neighbor_active and not pre_softmax_drop:
        return result
    if result is None:
        result = torch.zeros(
            batch_size,
            1,
            query_length,
            key_length,
            dtype=dtype,
            device=device,
        )
    else:
        result = result.clone()

    if neighbor_active:
        visual_tokens = visual_end - visual_start
        allow = build_yopo_runtime_neighbor_allow_mask(
            visual_tokens,
            window_size=int(neighbor_window),
            grid_width=cfg.get("neighbor_grid_width"),
            runtime=runtime,
            device=device,
        )
        local = torch.zeros(
            visual_tokens, visual_tokens, dtype=result.dtype, device=result.device
        )
        local.masked_fill_(~allow, float("-inf"))
        result[
            ..., visual_start:visual_end, visual_start:visual_end
        ] = local
        neighbor_bits = int(runtime.get("yopo_neighbor_attention_layer_bits", 0))
        neighbor_bits |= 1 << int(layer_idx)
        runtime["yopo_neighbor_attention_layer_bits"] = neighbor_bits
        runtime["yopo_neighbor_attention_layers"] = neighbor_bits.bit_count()

    if pre_softmax_drop:
        absolute_queries = torch.arange(
            int(past_len), int(past_len) + query_length, device=device
        )
        text_queries = absolute_queries >= visual_end
        if bool(text_queries.any()):
            result[:, :, text_queries, visual_start:visual_end] = float("-inf")
            drop_bits = int(runtime.get("yopo_layer_drop_layer_bits", 0))
            drop_bits |= 1 << int(layer_idx)
            runtime["yopo_layer_drop_layer_bits"] = drop_bits
            runtime["yopo_layer_drop_layers"] = drop_bits.bit_count()
    return result


def apply_yopo_value_head_mask(
    value_states: torch.Tensor,
    *,
    cfg: Optional[Dict[str, Any]],
    runtime: Optional[dict],
    active_head_mask: Optional[torch.Tensor],
    layer_idx: int,
    past_len: int,
) -> torch.Tensor:
    """Move YOPO's post-softmax visual-column mask onto V exactly."""

    if (
        not cfg
        or not runtime
        or active_head_mask is None
        or yopo_uses_indexed_text_only(cfg, layer_idx)
    ):
        return value_states
    if bool(cfg.get("prefill_only", False)) and (
        int(past_len) != 0 or int(value_states.shape[-2]) <= 1
    ):
        return value_states
    if int(layer_idx) < int(cfg["head_prune_start_layer"]):
        return value_states
    visual_start = runtime.get("visual_start")
    visual_end = runtime.get("visual_end")
    if visual_start is None or visual_end is None:
        return value_states
    visual_start, visual_end = int(visual_start), int(visual_end)
    if not 0 <= visual_start < visual_end <= int(value_states.shape[-2]):
        raise ValueError("YOPO visual span is outside the value tensor")
    row = active_head_mask[int(layer_idx)].to(
        device=value_states.device, dtype=value_states.dtype
    )
    if int(row.numel()) != int(value_states.shape[1]):
        raise ValueError("YOPO active-head row does not match repeated value heads")
    if (
        str(cfg.get("layer_drop_backend")) == "llava_post_softmax"
        and cfg.get("layer_drop_start_layer") is not None
        and int(layer_idx) >= int(cfg["layer_drop_start_layer"])
    ):
        row = torch.zeros_like(row)
        drop_bits = int(runtime.get("yopo_layer_drop_layer_bits", 0))
        drop_bits |= 1 << int(layer_idx)
        runtime["yopo_layer_drop_layer_bits"] = drop_bits
        runtime["yopo_layer_drop_layers"] = drop_bits.bit_count()
    if bool(row.all()):
        return value_states
    head_bits = int(runtime.get("yopo_head_mask_layer_bits", 0))
    head_bits |= 1 << int(layer_idx)
    runtime["yopo_head_mask_layer_bits"] = head_bits
    runtime["yopo_head_mask_layers"] = head_bits.bit_count()
    result = value_states.clone()
    result[:, :, visual_start:visual_end, :].mul_(row.view(1, -1, 1, 1))
    return result


def apply_yopo_visual_query_head_mask(
    attention_output: torch.Tensor,
    *,
    cfg: Optional[Dict[str, Any]],
    runtime: Optional[dict],
    active_head_mask: Optional[torch.Tensor],
    layer_idx: int,
    past_len: int,
) -> torch.Tensor:
    """Reproduce the released LLaVA visual-query output-head mask."""

    if (
        not cfg
        or not bool(cfg.get("visual_query_head_mask", False))
        or not runtime
        or active_head_mask is None
        or attention_output.ndim != 4
        or yopo_uses_indexed_text_only(cfg, layer_idx)
    ):
        return attention_output
    visual_start = runtime.get("visual_start")
    visual_end = runtime.get("visual_end")
    if visual_start is None or visual_end is None:
        return attention_output
    q_len = int(attention_output.shape[-2])
    q_positions = torch.arange(
        int(past_len), int(past_len) + q_len, device=attention_output.device
    )
    visual_queries = (q_positions >= int(visual_start)) & (
        q_positions < int(visual_end)
    )
    if not bool(visual_queries.any()):
        return attention_output
    row = active_head_mask[int(layer_idx)].to(
        device=attention_output.device, dtype=attention_output.dtype
    )
    if bool(row.all()):
        return attention_output
    query_bits = int(runtime.get("yopo_visual_query_mask_layer_bits", 0))
    query_bits |= 1 << int(layer_idx)
    runtime["yopo_visual_query_mask_layer_bits"] = query_bits
    runtime["yopo_visual_query_mask_layers"] = query_bits.bit_count()
    result = attention_output.clone()
    result[:, :, visual_queries, :] = result[:, :, visual_queries, :] * row.view(
        1, -1, 1, 1
    )
    return result


def yopo_visual_attention_ratio(
    attention_probs: torch.Tensor,
    *,
    visual_start: int,
    visual_end: int,
    query_index: int = -1,
) -> torch.Tensor:
    """Return YOPO's per-head visual/non-visual calibration ratio.

    The public Qwen2-VL collector sums the selected query's probability mass
    over visual keys and divides it by all system/text-key mass.  Calculating
    this directly inside eager attention avoids retaining an O(sequence^2)
    tensor for every decoder layer.
    """

    if attention_probs.ndim != 4:
        raise ValueError("YOPO calibration expects [batch, heads, queries, keys]")
    key_length = int(attention_probs.shape[-1])
    query_length = int(attention_probs.shape[-2])
    visual_start, visual_end = int(visual_start), int(visual_end)
    query_index = int(query_index)
    if not 0 <= visual_start < visual_end <= key_length:
        raise ValueError("YOPO calibration visual span is invalid")
    if query_index < 0:
        query_index += query_length
    if not 0 <= query_index < query_length:
        raise ValueError("YOPO calibration query index is invalid")
    query_probs = attention_probs[:, :, query_index, :].float()
    visual_mass = query_probs[..., visual_start:visual_end].sum(dim=-1)
    nonvisual_mass = query_probs[..., :visual_start].sum(dim=-1)
    nonvisual_mass = nonvisual_mass + query_probs[..., visual_end:].sum(dim=-1)
    return visual_mass / nonvisual_mass.clamp_min(torch.finfo(torch.float32).tiny)


def apply_yopo_ffn_mask(
    gate_states: torch.Tensor,
    up_states: torch.Tensor,
    *,
    cfg: Optional[Dict[str, Any]],
    runtime: Optional[dict],
    layer_idx: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Mask the public implementation's suffix FFN channels for visual rows."""

    if (
        not cfg
        or not runtime
        or float(cfg.get("ffn_keep_ratio", 1.0)) >= 1.0
        or yopo_uses_indexed_text_only(cfg, layer_idx)
    ):
        return gate_states, up_states
    # Normal generation has one full prefill followed by q_len=1 decode.  No
    # visual query is recomputed during decode.
    if gate_states.ndim != 3 or gate_states.shape[1] <= 1:
        return gate_states, up_states
    visual_start = runtime.get("visual_start")
    visual_end = runtime.get("visual_end")
    if visual_start is None or visual_end is None:
        return gate_states, up_states
    visual_start, visual_end = int(visual_start), int(visual_end)
    visual_end = min(visual_end, int(gate_states.shape[1]))
    if visual_start >= visual_end:
        return gate_states, up_states
    keep_channels = max(
        1, round(int(gate_states.shape[-1]) * float(cfg["ffn_keep_ratio"]))
    )
    if keep_channels < gate_states.shape[-1]:
        gate_states[:, visual_start:visual_end, keep_channels:] = 0
        up_states[:, visual_start:visual_end, keep_channels:] = 0
        if layer_idx is not None and int(layer_idx) >= 0:
            ffn_bits = int(runtime.get("yopo_ffn_mask_layer_bits", 0))
            ffn_bits |= 1 << int(layer_idx)
            runtime["yopo_ffn_mask_layer_bits"] = ffn_bits
            runtime["yopo_ffn_mask_layers"] = ffn_bits.bit_count()
    return gate_states, up_states


def yopo_mask_statistics(mask: torch.Tensor, *, start_layer: int = 0) -> Dict[str, Any]:
    active = mask.to(dtype=torch.int64).sum(dim=1)
    effective = active[int(start_layer) :]
    return {
        "active_heads_per_layer": [int(x) for x in active.tolist()],
        "active_head_total": int(effective.sum().item()),
        "active_head_mean": float(effective.float().mean().item())
        if effective.numel()
        else 0.0,
    }


def select_yopo_active_head_mask(
    mean_visual_ratios: torch.Tensor,
    *,
    start_layer: int,
    target_active_heads_per_layer: float,
) -> torch.BoolTensor:
    """Select a checkpoint-specific YOPO mask at a fixed operation budget.

    The released generator thresholds each layer/head's calibration ratio and
    asks users to adjust that threshold for the desired number of active heads.
    A global top-k is the deterministic equivalent when the requested mean
    active-head count is fixed. Layers before ``start_layer`` are marked active
    because the runtime does not apply head pruning there.
    """

    ratios = torch.as_tensor(mean_visual_ratios, dtype=torch.float64, device="cpu")
    if ratios.ndim != 2 or not torch.isfinite(ratios).all():
        raise ValueError("YOPO calibration ratios must be a finite [layers, heads] matrix")
    layer_count, head_count = (int(ratios.shape[0]), int(ratios.shape[1]))
    start_layer = int(start_layer)
    target = float(target_active_heads_per_layer)
    if not 0 <= start_layer < layer_count:
        raise ValueError("YOPO calibration start_layer is outside the decoder")
    if not 0.0 < target <= head_count:
        raise ValueError("YOPO target_active_heads_per_layer must be in (0, head_count]")

    eligible = ratios[start_layer:].reshape(-1)
    keep_count = int(round(target * (layer_count - start_layer)))
    keep_count = min(max(keep_count, 1), int(eligible.numel()))
    # Stable sorting makes equal-valued calibration entries deterministic.
    order = torch.argsort(eligible, descending=True, stable=True)
    mask = torch.zeros(layer_count, head_count, dtype=torch.bool)
    mask[:start_layer] = True
    selected = torch.zeros_like(eligible, dtype=torch.bool)
    selected[order[:keep_count]] = True
    mask[start_layer:] = selected.view(layer_count - start_layer, head_count)
    return mask


def _yopo_layer_prefill_flops(
    *,
    visual_tokens: int,
    nonvisual_tokens: int,
    hidden_size: int,
    intermediate_size: int,
    num_attention_heads: int,
    num_key_value_heads: int,
    active_query_heads: int,
    active_kv_groups: int,
    neighbor_window_size: Optional[int],
    ffn_keep_ratio: float,
    structural_text_only: bool,
    visual_query_head_mask: bool,
) -> float:
    """Estimate one Qwen2 prefill layer under YOPO's sparse operations.

    The convention matches :func:`dop.qwen2_decoder_flops`: one multiply-add
    is counted as two FLOPs, attention uses a dense square baseline, and the
    Qwen2 GQA K/V width is handled independently from query heads.  YOPO's
    public PyTorch release is a mask simulation, so this is the *theoretical
    sparse-operation* cost, not the FLOPs executed by dense SDPA kernels.
    """

    v = int(visual_tokens)
    t = int(nonvisual_tokens)
    h = int(hidden_size)
    m = int(intermediate_size)
    qh = int(num_attention_heads)
    kvh = int(num_key_value_heads)
    active_q = int(active_query_heads)
    active_kv = int(active_kv_groups)
    if min(v, t, h, m, qh, kvh, active_q, active_kv) < 0:
        raise ValueError("YOPO FLOP dimensions must be non-negative")
    if qh < 1 or kvh < 1 or h % qh or qh % kvh:
        raise ValueError("YOPO requires valid Qwen2 GQA geometry")
    if active_q > qh or active_kv > kvh:
        raise ValueError("YOPO active head counts exceed model geometry")
    if not 0.0 < float(ffn_keep_ratio) <= 1.0:
        raise ValueError("YOPO ffn_keep_ratio must be in (0, 1]")

    head_dim = h // qh
    if structural_text_only:
        n = t
        return float(
            4 * n * h * h
            + 4 * n * h * (kvh * head_dim)
            + 4 * n * n * h
            + 6 * n * h * m
        )

    n = v + t
    query_fraction = active_q / float(qh)
    kv_fraction = active_kv / float(kvh)

    # Text Q/O always remain dense.  The released LLaVA profile additionally
    # gates visual-query outputs per head; the Qwen2-VL train-free profile does
    # not, so its visual Q/O projection cost stays dense.
    visual_qo_fraction = query_fraction if visual_query_head_mask else 1.0
    qo = 4.0 * h * h * (t + v * visual_qo_fraction)
    # Visual K/V groups are needed only for GQA groups that contain at least
    # one active query head; text K/V remain dense.
    kv = 4.0 * h * (kvh * head_dim) * (t + v * kv_fraction)

    # Text-text interactions stay dense.  The released Qwen2-VL branch masks
    # only the *visual-key columns* after softmax: visual-query -> text-key
    # work therefore remains dense, while text-query -> visual-key and
    # visual-query -> visual-key work follow the active-head fraction.  The
    # released LLaVA branch additionally masks visual-query head outputs, so
    # every pair with a visual query or visual key can follow that fraction.
    # A local window replaces V^2 only for the LLaVA neighbour-attention
    # profile; boundary effects make w^2 an upper bound.
    tt_pairs = t * t
    text_to_visual_pairs = t * v
    visual_to_text_pairs = v * t
    if neighbor_window_size is None:
        vv_pairs = v * v
    else:
        window = int(neighbor_window_size)
        if window < 1 or window % 2 != 1:
            raise ValueError("YOPO neighbor window must be a positive odd integer")
        vv_pairs = v * min(v, window * window)
    if visual_query_head_mask:
        attention_pairs = tt_pairs + query_fraction * (
            text_to_visual_pairs + visual_to_text_pairs + vv_pairs
        )
    else:
        attention_pairs = (
            tt_pairs
            + visual_to_text_pairs
            + query_fraction * (text_to_visual_pairs + vv_pairs)
        )
    attention = 4.0 * h * attention_pairs
    mlp = 6.0 * h * m * (t + v * float(ffn_keep_ratio))
    return float(qo + kv + attention + mlp)


def yopo_qwen2_prefill_flops(
    *,
    visual_tokens: int,
    nonvisual_tokens: int,
    active_head_mask: torch.Tensor,
    head_prune_start_layer: int,
    layer_drop_start_layer: Optional[int],
    hidden_size: int,
    intermediate_size: int,
    num_attention_heads: int,
    num_key_value_heads: int,
    neighbor_window_size: Optional[int],
    ffn_keep_ratio: float,
    visual_query_head_mask: bool,
) -> float:
    """Return total decoder-prefill FLOPs for one YOPO sample shape."""

    mask = torch.as_tensor(active_head_mask, dtype=torch.bool, device="cpu")
    if mask.ndim != 2:
        raise ValueError("YOPO active_head_mask must be [layers, query_heads]")
    layer_count, query_heads = map(int, mask.shape)
    if query_heads != int(num_attention_heads):
        raise ValueError("YOPO mask query-head width does not match the model")
    start = int(head_prune_start_layer)
    drop = layer_count if layer_drop_start_layer is None else int(
        layer_drop_start_layer
    )
    if not 0 <= start <= layer_count or not 0 <= drop <= layer_count:
        raise ValueError("YOPO layer boundary is outside the decoder")
    if query_heads % int(num_key_value_heads):
        raise ValueError("YOPO query heads must be divisible by KV heads")
    group_size = query_heads // int(num_key_value_heads)

    total = 0.0
    for layer_idx in range(layer_count):
        row = (
            mask[layer_idx]
            if layer_idx >= start
            else torch.ones(query_heads, dtype=torch.bool)
        )
        active_query = int(row.sum().item())
        active_groups = int(
            row.reshape(int(num_key_value_heads), group_size)
            .any(dim=1)
            .sum()
            .item()
        )
        total += _yopo_layer_prefill_flops(
            visual_tokens=visual_tokens,
            nonvisual_tokens=nonvisual_tokens,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            num_attention_heads=num_attention_heads,
            num_key_value_heads=num_key_value_heads,
            active_query_heads=active_query,
            active_kv_groups=active_groups,
            neighbor_window_size=neighbor_window_size,
            ffn_keep_ratio=ffn_keep_ratio,
            structural_text_only=layer_idx >= drop,
            visual_query_head_mask=visual_query_head_mask,
        )
    return float(total)


def select_yopo_flops_policy(
    *,
    mean_visual_ratios: torch.Tensor,
    sample_shapes: Any,
    target_flops_ratio: float,
    head_prune_start_layer: int,
    hidden_size: int,
    intermediate_size: int,
    num_attention_heads: int,
    num_key_value_heads: int,
    neighbor_window_size: Optional[int] = None,
    ffn_keep_ratio: float = 1.0,
    visual_query_head_mask: bool = False,
    minimum_active_heads_per_layer: int = 1,
    ffn_keep_ratio_bounds: Optional[Tuple[float, float]] = None,
) -> Dict[str, Any]:
    """Search YOPO's mean active-head budget and deep boundary.

    Candidate masks always follow the checkpoint-calibrated YOPO ranking.
    The objective first minimizes absolute error to the requested ratio of
    summed method and baseline sample FLOPs, then prefers the released FFN
    setting, more active heads, and a later deep cut.  ``ffn_keep_ratio_bounds`` is intended for
    adapting the released LLaVA training profile to a requested FLOP budget:
    it preserves YOPO's visual-channel operation while solving its continuous
    strength inside the supplied interval.  The result is rounded to a real
    Qwen2 intermediate-channel count before FLOPs are evaluated.
    """

    ratios = torch.as_tensor(mean_visual_ratios, dtype=torch.float64, device="cpu")
    if ratios.ndim != 2:
        raise ValueError("YOPO mean ratios must be [layers, heads]")
    layer_count, head_count = map(int, ratios.shape)
    target = float(target_flops_ratio)
    if not 0.0 < target <= 1.0:
        raise ValueError("YOPO target_flops_ratio must be in (0, 1]")
    parsed_shapes = []
    for raw in sample_shapes:
        v = int(raw["visual_tokens"])
        t = int(raw["nonvisual_tokens"])
        if v < 1 or t < 1:
            raise ValueError("YOPO calibration shapes must be positive")
        parsed_shapes.append((v, t))
    if not parsed_shapes:
        raise ValueError("YOPO policy search requires sample shapes")

    ffn_bounds = None
    if ffn_keep_ratio_bounds is not None:
        if len(ffn_keep_ratio_bounds) != 2:
            raise ValueError("YOPO FFN keep-ratio bounds must contain two values")
        lower, upper = map(float, ffn_keep_ratio_bounds)
        if not 0.0 < lower <= upper <= 1.0:
            raise ValueError(
                "YOPO FFN keep-ratio bounds must satisfy 0 < lower <= upper <= 1"
            )
        ffn_bounds = (lower, upper)

    full_mask = torch.ones(layer_count, head_count, dtype=torch.bool)
    full_costs = [
        yopo_qwen2_prefill_flops(
            visual_tokens=v,
            nonvisual_tokens=t,
            active_head_mask=full_mask,
            head_prune_start_layer=layer_count,
            layer_drop_start_layer=layer_count,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            num_attention_heads=num_attention_heads,
            num_key_value_heads=num_key_value_heads,
            neighbor_window_size=None,
            ffn_keep_ratio=1.0,
            visual_query_head_mask=False,
        )
        for v, t in parsed_shapes
    ]
    best = None
    best_payload = None
    for active_heads in range(
        int(minimum_active_heads_per_layer), head_count + 1
    ):
        mask = select_yopo_active_head_mask(
            ratios,
            start_layer=int(head_prune_start_layer),
            target_active_heads_per_layer=float(active_heads),
        )
        for drop_start in range(layer_count + 1):
            def evaluate(candidate_ffn_ratio: float) -> Dict[str, Any]:
                selected_costs = [
                    float(
                        yopo_qwen2_prefill_flops(
                            visual_tokens=v,
                            nonvisual_tokens=t,
                            active_head_mask=mask,
                            head_prune_start_layer=head_prune_start_layer,
                            layer_drop_start_layer=drop_start,
                            hidden_size=hidden_size,
                            intermediate_size=intermediate_size,
                            num_attention_heads=num_attention_heads,
                            num_key_value_heads=num_key_value_heads,
                            neighbor_window_size=neighbor_window_size,
                            ffn_keep_ratio=candidate_ffn_ratio,
                            visual_query_head_mask=visual_query_head_mask,
                        )
                    )
                    for (v, t), full in zip(parsed_shapes, full_costs)
                ]
                return aggregate_samplewise_flops(
                    selected_costs, full_costs
                )

            candidate_ffn_ratios = [float(ffn_keep_ratio)]
            released_ffn_ratio = float(ffn_keep_ratio)
            if ffn_bounds is not None:
                lower, upper = ffn_bounds
                released_ffn_ratio = lower
                lower_aggregate = evaluate(lower)
                upper_aggregate = evaluate(upper)
                lower_mean = float(lower_aggregate["mean_flops_ratio"])
                upper_mean = float(upper_aggregate["mean_flops_ratio"])
                if upper_mean != lower_mean:
                    ideal = lower + (target - lower_mean) * (
                        (upper - lower) / (upper_mean - lower_mean)
                    )
                else:
                    ideal = lower
                ideal = min(max(ideal, lower), upper)
                keep_channels = min(
                    max(round(ideal * int(intermediate_size)), 1),
                    int(intermediate_size),
                )
                rounded = keep_channels / float(intermediate_size)
                candidate_ffn_ratios = sorted({lower, rounded, upper})

            for candidate_ffn_ratio in candidate_ffn_ratios:
                aggregate = evaluate(candidate_ffn_ratio)
                mean_ratio = float(aggregate["mean_flops_ratio"])
                error = abs(mean_ratio - target)
                objective = (
                    error,
                    abs(candidate_ffn_ratio - released_ffn_ratio),
                    -active_heads,
                    -drop_start,
                )
                if best is None or objective < best:
                    best = objective
                    best_payload = {
                        "active_head_mask": mask,
                        "target_active_heads_per_layer": float(active_heads),
                        "layer_drop_start_layer": int(drop_start),
                        "ffn_keep_ratio": float(candidate_ffn_ratio),
                        "ffn_keep_channels": int(
                            round(candidate_ffn_ratio * int(intermediate_size))
                        ),
                        **aggregate,
                        "absolute_target_error": float(error),
                    }
    assert best_payload is not None
    return best_payload
