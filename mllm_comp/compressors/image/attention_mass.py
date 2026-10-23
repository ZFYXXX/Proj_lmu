"""Utilities for restoring visual cluster mass in decoder attention."""

from __future__ import annotations

from typing import Any


def add_visual_log_mass_bias(
    attention_mask: Any | None,
    visual_log_mass: Any,
    *,
    key_length: int,
) -> Any:
    """Add cluster log-mass to key logits, leaving non-visual keys unchanged.

    ``visual_log_mass`` is aligned to the compressed prefill sequence. During
    autoregressive decoding, newly generated keys receive zero additional mass.
    """

    import torch

    if not isinstance(visual_log_mass, torch.Tensor) or visual_log_mass.ndim != 1:
        raise ValueError("visual_log_mass must be a rank-1 tensor")
    if key_length < int(visual_log_mass.numel()):
        raise ValueError("attention key length is shorter than the visual mass layout")
    if attention_mask is None or attention_mask.ndim != 4:
        raise ValueError("visual mass correction requires a rank-4 additive attention mask")
    if int(attention_mask.shape[-1]) != key_length:
        raise ValueError("attention mask key dimension does not match key_length")
    bias = torch.zeros(key_length, device=attention_mask.device, dtype=torch.float32)
    count = int(visual_log_mass.numel())
    bias[:count] = visual_log_mass.to(device=bias.device, dtype=bias.dtype)
    return attention_mask + bias.to(dtype=attention_mask.dtype).view(1, 1, 1, -1)


__all__ = ["add_visual_log_mass_bias"]
