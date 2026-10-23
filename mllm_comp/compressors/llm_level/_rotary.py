"""Shared RoPE math for audio-token attention scorers.

This only rotates Q/K. Each strategy owns its scoring policy and gradient
context, including whether position embeddings are required.
"""

from __future__ import annotations

from typing import Any, Mapping

from mllm_comp.core.errors import InvariantError


def apply_rotary_position_embeddings(
    query: Any,
    key: Any,
    *,
    position_embeddings: Any,
    attention_module: Any,
    batch_size: int,
    sequence_length: int,
) -> tuple[Any, Any]:
    """Apply ordinary RoPE or Qwen's native ``[3, B, S, D]`` M-RoPE."""
    try:
        cos, sin = position_embeddings
    except (TypeError, ValueError) as exc:
        raise InvariantError("Expected a (cos, sin) position-embedding tuple.") from exc
    if (
        getattr(cos, "ndim", 0) == 4
        and int(cos.shape[0]) == 3
        and int(cos.shape[1]) == batch_size
        and int(cos.shape[2]) == sequence_length
    ):
        rope_scaling = getattr(attention_module, "rope_scaling", None)
        if not isinstance(rope_scaling, Mapping):
            rope_scaling = getattr(attention_module.config, "rope_scaling", None)
        sections = (
            rope_scaling.get("mrope_section")
            if isinstance(rope_scaling, Mapping)
            else None
        )
        if not isinstance(sections, (list, tuple)) or not sections:
            raise InvariantError("Cannot apply Qwen M-RoPE without mrope_section.")
        doubled_sections = [int(item) for item in sections] * 2
        if any(item <= 0 for item in doubled_sections) or sum(
            doubled_sections
        ) != int(cos.shape[-1]):
            raise InvariantError("Invalid Qwen mrope_section layout.")
        cos = _select_mrope_axes(cos, doubled_sections)
        sin = _select_mrope_axes(sin, doubled_sections)
    else:
        try:
            cos = _normalize_rope(cos, batch_size, sequence_length)
            sin = _normalize_rope(sin, batch_size, sequence_length)
        except AttributeError as exc:
            raise InvariantError("Unsupported rotary-embedding layout.") from exc
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)
    return (
        query * cos + _rotate_half(query) * sin,
        key * cos + _rotate_half(key) * sin,
    )


def _normalize_rope(value: Any, batch_size: int, sequence_length: int) -> Any:
    if value.ndim == 2 and int(value.shape[0]) == sequence_length:
        return value.unsqueeze(0).expand(batch_size, -1, -1)
    if value.ndim == 3 and int(value.shape[0]) in {1, batch_size}:
        return value.expand(batch_size, -1, -1)
    if value.ndim == 4 and int(value.shape[1]) == 1:
        return value[:, 0]
    raise InvariantError("Unsupported rotary-embedding layout.")


def _select_mrope_axes(value: Any, sections: list[int]) -> Any:
    import torch

    return torch.cat(
        [chunk[index % 3] for index, chunk in enumerate(value.split(sections, dim=-1))],
        dim=-1,
    )


def _rotate_half(value: Any) -> Any:
    import torch

    first, second = value.chunk(2, dim=-1)
    return torch.cat((-second, first), dim=-1)
