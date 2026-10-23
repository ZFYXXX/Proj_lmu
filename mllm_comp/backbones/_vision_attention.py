"""Pinned Qwen vision attention execution and video scoring signals."""

import inspect
from collections.abc import Mapping, Sequence
from typing import Any

from mllm_comp.core.errors import BackendIntegrationError, InvariantError


def execute_vision_attention(
    attention: Any,
    hidden_states: Any,
    *,
    counts: Sequence[int],
    position_embeddings: tuple[Any, Any],
    token_sizes: Sequence[Any | None],
    capture_mean_received: bool,
) -> tuple[Any, Any, Any, tuple[Any, ...] | None]:
    """Execute one pinned 4.57.3 vision attention call in bounded memory.

    This is the semantic equivalent of the Qwen2.5/Qwen3 native vision attention for
    the supported non-causal FORWARD envelope.  Splitting at this level is
    necessary for two reasons that output hooks cannot satisfy:

    * proportional ToMe attention adds ``log(token_size)`` to each key logit;
    * VisionZip must consume the final block's actual rotary keys/attention,
      not a second QKV projection of the post-MLP block output.

    Attention is evaluated independently for the same image windows encoded
    by provider ``cu_seqlens``.  Query chunking bounds the temporary score
    tensor while retaining exact per-window softmax semantics.
    """

    import torch
    if len(counts) != len(token_sizes):
        raise InvariantError(
            "Qwen vision attention counts and ToMe token-size streams differ."
        )
    total = sum(int(count) for count in counts)
    if total != int(hidden_states.shape[0]):
        raise InvariantError(
            "Qwen vision attention window counts do not match hidden states."
        )
    qkv = attention.qkv(hidden_states).reshape(
        total,
        3,
        int(attention.num_heads),
        -1,
    )
    query, key, value = qkv.unbind(dim=1)
    query, key = apply_vision_rotary(
        query,
        key,
        position_embeddings,
    )
    query_chunks = torch.split(query, tuple(counts))
    key_chunks = torch.split(key, tuple(counts))
    value_chunks = torch.split(value, tuple(counts))
    outputs: list[Any] = []
    received: list[Any] = []
    for query_chunk, key_chunk, value_chunk, sizes in zip(
        query_chunks,
        key_chunks,
        value_chunks,
        token_sizes,
    ):
        output, importance = streaming_vision_attention(
            query_chunk,
            key_chunk,
            value_chunk,
            scaling=float(attention.scaling),
            token_sizes=sizes,
            capture_mean_received=capture_mean_received,
        )
        outputs.append(output)
        if importance is not None:
            received.append(importance)
    projected = attention.proj(torch.cat(outputs, dim=0).reshape(total, -1))
    return (
        projected,
        query,
        key,
        tuple(received) if capture_mean_received else None,
    )


def apply_vision_rotary(
    query: Any,
    key: Any,
    position_embeddings: tuple[Any, Any],
) -> tuple[Any, Any]:
    """Source-pinned equivalent of ``apply_rotary_pos_emb_vision``."""

    query_dtype = query.dtype
    key_dtype = key.dtype
    cos, sin = position_embeddings
    query_float = query.float()
    key_float = key.float()
    cos = cos.unsqueeze(-2).float()
    sin = sin.unsqueeze(-2).float()
    query = query_float * cos + _rotate_half(query_float) * sin
    key = key_float * cos + _rotate_half(key_float) * sin
    return query.to(query_dtype), key.to(key_dtype)


def streaming_vision_attention(
    query: Any,
    key: Any,
    value: Any,
    *,
    scaling: float,
    token_sizes: Any | None,
    capture_mean_received: bool,
    query_chunk_size: int = 128,
) -> tuple[Any, Any | None]:
    """Exact non-causal attention with optional low-memory received mass.

    Inputs use provider layout ``[tokens, heads, head_dim]``.  The largest
    score allocation is ``heads x query_chunk_size x tokens``.  The optional
    result is the attention received by each key, averaged over heads and all
    queries; this is the only attention statistic VisionZip needs for a
    Qwen/SigLIP-like tower.
    """

    import torch
    if isinstance(query_chunk_size, bool) or query_chunk_size < 1:
        raise ValueError("query_chunk_size must be a positive integer.")
    if query.shape != key.shape or query.shape != value.shape:
        raise InvariantError("Qwen vision vision query/key/value shapes must match.")
    token_count = int(query.shape[0])
    head_count = int(query.shape[1])
    if token_count < 1 or head_count < 1:
        raise InvariantError("Qwen vision vision attention cannot be empty.")
    log_size: Any | None = None
    if token_sizes is not None:
        if int(token_sizes.shape[0]) != token_count:
            raise InvariantError(
                "Qwen vision ToMe token sizes do not match attention keys."
            )
        flat_sizes = token_sizes.reshape(token_count).float()
        if not bool((flat_sizes > 0).all().item()) or not bool(
            torch.isfinite(flat_sizes).all().item()
        ):
            raise InvariantError(
                "Qwen vision ToMe token sizes must be finite and positive."
            )
        # ToMe proportional attention: every subsequent layer adds log(s) to
        # the logits for a merged key.  A singleton query/head shape keeps the
        # bias broadcast-only rather than allocating a dense mask.
        log_size = flat_sizes.log().reshape(1, 1, token_count)

    query_heads = query.permute(1, 0, 2)
    key_heads = key.permute(1, 0, 2)
    value_heads = value.permute(1, 0, 2)
    output_chunks: list[Any] = []
    received_sum = (
        torch.zeros(token_count, dtype=torch.float32, device=query.device)
        if capture_mean_received
        else None
    )
    for start in range(0, token_count, int(query_chunk_size)):
        stop = min(start + int(query_chunk_size), token_count)
        logits = torch.matmul(
            query_heads[:, start:stop].float(),
            key_heads.float().transpose(-2, -1),
        ) * float(scaling)
        if log_size is not None:
            logits = logits + log_size
        probabilities = torch.softmax(logits, dim=-1, dtype=torch.float32)
        output_chunks.append(
            torch.matmul(probabilities.to(value_heads.dtype), value_heads)
        )
        if received_sum is not None:
            received_sum = received_sum + probabilities.sum(dim=(0, 1))
    output = torch.cat(output_chunks, dim=1).permute(1, 0, 2).contiguous()
    if received_sum is None:
        return output, None
    return output, received_sum / float(head_count * token_count)


def _rotate_half(value: Any) -> Any:
    import torch
    half = int(value.shape[-1]) // 2
    return torch.cat((-value[..., half:], value[..., :half]), dim=-1)


def bind_qwen_vision_block_call(
    signature: inspect.Signature,
    args: Sequence[Any],
    kwargs: Mapping[str, Any],
) -> tuple[Any, Any, tuple[Any, Any]]:
    """Read the native visual block inputs used by video signal providers."""
    bound = signature.bind_partial(*args, **kwargs)
    values = dict(bound.arguments)
    extras = values.pop("kwargs", None)
    if isinstance(extras, Mapping):
        values.update(extras)
    hidden = values.get("hidden_states", args[0] if args else None)
    cu_seqlens = values.get("cu_seqlens")
    position_embeddings = values.get("position_embeddings")
    if (
        hidden is None
        or cu_seqlens is None
        or not isinstance(position_embeddings, tuple)
        or len(position_embeddings) != 2
    ):
        raise BackendIntegrationError(
            "Video compression requires native Qwen block hidden/cu_seqlens/rotary inputs"
        )
    return hidden, cu_seqlens, position_embeddings


def qwen_vision_mean_received(
    attention: Any,
    hidden_states: Any,
    *,
    cu_seqlens: Any,
    position_embeddings: tuple[Any, Any],
) -> Any:
    """Reconstruct mean received attention within each native video window."""
    import torch
    total = int(hidden_states.shape[0])
    heads = int(getattr(attention, "num_heads", 0))
    scaling = getattr(attention, "scaling", None)
    qkv = getattr(attention, "qkv", None)
    if heads < 1 or not isinstance(scaling, (int, float)) or not callable(qkv):
        raise BackendIntegrationError(
            "Video compression requires native Qwen qkv/num_heads/scaling"
        )
    projected = qkv(hidden_states).reshape(total, 3, heads, -1)
    query, key, _value = projected.unbind(dim=1)
    # Video scoring retains FP32 rotary Q/K even when the native model is half precision.
    query, key = apply_vision_rotary(query.float(), key.float(), position_embeddings)
    boundaries = tuple(int(value) for value in cu_seqlens.detach().cpu().tolist())
    if (
        len(boundaries) < 2
        or boundaries[0] != 0
        or boundaries[-1] != total
        or any(left >= right for left, right in zip(boundaries, boundaries[1:]))
    ):
        raise BackendIntegrationError(
            "Video compression received invalid native Qwen attention boundaries"
        )
    received = []
    for start, stop in zip(boundaries, boundaries[1:]):
        query_chunk = query[start:stop]
        key_chunk = key[start:stop]
        accumulator = torch.zeros(
            stop - start,
            dtype=torch.float32,
            device=key.device,
        )
        for offset in range(0, stop - start, 128):
            current = query_chunk[offset : offset + 128]
            scores = torch.einsum("qhd,khd->hqk", current, key_chunk)
            probabilities = torch.softmax(
                scores * float(scaling),
                dim=-1,
                dtype=torch.float32,
            )
            accumulator = accumulator + probabilities.sum(dim=(0, 1))
        received.append(accumulator / float(heads * (stop - start)))
    return torch.cat(received, dim=0)
