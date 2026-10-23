"""Bounded-memory capture of Qwen3.5's final vision-layer attention statistics."""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Iterator


def _mean_received_attention(query: Any, key: Any, *, scaling: float, chunk_size: int) -> Any:
    """Compute per-key mean attention without materializing an N x N map."""
    import torch

    # The native Qwen vision attention uses [tokens, heads, head_dim].
    q = query.transpose(0, 1).float()
    k = key.transpose(0, 1).float()
    received = torch.zeros(k.shape[1], device=k.device, dtype=torch.float32)
    for start in range(0, q.shape[1], chunk_size):
        stop = min(q.shape[1], start + chunk_size)
        scores = torch.matmul(q[:, start:stop], k.transpose(-1, -2)) * float(scaling)
        probabilities = torch.softmax(scores, dim=-1)
        received += probabilities.sum(dim=1).sum(dim=0)
    return received / (q.shape[0] * q.shape[1])


@contextmanager
def capture_qwen3_5_final_vision_attention(
    visual: Any,
    *,
    query_chunk_size: int = 128,
) -> Iterator[list[dict[str, Any]]]:
    """Capture mean-received scores and rotary keys for each packed image.

    Qwen3.5's visual tower has no CLS token. We therefore use the mean attention
    received by each patch, matching VisionZip's non-CLS/SigLIP-style branch.
    Attention is reconstructed from the final block's native Q/K projections in
    query chunks, so memory use is O(chunk_size * tokens), not O(tokens^2).
    """
    import torch

    blocks = getattr(visual, "blocks", None)
    if not blocks:
        raise RuntimeError("Qwen3.5 vision tower exposes no transformer blocks.")
    attention_module = getattr(blocks[-1], "attn", None)
    if attention_module is None or not callable(getattr(attention_module, "qkv", None)):
        raise RuntimeError("Qwen3.5 final vision block does not expose native QKV projections.")
    if query_chunk_size < 1:
        raise ValueError("query_chunk_size must be positive.")

    from transformers.models.qwen3_5.modeling_qwen3_5 import apply_rotary_pos_emb_vision

    captured: list[dict[str, Any]] = []

    def pre_hook(module: Any, args: tuple[Any, ...], kwargs: dict[str, Any]) -> None:
        hidden = args[0] if args else kwargs.get("hidden_states")
        cu_seqlens = kwargs.get("cu_seqlens", args[1] if len(args) > 1 else None)
        position_embeddings = kwargs.get(
            "position_embeddings", args[2] if len(args) > 2 else None
        )
        if hidden is None or cu_seqlens is None or position_embeddings is None:
            raise RuntimeError("Qwen3.5 final vision attention inputs are incomplete.")
        if hidden.ndim != 2:
            raise RuntimeError(f"Expected packed vision states [tokens, hidden], got {tuple(hidden.shape)}.")

        token_count = int(hidden.shape[0])
        qkv = module.qkv(hidden).reshape(
            token_count, 3, module.num_heads, -1
        ).permute(1, 0, 2, 3)
        query, key = qkv[0], qkv[1]
        cos, sin = position_embeddings
        query, key = apply_rotary_pos_emb_vision(query, key, cos, sin)

        boundaries = [int(item) for item in cu_seqlens.detach().cpu().flatten().tolist()]
        if len(boundaries) < 2 or boundaries[0] != 0 or boundaries[-1] != token_count:
            raise RuntimeError("Qwen3.5 cu_seqlens do not partition the packed vision tokens.")
        scaling = float(getattr(module, "scaling", module.head_dim**-0.5))
        for start, stop in zip(boundaries[:-1], boundaries[1:]):
            if stop <= start:
                raise RuntimeError("Qwen3.5 vision sequence contains an empty image segment.")
            q_segment = query[start:stop]
            k_segment = key[start:stop]
            scores = _mean_received_attention(
                q_segment,
                k_segment,
                scaling=scaling,
                chunk_size=query_chunk_size,
            )
            captured.append(
                {
                    "mean_received": scores,
                    "keys": k_segment,
                    "patch_tokens": stop - start,
                    "attention_mode": "mean_received",
                    "key_position_encoding": "qwen3_5_rotary",
                }
            )

    handle = attention_module.register_forward_pre_hook(pre_hook, with_kwargs=True)
    try:
        yield captured
    finally:
        handle.remove()


@contextmanager
def capture_qwen3_5_premerger_features(visual: Any) -> Iterator[list[Any]]:
    """Capture final-layer patch features immediately before spatial merging."""

    import torch

    merger = getattr(visual, "merger", None)
    if merger is None or not callable(getattr(merger, "register_forward_pre_hook", None)):
        raise RuntimeError("Qwen3.8 visual tower does not expose its patch merger.")
    captured: list[Any] = []

    def pre_hook(_module: Any, args: tuple[Any, ...]) -> None:
        hidden = args[0] if args else None
        if not torch.is_tensor(hidden) or hidden.ndim != 2:
            raise RuntimeError("Expected pre-merger patch features shaped [tokens, hidden].")
        captured.append(hidden)

    handle = merger.register_forward_pre_hook(pre_hook)
    try:
        yield captured
    finally:
        handle.remove()


@contextmanager
def capture_qwen3_5_semantic_hidden_range(
    visual: Any, *, start_layer: int, end_layer: int
) -> Iterator[list[Any]]:
    """Capture native vision-block outputs in a zero-based half-open range."""
    blocks = getattr(visual, "blocks", None)
    if not blocks:
        raise RuntimeError("Qwen3.8 visual tower exposes no transformer blocks.")
    if start_layer < 0 or end_layer <= start_layer or end_layer > len(blocks):
        raise ValueError(
            f"Invalid Qwen3.8 semantic block range [{start_layer}, {end_layer}) "
            f"for vision depth {len(blocks)}."
        )
    captured: list[Any] = []
    handles = []

    def hook(_module: Any, _args: tuple[Any, ...], output: Any) -> None:
        import torch

        hidden = output[0] if isinstance(output, tuple) else output
        if not torch.is_tensor(hidden) or hidden.ndim != 2:
            raise RuntimeError(
                "Qwen3.8 semantic block output must be [patch_tokens, hidden]."
            )
        captured.append(hidden.detach())

    try:
        for index in range(start_layer, end_layer):
            handles.append(blocks[index].register_forward_hook(hook))
        yield captured
    finally:
        for handle in reversed(handles):
            handle.remove()


def pool_patch_statistics_to_merger_tokens(
    statistics: dict[str, Any],
    *,
    merged_token_count: int,
    spatial_merge_size: int,
) -> tuple[Any, Any]:
    """Map contiguous pre-merger patch statistics to Qwen3.5 merger tokens."""
    import torch

    if merged_token_count < 1 or spatial_merge_size < 1:
        raise ValueError("merged_token_count and spatial_merge_size must be positive.")
    patches_per_token = spatial_merge_size**2
    expected_patches = merged_token_count * patches_per_token
    attention = statistics["mean_received"]
    keys = statistics["keys"]
    if int(attention.shape[0]) != expected_patches or int(keys.shape[0]) != expected_patches:
        raise RuntimeError(
            "Qwen3.5 patch-to-merger mapping mismatch: expected "
            f"{expected_patches} patches for {merged_token_count} merger tokens, "
            f"got attention={int(attention.shape[0])}, keys={int(keys.shape[0])}."
        )
    pooled_attention = attention.reshape(merged_token_count, patches_per_token).mean(dim=1)
    pooled_keys = (
        keys.reshape(merged_token_count, patches_per_token, *keys.shape[1:])
        .mean(dim=1)
        .permute(1, 0, 2)
        .contiguous()
    )
    if not bool(torch.isfinite(pooled_attention).all()) or not bool(torch.isfinite(pooled_keys).all()):
        raise RuntimeError("Pooled Qwen3.5 VisionZip statistics contain non-finite values.")
    return pooled_attention, pooled_keys


def pool_premerger_features_to_merger_tokens(
    packed_features: Any,
    image_grid_thw: Any,
    *,
    spatial_merge_size: int,
) -> tuple[Any, ...]:
    """Average contiguous native patch groups into one visual vector per merger token."""

    import torch

    if not torch.is_tensor(packed_features) or packed_features.ndim != 2:
        raise ValueError("packed_features must have shape [patch_tokens, hidden].")
    if not torch.is_tensor(image_grid_thw) or image_grid_thw.ndim != 2:
        raise ValueError("image_grid_thw must have shape [images, 3].")
    if spatial_merge_size < 1:
        raise ValueError("spatial_merge_size must be positive.")
    patches_per_token = spatial_merge_size**2
    groups = []
    cursor = 0
    for grid in image_grid_thw:
        patch_count = int(grid.to(torch.long).prod().item())
        if patch_count < 1 or patch_count % patches_per_token:
            raise RuntimeError(
                f"Image patch count {patch_count} is not divisible by merger area {patches_per_token}."
            )
        token_count = patch_count // patches_per_token
        stop = cursor + patch_count
        if stop > int(packed_features.shape[0]):
            raise RuntimeError("image_grid_thw exceeds captured pre-merger features.")
        image_features = packed_features[cursor:stop]
        groups.append(
            image_features.reshape(token_count, patches_per_token, -1).mean(dim=1)
        )
        cursor = stop
    if cursor != int(packed_features.shape[0]):
        raise RuntimeError(
            f"Captured {int(packed_features.shape[0])} patch features, grid accounts for {cursor}."
        )
    if not all(bool(torch.isfinite(group).all()) for group in groups):
        raise RuntimeError("Pooled pre-merger visual features contain non-finite values.")
    return tuple(groups)


def build_qwen3_5_merger_inputs(
    packed_features: Any,
    image_grid_thw: Any,
    *,
    spatial_merge_size: int,
    merger: Any,
) -> tuple[Any, ...]:
    """Recreate the exact normalized input consumed by Qwen3.5's patch merger."""
    import torch

    if not torch.is_tensor(packed_features) or packed_features.ndim != 2:
        raise ValueError("packed_features must have shape [patch_tokens, hidden].")
    if spatial_merge_size < 1:
        raise ValueError("spatial_merge_size must be positive.")
    width = int(getattr(merger, "hidden_size", 0))
    if width < 1:
        raise ValueError("Qwen3.5 patch merger has no valid hidden_size.")
    groups = []
    cursor = 0
    for grid in image_grid_thw:
        patch_count = int(grid.to(torch.long).prod().item())
        output_count = patch_count // (spatial_merge_size**2)
        stop = cursor + patch_count
        patches = packed_features[cursor:stop]
        if patches.shape[0] != patch_count or patch_count % (spatial_merge_size**2):
            raise RuntimeError("Qwen3.5 patch merger input does not align with image_grid_thw.")
        if bool(getattr(merger, "use_postshuffle_norm", False)):
            merger_input = merger.norm(patches.reshape(output_count, width))
        else:
            merger_input = merger.norm(patches).reshape(output_count, width)
        groups.append(merger_input)
        cursor = stop
    if cursor != int(packed_features.shape[0]):
        raise RuntimeError("Captured patch features do not match image_grid_thw.")
    return tuple(groups)
