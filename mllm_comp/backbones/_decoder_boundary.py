"""Run pinned decoder attention, compression and FFN in one checkpoint scope."""

from mllm_comp.backbones._provider_patching import MethodPatch


def decoder_post_attention_forward(layer, hidden_states, *, apply_compression, tuple_output=False, **kwargs):
    """Run the pinned native Attention → compression → FFN value path.

    The caller owns checkpointing and the compression decision or replay.
    Attention and FFN remain native modules, including their device hooks.
    """
    update, weights = layer.self_attn(
        hidden_states=layer.input_layernorm(hidden_states), **kwargs,
    )
    residual = apply_compression(hidden_states + update)
    hidden = residual + layer.mlp(layer.post_attention_layernorm(residual))
    if tuple_output:
        return (hidden, weights) if kwargs.get("output_attentions", False) else (hidden,)
    return hidden


def decoder_post_attention_boundary(layer, hidden_states, *, apply_compression, tuple_output=False, **kwargs):
    """Insert compression between pinned Qwen residual and FFN operations.

    ``apply_compression`` returns the compressed hidden states and a value-only
    replay callable. Native layer/module hooks still own attention and devices.
    Each forward owns its own replay closure, including overlapping graphs.
    """
    import torch

    checkpointed = bool(getattr(layer, "gradient_checkpointing", False) and layer.training)
    replay = None

    def compress(residual):
        nonlocal replay, apply_compression
        if replay is None:
            with torch.no_grad():
                _, replay = apply_compression(residual)
            # Do not retain the provider request or its scoring activations.
            apply_compression = None
        # Forward and recomputation save the same differentiable value graph.
        return replay(residual)

    def boundary_forward(layer_self, value, **layer_kwargs):
        return decoder_post_attention_forward(
            layer_self, value, apply_compression=compress,
            tuple_output=tuple_output, **layer_kwargs,
        )

    def run(value):
        original_checkpointing = getattr(layer, "gradient_checkpointing", False)
        # Keep Accelerate's device hooks around the full layer implementation.
        method = "_old_forward" if hasattr(layer, "_hf_hook") else "forward"
        patch = MethodPatch.apply(layer, method, boundary_forward, getattr(layer, method), label="decoder boundary")
        layer.gradient_checkpointing = False
        try:
            return layer(value, **kwargs)
        finally:
            layer.gradient_checkpointing = original_checkpointing
            patch.restore()

    return layer._gradient_checkpointing_func(run, hidden_states) if checkpointed else run(hidden_states)
