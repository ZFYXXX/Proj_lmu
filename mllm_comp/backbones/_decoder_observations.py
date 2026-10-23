"""Temporary observations of native decoder module inputs and updates."""

from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any


@dataclass
class DecoderModuleTransitions:
    """Four detached outputs, complete when the native layer scope exits."""

    attention_before: Any = None
    attention_after: Any = None
    ffn_before: Any = None
    ffn_after: Any = None


@contextmanager
def capture_decoder_module_transitions(decoder_layer):
    """Observe normalized inputs and attention/FFN updates on the first forward.

    The caller keeps this scope outside checkpoint replay. Hooks do not replace
    native forwards or Accelerate device hooks and are removed on every exit.
    """
    sites = (
        ("input_layernorm", "attention_before"),
        ("self_attn", "attention_after"),
        ("post_attention_layernorm", "ffn_before"),
        ("mlp", "ffn_after"),
    )
    missing = tuple(name for name, _ in sites if not hasattr(decoder_layer, name))
    if missing:
        raise TypeError(f"Decoder transition capture is missing modules {missing!r}.")
    observation = DecoderModuleTransitions()

    def save(name):
        def hook(_module, _inputs, output):
            tensor = output[0] if isinstance(output, (tuple, list)) else output
            setattr(observation, name, tensor.detach())
        return hook

    handles = []
    try:
        for module_name, field_name in sites:
            handles.append(getattr(decoder_layer, module_name).register_forward_hook(save(field_name)))
        yield observation
        if any(getattr(observation, name) is None for _, name in sites):
            raise RuntimeError("Native decoder module observations are incomplete.")
    finally:
        for handle in handles:
            handle.remove()
