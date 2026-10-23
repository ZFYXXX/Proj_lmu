"""Exact eager attention-weight capture at the semantic Attention boundary."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from mllm_comp.analysis.bindings import DecoderLayerBinding
from mllm_comp.analysis.hooks import HookSession
from mllm_comp.core.errors import BackendIntegrationError


class AttentionWeightCapture:
    """Capture per-layer weights even when a provider drops top-level attentions."""

    def __init__(self, layers: Sequence[DecoderLayerBinding]) -> None:
        self.layers = tuple(layers)
        self.values: dict[int, Any] = {}
        self._errors: list[str] = []
        self._installed = False

    def install(self, hooks: HookSession) -> None:
        if self._installed:
            raise BackendIntegrationError(
                "Attention-weight capture hooks were already installed."
            )
        self._installed = True
        for layer in self.layers:
            hooks.register_forward_hook(layer.attention, self._hook(layer.index))

    def finish(self) -> dict[int, Any]:
        if self._errors:
            raise BackendIntegrationError(
                "Attention-weight capture failed: " + "; ".join(self._errors[:5])
            )
        expected = {layer.index for layer in self.layers}
        missing = sorted(expected - set(self.values))
        if missing:
            raise BackendIntegrationError(
                f"Attention modules returned no eager weights for layers {missing}."
            )
        return dict(self.values)

    def clear(self) -> None:
        self.values.clear()
        self._errors.clear()

    def _hook(self, layer: int):
        def hook(_module: Any, _inputs: Any, output: Any) -> Any:
            try:
                if layer in self.values:
                    raise BackendIntegrationError(
                        f"Attention layer {layer} fired more than once in one prefill."
                    )
                weights = attention_weight_tensor(output)
                self.values[layer] = weights.detach()
            except Exception as exc:
                self._errors.append(f"L{layer}: {type(exc).__name__}: {exc}")
            return output

        return hook


def attention_weight_tensor(output: Any) -> Any:
    """Extract one full ``[batch, head, query, key]`` attention tensor."""

    candidate = None
    if isinstance(output, Mapping):
        for key in ("attn_weights", "attention_weights", "attentions"):
            if key in output:
                candidate = output[key]
                break
    elif isinstance(output, (tuple, list)) and len(output) >= 2:
        candidate = output[1]
    else:
        for name in ("attn_weights", "attention_weights", "attentions"):
            candidate = getattr(output, name, None)
            if candidate is not None:
                break
    if candidate is None or not hasattr(candidate, "shape"):
        raise BackendIntegrationError(
            "Attention module returned no explicit weight tensor; use eager attention."
        )
    if len(candidate.shape) != 4:
        raise BackendIntegrationError(
            "Attention weights must have [batch, head, query, key] shape, got "
            f"{tuple(candidate.shape)}."
        )
    return candidate


__all__ = ["AttentionWeightCapture", "attention_weight_tensor"]
