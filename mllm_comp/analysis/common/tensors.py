"""Small lazy-torch helpers for extracting and replacing model outputs."""

from __future__ import annotations

from typing import Any, Mapping

from mllm_comp.core.errors import BackendIntegrationError


def hidden_tensor(value: Any) -> Any:
    if hasattr(value, "shape") and hasattr(value, "detach"):
        return value
    if isinstance(value, Mapping):
        for key in ("last_hidden_state", "hidden_states"):
            if key in value:
                return hidden_tensor(value[key])
    if isinstance(value, (tuple, list)) and value:
        return hidden_tensor(value[0])
    raise BackendIntegrationError(
        "Analysis hook could not extract a hidden-state tensor from "
        f"{type(value).__name__}."
    )


def replace_hidden_tensor(value: Any, replacement: Any) -> Any:
    """Preserve the decoder/sublayer output container while replacing item zero."""

    if hasattr(value, "shape") and hasattr(value, "detach"):
        return replacement
    if isinstance(value, tuple) and value:
        return (replacement, *value[1:])
    if isinstance(value, list) and value:
        return [replacement, *value[1:]]
    raise BackendIntegrationError(
        "Analysis intervention cannot replace hidden states in "
        f"{type(value).__name__}."
    )


def logits_tensor(output: Any) -> Any:
    value = getattr(output, "logits", None)
    if value is None and isinstance(output, Mapping):
        value = output.get("logits")
    if value is None and isinstance(output, (tuple, list)) and output:
        value = output[0]
    if not hasattr(value, "shape") or not hasattr(value, "detach"):
        raise BackendIntegrationError(
            f"Analysis forward returned no logits tensor in {type(output).__name__}."
        )
    if len(value.shape) != 3:
        raise BackendIntegrationError(
            "Analysis logits must have [batch, sequence, vocabulary] shape, got "
            f"{tuple(value.shape)}."
        )
    return value


__all__ = ["hidden_tensor", "logits_tensor", "replace_hidden_tensor"]
