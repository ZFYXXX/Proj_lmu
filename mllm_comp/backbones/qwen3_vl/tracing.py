"""Opt-in hooks for Qwen3-VL text-generation internals."""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Iterator

from mllm_comp.backbones.base import TraceArtifacts, TraceRequest


def _output_field(output: Any, name: str) -> Any | None:
    if output is None:
        return None
    if isinstance(output, dict):
        return output.get(name)
    return getattr(output, name, None)


@contextmanager
def capture_qwen3_vl_traces(
    model: Any,
    request: TraceRequest | None,
) -> Iterator[TraceArtifacts | None]:
    """Capture the requested last forward step without retaining by default."""

    if request is None or not request.enabled():
        yield None
        return

    artifacts = TraceArtifacts(
        metadata={
            "capture_semantics": "last_forward_step",
            "provider": "transformers-qwen3-vl-forward-hook",
        }
    )
    handles: list[Any] = []

    if hasattr(model, "register_forward_hook"):

        def model_hook(_module: Any, _args: Any, _kwargs: Any, output: Any) -> None:
            if request.text_logits:
                artifacts.text_logits = _output_field(output, "logits")
            if request.text_hidden_states:
                artifacts.text_hidden_states = _output_field(output, "hidden_states")
            if request.text_attentions:
                artifacts.text_attentions = _output_field(output, "attentions")
            if request.text_kv_cache:
                artifacts.text_kv_cache = _output_field(output, "past_key_values")

        handles.append(model.register_forward_hook(model_hook, with_kwargs=True))

    try:
        yield artifacts
    finally:
        for handle in reversed(handles):
            handle.remove()
