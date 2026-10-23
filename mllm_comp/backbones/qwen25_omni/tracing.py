"""Opt-in hooks for Qwen2.5-Omni generation internals."""

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
def capture_qwen25_omni_traces(
    model: Any,
    request: TraceRequest | None,
) -> Iterator[TraceArtifacts | None]:
    """Capture selected last-step forward artifacts and cross-stage tensors.

    The validated Transformers composite return does not expose Talker codec tokens or
    ``thinker_reply_part`` from the composite ``generate`` return value. The
    former is captured at Token2Wav's input and the latter at Talker's input.
    Ordinary generation installs no hooks and retains no model intermediates.
    """

    if request is None or not request.enabled():
        yield None
        return

    artifacts = TraceArtifacts(
        metadata={
            "capture_semantics": "last_forward_step_except_cross_stage_inputs",
            "provider": "transformers-qwen2.5-omni-hooks",
        }
    )
    handles: list[Any] = []

    thinker = getattr(model, "thinker", model)
    talker = getattr(model, "talker", None)
    token2wav = getattr(model, "token2wav", None)

    if thinker is not None and hasattr(thinker, "register_forward_hook"):
        def thinker_hook(_module: Any, _args: Any, _kwargs: Any, output: Any) -> None:
            if request.thinker_logits:
                artifacts.thinker_logits = _output_field(output, "logits")
            if request.thinker_hidden_states:
                artifacts.thinker_hidden_states = _output_field(output, "hidden_states")
            if request.thinker_attentions:
                artifacts.thinker_attentions = _output_field(output, "attentions")
            if request.thinker_kv_cache:
                artifacts.thinker_kv_cache = _output_field(output, "past_key_values")

        handles.append(thinker.register_forward_hook(thinker_hook, with_kwargs=True))

    if talker is not None and hasattr(talker, "register_forward_pre_hook"):
        def talker_pre_hook(_module: Any, _args: Any, kwargs: dict[str, Any]) -> None:
            if request.thinker_reply_part and artifacts.thinker_reply_part is None:
                value = kwargs.get("thinker_reply_part")
                if value is not None:
                    artifacts.thinker_reply_part = value

        handles.append(talker.register_forward_pre_hook(talker_pre_hook, with_kwargs=True))

    if talker is not None and hasattr(talker, "register_forward_hook"):
        def talker_hook(_module: Any, _args: Any, _kwargs: Any, output: Any) -> None:
            if request.talker_logits:
                artifacts.talker_logits = _output_field(output, "logits")
            if request.talker_hidden_states:
                artifacts.talker_hidden_states = _output_field(output, "hidden_states")
            if request.talker_attentions:
                artifacts.talker_attentions = _output_field(output, "attentions")
            if request.talker_kv_cache:
                artifacts.talker_kv_cache = _output_field(output, "past_key_values")

        handles.append(talker.register_forward_hook(talker_hook, with_kwargs=True))

    if token2wav is not None and hasattr(token2wav, "register_forward_pre_hook"):
        def token2wav_pre_hook(_module: Any, args: Any, kwargs: dict[str, Any]) -> None:
            codec_ids = args[0] if args else kwargs.get("code")
            if request.talker_codec_token_ids:
                artifacts.talker_codec_token_ids = codec_ids
            if request.token2wav_intermediates:
                artifacts.token2wav_intermediates = {
                    "code": codec_ids,
                    "conditioning": kwargs.get("conditioning"),
                    "reference_mel": kwargs.get("reference_mel"),
                }

        handles.append(token2wav.register_forward_pre_hook(token2wav_pre_hook, with_kwargs=True))

    try:
        yield artifacts
    finally:
        for handle in reversed(handles):
            handle.remove()
