"""Small helpers for provider-owned video text conditioning."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from mllm_comp.core.errors import BackendIntegrationError


def user_text_from_conversation(
    conversation: Sequence[Mapping[str, Any]], *, required: bool = False
) -> str | None:
    """Return user text without narrowing native text-free video inputs."""

    values: list[str] = []
    for message in conversation:
        if str(message.get("role", "")).lower() != "user":
            continue
        content = message.get("content")
        if isinstance(content, str):
            if content.strip():
                values.append(content.strip())
            continue
        if not isinstance(content, Sequence) or isinstance(content, (str, bytes)):
            continue
        for item in content:
            if not isinstance(item, Mapping) or item.get("type") != "text":
                continue
            text = item.get("text")
            if isinstance(text, str) and text.strip():
                values.append(text.strip())
    result = "\n".join(values)
    if not result and required:
        raise BackendIntegrationError(
            "Text-conditioned video compression requires non-empty user text"
        )
    return result or None


def single_video_conditioning_text(
    metadata: Mapping[str, Any], *, required: bool = False
) -> str | None:
    """Resolve optional batch-one text without burdening text-free methods."""

    values = metadata.get("video_conditioning_texts")
    if (
        not isinstance(values, Sequence)
        or isinstance(values, (str, bytes))
        or len(values) != 1
        or not isinstance(values[0], str)
        or not values[0].strip()
    ):
        if required:
            raise BackendIntegrationError(
                "Video compression requires exactly one non-empty conditioning text"
            )
        return None
    return values[0].strip()


__all__ = ["single_video_conditioning_text", "user_text_from_conversation"]
