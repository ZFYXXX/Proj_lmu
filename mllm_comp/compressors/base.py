"""Compressor plugin protocol."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Mapping, Protocol

from mllm_comp.core.plan import CompressionPlan
from mllm_comp.core.state import ForwardState


@dataclass
class CompressionContext:
    """Runtime context available to a compressor plugin."""

    phase: Literal["dry_run", "analyze", "train", "eval"]
    model: Any | None
    structural_handles: Mapping[str, Any] = field(default_factory=dict)
    analysis_records: list[Any] = field(default_factory=list)
    extra: dict[str, Any] = field(default_factory=dict)


class CompressorPlugin(Protocol):
    name: str
    version: str | None

    def validate_config(self, config: dict[str, Any]) -> None: ...

    def plan(
        self,
        *,
        state: ForwardState,
        context: CompressionContext,
        config: dict[str, Any],
    ) -> CompressionPlan: ...

    def state_dict(self) -> dict[str, Any]: ...

    def load_state_dict(self, state: dict[str, Any]) -> None: ...
