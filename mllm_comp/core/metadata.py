"""Per-stage metadata records."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .state import Tensor, TokenSpan


@dataclass
class StageMetadata:
    """What happened in one compression stage."""

    stage_name: str
    changed_indices: Tensor | None = None
    changed_spans: list[TokenSpan] = field(default_factory=list)
    stats: dict[str, Any] = field(default_factory=dict)
    artifacts: dict[str, str] = field(default_factory=dict)
