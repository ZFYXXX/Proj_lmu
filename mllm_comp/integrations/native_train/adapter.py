"""Native source training adapter skeleton."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from mllm_comp.core.errors import NotActivatedError


@dataclass
class TrainResult:
    checkpoint_path: str
    train_log_path: str | None
    metrics: dict[str, float] = field(default_factory=dict)
    backend_args: dict[str, Any] = field(default_factory=dict)


class NativeSourceTrainAdapter:
    """Reserved post-Phase-1a bridge to source-tree training scripts."""

    def train(
        self,
        *,
        config: Any,
        backbone: Any,
        pipeline: Any,
        registry: Any,
        manifest: Any,
    ) -> TrainResult:
        del config, backbone, pipeline, registry, manifest
        raise NotActivatedError(
            "Native source training is not implemented. "
            "Compressed training must prove the real forward path is routed through the platform pipeline."
        )
