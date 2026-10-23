"""Activate compression through the generic multimodal runtime session."""

from __future__ import annotations

from typing import Any

from mllm_comp.core.errors import BackendIntegrationError, ConfigError
from mllm_comp.core.vocabulary import CompressionMode, ExecutionPhase
from mllm_comp.pipeline.definition import resolve_compression_definition


def activate_runtime_compression(
    *,
    config: Any,
    facade: Any,
    runtime: Any,
    compression_mode: CompressionMode | str | None,
    execution_phase: ExecutionPhase | str = ExecutionPhase.GENERATE,
) -> Any | None:
    """Resolve, install and publish one transactional compression session."""

    split_spec = config.backbone.config.get("modality_split")
    split_checkpoint = config.backbone.config.get("modality_split_checkpoint")
    if split_spec is not None or split_checkpoint is not None:
        if CompressionMode(compression_mode or "none") is not CompressionMode.COMPRESSED:
            raise BackendIntegrationError("modality_split requires compression_mode=compressed.")
        if any(stage.enabled for stage in config.pipeline.stages):
            raise BackendIntegrationError("modality_split cannot combine with legacy pipeline stages.")
        from mllm_comp.runtime.split_support import activate_split

        return activate_split(runtime, config.backbone.name, split_spec, split_checkpoint)

    try:
        definition = resolve_compression_definition(
            config,
            compression_mode=compression_mode,
            backbone_name=facade.capabilities().name,
        )
    except ConfigError as exc:
        raise BackendIntegrationError(str(exc)) from exc
    if definition.compression_mode is CompressionMode.NONE:
        return None

    from mllm_comp.pipeline.runtime_session import RuntimeCompressionDispatcher

    return RuntimeCompressionDispatcher(contract_registry=definition.registry).activate(
        config=config,
        facade=facade,
        runtime=runtime,
        compression_mode=definition.compression_mode,
        execution_phase=ExecutionPhase(execution_phase),
        definition=definition,
    )


__all__ = ["activate_runtime_compression"]
