"""Composed backbone facade, component protocols, and capability profiles."""

from .base import (
    BackboneFacade,
    BackboneRuntime,
    GenerationResult,
    ProcessedBatch,
    SpeechGenerationOutput,
    TextGenerationOutput,
    TraceArtifacts,
    TraceRequest,
)
from .profiles import BackboneCapabilities

__all__ = [
    "BackboneCapabilities",
    "BackboneFacade",
    "BackboneRuntime",
    "GenerationResult",
    "ProcessedBatch",
    "SpeechGenerationOutput",
    "TextGenerationOutput",
    "TraceArtifacts",
    "TraceRequest",
]
