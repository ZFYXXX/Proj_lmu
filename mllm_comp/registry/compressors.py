"""Compressor registry.

Skeleton compressor files may exist before activation, but they must not be
added to ACTIVE_COMPRESSORS until their contract tests and smoke path exist.
Target candidates remain available only by exact name so development runs
never masquerade as active capability discovery.  Their execution contract
may be multimodal v2 or an explicitly injected external descriptor.
"""

from __future__ import annotations

from typing import Any

from mllm_comp.core.errors import RegistryError
from mllm_comp.registry.catalog import (
    CapabilityKind,
    CapabilityStatus,
    active_providers,
    get_capability_spec,
)

ACTIVE_COMPRESSORS: dict[str, type[Any]] = active_providers(CapabilityKind.COMPRESSOR)


def get_compressor(name: str) -> type[Any]:
    """Resolve an active provider or an explicitly named target candidate."""

    try:
        return ACTIVE_COMPRESSORS[name]
    except KeyError as exc:
        try:
            spec = get_capability_spec(CapabilityKind.COMPRESSOR, name)
        except RegistryError:
            spec = None
        if (
            spec is not None
            and spec.status is CapabilityStatus.TARGET
            and spec.provider is not None
        ):
            return spec.provider
        raise RegistryError(
            f"Compressor '{name}' is neither active nor an explicit target candidate."
        ) from exc
