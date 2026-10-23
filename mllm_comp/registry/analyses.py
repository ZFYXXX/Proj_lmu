"""Analysis plugin registry."""

from __future__ import annotations

from typing import Any

from mllm_comp.core.errors import RegistryError
from mllm_comp.registry.catalog import (
    CapabilityKind,
    active_providers,
)

ACTIVE_ANALYSES: dict[str, type[Any]] = active_providers(CapabilityKind.ANALYSIS)


def get_analysis(name: str) -> type[Any]:
    try:
        return ACTIVE_ANALYSES[name]
    except KeyError as exc:
        raise RegistryError(
            f"Analysis plugin '{name}' is not an active capability."
        ) from exc
