"""External integration registry."""

from __future__ import annotations

from typing import Any

from mllm_comp.core.errors import RegistryError
from mllm_comp.registry.catalog import CapabilityKind, active_providers

ACTIVE_INTEGRATIONS: dict[str, type[Any]] = active_providers(CapabilityKind.INTEGRATION)


def get_integration(name: str) -> type[Any]:
    try:
        return ACTIVE_INTEGRATIONS[name]
    except KeyError as exc:
        raise RegistryError(f"Integration '{name}' is not an active capability yet.") from exc
