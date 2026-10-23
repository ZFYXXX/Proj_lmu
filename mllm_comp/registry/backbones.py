"""Backbone registry.

Only implemented and validated adapters should be registered here.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import cast

from mllm_comp.backbones.base import BackboneFacade
from mllm_comp.core.errors import RegistryError
from mllm_comp.registry.catalog import CapabilityKind, active_providers

BackboneFactory = Callable[[], BackboneFacade]

ACTIVE_BACKBONES: dict[str, BackboneFactory] = cast(
    dict[str, BackboneFactory],
    active_providers(CapabilityKind.BACKBONE),
)


def get_backbone_factory(name: str) -> BackboneFactory:
    """Return the registered facade factory for an active backbone."""

    try:
        return ACTIVE_BACKBONES[name]
    except KeyError as exc:
        raise RegistryError(f"Backbone '{name}' is not an active capability yet.") from exc


def get_backbone(name: str) -> BackboneFacade:
    """Construct an active composed backbone facade."""

    facade = get_backbone_factory(name)()
    if not isinstance(facade, BackboneFacade):
        raise RegistryError(
            f"Backbone factory '{name}' returned {type(facade).__name__}, "
            "expected BackboneFacade."
        )
    facade.capabilities()
    return facade
