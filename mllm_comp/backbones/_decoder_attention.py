"""Transactional decoder-attention implementation overrides.

PACT adds a per-key ``log(cluster_mass)`` bias after its decoder boundary.
FlashAttention 2 does not expose an arbitrary additive-bias input, whereas the
pinned Transformers SDPA path does.  This helper switches only decoder config
objects for the lifetime of one PACT activation and restores the exact prior
values on every exit path.
"""

from __future__ import annotations

from typing import Any

from mllm_comp.core.errors import BackendIntegrationError


_MISSING = object()


class DecoderAttentionImplementationOverride:
    """Temporarily select one implementation on all decoder attention configs."""

    def __init__(
        self,
        model: Any,
        implementation: str = "sdpa",
        *,
        require_configs: bool = True,
    ) -> None:
        if not isinstance(implementation, str) or not implementation:
            raise ValueError("implementation must be a non-empty string")
        self.model = model
        self.implementation = implementation
        self.require_configs = require_configs
        self._entries: list[tuple[Any, Any]] = []

    @property
    def installed(self) -> bool:
        return bool(self._entries)

    def install(self) -> None:
        if self._entries:
            return
        configs = _decoder_attention_configs(self.model)
        if not configs:
            if self.require_configs:
                raise BackendIntegrationError(
                    "PACT could not resolve decoder attention configs for its "
                    "proportional-attention SDPA path."
                )
            return
        installed: list[tuple[Any, Any]] = []
        try:
            for config in configs:
                previous = getattr(config, "_attn_implementation", _MISSING)
                setattr(config, "_attn_implementation", self.implementation)
                installed.append((config, previous))
        except BaseException:
            for config, previous in reversed(installed):
                _restore_implementation(config, previous)
            raise
        self._entries = installed

    def restore(self) -> None:
        entries, self._entries = self._entries, []
        for config, previous in reversed(entries):
            _restore_implementation(config, previous)

    def metadata(self) -> dict[str, Any]:
        return {
            "installed": self.installed,
            "implementation": self.implementation,
            "decoder_config_count": len(self._entries),
            "reason": "pact_log_cluster_mass_attention_bias",
        }


def _decoder_attention_configs(model: Any) -> tuple[Any, ...]:
    roots: list[Any] = []
    model_core = getattr(model, "model", None)
    language_model = getattr(model_core, "language_model", None)
    if language_model is not None:
        roots.append(language_model)
    get_model = getattr(model, "get_model", None)
    if callable(get_model):
        core = get_model()
        if core is not None:
            roots.append(core)
    if model_core is not None:
        roots.append(model_core)

    configs: list[Any] = []
    seen: set[int] = set()

    def add_config(value: Any) -> None:
        if value is None or id(value) in seen:
            return
        seen.add(id(value))
        configs.append(value)

    for root in roots:
        layers = tuple(getattr(root, "layers", ()) or ())
        if not layers:
            nested = getattr(root, "model", None)
            layers = tuple(getattr(nested, "layers", ()) or ())
            if layers:
                add_config(getattr(nested, "config", None))
        if not layers:
            continue
        add_config(getattr(root, "config", None))
        for layer in layers:
            attention = getattr(layer, "self_attn", None)
            add_config(getattr(attention, "config", None))
        if configs:
            break
    return tuple(configs)


def _restore_implementation(config: Any, previous: Any) -> None:
    if previous is _MISSING:
        try:
            delattr(config, "_attn_implementation")
        except AttributeError:
            pass
    else:
        setattr(config, "_attn_implementation", previous)


__all__ = ["DecoderAttentionImplementationOverride"]
