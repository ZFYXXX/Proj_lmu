"""Shared native configuration boundary for decoder image methods."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import fields
import math
from numbers import Real
from typing import Any


def require_finite_real(value: Any, name: str) -> float:
    """Return a finite real executor value while rejecting booleans."""

    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number, not a boolean.")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite.")
    return result


def resolve_decoder_method_config(*, stage, binding, variant, binding_config=None):
    """Resolve the same method parameters for position planning and execution.

    Explicit legacy knobs override Variant defaults. Private stage parameters
    then override legacy knobs, and the selected binding has the final say.
    """
    keys = getattr(binding.strategy_factory, "legacy_config_keys", ())
    requested = {key: value for section in ("operation", "criterion", "schedule")
                 for key, value in (getattr(stage, section, {}) or {}).items() if key in keys}
    requested.update(getattr(stage, "config", {}) or {})
    requested.update(getattr(binding_config, "config", {}) or {})
    return dict(variant.resolve_config(requested))


def merge_decoder_executor_config(
    algorithm: str,
    *,
    stage: Any,
    binding_config: Any,
) -> dict[str, Any]:
    """Merge one decoder method's knobs and reject misspelled explicit keys.

    Operation, criterion, and schedule mappings are control-plane schemas, so
    only recognized algorithm knobs are selected from them. ``stage.config``
    and ``binding_config.config`` are explicit executor configuration and must
    therefore fail closed on unknown keys.
    """

    # Only the retained direct built-in entry points dispatch by name.
    # Registered methods already supply their own keys through the binding.
    # Import lazily: algorithm Specs use require_finite_real from this module.
    from . import decoder_selection as methods
    from .sgl import SGLSpec

    keys = {
        "pact": methods.PACTDecoderMethod.legacy_config_keys,
        "fastv": methods.FastVDecoderMethod.legacy_config_keys,
        "dart": methods.DARTDecoderMethod.legacy_config_keys,
        "d2pruner": methods.D2PrunerDecoderMethod.legacy_config_keys,
        "sparsevlm": methods.SparseVLMDecoderMethod.legacy_config_keys,
        "finepruner": methods.FinePrunerDecoderMethod.legacy_config_keys,
        "btp": methods.BTPDecoderMethod.legacy_config_keys,
        "halfv": methods.HalfVDecoderMethod.legacy_config_keys,
        "transprune": methods.TransPruneDecoderMethod.legacy_config_keys,
        "feather": methods.FEATHERDecoderMethod.legacy_config_keys,
        "ivc_prune": methods.IVCPruneDecoderMethod.legacy_config_keys,
        "sgl": frozenset(field.name for field in fields(SGLSpec)),
    }
    try:
        allowed = keys[algorithm]
    except KeyError as exc:
        raise ValueError(f"Unknown decoder image algorithm {algorithm!r}.") from exc

    result: dict[str, Any] = {}
    if stage is not None:
        for name in ("operation", "criterion", "schedule"):
            value = getattr(stage, name, None)
            if isinstance(value, Mapping):
                result.update(
                    {key: item for key, item in value.items() if key in allowed}
                )

    explicit_sources = (
        ("stage.config", getattr(stage, "config", None)),
        ("binding_config.config", getattr(binding_config, "config", None)),
    )
    for source_name, value in explicit_sources:
        if not isinstance(value, Mapping):
            continue
        unknown = tuple(sorted((key for key in value if key not in allowed), key=repr))
        if unknown:
            raise ValueError(
                f"Unknown {algorithm} executor config keys in {source_name}: "
                f"{list(unknown)!r}."
            )
        result.update(dict(value))
    return result


__all__ = [
    "merge_decoder_executor_config",
    "require_finite_real",
]
