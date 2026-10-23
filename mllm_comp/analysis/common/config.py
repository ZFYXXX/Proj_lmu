"""Strict shared config primitives for built-in analysis Probes."""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence


def validate_common_probe_config(
    config: Mapping[str, Any],
    *,
    context: str,
) -> None:
    """Reject ambiguous booleans shared by every Probe config."""

    for field in ("enabled", "save_raw"):
        if field in config and not isinstance(config[field], bool):
            raise ValueError(f"{context} {field} must be a boolean.")


def string_sequence(
    config: Mapping[str, Any],
    field: str,
    default: Sequence[str],
    *,
    context: str,
    allow_empty: bool = False,
) -> tuple[str, ...]:
    """Return an explicitly typed list/tuple of non-empty strings."""

    raw = config.get(field, default)
    if not isinstance(raw, (list, tuple)):
        raise ValueError(f"{context} {field} must be a list of strings.")
    if any(not isinstance(value, str) or not value for value in raw):
        raise ValueError(
            f"{context} {field} entries must be non-empty strings."
        )
    if not raw and not allow_empty:
        raise ValueError(f"{context} {field} cannot be empty.")
    return tuple(raw)


def validate_layer_selection(value: Any, *, context: str) -> None:
    """Validate ``all`` or a non-empty list of non-negative integer layers."""

    if value == "all":
        return
    if (
        not isinstance(value, list)
        or not value
        or any(
            isinstance(layer, bool)
            or not isinstance(layer, int)
            or layer < 0
            for layer in value
        )
    ):
        raise ValueError(
            f"{context} must be 'all' or a non-empty list of non-negative integers."
        )


def strict_boolean(
    config: Mapping[str, Any],
    field: str,
    default: bool,
    *,
    context: str,
) -> bool:
    value = config.get(field, default)
    if not isinstance(value, bool):
        raise ValueError(f"{context} {field} must be a boolean.")
    return value


def positive_integer(
    config: Mapping[str, Any],
    field: str,
    default: int,
    *,
    context: str,
) -> int:
    value = config.get(field, default)
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{context} {field} must be a positive integer.")
    return value


def nonnegative_finite_number(
    config: Mapping[str, Any],
    field: str,
    default: float,
    *,
    context: str,
) -> float:
    value = config.get(field, default)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{context} {field} must be numeric.")
    numeric = float(value)
    if not math.isfinite(numeric) or numeric < 0.0:
        raise ValueError(f"{context} {field} must be finite and non-negative.")
    return numeric


__all__ = [
    "nonnegative_finite_number",
    "positive_integer",
    "strict_boolean",
    "string_sequence",
    "validate_common_probe_config",
    "validate_layer_selection",
]
