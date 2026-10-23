"""Config loading helpers."""

from __future__ import annotations

import os
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from mllm_comp.core.errors import ConfigError

from .schema import ExperimentConfig, experiment_config_from_dict


_PORTABLE_ENVIRONMENT_VARIABLE = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")


def _expand_environment_variables(text: str) -> str:
    """Expand ``${NAME}`` on every host while preserving unresolved placeholders."""

    portable = _PORTABLE_ENVIRONMENT_VARIABLE.sub(
        lambda match: os.environ.get(match.group(1), match.group(0)),
        text,
    )
    return os.path.expandvars(portable)


def load_config(
    path: str | os.PathLike[str],
    *,
    overrides: Mapping[str, Any] | None = None,
) -> ExperimentConfig:
    """Expand YAML, apply explicit field overrides, then parse its typed schema.

    Overrides use the same nested field names as the config. Mappings update
    their supplied fields; scalar and list values replace the configured value.
    ``config.raw`` retains the original expanded YAML for provenance.
    """

    try:
        import yaml
    except ImportError as exc:
        raise NotImplementedError(
            "YAML config loading requires PyYAML."
        ) from exc

    config_path = Path(path)
    text = _expand_environment_variables(config_path.read_text(encoding="utf-8"))
    try:
        raw: dict[str, Any] = yaml.safe_load(text) or {}
    except yaml.YAMLError as exc:
        raise ConfigError(f"Invalid YAML config {config_path}: {exc}") from exc
    effective = _apply_overrides(raw, overrides) if overrides else raw
    try:
        config = experiment_config_from_dict(effective, path=str(config_path))
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"Invalid config value: {exc}") from exc
    config.raw = raw
    return config


def _apply_overrides(
    configured: Any,
    overrides: Mapping[str, Any],
    *,
    path: str = "config",
) -> dict[str, Any]:
    if not isinstance(configured, Mapping):
        raise ConfigError(f"{path} must be a mapping.")
    effective = dict(configured)
    for name, value in overrides.items():
        effective[name] = (
            _apply_overrides(configured.get(name, {}), value, path=f"{path}.{name}")
            if isinstance(value, Mapping)
            else value
        )
    return effective
