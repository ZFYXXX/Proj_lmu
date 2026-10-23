"""Configuration and activation-time loading for supported Python rule hooks.

Rule functions operate at existing algorithm boundaries. The runtime owns model
hooks, layouts and budgets; module factories use the existing replacement
lifecycle. Parsing rule descriptions never imports the configured rule functions.
"""

from __future__ import annotations

from collections.abc import Collection, Mapping
from dataclasses import replace
from functools import partial
from importlib import import_module
import inspect
import json
from typing import Any, Callable

from mllm_comp.core.errors import ConfigError, RegistryError


_LOCATIONS = {"scorer": "criterion", "selector": "operation",
              "grouper": "operation", "aggregator": "operation"}


def rule_specs(stage: Any, *, supported_rules: Collection[str]) -> dict[str, dict[str, Any]]:
    """Normalize the configured rule paths and JSON parameters, without imports."""
    operation = getattr(stage, "operation", {}) or {}
    configured = operation.get("rules", {})
    if not isinstance(configured, Mapping):
        raise ConfigError("operation.rules must be a mapping of rule names to descriptions.")
    if "rules" in (getattr(stage, "criterion", {}) or {}):
        raise ConfigError("Rules must be configured in operation.rules.")
    configured = {name: (raw, f"operation.rules.{name}") for name, raw in configured.items()}
    for name, location in _LOCATIONS.items():
        values = getattr(stage, location, {}) or {}
        other = "operation" if location == "criterion" else "criterion"
        if name in (getattr(stage, other, {}) or {}):
            raise ConfigError(f"Rule {name!r} must be configured in {location}.{name}.")
        if name not in values:
            continue
        if name in configured:
            raise ConfigError(f"Rule {name!r} is configured both in operation.rules and {location}.{name}.")
        configured[name] = (values[name], f"{location}.{name}")
    owner = getattr(stage, "variant", None) or getattr(stage, "plugin", None)
    result = {}
    for name, (raw, location) in configured.items():
        if name not in supported_rules:
            raise ConfigError(f"Variant or plugin {owner!r} does not support the {name!r} rule hook.")
        if not isinstance(raw, Mapping) or set(raw) - {"callable", "params"}:
            raise ConfigError(f"{location} requires a mapping with callable and optional params.")
        path = raw.get("callable")
        parts = path.split(":") if isinstance(path, str) else []
        if (len(parts) != 2 or not parts[1].isidentifier()
                or not all(part.isidentifier() for part in parts[0].split("."))):
            raise ConfigError(f"{location}.callable must be 'package.module:function'.")
        params = raw.get("params", {})
        if not isinstance(params, Mapping) or any(not isinstance(key, str) for key in params):
            raise ConfigError(f"{location}.params must be a mapping with string keys.")
        try:
            params = json.loads(json.dumps(dict(params), allow_nan=False))
        except (TypeError, ValueError) as exc:
            raise ConfigError(f"{location}.params must contain finite JSON values.") from exc
        result[name] = {"callable": path, "params": params}
    return result


def builtin_rule_specs(stage: Any) -> dict[str, dict[str, Any]]:
    """Read rules for a standalone executor that accepts only built-in variants."""
    from mllm_comp.compressors.multimodal_registry import builtin_multimodal_compression_contracts

    # Standalone video executors also allow stage=None for their built-in recipe.
    variant_name = getattr(stage, "variant", None)
    if variant_name is None:
        return rule_specs(stage, supported_rules=frozenset())
    try:
        variant = builtin_multimodal_compression_contracts().variant(variant_name)
    except RegistryError as exc:
        raise ConfigError(str(exc)) from exc
    return rule_specs(stage, supported_rules=variant.supported_rules)


def legacy_rule_specs(stage: Any) -> dict[str, dict[str, Any]]:
    """External legacy descriptors do not declare the v2 Python rule protocol."""
    return rule_specs(stage, supported_rules=())


def load_rules(stage: Any, *, supported_rules: Collection[str]) -> dict[str, Callable[..., Any]]:
    """Resolve plain functions once when constructing the existing executor."""
    return load_rule_specs(rule_specs(stage, supported_rules=supported_rules))


def load_rule_specs(
    specs: Mapping[str, Mapping[str, Any]]
) -> dict[str, Callable[..., Any]]:
    """Load functions from already normalized specs, also used as run metadata."""
    rules = {}
    for name, spec in specs.items():
        module, symbol = spec["callable"].split(":")
        try:
            function = getattr(import_module(module), symbol)
        except (ImportError, AttributeError) as exc:
            raise ConfigError(f"Cannot load {name} rule {spec['callable']!r}: {exc}") from exc
        if not inspect.isfunction(function):
            raise ConfigError(f"{name} rule {spec['callable']!r} must be a plain Python function.")
        try:
            inspect.signature(function).bind(object(), **spec["params"])
        except TypeError as exc:
            raise ConfigError(f"Invalid parameters for {name} rule {spec['callable']!r}: {exc}") from exc
        rules[name] = partial(function, **spec["params"])
    return rules


def resolve_rule_binding(*, binding: Any, rules: Mapping[str, Mapping[str, Any]]) -> Any:
    """Attach research-rule metadata to a copy of the selected physical binding."""
    if not rules:
        return binding
    return replace(binding, metadata={**binding.metadata, "rules_changed": True, "rules": dict(rules)})


def recipe_changed(binding: Any) -> bool:
    """Whether historical standard-recipe evidence is inapplicable to this run."""
    return bool(binding.metadata.get("position_changed") or binding.metadata.get("rules_changed"))
