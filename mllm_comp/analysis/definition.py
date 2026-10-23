"""Resolve and validate the effective definition for a real analysis run."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Any

from mllm_comp.config.schema import ExperimentConfig
from mllm_comp.core.errors import ConfigError, RegistryError
from mllm_comp.pipeline.definition import resolve_compression_definition
from mllm_comp.registry.catalog import (
    CapabilityKind,
    CapabilitySpec,
    get_capability_spec,
)


_PROBE_OVERRIDE_FIELDS = (
    "token_groups",
    "collect_points",
    "operations",
    "target_token_types",
    "attn_edges",
)


@dataclass
class AnalysisOverrides:
    """Temporary CLI overrides applied to one analysis run."""

    compression_mode: str | None = None
    phase: str | None = None
    tasks: str | None = None
    sample_limit: int | None = None
    runtime_batch_size: int | None = None
    probes: list[str] | None = None
    layers: str | None = None
    metrics: str | None = None
    token_groups: str | None = None
    collect_points: str | None = None
    operations: str | None = None
    target_token_types: str | None = None
    attn_edges: str | None = None
    save_raw: bool = False


@dataclass
class AnalysisDefinition:
    """Validated effective inputs consumed by the analysis runtime."""

    compression_mode: str
    phase: str
    tasks: list[str]
    sample_limit: int | None
    runtime_batch_size: int
    probes: list[dict[str, Any]]


def resolve_analysis_definition(
    config: ExperimentConfig,
    overrides: AnalysisOverrides,
) -> AnalysisDefinition:
    """Apply CLI overrides and validate inputs before loading the model."""

    analysis = config.analysis
    compression_mode = overrides.compression_mode or analysis.compression_mode
    phase = overrides.phase or analysis.phase
    tasks = _split_csv(overrides.tasks) if overrides.tasks else analysis.tasks
    sample_limit = overrides.sample_limit if overrides.sample_limit is not None else analysis.sample_limit
    runtime_batch_size = (
        overrides.runtime_batch_size
        if overrides.runtime_batch_size is not None
        else analysis.runtime_batch_size
    )
    probes = _select_probe_configs(analysis.probes, overrides.probes)

    _validate_analysis_phase(phase)
    resolve_compression_definition(
        config,
        compression_mode=compression_mode,
        interface="analysis",
        runtime_batch_size=runtime_batch_size,
    )
    _validate_effective_inputs(tasks=tasks, sample_limit=sample_limit)
    _apply_probe_overrides(probes, overrides)
    _validate_probe_configs(probes, backbone_name=config.backbone.name)

    return AnalysisDefinition(
        compression_mode=compression_mode,
        phase=phase,
        tasks=tasks,
        sample_limit=sample_limit,
        runtime_batch_size=runtime_batch_size,
        probes=probes,
    )


def _select_probe_configs(
    configured_probes: list[dict[str, Any]],
    requested_names: list[str] | None,
) -> list[dict[str, Any]]:
    enabled: list[dict[str, Any]] = []
    for probe in configured_probes:
        flag = probe.get("enabled", True)
        if not isinstance(flag, bool):
            raise ConfigError("analysis Probe enabled must be a boolean.")
        if flag:
            enabled.append(deepcopy(probe))
    seen_logical_names: set[str] = set()
    for probe in enabled:
        logical_name = _logical_probe_name(probe)
        if logical_name in seen_logical_names:
            raise ConfigError(
                f"analysis Probe logical name {logical_name!r} is duplicated."
            )
        seen_logical_names.add(logical_name)
    if not requested_names:
        if not enabled:
            raise ConfigError("analysis.probes must contain at least one enabled probe.")
        return enabled

    by_name: dict[str, dict[str, Any]] = {}
    for probe in enabled:
        name = str(probe.get("name", ""))
        method = str(probe.get("method", ""))
        if name:
            if name in by_name and by_name[name] is not probe:
                raise ConfigError(
                    f"Analysis Probe selector {name!r} is ambiguous."
                )
            by_name[name] = probe
        if method:
            if method in by_name and by_name[method] is not probe:
                raise ConfigError(
                    f"Analysis Probe selector {method!r} is ambiguous."
                )
            by_name[method] = probe

    selected: list[dict[str, Any]] = []
    for name in requested_names:
        selected.append(deepcopy(by_name.get(name, {"name": name, "enabled": True, "config": {}})))
    return selected


def _apply_probe_overrides(
    probes: list[dict[str, Any]],
    overrides: AnalysisOverrides,
) -> None:
    if overrides.layers is not None:
        layers = _parse_layers(overrides.layers)
        for probe in probes:
            probe["layers"] = layers

    if overrides.metrics is not None:
        _apply_metric_override(probes, _split_csv(overrides.metrics))

    for field_name in _PROBE_OVERRIDE_FIELDS:
        value = getattr(overrides, field_name)
        if value is None:
            continue
        parsed = _split_csv(value)
        matched = False
        for probe in probes:
            spec = _spec_for_probe(probe)
            if field_name in spec.override_fields:
                probe[field_name] = parsed
                matched = True
        if not matched:
            supported_probes = sorted(
                _probe_display_name(probe)
                for probe in probes
                if field_name in _spec_for_probe(probe).override_fields
            )
            support_message = ", ".join(supported_probes) or "none of the selected probes"
            raise ConfigError(
                f"--{field_name.replace('_', '-')} only applies to "
                f"{support_message}."
            )

    if overrides.save_raw:
        for probe in probes:
            probe["save_raw"] = True


def _apply_metric_override(probes: list[dict[str, Any]], metrics: list[str]) -> None:
    if not metrics:
        return

    if len(probes) == 1:
        probes[0]["metrics"] = metrics
        return

    unmatched: list[str] = []
    per_probe: dict[int, list[str]] = {}
    for metric in metrics:
        matched_indices: list[int] = []
        for index, probe in enumerate(probes):
            schema = _schema_for_probe(probe)
            supported_metrics = getattr(schema, "supported_metrics", set())
            if metric in supported_metrics:
                matched_indices.append(index)
        if not matched_indices:
            unmatched.append(metric)
            continue
        for index in matched_indices:
            per_probe.setdefault(index, []).append(metric)

    if unmatched:
        raise ConfigError(f"Unsupported analysis metrics in override: {unmatched}")

    for index, selected_metrics in per_probe.items():
        probes[index]["metrics"] = selected_metrics


def _validate_probe_configs(
    probes: list[dict[str, Any]],
    *,
    backbone_name: str,
) -> None:
    logical_names: set[str] = set()
    for probe in probes:
        logical_name = _logical_probe_name(probe)
        if logical_name in logical_names:
            raise ConfigError(
                f"analysis Probe logical name {logical_name!r} is duplicated."
            )
        logical_names.add(logical_name)
        spec = _spec_for_probe(probe)
        supported_backbones = spec.metadata.get("backbones")
        if supported_backbones is not None and backbone_name not in supported_backbones:
            raise ConfigError(
                f"Analysis probe {logical_name!r} does not declare "
                f"support for backbone {backbone_name!r}; supported backbones: "
                f"{sorted(supported_backbones)}."
            )
        spec.validate_config(probe)


def _validate_effective_inputs(
    *,
    tasks: Any,
    sample_limit: Any,
) -> None:
    if (
        not isinstance(tasks, list)
        or not tasks
        or any(not isinstance(task, str) or not task for task in tasks)
    ):
        raise ConfigError("analysis.tasks must be a non-empty list of task names.")
    if sample_limit is not None and (
        isinstance(sample_limit, bool)
        or not isinstance(sample_limit, int)
        or sample_limit < 1
    ):
        raise ConfigError("analysis.sample_limit must be null or a positive integer.")


def _schema_for_probe(probe: dict[str, Any]) -> type[Any]:
    spec = _spec_for_probe(probe)
    if not isinstance(spec.provider, type):
        raise ConfigError(
            f"Analysis probe '{_probe_display_name(probe)}' is not available for "
            "configuration validation."
        )
    return spec.provider


def _spec_for_probe(probe: dict[str, Any]) -> CapabilitySpec:
    key = _probe_key(probe)
    try:
        return get_capability_spec(CapabilityKind.ANALYSIS, key)
    except RegistryError as exc:
        raise ConfigError(
            f"Analysis probe '{_probe_display_name(probe)}' is not declared in the capability catalog."
        ) from exc


def _validate_analysis_phase(phase: str) -> None:
    if phase not in {"pre_compression", "post_compression"}:
        raise ConfigError(f"analysis.phase must be pre_compression or post_compression, got {phase!r}.")


def _parse_layers(value: str) -> str | list[int]:
    if value == "all":
        return "all"
    layers: list[int] = []
    for item in _split_csv(value):
        try:
            layer = int(item)
        except ValueError as exc:
            raise ConfigError(f"Layer override must be 'all' or comma-separated integers, got {value!r}.") from exc
        if layer < 0:
            raise ConfigError("Layer override cannot contain negative layer ids.")
        layers.append(layer)
    if not layers:
        raise ConfigError("Layer override cannot be empty.")
    return layers


def _split_csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def _probe_key(probe: dict[str, Any]) -> str:
    method = probe.get("method")
    if isinstance(method, str) and method:
        return method
    return str(probe.get("name", ""))


def _probe_display_name(probe: dict[str, Any]) -> str:
    name = str(probe.get("name", ""))
    method = str(probe.get("method", ""))
    if method and method != name:
        return f"{name}({method})"
    return name or method


def _logical_probe_name(probe: dict[str, Any]) -> str:
    name = probe.get("name")
    method = probe.get("method")
    if name is not None and (not isinstance(name, str) or not name):
        raise ConfigError("analysis Probe name must be a non-empty string.")
    if method is not None and (not isinstance(method, str) or not method):
        raise ConfigError("analysis Probe method must be a non-empty string.")
    logical_name = name or method
    if not logical_name:
        raise ConfigError("analysis Probe requires a non-empty name or method.")
    return logical_name
