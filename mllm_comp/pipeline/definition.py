"""Resolve and validate compression definitions used by real commands."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from mllm_comp.compressors.contracts import CompressionContractRegistry
from mllm_comp.compressors.extensions import compression_contracts
from mllm_comp.compressors.rules import legacy_rule_specs
from mllm_comp.config.schema import ExperimentConfig
from mllm_comp.config.validators import validate_runtime_batch_size
from mllm_comp.core.errors import BackendIntegrationError, ConfigError, RegistryError
from mllm_comp.core.vocabulary import CompressionMode
from mllm_comp.pipeline.stage import CompressionStageConfig
from mllm_comp.pipeline.v2_validation import (
    V2ContractProjection,
    V2ContractValidationError,
    selected_v2_binding_names,
    validate_v2_stage_contract,
)
from mllm_comp.registry.catalog import (
    CapabilityKind,
    CapabilityStatus,
    get_capability_spec,
)


@dataclass(frozen=True)
class CompressionDefinition:
    """Validated compression stages consumed by a real command."""

    compression_mode: CompressionMode
    stages: tuple[CompressionStageConfig, ...]
    projections: dict[str, V2ContractProjection] = field(default_factory=dict)
    registry: CompressionContractRegistry | None = field(default=None, repr=False, compare=False)


def resolve_compression_definition(
    config: ExperimentConfig,
    *,
    compression_mode: CompressionMode | str | None,
    interface: str | None = None,
    runtime_batch_size: int | None = None,
    backbone_name: str | None = None,
) -> CompressionDefinition:
    """Validate every enabled stage before a real compressed execution."""

    try:
        mode = CompressionMode(compression_mode or CompressionMode.NONE)
    except ValueError as exc:
        raise ConfigError(f"Unsupported compression mode {compression_mode!r}.") from exc
    _validate_command_context(
        config,
        interface=interface,
        runtime_batch_size=runtime_batch_size,
        backbone_name=backbone_name,
    )
    if mode is CompressionMode.NONE:
        return CompressionDefinition(compression_mode=mode, stages=())

    enabled_stages = _enabled_stages(config)
    from mllm_comp.fixed_mask.binding import validate_stages
    validate_stages(enabled_stages)
    if not enabled_stages:
        raise ConfigError(
            "compression_mode=compressed requires at least one enabled pipeline stage."
        )

    duplicate_names = _duplicate_stage_names(enabled_stages)
    if duplicate_names:
        raise ConfigError(
            f"pipeline stage names must be unique, duplicates: {duplicate_names}"
        )

    try:
        selected_binding_names = selected_v2_binding_names(enabled_stages)
    except V2ContractValidationError as exc:
        raise ConfigError(str(exc)) from exc

    registry = compression_contracts(config.pipeline.extensions)
    projections = {}
    for stage in enabled_stages:
        projection = _validate_stage(
            stage,
            backbone_name=backbone_name or config.backbone.name,
            interface=interface,
            selected_binding_names=selected_binding_names,
            selected_stages=enabled_stages,
            registry=registry,
        )
        if projection is not None:
            projections[stage.name] = projection
    return CompressionDefinition(
        compression_mode=mode,
        stages=tuple(enabled_stages),
        projections=projections,
        registry=registry,
    )


def _validate_command_context(
    config: ExperimentConfig,
    *,
    interface: str | None,
    runtime_batch_size: int | None,
    backbone_name: str | None,
) -> None:
    resolved_backbone_name = backbone_name or config.backbone.name
    try:
        backbone_spec = get_capability_spec(
            CapabilityKind.BACKBONE,
            resolved_backbone_name,
        )
    except RegistryError as exc:
        raise ConfigError(str(exc)) from exc
    if not backbone_spec.is_active:
        raise ConfigError(
            f"Backbone capability {resolved_backbone_name!r} is not active."
        )
    profile = backbone_spec.profile
    max_batch = profile.max_analysis_eval_batch_size if profile is not None else None
    if interface is not None:
        if interface not in {"analysis", "eval", "train"}:
            raise ConfigError(f"Unsupported compression interface {interface!r}.")
        section = getattr(config, interface)
        _validate_integration_backend(section.backend, interface=interface)

    if runtime_batch_size is not None:
        validate_runtime_batch_size(
            requested=runtime_batch_size,
            max_analysis_eval_batch_size=max_batch,
            backbone_name=resolved_backbone_name,
        )


def _validate_integration_backend(name: str | None, *, interface: str) -> None:
    if name is None:
        return
    try:
        spec = get_capability_spec(CapabilityKind.INTEGRATION, name)
    except RegistryError as exc:
        raise ConfigError(str(exc)) from exc
    if not spec.is_active:
        raise ConfigError(f"Integration capability {name!r} is not active.")
    if interface not in spec.interfaces:
        raise ConfigError(f"Integration capability {name!r} is not a {interface} backend.")


def _enabled_stages(config: ExperimentConfig) -> list[CompressionStageConfig]:
    return [stage for stage in config.pipeline.stages if stage.enabled]


def _duplicate_stage_names(stages: list[CompressionStageConfig]) -> list[str]:
    seen: set[str] = set()
    duplicates: list[str] = []
    for stage in stages:
        name = stage.name
        if name in seen and name not in duplicates:
            duplicates.append(name)
        seen.add(name)
    return duplicates


def _validate_stage(
    stage: CompressionStageConfig,
    *,
    backbone_name: str,
    interface: str | None,
    registry: CompressionContractRegistry,
    selected_binding_names: tuple[str, ...] = (),
    selected_stages: tuple[Any, ...] | list[Any] = (),
) -> V2ContractProjection | None:
    name = stage.name
    placement = stage.placement.value
    if stage.method is None:
        raise ConfigError(f"stage {name}: method is required.")
    if stage.object_family is None:
        raise ConfigError(f"stage {name}: object_family is required.")
    method = stage.method.value
    object_family = stage.object_family.value

    _validate_method_family(name, placement, method, object_family)
    if method == "token_compression" and not stage.modalities:
        raise ConfigError(f"stage {name}: modalities must be a non-empty list.")

    projection = None
    if stage.uses_v2_contract:
        projection = _validate_v2_contract(
            stage,
            backbone_name=backbone_name,
            interface=interface,
            selected_binding_names=selected_binding_names,
            selected_stages=selected_stages,
            registry=registry,
        )
    else:
        legacy_rule_specs(stage)
        if backbone_name == "qwen25_omni":
            raise ConfigError(
                f"stage {name}: Qwen2.5-Omni compression requires explicit "
                "algorithm, variant and binding v2 identities."
            )
        _validate_legacy_descriptor(stage, backbone_name=backbone_name)
    if projection is not None:
        _validate_v2_provider(stage, projection=projection, interface=interface)
    else:
        _run_stage_schema_validator(
            stage.plugin,
            stage,
            backbone_name=backbone_name,
            interface=interface,
        )
    return projection


def _validate_legacy_descriptor(
    stage: CompressionStageConfig,
    *,
    backbone_name: str,
) -> None:
    from mllm_comp.compressors.registry import get_compression_method_descriptor

    try:
        descriptor = get_compression_method_descriptor(stage.plugin)
    except BackendIntegrationError:
        return
    try:
        descriptor.validate_stage(stage, backbone=backbone_name)
    except (BackendIntegrationError, RegistryError) as exc:
        raise ConfigError(str(exc)) from exc


def _validate_v2_contract(
    stage: CompressionStageConfig,
    *,
    backbone_name: str,
    interface: str | None,
    registry: CompressionContractRegistry,
    selected_binding_names: tuple[str, ...] = (),
    selected_stages: tuple[Any, ...] | list[Any] = (),
) -> V2ContractProjection:
    try:
        projection = validate_v2_stage_contract(
            stage,
            registry=registry,
            backbone_name=backbone_name,
            selected_binding_names=selected_binding_names,
            selected_stages=selected_stages,
        )
    except V2ContractValidationError as exc:
        raise ConfigError(str(exc)) from exc

    # Execution depends on implemented mechanisms and their contracts. Support
    # evidence describes verification; it does not authorize research runs.
    for binding in projection.bindings:
        if not callable(binding.executor_factory):
            raise ConfigError(
                f"stage {stage.name}: binding {binding.name!r} has no executor factory."
            )
    return projection


def _validate_v2_provider(
    stage: CompressionStageConfig,
    *,
    projection: V2ContractProjection,
    interface: str | None,
) -> None:
    algorithm = projection.algorithm
    interfaces = algorithm.metadata.get("interfaces", ("analysis", "eval", "train"))
    if interface is not None and interfaces and interface not in interfaces:
        raise ConfigError(
            f"Compression capability {algorithm.name!r} is not available through "
            f"the {interface} interface."
        )
    validator = algorithm.stage_validator
    if validator is not None:
        try:
            plugin_config = stage.plugin_config()
            plugin_config["config"] = dict(projection.algorithm_config)
            validator(plugin_config)
        except ConfigError:
            raise
        except (TypeError, ValueError) as exc:
            raise ConfigError(f"Invalid config for algorithm {algorithm.name!r}: {exc}") from exc


def _validate_method_family(
    name: str,
    placement: str,
    method: str,
    object_family: str,
) -> None:
    expected = {
        "token_compression": {"tensor_slot"},
        "operation_compression": {"module_unit"},
        "quantization": {"weight", "weight_activation"},
    }[method]
    if object_family not in expected:
        raise ConfigError(
            f"stage {name}: method={method} expects object_family in {sorted(expected)}, "
            f"got {object_family!r}."
        )
    if method == "operation_compression" and placement != "llm_level":
        raise ConfigError(
            "Phase 1 operation compression is only allowed at placement=llm_level."
        )
    if method == "quantization":
        raise ConfigError("Quantization is reserved/future and cannot be enabled.")


def _run_stage_schema_validator(
    plugin: str,
    stage: CompressionStageConfig,
    *,
    backbone_name: str,
    interface: str | None,
) -> None:
    try:
        spec = get_capability_spec(CapabilityKind.COMPRESSOR, plugin)
    except RegistryError as exc:
        raise ConfigError(str(exc)) from exc
    if not stage.uses_v2_contract and not spec.is_active:
        from mllm_comp.compressors.registry import (
            get_compression_method_descriptor,
        )

        try:
            descriptor = get_compression_method_descriptor(plugin)
        except BackendIntegrationError as exc:
            raise ConfigError(
                f"Compression capability {plugin!r} is not active and has no "
                "explicit external descriptor candidate path."
            ) from exc
        if (
            spec.status is not CapabilityStatus.TARGET
            or descriptor.capability_status != CapabilityStatus.TARGET.value
        ):
            raise ConfigError(f"Compression capability {plugin!r} is not active.")
    supported_backbones = spec.metadata.get("backbones")
    if supported_backbones is not None and backbone_name not in supported_backbones:
        raise ConfigError(
            f"Compression capability {plugin!r} does not support backbone "
            f"{backbone_name!r}; expected one of {sorted(supported_backbones)}."
        )
    legacy_backbones = spec.metadata.get("legacy_backbones")
    if (legacy_backbones is not None and backbone_name not in legacy_backbones
            and {item.value for item in stage.modalities} != {"image"}):
        raise ConfigError(
            f"Compression capability {plugin!r} supports backbone {backbone_name!r} "
            "only with modalities=[image]."
        )
    if interface is not None and spec.interfaces and interface not in spec.interfaces:
        raise ConfigError(
            f"Compression capability {plugin!r} is not available through the "
            f"{interface} interface."
        )
    spec.validate_config(stage.plugin_config())
