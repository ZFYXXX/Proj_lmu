"""Shared semantic validation for v2 compression stages.

Definition resolution and runtime intentionally use the same validator. Callers translate
``V2ContractValidationError`` into their interface-specific public exception.
Model structural handles are optional during definition validation and become
mandatory only when a runtime facade can expose them.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any

from mllm_comp.compressors.rules import resolve_rule_binding, rule_specs
from mllm_comp.core.errors import ConfigError, RegistryError
from mllm_comp.core.vocabulary import (
    ExecutionPhase,
    MediaAlignment,
    Modality,
    ModalityCouplingPolicy,
    Placement,
)


class V2ContractValidationError(ValueError):
    """Raised when a stage and its selected physical bindings disagree."""


@dataclass(frozen=True)
class V2BindingSelection:
    """Normalized binding-local configuration shared by both validation paths."""

    binding: str
    target_modalities: frozenset[Modality] = frozenset()
    depends_on: tuple[str, ...] = ()
    order: int = 0
    config: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.binding, str) or not self.binding.strip():
            raise V2ContractValidationError(
                "A v2 binding selection must have a non-empty binding name."
            )
        try:
            modalities = frozenset(Modality(item) for item in self.target_modalities)
        except (TypeError, ValueError) as exc:
            raise V2ContractValidationError(
                f"Binding {self.binding!r} has invalid target modalities."
            ) from exc
        dependencies = tuple(self.depends_on)
        if any(
            not isinstance(item, str) or not item.strip() for item in dependencies
        ):
            raise V2ContractValidationError(
                f"Binding {self.binding!r} depends_on contains an empty name."
            )
        if isinstance(self.order, bool) or not isinstance(self.order, int):
            raise V2ContractValidationError(
                f"Binding {self.binding!r} order must be an integer."
            )
        if not isinstance(self.config, Mapping):
            raise V2ContractValidationError(
                f"Binding {self.binding!r} config must be a mapping."
            )
        object.__setattr__(self, "target_modalities", modalities)
        object.__setattr__(self, "depends_on", dependencies)
        object.__setattr__(self, "config", MappingProxyType(dict(self.config)))


@dataclass(frozen=True)
class V2ContractProjection:
    """Validated descriptors and normalized selections for one logical stage."""

    algorithm: Any
    variant: Any
    selections: tuple[V2BindingSelection, ...]
    bindings: tuple[Any, ...]
    required_handles: frozenset[str]
    algorithm_config: Mapping[str, Any]

    @property
    def binding_names(self) -> tuple[str, ...]:
        return tuple(item.binding for item in self.selections)


def normalize_v2_binding_selections(stage: Any) -> tuple[V2BindingSelection, ...]:
    """Normalize single or compound stage binding syntax without registry access."""

    compound = tuple(getattr(stage, "bindings", ()) or ())
    single = getattr(stage, "binding", None)
    if compound and single:
        raise V2ContractValidationError(
            f"stage {getattr(stage, 'name', '?')}: use either binding or bindings, "
            "not both."
        )
    if compound:
        selections = tuple(
            V2BindingSelection(
                binding=getattr(item, "binding", ""),
                target_modalities=frozenset(
                    getattr(item, "target_modalities", ()) or ()
                ),
                depends_on=tuple(getattr(item, "depends_on", ()) or ()),
                order=getattr(item, "order", 0),
                config=getattr(item, "config", {}) or {},
            )
            for item in sorted(compound, key=lambda value: getattr(value, "order", 0))
        )
    elif single:
        selections = (V2BindingSelection(binding=single),)
    else:
        raise V2ContractValidationError(
            f"stage {getattr(stage, 'name', '?')}: a v2 binding is required."
        )
    names = [item.binding for item in selections]
    if len(names) != len(set(names)):
        raise V2ContractValidationError(
            f"stage {getattr(stage, 'name', '?')}: duplicate v2 binding selection."
        )
    return selections


def validate_v2_stage_contract(
    stage: Any,
    *,
    registry: Any,
    backbone_name: str,
    execution_phase: ExecutionPhase | str | None = None,
    selected_binding_names: Iterable[str] = (),
    selected_stages: Iterable[Any] = (),
    available_handles: Mapping[str, Any] | Iterable[str] | None = None,
) -> V2ContractProjection:
    """Validate one stage against its algorithm, variant, and physical bindings."""

    stage_name = getattr(stage, "name", "?")
    algorithm_name = getattr(stage, "algorithm", None)
    variant_name = getattr(stage, "variant", None)
    try:
        algorithm = registry.algorithm(algorithm_name)
        variant = registry.variant(variant_name)
    except (RegistryError, KeyError, TypeError) as exc:
        raise V2ContractValidationError(f"stage {stage_name}: {exc}") from exc

    if variant.algorithm != algorithm.name:
        raise V2ContractValidationError(
            f"stage {stage_name}: variant {variant.name!r} belongs to algorithm "
            f"{variant.algorithm!r}, not {algorithm.name!r}."
        )
    stage_method = getattr(stage, "method", None)
    if stage_method is not None and stage_method != algorithm.method:
        raise V2ContractValidationError(
            f"stage {stage_name}: method {stage_method!r} does "
            f"not match algorithm {algorithm.name!r} method {algorithm.method.value!r}."
        )
    plugin_name = getattr(stage, "plugin", None)
    if plugin_name != algorithm.name:
        raise V2ContractValidationError(
            f"stage {stage_name}: plugin {plugin_name!r} does not match algorithm "
            f"{algorithm.name!r}."
        )
    try:
        placement = Placement(getattr(stage, "placement"))
    except (AttributeError, TypeError, ValueError) as exc:
        raise V2ContractValidationError(
            f"stage {stage_name}: placement is missing or invalid."
        ) from exc
    _validate_variant_semantics(stage, variant, stage_name=stage_name)
    try:
        algorithm_config = variant.resolve_config(
            getattr(stage, "config", {}) or {}
        )
    except (ConfigError, RegistryError, TypeError, ValueError) as exc:
        raise V2ContractValidationError(
            f"stage {stage_name}: invalid config for variant {variant.name!r}: {exc}"
        ) from exc
    try:
        rules = rule_specs(stage, supported_rules=variant.supported_rules)
    except ConfigError as exc:
        raise V2ContractValidationError(f"stage {stage_name}: {exc}") from exc
    selections = normalize_v2_binding_selections(stage)
    phase = _stage_execution_phase(
        stage,
        execution_phase=execution_phase,
        stage_name=stage_name,
    )
    selected_names = frozenset(selected_binding_names) or frozenset(
        item.binding for item in selections
    )
    bindings: list[Any] = []
    selected_stages = tuple(selected_stages)
    required_handles: set[str] = set()
    for selection in selections:
        try:
            binding = registry.binding(selection.binding)
        except (RegistryError, KeyError, TypeError) as exc:
            raise V2ContractValidationError(f"stage {stage_name}: {exc}") from exc
        if binding.backbone != backbone_name:
            raise V2ContractValidationError(
                f"stage {stage_name}: binding {binding.name!r} targets backbone "
                f"{binding.backbone!r}, not {backbone_name!r}."
            )
        if binding.variant != variant.name:
            raise V2ContractValidationError(
                f"stage {stage_name}: binding {binding.name!r} does not implement "
                f"variant {variant.name!r}."
            )
        position_resolver = getattr(binding, "position_resolver", None)
        if position_resolver is not None:
            try:
                binding = position_resolver(
                    stage=stage, binding=binding, algorithm=algorithm, variant=variant,
                )
            except (ConfigError, RegistryError, TypeError, ValueError) as exc:
                raise V2ContractValidationError(
                    f"stage {stage_name}: invalid compression position: {exc}"
                ) from exc
        elif placement not in algorithm.supported_placements:
            raise V2ContractValidationError(
                f"stage {stage_name}: algorithm {algorithm.name!r} does not support "
                f"placement {placement.value!r}."
            )
        binding = resolve_rule_binding(binding=binding, rules=rules)
        if phase not in binding.supported_phases:
            raise V2ContractValidationError(
                f"stage {stage_name}: binding {binding.name!r} does not support "
                f"phase {phase.value!r}."
            )
        binding_placements = frozenset(
            target.placement for target in binding.targets
        )
        if placement not in binding_placements:
            raise V2ContractValidationError(
                f"stage {stage_name}: binding {binding.name!r} has no physical "
                f"target at placement {placement.value!r}."
            )
        if (
            selection.target_modalities
            and selection.target_modalities != binding.target_modalities
        ):
            raise V2ContractValidationError(
                f"stage {stage_name}: binding selection {binding.name!r} targets "
                f"{_modality_names(selection.target_modalities)}, but its physical "
                f"contract targets {_modality_names(binding.target_modalities)}."
            )
        allowed_binding_config = getattr(binding, "metadata", {}).get(
            "binding_config_keys"
        )
        if allowed_binding_config is not None:
            allowed_keys = frozenset(allowed_binding_config)
            unknown_keys = sorted(set(selection.config) - allowed_keys)
            if unknown_keys:
                raise V2ContractValidationError(
                    f"stage {stage_name}: binding {binding.name!r} config contains "
                    f"unknown fields {unknown_keys}."
                )
        conflicts = frozenset(getattr(binding, "conflicts", ())) & selected_names
        conflicts -= {binding.name}
        slot = binding.exclusive_runtime_slot
        if slot and selected_stages:
            # A moved recipe occupies its resolved executor, so historical
            # binding-name conflicts cannot decide whether stages can coexist.
            conflicts = set()
            for other in selected_stages:
                if other is stage or not _uses_v2_contract(other):
                    continue
                for other_selection in normalize_v2_binding_selections(other):
                    try:
                        candidate = registry.binding(other_selection.binding)
                        if candidate.position_resolver is not None:
                            candidate = candidate.position_resolver(
                                stage=other, binding=candidate,
                                algorithm=registry.algorithm(other.algorithm),
                                variant=registry.variant(other.variant),
                            )
                    except (ConfigError, RegistryError, TypeError, ValueError) as exc:
                        raise V2ContractValidationError(f"stage {other.name}: invalid compression position: {exc}") from exc
                    if candidate.exclusive_runtime_slot == slot:
                        conflicts.add(f"{candidate.name} (stage {other.name})")
        if conflicts:
            raise V2ContractValidationError(
                f"stage {stage_name}: binding {binding.name!r} conflicts with "
                f"selected bindings {sorted(conflicts)}."
            )
        required_handles.update(_binding_required_handles(binding))
        bindings.append(binding)

    projection = V2ContractProjection(
        algorithm=algorithm,
        variant=variant,
        selections=selections,
        bindings=tuple(bindings),
        required_handles=frozenset(required_handles),
        algorithm_config=algorithm_config,
    )
    validate_v2_runtime_context(
        stage, projection, execution_phase=phase, available_handles=available_handles
    )
    return projection


def validate_v2_runtime_context(
    stage: Any,
    projection: V2ContractProjection,
    *,
    execution_phase: ExecutionPhase | str,
    available_handles: Mapping[str, Any] | Iterable[str] | None,
) -> None:
    """Check runtime-only requirements without rebuilding a validated contract."""
    stage_name = getattr(stage, "name", "?")
    _stage_execution_phase(
        stage, execution_phase=execution_phase, stage_name=stage_name
    )
    missing_handles = _missing_required_handles(
        projection.required_handles, available_handles
    )
    if missing_handles:
        raise V2ContractValidationError(
            f"stage {stage_name}: selected bindings require missing structural "
            f"handles {sorted(missing_handles)}."
        )


def selected_v2_binding_names(stages: Sequence[Any]) -> tuple[str, ...]:
    """Return every selected v2 binding name in stable config order."""

    names: list[str] = []
    for stage in stages:
        if not _uses_v2_contract(stage):
            continue
        names.extend(item.binding for item in normalize_v2_binding_selections(stage))
    return tuple(names)


def _validate_variant_semantics(stage: Any, variant: Any, *, stage_name: str) -> None:
    try:
        target = frozenset(
            Modality(item) for item in getattr(stage, "target_modalities", ())
        )
        conditioning = frozenset(
            Modality(item)
            for item in getattr(stage, "conditioning_modalities", ())
        )
        preserved = frozenset(
            Modality(item) for item in getattr(stage, "preserved_modalities", ())
        )
        coupling = ModalityCouplingPolicy(getattr(stage, "coupling_policy"))
        alignment = MediaAlignment(getattr(stage, "media_alignment"))
    except (AttributeError, TypeError, ValueError) as exc:
        raise V2ContractValidationError(
            f"stage {stage_name}: modality semantics are missing or invalid."
        ) from exc
    comparisons = (
        ("target modalities", target, variant.target_modalities),
        (
            "conditioning modalities",
            conditioning,
            variant.conditioning_modalities,
        ),
        ("preserved modalities", preserved, variant.preserved_modalities),
    )
    for label, actual, expected in comparisons:
        if actual != expected:
            raise V2ContractValidationError(
                f"stage {stage_name}: {label} {_modality_names(actual)} do not "
                f"match variant {variant.name!r} {_modality_names(expected)}."
            )
    if coupling is not variant.coupling_policy:
        raise V2ContractValidationError(
            f"stage {stage_name}: coupling_policy {coupling.value!r} does not "
            f"match variant {variant.name!r} {variant.coupling_policy.value!r}."
        )
    if alignment is not variant.media_alignment:
        raise V2ContractValidationError(
            f"stage {stage_name}: media_alignment {alignment.value!r} does not "
            f"match variant {variant.name!r} {variant.media_alignment.value!r}."
        )


def _stage_execution_phase(
    stage: Any,
    *,
    execution_phase: ExecutionPhase | str | None,
    stage_name: str,
) -> ExecutionPhase:
    try:
        stage_phase = ExecutionPhase(getattr(stage, "execution_phase"))
        requested = (
            stage_phase
            if execution_phase is None
            else ExecutionPhase(execution_phase)
        )
    except (AttributeError, TypeError, ValueError) as exc:
        raise V2ContractValidationError(
            f"stage {stage_name}: execution_phase is missing or invalid."
        ) from exc
    if stage_phase is not requested:
        raise V2ContractValidationError(
            f"stage {stage_name}: execution_phase={stage_phase.value!r}, but this "
            f"runtime requires {requested.value!r}."
        )
    return requested


def _binding_required_handles(binding: Any) -> frozenset[str]:
    declared = getattr(binding, "required_handles", None)
    if declared is not None:
        return frozenset(declared)
    return frozenset(
        handle
        for target in getattr(binding, "targets", ())
        for handle in getattr(target, "required_handles", ())
    )


def _missing_required_handles(
    required: Iterable[str],
    available: Mapping[str, Any] | Iterable[str] | None,
) -> frozenset[str]:
    if available is None:
        return frozenset()
    if isinstance(available, Mapping):
        return frozenset(
            name for name in required if not _mapping_exposes_handle(available, name)
        )
    names = frozenset(available)
    return frozenset(name for name in required if name not in names)


def _mapping_exposes_handle(handles: Mapping[str, Any], name: str) -> bool:
    if name in handles:
        return handles[name] is not None
    parts = name.split(".")
    for prefix_length in range(len(parts) - 1, 0, -1):
        prefix = ".".join(parts[:prefix_length])
        if prefix not in handles or handles[prefix] is None:
            continue
        value = handles[prefix]
        for attribute in parts[prefix_length:]:
            value = getattr(value, attribute, None)
            if value is None:
                break
        if value is not None:
            return True
    return False


def _modality_names(values: Iterable[Modality]) -> list[str]:
    return sorted(Modality(item).value for item in values)


def _uses_v2_contract(stage: Any) -> bool:
    return bool(
        getattr(stage, "uses_v2_contract", False)
        or getattr(stage, "binding", None)
        or getattr(stage, "bindings", ())
    )


__all__ = [
    "V2BindingSelection",
    "V2ContractProjection",
    "V2ContractValidationError",
    "normalize_v2_binding_selections",
    "selected_v2_binding_names",
    "validate_v2_stage_contract",
]
