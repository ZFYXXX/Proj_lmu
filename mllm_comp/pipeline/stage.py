"""Typed compression-stage configuration.

YAML stages are normalized into this dataclass exactly once by the config
loader. Runtime and definition validation consume the same representation.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, TypeVar

from mllm_comp.core.errors import ConfigError
from mllm_comp.core.vocabulary import (
    CompressionMethod,
    ModalityCouplingPolicy,
    ExecutionPhase,
    MediaAlignment,
    Modality,
    ObjectFamily,
    Placement,
    StringEnum,
    TokenRole,
    enum_values,
)


_STAGE_FIELDS = {
    "name",
    "placement",
    "execution_phase",
    "plugin",
    "enabled",
    "method",
    "object_family",
    "object_subtype",
    "modalities",
    "algorithm",
    "variant",
    "binding",
    "target_modalities",
    "conditioning_modalities",
    "preserved_modalities",
    "coupling_policy",
    "media_alignment",
    "bindings",
    "token_roles",
    "depends_on",
    "operation",
    "criterion",
    "schedule",
    "quantization",
    "activation_scope",
    "calibration",
    "config",
}

_EnumT = TypeVar("_EnumT", bound=StringEnum)


_BINDING_FIELDS = {
    "binding",
    "target_modalities",
    "depends_on",
    "order",
    "config",
}


@dataclass
class StageBindingConfig:
    """One physical binding selected inside a logical compound stage."""

    binding: str
    target_modalities: set[Modality] = field(default_factory=set)
    depends_on: list[str] = field(default_factory=list)
    order: int = 0
    config: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.binding, str) or not self.binding.strip():
            raise ConfigError("pipeline stage binding must be a non-empty string.")
        self.target_modalities = _coerce_enum_set(
            Modality,
            self.target_modalities,
            "bindings.target_modalities",
        )
        if not isinstance(self.order, int) or isinstance(self.order, bool):
            raise ConfigError("pipeline stage binding order must be an integer.")
        if any(not isinstance(item, str) or not item for item in self.depends_on):
            raise ConfigError(
                "pipeline stage binding depends_on must contain non-empty strings."
            )

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "StageBindingConfig":
        unknown = sorted(set(raw) - _BINDING_FIELDS)
        if unknown:
            raise ConfigError(f"Unknown compound binding fields: {unknown}")
        return cls(
            binding=raw.get("binding", ""),
            target_modalities=_set_field(raw, "target_modalities"),
            depends_on=_list_field(raw, "depends_on"),
            order=raw.get("order", 0),
            config=_dict_field(raw, "config"),
        )


@dataclass
class CompressionStageConfig:
    """One normalized experiment request, before a plugin creates a plan."""

    name: str
    placement: Placement
    execution_phase: ExecutionPhase
    plugin: str
    enabled: bool = True
    method: CompressionMethod | None = None
    object_family: ObjectFamily | None = None
    object_subtype: str | None = None
    modalities: set[Modality] = field(default_factory=set)
    algorithm: str | None = None
    variant: str | None = None
    binding: str | None = None
    target_modalities: set[Modality] = field(default_factory=set)
    conditioning_modalities: set[Modality] = field(default_factory=set)
    preserved_modalities: set[Modality] = field(default_factory=set)
    coupling_policy: ModalityCouplingPolicy = ModalityCouplingPolicy.INDEPENDENT
    media_alignment: MediaAlignment = MediaAlignment.NONE
    bindings: list[StageBindingConfig] = field(default_factory=list)
    token_roles: set[TokenRole] = field(default_factory=set)
    depends_on: list[str] = field(default_factory=list)
    operation: dict[str, Any] = field(default_factory=dict)
    criterion: dict[str, Any] = field(default_factory=dict)
    schedule: dict[str, Any] = field(default_factory=dict)
    quantization: dict[str, Any] = field(default_factory=dict)
    activation_scope: dict[str, Any] = field(default_factory=dict)
    calibration: dict[str, Any] = field(default_factory=dict)
    config: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.placement = _coerce_enum(Placement, self.placement, "placement")
        self.execution_phase = _coerce_enum(
            ExecutionPhase,
            self.execution_phase,
            "execution_phase",
        )
        self.method = _coerce_optional_enum(CompressionMethod, self.method, "method")
        self.object_family = _coerce_optional_enum(
            ObjectFamily,
            self.object_family,
            "object_family",
        )
        self.modalities = _coerce_enum_set(Modality, self.modalities, "modalities")
        self.target_modalities = _coerce_enum_set(
            Modality,
            self.target_modalities,
            "target_modalities",
        )
        self.conditioning_modalities = _coerce_enum_set(
            Modality,
            self.conditioning_modalities,
            "conditioning_modalities",
        )
        self.preserved_modalities = _coerce_enum_set(
            Modality,
            self.preserved_modalities,
            "preserved_modalities",
        )
        self.coupling_policy = _coerce_enum(
            ModalityCouplingPolicy,
            self.coupling_policy,
            "coupling_policy",
        )
        self.media_alignment = _coerce_enum(
            MediaAlignment,
            self.media_alignment,
            "media_alignment",
        )
        self.token_roles = _coerce_enum_set(TokenRole, self.token_roles, "token_roles")

        if self.modalities and self.target_modalities:
            if self.modalities != self.target_modalities:
                raise ConfigError(
                    f"stage {self.name}: legacy modalities and target_modalities "
                    "must match when both are provided."
                )
        elif self.modalities:
            self.target_modalities = set(self.modalities)
        elif self.target_modalities:
            self.modalities = set(self.target_modalities)

        if not isinstance(self.name, str) or not self.name:
            raise ConfigError("pipeline stage field 'name' must be a non-empty string.")
        if not isinstance(self.plugin, str) or not self.plugin:
            raise ConfigError("pipeline stage field 'plugin' must be a non-empty string.")
        if not isinstance(self.enabled, bool):
            raise ConfigError(f"stage {self.name}: enabled must be a bool.")
        if self.object_subtype is not None and not isinstance(self.object_subtype, str):
            raise ConfigError(f"stage {self.name}: object_subtype must be a string when provided.")
        for field_name in ("algorithm", "variant", "binding"):
            value = getattr(self, field_name)
            if value is not None and (not isinstance(value, str) or not value.strip()):
                raise ConfigError(
                    f"stage {self.name}: {field_name} must be a non-empty string."
                )
        if (self.algorithm is None) != (self.variant is None):
            raise ConfigError(
                f"stage {self.name}: algorithm and variant must be provided together."
            )
        if self.binding is not None and self.bindings:
            raise ConfigError(
                f"stage {self.name}: use either binding or bindings, not both."
            )
        overlap = self.target_modalities & self.preserved_modalities
        if overlap:
            raise ConfigError(
                f"stage {self.name}: target_modalities and preserved_modalities "
                f"overlap at {sorted(item.value for item in overlap)}."
            )
        if (
            self.uses_v2_contract
            and len(self.target_modalities) > 1
            and self.coupling_policy is ModalityCouplingPolicy.INDEPENDENT
        ):
            raise ConfigError(
                f"stage {self.name}: multiple target modalities require "
                "coupling_policy=synchronized or joint."
            )
        binding_names = [item.binding for item in self.bindings]
        binding_orders = [item.order for item in self.bindings]
        if len(binding_names) != len(set(binding_names)):
            raise ConfigError(f"stage {self.name}: duplicate compound binding name.")
        if len(binding_orders) != len(set(binding_orders)):
            raise ConfigError(f"stage {self.name}: duplicate compound binding order.")

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "CompressionStageConfig":
        """Normalize one YAML mapping and reject misspelled stable fields."""

        unknown = sorted(set(raw) - _STAGE_FIELDS)
        if unknown:
            raise ConfigError(f"Unknown pipeline stage fields: {unknown}")

        enabled = raw.get("enabled", True)
        if not isinstance(enabled, bool):
            raise ConfigError("pipeline stage field 'enabled' must be a bool.")

        return cls(
            name=raw.get("name", ""),
            placement=raw.get("placement", ""),
            execution_phase=raw.get("execution_phase", ""),
            plugin=raw.get("plugin", ""),
            enabled=enabled,
            method=raw.get("method"),
            object_family=raw.get("object_family"),
            object_subtype=raw.get("object_subtype"),
            modalities=_set_field(raw, "modalities"),
            algorithm=raw.get("algorithm"),
            variant=raw.get("variant"),
            binding=raw.get("binding"),
            target_modalities=_set_field(raw, "target_modalities"),
            conditioning_modalities=_set_field(raw, "conditioning_modalities"),
            preserved_modalities=_set_field(raw, "preserved_modalities"),
            coupling_policy=raw.get(
                "coupling_policy",
                ModalityCouplingPolicy.INDEPENDENT,
            ),
            media_alignment=raw.get("media_alignment", MediaAlignment.NONE),
            bindings=_binding_list_field(raw),
            token_roles=_set_field(raw, "token_roles"),
            depends_on=_list_field(raw, "depends_on"),
            operation=_dict_field(raw, "operation"),
            criterion=_dict_field(raw, "criterion"),
            schedule=_dict_field(raw, "schedule"),
            quantization=_dict_field(raw, "quantization"),
            activation_scope=_dict_field(raw, "activation_scope"),
            calibration=_dict_field(raw, "calibration"),
            config=_dict_field(raw, "config"),
        )

    def plugin_config(self) -> dict[str, Any]:
        """Return the algorithm-facing portion of this stage request."""

        result = {
            "operation": dict(self.operation),
            "criterion": dict(self.criterion),
            "schedule": dict(self.schedule),
            "activation_scope": dict(self.activation_scope),
            "config": dict(self.config),
        }
        if self.uses_v2_contract:
            result["modality_contract"] = {
                "algorithm": self.algorithm,
                "variant": self.variant,
                "binding": self.binding,
                "target_modalities": sorted(
                    item.value for item in self.target_modalities
                ),
                "conditioning_modalities": sorted(
                    item.value for item in self.conditioning_modalities
                ),
                "preserved_modalities": sorted(
                    item.value for item in self.preserved_modalities
                ),
                "coupling_policy": self.coupling_policy.value,
                "media_alignment": self.media_alignment.value,
                "bindings": [
                    {
                        "binding": item.binding,
                        "target_modalities": sorted(
                            modality.value for modality in item.target_modalities
                        ),
                        "depends_on": list(item.depends_on),
                        "order": item.order,
                        "config": dict(item.config),
                    }
                    for item in sorted(self.bindings, key=lambda value: value.order)
                ],
            }
        return result

    @property
    def uses_v2_contract(self) -> bool:
        return self.algorithm is not None


def _coerce_enum(enum_type: type[_EnumT], value: Any, field_name: str) -> _EnumT:
    try:
        return enum_type(value)
    except (TypeError, ValueError) as exc:
        raise ConfigError(
            f"pipeline stage field {field_name!r} must be one of "
            f"{list(enum_values(enum_type))}, got {value!r}."
        ) from exc


def _coerce_optional_enum(
    enum_type: type[_EnumT],
    value: Any,
    field_name: str,
) -> _EnumT | None:
    return None if value is None else _coerce_enum(enum_type, value, field_name)


def _coerce_enum_set(
    enum_type: type[_EnumT],
    values: set[Any],
    field_name: str,
) -> set[_EnumT]:
    return {_coerce_enum(enum_type, value, field_name) for value in values}


def _set_field(raw: Mapping[str, Any], field_name: str) -> set[Any]:
    value = raw.get(field_name, [])
    if not isinstance(value, (list, tuple, set)):
        raise ConfigError(f"pipeline stage field {field_name!r} must be a list.")
    return set(value)


def _list_field(raw: Mapping[str, Any], field_name: str) -> list[Any]:
    value = raw.get(field_name, [])
    if not isinstance(value, list):
        raise ConfigError(f"pipeline stage field {field_name!r} must be a list.")
    return list(value)


def _dict_field(raw: Mapping[str, Any], field_name: str) -> dict[str, Any]:
    value = raw.get(field_name, {})
    if not isinstance(value, Mapping):
        raise ConfigError(f"pipeline stage field {field_name!r} must be a mapping.")
    return dict(value)


def _binding_list_field(raw: Mapping[str, Any]) -> list[StageBindingConfig]:
    value = raw.get("bindings", [])
    if not isinstance(value, list) or any(not isinstance(item, Mapping) for item in value):
        raise ConfigError("pipeline stage field 'bindings' must be a list of mappings.")
    return [StageBindingConfig.from_dict(item) for item in value]
