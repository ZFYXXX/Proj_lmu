"""Modality- and backbone-neutral compression extension contracts."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any

from mllm_comp.core.errors import RegistryError
from mllm_comp.core.support import SupportRegistry, SupportUnit
from mllm_comp.core.vocabulary import (
    CompressionMethod,
    CompressionTokenUnit,
    ExecutionPhase,
    MediaAlignment,
    Modality,
    OperationKind,
    Placement,
    ModalityCouplingPolicy,
    TokenOperation,
)

# Researcher-facing short aliases.  The fully qualified vocabulary names avoid
# collisions with physical layout units and alignment-coupling policies.
CouplingPolicy = ModalityCouplingPolicy
TokenUnit = CompressionTokenUnit


@dataclass(frozen=True)
class AlgorithmDescriptor:
    """One algorithm implementation, independent of modality and backbone.

    ``operations`` describes the primary compression method: token transforms
    use ``TokenOperation``; computation transforms use ``OperationKind``.
    Bindings may also install cooperating mechanisms, such as a decode KV view.
    ``stage_validator`` optionally checks the full stage plugin configuration.
    Planning never constructs a kernel or discovers validation on its class;
    kernel construction belongs to the selected executor.
    """

    name: str
    version: str
    method: CompressionMethod
    kernel: Callable[..., Any]
    supported_placements: frozenset[Placement]
    operations: frozenset[TokenOperation] | frozenset[OperationKind]
    metadata: Mapping[str, Any] = field(default_factory=dict)
    stage_validator: Callable[[Mapping[str, Any]], Any] | None = None

    def __post_init__(self) -> None:
        if not self.name.strip() or not self.version.strip():
            raise RegistryError("Algorithm name and version must be non-empty.")
        object.__setattr__(self, "method", CompressionMethod(self.method))
        object.__setattr__(
            self,
            "supported_placements",
            frozenset(Placement(item) for item in self.supported_placements),
        )
        operation_type = {
            CompressionMethod.TOKEN_COMPRESSION: TokenOperation,
            CompressionMethod.OPERATION_COMPRESSION: OperationKind,
        }.get(self.method)
        if operation_type is None:
            raise RegistryError(
                f"Algorithm {self.name!r}: {self.method.value} has no implemented "
                "operation contract."
            )
        try:
            operations = frozenset(operation_type(item) for item in self.operations)
        except (TypeError, ValueError) as exc:
            raise RegistryError(
                f"Algorithm {self.name!r} operations must use "
                f"{operation_type.__name__} for {self.method.value}."
            ) from exc
        object.__setattr__(self, "operations", operations)
        if not callable(self.kernel):
            raise RegistryError(f"Algorithm {self.name!r} has no callable kernel.")
        if self.stage_validator is not None and not callable(self.stage_validator):
            raise RegistryError(f"Algorithm {self.name!r} stage_validator must be callable.")
        if not self.supported_placements:
            raise RegistryError(f"Algorithm {self.name!r} has no placement contract.")
        if not self.operations:
            raise RegistryError(f"Algorithm {self.name!r} has no compression operation.")


@dataclass(frozen=True)
class ModalityVariantDescriptor:
    """A concrete modality interpretation and its implemented Python rule hooks.

    ``supported_rules`` names hooks consumed by this variant's executor. Cloning
    the variant preserves that contract when reusing the same mechanism; a new
    strategy must declare the hooks it actually calls.
    """

    name: str
    algorithm: str
    target_modalities: frozenset[Modality]
    conditioning_modalities: frozenset[Modality] = frozenset()
    preserved_modalities: frozenset[Modality] = frozenset()
    coupling_policy: CouplingPolicy = CouplingPolicy.INDEPENDENT
    media_alignment: MediaAlignment = MediaAlignment.NONE
    token_unit: TokenUnit | None = TokenUnit.FEATURE_TOKEN
    supported_rules: frozenset[str] = frozenset()
    config_defaults: Mapping[str, Any] = field(default_factory=dict)
    config_validator: Callable[[Mapping[str, Any]], Mapping[str, Any]] | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.name.strip() or not self.algorithm.strip():
            raise RegistryError("Variant name and algorithm must be non-empty.")
        for field_name in (
            "target_modalities",
            "conditioning_modalities",
            "preserved_modalities",
        ):
            object.__setattr__(
                self,
                field_name,
                frozenset(Modality(item) for item in getattr(self, field_name)),
            )
        object.__setattr__(self, "coupling_policy", CouplingPolicy(self.coupling_policy))
        object.__setattr__(self, "media_alignment", MediaAlignment(self.media_alignment))
        if self.token_unit is not None:
            object.__setattr__(self, "token_unit", TokenUnit(self.token_unit))
        if not self.target_modalities:
            raise RegistryError(f"Variant {self.name!r} has no target modality.")
        overlap = self.target_modalities & self.preserved_modalities
        if overlap:
            raise RegistryError(
                f"Variant {self.name!r} cannot both target and preserve "
                f"{sorted(item.value for item in overlap)}."
            )
        if (
            len(self.target_modalities) > 1
            and self.coupling_policy is CouplingPolicy.INDEPENDENT
        ):
            raise RegistryError(
                f"Multi-target variant {self.name!r} must declare synchronized or "
                "joint coupling."
            )
        if isinstance(self.supported_rules, str):
            raise RegistryError(
                f"Variant {self.name!r} supported_rules must be a collection of hook names."
            )
        object.__setattr__(self, "supported_rules", frozenset(self.supported_rules))
        if any(
            not isinstance(name, str) or not name.isidentifier()
            for name in self.supported_rules
        ):
            raise RegistryError(
                f"Variant {self.name!r} supported_rules must contain Python identifiers."
            )
        if not isinstance(self.config_defaults, Mapping):
            raise RegistryError(
                f"Variant {self.name!r} config_defaults must be a mapping."
            )
        if self.config_validator is not None and not callable(
            self.config_validator
        ):
            raise RegistryError(
                f"Variant {self.name!r} config_validator must be callable."
            )

    def resolve_config(self, config: Mapping[str, Any]) -> Mapping[str, Any]:
        """Validate and normalize one variant-local algorithm configuration."""

        if not isinstance(config, Mapping):
            raise RegistryError(f"Variant {self.name!r} config must be a mapping.")
        if self.config_validator is None:
            resolved = {**dict(self.config_defaults), **dict(config)}
        else:
            resolved = self.config_validator(config)
        if not isinstance(resolved, Mapping):
            raise RegistryError(
                f"Variant {self.name!r} config validator must return a mapping."
            )
        return MappingProxyType(dict(resolved))


@dataclass(frozen=True)
class BindingTarget:
    """One physical insertion site within a logical compression stage."""

    name: str
    placement: Placement
    target_site: str
    target_modalities: frozenset[Modality]
    required_handles: frozenset[str] = frozenset()
    order: int = 0
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.name.strip() or not self.target_site.strip():
            raise RegistryError("Binding target name and target_site must be non-empty.")
        object.__setattr__(self, "placement", Placement(self.placement))
        object.__setattr__(
            self,
            "target_modalities",
            frozenset(Modality(item) for item in self.target_modalities),
        )
        object.__setattr__(self, "required_handles", frozenset(self.required_handles))
        if not self.target_modalities:
            raise RegistryError(f"Binding target {self.name!r} has no modality.")
        if any(
            not isinstance(handle, str) or not handle.strip()
            for handle in self.required_handles
        ):
            raise RegistryError(
                f"Binding target {self.name!r} required_handles must contain "
                "non-empty strings."
            )


@dataclass(frozen=True)
class BackbonePlacementBinding:
    """Bind one modality variant to physical sites on one backbone."""

    name: str
    backbone: str
    variant: str
    targets: tuple[BindingTarget, ...]
    supported_phases: frozenset[ExecutionPhase]
    executor_factory: Callable[..., Any] | None
    legalizer: str
    dependencies: tuple[str, ...] = ()
    conflicts: frozenset[str] = frozenset()
    metadata: Mapping[str, Any] = field(default_factory=dict)
    position_resolver: Callable[..., Any] | None = None
    strategy_factory: Callable[..., Any] | None = None
    executor_family: str | None = None
    canonical_executor_family: str | None = None
    exclusive_runtime_slot: str | None = None
    prepare_after_slots: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        for field_name in ("name", "backbone", "variant", "legalizer"):
            value = getattr(self, field_name)
            if not isinstance(value, str) or not value.strip():
                raise RegistryError(f"Binding {field_name} must be non-empty.")
        object.__setattr__(self, "targets", tuple(self.targets))
        object.__setattr__(
            self,
            "supported_phases",
            frozenset(ExecutionPhase(item) for item in self.supported_phases),
        )
        object.__setattr__(self, "dependencies", tuple(self.dependencies))
        object.__setattr__(self, "conflicts", frozenset(self.conflicts))
        object.__setattr__(self, "prepare_after_slots", tuple(self.prepare_after_slots))
        if not self.targets:
            raise RegistryError(f"Binding {self.name!r} has no physical target.")
        if not self.supported_phases:
            raise RegistryError(f"Binding {self.name!r} has no execution phase.")
        target_names = [item.name for item in self.targets]
        if len(target_names) != len(set(target_names)):
            raise RegistryError(f"Binding {self.name!r} has duplicate target names.")
        target_orders = [item.order for item in self.targets]
        if len(target_orders) != len(set(target_orders)):
            raise RegistryError(f"Binding {self.name!r} has duplicate target order.")
        for field_name, values in (
            ("dependencies", self.dependencies),
            ("conflicts", self.conflicts),
            ("prepare_after_slots", self.prepare_after_slots),
        ):
            if any(not isinstance(item, str) or not item.strip() for item in values):
                raise RegistryError(
                    f"Binding {self.name!r} {field_name} must contain non-empty "
                    "strings."
                )
        if self.name in self.conflicts:
            raise RegistryError(f"Binding {self.name!r} cannot conflict with itself.")
        if self.position_resolver is not None and not callable(self.position_resolver):
            raise RegistryError(f"Binding {self.name!r} position_resolver must be callable.")
        if self.strategy_factory is not None and not callable(self.strategy_factory):
            raise RegistryError(f"Binding {self.name!r} strategy_factory must be callable.")
        for field_name in ("executor_family", "canonical_executor_family", "exclusive_runtime_slot"):
            value = getattr(self, field_name)
            if value is not None and (not isinstance(value, str) or not value.strip()):
                raise RegistryError(f"Binding {self.name!r} {field_name} must be non-empty when provided.")

    @property
    def is_compound(self) -> bool:
        return len(self.targets) > 1

    @property
    def target_modalities(self) -> frozenset[Modality]:
        return frozenset(
            modality for target in self.targets for modality in target.target_modalities
        )

    def ordered_targets(self) -> tuple[BindingTarget, ...]:
        return tuple(sorted(self.targets, key=lambda item: item.order))

    @property
    def required_handles(self) -> frozenset[str]:
        """All structural handles needed by this physical binding."""

        return frozenset(
            handle for target in self.targets for handle in target.required_handles
        )


class CompressionContractRegistry:
    """Registry for v2 algorithms, variants, bindings, and exact support units."""

    def __init__(self) -> None:
        self._algorithms: dict[str, AlgorithmDescriptor] = {}
        self._variants: dict[str, ModalityVariantDescriptor] = {}
        self._bindings: dict[str, BackbonePlacementBinding] = {}
        self.support = SupportRegistry()

    def register_algorithm(self, descriptor: AlgorithmDescriptor) -> None:
        if descriptor.name in self._algorithms:
            raise RegistryError(f"Duplicate algorithm {descriptor.name!r}.")
        self._algorithms[descriptor.name] = descriptor

    def register_variant(self, descriptor: ModalityVariantDescriptor) -> None:
        if descriptor.name in self._variants:
            raise RegistryError(f"Duplicate modality variant {descriptor.name!r}.")
        if descriptor.algorithm not in self._algorithms:
            raise RegistryError(
                f"Variant {descriptor.name!r} references unknown algorithm "
                f"{descriptor.algorithm!r}."
            )
        algorithm = self._algorithms[descriptor.algorithm]
        if algorithm.method is CompressionMethod.TOKEN_COMPRESSION:
            if descriptor.token_unit is None:
                raise RegistryError(
                    f"Token variant {descriptor.name!r} must declare a token unit."
                )
        elif descriptor.token_unit is not None:
            raise RegistryError(
                f"Variant {descriptor.name!r} for {algorithm.method.value} must "
                "set token_unit=None; it does not transform token units."
            )
        self._variants[descriptor.name] = descriptor

    def register_binding(self, binding: BackbonePlacementBinding) -> None:
        if binding.name in self._bindings:
            raise RegistryError(f"Duplicate backbone binding {binding.name!r}.")
        try:
            variant = self._variants[binding.variant]
        except KeyError as exc:
            raise RegistryError(
                f"Binding {binding.name!r} references unknown variant "
                f"{binding.variant!r}."
            ) from exc
        algorithm = self._algorithms[variant.algorithm]
        unsupported_placements = {
            target.placement for target in binding.targets
        } - algorithm.supported_placements
        if unsupported_placements:
            raise RegistryError(
                f"Binding {binding.name!r} uses placements not supported by "
                f"algorithm {algorithm.name!r}: "
                f"{sorted(item.value for item in unsupported_placements)}."
            )
        if binding.target_modalities != variant.target_modalities:
            raise RegistryError(
                f"Binding {binding.name!r} targets "
                f"{sorted(item.value for item in binding.target_modalities)}, but "
                f"variant {variant.name!r} targets "
                f"{sorted(item.value for item in variant.target_modalities)}."
            )
        self._bindings[binding.name] = binding

    def register_support(self, unit: SupportUnit) -> None:
        try:
            binding = self._bindings[unit.binding]
            variant = self._variants[binding.variant]
        except KeyError as exc:
            raise RegistryError(
                f"Support unit {unit.identity.key!r} references unknown binding "
                f"{unit.binding!r}."
            ) from exc
        identity = unit.identity
        if identity.algorithm != variant.algorithm:
            raise RegistryError("Support unit algorithm does not match its binding variant.")
        if identity.variant != variant.name or identity.backbone != binding.backbone:
            raise RegistryError("Support unit variant/backbone does not match its binding.")
        if identity.target_modality not in variant.target_modalities:
            raise RegistryError("Support unit modality is not targeted by its variant.")
        if identity.phase not in binding.supported_phases:
            raise RegistryError("Support unit phase is not supported by its binding.")
        self.support.register(unit)

    def algorithm(self, name: str) -> AlgorithmDescriptor:
        try:
            return self._algorithms[name]
        except KeyError as exc:
            raise RegistryError(f"Unknown algorithm {name!r}.") from exc

    def variant(self, name: str) -> ModalityVariantDescriptor:
        try:
            return self._variants[name]
        except KeyError as exc:
            raise RegistryError(f"Unknown modality variant {name!r}.") from exc

    def binding(self, name: str) -> BackbonePlacementBinding:
        try:
            return self._bindings[name]
        except KeyError as exc:
            raise RegistryError(f"Unknown backbone binding {name!r}.") from exc

    def algorithms(self) -> tuple[AlgorithmDescriptor, ...]:
        return tuple(self._algorithms.values())

    def variants(self) -> tuple[ModalityVariantDescriptor, ...]:
        return tuple(self._variants.values())

    def bindings(self) -> tuple[BackbonePlacementBinding, ...]:
        return tuple(self._bindings.values())

    def extend_support(self, units: Iterable[SupportUnit]) -> None:
        for unit in units:
            self.register_support(unit)

    def clone(self) -> "CompressionContractRegistry":
        """Return an independent registry view over immutable descriptors.

        Support units are frozen values, while :class:`SupportRegistry` itself
        is mutable so validation code can advance candidate evidence without
        mutating the process-wide built-in registry.
        """

        clone = CompressionContractRegistry()
        for descriptor in self.algorithms():
            clone.register_algorithm(descriptor)
        for descriptor in self.variants():
            clone.register_variant(descriptor)
        for binding in self.bindings():
            clone.register_binding(binding)
        clone.extend_support(self.support.find())
        return clone


__all__ = [
    "AlgorithmDescriptor",
    "BackbonePlacementBinding",
    "BindingTarget",
    "CompressionContractRegistry",
    "CouplingPolicy",
    "MediaAlignment",
    "ModalityVariantDescriptor",
    "TokenUnit",
]
