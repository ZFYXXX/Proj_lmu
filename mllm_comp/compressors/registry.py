"""Public compression-method descriptors and registration API.

The descriptor is the single source of truth for an implemented compression
method.  Execution bridges dispatch only on ``executor_family``; algorithm
differences live behind the registered strategy object.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any, Literal

from mllm_comp.core.errors import BackendIntegrationError, RegistryError
from mllm_comp.core.vocabulary import (
    CompressionMethod,
    ExecutionPhase,
    Modality,
    ObjectFamily,
    Placement,
    TokenRole,
)


ExecutorFamily = Literal[
    "projector_token",
    "encoder_token",
    "llm_token",
    "llm_operation",
]
CapabilityLifecycleStatus = Literal["active", "target", "future"]

_STRATEGY_METHODS: dict[str, tuple[str, ...]] = {
    "projector_token": ("runtime_metadata", "compress"),
    "encoder_token": (
        "active_layer_indices",
        "runtime_metadata",
        "enabled",
        "output_lengths",
        "compress_layer",
    ),
    "llm_token": (
        "stage_layer_indices",
        "output_length",
        "score_audio_tokens",
        "select_indices",
        "runtime_metadata",
    ),
    "llm_operation": ("plan_operations", "runtime_metadata"),
}

_BACKBONE_EXECUTOR_TARGET_SITES: dict[str, dict[str, frozenset[str]]] = {
    "qwen25_omni": {
        "projector_token": frozenset(
            {
                "audio_encoder_output_pre_projector",
                "audio_projector_output_post_projector",
            }
        ),
        "encoder_token": frozenset(
            {
                "audio_encoder_layer_post_mhsa_pre_ffn",
                "audio_encoder_layer_output_post_block",
            }
        ),
        "llm_token": frozenset(
            {
                "llm_decoder_layer_output_audio_tokens",
                "llm_decoder_layer_post_attention_audio_tokens",
            }
        ),
        "llm_operation": frozenset({"llm.layers"}),
    }
}


@dataclass(frozen=True)
class CompressionMethodDescriptor:
    """Complete executable contract for one compression method.

    A method that reuses an existing executor family and target site can be
    added by implementing its plugin/spec/strategy and registering one of
    these descriptors.  Controllers, runtime bridges and integrations must
    never branch on ``name``.
    """

    name: str
    method: CompressionMethod
    placement: Placement
    target_site: str
    executor_family: ExecutorFamily
    plugin: type[Any] | Callable[[], Any]
    spec: type[Any]
    strategy: type[Any] | Callable[[Any], Any]
    supported_phases: frozenset[ExecutionPhase]
    interfaces: frozenset[str]
    required_context: frozenset[str]
    object_family: ObjectFamily
    object_subtype: str
    modalities: frozenset[Modality] = frozenset()
    token_roles: frozenset[TokenRole] = frozenset()
    backbones: frozenset[str] = frozenset({"qwen25_omni"})
    activation_phase: str = "compression_extension"
    capability_status: CapabilityLifecycleStatus = "active"
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.name:
            raise RegistryError("Compression descriptor name must be non-empty.")
        if not self.target_site:
            raise RegistryError(
                f"Compression descriptor {self.name!r} has no target_site."
            )
        if not self.supported_phases:
            raise RegistryError(
                f"Compression descriptor {self.name!r} has no supported phase."
            )
        if not self.interfaces:
            raise RegistryError(
                f"Compression descriptor {self.name!r} has no public interface."
            )
        if self.capability_status not in {"active", "target", "future"}:
            raise RegistryError(
                f"Compression descriptor {self.name!r} has invalid "
                f"capability_status={self.capability_status!r}."
            )
        if not callable(self.plugin) or not callable(self.spec) or not callable(
            self.strategy
        ):
            raise RegistryError(
                f"Compression descriptor {self.name!r} requires callable "
                "plugin, spec and strategy factories."
            )
        if not callable(getattr(self.spec, "from_config", None)):
            raise RegistryError(
                f"Compression descriptor {self.name!r} spec must expose "
                "from_config()."
            )
        family_contracts = {
            "projector_token": (
                CompressionMethod.TOKEN_COMPRESSION,
                Placement.PROJECTOR_LEVEL,
            ),
            "encoder_token": (
                CompressionMethod.TOKEN_COMPRESSION,
                Placement.ENCODER_LEVEL,
            ),
            "llm_token": (
                CompressionMethod.TOKEN_COMPRESSION,
                Placement.LLM_LEVEL,
            ),
            "llm_operation": (
                CompressionMethod.OPERATION_COMPRESSION,
                Placement.LLM_LEVEL,
            ),
        }
        expected_contract = family_contracts.get(self.executor_family)
        if expected_contract is None:
            raise RegistryError(
                f"Compression descriptor {self.name!r} has unsupported "
                f"executor_family={self.executor_family!r}."
            )
        if (self.method, self.placement) != expected_contract:
            raise RegistryError(
                f"Compression descriptor {self.name!r} executor_family="
                f"{self.executor_family!r} requires method="
                f"{expected_contract[0].value!r} and placement="
                f"{expected_contract[1].value!r}."
            )
        for backbone in self.backbones:
            supported_target_sites = _BACKBONE_EXECUTOR_TARGET_SITES.get(
                backbone, {}
            ).get(self.executor_family)
            if (
                supported_target_sites is not None
                and self.target_site not in supported_target_sites
            ):
                raise RegistryError(
                    f"Compression descriptor {self.name!r} executor_family="
                    f"{self.executor_family!r} cannot execute target_site="
                    f"{self.target_site!r} on backbone={backbone!r}; expected one "
                    f"of {sorted(supported_target_sites)}. A new target site "
                    "requires an executor-family contract change."
                )
        if self.executor_family == "llm_operation" and self.supported_phases != {
            ExecutionPhase.PREPARE
        }:
            raise RegistryError(
                f"Compression descriptor {self.name!r} LLM operations must activate "
                "only in execution_phase='prepare'."
            )

    def create_plugin(self) -> Any:
        plugin = self.plugin()
        plugin_name = getattr(plugin, "name", None)
        if plugin_name != self.name:
            raise RegistryError(
                f"Compression descriptor {self.name!r} created a plugin named "
                f"{plugin_name!r}."
            )
        missing = [
            name
            for name in ("validate_config", "plan")
            if not callable(getattr(plugin, name, None))
        ]
        if missing:
            raise RegistryError(
                f"Compression plugin for {self.name!r} is missing "
                f"{missing}."
            )
        return plugin

    def create_spec(self, config: Mapping[str, Any]) -> Any:
        factory = getattr(self.spec, "from_config", None)
        if not callable(factory):
            raise RegistryError(
                f"Compression spec {self.spec!r} for {self.name!r} does not expose "
                "from_config()."
            )
        return factory(config)

    def create_strategy(self, spec: Any) -> Any:
        strategy = self.strategy(spec)
        missing = [
            name
            for name in _STRATEGY_METHODS[self.executor_family]
            if not callable(getattr(strategy, name, None))
        ]
        if self.executor_family == "projector_token":
            missing.extend(
                name
                for name in ("score_source", "attention_layer_offset")
                if not hasattr(strategy, name)
            )
        if missing:
            raise RegistryError(
                f"Compression strategy for {self.name!r} does not implement the "
                f"{self.executor_family} contract: missing {sorted(set(missing))}."
            )
        return strategy

    def validate_stage(self, stage: Any, *, backbone: str | None = None) -> None:
        """Validate stable stage fields without algorithm-name dispatch."""

        if backbone is not None and backbone not in self.backbones:
            raise BackendIntegrationError(
                f"{self.name} does not support backbone {backbone!r}; expected one "
                f"of {sorted(self.backbones)}."
            )
        if stage.placement is not self.placement:
            raise BackendIntegrationError(
                f"{self.name} requires placement={self.placement.value}."
            )
        if stage.execution_phase not in self.supported_phases:
            raise BackendIntegrationError(
                f"{self.name} requires execution_phase in "
                f"{sorted(item.value for item in self.supported_phases)}."
            )
        target_site = stage.operation.get("target_site")
        if target_site not in {None, self.target_site}:
            raise BackendIntegrationError(
                f"{self.name} requires target_site={self.target_site!r}."
            )
        if stage.method not in {None, self.method}:
            raise BackendIntegrationError(
                f"{self.name} requires method={self.method.value}."
            )
        if stage.object_family not in {None, self.object_family}:
            raise BackendIntegrationError(
                f"{self.name} requires object_family={self.object_family.value}."
            )
        if stage.object_subtype not in {None, self.object_subtype}:
            raise BackendIntegrationError(
                f"{self.name} requires object_subtype={self.object_subtype!r}."
            )
        if stage.modalities and frozenset(stage.modalities) != self.modalities:
            raise BackendIntegrationError(
                f"{self.name} requires modalities="
                f"{sorted(item.value for item in self.modalities)}."
            )
        if stage.token_roles and frozenset(stage.token_roles) != self.token_roles:
            raise BackendIntegrationError(
                f"{self.name} requires token_roles="
                f"{sorted(item.value for item in self.token_roles)}."
            )


class CompressionMethodRegistry:
    """Mutable registry suitable for built-ins and isolated researcher tests."""

    def __init__(self, descriptors: Iterable[CompressionMethodDescriptor] = ()) -> None:
        self._descriptors: dict[str, CompressionMethodDescriptor] = {}
        for descriptor in descriptors:
            self.register(descriptor)

    def register(self, descriptor: CompressionMethodDescriptor) -> None:
        if descriptor.name in self._descriptors:
            raise RegistryError(
                f"Duplicate compression method descriptor {descriptor.name!r}."
            )
        self._descriptors[descriptor.name] = descriptor

    def get(self, name: str) -> CompressionMethodDescriptor:
        try:
            return self._descriptors[name]
        except KeyError as exc:
            raise BackendIntegrationError(
                f"Unknown compression method {name!r}; expected one of "
                f"{sorted(self._descriptors)}."
            ) from exc

    def names(self) -> frozenset[str]:
        return frozenset(self._descriptors)

    def descriptors(self) -> tuple[CompressionMethodDescriptor, ...]:
        return tuple(self._descriptors.values())

    def for_backbone(self, backbone: str) -> tuple[CompressionMethodDescriptor, ...]:
        return tuple(
            item for item in self._descriptors.values() if backbone in item.backbones
        )

    def clone(self) -> "CompressionMethodRegistry":
        return CompressionMethodRegistry(self.descriptors())


_BUILTIN_REGISTRY: CompressionMethodRegistry | None = None


def builtin_compression_methods() -> CompressionMethodRegistry:
    """Return legacy extension descriptors registered by external callers.

    Built-in multimodal methods live exclusively in the v2 contract registry.
    """

    global _BUILTIN_REGISTRY
    if _BUILTIN_REGISTRY is None:
        _BUILTIN_REGISTRY = CompressionMethodRegistry()
    return _BUILTIN_REGISTRY


def register_compression_method(descriptor: CompressionMethodDescriptor) -> None:
    """Public opt-in registration hook for external research methods."""

    builtin_compression_methods().register(descriptor)


def get_compression_method_descriptor(name: str) -> CompressionMethodDescriptor:
    return builtin_compression_methods().get(name)


__all__ = [
    "CapabilityLifecycleStatus",
    "CompressionMethodDescriptor",
    "CompressionMethodRegistry",
    "ExecutorFamily",
    "builtin_compression_methods",
    "get_compression_method_descriptor",
    "register_compression_method",
]
