"""Load explicitly named research registrations into an independent registry."""

from __future__ import annotations

from collections.abc import Callable, Collection, Mapping, Sequence
from dataclasses import replace
from functools import partial
from importlib import import_module
from typing import Any

from mllm_comp.compressors.contracts import AlgorithmDescriptor, CompressionContractRegistry
from mllm_comp.compressors.multimodal_registry import builtin_multimodal_compression_contracts
from mllm_comp.core.errors import ConfigError, RegistryError


def register_method(
    registry: CompressionContractRegistry,
    *,
    algorithm: AlgorithmDescriptor | str,
    variant: str | None = None,
    reuse_bindings: Sequence[str],
    strategy_factory: Callable[..., Any] | None = None,
    executor_factory: Callable[..., Any] | None = None,
    config_defaults: Mapping[str, Any] | None = None,
    config_validator: Callable[..., Any] | None = None,
    supported_rules: Collection[str] = (),
    conditioning_modalities: Collection[str] = (),
    position_resolver: Callable[..., Any] | None = None,
    target_name: str | None = None,
) -> None:
    """Register a new algorithm on explicitly selected existing mechanisms.

    This constructs ordinary Algorithm/Variant/Binding descriptors. A string
    algorithm name reuses the selected method's plan and configuration spec;
    supply a compatible strategy that retains those budget/operation semantics.
    A descriptor supplies a different plan, configuration or kernel explicitly.
    The variant name defaults to the algorithm name and target modalities.
    Authors provide either a strategy for the existing executor or an explicit executor
    factory; a new kernel alone does not establish protocol compatibility.
    Each native executor still validates and calls its own method interface.

    Modality layout, target sites, handles, phases and lifecycle dependencies
    come from the selected bindings. Method configuration, conditioning, rules
    and position resolvers belong to the new method and default to empty/None.
    Reusing a plan retains its spec and stage validator, but never its support
    evidence, paper metadata or capability claims. Strategy rules and method
    defaults must still be declared by their actual consumer.
    """
    if isinstance(reuse_bindings, str) or not reuse_bindings:
        raise RegistryError("register_method requires explicit reuse_bindings names.")
    sources = tuple(registry.binding(name) for name in reuse_bindings)
    if len({source.variant for source in sources}) != 1:
        raise RegistryError("Reused bindings must share one modality variant.")
    if len({source.backbone for source in sources}) != len(sources):
        raise RegistryError("Select one existing binding per backbone.")
    if (strategy_factory is None) == (executor_factory is None):
        raise RegistryError("Provide either strategy_factory or executor_factory explicitly.")
    if not callable(strategy_factory if strategy_factory is not None else executor_factory):
        raise RegistryError("The selected method factory must be callable.")
    if target_name is not None and any(len(source.targets) != 1 for source in sources):
        raise RegistryError("target_name renames one target; omit it for compound bindings.")
    source_variant = registry.variant(sources[0].variant)
    if isinstance(algorithm, str):
        source_algorithm = registry.algorithm(source_variant.algorithm)
        spec = source_algorithm.metadata.get("spec_factory")
        if (strategy_factory is None
                or not isinstance(spec, type)
                or not callable(getattr(spec, "from_config", None))):
            raise RegistryError(
                f"Algorithm {algorithm!r}: {source_algorithm.name!r} does not expose "
                "a reusable plan/spec with a strategy. Supply an AlgorithmDescriptor "
                "for a different execution interface."
            )
        algorithm = AlgorithmDescriptor(
            name=algorithm, version="1", method=source_algorithm.method,
            kernel=partial(_create_named_plan, source_algorithm.kernel, algorithm),
            supported_placements=source_algorithm.supported_placements,
            operations=source_algorithm.operations,
            metadata={"spec_factory": spec},
            stage_validator=source_algorithm.stage_validator,
        )
    if variant is None:
        modalities = ".".join(sorted(item.value for item in source_variant.target_modalities))
        variant = f"{algorithm.name}.{modalities}"
    registered_variant = replace(
        source_variant, name=variant, algorithm=algorithm.name,
        config_defaults=dict(config_defaults or {}), config_validator=config_validator,
        supported_rules=frozenset(supported_rules),
        conditioning_modalities=frozenset(conditioning_modalities), metadata={},
    )
    bindings = []
    for source in sources:
        if strategy_factory is not None and (
            source.strategy_factory is None or not callable(source.executor_factory)
        ):
            raise RegistryError(f"Binding {source.name!r} has no replaceable strategy executor.")
        conflicts = set(source.conflicts) | {source.name}
        # Include earlier research methods derived from conflicting mechanisms.
        # The ordinary stage validator checks either direction of a conflict.
        for existing in registry.bindings():
            if (existing.metadata.get("reused_binding") in conflicts
                    or source.name in existing.conflicts):
                conflicts.add(existing.name)
        conflicts.discard(f"{variant}.{source.backbone}")
        targets = tuple(replace(
            target, name=target_name if target_name is not None else target.name,
            metadata={key: value for key, value in target.metadata.items()
                      if key in {"component_role", "media_boundary"}},
        ) for target in source.targets)
        # Runtime requirements remain relevant to the reused physical mechanism.
        # Canonical configs, paper identity and evidence paths do not transfer.
        metadata = {key: source.metadata[key]
                    for key in ("required_runtime_context", "binding_config_keys")
                    if key in source.metadata}
        bindings.append(replace(
            source, name=f"{variant}.{source.backbone}", variant=variant, targets=targets,
            strategy_factory=strategy_factory,
            executor_factory=executor_factory if executor_factory is not None else source.executor_factory,
            position_resolver=position_resolver, conflicts=frozenset(conflicts),
            metadata={**metadata, "reused_binding": source.name},
        ))
    registry.register_algorithm(algorithm)
    registry.register_variant(registered_variant)
    for binding in bindings:
        registry.register_binding(binding)


def _create_named_plan(factory: Callable[..., Any], name: str) -> Any:
    """Give a reused plan its experiment-local identity without a rename-only subclass."""
    plugin = factory()
    plugin.name = name
    return plugin


def compression_contracts(extensions: Sequence[str] = ()) -> CompressionContractRegistry:
    """Call each ``package.module:register`` with this experiment's registry.

    Registration functions add descriptors to the supplied registry. They must
    be importable in every native worker and must not modify global registries.
    """

    builtin = builtin_multimodal_compression_contracts()
    if not extensions:
        return builtin
    registry = builtin.clone()
    for entry in extensions:
        module_name, separator, function_name = entry.partition(":")
        if not separator or not module_name or not function_name.isidentifier():
            raise ConfigError(
                f"Invalid pipeline.extensions entry {entry!r}; expected package.module:register."
            )
        try:
            register = getattr(import_module(module_name), function_name)
        except (ImportError, AttributeError) as exc:
            raise ConfigError(f"Cannot load compression extension {entry!r}: {exc}") from exc
        if not callable(register):
            raise ConfigError(f"Compression extension {entry!r} is not callable.")
        try:
            register(registry)
        except (RegistryError, TypeError, ValueError) as exc:
            raise ConfigError(f"Cannot register compression extension {entry!r}: {exc}") from exc
    return registry


__all__ = ["compression_contracts", "register_method"]
