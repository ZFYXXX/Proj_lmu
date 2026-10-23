"""Transactional activation of one or more runtime compression stages.

The session is deliberately independent of any model provider.  A backbone
binding resolves a stage into a :class:`CompressionActivationRecord`; the
session then owns ordering and lifecycle.  This keeps controller hooks out of
the generic pipeline while still making multi-stage activation auditable.

The native entry point is ``activate_runtime_compression``. Its validated
definition reaches ``RuntimeCompressionDispatcher.resolve`` here, which creates
records without constructing v2 executors. ``CompressionSession.install`` then
builds each executor in dependency order, after earlier prepare-time changes.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from mllm_comp.compressors.rules import legacy_rule_specs, recipe_changed
from mllm_comp.core.errors import BackendIntegrationError, ConfigError, RegistryError
from mllm_comp.core.vocabulary import (
    CompressionMode,
    ExecutionPhase,
)
from mllm_comp.pipeline.v2_validation import (
    V2ContractProjection,
    V2ContractValidationError,
    selected_v2_binding_names,
    validate_v2_stage_contract,
    validate_v2_runtime_context,
)

if TYPE_CHECKING:
    from mllm_comp.pipeline.definition import CompressionDefinition


@runtime_checkable
class CompressionActivation(Protocol):
    """Small lifecycle implemented by a dynamic or prepare-time executor."""

    def install(self) -> None:
        """Activate hooks or apply the requested prepare-time transform."""

    def uninstall(self) -> None:
        """Remove installed hooks, or perform the executor's best rollback."""

    def runtime_metadata(self) -> Mapping[str, Any]:
        """Return lightweight, serializable diagnostics for this activation."""


@dataclass
class CompressionActivationRecord:
    """Identity, dependencies, and lifecycle state for one activation.

    ``rollback_safe`` is evidence rather than a switch: the session always
    calls ``uninstall`` during rollback, but reports when an executor (for
    example an in-place structural rewrite) cannot promise full restoration.
    A binding may supply ``activation_factory`` instead of an eager executor;
    the session calls it in install order, after earlier structure changes.
    """

    name: str
    activation: CompressionActivation | None
    binding: str
    dependencies: tuple[str, ...] = ()
    config_order: int = 0
    rollback_safe: bool = True
    metadata: dict[str, Any] = field(default_factory=dict)
    activation_factory: Callable[[], CompressionActivation] | None = field(
        default=None, repr=False
    )
    _status: str = field(default="resolved", init=False, repr=False)
    _install_attempted: bool = field(default=False, init=False, repr=False)
    _installed: bool = field(default=False, init=False, repr=False)
    _error: str | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise BackendIntegrationError("Compression activation name is empty.")
        if not isinstance(self.binding, str) or not self.binding.strip():
            raise BackendIntegrationError(
                f"Compression activation {self.name!r} has no binding identity."
            )
        if not isinstance(self.config_order, int) or isinstance(
            self.config_order, bool
        ):
            raise BackendIntegrationError(
                f"Compression activation {self.name!r} has a non-integer order."
            )
        dependencies = tuple(self.dependencies)
        if any(not isinstance(item, str) or not item for item in dependencies):
            raise BackendIntegrationError(
                f"Compression activation {self.name!r} has an empty dependency."
            )
        if self.name in dependencies:
            raise BackendIntegrationError(
                f"Compression activation {self.name!r} depends on itself."
            )
        self.dependencies = dependencies
        if self.activation is None:
            if not callable(self.activation_factory):
                raise BackendIntegrationError(
                    f"Compression activation {self.name!r} has no executor or factory."
                )
        else:
            self._validate_activation()

    def _validate_activation(self) -> None:
        for method_name in ("install", "uninstall", "runtime_metadata"):
            if not callable(getattr(self.activation, method_name, None)):
                raise BackendIntegrationError(
                    f"Compression activation {self.name!r} does not implement "
                    f"{method_name}()."
                )

    @property
    def installed(self) -> bool:
        return self._installed

    @property
    def status(self) -> str:
        return self._status

    def install(self) -> None:
        """Install once; a failed attempt remains eligible for cleanup."""

        if self._installed:
            return
        self._status = "installing"
        self._install_attempted = True
        self._error = None
        try:
            if self.activation is None:
                # Previous prepare steps may have replaced modules. Resolve
                # physical references only now, before this executor installs.
                self.activation = self.activation_factory()
                self._validate_activation()
                self.activation_factory = None
                self.rollback_safe = bool(getattr(self.activation, "rollback_safe", True))
            self.activation.install()
        except BaseException as exc:
            self._status = "install_failed"
            self._error = _exception_summary(exc)
            raise
        self._installed = True
        self._status = "active"

    def uninstall(self, *, force: bool = False) -> None:
        """Uninstall an active (or possibly partially installed) executor."""

        if not self._installed and not (force and self._install_attempted):
            return
        self._status = "uninstalling"
        try:
            if self.activation is not None:
                self.activation.uninstall()
        except BaseException as exc:
            self._status = "uninstall_failed"
            self._error = _exception_summary(exc)
            raise
        self._installed = False
        self._install_attempted = False
        self._status = "uninstalled"

    def runtime_metadata(self) -> dict[str, Any]:
        metadata_error: str | None = None
        try:
            activation_metadata = (
                {} if self.activation is None else dict(self.activation.runtime_metadata())
            )
        except Exception as exc:  # Metadata must never hide a lifecycle failure.
            activation_metadata = {}
            metadata_error = _exception_summary(exc)
        result = {
            **dict(self.metadata),
            **activation_metadata,
            "name": self.name,
            "binding": self.binding,
            "dependencies": list(self.dependencies),
            "config_order": self.config_order,
            "rollback_safe": self.rollback_safe,
            "status": self._status,
            "installed": self._installed,
        }
        if self._error is not None:
            result["lifecycle_error"] = self._error
        if metadata_error is not None:
            result["metadata_error"] = metadata_error
        return result


class CompressionSession:
    """Transactional owner for an ordered set of compression activations."""

    def __init__(
        self,
        activations: Iterable[CompressionActivationRecord],
        *,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        self.activations = tuple(activations)
        self.metadata = dict(metadata or {})
        self._ordered = _dependency_order(self.activations)
        self._installed_order: list[CompressionActivationRecord] = []
        self._status = "resolved"
        self._rollback_errors: list[dict[str, str]] = []
        self._uninstall_errors: list[dict[str, str]] = []

    @property
    def status(self) -> str:
        return self._status

    @property
    def install_order(self) -> tuple[str, ...]:
        return tuple(item.name for item in self._ordered)

    @property
    def installed(self) -> bool:
        return self._status == "active"

    @property
    def active_records(self) -> tuple[CompressionActivationRecord, ...]:
        """Return installed activations in dependency-resolved execution order."""

        return tuple(self._installed_order)

    def install(self) -> "CompressionSession":
        """Install all activations or best-effort rollback the whole attempt."""

        if self._status == "active":
            return self
        if self._status == "installing":
            raise BackendIntegrationError("Compression session is already installing.")
        self._status = "installing"
        self._rollback_errors.clear()
        self._uninstall_errors.clear()
        self._installed_order.clear()
        for record in self._ordered:
            try:
                record.install()
            except BaseException as exc:
                rollback_records = (*self._installed_order, record)
                self._rollback_after_failure(record)
                incomplete = any(
                    not item.rollback_safe for item in rollback_records
                )
                self._status = (
                    "rollback_incomplete"
                    if incomplete or self._rollback_errors
                    else "rolled_back"
                )
                detail = (
                    f"; rollback diagnostics={self._rollback_errors!r}"
                    if self._rollback_errors
                    else ""
                )
                raise BackendIntegrationError(
                    f"Compression activation {record.name!r} failed: "
                    f"{_exception_summary(exc)}{detail}"
                ) from exc
            self._installed_order.append(record)
        self._status = "active"
        return self

    def uninstall(self) -> None:
        """Uninstall in exact reverse install order, continuing after errors."""

        if not self._installed_order:
            if self._status == "active":
                self._status = "uninstalled"
            return
        self._status = "uninstalling"
        self._uninstall_errors.clear()
        for record in reversed(self._installed_order):
            try:
                record.uninstall(force=True)
            except BaseException as exc:
                self._uninstall_errors.append(
                    {"name": record.name, "error": _exception_summary(exc)}
                )
        self._installed_order.clear()
        if self._uninstall_errors:
            self._status = "uninstall_failed"
            raise BackendIntegrationError(
                "One or more compression activations failed to uninstall: "
                f"{self._uninstall_errors!r}."
            )
        self._status = "uninstalled"

    def clear_diagnostic_records(self) -> None:
        """Begin a request, retaining lifecycle, counters and computational state."""

        for record in self._installed_order:
            clear = getattr(record.activation, "clear_diagnostic_records", None)
            if callable(clear):
                clear()

    def runtime_metadata(self) -> dict[str, Any]:
        """Return session- and activation-level lifecycle diagnostics."""

        activation_metadata = [item.runtime_metadata() for item in self._ordered]
        result = {
            **self.metadata,
            "enabled": bool(self.activations),
            "status": self._status,
            "activation_count": len(self.activations),
            "configured_order": [item.name for item in self.activations],
            "install_order": list(self.install_order),
            "active_activations": [
                item.name for item in self._installed_order if item.installed
            ],
            "rollback_errors": list(self._rollback_errors),
            "uninstall_errors": list(self._uninstall_errors),
            "activations": activation_metadata,
            "layout_delta_summary": _session_layout_delta_summary(
                activation_metadata
            ),
        }
        return result

    def _rollback_after_failure(
        self,
        failed: CompressionActivationRecord,
    ) -> None:
        rollback_order = [failed, *reversed(self._installed_order)]
        for record in rollback_order:
            try:
                record.uninstall(force=True)
            except BaseException as exc:
                self._rollback_errors.append(
                    {"name": record.name, "error": _exception_summary(exc)}
                )
        self._installed_order.clear()

    def __enter__(self) -> "CompressionSession":
        return self.install()

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        del exc_type, exc, traceback
        self.uninstall()


class RuntimeCompressionDispatcher:
    """Resolve v2 bindings or a backbone's explicit legacy compatibility layer."""

    def __init__(
        self,
        *,
        contract_registry: Any | None = None,
        legacy_bindings: Mapping[str, Any] | None = None,
    ) -> None:
        self.contract_registry = contract_registry
        self._legacy_bindings = dict(legacy_bindings or {})

    def resolve(
        self,
        *,
        config: Any,
        facade: Any,
        runtime: Any,
        compression_mode: CompressionMode | str | None,
        execution_phase: ExecutionPhase | str = ExecutionPhase.GENERATE,
        definition: CompressionDefinition | None = None,
    ) -> CompressionSession | None:
        """Resolve enabled config stages without mutating the model."""

        mode = CompressionMode(compression_mode or CompressionMode.NONE)
        if mode is CompressionMode.NONE:
            return None
        stages = (
            list(definition.stages)
            if definition is not None
            else [stage for stage in config.pipeline.stages if stage.enabled]
        )
        if not stages:
            raise BackendIntegrationError(
                "compression_mode=compressed requires an enabled pipeline stage."
            )
        phase = ExecutionPhase(execution_phase)
        from mllm_comp.fixed_mask.binding import validate_stages
        validate_stages(stages)
        backbone_name = facade.capabilities().name
        try:
            all_binding_names = selected_v2_binding_names(stages)
        except V2ContractValidationError as exc:
            raise BackendIntegrationError(str(exc)) from exc
        prepare_stages = tuple(
            stage.name for stage in stages
            if stage.execution_phase == ExecutionPhase.PREPARE
        )
        available_handles = (
            _runtime_structural_handles(facade, runtime)
            if all_binding_names and not prepare_stages
            else None
        )
        projections: dict[str, V2ContractProjection] = {}
        for stage in stages:
            if not _uses_v2_contract(stage):
                continue
            if self.contract_registry is None:
                raise BackendIntegrationError(
                    f"Stage {stage.name!r} selects a v2 binding, but no compression "
                    "contract registry was provided."
                )
            stage_phase = (
                ExecutionPhase.PREPARE
                if stage.execution_phase == ExecutionPhase.PREPARE else phase
            )
            try:
                if definition is None:
                    projections[stage.name] = validate_v2_stage_contract(
                        stage,
                        registry=self.contract_registry,
                        backbone_name=backbone_name,
                        execution_phase=stage_phase,
                        selected_binding_names=all_binding_names,
                        selected_stages=stages,
                        available_handles=available_handles,
                    )
                else:
                    projection = definition.projections[stage.name]
                    validate_v2_runtime_context(
                        stage,
                        projection,
                        execution_phase=stage_phase,
                        available_handles=available_handles,
                    )
                    projections[stage.name] = projection
            except V2ContractValidationError as exc:
                raise BackendIntegrationError(str(exc)) from exc

        records: list[CompressionActivationRecord] = []
        records_by_stage: dict[str, list[CompressionActivationRecord]] = {}
        binding_to_records: dict[str, list[CompressionActivationRecord]] = {}

        for config_order, stage in enumerate(stages):
            if _uses_v2_contract(stage):
                stage_records = self._resolve_v2_stage(
                    stage=stage,
                    facade=facade,
                    runtime=runtime,
                    execution_phase=(
                        ExecutionPhase.PREPARE
                        if stage.execution_phase == ExecutionPhase.PREPARE else phase
                    ),
                    config_order=config_order,
                    projection=projections[stage.name],
                    stage_projections=projections,
                )
            else:
                try:
                    legacy_rule_specs(stage)
                except ConfigError as exc:
                    raise BackendIntegrationError(str(exc)) from exc
                try:
                    legacy_binding = self._legacy_bindings[backbone_name]
                except KeyError as exc:
                    raise BackendIntegrationError(
                        f"Backbone {backbone_name!r} requires a selected compression binding; "
                        "external legacy execution requires an explicitly injected binding."
                    ) from exc
                stage_records = _as_records(
                    legacy_binding.resolve(
                        stage=stage,
                        facade=facade,
                        runtime=runtime,
                        execution_phase=phase,
                        config_order=config_order,
                    )
                )
            if stage.execution_phase != ExecutionPhase.PREPARE:
                for record in stage_records:
                    record.dependencies += prepare_stages
            records.extend(stage_records)
            records_by_stage[stage.name] = list(stage_records)
            for record in stage_records:
                binding_to_records.setdefault(record.binding, []).append(record)

        _resolve_dependency_aliases(
            records,
            records_by_stage=records_by_stage,
            binding_to_records=binding_to_records,
        )
        return CompressionSession(
            records,
            metadata={
                "mode": mode.value,
                "backbone": backbone_name,
                "execution_phase": phase.value,
            },
        )

    def activate(self, **kwargs: Any) -> CompressionSession | None:
        """Resolve, install, and publish one session on the runtime object."""

        runtime = kwargs["runtime"]
        candidate_stages = (kwargs["definition"].stages if kwargs.get("definition") is not None
                            else kwargs["config"].pipeline.stages)
        image_operations = any(
            getattr(stage, "enabled", True)
            and getattr(stage, "method", None) == "operation_compression"
            and getattr(stage, "modalities", set()) == {"image"}
            for stage in candidate_stages)
        if (CompressionMode(kwargs.get("compression_mode") or "none") is CompressionMode.COMPRESSED
                and image_operations):
            from mllm_comp.modality_split.pipeline import compile_pipeline_split, activate_pipeline_split
            split_plan = compile_pipeline_split(
                kwargs["config"], backbone_name=kwargs["facade"].capabilities().name,
                definition=kwargs.get("definition"))
            if split_plan is not None:
                return activate_pipeline_split(runtime, split_plan)
        session = self.resolve(**kwargs)
        if session is None:
            return None
        runtime.metadata["_compression_session"] = session
        try:
            session.install()
        finally:
            refresh_runtime_compression_metadata(runtime)
        return session

    def _resolve_v2_stage(
        self,
        *,
        stage: Any,
        facade: Any,
        runtime: Any,
        execution_phase: ExecutionPhase,
        config_order: int,
        projection: V2ContractProjection,
        stage_projections: Mapping[str, V2ContractProjection],
    ) -> tuple[CompressionActivationRecord, ...]:
        registry = self.contract_registry
        records: list[CompressionActivationRecord] = []
        for selection_order, (selection, binding) in enumerate(
            zip(projection.selections, projection.bindings)
        ):
            support_status = _binding_support_status(
                stage=stage,
                binding=binding,
                registry=registry,
                execution_phase=execution_phase,
            )
            factory = binding.executor_factory
            if not callable(factory):
                raise BackendIntegrationError(
                    f"Compression binding {binding.name!r} has no executor factory."
                )
            def create_activation(
                *, factory=factory, binding=binding, selection=selection,
            ) -> CompressionActivation:
                try:
                    validate_v2_runtime_context(
                        stage,
                        projection,
                        execution_phase=execution_phase,
                        available_handles=_runtime_structural_handles(
                            facade, runtime, refresh=True
                        ),
                    )
                except V2ContractValidationError as exc:
                    raise BackendIntegrationError(str(exc)) from exc
                return factory(
                    model=runtime.model,
                    runtime=runtime,
                    facade=facade,
                    stage=stage,
                    binding=binding,
                    binding_config=selection,
                    execution_phase=execution_phase,
                    contract_projection=projection,
                )
            generated_name = (
                stage.name
                if len(projection.selections) == 1
                else f"{stage.name}:{binding.name}"
            )
            raw_dependencies = tuple(stage.depends_on) + tuple(
                getattr(selection, "depends_on", ())
            ) + tuple(binding.dependencies)
            after_slots = binding.prepare_after_slots
            if after_slots:
                for other_name, other_projection in stage_projections.items():
                    if other_name != stage.name and any(
                        candidate.exclusive_runtime_slot in after_slots
                        for candidate in other_projection.bindings
                    ):
                        raw_dependencies += (other_name,)
            record = CompressionActivationRecord(
                name=generated_name,
                activation=None,
                activation_factory=create_activation,
                binding=binding.name,
                dependencies=raw_dependencies,
                config_order=(config_order * 1000) + selection_order,
                # The executor reports its actual rollback contract when built.
                rollback_safe=False,
                metadata={
                    "stage_name": stage.name,
                    "algorithm": stage.algorithm,
                    "variant": stage.variant,
                    "executor_family": "v2_binding",
                    "execution_phase": execution_phase.value,
                    "support_status": support_status,
                    "development_runtime": support_status != "active",
                },
            )
            if recipe_changed(binding):
                record.metadata.update(
                    canonical_evidence_applicable=False,
                    resolved_target_sites={
                        target.name: target.target_site for target in binding.targets
                    },
                )
                if binding.metadata.get("rules_changed"):
                    record.metadata.update(
                        rules=binding.metadata["rules"], rules_changed=True,
                    )
                application_positions = {
                    target.name: target.metadata["application_positions"]
                    for target in binding.targets
                    if "application_positions" in target.metadata
                }
                if application_positions:
                    record.metadata["application_positions"] = application_positions
            records.append(record)
        return tuple(records)


def _uses_v2_contract(stage: Any) -> bool:
    return bool(
        getattr(stage, "uses_v2_contract", False)
        or getattr(stage, "binding", None)
        or getattr(stage, "bindings", ())
    )


def _runtime_structural_handles(
    facade: Any,
    runtime: Any,
    *,
    refresh: bool = False,
) -> Mapping[str, Any] | None:
    """Read structural handles when the active facade exposes that contract."""

    model_adapter = getattr(facade, "model_adapter", None)
    inspect_handles = getattr(model_adapter, "structural_handles", None)
    metadata = getattr(runtime, "metadata", {})
    if isinstance(metadata, Mapping):
        published = metadata.get("structural_handles")
        if published is not None and not (refresh and callable(inspect_handles)):
            if not isinstance(published, Mapping):
                raise BackendIntegrationError(
                    "runtime.metadata['structural_handles'] must be a mapping."
                )
            return published
    if not callable(inspect_handles):
        return None
    try:
        handles = inspect_handles(runtime.model)
    except BaseException as exc:
        raise BackendIntegrationError(
            "Backbone structural handle inspection failed before compression "
            "activation."
        ) from exc
    if not isinstance(handles, Mapping):
        raise BackendIntegrationError(
            "Backbone model_adapter.structural_handles() must return a mapping."
        )
    return handles


def _binding_support_status(
    *,
    stage: Any,
    binding: Any,
    registry: Any,
    execution_phase: ExecutionPhase,
) -> str:
    """Report exact verification status without gating a legal research run."""

    from mllm_comp.core.support import SupportStatus, SupportUnitId

    if recipe_changed(binding):
        return SupportStatus.DECLARED.value

    statuses: set[str] = set()
    for modality in binding.target_modalities:
        identity = SupportUnitId(
            algorithm=stage.algorithm,
            variant=stage.variant,
            backbone=binding.backbone,
            target_modality=modality,
            phase=execution_phase,
        )
        try:
            unit = registry.support.get(identity)
        except RegistryError:
            statuses.add("undeclared")
            continue
        # Evidence from an alternate physical binding does not cover this run.
        statuses.add(unit.status.value if unit.binding == binding.name else "undeclared")
    return next(iter(statuses)) if len(statuses) == 1 else "mixed"



def _as_records(resolved: Any) -> tuple[CompressionActivationRecord, ...]:
    """Flatten legacy binding results; each record already owns its identity."""

    if isinstance(resolved, CompressionActivationRecord):
        return (resolved,)
    if isinstance(resolved, Sequence) and not isinstance(
        resolved, (str, bytes, bytearray)
    ):
        return tuple(record for item in resolved for record in _as_records(item))
    raise BackendIntegrationError(
        "A compression binding returned a bare activation without identity."
    )


def _resolve_dependency_aliases(
    records: list[CompressionActivationRecord],
    *,
    records_by_stage: Mapping[str, list[CompressionActivationRecord]],
    binding_to_records: Mapping[str, list[CompressionActivationRecord]],
) -> None:
    exact_names = {item.name for item in records}
    for record in records:
        resolved: list[str] = []
        for dependency in record.dependencies:
            if dependency in exact_names:
                targets = [dependency]
            elif dependency in records_by_stage:
                targets = [item.name for item in records_by_stage[dependency]]
            elif dependency in binding_to_records:
                candidates = binding_to_records[dependency]
                if len(candidates) != 1:
                    raise BackendIntegrationError(
                        f"Compression dependency {dependency!r} is ambiguous across "
                        f"{[item.name for item in candidates]}."
                    )
                targets = [candidates[0].name]
            else:
                targets = [dependency]
            for target in targets:
                if target != record.name and target not in resolved:
                    resolved.append(target)
        record.dependencies = tuple(resolved)


def _dependency_order(
    activations: tuple[CompressionActivationRecord, ...],
) -> tuple[CompressionActivationRecord, ...]:
    by_name: dict[str, CompressionActivationRecord] = {}
    input_order: dict[str, int] = {}
    for index, activation in enumerate(activations):
        if activation.name in by_name:
            raise BackendIntegrationError(
                f"Duplicate compression activation name {activation.name!r}."
            )
        by_name[activation.name] = activation
        input_order[activation.name] = index
    missing = {
        dependency
        for activation in activations
        for dependency in activation.dependencies
        if dependency not in by_name
    }
    if missing:
        raise BackendIntegrationError(
            "Compression activation dependencies are missing: "
            f"{sorted(missing)}."
        )
    remaining = {
        activation.name: set(activation.dependencies) for activation in activations
    }
    ordered: list[CompressionActivationRecord] = []
    while remaining:
        ready = [
            by_name[name]
            for name, dependencies in remaining.items()
            if not dependencies
        ]
        if not ready:
            cycle = sorted(remaining)
            raise BackendIntegrationError(
                f"Compression activation dependency cycle: {cycle}."
            )
        ready.sort(
            key=lambda item: (
                item.config_order,
                input_order[item.name],
                item.name,
            )
        )
        for activation in ready:
            ordered.append(activation)
            del remaining[activation.name]
            for dependencies in remaining.values():
                dependencies.discard(activation.name)
    return tuple(ordered)


def _exception_summary(exc: BaseException) -> str:
    return f"{type(exc).__name__}: {exc}"


def _session_layout_delta_summary(
    activations: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Project exact activation records into a compact session audit view."""

    records: list[dict[str, Any]] = []
    omitted = {
        "old_to_new",
        "dropped_indices",
        "merge_groups",
        "protected_indices",
    }
    for activation in activations:
        raw_records = activation.get("layout_delta_records", ())
        if not isinstance(raw_records, Sequence) or isinstance(
            raw_records, (str, bytes, bytearray)
        ):
            continue
        for raw in raw_records:
            if not isinstance(raw, Mapping):
                continue
            compact = {key: value for key, value in raw.items() if key not in omitted}
            compact.update(
                {
                    "activation": activation.get("name"),
                    "binding": activation.get("binding"),
                    "stage_name": activation.get("stage_name"),
                }
            )
            records.append(compact)
    return {
        "record_count": len(records),
        "records": records,
    }


def refresh_runtime_compression_metadata(runtime: Any) -> dict[str, Any] | None:
    """Refresh dynamic callback evidence published on a BackboneRuntime.

    Activations mutate their counters and layout history during model calls.
    Publishing only at install time would leave runtime manifests permanently
    empty, so evidence writers call this helper immediately before serialization.
    """

    metadata = getattr(runtime, "metadata", None)
    if not isinstance(metadata, dict):
        return None
    session = metadata.get("_compression_session")
    from mllm_comp.modality_split.pipeline import PipelineSplitSession
    if not isinstance(session, (CompressionSession, PipelineSplitSession)):
        value = metadata.get("compression")
        return dict(value) if isinstance(value, Mapping) else None
    value = session.runtime_metadata()
    metadata["compression"] = value
    return value


__all__ = [
    "CompressionActivation",
    "CompressionActivationRecord",
    "CompressionSession",
    "RuntimeCompressionDispatcher",
    "refresh_runtime_compression_metadata",
]
