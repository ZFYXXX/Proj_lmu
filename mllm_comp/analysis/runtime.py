"""Single lmms-eval sample loop shared by one or more analysis Probe sessions."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
import re
import sys
from typing import Any, Callable, Mapping

from mllm_comp.analysis.definition import AnalysisDefinition
from mllm_comp.backbones.base import BackboneFacade, BackboneRuntime
from mllm_comp.config.schema import ExperimentConfig
from mllm_comp.core.errors import BackendIntegrationError
from mllm_comp.core.manifest import AnalysisRecord, BackendRecord
from mllm_comp.core.vocabulary import ExecutionPhase, RuntimeMode


_SAFE_PROBE_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")


@dataclass
class AnalysisExecutionResult:
    records: list[AnalysisRecord]
    backend: BackendRecord
    sample_count: int
    compression_runtime_metadata: Mapping[str, Any] | None = None


def execute_analysis(
    config: ExperimentConfig,
    definition: AnalysisDefinition,
    *,
    device: str | None = None,
    runtime_loader: Callable[..., tuple[BackboneFacade, BackboneRuntime]] | None = None,
    adapter: Any | None = None,
    provider_resolver: Callable[..., Any] | None = None,
) -> AnalysisExecutionResult:
    """Execute all Probes, owning the loaded runtime until the experiment ends.

    An injected loader must return a fresh runtime for this experiment. Its
    hooks and model references are released on both success and failure.
    """

    probe_specs = _analysis_probe_specs(
        definition.probes,
        backbone_name=config.backbone.name,
    )
    _require_fresh_output_paths(config)
    if runtime_loader is None:
        from mllm_comp.runtime import load_backbone_runtime

        runtime_loader = load_backbone_runtime
    if provider_resolver is None:
        from mllm_comp.registry.analyses import get_analysis

        provider_resolver = get_analysis
    backend_name = config.analysis.backend or "lmms_eval"
    if adapter is None:
        from mllm_comp.registry.integrations import get_integration

        adapter = get_integration(backend_name)()

    backbone, runtime = runtime_loader(
        config,
        device=device,
        train=False,
        compression_mode=definition.compression_mode,
        runtime_mode=RuntimeMode.DEFAULT,
        execution_phase=_analysis_compression_execution_phase(
            config,
            definition,
        ),
    )
    sessions: list[Any] = []
    requires_teacher_forcing = False
    sample_count = 0
    try:
        for probe_config, execution_spec in zip(definition.probes, probe_specs):
            logical_name = execution_spec["name"]
            method = execution_spec["method"]
            provider = provider_resolver(method)
            plugin = provider() if isinstance(provider, type) else provider
            requires_teacher_forcing |= bool(getattr(plugin, "requires_teacher_forcing", False))
            opener = getattr(plugin, "open", None)
            if not callable(opener):
                raise BackendIntegrationError(
                    f"Analysis Probe {method!r} does not implement the session lifecycle."
                )
            effective_probe_config = dict(probe_config)
            effective_probe_config.update(
                {
                    "_capability_status": execution_spec["capability_status"],
                    "_supported_backbones": execution_spec[
                        "supported_backbones"
                    ],
                }
            )
            sessions.append(
                opener(
                    backbone=backbone,
                    runtime=runtime,
                    config=effective_probe_config,
                    phase=str(definition.phase),
                    output_dir=str(
                        Path(config.output.run_dir) / "analysis" / logical_name
                    ),
                )
            )

        sample_iter = adapter.iter_analysis_samples(
            config=config,
            backbone=backbone,
            runtime=runtime,
            sample_limit=definition.sample_limit,
            requires_teacher_forcing=requires_teacher_forcing,
        )
        for sample in sample_iter:
            sample_count += 1
            for session in sessions:
                session.process(sample)
        records = [session.finish() for session in sessions]
        from mllm_comp.pipeline.runtime_session import (
            refresh_runtime_compression_metadata,
        )

        # Teardown can clear activation histories; retain independent evidence.
        compression_metadata = deepcopy(refresh_runtime_compression_metadata(runtime))
    finally:
        active_error = sys.exc_info()[0] is not None
        cleanup_errors: list[str] = []
        for session in reversed(sessions):
            close = getattr(session, "close", None)
            if not callable(close):
                continue
            try:
                close()
            except Exception as exc:
                cleanup_errors.append(f"{type(exc).__name__}: {exc}")
        try:
            runtime.close()
        except Exception as exc:
            cleanup_errors.append(f"{type(exc).__name__}: {exc}")
        if cleanup_errors and not active_error:
            raise BackendIntegrationError(
                "Analysis session cleanup failed: " + "; ".join(cleanup_errors)
            )

    report = adapter.dependency_report() if hasattr(adapter, "dependency_report") else {}
    versions = report.get("versions", {}) if isinstance(report, Mapping) else {}
    version = "unknown"
    if isinstance(versions, Mapping):
        for key in ("lmms-eval", "lmms_eval", backend_name):
            if versions.get(key):
                version = str(versions[key])
                break
    return AnalysisExecutionResult(
        records=records,
        backend=BackendRecord(
            name=backend_name,
            version=version,
            args={
                "mode": "analysis",
                "sample_count": sample_count,
                "single_dataset_iteration": True,
                "probe_capabilities": probe_specs,
            },
        ),
        sample_count=sample_count,
        compression_runtime_metadata=compression_metadata,
    )


def _analysis_probe_specs(
    probes: list[Mapping[str, Any]],
    *,
    backbone_name: str,
) -> list[dict[str, Any]]:
    """Validate artifact identities before loading a model or opening sessions."""

    from mllm_comp.core.errors import RegistryError
    from mllm_comp.registry.catalog import (
        CapabilityKind,
        CapabilityStatus,
        get_capability_spec,
    )

    result: list[dict[str, Any]] = []
    names: set[str] = set()
    for probe in probes:
        raw_name = probe.get("name")
        raw_method = probe.get("method")
        if raw_name is not None and (
            not isinstance(raw_name, str) or not raw_name
        ):
            raise BackendIntegrationError(
                "Analysis Probe name must be a non-empty string."
            )
        if raw_method is not None and (
            not isinstance(raw_method, str) or not raw_method
        ):
            raise BackendIntegrationError(
                "Analysis Probe method must be a non-empty string."
            )
        logical_name = raw_name or raw_method
        if not logical_name or not _SAFE_PROBE_NAME.fullmatch(logical_name):
            raise BackendIntegrationError(
                "Analysis Probe name must be one safe path component, got "
                f"{logical_name!r}."
            )
        if logical_name in names:
            raise BackendIntegrationError(
                f"Analysis Probe logical name {logical_name!r} is duplicated."
            )
        names.add(logical_name)
        method = raw_method or logical_name
        capability_status = "external"
        supported_backbones: list[str] | None = None
        try:
            catalog_spec = get_capability_spec(CapabilityKind.ANALYSIS, method)
        except RegistryError:
            catalog_spec = None
        if catalog_spec is not None:
            capability_status = catalog_spec.status.value
            if catalog_spec.status is not CapabilityStatus.ACTIVE:
                raise BackendIntegrationError(
                    f"Analysis Probe {logical_name!r} is not active; catalog status="
                    f"{capability_status!r}."
                )
            declared = catalog_spec.metadata.get("backbones")
            if declared is not None:
                supported_backbones = sorted(str(value) for value in declared)
                if backbone_name not in supported_backbones:
                    raise BackendIntegrationError(
                        f"Analysis Probe {logical_name!r} does not declare support "
                        f"for backbone {backbone_name!r}."
                    )
        result.append(
            {
                "name": logical_name,
                "method": method,
                "capability_status": capability_status,
                "supported_backbones": supported_backbones,
            }
        )
    if not result:
        raise BackendIntegrationError(
            "Analysis execution requires at least one Probe."
        )
    return result


def _require_fresh_output_paths(config: ExperimentConfig) -> None:
    """Reject stale analysis outputs before loading a checkpoint."""

    manifest_path = Path(config.output.manifest_path)
    analysis_root = Path(config.output.run_dir) / "analysis"
    collisions: list[str] = []
    if manifest_path.exists():
        collisions.append(str(manifest_path))
    if analysis_root.exists():
        if not analysis_root.is_dir() or any(analysis_root.iterdir()):
            collisions.append(str(analysis_root))
    if collisions:
        raise BackendIntegrationError(
            "Analysis requires fresh output paths so a failed rerun cannot mix "
            "artifacts with an older manifest. Choose a new --output_dir/--manifest "
            "or explicitly remove the stale outputs: "
            + ", ".join(collisions)
        )


def _analysis_compression_execution_phase(
    config: ExperimentConfig,
    definition: AnalysisDefinition,
) -> ExecutionPhase:
    """Analysis requests are forward passes; prepare stages install beforehand."""

    if str(definition.compression_mode) == "none":
        return ExecutionPhase.FORWARD
    phases = {
        stage.execution_phase
        for stage in config.pipeline.stages
        if stage.enabled
    }
    if phases - {ExecutionPhase.FORWARD, ExecutionPhase.PREPARE}:
        raise BackendIntegrationError(
            "Analysis probes execute model forward passes, so compression stages "
            "must use execution_phase=forward or prepare."
        )
    return ExecutionPhase.FORWARD


__all__ = ["AnalysisExecutionResult", "execute_analysis"]
