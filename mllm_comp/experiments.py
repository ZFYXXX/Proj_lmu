"""Run one configured experiment from Python or the command line."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from datetime import datetime, timezone
import logging
from pathlib import Path
from typing import Any, Literal
from uuid import uuid4

from mllm_comp.analysis.definition import AnalysisOverrides, resolve_analysis_definition
from mllm_comp.analysis.runtime import execute_analysis
from mllm_comp.config import ExperimentConfig, config_to_dict
from mllm_comp.config.validators import validate_runnable_config
from mllm_comp.core.errors import MLLMCompError
from mllm_comp.core.serialization import atomic_write_json, to_jsonable
from mllm_comp.core.vocabulary import AnalysisPhase, CompressionMode, EvalTarget
from mllm_comp.integrations.native_cli import NativeCLIProcessError
from mllm_comp.pipeline.definition import resolve_compression_definition
from mllm_comp.registry.integrations import get_integration
from mllm_comp.restoration import restore_config
from mllm_comp.run_manifest import build_run_manifest


ExperimentTask = Literal["eval", "train", "analyze"]
ExperimentStatus = Literal["planned", "running", "succeeded", "failed"]


@dataclass(frozen=True)
class ExperimentPaths:
    """Absolute artifact locations resolved once in the caller's launch cwd."""

    run_dir: str
    effective_config: str
    result: str
    manifest: str


@dataclass(frozen=True)
class ExperimentError:
    type: str
    stage: str
    message: str


@dataclass
class ExperimentResult:
    """One machine-readable outcome; native logs keep their original streams.

    ``returncode`` is the CLI exit code. ``native_returncode`` retains the
    upstream process code, including negative POSIX signal codes. Planned and
    running experiments have no exit code yet. ``output`` holds task-specific
    metrics or the launch plan, without changing upstream metric semantics.
    """

    task: ExperimentTask
    status: ExperimentStatus = "planned"
    run_id: str | None = None
    paths: ExperimentPaths | None = None
    returncode: int | None = None
    native_returncode: int | None = None
    output: dict[str, Any] = field(default_factory=dict)
    error: ExperimentError | None = None
    schema_version: int = 1

    def to_dict(self) -> dict[str, Any]:
        return to_jsonable(self)

    def record_failure(self, error: BaseException, *, stage: str) -> None:
        self.status = "failed"
        self.returncode = 2 if isinstance(error, (MLLMCompError, OSError, ValueError)) else 1
        if isinstance(error, NativeCLIProcessError):
            self.native_returncode = error.returncode
            self.returncode = error.shell_returncode
        elif isinstance(error, KeyboardInterrupt):
            self.returncode = 130
        self.error = ExperimentError(type(error).__name__, stage, str(error))


def run_experiment(
    config: ExperimentConfig,
    *,
    task: ExperimentTask,
    device: str | None = None,
    analysis_overrides: AnalysisOverrides | None = None,
    plan_only: bool = False,
) -> ExperimentResult:
    """Resolve, execute and record one independent experiment.

    The caller's config is never mutated. An explicit ``output.run_dir`` is
    exact; otherwise a fresh directory is generated below ``runs/<name>``.
    Recorded paths and the effective output config use absolute locations.
    Existing nonempty run directories and existing manifests are rejected.
    Planning is read-only. Expected configuration/integration failures return
    a failed result; unexpected exceptions are recorded and re-raised.
    """

    result = ExperimentResult(task=task)
    stage = "configuration"
    prepared = False
    try:
        if task not in {"eval", "train", "analyze"}:
            raise ValueError(f"Unknown experiment task: {task!r}.")
        if task != "analyze" and analysis_overrides is not None:
            raise ValueError("analysis_overrides only applies to task='analyze'.")
        effective = deepcopy(config)
        if device is None:
            device = effective.backbone.config.get("device")
        if device is not None:
            effective.backbone.config["device"] = device
        definition = None
        if task == "analyze":
            effective.analysis.backend = effective.analysis.backend or "lmms_eval"
            definition = resolve_analysis_definition(
                effective, analysis_overrides or AnalysisOverrides()
            )
            effective.analysis.compression_mode = CompressionMode(
                definition.compression_mode
            )
            effective.analysis.phase = AnalysisPhase(definition.phase)
            effective.analysis.tasks = list(definition.tasks)
            effective.analysis.sample_limit = definition.sample_limit
            effective.analysis.runtime_batch_size = definition.runtime_batch_size
            effective.analysis.probes = deepcopy(definition.probes)
        else:
            section = getattr(effective, task)
            section.compression_mode = CompressionMode(section.compression_mode)
            section.backend = section.backend or (
                "lmms_eval" if task == "eval" else "ms_swift"
            )
            if task == "eval" and effective.eval.target is None:
                effective.eval.target = (
                    EvalTarget.COMPRESSED
                    if effective.eval.compression_mode is CompressionMode.COMPRESSED
                    else EvalTarget.BASELINE
                )
        validate_runnable_config(effective, task=task)
        section = getattr(effective, "analysis" if task == "analyze" else task)
        compression_definition = resolve_compression_definition(
            effective,
            compression_mode=section.compression_mode,
            interface="analysis" if task == "analyze" else task,
            runtime_batch_size=(section.runtime_batch_size if task != "train" else None),
        )
        result.paths = _resolve_output_paths(effective)
        result.run_id = effective.experiment.run_id
        stage = "prepare_outputs"
        _check_fresh_outputs(result.paths)

        if plan_only:
            stage = "plan"
            if task == "analyze":
                result.output = {"definition": definition}
            else:
                adapter = get_integration(getattr(effective, task).backend)()
                plan = (
                    adapter.build_launch_plan(
                        effective,
                        checkpoint_path=effective.eval.checkpoint_path,
                        device=device,
                    )
                    if task == "eval" else adapter.build_launch_plan(effective)
                )
                result.output = {
                    "plan": plan,
                    "versions": adapter.dependency_report().get("versions", {}),
                }
            result.output["effective_config"] = config_to_dict(effective)
            return result

        Path(result.paths.run_dir).mkdir(parents=True, exist_ok=True)
        prepared = True
        atomic_write_json(result.paths.effective_config, config_to_dict(effective))
        result.status = "running"
        atomic_write_json(result.paths.result, result)
        stage = "execute"
        if task == "analyze":
            execution = execute_analysis(effective, definition, device=device)
            result.output = {
                "analysis_records": execution.records,
                "sample_count": execution.sample_count,
            }
            manifest_args = {
                "eval_backend": execution.backend,
                "analysis_records": execution.records,
                "checkpoint_path": (
                    effective.backbone.model_path or effective.backbone.model_id
                ),
                "invariants_passed": analysis_invariants_passed(execution),
                "invariants_run": [
                    "AnalysisSemanticBinding",
                    "AnalysisTokenGroupLayout",
                    "AnalysisHookCleanup",
                ],
                "compression_runtime_metadata": execution.compression_runtime_metadata,
            }
        elif task == "eval":
            adapter = get_integration(effective.eval.backend)()
            execution = adapter.evaluate(
                config=effective,
                checkpoint_path=effective.eval.checkpoint_path,
                device=device,
                compressed=effective.eval.compression_mode is CompressionMode.COMPRESSED,
            )
            result.native_returncode = 0
            result.output = {
                "backend": execution.backend,
                "metrics": execution.metrics,
                "raw_output_path": execution.raw_output_path,
            }
            manifest_args = {
                "eval_backend": execution.backend,
                "metrics": execution.metrics,
                "checkpoint_path": (
                    effective.eval.checkpoint_path
                    or effective.backbone.model_path
                    or effective.backbone.model_id
                ),
                "invariants_passed": True,
                "invariants_run": _native_validator_invariants(
                    effective.backbone.name,
                    forward_only_probe=bool(
                        effective.eval.args.get("forward_only_probe", False)
                    ),
                ),
            }
        else:
            adapter = get_integration(effective.train.backend)()
            execution = adapter.train(config=effective)
            result.native_returncode = 0
            result.output = {
                "backend": execution.backend,
                "checkpoint_path": execution.checkpoint_path,
                "train_log_path": execution.train_log_path,
                "metrics": execution.metrics,
            }
            manifest_args = {
                "train_backend": execution.backend,
                "checkpoint_path": execution.checkpoint_path,
                "invariants_passed": True,
                "invariants_run": _training_boundary_invariants(
                    effective.backbone.name,
                    compression_plugin=(
                        next(
                            (s.plugin for s in effective.pipeline.stages if s.enabled),
                            None,
                        )
                        if effective.train.compression_mode is CompressionMode.COMPRESSED
                        else None
                    ),
                ),
            }
        stage = "write_artifacts"
        if Path(result.paths.manifest).exists():
            raise FileExistsError(
                "Refusing to overwrite an existing experiment artifact with the Manifest: "
                f"{result.paths.manifest}"
            )
        manifest = build_run_manifest(
            effective, compression_definition=compression_definition, **manifest_args
        )
        atomic_write_json(result.paths.manifest, manifest)
        result.status = "succeeded"
        result.returncode = 0
        atomic_write_json(result.paths.result, result)
        return result
    except (Exception, KeyboardInterrupt) as exc:
        result.record_failure(exc, stage=stage)
        if prepared:
            try:
                atomic_write_json(result.paths.result, result)
            except Exception:
                logging.getLogger(__name__).exception("Could not record experiment failure.")
        if isinstance(exc, (MLLMCompError, OSError, ValueError)):
            return result
        raise


def _resolve_output_paths(config: ExperimentConfig) -> ExperimentPaths:
    if config.experiment.run_id is None:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        config.experiment.run_id = f"{stamp}-{uuid4().hex[:8]}"
    if config.output.run_dir is None:
        config.output.run_dir = str(
            Path("runs") / config.experiment.name / config.experiment.run_id
        )
    run_dir = Path(config.output.run_dir).resolve()
    manifest = (
        run_dir / "manifest.json" if config.output.manifest_path is None
        else Path(config.output.manifest_path).resolve()
    )
    config.output.run_dir = str(run_dir)
    config.output.manifest_path = str(manifest)
    return ExperimentPaths(
        run_dir=config.output.run_dir,
        effective_config=str(run_dir / "effective_config.json"),
        result=str(run_dir / "result.json"),
        manifest=config.output.manifest_path,
    )


def _check_fresh_outputs(paths: ExperimentPaths) -> None:
    run_dir = Path(paths.run_dir)
    if run_dir.exists() and (not run_dir.is_dir() or any(run_dir.iterdir())):
        raise ValueError(
            f"Experiment run directory must be new or empty: {run_dir}. "
            "Choose a new output.run_dir / --output_dir, or omit output.run_dir "
            "and experiment.run_id to generate a new run directory."
        )
    manifest = Path(paths.manifest)
    if manifest.exists():
        raise ValueError(
            f"Experiment manifest already exists: {manifest}. "
            "Choose a new output.manifest_path / --manifest, or omit it to use "
            "the new run directory's manifest.json."
        )
    resolved_run_dir = run_dir.resolve()
    reserved_files = {
        Path(paths.effective_config).resolve(), Path(paths.result).resolve()
    }
    resolved_manifest = manifest.resolve()
    if (
        resolved_manifest == resolved_run_dir
        or resolved_manifest in resolved_run_dir.parents
        or resolved_manifest in reserved_files
        or reserved_files.intersection(resolved_manifest.parents)
    ):
        raise ValueError(
            "output.manifest_path must not conflict with the run directory or config/result files."
        )


_NATIVE_VALIDATOR_NAMES = {
    "llava_onevision": "LlavaOneVisionNativeValidator",
    "qwen25_omni": "Qwen25OmniNativeValidator",
    "qwen25_vl": "Qwen25VLNativeValidator",
    "qwen3_vl": "Qwen3VLNativeValidator",
    "qwen3_5": "Qwen3_5NativeValidator",
}


def _native_validator_invariants(
    backbone_name: str, *, forward_only_probe: bool = False,
) -> list[str]:
    try:
        validator_name = _NATIVE_VALIDATOR_NAMES[backbone_name]
    except KeyError as exc:
        supported = ", ".join(sorted(_NATIVE_VALIDATOR_NAMES))
        raise ValueError(
            f"No eval Manifest validator mapping for backbone {backbone_name!r}; "
            f"supported backbones: {supported}."
        ) from exc
    if forward_only_probe:
        return [
            f"{validator_name}.validate_state",
            "LMMSEvalGenerateUntilTransport",
            "ForwardOnlyCompressionProbe",
        ]
    return [
        f"{validator_name}.validate_state",
        f"{validator_name}.validate_generation",
        "LMMSEvalModelProtocol",
    ]


_TRAINING_BOUNDARY_NAMES = {
    "qwen25_omni": "ThinkerOnlyTrainingBoundary",
    "qwen25_vl": "Qwen25VLLanguageModelTrainingBoundary",
    "qwen3_vl": "Qwen3VLLanguageModelTrainingBoundary",
    "llava_onevision": "LlavaOneVisionLanguageModelTrainingBoundary",
}


def _training_boundary_invariants(
    backbone_name: str, *, compression_plugin: str | None = None,
) -> list[str]:
    try:
        boundary_name = _TRAINING_BOUNDARY_NAMES[backbone_name]
    except KeyError as exc:
        supported = ", ".join(sorted(_TRAINING_BOUNDARY_NAMES))
        raise ValueError(
            f"No train Manifest boundary mapping for backbone {backbone_name!r}; "
            f"supported backbones: {supported}."
        ) from exc
    result = ["MSSwiftSftArguments", boundary_name]
    if compression_plugin is not None:
        result.extend([
            "DynamicCompressionTrainingForward",
            f"CompressionPlugin:{compression_plugin}",
            "CompressionTrainingBackwardEvidence",
        ])
    return result


def analysis_invariants_passed(result: object) -> bool:
    """Return true only when every Probe accepted every iterated sample."""

    records = getattr(result, "records", None)
    sample_count = getattr(result, "sample_count", None)
    if not isinstance(records, list) or not records:
        return False
    if isinstance(sample_count, bool) or not isinstance(sample_count, int):
        return False
    for record in records:
        if record.sample_count != sample_count:
            return False
        skipped = record.summary.get("skipped_samples")
        if isinstance(skipped, bool) or not isinstance(skipped, int) or skipped != 0:
            return False
    return True
