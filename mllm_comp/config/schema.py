"""Typed experiment configuration and serialization."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field, fields
from typing import Any, TypeVar

from mllm_comp.core.errors import ConfigError
from mllm_comp.core.serialization import to_jsonable
from mllm_comp.core.vocabulary import (
    AnalysisPhase,
    CompressionMode,
    DistributedLauncher,
    EvalTarget,
    RuntimeMode,
    StringEnum,
    enum_values,
)
from mllm_comp.pipeline.stage import CompressionStageConfig
from .stage_completion import complete_stage_requests


@dataclass
class ExperimentSection:
    name: str
    run_id: str | None = None
    seed: int = 42


@dataclass
class BackboneSection:
    name: str
    model_path: str | None = None
    model_id: str | None = None
    model_revision: str | None = None
    source_path: str | None = None
    trust_remote_code: bool | None = None
    config: dict[str, Any] = field(default_factory=dict)


@dataclass
class PipelineSection:
    stages: list[CompressionStageConfig] = field(default_factory=list)
    extensions: list[str] = field(default_factory=list)


@dataclass
class AnalysisSection:
    backend: str | None = None
    compression_mode: CompressionMode = CompressionMode.NONE
    phase: AnalysisPhase = AnalysisPhase.PRE_COMPRESSION
    runtime_batch_size: int = 1
    tasks: list[str] = field(default_factory=list)
    sample_limit: int | None = None
    task_include_path: list[str] = field(default_factory=list)
    streaming_datasets: bool = False
    sampling: dict[str, Any] = field(default_factory=dict)
    probes: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class TrainSection:
    backend: str | None = None
    compression_mode: CompressionMode = CompressionMode.NONE
    runtime_mode: RuntimeMode = RuntimeMode.DEFAULT
    script_path: str | None = None
    args_path: str | None = None
    checkpoint_in: str | None = None
    resume: str | None = None
    max_steps: int | None = None
    args: dict[str, Any] = field(default_factory=dict)


@dataclass
class EvalSection:
    backend: str | None = None
    compression_mode: CompressionMode = CompressionMode.NONE
    tasks: list[str] = field(default_factory=list)
    runtime_batch_size: int = 1
    num_processes: int = 1
    distributed_launcher: DistributedLauncher = DistributedLauncher.ACCELERATE
    main_process_port: int | None = None
    sample_limit: int | None = None
    checkpoint_path: str | None = None
    target: EvalTarget | None = None
    runtime_mode: RuntimeMode = RuntimeMode.DEFAULT
    generation: dict[str, Any] = field(default_factory=dict)
    args: dict[str, Any] = field(default_factory=dict)


@dataclass
class DataSection:
    dataset_id: str | None = None
    dataset_commit: str | None = None
    sample_split: str | None = None


@dataclass
class OutputSection:
    run_dir: str | None = None
    manifest_path: str | None = None


@dataclass
class ExperimentConfig:
    """Structured top-level experiment config with the original raw mapping kept."""

    experiment: ExperimentSection
    backbone: BackboneSection
    pipeline: PipelineSection
    analysis: AnalysisSection
    train: TrainSection
    eval: EvalSection
    data: DataSection
    output: OutputSection
    raw: dict[str, Any]
    path: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


def config_to_dict(config: ExperimentConfig) -> dict[str, Any]:
    """Return an independent, JSON-compatible config accepted by ``load_config``.

    Source text, source path and transient runtime metadata are provenance, not
    executable configuration. Serialize only the typed configuration sections.
    """

    return {
        item.name: to_jsonable(getattr(config, item.name))
        for item in fields(config)
        if item.name not in {"raw", "path", "metadata"}
    }


def experiment_config_from_dict(raw: dict[str, Any], *, path: str | None = None) -> ExperimentConfig:
    if not isinstance(raw, dict):
        raise ConfigError("Experiment config root must be a mapping.")

    top_level_fields = {
        "experiment",
        "backbone",
        "pipeline",
        "analysis",
        "train",
        "eval",
        "data",
        "output",
    }
    unknown_top_level = sorted(set(raw) - top_level_fields)
    if unknown_top_level:
        raise ConfigError(f"Unknown top-level config fields: {unknown_top_level}")

    experiment_raw = _section(raw, "experiment", ExperimentSection)
    backbone_raw = _section(raw, "backbone", BackboneSection)
    pipeline_raw = _section(raw, "pipeline", PipelineSection)
    analysis_raw = _section(raw, "analysis", AnalysisSection)
    train_raw = _section(raw, "train", TrainSection)
    eval_raw = _section(raw, "eval", EvalSection)
    data_raw = _section(raw, "data", DataSection)
    output_raw = _section(raw, "output", OutputSection)

    stages_raw = pipeline_raw.get("stages", [])
    if not isinstance(stages_raw, list):
        raise ConfigError("pipeline.stages must be a list.")
    for index, stage_raw in enumerate(stages_raw):
        if not isinstance(stage_raw, dict):
            raise ConfigError(f"pipeline.stages[{index}] must be a mapping.")
    extensions = _string_list_value(pipeline_raw, "extensions", "pipeline")
    completed_stages = complete_stage_requests(
        stages_raw, backbone_name=backbone_raw.get("name", ""), extensions=extensions,
    )
    stages = [CompressionStageConfig.from_dict(stage) for stage in completed_stages]

    return ExperimentConfig(
        experiment=ExperimentSection(
            name=experiment_raw.get("name", ""),
            run_id=_optional_string_value(experiment_raw.get("run_id"), "experiment.run_id"),
            seed=int(experiment_raw.get("seed", 42)),
        ),
        backbone=BackboneSection(
            name=backbone_raw.get("name", ""),
            model_path=backbone_raw.get("model_path"),
            model_id=backbone_raw.get("model_id"),
            model_revision=backbone_raw.get("model_revision"),
            source_path=backbone_raw.get("source_path"),
            trust_remote_code=_optional_bool_value(
                backbone_raw.get("trust_remote_code"),
                "backbone.trust_remote_code",
            ),
            config=_mapping_value(backbone_raw, "config", "backbone"),
        ),
        pipeline=PipelineSection(
            stages=stages,
            extensions=extensions,
        ),
        analysis=AnalysisSection(
            backend=analysis_raw.get("backend"),
            compression_mode=_enum_value(
                CompressionMode,
                analysis_raw.get("compression_mode", CompressionMode.NONE),
                "analysis.compression_mode",
            ),
            phase=_enum_value(
                AnalysisPhase,
                analysis_raw.get("phase", AnalysisPhase.PRE_COMPRESSION),
                "analysis.phase",
            ),
            runtime_batch_size=_strict_integer_value(
                analysis_raw.get("runtime_batch_size", 1),
                "analysis.runtime_batch_size",
            ),
            tasks=_string_list_value(analysis_raw, "tasks", "analysis"),
            sample_limit=analysis_raw.get("sample_limit"),
            task_include_path=_string_list_value(
                analysis_raw, "task_include_path", "analysis"
            ),
            streaming_datasets=_optional_bool_value(
                analysis_raw.get("streaming_datasets"),
                "analysis.streaming_datasets",
            )
            if "streaming_datasets" in analysis_raw
            else False,
            sampling=_mapping_value(analysis_raw, "sampling", "analysis"),
            probes=_mapping_list_value(analysis_raw, "probes", "analysis"),
        ),
        train=TrainSection(
            backend=train_raw.get("backend"),
            compression_mode=_enum_value(
                CompressionMode,
                train_raw.get("compression_mode", CompressionMode.NONE),
                "train.compression_mode",
            ),
            runtime_mode=_enum_value(
                RuntimeMode,
                train_raw.get("runtime_mode", RuntimeMode.DEFAULT),
                "train.runtime_mode",
            ),
            script_path=train_raw.get("script_path"),
            args_path=train_raw.get("args_path"),
            checkpoint_in=train_raw.get("checkpoint_in"),
            resume=train_raw.get("resume"),
            max_steps=train_raw.get("max_steps"),
            args=_mapping_value(train_raw, "args", "train"),
        ),
        eval=EvalSection(
            backend=eval_raw.get("backend"),
            compression_mode=_enum_value(
                CompressionMode,
                eval_raw.get("compression_mode", CompressionMode.NONE),
                "eval.compression_mode",
            ),
            tasks=_string_list_value(eval_raw, "tasks", "eval"),
            runtime_batch_size=int(eval_raw.get("runtime_batch_size", 1)),
            num_processes=int(eval_raw.get("num_processes", 1)),
            distributed_launcher=_enum_value(
                DistributedLauncher,
                eval_raw.get(
                    "distributed_launcher",
                    DistributedLauncher.ACCELERATE,
                ),
                "eval.distributed_launcher",
            ),
            main_process_port=(
                None
                if eval_raw.get("main_process_port") is None
                else int(eval_raw["main_process_port"])
            ),
            sample_limit=eval_raw.get("sample_limit"),
            checkpoint_path=eval_raw.get("checkpoint_path"),
            target=_optional_enum_value(EvalTarget, eval_raw.get("target"), "eval.target"),
            runtime_mode=_enum_value(
                RuntimeMode,
                eval_raw.get("runtime_mode", RuntimeMode.DEFAULT),
                "eval.runtime_mode",
            ),
            generation=_mapping_value(eval_raw, "generation", "eval"),
            args=_mapping_value(eval_raw, "args", "eval"),
        ),
        data=DataSection(
            dataset_id=_optional_string_value(
                data_raw.get("dataset_id"),
                "data.dataset_id",
            ),
            dataset_commit=data_raw.get("dataset_commit"),
            sample_split=data_raw.get("sample_split"),
        ),
        output=OutputSection(
            run_dir=_optional_string_value(output_raw.get("run_dir"), "output.run_dir"),
            manifest_path=_optional_string_value(
                output_raw.get("manifest_path"), "output.manifest_path"
            ),
        ),
        raw=raw,
        path=path,
    )


def _section(
    raw: Mapping[str, Any],
    name: str,
    schema: type[Any],
) -> dict[str, Any]:
    """Return one strict section, deriving field names from its dataclass."""

    value = raw.get(name, {})
    if not isinstance(value, Mapping):
        raise ConfigError(f"Config section {name!r} must be a mapping.")
    allowed = {item.name for item in fields(schema)}
    unknown = sorted(set(value) - allowed)
    if unknown:
        raise ConfigError(f"Unknown fields in config section {name!r}: {unknown}")
    return dict(value)


_EnumT = TypeVar("_EnumT", bound=StringEnum)


def _enum_value(enum_type: type[_EnumT], value: Any, path: str) -> _EnumT:
    try:
        return enum_type(value)
    except (TypeError, ValueError) as exc:
        raise ConfigError(
            f"{path} must be one of {list(enum_values(enum_type))}, got {value!r}."
        ) from exc


def _optional_enum_value(
    enum_type: type[_EnumT],
    value: Any,
    path: str,
) -> _EnumT | None:
    return None if value is None else _enum_value(enum_type, value, path)


def _optional_bool_value(value: Any, path: str) -> bool | None:
    if value is None or isinstance(value, bool):
        return value
    raise ConfigError(f"{path} must be a bool when provided.")


def _optional_string_value(value: Any, path: str) -> str | None:
    if value is None or isinstance(value, str):
        return value
    raise ConfigError(f"{path} must be a string or null when provided.")


def _strict_integer_value(value: Any, path: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ConfigError(f"{path} must be an integer.")
    return value


def _mapping_value(
    section: Mapping[str, Any],
    key: str,
    section_name: str,
) -> dict[str, Any]:
    value = section.get(key, {})
    if not isinstance(value, Mapping):
        raise ConfigError(f"{section_name}.{key} must be a mapping.")
    return dict(value)


def _string_list_value(
    section: Mapping[str, Any],
    key: str,
    section_name: str,
) -> list[str]:
    value = section.get(key, [])
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        raise ConfigError(f"{section_name}.{key} must be a list of strings.")
    return list(value)


def _mapping_list_value(
    section: Mapping[str, Any],
    key: str,
    section_name: str,
) -> list[dict[str, Any]]:
    value = section.get(key, [])
    if not isinstance(value, list) or any(not isinstance(item, Mapping) for item in value):
        raise ConfigError(f"{section_name}.{key} must be a list of mappings.")
    return [dict(item) for item in value]
