"""Config validation helpers."""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any, Literal

from mllm_comp.core.errors import ConfigError

from .schema import ExperimentConfig, config_to_dict


_ENV_PLACEHOLDER = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")


def validate_runnable_config(
    config: ExperimentConfig,
    *,
    task: Literal["eval", "train", "analyze"] | None = None,
) -> None:
    """Validate a complete recipe, or only the inputs used by one task.

    Task-specific model and dataset choices follow the existing native
    adapters. An unused task or disabled compression stage may retain template
    placeholders, so the same recipe can be used for baseline and compression.
    """

    missing: list[str] = []
    if not _is_nonempty_string(config.experiment.name):
        missing.append("experiment.name")
    if config.experiment.run_id is not None and not _is_nonempty_string(config.experiment.run_id):
        missing.append("experiment.run_id must be non-empty when provided")
    if not _is_nonempty_string(config.backbone.name):
        missing.append("backbone.name")
    effective = config_to_dict(config)
    model_fields = [("backbone", "model_path"), ("backbone", "model_id")]
    if task == "train":
        model_fields.insert(0, ("train", "checkpoint_in"))
    elif task == "eval":
        model_fields.insert(0, ("eval", "checkpoint_path"))
    elif task is None:
        model_fields.append(("backbone", "source_path"))
    selected_model = next(
        ((section, name) for section, name in model_fields if effective[section][name]),
        None,
    )
    if selected_model is None or not _is_nonempty_string(effective[selected_model[0]][selected_model[1]]):
        missing.append("one of " + ", ".join(f"{section}.{name}" for section, name in model_fields))

    if task in {"eval", "analyze"}:
        section = "analysis" if task == "analyze" else "eval"
        if not getattr(config, section).tasks:
            missing.append(f"{section}.tasks (set task names in the recipe or pass --tasks)")
    elif task == "train":
        datasets = config.train.args.get("datasets")
        dataset_field = "train.args.datasets"
        if datasets is None:
            datasets = config.data.dataset_id
            dataset_field = "train.args.datasets or data.dataset_id"
        values = [datasets] if isinstance(datasets, str) else datasets
        if not isinstance(values, (list, tuple)) or not values or not all(_is_nonempty_string(item) for item in values):
            missing.append(f"{dataset_field} (set a non-empty dataset name/path or list)")
    elif not _has_runtime_data_source(config):
        missing.append("a data source in data.dataset_id, train.args.datasets, analysis.tasks, or eval.tasks")
    for name in ("run_dir", "manifest_path"):
        value = getattr(config.output, name)
        if value is not None and not _is_nonempty_string(value):
            missing.append(f"output.{name} must be non-empty when provided")
    if missing:
        raise ConfigError("Runnable config is missing required values: " + ", ".join(missing))

    if task is not None:
        section = "analysis" if task == "analyze" else task
        for unused in {"train", "eval", "analysis"} - {section}:
            effective.pop(unused)
        # Each backend chooses one model source; a higher-priority checkpoint
        # makes the recipe's default model placeholders irrelevant.
        for name in ("model_path", "model_id", "source_path"):
            if selected_model != ("backbone", name):
                effective["backbone"].pop(name)
        if task != "train" or config.train.args.get("datasets") is not None:
            effective.pop("data")
        if str(getattr(config, section).compression_mode) == "compressed":
            effective["pipeline"]["stages"] = [
                stage if stage["enabled"] else None
                for stage in effective["pipeline"]["stages"]
            ]
        else:
            effective.pop("pipeline")
    unresolved = _unresolved_environment_variables(effective)
    if unresolved:
        fields = "; ".join(f"{path}: ${{{name}}}" for path, name in unresolved)
        raise ConfigError(
            f"Runnable config contains unresolved environment variables: {fields}. "
            "Set these environment variables before loading the recipe, or replace the listed fields."
        )


def _has_runtime_data_source(config: ExperimentConfig) -> bool:
    if _is_nonempty_string(config.data.dataset_id):
        return True
    datasets = config.train.args.get("datasets")
    if isinstance(datasets, list) and any(_is_nonempty_string(item) for item in datasets):
        return True
    return bool(config.analysis.tasks or config.eval.tasks)


def _is_nonempty_string(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _unresolved_environment_variables(value: Any, path: str = "") -> list[tuple[str, str]]:
    if isinstance(value, str):
        return [(path, name) for name in dict.fromkeys(_ENV_PLACEHOLDER.findall(value))]
    if isinstance(value, Mapping):
        return [
            match
            for key, item in value.items()
            for match in _unresolved_environment_variables(item, f"{path}.{key}" if path else str(key))
        ]
    if isinstance(value, list):
        return [
            match for index, item in enumerate(value)
            for match in _unresolved_environment_variables(item, f"{path}[{index}]")
        ]
    return []


def validate_runtime_batch_size(
    *,
    requested: int,
    max_analysis_eval_batch_size: int | None,
    backbone_name: str,
) -> None:
    """Validate analysis/eval runtime batch size against a profile."""

    if requested < 1:
        raise ConfigError(f"runtime batch size must be >= 1, got {requested}")
    if max_analysis_eval_batch_size is not None and requested > max_analysis_eval_batch_size:
        raise ConfigError(
            f"{backbone_name} allows runtime analysis/eval batch size "
            f"<= {max_analysis_eval_batch_size}, got {requested}"
        )
