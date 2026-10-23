"""Restore a recorded experiment through existing model and trainer loaders."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal

from mllm_comp.compressors.extensions import compression_contracts
from mllm_comp.config import ExperimentConfig
from mllm_comp.config.schema import experiment_config_from_dict
from mllm_comp.core.errors import ConfigError
from mllm_comp.core.vocabulary import CompressionMethod, CompressionMode, EvalTarget, ExecutionPhase


def restore_config(
    run_dir: str | Path,
    *,
    task: Literal["eval", "train", "analyze"],
    checkpoint: str | Path | None = None,
) -> ExperimentConfig:
    """Load a run's effective recipe and select an actual saved checkpoint.

    The returned configuration starts an independent experiment. Training keeps
    the original model and compression construction and delegates checkpoint
    state restoration to Swift. Evaluation/analysis load saved model weights;
    materialized operations are not applied again, while learned HTA modules
    are reconstructed at their original locations before loading their weights.
    No models or tensors are loaded here.
    """

    if task not in {"eval", "train", "analyze"}:
        raise ConfigError(f"Unknown restore task: {task!r}.")
    directory = Path(run_dir).resolve()
    snapshot = directory / "effective_config.json"
    config = experiment_config_from_dict(_read_json(snapshot), path=str(snapshot))
    result = _read_json(directory / "result.json")
    source_task = result.get("task")
    if source_task not in {"eval", "train", "analyze"}:
        raise ConfigError("The recorded result must identify an eval, train or analyze task.")
    if checkpoint is None:
        recorded = (result.get("output") or {}).get("checkpoint_path")
        if source_task != "train" or not recorded:
            raise ConfigError(
                "This run did not record a produced training checkpoint. "
                "Provide --checkpoint explicitly for an existing saved artifact; "
                "an evaluation input model is not a produced checkpoint."
            )
        checkpoint_path = Path(recorded)
        if not checkpoint_path.is_absolute():
            raise ConfigError("The recorded checkpoint path is relative; provide an explicit checkpoint path.")
    else:
        checkpoint_path = Path(checkpoint).resolve()
    if not checkpoint_path.exists():
        raise ConfigError(f"Saved checkpoint does not exist: {checkpoint_path}")

    artifact_kind, model_config = _checkpoint_kind(checkpoint_path)
    from mllm_comp.integrations.ms_swift.split_lifecycle import inspect_split_checkpoint, validate_checkpoint_plan
    split_contract = inspect_split_checkpoint(checkpoint_path) if checkpoint_path.is_dir() else None
    if split_contract is not None:
        from mllm_comp.modality_split.pipeline import compile_pipeline_split
        validate_checkpoint_plan(checkpoint_path, compile_pipeline_split(config))
    source_section = getattr(config, "analysis" if source_task == "analyze" else source_task)
    source_mode = CompressionMode(source_section.compression_mode)
    section = getattr(config, "analysis" if task == "analyze" else task)
    section.compression_mode = source_mode
    enabled = [stage for stage in config.pipeline.stages if stage.enabled]
    active = enabled if source_mode is CompressionMode.COMPRESSED else []
    hta_stages = [stage for stage in active if stage.algorithm == "fastslm_hta_audio"]
    phase = (
        ExecutionPhase.GENERATE
        if task == "eval" and config.eval.args.get("forward_only_probe") is not True
        else ExecutionPhase.FORWARD
    )
    for stage in active:
        if stage.execution_phase is not ExecutionPhase.PREPARE:
            stage.execution_phase = phase

    if task == "train":
        if artifact_kind == "hta_module":
            raise ConfigError("An HTA module file has no native Swift training state; use a Swift checkpoint directory.")
        if not (checkpoint_path / "trainer_state.json").is_file():
            raise ConfigError("Training restore requires the checkpoint's native Swift trainer_state.json.")
        # Swift must build the same model and modules before restoring its own
        # weights, optimizer, scheduler and RNG state. Loading a pruned model
        # here would apply the original structural stage twice.
        config.train.resume = str(checkpoint_path)
        config.train.args["resume_only_model"] = False
    else:
        if artifact_kind == "peft" and split_contract is None:
            raise ConfigError(
                "This is a PEFT adapter checkpoint. Framework eval/analyze do not "
                "load or merge PEFT adapters; supply a complete merged model checkpoint. "
                "Native Swift training resume remains available."
            )
        if artifact_kind == "hta_module":
            if len(hta_stages) != 1:
                raise ConfigError(
                    "An HTA-only file requires exactly one enabled fastslm_hta_audio stage; "
                    "multiple HTA locations require a full checkpoint with their named modules."
                )
            hta_stages[0].config["checkpoint_path"] = str(checkpoint_path)
            hta_stages[0].config["checkpoint_strict"] = True
            # A module-only artifact must be paired with the model actually
            # selected by the source task, including its checkpoint override.
            source_model = (
                config.train.checkpoint_in if source_task == "train"
                else config.eval.checkpoint_path if source_task == "eval"
                else None
            ) or config.backbone.model_path or config.backbone.model_id
            config.backbone.model_path = source_model
            if task == "eval":
                config.eval.checkpoint_path = source_model
        else:
            if split_contract is None:
                _disable_materialized_operations(config, active, checkpoint_path, model_config)
            config.backbone.model_path = str(checkpoint_path)
            if model_config.get("model_type") == "qwen2_5_omni_thinker":
                config.backbone.config["enable_talker"] = False
            if task == "eval":
                config.eval.checkpoint_path = str(checkpoint_path)
            for stage in hta_stages:
                stage.config["checkpoint_path"] = str(checkpoint_path)
                stage.config["checkpoint_strict"] = True
        if source_mode is CompressionMode.COMPRESSED and not any(stage.enabled for stage in config.pipeline.stages):
            section.compression_mode = CompressionMode.NONE
        if task == "eval":
            config.eval.target = EvalTarget.COMPRESSED if source_mode is CompressionMode.COMPRESSED else EvalTarget.BASELINE
        config.train.resume = None

    config.experiment.run_id = None
    config.output.run_dir = None
    config.output.manifest_path = None
    # Explicit diagnostic filenames identify the old run, not the replay.
    config.eval.args.pop("diagnostic_trace_path", None)
    config.train.args.pop("output_dir", None)
    return config


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ConfigError(f"{path}: expected a JSON object.")
    return payload


def _checkpoint_kind(path: Path) -> tuple[str, dict[str, Any]]:
    if path.is_file():
        if path.suffix not in {".pt", ".pth", ".bin", ".safetensors"}:
            raise ConfigError("A standalone checkpoint must be an HTA state-dict file.")
        return "hta_module", {}
    adapter_path = path / "adapter_config.json"
    if adapter_path.is_file():
        adapter = _read_json(adapter_path)
        if adapter.get("format") == "mllm_comp_selected_parameter_checkpoint":
            raise ConfigError(
                "This selected-parameter checkpoint is a diagnostic final-norm artifact, "
                "not full model weights or a native Swift training checkpoint."
            )
        if adapter.get("peft_type"):
            _require_weights(path, stem="adapter_model", binary="adapter_model.bin")
            return "peft", adapter
        raise ConfigError("Unrecognized adapter checkpoint format; no generic adapter loader is available.")
    model_config_path = path / "config.json"
    if not model_config_path.is_file():
        raise ConfigError("A full model checkpoint requires config.json and its model weight files.")
    _require_weights(path, stem="model", binary="pytorch_model.bin")
    return "full_model", _read_json(model_config_path)


def _require_weights(path: Path, *, stem: str, binary: str) -> None:
    for filename in (f"{stem}.safetensors", binary):
        if (path / filename).is_file():
            return
    for filename in (f"{stem}.safetensors.index.json", f"{binary}.index.json"):
        index_path = path / filename
        if not index_path.is_file():
            continue
        weight_map = _read_json(index_path).get("weight_map")
        if not isinstance(weight_map, dict) or not weight_map:
            raise ConfigError(f"Checkpoint index has no weight_map: {index_path}")
        if any(not isinstance(shard, str) or not shard for shard in weight_map.values()):
            raise ConfigError(f"Checkpoint index contains an invalid shard name: {index_path}")
        missing = [shard for shard in sorted(set(weight_map.values())) if not (path / shard).is_file()]
        if missing:
            raise ConfigError(f"Checkpoint index references missing weight shards: {missing}")
        return
    raise ConfigError(f"Checkpoint has no saved model weights: {path}")


def _disable_materialized_operations(
    config: ExperimentConfig, active: list[Any], checkpoint: Path, model_config: dict[str, Any],
) -> None:
    registry = compression_contracts(config.pipeline.extensions) if any(stage.algorithm for stage in active) else None
    operations = [
        stage for stage in active
        if (registry.algorithm(stage.algorithm).method if stage.algorithm else stage.method)
        is CompressionMethod.OPERATION_COMPRESSION
    ]
    if not operations:
        return
    records = model_config.get("mllm_comp_operations", [])
    if not records and (checkpoint / "operation_compression_manifest.json").is_file():
        exported = _read_json(checkpoint / "operation_compression_manifest.json")
        records = [exported.get("operation_compression", {})]
    if not isinstance(records, list) or any(not isinstance(record, dict) for record in records):
        raise ConfigError("The checkpoint has invalid materialized Operation metadata.")
    materialized = {record.get("stage_name") for record in records}
    missing = [stage.name for stage in operations if stage.name not in materialized]
    if missing:
        raise ConfigError(
            "The checkpoint does not record materialized Operation stages "
            f"{missing}; restore cannot safely decide whether to apply structural edits."
        )
    names = {stage.name for stage in operations}
    bindings = {
        binding for stage in operations
        for binding in ([stage.binding] if stage.binding else [item.binding for item in stage.bindings])
    }
    for stage in config.pipeline.stages:
        if stage.name in names:
            stage.enabled = False
        else:
            stage.depends_on = [name for name in stage.depends_on if name not in names]
            for binding in stage.bindings:
                binding.depends_on = [name for name in binding.depends_on if name not in bindings]


__all__ = ["restore_config"]
