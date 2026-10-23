"""Native-CLI-first ms-swift training backend."""

from __future__ import annotations

import base64
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import sys
from typing import Any, Callable

from mllm_comp.backbones.qwen25_omni.v2_compression import (
    resolve_qwen25_omni_compression_execution,
)
from mllm_comp.compressors.extensions import compression_contracts
from mllm_comp.core.errors import BackendIntegrationError, RegistryError
from mllm_comp.core.manifest import BackendRecord
from mllm_comp.core.serialization import atomic_write_text, to_jsonable
from mllm_comp.core.vocabulary import CompressionMode, ExecutionPhase
from mllm_comp.integrations.ms_swift.diagnostics import (
    _pretty_training_diagnostic_path,
    _pretty_training_diagnostic_records,
)
from mllm_comp.integrations.lmms_eval.diagnostics import _pretty_diagnostic_json
from mllm_comp.integrations.native_cli import (
    NativeCLIPlan,
    NativeCLIProcessError,
    NativeCLIResult,
    installed_versions,
    resolved_cwd,
    run_native_cli,
)


_ARGUMENT_NAME = re.compile(r"^[A-Za-z][A-Za-z0-9_]*$")
_CHECKPOINT_NAME = re.compile(r"^checkpoint-(\d+)$")
_PLATFORM_ONLY_KEYS = {
    "artifact_format",
    "enable_audio_output",
    "trainable_branch",
    "native_argv",
}
_VSCODE_DEBUG_SUBPROCESSES_ENV = "MLLM_COMP_VSCODE_DEBUG_SUBPROCESSES"
_PLATFORM_OWNED_SWIFT_KEYS = {
    "model",
    "model_type",
    "template",
    "dataset",
    "output_dir",
    "max_steps",
    "resume_from_checkpoint",
}
_QWEN25_OMNI_ENVIRONMENT_FIELDS = {
    "max_pixels": "MAX_PIXELS",
    "use_audio_in_video": "USE_AUDIO_IN_VIDEO",
    "sampling_rate": "SAMPLING_RATE",
    "video_min_pixels": "VIDEO_MIN_PIXELS",
    "video_max_pixels": "VIDEO_MAX_PIXELS",
    "fps": "FPS",
    "fps_min_frames": "FPS_MIN_FRAMES",
    "fps_max_frames": "FPS_MAX_FRAMES",
    "frame_factor": "FRAME_FACTOR",
}
_QWEN25_OMNI_MEDIA_FIELDS = frozenset(_QWEN25_OMNI_ENVIRONMENT_FIELDS)
_QWEN25_OMNI_POSITIVE_INTEGER_FIELDS = frozenset(
    {
        "max_pixels",
        "sampling_rate",
        "video_min_pixels",
        "video_max_pixels",
        "fps_min_frames",
        "fps_max_frames",
        "frame_factor",
    }
)
_QWEN25_OMNI_UNSET_ENVIRONMENT = tuple(
    _QWEN25_OMNI_ENVIRONMENT_FIELDS.values()
)


@dataclass(frozen=True)
class MSSwiftLaunchPlan(NativeCLIPlan):
    """Deterministic official ``swift sft`` invocation."""

    model_path: str
    datasets: list[str]
    output_dir: str
    compression_mode: str
    artifact_format: str | None
    backbone_name: str
    model_type: str
    template: str
    trainable_branch: str = "thinker"
    talker_training: bool = False
    compression_training: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class _MSSwiftBackboneSpec:
    backbone_name: str
    model_type: str
    template: str
    default_trainable_branch: str
    trainable_branches: tuple[str, ...]
    environment: dict[str, str]
    compression_stage_limit: int | None = 1


_BACKBONE_SPECS = {
    "qwen25_omni": _MSSwiftBackboneSpec(
        backbone_name="qwen25_omni",
        model_type="qwen2_5_omni",
        template="qwen2_5_omni",
        default_trainable_branch="thinker",
        trainable_branches=("thinker",),
        environment={"ENABLE_AUDIO_OUTPUT": "0"},
        compression_stage_limit=None,
    ),
    "qwen3_vl": _MSSwiftBackboneSpec(
        backbone_name="qwen3_vl",
        model_type="qwen3_vl",
        template="qwen3_vl",
        default_trainable_branch="language_model",
        trainable_branches=("language_model",),
        environment={},
    ),
    "qwen25_vl": _MSSwiftBackboneSpec(
        backbone_name="qwen25_vl",
        model_type="qwen2_5_vl",
        template="qwen2_5_vl",
        default_trainable_branch="language_model",
        trainable_branches=("language_model",),
        environment={},
    ),
    "llava_onevision": _MSSwiftBackboneSpec(
        backbone_name="llava_onevision",
        model_type="llava_onevision_qwen",
        template="llava_onevision_qwen",
        default_trainable_branch="language_model",
        trainable_branches=("language_model",),
        environment={},
    ),
}


@dataclass
class MSSwiftTrainResult:
    checkpoint_path: str
    train_log_path: str | None
    metrics: dict[str, float] = field(default_factory=dict)
    backend_args: dict[str, Any] = field(default_factory=dict)
    backend: BackendRecord | None = None
    raw_result: dict[str, Any] = field(default_factory=dict)


def _swift_versions() -> dict[str, str]:
    return installed_versions(primary={"ms_swift": ("ms-swift", "ms_swift")})


def _as_string_list(value: Any, *, name: str) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        values = list(value)
        if all(isinstance(item, str) for item in values):
            return values
    raise BackendIntegrationError(
        f"ms-swift argument {name!r} must be a string or list of strings."
    )


def _argument_values(value: Any, *, name: str) -> list[str]:
    if isinstance(value, bool):
        return ["true" if value else "false"]
    if isinstance(value, (str, int, float)):
        return [str(value)]
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        values: list[str] = []
        for item in value:
            if not isinstance(item, (str, int, float, bool)):
                raise BackendIntegrationError(
                    f"ms-swift argument {name!r} contains unsupported value {item!r}."
                )
            values.extend(_argument_values(item, name=name))
        return values
    raise BackendIntegrationError(
        f"ms-swift argument {name!r} must be a scalar or list of scalar values."
    )


def _append_argument(argv: list[str], name: str, value: Any) -> None:
    if value is None:
        return
    if not _ARGUMENT_NAME.fullmatch(name):
        raise BackendIntegrationError(f"Invalid ms-swift argument name {name!r}.")
    argv.append(f"--{name}")
    argv.extend(_argument_values(value, name=name))


def _qwen25_omni_media_overrides(
    *,
    backbone_name: str,
    backbone_config: Mapping[str, Any],
    swift_args: Mapping[str, Any],
) -> tuple[dict[str, str], list[str]]:
    declared = {
        name: backbone_config[name]
        for name in _QWEN25_OMNI_MEDIA_FIELDS
        if name in backbone_config
    }
    if backbone_name != "qwen25_omni":
        if declared:
            raise BackendIntegrationError(
                "Qwen2.5-Omni media controls in backbone.config are not supported for "
                f"backbone {backbone_name!r}: {sorted(declared)}."
            )
        return {}, []

    misplaced = sorted(_QWEN25_OMNI_MEDIA_FIELDS.intersection(swift_args))
    if misplaced:
        raise BackendIntegrationError(
            "Qwen2.5-Omni media controls belong in backbone.config, not train.args: "
            f"{misplaced}."
        )

    if "use_audio_in_video" in declared and not isinstance(
        declared["use_audio_in_video"], bool
    ):
        raise BackendIntegrationError(
            "backbone.config.use_audio_in_video must be a boolean."
        )

    for name in _QWEN25_OMNI_POSITIVE_INTEGER_FIELDS:
        if name not in declared:
            continue
        value = declared[name]
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise BackendIntegrationError(
                f"backbone.config.{name} must be a positive integer."
            )

    if "fps" in declared:
        fps = declared["fps"]
        if isinstance(fps, bool) or not isinstance(fps, (int, float)) or fps <= 0:
            raise BackendIntegrationError(
                "backbone.config.fps must be a positive number."
            )

    bounded_pairs = (
        ("video_min_pixels", "video_max_pixels"),
        ("fps_min_frames", "fps_max_frames"),
    )
    for minimum_name, maximum_name in bounded_pairs:
        if minimum_name in declared and maximum_name in declared:
            if declared[minimum_name] > declared[maximum_name]:
                raise BackendIntegrationError(
                    f"backbone.config.{minimum_name} must be <= "
                    f"backbone.config.{maximum_name}."
                )

    environment: dict[str, str] = {}
    for config_name, environment_name in _QWEN25_OMNI_ENVIRONMENT_FIELDS.items():
        if config_name not in declared:
            continue
        value = declared[config_name]
        if isinstance(value, bool):
            environment[environment_name] = "1" if value else "0"
        else:
            environment[environment_name] = str(value)

    return environment, list(_QWEN25_OMNI_UNSET_ENVIRONMENT)


def _checkpoint_sort_key(path: Path) -> tuple[int, int]:
    match = _CHECKPOINT_NAME.fullmatch(path.name)
    step = int(match.group(1)) if match else -1
    return step, path.stat().st_mtime_ns


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _collect_swift_outputs(
    output_dir: Path,
) -> tuple[Path, Path | None, dict[str, float], dict[str, Any]]:
    if not output_dir.exists():
        raise BackendIntegrationError(
            "ms-swift exited successfully but its native output directory does not exist: "
            f"{output_dir}."
        )
    checkpoints = [
        path
        for path in output_dir.rglob("checkpoint-*")
        if path.is_dir() and _CHECKPOINT_NAME.fullmatch(path.name)
    ]
    checkpoint = max(checkpoints, key=_checkpoint_sort_key) if checkpoints else output_dir
    log_candidates = [path for path in output_dir.rglob("logging.jsonl") if path.is_file()]
    logging_path = (
        max(log_candidates, key=lambda path: path.stat().st_mtime_ns)
        if log_candidates
        else None
    )

    metrics: dict[str, float] = {}
    last_log: dict[str, Any] = {}
    if logging_path is not None:
        for line in logging_path.read_text(encoding="utf-8").splitlines():
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(payload, dict):
                last_log = payload
        for key, value in last_log.items():
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                metrics[f"last/{key}"] = float(value)

    trainer_state_candidates = [
        path for path in output_dir.rglob("trainer_state.json") if path.is_file()
    ]
    trainer_state: dict[str, Any] = {}
    if trainer_state_candidates:
        trainer_state = _read_json(
            max(trainer_state_candidates, key=lambda path: path.stat().st_mtime_ns)
        )
        for key in ("best_metric", "global_step", "epoch"):
            value = trainer_state.get(key)
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                metrics[key] = float(value)

    raw_result = {
        "checkpoint_path": str(checkpoint),
        "logging_path": str(logging_path) if logging_path is not None else None,
        "last_log": last_log,
        "trainer_state": trainer_state,
    }
    return checkpoint, logging_path, metrics, raw_result


def _training_compression_spec(config: Any) -> tuple[str, dict[str, Any]]:
    enabled = [stage for stage in config.pipeline.stages if stage.enabled]
    if not enabled:
        raise BackendIntegrationError(
            "Compressed training requires an enabled pipeline stage."
        )
    backbone_name = config.backbone.name
    if any(stage.algorithm == "fixed_mask" for stage in enabled):
        return _fixed_mask_training_compression_spec(config, enabled)
    if backbone_name in {"qwen25_vl", "qwen3_vl"}:
        from mllm_comp.modality_split.pipeline import compile_pipeline_split
        plan = compile_pipeline_split(config)
        if plan is not None:
            return _split_training_compression_spec(config, plan)
    limit = _BACKBONE_SPECS[backbone_name].compression_stage_limit
    if limit is not None and len(enabled) > limit:
        raise BackendIntegrationError(
            "Image compressed training requires exactly one enabled pipeline stage."
        )
    if backbone_name != "qwen25_omni":
        return _image_training_compression_spec(
            config,
            stage=enabled[0],
            backbone_name=backbone_name,
        )
    if config.train.resume and bool(config.train.args.get("resume_only_model", False)):
        raise BackendIntegrationError(
            "Compressed train.resume requires Swift's complete Trainer restore; "
            "resume_only_model would load an already transformed model before stage installation."
        )
    registry = compression_contracts(config.pipeline.extensions)
    stages = []
    for stage in enabled:
        if not stage.binding or not stage.variant or not stage.algorithm:
            raise BackendIntegrationError(
                "Qwen2.5-Omni compression training requires an exact "
                "algorithm, variant and binding."
            )
        try:
            execution = resolve_qwen25_omni_compression_execution(
                stage=stage,
                binding=registry.binding(stage.binding),
                algorithm=registry.algorithm(stage.algorithm),
                variant=registry.variant(stage.variant),
            )
        except RegistryError as exc:
            raise BackendIntegrationError(str(exc)) from exc
        required_phase = (
            ExecutionPhase.PREPARE
            if execution.executor_family == "llm_operation"
            else ExecutionPhase.FORWARD
        )
        if stage.execution_phase is not required_phase:
            raise BackendIntegrationError(
                f"Compressed training stage {stage.name!r} requires "
                f"execution_phase={required_phase.value}."
            )
        stages.append({
            "stage_name": stage.name,
            "plugin": stage.plugin,
            "executor_family": execution.executor_family,
            "execution_phase": stage.execution_phase.value,
        })
    unsupported_layout_modes = [
        name
        for name in ("packing", "padding_free")
        if bool(config.train.args.get(name, False))
    ]
    if unsupported_layout_modes and any(
        item["executor_family"] != "llm_operation" for item in stages
    ):
        raise BackendIntegrationError(
            "Dynamic Qwen2.5-Omni compressed training does not yet support "
            f"{unsupported_layout_modes}."
        )
    payload = {
        "schema_version": 3,
        "backbone_name": backbone_name,
        "runtime_mode": config.train.runtime_mode.value,
        "extensions": list(config.pipeline.extensions),
        "stages": [to_jsonable(stage) for stage in enabled],
    }
    raw = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    encoded = base64.urlsafe_b64encode(raw).decode("ascii")
    return encoded, {
        "enabled": True,
        **(stages[0] if len(stages) == 1 else {
            "executor_family": "omni_multistage", "stages": stages,
        }),
        "runtime_mode": config.train.runtime_mode.value,
        "spec_hash": hashlib.sha256(raw).hexdigest(),
    }


def _fixed_mask_training_compression_spec(config, stages):
    from mllm_comp.fixed_mask.binding import validate_stages
    from mllm_comp.fixed_mask.checkpoint import validate_checkpoint
    validate_stages(stages)
    if config.backbone.name not in {"qwen25_vl", "llava_onevision"}:
        raise BackendIntegrationError("fixed_mask supports Qwen2.5-VL and LLaVA-OneVision")
    for name in ("packing", "padding_free", "use_liger_kernel"):
        if config.train.args.get(name):
            raise BackendIntegrationError(f"fixed_mask training does not support {name}")
    if config.train.args.get("per_device_train_batch_size", 1) != 1 or config.train.args.get("per_device_eval_batch_size", 1) != 1:
        raise BackendIntegrationError("fixed_mask requires per-device batch size one")
    for name in ("train_type", "tuner_type"):
        if config.train.args.get(name, "lora") not in {"full", "lora"}:
            raise BackendIntegrationError("fixed_mask recovery supports full and lora training")
    for stage in stages:
        if stage.algorithm == "fixed_mask":
            from mllm_comp.fixed_mask.spec import normalize_spec
            normalize_spec(stage.config)
            if stage.execution_phase is not ExecutionPhase.PREPARE:
                raise BackendIntegrationError("fixed_mask requires execution_phase=prepare")
        else:
            _image_training_compression_spec(config, stage=stage, backbone_name=config.backbone.name)
    checkpoint = config.train.resume or config.train.checkpoint_in
    if checkpoint:
        validate_checkpoint(checkpoint, config.backbone.name, stages)
    payload = {"schema_version": 5, "backbone_name": config.backbone.name,
               "runtime_mode": config.train.runtime_mode.value,
               "extensions": list(config.pipeline.extensions), "stages": [to_jsonable(s) for s in stages],
               "checkpoint": checkpoint}
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return base64.urlsafe_b64encode(raw).decode(), {
        "enabled": True, "executor_family": "image_fixed_mask", "pipeline_stages": payload["stages"],
        "runtime_mode": config.train.runtime_mode.value, "spec_hash": hashlib.sha256(raw).hexdigest()}


def _split_training_compression_spec(config: Any, plan: Any) -> tuple[str, dict[str, Any]]:
    # Fail before launching Swift rather than silently saving partition shards or
    # losing modality boundaries. Broader backend modes need their own acceptance.
    options = config.train.args
    for name in ("packing", "padding_free", "deepspeed", "fsdp", "load_best_model_at_end",
                 "use_dora", "quant_method", "quant_bits", "modules_to_save", "resume_only_model",
                 "native_argv", "use_liger_kernel", "sequence_parallel_size"):
        value = options.get(name)
        if value and not (name == "sequence_parallel_size" and value == 1):
            raise BackendIntegrationError(f"Image pipeline operations do not yet support {name}.")
    for name in ("tuner_type", "train_type"):
        if options.get(name, "lora") not in {"full", "lora"}:
            raise BackendIntegrationError(f"Image pipeline operations require {name}=full or lora.")
    if options.get("lora_bias", "none") != "none":
        raise BackendIntegrationError("Image pipeline operations require lora_bias=none.")
    checkpoint = config.train.resume
    if not checkpoint and config.train.checkpoint_in:
        from mllm_comp.integrations.ms_swift.split_lifecycle import inspect_split_checkpoint
        if inspect_split_checkpoint(config.train.checkpoint_in) is not None:
            checkpoint = config.train.checkpoint_in
    payload = {"schema_version": 4, "backbone_name": config.backbone.name,
               "runtime_mode": config.train.runtime_mode.value,
               "extensions": plan.contract["extensions"], "stages": plan.contract["stages"],
               "checkpoint": checkpoint}
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    stages = [{**item, "executor_family": "image_operation"} for item in plan.stages]
    return base64.urlsafe_b64encode(raw).decode(), {
        "enabled": True, "executor_family": "image_operation_multistage", "stages": stages,
        "runtime_mode": config.train.runtime_mode.value, "spec_hash": hashlib.sha256(raw).hexdigest()}


def _image_training_compression_spec(
    config: Any,
    *,
    stage: Any,
    backbone_name: str,
) -> tuple[str, dict[str, Any]]:
    if backbone_name not in {"qwen25_vl", "qwen3_vl", "llava_onevision"}:
        raise BackendIntegrationError(
            f"Dynamic image compression training does not support {backbone_name!r}."
        )
    if not stage.binding or not stage.variant or not stage.algorithm:
        raise BackendIntegrationError(
            "Dynamic image compression training requires an exact algorithm, "
            "variant and binding."
        )
    registry = compression_contracts(config.pipeline.extensions)
    try:
        binding = registry.binding(stage.binding)
        variant = registry.variant(stage.variant)
        algorithm = registry.algorithm(stage.algorithm)
    except RegistryError as exc:
        raise BackendIntegrationError(str(exc)) from exc
    if (
        binding.backbone != backbone_name
        or binding.variant != variant.name
        or variant.algorithm != algorithm.name
    ):
        raise BackendIntegrationError(
            "Dynamic image compression training stage does not match its exact "
            "backbone/variant/binding contract."
        )
    if stage.execution_phase not in binding.supported_phases:
        raise BackendIntegrationError(
            f"Compression binding {binding.name!r} does not support "
            f"execution_phase={stage.execution_phase.value}."
        )
    comparison_operation = (binding.exclusive_runtime_slot == "image_comparison"
                            and stage.execution_phase is ExecutionPhase.PREPARE)
    if stage.execution_phase is not ExecutionPhase.FORWARD and not comparison_operation:
        raise BackendIntegrationError(
            "Dynamic image compression training requires execution_phase=forward."
        )
    if bool(config.train.args.get("packing", False)) or bool(
        config.train.args.get("padding_free", False)
    ):
        raise BackendIntegrationError(
            "Dynamic image compression training does not support packing or "
            "padding_free batches."
        )
    if (
        backbone_name == "llava_onevision"
        and algorithm.name == "tome"
        and config.train.args.get("vit_gradient_checkpointing") is not False
    ):
        raise BackendIntegrationError(
            "LLaVA-OneVision ToMe training requires "
            "train.args.vit_gradient_checkpointing=false because its pinned "
            "vision callback cannot be replayed safely."
        )
    payload = {
        "schema_version": 2,
        "backbone_name": backbone_name,
        "runtime_mode": config.train.runtime_mode.value,
        "executor_family": "image_multimodal",
        "extensions": list(config.pipeline.extensions),
        "stage": to_jsonable(stage),
    }
    raw = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    encoded = base64.urlsafe_b64encode(raw).decode("ascii")
    return encoded, {
        "enabled": True,
        "stage_name": stage.name,
        "plugin": stage.plugin,
        "executor_family": "image_multimodal",
        "execution_phase": stage.execution_phase.value,
        "runtime_mode": config.train.runtime_mode.value,
        "backbone_name": backbone_name,
        "spec_hash": hashlib.sha256(raw).hexdigest(),
    }


def _collect_training_compression_evidence(
    evidence_dir: Path,
    *,
    expected_spec_hash: str,
    expected_runtime_mode: str,
    expected_executor_family: str = "projector_token",
    expected_stages: Sequence[Mapping[str, Any]] = (),
    expected_checkpoint: bool = True,
) -> dict[str, Any]:
    paths = sorted(
        {
            *evidence_dir.glob("compression_training_rank_*.jsonl"),
            *evidence_dir.glob("debug_diagnostic.jsonl"),
            *evidence_dir.glob("debug_diagnostic_rank_*.jsonl"),
        }
    )
    records: list[dict[str, Any]] = []
    records_by_path: dict[Path, list[dict[str, Any]]] = {}
    for path in paths:
        path_records: list[dict[str, Any]] = []
        for line_number, line in enumerate(
            path.read_text(encoding="utf-8").splitlines(),
            start=1,
        ):
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise BackendIntegrationError(
                    f"Malformed compression training evidence {path}:{line_number}."
                ) from exc
            if not isinstance(payload, dict):
                raise BackendIntegrationError(
                    f"Compression training evidence {path}:{line_number} is not a mapping."
                )
            records.append(payload)
            path_records.append(payload)
        records_by_path[path] = path_records
    if not records:
        raise BackendIntegrationError(
            "ms-swift returned successfully without dynamic-compression training evidence."
        )
    wrong_hashes = sorted(
        {
            str(record.get("spec_hash"))
            for record in records
            if record.get("spec_hash") != expected_spec_hash
        }
    )
    if wrong_hashes:
        raise BackendIntegrationError(
            "Compression training evidence does not match the launched stage spec: "
            f"{wrong_hashes}."
        )
    if expected_stages:
        summaries = []
        for stage in expected_stages:
            stage_records = [record for record in records if record.get("stage_name") == stage["stage_name"]]
            summary = _collect_stage_training_evidence(
                paths=paths, records=stage_records,
                expected_runtime_mode=expected_runtime_mode,
                expected_executor_family=str(stage["executor_family"]),
            )
            summaries.append({**dict(stage), **summary})
        result = {
            "verified": True, "executor_family": expected_executor_family if expected_executor_family == "image_operation_multistage" else "omni_multistage",
            "runtime_mode": expected_runtime_mode,
            "evidence_paths": [str(path) for path in paths],
            "stages": summaries,
        }
    else:
        result = _collect_stage_training_evidence(
            paths=paths, records=records,
            expected_runtime_mode=expected_runtime_mode,
            expected_executor_family=expected_executor_family,
        )
    if expected_executor_family in {"image_operation", "image_operation_multistage"}:
        if expected_checkpoint:
            result.update(_collect_image_save_evidence(records))
        else:
            # The native launch explicitly disabled saving. Installation and
            # forward/backward execution were verified above; saved weights and
            # independent parameter updates have deliberately not been verified.
            result.update(checkpoint_verified=False,
                          image_parameter_training_verified=False,
                          verification_scope="native_training_execution_only",
                          checkpoint_status="not_requested")
    if expected_executor_family == "image_fixed_mask" and expected_checkpoint:
        from mllm_comp.fixed_mask.checkpoint import inspect_checkpoint
        saves = [r for r in records if r.get("event") == "save" and r.get("fixed_mask")]
        if not saves or inspect_checkpoint(saves[-1]["output_dir"]) != saves[-1]["fixed_mask"]:
            raise BackendIntegrationError("Fixed-mask training requires its native checkpoint sidecar")
        result.update(checkpoint_verified=True, checkpoint_path=saves[-1]["output_dir"])
    diagnostic_pretty_paths: list[str] = []
    if expected_runtime_mode == "debug":
        for path in paths:
            if not path.name.startswith("debug_diagnostic"):
                continue
            pretty_path = _pretty_training_diagnostic_path(path)
            pretty_records = _pretty_training_diagnostic_records(
                records_by_path[path],
                source_path=path,
            )
            atomic_write_text(
                pretty_path,
                _pretty_diagnostic_json(pretty_records),
            )
            diagnostic_pretty_paths.append(str(pretty_path))
    result["diagnostic_pretty_paths"] = diagnostic_pretty_paths
    return result


def _collect_image_save_evidence(records):
    """Verify global native save evidence, separately from per-stage execution.

    Zero-output/frozen image branches have no independent trainable parameters.
    Otherwise require some finite gradient and update in each independent image
    module, not in every tensor (e.g. zero-initialized LoRA A/B and BF16 rounding).
    """
    from mllm_comp.integrations.ms_swift.split_lifecycle import inspect_split_checkpoint, SIDECARS
    ready = [r for r in records if r.get("event") == "tuner_ready"
             and r.get("scope") == "all_image_stages"]
    saves = [r for r in records if r.get("event") == "save"
             and r.get("scope") == "all_image_stages"]
    if not ready or not saves or "trainable_image_parameters" not in ready[-1]:
        raise BackendIntegrationError("Image training requires tuner and native checkpoint save evidence.")
    save = saves[-1]
    directory = Path(save.get("output_dir", ""))
    try:
        contract = inspect_split_checkpoint(directory)
        if contract is None or contract != save.get("pipeline_contract"):
            raise ValueError("checkpoint contract differs from the save event")
        if any((directory / name).stat().st_size == 0 for name in SIDECARS):
            raise ValueError("empty checkpoint sidecar")
        # Full and PEFT checkpoints can be sharded. Check actual payload files,
        # not just a JSON index that can survive removal of all model weights.
        indices = list(directory.glob("*.index.json"))
        payloads = []
        for index in indices:
            payloads.extend(directory / n for n in json.loads(index.read_text()).get("weight_map", {}).values())
        if not payloads:
            payloads = [directory / n for n in ("model.safetensors", "pytorch_model.bin",
                        "adapter_model.safetensors", "adapter_model.bin") if (directory / n).is_file()]
        if not payloads or any(not p.is_file() or p.stat().st_size == 0 for p in payloads):
            raise ValueError("missing native checkpoint weights")
    except (ValueError, OSError, TypeError) as exc:
        raise BackendIntegrationError(f"Invalid image-operation checkpoint: {exc}") from exc
    names = set(ready[-1]["trainable_image_parameters"])
    updates = {r["parameter"]: r for r in save.get("image_parameter_updates", [])}
    gradients = {r["parameter"]: r for r in save.get("image_parameter_gradients", [])}
    if set(updates) != names or set(gradients) != names:
        raise BackendIntegrationError("Image parameter update/gradient evidence is incomplete.")
    groups = {}
    for name in names:
        grad, update = gradients[name], updates[name]
        norm, delta = grad.get("norm"), update.get("max_abs_delta")
        if (not grad.get("finite") or not isinstance(norm, (int, float)) or not math.isfinite(norm)
                or norm < 0 or not isinstance(delta, (int, float)) or not math.isfinite(delta) or delta < 0):
            raise BackendIntegrationError(f"Image parameter gradient/update is not finite: {name}")
        group = groups.setdefault(name.split(".image_op.")[0], [False, False])
        group[0] |= norm > 0
        group[1] |= update.get("changed") is True and delta > 0
    if any(not gradient or not changed for gradient, changed in groups.values()):
        raise BackendIntegrationError("An independent image parameter group has no nonzero gradient or parameter update.")
    return {"checkpoint_verified": True, "checkpoint_path": str(directory),
            "image_parameter_training_verified": bool(names),
            "image_parameter_training_status": "verified" if names else "not_applicable_no_independent_trainable_parameters",
            "independent_image_parameter_count": len(names)}


def _collect_stage_training_evidence(
    *, paths: Sequence[Path], records: list[dict[str, Any]],
    expected_runtime_mode: str, expected_executor_family: str,
) -> dict[str, Any]:
    installs = [record for record in records if record.get("event") == "install"]
    forwards = [record for record in records if record.get("event") == "forward"]
    backwards = [record for record in records if record.get("event") == "backward"]
    if expected_executor_family == "image_fixed_mask":
        good = [r for r in installs if r.get("executor_family") == "image_fixed_mask"]
        forward = {r["forward_index"] for r in forwards if r.get("loss_requires_grad")
                   and r.get("runtime_mode") == expected_runtime_mode and r.get("use_cache") is False
                   and any(item.get("fixed_mask", {}).get("masked_layer_calls", 0) > 0
                           for item in r.get("layout_records", []))}
        backward = {r["forward_index"] for r in backwards if r.get("gradient_present")}
        if not good or not forward or not forward.issubset(backward):
            raise BackendIntegrationError("Fixed-mask training install/forward/backward evidence is incomplete")
        return {"verified": True, "executor_family": "image_fixed_mask", "runtime_mode": expected_runtime_mode,
                "install_events": len(good), "forward_events": len(forward), "backward_events": len(backward),
                "physical_acceleration": False, "evidence_paths": [str(path) for path in paths]}
    if expected_executor_family == "image_operation":
        good = [r for r in installs if r.get("lifecycle") == "post_load_pre_tuner_pre_optimizer"
                and r.get("replacement_count", 0) > 0
                and r.get("replacement_count") == len(r.get("installed_handles", []))]
        fwd = {r["forward_index"] for r in forwards if r.get("loss_requires_grad")}
        bwd = {r["forward_index"] for r in backwards if r.get("gradient_present")}
        if not good or not fwd or not fwd.issubset(bwd):
            raise BackendIntegrationError("Image operation training install/forward/backward evidence is incomplete.")
        return {"verified": True, "executor_family": "image_operation", "install_events": len(good),
                "forward_events": len(fwd), "backward_events": len(bwd),
                "replacement_count": good[-1]["replacement_count"], "installed_handles": good[-1]["installed_handles"],
                "pre_optimizer_install_verified": True, "runtime_mode": expected_runtime_mode}
    if expected_executor_family == "llm_operation":
        return _collect_operation_training_evidence(
            paths=paths,
            installs=installs,
            forwards=forwards,
            backwards=backwards,
            expected_runtime_mode=expected_runtime_mode,
        )
    if expected_executor_family == "image_multimodal":
        return _collect_image_training_evidence(
            paths=paths,
            installs=installs,
            forwards=forwards,
            backwards=backwards,
            expected_runtime_mode=expected_runtime_mode,
        )
    audio_forwards = [
        record
        for record in forwards
        if int(
            (record.get("compression") or {}).get("original_audio_tokens", 0)
        )
        > 0
    ]
    if not installs or not forwards or not audio_forwards or not backwards:
        raise BackendIntegrationError(
            "Dynamic compressed training evidence is incomplete: expected loader "
            "installation, an audio compression forward, and a loss backward; got "
            f"install={len(installs)}, forward={len(forwards)}, "
            f"audio_forward={len(audio_forwards)}, backward={len(backwards)}."
        )
    non_training_forwards = [
        int(record.get("forward_index", -1))
        for record in forwards
        if record.get("loss_requires_grad") is not True
    ]
    if non_training_forwards:
        raise BackendIntegrationError(
            "Compression training evidence must not contain validation/no-grad "
            f"forwards: {non_training_forwards}."
        )
    backward_indices = {
        int(record["forward_index"])
        for record in backwards
        if record.get("gradient_present") and "forward_index" in record
    }
    audio_forward_indices = {
        int(record["forward_index"])
        for record in audio_forwards
        if "forward_index" in record
    }
    missing_backward_indices = sorted(audio_forward_indices - backward_indices)
    if missing_backward_indices:
        raise BackendIntegrationError(
            "Audio-compressed training forwards are missing backward evidence: "
            f"{missing_backward_indices}."
        )
    template_audio_forwards = [
        record
        for record in audio_forwards
        if record.get("bridge") == "swift_template_pre_forward_hook"
    ]
    if not template_audio_forwards:
        raise BackendIntegrationError(
            "No audio-compressed forward was injected through the ms-swift "
            "multimodal template bridge."
        )
    observed_runtime_modes = {
        str(record.get("runtime_mode")) for record in audio_forwards
    }
    if observed_runtime_modes != {expected_runtime_mode}:
        raise BackendIntegrationError(
            "Compression training runtime-mode evidence does not match the launch: "
            f"expected {expected_runtime_mode!r}, got {sorted(observed_runtime_modes)}."
        )
    if expected_runtime_mode == "profile":
        profile_ok = all(
            isinstance(record.get("timing"), dict)
            and isinstance(record["timing"].get("forward_seconds"), (int, float))
            for record in audio_forwards
        )
        if not profile_ok:
            raise BackendIntegrationError(
                "runtime_mode=profile did not record compressed training timing."
            )
    else:
        profile_ok = all(record.get("timing") is None for record in audio_forwards)
        if not profile_ok:
            raise BackendIntegrationError(
                f"runtime_mode={expected_runtime_mode} unexpectedly recorded profiling data."
            )
    if expected_runtime_mode == "debug":
        debug_ok = all(
            isinstance(record.get("token_debug"), dict)
            and isinstance(record["token_debug"].get("layers"), list)
            for record in audio_forwards
        )
        if not debug_ok:
            raise BackendIntegrationError(
                "runtime_mode=debug did not record per-layer token counts."
            )
    else:
        debug_ok = all(record.get("token_debug") is None for record in audio_forwards)
        if not debug_ok:
            raise BackendIntegrationError(
                f"runtime_mode={expected_runtime_mode} unexpectedly recorded token-debug data."
            )
    original_tokens = sum(
        int((record.get("compression") or {}).get("original_audio_tokens", 0))
        for record in audio_forwards
    )
    kept_tokens = sum(
        int((record.get("compression") or {}).get("kept_audio_tokens", 0))
        for record in audio_forwards
    )
    return {
        "verified": True,
        "evidence_paths": [str(path) for path in paths],
        "install_events": len(installs),
        "forward_events": len(forwards),
        "audio_forward_events": len(audio_forwards),
        "backward_events": len(backwards),
        "audio_backward_verified": True,
        "training_forward_only_verified": True,
        "template_bridge_verified": True,
        "runtime_mode": expected_runtime_mode,
        "profile_evidence_verified": profile_ok,
        "debug_evidence_verified": debug_ok,
        "observed_original_audio_tokens": original_tokens,
        "observed_kept_audio_tokens": kept_tokens,
        "observed_token_reduction": kept_tokens < original_tokens,
        "executor_family": expected_executor_family,
    }


def _collect_operation_training_evidence(
    *,
    paths: Sequence[Path],
    installs: list[dict[str, Any]],
    forwards: list[dict[str, Any]],
    backwards: list[dict[str, Any]],
    expected_runtime_mode: str,
) -> dict[str, Any]:
    operation_installs = [
        item
        for item in installs
        if item.get("executor_family") == "llm_operation"
        and isinstance(item.get("operation"), Mapping)
        and int(item["operation"].get("result_num_hidden_layers", 0)) > 0
    ]
    training_forwards = [
        item
        for item in forwards
        if item.get("loss_requires_grad") is True
        and item.get("executor_family") == "llm_operation"
    ]
    if not operation_installs or not training_forwards or not backwards:
        raise BackendIntegrationError(
            "Operation-compressed training evidence is incomplete: expected a "
            "pre-optimizer structural install, a loss forward and a backward; got "
            f"install={len(operation_installs)}, forward={len(training_forwards)}, "
            f"backward={len(backwards)}."
        )
    forward_indices = {
        int(item["forward_index"])
        for item in training_forwards
        if "forward_index" in item
    }
    backward_indices = {
        int(item["forward_index"])
        for item in backwards
        if item.get("gradient_present") and "forward_index" in item
    }
    missing = sorted(forward_indices - backward_indices)
    if missing:
        raise BackendIntegrationError(
            f"Operation-compressed forwards are missing backward evidence: {missing}."
        )
    modes = {str(item.get("runtime_mode")) for item in training_forwards}
    if modes != {expected_runtime_mode}:
        raise BackendIntegrationError(
            "Operation-compressed runtime-mode evidence does not match the launch: "
            f"expected {expected_runtime_mode!r}, got {sorted(modes)}."
        )
    install = operation_installs[-1]["operation"]
    return {
        "verified": True,
        "executor_family": "llm_operation",
        "evidence_paths": [str(path) for path in paths],
        "diagnostic_pretty_paths": [],
        "install_events": len(operation_installs),
        "forward_events": len(training_forwards),
        "backward_events": len(backwards),
        "operation_backward_verified": True,
        "pre_optimizer_install_verified": True,
        "training_forward_only_verified": True,
        "runtime_mode": expected_runtime_mode,
        "source_num_hidden_layers": install.get("source_num_hidden_layers"),
        "result_num_hidden_layers": install.get("result_num_hidden_layers"),
        "pruned_source_layers": install.get("pruned_source_layers"),
        "parameter_reduction": install.get("parameter_reduction"),
        "requires_checkpoint_export": install.get("requires_checkpoint_export"),
    }


def _collect_image_training_evidence(
    *,
    paths: Sequence[Path],
    installs: list[dict[str, Any]],
    forwards: list[dict[str, Any]],
    backwards: list[dict[str, Any]],
    expected_runtime_mode: str,
) -> dict[str, Any]:
    image_installs = [
        item for item in installs if item.get("executor_family") == "image_multimodal"
    ]
    image_forwards = [
        item
        for item in forwards
        if item.get("executor_family") == "image_multimodal"
        and item.get("loss_requires_grad") is True
    ]
    reduced_forwards = [
        item
        for item in image_forwards
        if any(
            (int(record.get("result_count", 0)) < int(record.get("source_count", 0))
             or record.get("comparison_operation_executed") is True)
            for record in item.get("layout_records", [])
            if isinstance(record, Mapping)
        )
    ]
    if not image_installs or not image_forwards or not reduced_forwards or not backwards:
        raise BackendIntegrationError(
            "Image-compressed training evidence is incomplete: expected an exact "
            "binding install, a token-reducing loss forward and a backward; got "
            f"install={len(image_installs)}, forward={len(image_forwards)}, "
            f"reduced_forward={len(reduced_forwards)}, backward={len(backwards)}."
        )
    forward_indices = {
        int(item["forward_index"])
        for item in reduced_forwards
        if "forward_index" in item
    }
    backward_indices = {
        int(item["forward_index"])
        for item in backwards
        if item.get("gradient_present") and "forward_index" in item
    }
    missing = sorted(forward_indices - backward_indices)
    if missing:
        raise BackendIntegrationError(
            f"Image-compressed forwards are missing backward evidence: {missing}."
        )
    modes = {str(item.get("runtime_mode")) for item in image_forwards}
    if modes != {expected_runtime_mode}:
        raise BackendIntegrationError(
            "Image-compressed runtime-mode evidence does not match the launch: "
            f"expected {expected_runtime_mode!r}, got {sorted(modes)}."
        )
    return {
        "verified": True,
        "executor_family": "image_multimodal",
        "evidence_paths": [str(path) for path in paths],
        "diagnostic_pretty_paths": [],
        "install_events": len(image_installs),
        "forward_events": len(image_forwards),
        "backward_events": len(backwards),
        "image_backward_verified": True,
        "token_reduction_verified": any(
            int(record.get("result_count", 0)) < int(record.get("source_count", 0))
            for item in reduced_forwards for record in item.get("layout_records", [])),
        "comparison_operation_verified": any(
            record.get("comparison_operation_executed") is True
            for item in reduced_forwards for record in item.get("layout_records", [])),
        "teacher_forcing_verified": all(
            item.get("use_cache") is False for item in image_forwards
        ),
        "runtime_mode": expected_runtime_mode,
    }


class MSSwiftTrainAdapter:
    """Delegate the complete training lifecycle to the official Swift CLI."""

    def __init__(
        self,
        *,
        command_runner: Callable[[NativeCLIPlan], NativeCLIResult] = run_native_cli,
    ) -> None:
        self._command_runner = command_runner

    @staticmethod
    def capability_metadata() -> dict[str, Any]:
        """Describe the same backbone policies used to build training plans."""

        return {
            "backbones": list(_BACKBONE_SPECS),
            "trainable_branches": {
                name: list(spec.trainable_branches)
                for name, spec in _BACKBONE_SPECS.items()
            },
            "talker_training": False,
            "compressed_training": {
                "transformers_checkpoint": True,
                "single_stage_only": all(
                    spec.compression_stage_limit == 1 for spec in _BACKBONE_SPECS.values()
                ),
                "stage_limit_by_backbone": {
                    name: spec.compression_stage_limit
                    for name, spec in _BACKBONE_SPECS.items()
                },
                "executor_families": [
                    "projector_token", "encoder_token", "llm_token",
                    "llm_operation", "image_multimodal",
                ],
            },
        }

    @staticmethod
    def validate_config(config: Mapping[str, Any]) -> None:
        branch = config.get("trainable_branch")
        if branch == "talker":
            raise ValueError("The platform ms-swift backend does not support Talker training.")
        if config.get("enable_audio_output") not in (None, False, 0):
            raise ValueError(
                "ENABLE_AUDIO_OUTPUT must be false for platform-controlled ms-swift training."
            )

    @staticmethod
    def _backbone_spec(backbone_name: str) -> _MSSwiftBackboneSpec:
        try:
            return _BACKBONE_SPECS[backbone_name]
        except KeyError as exc:
            supported = ", ".join(sorted(_BACKBONE_SPECS))
            raise BackendIntegrationError(
                f"The ms-swift adapter does not support backbone.name={backbone_name!r}; "
                f"supported backbones: {supported}."
            ) from exc

    @staticmethod
    def _validate_backbone_policy(
        spec: _MSSwiftBackboneSpec,
        *,
        trainable_branch: str,
        enable_audio_output: Any,
        audio_output_was_declared: bool,
    ) -> None:
        if trainable_branch not in spec.trainable_branches:
            allowed = ", ".join(spec.trainable_branches)
            raise BackendIntegrationError(
                f"ms-swift backbone {spec.backbone_name!r} requires "
                f"trainable_branch to be one of: {allowed}."
            )
        if enable_audio_output not in (None, False, 0):
            raise BackendIntegrationError(
                "ENABLE_AUDIO_OUTPUT must be false for platform-controlled ms-swift training."
            )
        if spec.backbone_name != "qwen25_omni" and audio_output_was_declared:
            raise BackendIntegrationError(
                "train.args.enable_audio_output is a Qwen2.5-Omni-only control and "
                f"does not apply to {spec.backbone_name}."
            )

    @staticmethod
    def dependency_report() -> dict[str, Any]:
        versions = _swift_versions()
        executable = shutil.which("swift")
        return {
            "available": (
                versions["ms_swift"] != "not-installed" and executable is not None
            ),
            "version": versions["ms_swift"],
            "versions": versions,
            "executable": executable,
            "model_load_attempted": False,
        }

    def build_launch_plan(self, config: Any) -> MSSwiftLaunchPlan:
        spec = self._backbone_spec(config.backbone.name)
        train = config.train
        extra = dict(train.args)
        conflicts = sorted(_PLATFORM_OWNED_SWIFT_KEYS.intersection(extra))
        if conflicts:
            raise BackendIntegrationError(
                "train.args cannot override platform-owned ms-swift arguments: "
                f"{conflicts}."
            )
        native_argv = extra.pop("native_argv", [])
        if not isinstance(native_argv, list) or any(
            not isinstance(item, str) for item in native_argv
        ):
            raise BackendIntegrationError("train.args.native_argv must be a list of strings.")
        # Native argv is appended last and keeps its explicit override. Avoid
        # adding a competing default when it already supplies the seed.
        if any(arg == "--seed" or arg.startswith("--seed=") for arg in native_argv):
            extra.pop("seed", None)
        else:
            extra.setdefault("seed", config.experiment.seed)
        external_plugins_value = extra.pop("external_plugins", [])
        external_plugins = _as_string_list(
            external_plugins_value,
            name="external_plugins",
        )
        trainable_branch = str(
            extra.pop("trainable_branch", spec.default_trainable_branch)
        )
        audio_output_was_declared = "enable_audio_output" in extra
        enable_audio_output = extra.pop("enable_audio_output", None)
        self._validate_backbone_policy(
            spec,
            trainable_branch=trainable_branch,
            enable_audio_output=enable_audio_output,
            audio_output_was_declared=audio_output_was_declared,
        )
        artifact_format_value = extra.pop("artifact_format", None)
        artifact_format = (
            str(artifact_format_value) if artifact_format_value is not None else None
        )
        if artifact_format not in {None, "transformers"}:
            raise BackendIntegrationError(
                "train.args.artifact_format must be 'transformers' when set."
            )
        media_environment, unset_environment = (
            _qwen25_omni_media_overrides(
                backbone_name=spec.backbone_name,
                backbone_config=config.backbone.config,
                swift_args=extra,
            )
        )

        compression_mode = str(
            getattr(train.compression_mode, "value", train.compression_mode)
        )
        compression_training: dict[str, Any] = {"enabled": False}
        if compression_mode == CompressionMode.COMPRESSED.value:
            encoded_spec, compression_training = _training_compression_spec(config)

        model_path = (
            train.checkpoint_in or config.backbone.model_path or config.backbone.model_id
        )
        if not model_path:
            raise BackendIntegrationError(
                "ms-swift training needs train.checkpoint_in, backbone.model_path or backbone.model_id."
            )
        dataset_value = extra.pop("datasets", None)
        if dataset_value is None:
            dataset_value = config.data.dataset_id
        datasets = _as_string_list(dataset_value, name="dataset")
        if not datasets or any(not item for item in datasets):
            raise BackendIntegrationError(
                "ms-swift training requires at least one non-empty dataset."
            )

        output_dir = str(Path(config.output.run_dir).resolve())
        environment = dict(spec.environment)
        environment.update(media_environment)
        if compression_mode == CompressionMode.COMPRESSED.value:
            plugin_path = str(Path(__file__).with_name("training_plugin.py").resolve())
            evidence_dir = output_dir
            external_plugins = [plugin_path, *external_plugins]
            environment["MLLM_COMP_TRAIN_COMPRESSION_SPEC"] = encoded_spec
            environment["MLLM_COMP_TRAIN_COMPRESSION_EVIDENCE_DIR"] = evidence_dir
            compression_training.update(
                {
                    "plugin_path": plugin_path,
                    "evidence_dir": evidence_dir,
                }
            )
        if os.environ.get(_VSCODE_DEBUG_SUBPROCESSES_ENV) == "1":
            # debugpy can reliably auto-attach only when the first child process is
            # launched through the active Python interpreter. ``swift sft`` is a
            # console-script shim, whereas this module invocation reaches the same
            # upstream dispatcher without changing normal training launches.
            argv = [sys.executable, "-m", "swift.cli.main", "sft"]
        else:
            argv = ["swift", "sft"]
        _append_argument(argv, "model", str(model_path))
        _append_argument(argv, "model_type", spec.model_type)
        _append_argument(argv, "template", spec.template)
        _append_argument(argv, "dataset", datasets)
        _append_argument(argv, "output_dir", output_dir)
        _append_argument(argv, "max_steps", train.max_steps)
        _append_argument(argv, "resume_from_checkpoint", train.resume)
        if external_plugins:
            _append_argument(argv, "external_plugins", external_plugins)
        for name, value in extra.items():
            if name not in _PLATFORM_ONLY_KEYS:
                _append_argument(argv, name, value)
        argv.extend(native_argv)
        return MSSwiftLaunchPlan(
            argv=argv,
            environment=environment,
            unset_environment=unset_environment,
            cwd=resolved_cwd(),
            expected_output_paths=[
                str(Path(output_dir) / "**" / "checkpoint-*"),
                str(Path(output_dir) / "**" / "logging.jsonl"),
                str(Path(output_dir) / "**" / "trainer_state.json"),
            ],
            model_path=str(model_path),
            datasets=datasets,
            output_dir=output_dir,
            compression_mode=compression_mode,
            artifact_format=artifact_format,
            backbone_name=spec.backbone_name,
            model_type=spec.model_type,
            template=spec.template,
            trainable_branch=trainable_branch,
            talker_training=False,
            compression_training=compression_training,
        )

    def train(self, *, config: Any, manifest: Any | None = None) -> MSSwiftTrainResult:
        del manifest
        plan = self.build_launch_plan(config)
        process_result = self._command_runner(plan)
        if process_result.returncode != 0:
            raise NativeCLIProcessError("ms-swift", process_result.returncode)
        checkpoint, logging_path, metrics, raw_result = _collect_swift_outputs(
            Path(plan.output_dir)
        )
        compression_training = dict(plan.compression_training)
        if compression_training.get("enabled"):
            if logging_path is None:
                raise BackendIntegrationError(
                    "Compressed ms-swift training did not produce logging.jsonl, so "
                    "its native artifact directory cannot be resolved."
                )
            evidence_dir = logging_path.parent
            compression_training["evidence_dir"] = str(evidence_dir)
            evidence = _collect_training_compression_evidence(
                evidence_dir,
                expected_spec_hash=str(compression_training["spec_hash"]),
                expected_runtime_mode=str(compression_training["runtime_mode"]),
                expected_executor_family=str(
                    compression_training["executor_family"]
                ),
                expected_stages=compression_training.get("stages", ()),
                expected_checkpoint=config.train.args.get("save_strategy") != "no",
            )
            compression_training.update(evidence)
            raw_result["compression_training"] = evidence
        versions = _swift_versions()
        backend_args = {
            "argv": list(plan.argv),
            "environment": dict(plan.environment),
            "unset_environment": list(plan.unset_environment),
            "cwd": plan.cwd,
            "versions": versions,
            "model_path": plan.model_path,
            "datasets": list(plan.datasets),
            "compression_mode": plan.compression_mode,
            "artifact_format": plan.artifact_format,
            "backbone_name": plan.backbone_name,
            "model_type": plan.model_type,
            "template": plan.template,
            "trainable_branch": plan.trainable_branch,
            "talker_training": plan.talker_training,
            "compression_training": compression_training,
            "output_dir": plan.output_dir,
            "checkpoint_path": str(checkpoint),
            "logging_path": str(logging_path) if logging_path is not None else None,
        }
        return MSSwiftTrainResult(
            checkpoint_path=str(checkpoint),
            train_log_path=(str(logging_path) if logging_path is not None else None),
            metrics=metrics,
            backend_args=backend_args,
            backend=BackendRecord(
                name="ms_swift",
                version=versions["ms_swift"],
                config_path=config.path,
                args=backend_args,
            ),
            raw_result=raw_result,
        )
