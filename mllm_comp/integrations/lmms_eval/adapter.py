"""Native-CLI-first lmms-eval integration.

Evaluation lifecycle, logging, progress, caching, result persistence, and table
rendering remain owned by lmms-eval.  This module only builds the upstream
command, writes the platform model-plugin specification, launches the command,
and collects its standard result files afterwards.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from collections import Counter
from copy import deepcopy
from dataclasses import dataclass, field
from importlib import metadata
import json
import os
from pathlib import Path
import sys
from typing import Any, Callable, Iterable

from mllm_comp.analysis.records import AnalysisSample
from mllm_comp.backbones.base import BackboneFacade, BackboneRuntime
from mllm_comp.config import config_to_dict
from mllm_comp.core.errors import BackendIntegrationError
from mllm_comp.core.manifest import BackendRecord, MetricRecord
from mllm_comp.core.serialization import (
    atomic_write_json,
    to_jsonable,
)
from mllm_comp.core.vocabulary import RuntimeMode
from mllm_comp.integrations.lmms_eval.diagnostics import (
    _pretty_diagnostic_path,
    _materialize_pretty_diagnostics,
)
from mllm_comp.integrations.native_cli import (
    NativeCLIPlan,
    NativeCLIProcessError,
    NativeCLIResult,
    installed_versions,
    resolved_cwd,
    run_native_cli,
)

from .model_bridge import LMMSEvalRequestBridge


_PLUGIN_ENTRYPOINT = "mllm_comp.integrations.lmms_eval.plugin:get_model_manifests"
_PLUGIN_MODEL_IDS = {
    "llava_onevision": "mllm_comp_llava_onevision",
    "qwen25_omni": "qwen2_5_omni",
    "qwen25_vl": "mllm_comp_qwen25_vl",
    "qwen3_vl": "mllm_comp_qwen3_vl",
    "qwen3_5": "mllm_comp_qwen3_5",
}
_PLATFORM_OWNED_ARGS = {
    "model",
    "model_args",
    "tasks",
    "batch_size",
    "device",
    "limit",
    "output_path",
}
_PLUGIN_ONLY_ARGS = {
    "system_prompt",
    "use_audio_in_video",
    "diagnostic_trace_path",
    "forward_only_probe",
}
_UPSTREAM_CLI_GAPS = {
    "bootstrap_iters": (
        "lmms-eval 0.7.2 exposes bootstrap_iters in simple_evaluate(), but not "
        "in its native CLI parser. Remove eval.args.bootstrap_iters or use an "
        "upstream version that provides the CLI flag."
    ),
}


def _analysis_target(
    task: Any,
    raw_doc: Any,
    *,
    task_name: str,
) -> tuple[str | None, str | None]:
    """Resolve one deterministic textual reference without inventing a target."""

    # The upstream TextVQA task config currently declares doc_to_target="answer",
    # while its validation records contain the ten human references in "answers".
    # lmms-eval consequently renders the literal word "answer" as a target.
    # Prefer the actual reference list and select a deterministic plurality label.
    if "textvqa" in task_name.casefold() and isinstance(raw_doc, Mapping):
        references = raw_doc.get("answers")
        if isinstance(references, Sequence) and not isinstance(references, (str, bytes)):
            cleaned = [str(value).strip() for value in references if str(value).strip()]
            if cleaned:
                normalized = [_normalize_textvqa_reference(value) for value in cleaned]
                counts = Counter(normalized)
                winner = max(counts, key=counts.get)  # ties follow first appearance
                target = next(value for value, key in zip(cleaned, normalized) if key == winner)
                return target, None

    callback = getattr(task, "doc_to_target", None)
    if not callable(callback):
        return None, (
            f"Analysis task {task_name!r} does not expose doc_to_target."
        )
    value = callback(raw_doc)
    if not isinstance(value, str):
        return None, (
            f"Analysis task {task_name!r} returned non-string doc_to_target "
            f"type {type(value).__name__!r}."
        )
    if not value.strip():
        return None, (
            f"Analysis task {task_name!r} returned an empty doc_to_target."
        )
    return value, None


def _normalize_textvqa_reference(value: str) -> str:
    """Normalize only case, punctuation, and whitespace for plurality counting."""

    import re
    import unicodedata

    value = unicodedata.normalize("NFKC", value).casefold()
    value = "".join(" " if unicodedata.category(char).startswith("P") else char for char in value)
    return re.sub(r"\s+", " ", value).strip()


@dataclass(frozen=True)
class LMMSEvalLaunchPlan(NativeCLIPlan):
    """Resolved official lmms-eval invocation and platform plugin inputs."""

    model_name: str
    checkpoint_path: str
    output_path: str
    plugin_spec_path: str
    plugin_spec: dict[str, Any]
    lmms_eval_argv: list[str]
    num_processes: int
    distributed_launcher: str
    main_process_port: int | None


@dataclass
class EvalResult:
    metrics: list[MetricRecord]
    raw_output_path: str
    sample_output_paths: list[str] = field(default_factory=list)
    backend_args: dict[str, Any] = field(default_factory=dict)
    backend: BackendRecord | None = None


def _entrypoint_registered() -> bool:
    entrypoints = metadata.entry_points()
    selected = (
        entrypoints.select(group="lmms_eval.models")
        if hasattr(entrypoints, "select")
        else entrypoints.get("lmms_eval.models", [])
    )
    return any(
        getattr(entrypoint, "value", None) == _PLUGIN_ENTRYPOINT
        for entrypoint in selected
    )


def _metric_records(raw: dict[str, Any]) -> list[MetricRecord]:
    task_results = raw.get("results", {})
    if not isinstance(task_results, dict):
        raise BackendIntegrationError("lmms-eval output field 'results' must be a mapping.")
    configs = raw.get("configs", {}) if isinstance(raw.get("configs"), dict) else {}
    sample_counts = (
        raw.get("n-samples", {}) if isinstance(raw.get("n-samples"), dict) else {}
    )
    records: list[MetricRecord] = []
    for task, values in task_results.items():
        if not isinstance(values, dict):
            continue
        metrics = {
            str(name): float(value)
            for name, value in values.items()
            if isinstance(value, (int, float)) and not isinstance(value, bool)
        }
        task_config = configs.get(task, {}) if isinstance(configs.get(task), dict) else {}
        split = next(
            (
                str(task_config[key])
                for key in ("test_split", "validation_split", "split")
                if task_config.get(key) is not None
            ),
            None,
        )
        count = sample_counts.get(task, {})
        effective = count.get("effective") if isinstance(count, dict) else None
        records.append(
            MetricRecord(
                task=str(task),
                split=split,
                metrics=metrics,
                num_samples=int(effective) if effective is not None else None,
            )
        )
    return records


def _native_arg_values(value: Any, *, name: str) -> list[str]:
    if isinstance(value, bool):
        return [] if value else []
    if isinstance(value, (str, int, float)):
        return [str(value)]
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        if not all(isinstance(item, (str, int, float)) for item in value):
            raise BackendIntegrationError(
                f"lmms-eval argument {name!r} must contain only scalar values."
            )
        return [",".join(str(item) for item in value)]
    raise BackendIntegrationError(
        f"lmms-eval argument {name!r} must be a scalar, bool, or list of scalars."
    )


def _append_native_arg(argv: list[str], name: str, value: Any) -> None:
    if value is None or value is False:
        return
    if not name or name.startswith("-") or any(char.isspace() for char in name):
        raise BackendIntegrationError(f"Invalid lmms-eval argument name {name!r}.")
    argv.append(f"--{name}")
    if value is not True:
        argv.extend(_native_arg_values(value, name=name))


def _distributed_settings(config: Any, *, device: str | None) -> tuple[int, str, int | None]:
    raw_num_processes = getattr(config.eval, "num_processes", 1)
    if isinstance(raw_num_processes, bool) or not isinstance(raw_num_processes, int):
        raise BackendIntegrationError("eval.num_processes must be an integer.")
    if raw_num_processes < 1:
        raise BackendIntegrationError("eval.num_processes must be at least 1.")

    launcher = str(getattr(config.eval, "distributed_launcher", "accelerate"))
    if launcher not in {"accelerate", "torchrun"}:
        raise BackendIntegrationError(
            "eval.distributed_launcher must be 'accelerate' or 'torchrun'."
        )

    raw_port = getattr(config.eval, "main_process_port", None)
    if raw_port is not None:
        if isinstance(raw_port, bool) or not isinstance(raw_port, int):
            raise BackendIntegrationError("eval.main_process_port must be an integer.")
        if not 1 <= raw_port <= 65535:
            raise BackendIntegrationError(
                "eval.main_process_port must be between 1 and 65535."
            )

    if raw_num_processes > 1 and device is not None:
        raise BackendIntegrationError(
            "Do not pass --device for distributed eval. Each worker must use the "
            "device selected from its LOCAL_RANK by Accelerate/Torchrun."
        )

    visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
    if raw_num_processes > 1 and visible_devices is not None:
        visible = [item.strip() for item in visible_devices.split(",") if item.strip()]
        if visible_devices.strip() == "-1":
            visible = []
        if len(visible) < raw_num_processes:
            raise BackendIntegrationError(
                "eval.num_processes exceeds CUDA_VISIBLE_DEVICES: "
                f"requested {raw_num_processes}, visible {len(visible)}."
            )

    return raw_num_processes, launcher, raw_port


def _distributed_argv(
    lmms_eval_argv: list[str],
    *,
    num_processes: int,
    launcher: str,
    main_process_port: int | None,
) -> list[str]:
    if num_processes == 1:
        return list(lmms_eval_argv)

    child_argv = lmms_eval_argv[3:]
    if launcher == "accelerate":
        argv = [
            sys.executable,
            "-m",
            "accelerate.commands.launch",
            "--multi_gpu",
            "--num_processes",
            str(num_processes),
        ]
        if main_process_port is not None:
            argv.extend(["--main_process_port", str(main_process_port)])
    else:
        argv = [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--nproc_per_node",
            str(num_processes),
        ]
        if main_process_port is not None:
            argv.extend(["--master_port", str(main_process_port)])
    argv.extend(["-m", "lmms_eval", *child_argv])
    return argv


def _result_snapshot(output_path: Path) -> dict[Path, int]:
    if not output_path.exists():
        return {}
    return {
        path: path.stat().st_mtime_ns
        for path in output_path.rglob("*_results.json")
        if path.is_file()
    }


def _collect_native_result(
    output_path: Path,
    before: Mapping[Path, int],
) -> tuple[Path, list[Path], dict[str, Any]]:
    candidates = [
        path
        for path in output_path.rglob("*_results.json")
        if path.is_file() and before.get(path) != path.stat().st_mtime_ns
    ]
    if not candidates:
        raise BackendIntegrationError(
            "lmms-eval exited successfully but did not create or update a native "
            f"aggregated result under {output_path}."
        )
    result_path = max(candidates, key=lambda path: path.stat().st_mtime_ns)
    try:
        raw = json.loads(result_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise BackendIntegrationError(
            f"Cannot read lmms-eval native result {result_path}: {exc}"
        ) from exc
    if not isinstance(raw, dict):
        raise BackendIntegrationError("lmms-eval aggregated result must be a JSON object.")
    prefix = result_path.name.removesuffix("_results.json")
    samples = sorted(result_path.parent.glob(f"{prefix}_samples_*.jsonl"))
    return result_path, samples, raw


def _diagnostic_trace_paths(plan: LMMSEvalLaunchPlan) -> list[Path]:
    """Resolve the exact per-rank diagnostic files owned by one launch."""

    value = plan.plugin_spec.get("diagnostic_trace_path")
    if not isinstance(value, str) or not value:
        return []
    path = Path(value)
    if plan.num_processes == 1:
        return [path]
    suffix = path.suffix
    stem = path.name[: -len(suffix)] if suffix else path.name
    return [
        path.with_name(f"{stem}.rank{rank}{suffix}")
        for rank in range(plan.num_processes)
    ]


def _compression_manifest_paths(plan: LMMSEvalLaunchPlan) -> list[Path]:
    """Resolve per-rank compression lifecycle evidence files."""

    value = plan.plugin_spec.get("compression_manifest_path")
    if not isinstance(value, str) or not value:
        return []
    path = Path(value)
    if plan.num_processes == 1:
        return [path]
    suffix = path.suffix
    stem = path.name[: -len(suffix)] if suffix else path.name
    return [
        path.with_name(f"{stem}.rank{rank}{suffix}")
        for rank in range(plan.num_processes)
    ]


def _has_compression_invocation(payload: Mapping[str, Any]) -> bool:
    """Return whether a runtime manifest proves a dynamic callback executed."""

    runtime = payload.get("runtime")
    if not isinstance(runtime, Mapping):
        return False
    activations = runtime.get("activations")
    if not isinstance(activations, Sequence) or isinstance(
        activations, (str, bytes, bytearray)
    ):
        return False
    for activation in activations:
        if not isinstance(activation, Mapping):
            continue
        invocation_count = activation.get("invocation_count", 0)
        records = activation.get("layout_delta_records")
        record_count = activation.get("layout_delta_records_count")
        if record_count is None:
            record_count = len(records) if isinstance(records, (list, tuple)) else 0
        if (
            isinstance(invocation_count, int)
            and invocation_count > 0
            and isinstance(record_count, int)
            and record_count > 0
        ):
            return True
    return False


class LMMSEvalAdapter:
    """Launch lmms-eval and expose a platform runtime through its model plugin."""

    def __init__(
        self,
        *,
        command_runner: Callable[[NativeCLIPlan], NativeCLIResult] = run_native_cli,
        task_manager_cls: type[Any] | None = None,
        task_dict_factory: Callable[..., dict[str, Any]] | None = None,
    ) -> None:
        self._command_runner = command_runner
        self._task_manager_cls = task_manager_cls
        self._task_dict_factory = task_dict_factory

    @staticmethod
    def validate_config(config: dict[str, Any]) -> None:
        if config.get("return_audio") is True:
            raise ValueError(
                "The platform lmms-eval plugin evaluates text responses only; "
                "return_audio must be false."
            )

    @staticmethod
    def dependency_report() -> dict[str, Any]:
        versions = installed_versions(
            primary={
                "lmms_eval": ("lmms-eval", "lmms_eval"),
                "accelerate": ("accelerate",),
            }
        )
        plugin_registered = _entrypoint_registered()
        return {
            "available": (
                versions["lmms_eval"] != "not-installed"
                and versions["accelerate"] != "not-installed"
                and plugin_registered
            ),
            "version": versions["lmms_eval"],
            "versions": versions,
            "plugin_entrypoint": _PLUGIN_ENTRYPOINT,
            "plugin_registered": plugin_registered,
            "model_load_attempted": False,
        }

    def build_launch_plan(
        self,
        config: Any,
        *,
        checkpoint_path: str | None = None,
        device: str | None = None,
    ) -> LMMSEvalLaunchPlan:
        if device is None:
            device = config.backbone.config.get("device")
        tasks = list(config.eval.tasks)
        if not tasks:
            raise BackendIntegrationError("eval.tasks must contain at least one lmms-eval task.")
        try:
            model_name = _PLUGIN_MODEL_IDS[config.backbone.name]
        except KeyError as exc:
            supported = ", ".join(sorted(_PLUGIN_MODEL_IDS))
            raise BackendIntegrationError(
                f"The lmms-eval plugin does not map backbone.name={config.backbone.name!r}; "
                f"supported backbones: {supported}."
            ) from exc

        resolved_checkpoint = (
            checkpoint_path
            or config.eval.checkpoint_path
            or config.backbone.model_path
            or config.backbone.model_id
        )
        if not resolved_checkpoint:
            raise BackendIntegrationError(
                "lmms-eval needs eval.checkpoint_path, backbone.model_path, or backbone.model_id."
            )
        num_processes, distributed_launcher, main_process_port = _distributed_settings(
            config,
            device=device,
        )

        framework_args = dict(config.eval.args)
        lmms_seed = framework_args.pop("seed", config.experiment.seed)
        for name, message in _UPSTREAM_CLI_GAPS.items():
            if name in framework_args:
                raise BackendIntegrationError(message)
        conflicts = sorted(_PLATFORM_OWNED_ARGS.intersection(framework_args))
        if conflicts:
            raise BackendIntegrationError(
                "eval.args cannot override platform-owned lmms-eval arguments: "
                f"{conflicts}."
            )
        if "result_filename" in framework_args:
            raise BackendIntegrationError(
                "eval.args.result_filename belonged to the retired direct-API bridge. "
                "Remove it; native lmms-eval owns aggregated result filenames."
            )
        native_argv = framework_args.pop("native_argv", [])
        if not isinstance(native_argv, list) or any(
            not isinstance(item, str) for item in native_argv
        ):
            raise BackendIntegrationError("eval.args.native_argv must be a list of strings.")
        force_simple = framework_args.pop("force_simple", False)
        if not isinstance(force_simple, bool):
            raise BackendIntegrationError("eval.args.force_simple must be a boolean.")
        plugin_values = {
            name: framework_args.pop(name, None) for name in _PLUGIN_ONLY_ARGS
        }
        forward_only_probe = plugin_values["forward_only_probe"]
        if forward_only_probe is None:
            forward_only_probe = False
        if not isinstance(forward_only_probe, bool):
            raise BackendIntegrationError(
                "eval.args.forward_only_probe must be a boolean."
            )
        if forward_only_probe and config.backbone.name == "qwen25_omni":
            raise BackendIntegrationError(
                "forward_only_probe is scoped to the ImageLLM LLaVA/Qwen-VL "
                "backbones, not qwen25_omni."
            )
        if forward_only_probe and config.eval.runtime_batch_size != 1:
            raise BackendIntegrationError(
                "forward_only_probe requires eval.runtime_batch_size=1."
            )

        cwd = resolved_cwd()
        output_path = str(Path(config.output.run_dir).resolve())
        plugin_spec_path = str(
            (Path(output_path) / "mllm_comp_lmms_plugin.json").resolve()
        )
        compression_mode = str(
            getattr(config.eval.compression_mode, "value", config.eval.compression_mode)
        )
        runtime_mode = RuntimeMode(config.eval.runtime_mode)
        compression_manifest_path = (
            str((Path(output_path) / "compression_runtime.json").resolve())
            if compression_mode == "compressed"
            else None
        )
        if (
            runtime_mode is not RuntimeMode.DEFAULT
            and config.backbone.name != "qwen25_omni"
        ):
            raise BackendIntegrationError(
                "profile/debug runtime modes are currently implemented only for qwen25_omni."
            )
        diagnostic_trace_path = plugin_values["diagnostic_trace_path"]
        if runtime_mode is RuntimeMode.DEFAULT and diagnostic_trace_path is not None:
            raise BackendIntegrationError(
                "eval.args.diagnostic_trace_path requires runtime_mode=profile or debug."
            )
        if (
            diagnostic_trace_path is None
            and config.backbone.name == "qwen25_omni"
            and runtime_mode is not RuntimeMode.DEFAULT
        ):
            if runtime_mode is RuntimeMode.DEBUG:
                diagnostic_name = "debug_diagnostic.jsonl"
            else:
                diagnostic_name = (
                    "compression_diagnostic.jsonl"
                    if compression_mode == "compressed"
                    else "baseline_diagnostic.jsonl"
                )
            diagnostic_trace_path = str(
                (Path(output_path) / diagnostic_name).resolve()
            )
        elif diagnostic_trace_path is not None:
            diagnostic_trace_path = str(Path(diagnostic_trace_path).resolve())
        plugin_spec = {
            "schema_version": 2,
            "config_path": str(Path(config.path).resolve()) if config.path else None,
            "effective_config": config_to_dict(config),
            "checkpoint_path": str(resolved_checkpoint),
            "device": device,
            "expected_world_size": num_processes,
            "distributed_launcher": (
                distributed_launcher if num_processes > 1 else None
            ),
            "backbone_name": config.backbone.name,
            "compression_mode": compression_mode,
            "compression_manifest_path": compression_manifest_path,
            "runtime_mode": runtime_mode.value,
            "request_mode": (
                "forward_probe" if forward_only_probe else "generate"
            ),
            "generation": to_jsonable(dict(config.eval.generation)),
            "system_prompt": plugin_values["system_prompt"],
            "use_audio_in_video": bool(plugin_values["use_audio_in_video"] or False),
            "diagnostic_trace_path": diagnostic_trace_path,
        }
        model_args = (
            f"pretrained={resolved_checkpoint},mllm_comp_spec={plugin_spec_path}"
        )
        lmms_eval_argv = [
            sys.executable,
            "-m",
            "lmms_eval",
            "--model",
            model_name,
            "--model_args",
            model_args,
            "--tasks",
            ",".join(tasks),
            "--batch_size",
            str(config.eval.runtime_batch_size),
            "--output_path",
            output_path,
            "--seed",
        ]
        seed_values = _native_arg_values(lmms_seed, name="seed")
        if len(seed_values) != 1:
            raise BackendIntegrationError(
                "eval.args.seed must resolve to one lmms-eval seed argument."
            )
        lmms_eval_argv.append(seed_values[0])
        if force_simple:
            lmms_eval_argv.append("--force_simple")
        if device:
            lmms_eval_argv.extend(["--device", device])
        if config.eval.sample_limit is not None:
            lmms_eval_argv.extend(["--limit", str(config.eval.sample_limit)])
        for name, value in framework_args.items():
            _append_native_arg(lmms_eval_argv, name, value)
        lmms_eval_argv.extend(native_argv)
        argv = _distributed_argv(
            lmms_eval_argv,
            num_processes=num_processes,
            launcher=distributed_launcher,
            main_process_port=main_process_port,
        )
        launch_environment: dict[str, str] = {}
        if "CUDA_VISIBLE_DEVICES" in os.environ:
            launch_environment["CUDA_VISIBLE_DEVICES"] = os.environ[
                "CUDA_VISIBLE_DEVICES"
            ]
        expected_output_paths = [
            str(Path(output_path) / "<model>" / "*_results.json"),
            str(Path(output_path) / "<model>" / "*_samples_*.jsonl"),
        ]
        if diagnostic_trace_path is not None:
            trace_path = Path(diagnostic_trace_path)
            if num_processes > 1:
                suffix = trace_path.suffix
                stem = (
                    trace_path.name[: -len(suffix)]
                    if suffix
                    else trace_path.name
                )
                expected_output_paths.append(
                    str(trace_path.with_name(f"{stem}.rank*{suffix}"))
                )
                expected_output_paths.append(
                    str(trace_path.with_name(f"{stem}.rank*.pretty.json"))
                )
            else:
                expected_output_paths.append(str(trace_path))
                expected_output_paths.append(str(_pretty_diagnostic_path(trace_path)))
        if compression_manifest_path is not None:
            manifest_path = Path(compression_manifest_path)
            if num_processes > 1:
                suffix = manifest_path.suffix
                stem = (
                    manifest_path.name[: -len(suffix)]
                    if suffix
                    else manifest_path.name
                )
                expected_output_paths.append(
                    str(manifest_path.with_name(f"{stem}.rank*{suffix}"))
                )
            else:
                expected_output_paths.append(str(manifest_path))
        return LMMSEvalLaunchPlan(
            argv=argv,
            environment=launch_environment,
            cwd=cwd,
            expected_output_paths=expected_output_paths,
            model_name=model_name,
            checkpoint_path=str(resolved_checkpoint),
            output_path=output_path,
            plugin_spec_path=plugin_spec_path,
            plugin_spec=plugin_spec,
            lmms_eval_argv=lmms_eval_argv,
            num_processes=num_processes,
            distributed_launcher=distributed_launcher,
            main_process_port=main_process_port,
        )

    def iter_analysis_samples(
        self,
        *,
        config: Any,
        backbone: BackboneFacade,
        runtime: BackboneRuntime,
        sample_limit: int | None,
        requires_teacher_forcing: bool = False,
    ) -> Iterable[AnalysisSample]:
        """Keep the analysis-only task iterator independent from eval launching."""

        TaskManager = self._task_manager_cls
        get_task_dict = self._task_dict_factory
        if TaskManager is None or get_task_dict is None:
            try:
                from lmms_eval.tasks import TaskManager as ImportedTaskManager
                from lmms_eval.tasks import get_task_dict as imported_get_task_dict
            except ImportError as exc:
                raise BackendIntegrationError("lmms-eval is not importable.") from exc
            TaskManager = ImportedTaskManager
            get_task_dict = imported_get_task_dict
        tasks = list(config.analysis.tasks)
        if not tasks:
            raise BackendIntegrationError(
                "analysis.tasks must contain at least one lmms-eval task."
            )
        model_name = _PLUGIN_MODEL_IDS.get(
            backbone.capabilities().name,
            backbone.capabilities().name,
        )
        task_include_path = getattr(config.analysis, "task_include_path", [])
        manager = TaskManager(
            model_name=model_name,
            include_path=task_include_path or None,
            include_defaults=not bool(task_include_path),
        )
        if bool(getattr(config.analysis, "streaming_datasets", False)):
            # Analysis pilots typically read only a few records from large
            # image datasets. Load the platform-selected task configs as
            # streams so `sample_limit` avoids materializing whole datasets.
            task_dict = {}
            for task_name in tasks:
                task_config = manager._get_config(task_name)
                if not task_config or not isinstance(task_config, dict):
                    raise BackendIntegrationError(
                        f"Streaming analysis requires a leaf task config for {task_name!r}."
                    )
                dataset_kwargs = dict(task_config.get("dataset_kwargs", {}))
                dataset_kwargs.update({"streaming": True, "num_proc": None})
                task_config["dataset_kwargs"] = dataset_kwargs
                task_dict.update(manager.load_config(task_config))
        else:
            task_dict = get_task_dict(tasks, manager, "simple")
        bridge = LMMSEvalRequestBridge(
            facade=backbone,
            runtime=runtime,
            use_audio_in_video=bool(
                config.backbone.config.get("use_audio_in_video", False)
            ),
        )
        limit = sample_limit if sample_limit is not None else config.analysis.sample_limit
        sampling = dict(getattr(config.analysis, "sampling", {}) or {})
        excluded_images: set[str] = set()
        exclusion_sources: list[dict[str, str]] = []
        if sampling:
            from mllm_comp.analysis.common.discovery_sampling import load_excluded_images
            unknown = set(sampling) - {"unique_images_per_task", "seed", "exclude_samples", "manifest_path", "manifest_stage", "manifest_shard"}
            if unknown:
                raise BackendIntegrationError(f"Unknown analysis sampling fields: {sorted(unknown)}")
            paths = sampling.get("exclude_samples", [])
            if not isinstance(paths, list) or any(not isinstance(path, str) for path in paths):
                raise BackendIntegrationError("analysis.sampling.exclude_samples must be a list of paths.")
            excluded_images, exclusion_sources = load_excluded_images(paths)

        def leaf_tasks(values: dict[str, Any]) -> Iterable[tuple[str, Any]]:
            for task_name, task_value in values.items():
                if isinstance(task_value, dict):
                    yield from leaf_tasks(task_value)
                    continue
                if isinstance(task_value, tuple):
                    _, task_value = task_value
                if task_value is not None:
                    yield str(task_name), task_value

        for task_name, task in leaf_tasks(task_dict):
            output_type = str(task.get_config("output_type"))
            if "generate_until" not in output_type:
                raise BackendIntegrationError(
                    f"Analysis task {task_name!r} uses unsupported output_type={output_type!r}."
                )
            documents = task.doc_iterator(limit=limit)
            selection_metadata: dict[str, Any] = {}
            manifest_entries = {}
            if sampling.get("manifest_path"):
                from mllm_comp.analysis.common.evidence_cohort import select_manifest_entries
                manifest_entries, selection_metadata = select_manifest_entries(
                    sampling["manifest_path"], task_name=task_name,
                    stage=sampling.get("manifest_stage"), shard=sampling.get("manifest_shard"),
                )
                documents = ((doc_id, doc) for doc_id, doc in documents if int(doc_id) in manifest_entries)
            elif sampling:
                from mllm_comp.analysis.common.discovery_sampling import sample_unique_image_docs
                documents, selection_metadata = sample_unique_image_docs(
                    documents, doc_to_visual=task.doc_to_visual,
                    count=sampling.get("unique_images_per_task"),
                    seed=sampling.get("seed", config.experiment.seed), excluded=excluded_images,
                )
                selection_metadata["exclusion_sources"] = exclusion_sources
            for doc_id, raw_doc in documents:
                context = task.doc_to_text(raw_doc)
                if not isinstance(context, str):
                    raise BackendIntegrationError(
                        f"Analysis task {task_name!r} returned a non-string prompt."
                    )
                visual = task.doc_to_visual(raw_doc)
                target_text, target_unavailable_reason = _analysis_target(
                    task,
                    raw_doc,
                    task_name=task_name,
                )
                sample = bridge.build_sample(
                    context=context,
                    visual=visual,
                    sample_id=f"{task_name}:{doc_id}",
                )
                processed = backbone.processor_adapter.prepare_inputs(
                    runtime.processor,
                    sample,
                    phase="eval",
                    runtime_mode=str(
                        runtime.metadata.get(
                            "runtime_mode",
                            RuntimeMode.DEFAULT.value,
                        )
                    ),
                )
                record: AnalysisSample = AnalysisSample(
                    sample_id=sample.sample_id,
                    task_name=task_name,
                    processor_outputs=dict(processed.model_inputs),
                    raw_doc=raw_doc,
                    media=visual,
                    metadata={
                        "prompt": context,
                        "prompt_lengths": list(processed.prompt_lengths),
                        **dict(processed.metadata),
                        "discovery_sampling": selection_metadata,
                        "evidence_cohort": manifest_entries.get(int(doc_id), {}),
                    },
                )
                if callable(getattr(task, "process_results", None)):
                    # Native scorers can normalize the human answer list in
                    # place. Copy it for every condition so evaluation cannot
                    # modify subsequent conditions or the recorded references.
                    record["score_prediction"] = lambda prediction, task=task, doc=raw_doc: to_jsonable(
                        task.process_results(deepcopy(doc), [prediction])
                    )
                    native_generation = task.get_config("generation_kwargs") or {}
                    record["generation_stops"] = native_generation.get("until", [])
                if target_text is None:
                    record["target_unavailable_reason"] = str(
                        target_unavailable_reason
                    )
                else:
                    record["target_text"] = target_text
                    record["target_source"] = (
                        "raw_doc.answers_plurality_case_punctuation_whitespace"
                        if "textvqa" in task_name.casefold()
                        and isinstance(raw_doc, Mapping)
                        and isinstance(raw_doc.get("answers"), Sequence)
                        else "lmms_eval.doc_to_target"
                    )
                    if (
                        "textvqa" in task_name.casefold()
                        and isinstance(raw_doc, Mapping)
                        and isinstance(raw_doc.get("answers"), Sequence)
                        and not isinstance(raw_doc.get("answers"), (str, bytes))
                    ):
                        record["target_references"] = [
                            str(value) for value in raw_doc["answers"]
                        ]
                if target_text is not None and requires_teacher_forcing:
                    prepare_teacher_forced = getattr(
                        backbone.processor_adapter,
                        "prepare_teacher_forced_inputs",
                        None,
                    )
                    if not callable(prepare_teacher_forced):
                        record["teacher_forcing_unavailable_reason"] = (
                            f"Backbone {backbone.capabilities().name!r} does not expose "
                            "teacher-forced analysis preprocessing."
                        )
                    else:
                        teacher_forced = prepare_teacher_forced(
                            runtime.processor,
                            sample,
                            target_text=target_text,
                            prompt_batch=processed,
                            phase="eval",
                            runtime_mode=str(
                                runtime.metadata.get(
                                    "runtime_mode",
                                    RuntimeMode.DEFAULT.value,
                                )
                            ),
                        )
                        objective = teacher_forced.metadata.get("teacher_forcing")
                        if not isinstance(objective, Mapping):
                            raise BackendIntegrationError(
                                "Teacher-forced preprocessing did not return objective metadata."
                            )
                        record["teacher_forced_processor_outputs"] = dict(
                            teacher_forced.model_inputs
                        )
                        record["objective"] = dict(objective)
                yield record

    def evaluate(
        self,
        *,
        config: Any,
        checkpoint_path: str | None = None,
        device: str | None = None,
        compressed: bool = False,
        manifest: Any | None = None,
    ) -> EvalResult:
        del manifest
        plan = self.build_launch_plan(
            config,
            checkpoint_path=checkpoint_path,
            device=device,
        )
        atomic_write_json(plan.plugin_spec_path, plan.plugin_spec)
        output_path = Path(plan.output_path)
        before = _result_snapshot(output_path)
        process_result = self._command_runner(plan)
        if process_result.returncode != 0:
            raise NativeCLIProcessError("lmms-eval", process_result.returncode)
        diagnostic_trace_paths = _diagnostic_trace_paths(plan)
        missing_traces = [
            str(path) for path in diagnostic_trace_paths if not path.is_file()
        ]
        if missing_traces:
            raise BackendIntegrationError(
                "lmms-eval completed without the expected diagnostic trace files: "
                f"{missing_traces}."
            )
        if diagnostic_trace_paths and not any(
            path.stat().st_size > 0 for path in diagnostic_trace_paths
        ):
            raise BackendIntegrationError(
                "lmms-eval diagnostic trace files are all empty; no per-sample "
                "runtime evidence was recorded."
            )
        diagnostic_pretty_paths = _materialize_pretty_diagnostics(
            diagnostic_trace_paths
        )
        compression_manifest_paths = _compression_manifest_paths(plan)
        missing_compression_manifests = [
            str(path) for path in compression_manifest_paths if not path.is_file()
        ]
        if missing_compression_manifests:
            raise BackendIntegrationError(
                "lmms-eval completed without compression lifecycle evidence: "
                f"{missing_compression_manifests}."
            )
        compression_manifests: list[dict[str, Any]] = []
        for path in compression_manifest_paths:
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                raise BackendIntegrationError(
                    f"Cannot read compression lifecycle evidence {path}: {exc}"
                ) from exc
            if not isinstance(payload, dict) or payload.get("enabled") is not True:
                raise BackendIntegrationError(
                    f"Compression lifecycle evidence {path} is incomplete."
                )
            compression_manifests.append(payload)
        if compression_manifests and not any(
            isinstance(item.get("last_request"), dict)
            for item in compression_manifests
        ):
            raise BackendIntegrationError(
                "Compression lifecycle evidence contains no completed eval request."
            )
        if plan.plugin_spec.get("request_mode") == "forward_probe":
            inactive = [
                index
                for index, payload in enumerate(compression_manifests)
                if not _has_compression_invocation(payload)
            ]
            if inactive:
                raise BackendIntegrationError(
                    "lmms-eval forward probe completed without a compression "
                    f"callback invocation in worker manifests {inactive}."
                )
        result_path, sample_paths, raw = _collect_native_result(output_path, before)
        versions = installed_versions(
            primary={
                "lmms_eval": ("lmms-eval", "lmms_eval"),
                "accelerate": ("accelerate",),
            }
        )
        backend_args = {
            "argv": list(plan.argv),
            "lmms_eval_argv": list(plan.lmms_eval_argv),
            "num_processes": plan.num_processes,
            "distributed_launcher": (
                plan.distributed_launcher if plan.num_processes > 1 else None
            ),
            "main_process_port": plan.main_process_port,
            "environment": dict(plan.environment),
            "cwd": plan.cwd,
            "versions": versions,
            "output_path": plan.output_path,
            "result_path": str(result_path),
            "sample_output_paths": [str(path) for path in sample_paths],
            "diagnostic_trace_paths": [
                str(path.resolve()) for path in diagnostic_trace_paths
            ],
            "diagnostic_pretty_paths": [
                str(path.resolve()) for path in diagnostic_pretty_paths
            ],
            "compression_manifest_paths": [
                str(path.resolve()) for path in compression_manifest_paths
            ],
            "compression_manifests": compression_manifests,
            "plugin_spec_path": plan.plugin_spec_path,
            "plugin_spec": dict(plan.plugin_spec),
            "runtime_mode": plan.plugin_spec["runtime_mode"],
            "model_name": plan.model_name,
            "checkpoint_path": plan.checkpoint_path,
            "compressed": bool(compressed),
        }
        return EvalResult(
            metrics=_metric_records(raw),
            raw_output_path=str(result_path),
            sample_output_paths=[str(path) for path in sample_paths],
            backend_args=backend_args,
            backend=BackendRecord(
                name="lmms_eval",
                version=versions["lmms_eval"],
                config_path=config.path,
                args=backend_args,
            ),
        )
