"""Read-only comparison of two recorded framework run directories."""

from __future__ import annotations

from collections.abc import Mapping
import json
import math
from pathlib import Path
from typing import Any


def compare_runs(left_run: str | Path, right_run: str | Path) -> dict[str, Any]:
    """Compare recorded values, reporting protocol differences and missing data.

    Inputs are run directories. Read their default manifest or the absolute
    manifest path recorded in result.json. Legacy directories with only the
    default manifest remain readable; arbitrary relative custom paths are not
    resolved against a guessed launch directory.
    """

    left = _read_run(Path(left_run))
    right = _read_run(Path(right_run))
    left_metrics = _metrics(left)
    right_metrics = _metrics(right)
    metrics = []
    for key in sorted(left_metrics.keys() | right_metrics.keys(), key=str):
        left_value = left_metrics.get(key)
        right_value = right_metrics.get(key)
        before = None if left_value is None else left_value["value"]
        after = None if right_value is None else right_value["value"]
        delta = None if before is None or after is None else after - before
        metrics.append({
            "task": key[0], "split": key[1], "metric": key[2],
            "left": before, "right": after,
            "left_present": left_value is not None,
            "right_present": right_value is not None,
            "absolute_delta": delta,
            "relative_delta": None if delta is None or before == 0 else delta / before,
            "left_num_samples": None if left_value is None else left_value["num_samples"],
            "right_num_samples": None if right_value is None else right_value["num_samples"],
        })
    left_protocol, left_missing = _protocol(left)
    right_protocol, right_missing = _protocol(right)
    differences = _differences(left_protocol, right_protocol)
    left_config = _comparison_config(left)
    right_config = _comparison_config(right)
    return {
        "schema_version": 1,
        "left": left["summary"],
        "right": right["summary"],
        "metrics": metrics,
        "analysis_records": {
            "left": (left["manifest"] or {}).get("analysis_records", []),
            "right": (right["manifest"] or {}).get("analysis_records", []),
        },
        "protocol": {
            "matched": False if differences else None if left_missing or right_missing else True,
            "differences": differences,
            "missing": {"left": left_missing, "right": right_missing},
        },
        "config_differences": (
            _differences(left_config, right_config)
            if left_config is not None and right_config is not None else None
        ),
        "audio_efficiency": _audio_efficiency(left, right),
    }


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path}: expected a JSON object.")
    return payload


def _read_run(directory: Path) -> dict[str, Any]:
    directory = directory.resolve()
    if not directory.is_dir():
        raise ValueError(f"Run directory does not exist: {directory}")
    result_path = directory / "result.json"
    result = _read_json(result_path) if result_path.is_file() else {}
    manifest_path = directory / "manifest.json"
    paths = result.get("paths") or {}
    recorded_manifest = paths.get("manifest")
    if recorded_manifest:
        recorded = Path(recorded_manifest)
        if recorded.is_absolute():
            manifest_path = recorded
        elif recorded != Path(paths.get("run_dir", ".")) / "manifest.json":
            raise ValueError(
                f"{directory}: custom manifest path {recorded_manifest!r} is relative; "
                "record an absolute manifest path instead of relying on the launch cwd."
            )
    manifest = _read_json(manifest_path) if manifest_path.is_file() else None
    config_path = directory / "effective_config.json"
    config = (manifest or {}).get("effective_config") or (
        _read_json(config_path) if config_path.is_file() else None
    )
    if not result and manifest is None:
        raise ValueError(f"{directory}: neither result.json nor a default manifest is available.")
    task = result.get("task")
    if task is None and manifest is not None:
        task = (
            "analyze" if manifest.get("analysis_records")
            else "train" if manifest.get("train_backend") else "eval"
        )
    return {
        "directory": directory,
        "result": result,
        "manifest": manifest,
        "config": config,
        "summary": {
            "run_dir": str(directory),
            "run_id": result.get("run_id") or (manifest or {}).get("run_id"),
            "task": task,
            "status": result.get("status", "unknown"),
            "returncode": result.get("returncode"),
            "native_returncode": result.get("native_returncode"),
            "error": result.get("error"),
            "manifest_path": str(manifest_path) if manifest is not None else None,
            "missing": [] if manifest is not None else ["manifest"],
        },
    }


def _metrics(run: dict[str, Any]) -> dict[tuple[str, str | None, str], dict[str, Any]]:
    values = {}
    manifest = run["manifest"] or {}
    records = [*manifest.get("baseline_metrics", []), *manifest.get("compressed_metrics", [])]
    if run["summary"]["task"] == "train":
        records.append({
            "task": "train", "split": None,
            "metrics": (run["result"].get("output") or {}).get("metrics", {}),
        })
    for record in records:
        for name, value in record["metrics"].items():
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError(f"{run['directory']}: metric {name!r} must be finite and numeric.")
            key = (record["task"], record.get("split"), name)
            if key in values:
                raise ValueError(f"{run['directory']}: ambiguous duplicate metric {key!r}.")
            values[key] = {"value": value, "num_samples": record.get("num_samples")}
    return values


def _protocol(run: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    manifest = run["manifest"]
    if manifest is None:
        return {}, ["manifest"]
    task = run["summary"]["task"]
    backend_key = "train_backend" if task == "train" else "eval_backend"
    backend = manifest.get(backend_key) or {}
    args = backend.get("args") or {}
    protocol = {"task": task}
    missing = []
    for name in ("backbone_name", "source_tree_hash", "source_scope"):
        if name not in manifest or manifest[name] is None:
            missing.append(name)
        else:
            protocol[name] = manifest[name]
    checkpoint = (
        args.get("model_path") or manifest.get("model_path_or_id")
        if task == "train"
        else manifest.get("checkpoint_path") or manifest.get("model_path_or_id")
    )
    if checkpoint:
        protocol["checkpoint"] = checkpoint
    else:
        missing.append("checkpoint")
    if task != "analyze":
        for name in ("name", "version"):
            if not backend.get(name):
                missing.append(f"backend.{name}")
            else:
                protocol[f"backend.{name}"] = backend[name]
    if args.get("versions"):
        protocol["runtime_versions"] = args["versions"]
    else:
        missing.append("runtime_versions")
    if task == "eval":
        from mllm_comp.analysis.audio_comparison import _evaluation_protocol

        try:
            protocol["native_evaluation"] = _evaluation_protocol(manifest, label="Run")
        except ValueError as exc:
            missing.append(f"native_evaluation: {exc}")
    elif task == "train":
        protocol["native_training"] = {
            name: args[name] for name in (
                "model_path", "datasets", "model_type", "template", "trainable_branch", "talker_training",
            ) if name in args
        }
    config = run["config"] or {}
    section_name = "analysis" if task == "analyze" else task
    section = config.get(section_name) or {}
    if section:
        protocol[section_name] = {
            key: value for key, value in section.items()
            if key not in {"compression_mode", "target", "checkpoint_path", "main_process_port"}
        }
        if task == "train":
            protocol[section_name]["args"] = {
                key: value for key, value in section.get("args", {}).items()
                if key != "output_dir"
            }
        elif task == "eval":
            protocol[section_name]["args"] = {
                key: value for key, value in section.get("args", {}).items()
                if key != "diagnostic_trace_path"
            }
    else:
        missing.append(section_name)
    for name in ("backbone", "data"):
        if name in config:
            protocol[name] = config[name]
        else:
            missing.append(name)
    records = [*manifest.get("baseline_metrics", []), *manifest.get("compressed_metrics", [])]
    protocol["sample_counts"] = sorted(
        ({"task": record["task"], "split": record.get("split"), "count": record.get("num_samples")} for record in records),
        key=lambda record: (record["task"], str(record["split"])),
    )
    if task == "eval":
        for record in protocol["sample_counts"]:
            if record["count"] is None:
                missing.append(f"sample_counts.{record['task']}.{record['split']}")
    experiment = config.get("experiment") or {}
    if "seed" in experiment:
        protocol["seed"] = experiment["seed"]
    else:
        missing.append("experiment.seed")
    return protocol, missing


def _comparison_config(run: dict[str, Any]) -> dict[str, Any] | None:
    config = run["config"]
    if not config:
        return None
    result = {
        **{name: value for name, value in config.items() if name not in {"output", "experiment"}},
        "experiment": {name: value for name, value in config.get("experiment", {}).items() if name != "run_id"},
    }
    if "train" in result:
        result["train"] = {
            **result["train"],
            "args": {name: value for name, value in result["train"].get("args", {}).items() if name != "output_dir"},
        }
    return result


def _differences(left: Mapping[str, Any], right: Mapping[str, Any], prefix: str = "") -> list[dict[str, Any]]:
    differences = []
    for key in sorted(left.keys() | right.keys()):
        path = f"{prefix}.{key}" if prefix else key
        before, after = left.get(key), right.get(key)
        if key in left and key in right and isinstance(before, Mapping) and isinstance(after, Mapping):
            differences.extend(_differences(before, after, path))
        elif key not in left or key not in right or before != after:
            differences.append({
                "field": path, "left": before, "right": after,
                "left_present": key in left, "right_present": key in right,
            })
    return differences


def _audio_efficiency(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    from mllm_comp.analysis.audio_comparison import compare_runs as compare_audio_runs
    from mllm_comp.analysis.audio_summary import _read_records, summarize_records

    manifests = [left["manifest"], right["manifest"]]
    if any(manifest is None or manifest.get("backbone_name") != "qwen25_omni" for manifest in manifests):
        return {"status": "unavailable", "reason": "Audio efficiency requires two Qwen2.5-Omni profile manifests."}
    try:
        modes, summaries, quality = [], [], []
        for manifest in manifests:
            spec = ((manifest.get("eval_backend") or {}).get("args") or {}).get("plugin_spec") or {}
            if spec.get("runtime_mode") != "profile":
                raise ValueError("Audio efficiency requires runtime_mode=profile on both runs.")
            modes.append(spec.get("compression_mode"))
            trace_paths = sorted(
                value for name, value in manifest.get("trace_paths", {}).items()
                if name.startswith("lmms_eval_diagnostic_")
                and name.removeprefix("lmms_eval_diagnostic_").isdigit()
            )
            if not trace_paths or any(not Path(path).is_absolute() for path in trace_paths):
                raise ValueError("Audio efficiency requires recorded absolute diagnostic trace paths.")
            summary = summarize_records(_read_records([Path(path) for path in trace_paths]))
            summary["source_files"] = trace_paths
            summaries.append(summary)
            records = [*manifest.get("baseline_metrics", []), *manifest.get("compressed_metrics", [])]
            results = {}
            for record in records:
                results.setdefault(f"{record['task']}/{record.get('split')}", {}).update(record["metrics"])
            quality.append({"results": results})
        if set(modes) != {"compressed", "none"}:
            raise ValueError("Audio efficiency requires one baseline and one compressed run.")
        baseline, compressed = modes.index("none"), modes.index("compressed")
        report = compare_audio_runs(
            baseline_result=quality[baseline], compressed_result=quality[compressed],
            baseline_efficiency=summaries[baseline], compressed_efficiency=summaries[compressed],
            baseline_manifest=manifests[baseline], compressed_manifest=manifests[compressed],
        )
        return {"status": "compared", "baseline": ("left", "right")[baseline], "comparison": report}
    except (OSError, ValueError) as exc:
        return {"status": "unavailable", "reason": str(exc)}
