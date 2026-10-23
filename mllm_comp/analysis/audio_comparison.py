"""Compare quality and efficiency under the existing strict audio protocol."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


def _flatten_numeric(
    value: Any,
    *,
    prefix: str = "",
) -> dict[str, float]:
    if isinstance(value, bool):
        return {}
    if isinstance(value, (int, float)):
        return {prefix: float(value)} if prefix else {}
    if not isinstance(value, Mapping):
        return {}
    flattened: dict[str, float] = {}
    for name, child in value.items():
        child_prefix = f"{prefix}.{name}" if prefix else str(name)
        flattened.update(_flatten_numeric(child, prefix=child_prefix))
    return flattened


def _comparison(
    baseline: float,
    compressed: float,
    *,
    include_speedup: bool,
) -> dict[str, float | None]:
    output: dict[str, float | None] = {
        "baseline": baseline,
        "compressed": compressed,
        "absolute_delta": compressed - baseline,
        "relative_delta": (
            (compressed - baseline) / baseline if baseline != 0 else None
        ),
    }
    if include_speedup:
        output["baseline_over_compressed"] = (
            baseline / compressed if compressed != 0 else None
        )
    return output


def _nested_numeric(payload: Mapping[str, Any], dotted_path: str) -> float | None:
    value: Any = payload
    for component in dotted_path.split("."):
        if not isinstance(value, Mapping) or component not in value:
            return None
        value = value[component]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def _argv_value(argv: Any, name: str) -> str | None:
    if not isinstance(argv, list) or name not in argv:
        return None
    index = argv.index(name)
    if index + 1 >= len(argv):
        raise ValueError(f"lmms-eval argv ends after {name}.")
    return str(argv[index + 1])


def _evaluation_protocol(manifest: Mapping[str, Any], *, label: str) -> dict[str, Any]:
    backend = manifest.get("eval_backend")
    if not isinstance(backend, Mapping):
        raise ValueError(f"{label} manifest is missing eval_backend.")
    args = backend.get("args")
    if not isinstance(args, Mapping):
        raise ValueError(f"{label} manifest is missing eval_backend.args.")
    plugin_spec = args.get("plugin_spec")
    if not isinstance(plugin_spec, Mapping):
        raise ValueError(f"{label} manifest is missing the effective plugin_spec.")
    runtime_packages = args.get("versions")
    if not isinstance(runtime_packages, Mapping) or any(
        not isinstance(runtime_packages.get(name), str)
        or not runtime_packages.get(name)
        for name in ("transformers", "torch")
    ):
        raise ValueError(
            f"{label} manifest is missing Transformers/PyTorch version evidence."
        )
    argv = args.get("lmms_eval_argv")
    checkpoint = manifest.get("checkpoint_path") or manifest.get("model_path_or_id")
    if not isinstance(checkpoint, str) or not checkpoint:
        raise ValueError(f"{label} manifest is missing the checkpoint identity.")
    source_commit = manifest.get("source_commit")
    source_tree_hash = manifest.get("source_tree_hash")
    if (
        not isinstance(source_commit, str)
        or not source_commit
        or not isinstance(source_tree_hash, str)
        or not source_tree_hash
    ):
        raise ValueError(
            f"{label} manifest is missing Git/source-tree identity evidence."
        )
    tasks = manifest.get("eval_tasks")
    if not isinstance(tasks, list) or not tasks:
        raise ValueError(f"{label} manifest is missing eval_tasks.")
    source_identity = {"source_tree_hash": source_tree_hash}
    if manifest.get("source_scope") == "runtime_sources_v1":
        source_identity["source_scope"] = "runtime_sources_v1"
    else:
        # Existing manifests used a whole-worktree hash and remain comparable
        # to each other under their original, stricter provenance contract.
        source_identity.update({
            "source_scope": manifest.get("source_scope"),
            "source_commit": source_commit,
            "source_dirty": manifest.get("source_dirty"),
        })
    return {
        "backbone_name": manifest.get("backbone_name"),
        "checkpoint_path": checkpoint,
        **source_identity,
        "eval_tasks": tasks,
        "backend_name": backend.get("name"),
        "backend_version": backend.get("version"),
        "runtime_packages": dict(runtime_packages),
        "model_name": args.get("model_name"),
        "num_processes": args.get("num_processes"),
        "distributed_launcher": args.get("distributed_launcher"),
        "batch_size": _argv_value(argv, "--batch_size"),
        "sample_limit": _argv_value(argv, "--limit"),
        "seed": _argv_value(argv, "--seed"),
        "force_simple": isinstance(argv, list) and "--force_simple" in argv,
        "device": plugin_spec.get("device"),
        "expected_world_size": plugin_spec.get("expected_world_size"),
        "generation": plugin_spec.get("generation"),
        "system_prompt": plugin_spec.get("system_prompt"),
        "use_audio_in_video": plugin_spec.get("use_audio_in_video"),
    }


def _manifest_trace_paths(
    manifest: Mapping[str, Any],
    efficiency: Mapping[str, Any],
    *,
    label: str,
) -> list[str]:
    raw_paths = manifest.get("trace_paths")
    if not isinstance(raw_paths, Mapping) or not raw_paths:
        raise ValueError(f"{label} manifest has no trace_paths evidence.")
    manifest_paths = sorted(
        str(path) for name, path in raw_paths.items()
        if name.startswith("lmms_eval_diagnostic_")
        and name.removeprefix("lmms_eval_diagnostic_").isdigit()
        and isinstance(path, str) and path
    )
    source_files = efficiency.get("source_files")
    if not isinstance(source_files, list) or not source_files:
        raise ValueError(f"{label} efficiency summary has no source_files evidence.")
    summary_paths = sorted(str(path) for path in source_files)
    if manifest_paths != summary_paths:
        raise ValueError(
            f"{label} manifest trace_paths do not match efficiency source_files."
        )
    return manifest_paths


_EFFICIENCY_PATHS = (
    "timing_seconds.generation.median",
    "timing_seconds.prefill.median",
    "timing_seconds.decode_per_token.median",
    "timing_seconds.ttft.median",
    "timing_seconds.attention_score.median",
    "llm_input_tokens.median",
    "cuda_memory_bytes.max_allocated.median",
    "cuda_memory_bytes.peak_allocation_delta.median",
    "cuda_memory_bytes.max_reserved.median",
    "cuda_memory_bytes.peak_reserved_delta.median",
)

_REQUIRED_EFFICIENCY_PATHS = (
    "timing_seconds.generation.median",
    "timing_seconds.prefill.median",
    "llm_input_tokens.median",
    "cuda_memory_bytes.peak_allocation_delta.median",
)


def compare_runs(
    *,
    baseline_result: Mapping[str, Any],
    compressed_result: Mapping[str, Any],
    baseline_efficiency: Mapping[str, Any],
    compressed_efficiency: Mapping[str, Any],
    baseline_manifest: Mapping[str, Any] | None = None,
    compressed_manifest: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    baseline_modes = baseline_efficiency.get("compression_modes", [])
    compressed_modes = compressed_efficiency.get("compression_modes", [])
    if baseline_modes != ["none"]:
        raise ValueError(
            f"baseline efficiency summary has unexpected modes {baseline_modes!r}"
        )
    if compressed_modes != ["compressed"]:
        raise ValueError(
            "compressed efficiency summary has unexpected modes "
            f"{compressed_modes!r}"
        )

    baseline_trace_count = baseline_efficiency.get("num_trace_records")
    compressed_trace_count = compressed_efficiency.get("num_trace_records")
    if (
        isinstance(baseline_trace_count, bool)
        or not isinstance(baseline_trace_count, int)
        or baseline_trace_count < 1
        or compressed_trace_count != baseline_trace_count
    ):
        raise ValueError(
            "Baseline and compressed efficiency summaries must contain the same "
            "positive number of trace records."
        )
    compressed_record_count = compressed_efficiency.get("num_compressed_records")
    if compressed_record_count != compressed_trace_count:
        raise ValueError(
            "Every compressed trace record must contain compression evidence."
        )
    chronological_violations = (
        compressed_efficiency.get("token_summary", {}).get(
            "chronological_index_violations"
        )
        if isinstance(compressed_efficiency.get("token_summary"), Mapping)
        else None
    )
    if chronological_violations != []:
        raise ValueError(
            "Compressed efficiency summary contains chronological index violations "
            "or omitted the invariant field."
        )

    protocol: dict[str, Any] | None = None
    if (baseline_manifest is None) != (compressed_manifest is None):
        raise ValueError("Provide both baseline and compressed manifests, or neither.")
    if baseline_manifest is not None and compressed_manifest is not None:
        baseline_protocol = _evaluation_protocol(
            baseline_manifest,
            label="baseline",
        )
        compressed_protocol = _evaluation_protocol(
            compressed_manifest,
            label="compressed",
        )
        if baseline_protocol != compressed_protocol:
            raise ValueError(
                "Baseline and compressed manifests describe different evaluation "
                f"protocols: baseline={baseline_protocol}, compressed={compressed_protocol}."
            )
        baseline_trace_paths = _manifest_trace_paths(
            baseline_manifest,
            baseline_efficiency,
            label="baseline",
        )
        compressed_trace_paths = _manifest_trace_paths(
            compressed_manifest,
            compressed_efficiency,
            label="compressed",
        )
        baseline_environments = baseline_efficiency.get("runtime_environments")
        compressed_environments = compressed_efficiency.get("runtime_environments")
        if (
            not isinstance(baseline_environments, list)
            or not baseline_environments
            or baseline_environments != compressed_environments
        ):
            raise ValueError(
                "Baseline and compressed summaries must record the same non-empty "
                "runtime environment set."
            )
        expected_world_size = baseline_protocol.get("expected_world_size")
        if (
            isinstance(expected_world_size, bool)
            or not isinstance(expected_world_size, int)
            or len(baseline_trace_paths) != expected_world_size
            or len(compressed_trace_paths) != expected_world_size
        ):
            raise ValueError(
                "Manifest trace counts do not match the protocol world size."
            )
        protocol = {
            "matched": True,
            "identity": baseline_protocol,
            "baseline_trace_paths": baseline_trace_paths,
            "compressed_trace_paths": compressed_trace_paths,
            "runtime_environments": baseline_environments,
        }

    for path in _REQUIRED_EFFICIENCY_PATHS:
        if _nested_numeric(baseline_efficiency, path) is None:
            raise ValueError(f"Baseline efficiency summary is missing {path}.")
        if _nested_numeric(compressed_efficiency, path) is None:
            raise ValueError(f"Compressed efficiency summary is missing {path}.")

    baseline_quality = _flatten_numeric(
        baseline_result.get("results", baseline_result)
    )
    compressed_quality = _flatten_numeric(
        compressed_result.get("results", compressed_result)
    )
    common_quality = sorted(set(baseline_quality) & set(compressed_quality))
    if not common_quality:
        raise ValueError("The two lmms-eval result files share no numeric metrics.")

    efficiency: dict[str, Any] = {}
    for path in _EFFICIENCY_PATHS:
        baseline_value = _nested_numeric(baseline_efficiency, path)
        compressed_value = _nested_numeric(compressed_efficiency, path)
        if baseline_value is None or compressed_value is None:
            continue
        efficiency[path] = _comparison(
            baseline_value,
            compressed_value,
            include_speedup=True,
        )

    return {
        "evaluation_protocol": protocol,
        "quality_metrics": {
            path: _comparison(
                baseline_quality[path],
                compressed_quality[path],
                include_speedup=False,
            )
            for path in common_quality
        },
        "efficiency_metrics": efficiency,
        "trace_counts": {
            "baseline": baseline_trace_count,
            "compressed": compressed_trace_count,
        },
        "token_totals": {
            "original": _nested_numeric(
                compressed_efficiency,
                "token_summary.original_audio_tokens",
            ),
            "kept": _nested_numeric(
                compressed_efficiency,
                "token_summary.kept_audio_tokens",
            ),
            "achieved_keep_ratio": _nested_numeric(
                compressed_efficiency,
                "token_summary.achieved_keep_ratio",
            ),
        },
        "compression_only_metrics": {
            "attention_score_seconds_median": _nested_numeric(
                compressed_efficiency,
                "timing_seconds.attention_score.median",
            ),
        },
    }
