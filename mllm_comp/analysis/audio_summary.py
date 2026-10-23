"""Summarize the existing Qwen2.5-Omni audio profile trace format."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
import glob
import json
from pathlib import Path
import statistics
from typing import Any


def _numeric(values: Iterable[Any]) -> list[float]:
    return [
        float(value)
        for value in values
        if isinstance(value, (int, float)) and not isinstance(value, bool)
    ]


def _percentile(values: Sequence[float], percentile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * percentile
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def _distribution(values: Iterable[Any]) -> dict[str, Any]:
    numeric = _numeric(values)
    return {
        "count": len(numeric),
        "median": statistics.median(numeric) if numeric else None,
        "p90": _percentile(numeric, 0.9),
        "min": min(numeric) if numeric else None,
        "max": max(numeric) if numeric else None,
    }


def summarize_records(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Aggregate per-sample token and latency evidence."""

    compression_records: list[Mapping[str, Any]] = []
    timings: list[Mapping[str, Any]] = []
    compression_modes: set[str] = set()
    runtime_modes: set[str] = set()
    chronological_violations: list[dict[str, Any]] = []
    runtime_environments: dict[str, Mapping[str, Any]] = {}
    for record in records:
        runtime_environment = record.get("runtime_environment")
        if isinstance(runtime_environment, Mapping):
            canonical = json.dumps(
                runtime_environment,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
            runtime_environments[canonical] = runtime_environment
        generation = record.get("generation_metadata", {})
        if not isinstance(generation, Mapping):
            continue
        compression_mode = generation.get("compression_mode")
        if isinstance(compression_mode, str) and compression_mode:
            compression_modes.add(compression_mode)
        runtime_mode = generation.get("runtime_mode")
        if isinstance(runtime_mode, str) and runtime_mode:
            runtime_modes.add(runtime_mode)
        compression = generation.get("compression")
        if isinstance(compression, Mapping):
            compression_records.append(compression)
            for item in compression.get("items", []):
                if isinstance(item, Mapping) and item.get("indices_chronological") is not True:
                    chronological_violations.append(
                        {
                            "sample_id": record.get("sample_id"),
                            "audio_unit_index": item.get("audio_unit_index"),
                        }
                    )
        timing = generation.get("timing")
        if isinstance(timing, Mapping):
            timings.append(timing)

    if len(compression_modes) > 1:
        raise ValueError(
            "Diagnostic inputs mix baseline and compressed records. Summarize "
            "each run separately before comparing their output JSON files."
        )
    if runtime_modes != {"profile"}:
        raise ValueError(
            "Efficiency summaries require only runtime_mode=profile records; "
            f"found {sorted(runtime_modes) or ['missing']}."
        )

    original_total = sum(
        int(record.get("original_audio_tokens", 0))
        for record in compression_records
    )
    kept_total = sum(
        int(record.get("kept_audio_tokens", 0))
        for record in compression_records
    )
    return {
        "num_trace_records": len(records),
        "num_compressed_records": len(compression_records),
        "num_timing_records": len(timings),
        "compression_modes": sorted(compression_modes),
        "runtime_modes": sorted(runtime_modes),
        "runtime_environments": [
            dict(runtime_environments[key]) for key in sorted(runtime_environments)
        ],
        "token_summary": {
            "original_audio_tokens": original_total,
            "kept_audio_tokens": kept_total,
            "achieved_keep_ratio": (
                kept_total / original_total if original_total else None
            ),
            "per_sample_original": _distribution(
                record.get("original_audio_tokens")
                for record in compression_records
            ),
            "per_sample_kept": _distribution(
                record.get("kept_audio_tokens")
                for record in compression_records
            ),
            "chronological_index_violations": chronological_violations,
        },
        "timing_seconds": {
            "generation": _distribution(
                timing.get("generation_seconds") for timing in timings
            ),
            "prefill": _distribution(
                timing.get("prefill_seconds") for timing in timings
            ),
            "ttft": _distribution(
                timing.get("ttft_seconds") for timing in timings
            ),
            "decode_per_token": _distribution(
                timing.get("decode_seconds_per_token") for timing in timings
            ),
            "attention_score": _distribution(
                record.get("attention_score_seconds")
                for record in compression_records
            ),
        },
        "llm_input_tokens": _distribution(
            timing.get("llm_input_tokens") for timing in timings
        ),
        "cuda_memory_bytes": {
            "allocated_before": _distribution(
                timing.get("cuda_memory_allocated_before_bytes")
                for timing in timings
            ),
            "max_allocated": _distribution(
                timing.get("cuda_max_memory_allocated_bytes")
                for timing in timings
            ),
            "peak_allocation_delta": _distribution(
                timing.get("cuda_peak_memory_delta_bytes")
                for timing in timings
            ),
            "reserved_before": _distribution(
                timing.get("cuda_memory_reserved_before_bytes")
                for timing in timings
            ),
            "max_reserved": _distribution(
                timing.get("cuda_max_memory_reserved_bytes")
                for timing in timings
            ),
            "peak_reserved_delta": _distribution(
                timing.get("cuda_peak_reserved_delta_bytes")
                for timing in timings
            ),
        },
    }


def _resolve_paths(values: Sequence[str]) -> list[Path]:
    paths: set[Path] = set()
    for value in values:
        expanded = [Path(item) for item in glob.glob(value)]
        if not expanded:
            expanded = [Path(value)]
        for path in expanded:
            if path.is_dir():
                owned = _plugin_owned_trace_paths(path)
                if owned is None:
                    paths.update(path.rglob("compression_diagnostic*.jsonl"))
                    paths.update(path.rglob("baseline_diagnostic*.jsonl"))
                else:
                    paths.update(owned)
            elif path.is_file():
                paths.add(path)
    return sorted(paths)


def _plugin_owned_trace_paths(run_dir: Path) -> list[Path] | None:
    """Use the launch spec to exclude stale rank files from earlier runs."""

    spec_path = run_dir / "mllm_comp_lmms_plugin.json"
    if not spec_path.is_file():
        return None
    try:
        payload = json.loads(spec_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Cannot read {spec_path}: {exc}") from exc
    if not isinstance(payload, Mapping):
        raise ValueError(f"{spec_path}: expected a JSON object")
    if payload.get("runtime_mode") != "profile":
        raise ValueError(
            f"{spec_path}: efficiency summaries require runtime_mode=profile"
        )
    value = payload.get("diagnostic_trace_path")
    if not isinstance(value, str) or not value:
        return None
    world_size = payload.get("expected_world_size", 1)
    if isinstance(world_size, bool) or not isinstance(world_size, int) or world_size < 1:
        raise ValueError(f"{spec_path}: expected_world_size must be a positive integer")
    trace_path = Path(value)
    if world_size == 1:
        expected = [trace_path]
    else:
        suffix = trace_path.suffix
        stem = trace_path.name[: -len(suffix)] if suffix else trace_path.name
        expected = [
            trace_path.with_name(f"{stem}.rank{rank}{suffix}")
            for rank in range(world_size)
        ]
    missing = [str(path) for path in expected if not path.is_file()]
    if missing:
        raise ValueError(
            f"{spec_path}: missing expected diagnostic trace files {missing}"
        )
    return expected


def _read_records(paths: Sequence[Path]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for path in paths:
        for line_number, line in enumerate(
            path.read_text(encoding="utf-8").splitlines(),
            start=1,
        ):
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
            if not isinstance(payload, dict):
                raise ValueError(f"{path}:{line_number}: expected a JSON object")
            records.append(payload)
    return records
