"""Typed JSON artifact envelopes and AnalysisRecord construction."""

from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import tempfile
from typing import Any, Iterable, Mapping

from mllm_comp.analysis.common.streaming import StreamingAnalysisRecords
from mllm_comp.core.errors import ConfigError
from mllm_comp.core.manifest import AnalysisRecord
from mllm_comp.core.serialization import atomic_write_json, to_jsonable


ANALYSIS_ARTIFACT_SCHEMA_VERSION = 1
_SAFE_FILENAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*\.json$")


def write_streaming_analysis_result(
    *,
    output_dir: str | Path,
    filename_stem: str,
    probe_name: str,
    method: str,
    method_version: str | None,
    phase: str,
    backbone_name: str,
    checkpoint: str,
    checkpoint_revision: str | None,
    capability_status: str | None,
    records: StreamingAnalysisRecords,
    aggregate: Mapping[str, Any],
    summary: Mapping[str, Any],
    aggregate_metadata: Mapping[str, Any] | None = None,
) -> AnalysisRecord:
    """Publish an aggregate, optional streamed raw samples, and their record.

    The Probe computes its aggregate and method-specific summary before calling
    this function. It also owns finish/close and rejects an empty successful set.
    ``filename_stem`` stays independent of the configured logical Probe name.
    """

    identity = {
        "output_dir": output_dir,
        "probe_name": probe_name,
        "method": method,
        "method_version": method_version,
        "phase": phase,
        "backbone_name": backbone_name,
        "checkpoint": checkpoint,
        "checkpoint_revision": checkpoint_revision,
        "capability_status": capability_status,
    }
    counts = {
        "successful_samples": records.successful_count,
        "skipped_samples": records.skipped_count,
    }
    artifact_paths = {
        "aggregate": write_analysis_artifact(
            **identity,
            filename=f"{filename_stem}.aggregate.json",
            artifact_kind="aggregate",
            payload=aggregate,
            metadata={**counts, **(aggregate_metadata or {})},
        )
    }
    if records.save_raw:
        artifact_paths["per_sample"] = write_analysis_sample_artifact(
            **identity,
            filename=f"{filename_stem}.raw.json",
            artifact_kind="per_sample",
            samples=records.iter_raw_samples(),
            skipped_samples=records.iter_raw_skipped(),
            metadata={"sample_count": records.successful_count},
        )
    return make_analysis_record(
        name=probe_name,
        phase=phase,
        method=method,
        summary={
            **counts,
            **summary,
            "aggregation_contract": records.execution_evidence()["contract"],
        },
        artifact_paths=artifact_paths,
        sample_count=records.successful_count,
    )


def write_analysis_artifact(
    *,
    output_dir: str | Path,
    filename: str,
    artifact_kind: str,
    probe_name: str,
    method: str,
    method_version: str | None,
    phase: str,
    backbone_name: str,
    checkpoint: str,
    checkpoint_revision: str | None,
    capability_status: str | None = None,
    payload: Any,
    metadata: Mapping[str, Any] | None = None,
) -> str:
    """Write one self-describing artifact and return its manifest path."""

    target = _artifact_target(output_dir, filename, phase)
    envelope = _artifact_envelope(
        artifact_kind=artifact_kind,
        probe_name=probe_name,
        method=method,
        method_version=method_version,
        phase=phase,
        backbone_name=backbone_name,
        checkpoint=checkpoint,
        checkpoint_revision=checkpoint_revision,
        capability_status=capability_status,
        metadata=metadata,
    )
    envelope["payload"] = to_jsonable(payload)
    atomic_write_json(target, envelope, ensure_ascii=False)
    return str(target)


def write_analysis_sample_artifact(
    *,
    output_dir: str | Path,
    filename: str,
    artifact_kind: str,
    probe_name: str,
    method: str,
    method_version: str | None,
    phase: str,
    backbone_name: str,
    checkpoint: str,
    checkpoint_revision: str | None,
    samples: Iterable[Mapping[str, Any]],
    skipped_samples: Iterable[Mapping[str, Any]],
    capability_status: str | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> str:
    """Atomically write per-sample arrays without materializing either iterable."""

    target = _artifact_target(output_dir, filename, phase)
    envelope = _artifact_envelope(
        artifact_kind=artifact_kind,
        probe_name=probe_name,
        method=method,
        method_version=method_version,
        phase=phase,
        backbone_name=backbone_name,
        checkpoint=checkpoint,
        checkpoint_revision=checkpoint_revision,
        capability_status=capability_status,
        metadata=metadata,
    )
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            dir=target.parent,
            delete=False,
            suffix=".tmp",
        ) as handle:
            tmp_name = handle.name
            handle.write("{\n")
            items = sorted(envelope.items())
            for key, value in items:
                handle.write("  ")
                json.dump(key, handle, ensure_ascii=False)
                handle.write(": ")
                json.dump(
                    value,
                    handle,
                    ensure_ascii=False,
                    indent=2,
                    sort_keys=True,
                )
                handle.write(",\n")
            handle.write('  "payload": {\n    "samples": [')
            _write_json_array_items(handle, samples)
            handle.write('],\n    "skipped_samples": [')
            _write_json_array_items(handle, skipped_samples)
            handle.write("]\n  }\n}\n")
        os.replace(tmp_name, target)
    except Exception:
        if tmp_name is not None:
            Path(tmp_name).unlink(missing_ok=True)
        raise
    return str(target)


def _artifact_target(
    output_dir: str | Path,
    filename: str,
    phase: str,
) -> Path:
    if not _SAFE_FILENAME.fullmatch(filename):
        raise ConfigError(
            "Analysis artifact filename must be one safe .json basename, got "
            f"{filename!r}."
        )
    if phase not in {"pre_compression", "post_compression"}:
        raise ConfigError(f"Unsupported analysis artifact phase {phase!r}.")
    return Path(output_dir) / filename


def _artifact_envelope(
    *,
    artifact_kind: str,
    probe_name: str,
    method: str,
    method_version: str | None,
    phase: str,
    backbone_name: str,
    checkpoint: str,
    checkpoint_revision: str | None,
    capability_status: str | None,
    metadata: Mapping[str, Any] | None,
) -> dict[str, Any]:
    return {
        "schema_version": ANALYSIS_ARTIFACT_SCHEMA_VERSION,
        "artifact_kind": artifact_kind,
        "probe": {
            "name": probe_name,
            "method": method,
            "method_version": method_version,
            "capability_status": capability_status,
        },
        "phase": phase,
        "backbone": {
            "name": backbone_name,
            "checkpoint": checkpoint,
            "revision": checkpoint_revision,
        },
        "created_at": datetime.now(timezone.utc).isoformat(),
        "metadata": to_jsonable(dict(metadata or {})),
    }


def _write_json_array_items(
    handle: Any,
    values: Iterable[Mapping[str, Any]],
) -> None:
    first = True
    for value in values:
        if not first:
            handle.write(",")
        handle.write("\n      ")
        json.dump(
            to_jsonable(dict(value)),
            handle,
            ensure_ascii=False,
            sort_keys=True,
        )
        first = False
    if not first:
        handle.write("\n    ")


def make_analysis_record(
    *,
    name: str,
    phase: str,
    method: str,
    summary: Mapping[str, Any],
    artifact_paths: Mapping[str, str],
    sample_count: int,
) -> AnalysisRecord:
    """Validate stable AnalysisRecord semantics before manifest insertion."""

    if not name or not method:
        raise ConfigError("AnalysisRecord name and method must be non-empty.")
    if phase not in {"pre_compression", "post_compression"}:
        raise ConfigError(f"Unsupported AnalysisRecord phase {phase!r}.")
    if sample_count < 0:
        raise ConfigError("AnalysisRecord sample_count cannot be negative.")
    normalized_paths = {
        str(kind): str(path)
        for kind, path in artifact_paths.items()
        if str(kind) and str(path)
    }
    if not normalized_paths:
        raise ConfigError("AnalysisRecord must reference at least one artifact.")
    return AnalysisRecord(
        name=name,
        phase=phase,
        method=method,
        summary=to_jsonable(dict(summary)),
        artifact_paths=normalized_paths,
        sample_count=sample_count,
    )


__all__ = [
    "ANALYSIS_ARTIFACT_SCHEMA_VERSION",
    "make_analysis_record",
    "write_analysis_artifact",
    "write_analysis_sample_artifact",
    "write_streaming_analysis_result",
]
