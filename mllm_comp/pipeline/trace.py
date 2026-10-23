"""Trace records and helpers."""

from __future__ import annotations

from dataclasses import dataclass, field
from collections.abc import Mapping
import json
from pathlib import Path
from typing import Any

from mllm_comp.core.actions import OperationAction
from mllm_comp.core.state import TokenSpan
from mllm_comp.core.serialization import atomic_write_json, to_jsonable


TRACE_FORMAT = "jsonl"


@dataclass
class TokenTraceRecord:
    """One sample-stage token compression trace record.

    Trace files are JSONL, with one record per sample and stage.
    """

    sample_id: str | int
    stage_name: str
    placement: str
    original_token_count: int
    kept_token_count: int
    affected_spans: list[TokenSpan] = field(default_factory=list)
    changed_indices: list[int] | list[list[int]] | None = None
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class OperationTraceRecord:
    """One operation-compression trace record for a stage."""

    stage_name: str
    placement: str
    action: OperationAction
    dry_run: bool = False
    installed_handles: list[str] = field(default_factory=list)
    reversible: bool = True
    extra: dict[str, Any] = field(default_factory=dict)


TraceRecord = TokenTraceRecord


def _scalar_fields(value: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: item
        for key, item in value.items()
        if item is None or isinstance(item, (str, int, float, bool))
    }


def compression_runtime_summary(runtime: Mapping[str, Any]) -> dict[str, Any]:
    """Summarize lifecycle, counters and layout sizes without token mappings."""

    result = _scalar_fields(runtime)
    for key in (
        "configured_order", "install_order", "active_activations",
        "rollback_errors", "uninstall_errors", "canonical_config",
        "callback_counts", "dependencies", "target_sites", "callback_sites",
        "installed_handles",
    ):
        if key in runtime:
            result[key] = runtime[key]
    activations = runtime.get("activations")
    if isinstance(activations, (list, tuple)):
        result["activations"] = [
            compression_runtime_summary(item) for item in activations
            if isinstance(item, Mapping)
        ]
    for key, value in runtime.items():
        if isinstance(value, (list, tuple)) and key.endswith(
            ("_records", "_deltas", "_repairs", "_views")
        ):
            result[f"{key}_count"] = len(value)
    layout = runtime.get("layout_delta_summary")
    if isinstance(layout, Mapping):
        result["layout_delta_summary"] = {
            "record_count": layout.get("record_count", 0),
            "records": [_scalar_fields(item) for item in layout.get("records", ())],
        }
    return result


class CompressionTraceWriter:
    """Append request details once; atomically publish a small current manifest."""

    def __init__(self, manifest_path: str, *, model_class: str) -> None:
        self.manifest_path = Path(manifest_path)
        self.trace_path = self.manifest_path.with_suffix(".requests.jsonl")
        self.trace_path.parent.mkdir(parents=True, exist_ok=True)
        self.trace_path.write_text("", encoding="utf-8")
        self.model_class = model_class
        self.request_count = 0

    def write(
        self,
        runtime: Mapping[str, Any],
        *,
        last_request: Mapping[str, Any] | None,
    ) -> None:
        if last_request is not None:
            record = {
                "request_index": self.request_count,
                "request": dict(last_request),
                "runtime": dict(runtime),
            }
            encoded = json.dumps(to_jsonable(record), ensure_ascii=False)
            with self.trace_path.open("a", encoding="utf-8") as stream:
                stream.write(encoded + "\n")
            self.request_count += 1
        request_summary = None
        if last_request is not None:
            request_summary = _scalar_fields(last_request)
            for key, value in last_request.items():
                if isinstance(value, Mapping):
                    request_summary[key] = _scalar_fields(value)
        atomic_write_json(
            self.manifest_path,
            {
                "schema_version": 2,
                "enabled": True,
                "runtime": compression_runtime_summary(runtime),
                "model_class": self.model_class,
                "last_request": request_summary,
                "request_count": self.request_count,
                "trace": {
                    "path": str(self.trace_path.resolve()),
                    "format": "jsonl",
                    "record_count": self.request_count,
                },
            },
        )
