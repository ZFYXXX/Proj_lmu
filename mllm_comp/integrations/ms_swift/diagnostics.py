"""Readable training records and summaries; native evidence stays unchanged."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence


def _pretty_training_diagnostic_path(evidence_path: Path) -> Path:
    if evidence_path.suffix:
        return evidence_path.with_suffix(".pretty.json")
    return evidence_path.with_name(f"{evidence_path.name}.pretty.json")


def _half_open_spans(values: Any) -> list[list[int]]:
    if not isinstance(values, list):
        return []
    positions = sorted(
        {
            int(value)
            for value in values
            if isinstance(value, int) and not isinstance(value, bool)
        }
    )
    if not positions:
        return []
    spans: list[list[int]] = []
    start = previous = positions[0]
    for position in positions[1:]:
        if position == previous + 1:
            previous = position
            continue
        spans.append([start, previous + 1])
        start = previous = position
    spans.append([start, previous + 1])
    return spans


def _compact_removed_positions(value: Any) -> Any:
    if not isinstance(value, list):
        return value
    return [
        {
            "sample": sample_index,
            "count": len(positions) if isinstance(positions, list) else 0,
            "half_open_spans": _half_open_spans(positions),
        }
        for sample_index, positions in enumerate(value)
    ]


def _compact_training_compression_layer(value: Any) -> Any:
    if not isinstance(value, Mapping):
        return value
    compact = {
        str(name): child
        for name, child in value.items()
        if name != "chunks"
    }
    chunks = value.get("chunks")
    if not isinstance(chunks, list):
        return compact
    chunk_mappings = [chunk for chunk in chunks if isinstance(chunk, Mapping)]
    merged_pairs = [chunk.get("merged_pairs") for chunk in chunk_mappings]
    similarity_mins = [
        chunk.get("similarity_min")
        for chunk in chunk_mappings
        if isinstance(chunk.get("similarity_min"), (int, float))
        and not isinstance(chunk.get("similarity_min"), bool)
    ]
    similarity_maxes = [
        chunk.get("similarity_max")
        for chunk in chunk_mappings
        if isinstance(chunk.get("similarity_max"), (int, float))
        and not isinstance(chunk.get("similarity_max"), bool)
    ]
    chunk_summary: dict[str, Any] = {
        "count": len(chunks),
        "merged_pairs_by_chunk": merged_pairs,
    }
    if similarity_mins or similarity_maxes:
        chunk_summary["similarity_range"] = [
            min(similarity_mins) if similarity_mins else None,
            max(similarity_maxes) if similarity_maxes else None,
        ]
    compact["chunk_summary"] = chunk_summary
    return compact


def _compact_training_diagnostic_record(record: Mapping[str, Any]) -> dict[str, Any]:
    compact = dict(record)
    compression = compact.get("compression")
    if isinstance(compression, Mapping):
        compact_compression = dict(compression)
        layers = compact_compression.get("layers")
        if isinstance(layers, list):
            compact_compression["layers"] = [
                _compact_training_compression_layer(layer) for layer in layers
            ]
        compact["compression"] = compact_compression
    compression_state = compact.get("compression_state")
    if isinstance(compression_state, Mapping):
        compact_state = dict(compression_state)
        if "removed_sequence_positions_by_sample" in compact_state:
            compact_state["removed_sequence_positions_by_sample"] = (
                _compact_removed_positions(
                    compact_state["removed_sequence_positions_by_sample"]
                )
            )
        compact["compression_state"] = compact_state
    return compact


def _training_diagnostic_summary(
    records: Sequence[Mapping[str, Any]],
    *,
    source_path: Path,
) -> dict[str, Any]:
    forwards = [record for record in records if record.get("event") == "forward"]
    backwards = [record for record in records if record.get("event") == "backward"]
    original_tokens = sum(
        int((record.get("compression") or {}).get("original_audio_tokens", 0))
        for record in forwards
    )
    kept_tokens = sum(
        int((record.get("compression") or {}).get("kept_audio_tokens", 0))
        for record in forwards
    )
    forward_indices = {
        int(record["forward_index"])
        for record in forwards
        if "forward_index" in record
    }
    backward_indices = {
        int(record["forward_index"])
        for record in backwards
        if record.get("gradient_present") and "forward_index" in record
    }
    first = records[0] if records else {}
    summary = {
        "event": "summary",
        "schema_version": first.get("schema_version"),
        "runtime_mode": "debug",
        "stage_name": first.get("stage_name"),
        "plugin": first.get("plugin"),
        "source_jsonl": source_path.name,
        "install_events": sum(record.get("event") == "install" for record in records),
        "forward_events": len(forwards),
        "backward_events": len(backwards),
        "all_forwards_have_backward": forward_indices <= backward_indices,
        "audio_tokens": {
            "original": original_tokens,
            "kept": kept_tokens,
            "removed": original_tokens - kept_tokens,
            "keep_ratio": (
                kept_tokens / original_tokens if original_tokens else 1.0
            ),
        },
        "detail_policy": {
            "pretty": "chunk selections and removed positions are summarized",
            "complete": source_path.name,
        },
    }
    return summary


def _pretty_training_diagnostic_records(
    records: Sequence[Mapping[str, Any]], *, source_path: Path,
) -> list[dict[str, Any]]:
    stage_names = list(dict.fromkeys(record.get("stage_name") for record in records))
    if len(stage_names) > 1:
        # Budgets refer to each stage's input; adding them would misrepresent
        # the end-to-end reduction and label all stages as the first method.
        summary = {
            "event": "summary", "runtime_mode": "debug", "multi_stage": True,
            "source_jsonl": source_path.name,
            "stages": [
                _training_diagnostic_summary(
                    [record for record in records if record.get("stage_name") == name],
                    source_path=source_path,
                ) for name in stage_names
            ],
        }
    else:
        summary = _training_diagnostic_summary(records, source_path=source_path)
    return [summary, *[_compact_training_diagnostic_record(record) for record in records]]
