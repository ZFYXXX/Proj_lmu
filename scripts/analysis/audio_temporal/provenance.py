"""Time provenance for original, substituted, and merged audio tokens."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class TokenTrace:
    """The original temporal support represented by one current token."""

    source_indices: tuple[int, ...]
    intervals_ms: tuple[tuple[float, float], ...]
    mass: float
    center_ms: float
    hard_chunks: tuple[int, ...]

    def as_dict(self):
        union = union_intervals(self.intervals_ms)
        coverage = sum(end - start for start, end in union)
        span = union[-1][1] - union[0][0] if union else 0.0
        gaps = [right[0] - left[1] for left, right in zip(union, union[1:])]
        return {
            "source_indices": list(self.source_indices),
            "intervals_ms": [[float(a), float(b)] for a, b in union],
            "mass": float(self.mass),
            "center_ms": float(self.center_ms),
            "hard_chunks": list(self.hard_chunks),
            "nominal_span_ms": float(span),
            "effective_coverage_ms": float(coverage),
            "max_gap_ms": float(max(gaps, default=0.0)),
            "fragments": len(union),
            "crosses_hard_chunk": len(self.hard_chunks) > 1,
        }


def union_intervals(intervals):
    ordered = sorted((float(a), float(b)) for a, b in intervals if b >= a)
    merged = []
    for start, end in ordered:
        if not merged or start > merged[-1][1] + 1e-6:
            merged.append([start, end])
        else:
            merged[-1][1] = max(merged[-1][1], end)
    return tuple((float(start), float(end)) for start, end in merged)


def initialize_traces(time_ms, chunk_ids):
    """Assign a Voronoi-like time interval to every native representation token."""
    time_ms = np.asarray(time_ms, dtype=float)
    chunk_ids = np.asarray(chunk_ids, dtype=int)
    traces = [None] * len(time_ms)
    global_steps = np.diff(np.sort(time_ms))
    global_steps = global_steps[global_steps > 0]
    fallback_step = float(np.median(global_steps)) if len(global_steps) else 1.0
    for chunk in np.unique(chunk_ids):
        indices = np.flatnonzero(chunk_ids == chunk)
        times = time_ms[indices]
        if len(times) == 1:
            starts, ends = times - fallback_step / 2, times + fallback_step / 2
        else:
            midpoints = (times[:-1] + times[1:]) / 2
            starts = np.r_[times[0] - (times[1] - times[0]) / 2, midpoints]
            ends = np.r_[midpoints, times[-1] + (times[-1] - times[-2]) / 2]
        for index, start, end, center in zip(indices, starts, ends, times):
            traces[int(index)] = TokenTrace(
                source_indices=(int(index),),
                intervals_ms=((float(start), float(end)),),
                mass=1.0,
                center_ms=float(center),
                hard_chunks=(int(chunk),),
            )
    return traces


def merge_traces(traces, *, allow_cross_chunk=False):
    if not traces:
        raise ValueError("Cannot merge an empty provenance group")
    chunks = tuple(sorted({chunk for trace in traces for chunk in trace.hard_chunks}))
    if len(chunks) != 1 and not allow_cross_chunk:
        raise ValueError("Temporal analysis never merges across hard audio chunks")
    mass = float(sum(trace.mass for trace in traces))
    center = sum(trace.center_ms * trace.mass for trace in traces) / mass
    return TokenTrace(
        source_indices=tuple(sorted({index for trace in traces for index in trace.source_indices})),
        intervals_ms=union_intervals(
            interval for trace in traces for interval in trace.intervals_ms),
        mass=mass,
        center_ms=float(center),
        hard_chunks=chunks,
    )


def replay_ltbm_rounds(original_traces, merge_rounds, *, kept_tokens):
    """Replay actual LTBM round-local pairs; never reselect merge candidates."""
    current = list(original_traces)
    for round_number, round_info in enumerate(merge_rounds):
        if (round_info["round_index"] != round_number
                or round_info["input_tokens"] != len(current)):
            raise ValueError("LTBM round order or input token count differs from the runtime")
        by_destination = {}
        removed = set()
        for pair in round_info["merge_pairs"]:
            source, destination = (int(value) for value in pair)
            if (source < 0 or source >= len(current) or source % 2 != 0
                    or destination < 0 or destination >= len(current)
                    or destination % 2 != 1 or source in removed):
                raise ValueError("Invalid LTBM bipartite merge pair")
            removed.add(source)
            by_destination.setdefault(destination, []).append(source)
        if round_info["merge_count"] != len(removed):
            raise ValueError("LTBM round merge count differs from reported pairs")
        next_traces = []
        for index, trace in enumerate(current):
            if index in removed:
                continue
            sources = by_destination.get(index, ())
            next_traces.append(merge_traces(
                [trace, *(current[source] for source in sources)],
                allow_cross_chunk=True) if sources else trace)
        if len(next_traces) != round_info["output_tokens"]:
            raise ValueError("LTBM round output count differs from reported pairs")
        current = next_traces
    if len(current) != kept_tokens:
        raise ValueError("LTBM final token count differs from runtime report")
    sources = [index for trace in current for index in trace.source_indices]
    if sorted(sources) != list(range(len(original_traces))):
        raise ValueError("LTBM provenance lost or duplicated original P_in tokens")
    return current


def trace_summary(traces, *, boundary_times_ms=()):
    rows = [trace.as_dict() for trace in traces]
    boundaries = np.asarray(boundary_times_ms, dtype=float)
    for trace, row in zip(traces, rows):
        intervals = union_intervals(trace.intervals_ms)
        row["crosses_input_boundary"] = bool(intervals and any(
            intervals[0][0] < boundary < intervals[-1][1] for boundary in boundaries))
    def values(name):
        return np.asarray([row[name] for row in rows], dtype=float)
    return {
        "tokens": len(rows),
        "nominal_span_ms_mean": float(values("nominal_span_ms").mean()) if rows else None,
        "nominal_span_ms_max": float(values("nominal_span_ms").max()) if rows else None,
        "effective_coverage_ms_mean": float(values("effective_coverage_ms").mean()) if rows else None,
        "max_gap_ms_mean": float(values("max_gap_ms").mean()) if rows else None,
        "max_gap_ms_max": float(values("max_gap_ms").max()) if rows else None,
        "fragments_mean": float(values("fragments").mean()) if rows else None,
        "fragmented_tokens": int(sum(row["fragments"] > 1 for row in rows)),
        "hard_chunk_crossings": int(sum(row["crosses_hard_chunk"] for row in rows)),
        "input_boundary_crossings": int(sum(row["crosses_input_boundary"] for row in rows)),
    }


def chronological_anchor_indices(traces):
    """One native coordinate per output token, ordered for placeholder compaction."""
    anchors = []
    for trace in traces:
        anchors.append(min(trace.source_indices, key=lambda index: abs(index - np.mean(trace.source_indices))))
    return np.asarray(anchors, dtype=int)
