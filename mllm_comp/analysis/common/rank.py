"""Task-wise percentile aggregation that preserves raw analysis scores."""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence


def finite_metric_count(value: Any, metrics: Sequence[str]) -> int:
    """Count finite numeric values whose mapping key names a selected metric."""

    if isinstance(value, Mapping):
        return sum(
            1
            if key in metrics
            and item is not None
            and isinstance(item, (int, float))
            and math.isfinite(float(item))
            else finite_metric_count(item, metrics)
            for key, item in value.items()
        )
    if isinstance(value, (list, tuple)):
        return sum(finite_metric_count(item, metrics) for item in value)
    return 0


def select_ranking_metric(
    metrics: Sequence[str],
    configured: Any,
    directions: Mapping[str, bool],
) -> str:
    """Select the configured ranking metric or the preferred rankable default."""

    if configured is not None:
        return str(configured)
    if "js" in metrics:
        return "js"
    return next(metric for metric in metrics if metric in directions)


def aggregate_task_percentiles(
    task_scores: Mapping[str, Mapping[str, float]],
    *,
    higher_is_more_important: bool = True,
) -> dict:
    """Return raw scores, per-task average-tie percentiles, and their mean."""

    components: dict[str, dict[str, dict[str, float]]] = {}
    skipped: list[dict[str, str]] = []
    for task_name, scores in task_scores.items():
        finite: dict[str, float] = {}
        for component, raw_score in scores.items():
            score = float(raw_score)
            if not math.isfinite(score):
                skipped.append(
                    {
                        "task": str(task_name),
                        "component": str(component),
                        "reason": "non_finite_raw_score",
                    }
                )
                continue
            finite[str(component)] = score
        ranks = _average_tie_percentiles(
            finite,
            higher_is_more_important=higher_is_more_important,
        )
        for component, score in finite.items():
            entry = components.setdefault(
                component,
                {"raw_scores_by_task": {}, "percentiles_by_task": {}},
            )
            entry["raw_scores_by_task"][str(task_name)] = score
            entry["percentiles_by_task"][str(task_name)] = ranks[component]

    result_components: dict[str, dict] = {}
    for component, entry in sorted(components.items()):
        percentiles = entry["percentiles_by_task"]
        result_components[component] = {
            **entry,
            "mean_percentile": sum(percentiles.values()) / len(percentiles),
            "task_count": len(percentiles),
        }
    return {
        "schema_version": 1,
        "aggregation": "mean_task_percentile_average_ties_v1",
        "higher_is_more_important": higher_is_more_important,
        "task_count": len(task_scores),
        "components": result_components,
        "skipped": skipped,
    }


def _average_tie_percentiles(
    scores: Mapping[str, float],
    *,
    higher_is_more_important: bool,
) -> dict[str, float]:
    if not scores:
        return {}
    ordered = sorted(
        scores.items(),
        key=lambda item: item[1],
        reverse=not higher_is_more_important,
    )
    result: dict[str, float] = {}
    index = 0
    size = len(ordered)
    while index < size:
        end = index + 1
        while end < size and ordered[end][1] == ordered[index][1]:
            end += 1
        average_rank = ((index + 1) + end) / 2.0
        percentile = average_rank / size
        for component, _score in ordered[index:end]:
            result[component] = percentile
        index = end
    return result


def aggregate_task_vector_percentiles(
    task_vectors: Mapping[str, Mapping[str, Sequence[float]]],
    *,
    higher_is_more_important: bool = True,
) -> dict:
    """Rank compact component vectors independently, then average across tasks."""

    vectors: dict[str, dict] = {}
    skipped: list[dict[str, str | int]] = []
    vector_ids = sorted(
        {
            str(vector_id)
            for values in task_vectors.values()
            for vector_id in values
        }
    )
    for vector_id in vector_ids:
        raw_scores_by_task: dict[str, list[float | None]] = {}
        percentiles_by_task: dict[str, list[float | None]] = {}
        expected_length: int | None = None
        for task, task_values in task_vectors.items():
            if vector_id not in task_values:
                continue
            raw_values = [float(value) for value in task_values[vector_id]]
            if expected_length is None:
                expected_length = len(raw_values)
            elif len(raw_values) != expected_length:
                raise ValueError(
                    f"Component vector {vector_id!r} changes length across tasks."
                )
            finite = {
                index: value
                for index, value in enumerate(raw_values)
                if math.isfinite(value)
            }
            for index, value in enumerate(raw_values):
                if not math.isfinite(value):
                    skipped.append(
                        {
                            "task": str(task),
                            "vector": vector_id,
                            "index": index,
                            "reason": "non_finite_raw_score",
                        }
                    )
            ranked = _average_tie_percentile_vector(
                finite,
                higher_is_more_important=higher_is_more_important,
            )
            raw_scores_by_task[str(task)] = [
                value if math.isfinite(value) else None for value in raw_values
            ]
            percentiles_by_task[str(task)] = [
                ranked.get(index) for index in range(len(raw_values))
            ]
        if expected_length is None:
            continue
        mean_percentile: list[float | None] = []
        task_count: list[int] = []
        for index in range(expected_length):
            values = [
                task_values[index]
                for task_values in percentiles_by_task.values()
                if task_values[index] is not None
            ]
            mean_percentile.append(
                sum(values) / len(values) if values else None
            )
            task_count.append(len(values))
        vectors[vector_id] = {
            "length": expected_length,
            "raw_scores_by_task": raw_scores_by_task,
            "percentiles_by_task": percentiles_by_task,
            "mean_percentile": mean_percentile,
            "task_count": task_count,
        }
    return {
        "schema_version": 2,
        "aggregation": (
            "mean_task_percentile_average_ties_within_component_vector_v2"
        ),
        "partitioning": "component_vector_identity_v1",
        "higher_is_more_important": higher_is_more_important,
        "task_count": len(task_vectors),
        "vectors": vectors,
        "skipped": skipped,
    }


def _average_tie_percentile_vector(
    scores: Mapping[int, float],
    *,
    higher_is_more_important: bool,
) -> dict[int, float]:
    if not scores:
        return {}
    ordered = sorted(
        scores.items(),
        key=lambda item: item[1],
        reverse=not higher_is_more_important,
    )
    result: dict[int, float] = {}
    index = 0
    size = len(ordered)
    while index < size:
        end = index + 1
        while end < size and ordered[end][1] == ordered[index][1]:
            end += 1
        percentile = (((index + 1) + end) / 2.0) / size
        for component_index, _score in ordered[index:end]:
            result[component_index] = percentile
        index = end
    return result


__all__ = [
    "aggregate_task_percentiles",
    "aggregate_task_vector_percentiles",
    "finite_metric_count",
    "select_ranking_metric",
]
