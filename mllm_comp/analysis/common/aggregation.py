"""Small online accumulators shared by analysis aggregations."""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence


class FiniteMetricMeans:
    """Accumulate finite metric means without retaining source reports."""

    def __init__(self, metrics: Sequence[str]) -> None:
        self.metrics = tuple(str(metric) for metric in metrics)
        self._totals = {metric: 0.0 for metric in self.metrics}
        self._counts = {metric: 0 for metric in self.metrics}

    def add(self, report: Mapping[str, Any]) -> None:
        for metric in self.metrics:
            value = report.get(metric)
            if value is None:
                continue
            numeric = float(value)
            if not math.isfinite(numeric):
                continue
            self._totals[metric] += numeric
            self._counts[metric] += 1

    def result(self) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for metric in self.metrics:
            count = self._counts[metric]
            result[metric] = self._totals[metric] / count if count else None
            result[f"{metric}_finite_sample_count"] = count
        return result


__all__ = ["FiniteMetricMeans"]
