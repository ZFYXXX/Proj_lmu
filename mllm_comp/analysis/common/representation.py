"""Pure representation-dynamics formulas and sample aggregation."""

from __future__ import annotations

import math
from typing import Any, Iterable, Mapping, Sequence


def cosine_similarity_rows(
    left: Sequence[Sequence[float]],
    right: Sequence[Sequence[float]],
    *,
    eps: float = 1e-12,
) -> list[float]:
    if len(left) != len(right):
        raise ValueError("Cosine inputs must contain the same number of rows.")
    result: list[float] = []
    for left_row, right_row in zip(left, right):
        if len(left_row) != len(right_row) or not left_row:
            raise ValueError("Cosine rows must be non-empty and shape-aligned.")
        dot = sum(float(a) * float(b) for a, b in zip(left_row, right_row))
        left_norm = math.sqrt(sum(float(value) ** 2 for value in left_row))
        right_norm = math.sqrt(sum(float(value) ** 2 for value in right_row))
        result.append(dot / max(left_norm * right_norm, eps))
    return result


class RepresentationSampleAccumulator:
    """Online representation aggregation with fixed layer/group state."""

    def __init__(self) -> None:
        self._totals: dict[tuple[str, str, str], float] = {}
        self._counts: dict[tuple[str, str, str], int] = {}
        self._absent: dict[tuple[str, str, str], int] = {}
        self._sample_count = 0

    def add(self, sample: Mapping[str, Any]) -> None:
        self._sample_count += 1
        for layer, comparisons in sample.get("layers", {}).items():
            for comparison, groups in comparisons.items():
                for group, metrics in groups.items():
                    key = (str(layer), str(comparison), str(group))
                    value = metrics.get("cosine")
                    if value is None:
                        self._absent[key] = self._absent.get(key, 0) + 1
                        continue
                    numeric = float(value)
                    if math.isfinite(numeric):
                        self._totals[key] = self._totals.get(key, 0.0) + numeric
                        self._counts[key] = self._counts.get(key, 0) + 1
                    else:
                        self._absent[key] = self._absent.get(key, 0) + 1

    def result(self) -> dict[str, Any]:
        layers: dict[str, dict] = {}
        keys = sorted(set(self._totals) | set(self._absent))
        for layer, comparison, group in keys:
            key = (layer, comparison, group)
            count = self._counts.get(key, 0)
            layers.setdefault(layer, {}).setdefault(comparison, {})[group] = {
                "cosine": self._totals.get(key, 0.0) / count if count else None,
                "finite_sample_count": count,
                "absent_or_nonfinite_sample_count": self._absent.get(key, 0),
            }
        return {
            "aggregation": "mean_over_samples_v1",
            "sample_count": self._sample_count,
            "layers": layers,
        }


def aggregate_representation_samples(samples: Iterable[Mapping]) -> dict:
    accumulator = RepresentationSampleAccumulator()
    for sample in samples:
        accumulator.add(sample)
    return accumulator.result()


__all__ = [
    "RepresentationSampleAccumulator",
    "aggregate_representation_samples",
    "cosine_similarity_rows",
]
