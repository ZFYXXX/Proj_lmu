"""Streaming task aggregation for hierarchical Taylor component scores."""

from __future__ import annotations

import math
from typing import Any, Mapping

from mllm_comp.analysis.common.taylor import (
    PATH_TYPES,
    modality_neuron_partition,
    spearman_average_ties,
)


SCORE_FAMILIES = (
    "normalized_taylor",
    "direct_removal_taylor",
    "activation_magnitude",
)
TAYLOR_GRANULARITIES = (
    "path",
    "path_child_mass",
    "head",
    "ffn_neuron",
    "sublayer",
    "whole_layer",
)


class TaylorTaskAccumulator:
    """Accumulate finite component values without retaining sample tensors."""

    def __init__(self) -> None:
        self._sums: dict[str, dict[str, dict[str, float]]] = {}
        self._counts: dict[str, dict[str, dict[str, int]]] = {}
        self._vector_sums: dict[
            str, dict[str, dict[str, list[float]]]
        ] = {}
        self._vector_counts: dict[
            str, dict[str, dict[str, list[int]]]
        ] = {}
        self.sample_counts: dict[str, int] = {}

    def update(self, task: str, report: Mapping[str, Any]) -> int:
        flattened = flatten_hierarchical_taylor(
            report,
            include_ffn_neurons=False,
        )
        task_name = str(task)
        self.sample_counts[task_name] = self.sample_counts.get(task_name, 0) + 1
        accepted = 0
        for family, values in flattened.items():
            family_sums = self._sums.setdefault(task_name, {}).setdefault(family, {})
            family_counts = self._counts.setdefault(task_name, {}).setdefault(
                family, {}
            )
            for component, raw_value in values.items():
                value = float(raw_value)
                if not math.isfinite(value):
                    continue
                family_sums[component] = family_sums.get(component, 0.0) + value
                family_counts[component] = family_counts.get(component, 0) + 1
                accepted += 1
        for family, vector_id, values in _ffn_score_vectors(report):
            sums = self._vector_sums.setdefault(task_name, {}).setdefault(
                family, {}
            )
            counts = self._vector_counts.setdefault(task_name, {}).setdefault(
                family, {}
            )
            if vector_id not in sums:
                sums[vector_id] = [0.0] * len(values)
                counts[vector_id] = [0] * len(values)
            elif len(sums[vector_id]) != len(values):
                raise ValueError(
                    f"Taylor component vector {vector_id!r} changes length."
                )
            for index, raw_value in enumerate(values):
                value = float(raw_value)
                if not math.isfinite(value):
                    continue
                sums[vector_id][index] += value
                counts[vector_id][index] += 1
                accepted += 1
        return accepted

    def task_means(self) -> dict[str, dict[str, dict[str, float]]]:
        result: dict[str, dict[str, dict[str, float]]] = {}
        for task, families in self._sums.items():
            result[task] = {}
            for family, values in families.items():
                result[task][family] = {
                    component: value / self._counts[task][family][component]
                    for component, value in values.items()
                    if self._counts[task][family][component] > 0
                }
        return result

    def scores_by_family(self) -> dict[str, dict[str, dict[str, float]]]:
        means = self.task_means()
        return {
            family: {
                task: task_families.get(family, {})
                for task, task_families in means.items()
            }
            for family in SCORE_FAMILIES
        }

    def vector_scores_by_family(
        self,
    ) -> dict[str, dict[str, dict[str, list[float]]]]:
        """Return compact FFN-neuron vectors without per-neuron string keys."""

        result: dict[str, dict[str, dict[str, list[float]]]] = {
            family: {} for family in SCORE_FAMILIES
        }
        for task, families in self._vector_sums.items():
            for family, vectors in families.items():
                result[family][task] = {}
                for vector_id, sums in vectors.items():
                    counts = self._vector_counts[task][family][vector_id]
                    if any(count <= 0 for count in counts):
                        raise ValueError(
                            f"Taylor component vector {vector_id!r} has a value "
                            "with no finite samples."
                        )
                    result[family][task][vector_id] = [
                        value / counts[index]
                        for index, value in enumerate(sums)
                    ]
        return result


def flatten_hierarchical_taylor(
    report: Mapping[str, Any],
    *,
    include_ffn_neurons: bool = True,
) -> dict[str, dict[str, float]]:
    """Flatten stable component identities while keeping estimator families typed."""

    result = {family: {} for family in SCORE_FAMILIES}
    for layer_key, layer in report.get("layers", {}).items():
        layer_prefix = f"layer:{int(layer_key)}"
        for path_name, path in layer.get("path", {}).items():
            _append_vector_scores(
                result,
                path,
                prefix=f"{layer_prefix}/path:{path_name}/head",
            )
        path_child_mass = layer.get("path_to_head", {}).get(
            "measured_path_normalized_abs_child_mass_by_head"
        )
        if isinstance(path_child_mass, list):
            for index, value in enumerate(path_child_mass):
                result["normalized_taylor"][
                    f"{layer_prefix}/path_child_mass/head:{index}"
                ] = float(value)
        for group, head in layer.get("head", {}).items():
            _append_vector_scores(
                result,
                head,
                prefix=f"{layer_prefix}/head",
                suffix=f"/group:{group}",
            )
        if include_ffn_neurons:
            neuron = layer.get("ffn_neuron", {})
            for group, taylor in neuron.get("taylor", {}).items():
                _append_vector_scores(
                    result,
                    taylor,
                    prefix=f"{layer_prefix}/ffn_neuron",
                    suffix=f"/group:{group}",
                )
            for group, activation in neuron.get(
                "activation_magnitude", {}
            ).items():
                values = activation.get("mean_abs_activation")
                if isinstance(values, list):
                    for index, value in enumerate(values):
                        result["activation_magnitude"][
                            f"{layer_prefix}/ffn_neuron:{index}/group:{group}"
                        ] = float(value)
        for sublayer_name, sublayer in layer.get("sublayer", {}).items():
            for group, values in sublayer.items():
                _append_scalar_scores(
                    result,
                    values,
                    component=(
                        f"{layer_prefix}/sublayer:{sublayer_name}/group:{group}"
                    ),
                )
        for group, values in layer.get("layer", {}).items():
            _append_scalar_scores(
                result,
                values,
                component=f"{layer_prefix}/whole_layer/group:{group}",
            )
    return result


def _ffn_score_vectors(
    report: Mapping[str, Any],
) -> list[tuple[str, str, list[float]]]:
    result: list[tuple[str, str, list[float]]] = []
    for layer_key, layer in report.get("layers", {}).items():
        prefix = f"layer:{int(layer_key)}/ffn_neuron"
        neuron = layer.get("ffn_neuron", {})
        for group, taylor in neuron.get("taylor", {}).items():
            vector_id = f"{prefix}/group:{group}"
            for family, field in (
                ("normalized_taylor", "normalized_score"),
                ("direct_removal_taylor", "direct_removal_score"),
            ):
                values = taylor.get(field)
                if isinstance(values, list):
                    result.append(
                        (family, vector_id, [float(value) for value in values])
                    )
        for group, activation in neuron.get(
            "activation_magnitude", {}
        ).items():
            values = activation.get("mean_abs_activation")
            if isinstance(values, list):
                result.append(
                    (
                        "activation_magnitude",
                        f"{prefix}/group:{group}",
                        [float(value) for value in values],
                    )
                )
    return result


def partition_taylor_task_scores(
    scores_by_family: Mapping[str, Mapping[str, Mapping[str, float]]],
) -> dict[str, dict[str, dict[str, dict[str, float]]]]:
    """Partition rankings by estimator family and comparable granularity."""

    result: dict[str, dict[str, dict[str, dict[str, float]]]] = {}
    for family, tasks in scores_by_family.items():
        family_result = {
            granularity: {str(task): {} for task in tasks}
            for granularity in TAYLOR_GRANULARITIES
        }
        for task, components in tasks.items():
            for component, score in components.items():
                granularity = taylor_component_granularity(component)
                family_result[granularity][str(task)][str(component)] = float(score)
        result[str(family)] = family_result
    return result


def taylor_component_granularity(component: str) -> str:
    markers = (
        ("/path_child_mass/", "path_child_mass"),
        ("/path:", "path"),
        ("/ffn_neuron:", "ffn_neuron"),
        ("/sublayer:", "sublayer"),
        ("/whole_layer/", "whole_layer"),
        ("/head:", "head"),
        ("/head/", "head"),
    )
    for marker, granularity in markers:
        if marker in component:
            return granularity
    raise ValueError(f"Unknown Taylor component granularity: {component!r}.")


def add_bottom_up_mass_views(report: dict[str, Any]) -> None:
    """Attach child-score mass beside, never in place of, direct parent Taylor."""

    component_groups = tuple(
        report.get("component_groups") or ("all", "vision", "text")
    )
    path_types = tuple(report.get("path_types") or PATH_TYPES)
    for layer in report.get("layers", {}).values():
        layer["path_to_head"] = _path_to_head_view(
            layer,
            path_types=path_types,
        )
        views: dict[str, Any] = {}
        for group in component_groups:
            head = layer.get("head", {}).get(group, {})
            neuron = layer.get("ffn_neuron", {}).get("taylor", {}).get(group, {})
            attention = layer.get("sublayer", {}).get("attention", {}).get(group, {})
            ffn = layer.get("sublayer", {}).get("ffn", {}).get(group, {})
            whole = layer.get("layer", {}).get(group, {})
            if whole.get("status") != "finite":
                views[group] = {"status": whole.get("status", "absent")}
                continue
            views[group] = {
                "status": "finite",
                "head_normalized_abs_child_mass": _finite_sum(
                    head.get("normalized_score")
                ),
                "ffn_neuron_normalized_abs_child_mass": _finite_sum(
                    neuron.get("normalized_score")
                ),
                "sublayer_normalized_abs_child_mass": _finite_sum(
                    [
                        attention.get("normalized_score"),
                        ffn.get("normalized_score"),
                    ]
                ),
                "direct_whole_layer_removal_score": whole.get(
                    "direct_removal_score"
                ),
                "interpretation": (
                    "child masses are sums of normalized absolute scores; the direct "
                    "whole-layer score is measured independently from the signed layer update"
                ),
            }
        layer["bottom_up_mass"] = views


def _path_to_head_view(
    layer: Mapping[str, Any],
    *,
    path_types: tuple[str, ...],
) -> dict[str, Any]:
    """Aggregate measured Path ranking scores per Head without claiming removal."""

    measured: list[str] = []
    vectors: list[list[float]] = []
    head_count: int | None = None
    for path_name in path_types:
        path = layer.get("path", {}).get(path_name, {})
        values = path.get("normalized_score")
        if not isinstance(values, list):
            continue
        numeric = [float(value) for value in values]
        if not numeric or any(not math.isfinite(value) for value in numeric):
            continue
        if head_count is None:
            head_count = len(numeric)
        elif len(numeric) != head_count:
            raise ValueError("Taylor Path vectors disagree on the query-head count.")
        measured.append(path_name)
        vectors.append(numeric)

    if head_count is None:
        return {
            "status": "unavailable",
            "measured_path_types": [],
            "missing_path_types": list(path_types),
        }

    child_mass = [
        math.fsum(vector[index] for vector in vectors)
        for index in range(head_count)
    ]
    head = layer.get("head", {}).get("all", {})
    direct_normalized = head.get("normalized_score")
    direct_removal = head.get("direct_removal_score")
    if isinstance(direct_normalized, list) and len(direct_normalized) != head_count:
        raise ValueError("Taylor Path and direct Head vectors disagree on head count.")
    if isinstance(direct_removal, list) and len(direct_removal) != head_count:
        raise ValueError("Taylor Path and direct Head vectors disagree on head count.")
    missing = [path_name for path_name in path_types if path_name not in measured]
    return {
        "status": "finite",
        "declared_path_coverage": "complete" if not missing else "partial",
        "head_count": head_count,
        "measured_path_types": measured,
        "missing_path_types": missing,
        "measured_path_normalized_abs_child_mass_by_head": child_mass,
        "direct_head_normalized_score": (
            [float(value) for value in direct_normalized]
            if isinstance(direct_normalized, list)
            else None
        ),
        "direct_head_removal_score": (
            [float(value) for value in direct_removal]
            if isinstance(direct_removal, list)
            else None
        ),
        "interpretation": (
            "the measured Path score sum is a normalized child-mass view; "
            "direct Head removal is measured independently and is not reconstructed "
            "from absolute Path scores"
        ),
    }


def ffn_modality_cross_check(
    scores_by_family: Mapping[str, Mapping[str, Mapping[str, float]]],
    *,
    bottom_fraction: float,
    vector_scores_by_family: Mapping[
        str, Mapping[str, Mapping[str, list[float]]]
    ]
    | None = None,
    modality_group: str = "vision",
) -> dict[str, Any]:
    """Compare Taylor and activation modality rankings after equal-task averaging."""

    if vector_scores_by_family is not None:
        return _ffn_modality_cross_check_vectors(
            vector_scores_by_family,
            bottom_fraction=bottom_fraction,
            modality_group=modality_group,
        )
    taylor = _equal_task_component_means(
        scores_by_family.get("normalized_taylor", {})
    )
    activation = _equal_task_component_means(
        scores_by_family.get("activation_magnitude", {})
    )
    layer_indices = sorted(
        {
            int(component.split("/", 1)[0].split(":", 1)[1])
            for component in (*taylor, *activation)
            if "/ffn_neuron:" in component
        }
    )
    result: dict[str, Any] = {}
    for layer in layer_indices:
        layer_result: dict[str, Any] = {}
        for estimator, values in (("taylor", taylor), ("activation", activation)):
            modality, text = _neuron_vectors(
                values,
                layer=layer,
                modality_group=modality_group,
            )
            if not modality or not text:
                layer_result[estimator] = {"status": "unavailable"}
                continue
            correlation = spearman_average_ties(modality, text)
            layer_result[estimator] = {
                "status": "finite" if math.isfinite(correlation) else "constant_rank",
                f"{modality_group}_text_spearman": (
                    correlation if math.isfinite(correlation) else None
                ),
                "partition": modality_neuron_partition(
                    modality,
                    text,
                    bottom_fraction=bottom_fraction,
                    modality_name=modality_group,
                ),
                "neuron_count": len(modality),
            }
        result[str(layer)] = layer_result
    return {
        "bottom_fraction": float(bottom_fraction),
        "task_weighting": "equal_task_mean_raw_score_v1",
        "layers": result,
        "warning": (
            "Taylor/activation disagreement is preserved; terminal-layer Taylor "
            "modality claims require the activation cross-check"
        ),
    }


def _ffn_modality_cross_check_vectors(
    scores_by_family: Mapping[
        str, Mapping[str, Mapping[str, list[float]]]
    ],
    *,
    bottom_fraction: float,
    modality_group: str,
) -> dict[str, Any]:
    families = {
        "taylor": _equal_task_vector_means(
            scores_by_family.get("normalized_taylor", {})
        ),
        "activation": _equal_task_vector_means(
            scores_by_family.get("activation_magnitude", {})
        ),
    }
    layer_indices = sorted(
        {
            int(vector_id.split("/", 1)[0].split(":", 1)[1])
            for values in families.values()
            for vector_id in values
        }
    )
    result: dict[str, Any] = {}
    for layer in layer_indices:
        layer_result: dict[str, Any] = {}
        for estimator, values in families.items():
            modality = values.get(
                f"layer:{layer}/ffn_neuron/group:{modality_group}",
                [],
            )
            text = values.get(f"layer:{layer}/ffn_neuron/group:text", [])
            if not modality or not text or len(modality) != len(text):
                layer_result[estimator] = {"status": "unavailable"}
                continue
            correlation = spearman_average_ties(modality, text)
            layer_result[estimator] = {
                "status": (
                    "finite" if math.isfinite(correlation) else "constant_rank"
                ),
                f"{modality_group}_text_spearman": (
                    correlation if math.isfinite(correlation) else None
                ),
                "partition": modality_neuron_partition(
                    modality,
                    text,
                    bottom_fraction=bottom_fraction,
                    modality_name=modality_group,
                ),
                "neuron_count": len(modality),
            }
        result[str(layer)] = layer_result
    return {
        "bottom_fraction": float(bottom_fraction),
        "task_weighting": "equal_task_mean_raw_score_v1",
        "score_storage": "compact_layer_group_vectors_v1",
        "layers": result,
        "warning": (
            "Taylor/activation disagreement is preserved; terminal-layer Taylor "
            "modality claims require the activation cross-check"
        ),
    }


def _equal_task_vector_means(
    values_by_task: Mapping[str, Mapping[str, list[float]]],
) -> dict[str, list[float]]:
    sums: dict[str, list[float]] = {}
    counts: dict[str, list[int]] = {}
    for vectors in values_by_task.values():
        for vector_id, values in vectors.items():
            if vector_id not in sums:
                sums[vector_id] = [0.0] * len(values)
                counts[vector_id] = [0] * len(values)
            elif len(sums[vector_id]) != len(values):
                raise ValueError(
                    f"Taylor vector {vector_id!r} changes length across tasks."
                )
            for index, raw_value in enumerate(values):
                value = float(raw_value)
                if math.isfinite(value):
                    sums[vector_id][index] += value
                    counts[vector_id][index] += 1
    result: dict[str, list[float]] = {}
    for vector_id, values in sums.items():
        if any(count <= 0 for count in counts[vector_id]):
            raise ValueError(
                f"Taylor vector {vector_id!r} has an index with no finite task score."
            )
        result[vector_id] = [
            total / counts[vector_id][index]
            for index, total in enumerate(values)
        ]
    return result


def _append_vector_scores(
    result: dict[str, dict[str, float]],
    values: Mapping[str, Any],
    *,
    prefix: str,
    suffix: str = "",
) -> None:
    normalized = values.get("normalized_score")
    direct = values.get("direct_removal_score")
    if isinstance(normalized, list):
        for index, value in enumerate(normalized):
            result["normalized_taylor"][f"{prefix}:{index}{suffix}"] = float(value)
    if isinstance(direct, list):
        for index, value in enumerate(direct):
            result["direct_removal_taylor"][f"{prefix}:{index}{suffix}"] = float(value)


def _append_scalar_scores(
    result: dict[str, dict[str, float]],
    values: Mapping[str, Any],
    *,
    component: str,
) -> None:
    normalized = values.get("normalized_score")
    direct = values.get("direct_removal_score")
    if normalized is not None:
        result["normalized_taylor"][component] = float(normalized)
    if direct is not None:
        result["direct_removal_taylor"][component] = float(direct)


def _equal_task_component_means(
    values_by_task: Mapping[str, Mapping[str, float]],
) -> dict[str, float]:
    sums: dict[str, float] = {}
    counts: dict[str, int] = {}
    for values in values_by_task.values():
        for component, value in values.items():
            numeric = float(value)
            if not math.isfinite(numeric):
                continue
            sums[component] = sums.get(component, 0.0) + numeric
            counts[component] = counts.get(component, 0) + 1
    return {
        component: total / counts[component]
        for component, total in sums.items()
        if counts[component] > 0
    }


def _neuron_vectors(
    values: Mapping[str, float],
    *,
    layer: int,
    modality_group: str,
) -> tuple[list[float], list[float]]:
    prefix = f"layer:{layer}/ffn_neuron:"
    modality: dict[int, float] = {}
    text: dict[int, float] = {}
    for component, value in values.items():
        if not component.startswith(prefix):
            continue
        remainder = component[len(prefix) :]
        index_text, separator, group = remainder.partition("/group:")
        if not separator:
            continue
        index = int(index_text)
        if group == modality_group:
            modality[index] = float(value)
        elif group == "text":
            text[index] = float(value)
    common = sorted(set(modality) & set(text))
    if common != list(range(len(common))):
        return [], []
    return [modality[index] for index in common], [text[index] for index in common]


def _finite_sum(values: Any) -> float | None:
    if not isinstance(values, (list, tuple)):
        values = [values]
    finite = [
        float(value)
        for value in values
        if value is not None and math.isfinite(float(value))
    ]
    return math.fsum(finite) if finite else None


__all__ = [
    "SCORE_FAMILIES",
    "TAYLOR_GRANULARITIES",
    "TaylorTaskAccumulator",
    "add_bottom_up_mass_views",
    "ffn_modality_cross_check",
    "flatten_hierarchical_taylor",
    "partition_taylor_task_scores",
    "taylor_component_granularity",
]
