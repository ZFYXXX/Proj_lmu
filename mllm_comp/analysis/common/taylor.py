"""Shared first-order Taylor math with dependency-free reference formulas."""

from __future__ import annotations

from contextlib import AbstractContextManager
import math
from typing import Any, Iterable, Sequence

from mllm_comp.core.errors import BackendIntegrationError


TAYLOR_ESTIMATOR_VERSION = "modality_hierarchical_taylor_v1"
NORMALIZED_SCORE = "mean_abs_gradient_times_activation"
DIRECT_REMOVAL_SCORE = "abs_sum_gradient_times_activation"
PATH_NORMALIZED_SCORE = "abs_sum_gradient_times_path_contribution_per_query"
PATH_TYPES = ("v2v", "t2v", "t2t")
TAYLOR_TARGET_MODALITIES = ("vision", "audio")


def taylor_component_groups(target_modality: str) -> tuple[str, str, str]:
    if target_modality not in TAYLOR_TARGET_MODALITIES:
        raise ValueError(f"Unsupported Taylor target modality {target_modality!r}.")
    return ("all", target_modality, "text")


def taylor_path_groups(
    target_modality: str,
) -> dict[str, tuple[str, str]]:
    if target_modality not in TAYLOR_TARGET_MODALITIES:
        raise ValueError(f"Unsupported Taylor target modality {target_modality!r}.")
    alias = "v" if target_modality == "vision" else "a"
    return {
        f"{alias}2{alias}": (target_modality, target_modality),
        f"t2{alias}": ("text", target_modality),
        "t2t": ("text", "text"),
    }


def scalar_taylor_reductions(products: Iterable[float]) -> dict[str, Any]:
    """Reduce elementwise ``gradient * activation`` without losing sign order.

    The normalized ranking view takes the mean *after* elementwise absolute
    value.  The direct-removal view takes the absolute value only after the
    signed component sum.  Keeping these operations separate prevents a sum of
    absolute child scores from masquerading as whole-component removal.
    """

    values = [float(value) for value in products]
    if not values:
        raise ValueError("Taylor reduction requires at least one product value.")
    non_finite = sum(not math.isfinite(value) for value in values)
    if non_finite:
        return {
            "status": "non_finite",
            "element_count": len(values),
            "finite_count": len(values) - non_finite,
            "non_finite_count": non_finite,
            "normalized_score": None,
            "direct_removal_score": None,
            "signed_sum": None,
            "normalized_score_definition": NORMALIZED_SCORE,
            "direct_removal_score_definition": DIRECT_REMOVAL_SCORE,
        }
    signed_sum = math.fsum(values)
    return {
        "status": "finite",
        "element_count": len(values),
        "finite_count": len(values),
        "non_finite_count": 0,
        "normalized_score": math.fsum(abs(value) for value in values) / len(values),
        "direct_removal_score": abs(signed_sum),
        "signed_sum": signed_sum,
        "normalized_score_definition": NORMALIZED_SCORE,
        "direct_removal_score_definition": DIRECT_REMOVAL_SCORE,
    }


def path_taylor_reduction(*, signed_sum: float, query_count: int) -> dict[str, Any]:
    """Return the exact path ranking and removal views for one head/block."""

    if query_count <= 0:
        raise ValueError("Path Taylor query_count must be positive.")
    value = float(signed_sum)
    if not math.isfinite(value):
        return {
            "status": "non_finite",
            "query_count": int(query_count),
            "normalized_score": None,
            "direct_removal_score": None,
            "signed_sum": None,
            "normalized_score_definition": PATH_NORMALIZED_SCORE,
            "direct_removal_score_definition": DIRECT_REMOVAL_SCORE,
        }
    return {
        "status": "finite",
        "query_count": int(query_count),
        "normalized_score": abs(value) / query_count,
        "direct_removal_score": abs(value),
        "signed_sum": value,
        "normalized_score_definition": PATH_NORMALIZED_SCORE,
        "direct_removal_score_definition": DIRECT_REMOVAL_SCORE,
    }


def path_block_taylor_reference(
    attention: Sequence[Sequence[Sequence[float]]],
    values: Sequence[Sequence[Sequence[float]]],
    gradients: Sequence[Sequence[Sequence[float]]],
) -> list[dict[str, Any]]:
    """Dependency-free reference for ``sum(g * (attention_block @ V))``.

    Shapes are ``attention=[head, query, key]``, ``values=[head, key, dim]``
    and ``gradients=[head, query, dim]``.  The production tensor path is tested
    against this intentionally small implementation.
    """

    if not attention:
        raise ValueError("Path Taylor requires at least one head.")
    if len(attention) != len(values) or len(attention) != len(gradients):
        raise ValueError("Path Taylor head dimensions do not match.")
    reports: list[dict[str, Any]] = []
    for head_attention, head_values, head_gradients in zip(
        attention, values, gradients
    ):
        query_count = len(head_attention)
        key_count = len(head_values)
        if query_count <= 0 or key_count <= 0:
            raise ValueError("Path Taylor query/key blocks must be non-empty.")
        dimension = len(head_values[0])
        if dimension <= 0 or any(len(row) != dimension for row in head_values):
            raise ValueError("Path Taylor value rows must have one stable dimension.")
        if len(head_gradients) != query_count or any(
            len(row) != dimension for row in head_gradients
        ):
            raise ValueError("Path Taylor gradient shape does not match values.")
        products: list[float] = []
        for query_index, attention_row in enumerate(head_attention):
            if len(attention_row) != key_count:
                raise ValueError("Path Taylor attention key dimensions do not match.")
            contribution = [
                math.fsum(
                    float(attention_row[key_index])
                    * float(head_values[key_index][dim_index])
                    for key_index in range(key_count)
                )
                for dim_index in range(dimension)
            ]
            products.extend(
                float(head_gradients[query_index][dim_index])
                * contribution[dim_index]
                for dim_index in range(dimension)
            )
        elementwise = scalar_taylor_reductions(products)
        report = path_taylor_reduction(
            signed_sum=float(elementwise["signed_sum"])
            if elementwise["signed_sum"] is not None
            else float("nan"),
            query_count=query_count,
        )
        report["elementwise_normalized_score"] = elementwise["normalized_score"]
        report["element_count"] = elementwise["element_count"]
        reports.append(report)
    return reports


def neuron_taylor_reference(
    activations: Sequence[Sequence[float]],
    gradients: Sequence[Sequence[float]],
    *,
    positions: Sequence[int] | None = None,
) -> list[dict[str, Any]]:
    """Reference token-normalized and whole-neuron Taylor scores."""

    if len(activations) != len(gradients) or not activations:
        raise ValueError("Neuron Taylor activation/gradient token shapes do not match.")
    width = len(activations[0])
    if width <= 0 or any(len(row) != width for row in activations) or any(
        len(row) != width for row in gradients
    ):
        raise ValueError("Neuron Taylor rows must have one stable width.")
    selected = tuple(range(len(activations))) if positions is None else tuple(positions)
    if not selected or any(index < 0 or index >= len(activations) for index in selected):
        raise ValueError("Neuron Taylor positions must select valid tokens.")
    return [
        scalar_taylor_reductions(
            float(activations[token][neuron]) * float(gradients[token][neuron])
            for token in selected
        )
        for neuron in range(width)
    ]


def activation_magnitude_reference(
    activations: Sequence[Sequence[float]],
    *,
    positions: Sequence[int] | None = None,
) -> list[float]:
    """Mean absolute FFN intermediate activation per selected token."""

    if not activations:
        raise ValueError("Activation magnitude requires at least one token.")
    width = len(activations[0])
    if width <= 0 or any(len(row) != width for row in activations):
        raise ValueError("Activation rows must have one stable width.")
    selected = tuple(range(len(activations))) if positions is None else tuple(positions)
    if not selected or any(index < 0 or index >= len(activations) for index in selected):
        raise ValueError("Activation positions must select valid tokens.")
    return [
        math.fsum(abs(float(activations[token][neuron])) for token in selected)
        / len(selected)
        for neuron in range(width)
    ]


def modality_neuron_partition(
    vision_scores: Sequence[float],
    text_scores: Sequence[float],
    *,
    bottom_fraction: float,
    modality_name: str = "vision",
) -> dict[str, Any]:
    """Partition neurons by bottom-fraction importance in each modality."""

    if len(vision_scores) != len(text_scores) or not vision_scores:
        raise ValueError("Vision/text neuron score vectors must have equal non-zero length.")
    if not 0.0 < bottom_fraction < 1.0:
        raise ValueError("bottom_fraction must be strictly between zero and one.")
    values = [float(value) for value in (*vision_scores, *text_scores)]
    if any(not math.isfinite(value) for value in values):
        raise ValueError("Neuron partition scores must be finite.")
    count = len(vision_scores)
    selected_count = max(1, min(count - 1, int(round(bottom_fraction * count))))
    vision_bottom = set(
        sorted(range(count), key=lambda index: (vision_scores[index], index))[
            :selected_count
        ]
    )
    text_bottom = set(
        sorted(range(count), key=lambda index: (text_scores[index], index))[
            :selected_count
        ]
    )
    modality_redundant_label = f"{modality_name}_redundant_text_important"
    text_redundant_label = f"text_redundant_{modality_name}_important"
    categories: list[str] = []
    for index in range(count):
        vision_redundant = index in vision_bottom
        text_redundant = index in text_bottom
        if vision_redundant and text_redundant:
            categories.append("both_redundant")
        elif vision_redundant:
            categories.append(modality_redundant_label)
        elif text_redundant:
            categories.append(text_redundant_label)
        else:
            categories.append("both_important")
    counts = {
        category: categories.count(category)
        for category in (
            "both_redundant",
            modality_redundant_label,
            text_redundant_label,
            "both_important",
        )
    }
    return {
        "bottom_fraction": float(bottom_fraction),
        "selected_per_modality": selected_count,
        "categories": categories,
        "counts": counts,
        "fractions": {
            category: value / count for category, value in counts.items()
        },
    }


def spearman_average_ties(left: Sequence[float], right: Sequence[float]) -> float:
    """Spearman correlation with average ranks for ties."""

    if len(left) != len(right) or len(left) < 2:
        raise ValueError("Spearman inputs must have equal length of at least two.")
    left_values = [float(value) for value in left]
    right_values = [float(value) for value in right]
    if any(not math.isfinite(value) for value in (*left_values, *right_values)):
        raise ValueError("Spearman inputs must be finite.")
    left_ranks = _average_ranks(left_values)
    right_ranks = _average_ranks(right_values)
    left_mean = math.fsum(left_ranks) / len(left_ranks)
    right_mean = math.fsum(right_ranks) / len(right_ranks)
    covariance = math.fsum(
        (x - left_mean) * (y - right_mean)
        for x, y in zip(left_ranks, right_ranks)
    )
    left_scale = math.sqrt(
        math.fsum((value - left_mean) ** 2 for value in left_ranks)
    )
    right_scale = math.sqrt(
        math.fsum((value - right_mean) ** 2 for value in right_ranks)
    )
    if left_scale == 0.0 or right_scale == 0.0:
        return float("nan")
    return covariance / (left_scale * right_scale)


def tensor_taylor_reductions(
    product: Any,
    *,
    reduce_dims: int | tuple[int, ...],
    normalization_count: int | None = None,
) -> dict[str, Any]:
    """Vectorized production reduction, imported lazily to keep CPU tests light."""

    try:
        import torch
    except ImportError as exc:  # pragma: no cover - real runtime requires torch
        raise BackendIntegrationError("Taylor tensor reductions require PyTorch.") from exc
    if not isinstance(product, torch.Tensor) or product.numel() == 0:
        raise BackendIntegrationError("Taylor product must be one non-empty tensor.")
    if not bool(torch.isfinite(product).all().item()):
        raise BackendIntegrationError("Taylor product contains non-finite values.")
    dims = (reduce_dims,) if isinstance(reduce_dims, int) else tuple(reduce_dims)
    absolute_sum = product.float().abs().sum(dim=dims)
    signed_sum = product.float().sum(dim=dims)
    if normalization_count is None:
        normalization_count = math.prod(int(product.shape[index]) for index in dims)
    if normalization_count <= 0:
        raise BackendIntegrationError("Taylor normalization count must be positive.")
    return {
        "normalized": absolute_sum / normalization_count,
        "direct_removal": signed_sum.abs(),
        "signed_sum": signed_sum,
        "normalization_count": int(normalization_count),
    }


class FrozenModelParameters(AbstractContextManager["FrozenModelParameters"]):
    """Temporarily freeze model parameters and restore their original flags."""

    def __init__(self, model: Any) -> None:
        self.model = model
        self._states: list[tuple[Any, bool]] = []

    def __enter__(self) -> "FrozenModelParameters":
        parameters = getattr(self.model, "parameters", None)
        if not callable(parameters):
            raise BackendIntegrationError("Taylor runtime model exposes no parameters().")
        self._states = [
            (parameter, bool(getattr(parameter, "requires_grad", False)))
            for parameter in parameters()
        ]
        for parameter, _ in self._states:
            setter = getattr(parameter, "requires_grad_", None)
            if not callable(setter):
                raise BackendIntegrationError(
                    "Taylor runtime parameter does not support requires_grad_()."
                )
            setter(False)
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> bool:
        errors: list[str] = []
        for parameter, state in self._states:
            try:
                parameter.requires_grad_(state)
            except Exception as restore_error:  # cleanup must try every parameter
                errors.append(f"{type(restore_error).__name__}: {restore_error}")
        self._states.clear()
        if errors and exc is None:
            raise BackendIntegrationError(
                "Taylor parameter-state restoration failed: " + "; ".join(errors)
            )
        return False


def _average_ranks(values: Sequence[float]) -> list[float]:
    ordered = sorted(range(len(values)), key=lambda index: (values[index], index))
    result = [0.0] * len(values)
    cursor = 0
    while cursor < len(ordered):
        end = cursor + 1
        while end < len(ordered) and values[ordered[end]] == values[ordered[cursor]]:
            end += 1
        average = ((cursor + 1) + end) / 2.0
        for position in range(cursor, end):
            result[ordered[position]] = average
        cursor = end
    return result


__all__ = [
    "DIRECT_REMOVAL_SCORE",
    "FrozenModelParameters",
    "NORMALIZED_SCORE",
    "PATH_NORMALIZED_SCORE",
    "PATH_TYPES",
    "TAYLOR_TARGET_MODALITIES",
    "TAYLOR_ESTIMATOR_VERSION",
    "activation_magnitude_reference",
    "modality_neuron_partition",
    "taylor_component_groups",
    "taylor_path_groups",
    "neuron_taylor_reference",
    "path_block_taylor_reference",
    "path_taylor_reduction",
    "scalar_taylor_reductions",
    "spearman_average_ties",
    "tensor_taylor_reductions",
]
