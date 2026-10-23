"""Direct per-layer, per-head multimodal attention routing metrics."""

from __future__ import annotations

import math
from typing import Any, Iterable, Mapping, Sequence

from mllm_comp.analysis.common.aggregation import FiniteMetricMeans
from mllm_comp.analysis.common.token_mask import TokenGroupLayout
from mllm_comp.core.errors import BackendIntegrationError


DEFAULT_ATTENTION_GROUPS = ("system", "text", "vision")
ATTENTION_GROUPS = (*DEFAULT_ATTENTION_GROUPS, "audio")


def attention_flow_metrics(groups: Sequence[str]) -> tuple[str, ...]:
    """Return the exact metric vocabulary implied by selected token groups."""

    selected = tuple(dict.fromkeys(str(group) for group in groups))
    metrics = [
        f"{query}_to_{key}_mass"
        for query in selected
        for key in selected
    ]
    metrics.extend(f"decision_to_{key}_mass" for key in selected)
    for target in ("vision", "audio"):
        if target not in selected:
            continue
        metrics.extend(
            (
                f"decision_to_{target}_entropy",
                f"decision_to_{target}_entropy_norm",
                f"text_to_{target}_entropy",
                f"text_to_{target}_entropy_norm",
            )
        )
    return tuple(metrics)


DEFAULT_ATTENTION_FLOW_METRICS = attention_flow_metrics(DEFAULT_ATTENTION_GROUPS)
ATTENTION_FLOW_METRICS = attention_flow_metrics(ATTENTION_GROUPS)


def attention_head_metrics(
    attention: Any,
    *,
    labels: Sequence[str],
    decision_position: int,
    groups: Sequence[str] = DEFAULT_ATTENTION_GROUPS,
    mass_tolerance: float = 1e-5,
) -> dict[str, float | None]:
    """Compute direct routing metrics for one [query, key] attention head."""

    if hasattr(attention, "detach"):
        return _torch_head_metrics(
            attention,
            labels=labels,
            decision_position=decision_position,
            groups=groups,
        )
    rows = [[float(value) for value in row] for row in attention]
    if not rows or any(len(row) != len(rows) for row in rows):
        raise ValueError("Attention head must be a non-empty square matrix.")
    if len(labels) != len(rows):
        raise ValueError("Attention labels do not match matrix length.")
    valid = [index for index, label in enumerate(labels) if label != "padding"]
    if not valid:
        raise ValueError("Attention matrix has no valid query rows.")
    max_error = max(abs(sum(rows[index]) - 1.0) for index in valid)
    if max_error > mass_tolerance:
        raise ValueError(
            f"Attention row mass is not conserved (max_error={max_error:.6g})."
        )
    masks = {group: _positions(labels, group) for group in groups}
    result: dict[str, float | None] = {}
    for query in groups:
        for key in groups:
            result[f"{query}_to_{key}_mass"] = _python_mass(
                rows,
                masks[query],
                masks[key],
            )
    decision = (
        [decision_position]
        if decision_position in valid
        else []
    )
    for key in groups:
        result[f"decision_to_{key}_mass"] = _python_mass(
            rows,
            decision,
            masks[key],
        )
    for target in ("vision", "audio"):
        if target not in groups:
            continue
        decision_entropy = _python_entropy(rows, decision, masks[target])
        text_entropy = _python_entropy(rows, masks.get("text", []), masks[target])
        result[f"decision_to_{target}_entropy"] = decision_entropy[0]
        result[f"decision_to_{target}_entropy_norm"] = decision_entropy[1]
        result[f"text_to_{target}_entropy"] = text_entropy[0]
        result[f"text_to_{target}_entropy_norm"] = text_entropy[1]
    return result


def compute_attention_flow_layer(
    attention: Any,
    *,
    layout: TokenGroupLayout,
    decision_position: int,
    groups: Sequence[str] = DEFAULT_ATTENTION_GROUPS,
    mass_tolerance: float = 2e-3,
) -> dict[str, Any]:
    """Compute each head plus layer mean from [1, head, query, key] weights."""

    try:
        import torch
    except ImportError as exc:  # pragma: no cover
        raise BackendIntegrationError("Attention flow requires PyTorch.") from exc
    if len(attention.shape) != 4 or int(attention.shape[0]) != 1:
        raise BackendIntegrationError(
            "Attention flow requires weights shaped [1, head, query, key]."
        )
    query_length = int(attention.shape[-2])
    key_length = int(attention.shape[-1])
    if query_length != key_length:
        raise BackendIntegrationError(
            f"Attention flow requires square self-attention, got {query_length}x{key_length}."
        )
    if len(layout.labels_by_sample) != 1 or len(layout.labels_by_sample[0]) != query_length:
        raise BackendIntegrationError("Attention flow token layout length mismatch.")
    labels = layout.labels_by_sample[0]
    valid_positions = _positions(labels, "all")
    if not valid_positions:
        raise BackendIntegrationError("Attention flow has no valid query positions.")
    if not bool(attention.is_floating_point()):
        raise BackendIntegrationError("Attention flow weights must be floating point.")
    dtype_epsilon = float(torch.finfo(attention.dtype).eps)
    effective_tolerance = max(mass_tolerance, dtype_epsilon)
    indices = torch.tensor(valid_positions, dtype=torch.long, device=attention.device)
    row_sums = attention[0].float().sum(dim=-1).index_select(1, indices)
    max_error = float((row_sums - 1.0).abs().max().detach().cpu().item())
    if not math.isfinite(max_error) or max_error > effective_tolerance:
        raise BackendIntegrationError(
            "Attention row mass is not conserved on valid queries "
            f"(max_error={max_error:.6g}, "
            f"effective_tolerance={effective_tolerance:.6g}, "
            f"configured_tolerance={mass_tolerance:.6g}, dtype={attention.dtype})."
        )
    heads = {
        str(head): attention_head_metrics(
            attention[0, head],
            labels=labels,
            decision_position=decision_position,
            groups=groups,
        )
        for head in range(int(attention.shape[1]))
    }
    mean: dict[str, float | None] = {}
    metric_names = sorted({key for metrics in heads.values() for key in metrics})
    for metric in metric_names:
        values = [
            float(metrics[metric])
            for metrics in heads.values()
            if metrics.get(metric) is not None
            and math.isfinite(float(metrics[metric]))
        ]
        mean[metric] = sum(values) / len(values) if values else None
    return {
        "mean": mean,
        "heads": heads,
        "mass_conservation": {
            "passed": True,
            "max_abs_row_sum_error": max_error,
            "tolerance": effective_tolerance,
            "configured_tolerance": mass_tolerance,
            "dtype_epsilon": dtype_epsilon,
            "weight_dtype": str(attention.dtype).removeprefix("torch."),
            "valid_query_count": len(valid_positions),
        },
    }


class AttentionFlowSampleAccumulator:
    """Online per-layer and per-head routing metric means."""

    def __init__(self, *, metrics: Sequence[str]) -> None:
        self.metrics = tuple(str(metric) for metric in metrics)
        self._sample_count = 0
        self._layers: dict[str, dict[str, Any]] = {}

    def add(self, sample: Mapping[str, Any]) -> None:
        self._sample_count += 1
        for layer, report in sample.get("layers", {}).items():
            layer_key = str(layer)
            bucket = self._layers.get(layer_key)
            if bucket is None:
                bucket = {
                    "mean": FiniteMetricMeans(self.metrics),
                    "heads": {},
                    "mass_conservation": _AttentionMassConservationAccumulator(),
                }
                self._layers[layer_key] = bucket
            bucket["mean"].add(report.get("mean", {}))
            bucket["mass_conservation"].add(report)
            for head, values in report.get("heads", {}).items():
                head_key = str(head)
                accumulator = bucket["heads"].get(head_key)
                if accumulator is None:
                    accumulator = FiniteMetricMeans(self.metrics)
                    bucket["heads"][head_key] = accumulator
                accumulator.add(values)

    def result(self) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for layer in sorted(self._layers, key=lambda value: int(value)):
            result[layer] = {
                "mean": self._layers[layer]["mean"].result(),
                "mass_conservation": self._layers[layer][
                    "mass_conservation"
                ].result(),
                "heads": {
                    head: accumulator.result()
                    for head, accumulator in sorted(
                        self._layers[layer]["heads"].items(),
                        key=lambda item: int(item[0]),
                    )
                },
            }
        return {"sample_count": self._sample_count, "layers": result}


class _AttentionMassConservationAccumulator:
    """Retain the validation contract when raw sample artifacts are disabled."""

    def __init__(self) -> None:
        self._sample_count = 0
        self._max_error = 0.0
        self._configured_tolerance: float | None = None
        self._dtype_contracts: dict[str, tuple[float, float]] = {}
        self._valid_query_count_min: int | None = None
        self._valid_query_count_max: int | None = None

    def add(self, report: Mapping[str, Any]) -> None:
        evidence = report.get("mass_conservation")
        if not isinstance(evidence, Mapping) or evidence.get("passed") is not True:
            raise BackendIntegrationError(
                "Attention-flow aggregation requires passed mass-conservation evidence."
            )
        max_error = _finite_evidence_float(evidence, "max_abs_row_sum_error")
        configured = _finite_evidence_float(evidence, "configured_tolerance")
        effective = _finite_evidence_float(evidence, "tolerance")
        dtype_epsilon = _finite_evidence_float(evidence, "dtype_epsilon")
        weight_dtype = evidence.get("weight_dtype")
        valid_query_count = evidence.get("valid_query_count")
        if (
            max_error < 0.0
            or configured < 0.0
            or effective < 0.0
            or dtype_epsilon < 0.0
            or not isinstance(weight_dtype, str)
            or not weight_dtype
            or isinstance(valid_query_count, bool)
            or not isinstance(valid_query_count, int)
            or valid_query_count <= 0
        ):
            raise BackendIntegrationError(
                "Attention-flow mass-conservation evidence is invalid."
            )
        expected_effective = max(configured, dtype_epsilon)
        if not math.isclose(effective, expected_effective, rel_tol=1e-12, abs_tol=0.0):
            raise BackendIntegrationError(
                "Attention-flow effective mass tolerance does not match its "
                "configured and dtype tolerances."
            )
        if max_error > effective:
            raise BackendIntegrationError(
                "Attention-flow aggregation received failed mass-conservation evidence."
            )
        if self._configured_tolerance is None:
            self._configured_tolerance = configured
        elif not math.isclose(
            configured,
            self._configured_tolerance,
            rel_tol=1e-12,
            abs_tol=0.0,
        ):
            raise BackendIntegrationError(
                "Attention-flow samples used different configured mass tolerances."
            )
        dtype_contract = self._dtype_contracts.get(weight_dtype)
        if dtype_contract is None:
            self._dtype_contracts[weight_dtype] = (dtype_epsilon, effective)
        elif not (
            math.isclose(dtype_epsilon, dtype_contract[0], rel_tol=1e-12, abs_tol=0.0)
            and math.isclose(effective, dtype_contract[1], rel_tol=1e-12, abs_tol=0.0)
        ):
            raise BackendIntegrationError(
                f"Attention-flow dtype contract changed for {weight_dtype}."
            )
        self._sample_count += 1
        self._max_error = max(self._max_error, max_error)
        self._valid_query_count_min = (
            valid_query_count
            if self._valid_query_count_min is None
            else min(self._valid_query_count_min, valid_query_count)
        )
        self._valid_query_count_max = (
            valid_query_count
            if self._valid_query_count_max is None
            else max(self._valid_query_count_max, valid_query_count)
        )

    def result(self) -> dict[str, Any]:
        return {
            "contract": "dtype_aware_valid_query_row_mass_v1",
            "passed": self._sample_count > 0,
            "sample_count": self._sample_count,
            "max_abs_row_sum_error": self._max_error,
            "configured_tolerance": self._configured_tolerance,
            "weight_dtypes": sorted(self._dtype_contracts),
            "dtype_epsilon_by_weight_dtype": {
                dtype: contract[0]
                for dtype, contract in sorted(self._dtype_contracts.items())
            },
            "effective_tolerance_by_weight_dtype": {
                dtype: contract[1]
                for dtype, contract in sorted(self._dtype_contracts.items())
            },
            "valid_query_count_min": self._valid_query_count_min,
            "valid_query_count_max": self._valid_query_count_max,
        }


def _finite_evidence_float(evidence: Mapping[str, Any], key: str) -> float:
    value = evidence.get(key)
    if isinstance(value, bool):
        raise BackendIntegrationError(
            f"Attention-flow mass-conservation evidence {key} must be finite."
        )
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise BackendIntegrationError(
            f"Attention-flow mass-conservation evidence {key} must be finite."
        ) from exc
    if not math.isfinite(number):
        raise BackendIntegrationError(
            f"Attention-flow mass-conservation evidence {key} must be finite."
        )
    return number


def aggregate_attention_flow_samples(
    samples: Iterable[Mapping[str, Any]],
    *,
    metrics: Sequence[str],
) -> dict[str, Any]:
    accumulator = AttentionFlowSampleAccumulator(metrics=metrics)
    for sample in samples:
        accumulator.add(sample)
    return accumulator.result()


def _torch_head_metrics(
    attention: Any,
    *,
    labels: Sequence[str],
    decision_position: int,
    groups: Sequence[str],
) -> dict[str, float | None]:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover
        raise BackendIntegrationError("Tensor attention metrics require PyTorch.") from exc
    if len(attention.shape) != 2 or int(attention.shape[0]) != int(attention.shape[1]):
        raise BackendIntegrationError("Attention head must be a square matrix.")
    if len(labels) != int(attention.shape[0]):
        raise BackendIntegrationError("Attention labels do not match matrix length.")
    masks = {
        group: torch.tensor(
            [index in _positions(labels, group) for index in range(len(labels))],
            dtype=torch.bool,
            device=attention.device,
        )
        for group in groups
    }
    decision = torch.zeros(len(labels), dtype=torch.bool, device=attention.device)
    if 0 <= decision_position < len(labels) and labels[decision_position] != "padding":
        decision[decision_position] = True
    result: dict[str, float | None] = {}
    for query in groups:
        for key in groups:
            result[f"{query}_to_{key}_mass"] = _torch_mass(
                attention,
                masks[query],
                masks[key],
            )
    for key in groups:
        result[f"decision_to_{key}_mass"] = _torch_mass(
            attention,
            decision,
            masks[key],
        )
    for target in ("vision", "audio"):
        if target not in groups:
            continue
        decision_entropy = _torch_entropy(attention, decision, masks[target])
        text_entropy = _torch_entropy(
            attention,
            masks.get("text", torch.zeros_like(decision)),
            masks[target],
        )
        result[f"decision_to_{target}_entropy"] = decision_entropy[0]
        result[f"decision_to_{target}_entropy_norm"] = decision_entropy[1]
        result[f"text_to_{target}_entropy"] = text_entropy[0]
        result[f"text_to_{target}_entropy_norm"] = text_entropy[1]
    return result


def _torch_mass(attention: Any, query_mask: Any, key_mask: Any) -> float | None:
    if int(query_mask.sum().item()) == 0 or int(key_mask.sum().item()) == 0:
        return None
    selected = attention[query_mask][:, key_mask]
    return float(selected.sum(dim=-1).float().mean().detach().cpu().item())


def _torch_entropy(attention: Any, query_mask: Any, key_mask: Any) -> tuple[float | None, float | None]:
    key_count = int(key_mask.sum().item())
    if int(query_mask.sum().item()) == 0 or key_count == 0:
        return None, None
    selected = attention[query_mask][:, key_mask].float()
    totals = selected.sum(dim=-1)
    reachable = totals > 0.0
    if int(reachable.sum().item()) == 0:
        return None, None
    selected = selected[reachable]
    probabilities = selected / totals[reachable].unsqueeze(-1)
    entropy = -(
        probabilities * probabilities.clamp_min(1e-12).log()
    ).sum(dim=-1).mean()
    value = float(entropy.detach().cpu().item())
    return value, value / math.log(max(key_count, 2))


def _python_mass(
    attention: Sequence[Sequence[float]],
    queries: Sequence[int],
    keys: Sequence[int],
) -> float | None:
    if not queries or not keys:
        return None
    return sum(
        sum(attention[query][key] for key in keys)
        for query in queries
    ) / len(queries)


def _python_entropy(
    attention: Sequence[Sequence[float]],
    queries: Sequence[int],
    keys: Sequence[int],
) -> tuple[float | None, float | None]:
    if not queries or not keys:
        return None, None
    entropies: list[float] = []
    for query in queries:
        selected = [attention[query][key] for key in keys]
        total = sum(selected)
        if total <= 0.0:
            continue
        probabilities = [value / total for value in selected]
        entropies.append(
            -sum(value * math.log(max(value, 1e-12)) for value in probabilities)
        )
    if not entropies:
        return None, None
    value = sum(entropies) / len(entropies)
    return value, value / math.log(max(len(keys), 2))


def _positions(labels: Sequence[str], group: str) -> list[int]:
    if group == "all":
        return [index for index, label in enumerate(labels) if label != "padding"]
    if group == "vision":
        return [
            index for index, label in enumerate(labels) if label in {"image", "video"}
        ]
    return [index for index, label in enumerate(labels) if label == group]


__all__ = [
    "ATTENTION_FLOW_METRICS",
    "ATTENTION_GROUPS",
    "DEFAULT_ATTENTION_FLOW_METRICS",
    "DEFAULT_ATTENTION_GROUPS",
    "AttentionFlowSampleAccumulator",
    "aggregate_attention_flow_samples",
    "attention_head_metrics",
    "attention_flow_metrics",
    "compute_attention_flow_layer",
]
