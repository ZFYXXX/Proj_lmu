"""Distribution-shift metrics shared by intervention-based analyses."""

from __future__ import annotations

import math
from typing import Any, Iterable, Mapping, Sequence

from mllm_comp.analysis.common.aggregation import FiniteMetricMeans
from mllm_comp.analysis.common.token_mask import TokenGroupLayout
from mllm_comp.core.errors import BackendIntegrationError


DISTRIBUTION_METRICS = (
    "kl_pq",
    "kl_qp",
    "js",
    "base_entropy",
    "changed_entropy",
    "base_top1_prob",
    "changed_top1_prob",
    "changed_prob_on_base_top1",
    "top1_changed",
)

# Only metrics with an unambiguous intervention-importance direction belong in
# task rankings. Baseline-only quantities are constant across interventions;
# changed entropy/top-1 confidence may move in either scientifically meaningful
# direction and therefore are intentionally excluded.
IMPORTANCE_RANKING_DIRECTIONS = {
    "kl_pq": True,
    "kl_qp": True,
    "js": True,
    "changed_prob_on_base_top1": False,
    "top1_changed": True,
}
UNAVAILABLE_REASON_EXAMPLE_LIMIT = 20


def categorical_distribution_metrics(
    base_logits: Any,
    changed_logits: Any,
) -> dict[str, float]:
    """Compute P-vs-Q metrics for one or more categorical logit rows.

    Plain sequences use a dependency-free reference implementation for contract
    tests. Runtime tensors use a vectorized lazy-PyTorch implementation.
    """

    if hasattr(base_logits, "detach") or hasattr(changed_logits, "detach"):
        return _torch_categorical_metrics(base_logits, changed_logits)
    return _python_categorical_metrics(base_logits, changed_logits)


def distribution_shift_report(
    base_logits: Any,
    changed_logits: Any,
    *,
    layout: TokenGroupLayout,
    token_groups: Sequence[str],
    metrics: Sequence[str] = DISTRIBUTION_METRICS,
) -> dict[str, Any]:
    """Compare the final prompt decision and modality-grouped prefill logits."""

    try:
        import torch
    except ImportError as exc:  # pragma: no cover - real runtime requires torch
        raise BackendIntegrationError(
            "Distribution-shift reports require PyTorch tensors."
        ) from exc
    if tuple(base_logits.shape) != tuple(changed_logits.shape):
        raise BackendIntegrationError(
            "Baseline and intervention logits shapes disagree: "
            f"{tuple(base_logits.shape)} vs {tuple(changed_logits.shape)}."
        )
    if len(base_logits.shape) != 3 or int(base_logits.shape[0]) != 1:
        raise BackendIntegrationError(
            "Distribution-shift reports currently require logits shaped "
            "[1, sequence, vocabulary]."
        )
    sequence_length = int(base_logits.shape[1])
    if len(layout.labels_by_sample) != 1 or len(layout.labels_by_sample[0]) != sequence_length:
        raise BackendIntegrationError(
            "Token-group layout does not match the intervention logits sequence."
        )
    selected_metrics = tuple(dict.fromkeys(str(value) for value in metrics))
    unknown = sorted(set(selected_metrics) - set(DISTRIBUTION_METRICS))
    if unknown:
        raise BackendIntegrationError(f"Unknown distribution metrics: {unknown}.")

    valid_positions = tuple(
        index
        for index, label in enumerate(layout.labels_by_sample[0])
        if label != "padding"
    )
    if not valid_positions:
        raise BackendIntegrationError("Analysis prompt has no valid decision token.")
    decision_position = valid_positions[-1]
    decision = categorical_distribution_metrics(
        base_logits[0, decision_position],
        changed_logits[0, decision_position],
    )
    report: dict[str, Any] = {
        "decision_position": decision_position,
        "decision": {name: decision[name] for name in selected_metrics},
        "token_groups": {},
    }
    for group in token_groups:
        positions = (
            valid_positions if group == "all" else layout.positions(0, group)
        )
        if not positions:
            report["token_groups"][group] = {
                **{name: None for name in selected_metrics},
                "token_count": 0,
                "status": "absent",
            }
            continue
        indices = torch.tensor(positions, dtype=torch.long, device=base_logits.device)
        values = categorical_distribution_metrics(
            base_logits[0].index_select(0, indices),
            changed_logits[0].index_select(0, indices),
        )
        report["token_groups"][group] = {
            **{name: values[name] for name in selected_metrics},
            "token_count": len(positions),
            "status": "finite",
        }
    return report


class LayerImportanceSampleAccumulator:
    """Online means for intervention reports without zero-filling absences."""

    def __init__(
        self,
        *,
        metrics: Sequence[str],
        token_groups: Sequence[str],
    ) -> None:
        self.metrics = tuple(str(metric) for metric in metrics)
        self.token_groups = tuple(str(group) for group in token_groups)
        self._sample_count = 0
        self._states: dict[str, dict[str, dict[str, Any]]] = {}

    def add(self, sample: Mapping[str, Any]) -> None:
        self._sample_count += 1
        for layer, experiments in sample.get("layers", {}).items():
            for experiment, report in experiments.items():
                layer_key = str(layer)
                experiment_key = str(experiment)
                state = self._states.setdefault(layer_key, {}).get(experiment_key)
                if state is None:
                    state = {
                        "finite_count": 0,
                        "unavailable_count": 0,
                        "unavailable_reasons": set(),
                        "unavailable_reasons_truncated": False,
                        "decision": FiniteMetricMeans(self.metrics),
                        "token_groups": {
                            group: FiniteMetricMeans(self.metrics)
                            for group in self.token_groups
                        },
                    }
                    self._states[layer_key][experiment_key] = state
                if report.get("status", "finite") != "finite":
                    state["unavailable_count"] += 1
                    reason = str(
                        report.get("reason", report.get("status", "unavailable"))
                    )
                    if reason not in state["unavailable_reasons"]:
                        if (
                            len(state["unavailable_reasons"])
                            < UNAVAILABLE_REASON_EXAMPLE_LIMIT
                        ):
                            state["unavailable_reasons"].add(reason)
                        else:
                            state["unavailable_reasons_truncated"] = True
                    continue
                state["finite_count"] += 1
                state["decision"].add(report.get("decision", {}))
                for group, accumulator in state["token_groups"].items():
                    accumulator.add(report.get("token_groups", {}).get(group, {}))

    def result(self) -> dict[str, Any]:
        result_layers: dict[str, Any] = {}
        for layer in sorted(self._states, key=_layer_sort_key):
            result_layers[layer] = {}
            for experiment in sorted(self._states[layer]):
                state = self._states[layer][experiment]
                result_layers[layer][experiment] = {
                    "status": (
                        "finite" if state["finite_count"] else "unavailable"
                    ),
                    "sample_count": state["finite_count"],
                    "unavailable_sample_count": state["unavailable_count"],
                    "unavailable_reasons": sorted(state["unavailable_reasons"]),
                    "unavailable_reasons_truncated": state[
                        "unavailable_reasons_truncated"
                    ],
                    "unavailable_reason_example_limit": (
                        UNAVAILABLE_REASON_EXAMPLE_LIMIT
                    ),
                    "decision": state["decision"].result(),
                    "token_groups": {
                        group: accumulator.result()
                        for group, accumulator in state["token_groups"].items()
                    },
                }
        return {"sample_count": self._sample_count, "layers": result_layers}


def aggregate_layer_importance_samples(
    samples: Iterable[Mapping[str, Any]],
    *,
    metrics: Sequence[str],
    token_groups: Sequence[str],
) -> dict[str, Any]:
    """Mean valid intervention reports in one pass over any iterable."""

    accumulator = LayerImportanceSampleAccumulator(
        metrics=metrics,
        token_groups=token_groups,
    )
    for sample in samples:
        accumulator.add(sample)
    return accumulator.result()


def _torch_categorical_metrics(base_logits: Any, changed_logits: Any) -> dict[str, float]:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover
        raise BackendIntegrationError("Tensor distribution metrics require PyTorch.") from exc
    if tuple(base_logits.shape) != tuple(changed_logits.shape):
        raise BackendIntegrationError(
            "Categorical logit shapes disagree: "
            f"{tuple(base_logits.shape)} vs {tuple(changed_logits.shape)}."
        )
    if len(base_logits.shape) == 1:
        base_logits = base_logits.unsqueeze(0)
        changed_logits = changed_logits.unsqueeze(0)
    if len(base_logits.shape) != 2:
        raise BackendIntegrationError("Categorical logits must be [rows, vocabulary].")
    base_logp = torch.log_softmax(base_logits.float(), dim=-1)
    changed_logp = torch.log_softmax(changed_logits.float(), dim=-1)
    base_p = base_logp.exp()
    changed_p = changed_logp.exp()
    mixture = 0.5 * (base_p + changed_p)
    mixture_logp = mixture.clamp_min(torch.finfo(mixture.dtype).tiny).log()
    base_top1 = base_p.argmax(dim=-1)
    changed_top1 = changed_p.argmax(dim=-1)
    values = {
        "kl_pq": (base_p * (base_logp - changed_logp)).sum(dim=-1),
        "kl_qp": (changed_p * (changed_logp - base_logp)).sum(dim=-1),
        "js": 0.5
        * (
            (base_p * (base_logp - mixture_logp)).sum(dim=-1)
            + (changed_p * (changed_logp - mixture_logp)).sum(dim=-1)
        ),
        "base_entropy": -(base_p * base_logp).sum(dim=-1),
        "changed_entropy": -(changed_p * changed_logp).sum(dim=-1),
        "base_top1_prob": base_p.gather(-1, base_top1.unsqueeze(-1)).squeeze(-1),
        "changed_top1_prob": changed_p.gather(
            -1, changed_top1.unsqueeze(-1)
        ).squeeze(-1),
        "changed_prob_on_base_top1": changed_p.gather(
            -1, base_top1.unsqueeze(-1)
        ).squeeze(-1),
        "top1_changed": (base_top1 != changed_top1).float(),
    }
    return {
        name: float(value.mean().detach().cpu().item())
        for name, value in values.items()
    }


def _python_categorical_metrics(base_logits: Any, changed_logits: Any) -> dict[str, float]:
    base_rows = _logit_rows(base_logits)
    changed_rows = _logit_rows(changed_logits)
    if len(base_rows) != len(changed_rows) or any(
        len(base) != len(changed)
        for base, changed in zip(base_rows, changed_rows)
    ):
        raise ValueError("Categorical logit shapes disagree.")
    if not base_rows or not base_rows[0]:
        raise ValueError("Categorical logits cannot be empty.")
    rows: list[dict[str, float]] = []
    for base, changed in zip(base_rows, changed_rows):
        base_p, base_logp = _softmax_and_log(base)
        changed_p, changed_logp = _softmax_and_log(changed)
        mixture = [(left + right) / 2.0 for left, right in zip(base_p, changed_p)]
        base_top1 = max(range(len(base_p)), key=base_p.__getitem__)
        changed_top1 = max(range(len(changed_p)), key=changed_p.__getitem__)
        rows.append(
            {
                "kl_pq": sum(
                    probability * (left - right)
                    for probability, left, right in zip(base_p, base_logp, changed_logp)
                ),
                "kl_qp": sum(
                    probability * (right - left)
                    for probability, left, right in zip(
                        changed_p, base_logp, changed_logp
                    )
                ),
                "js": 0.5
                * (
                    sum(
                        probability * (logp - math.log(mix))
                        for probability, logp, mix in zip(base_p, base_logp, mixture)
                    )
                    + sum(
                        probability * (logp - math.log(mix))
                        for probability, logp, mix in zip(
                            changed_p, changed_logp, mixture
                        )
                    )
                ),
                "base_entropy": -sum(
                    probability * logp for probability, logp in zip(base_p, base_logp)
                ),
                "changed_entropy": -sum(
                    probability * logp
                    for probability, logp in zip(changed_p, changed_logp)
                ),
                "base_top1_prob": base_p[base_top1],
                "changed_top1_prob": changed_p[changed_top1],
                "changed_prob_on_base_top1": changed_p[base_top1],
                "top1_changed": float(base_top1 != changed_top1),
            }
        )
    return {
        name: sum(row[name] for row in rows) / len(rows)
        for name in DISTRIBUTION_METRICS
    }


def _logit_rows(value: Any) -> list[list[float]]:
    if not isinstance(value, (list, tuple)):
        raise ValueError("Plain categorical logits must be sequences.")
    if not value:
        return []
    if isinstance(value[0], (int, float)):
        return [[float(item) for item in value]]
    return [[float(item) for item in row] for row in value]


def _softmax_and_log(logits: Sequence[float]) -> tuple[list[float], list[float]]:
    maximum = max(logits)
    shifted = [float(value) - maximum for value in logits]
    denominator = sum(math.exp(value) for value in shifted)
    log_denominator = math.log(denominator)
    logp = [value - log_denominator for value in shifted]
    return [math.exp(value) for value in logp], logp


def _layer_sort_key(value: str) -> tuple[int, str]:
    try:
        return (0, f"{int(value):012d}")
    except ValueError:
        return (1, value)


__all__ = [
    "DISTRIBUTION_METRICS",
    "IMPORTANCE_RANKING_DIRECTIONS",
    "LayerImportanceSampleAccumulator",
    "UNAVAILABLE_REASON_EXAMPLE_LIMIT",
    "aggregate_layer_importance_samples",
    "categorical_distribution_metrics",
    "distribution_shift_report",
]
