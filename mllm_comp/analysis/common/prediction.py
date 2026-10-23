"""First-answer-token logit-lens metrics and aggregation."""

from __future__ import annotations

import math
from typing import Any, Iterable, Mapping, Sequence

from mllm_comp.analysis.common.aggregation import FiniteMetricMeans
from mllm_comp.analysis.common.distribution import categorical_distribution_metrics
from mllm_comp.core.errors import BackendIntegrationError


PREDICTION_METRICS = (
    "kl_final_to_layer",
    "kl_layer_to_final",
    "js",
    "layer_entropy",
    "final_entropy",
    "top1_matches_final",
    "layer_top1_prob",
    "final_top1_prob",
    "final_top1_prob_under_layer",
    "layer_top1_prob_under_final",
    "rank_of_final_top1",
)


def prediction_dynamics_metrics(
    layer_logits: Any,
    final_logits: Any,
    *,
    top_k: int,
    tokenizer: Any | None = None,
) -> dict[str, Any]:
    """Compare one layer's logit lens to the final next-token distribution."""

    if top_k < 1:
        raise ValueError("prediction dynamics top_k must be positive.")
    distribution = categorical_distribution_metrics(final_logits, layer_logits)
    if hasattr(layer_logits, "detach") or hasattr(final_logits, "detach"):
        details = _torch_prediction_details(
            layer_logits,
            final_logits,
            top_k=top_k,
            tokenizer=tokenizer,
        )
    else:
        details = _python_prediction_details(
            layer_logits,
            final_logits,
            top_k=top_k,
            tokenizer=tokenizer,
        )
    return {
        "metrics": {
            "kl_final_to_layer": distribution["kl_pq"],
            "kl_layer_to_final": distribution["kl_qp"],
            "js": distribution["js"],
            "layer_entropy": distribution["changed_entropy"],
            "final_entropy": distribution["base_entropy"],
            "top1_matches_final": 1.0 - distribution["top1_changed"],
            "layer_top1_prob": distribution["changed_top1_prob"],
            "final_top1_prob": distribution["base_top1_prob"],
            "final_top1_prob_under_layer": distribution[
                "changed_prob_on_base_top1"
            ],
            "layer_top1_prob_under_final": details[
                "layer_top1_prob_under_final"
            ],
            "rank_of_final_top1": float(details["rank_of_final_top1"]),
        },
        "layer_top1": details["layer_top1"],
        "final_top1": details["final_top1"],
        "top_tokens": details["top_tokens"],
    }


class PredictionSampleAccumulator:
    """Online layer-wise logit-lens metric means."""

    def __init__(self, *, metrics: Sequence[str]) -> None:
        self.metrics = tuple(str(metric) for metric in metrics)
        self._sample_count = 0
        self._layers: dict[str, FiniteMetricMeans] = {}

    def add(self, sample: Mapping[str, Any]) -> None:
        self._sample_count += 1
        for layer, report in sample.get("layers", {}).items():
            layer_key = str(layer)
            accumulator = self._layers.get(layer_key)
            if accumulator is None:
                accumulator = FiniteMetricMeans(self.metrics)
                self._layers[layer_key] = accumulator
            accumulator.add(report.get("metrics", {}))

    def result(self) -> dict[str, Any]:
        return {
            "sample_count": self._sample_count,
            "layers": {
                layer: self._layers[layer].result()
                for layer in sorted(self._layers, key=lambda value: int(value))
            },
        }


def aggregate_prediction_samples(
    samples: Iterable[Mapping[str, Any]],
    *,
    metrics: Sequence[str],
) -> dict[str, Any]:
    accumulator = PredictionSampleAccumulator(metrics=metrics)
    for sample in samples:
        accumulator.add(sample)
    return accumulator.result()


def _torch_prediction_details(
    layer_logits: Any,
    final_logits: Any,
    *,
    top_k: int,
    tokenizer: Any | None,
) -> dict[str, Any]:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover
        raise BackendIntegrationError("Tensor prediction metrics require PyTorch.") from exc
    if len(layer_logits.shape) != 1 or tuple(layer_logits.shape) != tuple(final_logits.shape):
        raise BackendIntegrationError("Prediction logits must share [vocabulary] shape.")
    layer_p = torch.softmax(layer_logits.float(), dim=-1)
    final_p = torch.softmax(final_logits.float(), dim=-1)
    layer_top1 = int(layer_p.argmax().detach().cpu().item())
    final_top1 = int(final_p.argmax().detach().cpu().item())
    rank = int((layer_p > layer_p[final_top1]).sum().detach().cpu().item()) + 1
    count = min(top_k, int(layer_p.numel()))
    probabilities, token_ids = torch.topk(layer_p, k=count)
    return {
        "layer_top1_prob_under_final": float(
            final_p[layer_top1].detach().cpu().item()
        ),
        "rank_of_final_top1": rank,
        "layer_top1": _token_record(
            layer_top1,
            float(layer_p[layer_top1].detach().cpu().item()),
            tokenizer,
        ),
        "final_top1": _token_record(
            final_top1,
            float(final_p[final_top1].detach().cpu().item()),
            tokenizer,
        ),
        "top_tokens": [
            _token_record(
                int(token_id.detach().cpu().item()),
                float(probability.detach().cpu().item()),
                tokenizer,
            )
            for probability, token_id in zip(probabilities, token_ids)
        ],
    }


def _python_prediction_details(
    layer_logits: Any,
    final_logits: Any,
    *,
    top_k: int,
    tokenizer: Any | None,
) -> dict[str, Any]:
    layer_values = [float(value) for value in layer_logits]
    final_values = [float(value) for value in final_logits]
    if not layer_values or len(layer_values) != len(final_values):
        raise ValueError("Prediction logits must share a non-empty vocabulary shape.")
    layer_p = _softmax(layer_values)
    final_p = _softmax(final_values)
    layer_top1 = max(range(len(layer_p)), key=layer_p.__getitem__)
    final_top1 = max(range(len(final_p)), key=final_p.__getitem__)
    ordered = sorted(range(len(layer_p)), key=layer_p.__getitem__, reverse=True)
    return {
        "layer_top1_prob_under_final": final_p[layer_top1],
        "rank_of_final_top1": ordered.index(final_top1) + 1,
        "layer_top1": _token_record(layer_top1, layer_p[layer_top1], tokenizer),
        "final_top1": _token_record(final_top1, final_p[final_top1], tokenizer),
        "top_tokens": [
            _token_record(token_id, layer_p[token_id], tokenizer)
            for token_id in ordered[: min(top_k, len(ordered))]
        ],
    }


def _token_record(token_id: int, probability: float, tokenizer: Any | None) -> dict[str, Any]:
    decoded = ""
    decode = getattr(tokenizer, "decode", None)
    if callable(decode):
        try:
            decoded = str(decode([int(token_id)]))
        except Exception:
            decoded = ""
    return {"token_id": int(token_id), "token": decoded, "prob": float(probability)}


def _softmax(values: Sequence[float]) -> list[float]:
    maximum = max(values)
    exponentials = [math.exp(value - maximum) for value in values]
    denominator = sum(exponentials)
    return [value / denominator for value in exponentials]


__all__ = [
    "PREDICTION_METRICS",
    "PredictionSampleAccumulator",
    "aggregate_prediction_samples",
    "prediction_dynamics_metrics",
]
