"""Residual-node capture, update metrics, and aggregation."""

from __future__ import annotations

import math
from typing import Any, Iterable, Mapping, Sequence

from mllm_comp.analysis.bindings import DecoderLayerBinding
from mllm_comp.analysis.common.aggregation import FiniteMetricMeans
from mllm_comp.analysis.common.distribution import LayerImportanceSampleAccumulator
from mllm_comp.analysis.common.tensors import hidden_tensor
from mllm_comp.analysis.common.token_mask import TokenGroupLayout
from mllm_comp.analysis.hooks import HookSession
from mllm_comp.core.errors import BackendIntegrationError


RESIDUAL_METRICS = (
    "input_norm",
    "output_norm",
    "update_norm",
    "residual_ratio",
    "cosine_to_input",
)


class ResidualNodeCapture:
    """Collect the five semantic pre-norm decoder nodes and verify reconstruction."""

    def __init__(self, layers: Sequence[DecoderLayerBinding]) -> None:
        self.layers = tuple(layers)
        self.values: dict[int, dict[str, Any]] = {}

    def install(self, hooks: HookSession) -> None:
        for binding in self.layers:
            index = binding.index
            hooks.register_forward_pre_hook(
                binding.layer,
                lambda _module, args, index=index: self._store(
                    index,
                    "before_attn",
                    hidden_tensor(args),
                ),
            )
            hooks.register_forward_hook(
                binding.attention,
                lambda _module, _args, output, index=index: self._store(
                    index,
                    "attn_only",
                    hidden_tensor(output),
                ),
            )
            hooks.register_forward_hook(
                binding.ffn,
                lambda _module, _args, output, index=index: self._store(
                    index,
                    "ffn_only",
                    hidden_tensor(output),
                ),
            )
            hooks.register_forward_hook(
                binding.layer,
                lambda _module, _args, output, index=index: self._store(
                    index,
                    "after_ffn",
                    hidden_tensor(output),
                ),
            )

    def _store(self, layer: int, point: str, value: Any) -> None:
        bucket = self.values.setdefault(layer, {})
        if point in bucket:
            raise BackendIntegrationError(
                f"Residual hook {point} at layer {layer} fired more than once."
            )
        # Provider code may mutate a decoder layer's returned tensor after the
        # layer forward hook fires (Qwen3-VL DeepStack does this at visual
        # positions). Keep a point-in-time residual node, not an aliased view.
        bucket[point] = value.detach().clone()

    def finish(self, *, atol: float, rtol: float) -> tuple[dict, dict]:
        try:
            import torch
        except ImportError as exc:  # pragma: no cover
            raise BackendIntegrationError("Residual checks require PyTorch.") from exc
        nodes: dict[int, dict[str, Any]] = {}
        evidence: dict[str, Any] = {}
        for binding in self.layers:
            values = self.values.get(binding.index, {})
            missing = sorted(
                {"before_attn", "attn_only", "ffn_only", "after_ffn"}
                - set(values)
            )
            if missing:
                raise BackendIntegrationError(
                    f"Residual layer {binding.index} missed hooks: {missing}."
                )
            shapes = {point: tuple(value.shape) for point, value in values.items()}
            if len(set(shapes.values())) != 1 or len(next(iter(shapes.values()))) != 3:
                raise BackendIntegrationError(
                    f"Residual layer {binding.index} hidden shapes disagree: {shapes}."
                )
            after_attn = values["before_attn"] + values["attn_only"]
            reconstructed = after_attn + values["ffn_only"]
            difference = (reconstructed.float() - values["after_ffn"].float()).abs()
            max_abs_error = float(difference.max().detach().cpu().item())
            if not bool(
                torch.allclose(
                    reconstructed.float(),
                    values["after_ffn"].float(),
                    atol=atol,
                    rtol=rtol,
                )
            ):
                raise BackendIntegrationError(
                    f"Residual layer {binding.index} failed the pre-norm contract "
                    f"(max_abs_error={max_abs_error:.6g})."
                )
            nodes[binding.index] = {**values, "after_attn": after_attn}
            evidence[str(binding.index)] = {
                "passed": True,
                "max_abs_error": max_abs_error,
                "atol": atol,
                "rtol": rtol,
            }
        return nodes, evidence

    def clear(self) -> None:
        self.values.clear()


def residual_vector_metrics(before: Any, after: Any) -> dict[str, float] | None:
    """Mean residual metrics over selected [token, hidden] vectors."""

    if hasattr(before, "detach") or hasattr(after, "detach"):
        return _torch_residual_vector_metrics(before, after)
    return _python_residual_vector_metrics(before, after)


def summarize_residual_updates(
    before: Any,
    after: Any,
    *,
    layout: TokenGroupLayout,
    token_groups: Sequence[str],
) -> dict[str, Any]:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover
        raise BackendIntegrationError("Residual update summaries require PyTorch.") from exc
    if tuple(before.shape) != tuple(after.shape) or len(before.shape) != 3:
        raise BackendIntegrationError(
            "Residual update tensors must share [batch, sequence, hidden] shape."
        )
    if int(before.shape[0]) != 1 or len(layout.labels_by_sample) != 1:
        raise BackendIntegrationError("Residual update summaries require batch size one.")
    if len(layout.labels_by_sample[0]) != int(before.shape[1]):
        raise BackendIntegrationError("Residual layout does not match hidden sequence length.")
    valid = tuple(
        index
        for index, label in enumerate(layout.labels_by_sample[0])
        if label != "padding"
    )
    result: dict[str, Any] = {}
    for group in token_groups:
        positions = valid if group == "all" else layout.positions(0, group)
        if not positions:
            result[group] = {
                **{metric: None for metric in RESIDUAL_METRICS},
                "token_count": 0,
                "status": "absent",
            }
            continue
        indices = torch.tensor(positions, dtype=torch.long, device=before.device)
        metrics = residual_vector_metrics(
            before[0].index_select(0, indices),
            after[0].index_select(0, indices),
        )
        assert metrics is not None
        result[group] = {
            **metrics,
            "token_count": len(positions),
            "status": "finite",
        }
    return result


class ResidualSampleAccumulator:
    """Online residual-stat and causal-intervention aggregation."""

    def __init__(
        self,
        *,
        residual_metrics: Sequence[str],
        distribution_metrics: Sequence[str],
        token_groups: Sequence[str],
    ) -> None:
        self.residual_metrics = tuple(str(metric) for metric in residual_metrics)
        self._sample_count = 0
        self._residual_states: dict[
            str,
            dict[str, dict[str, FiniteMetricMeans]],
        ] = {}
        self._causal = LayerImportanceSampleAccumulator(
            metrics=distribution_metrics,
            token_groups=[group for group in token_groups if group != "all"],
        )

    def add(self, sample: Mapping[str, Any]) -> None:
        self._sample_count += 1
        for layer, operations in sample.get("residual_stats", {}).items():
            for operation, groups in operations.items():
                for group, metrics in groups.items():
                    layer_key = str(layer)
                    operation_key = str(operation)
                    group_key = str(group)
                    accumulator = (
                        self._residual_states.setdefault(layer_key, {})
                        .setdefault(operation_key, {})
                        .get(group_key)
                    )
                    if accumulator is None:
                        accumulator = FiniteMetricMeans(self.residual_metrics)
                        self._residual_states[layer_key][operation_key][
                            group_key
                        ] = accumulator
                    accumulator.add(metrics)
        self._causal.add({"layers": sample.get("causal", {})})

    def result(self) -> dict[str, Any]:
        residual_stats: dict[str, Any] = {}
        for layer in sorted(self._residual_states, key=lambda value: int(value)):
            residual_stats[layer] = {}
            for operation, groups in sorted(self._residual_states[layer].items()):
                residual_stats[layer][operation] = {
                    group: accumulator.result()
                    for group, accumulator in sorted(groups.items())
                }
        causal = self._causal.result()["layers"]
        return {
            "sample_count": self._sample_count,
            "residual_stats": residual_stats,
            "causal": causal,
        }


def aggregate_residual_samples(
    samples: Iterable[Mapping[str, Any]],
    *,
    residual_metrics: Sequence[str],
    distribution_metrics: Sequence[str],
    token_groups: Sequence[str],
) -> dict[str, Any]:
    accumulator = ResidualSampleAccumulator(
        residual_metrics=residual_metrics,
        distribution_metrics=distribution_metrics,
        token_groups=token_groups,
    )
    for sample in samples:
        accumulator.add(sample)
    return accumulator.result()


def _torch_residual_vector_metrics(before: Any, after: Any) -> dict[str, float] | None:
    try:
        import torch.nn.functional as functional
    except ImportError as exc:  # pragma: no cover
        raise BackendIntegrationError("Tensor residual metrics require PyTorch.") from exc
    if tuple(before.shape) != tuple(after.shape) or len(before.shape) != 2:
        raise BackendIntegrationError("Residual vectors must share [token, hidden] shape.")
    if int(before.shape[0]) == 0:
        return None
    left = before.float()
    right = after.float()
    update = right - left
    input_norm = left.norm(dim=-1)
    values = {
        "input_norm": input_norm,
        "output_norm": right.norm(dim=-1),
        "update_norm": update.norm(dim=-1),
        "residual_ratio": update.norm(dim=-1) / input_norm.clamp_min(1e-12),
        "cosine_to_input": functional.cosine_similarity(
            left,
            right,
            dim=-1,
            eps=1e-12,
        ),
    }
    return {
        name: float(value.mean().detach().cpu().item())
        for name, value in values.items()
    }


def _python_residual_vector_metrics(before: Any, after: Any) -> dict[str, float] | None:
    left_rows = [[float(value) for value in row] for row in before]
    right_rows = [[float(value) for value in row] for row in after]
    if len(left_rows) != len(right_rows) or any(
        len(left) != len(right)
        for left, right in zip(left_rows, right_rows)
    ):
        raise ValueError("Residual vector shapes disagree.")
    if not left_rows:
        return None
    rows: list[dict[str, float]] = []
    for left, right in zip(left_rows, right_rows):
        input_norm = math.sqrt(sum(value * value for value in left))
        output_norm = math.sqrt(sum(value * value for value in right))
        update_norm = math.sqrt(
            sum((right_value - left_value) ** 2 for left_value, right_value in zip(left, right))
        )
        denominator = max(input_norm * output_norm, 1e-12)
        rows.append(
            {
                "input_norm": input_norm,
                "output_norm": output_norm,
                "update_norm": update_norm,
                "residual_ratio": update_norm / max(input_norm, 1e-12),
                "cosine_to_input": sum(
                    left_value * right_value
                    for left_value, right_value in zip(left, right)
                )
                / denominator,
            }
        )
    return {
        metric: sum(row[metric] for row in rows) / len(rows)
        for metric in RESIDUAL_METRICS
    }


__all__ = [
    "RESIDUAL_METRICS",
    "ResidualNodeCapture",
    "ResidualSampleAccumulator",
    "aggregate_residual_samples",
    "residual_vector_metrics",
    "summarize_residual_updates",
]
