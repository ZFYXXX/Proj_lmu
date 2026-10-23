"""Observation and causal deletion of decoder residual updates."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping, Sequence

from mllm_comp.analysis.artifacts import write_streaming_analysis_result
from mllm_comp.analysis.bindings import (
    resolve_analysis_binding,
    select_decoder_layers,
)
from mllm_comp.analysis.common.distribution import (
    DISTRIBUTION_METRICS,
    IMPORTANCE_RANKING_DIRECTIONS,
    distribution_shift_report,
)
from mllm_comp.analysis.common.config import (
    nonnegative_finite_number,
    strict_boolean,
    string_sequence,
    validate_common_probe_config,
    validate_layer_selection,
)
from mllm_comp.analysis.common.interventions import install_sublayer_update_skip
from mllm_comp.analysis.common.rank import (
    aggregate_task_percentiles,
    finite_metric_count,
    select_ranking_metric,
)
from mllm_comp.analysis.common.residual import (
    RESIDUAL_METRICS,
    ResidualNodeCapture,
    ResidualSampleAccumulator,
    summarize_residual_updates,
)
from mllm_comp.analysis.common.sample import prepare_analysis_forward_state
from mllm_comp.analysis.common.streaming import StreamingAnalysisRecords
from mllm_comp.analysis.common.tensors import logits_tensor
from mllm_comp.analysis.common.token_mask import build_token_group_layout
from mllm_comp.analysis.hooks import HookSession
from mllm_comp.core.errors import BackendIntegrationError
from mllm_comp.core.manifest import AnalysisRecord


_CAUSAL_OPERATIONS = {"attn", "ffn"}
_TOKEN_GROUPS = {
    "all",
    "system",
    "text",
    "vision",
    "image",
    "video",
    "audio",
}


class ResidualStreamDecomposition:
    """Quantify Attention/FFN writes and their actual deletion effects."""

    name = "residual_stream_decomposition"
    version = "1.0.0"
    supported_metrics = set(RESIDUAL_METRICS) | set(DISTRIBUTION_METRICS)
    supported_operations = _CAUSAL_OPERATIONS
    supported_token_groups = _TOKEN_GROUPS

    def validate_config(self, config: dict[str, Any]) -> None:
        validate_common_probe_config(config, context="residual decomposition")
        metrics = set(
            string_sequence(
                config,
                "metrics",
                (*RESIDUAL_METRICS, *DISTRIBUTION_METRICS),
                context="residual decomposition",
            )
        )
        if not metrics or not metrics.issubset(self.supported_metrics):
            unknown = sorted(metrics - self.supported_metrics)
            raise ValueError(
                f"Unsupported residual-decomposition metrics: "
                f"{unknown or sorted(metrics)}"
            )
        operations = set(_causal_operations(config))
        if not operations.issubset(self.supported_operations):
            unknown = sorted(operations - self.supported_operations)
            raise ValueError(f"Unsupported residual causal operations: {unknown}")
        if operations and not (metrics & set(DISTRIBUTION_METRICS)):
            raise ValueError(
                "Residual causal operations require at least one distribution metric."
            )
        collect_residual_stats = strict_boolean(
            config,
            "collect_residual_stats",
            True,
            context="residual decomposition",
        )
        if collect_residual_stats and not (
            metrics & set(RESIDUAL_METRICS)
        ):
            raise ValueError(
                "collect_residual_stats requires at least one residual metric."
            )
        groups = set(
            string_sequence(
                config,
                "token_groups",
                ("all", "system", "text", "vision"),
                context="residual decomposition",
            )
        )
        if not groups or not groups.issubset(self.supported_token_groups):
            unknown = sorted(groups - self.supported_token_groups)
            raise ValueError(
                f"Unsupported residual token groups: {unknown or sorted(groups)}"
            )
        _validate_layers(config.get("layers", "all"), field="layers")
        _validate_layers(config.get("causal_layers", config.get("layers", "all")), field="causal_layers")
        distribution_metrics = metrics & set(DISTRIBUTION_METRICS)
        ranking_metric = config.get("ranking_metric")
        if ranking_metric is not None and (
            not isinstance(ranking_metric, str)
            or ranking_metric not in distribution_metrics
        ):
            raise ValueError(
                "residual ranking_metric must be a configured distribution metric."
            )
        rankable = distribution_metrics & set(IMPORTANCE_RANKING_DIRECTIONS)
        if ranking_metric is not None and ranking_metric not in rankable:
            raise ValueError(
                "residual ranking_metric has no unambiguous importance direction."
            )
        if operations and ranking_metric is None and not rankable:
            raise ValueError(
                "residual causal metrics must include at least one rank-compatible "
                "distribution-shift metric."
            )
        nonnegative_finite_number(
            config,
            "residual_atol",
            2e-3,
            context="residual decomposition",
        )
        nonnegative_finite_number(
            config,
            "residual_rtol",
            2e-3,
            context="residual decomposition",
        )

    def open(
        self,
        *,
        backbone: Any,
        runtime: Any,
        config: dict[str, Any],
        phase: str,
        output_dir: str,
    ) -> "_ResidualStreamSession":
        self.validate_config(config)
        return _ResidualStreamSession(
            plugin=self,
            backbone=backbone,
            runtime=runtime,
            config=config,
            phase=phase,
            output_dir=output_dir,
        )

@dataclass
class _ResidualStreamSession:
    plugin: ResidualStreamDecomposition
    backbone: Any
    runtime: Any
    config: dict[str, Any]
    phase: str
    output_dir: str

    def __post_init__(self) -> None:
        self.binding = resolve_analysis_binding(self.backbone, self.runtime.model)
        self.selected_layers = select_decoder_layers(
            self.binding,
            self.config.get("layers", "all"),
            context="Residual observation",
        )
        self.causal_layers = select_decoder_layers(
            self.binding,
            self.config.get("causal_layers", self.config.get("layers", "all")),
            context="Residual causal",
        )
        self.causal_operations = tuple(dict.fromkeys(_causal_operations(self.config)))
        configured_metrics = tuple(
            dict.fromkeys(
                string_sequence(
                    self.config,
                    "metrics",
                    (*RESIDUAL_METRICS, *DISTRIBUTION_METRICS),
                    context="residual decomposition",
                )
            )
        )
        self.residual_metrics = tuple(
            metric for metric in RESIDUAL_METRICS if metric in configured_metrics
        )
        self.distribution_metrics = tuple(
            metric for metric in DISTRIBUTION_METRICS if metric in configured_metrics
        )
        self.token_groups = tuple(
            dict.fromkeys(
                string_sequence(
                    self.config,
                    "token_groups",
                    ("all", "system", "text", "vision"),
                    context="residual decomposition",
                )
            )
        )
        self.ranking_metric = (
            select_ranking_metric(
                self.distribution_metrics,
                self.config.get("ranking_metric"),
                IMPORTANCE_RANKING_DIRECTIONS,
            )
            if self.causal_operations and self.distribution_metrics
            else None
        )
        if self.ranking_metric is not None and self.ranking_metric not in self.distribution_metrics:
            raise BackendIntegrationError(
                "Residual ranking_metric must be a configured distribution metric."
            )
        self.ranking_higher_is_more_important = (
            IMPORTANCE_RANKING_DIRECTIONS[self.ranking_metric]
            if self.ranking_metric is not None
            else None
        )
        self.collect_residual_stats = strict_boolean(
            self.config,
            "collect_residual_stats",
            True,
            context="residual decomposition",
        )
        self.records = StreamingAnalysisRecords(
            output_dir=self.output_dir,
            prefix=self.plugin.name,
            save_raw=bool(self.config.get("save_raw", False)),
            accumulator_factory=lambda: ResidualSampleAccumulator(
                residual_metrics=self.residual_metrics,
                distribution_metrics=self.distribution_metrics,
                token_groups=self.token_groups,
            ),
            aggregate_by_task=True,
        )
        self._finished = False
        self._closed = False

    def process(self, sample: Mapping[str, Any]) -> None:
        if self._finished or self._closed:
            raise BackendIntegrationError(
                "Cannot process a sample after residual-decomposition completion."
            )
        try:
            report = self._process_sample(sample)
        except (BackendIntegrationError, ValueError) as exc:
            self.records.add_skipped(
                {
                    "sample_id": sample.get("sample_id"),
                    "task_name": sample.get("task_name"),
                    "reason": f"{type(exc).__name__}: {exc}",
                }
            )
            return
        self.records.add_success(report)

    def _process_sample(self, sample: Mapping[str, Any]) -> dict[str, Any]:
        state = prepare_analysis_forward_state(
            backbone=self.backbone,
            runtime=self.runtime,
            sample=sample,
            probe_name="Residual stream decomposition",
        )
        try:
            import torch
        except ImportError as exc:  # pragma: no cover
            raise BackendIntegrationError(
                "Residual stream decomposition requires PyTorch."
            ) from exc

        capture = ResidualNodeCapture(self.selected_layers)
        baseline_output: Any | None = None
        baseline_logits: Any | None = None
        try:
            with torch.inference_mode():
                with HookSession() as hooks:
                    capture.install(hooks)
                    baseline_output = self.backbone.runner.forward_train(
                        self.runtime,
                        state,
                    )
                baseline_logits = logits_tensor(baseline_output).detach()
                nodes, residual_evidence = capture.finish(
                    atol=float(self.config.get("residual_atol", 2e-3)),
                    rtol=float(self.config.get("residual_rtol", 2e-3)),
                )
                sequence_length = int(baseline_logits.shape[1])
                layout = build_token_group_layout(
                    state,
                    sequence_length=sequence_length,
                    backbone_name=self.binding.backbone_name,
                    runtime=self.runtime,
                )
                residual_stats = (
                    _residual_stats(
                        nodes,
                        layout=layout,
                        token_groups=self.token_groups,
                        metrics=self.residual_metrics,
                    )
                    if self.collect_residual_stats
                    else {}
                )
                causal: dict[str, Any] = {}
                for layer in self.causal_layers:
                    causal[str(layer.index)] = {}
                    for operation in self.causal_operations:
                        changed_output: Any | None = None
                        changed_logits: Any | None = None
                        try:
                            sublayer = (
                                layer.attention if operation == "attn" else layer.ffn
                            )
                            with HookSession() as hooks:
                                install_sublayer_update_skip(
                                    hooks,
                                    sublayer=sublayer,
                                    layout=layout,
                                    target_group="all",
                                )
                                changed_output = self.backbone.runner.forward_train(
                                    self.runtime,
                                    state,
                                )
                            changed_logits = logits_tensor(changed_output).detach()
                            report = distribution_shift_report(
                                baseline_logits,
                                changed_logits,
                                layout=layout,
                                token_groups=tuple(
                                    group for group in self.token_groups if group != "all"
                                ),
                                metrics=self.distribution_metrics,
                            )
                            causal[str(layer.index)][f"remove_{operation}_update"] = {
                                "status": "finite",
                                "intervention": {
                                    "operation": operation,
                                    "scope": "all_nonpadding_tokens",
                                    "semantic_target": (
                                        "attention_residual_update"
                                        if operation == "attn"
                                        else "ffn_residual_update"
                                    ),
                                },
                                **report,
                            }
                        finally:
                            del changed_logits
                            del changed_output
            return {
                "sample_id": sample["sample_id"],
                "task_name": sample["task_name"],
                "token_counts": layout.counts(),
                "token_layout_evidence": dict(layout.evidence),
                "residual_contract": residual_evidence,
                "residual_stats": residual_stats,
                "causal": causal,
            }
        finally:
            capture.clear()
            del baseline_logits
            del baseline_output

    def finish(self) -> AnalysisRecord:
        if self._finished:
            raise BackendIntegrationError("Residual session was already finished.")
        self._finished = True
        if self.records.successful_count == 0:
            reasons = "; ".join(
                str(item.get("reason"))
                for item in self.records.skipped_examples[:3]
            )
            raise BackendIntegrationError(
                "Residual stream decomposition produced no successful samples. "
                + reasons
            )
        aggregate = self.records.overall_result()
        task_aggregates = self.records.task_results()
        rank_aggregation = (
            aggregate_task_percentiles(
                {
                    task: _causal_scores(value, self.ranking_metric)
                    for task, value in task_aggregates.items()
                    if int(value.get("sample_count", 0)) > 0
                },
                higher_is_more_important=bool(
                    self.ranking_higher_is_more_important
                ),
            )
            if self.ranking_metric is not None
            else None
        )
        aggregate.update(
            {
                "task_aggregates": task_aggregates,
                "task_rank_aggregation": rank_aggregation,
                "ranking_metric": self.ranking_metric,
                "ranking_higher_is_more_important": (
                    self.ranking_higher_is_more_important
                ),
                "residual_metrics": list(self.residual_metrics),
                "distribution_metrics": list(self.distribution_metrics),
                "causal_operations": list(self.causal_operations),
                "observed_layers": [layer.index for layer in self.selected_layers],
                "causal_layers": [layer.index for layer in self.causal_layers],
                "token_groups": list(self.token_groups),
                "binding_evidence": dict(self.binding.evidence),
                **self.records.skipped_evidence(),
                "streaming_aggregation": self.records.execution_evidence(),
            }
        )
        return write_streaming_analysis_result(
            output_dir=self.output_dir,
            probe_name=str(self.config.get("name", self.plugin.name)),
            method=str(self.config.get("method", self.plugin.name)),
            method_version=self.plugin.version,
            phase=self.phase,
            backbone_name=self.binding.backbone_name,
            checkpoint=str(self.runtime.model_path_or_id or ""),
            checkpoint_revision=self.runtime.revision,
            capability_status=self.config.get("_capability_status"),
            filename_stem="residual_stream_decomposition",
            records=self.records,
            aggregate=aggregate,
            summary={
                "observed_layer_count": len(self.selected_layers),
                "causal_layer_count": len(self.causal_layers),
                "causal_operations": list(self.causal_operations),
                "finite_aggregate_values": finite_metric_count(
                    aggregate["residual_stats"],
                    self.residual_metrics,
                )
                + finite_metric_count(
                    aggregate["causal"],
                    self.distribution_metrics,
                ),
            },
        )

    def close(self) -> None:
        self._closed = True
        self.records.close()


def _residual_stats(
    nodes: Mapping[int, Mapping[str, Any]],
    *,
    layout: Any,
    token_groups: Sequence[str],
    metrics: Sequence[str],
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for layer, values in nodes.items():
        operations = {
            "attn": summarize_residual_updates(
                values["before_attn"],
                values["after_attn"],
                layout=layout,
                token_groups=token_groups,
            ),
            "ffn": summarize_residual_updates(
                values["after_attn"],
                values["after_ffn"],
                layout=layout,
                token_groups=token_groups,
            ),
        }
        result[str(layer)] = {
            operation: {
                group: {
                    **{metric: report[metric] for metric in metrics},
                    "token_count": report["token_count"],
                    "status": report["status"],
                }
                for group, report in groups.items()
            }
            for operation, groups in operations.items()
        }
    return result


def _causal_scores(aggregate: Mapping[str, Any], metric: str | None) -> dict[str, float]:
    if metric is None:
        return {}
    result: dict[str, float] = {}
    for layer, operations in aggregate.get("causal", {}).items():
        for operation, report in operations.items():
            value = report.get("decision", {}).get(metric)
            if value is not None and math.isfinite(float(value)):
                result[f"layer:{layer}/{operation}"] = float(value)
    return result


def _causal_operations(config: Mapping[str, Any]) -> tuple[str, ...]:
    if "causal_operations" in config:
        field = "causal_operations"
    else:
        field = "operations"
    return string_sequence(
        config,
        field,
        ("attn", "ffn"),
        context="residual decomposition",
        allow_empty=True,
    )


def _validate_layers(value: Any, *, field: str) -> None:
    validate_layer_selection(
        value,
        context=f"residual decomposition {field}",
    )


__all__ = ["ResidualStreamDecomposition"]
