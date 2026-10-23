"""Direct multimodal self-attention routing without cross-layer rollout."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from mllm_comp.analysis.artifacts import write_streaming_analysis_result
from mllm_comp.analysis.bindings import (
    layer_batches,
    require_eager_attention,
    resolve_analysis_binding,
    select_decoder_layers,
)
from mllm_comp.analysis.common.attention_flow import (
    ATTENTION_FLOW_METRICS,
    ATTENTION_GROUPS,
    DEFAULT_ATTENTION_GROUPS,
    AttentionFlowSampleAccumulator,
    attention_flow_metrics,
    compute_attention_flow_layer,
)
from mllm_comp.analysis.common.attention_weights import AttentionWeightCapture
from mllm_comp.analysis.common.config import (
    nonnegative_finite_number,
    positive_integer,
    string_sequence,
    validate_common_probe_config,
    validate_layer_selection,
)
from mllm_comp.analysis.common.sample import prepare_analysis_forward_state
from mllm_comp.analysis.common.rank import finite_metric_count
from mllm_comp.analysis.common.streaming import StreamingAnalysisRecords
from mllm_comp.analysis.common.token_mask import build_token_group_layout
from mllm_comp.analysis.hooks import HookSession
from mllm_comp.core.errors import BackendIntegrationError
from mllm_comp.core.manifest import AnalysisRecord


class AttentionFlow:
    """Measure direct query-to-key modality mass for every selected head."""

    name = "attention_flow"
    version = "1.0.0"
    supported_metrics = set(ATTENTION_FLOW_METRICS)
    supported_token_groups = set(ATTENTION_GROUPS)

    def validate_config(self, config: dict[str, Any]) -> None:
        validate_common_probe_config(config, context="attention flow")
        groups = set(
            string_sequence(
                config,
                "token_groups",
                DEFAULT_ATTENTION_GROUPS,
                context="attention flow",
            )
        )
        if not groups or not groups.issubset(self.supported_token_groups):
            unknown = sorted(groups - self.supported_token_groups)
            raise ValueError(
                f"Unsupported attention-flow token groups: {unknown or sorted(groups)}"
            )
        default_metrics = attention_flow_metrics(
            group for group in ATTENTION_GROUPS if group in groups
        )
        metrics = set(
            string_sequence(
                config,
                "metrics",
                default_metrics,
                context="attention flow",
            )
        )
        if not metrics or not metrics.issubset(self.supported_metrics):
            unknown = sorted(metrics - self.supported_metrics)
            raise ValueError(
                f"Unsupported attention-flow metrics: {unknown or sorted(metrics)}"
            )
        missing_groups = {
            metric: sorted(_metric_groups(metric) - groups)
            for metric in metrics
            if _metric_groups(metric) - groups
        }
        if missing_groups:
            raise ValueError(
                "attention-flow metrics require token groups that were not selected: "
                f"{missing_groups}."
            )
        validate_layer_selection(
            config.get("layers", "all"),
            context="attention flow layers",
        )
        nonnegative_finite_number(
            config,
            "mass_conservation_tolerance",
            2e-3,
            context="attention flow",
        )
        positive_integer(
            config,
            "layer_batch_size",
            1,
            context="attention flow",
        )

    def open(
        self,
        *,
        backbone: Any,
        runtime: Any,
        config: dict[str, Any],
        phase: str,
        output_dir: str,
    ) -> "_AttentionFlowSession":
        self.validate_config(config)
        return _AttentionFlowSession(
            plugin=self,
            backbone=backbone,
            runtime=runtime,
            config=config,
            phase=phase,
            output_dir=output_dir,
        )

@dataclass
class _AttentionFlowSession:
    plugin: AttentionFlow
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
            context="Attention flow",
        )
        configured_metrics = set(
            string_sequence(
                self.config,
                "metrics",
                attention_flow_metrics(
                    string_sequence(
                        self.config,
                        "token_groups",
                        DEFAULT_ATTENTION_GROUPS,
                        context="attention flow",
                    )
                ),
                context="attention flow",
            )
        )
        self.metrics = tuple(
            metric for metric in ATTENTION_FLOW_METRICS if metric in configured_metrics
        )
        configured_groups = set(
            string_sequence(
                self.config,
                "token_groups",
                DEFAULT_ATTENTION_GROUPS,
                context="attention flow",
            )
        )
        self.token_groups = tuple(
            group for group in ATTENTION_GROUPS if group in configured_groups
        )
        self.attention_backends = require_eager_attention(
            self.selected_layers,
            context="Attention flow",
        )
        self.layer_batch_size = int(self.config.get("layer_batch_size", 1))
        self.capture_execution = {
            "contract": "bounded_selected_layer_batches_v1",
            "layer_batch_size": self.layer_batch_size,
            "capture_pass_count": len(
                layer_batches(self.selected_layers, self.layer_batch_size)
            ),
            "selected_layer_count": len(self.selected_layers),
        }
        self.records = StreamingAnalysisRecords(
            output_dir=self.output_dir,
            prefix=self.plugin.name,
            save_raw=bool(self.config.get("save_raw", False)),
            accumulator_factory=lambda: AttentionFlowSampleAccumulator(
                metrics=self.metrics
            ),
            aggregate_by_task=True,
        )
        self._finished = False
        self._closed = False

    def process(self, sample: Mapping[str, Any]) -> None:
        if self._finished or self._closed:
            raise BackendIntegrationError(
                "Cannot process a sample after attention-flow completion."
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
        self._record_success(report)

    def _record_success(self, report: Mapping[str, Any]) -> None:
        if report.get("capture_execution") != self.capture_execution:
            raise BackendIntegrationError(
                "Attention-flow sample capture evidence does not match the "
                "configured bounded-layer execution contract."
            )
        self.records.add_success(report)

    def _process_sample(self, sample: Mapping[str, Any]) -> dict[str, Any]:
        state = prepare_analysis_forward_state(
            backbone=self.backbone,
            runtime=self.runtime,
            sample=sample,
            probe_name="Attention flow",
            output_attentions=False,
        )
        try:
            import torch
        except ImportError as exc:  # pragma: no cover
            raise BackendIntegrationError("Attention flow requires PyTorch.") from exc
        layout: Any | None = None
        decision_position: int | None = None
        layer_results: dict[str, Any] = {}
        capture_pass_count = 0
        for layer_batch in layer_batches(
            self.selected_layers,
            self.layer_batch_size,
        ):
            attentions: Mapping[int, Any] | None = None
            first: Any | None = None
            attention: Any | None = None
            capture = AttentionWeightCapture(layer_batch)
            try:
                with HookSession() as hooks:
                    capture.install(hooks)
                    with torch.inference_mode():
                        self.backbone.runner.forward_train(self.runtime, state)
                    attentions = capture.finish()
                first = attentions[layer_batch[0].index]
                if first is None or len(first.shape) != 4:
                    raise BackendIntegrationError(
                        "Attention flow received no full attention matrix. Use "
                        "attn_implementation=eager on the pinned checkpoint."
                    )
                batch_layout = build_token_group_layout(
                    state,
                    sequence_length=int(first.shape[-1]),
                    backbone_name=self.binding.backbone_name,
                    runtime=self.runtime,
                )
                if layout is None:
                    layout = batch_layout
                    valid_positions = tuple(
                        index
                        for index, label in enumerate(layout.labels_by_sample[0])
                        if label != "padding"
                    )
                    if not valid_positions:
                        raise BackendIntegrationError(
                            "Attention flow found no decision token."
                        )
                    decision_position = valid_positions[-1]
                elif batch_layout.labels_by_sample != layout.labels_by_sample:
                    raise BackendIntegrationError(
                        "Bounded attention-flow layer batches produced different "
                        "token layouts."
                    )
                if decision_position is None:
                    raise BackendIntegrationError(
                        "Attention flow did not resolve a decision token."
                    )
                for layer in layer_batch:
                    attention = attentions[layer.index]
                    if attention is None:
                        raise BackendIntegrationError(
                            f"Attention flow layer {layer.index} returned no weights."
                        )
                    if (
                        int(attention.shape[1])
                        != self.binding.attention_geometry.num_query_heads
                    ):
                        raise BackendIntegrationError(
                            f"Attention flow layer {layer.index} head count "
                            f"{int(attention.shape[1])} does not match binding geometry "
                            f"{self.binding.attention_geometry.num_query_heads}."
                        )
                    layer_results[str(layer.index)] = compute_attention_flow_layer(
                        attention.detach(),
                        layout=layout,
                        decision_position=decision_position,
                        groups=self.token_groups,
                        mass_tolerance=float(
                            self.config.get("mass_conservation_tolerance", 2e-3)
                        ),
                    )
                capture_pass_count += 1
            finally:
                attention = first = attentions = None
                capture.clear()
        if layout is None or decision_position is None:
            raise BackendIntegrationError(
                "Attention flow produced no bounded capture batch."
            )
        return {
            "sample_id": sample["sample_id"],
            "task_name": sample["task_name"],
            "decision_position": decision_position,
            "token_counts": layout.counts(),
            "token_layout_evidence": dict(layout.evidence),
            "routing_semantics": "direct_attention_mass_no_rollout_v1",
            "capture_execution": {
                **self.capture_execution,
                "capture_pass_count": capture_pass_count,
            },
            "layers": layer_results,
        }

    def finish(self) -> AnalysisRecord:
        if self._finished:
            raise BackendIntegrationError("Attention-flow session was already finished.")
        self._finished = True
        if self.records.successful_count == 0:
            reasons = "; ".join(
                str(item.get("reason"))
                for item in self.records.skipped_examples[:3]
            )
            raise BackendIntegrationError(
                "Attention flow produced no successful samples. " + reasons
            )
        aggregate = self.records.overall_result()
        aggregate.update(
            {
                "task_aggregates": self.records.task_results(),
                "metrics": list(self.metrics),
                "token_groups": list(self.token_groups),
                "configured_layers": [layer.index for layer in self.selected_layers],
                "routing_semantics": "direct_attention_mass_no_rollout_v1",
                "attention_backends": self.attention_backends,
                "capture_execution": {
                    **self.capture_execution,
                    "successful_sample_count": self.records.successful_count,
                },
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
            filename_stem="attention_flow",
            records=self.records,
            aggregate=aggregate,
            summary={
                "layer_count": len(self.selected_layers),
                "head_count": self.binding.attention_geometry.num_query_heads,
                "finite_aggregate_values": finite_metric_count(
                    aggregate["layers"],
                    self.metrics,
                ),
                "routing_semantics": "direct_attention_mass_no_rollout_v1",
                "layer_batch_size": self.layer_batch_size,
                "capture_pass_count_per_sample": self.capture_execution[
                    "capture_pass_count"
                ],
            },
        )

    def close(self) -> None:
        self._closed = True
        self.records.close()


def _metric_groups(metric: str) -> set[str]:
    if metric.startswith("decision_to_"):
        remainder = metric[len("decision_to_") :]
        key = remainder.split("_", 1)[0]
        return {key}
    query, separator, remainder = metric.partition("_to_")
    if separator:
        target = remainder.split("_", 1)[0]
        if target:
            return {query, target}
    return set()


__all__ = ["AttentionFlow"]
