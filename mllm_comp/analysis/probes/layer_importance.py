"""Modality-aware Layer/Attention/FFN intervention importance."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping, Sequence

from mllm_comp.analysis.artifacts import write_streaming_analysis_result
from mllm_comp.analysis.bindings import (
    DecoderLayerBinding,
    resolve_analysis_binding,
    select_decoder_layers,
)
from mllm_comp.analysis.common.distribution import (
    DISTRIBUTION_METRICS,
    IMPORTANCE_RANKING_DIRECTIONS,
    LayerImportanceSampleAccumulator,
    distribution_shift_report,
)
from mllm_comp.analysis.common.config import (
    string_sequence,
    validate_common_probe_config,
    validate_layer_selection,
)
from mllm_comp.analysis.common.interventions import (
    install_attention_edge_block,
    install_ffn_update_skip,
    install_layer_update_skip,
)
from mllm_comp.analysis.common.rank import (
    aggregate_task_percentiles,
    finite_metric_count,
    select_ranking_metric,
)
from mllm_comp.analysis.common.sample import prepare_analysis_forward_state
from mllm_comp.analysis.common.streaming import StreamingAnalysisRecords
from mllm_comp.analysis.common.tensors import logits_tensor
from mllm_comp.analysis.common.token_mask import TokenGroupLayout, build_token_group_layout
from mllm_comp.analysis.hooks import HookSession
from mllm_comp.core.errors import BackendIntegrationError
from mllm_comp.core.manifest import AnalysisRecord


_TARGET_GROUPS = {
    "system",
    "text",
    "vision",
    "image",
    "video",
    "audio",
    "all",
}
_EDGE_GROUPS = {"s": "system", "t": "text", "v": "vision", "a": "audio"}
_ATTENTION_EDGES = {
    f"{target}2{source}" for target in _EDGE_GROUPS for source in _EDGE_GROUPS
}
_REPORT_TOKEN_GROUPS = ("system", "text", "vision")
_MASK_CAPABLE_ATTENTION_BACKENDS = {"eager", "sdpa"}


@dataclass(frozen=True)
class InterventionSpec:
    operation: str
    result_key: str
    target_group: str | None = None
    source_group: str | None = None
    attention_edge: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "operation": self.operation,
            "result_key": self.result_key,
            "target_group": self.target_group,
            "source_group": self.source_group,
            "attention_edge": self.attention_edge,
        }


class LayerImportance:
    """Measure output-distribution shifts under semantic decoder interventions."""

    name = "layer_importance"
    version = "1.0.0"
    supported_operations = {"layer", "layer_strict", "attn", "ffn"}
    supported_token_types = _TARGET_GROUPS
    supported_attention_edges = _ATTENTION_EDGES
    supported_metrics = set(DISTRIBUTION_METRICS)

    def validate_config(self, config: dict[str, Any]) -> None:
        validate_common_probe_config(config, context="layer importance")
        operations = set(
            string_sequence(
                config,
                "operations",
                ("layer", "attn", "ffn"),
                context="layer importance",
            )
        )
        if not operations or not operations.issubset(self.supported_operations):
            unknown = sorted(operations - self.supported_operations)
            raise ValueError(
                f"Unsupported layer importance operations: {unknown or sorted(operations)}"
            )

        token_types = set(
            string_sequence(
                config,
                "target_token_types",
                ("system", "text", "vision"),
                context="layer importance",
            )
        )
        if not token_types or not token_types.issubset(self.supported_token_types):
            unknown = sorted(token_types - self.supported_token_types)
            raise ValueError(
                f"Unsupported layer importance token types: "
                f"{unknown or sorted(token_types)}"
            )

        attn_edges = set(
            string_sequence(
                config,
                "attn_edges",
                ("v2v", "v2t", "t2v", "t2t"),
                context="layer importance",
            )
        )
        if "attn" in operations and (
            not attn_edges or not attn_edges.issubset(self.supported_attention_edges)
        ):
            unknown = sorted(attn_edges - self.supported_attention_edges)
            raise ValueError(
                f"Unsupported layer importance attention edges: "
                f"{unknown or sorted(attn_edges)}"
            )

        metrics = set(
            string_sequence(
                config,
                "metrics",
                DISTRIBUTION_METRICS,
                context="layer importance",
            )
        )
        if not metrics or not metrics.issubset(self.supported_metrics):
            unknown = sorted(metrics - self.supported_metrics)
            raise ValueError(
                f"Unsupported layer importance metrics: {unknown or sorted(metrics)}"
            )
        ranking_metric = config.get("ranking_metric")
        if ranking_metric is not None and (
            not isinstance(ranking_metric, str) or ranking_metric not in metrics
        ):
            raise ValueError(
                "layer importance ranking_metric must also be present in metrics."
            )
        rankable = metrics & set(IMPORTANCE_RANKING_DIRECTIONS)
        if ranking_metric is not None and ranking_metric not in rankable:
            raise ValueError(
                "layer importance ranking_metric has no unambiguous importance "
                "direction. Use KL, JS, changed_prob_on_base_top1, or top1_changed."
            )
        if ranking_metric is None and not rankable:
            raise ValueError(
                "layer importance metrics must include at least one rank-compatible "
                "distribution-shift metric."
            )

        validate_layer_selection(
            config.get("layers", "all"),
            context="layer importance layers",
        )

    def open(
        self,
        *,
        backbone: Any,
        runtime: Any,
        config: dict[str, Any],
        phase: str,
        output_dir: str,
    ) -> "_LayerImportanceSession":
        self.validate_config(config)
        return _LayerImportanceSession(
            plugin=self,
            backbone=backbone,
            runtime=runtime,
            config=config,
            phase=phase,
            output_dir=output_dir,
        )

@dataclass
class _LayerImportanceSession:
    plugin: LayerImportance
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
            context="Layer importance",
        )
        self.operations = tuple(
            dict.fromkeys(
                string_sequence(
                    self.config,
                    "operations",
                    ("layer", "attn", "ffn"),
                    context="layer importance",
                )
            )
        )
        self.target_groups = tuple(
            dict.fromkeys(
                string_sequence(
                    self.config,
                    "target_token_types",
                    ("system", "text", "vision"),
                    context="layer importance",
                )
            )
        )
        self.attention_edges = tuple(
            dict.fromkeys(
                string_sequence(
                    self.config,
                    "attn_edges",
                    ("v2v", "v2t", "t2v", "t2t"),
                    context="layer importance",
                )
            )
        )
        self.metrics = tuple(
            dict.fromkeys(
                string_sequence(
                    self.config,
                    "metrics",
                    DISTRIBUTION_METRICS,
                    context="layer importance",
                )
            )
        )
        self.ranking_metric = select_ranking_metric(
            self.metrics,
            self.config.get("ranking_metric"),
            IMPORTANCE_RANKING_DIRECTIONS,
        )
        self.ranking_higher_is_more_important = IMPORTANCE_RANKING_DIRECTIONS[
            self.ranking_metric
        ]
        self.report_token_groups = _report_token_groups(
            self.target_groups,
            self.attention_edges,
        )
        self.specs = build_intervention_specs(
            operations=self.operations,
            target_groups=self.target_groups,
            attention_edges=self.attention_edges,
        )
        self.attention_backends = _validate_attention_backends(
            self.selected_layers,
            self.specs,
        )
        self.records = StreamingAnalysisRecords(
            output_dir=self.output_dir,
            prefix=self.plugin.name,
            save_raw=bool(self.config.get("save_raw", False)),
            accumulator_factory=lambda: LayerImportanceSampleAccumulator(
                metrics=self.metrics,
                token_groups=self.report_token_groups,
            ),
            aggregate_by_task=True,
        )
        self._finished = False
        self._closed = False

    def process(self, sample: Mapping[str, Any]) -> None:
        if self._finished or self._closed:
            raise BackendIntegrationError(
                "Cannot process a sample after layer-importance session completion."
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
            probe_name="Layer importance",
        )
        try:
            import torch
        except ImportError as exc:  # pragma: no cover - real runtime requires torch
            raise BackendIntegrationError("Layer importance requires PyTorch.") from exc

        layers: dict[str, dict[str, Any]] = {}
        baseline_output: Any | None = None
        baseline_logits: Any | None = None
        finite_interventions = 0
        try:
            with torch.inference_mode():
                baseline_output = self.backbone.runner.forward_train(self.runtime, state)
                baseline_logits = logits_tensor(baseline_output).detach()
                sequence_length = int(baseline_logits.shape[1])
                layout = build_token_group_layout(
                    state,
                    sequence_length=sequence_length,
                    backbone_name=self.binding.backbone_name,
                    runtime=self.runtime,
                )
                for layer in self.selected_layers:
                    layer_results: dict[str, Any] = {}
                    for spec in self.specs:
                        unavailable_reason = _unavailable_reason(spec, layout)
                        if unavailable_reason is not None:
                            layer_results[spec.result_key] = {
                                "status": "unavailable",
                                "reason": unavailable_reason,
                                "intervention": spec.as_dict(),
                            }
                            continue
                        evidence: dict[str, Any] = {}
                        changed_output: Any | None = None
                        changed_logits: Any | None = None
                        try:
                            with HookSession() as hooks:
                                _install_intervention(
                                    hooks,
                                    layer=layer,
                                    layout=layout,
                                    spec=spec,
                                    evidence=evidence,
                                )
                                changed_output = self.backbone.runner.forward_train(
                                    self.runtime,
                                    state,
                                )
                            if (
                                spec.operation in {"attn", "layer_strict"}
                                and evidence.get("newly_blocked_edge_count") == 0
                            ):
                                layer_results[spec.result_key] = {
                                    "status": "unavailable",
                                    "reason": "no_safely_blockable_causally_visible_edge",
                                    "intervention": {
                                        **spec.as_dict(),
                                        **evidence,
                                    },
                                }
                                continue
                            changed_logits = logits_tensor(changed_output).detach()
                            report = distribution_shift_report(
                                baseline_logits,
                                changed_logits,
                                layout=layout,
                                token_groups=self.report_token_groups,
                                metrics=self.metrics,
                            )
                            layer_results[spec.result_key] = {
                                "status": "finite",
                                "intervention": {
                                    **spec.as_dict(),
                                    **evidence,
                                },
                                **report,
                            }
                            finite_interventions += 1
                        except BackendIntegrationError as exc:
                            layer_results[spec.result_key] = {
                                "status": "unavailable",
                                "reason": f"BackendIntegrationError: {exc}",
                                "intervention": {
                                    **spec.as_dict(),
                                    **evidence,
                                },
                            }
                        finally:
                            del changed_logits
                            del changed_output
                    layers[str(layer.index)] = layer_results
        finally:
            del baseline_logits
            del baseline_output
        if finite_interventions == 0:
            raise BackendIntegrationError(
                "Layer importance found no available intervention for this sample."
            )
        return {
            "sample_id": sample["sample_id"],
            "task_name": sample["task_name"],
            "token_counts": layout.counts(),
            "token_layout_evidence": dict(layout.evidence),
            "layers": layers,
        }

    def finish(self) -> AnalysisRecord:
        if self._finished:
            raise BackendIntegrationError("Layer-importance session was already finished.")
        self._finished = True
        if self.records.successful_count == 0:
            reasons = "; ".join(
                str(item.get("reason"))
                for item in self.records.skipped_examples[:3]
            )
            raise BackendIntegrationError(
                "Layer importance produced no successful samples. " + reasons
            )

        aggregate = self.records.overall_result()
        task_aggregates = self.records.task_results()
        task_scores = {
            task: _component_scores(task_aggregate, self.ranking_metric)
            for task, task_aggregate in task_aggregates.items()
            if int(task_aggregate.get("sample_count", 0)) > 0
        }
        aggregate.update(
            {
                "task_aggregates": task_aggregates,
                "task_rank_aggregation": aggregate_task_percentiles(
                    task_scores,
                    higher_is_more_important=(
                        self.ranking_higher_is_more_important
                    ),
                ),
                "ranking_metric": self.ranking_metric,
                "ranking_higher_is_more_important": (
                    self.ranking_higher_is_more_important
                ),
                "metrics": list(self.metrics),
                "operations": list(self.operations),
                "target_token_types": list(self.target_groups),
                "attention_edges": list(self.attention_edges),
                "interventions": [spec.as_dict() for spec in self.specs],
                "configured_layers": [layer.index for layer in self.selected_layers],
                "binding_evidence": dict(self.binding.evidence),
                "attention_backends": self.attention_backends,
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
            filename_stem="layer_importance",
            records=self.records,
            aggregate=aggregate,
            aggregate_metadata={"intervention_count": len(self.specs)},
            summary={
                "layer_count": len(self.selected_layers),
                "intervention_count": len(self.specs),
                "finite_aggregate_values": finite_metric_count(
                    aggregate,
                    self.metrics,
                ),
                "ranking_metric": self.ranking_metric,
            },
        )

    def close(self) -> None:
        self._closed = True
        self.records.close()


def build_intervention_specs(
    *,
    operations: Sequence[str],
    target_groups: Sequence[str],
    attention_edges: Sequence[str],
) -> tuple[InterventionSpec, ...]:
    specs: list[InterventionSpec] = []
    for operation in operations:
        if operation == "attn":
            for edge in attention_edges:
                target_alias, source_alias = edge.split("2", maxsplit=1)
                specs.append(
                    InterventionSpec(
                        operation="attn",
                        result_key=f"attn:{edge}",
                        target_group=_EDGE_GROUPS[target_alias],
                        source_group=_EDGE_GROUPS[source_alias],
                        attention_edge=edge,
                    )
                )
            continue
        for target_group in target_groups:
            specs.append(
                InterventionSpec(
                    operation=operation,
                    result_key=f"{operation}:{target_group}",
                    target_group=target_group,
                )
            )
    return tuple(specs)


def _report_token_groups(
    target_groups: Sequence[str],
    attention_edges: Sequence[str],
) -> tuple[str, ...]:
    """Keep the visual report schema stable and add only selected new groups."""

    groups = list(_REPORT_TOKEN_GROUPS)
    for group in target_groups:
        if group not in {"all", "image", "video"} and group not in groups:
            groups.append(group)
    for edge in attention_edges:
        target_alias, source_alias = edge.split("2", maxsplit=1)
        for alias in (target_alias, source_alias):
            group = _EDGE_GROUPS[alias]
            if group not in groups:
                groups.append(group)
    return tuple(groups)


def _install_intervention(
    hooks: HookSession,
    *,
    layer: DecoderLayerBinding,
    layout: TokenGroupLayout,
    spec: InterventionSpec,
    evidence: dict[str, Any],
) -> None:
    if spec.operation == "layer":
        assert spec.target_group is not None
        install_layer_update_skip(
            hooks,
            layer=layer.layer,
            layout=layout,
            target_group=spec.target_group,
        )
        return
    if spec.operation == "ffn":
        assert spec.target_group is not None
        install_ffn_update_skip(
            hooks,
            ffn=layer.ffn,
            layout=layout,
            target_group=spec.target_group,
        )
        return
    if spec.operation == "attn":
        assert spec.target_group is not None and spec.source_group is not None
        install_attention_edge_block(
            hooks,
            attention=layer.attention,
            layout=layout,
            source_group=spec.source_group,
            target_group=spec.target_group,
            evidence=evidence,
        )
        return
    if spec.operation == "layer_strict":
        assert spec.target_group is not None
        if spec.target_group != "all":
            install_attention_edge_block(
                hooks,
                attention=layer.attention,
                layout=layout,
                source_group=spec.target_group,
                target_group=None,
                non_source_queries=True,
                evidence=evidence,
            )
        install_layer_update_skip(
            hooks,
            layer=layer.layer,
            layout=layout,
            target_group=spec.target_group,
        )
        return
    raise BackendIntegrationError(
        f"Unsupported layer importance operation {spec.operation!r}."
    )


def _unavailable_reason(
    spec: InterventionSpec,
    layout: TokenGroupLayout,
) -> str | None:
    if spec.target_group is not None and not _group_positions(layout, spec.target_group):
        return f"target token group {spec.target_group!r} is absent"
    if spec.source_group is not None and not _group_positions(layout, spec.source_group):
        return f"source token group {spec.source_group!r} is absent"
    return None


def _group_positions(layout: TokenGroupLayout, group: str) -> tuple[int, ...]:
    if group == "all":
        return tuple(
            index
            for index, label in enumerate(layout.labels_by_sample[0])
            if label != "padding"
        )
    return layout.positions(0, group)


def _validate_attention_backends(
    layers: Sequence[DecoderLayerBinding],
    specs: Sequence[InterventionSpec],
) -> dict[str, str]:
    needs_mask_intervention = any(
        spec.operation == "attn"
        or (spec.operation == "layer_strict" and spec.target_group != "all")
        for spec in specs
    )
    result: dict[str, str] = {}
    for layer in layers:
        config = getattr(layer.attention, "config", None)
        backend = getattr(config, "_attn_implementation", None)
        normalized = str(backend or "unknown")
        result[str(layer.index)] = normalized
        if needs_mask_intervention and normalized not in _MASK_CAPABLE_ATTENTION_BACKENDS:
            raise BackendIntegrationError(
                "Attention edge/strict interventions require eager or SDPA attention; "
                f"layer {layer.index} reports {normalized!r}. Configure the pinned "
                "checkpoint with attn_implementation=eager or sdpa."
            )
    return result


def _component_scores(aggregate: Mapping[str, Any], metric: str) -> dict[str, float]:
    result: dict[str, float] = {}
    for layer, experiments in aggregate.get("layers", {}).items():
        for experiment, report in experiments.items():
            value = report.get("decision", {}).get(metric)
            if value is not None and math.isfinite(float(value)):
                result[f"layer:{layer}/{experiment}"] = float(value)
    return result


__all__ = ["InterventionSpec", "LayerImportance", "build_intervention_specs"]
