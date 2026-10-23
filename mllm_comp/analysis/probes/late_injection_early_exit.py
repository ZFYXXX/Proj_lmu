"""Timing analysis for cross-modal early exit and late modality injection."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Iterable, Mapping, Sequence

from mllm_comp.analysis.artifacts import write_streaming_analysis_result
from mllm_comp.analysis.bindings import (
    AnalysisBackboneBinding,
    DecoderLayerBinding,
    resolve_analysis_binding,
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
    install_token_hidden_replacement,
)
from mllm_comp.analysis.common.rank import (
    aggregate_task_percentiles,
    finite_metric_count,
    select_ranking_metric,
)
from mllm_comp.analysis.common.sample import prepare_analysis_forward_state
from mllm_comp.analysis.common.streaming import StreamingAnalysisRecords
from mllm_comp.analysis.common.tensors import hidden_tensor, logits_tensor
from mllm_comp.analysis.common.token_mask import TokenGroupLayout, build_token_group_layout
from mllm_comp.analysis.hooks import HookSession
from mllm_comp.core.errors import BackendIntegrationError
from mllm_comp.core.manifest import AnalysisRecord


_EXPERIMENT_TYPES = ("early_exit", "late_injection")
_TOKEN_GROUPS = ("system", "text", "vision")
_TARGET_MODALITIES = {"vision", "audio"}
_MASK_BACKENDS = {"eager", "sdpa"}


@dataclass(frozen=True)
class TimingExperiment:
    experiment_type: str
    threshold: int

    @property
    def key(self) -> str:
        return f"{self.experiment_type}_{self.threshold}"


class LateInjectionEarlyExit:
    """Locate when selected modality information must enter and remain available."""

    name = "late_injection_early_exit"
    version = "1.0.0"
    supported_metrics = set(DISTRIBUTION_METRICS)
    supported_experiment_types = set(_EXPERIMENT_TYPES)

    def validate_config(self, config: dict[str, Any]) -> None:
        validate_common_probe_config(config, context="timing analysis")
        metrics = set(
            string_sequence(
                config,
                "metrics",
                DISTRIBUTION_METRICS,
                context="timing analysis",
            )
        )
        if not metrics or not metrics.issubset(self.supported_metrics):
            unknown = sorted(metrics - self.supported_metrics)
            raise ValueError(
                f"Unsupported timing-analysis metrics: {unknown or sorted(metrics)}"
            )
        experiment_types = set(
            string_sequence(
                config,
                "experiment_types",
                _EXPERIMENT_TYPES,
                context="timing analysis",
            )
        )
        if not experiment_types or not experiment_types.issubset(
            self.supported_experiment_types
        ):
            unknown = sorted(experiment_types - self.supported_experiment_types)
            raise ValueError(
                f"Unsupported timing experiment types: "
                f"{unknown or sorted(experiment_types)}"
            )
        target_modality = config.get("target_modality", "vision")
        if (
            not isinstance(target_modality, str)
            or target_modality not in _TARGET_MODALITIES
        ):
            raise ValueError(
                "timing-analysis target_modality must be 'vision' or 'audio'."
            )
        validate_layer_selection(
            config.get("layers", "all"),
            context="timing-analysis layers",
        )
        for field in ("early_exit_thresholds", "late_injection_thresholds"):
            value = config.get(field, "all")
            validate_layer_selection(
                value,
                context=f"timing-analysis {field}",
            )
        ranking_metric = config.get("ranking_metric")
        if ranking_metric is not None and (
            not isinstance(ranking_metric, str) or ranking_metric not in metrics
        ):
            raise ValueError(
                "timing-analysis ranking_metric must also be present in metrics."
            )
        rankable = metrics & set(IMPORTANCE_RANKING_DIRECTIONS)
        if ranking_metric is not None and ranking_metric not in rankable:
            raise ValueError(
                "timing-analysis ranking_metric has no unambiguous importance "
                "direction. Use KL, JS, changed_prob_on_base_top1, or top1_changed."
            )
        if ranking_metric is None and not rankable:
            raise ValueError(
                "timing-analysis metrics must include at least one rank-compatible "
                "distribution-shift metric."
            )

    def open(
        self,
        *,
        backbone: Any,
        runtime: Any,
        config: dict[str, Any],
        phase: str,
        output_dir: str,
    ) -> "_TimingSession":
        self.validate_config(config)
        return _TimingSession(
            plugin=self,
            backbone=backbone,
            runtime=runtime,
            config=config,
            phase=phase,
            output_dir=output_dir,
        )

@dataclass
class _TimingSession:
    plugin: LateInjectionEarlyExit
    backbone: Any
    runtime: Any
    config: dict[str, Any]
    phase: str
    output_dir: str

    def __post_init__(self) -> None:
        self.binding = resolve_analysis_binding(self.backbone, self.runtime.model)
        self.target_modality = str(self.config.get("target_modality", "vision"))
        self.report_token_groups = (
            _TOKEN_GROUPS
            if self.target_modality == "vision"
            else ("system", "text", self.target_modality)
        )
        self.experiments = _timing_experiments(
            layer_count=len(self.binding.layers),
            config=self.config,
        )
        configured_metrics = set(
            string_sequence(
                self.config,
                "metrics",
                DISTRIBUTION_METRICS,
                context="timing analysis",
            )
        )
        self.metrics = tuple(
            metric for metric in DISTRIBUTION_METRICS if metric in configured_metrics
        )
        self.ranking_metric = select_ranking_metric(
            self.metrics,
            self.config.get("ranking_metric"),
            IMPORTANCE_RANKING_DIRECTIONS,
        )
        self.ranking_higher_is_more_important = IMPORTANCE_RANKING_DIRECTIONS[
            self.ranking_metric
        ]
        self.attention_backends = _require_mask_backends(self.binding.layers)
        self.records = StreamingAnalysisRecords(
            output_dir=self.output_dir,
            prefix=self.plugin.name,
            save_raw=bool(self.config.get("save_raw", False)),
            accumulator_factory=lambda: _TimingSampleAccumulator(
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
                "Cannot process a sample after timing-analysis completion."
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
            probe_name="Late injection / early exit",
        )
        try:
            import torch
        except ImportError as exc:  # pragma: no cover
            raise BackendIntegrationError(
                "Late injection / early exit requires PyTorch."
            ) from exc

        injection_capture = _ModalityInjectionCapture(
            self.binding,
            target_modality=self.target_modality,
        )
        baseline_output: Any | None = None
        baseline_logits: Any | None = None
        try:
            with torch.inference_mode():
                with HookSession() as hooks:
                    injection_capture.install(hooks)
                    baseline_output = self.backbone.runner.forward_train(
                        self.runtime,
                        state,
                    )
                baseline_logits = logits_tensor(baseline_output).detach()
                sequence_length = int(baseline_logits.shape[1])
                layout = build_token_group_layout(
                    state,
                    sequence_length=sequence_length,
                    backbone_name=self.binding.backbone_name,
                    runtime=self.runtime,
                )
                base_modality, deepstack_features, injection_evidence = (
                    injection_capture.finish(layout=layout)
                )
                results: dict[str, Any] = {}
                for experiment in self.experiments:
                    changed_output: Any | None = None
                    changed_logits: Any | None = None
                    evidence: dict[str, Any] = {
                        "experiment_type": experiment.experiment_type,
                        "threshold": experiment.threshold,
                        "blocked_layers": [],
                    }
                    try:
                        with HookSession() as hooks:
                            for layer in self.binding.layers:
                                if not _blocks_layer(experiment, layer.index):
                                    continue
                                layer_evidence: dict[str, Any] = {}
                                _install_bidirectional_cross_modal_block(
                                    hooks,
                                    layer=layer,
                                    layout=layout,
                                    target_modality=self.target_modality,
                                    evidence=layer_evidence,
                                )
                                evidence["blocked_layers"].append(
                                    {
                                        "layer": layer.index,
                                        **layer_evidence,
                                    }
                                )
                            if (
                                experiment.experiment_type == "late_injection"
                                and experiment.threshold < len(self.binding.layers)
                            ):
                                reinjection_evidence: dict[str, Any] = {}
                                bundle = _reinjection_bundle(
                                    base_modality,
                                    deepstack_features,
                                    threshold=experiment.threshold,
                                )
                                install_token_hidden_replacement(
                                    hooks,
                                    layer=self.binding.layers[
                                        experiment.threshold
                                    ].layer,
                                    layout=layout,
                                    target_group=self.target_modality,
                                    replacement=bundle,
                                    evidence=reinjection_evidence,
                                )
                                reinjection_evidence.update(
                                    {
                                        "layer": experiment.threshold,
                                        "deepstack_branches_included": min(
                                            experiment.threshold,
                                            len(deepstack_features),
                                        ),
                                    }
                                )
                                evidence["reinjection"] = reinjection_evidence
                            elif experiment.experiment_type == "late_injection":
                                evidence["reinjection"] = {
                                    "layer": None,
                                    "semantics": "never_reinjected_inside_decoder",
                                }
                            changed_output = self.backbone.runner.forward_train(
                                self.runtime,
                                state,
                            )
                        changed_logits = logits_tensor(changed_output).detach()
                        report = distribution_shift_report(
                            baseline_logits,
                            changed_logits,
                            layout=layout,
                            token_groups=self.report_token_groups,
                            metrics=self.metrics,
                        )
                        results[experiment.key] = {
                            "status": "finite",
                            "intervention": evidence,
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
                "injection_evidence": injection_evidence,
                "experiments": results,
            }
        finally:
            injection_capture.clear()
            del baseline_logits
            del baseline_output

    def finish(self) -> AnalysisRecord:
        if self._finished:
            raise BackendIntegrationError("Timing-analysis session was already finished.")
        self._finished = True
        if self.records.successful_count == 0:
            reasons = "; ".join(
                str(item.get("reason"))
                for item in self.records.skipped_examples[:3]
            )
            raise BackendIntegrationError(
                "Late injection / early exit produced no successful samples. "
                + reasons
            )
        aggregate = self.records.overall_result()
        task_aggregates = self.records.task_results()
        aggregate.update(
            {
                "task_aggregates": task_aggregates,
                "task_rank_aggregation": aggregate_task_percentiles(
                    {
                        task: _timing_scores(value, self.ranking_metric)
                        for task, value in task_aggregates.items()
                        if int(value.get("sample_count", 0)) > 0
                    },
                    higher_is_more_important=(
                        self.ranking_higher_is_more_important
                    ),
                ),
                "ranking_metric": self.ranking_metric,
                "ranking_higher_is_more_important": (
                    self.ranking_higher_is_more_important
                ),
                "metrics": list(self.metrics),
                "target_modality": self.target_modality,
                "experiments": [
                    {
                        "experiment_type": item.experiment_type,
                        "threshold": item.threshold,
                        "key": item.key,
                    }
                    for item in self.experiments
                ],
                "attention_backends": self.attention_backends,
                "injection_contract": _injection_contract(
                    self.binding,
                    target_modality=self.target_modality,
                ),
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
            filename_stem="late_injection_early_exit",
            records=self.records,
            aggregate=aggregate,
            aggregate_metadata={"experiment_count": len(self.experiments)},
            summary={
                "experiment_count": len(self.experiments),
                "finite_aggregate_values": finite_metric_count(
                    {
                        experiment_type: aggregate[experiment_type]
                        for experiment_type in _EXPERIMENT_TYPES
                    },
                    self.metrics,
                ),
                "ranking_metric": self.ranking_metric,
            },
        )

    def close(self) -> None:
        self._closed = True
        self.records.close()


class _ModalityInjectionCapture:
    """Capture exact base modality tokens and applicable provider branches."""

    def __init__(
        self,
        binding: AnalysisBackboneBinding,
        *,
        target_modality: str,
    ) -> None:
        self.binding = binding
        self.target_modality = target_modality
        self.base_hidden: Any | None = None
        self.deepstack_values: dict[int, list[Any]] = {}

    def install(self, hooks: HookSession) -> None:
        hooks.register_forward_pre_hook(
            self.binding.layers[0].layer,
            lambda _module, args: self._store_base(hidden_tensor(args)),
        )
        projectors = (
            self.binding.deepstack_projectors
            if self.target_modality == "vision"
            else ()
        )
        for index, projector in enumerate(projectors):
            hooks.register_forward_hook(
                projector,
                lambda _module, _args, output, index=index: self.deepstack_values.setdefault(
                    index,
                    [],
                ).append(hidden_tensor(output).detach()),
            )

    def _store_base(self, hidden: Any) -> None:
        if self.base_hidden is not None:
            raise BackendIntegrationError("Base modality embedding hook fired more than once.")
        self.base_hidden = hidden.detach()

    def finish(
        self,
        *,
        layout: TokenGroupLayout,
    ) -> tuple[Any, tuple[Any, ...], dict[str, Any]]:
        try:
            import torch
        except ImportError as exc:  # pragma: no cover
            raise BackendIntegrationError("Modality injection capture requires PyTorch.") from exc
        if self.base_hidden is None or len(self.base_hidden.shape) != 3:
            raise BackendIntegrationError("Timing analysis missed first-layer hidden states.")
        positions = layout.positions(0, self.target_modality)
        if not positions:
            raise BackendIntegrationError(
                "Late injection / early exit requires at least one "
                f"{self.target_modality} token."
            )
        indices = torch.tensor(
            positions,
            dtype=torch.long,
            device=self.base_hidden.device,
        )
        base_modality = self.base_hidden[0].index_select(0, indices).unsqueeze(0)
        features: list[Any] = []
        for index in range(len(self.binding.deepstack_projectors)):
            values = self.deepstack_values.get(index, [])
            if len(values) != 1:
                raise BackendIntegrationError(
                    "Qwen3 DeepStack projector must fire exactly once for the "
                    f"single-image timing contract; branch {index} fired {len(values)} times."
                )
            value = values[0]
            if len(value.shape) == 2:
                value = value.unsqueeze(0)
            if tuple(value.shape) != tuple(base_modality.shape):
                raise BackendIntegrationError(
                    "Qwen3 DeepStack branch shape does not match base visual tokens: "
                    f"branch={tuple(value.shape)}, base={tuple(base_modality.shape)}."
                )
            features.append(value)
        contract_modality = (
            "visual" if self.target_modality == "vision" else self.target_modality
        )
        return base_modality, tuple(features), {
            "contract": f"native_{contract_modality}_reinjection_bundle_v1",
            "base_route": "first_decoder_layer_input",
            f"{self.target_modality}_token_count": len(positions),
            "deepstack_branch_count": len(features),
            "deepstack_mapping": (
                "branch_i_is_added_after_decoder_layer_i"
                if self.binding.backbone_name == "qwen3_vl"
                else "not_applicable"
            ),
        }

    def clear(self) -> None:
        self.base_hidden = None
        self.deepstack_values.clear()


def _install_bidirectional_cross_modal_block(
    hooks: HookSession,
    *,
    layer: DecoderLayerBinding,
    layout: TokenGroupLayout,
    target_modality: str,
    evidence: dict[str, Any],
) -> None:
    other_to_target: dict[str, Any] = {}
    target_to_other: dict[str, Any] = {}
    other_group = f"non{target_modality}"
    install_attention_edge_block(
        hooks,
        attention=layer.attention,
        layout=layout,
        source_group=target_modality,
        target_group=None,
        non_source_queries=True,
        evidence=other_to_target,
    )
    install_attention_edge_block(
        hooks,
        attention=layer.attention,
        layout=layout,
        source_group=other_group,
        target_group=target_modality,
        evidence=target_to_other,
    )
    evidence.update(
        {
            f"{other_group}_query_to_{target_modality}_key": other_to_target,
            f"{target_modality}_query_to_{other_group}_key": target_to_other,
            "semantics": "bidirectional_cross_modal_attention_block_v1",
        }
    )


def _reinjection_bundle(
    base_modality: Any,
    deepstack_features: Sequence[Any],
    *,
    threshold: int,
) -> Any:
    bundle = base_modality.clone()
    for feature in deepstack_features[:threshold]:
        bundle = bundle + feature.to(device=bundle.device, dtype=bundle.dtype)
    return bundle


def _blocks_layer(experiment: TimingExperiment, layer: int) -> bool:
    if experiment.experiment_type == "early_exit":
        return layer > experiment.threshold
    if experiment.experiment_type == "late_injection":
        return layer < experiment.threshold
    raise BackendIntegrationError(
        f"Unknown timing experiment {experiment.experiment_type!r}."
    )


def _timing_experiments(
    *,
    layer_count: int,
    config: Mapping[str, Any],
) -> tuple[TimingExperiment, ...]:
    types = tuple(
        dict.fromkeys(
            string_sequence(
                config,
                "experiment_types",
                _EXPERIMENT_TYPES,
                context="timing analysis",
            )
        )
    )
    configured_layers = config.get("layers", "all")
    default_early = (
        list(range(layer_count))
        if configured_layers == "all"
        else list(dict.fromkeys(int(value) for value in configured_layers))
    )
    default_late = (
        list(range(1, layer_count + 1))
        if configured_layers == "all"
        else [int(value) + 1 for value in dict.fromkeys(configured_layers)]
    )
    early = config.get("early_exit_thresholds", default_early)
    late = config.get("late_injection_thresholds", default_late)
    if early == "all":
        early = list(range(layer_count))
    if late == "all":
        late = list(range(1, layer_count + 1))
    early = list(dict.fromkeys(int(value) for value in early))
    late = list(dict.fromkeys(int(value) for value in late))
    if any(value < 0 or value >= layer_count for value in early):
        raise BackendIntegrationError(
            f"Early-exit thresholds must be in [0, {layer_count - 1}], got {early}."
        )
    if any(value < 1 or value > layer_count for value in late):
        raise BackendIntegrationError(
            f"Late-injection thresholds must be in [1, {layer_count}], got {late}."
        )
    result: list[TimingExperiment] = []
    if "early_exit" in types:
        result.extend(TimingExperiment("early_exit", value) for value in early)
    if "late_injection" in types:
        result.extend(TimingExperiment("late_injection", value) for value in late)
    return tuple(result)


def _aggregate_timing_samples(
    samples: Iterable[Mapping[str, Any]],
    *,
    metrics: Sequence[str],
) -> dict[str, Any]:
    accumulator = _TimingSampleAccumulator(metrics=metrics)
    for sample in samples:
        accumulator.add(sample)
    return accumulator.result()


class _TimingSampleAccumulator:
    def __init__(
        self,
        *,
        metrics: Sequence[str],
        token_groups: Sequence[str] = _TOKEN_GROUPS,
    ) -> None:
        self._inner = LayerImportanceSampleAccumulator(
            metrics=metrics,
            token_groups=token_groups,
        )

    def add(self, sample: Mapping[str, Any]) -> None:
        layers: dict[str, dict[str, Any]] = {}
        for key, report in sample.get("experiments", {}).items():
            experiment_type, threshold = key.rsplit("_", maxsplit=1)
            layers.setdefault(experiment_type, {})[threshold] = report
        self._inner.add({"layers": layers})

    def result(self) -> dict[str, Any]:
        aggregate = self._inner.result()
        layers = aggregate["layers"]
        return {
            "sample_count": aggregate["sample_count"],
            "early_exit": layers.get("early_exit", {}),
            "late_injection": layers.get("late_injection", {}),
        }


def _timing_scores(aggregate: Mapping[str, Any], metric: str) -> dict[str, float]:
    result: dict[str, float] = {}
    for experiment_type in _EXPERIMENT_TYPES:
        for threshold, report in aggregate.get(experiment_type, {}).items():
            value = report.get("decision", {}).get(metric)
            if value is not None and math.isfinite(float(value)):
                result[f"{experiment_type}:{threshold}"] = float(value)
    return result


def _require_mask_backends(
    layers: Sequence[DecoderLayerBinding],
) -> dict[str, str]:
    result: dict[str, str] = {}
    for layer in layers:
        backend = getattr(
            getattr(layer.attention, "config", None),
            "_attn_implementation",
            None,
        )
        normalized = str(backend or "unknown")
        result[str(layer.index)] = normalized
        if normalized not in _MASK_BACKENDS:
            raise BackendIntegrationError(
                "Late injection / early exit requires eager or SDPA causal masks; "
                f"layer {layer.index} reports {normalized!r}."
            )
    return result


def _injection_contract(
    binding: AnalysisBackboneBinding,
    *,
    target_modality: str,
) -> dict[str, Any]:
    contract_modality = "visual" if target_modality == "vision" else target_modality
    base_state_key = (
        "base_visual_state"
        if target_modality == "vision"
        else f"base_{target_modality}_state"
    )
    return {
        "contract": f"native_{contract_modality}_reinjection_bundle_v1",
        base_state_key: (
            f"first_decoder_layer_input_{target_modality}_positions"
        ),
        "decoder_state_leakage": "excluded_by_threshold_layer_reset",
        "qwen3_deepstack": (
            {
                "projector_count": len(binding.deepstack_projectors),
                "mapping": "branch_i_is_added_after_decoder_layer_i",
                "threshold_bundle": "base_plus_sum_of_branches_with_i_below_K",
            }
            if binding.backbone_name == "qwen3_vl" and target_modality == "vision"
            else "not_applicable"
        ),
    }


__all__ = ["LateInjectionEarlyExit", "TimingExperiment"]
