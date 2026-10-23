"""Modality-aware hierarchical first-order Taylor analysis."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping

from mllm_comp.analysis.artifacts import make_analysis_record, write_analysis_artifact
from mllm_comp.analysis.bindings import (
    layer_batches,
    require_eager_attention,
    require_taylor_semantics,
    resolve_analysis_binding,
    select_decoder_layers,
)
from mllm_comp.analysis.common.rank import (
    aggregate_task_percentiles,
    aggregate_task_vector_percentiles,
)
from mllm_comp.analysis.common.attention_weights import AttentionWeightCapture
from mllm_comp.analysis.common.config import (
    positive_integer,
    validate_common_probe_config,
    validate_layer_selection,
)
from mllm_comp.analysis.common.sample import (
    prepare_teacher_forced_analysis_forward_state,
)
from mllm_comp.analysis.common.taylor import (
    FrozenModelParameters,
    TAYLOR_TARGET_MODALITIES,
    TAYLOR_ESTIMATOR_VERSION,
)
from mllm_comp.analysis.common.taylor_aggregate import (
    SCORE_FAMILIES,
    TaylorTaskAccumulator,
    add_bottom_up_mass_views,
    ffn_modality_cross_check,
    partition_taylor_task_scores,
)
from mllm_comp.analysis.common.taylor_capture import HierarchicalTaylorCapture
from mllm_comp.analysis.common.taylor_validation import (
    run_causal_validation,
    select_causal_validation_candidates,
    validate_causal_validation_config,
)
from mllm_comp.analysis.common.token_mask import build_token_group_layout
from mllm_comp.analysis.hooks import HookSession
from mllm_comp.core.errors import BackendIntegrationError
from mllm_comp.core.manifest import AnalysisRecord


class HierarchicalTaylor:
    """Collect the full hierarchy in bounded selected-layer batches."""

    name = "hierarchical_taylor"
    version = "1.0.0"
    requires_teacher_forcing = True

    def validate_config(self, config: dict[str, Any]) -> None:
        validate_common_probe_config(config, context="hierarchical Taylor")
        validate_layer_selection(
            config.get("layers", "all"),
            context="hierarchical Taylor layers",
        )
        positive_integer(
            config,
            "layer_batch_size",
            1,
            context="hierarchical Taylor",
        )
        bottom_fraction = config.get("ffn_bottom_fraction", 0.3)
        if isinstance(bottom_fraction, bool) or not isinstance(
            bottom_fraction, (int, float)
        ):
            raise ValueError("hierarchical Taylor ffn_bottom_fraction must be numeric.")
        if not 0.0 < float(bottom_fraction) < 1.0:
            raise ValueError(
                "hierarchical Taylor ffn_bottom_fraction must be between zero and one."
            )
        if config.get("objective") not in {
            None,
            "teacher_forced_answer_cross_entropy_v1",
        }:
            raise ValueError(
                "hierarchical Taylor only supports the native teacher-forced answer CE objective."
            )
        target_modality = config.get("target_modality", "vision")
        if (
            not isinstance(target_modality, str)
            or target_modality not in TAYLOR_TARGET_MODALITIES
        ):
            raise ValueError(
                "hierarchical Taylor target_modality must be 'vision' or 'audio'."
            )
        validate_causal_validation_config(config)

    def open(
        self,
        *,
        backbone: Any,
        runtime: Any,
        config: dict[str, Any],
        phase: str,
        output_dir: str,
    ) -> "_HierarchicalTaylorSession":
        self.validate_config(config)
        return _HierarchicalTaylorSession(
            plugin=self,
            backbone=backbone,
            runtime=runtime,
            config=config,
            phase=phase,
            output_dir=output_dir,
        )

@dataclass
class _HierarchicalTaylorSession:
    plugin: HierarchicalTaylor
    backbone: Any
    runtime: Any
    config: dict[str, Any]
    phase: str
    output_dir: str

    def __post_init__(self) -> None:
        self.binding = resolve_analysis_binding(self.backbone, self.runtime.model)
        self.target_modality = str(self.config.get("target_modality", "vision"))
        self.selected_layers = select_decoder_layers(
            self.binding,
            self.config.get("layers", "all"),
            context="Hierarchical Taylor",
        )
        self.layer_batch_size = int(self.config.get("layer_batch_size", 1))
        require_taylor_semantics(self.binding, self.selected_layers)
        self.attention_backends = require_eager_attention(
            self.selected_layers,
            context="Hierarchical Taylor Path analysis",
        )
        self.causal_validation = validate_causal_validation_config(self.config)
        self.accumulator = TaylorTaskAccumulator()
        self.sample_summaries: list[dict[str, Any]] = []
        self.raw_samples: list[dict[str, Any]] = []
        self.skipped_samples: list[dict[str, Any]] = []
        self._finished = False
        self._closed = False

    def process(self, sample: Mapping[str, Any]) -> None:
        if self._finished or self._closed:
            raise BackendIntegrationError(
                "Cannot process a sample after hierarchical Taylor completion."
            )
        try:
            result = self._process_sample(sample)
            accepted = self.accumulator.update(str(sample["task_name"]), result["report"])
            summary = {
                key: value for key, value in result.items() if key != "report"
            }
            summary["finite_component_scores"] = accepted
            self.sample_summaries.append(summary)
            if bool(self.config.get("save_raw", False)):
                self.raw_samples.append(
                    {
                        **summary,
                        "report": result["report"],
                    }
                )
        except (BackendIntegrationError, ValueError, FloatingPointError) as exc:
            self.skipped_samples.append(
                {
                    "sample_id": sample.get("sample_id"),
                    "task_name": sample.get("task_name"),
                    "reason": f"{type(exc).__name__}: {exc}",
                }
            )

    def _process_sample(self, sample: Mapping[str, Any]) -> dict[str, Any]:
        state, objective = prepare_teacher_forced_analysis_forward_state(
            backbone=self.backbone,
            runtime=self.runtime,
            sample=sample,
            probe_name="Hierarchical Taylor",
            output_attentions=False,
        )
        try:
            import torch
        except ImportError as exc:  # pragma: no cover - real runtime requires torch
            raise BackendIntegrationError(
                "Hierarchical Taylor requires PyTorch."
            ) from exc

        model = self.runtime.model
        zero_grad = getattr(model, "zero_grad", None)
        loss_value: float | None = None
        layout: Any | None = None
        report: dict[str, Any] | None = None
        capture_pass_count = 0
        try:
            with FrozenModelParameters(model):
                for layer_batch in layer_batches(
                    self.selected_layers,
                    self.layer_batch_size,
                ):
                    capture = HierarchicalTaylorCapture(
                        layer_batch,
                        self.binding.attention_geometry,
                        target_modality=self.target_modality,
                    )
                    attention_capture = AttentionWeightCapture(layer_batch)
                    output: Any | None = None
                    loss: Any | None = None
                    attentions: Mapping[int, Any] | None = None
                    if callable(zero_grad):
                        zero_grad(set_to_none=True)
                    try:
                        with HookSession() as hooks, torch.enable_grad():
                            capture.install(
                                hooks,
                                input_embeddings=self.binding.input_embeddings,
                            )
                            attention_capture.install(hooks)
                            output = self.backbone.runner.forward_train(
                                self.runtime,
                                state,
                            )
                            loss = _output_loss(output)
                            attentions = attention_capture.finish()
                            first_attention = attentions[layer_batch[0].index]
                            if (
                                first_attention is None
                                or len(first_attention.shape) != 4
                            ):
                                raise BackendIntegrationError(
                                    "Hierarchical Taylor requires full eager "
                                    "attention matrices."
                                )
                            batch_layout = build_token_group_layout(
                                state,
                                sequence_length=int(first_attention.shape[-1]),
                                backbone_name=self.binding.backbone_name,
                                runtime=self.runtime,
                            )
                            capture.bind_layout(batch_layout)
                            batch_loss_value = float(loss.detach().cpu().item())
                            if not math.isfinite(batch_loss_value):
                                raise FloatingPointError(
                                    "Teacher-forced Taylor loss is non-finite."
                                )
                            if loss_value is None:
                                loss_value = batch_loss_value
                                layout = batch_layout
                            else:
                                if not math.isclose(
                                    batch_loss_value,
                                    loss_value,
                                    rel_tol=1e-6,
                                    abs_tol=1e-6,
                                ):
                                    raise BackendIntegrationError(
                                        "Bounded Taylor layer batches produced "
                                        "different baseline losses."
                                    )
                                if (
                                    layout is None
                                    or batch_layout.labels_by_sample
                                    != layout.labels_by_sample
                                ):
                                    raise BackendIntegrationError(
                                        "Bounded Taylor layer batches produced "
                                        "different token layouts."
                                    )
                            loss.backward()
                            batch_report = capture.finish(attentions)
                            add_bottom_up_mass_views(batch_report)
                        report = _merge_taylor_reports(report, batch_report)
                        capture_pass_count += 1
                    finally:
                        output = loss = attentions = None
                        capture.clear()
                        attention_capture.clear()
                        if callable(zero_grad):
                            zero_grad(set_to_none=True)
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                if report is None or loss_value is None or layout is None:
                    raise BackendIntegrationError(
                        "Hierarchical Taylor produced no bounded capture batch."
                    )
                report["capture_execution"] = {
                    "contract": "bounded_selected_layer_batches_v2",
                    "layer_batch_size": self.layer_batch_size,
                    "capture_pass_count": capture_pass_count,
                    "selected_layer_count": len(self.selected_layers),
                }
                if self.causal_validation["enabled"]:
                    selections = select_causal_validation_candidates(
                        report,
                        kinds=self.causal_validation["kinds"],
                        component_count=self.causal_validation["component_count"],
                        seed=self.causal_validation["seed"],
                        sample_id=str(sample["sample_id"]),
                    )
                    report["causal_validation"] = run_causal_validation(
                        binding=self.binding,
                        layout=layout,
                        state=state,
                        forward=lambda: self.backbone.runner.forward_train(
                            self.runtime, state
                        ),
                        baseline_loss=loss_value,
                        selections=selections,
                        target_modality=self.target_modality,
                    )
            return {
                "sample_id": sample["sample_id"],
                "task_name": sample["task_name"],
                "loss": loss_value,
                "objective": objective,
                "target_source": sample.get("target_source"),
                "token_counts": layout.counts(),
                "token_layout_evidence": dict(layout.evidence),
                "capture_pass_count": capture_pass_count,
                "layer_batch_size": self.layer_batch_size,
                "causal_validation": report.get("causal_validation"),
                "report": report,
            }
        finally:
            if callable(zero_grad):
                zero_grad(set_to_none=True)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def finish(self) -> AnalysisRecord:
        if self._finished:
            raise BackendIntegrationError(
                "Hierarchical Taylor session was already finished."
            )
        self._finished = True
        if not self.sample_summaries:
            reasons = "; ".join(
                str(item.get("reason")) for item in self.skipped_samples[:3]
            )
            raise BackendIntegrationError(
                "Hierarchical Taylor produced no successful samples. " + reasons
            )
        scores_by_family = self.accumulator.scores_by_family()
        vector_scores_by_family = self.accumulator.vector_scores_by_family()
        partitioned_scores = partition_taylor_task_scores(scores_by_family)
        ranks = {
            family: {
                granularity: aggregate_task_percentiles(task_scores)
                for granularity, task_scores in partitioned_scores[family].items()
                if any(task_scores.values())
            }
            for family in SCORE_FAMILIES
        }
        vector_ranks = {
            family: aggregate_task_vector_percentiles(
                vector_scores_by_family[family]
            )
            for family in SCORE_FAMILIES
            if vector_scores_by_family[family]
        }
        bottom_fraction = float(self.config.get("ffn_bottom_fraction", 0.3))
        cross_check = ffn_modality_cross_check(
            scores_by_family,
            bottom_fraction=bottom_fraction,
            vector_scores_by_family=vector_scores_by_family,
            modality_group=self.target_modality,
        )
        common = {
            "output_dir": self.output_dir,
            "probe_name": str(self.config.get("name", self.plugin.name)),
            "method": str(self.config.get("method", self.plugin.name)),
            "method_version": self.plugin.version,
            "phase": self.phase,
            "backbone_name": self.binding.backbone_name,
            "checkpoint": str(self.runtime.model_path_or_id or ""),
            "checkpoint_revision": self.runtime.revision,
            "capability_status": self.config.get("_capability_status"),
        }
        shared_metadata = {
            "successful_samples": len(self.sample_summaries),
            "skipped_samples": len(self.skipped_samples),
            "estimator_version": TAYLOR_ESTIMATOR_VERSION,
        }
        scores_path = write_analysis_artifact(
            **common,
            filename="hierarchical_taylor.task_scores.json",
            artifact_kind="task_raw_scores",
            payload={
                "estimator_version": TAYLOR_ESTIMATOR_VERSION,
                "task_sample_counts": dict(self.accumulator.sample_counts),
                "scores_by_family": scores_by_family,
                "score_vectors_by_family": vector_scores_by_family,
                "score_storage": "compact_ffn_layer_group_vectors_v2",
                "rank_partitioning": "score_family_and_granularity_v2",
                "ffn_modality_cross_check": cross_check,
                "target_modality": self.target_modality,
                "sample_summaries": self.sample_summaries,
                "skipped_samples": self.skipped_samples,
                "attention_backends": self.attention_backends,
                "binding_evidence": dict(self.binding.evidence),
            },
            metadata=shared_metadata,
        )
        ranks_path = write_analysis_artifact(
            **common,
            filename="hierarchical_taylor.task_ranks.json",
            artifact_kind="cross_task_percentile_ranks",
            payload={
                "estimator_version": TAYLOR_ESTIMATOR_VERSION,
                "aggregation": (
                    "mean_task_percentile_average_ties_by_granularity_v2"
                ),
                "partitioning": "score_family_and_granularity_v2",
                "score_families": ranks,
                "score_vector_families": vector_ranks,
            },
            metadata=shared_metadata,
        )
        artifact_paths = {
            "task_scores": scores_path,
            "task_ranks": ranks_path,
        }
        if bool(self.config.get("save_raw", False)):
            artifact_paths["per_sample"] = write_analysis_artifact(
                **common,
                filename="hierarchical_taylor.raw.json",
                artifact_kind="per_sample",
                payload={
                    "samples": self.raw_samples,
                    "skipped_samples": self.skipped_samples,
                },
                metadata=shared_metadata,
            )
        component_counts = {
            family: sum(
                len(values) for values in scores_by_family[family].values()
            )
            + sum(
                len(vector)
                for task_vectors in vector_scores_by_family[family].values()
                for vector in task_vectors.values()
            )
            for family in SCORE_FAMILIES
        }
        return make_analysis_record(
            name=str(self.config.get("name", self.plugin.name)),
            phase=self.phase,
            method=str(self.config.get("method", self.plugin.name)),
            summary={
                "successful_samples": len(self.sample_summaries),
                "skipped_samples": len(self.skipped_samples),
                "layer_count": len(self.selected_layers),
                "head_count": self.binding.attention_geometry.num_query_heads,
                "component_score_counts": component_counts,
                "estimator_version": TAYLOR_ESTIMATOR_VERSION,
                "objective": "teacher_forced_answer_cross_entropy_v1",
                "target_modality": self.target_modality,
                "task_aggregation": (
                    "mean_task_percentile_average_ties_by_granularity_v2"
                ),
                "task_rank_partitioning": "score_family_and_granularity_v2",
                "layer_batch_size": self.layer_batch_size,
                "activation_cross_check": True,
                "causal_validation": dict(self.causal_validation),
            },
            artifact_paths=artifact_paths,
            sample_count=len(self.sample_summaries),
        )

    def close(self) -> None:
        self._closed = True
        self.raw_samples.clear()


def _merge_taylor_reports(
    current: dict[str, Any] | None,
    incoming: Mapping[str, Any],
) -> dict[str, Any]:
    stable_fields = (
        "score_definitions",
        "target_modality",
        "component_groups",
        "path_types",
    )
    incoming_layers = dict(incoming.get("layers", {}))
    if not incoming_layers:
        raise BackendIntegrationError("Taylor capture batch returned no layers.")
    if current is None:
        return {
            **{field: incoming.get(field) for field in stable_fields},
            "layers": incoming_layers,
        }
    for field in stable_fields:
        if current.get(field) != incoming.get(field):
            raise BackendIntegrationError(
                f"Taylor capture batches disagree on {field}."
            )
    overlap = sorted(set(current["layers"]) & set(incoming_layers))
    if overlap:
        raise BackendIntegrationError(
            f"Taylor capture batches overlap decoder layers {overlap}."
        )
    current["layers"].update(incoming_layers)
    return current


def _output_loss(output: Any) -> Any:
    loss = getattr(output, "loss", None)
    if loss is None and isinstance(output, Mapping):
        loss = output.get("loss")
    if loss is None or not hasattr(loss, "backward"):
        raise BackendIntegrationError(
            "Teacher-forced model output contains no differentiable native loss."
        )
    if getattr(loss, "ndim", None) != 0:
        raise BackendIntegrationError("Teacher-forced Taylor loss must be scalar.")
    try:
        import torch
    except ImportError as exc:  # pragma: no cover
        raise BackendIntegrationError("Taylor loss validation requires PyTorch.") from exc
    if not bool(torch.isfinite(loss).item()):
        raise FloatingPointError("Teacher-forced Taylor loss is non-finite.")
    if not bool(getattr(loss, "requires_grad", False)):
        raise BackendIntegrationError(
            "Teacher-forced Taylor loss does not retain an activation graph."
        )
    return loss


__all__ = ["HierarchicalTaylor"]
