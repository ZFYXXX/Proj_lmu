"""Provider-aware prefill logit-lens trajectory for the first answer token."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from mllm_comp.analysis.artifacts import write_streaming_analysis_result
from mllm_comp.analysis.bindings import (
    AnalysisBackboneBinding,
    DecoderLayerBinding,
    resolve_analysis_binding,
    select_decoder_layers,
)
from mllm_comp.analysis.common.prediction import (
    PREDICTION_METRICS,
    PredictionSampleAccumulator,
    prediction_dynamics_metrics,
)
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
from mllm_comp.analysis.common.tensors import hidden_tensor, logits_tensor
from mllm_comp.analysis.common.token_mask import build_token_group_layout
from mllm_comp.analysis.hooks import HookSession
from mllm_comp.core.errors import BackendIntegrationError
from mllm_comp.core.manifest import AnalysisRecord


class PredictionDynamics:
    """Read every selected provider layer through the native final norm and head."""

    name = "prediction_dynamics"
    version = "1.0.0"
    supported_metrics = set(PREDICTION_METRICS)

    def validate_config(self, config: dict[str, Any]) -> None:
        validate_common_probe_config(config, context="prediction dynamics")
        metrics = set(
            string_sequence(
                config,
                "metrics",
                PREDICTION_METRICS,
                context="prediction dynamics",
            )
        )
        if not metrics or not metrics.issubset(self.supported_metrics):
            unknown = sorted(metrics - self.supported_metrics)
            raise ValueError(
                f"Unsupported prediction-dynamics metrics: "
                f"{unknown or sorted(metrics)}"
            )
        validate_layer_selection(
            config.get("layers", "all"),
            context="prediction dynamics layers",
        )
        positive_integer(
            config,
            "top_k",
            10,
            context="prediction dynamics",
        )
        nonnegative_finite_number(
            config,
            "final_lens_js_tolerance",
            1e-7,
            context="prediction dynamics",
        )

    def open(
        self,
        *,
        backbone: Any,
        runtime: Any,
        config: dict[str, Any],
        phase: str,
        output_dir: str,
    ) -> "_PredictionDynamicsSession":
        self.validate_config(config)
        return _PredictionDynamicsSession(
            plugin=self,
            backbone=backbone,
            runtime=runtime,
            config=config,
            phase=phase,
            output_dir=output_dir,
        )

@dataclass
class _PredictionDynamicsSession:
    plugin: PredictionDynamics
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
            context="Prediction dynamics",
        )
        configured_metrics = set(
            string_sequence(
                self.config,
                "metrics",
                PREDICTION_METRICS,
                context="prediction dynamics",
            )
        )
        self.metrics = tuple(
            metric for metric in PREDICTION_METRICS if metric in configured_metrics
        )
        self.top_k = int(self.config.get("top_k", 10))
        processor = getattr(self.runtime, "processor", None)
        self.tokenizer = getattr(processor, "tokenizer", processor)
        self.records = StreamingAnalysisRecords(
            output_dir=self.output_dir,
            prefix=self.plugin.name,
            save_raw=bool(self.config.get("save_raw", False)),
            accumulator_factory=lambda: PredictionSampleAccumulator(
                metrics=self.metrics
            ),
            aggregate_by_task=True,
        )
        self._finished = False
        self._closed = False

    def process(self, sample: Mapping[str, Any]) -> None:
        if self._finished or self._closed:
            raise BackendIntegrationError(
                "Cannot process a sample after prediction-dynamics completion."
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
            probe_name="Prediction dynamics",
        )
        try:
            import torch
        except ImportError as exc:  # pragma: no cover
            raise BackendIntegrationError("Prediction dynamics requires PyTorch.") from exc

        capture = _PredictionBoundaryCapture(self.binding, self.selected_layers)
        output: Any | None = None
        try:
            with HookSession() as hooks:
                capture.install(hooks)
                with torch.inference_mode():
                    output = self.backbone.runner.forward_train(self.runtime, state)
            final_logits = logits_tensor(output).detach()
            if int(final_logits.shape[0]) != 1:
                raise BackendIntegrationError(
                    "Prediction dynamics currently requires batch size one."
                )
            sequence_length = int(final_logits.shape[1])
            layout = build_token_group_layout(
                state,
                sequence_length=sequence_length,
                backbone_name=self.binding.backbone_name,
                runtime=self.runtime,
            )
            valid_positions = tuple(
                index
                for index, label in enumerate(layout.labels_by_sample[0])
                if label != "padding"
            )
            if not valid_positions:
                raise BackendIntegrationError(
                    "Prediction dynamics found no valid decision token."
                )
            decision_position = valid_positions[-1]
            final_decision_logits = final_logits[0, decision_position].float()
            boundaries = capture.finish(sequence_length=sequence_length)
            layer_results: dict[str, Any] = {}
            with torch.inference_mode():
                for layer in self.selected_layers:
                    hidden = boundaries[layer.index]
                    decision_hidden = hidden[0, decision_position].reshape(1, 1, -1)
                    lens_logits = self.binding.lm_head(
                        self.binding.final_norm(decision_hidden)
                    )[0, 0].float()
                    layer_results[str(layer.index)] = prediction_dynamics_metrics(
                        lens_logits,
                        final_decision_logits,
                        top_k=self.top_k,
                        tokenizer=self.tokenizer,
                    )
            last_layer = self.binding.layers[-1].index
            if last_layer in boundaries:
                last_metrics = layer_results[str(last_layer)]["metrics"]
                tolerance = float(self.config.get("final_lens_js_tolerance", 1e-7))
                if float(last_metrics["js"]) > tolerance:
                    raise BackendIntegrationError(
                        "Final-layer logit lens does not reproduce the model's final "
                        f"decision distribution (js={last_metrics['js']:.6g}, "
                        f"tolerance={tolerance:.6g})."
                    )
            return {
                "sample_id": sample["sample_id"],
                "task_name": sample["task_name"],
                "decision_position": decision_position,
                "decision_token_group": layout.labels_by_sample[0][decision_position],
                "token_counts": layout.counts(),
                "token_layout_evidence": dict(layout.evidence),
                "boundary_semantics": capture.evidence(),
                "layers": layer_results,
            }
        finally:
            capture.clear()
            del output

    def finish(self) -> AnalysisRecord:
        if self._finished:
            raise BackendIntegrationError("Prediction-dynamics session was already finished.")
        self._finished = True
        if self.records.successful_count == 0:
            reasons = "; ".join(
                str(item.get("reason"))
                for item in self.records.skipped_examples[:3]
            )
            raise BackendIntegrationError(
                "Prediction dynamics produced no successful samples. " + reasons
            )
        aggregate = self.records.overall_result()
        aggregate.update(
            {
                "task_aggregates": self.records.task_results(),
                "metrics": list(self.metrics),
                "configured_layers": [layer.index for layer in self.selected_layers],
                "top_k": self.top_k,
                "trajectory": "prefill_first_answer_token_logit_lens_v1",
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
            filename_stem="prediction_dynamics",
            records=self.records,
            aggregate=aggregate,
            summary={
                "layer_count": len(self.selected_layers),
                "finite_aggregate_values": finite_metric_count(
                    aggregate["layers"],
                    self.metrics,
                ),
                "trajectory": "prefill_first_answer_token_logit_lens_v1",
            },
        )

    def close(self) -> None:
        self._closed = True
        self.records.close()


class _PredictionBoundaryCapture:
    """Capture provider post-layer boundaries, including Qwen3 DeepStack injection."""

    def __init__(
        self,
        binding: AnalysisBackboneBinding,
        selected_layers: Sequence[DecoderLayerBinding],
    ) -> None:
        self.binding = binding
        self.selected_layers = tuple(selected_layers)
        self.values: dict[int, Any] = {}
        self.routes: dict[str, str] = {}

    def install(self, hooks: HookSession) -> None:
        final_index = self.binding.layers[-1].index
        by_index = {layer.index: layer for layer in self.binding.layers}
        for layer in self.selected_layers:
            if layer.index == final_index:
                module = self.binding.final_norm
                route = "final_norm_input"
            else:
                module = by_index[layer.index + 1].layer
                route = f"next_layer_input:{layer.index + 1}"
            self.routes[str(layer.index)] = route
            hooks.register_forward_pre_hook(
                module,
                lambda _module, args, index=layer.index: self._store(
                    index,
                    hidden_tensor(args),
                ),
            )

    def _store(self, layer: int, hidden: Any) -> None:
        if layer in self.values:
            raise BackendIntegrationError(
                f"Prediction boundary for layer {layer} fired more than once."
            )
        self.values[layer] = hidden.detach()

    def finish(self, *, sequence_length: int) -> dict[int, Any]:
        missing = sorted(
            layer.index for layer in self.selected_layers if layer.index not in self.values
        )
        if missing:
            raise BackendIntegrationError(
                f"Prediction dynamics missed provider boundaries for layers {missing}."
            )
        shapes = {str(layer): tuple(value.shape) for layer, value in self.values.items()}
        invalid = {
            layer: shape
            for layer, shape in shapes.items()
            if len(shape) != 3 or shape[0] != 1 or shape[1] != sequence_length
        }
        if invalid:
            raise BackendIntegrationError(
                "Prediction boundaries do not match final prefill sequence: "
                f"{invalid}."
            )
        return dict(self.values)

    def evidence(self) -> dict[str, Any]:
        return {
            "contract": "provider_post_layer_boundary_v1",
            "routes": dict(self.routes),
            "qwen3_deepstack_semantics": (
                "captured_before_next_decoder_layer_after_provider_injection"
                if self.binding.backbone_name == "qwen3_vl"
                else "not_applicable"
            ),
        }

    def clear(self) -> None:
        self.values.clear()


__all__ = ["PredictionDynamics"]
