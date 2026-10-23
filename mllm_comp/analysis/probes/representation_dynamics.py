"""Residual-stream representation dynamics over semantic decoder hooks."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Iterable, Mapping

from mllm_comp.analysis.artifacts import write_streaming_analysis_result
from mllm_comp.analysis.bindings import (
    resolve_analysis_binding,
    select_decoder_layers,
)
from mllm_comp.analysis.common.config import (
    nonnegative_finite_number,
    string_sequence,
    validate_common_probe_config,
    validate_layer_selection,
)
from mllm_comp.analysis.common.representation import RepresentationSampleAccumulator
from mllm_comp.analysis.common.residual import ResidualNodeCapture
from mllm_comp.analysis.common.sample import prepare_analysis_forward_state
from mllm_comp.analysis.common.streaming import StreamingAnalysisRecords
from mllm_comp.analysis.common.token_mask import TokenGroupLayout, build_token_group_layout
from mllm_comp.analysis.hooks import HookSession
from mllm_comp.core.errors import BackendIntegrationError
from mllm_comp.core.manifest import AnalysisRecord


_COMPARISONS = {
    "attn_res": ("before_attn", "after_attn"),
    "attn_pure": ("before_attn", "attn_only"),
    "ffn_res": ("after_attn", "after_ffn"),
    "ffn_pure": ("after_attn", "ffn_only"),
    "full_layer_res": ("before_attn", "after_ffn"),
}

_COLLECT_POINT_ALIASES = {
    "layer_input": ("before_attn",),
    "after_mlp": ("after_ffn",),
    "layer_output": ("after_ffn",),
    "residual_stream": (
        "before_attn",
        "attn_only",
        "after_attn",
        "ffn_only",
        "after_ffn",
    ),
}


class RepresentationDynamics:
    """Layer-wise modality-aware cosine changes in the decoder residual stream."""

    name = "representation_dynamics"
    version = "1.0.0"
    supported_metrics = {"cosine"}
    supported_collect_points = {
        "before_attn",
        "attn_only",
        "after_attn",
        "ffn_only",
        "after_ffn",
        # Existing config vocabulary retained as aliases for raw-node requests.
        "layer_input",
        "after_mlp",
        "layer_output",
        "residual_stream",
    }
    supported_token_groups = {
        "system",
        "text",
        "vision",
        "image",
        "video",
        "audio",
        "all",
    }

    def validate_config(self, config: dict[str, Any]) -> None:
        validate_common_probe_config(config, context="representation dynamics")
        metrics = set(
            string_sequence(
                config,
                "metrics",
                ("cosine",),
                context="representation dynamics",
            )
        )
        if not metrics or not metrics.issubset(self.supported_metrics):
            unknown = sorted(metrics - self.supported_metrics)
            raise ValueError(
                f"Unsupported representation-dynamics metrics: {unknown or sorted(metrics)}"
            )

        collect_points = set(
            string_sequence(
                config,
                "collect_points",
                (),
                context="representation dynamics",
                allow_empty=True,
            )
        )
        if not collect_points.issubset(self.supported_collect_points):
            unknown = sorted(collect_points - self.supported_collect_points)
            raise ValueError(
                f"Unsupported representation-dynamics collect points: {unknown}"
            )
        if collect_points and not _comparisons_for_collect_points(collect_points):
            raise ValueError(
                "representation dynamics collect_points must select both ends of "
                "at least one supported comparison."
            )

        token_groups = set(
            string_sequence(
                config,
                "token_groups",
                ("system", "text", "vision"),
                context="representation dynamics",
            )
        )
        if not token_groups or not token_groups.issubset(self.supported_token_groups):
            unknown = sorted(token_groups - self.supported_token_groups)
            raise ValueError(
                f"Unsupported representation-dynamics token groups: "
                f"{unknown or sorted(token_groups)}"
            )
        validate_layer_selection(
            config.get("layers", "all"),
            context="representation dynamics layers",
        )
        nonnegative_finite_number(
            config,
            "residual_atol",
            2e-3,
            context="representation dynamics",
        )
        nonnegative_finite_number(
            config,
            "residual_rtol",
            2e-3,
            context="representation dynamics",
        )

    def open(
        self,
        *,
        backbone: Any,
        runtime: Any,
        config: dict[str, Any],
        phase: str,
        output_dir: str,
    ) -> "_RepresentationDynamicsSession":
        self.validate_config(config)
        return _RepresentationDynamicsSession(
            plugin=self,
            backbone=backbone,
            runtime=runtime,
            config=config,
            phase=phase,
            output_dir=output_dir,
        )

@dataclass
class _RepresentationDynamicsSession:
    plugin: RepresentationDynamics
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
            context="Representation",
        )
        self.token_groups = list(
            dict.fromkeys(
                string_sequence(
                    self.config,
                    "token_groups",
                    ("system", "text", "vision"),
                    context="representation dynamics",
                )
            )
        )
        self.comparisons = _comparisons_for_collect_points(
            string_sequence(
                self.config,
                "collect_points",
                (),
                context="representation dynamics",
                allow_empty=True,
            )
        )
        self.records = StreamingAnalysisRecords(
            output_dir=self.output_dir,
            prefix=self.plugin.name,
            save_raw=bool(self.config.get("save_raw", False)),
            accumulator_factory=RepresentationSampleAccumulator,
            aggregate_by_task=True,
        )
        self._finished = False
        self._closed = False

    def process(self, sample: Mapping[str, Any]) -> None:
        if self._finished or self._closed:
            raise BackendIntegrationError(
                "Cannot process a sample after representation session completion."
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
            probe_name="Representation dynamics",
        )

        capture = ResidualNodeCapture(self.selected_layers)
        try:
            try:
                import torch
            except ImportError as exc:  # pragma: no cover - real runtime requires torch
                raise BackendIntegrationError(
                    "Representation dynamics requires PyTorch."
                ) from exc
            with HookSession() as hooks:
                capture.install(hooks)
                with torch.inference_mode():
                    output = self.backbone.runner.forward_train(self.runtime, state)
                    del output
                    nodes, residual_evidence = capture.finish(
                        atol=float(self.config.get("residual_atol", 2e-3)),
                        rtol=float(self.config.get("residual_rtol", 2e-3)),
                    )
                    sequence_length = int(
                        next(iter(nodes.values()))["before_attn"].shape[1]
                    )
                    layout = build_token_group_layout(
                        state,
                        sequence_length=sequence_length,
                        backbone_name=self.binding.backbone_name,
                        runtime=self.runtime,
                        layer_index=(
                            self.selected_layers[0].index
                            if len(self.selected_layers) == 1
                            else None
                        ),
                    )
                    layer_metrics = _sample_layer_metrics(
                        nodes,
                        layout=layout,
                        token_groups=self.token_groups,
                        comparisons=self.comparisons,
                    )
            return {
                "sample_id": sample["sample_id"],
                "task_name": sample["task_name"],
                "token_counts": layout.counts(),
                "token_layout_evidence": dict(layout.evidence),
                "residual_contract": residual_evidence,
                "layers": layer_metrics,
            }
        finally:
            capture.clear()

    def finish(self) -> AnalysisRecord:
        if self._finished:
            raise BackendIntegrationError("Representation session was already finished.")
        self._finished = True
        if self.records.successful_count == 0:
            reasons = "; ".join(
                str(item.get("reason"))
                for item in self.records.skipped_examples[:3]
            )
            raise BackendIntegrationError(
                "Representation dynamics produced no successful samples. " + reasons
            )
        aggregate = self.records.overall_result()
        aggregate.update(self.records.skipped_evidence())
        aggregate["streaming_aggregation"] = self.records.execution_evidence()
        aggregate["task_aggregates"] = self.records.task_results()
        aggregate["binding_evidence"] = dict(self.binding.evidence)
        aggregate["configured_layers"] = [item.index for item in self.selected_layers]
        aggregate["comparisons"] = list(self.comparisons)
        aggregate["token_groups"] = list(self.token_groups)
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
            filename_stem="representation_dynamics",
            records=self.records,
            aggregate=aggregate,
            summary={
                "layer_count": len(self.selected_layers),
                "finite_aggregate_values": _finite_aggregate_value_count(aggregate),
                "comparisons": list(self.comparisons),
                "token_groups": list(self.token_groups),
            },
        )

    def close(self) -> None:
        self._closed = True
        self.records.close()


def _sample_layer_metrics(
    nodes: Mapping[int, Mapping[str, Any]],
    *,
    layout: TokenGroupLayout,
    token_groups: list[str],
    comparisons: Mapping[str, tuple[str, str]],
) -> dict[str, dict]:
    try:
        import torch
        import torch.nn.functional as functional
    except ImportError as exc:  # pragma: no cover
        raise BackendIntegrationError("Representation metrics require PyTorch.") from exc
    result: dict[str, dict] = {}
    for layer, layer_nodes in nodes.items():
        comparison_results: dict[str, dict] = {}
        for comparison, (left_name, right_name) in comparisons.items():
            left = layer_nodes[left_name]
            right = layer_nodes[right_name]
            cosine = functional.cosine_similarity(
                left.float(), right.float(), dim=-1, eps=1e-12
            )
            if cosine.ndim != 2 or int(cosine.shape[0]) != 1:
                raise BackendIntegrationError(
                    "Representation cosine expects [1, sequence] values."
                )
            groups: dict[str, dict[str, Any]] = {}
            for group in token_groups:
                if group == "all":
                    positions = tuple(
                        index
                        for index, label in enumerate(layout.labels_by_sample[0])
                        if label != "padding"
                    )
                else:
                    positions = layout.positions(0, group)
                if not positions:
                    groups[group] = {
                        "cosine": None,
                        "token_count": 0,
                        "status": "absent",
                    }
                    continue
                indices = torch.tensor(
                    positions,
                    dtype=torch.long,
                    device=cosine.device,
                )
                selected = cosine[0].index_select(0, indices)
                value = float(selected.mean().detach().cpu().item())
                groups[group] = {
                    "cosine": value if math.isfinite(value) else None,
                    "token_count": len(positions),
                    "status": "finite" if math.isfinite(value) else "non_finite",
                }
            comparison_results[comparison] = groups
        result[str(layer)] = comparison_results
    return result


def _comparisons_for_collect_points(
    collect_points: Iterable[str],
) -> dict[str, tuple[str, str]]:
    requested = tuple(collect_points)
    if not requested:
        return dict(_COMPARISONS)
    canonical: set[str] = set()
    for point in requested:
        canonical.update(_COLLECT_POINT_ALIASES.get(point, (point,)))
    return {
        name: endpoints
        for name, endpoints in _COMPARISONS.items()
        if endpoints[0] in canonical and endpoints[1] in canonical
    }


def _finite_aggregate_value_count(aggregate: Mapping[str, Any]) -> int:
    count = 0
    for comparisons in aggregate.get("layers", {}).values():
        for groups in comparisons.values():
            for metrics in groups.values():
                value = metrics.get("cosine")
                if value is not None and math.isfinite(float(value)):
                    count += 1
    return count


__all__ = ["RepresentationDynamics"]
