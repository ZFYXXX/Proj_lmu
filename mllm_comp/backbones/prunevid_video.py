"""Installable PruneVid video executors for the three canonical backbones.

The existing video bridges own projector payload construction and native
prompt legalization.  This module composes the canonical projector kernel
with shared native forward/prefill pruning. The method retains its fixed
VTP scoring source and independently configured application boundary.
It never activates a support unit; real-checkpoint and train/reload evidence
remain separate registry gates.
"""

from __future__ import annotations

import inspect
from contextvars import ContextVar
from dataclasses import dataclass, replace
from typing import Any, Callable, Mapping, Sequence

from mllm_comp.backbones.video_decoder import VideoDecoderPruningExecutor
from mllm_comp.compressors.video.prunevid import PruneVidDecoderCall
from mllm_comp.backbones._video_bridge import (
    install_video_bridge,
    register_video_callback,
    release_video_callback,
)
from mllm_comp.compressors.video._rules import load_video_rules, video_rule_metadata
from mllm_comp.backbones._provider_patching import MethodPatch
from mllm_comp.backbones.image_training import attach_shifted_causal_loss
from mllm_comp.compressors.image.decoder_lifecycle import (
    DecoderBoundary,
    compact_dynamic_cache_prefix,
)
from mllm_comp.compressors.video import (
    PruneVidVTPResult,
    PruneVidVideoKernel,
    VideoCompressionResult,
    merge_prunevid_source_groups,
)
from mllm_comp.compressors.video.config import resolve_canonical_video_config
from mllm_comp.compressors.video.positions import video_decoder_positions, video_projector_position, video_projector_site
from mllm_comp.core.errors import BackendIntegrationError, InvariantError
from mllm_comp.core.layout import layout_delta_record
from mllm_comp.core.vocabulary import ExecutionPhase


_BACKBONES = ("llava_onevision", "qwen25_vl", "qwen3_vl")


@dataclass(frozen=True)
class _ActiveRequest:
    request: Any
    projector_result: VideoCompressionResult
    response: Any | None = None


class PruneVidVideoExecutor(VideoDecoderPruningExecutor):
    """Compound projector/decoder activation for one exact video binding."""

    algorithm = "prunevid"
    decoder_label = "PruneVid"
    variant = "prunevid.video"

    def __init__(
        self,
        *,
        backbone: str,
        model: Any,
        runtime: Any | None = None,
        facade: Any | None = None,
        stage: Any | None = None,
        binding: Any | None = None,
        binding_config: Any | None = None,
        contract_projection: Any | None = None,
        execution_phase: ExecutionPhase | str = ExecutionPhase.GENERATE,
    ) -> None:
        if backbone not in _BACKBONES:
            raise ValueError(f"Unknown PruneVid video backbone {backbone!r}")
        self.backbone = backbone
        self.model = model
        self.runtime = runtime
        self.facade = facade
        self.stage = stage
        self.binding = binding
        self.binding_config = binding_config
        self.execution_phase = ExecutionPhase(execution_phase)
        self.config = (
            contract_projection.algorithm_config
            if contract_projection is not None
            else _validated_config(stage, binding_config)
        )
        self.selected_layer = int(self.config["vtp_selected_layer"])
        self.alpha = float(self.config["vtp_alpha"])
        self.projector_position = video_projector_position(stage, backbone, target_name="temporal_spatial_reduce")
        self.projector_site = video_projector_site(backbone, self.projector_position)
        self.application_position = video_decoder_positions(stage, (self.selected_layer,), algorithm=self.algorithm, backbone=backbone)[0]
        self.kernel = PruneVidVideoKernel()
        self._rules, self._rule_specs = load_video_rules(stage, contract_projection=contract_projection)
        _validate_binding(binding, stage, backbone, self.execution_phase)

        self._active: ContextVar[_ActiveRequest | None] = ContextVar(
            f"prunevid_video_active_{backbone}_{id(self)}",
            default=None,
        )
        self._bridge: Any | None = None
        self._bridge_handle: Any | None = None
        self._owns_bridge = False
        self._patches: list[MethodPatch] = []
        self._installed = False
        self._projector_count = 0
        self._vtp_count = 0
        self._projector_records: list[dict[str, Any]] = []
        self._algorithm_records: list[dict[str, Any]] = []
        self._vtp_records: list[dict[str, Any]] = []
        self._cache_records: list[dict[str, Any]] = []

    @property
    def installed(self) -> bool:
        return self._installed

    def install(self) -> None:
        if self._installed:
            return
        patches: list[MethodPatch] = []
        bridge = None
        handle = None
        owns_bridge = False
        try:
            decoder = _decoder_owner(self.model, self.backbone)
            try:
                self.application_position = video_decoder_positions(
                    self.stage, (self.selected_layer,), len(decoder.layers), algorithm=self.algorithm, backbone=self.backbone)[0]
            except (TypeError, ValueError) as exc:
                raise BackendIntegrationError(str(exc)) from exc
            native_decoder_forward = getattr(decoder, "forward", None)
            if not callable(native_decoder_forward):
                raise BackendIntegrationError(
                    "PruneVid could not resolve the native decoder forward"
                )
            patches.append(
                MethodPatch.apply(
                    decoder,
                    "forward",
                    self._decoder_forward_replacement(native_decoder_forward),
                    native_decoder_forward,
                    label="PruneVid",
                )
            )

            native_model_forward = getattr(self.model, "forward", None)
            if not callable(native_model_forward):
                raise BackendIntegrationError(
                    "PruneVid could not resolve the native model forward"
                )
            patches.append(
                MethodPatch.apply(
                    self.model,
                    "forward",
                    self._model_forward_replacement(native_model_forward),
                    native_model_forward,
                    label="PruneVid",
                )
            )

            bridge, owns_bridge, projector_site = install_video_bridge(self.model, self.backbone)
            handle = register_video_callback(
                bridge,
                projector_site,
                self._projector_callback,
                label="PruneVid",
                **(
                    {"projector_position": "before", "preserve_projected_values": True}
                    if self.projector_position == "before"
                    else {}
                ),
            )
        except BaseException:
            release_video_callback(bridge, handle, owns_bridge)
            for patch in reversed(patches):
                try:
                    patch.restore()
                except Exception:
                    pass
            raise
        self._bridge = bridge
        self._bridge_handle = handle
        self._owns_bridge = owns_bridge
        self._patches = patches
        self._installed = True

    def uninstall(self) -> None:
        if not self._installed:
            return
        errors = release_video_callback(
            self._bridge, self._bridge_handle, self._owns_bridge
        )
        for patch in reversed(self._patches):
            try:
                patch.restore()
            except Exception as exc:
                errors.append(f"{type(exc).__name__}: {exc}")
        self._bridge_handle = None
        self._bridge = None
        self._owns_bridge = False
        self._patches.clear()
        self._active.set(None)
        self._installed = False
        if errors:
            raise BackendIntegrationError(f"PruneVid video uninstall failed: {errors}")

    def runtime_metadata(self) -> dict[str, Any]:
        bridge_metadata = (
            self._bridge.runtime_metadata()
            if self._bridge is not None
            and callable(getattr(self._bridge, "runtime_metadata", None))
            else None
        )
        return {
            "executor": f"{self.backbone}_prunevid_video",
            "algorithm": self.algorithm,
            "variant": self.variant,
            "backbone": self.backbone,
            "binding": getattr(self.binding, "name", None),
            "execution_phase": self.execution_phase.value,
            "installed": self._installed,
            "canonical_config": dict(self.config),
            "projector_position": self.projector_position,
            "target_site": self.projector_site,
            "application_positions": [self.application_position.as_dict()],
            "score_layer": self.selected_layer,
            "canonical_evidence_applicable": self.projector_position == "after" and self.selected_layer == 10
                and self.application_position.boundary == "inter_layer" and self.application_position.index == 11,
            "dense_projector_metric_source": self.projector_position == "before",
            "projector_invocations": self._projector_count,
            "vtp_invocations": self._vtp_count,
            "records_scope": "latest_video_request",
            "projector_layout_deltas": list(self._projector_records),
            "algorithm_records": list(self._algorithm_records),
            "vtp_layout_deltas": list(self._vtp_records),
            "cache_repairs": list(self._cache_records),
            "bridge": bridge_metadata,
            "compound_projector_llm": True,
            "training_gradients_preserved": True,
            "real_checkpoint_verified": False,
            "support_activation_claimed": False,
            **video_rule_metadata(self._rule_specs),
        }

    def clear_diagnostic_records(self) -> None:
        """Discard only diagnostic history; keep counters and replay state."""

        self._projector_records.clear()
        self._algorithm_records.clear()
        self._vtp_records.clear()
        self._cache_records.clear()

    def _projector_callback(self, request: Any) -> Any:
        self.clear_diagnostic_records()
        if self._active.get() is not None:
            raise BackendIntegrationError(
                "Nested PruneVid video projector requests are unsupported"
            )
        payload = getattr(request, "payload", None)
        reference = payload
        if self.projector_position == "before":
            projected = payload.signals.get("projector_reference_features")
            if projected is None:
                raise BackendIntegrationError("PruneVid projector input requires the original projected clustering reference.")
            reference = replace(payload, features=projected)
        result = self.kernel(reference, config=self.config, **({"rules": {k: v for k, v in self._rules.items() if self.projector_position != "before" or k != "aggregator"}} if self._rules else {}))
        if self.projector_position == "before":
            result = replace(result, features=merge_prunevid_source_groups(payload.features, result, **({"rules": self._rules} if self._rules else {})))
        result = replace(result, metadata={**result.metadata, "projector_position": self.projector_position,
            "target_site": self.projector_site, "clustering_source": "original_projected_features"})
        response = None
        if self.backbone == "qwen3_vl":
            from mllm_comp.backbones.qwen3_vl.video_compression import (
                Qwen3VLVideoCompressionResponse,
            )

            branches = tuple(
                merge_prunevid_source_groups(branch, result, **({"rules": self._rules} if self._rules else {}))
                for branch in request.deepstack_features
            )
            response = Qwen3VLVideoCompressionResponse.from_result(
                request,
                result,
                deepstack_features=branches,
            )
            returned = response
        else:
            returned = result
        self._active.set(_ActiveRequest(request, result, response))
        self._projector_count += 1
        record = layout_delta_record(result.layout_deltas[0])
        record.update(
            {
                "stage": "temporal_spatial_reduce",
                "projector_position": self.projector_position,
                "target_site": self.projector_site,
                "invocation_id": getattr(request, "invocation_id", None),
                "media_id": getattr(request, "media_id", None),
            }
        )
        self._projector_records.append(record)
        self._algorithm_records.append(
            _prunevid_invocation_record(request, result, response=response, scoring_payload=reference)
        )
        return returned

    def _model_forward_replacement(
        self,
        original: Callable[..., Any],
    ) -> Callable[..., Any]:
        signature = inspect.signature(original)

        def replacement(_model: Any, *args: Any, **kwargs: Any) -> Any:
            values = _bound_values(signature, args, kwargs)
            if not _is_video_call(values, self.backbone):
                return original(*args, **kwargs)
            if self._active.get() is not None:
                raise BackendIntegrationError(
                    "Nested PruneVid video model calls are unsupported"
                )
            try:
                if self.backbone != "llava_onevision":
                    return original(*args, **kwargs)
                bridge = self._bridge
                if bridge is None:
                    raise BackendIntegrationError(
                        "LLaVA PruneVid video bridge is not installed"
                    )
                labels = values.get("labels")
                if labels is not None and values.get("return_dict") is False:
                    raise BackendIntegrationError(
                        "LLaVA PruneVid training requires return_dict=True"
                    )
                with bridge.prunevid_decoder_context(
                    external_training_loss=labels is not None
                ):
                    output = original(*args, **kwargs)
                    if labels is None:
                        return output
                    repaired = bridge.prunevid_training_labels()
                    if repaired is None:
                        raise InvariantError(
                            "LLaVA PruneVid training labels were not produced"
                        )
                    return attach_shifted_causal_loss(
                        output,
                        repaired,
                        operation="LLaVA PruneVid video compression",
                    )
            finally:
                self._active.set(None)

        return replacement

    def _record_vtp(
        self,
        result: PruneVidVTPResult,
        cache_record: Mapping[str, Any],
    ) -> None:
        self._vtp_count += 1
        visual = layout_delta_record(result.visual_layout_delta)
        visual.update(
            {
                "stage": "vtp_prefill_prune",
                "selected_layer": result.selected_layer,
                "score_layer": result.selected_layer,
                "application_position": self.application_position.as_dict(),
                "alpha": result.alpha,
                "kept_visual_indices": list(result.kept_visual_indices),
                "dropped_visual_indices": list(result.dropped_visual_indices),
                "sequence_layout_delta": layout_delta_record(
                    result.sequence_layout_delta
                ),
                "attention_reduction": result.metadata.get("attention_reduction"),
                "selection": result.metadata.get("selection"),
                "cache_policy": result.metadata.get("cache_policy"),
                "position_policy": result.metadata.get("position_policy"),
                "cache": dict(cache_record),
            }
        )
        self._vtp_records.append(visual)
        self._cache_records.append(dict(cache_record))

    def _new_decoder_call(self, projector_result, sequence_length, num_layers):
        try:
            positions = video_decoder_positions(
                self.stage,
                (self.selected_layer,),
                num_layers,
                algorithm=self.algorithm,
                backbone=self.backbone,
            )
        except (TypeError, ValueError) as exc:
            raise BackendIntegrationError(str(exc)) from exc
        self.application_position = positions[0]
        return PruneVidDecoderCall(
            projector_result,
            positions=positions,
            selected_layer=self.selected_layer,
            alpha=self.alpha,
            rules=self._rules,
        )

    def _publish_decoder_pruning(
        self, result, cache_record, point, *, deepstack_branch_count=0
    ):
        kwargs = (
            {
                "deepstack_consumed_branch_count": min(
                    point.index, deepstack_branch_count
                )
            }
            if self.backbone == "qwen3_vl"
            else {}
        )
        self._bridge.apply_prunevid_vtp_layout(result, **kwargs)
        self._record_vtp(result, cache_record)

    def _repair_decoder_cache(
        self,
        past_key_values: Any,
        result: PruneVidVTPResult,
        *,
        use_cache: bool,
        position: Any | None = None,
    ) -> Mapping[str, Any]:
        if not use_cache:
            return {
                "policy": "none_training_forward",
                "source_length": result.sequence_layout_delta.source_count,
                "result_length": result.sequence_layout_delta.result_count,
                "touched_layers": [],
            }
        if (
            position is not None
            and position.boundary == "inter_layer"
            and position.index == result.selected_layer + 1
        ):
            position = None  # Preserve the original default cache telemetry.
        telemetry = compact_dynamic_cache_prefix(
            past_key_values,
            result.kept_sequence_indices,
            source_length=result.sequence_layout_delta.source_count,
            result_length=result.sequence_layout_delta.result_count,
            layer_count=(
                result.selected_layer + 1
                if position is None
                else position.index + int(position.boundary == "post_attention_pre_ffn")
            ),
            boundary_layer=(
                result.selected_layer if position is None else position.index
            ),
            boundary=(
                DecoderBoundary.AFTER_LAYER
                if position is None
                else (
                    DecoderBoundary.POST_ATTENTION_PRE_FFN
                    if position.boundary == "post_attention_pre_ffn"
                    else DecoderBoundary.BEFORE_LAYER
                )
            ),
        )
        return telemetry.as_dict()


def create_prunevid_video_executor(
    *,
    backbone: str,
    **kwargs: Any,
) -> PruneVidVideoExecutor:
    return PruneVidVideoExecutor(backbone=backbone, **kwargs)


def _prunevid_invocation_record(
    request: Any,
    result: Any,
    *,
    response: Any | None = None,
    scoring_payload: Any | None = None,
) -> dict[str, Any]:
    """Record projector decisions and branch shapes without replaying values."""

    # Scores may use a projected reference while the real request and its
    # synchronized branches hold pre-projector values of another width.
    payload = getattr(request, "payload", None) if scoring_payload is None else scoring_payload
    features = getattr(payload, "features", None)
    result_features = getattr(result, "features", None)
    source_count = int(getattr(result, "source_count", 0))
    result_count = int(getattr(result, "result_count", 0))
    if (
        getattr(features, "ndim", None) != 2
        or getattr(result_features, "ndim", None) != 2
        or int(features.shape[0]) != source_count
        or int(result_features.shape[0]) != result_count
        or source_count < 1
        or result_count < 1
    ):
        raise BackendIntegrationError(
            "PruneVid invocation evidence lost its projected feature layout"
        )
    torch = _torch()
    projected_finite = bool(torch.isfinite(features).all().item())
    result_finite = bool(torch.isfinite(result_features).all().item())
    if not projected_finite or not result_finite:
        raise BackendIntegrationError(
            "PruneVid invocation evidence contains non-finite projected features"
        )

    if int(getattr(result, "video_count", 0)) != 1:
        raise BackendIntegrationError(
            "PruneVid invocation evidence requires exactly one video"
        )
    coordinates = tuple(getattr(payload, "coordinates", ()))
    if len(coordinates) != source_count:
        raise BackendIntegrationError(
            "PruneVid invocation evidence lost its source coordinates"
        )
    temporal_indices = tuple(int(getattr(item, "t")) for item in coordinates)
    frame_ids = tuple(sorted(set(temporal_indices)))
    if not frame_ids or frame_ids != tuple(range(len(frame_ids))):
        raise BackendIntegrationError(
            "PruneVid invocation evidence requires dense temporal indices"
        )
    frame_count = len(frame_ids)
    if source_count % frame_count:
        raise BackendIntegrationError(
            "PruneVid invocation evidence has a non-rectangular frame layout"
        )
    tokens_per_frame = source_count // frame_count
    grid_height = max(int(getattr(item, "h")) for item in coordinates) + 1
    grid_width = max(int(getattr(item, "w")) for item in coordinates) + 1
    expected_raster = {
        (row, column) for row in range(grid_height) for column in range(grid_width)
    }
    if tokens_per_frame != grid_height * grid_width or any(
        {
            (int(getattr(item, "h")), int(getattr(item, "w")))
            for item in coordinates
            if int(getattr(item, "t")) == frame
        }
        != expected_raster
        for frame in frame_ids
    ):
        raise BackendIntegrationError(
            "PruneVid invocation evidence requires one dense rectangular raster per frame"
        )

    metadata = getattr(result, "metadata", {})
    windows_by_video = metadata.get("prunevid_windows")
    groups_by_video = getattr(result, "source_groups", ())
    representatives_by_video = getattr(result, "representative_indices", ())
    if (
        not isinstance(windows_by_video, tuple)
        or len(windows_by_video) != 1
        or not isinstance(groups_by_video, tuple)
        or len(groups_by_video) != 1
        or not isinstance(representatives_by_video, tuple)
        or len(representatives_by_video) != 1
    ):
        raise BackendIntegrationError(
            "PruneVid invocation evidence lost its single-video projector decisions"
        )
    groups = tuple(tuple(int(item) for item in group) for group in groups_by_video[0])
    representatives = tuple(int(item) for item in representatives_by_video[0])
    if len(groups) != result_count or len(representatives) != result_count:
        raise BackendIntegrationError(
            "PruneVid invocation evidence result groups do not match its token count"
        )
    flattened_sources = tuple(source for group in groups for source in group)
    source_partition_exact = len(flattened_sources) == source_count and tuple(
        sorted(flattened_sources)
    ) == tuple(range(source_count))
    representatives_source_ordered = representatives == tuple(
        sorted(set(representatives))
    )
    representatives_inside_groups = all(
        representative in group
        for representative, group in zip(representatives, groups)
    )
    if not (
        source_partition_exact
        and representatives_source_ordered
        and representatives_inside_groups
    ):
        raise BackendIntegrationError(
            "PruneVid invocation evidence lost source partition or causal anchors"
        )

    alpha = float(metadata.get("canonical_config", {}).get("vtp_alpha", 0.0))
    expected_frame_start = 0
    owned_result_indices: list[int] = []
    window_summaries: list[dict[str, Any]] = []
    static_output_count = 0
    dynamic_output_count = 0
    expected_vtp_keep_count = 0
    static_group_frame_coverage_exact = True
    dynamic_group_frame_ownership_exact = True
    raw_windows = tuple(windows_by_video[0])
    for expected_window_index, raw in enumerate(raw_windows):
        if not isinstance(raw, Mapping):
            raise BackendIntegrationError("PruneVid window record changed shape")
        window_index = int(raw.get("window_index", -1))
        source_frame_start = int(raw.get("source_frame_start", -1))
        window_frame_count = int(raw.get("frame_count", 0))
        static_indices = tuple(
            int(item) for item in raw.get("static_result_indices", ())
        )
        dynamic_by_frame = tuple(
            tuple(int(item) for item in row)
            for row in raw.get("dynamic_result_indices_by_frame", ())
        )
        if (
            window_index != expected_window_index
            or source_frame_start != expected_frame_start
            or window_frame_count < 1
            or len(dynamic_by_frame) != window_frame_count
        ):
            raise BackendIntegrationError(
                "PruneVid temporal windows do not preserve contiguous frame ownership"
            )
        result_indices = (
            *static_indices,
            *(item for row in dynamic_by_frame for item in row),
        )
        if len(set(result_indices)) != len(result_indices) or any(
            not 0 <= item < result_count for item in result_indices
        ):
            raise BackendIntegrationError(
                "PruneVid window outputs are duplicated or outside the result layout"
            )
        expected_frames = set(
            range(source_frame_start, source_frame_start + window_frame_count)
        )
        static_group_frame_coverage_exact = static_group_frame_coverage_exact and all(
            {source // tokens_per_frame for source in groups[item]} == expected_frames
            for item in static_indices
        )
        dynamic_group_frame_ownership_exact = (
            dynamic_group_frame_ownership_exact
            and all(
                {source // tokens_per_frame for source in groups[result_index]}
                == {source_frame_start + frame_offset}
                for frame_offset, row in enumerate(dynamic_by_frame)
                for result_index in row
            )
        )
        static_count = len(static_indices)
        dynamic_counts = tuple(len(row) for row in dynamic_by_frame)
        static_keep_count = int(static_count * alpha)
        dynamic_keep_counts = tuple(int(count * alpha) for count in dynamic_counts)
        expected_vtp_keep_count += static_keep_count + sum(dynamic_keep_counts)
        static_output_count += static_count
        dynamic_output_count += sum(dynamic_counts)
        owned_result_indices.extend(result_indices)
        window_summaries.append(
            {
                "window_index": window_index,
                "source_frame_start": source_frame_start,
                "frame_count": window_frame_count,
                "static_result_count": static_count,
                "dynamic_result_counts_by_frame": dynamic_counts,
                "static_result_indices": static_indices,
                "dynamic_result_indices_by_frame": dynamic_by_frame,
                "expected_vtp_static_keep_count": static_keep_count,
                "expected_vtp_dynamic_keep_counts_by_frame": dynamic_keep_counts,
            }
        )
        expected_frame_start += window_frame_count
    result_window_partition_exact = tuple(sorted(owned_result_indices)) == tuple(
        range(result_count)
    )
    window_frame_coverage_exact = expected_frame_start == frame_count
    if not (
        raw_windows
        and result_window_partition_exact
        and window_frame_coverage_exact
        and static_group_frame_coverage_exact
        and dynamic_group_frame_ownership_exact
        and expected_vtp_keep_count > 0
    ):
        raise BackendIntegrationError(
            "PruneVid invocation evidence lost its static/dynamic window semantics"
        )

    anchor_legalizations = metadata.get("anchor_legalizations_by_video")
    if not isinstance(anchor_legalizations, tuple) or len(anchor_legalizations) != 1:
        raise BackendIntegrationError(
            "PruneVid invocation evidence lost anchor legalization telemetry"
        )
    canonical_config = dict(metadata.get("canonical_config", {}))
    source_deepstack = tuple(getattr(request, "deepstack_features", ()))
    result_deepstack = tuple(getattr(response, "deepstack_features", ()))
    deepstack_source_shapes = tuple(tuple(branch.shape) for branch in source_deepstack)
    deepstack_result_shapes = tuple(tuple(branch.shape) for branch in result_deepstack)

    return {
        "invocation_id": getattr(request, "invocation_id", None),
        "media_id": getattr(request, "media_id", None),
        "source_count": source_count,
        "result_count": result_count,
        "canonical_config": canonical_config,
        "projected_features_shape": tuple(int(item) for item in features.shape),
        "result_features_shape": tuple(int(item) for item in result_features.shape),
        "projected_features_finite": projected_finite,
        "result_features_finite": result_finite,
        "frame_count": frame_count,
        "grid_height": grid_height,
        "grid_width": grid_width,
        "tokens_per_frame": tokens_per_frame,
        "temporal_cluster_target": max(
            1, int(frame_count * float(canonical_config["temporal_ratio"]))
        ),
        "window_count": len(window_summaries),
        "windows": tuple(window_summaries),
        "window_frame_coverage_exact": window_frame_coverage_exact,
        "result_window_partition_exact": result_window_partition_exact,
        "static_output_count": static_output_count,
        "dynamic_output_count": dynamic_output_count,
        "static_group_frame_coverage_exact": static_group_frame_coverage_exact,
        "dynamic_group_frame_ownership_exact": dynamic_group_frame_ownership_exact,
        "expected_vtp_keep_count": expected_vtp_keep_count,
        "source_partition_exact": source_partition_exact,
        "representative_indices": representatives,
        "representatives_source_ordered": representatives_source_ordered,
        "representatives_inside_groups": representatives_inside_groups,
        "anchor_legalization_count": int(anchor_legalizations[0]),
        "source_group_count": len(groups),
        "non_singleton_source_group_count": sum(len(group) > 1 for group in groups),
        "max_source_group_size": max(len(group) for group in groups),
        "projector_stage": metadata.get("projector_stage"),
        "vtp_required": bool(metadata.get("vtp_required")),
        "temporal_clustering": "deterministic_frame_mean_dpc_knn_contiguous_refinement",
        "static_dynamic_partition": "window_mean_cosine_gt_tau",
        "spatial_clustering": "deterministic_dpc_knn_floor_cluster_ratio",
        "temporal_tie_policy": metadata.get("temporal_tie_policy"),
        "output_order": metadata.get("output_order"),
        "representative_policy": metadata.get("representative_policy"),
        "deepstack_branch_count": len(source_deepstack),
        "deepstack_source_shapes": deepstack_source_shapes,
        "deepstack_result_shapes": deepstack_result_shapes,
    }


def _validated_config(stage: Any, binding_config: Any) -> Mapping[str, Any]:
    stage_config = getattr(stage, "config", {}) if stage is not None else {}
    if stage_config is None:
        stage_config = {}
    if not isinstance(stage_config, Mapping):
        raise BackendIntegrationError("PruneVid stage config must be a mapping")
    binding_values = (
        getattr(binding_config, "config", {}) if binding_config is not None else {}
    )
    if binding_values not in ({}, None):
        raise BackendIntegrationError(
            "PruneVid video binding does not accept binding-local config"
        )
    return resolve_canonical_video_config("prunevid", stage_config)


def _validate_binding(
    binding: Any,
    stage: Any,
    backbone: str,
    phase: ExecutionPhase,
) -> None:
    expected = f"prunevid.video.{backbone}"
    if binding is not None:
        if getattr(binding, "name", None) != expected:
            raise BackendIntegrationError(
                f"PruneVid executor requires binding {expected!r}"
            )
        if phase not in getattr(binding, "supported_phases", ()):
            raise BackendIntegrationError(
                f"PruneVid binding {expected!r} does not support {phase.value}"
            )
        targets = tuple(
            getattr(item, "name", None) for item in binding.ordered_targets()
        )
        if targets != ("temporal_spatial_reduce", "vtp_prefill_prune"):
            raise BackendIntegrationError("PruneVid compound target order changed")
    if stage is not None:
        if getattr(stage, "algorithm", None) != "prunevid":
            raise BackendIntegrationError(
                "PruneVid executor received another algorithm"
            )
        if getattr(stage, "variant", None) != "prunevid.video":
            raise BackendIntegrationError("PruneVid executor received another variant")


def _decoder_owner(model: Any, backbone: str) -> Any:
    if backbone == "llava_onevision":
        from mllm_comp.backbones.llava_onevision.provider_bridge import (
            _resolve_components,
        )

        return _resolve_components(model).core
    core = getattr(model, "model", None)
    owner = getattr(core, "language_model", None)
    if owner is None:
        raise BackendIntegrationError(
            f"{backbone} PruneVid could not resolve language_model"
        )
    return owner


def _bound_values(
    signature: inspect.Signature,
    args: Sequence[Any],
    kwargs: Mapping[str, Any],
) -> dict[str, Any]:
    bound = signature.bind_partial(*args, **kwargs)
    values = dict(bound.arguments)
    extras = values.pop("kwargs", None)
    if isinstance(extras, Mapping):
        values.update(extras)
    return values


def _is_video_call(values: Mapping[str, Any], backbone: str) -> bool:
    if backbone == "llava_onevision":
        modalities = values.get("modalities")
        if isinstance(modalities, str):
            return modalities == "video"
        if isinstance(modalities, Sequence):
            return tuple(modalities) == ("video",)
        return False
    return any(
        values.get(name) is not None
        for name in (
            "pixel_values_videos",
            "video_grid_thw",
            "second_per_grid_ts",
        )
    )


def _torch() -> Any:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - dependency-light control plane
        raise BackendIntegrationError(
            "PruneVid provider execution requires the existing PyTorch environment"
        ) from exc
    return torch


__all__ = [
    "PruneVidVideoExecutor",
    "create_prunevid_video_executor",
]
