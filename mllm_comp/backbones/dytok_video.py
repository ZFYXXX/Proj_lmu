"""DyToK detection capability and independent three-backbone executor."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

from mllm_comp.backbones._qwen_video_inputs import qwen_video_pixels_to_siglip
from mllm_comp.backbones._video_bridge import (
    install_video_bridge,
    register_video_callback,
    release_video_callback,
)
from mllm_comp.compressors.video._rules import load_video_rules, video_rule_metadata
from mllm_comp.compressors.video import (
    DyToKVideoKernel,
    VideoTokenPayload,
    merge_visionzip_source_groups,
)
from mllm_comp.compressors.video.config import resolve_canonical_video_config
from mllm_comp.compressors.video.positions import video_projector_position, video_projector_site, video_detection_layers
from mllm_comp.core.errors import BackendIntegrationError
from mllm_comp.core.layout import layout_delta_record
from mllm_comp.core.vocabulary import ExecutionPhase

from .visionzip_video import (
    VisionZipVideoSignalProvider,
)


_BACKBONES = ("llava_onevision", "qwen25_vl", "qwen3_vl")
_PROVIDER_ATTRIBUTE = "_mllm_comp_dytok_video_signal_provider"


@dataclass
class _DyToKCapture:
    visionzip_capture: Any
    frames: Any | None = None
    grid_thw: Any | None = None


class DyToKVideoSignalProvider:
    """Combine VisionZip encoder signals with one frozen detector pass."""

    def __init__(
        self,
        *,
        backbone: str,
        model: Any,
        detection_model_path: str | Path | None = None,
        detection_siglip_path: str | Path | None = None,
        target_layers: tuple[int, ...] = tuple(range(16, 24)),
        detection_device_map: Any = "cpu",
        detection_dtype: Any = "bfloat16",
        detection_runner: Any | None = None,
        visionzip_provider: Any | None = None,
    ) -> None:
        if backbone not in _BACKBONES:
            raise ValueError(f"Unknown DyToK video backbone {backbone!r}")
        self.backbone = backbone
        self.model = model
        self.detection_model_path = (
            None if detection_model_path is None else Path(detection_model_path)
        )
        self.detection_siglip_path = (
            None if detection_siglip_path is None else Path(detection_siglip_path)
        )
        self.target_layers = video_detection_layers(None, target_layers)
        self.detection_device_map = detection_device_map
        self.detection_dtype = detection_dtype
        self._detection_runner = detection_runner
        self._visionzip = visionzip_provider or VisionZipVideoSignalProvider(
            backbone=backbone,
            model=model,
        )
        self._installed = False
        self._capture_count = 0

    @property
    def installed(self) -> bool:
        return self._installed

    def prepare(self) -> None:
        if self._detection_runner is None:
            if self.detection_model_path is None or self.detection_siglip_path is None:
                raise BackendIntegrationError(
                    "DyToK requires local dytok_detection_model_path and "
                    "dytok_detection_siglip_path runtime metadata; downloads are "
                    "forbidden"
                )
            from mllm_comp.backbones._dytok_torch import FrozenDyToKDetector

            self._detection_runner = FrozenDyToKDetector(
                self.detection_model_path,
                siglip_path=self.detection_siglip_path,
                target_layers=self.target_layers,
                device_map=self.detection_device_map,
                torch_dtype=self.detection_dtype,
                owner_model=self.model,
            )
        if not callable(getattr(self._detection_runner, "frame_importance", None)):
            raise BackendIntegrationError(
                "DyToK detection runner lacks frame_importance(text, frames)"
            )
        configure = getattr(self._detection_runner, "configure_target_layers", None)
        if callable(configure):
            configure(self.target_layers)
        elif self.target_layers != tuple(range(16, 24)):
            raise BackendIntegrationError(
                "Injected DyToK detector does not support changing capture layers."
            )
        prepare = getattr(self._detection_runner, "prepare", None)
        if callable(prepare):
            prepare()

    def install(self) -> None:
        if self._installed:
            return
        self.prepare()
        existing = getattr(self.model, _PROVIDER_ATTRIBUTE, None)
        if existing is not None and existing is not self:
            raise BackendIntegrationError(
                "A different DyToK video signal provider is already installed"
            )
        self._visionzip.install()
        try:
            setattr(self.model, _PROVIDER_ATTRIBUTE, self)
        except BaseException:
            self._visionzip.uninstall()
            raise
        self._installed = True

    def uninstall(self) -> None:
        if not self._installed:
            return
        errors: list[str] = []
        if getattr(self.model, _PROVIDER_ATTRIBUTE, None) is not self:
            errors.append("provider ownership changed while active")
        elif _PROVIDER_ATTRIBUTE in vars(self.model):
            delattr(self.model, _PROVIDER_ATTRIBUTE)
        try:
            self._visionzip.uninstall()
        except Exception as exc:
            errors.append(f"{type(exc).__name__}: {exc}")
        self._installed = False
        if errors:
            raise BackendIntegrationError(
                f"DyToK video signal-provider uninstall failed: {errors}"
            )

    @contextmanager
    def capture_context(
        self,
        *,
        frames: Any | None = None,
        grid_thw: Any | None = None,
    ) -> Iterator[_DyToKCapture]:
        if not self._installed:
            raise BackendIntegrationError("DyToK signal provider is not installed")
        with self._visionzip.capture_context(
            frames=frames,
            grid_thw=grid_thw,
        ) as capture:
            yield _DyToKCapture(
                visionzip_capture=capture,
                frames=frames,
                grid_thw=grid_thw,
            )

    def payload_signals(
        self,
        capture: _DyToKCapture,
        *,
        token_count: int,
        coordinates: Sequence[Any],
    ) -> Mapping[str, Any]:
        if not isinstance(capture, _DyToKCapture):
            raise TypeError("DyToK received an unknown signal capture")
        return self._visionzip.payload_signals(
            capture.visionzip_capture,
            token_count=token_count,
            coordinates=coordinates,
        )

    def enrich_payload(
        self,
        payload: VideoTokenPayload,
        capture: _DyToKCapture,
    ) -> VideoTokenPayload:
        torch = _torch()
        if not isinstance(payload, VideoTokenPayload):
            raise TypeError("payload must be a VideoTokenPayload")
        if payload.video_count != 1:
            raise BackendIntegrationError("DyToK provider requires exactly one video")
        text = payload.signals.get("text_context")
        if not isinstance(text, str) or not text.strip():
            raise BackendIntegrationError(
                "DyToK requires user/system prompt text without the assistant answer"
            )
        frame_count = _projected_frame_count(payload)
        if self.backbone == "llava_onevision":
            frames = capture.frames
            if (
                getattr(frames, "ndim", None) != 4
                or int(frames.shape[0]) != frame_count
            ):
                raise BackendIntegrationError(
                    "LLaVA DyToK detector frames do not match projected time slices"
                )
            frame_source = "native_llava_siglip_pixels"
        else:
            pixel_key = (
                "qwen25_vl.pixel_values_videos"
                if self.backbone == "qwen25_vl"
                else "qwen3_vl.pixel_values_videos"
            )
            frames = self._qwen_pixels_to_siglip(
                payload.signals.get(pixel_key),
                capture.grid_thw,
            )
            if int(frames.shape[0]) != frame_count:
                raise BackendIntegrationError(
                    "Qwen DyToK detector frames do not match projected tubelets"
                )
            frame_source = "qwen_patchified_video_inverse_siglip_384"
        self.prepare()
        weights = self._detection_runner.frame_importance(text.strip(), frames)
        if not isinstance(weights, torch.Tensor) or weights.ndim != 1:
            raise BackendIntegrationError(
                "DyToK detector must return one rank-one frame-importance tensor"
            )
        if int(weights.numel()) != frame_count:
            raise BackendIntegrationError(
                "DyToK detector frame importance differs from projected time slices"
            )
        weights = weights.detach().to(
            device=payload.features.device,
            dtype=payload.features.dtype,
        )
        if not bool(torch.isfinite(weights).all().item()) or bool(
            (weights < 0).any().item()
        ):
            raise ValueError(
                "DyToK detector frame importance must be finite and non-negative"
            )
        total = weights.sum()
        if not bool(torch.isfinite(total).item()) or float(total.item()) <= 0:
            raise ValueError("DyToK detector frame importance has no positive mass")
        weights = weights / total
        self._capture_count += 1
        return VideoTokenPayload(
            features=payload.features,
            media_ids=payload.media_ids,
            video_offsets=payload.video_offsets,
            coordinates=payload.coordinates,
            raw_features=payload.raw_features,
            raw_video_offsets=payload.raw_video_offsets,
            raw_coordinates=payload.raw_coordinates,
            token_mask=payload.token_mask,
            signals={
                **dict(payload.signals),
                "dytok.frame_importance_by_video": (weights,),
                "detection_attention": {
                    "target_layers": self.target_layers,
                    "query": "last_prompt_token",
                    "keys": "visual_patch_tokens",
                    "layer_aggregation": "mean_then_normalize",
                },
                "prompt_layout": {
                    "single_visual_placeholder": True,
                    "assistant_answer_excluded": True,
                },
            },
            metadata={
                **dict(payload.metadata),
                "dytok_detection_frame_source": frame_source,
                "dytok_detection_model_frozen": True,
                "dytok_detection_local_only": True,
                "dytok_conditioning_excludes_assistant_answer": True,
            },
        )

    def runtime_metadata(self) -> dict[str, Any]:
        detector_metadata = None
        if self._detection_runner is not None:
            describe = getattr(self._detection_runner, "runtime_metadata", None)
            detector_metadata = (
                describe()
                if callable(describe)
                else {"provider": type(self._detection_runner).__name__}
            )
        return {
            "provider": f"{self.backbone}_dytok_video_signals",
            "installed": self._installed,
            "capture_count": self._capture_count,
            "video_only": True,
            "target_layers": self.target_layers,
            "detection_model_path": (
                None
                if self.detection_model_path is None
                else str(self.detection_model_path)
            ),
            "detection_siglip_path": (
                None
                if self.detection_siglip_path is None
                else str(self.detection_siglip_path)
            ),
            "detector": detector_metadata,
            "visionzip_signals": self._visionzip.runtime_metadata(),
            "detector_frozen": True,
            "detector_local_only": True,
            "assistant_answer_excluded": True,
            "real_checkpoint_verified": False,
        }

    def _qwen_pixels_to_siglip(self, values: Any, grid: Any) -> Any:
        torch = _torch()
        if not isinstance(values, torch.Tensor) or values.ndim != 2:
            raise BackendIntegrationError(
                "Qwen DyToK requires native rank-two pixel_values_videos"
            )
        if getattr(grid, "ndim", None) != 2 or int(grid.shape[0]) != 1:
            raise BackendIntegrationError(
                "Qwen DyToK requires exactly one video_grid_thw row"
            )
        visual = getattr(getattr(self.model, "model", None), "visual", None)
        config = getattr(visual, "config", None)
        patch_embed = getattr(visual, "patch_embed", None)
        patch_size = _first_positive_integer(
            getattr(patch_embed, "patch_size", None),
            getattr(config, "patch_size", None),
        )
        temporal_patch = _first_positive_integer(
            getattr(patch_embed, "temporal_patch_size", None),
            getattr(config, "temporal_patch_size", None),
        )
        merge_size = _first_positive_integer(
            getattr(visual, "spatial_merge_size", None),
            getattr(config, "spatial_merge_size", None),
        )
        return qwen_video_pixels_to_siglip(
            values,
            grid,
            backbone=self.backbone,
            patch_size=patch_size,
            temporal_patch=temporal_patch,
            merge_size=merge_size,
            method_name="DyToK",
        )


class DyToKVideoExecutor:
    """Run the detector provider before dynamic VisionZip at one splice."""

    algorithm = "dytok"
    variant = "dytok.video"

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
        signal_provider: Any | None = None,
        detection_runner: Any | None = None,
    ) -> None:
        if backbone not in _BACKBONES:
            raise ValueError(f"Unknown DyToK video backbone {backbone!r}")
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
        self.projector_position = video_projector_position(stage, backbone, target_name="dynamic_visionzip_reduce")
        self.detection_layers = video_detection_layers(stage, self.config["llm_config"]["target_layers"])
        _validate_binding(binding, stage, backbone, self.execution_phase)
        runtime_metadata = (
            getattr(runtime, "metadata", {}) if runtime is not None else {}
        )
        runtime_metadata = (
            runtime_metadata if isinstance(runtime_metadata, Mapping) else {}
        )
        detection_path = runtime_metadata.get("dytok_detection_model_path")
        siglip_path = runtime_metadata.get(
            "dytok_detection_siglip_path",
            runtime_metadata.get("metok_siglip_path"),
        )
        device_map = runtime_metadata.get("dytok_detection_device_map")
        if device_map is None:
            runtime_device = (
                getattr(runtime, "device", None) if runtime is not None else None
            )
            device_map = "cpu" if runtime_device is None else str(runtime_device)
        detection_dtype = runtime_metadata.get(
            "dytok_detection_dtype",
            getattr(runtime, "dtype", None) if runtime is not None else None,
        )
        if detection_dtype is None:
            detection_dtype = "bfloat16"
        target_layers = self.detection_layers
        self.kernel = DyToKVideoKernel()
        self._rules, self._rule_specs = load_video_rules(stage, contract_projection=contract_projection)
        self.signal_provider = signal_provider or DyToKVideoSignalProvider(
            backbone=backbone,
            model=model,
            detection_model_path=detection_path,
            detection_siglip_path=siglip_path,
            target_layers=target_layers,
            detection_device_map=device_map,
            detection_dtype=detection_dtype,
            detection_runner=detection_runner,
        )
        if isinstance(self.signal_provider, DyToKVideoSignalProvider):
            self.signal_provider.target_layers = self.detection_layers
        elif self.detection_layers != tuple(self.config["llm_config"]["target_layers"]):
            raise BackendIntegrationError(
                "Injected DyToK signal provider does not support changing detector layers."
            )
        self._bridge: Any | None = None
        self._bridge_handle: Any | None = None
        self._owns_bridge = False
        self._installed = False
        self._invocations = 0
        self._layout_records: list[dict[str, Any]] = []
        self._algorithm_records: list[dict[str, Any]] = []

    @property
    def installed(self) -> bool:
        return self._installed

    def install(self) -> None:
        if self._installed:
            return
        bridge = None
        handle = None
        owns_bridge = False
        try:
            self.signal_provider.install()
            bridge, owns_bridge, projector_site = install_video_bridge(self.model, self.backbone)
            handle = register_video_callback(
                bridge,
                projector_site,
                self._projector_callback,
                label="DyToK",
                signal_provider=self.signal_provider,
                **(
                    {"projector_position": "before"}
                    if self.projector_position == "before"
                    else {}
                ),
            )
        except BaseException:
            release_video_callback(bridge, handle, owns_bridge)
            try:
                self.signal_provider.uninstall()
            except Exception:
                pass
            raise
        self._bridge = bridge
        self._bridge_handle = handle
        self._owns_bridge = owns_bridge
        self._installed = True

    def uninstall(self) -> None:
        if not self._installed:
            return
        errors = release_video_callback(
            self._bridge, self._bridge_handle, self._owns_bridge
        )
        try:
            self.signal_provider.uninstall()
        except Exception as exc:
            errors.append(f"{type(exc).__name__}: {exc}")
        self._bridge_handle = None
        self._bridge = None
        self._owns_bridge = False
        self._installed = False
        if errors:
            raise BackendIntegrationError(f"DyToK video uninstall failed: {errors}")

    def runtime_metadata(self) -> dict[str, Any]:
        bridge_metadata = (
            self._bridge.runtime_metadata()
            if self._bridge is not None
            and callable(getattr(self._bridge, "runtime_metadata", None))
            else None
        )
        return {
            "executor": f"{self.backbone}_dytok_video",
            "algorithm": self.algorithm,
            "variant": self.variant,
            "backbone": self.backbone,
            "binding": getattr(self.binding, "name", None),
            "execution_phase": self.execution_phase.value,
            "installed": self._installed,
            "canonical_config": _plain_runtime_value(self.config),
            "projector_position": self.projector_position,
            "target_site": video_projector_site(self.backbone, self.projector_position),
            "detection_layers": self.detection_layers,
            "canonical_evidence_applicable": self.projector_position == "after"
                and self.detection_layers == tuple(self.config["llm_config"]["target_layers"]),
            "projector_invocations": self._invocations,
            "records_scope": "latest_video_request",
            "projector_layout_deltas": list(self._layout_records),
            "algorithm_records": list(self._algorithm_records),
            "signal_provider": self.signal_provider.runtime_metadata(),
            "bridge": bridge_metadata,
            "stages": ("detection_budget_provider", "dynamic_visionzip_reduce"),
            "base_budget_per_temporal_slice": 147,
            "upper_limit": 196,
            "detector_registered_as_trainable_child": False,
            "training_gradients_preserved": True,
            "image_variant_shared": False,
            "real_checkpoint_verified": False,
            "support_activation_claimed": False,
            **video_rule_metadata(self._rule_specs),
        }

    def clear_diagnostic_records(self) -> None:
        """Discard only diagnostic history; keep counters and replay state."""

        self._layout_records.clear()
        self._algorithm_records.clear()

    def _projector_callback(self, request: Any) -> Any:
        self.clear_diagnostic_records()
        result = self.kernel(getattr(request, "payload", None), config=self.config, **({"rules": self._rules} if self._rules else {}))
        replay_record: dict[str, Any] = {}
        if self.backbone == "qwen3_vl":
            from mllm_comp.backbones.qwen3_vl.video_compression import (
                Qwen3VLVideoCompressionResponse,
            )

            deepstack = tuple(
                merge_visionzip_source_groups(branch, result, **({"rules": self._rules} if self._rules else {}))
                for branch in request.deepstack_features
            )
            returned = Qwen3VLVideoCompressionResponse.from_result(
                request,
                result,
                deepstack_features=deepstack,
            )
            replay_record = {
                "deepstack_branch_count": len(deepstack),
                "deepstack_result_shapes": tuple(
                    tuple(int(item) for item in branch.shape) for branch in deepstack
                ),
            }
        else:
            returned = result
        self._invocations += 1
        record = layout_delta_record(result.layout_deltas[0])
        record.update(
            {
                "stage": "dynamic_visionzip_reduce",
                "projector_position": self.projector_position,
                "target_site": video_projector_site(self.backbone, self.projector_position),
                "detection_layers": self.detection_layers,
                "invocation_id": getattr(request, "invocation_id", None),
                "media_id": getattr(request, "media_id", None),
                "total_budgets": result.metadata["total_budgets_by_video"][0],
            }
        )
        self._layout_records.append(record)
        algorithm_record = _dytok_invocation_record(request, result)
        algorithm_record.update(replay_record)
        self._algorithm_records.append(algorithm_record)
        return returned


def create_dytok_video_executor(
    *,
    backbone: str,
    **kwargs: Any,
) -> DyToKVideoExecutor:
    return DyToKVideoExecutor(backbone=backbone, **kwargs)


def _dytok_invocation_record(request: Any, result: Any) -> dict[str, Any]:
    payload = getattr(request, "payload", None)
    signals = getattr(payload, "signals", {})
    payload_metadata = getattr(payload, "metadata", {})
    attention_by_layer = signals.get("vision_encoder_attention", {})
    hidden_by_layer = signals.get("vision_encoder_hidden_states", {})
    importance_by_video = signals.get("dytok.frame_importance_by_video")
    attention = (
        attention_by_layer.get(-2) if isinstance(attention_by_layer, Mapping) else None
    )
    hidden = hidden_by_layer.get(-2) if isinstance(hidden_by_layer, Mapping) else None
    source_count = int(getattr(result, "source_count", 0))
    result_count = int(getattr(result, "result_count", 0))
    if (
        getattr(attention, "ndim", None) != 1
        or getattr(hidden, "ndim", None) != 2
        or int(attention.shape[0]) != source_count
        or int(hidden.shape[0]) != source_count
    ):
        raise BackendIntegrationError(
            "DyToK invocation evidence lost aligned VisionZip signals"
        )
    if (
        not isinstance(importance_by_video, Sequence)
        or isinstance(importance_by_video, (str, bytes))
        or len(importance_by_video) != 1
        or getattr(importance_by_video[0], "ndim", None) != 1
    ):
        raise BackendIntegrationError(
            "DyToK invocation evidence requires one frame-importance vector"
        )
    torch = _torch()
    importance = importance_by_video[0]
    if not (
        bool(torch.isfinite(attention).all().item())
        and bool(torch.isfinite(hidden).all().item())
        and bool(torch.isfinite(importance).all().item())
    ):
        raise BackendIntegrationError("DyToK invocation evidence is non-finite")

    metadata = getattr(result, "metadata", {})
    records_by_video = metadata.get("frame_records_by_video")
    targets_by_video = metadata.get("merge_target_indices_by_video")
    groups_by_video = getattr(result, "source_groups", ())
    representatives_by_video = getattr(result, "representative_indices", ())
    if (
        not isinstance(records_by_video, tuple)
        or len(records_by_video) != 1
        or not isinstance(targets_by_video, tuple)
        or len(targets_by_video) != 1
        or len(groups_by_video) != 1
        or len(representatives_by_video) != 1
    ):
        raise BackendIntegrationError(
            "DyToK invocation evidence requires exactly one video"
        )
    frame_records = tuple(records_by_video[0])
    if int(importance.numel()) != len(frame_records):
        raise BackendIntegrationError(
            "DyToK frame-importance evidence differs from temporal slices"
        )
    groups = tuple(tuple(int(item) for item in group) for group in groups_by_video[0])
    targets = tuple(int(item) for item in targets_by_video[0])
    representatives = tuple(int(item) for item in representatives_by_video[0])
    if len(groups) != result_count or len(targets) != result_count:
        raise BackendIntegrationError("DyToK invocation group evidence drifted")

    frame_summaries: list[dict[str, Any]] = []
    group_offset = 0
    covered_sources: list[int] = []
    zero_budget_sources: list[int] = []
    non_singleton_contextual_groups = 0
    assigned_contextual_sources = 0
    source_offset = 0
    for frame in frame_records:
        frame_source_count = int(frame["source_count"])
        frame_budget = int(frame["total_budget"])
        stop = group_offset + frame_budget
        frame_groups = groups[group_offset:stop]
        frame_targets = targets[group_offset:stop]
        dominant = {int(item) for item in frame["dominant_indices"]}
        contextual = {int(item) for item in frame["contextual_target_indices"]}
        dominant_pairs = tuple(
            (group, target)
            for group, target in zip(frame_groups, frame_targets)
            if target in dominant
        )
        contextual_pairs = tuple(
            (group, target)
            for group, target in zip(frame_groups, frame_targets)
            if target in contextual
        )
        if (
            len(dominant_pairs) != int(frame["dominant_budget"])
            or len(contextual_pairs) != int(frame["contextual_budget"])
            or any(group != (target,) for group, target in dominant_pairs)
            or any(target not in group for group, target in contextual_pairs)
        ):
            raise BackendIntegrationError(
                "DyToK invocation does not match its dynamic VisionZip budgets"
            )
        assigned = sum(len(group) - 1 for group, _target in contextual_pairs)
        expected_assigned = frame_source_count - frame_budget if frame_budget else 0
        if assigned != expected_assigned:
            raise BackendIntegrationError(
                "DyToK contextual assignments do not match the frame budget"
            )
        covered_sources.extend(item for group in frame_groups for item in group)
        if frame_budget == 0:
            zero_budget_sources.extend(
                range(source_offset, source_offset + frame_source_count)
            )
        non_singleton = sum(len(group) > 1 for group, _target in contextual_pairs)
        non_singleton_contextual_groups += non_singleton
        assigned_contextual_sources += assigned
        frame_summaries.append(
            {
                "temporal_index": int(frame["temporal_index"]),
                "source_count": frame_source_count,
                "frame_importance": float(frame["frame_importance"]),
                "total_budget": frame_budget,
                "dominant_budget": int(frame["dominant_budget"]),
                "contextual_budget": int(frame["contextual_budget"]),
                "assigned_contextual_source_count": assigned,
                "non_singleton_contextual_group_count": non_singleton,
                "anchor_legalizations": int(frame["anchor_legalizations"]),
            }
        )
        group_offset = stop
        source_offset += frame_source_count
    if group_offset != result_count or source_offset != source_count:
        raise BackendIntegrationError(
            "DyToK frame evidence changed source/output counts"
        )

    covered = tuple(sorted(covered_sources))
    dropped = tuple(sorted(zero_budget_sources))
    provider_importance_values = tuple(
        float(item) for item in importance.detach().cpu().tolist()
    )
    importance_values = tuple(
        float(item["frame_importance"]) for item in frame_summaries
    )
    budgets = tuple(int(item["total_budget"]) for item in frame_summaries)
    return {
        "invocation_id": getattr(request, "invocation_id", None),
        "media_id": getattr(request, "media_id", None),
        "source_count": source_count,
        "result_count": result_count,
        "canonical_config": _plain_runtime_value(metadata.get("canonical_config", {})),
        "attention_shape": tuple(int(item) for item in attention.shape),
        "hidden_state_shape": tuple(int(item) for item in hidden.shape),
        "visionzip_signals_finite": True,
        "frame_importance": importance_values,
        "frame_importance_sum": sum(importance_values),
        "frame_importance_nonnegative": all(value >= 0 for value in importance_values),
        "frame_importance_varies": len(set(importance_values)) > 1,
        "provider_frame_importance": provider_importance_values,
        "provider_frame_importance_sum": sum(provider_importance_values),
        "provider_kernel_importance_max_abs_delta": max(
            (
                abs(provider - kernel)
                for provider, kernel in zip(
                    provider_importance_values, importance_values
                )
            ),
            default=0.0,
        ),
        "detection_frame_source": payload_metadata.get("dytok_detection_frame_source"),
        "detection_model_frozen": payload_metadata.get("dytok_detection_model_frozen"),
        "detection_model_local_only": payload_metadata.get(
            "dytok_detection_local_only"
        ),
        "conditioning_excludes_assistant_answer": payload_metadata.get(
            "dytok_conditioning_excludes_assistant_answer"
        ),
        "detection_attention": dict(signals.get("detection_attention", {})),
        "prompt_layout": dict(signals.get("prompt_layout", {})),
        "target_layers": tuple(int(item) for item in metadata.get("target_layers", ())),
        "detection_stage": metadata.get("detection_stage"),
        "base_compressor": metadata.get("base_compressor"),
        "temporal_slice_count": len(frame_summaries),
        "base_budget_per_temporal_slice": int(
            metadata.get("base_budget_per_temporal_slice", 0)
        ),
        "upper_limit": int(metadata.get("upper_limit", 0)),
        "total_budgets": budgets,
        "dominant_budgets": tuple(
            int(item["dominant_budget"]) for item in frame_summaries
        ),
        "contextual_budgets": tuple(
            int(item["contextual_budget"]) for item in frame_summaries
        ),
        "budget_total": sum(budgets),
        "budgets_vary": len(set(budgets)) > 1,
        "budgets_within_source_cap": all(
            0 <= int(item["total_budget"]) <= int(item["source_count"])
            for item in frame_summaries
        ),
        "dominant_contextual_split_exact": all(
            int(item["dominant_budget"]) == (6 * int(item["total_budget"])) // 7
            and int(item["contextual_budget"])
            == int(item["total_budget"]) - int(item["dominant_budget"])
            for item in frame_summaries
        ),
        "importance_budget_order_preserved": all(
            importance_values[left] <= importance_values[right]
            or budgets[left] >= budgets[right]
            for left in range(len(budgets))
            for right in range(len(budgets))
        ),
        "source_partition_or_zero_budget_drop_exact": tuple(
            sorted((*covered, *dropped))
        )
        == tuple(range(source_count)),
        "covered_sources_unique": len(covered) == len(set(covered)),
        "merge_targets_inside_groups": all(
            target in group for group, target in zip(groups, targets)
        ),
        "representatives_source_ordered": representatives
        == tuple(sorted(representatives)),
        "assigned_contextual_source_count": assigned_contextual_sources,
        "non_singleton_contextual_group_count": non_singleton_contextual_groups,
        "max_source_group_size": max((len(group) for group in groups), default=0),
        "selection_scope": metadata.get("selection_scope"),
        "attention_statistic": metadata.get("attention_statistic"),
        "similarity": metadata.get("similarity"),
        "contextual_target_sampling": metadata.get("contextual_target_sampling"),
        "merge_value": metadata.get("merge_value"),
        "representative_policy": metadata.get("representative_policy"),
        "training_free_compressor": metadata.get("training_free_compressor"),
        "image_variant_shared": metadata.get("image_variant_shared"),
        "frame_summaries": frame_summaries,
    }


def _validated_config(stage: Any, binding_config: Any) -> Mapping[str, Any]:
    stage_config = getattr(stage, "config", {}) if stage is not None else {}
    if stage_config is None:
        stage_config = {}
    if not isinstance(stage_config, Mapping):
        raise BackendIntegrationError("DyToK stage config must be a mapping")
    binding_values = (
        getattr(binding_config, "config", {}) if binding_config is not None else {}
    )
    if binding_values not in ({}, None):
        raise BackendIntegrationError(
            "DyToK video binding does not accept binding-local config"
        )
    return resolve_canonical_video_config("dytok", stage_config)


def _plain_runtime_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _plain_runtime_value(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(_plain_runtime_value(item) for item in value)
    if isinstance(value, list):
        return [_plain_runtime_value(item) for item in value]
    return value


def _validate_binding(
    binding: Any,
    stage: Any,
    backbone: str,
    phase: ExecutionPhase,
) -> None:
    expected = f"dytok.video.{backbone}"
    if binding is not None:
        if getattr(binding, "name", None) != expected:
            raise BackendIntegrationError(
                f"DyToK executor requires binding {expected!r}"
            )
        if phase not in getattr(binding, "supported_phases", ()):
            raise BackendIntegrationError(
                f"DyToK binding {expected!r} does not support {phase.value}"
            )
        targets = tuple(
            getattr(item, "name", None) for item in binding.ordered_targets()
        )
        if targets != ("detection_budget_provider", "dynamic_visionzip_reduce"):
            raise BackendIntegrationError("DyToK compound target order changed")
    if stage is not None:
        if getattr(stage, "algorithm", None) != "dytok":
            raise BackendIntegrationError("DyToK executor received another algorithm")
        if getattr(stage, "variant", None) != "dytok.video":
            raise BackendIntegrationError("DyToK executor received another variant")


def _projected_frame_count(payload: VideoTokenPayload) -> int:
    temporal = tuple(int(coordinate.t) for coordinate in payload.coordinates)
    distinct = tuple(dict.fromkeys(temporal))
    if not distinct or distinct != tuple(range(len(distinct))):
        raise BackendIntegrationError(
            "DyToK projected temporal slices must be contiguous and zero based"
        )
    return len(distinct)


def _first_positive_integer(*values: Any) -> int:
    for value in values:
        if isinstance(value, int) and not isinstance(value, bool) and value > 0:
            return value
        if (
            isinstance(value, Sequence)
            and not isinstance(value, (str, bytes))
            and value
        ):
            first = value[0]
            if isinstance(first, int) and not isinstance(first, bool) and first > 0:
                return first
    raise BackendIntegrationError("DyToK could not resolve a positive model dimension")


def _torch() -> Any:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - dependency-light control plane
        raise BackendIntegrationError(
            "DyToK provider execution requires the existing PyTorch environment"
        ) from exc
    return torch


__all__ = [
    "DyToKVideoExecutor",
    "DyToKVideoSignalProvider",
    "create_dytok_video_executor",
]
