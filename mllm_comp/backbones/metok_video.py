"""METok signal provider and registration of shared native video pruning.

The provider keeps SigLIP outside base-model checkpoint ownership and loads it
only from an explicitly supplied local directory. The executor combines the
event-aware projector kernel and request-local pruning method with the shared
native decoder loops and existing label/cache repair in each video bridge.
"""

from __future__ import annotations

import inspect
from collections.abc import Callable, Mapping, Sequence
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Iterator

from mllm_comp.backbones._qwen_video_inputs import (
    qwen_video_pixels_to_siglip,
    restore_qwen_video_raster,
)
from mllm_comp.backbones.video_decoder import VideoDecoderPruningExecutor
from mllm_comp.compressors.video.metok import METokDecoderCall
from mllm_comp.backbones._video_bridge import (
    install_video_bridge,
    register_video_callback,
    release_video_callback,
)
from mllm_comp.compressors.video._rules import load_video_rules, video_rule_metadata
from mllm_comp.compressors.layer_positions import LayerPosition
from mllm_comp.compressors.video.positions import video_decoder_positions, video_projector_position, video_projector_site
from mllm_comp.backbones._provider_patching import MethodPatch
from mllm_comp.backbones.image_training import attach_shifted_causal_loss
from mllm_comp.compressors.image.decoder_lifecycle import (
    DecoderBoundary,
    compact_dynamic_cache_prefix,
)
from mllm_comp.compressors.video import (
    METokPruningResult,
    METokVideoKernel,
    VideoCompressionResult,
    VideoTokenCoordinate,
    VideoTokenPayload,
    replay_metok_projector_layout,
)
from mllm_comp.compressors.video.config import resolve_canonical_video_config
from mllm_comp.core.errors import BackendIntegrationError, InvariantError
from mllm_comp.core.layout import layout_delta_record
from mllm_comp.core.vocabulary import ExecutionPhase


_BACKBONES = ("llava_onevision", "qwen25_vl", "qwen3_vl")
_PROVIDER_ATTRIBUTE = "_mllm_comp_metok_video_signal_provider"


@dataclass
class _METokCapture:
    frames: Any | None = None
    grid_thw: Any | None = None
    raw_features: Any | None = None
    calls: int = 0


class METokVideoSignalProvider:
    """Attach frozen SigLIP frame-text similarity and native raw provenance."""

    def __init__(
        self,
        *,
        backbone: str,
        model: Any,
        siglip_path: str | Path | None = None,
        siglip_encoder: Any | None = None,
    ) -> None:
        if backbone not in _BACKBONES:
            raise ValueError(f"Unknown METok video backbone {backbone!r}")
        self.backbone = backbone
        self.model = model
        self.siglip_path = None if siglip_path is None else Path(siglip_path)
        self._siglip_encoder = siglip_encoder
        self._active: ContextVar[_METokCapture | None] = ContextVar(
            f"metok_video_signals_{backbone}_{id(self)}",
            default=None,
        )
        self._patch: MethodPatch | None = None
        self._installed = False
        self._capture_count = 0

    @property
    def installed(self) -> bool:
        return self._installed

    def prepare(self) -> None:
        if self._siglip_encoder is not None:
            return
        if self.siglip_path is None:
            raise BackendIntegrationError(
                "METok requires runtime.metadata['metok_siglip_path'] naming an "
                "existing local SigLIP SO400M directory; downloads are forbidden"
            )
        from mllm_comp.backbones._metok_torch import FrozenSigLIPEncoder

        self._siglip_encoder = FrozenSigLIPEncoder(
            self.siglip_path,
            include_vision=self.backbone != "llava_onevision",
        )

    def install(self) -> None:
        if self._installed:
            return
        self.prepare()
        existing = getattr(self.model, _PROVIDER_ATTRIBUTE, None)
        if existing is not None and existing is not self:
            raise BackendIntegrationError(
                "A different METok video signal provider is already installed"
            )
        owner = self._capture_owner()
        original = getattr(owner, "forward", None)
        if not callable(original):
            raise BackendIntegrationError(
                "METok could not resolve the native visual forward"
            )
        patch = MethodPatch.apply(
            owner,
            "forward",
            self._capture_replacement(original),
            original,
            label="METok",
        )
        try:
            setattr(self.model, _PROVIDER_ATTRIBUTE, self)
        except BaseException:
            patch.restore()
            raise
        self._patch = patch
        self._installed = True

    def uninstall(self) -> None:
        if not self._installed:
            return
        errors: list[str] = []
        if getattr(self.model, _PROVIDER_ATTRIBUTE, None) is not self:
            errors.append("provider ownership changed while active")
        elif _PROVIDER_ATTRIBUTE in vars(self.model):
            delattr(self.model, _PROVIDER_ATTRIBUTE)
        if self._patch is not None:
            try:
                self._patch.restore()
            except Exception as exc:
                errors.append(f"{type(exc).__name__}: {exc}")
        self._patch = None
        self._active.set(None)
        self._installed = False
        if errors:
            raise BackendIntegrationError(
                f"METok video signal-provider uninstall failed: {errors}"
            )

    @contextmanager
    def capture_context(
        self,
        *,
        frames: Any | None = None,
        grid_thw: Any | None = None,
    ) -> Iterator[_METokCapture]:
        if not self._installed:
            raise BackendIntegrationError("METok signal provider is not installed")
        if self._active.get() is not None:
            raise BackendIntegrationError("Nested METok capture is unsupported")
        if self.backbone == "llava_onevision":
            if getattr(frames, "ndim", None) != 4 or grid_thw is not None:
                raise BackendIntegrationError(
                    "LLaVA METok capture requires one rank-four frame tensor"
                )
            capture = _METokCapture(frames=frames)
        else:
            if frames is not None or getattr(grid_thw, "ndim", None) != 2:
                raise BackendIntegrationError(
                    "Qwen METok capture requires rank-two video_grid_thw"
                )
            if int(grid_thw.shape[0]) != 1:
                raise BackendIntegrationError(
                    "METok supports exactly one Qwen video grid row"
                )
            capture = _METokCapture(grid_thw=grid_thw)
        token = self._active.set(capture)
        try:
            yield capture
        finally:
            self._active.reset(token)

    def payload_signals(
        self,
        capture: _METokCapture,
        *,
        token_count: int,
        coordinates: Sequence[Any],
    ) -> Mapping[str, Any]:
        if not isinstance(capture, _METokCapture) or capture.calls != 1:
            raise BackendIntegrationError(
                "METok capture did not observe exactly one visual encoder pass"
            )
        if capture.raw_features is None:
            raise BackendIntegrationError("METok capture is missing raw features")
        if token_count < 1 or len(tuple(coordinates)) != token_count:
            raise BackendIntegrationError(
                "METok projected coordinates do not match the payload"
            )
        return {}

    def enrich_payload(
        self,
        payload: VideoTokenPayload,
        capture: _METokCapture,
    ) -> VideoTokenPayload:
        if not isinstance(payload, VideoTokenPayload):
            raise TypeError("payload must be a VideoTokenPayload")
        if payload.video_count != 1:
            raise BackendIntegrationError("METok provider requires exactly one video")
        if capture.calls != 1 or capture.raw_features is None:
            raise BackendIntegrationError("METok capture is incomplete")
        text = payload.signals.get("text_context")
        if not isinstance(text, str) or not text.strip():
            raise BackendIntegrationError(
                "METok requires user/system prompt text without the assistant answer"
            )
        self.prepare()
        if self.backbone == "llava_onevision":
            raw, raw_coordinates, frame_count, raw_tokens = self._llava_raw(capture)
            similarity = self._siglip_encoder.native_frame_similarity(
                text,
                raw,
                frame_count=frame_count,
                tokens_per_frame=raw_tokens,
            )
            pixel_source = "native_siglip_visual_features"
        else:
            raw, raw_coordinates = self._qwen_raw(capture)
            pixel_key = (
                "qwen25_vl.pixel_values_videos"
                if self.backbone == "qwen25_vl"
                else "qwen3_vl.pixel_values_videos"
            )
            pixels = self._qwen_pixels_to_siglip(
                payload.signals.get(pixel_key),
                capture.grid_thw,
            )
            pixels = pixels.to(
                device=payload.features.device,
                dtype=payload.features.dtype,
            )
            similarity = self._siglip_encoder.pixel_frame_similarity(text, pixels)
            pixel_source = "qwen_patchified_video_inverse_siglip_384"
        raw = raw.to(device=payload.features.device, dtype=payload.features.dtype)
        similarity = similarity.to(
            device=payload.features.device,
            dtype=payload.features.dtype,
        )
        self._capture_count += 1
        return VideoTokenPayload(
            features=payload.features,
            media_ids=payload.media_ids,
            video_offsets=payload.video_offsets,
            coordinates=payload.coordinates,
            raw_features=raw,
            raw_video_offsets=(0, int(raw.shape[0])),
            raw_coordinates=raw_coordinates,
            token_mask=payload.token_mask,
            signals={
                **dict(payload.signals),
                "metok.frame_text_similarity": similarity,
                "siglip_text_encoder": "frozen_local_siglip",
                "prefill_decoder_attention": "native_full_key_qk",
                "prefill_layout_state": "before_layers_3_10_19",
            },
            metadata={
                **dict(payload.metadata),
                "metok_raw_stream": "native_visual_pre_projector",
                "metok_similarity_source": pixel_source,
                "metok_conditioning_excludes_assistant_answer": True,
                "metok_siglip_local_only": True,
            },
        )

    def runtime_metadata(self) -> dict[str, Any]:
        vision_enabled = getattr(self._siglip_encoder, "vision_enabled", None)
        return {
            "provider": f"{self.backbone}_metok_video_signals",
            "installed": self._installed,
            "capture_count": self._capture_count,
            "video_only": True,
            "raw_stream_attached": True,
            "siglip_frozen": True,
            "siglip_local_only": True,
            "siglip_path": None if self.siglip_path is None else str(self.siglip_path),
            "siglip_loaded": self._siglip_encoder is not None,
            "siglip_vision_enabled": vision_enabled,
            "assistant_answer_excluded": True,
            "real_checkpoint_verified": False,
        }

    def _capture_owner(self) -> Any:
        if self.backbone == "llava_onevision":
            get_vision = getattr(self.model, "get_vision_tower", None)
            outer = get_vision() if callable(get_vision) else None
            if outer is None:
                raise BackendIntegrationError(
                    "LLaVA METok requires the native outer vision tower"
                )
            return outer
        visual = getattr(getattr(self.model, "model", None), "visual", None)
        blocks = tuple(getattr(visual, "blocks", ()))
        if not blocks:
            raise BackendIntegrationError("METok requires native Qwen visual blocks")
        return blocks[-1]

    def _capture_replacement(
        self,
        original: Callable[..., Any],
    ) -> Callable[..., Any]:
        def replacement(_owner: Any, *args: Any, **kwargs: Any) -> Any:
            capture = self._active.get()
            output = original(*args, **kwargs)
            if capture is None:
                return output
            if capture.calls:
                raise BackendIntegrationError(
                    "METok observed multiple visual encoder calls"
                )
            if getattr(output, "ndim", None) not in {2, 3}:
                raise BackendIntegrationError(
                    "METok native visual encoder returned an unknown raw shape"
                )
            capture.raw_features = output
            capture.calls = 1
            return output

        return replacement

    def _llava_raw(
        self,
        capture: _METokCapture,
    ) -> tuple[Any, tuple[VideoTokenCoordinate, ...], int, int]:
        raw = capture.raw_features
        frames = capture.frames
        if getattr(raw, "ndim", None) != 3 or int(raw.shape[0]) != int(frames.shape[0]):
            raise BackendIntegrationError(
                "LLaVA METok raw features do not match the frame count"
            )
        token_count = int(raw.shape[1])
        side = int(token_count**0.5)
        if side * side != token_count:
            raise BackendIntegrationError(
                "LLaVA METok raw features are not a square patch raster"
            )
        frame_count = int(raw.shape[0])
        coordinates = tuple(
            VideoTokenCoordinate(t=frame, h=row, w=column, frame_index=frame)
            for frame in range(frame_count)
            for row in range(side)
            for column in range(side)
        )
        return (
            raw.reshape(-1, int(raw.shape[-1])),
            coordinates,
            frame_count,
            token_count,
        )

    def _qwen_raw(
        self,
        capture: _METokCapture,
    ) -> tuple[Any, tuple[VideoTokenCoordinate, ...]]:
        visual = getattr(getattr(self.model, "model", None), "visual", None)
        merge_size = _first_positive_integer(
            getattr(visual, "spatial_merge_size", None),
            getattr(getattr(visual, "config", None), "spatial_merge_size", None),
        )
        return restore_qwen_video_raster(
            capture.raw_features,
            capture.grid_thw,
            visual,
            backbone=self.backbone,
            merge_size=merge_size,
            merge_unit=_first_positive_integer(
                getattr(visual, "spatial_merge_unit", None), merge_size * merge_size
            ),
            method_name="METok",
        )

    def _qwen_pixels_to_siglip(self, values: Any, grid: Any) -> Any:
        torch = _torch()
        if not isinstance(values, torch.Tensor) or values.ndim != 2:
            raise BackendIntegrationError(
                "Qwen METok requires native rank-two pixel_values_videos"
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
            method_name="METok",
        )


@dataclass(frozen=True)
class _ActiveRequest:
    request: Any
    projector_result: VideoCompressionResult
    response: Any | None = None


class METokVideoExecutor(VideoDecoderPruningExecutor):
    """Compound event-pool and progressive decoder executor."""

    algorithm = "metok"
    decoder_label = "METok"
    variant = "metok.video"

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
    ) -> None:
        if backbone not in _BACKBONES:
            raise ValueError(f"Unknown METok video backbone {backbone!r}")
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
        self.pruning_layers = tuple(
            int(value) for value in self.config["pruning_layers"]
        )
        self.projector_position = video_projector_position(stage, backbone, target_name="event_pool")
        self.application_positions = video_decoder_positions(stage, self.pruning_layers, algorithm="metok", backbone=backbone)
        self.ratio = float(self.config["ratio"])
        _validate_binding(binding, stage, backbone, self.execution_phase)
        runtime_metadata = (
            getattr(runtime, "metadata", {}) if runtime is not None else {}
        )
        siglip_path = (
            runtime_metadata.get("metok_siglip_path")
            if isinstance(runtime_metadata, Mapping)
            else None
        )
        self.kernel = METokVideoKernel()
        self._rules, self._rule_specs = load_video_rules(stage, contract_projection=contract_projection)
        self.signal_provider = signal_provider or METokVideoSignalProvider(
            backbone=backbone,
            model=model,
            siglip_path=siglip_path,
        )
        self._active: ContextVar[_ActiveRequest | None] = ContextVar(
            f"metok_video_active_{backbone}_{id(self)}",
            default=None,
        )
        self._bridge: Any | None = None
        self._bridge_handle: Any | None = None
        self._owns_bridge = False
        self._patches: list[MethodPatch] = []
        self._installed = False
        self._projector_count = 0
        self._pruning_count = 0
        self._projector_records: list[dict[str, Any]] = []
        self._algorithm_records: list[dict[str, Any]] = []
        self._pruning_records: list[dict[str, Any]] = []
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
            self.signal_provider.install()
            decoder = _decoder_owner(self.model, self.backbone)
            native_decoder_forward = getattr(decoder, "forward", None)
            if not callable(native_decoder_forward):
                raise BackendIntegrationError(
                    "METok could not resolve the native decoder forward"
                )
            patches.append(
                MethodPatch.apply(
                    decoder,
                    "forward",
                    self._decoder_forward_replacement(native_decoder_forward),
                    native_decoder_forward,
                    label="METok",
                )
            )
            native_model_forward = getattr(self.model, "forward", None)
            if not callable(native_model_forward):
                raise BackendIntegrationError(
                    "METok could not resolve the native model forward"
                )
            patches.append(
                MethodPatch.apply(
                    self.model,
                    "forward",
                    self._model_forward_replacement(native_model_forward),
                    native_model_forward,
                    label="METok",
                )
            )
            bridge, owns_bridge, projector_site = install_video_bridge(self.model, self.backbone)
            handle = register_video_callback(
                bridge,
                projector_site,
                self._projector_callback,
                label="METok",
                signal_provider=self.signal_provider,
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
            try:
                self.signal_provider.uninstall()
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
        try:
            self.signal_provider.uninstall()
        except Exception as exc:
            errors.append(f"{type(exc).__name__}: {exc}")
        self._bridge_handle = None
        self._bridge = None
        self._owns_bridge = False
        self._patches.clear()
        self._active.set(None)
        self._installed = False
        if errors:
            raise BackendIntegrationError(f"METok video uninstall failed: {errors}")

    def runtime_metadata(self) -> dict[str, Any]:
        provider_metadata = getattr(self.signal_provider, "runtime_metadata", None)
        return {
            "executor": f"{self.backbone}_metok_video",
            "algorithm": self.algorithm,
            "variant": self.variant,
            "backbone": self.backbone,
            "binding": getattr(self.binding, "name", None),
            "execution_phase": self.execution_phase.value,
            "installed": self._installed,
            "canonical_config": dict(self.config),
            "projector_position": self.projector_position,
            "projector_site": video_projector_site(self.backbone, self.projector_position),
            "application_positions": [point.as_dict() for point in self.application_positions],
            "score_layers": list(self.pruning_layers[:2]),
            "canonical_evidence_applicable": self.projector_position == "after"
                and self.pruning_layers[:2] == (3, 10)
                and self.application_positions == tuple(LayerPosition("inter_layer", layer) for layer in (3, 10, 19)),
            "projector_invocations": self._projector_count,
            "pruning_invocations": self._pruning_count,
            "records_scope": "latest_video_request",
            "projector_layout_deltas": list(self._projector_records),
            "algorithm_records": list(self._algorithm_records),
            "pruning_layout_deltas": list(self._pruning_records),
            "cache_repairs": list(self._cache_records),
            "signal_provider": (
                provider_metadata() if callable(provider_metadata) else None
            ),
            "bridge": (
                self._bridge.runtime_metadata()
                if self._bridge is not None
                and callable(getattr(self._bridge, "runtime_metadata", None))
                else None
            ),
            "compound_projector_llm": True,
            "training_gradients_preserved": True,
            "siglip_trainable": False,
            "real_checkpoint_verified": False,
            "support_activation_claimed": False,
            **video_rule_metadata(self._rule_specs),
        }

    def clear_diagnostic_records(self) -> None:
        """Discard only diagnostic history; keep counters and replay state."""

        self._projector_records.clear()
        self._algorithm_records.clear()
        self._pruning_records.clear()
        self._cache_records.clear()

    def _projector_callback(self, request: Any) -> Any:
        self.clear_diagnostic_records()
        if self._active.get() is not None:
            raise BackendIntegrationError(
                "Nested METok video projector requests are unsupported"
            )
        payload = getattr(request, "payload", None)
        if self.projector_position == "before":
            reference = payload.signals.get("projector_reference_features")
            if reference is None:
                raise BackendIntegrationError("METok before-projector pooling requires the complete projected reference")
            result = self.kernel(replace(payload, features=reference), config=self.config, **({"rules": {k: v for k, v in self._rules.items() if k != "pool_transform"}} if self._rules else {}))
            result = replace(result, features=replay_metok_projector_layout(payload.features, result, **({"rules": self._rules} if self._rules else {})))
        else:
            result = self.kernel(payload, config=self.config, **({"rules": self._rules} if self._rules else {}))
        if "pool_transform" in self._rules:
            result = replace(result, metadata={**dict(result.metadata),
                "value_operation": "custom",
                "metok_frames": tuple(tuple({**dict(record), "interpolation": "custom"}
                    for record in records) for records in result.metadata["metok_frames"])})
        result = replace(result, metadata={**dict(result.metadata),
            "projector_position": self.projector_position,
            "target_site": video_projector_site(self.backbone, self.projector_position),
            "projected_plan_reference_tokens": payload.token_count if self.projector_position == "before" else 0})
        response = None
        if self.backbone == "qwen3_vl":
            from mllm_comp.backbones.qwen3_vl.video_compression import (
                Qwen3VLVideoCompressionResponse,
            )

            branches = tuple(
                replay_metok_projector_layout(branch, result, **({"rules": self._rules} if self._rules else {}))
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
                "stage": "event_pool",
                "projector_position": self.projector_position,
                "target_site": video_projector_site(self.backbone, self.projector_position),
                "invocation_id": getattr(request, "invocation_id", None),
                "media_id": getattr(request, "media_id", None),
                "event_lengths": result.metadata["metok_event_lengths"][0],
                "important_events": result.metadata["metok_important_events"][0],
            }
        )
        self._projector_records.append(record)
        self._algorithm_records.append(
            _metok_invocation_record(request, result, response=response, **({"rules": self._rules} if self._rules else {}))
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
                    "Nested METok video model calls are unsupported"
                )
            try:
                if self.backbone != "llava_onevision":
                    return original(*args, **kwargs)
                if self._bridge is None:
                    raise BackendIntegrationError(
                        "LLaVA METok video bridge is not installed"
                    )
                labels = values.get("labels")
                if labels is not None and values.get("return_dict") is False:
                    raise BackendIntegrationError(
                        "LLaVA METok training requires return_dict=True"
                    )
                with self._bridge.metok_decoder_context(
                    external_training_loss=labels is not None
                ):
                    output = original(*args, **kwargs)
                    if labels is None:
                        return output
                    repaired = self._bridge.metok_training_labels()
                    if repaired is None:
                        raise InvariantError(
                            "LLaVA METok training labels were not produced"
                        )
                    return attach_shifted_causal_loss(
                        output,
                        repaired,
                        operation="LLaVA METok video compression",
                    )
            finally:
                self._active.set(None)

        return replacement

    def _record_pruning(
        self,
        result: METokPruningResult,
        cache_record: Mapping[str, Any],
    ) -> None:
        self._pruning_count += 1
        visual = layout_delta_record(result.visual_layout_delta)
        point = result.metadata.get("application_position", {"boundary": "inter_layer", "index": result.boundary_layer})
        visual.update(
            {
                "stage": f"{'before_layer' if point['boundary'] == 'inter_layer' else 'post_attention_pre_ffn'}_{point['index']}",
                "boundary": "post_attention_pre_ffn" if point["boundary"] == "post_attention_pre_ffn" else "before_layer",
                "boundary_layer": result.boundary_layer,
                "application_position": dict(point),
                "score_layer": self.pruning_layers[result.stage_index] if result.stage_index < 2 else None,
                "stage_index": result.stage_index,
                "ratio": result.ratio,
                "event_lengths_before": result.event_lengths_before,
                "event_lengths_after": result.event_lengths_after,
                "important_event_indices": result.important_event_indices,
                "kept_visual_indices": result.kept_visual_indices,
                "dropped_visual_indices": result.dropped_visual_indices,
                "kept_sequence_indices": result.kept_sequence_indices,
                "cumulative_retained_projector_indices": (
                    result.retained_projector_indices
                ),
                "dropped_projector_indices": result.dropped_projector_indices,
                "cumulative_prompt_keep_indices": (
                    result.cumulative_prompt_keep_indices
                ),
                "sequence_layout_delta": layout_delta_record(
                    result.sequence_layout_delta
                ),
                "attention_reduction": result.metadata["attention_reduction"],
                "selection": result.metadata["selection"],
                "cache_policy": result.metadata["cache_policy"],
                "position_policy": result.metadata["position_policy"],
                "final_visual_removal": result.metadata["final_visual_removal"],
                "cache": dict(cache_record),
            }
        )
        self._pruning_records.append(visual)
        self._cache_records.append(dict(cache_record))

    def _new_decoder_call(self, projector_result, sequence_length, num_layers):
        try:
            positions = video_decoder_positions(
                self.stage,
                self.pruning_layers,
                num_layers,
                algorithm=self.algorithm,
                backbone=self.backbone,
            )
        except (TypeError, ValueError) as exc:
            raise BackendIntegrationError(str(exc)) from exc
        self.application_positions = positions
        return METokDecoderCall(
            projector_result,
            sequence_length=sequence_length,
            positions=positions,
            pruning_layers=self.pruning_layers,
            ratio=self.ratio,
            rules=self._rules,
        )

    def _publish_decoder_pruning(
        self, result, cache_record, point, *, deepstack_branch_count=0
    ):
        kwargs = (
            {
                "deepstack_branch_count": deepstack_branch_count,
                "deepstack_consumed_branch_count": min(
                    point.index, deepstack_branch_count
                ),
            }
            if self.backbone == "qwen3_vl"
            else {}
        )
        self._bridge.apply_metok_pruning_layout(result, **kwargs)
        self._record_pruning(result, cache_record)

    def _repair_decoder_cache(
        self,
        past_key_values: Any,
        result: METokPruningResult,
        *,
        use_cache: bool,
        position: LayerPosition | None = None,
    ) -> Mapping[str, Any]:
        point = position or LayerPosition("inter_layer", result.boundary_layer)
        mid = point.boundary == "post_attention_pre_ffn"
        boundary = (
            DecoderBoundary.POST_ATTENTION_PRE_FFN
            if mid
            else DecoderBoundary.BEFORE_LAYER
        )
        if not use_cache:
            return {
                "policy": "none_training_forward",
                "boundary": boundary.value,
                "boundary_layer": point.index,
                "source_length": result.sequence_layout_delta.source_count,
                "result_length": result.sequence_layout_delta.result_count,
                "touched_layers": [],
                "application_position": point.as_dict(),
            }
        telemetry = compact_dynamic_cache_prefix(
            past_key_values,
            result.kept_sequence_indices,
            source_length=result.sequence_layout_delta.source_count,
            result_length=result.sequence_layout_delta.result_count,
            layer_count=point.index + int(mid),
            boundary_layer=point.index,
            boundary=boundary,
        )
        return {**telemetry.as_dict(), "application_position": point.as_dict()}


def create_metok_video_executor(
    *,
    backbone: str,
    **kwargs: Any,
) -> METokVideoExecutor:
    return METokVideoExecutor(backbone=backbone, **kwargs)


def _metok_invocation_record(
    request: Any,
    result: VideoCompressionResult,
    *,
    response: Any | None,
    rules: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Record projector decisions and branch shapes without replaying values."""

    payload = getattr(request, "payload", None)
    actual_features = getattr(payload, "features", None)
    features = getattr(payload, "signals", {}).get("projector_reference_features", actual_features)
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
            "METok invocation evidence lost its projected feature layout"
        )
    torch = _torch()
    projected_finite = bool(torch.isfinite(features).all().item())
    result_finite = bool(torch.isfinite(result_features).all().item())
    if not projected_finite or not result_finite:
        raise BackendIntegrationError(
            "METok invocation evidence contains non-finite projected features"
        )

    if int(getattr(result, "video_count", 0)) != 1:
        raise BackendIntegrationError(
            "METok invocation evidence requires exactly one video"
        )
    coordinates = tuple(getattr(payload, "coordinates", ()))
    if len(coordinates) != source_count:
        raise BackendIntegrationError(
            "METok invocation evidence lost its source coordinates"
        )
    temporal = tuple(sorted({int(getattr(item, "t")) for item in coordinates}))
    if not temporal or temporal != tuple(range(len(temporal))):
        raise BackendIntegrationError(
            "METok invocation evidence requires dense temporal indices"
        )
    frame_count = len(temporal)
    grid_height = max(int(getattr(item, "h")) for item in coordinates) + 1
    grid_width = max(int(getattr(item, "w")) for item in coordinates) + 1
    tokens_per_frame = grid_height * grid_width
    expected_coordinates = tuple(
        (frame, row, column)
        for frame in range(frame_count)
        for row in range(grid_height)
        for column in range(grid_width)
    )
    actual_coordinates = tuple(
        (
            int(getattr(item, "t")),
            int(getattr(item, "h")),
            int(getattr(item, "w")),
        )
        for item in coordinates
    )
    if (
        source_count != frame_count * tokens_per_frame
        or actual_coordinates != expected_coordinates
    ):
        raise BackendIntegrationError(
            "METok invocation evidence requires source-ordered frame rasters"
        )

    signals = getattr(payload, "signals", {})
    similarities = signals.get("metok.frame_text_similarity")
    if (
        getattr(similarities, "ndim", None) != 1
        or int(similarities.shape[0]) != frame_count
        or not bool(torch.isfinite(similarities).all().item())
    ):
        raise BackendIntegrationError(
            "METok invocation evidence lost finite frame-text similarities"
        )
    similarity_values = tuple(float(item) for item in similarities.detach().cpu())
    raw_features = getattr(payload, "raw_features", None)
    raw_count = int(getattr(raw_features, "shape", (0,))[0])
    if (
        getattr(raw_features, "ndim", None) != 2
        or raw_count < frame_count
        or raw_count % frame_count
        or not bool(torch.isfinite(raw_features).all().item())
    ):
        raise BackendIntegrationError(
            "METok invocation evidence lost its finite native raw stream"
        )
    raw_tokens_per_frame = raw_count // frame_count

    metadata = getattr(result, "metadata", {})
    canonical_config = dict(metadata.get("canonical_config", {}))
    segments_by_video = metadata.get("metok_segments")
    important_by_video = metadata.get("metok_important_events")
    keyframes_by_video = metadata.get("metok_keyframes")
    lengths_by_video = metadata.get("metok_event_lengths")
    frames_by_video = metadata.get("metok_frames")
    groups_by_video = getattr(result, "source_groups", ())
    representatives_by_video = getattr(result, "representative_indices", ())
    single_video_records = (
        segments_by_video,
        important_by_video,
        keyframes_by_video,
        lengths_by_video,
        frames_by_video,
        groups_by_video,
        representatives_by_video,
    )
    if any(
        not isinstance(item, tuple) or len(item) != 1 for item in single_video_records
    ):
        raise BackendIntegrationError(
            "METok invocation evidence lost its single-video decisions"
        )
    segments = tuple((int(start), int(end)) for start, end in segments_by_video[0])
    important = tuple(int(item) for item in important_by_video[0])
    keyframes = tuple(tuple(int(item) for item in row) for row in keyframes_by_video[0])
    event_lengths = tuple(int(item) for item in lengths_by_video[0])
    frame_records = tuple(dict(item) for item in frames_by_video[0])
    groups = tuple(tuple(int(item) for item in group) for group in groups_by_video[0])
    representatives = tuple(int(item) for item in representatives_by_video[0])

    segment_cursor = 0
    segment_coverage_exact = bool(segments) if rules and "frame_grouper" in rules else len(segments) == min(
        int(canonical_config["num_segments"]), frame_count
    )
    for start, end in segments:
        segment_coverage_exact = (
            segment_coverage_exact
            and start == segment_cursor
            and end > start
            and end <= frame_count
        )
        segment_cursor = end
    segment_coverage_exact = segment_coverage_exact and segment_cursor == frame_count
    important_exact = (
        important == tuple(sorted(set(important)))
        and len(important) == min(int(canonical_config["top_k_events"]), len(segments))
        and all(0 <= item < len(segments) for item in important)
    )
    keyframes_exact = len(keyframes) == len(segments) and all(
        len(row)
        == max(
            1,
            int((end - start) * float(canonical_config["keyframe_ratio"])),
        )
        and row == tuple(sorted(set(row)))
        and all(start <= frame < end for frame in row)
        for row, (start, end) in zip(keyframes, segments)
    )
    if not segment_coverage_exact or not important_exact or not keyframes_exact:
        raise BackendIntegrationError(
            "METok invocation evidence lost event or keyframe semantics"
        )

    if len(groups) != result_count or len(representatives) != result_count:
        raise BackendIntegrationError(
            "METok invocation evidence result groups do not match its token count"
        )
    flattened_sources = tuple(source for group in groups for source in group)
    source_partition_exact = len(flattened_sources) == source_count and tuple(
        sorted(flattened_sources)
    ) == tuple(range(source_count))
    representatives_source_ordered = representatives == tuple(
        sorted(set(representatives))
    )
    representatives_inside_groups = all(
        representative == group[0]
        for representative, group in zip(representatives, groups)
    )
    if not (
        source_partition_exact
        and representatives_source_ordered
        and representatives_inside_groups
    ):
        raise BackendIntegrationError(
            "METok invocation evidence lost source partition or causal anchors"
        )

    important_set = set(important)
    keyframe_set = {frame for row in keyframes for frame in row}
    event_for_frame = {
        frame: event_index
        for event_index, (start, end) in enumerate(segments)
        for frame in range(start, end)
    }
    event_result_counts = [0 for _ in segments]
    result_cursor = 0
    frame_records_exact = len(frame_records) == frame_count
    frozen_frame_records: list[dict[str, Any]] = []
    for expected_frame, raw_record in enumerate(frame_records):
        record = dict(raw_record)
        event_index = event_for_frame.get(expected_frame, -1)
        is_important = event_index in important_set
        is_keyframe = expected_frame in keyframe_set
        if is_important:
            stride_key = "s1_important" if is_keyframe else "s2_important"
        else:
            stride_key = "s1_normal" if is_keyframe else "s2_normal"
        stride = int(canonical_config[stride_key])
        output_height = (grid_height + stride - 1) // stride
        output_width = (grid_width + stride - 1) // stride
        result_span = tuple(int(item) for item in record.get("result_span", ()))
        expected_span = (
            result_cursor,
            result_cursor + output_height * output_width,
        )
        frame_groups = groups[expected_span[0] : expected_span[1]]
        frame_sources = tuple(source for group in frame_groups for source in group)
        expected_sources = tuple(
            range(
                expected_frame * tokens_per_frame,
                (expected_frame + 1) * tokens_per_frame,
            )
        )
        frame_records_exact = frame_records_exact and (
            int(record.get("frame_index", -1)) == expected_frame
            and int(record.get("event_index", -1)) == event_index
            and bool(record.get("important")) is is_important
            and bool(record.get("keyframe")) is is_keyframe
            and int(record.get("stride", 0)) == stride
            and tuple(int(item) for item in record.get("source_shape", ()))
            == (grid_height, grid_width)
            and tuple(int(item) for item in record.get("result_shape", ()))
            == (output_height, output_width)
            and result_span == expected_span
            and record.get("interpolation") == ("custom" if rules and "pool_transform" in rules else "bilinear")
            and record.get("align_corners") is False
            and tuple(sorted(frame_sources)) == expected_sources
        )
        result_cursor = expected_span[1]
        if 0 <= event_index < len(event_result_counts):
            event_result_counts[event_index] += output_height * output_width
        record.update(
            {
                "frame_index": expected_frame,
                "event_index": event_index,
                "stride": stride,
                "source_shape": (grid_height, grid_width),
                "result_shape": (output_height, output_width),
                "result_span": expected_span,
            }
        )
        frozen_frame_records.append(record)
    frame_records_exact = (
        frame_records_exact
        and result_cursor == result_count
        and tuple(event_result_counts) == event_lengths
        and sum(event_lengths) == result_count
    )
    if not frame_records_exact:
        raise BackendIntegrationError(
            "METok invocation evidence lost its per-frame stride/layout decisions"
        )

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
        "actual_features_shape": tuple(int(item) for item in actual_features.shape),
        "projector_position": metadata.get("projector_position", "after"),
        "target_site": metadata.get("target_site"),
        "projected_plan_reference_tokens": metadata.get("projected_plan_reference_tokens", 0),
        "result_features_shape": tuple(int(item) for item in result_features.shape),
        "projected_features_finite": projected_finite,
        "result_features_finite": result_finite,
        "raw_features_shape": tuple(int(item) for item in raw_features.shape),
        "raw_features_finite": True,
        "raw_tokens_per_frame": raw_tokens_per_frame,
        "frame_text_similarities": similarity_values,
        "frame_text_similarity_finite": True,
        "frame_count": frame_count,
        "grid_height": grid_height,
        "grid_width": grid_width,
        "tokens_per_frame": tokens_per_frame,
        "segments": segments,
        "segment_coverage_exact": segment_coverage_exact,
        "important_events": important,
        "important_events_exact": important_exact,
        "keyframes_by_event": keyframes,
        "keyframe_counts_exact": keyframes_exact,
        "event_lengths": event_lengths,
        "frame_records": tuple(frozen_frame_records),
        "frame_records_exact": frame_records_exact,
        "source_groups": groups,
        "representative_indices": representatives,
        "source_partition_exact": source_partition_exact,
        "representatives_source_ordered": representatives_source_ordered,
        "representatives_first_in_group": representatives_inside_groups,
        "source_group_count": len(groups),
        "non_singleton_source_group_count": sum(len(group) > 1 for group in groups),
        "max_source_group_size": max(len(group) for group in groups),
        "projector_stage": metadata.get("projector_stage"),
        "segmentation_tie_policy": metadata.get("segmentation_tie_policy"),
        "importance_reduction": metadata.get("importance_reduction"),
        "keyframe_count_policy": metadata.get("keyframe_count"),
        "output_order": metadata.get("output_order"),
        "value_operation": metadata.get("value_operation"),
        "layout_source_groups": metadata.get("layout_source_groups"),
        "similarity_source": getattr(payload, "metadata", {}).get(
            "metok_similarity_source"
        ),
        "assistant_answer_excluded": bool(
            getattr(payload, "metadata", {}).get(
                "metok_conditioning_excludes_assistant_answer"
            )
        ),
        "siglip_local_only": bool(
            getattr(payload, "metadata", {}).get("metok_siglip_local_only")
        ),
        "deepstack_branch_count": len(source_deepstack),
        "deepstack_source_shapes": deepstack_source_shapes,
        "deepstack_result_shapes": deepstack_result_shapes,
    }


def _validated_config(stage: Any, binding_config: Any) -> Mapping[str, Any]:
    stage_config = getattr(stage, "config", {}) if stage is not None else {}
    if stage_config is None:
        stage_config = {}
    if not isinstance(stage_config, Mapping):
        raise BackendIntegrationError("METok stage config must be a mapping")
    binding_values = (
        getattr(binding_config, "config", {}) if binding_config is not None else {}
    )
    if binding_values not in ({}, None):
        raise BackendIntegrationError(
            "METok video binding does not accept binding-local config"
        )
    return resolve_canonical_video_config("metok", stage_config)


def _validate_binding(
    binding: Any,
    stage: Any,
    backbone: str,
    phase: ExecutionPhase,
) -> None:
    expected = f"metok.video.{backbone}"
    if binding is not None:
        if getattr(binding, "name", None) != expected:
            raise BackendIntegrationError(
                f"METok executor requires binding {expected!r}"
            )
        if phase not in getattr(binding, "supported_phases", ()):
            raise BackendIntegrationError(
                f"METok binding {expected!r} does not support {phase.value}"
            )
        targets = tuple(
            getattr(item, "name", None) for item in binding.ordered_targets()
        )
        if targets != ("event_pool", "prefill_attention_prune"):
            raise BackendIntegrationError("METok compound target order changed")
    if stage is not None:
        if getattr(stage, "algorithm", None) != "metok":
            raise BackendIntegrationError("METok executor received another algorithm")
        if getattr(stage, "variant", None) != "metok.video":
            raise BackendIntegrationError("METok executor received another variant")


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
            f"{backbone} METok could not resolve language_model"
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
    raise BackendIntegrationError("METok could not resolve a positive model dimension")


def _torch() -> Any:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - dependency-light control plane
        raise BackendIntegrationError(
            "METok provider execution requires the existing PyTorch environment"
        ) from exc
    return torch


__all__ = [
    "METokVideoExecutor",
    "METokVideoSignalProvider",
    "create_metok_video_executor",
]
