"""LongVU capability provider and independent three-backbone video executor.

DINOv2 is loaded only from an explicitly supplied local directory and remains
frozen outside the base-model state.  The reference-initialized SVA fusion is
registered on the loaded model before the bridge is installed, so native
training creates its optimizer after the trainable LongVU parameters exist.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

from mllm_comp.backbones._qwen_video_inputs import restore_qwen_video_raster
from mllm_comp.backbones._video_bridge import (
    install_video_bridge,
    register_video_callback,
    release_video_callback,
)
from mllm_comp.compressors.video._rules import load_video_rules, video_rule_metadata
from mllm_comp.backbones._provider_patching import MethodPatch
from mllm_comp.compressors.video import (
    LongVUVideoKernel,
    VideoTokenCoordinate,
    VideoTokenPayload,
    pool_longvu_source_groups,
)
from mllm_comp.compressors.video.config import resolve_canonical_video_config
from mllm_comp.compressors.video.positions import video_longvu_frame_selection_site
from mllm_comp.core.errors import BackendIntegrationError
from mllm_comp.core.layout import layout_delta_record
from mllm_comp.core.vocabulary import ExecutionPhase


_BACKBONES = ("llava_onevision", "qwen25_vl", "qwen3_vl")
_PROVIDER_ATTRIBUTE = "_mllm_comp_longvu_video_signal_provider"
_FUSION_ATTRIBUTE = "mllm_comp_longvu_fusion"
_CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
_CLIP_STD = (0.26862954, 0.26130258, 0.27577711)
_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)


@dataclass
class _LongVUCapture:
    frames: Any | None = None
    grid_thw: Any | None = None
    raw_features: Any | None = None
    calls: int = 0


class LongVUVideoSignalProvider:
    """Attach native raw features, normalized pixels, and frozen DINO features."""

    def __init__(
        self,
        *,
        backbone: str,
        model: Any,
        dino_path: str | Path | None = None,
        dino_encoder: Any | None = None,
    ) -> None:
        if backbone not in _BACKBONES:
            raise ValueError(f"Unknown LongVU video backbone {backbone!r}")
        self.backbone = backbone
        self.model = model
        self.dino_path = None if dino_path is None else Path(dino_path)
        self._dino_encoder = dino_encoder
        self._active: ContextVar[_LongVUCapture | None] = ContextVar(
            f"longvu_video_signals_{backbone}_{id(self)}",
            default=None,
        )
        self._patch: MethodPatch | None = None
        self._installed = False
        self._capture_count = 0

    @property
    def installed(self) -> bool:
        return self._installed

    @property
    def dino_hidden_size(self) -> int:
        self.prepare()
        value = getattr(self._dino_encoder, "hidden_size", None)
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise BackendIntegrationError(
                "LongVU DINOv2 provider does not publish a positive hidden size"
            )
        return value

    def prepare(self) -> None:
        """Resolve and load only an already-present local DINOv2 asset."""

        if self._dino_encoder is not None:
            return
        if self.dino_path is None:
            raise BackendIntegrationError(
                "LongVU requires runtime.metadata['longvu_dino_path'] naming an "
                "existing local DINOv2 Giant directory; downloads are forbidden"
            )
        from mllm_comp.backbones._longvu_torch import FrozenDinov2Encoder

        self._dino_encoder = FrozenDinov2Encoder(self.dino_path, interp_size=576)

    def install(self) -> None:
        if self._installed:
            return
        self.prepare()
        existing = getattr(self.model, _PROVIDER_ATTRIBUTE, None)
        if existing is not None and existing is not self:
            raise BackendIntegrationError(
                "A different LongVU video signal provider is already installed"
            )
        owner = self._capture_owner()
        original = getattr(owner, "forward", None)
        if not callable(original):
            raise BackendIntegrationError(
                "LongVU could not resolve the native visual forward"
            )
        patch = MethodPatch.apply(
            owner,
            "forward",
            self._capture_replacement(original),
            original,
            label="LongVU video",
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
        errors = []
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
                f"LongVU video signal-provider uninstall failed: {errors}"
            )

    @contextmanager
    def capture_context(
        self,
        *,
        frames: Any | None = None,
        grid_thw: Any | None = None,
    ) -> Iterator[_LongVUCapture]:
        if not self._installed:
            raise BackendIntegrationError("LongVU signal provider is not installed")
        if self._active.get() is not None:
            raise BackendIntegrationError("Nested LongVU capture is unsupported")
        if self.backbone == "llava_onevision":
            if getattr(frames, "ndim", None) != 4 or grid_thw is not None:
                raise BackendIntegrationError(
                    "LLaVA LongVU capture requires one rank-four frame tensor"
                )
            capture = _LongVUCapture(frames=frames)
        else:
            if frames is not None or getattr(grid_thw, "ndim", None) != 2:
                raise BackendIntegrationError(
                    "Qwen LongVU capture requires rank-two video_grid_thw"
                )
            if int(grid_thw.shape[0]) != 1:
                raise BackendIntegrationError(
                    "LongVU supports exactly one Qwen video grid row per sample"
                )
            capture = _LongVUCapture(grid_thw=grid_thw)
        token = self._active.set(capture)
        try:
            yield capture
        finally:
            self._active.reset(token)

    def payload_signals(
        self,
        capture: _LongVUCapture,
        *,
        token_count: int,
        coordinates: Sequence[Any],
    ) -> Mapping[str, Any]:
        if not isinstance(capture, _LongVUCapture) or capture.calls != 1:
            raise BackendIntegrationError(
                "LongVU capture did not observe exactly one visual encoder pass"
            )
        if capture.raw_features is None:
            raise BackendIntegrationError("LongVU capture is missing raw features")
        if token_count < 1 or len(tuple(coordinates)) != token_count:
            raise BackendIntegrationError(
                "LongVU projected coordinates do not match the payload"
            )
        return {}

    def enrich_payload(
        self,
        payload: VideoTokenPayload,
        capture: _LongVUCapture,
    ) -> VideoTokenPayload:
        torch = _torch()
        if not isinstance(payload, VideoTokenPayload):
            raise TypeError("payload must be a VideoTokenPayload")
        if payload.video_count != 1:
            raise BackendIntegrationError("LongVU provider requires exactly one video")
        if capture.calls != 1 or capture.raw_features is None:
            raise BackendIntegrationError("LongVU capture is incomplete")
        if self.backbone == "llava_onevision":
            raw, raw_coordinates = self._llava_raw(capture)
            pixels = _siglip_pixels_to_imagenet(capture.frames, torch=torch)
            pixel_source = "llava_siglip_frames"
        else:
            raw, raw_coordinates = self._qwen_raw(capture)
            pixel_key = (
                "qwen25_vl.pixel_values_videos"
                if self.backbone == "qwen25_vl"
                else "qwen3_vl.pixel_values_videos"
            )
            pixel_values = payload.signals.get(pixel_key)
            pixels = self._qwen_pixels_to_imagenet(pixel_values, capture.grid_thw)
            pixel_source = "qwen_patchified_video_inverse"
        pixels = pixels.to(
            device=payload.features.device,
            dtype=payload.features.dtype,
        )
        self.prepare()
        dino = self._dino_encoder(pixels)
        if (
            getattr(dino, "ndim", None) != 3
            or int(dino.shape[0]) != int(pixels.shape[0])
            or int(dino.shape[1]) != 576
        ):
            raise BackendIntegrationError(
                "LongVU DINOv2 provider returned an incompatible feature raster"
            )
        dino = dino.to(device=payload.features.device, dtype=payload.features.dtype)
        raw = raw.to(device=payload.features.device, dtype=payload.features.dtype)
        frame_count = int(pixels.shape[0])
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
                "pixels": pixels,
                "pixel_video_offsets": (0, frame_count),
                "dinov2_features": dino,
                "dinov2_video_offsets": (0, frame_count),
            },
            metadata={
                **dict(payload.metadata),
                "longvu_raw_stream": "native_visual_pre_projector",
                "longvu_pixel_source": pixel_source,
                "longvu_source_normalization": (
                    "siglip_half"
                    if self.backbone == "llava_onevision"
                    else (
                        "qwen3_half" if self.backbone == "qwen3_vl" else "openai_clip"
                    )
                ),
                "longvu_dino_normalization": "imagenet",
            },
        )

    def runtime_metadata(self) -> dict[str, Any]:
        return {
            "provider": f"{self.backbone}_longvu_video_signals",
            "installed": self._installed,
            "capture_count": self._capture_count,
            "video_only": True,
            "raw_stream_attached": True,
            "dino_frozen": True,
            "dino_path": None if self.dino_path is None else str(self.dino_path),
            "dino_loaded": self._dino_encoder is not None,
            "real_checkpoint_verified": False,
        }

    def _capture_owner(self) -> Any:
        if self.backbone == "llava_onevision":
            get_vision = getattr(self.model, "get_vision_tower", None)
            outer = get_vision() if callable(get_vision) else None
            if outer is None:
                raise BackendIntegrationError(
                    "LLaVA LongVU requires the native outer vision tower"
                )
            return outer
        visual = getattr(getattr(self.model, "model", None), "visual", None)
        blocks = tuple(getattr(visual, "blocks", ()))
        if not blocks:
            raise BackendIntegrationError("LongVU requires native Qwen visual blocks")
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
                    "LongVU observed multiple visual encoder calls"
                )
            if getattr(output, "ndim", None) not in {2, 3}:
                raise BackendIntegrationError(
                    "LongVU native visual encoder returned an unknown raw shape"
                )
            capture.raw_features = output
            capture.calls = 1
            return output

        return replacement

    def _llava_raw(
        self,
        capture: _LongVUCapture,
    ) -> tuple[Any, tuple[VideoTokenCoordinate, ...]]:
        raw = capture.raw_features
        frames = capture.frames
        if getattr(raw, "ndim", None) != 3 or int(raw.shape[0]) != int(frames.shape[0]):
            raise BackendIntegrationError(
                "LLaVA LongVU raw features do not match the frame count"
            )
        token_count = int(raw.shape[1])
        side = int(token_count**0.5)
        if side * side != token_count:
            raise BackendIntegrationError(
                "LLaVA LongVU raw features are not a square patch raster"
            )
        coordinates = tuple(
            VideoTokenCoordinate(t=frame, h=row, w=column, frame_index=frame)
            for frame in range(int(raw.shape[0]))
            for row in range(side)
            for column in range(side)
        )
        return raw.reshape(-1, int(raw.shape[-1])), coordinates

    def _qwen_raw(
        self,
        capture: _LongVUCapture,
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
            method_name="LongVU",
        )

    def _qwen_pixels_to_imagenet(self, values: Any, grid: Any) -> Any:
        torch = _torch()
        if not isinstance(values, torch.Tensor) or values.ndim != 2:
            raise BackendIntegrationError(
                "Qwen LongVU requires native rank-two pixel_values_videos"
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
        temporal, height, width = tuple(int(value) for value in grid[0].tolist())
        expected_width = 3 * temporal_patch * patch_size * patch_size
        if (
            int(values.shape[0]) != temporal * height * width
            or int(values.shape[1]) != expected_width
            or height % merge_size
            or width % merge_size
        ):
            raise BackendIntegrationError(
                "Qwen LongVU patchified pixels do not match video_grid_thw"
            )
        patches = values.reshape(
            temporal,
            height // merge_size,
            width // merge_size,
            merge_size,
            merge_size,
            3,
            temporal_patch,
            patch_size,
            patch_size,
        )
        frames = (
            patches.permute(0, 6, 5, 1, 3, 7, 2, 4, 8)
            .contiguous()
            .reshape(
                temporal,
                temporal_patch,
                3,
                height * patch_size,
                width * patch_size,
            )
        )
        source_mean, source_std = (
            (_CLIP_MEAN, _CLIP_STD)
            if self.backbone == "qwen25_vl"
            else ((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        )
        normalisation_mean = _channel_tensor(source_mean, frames, torch=torch)
        normalisation_std = _channel_tensor(source_std, frames, torch=torch)
        frames_01 = frames * normalisation_std[:, None] + normalisation_mean[:, None]
        frames_01 = frames_01.mean(dim=1)
        imagenet_mean = _channel_tensor(_IMAGENET_MEAN, frames_01, torch=torch)
        imagenet_std = _channel_tensor(_IMAGENET_STD, frames_01, torch=torch)
        return (frames_01 - imagenet_mean) / imagenet_std


class LongVUVideoExecutor:
    """Install one canonical LongVU provider, trainable SVA, and callback."""

    algorithm = "longvu"
    variant = "longvu.video"

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
        fusion_module: Any | None = None,
    ) -> None:
        if backbone not in _BACKBONES:
            raise ValueError(f"Unknown LongVU video backbone {backbone!r}")
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
        self.frame_selection_site = video_longvu_frame_selection_site(stage, backbone)
        _validate_binding(binding, stage, backbone, self.execution_phase)
        runtime_metadata = (
            getattr(runtime, "metadata", {}) if runtime is not None else {}
        )
        dino_path = (
            runtime_metadata.get("longvu_dino_path")
            if isinstance(runtime_metadata, Mapping)
            else None
        )
        self.kernel = LongVUVideoKernel()
        self._rules, self._rule_specs = load_video_rules(stage, contract_projection=contract_projection)
        self.signal_provider = signal_provider or LongVUVideoSignalProvider(
            backbone=backbone,
            model=model,
            dino_path=dino_path,
        )
        self.fusion_module = fusion_module
        self._fusion_attached = False
        self._owns_fusion_attribute = False
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
            self._prepare_fusion_module()
            self._attach_fusion_module()
            self.signal_provider.install()
            bridge, owns_bridge, projector_site = install_video_bridge(self.model, self.backbone)
            handle = register_video_callback(
                bridge,
                projector_site,
                self._projector_callback,
                label="LongVU",
                signal_provider=self.signal_provider,
            )
        except BaseException:
            release_video_callback(bridge, handle, owns_bridge)
            try:
                self.signal_provider.uninstall()
            except Exception:
                pass
            self._detach_fusion_module()
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
        try:
            self._detach_fusion_module()
        except Exception as exc:
            errors.append(f"{type(exc).__name__}: {exc}")
        self._bridge_handle = None
        self._bridge = None
        self._owns_bridge = False
        self._installed = False
        if errors:
            raise BackendIntegrationError(f"LongVU video uninstall failed: {errors}")

    def compression_state_dict(self) -> Mapping[str, Any]:
        self._prepare_fusion_module()
        state_dict = getattr(self.fusion_module, "state_dict", None)
        if not callable(state_dict):
            raise BackendIntegrationError(
                "LongVU fusion module does not expose state_dict()"
            )
        return state_dict()

    def load_compression_state_dict(self, state: Mapping[str, Any]) -> None:
        if not isinstance(state, Mapping):
            raise TypeError("LongVU compression state must be a mapping")
        self._prepare_fusion_module()
        loader = getattr(self.fusion_module, "load_state_dict", None)
        if not callable(loader):
            raise BackendIntegrationError(
                "LongVU fusion module does not expose load_state_dict()"
            )
        loader(state, strict=True)

    def runtime_metadata(self) -> dict[str, Any]:
        binding_metadata = getattr(self.binding, "metadata", {})
        if not isinstance(binding_metadata, Mapping):
            binding_metadata = {}
        verified_phases = tuple(binding_metadata.get("verified_phases", ()))
        verified = self.execution_phase.value in verified_phases and self.frame_selection_site == "pre_sva"
        parameter_count, trainable_count = _parameter_counts(self.fusion_module)
        provider_metadata = getattr(self.signal_provider, "runtime_metadata", None)
        return {
            "executor": f"{self.backbone}_longvu_video",
            "algorithm": self.algorithm,
            "variant": self.variant,
            "backbone": self.backbone,
            "binding": getattr(self.binding, "name", None),
            "execution_phase": self.execution_phase.value,
            "installed": self._installed,
            "canonical_config": dict(self.config),
            "frame_selection_site": self.frame_selection_site,
            "sva_site": "pre_output_projector",
            "canonical_evidence_applicable": self.frame_selection_site == "pre_sva",
            "projector_invocations": self._invocations,
            "records_scope": "latest_video_request",
            "projector_layout_deltas": list(self._layout_records),
            "algorithm_records": list(self._algorithm_records),
            "signal_provider": (
                provider_metadata() if callable(provider_metadata) else None
            ),
            "bridge": (
                self._bridge.runtime_metadata()
                if self._bridge is not None
                and callable(getattr(self._bridge, "runtime_metadata", None))
                else None
            ),
            "stages": {
                "pre_sva": ("dinov2_frame_selection", "sva_fusion", "output_projection"),
                "pre_output_projector": ("sva_fusion", "dinov2_frame_selection", "output_projection"),
                "post_output_projector": ("sva_fusion", "output_projection", "dinov2_frame_selection"),
            }[self.frame_selection_site],
            "fusion_attribute": _FUSION_ATTRIBUTE,
            "fusion_registered_pre_optimizer": self._fusion_attached,
            "fusion_parameter_count": parameter_count,
            "fusion_trainable_parameter_count": trainable_count,
            "dino_frozen": True,
            "compression_checkpoint_api": (
                "compression_state_dict",
                "load_compression_state_dict",
            ),
            "real_checkpoint_verified": verified,
            "support_activation_claimed": verified,
            "evidence_artifact": (
                binding_metadata.get(f"{self.execution_phase.value}_evidence_artifact")
                if verified
                else None
            ),
            **video_rule_metadata(self._rule_specs),
        }

    def clear_diagnostic_records(self) -> None:
        """Discard only diagnostic history; keep counters and replay state."""

        self._layout_records.clear()
        self._algorithm_records.clear()

    def _projector_callback(self, request: Any) -> Any:
        self.clear_diagnostic_records()
        result = self.kernel(
            getattr(request, "payload", None),
            fusion_module=self.fusion_module,
            config=self.config,
            frame_selection_site=self.frame_selection_site,
        **({"rules": self._rules} if self._rules else {}))
        replay_record: dict[str, Any] = {}
        if self.backbone == "qwen3_vl":
            from mllm_comp.backbones.qwen3_vl.video_compression import (
                Qwen3VLVideoCompressionResponse,
            )

            deepstack = tuple(
                pool_longvu_source_groups(branch, result)
                for branch in request.deepstack_features
            )
            returned = Qwen3VLVideoCompressionResponse.from_result(
                request,
                result,
                deepstack_features=deepstack,
            )
            replay_record = {
                "deepstack_branch_count": len(deepstack),
                "deepstack_source_shapes": tuple(
                    tuple(int(item) for item in branch.shape)
                    for branch in request.deepstack_features
                ),
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
                "stage": "dino_frame_select_sva_fusion",
                "invocation_id": getattr(request, "invocation_id", None),
                "media_id": getattr(request, "media_id", None),
                "selected_frames": len(
                    result.metadata["selected_frame_indices_by_video"][0]
                ),
                "tokens_per_selected_frame": 144,
                **{key: result.metadata[key] for key in (
                    "frame_selection_site", "sva_site", "sva_frame_counts_by_video",
                    "output_projector_frame_counts_by_video", "extra_sva_frames_by_video",
                    "extra_output_projector_frames_by_video",
                )},
            }
        )
        self._layout_records.append(record)
        algorithm_record = _longvu_invocation_record(request, result)
        algorithm_record.update(replay_record)
        self._algorithm_records.append(algorithm_record)
        return returned

    def _prepare_fusion_module(self) -> None:
        if self.fusion_module is not None:
            return
        prepare = getattr(self.signal_provider, "prepare", None)
        if callable(prepare):
            prepare()
        dino_hidden = getattr(self.signal_provider, "dino_hidden_size", None)
        if callable(dino_hidden):
            dino_hidden = dino_hidden()
        if isinstance(dino_hidden, bool) or not isinstance(dino_hidden, int):
            raise BackendIntegrationError(
                "LongVU signal provider does not expose dino_hidden_size"
            )
        raw_hidden, output_hidden = _model_dimensions(self.model, self.backbone)
        from mllm_comp.backbones._longvu_torch import LongVUFusionModule

        self.fusion_module = LongVUFusionModule(
            raw_hidden_size=raw_hidden,
            dino_hidden_size=dino_hidden,
            output_hidden_size=output_hidden,
            vision_hidden_size=int(self.config["vision_hidden_size"]),
            dino_interp_size=int(self.config["dino_interp_size"]),
            query_num=int(self.config["query_num"]),
            connector_depth=int(self.config["connector_depth"]),
        )
        device, dtype = _model_device_dtype(self.model)
        self.fusion_module.to(device=device, dtype=dtype)

    def _attach_fusion_module(self) -> None:
        if self._fusion_attached:
            return
        existing = getattr(self.model, _FUSION_ATTRIBUTE, None)
        if existing is not None and existing is not self.fusion_module:
            raise BackendIntegrationError(
                "A different LongVU fusion module is already registered"
            )
        if existing is None:
            setattr(self.model, _FUSION_ATTRIBUTE, self.fusion_module)
            self._owns_fusion_attribute = True
        self._fusion_attached = True

    def _detach_fusion_module(self) -> None:
        if not self._fusion_attached:
            return
        if getattr(self.model, _FUSION_ATTRIBUTE, None) is not self.fusion_module:
            raise BackendIntegrationError(
                "LongVU fusion-module ownership changed while active"
            )
        if self._owns_fusion_attribute:
            delattr(self.model, _FUSION_ATTRIBUTE)
        self._fusion_attached = False
        self._owns_fusion_attribute = False


def _longvu_invocation_record(request: Any, result: Any) -> dict[str, Any]:
    """Freeze the defining provider signals and DINO/SVA decisions."""

    torch = _torch()
    payload = getattr(request, "payload", None)
    features = getattr(payload, "features", None)
    raw_features = getattr(payload, "raw_features", None)
    signals = getattr(payload, "signals", {})
    pixels = signals.get("pixels") if isinstance(signals, Mapping) else None
    dino = signals.get("dinov2_features") if isinstance(signals, Mapping) else None
    source_count = int(getattr(result, "source_count", 0))
    result_count = int(getattr(result, "result_count", 0))
    if (
        getattr(features, "ndim", None) != 2
        or getattr(raw_features, "ndim", None) != 2
        or getattr(pixels, "ndim", None) != 4
        or getattr(dino, "ndim", None) != 3
        or int(features.shape[0]) != source_count
    ):
        raise BackendIntegrationError(
            "LongVU invocation evidence lost an aligned projected/raw/pixel/DINO signal"
        )
    finite = {
        "projected_features_finite": bool(torch.isfinite(features).all().item()),
        "raw_features_finite": bool(torch.isfinite(raw_features).all().item()),
        "pixels_finite": bool(torch.isfinite(pixels).all().item()),
        "dinov2_features_finite": bool(torch.isfinite(dino).all().item()),
        "result_features_finite": bool(
            torch.isfinite(getattr(result, "features", None)).all().item()
        ),
    }
    if not all(finite.values()):
        raise BackendIntegrationError(
            "LongVU invocation evidence contains non-finite data"
        )

    metadata = getattr(result, "metadata", {})
    media_ids = tuple(getattr(result, "media_ids", ()))
    source_offsets = tuple(int(value) for value in result.source_video_offsets)
    result_offsets = tuple(int(value) for value in result.video_offsets)
    if len(media_ids) != 1 or len(source_offsets) != 2 or len(result_offsets) != 2:
        raise BackendIntegrationError(
            "LongVU invocation evidence requires exactly one video"
        )
    selected = tuple(
        tuple(int(index) for index in values)
        for values in metadata.get("selected_frame_indices_by_video", ())
    )
    scores = tuple(metadata.get("frame_similarity_scores_by_video", ()))
    groups = tuple(
        tuple(tuple(int(index) for index in group) for group in video_groups)
        for video_groups in result.source_groups
    )
    representatives = tuple(
        tuple(int(index) for index in values)
        for values in result.representative_indices
    )
    if (
        len(selected) != 1
        or len(scores) != 1
        or len(groups) != 1
        or len(representatives) != 1
        or len(groups[0]) != result_count
        or len(representatives[0]) != result_count
    ):
        raise BackendIntegrationError("LongVU invocation decision records drifted")
    payload_metadata = getattr(payload, "metadata", {})
    return {
        **dict(metadata),
        **finite,
        "algorithm": "longvu",
        "variant": "longvu.video",
        "source_count": source_count,
        "result_count": result_count,
        "video_count": 1,
        "per_video_source_counts": (source_offsets[1] - source_offsets[0],),
        "per_video_result_counts": (result_offsets[1] - result_offsets[0],),
        "exact_budgets": tuple(
            int(value) for value in metadata.get("exact_budgets", ())
        ),
        "projected_features_shape": tuple(int(item) for item in features.shape),
        "raw_features_shape": tuple(int(item) for item in raw_features.shape),
        "pixels_shape": tuple(int(item) for item in pixels.shape),
        "dinov2_features_shape": tuple(int(item) for item in dino.shape),
        "result_features_shape": tuple(int(item) for item in result.features.shape),
        "source_groups": groups,
        "representative_indices": representatives,
        "selected_frame_count": len(selected[0]),
        "projected_pixel_source": payload_metadata.get("longvu_pixel_source"),
        "source_normalization": payload_metadata.get("longvu_source_normalization"),
        "dino_normalization": payload_metadata.get("longvu_dino_normalization"),
    }


def create_longvu_video_executor(
    *,
    backbone: str,
    **kwargs: Any,
) -> LongVUVideoExecutor:
    return LongVUVideoExecutor(backbone=backbone, **kwargs)


def _validated_config(stage: Any, binding_config: Any) -> Mapping[str, Any]:
    stage_config = getattr(stage, "config", {}) if stage is not None else {}
    if stage_config is None:
        stage_config = {}
    if not isinstance(stage_config, Mapping):
        raise BackendIntegrationError("LongVU stage config must be a mapping")
    binding_values = (
        getattr(binding_config, "config", {}) if binding_config is not None else {}
    )
    if binding_values not in ({}, None):
        raise BackendIntegrationError(
            "LongVU video binding does not accept binding-local config"
        )
    return resolve_canonical_video_config("longvu", stage_config)


def _validate_binding(
    binding: Any,
    stage: Any,
    backbone: str,
    phase: ExecutionPhase,
) -> None:
    expected = f"longvu.video.{backbone}"
    if binding is not None:
        if getattr(binding, "name", None) != expected:
            raise BackendIntegrationError(
                f"LongVU executor requires binding {expected!r}"
            )
        if phase not in getattr(binding, "supported_phases", ()):
            raise BackendIntegrationError(
                f"LongVU binding {expected!r} does not support {phase.value}"
            )
        targets = tuple(
            getattr(item, "name", None) for item in binding.ordered_targets()
        )
        if targets != ("dino_frame_select_sva_fusion",):
            raise BackendIntegrationError("LongVU binding target changed")
    if stage is not None:
        if getattr(stage, "algorithm", None) != "longvu":
            raise BackendIntegrationError("LongVU executor received another algorithm")
        if getattr(stage, "variant", None) != "longvu.video":
            raise BackendIntegrationError("LongVU executor received another variant")


def _model_dimensions(model: Any, backbone: str) -> tuple[int, int]:
    if backbone == "llava_onevision":
        get_vision = getattr(model, "get_vision_tower", None)
        outer = get_vision() if callable(get_vision) else None
        raw = _first_positive_integer(
            getattr(outer, "hidden_size", None),
            getattr(getattr(outer, "config", None), "hidden_size", None),
            getattr(
                getattr(getattr(outer, "vision_tower", None), "config", None),
                "hidden_size",
                None,
            ),
        )
        config = getattr(model, "config", None)
        output = _first_positive_integer(
            getattr(config, "hidden_size", None),
            getattr(getattr(config, "text_config", None), "hidden_size", None),
        )
        return raw, output
    visual = getattr(getattr(model, "model", None), "visual", None)
    blocks = tuple(getattr(visual, "blocks", ()))
    if not blocks:
        raise BackendIntegrationError("LongVU could not infer Qwen visual dimensions")
    norm = getattr(blocks[-1], "norm1", None)
    shape = getattr(norm, "normalized_shape", None)
    norm_hidden = shape[-1] if isinstance(shape, Sequence) and shape else None
    raw = _first_positive_integer(
        norm_hidden,
        getattr(getattr(visual, "config", None), "hidden_size", None),
        getattr(getattr(visual, "config", None), "embed_dim", None),
    )
    config = getattr(model, "config", None)
    output = _first_positive_integer(
        getattr(getattr(config, "text_config", None), "hidden_size", None),
        getattr(config, "hidden_size", None),
    )
    return raw, output


def _model_device_dtype(model: Any) -> tuple[Any, Any]:
    torch = _torch()
    parameters = getattr(model, "parameters", None)
    if not callable(parameters):
        raise BackendIntegrationError(
            "LongVU requires a loaded PyTorch model before fusion registration"
        )
    parameter = next(iter(parameters()), None)
    if parameter is None:
        return torch.device("cpu"), torch.float32
    return parameter.device, parameter.dtype


def _parameter_counts(module: Any) -> tuple[int, int]:
    parameters = getattr(module, "parameters", None)
    if not callable(parameters):
        return 0, 0
    values = tuple(parameters())
    return (
        sum(int(value.numel()) for value in values),
        sum(int(value.numel()) for value in values if bool(value.requires_grad)),
    )


def _siglip_pixels_to_imagenet(values: Any, *, torch: Any) -> Any:
    if (
        not isinstance(values, torch.Tensor)
        or values.ndim != 4
        or int(values.shape[1]) != 3
    ):
        raise BackendIntegrationError(
            "LLaVA LongVU requires rank-four SigLIP-normalized frames"
        )
    pixels_01 = values * 0.5 + 0.5
    mean = _channel_tensor(_IMAGENET_MEAN, pixels_01, torch=torch)
    std = _channel_tensor(_IMAGENET_STD, pixels_01, torch=torch)
    return (pixels_01 - mean) / std


def _channel_tensor(values: Sequence[float], like: Any, *, torch: Any) -> Any:
    return torch.tensor(values, device=like.device, dtype=like.dtype).view(1, 3, 1, 1)


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
    raise BackendIntegrationError("LongVU could not resolve a positive model dimension")


def _torch() -> Any:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - dependency-light control plane
        raise BackendIntegrationError(
            "LongVU video runtime requires the existing PyTorch environment"
        ) from exc
    return torch


__all__ = [
    "LongVUVideoExecutor",
    "LongVUVideoSignalProvider",
    "create_longvu_video_executor",
]
