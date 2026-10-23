"""Transformers 4.57.3 semantic hook bridge for Qwen3-VL image compression.

The generic Qwen3-VL compression adapter deliberately depends on semantic
registration points instead of private output hooks.  This module supplies
those points for one pinned provider implementation: Hugging Face
Transformers 4.57.3 ``Qwen3VLForConditionalGeneration``.

The bridge is intentionally fail closed:

* the Transformers version and dense Qwen3-VL class/structure signatures must
  match exactly;
* image and video visual calls are distinguished at ``get_*_features`` rather
  than guessed from tensor shape;
* decoder PyramidDrop is integrated inside the decoder loop after DeepStack,
  never through a decoder output hook, and its guarded generate phase uses
  the shared layer-local DynamicCache lifecycle;
* every instance monkey-patch and registration is removable, and multiple
  distinct activations can coexist on one model.

No support unit or evidence lifecycle is advanced by installing this bridge.
Real-checkpoint evidence must still be recorded by the framework's support
registry before a binding can become active.
"""

from __future__ import annotations

from mllm_comp.compressors.image._rules import replay_rule_groups

from mllm_comp.compressors.image.positions import (
    tome_encoder_positions,
    tome_encoder_score_layer,
)
from mllm_comp.compressors.image.tome import tome_merge_reference, replay_tome_image
from mllm_comp.compressors.layer_positions import LayerPosition

from collections.abc import Callable, Mapping, Sequence
from contextlib import nullcontext
from contextvars import ContextVar
from dataclasses import dataclass, field
import importlib.metadata
import inspect
from threading import RLock
from types import MappingProxyType, MethodType
from typing import Any
from uuid import uuid4

from mllm_comp.backbones._vision_attention import (
    execute_vision_attention,
    apply_vision_rotary,
)
from functools import partial
from mllm_comp.backbones._decoder_boundary import decoder_post_attention_boundary
from mllm_comp.backbones._decoder_observations import capture_decoder_module_transitions
from mllm_comp.backbones._provider_patching import preserve_bound_method_signature
from mllm_comp.backbones.image_training import (
    attach_shifted_causal_loss,
    compact_ignored_labels,
    repair_labels_for_layout_delta,
)
from mllm_comp.compressors.image.pyramid_drop import (
    pyramid_attention_key_visibility,
)
from mllm_comp.compressors.image.decoder_lifecycle import layout_merge_groups
from mllm_comp.compressors.image.decoder_registration import (
    resolve_decoder_selection_depth,
)
from mllm_comp.compressors.image.decoder_selection import capture_decoder_attention, decoder_layout_replay
from mllm_comp.compressors.image.restore import RestoreAttentionState
from mllm_comp.compressors.image.decoder_lifecycle import (
    CacheRepairPolicy,
    DecoderBoundary,
    DecoderCallPhase,
    ImageDecoderLifecycle,
    LayerLocalCacheView,
    dynamic_cache_layer_length,
    layout_result_representatives,
)
from mllm_comp.core.errors import BackendIntegrationError, InvariantError
from mllm_comp.core.layout import LayoutDelta
from mllm_comp.core.vocabulary import ExecutionPhase, Modality


SUPPORTED_TRANSFORMERS_VERSION = "4.57.3"
_BRIDGE_ATTRIBUTE = "_mllm_comp_qwen3_vl_provider_bridge"
_SITE_TOME = "encoder_dense_boundary"
_SITE_VISIONZIP = "post_visual_merger_with_deepstack"
_SITE_PYRAMID = "decoder_layer_loop_after_last_deepstack"


@dataclass(frozen=True)
class Qwen3VLProviderBridgeContract:
    """Pinned provider identity and the supported execution envelope."""

    provider: str = "transformers"
    provider_version: str = SUPPORTED_TRANSFORMERS_VERSION
    model_class: str = "Qwen3VLForConditionalGeneration"
    core_class: str = "Qwen3VLModel"
    visual_class: str = "Qwen3VLVisionModel"
    language_model_class: str = "Qwen3VLTextModel"
    supported_phases: frozenset[ExecutionPhase] = frozenset(
        {ExecutionPhase.FORWARD, ExecutionPhase.GENERATE}
    )
    use_cache_by_phase: Mapping[ExecutionPhase, bool] = field(
        default_factory=lambda: MappingProxyType(
            {
                ExecutionPhase.FORWARD: False,
                ExecutionPhase.GENERATE: True,
            }
        )
    )
    target_modality: Modality = Modality.IMAGE
    supports_video: bool = False
    supports_generate: bool = True
    supports_labels: bool = True
    metadata: Mapping[str, Any] = field(
        default_factory=lambda: MappingProxyType(
            {
                "provider_source": ("transformers.models.qwen3_vl.modeling_qwen3_vl"),
                "checkpoint_family": "Qwen3-VL dense",
                "real_checkpoint_verified": False,
                "support_lifecycle_mutated": False,
                "generate_scope": "decoder_methods_with_dynamic_cache",
            }
        )
    )


QWEN3_VL_TRANSFORMERS_4573_CONTRACT = Qwen3VLProviderBridgeContract()


@dataclass(frozen=True)
class _VisualCompressionPlan:
    """Image-feature layout emitted by one visual call."""

    source_counts: tuple[int, ...]
    result_counts: tuple[int, ...]
    layout_deltas: tuple[LayoutDelta, ...]

    @property
    def compressed(self) -> bool:
        return self.source_counts != self.result_counts


@dataclass(frozen=True)
class _ToMeStep:
    controller: Any
    deltas: tuple[LayoutDelta, ...]
    before_counts: tuple[int, ...]
    after_counts: tuple[int, ...]


@dataclass(frozen=True)
class _VisionAttentionCapture:
    """Low-memory statistics captured at one native vision attention call.

    Qwen3-VL does not expose vision attention weights when FlashAttention is
    selected.  The bridge therefore executes the selected block's attention
    in query chunks and retains only the mean attention received by each key,
    plus the *actual rotary key states* used by that call.  No post-block QKV
    projection or dense ``tokens x tokens`` attention tensor is reconstructed.
    """

    counts: tuple[int, ...]
    mean_received: tuple[Any, ...]
    keys: tuple[Any, ...]
    token_sizes: tuple[Any | None, ...]
    restore_steps: tuple[_ToMeStep, ...]


@dataclass(frozen=True)
class _Registration:
    token: str
    site: str
    controller: Any
    metadata: Mapping[str, Any]


class Qwen3VLProviderRegistrationHandle:
    """One idempotent removable controller registration."""

    def __init__(self, bridge: "Qwen3VLTransformers4573Bridge", token: str) -> None:
        self._bridge = bridge
        self.token = token

    @property
    def removed(self) -> bool:
        return not self._bridge._contains_registration(self.token)

    def remove(self) -> None:
        self._bridge._remove_registration(self.token)


@dataclass
class _InstancePatch:
    target: Any
    name: str
    existed: bool
    previous: Any

    @classmethod
    def apply(cls, target: Any, name: str, value: Any) -> "_InstancePatch":
        namespace = getattr(target, "__dict__", {})
        existed = name in namespace
        previous = namespace.get(name)
        setattr(target, name, value)
        return cls(target=target, name=name, existed=existed, previous=previous)

    def restore(self) -> None:
        if self.existed:
            setattr(self.target, self.name, self.previous)
        elif self.name in getattr(self.target, "__dict__", {}):
            delattr(self.target, self.name)


class Qwen3VLTransformers4573Bridge:
    """Reversible semantic source bridge for one Qwen3-VL model instance."""

    contract = QWEN3_VL_TRANSFORMERS_4573_CONTRACT

    def __init__(self, model: Any) -> None:
        self.model = model
        self.core = getattr(model, "model", None)
        self.visual = getattr(self.core, "visual", None)
        self.language_model = getattr(self.core, "language_model", None)
        self._lock = RLock()
        self._registrations: dict[str, _Registration] = {}
        self._patches: list[_InstancePatch] = []
        self._installed = False
        self._invocation_counts = {
            _SITE_TOME: 0,
            _SITE_VISIONZIP: 0,
            _SITE_PYRAMID: 0,
        }
        self._decoder_request_counter = 0
        self._latest_decoder_lifecycle: dict[str, Any] | None = None
        self._latest_restore_calibration: dict[str, Any] | None = None
        suffix = uuid4().hex
        self._visual_modality: ContextVar[str | None] = ContextVar(
            f"qwen3_vl_visual_modality_{suffix}",
            default=None,
        )
        self._visual_plan: ContextVar[_VisualCompressionPlan | None] = ContextVar(
            f"qwen3_vl_visual_plan_{suffix}",
            default=None,
        )
        self._prompt_removal_mask: ContextVar[Any | None] = ContextVar(
            f"qwen3_vl_prompt_removal_{suffix}",
            default=None,
        )
        self._comparison_grid = ContextVar(f"comparison_grid_{suffix}", default=None)
        self._image_count: ContextVar[int | None] = ContextVar(
            f"qwen3_vl_image_count_{suffix}",
            default=None,
        )
        self._prompt_text_embeddings: ContextVar[Any | None] = ContextVar(
            f"qwen3_vl_prompt_text_embeddings_{suffix}", default=None
        )
        self._training_labels: ContextVar[Any | None] = ContextVar(
            f"qwen3_vl_training_labels_{suffix}",
            default=None,
        )
        self._decoder_lifecycle: ContextVar[ImageDecoderLifecycle | None] = ContextVar(
            f"qwen3_vl_decoder_lifecycle_{suffix}",
            default=None,
        )
        self._restore_runtime: ContextVar[dict[str, Any] | None] = ContextVar(
            f"qwen3_vl_restore_runtime_{suffix}",
            default=None,
        )
        self._original_model_forward: Callable[..., Any] | None = None
        self._original_generate: Callable[..., Any] | None = None
        self._original_core_forward: Callable[..., Any] | None = None
        self._original_visual_forward: Callable[..., Any] | None = None
        self._original_language_forward: Callable[..., Any] | None = None
        self._original_get_placeholder_mask: Callable[..., Any] | None = None

    @property
    def installed(self) -> bool:
        return self._installed

    def install(self) -> "Qwen3VLTransformers4573Bridge":
        """Validate then install all semantic endpoints transactionally."""

        with self._lock:
            if self._installed:
                return self
            _validate_provider_version()
            _validate_qwen3_vl_structure(
                self.model,
                core=self.core,
                visual=self.visual,
                language_model=self.language_model,
            )
            existing = getattr(self.model, _BRIDGE_ATTRIBUTE, None)
            if existing is not None and existing is not self:
                raise BackendIntegrationError(
                    "This Qwen3-VL model already has a different provider bridge."
                )
            if getattr(self.model, "mllm_video_compression_bridge", None) is not None:
                raise BackendIntegrationError(
                    "Qwen3-VL image and video provider bridges cannot patch the "
                    "same model instance simultaneously."
                )
            patches: list[_InstancePatch] = []
            try:
                self._capture_original_methods()
                patches.extend(self._semantic_endpoint_patches())
                patches.extend(self._execution_patches())
                patches.append(
                    _InstancePatch.apply(self.model, _BRIDGE_ATTRIBUTE, self)
                )
            except BaseException:
                for patch in reversed(patches):
                    patch.restore()
                raise
            self._patches = patches
            self._installed = True
            return self

    def uninstall(self, *, force: bool = False) -> None:
        """Restore every instance attribute; optionally evict live controllers."""

        with self._lock:
            if not self._installed:
                return
            if self._registrations and not force:
                names = [
                    f"{item.site}:{getattr(item.controller, 'algorithm', '?')}"
                    for item in self._registrations.values()
                ]
                raise BackendIntegrationError(
                    "Cannot uninstall the Qwen3-VL provider bridge while "
                    f"activations remain registered: {names}."
                )
            self._registrations.clear()
            for patch in reversed(self._patches):
                patch.restore()
            self._patches.clear()
            self._installed = False

    def runtime_metadata(self) -> dict[str, Any]:
        with self._lock:
            registrations = tuple(self._registrations.values())
        return {
            "provider": self.contract.provider,
            "provider_version": self.contract.provider_version,
            "backbone": "qwen3_vl",
            "installed": self._installed,
            "supported_phases": sorted(
                phase.value for phase in self.contract.supported_phases
            ),
            "use_cache_by_phase": {
                phase.value: enabled
                for phase, enabled in self.contract.use_cache_by_phase.items()
            },
            "supports_generate": self.contract.supports_generate,
            "active_generate_supported": _registrations_support_generate(
                registrations
            ),
            "supports_video": self.contract.supports_video,
            "supports_labels": self.contract.supports_labels,
            "registration_count": len(registrations),
            "registrations": [
                {
                    "site": item.site,
                    "algorithm": getattr(item.controller, "algorithm", None),
                    "variant": getattr(item.controller, "variant", None),
                }
                for item in registrations
            ],
            "invocation_counts": dict(self._invocation_counts),
            "latest_decoder_lifecycle": (
                dict(self._latest_decoder_lifecycle)
                if self._latest_decoder_lifecycle is not None
                else None
            ),
            "latest_restore_calibration": (
                dict(self._latest_restore_calibration)
                if self._latest_restore_calibration is not None
                else None
            ),
            "real_checkpoint_verified": False,
            "support_lifecycle_mutated": False,
        }

    def _capture_original_methods(self) -> None:
        self._original_model_forward = self.model.forward
        self._original_generate = self.model.generate
        self._original_core_forward = self.core.forward
        self._original_visual_forward = self.visual.forward
        self._original_language_forward = self.language_model.forward
        self._original_get_placeholder_mask = self.core.get_placeholder_mask

    def _semantic_endpoint_patches(self) -> list[_InstancePatch]:
        bridge = self

        def register_tome(
            _visual: Any,
            controller: Any,
            **metadata: Any,
        ) -> Qwen3VLProviderRegistrationHandle:
            return bridge._register(_SITE_TOME, controller, metadata)

        def register_visionzip(
            _visual: Any,
            controller: Any,
            **metadata: Any,
        ) -> Qwen3VLProviderRegistrationHandle:
            return bridge._register(_SITE_VISIONZIP, controller, metadata)

        def register_pyramid(
            _language_model: Any,
            controller: Any,
            **metadata: Any,
        ) -> Qwen3VLProviderRegistrationHandle:
            return bridge._register(_SITE_PYRAMID, controller, metadata)

        return [
            _InstancePatch.apply(
                self.visual,
                "register_encoder_dense_boundary_controller",
                MethodType(register_tome, self.visual),
            ),
            _InstancePatch.apply(
                self.visual,
                "register_post_visual_merger_controller",
                MethodType(register_visionzip, self.visual),
            ),
            _InstancePatch.apply(
                self.language_model,
                "register_decoder_layer_loop_controller",
                MethodType(register_pyramid, self.language_model),
            ),
        ]

    def _execution_patches(self) -> list[_InstancePatch]:
        bridge = self

        def model_forward(module: Any, *args: Any, **kwargs: Any) -> Any:
            return bridge._model_forward(module, *args, **kwargs)

        def generate(module: Any, *args: Any, **kwargs: Any) -> Any:
            return bridge._generate(module, *args, **kwargs)

        def core_forward(module: Any, *args: Any, **kwargs: Any) -> Any:
            return bridge._core_forward(module, *args, **kwargs)

        def visual_forward(module: Any, *args: Any, **kwargs: Any) -> Any:
            return bridge._visual_forward(module, *args, **kwargs)

        def language_forward(module: Any, *args: Any, **kwargs: Any) -> Any:
            return bridge._language_forward(module, *args, **kwargs)

        def get_image_features(
            module: Any,
            pixel_values: Any,
            image_grid_thw: Any = None,
        ) -> Any:
            return bridge._get_visual_features(
                module,
                pixel_values,
                image_grid_thw,
                modality=Modality.IMAGE,
            )

        def get_video_features(
            module: Any,
            pixel_values_videos: Any,
            video_grid_thw: Any = None,
        ) -> Any:
            return bridge._get_visual_features(
                module,
                pixel_values_videos,
                video_grid_thw,
                modality=Modality.VIDEO,
            )

        def get_placeholder_mask(
            module: Any,
            input_ids: Any,
            inputs_embeds: Any,
            image_features: Any = None,
            video_features: Any = None,
        ) -> Any:
            return bridge._get_placeholder_mask(
                module,
                input_ids,
                inputs_embeds,
                image_features=image_features,
                video_features=video_features,
            )

        assert self._original_model_forward is not None
        preserve_bound_method_signature(model_forward, self._original_model_forward)

        return [
            _InstancePatch.apply(
                self.model,
                "forward",
                MethodType(model_forward, self.model),
            ),
            _InstancePatch.apply(
                self.model,
                "generate",
                MethodType(generate, self.model),
            ),
            _InstancePatch.apply(
                self.core,
                "forward",
                MethodType(core_forward, self.core),
            ),
            _InstancePatch.apply(
                self.visual,
                "forward",
                MethodType(visual_forward, self.visual),
            ),
            _InstancePatch.apply(
                self.language_model,
                "forward",
                MethodType(language_forward, self.language_model),
            ),
            _InstancePatch.apply(
                self.core,
                "get_image_features",
                MethodType(get_image_features, self.core),
            ),
            _InstancePatch.apply(
                self.core,
                "get_video_features",
                MethodType(get_video_features, self.core),
            ),
            _InstancePatch.apply(
                self.core,
                "get_placeholder_mask",
                MethodType(get_placeholder_mask, self.core),
            ),
        ]

    def _register(
        self,
        site: str,
        controller: Any,
        metadata: Mapping[str, Any],
    ) -> Qwen3VLProviderRegistrationHandle:
        with self._lock:
            if not self._installed:
                raise BackendIntegrationError(
                    "Qwen3-VL provider bridge must be installed before an "
                    "activation registers."
                )
            _validate_controller_registration(
                site,
                controller,
                metadata,
                visual=self.visual,
                language_model=self.language_model,
            )
            existing_controller_sites = {
                item.site
                for item in self._registrations.values()
                if item.controller is controller
            }
            if existing_controller_sites and not (
                getattr(controller, "algorithm", None) in {"nuwa", "restore"}
                and existing_controller_sites == {_SITE_VISIONZIP}
                and site == _SITE_PYRAMID
            ):
                raise BackendIntegrationError(
                    "The same Qwen3-VL compression activation is already registered."
                )
            token = uuid4().hex
            self._registrations[token] = _Registration(
                token=token,
                site=site,
                controller=controller,
                metadata=MappingProxyType(dict(metadata)),
            )
            return Qwen3VLProviderRegistrationHandle(self, token)

    def _contains_registration(self, token: str) -> bool:
        with self._lock:
            return token in self._registrations

    def _remove_registration(self, token: str) -> None:
        with self._lock:
            self._registrations.pop(token, None)

    def _controllers(self, site: str) -> tuple[Any, ...]:
        with self._lock:
            return tuple(
                item.controller
                for item in self._registrations.values()
                if item.site == site
            )

    def _has_controllers(self) -> bool:
        with self._lock:
            return bool(self._registrations)

    def _model_forward(self, _module: Any, *args: Any, **kwargs: Any) -> Any:
        assert self._original_model_forward is not None
        if not self._has_controllers():
            return self._original_model_forward(*args, **kwargs)
        bound = _bind_forward_arguments(
            self._original_model_forward,
            args,
            kwargs,
        )
        use_cache = bound.get("use_cache", kwargs.get("use_cache"))
        past_key_values = bound.get(
            "past_key_values",
            kwargs.get("past_key_values"),
        )
        labels = bound.get("labels", kwargs.get("labels"))
        active_lifecycle = self._decoder_lifecycle.get()
        if (
            bound.get("pixel_values_videos") is not None
            or bound.get("video_grid_thw") is not None
            or bound.get("second_per_grid_ts") is not None
        ):
            raise BackendIntegrationError(
                "Pinned Qwen3-VL compression is image-only; video-only and "
                "mixed image/video inputs require separate support units."
            )
        if active_lifecycle is None and (
            bool(use_cache) or past_key_values is not None
        ):
            raise BackendIntegrationError(
                "Qwen3-VL image compression provider bridge does not support "
                "cached generation; use FORWARD with use_cache=False."
            )
        if active_lifecycle is not None and use_cache is not True:
            raise BackendIntegrationError(
                "Qwen3-VL compressed generation requires use_cache=True."
            )
        if active_lifecycle is not None and labels is not None:
            raise BackendIntegrationError(
                "Qwen3-VL compressed generation does not accept labels."
            )
        if (
            labels is not None
            and bound.get("return_dict", kwargs.get("return_dict")) is False
        ):
            raise BackendIntegrationError(
                "Compressed Qwen3-VL training requires return_dict=True."
            )
        owned_lifecycle_token = None
        owned_lifecycle: ImageDecoderLifecycle | None = None
        if active_lifecycle is None and self._controllers(_SITE_PYRAMID):
            owned_lifecycle = self._new_decoder_lifecycle(prefix="forward")
            owned_lifecycle_token = self._decoder_lifecycle.set(owned_lifecycle)
        owned_restore_token = None
        if self._restore_runtime.get() is None and any(
            getattr(item.controller, "algorithm", None) == "restore"
            for item in self._registrations.values()
        ):
            self._latest_restore_calibration = None
            owned_restore_token = self._restore_runtime.set({})
        try:
            if labels is None:
                return self._original_model_forward(*args, **kwargs)

            signature = inspect.signature(self._original_model_forward)
            call = signature.bind_partial(*args, **kwargs)
            call.arguments["labels"] = None
            label_token = self._training_labels.set(labels)
            try:
                output = self._original_model_forward(*call.args, **call.kwargs)
                repaired_labels = self._training_labels.get()
                if repaired_labels is None:
                    raise InvariantError(
                        "Qwen3-VL training labels disappeared during compression."
                    )
                return attach_shifted_causal_loss(
                    output,
                    repaired_labels,
                    operation="Qwen3-VL image compression",
                )
            finally:
                self._training_labels.reset(label_token)
        finally:
            if owned_restore_token is not None:
                self._restore_runtime.reset(owned_restore_token)
            if owned_lifecycle_token is not None:
                assert owned_lifecycle is not None
                self._latest_decoder_lifecycle = owned_lifecycle.telemetry()
                self._decoder_lifecycle.reset(owned_lifecycle_token)

    def _generate(self, _module: Any, *args: Any, **kwargs: Any) -> Any:
        assert self._original_generate is not None
        with self._lock:
            registrations = tuple(self._registrations.values())
        if not registrations:
            return self._original_generate(*args, **kwargs)
        if not _registrations_support_generate(registrations):
            raise BackendIntegrationError(
                "Pinned Qwen3-VL does not support generate for this activation; "
                "generation requires one GENERATE-phase decoder compression method."
            )
        if self._decoder_lifecycle.get() is not None:
            raise BackendIntegrationError(
                "Nested Qwen3-VL compressed generation is unsupported."
            )
        lifecycle = self._new_decoder_lifecycle(prefix="generate")
        token = self._decoder_lifecycle.set(lifecycle)
        restore_token = None
        if any(
            getattr(item.controller, "algorithm", None) == "restore"
            for item in registrations
        ):
            self._latest_restore_calibration = None
            restore_token = self._restore_runtime.set({})
        try:
            output = self._original_generate(*args, **kwargs)
            if not lifecycle.prefill_complete:
                raise InvariantError(
                    "Qwen3-VL generate returned before compressed prefill completed."
                )
            return output
        finally:
            self._latest_decoder_lifecycle = lifecycle.telemetry()
            if restore_token is not None:
                self._restore_runtime.reset(restore_token)
            self._decoder_lifecycle.reset(token)

    def _new_decoder_lifecycle(self, *, prefix: str) -> ImageDecoderLifecycle:
        controllers = self._controllers(_SITE_PYRAMID)
        if len(controllers) != 1:
            raise BackendIntegrationError(
                "Qwen3-VL decoder lifecycle requires exactly one image method."
            )
        self._decoder_request_counter += 1
        return ImageDecoderLifecycle(
            request_id=f"{prefix}-{self._decoder_request_counter}",
            algorithm=str(controllers[0].algorithm),
            num_layers=len(self.language_model.layers),
            requires_decoder_boundary=(controllers[0].algorithm != "restore"),
            on_layout_delta=self._repair_decoder_training_labels,
        )

    def _repair_decoder_training_labels(
        self,
        delta: LayoutDelta,
        layer_index: int,
        boundary: DecoderBoundary,
    ) -> None:
        labels = self._training_labels.get()
        if labels is not None:
            self._training_labels.set(
                repair_labels_for_layout_delta(
                    labels,
                    delta,
                    operation=f"Qwen3-VL decoder {boundary.value} layer {layer_index}",
                )
            )

    def _core_forward(self, _module: Any, *args: Any, **kwargs: Any) -> Any:
        assert self._original_core_forward is not None
        plan_token = self._visual_plan.set(None)
        removal_token = self._prompt_removal_mask.set(None)
        image_count_token = self._image_count.set(None)
        text_embedding_token = self._prompt_text_embeddings.set(None)
        try:
            if any(
                getattr(controller, "algorithm", None) == "mmtok"
                for controller in self._controllers(_SITE_VISIONZIP)
            ):
                call = _bind_forward_arguments(
                    self._original_core_forward, args, kwargs
                )
                input_ids = call.get("input_ids")
                if input_ids is None or int(input_ids.shape[0]) != 1:
                    raise BackendIntegrationError(
                        "Qwen3-VL MMTok requires one input_ids prompt batch."
                    )
                prompt_mask = input_ids[0] != self.core.config.image_token_id
                attention_mask = call.get("attention_mask")
                if attention_mask is not None and getattr(
                    attention_mask, "ndim", None
                ) == 2:
                    prompt_mask = prompt_mask & attention_mask[0].bool()
                prompt_ids = input_ids[0][prompt_mask]
                if int(prompt_ids.numel()) < 1:
                    raise BackendIntegrationError(
                        "Qwen3-VL MMTok found no prompt text tokens."
                    )
                self._prompt_text_embeddings.set(
                    self.core.get_input_embeddings()(prompt_ids)
                )
            return self._original_core_forward(*args, **kwargs)
        finally:
            self._prompt_text_embeddings.reset(text_embedding_token)
            self._image_count.reset(image_count_token)
            self._prompt_removal_mask.reset(removal_token)
            self._visual_plan.reset(plan_token)

    def _get_visual_features(
        self,
        _module: Any,
        pixel_values: Any,
        grid_thw: Any,
        *,
        modality: Modality,
    ) -> Any:
        torch = _torch()
        if grid_thw is None:
            raise BackendIntegrationError(
                f"Qwen3-VL {modality.value} compression requires grid_thw."
            )
        token = self._visual_modality.set(modality.value)
        if modality is Modality.IMAGE:
            self._image_count.set(int(grid_thw.shape[0]))
            self._comparison_grid.set(grid_thw)
            self._visual_plan.set(None)
        try:
            pixels = pixel_values.type(self.visual.dtype)
            embeds, deepstack = self.visual(pixels, grid_thw=grid_thw)
        finally:
            self._visual_modality.reset(token)
        dense_counts = tuple(
            int(value)
            for value in (
                grid_thw.prod(-1) // self.visual.spatial_merge_size**2
            ).tolist()
        )
        plan = self._visual_plan.get() if modality is Modality.IMAGE else None
        split_sizes = plan.result_counts if plan is not None else dense_counts
        if sum(split_sizes) != int(embeds.shape[0]):
            raise InvariantError(
                "Qwen3-VL visual result counts do not match projected features."
            )
        return torch.split(embeds, split_sizes), deepstack

    def _get_placeholder_mask(
        self,
        _module: Any,
        input_ids: Any,
        inputs_embeds: Any,
        *,
        image_features: Any = None,
        video_features: Any = None,
    ) -> Any:
        assert self._original_get_placeholder_mask is not None
        plan = self._visual_plan.get()
        if image_features is None or plan is None or not plan.compressed:
            return self._original_get_placeholder_mask(
                input_ids,
                inputs_embeds,
                image_features=image_features,
                video_features=video_features,
            )
        torch = _torch()
        if input_ids is None:
            raise BackendIntegrationError(
                "Compressed Qwen3-VL image forward currently requires input_ids "
                "so image placeholders remain unambiguous."
            )
        dense_image_mask = input_ids == self.core.config.image_token_id
        flat_positions = dense_image_mask.flatten().nonzero(as_tuple=False).flatten()
        if int(flat_positions.numel()) != sum(plan.source_counts):
            raise InvariantError(
                "Qwen3-VL dense image placeholder count does not match the "
                "compression source layout."
            )
        selected_offsets: list[int] = []
        offset = 0
        for count, delta in zip(plan.source_counts, plan.layout_deltas):
            if count != delta.source_count:
                raise InvariantError(
                    "Qwen3-VL image plan source count and LayoutDelta differ."
                )
            selected_offsets.extend(
                offset + index for index in _layout_representatives(delta)
            )
            offset += count
        selected_flat = torch.zeros(
            dense_image_mask.numel(),
            dtype=torch.bool,
            device=dense_image_mask.device,
        )
        offset_tensor = torch.tensor(
            selected_offsets,
            dtype=torch.long,
            device=flat_positions.device,
        )
        selected_flat[flat_positions.index_select(0, offset_tensor)] = True
        selected_image_mask = selected_flat.view_as(dense_image_mask)
        removal_mask = dense_image_mask & ~selected_image_mask
        self._prompt_removal_mask.set(removal_mask)
        image_mask = selected_image_mask.unsqueeze(-1).expand_as(inputs_embeds)
        if inputs_embeds[image_mask].numel() != image_features.numel():
            raise InvariantError(
                "Compressed Qwen3-VL image placeholders do not match features."
            )
        # Preserve provider-native video validation while disabling its dense
        # image cardinality assertion for this already legalized image branch.
        _, video_mask = self._original_get_placeholder_mask(
            input_ids,
            inputs_embeds,
            image_features=None,
            video_features=video_features,
        )
        return image_mask, video_mask

    def _visual_forward(self, _module: Any, *args: Any, **kwargs: Any) -> Any:
        assert self._original_visual_forward is not None
        tome = self._controllers(_SITE_TOME)
        visionzip = self._controllers(_SITE_VISIONZIP)
        if not tome and not visionzip:
            return self._original_visual_forward(*args, **kwargs)
        modality = self._visual_modality.get()
        if modality == Modality.VIDEO.value:
            return self._original_visual_forward(*args, **kwargs)
        if modality != Modality.IMAGE.value:
            raise BackendIntegrationError(
                "Qwen3-VL compressed visual.forward must be entered through "
                "get_image_features; direct calls cannot distinguish image/video."
            )
        hidden_states, grid_thw, provider_kwargs = _bind_visual_call(
            self._original_visual_forward,
            args,
            kwargs,
        )
        return self._execute_image_visual_forward(
            hidden_states,
            grid_thw,
            provider_kwargs,
            tome_controllers=tome,
            visionzip_controllers=visionzip,
        )

    def _execute_image_visual_forward(
        self,
        hidden_states: Any,
        grid_thw: Any,
        provider_kwargs: Mapping[str, Any],
        *,
        tome_controllers: Sequence[Any],
        visionzip_controllers: Sequence[Any],
    ) -> Any:
        torch = _torch()
        if int(grid_thw.shape[0]) < 1 or not bool((grid_thw[:, 0] == 1).all()):
            raise BackendIntegrationError(
                "The pinned Qwen3-VL image bridge currently requires T=1 per "
                "image; video is a separate support unit."
            )
        if int(grid_thw.shape[0]) != 1 and any(
            getattr(controller, "algorithm", None)
            in {"zoo_prune", "nuwa", "restore"}
            for controller in visionzip_controllers
        ):
            raise BackendIntegrationError(
                "Qwen3-VL ZOO-Prune/Nüwa/RESTORE supports exactly one image."
            )
        pre_projector_controllers = tuple(
            controller for controller in visionzip_controllers
            if any(site.startswith("pre_") for site in getattr(controller, "projector_target_sites", ()))
        )
        pre_tome_controllers = tuple(controller for controller in tome_controllers
            if getattr(controller, "output_target_site", "").startswith("pre_"))
        main_before_projector = any(
            controller.projector_target_sites[0] == "pre_final_merger"
            for controller in pre_projector_controllers
        ) or bool(pre_tome_controllers)
        deepstack_before_projector = any(
            controller.projector_target_sites[1] == "pre_deepstack_mergers"
            for controller in pre_projector_controllers
        ) or bool(pre_tome_controllers)
        visual = self.visual
        hidden_states = visual.patch_embed(hidden_states)
        pos_embeds = visual.fast_pos_embed_interpolate(grid_thw)
        hidden_states = hidden_states + pos_embeds
        rotary_pos_emb = visual.rot_pos_emb(grid_thw)
        sequence_length = int(hidden_states.shape[0])
        hidden_states = hidden_states.reshape(sequence_length, -1)
        rotary_pos_emb = rotary_pos_emb.reshape(sequence_length, -1)
        embedding = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
        position_embeddings = (embedding.cos(), embedding.sin())
        source_patch_counts = tuple(int(value) for value in grid_thw.prod(-1).tolist())
        current_counts = source_patch_counts
        cu_seqlens = _cumulative_lengths(
            current_counts,
            device=grid_thw.device,
        )
        token_sizes: tuple[Any | None, ...] = tuple(None for _ in current_counts)
        restore_steps: list[_ToMeStep] = []
        deepstack_features: list[Any] = []
        last_dense_hidden: Any | None = None
        visionzip_capture: _VisionAttentionCapture | None = None
        projector_requires_metrics = any(
            bool(getattr(controller, "requires_visual_metrics", False))
            for controller in visionzip_controllers
        )
        nuwa_controller = next(
            (
                controller
                for controller in visionzip_controllers
                if getattr(controller, "algorithm", None) == "nuwa"
            ),
            None,
        )
        nuwa_semantic_sum = None
        nuwa_semantic_count = 0
        nuwa_spec = getattr(nuwa_controller, "spec", None)
        if (
            nuwa_spec is not None
            and nuwa_spec.semantic_hidden_start <= 0 < nuwa_spec.semantic_hidden_end
        ):
            nuwa_semantic_sum = hidden_states.float()
            nuwa_semantic_count = 1

        tome_positions = {
            id(controller): tome_encoder_positions(controller.stage, "qwen3_vl", len(visual.blocks))
            for controller in tome_controllers
        }

        score_layers = {
            id(controller): tome_encoder_score_layer(controller.stage, "qwen3_vl", len(visual.blocks))
            for controller in tome_controllers
        }
        score_keys = {}

        def merge_at(position: LayerPosition, attention_keys: Any, layer_index: int) -> Any:
            nonlocal hidden_states, current_counts, token_sizes, position_embeddings, cu_seqlens
            source = position.index
            if position.boundary == "inter_layer" and source not in score_keys and any(
                position in tome_positions[id(c)] and score_layers[id(c)] == source for c in tome_controllers
            ):
                source_block = visual.blocks[source]
                values = source_block.norm1(hidden_states)
                qkv = source_block.attn.qkv(values).reshape(values.shape[0], 3, source_block.attn.num_heads, -1)
                _, reference = apply_vision_rotary(qkv[:, 0], qkv[:, 1], position_embeddings)
                score_keys[source] = reference.detach()
            hidden_chunks = torch.split(hidden_states, current_counts)
            key_chunks = torch.split(attention_keys, current_counts)
            for controller in tome_controllers:
                if position not in tome_positions[id(controller)]:
                    continue
                source = score_layers[id(controller)]
                scoring_chunks = key_chunks if source is None else torch.split(score_keys[source], current_counts)
                before_counts = current_counts
                before_sizes = token_sizes
                results = tuple(
                    controller.merge_encoder_tokens(
                        chunk,
                        key_chunk,
                        token_sizes=size,
                        layer_index=layer_index,
                        position=position,
                    )
                    for chunk, key_chunk, size in zip(
                        hidden_chunks,
                        scoring_chunks,
                        token_sizes,
                    )
                )
                deltas = tuple(result.layout_delta for result in results)
                hidden_chunks = tuple(result.hidden_states for result in results)
                token_sizes = tuple(result.token_sizes for result in results)
                after_counts = tuple(
                    int(result.hidden_states.shape[0]) for result in results
                )
                restore_steps.append(
                    _ToMeStep(
                        controller=controller,
                        deltas=deltas,
                        before_counts=before_counts,
                        after_counts=after_counts,
                    )
                )
                hidden_states = torch.cat(hidden_chunks, dim=0)
                position_embeddings = tuple(
                    _select_layout_representatives(
                        value,
                        deltas,
                        before_counts,
                    )
                    for value in position_embeddings
                )
                current_counts = after_counts
                cu_seqlens = _cumulative_lengths(
                    current_counts,
                    device=grid_thw.device,
                )
                key_chunks = tuple(
                    tome_merge_reference(
                        key_chunk,
                        delta,
                        token_sizes=size,
                    )
                    for key_chunk, delta, size in zip(
                        key_chunks,
                        deltas,
                        before_sizes,
                    )
                )
                for source, reference in score_keys.items():
                    score_keys[source] = torch.cat([
                        tome_merge_reference(chunk, delta, token_sizes=size)
                        for chunk, delta, size in zip(torch.split(reference, before_counts), deltas, before_sizes)
                    ], dim=0)
                self._invocation_counts[_SITE_TOME] += len(results)
            return torch.cat(key_chunks, dim=0)

        initial = LayerPosition("inter_layer", 0)
        if any(initial in positions for positions in tome_positions.values()):
            block = visual.blocks[0]
            normalized = block.norm1(hidden_states)
            qkv = block.attn.qkv(normalized).reshape(normalized.shape[0], 3, block.attn.num_heads, -1)
            _, keys = apply_vision_rotary(qkv[:, 0], qkv[:, 1], position_embeddings)
            if 0 in score_layers.values():
                score_keys[0] = keys.detach()
            merge_at(initial, keys, 0)

        for layer_index, block in enumerate(visual.blocks):
            capture_final_attention = bool(
                tome_controllers or projector_requires_metrics
            ) and (layer_index == len(visual.blocks) - 1)
            if tome_controllers or capture_final_attention:
                # Preserve native residual branches. The selected boundary
                # determines whether this block's MLP sees compact tokens.
                normalised = block.norm1(hidden_states)
                attention_output, _, attention_keys, received = (
                    execute_vision_attention(
                        block.attn,
                        normalised,
                        counts=current_counts,
                        position_embeddings=position_embeddings,
                        token_sizes=token_sizes,
                        capture_mean_received=capture_final_attention,
                    )
                )
                if layer_index in score_layers.values() and layer_index not in score_keys:
                    score_keys[layer_index] = attention_keys.detach()
                hidden_states = hidden_states + attention_output

                if capture_final_attention:
                    if received is None:
                        raise InvariantError(
                            "Qwen3-VL final vision attention capture is missing."
                        )
                    visionzip_capture = _VisionAttentionCapture(
                        counts=current_counts,
                        mean_received=received,
                        keys=tuple(torch.split(attention_keys, tuple(current_counts))),
                        token_sizes=token_sizes,
                        # The final block's own ToMe merge occurs after this
                        # attention call and must not be part of its input map.
                        restore_steps=tuple(restore_steps),
                    )

            if tome_controllers:
                for position in (LayerPosition("post_attention_pre_ffn", layer_index), LayerPosition("inter_layer", layer_index + 1)):
                    if position.boundary == "inter_layer":
                        hidden_states = hidden_states + block.mlp(block.norm2(hidden_states))
                    attention_keys = merge_at(position, attention_keys, layer_index)
            elif capture_final_attention:
                hidden_states = hidden_states + block.mlp(block.norm2(hidden_states))
            else:
                hidden_states = block(
                    hidden_states,
                    cu_seqlens=cu_seqlens,
                    position_embeddings=position_embeddings,
                    **dict(provider_kwargs),
                )

            if nuwa_spec is not None and (
                nuwa_spec.semantic_hidden_start
                <= layer_index + 1
                < nuwa_spec.semantic_hidden_end
            ):
                value = hidden_states.float()
                nuwa_semantic_sum = (
                    value if nuwa_semantic_sum is None else nuwa_semantic_sum + value
                )
                nuwa_semantic_count += 1

            if layer_index in visual.deepstack_visual_indexes:
                dense_hidden = _restore_tome_steps(
                    hidden_states,
                    current_counts,
                    restore_steps,
                )
                merger_index = visual.deepstack_visual_indexes.index(layer_index)
                # Pre-projector decisions use intact spatial groups from this
                # tap, before its own normalization and projection.
                deepstack_features.append(
                    dense_hidden.reshape(-1, int(dense_hidden.shape[-1]) * int(visual.spatial_merge_unit))
                    if deepstack_before_projector else
                    visual.deepstack_merger_list[merger_index](dense_hidden)
                )
            last_dense_hidden = _restore_tome_steps(
                hidden_states,
                current_counts,
                restore_steps,
            )

        assert last_dense_hidden is not None
        merge_unit = int(visual.spatial_merge_unit)
        # Match the native merger input by concatenating every spatial group;
        # an average would alias structurally different visual groups.
        vision_reference = last_dense_hidden.reshape(
            int(last_dense_hidden.shape[0]) // merge_unit,
            -1,
        )
        merged_hidden = vision_reference if main_before_projector else visual.merger(last_dense_hidden)
        projected_reference = None
        if main_before_projector and any(
            getattr(controller, "algorithm", None) in {"mmtok", "zoo_prune"}
            for controller in visionzip_controllers
        ):
            # These methods retain their original projected-space scoring.
            # Only the compact value projection contributes to the value graph.
            with torch.no_grad():
                projected_reference = visual.merger(last_dense_hidden)

        if nuwa_controller is not None:
            if nuwa_semantic_sum is None or nuwa_semantic_count < 1:
                raise BackendIntegrationError(
                    "Qwen3-VL Nüwa semantic hidden-state range was not captured."
                )
            semantic_reference = (
                nuwa_semantic_sum / float(nuwa_semantic_count)
            ).reshape(int(last_dense_hidden.shape[0]) // merge_unit, -1)
        else:
            semantic_reference = vision_reference
        projector_inputs = None
        projector_layers = None
        if any(
            getattr(controller, "algorithm", None) == "zoo_prune"
            for controller in visionzip_controllers
        ):
            merger = visual.merger
            hidden_size = int(merger.hidden_size)
            norm_device = getattr(getattr(merger.norm, "_hf_hook", None), "execution_device", None)
            if norm_device is None:
                norm_device = getattr(merger.norm, "weight", last_dense_hidden).device
            projector_device = getattr(getattr(merger.linear_fc1, "_hf_hook", None), "execution_device", None)
            if projector_device is None:
                projector_device = merger.linear_fc1.weight.device
            dense_projector_inputs = last_dense_hidden.to(device=norm_device)
            if bool(getattr(merger, "use_postshuffle_norm", False)):
                projector_inputs = merger.norm(
                    dense_projector_inputs.reshape(-1, hidden_size)
                )
            else:
                projector_inputs = merger.norm(dense_projector_inputs).reshape(
                    -1, hidden_size
                )
            projector_inputs = projector_inputs.to(device=projector_device)
            projector_layers = (
                merger.linear_fc1,
                merger.act_fn,
                merger.linear_fc2,
            )
        source_feature_counts = tuple(
            count // merge_unit for count in source_patch_counts
        )
        if not tome_controllers and not visionzip_controllers:
            identity_deltas = tuple(
                LayoutDelta.identity(count, media_unit="image_feature")
                for count in source_feature_counts
            )
            self._visual_plan.set(
                _VisualCompressionPlan(
                    source_counts=source_feature_counts,
                    result_counts=source_feature_counts,
                    layout_deltas=identity_deltas,
                )
            )
            return merged_hidden, deepstack_features

        if visionzip_capture is None:
            if tome_controllers or projector_requires_metrics:
                raise InvariantError(
                    "Qwen3-VL did not capture required final vision metrics."
                )
            attention_chunks = [None] * len(source_feature_counts)
            key_chunks = [None] * len(source_feature_counts)
        else:
            attention_chunks, key_chunks = _visionzip_metrics_from_capture(
                visionzip_capture,
                patch_counts=source_patch_counts,
                spatial_merge_unit=int(visual.spatial_merge_unit),
            )
        main_chunks = list(torch.split(merged_hidden, source_feature_counts))
        projected_reference_chunks = (
            list(torch.split(projected_reference, source_feature_counts))
            if projected_reference is not None else None
        )
        vision_reference_chunks = list(
            torch.split(vision_reference, source_feature_counts)
        )
        semantic_reference_chunks = list(
            torch.split(semantic_reference, source_feature_counts)
        )
        feature_grids = tuple(
            (
                int(row[1].item()) // int(visual.spatial_merge_size),
                int(row[2].item()) // int(visual.spatial_merge_size),
            )
            for row in grid_thw
        )
        projector_input_chunks = (
            list(torch.split(projector_inputs, source_feature_counts))
            if projector_inputs is not None
            else [None] * len(source_feature_counts)
        )
        deep_chunks_by_branch = [
            list(torch.split(branch, source_feature_counts))
            for branch in deepstack_features
        ]
        cumulative_deltas = [
            LayoutDelta.identity(count, media_unit="image_feature")
            for count in source_feature_counts
        ]
        def apply_projector_controllers(controllers):
            for controller in controllers:
                for image_index in range(len(main_chunks)):
                    branch_inputs = tuple(
                        chunks[image_index] for chunks in deep_chunks_by_branch
                    )
                    if getattr(controller, "algorithm", None) == "mmtok":
                        result = controller.compress_after_visual_mergers(
                            main_chunks[image_index],
                            branch_inputs,
                            vision_reference=vision_reference_chunks[image_index],
                            text_embeddings=self._prompt_text_embeddings.get(),
                            **({"projected_reference": projected_reference_chunks[image_index]}
                               if projected_reference_chunks is not None else {}),
                            image_index=image_index,
                        )
                    elif getattr(controller, "algorithm", None) == "vispruner":
                        result = controller.compress_after_visual_mergers(
                            main_chunks[image_index],
                            branch_inputs,
                            attention=attention_chunks[image_index],
                            vision_reference=vision_reference_chunks[image_index],
                            image_index=image_index,
                        )
                    elif getattr(controller, "algorithm", None) == "restore":
                        result = controller.compress_after_visual_mergers(
                            main_chunks[image_index],
                            branch_inputs,
                            attention=attention_chunks[image_index],
                            vision_reference=vision_reference_chunks[image_index],
                            image_index=image_index,
                        )
                    elif getattr(controller, "algorithm", None) == "zoo_prune":
                        result = controller.compress_after_visual_mergers(
                            main_chunks[image_index],
                            branch_inputs,
                            projector_inputs=projector_input_chunks[image_index],
                            projector_layers=projector_layers,
                            **({"projected_reference": projected_reference_chunks[image_index]}
                               if projected_reference_chunks is not None else {}),
                            image_index=image_index,
                        )
                    elif getattr(controller, "algorithm", None) == "nuwa":
                        result = controller.compress_after_visual_mergers(
                            main_chunks[image_index],
                            branch_inputs,
                            attention=attention_chunks[image_index],
                            keys=key_chunks[image_index],
                            semantic_reference=semantic_reference_chunks[image_index],
                            feature_grid=feature_grids[image_index],
                            image_index=image_index,
                        )
                    else:
                        metric_keys = key_chunks[image_index]
                        if getattr(controller, "algorithm", None) == "saint":
                            # Capture storage is heads-first; SAINT's kernel is
                            # tokens-first, including when both counts match.
                            metric_keys = metric_keys.permute(1, 0, 2).contiguous()
                        result = controller.compress_after_visual_mergers(
                            main_chunks[image_index],
                            branch_inputs,
                            attention=attention_chunks[image_index],
                            keys=metric_keys,
                            image_index=image_index,
                        )
                    main_chunks[image_index] = result.main_image_features
                    if getattr(controller, "algorithm", None) == "restore":
                        runtime = self._restore_runtime.get()
                        if runtime is None:
                            raise InvariantError(
                                "Qwen3-VL RESTORE visual merge ran outside its request lifecycle."
                            )
                        runtime["visual_sizes"] = result.kernel_result.token_sizes
                        runtime["source_visual_count"] = int(
                            result.layout_delta.source_count
                        )
                    for branch_index, value in enumerate(result.deepstack_image_features):
                        deep_chunks_by_branch[branch_index][image_index] = value
                    if getattr(controller, "algorithm", None) == "restore":
                        if not cumulative_deltas[image_index].is_identity:
                            raise InvariantError(
                                "Qwen3-VL RESTORE requires an unmodified source layout."
                            )
                        cumulative_deltas[image_index] = result.layout_delta
                    else:
                        cumulative_deltas[image_index] = _compose_layout_mapping(
                            cumulative_deltas[image_index],
                            result.layout_delta,
                        )
                    if attention_chunks[image_index] is not None:
                        attention_chunks[image_index] = _mean_square_by_delta(
                            attention_chunks[image_index],
                            result.layout_delta,
                        )
                    if key_chunks[image_index] is not None:
                        key_chunks[image_index] = _mean_keys_by_delta(
                            key_chunks[image_index],
                            result.layout_delta,
                        )
                    self._invocation_counts[_SITE_VISIONZIP] += 1
        def apply_output_tome(controllers: Sequence[Any]) -> None:
            for controller in controllers:
                for image_index in range(len(main_chunks)):
                    result = controller.merge_output_tokens(
                        main_chunks[image_index],
                        key_chunks[image_index].permute(1, 0, 2).contiguous(),
                    )
                    delta = result.layout_delta
                    main_chunks[image_index] = result.hidden_states
                    for branch_chunks in deep_chunks_by_branch:
                        output_rules = {name.removeprefix("output_"): rule
                            for name, rule in getattr(controller, "_rules", {}).items() if name.startswith("output_")}
                        branch_chunks[image_index] = (replay_tome_image(
                            branch_chunks[image_index], result, rules=output_rules) if output_rules
                            else _merge_values_by_tome_groups(branch_chunks[image_index], result.sequence_source_groups))
                    if any(
                        getattr(item, "algorithm", None) == "vispruner"
                        and item not in pre_projector_controllers
                        for item in visionzip_controllers
                    ):
                        # The fixed final-vision source must follow this existing
                        # output merge before a later VisPruner consumes it.
                        vision_reference_chunks[image_index] = _merge_values_by_tome_groups(
                            vision_reference_chunks[image_index], result.sequence_source_groups,
                        )
                    cumulative_deltas[image_index] = _compose_layout_mapping(
                        cumulative_deltas[image_index],
                        delta,
                    )
                    attention_chunks[image_index] = _mean_square_by_delta(
                        attention_chunks[image_index],
                        delta,
                    )
                    key_chunks[image_index] = _mean_keys_by_delta(
                        key_chunks[image_index],
                        delta,
                    )
                    self._invocation_counts[_SITE_TOME] += 1

        apply_projector_controllers(pre_projector_controllers)
        apply_output_tome(pre_tome_controllers)
        if main_before_projector:
            main_chunks = [visual.merger(chunk.reshape(-1, int(last_dense_hidden.shape[-1])))
                           for chunk in main_chunks]
        if deepstack_before_projector:
            deep_chunks_by_branch = [
                [merger(chunk.reshape(-1, int(last_dense_hidden.shape[-1]))) for chunk in chunks]
                for merger, chunks in zip(visual.deepstack_merger_list, deep_chunks_by_branch)
            ]
        apply_output_tome(tuple(controller for controller in tome_controllers if controller not in pre_tome_controllers))
        apply_projector_controllers(tuple(
            controller for controller in visionzip_controllers
            if controller not in pre_projector_controllers
        ))
        result_counts = tuple(int(chunk.shape[0]) for chunk in main_chunks)
        self._visual_plan.set(
            _VisualCompressionPlan(
                source_counts=source_feature_counts,
                result_counts=result_counts,
                layout_deltas=tuple(cumulative_deltas),
            )
        )
        return (
            torch.cat(main_chunks, dim=0),
            [torch.cat(chunks, dim=0) for chunks in deep_chunks_by_branch],
        )

    def _language_forward(self, _module: Any, *args: Any, **kwargs: Any) -> Any:
        assert self._original_language_forward is not None
        decoder_controllers = self._controllers(_SITE_PYRAMID)
        removal_mask = self._prompt_removal_mask.get()
        if not decoder_controllers and removal_mask is None:
            return self._original_language_forward(*args, **kwargs)
        call = _bind_language_call(
            self._original_language_forward,
            args,
            kwargs,
        )
        use_cache = bool(call["use_cache"])
        lifecycle = self._decoder_lifecycle.get()
        if lifecycle is None and (use_cache or call["past_key_values"] is not None):
            raise BackendIntegrationError(
                "Qwen3-VL provider bridge supports only FORWARD with "
                "use_cache=False and no past_key_values."
            )
        if lifecycle is not None:
            expected_cache = (
                self._training_labels.get() is None
                and (
                    call["past_key_values"] is not None
                    or ExecutionPhase(decoder_controllers[0].execution_phase)
                    is ExecutionPhase.GENERATE
                )
            )
            if use_cache != expected_cache:
                raise BackendIntegrationError(
                    "Qwen3-VL decoder lifecycle received an inconsistent cache mode."
                )
        if removal_mask is not None and bool(removal_mask.any().item()):
            labels = self._training_labels.get()
            if labels is not None:
                keep = (~removal_mask[0]).nonzero(as_tuple=False).flatten()
                self._training_labels.set(
                    compact_ignored_labels(
                        labels,
                        keep,
                        source_length=int(removal_mask.shape[-1]),
                        operation="Qwen3-VL prompt compaction",
                    )
                )
        call = _compact_prompt_call(call, removal_mask)
        if not decoder_controllers:
            return self._original_language_forward(**call)
        visual_mask = call.get("visual_pos_masks")
        if (visual_mask is None or not bool(visual_mask.any().item())) and call[
            "past_key_values"
        ] is None:
            return self._original_language_forward(**call)
        if lifecycle is None:
            raise BackendIntegrationError(
                "Qwen3-VL decoder compression requires a guarded request lifecycle."
            )
        return self._execute_decoder_loop(
            call,
            decoder_controllers,
            lifecycle=lifecycle,
            image_count=self._image_count.get(),
        )

    def _execute_decoder_loop(
        self,
        call: dict[str, Any],
        controllers: Sequence[Any],
        *,
        lifecycle: ImageDecoderLifecycle | None = None,
        image_count: int | None = None,
    ) -> Any:
        torch = _torch()
        if len(controllers) != 1:
            raise BackendIntegrationError(
                "Qwen3-VL decoder loop requires exactly one image method."
            )
        controller = controllers[0]
        decoder_algorithm = str(controller.algorithm)
        selection_method = getattr(controller, "decoder_selection", None)
        use_contiguous_positions = selection_method is not None and selection_method.position_policy == "contiguous"
        layer_local_contiguous = use_contiguous_positions and selection_method.cache_policy is CacheRepairPolicy.LAYER_LOCAL
        compact_contiguous = use_contiguous_positions and selection_method.cache_policy is CacheRepairPolicy.COMPACT_PREFIX
        if selection_method is None and decoder_algorithm not in {
            "pyramid_drop",
            "sgl",
            "nuwa",
            "restore",
        }:
            raise BackendIntegrationError(
                f"Unsupported Qwen3-VL decoder algorithm {decoder_algorithm!r}."
            )
        BaseModelOutputWithPast, create_causal_mask = _transformers_decoder_helpers()
        input_ids = call.pop("input_ids")
        inputs_embeds = call.pop("inputs_embeds")
        attention_mask = call.pop("attention_mask")
        position_ids = call.pop("position_ids")
        past_key_values = call.pop("past_key_values")
        use_cache = bool(call.pop("use_cache"))
        cache_position = call.pop("cache_position")
        visual_pos_masks = call.pop("visual_pos_masks")
        deepstack_visual_embeds = call.pop("deepstack_visual_embeds")
        if deepstack_visual_embeds is not None:
            deepstack_visual_embeds = list(deepstack_visual_embeds)
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError(
                "You must specify exactly one of input_ids or inputs_embeds"
            )
        if inputs_embeds is None:
            inputs_embeds = self.language_model.embed_tokens(input_ids)
        if int(inputs_embeds.shape[0]) != 1:
            raise BackendIntegrationError(
                "Pinned Qwen3-VL decoder compression currently supports batch "
                "size one so each LayoutDelta remains exact."
            )
        if use_cache and past_key_values is None:
            past_key_values = _new_dynamic_cache(self.language_model.config)
        if cache_position is None:
            cache_position = torch.arange(
                inputs_embeds.shape[1],
                device=inputs_embeds.device,
            )
        if position_ids is None:
            position_ids = cache_position.view(1, 1, -1).expand(
                3,
                inputs_embeds.shape[0],
                -1,
            )
        elif position_ids.ndim == 2:
            position_ids = position_ids[None, ...].expand(
                3,
                position_ids.shape[0],
                -1,
            )
        if position_ids.ndim != 3 or int(position_ids.shape[0]) not in (3, 4):
            raise BackendIntegrationError(
                "Qwen3-VL decoder compression requires 3-plane M-RoPE positions or "
                "the provider's packed 4-plane representation."
            )
        if int(position_ids.shape[-1]) != int(inputs_embeds.shape[1]):
            raise InvariantError("Qwen3-VL positions do not match decoder sequence.")
        if int(position_ids.shape[0]) == 4:
            text_position_ids = position_ids[0]
            rotary_position_ids = position_ids[1:]
        else:
            text_position_ids = position_ids[0]
            rotary_position_ids = position_ids
        padding_mask = attention_mask
        attention_mask = create_causal_mask(
            config=self.language_model.config,
            input_embeds=inputs_embeds,
            attention_mask=attention_mask,
            cache_position=cache_position,
            past_key_values=None,
            position_ids=text_position_ids,
        )
        if attention_mask is not None and not hasattr(attention_mask, "index_select"):
            raise BackendIntegrationError(
                "Pinned Qwen3-VL decoder compression cannot compact this provider "
                "attention-mask representation."
            )
        owned_lifecycle = lifecycle is None
        lifecycle = lifecycle or self._new_decoder_lifecycle(prefix="forward")
        decoder_call = lifecycle.begin_call(
            query_length=int(inputs_embeds.shape[1]),
            use_cache=use_cache,
            past_key_values=past_key_values,
        )
        layer_local_native_rotary_positions = (
            rotary_position_ids
            if decoder_call.phase is DecoderCallPhase.DECODE
            and layer_local_contiguous
            else None
        )
        try:
            if (
                decoder_call.phase is not DecoderCallPhase.DECODE
                and ((selection_method is not None and selection_method.single_image_only) or decoder_algorithm in {
                    "sgl",
                    "nuwa",
                    "restore",
                })
                and image_count != 1
            ):
                raise BackendIntegrationError(
                    f"Qwen3-VL {decoder_algorithm} is single-image only; "
                    f"received {image_count!r} images."
                )
            if decoder_call.phase is DecoderCallPhase.DECODE:
                if visual_pos_masks is not None or deepstack_visual_embeds is not None:
                    raise BackendIntegrationError(
                        "Qwen3-VL decode must not replay visual or DeepStack inputs."
                    )
                if compact_contiguous:
                    compact_start = dynamic_cache_layer_length(past_key_values, 0)
                    compact_positions = torch.arange(
                        compact_start,
                        compact_start + int(inputs_embeds.shape[1]),
                        dtype=position_ids.dtype,
                        device=inputs_embeds.device,
                    )
                    position_ids = compact_positions.view(1, 1, -1).expand(
                        int(position_ids.shape[0]),
                        int(inputs_embeds.shape[0]),
                        -1,
                    )
                    text_position_ids = position_ids[0]
                    rotary_position_ids = (
                        position_ids[1:]
                        if int(position_ids.shape[0]) == 4
                        else position_ids
                    )
            elif visual_pos_masks is None or not bool(visual_pos_masks.any().item()):
                raise BackendIntegrationError(
                    "Qwen3-VL compressed prefill requires exact visual positions."
                )
            hidden_states = inputs_embeds
            position_embeddings = self.language_model.rotary_emb(
                hidden_states,
                rotary_position_ids,
            )
            restore_attention = None
            if decoder_algorithm == "restore":
                runtime = self._restore_runtime.get()
                if runtime is None:
                    raise InvariantError(
                        "Qwen3-VL RESTORE decoder ran outside its request lifecycle."
                    )
                if decoder_call.phase is DecoderCallPhase.DECODE:
                    prior = runtime.get("attention_state")
                    if prior is None:
                        raise InvariantError(
                            "RESTORE decode has no calibrated prefill state."
                        )
                    restore_attention = prior.append(
                        rotary_position_ids, reference=hidden_states
                    )
                else:
                    assert visual_pos_masks is not None
                    restore_attention = RestoreAttentionState.prefill(
                        rotary_position_ids,
                        image_indices=visual_pos_masks[0]
                        .bool()
                        .nonzero(as_tuple=False)
                        .flatten(),
                        visual_sizes=runtime["visual_sizes"],
                        source_visual_count=runtime["source_visual_count"],
                        first_layer_size=controller.spec.first_layer_size,
                        reference=hidden_states,
                    )
                runtime["attention_state"] = restore_attention
            restore_layers_calibrated = 0
            selection = selection_method.new_call() if selection_method is not None else None
            if selection is not None and decoder_call.phase is not DecoderCallPhase.DECODE:
                if callable(getattr(selection, "set_visual_layout", None)):
                    from mllm_comp.compressors.comparison.native_decoder import qwen_visual_segments
                    selection.set_visual_layout(qwen_visual_segments(self._comparison_grid.get(), self.visual.spatial_merge_size))
                selection.initialize(inputs_embeds[0], visual_pos_masks[0].bool(),
                    padding_mask=padding_mask, labels=self._training_labels.get())
            boundary_replay = None
            mid_deepstack_keep = None
            application_positions = (
                selection_method.positions if selection_method is not None else ()
            )
            def apply_pre_layer_compression(layer_index, *, mid=False):
                nonlocal boundary_replay
                nonlocal hidden_states, attention_mask, padding_mask, position_ids, cache_position
                nonlocal visual_pos_masks, text_position_ids, rotary_position_ids, position_embeddings
                nonlocal mid_deepstack_keep
                point = LayerPosition("post_attention_pre_ffn" if mid else "inter_layer", layer_index)
                if decoder_call.phase is DecoderCallPhase.DECODE or point not in application_positions:
                    return None
                assert visual_pos_masks is not None
                source_image_indices = (
                    visual_pos_masks[0].bool().nonzero(as_tuple=False).flatten()
                )
                if int(source_image_indices.numel()) == 0:
                    return None
                result = controller.apply_decoder_selection(
                    selection,
                    point,
                    hidden_states[0],
                    image_indices=source_image_indices,
                )
                kept = torch.tensor(
                    layout_result_representatives(result.layout_delta),
                    dtype=torch.long,
                    device=hidden_states.device,
                )
                visual_representatives = kept[torch.isin(kept, source_image_indices.to(kept.device))]
                local_keep = torch.searchsorted(source_image_indices, visual_representatives.to(source_image_indices.device))
                boundary_replay = decoder_layout_replay(result.layout_delta, rules=getattr(controller, "_rules", {}))
                lifecycle.record_boundary(
                    layer_index=min(layer_index, len(self.language_model.layers) - 1),
                    boundary=DecoderBoundary.POST_ATTENTION_PRE_FFN if mid else (DecoderBoundary.AFTER_LAYER if layer_index == len(self.language_model.layers) else DecoderBoundary.BEFORE_LAYER),
                    layout_delta=result.layout_delta,
                    past_key_values=past_key_values,
                    cache_policy=selection_method.cache_policy if selection_method is not None else CacheRepairPolicy.COMPACT_PREFIX,
                    prefix_layer_count=layer_index + int(mid) if use_cache else 0,
                )
                if deepstack_visual_embeds is not None:
                    if result.layout_delta.result_layout.get("weighted_replay_rows") is not None:
                        result_image_rows = visual_pos_masks[0].index_select(0, kept.to(visual_pos_masks.device)).nonzero(as_tuple=True)[0]
                        source_count = result.layout_delta.source_count
                        def update_branch(branch):
                            dense = branch.new_zeros((1, source_count, branch.shape[-1]))
                            dense = dense.index_copy(1, source_image_indices.to(branch.device), branch[None])
                            return boundary_replay(dense)[0].index_select(0, result_image_rows.to(branch.device))
                    elif result.layout_delta.merge_groups:
                        groups = layout_merge_groups(result.layout_delta, source_image_indices)
                        full_groups = layout_merge_groups(result.layout_delta)
                        result_images = sorted({result.layout_delta.old_to_new[int(i)] for i in source_image_indices.tolist()
                                                if result.layout_delta.old_to_new[int(i)] is not None})
                        update_branch = partial(replay_rule_groups, groups=groups,
                            source_groups=tuple(full_groups[i] for i in result_images),
                            rules=getattr(controller, "_rules", {}))
                    else:
                        update_branch = lambda branch: branch.index_select(0, local_keep.to(branch.device))
                    if mid:
                        mid_deepstack_keep = update_branch
                    else:
                        for branch_index in range(layer_index, len(deepstack_visual_embeds)):
                            deepstack_visual_embeds[branch_index] = update_branch(deepstack_visual_embeds[branch_index])
                hidden_states = result.hidden_states.unsqueeze(0)
                position_ids = position_ids.index_select(
                    -1, kept.to(position_ids.device)
                )
                visual_pos_masks = visual_pos_masks.index_select(
                    -1, kept.to(visual_pos_masks.device)
                )
                if padding_mask is not None:
                    padding_mask = padding_mask.index_select(
                        -1,
                        kept.to(padding_mask.device),
                    )
                cache_position = torch.arange(
                    hidden_states.shape[1], device=hidden_states.device
                )
                if use_contiguous_positions:
                    position_ids = cache_position.to(position_ids.dtype).view(1, 1, -1).expand(int(position_ids.shape[0]), int(hidden_states.shape[0]), -1)
                if int(position_ids.shape[0]) == 4:
                    text_position_ids = position_ids[0]
                    rotary_position_ids = position_ids[1:]
                else:
                    text_position_ids = position_ids[0]
                    rotary_position_ids = position_ids
                attention_mask = create_causal_mask(
                    config=self.language_model.config,
                    input_embeds=hidden_states,
                    attention_mask=padding_mask,
                    cache_position=cache_position,
                    past_key_values=None,
                    position_ids=text_position_ids,
                )
                position_embeddings = self.language_model.rotary_emb(
                    hidden_states, rotary_position_ids
                )
                self._invocation_counts[_SITE_PYRAMID] += 1
                return kept.detach()
            def apply_post_attention(value, layer_index):
                nonlocal hidden_states
                hidden_states = value
                kept = apply_pre_layer_compression(layer_index, mid=True)
                return hidden_states, boundary_replay if kept is not None else lambda values: values
            def read_selection_attention(capture, image_indices):
                metric_layer = self.language_model.layers[capture.projection_layer]
                query, image_keys, attention_keys = _decoder_query_keys(metric_layer, hidden_states,
                    instruction_indices=capture.query_indices, image_indices=image_indices,
                    position_embeddings=position_embeddings, use_rope=capture.use_rope)
                visibility = _attention_key_visibility(attention_mask,
                    instruction_indices=capture.query_indices, key_count=int(attention_keys.shape[0]), padding_mask=padding_mask)
                return query, image_keys, attention_keys, visibility

            from mllm_comp.compressors.comparison.native_decoder import observe_comparison_input, layer_context
            for layer_index, decoder_layer in enumerate(self.language_model.layers):
                if selection is not None and decoder_call.phase is not DecoderCallPhase.DECODE:
                    observe_comparison_input(selection, layer_index, hidden_states, self.language_model.layers,
                        attention_mask=attention_mask, position_ids=text_position_ids,
                        position_embeddings=position_embeddings, cache_position=cache_position,
                        deepstack_visual_embeds=deepstack_visual_embeds, visual_pos_masks=visual_pos_masks)
                # Earlier stages at this entry run before a later stage samples this layer's input.
                apply_before_capture = selection.select_before_input(layer_index) if selection is not None else False
                if apply_before_capture:
                    apply_pre_layer_compression(layer_index)
                if selection is not None and decoder_call.phase is not DecoderCallPhase.DECODE:
                    image_row = visual_pos_masks[0].bool()
                    capture = selection.capture_input(layer_index, image_row,
                        padding_mask=padding_mask, labels=self._training_labels.get())
                    if capture is not None:
                        capture_decoder_attention(selection, capture, image_row.nonzero(as_tuple=False).flatten(), read_selection_attention)
                if not apply_before_capture:
                    apply_pre_layer_compression(layer_index)
                layer_cache_position = cache_position
                layer_text_position_ids = text_position_ids
                layer_position_embeddings = position_embeddings
                if decoder_call.phase is DecoderCallPhase.DECODE:
                    cache_length = dynamic_cache_layer_length(
                        past_key_values, layer_index
                    )
                    layer_cache_position = torch.arange(
                        cache_length,
                        cache_length + int(hidden_states.shape[1]),
                        device=hidden_states.device,
                    )
                    if layer_local_contiguous:
                        assert layer_local_native_rotary_positions is not None
                        layer_text_position_ids = layer_cache_position.to(
                            position_ids.dtype
                        ).view(1, -1).expand(int(hidden_states.shape[0]), -1)
                        if layer_index < (application_positions[0].index + int(application_positions[0].boundary == "post_attention_pre_ffn")):
                            layer_rotary_position_ids = (
                                layer_local_native_rotary_positions
                            )
                            rotary_policy = "native_mrope"
                        else:
                            layer_rotary_position_ids = layer_cache_position.to(
                                position_ids.dtype
                            ).view(1, 1, -1).expand(
                                3,
                                int(hidden_states.shape[0]),
                                -1,
                            )
                            rotary_policy = "contiguous_physical"
                        lifecycle.record_layer_local_decode_position(
                            layer_index=layer_index,
                            cache_position=layer_cache_position,
                            text_position_ids=layer_text_position_ids,
                            rotary_position_ids=layer_rotary_position_ids,
                            expected_rotary_position_ids=layer_rotary_position_ids,
                            rotary_policy=rotary_policy,
                        )
                        layer_position_embeddings = self.language_model.rotary_emb(
                            hidden_states,
                            layer_rotary_position_ids,
                        )
                    local_padding = _layer_local_decode_padding_mask(
                        padding_mask,
                        past_key_values,
                        layer_index=layer_index,
                        query_length=int(hidden_states.shape[1]),
                        reference=hidden_states,
                    )
                    attention_mask = create_causal_mask(
                        config=self.language_model.config,
                        input_embeds=hidden_states,
                        attention_mask=local_padding,
                        cache_position=layer_cache_position,
                        past_key_values=LayerLocalCacheView(
                            past_key_values, layer_index
                        ),
                        position_ids=layer_text_position_ids,
                    )
                layer_input_hidden = hidden_states
                layer_input_position_embeddings = layer_position_embeddings
                layer_attention_mask = attention_mask
                if decoder_algorithm == "restore":
                    assert restore_attention is not None
                    layer_attention_mask = restore_attention.attention_mask(
                        layer_attention_mask,
                        rotary_position_ids,
                        self.language_model.rotary_emb.inv_freq,
                        layer_index=layer_index,
                        reference=hidden_states,
                        calibration_constant=controller.spec.calibration_constant,
                        mrope_section=_restore_mrope_section(
                            self.language_model.config,
                            axis_count=int(rotary_position_ids.shape[0]),
                        ),
                        rules=getattr(controller, "_rules", {}),
                    )
                    restore_layers_calibrated += 1
                if selection is not None:
                    layer_attention_mask = selection.prepare_attention_mask(layer_attention_mask,
                        query_length=int(hidden_states.shape[1]), reference=hidden_states)
                capture_context = nullcontext(None)
                if (selection is not None and decoder_call.phase is not DecoderCallPhase.DECODE
                        and layer_index in selection.transition_layers):
                    image_indices = visual_pos_masks[0].bool().nonzero(as_tuple=False).flatten()
                    if selection.begin_transition(layer_index, image_indices):
                        capture_context = capture_decoder_module_transitions(decoder_layer)
                with capture_context as transition_observation:
                    layer_forward = decoder_layer
                    if (decoder_call.phase is not DecoderCallPhase.DECODE
                            and LayerPosition("post_attention_pre_ffn", layer_index) in application_positions):
                        layer_forward = partial(decoder_post_attention_boundary, decoder_layer,
                            apply_compression=partial(apply_post_attention, layer_index=layer_index))
                    hidden_states = layer_forward(
                        hidden_states,
                        attention_mask=layer_attention_mask,
                        position_ids=layer_text_position_ids,
                        past_key_values=past_key_values,
                        cache_position=layer_cache_position,
                        position_embeddings=layer_position_embeddings,
                        **call,
                        **layer_context(selection),
                    )
                if mid_deepstack_keep is not None:
                    # Branch values stay outside the discrete checkpoint decision.
                    # Map once with gradients before current/future injections.
                    for branch_index in range(layer_index, len(deepstack_visual_embeds)):
                        branch = deepstack_visual_embeds[branch_index]
                        deepstack_visual_embeds[branch_index] = mid_deepstack_keep(branch)
                    mid_deepstack_keep = None
                if decoder_call.phase is DecoderCallPhase.DECODE:
                    continue
                if deepstack_visual_embeds is not None and layer_index in range(
                    len(deepstack_visual_embeds)
                ):
                    hidden_states = self.language_model._deepstack_process(
                        hidden_states,
                        visual_pos_masks,
                        deepstack_visual_embeds[layer_index],
                    )
                if transition_observation is not None:
                    image_indices = visual_pos_masks[0].bool().nonzero(as_tuple=False).flatten()
                    selection.record_transition(transition_observation, image_indices=image_indices)
                    transition_observation = None
                if selection is not None:
                    if selection.output_hidden_layer == layer_index:
                        selection.record_output_hidden(layer_index, self.language_model.norm(hidden_states)[0])
                    image_row = visual_pos_masks[0].bool()
                    if layer_index in selection.output_feature_layers:
                        selection.record_output_features(layer_index, hidden_states[0],
                            decoder_layer=decoder_layer, position_embeddings=layer_input_position_embeddings,
                            image_mask=image_row, padding_mask=padding_mask)
                    if layer_index in selection.output_projection_layers:
                        projection_layer = selection.output_projection_layers[layer_index]
                        features = _decoder_projection_features(self.language_model.layers[projection_layer], hidden_states,
                            position_embeddings=position_embeddings)
                        selection.record_output_projection(layer_index, hidden_states[0], features=features,
                            image_indices=image_row.nonzero(as_tuple=False).flatten())
                    capture = selection.capture_output(layer_index, image_row,
                        padding_mask=padding_mask, labels=self._training_labels.get())
                    if capture is not None:
                        capture_decoder_attention(selection, capture, image_row.nonzero(as_tuple=False).flatten(), read_selection_attention)
                    continue
                if decoder_algorithm == "restore":
                    self._invocation_counts[_SITE_PYRAMID] += 1
                    continue
            apply_pre_layer_compression(len(self.language_model.layers))
            hidden_states = self.language_model.norm(hidden_states)
            output = BaseModelOutputWithPast(
                last_hidden_state=hidden_states,
                past_key_values=past_key_values,
            )
            lifecycle.finish_call(past_key_values=past_key_values)
            if decoder_algorithm == "restore":
                assert restore_attention is not None
                runtime = self._restore_runtime.get()
                self._latest_restore_calibration = {
                    "phase": decoder_call.phase.value,
                    "layers_calibrated": restore_layers_calibrated,
                    "query_tokens": int(hidden_states.shape[1]),
                    "key_tokens": int(restore_attention.key_sizes.shape[-1]),
                    "visual_source_tokens": int(
                        (runtime or {}).get("source_visual_count", 0)
                    ),
                    "first_layer_size": controller.spec.first_layer_size,
                    "position_policy": "native_sparse_mrope_representatives",
                    "cache_transform": "none",
                }
            if owned_lifecycle:
                self._latest_decoder_lifecycle = lifecycle.telemetry()
            return output
        except BaseException:
            lifecycle.abort_call()
            raise


def install_qwen3_vl_transformers_4573_bridge(
    model: Any,
) -> Qwen3VLTransformers4573Bridge:
    """Install or return the existing pinned bridge for ``model``."""

    existing = getattr(model, _BRIDGE_ATTRIBUTE, None)
    if isinstance(existing, Qwen3VLTransformers4573Bridge):
        return existing
    return Qwen3VLTransformers4573Bridge(model).install()


def uninstall_qwen3_vl_transformers_4573_bridge(
    model: Any,
    *,
    force: bool = False,
) -> None:
    """Uninstall the pinned bridge when present."""

    bridge = getattr(model, _BRIDGE_ATTRIBUTE, None)
    if bridge is None:
        return
    if not isinstance(bridge, Qwen3VLTransformers4573Bridge):
        raise BackendIntegrationError(
            "Qwen3-VL model bridge attribute is owned by an unknown provider."
        )
    bridge.uninstall(force=force)


def _installed_transformers_version() -> str:
    try:
        return importlib.metadata.version("transformers")
    except importlib.metadata.PackageNotFoundError as exc:
        raise BackendIntegrationError(
            "Qwen3-VL provider bridge requires transformers==4.57.3."
        ) from exc


def _validate_provider_version() -> None:
    actual = _installed_transformers_version()
    if actual != SUPPORTED_TRANSFORMERS_VERSION:
        raise BackendIntegrationError(
            "Qwen3-VL provider bridge is source-pinned to "
            f"transformers=={SUPPORTED_TRANSFORMERS_VERSION}; got {actual!r}."
        )


def _validate_qwen3_vl_structure(
    model: Any,
    *,
    core: Any,
    visual: Any,
    language_model: Any,
) -> None:
    expected_classes = (
        (model, "Qwen3VLForConditionalGeneration", "model"),
        (core, "Qwen3VLModel", "model.model"),
        (visual, "Qwen3VLVisionModel", "model.model.visual"),
        (language_model, "Qwen3VLTextModel", "model.model.language_model"),
    )
    for value, expected, path in expected_classes:
        actual = type(value).__name__ if value is not None else None
        if actual != expected:
            raise BackendIntegrationError(
                f"Pinned Qwen3-VL bridge requires {path} class {expected}, "
                f"got {actual!r}."
            )
    required_visual = (
        "blocks",
        "merger",
        "deepstack_visual_indexes",
        "deepstack_merger_list",
        "patch_embed",
        "fast_pos_embed_interpolate",
        "rot_pos_emb",
        "spatial_merge_size",
        "spatial_merge_unit",
    )
    required_language = (
        "layers",
        "norm",
        "rotary_emb",
        "embed_tokens",
        "_deepstack_process",
        "config",
    )
    required_core = (
        "get_image_features",
        "get_video_features",
        "get_placeholder_mask",
        "get_rope_index",
        "config",
    )
    _require_attributes(visual, required_visual, "model.model.visual")
    _require_attributes(language_model, required_language, "language_model")
    _require_attributes(core, required_core, "model.model")
    _require_attributes(model, ("forward", "generate"), "model")
    if len(visual.deepstack_visual_indexes) != len(visual.deepstack_merger_list):
        raise BackendIntegrationError("Qwen3-VL DeepStack index/merger counts differ.")
    if tuple(sorted(set(visual.deepstack_visual_indexes))) != tuple(
        visual.deepstack_visual_indexes
    ):
        raise BackendIntegrationError(
            "Qwen3-VL DeepStack visual indexes must be unique and ordered."
        )
    _require_signature(
        visual.forward,
        {"hidden_states", "grid_thw"},
        "Qwen3VLVisionModel.forward",
    )
    _require_signature(
        language_model.forward,
        {
            "inputs_embeds",
            "attention_mask",
            "position_ids",
            "past_key_values",
            "use_cache",
            "visual_pos_masks",
            "deepstack_visual_embeds",
        },
        "Qwen3VLTextModel.forward",
    )
    _require_signature(
        core.forward,
        {"pixel_values", "pixel_values_videos", "image_grid_thw"},
        "Qwen3VLModel.forward",
    )


def _require_attributes(value: Any, names: Sequence[str], path: str) -> None:
    missing = [name for name in names if not hasattr(value, name)]
    if missing:
        raise BackendIntegrationError(
            f"Pinned Qwen3-VL bridge structure mismatch at {path}: missing {missing}."
        )


def _require_signature(
    function: Callable[..., Any],
    required: set[str],
    name: str,
) -> None:
    try:
        parameters = set(inspect.signature(function).parameters)
    except (TypeError, ValueError) as exc:
        raise BackendIntegrationError(
            f"Cannot inspect pinned provider signature {name}."
        ) from exc
    missing = sorted(required - parameters)
    if missing:
        raise BackendIntegrationError(
            f"Pinned provider signature {name} is missing {missing}."
        )


def _validate_controller_registration(
    site: str,
    controller: Any,
    metadata: Mapping[str, Any],
    *,
    visual: Any,
    language_model: Any,
) -> None:
    phase = ExecutionPhase(
        getattr(controller, "execution_phase", ExecutionPhase.FORWARD)
    )
    algorithm = getattr(controller, "algorithm", None)
    if phase is ExecutionPhase.GENERATE and (
        site == _SITE_PYRAMID
        or (site == _SITE_VISIONZIP and algorithm in {"nuwa", "restore"})
    ):
        pass
    elif phase is not ExecutionPhase.FORWARD:
        raise BackendIntegrationError(
            "Pinned Qwen3-VL provider supports GENERATE only for decoder "
            "compression methods with the shared DynamicCache lifecycle."
        )
    if metadata.get("target_modality") != Modality.IMAGE.value:
        raise BackendIntegrationError(
            "Pinned Qwen3-VL provider registrations must target image only."
        )
    if site == _SITE_TOME:
        expected = tuple(
            f"deepstack_merger[{index}]"
            for index in range(len(visual.deepstack_merger_list))
        ) + ("final_merger",)
        if tuple(metadata.get("restore_before", ())) != expected:
            raise BackendIntegrationError(
                "Qwen3-VL ToMe dense-restore sites do not match provider structure."
            )
    elif site == _SITE_VISIONZIP:
        if metadata.get("synchronize_deepstack") is not True:
            raise BackendIntegrationError(
                "Qwen3-VL post-merger compression must synchronize every "
                "DeepStack branch."
            )
        requires_visual_metrics = metadata.get("requires_visual_metrics")
        if not isinstance(requires_visual_metrics, bool):
            raise BackendIntegrationError(
                "Qwen3-VL post-merger compression must declare whether final "
                "visual metrics are required."
            )
        if getattr(controller, "requires_visual_metrics", None) is not (
            requires_visual_metrics
        ):
            raise BackendIntegrationError(
                "Qwen3-VL post-merger controller and registration metadata "
                "disagree about final visual metrics."
            )
        if algorithm == "nuwa" and metadata.get(
            "requires_semantic_hidden_range"
        ) is not True:
            raise BackendIntegrationError(
                "Qwen3-VL Nüwa registration omitted semantic hidden-state capture."
            )
    elif site == _SITE_PYRAMID:
        if algorithm == "restore":
            if metadata.get("requires_mask_position_repair") is not False or (
                metadata.get("requires_kv_cache_repair") is not False
            ):
                raise BackendIntegrationError(
                    "Qwen3-VL RESTORE keeps cache/layout stable and must not request repair."
                )
            if metadata.get("cache_policy") != "stable" or metadata.get(
                "cache_transform"
            ) != "none":
                raise BackendIntegrationError(
                    "Qwen3-VL RESTORE decoder registration must keep cache shape stable."
                )
            if tuple(metadata.get("schedule", ())) != () or metadata.get(
                "attention_calibration"
            ) != "token_mass_plus_mrope_distance":
                raise BackendIntegrationError(
                    "Qwen3-VL RESTORE must calibrate every decoder layer without a prune schedule."
                )
            return
        if metadata.get("requires_mask_position_repair") is not True:
            raise BackendIntegrationError(
                "Qwen3-VL decoder registration omitted mask/position repair."
            )
        expected_cache_repair = phase is ExecutionPhase.GENERATE
        if metadata.get("requires_kv_cache_repair") is not expected_cache_repair:
            raise BackendIntegrationError(
                "Qwen3-VL decoder cache-repair metadata does not match its "
                "execution phase."
            )
        if (
            metadata.get("last_deepstack_injection_layer")
            != len(visual.deepstack_merger_list) - 1
        ):
            raise BackendIntegrationError(
                "Qwen3-VL decoder DeepStack boundary does not match provider."
            )
        method = getattr(controller, "decoder_selection", None)
        if method is not None:
            resolve_decoder_selection_depth(method, len(language_model.layers))
            if metadata.get("cache_policy") != method.cache_policy.value or metadata.get("synchronize_remaining_deepstack") is not True:
                raise BackendIntegrationError("Decoder selection registration must declare cache repair and remaining DeepStack synchronization.")
            if tuple(metadata.get("schedule", ())) != tuple(point.index for point in method.positions):
                raise BackendIntegrationError("Decoder selection registration does not match its application positions.")
            return
        if algorithm in {
            "sgl",
            "nuwa",
        }:
            if metadata.get("cache_policy") != "compact_prefix" or (
                metadata.get("synchronize_remaining_deepstack") is not True
            ):
                raise BackendIntegrationError(
                    f"Qwen3-VL {algorithm} registration must compact prefix cache and "
                    "synchronize remaining DeepStack branches."
                )
            schedule = tuple(metadata.get("schedule", ()))
            if algorithm == "sgl":
                expected_schedule = (getattr(controller, "application_position", LayerPosition("inter_layer", controller.prune_layer)).index,)
            elif algorithm == "nuwa":
                expected_schedule = (getattr(controller, "application_position", LayerPosition("inter_layer", controller.spec.stage2_after_layer)).index,)
            if schedule != expected_schedule:
                raise BackendIntegrationError(
                    f"Qwen3-VL {algorithm} registration schedule does not match "
                    "its configured boundary."
                )
        else:
            raise BackendIntegrationError(
                f"Unknown Qwen3-VL decoder algorithm {algorithm!r}."
            )
    else:
        raise BackendIntegrationError(
            f"Unknown Qwen3-VL provider semantic site {site!r}."
        )


def _registrations_support_generate(
    registrations: Sequence[_Registration],
) -> bool:
    if len(registrations) == 1:
        item = registrations[0]
        return (
            item.site == _SITE_PYRAMID
            and ExecutionPhase(item.controller.execution_phase)
            is ExecutionPhase.GENERATE
            and getattr(item.controller, "algorithm", None)
            not in {"nuwa", "restore"}
        )
    if len(registrations) != 2:
        return False
    controllers = {id(item.controller) for item in registrations}
    sites = {item.site for item in registrations}
    algorithms = {getattr(item.controller, "algorithm", None) for item in registrations}
    return (
        len(controllers) == 1
        and sites == {_SITE_VISIONZIP, _SITE_PYRAMID}
        and len(algorithms) == 1
        and algorithms <= {"nuwa", "restore"}
        and all(
            ExecutionPhase(item.controller.execution_phase)
            is ExecutionPhase.GENERATE
            for item in registrations
        )
    )


def _bind_forward_arguments(
    function: Callable[..., Any],
    args: Sequence[Any],
    kwargs: Mapping[str, Any],
) -> dict[str, Any]:
    try:
        bound = inspect.signature(function).bind_partial(*args, **kwargs)
    except (TypeError, ValueError):
        return dict(kwargs)
    return dict(bound.arguments)


def _bind_visual_call(
    function: Callable[..., Any],
    args: Sequence[Any],
    kwargs: Mapping[str, Any],
) -> tuple[Any, Any, dict[str, Any]]:
    values = _bind_forward_arguments(function, args, kwargs)
    hidden = values.pop("hidden_states", args[0] if args else None)
    grid = values.pop("grid_thw", None)
    values.pop("self", None)
    extras = values.pop("kwargs", None)
    if isinstance(extras, Mapping):
        values.update(extras)
    if hidden is None or grid is None:
        raise BackendIntegrationError(
            "Qwen3-VL visual bridge requires hidden_states and grid_thw."
        )
    return hidden, grid, values


def _bind_language_call(
    function: Callable[..., Any],
    args: Sequence[Any],
    kwargs: Mapping[str, Any],
) -> dict[str, Any]:
    values = _bind_forward_arguments(function, args, kwargs)
    values.pop("self", None)
    defaults = {
        "input_ids": None,
        "attention_mask": None,
        "position_ids": None,
        "past_key_values": None,
        "inputs_embeds": None,
        "use_cache": None,
        "cache_position": None,
        "visual_pos_masks": None,
        "deepstack_visual_embeds": None,
    }
    extras = values.pop("kwargs", None)
    if isinstance(extras, Mapping):
        values.update(extras)
    return {**defaults, **values}


def _compact_prompt_call(
    call: dict[str, Any],
    removal_mask: Any | None,
) -> dict[str, Any]:
    if removal_mask is None or not bool(removal_mask.any().item()):
        return call
    torch = _torch()
    inputs = call["inputs_embeds"]
    if inputs is None or int(inputs.shape[0]) != 1:
        raise BackendIntegrationError(
            "Compressed Qwen3-VL prompt legalization currently requires one "
            "inputs_embeds batch item."
        )
    keep = (~removal_mask[0]).nonzero(as_tuple=False).flatten()
    call["inputs_embeds"] = inputs.index_select(1, keep.to(inputs.device))
    call["input_ids"] = None
    for name in ("attention_mask", "position_ids", "visual_pos_masks"):
        value = call[name]
        if value is None:
            continue
        indices = keep.to(value.device)
        call[name] = value.index_select(-1, indices)
    call["cache_position"] = torch.arange(
        keep.numel(),
        device=inputs.device,
    )
    return call


def _cumulative_lengths(counts: Sequence[int], *, device: Any) -> Any:
    torch = _torch()
    return torch.tensor(
        (0, *tuple(_running_sums(counts))),
        dtype=torch.int32,
        device=device,
    )


def _running_sums(counts: Sequence[int]) -> Sequence[int]:
    result: list[int] = []
    total = 0
    for count in counts:
        total += int(count)
        result.append(total)
    return result


def _select_layout_representatives(
    value: Any,
    deltas: Sequence[LayoutDelta],
    counts: Sequence[int],
) -> Any:
    torch = _torch()
    chunks = torch.split(value, tuple(counts))
    selected = []
    for chunk, delta in zip(chunks, deltas):
        indices = torch.tensor(
            _layout_representatives(delta),
            dtype=torch.long,
            device=chunk.device,
        )
        selected.append(chunk.index_select(0, indices))
    return torch.cat(selected, dim=0)


def _restore_tome_steps(
    hidden_states: Any,
    current_counts: Sequence[int],
    steps: Sequence[_ToMeStep],
) -> Any:
    torch = _torch()
    value = hidden_states
    counts = tuple(current_counts)
    for step in reversed(steps):
        if counts != step.after_counts:
            raise InvariantError(
                "Qwen3-VL ToMe restore chain cardinality is inconsistent."
            )
        chunks = torch.split(value, counts)
        value = torch.cat(
            [
                step.controller.restore_dense_boundary(chunk, delta)
                for chunk, delta in zip(chunks, step.deltas)
            ],
            dim=0,
        )
        counts = step.before_counts
    return value


def _visionzip_metrics_from_capture(
    capture: _VisionAttentionCapture,
    *,
    patch_counts: Sequence[int],
    spatial_merge_unit: int,
) -> tuple[list[Any], list[Any]]:
    """Project captured final-block statistics onto native merger tokens.

    The attention argument expected by the backbone-neutral VisionZip kernel
    is square, even though Qwen-like towers only consume its column mean.  We
    therefore expose each exact mean-received vector as a stride-zero expanded
    view.  This preserves the ranking statistic with O(tokens) storage and
    avoids manufacturing a second dense softmax matrix after the block.
    """

    if spatial_merge_unit < 1:
        raise InvariantError("Qwen3-VL spatial merge unit must be positive.")
    patch_counts = tuple(int(value) for value in patch_counts)
    if any(value < 1 for value in patch_counts):
        raise InvariantError("Qwen3-VL patch counts must be positive.")
    attentions: list[Any] = []
    keys: list[Any] = []
    if not (
        len(capture.counts)
        == len(capture.mean_received)
        == len(capture.keys)
        == len(capture.token_sizes)
        == len(patch_counts)
    ):
        raise InvariantError(
            "Qwen3-VL VisionZip capture stream counts are inconsistent."
        )
    for stream_index, (
        compact_count,
        importance,
        compact_keys,
        compact_sizes,
        patch_count,
    ) in enumerate(
        zip(
            capture.counts,
            capture.mean_received,
            capture.keys,
            capture.token_sizes,
            patch_counts,
        )
    ):
        if int(importance.shape[0]) != int(compact_count) or int(
            compact_keys.shape[0]
        ) != int(compact_count):
            raise InvariantError(
                "Qwen3-VL VisionZip capture tensor counts are inconsistent."
            )
        dense_importance, dense_keys = _restore_attention_capture_to_dense(
            importance,
            compact_keys,
            compact_sizes,
            capture.restore_steps,
            stream_index=stream_index,
            source_count=int(patch_count),
        )
        if int(patch_count) % spatial_merge_unit:
            raise InvariantError(
                "Qwen3-VL patch count is not divisible by spatial merge unit."
            )
        feature_count = int(patch_count) // spatial_merge_unit
        feature_importance = dense_importance.reshape(
            feature_count,
            spatial_merge_unit,
        ).mean(dim=1)
        feature_keys = dense_keys.reshape(
            feature_count,
            spatial_merge_unit,
            dense_keys.shape[1],
            dense_keys.shape[2],
        ).mean(dim=1)
        # VisionZip's mean_received mode averages rows.  Expanding one exact
        # received-mass vector creates a zero-copy square Tensor view whose
        # row mean is unchanged.
        attentions.append(feature_importance.unsqueeze(0).expand(feature_count, -1))
        keys.append(feature_keys.permute(1, 0, 2).contiguous())
    return attentions, keys


def _restore_attention_capture_to_dense(
    importance: Any,
    keys: Any,
    token_sizes: Any | None,
    restore_steps: Sequence[_ToMeStep],
    *,
    stream_index: int,
    source_count: int,
) -> tuple[Any, Any]:
    """Distribute compact received mass back across original patch sources."""

    torch = _torch()
    mapping = list(range(source_count))
    current_count = source_count
    for step in restore_steps:
        if stream_index >= len(step.deltas):
            raise InvariantError("Qwen3-VL ToMe restore stream is missing.")
        delta = step.deltas[stream_index]
        if delta.source_count != current_count:
            raise InvariantError("Qwen3-VL ToMe capture restore chain is inconsistent.")
        mapping = [int(delta.old_to_new[index]) for index in mapping]
        current_count = delta.result_count
    if current_count != int(importance.shape[0]):
        raise InvariantError(
            "Qwen3-VL captured attention does not match its ToMe input layout."
        )
    gather = torch.tensor(mapping, dtype=torch.long, device=importance.device)
    dense_keys = keys.index_select(0, gather.to(keys.device))
    if token_sizes is None:
        compact_sizes = torch.ones_like(importance, dtype=torch.float32)
    else:
        compact_sizes = token_sizes.reshape(current_count).float()
    # Proportional attention assigns probability mass to a compact group.  A
    # dense consumer receives the group's mass divided by its carried size;
    # summing across preimages therefore preserves the captured distribution.
    dense_importance = importance.index_select(0, gather) / compact_sizes.index_select(
        0,
        gather.to(compact_sizes.device),
    )
    return dense_importance, dense_keys


def _restore_mrope_section(config: Any, *, axis_count: int) -> tuple[int, ...]:
    scaling = getattr(config, "rope_scaling", None)
    section = scaling.get("mrope_section") if isinstance(scaling, Mapping) else None
    if not isinstance(section, Sequence) or isinstance(section, (str, bytes)):
        raise BackendIntegrationError(
            "Qwen3-VL RESTORE requires native rope_scaling.mrope_section."
        )
    values = tuple(int(value) for value in section)
    if len(values) != axis_count or any(value < 0 for value in values):
        raise BackendIntegrationError(
            "Qwen3-VL RESTORE M-RoPE section does not match its position axes."
        )
    return values


def _layout_representatives(delta: LayoutDelta) -> tuple[int, ...]:
    explicit = delta.result_layout.get("representative_indices")
    if explicit is not None:
        values = tuple(int(value) for value in explicit)
        if (
            len(values) != delta.result_count
            or len(set(values)) != len(values)
            or any(value < 0 or value >= delta.source_count for value in values)
        ):
            raise InvariantError(
                "LayoutDelta representative_indices are not valid source positions."
            )
        for result_index, source in enumerate(values):
            if delta.old_to_new[source] != result_index:
                raise InvariantError(
                    "LayoutDelta representative does not belong to its result group."
                )
        return values
    result: list[int] = []
    for result_index in range(delta.result_count):
        preimage = [
            source
            for source, target in delta.old_to_new.items()
            if target == result_index
        ]
        if not preimage:
            raise InvariantError("LayoutDelta result has no source preimage.")
        result.append(preimage[0])
    return tuple(result)


def _compose_layout_mapping(first: LayoutDelta, second: LayoutDelta) -> LayoutDelta:
    if first.result_count != second.source_count:
        raise InvariantError("Qwen3-VL layout composition count mismatch.")
    # Keep the ordering policy declared by AIM.  Reconstructing this mapping
    # through from_mapping() silently selected PRESERVE_SOURCE and rejected
    # valid explicit bipartite-merge orderings on real checkpoint features.
    return first.compose(second)


def _delta_groups(delta: LayoutDelta) -> tuple[tuple[int, ...], ...]:
    return tuple(
        tuple(
            source
            for source, target in delta.old_to_new.items()
            if target == result_index
        )
        for result_index in range(delta.result_count)
    )


def _merge_values_by_tome_groups(
    value: Any,
    groups: Sequence[Sequence[int]],
) -> Any:
    """Replay one ToMe grouping on a synchronized DeepStack feature branch."""

    torch = _torch()
    outputs = []
    for group in groups:
        indices = torch.tensor(tuple(group), dtype=torch.long, device=value.device)
        outputs.append(value.index_select(0, indices).mean(dim=0))
    return torch.stack(outputs, dim=0)


def _mean_square_by_delta(value: Any, delta: LayoutDelta) -> Any:
    torch = _torch()
    groups = _delta_groups(delta)
    rows = []
    for query_group in groups:
        query_index = torch.tensor(
            query_group,
            dtype=torch.long,
            device=value.device,
        )
        columns = []
        for key_group in groups:
            key_index = torch.tensor(
                key_group,
                dtype=torch.long,
                device=value.device,
            )
            columns.append(
                value.index_select(-2, query_index)
                .index_select(-1, key_index)
                .mean(dim=(-2, -1))
            )
        rows.append(torch.stack(columns, dim=-1))
    return torch.stack(rows, dim=-2)


def _mean_keys_by_delta(value: Any, delta: LayoutDelta) -> Any:
    torch = _torch()
    outputs = []
    for group in _delta_groups(delta):
        indices = torch.tensor(group, dtype=torch.long, device=value.device)
        outputs.append(value.index_select(1, indices).mean(dim=1))
    return torch.stack(outputs, dim=1)


def _decoder_query_keys(
    decoder_layer: Any,
    hidden_states: Any,
    *,
    instruction_indices: Any,
    image_indices: Any,
    position_embeddings: tuple[Any, Any],
    use_rope: bool = True,
) -> tuple[Any, Any, Any]:
    torch = _torch()
    try:
        from transformers.models.qwen3_vl.modeling_qwen3_vl import (
            apply_rotary_pos_emb,
        )
    except ImportError as exc:  # pragma: no cover - exact provider guard covers this
        raise BackendIntegrationError(
            "transformers==4.57.3 Qwen3-VL RoPE helper is unavailable."
        ) from exc
    normalised = decoder_layer.input_layernorm(hidden_states)
    attention = decoder_layer.self_attn
    batch, sequence_length, _ = normalised.shape
    query = attention.q_norm(
        attention.q_proj(normalised).view(
            batch,
            sequence_length,
            -1,
            attention.head_dim,
        )
    ).transpose(1, 2)
    keys = attention.k_norm(
        attention.k_proj(normalised).view(
            batch,
            sequence_length,
            -1,
            attention.head_dim,
        )
    ).transpose(1, 2)
    if use_rope:
        cos, sin = (value.to(query.device) for value in position_embeddings)
        query, keys = apply_rotary_pos_emb(query, keys, cos, sin)
    instruction_indices = torch.as_tensor(
        instruction_indices,
        dtype=torch.long,
        device=query.device,
    )
    query = query[0].index_select(1, instruction_indices).transpose(0, 1)
    keys = keys[0].transpose(0, 1)
    groups = int(attention.num_key_value_groups)
    if groups > 1:
        keys = torch.repeat_interleave(keys, groups, dim=1)
    return query, keys.index_select(0, image_indices.to(keys.device)), keys


def _decoder_projection_features(
    decoder_layer: Any,
    hidden_states: Any,
    *,
    position_embeddings: tuple[Any, Any],
) -> tuple[Any, Any, Any]:
    """Extract normalized Qwen3 pre-RoPE Q/K and rotated clustering keys."""

    torch = _torch()
    try:
        from transformers.models.qwen3_vl.modeling_qwen3_vl import (
            apply_rotary_pos_emb,
        )
    except ImportError as exc:  # pragma: no cover - provider guard covers this
        raise BackendIntegrationError(
            "transformers==4.57.3 Qwen3-VL RoPE helper is unavailable."
        ) from exc
    normalised = decoder_layer.input_layernorm(hidden_states.detach())
    attention = decoder_layer.self_attn
    batch, sequence_length, _ = normalised.shape
    query = attention.q_norm(
        attention.q_proj(normalised).view(
            batch, sequence_length, -1, attention.head_dim
        )
    ).transpose(1, 2)
    keys = attention.k_norm(
        attention.k_proj(normalised).view(
            batch, sequence_length, -1, attention.head_dim
        )
    ).transpose(1, 2)
    cos, sin = position_embeddings
    _rotated_query, rotated_keys = apply_rotary_pos_emb(query, keys, cos.to(query.device), sin.to(query.device))
    groups = int(getattr(attention, "num_key_value_groups", 1))
    expanded_keys = (
        torch.repeat_interleave(keys, groups, dim=1) if groups > 1 else keys
    )
    expanded_rotated = (
        torch.repeat_interleave(rotated_keys, groups, dim=1)
        if groups > 1
        else rotated_keys
    )
    clustering = expanded_rotated.transpose(1, 2).flatten(2, 3)
    return query[0], expanded_keys[0], clustering[0]


def _attention_key_visibility(
    attention_mask: Any | None,
    *,
    instruction_indices: Any,
    key_count: int,
    padding_mask: Any | None,
) -> Any | None:
    """Project the provider causal mask to each instruction query."""

    try:
        return pyramid_attention_key_visibility(
            attention_mask,
            instruction_indices=instruction_indices,
            key_count=key_count,
            padding_mask=padding_mask,
        )
    except (TypeError, ValueError) as exc:
        raise BackendIntegrationError(
            "Qwen3-VL decoder compression could not derive causal key visibility."
        ) from exc


def _layer_local_decode_padding_mask(
    padding_mask: Any | None,
    past_key_values: Any,
    *,
    layer_index: int,
    query_length: int,
    reference: Any,
) -> Any | None:
    if padding_mask is None:
        return None
    if getattr(padding_mask, "ndim", None) != 2 or int(padding_mask.shape[0]) != 1:
        raise BackendIntegrationError(
            "Qwen3-VL cached decoder compression requires a rank-2 batch-one mask."
        )
    if not bool(padding_mask.bool().all().item()):
        raise BackendIntegrationError(
            "Qwen3-VL cached decoder compression does not support padded prompts."
        )
    length = dynamic_cache_layer_length(past_key_values, layer_index) + query_length
    return _torch().ones(
        (1, length),
        dtype=padding_mask.dtype,
        device=reference.device,
    )


def _new_dynamic_cache(config: Any) -> Any:
    try:
        from transformers.cache_utils import DynamicCache
    except ImportError as exc:  # pragma: no cover - exact version guard covers it
        raise BackendIntegrationError(
            "transformers==4.57.3 DynamicCache is unavailable."
        ) from exc
    return DynamicCache(config=config)


def _torch() -> Any:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - control-plane only env
        raise RuntimeError("Qwen3-VL provider execution requires PyTorch.") from exc
    return torch


def _transformers_decoder_helpers() -> tuple[Any, Callable[..., Any]]:
    try:
        from transformers.masking_utils import create_causal_mask
        from transformers.modeling_outputs import BaseModelOutputWithPast
    except ImportError as exc:  # pragma: no cover - exact env guard covers this
        raise BackendIntegrationError(
            "transformers==4.57.3 decoder helpers are unavailable."
        ) from exc
    return BaseModelOutputWithPast, create_causal_mask


__all__ = [
    "QWEN3_VL_TRANSFORMERS_4573_CONTRACT",
    "Qwen3VLProviderBridgeContract",
    "Qwen3VLProviderRegistrationHandle",
    "Qwen3VLTransformers4573Bridge",
    "SUPPORTED_TRANSFORMERS_VERSION",
    "install_qwen3_vl_transformers_4573_bridge",
    "uninstall_qwen3_vl_transformers_4573_bridge",
]
