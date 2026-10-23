"""Pinned Transformers 4.57.3 bridge for Qwen2.5-VL image compression.

This module is deliberately source-specific.  It implements the three
semantic callbacks exposed by :mod:`image_compression` against the exact
``transformers==4.57.3`` Qwen2.5-VL source layout:

* ToMe runs after each vision attention residual and before that block's MLP;
  the compact MLP result is restored for native windows, then a final ToMe
  legalizer permanently compacts post-merger image features for the LLM.
* VisionZip runs after the native 2x2 visual merger and reverse-window reorder,
  before image features are scattered into prompt placeholders.
* PyramidDrop runs inside the text decoder loop and rebuilds causal masks,
  M-RoPE inputs, and sequence-aligned image masks after every drop; its
  generate phase uses the shared layer-local DynamicCache lifecycle.

The executable envelope is intentionally narrow: image-only batch-one
``FORWARD(use_cache=False)`` plus guarded decoder-method generation with
``use_cache=True``.  Causal-LM labels are repaired across every forward
sequence mutation.  Videos and unsupported output modes fail closed.  Patches
are instance-local and reversible.

Installing this bridge does not advance support lifecycle state and does not
claim real-checkpoint verification.
"""

from __future__ import annotations


from mllm_comp.compressors.image.positions import (
    tome_encoder_positions,
    tome_encoder_score_layer,
)
from mllm_comp.compressors.image.tome import tome_merge_reference
from mllm_comp.compressors.layer_positions import LayerPosition

from collections.abc import Callable, Mapping, Sequence
from contextlib import nullcontext
from contextvars import ContextVar
from dataclasses import dataclass, field, replace
import importlib.metadata
import inspect
from threading import RLock
from types import MappingProxyType, MethodType
from typing import Any
from uuid import uuid4

from mllm_comp.backbones._vision_attention import execute_vision_attention
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
from mllm_comp.compressors.image.attention_mass import add_visual_log_mass_bias
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
_BRIDGE_ATTRIBUTE = "_mllm_comp_qwen25_vl_provider_bridge"
_IMAGE_BRIDGE_ATTRIBUTE = "_mllm_comp_qwen25_vl_image_compression_bridge"
_SITE_TOME = "encoder_dense_boundary"
_SITE_SAINT = "encoder_progressive_prune"
_SITE_VISIONZIP = "post_visual_merger"
_SITE_PYRAMID = "decoder_layer_loop"
_KNOWN_SITES = frozenset(
    {_SITE_TOME, _SITE_SAINT, _SITE_VISIONZIP, _SITE_PYRAMID}
)
_PROJECTOR_GENERATE_ALGORITHMS = frozenset(
    {"bifocal", "feature_anchor_mean", "grid_mean", "prumerge",
     "visionzip", "divprune", "aim", "gap", "mmtok"}
)
_CENTROID_MROPE_ALGORITHMS = frozenset(
    {"feature_anchor_mean", "grid_mean"}
)


@dataclass(frozen=True)
class Qwen25VLProviderBridgeContract:
    """Identity and execution envelope for the pinned provider source."""

    provider: str = "transformers"
    provider_version: str = SUPPORTED_TRANSFORMERS_VERSION
    model_class: str = "Qwen2_5_VLForConditionalGeneration"
    core_class: str = "Qwen2_5_VLModel"
    visual_class: str = "Qwen2_5_VisionTransformerPretrainedModel"
    language_model_class: str = "Qwen2_5_VLTextModel"
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
                "provider_source": (
                    "transformers.models.qwen2_5_vl.modeling_qwen2_5_vl"
                ),
                "checkpoint_family": "Qwen2.5-VL dense",
                "real_checkpoint_verified": False,
                "support_lifecycle_mutated": False,
                "visionzip_metric": "final_full_attention_qk",
                "mrope_repair": "retain_dense_source_coordinates",
                "generate_scope": "decoder_methods_with_dynamic_cache",
            }
        )
    )


QWEN25_VL_TRANSFORMERS_4573_CONTRACT = Qwen25VLProviderBridgeContract()


@dataclass(frozen=True)
class _VisualCompressionPlan:
    source_counts: tuple[int, ...]
    result_counts: tuple[int, ...]
    layout_deltas: tuple[LayoutDelta, ...]

    @property
    def compressed(self) -> bool:
        return self.source_counts != self.result_counts


@dataclass(frozen=True)
class _Registration:
    token: str
    site: str
    executor: Any
    schedule: tuple[int, ...] = ()


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


class Qwen25VLProviderRegistrationHandle:
    """One idempotent, independently removable activation registration."""

    def __init__(
        self,
        bridge: "Qwen25VLTransformers4573Bridge",
        token: str,
    ) -> None:
        self._bridge = bridge
        self.token = token

    @property
    def removed(self) -> bool:
        return not self._bridge._contains_registration(self.token)

    def remove(self) -> None:
        self._bridge._remove_registration(self.token)


class Qwen25VLCompositeProviderRegistrationHandle:
    """One removable handle for Nüwa's visual and decoder registrations."""

    def __init__(
        self,
        bridge: "Qwen25VLTransformers4573Bridge",
        tokens: tuple[str, str],
    ) -> None:
        self._bridge = bridge
        self.tokens = tokens

    @property
    def removed(self) -> bool:
        return all(not self._bridge._contains_registration(token) for token in self.tokens)

    def remove(self) -> None:
        for token in reversed(self.tokens):
            self._bridge._remove_registration(token)


class Qwen25VLTransformers4573Bridge:
    """Reversible semantic bridge installed on one Qwen2.5-VL instance."""

    contract = QWEN25_VL_TRANSFORMERS_4573_CONTRACT

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
            _SITE_SAINT: 0,
            _SITE_VISIONZIP: 0,
            _SITE_PYRAMID: 0,
        }
        self._latest_delta_by_activation: dict[str, dict[str, Any]] = {}
        self._delta_event_count = 0
        self._decoder_request_counter = 0
        self._latest_decoder_lifecycle: dict[str, Any] | None = None
        self._latest_decode_position: dict[str, Any] | None = None
        self._latest_rope_delta_repair: dict[str, Any] | None = None
        self._latest_restore_calibration: dict[str, Any] | None = None

        suffix = uuid4().hex
        self._visual_modality: ContextVar[str | None] = ContextVar(
            f"qwen25_vl_visual_modality_{suffix}", default=None
        )
        self._visual_plan: ContextVar[_VisualCompressionPlan | None] = ContextVar(
            f"qwen25_vl_visual_plan_{suffix}", default=None
        )
        self._prompt_removal_mask: ContextVar[Any | None] = ContextVar(
            f"qwen25_vl_prompt_removal_{suffix}", default=None
        )
        self._prompt_image_mask: ContextVar[Any | None] = ContextVar(
            f"qwen25_vl_prompt_image_mask_{suffix}", default=None
        )
        self._comparison_grid = ContextVar(f"comparison_grid_{suffix}", default=None)
        self._image_count: ContextVar[int | None] = ContextVar(
            f"qwen25_vl_image_count_{suffix}", default=None
        )
        self._prompt_text_embeddings: ContextVar[Any | None] = ContextVar(
            f"qwen25_vl_prompt_text_embeddings_{suffix}", default=None
        )
        self._training_labels: ContextVar[Any | None] = ContextVar(
            f"qwen25_vl_training_labels_{suffix}", default=None
        )
        self._decoder_lifecycle: ContextVar[ImageDecoderLifecycle | None] = ContextVar(
            f"qwen25_vl_decoder_lifecycle_{suffix}", default=None
        )
        self._layer_local_native_rope_delta: ContextVar[
            tuple[int, ...] | None
        ] = ContextVar(
            f"qwen25_vl_layer_local_native_rope_delta_{suffix}",
            default=(),
        )
        self._restore_runtime: ContextVar[dict[str, Any] | None] = ContextVar(
            f"qwen25_vl_restore_runtime_{suffix}",
            default=None,
        )
        self._prompt_input_ids: ContextVar[Any | None] = ContextVar(
            f"qwen25_vl_prompt_input_ids_{suffix}", default=None
        )
        self._projector_generate: ContextVar[dict[str, Any] | None] = ContextVar(
            f"qwen25_vl_projector_generate_{suffix}", default=None
        )
        self._latest_projector_generate: dict[str, Any] | None = None
        self._visual_log_mass: ContextVar[Any | None] = ContextVar(
            f"qwen25_vl_visual_log_mass_{suffix}", default=None
        )

        self._original_model_forward: Callable[..., Any] | None = None
        self._original_generate: Callable[..., Any] | None = None
        self._original_core_forward: Callable[..., Any] | None = None
        self._original_visual_forward: Callable[..., Any] | None = None
        self._original_language_forward: Callable[..., Any] | None = None
        self._original_get_image_features: Callable[..., Any] | None = None
        self._original_get_video_features: Callable[..., Any] | None = None
        self._original_get_placeholder_mask: Callable[..., Any] | None = None
        self._original_layer0_attention_forward: Callable[..., Any] | None = None

    @property
    def installed(self) -> bool:
        return self._installed

    def install(self) -> "Qwen25VLTransformers4573Bridge":
        """Validate the pinned source contract, then patch transactionally."""

        with self._lock:
            if self._installed:
                return self
            _validate_provider_version()
            _validate_qwen25_vl_structure(
                self.model,
                core=self.core,
                visual=self.visual,
                language_model=self.language_model,
            )
            for attribute in (_BRIDGE_ATTRIBUTE, _IMAGE_BRIDGE_ATTRIBUTE):
                existing = getattr(self.model, attribute, None)
                if existing is not None and existing is not self:
                    raise BackendIntegrationError(
                        "This Qwen2.5-VL model already has a different provider "
                        f"bridge at {attribute}."
                    )
            patches: list[_InstancePatch] = []
            try:
                self._capture_original_methods()
                patches.extend(self._execution_patches())
                patches.append(
                    _InstancePatch.apply(self.model, _BRIDGE_ATTRIBUTE, self)
                )
                patches.append(
                    _InstancePatch.apply(self.model, _IMAGE_BRIDGE_ATTRIBUTE, self)
                )
            except BaseException:
                for patch in reversed(patches):
                    patch.restore()
                raise
            self._patches = patches
            self._installed = True
            return self

    def uninstall(self, *, force: bool = False) -> None:
        """Restore all instance attributes, optionally evicting activations."""

        with self._lock:
            if not self._installed:
                return
            if self._registrations and not force:
                active = [
                    f"{item.site}:{getattr(item.executor.contract, 'binding', '?')}"
                    for item in self._registrations.values()
                ]
                raise BackendIntegrationError(
                    "Cannot uninstall the Qwen2.5-VL provider bridge while "
                    f"activations remain registered: {active}."
                )
            self._registrations.clear()
            for patch in reversed(self._patches):
                patch.restore()
            self._patches.clear()
            self._installed = False

    def install_image_compression(
        self,
        executor: Any,
    ) -> Qwen25VLProviderRegistrationHandle:
        """Register one adapter executor at its exact semantic callback site."""

        with self._lock:
            if not self._installed:
                raise BackendIntegrationError(
                    "Qwen2.5-VL provider bridge must be installed first."
                )
            if any(item.executor is executor for item in self._registrations.values()):
                raise BackendIntegrationError(
                    "The same Qwen2.5-VL compression activation is already registered."
                )
            composite_algorithm = getattr(
                getattr(executor, "contract", None), "algorithm", None
            )
            if composite_algorithm in {"nuwa", "restore"}:
                schedule = (
                    _validate_nuwa_composite_registration(
                        executor,
                        visual=self.visual,
                        language_model=self.language_model,
                    )
                    if composite_algorithm == "nuwa"
                    else _validate_restore_composite_registration(
                        executor,
                        visual=self.visual,
                    )
                )
                visual_token = uuid4().hex
                decoder_token = uuid4().hex
                self._registrations[visual_token] = _Registration(
                    token=visual_token,
                    site=_SITE_VISIONZIP,
                    executor=executor,
                    schedule=(),
                )
                self._registrations[decoder_token] = _Registration(
                    token=decoder_token,
                    site=_SITE_PYRAMID,
                    executor=executor,
                    schedule=schedule,
                )
                return Qwen25VLCompositeProviderRegistrationHandle(
                    self,
                    (visual_token, decoder_token),
                )
            site, schedule = _validate_executor_registration(
                executor,
                visual=self.visual,
                language_model=self.language_model,
            )
            token = uuid4().hex
            self._registrations[token] = _Registration(
                token=token,
                site=site,
                executor=executor,
                schedule=schedule,
            )
            return Qwen25VLProviderRegistrationHandle(self, token)

    def uninstall_image_compression(self, handle: Any) -> None:
        if not isinstance(
            handle,
            (
                Qwen25VLProviderRegistrationHandle,
                Qwen25VLCompositeProviderRegistrationHandle,
            ),
        ):
            raise BackendIntegrationError(
                "Qwen2.5-VL provider received an unknown activation handle."
            )
        if handle._bridge is not self:
            raise BackendIntegrationError(
                "Qwen2.5-VL activation handle belongs to another model bridge."
            )
        handle.remove()

    def runtime_metadata(self) -> dict[str, Any]:
        with self._lock:
            registrations = tuple(self._registrations.values())
            latest = [
                dict(value) for value in self._latest_delta_by_activation.values()
            ]
        return {
            "provider": self.contract.provider,
            "provider_version": self.contract.provider_version,
            "backbone": "qwen25_vl",
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
            "supports_video": False,
            "supports_labels": True,
            "registration_count": len(registrations),
            "registrations": [
                {
                    "activation_id": item.token,
                    "site": item.site,
                    "binding": getattr(item.executor.contract, "binding", None),
                    "algorithm": getattr(item.executor.contract, "algorithm", None),
                    "schedule": list(item.schedule),
                    **({"permanent_output_merge_site": item.executor.output_target_site}
                        if hasattr(item.executor, "output_target_site") else {}),
                    **({"projector_target_sites": list(item.executor.projector_target_sites)}
                        if hasattr(item.executor, "projector_target_sites") else {}),
                }
                for item in registrations
            ],
            "invocation_counts": dict(self._invocation_counts),
            "latest_layout_deltas": latest,
            "layout_delta_event_count": self._delta_event_count,
            "latest_decoder_lifecycle": (
                dict(self._latest_decoder_lifecycle)
                if self._latest_decoder_lifecycle is not None
                else None
            ),
            "latest_decode_position": (
                dict(self._latest_decode_position)
                if self._latest_decode_position is not None
                else None
            ),
            "latest_rope_delta_repair": (
                dict(self._latest_rope_delta_repair)
                if self._latest_rope_delta_repair is not None
                else None
            ),
            "latest_restore_calibration": (
                dict(self._latest_restore_calibration)
                if self._latest_restore_calibration is not None
                else None
            ),
            "latest_projector_generate": (
                dict(self._latest_projector_generate)
                if self._latest_projector_generate is not None
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
        self._original_get_image_features = self.core.get_image_features
        self._original_get_video_features = self.core.get_video_features
        self._original_get_placeholder_mask = self.core.get_placeholder_mask
        self._original_layer0_attention_forward = self.language_model.layers[0].self_attn.forward

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

        def layer0_attention_forward(module: Any, *args: Any, **kwargs: Any) -> Any:
            return bridge._layer0_attention_forward(module, *args, **kwargs)

        def get_image_features(
            module: Any,
            pixel_values: Any,
            image_grid_thw: Any = None,
        ) -> Any:
            return bridge._get_visual_features(
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
                input_ids,
                inputs_embeds,
                image_features=image_features,
                video_features=video_features,
            )

        assert self._original_model_forward is not None
        preserve_bound_method_signature(model_forward, self._original_model_forward)
        assert self._original_layer0_attention_forward is not None
        preserve_bound_method_signature(
            layer0_attention_forward, self._original_layer0_attention_forward
        )

        return [
            _InstancePatch.apply(
                self.model, "forward", MethodType(model_forward, self.model)
            ),
            _InstancePatch.apply(
                self.model, "generate", MethodType(generate, self.model)
            ),
            _InstancePatch.apply(
                self.core, "forward", MethodType(core_forward, self.core)
            ),
            _InstancePatch.apply(
                self.visual, "forward", MethodType(visual_forward, self.visual)
            ),
            _InstancePatch.apply(
                self.language_model,
                "forward",
                MethodType(language_forward, self.language_model),
            ),
            _InstancePatch.apply(
                self.language_model.layers[0].self_attn,
                "forward",
                MethodType(
                    layer0_attention_forward,
                    self.language_model.layers[0].self_attn,
                ),
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

    def _contains_registration(self, token: str) -> bool:
        with self._lock:
            return token in self._registrations

    def _remove_registration(self, token: str) -> None:
        with self._lock:
            self._registrations.pop(token, None)

    def _registrations_for(self, site: str) -> tuple[_Registration, ...]:
        with self._lock:
            return tuple(
                item for item in self._registrations.values() if item.site == site
            )

    def _has_registrations(self) -> bool:
        with self._lock:
            return bool(self._registrations)

    def _record_delta(
        self,
        registration: _Registration,
        delta: LayoutDelta,
        *,
        boundary_layer: int | None = None,
        image_index: int | None = None,
    ) -> None:
        if not isinstance(delta, LayoutDelta):
            raise InvariantError("Provider callback did not return LayoutDelta.")
        with self._lock:
            self._invocation_counts[registration.site] += 1
            self._delta_event_count += 1
            invocation_id = f"{registration.token}:{self._delta_event_count}"
            summary: dict[str, Any] = {
                "source_count": delta.source_count,
                "result_count": delta.result_count,
                "media_unit": delta.media_unit,
                "operation": _layout_operation(delta),
                "site": registration.site,
                "activation_id": registration.token,
                "invocation_id": invocation_id,
                "binding": getattr(registration.executor.contract, "binding", None),
            }
            if registration.site == _SITE_VISIONZIP and hasattr(registration.executor, "projector_target_sites"):
                summary["target_site"] = registration.executor.projector_target_sites[0]
            if boundary_layer is None and registration.site == _SITE_TOME and hasattr(registration.executor, "output_target_site"):
                summary["target_site"] = registration.executor.output_target_site
            if boundary_layer is not None:
                summary["boundary_layer"] = boundary_layer
            if image_index is not None:
                summary["image_index"] = image_index
            self._latest_delta_by_activation[registration.token] = summary

    def _model_forward(self, _module: Any, *args: Any, **kwargs: Any) -> Any:
        assert self._original_model_forward is not None
        if not self._has_registrations():
            return self._original_model_forward(*args, **kwargs)
        call = _bind_forward_arguments(self._original_model_forward, args, kwargs)
        use_cache = call.get("use_cache", kwargs.get("use_cache"))
        if self._projector_generate.get() is not None:
            if use_cache is not True:
                raise BackendIntegrationError(
                    "Qwen2.5-VL projector-only generation requires use_cache=True."
                )
            if call.get("labels") is not None:
                raise BackendIntegrationError(
                    "Qwen2.5-VL compressed generation does not accept labels."
                )
            if (
                call.get("pixel_values_videos") is not None
                or call.get("video_grid_thw") is not None
            ):
                raise BackendIntegrationError(
                    "Pinned Qwen2.5-VL compression is image-only."
                )
            input_ids = call.get("input_ids")
            if input_ids is not None and int(input_ids.shape[0]) != 1:
                raise BackendIntegrationError(
                    "Qwen2.5-VL projector-only generation supports batch size one."
                )
            return self._original_model_forward(*args, **kwargs)
        active_lifecycle = self._decoder_lifecycle.get()
        if active_lifecycle is None and use_cache is not False:
            raise BackendIntegrationError(
                "Pinned Qwen2.5-VL compression requires FORWARD with the "
                "explicit argument use_cache=False."
            )
        if active_lifecycle is None and call.get("past_key_values") is not None:
            raise BackendIntegrationError(
                "Pinned Qwen2.5-VL compression rejects cache state and cached "
                "generation."
            )
        if active_lifecycle is not None and use_cache is not True:
            raise BackendIntegrationError(
                "Qwen2.5-VL compressed generation requires use_cache=True."
            )
        labels = call.get("labels")
        if active_lifecycle is not None and labels is not None:
            raise BackendIntegrationError(
                "Qwen2.5-VL compressed generation does not accept labels."
            )
        if labels is not None and call.get("return_dict") is False:
            raise BackendIntegrationError(
                "Compressed Qwen2.5-VL training requires return_dict=True."
            )
        if (
            call.get("pixel_values_videos") is not None
            or call.get("video_grid_thw") is not None
            or call.get("second_per_grid_ts") is not None
        ):
            raise BackendIntegrationError(
                "Pinned Qwen2.5-VL compression is image-only; video remains a "
                "separate support unit."
            )
        if call.get("pixel_values") is not None and call.get("input_ids") is None:
            raise BackendIntegrationError(
                "Compressed Qwen2.5-VL image forward requires input_ids so dense "
                "source M-RoPE positions can be legalized explicitly."
            )
        input_ids = call.get("input_ids")
        if input_ids is not None and int(input_ids.shape[0]) != 1:
            raise BackendIntegrationError(
                "Pinned Qwen2.5-VL image compression currently supports one "
                "prompt batch item (which may contain multiple images)."
            )
        if self._registrations_for(_SITE_PYRAMID):
            output_attentions = _effective_bool(
                call.get("output_attentions"),
                getattr(self.core.config, "output_attentions", False),
            )
            output_hidden_states = _effective_bool(
                call.get("output_hidden_states"),
                getattr(self.core.config, "output_hidden_states", False),
            )
            if output_attentions or output_hidden_states:
                raise BackendIntegrationError(
                    "Pinned Qwen2.5-VL decoder compression does not expose attention or "
                    "hidden-state histories after sequence mutation."
                )
        owned_lifecycle_token = None
        owned_lifecycle: ImageDecoderLifecycle | None = None
        if active_lifecycle is None and self._registrations_for(_SITE_PYRAMID):
            owned_lifecycle = self._new_decoder_lifecycle(prefix="forward")
            owned_lifecycle_token = self._decoder_lifecycle.set(owned_lifecycle)
        owned_restore_token = None
        if self._restore_runtime.get() is None and any(
            getattr(item.executor.contract, "algorithm", None) == "restore"
            for item in self._registrations.values()
        ):
            self._latest_restore_calibration = None
            owned_restore_token = self._restore_runtime.set({})
        try:
            if labels is None:
                return self._original_model_forward(*args, **kwargs)

            signature = inspect.signature(self._original_model_forward)
            bound = signature.bind_partial(*args, **kwargs)
            bound.arguments["labels"] = None
            label_token = self._training_labels.set(labels)
            try:
                output = self._original_model_forward(*bound.args, **bound.kwargs)
                repaired_labels = self._training_labels.get()
                if repaired_labels is None:
                    raise InvariantError(
                        "Qwen2.5-VL training labels disappeared during compression."
                    )
                return attach_shifted_causal_loss(
                    output,
                    repaired_labels,
                    operation="Qwen2.5-VL image compression",
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
                "Pinned Qwen2.5-VL does not support generate for this activation; "
                "generation requires one GENERATE-phase decoder compression method."
            )
        if self._decoder_lifecycle.get() is not None or (
            self._projector_generate.get() is not None
        ):
            raise BackendIntegrationError(
                "Nested Qwen2.5-VL compressed generation is unsupported."
            )
        if _is_projector_only_generate(registrations):
            return self._generate_projector_only(
                registrations[0].executor, *args, **kwargs
            )
        lifecycle = self._new_decoder_lifecycle(prefix="generate")
        token = self._decoder_lifecycle.set(lifecycle)
        native_rope_token = self._layer_local_native_rope_delta.set(None)
        restore_token = None
        if any(
            getattr(item.executor.contract, "algorithm", None) == "restore"
            for item in registrations
        ):
            self._latest_restore_calibration = None
            restore_token = self._restore_runtime.set({})
        try:
            output = self._original_generate(*args, **kwargs)
            if not lifecycle.prefill_complete:
                raise InvariantError(
                    "Qwen2.5-VL generate returned before compressed prefill completed."
                )
            return output
        finally:
            self._latest_decoder_lifecycle = lifecycle.telemetry()
            if restore_token is not None:
                self._restore_runtime.reset(restore_token)
            self._layer_local_native_rope_delta.reset(native_rope_token)
            self._decoder_lifecycle.reset(token)

    def _generate_projector_only(self, executor: Any, *args: Any, **kwargs: Any) -> Any:
        assert self._original_generate is not None
        state: dict[str, Any] = {
            "prefill_calls": 0,
            "prompt_removed": 0,
            "expected_decode_position": None,
            "observed_decode_position": None,
            "decode_steps": 0,
            "decode_position_verified": None,
        }
        token = self._projector_generate.set(state)
        try:
            output = self._original_generate(*args, **kwargs)
            if state["prefill_calls"] != 1:
                raise InvariantError(
                    "Qwen2.5-VL projector-only generate expected exactly one image "
                    f"prefill, saw {state['prefill_calls']}."
                )
            return output
        finally:
            telemetry = {
                key: value for key, value in state.items() if key != "visual_log_mass"
            }
            mass = state.get("visual_log_mass")
            if mass is not None:
                telemetry["attention_mass_correction"] = {
                    "applied": True,
                    "sequence_positions": int(mass.numel()),
                    "nonzero_visual_positions": int((mass != 0).sum().item()),
                    "max_log_mass": float(mass.max().item()),
                }
            else:
                telemetry["attention_mass_correction"] = {"applied": False}
            self._latest_projector_generate = telemetry
            self._projector_generate.reset(token)
            recorder = getattr(executor, "record_projector_generate", None)
            if callable(recorder):
                recorder(dict(telemetry))

    def _new_decoder_lifecycle(self, *, prefix: str) -> ImageDecoderLifecycle:
        registrations = self._registrations_for(_SITE_PYRAMID)
        if len(registrations) != 1:
            raise BackendIntegrationError(
                "Qwen2.5-VL decoder lifecycle requires exactly one decoder image method."
            )
        self._decoder_request_counter += 1
        self._latest_decode_position = None
        self._latest_rope_delta_repair = None
        return ImageDecoderLifecycle(
            request_id=f"{prefix}-{self._decoder_request_counter}",
            algorithm=str(registrations[0].executor.contract.algorithm),
            num_layers=len(self.language_model.layers),
            requires_decoder_boundary=(
                registrations[0].executor.contract.algorithm != "restore"
            ),
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
                    operation=(
                        f"Qwen2.5-VL decoder {boundary.value} layer {layer_index}"
                    ),
                )
            )

    def _record_decode_position(
        self,
        *,
        cache_position: Any,
        position_ids: Any,
    ) -> None:
        if self._latest_decode_position is not None:
            return
        cache_values = _tensor_int_values(cache_position)
        rope_values = _tensor_int_values(getattr(self.core, "rope_deltas", None))
        rotary_positions = (
            position_ids[1:] if int(position_ids.shape[0]) == 4 else position_ids
        )
        observed = _tensor_int_values(rotary_positions[..., -1])
        expected = None
        matches = None
        if cache_values and rope_values:
            expected = cache_values[0] + rope_values[0]
            matches = bool(observed) and all(value == expected for value in observed)
        self._latest_decode_position = {
            "incoming_cache_position": cache_values,
            "provider_rope_delta": rope_values,
            "expected_rotary_position": expected,
            "observed_rotary_positions": observed,
            "matches_cache_position_plus_rope_delta": matches,
        }

    def _core_forward(self, _module: Any, *args: Any, **kwargs: Any) -> Any:
        assert self._original_core_forward is not None
        plan_token = self._visual_plan.set(None)
        removal_token = self._prompt_removal_mask.set(None)
        image_token = self._prompt_image_mask.set(None)
        image_count_token = self._image_count.set(None)
        text_embedding_token = self._prompt_text_embeddings.set(None)
        prompt_ids_token = self._prompt_input_ids.set(None)
        projector_registrations = self._registrations_for(_SITE_VISIONZIP)
        bifocal_active = _has_bifocal_registration(projector_registrations)
        projector_generate_active = self._projector_generate.get() is not None
        projector_generate_prefill = False
        try:
            if bifocal_active:
                call = _bind_forward_arguments(
                    self._original_core_forward, args, kwargs
                )
                input_ids = call.get("input_ids")
                projector_generate_prefill = call.get("pixel_values") is not None
                if input_ids is not None and projector_generate_prefill:
                    if int(input_ids.shape[0]) != 1:
                        raise BackendIntegrationError(
                            "Qwen2.5-VL Bifocal requires one input_ids prompt batch."
                        )
                    self._prompt_input_ids.set(input_ids[0])
            elif projector_generate_active:
                call = _bind_forward_arguments(
                    self._original_core_forward, args, kwargs
                )
                projector_generate_prefill = call.get("pixel_values") is not None
            if (not projector_generate_active or projector_generate_prefill) and any(
                getattr(item.executor.contract, "algorithm", None) == "mmtok"
                for item in self._registrations_for(_SITE_VISIONZIP)
            ):
                call = _bind_forward_arguments(
                    self._original_core_forward, args, kwargs
                )
                input_ids = call.get("input_ids")
                if input_ids is None or int(input_ids.shape[0]) != 1:
                    raise BackendIntegrationError(
                        "Qwen2.5-VL MMTok requires one input_ids prompt batch."
                    )
                attention_mask = call.get("attention_mask")
                mmtok_executor = next(item.executor for item in projector_registrations
                    if getattr(item.executor.contract, "algorithm", None) == "mmtok")
                self._prompt_text_embeddings.set(
                    mmtok_executor.mmtok_prompt_embeddings(input_ids, attention_mask)
                )
            output = self._original_core_forward(*args, **kwargs)
            removal = self._prompt_removal_mask.get()
            prompt_removed = int(removal.sum().item()) if removal is not None else 0
            lifecycle = self._decoder_lifecycle.get()
            selected_methods = [getattr(item.executor, "decoder_selection", None)
                for item in self._registrations_for(_SITE_PYRAMID)]
            sparse_contiguous = lifecycle is not None and any(
                method is not None and method.position_policy == "contiguous"
                for method in selected_methods)
            layer_local_contiguous = sparse_contiguous and any(
                method is not None and method.position_policy == "contiguous"
                and method.cache_policy is CacheRepairPolicy.LAYER_LOCAL
                for method in selected_methods)
            decoder_removed = 0
            if (
                lifecycle is not None
                and sparse_contiguous
                and lifecycle.source_length is not None
                and lifecycle.current_length is not None
            ):
                decoder_removed = lifecycle.source_length - lifecycle.current_length
            rope_deltas = getattr(output, "rope_deltas", None)
            native_rope_delta = _tensor_int_values(rope_deltas)
            if (
                lifecycle is not None
                and layer_local_contiguous
                and self._layer_local_native_rope_delta.get() is None
                and native_rope_delta
            ):
                self._layer_local_native_rope_delta.set(tuple(native_rope_delta))
            if sparse_contiguous and prompt_removed:
                raise InvariantError(
                    "Qwen2.5-VL contiguous decoder compression cannot compose with pre-LLM prompt "
                    "compression in the one-method image contract."
                )
            rope_adjustment = prompt_removed - decoder_removed
            if bifocal_active or projector_generate_active:
                rope_adjustment = 0
            if sparse_contiguous and rope_deltas is not None:
                adjusted = rope_deltas.new_full(
                    rope_deltas.shape,
                    -decoder_removed,
                )
                output.rope_deltas = adjusted
                self.core.rope_deltas = adjusted
            elif rope_adjustment and rope_deltas is not None:
                adjusted = rope_deltas + rope_adjustment
                output.rope_deltas = adjusted
                self.core.rope_deltas = adjusted
            if removal is not None or self._latest_rope_delta_repair is None:
                self._latest_rope_delta_repair = {
                    "native_rope_delta": native_rope_delta,
                    "prompt_removed_tokens": prompt_removed,
                    "decoder_removed_tokens_added": -decoder_removed,
                    "published_rope_delta": _tensor_int_values(
                        getattr(output, "rope_deltas", None)
                    ),
                    **(
                        {
                            "policy": (
                                "bifocal_no_correction"
                                if bifocal_active
                                else "projector_generate_no_correction"
                            )
                        }
                        if bifocal_active or projector_generate_active
                        else {}
                    ),
                }
            state = self._projector_generate.get()
            if state is not None and projector_generate_prefill:
                state["prefill_calls"] += 1
                state["prompt_removed"] = prompt_removed
            return output
        finally:
            self._prompt_input_ids.reset(prompt_ids_token)
            self._prompt_text_embeddings.reset(text_embedding_token)
            self._image_count.reset(image_count_token)
            self._prompt_image_mask.reset(image_token)
            self._prompt_removal_mask.reset(removal_token)
            self._visual_plan.reset(plan_token)

    def _get_visual_features(
        self,
        pixel_values: Any,
        grid_thw: Any,
        *,
        modality: Modality,
    ) -> Any:
        assert self._original_get_video_features is not None
        if grid_thw is None:
            raise BackendIntegrationError(
                f"Qwen2.5-VL {modality.value} features require grid_thw."
            )
        if modality is Modality.VIDEO:
            token = self._visual_modality.set(Modality.VIDEO.value)
            try:
                return self._original_get_video_features(
                    pixel_values, video_grid_thw=grid_thw
                )
            finally:
                self._visual_modality.reset(token)

        torch = _torch()
        token = self._visual_modality.set(Modality.IMAGE.value)
        self._image_count.set(int(grid_thw.shape[0]))
        self._comparison_grid.set(grid_thw)
        self._visual_plan.set(None)
        try:
            pixels = pixel_values.type(self.visual.dtype)
            embeds = self.visual(pixels, grid_thw=grid_thw)
        finally:
            self._visual_modality.reset(token)
        dense_counts = tuple(
            int(value)
            for value in (
                grid_thw.prod(-1) // self.visual.spatial_merge_size**2
            ).tolist()
        )
        plan = self._visual_plan.get()
        if plan is None:
            plan = _VisualCompressionPlan(
                source_counts=dense_counts,
                result_counts=dense_counts,
                layout_deltas=tuple(
                    LayoutDelta.identity(count, media_unit="image_feature")
                    for count in dense_counts
                ),
            )
            self._visual_plan.set(plan)
        if plan.source_counts != dense_counts:
            raise InvariantError(
                "Qwen2.5-VL visual plan source counts disagree with grid_thw."
            )
        if sum(plan.result_counts) != int(embeds.shape[0]):
            raise InvariantError(
                "Qwen2.5-VL visual plan result counts disagree with features."
            )
        return torch.split(embeds, plan.result_counts)

    def _get_placeholder_mask(
        self,
        input_ids: Any,
        inputs_embeds: Any,
        *,
        image_features: Any = None,
        video_features: Any = None,
    ) -> tuple[Any, Any]:
        assert self._original_get_placeholder_mask is not None
        plan = self._visual_plan.get()
        if image_features is None or plan is None or not plan.compressed:
            image_mask, video_mask = self._original_get_placeholder_mask(
                input_ids,
                inputs_embeds,
                image_features=image_features,
                video_features=video_features,
            )
            if image_features is not None:
                dense = image_mask[..., 0].bool()
                self._prompt_image_mask.set(dense)
                self._prompt_removal_mask.set(dense & False)
            return image_mask, video_mask

        torch = _torch()
        dense_image_mask, video_mask = self._original_get_placeholder_mask(
            input_ids,
            inputs_embeds,
            image_features=None,
            video_features=video_features,
        )
        dense = dense_image_mask[..., 0].bool()
        if int(dense.shape[0]) != 1:
            raise BackendIntegrationError(
                "Compressed Qwen2.5-VL placeholder repair requires batch size one."
            )
        flat_positions = dense.reshape(-1).nonzero(as_tuple=False).flatten()
        if int(flat_positions.numel()) != sum(plan.source_counts):
            raise InvariantError(
                "Dense image placeholders do not match compression source counts."
            )
        retained_offsets: list[int] = []
        source_offset = 0
        for count, delta in zip(plan.source_counts, plan.layout_deltas):
            if delta.source_count != count:
                raise InvariantError(
                    "Compression LayoutDelta source count differs from image plan."
                )
            retained_offsets.extend(
                source_offset + index for index in _layout_representatives(delta)
            )
            source_offset += count
        selected_flat = torch.zeros_like(dense.reshape(-1), dtype=torch.bool)
        offset_tensor = torch.tensor(
            retained_offsets,
            dtype=torch.long,
            device=flat_positions.device,
        )
        selected_flat[flat_positions.index_select(0, offset_tensor)] = True
        selected = selected_flat.view_as(dense)
        removal = dense & ~selected
        expanded = selected.unsqueeze(-1).expand_as(inputs_embeds)
        if inputs_embeds[expanded].numel() != image_features.numel():
            raise InvariantError(
                "Compressed image placeholders do not match image features."
            )
        self._prompt_image_mask.set(selected)
        self._prompt_removal_mask.set(removal)
        return expanded, video_mask

    def _visual_forward(self, _module: Any, *args: Any, **kwargs: Any) -> Any:
        assert self._original_visual_forward is not None
        tome = self._registrations_for(_SITE_TOME)
        saint = self._registrations_for(_SITE_SAINT)
        visionzip = self._registrations_for(_SITE_VISIONZIP)
        if not tome and not saint and not visionzip:
            return self._original_visual_forward(*args, **kwargs)
        modality = self._visual_modality.get()
        if modality == Modality.VIDEO.value:
            return self._original_visual_forward(*args, **kwargs)
        if modality != Modality.IMAGE.value:
            raise BackendIntegrationError(
                "Compressed visual.forward must be entered through "
                "Qwen2_5_VLModel.get_image_features; a direct call cannot prove "
                "image rather than video modality."
            )
        hidden_states, grid_thw, provider_kwargs = _bind_visual_call(
            self._original_visual_forward, args, kwargs
        )
        return self._execute_image_visual_forward(
            hidden_states,
            grid_thw,
            provider_kwargs,
            tome_registrations=tome,
            saint_registrations=saint,
            visionzip_registrations=visionzip,
        )

    def _execute_image_visual_forward(
        self,
        hidden_states: Any,
        grid_thw: Any,
        provider_kwargs: Mapping[str, Any],
        *,
        tome_registrations: Sequence[_Registration],
        saint_registrations: Sequence[_Registration],
        visionzip_registrations: Sequence[_Registration],
    ) -> Any:
        torch = _torch()
        if int(grid_thw.shape[0]) < 1 or not bool((grid_thw[:, 0] == 1).all()):
            raise BackendIntegrationError(
                "Pinned Qwen2.5-VL image compression requires T=1 for every "
                "image; video is a separate support unit."
            )
        if saint_registrations:
            if tome_registrations or visionzip_registrations:
                raise BackendIntegrationError(
                    "Qwen2.5-VL SAINT cannot share a model instance with another "
                    "image compression method."
                )
            return self._execute_saint_image_visual_forward(
                hidden_states,
                grid_thw,
                provider_kwargs,
                saint_registrations,
            )
        if int(grid_thw.shape[0]) != 1 and (self._projector_generate.get() is not None or any(
            getattr(registration.executor.contract, "algorithm", None)
            in {
                "zoo_prune",
                "nuwa",
                "restore",
                "bifocal",
                "feature_anchor_mean",
                "grid_mean",
                "prumerge",
            }
            for registration in visionzip_registrations
        )):
            raise BackendIntegrationError(
                "This Qwen2.5-VL projector compression path supports exactly one image."
            )
        visual = self.visual
        hidden_states = visual.patch_embed(hidden_states)
        rotary_pos_emb = visual.rot_pos_emb(grid_thw)
        window_index, cu_window_values = visual.get_window_index(grid_thw)
        cu_window_seqlens = torch.tensor(
            cu_window_values,
            device=hidden_states.device,
            dtype=torch.int32,
        )
        cu_window_seqlens = torch.unique_consecutive(cu_window_seqlens)
        sequence_length = int(hidden_states.shape[0])
        unit = int(visual.spatial_merge_unit)
        if sequence_length % unit:
            raise InvariantError(
                "Qwen2.5-VL visual patch count is not divisible by spatial merge unit."
            )
        hidden_states = hidden_states.reshape(sequence_length // unit, unit, -1)
        hidden_states = hidden_states[window_index].reshape(sequence_length, -1)
        rotary_pos_emb = rotary_pos_emb.reshape(sequence_length // unit, unit, -1)
        rotary_pos_emb = rotary_pos_emb[window_index].reshape(sequence_length, -1)
        embedding = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
        position_embeddings = (embedding.cos(), embedding.sin())

        patch_counts = tuple(int(value) for value in grid_thw.prod(-1).tolist())
        cu_full = _cumulative_lengths(patch_counts, device=grid_thw.device)
        final_query: Any | None = None
        final_key: Any | None = None
        final_key_sizes: Any | None = None
        projector_requires_metrics = any(
            bool(
                getattr(
                    getattr(registration.executor, "contract", None),
                    "requires_visual_metrics",
                    False,
                )
            )
            for registration in visionzip_registrations
        )
        nuwa_registration = next(
            (
                registration
                for registration in visionzip_registrations
                if getattr(registration.executor.contract, "algorithm", None)
                == "nuwa"
            ),
            None,
        )
        bifocal_registration = next(
            (
                registration
                for registration in visionzip_registrations
                if getattr(registration.executor.contract, "algorithm", None)
                == "bifocal"
            ),
            None,
        )
        bifocal_spec = (
            getattr(bifocal_registration.executor, "_bifocal_spec_value", None)
            if bifocal_registration is not None
            else None
        )
        bifocal_vit_block = (
            int(bifocal_spec.vit_block)
            if bifocal_spec is not None and bifocal_spec.needs_vit_maps
            else -1
        )
        if bifocal_vit_block >= 0 and bifocal_vit_block not in set(
            visual.fullatt_block_indexes
        ):
            raise BackendIntegrationError(
                f"Qwen2.5-VL Bifocal vit_block={bifocal_vit_block} is not a "
                "full-attention block; choose one of "
                f"{list(visual.fullatt_block_indexes)}."
            )
        bifocal_patch_maps: dict[str, Any] | None = None
        nuwa_semantic_sum = None
        nuwa_semantic_count = 0
        nuwa_spec = (
            getattr(nuwa_registration.executor, "_nuwa_spec_value", None)
            if nuwa_registration is not None
            else None
        )
        if (
            nuwa_spec is not None
            and nuwa_spec.semantic_hidden_start <= 0 < nuwa_spec.semantic_hidden_end
        ):
            nuwa_semantic_sum = hidden_states.float()
            nuwa_semantic_count = 1
        score_layers = {
            id(registration): tome_encoder_score_layer(registration.executor.stage, "qwen25_vl", len(visual.blocks))
            for registration in tome_registrations
        }
        score_keys = {}
        for layer_index, block in enumerate(visual.blocks):
            cu_now = (
                cu_full
                if layer_index in set(visual.fullatt_block_indexes)
                else cu_window_seqlens
            )
            normalised = block.norm1(hidden_states)
            if layer_index in score_layers.values() and layer_index not in score_keys:
                _, source_keys = _vision_rotated_qk(block, normalised, position_embeddings)
                score_keys[layer_index] = source_keys.detach()
            if layer_index == len(visual.blocks) - 1 and (
                tome_registrations or projector_requires_metrics
            ):
                final_query, final_key = _vision_rotated_qk(
                    block, normalised, position_embeddings
                )
            if layer_index == bifocal_vit_block:
                bifocal_patch_maps = _bifocal_vit_column_mass(
                    block, normalised, cu_full, position_embeddings
                )
            hidden_states, actual_reference = self._execute_visual_block(
                block,
                hidden_states,
                normalised,
                cu_now,
                position_embeddings,
                provider_kwargs,
                tome_registrations,
                layer_index=layer_index, score_layers=score_layers, score_keys=score_keys,
            )
            if layer_index == len(visual.blocks) - 1 and actual_reference is not None:
                final_query, final_key, final_key_sizes = actual_reference
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

        pre_projector_registrations = tuple(registration for registration in visionzip_registrations
            if getattr(registration.executor, "projector_target_sites", ()) == ("pre_vision_merger",))
        pre_tome_registrations = tuple(registration for registration in tome_registrations
            if getattr(registration.executor, "output_target_site", None) == "pre_vision_merger")
        reverse_indices = torch.argsort(window_index)
        bifocal_map_chunks: list[dict[str, Any]] | None = None
        if bifocal_patch_maps is not None:
            bifocal_sizes = tuple(count // unit for count in patch_counts)
            per_name = {
                name: torch.split(
                    value.view(-1, unit).mean(dim=-1)[reverse_indices], bifocal_sizes
                )
                for name, value in bifocal_patch_maps.items()
            }
            bifocal_map_chunks = [
                {name: chunks[index] for name, chunks in per_name.items()}
                for index in range(len(bifocal_sizes))
            ]
        # The original post path keeps its native merger invocation and order.
        merged = None if pre_projector_registrations or pre_tome_registrations else visual.merger(hidden_states)[reverse_indices]
        projected_reference = None
        if any(getattr(registration.executor.contract, "algorithm", None) in {"mmtok", "zoo_prune"}
                for registration in pre_projector_registrations):
            # Their fixed selection rules still require the dense projected
            # features; only selected raw groups take the differentiable path.
            with torch.no_grad():
                projected_reference = visual.merger(hidden_states)[reverse_indices]
        # The native merger concatenates each spatial group before projection;
        # keep that complete group structure for MMTok's visual reference.
        vision_reference = hidden_states.reshape(
            sequence_length // unit, -1
        )[reverse_indices]
        if nuwa_registration is not None:
            if nuwa_semantic_sum is None or nuwa_semantic_count < 1:
                raise BackendIntegrationError(
                    "Qwen2.5-VL Nüwa semantic hidden-state range was not captured."
                )
            semantic_reference = (
                (nuwa_semantic_sum / float(nuwa_semantic_count))
                .reshape(sequence_length // unit, -1)[reverse_indices]
                .to(vision_reference.dtype)
            )
        else:
            semantic_reference = vision_reference
        projector_inputs = None
        projector_layers = None
        if any(
            getattr(registration.executor.contract, "algorithm", None)
            == "zoo_prune"
            for registration in visionzip_registrations
        ):
            norm = visual.merger.ln_q
            first_linear = visual.merger.mlp[0]
            norm_device = getattr(getattr(norm, "_hf_hook", None), "execution_device", None)
            if norm_device is None:
                norm_device = getattr(norm, "weight", hidden_states).device
            projector_device = getattr(getattr(first_linear, "_hf_hook", None), "execution_device", None)
            if projector_device is None:
                projector_device = first_linear.weight.device
            projector_inputs = visual.merger.ln_q(hidden_states.to(norm_device)).reshape(
                sequence_length // unit, -1
            )[reverse_indices].to(projector_device)
            projector_layers = tuple(visual.merger.mlp.children())
        feature_counts = tuple(count // unit for count in patch_counts)
        identity = tuple(
            LayoutDelta.identity(count, media_unit="image_feature")
            for count in feature_counts
        )
        if not tome_registrations and not visionzip_registrations:
            self._visual_plan.set(
                _VisualCompressionPlan(feature_counts, feature_counts, identity)
            )
            return merged
        if final_query is None or final_key is None:
            if tome_registrations or projector_requires_metrics:
                raise InvariantError(
                    "Required final visual Q/K metrics were not captured."
                )
            attention_chunks = [None] * len(feature_counts)
            key_chunks = [None] * len(feature_counts)
        elif projector_requires_metrics:
            attention_chunks, key_chunks = _merged_vision_metrics(
                final_query,
                final_key,
                patch_counts=patch_counts,
                feature_counts=feature_counts,
                window_index=window_index,
                spatial_merge_unit=unit,
                scaling=float(visual.blocks[-1].attn.scaling),
                key_sizes=final_key_sizes,
            )
        else:
            attention_chunks = [None] * len(feature_counts)
            key_chunks = _merged_vision_keys(
                final_key,
                patch_counts=patch_counts,
                feature_counts=feature_counts,
                window_index=window_index,
                spatial_merge_unit=unit,
            )
        feature_grids = tuple(
            (
                int(row[1].item()) // int(visual.spatial_merge_size),
                int(row[2].item()) // int(visual.spatial_merge_size),
            )
            for row in grid_thw
        )
        cumulative_deltas = list(identity)
        current_feature_counts = feature_counts

        def compact_tome_output(registration: _Registration, chunks: list[Any]) -> tuple[LayoutDelta, ...]:
            deltas = []
            for image_index, chunk in enumerate(tuple(chunks)):
                result = registration.executor.compact_visual_output(
                    chunk, key_chunks[image_index].permute(1, 0, 2).contiguous(),
                )
                delta = result.layout_delta
                chunks[image_index] = result.hidden_states
                cumulative_deltas[image_index] = _compose_layout_mapping(cumulative_deltas[image_index], delta)
                key_chunks[image_index] = _mean_keys_by_delta(key_chunks[image_index], delta)
                if attention_chunks[image_index] is not None:
                    attention_chunks[image_index] = _mean_square_by_delta(attention_chunks[image_index], delta)
                self._record_delta(registration, delta, image_index=image_index)
                deltas.append(delta)
            return tuple(deltas)

        if pre_projector_registrations or pre_tome_registrations:
            # Invert the window permutation before selecting/merging complete
            # groups. Each output row retains its four within-group patches.
            raw_chunks = list(torch.split(vision_reference, feature_counts))
            original_semantic_chunks = torch.split(semantic_reference, feature_counts)
            projected_chunks = torch.split(projected_reference, feature_counts) if projected_reference is not None else [None] * len(feature_counts)
            proxy_input_chunks = torch.split(projector_inputs, feature_counts) if projector_inputs is not None else [None] * len(feature_counts)
            for registration in pre_projector_registrations:
                algorithm = getattr(registration.executor.contract, "algorithm", None)
                for image_index, chunk in enumerate(tuple(raw_chunks)):
                    count = int(chunk.shape[0])
                    result = registration.executor.post_visual_merger(
                        chunk, attention_chunks[image_index], key_chunks[image_index],
                        placeholder_indices=tuple(range(count)), source_sequence_length=count,
                        image_index=image_index,
                        vision_reference=original_semantic_chunks[image_index] if algorithm == "nuwa" else chunk,
                        projected_reference=projected_chunks[image_index],
                        text_embeddings=self._prompt_text_embeddings.get(),
                        projector_inputs=proxy_input_chunks[image_index], projector_layers=projector_layers,
                        **({"feature_grid": feature_grids[image_index]} if algorithm == "nuwa" else {}),
                    )
                    delta = result.prompt_repair.layout_delta
                    raw_chunks[image_index] = result.kernel_result.tokens
                    if algorithm == "restore":
                        runtime = self._restore_runtime.get()
                        if runtime is None:
                            raise InvariantError(
                                "Qwen2.5-VL RESTORE visual merge ran outside its request lifecycle."
                            )
                        runtime["visual_sizes"] = result.kernel_result.token_sizes
                        runtime["source_visual_count"] = int(delta.source_count)
                        if not cumulative_deltas[image_index].is_identity:
                            raise InvariantError(
                                "Qwen2.5-VL RESTORE requires an unmodified source layout."
                            )
                        # Preserve membership, explicit representatives and mass
                        # for the original decoder attention calibration.
                        cumulative_deltas[image_index] = delta
                    else:
                        cumulative_deltas[image_index] = _compose_layout_mapping(cumulative_deltas[image_index], delta)
                    if attention_chunks[image_index] is not None:
                        attention_chunks[image_index] = _mean_square_by_delta(attention_chunks[image_index], delta)
                    if key_chunks[image_index] is not None:
                        key_chunks[image_index] = _mean_keys_by_delta(key_chunks[image_index], delta)
                    self._record_delta(registration, delta, image_index=image_index)
            for registration in pre_tome_registrations:
                compact_tome_output(registration, raw_chunks)
            current_feature_counts = tuple(int(chunk.shape[0]) for chunk in raw_chunks)
            vision_reference = torch.cat(raw_chunks, dim=0)
            compact_patches = vision_reference.reshape(-1, hidden_states.shape[-1])
            # Start at the merger's original LN, and do not apply the old
            # N-group reverse permutation to these K already-spatial groups.
            merged = visual.merger(compact_patches)
            if projector_inputs is not None:
                if any(getattr(registration.executor.contract, "algorithm", None) == "zoo_prune"
                        and registration not in pre_projector_registrations for registration in visionzip_registrations):
                    projector_inputs = visual.merger.ln_q(compact_patches.to(norm_device)).reshape(sum(current_feature_counts), -1).to(projector_device)
                else:
                    projector_inputs = None
            if nuwa_registration is None:
                semantic_reference = vision_reference
            else:
                semantic_chunks = torch.split(semantic_reference, feature_counts)
                semantic_reference = torch.cat([
                    _mean_keys_by_delta(chunk.unsqueeze(0), delta).squeeze(0)
                    for chunk, delta in zip(semantic_chunks, cumulative_deltas)
                ], dim=0)
        token_chunks = list(torch.split(merged, current_feature_counts))
        vision_reference_chunks = list(
            torch.split(vision_reference, current_feature_counts)
        )
        semantic_reference_chunks = list(
            torch.split(semantic_reference, current_feature_counts)
        )
        projector_input_chunks = (
            list(torch.split(projector_inputs, current_feature_counts))
            if projector_inputs is not None
            else [None] * len(feature_counts)
        )
        for registration in tome_registrations:
            if registration in pre_tome_registrations:
                continue
            deltas = compact_tome_output(registration, token_chunks)
            if any(getattr(item.executor.contract, "algorithm", None) == "vispruner"
                    and item not in pre_projector_registrations for item in visionzip_registrations):
                for image_index, delta in enumerate(deltas):
                    vision_reference_chunks[image_index] = _mean_keys_by_delta(
                        vision_reference_chunks[image_index].unsqueeze(0), delta,
                    ).squeeze(0)
        for registration in visionzip_registrations:
            if registration in pre_projector_registrations:
                continue
            for image_index, chunk in enumerate(tuple(token_chunks)):
                count = int(chunk.shape[0])
                result = registration.executor.post_visual_merger(
                    chunk,
                    attention_chunks[image_index],
                    key_chunks[image_index],
                    placeholder_indices=tuple(range(count)),
                    source_sequence_length=count,
                    image_index=image_index,
                    vision_reference=(
                        semantic_reference_chunks[image_index]
                        if getattr(
                            registration.executor.contract, "algorithm", None
                        )
                        == "nuwa"
                        else vision_reference_chunks[image_index]
                    ),
                    text_embeddings=self._prompt_text_embeddings.get(),
                    projector_inputs=projector_input_chunks[image_index],
                    projector_layers=projector_layers,
                    **(
                        {"feature_grid": feature_grids[image_index]}
                        if getattr(
                            registration.executor.contract, "algorithm", None
                        )
                        in {"nuwa", "feature_anchor_mean", "grid_mean"}
                        else {}
                    ),
                    **(
                        {
                            "feature_grid": feature_grids[image_index],
                            "vision_maps": (
                                bifocal_map_chunks[image_index]
                                if bifocal_map_chunks is not None
                                else None
                            ),
                            "prompt_input_ids": self._prompt_input_ids.get(),
                        }
                        if registration is bifocal_registration
                        else {}
                    ),
                )
                kernel = result.kernel_result
                delta = result.prompt_repair.layout_delta
                if getattr(
                    registration.executor.contract, "algorithm", None
                ) == "restore":
                    runtime = self._restore_runtime.get()
                    if runtime is None:
                        raise InvariantError(
                            "Qwen2.5-VL RESTORE visual merge ran outside its request lifecycle."
                        )
                    runtime["visual_sizes"] = kernel.token_sizes
                    runtime["source_visual_count"] = int(delta.source_count)
                token_chunks[image_index] = kernel.tokens
                if getattr(
                    registration.executor.contract, "algorithm", None
                ) == "restore":
                    if not cumulative_deltas[image_index].is_identity:
                        raise InvariantError(
                            "Qwen2.5-VL RESTORE requires an unmodified source layout."
                        )
                    cumulative_deltas[image_index] = delta
                else:
                    cumulative_deltas[image_index] = _compose_layout_mapping(
                        cumulative_deltas[image_index], delta
                    )
                if attention_chunks[image_index] is not None:
                    attention_chunks[image_index] = _mean_square_by_delta(
                        attention_chunks[image_index], delta
                    )
                if key_chunks[image_index] is not None:
                    key_chunks[image_index] = _mean_keys_by_delta(
                        key_chunks[image_index], delta
                    )
                self._record_delta(registration, delta, image_index=image_index)
        result_counts = tuple(int(chunk.shape[0]) for chunk in token_chunks)
        self._visual_plan.set(
            _VisualCompressionPlan(
                source_counts=feature_counts,
                result_counts=result_counts,
                layout_deltas=tuple(cumulative_deltas),
            )
        )
        return torch.cat(token_chunks, dim=0)

    def _execute_saint_image_visual_forward(
        self,
        hidden_states: Any,
        grid_thw: Any,
        provider_kwargs: Mapping[str, Any],
        registrations: Sequence[_Registration],
    ) -> Any:
        """Run permanent progressive SAINT pruning in source patch order."""

        torch = _torch()
        if len(registrations) != 1:
            raise BackendIntegrationError(
                "Qwen2.5-VL SAINT requires exactly one encoder registration."
            )
        registration = registrations[0]
        visual = self.visual
        hidden_states = visual.patch_embed(hidden_states)
        rotary = visual.rot_pos_emb(grid_thw)
        source_patch_counts = tuple(int(value) for value in grid_thw.prod(-1).tolist())
        current_counts = source_patch_counts
        unit = int(visual.spatial_merge_unit)
        if any(count % unit for count in source_patch_counts):
            raise InvariantError(
                "Qwen2.5-VL SAINT requires every image patch span to align to "
                "the native spatial merge unit."
            )
        cu_full = _cumulative_lengths(current_counts, device=hidden_states.device)
        window_tokens = (
            int(visual.window_size)
            // int(visual.spatial_merge_size)
            // int(visual.patch_size)
        ) ** 2 * unit
        cu_window = _sequential_window_cu(
            cu_full, window_tokens=window_tokens, device=hidden_states.device
        )
        embedding = torch.cat((rotary, rotary), dim=-1)
        position_embeddings = (embedding.cos(), embedding.sin())
        tracked = list(
            torch.split(
                torch.arange(
                    int(hidden_states.shape[0]),
                    dtype=torch.long,
                    device=hidden_states.device,
                ),
                source_patch_counts,
            )
        )
        full_attention = set(int(value) for value in visual.fullatt_block_indexes)
        application_positions = registration.executor.application_positions

        def apply_pruning(point):
            nonlocal hidden_states, position_embeddings, tracked, current_counts, cu_full, cu_window
            if point not in application_positions:
                return
            stage_index = application_positions.index(point)
            key_projection_layer = registration.schedule[stage_index]
            metric_block = visual.blocks[key_projection_layer]
            # SAINT projects the current values through the configured module;
            # this is a raw-key transform, not a historical Attention snapshot.
            keys = _vision_raw_keys(metric_block, metric_block.norm1(hidden_states)).reshape(
                int(hidden_states.shape[0]), -1,
            ).to(device=hidden_states.device)
            token_chunks = list(torch.split(hidden_states, current_counts))
            key_chunks = list(torch.split(keys, current_counts))
            cos_chunks = list(torch.split(position_embeddings[0], current_counts))
            sin_chunks = list(torch.split(position_embeddings[1], current_counts))
            next_tokens: list[Any] = []
            next_cos: list[Any] = []
            next_sin: list[Any] = []
            next_tracked: list[Any] = []
            next_counts: list[int] = []
            for image_index, (tokens, image_keys) in enumerate(
                zip(token_chunks, key_chunks)
            ):
                result = registration.executor.encoder_progressive_prune(
                    tokens,
                    image_keys,
                    boundary_layer=key_projection_layer,
                    image_index=image_index,
                )
                selected = result.selected_indices
                next_tokens.append(result.tokens)
                next_cos.append(cos_chunks[image_index].index_select(0, selected.to(cos_chunks[image_index].device)))
                next_sin.append(sin_chunks[image_index].index_select(0, selected.to(sin_chunks[image_index].device)))
                next_tracked.append(tracked[image_index].index_select(0, selected.to(tracked[image_index].device)))
                next_counts.append(int(result.tokens.shape[0]))
                self._record_delta(
                    registration,
                    result.layout_delta,
                    image_index=image_index,
                    boundary_layer=key_projection_layer,
                )
            hidden_states = torch.cat(next_tokens, dim=0)
            position_embeddings = (torch.cat(next_cos, dim=0), torch.cat(next_sin, dim=0))
            tracked = next_tracked
            current_counts = tuple(next_counts)
            cu_full = _cumulative_lengths(current_counts, device=hidden_states.device)
            cu_window = _sequential_window_cu(
                cu_full, window_tokens=window_tokens, device=hidden_states.device
            )

        apply_pruning(LayerPosition("inter_layer", 0))
        for layer_index, block in enumerate(visual.blocks):
            cu_now = cu_full if layer_index in full_attention else cu_window
            normalised = block.norm1(hidden_states)
            attention = block.attn(
                normalised,
                cu_seqlens=cu_now,
                position_embeddings=position_embeddings,
                **dict(provider_kwargs),
            )
            hidden_states = hidden_states + attention
            apply_pruning(LayerPosition("post_attention_pre_ffn", layer_index))
            hidden_states = hidden_states + block.mlp(block.norm2(hidden_states))
            apply_pruning(LayerPosition("inter_layer", layer_index + 1))
        if any(count % unit for count in current_counts):
            raise InvariantError(
                "Qwen2.5-VL SAINT produced a patch span incompatible with the "
                "native spatial merger."
            )
        merged = visual.merger(hidden_states)
        source_feature_counts = tuple(count // unit for count in source_patch_counts)
        result_feature_counts = tuple(count // unit for count in current_counts)
        feature_deltas: list[LayoutDelta] = []
        for image_index, (
            original_patches,
            source_features,
            result_features,
        ) in enumerate(zip(tracked, source_feature_counts, result_feature_counts)):
            candidate = original_patches.reshape(result_features, unit)[:, 0]
            image_patch_offset = sum(source_patch_counts[:image_index])
            candidate = (candidate - image_patch_offset) // unit
            representatives = _strict_feature_representatives(
                candidate,
                source_count=source_features,
            )
            mapping: list[int | None] = [None] * source_features
            for result_index, source_index in enumerate(representatives):
                mapping[source_index] = result_index
            feature_delta = LayoutDelta.from_mapping(
                source_count=source_features,
                result_count=result_features,
                old_to_new=mapping,
                media_unit="image_feature",
                source_layout={
                    "backbone": "qwen25_vl",
                    "image_index": image_index,
                    "source": "pre_saint_spatial_merger",
                },
                result_layout={
                    "backbone": "qwen25_vl",
                    "image_index": image_index,
                    "representative_indices": representatives,
                    "position_policy": "sparse_mrope_rebuild",
                },
            )
            feature_deltas.append(feature_delta)
        self._visual_plan.set(
            _VisualCompressionPlan(
                source_counts=source_feature_counts,
                result_counts=result_feature_counts,
                layout_deltas=tuple(feature_deltas),
            )
        )
        return merged

    def _execute_visual_block(
        self,
        block: Any,
        hidden_states: Any,
        normalised: Any,
        cu_seqlens: Any,
        position_embeddings: tuple[Any, Any],
        provider_kwargs: Mapping[str, Any],
        registrations: Sequence[_Registration],
        *,
        layer_index: int,
        score_layers: Mapping[int, int | None],
        score_keys: dict[int, Any],
    ) -> tuple[Any, tuple[Any, Any, Any] | None]:
        torch = _torch()
        positions = {
            id(registration): tome_encoder_positions(registration.executor.stage, "qwen25_vl", len(self.visual.blocks))
            for registration in registrations
        }
        initial = LayerPosition("inter_layer", 0)
        initial_registrations = [r for r in registrations if layer_index == 0 and initial in positions[id(r)]]
        source_position_embeddings = position_embeddings
        initial_restore = []
        actual_reference = None

        def context_for(registration: _Registration, stream: int, chunk: Any, carried_sizes: Any, steps: Any) -> dict[str, Any]:
            config = registration.executor.config
            values = {key: tuple(config[key]) for key in ("protected_indices", "coordinates") if key in config}
            sizes = config.get("token_sizes")
            if sizes is not None:
                if torch.is_tensor(sizes) and sizes.device != chunk.device:
                    raise ValueError("token_sizes and hidden_states must be on the same device")
                sizes = torch.as_tensor(sizes, dtype=chunk.dtype, device=chunk.device)
                source_count = steps[0][1][stream] if steps else int(chunk.shape[0])
                if tuple(sizes.shape) not in {(source_count,), (source_count, 1)}:
                    raise ValueError("token_sizes for one sequence must have shape [T] or [T, 1]")
                sizes = sizes.reshape(source_count, 1)
            for results, _counts in steps:
                delta = results[stream].dense_restore.compact_delta
                if sizes is not None:
                    mapping = torch.tensor([delta.old_to_new[i] for i in range(delta.source_count)], device=chunk.device)
                    sizes = sizes.new_zeros((delta.result_count, 1)).index_add(0, mapping, sizes)
                if values.get("protected_indices"):
                    values["protected_indices"] = tuple(sorted({delta.old_to_new[int(i)] for i in values["protected_indices"]}))
                if values.get("coordinates"):
                    values["coordinates"] = tuple(values["coordinates"][i] for i in _layout_representatives(delta))
            if sizes is not None or carried_sizes is not None:
                values["token_sizes"] = sizes if sizes is not None else carried_sizes
            return values

        source_lengths = tuple(int(v) for v in (cu_seqlens[1:] - cu_seqlens[:-1]).tolist())
        merge_sizes = {}

        def merge_chunk(registration: _Registration, stream: int, chunk: Any, keys: Any, position: LayerPosition, context: Mapping[str, Any], steps: Any) -> Any:
            source = score_layers[id(registration)]
            if source is not None:
                keys = torch.split(score_keys[source], source_lengths)[stream]
                for results, _counts in steps:
                    prior = results[stream]
                    keys = tome_merge_reference(keys, prior.dense_restore.compact_delta, token_sizes=merge_sizes[id(prior)])
            result = registration.executor.encoder_dense_boundary(
                chunk, keys, boundary_layer=layer_index,
                application_position=position.as_dict(), **dict(context),
            )
            merge_sizes[id(result)] = context.get("token_sizes", registration.executor.config.get("token_sizes"))
            return result

        token_sizes = None
        if initial_registrations:
            lengths = tuple(int(v) for v in (cu_seqlens[1:] - cu_seqlens[:-1]).tolist())
            _, keys = _vision_rotated_qk(block, normalised, position_embeddings)
            token_sizes = tuple(None for _ in lengths)
            source_sizes = tuple(hidden_states.new_ones((count, 1)) for count in lengths)
            for index, registration in enumerate(initial_registrations):
                configured_sizes = registration.executor.config.get("token_sizes")
                if configured_sizes is not None:
                    source_sizes = tuple(torch.as_tensor(configured_sizes, device=hidden_states.device, dtype=hidden_states.dtype).reshape(-1, 1) for _ in lengths)
                results = tuple(
                    merge_chunk(registration, stream, chunk, key_chunk, initial,
                        context_for(registration, stream, chunk, size, initial_restore), initial_restore,
                    )
                    for stream, (chunk, key_chunk, size) in enumerate(zip(torch.split(hidden_states, lengths), torch.split(keys, lengths), token_sizes))
                )
                initial_restore.append((results, lengths))
                for result in results:
                    self._record_delta(registration, result.dense_restore.compact_delta, boundary_layer=0)
                selected_positions = []
                for value in position_embeddings:
                    selected_positions.append(torch.cat([
                        chunk.index_select(0, torch.tensor(
                            _layout_representatives(result.dense_restore.compact_delta),
                            device=chunk.device, dtype=torch.long,
                        ))
                        for chunk, result in zip(torch.split(value, lengths), results)
                    ], dim=0))
                position_embeddings = tuple(selected_positions)
                hidden_states = torch.cat([result.kernel_result.hidden_states for result in results], dim=0)
                token_sizes = tuple(result.kernel_result.token_sizes for result in results)
                lengths = tuple(int(result.kernel_result.hidden_states.shape[0]) for result in results)
                normalised = block.norm1(hidden_states)
                if index + 1 < len(initial_registrations):
                    _, keys = _vision_rotated_qk(block, normalised, position_embeddings)
            cu_seqlens = _cumulative_lengths(lengths, device=hidden_states.device)
            attention_output, query, keys, _ = execute_vision_attention(
                block.attn, normalised, counts=lengths, position_embeddings=position_embeddings,
                token_sizes=token_sizes, capture_mean_received=False,
            )
            actual_reference = (query, keys)
            reference_counts = lengths
            for results, before_counts in reversed(initial_restore):
                actual_reference = tuple(torch.cat([
                    result.dense_restore.restore(chunk, sequence_dim=0)
                    for result, chunk in zip(results, torch.split(value, reference_counts))
                ], dim=0) for value in actual_reference)
                reference_counts = before_counts
            actual_reference = (*actual_reference, torch.cat(source_sizes, dim=0))
            post_attention = hidden_states + attention_output
        else:
            attention_output = block.attn(
                normalised,
                cu_seqlens=cu_seqlens,
                position_embeddings=position_embeddings,
                **dict(provider_kwargs),
            )
            post_attention = hidden_states + attention_output
            if not registrations:
                return post_attention + block.mlp(block.norm2(post_attention)), None
            _, keys = _vision_rotated_qk(block, normalised, position_embeddings)
        lengths = tuple(int(v) for v in (cu_seqlens[1:] - cu_seqlens[:-1]).tolist())
        hidden_states = post_attention
        for position in (LayerPosition("post_attention_pre_ffn", layer_index), LayerPosition("inter_layer", layer_index + 1)):
            source = position.index
            if position.boundary == "inter_layer" and source not in score_keys and any(
                position in positions[id(r)] and score_layers[id(r)] == source for r in registrations
            ):
                # The next native layer sees the structurally restored input.
                dense_reference = hidden_states
                counts = lengths
                for results, before_counts in reversed(initial_restore):
                    dense_reference = torch.cat([
                        result.dense_restore.restore(chunk, sequence_dim=0)
                        for result, chunk in zip(results, torch.split(dense_reference, counts))
                    ], dim=0)
                    counts = before_counts
                source_block = self.visual.blocks[source]
                _, reference = _vision_rotated_qk(source_block, source_block.norm1(dense_reference), source_position_embeddings)
                score_keys[source] = reference.detach()
            chunks = list(torch.split(hidden_states, lengths))
            key_chunks = list(torch.split(keys, lengths))
            sizes = token_sizes
            restore_steps: list[tuple[_Registration, tuple[Any, ...], tuple[int, ...]]] = []
            for registration in registrations:
                if position not in positions[id(registration)]:
                    continue
                before_counts = tuple(int(chunk.shape[0]) for chunk in chunks)
                results = tuple(
                    merge_chunk(registration, i, chunk, key_chunk, position,
                        context_for(registration, i, chunk, sizes[i], initial_restore + [
                            (prior_results, prior_counts) for _, prior_results, prior_counts in restore_steps
                        ]) if sizes is not None else {},
                        initial_restore + [(prior_results, prior_counts) for _, prior_results, prior_counts in restore_steps],
                    )
                    for i, (chunk, key_chunk) in enumerate(zip(chunks, key_chunks))
                )
                chunks = [result.kernel_result.hidden_states for result in results]
                if sizes is not None:
                    sizes = tuple(result.kernel_result.token_sizes for result in results)
                restore_steps.append((registration, results, before_counts))
                for result in results:
                    self._record_delta(
                        registration,
                        result.dense_restore.compact_delta,
                        boundary_layer=layer_index,
                    )
                # A second activation receives a metric aligned with the first
                # activation's compact layout.
                key_chunks = [
                    _vision_raw_keys(block, block.norm1(chunk)) for chunk in chunks
                ]
            compact = torch.cat(chunks, dim=0)
            if position.boundary == "post_attention_pre_ffn":
                compact = compact + block.mlp(block.norm2(compact))
            counts = tuple(int(chunk.shape[0]) for chunk in chunks)
            for _registration, results, before_counts in reversed(restore_steps):
                compact_chunks = torch.split(compact, counts)
                chunks = [
                    result.dense_restore.restore(chunk, sequence_dim=0)
                    for chunk, result in zip(compact_chunks, results)
                ]
                compact = torch.cat(chunks, dim=0)
                counts = before_counts
            if counts != lengths:
                raise InvariantError(
                    "Qwen2.5-VL ToMe restore did not recover dense attention segments."
                )
            hidden_states = compact
        for results, before_counts in reversed(initial_restore):
            chunks = torch.split(hidden_states, lengths)
            hidden_states = torch.cat([result.dense_restore.restore(chunk, sequence_dim=0) for result, chunk in zip(results, chunks)], dim=0)
            lengths = before_counts
        return hidden_states, actual_reference

    def _language_forward(self, _module: Any, *args: Any, **kwargs: Any) -> Any:
        assert self._original_language_forward is not None
        decoder_methods = self._registrations_for(_SITE_PYRAMID)
        removal = self._prompt_removal_mask.get()
        projector_state = self._projector_generate.get()
        if projector_state is not None:
            return self._projector_generate_language_forward(
                projector_state, removal, args, kwargs
            )
        if not decoder_methods and (removal is None or not bool(removal.any().item())):
            return self._original_language_forward(*args, **kwargs)
        call = _bind_language_call(self._original_language_forward, args, kwargs)
        from mllm_comp.fixed_mask.runtime import attach_layout
        attach_layout(self.model, call)
        resolved_use_cache = _effective_bool(
            call["use_cache"],
            getattr(self.language_model.config, "use_cache", False),
        )
        lifecycle = self._decoder_lifecycle.get()
        if lifecycle is None and (
            resolved_use_cache or call["past_key_values"] is not None
        ):
            raise BackendIntegrationError(
                "Pinned Qwen2.5-VL language bridge requires use_cache=False and "
                "no past_key_values."
            )
        if lifecycle is not None:
            expected_cache = (
                self._training_labels.get() is None
                and (
                    call["past_key_values"] is not None
                    or ExecutionPhase(decoder_methods[0].executor.execution_phase)
                    is ExecutionPhase.GENERATE
                )
            )
            if resolved_use_cache != expected_cache:
                raise BackendIntegrationError(
                    "Qwen2.5-VL decoder lifecycle received an inconsistent cache mode."
                )
        if removal is not None and bool(removal.any().item()):
            labels = self._training_labels.get()
            if labels is not None:
                keep = (~removal[0]).nonzero(as_tuple=False).flatten()
                self._training_labels.set(
                    compact_ignored_labels(
                        labels,
                        keep,
                        source_length=int(removal.shape[-1]),
                        operation="Qwen2.5-VL prompt compaction",
                    )
                )
        call, image_mask = _compact_prompt_call(
            call,
            removal,
            self._prompt_image_mask.get(),
            visual_plan=self._visual_plan.get(),
            centroid_mrope=_has_centroid_mrope_registration(
                self._registrations_for(_SITE_VISIONZIP)
            ),
        )
        if not decoder_methods:
            return self._original_language_forward(**call)
        if (image_mask is None or not bool(image_mask.any().item())) and call[
            "past_key_values"
        ] is None:
            return self._original_language_forward(**call)
        if lifecycle is None:
            raise BackendIntegrationError(
                "Qwen2.5-VL decoder compression requires a guarded request lifecycle."
            )
        return self._execute_decoder_loop(
            call,
            image_mask,
            decoder_methods,
            lifecycle,
            image_count=self._image_count.get(),
        )

    def _projector_generate_language_forward(
        self,
        state: dict[str, Any],
        removal: Any | None,
        args: Sequence[Any],
        kwargs: Mapping[str, Any],
    ) -> Any:
        call = _bind_language_call(self._original_language_forward, args, kwargs)
        from mllm_comp.fixed_mask.runtime import attach_layout
        attach_layout(self.model, call)
        inputs = call.get("inputs_embeds")
        position_ids = call.get("position_ids")
        length = (
            int(inputs.shape[1])
            if inputs is not None
            else int(call["input_ids"].shape[1])
        )
        past = call.get("past_key_values")
        is_decode = length == 1 and _bifocal_cache_length(past) > 0
        if not is_decode:
            if position_ids is None:
                raise InvariantError(
                    "Qwen2.5-VL projector-only prefill needs dense M-RoPE position_ids."
                )
            rotary = (
                position_ids[1:] if int(position_ids.shape[0]) == 4 else position_ids
            )
            state["expected_decode_position"] = int(rotary.max().item()) + 1
            compacted = removal is not None and bool(removal.any().item())
            visual_log_mass = self._projector_visual_log_mass(
                call, removal, self._prompt_image_mask.get()
            )
            call, _ = _compact_prompt_call(
                call,
                removal,
                self._prompt_image_mask.get(),
                visual_plan=self._visual_plan.get(),
                centroid_mrope=_has_centroid_mrope_registration(
                    self._registrations_for(_SITE_VISIONZIP)
                ),
            )
            if compacted and int(call["position_ids"].shape[0]) == 4:
                call["position_ids"] = call["position_ids"][1:]
            state["visual_log_mass"] = visual_log_mass
            return self._run_with_visual_log_mass(
                visual_log_mass,
                lambda: self._original_language_forward(**call),
            )
        state["decode_steps"] += 1
        if state["decode_position_verified"] is None and position_ids is not None:
            rotary = (
                position_ids[1:] if int(position_ids.shape[0]) == 4 else position_ids
            )
            observed = sorted(
                {int(value) for value in rotary[..., 0].flatten().tolist()}
            )
            expected = state["expected_decode_position"]
            state["observed_decode_position"] = observed
            if expected is None or observed != [expected]:
                state["decode_position_verified"] = False
                raise InvariantError(
                    f"Qwen2.5-VL projector-only first decode position {observed} != "
                    f"expected [{expected}] (prefill max M-RoPE position + 1). The "
                    "rope_deltas / cache_position convention after Stage-1 "
                    "compaction is broken."
                )
            state["decode_position_verified"] = True
        return self._run_with_visual_log_mass(
            state.get("visual_log_mass"),
            lambda: self._original_language_forward(*args, **kwargs),
        )

    def _projector_visual_log_mass(
        self, call: Mapping[str, Any], removal: Any | None, image_mask: Any | None
    ) -> Any | None:
        registrations = self._registrations_for(_SITE_VISIONZIP)
        if not any(
            bool(
                getattr(item.executor, "config", {}).get(
                    "attention_mass_correction", False
                )
            )
            for item in registrations
        ):
            return None
        plan = self._visual_plan.get()
        if plan is None or len(plan.layout_deltas) != 1:
            raise BackendIntegrationError(
                "Visual mass correction requires one image compression layout."
            )
        if removal is None or image_mask is None:
            raise BackendIntegrationError(
                "Visual mass correction requires the repaired image placeholder layout."
            )
        delta = plan.layout_deltas[0]
        sizes = [len(group) for group in _delta_groups(delta)]
        # image_mask marks the retained placeholders, already in the same
        # representative order as the compressed visual features. Removed
        # source placeholders are not keys in the compact attention sequence.
        positions = image_mask[0].reshape(-1).nonzero(as_tuple=False).flatten()
        if int(positions.numel()) != len(sizes):
            raise InvariantError(
                "Cluster mass count does not match dense visual placeholders."
            )
        torch = _torch()
        dense_mass = torch.zeros(
            int(call["inputs_embeds"].shape[1]),
            device=positions.device,
            dtype=torch.float32,
        )
        dense_mass[positions] = torch.tensor(
            sizes, device=positions.device, dtype=torch.float32
        ).log()
        keep = (~removal[0]).nonzero(as_tuple=False).flatten()
        return dense_mass.index_select(0, keep.to(dense_mass.device))

    def _run_with_visual_log_mass(
        self, visual_log_mass: Any | None, function: Callable[[], Any]
    ) -> Any:
        if visual_log_mass is None:
            return function()
        token = self._visual_log_mass.set(visual_log_mass)
        try:
            return function()
        finally:
            self._visual_log_mass.reset(token)

    def _layer0_attention_forward(
        self, _module: Any, *args: Any, **kwargs: Any
    ) -> Any:
        """Apply O-Z visual mass correction only during opted-in requests."""

        assert self._original_layer0_attention_forward is not None
        visual_log_mass = self._visual_log_mass.get()
        if visual_log_mass is None:
            return self._original_layer0_attention_forward(*args, **kwargs)
        signature = inspect.signature(self._original_layer0_attention_forward)
        bound = signature.bind_partial(*args, **kwargs)
        attention_mask = bound.arguments.get("attention_mask")
        if attention_mask is None:
            raise BackendIntegrationError(
                "Visual mass correction needs the eager additive attention mask."
            )
        key_length = int(attention_mask.shape[-1])
        bound.arguments["attention_mask"] = add_visual_log_mass_bias(
            attention_mask, visual_log_mass, key_length=key_length
        )
        return self._original_layer0_attention_forward(*bound.args, **bound.kwargs)

    def _execute_decoder_loop(
        self,
        call: dict[str, Any],
        image_mask: Any | None,
        registrations: Sequence[_Registration],
        lifecycle: ImageDecoderLifecycle,
        *,
        image_count: int | None = None,
    ) -> Any:
        torch = _torch()
        (
            BaseModelOutputWithPast,
            create_causal_mask,
            create_sliding_window_causal_mask,
        ) = _transformers_decoder_helpers()
        if len(registrations) != 1:
            raise BackendIntegrationError(
                "Qwen2.5-VL decoder loop requires exactly one image method."
            )
        registration = registrations[0]
        decoder_algorithm = str(registration.executor.contract.algorithm)
        selection_method = getattr(registration.executor, "decoder_selection", None)
        use_contiguous_positions = selection_method is not None and selection_method.position_policy == "contiguous"
        layer_local_contiguous = use_contiguous_positions and selection_method.cache_policy is CacheRepairPolicy.LAYER_LOCAL
        compact_contiguous = use_contiguous_positions and selection_method.cache_policy is CacheRepairPolicy.COMPACT_PREFIX
        if selection_method is None and decoder_algorithm not in {
            "pyramid_drop",
            "fastv",
            "sgl",
            "nuwa",
            "restore",
        }:
            raise BackendIntegrationError(
                f"Unsupported Qwen2.5-VL decoder algorithm {decoder_algorithm!r}."
            )
        input_ids = call.pop("input_ids")
        inputs_embeds = call.pop("inputs_embeds")
        padding_mask = call.pop("attention_mask")
        position_ids = call.pop("position_ids")
        past_key_values = call.pop("past_key_values")
        use_cache = call.pop("use_cache")
        output_attentions = _effective_bool(
            call.pop("output_attentions"),
            getattr(self.language_model.config, "output_attentions", False),
        )
        output_hidden_states = _effective_bool(
            call.pop("output_hidden_states"),
            getattr(self.language_model.config, "output_hidden_states", False),
        )
        return_dict = _effective_bool(
            call.pop("return_dict"),
            getattr(self.language_model.config, "use_return_dict", True),
        )
        cache_position = call.pop("cache_position")
        use_cache = _effective_bool(
            use_cache,
            getattr(self.language_model.config, "use_cache", False),
        )
        if output_attentions or output_hidden_states or not return_dict:
            raise BackendIntegrationError(
                "Decoder image compression supports return_dict=True without attention or "
                "hidden-state histories."
            )
        if isinstance(padding_mask, Mapping):
            raise BackendIntegrationError(
                "Decoder image compression cannot rebuild a caller-supplied attention-mask "
                "mapping; pass the rank-2 padding mask instead."
            )
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError(
                "You must specify exactly one of input_ids or inputs_embeds"
            )
        if inputs_embeds is None:
            inputs_embeds = self.language_model.embed_tokens(input_ids)
        if int(inputs_embeds.shape[0]) != 1:
            raise BackendIntegrationError(
                "Pinned Qwen2.5-VL decoder compression supports batch size one."
            )
        if use_cache and past_key_values is None:
            past_key_values = _new_dynamic_cache(self.language_model.config)
        if cache_position is None:
            cache_position = torch.arange(
                inputs_embeds.shape[1], device=inputs_embeds.device
            )
        if position_ids is None:
            position_ids = cache_position.view(1, 1, -1).expand(
                3, inputs_embeds.shape[0], -1
            )
        elif position_ids.ndim == 2:
            position_ids = position_ids[None, ...].expand(3, position_ids.shape[0], -1)
        if position_ids.ndim != 3 or int(position_ids.shape[0]) not in (3, 4):
            raise BackendIntegrationError(
                "Qwen2.5-VL decoder compression requires 3-plane M-RoPE positions or "
                "the provider's packed 4-plane representation."
            )
        if int(position_ids.shape[-1]) != int(inputs_embeds.shape[1]):
            raise InvariantError("M-RoPE positions do not match decoder sequence.")

        decoder_call = lifecycle.begin_call(
            query_length=int(inputs_embeds.shape[1]),
            use_cache=use_cache,
            past_key_values=past_key_values,
        )
        layer_local_native_rotary_positions = None
        if (
            decoder_call.phase is DecoderCallPhase.DECODE
            and layer_local_contiguous
        ):
            native_rope_delta = self._layer_local_native_rope_delta.get()
            if native_rope_delta:
                native_positions = cache_position.to(position_ids.dtype) + int(
                    native_rope_delta[0]
                )
                layer_local_native_rotary_positions = native_positions.view(
                    1, 1, -1
                ).expand(3, int(inputs_embeds.shape[0]), -1)
            else:
                layer_local_native_rotary_positions = (
                    position_ids[1:]
                    if int(position_ids.shape[0]) == 4
                    else position_ids
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
                    f"Qwen2.5-VL {decoder_algorithm} is single-image only; "
                    f"received {image_count!r} images."
                )
            if decoder_call.phase is DecoderCallPhase.DECODE:
                if compact_contiguous:
                    compact_position = dynamic_cache_layer_length(past_key_values, 0)
                    position_ids = torch.full_like(
                        position_ids,
                        compact_position,
                    )
                if not use_contiguous_positions:
                    self._record_decode_position(
                        cache_position=cache_position,
                        position_ids=position_ids,
                    )
            if decoder_call.phase is DecoderCallPhase.DECODE and image_mask is not None:
                raise BackendIntegrationError(
                    "Qwen2.5-VL decode must not replay image placeholder state."
                )
            if decoder_call.phase is not DecoderCallPhase.DECODE and (
                image_mask is None or not bool(image_mask.any().item())
            ):
                raise BackendIntegrationError(
                    "Qwen2.5-VL compressed prefill requires exact image positions."
                )
            hidden_states = inputs_embeds
            causal_masks, text_position_ids, rotary_position_ids = _decoder_state(
                self.language_model,
                hidden_states,
                padding_mask,
                position_ids,
                cache_position,
                create_causal_mask,
                create_sliding_window_causal_mask,
            )
            position_embeddings = self.language_model.rotary_emb(
                hidden_states, rotary_position_ids
            )
            restore_attention = None
            selection = selection_method.new_call() if selection_method is not None else None
            if selection is not None and decoder_call.phase is not DecoderCallPhase.DECODE:
                if callable(getattr(selection, "set_visual_layout", None)):
                    from mllm_comp.compressors.comparison.native_decoder import qwen_visual_segments
                    selection.set_visual_layout(qwen_visual_segments(self._comparison_grid.get(), self.visual.spatial_merge_size))
                selection.initialize(inputs_embeds[0], image_mask[0].bool(),
                    padding_mask=padding_mask, labels=self._training_labels.get())
            boundary_replay = None
            restore_spec = getattr(
                registration.executor,
                "_restore_spec_value",
                None,
            )
            if decoder_algorithm == "restore" and restore_spec is None:
                raise InvariantError("Qwen2.5-VL RESTORE spec was not installed.")
            if decoder_algorithm == "restore":
                runtime = self._restore_runtime.get()
                if runtime is None:
                    raise InvariantError(
                        "Qwen2.5-VL RESTORE decoder ran outside its request lifecycle."
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
                    assert image_mask is not None
                    restore_attention = RestoreAttentionState.prefill(
                        rotary_position_ids,
                        image_indices=image_mask[0]
                        .bool()
                        .nonzero(as_tuple=False)
                        .flatten(),
                        visual_sizes=runtime["visual_sizes"],
                        source_visual_count=runtime["source_visual_count"],
                        first_layer_size=restore_spec.first_layer_size,
                        reference=hidden_states,
                    )
                runtime["attention_state"] = restore_attention
            restore_layers_calibrated = 0
            application_positions = (
                selection_method.positions if selection_method is not None else ()
            )
            def apply_pre_layer_compression(layer_index, *, mid=False):
                nonlocal boundary_replay
                nonlocal hidden_states, image_mask, padding_mask, position_ids, cache_position
                nonlocal causal_masks, text_position_ids, rotary_position_ids, position_embeddings
                point = LayerPosition("post_attention_pre_ffn" if mid else "inter_layer", layer_index)
                if decoder_call.phase is DecoderCallPhase.DECODE or point not in application_positions:
                    return None
                assert image_mask is not None
                current_image_indices = (
                    image_mask[0].bool().nonzero(as_tuple=False).flatten()
                )
                if int(current_image_indices.numel()) == 0:
                    return None
                result = registration.executor.apply_decoder_selection(
                    selection,
                    point,
                    hidden_states[0],
                    image_indices=current_image_indices,
                )
                kept = torch.tensor(
                    layout_result_representatives(result.decoder_repair.layout_delta),
                    dtype=torch.long,
                    device=hidden_states.device,
                )
                boundary_replay = decoder_layout_replay(result.decoder_repair.layout_delta, rules=getattr(registration.executor, "_rules", {}))
                delta = result.decoder_repair.layout_delta
                lifecycle.record_boundary(
                    layer_index=min(layer_index, len(self.language_model.layers) - 1),
                    boundary=DecoderBoundary.POST_ATTENTION_PRE_FFN if mid else (DecoderBoundary.AFTER_LAYER if layer_index == len(self.language_model.layers) else DecoderBoundary.BEFORE_LAYER),
                    layout_delta=delta,
                    past_key_values=past_key_values,
                    cache_policy=selection_method.cache_policy if selection_method is not None else CacheRepairPolicy.COMPACT_PREFIX,
                    prefix_layer_count=layer_index + int(mid) if use_cache else 0,
                )
                hidden_states = result.kernel_result.hidden_states.unsqueeze(0)
                image_mask = image_mask.index_select(-1, kept.to(image_mask.device))
                from mllm_comp.fixed_mask.runtime import compact_layout
                compact_layout(call, kept)
                if padding_mask is not None:
                    padding_mask = padding_mask.index_select(
                        -1, kept.to(padding_mask.device)
                    )
                position_ids = position_ids.index_select(
                    -1, kept.to(position_ids.device)
                )
                cache_position = torch.arange(
                    hidden_states.shape[1], device=hidden_states.device
                )
                if use_contiguous_positions:
                    position_ids = cache_position.to(position_ids.dtype).view(1, 1, -1).expand(int(position_ids.shape[0]), int(hidden_states.shape[0]), -1)
                causal_masks, text_position_ids, rotary_position_ids = (
                    _decoder_state(
                        self.language_model,
                        hidden_states,
                        padding_mask,
                        position_ids,
                        cache_position,
                        create_causal_mask,
                        create_sliding_window_causal_mask,
                    )
                )
                position_embeddings = self.language_model.rotary_emb(
                    hidden_states, rotary_position_ids
                )
                self._record_delta(
                    registration,
                    delta,
                    boundary_layer=layer_index,
                )
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
                visibility = pyramid_attention_key_visibility(causal_masks[metric_layer.attention_type],
                    instruction_indices=capture.query_indices, key_count=int(attention_keys.shape[0]), padding_mask=padding_mask)
                return query, image_keys, attention_keys, visibility

            from mllm_comp.compressors.comparison.native_decoder import observe_comparison_input, layer_context
            for layer_index, decoder_layer in enumerate(self.language_model.layers):
                if selection is not None and decoder_call.phase is not DecoderCallPhase.DECODE:
                    observe_comparison_input(selection, layer_index, hidden_states, self.language_model.layers,
                        attention_mask=causal_masks[decoder_layer.attention_type], position_ids=text_position_ids,
                        position_embeddings=position_embeddings, cache_position=cache_position)
                # Earlier stages at this entry run before a later stage samples this layer's input.
                apply_before_capture = selection.select_before_input(layer_index) if selection is not None else False
                if apply_before_capture:
                    apply_pre_layer_compression(layer_index)
                if selection is not None and decoder_call.phase is not DecoderCallPhase.DECODE:
                    image_row = image_mask[0].bool()
                    capture = selection.capture_input(layer_index, image_row,
                        padding_mask=padding_mask, labels=self._training_labels.get())
                    if capture is not None:
                        capture_decoder_attention(selection, capture, image_row.nonzero(as_tuple=False).flatten(), read_selection_attention)
                if not apply_before_capture:
                    apply_pre_layer_compression(layer_index)
                layer_cache_position = cache_position
                layer_position_ids = position_ids
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
                        physical_text_positions = layer_cache_position.to(
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
                        layer_position_ids = torch.cat(
                            (
                                physical_text_positions.unsqueeze(0),
                                layer_rotary_position_ids,
                            ),
                            dim=0,
                        )
                        lifecycle.record_layer_local_decode_position(
                            layer_index=layer_index,
                            cache_position=layer_cache_position,
                            text_position_ids=physical_text_positions,
                            rotary_position_ids=layer_rotary_position_ids,
                            expected_rotary_position_ids=layer_rotary_position_ids,
                            rotary_policy=rotary_policy,
                        )
                    local_padding = _layer_local_decode_padding_mask(
                        padding_mask,
                        past_key_values,
                        layer_index=layer_index,
                        query_length=int(hidden_states.shape[1]),
                        reference=hidden_states,
                    )
                    causal_masks, text_position_ids, rotary_position_ids = (
                        _decoder_state(
                            self.language_model,
                            hidden_states,
                            local_padding,
                            layer_position_ids,
                            layer_cache_position,
                            create_causal_mask,
                            create_sliding_window_causal_mask,
                            past_key_values=LayerLocalCacheView(
                                past_key_values, layer_index
                            ),
                        )
                    )
                    position_embeddings = self.language_model.rotary_emb(
                        hidden_states, rotary_position_ids
                    )
                layer_input_hidden = hidden_states
                layer_input_position_embeddings = position_embeddings
                capture_context = nullcontext(None)
                if (selection is not None and decoder_call.phase is not DecoderCallPhase.DECODE
                        and layer_index in selection.transition_layers):
                    image_indices = image_mask[0].bool().nonzero(as_tuple=False).flatten()
                    if selection.begin_transition(layer_index, image_indices):
                        capture_context = capture_decoder_module_transitions(decoder_layer)
                with capture_context as transition_observation:
                    layer_attention_mask = causal_masks[
                        decoder_layer.attention_type
                    ]
                    if decoder_algorithm == "restore":
                        assert restore_spec is not None
                        assert restore_attention is not None
                        layer_attention_mask = restore_attention.attention_mask(
                            layer_attention_mask,
                            rotary_position_ids,
                            self.language_model.rotary_emb.inv_freq,
                            layer_index=layer_index,
                            reference=hidden_states,
                            calibration_constant=restore_spec.calibration_constant,
                            mrope_section=_restore_mrope_section(
                                self.language_model.config,
                                axis_count=int(rotary_position_ids.shape[0]),
                            ),
                            rules=getattr(registration.executor, "_rules", {}),
                        )
                        restore_layers_calibrated += 1
                    if selection is not None:
                        layer_attention_mask = selection.prepare_attention_mask(layer_attention_mask,
                            query_length=int(hidden_states.shape[1]), reference=hidden_states)
                    layer_forward = decoder_layer
                    if (decoder_call.phase is not DecoderCallPhase.DECODE
                            and LayerPosition("post_attention_pre_ffn", layer_index) in application_positions):
                        layer_forward = partial(decoder_post_attention_boundary, decoder_layer, tuple_output=True,
                            apply_compression=partial(apply_post_attention, layer_index=layer_index))
                    layer_outputs = layer_forward(
                        hidden_states,
                        attention_mask=layer_attention_mask,
                        position_ids=text_position_ids,
                        past_key_values=past_key_values,
                        output_attentions=False,
                        use_cache=use_cache,
                        cache_position=layer_cache_position,
                        position_embeddings=position_embeddings,
                        **call,
                        **layer_context(selection),
                    )
                hidden_states = layer_outputs[0]
                if decoder_call.phase is DecoderCallPhase.DECODE:
                    continue
                assert image_mask is not None
                if transition_observation is not None:
                    image_indices = image_mask[0].bool().nonzero(as_tuple=False).flatten()
                    selection.record_transition(transition_observation, image_indices=image_indices)
                    transition_observation = None
                if selection is not None:
                    if selection.output_hidden_layer == layer_index:
                        selection.record_output_hidden(layer_index, self.language_model.norm(hidden_states)[0])
                    image_row = image_mask[0].bool()
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
            apply_pre_layer_compression(len(self.language_model.layers))
            hidden_states = self.language_model.norm(hidden_states)
            output = BaseModelOutputWithPast(
                last_hidden_state=hidden_states,
                past_key_values=past_key_values,
                hidden_states=None,
                attentions=None,
            )
            lifecycle.finish_call(past_key_values=past_key_values)
            if decoder_algorithm == "restore":
                assert restore_attention is not None
                self._latest_restore_calibration = {
                    "phase": decoder_call.phase.value,
                    "layers_calibrated": restore_layers_calibrated,
                    "query_tokens": int(hidden_states.shape[1]),
                    "key_tokens": int(restore_attention.key_sizes.shape[-1]),
                    "visual_source_tokens": int(runtime.get("source_visual_count", 0)),
                    "first_layer_size": restore_spec.first_layer_size,
                    "position_policy": "native_sparse_mrope_representatives",
                    "cache_transform": "none",
                }
            return output
        except BaseException:
            lifecycle.abort_call()
            raise


def install_qwen25_vl_transformers_4573_bridge(
    model: Any,
) -> Qwen25VLTransformers4573Bridge:
    """Install or return the existing pinned bridge for ``model``."""

    existing = getattr(model, _BRIDGE_ATTRIBUTE, None)
    if isinstance(existing, Qwen25VLTransformers4573Bridge):
        return existing
    return Qwen25VLTransformers4573Bridge(model).install()


def uninstall_qwen25_vl_transformers_4573_bridge(
    model: Any,
    *,
    force: bool = False,
) -> None:
    """Uninstall the pinned bridge when present."""

    bridge = getattr(model, _BRIDGE_ATTRIBUTE, None)
    if bridge is None:
        return
    if not isinstance(bridge, Qwen25VLTransformers4573Bridge):
        raise BackendIntegrationError(
            "Qwen2.5-VL provider bridge attribute has an unknown owner."
        )
    bridge.uninstall(force=force)


def _installed_transformers_version() -> str:
    try:
        return importlib.metadata.version("transformers")
    except importlib.metadata.PackageNotFoundError as exc:
        raise BackendIntegrationError(
            "Qwen2.5-VL provider bridge requires transformers==4.57.3."
        ) from exc


def _validate_provider_version() -> None:
    actual = _installed_transformers_version()
    if actual != SUPPORTED_TRANSFORMERS_VERSION:
        raise BackendIntegrationError(
            "Qwen2.5-VL provider bridge is source-pinned to "
            f"transformers=={SUPPORTED_TRANSFORMERS_VERSION}; got {actual!r}."
        )


def _validate_qwen25_vl_structure(
    model: Any,
    *,
    core: Any,
    visual: Any,
    language_model: Any,
) -> None:
    expected_classes = (
        (model, "Qwen2_5_VLForConditionalGeneration", "model"),
        (core, "Qwen2_5_VLModel", "model.model"),
        (
            visual,
            "Qwen2_5_VisionTransformerPretrainedModel",
            "model.model.visual",
        ),
        (language_model, "Qwen2_5_VLTextModel", "model.model.language_model"),
    )
    for value, expected, path in expected_classes:
        actual = type(value).__name__ if value is not None else None
        if actual != expected:
            raise BackendIntegrationError(
                f"Pinned Qwen2.5-VL bridge requires {path} class {expected}, "
                f"got {actual!r}."
            )
    _require_attributes(
        model,
        ("forward", "generate"),
        "model",
    )
    _require_attributes(
        core,
        (
            "forward",
            "visual",
            "language_model",
            "get_image_features",
            "get_video_features",
            "get_placeholder_mask",
            "get_rope_index",
            "config",
            "rope_deltas",
        ),
        "model.model",
    )
    _require_attributes(
        visual,
        (
            "forward",
            "blocks",
            "merger",
            "patch_embed",
            "rot_pos_emb",
            "get_window_index",
            "fullatt_block_indexes",
            "spatial_merge_size",
            "spatial_merge_unit",
            "dtype",
        ),
        "model.model.visual",
    )
    _require_attributes(
        language_model,
        (
            "forward",
            "layers",
            "norm",
            "rotary_emb",
            "embed_tokens",
            "has_sliding_layers",
            "config",
        ),
        "model.model.language_model",
    )
    if len(visual.blocks) < 1 or len(language_model.layers) < 2:
        raise BackendIntegrationError(
            "Pinned Qwen2.5-VL compression requires at least one visual block "
            "and two decoder layers."
        )
    for index, block in enumerate(visual.blocks):
        _require_attributes(
            block,
            ("norm1", "norm2", "attn", "mlp"),
            f"model.model.visual.blocks[{index}]",
        )
        _require_attributes(
            block.attn,
            ("qkv", "num_heads", "scaling"),
            f"model.model.visual.blocks[{index}].attn",
        )
    for index, layer in enumerate(language_model.layers):
        _require_attributes(
            layer,
            ("input_layernorm", "self_attn", "attention_type"),
            f"model.model.language_model.layers[{index}]",
        )
        _require_attributes(
            layer.self_attn,
            (
                "q_proj",
                "k_proj",
                "head_dim",
                "num_key_value_groups",
            ),
            f"model.model.language_model.layers[{index}].self_attn",
        )
    _require_signature(
        model.forward,
        {"input_ids", "labels", "use_cache", "pixel_values", "pixel_values_videos"},
        "Qwen2_5_VLForConditionalGeneration.forward",
    )
    _require_signature(
        core.forward,
        {
            "input_ids",
            "attention_mask",
            "position_ids",
            "past_key_values",
            "inputs_embeds",
            "use_cache",
            "pixel_values",
            "pixel_values_videos",
            "image_grid_thw",
            "video_grid_thw",
            "second_per_grid_ts",
        },
        "Qwen2_5_VLModel.forward",
    )
    _require_signature(
        visual.forward,
        {"hidden_states", "grid_thw"},
        "Qwen2_5_VisionTransformerPretrainedModel.forward",
    )
    _require_signature(
        language_model.forward,
        {
            "input_ids",
            "attention_mask",
            "position_ids",
            "past_key_values",
            "inputs_embeds",
            "use_cache",
            "output_attentions",
            "output_hidden_states",
            "return_dict",
            "cache_position",
        },
        "Qwen2_5_VLTextModel.forward",
    )


def _validate_executor_registration(
    executor: Any,
    *,
    visual: Any,
    language_model: Any,
) -> tuple[str, tuple[int, ...]]:
    contract = getattr(executor, "contract", None)
    site = getattr(contract, "callback", None)
    callback = "apply_decoder_selection" if site == _SITE_PYRAMID else site
    if site not in _KNOWN_SITES or not callable(getattr(executor, callback, None)):
        raise BackendIntegrationError(
            "Qwen2.5-VL provider received an executor without a known semantic "
            "callback."
        )
    phase = ExecutionPhase(getattr(executor, "execution_phase", ExecutionPhase.FORWARD))
    if phase is ExecutionPhase.GENERATE and site == _SITE_PYRAMID:
        pass
    elif (
        phase is ExecutionPhase.GENERATE
        and site == _SITE_VISIONZIP
        and getattr(contract, "algorithm", None)
        in _PROJECTOR_GENERATE_ALGORITHMS
    ):
        pass
    elif phase is not ExecutionPhase.FORWARD:
        raise BackendIntegrationError(
            "Pinned Qwen2.5-VL provider supports GENERATE only for approved "
            "projector-only methods or decoder compression methods with the "
            "shared DynamicCache lifecycle."
        )
    if getattr(contract, "image_only", False) is not True or getattr(
        contract, "supports_video", True
    ):
        raise BackendIntegrationError(
            "Pinned Qwen2.5-VL provider registrations must target image only."
        )
    config = getattr(executor, "config", {})
    if not isinstance(config, Mapping):
        raise BackendIntegrationError("Compression executor config must be a mapping.")
    if site == _SITE_TOME:
        r = config.get("r")
        if isinstance(r, bool) or not isinstance(r, int) or r < 0:
            raise BackendIntegrationError(
                "Qwen2.5-VL ToMe registration requires non-negative integer r."
            )
        return site, ()
    if site == _SITE_SAINT:
        _require_attributes(
            visual,
            ("window_size", "patch_size"),
            "model.model.visual (SAINT)",
        )
        layers = tuple(
            getattr(executor, "_saint_layer_indices_value", ())
        )
        ratios = tuple(getattr(executor, "_saint_keep_ratios_value", ()))
        if (
            not layers
            or len(layers) != len(ratios)
            or layers != tuple(sorted(set(layers)))
            or any(index < 0 or index >= len(visual.blocks) for index in layers)
        ):
            raise BackendIntegrationError(
                "Qwen2.5-VL SAINT requires a valid sorted encoder schedule "
                "matching keep_ratios."
            )
        from mllm_comp.compressors.image.positions import saint_encoder_positions
        try:
            executor.application_positions = saint_encoder_positions(getattr(executor, "stage", None), layers, len(visual.blocks))
        except (TypeError, ValueError) as exc:
            raise BackendIntegrationError(str(exc)) from exc
        return site, layers
    if site == _SITE_VISIONZIP:
        if getattr(contract, "requires_visual_metrics", False) and len(
            visual.blocks
        ) - 1 not in set(visual.fullatt_block_indexes):
            raise BackendIntegrationError(
                "Qwen2.5-VL visual-cue compression requires the final visual "
                "block to be full attention so its global metric is well-defined."
            )
        return site, ()
    if getattr(contract, "algorithm", None) == "sgl":
        try:
            prune_layer = executor.configure_sgl_decoder_depth(
                len(language_model.layers)
            )
        except (TypeError, ValueError) as exc:
            raise BackendIntegrationError(str(exc)) from exc
        return site, (prune_layer,)
    if getattr(executor, "decoder_selection", None) is not None:
        method = executor.decoder_selection
        resolve_decoder_selection_depth(method, len(language_model.layers))
        executor.application_positions = method.positions
        executor.application_position = method.positions[0]
        executor.target_site = method.target_site
        return site, tuple(point.index for point in method.positions)
    raise BackendIntegrationError("No decoder execution mechanism was registered.")


def _validate_nuwa_composite_registration(
    executor: Any,
    *,
    visual: Any,
    language_model: Any,
) -> tuple[int, ...]:
    """Validate Nüwa's one activation spanning visual and decoder boundaries."""

    contract = getattr(executor, "contract", None)
    if getattr(contract, "algorithm", None) != "nuwa" or not all(
        callable(getattr(executor, callback, None))
        for callback in ("post_visual_merger", "apply_decoder_selection")
    ):
        raise BackendIntegrationError(
            "Qwen2.5-VL Nüwa requires visual and decoder callbacks on one executor."
        )
    phase = ExecutionPhase(
        getattr(executor, "execution_phase", ExecutionPhase.FORWARD)
    )
    if phase not in {ExecutionPhase.FORWARD, ExecutionPhase.GENERATE}:
        raise BackendIntegrationError(
            "Qwen2.5-VL Nüwa supports FORWARD and GENERATE image phases only."
        )
    if getattr(contract, "image_only", False) is not True or getattr(
        contract, "supports_video", True
    ):
        raise BackendIntegrationError(
            "Qwen2.5-VL Nüwa must be registered as an image-only method."
        )
    if len(visual.blocks) - 1 not in set(visual.fullatt_block_indexes):
        raise BackendIntegrationError(
            "Qwen2.5-VL Nüwa requires the final visual block to expose global "
            "attention metrics."
        )
    spec = getattr(executor, "_nuwa_spec_value", None)
    if spec is None:
        raise BackendIntegrationError("Qwen2.5-VL Nüwa spec was not installed.")
    resolve_decoder_selection_depth(
        executor.decoder_selection, len(language_model.layers)
    )
    executor.application_position = executor.decoder_selection.positions[0]
    return (executor.application_position.index,)


def _validate_restore_composite_registration(
    executor: Any,
    *,
    visual: Any,
) -> tuple[int, ...]:
    """Validate RESTORE's visual merge and every-layer decoder calibration."""

    contract = getattr(executor, "contract", None)
    if getattr(contract, "algorithm", None) != "restore" or not callable(
        getattr(executor, "post_visual_merger", None)
    ):
        raise BackendIntegrationError(
            "Qwen2.5-VL RESTORE requires one composite visual/decoder executor."
        )
    phase = ExecutionPhase(
        getattr(executor, "execution_phase", ExecutionPhase.FORWARD)
    )
    if phase not in {ExecutionPhase.FORWARD, ExecutionPhase.GENERATE}:
        raise BackendIntegrationError(
            "Qwen2.5-VL RESTORE supports FORWARD and GENERATE image phases only."
        )
    if getattr(contract, "image_only", False) is not True or getattr(
        contract, "supports_video", True
    ):
        raise BackendIntegrationError(
            "Qwen2.5-VL RESTORE must be registered as an image-only method."
        )
    if len(visual.blocks) - 1 not in set(visual.fullatt_block_indexes):
        raise BackendIntegrationError(
            "Qwen2.5-VL RESTORE requires final global visual attention metrics."
        )
    if getattr(executor, "_restore_spec_value", None) is None:
        raise BackendIntegrationError("Qwen2.5-VL RESTORE spec was not installed.")
    return ()


def _is_projector_only_generate(registrations: Sequence[_Registration]) -> bool:
    if len(registrations) != 1:
        return False
    registration = registrations[0]
    return (
        registration.site == _SITE_VISIONZIP
        and getattr(registration.executor.contract, "algorithm", None)
        in _PROJECTOR_GENERATE_ALGORITHMS
        and ExecutionPhase(registration.executor.execution_phase)
        is ExecutionPhase.GENERATE
    )


def _registrations_support_generate(
    registrations: Sequence[_Registration],
) -> bool:
    """Return whether the registrations form one supported generate method."""

    if len(registrations) == 1:
        registration = registrations[0]
        if _is_projector_only_generate(registrations):
            return True
        return (
            registration.site == _SITE_PYRAMID
            and ExecutionPhase(registration.executor.execution_phase)
            is ExecutionPhase.GENERATE
        )
    if len(registrations) != 2:
        return False
    executors = {id(registration.executor) for registration in registrations}
    sites = {registration.site for registration in registrations}
    algorithms = {
        getattr(registration.executor.contract, "algorithm", None)
        for registration in registrations
    }
    return (
        len(executors) == 1
        and sites == {_SITE_VISIONZIP, _SITE_PYRAMID}
        and len(algorithms) == 1
        and algorithms <= {"nuwa", "restore"}
        and all(
            ExecutionPhase(registration.executor.execution_phase)
            is ExecutionPhase.GENERATE
            for registration in registrations
        )
    )


def _require_attributes(value: Any, names: Sequence[str], path: str) -> None:
    missing = [name for name in names if not hasattr(value, name)]
    if missing:
        raise BackendIntegrationError(
            f"Pinned Qwen2.5-VL structure mismatch at {path}: missing {missing}."
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


def _bind_forward_arguments(
    function: Callable[..., Any],
    args: Sequence[Any],
    kwargs: Mapping[str, Any],
) -> dict[str, Any]:
    try:
        bound = inspect.signature(function).bind_partial(*args, **kwargs)
    except (TypeError, ValueError):
        return dict(kwargs)
    values = dict(bound.arguments)
    extras = values.pop("kwargs", None)
    if isinstance(extras, Mapping):
        values.update(extras)
    return values


def _bind_visual_call(
    function: Callable[..., Any],
    args: Sequence[Any],
    kwargs: Mapping[str, Any],
) -> tuple[Any, Any, dict[str, Any]]:
    values = _bind_forward_arguments(function, args, kwargs)
    hidden = values.pop("hidden_states", args[0] if args else None)
    grid = values.pop("grid_thw", None)
    values.pop("self", None)
    if hidden is None or grid is None:
        raise BackendIntegrationError(
            "Qwen2.5-VL visual bridge requires hidden_states and grid_thw."
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
        "output_attentions": None,
        "output_hidden_states": None,
        "return_dict": None,
        "cache_position": None,
    }
    return {**defaults, **values}


def _compact_prompt_call(
    call: dict[str, Any],
    removal_mask: Any | None,
    image_mask: Any | None,
    *,
    visual_plan: _VisualCompressionPlan | None = None,
    centroid_mrope: bool = False,
) -> tuple[dict[str, Any], Any | None]:
    if removal_mask is None or not bool(removal_mask.any().item()):
        return call, image_mask
    torch = _torch()
    inputs = call["inputs_embeds"]
    if inputs is None or int(inputs.shape[0]) != 1:
        raise BackendIntegrationError(
            "Compressed-image prompt repair requires one inputs_embeds batch item."
        )
    keep = (~removal_mask[0]).nonzero(as_tuple=False).flatten()
    dense_positions = call.get("position_ids")
    dense_image_mask = (
        (image_mask | removal_mask)
        if image_mask is not None
        else None
    )
    call["inputs_embeds"] = inputs.index_select(1, keep.to(inputs.device))
    call["input_ids"] = None
    for name in ("attention_mask", "position_ids"):
        value = call[name]
        if value is not None:
            call[name] = value.index_select(-1, keep.to(value.device))
    if centroid_mrope:
        call["position_ids"] = _apply_visual_centroid_mrope(
            dense_positions=dense_positions,
            compact_positions=call.get("position_ids"),
            dense_image_mask=dense_image_mask,
            keep=keep,
            visual_plan=visual_plan,
        )
    call["cache_position"] = torch.arange(keep.numel(), device=inputs.device)
    if image_mask is not None:
        image_mask = image_mask.index_select(-1, keep.to(image_mask.device))
    from mllm_comp.fixed_mask.runtime import compact_layout
    compact_layout(call, keep)
    return call, image_mask


def _apply_visual_centroid_mrope(
    *,
    dense_positions: Any | None,
    compact_positions: Any | None,
    dense_image_mask: Any | None,
    keep: Any,
    visual_plan: _VisualCompressionPlan | None,
) -> Any:
    """Replace retained visual positions by rounded native cluster centroids."""

    if dense_positions is None or compact_positions is None:
        raise InvariantError("Centroid M-RoPE repair requires position_ids.")
    if dense_image_mask is None or visual_plan is None:
        raise InvariantError("Centroid M-RoPE repair requires a visual layout plan.")
    if int(dense_image_mask.shape[0]) != 1:
        raise BackendIntegrationError(
            "Centroid M-RoPE repair requires batch size one."
        )
    if getattr(dense_positions, "ndim", None) != 3 or int(dense_positions.shape[0]) < 3:
        raise InvariantError(
            "Centroid M-RoPE repair expected [axes, batch, sequence] position_ids."
        )
    torch = _torch()
    dense_flat = dense_image_mask[0].reshape(-1).nonzero(as_tuple=False).flatten()
    if int(dense_flat.numel()) != sum(visual_plan.source_counts):
        raise InvariantError(
            "Centroid M-RoPE repair image placeholders disagree with visual plan."
        )
    old_to_compact = torch.full(
        (int(dense_image_mask.shape[-1]),),
        -1,
        dtype=torch.long,
        device=keep.device,
    )
    old_to_compact[keep] = torch.arange(keep.numel(), device=keep.device)
    repaired = compact_positions.clone()
    source_offset = 0
    for source_count, delta in zip(
        visual_plan.source_counts, visual_plan.layout_deltas
    ):
        representatives = _layout_representatives(delta)
        groups = _delta_groups(delta)
        if len(representatives) != len(groups):
            raise InvariantError("Centroid M-RoPE group/representative mismatch.")
        for result_index, members in enumerate(groups):
            member_offsets = torch.tensor(
                tuple(source_offset + int(index) for index in members),
                dtype=torch.long,
                device=dense_flat.device,
            )
            absolute_members = dense_flat.index_select(0, member_offsets)
            centroid = (
                dense_positions[-3:, 0]
                .index_select(1, absolute_members.to(dense_positions.device))
                .float()
                .mean(dim=1)
                .round()
                .to(dtype=dense_positions.dtype)
            )
            representative_offset = source_offset + representatives[result_index]
            representative_absolute = int(dense_flat[representative_offset].item())
            compact_index = int(old_to_compact[representative_absolute].item())
            if compact_index < 0:
                raise InvariantError(
                    "Centroid M-RoPE representative was removed from the prompt."
                )
            repaired[-3:, 0, compact_index] = centroid.to(repaired.device)
        source_offset += source_count
    return repaired


def _cumulative_lengths(counts: Sequence[int], *, device: Any) -> Any:
    torch = _torch()
    total = 0
    values = [0]
    for count in counts:
        total += int(count)
        values.append(total)
    return torch.tensor(values, dtype=torch.int32, device=device)


def _sequential_window_cu(
    cu_seqlens: Any,
    *,
    window_tokens: int,
    device: Any,
) -> Any:
    torch = _torch()
    if window_tokens < 1:
        raise InvariantError("Sequential visual window size must be positive.")
    bounds = [int(value) for value in cu_seqlens.tolist()]
    values = [0]
    for start, end in zip(bounds, bounds[1:]):
        position = start + window_tokens
        while position < end:
            values.append(position)
            position += window_tokens
        if end > start:
            values.append(end)
    return torch.unique_consecutive(
        torch.tensor(values, dtype=torch.int32, device=device)
    )


def _strict_feature_representatives(
    candidates: Any,
    *,
    source_count: int,
) -> tuple[int, ...]:
    values = [int(value) for value in candidates.tolist()]
    result_count = len(values)
    if result_count > source_count:
        raise InvariantError(
            "SAINT feature representatives exceed the dense source layout."
        )
    result: list[int] = []
    for index, candidate in enumerate(values):
        lower = 0 if not result else result[-1] + 1
        upper = source_count - (result_count - index)
        result.append(min(max(candidate, lower), upper))
    if result != sorted(set(result)) or any(
        value < 0 or value >= source_count for value in result
    ):
        raise InvariantError("SAINT could not legalize sparse feature positions.")
    return tuple(result)


def _vision_raw_keys(block: Any, normalised: Any) -> Any:
    qkv = block.attn.qkv(normalised).reshape(
        normalised.shape[0], 3, block.attn.num_heads, -1
    )
    return qkv[:, 1]


def _apply_rotary_pos_emb_vision_4573(q: Any, k: Any, cos: Any, sin: Any) -> tuple[Any, Any]:
    orig_q_dtype, orig_k_dtype = q.dtype, k.dtype
    q, k = q.float(), k.float()
    cos, sin = cos.unsqueeze(-2).float(), sin.unsqueeze(-2).float()
    q_embed = (q * cos) + (_rotate_half(q) * sin)
    k_embed = (k * cos) + (_rotate_half(k) * sin)
    return q_embed.to(orig_q_dtype), k_embed.to(orig_k_dtype)


def _bifocal_cache_length(past: Any) -> int:
    if past is None:
        return 0
    getter = getattr(past, "get_seq_length", None)
    if callable(getter):
        return int(getter())
    layers = getattr(past, "layers", None)
    if layers:
        return int(layers[0].get_seq_length())
    return 0


def _has_bifocal_registration(registrations: Sequence[Any]) -> bool:
    return any(
        getattr(getattr(item.executor, "contract", None), "algorithm", None)
        == "bifocal"
        for item in registrations
    )


def _has_centroid_mrope_registration(registrations: Sequence[Any]) -> bool:
    return any(
        getattr(getattr(item.executor, "contract", None), "algorithm", None)
        in _CENTROID_MROPE_ALGORITHMS
        for item in registrations
    )


def _bifocal_vit_column_mass(
    block: Any,
    normalised: Any,
    cu_seqlens: Any,
    position_embeddings: tuple[Any, Any],
) -> dict[str, Any]:
    torch = _torch()
    try:
        from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import (
            apply_rotary_pos_emb_vision,
        )
    except ImportError:
        apply_rotary_pos_emb_vision = _apply_rotary_pos_emb_vision_4573

    attn = block.attn
    with torch.no_grad():
        length = int(normalised.shape[0])
        qkv = attn.qkv(normalised).reshape(length, 3, attn.num_heads, -1).permute(1, 0, 2, 3)
        q, k = apply_rotary_pos_emb_vision(qkv[0], qkv[1], *position_embeddings)
        q = q.transpose(0, 1).float()
        k = k.transpose(0, 1).float()
        device = normalised.device
        mass = torch.zeros(length, dtype=torch.float32, device=device)
        mass_h = torch.zeros(attn.num_heads, length, dtype=torch.float32, device=device)
        diag = torch.zeros(length, dtype=torch.float32, device=device)
        bounds = [int(value) for value in cu_seqlens.tolist()]
        scaling = float(attn.scaling)
        for start, end in zip(bounds[:-1], bounds[1:]):
            if end <= start:
                continue
            keys = k[:, start:end]
            for i0 in range(start, end, 512):
                i1 = min(i0 + 512, end)
                a = torch.softmax(q[:, i0:i1] @ keys.transpose(1, 2) * scaling, dim=-1)
                mass[start:end] += a.sum(dim=(0, 1))
                mass_h[:, start:end] += a.sum(dim=1)
                rows = torch.arange(i1 - i0, device=device)
                diag[i0:i1] += a[:, rows, i0 - start + rows].sum(dim=0)
        value_norm = qkv[2].norm(dim=-1).t().float()
        return {
            "mass": mass,
            "nodiag": mass - diag,
            "vmass": (mass_h * value_norm).sum(dim=0),
        }


def _vision_rotated_qk(
    block: Any,
    normalised: Any,
    position_embeddings: tuple[Any, Any],
) -> tuple[Any, Any]:
    qkv = block.attn.qkv(normalised).reshape(
        normalised.shape[0], 3, block.attn.num_heads, -1
    )
    query = qkv[:, 0]
    key = qkv[:, 1]
    cos, sin = position_embeddings
    cos = cos.unsqueeze(-2).to(dtype=query.dtype)
    sin = sin.unsqueeze(-2).to(dtype=query.dtype)
    return (
        query * cos + _rotate_half(query) * sin,
        key * cos + _rotate_half(key) * sin,
    )


def _merged_vision_metrics(
    query: Any,
    key: Any,
    *,
    patch_counts: Sequence[int],
    feature_counts: Sequence[int],
    window_index: Any,
    spatial_merge_unit: int,
    scaling: float,
    key_sizes: Any | None = None,
) -> tuple[list[Any], list[Any]]:
    torch = _torch()
    query_chunks = torch.split(query.float(), tuple(patch_counts))
    key_chunks = torch.split(key.float(), tuple(patch_counts))
    window_chunks = torch.split(window_index, tuple(feature_counts))
    attentions: list[Any] = []
    keys: list[Any] = []
    size_chunks = torch.split(key_sizes.reshape(-1), tuple(patch_counts)) if key_sizes is not None else [None] * len(patch_counts)
    feature_offset = 0
    for query_chunk, key_chunk, window_chunk, feature_count, sizes in zip(
        query_chunks, key_chunks, window_chunks, feature_counts, size_chunks
    ):
        score = torch.einsum("qhd,khd->hqk", query_chunk, key_chunk) * scaling
        if sizes is not None:
            score = score + sizes.float().log().reshape(1, 1, -1)
        score = torch.softmax(score, dim=-1, dtype=torch.float32)
        score = score.reshape(
            score.shape[0],
            feature_count,
            spatial_merge_unit,
            feature_count,
            spatial_merge_unit,
        ).mean(dim=(2, 4))
        merged_keys = (
            key_chunk.reshape(
                feature_count,
                spatial_merge_unit,
                key_chunk.shape[1],
                key_chunk.shape[2],
            )
            .mean(dim=1)
            .permute(1, 0, 2)
        )
        local_window = window_chunk - feature_offset
        reverse = torch.argsort(local_window).to(device=score.device)
        attentions.append(score.index_select(-2, reverse).index_select(-1, reverse))
        keys.append(merged_keys.index_select(1, reverse))
        feature_offset += int(feature_count)
    return attentions, keys


def _merged_vision_keys(
    key: Any,
    *,
    patch_counts: Sequence[int],
    feature_counts: Sequence[int],
    window_index: Any,
    spatial_merge_unit: int,
) -> list[Any]:
    """Project final attention keys to native post-merger feature order."""

    torch = _torch()
    key_chunks = torch.split(key.float(), tuple(patch_counts))
    window_chunks = torch.split(window_index, tuple(feature_counts))
    keys: list[Any] = []
    feature_offset = 0
    for key_chunk, window_chunk, feature_count in zip(
        key_chunks, window_chunks, feature_counts
    ):
        merged_keys = (
            key_chunk.reshape(
                feature_count,
                spatial_merge_unit,
                key_chunk.shape[1],
                key_chunk.shape[2],
            )
            .mean(dim=1)
            .permute(1, 0, 2)
        )
        local_window = window_chunk - feature_offset
        reverse = torch.argsort(local_window).to(device=merged_keys.device)
        keys.append(merged_keys.index_select(1, reverse))
        feature_offset += int(feature_count)
    return keys


def _rotate_half(value: Any) -> Any:
    half = int(value.shape[-1]) // 2
    return _torch().cat((-value[..., half:], value[..., :half]), dim=-1)


def _restore_mrope_section(config: Any, *, axis_count: int) -> tuple[int, ...]:
    scaling = getattr(config, "rope_scaling", None)
    section = scaling.get("mrope_section") if isinstance(scaling, Mapping) else None
    if not isinstance(section, Sequence) or isinstance(section, (str, bytes)):
        raise BackendIntegrationError(
            "Qwen2.5-VL RESTORE requires native rope_scaling.mrope_section."
        )
    values = tuple(int(value) for value in section)
    if len(values) != axis_count or any(value < 0 for value in values):
        raise BackendIntegrationError(
            "Qwen2.5-VL RESTORE M-RoPE section does not match its position axes."
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
        sources = [
            source
            for source, target in delta.old_to_new.items()
            if target == result_index
        ]
        if not sources:
            raise InvariantError("LayoutDelta result has no source preimage.")
        result.append(sources[0])
    return tuple(result)


def _compose_layout_mapping(first: LayoutDelta, second: LayoutDelta) -> LayoutDelta:
    if first.result_count != second.source_count:
        raise InvariantError("Qwen2.5-VL layout composition count mismatch.")
    # Delegate to the contract-level composition so an AIM delta keeps its
    # EXPLICIT result ordering.  Rebuilding the same mapping with
    # LayoutDelta.from_mapping's default PRESERVE_SOURCE policy rejects valid
    # AIM bipartite groups whose representative order is intentionally
    # explicit rather than monotonic in the first source index.
    composed = first.compose(second)
    if "representative_indices" in second.result_layout:
        original = _layout_representatives(first)
        representatives = _layout_representatives(second)
        composed = replace(composed, result_layout={**dict(composed.result_layout),
            "representative_indices": [original[index] for index in representatives]})
    return composed


def _delta_groups(delta: LayoutDelta) -> tuple[tuple[int, ...], ...]:
    return tuple(
        tuple(
            source
            for source, target in delta.old_to_new.items()
            if target == result_index
        )
        for result_index in range(delta.result_count)
    )


def _mean_square_by_delta(value: Any, delta: LayoutDelta) -> Any:
    torch = _torch()
    groups = _delta_groups(delta)
    rows = []
    for query_group in groups:
        query_index = torch.tensor(query_group, dtype=torch.long, device=value.device)
        columns = []
        for key_group in groups:
            key_index = torch.tensor(key_group, dtype=torch.long, device=value.device)
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


def _decoder_state(
    language_model: Any,
    hidden_states: Any,
    padding_mask: Any,
    position_ids: Any,
    cache_position: Any,
    create_causal_mask: Callable[..., Any],
    create_sliding_window_causal_mask: Callable[..., Any],
    *,
    past_key_values: Any | None = None,
) -> tuple[dict[str, Any], Any | None, Any]:
    if int(position_ids.shape[0]) == 4:
        text_position_ids = position_ids[0]
        rotary_position_ids = position_ids[1:]
    else:
        text_position_ids = None
        rotary_position_ids = position_ids
    mask_kwargs = {
        "config": language_model.config,
        "input_embeds": hidden_states,
        "attention_mask": padding_mask,
        "cache_position": cache_position,
        "past_key_values": past_key_values,
        "position_ids": text_position_ids,
    }
    masks = {"full_attention": create_causal_mask(**mask_kwargs)}
    if language_model.has_sliding_layers:
        masks["sliding_attention"] = create_sliding_window_causal_mask(**mask_kwargs)
    return masks, text_position_ids, rotary_position_ids


def _layer_local_decode_padding_mask(
    padding_mask: Any | None,
    past_key_values: Any,
    *,
    layer_index: int,
    query_length: int,
    reference: Any,
) -> Any | None:
    """Build a batch-one mask sized to one decoder-compression cache layer."""

    if padding_mask is None:
        return None
    if getattr(padding_mask, "ndim", None) != 2 or int(padding_mask.shape[0]) != 1:
        raise BackendIntegrationError(
            "Qwen2.5-VL cached decoder compression requires a rank-2 batch-one mask."
        )
    if not bool(padding_mask.bool().all().item()):
        raise BackendIntegrationError(
            "Qwen2.5-VL cached decoder compression does not support padded prompts."
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
        from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import (
            apply_multimodal_rotary_pos_emb,
        )
    except ImportError as exc:  # pragma: no cover - exact provider guard covers this
        raise BackendIntegrationError(
            "transformers==4.57.3 Qwen2.5-VL RoPE helper is unavailable."
        ) from exc
    normalised = decoder_layer.input_layernorm(hidden_states)
    attention = decoder_layer.self_attn
    batch, sequence_length, _ = normalised.shape
    query = (
        attention.q_proj(normalised)
        .view(batch, sequence_length, -1, attention.head_dim)
        .transpose(1, 2)
    )
    keys = (
        attention.k_proj(normalised)
        .view(batch, sequence_length, -1, attention.head_dim)
        .transpose(1, 2)
    )
    if use_rope:
        cos, sin = (value.to(query.device) for value in position_embeddings)
        query, keys = apply_multimodal_rotary_pos_emb(
            query,
            keys,
            cos,
            sin,
            attention.rope_scaling["mrope_section"],
        )
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
    """Extract PACT pre-RoPE EUTI Q/K and post-RoPE clustering keys."""

    torch = _torch()
    try:
        from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import (
            apply_multimodal_rotary_pos_emb,
        )
    except ImportError as exc:  # pragma: no cover
        raise BackendIntegrationError(
            "transformers==4.57.3 Qwen2.5-VL RoPE helper is unavailable."
        ) from exc
    normalised = decoder_layer.input_layernorm(hidden_states.detach())
    attention = decoder_layer.self_attn
    batch, sequence_length, _ = normalised.shape
    query = (
        attention.q_proj(normalised)
        .view(batch, sequence_length, -1, attention.head_dim)
        .transpose(1, 2)
    )
    keys = (
        attention.k_proj(normalised)
        .view(batch, sequence_length, -1, attention.head_dim)
        .transpose(1, 2)
    )
    cos, sin = position_embeddings
    _rotated_query, rotated_keys = apply_multimodal_rotary_pos_emb(
        query,
        keys,
        cos.to(query.device),
        sin.to(query.device),
        attention.rope_scaling["mrope_section"],
    )
    groups = int(attention.num_key_value_groups)
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


def _effective_bool(value: Any, default: Any) -> bool:
    return bool(default if value is None else value)


def _tensor_int_values(value: Any) -> list[int]:
    if value is None:
        return []
    if hasattr(value, "detach"):
        value = value.detach().reshape(-1).cpu().tolist()
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        value = list(value)
    else:
        value = [value]
    return [int(item) for item in value]


def _layout_operation(delta: LayoutDelta) -> str:
    if delta.is_identity:
        return "identity"
    if delta.dropped_indices:
        return "drop"
    if delta.merge_groups:
        return "merge"
    return "remap"


def _torch() -> Any:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - dependency-light control plane
        raise RuntimeError("Qwen2.5-VL provider execution requires PyTorch.") from exc
    return torch


def _transformers_decoder_helpers() -> tuple[
    Any,
    Callable[..., Any],
    Callable[..., Any],
]:
    try:
        from transformers.masking_utils import (
            create_causal_mask,
            create_sliding_window_causal_mask,
        )
        from transformers.modeling_outputs import BaseModelOutputWithPast
    except ImportError as exc:  # pragma: no cover - exact version guard covers it
        raise BackendIntegrationError(
            "transformers==4.57.3 decoder helpers are unavailable."
        ) from exc
    return (
        BaseModelOutputWithPast,
        create_causal_mask,
        create_sliding_window_causal_mask,
    )


__all__ = [
    "QWEN25_VL_TRANSFORMERS_4573_CONTRACT",
    "Qwen25VLProviderBridgeContract",
    "Qwen25VLProviderRegistrationHandle",
    "Qwen25VLTransformers4573Bridge",
    "SUPPORTED_TRANSFORMERS_VERSION",
    "install_qwen25_vl_transformers_4573_bridge",
    "uninstall_qwen25_vl_transformers_4573_bridge",
]
