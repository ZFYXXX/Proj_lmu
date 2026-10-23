"""Pinned image-compression bridge for vendored LLaVA-NeXT ``ee14dbd``.

The source revision exposes no callback API, so the bridge installs guarded,
instance-local replacements around the exact monolithic boundaries.  The
replacements are reversible and restricted to image-only, batch-one
execution.  Ordinary methods use ``FORWARD(use_cache=False)``; PyramidDrop
also uses the shared DynamicCache prefill/decode lifecycle during guarded
generation.  Video, source drift, and unsupported output modes fail closed.
"""

from __future__ import annotations


from mllm_comp.compressors.image.positions import (
    tome_encoder_positions,
    tome_encoder_score_layer,
)
from mllm_comp.compressors.image.tome import tome_merge_reference
from mllm_comp.compressors.layer_positions import LayerPosition

import hashlib
import importlib.metadata
import inspect
from collections.abc import Callable, Mapping
from contextlib import nullcontext
from contextvars import ContextVar
from dataclasses import dataclass, replace
from pathlib import Path
from types import MappingProxyType, MethodType
from typing import Any

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
    pyramid_instruction_indices,
)
from mllm_comp.compressors.image.pact import (
    pact_attention_mask as _pact_attention_mask,
    pact_token_scores,
)
from mllm_comp.compressors.image.decoder_lifecycle import (
    CacheRepairPolicy,
    DecoderBoundary,
    DecoderCallPhase,
    ImageDecoderLifecycle,
    LayerLocalCacheView,
    dynamic_cache_layer_length,
    layout_result_representatives,
)
from mllm_comp.compressors.image.decoder_registration import (
    resolve_decoder_selection_depth,
)
from mllm_comp.compressors.image.decoder_selection import (
    AttentionCapture,
    DecoderSelectionCall,
    DecoderSelectionRegistration,
    capture_decoder_attention,
    decoder_layout_replay,
)
from mllm_comp.compressors.image.fastv import fastv_scorer_query_indices
from mllm_comp.compressors.image.nuwa import build_nuwa_prefill_position_ids
from mllm_comp.compressors.image.restore import (
    build_restore_prefill_position_ids,
    RestoreAttentionState,
)
from mllm_comp.core.layout import LayoutDelta
from mllm_comp.core.errors import BackendIntegrationError, InvariantError
from mllm_comp.core.vocabulary import ExecutionPhase

from .image_compression import (
    LLAVA_ONEVISION_BRIDGE_API_VERSION,
    LLAVA_ONEVISION_SOURCE_COMMIT,
    LLAVA_ONEVISION_SOURCE_ROUTE,
    FASTV_SITE,
    PYRAMID_DROP_SITE,
    RESTORE_CALIBRATION_SITE,
    TOME_MERGE_SITE,
    TOME_PACK_SITE,
    TOME_RESTORE_SITE,
    VISIONZIP_SITE,
    LlavaEncoderDenseRestoreRequest,
    LlavaEncoderDenseRestoreResponse,
    LlavaEncoderMergeRequest,
    LlavaEncoderMergeResponse,
    LlavaFastVRequest,
    LlavaDecoderLayerRequest,
    LlavaDecoderLayerResponse,
    LlavaDecoderLayoutRepair,
    LlavaPACTRequest,
    LlavaPackedImageRequest,
    LlavaPackedImageResponse,
    _crop_source_ids_in_packed_layout,
)


SUPPORTED_TRANSFORMERS_VERSION = "4.57.3"
_IMAGE_TOKEN_INDEX = -200


_NORMALIZED_SOURCE_HASHES: Mapping[str, str] = MappingProxyType(
    {
        "llava/model/multimodal_encoder/siglip_encoder.py": (
            "fe9b28fca491285c2225b6e2dc7fb63e34d4b6b0e97e327ff7f141180e9cc51b"
        ),
        "llava/model/llava_arch.py": (
            "1ac99c40a704e2652553335ba8674e8fe8b11aa9b61eaee9f7e9697335d6d720"
        ),
        "llava/model/language_model/llava_qwen.py": (
            "f55015d1b7f618462bea4f902004a8a3f9ac1e41c2e8d4dfa9b73fb1fb0c1cba"
        ),
    }
)

_EXPECTED_SIGNATURES: Mapping[str, tuple[str, ...]] = MappingProxyType(
    {
        "LlavaQwenForCausalLM.forward": (
            "self",
            "input_ids",
            "attention_mask",
            "position_ids",
            "past_key_values",
            "inputs_embeds",
            "labels",
            "use_cache",
            "output_attentions",
            "output_hidden_states",
            "images",
            "image_sizes",
            "return_dict",
            "modalities",
            "dpo_forward",
            "cache_position",
        ),
        "LlavaQwenForCausalLM.generate": (
            "self",
            "inputs",
            "images",
            "image_sizes",
            "modalities",
            "kwargs",
        ),
        "SigLipEncoder.forward": (
            "self",
            "inputs_embeds",
            "attention_mask",
            "output_attentions",
            "output_hidden_states",
            "return_dict",
        ),
        "SigLipEncoderLayer.forward": (
            "self",
            "hidden_states",
            "attention_mask",
            "output_attentions",
        ),
        "LlavaMetaForCausalLM.prepare_inputs_labels_for_multimodal": (
            "self",
            "input_ids",
            "position_ids",
            "attention_mask",
            "past_key_values",
            "labels",
            "images",
            "modalities",
            "image_sizes",
        ),
        "Qwen2Model.forward": (
            "self",
            "input_ids",
            "attention_mask",
            "position_ids",
            "past_key_values",
            "inputs_embeds",
            "use_cache",
            "cache_position",
            "kwargs",
        ),
    }
)

_BLOCKED_SITES: Mapping[str, str] = MappingProxyType(
    {
        "video": "video serialization is a separate exact support unit.",
    }
)


@dataclass(frozen=True)
class _ProviderComponents:
    model: Any
    core: Any
    vision_tower: Any
    encoder: Any
    layers: tuple[Any, ...]


@dataclass
class _EncoderExecution:
    invocation_id: str
    stream_ids: tuple[str, ...]
    token_sizes: Any | None = None
    proportional_attention_bias: Any | None = None
    coordinates: tuple[Any, ...] = ()
    positions: tuple[LayerPosition, ...] = ()
    score_layer: int | None = None
    score_keys: Any | None = None


@dataclass(frozen=True)
class _RootExecution:
    invocation_id: str


@dataclass
class _SequenceExecution:
    invocation_id: str
    image_indices: Any
    newline_indices: tuple[int, ...]
    image_sizes: tuple[tuple[int, int], ...]
    modalities: tuple[str, ...]
    restore_attention: RestoreAttentionState | None = None
    restore_source_visual_count: int | None = None


@dataclass(frozen=True)
class _VisionSourceCapture:
    vision_crops: Any
    projected_crops: Any
    attentions: Any | None = None
    semantic_crops: Any | None = None
    keys: Any | None = None
    projector_layers: tuple[Any, ...] | None = None
    attention_layers: Mapping[int, Any] | None = None


@dataclass(frozen=True)
class _CallbackRegistration:
    token: int
    site: str
    callback: Callable[[Any], Any]
    requires_packed_metrics: bool = False
    execution_phase: ExecutionPhase | None = None
    algorithm: str | None = None
    prune_layer: int | None = None
    prune_layers: tuple[int, ...] = ()
    score_layers: tuple[int, ...] = ()
    score_source: str | None = None
    activation_id: int | None = None
    decoder_selection: DecoderSelectionRegistration | None = None


class _RawDecoderCallback:
    """Adapt the published raw LLaVA Q/K callback at registration time."""

    def __init__(self, callback, algorithm):
        self.callback = callback
        self.algorithm = algorithm
        self.position_policy = "sparse_source" if algorithm == "fastv" else "contiguous"
        self.single_image_only = False
        self.cache_policy = (CacheRepairPolicy.COMPACT_PREFIX if algorithm == "fastv"
                             else CacheRepairPolicy.LAYER_LOCAL)

    def resolve_depth(self, num_layers):
        self.num_layers = num_layers
        if self.algorithm == "fastv":
            self.prune_layer = _callback_prune_layer(self.callback)
            if (
                self.prune_layer is None
                or self.prune_layer > num_layers
                or (self.prune_layer == num_layers
                    and getattr(self.callback, "_mllm_application_position",
                                LayerPosition("inter_layer", self.prune_layer)).boundary == "post_attention_pre_ffn")
            ):
                raise BackendIntegrationError("LLaVA decoder pre-layer registration requires a valid boundary.")
            self.source_layers = _callback_score_layers(self.callback)
            self.score_source = _callback_score_source(self.callback) or "last_token"
            self.positions = (getattr(self.callback, "_mllm_application_position",
                                     LayerPosition("inter_layer", self.prune_layer)),)
        else:
            self.source_layers = _callback_prune_layers(self.callback) or tuple(range(num_layers))
            self.positions = getattr(self.callback, "_mllm_application_positions",
                                     tuple(LayerPosition("inter_layer", i + 1) for i in self.source_layers))

    def new_call(self):
        return _RawDecoderCallbackCall(self)

    def apply(self, call, point, hidden_states, *, sequence, **layout):
        source = self.source_layers[self.positions.index(point)] if self.algorithm == "pyramid_drop" else None
        query, image_keys, attention_keys, visibility = call.take_attention(source)
        common = dict(invocation_id=sequence.invocation_id, num_layers=self.num_layers,
                      hidden_states=hidden_states, image_indices=sequence.image_indices.to(hidden_states.device),
                      attention_keys=attention_keys, attention_key_mask=visibility,
                      image_sizes=sequence.image_sizes, modalities=sequence.modalities,
                      newline_indices=sequence.newline_indices, **layout)
        if self.algorithm == "fastv":
            request = LlavaFastVRequest(prune_layer=point.index, scorer_query=query, **common)
        else:
            request = LlavaDecoderLayerRequest(layer_index=source, instruction_query=query,
                                               image_keys=image_keys, **common)
        response = self.callback(request)
        if self.algorithm == "pyramid_drop" and isinstance(response, LlavaDecoderLayerResponse) and not response.dropped:
            return None
        return response


class _RawDecoderCallbackCall(DecoderSelectionCall):
    def __init__(self, method):
        self.method = method
        self.attention = {None: (None, None, None, None)} if method.algorithm == "fastv" and not method.source_layers else {}

    def capture_input(self, layer_index, image_mask, *, padding_mask, labels):
        if self.method.algorithm != "fastv" or layer_index not in self.method.source_layers:
            return None
        queries = fastv_scorer_query_indices(image_mask, score_source=self.method.score_source,
                                             attention_mask=padding_mask, labels=labels)
        return AttentionCapture(layer_index, layer_index, queries)

    def capture_output(self, layer_index, image_mask, *, padding_mask, labels):
        if self.method.algorithm != "pyramid_drop" or layer_index not in self.method.source_layers or not bool(image_mask.any().item()):
            return None
        queries = pyramid_instruction_indices(image_mask, attention_mask=padding_mask, labels=labels)
        return AttentionCapture(layer_index, min(layer_index + 1, self.method.num_layers - 1), queries)

    def record_attention(self, capture, *, query, image_keys, attention_keys, visibility, image_indices):
        source = capture.source_layer if self.method.algorithm == "pyramid_drop" else None
        self.attention[source] = (query, image_keys, attention_keys, visibility)

    def take_attention(self, source):
        if source not in self.attention:
            raise InvariantError("LLaVA raw callback Q/K were not captured before application.")
        return self.attention.pop(source)


@dataclass
class _InstancePatch:
    owner: Any
    name: str
    had_instance_attribute: bool
    prior_instance_value: Any

    def restore(self) -> None:
        if self.had_instance_attribute:
            setattr(self.owner, self.name, self.prior_instance_value)
        elif self.name in vars(self.owner):
            delattr(self.owner, self.name)


class LlavaOneVisionProviderCallbackHandle:
    """Idempotent callback registration owned by one provider bridge."""

    def __init__(
        self,
        bridge: "PinnedLlavaOneVisionEe14dbdBridge",
        *,
        token: int,
        site: str,
    ) -> None:
        self._bridge = bridge
        self.token = token
        self.site = site
        self.removed = False

    def remove(self) -> None:
        if self.removed:
            return
        self._bridge._remove_callback(self.token, self.site)
        self.removed = True


class PinnedLlavaOneVisionEe14dbdBridge:
    """Reversible instance bridge for the exact vendored source revision."""

    api_version = LLAVA_ONEVISION_BRIDGE_API_VERSION
    source_route = LLAVA_ONEVISION_SOURCE_ROUTE
    source_commit = LLAVA_ONEVISION_SOURCE_COMMIT
    capabilities: Mapping[str, bool] = MappingProxyType(
        {
            "crop_isolated_encoder_state": True,
            "anyres_layout_metadata": True,
            "image_newline_indices": True,
            "packed_attention_and_keys": True,
            "multisource_embedding_payload": True,
            "visual_attention_crop_payload": True,
            "projector_input_and_layers": True,
            "nuwa_visual_source_payload": True,
            "restore_visual_source_payload": True,
            "decoder_loop_control": True,
            "sequence_layout_repair": True,
            "training_label_repair": True,
            "kv_cache_repair": True,
        }
    )

    def __init__(
        self,
        model: Any,
        *,
        _contract_verifier: Callable[[Any, _ProviderComponents], None] | None = None,
    ) -> None:
        self.model = model
        self._components = _resolve_components(model)
        verifier = _contract_verifier or _verify_pinned_contract
        verifier(model, self._components)
        self._contract_mode = (
            "pinned_source" if _contract_verifier is None else "injected_test_double"
        )
        self._callbacks: dict[str, _CallbackRegistration] = {}
        self._handles: dict[int, LlavaOneVisionProviderCallbackHandle] = {}
        self._patches: list[_InstancePatch] = []
        self._next_token = 1
        self._root_counter = 0
        self._projector_generation = ContextVar(f"llava_projector_generation_{id(self)}", default=False)
        self._encoder_counter = 0
        self._callback_counts = {
            TOME_MERGE_SITE: 0,
            TOME_RESTORE_SITE: 0,
            TOME_PACK_SITE: 0,
            VISIONZIP_SITE: 0,
            PYRAMID_DROP_SITE: 0,
            RESTORE_CALIBRATION_SITE: 0,
            FASTV_SITE: 0,
        }
        self._latest_layout_deltas: list[dict[str, Any]] = []
        self._latest_decoder_lifecycle: dict[str, Any] | None = None
        self._latest_restore_calibration: dict[str, Any] | None = None
        self._decoder_request_counter = 0
        self._layout_by_invocation: dict[tuple[str, str], tuple[Any, ...]] = {}
        self._root_execution: ContextVar[_RootExecution | None] = ContextVar(
            f"llava_root_execution_{id(self)}", default=None
        )
        self._encoder_execution: ContextVar[_EncoderExecution | None] = ContextVar(
            f"llava_encoder_execution_{id(self)}", default=None
        )
        self._sequence_execution: ContextVar[_SequenceExecution | None] = ContextVar(
            f"llava_sequence_execution_{id(self)}", default=None
        )
        self._training_labels: ContextVar[Any | None] = ContextVar(
            f"llava_training_labels_{id(self)}", default=None
        )
        self._vision_source_capture: ContextVar[_VisionSourceCapture | None] = ContextVar(
            f"llava_vision_source_capture_{id(self)}", default=None
        )
        self._prompt_text_embeddings: ContextVar[Any | None] = ContextVar(
            f"llava_prompt_text_embeddings_{id(self)}", default=None
        )
        self._decoder_lifecycle: ContextVar[ImageDecoderLifecycle | None] = ContextVar(
            f"llava_decoder_lifecycle_{id(self)}", default=None
        )
        self._nuwa_next_position: ContextVar[int | None] = ContextVar(
            f"llava_nuwa_next_position_{id(self)}", default=None
        )
        self._restore_next_position: ContextVar[int | None] = ContextVar(
            f"llava_restore_next_position_{id(self)}", default=None
        )
        self._attached_attribute: str | None = None

    @property
    def patched(self) -> bool:
        return bool(self._patches)

    def available_sites(self) -> tuple[str, ...]:
        return (
            TOME_MERGE_SITE,
            TOME_RESTORE_SITE,
            TOME_PACK_SITE,
            VISIONZIP_SITE,
            PYRAMID_DROP_SITE,
            RESTORE_CALIBRATION_SITE,
            FASTV_SITE,
        )

    def prepare_training_forward(self) -> None:
        """Apply training-only state required by active pinned callbacks."""

        if TOME_RESTORE_SITE in self._callbacks:
            self._components.encoder.gradient_checkpointing = False

    def register_callback(
        self,
        site: str,
        callback: Callable[[Any], Any],
        *,
        requires_packed_metrics: bool = False,
        decoder_selection: DecoderSelectionRegistration | None = None,
    ) -> LlavaOneVisionProviderCallbackHandle:
        if site not in self.available_sites():
            reason = _BLOCKED_SITES.get(site, "site is not part of the pinned bridge")
            raise BackendIntegrationError(
                f"LLaVA provider site {site!r} is fail-closed: {reason}"
            )
        if not callable(callback):
            raise BackendIntegrationError("LLaVA provider callback must be callable.")
        if not isinstance(requires_packed_metrics, bool):
            raise BackendIntegrationError(
                "LLaVA packed-metric registration flag must be a bool."
            )
        if site in self._callbacks:
            raise BackendIntegrationError(
                f"LLaVA provider site {site!r} already has an activation; "
                "same-site composition is not layout-safe."
            )
        callback_algorithm = decoder_selection.method.algorithm if decoder_selection is not None else _callback_algorithm(callback)
        if (
            site in {FASTV_SITE, PYRAMID_DROP_SITE}
            and decoder_selection is None
            and callback_algorithm == "nuwa"
        ):
            raise BackendIntegrationError(
                "Nüwa Stage 2 requires DecoderSelectionRegistration."
            )
        if decoder_selection is None and callback_algorithm in {"fastv", "pyramid_drop"}:
            if hasattr(callback, "_mllm_pyramid_schedule"):
                raise BackendIntegrationError("Scheduled PyramidDrop requires an explicit DecoderSelectionRegistration.")
            method = _RawDecoderCallback(callback, callback_algorithm)
            decoder_selection = DecoderSelectionRegistration(method, method.apply,
                _callback_execution_phase(callback), _callback_activation_id(callback))
        if decoder_selection is not None:
            resolve_decoder_selection_depth(decoder_selection.method, len(self._components.core.layers))
        pre_projector = "pre_vision_projector" in getattr(callback, "_mllm_projector_target_sites", ())
        if not self._patches:
            self._install_instance_patches(
                include_visual_source=callback_algorithm
                in {"mmtok", "vispruner", "zoo_prune", "nuwa", "restore"} or pre_projector
                or getattr(callback, "_mllm_requires_vision_source", False)
            )
        elif (pre_projector or callback_algorithm == "vispruner") and not any(patch.owner is self.model and patch.name == "encode_images" for patch in self._patches):
            self._patch_bound_method(self.model, "encode_images", self._encode_images_replacement(), self._patches)
        token = self._next_token
        self._next_token += 1
        handle = LlavaOneVisionProviderCallbackHandle(self, token=token, site=site)
        registration = _CallbackRegistration(
            token=token,
            site=site,
            callback=callback,
            requires_packed_metrics=requires_packed_metrics,
            execution_phase=decoder_selection.execution_phase if decoder_selection is not None else _callback_execution_phase(callback),
            algorithm=callback_algorithm,
            prune_layer=decoder_selection.method.positions[0].index if decoder_selection is not None else _callback_prune_layer(callback),
            prune_layers=tuple(p.index for p in decoder_selection.method.positions) if decoder_selection is not None else _callback_prune_layers(callback),
            score_layers=_callback_score_layers(callback),
            score_source=_callback_score_source(callback),
            activation_id=decoder_selection.activation_id if decoder_selection is not None else _callback_activation_id(callback),
            decoder_selection=decoder_selection,
        )
        if site == FASTV_SITE:
            if decoder_selection is not None:
                invalid_pre_layer = False
            elif registration.algorithm == "sgl":
                point = getattr(callback, "_mllm_application_position", LayerPosition("inter_layer", registration.prune_layer))
                source = getattr(callback, "_mllm_score_boundary", registration.prune_layer)
                depth = len(self._components.core.layers)
                invalid_pre_layer = (registration.prune_layer is None or point.index > depth
                    or (point.boundary == "post_attention_pre_ffn" and point.index == depth)
                    or (registration.algorithm != "sgl" and (source < 1 or source > depth
                        or source > point.index)))
            elif registration.algorithm == "nuwa":
                invalid_pre_layer = (
                    registration.prune_layer is None
                    or registration.prune_layer > len(self._components.core.layers)
                    or (registration.prune_layer == len(self._components.core.layers)
                        and getattr(callback, "_mllm_application_position", LayerPosition("inter_layer", registration.prune_layer)).boundary == "post_attention_pre_ffn")
                    or len(registration.score_layers) > 1
                    or any(source >= len(self._components.core.layers) or source > registration.prune_layer
                           for source in registration.score_layers)
                )
            else:
                invalid_pre_layer = (
                    registration.algorithm
                    not in {
                        "sgl",
                        "nuwa",
                    }
                    or registration.prune_layer is None
                    or (
                        registration.algorithm
                        == "nuwa"
                        and registration.prune_layer < 1
                    )
                    or (
                        registration.prune_layer is not None
                        and registration.prune_layer
                        >= len(self._components.core.layers)
                    )
                )
            if invalid_pre_layer:
                if not self._callbacks:
                    self._restore_instance_patches()
                raise BackendIntegrationError(
                    "LLaVA decoder pre-layer registration requires a valid boundary."
                )
        if site == RESTORE_CALIBRATION_SITE and registration.algorithm != "restore":
            if not self._callbacks:
                self._restore_instance_patches()
            raise BackendIntegrationError(
                "LLaVA decoder attention calibration is reserved for RESTORE."
            )
        if site == PYRAMID_DROP_SITE and decoder_selection is None and registration.algorithm == "pact" and (
            registration.prune_layer is None
            or registration.prune_layer >= len(self._components.core.layers) - 1
        ):
            if not self._callbacks:
                self._restore_instance_patches()
            raise BackendIntegrationError(
                "LLaVA PACT requires a completed-layer boundary with a later "
                "decoder layer for its Q/K metric."
            )
        if site == PYRAMID_DROP_SITE and decoder_selection is None and registration.algorithm == "pact":
            from mllm_comp.compressors.image.positions import pact_decoder_position
            try:
                pact_decoder_position(getattr(callback, "_mllm_position_stage", None),
                    callback._mllm_pact_spec, len(self._components.core.layers))
            except (TypeError, ValueError) as exc:
                if not self._callbacks:
                    self._restore_instance_patches()
                raise BackendIntegrationError(str(exc)) from exc
        self._callbacks[site] = registration
        self._handles[token] = handle
        return handle

    def runtime_metadata(self) -> dict[str, Any]:
        return {
            "provider": "vendored_llava_next_ee14dbd_instance_bridge",
            "source_commit": self.source_commit,
            "source_route": self.source_route,
            "bridge_api_version": self.api_version,
            "contract_mode": self._contract_mode,
            "patched": self.patched,
            "active_sites": sorted(self._callbacks),
            "packed_metric_sites": sorted(
                site
                for site, registration in self._callbacks.items()
                if registration.requires_packed_metrics
            ),
            "available_sites": list(self.available_sites()),
            "blocked_sites": dict(_BLOCKED_SITES),
            "capabilities": dict(self.capabilities),
            "callback_counts": dict(self._callback_counts),
            "latest_layout_deltas": list(self._latest_layout_deltas),
            "patch_targets": [
                f"{type(item.owner).__name__}.{item.name}" for item in self._patches
            ],
            "supported_phases": [
                ExecutionPhase.FORWARD.value,
                ExecutionPhase.GENERATE.value,
            ],
            "use_cache_by_phase": {
                ExecutionPhase.FORWARD.value: False,
                ExecutionPhase.GENERATE.value: True,
            },
            "generate_supported": True,
            "active_generate_supported": _generate_registration(
                tuple(self._callbacks.values())
            )
            is not None or self._projector_generate_supported(),
            "labels_supported": True,
            "use_cache_supported": True,
            "latest_decoder_lifecycle": self._latest_decoder_lifecycle,
            "latest_restore_calibration": self._latest_restore_calibration,
            "support_activation_claimed": False,
        }

    def detach(self) -> None:
        if self._callbacks:
            raise BackendIntegrationError(
                "Cannot detach the LLaVA provider bridge while callbacks are active."
            )
        self._restore_instance_patches()
        if (
            self._attached_attribute is not None
            and getattr(self.model, self._attached_attribute, None) is self
        ):
            delattr(self.model, self._attached_attribute)
        self._attached_attribute = None

    def _remove_callback(self, token: int, site: str) -> None:
        registered = self._callbacks.get(site)
        if registered is None or registered.token != token:
            return
        del self._callbacks[site]
        self._handles.pop(token, None)
        if not self._callbacks:
            self._restore_instance_patches()

    def _install_instance_patches(
        self, *, include_visual_source: bool = False
    ) -> None:
        if self._patches:
            return
        installed: list[_InstancePatch] = []
        try:
            self._patch_bound_method(
                self.model,
                "forward",
                self._root_forward_replacement(),
                installed,
            )
            self._patch_bound_method(
                self.model,
                "generate",
                self._generate_replacement(),
                installed,
            )
            self._patch_bound_method(
                self.model,
                "prepare_inputs_labels_for_multimodal",
                self._prepare_multimodal_replacement(),
                installed,
            )
            if include_visual_source:
                self._patch_bound_method(
                    self.model,
                    "encode_images",
                    self._encode_images_replacement(),
                    installed,
                )
            self._patch_bound_method(
                self._components.core,
                "forward",
                self._decoder_forward_replacement(),
                installed,
            )
            self._patch_bound_method(
                self._components.encoder,
                "forward",
                self._encoder_forward_replacement(),
                installed,
            )
            for layer_index, layer in enumerate(self._components.layers):
                self._patch_bound_method(
                    layer,
                    "forward",
                    self._layer_forward_replacement(layer_index),
                    installed,
                )
        except BaseException:
            for patch in reversed(installed):
                patch.restore()
            raise
        self._patches = installed

    def _patch_bound_method(
        self,
        owner: Any,
        name: str,
        replacement: Callable[..., Any],
        installed: list[_InstancePatch],
    ) -> None:
        if not callable(getattr(owner, name, None)):
            raise BackendIntegrationError(
                f"Pinned LLaVA provider target {type(owner).__name__}.{name} is missing."
            )
        had_instance = name in vars(owner)
        prior = vars(owner).get(name)
        setattr(owner, name, MethodType(replacement, owner))
        installed.append(_InstancePatch(owner, name, had_instance, prior))

    def _restore_instance_patches(self) -> None:
        for patch in reversed(self._patches):
            patch.restore()
        self._patches.clear()
        self._layout_by_invocation.clear()

    def _root_forward_replacement(self) -> Callable[..., Any]:
        original = self.model.forward
        signature = inspect.signature(original)

        def replacement(_model: Any, *args: Any, **kwargs: Any) -> Any:
            bound = signature.bind_partial(*args, **kwargs)
            labels = bound.arguments.get("labels")
            use_cache = bound.arguments.get("use_cache")
            past_key_values = bound.arguments.get("past_key_values")
            modalities = bound.arguments.get("modalities", ("image",))
            active_lifecycle = self._decoder_lifecycle.get()
            if self._projector_generation.get():
                if labels is not None or self._root_execution.get() is None:
                    raise BackendIntegrationError("Projector generation requires its guarded unlabeled request")
                bound.arguments["use_cache"] = True
                return original(*bound.args, **bound.kwargs)
            if labels is not None and bound.arguments.get("return_dict") is False:
                raise BackendIntegrationError(
                    "Compressed LLaVA training requires return_dict=True."
                )
            if active_lifecycle is None:
                if use_cache not in (None, False):
                    raise BackendIntegrationError(
                        "Pinned LLaVA compression bridge requires use_cache=False "
                        "outside guarded decoder-compression generation."
                    )
                if past_key_values is not None:
                    raise BackendIntegrationError(
                        "Pinned LLaVA compression bridge rejects past_key_values "
                        "outside guarded decoder-compression generation."
                    )
            elif use_cache is not True:
                raise BackendIntegrationError(
                    "LLaVA decoder-compression generation requires use_cache=True."
                )
            elif labels is not None:
                raise BackendIntegrationError(
                    "LLaVA decoder-compression generation does not accept labels."
                )
            if bound.arguments.get("dpo_forward") not in (None, False):
                raise BackendIntegrationError(
                    "Pinned LLaVA compression bridge does not support dpo_forward."
                )
            if (
                PYRAMID_DROP_SITE in self._callbacks
                or RESTORE_CALIBRATION_SITE in self._callbacks
                or FASTV_SITE in self._callbacks
            ):
                if bound.arguments.get("output_attentions") not in (None, False):
                    raise BackendIntegrationError(
                        "Pinned LLaVA decoder compression does not expose attention histories."
                    )
                if bound.arguments.get("output_hidden_states") not in (None, False):
                    raise BackendIntegrationError(
                        "Pinned LLaVA decoder compression does not expose hidden-state histories."
                    )
                if bound.arguments.get("return_dict") is False:
                    raise BackendIntegrationError(
                        "Pinned LLaVA decoder compression requires return_dict=True."
                    )
            # Resolve the nullable vendored signature at the guarded boundary.
            bound.arguments["use_cache"] = active_lifecycle is not None
            _require_image_only_modalities(modalities)
            if active_lifecycle is not None:
                if self._root_execution.get() is None:
                    raise BackendIntegrationError(
                        "LLaVA generation decoder call lost its root request context."
                    )
                return original(*bound.args, **bound.kwargs)

            self._root_counter += 1
            state = _RootExecution(invocation_id=f"forward-{self._root_counter}")
            token = self._root_execution.set(state)
            sequence_token = self._sequence_execution.set(None)
            label_token = self._training_labels.set(labels)
            source_capture_token = self._vision_source_capture.set(None)
            text_embedding_token = self._prompt_text_embeddings.set(None)
            nuwa_position_token = self._nuwa_next_position.set(None)
            restore_position_token = self._restore_next_position.set(None)
            lifecycle_token = None
            owned_lifecycle: ImageDecoderLifecycle | None = None
            if (
                PYRAMID_DROP_SITE in self._callbacks
                or RESTORE_CALIBRATION_SITE in self._callbacks
                or FASTV_SITE in self._callbacks
            ):
                owned_lifecycle = self._new_decoder_lifecycle(prefix="forward")
                lifecycle_token = self._decoder_lifecycle.set(owned_lifecycle)
            if labels is not None:
                # Let the bridge own label repair and loss after all prompt and
                # decoder mutations; the provider must not compute against the
                # original dense sequence.
                bound.arguments["labels"] = None
            try:
                output = original(*bound.args, **bound.kwargs)
                if labels is None:
                    return output
                repaired_labels = self._training_labels.get()
                if repaired_labels is None:
                    raise BackendIntegrationError(
                        "LLaVA training labels disappeared during compression."
                    )
                return attach_shifted_causal_loss(
                    output,
                    repaired_labels,
                    operation="LLaVA-OneVision image compression",
                )
            finally:
                if lifecycle_token is not None:
                    assert owned_lifecycle is not None
                    self._latest_decoder_lifecycle = owned_lifecycle.telemetry()
                    self._decoder_lifecycle.reset(lifecycle_token)
                self._training_labels.reset(label_token)
                self._prompt_text_embeddings.reset(text_embedding_token)
                self._nuwa_next_position.reset(nuwa_position_token)
                self._restore_next_position.reset(restore_position_token)
                self._vision_source_capture.reset(source_capture_token)
                self._sequence_execution.reset(sequence_token)
                self._root_execution.reset(token)

        preserve_bound_method_signature(replacement, original)
        return replacement

    def _pre_projector_packed_site(self) -> str | None:
        sites = []
        for site in (VISIONZIP_SITE, TOME_PACK_SITE):
            registration = self._callbacks.get(site)
            if registration is not None and "pre_vision_projector" in getattr(
                registration.callback, "_mllm_projector_target_sites", ()
            ):
                sites.append(site)
        if len(sites) > 1:
            raise BackendIntegrationError(
                "LLaVA supports one packed compression callback before the projector; "
                "successive raw layouts cannot use the same source mapping."
            )
        return sites[0] if sites else None

    def _encode_images_replacement(self) -> Callable[..., Any]:
        original = self.model.encode_images

        def replacement(_model: Any, images: Any) -> Any:
            pre_projector_site = self._pre_projector_packed_site()
            pre_projector = pre_projector_site is not None
            registration = self._callbacks.get(pre_projector_site or VISIONZIP_SITE)
            if pre_projector and type(self._components.core.mm_projector).__name__ == "PoolerProjector":
                raise BackendIntegrationError("Pre-projector packed compression requires a tokenwise projector; PoolerProjector consumes a fixed spatial grid.")
            if registration is None or (not pre_projector and not getattr(registration.callback, "_mllm_requires_vision_source", False) and registration.algorithm not in {
                "mmtok", "vispruner", "zoo_prune", "nuwa", "restore",
            }):
                return original(images)
            semantic_crops = None
            vision_keys = None
            attention_layers = None
            torch = _torch()
            requested_attention_layers = getattr(registration.callback, "_mllm_vision_attention_layers", ())
            if registration.algorithm in {"vispruner", "nuwa", "restore"} or requested_attention_layers:
                if getattr(images, "ndim", None) != 4:
                    raise BackendIntegrationError(
                        "LLaVA-OneVision visual-cue compression requires one "
                        "rank-four crop tensor."
                    )
                inner = self._components.vision_tower.vision_tower
                outputs = inner(
                    images.to(
                        device=self._components.vision_tower.device,
                        dtype=self._components.vision_tower.dtype,
                    ),
                    output_hidden_states=True,
                    output_attentions=True,
                    return_dict=True,
                )
                hidden_states = getattr(outputs, "hidden_states", None)
                attentions = getattr(outputs, "attentions", None)
                if not hidden_states or not attentions:
                    raise BackendIntegrationError(
                        "LLaVA-OneVision visual-cue compression did not receive "
                        "final SigLIP "
                        "hidden states and attention."
                    )
                if requested_attention_layers:
                    if any(index >= len(attentions) or index < -len(attentions) for index in requested_attention_layers):
                        raise BackendIntegrationError("Requested comparison attention layer exceeds the vision tower depth")
                    attention_layers = {index: attentions[index] for index in requested_attention_layers}
                # RESTORE's fixed reproduction point is the penultimate retained
                # SigLIP layer; other visual-cue methods use the native final state.
                if registration.algorithm == "restore":
                    if len(hidden_states) < 3 or len(attentions) < 2:
                        raise BackendIntegrationError(
                            "LLaVA-OneVision RESTORE requires a penultimate SigLIP layer."
                        )
                    vision = hidden_states[-2].to(images.dtype)
                    final_attention = attentions[-2]
                    if getattr(
                        registration.callback, "_mllm_restore_spec", None
                    ) is None:
                        raise BackendIntegrationError(
                            "LLaVA-OneVision RESTORE spec metadata is unavailable."
                        )
                else:
                    vision = hidden_states[-1].to(images.dtype)
                    final_attention = attentions[-1]
                if registration.algorithm == "nuwa":
                    spec = getattr(registration.callback, "_mllm_nuwa_spec", None)
                    if spec is None:
                        raise BackendIntegrationError(
                            "LLaVA-OneVision Nüwa spec metadata is unavailable."
                        )
                    start = int(spec.semantic_hidden_start)
                    end = min(int(spec.semantic_hidden_end), len(hidden_states))
                    if start >= end:
                        raise BackendIntegrationError(
                            "LLaVA-OneVision Nüwa semantic hidden range exceeds "
                            "the pinned SigLip depth."
                        )
                    semantic_crops = torch.stack(
                        tuple(hidden_states[index].float() for index in range(start, end))
                    ).mean(dim=0)
                    final_layer = inner.vision_model.encoder.layers[-1]
                    attention_input = final_layer.layer_norm1(hidden_states[-2])
                    vision_keys = final_layer.self_attn.k_proj(attention_input)
            else:
                vision = self._components.vision_tower(images)
                final_attention = None
            # Capture visual cues once; pre uses the dense projection only as
            # the native packing/metric reference and retains the vision graph.
            with torch.no_grad() if pre_projector else nullcontext():
                projected = self._components.core.mm_projector(vision)
            projector_layers = (
                tuple(self._components.core.mm_projector.children())
                if registration.algorithm == "zoo_prune"
                else None
            )
            self._vision_source_capture.set(
                _VisionSourceCapture(
                    vision_crops=(hidden_states[requested_attention_layers[0] % len(attentions) + 1].to(images.dtype)
                        if registration.algorithm == "dop" else vision),
                    projected_crops=projected,
                    attentions=final_attention,
                    semantic_crops=semantic_crops,
                    keys=vision_keys,
                    projector_layers=projector_layers,
                    attention_layers=attention_layers,
                )
            )
            return projected

        preserve_bound_method_signature(replacement, original)
        return replacement

    def _prepare_multimodal_replacement(self) -> Callable[..., Any]:
        original = self.model.prepare_inputs_labels_for_multimodal
        signature = inspect.signature(original)

        def replacement(_model: Any, *args: Any, **kwargs: Any) -> Any:
            if not (
                TOME_PACK_SITE in self._callbacks
                or VISIONZIP_SITE in self._callbacks
                or PYRAMID_DROP_SITE in self._callbacks
                or RESTORE_CALIBRATION_SITE in self._callbacks
                or FASTV_SITE in self._callbacks
            ):
                return original(*args, **kwargs)
            root = self._root_execution.get()
            if root is None:
                raise BackendIntegrationError(
                    "Compressed multimodal preparation was called outside guarded "
                    "LlavaQwenForCausalLM.forward."
                )
            bound = signature.bind_partial(*args, **kwargs)
            call = dict(bound.arguments)
            if call.get("past_key_values") is not None:
                if self._decoder_lifecycle.get() is not None or self._projector_generation.get():
                    # GenerationMixin re-enters the multimodal helper for
                    # single-token decode.  The vendored implementation takes
                    # its cache fast path and must not rebuild prompt layout.
                    return original(*args, **kwargs)
                raise BackendIntegrationError(
                    "LLaVA packed-image compression rejects cache state outside "
                    "guarded decoder-compression generation."
                )
            input_ids = call.get("input_ids")
            if getattr(input_ids, "ndim", None) != 2 or int(input_ids.shape[0]) != 1:
                raise BackendIntegrationError(
                    "LLaVA packed-image compression requires one input_ids batch item."
                )
            registration = self._callbacks.get(VISIONZIP_SITE)
            pre_projector_site = self._pre_projector_packed_site()
            pre_projector = pre_projector_site is not None
            mmtok_enabled = (
                registration is not None and registration.algorithm == "mmtok"
            )
            vispruner_enabled = (
                registration is not None and registration.algorithm == "vispruner"
            )
            zoo_prune_enabled = (
                registration is not None and registration.algorithm == "zoo_prune"
            )
            nuwa_enabled = (
                registration is not None and registration.algorithm == "nuwa"
            )
            restore_enabled = (
                registration is not None and registration.algorithm == "restore"
            )
            comparison_source = registration is not None and getattr(registration.callback, "_mllm_requires_vision_source", False)
            if mmtok_enabled:
                prompt_ids = _active_prompt_ids(
                    input_ids, call.get("attention_mask")
                )
                prompt_ids = prompt_ids[prompt_ids >= 0]
                if int(prompt_ids.numel()) < 1:
                    raise BackendIntegrationError(
                        "LLaVA-OneVision MMTok found no prompt text tokens."
                    )
                self._prompt_text_embeddings.set(
                    self._components.core.embed_tokens(prompt_ids)
                )
            instruction_selector = getattr(registration.callback, "_mllm_instruction_selector", None) if registration else None
            if instruction_selector is not None:
                prompt_ids = instruction_selector(input_ids, call.get("attention_mask"), self._training_labels.get())
                self._prompt_text_embeddings.set(self._components.core.embed_tokens(prompt_ids))
            modalities = _normalise_single_image_modalities(call.get("modalities"))
            image_sizes = _normalise_single_image_sizes(call.get("image_sizes"))
            source_labels = self._training_labels.get()
            if source_labels is None:
                prepared = original(*args, **kwargs)
            else:
                prepared_call = signature.bind_partial(*args, **kwargs)
                prepared_call.arguments["labels"] = source_labels
                prepared = original(*prepared_call.args, **prepared_call.kwargs)
            if not isinstance(prepared, tuple) or len(prepared) != 6:
                raise BackendIntegrationError(
                    "Pinned multimodal prepare method returned an unknown structure."
                )
            (
                prepared_ids,
                prepared_positions,
                prepared_mask,
                prepared_past,
                embeds,
                prepared_labels,
            ) = prepared
            if prepared_ids is not None or prepared_past is not None:
                raise BackendIntegrationError(
                    "Pinned image preparation did not produce the proven embedding path."
                )
            if source_labels is not None and prepared_labels is None:
                raise BackendIntegrationError(
                    "Pinned LLaVA preparation did not expand training labels."
                )
            if getattr(embeds, "ndim", None) != 3 or int(embeds.shape[0]) != 1:
                raise BackendIntegrationError(
                    "Pinned image preparation must return [1, tokens, hidden] embeddings."
                )
            active_ids = _active_prompt_ids(input_ids, call.get("attention_mask"))
            placeholder_positions = (
                (active_ids == _IMAGE_TOKEN_INDEX).nonzero(as_tuple=False).flatten()
            )
            if int(placeholder_positions.numel()) != 1:
                raise BackendIntegrationError(
                    "LLaVA packed-image compression requires exactly one IMAGE_TOKEN_INDEX."
                )
            image_start = int(placeholder_positions[0].item())
            text_count = int(active_ids.numel()) - 1
            packed_count = int(embeds.shape[1]) - text_count
            if packed_count <= 0:
                raise BackendIntegrationError(
                    "Could not derive a non-empty packed image span from multimodal output."
                )
            image_end = image_start + packed_count
            if image_end > int(embeds.shape[1]):
                raise BackendIntegrationError(
                    "Derived packed image span is outside prepared embeddings."
                )
            tokens = embeds[0, image_start:image_end]
            local_newlines = _image_newline_indices(
                tokens, getattr(self._components.core, "image_newline", None)
            )

            anyres_layout: Mapping[str, Any] = MappingProxyType(
                {
                    "packed_after_anyres": True,
                    "pre_llm_splice": True,
                    "image_index": 0,
                    "packed_token_count": packed_count,
                    "newline_indices": local_newlines,
                    "crop_boundaries": ((0, packed_count),),
                    "source": "ee14dbd_prepare_inputs_labels_for_multimodal",
                }
            )
            source_capture = (
                self._vision_source_capture.get()
                if mmtok_enabled
                or vispruner_enabled
                or zoo_prune_enabled
                or nuwa_enabled
                or restore_enabled
                or pre_projector
                or comparison_source
                else None
            )
            if (
                mmtok_enabled
                or vispruner_enabled
                or zoo_prune_enabled
                or nuwa_enabled
                or restore_enabled
                or pre_projector
                or comparison_source
            ) and source_capture is None:
                raise BackendIntegrationError(
                    "LLaVA-OneVision crop selection did not capture its visual "
                    "source payload."
                )
            patches_per_side = None
            grid_width = None
            grid_height = None
            patch_merge_type = None
            if source_capture is not None:
                patches_per_side = int(
                    self._components.vision_tower.num_patches_per_side
                )
                patch_merge_type = str(
                    getattr(self.model.config, "mm_patch_merge_type", "flat")
                )
                crop_count = int(source_capture.projected_crops.shape[0])
                if crop_count > 1 and (not pre_projector or (
                    patch_merge_type.startswith("spatial") and (isinstance(call.get("images"), list) or getattr(call.get("images"), "ndim", None) == 5))):
                    from llava.mm_utils import get_anyres_image_grid_shape

                    grid_width, grid_height = get_anyres_image_grid_shape(
                        image_sizes[0],
                        self.model.config.image_grid_pinpoints,
                        self._components.vision_tower.image_size,
                    )
            nuwa_position_source: tuple[tuple[int, ...], int] | None = None
            representative_position_source: tuple[tuple[int, ...], int] | None = None
            restore_position_source: tuple[
                tuple[int, ...], int, Any
            ] | None = None
            packed_vision_values = None
            packed_source_delta = None
            if pre_projector:
                if self._callbacks[pre_projector_site].algorithm == "divprune" and local_newlines:
                    raise BackendIntegrationError("DivPrune pre-projector compression cannot preserve protected-centre cosine metrics for LLM-width image_newline tokens, which have no vision-input representation.")
                source_images = call.get("images")
                raw_grid_input = isinstance(source_images, list) or getattr(source_images, "ndim", None) == 5
                merge_type = patch_merge_type if raw_grid_input else "flat"
                if "maxpool2x2" in str(merge_type):
                    raise BackendIntegrationError("Pre-projector packed compression cannot move AnyRes max-pooling across the projector.")
                try:
                    source_ids = _crop_source_ids_in_packed_layout(
                        crop_count=int(source_capture.vision_crops.shape[0]),
                        patch_count=int(source_capture.vision_crops.shape[1]),
                        patches_per_side=patches_per_side, grid_width=grid_width, grid_height=grid_height,
                        patch_merge_type=merge_type, image_size=image_sizes[0], source_count=packed_count,
                        newline_indices=local_newlines, device=source_capture.vision_crops.device)
                except BackendIntegrationError as exc:
                    raise BackendIntegrationError("Pre-projector packed compression requires a pure AnyRes token layout; resizing/pooling cannot be represented by crop source indices.") from exc
                flat_vision = source_capture.vision_crops.flatten(0, 1)
                packed_vision_values = flat_vision.index_select(0, source_ids.clamp_min(0))
                packed_vision_values = packed_vision_values.masked_fill((source_ids < 0).unsqueeze(-1), 0)
            # Preserve the original callback order unless the other packed
            # reducer explicitly precedes ToMe's permanent output step.
            packed_sites = (VISIONZIP_SITE, TOME_PACK_SITE) if pre_projector_site == VISIONZIP_SITE else (TOME_PACK_SITE, VISIONZIP_SITE)
            for site in packed_sites:
                if site not in self._callbacks:
                    continue
                registration = self._callbacks[site]
                site_before_projector = site == pre_projector_site
                if registration.algorithm == "vispruner" and packed_source_delta is not None and not packed_source_delta.is_identity:
                    raise BackendIntegrationError("VisPruner post-projector crop selection requires the original per-crop layout; permanent ToMe groups no longer preserve its exact crop selection and budget. Use the pre-projector position before ToMe output merging.")
                if registration.requires_packed_metrics and not (site_before_projector and registration.algorithm == "aim"):
                    attention, keys = _packed_qk_metrics(
                        self._components.core.layers[0], tokens
                    )
                else:
                    attention = None
                    keys = None
                response = self._dispatch(
                    site,
                    LlavaPackedImageRequest(
                        invocation_id=root.invocation_id,
                        image_index=0,
                        tokens=packed_vision_values if site_before_projector else tokens,
                        attention=attention,
                        keys=keys,
                        image_sizes=image_sizes,
                        modalities=modalities,
                        newline_indices=local_newlines,
                        anyres_layout=anyres_layout,
                        projected_crops=(
                            source_capture.projected_crops
                            if source_capture is not None
                            else None
                        ),
                        vision_crops=(
                            source_capture.vision_crops
                            if source_capture is not None
                            else None
                        ),
                        vision_attentions=(
                            source_capture.attentions
                            if source_capture is not None
                            else None
                        ),
                        semantic_crops=(
                            source_capture.semantic_crops
                            if source_capture is not None
                            else None
                        ),
                        vision_keys=(
                            source_capture.keys
                            if source_capture is not None
                            else None
                        ),
                        text_embeddings=(
                            self._prompt_text_embeddings.get()
                            if source_capture is not None
                            else None
                        ),
                        vision_attention_layers=source_capture.attention_layers if source_capture is not None else None,
                        patches_per_side=patches_per_side,
                        grid_width=grid_width,
                        grid_height=grid_height,
                        patch_merge_type=merge_type if site_before_projector else patch_merge_type,
                        projector_layers=(
                            source_capture.projector_layers
                            if source_capture is not None
                            else None
                        ),
                    ),
                )
                if not isinstance(response, LlavaPackedImageResponse):
                    raise BackendIntegrationError(
                        f"{site} callback returned the wrong response type."
                    )
                if site_before_projector:
                    # Structural tokens retain their native LLM values and never
                    # enter the projector, even when vision/LLM widths differ.
                    torch = _torch()
                    structural = set(response.newline_indices)
                    content_indices = torch.tensor([i for i in range(response.tokens.shape[0]) if i not in structural], dtype=torch.long, device=response.tokens.device)
                    values = self._components.core.mm_projector(response.tokens.index_select(0, content_indices))
                    output = values.new_zeros((response.tokens.shape[0], values.shape[-1])).index_copy(0, content_indices.to(values.device), values)
                    if local_newlines:
                        old_newlines = torch.tensor(local_newlines, dtype=torch.long, device=tokens.device)
                        new_newlines = torch.tensor(response.newline_indices, dtype=torch.long, device=values.device)
                        output = output.index_copy(0, new_newlines, tokens.index_select(0, old_newlines).to(values.device))
                    response = replace(response, tokens=output)
                if (
                    response.image_index != 0
                    or response.image_sizes != image_sizes
                    or response.modalities != modalities
                ):
                    raise BackendIntegrationError(
                        f"{site} callback changed image batch identity."
                    )
                if response.layout_delta.source_count != packed_count:
                    raise BackendIntegrationError(
                        f"{site} LayoutDelta source count differs from packed image span."
                    )
                if registration.algorithm == "nuwa":
                    nuwa_position_source = (
                        tuple(response.layout_delta.kept_indices),
                        packed_count,
                    )
                elif registration.algorithm == "gap":
                    representative_position_source = (
                        tuple(response.layout_delta.kept_indices),
                        packed_count,
                    )
                elif registration.algorithm == "prumerge":
                    representatives = tuple(
                        int(value)
                        for value in response.layout_delta.result_layout.get(
                            "representative_indices", ()
                        )
                    )
                    if len(representatives) != response.layout_delta.result_count:
                        raise BackendIntegrationError(
                            "LLaVA-OneVision PruMerge response omitted source "
                            "representative positions."
                        )
                    representative_position_source = (
                        representatives,
                        packed_count,
                    )
                elif registration.algorithm == "restore":
                    if (
                        response.original_visual_length is None
                        or response.visual_sizes is None
                        or len(response.representative_indices)
                        != response.layout_delta.result_count
                    ):
                        raise BackendIntegrationError(
                            "LLaVA-OneVision RESTORE response omitted position/size metadata."
                        )
                    restore_position_source = (
                        response.representative_indices,
                        int(response.original_visual_length),
                        response.visual_sizes,
                    )
                if site == TOME_PACK_SITE:
                    packed_source_delta = response.layout_delta
                old_embeds = embeds
                embeds = _replace_packed_image_tokens(
                    old_embeds,
                    image_start=image_start,
                    image_end=image_end,
                    tokens=response.tokens,
                )
                keep = _layout_sequence_representatives(
                    response.layout_delta,
                    image_start=image_start,
                    image_end=image_end,
                    sequence_length=int(old_embeds.shape[1]),
                    device=old_embeds.device,
                )
                if site == TOME_PACK_SITE and representative_position_source is not None:
                    original_representatives, original_count = representative_position_source
                    image_keep = keep[image_start:image_start + response.layout_delta.result_count] - image_start
                    representative_position_source = (
                        tuple(original_representatives[index] for index in image_keep.tolist()), original_count)
                prepared_mask = _select_sequence_axis(prepared_mask, keep)
                prepared_positions = _select_sequence_axis(prepared_positions, keep)
                if prepared_labels is not None:
                    prepared_labels = compact_ignored_labels(
                        prepared_labels,
                        keep,
                        source_length=int(old_embeds.shape[1]),
                        operation=f"LLaVA packed compression at {site}",
                    )
                packed_count = int(response.layout_delta.result_count)
                image_end = image_start + packed_count
                local_newlines = response.newline_indices
                anyres_layout = response.anyres_layout
                tokens = response.tokens

            restore_attention = None
            restore_source_visual_count = None
            if restore_position_source is not None:
                representatives, original_visual_length, visual_sizes = (
                    restore_position_source
                )
                torch = _torch()
                selected = torch.tensor(
                    representatives,
                    dtype=torch.long,
                    device=embeds.device,
                )
                positions = build_restore_prefill_position_ids(
                    int(embeds.shape[1]),
                    visual_span=(image_start, image_end),
                    representative_positions=selected,
                    original_visual_length=original_visual_length,
                    device=embeds.device,
                )
                prepared_positions = positions.unsqueeze(0)
                spec = getattr(registration.callback, "_mllm_restore_spec", None)
                if spec is None:
                    raise BackendIntegrationError(
                        "LLaVA-OneVision RESTORE decoder spec is unavailable."
                    )
                restore_attention = RestoreAttentionState.prefill(
                    prepared_positions,
                    image_indices=torch.arange(
                        image_start, image_end, device=embeds.device
                    ),
                    visual_sizes=visual_sizes,
                    source_visual_count=original_visual_length,
                    first_layer_size=spec.first_layer_size,
                    reference=embeds,
                )
                restore_source_visual_count = original_visual_length
                self._restore_next_position.set(int(positions[-1].item()) + 1)
            elif representative_position_source is not None:
                representatives, original_visual_length = (
                    representative_position_source
                )
                torch = _torch()
                selected = torch.tensor(
                    representatives,
                    dtype=torch.long,
                    device=embeds.device,
                )
                prepared_positions = build_nuwa_prefill_position_ids(
                    int(embeds.shape[1]),
                    visual_span=(image_start, image_end),
                    selected_indices=selected,
                    original_visual_length=original_visual_length,
                    device=embeds.device,
                ).unsqueeze(0)
            elif nuwa_position_source is None:
                prepared_positions = _rebuild_text_position_ids(
                    prepared_mask,
                    reference=prepared_positions,
                    sequence_length=int(embeds.shape[1]),
                )
            else:
                representatives, original_visual_length = nuwa_position_source
                torch = _torch()
                selected = torch.tensor(
                    representatives,
                    dtype=torch.long,
                    device=embeds.device,
                )
                positions = build_nuwa_prefill_position_ids(
                    int(embeds.shape[1]),
                    visual_span=(image_start, image_end),
                    selected_indices=selected,
                    original_visual_length=original_visual_length,
                    device=embeds.device,
                )
                prepared_positions = positions.unsqueeze(0)
                self._nuwa_next_position.set(int(positions[-1].item()) + 1)

            torch = _torch()
            image_indices = torch.arange(
                image_start, image_end, dtype=torch.long, device=embeds.device
            )
            from mllm_comp.fixed_mask.runtime import record_llava_layout
            record_llava_layout(self.model, image_indices, embeds.shape[1], prepared_mask, embeds.device)
            comparison = getattr(self.model, "_mllm_comparison_operations", None)
            if comparison is not None:
                from mllm_comp.fixed_mask.runtime import make_layout
                vision_mask = torch.zeros(embeds.shape[:2], dtype=torch.bool, device=embeds.device)
                vision_mask[:, image_indices] = True
                comparison.incoming.set(make_layout(vision_mask, prepared_mask))
            self._sequence_execution.set(
                _SequenceExecution(
                    invocation_id=root.invocation_id,
                    image_indices=image_indices,
                    newline_indices=tuple(
                        image_start + item for item in local_newlines
                    ),
                    image_sizes=image_sizes,
                    modalities=modalities,
                    restore_attention=restore_attention,
                    restore_source_visual_count=restore_source_visual_count,
                )
            )
            if source_labels is not None:
                self._training_labels.set(prepared_labels)
            return (
                None,
                prepared_positions,
                prepared_mask,
                None,
                embeds,
                None,
            )

        return replacement

    def _generate_replacement(self) -> Callable[..., Any]:
        original = self.model.generate

        def replacement(_model: Any, *args: Any, **kwargs: Any) -> Any:
            registrations = tuple(self._callbacks.values())
            if self._projector_generate_supported():
                return self._projector_generate(original, args, kwargs)
            registration = _generate_registration(registrations)
            if registration is None:
                raise BackendIntegrationError(
                    "Pinned LLaVA does not support generate for this activation; "
                    "generation requires one GENERATE-phase decoder image method."
                )
            if self._decoder_lifecycle.get() is not None:
                raise BackendIntegrationError(
                    "Nested LLaVA compressed generation is unsupported."
                )
            self._root_counter += 1
            root = _RootExecution(invocation_id=f"generate-{self._root_counter}")
            lifecycle = self._new_decoder_lifecycle(
                prefix="generate", registration=registration
            )
            root_token = self._root_execution.set(root)
            sequence_token = self._sequence_execution.set(None)
            label_token = self._training_labels.set(None)
            lifecycle_token = self._decoder_lifecycle.set(lifecycle)
            nuwa_position_token = self._nuwa_next_position.set(None)
            restore_position_token = self._restore_next_position.set(None)
            source_capture_token = self._vision_source_capture.set(None)
            self._latest_restore_calibration = None
            try:
                output = original(*args, **kwargs)
                if not lifecycle.prefill_complete:
                    raise BackendIntegrationError(
                        "LLaVA generate returned before compressed prefill completed."
                    )
                return output
            finally:
                self._latest_decoder_lifecycle = lifecycle.telemetry()
                self._decoder_lifecycle.reset(lifecycle_token)
                self._nuwa_next_position.reset(nuwa_position_token)
                self._restore_next_position.reset(restore_position_token)
                self._vision_source_capture.reset(source_capture_token)
                self._training_labels.reset(label_token)
                self._sequence_execution.reset(sequence_token)
                self._root_execution.reset(root_token)

        return replacement

    def _projector_generate_supported(self):
        registrations = tuple(self._callbacks.values())
        return (len(registrations) == 1 and registrations[0].site == VISIONZIP_SITE
            and registrations[0].execution_phase is ExecutionPhase.GENERATE
            and registrations[0].algorithm in {"cdpruner", "dppruner", "eadp", "hiprune", "dop"})

    def _projector_generate(self, original, args, kwargs):
        if self._projector_generation.get() or self._root_execution.get() is not None:
            raise BackendIntegrationError("Nested LLaVA compressed generation is unsupported")
        self._root_counter += 1
        contexts = ((self._projector_generation, True),
            (self._root_execution, _RootExecution(invocation_id=f"projector-generate-{self._root_counter}")),
            (self._sequence_execution, None), (self._training_labels, None),
            (self._vision_source_capture, None), (self._prompt_text_embeddings, None))
        tokens = [(variable, variable.set(value)) for variable, value in contexts]
        try:
            # Packed selection finishes before the native generation loop.
            # Operation methods can further shorten KV at their own boundaries.
            return original(*args, **kwargs)
        finally:
            for variable, token in reversed(tokens):
                variable.reset(token)

    def _new_decoder_lifecycle(
        self,
        *,
        prefix: str,
        registration: _CallbackRegistration | None = None,
    ) -> ImageDecoderLifecycle:
        if registration is None:
            candidates = tuple(
                item
                for site, item in self._callbacks.items()
                if site
                in {PYRAMID_DROP_SITE, RESTORE_CALIBRATION_SITE, FASTV_SITE}
            )
            registration = candidates[0] if len(candidates) == 1 else None
        if registration is None or (registration.decoder_selection is None and registration.algorithm not in {
            "pyramid_drop",
            "fastv",
            "sgl",
            "nuwa",
            "restore",
            "pact",
        }):
            raise BackendIntegrationError(
                "LLaVA decoder lifecycle requires exactly one decoder image method."
            )
        self._decoder_request_counter += 1
        return ImageDecoderLifecycle(
            request_id=f"{prefix}-{self._decoder_request_counter}",
            algorithm=registration.algorithm,
            num_layers=len(self._components.core.layers),
            requires_decoder_boundary=(registration.algorithm != "restore"),
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
                    operation=f"LLaVA decoder {boundary.value} layer {layer_index}",
                )
            )

    def _decoder_forward_replacement(self) -> Callable[..., Any]:
        core = self._components.core
        original = core.forward
        signature = inspect.signature(original)

        def replacement(_core: Any, *args: Any, **kwargs: Any) -> Any:
            if (
                PYRAMID_DROP_SITE not in self._callbacks
                and RESTORE_CALIBRATION_SITE not in self._callbacks
                and FASTV_SITE not in self._callbacks
            ):
                return original(*args, **kwargs)
            root = self._root_execution.get()
            sequence = self._sequence_execution.get()
            if root is None or sequence is None:
                raise BackendIntegrationError(
                    "LLaVA decoder compression requires guarded multimodal preparation."
                )
            call = _bind_decoder_call(signature, args, kwargs)
            from mllm_comp.fixed_mask.runtime import attach_layout
            attach_layout(self.model, call)
            return self._execute_decoder_loop(call, sequence)

        return replacement

    def _execute_decoder_loop(
        self,
        call: dict[str, Any],
        sequence: _SequenceExecution,
    ) -> Any:
        (
            BaseModelOutputWithPast,
            create_causal_mask,
            create_sliding_window_causal_mask,
        ) = _transformers_decoder_helpers()
        input_ids = call.pop("input_ids")
        inputs_embeds = call.pop("inputs_embeds")
        padding_mask = call.pop("attention_mask")
        position_ids = call.pop("position_ids")
        past_key_values = call.pop("past_key_values")
        use_cache = call.pop("use_cache")
        cache_position = call.pop("cache_position")
        output_attentions = call.pop("output_attentions", None)
        output_hidden_states = call.pop("output_hidden_states", None)
        return_dict = call.pop("return_dict", None)
        lifecycle = self._decoder_lifecycle.get()
        if lifecycle is None:
            raise BackendIntegrationError(
                "LLaVA decoder compression requires a guarded request lifecycle."
            )
        if use_cache is None:
            use_cache = bool(getattr(self._components.core.config, "use_cache", False))
        if not isinstance(use_cache, bool):
            raise BackendIntegrationError(
                "Pinned LLaVA decoder compression requires a resolved bool use_cache."
            )
        if output_attentions not in (None, False) or output_hidden_states not in (
            None,
            False,
        ):
            raise BackendIntegrationError(
                "Pinned LLaVA decoder compression cannot expose sequence histories."
            )
        if return_dict is False:
            raise BackendIntegrationError(
                "Pinned LLaVA decoder compression requires return_dict=True."
            )
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise BackendIntegrationError(
                "LLaVA decoder requires exactly one of input_ids or inputs_embeds."
            )
        if inputs_embeds is None:
            inputs_embeds = self._components.core.embed_tokens(input_ids)
        if (
            getattr(inputs_embeds, "ndim", None) != 3
            or int(inputs_embeds.shape[0]) != 1
        ):
            raise BackendIntegrationError(
                "Pinned LLaVA decoder compression supports one decoder batch item."
            )
        sequence_length = int(inputs_embeds.shape[1])
        if use_cache and past_key_values is None:
            past_key_values = _new_dynamic_cache(self._components.core.config)
        decoder_call = lifecycle.begin_call(
            query_length=sequence_length,
            use_cache=use_cache,
            past_key_values=past_key_values,
        )
        try:
            if decoder_call.phase is not DecoderCallPhase.DECODE and (
                int(sequence.image_indices.numel()) == 0
                or int(sequence.image_indices.max().item()) >= sequence_length
            ):
                raise BackendIntegrationError(
                    "Prepared image indices do not match decoder embeddings."
                )
            if decoder_call.phase is DecoderCallPhase.DECODE and sequence_length != 1:
                raise BackendIntegrationError(
                    "LLaVA decoder compression requires one decode query token."
                )
            return self._run_decoder_layers(
                call=call,
                sequence=sequence,
                inputs_embeds=inputs_embeds,
                padding_mask=padding_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                cache_position=cache_position,
                decoder_call_phase=decoder_call.phase,
                lifecycle=lifecycle,
                BaseModelOutputWithPast=BaseModelOutputWithPast,
                create_causal_mask=create_causal_mask,
                create_sliding_window_causal_mask=create_sliding_window_causal_mask,
            )
        except BaseException:
            lifecycle.abort_call()
            raise

    def _run_decoder_layers(
        self,
        *,
        call: dict[str, Any],
        sequence: _SequenceExecution,
        inputs_embeds: Any,
        padding_mask: Any,
        position_ids: Any,
        past_key_values: Any,
        use_cache: bool,
        cache_position: Any,
        decoder_call_phase: DecoderCallPhase,
        lifecycle: ImageDecoderLifecycle,
        BaseModelOutputWithPast: Any,
        create_causal_mask: Callable[..., Any],
        create_sliding_window_causal_mask: Callable[..., Any],
    ) -> Any:
        torch = _torch()
        decoder_registrations = tuple(
            item
            for site, item in self._callbacks.items()
            if site in {PYRAMID_DROP_SITE, RESTORE_CALIBRATION_SITE, FASTV_SITE}
        )
        if len(decoder_registrations) != 1:
            raise BackendIntegrationError(
                "LLaVA decoder loop requires exactly one image method."
            )
        decoder_registration = decoder_registrations[0]
        decoder_algorithm = decoder_registration.algorithm
        if decoder_registration.decoder_selection is None and decoder_algorithm not in {
            "pyramid_drop",
            "fastv",
            "sgl",
            "nuwa",
            "restore",
            "pact",
        }:
            raise BackendIntegrationError(
                f"Unsupported LLaVA decoder algorithm {decoder_algorithm!r}."
            )
        sequence_length = int(inputs_embeds.shape[1])
        if isinstance(padding_mask, Mapping):
            raise BackendIntegrationError(
                "Pinned LLaVA decoder compression rejects caller-supplied mask mappings."
            )
        if decoder_call_phase is DecoderCallPhase.DECODE:
            if padding_mask is not None and (
                getattr(padding_mask, "ndim", None) != 2
                or int(padding_mask.shape[0]) != 1
                or not bool(padding_mask.bool().all().item())
            ):
                raise BackendIntegrationError(
                    "LLaVA cached decoder compression requires an unpadded batch-one mask."
                )
        else:
            if padding_mask is None:
                padding_mask = torch.ones(
                    (1, sequence_length),
                    dtype=torch.bool,
                    device=inputs_embeds.device,
                )
            elif getattr(padding_mask, "ndim", None) != 2 or tuple(
                padding_mask.shape
            ) != (1, sequence_length):
                raise BackendIntegrationError(
                    "Pinned LLaVA decoder compression requires a rank-2 batch-one padding mask."
                )
            if cache_position is None:
                cache_position = torch.arange(
                    sequence_length, device=inputs_embeds.device
                )
            if position_ids is None:
                position_ids = cache_position.unsqueeze(0)
            if getattr(position_ids, "ndim", None) != 2 or tuple(
                position_ids.shape
            ) != (1, sequence_length):
                raise BackendIntegrationError(
                    "Pinned LLaVA decoder compression requires rank-2 text position_ids."
                )

        hidden_states = inputs_embeds
        causal_masks: dict[str, Any] = {}
        position_embeddings: tuple[Any, Any] | None = None
        if decoder_call_phase is not DecoderCallPhase.DECODE:
            causal_masks = _decoder_masks(
                self._components.core,
                hidden_states,
                padding_mask,
                position_ids,
                cache_position,
                create_causal_mask,
                create_sliding_window_causal_mask,
            )
            position_embeddings = self._components.core.rotary_emb(
                hidden_states, position_ids
            )
        layers = tuple(self._components.core.layers)
        selection_registration = decoder_registration.decoder_selection
        selection_method = selection_registration.method if selection_registration is not None else None
        selection = selection_method.new_call() if selection_method is not None else None
        if (selection is not None and decoder_call_phase is not DecoderCallPhase.DECODE
                and callable(getattr(selection, "set_visual_layout", None))):
            from mllm_comp.compressors.comparison.decoder import llava_visual_segments
            first = int(sequence.image_indices[0])
            selection.set_visual_layout(llava_visual_segments(
                count=int(sequence.image_indices.numel()),
                newlines=[i - first for i in sequence.newline_indices],
                side=int(self._components.vision_tower.num_patches_per_side),
                merge_type=str(self.model.config.mm_patch_merge_type)))
        if selection is not None and decoder_call_phase is not DecoderCallPhase.DECODE:
            initial_image_mask = torch.zeros(inputs_embeds.shape[1], dtype=torch.bool, device=inputs_embeds.device)
            initial_image_mask[sequence.image_indices.to(inputs_embeds.device)] = True
            selection.initialize(inputs_embeds[0], initial_image_mask,
                padding_mask=padding_mask, labels=self._training_labels.get())
        boundary_replay = None
        if (decoder_call_phase is not DecoderCallPhase.DECODE and selection_method is not None
                and selection_method.single_image_only
                and (len(sequence.image_sizes) != 1 or tuple(sequence.modalities) != ("image",))):
            raise BackendIntegrationError(f"LLaVA {selection_method.algorithm} supports exactly one image per prompt.")
        if (
            decoder_algorithm == "sgl"
            and decoder_call_phase is not DecoderCallPhase.DECODE
        ):
            if len(sequence.image_sizes) != 1 or tuple(sequence.modalities) != (
                "image",
            ):
                raise BackendIntegrationError(
                    "LLaVA SGL supports exactly one image per prompt."
                )
        if (
            decoder_algorithm == "nuwa"
            and decoder_call_phase is not DecoderCallPhase.DECODE
        ):
            if len(sequence.image_sizes) != 1 or tuple(sequence.modalities) != (
                "image",
            ):
                raise BackendIntegrationError(
                    "LLaVA Nüwa supports exactly one image per prompt."
                )
        pact_token_sizes: Any | None = None
        pact_payload = None
        pact_spec = getattr(
            decoder_registration.callback,
            "_mllm_pact_spec",
            None,
        )
        if selection is None and decoder_algorithm == "pact":
            if pact_spec is None:
                raise InvariantError("LLaVA PACT spec was not registered.")
            if len(sequence.image_sizes) != 1 or tuple(sequence.modalities) != (
                "image",
            ):
                raise BackendIntegrationError(
                    "LLaVA PACT supports exactly one image per prompt so clusters "
                    "cannot cross media boundaries."
                )
        restore_spec = getattr(
            decoder_registration.callback,
            "_mllm_restore_spec",
            None,
        )
        restore_attention = sequence.restore_attention
        restore_layers_calibrated = 0
        if decoder_algorithm == "restore":
            if restore_spec is None:
                raise InvariantError("LLaVA RESTORE spec was not registered.")
            if len(sequence.image_sizes) != 1 or tuple(sequence.modalities) != (
                "image",
            ):
                raise BackendIntegrationError(
                    "LLaVA RESTORE supports exactly one image per prompt."
                )
            if restore_attention is None:
                raise InvariantError(
                    "LLaVA RESTORE decoder lost its prefill position/size state."
                )
            if decoder_call_phase is DecoderCallPhase.DECODE:
                next_position = self._restore_next_position.get()
                if next_position is None:
                    raise InvariantError(
                        "LLaVA RESTORE decode lost its source-position continuation."
                    )
                query_position = next_position + lifecycle.decode_forward_calls - 1
                current_positions = torch.full(
                    (1, int(hidden_states.shape[1])),
                    query_position,
                    dtype=restore_attention.key_positions.dtype,
                    device=hidden_states.device,
                )
                restore_attention = restore_attention.append(
                    current_positions, reference=hidden_states
                )
                sequence.restore_attention = restore_attention
        application_position = (
            getattr(
                decoder_registration.callback,
                "_mllm_application_position",
                LayerPosition(
                    "inter_layer",
                    decoder_registration.prune_layer + int(decoder_algorithm == "pact"),
                ),
            )
            if selection_method is None and decoder_algorithm in {"sgl", "pact"}
            else None
        )
        application_positions = (application_position,) if application_position is not None else ()
        if selection_method is not None:
            application_positions = selection_method.positions
        def apply_pre_layer_compression(layer_index, *, mid=False):
            nonlocal pact_token_sizes, pact_payload, boundary_replay
            nonlocal hidden_states, padding_mask, position_ids, past_key_values, cache_position
            nonlocal causal_masks, position_embeddings
            point = LayerPosition("post_attention_pre_ffn" if mid else "inter_layer", layer_index)
            if decoder_call_phase is DecoderCallPhase.DECODE or point not in application_positions:
                return None
            image_indices = sequence.image_indices.to(hidden_states.device)
            if selection is not None:
                if int(image_indices.numel()) == 0:
                    return None
                response = selection_registration.apply(selection, point, hidden_states[0], sequence=sequence,
                    attention_mask=padding_mask, position_ids=position_ids, past_key_values=past_key_values,
                    use_cache=use_cache, repair_layout=_decoder_repair_callback)
                self._callback_counts[decoder_registration.site] += 1
                self._record_callback_response(decoder_registration.site, sequence.invocation_id, response)
                if response is None:
                    return None
            elif decoder_algorithm == "pact":
                if pact_payload is None:
                    raise InvariantError("PACT fixed metric sources are unavailable at application.")
                scores, query_states, key_states, clustering_keys = pact_payload
                request = LlavaPACTRequest(
                    invocation_id=sequence.invocation_id, layer_index=pact_spec.boundary_layer, num_layers=len(layers),
                    hidden_states=hidden_states[0], image_indices=image_indices,
                    query_states=query_states, key_states=key_states, clustering_keys=clustering_keys, image_attention_scores=scores,
                    attention_mask=padding_mask, position_ids=position_ids, past_key_values=past_key_values,
                    use_cache=use_cache, image_sizes=sequence.image_sizes, modalities=sequence.modalities,
                    newline_indices=sequence.newline_indices, repair_layout=_decoder_repair_callback)
                pact_payload = None
            elif decoder_algorithm == "sgl":
                request = LlavaFastVRequest(
                    invocation_id=sequence.invocation_id,
                    prune_layer=layer_index,
                    num_layers=len(layers),
                    hidden_states=hidden_states[0],
                    image_indices=image_indices,
                    scorer_query=None,
                    attention_keys=None,
                    attention_key_mask=None,
                    attention_mask=padding_mask,
                    position_ids=position_ids,
                    past_key_values=past_key_values,
                    use_cache=use_cache,
                    image_sizes=sequence.image_sizes,
                    modalities=sequence.modalities,
                    newline_indices=sequence.newline_indices,
                    repair_layout=_decoder_repair_callback,
                )
            if selection is None:
                response = self._dispatch(PYRAMID_DROP_SITE if decoder_algorithm == "pact" else FASTV_SITE, request)
            if not isinstance(response, LlavaDecoderLayerResponse) or (
                response.layout_delta is None
            ):
                raise BackendIntegrationError(
                    f"{decoder_algorithm} decoder callback returned an invalid response."
                )
            if selection is None and decoder_algorithm == "pact":
                pact_token_sizes = response.token_sizes
            boundary_replay = decoder_layout_replay(response.layout_delta,
                rules=getattr(decoder_registration.callback, "_mllm_rules", {}))
            lifecycle.record_boundary(
                layer_index=min(layer_index, len(layers) - 1),
                boundary=DecoderBoundary.POST_ATTENTION_PRE_FFN if mid else (DecoderBoundary.AFTER_LAYER if layer_index == len(layers) else DecoderBoundary.BEFORE_LAYER),
                layout_delta=response.layout_delta,
                past_key_values=past_key_values,
                cache_policy=selection_method.cache_policy if selection_method is not None else CacheRepairPolicy.LAYER_LOCAL if decoder_algorithm == "pact" else CacheRepairPolicy.COMPACT_PREFIX,
                prefix_layer_count=layer_index + int(mid) if use_cache else 0,
            )
            hidden_states = response.hidden_states.unsqueeze(0)
            from mllm_comp.fixed_mask.runtime import KEY, compact_layout
            if call.get(KEY) is not None:
                compact_layout(call, torch.tensor(layout_result_representatives(response.layout_delta),
                                                 device=hidden_states.device, dtype=torch.long))
            padding_mask = response.attention_mask
            position_ids = response.position_ids
            past_key_values = response.past_key_values
            sequence.image_indices = response.image_indices
            sequence.newline_indices = response.newline_indices
            cache_position = torch.arange(
                hidden_states.shape[1], device=hidden_states.device
            )
            causal_masks = _decoder_masks(
                self._components.core,
                hidden_states,
                padding_mask,
                position_ids,
                cache_position,
                create_causal_mask,
                create_sliding_window_causal_mask,
            )
            position_embeddings = self._components.core.rotary_emb(
                hidden_states, position_ids
            )
            return torch.as_tensor(response.layout_delta.kept_indices, device=hidden_states.device)
        def apply_post_attention(value, layer_index):
            nonlocal hidden_states
            hidden_states = value
            kept = apply_pre_layer_compression(layer_index, mid=True)
            return hidden_states, boundary_replay if kept is not None else lambda values: values
        def read_selection_attention(capture, image_indices):
            metric_layer = layers[capture.projection_layer]
            query, image_keys, attention_keys = _decoder_query_keys(metric_layer, hidden_states,
                instruction_indices=capture.query_indices, image_indices=image_indices,
                position_embeddings=position_embeddings, use_rope=capture.use_rope)
            visibility = pyramid_attention_key_visibility(causal_masks[getattr(metric_layer, "attention_type", "full_attention")],
                instruction_indices=capture.query_indices, key_count=int(attention_keys.shape[0]), padding_mask=padding_mask)
            return query, image_keys, attention_keys, visibility

        from mllm_comp.compressors.comparison.native_decoder import observe_comparison_input, layer_context
        for layer_index, decoder_layer in enumerate(layers):
            if selection is not None and decoder_call_phase is not DecoderCallPhase.DECODE:
                observe_comparison_input(selection, layer_index, hidden_states, layers,
                    attention_mask=causal_masks[getattr(decoder_layer, "attention_type", "full_attention")],
                    position_ids=position_ids, position_embeddings=position_embeddings,
                    cache_position=cache_position)
            # Earlier stages at this entry run before a later stage samples this layer's input.
            apply_before_capture = selection.select_before_input(layer_index) if selection is not None else False
            if apply_before_capture:
                apply_pre_layer_compression(layer_index)
            if selection is not None and decoder_call_phase is not DecoderCallPhase.DECODE:
                image_indices = sequence.image_indices.to(hidden_states.device)
                image_row = torch.zeros(hidden_states.shape[1], dtype=torch.bool, device=hidden_states.device)
                image_row[image_indices] = True
                capture = selection.capture_input(layer_index, image_row,
                    padding_mask=padding_mask, labels=self._training_labels.get())
                capture_decoder_attention(selection, capture, image_indices, read_selection_attention)
            if not apply_before_capture:
                apply_pre_layer_compression(layer_index)
            layer_position_ids = position_ids
            layer_cache_position = cache_position
            if decoder_call_phase is DecoderCallPhase.DECODE:
                cache_length = dynamic_cache_layer_length(past_key_values, layer_index)
                layer_cache_position = torch.arange(
                    cache_length,
                    cache_length + int(hidden_states.shape[1]),
                    device=hidden_states.device,
                )
                position_dtype = getattr(position_ids, "dtype", torch.long)
                if decoder_algorithm in {"nuwa", "restore"}:
                    next_position = (
                        self._nuwa_next_position.get()
                        if decoder_algorithm == "nuwa"
                        else self._restore_next_position.get()
                    )
                    if next_position is None:
                        raise InvariantError(
                            f"LLaVA {decoder_algorithm} decode lost its source-position continuation."
                        )
                    original_position = (
                        next_position + lifecycle.decode_forward_calls - 1
                    )
                    layer_position_ids = torch.full(
                        (1, int(hidden_states.shape[1])),
                        original_position,
                        dtype=position_dtype,
                        device=hidden_states.device,
                    )
                elif (selection_method is not None and selection_method.position_policy == "sparse_source") or decoder_algorithm in {
                    "sgl",
                }:
                    original_position = (
                        lifecycle.source_length + lifecycle.decode_forward_calls - 1
                    )
                    layer_position_ids = torch.full(
                        (1, int(hidden_states.shape[1])),
                        original_position,
                        dtype=position_dtype,
                        device=hidden_states.device,
                    )
                else:
                    layer_position_ids = layer_cache_position.to(
                        dtype=position_dtype
                    ).unsqueeze(0)
                if (selection_method is not None
                        and selection_method.position_policy == "contiguous"
                        and selection_method.cache_policy is CacheRepairPolicy.LAYER_LOCAL):
                    lifecycle.record_layer_local_decode_position(
                        layer_index=layer_index,
                        cache_position=layer_cache_position,
                        text_position_ids=layer_position_ids,
                        rotary_position_ids=layer_position_ids,
                        expected_rotary_position_ids=layer_position_ids,
                        rotary_policy="contiguous_physical",
                    )
                local_padding = _layer_local_decode_padding_mask(
                    padding_mask,
                    past_key_values,
                    layer_index=layer_index,
                    query_length=int(hidden_states.shape[1]),
                    reference=hidden_states,
                )
                causal_masks = _decoder_masks(
                    self._components.core,
                    hidden_states,
                    local_padding,
                    layer_position_ids,
                    layer_cache_position,
                    create_causal_mask,
                    create_sliding_window_causal_mask,
                    past_key_values=LayerLocalCacheView(past_key_values, layer_index),
                )
                position_embeddings = self._components.core.rotary_emb(
                    hidden_states, layer_position_ids
                )
            attention_type = getattr(decoder_layer, "attention_type", "full_attention")
            if attention_type not in causal_masks:
                raise BackendIntegrationError(
                    f"Pinned Qwen2 decoder returned no {attention_type!r} causal mask."
                )
            assert position_embeddings is not None
            layer_input_hidden = hidden_states
            layer_input_position_embeddings = position_embeddings
            layer_attention_mask = causal_masks[attention_type]
            if decoder_algorithm == "restore":
                assert restore_spec is not None
                assert restore_attention is not None
                layer_attention_mask = restore_attention.attention_mask(
                    layer_attention_mask,
                    layer_position_ids,
                    self._components.core.rotary_emb.inv_freq,
                    layer_index=layer_index,
                    reference=hidden_states,
                    calibration_constant=restore_spec.calibration_constant,
                    rules=getattr(decoder_registration.callback, "_mllm_rules", {}),
                )
                restore_layers_calibrated += 1
            if selection is not None:
                layer_attention_mask = selection.prepare_attention_mask(layer_attention_mask,
                    query_length=int(hidden_states.shape[1]), reference=hidden_states)
            elif (
                decoder_algorithm == "pact"
                and pact_token_sizes is not None
                and pact_spec.proportional_attention
            ):
                layer_attention_mask = _pact_attention_mask(
                    layer_attention_mask,
                    pact_token_sizes,
                    query_length=int(hidden_states.shape[1]),
                    reference=hidden_states,
                )
            capture_context = nullcontext(None)
            if (selection is not None and decoder_call_phase is not DecoderCallPhase.DECODE
                    and layer_index in selection.transition_layers):
                image_indices = sequence.image_indices.to(hidden_states.device)
                if selection.begin_transition(layer_index, image_indices):
                    capture_context = capture_decoder_module_transitions(decoder_layer)
            with capture_context as transition_observation:
                layer_forward = decoder_layer
                if (decoder_call_phase is not DecoderCallPhase.DECODE
                        and LayerPosition("post_attention_pre_ffn", layer_index) in application_positions):
                    layer_forward = partial(decoder_post_attention_boundary, decoder_layer,
                        apply_compression=partial(apply_post_attention, layer_index=layer_index))
                layer_output = layer_forward(
                    hidden_states,
                    attention_mask=layer_attention_mask,
                    position_ids=layer_position_ids,
                    past_key_values=past_key_values,
                    use_cache=use_cache,
                    cache_position=layer_cache_position,
                    position_embeddings=position_embeddings,
                    **call,
                    **layer_context(selection),
                )
            hidden_states = (
                layer_output[0] if isinstance(layer_output, tuple) else layer_output
            )
            if decoder_call_phase is DecoderCallPhase.DECODE:
                continue
            if decoder_algorithm == "restore":
                continue
            image_indices = sequence.image_indices.to(hidden_states.device)
            image_mask = torch.zeros(
                hidden_states.shape[1], dtype=torch.bool, device=hidden_states.device
            )
            image_mask[image_indices] = True
            if transition_observation is not None:
                selection.record_transition(transition_observation, image_indices=image_indices)
                transition_observation = None
            if selection is not None:
                if selection.output_hidden_layer == layer_index:
                    selection.record_output_hidden(layer_index, self._components.core.norm(hidden_states)[0])
                if layer_index in selection.output_feature_layers:
                    selection.record_output_features(layer_index, hidden_states[0],
                        decoder_layer=decoder_layer, position_embeddings=layer_input_position_embeddings,
                        image_mask=image_mask, padding_mask=padding_mask)
                if layer_index in selection.output_projection_layers:
                    projection_layer = selection.output_projection_layers[layer_index]
                    features = _decoder_projection_features(layers[projection_layer], hidden_states,
                        position_embeddings=position_embeddings)
                    selection.record_output_projection(layer_index, hidden_states[0], features=features,
                        image_indices=image_indices, protected_indices=sequence.newline_indices)
                capture = selection.capture_output(layer_index, image_mask,
                    padding_mask=padding_mask, labels=self._training_labels.get())
                capture_decoder_attention(selection, capture, image_indices, read_selection_attention)
                continue
            if decoder_algorithm == "sgl":
                continue
            if decoder_algorithm == "pact":
                if layer_index != decoder_registration.prune_layer:
                    continue
                metric_layer = layers[layer_index + 1]
                query_states, key_states, clustering_keys = _decoder_projection_features(
                    metric_layer,
                    hidden_states,
                    position_embeddings=position_embeddings,
                )
                source_images = sequence.image_indices.to(hidden_states.device)
                source_images = source_images[~torch.isin(source_images, torch.tensor(sequence.newline_indices, device=source_images.device))]
                scores = pact_token_scores(query_states, key_states, source_images,
                    hidden_norms=hidden_states[0].detach().float().norm(dim=-1) if pact_spec.multiply_by_hidden_norm else None, rules=getattr(decoder_registration.callback, "_mllm_rules", {}),)
                pact_payload = (scores, query_states, key_states, clustering_keys)
                continue
        apply_pre_layer_compression(len(layers))
        hidden_states = self._components.core.norm(hidden_states)
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
                "phase": decoder_call_phase.value,
                "layers_calibrated": restore_layers_calibrated,
                "query_tokens": int(hidden_states.shape[1]),
                "key_tokens": int(restore_attention.key_sizes.shape[-1]),
                "visual_source_tokens": sequence.restore_source_visual_count,
                "first_layer_size": restore_spec.first_layer_size,
                "position_policy": "real_packed_source_representative",
                "cache_transform": "none",
            }
        return output

    def _encoder_forward_replacement(self) -> Callable[..., Any]:
        encoder = self._components.encoder
        original = encoder.forward
        signature = inspect.signature(original)

        def replacement(_encoder: Any, *args: Any, **kwargs: Any) -> Any:
            if TOME_RESTORE_SITE not in self._callbacks:
                return original(*args, **kwargs)
            root = self._root_execution.get()
            if root is None:
                raise BackendIntegrationError(
                    "Compressed SigLip encoder was called outside guarded "
                    "LlavaQwenForCausalLM.forward."
                )
            bound = signature.bind_partial(*args, **kwargs)
            inputs = bound.arguments.get("inputs_embeds")
            if inputs is None:
                raise BackendIntegrationError(
                    "SigLip encoder inputs_embeds is required."
                )
            if getattr(inputs, "ndim", None) != 3:
                raise BackendIntegrationError(
                    "Pinned SigLip encoder requires [batch, tokens, channels]."
                )
            resolved_return_dict = bound.arguments.get("return_dict")
            if resolved_return_dict is None:
                resolved_return_dict = getattr(encoder.config, "use_return_dict", None)
            if resolved_return_dict is not True:
                raise BackendIntegrationError(
                    "Pinned ToMe dense restore requires SigLip return_dict=True."
                )
            if bool(getattr(encoder, "gradient_checkpointing", False)) and bool(
                getattr(encoder, "training", False)
            ):
                raise BackendIntegrationError(
                    "Pinned ToMe callback state is not safe with vision gradient "
                    "checkpointing enabled."
                )
            batch_size, token_count = int(inputs.shape[0]), int(inputs.shape[1])
            self._encoder_counter += 1
            invocation_id = f"{root.invocation_id}/vision-{self._encoder_counter}"
            state = _EncoderExecution(
                invocation_id=invocation_id,
                stream_ids=tuple(
                    f"{invocation_id}/crop-{index}" for index in range(batch_size)
                ),
                coordinates=_grid_coordinates(token_count),
                positions=tome_encoder_positions(
                    getattr(getattr(self._callbacks.get(TOME_MERGE_SITE), "callback", None), "_mllm_position_stage", None),
                    "llava_onevision", len(self._components.layers),
                ),
                score_layer=tome_encoder_score_layer(
                    getattr(getattr(self._callbacks.get(TOME_MERGE_SITE), "callback", None), "_mllm_position_stage", None),
                    "llava_onevision", len(self._components.layers),
                    config=getattr(getattr(self._callbacks.get(TOME_MERGE_SITE), "callback", None), "_mllm_position_config", None),
                ),
            )
            context_token = self._encoder_execution.set(state)
            try:
                output = original(*args, **kwargs)
            finally:
                self._encoder_execution.reset(context_token)
            last_hidden = getattr(output, "last_hidden_state", None)
            if last_hidden is None:
                raise BackendIntegrationError(
                    "Pinned SigLip encoder did not return last_hidden_state."
                )
            response = self._dispatch(
                TOME_RESTORE_SITE,
                LlavaEncoderDenseRestoreRequest(
                    invocation_id=invocation_id,
                    hidden_states=last_hidden,
                    stream_ids=state.stream_ids,
                ),
            )
            if not isinstance(response, LlavaEncoderDenseRestoreResponse):
                raise BackendIntegrationError(
                    "ToMe dense callback returned the wrong response type."
                )
            return _replace_encoder_output(output, response.hidden_states)

        return replacement

    def _layer_forward_replacement(self, layer_index: int) -> Callable[..., Any]:
        layer = self._components.layers[layer_index]
        original = layer.forward

        def replacement(
            layer: Any,
            hidden_states: Any,
            attention_mask: Any = None,
            output_attentions: bool | None = False,
        ) -> tuple[Any, ...]:
            import torch

            if TOME_MERGE_SITE not in self._callbacks:
                return original(
                    hidden_states,
                    attention_mask=attention_mask,
                    output_attentions=output_attentions,
                )
            state = self._encoder_execution.get()
            if state is None:
                raise BackendIntegrationError(
                    "Compressed SigLip layer was called outside encoder context."
                )
            if getattr(hidden_states, "ndim", None) != 3:
                raise BackendIntegrationError(
                    "SigLip layer hidden_states must be rank 3."
                )
            def keys_for(normalized: Any, source_layer: Any = layer) -> Any:
                with torch.no_grad():
                    projected = source_layer.self_attn.k_proj(normalized)
                    return projected.view(
                        int(normalized.shape[0]), int(normalized.shape[1]),
                        int(source_layer.self_attn.num_heads), int(source_layer.self_attn.head_dim),
                    )

            def merge(value: Any, keys: Any, position: LayerPosition) -> Any:
                response = self._dispatch(
                    TOME_MERGE_SITE,
                    LlavaEncoderMergeRequest(
                        invocation_id=state.invocation_id, hidden_states=value,
                        keys=state.score_keys if state.score_keys is not None else keys,
                        layer_index=layer_index, stream_ids=state.stream_ids,
                        token_sizes=state.token_sizes, coordinates=state.coordinates,
                        position=position,
                    ),
                )
                if not isinstance(response, LlavaEncoderMergeResponse):
                    raise BackendIntegrationError("ToMe merge callback returned the wrong response type.")
                if state.score_keys is not None:
                    state.score_keys = torch.stack([
                        tome_merge_reference(key, delta, token_sizes=None if state.token_sizes is None else state.token_sizes[i])
                        for i, (key, delta) in enumerate(zip(state.score_keys, response.layout_deltas))
                    ])
                state.token_sizes = response.token_sizes
                state.proportional_attention_bias = response.proportional_attention_bias
                state.coordinates = ()
                return response.hidden_states

            normalized = layer.layer_norm1(hidden_states)
            keys = keys_for(normalized)
            if state.score_layer == layer_index and state.score_keys is None:
                state.score_keys = keys.detach()
            initial = LayerPosition("inter_layer", 0)
            if layer_index == 0 and initial in state.positions:
                hidden_states = merge(hidden_states, keys, initial)
                normalized = layer.layer_norm1(hidden_states)
                keys = keys_for(normalized)
            effective_mask = _add_proportional_attention_bias(
                attention_mask, state.proportional_attention_bias, hidden_states=normalized,
            )
            attended, attention_weights = layer.self_attn(
                hidden_states=normalized, attention_mask=effective_mask,
                output_attentions=output_attentions,
            )
            hidden_states = hidden_states + attended
            mid = LayerPosition("post_attention_pre_ffn", layer_index)
            full = LayerPosition("inter_layer", layer_index + 1)
            if mid in state.positions:
                hidden_states = merge(hidden_states, keys, mid)
            hidden_states = hidden_states + layer.mlp(layer.layer_norm2(hidden_states))
            if full in state.positions:
                if state.score_layer == layer_index + 1 and state.score_keys is None:
                    source_layer = self._components.layers[layer_index + 1]
                    state.score_keys = keys_for(source_layer.layer_norm1(hidden_states), source_layer).detach()
                hidden_states = merge(hidden_states, keys, full)
            outputs: tuple[Any, ...] = (hidden_states,)
            if output_attentions:
                outputs += (attention_weights,)
            return outputs

        return replacement

    def _dispatch(self, site: str, request: Any) -> Any:
        registered = self._callbacks.get(site)
        if registered is None:
            raise BackendIntegrationError(
                f"Pinned LLaVA provider callback {site!r} is not registered."
            )
        callback = registered.callback
        self._callback_counts[site] += 1
        response = callback(request)
        return self._record_callback_response(site, getattr(request, "invocation_id", None), response)

    def _record_callback_response(self, site, invocation_id, response):
        deltas = getattr(response, "layout_deltas", None)
        if deltas is None:
            delta = getattr(response, "layout_delta", None)
            deltas = () if delta is None else (delta,)
        if deltas:
            current = tuple(deltas)
            layout_site = TOME_MERGE_SITE if site == TOME_RESTORE_SITE else site
            layout_key = (layout_site, invocation_id)
            prior = self._layout_by_invocation.get(layout_key)
            if prior is not None:
                if len(prior) != len(current):
                    raise BackendIntegrationError(
                        "Provider LayoutDelta batch cardinality changed inside an invocation."
                    )
                current = tuple(
                    before.compose(after) for before, after in zip(prior, current)
                )
            if invocation_id is not None:
                self._layout_by_invocation[layout_key] = current
            self._latest_layout_deltas = [
                _layout_delta_record(delta) for delta in current
            ]
        elif site == TOME_RESTORE_SITE and invocation_id is not None:
            completed = self._layout_by_invocation.pop(
                (TOME_MERGE_SITE, invocation_id), None
            )
            if completed is not None:
                self._latest_layout_deltas = [
                    _layout_delta_record(delta) for delta in completed
                ]
        return response


def install_llava_onevision_ee14dbd_provider_bridge(
    model: Any,
) -> PinnedLlavaOneVisionEe14dbdBridge:
    """Attach or return the exact pinned provider bridge on one model instance."""

    for name in ("mllm_compression_bridge", "_mllm_compression_bridge"):
        existing = getattr(model, name, None)
        if existing is not None:
            if isinstance(existing, PinnedLlavaOneVisionEe14dbdBridge):
                return existing
            raise BackendIntegrationError(
                f"Model already exposes an incompatible compression bridge at {name}."
            )
    bridge = PinnedLlavaOneVisionEe14dbdBridge(model)
    setattr(model, "mllm_compression_bridge", bridge)
    bridge._attached_attribute = "mllm_compression_bridge"
    return bridge


def _resolve_components(model: Any) -> _ProviderComponents:
    get_model = getattr(model, "get_model", None)
    get_vision = getattr(model, "get_vision_tower", None)
    if not callable(get_model) or not callable(get_vision):
        raise BackendIntegrationError(
            "Pinned LLaVA bridge requires get_model/get_vision_tower."
        )
    core = get_model()
    vision_tower = get_vision()
    inner = getattr(vision_tower, "vision_tower", None)
    vision_model = getattr(inner, "vision_model", None)
    encoder = getattr(vision_model, "encoder", None)
    layers = tuple(getattr(encoder, "layers", ()) or ())
    if core is None or vision_tower is None or encoder is None or not layers:
        raise BackendIntegrationError(
            "Pinned LLaVA bridge could not resolve the SigLip encoder/layers."
        )
    return _ProviderComponents(model, core, vision_tower, encoder, layers)


def _verify_pinned_contract(model: Any, components: _ProviderComponents) -> None:
    try:
        installed_transformers = importlib.metadata.version("transformers")
    except importlib.metadata.PackageNotFoundError as exc:
        raise BackendIntegrationError(
            "Pinned LLaVA provider requires transformers==4.57.3."
        ) from exc
    if installed_transformers != SUPPORTED_TRANSFORMERS_VERSION:
        raise BackendIntegrationError(
            "Pinned LLaVA provider version guard failed: "
            f"expected={SUPPORTED_TRANSFORMERS_VERSION}, actual={installed_transformers}."
        )
    expected_classes = (
        (model, "llava.model.language_model.llava_qwen", "LlavaQwenForCausalLM"),
        (components.core, "llava.model.language_model.llava_qwen", "LlavaQwenModel"),
        (
            components.vision_tower,
            "llava.model.multimodal_encoder.siglip_encoder",
            "SigLipVisionTower",
        ),
        (
            components.encoder,
            "llava.model.multimodal_encoder.siglip_encoder",
            "SigLipEncoder",
        ),
    )
    expected_classes += tuple(
        (
            layer,
            "llava.model.multimodal_encoder.siglip_encoder",
            "SigLipEncoderLayer",
        )
        for layer in components.layers
    )
    for value, module_name, class_name in expected_classes:
        actual = (type(value).__module__, type(value).__name__)
        if actual != (module_name, class_name):
            raise BackendIntegrationError(
                "Pinned LLaVA class guard failed: "
                f"expected={module_name}.{class_name}, actual={actual[0]}.{actual[1]}."
            )
    if getattr(getattr(model, "config", None), "model_type", None) != "llava_qwen":
        raise BackendIntegrationError(
            "Pinned LLaVA provider requires config.model_type='llava_qwen'."
        )
    signature_targets = {
        "LlavaQwenForCausalLM.forward": type(model).forward,
        "LlavaQwenForCausalLM.generate": type(model).generate,
        "LlavaMetaForCausalLM.prepare_inputs_labels_for_multimodal": (
            type(model).prepare_inputs_labels_for_multimodal
        ),
        "Qwen2Model.forward": type(components.core).forward,
        "SigLipEncoder.forward": type(components.encoder).forward,
        "SigLipEncoderLayer.forward": type(components.layers[0]).forward,
    }
    for name, target in signature_targets.items():
        actual_names = tuple(inspect.signature(target).parameters)
        if actual_names != _EXPECTED_SIGNATURES[name]:
            raise BackendIntegrationError(
                f"Pinned LLaVA signature guard failed for {name}: "
                f"expected={_EXPECTED_SIGNATURES[name]}, actual={actual_names}."
            )
    source_root = (
        Path(__file__).resolve().parents[3] / "third_party" / "backbones" / "LLaVA-NeXT"
    )
    for relative_path, expected_hash in _NORMALIZED_SOURCE_HASHES.items():
        path = source_root / relative_path
        if not path.is_file():
            raise BackendIntegrationError(
                f"Pinned LLaVA source file is missing: {path}."
            )
        normalized = path.read_text(encoding="utf-8").replace("\r\n", "\n")
        actual_hash = hashlib.sha256(normalized.encode("utf-8")).hexdigest()
        if actual_hash != expected_hash:
            raise BackendIntegrationError(
                "Pinned LLaVA source hash guard failed for "
                f"{relative_path}: expected={expected_hash}, actual={actual_hash}."
            )


def _normalise_single_image_modalities(value: Any) -> tuple[str, ...]:
    if value is None:
        modalities = ("image",)
    elif isinstance(value, str):
        modalities = (value,)
    else:
        modalities = tuple(str(item) for item in value)
    if modalities != ("image",):
        raise BackendIntegrationError(
            "Pinned LLaVA packed-image compression requires exactly one image modality."
        )
    return modalities


def _normalise_single_image_sizes(value: Any) -> tuple[tuple[int, int], ...]:
    tolist = getattr(value, "tolist", None)
    raw = tolist() if callable(tolist) else value
    if not isinstance(raw, (list, tuple)) or len(raw) != 1:
        raise BackendIntegrationError(
            "Pinned LLaVA packed-image compression requires exactly one image_sizes entry."
        )
    pair_value = raw[0]
    pair_tolist = getattr(pair_value, "tolist", None)
    pair = pair_tolist() if callable(pair_tolist) else pair_value
    if (
        not isinstance(pair, (list, tuple))
        or len(pair) != 2
        or any(
            isinstance(item, bool) or not isinstance(item, int) or item <= 0
            for item in pair
        )
    ):
        raise BackendIntegrationError(
            "Pinned LLaVA image_sizes must contain one positive (width, height) pair."
        )
    return ((int(pair[0]), int(pair[1])),)


def _active_prompt_ids(input_ids: Any, attention_mask: Any) -> Any:
    if attention_mask is None:
        return input_ids[0]
    if getattr(attention_mask, "ndim", None) != 2 or tuple(
        attention_mask.shape
    ) != tuple(input_ids.shape):
        raise BackendIntegrationError(
            "Pinned LLaVA prompt attention_mask must match input_ids."
        )
    return input_ids[0][attention_mask[0].bool()]


def _image_newline_indices(tokens: Any, image_newline: Any) -> tuple[int, ...]:
    torch = _torch()
    if image_newline is None:
        return ()
    vector = image_newline
    if getattr(vector, "ndim", None) == 2 and int(vector.shape[0]) == 1:
        vector = vector[0]
    if getattr(vector, "ndim", None) != 1 or int(vector.shape[0]) != int(
        tokens.shape[-1]
    ):
        raise BackendIntegrationError(
            "Pinned LLaVA image_newline shape differs from packed image features."
        )
    matches = torch.isclose(
        tokens.detach().float(),
        vector.detach().to(tokens.device).float().unsqueeze(0),
        rtol=1e-5,
        atol=1e-6,
    ).all(dim=-1)
    return tuple(
        int(item) for item in matches.nonzero(as_tuple=False).flatten().tolist()
    )


def _packed_qk_metrics(decoder_layer: Any, tokens: Any) -> tuple[Any, Any]:
    torch = _torch()
    hidden = tokens.unsqueeze(0)
    normalised = decoder_layer.input_layernorm(hidden)
    attention = decoder_layer.self_attn
    token_count = int(tokens.shape[0])
    head_dim = int(attention.head_dim)
    query = attention.q_proj(normalised).view(1, token_count, -1, head_dim)[0]
    key = attention.k_proj(normalised).view(1, token_count, -1, head_dim)[0]
    groups = int(getattr(attention, "num_key_value_groups", 1))
    if groups > 1:
        key = torch.repeat_interleave(key, groups, dim=1)
    query_by_head = query.permute(1, 0, 2)
    key_by_head = key.permute(1, 0, 2)
    scale = float(getattr(attention, "scaling", head_dim**-0.5))
    scores = (
        torch.matmul(query_by_head.float(), key_by_head.float().transpose(-1, -2))
        * scale
    )
    weights = torch.softmax(scores, dim=-1, dtype=torch.float32)
    return weights, key_by_head


def _replace_packed_image_tokens(
    embeddings: Any,
    *,
    image_start: int,
    image_end: int,
    tokens: Any,
) -> Any:
    torch = _torch()
    if getattr(tokens, "ndim", None) != 2 or int(tokens.shape[-1]) != int(
        embeddings.shape[-1]
    ):
        raise BackendIntegrationError(
            "Packed-image compression returned tokens with an incompatible shape."
        )
    return torch.cat(
        (embeddings[:, :image_start], tokens.unsqueeze(0), embeddings[:, image_end:]),
        dim=1,
    )


def _layout_sequence_representatives(
    delta: Any,
    *,
    image_start: int,
    image_end: int,
    sequence_length: int,
    device: Any,
) -> Any:
    torch = _torch()
    explicit = delta.result_layout.get("representative_indices")
    if explicit is not None:
        values = tuple(int(value) for value in explicit)
        if (
            len(values) != int(delta.result_count)
            or len(set(values)) != len(values)
            or any(value < 0 or value >= int(delta.source_count) for value in values)
        ):
            raise BackendIntegrationError(
                "Packed-image representative_indices are not valid source positions."
            )
        for result_index, source in enumerate(values):
            if delta.old_to_new[source] != result_index:
                raise BackendIntegrationError(
                    "Packed-image representative does not belong to its result group."
                )
        representatives = [image_start + source for source in values]
        result = [
            *range(image_start),
            *representatives,
            *range(image_end, sequence_length),
        ]
        return torch.tensor(result, dtype=torch.long, device=device)
    representatives: list[int] = []
    for result_index in range(int(delta.result_count)):
        sources = [
            source
            for source in range(int(delta.source_count))
            if delta.old_to_new[source] == result_index
        ]
        if not sources:
            raise BackendIntegrationError(
                "Packed-image LayoutDelta result token has no source representative."
            )
        representatives.append(image_start + sources[0])
    values = [*range(image_start), *representatives, *range(image_end, sequence_length)]
    return torch.tensor(values, dtype=torch.long, device=device)


def _select_sequence_axis(value: Any, indices: Any) -> Any:
    if value is None:
        return None
    if not hasattr(value, "index_select") or int(value.shape[-1]) <= int(
        indices.max().item()
    ):
        raise BackendIntegrationError(
            "Packed-image compression cannot compact this mask/position representation."
        )
    return value.index_select(-1, indices.to(value.device))


def _bind_decoder_call(
    signature: inspect.Signature,
    args: tuple[Any, ...],
    kwargs: Mapping[str, Any],
) -> dict[str, Any]:
    bound = signature.bind_partial(*args, **kwargs)
    values = dict(bound.arguments)
    extras = values.pop("kwargs", {})
    if isinstance(extras, Mapping):
        values.update(extras)
    defaults = {
        "input_ids": None,
        "attention_mask": None,
        "position_ids": None,
        "past_key_values": None,
        "inputs_embeds": None,
        "use_cache": None,
        "cache_position": None,
    }
    return {**defaults, **values}


def _callback_execution_phase(
    callback: Callable[[Any], Any],
) -> ExecutionPhase | None:
    value = getattr(callback, "_mllm_execution_phase", None)
    if value is None:
        return None
    try:
        return ExecutionPhase(value)
    except (TypeError, ValueError) as exc:
        raise BackendIntegrationError(
            "LLaVA provider callback declared an unknown execution phase."
        ) from exc


def _callback_algorithm(callback: Callable[[Any], Any]) -> str | None:
    value = getattr(callback, "_mllm_algorithm", None)
    if value is None:
        return None
    if not isinstance(value, str) or not value:
        raise BackendIntegrationError(
            "LLaVA provider callback declared an invalid algorithm."
        )
    return value


def _callback_activation_id(callback: Callable[[Any], Any]) -> int | None:
    value = getattr(callback, "_mllm_activation_id", None)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise BackendIntegrationError(
            "LLaVA provider callback declared an invalid activation identity."
        )
    return value


def _callback_prune_layer(callback: Callable[[Any], Any]) -> int | None:
    value = getattr(callback, "_mllm_prune_layer", None)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise BackendIntegrationError(
            "LLaVA decoder callback declared an invalid prune layer."
        )
    return value


def _callback_prune_layers(callback: Callable[[Any], Any]) -> tuple[int, ...]:
    value = getattr(callback, "_mllm_prune_layers", ())
    if isinstance(value, (str, bytes)):
        raise BackendIntegrationError(
            "LLaVA decoder callback declared invalid prune layers."
        )
    try:
        result = tuple(value)
    except TypeError as exc:
        raise BackendIntegrationError(
            "LLaVA decoder callback declared invalid prune layers."
        ) from exc
    if any(
        isinstance(item, bool) or not isinstance(item, int) or item < 0
        for item in result
    ):
        raise BackendIntegrationError(
            "LLaVA decoder callback declared invalid prune layers."
        )
    return result


def _callback_score_layers(callback: Callable[[Any], Any]) -> tuple[int, ...]:
    # Bare public callbacks retain their original prune_layer-derived source.
    default = ()
    if _callback_algorithm(callback) in {"fastv", "nuwa"}:
        prune_layer = _callback_prune_layer(callback)
        default = (prune_layer - 1,) if prune_layer else ()
    value = getattr(callback, "_mllm_score_layers", default)
    if isinstance(value, (str, bytes)):
        raise BackendIntegrationError(
            "LLaVA decoder callback declared invalid score layers."
        )
    try:
        result = tuple(value)
    except TypeError as exc:
        raise BackendIntegrationError(
            "LLaVA decoder callback declared invalid score layers."
        ) from exc
    if any(
        isinstance(item, bool) or not isinstance(item, int) or item < 0
        for item in result
    ):
        raise BackendIntegrationError(
            "LLaVA decoder callback declared invalid score layers."
        )
    return result


def _callback_score_source(callback: Callable[[Any], Any]) -> str | None:
    value = getattr(callback, "_mllm_score_source", None)
    if value is None:
        return None
    if value not in {"last_token", "mean_query"}:
        raise BackendIntegrationError(
            "LLaVA FastV callback declared an invalid score source."
        )
    return str(value)


def _generate_registration(
    registrations: tuple[_CallbackRegistration, ...],
) -> _CallbackRegistration | None:
    if len(registrations) == 2:
        sites = {item.site for item in registrations}
        activation_ids = {item.activation_id for item in registrations}
        algorithms = {item.algorithm for item in registrations}
        if (
            (
                sites == {VISIONZIP_SITE, FASTV_SITE} and algorithms == {"nuwa"}
            )
            or (
                sites == {VISIONZIP_SITE, RESTORE_CALIBRATION_SITE}
                and algorithms == {"restore"}
            )
        ) and (
            len(activation_ids) == 1
            and None not in activation_ids
            and all(
                item.execution_phase is ExecutionPhase.GENERATE
                for item in registrations
            )
        ):
            return next(
                item
                for item in registrations
                if item.site
                in {FASTV_SITE, PYRAMID_DROP_SITE, RESTORE_CALIBRATION_SITE}
            )
        return None
    if len(registrations) != 1:
        return None
    registration = registrations[0]
    if (
        registration.execution_phase is not ExecutionPhase.GENERATE
        or registration.site not in {PYRAMID_DROP_SITE, FASTV_SITE}
        or (registration.decoder_selection is None and registration.algorithm
        not in {
            "pyramid_drop",
            "fastv",
            "sgl",
            "pact",
        })
    ):
        return None
    return registration


def _decoder_masks(
    core: Any,
    hidden_states: Any,
    padding_mask: Any,
    position_ids: Any,
    cache_position: Any,
    create_causal_mask: Callable[..., Any],
    create_sliding_window_causal_mask: Callable[..., Any],
    *,
    past_key_values: Any | None = None,
) -> dict[str, Any]:
    mask_kwargs = {
        "config": core.config,
        "input_embeds": hidden_states,
        "attention_mask": padding_mask,
        "cache_position": cache_position,
        "past_key_values": past_key_values,
        "position_ids": position_ids,
    }
    masks = {"full_attention": create_causal_mask(**mask_kwargs)}
    if bool(getattr(core, "has_sliding_layers", False)):
        masks["sliding_attention"] = create_sliding_window_causal_mask(**mask_kwargs)
    return masks


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
    length = dynamic_cache_layer_length(past_key_values, layer_index) + query_length
    return _torch().ones(
        (1, length),
        dtype=padding_mask.dtype,
        device=reference.device,
    )


def _new_dynamic_cache(config: Any) -> Any:
    try:
        from transformers.cache_utils import DynamicCache
    except ImportError as exc:  # pragma: no cover - exact version guard covers this
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
        from transformers.models.qwen2.modeling_qwen2 import apply_rotary_pos_emb
    except ImportError as exc:  # pragma: no cover - exact provider guard covers this
        raise BackendIntegrationError(
            "transformers==4.57.3 Qwen2 RoPE helper is unavailable."
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
        query, keys = apply_rotary_pos_emb(query, keys, cos.to(query.device), sin.to(query.device))
    instruction_indices = torch.as_tensor(
        instruction_indices,
        dtype=torch.long,
        device=query.device,
    )
    query = query[0].index_select(1, instruction_indices).transpose(0, 1)
    keys = keys[0].transpose(0, 1)
    groups = int(getattr(attention, "num_key_value_groups", 1))
    if groups > 1:
        keys = torch.repeat_interleave(keys, groups, dim=1)
    return query, keys.index_select(0, image_indices.to(keys.device)), keys


def _decoder_projection_features(
    decoder_layer: Any,
    hidden_states: Any,
    *,
    position_embeddings: tuple[Any, Any],
) -> tuple[Any, Any, Any]:
    """Extract Qwen2 pre-RoPE EUTI Q/K and post-RoPE clustering keys."""

    torch = _torch()
    try:
        from transformers.models.qwen2.modeling_qwen2 import apply_rotary_pos_emb
    except ImportError as exc:  # pragma: no cover - provider guard covers this
        raise BackendIntegrationError(
            "transformers==4.57.3 Qwen2 RoPE helper is unavailable."
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


def _decoder_repair_callback(
    *,
    kept_indices: Any,
    dropped_indices: Any,
    layout_delta: Any,
    hidden_states: Any,
    attention_mask: Any,
    position_ids: Any,
    past_key_values: Any,
    use_cache: bool,
    layer_index: int,
    image_indices: Any,
    newline_indices: tuple[int, ...],
    position_policy: str = "contiguous",
    cache_policy: str = "layer_local",
) -> LlavaDecoderLayoutRepair:
    del dropped_indices, hidden_states
    torch = _torch()
    if use_cache and past_key_values is None:
        raise BackendIntegrationError(
            "Cached LLaVA decoder repair requires its DynamicCache."
        )
    if not use_cache and past_key_values is not None:
        raise BackendIntegrationError(
            "Non-cached LLaVA decoder repair received cache state."
        )
    kept = kept_indices.to(dtype=torch.long)
    compact_mask = attention_mask.index_select(-1, kept.to(attention_mask.device))
    if position_policy == "contiguous":
        compact_positions = _rebuild_text_position_ids(
            compact_mask,
            reference=position_ids,
            sequence_length=int(layout_delta.result_count),
        )
    elif position_policy == "sparse_source":
        compact_positions = position_ids.index_select(-1, kept.to(position_ids.device))
    else:
        raise BackendIntegrationError(
            f"Unknown LLaVA decoder position policy {position_policy!r}."
        )
    if cache_policy not in {"layer_local", "compact_prefix"}:
        raise BackendIntegrationError(
            f"Unknown LLaVA decoder cache policy {cache_policy!r}."
        )
    source_images = (
        image_indices.tolist()
        if hasattr(image_indices, "tolist")
        else list(image_indices)
    )
    repaired_images = torch.tensor(
        [int(layout_delta.old_to_new[int(index)]) for index in source_images],
        dtype=torch.long,
        device=kept.device,
    )
    repaired_newlines = tuple(
        int(layout_delta.old_to_new[int(index)]) for index in newline_indices
    )
    return LlavaDecoderLayoutRepair(
        attention_mask=compact_mask,
        position_ids=compact_positions,
        past_key_values=past_key_values,
        image_indices=repaired_images,
        newline_indices=repaired_newlines,
        sequence_length=int(layout_delta.result_count),
        metadata={
            "provider": "vendored_llava_next_ee14dbd_instance_bridge",
            "layer_index": layer_index,
            "causal_mask_rebuilt_by_provider": True,
            "position_embeddings_rebuilt_by_provider": True,
            "position_ids_reindexed_contiguously": position_policy == "contiguous",
            "position_policy": position_policy,
            "cache_policy": cache_policy,
            "cache_validation_owner": "shared_decoder_lifecycle",
        },
    )


def _rebuild_text_position_ids(
    attention_mask: Any | None,
    *,
    reference: Any | None,
    sequence_length: int,
) -> Any | None:
    if reference is None:
        return None
    torch = _torch()
    if attention_mask is None:
        return torch.arange(
            sequence_length,
            dtype=reference.dtype,
            device=reference.device,
        ).unsqueeze(0)
    if getattr(attention_mask, "ndim", None) != 2 or tuple(
        int(item) for item in attention_mask.shape
    ) != (1, sequence_length):
        raise BackendIntegrationError(
            "LLaVA position-id rebuild requires a batch-one rank-2 padding mask."
        )
    valid = attention_mask.to(reference.device).bool()
    positions = valid.long().cumsum(dim=-1) - 1
    positions.masked_fill_(~valid, 0)
    return positions.to(dtype=reference.dtype)


def _transformers_decoder_helpers() -> (
    tuple[Any, Callable[..., Any], Callable[..., Any]]
):
    try:
        from transformers.masking_utils import (
            create_causal_mask,
            create_sliding_window_causal_mask,
        )
        from transformers.modeling_outputs import BaseModelOutputWithPast
    except ImportError as exc:
        raise BackendIntegrationError(
            "transformers==4.57.3 decoder helpers are unavailable."
        ) from exc
    return (
        BaseModelOutputWithPast,
        create_causal_mask,
        create_sliding_window_causal_mask,
    )


def _torch() -> Any:
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("Pinned LLaVA provider execution requires PyTorch.") from exc
    return torch


def _require_image_only_modalities(modalities: Any) -> None:
    if modalities is None:
        values = ("image",)
    elif isinstance(modalities, str):
        values = (modalities,)
    else:
        values = tuple(str(item) for item in modalities)
    unsupported = [item for item in values if item != "image"]
    if unsupported:
        raise BackendIntegrationError(
            "Pinned LLaVA compression bridge is image-only; "
            f"unsupported modalities={unsupported}."
        )


def _grid_coordinates(token_count: int) -> tuple[Any, ...]:
    side = int(token_count**0.5)
    if side * side == token_count:
        return tuple((index // side, index % side) for index in range(token_count))
    return tuple((index,) for index in range(token_count))


def _add_proportional_attention_bias(
    attention_mask: Any,
    bias: Any,
    *,
    hidden_states: Any,
) -> Any:
    if bias is None:
        return attention_mask
    batch_size, token_count = int(hidden_states.shape[0]), int(hidden_states.shape[1])
    if tuple(int(item) for item in bias.shape) != (
        batch_size,
        1,
        1,
        token_count,
    ):
        raise BackendIntegrationError(
            "ToMe proportional attention bias does not match compact token layout."
        )
    expanded = bias.expand(batch_size, 1, token_count, token_count)
    if attention_mask is None:
        return expanded
    if tuple(int(item) for item in attention_mask.shape) != tuple(expanded.shape):
        raise BackendIntegrationError(
            "SigLip attention mask cannot be combined with ToMe token-size bias."
        )
    return attention_mask + expanded


def _replace_encoder_output(output: Any, restored_hidden_states: Any) -> Any:
    try:
        values = dict(output)
    except Exception as exc:
        raise BackendIntegrationError(
            "Pinned SigLip dense restore requires a Transformers ModelOutput."
        ) from exc
    values["last_hidden_state"] = restored_hidden_states
    hidden_states = values.get("hidden_states")
    if hidden_states is not None:
        hidden_states = tuple(hidden_states)
        if not hidden_states:
            raise BackendIntegrationError(
                "SigLip hidden_states tuple is unexpectedly empty."
            )
        values["hidden_states"] = (*hidden_states[:-1], restored_hidden_states)
    try:
        return type(output)(**values)
    except Exception as exc:
        raise BackendIntegrationError(
            "Pinned SigLip dense restore could not rebuild ModelOutput."
        ) from exc


def _telemetry_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _telemetry_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_telemetry_value(item) for item in value]
    return value


def _layout_delta_record(delta: Any) -> dict[str, Any]:
    return {
        "source_count": int(delta.source_count),
        "result_count": int(delta.result_count),
        "operation": str(delta.operation),
        "media_unit": str(delta.media_unit),
        "ordering_policy": str(delta.ordering_policy.value),
        "kept_indices": list(delta.kept_indices),
        "dropped_indices": list(delta.dropped_indices),
        "merge_groups": [list(group) for group in delta.merge_groups],
        "old_to_new": [delta.old_to_new[index] for index in range(delta.source_count)],
        "protected_indices": list(delta.protected_indices),
        "source_layout": _telemetry_value(delta.source_layout),
        "result_layout": _telemetry_value(delta.result_layout),
    }


__all__ = [
    "LlavaOneVisionProviderCallbackHandle",
    "PinnedLlavaOneVisionEe14dbdBridge",
    "install_llava_onevision_ee14dbd_provider_bridge",
]
