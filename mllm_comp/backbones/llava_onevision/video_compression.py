"""Native LLaVA-OneVision video packing and pre-LLM splice legalization.

The pinned LLaVA source represents one video with one ``IMAGE_TOKEN_INDEX``.
Frames are projected, spatially pooled, optionally separated by ``image_newline``
tokens, flattened, and inserted into the language-model sequence.  Video
compressors normally operate on the projected patch tokens. The explicit
pre-projector position applies the same spatial pooling to raw vision features,
compresses that pooled grid, and projects only its result. Pooling and nonlinear
projection are not assumed to commute. Newlines remain structural throughout.

This module owns the reversible instance bridge at that exact packed boundary.
It deliberately does not implement an algorithm: a registered callback accepts
the shared :class:`VideoTokenPayload` and must return a validated
:class:`VideoCompressionResult`.
"""

from __future__ import annotations

import inspect
import math
from collections.abc import Callable, Mapping, Sequence
from contextlib import contextmanager, nullcontext
from contextvars import ContextVar
from dataclasses import dataclass, replace
from types import MappingProxyType, MethodType
from typing import Any, Iterator

from mllm_comp.backbones._provider_patching import preserve_bound_method_signature
from mllm_comp.compressors.video import (
    METokPruningResult,
    PruneVidVTPResult,
    VideoCompressionResult,
    VideoTokenCoordinate,
    VideoTokenPayload,
)
from mllm_comp.core.errors import BackendIntegrationError, InvariantError
from mllm_comp.core.layout import layout_delta_record

from .image_compression import (
    LLAVA_ONEVISION_SOURCE_COMMIT,
    LLAVA_ONEVISION_SOURCE_ROUTE,
)


LLAVA_ONEVISION_VIDEO_BRIDGE_API_VERSION = "mllm_comp.llava_onevision.video.v1"
LLAVA_ONEVISION_VIDEO_SPLICE_SITE = "video_projected_features_pre_llm_splice"
LLAVA_ONEVISION_VIDEO_PRE_PROJECTOR_SITE = "video_pre_vision_projector_pooled"
LLAVA_IMAGE_TOKEN_INDEX = -200
LLAVA_IGNORE_INDEX = -100


@dataclass(frozen=True)
class LlavaOneVisionVideoPacking:
    """Exact native packing of patch and structural tokens for one video."""

    frame_count: int
    grid_height: int
    grid_width: int
    patch_merge_type: str
    newline_position: str
    packed_token_count: int
    patch_positions: tuple[int, ...]
    structural_positions: tuple[int, ...]
    coordinates: tuple[VideoTokenCoordinate, ...]

    def __post_init__(self) -> None:
        for name in ("frame_count", "grid_height", "grid_width", "packed_token_count"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        patch_positions = tuple(self.patch_positions)
        structural_positions = tuple(self.structural_positions)
        if patch_positions != tuple(sorted(set(patch_positions))):
            raise ValueError("patch_positions must be unique and source ordered")
        if structural_positions != tuple(sorted(set(structural_positions))):
            raise ValueError("structural_positions must be unique and source ordered")
        if set(patch_positions) & set(structural_positions):
            raise ValueError("patch and structural positions must be disjoint")
        if tuple(sorted((*patch_positions, *structural_positions))) != tuple(
            range(self.packed_token_count)
        ):
            raise ValueError(
                "patch and structural positions must partition the packed span"
            )
        expected_patches = self.frame_count * self.grid_height * self.grid_width
        if len(patch_positions) != expected_patches:
            raise ValueError("patch_positions do not match the native frame grid")
        if len(self.coordinates) != expected_patches:
            raise ValueError("coordinates must contain one entry per patch token")

    @classmethod
    def from_native(
        cls,
        *,
        frame_count: int,
        grid_height: int,
        grid_width: int,
        patch_merge_type: str,
        newline_position: str,
        frame_indices: Sequence[int] | None = None,
        timestamps_seconds: Sequence[float] | None = None,
    ) -> "LlavaOneVisionVideoPacking":
        """Build the packing implemented by the pinned ``llava_arch.py`` route."""

        for name, value in (
            ("frame_count", frame_count),
            ("grid_height", grid_height),
            ("grid_width", grid_width),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        merge_type = str(patch_merge_type)
        newline = str(newline_position)
        if merge_type != "flat" and not merge_type.startswith("spatial"):
            raise BackendIntegrationError(
                f"Unsupported LLaVA video patch merge type {merge_type!r}."
            )
        if frame_indices is None:
            resolved_frames = tuple(range(frame_count))
        else:
            resolved_frames = tuple(frame_indices)
            if len(resolved_frames) != frame_count or any(
                isinstance(value, bool) or not isinstance(value, int) or value < 0
                for value in resolved_frames
            ):
                raise ValueError(
                    "frame_indices must match frame_count and be non-negative"
                )
        if timestamps_seconds is None:
            resolved_times: tuple[float | None, ...] = (None,) * frame_count
        else:
            values = tuple(timestamps_seconds)
            if len(values) != frame_count:
                raise ValueError("timestamps_seconds must match frame_count")
            resolved_times = tuple(float(value) for value in values)
            if any(not math.isfinite(value) or value < 0 for value in resolved_times):
                raise ValueError("timestamps_seconds must be finite and non-negative")

        patch_positions: list[int] = []
        structural_positions: list[int] = []
        coordinates: list[VideoTokenCoordinate] = []
        packed_cursor = 0
        use_grid_newlines = merge_type.startswith("spatial") and newline == "grid"
        use_frame_newlines = merge_type.startswith("spatial") and newline == "frame"
        use_single_newline = (
            merge_type.startswith("spatial")
            and newline == "one_token"
            and "unpad" in merge_type
        )
        if merge_type.startswith("spatial") and newline not in {
            "grid",
            "frame",
            "one_token",
            "no_token",
        }:
            raise BackendIntegrationError(
                f"Unsupported LLaVA video newline position {newline!r}."
            )

        for temporal_index in range(frame_count):
            for row in range(grid_height):
                for column in range(grid_width):
                    patch_positions.append(packed_cursor)
                    coordinates.append(
                        VideoTokenCoordinate(
                            t=temporal_index,
                            h=row,
                            w=column,
                            frame_index=resolved_frames[temporal_index],
                            timestamp_seconds=resolved_times[temporal_index],
                        )
                    )
                    packed_cursor += 1
                if use_grid_newlines:
                    structural_positions.append(packed_cursor)
                    packed_cursor += 1
            if use_frame_newlines:
                structural_positions.append(packed_cursor)
                packed_cursor += 1
        if use_single_newline:
            structural_positions.append(packed_cursor)
            packed_cursor += 1

        return cls(
            frame_count=frame_count,
            grid_height=grid_height,
            grid_width=grid_width,
            patch_merge_type=merge_type,
            newline_position=newline,
            packed_token_count=packed_cursor,
            patch_positions=tuple(patch_positions),
            structural_positions=tuple(structural_positions),
            coordinates=tuple(coordinates),
        )

    def payload(
        self,
        packed_tokens: Any,
        *,
        media_id: str,
        signals: Mapping[str, Any] | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> VideoTokenPayload:
        """Gather only compressible patches from one native packed span."""

        shape = _feature_shape(packed_tokens)
        if len(shape) != 2:
            raise BackendIntegrationError(
                "LLaVA packed video tokens must be rank two [tokens, hidden]."
            )
        if shape[0] != self.packed_token_count:
            raise BackendIntegrationError(
                "LLaVA packed video token count differs from its packing contract."
            )
        features = _gather_feature_rows(packed_tokens, self.patch_positions)
        resolved_signals = {
            "llava.packed_token_count": self.packed_token_count,
            "llava.patch_positions": self.patch_positions,
            "llava.structural_positions": self.structural_positions,
            **dict(signals or {}),
        }
        resolved_metadata = {
            "backbone": "llava_onevision",
            "packing": {
                "frame_count": self.frame_count,
                "grid_height": self.grid_height,
                "grid_width": self.grid_width,
                "patch_merge_type": self.patch_merge_type,
                "newline_position": self.newline_position,
                "packed_token_count": self.packed_token_count,
                "structural_token_count": len(self.structural_positions),
            },
            **dict(metadata or {}),
        }
        return VideoTokenPayload(
            features=features,
            media_ids=(media_id,),
            video_offsets=(0, len(self.patch_positions)),
            coordinates=self.coordinates,
            signals=resolved_signals,
            metadata=resolved_metadata,
        )


@dataclass(frozen=True)
class LlavaOneVisionPackedVideoRequest:
    """Provider callback payload at the native packed-video splice boundary."""

    invocation_id: str
    media_id: str
    payload: VideoTokenPayload
    packing: LlavaOneVisionVideoPacking
    source_visual_span: tuple[int, int]
    prompt_input_ids: Any
    prepared_attention_mask: Any | None
    prepared_position_ids: Any | None
    prepared_labels: Any | None
    modalities: tuple[str, ...] = ("video",)

    def __post_init__(self) -> None:
        if not isinstance(self.invocation_id, str) or not self.invocation_id:
            raise ValueError("invocation_id must be non-empty")
        if not isinstance(self.media_id, str) or not self.media_id:
            raise ValueError("media_id must be non-empty")
        if self.modalities != ("video",):
            raise ValueError("LLaVA packed video requests target exactly one video")
        if self.payload.media_ids != (self.media_id,):
            raise ValueError("request media_id differs from the shared payload")
        if self.payload.token_count != len(self.packing.patch_positions):
            raise ValueError("payload token count differs from native patch packing")
        start, end = self.source_visual_span
        if start < 0 or end - start != self.packing.packed_token_count:
            raise ValueError("source_visual_span differs from native packing")


@dataclass(frozen=True)
class LlavaOneVisionVideoSpliceResult:
    """Legalized native tuple plus auditable sequence/layout correspondence."""

    prepared: tuple[Any, Any, Any, Any, Any, Any]
    source_visual_span: tuple[int, int]
    result_visual_span: tuple[int, int]
    sequence_representative_indices: tuple[int, ...]
    structural_source_positions: tuple[int, ...]
    structural_result_positions: tuple[int, ...]
    compression_result: VideoCompressionResult

    def record(self) -> dict[str, Any]:
        layout = layout_delta_record(self.compression_result.layout_deltas[0])
        return {
            "source_visual_span": self.source_visual_span,
            "result_visual_span": self.result_visual_span,
            "sequence_representative_indices": self.sequence_representative_indices,
            "structural_source_positions": self.structural_source_positions,
            "structural_result_positions": self.structural_result_positions,
            "source_patch_count": self.compression_result.source_count,
            "result_patch_count": self.compression_result.result_count,
            "compression_applied": self.compression_result.compression_applied,
            "layout_delta": layout,
        }


@dataclass(frozen=True)
class LlavaOneVisionVideoSplicePlan:
    """Framework-neutral ordering plan used by the tensor legalizer."""

    source_visual_span: tuple[int, int]
    result_visual_span: tuple[int, int]
    packed_items: tuple[tuple[int, str, int], ...]
    sequence_representative_indices: tuple[int, ...]
    structural_source_positions: tuple[int, ...]
    structural_result_positions: tuple[int, ...]


def plan_llava_onevision_video_splice(
    request: LlavaOneVisionPackedVideoRequest,
    compression_result: VideoCompressionResult,
    *,
    sequence_length: int,
) -> LlavaOneVisionVideoSplicePlan:
    """Validate provenance and plan patch/newline interleaving without a tensor API."""

    if not isinstance(request, LlavaOneVisionPackedVideoRequest):
        raise TypeError("request must be LlavaOneVisionPackedVideoRequest")
    if not isinstance(compression_result, VideoCompressionResult):
        raise TypeError("compression_result must be VideoCompressionResult")
    if compression_result.media_ids != request.payload.media_ids:
        raise InvariantError("LLaVA video compression changed media identity.")
    if compression_result.source_video_offsets != request.payload.video_offsets:
        raise InvariantError("LLaVA video compression changed its source boundary.")
    if compression_result.video_count != 1:
        raise InvariantError("LLaVA video splice supports exactly one video.")
    if (
        isinstance(sequence_length, bool)
        or not isinstance(sequence_length, int)
        or sequence_length <= 0
    ):
        raise ValueError("sequence_length must be a positive integer")
    start, end = request.source_visual_span
    if not (0 <= start < end <= sequence_length):
        raise InvariantError(
            "LLaVA source video span is outside the embedding sequence."
        )

    patch_to_packed = request.packing.patch_positions
    representatives = compression_result.representative_indices[0]
    if representatives != tuple(sorted(representatives)):
        raise InvariantError(
            "LLaVA video representative anchors must preserve source order."
        )
    packed_items: list[tuple[int, str, int]] = [
        (patch_to_packed[source_index], "patch", result_index)
        for result_index, source_index in enumerate(representatives)
    ]
    packed_items.extend(
        (packed_index, "structural", packed_index)
        for packed_index in request.packing.structural_positions
    )
    packed_items.sort(key=lambda item: item[0])
    if not packed_items:
        raise InvariantError("LLaVA video splice cannot produce an empty visual span.")
    structural_result_positions = tuple(
        index
        for index, (_packed_position, kind, _value_index) in enumerate(packed_items)
        if kind == "structural"
    )
    visual_representatives = tuple(start + item[0] for item in packed_items)
    sequence_representatives = (
        *range(start),
        *visual_representatives,
        *range(end, sequence_length),
    )
    return LlavaOneVisionVideoSplicePlan(
        source_visual_span=(start, end),
        result_visual_span=(start, start + len(packed_items)),
        packed_items=tuple(packed_items),
        sequence_representative_indices=tuple(sequence_representatives),
        structural_source_positions=request.packing.structural_positions,
        structural_result_positions=structural_result_positions,
    )


def legalize_llava_onevision_video_splice(
    prepared: tuple[Any, ...],
    request: LlavaOneVisionPackedVideoRequest,
    compression_result: VideoCompressionResult,
    *,
    ignore_index: int = LLAVA_IGNORE_INDEX,
) -> LlavaOneVisionVideoSpliceResult:
    """Replace one native video span and repair every one-dimensional sequence."""

    if len(prepared) != 6:
        raise BackendIntegrationError(
            "Pinned LLaVA multimodal preparation must return a six-item tuple."
        )
    prepared_ids, positions, attention_mask, past, embeds, labels = prepared
    if prepared_ids is not None or past is not None:
        raise BackendIntegrationError(
            "LLaVA video compression requires the prefill embedding splice path."
        )
    if getattr(embeds, "ndim", None) != 3 or int(embeds.shape[0]) != 1:
        raise BackendIntegrationError(
            "LLaVA video splice requires [1, sequence, hidden] embeddings."
        )
    result_features = compression_result.features
    if getattr(result_features, "ndim", None) != 2:
        raise InvariantError("Compressed LLaVA video features must be rank two.")
    if int(result_features.shape[1]) != int(embeds.shape[2]):
        raise InvariantError("Compressed LLaVA video hidden size changed.")

    start, end = request.source_visual_span
    sequence_length = int(embeds.shape[1])
    plan = plan_llava_onevision_video_splice(
        request,
        compression_result,
        sequence_length=sequence_length,
    )
    source_packed = embeds[0, start:end]
    if int(source_packed.shape[0]) != request.packing.packed_token_count:
        raise InvariantError("LLaVA source video span differs from its packing.")

    torch = _torch()
    rows = []
    for packed_position, kind, value_index in plan.packed_items:
        if kind == "patch":
            rows.append(result_features[value_index])
        else:
            rows.append(source_packed[value_index])
    result_packed = torch.stack(rows, dim=0)
    new_embeds = torch.cat(
        (embeds[0, :start], result_packed, embeds[0, end:]), dim=0
    ).unsqueeze(0)

    keep = torch.tensor(
        plan.sequence_representative_indices,
        dtype=torch.long,
        device=embeds.device,
    )
    new_mask = _gather_sequence(attention_mask, keep, sequence_length, "attention_mask")
    new_labels = _gather_sequence(labels, keep, sequence_length, "labels")
    if labels is not None:
        visual_labels = _sequence_slice(labels, start, end, sequence_length, "labels")
        if bool((visual_labels != ignore_index).any().item()):
            raise InvariantError(
                "LLaVA native video labels must be ignored before compression."
            )
        result_start = start
        result_end = start + int(result_packed.shape[0])
        result_labels = _sequence_slice(
            new_labels,
            result_start,
            result_end,
            int(new_embeds.shape[1]),
            "labels",
        )
        if bool((result_labels != ignore_index).any().item()):
            raise InvariantError("LLaVA compressed video labels must remain ignored.")
    new_positions = _dense_position_ids(
        positions,
        sequence_length=int(new_embeds.shape[1]),
        reference=embeds,
    )
    return LlavaOneVisionVideoSpliceResult(
        prepared=(None, new_positions, new_mask, None, new_embeds, new_labels),
        source_visual_span=plan.source_visual_span,
        result_visual_span=plan.result_visual_span,
        sequence_representative_indices=plan.sequence_representative_indices,
        structural_source_positions=plan.structural_source_positions,
        structural_result_positions=plan.structural_result_positions,
        compression_result=compression_result,
    )


class LlavaOneVisionVideoCallbackHandle:
    def __init__(self, bridge: "PinnedLlavaOneVisionVideoBridge", token: int) -> None:
        self._bridge = bridge
        self._token = token
        self._removed = False

    def remove(self) -> None:
        if self._removed:
            return
        self._bridge._remove_callback(self._token)
        self._removed = True


class PinnedLlavaOneVisionVideoBridge:
    """Reversible instance-local callback bridge for the pinned video splice."""

    api_version = LLAVA_ONEVISION_VIDEO_BRIDGE_API_VERSION
    source_route = LLAVA_ONEVISION_SOURCE_ROUTE
    source_commit = LLAVA_ONEVISION_SOURCE_COMMIT
    capabilities: Mapping[str, bool] = MappingProxyType(
        {
            "native_video_projection": True,
            "native_video_packing": True,
            "projector_to_llm_splice": True,
            "structural_newline_preservation": True,
            "sequence_layout_repair": True,
            "training_label_repair": True,
            "one_dimensional_position_rebuild": True,
        }
    )

    def __init__(
        self,
        model: Any,
        *,
        _contract_verifier: Callable[[Any], None] | None = None,
    ) -> None:
        self.model = model
        (_contract_verifier or _verify_pinned_video_contract)(model)
        self._contract_mode = (
            "pinned_source" if _contract_verifier is None else "injected_test_double"
        )
        self._callback: Callable[[LlavaOneVisionPackedVideoRequest], Any] | None = None
        self._signal_provider: Any | None = None
        self._projector_position = "after"
        self._preserve_projected_values = False
        self._callback_token = 0
        self._next_token = 1
        self._original_prepare: Any | None = None
        self._prepare_had_instance = False
        self._prepare_prior: Any | None = None
        self._replacement: Any | None = None
        self._request_counter = 0
        self._callback_count = 0
        self._latest_splice: dict[str, Any] | None = None
        self._latest_metok: list[dict[str, Any]] | None = None
        self._latest_vtp: dict[str, Any] | None = None
        self._attached_attribute: str | None = None
        self._request_metadata: ContextVar[Mapping[str, Any] | None] = ContextVar(
            f"llava_video_request_metadata_{id(self)}", default=None
        )
        self._active_splice: ContextVar[LlavaOneVisionVideoSpliceResult | None] = (
            ContextVar(f"llava_video_active_splice_{id(self)}", default=None)
        )
        self._training_labels: ContextVar[Any | None] = ContextVar(
            f"llava_video_training_labels_{id(self)}", default=None
        )
        self._external_training_loss: ContextVar[bool] = ContextVar(
            f"llava_video_external_loss_{id(self)}", default=False
        )

    @property
    def patched(self) -> bool:
        return self._replacement is not None

    def available_sites(self) -> tuple[str, ...]:
        return (LLAVA_ONEVISION_VIDEO_SPLICE_SITE,)

    def register_callback(
        self,
        site: str,
        callback: Callable[[LlavaOneVisionPackedVideoRequest], Any],
        *,
        signal_provider: Any | None = None,
        projector_position: str = "after",
        preserve_projected_values: bool = False,
    ) -> LlavaOneVisionVideoCallbackHandle:
        if site != LLAVA_ONEVISION_VIDEO_SPLICE_SITE:
            raise BackendIntegrationError(
                f"Unknown LLaVA video provider site {site!r}."
            )
        if projector_position not in {"before", "after"}:
            raise BackendIntegrationError("LLaVA video projector_position must be before or after.")
        if projector_position == "before":
            projector = self.model.get_model().mm_projector
            if type(projector).__name__ == "PoolerProjector":
                raise BackendIntegrationError(
                    "Pre-projector video compression requires a tokenwise projector; "
                    "PoolerProjector consumes a complete fixed spatial grid, which "
                    "VQToken cluster outputs do not retain."
                )
        if not callable(callback):
            raise BackendIntegrationError(
                "LLaVA video provider callback must be callable."
            )
        if self._callback is not None:
            raise BackendIntegrationError(
                "LLaVA video splice already has an active compression callback."
            )
        if signal_provider is not None and not all(
            callable(getattr(signal_provider, name, None))
            for name in ("capture_context", "payload_signals")
        ):
            raise BackendIntegrationError(
                "LLaVA video signal provider has an incomplete capture contract."
            )
        if not self.patched:
            self._install_patch()
        token = self._next_token
        self._next_token += 1
        self._callback = callback
        self._signal_provider = signal_provider
        self._projector_position = projector_position
        self._preserve_projected_values = preserve_projected_values
        self._callback_token = token
        return LlavaOneVisionVideoCallbackHandle(self, token)

    @contextmanager
    def request_context(self, metadata: Mapping[str, Any]) -> Iterator[None]:
        if not isinstance(metadata, Mapping):
            raise TypeError("LLaVA video request metadata must be a mapping")
        token = self._request_metadata.set(dict(metadata))
        try:
            yield
        finally:
            self._request_metadata.reset(token)

    def runtime_metadata(self) -> dict[str, Any]:
        return {
            "provider": "vendored_llava_next_video_instance_bridge",
            "bridge_api_version": self.api_version,
            "source_route": self.source_route,
            "source_commit": self.source_commit,
            "contract_mode": self._contract_mode,
            "patched": self.patched,
            "active_sites": (
                [LLAVA_ONEVISION_VIDEO_SPLICE_SITE]
                if self._callback is not None
                else []
            ),
            "projector_position": self._projector_position,
            "target_site": LLAVA_ONEVISION_VIDEO_PRE_PROJECTOR_SITE if self._projector_position == "before" else LLAVA_ONEVISION_VIDEO_SPLICE_SITE,
            "dense_projector_packing_reference": self._projector_position == "before",
            "callback_count": self._callback_count,
            "capabilities": dict(self.capabilities),
            "latest_splice": self._latest_splice,
            "latest_metok": self._latest_metok,
            "latest_vtp": self._latest_vtp,
            "signal_provider": (
                self._signal_provider.runtime_metadata()
                if self._signal_provider is not None
                and callable(getattr(self._signal_provider, "runtime_metadata", None))
                else None
            ),
            "support_activation_claimed": False,
        }

    @contextmanager
    def prunevid_decoder_context(
        self,
        *,
        external_training_loss: bool,
    ) -> Iterator[None]:
        """Own request-local labels while PruneVid changes decoder length."""

        if not isinstance(external_training_loss, bool):
            raise TypeError("external_training_loss must be a boolean")
        splice_token = self._active_splice.set(None)
        label_token = self._training_labels.set(None)
        loss_token = self._external_training_loss.set(external_training_loss)
        try:
            yield
        finally:
            self._external_training_loss.reset(loss_token)
            self._training_labels.reset(label_token)
            self._active_splice.reset(splice_token)

    def prunevid_training_labels(self) -> Any | None:
        return self._training_labels.get()

    @contextmanager
    def metok_decoder_context(
        self,
        *,
        external_training_loss: bool,
    ) -> Iterator[None]:
        """Own request-local labels while METok prunes three decoder stages."""

        if not isinstance(external_training_loss, bool):
            raise TypeError("external_training_loss must be a boolean")
        splice_token = self._active_splice.set(None)
        label_token = self._training_labels.set(None)
        loss_token = self._external_training_loss.set(external_training_loss)
        self._latest_metok = []
        try:
            yield
        finally:
            self._external_training_loss.reset(loss_token)
            self._training_labels.reset(label_token)
            self._active_splice.reset(splice_token)

    def metok_training_labels(self) -> Any | None:
        return self._training_labels.get()

    def apply_metok_pruning_layout(self, result: METokPruningResult) -> None:
        """Compact SFT labels through one pre-layer METok keep vector."""

        if not isinstance(result, METokPruningResult):
            raise TypeError("result must be a METokPruningResult")
        splice = self._active_splice.get()
        if splice is None:
            raise BackendIntegrationError(
                "LLaVA METok pruning must run inside an active video prompt"
            )
        records = self._latest_metok
        if records is None:
            records = []
            self._latest_metok = records
        expected_sequence = (
            int(splice.prepared[4].shape[1])
            if not records
            else int(records[-1]["result_sequence_length"])
        )
        expected_visual = (
            splice.compression_result.result_count
            if not records
            else int(records[-1]["result_visual_count"])
        )
        if result.sequence_layout_delta.source_count != expected_sequence:
            raise InvariantError("LLaVA METok sequence length changed")
        if result.visual_layout_delta.source_count != expected_visual:
            raise InvariantError("LLaVA METok visual layout changed")
        labels = self._training_labels.get()
        if labels is not None:
            dropped = tuple(result.sequence_layout_delta.dropped_indices)
            if dropped and bool(
                (labels[0, list(dropped)] != LLAVA_IGNORE_INDEX).any().item()
            ):
                raise InvariantError("LLaVA METok may drop ignored visual labels only")
            torch = _torch()
            keep = torch.tensor(
                result.kept_sequence_indices,
                dtype=torch.long,
                device=labels.device,
            )
            self._training_labels.set(labels.index_select(1, keep))
        records.append(
            {
                "boundary_layer": result.boundary_layer,
                "application_position": result.metadata.get("application_position", {"boundary": "inter_layer", "index": result.boundary_layer}),
                "stage_index": result.stage_index,
                "source_visual_count": result.visual_layout_delta.source_count,
                "result_visual_count": result.visual_layout_delta.result_count,
                "source_sequence_length": result.sequence_layout_delta.source_count,
                "result_sequence_length": result.sequence_layout_delta.result_count,
                "structural_tokens_preserved": True,
                "position_policy": "gather_native_one_dimensional_rope",
            }
        )

    def apply_prunevid_vtp_layout(self, result: PruneVidVTPResult) -> None:
        """Compact externally-owned SFT labels at the decoder VTP boundary."""

        if not isinstance(result, PruneVidVTPResult):
            raise TypeError("result must be a PruneVidVTPResult")
        splice = self._active_splice.get()
        if splice is None:
            raise BackendIntegrationError(
                "LLaVA PruneVid VTP must run inside an active video prompt"
            )
        source_length = int(splice.prepared[4].shape[1])
        if result.sequence_layout_delta.source_count != source_length:
            raise InvariantError("LLaVA PruneVid VTP sequence length changed")
        if (
            result.visual_layout_delta.source_count
            != splice.compression_result.result_count
        ):
            raise InvariantError("LLaVA PruneVid VTP visual layout changed")
        labels = self._training_labels.get()
        if labels is not None:
            dropped = tuple(result.sequence_layout_delta.dropped_indices)
            if dropped and bool(
                (labels[0, list(dropped)] != LLAVA_IGNORE_INDEX).any().item()
            ):
                raise InvariantError(
                    "LLaVA PruneVid VTP may drop ignored visual labels only"
                )
            torch = _torch()
            keep = torch.tensor(
                result.kept_sequence_indices,
                dtype=torch.long,
                device=labels.device,
            )
            self._training_labels.set(labels.index_select(1, keep))
        self._latest_vtp = {
            "selected_layer": result.selected_layer,
            "application_position": result.metadata.get("application_position", {"boundary": "inter_layer", "index": result.selected_layer + 1}),
            "source_visual_count": result.visual_layout_delta.source_count,
            "result_visual_count": result.visual_layout_delta.result_count,
            "source_sequence_length": source_length,
            "result_sequence_length": result.sequence_layout_delta.result_count,
            "structural_tokens_preserved": True,
            "position_policy": "dense_one_dimensional_rebuild",
        }

    def detach(self) -> None:
        if self._callback is not None:
            raise BackendIntegrationError(
                "Cannot detach the LLaVA video bridge while a callback is active."
            )
        self._restore_patch()
        if (
            self._attached_attribute is not None
            and getattr(self.model, self._attached_attribute, None) is self
        ):
            delattr(self.model, self._attached_attribute)
        self._attached_attribute = None

    def _remove_callback(self, token: int) -> None:
        if self._callback is None or token != self._callback_token:
            return
        self._callback = None
        self._signal_provider = None
        self._projector_position = "after"
        self._preserve_projected_values = False
        self._callback_token = 0
        self._restore_patch()

    def _install_patch(self) -> None:
        original = getattr(self.model, "prepare_inputs_labels_for_multimodal", None)
        if not callable(original):
            raise BackendIntegrationError(
                "LLaVA video bridge requires prepare_inputs_labels_for_multimodal."
            )
        self._prepare_had_instance = "prepare_inputs_labels_for_multimodal" in vars(
            self.model
        )
        self._prepare_prior = vars(self.model).get(
            "prepare_inputs_labels_for_multimodal"
        )
        self._original_prepare = original
        replacement = self._prepare_replacement(original)
        bound = MethodType(replacement, self.model)
        setattr(self.model, "prepare_inputs_labels_for_multimodal", bound)
        self._replacement = bound

    def _restore_patch(self) -> None:
        if not self.patched:
            return
        current = getattr(self.model, "prepare_inputs_labels_for_multimodal", None)
        same_function = getattr(current, "__func__", None) is getattr(
            self._replacement,
            "__func__",
            None,
        )
        if current is not self._replacement and not same_function:
            raise BackendIntegrationError(
                "LLaVA video prepare hook changed while the bridge was active."
            )
        if self._prepare_had_instance:
            setattr(
                self.model,
                "prepare_inputs_labels_for_multimodal",
                self._prepare_prior,
            )
        else:
            delattr(self.model, "prepare_inputs_labels_for_multimodal")
        self._original_prepare = None
        self._prepare_prior = None
        self._replacement = None

    def _prepare_replacement(self, original: Callable[..., Any]) -> Callable[..., Any]:
        signature = inspect.signature(original)

        def replacement(_model: Any, *args: Any, **kwargs: Any) -> Any:
            callback = self._callback
            if callback is None:
                return original(*args, **kwargs)
            bound = signature.bind_partial(*args, **kwargs)
            call = dict(bound.arguments)
            if _normalise_modalities(call.get("modalities")) != ("video",):
                return original(*args, **kwargs)
            input_ids = call.get("input_ids")
            if getattr(input_ids, "ndim", None) != 2 or int(input_ids.shape[0]) != 1:
                raise BackendIntegrationError(
                    "LLaVA video compression requires one input_ids batch item."
                )
            if call.get("past_key_values") is not None or int(input_ids.shape[1]) == 1:
                return original(*args, **kwargs)
            frames = _single_video_frames(call.get("images"))
            signal_provider = self._signal_provider
            capture = (
                nullcontext(None)
                if signal_provider is None
                else signal_provider.capture_context(frames=frames)
            )
            before_projector = self._projector_position == "before"
            raw_sources: list[Any] = []
            projector = self.model.get_model().mm_projector if before_projector else None
            handle = (
                projector.register_forward_pre_hook(
                    lambda _module, inputs: raw_sources.append(inputs[0])
                )
                if before_projector else None
            )
            try:
                with capture as signal_capture:
                    prepared = original(*args, **kwargs)
            finally:
                if handle is not None:
                    handle.remove()
            if not isinstance(prepared, tuple) or len(prepared) != 6:
                raise BackendIntegrationError(
                    "Pinned LLaVA video preparation returned an unknown structure."
                )
            embeds = prepared[4]
            if getattr(embeds, "ndim", None) != 3 or int(embeds.shape[0]) != 1:
                raise BackendIntegrationError(
                    "Pinned LLaVA video preparation did not return batch-one embeddings."
                )

            active_ids = _active_prompt_ids(input_ids, call.get("attention_mask"))
            placeholders = (
                (active_ids == LLAVA_IMAGE_TOKEN_INDEX)
                .nonzero(as_tuple=False)
                .flatten()
            )
            if int(placeholders.numel()) != 1:
                raise BackendIntegrationError(
                    "LLaVA video compression requires exactly one IMAGE_TOKEN_INDEX."
                )
            visual_start = int(placeholders[0].item())
            config = getattr(self.model, "config", None)
            if config is None:
                raise BackendIntegrationError("LLaVA video model has no config.")
            if bool(getattr(config, "add_faster_video", False)):
                raise BackendIntegrationError(
                    "The canonical LLaVA video splice does not enable add_faster_video."
                )
            grid_height, grid_width = _pooled_grid_shape(self.model, config)
            context = dict(self._request_metadata.get() or {})
            video_context = context.get("video", context)
            if not isinstance(video_context, Mapping):
                video_context = {}
            frame_count = int(frames.shape[0])
            frame_indices = video_context.get("frame_indices")
            timestamps = video_context.get("timestamps_seconds")
            packing = LlavaOneVisionVideoPacking.from_native(
                frame_count=frame_count,
                grid_height=grid_height,
                grid_width=grid_width,
                patch_merge_type=str(getattr(config, "mm_patch_merge_type", "flat")),
                newline_position=str(
                    getattr(config, "mm_newline_position", "one_token")
                ),
                frame_indices=frame_indices,
                timestamps_seconds=timestamps,
            )
            text_count = int(active_ids.numel()) - 1
            expected_sequence = text_count + packing.packed_token_count
            if int(embeds.shape[1]) != expected_sequence:
                raise BackendIntegrationError(
                    "LLaVA video packed span was truncated or differs from the native "
                    f"packing contract: expected_sequence={expected_sequence}, "
                    f"actual_sequence={int(embeds.shape[1])}."
                )
            visual_end = visual_start + packing.packed_token_count
            packed_tokens = embeds[0, visual_start:visual_end]
            self._request_counter += 1
            invocation_id = f"video-splice-{self._request_counter}"
            media_id = str(
                video_context.get("media_id")
                or context.get("sample_id")
                or invocation_id
            )
            payload = packing.payload(
                packed_tokens,
                media_id=media_id,
                signals={
                    "llava.prompt_input_ids": active_ids,
                    "llava.visual_span": (visual_start, visual_end),
                    "text_context": video_context.get("conditioning_text"),
                    **(
                        {}
                        if signal_provider is None
                        else dict(
                            signal_provider.payload_signals(
                                signal_capture,
                                token_count=len(packing.patch_positions),
                                coordinates=packing.coordinates,
                            )
                        )
                    ),
                },
                metadata={"invocation_id": invocation_id},
            )
            enrich_payload = (
                None
                if signal_provider is None
                else getattr(signal_provider, "enrich_payload", None)
            )
            if callable(enrich_payload):
                payload = enrich_payload(payload, signal_capture)
                if not isinstance(payload, VideoTokenPayload):
                    raise BackendIntegrationError(
                        "LLaVA video signal provider returned an invalid payload"
                    )
            target_site = LLAVA_ONEVISION_VIDEO_SPLICE_SITE
            if before_projector:
                if len(raw_sources) != 1:
                    raise BackendIntegrationError(
                        "LLaVA pre-projector video compression requires the single "
                        "native dense projector input from video preparation."
                    )
                # This deliberately pools raw vision values on the original
                # video lattice. It does not commute pooling through projection
                # or expand irregular cluster outputs into a fabricated grid.
                raw_pooled = self.model.get_2dPool(raw_sources[0]).flatten(0, 1)
                if int(raw_pooled.shape[0]) != payload.token_count:
                    raise BackendIntegrationError(
                        "LLaVA raw pooled video values differ from the native "
                        "projected video token grid."
                    )
                target_site = LLAVA_ONEVISION_VIDEO_PRE_PROJECTOR_SITE
                payload = replace(payload, features=raw_pooled, signals={**payload.signals,
                    **({"projector_reference_features": payload.features.detach()} if self._preserve_projected_values else {})})
            payload = replace(payload, metadata={
                **payload.metadata,
                "projector_position": self._projector_position,
                "target_site": target_site,
                "value_token_domain": "spatially_pooled_vision" if before_projector else "spatially_pooled_projected",
                "dense_projector_packing_reference": before_projector,
            })
            request = LlavaOneVisionPackedVideoRequest(
                invocation_id=invocation_id,
                media_id=media_id,
                payload=payload,
                packing=packing,
                source_visual_span=(visual_start, visual_end),
                prompt_input_ids=active_ids,
                prepared_attention_mask=prepared[2],
                prepared_position_ids=prepared[1],
                prepared_labels=prepared[5],
            )
            response = callback(request)
            if not isinstance(response, VideoCompressionResult):
                raise BackendIntegrationError(
                    "LLaVA video callback must return VideoCompressionResult."
                )
            if before_projector:
                response = replace(
                    response,
                    features=projector(response.features).to(embeds.device, embeds.dtype),
                )
            splice = legalize_llava_onevision_video_splice(
                prepared,
                request,
                response,
            )
            self._active_splice.set(splice)
            # Native generate prepares embeddings before its first forward,
            # so a fresh prefill also owns a fresh progressive layout history.
            self._latest_metok = []
            if self._external_training_loss.get() and splice.prepared[5] is not None:
                self._training_labels.set(splice.prepared[5])
                prepared_without_labels = list(splice.prepared)
                prepared_without_labels[5] = None
                returned_prepared = tuple(prepared_without_labels)
            else:
                returned_prepared = splice.prepared
            self._callback_count += 1
            self._latest_splice = {
                "invocation_id": invocation_id,
                "media_id": media_id,
                "target_site": target_site,
                "callback_site": LLAVA_ONEVISION_VIDEO_SPLICE_SITE,
                "projector_position": self._projector_position,
                "dense_projector_packing_reference": before_projector,
                **splice.record(),
            }
            return returned_prepared

        preserve_bound_method_signature(replacement, original)
        return replacement


def install_llava_onevision_video_bridge(
    model: Any,
) -> PinnedLlavaOneVisionVideoBridge:
    """Attach or return the exact video bridge on one LLaVA model instance."""

    for name in (
        "mllm_video_compression_bridge",
        "_mllm_video_compression_bridge",
    ):
        existing = getattr(model, name, None)
        if existing is not None:
            if isinstance(existing, PinnedLlavaOneVisionVideoBridge):
                return existing
            raise BackendIntegrationError(
                f"Model already exposes an incompatible video bridge at {name}."
            )
    bridge = PinnedLlavaOneVisionVideoBridge(model)
    setattr(model, "mllm_video_compression_bridge", bridge)
    bridge._attached_attribute = "mllm_video_compression_bridge"
    return bridge


@contextmanager
def llava_onevision_video_request_context(
    model: Any,
    metadata: Mapping[str, Any],
) -> Iterator[None]:
    """Publish per-sample frame/timing identity only while one request runs."""

    bridge = getattr(model, "mllm_video_compression_bridge", None)
    request_context = getattr(bridge, "request_context", None)
    if not callable(request_context):
        yield
        return
    with request_context(metadata):
        yield


def _verify_pinned_video_contract(model: Any) -> None:
    # Share the already-frozen class/signature/hash guard with ImageLLM.  The
    # two bridges remain separate lifecycle objects and patch different exact
    # modality behavior on the same pinned source route.
    from .provider_bridge import _resolve_components, _verify_pinned_contract

    components = _resolve_components(model)
    _verify_pinned_contract(model, components)


def _normalise_modalities(value: Any) -> tuple[str, ...]:
    if value is None:
        return ("image",)
    if isinstance(value, str):
        return (value,)
    return tuple(str(item) for item in value)


def _single_video_frames(images: Any) -> Any:
    if isinstance(images, (list, tuple)):
        if len(images) != 1:
            raise BackendIntegrationError(
                "LLaVA video compression supports one video per sample."
            )
        frames = images[0]
    elif getattr(images, "ndim", None) == 5 and int(images.shape[0]) == 1:
        frames = images[0]
    else:
        raise BackendIntegrationError(
            "LLaVA video inputs must contain one rank-four frame tensor."
        )
    if getattr(frames, "ndim", None) != 4 or int(frames.shape[0]) <= 0:
        raise BackendIntegrationError(
            "LLaVA video inputs must contain one non-empty rank-four frame tensor."
        )
    return frames


def _pooled_grid_shape(model: Any, config: Any) -> tuple[int, int]:
    get_vision = getattr(model, "get_vision_tower", None)
    vision = get_vision() if callable(get_vision) else None
    source_side = getattr(vision, "num_patches_per_side", None)
    configured_stride = getattr(config, "mm_spatial_pool_stride", 2)
    if (
        isinstance(source_side, bool)
        or not isinstance(source_side, int)
        or source_side <= 0
        or isinstance(configured_stride, bool)
        or not isinstance(configured_stride, int)
    ):
        raise BackendIntegrationError(
            "LLaVA video packing requires positive vision grid and pool stride."
        )
    # The pinned route calls get_2dPool(image_feat) without forwarding the
    # config value, so its effective stride is the method default of two.  The
    # frozen canonical profile declares the same value and fails closed if a
    # different ablation is supplied.
    if configured_stride != 2:
        raise BackendIntegrationError(
            "The canonical LLaVA video splice requires mm_spatial_pool_stride=2."
        )
    stride = 2
    mode = str(getattr(config, "mm_spatial_pool_mode", "bilinear"))
    if mode == "bilinear":
        result_side = math.ceil(source_side / stride)
    elif mode in {"average", "max"}:
        result_side = source_side // stride
    else:
        raise BackendIntegrationError(
            f"Unsupported LLaVA video spatial pool mode {mode!r}."
        )
    if result_side <= 0:
        raise BackendIntegrationError("LLaVA video spatial pooling produced no grid.")
    return result_side, result_side


def _active_prompt_ids(input_ids: Any, attention_mask: Any | None) -> Any:
    if attention_mask is None:
        return input_ids[0]
    if tuple(int(item) for item in attention_mask.shape) != tuple(
        int(item) for item in input_ids.shape
    ):
        raise BackendIntegrationError(
            "LLaVA video prompt attention_mask must match input_ids."
        )
    return input_ids[0][attention_mask[0].bool()]


def _gather_sequence(
    value: Any | None,
    keep: Any,
    source_length: int,
    name: str,
) -> Any | None:
    if value is None:
        return None
    shape = tuple(int(item) for item in value.shape)
    if len(shape) == 1 and shape[0] == source_length:
        return value.index_select(0, keep.to(value.device))
    if len(shape) == 2 and shape == (1, source_length):
        return value.index_select(1, keep.to(value.device))
    raise InvariantError(
        f"LLaVA {name} must be [sequence] or [1, sequence], got {shape}."
    )


def _sequence_slice(
    value: Any,
    start: int,
    end: int,
    sequence_length: int,
    name: str,
) -> Any:
    shape = tuple(int(item) for item in value.shape)
    if len(shape) == 1 and shape[0] == sequence_length:
        return value[start:end]
    if len(shape) == 2 and shape == (1, sequence_length):
        return value[0, start:end]
    raise InvariantError(
        f"LLaVA {name} must be [sequence] or [1, sequence], got {shape}."
    )


def _dense_position_ids(
    value: Any | None,
    *,
    sequence_length: int,
    reference: Any,
) -> Any | None:
    if value is None:
        return None
    torch = _torch()
    shape = tuple(int(item) for item in value.shape)
    positions = torch.arange(
        sequence_length,
        dtype=value.dtype,
        device=value.device if hasattr(value, "device") else reference.device,
    )
    if len(shape) == 1:
        return positions
    if len(shape) == 2 and shape[0] == 1:
        return positions.unsqueeze(0)
    raise InvariantError(
        f"LLaVA position_ids must be [sequence] or [1, sequence], got {shape}."
    )


def _feature_shape(value: Any) -> tuple[int, ...]:
    shape = getattr(value, "shape", None)
    if shape is not None:
        return tuple(int(item) for item in shape)
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise BackendIntegrationError("LLaVA video features must be tensor-like.")
    rows = tuple(value)
    if not rows:
        return (0,)
    first = rows[0]
    if not isinstance(first, Sequence) or isinstance(first, (str, bytes, bytearray)):
        return (len(rows),)
    width = len(first)
    if any(
        not isinstance(row, Sequence)
        or isinstance(row, (str, bytes, bytearray))
        or len(row) != width
        for row in rows
    ):
        raise BackendIntegrationError("LLaVA video feature rows must be rectangular.")
    return (len(rows), width)


def _gather_feature_rows(value: Any, positions: tuple[int, ...]) -> Any:
    index_select = getattr(value, "index_select", None)
    if callable(index_select):
        torch = _torch()
        indices = torch.tensor(
            positions,
            dtype=torch.long,
            device=getattr(value, "device", None),
        )
        return index_select(0, indices)
    try:
        return value[list(positions)]
    except (IndexError, KeyError, TypeError):
        pass
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return tuple(value[index] for index in positions)
    raise BackendIntegrationError(
        "LLaVA packed video features do not support row gathering."
    )


def _torch() -> Any:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - environment preflight owns this
        raise RuntimeError("LLaVA video compression requires PyTorch.") from exc
    return torch


__all__ = [
    "LLAVA_ONEVISION_VIDEO_BRIDGE_API_VERSION",
    "LLAVA_ONEVISION_VIDEO_SPLICE_SITE",
    "LLAVA_ONEVISION_VIDEO_PRE_PROJECTOR_SITE",
    "LlavaOneVisionPackedVideoRequest",
    "LlavaOneVisionVideoCallbackHandle",
    "LlavaOneVisionVideoPacking",
    "LlavaOneVisionVideoSplicePlan",
    "LlavaOneVisionVideoSpliceResult",
    "PinnedLlavaOneVisionVideoBridge",
    "install_llava_onevision_video_bridge",
    "legalize_llava_onevision_video_splice",
    "llava_onevision_video_request_context",
    "plan_llava_onevision_video_splice",
]
