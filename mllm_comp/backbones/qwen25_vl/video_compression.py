"""Pinned Qwen2.5-VL video payload and sparse M-RoPE legalization.

Qwen2.5-VL serializes one video as a dense run of ``video_token_id``
placeholders.  By default the vision tower emits post-merger features in raster
``(t, h, w)`` order, while ``get_rope_index`` assigns native three-axis M-RoPE
positions from the *source* ``video_grid_thw`` and ``second_per_grid_ts``.

An irregular compressor cannot truthfully replace that source grid with a
smaller dense grid.  This module therefore keeps the original grid for vision
encoding and native position construction, scatters compressed features only
at source representative placeholders, removes the other placeholders at the
language-model boundary, and gathers the original M-RoPE anchors.  Attention
masks, ignored labels, cache positions, and generation decode inputs follow
the same physical compaction.

The bridge is instance-local, reversible, batch-one/single-video only, and
does not activate any support unit.  Image and text calls are delegated
unchanged.
"""

from __future__ import annotations

import inspect
import math
from collections.abc import Callable, Mapping, MutableMapping, Sequence
from contextlib import contextmanager, nullcontext
from contextvars import ContextVar
from dataclasses import dataclass, replace
from numbers import Integral
from types import MappingProxyType, MethodType
from typing import Any, Iterator

from mllm_comp.backbones._provider_patching import preserve_bound_method_signature
from mllm_comp.backbones._video_conditioning import single_video_conditioning_text
from mllm_comp.backbones.image_training import attach_shifted_causal_loss
from mllm_comp.compressors.video import (
    METokPruningResult,
    PruneVidVTPResult,
    VideoCompressionResult,
    VideoTokenCoordinate,
    VideoTokenPayload,
)
from mllm_comp.core.errors import BackendIntegrationError, InvariantError
from mllm_comp.core.layout import layout_delta_record


QWEN25_VL_VIDEO_BRIDGE_API_VERSION = "mllm_comp.qwen25_vl.video.v1"
QWEN25_VL_VIDEO_SPLICE_SITE = "video_post_merger_pre_llm_scatter"
QWEN25_VL_IGNORE_INDEX = -100
_VIDEO_BRIDGE_ATTRIBUTE = "_mllm_comp_qwen25_vl_video_bridge"
_PUBLIC_VIDEO_BRIDGE_ATTRIBUTE = "mllm_video_compression_bridge"
_IMAGE_PROVIDER_ATTRIBUTES = (
    "_mllm_comp_qwen25_vl_provider_bridge",
    "_mllm_comp_qwen25_vl_image_compression_bridge",
)


@dataclass(frozen=True)
class Qwen25VLVideoGrid:
    """One source video grid after the native spatial merger."""

    temporal: int
    source_height: int
    source_width: int
    spatial_merge_size: int
    second_per_grid_seconds: float

    def __post_init__(self) -> None:
        for name in (
            "temporal",
            "source_height",
            "source_width",
            "spatial_merge_size",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, Integral) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
            object.__setattr__(self, name, int(value))
        if self.source_height % self.spatial_merge_size:
            raise ValueError("source_height must be divisible by spatial_merge_size")
        if self.source_width % self.spatial_merge_size:
            raise ValueError("source_width must be divisible by spatial_merge_size")
        timing = float(self.second_per_grid_seconds)
        if not math.isfinite(timing) or timing <= 0:
            raise ValueError("second_per_grid_seconds must be finite and positive")

    @classmethod
    def from_native(
        cls,
        grid_thw: Sequence[int],
        *,
        spatial_merge_size: int,
        second_per_grid_seconds: float,
    ) -> "Qwen25VLVideoGrid":
        values = tuple(grid_thw)
        if len(values) != 3:
            raise ValueError("video_grid_thw must contain exactly (t, h, w)")
        if any(
            isinstance(value, bool) or not isinstance(value, Integral)
            for value in values
        ):
            raise TypeError("video_grid_thw entries must be integers")
        return cls(
            temporal=int(values[0]),
            source_height=int(values[1]),
            source_width=int(values[2]),
            spatial_merge_size=spatial_merge_size,
            second_per_grid_seconds=second_per_grid_seconds,
        )

    @property
    def merged_height(self) -> int:
        return self.source_height // self.spatial_merge_size

    @property
    def merged_width(self) -> int:
        return self.source_width // self.spatial_merge_size

    @property
    def token_count(self) -> int:
        return self.temporal * self.merged_height * self.merged_width

    @property
    def source_grid_thw(self) -> tuple[int, int, int]:
        return (self.temporal, self.source_height, self.source_width)

    @property
    def coordinates(self) -> tuple[VideoTokenCoordinate, ...]:
        return tuple(
            VideoTokenCoordinate(
                t=temporal,
                h=row,
                w=column,
                tubelet_index=temporal,
                timestamp_seconds=temporal * self.second_per_grid_seconds,
            )
            for temporal in range(self.temporal)
            for row in range(self.merged_height)
            for column in range(self.merged_width)
        )

    def payload(
        self,
        features: Any,
        *,
        media_id: str,
        signals: Mapping[str, Any] | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> VideoTokenPayload:
        shape = _feature_shape(features)
        if len(shape) != 2 or shape[0] != self.token_count:
            raise BackendIntegrationError(
                "Qwen2.5-VL spatial-group video features must be "
                f"[{self.token_count}, hidden], got {shape}."
            )
        return VideoTokenPayload(
            features=features,
            media_ids=(media_id,),
            video_offsets=(0, self.token_count),
            coordinates=self.coordinates,
            signals={
                "qwen25_vl.source_video_grid_thw": self.source_grid_thw,
                "qwen25_vl.spatial_merge_size": self.spatial_merge_size,
                "qwen25_vl.second_per_grid_seconds": self.second_per_grid_seconds,
                **dict(signals or {}),
            },
            metadata={
                "backbone": "qwen25_vl",
                "grid_policy": "preserve_source_grid_and_gather_sparse_mrope",
                "source_video_grid_thw": self.source_grid_thw,
                "merged_grid_thw": (
                    self.temporal,
                    self.merged_height,
                    self.merged_width,
                ),
                **dict(metadata or {}),
            },
        )


@dataclass(frozen=True)
class Qwen25VLPackedVideoRequest:
    """Algorithm request in raster group order at the selected merger boundary."""

    invocation_id: str
    media_id: str
    payload: VideoTokenPayload
    grid: Qwen25VLVideoGrid
    source_placeholder_positions: tuple[int, ...]
    prompt_input_ids: Any
    source_attention_mask: Any | None
    source_video_grid_thw: Any
    source_second_per_grid_ts: Any

    def __post_init__(self) -> None:
        if not isinstance(self.invocation_id, str) or not self.invocation_id:
            raise ValueError("invocation_id must be non-empty")
        if not isinstance(self.media_id, str) or not self.media_id:
            raise ValueError("media_id must be non-empty")
        if self.payload.media_ids != (self.media_id,):
            raise ValueError("request media_id differs from the shared payload")
        if self.payload.token_count != self.grid.token_count:
            raise ValueError("payload token count differs from the source video grid")
        positions = tuple(self.source_placeholder_positions)
        if len(positions) != self.grid.token_count:
            raise ValueError("source video placeholders must match the source grid")
        if any(
            isinstance(position, bool) or not isinstance(position, int) or position < 0
            for position in positions
        ):
            raise ValueError(
                "source video placeholder positions must be non-negative integers"
            )
        if positions != tuple(range(positions[0], positions[0] + len(positions))):
            raise ValueError("one Qwen2.5-VL video must occupy one contiguous span")


@dataclass(frozen=True)
class Qwen25VLVideoSplicePlan:
    """Framework-neutral prompt and source-anchor compaction plan."""

    source_visual_span: tuple[int, int]
    result_visual_span: tuple[int, int]
    source_placeholder_positions: tuple[int, ...]
    retained_placeholder_positions: tuple[int, ...]
    removed_placeholder_positions: tuple[int, ...]
    result_placeholder_positions: tuple[int, ...]
    kept_sequence_indices: tuple[int, ...]
    representative_indices: tuple[int, ...]


def plan_qwen25_vl_video_splice(
    request: Qwen25VLPackedVideoRequest,
    compression_result: VideoCompressionResult,
    *,
    sequence_length: int,
) -> Qwen25VLVideoSplicePlan:
    """Validate provenance and map sparse anchors into one compact prompt."""

    if not isinstance(request, Qwen25VLPackedVideoRequest):
        raise TypeError("request must be Qwen25VLPackedVideoRequest")
    if not isinstance(compression_result, VideoCompressionResult):
        raise TypeError("compression_result must be VideoCompressionResult")
    if compression_result.media_ids != request.payload.media_ids:
        raise InvariantError("Qwen2.5-VL video compression changed media identity.")
    if compression_result.source_video_offsets != request.payload.video_offsets:
        raise InvariantError(
            "Qwen2.5-VL video compression changed its source boundary."
        )
    if compression_result.video_count != 1:
        raise InvariantError("Qwen2.5-VL video splice supports exactly one video.")
    if (
        isinstance(sequence_length, bool)
        or not isinstance(sequence_length, int)
        or sequence_length <= 0
    ):
        raise ValueError("sequence_length must be a positive integer")

    source = request.source_placeholder_positions
    if source[0] < 0 or source[-1] >= sequence_length:
        raise InvariantError("Qwen2.5-VL source video span is outside the prompt.")
    representatives = compression_result.representative_indices[0]
    if representatives != tuple(sorted(representatives)):
        raise InvariantError(
            "Qwen2.5-VL video representative anchors must preserve source order."
        )
    retained = tuple(source[index] for index in representatives)
    retained_set = set(retained)
    removed = tuple(index for index in source if index not in retained_set)
    removed_set = set(removed)
    kept = tuple(index for index in range(sequence_length) if index not in removed_set)
    start = source[0]
    result_positions = tuple(range(start, start + len(retained)))
    return Qwen25VLVideoSplicePlan(
        source_visual_span=(source[0], source[-1] + 1),
        result_visual_span=(start, start + len(retained)),
        source_placeholder_positions=source,
        retained_placeholder_positions=retained,
        removed_placeholder_positions=removed,
        result_placeholder_positions=result_positions,
        kept_sequence_indices=kept,
        representative_indices=representatives,
    )


@dataclass(frozen=True)
class Qwen25VLVideoPromptSpliceResult:
    """Compacted prompt fields with gathered native M-RoPE coordinates."""

    input_ids: Any
    inputs_embeds: Any
    attention_mask: Any | None
    position_ids: Any
    cache_position: Any
    labels: Any | None
    rope_deltas: Any
    plan: Qwen25VLVideoSplicePlan
    compression_result: VideoCompressionResult
    source_visual_mrope: tuple[tuple[int, ...], ...]
    result_visual_mrope: tuple[tuple[int, ...], ...]

    def record(self) -> dict[str, Any]:
        return {
            "source_visual_span": self.plan.source_visual_span,
            "result_visual_span": self.plan.result_visual_span,
            "source_placeholder_count": len(self.plan.source_placeholder_positions),
            "result_placeholder_count": len(self.plan.result_placeholder_positions),
            "retained_placeholder_positions": self.plan.retained_placeholder_positions,
            "removed_placeholder_positions": self.plan.removed_placeholder_positions,
            "representative_indices": self.plan.representative_indices,
            "source_visual_mrope": self.source_visual_mrope,
            "result_visual_mrope": self.result_visual_mrope,
            "source_video_grid_preserved": True,
            "result_grid_policy": "irregular_sparse_source_anchors",
            "compression_applied": self.compression_result.compression_applied,
            "layout_delta": layout_delta_record(
                self.compression_result.layout_deltas[0]
            ),
        }


def legalize_qwen25_vl_video_prompt(
    *,
    input_ids: Any,
    inputs_embeds: Any,
    attention_mask: Any | None,
    position_ids: Any,
    cache_position: Any | None,
    labels: Any | None,
    request: Qwen25VLPackedVideoRequest,
    compression_result: VideoCompressionResult,
    ignore_index: int = QWEN25_VL_IGNORE_INDEX,
) -> Qwen25VLVideoPromptSpliceResult:
    """Gather one compressed prompt and compute its exact decode rope delta."""

    input_shape = _tensor_shape(input_ids)
    embed_shape = _tensor_shape(inputs_embeds)
    if len(input_shape) != 2 or input_shape[0] != 1:
        raise BackendIntegrationError(
            "Qwen2.5-VL video legalization requires [1, sequence] input_ids."
        )
    if len(embed_shape) != 3 or embed_shape[:2] != input_shape:
        raise BackendIntegrationError(
            "Qwen2.5-VL video legalization requires aligned [1, sequence, hidden] "
            "inputs_embeds."
        )
    source_length = input_shape[1]
    plan = plan_qwen25_vl_video_splice(
        request,
        compression_result,
        sequence_length=source_length,
    )
    position_shape = _tensor_shape(position_ids)
    if position_shape not in {
        (3, 1, source_length),
        (4, 1, source_length),
    }:
        raise BackendIntegrationError(
            "Qwen2.5-VL video M-RoPE positions must be [3, 1, sequence] or "
            "[4, 1, sequence]."
        )
    if attention_mask is not None and _tensor_shape(attention_mask) != input_shape:
        raise InvariantError("Qwen2.5-VL video attention_mask must match input_ids.")
    if labels is not None and _tensor_shape(labels) != input_shape:
        raise InvariantError("Qwen2.5-VL video labels must match input_ids.")

    torch = _torch()
    keep = torch.tensor(
        plan.kept_sequence_indices,
        dtype=torch.long,
        device=inputs_embeds.device,
    )
    compact_ids = input_ids.index_select(1, keep.to(input_ids.device))
    compact_embeds = inputs_embeds.index_select(1, keep)
    compact_mask = (
        None
        if attention_mask is None
        else attention_mask.index_select(1, keep.to(attention_mask.device))
    )
    compact_positions = position_ids.index_select(
        -1,
        keep.to(position_ids.device),
    )
    compact_labels = (
        None if labels is None else labels.index_select(1, keep.to(labels.device))
    )
    if labels is not None:
        source_visual_labels = labels[0, list(plan.source_placeholder_positions)]
        if bool((source_visual_labels != ignore_index).any().item()):
            raise InvariantError(
                "Qwen2.5-VL source video labels must be ignored before compression."
            )
        result_visual_labels = compact_labels[
            0,
            list(plan.result_placeholder_positions),
        ]
        if bool((result_visual_labels != ignore_index).any().item()):
            raise InvariantError(
                "Qwen2.5-VL compressed video labels must remain ignored."
            )

    if cache_position is None:
        compact_cache = torch.arange(
            len(plan.kept_sequence_indices),
            dtype=torch.long,
            device=inputs_embeds.device,
        )
    else:
        cache_shape = _tensor_shape(cache_position)
        if cache_shape != (source_length,):
            raise InvariantError(
                "Qwen2.5-VL prefill cache_position must cover the source prompt."
            )
        compact_cache = cache_position.index_select(
            0,
            keep.to(cache_position.device),
        )
        # Cache storage is physically compact even though rotary positions keep
        # their original sparse source coordinates.
        compact_cache = torch.arange(
            compact_cache.numel(),
            dtype=cache_position.dtype,
            device=cache_position.device,
        )

    rope_deltas = _exact_rope_deltas(compact_positions, compact_mask)
    rotary_source = position_ids[1:] if position_shape[0] == 4 else position_ids
    source_visual_mrope = _mrope_columns(
        rotary_source,
        plan.source_placeholder_positions,
    )
    result_visual_mrope = _mrope_columns(
        compact_positions[1:] if position_shape[0] == 4 else compact_positions,
        plan.result_placeholder_positions,
    )
    expected_result_mrope = tuple(
        source_visual_mrope[index] for index in plan.representative_indices
    )
    if result_visual_mrope != expected_result_mrope:
        raise InvariantError(
            "Qwen2.5-VL compressed video M-RoPE anchors changed during gather."
        )
    return Qwen25VLVideoPromptSpliceResult(
        input_ids=compact_ids,
        inputs_embeds=compact_embeds,
        attention_mask=compact_mask,
        position_ids=compact_positions,
        cache_position=compact_cache,
        labels=compact_labels,
        rope_deltas=rope_deltas,
        plan=plan,
        compression_result=compression_result,
        source_visual_mrope=source_visual_mrope,
        result_visual_mrope=result_visual_mrope,
    )


class Qwen25VLVideoCallbackHandle:
    def __init__(self, bridge: "PinnedQwen25VLVideoBridge", token: int) -> None:
        self._bridge = bridge
        self._token = token
        self._removed = False

    def remove(self) -> None:
        if self._removed:
            return
        self._bridge._remove_callback(self._token)
        self._removed = True


@dataclass
class _InstancePatch:
    owner: Any
    name: str
    had_instance: bool
    previous: Any
    installed: Any

    @classmethod
    def apply(cls, owner: Any, name: str, value: Any) -> "_InstancePatch":
        namespace = vars(owner)
        had_instance = name in namespace
        previous = namespace.get(name)
        setattr(owner, name, value)
        return cls(owner, name, had_instance, previous, value)

    def restore(self) -> None:
        current = getattr(self.owner, self.name, None)
        same_function = getattr(current, "__func__", None) is getattr(
            self.installed,
            "__func__",
            None,
        )
        if current is not self.installed and not same_function:
            raise BackendIntegrationError(
                f"Qwen2.5-VL video hook {self.name!r} changed while active."
            )
        if self.had_instance:
            setattr(self.owner, self.name, self.previous)
        elif self.name in vars(self.owner):
            delattr(self.owner, self.name)


@dataclass(frozen=True)
class _ActiveSource:
    input_ids: Any
    attention_mask: Any | None
    pixel_values_videos: Any
    video_grid_thw: Any
    second_per_grid_ts: Any
    grid: Qwen25VLVideoGrid
    source_placeholder_positions: tuple[int, ...]
    media_id: str
    conditioning_text: str | None


@dataclass(frozen=True)
class _ActiveSplice:
    request: Qwen25VLPackedVideoRequest
    result: VideoCompressionResult
    plan: Qwen25VLVideoSplicePlan


@dataclass
class _GenerationLifecycle:
    request_id: str
    prefill_complete: bool = False
    source_sequence_length: int | None = None
    result_sequence_length: int | None = None
    removed_placeholder_positions: tuple[int, ...] = ()
    vtp_removed_source_positions: tuple[int, ...] = ()
    vtp_result_sequence_length: int | None = None
    decode_repairs: int = 0
    last_decode_record: dict[str, Any] | None = None

    @property
    def removed_count(self) -> int:
        return len(self.removed_placeholder_positions)


class PinnedQwen25VLVideoBridge:
    """Reversible provider bridge for one pinned Qwen2.5-VL model."""

    api_version = QWEN25_VL_VIDEO_BRIDGE_API_VERSION
    capabilities: Mapping[str, bool] = MappingProxyType(
        {
            "native_post_merger_features": True,
            "source_video_grid_provenance": True,
            "sparse_mrope_anchor_gather": True,
            "placeholder_repair": True,
            "attention_mask_repair": True,
            "training_label_repair": True,
            "prefill_cache_position_repair": True,
            "decode_input_repair": True,
        }
    )

    def __init__(
        self,
        model: Any,
        *,
        _contract_verifier: Callable[[Any], None] | None = None,
    ) -> None:
        self.model = model
        self.core = getattr(model, "model", None)
        self.visual = getattr(self.core, "visual", None)
        self.language_model = getattr(self.core, "language_model", None)
        (_contract_verifier or _verify_pinned_video_contract)(model)
        self._contract_mode = (
            "pinned_transformers_4.57.3"
            if _contract_verifier is None
            else "injected_test_double"
        )
        self._callback: Callable[[Qwen25VLPackedVideoRequest], Any] | None = None
        self._signal_provider: Any | None = None
        self._projector_position = "after"
        self._preserve_projected_values = False
        self._callback_token = 0
        self._next_token = 1
        self._patches: list[_InstancePatch] = []
        self._attached_attribute: str | None = None
        self._invocation_counter = 0
        self._callback_count = 0
        self._latest_splice: dict[str, Any] | None = None
        self._latest_generation: dict[str, Any] | None = None
        self._latest_metok: list[dict[str, Any]] | None = None
        self._latest_vtp: dict[str, Any] | None = None

        suffix = str(id(self))
        self._request_metadata: ContextVar[Mapping[str, Any] | None] = ContextVar(
            f"qwen25_video_request_metadata_{suffix}", default=None
        )
        self._active_source: ContextVar[_ActiveSource | None] = ContextVar(
            f"qwen25_video_active_source_{suffix}", default=None
        )
        self._active_splice: ContextVar[_ActiveSplice | None] = ContextVar(
            f"qwen25_video_active_splice_{suffix}", default=None
        )
        self._selected_video_mask: ContextVar[Any | None] = ContextVar(
            f"qwen25_video_selected_mask_{suffix}", default=None
        )
        self._prompt_splice: ContextVar[Qwen25VLVideoPromptSpliceResult | None] = (
            ContextVar(f"qwen25_video_prompt_splice_{suffix}", default=None)
        )
        self._training_labels: ContextVar[Any | None] = ContextVar(
            f"qwen25_video_training_labels_{suffix}", default=None
        )
        self._vtp_rope_deltas: ContextVar[Any | None] = ContextVar(
            f"qwen25_video_vtp_rope_deltas_{suffix}", default=None
        )
        self._generation: ContextVar[_GenerationLifecycle | None] = ContextVar(
            f"qwen25_video_generation_{suffix}", default=None
        )

        self._original_model_forward: Callable[..., Any] | None = None
        self._original_generate: Callable[..., Any] | None = None
        self._original_core_forward: Callable[..., Any] | None = None
        self._original_get_video_features: Callable[..., Any] | None = None
        self._original_get_placeholder_mask: Callable[..., Any] | None = None
        self._original_language_forward: Callable[..., Any] | None = None

    @property
    def patched(self) -> bool:
        return bool(self._patches)

    def available_sites(self) -> tuple[str, ...]:
        return (QWEN25_VL_VIDEO_SPLICE_SITE,)

    def register_callback(
        self,
        site: str,
        callback: Callable[[Qwen25VLPackedVideoRequest], Any],
        *,
        signal_provider: Any | None = None,
        projector_position: str = "after",
        preserve_projected_values: bool = False,
    ) -> Qwen25VLVideoCallbackHandle:
        if site != QWEN25_VL_VIDEO_SPLICE_SITE:
            raise BackendIntegrationError(
                f"Unknown Qwen2.5-VL video provider site {site!r}."
            )
        if projector_position not in {"before", "after"}:
            raise BackendIntegrationError("Qwen2.5-VL video projector_position must be before or after.")
        if not callable(callback):
            raise BackendIntegrationError(
                "Qwen2.5-VL video provider callback must be callable."
            )
        if self._callback is not None:
            raise BackendIntegrationError(
                "Qwen2.5-VL video splice already has an active callback."
            )
        if signal_provider is not None and not all(
            callable(getattr(signal_provider, name, None))
            for name in ("capture_context", "payload_signals")
        ):
            raise BackendIntegrationError(
                "Qwen2.5-VL video signal provider has an incomplete capture contract."
            )
        if not self.patched:
            self._install_patches()
        token = self._next_token
        self._next_token += 1
        self._callback = callback
        self._signal_provider = signal_provider
        self._projector_position = projector_position
        self._preserve_projected_values = preserve_projected_values
        self._callback_token = token
        return Qwen25VLVideoCallbackHandle(self, token)

    @contextmanager
    def request_context(self, metadata: Mapping[str, Any]) -> Iterator[None]:
        if not isinstance(metadata, Mapping):
            raise TypeError("Qwen2.5-VL video request metadata must be a mapping")
        token = self._request_metadata.set(dict(metadata))
        try:
            yield
        finally:
            self._request_metadata.reset(token)

    def runtime_metadata(self) -> dict[str, Any]:
        return {
            "provider": "transformers_qwen25_vl_video_instance_bridge",
            "bridge_api_version": self.api_version,
            "contract_mode": self._contract_mode,
            "patched": self.patched,
            "active_sites": (
                [QWEN25_VL_VIDEO_SPLICE_SITE] if self._callback is not None else []
            ),
            "projector_position": self._projector_position,
            "dense_projected_reference_required": self._projector_position == "before"
                and bool(getattr(self._signal_provider, "requires_projected_reference", False)),
            "target_site": "video_pre_vision_merger" if self._projector_position == "before" else QWEN25_VL_VIDEO_SPLICE_SITE,
            "callback_count": self._callback_count,
            "capabilities": dict(self.capabilities),
            "latest_splice": self._latest_splice,
            "latest_generation": self._latest_generation,
            "latest_metok": self._latest_metok,
            "latest_vtp": self._latest_vtp,
            "signal_provider": (
                self._signal_provider.runtime_metadata()
                if self._signal_provider is not None
                and callable(getattr(self._signal_provider, "runtime_metadata", None))
                else None
            ),
            "support_activation_claimed": False,
            "real_checkpoint_verified": False,
        }

    def apply_metok_pruning_layout(self, result: METokPruningResult) -> None:
        """Repair native MRoPE, labels, and decode metadata for one METok stage."""

        if not isinstance(result, METokPruningResult):
            raise TypeError("result must be a METokPruningResult")
        splice = self._active_splice.get()
        prompt = self._prompt_splice.get()
        if splice is None or prompt is None:
            raise BackendIntegrationError(
                "Qwen2.5-VL METok must run inside an active video prompt"
            )
        if result.stage_index == 0:
            self._latest_metok = []
        records = self._latest_metok
        if records is None:
            raise InvariantError("Qwen2.5-VL METok stages began out of order")
        expected_sequence = (
            int(prompt.inputs_embeds.shape[1])
            if not records
            else int(records[-1]["result_sequence_length"])
        )
        expected_visual = (
            splice.result.result_count
            if not records
            else int(records[-1]["result_visual_count"])
        )
        if result.sequence_layout_delta.source_count != expected_sequence:
            raise InvariantError("Qwen2.5-VL METok sequence length changed")
        if result.visual_layout_delta.source_count != expected_visual:
            raise InvariantError("Qwen2.5-VL METok visual layout changed")
        torch = _torch()
        current_keep = torch.tensor(
            result.kept_sequence_indices,
            dtype=torch.long,
            device=prompt.inputs_embeds.device,
        )
        cumulative_keep = torch.tensor(
            result.cumulative_prompt_keep_indices,
            dtype=torch.long,
            device=prompt.position_ids.device,
        )
        if records:
            source_cumulative_keep = tuple(
                int(item) for item in records[-1]["cumulative_prompt_keep_indices"]
            )
            source_visual_sequence_indices = tuple(
                int(item) for item in records[-1]["result_visual_sequence_indices"]
            )
        else:
            source_cumulative_keep = tuple(range(expected_sequence))
            source_visual_sequence_indices = tuple(
                int(item) for item in prompt.plan.result_placeholder_positions
            )
        if (
            len(source_cumulative_keep) != expected_sequence
            or len(source_visual_sequence_indices) != expected_visual
        ):
            raise InvariantError("Qwen2.5-VL METok cumulative M-RoPE layout changed")
        source_keep = torch.tensor(
            source_cumulative_keep,
            dtype=torch.long,
            device=prompt.position_ids.device,
        )
        source_positions = prompt.position_ids.index_select(-1, source_keep)
        compact_positions = prompt.position_ids.index_select(-1, cumulative_keep)
        compact_mask = (
            None
            if prompt.attention_mask is None
            else prompt.attention_mask.index_select(
                1,
                cumulative_keep.to(prompt.attention_mask.device),
            )
        )
        rope_deltas = _exact_rope_deltas(compact_positions, compact_mask)
        source_rotary = (
            source_positions[1:]
            if int(source_positions.shape[0]) == 4
            else source_positions
        )
        compact_rotary = (
            compact_positions[1:]
            if int(compact_positions.shape[0]) == 4
            else compact_positions
        )
        source_visual_mrope = _mrope_columns(
            source_rotary,
            source_visual_sequence_indices,
        )
        current_to_result = {
            int(source_index): result_index
            for result_index, source_index in enumerate(result.kept_sequence_indices)
        }
        result_visual_sequence_indices = tuple(
            current_to_result[source_visual_sequence_indices[index]]
            for index in result.kept_visual_indices
        )
        result_visual_mrope = _mrope_columns(
            compact_rotary,
            result_visual_sequence_indices,
        )
        expected_result_visual_mrope = tuple(
            source_visual_mrope[index] for index in result.kept_visual_indices
        )
        if result_visual_mrope != expected_result_visual_mrope:
            raise InvariantError(
                "Qwen2.5-VL METok changed gathered native M-RoPE anchors"
            )
        if compact_mask is None:
            active_position_count = int(compact_rotary.shape[-1])
            active_rotary = compact_rotary.reshape(-1)
        else:
            active_mask = compact_mask[0].bool()
            active_position_count = int(active_mask.sum().item())
            active_rotary = compact_rotary[:, 0, active_mask].reshape(-1)
        max_active_rotary_position = int(active_rotary.max().item())
        expected_rope_delta = max_active_rotary_position + 1 - active_position_count
        if _tensor_int_values(rope_deltas) != [expected_rope_delta]:
            raise InvariantError("Qwen2.5-VL METok published an inexact rope delta")
        self._vtp_rope_deltas.set(rope_deltas)
        labels = self._training_labels.get()
        if labels is not None:
            dropped = tuple(result.sequence_layout_delta.dropped_indices)
            if dropped and bool(
                (labels[0, list(dropped)] != QWEN25_VL_IGNORE_INDEX).any().item()
            ):
                raise InvariantError(
                    "Qwen2.5-VL METok may drop ignored visual labels only"
                )
            self._training_labels.set(
                labels.index_select(1, current_keep.to(labels.device))
            )
        removed_source = tuple(
            splice.plan.retained_placeholder_positions[index]
            for index in result.dropped_projector_indices
        )
        lifecycle = self._generation.get()
        if lifecycle is not None:
            lifecycle.vtp_removed_source_positions = removed_source
            lifecycle.vtp_result_sequence_length = (
                result.sequence_layout_delta.result_count
            )
        records.append(
            {
                "boundary_layer": result.boundary_layer,
                "application_position": result.metadata.get("application_position", {"boundary": "inter_layer", "index": result.boundary_layer}),
                "stage_index": result.stage_index,
                "source_visual_count": result.visual_layout_delta.source_count,
                "result_visual_count": result.visual_layout_delta.result_count,
                "source_sequence_length": result.sequence_layout_delta.source_count,
                "result_sequence_length": result.sequence_layout_delta.result_count,
                "dropped_source_placeholder_positions": removed_source,
                "source_visual_sequence_indices": source_visual_sequence_indices,
                "result_visual_sequence_indices": result_visual_sequence_indices,
                "source_visual_mrope": source_visual_mrope,
                "result_visual_mrope": result_visual_mrope,
                "cumulative_prompt_keep_indices": tuple(
                    int(item) for item in result.cumulative_prompt_keep_indices
                ),
                "position_policy": "gather_native_mrope",
                "active_position_count": active_position_count,
                "max_active_rotary_position": max_active_rotary_position,
                "expected_rope_delta": expected_rope_delta,
                "rope_delta_exact": True,
                "rope_deltas": _tensor_int_values(rope_deltas),
            }
        )

    def apply_prunevid_vtp_layout(self, result: PruneVidVTPResult) -> None:
        """Repair labels and pending generate metadata for one VTP boundary."""

        if not isinstance(result, PruneVidVTPResult):
            raise TypeError("result must be a PruneVidVTPResult")
        splice = self._active_splice.get()
        prompt = self._prompt_splice.get()
        if splice is None or prompt is None:
            raise BackendIntegrationError(
                "Qwen2.5-VL VTP must run inside an active video prompt"
            )
        source_length = int(prompt.inputs_embeds.shape[1])
        if result.sequence_layout_delta.source_count != source_length:
            raise InvariantError("Qwen2.5-VL VTP sequence length changed")
        if result.visual_layout_delta.source_count != splice.result.result_count:
            raise InvariantError("Qwen2.5-VL VTP visual layout changed")
        torch = _torch()
        keep = torch.tensor(
            result.kept_sequence_indices,
            dtype=torch.long,
            device=prompt.inputs_embeds.device,
        )
        compact_positions = prompt.position_ids.index_select(
            -1,
            keep.to(prompt.position_ids.device),
        )
        compact_mask = (
            None
            if prompt.attention_mask is None
            else prompt.attention_mask.index_select(
                1,
                keep.to(prompt.attention_mask.device),
            )
        )
        source_visual_positions = tuple(
            int(item) for item in splice.plan.result_placeholder_positions
        )
        source_rotary = (
            prompt.position_ids[1:]
            if int(prompt.position_ids.shape[0]) == 4
            else prompt.position_ids
        )
        source_visual_mrope = _mrope_columns(
            source_rotary,
            source_visual_positions,
        )
        expected_result_mrope = tuple(
            source_visual_mrope[index] for index in result.kept_visual_indices
        )
        compact_index_by_source = {
            int(source_index): compact_index
            for compact_index, source_index in enumerate(result.kept_sequence_indices)
        }
        result_visual_sequence_indices = tuple(
            compact_index_by_source[source_visual_positions[index]]
            for index in result.kept_visual_indices
        )
        compact_rotary = (
            compact_positions[1:]
            if int(compact_positions.shape[0]) == 4
            else compact_positions
        )
        result_visual_mrope = _mrope_columns(
            compact_rotary,
            result_visual_sequence_indices,
        )
        if result_visual_mrope != expected_result_mrope:
            raise InvariantError(
                "Qwen2.5-VL PruneVid VTP changed gathered native M-RoPE anchors"
            )
        rope_deltas = _exact_rope_deltas(compact_positions, compact_mask)
        if compact_mask is None:
            active_position_count = int(compact_rotary.shape[-1])
            active_rotary = compact_rotary.reshape(-1)
        else:
            active_mask = compact_mask[0].bool()
            active_position_count = int(active_mask.sum().item())
            active_rotary = compact_rotary[:, 0, active_mask].reshape(-1)
        max_active_rotary_position = int(active_rotary.max().item())
        expected_rope_delta = max_active_rotary_position + 1 - active_position_count
        if _tensor_int_values(rope_deltas) != [expected_rope_delta]:
            raise InvariantError(
                "Qwen2.5-VL PruneVid VTP published an inexact rope delta"
            )
        self._vtp_rope_deltas.set(rope_deltas)
        labels = self._training_labels.get()
        if labels is not None:
            dropped = tuple(result.sequence_layout_delta.dropped_indices)
            if dropped and bool(
                (labels[0, list(dropped)] != QWEN25_VL_IGNORE_INDEX).any().item()
            ):
                raise InvariantError(
                    "Qwen2.5-VL PruneVid VTP may drop ignored visual labels only"
                )
            self._training_labels.set(labels.index_select(1, keep.to(labels.device)))
        lifecycle = self._generation.get()
        removed_source = tuple(
            splice.plan.retained_placeholder_positions[index]
            for index in result.dropped_visual_indices
        )
        if lifecycle is not None:
            lifecycle.vtp_removed_source_positions = removed_source
            lifecycle.vtp_result_sequence_length = (
                result.sequence_layout_delta.result_count
            )
        self._latest_vtp = {
            "selected_layer": result.selected_layer,
            "application_position": result.metadata.get("application_position", {"boundary": "inter_layer", "index": result.selected_layer + 1}),
            "source_visual_count": result.visual_layout_delta.source_count,
            "result_visual_count": result.visual_layout_delta.result_count,
            "source_sequence_length": source_length,
            "result_sequence_length": result.sequence_layout_delta.result_count,
            "dropped_source_placeholder_positions": removed_source,
            "kept_visual_indices": result.kept_visual_indices,
            "source_visual_sequence_indices": source_visual_positions,
            "result_visual_sequence_indices": result_visual_sequence_indices,
            "source_visual_mrope": source_visual_mrope,
            "result_visual_mrope": result_visual_mrope,
            "position_policy": "gather_native_mrope",
            "active_position_count": active_position_count,
            "max_active_rotary_position": max_active_rotary_position,
            "expected_rope_delta": expected_rope_delta,
            "rope_delta_exact": True,
            "rope_deltas": _tensor_int_values(rope_deltas),
        }

    def detach(self) -> None:
        if self._callback is not None:
            raise BackendIntegrationError(
                "Cannot detach the Qwen2.5-VL video bridge while a callback is active."
            )
        self._restore_patches()
        if (
            self._attached_attribute is not None
            and getattr(self.model, self._attached_attribute, None) is self
        ):
            delattr(self.model, self._attached_attribute)
        if getattr(self.model, _PUBLIC_VIDEO_BRIDGE_ATTRIBUTE, None) is self:
            delattr(self.model, _PUBLIC_VIDEO_BRIDGE_ATTRIBUTE)
        self._attached_attribute = None

    def _remove_callback(self, token: int) -> None:
        if self._callback is None or token != self._callback_token:
            return
        self._callback = None
        self._signal_provider = None
        self._projector_position = "after"
        self._preserve_projected_values = False
        self._callback_token = 0
        self._restore_patches()

    def _install_patches(self) -> None:
        for attribute in _IMAGE_PROVIDER_ATTRIBUTES:
            if getattr(self.model, attribute, None) is not None:
                raise BackendIntegrationError(
                    "Qwen2.5-VL image and video provider bridges cannot patch the "
                    "same model instance simultaneously."
                )
        self._original_model_forward = getattr(self.model, "forward", None)
        self._original_generate = getattr(self.model, "generate", None)
        self._original_core_forward = getattr(self.core, "forward", None)
        self._original_get_video_features = getattr(
            self.core, "get_video_features", None
        )
        self._original_get_placeholder_mask = getattr(
            self.core, "get_placeholder_mask", None
        )
        self._original_language_forward = getattr(self.language_model, "forward", None)
        originals = (
            self._original_model_forward,
            self._original_generate,
            self._original_core_forward,
            self._original_get_video_features,
            self._original_get_placeholder_mask,
            self._original_language_forward,
        )
        if not all(callable(value) for value in originals):
            raise BackendIntegrationError(
                "Pinned Qwen2.5-VL video bridge could not resolve native methods."
            )
        patches: list[_InstancePatch] = []
        try:
            patches.append(
                self._patch_bound(
                    self.model,
                    "forward",
                    self._model_forward_replacement(),
                    self._original_model_forward,
                )
            )
            patches.append(
                self._patch_bound(
                    self.model,
                    "generate",
                    self._generate_replacement(),
                    self._original_generate,
                )
            )
            patches.append(
                self._patch_bound(
                    self.core,
                    "forward",
                    self._core_forward_replacement(),
                    self._original_core_forward,
                )
            )
            patches.append(
                self._patch_bound(
                    self.core,
                    "get_video_features",
                    self._get_video_features_replacement(),
                    self._original_get_video_features,
                )
            )
            patches.append(
                self._patch_bound(
                    self.core,
                    "get_placeholder_mask",
                    self._get_placeholder_mask_replacement(),
                    self._original_get_placeholder_mask,
                )
            )
            patches.append(
                self._patch_bound(
                    self.language_model,
                    "forward",
                    self._language_forward_replacement(),
                    self._original_language_forward,
                )
            )
        except BaseException:
            for patch in reversed(patches):
                patch.restore()
            raise
        self._patches = patches

    def _patch_bound(
        self,
        owner: Any,
        name: str,
        replacement: Callable[..., Any],
        original: Callable[..., Any],
    ) -> _InstancePatch:
        preserve_bound_method_signature(replacement, original)
        return _InstancePatch.apply(owner, name, MethodType(replacement, owner))

    def _restore_patches(self) -> None:
        for patch in reversed(self._patches):
            patch.restore()
        self._patches.clear()
        self._original_model_forward = None
        self._original_generate = None
        self._original_core_forward = None
        self._original_get_video_features = None
        self._original_get_placeholder_mask = None
        self._original_language_forward = None

    def _model_forward_replacement(self) -> Callable[..., Any]:
        original = self._original_model_forward
        assert original is not None
        signature = inspect.signature(original)

        def replacement(_model: Any, *args: Any, **kwargs: Any) -> Any:
            callback = self._callback
            if callback is None:
                return original(*args, **kwargs)
            bound = signature.bind_partial(*args, **kwargs)
            call = bound.arguments
            lifecycle = self._generation.get()
            if (
                lifecycle is not None
                and lifecycle.prefill_complete
                and call.get("past_key_values") is not None
            ):
                # Transformers retains grid/timing metadata on cached decode
                # calls after correctly dropping the one-shot pixel tensor.
                # The completed lifecycle, rather than partial media kwargs,
                # is the authoritative boundary between prefill and decode.
                self._repair_decode_call(call, lifecycle)
                return original(*bound.args, **bound.kwargs)
            has_video = (
                call.get("pixel_values_videos") is not None
                or call.get("video_grid_thw") is not None
                or call.get("second_per_grid_ts") is not None
            )
            if not has_video:
                if lifecycle is not None and call.get("past_key_values") is not None:
                    self._repair_decode_call(call, lifecycle)
                return original(*bound.args, **bound.kwargs)

            if (
                call.get("pixel_values") is not None
                or call.get("image_grid_thw") is not None
            ):
                raise BackendIntegrationError(
                    "Compressed Qwen2.5-VL video requests cannot mix image media."
                )
            input_ids = call.get("input_ids")
            if _tensor_shape(input_ids)[:1] != (1,):
                raise BackendIntegrationError(
                    "Qwen2.5-VL video compression requires batch size one."
                )
            if call.get("inputs_embeds") is not None:
                raise BackendIntegrationError(
                    "Qwen2.5-VL video compression requires source input_ids."
                )
            labels = call.get("labels")
            use_cache = call.get("use_cache")
            if lifecycle is None and labels is not None and use_cache is None:
                # Teacher forcing uses the shortened prompt without a cache,
                # including through the public runner's native input mapping.
                call["use_cache"] = False
                use_cache = False
            required_video_fields = {
                "pixel_values_videos": call.get("pixel_values_videos"),
                "video_grid_thw": call.get("video_grid_thw"),
                "second_per_grid_ts": call.get("second_per_grid_ts"),
            }
            missing_video_fields = tuple(
                name for name, value in required_video_fields.items() if value is None
            )
            if missing_video_fields:
                raise BackendIntegrationError(
                    "Compressed Qwen2.5-VL video input is missing native field(s): "
                    + ", ".join(missing_video_fields)
                    + "."
                )
            past_key_values = call.get("past_key_values")
            if past_key_values is not None:
                cache_length = _cache_sequence_length(past_key_values)
                if lifecycle is None or cache_length != 0:
                    raise BackendIntegrationError(
                        "Qwen2.5-VL compressed video media is valid only with an "
                        "empty prefill cache."
                    )
            if lifecycle is None and use_cache is not False:
                raise BackendIntegrationError(
                    "Direct Qwen2.5-VL compressed forward requires use_cache=False."
                )
            if lifecycle is not None and use_cache is not True:
                raise BackendIntegrationError(
                    "Qwen2.5-VL compressed generation prefill requires use_cache=True."
                )
            if lifecycle is not None and labels is not None:
                raise BackendIntegrationError(
                    "Qwen2.5-VL compressed generation does not accept labels."
                )
            if labels is not None and call.get("return_dict") is False:
                raise BackendIntegrationError(
                    "Compressed Qwen2.5-VL video training requires return_dict=True."
                )

            label_token = self._training_labels.set(labels)
            if labels is not None:
                call["labels"] = None
            try:
                output = original(*bound.args, **bound.kwargs)
                if labels is None:
                    return output
                repaired = self._training_labels.get()
                if repaired is None:
                    raise InvariantError(
                        "Qwen2.5-VL compressed video labels disappeared."
                    )
                return attach_shifted_causal_loss(
                    output,
                    repaired,
                    operation="Qwen2.5-VL video compression",
                )
            finally:
                self._training_labels.reset(label_token)

        return replacement

    def _generate_replacement(self) -> Callable[..., Any]:
        original = self._original_generate
        assert original is not None

        def replacement(_model: Any, *args: Any, **kwargs: Any) -> Any:
            if self._callback is None or not any(
                kwargs.get(name) is not None
                for name in (
                    "pixel_values_videos",
                    "video_grid_thw",
                    "second_per_grid_ts",
                )
            ):
                return original(*args, **kwargs)
            if self._generation.get() is not None:
                raise BackendIntegrationError(
                    "Nested Qwen2.5-VL compressed generation is unsupported."
                )
            lifecycle = _GenerationLifecycle(
                request_id=f"generate-{self._invocation_counter + 1}"
            )
            token = self._generation.set(lifecycle)
            try:
                output = original(*args, **kwargs)
                if not lifecycle.prefill_complete:
                    raise InvariantError(
                        "Qwen2.5-VL generate returned before compressed prefill."
                    )
                return output
            finally:
                self._latest_generation = {
                    "request_id": lifecycle.request_id,
                    "prefill_complete": lifecycle.prefill_complete,
                    "source_sequence_length": lifecycle.source_sequence_length,
                    "result_sequence_length": lifecycle.result_sequence_length,
                    "removed_count": lifecycle.removed_count,
                    "decode_repairs": lifecycle.decode_repairs,
                    "last_decode_record": lifecycle.last_decode_record,
                }
                self._generation.reset(token)

        return replacement

    def _core_forward_replacement(self) -> Callable[..., Any]:
        original = self._original_core_forward
        assert original is not None
        signature = inspect.signature(original)

        def replacement(_core: Any, *args: Any, **kwargs: Any) -> Any:
            if self._callback is None:
                return original(*args, **kwargs)
            bound = signature.bind_partial(*args, **kwargs)
            call = bound.arguments
            if call.get("pixel_values_videos") is None:
                return original(*bound.args, **bound.kwargs)
            input_ids = call.get("input_ids")
            if _tensor_shape(input_ids)[:1] != (1,):
                raise BackendIntegrationError(
                    "Qwen2.5-VL compressed video core requires one input_ids row."
                )
            grid_rows = _native_rows(call.get("video_grid_thw"))
            if len(grid_rows) != 1:
                raise BackendIntegrationError(
                    "Qwen2.5-VL video compression supports exactly one video."
                )
            timing = _single_float(call.get("second_per_grid_ts"), "second_per_grid_ts")
            merge_size = int(getattr(self.visual, "spatial_merge_size", 0))
            grid = Qwen25VLVideoGrid.from_native(
                grid_rows[0],
                spatial_merge_size=merge_size,
                second_per_grid_seconds=timing,
            )
            video_token_id = int(getattr(self.core.config, "video_token_id"))
            placeholders = (
                (input_ids[0] == video_token_id).nonzero(as_tuple=False).flatten()
            )
            positions = tuple(int(value) for value in placeholders.tolist())
            if len(positions) != grid.token_count:
                raise BackendIntegrationError(
                    "Qwen2.5-VL video placeholder count differs from source grid."
                )
            if positions != tuple(range(positions[0], positions[-1] + 1)):
                raise BackendIntegrationError(
                    "One Qwen2.5-VL video must occupy one contiguous placeholder span."
                )
            attention_mask = call.get("attention_mask")
            if attention_mask is not None:
                active = attention_mask[0, placeholders].bool()
                if not bool(active.all().item()):
                    raise BackendIntegrationError(
                        "Qwen2.5-VL active video placeholders cannot be padded."
                    )
            metadata = dict(self._request_metadata.get() or {})
            media_id = _single_video_media_id(
                metadata,
                fallback=f"video-{self._invocation_counter + 1}",
            )
            source = _ActiveSource(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pixel_values_videos=call.get("pixel_values_videos"),
                video_grid_thw=call.get("video_grid_thw"),
                second_per_grid_ts=call.get("second_per_grid_ts"),
                grid=grid,
                source_placeholder_positions=positions,
                media_id=media_id,
                conditioning_text=single_video_conditioning_text(metadata),
            )
            source_token = self._active_source.set(source)
            splice_token = self._active_splice.set(None)
            mask_token = self._selected_video_mask.set(None)
            prompt_token = self._prompt_splice.set(None)
            rope_token = self._vtp_rope_deltas.set(None)
            try:
                output = original(*bound.args, **bound.kwargs)
                splice = self._active_splice.get()
                prompt = self._prompt_splice.get()
                if splice is None or prompt is None:
                    raise InvariantError(
                        "Qwen2.5-VL video prefill missed splice legalization."
                    )
                rope_deltas = self._vtp_rope_deltas.get()
                if rope_deltas is None:
                    rope_deltas = prompt.rope_deltas
                _set_output_field(output, "rope_deltas", rope_deltas)
                self.core.rope_deltas = rope_deltas
                self._callback_count += 1
                self._latest_splice = {
                    "invocation_id": splice.request.invocation_id,
                    "target_site": "video_pre_vision_merger" if self._projector_position == "before" else QWEN25_VL_VIDEO_SPLICE_SITE,
                    "dense_projected_reference_tokens": splice.request.payload.metadata.get("dense_projected_reference_tokens", 0),
                    "media_id": splice.request.media_id,
                    "source_video_grid_thw": source.grid.source_grid_thw,
                    "second_per_grid_seconds": source.grid.second_per_grid_seconds,
                    **prompt.record(),
                    "rope_deltas": _tensor_int_values(rope_deltas),
                }
                lifecycle = self._generation.get()
                if lifecycle is not None:
                    lifecycle.prefill_complete = True
                    lifecycle.source_sequence_length = int(input_ids.shape[1])
                    lifecycle.result_sequence_length = (
                        lifecycle.vtp_result_sequence_length
                        if lifecycle.vtp_result_sequence_length is not None
                        else int(prompt.input_ids.shape[1])
                    )
                    lifecycle.removed_placeholder_positions = tuple(
                        sorted(
                            {
                                *prompt.plan.removed_placeholder_positions,
                                *lifecycle.vtp_removed_source_positions,
                            }
                        )
                    )
                return output
            finally:
                self._vtp_rope_deltas.reset(rope_token)
                self._prompt_splice.reset(prompt_token)
                self._selected_video_mask.reset(mask_token)
                self._active_splice.reset(splice_token)
                self._active_source.reset(source_token)

        return replacement

    def _get_video_features_replacement(self) -> Callable[..., Any]:
        original = self._original_get_video_features
        assert original is not None
        signature = inspect.signature(original)

        def replacement(_core: Any, *args: Any, **kwargs: Any) -> Any:
            source = self._active_source.get()
            if self._callback is None or source is None:
                return original(*args, **kwargs)
            bound = signature.bind_partial(*args, **kwargs)
            signal_provider = self._signal_provider
            capture = (
                nullcontext(None)
                if signal_provider is None
                else signal_provider.capture_context(
                    grid_thw=source.video_grid_thw,
                )
            )
            merger_patch = None
            unit = int(self.visual.spatial_merge_size) ** 2
            try:
                if self._projector_position == "before":
                    def raw_groups(_merger: Any, hidden_states: Any) -> Any:
                        # Keep complete spatial groups. The native visual forward
                        # still applies its window inverse and get_video_features
                        # still splits the original per-video group counts.
                        return hidden_states.reshape(-1, unit * hidden_states.shape[-1])

                    merger_patch = _InstancePatch.apply(
                        self.visual.merger, "forward", MethodType(raw_groups, self.visual.merger)
                    )
                with capture as signal_capture:
                    features_by_video = original(*bound.args, **bound.kwargs)
            finally:
                if merger_patch is not None:
                    merger_patch.restore()
            if (
                not isinstance(features_by_video, (tuple, list))
                or len(features_by_video) != 1
            ):
                raise BackendIntegrationError(
                    "Pinned Qwen2.5-VL get_video_features must return one feature chunk."
                )
            features = features_by_video[0]
            self._invocation_counter += 1
            invocation_id = f"video-splice-{self._invocation_counter}"
            payload = source.grid.payload(
                features,
                media_id=source.media_id,
                signals={
                    "qwen25_vl.prompt_input_ids": source.input_ids,
                    "qwen25_vl.pixel_values_videos": source.pixel_values_videos,
                    "text_context": source.conditioning_text,
                    **(
                        {}
                        if signal_provider is None
                        else dict(
                            signal_provider.payload_signals(
                                signal_capture,
                                token_count=source.grid.token_count,
                                coordinates=source.grid.coordinates,
                            )
                        )
                    ),
                },
                metadata={"invocation_id": invocation_id,
                    "target_site": "video_pre_vision_merger" if self._projector_position == "before" else QWEN25_VL_VIDEO_SPLICE_SITE},
            )
            enrich_payload = (
                None
                if signal_provider is None
                else getattr(signal_provider, "enrich_payload", None)
            )
            reference_required = self._projector_position == "before" and (
                self._preserve_projected_values or getattr(signal_provider, "requires_projected_reference", False))
            reference_payload = payload
            if reference_required:
                # Native raster order is already restored; this independent
                # complete scoring projection needs no second window inverse.
                with _torch().no_grad():
                    reference = self.visual.merger(features.reshape(-1, features.shape[-1] // unit))
                reference_payload = replace(payload, features=reference,
                    metadata={**dict(payload.metadata),
                        "dense_projected_reference_tokens": source.grid.token_count})
            if callable(enrich_payload):
                payload = enrich_payload(reference_payload, signal_capture)
                if not isinstance(payload, VideoTokenPayload):
                    raise BackendIntegrationError(
                        "Qwen2.5-VL video signal provider returned an invalid payload"
                    )
            else:
                payload = reference_payload
            if reference_required:
                payload = replace(payload, features=features, signals={**payload.signals,
                    **({"projector_reference_features": reference} if self._preserve_projected_values else {})})
            request = Qwen25VLPackedVideoRequest(
                invocation_id=invocation_id,
                media_id=source.media_id,
                payload=payload,
                grid=source.grid,
                source_placeholder_positions=source.source_placeholder_positions,
                prompt_input_ids=source.input_ids,
                source_attention_mask=source.attention_mask,
                source_video_grid_thw=source.video_grid_thw,
                source_second_per_grid_ts=source.second_per_grid_ts,
            )
            response = self._callback(request)
            if not isinstance(response, VideoCompressionResult):
                raise BackendIntegrationError(
                    "Qwen2.5-VL video callback must return VideoCompressionResult."
                )
            result_shape = _feature_shape(response.features)
            source_shape = _feature_shape(features)
            if len(result_shape) != 2 or result_shape[1:] != source_shape[1:]:
                raise InvariantError("Qwen2.5-VL compressed video hidden size changed.")
            if self._projector_position == "before":
                # Callback values are already in raster order. Apply the original
                # LN and MLP to K complete groups without the old N-row inverse.
                compact_patches = response.features.reshape(-1, source_shape[1] // unit)
                response = replace(response, features=self.visual.merger(compact_patches))
            plan = plan_qwen25_vl_video_splice(
                request,
                response,
                sequence_length=int(source.input_ids.shape[1]),
            )
            self._active_splice.set(_ActiveSplice(request, response, plan))
            return (response.features,)

        return replacement

    def _get_placeholder_mask_replacement(self) -> Callable[..., Any]:
        original = self._original_get_placeholder_mask
        assert original is not None
        signature = inspect.signature(original)

        def replacement(_core: Any, *args: Any, **kwargs: Any) -> Any:
            splice = self._active_splice.get()
            if splice is None:
                return original(*args, **kwargs)
            bound = signature.bind_partial(*args, **kwargs)
            call = bound.arguments
            input_ids = call.get("input_ids")
            inputs_embeds = call.get("inputs_embeds")
            supplied_video_features = call.get("video_features")
            call["video_features"] = None
            image_mask, dense_video_mask = original(*bound.args, **bound.kwargs)
            dense = dense_video_mask[..., 0].bool()
            if _tensor_shape(dense) != _tensor_shape(input_ids):
                raise InvariantError(
                    "Qwen2.5-VL dense video placeholder mask changed shape."
                )
            positions = tuple(
                int(value)
                for value in dense[0].nonzero(as_tuple=False).flatten().tolist()
            )
            if positions != splice.plan.source_placeholder_positions:
                raise InvariantError(
                    "Qwen2.5-VL native video placeholder mask changed source order."
                )
            torch = _torch()
            selected = torch.zeros_like(dense, dtype=torch.bool)
            selected[0, list(splice.plan.retained_placeholder_positions)] = True
            expanded = selected.unsqueeze(-1).expand_as(inputs_embeds)
            if supplied_video_features is None or (
                inputs_embeds[expanded].numel() != supplied_video_features.numel()
            ):
                raise InvariantError(
                    "Qwen2.5-VL compressed placeholders do not match video features."
                )
            self._selected_video_mask.set(selected)
            return image_mask, expanded

        return replacement

    def _language_forward_replacement(self) -> Callable[..., Any]:
        original = self._original_language_forward
        assert original is not None
        signature = inspect.signature(original)

        def replacement(_language: Any, *args: Any, **kwargs: Any) -> Any:
            splice = self._active_splice.get()
            source = self._active_source.get()
            selected = self._selected_video_mask.get()
            if splice is None or source is None or selected is None:
                return original(*args, **kwargs)
            bound = signature.bind_partial(*args, **kwargs)
            call = bound.arguments
            prompt = legalize_qwen25_vl_video_prompt(
                input_ids=source.input_ids,
                inputs_embeds=call.get("inputs_embeds"),
                attention_mask=call.get("attention_mask"),
                position_ids=call.get("position_ids"),
                cache_position=call.get("cache_position"),
                labels=self._training_labels.get(),
                request=splice.request,
                compression_result=splice.result,
            )
            call["input_ids"] = None
            call["inputs_embeds"] = prompt.inputs_embeds
            call["attention_mask"] = prompt.attention_mask
            call["position_ids"] = prompt.position_ids
            call["cache_position"] = prompt.cache_position
            if self._training_labels.get() is not None:
                self._training_labels.set(prompt.labels)
            self._prompt_splice.set(prompt)
            return original(*bound.args, **bound.kwargs)

        return replacement

    def _repair_decode_call(
        self,
        call: MutableMapping[str, Any],
        lifecycle: _GenerationLifecycle,
    ) -> None:
        if not lifecycle.prefill_complete:
            return
        source_length = lifecycle.source_sequence_length
        result_length = lifecycle.result_sequence_length
        if source_length is None or result_length is None:
            raise InvariantError("Qwen2.5-VL generation lifecycle lost prompt lengths.")
        removed = lifecycle.removed_placeholder_positions
        removed_count = len(removed)
        cache_position = call.get("cache_position")
        cache_values = _tensor_int_values(cache_position)
        attention_mask = call.get("attention_mask")
        attention_shape = _tensor_shape(attention_mask)
        dense_cache = bool(cache_values and cache_values[0] >= source_length)
        dense_mask = bool(
            len(attention_shape) == 2
            and attention_shape[0] == 1
            and attention_shape[1] >= source_length
        )
        if not dense_cache and not dense_mask and removed_count:
            # Generation implementations may already follow the physical cache
            # length. In that case no second repair is allowed.
            lifecycle.decode_repairs += 1
            lifecycle.last_decode_record = {
                "input_cache_position": cache_values,
                "input_attention_length": (
                    attention_shape[1] if len(attention_shape) == 2 else None
                ),
                "already_compact": True,
            }
            return

        if dense_mask:
            if len(attention_shape) != 2 or attention_shape[0] != 1:
                raise InvariantError(
                    "Qwen2.5-VL decode attention_mask cannot be compacted safely."
                )
            removed_set = set(removed)
            keep = tuple(
                index
                for index in range(attention_shape[1])
                if index >= source_length or index not in removed_set
            )
            torch = _torch()
            keep_tensor = torch.tensor(
                keep,
                dtype=torch.long,
                device=attention_mask.device,
            )
            call["attention_mask"] = attention_mask.index_select(1, keep_tensor)
        if dense_cache and removed_count:
            call["cache_position"] = cache_position - removed_count
        position_ids = call.get("position_ids")
        if position_ids is not None and removed_count and (dense_cache or dense_mask):
            call["position_ids"] = position_ids - removed_count
        lifecycle.decode_repairs += 1
        lifecycle.last_decode_record = {
            "input_cache_position": cache_values,
            "output_cache_position": _tensor_int_values(call.get("cache_position")),
            "input_attention_length": (
                attention_shape[1] if len(attention_shape) == 2 else None
            ),
            "output_attention_length": (
                _tensor_shape(call.get("attention_mask"))[1]
                if len(_tensor_shape(call.get("attention_mask"))) == 2
                else None
            ),
            "removed_count": removed_count,
            "already_compact": False,
        }


def install_qwen25_vl_video_bridge(model: Any) -> PinnedQwen25VLVideoBridge:
    """Attach or return the exact video bridge on one Qwen2.5-VL instance."""

    for attribute in (_VIDEO_BRIDGE_ATTRIBUTE, _PUBLIC_VIDEO_BRIDGE_ATTRIBUTE):
        existing = getattr(model, attribute, None)
        if existing is not None:
            if isinstance(existing, PinnedQwen25VLVideoBridge):
                return existing
            raise BackendIntegrationError(
                f"Model already exposes an incompatible video bridge at {attribute}."
            )
    bridge = PinnedQwen25VLVideoBridge(model)
    setattr(model, _VIDEO_BRIDGE_ATTRIBUTE, bridge)
    setattr(model, _PUBLIC_VIDEO_BRIDGE_ATTRIBUTE, bridge)
    bridge._attached_attribute = _VIDEO_BRIDGE_ATTRIBUTE
    return bridge


@contextmanager
def qwen25_vl_video_request_context(
    model: Any,
    metadata: Mapping[str, Any],
) -> Iterator[None]:
    """Publish media identity only for the duration of one native request."""

    bridge = getattr(model, _VIDEO_BRIDGE_ATTRIBUTE, None)
    request_context = getattr(bridge, "request_context", None)
    if not callable(request_context):
        yield
        return
    with request_context(metadata):
        yield


def _verify_pinned_video_contract(model: Any) -> None:
    from .provider_bridge import (
        _validate_provider_version,
        _validate_qwen25_vl_structure,
    )

    core = getattr(model, "model", None)
    visual = getattr(core, "visual", None)
    language_model = getattr(core, "language_model", None)
    _validate_provider_version()
    _validate_qwen25_vl_structure(
        model,
        core=core,
        visual=visual,
        language_model=language_model,
    )


def _native_rows(value: Any) -> tuple[tuple[int, int, int], ...]:
    raw = value.tolist() if callable(getattr(value, "tolist", None)) else value
    if not isinstance(raw, (tuple, list)):
        raise BackendIntegrationError("video_grid_thw must be a rank-two sequence.")
    rows: list[tuple[int, int, int]] = []
    for row in raw:
        if not isinstance(row, (tuple, list)) or len(row) != 3:
            raise BackendIntegrationError(
                "video_grid_thw rows must contain exactly three integers."
            )
        if any(
            isinstance(item, bool) or not isinstance(item, Integral) for item in row
        ):
            raise BackendIntegrationError(
                "video_grid_thw rows must contain exactly three integers."
            )
        values = tuple(int(item) for item in row)
        if any(value <= 0 for value in values):
            raise BackendIntegrationError("video_grid_thw entries must be positive.")
        rows.append(values)
    return tuple(rows)


def _single_float(value: Any, name: str) -> float:
    raw = value.tolist() if callable(getattr(value, "tolist", None)) else value
    if isinstance(raw, (tuple, list)):
        if len(raw) != 1:
            raise BackendIntegrationError(f"{name} must contain one value.")
        raw = raw[0]
    if isinstance(raw, bool) or not isinstance(raw, (int, float)):
        raise BackendIntegrationError(f"{name} must contain one finite number.")
    try:
        result = float(raw)
    except (TypeError, ValueError) as exc:
        raise BackendIntegrationError(
            f"{name} must contain one finite number."
        ) from exc
    if not math.isfinite(result) or result <= 0:
        raise BackendIntegrationError(f"{name} must be finite and positive.")
    return result


def _single_video_media_id(
    metadata: Mapping[str, Any],
    *,
    fallback: str,
) -> str:
    if "video_media_ids" in metadata:
        raw = metadata["video_media_ids"]
        if isinstance(raw, (str, bytes)) or not isinstance(raw, Sequence):
            raise BackendIntegrationError(
                "Qwen2.5-VL video_media_ids must contain exactly one identity."
            )
        media_ids = tuple(raw)
        if len(media_ids) != 1 or not isinstance(media_ids[0], str):
            raise BackendIntegrationError(
                "Qwen2.5-VL video_media_ids must contain exactly one identity."
            )
        media_id = media_ids[0].strip()
        if not media_id:
            raise BackendIntegrationError(
                "Qwen2.5-VL video media identity must be non-empty."
            )
        return media_id
    sample_id = metadata.get("sample_id")
    if sample_id is not None and str(sample_id).strip():
        return str(sample_id)
    return fallback


def _feature_shape(value: Any) -> tuple[int, ...]:
    shape = getattr(value, "shape", None)
    if shape is not None:
        return tuple(int(item) for item in shape)
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise BackendIntegrationError("Qwen2.5-VL video features must be tensor-like.")
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
        raise BackendIntegrationError(
            "Qwen2.5-VL video feature rows must be rectangular."
        )
    return (len(rows), width)


def _tensor_shape(value: Any) -> tuple[int, ...]:
    if value is None:
        return ()
    shape = getattr(value, "shape", None)
    if shape is None:
        return _feature_shape(value)
    return tuple(int(item) for item in shape)


def _exact_rope_deltas(position_ids: Any, attention_mask: Any | None) -> Any:
    torch = _torch()
    rotary = position_ids[1:] if int(position_ids.shape[0]) == 4 else position_ids
    sequence_length = int(rotary.shape[-1])
    if attention_mask is None:
        active = rotary.reshape(-1)
        active_length = sequence_length
    else:
        valid = attention_mask[0].bool()
        active = rotary[:, 0, valid].reshape(-1)
        active_length = int(valid.sum().item())
    if active_length <= 0 or int(active.numel()) <= 0:
        raise InvariantError("Qwen2.5-VL compressed prompt has no active tokens.")
    maximum = active.max()
    return (maximum + 1 - active_length).to(dtype=torch.long).reshape(1, 1)


def _mrope_columns(
    position_ids: Any,
    positions: Sequence[int],
) -> tuple[tuple[int, ...], ...]:
    return tuple(
        tuple(int(value) for value in position_ids[:, 0, index].tolist())
        for index in positions
    )


def _set_output_field(output: Any, name: str, value: Any) -> None:
    if isinstance(output, dict):
        output[name] = value
    else:
        setattr(output, name, value)


def _tensor_int_values(value: Any) -> list[int]:
    if value is None:
        return []
    raw = value.detach().cpu().reshape(-1).tolist()
    return [int(item) for item in raw]


def _cache_sequence_length(value: Any) -> int | None:
    get_length = getattr(value, "get_seq_length", None)
    if callable(get_length):
        try:
            return int(get_length())
        except (TypeError, ValueError):
            return None
    layers = getattr(value, "layers", None)
    if layers:
        layer_length = getattr(layers[0], "get_seq_length", None)
        if callable(layer_length):
            try:
                return int(layer_length())
            except (TypeError, ValueError):
                return None
    return None


def _torch() -> Any:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - environment preflight owns this
        raise RuntimeError("Qwen2.5-VL video compression requires PyTorch.") from exc
    return torch


__all__ = [
    "QWEN25_VL_VIDEO_BRIDGE_API_VERSION",
    "QWEN25_VL_VIDEO_SPLICE_SITE",
    "PinnedQwen25VLVideoBridge",
    "Qwen25VLPackedVideoRequest",
    "Qwen25VLVideoCallbackHandle",
    "Qwen25VLVideoGrid",
    "Qwen25VLVideoPromptSpliceResult",
    "Qwen25VLVideoSplicePlan",
    "install_qwen25_vl_video_bridge",
    "legalize_qwen25_vl_video_prompt",
    "plan_qwen25_vl_video_splice",
    "qwen25_vl_video_request_context",
]
