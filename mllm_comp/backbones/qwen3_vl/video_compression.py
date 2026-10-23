"""Pinned Qwen3-VL video payload, DeepStack, and sparse M-RoPE repair.

Qwen3-VL emits one post-merger feature stream plus one feature stream for
every DeepStack projector.  All streams share the same raster token order and
are injected at positions selected by ``visual_pos_masks``.  A video
compressor must therefore apply one source layout to the main stream and every
DeepStack stream; shortening only the main stream is not a legal Qwen3-VL
execution.

The bridge in this module keeps the native source ``video_grid_thw`` and any
provider timing kwargs for vision encoding and M-RoPE construction.  It
scatters synchronized compressed streams at representative source
placeholders, gathers the native M-RoPE anchors at the language boundary, and
compacts the physical prompt, attention mask, labels, cache positions, and
``visual_pos_masks`` together.  It is instance-local, reversible, batch-one /
single-video only, and deliberately does not activate any support unit.
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


QWEN3_VL_VIDEO_BRIDGE_API_VERSION = "mllm_comp.qwen3_vl.video.v1"
QWEN3_VL_VIDEO_SPLICE_SITE = "video_post_merger_with_deepstack_pre_llm_scatter"
QWEN3_VL_VIDEO_PRE_PROJECTOR_SITE = "video_pre_main_and_deepstack_mergers"
QWEN3_VL_IGNORE_INDEX = -100
_VIDEO_BRIDGE_ATTRIBUTE = "_mllm_comp_qwen3_vl_video_bridge"
_PUBLIC_VIDEO_BRIDGE_ATTRIBUTE = "mllm_video_compression_bridge"
_IMAGE_PROVIDER_ATTRIBUTES = ("_mllm_comp_qwen3_vl_provider_bridge",)


@dataclass(frozen=True)
class Qwen3VLVideoGrid:
    """One source video grid after the native spatial merger."""

    temporal: int
    source_height: int
    source_width: int
    spatial_merge_size: int
    second_per_grid_seconds: float | None = None

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
        if self.second_per_grid_seconds is not None:
            timing = float(self.second_per_grid_seconds)
            if not math.isfinite(timing) or timing <= 0:
                raise ValueError("second_per_grid_seconds must be finite and positive")
            object.__setattr__(self, "second_per_grid_seconds", timing)

    @classmethod
    def from_native(
        cls,
        grid_thw: Sequence[int],
        *,
        spatial_merge_size: int,
        second_per_grid_seconds: float | None = None,
    ) -> "Qwen3VLVideoGrid":
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
                timestamp_seconds=(
                    None
                    if self.second_per_grid_seconds is None
                    else temporal * self.second_per_grid_seconds
                ),
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
                "Qwen3-VL post-merger video features must be "
                f"[{self.token_count}, hidden], got {shape}."
            )
        timing_signal = (
            {}
            if self.second_per_grid_seconds is None
            else {"qwen3_vl.second_per_grid_seconds": self.second_per_grid_seconds}
        )
        return VideoTokenPayload(
            features=features,
            media_ids=(media_id,),
            video_offsets=(0, self.token_count),
            coordinates=self.coordinates,
            signals={
                "qwen3_vl.source_video_grid_thw": self.source_grid_thw,
                "qwen3_vl.spatial_merge_size": self.spatial_merge_size,
                **timing_signal,
                **dict(signals or {}),
            },
            metadata={
                "backbone": "qwen3_vl",
                "grid_policy": "preserve_source_grid_and_gather_sparse_mrope",
                "deepstack_policy": "shared_source_groups_all_branches",
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
class Qwen3VLPackedVideoRequest:
    """Algorithm request at the native main-plus-DeepStack boundary."""

    invocation_id: str
    media_id: str
    payload: VideoTokenPayload
    deepstack_features: tuple[Any, ...]
    grid: Qwen3VLVideoGrid
    source_placeholder_positions: tuple[int, ...]
    prompt_input_ids: Any
    source_attention_mask: Any | None
    source_video_grid_thw: Any
    source_second_per_grid_ts: Any | None

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
        if positions != tuple(sorted(set(positions))):
            raise ValueError(
                "Qwen3-VL video placeholder positions must be strictly increasing"
            )
        main_shape = _feature_shape(self.payload.features)
        branches = tuple(self.deepstack_features)
        for index, branch in enumerate(branches):
            shape = _feature_shape(branch)
            if len(shape) != 2 or shape != main_shape:
                raise ValueError(
                    "Qwen3-VL DeepStack source branch "
                    f"{index} must match main features {main_shape}, got {shape}."
                )
        object.__setattr__(self, "deepstack_features", branches)


@dataclass(frozen=True)
class Qwen3VLVideoCompressionResponse:
    """One main result and all DeepStack branches under the same layout."""

    compression_result: VideoCompressionResult
    deepstack_features: tuple[Any, ...]
    synchronization_source_groups: tuple[tuple[tuple[int, ...], ...], ...]

    def __post_init__(self) -> None:
        result = self.compression_result
        if not isinstance(result, VideoCompressionResult):
            raise TypeError("compression_result must be VideoCompressionResult")
        groups = tuple(
            tuple(tuple(group) for group in video_groups)
            for video_groups in self.synchronization_source_groups
        )
        if groups != result.source_groups:
            raise ValueError(
                "DeepStack synchronization groups must equal the main result groups"
            )
        result_shape = _feature_shape(result.features)
        branches = tuple(self.deepstack_features)
        for index, branch in enumerate(branches):
            shape = _feature_shape(branch)
            if len(shape) != 2 or shape != result_shape:
                raise ValueError(
                    "Qwen3-VL DeepStack result branch "
                    f"{index} must match main features {result_shape}, got {shape}."
                )
        object.__setattr__(self, "deepstack_features", branches)
        object.__setattr__(self, "synchronization_source_groups", groups)

    @classmethod
    def identity(
        cls,
        request: Qwen3VLPackedVideoRequest,
    ) -> "Qwen3VLVideoCompressionResponse":
        result = VideoCompressionResult.identity(request.payload)
        return cls(
            compression_result=result,
            deepstack_features=request.deepstack_features,
            synchronization_source_groups=result.source_groups,
        )

    @classmethod
    def from_result(
        cls,
        request: Qwen3VLPackedVideoRequest,
        compression_result: VideoCompressionResult,
        *,
        deepstack_features: Sequence[Any],
    ) -> "Qwen3VLVideoCompressionResponse":
        if compression_result.media_ids != request.payload.media_ids:
            raise InvariantError("Qwen3-VL video compression changed media identity.")
        if compression_result.source_video_offsets != request.payload.video_offsets:
            raise InvariantError("Qwen3-VL video compression changed source bounds.")
        branches = tuple(deepstack_features)
        if len(branches) != len(request.deepstack_features):
            raise InvariantError(
                "Qwen3-VL video compression changed the DeepStack branch count."
            )
        return cls(
            compression_result=compression_result,
            deepstack_features=branches,
            synchronization_source_groups=compression_result.source_groups,
        )


@dataclass(frozen=True)
class Qwen3VLVideoSplicePlan:
    """Framework-neutral sequence and visual-mask compaction plan."""

    source_visual_span: tuple[int, int]
    result_visual_span: tuple[int, int]
    source_placeholder_positions: tuple[int, ...]
    retained_placeholder_positions: tuple[int, ...]
    removed_placeholder_positions: tuple[int, ...]
    result_placeholder_positions: tuple[int, ...]
    kept_sequence_indices: tuple[int, ...]
    representative_indices: tuple[int, ...]


def plan_qwen3_vl_video_splice(
    request: Qwen3VLPackedVideoRequest,
    response: Qwen3VLVideoCompressionResponse,
    *,
    sequence_length: int,
) -> Qwen3VLVideoSplicePlan:
    """Validate shared provenance and map sparse anchors into one prompt."""

    if not isinstance(request, Qwen3VLPackedVideoRequest):
        raise TypeError("request must be Qwen3VLPackedVideoRequest")
    if not isinstance(response, Qwen3VLVideoCompressionResponse):
        raise TypeError("response must be Qwen3VLVideoCompressionResponse")
    result = response.compression_result
    if result.media_ids != request.payload.media_ids:
        raise InvariantError("Qwen3-VL video compression changed media identity.")
    if result.source_video_offsets != request.payload.video_offsets:
        raise InvariantError("Qwen3-VL video compression changed source bounds.")
    if result.video_count != 1:
        raise InvariantError("Qwen3-VL video splice supports exactly one video.")
    if len(response.deepstack_features) != len(request.deepstack_features):
        raise InvariantError("Qwen3-VL DeepStack branch count changed.")
    if (
        isinstance(sequence_length, bool)
        or not isinstance(sequence_length, int)
        or sequence_length <= 0
    ):
        raise ValueError("sequence_length must be a positive integer")

    source = request.source_placeholder_positions
    if source[0] < 0 or source[-1] >= sequence_length:
        raise InvariantError("Qwen3-VL source video span is outside the prompt.")
    representatives = result.representative_indices[0]
    if representatives != tuple(sorted(representatives)):
        raise InvariantError(
            "Qwen3-VL video representative anchors must preserve source order."
        )
    retained = tuple(source[index] for index in representatives)
    retained_set = set(retained)
    removed = tuple(index for index in source if index not in retained_set)
    removed_set = set(removed)
    kept = tuple(index for index in range(sequence_length) if index not in removed_set)
    compact_positions = {
        source_index: result_index for result_index, source_index in enumerate(kept)
    }
    result_positions = tuple(compact_positions[index] for index in retained)
    return Qwen3VLVideoSplicePlan(
        source_visual_span=(source[0], source[-1] + 1),
        result_visual_span=(result_positions[0], result_positions[-1] + 1),
        source_placeholder_positions=source,
        retained_placeholder_positions=retained,
        removed_placeholder_positions=removed,
        result_placeholder_positions=result_positions,
        kept_sequence_indices=kept,
        representative_indices=representatives,
    )


def _position_spans(positions: Sequence[int]) -> tuple[tuple[int, int], ...]:
    values = tuple(int(position) for position in positions)
    if not values:
        return ()
    spans = []
    start = previous = values[0]
    for position in values[1:]:
        if position != previous + 1:
            spans.append((start, previous + 1))
            start = position
        previous = position
    spans.append((start, previous + 1))
    return tuple(spans)


@dataclass(frozen=True)
class Qwen3VLVideoPromptSpliceResult:
    """Compacted Qwen3 prompt with synchronized visual and DeepStack fields."""

    input_ids: Any
    inputs_embeds: Any
    attention_mask: Any | None
    position_ids: Any
    cache_position: Any
    labels: Any | None
    visual_pos_masks: Any
    deepstack_visual_embeds: tuple[Any, ...]
    rope_deltas: Any
    plan: Qwen3VLVideoSplicePlan
    response: Qwen3VLVideoCompressionResponse
    source_visual_mrope: tuple[tuple[int, ...], ...]
    result_visual_mrope: tuple[tuple[int, ...], ...]

    def record(self) -> dict[str, Any]:
        result = self.response.compression_result
        result_length = int(self.input_ids.shape[1])
        labels_present = self.labels is not None
        visual_labels_ignored = bool(
            not labels_present
            or (
                self.labels[0, list(self.plan.result_placeholder_positions)]
                == QWEN3_VL_IGNORE_INDEX
            )
            .all()
            .item()
        )
        return {
            "source_visual_span": self.plan.source_visual_span,
            "result_visual_span": self.plan.result_visual_span,
            "source_placeholder_count": len(self.plan.source_placeholder_positions),
            "result_placeholder_count": len(self.plan.result_placeholder_positions),
            "source_placeholder_spans": _position_spans(
                self.plan.source_placeholder_positions
            ),
            "result_placeholder_spans": _position_spans(
                self.plan.result_placeholder_positions
            ),
            "retained_placeholder_positions": self.plan.retained_placeholder_positions,
            "removed_placeholder_positions": self.plan.removed_placeholder_positions,
            "representative_indices": self.plan.representative_indices,
            "source_visual_mrope": self.source_visual_mrope,
            "result_visual_mrope": self.result_visual_mrope,
            "source_position_plane_count": int(self.position_ids.shape[0]),
            "result_position_plane_count": int(self.position_ids.shape[0]),
            "source_video_grid_preserved": True,
            "result_grid_policy": "irregular_sparse_source_anchors",
            "visual_pos_masks_compacted": True,
            "visual_pos_masks_result_count": int(
                self.visual_pos_masks[0].bool().sum().item()
            ),
            "attention_mask_compacted": bool(
                self.attention_mask is None
                or tuple(self.attention_mask.shape) == tuple(self.input_ids.shape)
            ),
            "position_ids_compacted": bool(
                int(self.position_ids.shape[-1]) == result_length
            ),
            "cache_position_dense": tuple(
                int(value) for value in self.cache_position.tolist()
            )
            == tuple(range(result_length)),
            "labels_present": labels_present,
            "training_labels_compacted": bool(
                not labels_present
                or tuple(self.labels.shape) == tuple(self.input_ids.shape)
            ),
            "visual_labels_ignored": visual_labels_ignored,
            "deepstack_branch_count": len(self.deepstack_visual_embeds),
            "deepstack_result_shapes": tuple(
                _feature_shape(branch) for branch in self.deepstack_visual_embeds
            ),
            "shared_main_deepstack_layout": True,
            "deepstack_group_count": len(
                self.response.synchronization_source_groups[0]
            ),
            "compression_applied": result.compression_applied,
            "layout_delta": layout_delta_record(result.layout_deltas[0]),
        }


def legalize_qwen3_vl_video_prompt(
    *,
    input_ids: Any,
    inputs_embeds: Any,
    attention_mask: Any | None,
    position_ids: Any,
    cache_position: Any | None,
    labels: Any | None,
    visual_pos_masks: Any,
    deepstack_visual_embeds: Sequence[Any] | None,
    request: Qwen3VLPackedVideoRequest,
    response: Qwen3VLVideoCompressionResponse,
    ignore_index: int = QWEN3_VL_IGNORE_INDEX,
) -> Qwen3VLVideoPromptSpliceResult:
    """Gather one compressed prompt and its native interleaved M-RoPE anchors."""

    input_shape = _tensor_shape(input_ids)
    embed_shape = _tensor_shape(inputs_embeds)
    if len(input_shape) != 2 or input_shape[0] != 1:
        raise BackendIntegrationError(
            "Qwen3-VL video legalization requires [1, sequence] input_ids."
        )
    if len(embed_shape) != 3 or embed_shape[:2] != input_shape:
        raise BackendIntegrationError(
            "Qwen3-VL video legalization requires aligned [1, sequence, hidden] "
            "inputs_embeds."
        )
    source_length = input_shape[1]
    plan = plan_qwen3_vl_video_splice(
        request,
        response,
        sequence_length=source_length,
    )
    position_shape = _tensor_shape(position_ids)
    if position_shape not in {(3, 1, source_length), (4, 1, source_length)}:
        raise BackendIntegrationError(
            "Qwen3-VL video M-RoPE positions must be [3, 1, sequence] or "
            "[4, 1, sequence]."
        )
    if attention_mask is not None and _tensor_shape(attention_mask) != input_shape:
        raise InvariantError("Qwen3-VL video attention_mask must match input_ids.")
    if labels is not None and _tensor_shape(labels) != input_shape:
        raise InvariantError("Qwen3-VL video labels must match input_ids.")
    if _tensor_shape(visual_pos_masks) != input_shape:
        raise InvariantError("Qwen3-VL visual_pos_masks must match input_ids.")
    source_visual_positions = tuple(
        int(value)
        for value in visual_pos_masks[0]
        .bool()
        .nonzero(as_tuple=False)
        .flatten()
        .tolist()
    )
    if source_visual_positions != plan.retained_placeholder_positions:
        raise InvariantError(
            "Qwen3-VL visual_pos_masks do not select the shared video anchors."
        )

    branches = tuple(deepstack_visual_embeds or ())
    if len(branches) != len(response.deepstack_features):
        raise InvariantError("Qwen3-VL language DeepStack branch count changed.")
    result_shape = _feature_shape(response.compression_result.features)
    for index, branch in enumerate(branches):
        if _feature_shape(branch) != result_shape:
            raise InvariantError(
                f"Qwen3-VL language DeepStack branch {index} changed layout."
            )

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
    compact_positions = position_ids.index_select(-1, keep.to(position_ids.device))
    compact_labels = (
        None if labels is None else labels.index_select(1, keep.to(labels.device))
    )
    compact_visual_mask = visual_pos_masks.index_select(
        1,
        keep.to(visual_pos_masks.device),
    )
    result_visual_positions = tuple(
        int(value)
        for value in compact_visual_mask[0]
        .bool()
        .nonzero(as_tuple=False)
        .flatten()
        .tolist()
    )
    if result_visual_positions != plan.result_placeholder_positions:
        raise InvariantError(
            "Qwen3-VL compact visual_pos_masks changed result placeholder order."
        )

    if labels is not None:
        source_visual_labels = labels[0, list(plan.source_placeholder_positions)]
        if bool((source_visual_labels != ignore_index).any().item()):
            raise InvariantError(
                "Qwen3-VL source video labels must be ignored before compression."
            )
        result_visual_labels = compact_labels[
            0, list(plan.result_placeholder_positions)
        ]
        if bool((result_visual_labels != ignore_index).any().item()):
            raise InvariantError(
                "Qwen3-VL compressed video labels must remain ignored."
            )

    if cache_position is None:
        compact_cache = torch.arange(
            len(plan.kept_sequence_indices),
            dtype=torch.long,
            device=inputs_embeds.device,
        )
    else:
        if _tensor_shape(cache_position) != (source_length,):
            raise InvariantError(
                "Qwen3-VL prefill cache_position must cover the source prompt."
            )
        compact_cache = torch.arange(
            len(plan.kept_sequence_indices),
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
            "Qwen3-VL compressed video M-RoPE anchors changed during gather."
        )
    return Qwen3VLVideoPromptSpliceResult(
        input_ids=compact_ids,
        inputs_embeds=compact_embeds,
        attention_mask=compact_mask,
        position_ids=compact_positions,
        cache_position=compact_cache,
        labels=compact_labels,
        visual_pos_masks=compact_visual_mask,
        deepstack_visual_embeds=branches,
        rope_deltas=rope_deltas,
        plan=plan,
        response=response,
        source_visual_mrope=source_visual_mrope,
        result_visual_mrope=result_visual_mrope,
    )


class Qwen3VLVideoCallbackHandle:
    def __init__(self, bridge: "PinnedQwen3VLVideoBridge", token: int) -> None:
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
                f"Qwen3-VL video hook {self.name!r} changed while active."
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
    second_per_grid_ts: Any | None
    grid: Qwen3VLVideoGrid
    source_placeholder_positions: tuple[int, ...]
    media_id: str
    conditioning_text: str | None


@dataclass(frozen=True)
class _ActiveSplice:
    request: Qwen3VLPackedVideoRequest
    response: Qwen3VLVideoCompressionResponse
    plan: Qwen3VLVideoSplicePlan


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


class PinnedQwen3VLVideoBridge:
    """Reversible provider bridge for one pinned Qwen3-VL model."""

    api_version = QWEN3_VL_VIDEO_BRIDGE_API_VERSION
    capabilities: Mapping[str, bool] = MappingProxyType(
        {
            "native_post_merger_features": True,
            "native_pre_merger_spatial_groups": True,
            "source_video_grid_provenance": True,
            "sparse_mrope_anchor_gather": True,
            "placeholder_repair": True,
            "attention_mask_repair": True,
            "training_label_repair": True,
            "prefill_cache_position_repair": True,
            "decode_input_repair": True,
            "visual_pos_masks_repair": True,
            "deepstack_shared_layout": True,
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
        self._callback: Callable[[Qwen3VLPackedVideoRequest], Any] | None = None
        self._signal_provider: Any | None = None
        self._projector_position = "after"
        self._preserve_projected_values = False
        self._latest_projector: dict[str, Any] | None = None
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
            f"qwen3_video_request_metadata_{suffix}", default=None
        )
        self._active_source: ContextVar[_ActiveSource | None] = ContextVar(
            f"qwen3_video_active_source_{suffix}", default=None
        )
        self._active_splice: ContextVar[_ActiveSplice | None] = ContextVar(
            f"qwen3_video_active_splice_{suffix}", default=None
        )
        self._selected_video_mask: ContextVar[Any | None] = ContextVar(
            f"qwen3_video_selected_mask_{suffix}", default=None
        )
        self._prompt_splice: ContextVar[Qwen3VLVideoPromptSpliceResult | None] = (
            ContextVar(f"qwen3_video_prompt_splice_{suffix}", default=None)
        )
        self._training_labels: ContextVar[Any | None] = ContextVar(
            f"qwen3_video_training_labels_{suffix}", default=None
        )
        self._vtp_rope_deltas: ContextVar[Any | None] = ContextVar(
            f"qwen3_video_vtp_rope_deltas_{suffix}", default=None
        )
        self._generation: ContextVar[_GenerationLifecycle | None] = ContextVar(
            f"qwen3_video_generation_{suffix}", default=None
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
        return (QWEN3_VL_VIDEO_SPLICE_SITE,)

    def register_callback(
        self,
        site: str,
        callback: Callable[[Qwen3VLPackedVideoRequest], Any],
        *,
        signal_provider: Any | None = None,
        projector_position: str = "after",
        preserve_projected_values: bool = False,
    ) -> Qwen3VLVideoCallbackHandle:
        if site != QWEN3_VL_VIDEO_SPLICE_SITE:
            raise BackendIntegrationError(
                f"Unknown Qwen3-VL video provider site {site!r}."
            )
        if not callable(callback):
            raise BackendIntegrationError(
                "Qwen3-VL video provider callback must be callable."
            )
        if projector_position not in ("before", "after"):
            raise BackendIntegrationError("Qwen3-VL projector_position must be before or after.")
        if self._callback is not None:
            raise BackendIntegrationError(
                "Qwen3-VL video splice already has an active callback."
            )
        if signal_provider is not None and not all(
            callable(getattr(signal_provider, name, None))
            for name in ("capture_context", "payload_signals")
        ):
            raise BackendIntegrationError(
                "Qwen3-VL video signal provider has an incomplete capture contract."
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
        return Qwen3VLVideoCallbackHandle(self, token)

    @contextmanager
    def request_context(self, metadata: Mapping[str, Any]) -> Iterator[None]:
        if not isinstance(metadata, Mapping):
            raise TypeError("Qwen3-VL video request metadata must be a mapping")
        token = self._request_metadata.set(dict(metadata))
        try:
            yield
        finally:
            self._request_metadata.reset(token)

    def runtime_metadata(self) -> dict[str, Any]:
        return {
            "provider": "transformers_qwen3_vl_video_instance_bridge",
            "bridge_api_version": self.api_version,
            "contract_mode": self._contract_mode,
            "patched": self.patched,
            "active_sites": (
                [QWEN3_VL_VIDEO_PRE_PROJECTOR_SITE if self._projector_position == "before"
                 else QWEN3_VL_VIDEO_SPLICE_SITE] if self._callback is not None else []
            ),
            "callback_count": self._callback_count,
            "projector_position": self._projector_position,
            "actual_projector_site": (
                QWEN3_VL_VIDEO_PRE_PROJECTOR_SITE
                if self._projector_position == "before" else QWEN3_VL_VIDEO_SPLICE_SITE
            ),
            "latest_projector": self._latest_projector,
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

    def apply_metok_pruning_layout(
        self,
        result: METokPruningResult,
        *,
        deepstack_branch_count: int | None = None,
        deepstack_consumed_branch_count: int | None = None,
    ) -> None:
        """Repair native MRoPE, masks, labels, and decode state per METok stage."""

        if not isinstance(result, METokPruningResult):
            raise TypeError("result must be a METokPruningResult")
        splice = self._active_splice.get()
        prompt = self._prompt_splice.get()
        if splice is None or prompt is None:
            raise BackendIntegrationError(
                "Qwen3-VL METok must run inside an active video prompt"
            )
        if result.stage_index == 0:
            self._latest_metok = []
        records = self._latest_metok
        if records is None:
            raise InvariantError("Qwen3-VL METok stages began out of order")
        projector = splice.response.compression_result
        expected_sequence = (
            int(prompt.inputs_embeds.shape[1])
            if not records
            else int(records[-1]["result_sequence_length"])
        )
        expected_visual = (
            projector.result_count
            if not records
            else int(records[-1]["result_visual_count"])
        )
        if result.sequence_layout_delta.source_count != expected_sequence:
            raise InvariantError("Qwen3-VL METok sequence length changed")
        if result.visual_layout_delta.source_count != expected_visual:
            raise InvariantError("Qwen3-VL METok visual layout changed")
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
            raise InvariantError("Qwen3-VL METok cumulative M-RoPE layout changed")
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
                "Qwen3-VL METok changed gathered native M-RoPE anchors"
            )
        source_visual_mask = prompt.visual_pos_masks.index_select(
            1,
            source_keep.to(prompt.visual_pos_masks.device),
        )
        compact_visual_mask = prompt.visual_pos_masks.index_select(
            1,
            cumulative_keep.to(prompt.visual_pos_masks.device),
        )
        source_visual_pos_mask_indices = tuple(
            int(item)
            for item in source_visual_mask[0]
            .bool()
            .nonzero(as_tuple=False)
            .flatten()
            .tolist()
        )
        result_visual_pos_mask_indices = tuple(
            int(item)
            for item in compact_visual_mask[0]
            .bool()
            .nonzero(as_tuple=False)
            .flatten()
            .tolist()
        )
        if (
            source_visual_pos_mask_indices != source_visual_sequence_indices
            or result_visual_pos_mask_indices != result_visual_sequence_indices
        ):
            raise InvariantError("Qwen3-VL METok visual_pos_masks changed layout")
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
            raise InvariantError("Qwen3-VL METok published an inexact rope delta")
        expected_deepstack_branches = len(
            tuple(getattr(splice.response, "deepstack_features", ()))
        )
        observed_deepstack_branches = (
            expected_deepstack_branches
            if deepstack_branch_count is None
            else int(deepstack_branch_count)
        )
        consumed_deepstack_branches = (
            observed_deepstack_branches
            if deepstack_consumed_branch_count is None
            else int(deepstack_consumed_branch_count)
        )
        if observed_deepstack_branches != expected_deepstack_branches:
            raise InvariantError("Qwen3-VL METok DeepStack branch count changed")
        if not 0 <= consumed_deepstack_branches <= observed_deepstack_branches:
            raise InvariantError("Qwen3-VL METok consumed an invalid number of DeepStack branches")
        self._vtp_rope_deltas.set(rope_deltas)
        labels = self._training_labels.get()
        if labels is not None:
            dropped = tuple(result.sequence_layout_delta.dropped_indices)
            if dropped and bool(
                (labels[0, list(dropped)] != QWEN3_VL_IGNORE_INDEX).any().item()
            ):
                raise InvariantError(
                    "Qwen3-VL METok may drop ignored visual labels only"
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
                "source_visual_pos_mask_indices": (source_visual_pos_mask_indices),
                "result_visual_pos_mask_indices": (result_visual_pos_mask_indices),
                "visual_pos_masks_compacted": True,
                "visual_pos_masks_compacted_exact": True,
                "deepstack_branch_count": observed_deepstack_branches,
                "deepstack_consumed_branch_count": consumed_deepstack_branches,
                "deepstack_completed_before_pruning": consumed_deepstack_branches == observed_deepstack_branches,
                "deepstack_projector_result_shapes": tuple(
                    _feature_shape(branch)
                    for branch in tuple(
                        getattr(splice.response, "deepstack_features", ())
                    )
                ),
                "deepstack_keep_synchronized": True,
                "position_policy": "gather_native_mrope",
                "active_position_count": active_position_count,
                "max_active_rotary_position": max_active_rotary_position,
                "expected_rope_delta": expected_rope_delta,
                "rope_delta_exact": True,
                "rope_deltas": _tensor_int_values(rope_deltas),
            }
        )

    def apply_prunevid_vtp_layout(self, result: PruneVidVTPResult, *, deepstack_consumed_branch_count: int | None = None) -> None:
        """Repair and freeze the exact Qwen3 layout at one VTP boundary."""

        if not isinstance(result, PruneVidVTPResult):
            raise TypeError("result must be a PruneVidVTPResult")
        splice = self._active_splice.get()
        prompt = self._prompt_splice.get()
        if splice is None or prompt is None:
            raise BackendIntegrationError(
                "Qwen3-VL VTP must run inside an active video prompt"
            )
        source_length = int(prompt.inputs_embeds.shape[1])
        projector = splice.response.compression_result
        if result.sequence_layout_delta.source_count != source_length:
            raise InvariantError("Qwen3-VL VTP sequence length changed")
        if result.visual_layout_delta.source_count != projector.result_count:
            raise InvariantError("Qwen3-VL VTP visual layout changed")
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
            int(item) for item in prompt.plan.result_placeholder_positions
        )
        source_mask_positions = tuple(
            int(item)
            for item in prompt.visual_pos_masks[0]
            .bool()
            .nonzero(as_tuple=False)
            .flatten()
            .tolist()
        )
        if source_mask_positions != source_visual_positions:
            raise InvariantError(
                "Qwen3-VL PruneVid VTP received divergent visual_pos_masks"
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
                "Qwen3-VL PruneVid VTP changed gathered native M-RoPE anchors"
            )
        compact_visual_mask = prompt.visual_pos_masks.index_select(
            1,
            keep.to(prompt.visual_pos_masks.device),
        )
        result_mask_positions = tuple(
            int(item)
            for item in compact_visual_mask[0]
            .bool()
            .nonzero(as_tuple=False)
            .flatten()
            .tolist()
        )
        if result_mask_positions != result_visual_sequence_indices:
            raise InvariantError(
                "Qwen3-VL PruneVid VTP changed compact visual_pos_masks"
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
                "Qwen3-VL PruneVid VTP published an inexact rope delta"
            )
        deepstack_shapes = tuple(
            _feature_shape(branch) for branch in prompt.deepstack_visual_embeds
        )
        deepstack_consumed_count = (min(len(deepstack_shapes), result.selected_layer + 1)
            if deepstack_consumed_branch_count is None else int(deepstack_consumed_branch_count))
        if not 0 <= deepstack_consumed_count <= len(deepstack_shapes):
            raise InvariantError("Qwen3-VL PruneVid consumed an invalid number of DeepStack branches")
        self._vtp_rope_deltas.set(rope_deltas)
        labels = self._training_labels.get()
        if labels is not None:
            dropped = tuple(result.sequence_layout_delta.dropped_indices)
            if dropped and bool(
                (labels[0, list(dropped)] != QWEN3_VL_IGNORE_INDEX).any().item()
            ):
                raise InvariantError(
                    "Qwen3-VL PruneVid VTP may drop ignored visual labels only"
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
            "source_visual_pos_mask_indices": source_mask_positions,
            "result_visual_pos_mask_indices": result_mask_positions,
            "visual_pos_masks_compacted": True,
            "visual_pos_masks_compacted_exact": True,
            "deepstack_branch_count": len(deepstack_shapes),
            "deepstack_projector_result_shapes": deepstack_shapes,
            "deepstack_consumed_branch_count": deepstack_consumed_count,
            "deepstack_completed_before_vtp": deepstack_consumed_count == len(deepstack_shapes),
            "active_position_count": active_position_count,
            "max_active_rotary_position": max_active_rotary_position,
            "expected_rope_delta": expected_rope_delta,
            "rope_delta_exact": True,
            "rope_deltas": _tensor_int_values(rope_deltas),
        }

    def detach(self) -> None:
        if self._callback is not None:
            raise BackendIntegrationError(
                "Cannot detach the Qwen3-VL video bridge while a callback is active."
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
                    "Qwen3-VL image and video provider bridges cannot patch the "
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
                "Pinned Qwen3-VL video bridge could not resolve native methods."
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
            if self._callback is None:
                return original(*args, **kwargs)
            bound = signature.bind_partial(*args, **kwargs)
            lifecycle = self._generation.get()
            if (
                lifecycle is not None
                and lifecycle.prefill_complete
                and _bound_value(bound, "past_key_values") is not None
            ):
                # Transformers retains video_grid_thw on cached decode calls
                # after correctly dropping the one-shot pixel tensor.  The
                # completed lifecycle, rather than partial media kwargs, is
                # the authoritative boundary between prefill and decode.
                self._repair_decode_call(bound, lifecycle)
                return original(*bound.args, **bound.kwargs)
            has_video = any(
                _bound_value(bound, name) is not None
                for name in ("pixel_values_videos", "video_grid_thw")
            )
            if not has_video:
                if (
                    lifecycle is not None
                    and _bound_value(bound, "past_key_values") is not None
                ):
                    self._repair_decode_call(bound, lifecycle)
                return original(*bound.args, **bound.kwargs)

            if any(
                _bound_value(bound, name) is not None
                for name in ("pixel_values", "image_grid_thw")
            ):
                raise BackendIntegrationError(
                    "Compressed Qwen3-VL video requests cannot mix image media."
                )
            input_ids = _bound_value(bound, "input_ids")
            if _tensor_shape(input_ids)[:1] != (1,):
                raise BackendIntegrationError(
                    "Qwen3-VL video compression requires batch size one."
                )
            if _bound_value(bound, "inputs_embeds") is not None:
                raise BackendIntegrationError(
                    "Qwen3-VL video compression requires source input_ids."
                )
            required_video_fields = {
                "pixel_values_videos": _bound_value(bound, "pixel_values_videos"),
                "video_grid_thw": _bound_value(bound, "video_grid_thw"),
            }
            missing = tuple(
                name for name, value in required_video_fields.items() if value is None
            )
            if missing:
                raise BackendIntegrationError(
                    "Compressed Qwen3-VL video input is missing native field(s): "
                    + ", ".join(missing)
                    + "."
                )
            labels = _bound_value(bound, "labels")
            use_cache = _bound_value(bound, "use_cache")
            if lifecycle is None and labels is not None and use_cache is None:
                # The platform teacher-forcing runner supplies labels but does
                # not add provider cache kwargs.  Compressed training owns a
                # physically shortened prompt, so make the required no-cache
                # mode explicit without changing native text/image calls.
                _bound_assign(bound, "use_cache", False)
                use_cache = False
            past_key_values = _bound_value(bound, "past_key_values")
            if past_key_values is not None:
                cache_length = _cache_sequence_length(past_key_values)
                if lifecycle is None or cache_length != 0:
                    raise BackendIntegrationError(
                        "Qwen3-VL compressed video media is valid only with an "
                        "empty prefill cache."
                    )
            if lifecycle is None and use_cache is not False:
                raise BackendIntegrationError(
                    "Direct Qwen3-VL compressed forward requires use_cache=False."
                )
            if lifecycle is not None and use_cache is not True:
                raise BackendIntegrationError(
                    "Qwen3-VL compressed generation prefill requires use_cache=True."
                )
            if lifecycle is not None and labels is not None:
                raise BackendIntegrationError(
                    "Qwen3-VL compressed generation does not accept labels."
                )
            if labels is not None and _bound_value(bound, "return_dict") is False:
                raise BackendIntegrationError(
                    "Compressed Qwen3-VL video training requires return_dict=True."
                )

            label_token = self._training_labels.set(labels)
            if labels is not None:
                _bound_assign(bound, "labels", None)
            try:
                output = original(*bound.args, **bound.kwargs)
                if labels is None:
                    return output
                repaired = self._training_labels.get()
                if repaired is None:
                    raise InvariantError(
                        "Qwen3-VL compressed video labels disappeared."
                    )
                return attach_shifted_causal_loss(
                    output,
                    repaired,
                    operation="Qwen3-VL video compression",
                )
            finally:
                self._training_labels.reset(label_token)

        return replacement

    def _generate_replacement(self) -> Callable[..., Any]:
        original = self._original_generate
        assert original is not None
        signature = inspect.signature(original)

        def replacement(_model: Any, *args: Any, **kwargs: Any) -> Any:
            if self._callback is None:
                return original(*args, **kwargs)
            bound = signature.bind_partial(*args, **kwargs)
            if not any(
                _bound_value(bound, name) is not None
                for name in ("pixel_values_videos", "video_grid_thw")
            ):
                return original(*bound.args, **bound.kwargs)
            if self._generation.get() is not None:
                raise BackendIntegrationError(
                    "Nested Qwen3-VL compressed generation is unsupported."
                )
            lifecycle = _GenerationLifecycle(
                request_id=f"generate-{self._invocation_counter + 1}"
            )
            token = self._generation.set(lifecycle)
            try:
                output = original(*bound.args, **bound.kwargs)
                if not lifecycle.prefill_complete:
                    raise InvariantError(
                        "Qwen3-VL generate returned before compressed prefill."
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
            if _bound_value(bound, "pixel_values_videos") is None:
                return original(*bound.args, **bound.kwargs)
            if _bound_value(bound, "pixel_values") is not None:
                raise BackendIntegrationError(
                    "Compressed Qwen3-VL video requests cannot mix image media."
                )
            input_ids = _bound_value(bound, "input_ids")
            if _tensor_shape(input_ids)[:1] != (1,):
                raise BackendIntegrationError(
                    "Qwen3-VL compressed video core requires one input_ids row."
                )
            video_grid_thw = _bound_value(bound, "video_grid_thw")
            grid_rows = _native_rows(video_grid_thw)
            if len(grid_rows) != 1:
                raise BackendIntegrationError(
                    "Qwen3-VL video compression supports exactly one video."
                )
            raw_timing = _bound_value(bound, "second_per_grid_ts")
            timing = _optional_single_float(raw_timing, "second_per_grid_ts")
            merge_size = int(getattr(self.visual, "spatial_merge_size", 0))
            grid = Qwen3VLVideoGrid.from_native(
                grid_rows[0],
                spatial_merge_size=merge_size,
                second_per_grid_seconds=timing,
            )
            video_token_id = getattr(self.core.config, "video_token_id", None)
            if isinstance(video_token_id, bool) or not isinstance(
                video_token_id, Integral
            ):
                raise BackendIntegrationError(
                    "Pinned Qwen3-VL config does not expose video_token_id."
                )
            placeholders = (
                (input_ids[0] == int(video_token_id)).nonzero(as_tuple=False).flatten()
            )
            positions = tuple(int(value) for value in placeholders.tolist())
            if len(positions) != grid.token_count:
                raise BackendIntegrationError(
                    "Qwen3-VL video placeholder count differs from source grid."
                )
            attention_mask = _bound_value(bound, "attention_mask")
            if attention_mask is not None:
                active = attention_mask[0, placeholders].bool()
                if not bool(active.all().item()):
                    raise BackendIntegrationError(
                        "Qwen3-VL active video placeholders cannot be padded."
                    )
            metadata = dict(self._request_metadata.get() or {})
            media_id = _single_video_media_id(
                metadata,
                fallback=f"video-{self._invocation_counter + 1}",
            )
            source = _ActiveSource(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pixel_values_videos=_bound_value(bound, "pixel_values_videos"),
                video_grid_thw=video_grid_thw,
                second_per_grid_ts=raw_timing,
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
                        "Qwen3-VL video prefill missed splice legalization."
                    )
                rope_deltas = self._vtp_rope_deltas.get()
                if rope_deltas is None:
                    rope_deltas = prompt.rope_deltas
                _set_output_field(output, "rope_deltas", rope_deltas)
                self.core.rope_deltas = rope_deltas
                self._callback_count += 1
                self._latest_splice = {
                    "invocation_id": splice.request.invocation_id,
                    "media_id": splice.request.media_id,
                    "source_video_grid_thw": source.grid.source_grid_thw,
                    "second_per_grid_seconds": source.grid.second_per_grid_seconds,
                    **prompt.record(),
                    "projector": self._latest_projector,
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

    @contextmanager
    def _raw_projector_groups(self, token_count: int) -> Iterator[list[int]]:
        """Defer native projectors while retaining their complete spatial groups."""
        mergers = (self.visual.merger, *self.visual.deepstack_merger_list)
        merge_unit = int(self.visual.spatial_merge_size) ** 2
        widths: list[int] = [0] * len(mergers)
        patches: list[_InstancePatch] = []

        def deferred(index: int) -> Callable[..., Any]:
            def forward(_merger: Any, hidden_states: Any) -> Any:
                shape = _feature_shape(hidden_states)
                if len(shape) != 2 or shape[0] != token_count * merge_unit:
                    raise InvariantError("Qwen3-VL raw projector input lost complete spatial groups.")
                widths[index] = shape[1]
                return hidden_states.reshape(token_count, merge_unit * shape[1])
            return forward

        try:
            for index, merger in enumerate(mergers):
                patches.append(self._patch_bound(merger, "forward", deferred(index), merger.forward))
            yield widths
        finally:
            for patch in reversed(patches):
                patch.restore()

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
            before = self._projector_position == "before"
            raw_groups = self._raw_projector_groups(source.grid.token_count) if before else nullcontext(())
            with capture as signal_capture, raw_groups as raw_widths:
                native = original(*bound.args, **bound.kwargs)
            if not isinstance(native, (tuple, list)) or len(native) != 2:
                raise BackendIntegrationError(
                    "Pinned Qwen3-VL get_video_features must return "
                    "(feature_chunks, deepstack_features)."
                )
            features_by_video, deepstack_features = native
            if (
                not isinstance(features_by_video, (tuple, list))
                or len(features_by_video) != 1
            ):
                raise BackendIntegrationError(
                    "Pinned Qwen3-VL get_video_features must return one feature chunk."
                )
            if not isinstance(deepstack_features, (tuple, list)):
                raise BackendIntegrationError(
                    "Pinned Qwen3-VL get_video_features changed DeepStack output."
                )
            expected_branches = len(getattr(self.visual, "deepstack_merger_list", ()))
            if len(deepstack_features) != expected_branches:
                raise InvariantError(
                    "Qwen3-VL video DeepStack branch count differs from the visual "
                    "projector count."
                )
            features = features_by_video[0]
            self._invocation_counter += 1
            invocation_id = f"video-splice-{self._invocation_counter}"
            payload = source.grid.payload(
                features,
                media_id=source.media_id,
                signals={
                    "qwen3_vl.prompt_input_ids": source.input_ids,
                    "qwen3_vl.pixel_values_videos": source.pixel_values_videos,
                    "qwen3_vl.deepstack_features": tuple(deepstack_features),
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
                metadata={
                    "invocation_id": invocation_id,
                    "projector_position": self._projector_position,
                    "actual_projector_site": QWEN3_VL_VIDEO_PRE_PROJECTOR_SITE if before else QWEN3_VL_VIDEO_SPLICE_SITE,
                    "value_space": "raw_spatial_groups" if before else "projected_features",
                },
            )
            enrich_payload = (
                None
                if signal_provider is None
                else getattr(signal_provider, "enrich_payload", None)
            )
            dense_reference_rows = 0
            needs_projected_reference = before and (
                self._preserve_projected_values or getattr(signal_provider, "requires_projected_reference", False))
            enrichment_payload = payload
            if needs_projected_reference:
                with _torch().no_grad():
                    reference = self.visual.merger(features.reshape(-1, raw_widths[0]))
                enrichment_payload = replace(payload, features=reference,
                    metadata={**payload.metadata, "dense_projected_reference_tokens": source.grid.token_count})
                dense_reference_rows = source.grid.token_count
            if callable(enrich_payload):
                payload = enrich_payload(enrichment_payload, signal_capture)
                if not isinstance(payload, VideoTokenPayload):
                    raise BackendIntegrationError(
                        "Qwen3-VL video signal provider returned an invalid payload"
                    )
            else:
                payload = enrichment_payload
            if needs_projected_reference:
                payload = replace(payload, features=features, signals={**payload.signals,
                    **({"projector_reference_features": reference} if self._preserve_projected_values else {})})
            request = Qwen3VLPackedVideoRequest(
                invocation_id=invocation_id,
                media_id=source.media_id,
                payload=payload,
                deepstack_features=tuple(deepstack_features),
                grid=source.grid,
                source_placeholder_positions=source.source_placeholder_positions,
                prompt_input_ids=source.input_ids,
                source_attention_mask=source.attention_mask,
                source_video_grid_thw=source.video_grid_thw,
                source_second_per_grid_ts=source.second_per_grid_ts,
            )
            response = self._callback(request)
            if not isinstance(response, Qwen3VLVideoCompressionResponse):
                raise BackendIntegrationError(
                    "Qwen3-VL video callback must return "
                    "Qwen3VLVideoCompressionResponse."
                )
            result_shape = _feature_shape(response.compression_result.features)
            source_shape = _feature_shape(features)
            if len(result_shape) != 2 or result_shape[1:] != source_shape[1:]:
                raise InvariantError("Qwen3-VL compressed video hidden size changed.")
            plan = plan_qwen3_vl_video_splice(
                request,
                response,
                sequence_length=int(source.input_ids.shape[1]),
            )
            if before:
                mergers = (self.visual.merger, *self.visual.deepstack_merger_list)
                values = (response.compression_result.features, *response.deepstack_features)
                if any(width <= 0 for width in raw_widths):
                    raise InvariantError("Qwen3-VL native visual pass did not visit every projector.")
                projected = tuple(merger(value.reshape(-1, width))
                                  for merger, value, width in zip(mergers, values, raw_widths))
                response = replace(response,
                    compression_result=replace(response.compression_result, features=projected[0]),
                    deepstack_features=projected[1:])
                expected_width = int(self.language_model.embed_tokens.weight.shape[1])
                if _feature_shape(projected[0]) != (result_shape[0], expected_width):
                    raise InvariantError("Qwen3-VL compressed projectors changed the language hidden size.")
            self._latest_projector = {
                "projector_position": self._projector_position,
                "actual_projector_site": QWEN3_VL_VIDEO_PRE_PROJECTOR_SITE if before else QWEN3_VL_VIDEO_SPLICE_SITE,
                "value_space": "raw_spatial_groups" if before else "projected_features",
                "source_group_count": source.grid.token_count,
                "result_group_count": result_shape[0],
                "callback_hidden_size": source_shape[1],
                "projector_input_patch_rows": ((result_shape[0] if before else source.grid.token_count) + dense_reference_rows) * source.grid.spatial_merge_size ** 2,
                "projector_output_rows": (result_shape[0] if before else source.grid.token_count) + dense_reference_rows,
                "dense_main_scoring_reference_rows": dense_reference_rows,
                "deepstack_projector_input_patch_rows": (result_shape[0] if before else source.grid.token_count) * source.grid.spatial_merge_size ** 2,
                "deepstack_branch_count": len(response.deepstack_features),
            }
            self._active_splice.set(_ActiveSplice(request, response, plan))
            return (
                (response.compression_result.features,),
                list(response.deepstack_features),
            )

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
            input_ids = _bound_value(bound, "input_ids")
            inputs_embeds = _bound_value(bound, "inputs_embeds")
            supplied_video_features = _bound_value(bound, "video_features")
            _bound_assign(bound, "video_features", None)
            image_mask, dense_video_mask = original(*bound.args, **bound.kwargs)
            dense = dense_video_mask[..., 0].bool()
            if _tensor_shape(dense) != _tensor_shape(input_ids):
                raise InvariantError(
                    "Qwen3-VL dense video placeholder mask changed shape."
                )
            positions = tuple(
                int(value)
                for value in dense[0].nonzero(as_tuple=False).flatten().tolist()
            )
            if positions != splice.plan.source_placeholder_positions:
                raise InvariantError(
                    "Qwen3-VL native video placeholder mask changed source order."
                )
            torch = _torch()
            selected = torch.zeros_like(dense, dtype=torch.bool)
            selected[0, list(splice.plan.retained_placeholder_positions)] = True
            expanded = selected.unsqueeze(-1).expand_as(inputs_embeds)
            if supplied_video_features is None or (
                inputs_embeds[expanded].numel() != supplied_video_features.numel()
            ):
                raise InvariantError(
                    "Qwen3-VL compressed placeholders do not match video features."
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
            prompt = legalize_qwen3_vl_video_prompt(
                input_ids=source.input_ids,
                inputs_embeds=_bound_value(bound, "inputs_embeds"),
                attention_mask=_bound_value(bound, "attention_mask"),
                position_ids=_bound_value(bound, "position_ids"),
                cache_position=_bound_value(bound, "cache_position"),
                labels=self._training_labels.get(),
                visual_pos_masks=_bound_value(bound, "visual_pos_masks"),
                deepstack_visual_embeds=_bound_value(bound, "deepstack_visual_embeds"),
                request=splice.request,
                response=splice.response,
            )
            _bound_assign(bound, "input_ids", None)
            _bound_assign(bound, "inputs_embeds", prompt.inputs_embeds)
            _bound_assign(bound, "attention_mask", prompt.attention_mask)
            _bound_assign(bound, "position_ids", prompt.position_ids)
            _bound_assign(bound, "cache_position", prompt.cache_position)
            _bound_assign(bound, "visual_pos_masks", prompt.visual_pos_masks)
            _bound_assign(
                bound,
                "deepstack_visual_embeds",
                list(prompt.deepstack_visual_embeds),
            )
            if self._training_labels.get() is not None:
                self._training_labels.set(prompt.labels)
            self._prompt_splice.set(prompt)
            return _call_bound_method(original, bound)

        return replacement

    def _repair_decode_call(
        self,
        bound: inspect.BoundArguments,
        lifecycle: _GenerationLifecycle,
    ) -> None:
        if not lifecycle.prefill_complete:
            return
        source_length = lifecycle.source_sequence_length
        result_length = lifecycle.result_sequence_length
        if source_length is None or result_length is None:
            raise InvariantError("Qwen3-VL generation lifecycle lost prompt lengths.")
        removed = lifecycle.removed_placeholder_positions
        removed_count = len(removed)
        cache_position = _bound_value(bound, "cache_position")
        cache_values = _tensor_int_values(cache_position)
        attention_mask = _bound_value(bound, "attention_mask")
        attention_shape = _tensor_shape(attention_mask)
        dense_cache = bool(cache_values and cache_values[0] >= source_length)
        dense_mask = bool(
            len(attention_shape) == 2
            and attention_shape[0] == 1
            and attention_shape[1] >= source_length
        )
        if not dense_cache and not dense_mask and removed_count:
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
            _bound_assign(
                bound,
                "attention_mask",
                attention_mask.index_select(1, keep_tensor),
            )
        if dense_cache and removed_count:
            _bound_assign(bound, "cache_position", cache_position - removed_count)
        position_ids = _bound_value(bound, "position_ids")
        if position_ids is not None and removed_count and (dense_cache or dense_mask):
            _bound_assign(bound, "position_ids", position_ids - removed_count)
        lifecycle.decode_repairs += 1
        output_attention = _bound_value(bound, "attention_mask")
        output_attention_shape = _tensor_shape(output_attention)
        lifecycle.last_decode_record = {
            "input_cache_position": cache_values,
            "output_cache_position": _tensor_int_values(
                _bound_value(bound, "cache_position")
            ),
            "input_attention_length": (
                attention_shape[1] if len(attention_shape) == 2 else None
            ),
            "output_attention_length": (
                output_attention_shape[1] if len(output_attention_shape) == 2 else None
            ),
            "removed_count": removed_count,
            "already_compact": False,
        }


def install_qwen3_vl_video_bridge(model: Any) -> PinnedQwen3VLVideoBridge:
    """Attach or return the exact video bridge on one Qwen3-VL instance."""

    for attribute in (_VIDEO_BRIDGE_ATTRIBUTE, _PUBLIC_VIDEO_BRIDGE_ATTRIBUTE):
        existing = getattr(model, attribute, None)
        if existing is not None:
            if isinstance(existing, PinnedQwen3VLVideoBridge):
                return existing
            raise BackendIntegrationError(
                f"Model already exposes an incompatible video bridge at {attribute}."
            )
    if any(
        getattr(model, attribute, None) is not None
        for attribute in _IMAGE_PROVIDER_ATTRIBUTES
    ):
        raise BackendIntegrationError(
            "Qwen3-VL image and video provider bridges cannot patch the same "
            "model instance simultaneously."
        )
    bridge = PinnedQwen3VLVideoBridge(model)
    setattr(model, _VIDEO_BRIDGE_ATTRIBUTE, bridge)
    setattr(model, _PUBLIC_VIDEO_BRIDGE_ATTRIBUTE, bridge)
    bridge._attached_attribute = _VIDEO_BRIDGE_ATTRIBUTE
    return bridge


@contextmanager
def qwen3_vl_video_request_context(
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
        _validate_qwen3_vl_structure,
    )

    core = getattr(model, "model", None)
    visual = getattr(core, "visual", None)
    language_model = getattr(core, "language_model", None)
    _validate_provider_version()
    _validate_qwen3_vl_structure(
        model,
        core=core,
        visual=visual,
        language_model=language_model,
    )
    required_signatures = (
        (model.forward, {"pixel_values_videos", "video_grid_thw"}, "model.forward"),
        (
            core.forward,
            {"pixel_values_videos", "video_grid_thw"},
            "model.model.forward",
        ),
        (
            core.get_video_features,
            {"pixel_values_videos", "video_grid_thw"},
            "model.model.get_video_features",
        ),
        (
            language_model.forward,
            {"visual_pos_masks", "deepstack_visual_embeds"},
            "model.model.language_model.forward",
        ),
    )
    for function, required, name in required_signatures:
        parameters = set(inspect.signature(function).parameters)
        missing = sorted(required - parameters)
        if missing:
            raise BackendIntegrationError(
                f"Pinned Qwen3-VL video signature {name} is missing {missing}."
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


def _optional_single_float(value: Any, name: str) -> float | None:
    if value is None:
        return None
    raw = value.tolist() if callable(getattr(value, "tolist", None)) else value
    if isinstance(raw, (tuple, list)):
        if len(raw) != 1:
            raise BackendIntegrationError(f"{name} must contain one value.")
        raw = raw[0]
    if isinstance(raw, bool) or not isinstance(raw, (int, float)):
        raise BackendIntegrationError(f"{name} must contain one finite number.")
    result = float(raw)
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
                "Qwen3-VL video_media_ids must contain exactly one identity."
            )
        media_ids = tuple(raw)
        if len(media_ids) != 1 or not isinstance(media_ids[0], str):
            raise BackendIntegrationError(
                "Qwen3-VL video_media_ids must contain exactly one identity."
            )
        media_id = media_ids[0].strip()
        if not media_id:
            raise BackendIntegrationError(
                "Qwen3-VL video media identity must be non-empty."
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
        raise BackendIntegrationError("Qwen3-VL video features must be tensor-like.")
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
            "Qwen3-VL video feature rows must be rectangular."
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
        raise InvariantError("Qwen3-VL compressed prompt has no active tokens.")
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


def _bound_value(bound: inspect.BoundArguments, name: str) -> Any:
    if name in bound.arguments:
        return bound.arguments[name]
    for parameter_name, parameter in bound.signature.parameters.items():
        if parameter.kind is inspect.Parameter.VAR_KEYWORD:
            extras = bound.arguments.get(parameter_name)
            if isinstance(extras, Mapping) and name in extras:
                return extras[name]
    return None


def _bound_assign(bound: inspect.BoundArguments, name: str, value: Any) -> None:
    if name in bound.signature.parameters:
        bound.arguments[name] = value
        return
    for parameter_name, parameter in bound.signature.parameters.items():
        if parameter.kind is inspect.Parameter.VAR_KEYWORD:
            extras = dict(bound.arguments.get(parameter_name) or {})
            extras[name] = value
            bound.arguments[parameter_name] = extras
            return
    raise BackendIntegrationError(
        f"Pinned Qwen3-VL provider call cannot assign field {name!r}."
    )


def _call_bound_method(
    method: Callable[..., Any],
    bound: inspect.BoundArguments,
) -> Any:
    positional = []
    keyword: dict[str, Any] = {}
    extras: Mapping[str, Any] = {}
    for name, parameter in bound.signature.parameters.items():
        if name not in bound.arguments:
            continue
        value = bound.arguments[name]
        if parameter.kind is inspect.Parameter.POSITIONAL_ONLY:
            positional.append(value)
        elif parameter.kind is inspect.Parameter.VAR_POSITIONAL:
            positional.extend(value)
        elif parameter.kind is inspect.Parameter.VAR_KEYWORD:
            if not isinstance(value, Mapping):
                raise BackendIntegrationError(
                    "Pinned Qwen3-VL provider **kwargs changed type."
                )
            extras = value
        else:
            keyword[name] = value
    for name, value in extras.items():
        # Transformers' return-dict decorator can retain a field in its
        # variadic kwargs while exposing the same field in ``__signature__``.
        # The explicit pinned parameter is authoritative in that case.
        keyword.setdefault(name, value)
    return method(*positional, **keyword)


def _set_output_field(output: Any, name: str, value: Any) -> None:
    if isinstance(output, MutableMapping):
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
        raise RuntimeError("Qwen3-VL video compression requires PyTorch.") from exc
    return torch


__all__ = [
    "QWEN3_VL_VIDEO_BRIDGE_API_VERSION",
    "QWEN3_VL_VIDEO_SPLICE_SITE",
    "PinnedQwen3VLVideoBridge",
    "Qwen3VLPackedVideoRequest",
    "Qwen3VLVideoCallbackHandle",
    "Qwen3VLVideoCompressionResponse",
    "Qwen3VLVideoGrid",
    "Qwen3VLVideoPromptSpliceResult",
    "Qwen3VLVideoSplicePlan",
    "install_qwen3_vl_video_bridge",
    "legalize_qwen3_vl_video_prompt",
    "plan_qwen3_vl_video_splice",
    "qwen3_vl_video_request_context",
]
