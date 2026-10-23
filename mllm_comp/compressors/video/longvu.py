"""Canonical LongVU frame selection and trainable SVA video fusion.

The DINOv2 encoder and the learnable fusion module are capability-provider
owned.  This module owns only the backbone-neutral algorithm boundary:
deterministic DINO frame selection, dense raw-raster interpolation, exact
12x12 structural source groups, and the result/LayoutDelta contract.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

from mllm_comp.core.errors import BackendIntegrationError, ConfigError, InvariantError

from .config import (
    VIDEO_REFERENCE_REVISION,
    resolve_canonical_video_config,
    validate_video_payload_signals,
)
from .contracts import VideoCompressionResult, VideoTokenPayload
from . import _rules as rule_contract


_ALGORITHM = "longvu"
_VARIANT = "longvu.video"


class LongVUVideoKernel:
    """Registry kernel for the single frozen LongVU video profile."""

    name = _ALGORITHM
    version = f"videollm-comp-{VIDEO_REFERENCE_REVISION[:8]}-longvu-v1"

    def validate_config(self, plugin_config: Mapping[str, Any]) -> None:
        if not isinstance(plugin_config, Mapping):
            raise ConfigError("video plugin config must be a mapping")
        contract = plugin_config.get("modality_contract", {})
        if contract:
            if not isinstance(contract, Mapping):
                raise ConfigError("modality_contract must be a mapping")
            if contract.get("algorithm") != self.name:
                raise ConfigError("video plugin algorithm must be 'longvu'")
            if contract.get("variant") != _VARIANT:
                raise ConfigError("video plugin variant must be 'longvu.video'")
        resolve_canonical_video_config(
            self.name,
            plugin_config.get("config", {}),
        )

    def __call__(
        self,
        payload: VideoTokenPayload,
        *,
        fusion_module: Any | None = None,
        config: Mapping[str, Any] | None = None,
        frame_selection_site: str = "pre_sva",
        rules: Mapping[str, Any] | None = None,
    ) -> VideoCompressionResult:
        return longvu_video(
            payload,
            fusion_module=fusion_module,
            config=config,
            frame_selection_site=frame_selection_site,
            rules=rules,
        )


LongVUKernel = LongVUVideoKernel


def longvu_video(
    payload: VideoTokenPayload,
    *,
    fusion_module: Any | None,
    config: Mapping[str, Any] | None = None,
    frame_selection_site: str = "pre_sva",
    rules: Mapping[str, Any] | None = None,
) -> VideoCompressionResult:
    """Run frozen DINO frame selection and learnable SVA fusion per video."""

    torch = _torch()
    if frame_selection_site not in {"pre_sva", "pre_output_projector", "post_output_projector"}:
        raise ConfigError(f"Unsupported LongVU frame selection site: {frame_selection_site!r}")
    if not isinstance(payload, VideoTokenPayload):
        raise TypeError("payload must be a VideoTokenPayload")
    _validate_feature_tensor(payload.features, "projected features", torch=torch)
    if payload.raw_features is None:
        raise BackendIntegrationError(
            "LongVU requires provider-owned pre-projector/native raw features"
        )
    _validate_feature_tensor(payload.raw_features, "raw features", torch=torch)
    validate_video_payload_signals(_ALGORITHM, payload)
    if fusion_module is None or not callable(fusion_module):
        raise BackendIntegrationError(
            "LongVU requires a registered trainable SVA fusion module"
        )
    if frame_selection_site != "pre_sva" and frame_selection_site not in getattr(
        fusion_module, "supported_frame_selection_sites", (),
    ):
        raise BackendIntegrationError("LongVU fusion module does not support moved frame selection")
    if payload.token_mask is not None:
        mask = torch.as_tensor(payload.token_mask, device=payload.features.device)
        if mask.ndim != 1 or int(mask.numel()) != payload.token_count:
            raise ValueError("LongVU token_mask must match the projected layout")
        if not bool(mask.bool().all().item()):
            raise BackendIntegrationError(
                "LongVU requires provider-trimmed videos without padded tokens"
            )

    resolved = resolve_canonical_video_config(_ALGORITHM, config or {})
    interp_side = math.isqrt(int(resolved["dino_interp_size"]))
    query_side = math.isqrt(int(resolved["query_num"]))
    if (
        interp_side * interp_side != int(resolved["dino_interp_size"])
        or query_side * query_side != int(resolved["query_num"])
        or interp_side % query_side
    ):
        raise InvariantError("Canonical LongVU spatial sizes are inconsistent")

    dino = payload.signals["dinov2_features"]
    pixels = payload.signals["pixels"]
    _validate_rank_three(dino, "dinov2_features", torch=torch)
    if getattr(pixels, "ndim", None) != 4:
        raise BackendIntegrationError(
            "LongVU pixels must be a rank-four provider-normalized frame tensor"
        )
    if (
        dino.device != payload.features.device
        or pixels.device != payload.features.device
    ):
        raise BackendIntegrationError(
            "LongVU raw, DINO, pixels, and projected features must share a device"
        )
    if payload.raw_features.device != payload.features.device:
        raise BackendIntegrationError(
            "LongVU raw and projected features must share a device"
        )
    if int(dino.shape[1]) != int(resolved["dino_interp_size"]):
        raise BackendIntegrationError(
            "LongVU DINO provider did not emit the canonical 24x24 raster"
        )
    if not bool(torch.isfinite(dino).all().item()):
        raise BackendIntegrationError("LongVU DINO features must be finite")

    projected_frames_by_video = tuple(
        _dense_frame_spans(
            payload.coordinates[payload.video_span(video_index)],
            name="projected",
        )
        for video_index in range(payload.video_count)
    )
    raw_frames_by_video = tuple(
        _dense_frame_spans(
            payload.raw_coordinates[payload.raw_video_span(video_index)],
            name="raw",
        )
        for video_index in range(payload.video_count)
    )
    frame_counts = tuple(len(frames) for frames in projected_frames_by_video)
    if any(
        tuple(item[0] for item in projected) != tuple(item[0] for item in raw)
        for projected, raw in zip(projected_frames_by_video, raw_frames_by_video)
    ):
        raise BackendIntegrationError(
            "LongVU projected and raw streams have different temporal identities"
        )
    dino_offsets = _frame_offsets(
        payload.signals.get("dinov2_video_offsets"),
        frame_counts,
        total=int(dino.shape[0]),
        name="dinov2_video_offsets",
    )
    pixel_offsets = _frame_offsets(
        payload.signals.get("pixel_video_offsets"),
        frame_counts,
        total=int(pixels.shape[0]),
        name="pixel_video_offsets",
    )
    if dino_offsets != pixel_offsets:
        raise BackendIntegrationError(
            "LongVU DINO and pixel frame boundaries must be identical"
        )

    outputs = []
    groups_by_video = []
    representatives_by_video = []
    exact_budgets = []
    selected_by_video = []
    frame_scores_by_video = []
    for video_index, (projected_frames, raw_frames) in enumerate(
        zip(projected_frames_by_video, raw_frames_by_video)
    ):
        dino_start, dino_stop = dino_offsets[video_index : video_index + 2]
        dino_video = dino[dino_start:dino_stop]
        selected, scores = select_longvu_frames(
            dino_video,
            window_size=int(resolved["window_size"]),
            threshold=float(resolved["threshold"]),
            max_frames=int(resolved["max_frames"]),
            min_frames=int(resolved["min_frames"]),
            rules=rules,
        )
        selected_indices = tuple(int(value) for value in selected.tolist())
        raw_rasters = []
        dino_rasters = []
        groups = []
        anchors = []
        fusion_indices = selected_indices if frame_selection_site == "pre_sva" else range(len(raw_frames))
        for frame_index in fusion_indices:
            _temporal, raw_start, raw_stop, raw_height, raw_width = raw_frames[
                frame_index
            ]
            raw_frame = payload.raw_features[
                payload.raw_video_offsets[video_index]
                + raw_start : payload.raw_video_offsets[video_index]
                + raw_stop
            ]
            raw_rasters.append(
                _interpolate_raster(
                    raw_frame,
                    height=raw_height,
                    width=raw_width,
                    side=interp_side,
                    torch=torch,
                )
            )
            dino_rasters.append(dino_video[frame_index])

        for selected_index in selected_indices:
            _temporal, start, stop, height, width = projected_frames[selected_index]
            frame_groups = longvu_structural_groups(
                height,
                width,
                query_side=query_side,
                source_offset=start,
            )
            if frame_groups[-1][-1] >= stop:
                raise InvariantError("LongVU structural group escaped its frame span")
            groups.extend(frame_groups)
            anchors.extend(group[0] for group in frame_groups)

        raw_batch = torch.stack(raw_rasters, dim=0)
        dino_batch = torch.stack(dino_rasters, dim=0)
        def default_transform():
            if frame_selection_site == "pre_sva":
                return fusion_module(raw_batch, dino_batch)
            return fusion_module(raw_batch, dino_batch,
                selected_frame_indices=selected, frame_selection_site=frame_selection_site)
        expected_shape = (
            len(selected_indices),
            int(resolved["query_num"]),
            int(payload.features.shape[-1]),
        )
        if rules and "fusion_transform" in rules:
            fused = rule_contract.value(rules["fusion_transform"],
                {"values": raw_batch, "dino_features": dino_batch, "module": fusion_module,
                 "selected_frame_indices": selected, "frame_selection_site": frame_selection_site,
                 "default_transform": default_transform}, reference=raw_batch,
                shape=expected_shape, name="fusion_transform")
        else:
            fused = default_transform()
        if tuple(fused.shape) != expected_shape:
            raise BackendIntegrationError(
                "LongVU SVA fusion returned an incompatible shape: "
                f"expected={expected_shape}, actual={tuple(fused.shape)}"
            )
        fused = fused.reshape(-1, int(fused.shape[-1]))
        if not bool(torch.isfinite(fused).all().item()):
            raise BackendIntegrationError("LongVU SVA fusion emitted non-finite values")
        if int(fused.shape[0]) >= (
            payload.video_offsets[video_index + 1] - payload.video_offsets[video_index]
        ):
            raise BackendIntegrationError(
                "Canonical LongVU did not produce a non-identity token reduction"
            )
        ordered_anchors = tuple(anchors)
        if ordered_anchors != tuple(sorted(ordered_anchors)) or len(
            set(ordered_anchors)
        ) != len(ordered_anchors):
            raise InvariantError(
                "LongVU structural anchors must be unique and source ordered"
            )
        outputs.append(fused)
        groups_by_video.append(tuple(groups))
        representatives_by_video.append(ordered_anchors)
        exact_budgets.append(len(groups))
        selected_by_video.append(selected_indices)
        frame_scores_by_video.append(scores)

    return VideoCompressionResult.from_source_groups(
        payload,
        torch.cat(outputs, dim=0),
        source_groups=groups_by_video,
        representative_indices=representatives_by_video,
        exact_budgets=exact_budgets,
        metadata={
            "algorithm": _VARIANT,
            "reference_revision": VIDEO_REFERENCE_REVISION,
            "canonical_config": dict(resolved),
            "stages": {
                "pre_sva": ("dinov2_frame_selection", "sva_fusion", "output_projection"),
                "pre_output_projector": ("sva_fusion", "dinov2_frame_selection", "output_projection"),
                "post_output_projector": ("sva_fusion", "output_projection", "dinov2_frame_selection"),
            }[frame_selection_site],
            "frame_selection_site": frame_selection_site,
            "sva_site": "pre_output_projector",
            "sva_frame_counts_by_video": tuple(len(selected) if frame_selection_site == "pre_sva" else count
                for selected, count in zip(selected_by_video, frame_counts)),
            "output_projector_frame_counts_by_video": tuple(count if frame_selection_site == "post_output_projector" else len(selected)
                for selected, count in zip(selected_by_video, frame_counts)),
            "extra_sva_frames_by_video": tuple(0 if frame_selection_site == "pre_sva" else count - len(selected)
                for selected, count in zip(selected_by_video, frame_counts)),
            "extra_output_projector_frames_by_video": tuple(count - len(selected) if frame_selection_site == "post_output_projector" else 0
                for selected, count in zip(selected_by_video, frame_counts)),
            "selected_frame_indices_by_video": tuple(selected_by_video),
            "frame_similarity_scores_by_video": tuple(frame_scores_by_video),
            "tokens_per_selected_frame": int(resolved["query_num"]),
            "frame_selection_tie_policy": "custom" if rules and ('frame_selector' in rules) else "stable_source_order_middle_forced",
            "layout_source_groups": "dense_raster_partition_12x12",
            "layout_anchor_policy": "first_source_in_structural_query_cell",
            "value_lineage_policy": "custom" if rules and ('fusion_transform' in rules) else "trainable_sva_native_dino_cross_attention",
            "trainable_compressor": True,
            "dino_frozen": True,
        },
    )


def select_longvu_frames(
    dino_features: Any,
    *,
    window_size: int,
    threshold: float,
    max_frames: int,
    min_frames: int,
    rules: Mapping[str, Any] | None = None,
) -> tuple[Any, tuple[tuple[float, ...], ...]]:
    """Return source-ordered frame indices using the released LongVU rule."""

    torch = _torch()
    _validate_rank_three(dino_features, "dinov2_features", torch=torch)
    frame_count = int(dino_features.shape[0])
    if frame_count < 1:
        raise BackendIntegrationError("LongVU requires at least one frame")
    selected = []
    score_records = []
    for start in range(0, frame_count, window_size):
        stop = min(start + window_size, frame_count)
        flattened = dino_features[start:stop].flatten(1)
        norms = torch.linalg.vector_norm(flattened, dim=1)
        if bool((norms <= 0).any().item()) or not bool(
            torch.isfinite(norms).all().item()
        ):
            raise BackendIntegrationError(
                "LongVU DINO frame vectors must be finite and non-zero"
            )
        normalised = flattened / norms.unsqueeze(1)
        scores = (normalised @ normalised.transpose(0, 1)).mean(dim=1)
        if rules and "frame_scorer" in rules:
            scores = rule_contract.scores(rules["frame_scorer"],
                {"values": dino_features[start:stop], "default_scores": scores,
                 "frame_offset": start}, count=stop-start,
                reference=dino_features, name="frame_scorer")
        if not bool(torch.isfinite(scores).all().item()):
            raise BackendIntegrationError("LongVU frame similarities must be finite")
        forced = scores.clone()
        forced[(stop - start) // 2] = 0.0
        local = torch.where(forced < threshold)[0]
        if int(local.numel()) == 0:
            local = torch.tensor(
                [(stop - start) // 2],
                device=dino_features.device,
                dtype=torch.long,
            )
        selected.extend(int(value) + start for value in local.tolist())
        score_records.append(
            tuple(float(value) for value in forced.detach().cpu().tolist())
        )

    if len(selected) > max_frames:
        interval = len(selected) / float(max_frames)
        selected = [selected[int(interval * index)] for index in range(max_frames)]
    if len(selected) < min_frames:
        step = max(1, frame_count // min_frames)
        selected = list(range(0, frame_count, step))[:min_frames]
    if rules and "frame_selector" in rules:
        selected = sorted(rule_contract.select(rules["frame_selector"],
            {"values": dino_features, "window_scores": tuple(score_records),
             "default_selected_indices": tuple(selected), "min_frames": min_frames,
             "max_frames": max_frames}, count=None, candidates=range(frame_count), name="frame_selector"))
        if not min(min_frames, frame_count) <= len(selected) <= min(max_frames, frame_count):
            raise BackendIntegrationError("LongVU frame_selector must respect the frame budget bounds")
    if not selected or selected != sorted(set(selected)):
        raise InvariantError("LongVU frame selection lost stable source order")
    return (
        torch.tensor(selected, device=dino_features.device, dtype=torch.long),
        tuple(score_records),
    )


def longvu_structural_groups(
    height: int,
    width: int,
    *,
    query_side: int = 12,
    source_offset: int = 0,
) -> tuple[tuple[int, ...], ...]:
    """Partition one dense raster into ordered query-cell source groups."""

    for name, value in (
        ("height", height),
        ("width", width),
        ("query_side", query_side),
        ("source_offset", source_offset),
    ):
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(f"{name} must be an integer")
    if height < query_side or width < query_side or query_side < 1:
        raise BackendIntegrationError(
            "LongVU requires each projected frame raster to be at least 12x12"
        )
    if source_offset < 0:
        raise ValueError("source_offset must be non-negative")
    groups = []
    for query_row in range(query_side):
        row_start = query_row * height // query_side
        row_stop = (query_row + 1) * height // query_side
        for query_column in range(query_side):
            column_start = query_column * width // query_side
            column_stop = (query_column + 1) * width // query_side
            group = tuple(
                source_offset + row * width + column
                for row in range(row_start, row_stop)
                for column in range(column_start, column_stop)
            )
            if not group:
                raise InvariantError("LongVU structural query cell is empty")
            groups.append(group)
    flattened = tuple(index for group in groups for index in group)
    expected = tuple(range(source_offset, source_offset + height * width))
    if tuple(sorted(flattened)) != expected or len(set(flattened)) != len(flattened):
        raise InvariantError("LongVU structural groups must partition the raster")
    return tuple(groups)


def pool_longvu_source_groups(
    features: Any,
    result: VideoCompressionResult,
    *,
    video_index: int = 0,
) -> Any:
    """Mean-pool the same structural groups on one Qwen3 DeepStack branch."""

    torch = _torch()
    _validate_feature_tensor(features, "synchronized features", torch=torch)
    if not isinstance(result, VideoCompressionResult):
        raise TypeError("result must be a VideoCompressionResult")
    source_span = result.source_video_span(video_index)
    if int(features.shape[0]) != source_span.stop - source_span.start:
        raise InvariantError(
            "synchronized LongVU branch does not match the source video layout"
        )
    values = [
        features.index_select(
            0,
            torch.tensor(group, device=features.device, dtype=torch.long),
        ).mean(dim=0)
        for group in result.source_groups[video_index]
    ]
    if not values:
        raise InvariantError("LongVU result has no structural source groups")
    return torch.stack(values, dim=0)


def _dense_frame_spans(
    coordinates: Sequence[Any],
    *,
    name: str,
) -> tuple[tuple[int, int, int, int, int], ...]:
    if not coordinates:
        raise BackendIntegrationError(f"LongVU {name} coordinates are empty")
    spans = []
    start = 0
    while start < len(coordinates):
        temporal = int(coordinates[start].t)
        stop = start + 1
        while stop < len(coordinates) and int(coordinates[stop].t) == temporal:
            stop += 1
        frame = coordinates[start:stop]
        rows = tuple(int(item.h) for item in frame)
        columns = tuple(int(item.w) for item in frame)
        height = max(rows) + 1
        width = max(columns) + 1
        expected = tuple(
            (row, column) for row in range(height) for column in range(width)
        )
        actual = tuple((int(item.h), int(item.w)) for item in frame)
        if actual != expected:
            raise BackendIntegrationError(
                f"LongVU {name} temporal slice is not a source-ordered dense raster"
            )
        spans.append((temporal, start, stop, height, width))
        start = stop
    if tuple(item[0] for item in spans) != tuple(
        sorted(set(item[0] for item in spans))
    ):
        raise BackendIntegrationError(
            f"LongVU {name} temporal coordinates must be unique and ordered"
        )
    return tuple(spans)


def _interpolate_raster(
    features: Any,
    *,
    height: int,
    width: int,
    side: int,
    torch: Any,
) -> Any:
    if int(features.shape[0]) != height * width:
        raise InvariantError("LongVU raw raster shape changed during interpolation")
    if height == side and width == side:
        return features
    image = features.reshape(height, width, -1).permute(2, 0, 1).unsqueeze(0)
    resized = torch.nn.functional.interpolate(
        image.float(),
        size=(side, side),
        mode="bilinear",
        align_corners=False,
    ).to(features.dtype)
    return resized.squeeze(0).permute(1, 2, 0).reshape(side * side, -1)


def _frame_offsets(
    supplied: Any,
    frame_counts: tuple[int, ...],
    *,
    total: int,
    name: str,
) -> tuple[int, ...]:
    expected = [0]
    for count in frame_counts:
        expected.append(expected[-1] + count)
    expected_tuple = tuple(expected)
    if supplied is None:
        if len(frame_counts) != 1:
            raise BackendIntegrationError(
                f"LongVU multi-video payload requires explicit {name}"
            )
        supplied_tuple = expected_tuple
    else:
        supplied_tuple = tuple(int(value) for value in supplied)
    if supplied_tuple != expected_tuple or expected_tuple[-1] != total:
        raise BackendIntegrationError(f"LongVU {name} does not match frame layout")
    return supplied_tuple


def _validate_feature_tensor(value: Any, name: str, *, torch: Any) -> None:
    if not isinstance(value, torch.Tensor) or value.ndim != 2:
        raise TypeError(f"LongVU {name} must be a rank-two PyTorch tensor")
    if int(value.shape[0]) < 1 or int(value.shape[1]) < 1:
        raise ValueError(f"LongVU {name} must be non-empty")
    if not bool(torch.isfinite(value).all().item()):
        raise BackendIntegrationError(f"LongVU {name} must be finite")


def _validate_rank_three(value: Any, name: str, *, torch: Any) -> None:
    if not isinstance(value, torch.Tensor) or value.ndim != 3:
        raise TypeError(f"LongVU {name} must be a rank-three PyTorch tensor")
    if any(int(size) < 1 for size in value.shape):
        raise ValueError(f"LongVU {name} must be non-empty")


def _torch() -> Any:
    try:
        import torch
    except (
        ImportError
    ) as exc:  # pragma: no cover - local control env is dependency-light
        raise BackendIntegrationError(
            "LongVU video runtime requires the existing PyTorch environment"
        ) from exc
    return torch


__all__ = [
    "LongVUKernel",
    "LongVUVideoKernel",
    "longvu_structural_groups",
    "longvu_video",
    "pool_longvu_source_groups",
    "select_longvu_frames",
]
