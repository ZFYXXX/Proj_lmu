"""Canonical FastVID DySeg/STPrune/DTM video compression.

The pinned recipe first segments adjacent frames from global visual features,
keeps attention-salient projected tokens, and then emits density-selected DTM
context tokens.  Multi-frame DTM intentionally reuses every filtered token in
the segment for each target frame.  Consequently its numerical lineage can
overlap and cannot be represented by the single-valued placeholder mapping in
``LayoutDelta``.  Results therefore use the released sampled centers as their
layout source groups and record the complete, potentially overlapping value
lineage separately for exact DeepStack replay and evidence.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

from mllm_comp.core.errors import BackendIntegrationError, ConfigError, InvariantError

from .config import VIDEO_REFERENCE_REVISION, resolve_canonical_video_config
from .contracts import VideoCompressionResult, VideoTokenPayload
from . import _rules as rule_contract


_ALGORITHM = "fastvid"
_VARIANT = "fastvid.video"


class FastVIDVideoKernel:
    """Registry kernel for the single frozen FastVID video profile."""

    name = _ALGORITHM
    version = f"videollm-comp-{VIDEO_REFERENCE_REVISION[:8]}-fastvid-v1"

    def validate_config(self, plugin_config: Mapping[str, Any]) -> None:
        if not isinstance(plugin_config, Mapping):
            raise ConfigError("video plugin config must be a mapping")
        contract = plugin_config.get("modality_contract", {})
        if contract:
            if not isinstance(contract, Mapping):
                raise ConfigError("modality_contract must be a mapping")
            if contract.get("algorithm") != self.name:
                raise ConfigError("video plugin algorithm must be 'fastvid'")
            if contract.get("variant") != _VARIANT:
                raise ConfigError("video plugin variant must be 'fastvid.video'")
        resolve_canonical_video_config(
            self.name,
            plugin_config.get("config", {}),
        )

    def __call__(
        self,
        payload: VideoTokenPayload,
        *,
        config: Mapping[str, Any] | None = None,
        rules: Mapping[str, Any] | None = None,
    ) -> VideoCompressionResult:
        return fastvid_video(payload, config=config, rules=rules)

    def replay(self, features: Any, result: VideoCompressionResult, *,
               config: Mapping[str, Any], rules: Mapping[str, Any]) -> Any:
        """Replay saved, possibly overlapping value lineage on another branch."""
        return replay_fastvid_value_lineage(features, result, rules=rules)


FastVIDKernel = FastVIDVideoKernel


def fastvid_video(
    payload: VideoTokenPayload,
    *,
    config: Mapping[str, Any] | None = None,
    rules: Mapping[str, Any] | None = None,
) -> VideoCompressionResult:
    """Run the frozen DySeg, STPrune, and DTM stages per video."""

    torch = _torch()
    if not isinstance(payload, VideoTokenPayload):
        raise TypeError("payload must be a VideoTokenPayload")
    _validate_feature_tensor(payload.features, "projected features", torch=torch)
    if payload.raw_features is None:
        raise BackendIntegrationError(
            "FastVID requires provider-owned pre-projector/native raw features"
        )
    _validate_feature_tensor(payload.raw_features, "raw features", torch=torch)
    if payload.token_mask is not None:
        mask = torch.as_tensor(payload.token_mask, device=payload.features.device)
        if mask.ndim != 1 or int(mask.numel()) != payload.token_count:
            raise ValueError("FastVID token_mask must match the projected layout")
        if not bool(mask.bool().all().item()):
            raise BackendIntegrationError(
                "FastVID requires provider-trimmed videos without padded tokens"
            )

    resolved = resolve_canonical_video_config(_ALGORITHM, config or {})
    frame_globals = _aligned_signal(
        payload,
        "frame_global_features",
        rank=2,
        torch=torch,
    )
    saliency = _aligned_signal(
        payload,
        "token_saliency_attention",
        rank=1,
        first_dimension=payload.token_count,
        torch=torch,
    )
    frame_globals = frame_globals.to(payload.features.device)
    saliency = saliency.to(payload.features.device)

    projected_frames_by_video = []
    raw_frames_by_video = []
    total_frames = 0
    for video_index in range(payload.video_count):
        projected = _dense_frame_spans(
            payload.coordinates[payload.video_span(video_index)],
            name="projected",
        )
        raw = _dense_frame_spans(
            payload.raw_coordinates[payload.raw_video_span(video_index)],
            name="raw",
        )
        if len(projected) != len(raw):
            raise BackendIntegrationError(
                "FastVID raw/projected temporal boundaries do not match"
            )
        projected_frames_by_video.append(projected)
        raw_frames_by_video.append(raw)
        total_frames += len(projected)
    if int(frame_globals.shape[0]) != total_frames:
        raise BackendIntegrationError(
            "FastVID frame_global_features must contain one row per temporal slice"
        )

    outputs: list[Any] = []
    groups_by_video: list[tuple[tuple[int, ...], ...]] = []
    representatives_by_video: list[tuple[int, ...]] = []
    exact_budgets: list[int] = []
    value_records_by_video: list[tuple[dict[str, Any], ...]] = []
    segment_sizes_by_video: list[tuple[int, ...]] = []
    context_allocations_by_video: list[tuple[int, ...]] = []
    frame_output_counts_by_video: list[tuple[int, ...]] = []
    global_cursor = 0

    for video_index in range(payload.video_count):
        video_span = payload.video_span(video_index)
        features = payload.features[video_span]
        attention = saliency[video_span]
        frame_spans = projected_frames_by_video[video_index]
        frame_count = len(frame_spans)
        globals_for_video = frame_globals[global_cursor : global_cursor + frame_count]
        global_cursor += frame_count

        counts = tuple(stop - start for _temporal, start, stop in frame_spans)
        if len(set(counts)) != 1:
            raise BackendIntegrationError(
                "Canonical FastVID requires a uniform projected raster per frame"
            )
        tokens_per_frame = counts[0]
        retain_num = int(tokens_per_frame * float(resolved["retention_ratio"]))
        context_num = int(retain_num * float(resolved["stprune_d"]))
        salient_num = retain_num - context_num
        filtered_num = tokens_per_frame - salient_num
        if (
            retain_num < 2
            or salient_num < 1
            or context_num < 1
            or filtered_num < 4
            or retain_num >= tokens_per_frame
        ):
            raise BackendIntegrationError(
                "Canonical FastVID requires a non-identity raster with at least "
                "one salient token, one context token, and four filtered tokens"
            )

        segment_sizes = _dynamic_segmentation(
            globals_for_video,
            dyseg_c=int(resolved["dyseg_c"]),
            dyseg_tau=float(resolved["dyseg_tau"]),
            torch=torch,
        )
        if rules and "frame_grouper" in rules:
            grouped_frames = rule_contract.groups(rules["frame_grouper"],
                {"values": globals_for_video, "default_segment_sizes": segment_sizes,
                 "group_count": len(segment_sizes)}, count=None,
                candidates=range(frame_count), name="frame_grouper")
            if tuple(i for group in grouped_frames for i in group) != tuple(range(frame_count)):
                raise BackendIntegrationError("FastVID frame_grouper must return contiguous ordered segments")
            segment_sizes = tuple(len(group) for group in grouped_frames)
        allocations = _context_allocations(
            segment_sizes,
            context_num=context_num,
            dtm_p=int(resolved["dtm_p"]),
        )
        if rules and "context_allocator" in rules:
            with torch.no_grad():
                custom_allocations = tuple(rules["context_allocator"]({"segment_sizes": segment_sizes,
                    "context_count": context_num, "default_allocations": allocations,
                    "total_budget": sum(allocations), "capacity_per_frame": filtered_num}))
            if (len(custom_allocations) != frame_count or sum(custom_allocations) != sum(allocations)
                    or any(isinstance(n, bool) or not isinstance(n, int) or not 0 <= n <= filtered_num
                           for n in custom_allocations)):
                raise BackendIntegrationError("FastVID context_allocator must preserve budget and frame capacities")
            allocations = custom_allocations

        salient_by_frame: list[tuple[int, ...]] = []
        filtered_by_frame: list[tuple[int, ...]] = []
        density_by_frame: list[Any] = []
        for _temporal, start, stop in frame_spans:
            frame_attention = attention[start:stop]
            if rules and "saliency_scorer" in rules:
                frame_attention = rule_contract.scores(rules["saliency_scorer"],
                    {"values": features[start:stop], "default_scores": frame_attention,
                     "frame_index": _temporal}, count=tokens_per_frame,
                    reference=features, name="saliency_scorer")
            salient_local = _stable_rank(
                frame_attention,
                count=salient_num,
                descending=True,
                torch=torch,
            )
            if rules and "saliency_selector" in rules:
                salient_local = rule_contract.select(rules["saliency_selector"],
                    {"values": features[start:stop], "scores": frame_attention, "keep_count": salient_num},
                    count=salient_num, candidates=range(tokens_per_frame), name="saliency_selector")
            salient_set = set(salient_local)
            filtered_local = tuple(
                index for index in range(tokens_per_frame) if index not in salient_set
            )
            filtered_index = torch.tensor(
                filtered_local,
                dtype=torch.long,
                device=features.device,
            )
            filtered_features = features[start:stop].index_select(0, filtered_index)
            density = _density_score(filtered_features, torch=torch)
            if rules and "center_scorer" in rules:
                density = rule_contract.scores(rules["center_scorer"],
                    {"values": filtered_features, "default_scores": density},
                    count=filtered_num, reference=features, name="center_scorer")
            salient_by_frame.append(tuple(start + index for index in salient_local))
            filtered_by_frame.append(tuple(start + index for index in filtered_local))
            density_by_frame.append(density)

        records: list[dict[str, Any]] = []
        values: list[Any] = []
        for frame_index, anchors in enumerate(salient_by_frame):
            for anchor in anchors:
                records.append(
                    {
                        "kind": "salient",
                        "target_frame": frame_index,
                        "center_index": anchor,
                        "assigned_indices": (),
                        "center_weight": 1.0,
                    }
                )
                values.append(features[anchor])

        segment_offset = 0
        for segment_index, segment_length in enumerate(segment_sizes):
            segment_frames = tuple(
                range(segment_offset, segment_offset + segment_length)
            )
            target_frames = tuple(
                frame for frame in segment_frames if allocations[frame] > 0
            )
            if segment_length == 1 and not (rules and "context_allocator" in rules):
                if allocations[segment_frames[0]] != context_num:
                    raise InvariantError(
                        "FastVID single-frame context allocation changed"
                    )
            elif not (rules and "context_allocator" in rules) and any(allocations[frame] <= context_num for frame in target_frames):
                raise InvariantError("FastVID multi-frame context allocation changed")

            all_filtered_indices = tuple(
                index for frame in segment_frames for index in filtered_by_frame[frame]
            )
            all_index = torch.tensor(
                all_filtered_indices,
                dtype=torch.long,
                device=features.device,
            )
            all_filtered = features.index_select(0, all_index)
            all_normalised = _normalise_nonzero(
                all_filtered.detach().float(),
                name="filtered projected tokens",
                torch=torch,
            )

            for target_frame in target_frames:
                target_count = allocations[target_frame]
                center_positions = _stable_rank(
                    density_by_frame[target_frame],
                    count=target_count,
                    descending=True,
                    torch=torch,
                )
                if rules and "center_selector" in rules:
                    center_positions = rule_contract.select(rules["center_selector"],
                        {"values": features[list(filtered_by_frame[target_frame])],
                         "scores": density_by_frame[target_frame], "keep_count": target_count},
                        count=target_count, candidates=range(filtered_num), name="center_selector")
                center_indices = tuple(
                    filtered_by_frame[target_frame][position]
                    for position in center_positions
                )
                center_index = torch.tensor(
                    center_indices,
                    dtype=torch.long,
                    device=features.device,
                )
                centers = features.index_select(0, center_index)
                normalised_centers = _normalise_nonzero(
                    centers.detach().float(),
                    name="DTM centers",
                    torch=torch,
                )
                assignments = (
                    all_normalised @ normalised_centers.transpose(0, 1)
                ).argmax(dim=-1)
                if rules and "grouper" in rules:
                    custom_groups = rule_contract.groups(rules["grouper"],
                        {"values": features, "candidates": all_filtered_indices,
                         "targets": center_indices}, count=len(center_indices),
                        candidates=all_filtered_indices, name="grouper", allow_empty=True)
                    assignments = torch.empty(len(all_filtered_indices), dtype=torch.long, device=features.device)
                    local_lookup = {source: index for index, source in enumerate(all_filtered_indices)}
                    for group_index, group in enumerate(custom_groups):
                        assignments[[local_lookup[source] for source in group]] = group_index
                for center_position, center in enumerate(center_indices):
                    member_mask = assignments == center_position
                    member_indices_tensor = all_index[member_mask]
                    assigned = tuple(
                        int(value) for value in member_indices_tensor.tolist()
                    )
                    count = len(assigned)
                    center_weight = max(
                        1.0 / float(count + 1),
                        float(resolved["dtm_beta"]),
                    )
                    value = features[center]
                    if count:
                        aggregated = features.index_select(
                            0, member_indices_tensor
                        ).mean(dim=0)
                        value = (
                            center_weight * value + (1.0 - center_weight) * aggregated
                        )
                    if rules and "aggregator" in rules:
                        value = _aggregate_value(features, center, assigned, center_weight, rules["aggregator"])
                    records.append(
                        {
                            "kind": (
                                "single_frame_dtm"
                                if segment_length == 1
                                else "multi_frame_dtm"
                            ),
                            "segment_index": segment_index,
                            "target_frame": target_frame,
                            "center_index": center,
                            "assigned_indices": assigned,
                            "center_weight": center_weight,
                        }
                    )
                    values.append(value)
            segment_offset += segment_length

        order = tuple(
            sorted(
                range(len(records)), key=lambda index: records[index]["center_index"]
            )
        )
        ordered_records = tuple(records[index] for index in order)
        ordered_values = torch.stack(tuple(values[index] for index in order), dim=0)
        anchors = tuple(record["center_index"] for record in ordered_records)
        if anchors != tuple(sorted(anchors)) or len(set(anchors)) != len(anchors):
            raise InvariantError(
                "FastVID sampled centers do not preserve unique source order"
            )
        if len(anchors) >= int(features.shape[0]):
            raise BackendIntegrationError(
                "Canonical FastVID did not produce a non-identity reduction"
            )

        outputs.append(ordered_values)
        groups_by_video.append(tuple((anchor,) for anchor in anchors))
        representatives_by_video.append(anchors)
        exact_budgets.append(len(anchors))
        value_records_by_video.append(ordered_records)
        segment_sizes_by_video.append(segment_sizes)
        context_allocations_by_video.append(allocations)
        frame_output_counts_by_video.append(
            tuple(
                sum(record["target_frame"] == frame for record in ordered_records)
                for frame in range(frame_count)
            )
        )

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
            "stages": ("dyseg", "stprune", "dtm"),
            "dyseg_boundary_rule": "custom" if rules and ('frame_grouper' in rules) else "stable_lowest_adjacent_cosine_union_below_tau",
            "stprune_rule": "custom" if rules and ('saliency_scorer' in rules or 'saliency_selector' in rules) else "stable_per_frame_attention_topk",
            "density_tie_policy": "custom" if rules and ('center_scorer' in rules or 'center_selector' in rules) else "stable_source_order_without_random_noise",
            "dtm_merge_rule": "custom" if rules and ('aggregator' in rules) else "center_weighted_with_assigned_mean",
            "layout_source_groups": "sampled_center_singletons",
            "layout_anchor_policy": "released_sampled_center",
            "value_lineage_policy": "custom" if rules and ('aggregator' in rules) else "exact_potentially_overlapping_assignments",
            "value_records_by_video": tuple(value_records_by_video),
            "segment_sizes_by_video": tuple(segment_sizes_by_video),
            "context_allocations_by_video": tuple(context_allocations_by_video),
            "frame_output_counts_by_video": tuple(frame_output_counts_by_video),
            "training_free_compressor": True,
        },
    )


def replay_fastvid_value_lineage(
    features: Any,
    result: VideoCompressionResult,
    *,
    video_index: int = 0,
    rules: Mapping[str, Any] | None = None,
) -> Any:
    """Replay the exact FastVID value lineage on one DeepStack branch."""

    torch = _torch()
    _validate_feature_tensor(features, "synchronized features", torch=torch)
    if not isinstance(result, VideoCompressionResult):
        raise TypeError("result must be a VideoCompressionResult")
    source_span = result.source_video_span(video_index)
    if int(features.shape[0]) != source_span.stop - source_span.start:
        raise InvariantError(
            "synchronized FastVID branch does not match the source video layout"
        )
    records_by_video = result.metadata.get("value_records_by_video")
    if not isinstance(records_by_video, tuple) or len(records_by_video) != len(
        result.media_ids
    ):
        raise InvariantError("FastVID result is missing exact value lineage")

    values = []
    for record in records_by_video[video_index]:
        if not isinstance(record, Mapping):
            raise InvariantError("FastVID value-lineage record changed")
        center = int(record["center_index"])
        assigned = tuple(int(value) for value in record["assigned_indices"])
        center_weight = float(record["center_weight"])
        value = features[center]
        if assigned:
            index = torch.tensor(
                assigned,
                dtype=torch.long,
                device=features.device,
            )
            value = center_weight * value + (
                1.0 - center_weight
            ) * features.index_select(0, index).mean(dim=0)
        if record["kind"] != "salient" and rules and "aggregator" in rules:
            value = _aggregate_value(features, center, assigned, center_weight, rules["aggregator"])
        values.append(value)
    return torch.stack(values, dim=0)


def _aggregate_value(features: Any, center: int, assigned: tuple[int, ...], weight: float, rule: Any) -> Any:
    return rule_contract.value(rule, {"values": features, "target": center,
        "assigned_indices": assigned, "center_weight": weight}, reference=features,
        shape=(int(features.shape[-1]),), name="aggregator")


def _dynamic_segmentation(
    frame_global_features: Any,
    *,
    dyseg_c: int,
    dyseg_tau: float,
    torch: Any,
) -> tuple[int, ...]:
    frame_count = int(frame_global_features.shape[0])
    if frame_count < 1:
        raise BackendIntegrationError("FastVID requires at least one frame")
    if frame_count == 1:
        return (1,)
    normalised = _normalise_nonzero(
        frame_global_features.detach().float(),
        name="frame global features",
        torch=torch,
    )
    similarities = (normalised[:-1] * normalised[1:]).sum(dim=1)
    count = min(dyseg_c - 1, int(similarities.shape[0]))
    lowest = set(
        _stable_rank(
            similarities,
            count=count,
            descending=False,
            torch=torch,
        )
    )
    below = set(
        int(value)
        for value in (similarities < dyseg_tau)
        .nonzero(as_tuple=False)
        .flatten()
        .tolist()
    )
    cuts = tuple(sorted(lowest | below))
    boundaries = (-1, *cuts, frame_count - 1)
    sizes = tuple(right - left for left, right in zip(boundaries, boundaries[1:]))
    if sum(sizes) != frame_count or any(size < 1 for size in sizes):
        raise InvariantError("FastVID DySeg emitted invalid temporal segments")
    return sizes


def _context_allocations(
    segment_sizes: Sequence[int],
    *,
    context_num: int,
    dtm_p: int,
) -> tuple[int, ...]:
    allocations: list[int] = []
    for segment_length in segment_sizes:
        target_count = (int(segment_length) + dtm_p - 1) // dtm_p
        per_target = context_num * int(segment_length) // target_count
        allocations.extend(
            per_target if (int(segment_length) - 1 - local) % dtm_p == 0 else 0
            for local in range(int(segment_length))
        )
    return tuple(allocations)


def _density_score(filtered_tokens: Any, *, torch: Any) -> Any:
    hidden = int(filtered_tokens.shape[-1])
    distance = torch.cdist(
        filtered_tokens.detach().float(),
        filtered_tokens.detach().float(),
    ) / math.sqrt(hidden)
    nearest = torch.sort(distance, dim=-1, stable=True).values[..., :4]
    density = (-(nearest**2).mean(dim=-1)).exp()
    higher = density.unsqueeze(-2) > density.unsqueeze(-1)
    distance_max = distance.flatten().max()
    nearest_higher = torch.where(higher, distance, distance_max).min(dim=-1).values
    return nearest_higher * density


def _stable_rank(
    values: Any,
    *,
    count: int,
    descending: bool,
    torch: Any,
) -> tuple[int, ...]:
    if count < 0 or count > int(values.shape[0]):
        raise BackendIntegrationError("FastVID requested an infeasible stable rank")
    if count == 0:
        return ()
    order = torch.argsort(
        values.detach().float(),
        descending=descending,
        stable=True,
    )
    return tuple(int(value) for value in order[:count].tolist())


def _normalise_nonzero(value: Any, *, name: str, torch: Any) -> Any:
    norms = value.norm(dim=-1, keepdim=True)
    if not bool(torch.isfinite(norms).all().item()) or bool((norms == 0).any().item()):
        raise BackendIntegrationError(f"FastVID {name} must have finite non-zero norms")
    return value / norms


def _aligned_signal(
    payload: VideoTokenPayload,
    name: str,
    *,
    rank: int,
    first_dimension: int | None = None,
    torch: Any,
) -> Any:
    value = payload.signals.get(name)
    if not isinstance(value, torch.Tensor):
        raise BackendIntegrationError(f"FastVID requires tensor signal {name}")
    if value.ndim != rank:
        raise BackendIntegrationError(f"FastVID {name} rank changed")
    if first_dimension is not None and int(value.shape[0]) != first_dimension:
        raise BackendIntegrationError(f"FastVID {name} does not align with payload")
    if not value.is_floating_point():
        raise TypeError(f"FastVID {name} must use a floating dtype")
    if not bool(torch.isfinite(value).all().item()):
        raise ValueError(f"FastVID {name} must be finite")
    return value


def _validate_feature_tensor(value: Any, name: str, *, torch: Any) -> None:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"FastVID {name} must be a torch.Tensor")
    if value.ndim != 2 or not value.is_floating_point():
        raise ValueError(f"FastVID {name} must be floating [tokens, hidden]")
    if int(value.shape[0]) < 1 or int(value.shape[1]) < 1:
        raise ValueError(f"FastVID {name} must be non-empty")
    if not bool(torch.isfinite(value).all().item()):
        raise ValueError(f"FastVID {name} must be finite")


def _dense_frame_spans(
    coordinates: tuple[Any, ...],
    *,
    name: str,
) -> tuple[tuple[int, int, int], ...]:
    if not coordinates:
        raise BackendIntegrationError(f"FastVID {name} video has no coordinates")
    spans: list[tuple[int, int, int]] = []
    start = 0
    temporal = int(coordinates[0].t)
    seen = {temporal}
    for index, coordinate in enumerate(coordinates[1:], start=1):
        current = int(coordinate.t)
        if current == temporal:
            continue
        if current in seen or current < temporal:
            raise BackendIntegrationError(
                f"FastVID {name} temporal slices must be contiguous and ordered"
            )
        spans.append((temporal, start, index))
        start = index
        temporal = current
        seen.add(current)
    spans.append((temporal, start, len(coordinates)))

    expected: tuple[tuple[int, int], ...] | None = None
    for _temporal, frame_start, frame_stop in spans:
        raster = tuple(
            (int(coordinate.h), int(coordinate.w))
            for coordinate in coordinates[frame_start:frame_stop]
        )
        heights = tuple(sorted({height for height, _width in raster}))
        widths = tuple(sorted({width for _height, width in raster}))
        dense = tuple((height, width) for height in heights for width in widths)
        if (
            not heights
            or not widths
            or heights != tuple(range(len(heights)))
            or widths != tuple(range(len(widths)))
            or raster != dense
        ):
            raise BackendIntegrationError(
                f"FastVID {name} frames must be dense row-major rasters"
            )
        if expected is None:
            expected = raster
        elif raster != expected:
            raise BackendIntegrationError(
                f"FastVID {name} raster must be uniform across frames"
            )
    return tuple(spans)


def _torch() -> Any:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - dependency-light control plane
        raise BackendIntegrationError(
            "FastVID video runtime requires the existing PyTorch environment"
        ) from exc
    return torch


__all__ = [
    "FastVIDKernel",
    "FastVIDVideoKernel",
    "fastvid_video",
    "replay_fastvid_value_lineage",
]
