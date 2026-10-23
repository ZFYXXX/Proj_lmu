"""Canonical per-temporal-slice VisionZip video reduction.

The pinned video recipe keeps 126 attention-dominant tokens and 21 uniformly
sampled contextual targets in every temporal slice.  Remaining tokens are
assigned to contextual targets by cosine similarity in the penultimate vision
encoder hidden state.  The released merge value is ``target + mean(assigned)``
rather than the mean of the complete group.

This module is intentionally independent from ``visionzip.image``.  The image
variant owns a different 27:5 scaled-budget contract and existing evidence;
the video variant owns fixed per-slice budgets, aligned penultimate encoder
signals, spatiotemporal source groups, and its own registry bindings.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from mllm_comp.core.errors import BackendIntegrationError, ConfigError, InvariantError

from .config import VIDEO_REFERENCE_REVISION, resolve_canonical_video_config
from .contracts import VideoCompressionResult, VideoTokenPayload
from . import _rules as rule_contract


_ALGORITHM = "visionzip"
_VARIANT = "visionzip.video"


class VisionZipVideoKernel:
    """Registry kernel for the single frozen ``visionzip.video`` profile."""

    name = _ALGORITHM
    version = f"videollm-comp-{VIDEO_REFERENCE_REVISION[:8]}-visionzip-v1"

    def validate_config(self, plugin_config: Mapping[str, Any]) -> None:
        if not isinstance(plugin_config, Mapping):
            raise ConfigError("video plugin config must be a mapping")
        contract = plugin_config.get("modality_contract", {})
        if contract:
            if not isinstance(contract, Mapping):
                raise ConfigError("modality_contract must be a mapping")
            if contract.get("algorithm") != self.name:
                raise ConfigError("video plugin algorithm must be 'visionzip'")
            if contract.get("variant") != _VARIANT:
                raise ConfigError("video plugin variant must be 'visionzip.video'")
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
        return visionzip_video(payload, config=config, rules=rules)

    def replay(self, features: Any, result: VideoCompressionResult, *,
               config: Mapping[str, Any], rules: Mapping[str, Any]) -> Any:
        """Replay saved dominant/contextual groups on another feature branch."""
        return merge_visionzip_source_groups(features, result, rules=rules)


VisionZipKernel = VisionZipVideoKernel


def visionzip_video(
    payload: VideoTokenPayload,
    *,
    config: Mapping[str, Any] | None = None,
    rules: Mapping[str, Any] | None = None,
) -> VideoCompressionResult:
    """Run fixed 126+21 VisionZip independently in every temporal slice.

    Provider signals are mappings keyed by the frozen layer index.  Both
    tensors must already align one-for-one with the projected payload rows:
    ``vision_encoder_attention[-2]`` is a scalar mean-received score per token
    and ``vision_encoder_hidden_states[-2]`` is its similarity feature.
    Backbone providers own spatial-merger/pooling alignment before this kernel.
    """

    torch = _torch()
    if not isinstance(payload, VideoTokenPayload):
        raise TypeError("payload must be a VideoTokenPayload")
    if not isinstance(payload.features, torch.Tensor):
        raise TypeError("VisionZip payload features must be a torch.Tensor")
    if payload.features.ndim != 2 or not payload.features.is_floating_point():
        raise ValueError(
            "VisionZip payload features must have floating shape [tokens, hidden]"
        )
    if not bool(torch.isfinite(payload.features).all().item()):
        raise ValueError("VisionZip payload features must be finite")
    if payload.token_mask is not None:
        mask = torch.as_tensor(payload.token_mask, device=payload.features.device)
        if mask.ndim != 1 or int(mask.numel()) != payload.token_count:
            raise ValueError("VisionZip token_mask must match the projected layout")
        if not bool(mask.bool().all().item()):
            raise BackendIntegrationError(
                "VisionZip requires provider-trimmed videos without padded tokens"
            )

    resolved = resolve_canonical_video_config(_ALGORITHM, config or {})
    attention_layer = int(resolved["attn_layer"])
    hidden_layer = int(resolved["hidden_state_layer"])
    attention = _aligned_layer_signal(
        payload,
        name="vision_encoder_attention",
        layer=attention_layer,
        expected_rank=1,
        torch=torch,
    )
    similarity_features = _aligned_layer_signal(
        payload,
        name="vision_encoder_hidden_states",
        layer=hidden_layer,
        expected_rank=2,
        torch=torch,
    )
    attention = attention.to(payload.features.device)
    similarity_features = similarity_features.to(payload.features.device)

    dominant_num = int(resolved["dominant_num"])
    contextual_num = int(resolved["contextual_num"])
    per_slice_budget = dominant_num + contextual_num
    outputs: list[Any] = []
    groups_by_video: list[tuple[tuple[int, ...], ...]] = []
    representatives_by_video: list[tuple[int, ...]] = []
    merge_targets_by_video: list[tuple[int, ...]] = []
    exact_budgets: list[int] = []
    frame_records_by_video: list[tuple[dict[str, Any], ...]] = []
    anchor_legalizations_by_video: list[int] = []

    for video_index in range(payload.video_count):
        span = payload.video_span(video_index)
        source = payload.features[span]
        scores = attention[span]
        similarity = similarity_features[span]
        video_coordinates = payload.coordinates[span]
        temporal_spans = _temporal_spans(video_coordinates)
        _validate_uniform_raster(video_coordinates, temporal_spans)
        counts = tuple(stop - start for _, start, stop in temporal_spans)
        if len(set(counts)) != 1:
            raise BackendIntegrationError(
                "Canonical VisionZip requires a uniform raster per temporal slice"
            )

        video_outputs: list[Any] = []
        video_groups: list[tuple[int, ...]] = []
        video_targets: list[int] = []
        video_records: list[dict[str, Any]] = []
        video_legalizations = 0
        for temporal, start, stop in temporal_spans:
            source_count = stop - start
            if source_count <= per_slice_budget:
                raise BackendIntegrationError(
                    "Canonical VisionZip requires more than 147 source tokens in "
                    "every temporal slice so 126+21 is a non-identity reduction"
                )
            frame_groups, frame_targets, dominant, contextual = _visionzip_frame_groups(
                scores[start:stop],
                similarity[start:stop],
                dominant_num=dominant_num,
                contextual_num=contextual_num,
                torch=torch,
                rules=rules,
                values=source[start:stop],
            )
            frame_representatives, legalizations = _causal_layout_anchors(
                frame_groups,
                preferred=frame_targets,
            )
            frame_output = _merge_frame_values(
                source[start:stop],
                frame_groups,
                merge_targets=frame_targets,
                torch=torch,
                rules=rules,
            )
            video_outputs.append(frame_output)
            video_groups.extend(
                tuple(index + start for index in group) for group in frame_groups
            )
            video_targets.extend(index + start for index in frame_targets)
            representatives_by_frame = tuple(
                index + start for index in frame_representatives
            )
            video_records.append(
                {
                    "temporal_index": temporal,
                    "source_count": source_count,
                    "result_count": len(frame_groups),
                    "dominant_indices": tuple(index + start for index in dominant),
                    "contextual_target_indices": tuple(
                        index + start for index in contextual
                    ),
                    "representative_indices": representatives_by_frame,
                    "anchor_legalizations": legalizations,
                }
            )
            video_legalizations += legalizations

        merged_video_groups = tuple(video_groups)
        merged_video_targets = tuple(video_targets)
        video_representatives = tuple(
            representative
            for record in video_records
            for representative in record["representative_indices"]
        )
        outputs.append(torch.cat(video_outputs, dim=0))
        groups_by_video.append(merged_video_groups)
        merge_targets_by_video.append(merged_video_targets)
        representatives_by_video.append(video_representatives)
        exact_budgets.append(len(merged_video_groups))
        frame_records_by_video.append(tuple(video_records))
        anchor_legalizations_by_video.append(video_legalizations)

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
            "selection_scope": "independent_per_temporal_slice",
            "attention_statistic": "custom" if rules and ('scorer' in rules) else "penultimate_mean_received",
            "similarity": "custom" if rules and ('grouper' in rules) else "penultimate_hidden_cosine",
            "contextual_target_sampling": "custom" if rules and ('context_selector' in rules) else "released_prefix_stride",
            "merge_value": "custom" if rules and ('aggregator' in rules) else "target_plus_mean_assigned_excluding_target",
            "merge_target_indices_by_video": tuple(merge_targets_by_video),
            "representative_policy": (
                "merge_target_when_causal_else_source_group_first_index"
            ),
            "anchor_legalizations_by_video": tuple(anchor_legalizations_by_video),
            "output_order": "source_group_first_index_within_temporal_slice",
            "per_temporal_slice_budget": per_slice_budget,
            "frame_records_by_video": tuple(frame_records_by_video),
            "image_variant_shared": False,
        },
    )


def merge_visionzip_source_groups(
    features: Any,
    result: VideoCompressionResult,
    *,
    video_index: int = 0,
    rules: Mapping[str, Any] | None = None,
) -> Any:
    """Replay VisionZip merge targets on a synchronized DeepStack branch."""

    torch = _torch()
    if not isinstance(features, torch.Tensor):
        raise TypeError("synchronized VisionZip features must be a torch.Tensor")
    if features.ndim != 2 or not features.is_floating_point():
        raise ValueError(
            "synchronized VisionZip features must be floating [tokens, hidden]"
        )
    if not bool(torch.isfinite(features).all().item()):
        raise ValueError("synchronized VisionZip features must be finite")
    if not isinstance(result, VideoCompressionResult):
        raise TypeError("result must be a VideoCompressionResult")
    source_span = result.source_video_span(video_index)
    source_count = source_span.stop - source_span.start
    if int(features.shape[0]) != source_count:
        raise InvariantError(
            "synchronized VisionZip branch does not match the source video layout"
        )
    targets_by_video = result.metadata.get("merge_target_indices_by_video")
    if not isinstance(targets_by_video, tuple) or len(targets_by_video) != len(
        result.media_ids
    ):
        raise InvariantError("VisionZip result is missing exact merge targets")
    targets = tuple(int(value) for value in targets_by_video[video_index])
    groups = result.source_groups[video_index]
    if len(targets) != len(groups):
        raise InvariantError("VisionZip merge targets do not match source groups")
    if rules and "aggregator" in rules:
        # The first value pass exposes one frame, including frame-local group
        # and target indices. Replay must retain that same callable context.
        rows = []
        source_start = result_start = 0
        for record in result.metadata["frame_records_by_video"][video_index]:
            source_stop = source_start + int(record["source_count"])
            result_stop = result_start + len(record["representative_indices"])
            if result_stop > result_start:
                rows.append(_merge_frame_values(
                    features[source_start:source_stop],
                    tuple(tuple(index - source_start for index in group)
                          for group in groups[result_start:result_stop]),
                    merge_targets=tuple(index - source_start for index in targets[result_start:result_stop]),
                    torch=torch, rules=rules))
            source_start, result_start = source_stop, result_stop
        return torch.cat(rows, dim=0)
    return _merge_frame_values(
        features,
        groups,
        merge_targets=targets,
        torch=torch,
        rules=rules,
    )


def _visionzip_frame_groups(
    attention: Any,
    similarity_features: Any,
    *,
    dominant_num: int,
    contextual_num: int,
    torch: Any,
    rules: Mapping[str, Any] | None = None,
    values: Any | None = None,
) -> tuple[
    tuple[tuple[int, ...], ...],
    tuple[int, ...],
    tuple[int, ...],
    tuple[int, ...],
]:
    count = int(attention.shape[0])
    if int(similarity_features.shape[0]) != count:
        raise InvariantError("VisionZip signal token counts changed")
    rules = rules or {}
    request = {"values": similarity_features if values is None else values,
               "attention_scores": attention, "similarity_features": similarity_features,
               "dominant_count": dominant_num, "contextual_count": contextual_num}
    if "scorer" in rules:
        attention = rule_contract.scores(rules["scorer"], request, count=count,
            reference=attention, name="scorer")
    order = torch.argsort(
        attention.detach().float(),
        descending=True,
        stable=True,
    )
    dominant_ranked = tuple(int(value) for value in order[:dominant_num].tolist())
    if "dominant_selector" in rules:
        dominant_ranked = rule_contract.select(rules["dominant_selector"],
            {**request, "scores": attention, "keep_count": dominant_num},
            count=dominant_num, candidates=range(count), name="dominant_selector")
    dominant_set = set(dominant_ranked)
    non_dominant = tuple(index for index in range(count) if index not in dominant_set)
    step = max(1, len(non_dominant) // contextual_num)
    contextual_targets = tuple(
        non_dominant[position * step] for position in range(contextual_num)
    )
    if "context_selector" in rules:
        contextual_targets = rule_contract.select(rules["context_selector"],
            {**request, "scores": attention, "candidates": non_dominant,
             "keep_count": contextual_num}, count=contextual_num,
            candidates=non_dominant, name="context_selector")
    contextual_set = set(contextual_targets)
    unassigned = tuple(index for index in non_dominant if index not in contextual_set)

    normalized = torch.nn.functional.normalize(
        similarity_features.detach().float(),
        p=2.0,
        dim=-1,
    )
    target_index = torch.tensor(
        contextual_targets,
        dtype=torch.long,
        device=normalized.device,
    )
    source_index = torch.tensor(
        unassigned,
        dtype=torch.long,
        device=normalized.device,
    )
    assignments = (
        normalized.index_select(0, source_index)
        @ normalized.index_select(0, target_index).transpose(0, 1)
    ).argmax(dim=-1)
    contextual_groups: list[list[int]] = [[target] for target in contextual_targets]
    for source, target_position in zip(unassigned, assignments.tolist()):
        contextual_groups[int(target_position)].append(source)
    if "grouper" in rules:
        contextual_groups = rule_contract.groups(rules["grouper"],
            {**request, "candidates": non_dominant, "targets": contextual_targets},
            count=contextual_num, candidates=non_dominant, name="grouper")
        if any(target not in group for target, group in zip(contextual_targets, contextual_groups)):
            raise BackendIntegrationError("VisionZip grouper must keep each target in its corresponding group")

    records: list[tuple[tuple[int, ...], int]] = [
        ((index,), index) for index in dominant_ranked
    ]
    records.extend(
        (tuple(sorted(group)), target)
        for group, target in zip(contextual_groups, contextual_targets)
    )
    records.sort(key=lambda record: record[0][0])
    groups = tuple(record[0] for record in records)
    targets = tuple(record[1] for record in records)
    if len(groups) != dominant_num + contextual_num:
        raise InvariantError("VisionZip did not emit the exact canonical budget")
    return groups, targets, dominant_ranked, contextual_targets


def _merge_frame_values(
    features: Any,
    groups: tuple[tuple[int, ...], ...],
    *,
    merge_targets: tuple[int, ...],
    torch: Any,
    rules: Mapping[str, Any] | None = None,
) -> Any:
    if len(groups) != len(merge_targets):
        raise InvariantError("VisionZip group/merge-target counts differ")
    values = []
    for group, target in zip(groups, merge_targets):
        if target not in group:
            raise InvariantError("VisionZip merge target left its source group")
        assigned = tuple(index for index in group if index != target)
        value = features[target]
        if assigned:
            index = torch.tensor(
                assigned,
                dtype=torch.long,
                device=features.device,
            )
            value = value + features.index_select(0, index).mean(dim=0)
        if rules and "aggregator" in rules:
            value = rule_contract.value(rules["aggregator"],
                {"values": features, "group": group, "target": target,
                 "assigned_indices": assigned}, reference=features,
                shape=(int(features.shape[-1]),), name="aggregator")
        values.append(value)
    return torch.stack(values, dim=0)


def _causal_layout_anchors(
    groups: tuple[tuple[int, ...], ...],
    *,
    preferred: tuple[int, ...],
) -> tuple[tuple[int, ...], int]:
    if len(groups) != len(preferred):
        raise InvariantError("VisionZip group/anchor counts differ")
    representatives = []
    legalizations = 0
    for index, (group, target) in enumerate(zip(groups, preferred)):
        if target not in group:
            raise InvariantError("VisionZip preferred anchor left its source group")
        upper = groups[index + 1][0] if index + 1 < len(groups) else None
        representative = target if upper is None or target < upper else group[0]
        representatives.append(representative)
        legalizations += int(representative != target)
    result = tuple(representatives)
    if result != tuple(sorted(result)) or len(set(result)) != len(result):
        raise InvariantError("VisionZip anchors do not preserve causal source order")
    return result, legalizations


def _temporal_spans(
    coordinates: tuple[Any, ...],
) -> tuple[tuple[int, int, int], ...]:
    if not coordinates:
        raise BackendIntegrationError("VisionZip video has no source coordinates")
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
                "VisionZip temporal slices must be contiguous and source ordered"
            )
        spans.append((temporal, start, index))
        start = index
        temporal = current
        seen.add(current)
    spans.append((temporal, start, len(coordinates)))
    return tuple(spans)


def _validate_uniform_raster(
    coordinates: tuple[Any, ...],
    temporal_spans: tuple[tuple[int, int, int], ...],
) -> None:
    expected: tuple[tuple[int, int], ...] | None = None
    for _temporal, start, stop in temporal_spans:
        raster = tuple(
            (int(coordinate.h), int(coordinate.w))
            for coordinate in coordinates[start:stop]
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
                "Canonical VisionZip requires each temporal slice to be a "
                "dense row-major projected raster"
            )
        if expected is None:
            expected = raster
        elif raster != expected:
            raise BackendIntegrationError(
                "Canonical VisionZip requires a uniform raster across temporal slices"
            )


def _aligned_layer_signal(
    payload: VideoTokenPayload,
    *,
    name: str,
    layer: int,
    expected_rank: int,
    torch: Any,
) -> Any:
    layers = payload.signals.get(name)
    if not isinstance(layers, Mapping) or layer not in layers:
        raise BackendIntegrationError(f"VisionZip requires aligned {name}[{layer}]")
    value = layers[layer]
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"VisionZip {name}[{layer}] must be a torch.Tensor")
    if value.ndim != expected_rank or int(value.shape[0]) != payload.token_count:
        raise BackendIntegrationError(
            f"VisionZip {name}[{layer}] must align one-for-one with payload tokens"
        )
    if not value.is_floating_point():
        raise TypeError(f"VisionZip {name}[{layer}] must use a floating dtype")
    if not bool(torch.isfinite(value).all().item()):
        raise ValueError(f"VisionZip {name}[{layer}] must be finite")
    return value


def _torch() -> Any:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - dependency-light control plane
        raise BackendIntegrationError(
            "VisionZip video runtime requires the existing PyTorch environment"
        ) from exc
    return torch


__all__ = [
    "VisionZipKernel",
    "VisionZipVideoKernel",
    "merge_visionzip_source_groups",
    "visionzip_video",
]
