"""Canonical DyToK detection budgets and dynamic VisionZip reduction.

The pinned dynamic recipe uses a frozen detection-model pass to obtain one
importance value per temporal slice.  It redistributes the fixed VisionZip
total budget (126 dominant plus 21 contextual tokens per slice), caps every
slice at 196 or its source cardinality, and applies the released 6:1
dominant/contextual split.  Detection is a capability-provider stage; this
module owns only deterministic budget allocation and token reduction.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

from mllm_comp.core.errors import BackendIntegrationError, ConfigError, InvariantError

from .config import VIDEO_REFERENCE_REVISION, resolve_canonical_video_config
from .contracts import VideoCompressionResult, VideoTokenPayload
from . import _rules as rule_contract
from .visionzip import (
    _aligned_layer_signal,
    _causal_layout_anchors,
    _merge_frame_values,
    _temporal_spans,
    _validate_uniform_raster,
    _visionzip_frame_groups,
)


_ALGORITHM = "dytok"
_VARIANT = "dytok.video"
_IMPORTANCE_SIGNAL = "dytok.frame_importance_by_video"


class DyToKVideoKernel:
    """Registry kernel for the single frozen dynamic DyToK profile."""

    name = _ALGORITHM
    version = f"videollm-comp-{VIDEO_REFERENCE_REVISION[:8]}-dytok-v1"

    def validate_config(self, plugin_config: Mapping[str, Any]) -> None:
        if not isinstance(plugin_config, Mapping):
            raise ConfigError("video plugin config must be a mapping")
        contract = plugin_config.get("modality_contract", {})
        if contract:
            if not isinstance(contract, Mapping):
                raise ConfigError("modality_contract must be a mapping")
            if contract.get("algorithm") != self.name:
                raise ConfigError("video plugin algorithm must be 'dytok'")
            if contract.get("variant") != _VARIANT:
                raise ConfigError("video plugin variant must be 'dytok.video'")
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
        return dytok_video(payload, config=config, rules=rules)


DyToKKernel = DyToKVideoKernel


def allocate_dytok_frame_budgets(
    frame_weights: Any,
    source_counts: Sequence[int],
    *,
    base_budget_per_frame: int = 147,
    upper_limit: int = 196,
) -> tuple[int, ...]:
    """Allocate the exact fixed total with stable rounding and cap replay.

    This follows the pinned implementation's floor/largest-remainder step,
    then its weighted round-robin redistribution after per-frame clamping.
    The preserved total is ``min(147 * frames, sum(frame capacities))``.
    """

    torch = _torch()
    if not isinstance(frame_weights, torch.Tensor):
        frame_weights = torch.as_tensor(frame_weights, dtype=torch.float32)
    if frame_weights.ndim != 1 or not frame_weights.is_floating_point():
        raise TypeError("DyToK frame weights must be a floating rank-one tensor")
    counts = tuple(source_counts)
    if len(counts) != int(frame_weights.numel()) or not counts:
        raise ValueError("DyToK frame weights must match non-empty source counts")
    if any(
        isinstance(value, bool) or not isinstance(value, int) or value < 1
        for value in counts
    ):
        raise ValueError("DyToK source counts must be positive integers")
    if (
        isinstance(base_budget_per_frame, bool)
        or not isinstance(base_budget_per_frame, int)
        or base_budget_per_frame < 1
        or isinstance(upper_limit, bool)
        or not isinstance(upper_limit, int)
        or upper_limit < 1
    ):
        raise ValueError("DyToK budget and upper limit must be positive integers")
    weights = frame_weights.detach().float()
    if not bool(torch.isfinite(weights).all().item()) or bool(
        (weights < 0).any().item()
    ):
        raise ValueError("DyToK frame weights must be finite and non-negative")
    weight_sum = float(weights.sum().item())
    if not math.isfinite(weight_sum) or weight_sum <= 0:
        raise ValueError("DyToK frame weights must have positive total mass")
    weights = weights / weight_sum

    capacities = tuple(min(upper_limit, count) for count in counts)
    target_total = min(base_budget_per_frame * len(counts), sum(capacities))
    if target_total < 1:
        raise InvariantError("DyToK source capacity cannot produce an output")
    allocations = weights * float(target_total)
    integer = torch.floor(allocations).to(dtype=torch.long)
    remainders = allocations - integer.float()
    remaining = target_total - int(integer.sum().item())
    remainder_order = sorted(
        range(len(counts)),
        key=lambda index: (-float(remainders[index].item()), index),
    )
    for index in remainder_order[:remaining]:
        integer[index] += 1

    current = [
        min(int(integer[index].item()), capacities[index])
        for index in range(len(counts))
    ]
    missing = target_total - sum(current)
    importance_order = sorted(
        range(len(counts)),
        key=lambda index: (-float(weights[index].item()), index),
    )
    while missing > 0:
        candidates = [
            index for index in importance_order if current[index] < capacities[index]
        ]
        if not candidates:
            raise InvariantError("DyToK could not redistribute its capped budget")
        for index in candidates:
            if missing == 0:
                break
            current[index] += 1
            missing -= 1
    result = tuple(current)
    if sum(result) != target_total or any(
        value > capacity for value, capacity in zip(result, capacities)
    ):
        raise InvariantError("DyToK budget allocation violated its exact total")
    return result


def dytok_frame_attention_from_qk(
    query_states: Any,
    visual_key_states: Any,
    *,
    frame_token_counts: Sequence[int],
) -> Any:
    """Return reference-style visual-only last-query attention per frame."""

    torch = _torch()
    if not isinstance(query_states, torch.Tensor) or not isinstance(
        visual_key_states, torch.Tensor
    ):
        raise TypeError("DyToK detection Q/K states must be tensors")
    if query_states.ndim != 3 or visual_key_states.ndim != 3:
        raise ValueError("DyToK Q/K states must be [queries/tokens, heads, width]")
    if int(query_states.shape[0]) != 1:
        raise ValueError("Canonical DyToK uses exactly the last prompt query")
    if tuple(query_states.shape[1:]) != tuple(visual_key_states.shape[1:]):
        raise ValueError("DyToK detection Q/K head layouts differ")
    counts = tuple(frame_token_counts)
    if not counts or any(
        isinstance(value, bool) or not isinstance(value, int) or value < 1
        for value in counts
    ):
        raise ValueError("DyToK frame token counts must be positive integers")
    if sum(counts) != int(visual_key_states.shape[0]):
        raise ValueError("DyToK frame token counts do not cover visual keys")
    width = int(query_states.shape[-1])
    scores = torch.einsum(
        "qhd,khd->hqk",
        query_states.float(),
        visual_key_states.float(),
    ) / math.sqrt(width)
    token_attention = torch.softmax(scores, dim=-1).mean(dim=(0, 1))
    return torch.stack(
        tuple(chunk.mean() for chunk in token_attention.split(counts)),
        dim=0,
    )


def aggregate_dytok_frame_importance(layer_frame_weights: Sequence[Any]) -> Any:
    """Average target-layer frame signals and normalize their total to one."""

    torch = _torch()
    values = tuple(layer_frame_weights)
    if not values or any(not isinstance(value, torch.Tensor) for value in values):
        raise ValueError("DyToK requires tensor frame weights from target layers")
    shape = tuple(values[0].shape)
    if (
        len(shape) != 1
        or not shape
        or any(tuple(value.shape) != shape for value in values)
    ):
        raise ValueError("DyToK target-layer frame-weight shapes differ")
    stacked = torch.stack(tuple(value.detach().float().to(values[0].device) for value in values), dim=0)
    if not bool(torch.isfinite(stacked).all().item()) or bool(
        (stacked < 0).any().item()
    ):
        raise ValueError(
            "DyToK target-layer frame weights must be finite and non-negative"
        )
    result = stacked.mean(dim=0)
    total = result.sum()
    if not bool(torch.isfinite(total).item()) or float(total.item()) <= 0:
        raise ValueError("DyToK aggregated frame importance has no positive mass")
    return result / total


def dytok_video(
    payload: VideoTokenPayload,
    *,
    config: Mapping[str, Any] | None = None,
    rules: Mapping[str, Any] | None = None,
) -> VideoCompressionResult:
    """Apply detection-derived per-frame budgets to canonical VisionZip."""

    torch = _torch()
    if not isinstance(payload, VideoTokenPayload):
        raise TypeError("payload must be a VideoTokenPayload")
    if not isinstance(payload.features, torch.Tensor):
        raise TypeError("DyToK payload features must be a torch.Tensor")
    if payload.features.ndim != 2 or not payload.features.is_floating_point():
        raise ValueError("DyToK payload features must be floating [tokens, hidden]")
    if not bool(torch.isfinite(payload.features).all().item()):
        raise ValueError("DyToK payload features must be finite")
    if payload.token_mask is not None:
        mask = torch.as_tensor(payload.token_mask, device=payload.features.device)
        if mask.ndim != 1 or int(mask.numel()) != payload.token_count:
            raise ValueError("DyToK token_mask must match the projected layout")
        if not bool(mask.bool().all().item()):
            raise BackendIntegrationError(
                "DyToK requires provider-trimmed videos without padded tokens"
            )

    resolved = resolve_canonical_video_config(_ALGORITHM, config or {})
    base = resolved["base_config"]
    attention_layer = int(base["attn_layer"])
    hidden_layer = int(base["hidden_state_layer"])
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
    importance_by_video = _importance_by_video(payload, torch=torch)
    default_dominant = int(base["dominant_num"])
    default_contextual = int(base["contextual_num"])
    base_budget = default_dominant + default_contextual
    upper_limit = int(resolved["upper_limit"])

    outputs: list[Any] = []
    groups_by_video: list[tuple[tuple[int, ...], ...]] = []
    representatives_by_video: list[tuple[int, ...]] = []
    merge_targets_by_video: list[tuple[int, ...]] = []
    exact_budgets: list[int] = []
    total_budgets_by_video: list[tuple[int, ...]] = []
    dominant_budgets_by_video: list[tuple[int, ...]] = []
    contextual_budgets_by_video: list[tuple[int, ...]] = []
    frame_records_by_video: list[tuple[dict[str, Any], ...]] = []
    anchor_legalizations_by_video: list[int] = []

    for video_index in range(payload.video_count):
        span = payload.video_span(video_index)
        source = payload.features[span]
        scores = attention[span]
        similarity = similarity_features[span]
        coordinates = payload.coordinates[span]
        temporal_spans = _temporal_spans(coordinates)
        _validate_uniform_raster(coordinates, temporal_spans)
        source_counts = tuple(stop - start for _temporal, start, stop in temporal_spans)
        weights = importance_by_video[video_index]
        if rules and "frame_scorer" in rules:
            weights = rule_contract.scores(rules["frame_scorer"],
                {"values": source, "coordinates": coordinates,
                 "frame_spans": temporal_spans, "default_scores": weights,
                 "signals": payload.signals, "video_index": video_index},
                count=len(temporal_spans), reference=source, name="frame_scorer")
        if int(weights.numel()) != len(temporal_spans):
            raise BackendIntegrationError(
                "DyToK detection frame count differs from projected temporal slices"
            )
        total_budgets = allocate_dytok_frame_budgets(
            weights,
            source_counts,
            base_budget_per_frame=base_budget,
            upper_limit=upper_limit,
        )
        if rules and "budget_allocator" in rules:
            with torch.no_grad():
                allocated = rules["budget_allocator"]({"scores": weights,
                    "source_counts": source_counts, "upper_limit": upper_limit,
                    "total_budget": sum(total_budgets), "default_budgets": total_budgets})
            if isinstance(allocated, torch.Tensor):
                if allocated.ndim != 1 or allocated.dtype not in (torch.int32, torch.int64):
                    raise BackendIntegrationError("DyTok budget_allocator must return integer frame budgets")
                allocated = allocated.tolist()
            allocated = tuple(allocated)
            if (len(allocated) != len(source_counts) or sum(allocated) != sum(total_budgets)
                    or any(isinstance(n, bool) or not isinstance(n, int) or n < 0
                           or n > min(cap, upper_limit) for n, cap in zip(allocated, source_counts))):
                raise BackendIntegrationError("DyTok budget_allocator must preserve the total and frame capacities")
            total_budgets = allocated
        dominant_budgets = tuple((6 * value) // 7 for value in total_budgets)
        contextual_budgets = tuple(
            total - dominant for total, dominant in zip(total_budgets, dominant_budgets)
        )

        video_outputs: list[Any] = []
        video_groups: list[tuple[int, ...]] = []
        video_targets: list[int] = []
        video_representatives: list[int] = []
        video_records: list[dict[str, Any]] = []
        video_legalizations = 0
        for frame_position, (temporal, start, stop) in enumerate(temporal_spans):
            dominant_num = dominant_budgets[frame_position]
            contextual_num = contextual_budgets[frame_position]
            total_budget = total_budgets[frame_position]
            if total_budget == 0:
                frame_groups: tuple[tuple[int, ...], ...] = ()
                frame_targets: tuple[int, ...] = ()
                dominant: tuple[int, ...] = ()
                contextual: tuple[int, ...] = ()
                frame_representatives: tuple[int, ...] = ()
                legalizations = 0
            else:
                frame_groups, frame_targets, dominant, contextual = (
                    _visionzip_frame_groups(
                        scores[start:stop],
                        similarity[start:stop],
                        dominant_num=dominant_num,
                        contextual_num=contextual_num,
                        torch=torch,
                        rules=rules,
                        values=source[start:stop],
                    )
                )
                frame_representatives, legalizations = _causal_layout_anchors(
                    frame_groups,
                    preferred=frame_targets,
                )
                video_outputs.append(
                    _merge_frame_values(
                        source[start:stop],
                        frame_groups,
                        merge_targets=frame_targets,
                        torch=torch,
                        rules=rules,
                    )
                )
            video_groups.extend(
                tuple(index + start for index in group) for group in frame_groups
            )
            video_targets.extend(index + start for index in frame_targets)
            video_representatives.extend(
                index + start for index in frame_representatives
            )
            video_records.append(
                {
                    "temporal_index": temporal,
                    "source_count": stop - start,
                    "frame_importance": float(weights[frame_position].item()),
                    "total_budget": total_budget,
                    "dominant_budget": dominant_num,
                    "contextual_budget": contextual_num,
                    "dominant_indices": tuple(index + start for index in dominant),
                    "contextual_target_indices": tuple(
                        index + start for index in contextual
                    ),
                    "representative_indices": tuple(
                        index + start for index in frame_representatives
                    ),
                    "anchor_legalizations": legalizations,
                }
            )
            video_legalizations += legalizations
        if not video_outputs:
            raise InvariantError("DyToK must produce at least one token per video")
        outputs.append(torch.cat(video_outputs, dim=0))
        groups_by_video.append(tuple(video_groups))
        merge_targets_by_video.append(tuple(video_targets))
        representatives_by_video.append(tuple(video_representatives))
        exact_budgets.append(sum(total_budgets))
        total_budgets_by_video.append(total_budgets)
        dominant_budgets_by_video.append(dominant_budgets)
        contextual_budgets_by_video.append(contextual_budgets)
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
            "mode": "dynamic",
            "detection_stage": "custom" if rules and ('frame_scorer' in rules) else "frozen_llm_last_query_visual_attention",
            "target_layers": tuple(
                int(value) for value in payload.signals.get("detection_attention", {}).get(
                    "target_layers", resolved["llm_config"]["target_layers"])
            ),
            "base_compressor": "visionzip.video",
            "base_budget_per_temporal_slice": base_budget,
            "upper_limit": upper_limit,
            "total_budgets_by_video": tuple(total_budgets_by_video),
            "dominant_budgets_by_video": tuple(dominant_budgets_by_video),
            "contextual_budgets_by_video": tuple(contextual_budgets_by_video),
            "budget_total_policy": "fixed_total_subject_to_source_cap",
            "dominant_contextual_split": "floor_6_over_7_then_remainder",
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
            "frame_records_by_video": tuple(frame_records_by_video),
            "training_free_compressor": True,
            "image_variant_shared": False,
        },
    )


def _importance_by_video(
    payload: VideoTokenPayload,
    *,
    torch: Any,
) -> tuple[Any, ...]:
    raw = payload.signals.get(_IMPORTANCE_SIGNAL)
    if isinstance(raw, torch.Tensor) and payload.video_count == 1:
        raw = (raw,)
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
        raise BackendIntegrationError(
            f"DyToK requires {_IMPORTANCE_SIGNAL} with one tensor per video"
        )
    values = tuple(raw)
    if len(values) != payload.video_count:
        raise BackendIntegrationError(
            "DyToK frame importance must contain one entry per video"
        )
    result = []
    for value in values:
        if not isinstance(value, torch.Tensor) or value.ndim != 1:
            raise TypeError("DyToK frame importance entries must be rank-one tensors")
        normalized = value.detach().float().to(payload.features.device)
        if not bool(torch.isfinite(normalized).all().item()) or bool(
            (normalized < 0).any().item()
        ):
            raise ValueError("DyToK frame importance must be finite and non-negative")
        total = normalized.sum()
        if not bool(torch.isfinite(total).item()) or float(total.item()) <= 0:
            raise ValueError("DyToK frame importance must have positive total mass")
        result.append(normalized / total)
    return tuple(result)


def _torch() -> Any:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - dependency-light control plane
        raise BackendIntegrationError(
            "DyToK video runtime requires the existing PyTorch environment"
        ) from exc
    return torch


__all__ = [
    "DyToKKernel",
    "DyToKVideoKernel",
    "aggregate_dytok_frame_importance",
    "allocate_dytok_frame_budgets",
    "dytok_frame_attention_from_qk",
    "dytok_video",
]
