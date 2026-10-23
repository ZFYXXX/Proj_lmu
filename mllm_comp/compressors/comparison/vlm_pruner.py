"""VLM-Pruner BSS selection and selective-weighted aggregation for OV/Qwen2.

The official LLaVA implementation prunes immediately before decoder layer 3
(after two full layers), seeds four key-space pivots, expands the retained set
with Buffering for Spatial Sparsity (BSS), and fuses discarded states back into
their nearest retained tokens.  This module keeps those operations isolated
from the stock Qwen2 path and accepts OneVision's actual base/AnyRes layouts.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F


VLM_PRUNER_DEFAULTS: Dict[str, Any] = {
    "method": "vlm_pruner",
    "prune_after_layer": 2,
    "retain_ratio": 0.2222222222222222,
    "pivot_image_tokens": 4,
    "threshold": 0.8,
    "token_batch": 16,
    "variance_channels": 256,
    "spatial_reward_weight": 0.5,
    "retained_state_weight": 0.3,
    "aggregated_state_weight": 0.7,
    "position_mode": "preserve",
    "spatial_layout": "ov_base_and_anyres_segments",
    "single_image_only": True,
    "min_visual_tokens": 1,
    "log_first_n": 1,
}


def normalize_vlm_pruner_config(
    raw: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    cfg = dict(VLM_PRUNER_DEFAULTS)
    if raw:
        unknown = sorted(set(raw) - set(cfg))
        if unknown:
            raise ValueError(
                f"Unknown VLM-Pruner config keys: {', '.join(unknown)}"
            )
        cfg.update(raw)
    cfg["method"] = "vlm_pruner"
    for key in (
        "prune_after_layer",
        "pivot_image_tokens",
        "token_batch",
        "variance_channels",
        "min_visual_tokens",
        "log_first_n",
    ):
        cfg[key] = int(cfg[key])
    for key in (
        "retain_ratio",
        "threshold",
        "spatial_reward_weight",
        "retained_state_weight",
        "aggregated_state_weight",
    ):
        cfg[key] = float(cfg[key])
    cfg["position_mode"] = str(cfg["position_mode"]).strip().lower()
    cfg["spatial_layout"] = str(cfg["spatial_layout"]).strip().lower()
    cfg["single_image_only"] = bool(cfg["single_image_only"])

    if cfg["prune_after_layer"] < 1:
        raise ValueError("VLM-Pruner must leave at least one full scorer layer")
    if not 0.0 < cfg["retain_ratio"] <= 1.0:
        raise ValueError("VLM-Pruner retain_ratio must lie in (0, 1]")
    if cfg["pivot_image_tokens"] < 1:
        raise ValueError("VLM-Pruner pivot_image_tokens must be >= 1")
    if cfg["token_batch"] < 1:
        raise ValueError("VLM-Pruner token_batch must be >= 1")
    if cfg["variance_channels"] < 1:
        raise ValueError("VLM-Pruner variance_channels must be >= 1")
    if cfg["spatial_reward_weight"] < 0:
        raise ValueError("VLM-Pruner spatial_reward_weight must be non-negative")
    if cfg["retained_state_weight"] < 0 or cfg["aggregated_state_weight"] < 0:
        raise ValueError("VLM-Pruner aggregation weights must be non-negative")
    if not math.isclose(
        cfg["retained_state_weight"] + cfg["aggregated_state_weight"],
        1.0,
        rel_tol=0,
        abs_tol=1e-12,
    ):
        raise ValueError("VLM-Pruner aggregation weights must sum to one")
    if cfg["position_mode"] != "preserve":
        raise ValueError("VLM-Pruner reproduction preserves original positions")
    if cfg["spatial_layout"] != "ov_base_and_anyres_segments":
        raise ValueError(
            "OV VLM-Pruner requires spatial_layout='ov_base_and_anyres_segments'"
        )
    if cfg["min_visual_tokens"] < 1:
        raise ValueError("VLM-Pruner min_visual_tokens must be >= 1")
    if cfg["log_first_n"] < 0:
        raise ValueError("VLM-Pruner log_first_n must be >= 0")
    return cfg


def ov_spatial_distance_matrix(
    visual_tokens: int,
    segments: Optional[Sequence[Dict[str, Any]]],
    *,
    device: torch.device,
) -> torch.Tensor:
    """Build the paper's normalized spatial-distance matrix for OV layouts.

    The release lays one image out row-major and divides Euclidean distance by
    ``sqrt(height**2 + width**2)``.  OneVision has a base view followed by an
    independently reconstructed AnyRes view (and flat mode can contain several
    crops), so distances are computed with that exact formula inside each real
    view.  Tokens from different views receive normalized distance one: there
    is no defensible shared pixel coordinate, and this rewards representing an
    as-yet unrepresented view without fabricating spatial adjacency.
    """

    count = int(visual_tokens)
    if count < 1:
        raise ValueError("VLM-Pruner needs a non-empty visual span")
    if not segments:
        width = max(1, int(round(math.sqrt(count))))
        segments = [{"start": 0, "end": count, "grid_width": width}]

    coordinates = torch.empty(count, 2, device=device, dtype=torch.float32)
    segment_ids = torch.empty(count, device=device, dtype=torch.long)
    normalizers = torch.empty(count, device=device, dtype=torch.float32)
    covered = torch.zeros(count, device=device, dtype=torch.bool)
    for segment_id, segment in enumerate(segments):
        start = int(segment["start"])
        end = int(segment["end"])
        width = int(segment["grid_width"])
        if not (0 <= start < end <= count) or width < 1:
            raise ValueError(f"Invalid VLM-Pruner spatial segment: {segment}")
        length = end - start
        height = int(math.ceil(length / float(width)))
        local = torch.arange(length, device=device)
        x = (local % width).float()
        y = torch.div(local, width, rounding_mode="floor").float()
        coordinates[start:end] = torch.stack((x, y), dim=-1)
        segment_ids[start:end] = int(segment_id)
        normalizers[start:end] = float(
            math.sqrt(height * height + width * width)
        )
        covered[start:end] = True
    if not bool(covered.all()):
        raise ValueError("VLM-Pruner spatial segments do not cover the visual span")
    distance = torch.cdist(coordinates, coordinates)
    same_segment = segment_ids[:, None] == segment_ids[None, :]
    distance = distance / normalizers[:, None]
    return torch.where(same_segment, distance, torch.ones_like(distance))


@torch.no_grad()
def _select_bss_indices(
    visual_hidden: torch.Tensor,
    visual_keys: torch.Tensor,
    spatial_distance: torch.Tensor,
    *,
    target: int,
    cfg: Dict[str, Any],
) -> Tuple[torch.LongTensor, torch.Tensor, Dict[str, Any]]:
    """Official pivot+BSS selection; returns indices and detached fusion weights."""

    visual_count, hidden_size = visual_hidden.shape
    if visual_keys.ndim != 2 or visual_keys.shape[0] != visual_count:
        raise ValueError("VLM-Pruner key states do not match visual hidden states")
    if spatial_distance.shape != (visual_count, visual_count):
        raise ValueError("VLM-Pruner spatial distances do not match visual tokens")
    retain = min(visual_count, max(int(cfg["min_visual_tokens"]), int(target)))
    if retain >= visual_count:
        selected = torch.arange(visual_count, device=visual_hidden.device)
        empty = visual_hidden.new_empty((0, visual_count), dtype=torch.float32)
        return selected, empty, {
            "pivot_tokens": 0,
            "selection_fallback_tokens": 0,
        }

    pivot_count = min(int(cfg["pivot_image_tokens"]), retain, visual_count)
    pivots = torch.empty(pivot_count, dtype=torch.long, device=visual_hidden.device)
    pivots[0] = visual_keys.norm(p=1, dim=-1).argmax()
    for pivot_index in range(1, pivot_count):
        distances = torch.cdist(
            visual_keys.float(), visual_keys[pivots[:pivot_index]].float()
        )
        minimum = distances.min(dim=1).values
        minimum[pivots[:pivot_index]] = -torch.inf
        pivots[pivot_index] = minimum.argmax()
    selected = [int(value) for value in pivots.tolist()]

    channel_count = min(int(cfg["variance_channels"]), hidden_size)
    # The public implementation uses torch.var's default unbiased estimator.
    variances = visual_hidden.float().var(dim=0)
    channels = torch.topk(variances, k=channel_count).indices
    screened = F.normalize(
        visual_hidden[:, channels].float(), dim=-1, eps=1e-8
    )
    cosine = screened @ screened.transpose(0, 1)
    def adjusted(rows: torch.Tensor, cols: torch.Tensor) -> torch.Tensor:
        semantic = cosine.index_select(0, rows).index_select(1, cols)
        distance = spatial_distance.index_select(0, rows).index_select(1, cols)
        # The release first takes the minimum spatial distance to the retained
        # set, then broadcasts that one reward over every semantic pair.
        spatial_reward = distance.amin(dim=1, keepdim=True)
        return semantic * (
            1.0 + float(cfg["spatial_reward_weight"]) * spatial_reward
        )

    threshold = float(cfg["threshold"]) - 0.1
    token_batch = int(cfg["token_batch"])
    fallback_count = 0
    while len(selected) < retain:
        threshold += 0.1
        selected_tensor = torch.tensor(
            selected, dtype=torch.long, device=visual_hidden.device
        )
        candidate_mask = torch.ones(
            visual_count, dtype=torch.bool, device=visual_hidden.device
        )
        candidate_mask[selected_tensor] = False
        candidates = candidate_mask.nonzero(as_tuple=True)[0]
        if candidates.numel() == 0:
            break
        nonduplicate = 1.0 - adjusted(candidates, selected_tensor).max(dim=1).values
        ordered = candidates[nonduplicate.argsort(descending=True)]
        before = len(selected)
        for offset in range(0, int(ordered.numel()), token_batch):
            if offset == token_batch and before == len(selected):
                break
            batch = ordered[offset : offset + token_batch]
            current = torch.tensor(
                selected, dtype=torch.long, device=visual_hidden.device
            )
            maximum = adjusted(batch, current).max(dim=1).values
            valid = batch[maximum < threshold]
            remaining = retain - len(selected)
            if int(valid.numel()) > remaining:
                # The released implementation chooses the most non-duplicate
                # rows from the whole final batch at this boundary.
                valid = batch[torch.topk(1.0 - maximum, k=remaining).indices]
            if valid.numel():
                selected.extend(int(value) for value in valid.tolist())
                if len(selected) >= retain:
                    break
        if len(selected) == before:
            # The public loop can stop below its nominal budget at this point.
            # The experiment contract requires an exact token rate, so retain
            # the same centrifugal ordering and fill the next batch instead.
            remaining = retain - len(selected)
            selected.extend(int(value) for value in ordered[:remaining].tolist())
            fallback_count += min(remaining, int(ordered.numel()))
            break

    if len(selected) < retain:
        missing = retain - len(selected)
        selected_tensor = torch.tensor(
            selected, dtype=torch.long, device=visual_hidden.device
        )
        mask = torch.ones(visual_count, dtype=torch.bool, device=visual_hidden.device)
        mask[selected_tensor] = False
        candidates = mask.nonzero(as_tuple=True)[0]
        nonduplicate = 1.0 - adjusted(candidates, selected_tensor).max(dim=1).values
        fill = candidates[nonduplicate.argsort(descending=True)[:missing]]
        selected.extend(int(value) for value in fill.tolist())
        fallback_count += int(fill.numel())

    selected_tensor = torch.tensor(
        selected[:retain], dtype=torch.long, device=visual_hidden.device
    )
    if int(torch.unique(selected_tensor).numel()) != retain:
        raise RuntimeError("VLM-Pruner BSS produced duplicate retained tokens")

    all_indices = torch.arange(visual_count, device=visual_hidden.device)
    discarded_mask = torch.ones(
        visual_count, dtype=torch.bool, device=visual_hidden.device
    )
    discarded_mask[selected_tensor] = False
    discarded = all_indices[discarded_mask]
    if discarded.numel():
        discarded_similarity = cosine.index_select(0, discarded).index_select(
            1, selected_tensor
        )
        nearest = discarded_similarity.argmax(dim=1)
        one_hot = torch.zeros_like(discarded_similarity)
        one_hot.scatter_(1, nearest.unsqueeze(1), 1.0)
        weights = discarded_similarity * one_hot
        weights = weights / weights.sum(dim=0, keepdim=True).clamp(min=1e-8)
    else:
        weights = cosine.new_empty((0, retain))
    return selected_tensor, weights, {
        "pivot_tokens": int(pivot_count),
        "variance_channels_used": int(channel_count),
        "selection_fallback_tokens": int(fallback_count),
        "discarded_tokens": int(discarded.numel()),
    }


def build_vlm_pruner_plan(
    hidden_states: torch.Tensor,
    key_states: torch.Tensor,
    *,
    visual_span: Tuple[int, int],
    cfg: Dict[str, Any],
    spatial_segments: Optional[Sequence[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """Select, fuse, and compact one Qwen2 prefill sequence."""

    if hidden_states.ndim != 3 or hidden_states.shape[0] != 1:
        raise ValueError("VLM-Pruner requires hidden states [1,S,D]")
    if key_states.ndim != 3 or key_states.shape[:2] != hidden_states.shape[:2]:
        raise ValueError("VLM-Pruner projected keys must have shape [1,S,KD]")
    start, end = map(int, visual_span)
    sequence_length = int(hidden_states.shape[1])
    if not 0 <= start < end <= sequence_length:
        raise ValueError("VLM-Pruner visual span is invalid")
    visual_count = end - start
    target = min(
        visual_count,
        max(
            int(cfg["min_visual_tokens"]),
            int(round(visual_count * float(cfg["retain_ratio"]))),
        ),
    )
    spatial_distance = ov_spatial_distance_matrix(
        visual_count, spatial_segments, device=hidden_states.device
    )
    selected, weights, selection_stats = _select_bss_indices(
        hidden_states[0, start:end].detach(),
        key_states[0, start:end].detach(),
        spatial_distance,
        target=target,
        cfg=cfg,
    )

    updated = hidden_states.clone()
    all_visual = torch.arange(visual_count, device=hidden_states.device)
    discarded_mask = torch.ones(
        visual_count, dtype=torch.bool, device=hidden_states.device
    )
    discarded_mask[selected] = False
    discarded = all_visual[discarded_mask]
    if discarded.numel():
        discarded_hidden = hidden_states[0, start:end].index_select(0, discarded)
        aggregated = weights.transpose(0, 1).to(hidden_states.dtype) @ discarded_hidden
        update_mask = weights.sum(dim=0) > 0
        selected_global = start + selected[update_mask]
        updated[0, selected_global] = (
            float(cfg["retained_state_weight"])
            * hidden_states[0, selected_global]
            + float(cfg["aggregated_state_weight"])
            * aggregated[update_mask]
        )

    selected_sorted = selected.sort().values
    keep_indices = torch.cat(
        (
            torch.arange(start, device=hidden_states.device),
            start + selected_sorted,
            torch.arange(end, sequence_length, device=hidden_states.device),
        )
    )
    return {
        "hidden_states": updated.index_select(1, keep_indices),
        "keep_indices": keep_indices,
        "selected_visual_indices": selected_sorted,
        "visual_tokens_before": int(visual_count),
        "visual_tokens_after": int(selected_sorted.numel()),
        "sequence_tokens_before": int(sequence_length),
        "sequence_tokens_after": int(keep_indices.numel()),
        "fusion_weights": weights,
        "selection_order": selected,
        "discarded_visual_indices": discarded,
        **selection_stats,
    }
