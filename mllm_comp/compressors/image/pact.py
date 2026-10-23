"""PACT pruning, density-peak clustering and proportional token weights.

This is the shared single-sample kernel from the supplied Qwen2.5-VL port.
The provider owns Q/K extraction, the decoder boundary, sequence-field repair
and application of the returned proportional-attention weights.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from typing import Any, Mapping

from mllm_comp.core.layout import LayoutDelta, OrderingPolicy
from mllm_comp.core.errors import BackendIntegrationError, InvariantError
from ._rules import score_rule, selection_rule, rule_metadata, replay_rule_groups
from .decoder_lifecycle import layout_merge_groups


@dataclass(frozen=True)
class PACTSpec:
    reduction_layer: int = 4
    prune_keep_ratio: float = 1.0
    target_keep_ratio: float = 0.3
    cutoff: float | None = None
    density_scale: float = 2.0
    include_pruned: bool = True
    pruned_cutoff_coefficient: float = 1.5
    multiply_by_hidden_norm: bool = True
    proportional_attention: bool = True
    cutoff_search_steps: int = 20
    min_visual_tokens: int = 1

    def __post_init__(self) -> None:
        if (
            isinstance(self.reduction_layer, bool)
            or not isinstance(self.reduction_layer, int)
            or self.reduction_layer < 1
        ):
            raise ValueError("reduction_layer must be a positive 1-indexed layer.")
        for name in (
            "prune_keep_ratio",
            "target_keep_ratio",
            "density_scale",
            "pruned_cutoff_coefficient",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(f"{name} must be a real number.")
            if float(value) <= 0.0:
                raise ValueError(f"{name} must be positive.")
            object.__setattr__(self, name, float(value))
        if self.prune_keep_ratio > 1.0 or self.target_keep_ratio > 1.0:
            raise ValueError("PACT keep ratios must be at most one.")
        if self.cutoff is not None and (
            isinstance(self.cutoff, bool)
            or not isinstance(self.cutoff, (int, float))
            or not 0.0 <= float(self.cutoff) <= 2.0
        ):
            raise ValueError("cutoff must be in [0, 2] or None.")
        if self.cutoff is not None:
            object.__setattr__(self, "cutoff", float(self.cutoff))
        for name in ("include_pruned", "multiply_by_hidden_norm", "proportional_attention"):
            if not isinstance(getattr(self, name), bool):
                raise TypeError(f"{name} must be a boolean.")
        for name in ("cutoff_search_steps", "min_visual_tokens"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer.")

    @property
    def boundary_layer(self) -> int:
        """Zero-based completed-layer boundary used by framework adapters."""

        return self.reduction_layer - 1


@dataclass(frozen=True)
class PACTClusterAssignment:
    centers: Any
    labels: Any
    cutoff: float


@dataclass(frozen=True)
class PACTResult:
    hidden_states: Any
    kept_indices: Any
    result_image_indices: Any
    representative_indices: Any
    token_sizes: Any
    scores: Any
    cluster_assignment: PACTClusterAssignment
    layout_delta: LayoutDelta
    metadata: Mapping[str, Any] = field(default_factory=dict)


def pact_image_tokens(
    hidden_states: Any,
    *,
    image_indices: Any,
    query_states: Any = None,
    key_states: Any = None,
    clustering_keys: Any,
    image_attention_scores: Any | None = None,
    spec: PACTSpec | None = None,
    media_unit: str = "llm_token",
    source_layout: Mapping[str, Any] | None = None,
    result_layout: Mapping[str, Any] | None = None,
    rules: Mapping[str, Any] | None = None,
) -> PACTResult:
    """Apply PACT to one serialized image span in a decoder sequence."""

    torch = _torch()
    active = spec or PACTSpec()
    if (
        not isinstance(hidden_states, torch.Tensor)
        or hidden_states.ndim != 2
        or not hidden_states.is_floating_point()
    ):
        raise ValueError("hidden_states must be a floating rank-2 tensor.")
    length = int(hidden_states.shape[0])
    images = _index_tensor(image_indices, device=hidden_states.device)
    _validate_indices(images, length)
    image_count = int(images.numel())
    if image_count < 1:
        raise ValueError("PACT requires at least one image token.")
    for name, value in (() if image_attention_scores is not None else (
        ("query_states", query_states),
        ("key_states", key_states),
    )):
        if (
            not isinstance(value, torch.Tensor)
            or value.ndim != 3
            or not value.is_floating_point()
            or int(value.shape[1]) != length
        ):
            raise ValueError(f"{name} must be [heads, sequence, head_dim].")
    if image_attention_scores is None and query_states.shape != key_states.shape:
        raise ValueError("query_states and key_states must have equal shapes.")
    if (
        not isinstance(clustering_keys, torch.Tensor)
        or clustering_keys.ndim != 2
        or not clustering_keys.is_floating_point()
        or int(clustering_keys.shape[0]) != length
    ):
        raise ValueError("clustering_keys must be [sequence, feature_dim].")

    scores = pact_token_scores(
        query_states,
        key_states,
        images,
        hidden_norms=(
            hidden_states.detach().float().norm(dim=-1)
            if active.multiply_by_hidden_norm
            else None
        ),
        rules=rules,
    ) if image_attention_scores is None else image_attention_scores
    if not isinstance(scores, torch.Tensor) or scores.ndim != 1 or int(scores.numel()) != image_count:
        raise ValueError("image_attention_scores must contain one score per image token.")
    scores = scores.to(hidden_states.device)
    clustering_keys = clustering_keys.to(hidden_states.device)
    prune_count = max(
        active.min_visual_tokens,
        int(image_count * active.prune_keep_ratio),
    )
    prune_count = min(image_count, prune_count)
    scores = score_rule(rules, "scorer", {
        "hidden_states": hidden_states, "image_indices": images,
        "image_attention_scores": scores, "clustering_keys": clustering_keys,
        "keep_count": prune_count,
    }, scores)
    keep_mask = pact_score_keep_mask(scores, prune_count)
    if prune_count < image_count:
        selected = selection_rule(rules, "selector", {"hidden_states": hidden_states,
            "image_indices": images, "scores": scores, "keep_count": prune_count},
            count=image_count, keep_count=prune_count, device=images.device)
        if selected is not None:
            keep_mask = torch.zeros_like(keep_mask)
            keep_mask[selected] = True
    kept_images = images[keep_mask]
    pruned_images = images[~keep_mask]
    kept_keys = clustering_keys.index_select(0, kept_images).detach()
    include_pruned = active.include_pruned and int(pruned_images.numel()) > 0
    pruned_keys = (
        clustering_keys.index_select(0, pruned_images).detach()
        if include_pruned
        else None
    )
    target_centers = max(
        active.min_visual_tokens,
        int(image_count * active.target_keep_ratio),
    )
    target_centers = min(int(kept_images.numel()), target_centers)
    clusterer = PACTDensityPeakClusterer(density_scale=active.density_scale)
    if active.cutoff is None:
        assignment = clusterer.fit_with_target(
            kept_keys,
            target_centers,
            pruned_keys=pruned_keys,
            pruned_cutoff_coefficient=active.pruned_cutoff_coefficient,
            search_steps=active.cutoff_search_steps,
        )
    else:
        assignment = clusterer.fit(
            kept_keys,
            active.cutoff,
            pruned_keys=pruned_keys,
            pruned_cutoff_coefficient=active.pruned_cutoff_coefficient,
        )
    if rules and "grouper" in rules and int(kept_images.numel()) > 1:
        from types import MappingProxyType
        with torch.no_grad():
            plan = rules["grouper"](MappingProxyType({
                "kept_keys": kept_keys, "pruned_keys": pruned_keys,
                "kept_indices": kept_images, "pruned_indices": pruned_images if include_pruned else pruned_images[:0],
                "center_count": int(assignment.centers.numel()),
                "default_centers": assignment.centers, "default_labels": assignment.labels,
                "cutoff": assignment.cutoff,
            }))
        if not isinstance(plan, Mapping) or set(plan) != {"centers", "labels"}:
            raise ValueError("PACT grouper must return centers and labels.")
        centers, labels = plan["centers"], plan["labels"]
        integer_types = (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8)
        if any(not isinstance(v, torch.Tensor) or v.ndim != 1 or v.dtype not in integer_types for v in (centers, labels)):
            raise ValueError("PACT grouper centers and labels must be integer vectors.")
        centers = centers.detach().to(device=images.device, dtype=torch.long).sort().values
        labels = labels.detach().to(device=images.device, dtype=torch.long)
        if (centers.numel() != assignment.centers.numel() or centers.unique().numel() != centers.numel()
                or bool(((centers < 0) | (centers >= kept_images.numel())).any())
                or labels.shape != assignment.labels.shape
                or not set(labels.tolist()).issubset({-1, *centers.tolist()})
                or (not include_pruned and bool((labels[:kept_images.numel()] < 0).any()))
                or not torch.equal(labels.index_select(0, centers), centers)):
            raise ValueError("PACT grouper must preserve center budget, retain kept tokens without pruned reassignment, and assign each center to itself.")
        assignment = PACTClusterAssignment(centers, labels, assignment.cutoff)
    merged, center_mask, center_sizes = pact_merge_clusters(
        hidden_states.index_select(0, kept_images),
        assignment,
        pruned_hidden_states=(
            hidden_states.index_select(0, pruned_images) if include_pruned else None
        ),
    )
    center_local = center_mask.nonzero(as_tuple=False).flatten()
    center_sources = kept_images.index_select(0, center_local)
    updated = hidden_states.clone()
    updated[kept_images] = merged
    image_set = set(int(value) for value in images.tolist())
    text_indices = torch.tensor(
        [index for index in range(length) if index not in image_set],
        dtype=torch.long,
        device=hidden_states.device,
    )
    kept = torch.cat((text_indices, center_sources)).sort().values
    output = updated.index_select(0, kept)
    center_positions = torch.searchsorted(kept, center_sources)
    result_images = center_positions.sort().values
    sizes = torch.ones(int(kept.numel()), dtype=torch.float32, device=kept.device)
    sizes[center_positions] = center_sizes.index_select(0, center_local)

    output_position = {int(source): index for index, source in enumerate(kept.tolist())}
    old_to_new: list[int | None] = [None] * length
    for source in text_indices.tolist():
        old_to_new[int(source)] = output_position[int(source)]
    centers = assignment.centers.tolist()
    labels = assignment.labels.tolist()
    center_result = {
        int(center): output_position[int(kept_images[int(center)].item())]
        for center in centers
    }
    combined_sources = kept_images.tolist()
    if include_pruned:
        combined_sources.extend(pruned_images.tolist())
    for local, center in enumerate(labels):
        if center >= 0:
            old_to_new[int(combined_sources[local])] = center_result[int(center)]
    representative_values = tuple(int(value) for value in kept.tolist())
    resolved_layout = dict(result_layout or {})
    resolved_layout.setdefault("representative_indices", representative_values)
    resolved_layout.setdefault(
        "token_sizes", tuple(float(value) for value in sizes.tolist())
    )
    delta = LayoutDelta.from_mapping(
        source_count=length,
        result_count=int(output.shape[0]),
        old_to_new=old_to_new,
        media_unit=media_unit,
        protected_indices=tuple(int(value) for value in text_indices.tolist()),
        source_layout=source_layout or {},
        result_layout=resolved_layout,
        ordering_policy=OrderingPolicy.EXPLICIT,
    )
    result = PACTResult(
        hidden_states=output,
        kept_indices=kept,
        result_image_indices=result_images,
        representative_indices=kept,
        token_sizes=sizes,
        scores=scores,
        cluster_assignment=assignment,
        layout_delta=delta,
        metadata={
            **rule_metadata(rules),
            "algorithm": "pact",
            "source_tokens": length,
            "result_tokens": int(output.shape[0]),
            "source_image_tokens": image_count,
            "prune_survivors": int(kept_images.numel()),
            "target_centers": target_centers,
            "result_image_tokens": int(center_sources.numel()),
            "cutoff": assignment.cutoff,
            "include_pruned": include_pruned,
            "proportional_attention": active.proportional_attention,
            "compression_applied": not delta.is_identity,
        },
    )

    if rules and "aggregator" in rules:
        result = replace(result, hidden_states=replay_rule_groups(
            hidden_states, layout_merge_groups(delta), rules=rules))
    return result


def pact_token_scores(
    query_states: Any,
    key_states: Any,
    image_indices: Any,
    *,
    hidden_norms: Any | None = None,
    rules: Mapping[str, Any] | None = None,
) -> Any:
    """Compute PACT's EUTI score from pre-RoPE Q/K states."""

    torch = _torch()
    functional = _functional()
    images = _index_tensor(image_indices, device=query_states.device)
    global_query = query_states.index_select(1, images).mean(dim=1, keepdim=True)
    image_keys = key_states.index_select(1, images)
    attention = torch.matmul(
        global_query.float() / math.sqrt(int(query_states.shape[-1])),
        image_keys.float().transpose(-1, -2),
    )
    attention = functional.softmax(attention, dim=-1, dtype=torch.float32)
    scores = torch.nan_to_num(attention, nan=0.0).mean(dim=0).squeeze(0)
    if hidden_norms is not None:
        scores = scores * hidden_norms.to(scores.device).index_select(0, images)
    return score_rule(rules, "source_scorer", {"query_states": query_states,
        "key_states": key_states, "image_indices": images, "hidden_norms": hidden_norms,
        "attention": attention}, scores)


def pact_score_keep_mask(scores: Any, keep_count: int) -> Any:
    torch = _torch()
    count = int(scores.numel())
    if keep_count >= count:
        return torch.ones(count, dtype=torch.bool, device=scores.device)
    if keep_count <= 0:
        return torch.zeros(count, dtype=torch.bool, device=scores.device)
    order = torch.argsort(scores, descending=True, stable=True)
    mask = torch.zeros(count, dtype=torch.bool, device=scores.device)
    mask[order[:keep_count]] = True
    return mask


class PACTDensityPeakClusterer:
    """Rank-stabilized DBDPC used by the supplied implementation."""

    def __init__(self, *, density_scale: float = 2.0) -> None:
        self.density_scale = float(density_scale)

    def fit(
        self,
        features: Any,
        cutoff: float,
        *,
        pruned_keys: Any | None = None,
        pruned_cutoff_coefficient: float = 1.5,
    ) -> PACTClusterAssignment:
        torch = _torch()
        with torch.no_grad():
            count = int(features.shape[0])
            if count == 0:
                empty = torch.empty(0, dtype=torch.long, device=features.device)
                return PACTClusterAssignment(empty, empty, float(cutoff))
            if count == 1:
                total = count + (0 if pruned_keys is None else int(pruned_keys.shape[0]))
                return PACTClusterAssignment(
                    torch.zeros(1, dtype=torch.long, device=features.device),
                    torch.zeros(total, dtype=torch.long, device=features.device),
                    float(cutoff),
                )
            distances = pact_cosine_pairwise_distance(features)
            density = torch.exp(-((distances / self.density_scale) ** 2)).sum(dim=1)
            density_order = torch.argsort(density, stable=True)
            ranks = torch.empty_like(density_order, dtype=density.dtype)
            ranks[density_order] = torch.arange(
                count, dtype=density.dtype, device=features.device
            )
            centers: list[int] = []
            unassigned = torch.ones(count, dtype=torch.bool, device=features.device)
            while True:
                unassigned_indices = unassigned.nonzero(as_tuple=False).flatten()
                active_count = int(unassigned_indices.numel())
                if active_count == 0:
                    break
                active_distances = distances[unassigned_indices][:, unassigned_indices]
                active_density = ranks[unassigned_indices]
                rows = active_density.unsqueeze(1).expand(active_count, active_count)
                higher = ~torch.gt(rows, rows.transpose(0, 1))
                higher.fill_diagonal_(False)
                conditional = torch.where(
                    higher,
                    active_distances,
                    torch.full_like(active_distances, float("inf")),
                )
                delta = conditional.min(dim=1).values
                delta[active_density == active_density.max()] = float("inf")
                new_centers = unassigned_indices[delta > cutoff]
                new_count = int(new_centers.numel())
                if new_count == 0:
                    break
                centers.extend(int(value) for value in new_centers.tolist())
                covered = (distances[unassigned_indices][:, new_centers] <= cutoff).any(dim=1)
                unassigned[unassigned_indices[covered]] = False
                unassigned[new_centers] = False
                if new_count < 10:
                    delta_order = torch.argsort(delta, descending=True, stable=True)
                    ordered_indices = unassigned_indices[delta_order]
                    ordered_delta = delta[delta_order]
                    first = int(torch.searchsorted(-ordered_delta, -cutoff).item())
                    centers.extend(int(value) for value in ordered_indices[:first].tolist())
                    remaining = ordered_indices[first:]
                    if int(remaining.numel()):
                        remaining = remaining[
                            torch.argsort(ranks[remaining], descending=True, stable=True)
                        ]
                        if centers:
                            centers_tensor = torch.tensor(
                                centers, dtype=torch.long, device=features.device
                            )
                            remaining = remaining[
                                (distances[remaining][:, centers_tensor] > cutoff).all(dim=1)
                            ]
                        if int(remaining.numel()):
                            neighbourhood = distances[remaining][:, remaining] <= cutoff
                            neighbourhood.fill_diagonal_(False)
                            claimed = torch.zeros(
                                int(remaining.numel()),
                                dtype=torch.bool,
                                device=features.device,
                            )
                            for index in range(int(remaining.numel())):
                                if not bool(claimed[index].item()):
                                    centers.append(int(remaining[index].item()))
                                    claimed[index + 1 :] |= neighbourhood[index, index + 1 :]
                    break
            centers = list(dict.fromkeys(centers))
            if not centers:
                centers = [int(torch.argmax(ranks).item())]
            centers_tensor = torch.tensor(
                centers, dtype=torch.long, device=features.device
            )
            centers_tensor = torch.unique(centers_tensor, sorted=True)
            if pruned_keys is None:
                nearest = distances[:, centers_tensor].argmin(dim=1)
                labels = centers_tensor[nearest]
                labels[centers_tensor] = centers_tensor
            else:
                functional = _functional()
                normalised = functional.normalize(features.float(), p=2.0, dim=1)
                pruned = functional.normalize(pruned_keys.float(), p=2.0, dim=1)
                pruned_distances = (1.0 - pruned @ normalised[centers_tensor].transpose(0, 1)).clamp(min=0.0)
                all_distances = torch.cat(
                    (distances[:, centers_tensor], pruned_distances), dim=0
                )
                minimum, nearest = all_distances.min(dim=1)
                labels = torch.full(
                    (int(all_distances.shape[0]),),
                    -1,
                    dtype=torch.long,
                    device=features.device,
                )
                accepted = minimum <= cutoff * pruned_cutoff_coefficient
                labels[accepted] = centers_tensor[nearest[accepted]]
                labels[centers_tensor] = centers_tensor
            used_centers = torch.unique(labels[labels >= 0], sorted=True)
            return PACTClusterAssignment(used_centers, labels, float(cutoff))

    def fit_with_target(
        self,
        features: Any,
        target_count: int,
        *,
        pruned_keys: Any | None = None,
        pruned_cutoff_coefficient: float = 1.5,
        search_steps: int = 20,
    ) -> PACTClusterAssignment:
        lower, upper = 0.0, 2.0
        result: PACTClusterAssignment | None = None
        for _ in range(search_steps):
            cutoff = (lower + upper) / 2.0
            result = self.fit(
                features,
                cutoff,
                pruned_keys=pruned_keys,
                pruned_cutoff_coefficient=pruned_cutoff_coefficient,
            )
            if int(result.centers.numel()) > target_count:
                lower = cutoff
            else:
                upper = cutoff
        assert result is not None
        return result


def pact_merge_clusters(
    hidden_states: Any,
    assignment: PACTClusterAssignment,
    *,
    pruned_hidden_states: Any | None = None,
) -> tuple[Any, Any, Any]:
    """Differentiably replace cluster centers with their member means."""

    torch = _torch()
    functional = _functional()
    count = int(hidden_states.shape[0])
    centers = assignment.centers
    if int(centers.numel()) == 0:
        return (
            hidden_states,
            torch.ones(count, dtype=torch.bool, device=hidden_states.device),
            torch.ones(count, dtype=torch.float32, device=hidden_states.device),
        )
    full = (
        hidden_states
        if pruned_hidden_states is None
        else torch.cat((hidden_states, pruned_hidden_states), dim=0)
    )
    valid = (assignment.labels >= 0).nonzero(as_tuple=False).flatten()
    cluster_positions = torch.searchsorted(centers, assignment.labels[valid])
    order = torch.argsort(cluster_positions, stable=True)
    members = valid[order]
    cluster_positions = cluster_positions[order]
    sizes = torch.bincount(
        cluster_positions, minlength=int(centers.numel())
    ).to(hidden_states.dtype)
    one_hot = functional.one_hot(
        cluster_positions, num_classes=int(centers.numel())
    ).to(hidden_states.dtype)
    means = one_hot.transpose(0, 1) @ full.index_select(0, members)
    means = means / sizes.unsqueeze(1)
    output = hidden_states.clone()
    output[centers] = means
    center_mask = torch.zeros(count, dtype=torch.bool, device=hidden_states.device)
    center_mask[centers] = True
    weights = torch.ones(count, dtype=torch.float32, device=hidden_states.device)
    weights[centers] = sizes.float()
    return output, center_mask, weights


def pact_cosine_pairwise_distance(features: Any) -> Any:
    functional = _functional()
    normalised = functional.normalize(features.float(), p=2.0, dim=-1)
    distances = (1.0 - normalised @ normalised.transpose(0, 1)).clamp(min=0.0)
    distances.fill_diagonal_(0.0)
    return distances


def pact_proportional_attention_bias(token_sizes: Any, *, dtype: Any) -> Any:
    """Return broadcastable log-mass bias for subsequent eager attention."""

    if bool((token_sizes <= 0).any().item()):
        raise ValueError("PACT token sizes must be positive.")
    return token_sizes.to(dtype=dtype).log().view(1, 1, 1, -1)


def pact_attention_mask(
    base_mask: Any | None,
    token_sizes: Any,
    *,
    query_length: int,
    reference: Any,
) -> Any:
    """Compose native causal visibility with PACT log cluster-mass bias."""

    torch = _torch()
    key_length = int(token_sizes.numel())
    bias = pact_proportional_attention_bias(
        token_sizes,
        dtype=reference.dtype,
    ).to(reference.device)
    if base_mask is None:
        query = torch.arange(query_length, device=reference.device)[:, None]
        key = torch.arange(key_length, device=reference.device)[None, :]
        blocked = key > (key_length - query_length + query)
        floor = torch.finfo(reference.dtype).min
        base_mask = torch.zeros(
            (1, 1, query_length, key_length),
            dtype=reference.dtype,
            device=reference.device,
        ).masked_fill(blocked.view(1, 1, query_length, key_length), floor)
    elif not isinstance(base_mask, torch.Tensor):
        raise BackendIntegrationError(
            "PACT requires a tensor causal attention mask."
        )
    elif base_mask.dtype == torch.bool:
        floor = torch.finfo(reference.dtype).min
        base_mask = torch.where(
            base_mask,
            torch.zeros((), dtype=reference.dtype, device=base_mask.device),
            torch.full((), floor, dtype=reference.dtype, device=base_mask.device),
        )
    if int(base_mask.shape[-1]) != key_length:
        raise InvariantError("PACT causal-mask key length does not match token sizes.")
    return base_mask.to(dtype=reference.dtype) + bias


def _index_tensor(value: Any, *, device: Any) -> Any:
    torch = _torch()
    if isinstance(value, torch.Tensor):
        result = value.to(device=device, dtype=torch.long)
    else:
        result = torch.tensor(tuple(value), dtype=torch.long, device=device)
    if result.ndim != 1:
        raise ValueError("image_indices must be one-dimensional.")
    return result


def _validate_indices(indices: Any, upper: int) -> None:
    values = [int(value) for value in indices.tolist()]
    if values != sorted(set(values)):
        raise ValueError("image_indices must be unique and source ordered.")
    if any(value < 0 or value >= upper for value in values):
        raise ValueError("image_indices contains an out-of-range index.")


def _torch() -> Any:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("PACT execution requires PyTorch.") from exc
    return torch


def _functional() -> Any:
    try:
        import torch.nn.functional as functional
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("PACT execution requires PyTorch.") from exc
    return functional


__all__ = [
    "PACTClusterAssignment",
    "PACTDensityPeakClusterer",
    "PACTResult",
    "PACTSpec",
    "pact_cosine_pairwise_distance",
    "pact_image_tokens",
    "pact_merge_clusters",
    "pact_proportional_attention_bias",
    "pact_score_keep_mask",
    "pact_token_scores",
]
