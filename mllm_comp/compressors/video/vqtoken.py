"""Canonical fixed-K VQToken video clustering.

The pinned recipe clusters every projected token from one video in a single
cosine K-means problem and returns the empirical means of the L2-normalized
members.  Adaptive K and the cross-attention expansion path are intentionally
outside the canonical profile.

The reference samples K-means++ centers randomly and can emit empty clusters.
That is incompatible with an exact, replayable :class:`LayoutDelta`.  This
port uses source-order-stable farthest-first initialization and deterministically
reassigns the least-similar member of an overfull cluster whenever a cluster is
empty.  The defining fixed-K/global/normalized-mean recipe is unchanged while
every output owns one non-empty source group and one explicit representative.
"""

from __future__ import annotations

from typing import Any, Mapping

from mllm_comp.core.errors import BackendIntegrationError, ConfigError, InvariantError

from .config import VIDEO_REFERENCE_REVISION, resolve_canonical_video_config
from .contracts import VideoCompressionResult, VideoTokenPayload
from . import _rules as rule_contract


_ALGORITHM = "vqtoken"
_VARIANT = "vqtoken.video"
_MAX_ITERATIONS = 50
_TOLERANCE = 1e-4


class VQTokenVideoKernel:
    """Registry kernel for the single frozen ``vqtoken.video`` profile."""

    name = _ALGORITHM
    version = f"videollm-comp-{VIDEO_REFERENCE_REVISION[:8]}-vqtoken-v1"

    def validate_config(self, plugin_config: Mapping[str, Any]) -> None:
        if not isinstance(plugin_config, Mapping):
            raise ConfigError("video plugin config must be a mapping")
        contract = plugin_config.get("modality_contract", {})
        if contract:
            if not isinstance(contract, Mapping):
                raise ConfigError("modality_contract must be a mapping")
            if contract.get("algorithm") != self.name:
                raise ConfigError("video plugin algorithm must be 'vqtoken'")
            if contract.get("variant") != _VARIANT:
                raise ConfigError("video plugin variant must be 'vqtoken.video'")
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
        return vqtoken_video(payload, config=config, rules=rules)

    def replay(
        self, features: Any, result: VideoCompressionResult, *,
        config: Mapping[str, Any], rules: Mapping[str, Any],
    ) -> Any:
        """Apply the saved normalized groups to another visual feature branch."""
        return merge_vqtoken_source_groups(features, result, rules=rules)


VQTokenKernel = VQTokenVideoKernel


def vqtoken_video(
    payload: VideoTokenPayload,
    *,
    config: Mapping[str, Any] | None = None,
    rules: Mapping[str, Any] | None = None,
) -> VideoCompressionResult:
    """Cluster every video's complete spatiotemporal token stream into K=32.

    Assignments are selected without gradients.  Output features are rebuilt
    from the differentiable L2-normalized source tokens, so the canonical
    training path still propagates gradients into the visual projector and any
    intended backbone/adapter parameters.
    """

    torch = _torch()
    if not isinstance(payload, VideoTokenPayload):
        raise TypeError("payload must be a VideoTokenPayload")
    if not isinstance(payload.features, torch.Tensor):
        raise TypeError("VQToken payload features must be a torch.Tensor")
    if payload.features.ndim != 2 or not payload.features.is_floating_point():
        raise ValueError(
            "VQToken payload features must have floating shape [tokens, hidden]"
        )
    if not bool(torch.isfinite(payload.features).all().item()):
        raise ValueError("VQToken payload features must be finite")
    if payload.token_mask is not None:
        mask = torch.as_tensor(payload.token_mask, device=payload.features.device)
        if mask.ndim != 1 or int(mask.numel()) != payload.token_count:
            raise ValueError("VQToken token_mask must match the projected layout")
        if not bool(mask.bool().all().item()):
            raise BackendIntegrationError(
                "VQToken requires provider-trimmed videos without padded tokens"
            )

    resolved = resolve_canonical_video_config(_ALGORITHM, config or {})
    cluster_count = int(resolved["num_clusters"])
    if resolved["adaptive"] or resolved["use_cross_attention"]:
        raise ConfigError("Canonical VQToken requires fixed K without cross-attention")

    outputs: list[Any] = []
    groups_by_video: list[tuple[tuple[int, ...], ...]] = []
    representatives_by_video: list[tuple[int, ...]] = []
    iterations_by_video: list[int] = []
    anchor_legalizations_by_video: list[int] = []

    for video_index in range(payload.video_count):
        span = payload.video_span(video_index)
        source = payload.features[span]
        if int(source.shape[0]) <= cluster_count:
            raise BackendIntegrationError(
                "Canonical VQToken requires more than 32 source tokens so fixed-K "
                "clustering is a non-identity reduction"
            )
        normalized = _transform_values(source, rules=rules, torch=torch)
        if rules and "grouper" in rules:
            custom_groups = rule_contract.groups(rules["grouper"],
                {"values": normalized.detach(), "group_count": cluster_count},
                count=cluster_count, candidates=range(int(source.shape[0])), name="grouper")
            labels = torch.empty(int(source.shape[0]), dtype=torch.long, device=source.device)
            for group_index, group in enumerate(custom_groups):
                labels[list(group)] = group_index
            iterations = 0
        else:
            labels, iterations = _stable_cosine_kmeans(
                normalized.detach().float(),
                cluster_count=cluster_count,
                max_iterations=_MAX_ITERATIONS,
                tolerance=_TOLERANCE,
                torch=torch,
            )
        with torch.no_grad():
            records = _cluster_records(
                normalized,
                labels,
                cluster_count=cluster_count,
                torch=torch,
            )
            records.sort(key=lambda record: record[0][0])
            groups = tuple(record[0] for record in records)
            preferred_representatives = tuple(record[1] for record in records)
            representatives, legalization_count = _causal_representatives(
                normalized,
                groups,
                preferred=preferred_representatives,
                torch=torch,
            )
        compressed = torch.stack(
            [
                normalized.index_select(
                    0,
                    torch.tensor(group, dtype=torch.long, device=source.device),
                ).mean(dim=0)
                for group in groups
            ],
            dim=0,
        )
        if rules and "aggregator" in rules:
            compressed = _aggregate_values(normalized, groups, rules["aggregator"], torch=torch)
        outputs.append(compressed)
        groups_by_video.append(groups)
        representatives_by_video.append(representatives)
        iterations_by_video.append(iterations)
        anchor_legalizations_by_video.append(legalization_count)

    return VideoCompressionResult.from_source_groups(
        payload,
        torch.cat(outputs, dim=0),
        source_groups=groups_by_video,
        representative_indices=representatives_by_video,
        exact_budgets=tuple(cluster_count for _ in range(payload.video_count)),
        metadata={
            "algorithm": _VARIANT,
            "reference_revision": VIDEO_REFERENCE_REVISION,
            "canonical_config": dict(resolved),
            "clustering_scope": "global_per_video",
            "distance": "custom" if rules and ('transform' in rules or 'grouper' in rules) else "cosine_on_l2_normalized_tokens",
            "output_feature_rule": "custom" if rules and ('transform' in rules or 'aggregator' in rules) else "mean_l2_normalized_members",
            "representative_policy": (
                "closest_member_to_normalized_cluster_mean_before_next_group"
            ),
            "anchor_legalizations_by_video": tuple(anchor_legalizations_by_video),
            "initialization": "custom" if rules and ('grouper' in rules) else "deterministic_farthest_first_source_order",
            "empty_cluster_policy": "custom" if rules and ('grouper' in rules) else "least_similar_overfull_donor_source_order",
            "max_iterations": _MAX_ITERATIONS,
            "tolerance": _TOLERANCE,
            "iterations_by_video": tuple(iterations_by_video),
            "output_order": "source_group_first_index",
        },
    )


def merge_vqtoken_source_groups(
    features: Any,
    result: VideoCompressionResult,
    *,
    video_index: int = 0,
    rules: Mapping[str, Any] | None = None,
) -> Any:
    """Replay one VQToken assignment on a synchronized Qwen3 DeepStack branch."""

    torch = _torch()
    if not isinstance(features, torch.Tensor):
        raise TypeError("synchronized VQToken features must be a torch.Tensor")
    if features.ndim != 2 or not features.is_floating_point():
        raise ValueError(
            "synchronized VQToken features must be floating [tokens, hidden]"
        )
    if not bool(torch.isfinite(features).all().item()):
        raise ValueError("synchronized VQToken features must be finite")
    if not isinstance(result, VideoCompressionResult):
        raise TypeError("result must be a VideoCompressionResult")
    source_span = result.source_video_span(video_index)
    source_count = source_span.stop - source_span.start
    if int(features.shape[0]) != source_count:
        raise InvariantError(
            "synchronized VQToken branch does not match the source video layout"
        )
    normalized = _transform_values(features, rules=rules, torch=torch)
    if rules and "aggregator" in rules:
        return _aggregate_values(normalized, result.source_groups[video_index], rules["aggregator"], torch=torch)
    rows = []
    for group in result.source_groups[video_index]:
        index = torch.tensor(group, dtype=torch.long, device=features.device)
        rows.append(normalized.index_select(0, index).mean(dim=0))
    return torch.stack(rows, dim=0)


def _transform_values(features: Any, *, rules: Any, torch: Any) -> Any:
    if rules and "transform" in rules:
        return rule_contract.value(rules["transform"],
            {"values": features, "default_transform": lambda: _l2_normalize_tokens(features, torch=torch)},
            reference=features, shape=tuple(features.shape), name="transform")
    return _l2_normalize_tokens(features, torch=torch)


def _aggregate_values(features: Any, groups: Any, rule: Any, *, torch: Any) -> Any:
    return torch.stack(tuple(rule_contract.value(rule,
        {"values": features, "group": group}, reference=features,
        shape=(int(features.shape[-1]),), name="aggregator") for group in groups), dim=0)


def _stable_cosine_kmeans(
    features: Any,
    *,
    cluster_count: int,
    max_iterations: int,
    tolerance: float,
    torch: Any,
) -> tuple[Any, int]:
    count, hidden = (int(value) for value in features.shape)
    if not 1 < cluster_count < count:
        raise ValueError("VQToken cluster_count must be between one and token count")
    if hidden < 1:
        raise ValueError("VQToken hidden size must be positive")
    center_indices = _farthest_first_indices(
        features,
        cluster_count=cluster_count,
        torch=torch,
    )
    centroids = features.index_select(0, center_indices).clone()
    labels = None
    iterations = 0
    with torch.no_grad():
        for iterations in range(1, max_iterations + 1):
            centroid_norms = torch.linalg.vector_norm(centroids, dim=-1, keepdim=True)
            if bool((centroid_norms <= 0).any().item()):
                raise InvariantError("VQToken K-means produced a zero centroid")
            similarities = features @ (centroids / centroid_norms).transpose(0, 1)
            labels = similarities.argmax(dim=-1)
            labels = _repair_empty_clusters(
                labels,
                similarities,
                cluster_count=cluster_count,
                torch=torch,
            )
            new_centroids = torch.zeros(
                cluster_count,
                hidden,
                dtype=features.dtype,
                device=features.device,
            )
            new_centroids.scatter_add_(
                0,
                labels.unsqueeze(1).expand(-1, hidden),
                features,
            )
            counts = torch.bincount(labels, minlength=cluster_count).to(
                dtype=features.dtype
            )
            if bool((counts <= 0).any().item()):
                raise InvariantError("VQToken empty-cluster repair was incomplete")
            new_centroids = new_centroids / counts.unsqueeze(1)
            shift = torch.linalg.vector_norm(centroids - new_centroids)
            centroids = new_centroids
            if float(shift.item()) < tolerance:
                break
    if labels is None:
        raise InvariantError("VQToken K-means did not execute")
    return labels, iterations


def _farthest_first_indices(
    features: Any,
    *,
    cluster_count: int,
    torch: Any,
) -> Any:
    selected = [0]
    similarities = features @ features[0]
    minimum_distances = 1.0 - similarities
    for _ in range(1, cluster_count):
        available = minimum_distances.clone()
        available[torch.tensor(selected, dtype=torch.long, device=features.device)] = (
            float("-inf")
        )
        next_index = int(available.argmax().item())
        selected.append(next_index)
        candidate_distances = 1.0 - (features @ features[next_index])
        minimum_distances = torch.minimum(minimum_distances, candidate_distances)
    return torch.tensor(selected, dtype=torch.long, device=features.device)


def _repair_empty_clusters(
    labels: Any,
    similarities: Any,
    *,
    cluster_count: int,
    torch: Any,
) -> Any:
    labels = labels.clone()
    counts = torch.bincount(labels, minlength=cluster_count)
    empty = (counts == 0).nonzero(as_tuple=False).flatten().tolist()
    if not empty:
        return labels
    source_indices = torch.arange(labels.numel(), device=labels.device)
    assigned = similarities[source_indices, labels]
    donor_order = torch.argsort(assigned, descending=False, stable=True).tolist()
    reserved: set[int] = set()
    for cluster in empty:
        donor = next(
            (
                index
                for index in donor_order
                if index not in reserved and int(counts[int(labels[index])].item()) > 1
            ),
            None,
        )
        if donor is None:
            raise InvariantError("VQToken could not repair an empty cluster")
        old_cluster = int(labels[donor].item())
        labels[donor] = cluster
        counts[old_cluster] -= 1
        counts[cluster] += 1
        reserved.add(donor)
    return labels


def _cluster_records(
    normalized: Any,
    labels: Any,
    *,
    cluster_count: int,
    torch: Any,
) -> list[tuple[tuple[int, ...], int]]:
    records = []
    for cluster in range(cluster_count):
        members = (labels == cluster).nonzero(as_tuple=False).flatten()
        if int(members.numel()) < 1:
            raise InvariantError("VQToken must emit one non-empty group per cluster")
        group = tuple(int(value) for value in members.tolist())
        member_features = normalized.index_select(0, members).float()
        center = member_features.mean(dim=0, keepdim=True)
        scores = torch.nn.functional.cosine_similarity(
            member_features,
            center,
            dim=-1,
        )
        representative = group[int(scores.argmax().item())]
        records.append((group, representative))
    return records


def _causal_representatives(
    normalized: Any,
    groups: tuple[tuple[int, ...], ...],
    *,
    preferred: tuple[int, ...],
    torch: Any,
) -> tuple[tuple[int, ...], int]:
    representatives = []
    legalized = 0
    for index, group in enumerate(groups):
        upper = groups[index + 1][0] if index + 1 < len(groups) else None
        eligible = tuple(value for value in group if upper is None or value < upper)
        if not eligible:
            raise InvariantError("VQToken has no causal representative for a group")
        member_index = torch.tensor(
            group,
            dtype=torch.long,
            device=normalized.device,
        )
        eligible_index = torch.tensor(
            eligible,
            dtype=torch.long,
            device=normalized.device,
        )
        center = (
            normalized.index_select(0, member_index)
            .float()
            .mean(
                dim=0,
                keepdim=True,
            )
        )
        scores = torch.nn.functional.cosine_similarity(
            normalized.index_select(0, eligible_index).float(),
            center,
            dim=-1,
        )
        representative = eligible[int(scores.argmax().item())]
        representatives.append(representative)
        legalized += int(representative != preferred[index])
    result = tuple(representatives)
    if result != tuple(sorted(result)) or len(set(result)) != len(result):
        raise InvariantError("VQToken representatives do not preserve causal order")
    return result, legalized


def _l2_normalize_tokens(features: Any, *, torch: Any) -> Any:
    norms = torch.linalg.vector_norm(features, dim=-1, keepdim=True)
    if not bool(torch.isfinite(norms).all().item()) or bool((norms <= 0).any().item()):
        raise BackendIntegrationError(
            "VQToken requires finite non-zero projected token vectors"
        )
    return features / norms


def _torch() -> Any:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - dependency-light control plane
        raise BackendIntegrationError(
            "VQToken runtime requires the existing PyTorch environment"
        ) from exc
    return torch


__all__ = [
    "VQTokenKernel",
    "VQTokenVideoKernel",
    "merge_vqtoken_source_groups",
    "vqtoken_video",
]
