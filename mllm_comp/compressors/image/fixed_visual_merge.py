"""Deterministic fixed-budget post-merger visual-token aggregation.

This module turns the two Stage-1 diagnostic partitions into deployable
projector-level kernels.  It deliberately reuses the exact partition
implementations used by ``visual_merge_attribution`` so the quality run tests
the diagnosed proxy rather than a subtly different clustering rule.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Mapping

from mllm_comp.analysis.common.visual_merge_attribution import (
    feature_anchor_partition,
    grid_partition,
)
from mllm_comp.core.layout import LayoutDelta, LayoutTokenUnit, OrderingPolicy


@dataclass(frozen=True)
class FixedVisualMergeSpec:
    """Configuration for a deterministic exact-count visual merge."""

    partition: str = "feature_anchor"
    target_count: int = 64

    def __post_init__(self) -> None:
        if self.partition not in {"feature_anchor", "grid"}:
            raise ValueError("partition must be 'feature_anchor' or 'grid'")
        if isinstance(self.target_count, bool) or self.target_count < 1:
            raise ValueError("target_count must be a positive integer")


@dataclass(frozen=True)
class FixedVisualMergeOutput:
    tokens: Any
    indices: Any
    layout_delta: LayoutDelta
    metadata: Mapping[str, Any]

    @property
    def kept_indices(self) -> Any:
        return self.indices


def fixed_visual_merge(
    tokens: Any,
    *,
    grid_height: int,
    grid_width: int,
    spec: FixedVisualMergeSpec = FixedVisualMergeSpec(),
    media_unit: str | LayoutTokenUnit = "image_feature",
    source_layout: Mapping[str, Any] | None = None,
    result_layout: Mapping[str, Any] | None = None,
    projector: Any | None = None,
) -> FixedVisualMergeOutput:
    """Mean-aggregate one rectangular post-merger feature grid exactly to K.

    Cluster values are accumulated in FP32.  Output clusters are serialized by
    the source index closest to their rounded two-dimensional centroid, which
    gives the prompt legalizer a deterministic source placeholder for every
    result token.  The provider later replaces that placeholder's native
    M-RoPE position with the exact rounded three-axis cluster centroid.
    """

    import torch

    if not isinstance(tokens, torch.Tensor) or tokens.ndim != 2:
        raise ValueError("tokens must have shape [tokens, hidden]")
    if not tokens.is_floating_point():
        raise ValueError("tokens must be floating point")
    source_count = int(tokens.shape[0])
    if grid_height * grid_width != source_count:
        raise ValueError(
            f"feature grid {grid_height}x{grid_width} does not match "
            f"{source_count} source tokens"
        )
    if source_count < spec.target_count:
        raise ValueError(
            f"source token count {source_count} is below target_count "
            f"{spec.target_count}"
        )

    partition = (
        grid_partition(grid_height, grid_width, spec.target_count)
        if spec.partition == "grid"
        else feature_anchor_partition(
            tokens,
            height=grid_height,
            width=grid_width,
            target_tokens=spec.target_count,
        )
    )

    records: list[tuple[int, tuple[int, ...], tuple[int, int], Any]] = []
    for members in partition.members:
        member_tensor = torch.tensor(members, dtype=torch.long, device=tokens.device)
        rows = member_tensor.div(grid_width, rounding_mode="floor")
        cols = member_tensor.remainder(grid_width)
        centroid_row = int(torch.round(rows.float().mean()).item())
        centroid_col = int(torch.round(cols.float().mean()).item())
        squared_distance = (rows - centroid_row).square() + (cols - centroid_col).square()
        nearest_distance = squared_distance.min()
        nearest = member_tensor[squared_distance == nearest_distance]
        representative = int(nearest.min().item())
        member_values = tokens.index_select(0, member_tensor)
        value = (
            member_values.float().mean(dim=0).to(tokens.dtype)
            if projector is None
            else projector(
                member_values,
                rows=rows,
                cols=cols,
                grid_height=grid_height,
                grid_width=grid_width,
                source_count=source_count,
            )
        )
        records.append(
            (
                representative,
                tuple(int(index) for index in members),
                (centroid_row, centroid_col),
                value,
            )
        )

    # Prompt placeholders are emitted in source order.  Token order must match
    # the selected placeholder order even when farthest-first anchor IDs do not.
    records.sort(key=lambda item: item[0])
    representatives = tuple(item[0] for item in records)
    cluster_members = tuple(item[1] for item in records)
    centroids = tuple(item[2] for item in records)
    output = torch.stack([item[3] for item in records])
    mapping: dict[int, int] = {}
    for result_index, members in enumerate(cluster_members):
        for source_index in members:
            mapping[source_index] = result_index
    delta = LayoutDelta.from_mapping(
        source_count=source_count,
        old_to_new=mapping,
        result_count=spec.target_count,
        media_unit=media_unit,
        source_layout=source_layout,
        result_layout={
            **dict(result_layout or {}),
            "representative_indices": representatives,
            "cluster_members": cluster_members,
            "grid_centroids": centroids,
            "partition_hash": partition.partition_hash,
            "position_policy": "rounded_native_mrope_cluster_centroid",
        },
        ordering_policy=OrderingPolicy.EXPLICIT,
    )
    metadata = MappingProxyType(
        {
            "algorithm": f"{spec.partition}_mean",
            "partition": spec.partition,
            "partition_hash": partition.partition_hash,
            "source_count": source_count,
            "result_count": spec.target_count,
            "target_count": spec.target_count,
            "cluster_sizes": tuple(len(members) for members in cluster_members),
            "cluster_members": cluster_members,
            "grid_centroids": centroids,
            "representative_indices": representatives,
            "aggregation": (
                "fp32_mean"
                if projector is None
                else f"stage2_{projector.spec.variant}"
            ),
            "position_policy": "rounded_native_mrope_cluster_centroid",
            "token_order": "source_representative",
        }
    )
    return FixedVisualMergeOutput(
        tokens=output,
        indices=torch.tensor(representatives, dtype=torch.long, device=tokens.device),
        layout_delta=delta,
        metadata=metadata,
    )


__all__ = [
    "FixedVisualMergeOutput",
    "FixedVisualMergeSpec",
    "fixed_visual_merge",
]
