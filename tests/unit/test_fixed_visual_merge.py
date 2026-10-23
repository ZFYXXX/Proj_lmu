from __future__ import annotations

import pytest

from mllm_comp.compressors.image.fixed_visual_merge import (
    FixedVisualMergeSpec,
    fixed_visual_merge,
)


@pytest.mark.parametrize("partition", ["grid", "feature_anchor"])
def test_fixed_visual_merge_is_exact_deterministic_and_exhaustive(partition: str) -> None:
    torch = pytest.importorskip("torch")
    features = torch.arange(80 * 6, dtype=torch.float32).reshape(80, 6)
    spec = FixedVisualMergeSpec(partition=partition, target_count=64)

    first = fixed_visual_merge(
        features, grid_height=8, grid_width=10, spec=spec
    )
    second = fixed_visual_merge(
        features, grid_height=8, grid_width=10, spec=spec
    )

    assert first.tokens.shape == (64, 6)
    assert torch.equal(first.tokens, second.tokens)
    assert first.metadata["partition_hash"] == second.metadata["partition_hash"]
    members = first.metadata["cluster_members"]
    assert len(members) == 64
    assert all(members)
    assert sorted(index for cluster in members for index in cluster) == list(range(80))
    assert tuple(first.indices.tolist()) == tuple(sorted(first.indices.tolist()))
    assert first.layout_delta.result_count == 64


def test_fixed_visual_merge_values_are_fp32_cluster_means() -> None:
    torch = pytest.importorskip("torch")
    features = torch.arange(4 * 3, dtype=torch.bfloat16).reshape(4, 3)
    result = fixed_visual_merge(
        features,
        grid_height=2,
        grid_width=2,
        spec=FixedVisualMergeSpec(partition="grid", target_count=2),
    )

    for row, members in enumerate(result.metadata["cluster_members"]):
        expected = features[list(members)].float().mean(dim=0).to(features.dtype)
        assert torch.equal(result.tokens[row], expected)


def test_fixed_visual_merge_rejects_below_budget_and_bad_grid() -> None:
    torch = pytest.importorskip("torch")
    features = torch.randn(8, 4)
    with pytest.raises(ValueError, match="below target_count"):
        fixed_visual_merge(
            features,
            grid_height=2,
            grid_width=4,
            spec=FixedVisualMergeSpec(target_count=9),
        )
    with pytest.raises(ValueError, match="does not match"):
        fixed_visual_merge(features, grid_height=3, grid_width=3)
