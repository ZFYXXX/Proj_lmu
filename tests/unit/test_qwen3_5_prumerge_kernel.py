from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from mllm_comp.compressors.image.prumerge import PruMergeSpec, prumerge_image_tokens


def test_prumerge_plus_fixed_budget_includes_one_global_residual() -> None:
    count = 16
    tokens = torch.randn(count, 8)
    received = torch.linspace(0.0, 1.0, count)
    attention = received.unsqueeze(0).expand(count, -1)
    keys = torch.eye(count)
    result = prumerge_image_tokens(
        tokens,
        attention,
        keys,
        spec=PruMergeSpec(
            variant="plus",
            adaptive=False,
            target_count=8,
            cluster_size=4,
        ),
        media_unit="image_feature",
    )

    assert result.tokens.shape == (8, 8)
    assert result.indices.shape == (8,)
    assert result.layout_delta.source_count == count
    assert result.layout_delta.result_count == 8
    assert result.metadata["center_count"] == 7
    assert result.metadata["global_count"] == 1
    assert set(result.layout_delta.old_to_new.values()) == set(range(8))
