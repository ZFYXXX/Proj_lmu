from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from mllm_comp.compressors.image.aim import AIMSpec, aim_image_tokens


def test_aim_merges_to_fixed_budget_with_complete_source_mapping() -> None:
    tokens = torch.randn(16, 12)
    result = aim_image_tokens(
        tokens,
        spec=AIMSpec(retain_ratio=0.5, target_count=8),
        media_unit="image_feature",
    )

    assert result.tokens.shape == (8, 12)
    assert result.representative_indices.shape == (8,)
    assert result.layout_delta.source_count == 16
    assert result.layout_delta.result_count == 8
    assert result.layout_delta.operation == "merge"
    assert len(result.token_sizes) == 8
