from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from mllm_comp.compressors.image.gap import GAPSpec, gap_image_tokens


def test_gap_mean_received_topk_selects_fixed_budget_in_source_order() -> None:
    count = 8
    tokens = torch.arange(count * 3, dtype=torch.float32).reshape(count, 3)
    received = torch.arange(count, dtype=torch.float32)
    attention = received.unsqueeze(0).expand(count, -1)
    result = gap_image_tokens(
        tokens,
        attention,
        spec=GAPSpec(retain_ratio=0.5, target_count=4),
        media_unit="image_feature",
    )

    assert result.tokens.shape == (4, 3)
    assert result.indices.tolist() == [4, 5, 6, 7]
    assert result.layout_delta.source_count == 8
    assert result.layout_delta.result_count == 4
    assert result.layout_delta.result_layout["position_policy"] == "preserve_original"
    assert result.metadata["importance_source"] == "mean_received_attention"
