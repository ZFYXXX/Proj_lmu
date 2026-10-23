from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from mllm_comp.compressors.image.vispruner import VisPrunerSpec, vispruner_image_tokens


def test_vispruner_combines_attention_saliency_with_visual_diversity() -> None:
    count = 16
    output = torch.randn(count, 12)
    selection = torch.randn(count, 5)
    received = torch.linspace(0.0, 1.0, count)
    result = vispruner_image_tokens(
        output,
        selection,
        received,
        spec=VisPrunerSpec(target_count=8, important_ratio=0.5, removal_step=2),
        media_unit="image_feature",
    )

    assert result.tokens.shape == (8, 12)
    assert result.indices.numel() == 8
    assert result.layout_delta.source_count == count
    assert result.layout_delta.result_count == 8
    assert result.metadata["important_count"] == 4
    assert result.metadata["diverse_count"] == 4
