from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from mllm_comp.backbones.qwen3_5.vision_attention import (
    pool_premerger_features_to_merger_tokens,
)
from mllm_comp.compressors.image.mmtok import MMTokSpec, mmtok_image_tokens


def test_premerger_patch_features_pool_to_native_merger_tokens() -> None:
    patches = torch.arange(16 * 3, dtype=torch.float32).reshape(16, 3)
    grid = torch.tensor([[1, 4, 4]], dtype=torch.long)

    (pooled,) = pool_premerger_features_to_merger_tokens(
        patches, grid, spatial_merge_size=2
    )

    assert pooled.shape == (4, 3)
    torch.testing.assert_close(
        pooled,
        patches.reshape(4, 4, 3).mean(dim=1),
    )


def test_mmtok_selects_fixed_fraction_using_text_and_visual_features() -> None:
    projected = torch.randn(8, 12)
    vision = torch.randn(8, 5)
    text = torch.randn(3, 12)
    result = mmtok_image_tokens(
        projected,
        vision,
        text,
        spec=MMTokSpec(retain_ratio=0.5, alpha=0.5),
    )

    assert result.tokens.shape == (4, 12)
    assert result.indices.shape == (4,)
    assert result.layout_delta.source_count == 8
    assert result.layout_delta.result_count == 4
    assert result.metadata["coverage_objective"] == "text_vision_plus_alpha_vision_vision"
