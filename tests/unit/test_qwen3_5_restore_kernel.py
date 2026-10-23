from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from mllm_comp.compressors.image.restore import RestoreSpec, restore_image_tokens


def test_restore_pristine_plus_anchors_hits_fixed_budget_and_preserves_mass() -> None:
    count = 16
    output = torch.randn(count, 12)
    selection = torch.randn(count, 5)
    received = torch.linspace(0.0, 1.0, count)
    result = restore_image_tokens(
        output,
        selection,
        received,
        spec=RestoreSpec(target_count=8, min_visual_tokens=2),
        media_unit="image_feature",
    )

    assert result.tokens.shape == (8, 12)
    assert result.layout_delta.source_count == count
    assert result.layout_delta.result_count == 8
    assert result.metadata["pristine_count"] + result.metadata["anchor_count"] == 8
    assert result.metadata["mass_conserved"] is True
