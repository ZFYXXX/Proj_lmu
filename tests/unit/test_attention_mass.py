from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from mllm_comp.compressors.image.attention_mass import add_visual_log_mass_bias


def test_visual_log_mass_is_added_only_to_prefill_keys() -> None:
    mask = torch.zeros(1, 1, 2, 5, dtype=torch.float32)
    mass = torch.tensor([0.0, 1.0, 2.0])
    result = add_visual_log_mass_bias(mask, mass, key_length=5)
    expected = torch.tensor([0.0, 1.0, 2.0, 0.0, 0.0])
    torch.testing.assert_close(result[0, 0, 0], expected)
    torch.testing.assert_close(result[0, 0, 1], expected)


def test_visual_log_mass_rejects_invalid_mask_and_lengths() -> None:
    mass = torch.ones(3)
    with pytest.raises(ValueError, match="rank-4"):
        add_visual_log_mass_bias(torch.zeros(1, 5), mass, key_length=5)
    with pytest.raises(ValueError, match="shorter"):
        add_visual_log_mass_bias(torch.zeros(1, 1, 1, 2), mass, key_length=2)
