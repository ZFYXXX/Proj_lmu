from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from mllm_comp.compressors.image.otprune import OTPruneSpec, otprune_image


def test_otprune_selects_exact_source_ordered_budget_and_drop_layout() -> None:
    tokens = torch.arange(64 * 8, dtype=torch.float32).reshape(64, 8)
    result = otprune_image(tokens, spec=OTPruneSpec(retain_ratio=0.5, gamma=0.01))

    assert result.tokens.shape == (32, 8)
    assert result.indices.shape == (32,)
    assert result.indices.tolist() == sorted(result.indices.tolist())
    assert len(set(result.indices.tolist())) == 32
    assert result.layout_delta.source_count == 64
    assert result.layout_delta.result_count == 32
    assert result.layout_delta.operation == "drop"
    assert len(result.layout_delta.dropped_indices) == 32
    assert result.metadata["selection"] == "greedy_logdet_cholesky"


def test_otprune_is_identity_at_full_retention() -> None:
    tokens = torch.randn(9, 4)
    result = otprune_image(tokens, spec=OTPruneSpec(retain_ratio=1.0))

    assert torch.equal(result.tokens, tokens)
    assert result.indices.tolist() == list(range(9))
    assert result.layout_delta.result_count == 9
    assert result.layout_delta.dropped_indices == ()


@pytest.mark.parametrize(
    "kwargs, error",
    [
        ({"retain_ratio": 0.0}, ValueError),
        ({"target_count": 0}, ValueError),
        ({"gamma": -0.1}, ValueError),
    ],
)
def test_otprune_rejects_invalid_configuration(kwargs, error) -> None:
    with pytest.raises(error):
        OTPruneSpec(**kwargs)


def test_otprune_rejects_nonfinite_or_nonmatrix_features() -> None:
    with pytest.raises(ValueError, match="shape"):
        otprune_image(torch.ones(2, 3, 4))
    bad = torch.ones(4, 3)
    bad[0, 0] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        otprune_image(bad)
