from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from mllm_comp.compressors.image.pact import (
    PACTSpec,
    pact_image_tokens,
    pact_proportional_attention_bias,
    pact_score_keep_mask,
)


def test_pact_compacts_only_image_tokens_and_preserves_gradients() -> None:
    generator = torch.Generator().manual_seed(19)
    hidden = torch.randn(14, 8, generator=generator, requires_grad=True)
    query = torch.randn(2, 14, 4, generator=generator)
    keys = torch.randn(2, 14, 4, generator=generator)
    images = torch.arange(2, 12)
    result = pact_image_tokens(
        hidden,
        image_indices=images,
        query_states=query,
        key_states=keys,
        clustering_keys=hidden.detach(),
        spec=PACTSpec(
            target_keep_ratio=0.3,
            cutoff_search_steps=8,
            multiply_by_hidden_norm=True,
        ),
    )

    assert result.hidden_states.shape[0] < hidden.shape[0]
    assert result.kept_indices[:2].tolist() == [0, 1]
    assert result.kept_indices[-2:].tolist() == [12, 13]
    assert float(result.token_sizes.sum()) == pytest.approx(14.0)
    result.hidden_states.sum().backward()
    assert hidden.grad is not None
    assert float(hidden.grad.abs().sum()) > 0.0


def test_pact_stable_score_ties_and_log_mass_bias() -> None:
    mask = pact_score_keep_mask(torch.ones(5), 2)
    assert mask.tolist() == [True, True, False, False, False]
    bias = pact_proportional_attention_bias(
        torch.tensor([1.0, 2.0, 4.0]), dtype=torch.float32
    )
    assert bias.shape == (1, 1, 1, 3)
    assert bias.flatten().tolist() == pytest.approx([0.0, 0.693147, 1.386294])
