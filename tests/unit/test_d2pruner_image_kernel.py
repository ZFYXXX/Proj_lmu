from __future__ import annotations

import pytest


torch = pytest.importorskip("torch")

from mllm_comp.compressors.image.d2pruner import (  # noqa: E402
    D2PrunerSpec,
    d2pruner_image_tokens,
    load_d2pruner_bias_prior,
)


def test_d2pruner_rejects_non_finite_numeric_config() -> None:
    with pytest.raises(ValueError, match="finite"):
        D2PrunerSpec(epsilon=float("nan"))


def test_d2pruner_exact_budget_preserves_text_and_structural_tokens() -> None:
    torch.manual_seed(5)
    hidden = torch.randn(8, 6)
    result = d2pruner_image_tokens(
        hidden,
        image_indices=torch.arange(1, 7),
        image_attention_scores=torch.tensor([0.9, 0.8, 0.7, 0.6, 0.5, 0.4]),
        bias_prior=torch.ones(16),
        spec=D2PrunerSpec(retain_ratio=0.5, pivot_ratio=0.5),
        protected_indices=(6,),
        source_layout={"backbone": "fixture"},
        result_layout={"position_policy": "sparse_source"},
    )

    assert result.kept_indices[0].item() == 0
    assert result.kept_indices[-1].item() == 7
    assert 6 in result.kept_indices.tolist()
    assert int(result.kept_image_indices.numel()) == 3
    assert result.layout_delta.source_count == 8
    assert result.layout_delta.result_count == 5
    assert result.adjacency_edges >= 0


def test_d2pruner_debias_and_graph_mis_change_the_selected_tokens() -> None:
    hidden = torch.tensor(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [-1.0, 0.0],
            [0.0, -1.0],
            [1.0, 1.0],
            [-1.0, -1.0],
            [2.0, 2.0],
        ]
    )
    result = d2pruner_image_tokens(
        hidden,
        image_indices=torch.tensor([1, 2, 3, 4, 5, 6]),
        image_attention_scores=torch.tensor([10.0, 6.0, 5.0, 4.0, 3.0, 2.0]),
        bias_prior=torch.tensor([10.0, 1.0, 1.0, 1.0, 1.0, 1.0]),
        spec=D2PrunerSpec(
            retain_ratio=1.0 / 3.0,
            pivot_ratio=0.5,
            semantic_weight=0.0,
            similarity_threshold=0.99,
            epsilon=1e-6,
        ),
    )

    assert torch.allclose(
        result.relative_scores,
        torch.tensor([1.0, 6.0, 5.0, 4.0, 3.0, 2.0]),
        atol=1e-5,
    )
    # Local token 1 is the debiased pivot. Local token 2 has the next score
    # but is its spatial neighbour, so graph MIS takes lower-scored local 4.
    assert result.kept_image_indices.tolist() == [2, 5]
    assert result.kept_indices.tolist() == [0, 2, 5, 7]
    assert result.adjacency_edges == 22


def test_d2pruner_prior_loader_requires_real_calibration_asset(tmp_path) -> None:
    prior_path = tmp_path / "prior.pt"
    torch.save(
        {
            "bias": torch.ones(8),
            "sample_count": 1000,
            "model_path": "/checkpoints/fixture-model",
            "scorer_layer": 1,
        },
        prior_path,
    )
    prior, metadata = load_d2pruner_bias_prior(
        prior_path,
        expected_model_id="/checkpoints/fixture-model",
        expected_scorer_layer=1,
    )

    assert prior.shape == (8,)
    assert metadata["sample_count"] == 1000
    assert metadata["validated_model_identity"] == "/checkpoints/fixture-model"
    with pytest.raises(ValueError, match="checkpoint mismatch"):
        load_d2pruner_bias_prior(
            prior_path,
            expected_model_id="/another-root/fixture-model",
            expected_scorer_layer=1,
        )
    with pytest.raises(ValueError, match="scorer-layer mismatch"):
        load_d2pruner_bias_prior(
            prior_path,
            expected_model_id="/checkpoints/fixture-model",
            expected_scorer_layer=2,
        )
    with pytest.raises(FileNotFoundError):
        load_d2pruner_bias_prior(tmp_path / "missing.pt")
