from __future__ import annotations

import math

import pytest

torch = pytest.importorskip("torch")

from mllm_comp.compressors.image.btp import (  # noqa: E402
    BalancedTokenPruningSpec,
    balanced_token_pruning_image_tokens,
    btp_maxmin_diversity_fill,
    btp_rebalanced_attention_topk,
)


def test_published_qwen25_vl_recipe_and_native_boundary_mapping() -> None:
    spec = BalancedTokenPruningSpec()

    assert spec.prune_before_layers == (4, 7, 16, 23)
    assert spec.boundary_after_layer_indices == (2, 5, 14, 21)
    assert spec.retain_ratios == (0.5, 0.5, 0.5, 0.0)
    assert spec.attention_fractions == (0.1, 0.4, 1.0, 0.0)
    assert spec.attention_oversample == (1.2, 1.0, 1.2, 1.0)
    assert spec.rebalance_stages == (False, False, True, False)
    spec.validate_decoder_depth(23)


@pytest.mark.parametrize(
    ("kwargs", "error"),
    (
        ({"retain_ratios": (0.5,)}, "exactly 4"),
        ({"retain_ratios": (0.5, 0.5, 0.5, 0.5)}, "FinalWipe"),
        ({"retain_ratios": (0.5, 0.0, 0.5, 0.0)}, "exactly one"),
        ({"retain_ratios": (0.5, 0.5, 0.25, 0.0)}, "retain_ratio=0.5"),
        ({"attention_fractions": (True, 0.4, 1.0, 0.0)}, "real number"),
        ({"attention_oversample": (1.2, 1.0, math.nan, 1.0)}, "finite"),
        ({"position_policy": "contiguous"}, "preserve_original"),
    ),
)
def test_spec_rejects_invalid_stage_contracts(
    kwargs: dict[str, object],
    error: str | None,
) -> None:
    if error is None:
        BalancedTokenPruningSpec(**kwargs)
    else:
        with pytest.raises((TypeError, ValueError), match=error):
            BalancedTokenPruningSpec(**kwargs)


def test_rebalanced_attention_prefers_ranked_early_half_then_later_half() -> None:
    scores = torch.tensor([7.0, 1.0, 0.0, 8.0, 10.0, 9.0])

    selected = btp_rebalanced_attention_topk(
        scores,
        target_count=3,
        oversample=1.5,
    )

    assert selected.tolist() == [0, 4, 5]


def test_maxmin_preserves_attention_seed_and_matches_release_initialization() -> None:
    features = torch.tensor(
        [
            [1.0, 0.0],
            [0.99, 0.01],
            [0.0, 1.0],
            [-1.0, 0.0],
        ]
    )

    seeded = btp_maxmin_diversity_fill(
        features,
        target_count=2,
        seed_indices=torch.tensor([0]),
    )
    unseeded = btp_maxmin_diversity_fill(features, target_count=2)

    assert seeded.tolist() == [0, 3]
    assert unseeded.tolist() == [3, 0]


def test_hybrid_stage_uses_current_span_budget_and_preserves_source_order() -> None:
    hidden = torch.tensor(
        [
            [9.0, 9.0],
            [8.0, 8.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [-1.0, 0.0],
            [0.0, -1.0],
            [7.0, 7.0],
            [6.0, 6.0],
        ]
    )
    spec = BalancedTokenPruningSpec(
        prune_before_layers=(2, 3),
        retain_ratios=(0.5, 0.0),
        attention_fractions=(0.5, 0.0),
        attention_oversample=(1.0, 1.0),
        rebalance_stages=(False, False),
    )

    result = balanced_token_pruning_image_tokens(
        hidden,
        image_indices=torch.arange(2, 6),
        boundary_after_layer=0,
        image_attention_scores=torch.tensor([0.1, 0.9, 0.2, 0.8]),
        spec=spec,
        source_layout={"stage": 0},
        result_layout={"position_policy": "preserve_original"},
    )

    assert result.selected_visual_indices.tolist() == [1, 3]
    assert result.kept_image_indices.tolist() == [3, 5]
    assert result.kept_indices.tolist() == [0, 1, 3, 5, 6, 7]
    assert result.layout_delta.dropped_indices == (2, 4)
    assert result.metadata["attention_seed_count"] == 1
    assert result.metadata["diversity_fill_count"] == 1
    assert result.layout_delta.result_layout["position_policy"] == "preserve_original"


def test_finalwipe_removes_all_image_tokens_without_attention() -> None:
    hidden = torch.arange(14, dtype=torch.float32).reshape(7, 2)
    spec = BalancedTokenPruningSpec(
        prune_before_layers=(2,),
        retain_ratios=(0.0,),
        attention_fractions=(0.0,),
        attention_oversample=(1.0,),
        rebalance_stages=(False,),
    )

    result = balanced_token_pruning_image_tokens(
        hidden,
        image_indices=torch.arange(2, 5),
        boundary_after_layer=0,
        image_attention_scores=None,
        spec=spec,
    )

    assert result.metadata["stage_kind"] == "final_wipe"
    assert result.kept_image_indices.numel() == 0
    assert result.kept_indices.tolist() == [0, 1, 5, 6]
    assert result.layout_delta.protected_indices == (0, 1, 5, 6)
    assert result.layout_delta.old_to_new[2] is None
    assert result.hidden_states.tolist() == hidden[[0, 1, 5, 6]].tolist()


def test_default_small_span_survives_positive_stages_until_finalwipe() -> None:
    hidden = torch.arange(14, dtype=torch.float32).reshape(7, 2)
    image_indices = torch.arange(2, 6)
    spec = BalancedTokenPruningSpec()
    image_counts: list[int] = []
    sequence_counts: list[int] = []
    stage_kinds: list[str] = []

    for boundary in spec.boundary_after_layer_indices:
        result = balanced_token_pruning_image_tokens(
            hidden,
            image_indices=image_indices,
            boundary_after_layer=boundary,
            image_attention_scores=torch.arange(
                int(image_indices.numel()),
                dtype=torch.float32,
            ),
            spec=spec,
        )
        hidden = result.hidden_states
        image_indices = torch.arange(
            2,
            2 + int(result.kept_image_indices.numel()),
        )
        image_counts.append(int(result.kept_image_indices.numel()))
        sequence_counts.append(int(result.layout_delta.result_count))
        stage_kinds.append(str(result.metadata["stage_kind"]))

    assert image_counts == [2, 1, 1, 0]
    assert sequence_counts == [5, 4, 4, 3]
    assert stage_kinds == [
        "attention_seeded_diversity",
        "attention_seeded_diversity",
        "identity",
        "final_wipe",
    ]
