from __future__ import annotations

import pytest


torch = pytest.importorskip("torch")

from mllm_comp.compressors.image.finepruner import (  # noqa: E402
    FinePrunerSpec,
    finepruner_arp_image_tokens,
    finepruner_attention_head_slice,
    finepruner_first_representatives,
    finepruner_iac_image_tokens,
    finepruner_lsh_codes,
    finepruner_select_unbiased_heads,
)


def test_finepruner_defaults_derive_exact_completed_layer_schedule() -> None:
    spec = FinePrunerSpec()
    assert spec.iac_after_layers == (7, 14)
    assert spec.iac_hash_bits == (48, 41)
    assert spec.arp_target_after_layers == (20, 21, 22)
    assert spec.arp_apply_after_layer == 22
    assert spec.resolve_final_visual_count(9) == 5
    assert spec.resolve_final_visual_count(3) == 1
    spec.validate_decoder_depth(24)
    with pytest.raises(ValueError, match="decoder depth"):
        spec.validate_decoder_depth(21)
    with pytest.raises(TypeError, match="boolean"):
        FinePrunerSpec(head_ratio=True)
    with pytest.raises(ValueError, match="sorted and unique"):
        FinePrunerSpec(arp_layer_offsets=(0, -1, 1))


def test_lsh_is_deterministic_without_mutating_global_rng() -> None:
    hidden = torch.arange(30, dtype=torch.float32).view(6, 5)
    state = torch.random.get_rng_state().clone()
    first = finepruner_lsh_codes(hidden, hash_bits=8, random_seed=126)
    second = finepruner_lsh_codes(hidden, hash_bits=8, random_seed=126)
    assert torch.equal(first, second)
    assert torch.equal(state, torch.random.get_rng_state())
    assert finepruner_first_representatives(
        torch.tensor([7, 3, 7, 9, 3, 9])
    ).tolist() == [0, 1, 3]


def test_iac_has_exact_source_mapping_and_budget_guard() -> None:
    hidden = torch.tensor(
        [[10.0], [-3.0], [-2.0], [-1.0], [1.0], [2.0], [3.0], [20.0]]
    )
    spec = FinePrunerSpec(
        iac_after_layers=(1,),
        iac_hash_bits=(1,),
        arp_center_after_layer=3,
        arp_layer_offsets=(0,),
        arp_retain_ratio=0.9,
        random_seed=126,
    )
    result = finepruner_iac_image_tokens(
        hidden,
        image_indices=torch.arange(1, 7),
        completed_layers=1,
        initial_visual_tokens=6,
        spec=spec,
        source_layout={"stage": 0},
        result_layout={"position_policy": "contiguous"},
    )
    assert result.kept_image_indices.tolist() == [1, 4]
    assert result.dropped_indices.tolist() == [2, 3, 5, 6]
    assert result.kept_indices.tolist() == [0, 1, 4, 7]
    assert tuple(result.layout_delta.old_to_new.values()) == (
        0,
        1,
        None,
        None,
        2,
        None,
        None,
        3,
    )
    assert result.layout_delta.merge_groups == ()
    assert result.metadata["budget_guarded"] is False

    guarded = finepruner_iac_image_tokens(
        hidden,
        image_indices=torch.arange(1, 7),
        completed_layers=1,
        initial_visual_tokens=6,
        spec=FinePrunerSpec(
            iac_after_layers=(1,),
            iac_hash_bits=(1,),
            arp_center_after_layer=3,
            arp_layer_offsets=(0,),
            target_visual_tokens=5,
        ),
    )
    assert guarded.metadata["budget_guarded"] is True
    assert guarded.kept_indices.tolist() == list(range(8))


def test_arp_averages_layers_selects_low_kl_heads_and_restores_source_order() -> None:
    hidden = torch.arange(8, dtype=torch.float32).view(8, 1)
    layer_a = torch.tensor(
        [
            [0.1, 0.9, 0.2, 0.8],
            [8.0, 0.0, 0.0, 0.0],
        ]
    )
    layer_b = torch.tensor(
        [
            [0.2, 0.8, 0.1, 0.7],
            [8.0, 0.0, 0.0, 0.0],
        ]
    )
    spec = FinePrunerSpec(
        iac_after_layers=(1,),
        iac_hash_bits=(3,),
        arp_center_after_layer=3,
        arp_layer_offsets=(-1, 0),
        arp_retain_ratio=0.5,
        head_ratio=0.5,
    )
    result = finepruner_arp_image_tokens(
        hidden,
        [layer_a, layer_b],
        image_indices=torch.arange(2, 6),
        completed_layers=3,
        initial_visual_tokens=8,
        spec=spec,
    )
    assert result.selected_heads.tolist() == [0]
    assert result.head_kl[0] < result.head_kl[1]
    assert result.kept_image_indices.tolist() == [3, 5]
    assert result.kept_indices.tolist() == [0, 1, 3, 5, 6, 7]
    assert result.layout_delta.dropped_indices == (2, 4)
    assert result.metadata["target_visual_tokens"] == 2


def test_attention_slice_normalizes_over_all_visible_keys_per_head() -> None:
    query = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]])
    keys = torch.tensor(
        [
            [[1.0, 0.0], [0.0, 1.0]],
            [[0.0, 1.0], [1.0, 0.0]],
            [[-1.0, 0.0], [0.0, -1.0]],
        ]
    )
    scores = finepruner_attention_head_slice(
        query=query,
        attention_keys=keys,
        image_key_indices=torch.tensor([0, 2]),
        attention_key_mask=torch.tensor([[True, True, True]]),
    )
    logits = torch.tensor([1.0, 0.0, -1.0]) / (2.0**0.5)
    expected = torch.softmax(logits, dim=0)[torch.tensor([0, 2])]
    assert scores.shape == (2, 2)
    assert torch.allclose(scores[0], expected)
    assert torch.allclose(scores[1], expected)


def test_unbiased_head_selection_rejects_non_finite_and_boolean_ratio() -> None:
    attention = torch.tensor([[0.0, 0.0], [5.0, 0.0]])
    selected, kl = finepruner_select_unbiased_heads(attention, head_ratio=0.5)
    assert selected.tolist() == [0]
    assert kl[0] < kl[1]
    with pytest.raises(TypeError, match="boolean"):
        finepruner_select_unbiased_heads(attention, head_ratio=True)
