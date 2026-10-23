from __future__ import annotations

import math

import pytest


torch = pytest.importorskip("torch")

from mllm_comp.compressors.image.transprune import (  # noqa: E402
    TransPruneSpec,
    select_transprune_ttv_history,
    transprune_image_tokens,
    transprune_instruction_guided_attention,
    transprune_module_transition_variation,
)


def test_public_recipe_and_next_layer_depth_contract() -> None:
    spec = TransPruneSpec()

    assert spec.accumulating_layer_indices == (6, 7, 8, 9, 10, 11)
    assert spec.boundary_after_layer_indices == (6, 8, 11)
    assert spec.retain_ratios == (0.875, 0.625, 0.125)
    assert spec.alpha == 0.5
    spec.validate_decoder_depth(13)
    with pytest.raises(ValueError, match="following decoder layer"):
        spec.validate_decoder_depth(12)
    with pytest.raises(ValueError, match="strictly decreasing"):
        TransPruneSpec(retain_ratios=(0.5, 0.5, 0.25))


def test_release_ttv_uses_raw_module_output_and_full_prompt_softmax() -> None:
    before = torch.tensor([[[1.0, 0.0]] * 4])
    after = torch.tensor(
        [[[1.0, 0.0], [0.0, 2.0], [-3.0, 0.0], [1.0, 0.0]]]
    )
    images = torch.tensor([1, 2])

    release = transprune_module_transition_variation(
        before,
        after,
        image_indices=images,
        direction_rule="release_1_minus_cosine",
        softmax_scope="release_full_sequence",
    )
    paper_scope = transprune_module_transition_variation(
        before,
        after,
        image_indices=images,
        direction_rule="release_1_minus_cosine",
        softmax_scope="paper_visual_tokens",
    )

    full_denominator = 2.0 + math.e + math.e**2
    visual_denominator = math.e + math.e**2
    assert torch.allclose(
        release,
        torch.tensor(
            [2.0 * math.e / full_denominator, 3.0 * math.e**2 / full_denominator]
        ),
    )
    assert torch.allclose(
        paper_scope,
        torch.tensor(
            [
                2.0 * math.e / visual_denominator,
                3.0 * math.e**2 / visual_denominator,
            ]
        ),
    )


def test_iga_normalizes_over_all_visible_keys_and_averages_queries() -> None:
    queries = torch.tensor([[[1.0]], [[2.0]]])
    keys = torch.tensor([[[0.0]], [[1.0]], [[2.0]], [[3.0]]])
    image_indices = torch.tensor([1, 2])
    visibility = torch.tensor(
        [[True, True, True, False], [True, True, True, True]]
    )

    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        scores = transprune_instruction_guided_attention(
            instruction_queries=queries.bfloat16(),
            image_keys=keys.index_select(0, image_indices).bfloat16(),
            attention_keys=keys.bfloat16(),
            image_indices=image_indices,
            attention_key_mask=visibility,
        )
    first = torch.softmax(torch.tensor([0.0, 1.0, 2.0]), dim=0)
    second = torch.softmax(torch.tensor([0.0, 2.0, 4.0, 6.0]), dim=0)
    expected = torch.stack(
        ((first[1] + second[1]) / 2.0, (first[2] + second[2]) / 2.0)
    )

    assert scores.dtype == torch.float32
    assert torch.allclose(scores, expected, atol=1e-6, rtol=1e-6)


def test_progressive_selection_uses_original_budget_and_realigns_history() -> None:
    hidden = torch.arange(20, dtype=torch.float32).reshape(10, 2)
    spec = TransPruneSpec(
        accumulating_layers=(1, 2, 3),
        prune_after_layers=(1, 2, 3),
        retain_ratios=(0.75, 0.5, 0.25),
    )
    history = [torch.arange(1, 9, dtype=torch.float32)]

    first = transprune_image_tokens(
        hidden,
        image_indices=torch.arange(1, 9),
        boundary_after_layer=0,
        ttv_history=history,
        image_attention_scores=torch.arange(1, 9, dtype=torch.float32),
        initial_visual_tokens=8,
        spec=spec,
        source_layout={"stage": 0},
        result_layout={"position_policy": "contiguous"},
    )

    assert first.selected_visual_indices.tolist() == [2, 3, 4, 5, 6, 7]
    assert first.kept_image_indices.tolist() == [3, 4, 5, 6, 7, 8]
    assert first.kept_indices.tolist() == [0, 3, 4, 5, 6, 7, 8, 9]
    assert first.layout_delta.dropped_indices == (1, 2)

    aligned = select_transprune_ttv_history(
        history,
        first.selected_visual_indices,
    )
    second = transprune_image_tokens(
        first.hidden_states,
        image_indices=torch.arange(1, 7),
        boundary_after_layer=1,
        ttv_history=aligned,
        image_attention_scores=torch.arange(1, 7, dtype=torch.float32),
        initial_visual_tokens=8,
        spec=spec,
    )

    assert aligned[0].tolist() == [3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
    assert second.metadata["result_image_tokens"] == 4
    assert second.selected_visual_indices.tolist() == [2, 3, 4, 5]
    assert second.kept_indices.tolist() == [0, 3, 4, 5, 6, 7]
