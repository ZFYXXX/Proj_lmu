from __future__ import annotations

import pytest


torch = pytest.importorskip("torch")

from mllm_comp.compressors.image.sparsevlm import (  # noqa: E402
    SparseVLMSpec,
    density_peak_average,
    select_sparsevlm_text_raters,
    sparsevlm_image_tokens,
    remap_sparsevlm_attention_scores,
)


def test_pending_attention_mass_and_mean_replay_follow_the_same_layout():
    from mllm_comp.core.layout import LayoutDelta
    from mllm_comp.compressors.image.decoder_lifecycle import layout_merge_groups, replay_mean_groups

    delta = LayoutDelta.from_mapping(source_count=8, result_count=5,
        old_to_new=[0, 1, 2, 2, None, 3, 3, 4], media_unit="llm_token")
    images = torch.arange(1, 7)
    scores = torch.tensor([0.1, 0.2, 0.3, 0.9, 0.04, 0.06])
    torch.testing.assert_close(remap_sparsevlm_attention_scores(scores, delta, images), torch.tensor([0.1, 0.5, 0.1]))
    values = torch.arange(16, dtype=torch.float32).reshape(8, 2).requires_grad_()
    groups = layout_merge_groups(delta)
    merged = replay_mean_groups(values, groups)
    torch.testing.assert_close(merged, torch.stack([values[0], values[1], values[2:4].mean(0), values[5:7].mean(0), values[7]]))
    visual = replay_mean_groups(values[images], layout_merge_groups(delta, images))
    torch.testing.assert_close(visual, merged[1:4])
    merged.sum().backward()
    torch.testing.assert_close(values.grad, torch.tensor([1, 1, 0.5, 0.5, 0, 0.5, 0.5, 1]).view(8, 1).expand(8, 2))


def test_sparsevlm_defaults_and_onevision_schedule_scaling() -> None:
    spec = SparseVLMSpec()
    assert spec.prune_layers == (2, 6, 15)
    assert spec.stage_target(2, 1152) == 600
    assert spec.stage_target(15, 288) == 55
    with pytest.raises(ValueError, match="equal length"):
        SparseVLMSpec(prune_layers=(2, 6), schedule_keep_tokens=(300,))
    with pytest.raises(TypeError, match="boolean"):
        SparseVLMSpec(recycle_ratio=True)


def test_sparsevlm_text_raters_use_post_image_relative_attention() -> None:
    embeddings = torch.tensor(
        [
            [0.0, 0.0],
            [4.0, 0.0],
            [3.0, 0.0],
            [2.0, 0.0],
            [0.0, 1.0],
            [-1.0, 0.0],
        ]
    )
    result = select_sparsevlm_text_raters(
        embeddings,
        image_indices=torch.tensor([1, 2]),
        text_candidate_indices=torch.tensor([3, 4, 5]),
    )
    assert 3 in result.rater_indices.tolist()
    assert set(result.rater_indices.tolist()) <= {3, 4, 5}


def test_sparsevlm_stage_has_exact_selection_merge_and_layout_mapping() -> None:
    hidden = torch.arange(11, dtype=torch.float32).view(11, 1)
    result = sparsevlm_image_tokens(
        hidden,
        image_indices=torch.arange(1, 9),
        image_attention_scores=torch.tensor([0.9, 0.1, 0.8, 0.2, 0.7, 0.6, 0.3, 0.4]),
        layer_index=2,
        initial_visual_tokens=8,
        spec=SparseVLMSpec(
            prune_layers=(2,),
            schedule_keep_tokens=(2,),
            schedule_base_visual_tokens=8,
            scale_schedule_to_input=False,
        ),
        source_layout={"stage": 0},
        result_layout={"position_policy": "contiguous"},
    )

    assert result.selected_image_indices.tolist() == [1, 3]
    assert result.recycled_image_indices.tolist() == [5, 6]
    assert result.dropped_image_indices.tolist() == [2, 4, 7, 8]
    assert result.result_image_indices.tolist() == [1, 2, 3]
    assert result.hidden_states.flatten().tolist() == [0.0, 1.0, 3.0, 5.5, 9.0, 10.0]
    assert result.representative_indices.tolist() == [0, 1, 3, 5, 9, 10]
    assert result.layout_delta.merge_groups == ((5, 6),)
    assert result.layout_delta.dropped_indices == (2, 4, 7, 8)
    assert tuple(result.layout_delta.old_to_new[index] for index in range(11)) == (
        0,
        1,
        None,
        2,
        None,
        3,
        3,
        None,
        None,
        4,
        5,
    )
    assert result.layout_delta.result_layout["position_policy"] == "contiguous"


def test_sparsevlm_stage_is_identity_when_target_already_fits() -> None:
    hidden = torch.arange(6, dtype=torch.float32).view(6, 1)
    result = sparsevlm_image_tokens(
        hidden,
        image_indices=torch.arange(1, 5),
        image_attention_scores=torch.ones(4),
        layer_index=2,
        initial_visual_tokens=4,
        spec=SparseVLMSpec(
            prune_layers=(2,),
            schedule_keep_tokens=(4,),
            schedule_base_visual_tokens=4,
            scale_schedule_to_input=False,
        ),
        source_layout={"rule_layer": 2},
        result_layout={"application_position": {"boundary": "post_attention_pre_ffn", "index": 3}},
    )

    assert torch.equal(result.hidden_states, hidden)
    assert result.representative_indices.tolist() == list(range(6))
    assert result.layout_delta.dropped_indices == ()
    assert result.layout_delta.merge_groups == ()
    assert result.layout_delta.source_layout == {"rule_layer": 2}
    assert result.layout_delta.result_layout["application_position"]["index"] == 3


def test_sparsevlm_density_peak_average_is_seed_deterministic() -> None:
    tokens = torch.tensor([[0.0], [0.1], [10.0], [10.1]])
    first, first_assignments, first_centers = density_peak_average(
        tokens,
        2,
        random_seed=7,
    )
    second, second_assignments, second_centers = density_peak_average(
        tokens,
        2,
        random_seed=7,
    )
    assert torch.equal(first, second)
    assert torch.equal(first_assignments, second_assignments)
    assert torch.equal(first_centers, second_centers)
    assert first.shape == (2, 1)
