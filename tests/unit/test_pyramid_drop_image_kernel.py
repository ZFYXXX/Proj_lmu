from __future__ import annotations

import pytest

from mllm_comp.compressors.image.pyramid_drop import (
    PyramidDropSchedule,
    PyramidDropSpec,
    pyramid_attention_key_visibility,
    pyramid_attention_scores,
    pyramid_drop,
    pyramid_instruction_indices,
)


torch = pytest.importorskip("torch")


def test_paper_default_schedule_matches_32_layer_boundaries() -> None:
    schedule = PyramidDropSchedule.uniform(32)

    assert schedule.layer_indices == (7, 15, 23)
    assert schedule.keep_ratio == 0.5
    assert schedule.num_stages == 4


def test_training_instruction_queries_are_answer_prefixes() -> None:
    image_mask = torch.tensor([0, 1, 1, 0, 0, 0, 0], dtype=torch.bool)
    labels = torch.tensor([[-100, -100, -100, -100, 17, -100, 23]])

    indices = pyramid_instruction_indices(
        image_mask,
        attention_mask=torch.ones(1, 7),
        labels=labels,
    )

    assert indices.tolist() == [3, 5]


def test_attention_visibility_selects_each_causal_query_row() -> None:
    mask = torch.full((1, 1, 5, 5), torch.finfo(torch.float32).min)
    mask[0, 0] = torch.triu(mask[0, 0], diagonal=1)

    visible = pyramid_attention_key_visibility(
        mask,
        instruction_indices=torch.tensor([2, 4]),
    )

    assert visible.tolist() == [
        [True, True, True, False, False],
        [True, True, True, True, True],
    ]


def test_schedule_rejects_drop_before_deepstack_is_consumed() -> None:
    schedule = PyramidDropSchedule.uniform(32)

    schedule.validate_after_layer(6)
    with pytest.raises(ValueError, match="must be after"):
        schedule.validate_after_layer(7)


def test_last_instruction_query_ranks_image_keys_and_restores_order() -> None:
    hidden = torch.arange(18, dtype=torch.float32).reshape(6, 3)
    image_indices = torch.tensor([1, 2, 3, 4])
    query = torch.tensor([1.0, 0.0])
    keys = torch.tensor(
        [
            [0.1, 0.0],
            [0.9, 0.0],
            [0.2, 0.0],
            [0.8, 0.0],
        ]
    )

    result = pyramid_drop(
        hidden,
        image_indices=image_indices,
        instruction_query=query,
        image_keys=keys,
    )

    assert result.kept_image_indices.tolist() == [2, 4]
    assert result.kept_indices.tolist() == [0, 2, 4, 5]
    assert result.dropped_indices.tolist() == [1, 3]
    assert torch.equal(result.hidden_states, hidden[[0, 2, 4, 5]])
    assert result.layout_delta.dropped_indices == (1, 3)
    assert result.layout_delta.protected_indices == (0, 5)


def test_paper_rounding_removes_ceil_fraction_for_odd_image_count() -> None:
    hidden = torch.randn(7, 4)
    result = pyramid_drop(
        hidden,
        image_indices=torch.tensor([1, 2, 3, 4, 5]),
        instruction_query=torch.tensor([1.0, 0.0]),
        image_keys=torch.tensor(
            [[0.1, 0.0], [0.2, 0.0], [0.3, 0.0], [0.4, 0.0], [0.5, 0.0]]
        ),
    )

    assert result.metadata["source_image_tokens"] == 5
    assert result.metadata["result_image_tokens"] == 2


def test_protected_image_token_is_kept_even_below_score_budget() -> None:
    hidden = torch.randn(6, 4)
    result = pyramid_drop(
        hidden,
        image_indices=[1, 2, 3, 4],
        instruction_query=torch.tensor([1.0, 0.0]),
        image_keys=torch.tensor(
            [[0.9, 0.0], [0.8, 0.0], [-1.0, 0.0], [0.7, 0.0]]
        ),
        protected_indices=(3,),
    )

    assert result.kept_image_indices.tolist() == [1, 3]
    assert 3 in result.layout_delta.protected_indices


def test_all_non_image_tokens_are_preserved_and_marked_protected() -> None:
    hidden = torch.randn(5, 3)
    result = pyramid_drop(
        hidden,
        image_indices=[2, 3],
        instruction_query=torch.tensor([1.0, 0.0]),
        image_keys=torch.tensor([[1.0, 0.0], [0.0, 0.0]]),
    )

    assert result.kept_indices.tolist() == [0, 1, 2, 4]
    assert result.layout_delta.protected_indices == (0, 1, 4)


def test_multi_head_scores_average_headwise_attention_probabilities() -> None:
    query = torch.tensor([[2.0, 0.0], [0.0, 2.0]])
    keys = torch.tensor(
        [
            [[1.0, 0.0], [0.0, 1.0]],
            [[0.5, 0.0], [0.0, 0.5]],
        ]
    )

    scores = pyramid_attention_scores(
        instruction_query=query,
        image_keys=keys,
        image_count=2,
    )

    logits = torch.tensor([2.0, 1.0]) * (2.0 ** -0.5)
    expected = torch.softmax(logits, dim=0)
    assert torch.allclose(scores, expected)


def test_scores_normalize_against_complete_visible_key_sequence() -> None:
    query = torch.tensor([[2.0, 0.0], [0.0, 2.0]])
    all_keys = torch.tensor(
        [
            [[4.0, 0.0], [0.0, 0.0]],
            [[1.0, 0.0], [0.0, 1.0]],
            [[0.5, 0.0], [0.0, 0.5]],
            [[9.0, 0.0], [0.0, 9.0]],
        ]
    )
    image_indices = torch.tensor([1, 2])
    scores = pyramid_attention_scores(
        instruction_query=query,
        image_keys=all_keys.index_select(0, image_indices),
        image_count=2,
        attention_keys=all_keys,
        attention_key_mask=torch.tensor([1, 1, 1, 0]),
        image_key_indices=image_indices,
    )

    logits = (all_keys * query.unsqueeze(0)).sum(-1) * (2.0 ** -0.5)
    logits[3] = float("-inf")
    expected = torch.softmax(logits, dim=0).index_select(0, image_indices).mean(-1)
    assert torch.allclose(scores, expected)


def test_multiple_instruction_queries_use_their_own_causal_visibility() -> None:
    queries = torch.tensor([[[1.0]], [[2.0]]])
    all_keys = torch.tensor([[[0.0]], [[1.0]], [[2.0]], [[4.0]]])
    image_indices = torch.tensor([1, 2])
    visibility = torch.tensor(
        [
            [1, 1, 1, 0],
            [1, 1, 1, 1],
        ]
    )

    scores = pyramid_attention_scores(
        instruction_query=queries,
        image_keys=all_keys.index_select(0, image_indices),
        image_count=2,
        attention_keys=all_keys,
        attention_key_mask=visibility,
        image_key_indices=image_indices,
    )

    logits = torch.einsum("qhd,thd->tqh", queries, all_keys)
    logits = logits.masked_fill(~visibility.T.unsqueeze(-1).bool(), float("-inf"))
    expected = (
        torch.softmax(logits, dim=0)
        .index_select(0, image_indices)
        .mean(dim=(-1, -2))
    )
    assert torch.allclose(scores, expected)


def test_ties_are_deterministic_and_keep_earlier_image_tokens() -> None:
    hidden = torch.randn(6, 4)
    result = pyramid_drop(
        hidden,
        image_indices=[1, 2, 3, 4],
        instruction_query=torch.tensor([1.0, 0.0]),
        image_keys=torch.ones(4, 2),
    )

    assert result.kept_image_indices.tolist() == [1, 2]


def test_hidden_state_gather_remains_differentiable() -> None:
    hidden = torch.arange(24, dtype=torch.float32).reshape(6, 4).requires_grad_()
    keys = torch.randn(4, 2, requires_grad=True)
    result = pyramid_drop(
        hidden,
        image_indices=[1, 2, 3, 4],
        instruction_query=torch.tensor([1.0, 0.0], requires_grad=True),
        image_keys=keys,
    )

    (result.hidden_states.sum() + result.image_scores.sum()).backward()

    assert hidden.grad is not None
    assert keys.grad is not None


def test_empty_image_stream_is_an_identity() -> None:
    hidden = torch.randn(3, 4)
    result = pyramid_drop(
        hidden,
        image_indices=[],
        instruction_query=torch.tensor([1.0, 0.0]),
        image_keys=torch.empty(0, 2),
    )

    assert result.layout_delta.is_identity
    assert result.hidden_states is hidden


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"image_indices": [2, 1]}, "source order"),
        ({"image_indices": [1, 1]}, "duplicates"),
        ({"image_indices": [9]}, "outside"),
    ],
)
def test_invalid_image_index_layout_fails_early(
    kwargs: dict[str, object], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        pyramid_drop(
            torch.randn(4, 3),
            instruction_query=torch.tensor([1.0, 0.0]),
            image_keys=torch.randn(len(kwargs["image_indices"]), 2),
            **kwargs,
        )


def test_spec_validation_is_strict() -> None:
    with pytest.raises(ValueError, match="keep_ratio"):
        PyramidDropSpec(keep_ratio=0.0)
    with pytest.raises(ValueError, match="num_stages"):
        PyramidDropSpec(num_stages=1)
