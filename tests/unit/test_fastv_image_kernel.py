from __future__ import annotations

import pytest

from mllm_comp.compressors.image.fastv import (
    FastVResult,
    FastVSpec,
    fastv_image_tokens,
    fastv_scorer_query_indices,
)


torch = pytest.importorskip("torch")


def _attention_keys(scores: list[float], *, heads: int = 2) -> torch.Tensor:
    keys = torch.zeros(len(scores), heads, 2)
    keys[:, :, 0] = torch.tensor(scores).unsqueeze(1)
    return keys


def test_released_last_token_topk_preserves_every_text_token_and_layout() -> None:
    hidden = torch.arange(24, dtype=torch.float32).reshape(8, 3)
    keys = _attention_keys([0.0, 0.0, 0.1, 0.7, 0.2, 0.9, 0.0, 0.0])
    query = torch.tensor([[1.0, 0.0], [1.0, 0.0]])

    result = fastv_image_tokens(
        hidden,
        image_indices=torch.tensor([2, 3, 4, 5]),
        scorer_query=query,
        attention_keys=keys,
    )

    assert type(result) is FastVResult
    assert result.kept_indices.tolist() == [0, 1, 3, 5, 6, 7]
    assert result.kept_image_indices.tolist() == [3, 5]
    assert result.dropped_indices.tolist() == [2, 4]
    assert torch.equal(result.hidden_states, hidden[[0, 1, 3, 5, 6, 7]])
    assert result.layout_delta.kept_indices == (0, 1, 3, 5, 6, 7)
    assert result.layout_delta.protected_indices == (0, 1, 6, 7)
    assert result.metadata["ranking"] == "last_token"


def test_mean_query_averages_visible_query_probabilities() -> None:
    hidden = torch.randn(6, 3)
    keys = _attention_keys([0.0, 0.9, 0.8, 0.2, 0.1, 0.0], heads=1)
    query = torch.tensor([[[1.0, 0.0]], [[0.5, 0.0]]])
    visibility = torch.tensor(
        [
            [True, True, True, False, False, False],
            [True, True, True, True, True, True],
        ]
    )

    result = fastv_image_tokens(
        hidden,
        image_indices=[1, 2, 3, 4],
        scorer_query=query,
        attention_keys=keys,
        attention_key_mask=visibility,
        spec=FastVSpec(score_source="mean_query"),
    )

    assert result.kept_image_indices.tolist() == [1, 2]
    assert result.metadata["ranking"] == "mean_query"


def test_layer_zero_seeded_random_selection_is_reproducible() -> None:
    hidden = torch.randn(10, 4)
    kwargs = {
        "hidden_states": hidden,
        "image_indices": torch.arange(2, 8),
        "scorer_query": None,
        "attention_keys": None,
        "spec": FastVSpec(prune_layer=0, random_seed=17),
    }

    first = fastv_image_tokens(**kwargs)
    second = fastv_image_tokens(**kwargs)

    assert first.kept_indices.tolist() == second.kept_indices.tolist()
    assert first.metadata["ranking"] == "seeded_random"


def test_training_query_stops_before_teacher_forced_answer() -> None:
    image_mask = torch.tensor([0, 1, 1, 0, 0, 0, 0, 0, 0], dtype=torch.bool)
    labels = torch.tensor([[-100, -100, -100, -100, -100, -100, -100, 4, 5]])

    last = fastv_scorer_query_indices(
        image_mask,
        score_source="last_token",
        attention_mask=torch.ones(1, 9),
        labels=labels,
    )
    mean = fastv_scorer_query_indices(
        image_mask,
        score_source="mean_query",
        attention_mask=torch.ones(1, 9),
        labels=labels,
    )

    assert last.tolist() == [6]
    assert mean.tolist() == list(range(7))


def test_protected_image_token_is_retained_inside_exact_budget() -> None:
    hidden = torch.randn(6, 4)
    keys = _attention_keys([0.0, 0.9, 0.8, -1.0, 0.7, 0.0])
    query = torch.tensor([[1.0, 0.0], [1.0, 0.0]])

    result = fastv_image_tokens(
        hidden,
        image_indices=[1, 2, 3, 4],
        scorer_query=query,
        attention_keys=keys,
        protected_indices=(3,),
    )

    assert result.kept_image_indices.tolist() == [1, 3]
    assert 3 in result.layout_delta.protected_indices


@pytest.mark.parametrize(
    "kwargs,match",
    (
        ({"prune_layer": -1}, "prune_layer"),
        ({"prune_ratio": 1.0}, "prune_ratio"),
        ({"min_visual_tokens": 0}, "min_visual_tokens"),
        ({"score_source": "received"}, "score_source"),
    ),
)
def test_invalid_config_is_rejected(kwargs: dict[str, object], match: str) -> None:
    with pytest.raises((TypeError, ValueError), match=match):
        FastVSpec(**kwargs)  # type: ignore[arg-type]
