from __future__ import annotations

import pytest

from mllm_comp.compressors.image.divprune import (
    DivPruneSpec,
    divprune_image,
    select_divprune_indices,
)


torch = pytest.importorskip("torch")


def _pinned_reference(tokens: torch.Tensor, target_count: int) -> torch.Tensor:
    normalized = torch.nn.functional.normalize(tokens, dim=-1)
    distances = 1.0 - normalized @ normalized.T
    selected = torch.empty(target_count, dtype=torch.long)
    for index in range(target_count):
        if index == 0:
            scores = torch.topk(distances, 2, dim=0, largest=False).values[1]
        else:
            scores = distances.index_select(0, selected[:index]).min(dim=0).values
            scores[selected[:index]] = float("-inf")
        selected[index] = torch.argmax(scores)
    return selected


def test_chunked_cosine_maxmin_matches_pinned_fuxian_selection() -> None:
    tokens = torch.tensor(
        [[1.0, 0.0], [0.9, 0.1], [0.0, 1.0], [-1.0, 0.0], [0.0, -1.0]]
    )

    expected = _pinned_reference(tokens, 3)
    actual = select_divprune_indices(tokens, 3, distance_chunk_size=2)

    assert actual.tolist() == expected.tolist()


def test_default_ratio_matches_released_576_token_budget() -> None:
    spec = DivPruneSpec()

    assert spec.resolve_target_count(576) == 56


def test_exact_budget_restores_source_order_and_publishes_complete_drop_delta() -> None:
    tokens = torch.tensor(
        [[1.0, 0.0], [0.9, 0.1], [0.0, 1.0], [-1.0, 0.0], [0.0, -1.0]]
    )

    output = divprune_image(tokens, spec=DivPruneSpec(target_count=3))

    assert output.tokens.shape == (3, 2)
    assert output.indices.tolist() == sorted(output.indices.tolist())
    assert output.selection_order.tolist() == _pinned_reference(tokens, 3).tolist()
    torch.testing.assert_close(output.tokens, tokens.index_select(0, output.indices))
    assert output.layout_delta.source_count == 5
    assert output.layout_delta.result_count == 3
    assert output.layout_delta.operation == "drop"
    assert output.layout_delta.kept_indices == tuple(output.indices.tolist())
    assert output.metadata["effective_target_count"] == 3
    assert output.metadata["selected_indices"] == tuple(output.indices.tolist())


def test_structural_indices_seed_selection_and_count_toward_exact_budget() -> None:
    tokens = torch.tensor(
        [[1.0, 0.0], [0.9, 0.1], [0.0, 1.0], [-1.0, 0.0], [0.0, -1.0]]
    )

    output = divprune_image(
        tokens,
        spec=DivPruneSpec(target_count=3),
        protected_indices=(4,),
    )

    assert output.selection_order[0].item() == 4
    assert 4 in output.indices.tolist()
    assert output.layout_delta.protected_indices == (4,)
    assert output.layout_delta.old_to_new[4] is not None
    assert output.tokens.shape[0] == 3


def test_identity_reuses_input_and_minimum_keeps_one_token() -> None:
    tokens = torch.randn(4, 3)
    identity = divprune_image(tokens, spec=DivPruneSpec(retain_ratio=1.0))
    minimum = divprune_image(tokens, spec=DivPruneSpec(target_count=1))

    assert identity.tokens is tokens
    assert identity.indices.tolist() == [0, 1, 2, 3]
    assert identity.layout_delta.is_identity
    assert identity.metadata["compression_applied"] is False
    assert minimum.tokens.shape == (1, 3)
    assert minimum.layout_delta.result_count == 1


def test_zero_vectors_do_not_create_nan_or_duplicate_indices() -> None:
    tokens = torch.tensor([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])

    selected = select_divprune_indices(tokens, 3)

    assert sorted(selected.tolist()) == [0, 1, 2]


def test_single_sequence_contract_prevents_cross_image_selection() -> None:
    with pytest.raises(ValueError, match="single-sequence contract"):
        divprune_image(torch.zeros(2, 4, 3))

    first = divprune_image(
        torch.eye(4),
        spec=DivPruneSpec(target_count=2),
        source_layout={"image_index": 0},
    )
    second = divprune_image(
        -torch.eye(4),
        spec=DivPruneSpec(target_count=2),
        source_layout={"image_index": 1},
    )
    assert first.layout_delta.source_layout["image_index"] == 0
    assert second.layout_delta.source_layout["image_index"] == 1
    assert first.layout_delta.source_count == second.layout_delta.source_count == 4


@pytest.mark.parametrize(
    "kwargs,match",
    (
        ({"retain_ratio": 0.0}, "retain_ratio"),
        ({"target_count": 0}, "target_count"),
        ({"prune_layer": 1}, "prune_layer=0"),
        ({"distance": "euclidean"}, "distance='cosine'"),
        ({"selection": "random"}, "official_greedy_maxmin"),
        ({"distance_chunk_size": 0}, "distance_chunk_size"),
    ),
)
def test_invalid_config_is_rejected(kwargs: dict[str, object], match: str) -> None:
    with pytest.raises((TypeError, ValueError), match=match):
        DivPruneSpec(**kwargs)  # type: ignore[arg-type]


def test_invalid_tensor_and_protected_budget_are_rejected() -> None:
    with pytest.raises(TypeError, match="floating dtype"):
        divprune_image(torch.ones(3, 2, dtype=torch.long))
    with pytest.raises(ValueError, match="finite"):
        select_divprune_indices(torch.tensor([[float("nan")], [1.0]]), 1)
    with pytest.raises(ValueError, match="protected structural set"):
        divprune_image(
            torch.randn(3, 2),
            spec=DivPruneSpec(target_count=1),
            protected_indices=(0, 2),
        )


@pytest.mark.parametrize("protected,expected", [((), [2, 3, 1]), ((0,), [0, 2, 1])])
def test_custom_distances_drive_maxmin_without_computing_cosine_distances(
    monkeypatch, protected, expected
) -> None:
    tokens = torch.tensor([[1.0, 0.0], [0.9, 0.1], [0.0, 1.0], [-1.0, 0.0]])
    distances = torch.tensor(
        [[0.0, 2.0, 3.0, 1.0], [2.0, 0.0, 4.0, 5.0],
         [3.0, 4.0, 0.0, 6.0], [1.0, 5.0, 6.0, 0.0]]
    )

    def forbidden(*args, **kwargs):
        raise AssertionError("Custom distances must replace the cosine computation")

    # Scoring is supplied independently; no matrix product is needed by max-min selection.
    monkeypatch.setattr(torch.Tensor, "__matmul__", forbidden)
    monkeypatch.setattr(torch, "matmul", forbidden)
    monkeypatch.setattr(torch, "mv", forbidden)
    selected = select_divprune_indices(
        tokens, 3, protected_indices=protected, rules={"scorer": lambda request: distances}
    )

    assert selected.tolist() == expected


@pytest.mark.parametrize("autocast", [False, True])
def test_custom_distances_keep_score_precision_and_detach_before_selection(autocast) -> None:
    tokens = torch.ones(4, 2)
    distances = torch.full((4, 4), 0.75, dtype=torch.float64, requires_grad=True)
    expected_dtype = torch.bfloat16 if autocast else tokens.dtype

    def selector(request):
        scores = request["scores"]
        assert scores.dtype == expected_dtype
        assert scores.device == tokens.device
        assert not scores.requires_grad
        torch.testing.assert_close(scores, torch.full((4, 4), 0.75, dtype=expected_dtype))
        return torch.tensor([0, 2])

    with torch.autocast("cpu", dtype=torch.bfloat16, enabled=autocast):
        selected = select_divprune_indices(
            tokens, 2, rules={"scorer": lambda request: distances, "selector": selector}
        )

    assert selected.tolist() == [0, 2]


@pytest.mark.parametrize(
    "output,match",
    [(torch.zeros(4, 3), "shape"), (torch.ones(4, 4, dtype=torch.long), "floating"),
     (torch.full((4, 4), float("nan")), "finite")],
)
def test_custom_distances_require_finite_floating_pairwise_scores(output, match) -> None:
    with pytest.raises(ValueError, match=match):
        select_divprune_indices(torch.ones(4, 2), 2, rules={"scorer": lambda request: output})
