from __future__ import annotations

import pytest

from mllm_comp.compressors.image.visionzip import (
    scaled_visionzip_allocation,
    visionzip_image,
)


torch = pytest.importorskip("torch")


def _received_attention(scores: torch.Tensor) -> torch.Tensor:
    """Attention whose mean-received importance equals ``scores``."""

    return scores.unsqueeze(0).expand(scores.numel(), -1).clone()


def test_default_27_to_5_budget_and_complete_layout_delta() -> None:
    count = 64
    tokens = torch.arange(count * 3, dtype=torch.float32).reshape(count, 3)
    attention = _received_attention(torch.arange(count, dtype=torch.float32))
    keys = torch.stack(
        (
            torch.linspace(0.0, 1.0, count),
            torch.linspace(1.0, 0.0, count),
        ),
        dim=-1,
    )

    output = visionzip_image(tokens, attention, keys)

    assert output.tokens.shape == (32, 3)
    assert output.indices.shape == (32,)
    assert output.metadata["dominant_count"] == 27
    assert output.metadata["contextual_count"] == 5
    assert output.metadata["dominant_contextual_weights"] == (27, 5)
    assert output.metadata["dominant_indices"] == tuple(range(63, 36, -1))
    assert output.layout_delta.source_count == 64
    assert output.layout_delta.result_count == 32
    assert output.layout_delta.operation == "merge"
    assert set(index for group in output.source_groups for index in group) == set(
        range(count)
    )
    assert sum(len(group) for group in output.source_groups) == count


def test_budget_scaling_preserves_paper_ratio_with_half_up_rounding() -> None:
    assert scaled_visionzip_allocation(32) == (27, 5)
    assert scaled_visionzip_allocation(16) == (14, 2)
    assert scaled_visionzip_allocation(2) == (1, 1)
    with pytest.raises(ValueError, match="at least one dominant"):
        scaled_visionzip_allocation(1)


def test_clip_uses_cls_attention_and_automatically_protects_cls() -> None:
    tokens = torch.arange(12, dtype=torch.float32).reshape(6, 2)
    attention = torch.zeros(2, 6, 6)
    attention[:, 0, 4] = 10.0
    # Mean-received attention would prefer token 2, demonstrating that the CLS
    # path is actually used rather than merely labelled in metadata.
    attention[:, :, 2] += 5.0
    attention[:, 0, 4] = 20.0
    keys = torch.eye(6, dtype=torch.float32)

    output = visionzip_image(
        tokens,
        attention,
        keys,
        budget=3,
        cls_index=0,
    )

    assert output.metadata["attention_mode"] == "cls"
    assert output.metadata["dominant_indices"] == (4,)
    assert output.metadata["protected_indices"] == (0,)
    assert output.layout_delta.protected_indices == (0,)
    assert next(group for group in output.source_groups if 0 in group) == (0,)


def test_siglip_uses_mean_received_attention_without_cls() -> None:
    tokens = torch.arange(10, dtype=torch.float32).reshape(5, 2)
    scores = torch.tensor([0.0, 9.0, 2.0, 1.0, 3.0])
    attention = _received_attention(scores).unsqueeze(0)  # one attention head
    keys = torch.eye(5, dtype=torch.float32).unsqueeze(0)  # one key head

    output = visionzip_image(tokens, attention, keys, budget=2)
    precomputed_output = visionzip_image(tokens, scores, keys, budget=2)

    assert output.metadata["attention_mode"] == "mean_received"
    assert output.metadata["dominant_indices"] == (1,)
    assert output.metadata["protected_indices"] == ()
    assert precomputed_output.metadata["dominant_indices"] == output.metadata["dominant_indices"]
    assert torch.equal(precomputed_output.indices, output.indices)
    assert torch.equal(precomputed_output.tokens, output.tokens)


def test_released_context_targets_and_target_plus_assigned_mean() -> None:
    tokens = torch.arange(8, dtype=torch.float32).unsqueeze(-1)
    received = torch.tensor([10.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 9.0])
    attention = _received_attention(received)
    keys = torch.zeros(8, 2)
    keys[[1, 2, 3], 0] = 1.0
    keys[[4, 5, 6], 1] = 1.0

    output = visionzip_image(
        tokens,
        attention,
        keys,
        budget=4,
        dominant_weight=1,
        contextual_weight=1,
    )

    assert output.metadata["dominant_indices"] == (0, 7)
    assert output.metadata["contextual_indices"] == (1, 4)
    assert output.metadata["contextual_target_sampling"] == (
        "released_prefix_stride"
    )
    assert output.metadata["merge_value"] == "target_plus_assigned_mean"
    assert output.indices.tolist() == [0, 1, 4, 7]
    assert output.source_groups == ((0,), (1, 2, 3), (4, 5, 6), (7,))
    torch.testing.assert_close(
        output.tokens.squeeze(-1),
        torch.tensor([0.0, 3.5, 9.5, 7.0]),
    )
    assert dict(output.layout_delta.old_to_new) == {
        0: 0,
        1: 1,
        2: 1,
        3: 1,
        4: 2,
        5: 2,
        6: 2,
        7: 3,
    }
    assert output.layout_delta.merge_groups == ((1, 2, 3), (4, 5, 6))


def test_contextual_matching_uses_direction_not_key_magnitude() -> None:
    tokens = torch.arange(8, dtype=torch.float32).unsqueeze(-1)
    received = torch.tensor([10.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 9.0])
    attention = _received_attention(received)
    keys = torch.tensor(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [1.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [10.0, 10.0],
            [0.0, 1.0],
            [0.0, 0.0],
        ]
    )

    output = visionzip_image(
        tokens,
        attention,
        keys,
        budget=4,
        dominant_weight=1,
        contextual_weight=1,
    )

    # Context targets are 1 and 4.  Source 3 is parallel to target 1; without
    # L2 normalisation a high-norm target could win only because its norm is
    # much larger.
    assert output.metadata["contextual_indices"] == (1, 4)
    source_three_group = next(group for group in output.source_groups if 3 in group)
    assert 1 in source_three_group
    assert 4 not in source_three_group
    assert output.metadata["similarity"] == (
        "mean_head_l2_normalized_key_dot_product"
    )


def test_special_tokens_are_singletons_and_count_toward_total_budget() -> None:
    tokens = torch.arange(16, dtype=torch.float32).reshape(8, 2)
    attention = torch.zeros(8, 8)
    attention[0, 1] = 10.0
    keys = torch.eye(8)

    output = visionzip_image(
        tokens,
        attention,
        keys,
        budget=4,
        cls_index=0,
        protected_indices=(7,),
        dominant_weight=1,
        contextual_weight=1,
    )

    assert output.tokens.shape[0] == 4
    assert output.metadata["protected_count"] == 2
    assert output.metadata["dominant_count"] == 1
    assert output.metadata["contextual_count"] == 1
    assert output.layout_delta.protected_indices == (0, 7)
    for protected in (0, 7):
        assert next(group for group in output.source_groups if protected in group) == (
            protected,
        )


def test_selection_is_deterministic_under_attention_and_key_ties() -> None:
    tokens = torch.arange(20, dtype=torch.float32).reshape(10, 2)
    attention = torch.zeros(10, 10)
    keys = torch.zeros(10, 3)

    first = visionzip_image(tokens, attention, keys, budget=4)
    second = visionzip_image(tokens, attention, keys, budget=4)

    assert first.metadata["dominant_indices"] == (0, 1, 2)
    assert first.metadata["contextual_indices"] == (3,)
    assert first.source_groups == second.source_groups
    assert torch.equal(first.indices, second.indices)
    assert torch.equal(first.tokens, second.tokens)


def test_contextual_means_remain_differentiable_for_every_source_token() -> None:
    tokens = (
        torch.arange(1, 17, dtype=torch.float32).reshape(8, 2).clone().requires_grad_()
    )
    attention = _received_attention(
        torch.tensor([10.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 9.0])
    )
    keys = torch.zeros(8, 2)
    keys[[1, 2, 3], 0] = 1.0
    keys[[4, 5, 6], 1] = 1.0

    output = visionzip_image(
        tokens,
        attention,
        keys,
        budget=4,
        dominant_weight=1,
        contextual_weight=1,
    )
    output.tokens.sum().backward()

    assert tokens.grad is not None
    assert bool(torch.isfinite(tokens.grad).all())
    assert bool((tokens.grad > 0).all())


def test_noop_is_identity_and_reuses_the_input_tensor() -> None:
    tokens = torch.randn(5, 3)
    attention = torch.zeros(5, 5)
    keys = torch.randn(5, 4)

    output = visionzip_image(tokens, attention, keys, budget=32)

    assert output.tokens is tokens
    assert output.indices.tolist() == list(range(5))
    assert output.layout_delta.is_identity
    assert output.metadata["compression_applied"] is False
    assert output.metadata["effective_budget"] == 5


def test_single_sequence_contract_and_budget_validation_are_explicit() -> None:
    with pytest.raises(ValueError, match="single-sequence contract"):
        visionzip_image(
            torch.zeros(2, 4, 3),
            torch.zeros(2, 4, 4),
            torch.zeros(2, 4, 3),
        )

    with pytest.raises(ValueError, match="room for at least one dominant"):
        visionzip_image(
            torch.zeros(5, 2),
            torch.zeros(5, 5),
            torch.zeros(5, 2),
            budget=2,
            protected_indices=(0,),
        )

    with pytest.raises(ValueError, match="requires cls_index"):
        visionzip_image(
            torch.zeros(5, 2),
            torch.zeros(5, 5),
            torch.zeros(5, 2),
            budget=3,
            attention_mode="cls",
        )
