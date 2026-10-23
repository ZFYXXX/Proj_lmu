from __future__ import annotations

import importlib
from pathlib import Path
import subprocess
import sys

import pytest


def _torch():
    return pytest.importorskip("torch")


def _tome():
    return importlib.import_module("mllm_comp.compressors.image.tome")


def test_module_import_does_not_require_torch() -> None:
    script = """
import builtins

original_import = builtins.__import__
def reject_torch(name, *args, **kwargs):
    if name == "torch" or name.startswith("torch."):
        raise AssertionError("ToMe imported torch eagerly")
    return original_import(name, *args, **kwargs)

builtins.__import__ = reject_torch
from mllm_comp.compressors.image.tome import tome_merge_image
assert callable(tome_merge_image)
"""
    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=Path(__file__).resolve().parents[2],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr


def test_single_sequence_uses_mean_head_cosine_and_paper_ordering() -> None:
    torch = _torch()
    tome = _tome()
    hidden = torch.tensor([[0.0], [10.0], [2.0], [30.0], [4.0], [50.0]])
    sizes = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    mean_keys = torch.tensor(
        [
            [-1.0, 0.0],
            [1.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [0.0, 1.0],
            [-1.0, 0.0],
        ]
    )
    # The two heads average to mean_keys, verifying head reduction happens
    # before cosine matching.
    keys = torch.stack((mean_keys + 0.25, mean_keys - 0.25), dim=1)

    result = tome.tome_merge_image(
        hidden,
        keys,
        r=2,
        token_sizes=sizes,
        protected_indices=(0,),
        coordinates=tuple((index // 3, index % 3) for index in range(6)),
    )

    expected = torch.tensor([[0.0], [5.2], [140.0 / 9.0], [50.0]])
    assert torch.allclose(result.hidden_states, expected)
    assert torch.equal(result.token_sizes, torch.tensor([[1.0], [5.0], [9.0], [6.0]]))
    assert result.proportional_attention_bias.shape == (1, 1, 4)
    assert torch.allclose(
        result.proportional_attention_bias.reshape(-1),
        result.token_sizes.reshape(-1).log(),
    )
    assert result.sequence_source_groups == ((0,), (1, 2), (3, 4), (5,))
    assert result.layout_delta.merge_groups == ((1, 2), (3, 4))
    assert result.layout_delta.protected_indices == (0,)
    assert result.layout_delta.coordinates[4] == (1, 1)
    assert dict(result.layout_delta.old_to_new) == {
        0: 0,
        1: 1,
        2: 1,
        3: 2,
        4: 2,
        5: 3,
    }
    assert result.metadata["score_metric"] == "mean_head_key_l2_cosine"
    assert result.metadata["ordering"] == "source_a_unmerged_then_destination_b"
    assert result.metadata["r_requested"] == result.metadata["r_effective"] == 2


def test_batch_selects_edges_per_sequence_and_keeps_a_uniform_shape() -> None:
    torch = _torch()
    tome = _tome()
    hidden = torch.tensor(
        [
            [[0.0], [10.0], [2.0], [30.0]],
            [[100.0], [110.0], [102.0], [130.0]],
        ]
    )
    keys = torch.tensor(
        [
            [[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]],
            [[-1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0]],
        ]
    )

    result = tome.tome_merge_image(hidden, keys, r=1)

    assert result.hidden_states.shape == (2, 3, 1)
    assert result.token_sizes.shape == (2, 3, 1)
    assert result.proportional_attention_bias.shape == (2, 1, 1, 3)
    assert result.source_groups == (
        ((2,), (0, 1), (3,)),
        ((0,), (1,), (2, 3)),
    )
    assert torch.allclose(
        result.hidden_states,
        torch.tensor(
            [
                [[2.0], [5.0], [30.0]],
                [[100.0], [110.0], [116.0]],
            ]
        ),
    )
    with pytest.raises(ValueError, match="single-sequence"):
        _ = result.layout_delta

    restored = tome.tome_restore_dense_boundary(
        result.hidden_states,
        result.layout_deltas,
    )
    assert restored.shape == hidden.shape
    for batch_index, delta in enumerate(result.layout_deltas):
        for source_index, compact_index in delta.old_to_new.items():
            assert torch.equal(
                restored[batch_index, source_index],
                result.hidden_states[batch_index, compact_index],
            )


def test_dense_boundary_restore_broadcasts_merged_representatives() -> None:
    torch = _torch()
    tome = _tome()
    hidden = torch.tensor([[0.0], [10.0], [2.0], [30.0]])
    keys = torch.tensor([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]])

    result = tome.tome_merge_image(hidden, keys, r=1)
    restored = tome.tome_restore_dense_boundary(
        result.hidden_states,
        result.layout_delta,
    )

    assert restored.shape == hidden.shape
    assert torch.equal(
        restored,
        result.hidden_states.index_select(
            0,
            torch.tensor([result.layout_delta.old_to_new[index] for index in range(4)]),
        ),
    )


def test_causal_prompt_legalizer_reorders_whole_groups_by_source() -> None:
    torch = _torch()
    tome = _tome()
    hidden = torch.tensor([[0.0], [10.0], [2.0], [30.0]], requires_grad=True)
    keys = torch.tensor([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]])

    paper = tome.tome_merge_image(hidden, keys, r=1)
    legal = tome.tome_preserve_source_order(paper)

    assert paper.sequence_source_groups == ((2,), (0, 1), (3,))
    assert legal.sequence_source_groups == ((0, 1), (2,), (3,))
    assert torch.equal(legal.hidden_states, torch.tensor([[5.0], [2.0], [30.0]]))
    assert dict(legal.layout_delta.old_to_new) == {0: 0, 1: 0, 2: 1, 3: 2}
    assert legal.layout_delta.ordering_policy.value == "preserve_source"
    assert legal.metadata["upstream_ordering"] == (
        "source_a_unmerged_then_destination_b"
    )
    legal.hidden_states.sum().backward()
    assert hidden.grad is not None


def test_dense_boundary_restore_rejects_drop_deltas() -> None:
    torch = _torch()
    tome = _tome()
    layout = importlib.import_module("mllm_comp.core.layout")
    delta = layout.LayoutDelta.drop(4, (2,), media_unit="image_patch")
    with pytest.raises(ValueError, match="merge-only"):
        tome.tome_restore_dense_boundary(torch.ones(3, 2), delta)


def test_ties_are_stable_by_source_then_destination_index() -> None:
    torch = _torch()
    tome = _tome()
    hidden = torch.arange(6.0).unsqueeze(-1)
    keys = torch.ones(6, 2)

    first = tome.tome_merge(hidden, keys, r=1)
    second = tome.tome_merge(hidden, keys, r=1)

    # A0 wins the source tie and B1 wins its destination tie.
    expected_groups = ((2,), (4,), (0, 1), (3,), (5,))
    assert first.sequence_source_groups == expected_groups
    assert second.sequence_source_groups == expected_groups
    assert torch.equal(first.hidden_states, second.hidden_states)


def test_multiple_sources_scatter_into_one_destination() -> None:
    torch = _torch()
    tome = _tome()
    hidden = torch.arange(6.0).unsqueeze(-1)
    keys = torch.tensor(
        [
            [1.0, 0.0],
            [1.0, 0.0],
            [1.0, 0.0],
            [-1.0, 0.0],
            [0.0, 1.0],
            [-1.0, 0.0],
        ]
    )

    result = tome.tome_merge_image(hidden, keys, r=2)

    assert result.sequence_source_groups == ((4,), (0, 1, 2), (3,), (5,))
    assert torch.equal(result.token_sizes, torch.tensor([[1.0], [3.0], [1.0], [1.0]]))
    assert torch.allclose(
        result.hidden_states,
        torch.tensor([[4.0], [1.0], [3.0], [5.0]]),
    )


def test_protected_tokens_are_neither_sources_nor_destinations() -> None:
    torch = _torch()
    tome = _tome()
    hidden = torch.arange(4.0).unsqueeze(-1)
    keys = torch.ones(4, 2)

    result = tome.tome_merge_image(
        hidden,
        keys,
        r=1,
        protected_indices=(0, 1),
    )

    assert result.sequence_source_groups == ((0,), (1,), (2, 3))
    assert result.layout_delta.protected_indices == (0, 1)
    assert all(
        protected not in group
        for protected in (0, 1)
        for group in result.layout_delta.merge_groups
    )


def test_layout_records_destination_b_as_the_merged_coordinate_representative() -> None:
    torch = _torch()
    tome = _tome()
    hidden = torch.arange(4.0).unsqueeze(-1)
    keys = torch.tensor([[0.0, 1.0], [-1.0, 0.0], [-1.0, 0.0], [0.0, 1.0]])
    coordinates = ((0, 0), (0, 1), (1, 0), (1, 1))

    result = tome.tome_merge_image(
        hidden,
        keys,
        r=1,
        coordinates=coordinates,
    )

    # A0 merges into B3.  Although the canonical preimage is sorted (0, 3),
    # the result position is B3's position, exactly as the ToMe destination.
    assert result.sequence_source_groups == ((2,), (1,), (0, 3))
    assert result.layout_delta.result_layout["representative_indices"] == (2, 1, 3)
    assert result.layout_delta.result_layout["result_coordinates"] == (
        (1, 0),
        (0, 1),
        (1, 1),
    )


def test_paper_cap_is_auditable_and_can_be_strict() -> None:
    torch = _torch()
    tome = _tome()
    hidden = torch.arange(3.0).unsqueeze(-1)
    keys = torch.ones(3, 2)

    clamped = tome.tome_merge_image(
        hidden,
        keys,
        r=3,
        protected_indices=(0, 1),
    )
    assert clamped.layout_delta.is_identity
    assert clamped.metadata["r_effective"] == 0
    assert clamped.metadata["r_was_clamped"] is True
    assert clamped.metadata["ordering"] == "source_order_identity"

    with pytest.raises(ValueError, match="maximum is 0"):
        tome.tome_merge_image(
            hidden,
            keys,
            r=1,
            protected_indices=(0, 1),
            clamp_r=False,
        )


def test_size_weighted_merge_is_differentiable() -> None:
    torch = _torch()
    tome = _tome()
    keys = torch.tensor(
        [[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]],
        dtype=torch.double,
    )
    hidden = torch.tensor(
        [[0.2, -0.1], [0.4, 0.3], [0.7, -0.8], [0.5, 0.9]],
        dtype=torch.double,
        requires_grad=True,
    )
    sizes = torch.tensor(
        [[1.1], [1.4], [0.8], [1.7]],
        dtype=torch.double,
        requires_grad=True,
    )

    def merged_values(values, masses):
        return tome.tome_merge_image(
            values,
            keys,
            r=1,
            token_sizes=masses,
        ).hidden_states

    assert torch.autograd.gradcheck(
        merged_values,
        (hidden, sizes),
        eps=1e-6,
        atol=1e-4,
        rtol=1e-3,
    )
    loss = merged_values(hidden, sizes).square().sum()
    loss.backward()
    assert hidden.grad is not None and bool(torch.isfinite(hidden.grad).all())
    assert sizes.grad is not None and bool(torch.isfinite(sizes.grad).all())


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"r": -1}, "non-negative"),
        ({"r": True}, "integer"),
        ({"r": 1, "protected_indices": (4,)}, "outside"),
        ({"r": 1, "token_sizes": [1.0, 0.0, 1.0, 1.0]}, "positive"),
    ],
)
def test_invalid_contract_inputs_fail_loudly(kwargs: dict, message: str) -> None:
    torch = _torch()
    tome = _tome()
    hidden = torch.ones(4, 2)
    keys = torch.ones(4, 2)
    with pytest.raises((TypeError, ValueError), match=message):
        tome.tome_merge_image(hidden, keys, **kwargs)


def test_empty_batch_is_rejected_before_matching() -> None:
    torch = _torch()
    tome = _tome()
    with pytest.raises(ValueError, match="at least one sequence"):
        tome.tome_merge_image(
            torch.empty(0, 4, 2),
            torch.empty(0, 4, 2),
            r=1,
        )
