from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from mllm_comp.compressors.image.aim import AIMSpec, aim_image_tokens
from mllm_comp.compressors.image.gap import GAPSpec, gap_image_tokens
from mllm_comp.compressors.image.prumerge import PruMergeSpec, prumerge_image_tokens
from mllm_comp.compressors.image.saint import SAINTSpec, saint_image_tokens


def _received_attention(values: list[float]) -> torch.Tensor:
    return torch.tensor(values, dtype=torch.float32).repeat(len(values), 1)


def test_saint_reaches_aligned_target_and_preserves_source_order() -> None:
    tokens = torch.arange(16 * 4, dtype=torch.float32).reshape(16, 4)
    keys = torch.tensor(
        [[1.0, float(index % 3), float(index), 0.5] for index in range(16)]
    )
    result = saint_image_tokens(
        tokens,
        keys,
        spec=SAINTSpec(keep_ratio=0.25, group_size=4, similarity_threshold=0.2),
    )

    assert result.tokens.shape == (4, 4)
    assert result.selected_indices.tolist() == sorted(result.selected_indices.tolist())
    assert result.layout_delta.source_count == 16
    assert result.layout_delta.result_count == 4


def test_aim_tracks_merge_groups_without_detaching_hidden_states() -> None:
    generator = torch.Generator().manual_seed(11)
    tokens = torch.randn(12, 6, generator=generator, requires_grad=True)
    result = aim_image_tokens(tokens, spec=AIMSpec(retain_ratio=0.25))

    assert result.tokens.shape == (3, 6)
    assert sum(result.token_sizes.tolist()) == pytest.approx(12.0)
    assert all(target is not None for target in result.layout_delta.old_to_new.values())
    result.tokens.square().sum().backward()
    assert tokens.grad is not None
    assert torch.isfinite(tokens.grad).all()
    assert float(tokens.grad.abs().sum()) > 0.0


def test_prumerge_uses_released_unnormalized_merge_equations() -> None:
    tokens = torch.tensor(
        [[1.0, 0.0], [0.0, 2.0], [3.0, 0.0], [0.0, 4.0]]
    )
    attention = _received_attention([4.0, 3.0, 2.0, 1.0])
    keys = torch.tensor([[1.0, 0.0], [0.9, 0.0], [0.0, 1.0], [-1.0, 0.0]])

    result = prumerge_image_tokens(
        tokens,
        attention,
        keys,
        spec=PruMergeSpec(variant="base", target_count=2, cluster_size=1),
    )

    assert result.metadata["center_indices"] == (0,)
    assert result.metadata["neighbour_indices"] == ((1,),)
    torch.testing.assert_close(result.tokens[0], tokens[0] + tokens[1] * 3.0)
    torch.testing.assert_close(
        result.tokens[1], tokens[1] * 3.0 + tokens[2] * 2.0 + tokens[3]
    )


def test_gap_applies_ratio_to_prunable_tokens_and_restores_source_order() -> None:
    tokens = torch.arange(14, dtype=torch.float32).view(7, 2)
    result = gap_image_tokens(
        tokens,
        _received_attention([1.0, 9.0, 2.0, 8.0, 3.0, 7.0, 0.0]),
        spec=GAPSpec(retain_ratio=0.5),
        protected_indices=(6,),
    )

    assert result.indices.tolist() == [1, 3, 5, 6]
    assert result.layout_delta.result_layout["position_policy"] == "preserve_original"
