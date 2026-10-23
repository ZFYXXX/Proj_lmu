from __future__ import annotations

import pytest


torch = pytest.importorskip("torch")

from mllm_comp.compressors.image.dart import (  # noqa: E402
    DARTSpec,
    dart_image_tokens,
    dart_post_image_text_indices,
)


def test_dart_rejects_boolean_ratio() -> None:
    with pytest.raises(TypeError, match="real number"):
        DARTSpec(retain_ratio=True)


def test_dart_exact_pivots_select_from_remaining_and_preserve_source_order() -> None:
    normalized = torch.tensor(
        [
            [1.0, 1.0],
            [1.0, 0.0],
            [0.9, 0.1],
            [0.0, 1.0],
            [-1.0, 0.0],
            [0.0, -1.0],
            [1.0, 1.0],
        ]
    )
    hidden = normalized * 3.0
    keys = normalized.clone()
    keys[1] *= 10.0
    keys[5] *= 20.0
    result = dart_image_tokens(
        hidden,
        normalized,
        keys,
        image_indices=torch.arange(1, 5),
        text_candidate_indices=torch.arange(5, 7),
        spec=DARTSpec(
            target_visual_tokens=3,
            pivot_image_tokens=1,
            pivot_text_tokens=1,
        ),
        source_layout={"backbone": "fixture"},
        result_layout={"position_policy": "sparse_source"},
    )

    assert result.image_pivot_indices.tolist() == [1]
    assert result.text_pivot_indices.tolist() == [5]
    assert result.kept_image_indices.tolist() == [1, 3, 4]
    assert result.kept_indices.tolist() == [0, 1, 3, 4, 5, 6]
    assert torch.equal(result.hidden_states, hidden.index_select(0, result.kept_indices))
    assert result.layout_delta.source_count == 7
    assert result.layout_delta.result_count == 6
    assert result.layout_delta.result_layout["position_policy"] == "sparse_source"


def test_dart_text_pivots_are_post_image_prompt_only() -> None:
    image_mask = torch.tensor([False, True, True, True, False, False, False])
    labels = torch.tensor([[-100, -100, -100, -100, -100, 7, 8]])
    candidates = dart_post_image_text_indices(
        image_mask,
        attention_mask=torch.ones(1, 7, dtype=torch.long),
        labels=labels,
    )

    assert candidates.tolist() == [4]
