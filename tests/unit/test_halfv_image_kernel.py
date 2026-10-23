from __future__ import annotations

import math

import pytest

torch = pytest.importorskip("torch")

from mllm_comp.compressors.image.halfv import (  # noqa: E402
    HalfVSpec,
    halfv_anchor_cover_select,
    halfv_image_tokens,
    halfv_last_text_attention_scores,
)


def test_released_qwen25_vl_recipe_and_native_boundaries() -> None:
    spec = HalfVSpec()

    assert spec.boundary_after_layer_indices == (1, 20)
    assert spec.ivr_retention_ratio == 0.25
    assert spec.anchor_ratio == 0.1
    assert spec.ssr_retention_ratio == 0.05
    assert spec.ivr_use_rope is False
    assert spec.ssr_use_rope is True
    assert spec.position_policy == "preserve_original"
    spec.validate_decoder_depth(24)


@pytest.mark.parametrize(
    ("kwargs", "error"),
    (
        ({"ivr_layer_index": True}, "integer"),
        ({"ssr_layer_index": 2}, "follow"),
        ({"ivr_retention_ratio": math.nan}, "finite"),
        ({"anchor_ratio": 1.1}, r"\[0, 1\]"),
        ({"ssr_use_rope": False}, "requires RoPE"),
        ({"position_policy": "contiguous"}, "preserve_original"),
    ),
)
def test_spec_rejects_non_release_contracts(
    kwargs: dict[str, object],
    error: str,
) -> None:
    with pytest.raises((TypeError, ValueError), match=error):
        HalfVSpec(**kwargs)


def test_scorer_is_unscaled_head_softmax_then_head_mean() -> None:
    query = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]])
    image_keys = torch.tensor(
        [
            [[1.0, 0.0], [1.0, 0.0]],
            [[0.0, 1.0], [0.0, 1.0]],
            [[0.0, 0.0], [0.0, 0.0]],
        ]
    )

    scores = halfv_last_text_attention_scores(query, image_keys)
    denominator = math.e + 2.0
    expected = torch.tensor(
        [
            (math.e + 1.0) / (2.0 * denominator),
            (math.e + 1.0) / (2.0 * denominator),
            1.0 / denominator,
        ]
    )

    assert scores.dtype == torch.float32
    assert torch.allclose(scores, expected)
    assert torch.allclose(scores.sum(), torch.tensor(1.0))


def test_anchor_cover_keeps_attention_anchor_then_farthest_cover() -> None:
    features = torch.tensor(
        [
            [1.0, 0.0],
            [0.99, 0.01],
            [-1.0, 0.0],
            [0.0, -1.0],
        ]
    )

    selected, anchor_count = halfv_anchor_cover_select(
        features,
        torch.tensor([9.0, 1.0, 0.0, 0.0]),
        target_count=3,
        anchor_ratio=0.34,
    )

    assert anchor_count == 1
    assert selected.tolist() == [0, 2, 3]


def test_ivr_then_ssr_use_current_and_original_visual_budgets() -> None:
    hidden = torch.arange(22, dtype=torch.float32).reshape(11, 2)
    selection_hidden = hidden.clone()
    selection_hidden[2:10] = torch.tensor(
        [
            [1.0, 0.0],
            [0.99, 0.01],
            [0.0, 1.0],
            [0.0, 0.99],
            [-1.0, 0.0],
            [-0.99, 0.01],
            [0.0, -1.0],
            [0.01, -0.99],
        ]
    )
    spec = HalfVSpec(
        ivr_layer_index=2,
        ivr_retention_ratio=0.25,
        anchor_ratio=0.5,
        ssr_layer_index=4,
        ssr_retention_ratio=0.125,
    )

    ivr = halfv_image_tokens(
        hidden,
        image_indices=torch.arange(2, 10),
        boundary_after_layer=1,
        image_attention_scores=torch.tensor(
            [9.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        ),
        initial_visual_tokens=8,
        selection_hidden_states=selection_hidden,
        spec=spec,
        source_layout={"position_policy": "native"},
        result_layout={"position_policy": "preserve_original"},
    )

    assert ivr.metadata["stage"] == "ivr"
    assert ivr.metadata["target_relative_to"] == "current_visual"
    assert ivr.metadata["attention_anchor_count"] == 1
    assert ivr.selected_visual_indices.tolist() == [0, 4]
    assert ivr.kept_image_indices.tolist() == [2, 6]
    assert ivr.kept_indices.tolist() == [0, 1, 2, 6, 10]
    assert ivr.layout_delta.dropped_indices == (3, 4, 5, 7, 8, 9)

    ssr = halfv_image_tokens(
        ivr.hidden_states,
        image_indices=torch.tensor([2, 3]),
        boundary_after_layer=3,
        image_attention_scores=torch.tensor([0.1, 0.9]),
        initial_visual_tokens=8,
        spec=spec,
    )

    assert ssr.metadata["stage"] == "ssr"
    assert ssr.metadata["target_relative_to"] == "original_visual"
    assert ssr.selected_visual_indices.tolist() == [1]
    assert ssr.kept_image_indices.tolist() == [3]
    assert ssr.kept_indices.tolist() == [0, 1, 3, 4]
    assert ssr.layout_delta.dropped_indices == (2,)
    assert ssr.metadata["result_image_tokens"] == 1
