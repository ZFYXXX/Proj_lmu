from __future__ import annotations

from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from mllm_comp.compressors.image.feather import (
    FEATHERSpec,
    feather_image_tokens,
    feather_no_rope_scores,
)
from mllm_comp.compressors.image.ivc_prune import (
    IVCPruneSpec,
    ivc_coordinate_scores,
    ivc_post_image_text_indices,
    ivc_prune_image_tokens,
    project_ivc_value_states,
)


def test_feather_preserves_nonvisual_tokens_and_reaches_two_stage_targets() -> None:
    hidden = torch.arange(24, dtype=torch.float32).view(12, 2)
    image = torch.arange(1, 10)
    scores = torch.arange(9, dtype=torch.float32)
    spec = FEATHERSpec(stage_layers=(2, 4), reduction_ratio=0.7, uniform_stride=3)

    first = feather_image_tokens(
        hidden,
        image_indices=image,
        image_attention_scores=scores,
        before_layer=2,
        initial_visual_tokens=9,
        spec=spec,
    )
    second_image = torch.arange(1, 1 + int(first.kept_image_indices.numel()))
    second = feather_image_tokens(
        first.hidden_states,
        image_indices=second_image,
        image_attention_scores=torch.arange(second_image.numel(), dtype=torch.float32),
        before_layer=4,
        initial_visual_tokens=9,
        spec=spec,
    )

    assert first.metadata["stage"] == 1
    assert first.metadata["uniform_tokens_added"] >= 1
    assert second.metadata["stage"] == 2
    assert second.kept_image_indices.numel() == 1
    assert second.kept_indices[0].item() == 0
    assert second.kept_indices[-1].item() == first.hidden_states.shape[0] - 1


def test_feather_second_stage_keeps_best_optional_tokens_with_protected_budget() -> None:
    hidden = torch.arange(20, dtype=torch.float32).view(10, 2)
    result = feather_image_tokens(
        hidden,
        image_indices=torch.arange(1, 9),
        image_attention_scores=torch.arange(8, dtype=torch.float32),
        before_layer=4,
        initial_visual_tokens=8,
        spec=FEATHERSpec(stage_layers=(2, 4), target_final_retain_ratio=0.5),
        protected_indices=(1, 2, 3),
    )

    # Three protected newlines leave room for only the highest-scoring token.
    assert result.kept_image_indices.tolist() == [1, 2, 3, 8]
    assert result.kept_indices.tolist() == [0, 1, 2, 3, 8, 9]
    torch.testing.assert_close(result.hidden_states, hidden[[0, 1, 2, 3, 8, 9]])


def test_ivc_prune_supports_native_mrope_and_strict_budget() -> None:
    sequence_length = 9
    cosine = torch.arange(3 * sequence_length * 4, dtype=torch.float32).view(
        3, 1, sequence_length, 4
    )
    sine = cosine.flip(2)
    image = torch.arange(1, 7)
    cosine_scores, sine_scores = ivc_coordinate_scores(
        (cosine, sine), image_indices=image, sequence_length=sequence_length
    )

    hidden = torch.arange(sequence_length * 4, dtype=torch.float32).view(
        sequence_length, 4
    )
    result = ivc_prune_image_tokens(
        hidden,
        image_indices=image,
        cosine_scores=cosine_scores,
        sine_scores=sine_scores,
        value_states=hidden,
        text_query_indices=torch.tensor([7, 8]),
        before_layer=3,
        spec=IVCPruneSpec(
            prune_layer=3,
            total_retain_ratio=0.5,
            axis_ratio_each=0.1,
            semantic_seed_ratio=0.01,
            strict_total_budget=True,
        ),
    )

    assert result.kept_image_indices.numel() == 3
    assert result.kept_image_indices.tolist() == sorted(result.kept_image_indices.tolist())
    assert result.layout_delta.result_layout["position_policy"] == "preserve_original"
    mask = torch.tensor([0, 1, 1, 0, 0], dtype=torch.bool)
    assert ivc_post_image_text_indices(mask, skip=1).tolist() == [4]


def test_decoder_projection_helpers_accept_config_owned_head_counts() -> None:
    class Attention(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.config = SimpleNamespace(num_attention_heads=2, num_key_value_heads=1)
            self.head_dim = 2
            self.num_key_value_groups = 2
            self.scaling = self.head_dim**-0.5
            self.q_proj = torch.nn.Linear(4, 4, bias=False)
            self.k_proj = torch.nn.Linear(4, 2, bias=False)
            self.v_proj = torch.nn.Linear(4, 2, bias=False)
            self.q_norm = torch.nn.Identity()
            self.k_norm = torch.nn.Identity()

    layer = SimpleNamespace(input_layernorm=torch.nn.Identity(), self_attn=Attention())
    hidden = torch.arange(20, dtype=torch.float32).view(5, 4)

    scores = feather_no_rope_scores(
        hidden,
        layer,
        image_indices=torch.tensor([1, 2, 3]),
        query_indices=torch.tensor([4]),
    )
    values = project_ivc_value_states(hidden, layer)

    assert scores.shape == (3,)
    torch.testing.assert_close(scores.sum(), torch.tensor(1.0))
    assert values.shape == (5, 2)
