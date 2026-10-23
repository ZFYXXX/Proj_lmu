from __future__ import annotations

from dataclasses import dataclass, field

import pytest


torch = pytest.importorskip("torch")

from mllm_comp.compressors.image.nuwa import (  # noqa: E402
    NuwaSpec,
    build_nuwa_prefill_position_ids,
    merge_nuwa_executor_config,
    nuwa_stage1_image_tokens,
    nuwa_stage2_image_tokens,
    nuwa_visual_salience,
    regional_candidate_indices,
    relative_position_mapping,
)


@dataclass
class _ConfigSource:
    config: dict[str, object] = field(default_factory=dict)
    operation: dict[str, object] = field(default_factory=dict)
    criterion: dict[str, object] = field(default_factory=dict)
    schedule: dict[str, object] = field(default_factory=dict)


def test_nuwa_config_is_explicit_and_depth_checked() -> None:
    spec = NuwaSpec()
    assert spec.stage1_total_tokens == 582
    assert spec.stage2_after_layer == 16
    spec.validate_decoder_depth(32)
    with pytest.raises(ValueError, match="Unknown Nüwa"):
        merge_nuwa_executor_config(
            stage=_ConfigSource(config={"retain_rato": 0.5}),
            binding_config=_ConfigSource(),
        )
    with pytest.raises(ValueError, match="precede"):
        spec.validate_decoder_depth(16)


def test_regional_candidates_include_partial_odd_borders() -> None:
    candidates = regional_candidate_indices(
        torch.arange(9, dtype=torch.float32),
        grid_height=3,
        grid_width=3,
        region_size=2,
        top_n_per_region=1,
    )
    assert candidates.tolist() == [4, 5, 7, 8]


def test_visual_salience_norms_each_complete_multihead_token_key() -> None:
    keys = torch.tensor(
        [
            [[10.0], [6.0]],
            [[0.0], [6.0]],
        ]
    )
    salience, key_norms = nuwa_visual_salience(torch.ones(2), keys)

    torch.testing.assert_close(
        key_norms,
        torch.tensor([10.0, 72.0**0.5]),
    )
    assert int(salience.argmax().item()) == 0


def test_stage1_selects_source_order_and_applies_released_self_edge() -> None:
    aggregation = torch.arange(18, dtype=torch.float32).reshape(9, 2)
    semantic = torch.eye(9, dtype=torch.float32)
    attention = torch.diag(torch.arange(1, 10, dtype=torch.float32))
    keys = torch.ones(1, 9, 2)
    result = nuwa_stage1_image_tokens(
        aggregation,
        semantic,
        attention,
        keys,
        grid_height=3,
        grid_width=3,
        spec=NuwaSpec(
            stage1_total_tokens=4,
            region_size=2,
            top_n_per_region=1,
            semantic_hidden_start=1,
            semantic_hidden_end=2,
        ),
    )
    assert result.selected_indices.tolist() == [4, 5, 7, 8]
    assert result.layout_delta.kept_indices == (4, 5, 7, 8)
    assert result.tokens.shape == (4, 2)
    assert torch.isfinite(result.tokens).all()
    assert result.metadata["pillar_tokens"] == 4
    # Every pillar is a unit self edge and therefore preserves its own value.
    assert torch.equal(result.tokens, aggregation.index_select(0, result.selected_indices))


def test_stage1_supports_rectangular_native_qwen_grid() -> None:
    aggregation = torch.arange(24, dtype=torch.float32).reshape(6, 4)
    semantic = aggregation.flip(0)
    attention = torch.ones(2, 6, 6)
    keys = torch.arange(24, dtype=torch.float32).reshape(1, 6, 4).add(1)
    result = nuwa_stage1_image_tokens(
        aggregation,
        semantic,
        attention,
        keys,
        grid_height=2,
        grid_width=3,
        spec=NuwaSpec(
            stage1_total_tokens=2,
            semantic_hidden_start=1,
            semantic_hidden_end=2,
        ),
    )
    assert result.tokens.shape == (2, 4)
    assert result.selected_indices.tolist() == sorted(result.selected_indices.tolist())


def test_stage2_released_attention_selects_exact_visual_tokens() -> None:
    hidden = torch.arange(32, dtype=torch.float32).reshape(8, 4)
    result = nuwa_stage2_image_tokens(
        hidden,
        image_indices=torch.tensor([2, 3, 4, 5]),
        image_attention_scores=torch.tensor([0.1, 0.8, 0.2, 0.7]),
        spec=NuwaSpec(
            stage2_retain_ratio=0.5,
            semantic_hidden_start=1,
            semantic_hidden_end=2,
        ),
    )
    assert result.selected_visual_indices.tolist() == [1, 3]
    assert result.kept_image_indices.tolist() == [3, 5]
    assert result.kept_indices.tolist() == [0, 1, 3, 5, 6, 7]
    assert result.layout_delta.dropped_indices == (2, 4)


def test_stage2_random_selection_preserves_global_rng() -> None:
    hidden = torch.randn(10, 4)
    before = torch.random.get_rng_state().clone()
    first = nuwa_stage2_image_tokens(
        hidden,
        image_indices=torch.arange(2, 8),
        image_attention_scores=None,
        spec=NuwaSpec(
            stage2_retain_ratio=0.5,
            random_stage2=True,
            semantic_hidden_start=1,
            semantic_hidden_end=2,
        ),
    )
    after = torch.random.get_rng_state()
    second = nuwa_stage2_image_tokens(
        hidden,
        image_indices=torch.arange(2, 8),
        image_attention_scores=None,
        spec=NuwaSpec(
            stage2_retain_ratio=0.5,
            random_stage2=True,
            semantic_hidden_start=1,
            semantic_hidden_end=2,
        ),
    )
    assert torch.equal(before, after)
    assert torch.equal(first.selected_visual_indices, second.selected_visual_indices)


def test_rpme_spans_original_visual_range_and_shifts_suffix() -> None:
    selected = torch.tensor([2, 5, 9])
    assert relative_position_mapping(
        selected, original_length=10, visual_start=3
    ).tolist() == [3, 8, 12]
    positions = build_nuwa_prefill_position_ids(
        8,
        visual_span=(2, 5),
        selected_indices=selected,
        original_visual_length=10,
        device=torch.device("cpu"),
    )
    assert positions.tolist() == [0, 1, 2, 7, 11, 12, 13, 14]
