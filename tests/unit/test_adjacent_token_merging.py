"""Contracts for the paper-default encoder-level A-ToMe method."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from mllm_comp.compressors.base import CompressionContext
from mllm_comp.compressors.encoder_level import (
    AdjacentTokenMergingAudio,
    AdjacentTokenMergingSpec,
    adjacent_merge_count,
    adjacent_merge_output_length,
    qwen_audio_chunk_lengths_after_cnn,
    qwen_planned_a_tome_lengths,
)
from mllm_comp.core.state import ForwardState, TokenSpan
from mllm_comp.core.vocabulary import ExecutionPhase, Placement


def test_paper_defaults_resolve_every_three_layers() -> None:
    spec = AdjacentTokenMergingSpec.from_config({})

    assert spec.merge_ratio == 0.10
    assert spec.first_layer == 2
    assert spec.layer_interval == 3
    assert spec.resolve_layer_indices(18) == (2, 5, 8, 11, 14, 17)
    assert spec.resolve_layer_indices(32) == (
        2,
        5,
        8,
        11,
        14,
        17,
        20,
        23,
        26,
        29,
    )


def test_merge_budget_is_bounded_and_uses_current_length() -> None:
    assert adjacent_merge_count(100, 0.10) == 10
    assert adjacent_merge_output_length(100, 0.10) == 90
    assert adjacent_merge_count(5, 0.50) == 2
    assert adjacent_merge_output_length(5, 0.50) == 3
    assert adjacent_merge_output_length(9, 0.10) == 9
    with pytest.raises(ValueError, match="merge_ratio"):
        adjacent_merge_count(10, 0.6)


def test_qwen_length_plan_keeps_chunks_independent() -> None:
    assert qwen_audio_chunk_lengths_after_cnn(200, 100) == [100]
    initial, final = qwen_planned_a_tome_lengths(
        200,
        n_window=100,
        merge_ratio=0.10,
        active_layer_count=6,
    )

    assert initial == [100]
    assert final == [54]

    chunked_initial, chunked_final = qwen_planned_a_tome_lengths(
        400,
        n_window=100,
        merge_ratio=0.10,
        active_layer_count=1,
    )
    assert chunked_initial == [100, 100]
    assert chunked_final == [90, 90]


def test_plugin_plans_encoder_merge_and_compacted_llm_length() -> None:
    encoder = SimpleNamespace(
        n_window=100,
        layers=[object() for _ in range(18)],
    )
    state = ForwardState(
        audio_feature_lengths=[200],
        modality_token_spans=[
            TokenSpan(sample_index=0, start=2, end=52, modality="audio")
        ],
    )
    plugin = AdjacentTokenMergingAudio()

    plan = plugin.plan(
        state=state,
        context=CompressionContext(
            phase="eval",
            model=None,
            structural_handles={"audio_encoder": encoder},
        ),
        config={"_stage_name": "atome", "_execution_phase": "forward"},
    )

    assert plan.placement is Placement.ENCODER_LEVEL
    assert plan.execution_phase is ExecutionPhase.FORWARD
    assert len(plan.token_actions) == 1
    action = plan.token_actions[0]
    assert action.op == "merge"
    assert action.metadata["initial_encoder_count"] == 100
    assert action.metadata["final_encoder_count"] == 54
    assert action.metadata["original_count"] == 50
    assert action.metadata["keep_count"] == 27
    assert action.metadata["active_layers"] == [2, 5, 8, 11, 14, 17]


def test_explicit_layer_scope_overrides_paper_cadence() -> None:
    spec = AdjacentTokenMergingSpec.from_config(
        {"activation_scope": {"layers": [1, 4]}}
    )

    assert spec.resolve_layer_indices(6) == (1, 4)
    with pytest.raises(ValueError, match="outside"):
        spec.resolve_layer_indices(4)
