"""Contracts for attention Top-K and progressive encoder pruning."""

from __future__ import annotations

from types import SimpleNamespace

from mllm_comp.compressors.base import CompressionContext
from mllm_comp.compressors.encoder_level import (
    EncoderAttentionTopKAudio,
    EncoderAttentionTopKSpec,
    ProgressiveAttentionPruningAudio,
    ProgressiveAttentionPruningSpec,
    attention_topk_output_length,
    progressive_pruning_output_length,
)
from mllm_comp.core.state import ForwardState, TokenSpan
from mllm_comp.core.vocabulary import ExecutionPhase, Placement


def _state() -> ForwardState:
    return ForwardState(
        audio_feature_lengths=[64],
        modality_token_spans=[
            TokenSpan(sample_index=0, start=1, end=17, modality="audio")
        ],
    )


def _context(num_layers: int = 12) -> CompressionContext:
    encoder = SimpleNamespace(
        layers=[object() for _ in range(num_layers)],
        n_window=16,
    )
    return CompressionContext(
        phase="train",
        model=None,
        structural_handles={"audio_encoder": encoder},
    )


def test_encoder_attention_topk_uses_paper_blocks_and_pre_ffn_site() -> None:
    spec = EncoderAttentionTopKSpec.from_config({})
    assert spec.resolve_layer_indices(12) == (3, 6, 9)
    assert attention_topk_output_length(17, 0.5) == 8

    plan = EncoderAttentionTopKAudio().plan(
        state=_state(),
        context=_context(),
        config={
            "_stage_name": "encoder_topk",
            "_execution_phase": "forward",
            "operation": {
                "type": "keep",
                "target_site": "audio_encoder_layer_post_mhsa_pre_ffn",
            },
            "criterion": {"score_source": "mean_received_self_attention"},
            "schedule": {"keep_ratio": 0.5},
            "activation_scope": {"layers": [0, 1]},
        },
    )

    action = plan.token_actions[0]
    assert plan.placement is Placement.ENCODER_LEVEL
    assert plan.execution_phase is ExecutionPhase.FORWARD
    assert action.metadata["initial_chunk_lengths"] == [16, 16]
    assert action.metadata["final_chunk_lengths"] == [4, 4]
    assert action.metadata["keep_count"] == 4


def test_progressive_pruning_supports_fixed_and_linear_schedules() -> None:
    fixed = ProgressiveAttentionPruningSpec.from_config(
        {"activation_scope": {"layers": [2, 4]}}
    )
    assert [fixed.pruning_rate_for_layer(i, (2, 4)) for i in (2, 4)] == [
        0.1,
        0.1,
    ]
    linear = ProgressiveAttentionPruningSpec.from_config(
        {
            "schedule": {
                "policy": "linear",
                "start_pruning_rate": 0.1,
                "end_pruning_rate": 0.3,
            },
            "activation_scope": {"layers": [1, 3, 5]},
        }
    )
    rates = [linear.pruning_rate_for_layer(i, (1, 3, 5)) for i in (1, 3, 5)]
    assert rates == [0.1, 0.2, 0.3]
    assert progressive_pruning_output_length(10, 0.2) == 8


def test_qwen_progressive_three_stage_schedule_matches_recorded_budget() -> None:
    layers = (6, 12, 18)
    spec = ProgressiveAttentionPruningSpec.from_config(
        {
            "schedule": {
                "policy": "linear",
                "start_pruning_rate": 0.05,
                "end_pruning_rate": 0.20,
                "minimum_keep_tokens": 2,
            },
            "activation_scope": {"layers": list(layers)},
        }
    )

    length = 50
    trajectory: list[int] = []
    for layer in layers:
        length = progressive_pruning_output_length(
            length,
            spec.pruning_rate_for_layer(layer, layers),
            minimum_keep_tokens=spec.minimum_keep_tokens,
        )
        trajectory.append(length)

    assert trajectory == [47, 41, 32]


def test_progressive_plan_reduces_lengths_between_blocks() -> None:
    plan = ProgressiveAttentionPruningAudio().plan(
        state=_state(),
        context=_context(4),
        config={
            "_stage_name": "progressive",
            "operation": {
                "type": "keep",
                "target_site": "audio_encoder_layer_output_post_block",
            },
            "criterion": {"score_source": "mean_received_self_attention"},
            "schedule": {"policy": "fixed", "pruning_rate": 0.5},
            "activation_scope": {"layers": [0, 1]},
        },
    )

    action = plan.token_actions[0]
    assert action.metadata["initial_chunk_lengths"] == [16, 16]
    assert action.metadata["final_chunk_lengths"] == [4, 4]
    assert action.metadata["pruning_rates"] == [0.5, 0.5]
    assert action.metadata["keep_count"] == 4
    assert plan.object_subtype == "audio_encoder_layer_output_post_block"
