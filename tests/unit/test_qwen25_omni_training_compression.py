"""Training-input contracts for dynamic Qwen2.5-Omni compression."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from mllm_comp.backbones.qwen25_omni.training_compression import (
    build_qwen25_omni_training_state,
    model_inputs_from_forward_call,
)


torch = pytest.importorskip("torch")


def test_training_state_preserves_native_inputs_and_counts_media_units() -> None:
    config = SimpleNamespace(
        image_token_index=10,
        video_token_index=11,
        audio_token_index=12,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=[2, 3],
        audio_bos_token_id=13,
        audio_eos_token_id=14,
        vision_start_token_id=15,
        vision_end_token_id=16,
    )
    model = SimpleNamespace(thinker=SimpleNamespace(config=config))
    input_ids = torch.tensor(
        [
            [1, 10, 10, 2, 12, 12, 12, 3, 0],
            [4, 11, 11, 12, 12, 5, 0, 0, 0],
        ]
    )
    attention_mask = torch.tensor(
        [
            [1, 1, 1, 1, 1, 1, 1, 1, 0],
            [1, 1, 1, 1, 1, 1, 0, 0, 0],
        ]
    )
    labels = input_ids.clone()
    feature_attention_mask = torch.tensor(
        [
            [1, 1, 1, 1, 0],
            [1, 1, 1, 0, 0],
        ]
    )
    native_inputs = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
        "input_features": torch.ones(2, 4, 5),
        "feature_attention_mask": feature_attention_mask,
        "custom_provider_kwarg": "preserved",
    }

    state = build_qwen25_omni_training_state(
        model,
        native_inputs,
        debug_semantics=True,
    )

    assert set(state.native_model_inputs) == set(native_inputs)
    for name, value in native_inputs.items():
        assert state.native_model_inputs[name] is value
    assert state.audio_feature_lengths.tolist() == [4, 3]
    assert state.modality_metadata["padding_side"] == "right"
    assert state.modality_metadata["eos_token_id"] == 2
    assert state.modality_metadata["special_token_ids"] == [
        0,
        1,
        2,
        3,
        13,
        14,
        15,
        16,
    ]
    assert state.modality_metadata["global_special_token_ids"] == [1]
    assert state.modality_metadata["modality_special_token_ids"]["audio"] == [
        13,
        14,
    ]
    assert state.modality_metadata["vision_special_token_ids"] == [15, 16]
    assert state.modality_metadata["expected_token_counts_by_sample"] == [
        {"image": 2, "audio": 3},
        {"video": 2, "audio": 2},
    ]
    assert state.modality_metadata["modality_counts_by_sample"] == [
        {"image": 1, "audio": 1},
        {"video": 1, "audio": 1},
    ]

    default_state = build_qwen25_omni_training_state(model, native_inputs)
    assert "special_token_ids" not in default_state.modality_metadata


def test_forward_call_binding_keeps_positional_and_variadic_kwargs() -> None:
    def forward(input_ids: object, attention_mask: object = None, **kwargs: object) -> None:
        del input_ids, attention_mask, kwargs

    inputs = model_inputs_from_forward_call(
        forward,
        ("ids",),
        {"attention_mask": "mask", "input_features": "audio"},
    )

    assert inputs == {
        "input_ids": "ids",
        "attention_mask": "mask",
        "input_features": "audio",
    }
