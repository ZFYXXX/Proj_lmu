from __future__ import annotations

import pytest

from mllm_comp.backbones.base import ProcessedBatch
from mllm_comp.backbones.teacher_forcing import (
    OBJECTIVE_NAME,
    build_teacher_forced_batch,
)
from mllm_comp.core.errors import BackendIntegrationError


def _batch(ids: list[int], mask: list[int] | None = None) -> ProcessedBatch:
    return ProcessedBatch(
        model_inputs={
            "input_ids": [ids],
            "attention_mask": [mask or [1] * len(ids)],
        },
        sample_ids=["sample"],
        prompt_lengths=[sum(mask or [1] * len(ids))],
        metadata={"provider": "fixture"},
    )


def test_teacher_forcing_labels_only_non_special_completion_tokens() -> None:
    result = build_teacher_forced_batch(
        prompt_batch=_batch([1, 2, 3]),
        full_batch=_batch([1, 2, 3, 20, 99]),
        special_token_ids=[99],
    )

    assert result.model_inputs["labels"] == [[-100, -100, -100, 20, -100]]
    assert result.metadata["teacher_forcing"] == {
        "objective": OBJECTIVE_NAME,
        "ignore_index": -100,
        "normalization": "mean_over_answer_tokens",
        "prefix_token_counts": [3],
        "completion_token_counts": [2],
        "content_completion_token_counts": [2],
        "answer_token_counts": [1],
        "answer_token_positions_by_sample": [(3,)],
        "special_token_ids_masked": [99],
        "completion_end_token_ids": [],
    }


def test_teacher_forcing_stops_at_assistant_message_end() -> None:
    result = build_teacher_forced_batch(
        prompt_batch=_batch([1, 2]),
        full_batch=_batch([1, 2, 20, 21, 99, 30]),
        special_token_ids=[99],
        completion_end_token_ids=[99],
    )

    assert result.model_inputs["labels"] == [[-100, -100, 20, 21, -100, -100]]
    objective = result.metadata["teacher_forcing"]
    assert objective["completion_token_counts"] == [4]
    assert objective["content_completion_token_counts"] == [2]
    assert objective["answer_token_counts"] == [2]
    assert objective["answer_token_positions_by_sample"] == [(2, 3)]


def test_teacher_forcing_uses_valid_tokens_with_padding() -> None:
    result = build_teacher_forced_batch(
        prompt_batch=_batch([0, 1, 2], [0, 1, 1]),
        full_batch=_batch([0, 1, 2, 20], [0, 1, 1, 1]),
    )

    assert result.model_inputs["labels"] == [[-100, -100, -100, 20]]
    assert result.metadata["teacher_forcing"]["prefix_token_counts"] == [2]


@pytest.mark.parametrize(
    ("prompt", "full", "message"),
    [
        ([1, 2], [1, 7, 20], "changed the native prompt prefix"),
        ([1, 2], [1, 2, 99], "no non-special answer tokens"),
    ],
)
def test_teacher_forcing_rejects_ambiguous_objectives(
    prompt: list[int],
    full: list[int],
    message: str,
) -> None:
    with pytest.raises(BackendIntegrationError, match=message):
        build_teacher_forced_batch(
            prompt_batch=_batch(prompt),
            full_batch=_batch(full),
            special_token_ids=[99],
        )
