"""Shared completion-only label construction for analysis objectives."""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Iterable

from mllm_comp.backbones.base import ProcessedBatch
from mllm_comp.core.errors import BackendIntegrationError


IGNORE_INDEX = -100
OBJECTIVE_NAME = "teacher_forced_answer_cross_entropy_v1"


def processor_special_token_ids(processor: Any) -> tuple[int, ...]:
    """Return the tokenizer's declared special ids without assuming HF types."""

    tokenizer = getattr(processor, "tokenizer", processor)
    values = getattr(tokenizer, "all_special_ids", ()) or ()
    return tuple(sorted({int(value) for value in values if value is not None}))


def build_teacher_forced_batch(
    *,
    prompt_batch: ProcessedBatch,
    full_batch: ProcessedBatch,
    special_token_ids: Iterable[int] = (),
    completion_end_token_ids: Iterable[int] = (),
) -> ProcessedBatch:
    """Attach labels for answer tokens after an exact native prompt prefix.

    The implementation deliberately uses a small tensor/list protocol instead
    of importing torch. Production tensors provide ``tolist``, ``clone`` and
    ``fill_``; unit fixtures can use ordinary nested lists.
    """

    prompt_ids = prompt_batch.model_inputs.get("input_ids")
    full_ids = full_batch.model_inputs.get("input_ids")
    prompt_rows = _rows(prompt_ids, field="prompt input_ids")
    full_rows = _rows(full_ids, field="teacher-forced input_ids")
    if len(prompt_rows) != len(full_rows) or not full_rows:
        raise BackendIntegrationError(
            "Teacher-forced prompt/full batches must have the same non-zero batch size."
        )

    prompt_valid = _valid_positions(
        prompt_batch.model_inputs.get("attention_mask"), prompt_rows
    )
    full_valid = _valid_positions(
        full_batch.model_inputs.get("attention_mask"), full_rows
    )
    special_ids = {int(value) for value in special_token_ids}
    end_ids = {int(value) for value in completion_end_token_ids}
    answer_positions: list[list[int]] = []
    content_counts: list[int] = []
    completion_counts: list[int] = []
    prefix_counts: list[int] = []

    for row_index, (prompt_row, full_row) in enumerate(zip(prompt_rows, full_rows)):
        prompt_positions = prompt_valid[row_index]
        full_positions = full_valid[row_index]
        prompt_tokens = [prompt_row[index] for index in prompt_positions]
        full_tokens = [full_row[index] for index in full_positions]
        if len(full_tokens) <= len(prompt_tokens):
            raise BackendIntegrationError(
                "Teacher-forced rendering added no completion tokens for "
                f"sample {row_index}."
            )
        if full_tokens[: len(prompt_tokens)] != prompt_tokens:
            raise BackendIntegrationError(
                "Teacher-forced rendering changed the native prompt prefix for "
                f"sample {row_index}."
            )
        completion = full_positions[len(prompt_tokens) :]
        content_completion: list[int] = []
        for index in completion:
            if full_row[index] in end_ids:
                break
            content_completion.append(index)
        answer = [index for index in content_completion if full_row[index] not in special_ids]
        if not answer:
            raise BackendIntegrationError(
                "Teacher-forced completion contains no non-special answer tokens for "
                f"sample {row_index}."
            )
        prefix_counts.append(len(prompt_tokens))
        completion_counts.append(len(completion))
        content_counts.append(len(content_completion))
        answer_positions.append(answer)

    labels = _ignored_like(full_ids, full_rows)
    for row_index, positions in enumerate(answer_positions):
        for position in positions:
            _set_value(labels, row_index, position, full_rows[row_index][position])

    model_inputs = dict(full_batch.model_inputs)
    model_inputs["labels"] = labels
    metadata = deepcopy(full_batch.metadata)
    metadata["teacher_forcing"] = {
        "objective": OBJECTIVE_NAME,
        "ignore_index": IGNORE_INDEX,
        "normalization": "mean_over_answer_tokens",
        "prefix_token_counts": prefix_counts,
        "completion_token_counts": completion_counts,
        "content_completion_token_counts": content_counts,
        "answer_token_counts": [len(value) for value in answer_positions],
        "answer_token_positions_by_sample": [tuple(value) for value in answer_positions],
        "special_token_ids_masked": sorted(special_ids),
        "completion_end_token_ids": sorted(end_ids),
    }
    return ProcessedBatch(
        model_inputs=model_inputs,
        sample_ids=list(full_batch.sample_ids),
        prompt_lengths=list(full_batch.prompt_lengths),
        metadata=metadata,
    )


def _rows(value: Any, *, field: str) -> list[list[int]]:
    if value is None:
        raise BackendIntegrationError(f"{field} are required for teacher forcing.")
    tolist = getattr(value, "tolist", None)
    raw = tolist() if callable(tolist) else value
    if not isinstance(raw, (list, tuple)) or not raw:
        raise BackendIntegrationError(f"{field} must have shape [batch, sequence].")
    if not isinstance(raw[0], (list, tuple)):
        raw = [raw]
    rows = [[int(item) for item in row] for row in raw]
    if any(not row for row in rows):
        raise BackendIntegrationError(f"{field} cannot contain an empty sequence.")
    return rows


def _valid_positions(mask: Any, rows: list[list[int]]) -> list[list[int]]:
    if mask is None:
        return [list(range(len(row))) for row in rows]
    mask_rows = _rows(mask, field="attention_mask")
    if len(mask_rows) != len(rows):
        raise BackendIntegrationError(
            "Teacher-forced attention_mask batch size does not match input_ids."
        )
    positions: list[list[int]] = []
    for row, mask_row in zip(rows, mask_rows):
        if len(mask_row) != len(row):
            raise BackendIntegrationError(
                "Teacher-forced attention_mask length does not match input_ids."
            )
        positions.append([index for index, value in enumerate(mask_row) if value])
    return positions


def _ignored_like(value: Any, rows: list[list[int]]) -> Any:
    clone = getattr(value, "clone", None)
    if callable(clone):
        labels = clone()
        fill = getattr(labels, "fill_", None)
        if not callable(fill):
            raise BackendIntegrationError(
                "Teacher-forced tensor labels require a fill_ operation."
            )
        fill(IGNORE_INDEX)
        return labels
    return [[IGNORE_INDEX for _ in row] for row in rows]


def _set_value(value: Any, row: int, column: int, token: int) -> None:
    try:
        value[row, column] = token
    except (TypeError, IndexError):
        value[row][column] = token


__all__ = [
    "IGNORE_INDEX",
    "OBJECTIVE_NAME",
    "build_teacher_forced_batch",
    "processor_special_token_ids",
]
