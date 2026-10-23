"""Opt-in token-count diagnostics for Qwen2.5-Omni.

The collector records shapes and modality counts only. It never retains hidden
states, logits, attention matrices, or KV-cache tensors.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from contextlib import contextmanager
from copy import deepcopy
from math import prod
from typing import Any, Iterator

from mllm_comp.backbones.base import ProcessedBatch
from mllm_comp.core.errors import InvariantError
from mllm_comp.core.state import ForwardState
from mllm_comp.core.vocabulary import Placement


TOKEN_TYPES: tuple[str, ...] = (
    "system_prompt",
    "sys_special",
    "instruction",
    "ins_special",
    "response",
    "resp_special",
    "audio",
    "audio_special",
    "image",
    "image_special",
    "video",
    "video_special",
    "global_special",
    "pad",
    "unknown",
)

_MESSAGE_SPECIAL_BY_CONTENT: dict[str, str] = {
    "system_prompt": "sys_special",
    "instruction": "ins_special",
    "response": "resp_special",
}
_MODALITY_SPECIAL_BY_CONTENT: dict[str, str] = {
    "audio": "audio_special",
    "image": "image_special",
    "video": "video_special",
}


def _empty_counts() -> dict[str, int]:
    return {name: 0 for name in TOKEN_TYPES}


def _shape(value: Any) -> tuple[int, ...] | None:
    shape = getattr(value, "shape", None)
    if shape is not None:
        try:
            return tuple(int(dim) for dim in shape)
        except (TypeError, ValueError):
            return None
    if isinstance(value, (list, tuple)):
        if not value:
            return (0,)
        child = _shape(value[0])
        return (len(value), *(child or ()))
    return None


def _rows(value: Any) -> list[list[int]]:
    tolist = getattr(value, "tolist", None)
    raw = tolist() if callable(tolist) else value
    if raw is None:
        return []
    if not isinstance(raw, (list, tuple)):
        return [[int(raw)]]
    if not raw:
        return []
    if not isinstance(raw[0], (list, tuple)):
        return [[int(item) for item in raw]]
    return [[int(item) for item in row] for row in raw]


def _tolist(value: Any) -> Any:
    tolist = getattr(value, "tolist", None)
    return tolist() if callable(tolist) else value


def _normalized_semantic_rows(value: Any) -> list[list[str]] | None:
    if not isinstance(value, list):
        return None
    rows: list[list[str]] = []
    for row in value:
        if not isinstance(row, list):
            return None
        rows.append([str(item) for item in row])
    return rows


def _merge_counts(items: Sequence[Mapping[str, int]]) -> dict[str, int]:
    merged = _empty_counts()
    for item in items:
        for name in TOKEN_TYPES:
            merged[name] += int(item.get(name, 0))
    return merged


def _sample_record(sample_id: str | int, counts: Mapping[str, int]) -> dict[str, Any]:
    unexpected = set(counts) - set(TOKEN_TYPES)
    if unexpected:
        raise InvariantError(
            f"Debug token counts contain unsupported fields: {sorted(unexpected)}."
        )
    normalized = _empty_counts()
    normalized.update({name: int(value) for name, value in counts.items()})
    if any(value < 0 for value in normalized.values()):
        raise InvariantError("Debug token counts cannot be negative.")
    return {
        "sample_id": sample_id,
        "total_tokens": sum(normalized.values()),
        "token_counts": normalized,
    }


def _llm_snapshot_from_expected(
    *,
    stage: str,
    input_ids: Any,
    attention_mask: Any,
    sample_ids: Sequence[str | int],
    expected_by_sample: Sequence[Mapping[str, Any]],
    token_domain: str = "llm_sequence",
) -> dict[str, Any]:
    input_shape = _shape(input_ids)
    if input_shape is None or len(input_shape) != 2:
        raise InvariantError(f"{stage} debug snapshot requires rank-2 input_ids.")
    batch_size, width = input_shape
    if batch_size != len(sample_ids):
        raise InvariantError(
            f"{stage} debug snapshot batch size {batch_size} does not match "
            f"{len(sample_ids)} sample ids."
        )
    mask_rows = _rows(attention_mask)
    records: list[dict[str, Any]] = []
    for sample_index, sample_id in enumerate(sample_ids):
        valid = sum(mask_rows[sample_index]) if sample_index < len(mask_rows) else width
        expected = (
            expected_by_sample[sample_index]
            if sample_index < len(expected_by_sample)
            else {}
        )
        counts = _empty_counts()
        for modality in ("audio", "image", "video"):
            counts[modality] = int(expected.get(modality, 0))
        media = counts["audio"] + counts["image"] + counts["video"]
        if media > valid:
            raise InvariantError(
                f"{stage} sample {sample_id!r} has {media} media tokens but only "
                f"{valid} valid sequence positions."
            )
        counts["unknown"] = max(0, int(valid) - media)
        counts["pad"] = max(0, width - int(valid))
        classified = sum(counts.values())
        if classified < width:
            counts["unknown"] = width - classified
        records.append(_sample_record(sample_id, counts))
    return {
        "stage": stage,
        "token_domain": token_domain,
        "total_tokens": batch_size * width,
        "token_counts": _merge_counts(
            [record["token_counts"] for record in records]
        ),
        "per_sample": records,
    }


def _llm_snapshot_from_semantic_labels(
    *,
    stage: str,
    input_ids: Any,
    attention_mask: Any,
    sample_ids: Sequence[str | int],
    semantic_rows: Sequence[Sequence[str]],
    classification_source: str,
    token_domain: str = "llm_sequence",
) -> dict[str, Any]:
    input_shape = _shape(input_ids)
    if input_shape is None or len(input_shape) != 2:
        raise InvariantError(f"{stage} debug snapshot requires rank-2 input_ids.")
    batch_size, width = input_shape
    if batch_size != len(sample_ids) or batch_size != len(semantic_rows):
        raise InvariantError(
            f"{stage} semantic labels do not match batch size {batch_size}."
        )
    mask_rows = _rows(attention_mask)
    records: list[dict[str, Any]] = []
    for sample_index, sample_id in enumerate(sample_ids):
        labels = list(semantic_rows[sample_index])
        if len(labels) != width:
            raise InvariantError(
                f"{stage} sample {sample_id!r} has {len(labels)} semantic labels "
                f"for sequence width {width}."
            )
        mask = mask_rows[sample_index] if sample_index < len(mask_rows) else [1] * width
        counts = _empty_counts()
        for position, label in enumerate(labels):
            if position < len(mask) and not int(mask[position]):
                counts["pad"] += 1
            elif label in counts:
                counts[label] += 1
            else:
                counts["unknown"] += 1
        records.append(_sample_record(sample_id, counts))
    return {
        "stage": stage,
        "token_domain": token_domain,
        "classification_source": classification_source,
        "total_tokens": batch_size * width,
        "token_counts": _merge_counts(
            [record["token_counts"] for record in records]
        ),
        "per_sample": records,
    }


def _llm_snapshot_from_state(
    state: ForwardState,
    *,
    stage: str,
    sample_ids: Sequence[str | int],
) -> dict[str, Any]:
    input_shape = _shape(state.input_ids)
    if input_shape is None or len(input_shape) != 2:
        raise InvariantError(f"{stage} debug snapshot requires rank-2 input_ids.")
    _, width = input_shape
    expected = [_empty_counts() for _ in sample_ids]
    for span in state.modality_token_spans:
        if not 0 <= span.sample_index < len(expected):
            raise InvariantError(
                f"Debug token span sample index {span.sample_index} is outside the batch."
            )
        if span.modality in expected[span.sample_index]:
            expected[span.sample_index][span.modality] += max(
                0,
                min(width, int(span.end)) - max(0, int(span.start)),
            )
    return _llm_snapshot_from_expected(
        stage=stage,
        input_ids=state.input_ids,
        attention_mask=state.attention_mask,
        sample_ids=sample_ids,
        expected_by_sample=expected,
    )


def _audio_feature_snapshot(
    processed_batch: ProcessedBatch,
) -> dict[str, Any]:
    sample_ids = processed_batch.sample_ids
    mask_rows = _rows(
        processed_batch.model_inputs.get("feature_attention_mask")
    )
    modality_counts = processed_batch.metadata.get("modality_counts_by_sample", [])
    records: list[dict[str, Any]] = []
    row_offset = 0
    for sample_index, sample_id in enumerate(sample_ids):
        item_count = 0
        if sample_index < len(modality_counts):
            item_count = int(modality_counts[sample_index].get("audio", 0))
        rows = mask_rows[row_offset : row_offset + item_count]
        row_offset += item_count
        counts = _empty_counts()
        counts["audio"] = sum(sum(row) for row in rows)
        counts["pad"] = sum(len(row) - sum(row) for row in rows)
        records.append(_sample_record(sample_id, counts))
    if row_offset != len(mask_rows):
        raise InvariantError(
            "Debug audio feature rows do not match modality_counts_by_sample: "
            f"consumed {row_offset}, found {len(mask_rows)}."
        )
    return {
        "stage": "audio_feature_input",
        "token_domain": "audio_feature_frame",
        "total_tokens": sum(record["total_tokens"] for record in records),
        "token_counts": _merge_counts(
            [record["token_counts"] for record in records]
        ),
        "per_sample": records,
    }


def _encoded_input_ids(encoded: Any) -> list[int] | None:
    if isinstance(encoded, Mapping):
        value = encoded.get("input_ids")
    else:
        value = getattr(encoded, "input_ids", None)
    input_ids = _tolist(value)
    if (
        isinstance(input_ids, list)
        and input_ids
        and isinstance(input_ids[0], list)
    ):
        input_ids = input_ids[0]
    if not isinstance(input_ids, list):
        return None
    try:
        return [int(value) for value in input_ids]
    except (TypeError, ValueError):
        return None


def _tokenizer_input_ids(tokenizer: Any, text: str) -> list[int] | None:
    if not callable(tokenizer):
        return None
    try:
        encoded = tokenizer(text, add_special_tokens=False)
    except (NotImplementedError, TypeError, ValueError):
        return None
    return _encoded_input_ids(encoded)


def _tokenizer_template_payload(
    tokenizer: Any,
    rendered_text: str,
) -> tuple[list[int], list[tuple[int, int]] | None] | None:
    if not callable(tokenizer):
        return None
    try:
        encoded = tokenizer(
            rendered_text,
            add_special_tokens=False,
            return_offsets_mapping=True,
        )
    except (NotImplementedError, TypeError, ValueError):
        input_ids = _tokenizer_input_ids(tokenizer, rendered_text)
        return None if input_ids is None else (input_ids, None)
    input_ids = _encoded_input_ids(encoded)
    if input_ids is None:
        return None
    offsets = (
        _tolist(encoded.get("offset_mapping"))
        if isinstance(encoded, Mapping)
        else _tolist(getattr(encoded, "offset_mapping", None))
    )
    if (
        isinstance(offsets, list)
        and offsets
        and isinstance(offsets[0], list)
        and offsets[0]
        and isinstance(offsets[0][0], (list, tuple))
    ):
        offsets = offsets[0]
    if not isinstance(offsets, list):
        return input_ids, None
    if len(input_ids) != len(offsets):
        return None
    try:
        return (
            input_ids,
            [(int(item[0]), int(item[1])) for item in offsets],
        )
    except (IndexError, TypeError, ValueError):
        return None


def _message_region_labels(
    *,
    tokenizer: Any,
    template_ids: Sequence[int],
    message_prefixes: Sequence[Mapping[str, Any]],
    special_token_ids: set[int],
    global_special_token_ids: set[int],
) -> list[str] | None:
    labels = [
        "global_special" if int(token_id) in special_token_ids else "unknown"
        for token_id in template_ids
    ]
    start = 0
    for prefix in message_prefixes:
        prefix_ids = _tokenizer_input_ids(tokenizer, str(prefix.get("text", "")))
        if (
            prefix_ids is None
            or len(prefix_ids) < start
            or len(prefix_ids) > len(template_ids)
            or list(template_ids[: len(prefix_ids)]) != prefix_ids
        ):
            return None
        content_label = str(prefix.get("label", ""))
        special_label = _MESSAGE_SPECIAL_BY_CONTENT.get(content_label)
        if special_label is None:
            return None
        labels[start : len(prefix_ids)] = [special_label] * (
            len(prefix_ids) - start
        )
        start = len(prefix_ids)
    if start < len(template_ids):
        labels[start:] = ["resp_special"] * (len(template_ids) - start)
    for index, token_id in enumerate(template_ids):
        if int(token_id) in global_special_token_ids:
            labels[index] = "global_special"
    return labels


def _modality_special_sets(
    metadata: Mapping[str, Any],
) -> tuple[dict[int, str], set[int]]:
    explicit: dict[int, str] = {}
    configured = metadata.get("modality_special_token_ids", {})
    if isinstance(configured, Mapping):
        for modality, values in configured.items():
            if str(modality) not in _MODALITY_SPECIAL_BY_CONTENT:
                continue
            if isinstance(values, Sequence) and not isinstance(values, (str, bytes)):
                for value in values:
                    explicit[int(value)] = str(modality)
    token_text_by_id = metadata.get("special_token_text_by_id", {})
    vision_special_ids: set[int] = {
        int(value) for value in metadata.get("vision_special_token_ids", [])
    }
    if isinstance(token_text_by_id, Mapping):
        for raw_token_id, raw_text in token_text_by_id.items():
            token_id = int(raw_token_id)
            normalized = str(raw_text).lower()
            if "audio" in normalized:
                explicit[token_id] = "audio"
            elif "image" in normalized:
                explicit[token_id] = "image"
            elif "video" in normalized:
                explicit[token_id] = "video"
            elif "vision" in normalized:
                vision_special_ids.add(token_id)
    return explicit, vision_special_ids


def _apply_modality_special_labels(
    *,
    input_ids: Sequence[int],
    labels: list[str],
    modality_by_id: Mapping[int, str],
    explicit_special_by_id: Mapping[int, str],
    vision_special_ids: set[int],
) -> None:
    for index, token_id in enumerate(input_ids):
        normalized_id = int(token_id)
        if normalized_id in modality_by_id:
            labels[index] = modality_by_id[normalized_id]
            continue
        modality = explicit_special_by_id.get(normalized_id)
        if modality is not None:
            labels[index] = _MODALITY_SPECIAL_BY_CONTENT[modality]
    for index, token_id in enumerate(input_ids):
        if int(token_id) not in vision_special_ids:
            continue
        immediate_candidates = {
            modality_by_id.get(int(input_ids[neighbor]))
            for neighbor in (index - 1, index + 1)
            if 0 <= neighbor < len(input_ids)
        } & {"image", "video"}
        if len(immediate_candidates) == 1:
            labels[index] = _MODALITY_SPECIAL_BY_CONTENT[
                immediate_candidates.pop()
            ]
            continue
        candidates: list[str] = []
        for direction in (-1, 1):
            neighbor = index + direction
            while (
                0 <= neighbor < len(input_ids)
                and int(input_ids[neighbor]) in vision_special_ids
            ):
                neighbor += direction
            if 0 <= neighbor < len(input_ids):
                modality = modality_by_id.get(int(input_ids[neighbor]))
                if modality in {"image", "video"}:
                    candidates.append(modality)
        if candidates and len(set(candidates)) == 1:
            labels[index] = _MODALITY_SPECIAL_BY_CONTENT[candidates[0]]


def _template_semantic_labels(
    *,
    input_ids: Sequence[int],
    offsets: Sequence[tuple[int, int]],
    char_spans: Sequence[Mapping[str, Any]],
    region_labels: Sequence[str],
    modality_by_id: Mapping[int, str],
    explicit_special_by_id: Mapping[int, str],
    vision_special_ids: set[int],
) -> list[str]:
    labels = list(region_labels)
    for index, (token_id, (start, end)) in enumerate(zip(input_ids, offsets)):
        modality = modality_by_id.get(int(token_id))
        if modality is not None:
            labels[index] = modality
            continue
        if end <= start:
            continue
        for span in char_spans:
            span_start = int(span.get("template_start", span.get("start", -1)))
            span_end = int(span.get("template_end", span.get("end", -1)))
            if max(start, span_start) < min(end, span_end):
                candidate = str(span.get("label", "unknown"))
                labels[index] = candidate if candidate in TOKEN_TYPES else "unknown"
                break
    _apply_modality_special_labels(
        input_ids=input_ids,
        labels=labels,
        modality_by_id=modality_by_id,
        explicit_special_by_id=explicit_special_by_id,
        vision_special_ids=vision_special_ids,
    )
    return labels


def _template_semantic_labels_from_prefixes(
    *,
    tokenizer: Any,
    rendered_text: str,
    input_ids: Sequence[int],
    char_spans: Sequence[Mapping[str, Any]],
    region_labels: Sequence[str],
    modality_by_id: Mapping[int, str],
    explicit_special_by_id: Mapping[int, str],
    vision_special_ids: set[int],
) -> list[str] | None:
    labels = list(region_labels)
    for span in char_spans:
        start = int(span.get("template_start", span.get("start", -1)))
        end = int(span.get("template_end", span.get("end", -1)))
        if not 0 <= start <= end <= len(rendered_text):
            return None
        start_ids = _tokenizer_input_ids(tokenizer, rendered_text[:start])
        end_ids = _tokenizer_input_ids(tokenizer, rendered_text[:end])
        if (
            start_ids is None
            or end_ids is None
            or not 0 <= len(start_ids) <= len(end_ids) <= len(input_ids)
            or list(input_ids[: len(start_ids)]) != start_ids
            or list(input_ids[: len(end_ids)]) != end_ids
        ):
            return None
        candidate = str(span.get("label", "unknown"))
        label = candidate if candidate in TOKEN_TYPES else "unknown"
        for index in range(len(start_ids), len(end_ids)):
            if int(input_ids[index]) not in modality_by_id:
                labels[index] = label
    _apply_modality_special_labels(
        input_ids=input_ids,
        labels=labels,
        modality_by_id=modality_by_id,
        explicit_special_by_id=explicit_special_by_id,
        vision_special_ids=vision_special_ids,
    )
    return labels


def _align_template_labels(
    *,
    template_ids: Sequence[int],
    template_labels: Sequence[str],
    final_ids: Sequence[int],
    modality_by_id: Mapping[int, str],
) -> list[str] | None:
    aligned: list[str] = []
    template_index = 0
    final_index = 0
    while template_index < len(template_ids):
        token_id = int(template_ids[template_index])
        modality = modality_by_id.get(token_id)
        if modality is not None:
            while (
                template_index + 1 < len(template_ids)
                and int(template_ids[template_index + 1]) == token_id
            ):
                template_index += 1
            start = final_index
            while (
                final_index < len(final_ids)
                and int(final_ids[final_index]) == token_id
            ):
                aligned.append(modality)
                final_index += 1
            if final_index == start:
                return None
        else:
            if final_index >= len(final_ids) or int(final_ids[final_index]) != token_id:
                return None
            aligned.append(str(template_labels[template_index]))
            final_index += 1
        template_index += 1
    return aligned if final_index == len(final_ids) else None


def _mark_user_instruction_regions(
    *,
    input_ids: Sequence[int],
    labels: list[str],
    metadata: Mapping[str, Any],
) -> None:
    """Recover Qwen user-message regions when chat-template metadata is absent.

    ms-swift supplies the final tokenized batch rather than the original
    conversation. Qwen still exposes an unambiguous ``im_start, user`` marker,
    so prompt text can be separated from the preceding system message without
    treating answer labels or modality placeholders as instruction queries.
    """

    raw_user = metadata.get("user_token_id")
    raw_start = metadata.get("bos_token_id")
    raw_end = metadata.get("eos_token_id")
    if raw_user is None or raw_start is None or raw_end is None:
        return
    user_token_id = int(raw_user)
    start_token_id = int(raw_start)
    raw_end_ids = raw_end if isinstance(raw_end, (list, tuple, set)) else [raw_end]
    end_token_ids = {int(value) for value in raw_end_ids if value is not None}
    if not end_token_ids:
        return
    inside_user = False
    index = 0
    while index < len(input_ids):
        token_id = int(input_ids[index])
        if (
            token_id == start_token_id
            and index + 1 < len(input_ids)
            and int(input_ids[index + 1]) == user_token_id
        ):
            inside_user = True
            if labels[index + 1] == "unknown":
                labels[index + 1] = "ins_special"
            index += 2
            continue
        if inside_user and token_id in end_token_ids:
            inside_user = False
        elif inside_user and labels[index] == "unknown":
            labels[index] = "instruction"
        index += 1


def _fallback_semantic_labels(state: ForwardState) -> list[list[str]]:
    input_rows = _rows(state.input_ids)
    mask_rows = _rows(state.attention_mask)
    label_rows = _rows(state.labels) if state.labels is not None else []
    metadata = state.modality_metadata
    modality_by_id = {
        int(token_id): str(modality)
        for modality, token_id in metadata.get("token_ids", {}).items()
        if token_id is not None
    }
    special_token_ids = {
        int(value) for value in metadata.get("special_token_ids", [])
    }
    global_special_token_ids = {
        int(value) for value in metadata.get("global_special_token_ids", [])
    }
    explicit_special_by_id, vision_special_ids = _modality_special_sets(metadata)
    result: list[list[str]] = []
    for sample_index, input_row in enumerate(input_rows):
        mask = (
            mask_rows[sample_index]
            if sample_index < len(mask_rows)
            else [1] * len(input_row)
        )
        training_labels = (
            label_rows[sample_index]
            if sample_index < len(label_rows)
            else []
        )
        row: list[str] = []
        for position, token_id in enumerate(input_row):
            normalized_id = int(token_id)
            supervised = (
                position < len(training_labels)
                and int(training_labels[position]) != -100
            )
            if position < len(mask) and not int(mask[position]):
                row.append("pad")
            elif normalized_id in modality_by_id:
                row.append(modality_by_id[normalized_id])
            elif normalized_id in explicit_special_by_id:
                row.append(
                    _MODALITY_SPECIAL_BY_CONTENT[
                        explicit_special_by_id[normalized_id]
                    ]
                )
            elif normalized_id in global_special_token_ids:
                row.append("global_special")
            elif normalized_id in special_token_ids:
                row.append("resp_special" if supervised else "global_special")
            elif supervised:
                row.append("response")
            else:
                row.append("unknown")
        _apply_modality_special_labels(
            input_ids=input_row,
            labels=row,
            modality_by_id=modality_by_id,
            explicit_special_by_id=explicit_special_by_id,
            vision_special_ids=vision_special_ids,
        )
        _mark_user_instruction_regions(
            input_ids=input_row,
            labels=row,
            metadata=metadata,
        )
        for position, value in enumerate(mask):
            if position < len(row) and not int(value):
                row[position] = "pad"
        result.append(row)
    return result


def _first_tensor_like(value: Any) -> Any | None:
    if getattr(value, "shape", None) is not None:
        return value
    if isinstance(value, Mapping):
        for child in value.values():
            result = _first_tensor_like(child)
            if result is not None:
                return result
    elif isinstance(value, (list, tuple)):
        for child in value:
            result = _first_tensor_like(child)
            if result is not None:
                return result
    return None


def _carrier(args: tuple[Any, ...], kwargs: Mapping[str, Any]) -> Any | None:
    for name in ("hidden_states", "inputs_embeds", "input_ids"):
        value = kwargs.get(name)
        if value is not None:
            return value
    return _first_tensor_like(args)


def _token_count(value: Any, *, stage: str) -> int | None:
    shape = _shape(value)
    if not shape:
        return None
    if stage == "llm" and len(shape) >= 3:
        return int(prod(shape[:-1]))
    if stage == "audio_pooler" and len(shape) >= 2:
        # Qwen transposes one audio item to [channels, tokens] around AvgPool1d.
        return int(shape[-1])
    return int(shape[0])


def _phase_record() -> dict[str, Any]:
    return {
        "calls": 0,
        "input": {
            "total_tokens": 0,
            "classified_tokens": 0,
            "classification_delta": 0,
            "token_counts": _empty_counts(),
        },
        "output": {
            "total_tokens": 0,
            "classified_tokens": 0,
            "classification_delta": 0,
            "token_counts": _empty_counts(),
        },
    }


class Qwen25OmniTokenDebugCollector:
    """Collect per-stage and per-layer token counts during one request."""

    def __init__(self, model: Any, processor: Any | None = None) -> None:
        self.model = model
        self.processor = processor
        self.thinker = getattr(model, "thinker", model)
        self.audio_tower = getattr(self.thinker, "audio_tower", None)
        self.llm = getattr(self.thinker, "model", None)
        self._targets = self._resolve_targets()
        self._hook_handles: list[Any] = []
        self._installed = False
        self._active: dict[str, Any] | None = None
        self.last_report: dict[str, Any] | None = None

    @staticmethod
    def _chat_template_semantic_labels(
        state: ForwardState,
        processed_batch: ProcessedBatch,
        *,
        processor: Any | None = None,
    ) -> tuple[list[list[str]] | None, list[str], str]:
        tokenizer = getattr(processor, "tokenizer", None)
        metadata = processed_batch.metadata
        rendered_texts = metadata.get("chat_template_text_by_sample")
        char_spans_by_sample = metadata.get("semantic_char_spans_by_sample")
        message_prefixes_by_sample = metadata.get(
            "semantic_message_prefixes_by_sample"
        )
        if (
            tokenizer is None
            or not isinstance(rendered_texts, list)
            or not isinstance(char_spans_by_sample, list)
            or not isinstance(message_prefixes_by_sample, list)
        ):
            return (
                None,
                ["chat-template semantic inputs are unavailable"],
                "unavailable",
            )
        input_rows = _rows(state.input_ids)
        mask_rows = _rows(state.attention_mask)
        if not (
            len(rendered_texts)
            == len(char_spans_by_sample)
            == len(message_prefixes_by_sample)
            == len(input_rows)
        ):
            return (
                None,
                ["chat-template semantic batch sizes do not match"],
                "unavailable",
            )
        modality_by_id = {
            int(token_id): str(modality)
            for modality, token_id in metadata.get("token_ids", {}).items()
            if token_id is not None
        }
        special_token_ids = {
            int(value) for value in metadata.get("special_token_ids", [])
        }
        global_special_token_ids = {
            int(value) for value in metadata.get("global_special_token_ids", [])
        }
        explicit_special_by_id, vision_special_ids = _modality_special_sets(
            metadata
        )
        semantic_rows: list[list[str]] = []
        warnings: list[str] = []
        used_prefix_alignment = False
        for sample_index, (
            rendered_text,
            char_spans,
            message_prefixes,
            input_row,
        ) in enumerate(
            zip(
                rendered_texts,
                char_spans_by_sample,
                message_prefixes_by_sample,
                input_rows,
            )
        ):
            payload = _tokenizer_template_payload(tokenizer, str(rendered_text))
            if payload is None:
                return (
                    None,
                    [f"sample {sample_index} tokenizer tokenization is unavailable"],
                    "unavailable",
                )
            template_ids, offsets = payload
            region_labels = _message_region_labels(
                tokenizer=tokenizer,
                template_ids=template_ids,
                message_prefixes=message_prefixes,
                special_token_ids=special_token_ids,
                global_special_token_ids=global_special_token_ids,
            )
            if region_labels is None:
                return (
                    None,
                    [f"sample {sample_index} message prefixes do not align"],
                    "unavailable",
                )
            if offsets is None:
                template_labels = _template_semantic_labels_from_prefixes(
                    tokenizer=tokenizer,
                    rendered_text=str(rendered_text),
                    input_ids=template_ids,
                    char_spans=char_spans,
                    region_labels=region_labels,
                    modality_by_id=modality_by_id,
                    explicit_special_by_id=explicit_special_by_id,
                    vision_special_ids=vision_special_ids,
                )
                used_prefix_alignment = True
            else:
                template_labels = _template_semantic_labels(
                    input_ids=template_ids,
                    offsets=offsets,
                    char_spans=char_spans,
                    region_labels=region_labels,
                    modality_by_id=modality_by_id,
                    explicit_special_by_id=explicit_special_by_id,
                    vision_special_ids=vision_special_ids,
                )
            if template_labels is None:
                return (
                    None,
                    [f"sample {sample_index} chat-template spans cannot be tokenized"],
                    "unavailable",
                )
            mask = (
                mask_rows[sample_index]
                if sample_index < len(mask_rows)
                else [1] * len(input_row)
            )
            valid_positions = [
                position
                for position in range(len(input_row))
                if position >= len(mask) or int(mask[position])
            ]
            final_ids = [int(input_row[position]) for position in valid_positions]
            aligned = _align_template_labels(
                template_ids=template_ids,
                template_labels=template_labels,
                final_ids=final_ids,
                modality_by_id=modality_by_id,
            )
            if aligned is None:
                return (
                    None,
                    [
                        f"sample {sample_index} chat-template tokens do not align "
                        "with processor input_ids"
                    ],
                    "unavailable",
                )
            row = ["pad"] * len(input_row)
            for position, label in zip(valid_positions, aligned):
                row[position] = label
            semantic_rows.append(row)
            sample_warnings = metadata.get(
                "semantic_classification_warnings_by_sample", []
            )
            if isinstance(sample_warnings, list) and sample_index < len(
                sample_warnings
            ):
                warnings.extend(
                    f"sample {sample_index}: {item}"
                    for item in sample_warnings[sample_index]
                )
        source = (
            "chat_template_prefix_alignment"
            if used_prefix_alignment
            else "chat_template_offsets"
        )
        return semantic_rows, warnings, source

    def prepare_sequence_labels(
        self,
        state: ForwardState,
        processed_batch: ProcessedBatch,
    ) -> None:
        """Attach exclusive semantic labels before any sequence compaction."""
        prepare_qwen25_omni_semantic_labels(
            state,
            processed_batch,
            processor=self.processor,
        )

    def _resolve_targets(self) -> list[dict[str, Any]]:
        audio_layers = list(getattr(self.audio_tower, "layers", ()))
        llm_layers = list(getattr(self.llm, "layers", ()))
        if self.thinker is None or self.audio_tower is None or not audio_layers:
            raise InvariantError(
                "Qwen debug mode could not resolve thinker.audio_tower.layers."
            )
        if self.llm is None or not llm_layers:
            raise InvariantError("Qwen debug mode could not resolve thinker.model.layers.")
        targets: list[dict[str, Any]] = []
        for index, module in enumerate(audio_layers):
            targets.append(
                {
                    "stage": "audio_encoder",
                    "layer_index": index,
                    "module_path": f"thinker.audio_tower.layers.{index}",
                    "module": module,
                }
            )
        for stage, name in (
            ("audio_pooler", "avg_pooler"),
            ("audio_post_norm", "ln_post"),
            ("audio_projector", "proj"),
        ):
            module = getattr(self.audio_tower, name, None)
            if module is None:
                raise InvariantError(
                    f"Qwen debug mode could not resolve thinker.audio_tower.{name}."
                )
            targets.append(
                {
                    "stage": stage,
                    "layer_index": None,
                    "module_path": f"thinker.audio_tower.{name}",
                    "module": module,
                }
            )
        for index, module in enumerate(llm_layers):
            targets.append(
                {
                    "stage": "llm",
                    "layer_index": index,
                    "module_path": f"thinker.model.layers.{index}",
                    "module": module,
                }
            )
        return targets

    def install(self) -> None:
        if self._installed:
            return
        for target in self._targets:
            register = getattr(target["module"], "register_forward_hook", None)
            if not callable(register):
                raise InvariantError(
                    f"Debug target {target['module_path']} does not support forward hooks."
                )
            self._hook_handles.append(
                register(self._hook(target), with_kwargs=True)
            )
        self._installed = True

    def uninstall(self) -> None:
        for handle in reversed(self._hook_handles):
            handle.remove()
        self._hook_handles.clear()
        self._installed = False

    def _hook(self, target: Mapping[str, Any]) -> Any:
        def record(
            _module: Any,
            args: tuple[Any, ...],
            kwargs: dict[str, Any],
            output: Any,
        ) -> None:
            active = self._active
            if active is None:
                return
            module_record = active["module_records"][target["module_path"]]
            if target["stage"] == "llm":
                phase = (
                    "prefill"
                    if module_record["prefill"]["calls"] == 0
                    else "decode"
                )
            else:
                phase = "encode"
            input_tokens = _token_count(
                _carrier(args, kwargs),
                stage=str(target["stage"]),
            )
            output_tokens = _token_count(
                _first_tensor_like(output),
                stage=str(target["stage"]),
            )
            self._accumulate(
                module_record[phase],
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                stage=str(target["stage"]),
                phase=phase,
                llm_prefill_counts=active["llm_prefill_counts"],
            )

        return record

    def apply_compression_report(
        self,
        compression_report: Mapping[str, Any] | None,
    ) -> None:
        """Reconcile one compression report with hook-collected diagnostics.

        Most placements are fully visible to the collector's module hooks. A
        placement-specific handler is used only when a compressor bypasses a
        hooked module boundary, as A-ToMe does for active audio encoder layers.
        """

        if not compression_report:
            return
        placement = str(compression_report.get("placement") or "")
        if placement == Placement.ENCODER_LEVEL.value:
            self._apply_encoder_level_report(compression_report)
        elif placement == Placement.PROJECTOR_LEVEL.value:
            self._apply_projector_level_report(compression_report)
        elif placement == Placement.LLM_LEVEL.value:
            self._apply_llm_level_report(compression_report)
        elif placement == Placement.INPUT_LEVEL.value:
            self._apply_input_level_report(compression_report)

    def apply_generated_response(
        self,
        generated_ids: Any,
        *,
        semantic_metadata: Mapping[str, Any],
    ) -> None:
        """Append generated-response and complete-sequence semantic snapshots."""

        if self.last_report is None:
            raise InvariantError(
                "Cannot apply generated response diagnostics before debug collection."
            )
        rows = _rows(generated_ids)
        sample_ids = list(self.last_report.get("sample_ids", []))
        if len(rows) != len(sample_ids):
            raise InvariantError(
                "Generated response batch does not match debug sample ids."
            )
        special_token_ids = {
            int(value) for value in semantic_metadata.get("special_token_ids", [])
        }
        modality_by_id = {
            int(token_id): str(modality)
            for modality, token_id in semantic_metadata.get("token_ids", {}).items()
            if token_id is not None
        }
        explicit_special_by_id, vision_special_ids = _modality_special_sets(
            semantic_metadata
        )
        generated_records: list[dict[str, Any]] = []
        for sample_id, row in zip(sample_ids, rows):
            labels: list[str] = []
            for token_id in row:
                normalized_id = int(token_id)
                modality = modality_by_id.get(normalized_id)
                if modality is not None:
                    labels.append(modality)
                elif normalized_id in explicit_special_by_id:
                    labels.append(
                        _MODALITY_SPECIAL_BY_CONTENT[
                            explicit_special_by_id[normalized_id]
                        ]
                    )
                elif normalized_id in special_token_ids:
                    labels.append("resp_special")
                else:
                    labels.append("response")
            _apply_modality_special_labels(
                input_ids=row,
                labels=labels,
                modality_by_id=modality_by_id,
                explicit_special_by_id=explicit_special_by_id,
                vision_special_ids=vision_special_ids,
            )
            counts = _empty_counts()
            for label in labels:
                counts[label if label in counts else "unknown"] += 1
            generated_records.append(_sample_record(sample_id, counts))
        generated_snapshot = {
            "stage": "generated_response",
            "token_domain": "llm_generated_response",
            "classification_source": "generated_token_ids",
            "total_tokens": sum(item["total_tokens"] for item in generated_records),
            "token_counts": _merge_counts(
                [item["token_counts"] for item in generated_records]
            ),
            "per_sample": generated_records,
        }
        prefill_snapshot = next(
            (
                item
                for item in self.last_report["stage_snapshots"]
                if item.get("stage") == "post_legalizer_llm_input"
            ),
            None,
        )
        if prefill_snapshot is None:
            raise InvariantError(
                "Debug report omitted post_legalizer_llm_input snapshot."
            )
        complete_records: list[dict[str, Any]] = []
        for prompt_record, response_record in zip(
            prefill_snapshot["per_sample"],
            generated_records,
        ):
            counts = _empty_counts()
            for name in TOKEN_TYPES:
                counts[name] = int(prompt_record["token_counts"].get(name, 0)) + int(
                    response_record["token_counts"].get(name, 0)
                )
            complete_records.append(_sample_record(prompt_record["sample_id"], counts))
        complete_snapshot = {
            "stage": "complete_llm_sequence",
            "token_domain": "llm_complete_sequence",
            "classification_source": "prompt_labels_plus_generated_token_ids",
            "total_tokens": sum(item["total_tokens"] for item in complete_records),
            "token_counts": _merge_counts(
                [item["token_counts"] for item in complete_records]
            ),
            "per_sample": complete_records,
        }
        self.last_report["stage_snapshots"].extend(
            [generated_snapshot, complete_snapshot]
        )

    def _apply_encoder_level_report(
        self,
        compression_report: Mapping[str, Any],
    ) -> None:
        """Fill layer counts when an encoder compressor bypasses module hooks."""

        if compression_report.get("placement") != "encoder_level":
            return
        layer_reports = compression_report.get("layers")
        if not isinstance(layer_reports, list):
            return
        if self.last_report is None:
            raise InvariantError(
                "Cannot apply encoder compression diagnostics before debug collection."
            )
        by_index = {
            int(record["layer_index"]): record
            for record in self.last_report["layers"]
            if record.get("stage") == "audio_encoder"
        }
        for layer_report in layer_reports:
            if layer_report.get("application_position") == {"boundary": "inter_layer", "index": 0}:
                # Encoder initial compaction happens outside layer.__call__.
                # Its module hooks already see the compact input and output.
                continue
            layer_index = int(layer_report["layer_index"])
            record = by_index.get(layer_index)
            if record is None:
                raise InvariantError(
                    f"Debug collector has no audio encoder layer {layer_index}."
                )
            encode = _phase_record()
            self._accumulate(
                encode,
                input_tokens=int(layer_report["input_tokens"]),
                output_tokens=int(layer_report["output_tokens"]),
                stage="audio_encoder",
                phase="encode",
                llm_prefill_counts={},
            )
            record["encode"] = encode

    @staticmethod
    def _apply_projector_level_report(
        compression_report: Mapping[str, Any],
    ) -> None:
        """Keep hook-observed projector diagnostics as the authoritative counts."""

        del compression_report

    def _apply_llm_level_report(
        self,
        compression_report: Mapping[str, Any],
    ) -> None:
        """Reconcile hook totals with Audio-Pyramid's per-layer audio layout."""

        stage_reports = compression_report.get("stages")
        if not isinstance(stage_reports, list) or not stage_reports:
            return
        if self.last_report is None:
            raise InvariantError(
                "Cannot apply LLM compression diagnostics before debug collection."
            )
        post_legalizer = next(
            (
                item
                for item in self.last_report["stage_snapshots"]
                if item.get("stage") == "post_legalizer_llm_input"
            ),
            None,
        )
        if post_legalizer is None:
            raise InvariantError(
                "Debug report omitted post_legalizer_llm_input snapshot."
            )
        stages_by_layer: dict[int, list[Mapping[str, Any]]] = {}
        for item in stage_reports:
            if not isinstance(item, Mapping):
                continue
            position = item.get("application_position")
            if position == {"boundary": "inter_layer", "index": 0}:
                # Module hooks surround layer.forward: its input is the full
                # initial sequence, and its output includes initial compaction.
                layer_index = 0
            elif item.get("layer_index_zero_based") is not None:
                layer_index = int(item["layer_index_zero_based"])
            else:
                continue
            stages_by_layer.setdefault(layer_index, []).append(item)
        current_counts = {
            name: int(post_legalizer["token_counts"].get(name, 0))
            for name in TOKEN_TYPES
        }
        for layer_record in self.last_report["layers"]:
            if layer_record.get("stage") != "llm":
                continue
            prefill = layer_record["prefill"]
            if int(prefill["calls"]) == 0:
                continue
            input_counts = self._llm_layout_counts(
                reference_counts=current_counts,
                observed=int(prefill["input"]["total_tokens"]),
                audio_tokens=int(current_counts["audio"]),
            )
            self._replace_side_counts(prefill["input"], input_counts)

            output_audio_tokens = int(input_counts["audio"])
            for stage_report in stages_by_layer.get(int(layer_record["layer_index"]), ()):
                reported_input_audio = int(stage_report["input_audio_tokens"])
                if reported_input_audio != output_audio_tokens:
                    raise InvariantError(
                        "LLM compression report input audio count does not match the "
                        f"tracked layout at layer {layer_record['layer_index']}: "
                        f"reported={reported_input_audio}, tracked={output_audio_tokens}."
                    )
                output_audio_tokens = int(stage_report["output_audio_tokens"])
            output_counts = self._llm_layout_counts(
                reference_counts=input_counts,
                observed=int(prefill["output"]["total_tokens"]),
                audio_tokens=output_audio_tokens,
            )
            self._replace_side_counts(prefill["output"], output_counts)
            current_counts = output_counts

    @staticmethod
    def _llm_layout_counts(
        *,
        reference_counts: Mapping[str, int],
        observed: int,
        audio_tokens: int,
    ) -> dict[str, int]:
        counts = _empty_counts()
        counts.update(
            {name: int(reference_counts.get(name, 0)) for name in TOKEN_TYPES}
        )
        counts["audio"] = audio_tokens
        counts["pad"] = 0
        classified_without_padding = sum(counts.values())
        if classified_without_padding > observed:
            raise InvariantError(
                "LLM compression semantic counts exceed the observed compact layout: "
                f"classified={classified_without_padding}, observed={observed}."
            )
        counts["pad"] = observed - classified_without_padding
        return counts

    @staticmethod
    def _replace_side_counts(
        side_record: dict[str, Any],
        counts: Mapping[str, int],
    ) -> None:
        total_tokens = int(side_record["total_tokens"])
        normalized = {name: int(counts.get(name, 0)) for name in TOKEN_TYPES}
        if sum(normalized.values()) != total_tokens:
            raise InvariantError(
                "Reconciled LLM debug counts do not match the hook-observed total."
            )
        side_record["token_counts"] = normalized
        side_record["classified_tokens"] = total_tokens
        side_record["classification_delta"] = 0

    @staticmethod
    def _apply_input_level_report(
        compression_report: Mapping[str, Any],
    ) -> None:
        """Keep state snapshots as the authoritative input-level counts."""

        del compression_report

    @staticmethod
    def _classified_counts(
        *,
        observed: int | None,
        stage: str,
        phase: str,
        llm_prefill_counts: Mapping[str, int],
    ) -> dict[str, int]:
        if observed is None:
            return _empty_counts()
        if stage == "llm" and phase == "prefill":
            counts = _empty_counts()
            counts.update(
                {name: int(llm_prefill_counts.get(name, 0)) for name in TOKEN_TYPES}
            )
            classified = sum(counts.values())
            if classified < observed:
                counts["unknown"] += observed - classified
            elif classified > observed:
                # Shape-only hooks cannot know which semantic tokens a dynamic
                # LLM compressor removed. Keep the observation conserved and let
                # apply_compression_report() restore exact per-layer semantics.
                counts = _empty_counts()
                counts["unknown"] = observed
            return counts
        counts = _empty_counts()
        counts["response" if stage == "llm" else "audio"] = observed
        return counts

    @classmethod
    def _accumulate(
        cls,
        phase_record: dict[str, Any],
        *,
        input_tokens: int | None,
        output_tokens: int | None,
        stage: str,
        phase: str,
        llm_prefill_counts: Mapping[str, int],
    ) -> None:
        phase_record["calls"] += 1
        for side, observed in (("input", input_tokens), ("output", output_tokens)):
            if observed is None:
                continue
            phase_record[side]["total_tokens"] += observed
            counts = cls._classified_counts(
                observed=observed,
                stage=stage,
                phase=phase,
                llm_prefill_counts=llm_prefill_counts,
            )
            if sum(counts.values()) != observed:
                raise InvariantError(
                    f"Debug {stage}/{phase}/{side} token counts are not conserved."
                )
            for name in TOKEN_TYPES:
                phase_record[side]["token_counts"][name] += counts[name]
            phase_record[side]["classified_tokens"] += sum(counts.values())
            phase_record[side]["classification_delta"] = (
                phase_record[side]["total_tokens"]
                - phase_record[side]["classified_tokens"]
            )

    @contextmanager
    def request(
        self,
        *,
        original_state: ForwardState,
        active_state: ForwardState,
        processed_batch: ProcessedBatch,
    ) -> Iterator[None]:
        if not self._installed:
            raise InvariantError("Qwen token debug collector is not installed.")
        if self._active is not None:
            raise InvariantError("Nested Qwen token debug requests are unsupported.")
        original_expected = processed_batch.metadata.get(
            "expected_token_counts_by_sample", []
        )
        original_semantic_rows = _normalized_semantic_rows(
            processed_batch.metadata.get("semantic_token_labels_by_sample")
        )
        active_semantic_rows = _normalized_semantic_rows(
            active_state.modality_metadata.get("semantic_token_labels_by_sample")
        )
        original_source = str(
            processed_batch.metadata.get(
                "semantic_classification_source",
                "legacy_length_fallback",
            )
        )
        active_source = str(
            active_state.modality_metadata.get(
                "semantic_classification_source",
                original_source,
            )
        )
        processor_snapshot = (
            _llm_snapshot_from_semantic_labels(
                stage="processor_output",
                input_ids=original_state.input_ids,
                attention_mask=original_state.attention_mask,
                sample_ids=processed_batch.sample_ids,
                semantic_rows=original_semantic_rows,
                classification_source=original_source,
            )
            if original_semantic_rows is not None
            else _llm_snapshot_from_expected(
                stage="processor_output",
                input_ids=original_state.input_ids,
                attention_mask=original_state.attention_mask,
                sample_ids=processed_batch.sample_ids,
                expected_by_sample=original_expected,
            )
        )
        audio_feature_snapshot = _audio_feature_snapshot(processed_batch)
        post_legalizer_snapshot = (
            _llm_snapshot_from_semantic_labels(
                stage="post_legalizer_llm_input",
                input_ids=active_state.input_ids,
                attention_mask=active_state.attention_mask,
                sample_ids=processed_batch.sample_ids,
                semantic_rows=active_semantic_rows,
                classification_source=active_source,
            )
            if active_semantic_rows is not None
            else _llm_snapshot_from_state(
                active_state,
                stage="post_legalizer_llm_input",
                sample_ids=processed_batch.sample_ids,
            )
        )
        snapshots = [
            processor_snapshot,
            audio_feature_snapshot,
            post_legalizer_snapshot,
        ]
        module_records: dict[str, dict[str, Any]] = {}
        for target in self._targets:
            phases = (
                {"prefill": _phase_record(), "decode": _phase_record()}
                if target["stage"] == "llm"
                else {"encode": _phase_record()}
            )
            module_records[target["module_path"]] = {
                "stage": target["stage"],
                "layer_index": target["layer_index"],
                "module_path": target["module_path"],
                **phases,
            }
        self._active = {
            "stage_snapshots": snapshots,
            "module_records": module_records,
            "llm_prefill_counts": post_legalizer_snapshot["token_counts"],
        }
        self.last_report = None
        try:
            yield
        finally:
            active = self._active
            self._active = None
            if active is not None:
                ordered_records = [
                    active["module_records"][target["module_path"]]
                    for target in self._targets
                ]
                self.last_report = {
                    "schema_version": 3,
                    "mode": "debug",
                    "sample_ids": list(processed_batch.sample_ids),
                    "semantic_classification": {
                        "source": active_source,
                        "warnings": list(
                            active_state.modality_metadata.get(
                                "semantic_classification_warnings", []
                            )
                        ),
                    },
                    "stage_snapshots": deepcopy(active["stage_snapshots"]),
                    "layers": deepcopy(
                        [
                            record
                            for record in ordered_records
                            if record["layer_index"] is not None
                        ]
                    ),
                    "modules": deepcopy(
                        [
                            record
                            for record in ordered_records
                            if record["layer_index"] is None
                        ]
                    ),
                }


def prepare_qwen25_omni_semantic_labels(
    state: ForwardState,
    processed_batch: ProcessedBatch,
    *,
    processor: Any | None = None,
) -> None:
    """Attach one exclusive semantic role to every final processor token.

    This classifier is shared by debug collection and LLM-level compression.
    Existing labels are accepted only when their batch and sequence dimensions
    match the actual ``input_ids``; stale pre-expansion labels are recomputed.
    """

    input_rows = _rows(state.input_ids)
    existing = _normalized_semantic_rows(
        state.modality_metadata.get("semantic_token_labels_by_sample")
    )
    existing_matches = existing is not None and len(existing) == len(input_rows)
    if existing_matches:
        existing_matches = all(
            len(row) == len(input_row)
            for row, input_row in zip(existing, input_rows)
        )
    if existing_matches:
        assert existing is not None
        for metadata in (state.modality_metadata, processed_batch.metadata):
            metadata["semantic_token_labels_by_sample"] = deepcopy(existing)
            for name in (
                "semantic_classification_source",
                "semantic_classification_warnings",
            ):
                if name in state.modality_metadata:
                    metadata[name] = deepcopy(state.modality_metadata[name])
        return

    warnings: list[str] = []
    if existing is not None:
        warnings.append(
            "existing semantic labels did not match final processor input_ids"
        )
    semantic_rows, template_warnings, source = (
        Qwen25OmniTokenDebugCollector._chat_template_semantic_labels(
            state,
            processed_batch,
            processor=processor,
        )
    )
    warnings.extend(template_warnings)
    if semantic_rows is None:
        semantic_rows = _fallback_semantic_labels(state)
        source = (
            "training_labels_fallback"
            if state.labels is not None
            else "unresolved_fallback"
        )
    for metadata in (state.modality_metadata, processed_batch.metadata):
        metadata["semantic_token_labels_by_sample"] = deepcopy(semantic_rows)
        metadata["semantic_classification_source"] = source
        metadata["semantic_classification_warnings"] = list(warnings)


__all__ = [
    "Qwen25OmniTokenDebugCollector",
    "TOKEN_TYPES",
    "prepare_qwen25_omni_semantic_labels",
]
