"""Exact token-group layouts for analysis probes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping

from mllm_comp.core.errors import BackendIntegrationError


@dataclass(frozen=True)
class TokenGroupLayout:
    labels_by_sample: tuple[tuple[str, ...], ...]
    evidence: Mapping[str, Any]

    def positions(self, sample_index: int, group: str) -> tuple[int, ...]:
        return tuple(
            index
            for index, label in enumerate(self.labels_by_sample[sample_index])
            if label == group or (group == "vision" and label in {"image", "video"})
        )

    def counts(self) -> dict[str, int]:
        result: dict[str, int] = {}
        for row in self.labels_by_sample:
            for label in row:
                result[label] = result.get(label, 0) + 1
        result["vision"] = result.get("image", 0) + result.get("video", 0)
        return result


def build_token_group_layout(
    state: Any,
    *,
    sequence_length: int,
    backbone_name: str,
    runtime: Any | None = None,
    layer_index: int | None = None,
) -> TokenGroupLayout:
    """Build expanded-sequence labels and report the exact evidence route."""

    if sequence_length < 1:
        raise BackendIntegrationError("Analysis sequence_length must be positive.")
    input_rows = _rows(getattr(state, "input_ids", None))
    batch_size = len(getattr(state, "sample_ids", ()) or ()) or len(input_rows) or 1
    if batch_size != 1:
        raise BackendIntegrationError(
            "The first analysis token-layout contract is single-sample."
        )

    if backbone_name == "llava_onevision" and (
        not input_rows or len(input_rows[0]) != sequence_length
    ):
        return _llava_expanded_layout(state, input_rows, sequence_length)

    if backbone_name == "qwen25_omni":
        semantic_layout = _qwen_omni_semantic_layout(
            state,
            runtime=runtime,
            batch_size=batch_size,
            sequence_length=sequence_length,
            layer_index=layer_index,
        )
        if semantic_layout is not None:
            return semantic_layout

    if input_rows and any(len(row) != sequence_length for row in input_rows):
        raise BackendIntegrationError(
            f"Backbone {backbone_name!r} input_ids length does not match collected "
            f"hidden sequence length {sequence_length}."
        )
    labels = [["text"] * sequence_length for _ in range(batch_size)]
    _apply_padding(labels, getattr(state, "attention_mask", None))
    _apply_system_spans(labels, getattr(state, "modality_metadata", {}))
    span_count = _apply_modality_spans(
        labels,
        getattr(state, "modality_token_spans", ()),
        sequence_length=sequence_length,
    )
    metadata = getattr(state, "modality_metadata", {})
    return TokenGroupLayout(
        labels_by_sample=tuple(tuple(row) for row in labels),
        evidence={
            "layout_version": "analysis_token_groups_v1",
            "route": "forward_state_exact_spans",
            "backbone": backbone_name,
            "sequence_length": sequence_length,
            "modality_span_count": span_count,
            "system_classification": (
                "exact_spans"
                if _system_spans(metadata)
                else "unavailable_nonvisual_tokens_are_text"
            ),
        },
    )


def _qwen_omni_semantic_layout(
    state: Any,
    *,
    runtime: Any | None,
    batch_size: int,
    sequence_length: int,
    layer_index: int | None,
) -> TokenGroupLayout | None:
    candidates: list[tuple[str, Any, Any, Any]] = []
    runtime_metadata = getattr(runtime, "metadata", {})
    if isinstance(runtime_metadata, Mapping):
        last_forward = runtime_metadata.get("last_training_forward")
        if isinstance(last_forward, Mapping):
            snapshot = last_forward.get("effective_token_layout")
            if isinstance(snapshot, Mapping):
                layer_rows = None
                if layer_index is not None:
                    by_layer = snapshot.get("semantic_labels_by_layer_entry")
                    if isinstance(by_layer, Mapping):
                        layer_rows = by_layer.get(str(int(layer_index)))
                if layer_rows is not None:
                    candidates.append(
                        (
                            "runtime_layer_entry_semantic_labels",
                            layer_rows,
                            sequence_length,
                            snapshot.get("sample_ids"),
                        )
                    )
                candidates.append(
                    (
                        "runtime_effective_semantic_labels",
                        snapshot.get("semantic_labels_by_sample"),
                        snapshot.get("sequence_length"),
                        snapshot.get("sample_ids"),
                    )
                )
    metadata = getattr(state, "modality_metadata", {})
    if isinstance(metadata, Mapping):
        candidates.append(
            (
                "forward_state_semantic_labels",
                metadata.get("semantic_token_labels_by_sample"),
                sequence_length,
                getattr(state, "sample_ids", None),
            )
        )

    state_sample_ids = list(getattr(state, "sample_ids", ()) or ())
    for route, raw_rows, raw_length, raw_sample_ids in candidates:
        if raw_length is None or int(raw_length) != sequence_length:
            continue
        rows = _semantic_rows(raw_rows)
        if rows is None or len(rows) != batch_size:
            continue
        if any(len(row) != sequence_length for row in rows):
            continue
        candidate_sample_ids = list(raw_sample_ids or ())
        if (
            state_sample_ids
            and candidate_sample_ids
            and candidate_sample_ids != state_sample_ids
        ):
            continue
        labels = [[_analysis_semantic_label(value) for value in row] for row in rows]
        if route == "forward_state_semantic_labels":
            _apply_padding(labels, getattr(state, "attention_mask", None))
        return TokenGroupLayout(
            labels_by_sample=tuple(tuple(row) for row in labels),
            evidence={
                "layout_version": "analysis_token_groups_v1",
                "route": route,
                "backbone": "qwen25_omni",
                "sequence_length": sequence_length,
                "semantic_mapping": "qwen25_omni_semantic_roles_v1",
                "system_classification": "exact_semantic_labels",
                "layer_index_zero_based": layer_index,
            },
        )
    return None


def _semantic_rows(value: Any) -> list[list[str]] | None:
    if not isinstance(value, (list, tuple)):
        return None
    rows: list[list[str]] = []
    for row in value:
        if not isinstance(row, (list, tuple)):
            return None
        rows.append([str(item) for item in row])
    return rows


def _analysis_semantic_label(value: str) -> str:
    if value in {"system_prompt", "sys_special"}:
        return "system"
    if value in {"image", "video", "audio"}:
        return value
    if value in {"pad", "padding"}:
        return "padding"
    return "text"


def _llava_expanded_layout(
    state: Any,
    input_rows: list[list[int]],
    sequence_length: int,
) -> TokenGroupLayout:
    if len(input_rows) != 1:
        raise BackendIntegrationError(
            "LLaVA expanded token fallback requires one unpadded input row."
        )
    attention_rows = _rows(getattr(state, "attention_mask", None))
    if attention_rows and (
        len(attention_rows) != 1
        or len(attention_rows[0]) != len(input_rows[0])
        or any(value != 1 for value in attention_rows[0])
    ):
        raise BackendIntegrationError(
            "LLaVA expanded token fallback requires one fully valid prompt row."
        )
    spans = list(getattr(state, "modality_token_spans", ()) or ())
    image_spans = [span for span in spans if getattr(span, "modality", None) == "image"]
    if len(image_spans) != 1:
        raise BackendIntegrationError(
            "LLaVA expanded token fallback is exact only for one image placeholder; "
            f"found {len(image_spans)}."
        )
    placeholder = image_spans[0]
    if int(placeholder.end) - int(placeholder.start) != 1:
        raise BackendIntegrationError("LLaVA image placeholder span must have width one.")
    expanded_image_tokens = sequence_length - (len(input_rows[0]) - 1)
    if expanded_image_tokens < 1:
        raise BackendIntegrationError(
            "LLaVA expanded sequence does not contain a positive image-token span."
        )
    start = int(placeholder.start)
    end = start + expanded_image_tokens
    labels = ["text"] * sequence_length
    labels[start:end] = ["image"] * expanded_image_tokens
    metadata = getattr(state, "modality_metadata", {}) or {}
    if _system_spans(metadata):
        raise BackendIntegrationError(
            "LLaVA pre-expansion system spans require an explicit expansion mapping."
        )
    return TokenGroupLayout(
        labels_by_sample=(tuple(labels),),
        evidence={
            "layout_version": "analysis_token_groups_v1",
            "route": "llava_single_image_placeholder_expansion",
            "backbone": "llava_onevision",
            "sequence_length": sequence_length,
            "original_sequence_length": len(input_rows[0]),
            "placeholder_position": start,
            "expanded_image_token_count": expanded_image_tokens,
            "system_classification": "unavailable_nonvisual_tokens_are_text",
            "fallback_limit": "single_image_only",
        },
    )


def _apply_padding(labels: list[list[str]], attention_mask: Any) -> None:
    rows = _rows(attention_mask)
    if not rows:
        return
    if len(rows) != len(labels):
        raise BackendIntegrationError("Analysis attention-mask batch size mismatch.")
    for label_row, mask_row in zip(labels, rows):
        if len(label_row) != len(mask_row):
            raise BackendIntegrationError("Analysis attention-mask sequence mismatch.")
        for index, value in enumerate(mask_row):
            if not value:
                label_row[index] = "padding"


def _apply_system_spans(labels: list[list[str]], metadata: Any) -> None:
    for sample_index, start, end in _system_spans(metadata):
        if not 0 <= sample_index < len(labels) or not 0 <= start <= end <= len(
            labels[sample_index]
        ):
            raise BackendIntegrationError("Analysis system token span is out of bounds.")
        labels[sample_index][start:end] = ["system"] * (end - start)


def _apply_modality_spans(
    labels: list[list[str]],
    spans: Iterable[Any],
    *,
    sequence_length: int,
) -> int:
    count = 0
    for span in spans:
        modality = str(getattr(span, "modality", ""))
        if modality not in {"image", "video", "audio"}:
            continue
        sample_index = int(getattr(span, "sample_index", 0))
        start = int(getattr(span, "start"))
        end = int(getattr(span, "end"))
        if (
            not 0 <= sample_index < len(labels)
            or not 0 <= start < end <= sequence_length
        ):
            raise BackendIntegrationError("Analysis modality token span is out of bounds.")
        labels[sample_index][start:end] = [modality] * (end - start)
        count += 1
    return count


def _system_spans(metadata: Any) -> list[tuple[int, int, int]]:
    if not isinstance(metadata, Mapping):
        return []
    raw = metadata.get("system_token_spans", ()) or ()
    result: list[tuple[int, int, int]] = []
    for item in raw:
        if isinstance(item, Mapping):
            result.append(
                (
                    int(item.get("sample_index", 0)),
                    int(item["start"]),
                    int(item["end"]),
                )
            )
        else:
            values = tuple(int(value) for value in item)
            if len(values) == 2:
                result.append((0, values[0], values[1]))
            elif len(values) == 3:
                result.append(values)
            else:
                raise BackendIntegrationError(
                    "system_token_spans entries must be (start, end) or "
                    "(sample, start, end)."
                )
    return result


def _rows(value: Any) -> list[list[int]]:
    if value is None:
        return []
    tolist = getattr(value, "tolist", None)
    raw = tolist() if callable(tolist) else value
    if not isinstance(raw, (list, tuple)):
        return [[int(raw)]]
    if not raw:
        return []
    if not isinstance(raw[0], (list, tuple)):
        raw = [raw]
    return [[int(item) for item in row] for row in raw]


__all__ = [
    "TokenGroupLayout",
    "build_token_group_layout",
]
