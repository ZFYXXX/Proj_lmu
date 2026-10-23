"""Shared conversion from an lmms-eval AnalysisSample to a native forward state."""

from __future__ import annotations

from typing import Any, Mapping

from mllm_comp.backbones.base import ProcessedBatch
from mllm_comp.backbones.teacher_forcing import IGNORE_INDEX, OBJECTIVE_NAME
from mllm_comp.core.errors import BackendIntegrationError


def prepare_analysis_forward_state(
    *,
    backbone: Any,
    runtime: Any,
    sample: Mapping[str, Any],
    probe_name: str,
    output_attentions: bool = False,
    output_hidden_states: bool = False,
) -> Any:
    """Build a single-sample direct prompt forward state without a KV cache."""

    return _prepare_forward_state(
        backbone=backbone,
        runtime=runtime,
        sample=sample,
        probe_name=probe_name,
        processor_outputs_field="processor_outputs",
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
    )


def prepare_teacher_forced_analysis_forward_state(
    *,
    backbone: Any,
    runtime: Any,
    sample: Mapping[str, Any],
    probe_name: str,
    output_attentions: bool = False,
    output_hidden_states: bool = False,
) -> tuple[Any, dict[str, Any]]:
    """Build a validated answer-supervised state and return its provenance."""

    objective = teacher_forced_objective(sample, probe_name=probe_name)
    state = _prepare_forward_state(
        backbone=backbone,
        runtime=runtime,
        sample=sample,
        probe_name=probe_name,
        processor_outputs_field="teacher_forced_processor_outputs",
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        extra_metadata={"teacher_forcing": objective},
    )
    return state, objective


def _prepare_forward_state(
    *,
    backbone: Any,
    runtime: Any,
    sample: Mapping[str, Any],
    probe_name: str,
    processor_outputs_field: str,
    output_attentions: bool,
    output_hidden_states: bool,
    extra_metadata: Mapping[str, Any] | None = None,
) -> Any:
    processed = analysis_processed_batch(
        sample,
        probe_name=probe_name,
        processor_outputs_field=processor_outputs_field,
        extra_metadata=extra_metadata,
    )
    state = backbone.state_builder.build(
        processed,
        raw_batch=sample.get("raw_doc"),
        phase="eval",
    )
    native_inputs = getattr(state, "native_model_inputs", None)
    if not isinstance(native_inputs, dict):
        raise BackendIntegrationError(
            f"{probe_name} requires ForwardState.native_model_inputs."
        )
    native_inputs["use_cache"] = False
    native_inputs["return_dict"] = True
    native_inputs["output_attentions"] = output_attentions
    native_inputs["output_hidden_states"] = output_hidden_states

    validator = getattr(backbone.runner, "native_validator", None)
    validate_state = getattr(validator, "validate_state", None)
    if callable(validate_state):
        violations = validate_state(runtime, state, processed)
        if violations:
            details = "; ".join(
                f"{item.name}[{item.field}]" for item in violations
            )
            raise BackendIntegrationError(
                f"{probe_name} sample failed native validation: {details}"
            )
    return state


def analysis_processed_batch(
    sample: Mapping[str, Any],
    *,
    probe_name: str,
    processor_outputs_field: str = "processor_outputs",
    extra_metadata: Mapping[str, Any] | None = None,
) -> ProcessedBatch:
    outputs = sample.get(processor_outputs_field)
    if not isinstance(outputs, Mapping):
        raise BackendIntegrationError(
            f"AnalysisSample.{processor_outputs_field} must be a mapping."
        )
    metadata = dict(sample.get("metadata", {}))
    metadata.update(dict(extra_metadata or {}))
    prompt_lengths = (
        [int(value) for value in metadata.get("prompt_lengths", ())]
        if processor_outputs_field == "processor_outputs"
        else []
    )
    if not prompt_lengths:
        prompt_lengths = _valid_sequence_lengths(outputs)
    if len(prompt_lengths) != 1:
        raise BackendIntegrationError(
            f"{probe_name} currently requires exactly one prompt length."
        )
    return ProcessedBatch(
        model_inputs=dict(outputs),
        sample_ids=[sample["sample_id"]],
        prompt_lengths=prompt_lengths,
        metadata=metadata,
    )


def teacher_forced_objective(
    sample: Mapping[str, Any],
    *,
    probe_name: str,
) -> dict[str, Any]:
    """Validate that Taylor consumes the one documented native CE objective."""

    objective = sample.get("objective")
    if not isinstance(objective, Mapping):
        reason = sample.get("teacher_forcing_unavailable_reason")
        if reason is None:
            reason = sample.get("target_unavailable_reason")
        suffix = f" Reason: {reason}" if reason else ""
        raise BackendIntegrationError(
            f"{probe_name} requires teacher-forced objective metadata.{suffix}"
        )
    normalized = dict(objective)
    if normalized.get("objective") != OBJECTIVE_NAME:
        raise BackendIntegrationError(
            f"{probe_name} requires objective {OBJECTIVE_NAME!r}, got "
            f"{normalized.get('objective')!r}."
        )
    if normalized.get("normalization") != "mean_over_answer_tokens":
        raise BackendIntegrationError(
            f"{probe_name} requires mean-over-answer-token normalization."
        )
    if normalized.get("ignore_index") != IGNORE_INDEX:
        raise BackendIntegrationError(
            f"{probe_name} requires teacher-forced ignore_index {IGNORE_INDEX}."
        )
    answer_counts = normalized.get("answer_token_counts")
    if (
        not isinstance(answer_counts, (list, tuple))
        or len(answer_counts) != 1
        or int(answer_counts[0]) <= 0
    ):
        raise BackendIntegrationError(
            f"{probe_name} requires exactly one positive answer-token count."
        )
    outputs = sample.get("teacher_forced_processor_outputs")
    if not isinstance(outputs, Mapping) or outputs.get("labels") is None:
        raise BackendIntegrationError(
            f"{probe_name} requires teacher-forced labels."
        )
    label_rows = _rows(outputs["labels"])
    if len(label_rows) != 1:
        raise BackendIntegrationError(
            f"{probe_name} requires exactly one teacher-forced label row."
        )
    labeled_positions = tuple(
        index
        for index, value in enumerate(label_rows[0])
        if int(value) != IGNORE_INDEX
    )
    if len(labeled_positions) != int(answer_counts[0]):
        raise BackendIntegrationError(
            f"{probe_name} answer-token metadata does not match its labels."
        )
    declared_positions = normalized.get("answer_token_positions_by_sample")
    if (
        not isinstance(declared_positions, (list, tuple))
        or len(declared_positions) != 1
        or not isinstance(declared_positions[0], (list, tuple))
    ):
        raise BackendIntegrationError(
            f"{probe_name} answer-token positions do not match its labels."
        )
    try:
        normalized_positions = tuple(int(value) for value in declared_positions[0])
    except (TypeError, ValueError) as exc:
        raise BackendIntegrationError(
            f"{probe_name} answer-token positions must be integers."
        ) from exc
    if normalized_positions != labeled_positions:
        raise BackendIntegrationError(
            f"{probe_name} answer-token positions do not match its labels."
        )
    return normalized


def _valid_sequence_lengths(outputs: Mapping[str, Any]) -> list[int]:
    attention_mask = outputs.get("attention_mask")
    rows = _rows(attention_mask)
    if rows:
        return [sum(1 for value in row if int(value) != 0) for row in rows]
    input_rows = _rows(outputs.get("input_ids"))
    return [len(row) for row in input_rows]


def _rows(value: Any) -> list[list[Any]]:
    if value is None:
        return []
    tolist = getattr(value, "tolist", None)
    raw = tolist() if callable(tolist) else value
    if not isinstance(raw, (list, tuple)) or not raw:
        return []
    if not isinstance(raw[0], (list, tuple)):
        raw = [raw]
    return [list(row) for row in raw]


__all__ = [
    "analysis_processed_batch",
    "prepare_analysis_forward_state",
    "prepare_teacher_forced_analysis_forward_state",
    "teacher_forced_objective",
]
