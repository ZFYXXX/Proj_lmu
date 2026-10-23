"""Native Qwen3-VL input/output contract validation."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any

from mllm_comp.backbones.base import BackboneRuntime, GenerationResult, ProcessedBatch
from mllm_comp.core.invariants import ForwardStateInvariant, InvariantViolation
from mllm_comp.core.state import ForwardState


def _shape(value: Any) -> tuple[int, ...] | None:
    shape = getattr(value, "shape", None)
    if shape is not None:
        try:
            return tuple(int(dim) for dim in shape)
        except TypeError:
            return None
    if isinstance(value, (list, tuple)):
        if not value:
            return (0,)
        child = _shape(value[0])
        return (len(value), *(child or ()))
    return None


def _numel(value: Any) -> int | None:
    if value is None:
        return None
    method = getattr(value, "numel", None)
    if callable(method):
        return int(method())
    shape = _shape(value)
    if shape is None:
        return None
    result = 1
    for dim in shape:
        result *= dim
    return result


def _violation(
    name: str,
    field: str,
    expected: Any,
    actual: Any,
    sample_id: str | int | None = None,
) -> InvariantViolation:
    return InvariantViolation(
        name=name,
        field=field,
        expected=expected,
        actual=actual,
        sample_id=sample_id,
    )


@dataclass(frozen=True)
class Qwen3VLNativeValidator:
    """Validate Qwen3-VL native visual serialization and text output."""

    def validate_state(
        self,
        runtime: BackboneRuntime,
        state: ForwardState,
        processed_batch: ProcessedBatch,
    ) -> list[InvariantViolation]:
        del runtime
        violations = ForwardStateInvariant().check(state)
        input_shape = _shape(state.input_ids)
        attention_shape = _shape(state.attention_mask)
        if input_shape is None:
            violations.append(
                _violation("qwen3_vl_input_ids_required", "input_ids", "rank-2 tokens", None)
            )
        elif attention_shape is not None and attention_shape != input_shape:
            violations.append(
                _violation("qwen3_vl_attention_shape", "attention_mask", input_shape, attention_shape)
            )

        for value_name, metadata_name in (
            ("pixel_values", "image_grid_thw"),
            ("video_pixel_values", "video_grid_thw"),
        ):
            value = getattr(state, value_name)
            metadata = getattr(state, metadata_name)
            if (value is None) != (metadata is None):
                violations.append(
                    _violation(
                        "qwen3_vl_modality_pair",
                        value_name,
                        f"present iff {metadata_name} is present",
                        {value_name: value is not None, metadata_name: metadata is not None},
                    )
                )

        expected_by_sample = processed_batch.metadata.get("expected_token_counts_by_sample", [])
        actual_by_sample: dict[int, Counter[str]] = {}
        for span in state.modality_token_spans:
            actual_by_sample.setdefault(span.sample_index, Counter())[span.modality] += (
                span.end - span.start
            )
        for sample_index, expected in enumerate(expected_by_sample):
            actual = actual_by_sample.get(sample_index, Counter())
            for modality in ("image", "video"):
                expected_count = int(expected.get(modality, 0))
                actual_count = int(actual.get(modality, 0))
                if expected_count != actual_count:
                    violations.append(
                        _violation(
                            "qwen3_vl_placeholder_count",
                            f"modality_token_spans.{modality}",
                            expected_count,
                            actual_count,
                            processed_batch.sample_ids[sample_index],
                        )
                    )

        mm_token_type_ids = state.native_model_inputs.get("mm_token_type_ids")
        mm_shape = _shape(mm_token_type_ids)
        if input_shape is not None and mm_shape is not None and mm_shape != input_shape:
            violations.append(
                _violation(
                    "qwen3_vl_mm_token_type_shape",
                    "mm_token_type_ids",
                    input_shape,
                    mm_shape,
                )
            )
        if state.sample_ids != processed_batch.sample_ids:
            violations.append(
                _violation(
                    "qwen3_vl_sample_id_alignment",
                    "sample_ids",
                    processed_batch.sample_ids,
                    state.sample_ids,
                )
            )
        return violations

    def validate_generation(
        self,
        runtime: BackboneRuntime,
        state: ForwardState,
        processed_batch: ProcessedBatch,
        result: GenerationResult,
    ) -> list[InvariantViolation]:
        del runtime, state
        violations: list[InvariantViolation] = []
        batch_size = len(processed_batch.sample_ids)
        if result.text is None:
            violations.append(
                _violation("qwen3_vl_text_output_required", "text", "text output", None)
            )
        else:
            if len(result.text.text) != batch_size:
                violations.append(
                    _violation(
                        "qwen3_vl_text_batch",
                        "text.text",
                        batch_size,
                        len(result.text.text),
                    )
                )
            if _numel(result.text.token_ids) == 0:
                violations.append(
                    _violation(
                        "qwen3_vl_text_tokens_nonempty",
                        "text.token_ids",
                        "> 0 tokens",
                        0,
                    )
                )
        if result.speech is not None:
            violations.append(
                _violation("qwen3_vl_speech_unsupported", "speech", None, "present")
            )
        if result.sample_ids != processed_batch.sample_ids:
            violations.append(
                _violation(
                    "qwen3_vl_output_sample_id_alignment",
                    "sample_ids",
                    processed_batch.sample_ids,
                    result.sample_ids,
                )
            )
        return violations
