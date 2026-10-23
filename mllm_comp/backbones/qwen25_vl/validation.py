"""Native Qwen2.5-VL input/output contract validation."""

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
class Qwen25VLNativeValidator:
    """Validate visual serialization and model-owned M-RoPE inputs."""

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
        if input_shape is None or len(input_shape) != 2:
            violations.append(
                _violation(
                    "qwen25_vl_input_ids_required",
                    "input_ids",
                    "rank-2 tokens",
                    input_shape,
                )
            )
        elif attention_shape is not None and attention_shape != input_shape:
            violations.append(
                _violation(
                    "qwen25_vl_attention_shape",
                    "attention_mask",
                    input_shape,
                    attention_shape,
                )
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
                        "qwen25_vl_modality_pair",
                        value_name,
                        f"present iff {metadata_name} is present",
                        {
                            value_name: value is not None,
                            metadata_name: metadata is not None,
                        },
                    )
                )

        expected_by_sample = processed_batch.metadata.get(
            "expected_token_counts_by_sample", []
        )
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
                            "qwen25_vl_placeholder_count",
                            f"modality_token_spans.{modality}",
                            expected_count,
                            actual_count,
                            processed_batch.sample_ids[sample_index],
                        )
                    )

        native = state.native_model_inputs
        mm_token_type_ids = native.get("mm_token_type_ids")
        mm_shape = _shape(mm_token_type_ids)
        if input_shape is not None and mm_shape is not None and mm_shape != input_shape:
            violations.append(
                _violation(
                    "qwen25_vl_mm_token_type_shape",
                    "mm_token_type_ids",
                    input_shape,
                    mm_shape,
                )
            )

        position_shape = _shape(state.position_ids)
        if position_shape is not None and input_shape is not None:
            valid_position_shapes = {
                input_shape,
                (3, *input_shape),
                # Transformers 4.57 also accepts a packed form whose leading
                # plane carries text-only positions before the three M-RoPE
                # temporal/height/width planes.
                (4, *input_shape),
            }
            if position_shape not in valid_position_shapes:
                violations.append(
                    _violation(
                        "qwen25_vl_mrope_position_shape",
                        "position_ids",
                        sorted(valid_position_shapes),
                        position_shape,
                    )
                )

        video_grid_shape = _shape(state.video_grid_thw)
        second_per_grid_ts = native.get("second_per_grid_ts")
        if video_grid_shape is not None:
            expected_videos = video_grid_shape[0] if len(video_grid_shape) == 2 else None
            actual_seconds = _numel(second_per_grid_ts)
            if actual_seconds != expected_videos:
                violations.append(
                    _violation(
                        "qwen25_vl_video_mrope_timing",
                        "second_per_grid_ts",
                        f"one value per video ({expected_videos})",
                        actual_seconds,
                    )
                )
        elif second_per_grid_ts is not None:
            violations.append(
                _violation(
                    "qwen25_vl_video_mrope_timing",
                    "second_per_grid_ts",
                    None,
                    "present without video_grid_thw",
                )
            )

        if state.sample_ids != processed_batch.sample_ids:
            violations.append(
                _violation(
                    "qwen25_vl_sample_id_alignment",
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
                _violation(
                    "qwen25_vl_text_output_required",
                    "text",
                    "text output",
                    None,
                )
            )
        else:
            if len(result.text.text) != batch_size:
                violations.append(
                    _violation(
                        "qwen25_vl_text_batch",
                        "text.text",
                        batch_size,
                        len(result.text.text),
                    )
                )
            if _numel(result.text.token_ids) == 0:
                violations.append(
                    _violation(
                        "qwen25_vl_text_tokens_nonempty",
                        "text.token_ids",
                        "> 0 tokens",
                        0,
                    )
                )
        if result.speech is not None:
            violations.append(
                _violation("qwen25_vl_speech_unsupported", "speech", None, "present")
            )
        if result.sample_ids != processed_batch.sample_ids:
            violations.append(
                _violation(
                    "qwen25_vl_output_sample_id_alignment",
                    "sample_ids",
                    processed_batch.sample_ids,
                    result.sample_ids,
                )
            )
        return violations


__all__ = ["Qwen25VLNativeValidator"]
