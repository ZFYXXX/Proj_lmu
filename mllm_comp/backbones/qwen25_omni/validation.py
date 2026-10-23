"""Native Qwen2.5-Omni input/output contract validation."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any

from mllm_comp.backbones.base import (
    BackboneRuntime,
    GenerationResult,
    ProcessedBatch,
)
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
class Qwen25OmniNativeValidator:
    """Validate canonicalization separately from compression legalization."""

    output_sample_rate: int = 24000

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
            violations.append(_violation("qwen_input_ids_required", "input_ids", "rank-2 tokens", None))
        elif attention_shape is not None and attention_shape != input_shape:
            violations.append(
                _violation("qwen_attention_shape", "attention_mask", input_shape, attention_shape)
            )

        required_pairs = (
            ("pixel_values", "image_grid_thw"),
            ("video_pixel_values", "video_grid_thw"),
            ("audio_values", "audio_attention_mask"),
        )
        for value_name, metadata_name in required_pairs:
            value = getattr(state, value_name)
            metadata = getattr(state, metadata_name)
            if (value is None) != (metadata is None):
                violations.append(
                    _violation(
                        "qwen_modality_pair",
                        value_name,
                        f"present iff {metadata_name} is present",
                        {value_name: value is not None, metadata_name: metadata is not None},
                    )
                )

        expected_by_sample = state.modality_metadata.get(
            "expected_token_counts_by_sample",
            processed_batch.metadata.get("expected_token_counts_by_sample", []),
        )
        actual_by_sample: dict[int, Counter[str]] = {}
        for span in state.modality_token_spans:
            actual_by_sample.setdefault(span.sample_index, Counter())[span.modality] += span.end - span.start
        for sample_index, expected in enumerate(expected_by_sample):
            actual = actual_by_sample.get(sample_index, Counter())
            for modality in ("image", "video", "audio"):
                expected_count = int(expected.get(modality, 0))
                actual_count = int(actual.get(modality, 0))
                if expected_count != actual_count:
                    violations.append(
                        _violation(
                            "qwen_placeholder_count",
                            f"modality_token_spans.{modality}",
                            expected_count,
                            actual_count,
                            processed_batch.sample_ids[sample_index],
                        )
                    )

        video_grid_shape = _shape(state.video_grid_thw)
        video_time_shape = _shape(state.video_second_per_grid)
        if video_grid_shape is not None and video_time_shape is not None:
            if video_grid_shape[0] != video_time_shape[0]:
                violations.append(
                    _violation(
                        "qwen_video_time_grid_alignment",
                        "video_second_per_grid",
                        video_grid_shape[0],
                        video_time_shape[0],
                    )
                )

        if state.sample_ids != processed_batch.sample_ids:
            violations.append(
                _violation(
                    "qwen_sample_id_alignment",
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
        del state
        violations: list[InvariantViolation] = []
        batch_size = len(processed_batch.sample_ids)
        if result.text is None:
            violations.append(_violation("qwen_text_output_required", "text", "text output", None))
        else:
            if len(result.text.text) != batch_size:
                violations.append(
                    _violation("qwen_text_batch", "text.text", batch_size, len(result.text.text))
                )
            if _numel(result.text.token_ids) == 0:
                violations.append(
                    _violation("qwen_text_tokens_nonempty", "text.token_ids", "> 0 tokens", 0)
                )

        return_audio = bool(result.metadata.get("return_audio", False))
        if return_audio and batch_size != 1:
            violations.append(
                _violation("qwen_speech_batch_size", "sample_ids", 1, batch_size)
            )
        if return_audio and result.speech is None:
            violations.append(
                _violation("qwen_speech_output_required", "speech", "waveform output", None)
            )
        if not return_audio and result.speech is not None:
            violations.append(
                _violation("qwen_unrequested_speech", "speech", None, "present")
            )
        if result.speech is not None:
            if _numel(result.speech.waveform) == 0:
                violations.append(
                    _violation("qwen_waveform_nonempty", "speech.waveform", "> 0 samples", 0)
                )
            expected_rate = int(runtime.metadata.get("speech_sample_rate", self.output_sample_rate))
            if result.speech.sample_rate != expected_rate:
                violations.append(
                    _violation(
                        "qwen_speech_sample_rate",
                        "speech.sample_rate",
                        expected_rate,
                        result.speech.sample_rate,
                    )
                )
            speaker_map = getattr(runtime.model, "speaker_map", {})
            if speaker_map and result.speech.speaker not in speaker_map:
                violations.append(
                    _violation(
                        "qwen_speaker_available",
                        "speech.speaker",
                        sorted(speaker_map),
                        result.speech.speaker,
                    )
                )
            codec_ids = result.speech.codec_token_ids
            if codec_ids is not None and _numel(codec_ids) == 0:
                violations.append(
                    _violation(
                        "qwen_talker_codec_nonempty",
                        "speech.codec_token_ids",
                        "> 0 codec tokens",
                        0,
                    )
                )
            if getattr(runtime.model, "token2wav", None) is not None and codec_ids is None:
                violations.append(
                    _violation(
                        "qwen_talker_codec_required",
                        "speech.codec_token_ids",
                        "captured Talker codec tokens",
                        None,
                    )
                )

            if result.traces is not None:
                trace_codec_ids = result.traces.talker_codec_token_ids
                token2wav_code = result.traces.token2wav_intermediates.get("code")
                if trace_codec_ids is not None and trace_codec_ids is not codec_ids:
                    violations.append(
                        _violation(
                            "qwen_talker_codec_trace_consistency",
                            "traces.talker_codec_token_ids",
                            "same object as speech.codec_token_ids",
                            "different object",
                        )
                    )
                if token2wav_code is not None and token2wav_code is not codec_ids:
                    violations.append(
                        _violation(
                            "qwen_token2wav_codec_handoff",
                            "traces.token2wav_intermediates.code",
                            "same object as speech.codec_token_ids",
                            "different object",
                        )
                    )
        return violations
