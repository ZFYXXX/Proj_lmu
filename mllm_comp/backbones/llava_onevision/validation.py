"""Native contract validation for LLaVA-OneVision text, image, and video."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from mllm_comp.backbones.base import BackboneRuntime, GenerationResult, ProcessedBatch
from mllm_comp.core.invariants import ForwardStateInvariant, InvariantViolation
from mllm_comp.core.state import ForwardState


def _shape(value: Any) -> tuple[int, ...] | None:
    shape = getattr(value, "shape", None)
    if shape is not None:
        try:
            return tuple(int(item) for item in shape)
        except TypeError:
            return None
    if isinstance(value, (list, tuple)):
        if not value:
            return (0,)
        child = _shape(value[0])
        return (len(value), *(child or ()))
    return None


def _numel(value: Any) -> int | None:
    method = getattr(value, "numel", None)
    if callable(method):
        return int(method())
    shape = _shape(value)
    if shape is None:
        return None
    result = 1
    for item in shape:
        result *= item
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
class LlavaOneVisionNativeValidator:
    """Validate placeholder/media identity and source-native generation output."""

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
                    "llava_input_ids_required",
                    "input_ids",
                    "rank-2 single-sample tokens",
                    input_shape,
                )
            )
        elif input_shape[0] != 1:
            violations.append(
                _violation("llava_single_sample", "input_ids", 1, input_shape[0])
            )
        if input_shape is not None and attention_shape != input_shape:
            violations.append(
                _violation(
                    "llava_attention_shape",
                    "attention_mask",
                    input_shape,
                    attention_shape,
                )
            )
        metadata = state.modality_metadata
        image_count = int(metadata.get("image_count", 0))
        video_count = int(metadata.get("video_count", 0))
        media_count = image_count + video_count
        image_sizes = tuple(metadata.get("image_sizes", ()))
        modalities = tuple(metadata.get("modalities", ()))
        if image_count and video_count:
            violations.append(
                _violation(
                    "llava_mixed_media_unsupported",
                    "modality_metadata",
                    "image or one video",
                    {"image_count": image_count, "video_count": video_count},
                )
            )
        if video_count not in {0, 1}:
            violations.append(
                _violation(
                    "llava_single_video",
                    "modality_metadata.video_count",
                    "0 or 1",
                    video_count,
                )
            )
        if len(image_sizes) != media_count:
            violations.append(
                _violation(
                    "llava_image_size_count",
                    "modality_metadata.image_sizes",
                    media_count,
                    len(image_sizes),
                )
            )
        if image_count and modalities != ("image",):
            violations.append(
                _violation(
                    "llava_native_modalities",
                    "modality_metadata.modalities",
                    ("image",),
                    modalities,
                )
            )
        if video_count and modalities != ("video",):
            violations.append(
                _violation(
                    "llava_native_video_modalities",
                    "modality_metadata.modalities",
                    ("video",),
                    modalities,
                )
            )
        if not media_count and any(item not in {"text"} for item in modalities):
            violations.append(
                _violation(
                    "llava_text_only_modalities",
                    "modality_metadata.modalities",
                    ("text",),
                    modalities,
                )
            )
        image_spans = [
            span for span in state.modality_token_spans if span.modality == "image"
        ]
        video_spans = [
            span for span in state.modality_token_spans if span.modality == "video"
        ]
        if len(image_spans) != image_count:
            violations.append(
                _violation(
                    "llava_image_placeholder_count",
                    "modality_token_spans",
                    image_count,
                    len(image_spans),
                    (
                        processed_batch.sample_ids[0]
                        if processed_batch.sample_ids
                        else None
                    ),
                )
            )
        if len(video_spans) != video_count:
            violations.append(
                _violation(
                    "llava_video_placeholder_count",
                    "modality_token_spans",
                    video_count,
                    len(video_spans),
                    (
                        processed_batch.sample_ids[0]
                        if processed_batch.sample_ids
                        else None
                    ),
                )
            )
        native = state.native_model_inputs
        if native.get("images") is not state.pixel_values:
            violations.append(
                _violation(
                    "llava_native_image_identity",
                    "native_model_inputs.images",
                    "same object as pixel_values",
                    "different object",
                )
            )
        if tuple(native.get("image_sizes", ())) != image_sizes:
            violations.append(
                _violation(
                    "llava_native_image_sizes",
                    "native_model_inputs.image_sizes",
                    image_sizes,
                    native.get("image_sizes"),
                )
            )
        if tuple(native.get("modalities", ())) != modalities:
            violations.append(
                _violation(
                    "llava_native_modalities_identity",
                    "native_model_inputs.modalities",
                    modalities,
                    native.get("modalities"),
                )
            )
        if video_count:
            video = metadata.get("video")
            if not isinstance(video, dict):
                violations.append(
                    _violation(
                        "llava_video_metadata",
                        "modality_metadata.video",
                        "decoded video metadata",
                        video,
                    )
                )
            else:
                frame_count = video.get("frame_count")
                frame_indices = tuple(video.get("frame_indices", ()))
                timestamps = tuple(video.get("timestamps_seconds", ()))
                if (
                    isinstance(frame_count, bool)
                    or not isinstance(frame_count, int)
                    or frame_count <= 0
                    or len(frame_indices) != frame_count
                    or len(timestamps) != frame_count
                ):
                    violations.append(
                        _violation(
                            "llava_video_frame_identity",
                            "modality_metadata.video",
                            "positive frame_count with matching indices/timestamps",
                            video,
                        )
                    )
        anyres = metadata.get("anyres")
        if not isinstance(anyres, dict) or anyres.get("layout_phase") != (
            "pre_encoder_pre_anyres_pack"
        ):
            violations.append(
                _violation(
                    "llava_anyres_metadata",
                    "modality_metadata.anyres",
                    "pre_encoder_pre_anyres_pack descriptor",
                    anyres,
                )
            )
        if state.sample_ids != processed_batch.sample_ids:
            violations.append(
                _violation(
                    "llava_sample_id_alignment",
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
        if result.text is None:
            violations.append(
                _violation("llava_text_output_required", "text", "text output", None)
            )
        else:
            if len(result.text.text) != 1:
                violations.append(
                    _violation(
                        "llava_text_batch", "text.text", 1, len(result.text.text)
                    )
                )
            if _numel(result.text.token_ids) == 0:
                violations.append(
                    _violation(
                        "llava_text_tokens_nonempty",
                        "text.token_ids",
                        "> 0 tokens",
                        0,
                    )
                )
        if result.speech is not None:
            violations.append(
                _violation("llava_speech_unsupported", "speech", None, "present")
            )
        if result.sample_ids != processed_batch.sample_ids:
            violations.append(
                _violation(
                    "llava_output_sample_id_alignment",
                    "sample_ids",
                    processed_batch.sample_ids,
                    result.sample_ids,
                )
            )
        return violations


__all__ = ["LlavaOneVisionNativeValidator"]
