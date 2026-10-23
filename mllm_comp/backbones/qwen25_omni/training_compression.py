"""Training-side state construction for dynamic Qwen2.5-Omni compression."""

from __future__ import annotations

from collections import Counter
import inspect
from typing import Any, Callable, Mapping

from mllm_comp.core.errors import InvariantError
from mllm_comp.core.state import ForwardState

from .implementation import _contiguous_token_spans


def model_inputs_from_forward_call(
    forward: Callable[..., Any],
    args: tuple[Any, ...],
    kwargs: Mapping[str, Any],
) -> dict[str, Any]:
    """Normalize one trainer model call without dropping provider-native kwargs."""

    if not args:
        return dict(kwargs)
    try:
        signature = inspect.signature(forward)
        bound = signature.bind_partial(*args, **dict(kwargs))
    except (TypeError, ValueError) as exc:
        raise InvariantError(
            "Compressed Qwen2.5-Omni training could not bind positional model inputs."
        ) from exc
    values = dict(bound.arguments)
    values.pop("self", None)
    for parameter in signature.parameters.values():
        if parameter.kind is inspect.Parameter.VAR_KEYWORD:
            nested = values.pop(parameter.name, {})
            if isinstance(nested, Mapping):
                values.update(nested)
    return values


def _config_candidates(model: Any) -> list[Any]:
    thinker = getattr(model, "thinker", model)
    model_config = getattr(model, "config", None)
    return [
        getattr(thinker, "config", None),
        getattr(model_config, "thinker_config", None),
        model_config,
    ]


def _integer_config_value(model: Any, names: tuple[str, ...]) -> int | None:
    for config in _config_candidates(model):
        if config is None:
            continue
        for name in names:
            value = getattr(config, name, None)
            if value is None:
                continue
            if isinstance(value, (list, tuple, set)):
                candidates = sorted(value) if isinstance(value, set) else value
                for candidate in candidates:
                    if candidate is not None:
                        return int(candidate)
                continue
            return int(value)
    return None


def _integer_config_values(model: Any, names: tuple[str, ...]) -> list[int]:
    values: set[int] = set()
    for config in _config_candidates(model):
        if config is None:
            continue
        for name in names:
            value = getattr(config, name, None)
            if value is None:
                continue
            candidates = value if isinstance(value, (list, tuple, set)) else [value]
            values.update(int(item) for item in candidates if item is not None)
    return sorted(values)


def _padding_side(attention_mask: Any | None) -> str:
    if attention_mask is None:
        return "right"
    rows = attention_mask.tolist()
    saw_left_padding = False
    saw_right_padding = False
    for row in rows:
        values = [int(value) for value in row]
        nonzero = [index for index, value in enumerate(values) if value]
        if not nonzero:
            continue
        saw_left_padding = saw_left_padding or nonzero[0] > 0
        saw_right_padding = saw_right_padding or nonzero[-1] < len(values) - 1
    if saw_left_padding and saw_right_padding:
        raise InvariantError(
            "Compressed training does not support a batch mixing left and right padding."
        )
    return "left" if saw_left_padding else "right"


def build_qwen25_omni_training_state(
    model: Any,
    model_inputs: Mapping[str, Any],
    *,
    debug_semantics: bool = False,
) -> ForwardState:
    """Build the minimal legalizable state from an ms-swift training batch."""

    native_inputs = dict(model_inputs)
    input_ids = native_inputs.get("input_ids")
    shape = getattr(input_ids, "shape", None)
    if shape is None or len(shape) != 2:
        raise InvariantError(
            "Compressed Qwen2.5-Omni training requires rank-2 input_ids."
        )
    audio_token_id = _integer_config_value(
        model,
        ("audio_token_id", "audio_token_index"),
    )
    if audio_token_id is None:
        raise InvariantError(
            "Compressed Qwen2.5-Omni training could not resolve the audio token id "
            "from thinker.config."
        )
    pad_token_id = _integer_config_value(model, ("pad_token_id",)) or 0
    attention_mask = native_inputs.get("attention_mask")
    feature_attention_mask = native_inputs.get("feature_attention_mask")
    audio_feature_lengths: Any = []
    if feature_attention_mask is not None:
        audio_feature_lengths = feature_attention_mask.sum(dim=1)

    token_ids = {
        "image": _integer_config_value(
            model,
            ("image_token_id", "image_token_index"),
        ),
        "video": _integer_config_value(
            model,
            ("video_token_id", "video_token_index"),
        ),
        "audio": audio_token_id,
    }
    spans = _contiguous_token_spans(input_ids, token_ids)
    expected_by_sample = [Counter() for _ in range(int(shape[0]))]
    modality_counts_by_sample = [Counter() for _ in range(int(shape[0]))]
    for span in spans:
        expected_by_sample[span.sample_index][span.modality] += span.end - span.start
        modality_counts_by_sample[span.sample_index][span.modality] += 1
    metadata = {
        "phase": "train",
        "state_phase": "train",
        "provider": "ms-swift.Qwen2_5OmniTemplate",
        "token_ids": token_ids,
        "pad_token_id": pad_token_id,
        "bos_token_id": _integer_config_value(model, ("bos_token_id",)),
        "eos_token_id": _integer_config_value(model, ("eos_token_id",)),
        "user_token_id": _integer_config_value(model, ("user_token_id",)),
        "padding_side": _padding_side(attention_mask),
        "expected_token_counts_by_sample": [
            dict(counts) for counts in expected_by_sample
        ],
        "modality_counts_by_sample": [
            dict(counts) for counts in modality_counts_by_sample
        ],
    }
    if debug_semantics:
        modality_special_token_ids = {
            "audio": _integer_config_values(
                model,
                (
                    "audio_bos_token_id",
                    "audio_eos_token_id",
                    "audio_start_token_id",
                    "audio_end_token_id",
                ),
            ),
            "image": _integer_config_values(
                model,
                (
                    "image_bos_token_id",
                    "image_eos_token_id",
                    "image_start_token_id",
                    "image_end_token_id",
                ),
            ),
            "video": _integer_config_values(
                model,
                (
                    "video_bos_token_id",
                    "video_eos_token_id",
                    "video_start_token_id",
                    "video_end_token_id",
                ),
            ),
        }
        vision_special_token_ids = _integer_config_values(
            model,
            (
                "vision_start_token_id",
                "vision_end_token_id",
                "vision_bos_token_id",
                "vision_eos_token_id",
            ),
        )
        global_special_token_ids = _integer_config_values(
            model,
            ("bos_token_id",),
        )
        special_token_ids = set(global_special_token_ids)
        special_token_ids.update(
            _integer_config_values(
                model,
                ("eos_token_id", "pad_token_id"),
            )
        )
        special_token_ids.update(vision_special_token_ids)
        for values in modality_special_token_ids.values():
            special_token_ids.update(values)
        metadata.update(
            {
                "special_token_ids": sorted(special_token_ids),
                "global_special_token_ids": global_special_token_ids,
                "modality_special_token_ids": modality_special_token_ids,
                "vision_special_token_ids": vision_special_token_ids,
            }
        )
    return ForwardState(
        input_ids=input_ids,
        inputs_embeds=native_inputs.get("inputs_embeds"),
        attention_mask=attention_mask,
        position_ids=native_inputs.get("position_ids"),
        labels=native_inputs.get("labels"),
        token_type_mask=native_inputs.get("token_type_ids"),
        pixel_values=native_inputs.get("pixel_values"),
        image_grid_thw=native_inputs.get("image_grid_thw"),
        video_pixel_values=native_inputs.get("pixel_values_videos"),
        video_grid_thw=native_inputs.get("video_grid_thw"),
        audio_values=native_inputs.get("input_features"),
        audio_attention_mask=feature_attention_mask,
        audio_feature_lengths=audio_feature_lengths,
        video_second_per_grid=native_inputs.get("video_second_per_grid"),
        native_model_inputs=native_inputs,
        modality_token_spans=spans,
        modality_metadata=metadata,
        sample_ids=list(range(int(shape[0]))),
    )


__all__ = [
    "build_qwen25_omni_training_state",
    "model_inputs_from_forward_call",
]
