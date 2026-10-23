"""lmms-eval model protocol backed by a platform-owned runtime.

The bridge is intentionally created around an existing :class:`BackboneRuntime`.
It must never call ``from_pretrained``: loading, compression and runtime ownership
stay on the MLLM-Comp-Platform side of the boundary.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib
import json
from pathlib import Path
from typing import Any

from mllm_comp.backbones.base import BackboneFacade, BackboneRuntime
from mllm_comp.backbones.llava_onevision import LlavaOneVisionSample
from mllm_comp.backbones.qwen25_omni import (
    DEFAULT_AUDIO_SYSTEM_PROMPT,
    Qwen25OmniSample,
)
from mllm_comp.backbones.qwen25_vl import (
    DEFAULT_SYSTEM_PROMPT as QWEN25_VL_DEFAULT_SYSTEM_PROMPT,
    Qwen25VLSample,
)
from mllm_comp.backbones.qwen3_vl import (
    DEFAULT_SYSTEM_PROMPT as QWEN3_VL_DEFAULT_SYSTEM_PROMPT,
    Qwen3VLSample,
)
from mllm_comp.backbones.qwen3_5 import Qwen3_5Sample
from mllm_comp.core.errors import BackendIntegrationError
from mllm_comp.core.serialization import atomic_write_json, to_jsonable
from mllm_comp.core.vocabulary import RuntimeMode
from mllm_comp.compressors.image.sgl import sgl_target_generation
from mllm_comp.integrations.native_cli import installed_versions
from mllm_comp.pipeline.runtime_session import (
    refresh_runtime_compression_metadata,
)
from mllm_comp.pipeline.trace import CompressionTraceWriter


_VIDEO_SUFFIXES = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
_AUDIO_SUFFIXES = {".wav", ".mp3", ".flac", ".m4a", ".ogg", ".opus"}
_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif", ".tiff"}
_QWEN25_CHAT_IMAGE_KWARGS = {"max_pixels": 512 * 512}
_QWEN25_CHAT_VIDEO_KWARGS = {
    "fps": 2,
    "max_frames": 16,
    "max_pixels": 384 * 384,
}
_QWEN3_CHAT_VIDEO_KWARGS = {
    "min_pixels": 256 * 28 * 28,
    "nframes": 32,
    "max_pixels": 1605632,
}
_QWEN_VL_DEFAULT_MIN_PIXELS = 256 * 28 * 28
_QWEN_VL_DEFAULT_MAX_PIXELS = 1605632
_OMNI_GENERATION_KEY_MAP = {
    "max_new_tokens": "thinker_max_new_tokens",
    "min_new_tokens": "thinker_min_new_tokens",
    "do_sample": "thinker_do_sample",
    "temperature": "thinker_temperature",
    "top_p": "thinker_top_p",
    "top_k": "thinker_top_k",
    "num_beams": "thinker_num_beams",
    "repetition_penalty": "thinker_repetition_penalty",
    "length_penalty": "thinker_length_penalty",
    "use_cache": "thinker_use_cache",
}
_INTERNAL_PROCESSOR_DIAGNOSTIC_FIELDS = frozenset(
    {
        "chat_template_text_by_sample",
        "semantic_char_spans_by_sample",
        "semantic_message_prefixes_by_sample",
        "semantic_token_labels_by_sample",
    }
)


def _is_nonstring_sequence(value: Any) -> bool:
    return isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray))


def _freeze_request_group(value: Any) -> Any:
    """Build a stable hashable key for lmms-eval generation kwargs."""

    if isinstance(value, Mapping):
        return tuple(
            (str(key), _freeze_request_group(item))
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        )
    if _is_nonstring_sequence(value):
        return tuple(_freeze_request_group(item) for item in value)
    try:
        hash(value)
    except TypeError:
        return repr(value)
    return value


def _native_simple_request_order(
    request_args: Sequence[tuple[Any, ...]],
    *,
    sort_key: Any,
) -> tuple[list[tuple[Any, ...]], Any]:
    """Mirror the pinned lmms-eval Collator behavior used by native VLMs.

    Requests remain grouped by generation kwargs, each group is sorted by the
    model-specific context key, and responses are restored to caller order.
    Keeping this small protocol local preserves import-time isolation when the
    optional lmms-eval package is not installed.
    """

    groups: dict[Any, list[tuple[int, tuple[Any, ...]]]] = {}
    for index, args in enumerate(request_args):
        generation_kwargs = args[1] if len(args) > 1 else None
        key = _freeze_request_group(generation_kwargs)
        groups.setdefault(key, []).append((index, args))

    ordered: list[tuple[Any, ...]] = []
    original_indices: list[int] = []
    for entries in groups.values():
        for original_index, args in sorted(
            entries,
            key=lambda item: sort_key(item[1]),
        ):
            original_indices.append(original_index)
            ordered.append(args)

    def restore(responses: Sequence[str]) -> list[str]:
        if len(responses) != len(original_indices):
            raise BackendIntegrationError(
                "lmms-eval response count does not match the reordered requests."
            )
        result: list[str | None] = [None] * len(original_indices)
        for original_index, response in zip(original_indices, responses):
            result[original_index] = response
        if any(response is None for response in result):
            raise BackendIntegrationError(
                "lmms-eval request ordering did not cover every response."
            )
        return [str(response) for response in result]

    return ordered, restore


def _is_pil_image(value: Any) -> bool:
    value_type = type(value)
    return value_type.__module__.startswith("PIL.") and value_type.__name__ != "ImageModule"


def _normalize_task_media(value: Any) -> list[Any]:
    """Flatten task media and translate lmms-eval's typed URL records."""
    if value is None:
        return []
    if _is_nonstring_sequence(value):
        flattened: list[Any] = []
        for item in value:
            flattened.extend(_normalize_task_media(item))
        return flattened
    if (isinstance(value, Mapping)
            and value.get("type") in {"image", "video", "audio"}
            and "url" in value):
        value = dict(value)
        value[value["type"]] = value.pop("url")
    return [value]


def _llava_media_item(value: Any) -> tuple[str, Any]:
    """Classify video/audio explicitly while preserving native image objects."""

    if isinstance(value, (str, Path)):
        path = str(value)
        suffix = Path(path).suffix.lower()
        if suffix in _VIDEO_SUFFIXES:
            return "video", path
        if suffix in _AUDIO_SUFFIXES:
            return "audio", path
        return "image", value
    if isinstance(value, Mapping):
        media_types = tuple(
            media_type
            for media_type in ("video", "audio", "image")
            if media_type in value
        )
        if len(media_types) > 1:
            raise BackendIntegrationError(
                "LLaVA-OneVision media records must name exactly one media type."
            )
        if media_types:
            media_type = media_types[0]
            return media_type, value[media_type]
    return "image", value


def _resample_audio(array: Any, source_rate: int, target_rate: int) -> Any:
    if source_rate == target_rate:
        return array
    try:
        import librosa
        import numpy as np
    except ImportError as exc:  # pragma: no cover - exercised in dependency preflight
        raise BackendIntegrationError(
            "lmms-eval audio requests need librosa and numpy when the dataset "
            f"sample rate ({source_rate}) differs from the processor rate ({target_rate})."
        ) from exc
    return librosa.resample(
        np.asarray(array),
        orig_sr=source_rate,
        target_sr=target_rate,
    ).astype("float32")


def _media_content_item(value: Any, *, audio_sample_rate: int) -> dict[str, Any]:
    if _is_pil_image(value):
        return {"type": "image", "image": value}

    if isinstance(value, (str, Path)):
        path = str(value)
        suffix = Path(path).suffix.lower()
        if suffix in _VIDEO_SUFFIXES:
            return {"type": "video", "video": path}
        if suffix in _AUDIO_SUFFIXES:
            return {"type": "audio", "audio": path}
        if suffix in _IMAGE_SUFFIXES or path.startswith(("http://", "https://", "data:image/")):
            return {"type": "image", "image": path}
        raise BackendIntegrationError(
            f"lmms-eval returned an unsupported media path {path!r}; "
            "expected an image, audio or video extension."
        )

    if isinstance(value, Mapping):
        if "array" in value and "sampling_rate" in value:
            array = _resample_audio(
                value["array"],
                int(value["sampling_rate"]),
                audio_sample_rate,
            )
            return {"type": "audio", "audio": array}
        if value.get("type") == "audio" and "url" in value:
            normalized = dict(value)
            normalized["audio"] = normalized.pop("url")
            return normalized
        for media_type, key in (("image", "image"), ("video", "video"), ("audio", "audio")):
            if key in value:
                return {**dict(value), "type": media_type, key: value[key]}

    raise BackendIntegrationError(
        "lmms-eval returned an unsupported media value of type "
        f"{type(value).__module__}.{type(value).__qualname__}."
    )


def _encode_qwen25_vl_image(value: Any) -> str:
    """Use the same lossy data-URL conversion as lmms-eval's Qwen2.5-VL model."""

    try:
        from lmms_eval.models.model_utils.media_encoder import encode_image_to_data_url
    except ImportError as exc:
        raise BackendIntegrationError(
            "Qwen2.5-VL lmms-eval image parity requires lmms-eval's media encoder."
        ) from exc
    return encode_image_to_data_url(
        value,
        image_format="JPEG",
        mime_type="image/jpeg",
        convert_rgb=True,
        quality=85,
    )


def _processor_pixel_bounds(processor: Any) -> tuple[int, int]:
    image_processor = getattr(processor, "image_processor", None)
    size = getattr(image_processor, "size", {})
    if not isinstance(size, Mapping):
        size = {}
    min_pixels = getattr(image_processor, "min_pixels", None)
    max_pixels = getattr(image_processor, "max_pixels", None)
    if min_pixels is None:
        min_pixels = size.get("shortest_edge", _QWEN_VL_DEFAULT_MIN_PIXELS)
    if max_pixels is None:
        max_pixels = size.get("longest_edge", _QWEN_VL_DEFAULT_MAX_PIXELS)
    return int(min_pixels), int(max_pixels)


def _normalize_until(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if _is_nonstring_sequence(value) and all(isinstance(item, str) for item in value):
        return list(value)
    raise BackendIntegrationError("lmms-eval generation key 'until' must be a string or list of strings.")


def _truncate_at_stop(text: str, stops: list[str]) -> str:
    indexes = [text.find(stop) for stop in stops if stop and stop in text]
    return text[: min(indexes)] if indexes else text


def _trace_jsonable(value: Any, *, include_values: bool = True) -> Any:
    if isinstance(value, Mapping):
        return {
            str(key): _trace_jsonable(inner, include_values=include_values)
            for key, inner in value.items()
        }
    if _is_nonstring_sequence(value):
        return [_trace_jsonable(item, include_values=include_values) for item in value]
    shape = getattr(value, "shape", None)
    dtype = getattr(value, "dtype", None)
    if shape is not None:
        record: dict[str, Any] = {
            "shape": [int(item) for item in shape],
            "dtype": str(dtype),
        }
        if include_values:
            tolist = getattr(value, "tolist", None)
            if callable(tolist):
                record["values"] = tolist()
        else:
            detach = getattr(value, "detach", None)
            cpu = getattr(value, "cpu", None)
            if callable(detach) and callable(cpu):
                tensor = value.detach().cpu()
                if str(getattr(tensor, "dtype", "")) == "torch.bfloat16":
                    tensor = tensor.float()
                numpy = getattr(tensor, "numpy", None)
                if callable(numpy):
                    record["sha256"] = hashlib.sha256(
                        numpy().tobytes()
                    ).hexdigest()
        return record
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)


def _output_field(output: Any, name: str) -> Any | None:
    if output is None:
        return None
    if isinstance(output, Mapping):
        return output.get(name)
    return getattr(output, name, None)


def _semantic_token_spans_by_sample(value: Any) -> list[list[dict[str, Any]]] | None:
    """Run-length encode per-token labels for human-readable diagnostics."""

    if not isinstance(value, list):
        return None
    encoded: list[list[dict[str, Any]]] = []
    for row in value:
        if not isinstance(row, list):
            return None
        spans: list[dict[str, Any]] = []
        for position, raw_label in enumerate(row):
            label = str(raw_label)
            if spans and spans[-1]["label"] == label:
                spans[-1]["end"] = position + 1
                spans[-1]["length"] += 1
                continue
            spans.append(
                {
                    "start": position,
                    "end": position + 1,
                    "length": 1,
                    "label": label,
                }
            )
        encoded.append(spans)
    return encoded


def _diagnostic_processor_metadata(metadata: Mapping[str, Any]) -> dict[str, Any]:
    result = {
        name: value
        for name, value in metadata.items()
        if name not in _INTERNAL_PROCESSOR_DIAGNOSTIC_FIELDS
    }
    semantic_spans = _semantic_token_spans_by_sample(
        metadata.get("semantic_token_labels_by_sample")
    )
    if semantic_spans is not None:
        result["semantic_token_span_format"] = {
            "coordinate_space": "processor_input_token_index",
            "interval": "[start,end)",
        }
        result["semantic_token_spans_by_sample"] = semantic_spans
    return result


def _runtime_environment(runtime: BackboneRuntime) -> dict[str, Any]:
    environment: dict[str, Any] = {
        "packages": installed_versions(primary={}),
        "device": str(runtime.device),
        "dtype": str(getattr(runtime, "dtype", None)),
        "cuda_runtime": None,
        "cudnn": None,
        "gpu_name": None,
        "gpu_capability": None,
    }
    try:
        import torch
    except ImportError:
        return environment
    cuda_runtime = torch.version.cuda
    environment["cuda_runtime"] = (
        str(cuda_runtime) if cuda_runtime is not None else None
    )
    environment["cudnn"] = (
        int(torch.backends.cudnn.version())
        if torch.backends.cudnn.is_available()
        else None
    )
    device = torch.device(runtime.device) if runtime.device is not None else None
    if device is not None and device.type == "cuda" and torch.cuda.is_available():
        environment["gpu_name"] = torch.cuda.get_device_name(device)
        environment["gpu_capability"] = list(torch.cuda.get_device_capability(device))
    return environment


class LMMSEvalRequestBridge:
    """Translate lmms-eval ``Instance`` objects to the platform backbone API."""

    def __init__(
        self,
        *,
        facade: BackboneFacade,
        runtime: BackboneRuntime,
        system_prompt: str | None = None,
        use_audio_in_video: bool = False,
        generation_overrides: Mapping[str, Any] | None = None,
        diagnostic_trace_path: str | None = None,
        compression_manifest_path: str | None = None,
        request_mode: str = "generate",
    ) -> None:
        backbone_name = facade.capabilities().name
        if backbone_name not in {
            "llava_onevision",
            "qwen25_omni",
            "qwen25_vl",
            "qwen3_vl",
            "qwen3_5",
        }:
            raise BackendIntegrationError(
                "The active lmms-eval bridge supports llava_onevision, "
                "qwen25_omni, qwen25_vl, qwen3_vl and qwen3_5 backbones."
            )
        if backbone_name in {"qwen25_vl", "qwen3_vl", "qwen3_5"} and use_audio_in_video:
            raise BackendIntegrationError(
                f"{backbone_name} has no audio-in-video branch."
            )
        if request_mode not in {"generate", "forward_probe"}:
            raise BackendIntegrationError(
                f"Unsupported lmms-eval request_mode={request_mode!r}."
            )
        if request_mode == "forward_probe" and backbone_name == "qwen25_omni":
            raise BackendIntegrationError(
                "forward_probe is implemented only for ImageLLM backbones."
            )
        self.facade = facade
        self.runtime = runtime
        self.backbone_name = backbone_name
        self.request_mode = request_mode
        if system_prompt is not None:
            self.system_prompt = system_prompt
        elif backbone_name == "qwen25_omni":
            self.system_prompt = DEFAULT_AUDIO_SYSTEM_PROMPT
        elif backbone_name == "qwen25_vl":
            self.system_prompt = QWEN25_VL_DEFAULT_SYSTEM_PROMPT
        elif backbone_name == "qwen3_vl":
            self.system_prompt = QWEN3_VL_DEFAULT_SYSTEM_PROMPT
        elif backbone_name == "qwen3_5":
            self.system_prompt = QWEN3_VL_DEFAULT_SYSTEM_PROMPT
        else:
            self.system_prompt = None
        self.use_audio_in_video = bool(use_audio_in_video)
        self.generation_overrides = dict(generation_overrides or {})
        self.min_pixels, self.max_pixels = _processor_pixel_bounds(runtime.processor)
        self.diagnostic_trace_path = diagnostic_trace_path
        self.compression_manifest_path = compression_manifest_path
        self._compression_trace = (
            CompressionTraceWriter(
                compression_manifest_path,
                model_class=type(self.runtime.model).__qualname__,
            )
            if compression_manifest_path else None
        )
        self.runtime_environment = (
            _runtime_environment(runtime) if self.diagnostic_trace_path else None
        )
        self.task_dict: dict[str, Any] = {}
        if self.diagnostic_trace_path:
            # Each distributed worker receives a rank-specific path. Start a
            # fresh trace for every model instance so a reused run directory
            # cannot silently mix records from earlier evaluations.
            path = Path(self.diagnostic_trace_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("", encoding="utf-8")
        if self.compression_manifest_path:
            self._write_compression_manifest(last_request=None)

    def _write_compression_manifest(
        self,
        *,
        last_request: Mapping[str, Any] | None,
    ) -> None:
        if not self.compression_manifest_path:
            return
        pipeline_split = self.runtime.metadata.get("_pipeline_split_session")
        if pipeline_split is not None:
            self._compression_trace.write(pipeline_split.metadata(), last_request=last_request)
            return
        split_session = self.runtime.metadata.get("_modality_split_session")
        if split_session is not None:
            from mllm_comp.core.serialization import atomic_write_json
            atomic_write_json(self.compression_manifest_path, {
                "schema_version": 1, "compression_mode": "compressed",
                "modality_split": split_session.metadata(), "last_request": last_request,
            })
            return
        compression = refresh_runtime_compression_metadata(self.runtime)
        if not isinstance(compression, Mapping) or compression.get("enabled") is not True:
            raise BackendIntegrationError(
                "Compression manifest was requested without an active compression stage."
            )
        self._compression_trace.write(compression, last_request=last_request)

    def _write_diagnostic_trace(self, payload: Mapping[str, Any]) -> None:
        if not self.diagnostic_trace_path:
            return
        path = Path(self.diagnostic_trace_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(_trace_jsonable(payload), ensure_ascii=False))
            handle.write("\n")

    @property
    def tokenizer(self) -> Any:
        return self.runtime.processor.tokenizer

    @property
    def config(self) -> Any:
        return self.runtime.model.config

    @property
    def device(self) -> Any:
        return self.runtime.device or getattr(self.runtime.model, "device", None)

    def _document(self, task: str, split: str, doc_id: Any) -> Any:
        try:
            return self.task_dict[task][split][doc_id]
        except (KeyError, IndexError, TypeError) as exc:
            raise BackendIntegrationError(
                f"Cannot resolve lmms-eval document task={task!r}, split={split!r}, doc_id={doc_id!r}."
            ) from exc

    def _conversation(self, context: Any, visual: Any, *, audio_sample_rate: int) -> list[dict[str, Any]]:
        if not isinstance(context, str):
            raise BackendIntegrationError(
                "The simple lmms-eval bridge expects a string context; chat/multi-round tasks "
                "need a dedicated chat bridge."
            )
        if self.backbone_name in {"qwen25_vl", "qwen3_vl", "qwen3_5"}:
            # Both native lmms-eval Qwen-VL models remove the task placeholder;
            # the processor inserts visual placeholders from the structured media.
            context = context.replace("<image>", "")
        content = []
        for value in _normalize_task_media(visual):
            item = _media_content_item(value, audio_sample_rate=audio_sample_rate)
            if self.backbone_name == "qwen25_vl" and _is_pil_image(value):
                item["image"] = _encode_qwen25_vl_image(value)
            if self.backbone_name in {"qwen25_vl", "qwen3_vl", "qwen3_5"}:
                if item.get("type") == "image":
                    item.update(
                        min_pixels=self.min_pixels,
                        max_pixels=self.max_pixels,
                    )
                elif item.get("type") == "video":
                    item.update(
                        min_pixels=self.min_pixels,
                        max_pixels=self.max_pixels,
                    )
                    if self.backbone_name == "qwen3_vl":
                        item.setdefault("nframes", 32)
            content.append(item)
        if self.backbone_name in {"qwen25_vl", "qwen3_vl", "qwen3_5"} and any(
            item.get("type") == "audio" for item in content
        ):
            raise BackendIntegrationError(
                f"{self.backbone_name} lmms-eval requests cannot contain audio."
            )
        content.append({"type": "text", "text": context})
        messages: list[dict[str, Any]] = []
        if self.system_prompt:
            messages.append(
                {
                    "role": "system",
                    "content": [{"type": "text", "text": self.system_prompt}],
                }
            )
        messages.append({"role": "user", "content": content})
        return messages

    def _chat_media_kwargs(self) -> tuple[dict[str, Any], dict[str, Any]]:
        """Return the media arguments used by lmms-eval's native chat model."""

        if self.backbone_name == "qwen25_vl":
            # Match the simple path and scorer: qwen-vl-utils must perform the
            # final resize, not round once at default bounds and again inside
            # the processor. Double rounding changes grounding pixel geometry.
            return ({"min_pixels": self.min_pixels, "max_pixels": self.max_pixels},
                    dict(_QWEN3_CHAT_VIDEO_KWARGS))
        if self.backbone_name == "qwen25_omni":
            try:
                from lmms_eval.models.chat._chat_base import (
                    IMAGE_KWARGS,
                    VIDEO_KWARGS,
                )
            except ImportError:
                image_kwargs = _QWEN25_CHAT_IMAGE_KWARGS
                video_kwargs = _QWEN25_CHAT_VIDEO_KWARGS
            else:
                image_kwargs = IMAGE_KWARGS
                video_kwargs = VIDEO_KWARGS
            return (
                dict(image_kwargs),
                dict(video_kwargs),
            )
        return {}, dict(_QWEN3_CHAT_VIDEO_KWARGS)

    def _chat_conversation(self, raw_messages: Any) -> list[dict[str, Any]]:
        """Normalize ``doc_to_messages`` with lmms-eval's own chat protocol."""

        if not _is_nonstring_sequence(raw_messages):
            raise BackendIntegrationError(
                "lmms-eval doc_to_messages must return a sequence of chat messages."
            )
        messages = list(raw_messages)
        has_task_system_prompt = any(
            isinstance(message, Mapping) and message.get("role") == "system"
            for message in messages
        )
        if self.system_prompt is not None and not has_task_system_prompt:
            messages.insert(
                0,
                {
                    "role": "system",
                    "content": [{"type": "text", "text": self.system_prompt}],
                },
            )
        try:
            from lmms_eval.protocol import ChatMessages

            chat_messages = ChatMessages(**{"messages": messages})
            image_kwargs, video_kwargs = self._chat_media_kwargs()
            conversation = chat_messages.to_hf_messages(
                image_kwargs=image_kwargs,
                video_kwargs=video_kwargs,
            )
        except (ImportError, TypeError, ValueError) as exc:
            raise BackendIntegrationError(
                f"Cannot normalize lmms-eval doc_to_messages output: {exc}"
            ) from exc
        if self.backbone_name in {"qwen25_vl", "qwen3_vl", "qwen3_5"} and any(
            content.get("type") == "audio"
            for message in conversation
            for content in message.get("content", [])
        ):
            raise BackendIntegrationError(
                f"{self.backbone_name} lmms-eval requests cannot contain audio."
            )
        return conversation

    def _sample_from_conversation(
        self,
        conversation: list[dict[str, Any]],
        *,
        sample_id: str | int,
    ) -> Qwen25OmniSample | Qwen25VLSample | Qwen3VLSample | Qwen3_5Sample:
        if self.backbone_name == "qwen25_omni":
            return Qwen25OmniSample(
                sample_id=sample_id,
                conversation=conversation,
                use_audio_in_video=self.use_audio_in_video,
            )
        if self.backbone_name == "qwen25_vl":
            return Qwen25VLSample(sample_id=sample_id, conversation=conversation)
        if self.backbone_name == "qwen3_vl":
            return Qwen3VLSample(sample_id=sample_id, conversation=conversation)
        if self.backbone_name == "qwen3_5":
            return Qwen3_5Sample(sample_id=sample_id, conversation=conversation)
        raise BackendIntegrationError(
            "LLaVA-OneVision is activated through lmms-eval's simple "
            "doc_to_visual protocol; chat requests are not part of this slice."
        )

    def build_sample(
        self,
        *,
        context: str,
        visual: Any,
        sample_id: str | int,
    ) -> LlavaOneVisionSample | Qwen25OmniSample | Qwen25VLSample | Qwen3VLSample | Qwen3_5Sample:
        """Build the same platform sample used by evaluation and analysis."""

        if self.backbone_name == "llava_onevision":
            if not isinstance(context, str):
                raise BackendIntegrationError(
                    "The LLaVA-OneVision lmms-eval bridge expects a string context."
                )
            media = _normalize_task_media(visual)
            content = [_llava_media_item(item) for item in media]
            videos = [value for media_type, value in content if media_type == "video"]
            if any(media_type == "audio" for media_type, _value in content):
                raise BackendIntegrationError(
                    "LLaVA-OneVision lmms-eval requests cannot contain audio."
                )
            if len(videos) > 1 or (videos and len(content) != 1):
                raise BackendIntegrationError(
                    "LLaVA-OneVision lmms-eval supports one video or image media, "
                    "not multi-video/mixed requests."
                )
            return LlavaOneVisionSample(
                sample_id=sample_id,
                prompt=context,
                images=(
                    () if videos else tuple(value for _media_type, value in content)
                ),
                video=videos[0] if videos else None,
            )

        audio_sample_rate = int(
            getattr(
                getattr(self.runtime.processor, "feature_extractor", None),
                "sampling_rate",
                16000,
            )
        )
        conversation = self._conversation(
            context,
            visual,
            audio_sample_rate=audio_sample_rate,
        )
        return self._sample_from_conversation(conversation, sample_id=sample_id)

    def build_chat_sample(
        self,
        *,
        raw_messages: Any,
        sample_id: str | int,
    ) -> Qwen25OmniSample | Qwen25VLSample | Qwen3VLSample | Qwen3_5Sample:
        """Build a platform sample from lmms-eval ``doc_to_messages`` output."""

        return self._sample_from_conversation(
            self._chat_conversation(raw_messages),
            sample_id=sample_id,
        )

    def _generation_kwargs(self, request_kwargs: Mapping[str, Any]) -> tuple[dict[str, Any], list[str]]:
        # lmms-eval task YAML owns the request contract.  Platform generation
        # values are defaults only; an explicit task value must never be
        # overwritten on the way to the model.
        merged = dict(self.generation_overrides)
        merged.update(request_kwargs)
        until_present = "until" in merged
        raw_until = merged.pop("until", None)
        if (
            not until_present
            and self.backbone_name in {"qwen25_vl", "qwen3_vl", "qwen3_5"}
        ):
            eos_token_id = getattr(self.tokenizer, "eos_token_id", None)
            raw_until = (
                []
                if eos_token_id is None
                else [self.tokenizer.decode(eos_token_id)]
            )
        stops = _normalize_until(raw_until)
        if self.backbone_name in {"qwen25_vl", "qwen3_vl", "qwen3_5"}:
            # Native Qwen-VL deliberately ignores this common task stopper.
            stops = [item for item in stops if item != "\n\n"]
        if merged.pop("return_audio", False):
            raise BackendIntegrationError(
                "lmms-eval metric tasks currently accept text responses only; "
                "Talker output must be validated through the platform speech path."
            )

        translated: dict[str, Any] = {}
        if self.backbone_name == "llava_onevision":
            # Match lmms_eval.models.simple.llava_onevision.generate_until.
            # In particular, absence of do_sample is meaningful: transformers
            # must inherit it from the checkpoint's generation_config.
            translated.update(merged)
            translated.pop("image_aspect_ratio", None)
            translated.setdefault("max_new_tokens", 1024)
            translated.setdefault("use_cache", True)
            pad_token_id = getattr(self.tokenizer, "pad_token_id", None)
            if pad_token_id is None:
                pad_token_id = getattr(self.tokenizer, "eos_token_id", None)
            if pad_token_id is not None:
                translated.setdefault("pad_token_id", pad_token_id)
            if not translated.get("do_sample", False):
                translated.pop("temperature", None)
                translated.pop("top_p", None)
                translated.pop("top_k", None)
            # The upstream image path removes `until` but does not install a
            # string stopping criterion.  Preserve that behavior for parity.
            return translated, []

        if self.backbone_name == "qwen25_omni":
            translated["return_audio"] = False
            for key, value in merged.items():
                target = _OMNI_GENERATION_KEY_MAP.get(key, key)
                if target in translated:
                    raise BackendIntegrationError(
                        f"Generation keys {key!r} and {target!r} resolve to the same model argument."
                    )
                translated[target] = value
            temperature = translated.get("thinker_temperature")
            translated.setdefault(
                "thinker_do_sample", bool(temperature and float(temperature) > 0)
            )
            translated.setdefault("thinker_max_new_tokens", 128)
            if not translated["thinker_do_sample"]:
                if translated.get("thinker_temperature") in (0, 0.0):
                    translated.pop("thinker_temperature", None)
                translated.pop("thinker_top_p", None)
                translated.pop("thinker_top_k", None)
            return translated, stops

        if self.backbone_name == "qwen25_vl":
            # Match lmms_eval.models.simple.qwen2_5_vl exactly. The native
            # wrapper derives sampling from temperature and ignores do_sample.
            current = {
                "max_new_tokens": 32768,
                "temperature": 0.0,
                "top_p": None,
                "num_beams": 1,
                **merged,
            }
            do_sample = bool(float(current["temperature"]) > 0)
            translated = {
                "eos_token_id": getattr(self.tokenizer, "eos_token_id", None),
                "pad_token_id": getattr(self.tokenizer, "pad_token_id", None),
                "do_sample": do_sample,
                "temperature": current["temperature"] if do_sample else None,
                "top_p": current.get("top_p") if do_sample else None,
                "num_beams": current["num_beams"],
                "max_new_tokens": current["max_new_tokens"],
                "use_cache": True,
            }
            return translated, stops

        if self.backbone_name in {"qwen3_vl", "qwen3_5"}:
            # Match Qwen3_VL._build_generate_kwargs. Only the keys forwarded by
            # the native wrapper are allowed to reach model.generate().
            current = {
                "max_new_tokens": 128,
                "temperature": 0.0,
                "top_p": None,
                "num_beams": 1,
                **merged,
            }
            do_sample = bool(float(current["temperature"]) > 0)
            translated = {
                "eos_token_id": getattr(self.tokenizer, "eos_token_id", None),
                "pad_token_id": getattr(self.tokenizer, "pad_token_id", None),
                "max_new_tokens": current["max_new_tokens"],
                "use_cache": True,
                "do_sample": do_sample,
            }
            if do_sample:
                for key in ("temperature", "top_p", "top_k"):
                    if current.get(key) is not None:
                        translated[key] = current[key]
            if current.get("num_beams") is not None:
                translated["num_beams"] = current["num_beams"]
            return translated, stops

        translated.update(merged)
        temperature = translated.get("temperature")
        translated.setdefault("do_sample", bool(temperature and float(temperature) > 0))
        translated.setdefault("max_new_tokens", 128)
        if not translated["do_sample"]:
            if translated.get("temperature") in (0, 0.0):
                translated.pop("temperature", None)
            translated.pop("top_p", None)
            translated.pop("top_k", None)
        return translated, stops

    def _forward_probe_sample(
        self,
        *,
        sample: LlavaOneVisionSample | Qwen25VLSample | Qwen3VLSample,
        processed: Any,
        state: Any,
        request_generation: Mapping[str, Any],
        context: Any,
        task: str,
        split: str,
        doc_id: Any,
        request_mode: str,
        cache_hook: Any | None,
    ) -> str:
        """Execute one eval-time ``FORWARD(use_cache=False)`` activation probe.

        lmms-eval still owns task loading, request collation, documents, media,
        result persistence, and metrics. This explicit diagnostic mode replaces
        autoregressive generation with exactly one model forward and returns
        the argmax next token only as protocol transport. Its benchmark score
        is not a quality result.
        """

        native_inputs = getattr(state, "native_model_inputs", None)
        if not isinstance(native_inputs, dict):
            raise BackendIntegrationError(
                "forward_probe requires a ForwardState with native_model_inputs."
            )
        if native_inputs.get("labels") is not None:
            raise BackendIntegrationError("forward_probe does not accept labels.")
        native_inputs["use_cache"] = False
        native_inputs["return_dict"] = True
        native_inputs["output_attentions"] = False
        native_inputs["output_hidden_states"] = False

        validator = getattr(self.facade.runner, "native_validator", None)
        validate_state = getattr(validator, "validate_state", None)
        if callable(validate_state):
            violations = validate_state(self.runtime, state, processed)
            if violations:
                details = "; ".join(
                    f"{item.name}[{item.field}]: expected={item.expected!r}, "
                    f"actual={item.actual!r}"
                    for item in violations
                )
                raise BackendIntegrationError(
                    f"forward_probe native input validation failed: {details}"
                )

        try:
            import torch
        except ImportError as exc:  # pragma: no cover - runtime loader requires torch
            raise BackendIntegrationError("forward_probe requires PyTorch.") from exc
        with torch.inference_mode():
            output = self.facade.runner.forward_train(self.runtime, state)
        logits = _output_field(output, "logits")
        if not isinstance(logits, torch.Tensor) or logits.ndim != 3:
            raise BackendIntegrationError(
                "forward_probe expected rank-3 model logits [batch, sequence, vocab]."
            )
        if int(logits.shape[0]) != 1 or int(logits.shape[1]) < 1:
            raise BackendIntegrationError(
                "forward_probe requires one non-empty output sequence."
            )
        token_ids = logits[:, -1, :].argmax(dim=-1, keepdim=True)
        decoded = self.facade.processor_adapter.detokenize(
            self.runtime.processor,
            token_ids,
        )
        if len(decoded) != 1:
            raise BackendIntegrationError(
                "forward_probe did not decode exactly one next-token response."
            )
        answer = (
            decoded[0].strip()
            if self.backbone_name == "llava_onevision"
            else decoded[0]
        )
        probe_record = {
            "sample_id": sample.sample_id,
            "request_mode": "forward_probe",
            "execution_phase": "forward",
            "use_cache": False,
            "logits_shape": [int(value) for value in logits.shape],
            "next_token_id": int(token_ids[0, 0].item()),
            "benchmark_quality_claimed": False,
        }
        if self.compression_manifest_path:
            self._write_compression_manifest(last_request=probe_record)
        if self.diagnostic_trace_path:
            self._write_diagnostic_trace(
                {
                    **probe_record,
                    "transport_request_mode": request_mode,
                    "task": task,
                    "split": split,
                    "doc_id": doc_id,
                    "context": context,
                    "processor_metadata": _diagnostic_processor_metadata(
                        processed.metadata
                    ),
                    "request_generation_ignored": dict(request_generation),
                    "decoded": decoded[0],
                    "answer": answer,
                }
            )
        if cache_hook is not None:
            cache_hook.add_partial(
                "generate_until",
                (context, dict(request_generation)),
                answer,
            )
        return answer

    def _generate_sample(
        self,
        sample: LlavaOneVisionSample | Qwen25OmniSample | Qwen25VLSample | Qwen3VLSample,
        *,
        request_generation: Mapping[str, Any],
        context: Any,
        task: str,
        split: str,
        doc_id: Any,
        request_mode: str,
        cache_hook: Any | None,
    ) -> str:
        session = self.runtime.metadata.get("_compression_session")
        if session is not None:
            session.clear_diagnostic_records()
        processed = self.facade.processor_adapter.prepare_inputs(
            self.runtime.processor,
            sample,
            phase="eval",
            runtime_mode=str(
                self.runtime.metadata.get("runtime_mode", RuntimeMode.DEFAULT.value)
            ),
        )
        # Qwen2.5 boxes refer to the actual resized image, not the original.
        # Keep the proven Platform1 geometry guard for all RefCOCO families.
        from .grounding import document_input_size, is_refcoco_rec

        if self.backbone_name == "qwen25_vl" and is_refcoco_rec(task):
            expected_size = document_input_size(
                self._document(task, split, doc_id), self.min_pixels, self.max_pixels
            )
            grid = processed.model_inputs.get("image_grid_thw")
            patch_size = int(self.runtime.processor.image_processor.patch_size)
            if grid is None or len(grid) != 1:
                raise BackendIntegrationError("RefCOCO grounding requires exactly one image grid")
            actual_size = (int(grid[0][2]) * patch_size, int(grid[0][1]) * patch_size)
            if actual_size != expected_size:
                raise BackendIntegrationError(
                    f"Grounding coordinate geometry mismatch: {actual_size} != {expected_size}"
                )
        state = self.facade.state_builder.build(
            processed,
            raw_batch=sample,
            phase="eval",
        )
        if self.request_mode == "forward_probe":
            if self.backbone_name == "qwen25_omni":  # defensive constructor guard
                raise BackendIntegrationError(
                    "forward_probe is not implemented for qwen25_omni."
                )
            return self._forward_probe_sample(
                sample=sample,
                processed=processed,
                state=state,
                request_generation=request_generation,
                context=context,
                task=task,
                split=split,
                doc_id=doc_id,
                request_mode=request_mode,
                cache_hook=cache_hook,
            )
        generation_kwargs, stops = self._generation_kwargs(request_generation)
        from .request_performance import request_performance
        with request_performance(self.runtime, processed, sample.sample_id), sgl_target_generation(self.runtime, processed, generation_kwargs):
            result = self.facade.runner.generate(
                self.runtime,
                state,
                processed_batch=processed,
                **generation_kwargs,
            )
        if self.compression_manifest_path:
            self._write_compression_manifest(
                last_request={
                    "sample_id": sample.sample_id,
                    "compression_mode": result.metadata.get("compression_mode"),
                    "compression": result.metadata.get("compression"),
                    "compression_state": result.metadata.get("compression_state"),
                }
            )
        if result.text is None or len(result.text.text) != 1:
            raise BackendIntegrationError(
                "The platform runtime did not return exactly one text response for an lmms-eval request."
            )
        answer = _truncate_at_stop(result.text.text[0], stops)
        if self.backbone_name == "llava_onevision":
            answer = answer.strip()
        model_config = getattr(self.runtime.model, "config", None)
        attention_implementation = next(
            (
                getattr(model_config, name)
                for name in ("_attn_implementation", "attn_implementation")
                if getattr(model_config, name, None) is not None
            ),
            None,
        )
        if self.diagnostic_trace_path:
            self._write_diagnostic_trace(
                {
                    "sample_id": sample.sample_id,
                    "request_mode": request_mode,
                    "task": task,
                    "split": split,
                    "doc_id": doc_id,
                    "context": context,
                    "conversation": getattr(sample, "conversation", None),
                    "prompt": getattr(sample, "prompt", None),
                    "system_prompt": self.system_prompt,
                    "processor_metadata": _diagnostic_processor_metadata(
                        processed.metadata
                    ),
                    "processor_inputs": {
                        name: _trace_jsonable(
                            value,
                            include_values=name
                            in {
                                "input_ids",
                                "attention_mask",
                                "image_grid_thw",
                                "video_grid_thw",
                            },
                        )
                        for name, value in processed.model_inputs.items()
                    },
                    "runtime_dtype": str(self.runtime.dtype),
                    "runtime_environment": dict(self.runtime_environment or {}),
                    "model_dtype": str(getattr(self.runtime.model, "dtype", None)),
                    "attention_implementation": attention_implementation,
                    "use_audio_in_video": self.use_audio_in_video,
                    "generation_kwargs": generation_kwargs,
                    "generation_metadata": dict(result.metadata),
                    "compression_state": result.metadata.get("compression_state"),
                    "generated_token_ids": result.text.token_ids,
                    "decoded": result.text.text[0],
                    "stop_sequences": stops,
                    "answer": answer,
                }
            )
        if cache_hook is not None:
            cache_hook.add_partial(
                "generate_until",
                (context, dict(request_generation)),
                answer,
            )
        return answer

    @staticmethod
    def _request_iterator(requests: Sequence[Any], *, show_progress: bool) -> Any:
        if not show_progress:
            return requests
        try:
            from tqdm import tqdm
        except ImportError:
            return requests
        return tqdm(requests, desc="Model Responding")

    def generate_until(
        self,
        requests: Sequence[Any],
        *,
        cache_hook: Any | None = None,
        show_progress: bool = False,
    ) -> list[str]:
        """Execute lmms-eval's simple ``doc_to_visual`` request protocol."""

        request_args = [tuple(getattr(request, "args", ())) for request in requests]
        restore_order = None
        if self.backbone_name in {
            "llava_onevision",
            "qwen25_vl",
            "qwen3_vl",
            "qwen3_5",
        } and request_args:
            # All three native lmms-eval models group by generation kwargs and
            # sort longest contexts first. This also fixes which sample sees
            # each RNG state when checkpoint sampling is enabled.
            def _collate(args: tuple[Any, ...]) -> tuple[int, Any]:
                context = args[0]
                if self.backbone_name == "llava_onevision":
                    tokens = self.tokenizer.encode(
                        context,
                        add_special_tokens=False,
                    )
                else:
                    tokens = self.tokenizer.encode(context)
                return -len(tokens), context

            request_args, restore_order = _native_simple_request_order(
                request_args,
                sort_key=_collate,
            )

        responses: list[str] = []
        iterator = self._request_iterator(request_args, show_progress=show_progress)
        for request_index, args in enumerate(iterator):
            if len(args) != 6:
                raise BackendIntegrationError(
                    "lmms-eval generate_until requests must expose six args: "
                    "(context, generation_kwargs, doc_to_visual, doc_id, task, split)."
                )
            context, request_generation, doc_to_visual, doc_id, task, split = args
            if not isinstance(request_generation, Mapping):
                raise BackendIntegrationError("lmms-eval generation_kwargs must be a mapping.")
            if not callable(doc_to_visual):
                raise BackendIntegrationError("lmms-eval doc_to_visual must be callable.")

            document = self._document(str(task), str(split), doc_id)
            sample = self.build_sample(
                context=context,
                visual=doc_to_visual(document),
                sample_id=f"{task}:{split}:{doc_id}:{request_index}",
            )
            responses.append(
                self._generate_sample(
                    sample,
                    request_generation=request_generation,
                    context=context,
                    task=str(task),
                    split=str(split),
                    doc_id=doc_id,
                    request_mode="simple",
                    cache_hook=cache_hook,
                )
            )
        return restore_order(responses) if restore_order is not None else responses

    def generate_until_chat(
        self,
        requests: Sequence[Any],
        *,
        cache_hook: Any | None = None,
        show_progress: bool = False,
    ) -> list[str]:
        """Execute lmms-eval's chat ``doc_to_messages`` request protocol."""

        responses: list[str] = []
        for request_index, request in enumerate(
            self._request_iterator(requests, show_progress=show_progress)
        ):
            args = tuple(getattr(request, "args", ()))
            if len(args) != 6:
                raise BackendIntegrationError(
                    "lmms-eval chat requests must expose six args: "
                    "(context, doc_to_messages, generation_kwargs, doc_id, task, split)."
                )
            context, doc_to_messages, request_generation, doc_id, task, split = args
            if not callable(doc_to_messages):
                raise BackendIntegrationError("lmms-eval doc_to_messages must be callable.")
            if not isinstance(request_generation, Mapping):
                raise BackendIntegrationError("lmms-eval generation_kwargs must be a mapping.")

            document = self._document(str(task), str(split), doc_id)
            sample = self.build_chat_sample(
                raw_messages=doc_to_messages(document),
                sample_id=f"{task}:{split}:{doc_id}:{request_index}",
            )
            responses.append(
                self._generate_sample(
                    sample,
                    request_generation=request_generation,
                    context=context,
                    task=str(task),
                    split=str(split),
                    doc_id=doc_id,
                    request_mode="chat",
                    cache_hook=cache_hook,
                )
            )
        return responses

    def loglikelihood(self, requests: Sequence[Any]) -> list[tuple[float, bool]]:
        del requests
        raise BackendIntegrationError(
            f"{self.backbone_name} loglikelihood tasks are not supported by the active lmms-eval bridge; "
            "select a generate_until task."
        )

    def generate_until_multi_round(self, requests: Sequence[Any]) -> list[str]:
        del requests
        raise BackendIntegrationError(
            f"Multi-round lmms-eval tasks are not supported by the active {self.backbone_name} bridge."
        )
