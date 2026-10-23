"""Qwen2.5-Omni transformers-native backbone implementation."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from contextlib import ExitStack, contextmanager, nullcontext
from dataclasses import dataclass, field, replace
import json
from pathlib import Path
from time import perf_counter
from typing import Any, Literal

from mllm_comp.backbones.base import (
    BackboneFacade,
    BackboneRuntime,
    GenerationResult,
    ProcessedBatch,
    SpeechGenerationOutput,
    TextGenerationOutput,
    TraceRequest,
)
from mllm_comp.backbones.profiles import QWEN25_OMNI_PROFILE, BackboneCapabilities
from mllm_comp.backbones.teacher_forcing import (
    build_teacher_forced_batch,
    processor_special_token_ids,
)
from mllm_comp.core.actions import LegalizationPolicy, TokenAction
from mllm_comp.core.errors import (
    BackendIntegrationError,
    InvariantError,
    NotActivatedError,
)
from mllm_comp.core.state import ForwardState, TokenSpan
from mllm_comp.core.vocabulary import RuntimeMode
from mllm_comp.pipeline.operation_apply import Qwen25OmniOperationApplier

from .tracing import capture_qwen25_omni_traces
from .validation import Qwen25OmniNativeValidator


DEFAULT_AUDIO_SYSTEM_PROMPT = (
    "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, "
    "capable of perceiving auditory and visual inputs, as well as generating text and speech."
)
QWEN25_OMNI_OUTPUT_SAMPLE_RATE = 24000
Qwen25OmniInputModality = Literal["image", "video", "audio"]
QWEN25_OMNI_INPUT_MODALITIES: tuple[Qwen25OmniInputModality, ...] = (
    "image",
    "video",
    "audio",
)


@dataclass(frozen=True)
class Qwen25OmniSample:
    """One chat-formatted Omni input with an explicit sample identity."""

    sample_id: str | int
    conversation: list[dict[str, Any]]
    use_audio_in_video: bool = False


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


def _tolist(value: Any) -> Any:
    method = getattr(value, "tolist", None)
    return method() if callable(method) else value


def _rows(value: Any) -> list[list[int]]:
    raw = _tolist(value)
    if raw is None:
        return []
    if not isinstance(raw, (list, tuple)):
        return [[int(raw)]]
    if not raw:
        return []
    if not isinstance(raw[0], (list, tuple)):
        return [[int(item) for item in raw]]
    return [[int(item) for item in row] for row in raw]


def _row_sums(value: Any) -> list[int]:
    return [sum(row) for row in _rows(value)]


def _dtype_name(value: Any) -> str | None:
    dtype = getattr(value, "dtype", None)
    return None if dtype is None else str(dtype)


def _normalize_samples(raw_sample: Any) -> list[Qwen25OmniSample]:
    if isinstance(raw_sample, Qwen25OmniSample):
        return [raw_sample]
    if isinstance(raw_sample, Mapping):
        conversation = raw_sample.get("conversation")
        if conversation is None:
            raise ValueError("Qwen2.5-Omni sample mapping requires 'conversation'.")
        return [
            Qwen25OmniSample(
                sample_id=raw_sample.get("sample_id", "sample-0"),
                conversation=list(conversation),
                use_audio_in_video=bool(raw_sample.get("use_audio_in_video", False)),
            )
        ]
    if isinstance(raw_sample, Sequence) and not isinstance(raw_sample, (str, bytes)):
        values = list(raw_sample)
        if values and all(isinstance(item, Mapping) and "role" in item for item in values):
            return [Qwen25OmniSample(sample_id="sample-0", conversation=values)]
        samples: list[Qwen25OmniSample] = []
        for index, item in enumerate(values):
            if isinstance(item, Qwen25OmniSample):
                samples.append(item)
            elif isinstance(item, Mapping) and "conversation" in item:
                samples.append(
                    Qwen25OmniSample(
                        sample_id=item.get("sample_id", f"sample-{index}"),
                        conversation=list(item["conversation"]),
                        use_audio_in_video=bool(item.get("use_audio_in_video", False)),
                    )
                )
            else:
                raise TypeError(
                    "A Qwen2.5-Omni batch must contain Qwen25OmniSample objects "
                    "or mappings with a 'conversation' field."
                )
        if samples:
            return samples
    raise TypeError(
        "raw_sample must be a Qwen25OmniSample, a conversation, or a batch of samples."
    )


def _conversation_modality_counts(
    conversation: list[dict[str, Any]],
    *,
    use_audio_in_video: bool,
) -> dict[str, int]:
    counts = {"image": 0, "video": 0, "audio": 0}
    for message in conversation:
        content = message.get("content", [])
        if isinstance(content, str):
            continue
        for item in content:
            if not isinstance(item, Mapping):
                continue
            kind = item.get("type")
            if kind in {"image", "image_url"}:
                counts["image"] += 1
            elif kind == "video":
                counts["video"] += 1
                if use_audio_in_video:
                    counts["audio"] += 1
            elif kind in {"audio", "audio_url"}:
                counts["audio"] += 1
    return counts


def _uses_default_audio_prompt(conversation: list[dict[str, Any]]) -> bool:
    if not conversation or conversation[0].get("role") != "system":
        return False
    content = conversation[0].get("content", [])
    return bool(
        isinstance(content, list)
        and content
        and isinstance(content[0], Mapping)
        and content[0].get("text") == DEFAULT_AUDIO_SYSTEM_PROMPT
    )


def _semantic_role(role: Any) -> str:
    normalized = str(role or "").lower()
    if normalized == "system":
        return "system_prompt"
    if normalized == "assistant":
        return "response"
    return "instruction"


def _conversation_text_segments(
    conversation: list[dict[str, Any]],
) -> list[dict[str, str]]:
    segments: list[dict[str, str]] = []
    for message in conversation:
        label = _semantic_role(message.get("role"))
        content = message.get("content", [])
        if isinstance(content, str):
            if content:
                segments.append({"label": label, "text": content})
            continue
        for item in content:
            if not isinstance(item, Mapping):
                continue
            text = item.get("text")
            if isinstance(text, str) and text:
                segments.append({"label": label, "text": text})
    return segments


def _chat_template_texts(value: Any, *, batch_size: int) -> list[str] | None:
    if isinstance(value, str) and batch_size == 1:
        return [value]
    if (
        isinstance(value, Sequence)
        and not isinstance(value, (str, bytes))
        and len(value) == batch_size
        and all(isinstance(item, str) for item in value)
    ):
        return [str(item) for item in value]
    return None


def _semantic_char_spans(
    rendered_text: str,
    segments: Sequence[Mapping[str, str]],
) -> tuple[list[dict[str, Any]], list[str]]:
    """Locate message content inside the complete rendered chat template.

    The offsets intentionally exclude role/control prefixes. For example, Qwen's
    system content starts after ``<|im_start|>system\n`` at template offset 19.
    """

    spans: list[dict[str, Any]] = []
    warnings: list[str] = []
    cursor = 0
    for segment_index, segment in enumerate(segments):
        text = str(segment["text"])
        template_start = rendered_text.find(text, cursor)
        if template_start < 0:
            warnings.append(
                f"message text segment {segment_index} was not found in chat template"
            )
            continue
        template_end = template_start + len(text)
        spans.append(
            {
                "template_start": template_start,
                "template_end": template_end,
                "label": str(segment["label"]),
            }
        )
        cursor = template_end
    return spans, warnings


def _semantic_message_prefixes(
    processor: Any,
    conversation: list[dict[str, Any]],
    chat_template_kwargs: Mapping[str, Any],
) -> tuple[list[dict[str, str]], list[str]]:
    prefixes: list[dict[str, str]] = []
    warnings: list[str] = []
    prefix_kwargs = dict(chat_template_kwargs)
    prefix_kwargs["tokenize"] = False
    prefix_kwargs["add_generation_prompt"] = False
    for message_index, message in enumerate(conversation):
        try:
            rendered = processor.apply_chat_template(
                conversation[: message_index + 1],
                **prefix_kwargs,
            )
        except (TypeError, ValueError) as exc:
            warnings.append(
                f"message prefix {message_index} could not be rendered: {exc}"
            )
            break
        if not isinstance(rendered, str):
            warnings.append(
                f"message prefix {message_index} did not render as one string"
            )
            break
        prefixes.append(
            {
                "label": _semantic_role(message.get("role")),
                "text": rendered,
            }
        )
    return prefixes, warnings


def _special_token_text_by_id(tokenizer: Any) -> dict[int, str]:
    token_ids = list(getattr(tokenizer, "all_special_ids", None) or [])
    token_texts = list(getattr(tokenizer, "all_special_tokens", None) or [])
    result = {
        int(token_id): str(token_text)
        for token_id, token_text in zip(token_ids, token_texts)
        if token_id is not None
    }
    convert = getattr(tokenizer, "convert_ids_to_tokens", None)
    if callable(convert):
        for token_id in token_ids:
            if token_id is None or int(token_id) in result:
                continue
            try:
                result[int(token_id)] = str(convert(int(token_id)))
            except (KeyError, TypeError, ValueError):
                continue
    return result


def _token_id(processor: Any, token: Any) -> int | None:
    if token is None:
        return None
    try:
        value = processor.tokenizer.convert_tokens_to_ids(token)
    except (KeyError, TypeError, ValueError):
        return None
    return None if value is None else int(value)


def _partition_expected_counts(
    modality_counts: list[dict[str, int]],
    *,
    image_grid_thw: Any,
    video_grid_thw: Any,
    feature_attention_mask: Any,
    image_merge_size: int,
    video_merge_size: int,
) -> list[dict[str, int]]:
    image_lengths = [
        row[0] * row[1] * row[2] // (image_merge_size**2)
        for row in _rows(image_grid_thw)
    ]
    video_lengths = [
        row[0] * row[1] * row[2] // (video_merge_size**2)
        for row in _rows(video_grid_thw)
    ]
    audio_lengths = []
    for feature_length in _row_sums(feature_attention_mask):
        input_length = (feature_length - 1) // 2 + 1
        audio_lengths.append((input_length - 2) // 2 + 1)

    offsets = {"image": 0, "video": 0, "audio": 0}
    lengths = {"image": image_lengths, "video": video_lengths, "audio": audio_lengths}
    expected: list[dict[str, int]] = []
    for counts in modality_counts:
        sample_expected: dict[str, int] = {}
        for modality in QWEN25_OMNI_INPUT_MODALITIES:
            start = offsets[modality]
            end = start + counts[modality]
            sample_expected[modality] = sum(lengths[modality][start:end])
            offsets[modality] = end
        expected.append(sample_expected)

    for length_modality, values in lengths.items():
        if offsets[length_modality] != len(values):
            raise ValueError(
                f"Qwen processor returned {len(values)} {length_modality} item(s), but "
                f"the conversations describe {offsets[length_modality]}."
            )
    return expected


@dataclass(frozen=True)
class Qwen25OmniProcessorAdapter:
    """Load Qwen's processor and prepare all native input modalities."""

    profile: BackboneCapabilities = QWEN25_OMNI_PROFILE
    media_loader: Callable[..., Any] | None = field(default=None, compare=False, repr=False)

    def capabilities(self) -> BackboneCapabilities:
        return self.profile

    def load_processor(
        self,
        model_path_or_id: str,
        *,
        revision: str | None = None,
        **load_kwargs: Any,
    ) -> Any:
        from transformers import Qwen2_5OmniProcessor

        if revision is not None:
            load_kwargs["revision"] = revision
        # Transformers 4.57 changes Auto/Omni image processing to fast by
        # default, even for checkpoints saved with the slow processor. Preserve
        # the checkpoint-era contract unless the caller opts in explicitly.
        load_kwargs.setdefault("use_fast", False)
        return Qwen2_5OmniProcessor.from_pretrained(model_path_or_id, **load_kwargs)

    def prepare_inputs(
        self,
        processor: Any,
        raw_sample: Any,
        *,
        phase: str,
        runtime_mode: str = "default",
        **processor_kwargs: Any,
    ) -> ProcessedBatch:
        # 让后续逻辑永远只处理 list[Qwen25OmniSample]，不再反复判断输入形式。
        samples = _normalize_samples(raw_sample)
        audio_in_video_values = {sample.use_audio_in_video for sample in samples}
        explicit_audio_in_video = processor_kwargs.pop("use_audio_in_video", None)
        if explicit_audio_in_video is not None:
            audio_in_video_values.add(bool(explicit_audio_in_video))
        if len(audio_in_video_values) != 1:
            raise ValueError("One processor batch cannot mix use_audio_in_video values.")
        use_audio_in_video = audio_in_video_values.pop()

        conversations = [sample.conversation for sample in samples]
        conversation_input: Any = conversations[0] if len(conversations) == 1 else conversations
        chat_template_kwargs = dict(processor_kwargs.pop("chat_template_kwargs", {}))
        chat_template_kwargs.setdefault("tokenize", False)
        chat_template_kwargs.setdefault("add_generation_prompt", True)
        text = processor.apply_chat_template(conversation_input, **chat_template_kwargs)
        debug_semantics = RuntimeMode(runtime_mode) is RuntimeMode.DEBUG
        collect_semantics = True
        rendered_texts: list[str] | None = None
        semantic_char_spans_by_sample: list[list[dict[str, Any]]] = []
        semantic_message_prefixes_by_sample: list[list[dict[str, str]]] = []
        semantic_warnings_by_sample: list[list[str]] = []
        if collect_semantics:
            rendered_texts = _chat_template_texts(text, batch_size=len(samples))
            if rendered_texts is None:
                semantic_char_spans_by_sample = [[] for _ in samples]
                semantic_warnings_by_sample = [
                    ["chat template did not return one rendered string per sample"]
                    for _ in samples
                ]
            else:
                for rendered_text, sample in zip(rendered_texts, samples):
                    spans, warnings = _semantic_char_spans(
                        rendered_text,
                        _conversation_text_segments(sample.conversation),
                    )
                    prefixes, prefix_warnings = _semantic_message_prefixes(
                        processor,
                        sample.conversation,
                        chat_template_kwargs,
                    )
                    semantic_char_spans_by_sample.append(spans)
                    semantic_message_prefixes_by_sample.append(prefixes)
                    semantic_warnings_by_sample.append(
                        [*warnings, *prefix_warnings]
                    )

        media_loader = self.media_loader
        if media_loader is None:
            from qwen_omni_utils import process_mm_info

            media_loader = process_mm_info
        audios, images, videos = media_loader(
            conversation_input,
            use_audio_in_video=use_audio_in_video,
        )

        processor_kwargs.setdefault("return_tensors", "pt")
        processor_kwargs.setdefault("padding", True)
        processor_kwargs["use_audio_in_video"] = use_audio_in_video
        outputs = processor(
            text=text,
            audio=audios,
            images=images,
            videos=videos,
            **processor_kwargs,
        )
        model_inputs = dict(outputs)
        attention_mask = model_inputs.get("attention_mask")
        prompt_lengths = _row_sums(attention_mask)
        if not prompt_lengths:
            input_shape = _shape(model_inputs.get("input_ids"))
            prompt_lengths = [input_shape[1]] * input_shape[0] if input_shape else []

        modality_counts = [
            _conversation_modality_counts(
                sample.conversation,
                use_audio_in_video=use_audio_in_video,
            )
            for sample in samples
        ]
        image_merge_size = int(getattr(processor.image_processor, "merge_size", 1))
        video_merge_size = int(getattr(processor.video_processor, "merge_size", 1))
        expected_counts = _partition_expected_counts(
            modality_counts,
            image_grid_thw=model_inputs.get("image_grid_thw"),
            video_grid_thw=model_inputs.get("video_grid_thw"),
            feature_attention_mask=model_inputs.get("feature_attention_mask"),
            image_merge_size=image_merge_size,
            video_merge_size=video_merge_size,
        )
        input_shape = _shape(model_inputs.get("input_ids"))
        tokenizer = getattr(processor, "tokenizer", None)
        metadata = {
            "phase": phase,
            "provider": "transformers.Qwen2_5OmniProcessor",
            "processor_output_keys": sorted(model_inputs),
            "processor_output_contract": {
                name: {"shape": _shape(value), "dtype": _dtype_name(value)}
                for name, value in model_inputs.items()
            },
            "input_sequence_length": input_shape[1] if input_shape and len(input_shape) > 1 else None,
            "use_audio_in_video": use_audio_in_video,
            "modality_counts_by_sample": modality_counts,
            "expected_token_counts_by_sample": expected_counts,
            "token_ids": {
                "image": _token_id(processor, getattr(processor, "image_token", None)),
                "video": _token_id(processor, getattr(processor, "video_token", None)),
                "audio": _token_id(processor, getattr(processor, "audio_token", None)),
            },
            "pad_token_id": getattr(tokenizer, "pad_token_id", None),
            "bos_token_id": getattr(tokenizer, "bos_token_id", None),
            "eos_token_id": getattr(tokenizer, "eos_token_id", None),
            "user_token_id": _token_id(processor, "user"),
            "padding_side": getattr(tokenizer, "padding_side", "right"),
            "image_merge_size": image_merge_size,
            "video_merge_size": video_merge_size,
            "audio_input_sample_rate": int(processor.feature_extractor.sampling_rate),
            "speech_output_sample_rate": QWEN25_OMNI_OUTPUT_SAMPLE_RATE,
            "uses_default_audio_system_prompt": [
                _uses_default_audio_prompt(sample.conversation) for sample in samples
            ],
            "chat_template_text": text,
            "processor_classes": {
                "processor": type(processor).__qualname__,
                "image_processor": type(processor.image_processor).__qualname__,
                "video_processor": type(processor.video_processor).__qualname__,
            },
        }
        semantic_context = {
            "chat_template_text_by_sample": rendered_texts,
            "semantic_char_spans_by_sample": semantic_char_spans_by_sample,
            "semantic_message_prefixes_by_sample": (
                semantic_message_prefixes_by_sample
            ),
            "semantic_classification_warnings_by_sample": (
                semantic_warnings_by_sample
            ),
            "special_token_ids": sorted(
                {
                    int(value)
                    for value in (getattr(tokenizer, "all_special_ids", None) or [])
                    if value is not None
                }
            ),
            "special_token_text_by_id": _special_token_text_by_id(tokenizer),
            "global_special_token_ids": [
                int(value)
                for value in (getattr(tokenizer, "bos_token_id", None),)
                if value is not None
            ],
        }
        semantic_metadata = {**metadata, **semantic_context}
        sample_ids = [sample.sample_id for sample in samples]
        semantic_batch = ProcessedBatch(
            model_inputs=model_inputs,
            sample_ids=sample_ids,
            prompt_lengths=prompt_lengths,
            metadata=semantic_metadata,
        )
        semantic_state = ForwardState(
            input_ids=model_inputs.get("input_ids"),
            attention_mask=model_inputs.get("attention_mask"),
            labels=model_inputs.get("labels"),
            modality_metadata=semantic_metadata,
        )
        from .token_debug import prepare_qwen25_omni_semantic_labels

        prepare_qwen25_omni_semantic_labels(
            semantic_state,
            semantic_batch,
            processor=processor,
        )
        for name in (
            "semantic_token_labels_by_sample",
            "semantic_classification_source",
            "semantic_classification_warnings",
        ):
            metadata[name] = semantic_metadata[name]
        if debug_semantics:
            metadata.update(semantic_context)
        return ProcessedBatch(
            model_inputs=model_inputs,
            sample_ids=sample_ids,
            prompt_lengths=prompt_lengths,
            metadata=metadata,
        )

    def prepare_teacher_forced_inputs(
        self,
        processor: Any,
        raw_sample: Any,
        *,
        target_text: str,
        prompt_batch: ProcessedBatch | None = None,
        phase: str,
        runtime_mode: str = "default",
    ) -> ProcessedBatch:
        """Render one native Omni answer and label only its text tokens."""

        if not isinstance(target_text, str) or not target_text.strip():
            raise BackendIntegrationError(
                "Qwen2.5-Omni teacher forcing requires a non-empty textual target."
            )
        samples = _normalize_samples(raw_sample)
        if len(samples) != 1:
            raise BackendIntegrationError(
                "Qwen2.5-Omni analysis teacher forcing is single-sample."
            )
        sample = samples[0]
        prompt = prompt_batch or self.prepare_inputs(
            processor,
            sample,
            phase=phase,
            runtime_mode=runtime_mode,
        )
        full_sample = Qwen25OmniSample(
            sample_id=sample.sample_id,
            conversation=[
                *sample.conversation,
                {
                    "role": "assistant",
                    "content": [{"type": "text", "text": target_text}],
                },
            ],
            use_audio_in_video=sample.use_audio_in_video,
        )
        full = self.prepare_inputs(
            processor,
            full_sample,
            phase=phase,
            runtime_mode=runtime_mode,
            chat_template_kwargs={"add_generation_prompt": False},
        )
        return build_teacher_forced_batch(
            prompt_batch=prompt,
            full_batch=full,
            special_token_ids=processor_special_token_ids(processor),
        )

    def detokenize(
        self,
        processor: Any,
        token_ids: Any,
        **decode_kwargs: Any,
    ) -> list[str]:
        decode_kwargs.setdefault("skip_special_tokens", True)
        decode_kwargs.setdefault("clean_up_tokenization_spaces", False)
        return list(processor.batch_decode(token_ids, **decode_kwargs))


@dataclass(frozen=True)
class Qwen25OmniModelAdapter:
    """Load composite or saved Thinker models and expose semantic handles."""

    profile: BackboneCapabilities = QWEN25_OMNI_PROFILE

    def capabilities(self) -> BackboneCapabilities:
        return self.profile

    def load_model(
        self,
        model_path_or_id: str,
        *,
        train: bool,
        revision: str | None = None,
        **load_kwargs: Any,
    ) -> Any:
        from transformers import Qwen2_5OmniForConditionalGeneration

        enable_talker = bool(load_kwargs.pop("enable_talker", True))
        if revision is not None:
            load_kwargs["revision"] = revision
        if "torch_dtype" in load_kwargs and "dtype" not in load_kwargs:
            load_kwargs["dtype"] = load_kwargs.pop("torch_dtype")
        model_class = Qwen2_5OmniForConditionalGeneration
        config_path = Path(model_path_or_id) / "config.json"
        if config_path.is_file():
            config = json.loads(config_path.read_text(encoding="utf-8"))
            if config.get("model_type") == "qwen2_5_omni_thinker":
                if enable_talker:
                    raise BackendIntegrationError(
                        "This checkpoint contains only Thinker weights; set enable_talker=false."
                    )
                from transformers import Qwen2_5OmniThinkerForConditionalGeneration

                model_class = Qwen2_5OmniThinkerForConditionalGeneration
        model = model_class.from_pretrained(
            model_path_or_id,
            **load_kwargs,
        )
        if not enable_talker and model_class is Qwen2_5OmniForConditionalGeneration:
            model.disable_talker()
        model.train(mode=train)
        return model

    def structural_handles(self, model: Any) -> Mapping[str, Any]:
        handles: dict[str, Any] = {}

        def add(path: str, value: Any) -> None:
            if value is not None:
                handles[path] = value

        thinker = getattr(model, "thinker", model)
        thinker_llm = getattr(thinker, "model", None)
        visual = getattr(thinker, "visual", None)
        audio = getattr(thinker, "audio_tower", None)
        add("thinker", thinker)
        add("thinker.llm", thinker_llm)
        add("llm", thinker_llm)
        add("thinker.lm_head", getattr(thinker, "lm_head", None))
        add("llm.input_embeddings", getattr(thinker_llm, "embed_tokens", None))
        add("llm.final_norm", getattr(thinker_llm, "norm", None))
        add("lm_head", getattr(thinker, "lm_head", None))
        add("vision_encoder", visual)
        add("vision_projector", getattr(visual, "merger", None))
        add("audio_encoder", audio)
        add("audio_projector", getattr(audio, "proj", None))

        for index, layer in enumerate(getattr(thinker_llm, "layers", ())):
            add(f"llm.layer[{index}]", layer)
            attention = getattr(layer, "self_attn", None)
            ffn = getattr(layer, "mlp", None)
            add(f"llm.layer[{index}].attn", attention)
            add(
                f"llm.layer[{index}].attn.v_proj",
                getattr(attention, "v_proj", None),
            )
            add(
                f"llm.layer[{index}].attn.o_proj",
                getattr(attention, "o_proj", None),
            )
            add(f"llm.layer[{index}].ffn", ffn)
            add(
                f"llm.layer[{index}].ffn.down_proj",
                getattr(ffn, "down_proj", None),
            )

        talker = getattr(model, "talker", None)
        talker_model = getattr(talker, "model", None)
        add("talker", talker)
        add("talker.model", talker_model)
        add("talker.thinker_projector", getattr(talker, "thinker_to_talker_proj", None))
        add("talker.lm_head", getattr(talker, "codec_head", None))
        for index, layer in enumerate(getattr(talker_model, "layers", ())):
            add(f"talker.layer[{index}]", layer)

        token2wav = getattr(model, "token2wav", None)
        add("token2wav", token2wav)
        add("token2wav.dit", getattr(token2wav, "code2wav_dit_model", None))
        add("token2wav.vocoder", getattr(token2wav, "code2wav_bigvgan_model", None))
        return handles

    def external_code_info(self) -> dict[str, Any]:
        return {
            "code_origin": self.profile.code_origin,
            "source_ref": self.profile.source_ref,
            "loader_strategy": self.profile.loader_strategy,
            "minimum_transformers_version": "4.52.3",
            "verified_transformers_version": "4.57.3",
        }


def _contiguous_token_spans(
    input_ids: Any,
    token_ids: Mapping[str, int | None],
) -> list[TokenSpan]:
    rows = _rows(input_ids)
    spans: list[TokenSpan] = []
    media_units: dict[Qwen25OmniInputModality, str] = {
        "image": "image_patch",
        "video": "video_patch",
        "audio": "audio_frame",
    }
    for sample_index, row in enumerate(rows):
        for modality in QWEN25_OMNI_INPUT_MODALITIES:
            token_id = token_ids.get(modality)
            if token_id is None:
                continue
            start: int | None = None
            for index, value in enumerate([*row, None]):
                if value == token_id and start is None:
                    start = index
                elif value != token_id and start is not None:
                    spans.append(
                        TokenSpan(
                            sample_index=sample_index,
                            start=start,
                            end=index,
                            modality=modality,
                            role="observation",
                            source="qwen25_omni_processor",
                            media_unit=media_units[modality],
                            encoder_origin=f"{modality}_encoder",
                        )
                    )
                    start = None
    return sorted(spans, key=lambda span: (span.sample_index, span.start, span.end))


@dataclass(frozen=True)
class Qwen25OmniStateBuilder:
    """Normalize native processor names while retaining exact model kwargs."""

    profile: BackboneCapabilities = QWEN25_OMNI_PROFILE

    def build(
        self,
        processed_batch: ProcessedBatch,
        *,
        raw_batch: Any | None = None,
        phase: Literal["dry_run", "train", "eval"] = "train",
    ) -> ForwardState:
        del raw_batch
        outputs = processed_batch.model_inputs
        audio_mask = outputs.get("feature_attention_mask")
        audio_feature_lengths = None
        if audio_mask is not None:
            sums = getattr(audio_mask, "sum", None)
            audio_feature_lengths = sums(dim=1) if callable(sums) else _row_sums(audio_mask)
        metadata = dict(processed_batch.metadata)
        metadata.update(
            {
                "state_phase": phase,
                "native_to_normalized_fields": {
                    "pixel_values_videos": "video_pixel_values",
                    "input_features": "audio_values",
                    "feature_attention_mask": "audio_attention_mask",
                },
            }
        )
        return ForwardState(
            input_ids=outputs.get("input_ids"),
            inputs_embeds=outputs.get("inputs_embeds"),
            attention_mask=outputs.get("attention_mask"),
            position_ids=outputs.get("position_ids"),
            labels=outputs.get("labels"),
            token_type_mask=outputs.get("token_type_ids"),
            pixel_values=outputs.get("pixel_values"),
            image_grid_thw=outputs.get("image_grid_thw"),
            video_pixel_values=outputs.get("pixel_values_videos"),
            video_grid_thw=outputs.get("video_grid_thw"),
            audio_values=outputs.get("input_features"),
            audio_attention_mask=audio_mask,
            audio_feature_lengths=audio_feature_lengths,
            video_second_per_grid=outputs.get("video_second_per_grid"),
            native_model_inputs=dict(outputs),
            modality_token_spans=_contiguous_token_spans(
                outputs.get("input_ids"),
                metadata.get("token_ids", {}),
            ),
            modality_metadata=metadata,
            sample_ids=list(processed_batch.sample_ids),
        )


@dataclass(frozen=True)
class Qwen25OmniLegalizer:
    """Compression-state legalizer; native validity is handled separately."""

    profile: BackboneCapabilities = QWEN25_OMNI_PROFILE

    def apply(
        self,
        state: ForwardState,
        *,
        token_actions: list[TokenAction],
        policy: LegalizationPolicy,
        stage_name: str,
    ) -> ForwardState:
        if token_actions:
            from .audio_compression import legalize_qwen25_omni_audio_tokens

            return legalize_qwen25_omni_audio_tokens(
                state,
                token_actions=token_actions,
                policy=policy,
                stage_name=stage_name,
            )
        return replace(state)


def _move_value(value: Any, *, device: Any | None, dtype: Any | None) -> Any:
    to = getattr(value, "to", None)
    if not callable(to):
        return value
    if device is not None:
        value = value.to(device)
    is_floating_point = getattr(value, "is_floating_point", None)
    if dtype is not None and callable(is_floating_point) and is_floating_point():
        value = value.to(dtype=dtype)
    return value


def _slice_generated_ids(sequences: Any, input_width: int) -> Any:
    try:
        return sequences[:, input_width:]
    except (TypeError, IndexError):
        return [list(row)[input_width:] for row in sequences]


def _raise_violations(prefix: str, violations: list[Any]) -> None:
    if not violations:
        return
    details = "; ".join(
        f"{item.name}[{item.field}]: expected={item.expected!r}, actual={item.actual!r}"
        for item in violations
    )
    raise InvariantError(f"{prefix}: {details}")


def _cuda_measurement_context(device: Any | None) -> tuple[Any, Any] | None:
    try:
        import torch
    except ImportError:
        return None
    if device is None or not torch.cuda.is_available():
        return None
    resolved = torch.device(device)
    if resolved.type != "cuda":
        return None
    return torch, resolved


@contextmanager
def _capture_thinker_generation_timing(
    model: Any,
    *,
    device: Any | None = None,
) -> Any:
    """Measure generation, LLM prefill, approximate TTFT and CUDA memory."""

    report: dict[str, Any] = {
        "generation_seconds": None,
        "prefill_seconds": None,
        "ttft_seconds": None,
        "ttft_is_approximate": True,
        "llm_input_tokens": None,
        "llm_forward_calls": 0,
        "decode_forward_seconds": [],
        "decode_seconds_per_token": None,
        "cuda_device": None,
        "cuda_memory_allocated_before_bytes": None,
        "cuda_memory_reserved_before_bytes": None,
        "cuda_max_memory_allocated_bytes": None,
        "cuda_max_memory_reserved_bytes": None,
        "cuda_peak_memory_delta_bytes": None,
        "cuda_peak_reserved_delta_bytes": None,
    }
    cuda_context = _cuda_measurement_context(device)
    if cuda_context is not None:
        torch, cuda_device = cuda_context
        torch.cuda.synchronize(cuda_device)
        report["cuda_device"] = str(cuda_device)
        report["cuda_memory_allocated_before_bytes"] = int(
            torch.cuda.memory_allocated(cuda_device)
        )
        report["cuda_memory_reserved_before_bytes"] = int(
            torch.cuda.memory_reserved(cuda_device)
        )
        torch.cuda.reset_peak_memory_stats(cuda_device)
    total_started = perf_counter()
    llm = getattr(getattr(model, "thinker", model), "model", None)
    register_pre = getattr(llm, "register_forward_pre_hook", None)
    register_post = getattr(llm, "register_forward_hook", None)
    handles: list[Any] = []
    call_started: list[float | None] = [None]
    cuda_call_started: list[Any | None] = [None]
    cuda_call_events: list[tuple[Any, Any]] = []

    if callable(register_pre) and callable(register_post):
        def pre_hook(
            _module: Any,
            args: tuple[Any, ...],
            kwargs: dict[str, Any],
        ) -> None:
            now = perf_counter()
            if report["llm_forward_calls"] == 1 and report["ttft_seconds"] is None:
                report["ttft_seconds"] = now - total_started
            carrier = kwargs.get("inputs_embeds")
            if carrier is None:
                carrier = kwargs.get("input_ids")
            if carrier is None and args:
                carrier = args[0]
            shape = getattr(carrier, "shape", None)
            if report["llm_forward_calls"] == 0 and shape is not None and len(shape) >= 2:
                sequence_length = int(shape[1] if len(shape) == 2 else shape[-2])
                report["llm_input_tokens"] = sequence_length * int(shape[0])
            if cuda_context is None:
                call_started[0] = now
            else:
                torch, cuda_device = cuda_context
                event = torch.cuda.Event(enable_timing=True)
                event.record(torch.cuda.current_stream(cuda_device))
                cuda_call_started[0] = event

        def post_hook(
            _module: Any,
            _args: tuple[Any, ...],
            _kwargs: dict[str, Any],
            _output: Any,
        ) -> None:
            if cuda_context is None:
                started = call_started[0]
                if started is None:
                    return
                elapsed = perf_counter() - started
                if report["llm_forward_calls"] == 0:
                    report["prefill_seconds"] = elapsed
                else:
                    report["decode_forward_seconds"].append(elapsed)
            else:
                started_event = cuda_call_started[0]
                if started_event is None:
                    return
                torch, cuda_device = cuda_context
                end_event = torch.cuda.Event(enable_timing=True)
                end_event.record(torch.cuda.current_stream(cuda_device))
                cuda_call_events.append((started_event, end_event))
            report["llm_forward_calls"] += 1
            call_started[0] = None
            cuda_call_started[0] = None

        handles.append(register_pre(pre_hook, with_kwargs=True))
        handles.append(register_post(post_hook, with_kwargs=True))
    try:
        yield report
    finally:
        if cuda_context is not None:
            torch, cuda_device = cuda_context
            torch.cuda.synchronize(cuda_device)
        report["generation_seconds"] = perf_counter() - total_started
        if report["ttft_seconds"] is None:
            report["ttft_seconds"] = report["generation_seconds"]
        if cuda_call_events:
            elapsed_seconds = [
                float(start.elapsed_time(end)) / 1000.0
                for start, end in cuda_call_events
            ]
            report["prefill_seconds"] = elapsed_seconds[0]
            report["decode_forward_seconds"] = elapsed_seconds[1:]
        if report["decode_forward_seconds"]:
            report["decode_seconds_per_token"] = sum(
                report["decode_forward_seconds"]
            ) / len(report["decode_forward_seconds"])
        if cuda_context is not None:
            torch, cuda_device = cuda_context
            allocated_before = report["cuda_memory_allocated_before_bytes"]
            reserved_before = report["cuda_memory_reserved_before_bytes"]
            max_allocated = int(torch.cuda.max_memory_allocated(cuda_device))
            max_reserved = int(torch.cuda.max_memory_reserved(cuda_device))
            report["cuda_max_memory_allocated_bytes"] = max_allocated
            report["cuda_max_memory_reserved_bytes"] = max_reserved
            report["cuda_peak_memory_delta_bytes"] = max(
                0,
                max_allocated - int(allocated_before),
            )
            report["cuda_peak_reserved_delta_bytes"] = max(
                0,
                max_reserved - int(reserved_before),
            )
        for handle in handles:
            handle.remove()


@contextmanager
def _capture_training_forward_timing(
    *,
    device: Any,
) -> Any:
    """Measure one training forward only when profile mode is requested."""

    report: dict[str, Any] = {
        "forward_seconds": None,
        "cuda_device": None,
        "cuda_memory_allocated_before_bytes": None,
        "cuda_memory_reserved_before_bytes": None,
        "cuda_max_memory_allocated_bytes": None,
        "cuda_max_memory_reserved_bytes": None,
        "cuda_peak_memory_delta_bytes": None,
        "cuda_peak_reserved_delta_bytes": None,
    }
    cuda_context = _cuda_measurement_context(device)
    if cuda_context is not None:
        torch, cuda_device = cuda_context
        torch.cuda.synchronize(cuda_device)
        report["cuda_device"] = str(cuda_device)
        report["cuda_memory_allocated_before_bytes"] = int(
            torch.cuda.memory_allocated(cuda_device)
        )
        report["cuda_memory_reserved_before_bytes"] = int(
            torch.cuda.memory_reserved(cuda_device)
        )
        torch.cuda.reset_peak_memory_stats(cuda_device)
    started = perf_counter()
    try:
        yield report
    finally:
        if cuda_context is not None:
            torch, cuda_device = cuda_context
            torch.cuda.synchronize(cuda_device)
        report["forward_seconds"] = perf_counter() - started
        if cuda_context is not None:
            torch, cuda_device = cuda_context
            allocated_before = int(report["cuda_memory_allocated_before_bytes"])
            reserved_before = int(report["cuda_memory_reserved_before_bytes"])
            max_allocated = int(torch.cuda.max_memory_allocated(cuda_device))
            max_reserved = int(torch.cuda.max_memory_reserved(cuda_device))
            report["cuda_max_memory_allocated_bytes"] = max_allocated
            report["cuda_max_memory_reserved_bytes"] = max_reserved
            report["cuda_peak_memory_delta_bytes"] = max(
                0,
                max_allocated - allocated_before,
            )
            report["cuda_peak_reserved_delta_bytes"] = max(
                0,
                max_reserved - reserved_before,
            )


@dataclass(frozen=True)
class Qwen25OmniRunner:
    """Run Thinker text generation and optional Talker/Token2Wav speech."""

    processor_adapter: Qwen25OmniProcessorAdapter = field(
        default_factory=Qwen25OmniProcessorAdapter
    )
    native_validator: Qwen25OmniNativeValidator = field(
        default_factory=Qwen25OmniNativeValidator
    )
    profile: BackboneCapabilities = QWEN25_OMNI_PROFILE

    def forward_train(self, runtime: BackboneRuntime, state: ForwardState) -> Any:
        runtime_mode = RuntimeMode(
            runtime.metadata.get("runtime_mode", RuntimeMode.DEFAULT.value)
        )
        original_state = state
        prepared_compression: tuple[_PreparedCompressionActivation, ...] = ()
        input_shape = _shape(original_state.input_ids)
        batch_size = input_shape[0] if input_shape else 1
        sample_ids = list(original_state.sample_ids or range(batch_size))
        processed_batch = ProcessedBatch(
            model_inputs=dict(original_state.native_model_inputs),
            sample_ids=sample_ids,
            prompt_lengths=[],
            metadata=dict(original_state.modality_metadata),
        )
        debug_collector = runtime.metadata.get("_token_debug_collector")
        if runtime_mode is RuntimeMode.DEBUG and debug_collector is None:
            raise InvariantError(
                "runtime_mode=debug requires an installed Qwen token debug collector."
            )
        if runtime_mode is RuntimeMode.DEBUG:
            debug_collector.prepare_sequence_labels(original_state, processed_batch)
        state, prepared_compression = _prepare_runtime_compression(runtime, state)
        _raise_violations(
            "Qwen2.5-Omni native training input validation failed",
            self.native_validator.validate_state(runtime, state, processed_batch),
        )
        model_inputs = {
            name: _move_value(value, device=runtime.device, dtype=runtime.dtype)
            for name, value in state.native_model_inputs.items()
        }
        target = getattr(runtime.model, "thinker", runtime.model)
        with ExitStack() as stack:
            timing = stack.enter_context(
                _capture_training_forward_timing(device=runtime.device)
                if runtime_mode is RuntimeMode.PROFILE
                else nullcontext(None)
            )
            if runtime_mode is RuntimeMode.DEBUG:
                stack.enter_context(
                    debug_collector.request(
                        original_state=original_state,
                        active_state=state,
                        processed_batch=processed_batch,
                    )
                )
            for prepared in prepared_compression:
                stack.enter_context(prepared.activation.request(prepared.plan))
            output = target(**model_inputs)
        if runtime_mode is RuntimeMode.DEBUG:
            _apply_debug_compression_reports(debug_collector, prepared_compression)
        compression_report = _runtime_compression_report(runtime, prepared_compression)
        runtime.metadata["last_training_forward"] = {
            "runtime_mode": runtime_mode.value,
            "compression_mode": (
                "compressed" if compression_report is not None else "none"
            ),
            "timing": timing,
            "token_debug": (
                debug_collector.last_report
                if runtime_mode is RuntimeMode.DEBUG
                else None
            ),
            "compression": compression_report,
            "compression_state": state.modality_metadata.get("audio_compression"),
            "effective_token_layout": _effective_token_layout_snapshot(
                state,
                prepared_compression=prepared_compression,
            ),
        }
        return output

    def generate(
        self,
        runtime: BackboneRuntime,
        state: ForwardState,
        *,
        processed_batch: ProcessedBatch,
        trace_request: TraceRequest | None = None,
        **gen_kwargs: Any,
    ) -> GenerationResult:
        runtime_mode = RuntimeMode(
            runtime.metadata.get("runtime_mode", RuntimeMode.DEFAULT.value)
        )
        original_state = state
        compression_enabled = _runtime_compression_enabled(runtime)
        prepared_compression: tuple[_PreparedCompressionActivation, ...] = ()
        debug_collector = runtime.metadata.get("_token_debug_collector")
        if runtime_mode is RuntimeMode.DEBUG and debug_collector is None:
            raise InvariantError(
                "runtime_mode=debug requires an installed Qwen token debug collector."
            )
        if runtime_mode is RuntimeMode.DEBUG:
            debug_collector.prepare_sequence_labels(original_state, processed_batch)
        state, prepared_compression = _prepare_runtime_compression(runtime, state)
        _raise_violations(
            "Qwen2.5-Omni native input validation failed",
            self.native_validator.validate_state(runtime, state, processed_batch),
        )
        if gen_kwargs.pop("stream", False) or gen_kwargs.get("streamer") is not None:
            raise NotActivatedError(
                "The validated Transformers Qwen2.5-Omni implementation exposes only "
                "synchronous composite generation; no public simultaneous text/speech "
                "streaming contract exists."
            )

        return_audio = bool(gen_kwargs.pop("return_audio", False))
        if compression_enabled and return_audio:
            raise NotActivatedError(
                "Qwen compression currently supports the Thinker text path only; "
                "Talker/Token2Wav validation is out of scope."
            )
        speaker = str(gen_kwargs.pop("speaker", "Chelsie"))
        model = runtime.model
        thinker_only = (
            getattr(getattr(model, "config", None), "model_type", None)
            == "qwen2_5_omni_thinker"
        )
        if return_audio:
            if not bool(getattr(model, "has_talker", False)):
                raise InvariantError("return_audio=True requires an enabled Talker and Token2Wav.")
            if len(processed_batch.sample_ids) != 1:
                raise InvariantError(
                    "The validated Qwen2.5-Omni speech path supports only batch size 1."
                )
            if not all(processed_batch.metadata.get("uses_default_audio_system_prompt", [])):
                raise InvariantError(
                    "Qwen2.5-Omni speech output requires the checkpoint's default audio system prompt."
                )
            speaker_map = getattr(model, "speaker_map", {})
            if speaker_map and speaker not in speaker_map:
                raise InvariantError(
                    f"Unknown speaker {speaker!r}; available speakers: {sorted(speaker_map)}."
                )

        expected_audio_in_video = bool(processed_batch.metadata.get("use_audio_in_video", False))
        requested_audio_in_video = gen_kwargs.pop("use_audio_in_video", expected_audio_in_video)
        if bool(requested_audio_in_video) != expected_audio_in_video:
            raise InvariantError(
                "use_audio_in_video must match process_mm_info and processor preprocessing."
            )

        request = trace_request or TraceRequest()
        generic_trace_fields = (
            "text_logits",
            "text_hidden_states",
            "text_attentions",
            "text_kv_cache",
        )
        unsupported_traces = [
            name for name in generic_trace_fields if getattr(request, name)
        ]
        if unsupported_traces:
            raise InvariantError(
                "Qwen2.5-Omni uses explicit Thinker trace fields; replace "
                f"{unsupported_traces} with the corresponding thinker_* fields."
            )
        if request.thinker_hidden_states:
            gen_kwargs.setdefault("thinker_output_hidden_states", True)
        if request.thinker_attentions:
            gen_kwargs.setdefault("thinker_output_attentions", True)
        if request.talker_hidden_states:
            gen_kwargs.setdefault("talker_output_hidden_states", True)
        if request.talker_attentions:
            gen_kwargs.setdefault("talker_output_attentions", True)
        if not return_audio and request.enabled():
            gen_kwargs.setdefault("thinker_return_dict_in_generate", True)

        if prepared_compression:
            # Dynamic compression plans the audio prefill once. Training
            # checkpoints can persist use_cache=False in Thinker's text config.
            generation_config = gen_kwargs.get(
                "thinker_generation_config", gen_kwargs.get("generation_config")
            )
            use_cache = gen_kwargs.get(
                "thinker_use_cache",
                gen_kwargs.get("use_cache", getattr(generation_config, "use_cache", None)),
            )
            if use_cache is False:
                raise InvariantError(
                    "Qwen2.5-Omni dynamic compression generation requires "
                    "thinker_use_cache=True so audio is processed only during prefill. "
                    "Remove the explicit use_cache=False setting or enable it."
                )
            # Pass the effective choice explicitly: native generation may
            # otherwise fall back from a supplied GenerationConfig to the model.
            gen_kwargs["thinker_use_cache"] = True if use_cache is None else use_cache

        model_inputs = {
            name: _move_value(value, device=runtime.device, dtype=runtime.dtype)
            for name, value in state.native_model_inputs.items()
        }
        input_shape = _shape(model_inputs.get("input_ids"))
        if input_shape is None or len(input_shape) != 2:
            raise InvariantError("Qwen2.5-Omni generation requires rank-2 input_ids.")
        input_width = input_shape[1]

        capture_request = request
        if return_audio and not request.talker_codec_token_ids:
            capture_request = replace(request, talker_codec_token_ids=True)

        with ExitStack() as stack:
            timing = stack.enter_context(
                _capture_thinker_generation_timing(model, device=runtime.device)
                if runtime_mode is RuntimeMode.PROFILE
                else nullcontext(None)
            )
            if runtime_mode is RuntimeMode.DEBUG:
                stack.enter_context(
                    debug_collector.request(
                        original_state=original_state,
                        active_state=state,
                        processed_batch=processed_batch,
                    )
                )
            for prepared in prepared_compression:
                stack.enter_context(prepared.activation.request(prepared.plan))
            captured_traces = stack.enter_context(
                capture_qwen25_omni_traces(model, capture_request)
            )
            if thinker_only:
                # The saved submodel accepts native generation arguments;
                # retain the composite entry's Thinker-over-shared precedence.
                thinker_kwargs = {"max_new_tokens": 1024}
                thinker_kwargs.update({
                    key.removeprefix("thinker_"): value
                    for key, value in gen_kwargs.items() if key.startswith("thinker_")
                })
                for key, value in gen_kwargs.items():
                    if not key.startswith(("thinker_", "talker_", "token2wav_")):
                        thinker_kwargs.setdefault(key, value)
                raw_output = model.generate(
                    **model_inputs, use_audio_in_video=expected_audio_in_video, **thinker_kwargs
                )
            else:
                raw_output = model.generate(
                    **model_inputs,
                    return_audio=return_audio,
                    speaker=speaker,
                    use_audio_in_video=expected_audio_in_video,
                    **gen_kwargs,
                )
        if runtime_mode is RuntimeMode.DEBUG:
            _apply_debug_compression_reports(debug_collector, prepared_compression)
        compression_report = _runtime_compression_report(runtime, prepared_compression)

        waveform = None
        sequence_output = raw_output
        if return_audio:
            sequence_output, waveform = raw_output
        sequences = getattr(sequence_output, "sequences", sequence_output)
        # 切除 prompt，只将 response token 送入文本解码。
        generated_ids = _slice_generated_ids(sequences, input_width)
        decoded = self.processor_adapter.detokenize(runtime.processor, generated_ids)
        if runtime_mode is RuntimeMode.DEBUG:
            debug_collector.apply_generated_response(
                generated_ids,
                semantic_metadata=state.modality_metadata,
            )

        speech = None
        if return_audio:
            codec_ids = (
                captured_traces.talker_codec_token_ids
                if captured_traces is not None
                else None
            )
            speech = SpeechGenerationOutput(
                waveform=waveform,
                sample_rate=int(
                    runtime.metadata.get(
                        "speech_sample_rate",
                        processed_batch.metadata.get(
                            "speech_output_sample_rate",
                            QWEN25_OMNI_OUTPUT_SAMPLE_RATE,
                        ),
                    )
                ),
                speaker=speaker,
                codec_token_ids=codec_ids,
            )
        if (
            captured_traces is not None
            and not request.talker_codec_token_ids
        ):
            captured_traces.talker_codec_token_ids = None
        result = GenerationResult(
            text=TextGenerationOutput(token_ids=generated_ids, text=decoded),
            speech=speech,
            traces=captured_traces if request.enabled() else None,
            sample_ids=list(processed_batch.sample_ids),
            metadata={
                "provider": (
                    "transformers.Qwen2_5OmniThinkerForConditionalGeneration.generate"
                    if thinker_only
                    else "transformers.Qwen2_5OmniForConditionalGeneration.generate"
                ),
                "runtime_mode": runtime_mode.value,
                "compression_mode": (
                    "compressed" if compression_report is not None else "none"
                ),
                "return_audio": return_audio,
                "speaker": speaker if return_audio else None,
                "use_audio_in_video": expected_audio_in_video,
                "input_sequence_length": input_width,
                "streaming_supported": False,
                "speech_batch_constraint": 1,
                "timing": timing,
                "token_debug": (
                    debug_collector.last_report
                    if runtime_mode is RuntimeMode.DEBUG
                    else None
                ),
                "compression": compression_report,
                "compression_state": state.modality_metadata.get(
                    "audio_compression"
                ),
            },
        )
        _raise_violations(
            "Qwen2.5-Omni native output validation failed",
            self.native_validator.validate_generation(runtime, state, processed_batch, result),
        )
        return result


@dataclass(frozen=True)
class _PreparedCompressionActivation:
    """One v2 dynamic activation prepared for the current model request."""

    name: str
    binding: str
    activation: Any
    plan: Any


def _active_runtime_compression_records(runtime: BackboneRuntime) -> tuple[Any, ...]:
    """Read installed activations only from the generic v2 session."""

    session = runtime.metadata.get("_compression_session")
    if session is None:
        return ()
    records = getattr(session, "active_records", ())
    try:
        return tuple(records)
    except TypeError as exc:
        raise InvariantError(
            "The installed compression session exposed invalid active records."
        ) from exc


def _runtime_compression_enabled(runtime: BackboneRuntime) -> bool:
    if _active_runtime_compression_records(runtime):
        return True
    configured = runtime.metadata.get("compression")
    return bool(
        isinstance(configured, Mapping) and configured.get("enabled") is True
    )


def _prepare_runtime_compression(
    runtime: BackboneRuntime,
    state: ForwardState,
) -> tuple[ForwardState, tuple[_PreparedCompressionActivation, ...]]:
    """Prepare all dynamic Qwen activations in dependency-resolved order."""

    prepared: list[_PreparedCompressionActivation] = []
    for record in _active_runtime_compression_records(runtime):
        activation = getattr(record, "activation", None)
        prepare_state = getattr(activation, "prepare_state", None)
        request = getattr(activation, "request", None)
        if not callable(prepare_state) and not callable(request):
            continue
        if not callable(prepare_state) or not callable(request):
            raise InvariantError(
                f"Compression activation {getattr(record, 'name', '<unknown>')!r} "
                "must implement prepare_state() and request() together."
            )
        # Data-dependent projector planning runs the real audio tower. Apply
        # already prepared upstream plans during that existing preflight.
        with ExitStack() as planning_requests:
            if (getattr(getattr(activation, "execution", None), "executor_family", None) != "llm_token"
                    and getattr(getattr(activation, "plugin", None), "requires_projector_hidden_state_plan", False)):
                for upstream in prepared:
                    planning_requests.enter_context(upstream.activation.request(upstream.plan))
            state, plan = prepare_state(state)
        if plan is None:
            raise InvariantError(
                f"Compression activation {getattr(record, 'name', '<unknown>')!r} "
                "did not produce a plan."
            )
        prepared.append(
            _PreparedCompressionActivation(
                name=str(getattr(record, "name", "")),
                binding=str(getattr(record, "binding", "")),
                activation=activation,
                plan=plan,
            )
        )
    return state, tuple(prepared)


def _apply_debug_compression_reports(
    debug_collector: Any,
    prepared: Sequence[_PreparedCompressionActivation],
) -> None:
    for item in prepared:
        debug_collector.apply_compression_report(item.activation.last_report)


def _runtime_compression_report(
    runtime: BackboneRuntime,
    prepared: Sequence[_PreparedCompressionActivation] | None,
) -> dict[str, Any] | None:
    """Return stage evidence while preserving the single-stage report shape."""

    session = runtime.metadata.get("_compression_session")
    runtime_metadata = getattr(session, "runtime_metadata", None)
    if callable(runtime_metadata):
        runtime.metadata["compression"] = dict(runtime_metadata())

    prepared_items = tuple(prepared or ())
    if len(prepared_items) == 1:
        report = prepared_items[0].activation.last_report
        return dict(report) if isinstance(report, Mapping) else report
    if len(prepared_items) > 1:
        stages = []
        for item in prepared_items:
            report = item.activation.last_report
            stages.append(
                {
                    "name": item.name,
                    "binding": item.binding,
                    "report": dict(report) if isinstance(report, Mapping) else report,
                }
            )
        return {
            "enabled": True,
            "multi_stage": True,
            "stage_count": len(stages),
            "stages": stages,
        }

    configured = runtime.metadata.get("compression")
    if isinstance(configured, Mapping) and configured.get("enabled") is True:
        return dict(configured)
    return None


def _effective_token_layout_snapshot(
    state: ForwardState,
    *,
    prepared_compression: Sequence[_PreparedCompressionActivation] = (),
) -> dict[str, Any] | None:
    """Retain only the lightweight semantic layout used by this forward."""

    input_shape = _shape(state.input_ids)
    if input_shape is None or len(input_shape) != 2:
        return None
    batch_size, sequence_length = input_shape
    raw_rows = state.modality_metadata.get("semantic_token_labels_by_sample")
    semantic_rows: list[list[str]] | None = None
    if (
        isinstance(raw_rows, list)
        and len(raw_rows) == batch_size
        and all(
            isinstance(row, list) and len(row) == sequence_length
            for row in raw_rows
        )
    ):
        semantic_rows = [[str(label) for label in row] for row in raw_rows]
    sample_ids = list(state.sample_ids or range(batch_size))
    if len(sample_ids) != batch_size:
        return None
    return {
        "schema_version": 1,
        "sample_ids": sample_ids,
        "sequence_length": sequence_length,
        "semantic_labels_by_sample": semantic_rows,
        "semantic_classification_source": state.modality_metadata.get(
            "semantic_classification_source"
        ),
        "semantic_labels_by_layer_entry": _prepared_layer_token_layouts(
            prepared_compression
        ),
    }


def _prepared_layer_token_layouts(
    prepared: Sequence[_PreparedCompressionActivation],
) -> dict[str, list[list[str]]]:
    result: dict[str, list[list[str]]] = {}
    for item in prepared:
        raw_layouts = getattr(item.activation, "last_token_layouts", None)
        if not isinstance(raw_layouts, Mapping):
            continue
        for raw_layer, raw_rows in raw_layouts.items():
            layer = str(int(raw_layer))
            if not isinstance(raw_rows, (list, tuple)) or not all(
                isinstance(row, (list, tuple)) for row in raw_rows
            ):
                raise InvariantError(
                    "Compression activation exposed invalid semantic token layouts."
                )
            rows = [[str(label) for label in row] for row in raw_rows]
            existing = result.get(layer)
            if existing is not None and existing != rows:
                raise InvariantError(
                    f"Compression activations disagree on layer {layer} token layout."
                )
            result[layer] = rows
    return result


def create_qwen25_omni_backbone() -> BackboneFacade:
    """Compose the Qwen2.5-Omni facade without retaining runtime objects."""

    profile = QWEN25_OMNI_PROFILE
    processor_adapter = Qwen25OmniProcessorAdapter(profile)
    native_validator = Qwen25OmniNativeValidator()
    return BackboneFacade(
        processor_adapter=processor_adapter,
        model_adapter=Qwen25OmniModelAdapter(profile),
        state_builder=Qwen25OmniStateBuilder(profile),
        legalizer=Qwen25OmniLegalizer(profile),
        runner=Qwen25OmniRunner(processor_adapter, native_validator, profile),
        operation_applier=Qwen25OmniOperationApplier(),
        native_validator=native_validator,
    )
