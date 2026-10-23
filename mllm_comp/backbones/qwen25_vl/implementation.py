"""Transformers-native Qwen2.5-VL baseline backbone implementation."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field, replace
from typing import Any, Iterator, Literal

from mllm_comp.backbones.base import (
    BackboneFacade,
    BackboneRuntime,
    GenerationResult,
    ProcessedBatch,
    TextGenerationOutput,
    TraceArtifacts,
    TraceRequest,
)
from mllm_comp.backbones.profiles import QWEN25_VL_PROFILE, BackboneCapabilities
from mllm_comp.backbones.teacher_forcing import (
    build_teacher_forced_batch,
    processor_special_token_ids,
)
from mllm_comp.backbones._video_conditioning import user_text_from_conversation
from mllm_comp.core.actions import LegalizationPolicy, TokenAction
from mllm_comp.core.errors import (
    BackendIntegrationError,
    InvariantError,
    NotActivatedError,
)
from mllm_comp.core.state import ForwardState, TokenSpan
from mllm_comp.compressors.image.sgl import sgl_target_generation

from .video_compression import qwen25_vl_video_request_context
from .validation import Qwen25VLNativeValidator


DEFAULT_SYSTEM_PROMPT = "You are a helpful assistant."
QWEN25_VL_IMAGE_PATCH_SIZE = 14
Qwen25VLInputModality = Literal["image", "video"]
QWEN25_VL_INPUT_MODALITIES: tuple[Qwen25VLInputModality, ...] = (
    "image",
    "video",
)


@dataclass(frozen=True)
class Qwen25VLSample:
    """One Qwen2.5-VL conversation with stable sample identity."""

    sample_id: str | int
    conversation: list[dict[str, Any]]


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


def _normalize_samples(raw_sample: Any) -> list[Qwen25VLSample]:
    if isinstance(raw_sample, Qwen25VLSample):
        return [raw_sample]
    if isinstance(raw_sample, Mapping):
        conversation = raw_sample.get("conversation")
        if conversation is None:
            raise ValueError("Qwen2.5-VL sample mapping requires 'conversation'.")
        return [
            Qwen25VLSample(
                sample_id=raw_sample.get("sample_id", "sample-0"),
                conversation=list(conversation),
            )
        ]
    if isinstance(raw_sample, Sequence) and not isinstance(raw_sample, (str, bytes)):
        values = list(raw_sample)
        if values and all(
            isinstance(item, Mapping) and "role" in item for item in values
        ):
            return [Qwen25VLSample(sample_id="sample-0", conversation=values)]
        samples: list[Qwen25VLSample] = []
        for index, item in enumerate(values):
            if isinstance(item, Qwen25VLSample):
                samples.append(item)
            elif isinstance(item, Mapping) and "conversation" in item:
                samples.append(
                    Qwen25VLSample(
                        sample_id=item.get("sample_id", f"sample-{index}"),
                        conversation=list(item["conversation"]),
                    )
                )
            else:
                raise TypeError(
                    "A Qwen2.5-VL batch must contain Qwen25VLSample objects or "
                    "mappings with a 'conversation' field."
                )
        if samples:
            return samples
    raise TypeError(
        "raw_sample must be a Qwen25VLSample, a conversation, or a batch of " "samples."
    )


def _conversation_modality_counts(
    conversation: list[dict[str, Any]],
) -> dict[str, int]:
    counts = {"image": 0, "video": 0}
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
            elif kind in {"audio", "audio_url"}:
                raise BackendIntegrationError(
                    "Qwen2.5-VL does not accept audio inputs."
                )
    return counts


def _token_id(processor: Any, attribute: str, fallback_token: str) -> int | None:
    token = getattr(processor, attribute, None) or fallback_token
    value = processor.tokenizer.convert_tokens_to_ids(token)
    return None if value is None else int(value)


def _qwen_completion_end_token_ids(processor: Any) -> tuple[int, ...]:
    """Return message terminators so formatting after assistant content is ignored."""

    token_id = _token_id(processor, "im_end_token", "<|im_end|>")
    return () if token_id is None else (token_id,)


def _spatial_merge_size(processor: Any) -> int:
    image_processor = getattr(processor, "image_processor", None)
    value = getattr(image_processor, "merge_size", None)
    if value is None:
        value = getattr(image_processor, "spatial_merge_size", None)
    if value is None:
        value = getattr(processor, "spatial_merge_size", 1)
    merge_size = int(value)
    if merge_size <= 0:
        raise ValueError("Qwen2.5-VL spatial_merge_size must be positive.")
    return merge_size


def _partition_expected_counts(
    modality_counts: list[dict[str, int]],
    *,
    image_grid_thw: Any,
    video_grid_thw: Any,
    spatial_merge_size: int,
) -> list[dict[str, int]]:
    def token_lengths(grid: Any, modality: str) -> list[int]:
        divisor = spatial_merge_size**2
        values: list[int] = []
        for row in _rows(grid):
            if len(row) != 3 or any(value <= 0 for value in row):
                raise ValueError(
                    f"Qwen2.5-VL {modality}_grid_thw rows must contain three "
                    "positive integers."
                )
            patch_count = row[0] * row[1] * row[2]
            if patch_count % divisor:
                raise ValueError(
                    f"Qwen2.5-VL {modality}_grid_thw patch count {patch_count} "
                    f"is not divisible by spatial merge area {divisor}."
                )
            values.append(patch_count // divisor)
        return values

    lengths = {
        "image": token_lengths(image_grid_thw, "image"),
        "video": token_lengths(video_grid_thw, "video"),
    }
    offsets = {"image": 0, "video": 0}
    expected: list[dict[str, int]] = []
    for counts in modality_counts:
        sample_expected: dict[str, int] = {}
        for modality in QWEN25_VL_INPUT_MODALITIES:
            start = offsets[modality]
            end = start + counts[modality]
            sample_expected[modality] = sum(lengths[modality][start:end])
            offsets[modality] = end
        expected.append(sample_expected)
    for modality, values in lengths.items():
        if offsets[modality] != len(values):
            raise ValueError(
                f"Qwen2.5-VL processor returned {len(values)} {modality} item(s), "
                f"but the conversations describe {offsets[modality]}."
            )
    return expected


@dataclass(frozen=True)
class Qwen25VLProcessorAdapter:
    """Load Qwen2.5-VL processor and retain all native visual/M-RoPE fields."""

    profile: BackboneCapabilities = QWEN25_VL_PROFILE
    media_loader: Callable[..., Any] | None = field(
        default=None,
        compare=False,
        repr=False,
    )

    def capabilities(self) -> BackboneCapabilities:
        return self.profile

    def load_processor(
        self,
        model_path_or_id: str,
        *,
        revision: str | None = None,
        **load_kwargs: Any,
    ) -> Any:
        try:
            from transformers import AutoProcessor, Qwen2_5_VLProcessor

            _ = Qwen2_5_VLProcessor
        except ImportError as exc:
            raise BackendIntegrationError(
                "Qwen2.5-VL requires transformers>=4.49,<5."
            ) from exc
        if revision is not None:
            load_kwargs["revision"] = revision
        return AutoProcessor.from_pretrained(model_path_or_id, **load_kwargs)

    def prepare_inputs(
        self,
        processor: Any,
        raw_sample: Any,
        *,
        phase: str,
        runtime_mode: str = "default",
        **processor_kwargs: Any,
    ) -> ProcessedBatch:
        del runtime_mode
        samples = _normalize_samples(raw_sample)
        if "enable_thinking" in processor_kwargs:
            raise BackendIntegrationError(
                "Qwen2.5-VL does not expose Qwen3-VL's enable_thinking "
                "processor contract."
            )
        modality_counts = [
            _conversation_modality_counts(sample.conversation) for sample in samples
        ]
        conversations = [sample.conversation for sample in samples]
        conversation_input: Any = (
            conversations[0] if len(conversations) == 1 else conversations
        )
        chat_template_kwargs = dict(processor_kwargs.pop("chat_template_kwargs", {}))
        if "enable_thinking" in chat_template_kwargs:
            raise BackendIntegrationError(
                "Qwen2.5-VL chat templates do not accept enable_thinking."
            )
        chat_template_kwargs.setdefault("tokenize", False)
        chat_template_kwargs.setdefault("add_generation_prompt", True)
        text = processor.apply_chat_template(
            conversation_input,
            **chat_template_kwargs,
        )

        media_loader = self.media_loader
        if media_loader is None:
            try:
                from qwen_vl_utils import process_vision_info
            except ImportError as exc:
                raise BackendIntegrationError(
                    "Qwen2.5-VL preprocessing requires qwen-vl-utils>=0.0.6."
                ) from exc
            media_loader = process_vision_info
        # Qwen2.5-VL's compatibility contract is the original binary return.
        # Do not pass Qwen3-only patch-size or video-metadata kwargs here.
        loaded = media_loader(conversation_input)
        if not isinstance(loaded, tuple) or len(loaded) != 2:
            raise BackendIntegrationError(
                "Qwen2.5-VL media loader must return (images, videos)."
            )
        images, videos = loaded

        processor_kwargs.setdefault("return_tensors", "pt")
        processor_kwargs.setdefault("padding", True)
        outputs = processor(
            text=text,
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

        spatial_merge_size = _spatial_merge_size(processor)
        expected_counts = _partition_expected_counts(
            modality_counts,
            image_grid_thw=model_inputs.get("image_grid_thw"),
            video_grid_thw=model_inputs.get("video_grid_thw"),
            spatial_merge_size=spatial_merge_size,
        )
        input_shape = _shape(model_inputs.get("input_ids"))
        video_media_ids = tuple(
            f"{sample.sample_id}:video-{video_index}"
            for sample, counts in zip(samples, modality_counts)
            for video_index in range(counts["video"])
        )
        video_conditioning_texts = tuple(
            user_text_from_conversation(sample.conversation)
            for sample, counts in zip(samples, modality_counts)
            for _ in range(counts["video"])
        )
        metadata = {
            "phase": phase,
            "provider": "transformers.Qwen2_5_VLProcessor",
            "processor_output_keys": sorted(model_inputs),
            "processor_output_contract": {
                name: {"shape": _shape(value), "dtype": _dtype_name(value)}
                for name, value in model_inputs.items()
            },
            "input_sequence_length": (
                input_shape[1] if input_shape and len(input_shape) > 1 else None
            ),
            "modality_counts_by_sample": modality_counts,
            "video_media_ids": video_media_ids,
            "video_conditioning_texts": video_conditioning_texts,
            "video_count": len(video_media_ids),
            "expected_token_counts_by_sample": expected_counts,
            "token_ids": {
                "image": _token_id(processor, "image_token", "<|image_pad|>"),
                "video": _token_id(processor, "video_token", "<|video_pad|>"),
            },
            "image_patch_size": QWEN25_VL_IMAGE_PATCH_SIZE,
            "spatial_merge_size": spatial_merge_size,
            "mrope": {
                "owner": "backbone",
                "dimensions": ("temporal", "height", "width"),
                "video_timing_key": "second_per_grid_ts",
            },
            "second_per_grid_ts_present": (
                model_inputs.get("second_per_grid_ts") is not None
            ),
            "media_loader_return_arity": 2,
        }
        return ProcessedBatch(
            model_inputs=model_inputs,
            sample_ids=[sample.sample_id for sample in samples],
            prompt_lengths=prompt_lengths,
            metadata=metadata,
            source_batch=raw_sample,
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
        if not isinstance(target_text, str) or not target_text.strip():
            raise BackendIntegrationError(
                "Qwen2.5-VL teacher forcing requires a non-empty textual target."
            )
        samples = _normalize_samples(raw_sample)
        if len(samples) != 1:
            raise BackendIntegrationError(
                "Qwen2.5-VL analysis teacher forcing is single-sample."
            )
        sample = samples[0]
        prompt = prompt_batch or self.prepare_inputs(
            processor,
            sample,
            phase=phase,
            runtime_mode=runtime_mode,
        )
        full_sample = Qwen25VLSample(
            sample_id=sample.sample_id,
            conversation=[
                *sample.conversation,
                {
                    "role": "assistant",
                    "content": [{"type": "text", "text": target_text}],
                },
            ],
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
            completion_end_token_ids=_qwen_completion_end_token_ids(processor),
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
class Qwen25VLModelAdapter:
    """Load Qwen2.5-VL and expose visual blocks, merger, and language model."""

    profile: BackboneCapabilities = QWEN25_VL_PROFILE

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
        try:
            from transformers import (
                AutoConfig,
                AutoModelForImageTextToText,
                Qwen2_5_VLForConditionalGeneration,
            )

            _ = Qwen2_5_VLForConditionalGeneration
        except ImportError as exc:
            raise BackendIntegrationError(
                "Qwen2.5-VL requires transformers>=4.49,<5."
            ) from exc
        if revision is not None:
            load_kwargs["revision"] = revision
        config_kwargs = {
            key: load_kwargs[key]
            for key in ("revision", "trust_remote_code", "local_files_only")
            if key in load_kwargs
        }
        config = AutoConfig.from_pretrained(model_path_or_id, **config_kwargs)
        # Qwen2.5's HF compatibility __getattribute__ delegates instance fields
        # to text_config, including model_type after save/reload. Validate the
        # actual top-level config class, not the delegated text-only value.
        model_type = getattr(type(config), "model_type", None) or getattr(config, "model_type", None)
        if model_type != "qwen2_5_vl":
            raise BackendIntegrationError(
                f"Checkpoint model_type={model_type!r} "
                "is not Qwen2.5-VL."
            )
        if "torch_dtype" in load_kwargs and "dtype" not in load_kwargs:
            load_kwargs["dtype"] = load_kwargs.pop("torch_dtype")
        model = AutoModelForImageTextToText.from_pretrained(
            model_path_or_id,
            **load_kwargs,
        )
        model.train(mode=train)
        return model

    def structural_handles(self, model: Any) -> Mapping[str, Any]:
        handles: dict[str, Any] = {}

        def add(path: str, value: Any) -> None:
            if value is not None:
                handles[path] = value

        core = getattr(model, "model", None)
        visual = getattr(core, "visual", None)
        if visual is None:
            visual = getattr(model, "visual", None)
        language_model = getattr(core, "language_model", None)
        if language_model is None:
            language_model = getattr(model, "language_model", None)
        visual_blocks = getattr(visual, "blocks", None)
        merger = getattr(visual, "merger", None)
        add("model", core)
        add("llm", language_model)
        add("language_model", language_model)
        get_input_embeddings = getattr(model, "get_input_embeddings", None)
        add(
            "llm.input_embeddings",
            (
                get_input_embeddings()
                if callable(get_input_embeddings)
                else getattr(language_model, "embed_tokens", None)
            ),
        )
        add("llm.final_norm", getattr(language_model, "norm", None))
        add("lm_head", getattr(model, "lm_head", None))
        add("vision_encoder", visual)
        add("vision_encoder.blocks", visual_blocks)
        add("vision_projector", merger)
        add("vision_projector.merger", merger)
        add("mrope_owner", core)
        add("vision_spatial_merge_size", getattr(visual, "spatial_merge_size", None))
        add(
            "video_features_pre_placeholder_scatter",
            getattr(core, "get_video_features", None),
        )
        add("video_grid_thw_sparse_mrope", getattr(core, "get_rope_index", None))
        add(
            "video_compression_bridge",
            getattr(model, "mllm_video_compression_bridge", None),
        )
        for index, block in enumerate(visual_blocks or ()):
            add(f"vision_encoder.block[{index}]", block)
            add(f"vision_encoder.block[{index}].attn", getattr(block, "attn", None))
            add(f"vision_encoder.block[{index}].ffn", getattr(block, "mlp", None))
        for index, layer in enumerate(getattr(language_model, "layers", ())):
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
        return handles

    def external_code_info(self) -> dict[str, Any]:
        return {
            "code_origin": self.profile.code_origin,
            "source_ref": self.profile.source_ref,
            "loader_strategy": self.profile.loader_strategy,
            "minimum_transformers_version": "4.49.0",
            "verified_transformers_version": "4.57.3",
            "qwen_vl_utils_return_contract": "(images, videos)",
            "image_patch_size": QWEN25_VL_IMAGE_PATCH_SIZE,
            "supported_input_slice": "text/image/single-video -> text",
            "video_facade_implemented": True,
            "video_bridge_provider": "transformers-4.57.3-instance-local",
            "video_real_checkpoint_verified": False,
        }


def _contiguous_token_spans(
    input_ids: Any,
    token_ids: Mapping[str, int | None],
) -> list[TokenSpan]:
    rows = _rows(input_ids)
    spans: list[TokenSpan] = []
    media_units = {"image": "image_patch", "video": "video_patch"}
    for sample_index, row in enumerate(rows):
        for modality in QWEN25_VL_INPUT_MODALITIES:
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
                            source="qwen25_vl_processor",
                            media_unit=media_units[modality],
                            encoder_origin="vision_encoder",
                        )
                    )
                    start = None
    return sorted(spans, key=lambda span: (span.sample_index, span.start, span.end))


@dataclass(frozen=True)
class Qwen25VLStateBuilder:
    """Normalize Qwen2.5-VL fields while preserving native M-RoPE kwargs."""

    profile: BackboneCapabilities = QWEN25_VL_PROFILE

    def build(
        self,
        processed_batch: ProcessedBatch,
        *,
        raw_batch: Any | None = None,
        phase: Literal["dry_run", "train", "eval"] = "train",
    ) -> ForwardState:
        del raw_batch
        outputs = processed_batch.model_inputs
        metadata = dict(processed_batch.metadata)
        metadata.update(
            {
                "state_phase": phase,
                "native_to_normalized_fields": {
                    "pixel_values_videos": "video_pixel_values",
                    "second_per_grid_ts": "video_second_per_grid",
                },
            }
        )
        return ForwardState(
            input_ids=outputs.get("input_ids"),
            inputs_embeds=outputs.get("inputs_embeds"),
            attention_mask=outputs.get("attention_mask"),
            position_ids=outputs.get("position_ids"),
            labels=outputs.get("labels"),
            pixel_values=outputs.get("pixel_values"),
            image_grid_thw=outputs.get("image_grid_thw"),
            video_pixel_values=outputs.get("pixel_values_videos"),
            video_grid_thw=outputs.get("video_grid_thw"),
            video_second_per_grid=outputs.get("second_per_grid_ts"),
            native_model_inputs=dict(outputs),
            token_type_mask=outputs.get("mm_token_type_ids"),
            modality_token_spans=_contiguous_token_spans(
                outputs.get("input_ids"),
                metadata.get("token_ids", {}),
            ),
            modality_metadata=metadata,
            sample_ids=list(processed_batch.sample_ids),
        )


@dataclass(frozen=True)
class Qwen25VLLegalizer:
    """Preserve native baseline state and reject unbound token mutation."""

    profile: BackboneCapabilities = QWEN25_VL_PROFILE

    def apply(
        self,
        state: ForwardState,
        *,
        token_actions: list[TokenAction],
        policy: LegalizationPolicy,
        stage_name: str,
    ) -> ForwardState:
        del policy, stage_name
        if token_actions:
            raise NotActivatedError(
                "Qwen2.5-VL token/grid/placeholder/M-RoPE repair requires an "
                "explicit image or video compression binding."
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
        f"{item.name}[{item.field}]: expected={item.expected!r}, "
        f"actual={item.actual!r}"
        for item in violations
    )
    raise InvariantError(f"{prefix}: {details}")


def _output_field(output: Any, name: str) -> Any | None:
    if output is None:
        return None
    if isinstance(output, Mapping):
        return output.get(name)
    return getattr(output, name, None)


@contextmanager
def _capture_qwen25_vl_traces(
    model: Any,
    request: TraceRequest | None,
) -> Iterator[TraceArtifacts | None]:
    if request is None or not request.enabled():
        yield None
        return
    artifacts = TraceArtifacts(
        metadata={
            "capture_semantics": "last_forward_step",
            "provider": "transformers-qwen25-vl-forward-hook",
        }
    )
    handles: list[Any] = []
    if hasattr(model, "register_forward_hook"):

        def model_hook(_module: Any, _args: Any, _kwargs: Any, output: Any) -> None:
            if request.text_logits:
                artifacts.text_logits = _output_field(output, "logits")
            if request.text_hidden_states:
                artifacts.text_hidden_states = _output_field(output, "hidden_states")
            if request.text_attentions:
                artifacts.text_attentions = _output_field(output, "attentions")
            if request.text_kv_cache:
                artifacts.text_kv_cache = _output_field(output, "past_key_values")

        handles.append(model.register_forward_hook(model_hook, with_kwargs=True))
    try:
        yield artifacts
    finally:
        for handle in reversed(handles):
            handle.remove()


@dataclass(frozen=True)
class Qwen25VLRunner:
    """Run standard Qwen2.5-VL text forward/generation."""

    processor_adapter: Qwen25VLProcessorAdapter = field(
        default_factory=Qwen25VLProcessorAdapter
    )
    native_validator: Qwen25VLNativeValidator = field(
        default_factory=Qwen25VLNativeValidator
    )
    profile: BackboneCapabilities = QWEN25_VL_PROFILE

    def forward_train(self, runtime: BackboneRuntime, state: ForwardState) -> Any:
        session = runtime.metadata.get("_compression_session")
        if session is not None:
            session.clear_diagnostic_records()
        model_inputs = {
            name: _move_value(value, device=runtime.device, dtype=runtime.dtype)
            for name, value in state.native_model_inputs.items()
        }
        with qwen25_vl_video_request_context(
            runtime.model,
            state.modality_metadata,
        ):
            return runtime.model(**model_inputs)

    def generate(
        self,
        runtime: BackboneRuntime,
        state: ForwardState,
        *,
        processed_batch: ProcessedBatch,
        trace_request: TraceRequest | None = None,
        **gen_kwargs: Any,
    ) -> GenerationResult:
        session = runtime.metadata.get("_compression_session")
        if session is not None:
            session.clear_diagnostic_records()
        _raise_violations(
            "Qwen2.5-VL native input validation failed",
            self.native_validator.validate_state(runtime, state, processed_batch),
        )
        if gen_kwargs.pop("return_audio", False):
            raise InvariantError("Qwen2.5-VL has no Talker or speech output branch.")
        if gen_kwargs.pop("stream", False) or gen_kwargs.get("streamer") is not None:
            raise NotActivatedError(
                "The platform Qwen2.5-VL runner currently returns complete "
                "GenerationResult objects."
            )

        request = trace_request or TraceRequest()
        omni_trace_fields = (
            "thinker_logits",
            "thinker_hidden_states",
            "thinker_attentions",
            "thinker_kv_cache",
            "thinker_reply_part",
            "talker_logits",
            "talker_hidden_states",
            "talker_attentions",
            "talker_kv_cache",
            "talker_codec_token_ids",
            "token2wav_intermediates",
        )
        unsupported_traces = [
            name for name in omni_trace_fields if getattr(request, name)
        ]
        if unsupported_traces:
            raise InvariantError(
                "Qwen2.5-VL has no Thinker/Talker/Token2Wav trace branches; "
                f"unsupported trace fields: {unsupported_traces}."
            )
        if request.text_hidden_states:
            gen_kwargs.setdefault("output_hidden_states", True)
        if request.text_attentions or request._full_generation_text_attentions:
            gen_kwargs.setdefault("output_attentions", True)
        if request.enabled():
            gen_kwargs.setdefault("return_dict_in_generate", True)
        if gen_kwargs.get("do_sample") is False:
            gen_kwargs.setdefault("temperature", None)
            gen_kwargs.setdefault("top_p", None)
            gen_kwargs.setdefault("top_k", None)

        model_inputs = {
            name: _move_value(value, device=runtime.device, dtype=runtime.dtype)
            for name, value in state.native_model_inputs.items()
        }
        input_shape = _shape(model_inputs.get("input_ids"))
        if input_shape is None or len(input_shape) != 2:
            raise InvariantError("Qwen2.5-VL generation requires rank-2 input_ids.")
        input_width = input_shape[1]
        with qwen25_vl_video_request_context(
            runtime.model,
            state.modality_metadata,
        ):
            with sgl_target_generation(runtime, processed_batch, gen_kwargs):
                with _capture_qwen25_vl_traces(
                    runtime.model, request
                ) as captured_traces:
                    raw_output = runtime.model.generate(**model_inputs, **gen_kwargs)
        if request._full_generation_text_attentions and captured_traces is not None:
            generated_attentions = _output_field(raw_output, "attentions")
            if generated_attentions is not None:
                captured_traces.text_attentions = generated_attentions
                captured_traces.metadata.setdefault("field_capture_semantics", {})[
                    "text_attentions"
                ] = "full_generate_steps"
        sequences = getattr(raw_output, "sequences", raw_output)
        generated_ids = _slice_generated_ids(sequences, input_width)
        decoded = self.processor_adapter.detokenize(
            runtime.processor,
            generated_ids,
        )
        result = GenerationResult(
            text=TextGenerationOutput(token_ids=generated_ids, text=decoded),
            speech=None,
            traces=captured_traces if request.enabled() else None,
            sample_ids=list(processed_batch.sample_ids),
            metadata={
                "provider": (
                    "transformers.Qwen2_5_VLForConditionalGeneration.generate"
                ),
                "return_audio": False,
                "input_sequence_length": input_width,
                "streaming_supported": False,
                "input_modalities": [
                    "text",
                    *[
                        modality
                        for modality, count in _total_modality_counts(
                            processed_batch
                        ).items()
                        if count
                    ],
                ],
                "output_modality": "text",
                "mrope_owner": "backbone",
            },
        )
        _raise_violations(
            "Qwen2.5-VL native output validation failed",
            self.native_validator.validate_generation(
                runtime,
                state,
                processed_batch,
                result,
            ),
        )
        return result


def _total_modality_counts(processed_batch: ProcessedBatch) -> dict[str, int]:
    totals = {"image": 0, "video": 0}
    for counts in processed_batch.metadata.get("modality_counts_by_sample", ()):
        for modality in totals:
            totals[modality] += int(counts.get(modality, 0))
    return totals


def create_qwen25_vl_backbone() -> BackboneFacade:
    """Compose the Qwen2.5-VL facade without retaining runtime objects."""

    profile = QWEN25_VL_PROFILE
    processor_adapter = Qwen25VLProcessorAdapter(profile)
    native_validator = Qwen25VLNativeValidator()
    return BackboneFacade(
        processor_adapter=processor_adapter,
        model_adapter=Qwen25VLModelAdapter(profile),
        state_builder=Qwen25VLStateBuilder(profile),
        legalizer=Qwen25VLLegalizer(profile),
        runner=Qwen25VLRunner(processor_adapter, native_validator, profile),
        operation_applier=None,
        native_validator=native_validator,
    )


__all__ = [
    "DEFAULT_SYSTEM_PROMPT",
    "QWEN25_VL_IMAGE_PATCH_SIZE",
    "Qwen25VLLegalizer",
    "Qwen25VLModelAdapter",
    "Qwen25VLProcessorAdapter",
    "Qwen25VLRunner",
    "Qwen25VLSample",
    "Qwen25VLStateBuilder",
    "create_qwen25_vl_backbone",
]
