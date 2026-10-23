"""Source-tree facade for native LLaVA-OneVision text, image, and video."""

from __future__ import annotations

import json
import math
import sys
from collections.abc import Callable, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field, replace
from pathlib import Path
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
from mllm_comp.backbones.profiles import (
    LLAVA_ONEVISION_PROFILE as _BASE_LLAVA_ONEVISION_PROFILE,
    BackboneCapabilities,
)
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
from mllm_comp.compressors.image.sgl import sgl_target_generation

from .image_compression import (
    LLAVA_ONEVISION_BRIDGE_API_VERSION,
    LLAVA_ONEVISION_SOURCE_COMMIT,
    LLAVA_ONEVISION_SOURCE_ROUTE,
)
from .video_compression import llava_onevision_video_request_context


# The facade capability describes the native source route. Exact compressed
# algorithm/backbone/phase support remains owned by the support registry and is
# not activated by this profile.
LLAVA_ONEVISION_NATIVE_PROFILE = replace(
    _BASE_LLAVA_ONEVISION_PROFILE,
    source_ref=(
        "vendored LLaVA-NeXT LLaVA-OneVision Qwen source tree "
        "(text/image/video slice)"
    ),
)
# Kept as a public alias because ImageLLM integrations already import it.  Its
# value now names the expanded native facade rather than an image-only claim.
LLAVA_ONEVISION_IMAGE_PROFILE = LLAVA_ONEVISION_NATIVE_PROFILE

DEFAULT_CONVERSATION_MODE = "qwen_1_5"
LLAVA_ONEVISION_MAX_VIDEO_FRAMES = 32


@dataclass(frozen=True)
class LlavaOneVisionSample:
    """One text, image, or single-video prompt."""

    sample_id: str | int
    prompt: str
    images: tuple[Any, ...] = ()
    video: Any | None = None
    conversation_mode: str = DEFAULT_CONVERSATION_MODE

    def __post_init__(self) -> None:
        if self.video is not None and self.images:
            raise ValueError("LLaVA samples cannot mix image and video media")


@dataclass(frozen=True)
class LlavaOneVisionDecodedVideo:
    """Uniformly sampled local video frames with immutable timing identity."""

    frames: Any
    frame_size: tuple[int, int]
    frame_indices: tuple[int, ...]
    timestamps_seconds: tuple[float, ...]
    source: str | None = None
    total_frames: int | None = None
    fps: float | None = None

    def __post_init__(self) -> None:
        frame_count = _leading_length(self.frames)
        if frame_count <= 0:
            raise ValueError("Decoded LLaVA video must contain at least one frame")
        width, height = self.frame_size
        if width <= 0 or height <= 0:
            raise ValueError("Decoded LLaVA video frame size must be positive")
        if len(self.frame_indices) != frame_count:
            raise ValueError("frame_indices must contain one entry per decoded frame")
        if len(self.timestamps_seconds) != frame_count:
            raise ValueError(
                "timestamps_seconds must contain one entry per decoded frame"
            )
        if any(index < 0 for index in self.frame_indices):
            raise ValueError("frame_indices must be non-negative")
        if any(
            not math.isfinite(value) or value < 0 for value in self.timestamps_seconds
        ):
            raise ValueError("timestamps_seconds must be finite and non-negative")


@dataclass(frozen=True)
class LlavaOneVisionSourceAPI:
    """Lazy imports from the selected vendored LLaVA-NeXT tree."""

    load_pretrained_model: Callable[..., Any]
    tokenizer_image_token: Callable[..., Any]
    process_images: Callable[..., Any]
    get_model_name_from_path: Callable[[str], str]
    conv_templates: Mapping[str, Any]
    image_token_index: int
    default_image_token: str
    default_im_start_token: str
    default_im_end_token: str


@dataclass(frozen=True)
class LlavaOneVisionLoadedArtifacts:
    tokenizer: Any
    model: Any
    image_processor: Any
    context_length: int
    model_name: str
    model_path_or_id: str
    revision: str | None
    load_signature: Mapping[str, Any]


class LlavaOneVisionSourceLoader:
    """Load one source-tree tuple and share it between facade components."""

    def __init__(
        self,
        *,
        api_loader: Callable[[], LlavaOneVisionSourceAPI] | None = None,
        snapshot_resolver: Callable[[str, str], str] | None = None,
    ) -> None:
        self._api_loader = api_loader or _import_source_api
        self._snapshot_resolver = snapshot_resolver or _resolve_checkpoint_snapshot
        self._api: LlavaOneVisionSourceAPI | None = None
        self._artifacts: dict[tuple[str, str | None], LlavaOneVisionLoadedArtifacts] = (
            {}
        )

    def api(self) -> LlavaOneVisionSourceAPI:
        if self._api is None:
            self._api = self._api_loader()
        return self._api

    def get(
        self,
        model_path_or_id: str,
        *,
        revision: str | None = None,
    ) -> LlavaOneVisionLoadedArtifacts | None:
        return self._artifacts.get((str(model_path_or_id), revision))

    def load(
        self,
        model_path_or_id: str,
        *,
        revision: str | None = None,
        **load_kwargs: Any,
    ) -> LlavaOneVisionLoadedArtifacts:
        requested_vision_revision = load_kwargs.get("vision_tower_revision")
        if requested_vision_revision is not None and (
            not isinstance(requested_vision_revision, str)
            or not requested_vision_revision.strip()
        ):
            raise BackendIntegrationError(
                "LLaVA vision_tower_revision must be a non-empty string."
            )
        if revision is not None and requested_vision_revision is None:
            raise BackendIntegrationError(
                "Pinned LLaVA checkpoint loading requires an explicit "
                "vision_tower_revision."
            )
        key = (str(model_path_or_id), revision)
        existing = self._artifacts.get(key)
        if existing is not None:
            existing_vision = existing.load_signature.get("vision_tower")
            existing_vision_revision = (
                existing_vision.get("revision")
                if isinstance(existing_vision, Mapping)
                else None
            )
            if (
                requested_vision_revision is not None
                and requested_vision_revision != existing_vision_revision
            ):
                raise BackendIntegrationError(
                    "LLaVA checkpoint is already loaded with vision tower "
                    f"revision {existing_vision_revision!r}; refusing requested "
                    f"revision {requested_vision_revision!r}."
                )
            return existing
        api = self.api()
        kwargs = dict(load_kwargs)
        model_base = kwargs.pop("model_base", None)
        model_name = str(
            kwargs.pop(
                "model_name", api.get_model_name_from_path(str(model_path_or_id))
            )
        )
        resolved_model_path = (
            self._snapshot_resolver(str(model_path_or_id), revision)
            if revision is not None
            else str(model_path_or_id)
        )
        vision_tower_revision = kwargs.pop("vision_tower_revision", None)
        kwargs.setdefault("multimodal", True)
        kwargs.setdefault("attn_implementation", "sdpa")
        raw_overwrite_config = kwargs.pop("overwrite_config", None)
        if raw_overwrite_config is not None and not isinstance(
            raw_overwrite_config, Mapping
        ):
            raise BackendIntegrationError("LLaVA overwrite_config must be a mapping.")
        overwrite_config = (
            {
                "mm_spatial_pool_stride": 2,
                "mm_spatial_pool_mode": "bilinear",
            }
            if raw_overwrite_config is None
            else dict(raw_overwrite_config)
        )
        vision_tower_source: dict[str, str] | None = None
        if vision_tower_revision is not None:
            vision_tower_id, resolved_vision_tower = _resolve_vision_tower_snapshot(
                resolved_model_path,
                vision_tower_revision,
                self._snapshot_resolver,
            )
            overwrite_config["mm_vision_tower"] = resolved_vision_tower
            vision_tower_source = {
                "model_id": vision_tower_id,
                "revision": vision_tower_revision,
                "resolved_source": resolved_vision_tower,
            }
        kwargs["overwrite_config"] = overwrite_config
        if "torch_dtype" in kwargs:
            kwargs["torch_dtype"] = _source_dtype_name(kwargs["torch_dtype"])
        loaded = api.load_pretrained_model(
            resolved_model_path,
            model_base,
            model_name,
            **kwargs,
        )
        if not isinstance(loaded, Sequence) or len(loaded) != 4:
            raise BackendIntegrationError(
                "LLaVA load_pretrained_model must return "
                "(tokenizer, model, image_processor, context_length)."
            )
        tokenizer, model, image_processor, context_length = loaded
        if tokenizer is None or model is None or image_processor is None:
            raise BackendIntegrationError(
                "LLaVA source loader returned an incomplete multimodal tuple."
            )
        source_evidence: dict[str, Any] = {
            "checkpoint": {
                "model_id": str(model_path_or_id),
                "revision": revision,
                "resolved_source": resolved_model_path,
            },
            "dependencies": {},
        }
        if vision_tower_source is not None:
            source_evidence["dependencies"]["vision_tower"] = vision_tower_source
        setattr(model, "_mllm_comp_source_evidence", source_evidence)
        safe_load_signature = {
            key: kwargs[key]
            for key in (
                "multimodal",
                "attn_implementation",
                "torch_dtype",
                "device_map",
                "overwrite_config",
            )
            if key in kwargs
        }
        artifacts = LlavaOneVisionLoadedArtifacts(
            tokenizer=tokenizer,
            model=model,
            image_processor=image_processor,
            context_length=int(context_length),
            model_name=model_name,
            model_path_or_id=str(model_path_or_id),
            revision=revision,
            load_signature={
                **safe_load_signature,
                "resolved_checkpoint_source": resolved_model_path,
                "snapshot_revision": revision,
                "vision_tower": vision_tower_source,
            },
        )
        self._artifacts[key] = artifacts
        return artifacts


def _resolve_checkpoint_snapshot(model_path_or_id: str, revision: str) -> str:
    """Resolve every checkpoint-owned artifact through one immutable snapshot."""

    local = Path(model_path_or_id).expanduser()
    if local.is_dir():
        return str(local.resolve())
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:  # pragma: no cover - runtime dependency
        raise BackendIntegrationError(
            "Pinned LLaVA checkpoint loading requires huggingface_hub."
        ) from exc
    try:
        return str(snapshot_download(repo_id=model_path_or_id, revision=revision))
    except Exception as exc:
        raise BackendIntegrationError(
            "Could not resolve the pinned LLaVA checkpoint snapshot "
            f"{model_path_or_id!r}@{revision}."
        ) from exc


def _resolve_vision_tower_snapshot(
    resolved_model_path: str,
    revision: str,
    snapshot_resolver: Callable[[str, str], str],
) -> tuple[str, str]:
    """Resolve the checkpoint-declared vision tower at an explicit revision."""

    config_path = Path(resolved_model_path) / "config.json"
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise BackendIntegrationError(
            "Pinned LLaVA vision-tower resolution requires a readable "
            f"checkpoint config at {config_path}."
        ) from exc
    if not isinstance(config, Mapping):
        raise BackendIntegrationError(
            "Pinned LLaVA checkpoint config must contain a JSON object."
        )
    vision_tower_id = config.get("mm_vision_tower")
    if not isinstance(vision_tower_id, str) or not vision_tower_id.strip():
        raise BackendIntegrationError(
            "Pinned LLaVA checkpoint config must declare mm_vision_tower."
        )
    try:
        resolved = snapshot_resolver(vision_tower_id, revision)
    except Exception as exc:
        raise BackendIntegrationError(
            "Could not resolve the pinned LLaVA vision tower "
            f"{vision_tower_id!r}@{revision}."
        ) from exc
    return vision_tower_id, str(resolved)


@dataclass
class LlavaOneVisionProcessorBundle:
    """Lazy source tuple projected to the platform's processor role."""

    model_path_or_id: str
    source_loader: LlavaOneVisionSourceLoader
    revision: str | None = None
    load_kwargs: dict[str, Any] = field(default_factory=dict)
    loaded_artifacts: LlavaOneVisionLoadedArtifacts | None = None

    def artifacts(self) -> LlavaOneVisionLoadedArtifacts:
        if self.loaded_artifacts is None:
            self.loaded_artifacts = self.source_loader.get(
                self.model_path_or_id,
                revision=self.revision,
            )
        if self.loaded_artifacts is None:
            self.loaded_artifacts = self.source_loader.load(
                self.model_path_or_id,
                revision=self.revision,
                **self.load_kwargs,
            )
        return self.loaded_artifacts

    @property
    def tokenizer(self) -> Any:
        return self.artifacts().tokenizer

    @property
    def image_processor(self) -> Any:
        return self.artifacts().image_processor

    @property
    def model_config(self) -> Any:
        return getattr(self.artifacts().model, "config", None)

    @property
    def context_length(self) -> int:
        return self.artifacts().context_length


@dataclass(frozen=True)
class LlavaOneVisionProcessorAdapter:
    """Use vendored conversation and native image/video preprocessing."""

    profile: BackboneCapabilities = LLAVA_ONEVISION_IMAGE_PROFILE
    source_loader: LlavaOneVisionSourceLoader = field(
        default_factory=LlavaOneVisionSourceLoader,
        compare=False,
        repr=False,
    )
    image_loader: Callable[[Any], Any] | None = field(
        default=None,
        compare=False,
        repr=False,
    )
    video_loader: Callable[[Any, int], LlavaOneVisionDecodedVideo] | None = field(
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
    ) -> LlavaOneVisionProcessorBundle:
        # Heavy loading is deferred until ModelAdapter.load_model so the source
        # tree's tuple loader runs once with the model's dtype/device settings.
        return LlavaOneVisionProcessorBundle(
            model_path_or_id=str(model_path_or_id),
            source_loader=self.source_loader,
            revision=revision,
            load_kwargs=dict(load_kwargs),
        )

    def prepare_inputs(
        self,
        processor: LlavaOneVisionProcessorBundle,
        raw_sample: Any,
        *,
        phase: str,
        runtime_mode: str = "default",
        **processor_kwargs: Any,
    ) -> ProcessedBatch:
        del runtime_mode
        samples = _normalise_samples(raw_sample)
        if len(samples) != 1:
            raise BackendIntegrationError(
                "The LLaVA-OneVision source-tree baseline is single-sample."
            )
        sample = samples[0]
        artifacts = processor.artifacts()
        api = processor.source_loader.api()
        model_config = getattr(artifacts.model, "config", None)
        if model_config is None:
            raise BackendIntegrationError("Loaded LLaVA model has no config.")
        conversation_mode = str(
            processor_kwargs.pop("conversation_mode", sample.conversation_mode)
        )
        if conversation_mode not in api.conv_templates:
            raise BackendIntegrationError(
                f"Unknown LLaVA conversation template {conversation_mode!r}."
            )
        images = tuple(
            (self.image_loader or _load_image)(item) for item in sample.images
        )
        max_video_frames = processor_kwargs.pop(
            "max_video_frames", LLAVA_ONEVISION_MAX_VIDEO_FRAMES
        )
        if max_video_frames != LLAVA_ONEVISION_MAX_VIDEO_FRAMES:
            raise BackendIntegrationError(
                "The frozen LLaVA validation profile requires "
                f"max_video_frames={LLAVA_ONEVISION_MAX_VIDEO_FRAMES}."
            )
        decoded_video = (
            (self.video_loader or _load_video)(sample.video, max_video_frames)
            if sample.video is not None
            else None
        )
        if decoded_video is not None and not isinstance(
            decoded_video, LlavaOneVisionDecodedVideo
        ):
            raise BackendIntegrationError(
                "LLaVA video_loader must return LlavaOneVisionDecodedVideo."
            )
        assistant_text = processor_kwargs.pop("assistant_text", None)
        media_count = len(images) + int(decoded_video is not None)
        prompt = _render_prompt(
            sample.prompt,
            media_count=media_count,
            conversation_mode=conversation_mode,
            api=api,
            model_config=model_config,
            assistant_text=assistant_text,
        )
        token_ids = api.tokenizer_image_token(
            prompt,
            artifacts.tokenizer,
            api.image_token_index,
            return_tensors="pt",
        )
        input_ids = _ensure_batch_dimension(token_ids)
        pad_token_id = getattr(artifacts.tokenizer, "pad_token_id", None)
        if pad_token_id is None:
            pad_token_id = getattr(artifacts.tokenizer, "eos_token_id", None)
        attention_mask = _attention_mask(input_ids, pad_token_id=pad_token_id)
        if decoded_video is not None:
            preprocess = getattr(artifacts.image_processor, "preprocess", None)
            if not callable(preprocess):
                raise BackendIntegrationError(
                    "LLaVA video processing requires image_processor.preprocess."
                )
            processed_video = preprocess(
                decoded_video.frames,
                return_tensors="pt",
            )
            if (
                not isinstance(processed_video, Mapping)
                or "pixel_values" not in processed_video
            ):
                raise BackendIntegrationError(
                    "LLaVA video processor did not return pixel_values."
                )
            pixel_values = [processed_video["pixel_values"]]
            image_sizes = (decoded_video.frame_size,)
            native_modalities = ("video",)
        else:
            image_sizes = tuple(_image_size(image) for image in images)
            pixel_values = (
                api.process_images(images, artifacts.image_processor, model_config)
                if images
                else None
            )
            native_modalities = ("image",) if images else ("text",)
        placeholder_positions = _find_token_positions(
            input_ids,
            api.image_token_index,
        )
        if len(placeholder_positions) != media_count:
            raise BackendIntegrationError(
                "LLaVA prompt/media mismatch after tokenizer_image_token: "
                f"placeholders={len(placeholder_positions)}, media={media_count}."
            )
        model_inputs: dict[str, Any] = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "images": pixel_values,
            "image_sizes": list(image_sizes),
            "modalities": native_modalities,
        }
        labels = processor_kwargs.pop("labels", None)
        if labels is not None:
            model_inputs["labels"] = labels
        if processor_kwargs:
            raise BackendIntegrationError(
                f"Unsupported LLaVA processor kwargs: {sorted(processor_kwargs)}."
            )
        patch_merge_type = str(getattr(model_config, "mm_patch_merge_type", "flat"))
        aspect_ratio = str(getattr(model_config, "image_aspect_ratio", "square"))
        newline_position = str(
            getattr(model_config, "mm_newline_position", "one_token")
        )
        prompt_lengths = _row_sums(attention_mask)
        metadata = {
            "phase": phase,
            "provider": "vendored.LLaVA-NeXT",
            "source_route": LLAVA_ONEVISION_SOURCE_ROUTE,
            "vendored_source_commit": LLAVA_ONEVISION_SOURCE_COMMIT,
            "model_name": artifacts.model_name,
            "context_length": artifacts.context_length,
            "conversation_mode": conversation_mode,
            "rendered_prompt": prompt,
            "render_mode": (
                "teacher_forced" if assistant_text is not None else "generation_prompt"
            ),
            "token_ids": {
                "image": api.image_token_index,
                "video": api.image_token_index,
            },
            "image_count": len(images),
            "video_count": int(decoded_video is not None),
            "media_count": media_count,
            "image_sizes": image_sizes,
            "modalities": native_modalities,
            "media_modalities": (
                ("video",)
                if decoded_video is not None
                else tuple("image" for _ in images)
            ),
            "media_placeholder_positions": tuple(placeholder_positions),
            "image_placeholder_positions": (
                tuple(placeholder_positions) if images else ()
            ),
            "video_placeholder_positions": (
                tuple(placeholder_positions) if decoded_video is not None else ()
            ),
            "anyres": {
                "image_aspect_ratio": aspect_ratio,
                "mm_patch_merge_type": patch_merge_type,
                "mm_newline_position": newline_position,
                "image_grid_pinpoints": tuple(
                    getattr(model_config, "image_grid_pinpoints", ()) or ()
                ),
                "image_newline_required": "unpad" in patch_merge_type,
                "layout_phase": "pre_encoder_pre_anyres_pack",
            },
            "compression_bridge": {
                "api_version": LLAVA_ONEVISION_BRIDGE_API_VERSION,
                "source_route": LLAVA_ONEVISION_SOURCE_ROUTE,
                "provider_callbacks_required_for_compression": True,
            },
            "video": (
                {
                    "media_id": f"{sample.sample_id}:video-0",
                    "conditioning_text": sample.prompt,
                    "source": decoded_video.source,
                    "frame_count": len(decoded_video.frame_indices),
                    "frame_size": decoded_video.frame_size,
                    "frame_indices": decoded_video.frame_indices,
                    "timestamps_seconds": decoded_video.timestamps_seconds,
                    "total_frames": decoded_video.total_frames,
                    "fps": decoded_video.fps,
                    "max_video_frames": LLAVA_ONEVISION_MAX_VIDEO_FRAMES,
                    "decode_policy": "uniform_local_decord",
                }
                if decoded_video is not None
                else None
            ),
            "video_supported_in_this_slice": True,
            "video_real_checkpoint_verified": False,
        }
        return ProcessedBatch(
            model_inputs=model_inputs,
            sample_ids=[sample.sample_id],
            prompt_lengths=prompt_lengths,
            metadata=metadata,
            source_batch=raw_sample,
        )

    def prepare_teacher_forced_inputs(
        self,
        processor: LlavaOneVisionProcessorBundle,
        raw_sample: Any,
        *,
        target_text: str,
        prompt_batch: ProcessedBatch | None = None,
        phase: str,
        runtime_mode: str = "default",
    ) -> ProcessedBatch:
        if not isinstance(target_text, str) or not target_text.strip():
            raise BackendIntegrationError(
                "LLaVA teacher forcing requires a non-empty textual target."
            )
        prompt = prompt_batch or self.prepare_inputs(
            processor,
            raw_sample,
            phase=phase,
            runtime_mode=runtime_mode,
        )
        full = self.prepare_inputs(
            processor,
            raw_sample,
            phase=phase,
            runtime_mode=runtime_mode,
            assistant_text=target_text,
        )
        return build_teacher_forced_batch(
            prompt_batch=prompt,
            full_batch=full,
            special_token_ids=processor_special_token_ids(processor),
        )

    def detokenize(
        self,
        processor: LlavaOneVisionProcessorBundle,
        token_ids: Any,
        **decode_kwargs: Any,
    ) -> list[str]:
        decode_kwargs.setdefault("skip_special_tokens", True)
        return list(processor.tokenizer.batch_decode(token_ids, **decode_kwargs))


@dataclass(frozen=True)
class LlavaOneVisionModelAdapter:
    """Load the Qwen-based OneVision checkpoint through the vendored builder."""

    profile: BackboneCapabilities = LLAVA_ONEVISION_IMAGE_PROFILE
    source_loader: LlavaOneVisionSourceLoader = field(
        default_factory=LlavaOneVisionSourceLoader,
        compare=False,
        repr=False,
    )

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
        artifacts = self.source_loader.load(
            model_path_or_id,
            revision=revision,
            **load_kwargs,
        )
        model_type = getattr(
            getattr(artifacts.model, "config", None), "model_type", None
        )
        if model_type not in {"llava_qwen", "llava_qwen_moe"}:
            raise BackendIntegrationError(
                f"Checkpoint model_type={model_type!r} is not the selected "
                "Qwen-based LLaVA-OneVision source route."
            )
        train_method = getattr(artifacts.model, "train", None)
        if callable(train_method):
            train_method(mode=train)
        return artifacts.model

    def structural_handles(self, model: Any) -> Mapping[str, Any]:
        handles: dict[str, Any] = {}

        def add(name: str, value: Any) -> None:
            if value is not None:
                handles[name] = value

        get_model = getattr(model, "get_model", None)
        core = get_model() if callable(get_model) else getattr(model, "model", None)
        get_vision = getattr(model, "get_vision_tower", None)
        vision_tower = (
            get_vision()
            if callable(get_vision)
            else getattr(core, "vision_tower", None)
        )
        language_model = core
        projector = getattr(core, "mm_projector", None)
        vision_layers = _vision_layers(vision_tower)
        llm_layers = getattr(language_model, "layers", ())
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
        add("vision_encoder", vision_tower)
        add("vision_encoder.layers", vision_layers)
        add("vision_projector", projector)
        add("mm_projector", projector)
        add("image_newline", getattr(core, "image_newline", None))
        add(
            "anyres_pre_llm_splice",
            getattr(model, "prepare_inputs_labels_for_multimodal", None),
        )
        add(
            "video_projector_llm_splice",
            getattr(model, "prepare_inputs_labels_for_multimodal", None),
        )
        add(
            "video_compression_bridge",
            getattr(model, "mllm_video_compression_bridge", None),
        )
        for index, layer in enumerate(vision_layers or ()):
            add(f"vision_encoder.layer[{index}]", layer)
            add(
                f"vision_encoder.layer[{index}].attn",
                getattr(layer, "self_attn", None),
            )
            add(
                f"vision_encoder.layer[{index}].ffn",
                getattr(layer, "mlp", None),
            )
        for index, layer in enumerate(llm_layers or ()):
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
            "source_route": LLAVA_ONEVISION_SOURCE_ROUTE,
            "vendored_source_commit": LLAVA_ONEVISION_SOURCE_COMMIT,
            "loader": "llava.model.builder.load_pretrained_model",
            "tokenizer": "llava.mm_utils.tokenizer_image_token",
            "image_processor": "llava.mm_utils.process_images",
            "conversation_template": DEFAULT_CONVERSATION_MODE,
            "supported_input_slice": "text/image/single-video -> text",
            "video_facade_implemented": True,
            "video_real_checkpoint_verified": False,
        }


@dataclass(frozen=True)
class LlavaOneVisionStateBuilder:
    profile: BackboneCapabilities = LLAVA_ONEVISION_IMAGE_PROFILE

    def build(
        self,
        processed_batch: ProcessedBatch,
        *,
        raw_batch: Any | None = None,
        phase: Literal["dry_run", "train", "eval"] = "train",
    ) -> ForwardState:
        del raw_batch
        outputs = dict(processed_batch.model_inputs)
        metadata = {**processed_batch.metadata, "state_phase": phase}
        media_modalities = tuple(metadata.get("media_modalities", ()))
        media_positions = tuple(metadata.get("media_placeholder_positions", ()))
        spans = [
            TokenSpan(
                sample_index=0,
                start=index,
                end=index + 1,
                modality=modality,
                role="observation",
                source="llava.tokenizer_image_token",
                media_unit=f"{modality}_placeholder",
                encoder_origin="vision_encoder",
                alignment_group=f"{modality}-{media_index}",
            )
            for media_index, (modality, index) in enumerate(
                zip(media_modalities, media_positions)
            )
        ]
        return ForwardState(
            input_ids=outputs.get("input_ids"),
            attention_mask=outputs.get("attention_mask"),
            labels=outputs.get("labels"),
            pixel_values=outputs.get("images"),
            native_model_inputs=outputs,
            modality_token_spans=spans,
            modality_metadata=metadata,
            sample_ids=list(processed_batch.sample_ids),
        )


@dataclass(frozen=True)
class LlavaOneVisionLegalizer:
    """Runtime bridges own image/video layout mutation; generic actions stay closed."""

    profile: BackboneCapabilities = LLAVA_ONEVISION_IMAGE_PROFILE

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
                "LLaVA token mutation requires an exact source-bridge executor; "
                "generic post-processor legalization is not active."
            )
        return replace(state)


@dataclass(frozen=True)
class LlavaOneVisionRunner:
    processor_adapter: LlavaOneVisionProcessorAdapter
    native_validator: Any
    profile: BackboneCapabilities = LLAVA_ONEVISION_IMAGE_PROFILE

    def forward_train(self, runtime: BackboneRuntime, state: ForwardState) -> Any:
        session = runtime.metadata.get("_compression_session")
        if session is not None:
            session.clear_diagnostic_records()
        model_inputs = _runtime_model_inputs(runtime, state, generate=False)
        with llava_onevision_video_request_context(
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
            "LLaVA-OneVision native input validation failed",
            self.native_validator.validate_state(runtime, state, processed_batch),
        )
        if gen_kwargs.pop("return_audio", False):
            raise InvariantError("LLaVA-OneVision has no speech/Talker output branch.")
        if gen_kwargs.pop("stream", False) or gen_kwargs.get("streamer") is not None:
            raise NotActivatedError(
                "The LLaVA-OneVision baseline returns complete generation results."
            )
        request = trace_request or TraceRequest()
        unsupported = [
            name
            for name in (
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
            if getattr(request, name)
        ]
        if unsupported:
            raise InvariantError(
                "LLaVA-OneVision has no Thinker/Talker/Token2Wav branches; "
                f"unsupported traces: {unsupported}."
            )
        if request.text_hidden_states:
            gen_kwargs.setdefault("output_hidden_states", True)
        if request.text_attentions or request._full_generation_text_attentions:
            gen_kwargs.setdefault("output_attentions", True)
        if request.enabled():
            gen_kwargs.setdefault("return_dict_in_generate", True)
        if gen_kwargs.get("do_sample") is False:
            gen_kwargs.pop("temperature", None)
            gen_kwargs.pop("top_p", None)
            gen_kwargs.pop("top_k", None)
        model_inputs = _runtime_model_inputs(runtime, state, generate=True)
        with llava_onevision_video_request_context(
            runtime.model,
            state.modality_metadata,
        ):
            with sgl_target_generation(runtime, processed_batch, gen_kwargs):
                with _capture_text_traces(runtime.model, request) as traces:
                    raw_output = runtime.model.generate(**model_inputs, **gen_kwargs)
        if request._full_generation_text_attentions and traces is not None:
            generated_attentions = _output_field(raw_output, "attentions")
            if generated_attentions is not None:
                traces.text_attentions = generated_attentions
                traces.metadata.setdefault("field_capture_semantics", {})[
                    "text_attentions"
                ] = "full_generate_steps"
        sequences = getattr(raw_output, "sequences", raw_output)
        generated_ids = _strip_prompt_prefix(sequences, state.input_ids)
        decoded = self.processor_adapter.detokenize(runtime.processor, generated_ids)
        result = GenerationResult(
            text=TextGenerationOutput(token_ids=generated_ids, text=decoded),
            speech=None,
            traces=traces if request.enabled() else None,
            sample_ids=list(processed_batch.sample_ids),
            metadata={
                "provider": "vendored.LlavaQwenForCausalLM.generate",
                "source_route": LLAVA_ONEVISION_SOURCE_ROUTE,
                "input_modalities": list(
                    processed_batch.metadata.get("modalities", ())
                ),
                "image_sizes": list(processed_batch.metadata.get("image_sizes", ())),
                "return_audio": False,
                "video_supported_in_this_slice": True,
                "video_real_checkpoint_verified": False,
            },
        )
        _raise_violations(
            "LLaVA-OneVision native output validation failed",
            self.native_validator.validate_generation(
                runtime, state, processed_batch, result
            ),
        )
        return result


def create_llava_onevision_backbone() -> BackboneFacade:
    """Compose one facade sharing the source tuple loader across components."""

    from .validation import LlavaOneVisionNativeValidator

    profile = LLAVA_ONEVISION_IMAGE_PROFILE
    loader = LlavaOneVisionSourceLoader()
    processor_adapter = LlavaOneVisionProcessorAdapter(profile, loader)
    validator = LlavaOneVisionNativeValidator()
    return BackboneFacade(
        processor_adapter=processor_adapter,
        model_adapter=LlavaOneVisionModelAdapter(profile, loader),
        state_builder=LlavaOneVisionStateBuilder(profile),
        legalizer=LlavaOneVisionLegalizer(profile),
        runner=LlavaOneVisionRunner(processor_adapter, validator, profile),
        operation_applier=None,
        native_validator=validator,
    )


def _import_source_api() -> LlavaOneVisionSourceAPI:
    # lmms-eval's native llava_onevision module enables this before model load.
    # Keep the platform route numerically aligned with that execution path.
    try:
        import torch
    except ImportError:
        pass
    else:
        torch.backends.cuda.matmul.allow_tf32 = True

    source_root = (
        Path(__file__).resolve().parents[3] / "third_party" / "backbones" / "LLaVA-NeXT"
    )
    if not source_root.is_dir():
        raise BackendIntegrationError(
            f"Vendored LLaVA-NeXT source tree is missing at {source_root}."
        )
    source_text = str(source_root)
    if source_text not in sys.path:
        sys.path.insert(0, source_text)
    try:
        from llava.constants import (
            DEFAULT_IMAGE_TOKEN,
            DEFAULT_IM_END_TOKEN,
            DEFAULT_IM_START_TOKEN,
            IMAGE_TOKEN_INDEX,
        )
        from llava.conversation import conv_templates
        from llava.mm_utils import (
            get_model_name_from_path,
            process_images,
            tokenizer_image_token,
        )
        from llava.model.builder import load_pretrained_model
    except ImportError as exc:
        raise BackendIntegrationError(
            "Failed to import the vendored LLaVA-NeXT source tree. Install its "
            "declared runtime dependencies in the current lane environment."
        ) from exc
    return LlavaOneVisionSourceAPI(
        load_pretrained_model=load_pretrained_model,
        tokenizer_image_token=tokenizer_image_token,
        process_images=process_images,
        get_model_name_from_path=get_model_name_from_path,
        conv_templates=conv_templates,
        image_token_index=int(IMAGE_TOKEN_INDEX),
        default_image_token=str(DEFAULT_IMAGE_TOKEN),
        default_im_start_token=str(DEFAULT_IM_START_TOKEN),
        default_im_end_token=str(DEFAULT_IM_END_TOKEN),
    )


def _source_dtype_name(value: Any) -> str:
    name = str(value).lower().replace("torch.", "")
    aliases = {"fp16": "float16", "bf16": "bfloat16"}
    name = aliases.get(name, name)
    if name not in {"float16", "bfloat16"}:
        raise BackendIntegrationError(
            "The selected LLaVA source loader supports torch_dtype float16 or "
            f"bfloat16, got {value!r}."
        )
    return name


def _normalise_samples(raw_sample: Any) -> list[LlavaOneVisionSample]:
    if isinstance(raw_sample, LlavaOneVisionSample):
        return [raw_sample]
    if isinstance(raw_sample, str):
        return [LlavaOneVisionSample(sample_id="sample-0", prompt=raw_sample)]
    if isinstance(raw_sample, Mapping):
        _reject_audio_media(raw_sample)
        prompt = raw_sample.get("prompt", raw_sample.get("text"))
        if not isinstance(prompt, str) or not prompt.strip():
            raise BackendIntegrationError(
                "LLaVA sample mapping requires a non-empty prompt/text string."
            )
        images = raw_sample.get("images", raw_sample.get("image", ()))
        if images is None:
            images = ()
        elif isinstance(images, Sequence) and not isinstance(images, (str, bytes)):
            images = tuple(images)
        else:
            images = (images,)
        videos = raw_sample.get("videos", raw_sample.get("video", ()))
        if videos is None:
            videos = ()
        elif isinstance(videos, Sequence) and not isinstance(
            videos, (str, bytes, bytearray)
        ):
            videos = tuple(videos)
        else:
            videos = (videos,)
        if len(videos) > 1:
            raise BackendIntegrationError(
                "LLaVA-OneVision supports one video per sample."
            )
        if images and videos:
            raise BackendIntegrationError(
                "LLaVA-OneVision does not support mixed image/video samples."
            )
        requested_modalities = raw_sample.get("modalities")
        if isinstance(requested_modalities, str):
            requested_modalities = (requested_modalities,)
        if requested_modalities is not None:
            requested = tuple(str(item).lower() for item in requested_modalities)
            if "video" in requested and not videos:
                raise BackendIntegrationError(
                    "LLaVA video modality requires one video payload."
                )
        return [
            LlavaOneVisionSample(
                sample_id=raw_sample.get("sample_id", "sample-0"),
                prompt=prompt,
                images=tuple(images),
                video=videos[0] if videos else None,
                conversation_mode=str(
                    raw_sample.get("conversation_mode", DEFAULT_CONVERSATION_MODE)
                ),
            )
        ]
    if isinstance(raw_sample, Sequence) and not isinstance(raw_sample, (str, bytes)):
        result: list[LlavaOneVisionSample] = []
        for item in raw_sample:
            result.extend(_normalise_samples(item))
        return result
    raise BackendIntegrationError(
        "LLaVA raw_sample must be text, LlavaOneVisionSample, or a sample mapping."
    )


def _reject_audio_media(sample: Mapping[str, Any]) -> None:
    forbidden = [
        key for key in ("audio", "audios") if sample.get(key) not in (None, (), [])
    ]
    modalities = sample.get("modalities")
    if isinstance(modalities, str):
        modalities = (modalities,)
    if modalities is not None:
        forbidden.extend(
            str(item)
            for item in modalities
            if str(item).lower() not in {"text", "image", "video"}
        )
    if forbidden:
        raise BackendIntegrationError(
            "LLaVA-OneVision supports text/image/video -> text only; "
            f"audio is unsupported (found {forbidden})."
        )


def _load_image(value: Any) -> Any:
    if hasattr(value, "size"):
        convert = getattr(value, "convert", None)
        return convert("RGB") if callable(convert) else value
    if isinstance(value, (str, Path)):
        text = str(value)
        if text.startswith(("http://", "https://")):
            raise BackendIntegrationError(
                "Remote image fetching is not implicit; provide a decoded image object."
            )
        try:
            from PIL import Image
        except ImportError as exc:
            raise BackendIntegrationError("LLaVA path images require Pillow.") from exc
        return Image.open(text).convert("RGB")
    raise BackendIntegrationError(
        "LLaVA image inputs must be PIL-like objects or local filesystem paths."
    )


def _load_video(value: Any, max_frames: int) -> LlavaOneVisionDecodedVideo:
    if isinstance(value, LlavaOneVisionDecodedVideo):
        return value
    if not isinstance(value, (str, Path)):
        raise BackendIntegrationError(
            "LLaVA video inputs must be decoded video records or local paths."
        )
    text = str(value)
    if text.startswith(("http://", "https://")):
        raise BackendIntegrationError(
            "Remote video fetching is not implicit; provide a local video path."
        )
    path = Path(text).expanduser()
    if not path.is_file():
        raise BackendIntegrationError(f"LLaVA video path does not exist: {path}.")
    if (
        isinstance(max_frames, bool)
        or not isinstance(max_frames, int)
        or max_frames <= 0
    ):
        raise BackendIntegrationError("max_frames must be a positive integer.")
    try:
        import numpy as np
        from decord import VideoReader, cpu
    except ImportError as exc:
        raise BackendIntegrationError(
            "LLaVA local video decoding requires the existing decord and numpy packages."
        ) from exc
    try:
        reader = VideoReader(str(path), ctx=cpu(0), num_threads=1)
        total_frames = int(len(reader))
        if total_frames <= 0:
            raise ValueError("video has no frames")
        frame_indices = tuple(
            int(value)
            for value in np.linspace(
                0,
                total_frames - 1,
                max_frames,
                dtype=int,
            ).tolist()
        )
        frames = reader.get_batch(list(frame_indices)).asnumpy()
        fps = float(reader.get_avg_fps())
    except Exception as exc:
        raise BackendIntegrationError(
            f"Could not decode local LLaVA video {path}."
        ) from exc
    if getattr(frames, "ndim", None) != 4:
        raise BackendIntegrationError(
            "LLaVA video decoder did not return [frames, height, width, channels]."
        )
    height, width = int(frames.shape[1]), int(frames.shape[2])
    timestamps = tuple(
        (float(index) / fps if fps > 0 else 0.0) for index in frame_indices
    )
    return LlavaOneVisionDecodedVideo(
        frames=frames,
        frame_size=(width, height),
        frame_indices=frame_indices,
        timestamps_seconds=timestamps,
        source=str(path.resolve()),
        total_frames=total_frames,
        fps=fps if fps > 0 else None,
    )


def _leading_length(value: Any) -> int:
    shape = getattr(value, "shape", None)
    if shape is not None and len(shape) > 0:
        return int(shape[0])
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return len(value)
    raise ValueError("Decoded LLaVA video frames must be a non-scalar sequence")


def _image_size(image: Any) -> tuple[int, int]:
    size = getattr(image, "size", None)
    if not isinstance(size, Sequence) or len(size) != 2:
        raise BackendIntegrationError(
            "Decoded LLaVA image has no (width, height) size."
        )
    width, height = (int(item) for item in size)
    if width <= 0 or height <= 0:
        raise BackendIntegrationError("Decoded LLaVA image size must be positive.")
    return width, height


def _render_prompt(
    prompt: str,
    *,
    media_count: int,
    conversation_mode: str,
    api: LlavaOneVisionSourceAPI,
    model_config: Any,
    assistant_text: str | None = None,
) -> str:
    placeholder_count = prompt.count(api.default_image_token)
    if media_count and placeholder_count == 0:
        prompt = "\n".join(
            [*(api.default_image_token for _ in range(media_count)), prompt]
        )
        placeholder_count = media_count
    if placeholder_count != media_count:
        raise BackendIntegrationError(
            "LLaVA prompt must contain exactly one <image> placeholder per media item."
        )
    if media_count and bool(getattr(model_config, "mm_use_im_start_end", False)):
        wrapped = (
            api.default_im_start_token
            + api.default_image_token
            + api.default_im_end_token
        )
        prompt = prompt.replace(api.default_image_token, wrapped)
    conversation = api.conv_templates[conversation_mode].copy()
    conversation.append_message(conversation.roles[0], prompt)
    conversation.append_message(conversation.roles[1], assistant_text)
    return str(conversation.get_prompt())


def _ensure_batch_dimension(value: Any) -> Any:
    unsqueeze = getattr(value, "unsqueeze", None)
    if callable(unsqueeze):
        return value.unsqueeze(0)
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise BackendIntegrationError(
            "tokenizer_image_token returned non-sequence ids."
        )
    values = list(value)
    if values and isinstance(values[0], Sequence):
        return values
    return [values]


def _attention_mask(value: Any, *, pad_token_id: int | None) -> Any:
    try:
        import torch
    except ImportError:
        torch = None
    if torch is not None and isinstance(value, torch.Tensor):
        if pad_token_id is None:
            return torch.ones_like(value, dtype=torch.long)
        return value.ne(pad_token_id)
    if pad_token_id is None:
        return [[1 for _ in row] for row in value]
    return [[int(token != pad_token_id) for token in row] for row in value]


def _rows(value: Any) -> list[list[int]]:
    tolist = getattr(value, "tolist", None)
    raw = tolist() if callable(tolist) else value
    if raw is None:
        return []
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
        return [[int(raw)]]
    if not raw:
        return []
    if not isinstance(raw[0], Sequence):
        return [[int(item) for item in raw]]
    return [[int(item) for item in row] for row in raw]


def _row_sums(value: Any) -> list[int]:
    return [sum(row) for row in _rows(value)]


def _find_token_positions(value: Any, token_id: int) -> list[int]:
    rows = _rows(value)
    if len(rows) != 1:
        raise BackendIntegrationError("LLaVA source baseline requires one token row.")
    return [index for index, item in enumerate(rows[0]) if item == token_id]


def _vision_layers(vision_tower: Any) -> Any | None:
    candidates = (
        ("vision_tower", "vision_model", "encoder", "layers"),
        ("vision_model", "encoder", "layers"),
        ("encoder", "layers"),
        ("layers",),
    )
    for path in candidates:
        value = vision_tower
        for name in path:
            value = getattr(value, name, None)
            if value is None:
                break
        if value is not None:
            return value
    return None


def _move_value(value: Any, *, device: Any | None, dtype: Any | None) -> Any:
    to = getattr(value, "to", None)
    if not callable(to):
        if isinstance(value, list):
            return [_move_value(item, device=device, dtype=dtype) for item in value]
        return value
    if device is not None:
        value = value.to(device)
    is_floating_point = getattr(value, "is_floating_point", None)
    if dtype is not None and callable(is_floating_point) and is_floating_point():
        value = value.to(dtype=dtype)
    return value


def _runtime_model_inputs(
    runtime: BackboneRuntime,
    state: ForwardState,
    *,
    generate: bool,
) -> dict[str, Any]:
    values = dict(state.native_model_inputs)
    input_ids = values.pop("input_ids", state.input_ids)
    if generate:
        values["inputs"] = input_ids
        values.pop("labels", None)
        # Preserve native lmms-eval parity: image uses the model default, while
        # video must explicitly select the source tree's video pooling/packing.
        if tuple(values.get("modalities", ())) == ("image",):
            values.pop("modalities", None)
    else:
        values["input_ids"] = input_ids
    values = {
        name: _move_value(value, device=runtime.device, dtype=runtime.dtype)
        for name, value in values.items()
        if value is not None
    }
    return values


def _strip_prompt_prefix(sequences: Any, prompt_ids: Any) -> Any:
    output_rows = _rows(sequences)
    prompt_rows = _rows(prompt_ids)
    if len(output_rows) != len(prompt_rows):
        return sequences
    stripped = []
    changed = False
    for output, prompt in zip(output_rows, prompt_rows):
        if len(output) >= len(prompt) and output[: len(prompt)] == prompt:
            stripped.append(output[len(prompt) :])
            changed = True
        else:
            stripped.append(output)
    if not changed:
        return sequences
    # Generation output is small; normalizing a prompt-prefixed provider result
    # to Python rows is acceptable and keeps tokenizer.batch_decode portable.
    return stripped


@contextmanager
def _capture_text_traces(
    model: Any,
    request: TraceRequest,
) -> Iterator[TraceArtifacts | None]:
    if not request.enabled():
        yield None
        return
    artifacts = TraceArtifacts(
        metadata={
            "provider": "vendored-llava-forward-hook",
            "capture_semantics": "last_forward_step",
        }
    )
    handle = None
    register = getattr(model, "register_forward_hook", None)
    if callable(register):

        def hook(_module: Any, _args: Any, _kwargs: Any, output: Any) -> None:
            if request.text_logits:
                artifacts.text_logits = _output_field(output, "logits")
            if request.text_hidden_states:
                artifacts.text_hidden_states = _output_field(output, "hidden_states")
            if request.text_attentions:
                artifacts.text_attentions = _output_field(output, "attentions")
            if request.text_kv_cache:
                artifacts.text_kv_cache = _output_field(output, "past_key_values")

        handle = register(hook, with_kwargs=True)
    try:
        yield artifacts
    finally:
        if handle is not None:
            handle.remove()


def _output_field(output: Any, name: str) -> Any:
    if isinstance(output, Mapping):
        return output.get(name)
    return getattr(output, name, None)


def _raise_violations(prefix: str, violations: list[Any]) -> None:
    if not violations:
        return
    details = "; ".join(
        f"{item.name}[{item.field}]: expected={item.expected!r}, actual={item.actual!r}"
        for item in violations
    )
    raise InvariantError(f"{prefix}: {details}")


__all__ = [
    "DEFAULT_CONVERSATION_MODE",
    "LLAVA_ONEVISION_IMAGE_PROFILE",
    "LLAVA_ONEVISION_MAX_VIDEO_FRAMES",
    "LLAVA_ONEVISION_NATIVE_PROFILE",
    "LlavaOneVisionDecodedVideo",
    "LlavaOneVisionLegalizer",
    "LlavaOneVisionLoadedArtifacts",
    "LlavaOneVisionModelAdapter",
    "LlavaOneVisionProcessorAdapter",
    "LlavaOneVisionProcessorBundle",
    "LlavaOneVisionRunner",
    "LlavaOneVisionSample",
    "LlavaOneVisionSourceAPI",
    "LlavaOneVisionSourceLoader",
    "LlavaOneVisionStateBuilder",
    "create_llava_onevision_backbone",
]
