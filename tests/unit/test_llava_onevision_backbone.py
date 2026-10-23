"""CPU-only contracts for the vendored LLaVA-OneVision native facade."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from mllm_comp.backbones.base import BackboneFacade, BackboneRuntime
from mllm_comp.backbones.llava_onevision import (
    LLAVA_ONEVISION_IMAGE_PROFILE,
    LlavaOneVisionDecodedVideo,
    LlavaOneVisionLegalizer,
    LlavaOneVisionLoadedArtifacts,
    LlavaOneVisionModelAdapter,
    LlavaOneVisionProcessorAdapter,
    LlavaOneVisionProcessorBundle,
    LlavaOneVisionSample,
    LlavaOneVisionSourceAPI,
    LlavaOneVisionSourceLoader,
    create_llava_onevision_backbone,
)
from mllm_comp.core.actions import LegalizationPolicy, TokenAction
from mllm_comp.core.errors import BackendIntegrationError, NotActivatedError


class _Conversation:
    roles = ("user", "assistant")

    def __init__(self) -> None:
        self.messages: list[tuple[str, Any]] = []

    def copy(self) -> "_Conversation":
        return _Conversation()

    def append_message(self, role: str, value: Any) -> None:
        self.messages.append((role, value))

    def get_prompt(self) -> str:
        return " | ".join(f"{role}:{value}" for role, value in self.messages)


class _Tokenizer:
    all_special_ids = [99]

    def batch_decode(self, rows: Any, **kwargs: Any) -> list[str]:
        assert kwargs["skip_special_tokens"] is True
        return [" ".join(str(item) for item in row) for row in rows]


class _Image:
    def __init__(self, size: tuple[int, int]) -> None:
        self.size = size

    def convert(self, mode: str) -> "_Image":
        assert mode == "RGB"
        return self


class _ImageProcessor:
    def __init__(self) -> None:
        self.video_calls: list[dict[str, Any]] = []

    def preprocess(self, frames: Any, *, return_tensors: str) -> dict[str, Any]:
        self.video_calls.append({"frames": frames, "return_tensors": return_tensors})
        return {"pixel_values": "video-pixel-values"}


class _Model:
    def __init__(self) -> None:
        self.config = SimpleNamespace(
            model_type="llava_qwen",
            image_aspect_ratio="anyres",
            mm_patch_merge_type="spatial_unpad",
            mm_newline_position="one_token",
            image_grid_pinpoints=[[384, 384], [768, 384]],
            mm_use_im_start_end=False,
        )
        self.train_mode: bool | None = None

    def train(self, *, mode: bool) -> None:
        self.train_mode = mode


def _api(
    *,
    load_pretrained_model: Any | None = None,
    process_calls: list[dict[str, Any]] | None = None,
) -> LlavaOneVisionSourceAPI:
    def tokenizer_image_token(
        prompt: str,
        tokenizer: Any,
        image_token_index: int,
        return_tensors: str,
    ) -> list[int]:
        del tokenizer
        assert return_tensors == "pt"
        return [
            1,
            *(
                [image_token_index, index + 10]
                for index in range(prompt.count("<image>"))
            ),
        ]

    # Flatten the small fixture token list constructed above.
    def tokenize(
        prompt: str, tokenizer: Any, image_token_index: int, return_tensors: str
    ) -> list[int]:
        raw = tokenizer_image_token(
            prompt, tokenizer, image_token_index, return_tensors
        )
        result: list[int] = []
        for item in raw:
            result.extend(item if isinstance(item, list) else [item])
        result.append(2)
        if "assistant:None" not in prompt:
            result.extend([20, 99])
        return result

    def process_images(images: Any, image_processor: Any, model_config: Any) -> str:
        if process_calls is not None:
            process_calls.append(
                {
                    "images": images,
                    "image_processor": image_processor,
                    "model_config": model_config,
                }
            )
        return "pixel-values"

    return LlavaOneVisionSourceAPI(
        load_pretrained_model=load_pretrained_model or (lambda *_args, **_kwargs: None),
        tokenizer_image_token=tokenize,
        process_images=process_images,
        get_model_name_from_path=lambda path: path.rsplit("/", 1)[-1],
        conv_templates={"qwen_1_5": _Conversation()},
        image_token_index=-200,
        default_image_token="<image>",
        default_im_start_token="<im_start>",
        default_im_end_token="<im_end>",
    )


def _processor_fixture() -> tuple[
    LlavaOneVisionProcessorAdapter,
    LlavaOneVisionProcessorBundle,
    list[dict[str, Any]],
]:
    process_calls: list[dict[str, Any]] = []
    source_loader = LlavaOneVisionSourceLoader(
        api_loader=lambda: _api(process_calls=process_calls)
    )
    model = _Model()
    artifacts = LlavaOneVisionLoadedArtifacts(
        tokenizer=_Tokenizer(),
        model=model,
        image_processor=_ImageProcessor(),
        context_length=32768,
        model_name="llava-onevision-qwen2-0.5b-ov",
        model_path_or_id="fixture",
        revision=None,
        load_signature={},
    )
    bundle = LlavaOneVisionProcessorBundle(
        model_path_or_id="fixture",
        source_loader=source_loader,
        loaded_artifacts=artifacts,
    )
    adapter = LlavaOneVisionProcessorAdapter(
        source_loader=source_loader,
        image_loader=lambda image: image,
    )
    return adapter, bundle, process_calls


def test_facade_declares_native_image_and_video_without_support_activation() -> None:
    facade = create_llava_onevision_backbone()

    assert isinstance(facade, BackboneFacade)
    assert facade.capabilities() == LLAVA_ONEVISION_IMAGE_PROFILE
    assert facade.capabilities().supports_image is True
    assert facade.capabilities().supports_video is True
    assert facade.capabilities().supports_audio is False
    assert facade.capabilities().output_modalities == ("text",)
    assert facade.operation_applier is None


def test_processor_state_and_validator_preserve_anyres_image_identity() -> None:
    processor_adapter, processor, process_calls = _processor_fixture()
    sample = LlavaOneVisionSample(
        sample_id="image-sample",
        prompt="Compare these images.",
        images=(_Image((640, 480)), _Image((320, 240))),
    )
    batch = processor_adapter.prepare_inputs(processor, sample, phase="eval")
    facade = create_llava_onevision_backbone()
    state = facade.state_builder.build(batch, phase="eval")

    assert batch.model_inputs["images"] == "pixel-values"
    assert batch.model_inputs["image_sizes"] == [(640, 480), (320, 240)]
    assert batch.model_inputs["modalities"] == ("image",)
    assert batch.metadata["image_placeholder_positions"] == (1, 3)
    assert batch.metadata["anyres"] == {
        "image_aspect_ratio": "anyres",
        "mm_patch_merge_type": "spatial_unpad",
        "mm_newline_position": "one_token",
        "image_grid_pinpoints": ([384, 384], [768, 384]),
        "image_newline_required": True,
        "layout_phase": "pre_encoder_pre_anyres_pack",
    }
    assert len(process_calls) == 1
    assert process_calls[0]["images"] == sample.images
    assert state.pixel_values is batch.model_inputs["images"]
    assert [span.start for span in state.modality_token_spans] == [1, 3]
    assert [span.alignment_group for span in state.modality_token_spans] == [
        "image-0",
        "image-1",
    ]
    assert facade.native_validator is not None
    assert (
        facade.native_validator.validate_state(
            BackboneRuntime(model=object(), processor=processor),
            state,
            batch,
        )
        == []
    )


def test_processor_builds_completion_only_teacher_forced_labels() -> None:
    processor_adapter, processor, _process_calls = _processor_fixture()
    sample = LlavaOneVisionSample(
        sample_id="teacher-forced",
        prompt="What is shown?",
        images=(_Image((640, 480)),),
    )
    prompt = processor_adapter.prepare_inputs(processor, sample, phase="eval")

    result = processor_adapter.prepare_teacher_forced_inputs(
        processor,
        sample,
        target_text="answer",
        prompt_batch=prompt,
        phase="eval",
    )

    labels = result.model_inputs["labels"][0]
    assert labels[: len(prompt.model_inputs["input_ids"][0])] == [
        -100
    ] * len(prompt.model_inputs["input_ids"][0])
    assert labels[-2:] == [20, -100]
    assert result.metadata["teacher_forcing"]["answer_token_counts"] == [1]


def test_processor_preserves_native_single_video_identity_and_timing() -> None:
    processor_adapter, processor, process_calls = _processor_fixture()
    decoded = LlavaOneVisionDecodedVideo(
        frames=("frame-0", "frame-1"),
        frame_size=(640, 360),
        frame_indices=(0, 9),
        timestamps_seconds=(0.0, 0.9),
        source="fixture.mp4",
        total_frames=10,
        fps=10.0,
    )
    sample = LlavaOneVisionSample(
        sample_id="video-sample",
        prompt="What happens?",
        video=decoded,
    )

    batch = processor_adapter.prepare_inputs(processor, sample, phase="eval")
    facade = create_llava_onevision_backbone()
    state = facade.state_builder.build(batch, phase="eval")

    assert process_calls == []
    assert batch.model_inputs["images"] == ["video-pixel-values"]
    assert batch.model_inputs["image_sizes"] == [(640, 360)]
    assert batch.model_inputs["modalities"] == ("video",)
    assert batch.metadata["image_count"] == 0
    assert batch.metadata["video_count"] == 1
    assert batch.metadata["video_placeholder_positions"] == (1,)
    assert batch.metadata["video"]["frame_indices"] == (0, 9)
    assert batch.metadata["video"]["timestamps_seconds"] == (0.0, 0.9)
    assert [span.modality for span in state.modality_token_spans] == ["video"]
    assert [span.alignment_group for span in state.modality_token_spans] == ["video-0"]
    assert (
        facade.native_validator.validate_state(
            BackboneRuntime(model=object(), processor=processor),
            state,
            batch,
        )
        == []
    )


def test_processor_rejects_audio_and_multi_sample_before_media_work() -> None:
    processor_adapter, processor, process_calls = _processor_fixture()
    with pytest.raises(BackendIntegrationError, match="audio is unsupported"):
        processor_adapter.prepare_inputs(
            processor,
            {"prompt": "audio", "modalities": ["audio"]},
            phase="eval",
        )
    with pytest.raises(BackendIntegrationError, match="single-sample"):
        processor_adapter.prepare_inputs(
            processor,
            ["first", "second"],
            phase="eval",
        )
    assert process_calls == []


def test_shared_source_loader_calls_load_pretrained_model_once_and_normalizes_dtype() -> (
    None
):
    calls: list[dict[str, Any]] = []
    model = _Model()

    def load_pretrained_model(*args: Any, **kwargs: Any) -> Any:
        calls.append({"args": args, "kwargs": kwargs})
        return _Tokenizer(), model, object(), 8192

    source_loader = LlavaOneVisionSourceLoader(
        api_loader=lambda: _api(load_pretrained_model=load_pretrained_model)
    )
    processor_adapter = LlavaOneVisionProcessorAdapter(source_loader=source_loader)
    model_adapter = LlavaOneVisionModelAdapter(source_loader=source_loader)
    processor = processor_adapter.load_processor("org/checkpoint")
    assert calls == []

    loaded_model = model_adapter.load_model(
        "org/checkpoint",
        train=False,
        torch_dtype="torch.bfloat16",
        device_map="cuda:0",
    )
    assert loaded_model is model
    assert model.train_mode is False
    assert processor.artifacts().model is model
    assert len(calls) == 1
    assert calls[0]["args"] == (
        "org/checkpoint",
        None,
        "checkpoint",
    )
    assert calls[0]["kwargs"]["torch_dtype"] == "bfloat16"
    assert calls[0]["kwargs"]["device_map"] == "cuda:0"
    assert calls[0]["kwargs"]["multimodal"] is True
    assert calls[0]["kwargs"]["attn_implementation"] == "sdpa"
    assert calls[0]["kwargs"]["overwrite_config"] == {
        "mm_spatial_pool_stride": 2,
        "mm_spatial_pool_mode": "bilinear",
    }


def test_pinned_llava_revision_requires_vision_tower_revision() -> None:
    source_loader = LlavaOneVisionSourceLoader(
        api_loader=lambda: _api(load_pretrained_model=lambda *args, **kwargs: None),
    )
    with pytest.raises(BackendIntegrationError, match="vision_tower_revision"):
        source_loader.load("org/checkpoint", revision="fixed-sha")


def test_pinned_llava_vision_tower_resolves_and_records_immutable_source(
    tmp_path: Path,
) -> None:
    load_calls: list[dict[str, Any]] = []
    snapshot_calls: list[tuple[str, str]] = []
    model = _Model()
    checkpoint_snapshot = tmp_path / "checkpoint"
    checkpoint_snapshot.mkdir()
    (checkpoint_snapshot / "config.json").write_text(
        json.dumps({"mm_vision_tower": "google/siglip-so400m-patch14-384"}),
        encoding="utf-8",
    )

    def load_pretrained_model(*args: Any, **kwargs: Any) -> Any:
        load_calls.append({"args": args, "kwargs": kwargs})
        return _Tokenizer(), model, object(), 8192

    def resolve_snapshot(repo_id: str, revision: str) -> str:
        snapshot_calls.append((repo_id, revision))
        if repo_id == "org/checkpoint":
            return str(checkpoint_snapshot)
        assert repo_id == "google/siglip-so400m-patch14-384"
        return "D:/hf-cache/immutable-siglip-snapshot"

    source_loader = LlavaOneVisionSourceLoader(
        api_loader=lambda: _api(load_pretrained_model=load_pretrained_model),
        snapshot_resolver=resolve_snapshot,
    )
    artifacts = source_loader.load(
        "org/checkpoint",
        revision="checkpoint-sha",
        vision_tower_revision="vision-sha",
    )

    assert snapshot_calls == [
        ("org/checkpoint", "checkpoint-sha"),
        ("google/siglip-so400m-patch14-384", "vision-sha"),
    ]
    assert load_calls[0]["kwargs"]["overwrite_config"]["mm_vision_tower"] == (
        "D:/hf-cache/immutable-siglip-snapshot"
    )
    expected_vision_source = {
        "model_id": "google/siglip-so400m-patch14-384",
        "revision": "vision-sha",
        "resolved_source": "D:/hf-cache/immutable-siglip-snapshot",
    }
    assert artifacts.load_signature["vision_tower"] == expected_vision_source
    assert model._mllm_comp_source_evidence["dependencies"]["vision_tower"] == (
        expected_vision_source
    )
    with pytest.raises(BackendIntegrationError, match="already loaded"):
        source_loader.load(
            "org/checkpoint",
            revision="checkpoint-sha",
            vision_tower_revision="different-vision-sha",
        )
    assert len(load_calls) == 1


def test_llava_source_loader_rejects_pin_after_unpinned_cache_entry() -> None:
    load_calls: list[dict[str, Any]] = []

    def load_pretrained_model(*args: Any, **kwargs: Any) -> Any:
        load_calls.append({"args": args, "kwargs": kwargs})
        return _Tokenizer(), _Model(), object(), 8192

    source_loader = LlavaOneVisionSourceLoader(
        api_loader=lambda: _api(load_pretrained_model=load_pretrained_model)
    )
    source_loader.load("org/checkpoint")

    with pytest.raises(BackendIntegrationError, match="already loaded"):
        source_loader.load(
            "org/checkpoint",
            vision_tower_revision="vision-sha",
        )
    assert len(load_calls) == 1


def test_llava_source_loader_does_not_retain_auth_token() -> None:
    load_calls: list[dict[str, Any]] = []

    def load_pretrained_model(*args: Any, **kwargs: Any) -> Any:
        load_calls.append({"args": args, "kwargs": kwargs})
        return _Tokenizer(), _Model(), object(), 8192

    source_loader = LlavaOneVisionSourceLoader(
        api_loader=lambda: _api(load_pretrained_model=load_pretrained_model)
    )
    artifacts = source_loader.load("org/checkpoint", token="do-not-retain")

    assert load_calls[0]["kwargs"]["token"] == "do-not-retain"
    assert "token" not in artifacts.load_signature
    assert "do-not-retain" not in repr(artifacts)


def test_pinned_llava_rejects_non_object_checkpoint_config(
    tmp_path: Path,
) -> None:
    checkpoint_snapshot = tmp_path / "checkpoint"
    checkpoint_snapshot.mkdir()
    (checkpoint_snapshot / "config.json").write_text("[]", encoding="utf-8")

    source_loader = LlavaOneVisionSourceLoader(
        api_loader=lambda: _api(load_pretrained_model=lambda *args, **kwargs: None),
        snapshot_resolver=lambda repo_id, revision: str(checkpoint_snapshot),
    )
    with pytest.raises(BackendIntegrationError, match="JSON object"):
        source_loader.load(
            "org/checkpoint",
            revision="checkpoint-sha",
            vision_tower_revision="vision-sha",
        )


def test_pinned_llava_wraps_invalid_vision_tower_revision(
    tmp_path: Path,
) -> None:
    checkpoint_snapshot = tmp_path / "checkpoint"
    checkpoint_snapshot.mkdir()
    (checkpoint_snapshot / "config.json").write_text(
        json.dumps({"mm_vision_tower": "google/siglip"}),
        encoding="utf-8",
    )

    def resolve_snapshot(repo_id: str, revision: str) -> str:
        if repo_id == "org/checkpoint":
            return str(checkpoint_snapshot)
        raise OSError(f"missing {repo_id}@{revision}")

    source_loader = LlavaOneVisionSourceLoader(
        api_loader=lambda: _api(load_pretrained_model=lambda *args, **kwargs: None),
        snapshot_resolver=resolve_snapshot,
    )
    with pytest.raises(BackendIntegrationError, match="Could not resolve"):
        source_loader.load(
            "org/checkpoint",
            revision="checkpoint-sha",
            vision_tower_revision="invalid-vision-sha",
        )


def test_structural_handles_cover_vision_projector_newline_and_language_layers() -> (
    None
):
    vision_layer = SimpleNamespace(self_attn=object(), mlp=object())
    vision_tower = SimpleNamespace(
        vision_tower=SimpleNamespace(
            vision_model=SimpleNamespace(encoder=SimpleNamespace(layers=[vision_layer]))
        )
    )
    llm_layer = SimpleNamespace(self_attn=object(), mlp=object())
    projector = object()
    newline = object()
    final_norm = object()
    core = SimpleNamespace(
        layers=[llm_layer],
        norm=final_norm,
        mm_projector=projector,
        image_newline=newline,
    )
    model = SimpleNamespace(
        get_model=lambda: core,
        get_vision_tower=lambda: vision_tower,
        lm_head=object(),
        prepare_inputs_labels_for_multimodal=lambda *args: args,
    )

    handles = LlavaOneVisionModelAdapter().structural_handles(model)

    assert handles["llm"] is core
    assert handles["llm.final_norm"] is final_norm
    assert handles["vision_encoder"] is vision_tower
    assert handles["vision_encoder.layers"] == [vision_layer]
    assert handles["vision_projector"] is projector
    assert handles["image_newline"] is newline
    assert handles["vision_encoder.layer[0]"] is vision_layer
    assert handles["llm.layer[0]"] is llm_layer
    assert callable(handles["anyres_pre_llm_splice"])


def test_runner_uses_llava_native_generate_kwargs_and_returns_only_new_tokens() -> None:
    processor_adapter, processor, _process_calls = _processor_fixture()
    batch = processor_adapter.prepare_inputs(
        processor,
        LlavaOneVisionSample(
            sample_id="generate",
            prompt="Describe.",
            images=(_Image((640, 480)),),
        ),
        phase="eval",
    )
    facade = create_llava_onevision_backbone()
    state = facade.state_builder.build(batch, phase="eval")

    class _GenerateModel:
        kwargs: dict[str, Any] | None = None

        def generate(self, **kwargs: Any) -> list[list[int]]:
            self.kwargs = kwargs
            return [list(kwargs["inputs"][0]) + [41, 42]]

    model = _GenerateModel()
    result = facade.runner.generate(
        BackboneRuntime(model=model, processor=processor),
        state,
        processed_batch=batch,
        do_sample=False,
        max_new_tokens=2,
    )

    assert model.kwargs is not None
    assert model.kwargs["images"] == "pixel-values"
    assert model.kwargs["image_sizes"] == [(640, 480)]
    assert "modalities" not in model.kwargs
    assert "temperature" not in model.kwargs
    assert result.text is not None
    assert result.text.token_ids == [[41, 42]]
    assert result.text.text == ["41 42"]
    assert result.speech is None
    assert result.metadata["video_supported_in_this_slice"] is True


def test_runner_preserves_explicit_native_video_modality() -> None:
    processor_adapter, processor, _process_calls = _processor_fixture()
    batch = processor_adapter.prepare_inputs(
        processor,
        LlavaOneVisionSample(
            sample_id="video-generate",
            prompt="Describe.",
            video=LlavaOneVisionDecodedVideo(
                frames=("frame-0", "frame-1"),
                frame_size=(640, 360),
                frame_indices=(0, 10),
                timestamps_seconds=(0.0, 1.0),
            ),
        ),
        phase="eval",
    )
    facade = create_llava_onevision_backbone()
    state = facade.state_builder.build(batch, phase="eval")

    class _GenerateModel:
        kwargs: dict[str, Any] | None = None

        def generate(self, **kwargs: Any) -> list[list[int]]:
            self.kwargs = kwargs
            return [list(kwargs["inputs"][0]) + [42]]

    model = _GenerateModel()
    result = facade.runner.generate(
        BackboneRuntime(model=model, processor=processor),
        state,
        processed_batch=batch,
        do_sample=False,
        max_new_tokens=1,
    )

    assert model.kwargs is not None
    assert model.kwargs["modalities"] == ("video",)
    assert model.kwargs["images"] == ["video-pixel-values"]
    assert result.metadata["input_modalities"] == ["video"]


def test_legalizer_rejects_generic_token_mutation_outside_source_bridge() -> None:
    processor_adapter, processor, _ = _processor_fixture()
    batch = processor_adapter.prepare_inputs(processor, "text only", phase="eval")
    state = create_llava_onevision_backbone().state_builder.build(batch, phase="eval")
    with pytest.raises(NotActivatedError, match="source-bridge executor"):
        LlavaOneVisionLegalizer().apply(
            state,
            token_actions=[TokenAction(op="drop", target_indices=[0])],
            policy=LegalizationPolicy(
                position_policy="backbone_default",
                mask_policy="backbone_default",
                label_policy="backbone_default",
            ),
            stage_name="fixture",
        )
