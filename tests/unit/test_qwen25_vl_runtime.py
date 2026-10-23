"""CPU-only contracts for the transformers-native Qwen2.5-VL facade."""

from __future__ import annotations

import sys
from contextlib import contextmanager
from types import SimpleNamespace
from typing import Any

import pytest

from mllm_comp.backbones.base import BackboneRuntime, TraceRequest
from mllm_comp.backbones.qwen25_vl import (
    QWEN25_VL_IMAGE_PATCH_SIZE,
    Qwen25VLModelAdapter,
    Qwen25VLProcessorAdapter,
    Qwen25VLSample,
    create_qwen25_vl_backbone,
)
from mllm_comp.core.actions import LegalizationPolicy, TokenAction
from mllm_comp.core.errors import (
    BackendIntegrationError,
    InvariantError,
    NotActivatedError,
)


class _FakeTokenizer:
    _ids = {"<|image_pad|>": 10, "<|video_pad|>": 11}
    all_special_ids = [99]

    def convert_tokens_to_ids(self, token: str) -> int:
        return self._ids[token]


class _FakeProcessor:
    image_token = "<|image_pad|>"
    video_token = "<|video_pad|>"

    def __init__(self, outputs: dict[str, Any]) -> None:
        self.outputs = outputs
        self.tokenizer = _FakeTokenizer()
        self.image_processor = SimpleNamespace(
            merge_size=2,
            spatial_merge_size=2,
            patch_size=14,
        )
        self.processor_call: dict[str, Any] | None = None
        self.template_call: dict[str, Any] | None = None

    def apply_chat_template(self, conversation: Any, **kwargs: Any) -> str:
        self.template_call = {"conversation": conversation, **kwargs}
        return "formatted"

    def __call__(self, **kwargs: Any) -> dict[str, Any]:
        self.processor_call = kwargs
        return self.outputs

    def batch_decode(self, token_ids: Any, **kwargs: Any) -> list[str]:
        del kwargs
        return [" ".join(str(token) for token in row) for row in token_ids]


def _conversation(*content: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": list(content)},
    ]


def _all_modality_batch() -> tuple[Any, Any, Any, list[dict[str, Any]]]:
    input_ids = [[1, *([10] * 4), *([11] * 8), 2]]
    outputs = {
        "input_ids": input_ids,
        "attention_mask": [[1] * len(input_ids[0])],
        "pixel_values": [[0.0]],
        "image_grid_thw": [[1, 4, 4]],
        "pixel_values_videos": [[0.0]],
        "video_grid_thw": [[2, 4, 4]],
        "second_per_grid_ts": [0.5],
        "mm_token_type_ids": [[0, *([1] * 4), *([2] * 8), 0]],
    }
    processor = _FakeProcessor(outputs)
    loader_calls: list[dict[str, Any]] = []

    def media_loader(conversation: Any) -> tuple[Any, Any]:
        loader_calls.append({"conversation": conversation})
        return ["image-tensor"], ["video-tensor"]

    adapter = Qwen25VLProcessorAdapter(media_loader=media_loader)
    sample = Qwen25VLSample(
        sample_id="all-modalities",
        conversation=_conversation(
            {"type": "image", "image": "image.png"},
            {"type": "video", "video": "video.mp4"},
            {"type": "text", "text": "Describe both."},
        ),
    )
    batch = adapter.prepare_inputs(processor, sample, phase="eval")
    return adapter, processor, batch, loader_calls


def test_prepare_inputs_uses_qwen25_binary_media_contract_and_mrope_fields() -> None:
    _adapter, processor, batch, loader_calls = _all_modality_batch()
    facade = create_qwen25_vl_backbone()
    state = facade.state_builder.build(batch, phase="eval")
    runtime = BackboneRuntime(model=object(), processor=processor)

    assert len(loader_calls) == 1
    assert loader_calls[0] == {"conversation": processor.template_call["conversation"]}
    assert processor.template_call is not None
    assert "enable_thinking" not in processor.template_call
    assert processor.processor_call is not None
    assert processor.processor_call == {
        "text": "formatted",
        "images": ["image-tensor"],
        "videos": ["video-tensor"],
        "return_tensors": "pt",
        "padding": True,
    }
    assert state.video_pixel_values == [[0.0]]
    assert state.video_second_per_grid == [0.5]
    assert state.token_type_mask == batch.model_inputs["mm_token_type_ids"]
    assert [
        (span.modality, span.end - span.start)
        for span in state.modality_token_spans
    ] == [("image", 4), ("video", 8)]
    assert batch.metadata["image_patch_size"] == QWEN25_VL_IMAGE_PATCH_SIZE == 14
    assert batch.metadata["spatial_merge_size"] == 2
    assert batch.metadata["media_loader_return_arity"] == 2
    assert batch.metadata["expected_token_counts_by_sample"] == [
        {"image": 4, "video": 8}
    ]
    assert batch.metadata["video_media_ids"] == ("all-modalities:video-0",)
    assert batch.metadata["video_count"] == 1
    assert facade.capabilities().name == "qwen25_vl"
    assert facade.native_validator is not None
    assert facade.native_validator.validate_state(runtime, state, batch) == []


def test_processor_builds_native_teacher_forced_answer_labels() -> None:
    class _TeacherProcessor(_FakeProcessor):
        def __init__(self) -> None:
            super().__init__({})

        def apply_chat_template(self, conversation: Any, **kwargs: Any) -> str:
            self.template_call = {"conversation": conversation, **kwargs}
            return "full" if conversation[-1]["role"] == "assistant" else "prompt"

        def __call__(self, **kwargs: Any) -> dict[str, Any]:
            self.processor_call = kwargs
            ids = [1, 2, 20, 99] if kwargs["text"] == "full" else [1, 2]
            return {"input_ids": [ids], "attention_mask": [[1] * len(ids)]}

    processor = _TeacherProcessor()
    adapter = Qwen25VLProcessorAdapter(
        media_loader=lambda _conversation: (None, None)
    )
    sample = Qwen25VLSample(
        "teacher-forced",
        _conversation({"type": "text", "text": "question"}),
    )
    prompt = adapter.prepare_inputs(processor, sample, phase="eval")

    result = adapter.prepare_teacher_forced_inputs(
        processor,
        sample,
        target_text="answer",
        prompt_batch=prompt,
        phase="eval",
    )

    assert result.model_inputs["labels"] == [[-100, -100, 20, -100]]
    assert result.metadata["teacher_forcing"]["answer_token_counts"] == [1]
    assert processor.template_call["add_generation_prompt"] is False
    assert processor.template_call["conversation"][-1] == {
        "role": "assistant",
        "content": [{"type": "text", "text": "answer"}],
    }


def test_processor_rejects_audio_before_media_loading_and_rejects_triple_return() -> None:
    processor = _FakeProcessor({"input_ids": [[1]], "attention_mask": [[1]]})
    loader_calls: list[Any] = []

    def loader(value: Any) -> tuple[Any, Any]:
        loader_calls.append(value)
        return None, None

    adapter = Qwen25VLProcessorAdapter(media_loader=loader)
    with pytest.raises(BackendIntegrationError, match="does not accept audio"):
        adapter.prepare_inputs(
            processor,
            Qwen25VLSample(
                "audio",
                _conversation({"type": "audio", "audio": "sample.wav"}),
            ),
            phase="eval",
        )
    assert loader_calls == []

    adapter = Qwen25VLProcessorAdapter(
        media_loader=lambda _messages: (None, None, {"fps": 2.0})
    )
    with pytest.raises(
        BackendIntegrationError,
        match=r"must return \(images, videos\)",
    ):
        adapter.prepare_inputs(
            processor,
            Qwen25VLSample(
                "text",
                _conversation({"type": "text", "text": "hello"}),
            ),
            phase="eval",
        )


def test_processor_batch_partitions_image_and_video_grids_by_sample() -> None:
    outputs = {
        "input_ids": [
            [1, 10, 10, 10, 10, 2, 0, 0, 0, 0],
            [1, 11, 11, 11, 11, 11, 11, 11, 11, 2],
        ],
        "attention_mask": [
            [1, 1, 1, 1, 1, 1, 0, 0, 0, 0],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        ],
        "pixel_values": [[0.0]],
        "image_grid_thw": [[1, 4, 4]],
        "pixel_values_videos": [[0.0]],
        "video_grid_thw": [[2, 4, 4]],
        "second_per_grid_ts": [0.5],
    }
    processor = _FakeProcessor(outputs)
    adapter = Qwen25VLProcessorAdapter(
        media_loader=lambda _messages: (["image"], ["video"])
    )
    samples = [
        Qwen25VLSample(
            "image-sample",
            _conversation(
                {"type": "image", "image": "image.png"},
                {"type": "text", "text": "image"},
            ),
        ),
        Qwen25VLSample(
            "video-sample",
            _conversation(
                {"type": "video", "video": "video.mp4"},
                {"type": "text", "text": "video"},
            ),
        ),
    ]

    batch = adapter.prepare_inputs(processor, samples, phase="eval")
    facade = create_qwen25_vl_backbone()
    state = facade.state_builder.build(batch, phase="eval")

    assert batch.sample_ids == ["image-sample", "video-sample"]
    assert batch.prompt_lengths == [6, 10]
    assert batch.metadata["expected_token_counts_by_sample"] == [
        {"image": 4, "video": 0},
        {"image": 0, "video": 8},
    ]
    assert batch.metadata["video_media_ids"] == ("video-sample:video-0",)
    assert [
        (span.sample_index, span.modality, span.end - span.start)
        for span in state.modality_token_spans
    ] == [(0, "image", 4), (1, "video", 8)]
    assert facade.native_validator is not None
    assert facade.native_validator.validate_state(
        BackboneRuntime(model=object(), processor=processor),
        state,
        batch,
    ) == []


def test_processor_rejects_malformed_or_nondivisible_visual_grid() -> None:
    sample = Qwen25VLSample(
        "bad-grid",
        _conversation(
            {"type": "image", "image": "image.png"},
            {"type": "text", "text": "image"},
        ),
    )
    adapter = Qwen25VLProcessorAdapter(
        media_loader=lambda _messages: (["image"], None)
    )

    malformed = _FakeProcessor(
        {
            "input_ids": [[1, 10, 2]],
            "attention_mask": [[1, 1, 1]],
            "pixel_values": [[0.0]],
            "image_grid_thw": [[1, 4]],
        }
    )
    with pytest.raises(ValueError, match="three positive integers"):
        adapter.prepare_inputs(malformed, sample, phase="eval")

    nondivisible = _FakeProcessor(
        {
            "input_ids": [[1, 10, 2]],
            "attention_mask": [[1, 1, 1]],
            "pixel_values": [[0.0]],
            "image_grid_thw": [[1, 3, 3]],
        }
    )
    with pytest.raises(ValueError, match="not divisible"):
        adapter.prepare_inputs(nondivisible, sample, phase="eval")


def test_native_validator_checks_video_timing_and_three_axis_mrope_shape() -> None:
    _adapter, processor, batch, _loader_calls = _all_modality_batch()
    facade = create_qwen25_vl_backbone()
    state = facade.state_builder.build(batch, phase="eval")
    runtime = BackboneRuntime(model=object(), processor=processor)
    assert facade.native_validator is not None

    state.native_model_inputs["second_per_grid_ts"] = [0.5, 1.0]
    state.position_ids = [[0] * len(batch.model_inputs["input_ids"][0])] * 2
    violations = facade.native_validator.validate_state(runtime, state, batch)

    assert {violation.name for violation in violations} >= {
        "qwen25_vl_video_mrope_timing",
        "qwen25_vl_mrope_position_shape",
    }


def test_runner_supports_text_image_video_to_text_and_rejects_audio_output() -> None:
    _adapter, processor, batch, _loader_calls = _all_modality_batch()
    facade = create_qwen25_vl_backbone()
    state = facade.state_builder.build(batch, phase="eval")

    class _Bridge:
        def __init__(self) -> None:
            self.active = False
            self.records: list[dict[str, Any]] = []

        @contextmanager
        def request_context(self, metadata: Any) -> Any:
            self.records.append(dict(metadata))
            self.active = True
            try:
                yield
            finally:
                self.active = False

    bridge = _Bridge()

    class _Model:
        generation_kwargs: dict[str, Any] | None = None

        def generate(self, **kwargs: Any) -> list[list[int]]:
            assert bridge.active
            self.generation_kwargs = kwargs
            return [list(kwargs["input_ids"][0]) + [21, 22]]

    model = _Model()
    model._mllm_comp_qwen25_vl_video_bridge = bridge
    runtime = BackboneRuntime(model=model, processor=processor)
    result = facade.runner.generate(
        runtime,
        state,
        processed_batch=batch,
        max_new_tokens=2,
        do_sample=False,
    )

    assert result.text is not None
    assert result.text.token_ids == [[21, 22]]
    assert result.text.text == ["21 22"]
    assert result.speech is None
    assert result.metadata["input_modalities"] == ["text", "image", "video"]
    assert result.metadata["output_modality"] == "text"
    assert result.metadata["mrope_owner"] == "backbone"
    assert model.generation_kwargs is not None
    assert model.generation_kwargs["second_per_grid_ts"] == [0.5]
    assert model.generation_kwargs["temperature"] is None
    assert model.generation_kwargs["top_p"] is None
    assert model.generation_kwargs["top_k"] is None
    assert bridge.records[-1]["video_media_ids"] == ("all-modalities:video-0",)
    with pytest.raises(InvariantError, match="no Talker"):
        facade.runner.generate(
            runtime,
            state,
            processed_batch=batch,
            return_audio=True,
        )
    with pytest.raises(InvariantError, match="no Thinker/Talker/Token2Wav"):
        facade.runner.generate(
            runtime,
            state,
            processed_batch=batch,
            trace_request=TraceRequest(thinker_logits=True),
        )


def test_runner_supports_text_only_to_text_without_visual_fields() -> None:
    processor = _FakeProcessor(
        {"input_ids": [[1, 2]], "attention_mask": [[1, 1]]}
    )
    adapter = Qwen25VLProcessorAdapter(
        media_loader=lambda _messages: (None, None)
    )
    batch = adapter.prepare_inputs(
        processor,
        Qwen25VLSample(
            "text-only",
            _conversation({"type": "text", "text": "hello"}),
        ),
        phase="eval",
    )
    facade = create_qwen25_vl_backbone()
    state = facade.state_builder.build(batch, phase="eval")
    model = SimpleNamespace(generate=lambda **kwargs: [kwargs["input_ids"][0] + [9]])

    result = facade.runner.generate(
        BackboneRuntime(model=model, processor=processor),
        state,
        processed_batch=batch,
        do_sample=False,
    )

    assert state.pixel_values is None
    assert state.video_pixel_values is None
    assert result.text is not None and result.text.token_ids == [[9]]
    assert result.metadata["input_modalities"] == ["text"]


def test_training_runner_publishes_stable_video_media_identity() -> None:
    _adapter, processor, batch, _loader_calls = _all_modality_batch()
    facade = create_qwen25_vl_backbone()
    state = facade.state_builder.build(batch, phase="train")
    records: list[dict[str, Any]] = []

    class _Bridge:
        @contextmanager
        def request_context(self, metadata: Any) -> Any:
            records.append(dict(metadata))
            yield

    class _Model:
        _mllm_comp_qwen25_vl_video_bridge = _Bridge()

        def __call__(self, **kwargs: Any) -> Any:
            assert records
            return kwargs

    output = facade.runner.forward_train(
        BackboneRuntime(model=_Model(), processor=processor),
        state,
    )

    assert output["second_per_grid_ts"] == [0.5]
    assert records[-1]["video_media_ids"] == ("all-modalities:video-0",)


def test_structural_handles_expose_visual_blocks_merger_and_language_model() -> None:
    vision_block = SimpleNamespace(attn=object(), mlp=object())
    merger = object()
    visual = SimpleNamespace(
        blocks=[vision_block],
        merger=merger,
        spatial_merge_size=2,
    )
    layer = SimpleNamespace(self_attn=object(), mlp=object())
    final_norm = object()
    language_model = SimpleNamespace(layers=[layer], norm=final_norm)
    get_video_features = object()
    get_rope_index = object()
    core = SimpleNamespace(
        visual=visual,
        language_model=language_model,
        get_video_features=get_video_features,
        get_rope_index=get_rope_index,
    )
    video_bridge = object()
    model = SimpleNamespace(
        model=core,
        lm_head=object(),
        mllm_video_compression_bridge=video_bridge,
    )

    handles = Qwen25VLModelAdapter().structural_handles(model)

    assert handles["vision_encoder"] is visual
    assert handles["vision_encoder.blocks"] is visual.blocks
    assert handles["vision_encoder.block[0]"] is vision_block
    assert handles["vision_projector"] is merger
    assert handles["vision_projector.merger"] is merger
    assert handles["llm"] is language_model
    assert handles["language_model"] is language_model
    assert handles["llm.final_norm"] is final_norm
    assert handles["mrope_owner"] is core
    assert handles["vision_spatial_merge_size"] == 2
    assert handles["video_features_pre_placeholder_scatter"] is get_video_features
    assert handles["video_grid_thw_sparse_mrope"] is get_rope_index
    assert handles["video_compression_bridge"] is video_bridge
    assert "vision_deepstack_projectors" not in handles

    info = Qwen25VLModelAdapter().external_code_info()
    assert info["supported_input_slice"] == "text/image/single-video -> text"
    assert info["video_facade_implemented"] is True
    assert info["video_real_checkpoint_verified"] is False


def test_model_loader_checks_family_and_normalizes_dtype(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, dict[str, Any]]] = []

    class _AutoConfig:
        model_type = "qwen2_5_vl"

        @classmethod
        def from_pretrained(cls, _path: str, **_kwargs: Any) -> Any:
            return SimpleNamespace(model_type=cls.model_type)

    class _LoadedModel:
        def train(self, *, mode: bool) -> None:
            self.train_mode = mode

    class _AutoModel:
        @classmethod
        def from_pretrained(cls, path: str, **kwargs: Any) -> _LoadedModel:
            calls.append((path, kwargs))
            return _LoadedModel()

    monkeypatch.setitem(
        sys.modules,
        "transformers",
        SimpleNamespace(
            AutoConfig=_AutoConfig,
            AutoModelForImageTextToText=_AutoModel,
            Qwen2_5_VLForConditionalGeneration=object,
        ),
    )

    model = Qwen25VLModelAdapter().load_model(
        "fixture",
        train=True,
        torch_dtype="bfloat16",
        local_files_only=True,
    )
    assert model.train_mode is True
    assert calls == [("fixture", {"local_files_only": True, "dtype": "bfloat16"})]

    _AutoConfig.model_type = "qwen3_vl"
    with pytest.raises(BackendIntegrationError, match="not Qwen2.5-VL"):
        Qwen25VLModelAdapter().load_model("fixture", train=False)


def test_baseline_legalizer_rejects_unbound_token_mutation() -> None:
    _adapter, _processor, batch, _loader_calls = _all_modality_batch()
    facade = create_qwen25_vl_backbone()
    state = facade.state_builder.build(batch, phase="eval")

    with pytest.raises(
        NotActivatedError,
        match="explicit image or video compression binding",
    ):
        facade.legalizer.apply(
            state,
            token_actions=[TokenAction(op="keep", target_indices=[0])],
            policy=LegalizationPolicy(
                position_policy="m_rope_rebuild",
                mask_policy="rebuild_from_tokens",
                label_policy="rebuild_ignore_index",
            ),
            stage_name="fixture",
        )
