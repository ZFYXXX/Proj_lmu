"""CPU-only contracts for the Qwen3-VL backbone implementation."""

from __future__ import annotations

import sys
from contextlib import contextmanager
from types import SimpleNamespace
from typing import Any

import pytest

from mllm_comp.backbones.base import BackboneRuntime, TraceRequest
from mllm_comp.backbones.qwen3_vl import (
    Qwen3VLModelAdapter,
    Qwen3VLProcessorAdapter,
    Qwen3VLSample,
    create_qwen3_vl_backbone,
)
from mllm_comp.backbones.qwen3_vl.tracing import capture_qwen3_vl_traces
from mllm_comp.core.errors import BackendIntegrationError, InvariantError


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
        self.image_processor = SimpleNamespace(merge_size=2)
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


def _all_modality_batch() -> tuple[Any, Any, Any]:
    input_ids = [[1, *([10] * 4), *([11] * 8), 2]]
    outputs = {
        "input_ids": input_ids,
        "attention_mask": [[1] * len(input_ids[0])],
        "pixel_values": [[0.0]],
        "image_grid_thw": [[1, 4, 4]],
        "pixel_values_videos": [[0.0]],
        "video_grid_thw": [[2, 4, 4]],
        "second_per_grid_ts": [0.5],
        "mm_token_type_ids": [[0] * len(input_ids[0])],
    }
    processor = _FakeProcessor(outputs)
    adapter = Qwen3VLProcessorAdapter(
        media_loader=lambda *_args, **_kwargs: (
            ["image-tensor"],
            [("video-tensor", {"fps": 4.0})],
            {"fps": 4.0},
        )
    )
    sample = Qwen3VLSample(
        sample_id="all-modalities",
        enable_thinking=False,
        conversation=_conversation(
            {"type": "image", "image": "image.png"},
            {"type": "video", "video": "video.mp4"},
            {"type": "text", "text": "Describe both."},
        ),
    )
    batch = adapter.prepare_inputs(processor, sample, phase="eval")
    return adapter, processor, batch


def test_prepare_inputs_state_and_validator_cover_qwen3_visual_contract() -> None:
    _adapter, processor, batch = _all_modality_batch()
    facade = create_qwen3_vl_backbone()
    state = facade.state_builder.build(batch, phase="eval")
    runtime = BackboneRuntime(model=object(), processor=processor)

    assert processor.template_call is not None
    assert processor.template_call["enable_thinking"] is False
    assert processor.processor_call is not None
    assert processor.processor_call["do_resize"] is False
    assert processor.processor_call["video_metadata"] == [{"fps": 4.0}]
    assert state.video_pixel_values == [[0.0]]
    assert state.token_type_mask == batch.model_inputs["mm_token_type_ids"]
    assert [(span.modality, span.end - span.start) for span in state.modality_token_spans] == [
        ("image", 4),
        ("video", 8),
    ]
    assert batch.metadata["image_patch_size"] == 16
    assert batch.metadata["expected_token_counts_by_sample"] == [
        {"image": 4, "video": 8}
    ]
    assert batch.metadata["video_media_ids"] == ("all-modalities:video-0",)
    assert batch.metadata["video_count"] == 1
    assert batch.metadata["native_video_timing_present"] is True
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
    adapter = Qwen3VLProcessorAdapter(
        media_loader=lambda *_args, **_kwargs: (None, None, {})
    )
    sample = Qwen3VLSample(
        "teacher-forced",
        _conversation({"type": "text", "text": "question"}),
        False,
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
    assert processor.template_call["enable_thinking"] is False


def test_processor_rejects_audio_and_mixed_thinking_modes() -> None:
    processor = _FakeProcessor(
        {"input_ids": [[1]], "attention_mask": [[1]]}
    )
    adapter = Qwen3VLProcessorAdapter(
        media_loader=lambda *_args, **_kwargs: (None, None, {})
    )
    with pytest.raises(BackendIntegrationError, match="does not accept audio"):
        adapter.prepare_inputs(
            processor,
            Qwen3VLSample(
                "audio",
                _conversation({"type": "audio", "audio": "sample.wav"}),
            ),
            phase="eval",
        )
    with pytest.raises(ValueError, match="mix enable_thinking"):
        adapter.prepare_inputs(
            processor,
            [
                Qwen3VLSample("a", _conversation({"type": "text", "text": "a"}), True),
                Qwen3VLSample("b", _conversation({"type": "text", "text": "b"}), False),
            ],
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
    }
    processor = _FakeProcessor(outputs)
    adapter = Qwen3VLProcessorAdapter(
        media_loader=lambda *_args, **_kwargs: (
            ["image-tensor"],
            [("video-tensor", {"fps": 4.0})],
            {"fps": 4.0},
        )
    )
    samples = [
        Qwen3VLSample(
            "image-sample",
            _conversation(
                {"type": "image", "image": "image.png"},
                {"type": "text", "text": "image"},
            ),
            False,
        ),
        Qwen3VLSample(
            "video-sample",
            _conversation(
                {"type": "video", "video": "video.mp4"},
                {"type": "text", "text": "video"},
            ),
            False,
        ),
    ]

    batch = adapter.prepare_inputs(processor, samples, phase="eval")
    facade = create_qwen3_vl_backbone()
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
        BackboneRuntime(model=object(), processor=processor), state, batch
    ) == []


def test_processor_rejects_malformed_or_nondivisible_visual_grid() -> None:
    sample = Qwen3VLSample(
        "bad-grid",
        _conversation(
            {"type": "image", "image": "image.png"},
            {"type": "text", "text": "image"},
        ),
    )
    adapter = Qwen3VLProcessorAdapter(
        media_loader=lambda *_args, **_kwargs: (["image"], None, {})
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


def test_runner_generates_text_and_rejects_speech() -> None:
    _adapter, processor, batch = _all_modality_batch()
    facade = create_qwen3_vl_backbone()
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
    model._mllm_comp_qwen3_vl_video_bridge = bridge
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
    assert result.traces is None
    assert model.generation_kwargs is not None
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


def test_training_runner_publishes_stable_video_media_identity() -> None:
    _adapter, processor, batch = _all_modality_batch()
    facade = create_qwen3_vl_backbone()
    state = facade.state_builder.build(batch, phase="train")
    records: list[dict[str, Any]] = []

    class _Bridge:
        @contextmanager
        def request_context(self, metadata: Any) -> Any:
            records.append(dict(metadata))
            yield

    class _Model:
        _mllm_comp_qwen3_vl_video_bridge = _Bridge()

        def __call__(self, **kwargs: Any) -> Any:
            assert records
            return kwargs

    output = facade.runner.forward_train(
        BackboneRuntime(model=_Model(), processor=processor),
        state,
    )

    assert output["second_per_grid_ts"] == [0.5]
    assert records[-1]["video_media_ids"] == ("all-modalities:video-0",)


def test_structural_handles_cover_deepstack_and_language_model() -> None:
    layer = SimpleNamespace(self_attn=object(), mlp=SimpleNamespace(experts=object()))
    merger = object()
    deepstack = [object(), object()]
    visual = SimpleNamespace(merger=merger, deepstack_merger_list=deepstack)
    final_norm = object()
    language_model = SimpleNamespace(layers=[layer], norm=final_norm)
    get_video_features = object()
    get_rope_index = object()
    core = SimpleNamespace(
        language_model=language_model,
        visual=visual,
        get_video_features=get_video_features,
        get_rope_index=get_rope_index,
    )
    video_bridge = object()
    model = SimpleNamespace(
        model=core,
        lm_head=object(),
        mllm_video_compression_bridge=video_bridge,
    )

    handles = Qwen3VLModelAdapter().structural_handles(model)

    assert handles["llm"] is language_model
    assert handles["language_model"] is language_model
    assert handles["llm.final_norm"] is final_norm
    assert handles["vision_encoder"] is visual
    assert handles["vision_projector"] is merger
    assert handles["vision_deepstack_projector[1]"] is deepstack[1]
    assert (
        handles["video_features_pre_placeholder_scatter_and_deepstack"]
        is get_video_features
    )
    assert handles["video_mrope_deepstack_visual_pos_masks"] is get_rope_index
    assert handles["video_compression_bridge"] is video_bridge
    assert handles["llm.layer[0]"] is layer
    assert handles["llm.layer[0].moe.experts"] is layer.mlp.experts

    info = Qwen3VLModelAdapter().external_code_info()
    assert info["supported_input_slice"] == "text/image/single-video -> text"
    assert info["video_facade_implemented"] is True
    assert info["video_deepstack_policy"] == "shared-source-groups-all-branches"
    assert info["video_real_checkpoint_verified"] is False


def test_model_loader_checks_checkpoint_family_and_normalizes_dtype(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, dict[str, Any]]] = []

    class _AutoConfig:
        model_type = "qwen3_vl"

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
            Qwen3VLForConditionalGeneration=object,
        ),
    )

    model = Qwen3VLModelAdapter().load_model(
        "fixture",
        train=True,
        torch_dtype="bfloat16",
        local_files_only=True,
    )
    assert model.train_mode is True
    assert calls == [
        ("fixture", {"local_files_only": True, "dtype": "bfloat16"})
    ]

    _AutoConfig.model_type = "not_qwen3"
    with pytest.raises(BackendIntegrationError, match="not Qwen3-VL"):
        Qwen3VLModelAdapter().load_model("fixture", train=False)


class _HookHandle:
    def __init__(self, owner: "_HookModule") -> None:
        self.owner = owner

    def remove(self) -> None:
        self.owner.hook = None


class _HookModule:
    def __init__(self) -> None:
        self.hook: Any | None = None

    def register_forward_hook(self, hook: Any, *, with_kwargs: bool) -> _HookHandle:
        assert with_kwargs is True
        self.hook = hook
        return _HookHandle(self)


def test_qwen3_trace_is_opt_in_and_removed() -> None:
    model = _HookModule()
    request = TraceRequest(
        text_logits=True,
        text_hidden_states=True,
        text_attentions=True,
        text_kv_cache=True,
    )

    with capture_qwen3_vl_traces(model, request) as artifacts:
        assert artifacts is not None and model.hook is not None
        model.hook(
            model,
            (),
            {},
            SimpleNamespace(
                logits="logits",
                hidden_states="hidden",
                attentions="attentions",
                past_key_values="kv-cache",
            ),
        )
        assert artifacts.text_logits == "logits"
        assert artifacts.text_hidden_states == "hidden"
        assert artifacts.text_attentions == "attentions"
        assert artifacts.text_kv_cache == "kv-cache"
    assert model.hook is None

    with capture_qwen3_vl_traces(model, None) as artifacts:
        assert artifacts is None
        assert model.hook is None
