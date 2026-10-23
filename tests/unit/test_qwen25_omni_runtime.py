"""CPU-only contract tests for the Qwen2.5-Omni backbone implementation."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import replace
import sys
from types import SimpleNamespace
from typing import Any

import pytest

from mllm_comp.backbones.base import BackboneRuntime, ProcessedBatch, TraceRequest
from mllm_comp.backbones.qwen25_omni import (
    DEFAULT_AUDIO_SYSTEM_PROMPT,
    Qwen25OmniModelAdapter,
    Qwen25OmniProcessorAdapter,
    Qwen25OmniSample,
    create_qwen25_omni_backbone,
)
from mllm_comp.backbones.qwen25_omni import (
    implementation as qwen25_omni_implementation,
)
from mllm_comp.core.errors import InvariantError, NotActivatedError
from mllm_comp.backbones.qwen25_omni.tracing import capture_qwen25_omni_traces
from mllm_comp.pipeline.runtime_session import (
    CompressionActivationRecord,
    CompressionSession,
)


class _FakeTokenizer:
    _ids = {"<image>": 10, "<video>": 11, "<audio>": 12}

    def convert_tokens_to_ids(self, token: str) -> int:
        return self._ids[token]


class _FakeProcessor:
    image_token = "<image>"
    video_token = "<video>"
    audio_token = "<audio>"

    def __init__(self, outputs: dict[str, Any]) -> None:
        self.outputs = outputs
        self.tokenizer = _FakeTokenizer()
        self.image_processor = SimpleNamespace(merge_size=2)
        self.video_processor = SimpleNamespace(merge_size=2)
        self.feature_extractor = SimpleNamespace(sampling_rate=16000)
        self.processor_call: dict[str, Any] | None = None

    def apply_chat_template(self, conversation: Any, **kwargs: Any) -> str:
        del conversation, kwargs
        return "formatted"

    def __call__(self, **kwargs: Any) -> dict[str, Any]:
        self.processor_call = kwargs
        return self.outputs

    def batch_decode(self, token_ids: Any, **kwargs: Any) -> list[str]:
        del kwargs
        return [" ".join(str(token) for token in row) for row in token_ids]


def test_transformers_457_loader_preserves_slow_processor_and_uses_dtype(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    processor_calls: list[tuple[str, dict[str, Any]]] = []
    model_calls: list[tuple[str, dict[str, Any]]] = []

    class _ProcessorClass:
        @classmethod
        def from_pretrained(cls, path: str, **kwargs: Any) -> object:
            processor_calls.append((path, kwargs))
            return object()

    class _LoadedModel:
        def train(self, *, mode: bool) -> None:
            self.train_mode = mode

        def disable_talker(self) -> None:
            self.talker_disabled = True

    class _ModelClass:
        @classmethod
        def from_pretrained(cls, path: str, **kwargs: Any) -> _LoadedModel:
            model_calls.append((path, kwargs))
            return _LoadedModel()

    monkeypatch.setitem(
        sys.modules,
        "transformers",
        SimpleNamespace(
            Qwen2_5OmniProcessor=_ProcessorClass,
            Qwen2_5OmniForConditionalGeneration=_ModelClass,
        ),
    )

    Qwen25OmniProcessorAdapter().load_processor("fixture")
    Qwen25OmniModelAdapter().load_model(
        "fixture",
        train=False,
        torch_dtype="bfloat16",
        enable_talker=False,
    )

    assert processor_calls == [("fixture", {"use_fast": False})]
    assert model_calls == [("fixture", {"dtype": "bfloat16"})]


def _system_message() -> dict[str, Any]:
    return {
        "role": "system",
        "content": [{"type": "text", "text": DEFAULT_AUDIO_SYSTEM_PROMPT}],
    }


def test_semantic_char_spans_use_full_template_content_offsets() -> None:
    rendered = (
        "<|im_start|>system\nSYS<|im_end|>\n"
        "<|im_start|>user\nASK<|im_end|>\n"
        "<|im_start|>assistant\n"
    )

    spans, warnings = qwen25_omni_implementation._semantic_char_spans(
        rendered,
        [
            {"label": "system_prompt", "text": "SYS"},
            {"label": "instruction", "text": "ASK"},
        ],
    )

    assert warnings == []
    assert spans[0] == {
        "template_start": 19,
        "template_end": 22,
        "label": "system_prompt",
    }
    assert all(
        rendered[span["template_start"] : span["template_end"]]
        in {"SYS", "ASK"}
        for span in spans
    )


def test_prepare_inputs_and_state_builder_cover_all_native_modalities() -> None:
    input_ids = [[1, *([10] * 4), *([11] * 8), *([12] * 4), 2]]
    outputs = {
        "input_ids": input_ids,
        "attention_mask": [[1] * len(input_ids[0])],
        "pixel_values": [[0.0]],
        "image_grid_thw": [[1, 4, 4]],
        "pixel_values_videos": [[0.0]],
        "video_grid_thw": [[2, 4, 4]],
        "video_second_per_grid": [0.5],
        "input_features": [[[0.0]], [[0.0]]],
        "feature_attention_mask": [[1] * 10, [1] * 10],
    }
    processor = _FakeProcessor(outputs)
    adapter = Qwen25OmniProcessorAdapter(
        media_loader=lambda *_args, **_kwargs: (["a", "v-a"], ["i"], ["v"])
    )
    sample = Qwen25OmniSample(
        sample_id="all-modalities",
        use_audio_in_video=True,
        conversation=[
            _system_message(),
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": "image.png"},
                    {"type": "audio", "audio": "audio.wav"},
                    {"type": "video", "video": "video.mp4"},
                    {"type": "text", "text": "Describe everything."},
                ],
            },
        ],
    )

    batch = adapter.prepare_inputs(processor, sample, phase="eval")
    facade = create_qwen25_omni_backbone()
    state = facade.state_builder.build(batch, phase="eval")
    runtime = BackboneRuntime(model=SimpleNamespace(), processor=processor)
    violations = facade.native_validator.validate_state(runtime, state, batch)

    assert violations == []
    assert batch.metadata["expected_token_counts_by_sample"] == [
        {"image": 4, "video": 8, "audio": 4}
    ]
    assert state.video_pixel_values is outputs["pixel_values_videos"]
    assert state.audio_values is outputs["input_features"]
    assert state.native_model_inputs == outputs
    assert [(span.modality, span.end - span.start) for span in state.modality_token_spans] == [
        ("image", 4),
        ("video", 8),
        ("audio", 4),
    ]
    assert processor.processor_call["use_audio_in_video"] is True
    assert "semantic_char_spans_by_sample" not in batch.metadata
    assert "semantic_message_prefixes_by_sample" not in batch.metadata
    assert "special_token_ids" not in batch.metadata
    assert len(batch.metadata["semantic_token_labels_by_sample"][0]) == len(
        input_ids[0]
    )

    debug_batch = adapter.prepare_inputs(
        processor,
        sample,
        phase="eval",
        runtime_mode="debug",
    )
    assert "semantic_char_spans_by_sample" in debug_batch.metadata
    assert "semantic_message_prefixes_by_sample" in debug_batch.metadata
    assert [
        item["label"]
        for item in debug_batch.metadata["semantic_message_prefixes_by_sample"][0]
    ] == ["system_prompt", "instruction"]
    assert "semantic_classification_warnings_by_sample" in debug_batch.metadata
    assert "special_token_ids" in debug_batch.metadata


def test_processor_builds_native_teacher_forced_answer_labels() -> None:
    class _TeacherTokenizer(_FakeTokenizer):
        all_special_ids = (99,)

    class _TeacherProcessor(_FakeProcessor):
        def __init__(self) -> None:
            super().__init__({})
            self.tokenizer = _TeacherTokenizer()

        def apply_chat_template(self, conversation: Any, **kwargs: Any) -> str:
            self.template_call = {"conversation": conversation, **kwargs}
            return "full" if conversation[-1]["role"] == "assistant" else "prompt"

        def __call__(self, **kwargs: Any) -> dict[str, Any]:
            self.processor_call = kwargs
            ids = [1, 2, 20, 99] if kwargs["text"] == "full" else [1, 2]
            return {"input_ids": [ids], "attention_mask": [[1] * len(ids)]}

    processor = _TeacherProcessor()
    adapter = Qwen25OmniProcessorAdapter(
        media_loader=lambda *_args, **_kwargs: (None, None, None)
    )
    sample = Qwen25OmniSample(
        sample_id="teacher-forced",
        conversation=[
            {"role": "user", "content": [{"type": "text", "text": "question"}]}
        ],
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


class _FakeToken2WavHandle:
    def __init__(self, owner: "_FakeToken2Wav") -> None:
        self.owner = owner

    def remove(self) -> None:
        self.owner.pre_hook = None


class _FakeToken2Wav:
    def __init__(self) -> None:
        self.pre_hook: Any = None

    def register_forward_pre_hook(
        self,
        hook: Any,
        *,
        with_kwargs: bool,
    ) -> _FakeToken2WavHandle:
        assert with_kwargs is True
        self.pre_hook = hook
        return _FakeToken2WavHandle(self)

    def emit(self, codec_ids: Any) -> None:
        assert self.pre_hook is not None
        self.pre_hook(self, (codec_ids,), {})


class _FakeModel:
    has_talker = True
    speaker_map = {"Chelsie": object(), "Ethan": object()}

    def __init__(self) -> None:
        self.kwargs: dict[str, Any] = {}
        self.token2wav = _FakeToken2Wav()

    def generate(self, **kwargs: Any) -> Any:
        self.kwargs = kwargs
        sequences = [[1, 2, 7, 8]]
        if kwargs["return_audio"]:
            self.token2wav.emit([[5, 6, 7]])
            return sequences, [0.1, -0.1]
        return sequences


class _FakeDynamicCompression:
    def __init__(
        self,
        name: str,
        placement: str,
        events: list[str],
    ) -> None:
        self.name = name
        self.placement = placement
        self.events = events
        self.last_report: dict[str, Any] | None = None

    def install(self) -> None:
        return None

    def uninstall(self) -> None:
        return None

    def runtime_metadata(self) -> dict[str, Any]:
        return {"algorithm": self.name}

    def prepare_state(self, state: Any) -> tuple[Any, object]:
        prepared_by = list(state.modality_metadata.get("prepared_by", ()))
        prepared_by.append(self.name)
        metadata = {**state.modality_metadata, "prepared_by": prepared_by}
        self.events.append(f"prepare:{self.name}")
        self.last_report = {
            "stage_name": self.name,
            "placement": self.placement,
            "prepared_by": prepared_by,
        }
        return replace(state, modality_metadata=metadata), object()

    @contextmanager
    def request(self, plan: object) -> Any:
        assert plan is not None
        self.events.append(f"enter:{self.name}")
        try:
            yield
        finally:
            self.events.append(f"exit:{self.name}")


class _FakeTrainThinker:
    def __init__(self, events: list[str]) -> None:
        self.events = events

    def __call__(self, **kwargs: Any) -> dict[str, Any]:
        self.events.append("forward")
        return kwargs


def _install_fake_compression_session(
    runtime: BackboneRuntime,
    activations: list[_FakeDynamicCompression],
) -> CompressionSession:
    records = [
        CompressionActivationRecord(
            name=activation.name,
            activation=activation,
            binding=f"{activation.name}.audio.qwen25_omni",
            dependencies=((activations[index - 1].name,) if index else ()),
            config_order=index,
        )
        for index, activation in enumerate(activations)
    ]
    session = CompressionSession(
        records,
        metadata={"mode": "compressed", "backbone": "qwen25_omni"},
    )
    runtime.metadata["_compression_session"] = session
    session.install()
    runtime.metadata["compression"] = session.runtime_metadata()
    return session


def _text_batch() -> ProcessedBatch:
    return ProcessedBatch(
        model_inputs={"input_ids": [[1, 2]], "attention_mask": [[1, 1]]},
        sample_ids=["sample"],
        prompt_lengths=[2],
        metadata={
            "token_ids": {"image": 10, "video": 11, "audio": 12},
            "expected_token_counts_by_sample": [{"image": 0, "video": 0, "audio": 0}],
            "use_audio_in_video": False,
            "uses_default_audio_system_prompt": [True],
            "speech_output_sample_rate": 24000,
        },
    )


def test_runner_normalizes_text_and_speech_outputs() -> None:
    facade = create_qwen25_omni_backbone()
    processor = _FakeProcessor({})
    model = _FakeModel()
    runtime = BackboneRuntime(model=model, processor=processor)
    batch = _text_batch()
    state = facade.state_builder.build(batch, phase="eval")

    text_result = facade.runner.generate(
        runtime,
        state,
        processed_batch=batch,
        return_audio=False,
    )
    speech_result = facade.runner.generate(
        runtime,
        state,
        processed_batch=batch,
        return_audio=True,
        speaker="Ethan",
    )

    assert text_result.text.token_ids == [[7, 8]]
    assert text_result.text.text == ["7 8"]
    assert text_result.speech is None
    assert text_result.metadata["runtime_mode"] == "default"
    assert text_result.metadata["timing"] is None
    assert text_result.metadata["token_debug"] is None
    assert speech_result.speech.waveform == [0.1, -0.1]
    assert speech_result.speech.sample_rate == 24000
    assert speech_result.speech.speaker == "Ethan"
    assert speech_result.speech.codec_token_ids == [[5, 6, 7]]
    assert speech_result.traces is None
    assert model.token2wav.pre_hook is None

    profile_runtime = BackboneRuntime(
        model=model,
        processor=processor,
        metadata={"runtime_mode": "profile"},
    )
    profile_result = facade.runner.generate(
        profile_runtime,
        state,
        processed_batch=batch,
        return_audio=False,
    )
    assert profile_result.metadata["runtime_mode"] == "profile"
    assert profile_result.metadata["timing"]["generation_seconds"] is not None
    assert profile_result.metadata["token_debug"] is None


def test_runner_uses_native_thinker_checkpoint_generation_arguments() -> None:
    facade = create_qwen25_omni_backbone()
    calls = []
    model = SimpleNamespace(
        config=SimpleNamespace(model_type="qwen2_5_omni_thinker"),
        generate=lambda **kwargs: calls.append(kwargs) or [[1, 2, 7, 8]],
    )
    runtime = BackboneRuntime(model=model, processor=_FakeProcessor({}))
    batch = _text_batch()
    state = facade.state_builder.build(batch, phase="eval")
    result = facade.runner.generate(
        runtime, state, processed_batch=batch,
        thinker_max_new_tokens=2, thinker_do_sample=False, temperature=0.7,
    )
    assert calls[0]["max_new_tokens"] == 2
    assert calls[0]["do_sample"] is False
    assert calls[0]["temperature"] == 0.7
    assert "return_audio" not in calls[0] and "speaker" not in calls[0]
    assert not any(key.startswith("thinker_") for key in calls[0])
    assert result.text.token_ids == [[7, 8]]
    assert "ThinkerForConditionalGeneration" in result.metadata["provider"]


def test_runner_consumes_single_v2_session_activation() -> None:
    events: list[str] = []
    facade = create_qwen25_omni_backbone()
    runtime = BackboneRuntime(
        model=_FakeModel(),
        processor=_FakeProcessor({}),
    )
    activation = _FakeDynamicCompression(
        "segmentwise_topk_audio",
        "projector_level",
        events,
    )
    _install_fake_compression_session(runtime, [activation])
    batch = _text_batch()
    state = facade.state_builder.build(batch, phase="eval")

    result = facade.runner.generate(runtime, state, processed_batch=batch)

    assert events == [
        "prepare:segmentwise_topk_audio",
        "enter:segmentwise_topk_audio",
        "exit:segmentwise_topk_audio",
    ]
    assert result.metadata["compression"] == activation.last_report
    assert result.metadata["compression"]["prepared_by"] == [
        "segmentwise_topk_audio"
    ]
    assert "_compression_controller" not in runtime.metadata


@pytest.mark.parametrize("thinker_only, generation_kwargs, cache_enabled", [
    (False, {}, True),
    (True, {}, True),
    (False, {"use_cache": False, "thinker_use_cache": True}, True),
    (True, {"use_cache": True, "thinker_use_cache": False}, False),
    (False, {"generation_config": SimpleNamespace(use_cache=False)}, False),
    (True, {"generation_config": SimpleNamespace(use_cache=False), "use_cache": True}, True),
    (False, {"generation_config": SimpleNamespace(use_cache=True),
      "thinker_generation_config": SimpleNamespace(use_cache=False)}, False),
    (True, {"generation_config": SimpleNamespace(use_cache=False),
      "thinker_generation_config": None, "thinker_use_cache": None}, True),
])
def test_dynamic_generation_cache_choice_respects_native_argument_precedence(
    thinker_only, generation_kwargs, cache_enabled,
) -> None:
    facade = create_qwen25_omni_backbone()
    calls = []
    events = []
    model = SimpleNamespace(
        config=SimpleNamespace(model_type=(
            "qwen2_5_omni_thinker" if thinker_only else "qwen2_5_omni"
        )),
        generate=lambda **kwargs: calls.append(kwargs) or [[1, 2, 7, 8]],
    )
    runtime = BackboneRuntime(model=model, processor=_FakeProcessor({}))
    _install_fake_compression_session(runtime, [
        _FakeDynamicCompression("audio", "projector_level", events)
    ])
    batch = _text_batch()
    state = facade.state_builder.build(batch, phase="eval")
    if cache_enabled:
        facade.runner.generate(runtime, state, processed_batch=batch, **generation_kwargs)
        cache_key = "use_cache" if thinker_only else "thinker_use_cache"
        assert calls[0][cache_key] is True
        assert events == ["prepare:audio", "enter:audio", "exit:audio"]
    else:
        with pytest.raises(InvariantError, match="requires thinker_use_cache=True"):
            facade.runner.generate(runtime, state, processed_batch=batch, **generation_kwargs)
        assert calls == []
        assert events == ["prepare:audio"]


@pytest.mark.parametrize("static_compression", [False, True])
def test_generation_without_dynamic_requests_preserves_disabled_cache(static_compression):
    facade = create_qwen25_omni_backbone()
    model = _FakeModel()
    runtime = BackboneRuntime(model=model, processor=_FakeProcessor({}))
    if static_compression:
        _install_fake_compression_session(runtime, [SimpleNamespace(
            name="operation", install=lambda: None, uninstall=lambda: None,
            runtime_metadata=lambda: {"algorithm": "operation"},
        )])
    batch = _text_batch()
    state = facade.state_builder.build(batch, phase="eval")
    facade.runner.generate(runtime, state, processed_batch=batch, use_cache=False)
    assert model.kwargs["use_cache"] is False
    assert "thinker_use_cache" not in model.kwargs


def test_runner_executes_multiple_v2_stages_in_dependency_order() -> None:
    events: list[str] = []
    facade = create_qwen25_omni_backbone()
    model = _FakeModel()
    model.thinker = _FakeTrainThinker(events)
    runtime = BackboneRuntime(model=model, processor=_FakeProcessor({}))
    activations = [
        _FakeDynamicCompression(
            "segmentwise_topk_audio",
            "projector_level",
            events,
        ),
        _FakeDynamicCompression(
            "audio_pyramid_attention_pruning",
            "llm_level",
            events,
        ),
    ]
    _install_fake_compression_session(runtime, activations)
    batch = _text_batch()
    batch.metadata["semantic_token_labels_by_sample"] = [
        ["system_prompt", "instruction"]
    ]
    state = facade.state_builder.build(batch, phase="train")

    output = facade.runner.forward_train(runtime, state)

    assert output["input_ids"] == [[1, 2]]
    assert events == [
        "prepare:segmentwise_topk_audio",
        "prepare:audio_pyramid_attention_pruning",
        "enter:segmentwise_topk_audio",
        "enter:audio_pyramid_attention_pruning",
        "forward",
        "exit:audio_pyramid_attention_pruning",
        "exit:segmentwise_topk_audio",
    ]
    report = runtime.metadata["last_training_forward"]["compression"]
    assert report["multi_stage"] is True
    assert report["stage_count"] == 2
    assert [stage["name"] for stage in report["stages"]] == [
        "segmentwise_topk_audio",
        "audio_pyramid_attention_pruning",
    ]
    assert report["stages"][1]["report"]["prepared_by"] == [
        "segmentwise_topk_audio",
        "audio_pyramid_attention_pruning",
    ]
    assert runtime.metadata["last_training_forward"]["effective_token_layout"] == {
        "schema_version": 1,
        "sample_ids": ["sample"],
        "sequence_length": 2,
        "semantic_labels_by_sample": [["system_prompt", "instruction"]],
        "semantic_classification_source": None,
        "semantic_labels_by_layer_entry": {},
    }


def test_runner_rejects_inconsistent_audio_in_video_and_fake_streaming() -> None:
    facade = create_qwen25_omni_backbone()
    runtime = BackboneRuntime(model=_FakeModel(), processor=_FakeProcessor({}))
    batch = _text_batch()
    state = facade.state_builder.build(batch, phase="eval")

    with pytest.raises(InvariantError, match="use_audio_in_video"):
        facade.runner.generate(
            runtime,
            state,
            processed_batch=batch,
            use_audio_in_video=True,
        )
    with pytest.raises(NotActivatedError, match="synchronous"):
        facade.runner.generate(runtime, state, processed_batch=batch, stream=True)
    with pytest.raises(InvariantError, match="Thinker trace fields"):
        facade.runner.generate(
            runtime,
            state,
            processed_batch=batch,
            trace_request=TraceRequest(text_logits=True),
        )


def test_structural_handles_cover_thinker_talker_and_token2wav() -> None:
    layer = SimpleNamespace(self_attn=object(), mlp=object())
    visual = SimpleNamespace(merger=object())
    audio = SimpleNamespace(proj=object())
    thinker = SimpleNamespace(
        model=SimpleNamespace(layers=[layer]),
        visual=visual,
        audio_tower=audio,
        lm_head=object(),
    )
    talker = SimpleNamespace(
        model=SimpleNamespace(layers=[object()]),
        codec_head=object(),
        thinker_to_talker_proj=object(),
    )
    token2wav = SimpleNamespace(
        code2wav_dit_model=object(),
        code2wav_bigvgan_model=object(),
    )
    model = SimpleNamespace(thinker=thinker, talker=talker, token2wav=token2wav)

    handles = Qwen25OmniModelAdapter().structural_handles(model)

    assert handles["llm.layer[0]"] is layer
    assert handles["vision_encoder"] is visual
    assert handles["audio_encoder"] is audio
    assert handles["talker"] is talker
    assert handles["talker.thinker_projector"] is talker.thinker_to_talker_proj
    assert handles["token2wav"] is token2wav
    assert handles["token2wav.dit"] is token2wav.code2wav_dit_model
    assert handles["token2wav.vocoder"] is token2wav.code2wav_bigvgan_model


class _HookHandle:
    def __init__(self, owner: "_HookModule", kind: str) -> None:
        self.owner = owner
        self.kind = kind

    def remove(self) -> None:
        setattr(self.owner, self.kind, None)


class _HookModule:
    forward_hook: Any = None
    pre_hook: Any = None

    def register_forward_hook(self, hook: Any, *, with_kwargs: bool) -> _HookHandle:
        assert with_kwargs is True
        self.forward_hook = hook
        return _HookHandle(self, "forward_hook")

    def register_forward_pre_hook(self, hook: Any, *, with_kwargs: bool) -> _HookHandle:
        assert with_kwargs is True
        self.pre_hook = hook
        return _HookHandle(self, "pre_hook")


def test_opt_in_hooks_capture_cross_stage_outputs_and_remove_themselves() -> None:
    thinker = _HookModule()
    talker = _HookModule()
    token2wav = _HookModule()
    model = SimpleNamespace(thinker=thinker, talker=talker, token2wav=token2wav)
    request = TraceRequest(
        thinker_logits=True,
        thinker_reply_part=True,
        talker_logits=True,
        talker_codec_token_ids=True,
        token2wav_intermediates=True,
    )

    with capture_qwen25_omni_traces(model, request) as artifacts:
        thinker.forward_hook(thinker, (), {}, SimpleNamespace(logits="thinker-logits"))
        talker.pre_hook(talker, (), {"thinker_reply_part": "reply"})
        talker.forward_hook(talker, (), {}, SimpleNamespace(logits="talker-logits"))
        token2wav.pre_hook(
            token2wav,
            ("codes",),
            {"conditioning": "cond", "reference_mel": "mel"},
        )

    assert artifacts.thinker_logits == "thinker-logits"
    assert artifacts.thinker_reply_part == "reply"
    assert artifacts.talker_logits == "talker-logits"
    assert artifacts.talker_codec_token_ids == "codes"
    assert artifacts.token2wav_intermediates["reference_mel"] == "mel"
    assert thinker.forward_hook is None
    assert talker.pre_hook is None
    assert token2wav.pre_hook is None
