"""Contract tests for the platform-owned lmms-eval bridge."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from types import ModuleType
import sys
from typing import Any

import pytest

from mllm_comp.backbones.base import (
    BackboneRuntime,
    GenerationResult,
    ProcessedBatch,
    TextGenerationOutput,
)
from mllm_comp.backbones.llava_onevision import LlavaOneVisionSample
from mllm_comp.backbones.qwen25_vl import Qwen25VLSample
from mllm_comp.backbones.qwen3_vl import Qwen3VLSample
from mllm_comp.config.schema import experiment_config_from_dict
from mllm_comp.config.loader import load_config
from mllm_comp.core.errors import BackendIntegrationError
from mllm_comp.core.state import ForwardState
from mllm_comp.core.vocabulary import RuntimeMode
from mllm_comp.integrations.lmms_eval import (
    LMMSEvalAdapter,
    LMMSEvalRequestBridge,
)
from mllm_comp.integrations.lmms_eval import adapter as lmms_eval_adapter_module
from mllm_comp.integrations.lmms_eval import diagnostics as lmms_eval_diagnostics
from mllm_comp.integrations.lmms_eval import model_bridge as model_bridge_module
from mllm_comp.integrations.native_cli import NativeCLIResult
from mllm_comp.run_manifest import build_run_manifest
from mllm_comp.experiments import run_experiment


class _ProcessorAdapter:
    def __init__(self) -> None:
        self.samples: list[Any] = []
        self.runtime_modes: list[str] = []

    def prepare_inputs(
        self,
        processor: Any,
        raw_sample: Any,
        *,
        phase: str,
        runtime_mode: str = "default",
    ) -> ProcessedBatch:
        del processor
        self.samples.append(raw_sample)
        self.runtime_modes.append(runtime_mode)
        return ProcessedBatch(
            model_inputs={"input_ids": [[1, 2]]},
            sample_ids=[raw_sample.sample_id],
            prompt_lengths=[2],
            metadata={"phase": phase},
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
        del processor, target_text, prompt_batch, runtime_mode
        return ProcessedBatch(
            model_inputs={
                "input_ids": [[1, 2, 3]],
                "labels": [[-100, -100, 3]],
            },
            sample_ids=[raw_sample.sample_id],
            prompt_lengths=[3],
            metadata={
                "phase": phase,
                "teacher_forcing": {
                    "objective": "teacher_forced_answer_cross_entropy_v1",
                    "answer_token_counts": [1],
                },
            },
        )


class _StateBuilder:
    def build(self, processed: ProcessedBatch, **kwargs: Any) -> Any:
        return {"processed": processed, **kwargs}


class _Runner:
    def __init__(self) -> None:
        self.generation_kwargs: list[dict[str, Any]] = []

    def generate(self, runtime: Any, state: Any, *, processed_batch: Any, **kwargs: Any) -> Any:
        del runtime, state, processed_batch
        self.generation_kwargs.append(kwargs)
        return GenerationResult(
            text=TextGenerationOutput(token_ids=[[3]], text=["answer<STOP>ignored"])
        )


class _Facade:
    def __init__(self, backbone_name: str = "qwen25_omni") -> None:
        self.backbone_name = backbone_name
        self.processor_adapter = _ProcessorAdapter()
        self.state_builder = _StateBuilder()
        self.runner = _Runner()

    def capabilities(self) -> Any:
        return SimpleNamespace(name=self.backbone_name)


class _Request:
    def __init__(self, args: tuple[Any, ...]) -> None:
        self.args = args


def _runtime() -> BackboneRuntime:
    processor = SimpleNamespace(
        tokenizer=SimpleNamespace(
            pad_token_id=99,
            eos_token_id=2,
            encode=lambda text, add_special_tokens=False: list(
                range(len(text.split()) + int(add_special_tokens))
            ),
            decode=lambda token_id: f"<eos:{token_id}>",
        ),
        feature_extractor=SimpleNamespace(sampling_rate=16000),
    )
    model = SimpleNamespace(config=SimpleNamespace(_name_or_path="fixture"))
    return BackboneRuntime(
        model=model,
        processor=processor,
        model_path_or_id="fixture",
        device="cuda:0",
    )


def _request(*, visual: Any = None, doc_id: int = 0) -> _Request:
    document = {"media": visual}
    return _Request(
        (
            "What is present?",
            {"max_new_tokens": 17, "temperature": 0.0, "until": ["<STOP>"]},
            lambda doc: doc["media"],
            doc_id,
            "fixture_task",
            "test",
        )
    ), document


def test_request_bridge_starts_a_fresh_diagnostic_trace(
    tmp_path: Path,
) -> None:
    trace_path = tmp_path / "diagnostic.jsonl"
    trace_path.write_text('{"stale":true}\n', encoding="utf-8")

    bridge = LMMSEvalRequestBridge(
        facade=_Facade(),
        runtime=_runtime(),
        diagnostic_trace_path=str(trace_path),
    )

    assert trace_path.read_text(encoding="utf-8") == ""
    bridge._write_diagnostic_trace({"fresh": True})
    assert [
        json.loads(line)
        for line in trace_path.read_text(encoding="utf-8").splitlines()
    ] == [{"fresh": True}]


def test_request_bridge_persists_default_mode_compression_lifecycle(
    tmp_path: Path,
) -> None:
    runtime = _runtime()
    runtime.metadata["compression"] = {
        "enabled": True,
        "plugin": "fixture_compressor",
        "executor_family": "llm_token",
    }
    manifest_path = tmp_path / "compression_runtime.json"
    bridge = LMMSEvalRequestBridge(
        facade=_Facade(),
        runtime=runtime,
        compression_manifest_path=str(manifest_path),
    )
    initial = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert initial["enabled"] is True
    assert initial["runtime"]["executor_family"] == "llm_token"
    assert initial["last_request"] is None

    first_request, first_document = _request(doc_id=0)
    second_request, second_document = _request(doc_id=1)
    bridge.task_dict = {
        "fixture_task": {"test": [first_document, second_document]}
    }
    assert bridge.generate_until([first_request, second_request]) == [
        "answer",
        "answer",
    ]

    completed = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert completed["last_request"]["sample_id"] == "fixture_task:test:1:1"
    assert completed["last_request"]["compression"] is None
    assert completed["request_count"] == 2
    trace_path = Path(completed["trace"]["path"])
    records = [json.loads(line) for line in trace_path.read_text(encoding="utf-8").splitlines()]
    assert [record["request"]["sample_id"] for record in records] == [
        "fixture_task:test:0:0", "fixture_task:test:1:1",
    ]


def test_request_bridge_clears_only_previous_request_records_and_preserves_failed_run_evidence(
    tmp_path: Path,
) -> None:
    from mllm_comp.pipeline.runtime_session import CompressionActivationRecord, CompressionSession

    class Activation:
        def __init__(self):
            self.records = []
            self.invocations = 0
            self.replay_state = object()

        def install(self):
            pass

        def uninstall(self):
            pass

        def clear_diagnostic_records(self):
            self.records.clear()

        def runtime_metadata(self):
            return {"invocation_count": self.invocations, "layout_delta_records": list(self.records)}

    activation = Activation()
    replay_state = activation.replay_state
    runtime = _runtime()
    runtime.metadata["_compression_session"] = CompressionSession([
        CompressionActivationRecord(name="fixture", binding="fixture.video", activation=activation),
    ]).install()
    facade = _Facade()
    generate = facade.runner.generate

    def generating(*args, **kwargs):
        assert activation.records == []
        assert activation.replay_state is replay_state
        activation.invocations += 1
        if activation.invocations == 3:
            raise RuntimeError("model failure")
        activation.records.extend([{"source_count": 8, "result_count": 4}] * 2)
        return generate(*args, **kwargs)

    facade.runner.generate = generating
    manifest_path = tmp_path / "compression.json"
    bridge = LMMSEvalRequestBridge(facade=facade, runtime=runtime, compression_manifest_path=str(manifest_path))
    requests, documents = zip(*(_request(doc_id=index) for index in range(3)))
    bridge.task_dict = {"fixture_task": {"test": list(documents)}}
    with pytest.raises(RuntimeError, match="model failure"):
        bridge.generate_until(list(requests))
    completed = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert completed["request_count"] == 2
    summary = completed["runtime"]["activations"][0]
    assert summary["invocation_count"] == summary["layout_delta_records_count"] == 2
    records = [json.loads(line) for line in Path(completed["trace"]["path"]).read_text(encoding="utf-8").splitlines()]
    assert len(records) == 2
    assert all(len(record["runtime"]["activations"][0]["layout_delta_records"]) == 2 for record in records)


def test_run_manifest_references_appended_compression_requests() -> None:
    from mllm_comp.core.manifest import BackendRecord

    config = experiment_config_from_dict({
        "experiment": {"name": "trace-fixture"},
        "backbone": {"name": "qwen25_omni", "model_id": "fixture"},
    })
    manifest = build_run_manifest(config, eval_backend=BackendRecord(
        name="lmms_eval", version="fixture", args={
            "compression_manifest_paths": ["compression.rank0.json"],
            "compression_manifests": [{"trace": {"path": "/run/compression.rank0.requests.jsonl"}}],
        },
    ))
    assert manifest.trace_paths["compression_runtime_0"] == "compression.rank0.json"
    assert manifest.trace_paths["compression_requests_0"] == "/run/compression.rank0.requests.jsonl"


def test_forward_probe_runs_one_no_cache_forward_and_records_non_quality_evidence(
    tmp_path: Path,
) -> None:
    torch = pytest.importorskip("torch")

    class _ForwardProcessorAdapter(_ProcessorAdapter):
        def prepare_inputs(
            self,
            processor: Any,
            raw_sample: Any,
            *,
            phase: str,
            runtime_mode: str = "default",
        ) -> ProcessedBatch:
            del processor
            self.samples.append(raw_sample)
            self.runtime_modes.append(runtime_mode)
            return ProcessedBatch(
                model_inputs={"input_ids": torch.tensor([[1, 2]])},
                sample_ids=[raw_sample.sample_id],
                prompt_lengths=[2],
                metadata={"phase": phase},
            )

        def detokenize(
            self,
            processor: Any,
            token_ids: Any,
            **kwargs: Any,
        ) -> list[str]:
            del processor, token_ids, kwargs
            return ["probe-token"]

    class _ForwardStateBuilder:
        def build(self, processed: ProcessedBatch, **kwargs: Any) -> ForwardState:
            del kwargs
            return ForwardState(native_model_inputs=dict(processed.model_inputs))

    class _ForwardRunner:
        def __init__(self) -> None:
            self.native_inputs: dict[str, Any] | None = None

        def forward_train(self, runtime: Any, state: ForwardState) -> Any:
            del runtime
            self.native_inputs = dict(state.native_model_inputs)
            logits = torch.zeros(1, 2, 5)
            logits[0, -1, 3] = 10
            return SimpleNamespace(logits=logits)

    facade = _Facade("qwen25_vl")
    facade.processor_adapter = _ForwardProcessorAdapter()
    facade.state_builder = _ForwardStateBuilder()
    facade.runner = _ForwardRunner()
    runtime = _runtime()
    runtime.metadata["compression"] = {
        "enabled": True,
        "activations": [{"invocation_count": 1}],
    }
    manifest_path = tmp_path / "forward_probe_compression.json"
    bridge = LMMSEvalRequestBridge(
        facade=facade,
        runtime=runtime,
        compression_manifest_path=str(manifest_path),
        request_mode="forward_probe",
    )
    sample = Qwen25VLSample(
        sample_id="probe:0",
        conversation=[
            {
                "role": "user",
                "content": [{"type": "text", "text": "What is present?"}],
            }
        ],
    )

    answer = bridge._generate_sample(
        sample,
        request_generation={"max_new_tokens": 16},
        context="What is present?",
        task="fixture_task",
        split="test",
        doc_id=0,
        request_mode="simple",
        cache_hook=None,
    )

    assert answer == "probe-token"
    assert facade.runner.native_inputs is not None
    assert facade.runner.native_inputs["use_cache"] is False
    assert facade.runner.native_inputs["return_dict"] is True
    completed = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert completed["last_request"]["request_mode"] == "forward_probe"
    assert completed["last_request"]["execution_phase"] == "forward"
    assert completed["last_request"]["benchmark_quality_claimed"] is False
    assert completed["last_request"]["next_token_id"] == 3


def test_request_diagnostic_filters_internal_chat_template_classification_data(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        model_bridge_module,
        "_runtime_environment",
        lambda _runtime: {"fixture": True},
    )

    class _TraceProcessorAdapter(_ProcessorAdapter):
        def prepare_inputs(
            self,
            processor: Any,
            raw_sample: Any,
            *,
            phase: str,
            runtime_mode: str = "default",
        ) -> ProcessedBatch:
            batch = super().prepare_inputs(
                processor,
                raw_sample,
                phase=phase,
                runtime_mode=runtime_mode,
            )
            batch.metadata.update(
                {
                    "chat_template_text": "rendered prompt",
                    "chat_template_text_by_sample": ["rendered prompt"],
                    "semantic_char_spans_by_sample": [
                        [{"start": 0, "end": 8, "label": "instruction"}]
                    ],
                    "semantic_message_prefixes_by_sample": [
                        [{"label": "instruction", "text": "rendered prompt"}]
                    ],
                    "semantic_token_labels_by_sample": [
                        [
                            "sys_special",
                            "system_prompt",
                            "system_prompt",
                            "ins_special",
                            "audio",
                            "audio",
                            "resp_special",
                        ]
                    ],
                }
            )
            return batch

    facade = _Facade()
    facade.processor_adapter = _TraceProcessorAdapter()
    trace_path = tmp_path / "diagnostic.jsonl"
    bridge = LMMSEvalRequestBridge(
        facade=facade,
        runtime=_runtime(),
        diagnostic_trace_path=str(trace_path),
    )
    request, document = _request()
    bridge.task_dict = {"fixture_task": {"test": [document]}}

    assert bridge.generate_until([request]) == ["answer"]
    record = json.loads(trace_path.read_text(encoding="utf-8"))
    assert "chat_template_text" not in record
    assert record["processor_metadata"]["chat_template_text"] == "rendered prompt"
    assert "chat_template_text_by_sample" not in record["processor_metadata"]
    assert "semantic_char_spans_by_sample" not in record["processor_metadata"]
    assert "semantic_message_prefixes_by_sample" not in record["processor_metadata"]
    assert "semantic_token_labels_by_sample" not in record["processor_metadata"]
    assert record["processor_metadata"]["semantic_token_span_format"] == {
        "coordinate_space": "processor_input_token_index",
        "interval": "[start,end)",
    }
    assert record["processor_metadata"]["semantic_token_spans_by_sample"] == [
        [
            {"start": 0, "end": 1, "length": 1, "label": "sys_special"},
            {"start": 1, "end": 3, "length": 2, "label": "system_prompt"},
            {"start": 3, "end": 4, "length": 1, "label": "ins_special"},
            {"start": 4, "end": 6, "length": 2, "label": "audio"},
            {"start": 6, "end": 7, "length": 1, "label": "resp_special"},
        ]
    ]


def test_request_bridge_default_mode_skips_runtime_environment_probe(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        model_bridge_module,
        "_runtime_environment",
        lambda _runtime: pytest.fail("default mode must not probe profile metadata"),
    )

    bridge = LMMSEvalRequestBridge(facade=_Facade(), runtime=_runtime())

    assert bridge.runtime_environment is None
    assert bridge.diagnostic_trace_path is None


def test_request_bridge_delegates_to_existing_runtime_and_maps_generation() -> None:
    facade = _Facade()
    bridge = LMMSEvalRequestBridge(
        facade=facade,
        runtime=_runtime(),
        generation_overrides={"top_p": 0.9, "use_cache": True},
    )
    request, document = _request(
        visual={"array": [0.0, 0.1, -0.1], "sampling_rate": 16000}
    )
    bridge.task_dict = {"fixture_task": {"test": [document]}}

    answers = bridge.generate_until([request])

    assert answers == ["answer"]
    assert facade.processor_adapter.runtime_modes == ["default"]
    sample = facade.processor_adapter.samples[0]
    assert sample.conversation[-1]["content"][0]["type"] == "audio"
    assert sample.conversation[-1]["content"][-1]["text"] == "What is present?"
    assert facade.runner.generation_kwargs == [
        {
            "return_audio": False,
            "thinker_max_new_tokens": 17,
            "thinker_do_sample": False,
            "thinker_use_cache": True,
        }
    ]


def test_chat_bridge_uses_doc_to_messages_and_lmms_media_kwargs(
    monkeypatch: Any,
) -> None:
    lmms_eval = ModuleType("lmms_eval")
    lmms_eval.__path__ = []  # type: ignore[attr-defined]
    protocol = ModuleType("lmms_eval.protocol")

    class _ChatMessages:
        def __init__(self, *, messages: list[dict[str, Any]]) -> None:
            self.messages = messages

        def to_hf_messages(
            self,
            *,
            image_kwargs: dict[str, Any],
            video_kwargs: dict[str, Any],
        ) -> list[dict[str, Any]]:
            normalized: list[dict[str, Any]] = []
            for message in self.messages:
                content = []
                for item in message["content"]:
                    if item["type"] == "text":
                        content.append({"type": "text", "text": item["text"]})
                    elif item["type"] == "image":
                        content.append(
                            {"type": "image", "image": item["url"], **image_kwargs}
                        )
                    elif item["type"] == "video":
                        content.append(
                            {"type": "video", "video": item["url"], **video_kwargs}
                        )
                    else:
                        content.append({"type": "audio", "audio": item["url"]})
                normalized.append({"role": message["role"], "content": content})
            return normalized

    protocol.ChatMessages = _ChatMessages  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "lmms_eval", lmms_eval)
    monkeypatch.setitem(sys.modules, "lmms_eval.protocol", protocol)

    facade = _Facade()
    bridge = LMMSEvalRequestBridge(facade=facade, runtime=_runtime())
    document = {"image": "fixture.png", "video": "fixture.mp4"}
    request = _Request(
        (
            "cache context",
            lambda doc: [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Compare both inputs."},
                        {"type": "image", "url": doc["image"]},
                        {"type": "video", "url": doc["video"]},
                    ],
                }
            ],
            {"max_new_tokens": 9, "temperature": 0},
            0,
            "fixture_task",
            "test",
        )
    )
    bridge.task_dict = {"fixture_task": {"test": [document]}}

    assert bridge.generate_until_chat([request]) == ["answer<STOP>ignored"]
    conversation = facade.processor_adapter.samples[0].conversation
    assert [message["role"] for message in conversation] == ["system", "user"]
    assert conversation[1]["content"][0]["text"] == "Compare both inputs."
    assert conversation[1]["content"][1] == {
        "type": "image",
        "image": "fixture.png",
        "max_pixels": 512 * 512,
    }
    assert conversation[1]["content"][2] == {
        "type": "video",
        "video": "fixture.mp4",
        "fps": 2,
        "max_frames": 16,
        "max_pixels": 384 * 384,
    }

    qwen3_facade = _Facade("qwen3_vl")
    qwen3_bridge = LMMSEvalRequestBridge(
        facade=qwen3_facade,
        runtime=_runtime(),
    )
    qwen3_bridge.task_dict = bridge.task_dict
    assert qwen3_bridge.generate_until_chat([request]) == ["answer<STOP>ignored"]
    qwen3_content = qwen3_facade.processor_adapter.samples[0].conversation[1]["content"]
    assert qwen3_content[1] == {"type": "image", "image": "fixture.png"}
    assert qwen3_content[2] == {
        "type": "video",
        "video": "fixture.mp4",
        "min_pixels": 256 * 28 * 28,
        "nframes": 32,
        "max_pixels": 1605632,
    }


def test_qwen3_vl_bridge_uses_platform_sample_and_native_generation_keys() -> None:
    facade = _Facade("qwen3_vl")
    bridge = LMMSEvalRequestBridge(
        facade=facade,
        runtime=_runtime(),
        generation_overrides={"top_p": 0.9},
    )
    request, document = _request(visual="fixture.png")
    bridge.task_dict = {"fixture_task": {"test": [document]}}

    assert bridge.generate_until([request]) == ["answer"]
    sample = facade.processor_adapter.samples[0]
    assert isinstance(sample, Qwen3VLSample)
    assert sample.conversation[-1]["content"][0] == {
        "type": "image",
        "image": "fixture.png",
        "min_pixels": 256 * 28 * 28,
        "max_pixels": 1605632,
    }
    assert facade.runner.generation_kwargs == [
        {
            "eos_token_id": 2,
            "pad_token_id": 99,
            "max_new_tokens": 17,
            "use_cache": True,
            "do_sample": False,
            "num_beams": 1,
        }
    ]


def test_qwen25_vl_bridge_matches_native_sample_and_generation_contract() -> None:
    facade = _Facade("qwen25_vl")
    bridge = LMMSEvalRequestBridge(
        facade=facade,
        runtime=_runtime(),
        generation_overrides={"max_new_tokens": 128, "temperature": 0.0},
    )
    document = {"media": "fixture.png"}
    request = _Request(
        (
            "<image>What is present?",
            {
                "max_new_tokens": 17,
                "temperature": 0.6,
                "top_p": 0.8,
                "num_beams": 2,
                "until": ["\n\n", "<STOP>"],
            },
            lambda doc: doc["media"],
            0,
            "fixture_task",
            "test",
        )
    )
    bridge.task_dict = {"fixture_task": {"test": [document]}}

    assert bridge.generate_until([request]) == ["answer"]
    sample = facade.processor_adapter.samples[0]
    assert isinstance(sample, Qwen25VLSample)
    assert sample.conversation[-1]["content"] == [
        {
            "type": "image",
            "image": "fixture.png",
            "min_pixels": 256 * 28 * 28,
            "max_pixels": 1605632,
        },
        {"type": "text", "text": "What is present?"},
    ]
    assert facade.runner.generation_kwargs == [
        {
            "eos_token_id": 2,
            "pad_token_id": 99,
            "do_sample": True,
            "temperature": 0.6,
            "top_p": 0.8,
            "num_beams": 2,
            "max_new_tokens": 17,
            "use_cache": True,
        }
    ]


@pytest.mark.parametrize(
    ("backbone_name", "expected_max_new_tokens"),
    [("qwen25_vl", 32768), ("qwen3_vl", 128)],
)
def test_qwen_vl_native_defaults_are_not_replaced_by_platform_fallbacks(
    backbone_name: str,
    expected_max_new_tokens: int,
) -> None:
    bridge = LMMSEvalRequestBridge(
        facade=_Facade(backbone_name),
        runtime=_runtime(),
    )

    kwargs, stops = bridge._generation_kwargs({})

    assert kwargs["max_new_tokens"] == expected_max_new_tokens
    assert kwargs["eos_token_id"] == 2
    assert kwargs["pad_token_id"] == 99
    assert kwargs["use_cache"] is True
    assert kwargs["do_sample"] is False
    assert kwargs["num_beams"] == 1
    assert stops == ["<eos:2>"]


@pytest.mark.parametrize("backbone_name", ["qwen25_vl", "qwen3_vl"])
def test_qwen_vl_bridge_matches_native_request_order(backbone_name: str) -> None:
    facade = _Facade(backbone_name)
    bridge = LMMSEvalRequestBridge(facade=facade, runtime=_runtime())
    documents = [{"media": None} for _ in range(3)]
    requests = [
        _Request(
            (
                context,
                {"max_new_tokens": 1},
                lambda doc: doc["media"],
                doc_id,
                "fixture_task",
                "test",
            )
        )
        for doc_id, context in enumerate(
            ["one", "one two three", "one two"]
        )
    ]
    bridge.task_dict = {"fixture_task": {"test": documents}}

    assert len(bridge.generate_until(requests)) == 3
    assert [
        sample.conversation[-1]["content"][-1]["text"]
        for sample in facade.processor_adapter.samples
    ] == ["one two three", "one two", "one"]


def test_visual_bridge_groups_request_order_by_generation_kwargs() -> None:
    facade = _Facade("qwen25_vl")
    bridge = LMMSEvalRequestBridge(facade=facade, runtime=_runtime())
    documents = [{"media": None} for _ in range(4)]
    cases = [
        ("one", {"max_new_tokens": 1, "until": ["<A>"]}),
        ("one two three", {"max_new_tokens": 2, "until": ["<B>"]}),
        ("one two", {"max_new_tokens": 1, "until": ["<A>"]}),
        ("one two three four", {"max_new_tokens": 2, "until": ["<B>"]}),
    ]
    requests = [
        _Request(
            (
                context,
                generation_kwargs,
                lambda doc: doc["media"],
                doc_id,
                "fixture_task",
                "test",
            )
        )
        for doc_id, (context, generation_kwargs) in enumerate(cases)
    ]
    bridge.task_dict = {"fixture_task": {"test": documents}}

    assert len(bridge.generate_until(requests)) == 4
    assert [
        sample.conversation[-1]["content"][-1]["text"]
        for sample in facade.processor_adapter.samples
    ] == ["one two", "one", "one two three four", "one two three"]


def test_llava_onevision_bridge_uses_simple_image_sample_and_no_system_prompt() -> None:
    facade = _Facade("llava_onevision")
    bridge = LMMSEvalRequestBridge(facade=facade, runtime=_runtime())
    image = SimpleNamespace(size=(384, 256))
    request, document = _request(visual=[image])
    bridge.task_dict = {"fixture_task": {"test": [document]}}

    assert bridge.generate_until([request]) == ["answer<STOP>ignored"]
    sample = facade.processor_adapter.samples[0]
    assert isinstance(sample, LlavaOneVisionSample)
    assert sample.prompt == "What is present?"
    assert sample.images == (image,)
    assert bridge.system_prompt is None
    assert facade.runner.generation_kwargs == [
        {"max_new_tokens": 17, "use_cache": True, "pad_token_id": 99}
    ]


def test_llava_onevision_bridge_routes_one_video_to_native_video_field() -> None:
    facade = _Facade("llava_onevision")
    bridge = LMMSEvalRequestBridge(facade=facade, runtime=_runtime())
    video = "D:/datasets/nextqa/clip.mp4"
    request, document = _request(visual=[video])
    bridge.task_dict = {"fixture_task": {"test": [document]}}

    assert bridge.generate_until([request]) == ["answer<STOP>ignored"]
    sample = facade.processor_adapter.samples[0]
    assert isinstance(sample, LlavaOneVisionSample)
    assert sample.images == ()
    assert sample.video == video

    with pytest.raises(BackendIntegrationError, match="multi-video/mixed"):
        bridge.build_sample(
            context="mixed",
            visual=[video, SimpleNamespace(size=(10, 10))],
            sample_id="mixed",
        )
    with pytest.raises(BackendIntegrationError, match="exactly one media type"):
        bridge.build_sample(
            context="ambiguous",
            visual=[{"video": video, "image": SimpleNamespace(size=(10, 10))}],
            sample_id="ambiguous",
        )


def test_llava_onevision_bridge_preserves_native_generation_defaults() -> None:
    facade = _Facade("llava_onevision")
    bridge = LMMSEvalRequestBridge(facade=facade, runtime=_runtime())

    kwargs, stops = bridge._generation_kwargs(
        {
            "until": ["<STOP>"],
            "temperature": 0.7,
            "top_p": 0.8,
            "top_k": 20,
        }
    )

    assert kwargs == {
        "max_new_tokens": 1024,
        "use_cache": True,
        "pad_token_id": 99,
    }
    assert stops == []
    assert "do_sample" not in kwargs


def test_llava_onevision_task_generation_values_win_over_platform_defaults() -> None:
    bridge = LMMSEvalRequestBridge(
        facade=_Facade("llava_onevision"),
        runtime=_runtime(),
        generation_overrides={
            "max_new_tokens": 128,
            "do_sample": False,
            "temperature": 0.0,
        },
    )

    kwargs, _ = bridge._generation_kwargs(
        {"max_new_tokens": 17, "do_sample": True, "temperature": 0.6}
    )

    assert kwargs == {
        "max_new_tokens": 17,
        "do_sample": True,
        "temperature": 0.6,
        "use_cache": True,
        "pad_token_id": 99,
    }


def test_llava_onevision_bridge_matches_native_request_order() -> None:
    facade = _Facade("llava_onevision")
    bridge = LMMSEvalRequestBridge(facade=facade, runtime=_runtime())
    documents = [{"media": None} for _ in range(3)]
    requests = [
        _Request(
            (
                context,
                {"max_new_tokens": 1},
                lambda doc: doc["media"],
                doc_id,
                "fixture_task",
                "test",
            )
        )
        for doc_id, context in enumerate(
            ["one", "one two three", "one two"]
        )
    ]
    bridge.task_dict = {"fixture_task": {"test": documents}}

    assert bridge.generate_until(requests) == [
        "answer<STOP>ignored",
        "answer<STOP>ignored",
        "answer<STOP>ignored",
    ]
    assert [sample.prompt for sample in facade.processor_adapter.samples] == [
        "one two three",
        "one two",
        "one",
    ]


def test_qwen3_vl_bridge_rejects_audio_and_audio_in_video() -> None:
    facade = _Facade("qwen3_vl")
    with pytest.raises(BackendIntegrationError, match="audio-in-video"):
        LMMSEvalRequestBridge(
            facade=facade,
            runtime=_runtime(),
            use_audio_in_video=True,
        )

    bridge = LMMSEvalRequestBridge(facade=facade, runtime=_runtime())
    request, document = _request(visual="fixture.wav")
    bridge.task_dict = {"fixture_task": {"test": [document]}}
    with pytest.raises(BackendIntegrationError, match="cannot contain audio"):
        bridge.generate_until([request])


def test_bridge_rejects_unsupported_contracts_explicitly() -> None:
    bridge = LMMSEvalRequestBridge(facade=_Facade(), runtime=_runtime())

    with pytest.raises(BackendIntegrationError, match="loglikelihood"):
        bridge.loglikelihood([])
    with pytest.raises(BackendIntegrationError, match="Multi-round"):
        bridge.generate_until_multi_round([])


@pytest.mark.parametrize("requires_teacher_forcing", [False, True])
def test_analysis_iterator_uses_lmms_task_and_platform_processor(requires_teacher_forcing, monkeypatch) -> None:
    # The installed MMAU task returns the native type/url form, not a bare path.
    media = [{"type": "audio", "url": "/fixture/mmau.wav"}]

    class _TaskManager:
        def __init__(self, **kwargs: Any) -> None:
            self.kwargs = kwargs

    class _Task:
        def get_config(self, key: str) -> str:
            assert key == "output_type"
            return "generate_until"

        def doc_iterator(self, *, limit: int | None) -> Any:
            assert limit == 1
            yield 4, {"prompt": "Describe", "media": media}

        def doc_to_text(self, doc: dict[str, Any]) -> str:
            return doc["prompt"]

        def doc_to_visual(self, doc: dict[str, Any]) -> Any:
            return doc["media"]

        def doc_to_target(self, doc: dict[str, Any]) -> str:
            del doc
            return "reference answer"

    facade = _Facade()
    if not requires_teacher_forcing:
        def unexpected_teacher_forcing(*args, **kwargs):
            pytest.fail("A prompt-only analysis must not preprocess the answer.")
        monkeypatch.setattr(facade.processor_adapter, "prepare_teacher_forced_inputs", unexpected_teacher_forcing)
    adapter = LMMSEvalAdapter(
        task_manager_cls=_TaskManager,
        task_dict_factory=lambda tasks, manager, task_type: {"fixture_task": _Task()},
    )
    config = experiment_config_from_dict(
        {
            "experiment": {"name": "analysis"},
            "backbone": {"name": "qwen25_omni"},
            "pipeline": {"stages": []},
            "analysis": {"tasks": ["fixture_task"], "sample_limit": 1},
            "train": {},
            "eval": {},
            "data": {"dataset_id": "fixture"},
            "output": {"run_dir": "runs/a", "manifest_path": "runs/a/manifest.json"},
        }
    )

    samples = list(
        adapter.iter_analysis_samples(
            config=config,
            backbone=facade,
            runtime=_runtime(),
            sample_limit=None,
            requires_teacher_forcing=requires_teacher_forcing,
        )
    )

    assert samples[0]["sample_id"] == "fixture_task:4"
    assert samples[0]["task_name"] == "fixture_task"
    assert samples[0]["processor_outputs"] == {"input_ids": [[1, 2]]}
    assert samples[0]["metadata"]["prompt"] == "Describe"
    assert samples[0]["target_text"] == "reference answer"
    assert samples[0]["target_source"] == "lmms_eval.doc_to_target"
    assert facade.processor_adapter.samples[0].conversation[-1]["content"] == [
        {"type": "audio", "audio": "/fixture/mmau.wav"},
        {"type": "text", "text": "Describe"},
    ]
    assert media == [{"type": "audio", "url": "/fixture/mmau.wav"}]
    if not requires_teacher_forcing:
        assert "teacher_forced_processor_outputs" not in samples[0]
        assert "objective" not in samples[0]
        return
    assert samples[0]["teacher_forced_processor_outputs"]["labels"] == [
        [-100, -100, 3]
    ]
    assert samples[0]["objective"]["objective"] == (
        "teacher_forced_answer_cross_entropy_v1"
    )


def test_adapter_launches_native_cli_and_collects_native_outputs(
    tmp_path: Path,
) -> None:
    calls: list[Any] = []

    def command_runner(plan: Any) -> NativeCLIResult:
        calls.append(plan)
        native_dir = Path(plan.output_path) / "fixture"
        native_dir.mkdir(parents=True)
        (native_dir / "2026-01-01T00-00-00_samples_fixture_task.jsonl").write_text(
            '{"response":"yes"}\n',
            encoding="utf-8",
        )
        (native_dir / "2026-01-01T00-00-00_results.json").write_text(
            json.dumps(
                {
                    "results": {
                        "fixture_task": {
                            "exact_match,none": 0.75,
                            "alias": "fixture",
                        }
                    },
                    "configs": {"fixture_task": {"test_split": "test"}},
                    "n-samples": {
                        "fixture_task": {"original": 8, "effective": 2}
                    },
                }
            ),
            encoding="utf-8",
        )
        return NativeCLIResult(returncode=0)

    adapter = LMMSEvalAdapter(command_runner=command_runner)
    output_path = tmp_path / "lmms_eval"
    config = experiment_config_from_dict(
        {
            "experiment": {"name": "eval", "seed": 7},
            "backbone": {"name": "qwen25_omni", "model_id": "fixture"},
            "pipeline": {"stages": []},
            "analysis": {},
            "train": {},
            "eval": {
                "backend": "lmms_eval",
                "tasks": ["fixture_task"],
                "runtime_batch_size": 1,
                "sample_limit": 2,
                "generation": {"max_new_tokens": 8},
                "args": {
                    "log_samples": True,
                    "verbosity": "DEBUG",
                    "include_path": "tasks/local",
                },
            },
            "data": {"dataset_id": "fixture"},
            "output": {
                "run_dir": str(output_path.parent),
                "manifest_path": str(tmp_path / "manifest.json"),
            },
        },
        path=str(tmp_path / "fixture.yaml"),
    )

    result = adapter.evaluate(
        config=config,
        device="cuda:0",
    )

    argv = calls[0].argv
    assert argv[:3] == [argv[0], "-m", "lmms_eval"]
    assert argv[argv.index("--model") + 1] == "qwen2_5_omni"
    assert "--force_simple" not in argv
    assert argv[argv.index("--tasks") + 1] == "fixture_task"
    assert argv[argv.index("--seed") + 1] == "7"
    assert argv[argv.index("--limit") + 1] == "2"
    assert "--log_samples" in argv
    assert argv[argv.index("--verbosity") + 1] == "DEBUG"
    assert argv[argv.index("--include_path") + 1] == "tasks/local"
    assert result.metrics[0].metrics == {"exact_match,none": 0.75}
    assert result.metrics[0].split == "test"
    assert result.metrics[0].num_samples == 2
    assert result.raw_output_path.endswith("_results.json")
    assert result.sample_output_paths[0].endswith("_samples_fixture_task.jsonl")
    assert result.backend is not None and result.backend.name == "lmms_eval"
    assert result.backend.args["argv"] == argv
    assert result.backend.args["lmms_eval_argv"] == argv
    assert result.backend.args["num_processes"] == 1
    assert result.backend.args["distributed_launcher"] is None
    assert result.backend.args["result_path"] == result.raw_output_path
    assert result.backend.args["plugin_spec"] == calls[0].plugin_spec
    assert calls[0].plugin_spec["runtime_mode"] == "default"
    assert calls[0].plugin_spec["diagnostic_trace_path"] is None
    assert result.backend.args["diagnostic_trace_paths"] == []
    assert result.backend.args["diagnostic_pretty_paths"] == []
    manifest = build_run_manifest(config, eval_backend=result.backend)
    assert manifest.trace_paths == {}


def test_launch_plan_accepts_native_lmms_seed_tuple(tmp_path: Path) -> None:
    config = experiment_config_from_dict(
        {
            "experiment": {"name": "eval", "seed": 42},
            "backbone": {"name": "llava_onevision", "model_id": "fixture"},
            "pipeline": {"stages": []},
            "analysis": {},
            "train": {},
            "eval": {
                "backend": "lmms_eval",
                "tasks": ["fixture_task"],
                "args": {"seed": [0, 1234, 1234, 1234]},
            },
            "data": {},
            "output": {
                "run_dir": str(tmp_path / "run"),
                "manifest_path": str(tmp_path / "manifest.json"),
            },
        },
        path=str(tmp_path / "fixture.yaml"),
    )

    plan = LMMSEvalAdapter().build_launch_plan(config)

    assert plan.lmms_eval_argv[plan.lmms_eval_argv.index("--seed") + 1] == (
        "0,1234,1234,1234"
    )


def test_compressed_eval_collects_runtime_manifest_into_run_manifest(
    tmp_path: Path,
) -> None:
    def command_runner(plan: Any) -> NativeCLIResult:
        native_dir = Path(plan.output_path) / "fixture"
        native_dir.mkdir(parents=True)
        (native_dir / "run_results.json").write_text(
            json.dumps({"results": {"fixture_task": {"score": 1.0}}}),
            encoding="utf-8",
        )
        compression_path = Path(plan.plugin_spec["compression_manifest_path"])
        compression_path.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "enabled": True,
                    "runtime": {"executor_family": "projector_token"},
                    "last_request": {
                        "compression_mode": "compressed",
                        "compression": {
                            "original_audio_tokens": 20,
                            "kept_audio_tokens": 10,
                        },
                    },
                }
            ),
            encoding="utf-8",
        )
        return NativeCLIResult(returncode=0)

    output_path = tmp_path / "compressed_eval"
    config = experiment_config_from_dict(
        {
            "experiment": {"name": "compressed-eval"},
            "backbone": {"name": "qwen25_omni", "model_id": "fixture"},
            "pipeline": {
                "stages": [
                    {
                        "name": "audio",
                        "placement": "projector_level",
                        "execution_phase": "generate",
                        "plugin": "segmentwise_topk_audio",
                        "method": "token_compression",
                        "object_family": "tensor_slot",
                        "modalities": ["audio"],
                        "token_roles": ["observation"],
                        "schedule": {"keep_ratio": 0.5},
                    }
                ]
            },
            "analysis": {},
            "train": {},
            "eval": {
                "backend": "lmms_eval",
                "compression_mode": "compressed",
                "tasks": ["fixture_task"],
            },
            "data": {"dataset_id": "fixture"},
            "output": {
                "run_dir": str(output_path),
                "manifest_path": str(output_path / "manifest.json"),
            },
        },
        path=str(tmp_path / "compressed.yaml"),
    )
    result = LMMSEvalAdapter(command_runner=command_runner).evaluate(config=config)

    evidence_path = result.backend_args["compression_manifest_paths"][0]
    assert result.backend_args["compression_manifests"][0]["last_request"][
        "compression"
    ]["kept_audio_tokens"] == 10
    manifest = build_run_manifest(config, eval_backend=result.backend)
    assert manifest.trace_paths["compression_runtime_0"] == evidence_path


def test_launch_plan_rejects_platform_arg_conflicts_and_retired_api_keys(
    tmp_path: Path,
) -> None:
    base = {
        "experiment": {"name": "eval"},
        "backbone": {"name": "qwen25_omni", "model_id": "fixture"},
        "pipeline": {"stages": []},
        "analysis": {},
        "train": {},
        "eval": {"tasks": ["mme"], "args": {"model": "other"}},
        "data": {"dataset_id": "fixture"},
        "output": {
            "run_dir": str(tmp_path),
            "manifest_path": str(tmp_path / "manifest.json"),
        },
    }
    adapter = LMMSEvalAdapter()
    with pytest.raises(BackendIntegrationError, match="platform-owned"):
        adapter.build_launch_plan(
            experiment_config_from_dict(base, path=str(tmp_path / "config.yaml"))
        )

    base["eval"]["args"] = {"result_filename": "legacy.json"}
    with pytest.raises(BackendIntegrationError, match="retired direct-API"):
        adapter.build_launch_plan(
            experiment_config_from_dict(base, path=str(tmp_path / "config.yaml"))
        )


def test_launch_plan_adds_force_simple_only_when_explicitly_enabled(
    tmp_path: Path,
) -> None:
    base = {
        "experiment": {"name": "eval"},
        "backbone": {"name": "qwen25_omni", "model_id": "fixture"},
        "pipeline": {"stages": []},
        "analysis": {},
        "train": {},
        "eval": {"tasks": ["mme"], "args": {}},
        "data": {"dataset_id": "fixture"},
        "output": {
            "run_dir": str(tmp_path),
            "manifest_path": str(tmp_path / "manifest.json"),
        },
    }
    adapter = LMMSEvalAdapter()

    default_plan = adapter.build_launch_plan(
        experiment_config_from_dict(base, path=str(tmp_path / "default.yaml"))
    )
    assert "--force_simple" not in default_plan.argv

    base["eval"]["args"] = {"force_simple": True}
    simple_plan = adapter.build_launch_plan(
        experiment_config_from_dict(base, path=str(tmp_path / "simple.yaml"))
    )
    assert simple_plan.argv.count("--force_simple") == 1

    base["eval"]["args"] = {"force_simple": "true"}
    with pytest.raises(BackendIntegrationError, match="must be a boolean"):
        adapter.build_launch_plan(
            experiment_config_from_dict(base, path=str(tmp_path / "invalid.yaml"))
        )


def test_launch_plan_marks_forward_probe_and_keeps_flag_out_of_native_cli(
    tmp_path: Path,
) -> None:
    raw = {
        "experiment": {"name": "forward-probe"},
        "backbone": {"name": "qwen25_vl", "model_id": "fixture"},
        "pipeline": {"stages": []},
        "analysis": {},
        "train": {},
        "eval": {
            "tasks": ["mme"],
            "runtime_batch_size": 1,
            "args": {"forward_only_probe": True, "force_simple": True},
        },
        "data": {},
        "output": {
            "run_dir": str(tmp_path / "forward-probe"),
            "manifest_path": str(tmp_path / "forward-probe" / "manifest.json"),
        },
    }
    adapter = LMMSEvalAdapter()
    plan = adapter.build_launch_plan(
        experiment_config_from_dict(raw, path=str(tmp_path / "probe.yaml"))
    )

    assert plan.plugin_spec["request_mode"] == "forward_probe"
    assert "--forward_only_probe" not in plan.lmms_eval_argv
    assert "--force_simple" in plan.lmms_eval_argv

    raw["eval"]["args"]["forward_only_probe"] = "true"
    with pytest.raises(BackendIntegrationError, match="must be a boolean"):
        adapter.build_launch_plan(
            experiment_config_from_dict(raw, path=str(tmp_path / "invalid.yaml"))
        )


def test_forward_probe_evidence_requires_invocation_and_layout_delta() -> None:
    payload = {
        "runtime": {
            "activations": [
                {"invocation_count": 1, "layout_delta_records": []}
            ]
        }
    }
    assert lmms_eval_adapter_module._has_compression_invocation(payload) is False

    payload["runtime"]["activations"][0] = {
        "invocation_count": 0,
        "layout_delta_records": [{"source_count": 10, "result_count": 5}],
    }
    assert lmms_eval_adapter_module._has_compression_invocation(payload) is False

    payload["runtime"]["activations"][0]["invocation_count"] = 1
    assert lmms_eval_adapter_module._has_compression_invocation(payload) is True


def test_segmentwise_topk_config_builds_dynamic_compressed_launch_plan(
    monkeypatch: Any,
) -> None:
    monkeypatch.setenv("QWEN25_OMNI_MODEL_PATH", "/models/qwen25-omni")
    config_path = (
        Path(__file__).resolve().parents[2]
        / "configs"
        / "eval"
        / "lmms_eval"
        / "qwen25_omni_aud_segmentwise_topk.yaml"
    )

    config = load_config(config_path)
    result = run_experiment(config, task="eval", plan_only=True)
    assert result.status == "planned", result.error
    plan = result.output["plan"]

    assert config.pipeline.stages[0].plugin == "segmentwise_topk_audio"
    assert plan.plugin_spec["compression_mode"] == "compressed"
    assert plan.plugin_spec["runtime_mode"] == "default"
    assert plan.plugin_spec["diagnostic_trace_path"] is None
    assert plan.plugin_spec["compression_manifest_path"].endswith(
        "compression_runtime.json"
    )
    assert any(
        path.endswith("compression_runtime.json")
        for path in plan.expected_output_paths
    )
    assert "--log_samples" in plan.argv


def test_qwen_audio_profile_gets_rank_safe_timing_trace(
    monkeypatch: Any,
) -> None:
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    monkeypatch.setenv("QWEN25_OMNI_MODEL_PATH", "/models/qwen25-omni")
    config_path = (
        Path(__file__).resolve().parents[2]
        / "configs"
        / "eval"
        / "lmms_eval"
        / "qwen25_omni_aud.yaml"
    )

    config = load_config(config_path)
    config.eval.num_processes = 2
    config.eval.runtime_mode = RuntimeMode.PROFILE
    result = run_experiment(config, task="eval", plan_only=True)
    assert result.status == "planned", result.error
    plan = result.output["plan"]

    assert plan.plugin_spec["compression_mode"] == "none"
    assert plan.plugin_spec["runtime_mode"] == "profile"
    assert plan.plugin_spec["diagnostic_trace_path"].endswith(
        "baseline_diagnostic.jsonl"
    )
    assert any(
        path.endswith("baseline_diagnostic.rank*.jsonl")
        for path in plan.expected_output_paths
    )


def test_qwen_audio_debug_gets_token_count_trace(monkeypatch: Any) -> None:
    monkeypatch.setenv("QWEN25_OMNI_MODEL_PATH", "/models/qwen25-omni")
    config_path = (
        Path(__file__).resolve().parents[2]
        / "configs"
        / "eval"
        / "lmms_eval"
        / "qwen25_omni_aud_segmentwise_topk.yaml"
    )

    config = load_config(config_path)
    config.eval.runtime_mode = RuntimeMode.DEBUG
    result = run_experiment(config, task="eval", plan_only=True)
    assert result.status == "planned", result.error
    plan = result.output["plan"]

    assert plan.plugin_spec["runtime_mode"] == "debug"
    assert plan.plugin_spec["diagnostic_trace_path"].endswith(
        "debug_diagnostic.jsonl"
    )
    assert any(
        path.endswith("debug_diagnostic.pretty.json")
        for path in plan.expected_output_paths
    )


def test_pretty_diagnostic_compacts_shapes_and_wraps_numeric_arrays(
    tmp_path: Path,
) -> None:
    trace_path = tmp_path / "debug_diagnostic.jsonl"
    payload = {
        "sample_id": "样本-1",
        "input_shape": list(range(12)),
        "token_counts": {"audio": 250, "unknown": 0},
        "kept_indices": list(range(11)),
        "long_mapping": {f"field_{index:02d}": index for index in range(20)},
    }
    raw = json.dumps(payload, ensure_ascii=False) + "\n"
    trace_path.write_text(raw, encoding="utf-8")

    [pretty_path] = lmms_eval_diagnostics._materialize_pretty_diagnostics([trace_path])
    content = pretty_path.read_text(encoding="utf-8")
    pretty_payload = json.loads(content)[0]
    assert pretty_payload == {**payload, "token_counts": {"audio": 250}}
    assert trace_path.read_text(encoding="utf-8") == raw
    assert '"input_shape": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]' in content
    assert '"token_counts": {"audio": 250}' in content
    lines = [line.strip() for line in content.splitlines()]
    assert '"long_mapping": {' in lines
    kept_start = lines.index('"kept_indices": [')
    assert lines[kept_start + 1 : kept_start + 3] == [
        "0, 1, 2, 3, 4, 5, 6, 7, 8, 9,",
        "10",
    ]


def test_pretty_diagnostic_compacts_repeated_debug_layers_and_modules(
    tmp_path: Path,
) -> None:
    def phase(input_tokens: int, output_tokens: int, *, calls: int = 1) -> dict:
        return {
            "calls": calls,
            "input": {"total_tokens": input_tokens},
            "output": {"total_tokens": output_tokens},
        }

    prompt_counts = {"instruction": 2, "audio": 4}
    prefill = {
        "calls": 1,
        "input": {"total_tokens": 6, "token_counts": prompt_counts},
        "output": {"total_tokens": 6, "token_counts": prompt_counts},
    }
    payload = {
        "generation_metadata": {
            "token_debug": {
                "mode": "debug",
                "stage_snapshots": [{
                    "stage": "post_legalizer_llm_input",
                    "token_counts": prompt_counts,
                }],
                "layers": [
                    *[{
                        "stage": "audio_encoder", "layer_index": index,
                        "module_path": f"thinker.audio_tower.layers.{index}",
                        "encode": phase(4, output_tokens),
                    } for index, output_tokens in enumerate((4, 4, 3))],
                    *[{
                        "stage": "llm", "layer_index": index,
                        "prefill": prefill, "decode": phase(2, 2, calls=2),
                    } for index in (0, 1)],
                ],
                "modules": [
                    {"stage": "audio_post_norm", "encode": phase(4, 4)},
                    {"stage": "audio_projector", "encode": phase(4, 2)},
                ],
            },
        },
    }
    trace_path = tmp_path / "debug_diagnostic.jsonl"
    raw = json.dumps(payload) + "\n"
    trace_path.write_text(raw, encoding="utf-8")

    [pretty_path] = lmms_eval_diagnostics._materialize_pretty_diagnostics([trace_path])
    pretty_debug = json.loads(pretty_path.read_text(encoding="utf-8"))[0][
        "generation_metadata"
    ]["token_debug"]

    assert pretty_debug["layers"] == [
        {"stage": "audio_encoder", "layer_range": [0, 1], "tokens": 4},
        {"stage": "audio_encoder", "layer": 2, "input_tokens": 4, "output_tokens": 3},
        {
            "stage": "llm", "layer_range": [0, 1],
            "prefill": {"tokens": 6, "token_counts_ref": "post_legalizer_llm_input"},
            "decode": {"calls": 2, "tokens": 2},
        },
    ]
    assert pretty_debug["modules"] == [
        {"stage": "audio_projector", "input_tokens": 4, "output_tokens": 2},
    ]
    assert trace_path.read_text(encoding="utf-8") == raw


def test_segmentwise_paper_sweep_template_resolves_one_setting(
    monkeypatch: Any,
) -> None:
    monkeypatch.setenv("QWEN25_OMNI_MODEL_PATH", "/models/qwen25-omni")
    monkeypatch.setenv("SEGMENTWISE_KEEP_RATIO", "0.10")
    monkeypatch.setenv("SEGMENTWISE_RUN_TAG", "r10")
    config_path = (
        Path(__file__).resolve().parents[2]
        / "configs"
        / "eval"
        / "lmms_eval"
        / "qwen25_omni_aud_segmentwise_topk_paper.yaml"
    )

    config = load_config(config_path)
    plan = LMMSEvalAdapter().build_launch_plan(config)

    assert config.pipeline.stages[0].schedule["allocation"] == "paper_floor"
    assert config.pipeline.stages[0].schedule["keep_ratio"] == 0.10
    assert config.output.run_dir.endswith("paper_r10_s10")
    assert plan.plugin_spec["runtime_mode"] == "default"
    assert plan.plugin_spec["diagnostic_trace_path"] is None


def test_qwen_backbones_share_plugin_class_but_preserve_native_model_identity(
    tmp_path: Path,
) -> None:
    adapter = LMMSEvalAdapter()
    model_names: list[str] = []
    for backbone_name in (
        "qwen25_omni",
        "qwen25_vl",
        "qwen3_vl",
        "llava_onevision",
    ):
        config = experiment_config_from_dict(
            {
                "experiment": {"name": backbone_name},
                "backbone": {"name": backbone_name, "model_id": "fixture"},
                "pipeline": {"stages": []},
                "analysis": {},
                "train": {},
                "eval": {"tasks": ["mme"]},
                "data": {"dataset_id": "fixture"},
                "output": {
                    "run_dir": str(tmp_path / backbone_name),
                    "manifest_path": str(tmp_path / backbone_name / "manifest.json"),
                },
            },
            path=str(tmp_path / f"{backbone_name}.yaml"),
        )
        model_names.append(adapter.build_launch_plan(config).model_name)

    assert model_names == [
        "qwen2_5_omni",
        "mllm_comp_qwen25_vl",
        "mllm_comp_qwen3_vl",
        "mllm_comp_llava_onevision",
    ]
