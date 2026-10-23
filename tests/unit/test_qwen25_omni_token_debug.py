"""Token-count-only debug instrumentation for Qwen2.5-Omni."""

from __future__ import annotations

import pytest

import json
from types import SimpleNamespace
from typing import Any

from mllm_comp.backbones.base import ProcessedBatch
from mllm_comp.backbones.qwen25_omni.token_debug import (
    Qwen25OmniTokenDebugCollector,
    TOKEN_TYPES,
    _apply_modality_special_labels,
    _modality_special_sets,
)
from mllm_comp.core.state import ForwardState, TokenSpan


class _ShapeValue:
    def __init__(self, *shape: int) -> None:
        self.shape = shape


class _Handle:
    def __init__(self, module: "_HookModule") -> None:
        self.module = module

    def remove(self) -> None:
        self.module.hook = None


class _HookModule:
    def __init__(self) -> None:
        self.hook: Any = None

    def register_forward_hook(self, hook: Any, *, with_kwargs: bool) -> _Handle:
        assert with_kwargs is True
        self.hook = hook
        return _Handle(self)

    def emit(self, input_shape: tuple[int, ...], output_shape: tuple[int, ...]) -> None:
        assert self.hook is not None
        self.hook(
            self,
            (_ShapeValue(*input_shape),),
            {},
            _ShapeValue(*output_shape),
        )


def _model() -> tuple[Any, dict[str, list[_HookModule] | _HookModule]]:
    audio_layers = [_HookModule(), _HookModule()]
    llm_layers = [_HookModule(), _HookModule()]
    modules: dict[str, list[_HookModule] | _HookModule] = {
        "audio_layers": audio_layers,
        "llm_layers": llm_layers,
        "pooler": _HookModule(),
        "post_norm": _HookModule(),
        "projector": _HookModule(),
    }
    audio_tower = SimpleNamespace(
        layers=audio_layers,
        avg_pooler=modules["pooler"],
        ln_post=modules["post_norm"],
        proj=modules["projector"],
    )
    model = SimpleNamespace(
        thinker=SimpleNamespace(
            audio_tower=audio_tower,
            model=SimpleNamespace(layers=llm_layers),
        )
    )
    return model, modules


def _assert_count_record_is_conserved(record: dict[str, Any]) -> None:
    counts = record["token_counts"]
    assert tuple(counts) == TOKEN_TYPES
    assert "total" not in counts
    assert sum(counts.values()) == record["total_tokens"]


def _assert_report_is_conserved(report: dict[str, Any]) -> None:
    for snapshot in report["stage_snapshots"]:
        _assert_count_record_is_conserved(snapshot)
        for sample in snapshot.get("per_sample", []):
            _assert_count_record_is_conserved(sample)
    for module in [*report["layers"], *report["modules"]]:
        for phase_name in ("encode", "prefill", "decode"):
            phase = module.get(phase_name)
            if phase is None:
                continue
            for side_name in ("input", "output"):
                side = phase[side_name]
                _assert_count_record_is_conserved(side)
                assert side["classified_tokens"] == side["total_tokens"]
                assert side["classification_delta"] == 0


def test_debug_collector_reports_stage_and_every_layer_token_counts() -> None:
    model, modules = _model()
    collector = Qwen25OmniTokenDebugCollector(model)
    collector.install()
    batch = ProcessedBatch(
        model_inputs={
            "input_ids": [[99, 1, 13, 12, 12, 12, 12, 14, 5, 0]],
            "attention_mask": [[1, 1, 1, 1, 1, 1, 1, 1, 1, 0]],
            "feature_attention_mask": [[1, 1, 1, 0]],
        },
        sample_ids=["sample-0"],
        prompt_lengths=[9],
        metadata={
            "expected_token_counts_by_sample": [{"audio": 4, "image": 0, "video": 0}],
            "semantic_token_labels_by_sample": [[
                "global_special", "ins_special", "audio_special",
                "audio", "audio", "audio", "audio", "audio_special", "resp_special", "pad",
            ]],
            "semantic_classification_source": "fixture",
            "modality_counts_by_sample": [{"audio": 1, "image": 0, "video": 0}],
        },
    )
    original_state = ForwardState(
        input_ids=batch.model_inputs["input_ids"],
        attention_mask=batch.model_inputs["attention_mask"],
    )
    active_state = ForwardState(
        input_ids=[[99, 1, 13, 12, 12, 14, 5, 0]],
        attention_mask=[[1, 1, 1, 1, 1, 1, 1, 0]],
        modality_token_spans=[TokenSpan(sample_index=0, start=3, end=5, modality="audio")],
        modality_metadata={
            "semantic_token_labels_by_sample": [[
                "global_special", "ins_special", "audio_special",
                "audio", "audio", "audio_special", "resp_special", "pad",
            ]],
            "semantic_classification_source": "fixture",
        },
    )

    with collector.request(
        original_state=original_state, active_state=active_state, processed_batch=batch,
    ):
        for layer in modules["audio_layers"]:
            layer.emit((4, 8), (4, 8))
        modules["pooler"].emit((8, 8), (4, 8))
        modules["post_norm"].emit((4, 8), (4, 8))
        modules["projector"].emit((2, 8), (2, 16))
        for layer in modules["llm_layers"]:
            layer.emit((1, 8, 16), (1, 8, 16))
            layer.emit((1, 1, 16), (1, 1, 16))
            layer.emit((1, 1, 16), (1, 1, 16))

    collector.apply_compression_report({
        "plugin": "adjacent_token_merging_audio", "placement": "encoder_level",
        "layers": [{"layer_index": 0, "input_tokens": 4, "output_tokens": 3}],
    })
    report = collector.last_report
    assert report is not None
    assert len(report["layers"]) == 4
    first_audio = report["layers"][0]
    assert first_audio["stage"] == "audio_encoder"
    assert first_audio["encode"]["input"]["token_counts"]["audio"] == 4
    assert first_audio["encode"]["output"]["token_counts"]["audio"] == 3
    first_llm = next(item for item in report["layers"] if item["stage"] == "llm")
    source_counts = {
        "ins_special": 1, "resp_special": 1, "audio": 4,
        "audio_special": 2, "global_special": 1, "pad": 1,
    }
    active_counts = {**source_counts, "audio": 2}
    assert {
        key: count for key, count in first_llm["prefill"]["input"]["token_counts"].items()
        if count
    } == active_counts
    assert first_llm["decode"]["calls"] == 2
    assert first_llm["decode"]["input"]["token_counts"]["response"] == 2
    projector = next(
        item for item in report["modules"] if item["stage"] == "audio_projector"
    )
    assert projector["encode"]["input"]["token_counts"]["audio"] == 2

    report_before_projector_dispatch = json.dumps(report, sort_keys=True)
    collector.apply_compression_report({
        "plugin": "segmentwise_topk_audio", "placement": "projector_level",
    })
    assert json.dumps(collector.last_report, sort_keys=True) == report_before_projector_dispatch
    collector.apply_generated_response(
        [[90, 5]], semantic_metadata={"special_token_ids": [5], "token_ids": {}},
    )
    counts = {
        item["stage"]: {key: count for key, count in item["token_counts"].items() if count}
        for item in collector.last_report["stage_snapshots"]
    }
    assert list(counts) == [
        "processor_output", "audio_feature_input", "post_legalizer_llm_input",
        "generated_response", "complete_llm_sequence",
    ]
    assert counts == {
        "processor_output": source_counts,
        "audio_feature_input": {"audio": 3, "pad": 1},
        "post_legalizer_llm_input": active_counts,
        "generated_response": {"response": 1, "resp_special": 1},
        "complete_llm_sequence": {**active_counts, "response": 1, "resp_special": 2},
    }
    assert report["schema_version"] == 3
    _assert_report_is_conserved(report)
    json.dumps(report)
    collector.uninstall()


@pytest.mark.parametrize("initial", [False, True])
def test_llm_debug_reconciles_audio_pyramid_compaction_per_layer(initial: bool) -> None:
    model, modules = _model()
    collector = Qwen25OmniTokenDebugCollector(model)
    collector.install()
    semantic_labels = [
        "global_special",
        "ins_special",
        "audio_special",
        "audio",
        "audio",
        "audio_special",
        "resp_special",
        "pad",
    ]
    batch = ProcessedBatch(
        model_inputs={
            "input_ids": [[99, 1, 13, 12, 12, 14, 5, 0]],
            "attention_mask": [[1, 1, 1, 1, 1, 1, 1, 0]],
        },
        sample_ids=["sample-0"],
        prompt_lengths=[7],
        metadata={
            "semantic_token_labels_by_sample": [semantic_labels],
            "semantic_classification_source": "fixture",
        },
    )
    state = ForwardState(
        input_ids=batch.model_inputs["input_ids"],
        attention_mask=batch.model_inputs["attention_mask"],
        modality_token_spans=[
            TokenSpan(sample_index=0, start=3, end=5, modality="audio")
        ],
        modality_metadata={
            "semantic_token_labels_by_sample": [semantic_labels],
            "semantic_classification_source": "fixture",
        },
    )

    with collector.request(
        original_state=state,
        active_state=state,
        processed_batch=batch,
    ):
        llm_layers = modules["llm_layers"]
        llm_layers[0].emit((1, 8, 16), (1, 6, 16))
        llm_layers[1].emit((1, 6, 16), (1, 6, 16))

    collector.apply_compression_report(
        {
            "plugin": "audio_pyramid_attention_pruning",
            "placement": "llm_level",
            "stages": [
                {
                    "layer_index_zero_based": None if initial else 0,
                    "application_position": {"boundary": "inter_layer", "index": 0 if initial else 1},
                    "input_sequence_width": 8,
                    "output_sequence_width": 6,
                    "input_audio_tokens": 2,
                    "output_audio_tokens": 1,
                    "position_policy": "preserve_absolute",
                },
                *([{
                    "layer_index_zero_based": 0,
                    "application_position": {"boundary": "inter_layer", "index": 1},
                    "input_audio_tokens": 1,
                    "output_audio_tokens": 1,
                }] if initial else []),
            ],
        }
    )

    report = collector.last_report
    assert report is not None
    llm_layers = [item for item in report["layers"] if item["stage"] == "llm"]
    assert llm_layers[0]["prefill"]["input"]["token_counts"]["audio"] == 2
    assert llm_layers[0]["prefill"]["input"]["token_counts"]["pad"] == 1
    assert llm_layers[0]["prefill"]["output"]["token_counts"]["audio"] == 1
    assert llm_layers[0]["prefill"]["output"]["token_counts"]["pad"] == 0
    assert llm_layers[1]["prefill"]["input"]["token_counts"]["audio"] == 1
    assert llm_layers[1]["prefill"]["input"]["token_counts"]["pad"] == 0
    assert llm_layers[1]["prefill"]["output"]["token_counts"]["audio"] == 1
    assert llm_layers[1]["prefill"]["output"]["token_counts"]["unknown"] == 0
    _assert_report_is_conserved(report)
    collector.uninstall()


_RENDERED_TEMPLATE = "<g><s>SYS</s><u><ab><A><ae>ASK</u><r>"
_SYSTEM_PREFIX = "<g><s>SYS</s>"
_USER_PREFIX = "<g><s>SYS</s><u><ab><A><ae>ASK</u>"
_TEMPLATE_IDS = [99, 1, 30, 2, 3, 13, 12, 14, 31, 4, 5]


class _OffsetTokenizer:
    all_special_ids = [99, 1, 2, 3, 13, 12, 14, 4, 5]
    all_special_tokens = [
        "<g>",
        "<s>",
        "</s>",
        "<u>",
        "<|audio_bos|>",
        "<|AUDIO|>",
        "<|audio_eos|>",
        "</u>",
        "<r>",
    ]
    bos_token_id = 99
    _input_ids_by_text = {
        _SYSTEM_PREFIX: [99, 1, 30, 2],
        _USER_PREFIX: [99, 1, 30, 2, 3, 13, 12, 14, 31, 4],
        _RENDERED_TEMPLATE: _TEMPLATE_IDS,
    }

    def __call__(
        self,
        text: str,
        *,
        add_special_tokens: bool,
        return_offsets_mapping: bool = False,
    ) -> dict[str, Any]:
        assert add_special_tokens is False
        result: dict[str, Any] = {"input_ids": self._input_ids_by_text[text]}
        if return_offsets_mapping:
            assert text == _RENDERED_TEMPLATE
            result["offset_mapping"] = [
                (0, 3),
                (3, 6),
                (6, 9),
                (9, 13),
                (13, 16),
                (16, 20),
                (20, 23),
                (23, 27),
                (27, 30),
                (30, 34),
                (34, 37),
            ]
        return result


def _semantic_fixture() -> tuple[ForwardState, ProcessedBatch]:
    state = ForwardState(
        input_ids=[[99, 1, 30, 2, 3, 13, 12, 12, 12, 12, 14, 31, 4, 5, 0]],
        attention_mask=[[1] * 14 + [0]],
        modality_metadata={},
    )
    batch = ProcessedBatch(
        model_inputs={
            "input_ids": state.input_ids,
            "attention_mask": state.attention_mask,
        },
        sample_ids=["sample-0"],
        prompt_lengths=[14],
        metadata={
            "chat_template_text_by_sample": [_RENDERED_TEMPLATE],
            "semantic_char_spans_by_sample": [
                [
                    {
                        "template_start": 6,
                        "template_end": 9,
                        "label": "system_prompt",
                    },
                    {
                        "template_start": 27,
                        "template_end": 30,
                        "label": "instruction",
                    },
                ]
            ],
            "semantic_message_prefixes_by_sample": [
                [
                    {"label": "system_prompt", "text": _SYSTEM_PREFIX},
                    {"label": "instruction", "text": _USER_PREFIX},
                ]
            ],
            "semantic_classification_warnings_by_sample": [[]],
            "token_ids": {"audio": 12, "image": 10, "video": 11},
            "special_token_ids": _OffsetTokenizer.all_special_ids,
            "special_token_text_by_id": dict(
                zip(
                    _OffsetTokenizer.all_special_ids,
                    _OffsetTokenizer.all_special_tokens,
                )
            ),
            "global_special_token_ids": [99],
        },
    )
    return state, batch


def test_debug_collector_builds_semantic_labels_from_chat_template_offsets() -> None:
    model, _ = _model()
    processor = SimpleNamespace(tokenizer=_OffsetTokenizer())
    collector = Qwen25OmniTokenDebugCollector(model, processor=processor)
    state, batch = _semantic_fixture()

    collector.prepare_sequence_labels(state, batch)

    assert state.modality_metadata["semantic_token_labels_by_sample"] == [
        [
            "global_special",
            "sys_special",
            "system_prompt",
            "sys_special",
            "ins_special",
            "audio_special",
            "audio",
            "audio",
            "audio",
            "audio",
            "audio_special",
            "instruction",
            "ins_special",
            "resp_special",
            "pad",
        ]
    ]
    assert state.modality_metadata["semantic_classification_source"] == (
        "chat_template_offsets"
    )
    assert batch.metadata["semantic_token_labels_by_sample"] == (
        state.modality_metadata["semantic_token_labels_by_sample"]
    )


class _SlowTokenizer:
    _input_ids_by_text = {
        **_OffsetTokenizer._input_ids_by_text,
        "<g><s>": [99, 1],
        "<g><s>SYS": [99, 1, 30],
        "<g><s>SYS</s><u><ab><A><ae>": [99, 1, 30, 2, 3, 13, 12, 14],
        "<g><s>SYS</s><u><ab><A><ae>ASK": [
            99,
            1,
            30,
            2,
            3,
            13,
            12,
            14,
            31,
        ],
    }

    def __call__(
        self,
        text: str,
        *,
        add_special_tokens: bool,
        return_offsets_mapping: bool = False,
    ) -> dict[str, Any]:
        assert add_special_tokens is False
        if return_offsets_mapping:
            raise NotImplementedError
        return {"input_ids": self._input_ids_by_text[text]}


def test_debug_collector_supports_tokenizer_without_offset_mapping() -> None:
    model, _ = _model()
    processor = SimpleNamespace(tokenizer=_SlowTokenizer())
    collector = Qwen25OmniTokenDebugCollector(model, processor=processor)
    state, batch = _semantic_fixture()

    collector.prepare_sequence_labels(state, batch)

    assert state.modality_metadata["semantic_token_labels_by_sample"] == [
        [
            "global_special",
            "sys_special",
            "system_prompt",
            "sys_special",
            "ins_special",
            "audio_special",
            "audio",
            "audio",
            "audio",
            "audio",
            "audio_special",
            "instruction",
            "ins_special",
            "resp_special",
            "pad",
        ]
    ]
    assert state.modality_metadata["semantic_classification_source"] == (
        "chat_template_prefix_alignment"
    )


def test_training_semantic_fallback_keeps_unresolved_prompt_tokens_unknown() -> None:
    model, _ = _model()
    collector = Qwen25OmniTokenDebugCollector(model)
    state = ForwardState(
        input_ids=[[1, 13, 12, 14, 30, 31, 2, 0]],
        attention_mask=[[1, 1, 1, 1, 1, 1, 1, 0]],
        labels=[[-100, -100, -100, -100, -100, 31, 2, -100]],
        modality_metadata={
            "token_ids": {"audio": 12, "image": 10, "video": 11},
            "special_token_ids": [0, 1, 2, 12, 13, 14],
            "global_special_token_ids": [1],
            "modality_special_token_ids": {
                "audio": [13, 14],
                "image": [],
                "video": [],
            },
        },
    )
    batch = ProcessedBatch(
        model_inputs={"input_ids": state.input_ids},
        sample_ids=[0],
        prompt_lengths=[],
        metadata=dict(state.modality_metadata),
    )

    collector.prepare_sequence_labels(state, batch)

    assert state.modality_metadata["semantic_token_labels_by_sample"] == [
        [
            "global_special",
            "audio_special",
            "audio",
            "audio_special",
            "unknown",
            "response",
            "resp_special",
            "pad",
        ]
    ]
    assert state.modality_metadata["semantic_classification_source"] == (
        "training_labels_fallback"
    )


def test_training_semantic_fallback_recovers_only_qwen_user_region() -> None:
    model, _ = _model()
    collector = Qwen25OmniTokenDebugCollector(model)
    state = ForwardState(
        input_ids=[
            [
                151644,
                8948,
                198,
                50,
                151645,
                198,
                151644,
                872,
                198,
                13,
                12,
                14,
                60,
                151645,
                198,
            ]
        ],
        attention_mask=[[1] * 15],
        labels=[[-100] * 15],
        modality_metadata={
            "token_ids": {"audio": 12, "image": 10, "video": 11},
            "special_token_ids": [13, 14, 151644, 151645],
            "global_special_token_ids": [151644],
            "modality_special_token_ids": {
                "audio": [13, 14],
                "image": [],
                "video": [],
            },
            "bos_token_id": 151644,
            "eos_token_id": [151645, 151646],
            "user_token_id": 872,
        },
    )
    batch = ProcessedBatch(
        model_inputs={"input_ids": state.input_ids},
        sample_ids=[0],
        prompt_lengths=[],
        metadata=dict(state.modality_metadata),
    )

    collector.prepare_sequence_labels(state, batch)

    semantic = state.modality_metadata["semantic_token_labels_by_sample"][0]
    assert semantic[3] == "unknown"
    assert semantic[7] == "ins_special"
    assert semantic[8] == "instruction"
    assert semantic[9:13] == [
        "audio_special",
        "audio",
        "audio_special",
        "instruction",
    ]
    assert semantic[13] != "instruction"


def test_modality_content_wins_over_special_and_vision_boundaries_are_typed() -> None:
    metadata = {
        "modality_special_token_ids": {"audio": [13, 14]},
        "vision_special_token_ids": [15, 16],
        "special_token_text_by_id": {
            10: "<|IMAGE|>",
            11: "<|VIDEO|>",
            12: "<|AUDIO|>",
            13: "<|audio_bos|>",
            14: "<|audio_eos|>",
            15: "<|vision_start|>",
            16: "<|vision_end|>",
        },
    }
    explicit, vision = _modality_special_sets(metadata)
    input_ids = [13, 12, 14, 15, 10, 16, 15, 11, 16]
    labels = ["ins_special"] * len(input_ids)

    _apply_modality_special_labels(
        input_ids=input_ids,
        labels=labels,
        modality_by_id={10: "image", 11: "video", 12: "audio"},
        explicit_special_by_id=explicit,
        vision_special_ids=vision,
    )

    assert labels == [
        "audio_special",
        "audio",
        "audio_special",
        "image_special",
        "image",
        "image_special",
        "video_special",
        "video",
        "video_special",
    ]
