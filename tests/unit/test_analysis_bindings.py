from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from mllm_comp.analysis.bindings import resolve_analysis_binding
from mllm_comp.analysis.common.token_mask import build_token_group_layout
from mllm_comp.backbones.llava_onevision import LlavaOneVisionModelAdapter
from mllm_comp.backbones.qwen25_vl import Qwen25VLModelAdapter
from mllm_comp.backbones.qwen25_omni import Qwen25OmniModelAdapter
from mllm_comp.backbones.qwen3_vl import Qwen3VLModelAdapter
from mllm_comp.core.errors import BackendIntegrationError
from mllm_comp.core.state import ForwardState, TokenSpan


def _attention() -> Any:
    return SimpleNamespace(
        num_heads=4,
        num_key_value_heads=2,
        head_dim=4,
        config=SimpleNamespace(
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=4,
            hidden_size=16,
        ),
    )


def _facade(name: str, adapter: Any) -> Any:
    return SimpleNamespace(
        capabilities=lambda: SimpleNamespace(name=name),
        model_adapter=adapter,
    )


def test_llava_analysis_binding_resolves_decoder_semantics() -> None:
    attention = _attention()
    layer = SimpleNamespace(self_attn=attention, mlp=object())
    norm = object()
    core = SimpleNamespace(layers=[layer], norm=norm)
    lm_head = object()
    model = SimpleNamespace(
        get_model=lambda: core,
        get_vision_tower=lambda: None,
        lm_head=lm_head,
        _mllm_comp_source_evidence={
            "checkpoint": {"model_id": "org/model", "revision": "model-sha"},
            "dependencies": {
                "vision_tower": {
                    "model_id": "org/vision",
                    "revision": "vision-sha",
                }
            },
        },
    )

    binding = resolve_analysis_binding(
        _facade("llava_onevision", LlavaOneVisionModelAdapter()), model
    )

    assert binding.layers[0].layer is layer
    assert binding.final_norm is norm
    assert binding.lm_head is lm_head
    assert binding.attention_geometry.query_heads_per_key_value_head == 2
    assert binding.evidence["model_source"]["dependencies"]["vision_tower"] == {
        "model_id": "org/vision",
        "revision": "vision-sha",
    }


def test_qwen25_analysis_binding_resolves_decoder_semantics() -> None:
    attention = _attention()
    layer = SimpleNamespace(self_attn=attention, mlp=object())
    norm = object()
    language_model = SimpleNamespace(layers=[layer], norm=norm)
    model = SimpleNamespace(
        model=SimpleNamespace(language_model=language_model, visual=None),
        lm_head=object(),
    )

    binding = resolve_analysis_binding(
        _facade("qwen25_vl", Qwen25VLModelAdapter()), model
    )

    assert binding.layers[0].attention is attention
    assert binding.final_norm is norm
    assert binding.attention_geometry.hidden_size == 16


def test_qwen3_analysis_binding_records_deepstack_mapping() -> None:
    attention = _attention()
    layer = SimpleNamespace(self_attn=attention, mlp=SimpleNamespace())
    norm = object()
    deepstack = [object(), object()]
    language_model = SimpleNamespace(layers=[layer], norm=norm)
    visual = SimpleNamespace(merger=object(), deepstack_merger_list=deepstack)
    model = SimpleNamespace(
        model=SimpleNamespace(language_model=language_model, visual=visual),
        lm_head=object(),
    )

    binding = resolve_analysis_binding(
        _facade("qwen3_vl", Qwen3VLModelAdapter()), model
    )

    assert binding.deepstack_projectors == tuple(deepstack)
    assert binding.evidence["deepstack_projector_count"] == 2


def test_qwen25_omni_analysis_binding_is_thinker_only() -> None:
    attention = _attention()
    attention.v_proj = object()
    attention.o_proj = object()
    ffn = SimpleNamespace(down_proj=object())
    layer = SimpleNamespace(self_attn=attention, mlp=ffn)
    thinker_llm = SimpleNamespace(
        layers=[layer],
        norm=object(),
        embed_tokens=object(),
    )
    thinker = SimpleNamespace(model=thinker_llm, lm_head=object())
    model = SimpleNamespace(
        thinker=thinker,
        talker=SimpleNamespace(model=SimpleNamespace(layers=[object()])),
        token2wav=object(),
    )

    binding = resolve_analysis_binding(
        _facade("qwen25_omni", Qwen25OmniModelAdapter()), model
    )

    assert binding.layers[0].layer is layer
    assert binding.input_embeddings is thinker_llm.embed_tokens
    assert binding.final_norm is thinker_llm.norm
    assert binding.lm_head is thinker.lm_head
    assert binding.evidence["semantic_scope"] == "thinker.llm"
    assert binding.evidence["excluded_branches"] == [
        "audio_encoder",
        "talker",
        "token2wav",
    ]


def test_analysis_binding_fails_before_collection_on_missing_semantics() -> None:
    facade = _facade(
        "qwen25_vl",
        SimpleNamespace(
            structural_handles=lambda _model: {
                "llm.layer[0]": object(),
                "llm.layer[0].attn": _attention(),
            }
        ),
    )
    with pytest.raises(BackendIntegrationError, match="lacks Attention or FFN"):
        resolve_analysis_binding(facade, object())


def test_qwen_token_layout_uses_exact_spans_system_and_padding() -> None:
    state = ForwardState(
        input_ids=[[1, 10, 10, 2, 0]],
        attention_mask=[[1, 1, 1, 1, 0]],
        sample_ids=["sample"],
        modality_token_spans=[
            TokenSpan(sample_index=0, start=1, end=3, modality="image")
        ],
        modality_metadata={"system_token_spans": [(0, 1)]},
    )

    layout = build_token_group_layout(
        state,
        sequence_length=5,
        backbone_name="qwen25_vl",
    )

    assert layout.labels_by_sample == (
        ("system", "image", "image", "text", "padding"),
    )
    assert layout.positions(0, "vision") == (1, 2)
    assert layout.evidence["route"] == "forward_state_exact_spans"


def test_qwen_omni_token_layout_keeps_exact_audio_group() -> None:
    state = ForwardState(
        input_ids=[[1, 15, 15, 15, 2]],
        attention_mask=[[1, 1, 1, 1, 1]],
        sample_ids=["sample"],
        modality_token_spans=[
            TokenSpan(sample_index=0, start=1, end=4, modality="audio")
        ],
    )

    layout = build_token_group_layout(
        state,
        sequence_length=5,
        backbone_name="qwen25_omni",
    )

    assert layout.labels_by_sample == (
        ("text", "audio", "audio", "audio", "text"),
    )
    assert layout.positions(0, "audio") == (1, 2, 3)
    assert layout.positions(0, "vision") == ()
    assert layout.counts()["audio"] == 3


def test_qwen_omni_token_layout_uses_effective_runtime_semantics() -> None:
    state = ForwardState(
        input_ids=[[1, 15, 15, 15, 2]],
        attention_mask=[[1, 1, 1, 1, 1]],
        sample_ids=["sample"],
        modality_metadata={
            "semantic_token_labels_by_sample": [
                ["system_prompt", "audio", "audio", "audio", "instruction"]
            ]
        },
    )
    runtime = SimpleNamespace(
        metadata={
            "last_training_forward": {
                "effective_token_layout": {
                    "schema_version": 1,
                    "sample_ids": ["sample"],
                    "sequence_length": 4,
                    "semantic_labels_by_sample": [
                        ["system_prompt", "audio", "audio", "instruction"]
                    ],
                    "semantic_labels_by_layer_entry": {
                        "9": [["system_prompt", "audio", "instruction"]]
                    },
                }
            }
        }
    )

    layout = build_token_group_layout(
        state,
        sequence_length=4,
        backbone_name="qwen25_omni",
        runtime=runtime,
    )

    assert layout.labels_by_sample == (("system", "audio", "audio", "text"),)
    assert layout.evidence["route"] == "runtime_effective_semantic_labels"

    layer_layout = build_token_group_layout(
        state,
        sequence_length=3,
        backbone_name="qwen25_omni",
        runtime=runtime,
        layer_index=9,
    )
    assert layer_layout.labels_by_sample == (("system", "audio", "text"),)
    assert layer_layout.evidence["route"] == (
        "runtime_layer_entry_semantic_labels"
    )
    assert layer_layout.evidence["layer_index_zero_based"] == 9


def test_llava_single_placeholder_expansion_is_explicit_and_exact() -> None:
    state = ForwardState(
        input_ids=[[1, -200, 2]],
        attention_mask=[[1, 1, 1]],
        sample_ids=["sample"],
        modality_token_spans=[
            TokenSpan(sample_index=0, start=1, end=2, modality="image")
        ],
    )

    layout = build_token_group_layout(
        state,
        sequence_length=6,
        backbone_name="llava_onevision",
    )

    assert layout.labels_by_sample == (
        ("text", "image", "image", "image", "image", "text"),
    )
    assert layout.evidence["route"] == "llava_single_image_placeholder_expansion"
    assert layout.evidence["fallback_limit"] == "single_image_only"


def test_llava_multi_image_expansion_refuses_an_unevidenced_split() -> None:
    state = ForwardState(
        input_ids=[[1, -200, 2, -200, 3]],
        attention_mask=[[1, 1, 1, 1, 1]],
        sample_ids=["sample"],
        modality_token_spans=[
            TokenSpan(sample_index=0, start=1, end=2, modality="image"),
            TokenSpan(sample_index=0, start=3, end=4, modality="image"),
        ],
    )

    with pytest.raises(BackendIntegrationError, match="exact only for one image"):
        build_token_group_layout(
            state,
            sequence_length=9,
            backbone_name="llava_onevision",
        )
