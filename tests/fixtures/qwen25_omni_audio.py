"""Shared native Omni models, audio inputs and compression recipes for tests."""
from __future__ import annotations
from types import SimpleNamespace
import pytest
from mllm_comp.backbones.qwen25_omni.v2_compression import (
    create_qwen25_omni_dynamic_controller, resolve_qwen25_omni_stage_execution,
)
from mllm_comp.core.vocabulary import RuntimeMode
from mllm_comp.core.state import ForwardState, TokenSpan
from mllm_comp.pipeline.stage import CompressionStageConfig


def compression_config(stage):
    """Use the public configuration defaults when activating a smoke recipe."""
    from mllm_comp.config.schema import experiment_config_from_dict

    config = experiment_config_from_dict({"backbone": {"name": "qwen25_omni"}})
    config.pipeline.stages = [stage]
    return config


def create_audio(path, *, sample_rate=16000):
    """Write the one-second, mono 440 Hz input used by checkpoint smoke tests."""
    import math
    import struct
    import wave

    with wave.open(str(path), "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(sample_rate)
        output.writeframes(b"".join(
            struct.pack("<h", int(0.15 * 32767 * math.sin(2 * math.pi * 440 * index / sample_rate)))
            for index in range(sample_rate)
        ))


class _MiniThinker:
    def __init__(self, audio_tower: object) -> None:
        self.audio_tower = audio_tower

    def get_rope_index(self, *args: object, **kwargs: object) -> object:
        return args, kwargs


def projector_stage(
    plugin: str = "segmentwise_topk_audio",
) -> CompressionStageConfig:
    methods = {
        "segmentwise_topk_audio": {
            "name": "audio_segmentwise_topk",
            "object_subtype": "audio_encoder_output_pre_projector",
            "operation": {
                "type": "keep",
                "target_site": "audio_encoder_output_pre_projector",
            },
            "criterion": {
                "score_source": "encoder_last_layer_attention",
                "num_segments": 2,
            },
            "schedule": {
                "keep_ratio": 0.5,
                "allocation": "balanced_exact",
                "minimum_keep_tokens": 1,
            },
        },
        "local_temporal_bipartite_merging_audio": {
            "name": "audio_ltbm",
            "object_subtype": "audio_encoder_output_pre_projector",
            "operation": {
                "type": "merge",
                "target_site": "audio_encoder_output_pre_projector",
                "merge_value": "mean",
            },
            "criterion": {
                "score_source": "projector_token_cosine",
                "pairing": "alternating_bipartite_local",
                "window_size": 4,
            },
            "schedule": {"keep_ratio": 0.5},
        },
        "similarity_attention_merge_prune_audio": {
            "name": "audio_sa_map",
            "object_subtype": "audio_projector_output_post_projector",
            "operation": {
                "type": "merge_prune",
                "target_site": "audio_projector_output_post_projector",
                "merge_value": "attention_weighted_mean",
            },
            "criterion": {
                "score_source": "encoder_penultimate_layer_attention",
                "similarity_threshold": 0.8,
                "pruning": "greedy_attention_dpp",
            },
            "schedule": {"keep_ratio": 0.5},
        },
        "uniform_sampling_audio": {
            "name": "uniform-sampling",
            "operation": {"type": "keep"},
            "schedule": {"compression_factor": 2},
        },
        "uniform_average_pooling_audio": {
            "name": "uniform-average",
            "operation": {"type": "merge"},
            "schedule": {"compression_factor": 2},
        },
        "unsupervised_segmentation_audio": {
            "name": "unseg",
            "operation": {"type": "merge"},
        },
        "fastslm_hta_audio": {
            "name": "fastslm-hta",
            "operation": {"type": "merge"},
            "schedule": {"keep_ratio": 0.5},
            "config": {
                "window_tokens": 16,
                "stage_query_tokens": 4,
                "output_query_tokens": 8,
                "detail_recovery_depth": 1,
                "num_attention_heads": 0,
            },
        },
    }
    method = methods[plugin]
    return CompressionStageConfig.from_dict(
        {
            **method,
            "placement": "projector_level",
            "execution_phase": "generate",
            "plugin": plugin,
            "algorithm": plugin,
            "variant": f"{plugin}.audio",
            "binding": f"{plugin}.audio.qwen25_omni",
            "method": "token_compression",
            "object_family": "tensor_slot",
            "modalities": ["audio"],
            "target_modalities": ["audio"],
            "conditioning_modalities": [],
            "preserved_modalities": ["text", "image", "video"],
            "coupling_policy": "independent",
            "media_alignment": "temporal",
            "token_roles": ["observation"],
        }
    )


def encoder_stage() -> CompressionStageConfig:
    return CompressionStageConfig.from_dict(
        {
            "name": "audio_adjacent_token_merging",
            "placement": "encoder_level",
            "execution_phase": "generate",
            "plugin": "adjacent_token_merging_audio",
            "algorithm": "adjacent_token_merging_audio",
            "variant": "adjacent_token_merging_audio.audio",
            "binding": "adjacent_token_merging_audio.audio.qwen25_omni",
            "method": "token_compression",
            "object_family": "tensor_slot",
            "object_subtype": "audio_encoder_layer_post_mhsa_pre_ffn",
            "modalities": ["audio"],
            "target_modalities": ["audio"],
            "conditioning_modalities": [],
            "preserved_modalities": ["text", "image", "video"],
            "coupling_policy": "independent",
            "media_alignment": "temporal",
            "token_roles": ["observation"],
            "operation": {
                "type": "merge",
                "target_site": "audio_encoder_layer_post_mhsa_pre_ffn",
                "merge_value": "mean",
            },
            "criterion": {
                "policy": "fixed_ratio",
                "score_source": "encoder_key_cosine",
                "pairing": "disjoint_adjacent",
            },
            "schedule": {"merge_ratio": 0.10},
            "activation_scope": {"first_layer": 2, "layer_interval": 3},
        }
    )


def encoder_extension_stage(plugin: str) -> CompressionStageConfig:
    methods = {
        "encoder_attention_topk_audio": {
            "name": "audio_encoder_topk",
            "object_subtype": "audio_encoder_layer_post_mhsa_pre_ffn",
            "operation": {
                "type": "keep",
                "target_site": "audio_encoder_layer_post_mhsa_pre_ffn",
            },
            "schedule": {"keep_ratio": 0.5},
        },
        "progressive_attention_pruning_audio": {
            "name": "audio_progressive_pruning",
            "object_subtype": "audio_encoder_layer_output_post_block",
            "operation": {
                "type": "keep",
                "target_site": "audio_encoder_layer_output_post_block",
            },
            "schedule": {"policy": "fixed", "pruning_rate": 0.5},
        },
        **{
            algorithm: {
                "name": algorithm,
                "object_subtype": "audio_encoder_layer_output_post_block",
                "operation": {"type": kind, "target_site": "audio_encoder_layer_output_post_block", "layers": [0, 2]},
                "schedule": {"keep_ratio": 0.5} if algorithm == "local_temporal_bipartite_merging_audio" else {"compression_factor": 2},
            }
            for algorithm, kind in [("uniform_sampling_audio", "keep"), ("uniform_average_pooling_audio", "merge"),
                                    ("local_temporal_bipartite_merging_audio", "merge")]
        },
    }
    methods["segmentwise_topk_audio"] = {
        "name": plugin, "object_subtype": "audio_encoder_layer_output_post_block",
        "operation": {"type": "keep", "target_site": "audio_encoder_layer_output_post_block", "layers": [0, 2]},
        "schedule": {"keep_ratio": 0.5},
    }
    methods["fastslm_hta_audio"] = {
        "name": plugin, "object_subtype": "audio_encoder_layer_output_post_block",
        "operation": {"type": "merge", "target_site": "audio_encoder_layer_output_post_block", "layers": [0, 2]},
        "schedule": {"keep_ratio": 0.5},
        "config": {"window_tokens": 8, "output_query_tokens": 4, "stage_query_tokens": 4, "detail_recovery_depth": 1},
    }
    method = methods[plugin]
    return CompressionStageConfig.from_dict(
        {
            **method,
            "placement": "encoder_level",
            "execution_phase": "forward",
            "plugin": plugin,
            "algorithm": plugin,
            "variant": f"{plugin}.audio",
            "binding": f"{plugin}.audio.qwen25_omni",
            "method": "token_compression",
            "object_family": "tensor_slot",
            "modalities": ["audio"],
            "target_modalities": ["audio"],
            "conditioning_modalities": [],
            "preserved_modalities": ["text", "image", "video"],
            "coupling_policy": "independent",
            "media_alignment": "temporal",
            "token_roles": ["observation"],
            "criterion": {} if plugin.startswith("uniform_") or plugin in {"local_temporal_bipartite_merging_audio", "fastslm_hta_audio"} else {"score_source": "mean_received_self_attention"},
            "activation_scope": {"layers": [0, 2]},
        }
    )


def llm_stage(*, phase: str) -> CompressionStageConfig:
    return CompressionStageConfig.from_dict(
        {
            "name": "audio_pyramid_tiny",
            "placement": "llm_level",
            "execution_phase": phase,
            "plugin": "audio_pyramid_attention_pruning",
            "algorithm": "audio_pyramid_attention_pruning",
            "variant": "audio_pyramid_attention_pruning.audio",
            "binding": "audio_pyramid_attention_pruning.audio.qwen25_omni",
            "method": "token_compression",
            "object_family": "tensor_slot",
            "object_subtype": "llm_decoder_layer_output_audio_tokens",
            "modalities": ["audio"],
            "target_modalities": ["audio"],
            "conditioning_modalities": ["text"],
            "preserved_modalities": ["text", "image", "video"],
            "coupling_policy": "independent",
            "media_alignment": "token",
            "token_roles": ["observation"],
            "operation": {
                "type": "keep",
                "target_site": "llm_decoder_layer_output_audio_tokens",
            },
            "criterion": {
                "score_source": "instruction_to_audio_received_attention"
            },
            "schedule": {"keep_ratio": 0.5, "minimum_keep_tokens": 1},
            "activation_scope": {
                "layers": [0, 1, 2],
                "lifecycle": "prefill_only",
            },
        }
    )


def _operation_stage() -> CompressionStageConfig:
    return CompressionStageConfig.from_dict(
        {
            "name": "spade_operation_only_tiny",
            "placement": "llm_level",
            "execution_phase": "prepare",
            "plugin": "structured_llm_layer_pruning",
            "algorithm": "structured_llm_layer_pruning",
            "variant": "structured_llm_layer_pruning.audio",
            "binding": "structured_llm_layer_pruning.audio.qwen25_omni",
            "method": "operation_compression",
            "object_family": "module_unit",
            "object_subtype": "transformer_layer",
            "modalities": ["audio"],
            "target_modalities": ["audio"],
            "conditioning_modalities": [],
            "preserved_modalities": ["text", "image", "video"],
            "coupling_policy": "independent",
            "media_alignment": "none",
            "operation": {
                "type": "prune",
                "target_site": "llm.layers",
                "target_scope": "llm",
                "target_kind": "layer",
                "apply_mode": "structural_rewrite",
                "shape_effect": "remove_module",
            },
            "criterion": {"selection_source": "explicit"},
            "activation_scope": {"layers": [1, 3]},
        }
    )


def _create_controller(
    model: object,
    stage: CompressionStageConfig,
    *,
    profile_enabled: bool = False,
) -> object:
    execution = resolve_qwen25_omni_stage_execution(stage)
    return create_qwen25_omni_dynamic_controller(
        model=model,
        stage=stage,
        binding=execution.binding,
        runtime_mode=(RuntimeMode.PROFILE if profile_enabled else RuntimeMode.DEFAULT),
    )


def _state(torch, *, training: bool) -> ForwardState:
    labels = None
    if training:
        labels = torch.tensor([[-100, -100, -100, -100, -100, -100, 3]])
    return ForwardState(
        input_ids=torch.tensor([[1, 99, 99, 99, 99, 2, 3]]),
        attention_mask=torch.ones((1, 7), dtype=torch.long),
        labels=labels,
        modality_token_spans=[
            TokenSpan(0, 1, 5, modality="audio", role="observation")
        ],
        modality_metadata={
            "semantic_token_labels_by_sample": [
                [
                    "ins_special",
                    "audio",
                    "audio",
                    "audio",
                    "audio",
                    "instruction",
                    "response",
                ]
            ]
        },
        native_model_inputs={},
    )


def _tiny_model(torch, *, audio_encoder_layers=1):
    transformers = pytest.importorskip("transformers")
    assert transformers.__version__ == "4.57.3"
    from transformers.models.qwen2_5_omni.configuration_qwen2_5_omni import (
        Qwen2_5OmniAudioEncoderConfig,
        Qwen2_5OmniTextConfig,
        Qwen2_5OmniThinkerConfig,
        Qwen2_5OmniVisionEncoderConfig,
    )
    from transformers.models.qwen2_5_omni.modeling_qwen2_5_omni import (
        Qwen2_5OmniThinkerForConditionalGeneration,
    )

    text = Qwen2_5OmniTextConfig(
        vocab_size=128,
        hidden_size=12,
        intermediate_size=24,
        num_hidden_layers=4,
        num_attention_heads=2,
        num_key_value_heads=2,
        max_position_embeddings=64,
        layer_types=["full_attention"] * 4,
    )
    text.rope_scaling = {
        "mrope_section": [1, 1, 1],
        "rope_type": "default",
        "type": "default",
    }
    audio = Qwen2_5OmniAudioEncoderConfig(
        num_mel_bins=4,
        encoder_layers=audio_encoder_layers,
        encoder_attention_heads=2,
        encoder_ffn_dim=24,
        d_model=12,
        max_source_positions=8,
        n_window=4,
        output_dim=12,
    )
    vision = Qwen2_5OmniVisionEncoderConfig(
        depth=1,
        # Native 2-D vision RoPE needs a head width divisible by four. Swift's
        # audio-only template also executes this tower for its zero-loss graph.
        hidden_size=16,
        intermediate_size=24,
        num_heads=2,
        patch_size=2,
        spatial_merge_size=1,
        temporal_patch_size=1,
        window_size=4,
        out_hidden_size=12,
        fullatt_block_indexes=[0],
    )
    config = Qwen2_5OmniThinkerConfig(
        audio_config=audio,
        vision_config=vision,
        text_config=text,
        audio_token_index=99,
        image_token_index=100,
        video_token_index=101,
        audio_start_token_id=102,
        audio_end_token_id=103,
        user_token_id=104,
        vision_start_token_id=105,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=127,
    )
    thinker = Qwen2_5OmniThinkerForConditionalGeneration(config)

    class _Composite(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.thinker = thinker
            self.config = SimpleNamespace(thinker_config=config)

    return _Composite()


def _hta_layer_stage(family, *, initial=False, identity=False):
    site = {"encoder": "audio_encoder_layer_output_post_block", "projector": "audio_encoder_output_pre_projector",
            "llm": "llm_decoder_layer_post_attention_audio_tokens"}[family]
    return CompressionStageConfig.from_dict({
        "name": family, "placement": family + "_level", "execution_phase": "forward",
        "plugin": "fastslm_hta_audio", "algorithm": "fastslm_hta_audio", "variant": "fastslm_hta_audio.audio",
        "binding": "fastslm_hta_audio.audio.qwen25_omni", "target_modalities": ["audio"],
        "operation": {"type": "merge", "target_site": site,
                      **({"layers": [0 if initial else 1]} if family != "projector" else {}),
                      **({"layer_edge": "before"} if initial else {})},
        "schedule": {"keep_ratio": 1.0 if identity else 0.5},
        "config": {"window_tokens": 8, "output_query_tokens": 8 if identity else 4,
                   "stage_query_tokens": 4, "detail_recovery_depth": 1},
    })


def _hta_audio_state(torch):
    features = torch.randn(1, 4, 32, requires_grad=True)
    ids = torch.tensor([[1, 102, *([99] * 8), 103, 2, 3]])
    mask = torch.ones_like(ids)
    labels = torch.tensor([[-100] * 12 + [3]])
    return ForwardState(input_ids=ids, attention_mask=mask, labels=labels, audio_values=features,
        audio_feature_lengths=torch.tensor([32]), modality_token_spans=[TokenSpan(0, 2, 10, modality="audio")],
        modality_metadata={"token_ids": {"audio": 99}, "pad_token_id": 0, "expected_token_counts_by_sample": [{"audio": 8}]},
        native_model_inputs={"input_ids": ids, "attention_mask": mask, "labels": labels,
                             "input_features": features, "feature_attention_mask": torch.ones((1, 32), dtype=torch.long)})
