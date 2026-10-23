"""Opt-in real-checkpoint forward/backward smoke for both audio compressors."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
from tests.fixtures.qwen25_omni_audio import (
    compression_config, create_audio as _create_audio,
)

from mllm_comp.backbones.base import BackboneRuntime
from mllm_comp.backbones.qwen25_omni import (
    DEFAULT_AUDIO_SYSTEM_PROMPT,
    Qwen25OmniSample,
    create_qwen25_omni_backbone,
)
from mllm_comp.core.vocabulary import CompressionMode
from mllm_comp.pipeline.runtime_compression import activate_runtime_compression
from mllm_comp.pipeline.stage import CompressionStageConfig


def _stage(plugin: str) -> CompressionStageConfig:
    methods = {
        "segmentwise_topk_audio": {
            "name": "audio_segmentwise_topk_train",
            "placement": "projector_level",
            "object_subtype": "audio_encoder_output_pre_projector",
            "operation": {
                "type": "keep",
                "budget_type": "keep_ratio",
                "target_site": "audio_encoder_output_pre_projector",
            },
            "criterion": {
                "ranking": "descending",
                "score_source": "encoder_last_layer_attention",
                "num_segments": 10,
            },
            "schedule": {
                "granularity": "per_audio",
                "keep_ratio": 0.25,
                "allocation": "balanced_exact",
                "minimum_keep_tokens": 1,
                "selector": {"modality": "audio"},
            },
        },
        "local_temporal_bipartite_merging_audio": {
            "name": "audio_ltbm_train",
            "placement": "projector_level",
            "object_subtype": "audio_encoder_output_pre_projector",
            "operation": {
                "type": "merge",
                "target_site": "audio_encoder_output_pre_projector",
                "merge_value": "mean",
            },
            "criterion": {
                "score_source": "projector_token_cosine",
                "pairing": "alternating_bipartite_local",
                "window_size": 8,
            },
            "schedule": {"keep_ratio": 0.5, "minimum_keep_tokens": 1},
        },
        "similarity_attention_merge_prune_audio": {
            "name": "audio_sa_map_train",
            "placement": "projector_level",
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
            "schedule": {"keep_ratio": 0.5, "minimum_keep_tokens": 1},
        },
        "adjacent_token_merging_audio": {
            "name": "audio_atome_train",
            "placement": "encoder_level",
            "object_subtype": "audio_encoder_layer_post_mhsa_pre_ffn",
            "operation": {
                "type": "merge",
                "target_site": "audio_encoder_layer_post_mhsa_pre_ffn",
                "merge_value": "mean",
            },
            "criterion": {
                "ranking": "descending",
                "policy": "fixed_ratio",
                "score_source": "encoder_key_cosine",
                "pairing": "disjoint_adjacent",
            },
            "schedule": {
                "granularity": "per_layer",
                "merge_ratio": 0.10,
                "selector": {"modality": "audio"},
            },
            "activation_scope": {"first_layer": 2, "layer_interval": 3},
        },
        "encoder_attention_topk_audio": {
            "name": "audio_encoder_topk_train",
            "placement": "encoder_level",
            "object_subtype": "audio_encoder_layer_post_mhsa_pre_ffn",
            "operation": {
                "type": "keep",
                "target_site": "audio_encoder_layer_post_mhsa_pre_ffn",
            },
            "criterion": {"score_source": "mean_received_self_attention"},
            "schedule": {"keep_ratio": 0.9, "minimum_keep_tokens": 2},
            "activation_scope": {"layers": [3, 6, 9]},
        },
        "progressive_attention_pruning_audio": {
            "name": "audio_progressive_pruning_train",
            "placement": "encoder_level",
            "object_subtype": "audio_encoder_layer_output_post_block",
            "operation": {
                "type": "keep",
                "target_site": "audio_encoder_layer_output_post_block",
            },
            "criterion": {"score_source": "mean_received_self_attention"},
            "schedule": {
                "policy": "linear",
                "start_pruning_rate": 0.05,
                "end_pruning_rate": 0.20,
                "minimum_keep_tokens": 2,
            },
            "activation_scope": {"layers": [6, 12, 18]},
        },
    }
    payload = {
        **methods[plugin],
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
    }
    return CompressionStageConfig.from_dict(payload)


@pytest.mark.gpu
@pytest.mark.parametrize(
    "plugin,gradient_checkpointing",
    [
        ("segmentwise_topk_audio", False),
        ("adjacent_token_merging_audio", True),
        ("local_temporal_bipartite_merging_audio", False),
        ("similarity_attention_merge_prune_audio", False),
        ("encoder_attention_topk_audio", True),
        ("progressive_attention_pruning_audio", True),
    ],
)
@pytest.mark.skipif(
    os.environ.get("MLLM_COMP_RUN_QWEN25_OMNI_COMPRESSED_TRAIN_GPU_SMOKE") != "1",
    reason="Set MLLM_COMP_RUN_QWEN25_OMNI_COMPRESSED_TRAIN_GPU_SMOKE=1 to run.",
)
def test_compressed_training_forward_reaches_audio_input_gradient(
    tmp_path: Path,
    plugin: str,
    gradient_checkpointing: bool,
) -> None:
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    assert transformers.__version__ == "4.57.3"
    assert torch.cuda.is_available() and torch.cuda.device_count() == 1

    model_path_value = os.environ.get("QWEN25_OMNI_MODEL_PATH")
    if not model_path_value:
        pytest.fail("QWEN25_OMNI_MODEL_PATH must point to a local checkpoint.")
    model_path = Path(model_path_value)
    assert model_path.is_dir()
    torch.cuda.reset_peak_memory_stats()

    audio_path = tmp_path / f"{plugin}.wav"
    _create_audio(audio_path)
    facade = create_qwen25_omni_backbone()
    processor = facade.processor_adapter.load_processor(
        str(model_path),
        local_files_only=True,
    )
    model = facade.model_adapter.load_model(
        str(model_path),
        train=True,
        local_files_only=True,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map="cuda:0",
        enable_talker=False,
    )
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    if gradient_checkpointing:
        model.gradient_checkpointing_enable()
    else:
        model.gradient_checkpointing_disable()
    runtime = BackboneRuntime(
        model=model,
        processor=processor,
        model_path_or_id=str(model_path),
        device=torch.device("cuda:0"),
        dtype=torch.bfloat16,
        metadata={"enable_talker": False, "runtime_mode": "default"},
    )
    sample = Qwen25OmniSample(
        sample_id=f"train-{plugin}",
        conversation=[
            {
                "role": "system",
                "content": [{"type": "text", "text": DEFAULT_AUDIO_SYSTEM_PROMPT}],
            },
            {
                "role": "user",
                "content": [
                    {"type": "audio", "audio": str(audio_path)},
                    {"type": "text", "text": "Briefly describe the sound."},
                ],
            },
        ],
    )
    batch = facade.processor_adapter.prepare_inputs(processor, sample, phase="train")
    audio_input = batch.model_inputs["input_features"].detach().requires_grad_(True)
    batch.model_inputs["input_features"] = audio_input
    labels = batch.model_inputs["input_ids"].clone()
    labels[labels == model.thinker.config.audio_token_index] = -100
    batch.model_inputs["labels"] = labels
    state = facade.state_builder.build(batch, phase="train")
    config = compression_config(_stage(plugin))
    controller = activate_runtime_compression(
        config=config,
        facade=facade,
        runtime=runtime,
        compression_mode=CompressionMode.COMPRESSED,
        execution_phase="forward",
    )

    try:
        output = facade.runner.forward_train(runtime, state)
        assert output.loss is not None and output.loss.requires_grad
        output.loss.backward()
    finally:
        assert controller is not None
        controller.uninstall()

    assert audio_input.grad is not None
    assert torch.isfinite(audio_input.grad).all()
    assert int(torch.count_nonzero(audio_input.grad).item()) > 0
    report = runtime.metadata["last_training_forward"]["compression"]
    assert report["plugin"] == plugin
    assert report["execution_phase"] == "forward"
    assert report["kept_audio_tokens"] < report["original_audio_tokens"]
    assert runtime.metadata["last_training_forward"]["timing"] is None

    report_path = Path(
        os.environ.get(
            "QWEN25_OMNI_COMPRESSED_TRAIN_SMOKE_REPORT_DIR",
            str(tmp_path),
        )
    ) / f"{plugin}.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(
            {
                "plugin": plugin,
                "gradient_checkpointing": gradient_checkpointing,
                "loss": float(output.loss.detach().float().cpu()),
                "peak_memory_bytes": int(torch.cuda.max_memory_allocated()),
                "nonzero_audio_input_gradients": int(
                    torch.count_nonzero(audio_input.grad).item()
                ),
                "compression": report,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
