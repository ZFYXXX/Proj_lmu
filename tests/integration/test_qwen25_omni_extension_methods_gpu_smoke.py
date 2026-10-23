"""Opt-in one-load checkpoint smoke for the four framework-extension methods."""

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


_METHODS = {
    "local_temporal_bipartite_merging_audio": {
        "name": "audio_ltbm_eval",
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
        "schedule": {"keep_ratio": 0.5},
    },
    "similarity_attention_merge_prune_audio": {
        "name": "audio_sa_map_eval",
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
        "schedule": {"keep_ratio": 0.5},
    },
    "encoder_attention_topk_audio": {
        "name": "audio_encoder_topk_eval",
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
        "name": "audio_progressive_pruning_eval",
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


def _stage(plugin: str) -> CompressionStageConfig:
    return CompressionStageConfig.from_dict(
        {
            **_METHODS[plugin],
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


@pytest.mark.gpu
@pytest.mark.skipif(
    os.environ.get("MLLM_COMP_RUN_QWEN25_OMNI_EXTENSION_GPU_SMOKE") != "1",
    reason="Set MLLM_COMP_RUN_QWEN25_OMNI_EXTENSION_GPU_SMOKE=1 to run.",
)
def test_four_extension_methods_generate_from_real_checkpoint(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    assert transformers.__version__ == "4.57.3"
    assert torch.cuda.is_available() and torch.cuda.device_count() == 1
    model_path = Path(os.environ["QWEN25_OMNI_MODEL_PATH"])
    assert model_path.is_dir()

    audio_path = tmp_path / "extension-methods.wav"
    _create_audio(audio_path)
    facade = create_qwen25_omni_backbone()
    processor = facade.processor_adapter.load_processor(
        str(model_path),
        local_files_only=True,
    )
    model = facade.model_adapter.load_model(
        str(model_path),
        train=False,
        local_files_only=True,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map="cuda:0",
        enable_talker=False,
    )
    runtime = BackboneRuntime(
        model=model,
        processor=processor,
        model_path_or_id=str(model_path),
        device=torch.device("cuda:0"),
        dtype=torch.bfloat16,
        metadata={"enable_talker": False, "runtime_mode": "default"},
    )
    sample = Qwen25OmniSample(
        sample_id="extension-methods",
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
    batch = facade.processor_adapter.prepare_inputs(processor, sample, phase="eval")
    state = facade.state_builder.build(batch, phase="eval")
    reports: dict[str, object] = {}
    for plugin in _METHODS:
        config = compression_config(_stage(plugin))
        controller = activate_runtime_compression(
            config=config,
            facade=facade,
            runtime=runtime,
            compression_mode=CompressionMode.COMPRESSED,
        )
        try:
            result = facade.runner.generate(
                runtime,
                state,
                processed_batch=batch,
                return_audio=False,
                thinker_max_new_tokens=1,
                thinker_do_sample=False,
            )
        finally:
            assert controller is not None
            controller.uninstall()
        assert result.text is not None
        assert int(result.text.token_ids.numel()) >= 1
        report = result.metadata["compression"]
        assert report["plugin"] == plugin
        assert 0 < report["kept_audio_tokens"] < report["original_audio_tokens"]
        reports[plugin] = report

    report_dir = Path(
        os.environ.get("QWEN25_OMNI_EXTENSION_SMOKE_REPORT_DIR", str(tmp_path))
    )
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "eval.json").write_text(
        json.dumps(reports, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
