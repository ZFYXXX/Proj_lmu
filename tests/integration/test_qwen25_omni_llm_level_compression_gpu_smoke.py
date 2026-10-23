"""Opt-in real-checkpoint smoke for Qwen2.5-Omni LLM-level compression.

This test deliberately covers lifecycle behavior that tensor-only unit tests
cannot prove: Audio-Pyramid performs one prefill selection and bypasses every
decode step, while structured layer pruning produces a trainable and
reloadable Transformers checkpoint.
"""

from __future__ import annotations

import gc
import hashlib
import json
import os
from pathlib import Path

import pytest
from tests.fixtures.qwen25_omni_audio import (
    compression_config as _config, create_audio as _create_audio,
)

from mllm_comp.backbones.base import BackboneRuntime
from mllm_comp.backbones.qwen25_omni import (
    DEFAULT_AUDIO_SYSTEM_PROMPT,
    Qwen25OmniSample,
    create_qwen25_omni_backbone,
    export_qwen25_omni_operation_checkpoint,
)
from mllm_comp.core.vocabulary import CompressionMode
from mllm_comp.pipeline.runtime_compression import activate_runtime_compression
from mllm_comp.pipeline.stage import CompressionStageConfig


_PYRAMID_PLUGIN = "audio_pyramid_attention_pruning"
_SPADE_PLUGIN = "structured_llm_layer_pruning"


def _sample(
    audio_path: Path,
    *,
    sample_id: str,
    response_text: str | None = None,
) -> Qwen25OmniSample:
    conversation = [
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
    ]
    if response_text is not None:
        conversation.append(
            {
                "role": "assistant",
                "content": [{"type": "text", "text": response_text}],
            }
        )
    return Qwen25OmniSample(
        sample_id=sample_id,
        conversation=conversation,
    )


def _audio_pyramid_stage(*, phase: str) -> CompressionStageConfig:
    return CompressionStageConfig.from_dict(
        {
            "name": "audio_pyramid_prefill",
            "placement": "llm_level",
            "execution_phase": phase,
            "plugin": _PYRAMID_PLUGIN,
            "algorithm": _PYRAMID_PLUGIN,
            "variant": f"{_PYRAMID_PLUGIN}.audio",
            "binding": f"{_PYRAMID_PLUGIN}.audio.qwen25_omni",
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
                "layers": [8, 17, 26],
                "lifecycle": "prefill_only",
            },
        }
    )


def _spade_stage() -> CompressionStageConfig:
    return CompressionStageConfig.from_dict(
        {
            "name": "spade_operation_only",
            "placement": "llm_level",
            "execution_phase": "prepare",
            "plugin": _SPADE_PLUGIN,
            "algorithm": _SPADE_PLUGIN,
            "variant": f"{_SPADE_PLUGIN}.audio",
            "binding": f"{_SPADE_PLUGIN}.audio.qwen25_omni",
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
            "activation_scope": {"layers": [8, 17, 26]},
        }
    )


def _runtime(torch, *, model, processor, model_path: Path) -> BackboneRuntime:
    return BackboneRuntime(
        model=model,
        processor=processor,
        model_path_or_id=str(model_path),
        device=torch.device("cuda:0"),
        dtype=torch.bfloat16,
        metadata={"enable_talker": False, "runtime_mode": "default"},
    )


def _training_state(torch, facade, processor, sample):
    batch = facade.processor_adapter.prepare_inputs(processor, sample, phase="train")
    audio_input = batch.model_inputs["input_features"].detach().requires_grad_(True)
    batch.model_inputs["input_features"] = audio_input
    labels = batch.model_inputs["input_ids"].clone()
    semantic = batch.metadata["semantic_token_labels_by_sample"]
    response_mask = torch.tensor(
        [[item == "response" for item in row] for row in semantic],
        dtype=torch.bool,
        device=labels.device,
    )
    assert bool(response_mask.any().item())
    labels[~response_mask] = -100
    batch.model_inputs["labels"] = labels
    return batch, facade.state_builder.build(batch, phase="train"), audio_input


def _assert_finite_nonzero_gradient(torch, value) -> int:
    assert value.grad is not None
    assert torch.isfinite(value.grad).all()
    nonzero = int(torch.count_nonzero(value.grad).item())
    assert nonzero > 0
    return nonzero


def _generation_kwargs() -> dict[str, object]:
    return {
        "return_audio": False,
        "thinker_min_new_tokens": 3,
        "thinker_max_new_tokens": 4,
        "thinker_do_sample": False,
        "thinker_use_cache": True,
    }


@pytest.mark.gpu
@pytest.mark.skipif(
    os.environ.get("MLLM_COMP_RUN_QWEN25_OMNI_LLM_LEVEL_GPU_SMOKE") != "1",
    reason="Set MLLM_COMP_RUN_QWEN25_OMNI_LLM_LEVEL_GPU_SMOKE=1 to run.",
)
def test_llm_level_eval_train_and_operation_export_reload(
    tmp_path: Path,
) -> None:
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    assert transformers.__version__ == "4.57.3"
    assert os.environ.get("CUDA_VISIBLE_DEVICES") is not None
    assert torch.cuda.is_available() and torch.cuda.device_count() == 1
    torch.cuda.reset_peak_memory_stats()

    model_path = Path(os.environ["QWEN25_OMNI_MODEL_PATH"])
    assert model_path.is_dir()
    audio_path = tmp_path / "llm-level.wav"
    _create_audio(audio_path)
    facade = create_qwen25_omni_backbone()
    processor = facade.processor_adapter.load_processor(
        str(model_path), local_files_only=True
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
    runtime = _runtime(
        torch, model=model, processor=processor, model_path=model_path
    )
    sample = _sample(audio_path, sample_id="llm-level-eval")

    eval_batch = facade.processor_adapter.prepare_inputs(
        processor, sample, phase="eval"
    )
    eval_state = facade.state_builder.build(eval_batch, phase="eval")
    pyramid = activate_runtime_compression(
        config=_config(_audio_pyramid_stage(phase="generate")),
        facade=facade,
        runtime=runtime,
        compression_mode=CompressionMode.COMPRESSED,
    )
    try:
        pyramid_generation = facade.runner.generate(
            runtime,
            eval_state,
            processed_batch=eval_batch,
            **_generation_kwargs(),
        )
    finally:
        assert pyramid is not None
        pyramid.uninstall()
    pyramid_report = pyramid_generation.metadata["compression"]
    assert pyramid_report["plugin"] == _PYRAMID_PLUGIN
    assert pyramid_report["prefill_forward_calls"] == 1
    assert pyramid_report["decode_forward_calls"] >= 2
    assert pyramid_report["decode_bypassed"] is True
    assert pyramid_report["selection_repeated_during_decode"] is False
    assert pyramid_report["kept_audio_tokens"] < pyramid_report["original_audio_tokens"]
    assert len(pyramid_report["stages"]) == 3

    for parameter in model.parameters():
        parameter.requires_grad_(False)
    model.gradient_checkpointing_enable()
    model.train()
    train_sample = _sample(
        audio_path,
        sample_id="llm-level-train",
        response_text="A steady electronic tone is audible.",
    )
    train_batch, train_state, pyramid_audio_input = _training_state(
        torch, facade, processor, train_sample
    )
    del train_batch
    pyramid_train = activate_runtime_compression(
        config=_config(_audio_pyramid_stage(phase="forward")),
        facade=facade,
        runtime=runtime,
        compression_mode=CompressionMode.COMPRESSED,
        execution_phase="forward",
    )
    try:
        pyramid_output = facade.runner.forward_train(runtime, train_state)
        assert pyramid_output.loss is not None
        assert pyramid_output.loss.requires_grad
        assert torch.isfinite(pyramid_output.loss.detach()).all()
        pyramid_output.loss.backward()
        pyramid_nonzero_gradients = _assert_finite_nonzero_gradient(
            torch, pyramid_audio_input
        )
        pyramid_train_report = runtime.metadata["last_training_forward"][
            "compression"
        ]
        pyramid_loss = float(pyramid_output.loss.detach().float().cpu())
    finally:
        assert pyramid_train is not None
        pyramid_train.uninstall()
    assert pyramid_train_report["prefill_forward_calls"] == 1
    assert pyramid_train_report["decode_forward_calls"] == 0
    assert pyramid_train_report["kept_audio_tokens"] < pyramid_train_report[
        "original_audio_tokens"
    ]

    model.gradient_checkpointing_disable()
    model.eval()
    source_num_layers = len(model.thinker.model.layers)
    operation = activate_runtime_compression(
        config=_config(_spade_stage()),
        facade=facade,
        runtime=runtime,
        compression_mode=CompressionMode.COMPRESSED,
        execution_phase="prepare",
    )
    assert operation is not None
    operation_activation = operation.activations[0].activation
    apply_result = operation_activation.apply_result
    assert apply_result is not None
    assert not apply_result.violations
    assert len(model.thinker.model.layers) == source_num_layers - 3
    assert apply_result.metadata["parameter_reduction"] > 0
    assert apply_result.metadata["state_dict_layer_indices_contiguous"] is True
    assert apply_result.metadata["config_layer_types_reindexed"] is True

    spade_generation = facade.runner.generate(
        runtime,
        eval_state,
        processed_batch=eval_batch,
        **_generation_kwargs(),
    )
    assert spade_generation.text is not None
    spade_session_report = spade_generation.metadata["compression"]
    spade_report = spade_session_report["activations"][0]
    assert spade_report["algorithm"] == _SPADE_PLUGIN
    assert spade_report["result_num_hidden_layers"] == source_num_layers - 3

    model.train()
    spade_batch, spade_state, spade_audio_input = _training_state(
        torch, facade, processor, train_sample
    )
    del spade_batch
    spade_output = facade.runner.forward_train(runtime, spade_state)
    assert spade_output.loss is not None and spade_output.loss.requires_grad
    assert torch.isfinite(spade_output.loss.detach()).all()
    spade_output.loss.backward()
    spade_nonzero_gradients = _assert_finite_nonzero_gradient(torch, spade_audio_input)
    spade_loss = float(spade_output.loss.detach().float().cpu())

    if os.environ.get("MLLM_COMP_QWEN25_OMNI_LLM_LEVEL_MINIMAL_ONLY") == "1":
        report_dir = Path(
            os.environ.get("QWEN25_OMNI_LLM_LEVEL_SMOKE_REPORT_DIR", str(tmp_path))
        )
        report_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "transformers": transformers.__version__,
            "torch": torch.__version__,
            "checkpoint": str(model_path.resolve()),
            "accelerator": torch.cuda.get_device_name(0),
            "peak_memory_bytes": int(torch.cuda.max_memory_allocated()),
            "audio_pyramid_generate": pyramid_report,
            "audio_pyramid_train": {
                "loss": pyramid_loss,
                "nonzero_audio_input_gradients": pyramid_nonzero_gradients,
                "compression": pyramid_train_report,
            },
            "spade_generate": spade_report,
            "spade_train": {
                "loss": spade_loss,
                "nonzero_audio_input_gradients": spade_nonzero_gradients,
            },
            "spade_apply": apply_result.metadata,
            "checkpoint_exported": False,
        }
        (report_dir / "llm_level_minimal_eval_train.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(json.dumps(payload, ensure_ascii=False))
        return

    model.eval()
    export_dir = tmp_path / "spade-export"
    export_manifest = export_qwen25_omni_operation_checkpoint(
        model=model,
        processor=processor,
        output_dir=export_dir,
        apply_result=apply_result,
        source_model_dir=model_path,
    )
    manifest_path = Path(export_manifest["manifest_path"])
    assert export_manifest["manifest_sha256"] == hashlib.sha256(
        manifest_path.read_bytes()
    ).hexdigest()
    assert export_manifest["operation_compression"]["artifact_scope"] == (
        "thinker_only"
    )
    assert export_manifest["auxiliary_artifacts"] == ["spk_dict.pt"]

    del (
        eval_state,
        pyramid,
        pyramid_audio_input,
        pyramid_generation,
        pyramid_output,
        pyramid_train,
        runtime,
        spade_audio_input,
        spade_generation,
        spade_output,
        spade_state,
        train_state,
        model,
    )
    gc.collect()
    torch.cuda.empty_cache()

    reloaded_processor = facade.processor_adapter.load_processor(
        str(export_dir), local_files_only=True
    )
    reloaded_model = facade.model_adapter.load_model(
        str(export_dir),
        train=False,
        local_files_only=True,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map="cuda:0",
        enable_talker=False,
    )
    assert reloaded_model.config.enable_audio_output is False
    assert reloaded_model.has_talker is False
    assert len(reloaded_model.thinker.model.layers) == source_num_layers - 3
    assert (
        reloaded_model.thinker.model.config.num_hidden_layers
        == source_num_layers - 3
    )
    assert [
        int(layer.self_attn.layer_idx)
        for layer in reloaded_model.thinker.model.layers
    ] == list(range(source_num_layers - 3))
    reloaded_runtime = _runtime(
        torch,
        model=reloaded_model,
        processor=reloaded_processor,
        model_path=export_dir,
    )
    reloaded_batch = facade.processor_adapter.prepare_inputs(
        reloaded_processor, sample, phase="eval"
    )
    reloaded_state = facade.state_builder.build(reloaded_batch, phase="eval")
    reloaded_generation = facade.runner.generate(
        reloaded_runtime,
        reloaded_state,
        processed_batch=reloaded_batch,
        **_generation_kwargs(),
    )
    assert reloaded_generation.text is not None
    assert int(reloaded_generation.text.token_ids.numel()) >= 3

    report_dir = Path(
        os.environ.get("QWEN25_OMNI_LLM_LEVEL_SMOKE_REPORT_DIR", str(tmp_path))
    )
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "llm_level_eval_train_export_reload.json").write_text(
        json.dumps(
            {
                "transformers": transformers.__version__,
                "torch": torch.__version__,
                "checkpoint": str(model_path.resolve()),
                "accelerator": torch.cuda.get_device_name(0),
                "audio_pyramid_generate": pyramid_report,
                "audio_pyramid_train": {
                    "loss": pyramid_loss,
                    "nonzero_audio_input_gradients": pyramid_nonzero_gradients,
                    "compression": pyramid_train_report,
                },
                "spade_generate": spade_report,
                "spade_train": {
                    "loss": spade_loss,
                    "nonzero_audio_input_gradients": spade_nonzero_gradients,
                },
                "spade_apply": apply_result.metadata,
                "export": {
                    "manifest_sha256": export_manifest["manifest_sha256"],
                    "manifest_payload_sha256": export_manifest[
                        "manifest_payload_sha256"
                    ],
                    "reload_num_hidden_layers": len(
                        reloaded_model.thinker.model.layers
                    ),
                    "reload_generated_tokens": int(
                        reloaded_generation.text.token_ids.numel()
                    ),
                },
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
