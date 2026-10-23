"""Opt-in real-checkpoint smoke for Qwen2.5-Omni audio pruning."""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
import struct
from types import SimpleNamespace
import wave

import pytest

from mllm_comp.backbones.base import BackboneRuntime
from mllm_comp.backbones.qwen25_omni import (
    DEFAULT_AUDIO_SYSTEM_PROMPT,
    Qwen25OmniSample,
    create_qwen25_omni_backbone,
)
from mllm_comp.core.vocabulary import CompressionMode
from mllm_comp.pipeline.runtime_compression import activate_runtime_compression
from mllm_comp.pipeline.stage import CompressionStageConfig


def _create_audio(path: Path, *, sample_rate: int = 16000, seconds: float = 1.0) -> None:
    frames = int(sample_rate * seconds)
    with wave.open(str(path), "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(sample_rate)
        samples = (
            struct.pack(
                "<h",
                int(0.15 * 32767 * math.sin(2 * math.pi * 440 * index / sample_rate)),
            )
            for index in range(frames)
        )
        output.writeframes(b"".join(samples))


def _stage(*, name: str, keep_ratio: float) -> CompressionStageConfig:
    return CompressionStageConfig.from_dict(
        {
            "name": name,
            "placement": "projector_level",
            "execution_phase": "generate",
            "plugin": "segmentwise_topk_audio",
            "algorithm": "segmentwise_topk_audio",
            "variant": "segmentwise_topk_audio.audio",
            "binding": "segmentwise_topk_audio.audio.qwen25_omni",
            "method": "token_compression",
            "object_family": "tensor_slot",
            "object_subtype": "audio_encoder_output_pre_projector",
            "modalities": ["audio"],
            "target_modalities": ["audio"],
            "conditioning_modalities": [],
            "preserved_modalities": ["text", "image", "video"],
            "coupling_policy": "independent",
            "media_alignment": "temporal",
            "token_roles": ["observation"],
            "operation": {
                "type": "keep",
                "target_site": "audio_encoder_output_pre_projector",
            },
            "criterion": {
                "score_source": "encoder_last_layer_attention",
                "num_segments": 10,
            },
            "schedule": {
                "keep_ratio": keep_ratio,
                "allocation": "balanced_exact",
                "minimum_keep_tokens": 1,
            },
        }
    )


def _tokens(value: object) -> list[list[int]]:
    return [[int(token) for token in row] for row in value.tolist()]


def _run_checkpoint_smoke(
    tmp_path: Path,
    *,
    device_spec: str,
    torch_dtype: object,
    attn_implementation: str,
    max_new_tokens: int,
    report_environment_variable: str,
    require_decoded_output: bool,
) -> None:
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    assert transformers.__version__ == "4.57.3"

    model_path_value = os.environ.get("QWEN25_OMNI_MODEL_PATH")
    if not model_path_value:
        pytest.fail("QWEN25_OMNI_MODEL_PATH must point to an existing local checkpoint.")
    model_path = Path(model_path_value)
    assert model_path.is_dir()
    if device_spec.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()

    audio_path = tmp_path / "segmentwise_topk.wav"
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
        torch_dtype=torch_dtype,
        attn_implementation=attn_implementation,
        device_map=device_spec,
        enable_talker=False,
    )
    runtime = BackboneRuntime(
        model=model,
        processor=processor,
        model_path_or_id=str(model_path),
        device=torch.device(device_spec),
        dtype=torch_dtype,
        metadata={"enable_talker": False, "runtime_mode": "profile"},
    )
    sample = Qwen25OmniSample(
        sample_id="audio-compression",
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
    generation = {
        "return_audio": False,
        "thinker_max_new_tokens": max_new_tokens,
        "thinker_do_sample": False,
    }

    baseline = facade.runner.generate(
        runtime,
        state,
        processed_batch=batch,
        **generation,
    )
    assert baseline.text is not None
    baseline_tokens = _tokens(baseline.text.token_ids)
    assert baseline_tokens and all(baseline_tokens)

    identity_session = activate_runtime_compression(
        config=SimpleNamespace(
            pipeline=SimpleNamespace(
                stages=[_stage(name="identity", keep_ratio=1.0)]
            )
        ),
        facade=facade,
        runtime=runtime,
        compression_mode=CompressionMode.COMPRESSED,
    )
    assert identity_session is not None
    identity = facade.runner.generate(
        runtime,
        state,
        processed_batch=batch,
        **generation,
    )
    assert identity.text is not None
    identity_tokens = _tokens(identity.text.token_ids)
    assert identity_tokens and all(identity_tokens)
    assert identity_tokens == baseline_tokens
    assert identity.metadata["compression"]["achieved_keep_ratio"] == 1.0
    identity_session.uninstall()

    compressed_session = activate_runtime_compression(
        config=SimpleNamespace(
            pipeline=SimpleNamespace(
                stages=[_stage(name="segmentwise-r25", keep_ratio=0.25)]
            )
        ),
        facade=facade,
        runtime=runtime,
        compression_mode=CompressionMode.COMPRESSED,
    )
    assert compressed_session is not None
    compressed = facade.runner.generate(
        runtime,
        state,
        processed_batch=batch,
        **generation,
    )
    compressed_session.uninstall()

    assert compressed.text is not None
    compressed_tokens = _tokens(compressed.text.token_ids)
    assert compressed_tokens and all(compressed_tokens)
    if require_decoded_output:
        assert compressed.text.text[0].strip()
    report = compressed.metadata["compression"]
    assert report["kept_audio_tokens"] < report["original_audio_tokens"]
    assert report["achieved_keep_ratio"] <= 0.25
    assert all(item["indices_chronological"] for item in report["items"])
    assert compressed.metadata["timing"]["llm_input_tokens"] is not None

    output = {
        "transformers": transformers.__version__,
        "torch": torch.__version__,
        "checkpoint": str(model_path.resolve()),
        "device": device_spec,
        "torch_dtype": str(torch_dtype),
        "attention_implementation": attn_implementation,
        "max_new_tokens": max_new_tokens,
        "accelerator": (
            torch.cuda.get_device_name(0) if device_spec.startswith("cuda") else "cpu"
        ),
        "peak_memory_bytes": (
            max(
                int(item["cuda_max_memory_allocated_bytes"])
                for item in (
                    baseline.metadata["timing"],
                    identity.metadata["timing"],
                    compressed.metadata["timing"],
                )
            )
            if device_spec.startswith("cuda")
            else None
        ),
        "baseline_tokens": baseline_tokens,
        "identity_tokens": identity_tokens,
        "compressed_tokens": compressed_tokens,
        "baseline_timing": baseline.metadata["timing"],
        "identity_timing": identity.metadata["timing"],
        "compressed_timing": compressed.metadata["timing"],
        "compression": report,
    }
    report_path = Path(
        os.environ.get(
            report_environment_variable,
            str(tmp_path / "qwen25_omni_audio_compression_smoke.json"),
        )
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(output, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


@pytest.mark.gpu
@pytest.mark.skipif(
    os.environ.get("MLLM_COMP_RUN_QWEN25_OMNI_AUDIO_COMPRESSION_GPU_SMOKE") != "1",
    reason="Set MLLM_COMP_RUN_QWEN25_OMNI_AUDIO_COMPRESSION_GPU_SMOKE=1 to run.",
)
def test_qwen25_omni_audio_compression_identity_and_r25_gpu(
    tmp_path: Path,
) -> None:
    torch = pytest.importorskip("torch")
    assert os.environ.get("CUDA_VISIBLE_DEVICES") is not None
    assert torch.cuda.is_available()
    assert torch.cuda.device_count() == 1
    _run_checkpoint_smoke(
        tmp_path,
        device_spec="cuda:0",
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        max_new_tokens=8,
        report_environment_variable="QWEN25_OMNI_AUDIO_COMPRESSION_SMOKE_REPORT",
        require_decoded_output=True,
    )


@pytest.mark.skipif(
    os.environ.get("MLLM_COMP_RUN_QWEN25_OMNI_AUDIO_COMPRESSION_CPU_SMOKE") != "1",
    reason="Set MLLM_COMP_RUN_QWEN25_OMNI_AUDIO_COMPRESSION_CPU_SMOKE=1 to run.",
)
def test_qwen25_omni_audio_compression_identity_and_r25_cpu(
    tmp_path: Path,
) -> None:
    torch = pytest.importorskip("torch")
    torch.set_num_threads(int(os.environ.get("MLLM_COMP_CPU_SMOKE_THREADS", "8")))
    _run_checkpoint_smoke(
        tmp_path,
        device_spec="cpu",
        torch_dtype=torch.float32,
        attn_implementation="eager",
        max_new_tokens=2,
        report_environment_variable="QWEN25_OMNI_AUDIO_COMPRESSION_CPU_SMOKE_REPORT",
        require_decoded_output=False,
    )
