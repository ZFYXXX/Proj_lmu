"""Opt-in real-checkpoint smoke for Qwen2.5-Omni encoder-level A-ToMe."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
from tests.fixtures.qwen25_omni_audio import (
    compression_config, create_audio as _create_audio,
)
from tests.fixtures.qwen25_omni_audio import encoder_stage as _stage

from mllm_comp.backbones.base import BackboneRuntime
from mllm_comp.backbones.qwen25_omni import (
    DEFAULT_AUDIO_SYSTEM_PROMPT,
    Qwen25OmniSample,
    create_qwen25_omni_backbone,
)
from mllm_comp.core.vocabulary import CompressionMode
from mllm_comp.pipeline.runtime_compression import activate_runtime_compression


def _run_checkpoint_smoke(
    tmp_path: Path,
    *,
    device_spec: str,
    torch_dtype: object,
    attn_implementation: str,
    max_new_tokens: int,
    report_environment_variable: str,
) -> None:
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    assert transformers.__version__ == "4.57.3"

    model_path_value = os.environ.get("QWEN25_OMNI_MODEL_PATH")
    if not model_path_value:
        pytest.fail("QWEN25_OMNI_MODEL_PATH must point to a local checkpoint.")
    model_path = Path(model_path_value)
    assert model_path.is_dir()
    if device_spec.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()

    audio_path = tmp_path / "atome.wav"
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
        metadata={"enable_talker": False, "runtime_mode": "default"},
    )
    sample = Qwen25OmniSample(
        sample_id="atome-smoke",
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
    session = activate_runtime_compression(
        config=compression_config(_stage()),
        facade=facade,
        runtime=runtime,
        compression_mode=CompressionMode.COMPRESSED,
    )
    assert session is not None

    try:
        result = facade.runner.generate(
            runtime,
            state,
            processed_batch=batch,
            return_audio=False,
            thinker_max_new_tokens=max_new_tokens,
            thinker_do_sample=False,
        )
    finally:
        session.uninstall()

    assert result.text is not None
    generated = [[int(token) for token in row] for row in result.text.token_ids.tolist()]
    assert generated and all(generated)
    report = result.metadata["compression"]
    assert report["plugin"] == "adjacent_token_merging_audio"
    assert report["placement"] == "encoder_level"
    assert report["final_encoder_tokens"] < report["initial_encoder_tokens"]
    assert report["kept_audio_tokens"] < report["original_audio_tokens"]
    assert report["layers"] is None
    assert result.metadata["timing"] is None

    output = {
        "transformers": transformers.__version__,
        "torch": torch.__version__,
        "checkpoint": str(model_path.resolve()),
        "device": device_spec,
        "torch_dtype": str(torch_dtype),
        "attention_implementation": attn_implementation,
        "peak_memory_bytes": (
            int(torch.cuda.max_memory_allocated())
            if device_spec.startswith("cuda")
            else None
        ),
        "generated_tokens": generated,
        "compression": report,
    }
    report_path = Path(
        os.environ.get(
            report_environment_variable,
            str(tmp_path / "qwen25_omni_atome_smoke.json"),
        )
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(output, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


@pytest.mark.gpu
@pytest.mark.skipif(
    os.environ.get("MLLM_COMP_RUN_QWEN25_OMNI_ATOME_GPU_SMOKE") != "1",
    reason="Set MLLM_COMP_RUN_QWEN25_OMNI_ATOME_GPU_SMOKE=1 to run.",
)
def test_qwen25_omni_atome_default_mode_checkpoint_gpu_smoke(
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
        report_environment_variable="QWEN25_OMNI_ATOME_SMOKE_REPORT",
    )


@pytest.mark.skipif(
    os.environ.get("MLLM_COMP_RUN_QWEN25_OMNI_ATOME_CPU_SMOKE") != "1",
    reason="Set MLLM_COMP_RUN_QWEN25_OMNI_ATOME_CPU_SMOKE=1 to run.",
)
def test_qwen25_omni_atome_default_mode_checkpoint_cpu_smoke(
    tmp_path: Path,
) -> None:
    torch = pytest.importorskip("torch")
    torch.set_num_threads(int(os.environ.get("MLLM_COMP_CPU_SMOKE_THREADS", "16")))
    _run_checkpoint_smoke(
        tmp_path,
        device_spec="cpu",
        torch_dtype=torch.float32,
        attn_implementation="eager",
        max_new_tokens=2,
        report_environment_variable="QWEN25_OMNI_ATOME_CPU_SMOKE_REPORT",
    )
