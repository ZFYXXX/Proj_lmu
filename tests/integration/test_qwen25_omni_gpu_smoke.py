"""Opt-in real Qwen2.5-Omni smoke on one assigned GPU.

Run only with ``MLLM_COMP_RUN_QWEN25_OMNI_GPU_SMOKE=1`` and launch the Python
process exposing one assigned GPU through ``CUDA_VISIBLE_DEVICES``. It is
logical cuda:0 inside the process.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import struct
import subprocess
import wave
from pathlib import Path
from typing import Any

import pytest

from mllm_comp.backbones.base import BackboneRuntime, TraceRequest
from mllm_comp.backbones.qwen25_omni import (
    DEFAULT_AUDIO_SYSTEM_PROMPT,
    Qwen25OmniSample,
    create_qwen25_omni_backbone,
)


pytestmark = pytest.mark.gpu


def _enabled() -> bool:
    return os.environ.get("MLLM_COMP_RUN_QWEN25_OMNI_GPU_SMOKE") == "1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _shape(value: Any) -> list[int] | None:
    shape = getattr(value, "shape", None)
    return None if shape is None else [int(dim) for dim in shape]


def _create_audio(path: Path, *, sample_rate: int = 16000, seconds: float = 1.0) -> None:
    frames = int(sample_rate * seconds)
    with wave.open(str(path), "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(sample_rate)
        samples = (
            struct.pack("<h", int(0.15 * 32767 * math.sin(2 * math.pi * 440 * i / sample_rate)))
            for i in range(frames)
        )
        output.writeframes(b"".join(samples))


def _run_ffmpeg(arguments: list[str]) -> None:
    executable = shutil.which("ffmpeg")
    if executable is None:
        pytest.fail("ffmpeg is required to create deterministic video smoke fixtures.")
    subprocess.run(
        [executable, "-hide_banner", "-loglevel", "error", "-y", *arguments],
        check=True,
    )


def _create_fixtures(root: Path) -> dict[str, Path]:
    image_path = root / "qwen25_omni_smoke.png"
    audio_path = root / "qwen25_omni_smoke.wav"
    silent_video_path = root / "qwen25_omni_silent.mp4"
    av_video_path = root / "qwen25_omni_av.mp4"

    image_module = pytest.importorskip("PIL.Image")
    image_module.new("RGB", (224, 224), color=(32, 96, 192)).save(image_path)
    _create_audio(audio_path)
    _run_ffmpeg(
        [
            "-f",
            "lavfi",
            "-i",
            "color=c=blue:s=224x224:r=4:d=1",
            "-an",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            str(silent_video_path),
        ]
    )
    _run_ffmpeg(
        [
            "-f",
            "lavfi",
            "-i",
            "color=c=green:s=224x224:r=4:d=1",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=440:sample_rate=16000:duration=1",
            "-shortest",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            str(av_video_path),
        ]
    )
    return {
        "image": image_path,
        "audio": audio_path,
        "silent_video": silent_video_path,
        "av_video": av_video_path,
    }


def _conversation(content: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "role": "system",
            "content": [{"type": "text", "text": DEFAULT_AUDIO_SYSTEM_PROMPT}],
        },
        {"role": "user", "content": content},
    ]


@pytest.mark.skipif(not _enabled(), reason="real Qwen2.5-Omni GPU smoke is opt-in")
def test_qwen25_omni_all_native_modalities_and_speech(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    pytest.importorskip("qwen_omni_utils")
    assert torch.cuda.is_available()
    assert torch.cuda.device_count() == 1

    model_path_value = os.environ.get("QWEN25_OMNI_MODEL_PATH")
    if not model_path_value:
        pytest.fail("QWEN25_OMNI_MODEL_PATH must point to an existing local checkpoint.")
    model_path = Path(model_path_value)
    assert model_path.is_dir()

    fixtures = _create_fixtures(tmp_path)
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
        device_map={"": 0},
        enable_talker=True,
    )
    runtime = BackboneRuntime(
        model=model,
        processor=processor,
        model_path_or_id=str(model_path),
        device=torch.device("cuda:0"),
        dtype=torch.bfloat16,
        metadata={"speech_sample_rate": 24000},
    )

    cases = {
        "text": Qwen25OmniSample(
            "text",
            _conversation([{"type": "text", "text": "Reply with one short sentence."}]),
        ),
        "image": Qwen25OmniSample(
            "image",
            _conversation(
                [
                    {"type": "image", "image": str(fixtures["image"])},
                    {"type": "text", "text": "What is the dominant color?"},
                ]
            ),
        ),
        "audio": Qwen25OmniSample(
            "audio",
            _conversation(
                [
                    {"type": "audio", "audio": str(fixtures["audio"])},
                    {"type": "text", "text": "Briefly describe the sound."},
                ]
            ),
        ),
        "silent_video": Qwen25OmniSample(
            "silent-video",
            _conversation(
                [
                    {"type": "video", "video": str(fixtures["silent_video"])},
                    {"type": "text", "text": "What color dominates the video?"},
                ]
            ),
            use_audio_in_video=False,
        ),
        "av_video": Qwen25OmniSample(
            "av-video",
            _conversation(
                [
                    {"type": "video", "video": str(fixtures["av_video"])},
                    {"type": "text", "text": "Describe the video and its sound briefly."},
                ]
            ),
            use_audio_in_video=True,
        ),
    }

    report: dict[str, Any] = {
        "environment": {
            "transformers": transformers.__version__,
            "torch": torch.__version__,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "logical_device": "cuda:0",
            "gpu_name": torch.cuda.get_device_name(0),
            "dtype": "bfloat16",
            "model_path": str(model_path),
            "config_sha256": _sha256(model_path / "config.json"),
        },
        "fixtures": {name: _sha256(path) for name, path in fixtures.items()},
        "cases": {},
    }

    for name, sample in cases.items():
        batch = facade.processor_adapter.prepare_inputs(processor, sample, phase="eval")
        state = facade.state_builder.build(batch, phase="eval")
        result = facade.runner.generate(
            runtime,
            state,
            processed_batch=batch,
            return_audio=False,
            thinker_max_new_tokens=16,
            thinker_do_sample=False,
        )
        assert result.text is not None
        assert result.text.text[0].strip()
        report["cases"][name] = {
            "processor_contract": batch.metadata["processor_output_contract"],
            "spans": [
                {
                    "modality": span.modality,
                    "start": span.start,
                    "end": span.end,
                }
                for span in state.modality_token_spans
            ],
            "generated_token_shape": _shape(result.text.token_ids),
            "text": result.text.text,
        }

    speech_sample = cases["text"]
    speech_batch = facade.processor_adapter.prepare_inputs(processor, speech_sample, phase="eval")
    speech_state = facade.state_builder.build(speech_batch, phase="eval")
    speech_result = facade.runner.generate(
        runtime,
        speech_state,
        processed_batch=speech_batch,
        trace_request=TraceRequest(
            thinker_reply_part=True,
            token2wav_intermediates=True,
        ),
        return_audio=True,
        speaker="Chelsie",
        thinker_max_new_tokens=8,
        thinker_do_sample=False,
        talker_max_new_tokens=64,
        talker_do_sample=False,
        token2wav_num_steps=2,
    )
    assert speech_result.speech is not None
    assert speech_result.speech.sample_rate == 24000
    assert speech_result.speech.codec_token_ids is not None
    assert speech_result.traces is not None
    assert speech_result.traces.thinker_reply_part is not None
    assert speech_result.traces.talker_codec_token_ids is None
    assert (
        speech_result.traces.token2wav_intermediates["code"]
        is speech_result.speech.codec_token_ids
    )
    report["speech"] = {
        "text": speech_result.text.text,
        "waveform_shape": _shape(speech_result.speech.waveform),
        "codec_token_shape": _shape(speech_result.speech.codec_token_ids),
        "thinker_reply_part_shape": _shape(speech_result.traces.thinker_reply_part),
        "token2wav_code_shape": _shape(
            speech_result.traces.token2wav_intermediates["code"]
        ),
        "sample_rate": speech_result.speech.sample_rate,
        "speaker": speech_result.speech.speaker,
    }

    report_path = Path(
        os.environ.get("QWEN25_OMNI_SMOKE_REPORT", tmp_path / "qwen25_omni_smoke.json")
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    assert report_path.is_file()
