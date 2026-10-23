"""Opt-in real Qwen2.5-VL text/image/video baseline smoke on GPU 1."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest
from functools import partial
from tests.fixtures.native_image import conversation

from mllm_comp.backbones.base import BackboneRuntime, TraceRequest
from mllm_comp.backbones.qwen25_vl import (
    DEFAULT_SYSTEM_PROMPT,
    Qwen25VLSample,
    create_qwen25_vl_backbone,
)


pytestmark = pytest.mark.gpu


def _enabled() -> bool:
    return os.environ.get("MLLM_COMP_RUN_QWEN25_VL_GPU_SMOKE") == "1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _shape(value: Any) -> list[int] | None:
    shape = getattr(value, "shape", None)
    return None if shape is None else [int(dim) for dim in shape]


def _create_fixtures(root: Path) -> dict[str, Path]:
    image_path = root / "qwen25_vl_smoke.png"
    video_path = root / "qwen25_vl_smoke.mp4"
    image_module = pytest.importorskip("PIL.Image")
    image_module.new("RGB", (256, 256), color=(32, 96, 192)).save(image_path)
    environment_ffmpeg = Path(sys.executable).with_name("ffmpeg")
    ffmpeg = shutil.which("ffmpeg") or (
        str(environment_ffmpeg) if environment_ffmpeg.is_file() else None
    )
    if ffmpeg is None:
        pytest.fail("ffmpeg is required for the deterministic video fixture.")
    subprocess.run(
        [
            ffmpeg,
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-f",
            "lavfi",
            "-i",
            "color=c=yellow:s=256x256:r=4:d=1",
            "-an",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            str(video_path),
        ],
        check=True,
    )
    return {"image": image_path, "video": video_path}


_conversation = partial(conversation, system_prompt=DEFAULT_SYSTEM_PROMPT)


@pytest.mark.skipif(not _enabled(), reason="real Qwen2.5-VL GPU smoke is opt-in")
def test_qwen25_vl_all_native_modalities_and_text_traces(tmp_path: Path) -> None:
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == "1", (
        "Launch with physical GPU 1 only: CUDA_VISIBLE_DEVICES=1."
    )
    os.environ.setdefault("FORCE_QWENVL_VIDEO_READER", "torchcodec")
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    pytest.importorskip("qwen_vl_utils")
    from transformers import Qwen2_5_VLForConditionalGeneration  # noqa: F401

    assert torch.cuda.is_available()
    assert torch.cuda.device_count() == 1
    model_path_value = os.environ.get("QWEN25_VL_MODEL_PATH")
    if not model_path_value:
        pytest.fail("QWEN25_VL_MODEL_PATH must point to a local checkpoint.")
    model_path = Path(model_path_value)
    assert model_path.is_dir()

    fixtures = _create_fixtures(tmp_path)
    facade = create_qwen25_vl_backbone()
    processor = facade.processor_adapter.load_processor(
        str(model_path), local_files_only=True
    )
    model = facade.model_adapter.load_model(
        str(model_path),
        train=False,
        local_files_only=True,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map={"": 0},
    )
    runtime = BackboneRuntime(
        model=model,
        processor=processor,
        model_path_or_id=str(model_path),
        device=torch.device("cuda:0"),
        dtype=torch.bfloat16,
        metadata={"enable_talker": False, "speech_sample_rate": None},
    )
    cases = {
        "text": Qwen25VLSample(
            "text",
            _conversation(
                [{"type": "text", "text": "Reply with one short greeting."}]
            ),
        ),
        "image": Qwen25VLSample(
            "image",
            _conversation(
                [
                    {"type": "image", "image": str(fixtures["image"])},
                    {"type": "text", "text": "What is the dominant color?"},
                ]
            ),
        ),
        "video": Qwen25VLSample(
            "video",
            _conversation(
                [
                    {
                        "type": "video",
                        "video": str(fixtures["video"]),
                        "fps": 4.0,
                    },
                    {"type": "text", "text": "What color dominates the video?"},
                ]
            ),
        ),
    }
    report: dict[str, Any] = {
        "environment": {
            "transformers": transformers.__version__,
            "torch": torch.__version__,
            "cuda_visible_devices": os.environ["CUDA_VISIBLE_DEVICES"],
            "logical_device": "cuda:0",
            "gpu_name": torch.cuda.get_device_name(0),
            "dtype": "bfloat16",
            "model_path": str(model_path),
            "config_sha256": _sha256(model_path / "config.json"),
            "video_reader": os.environ["FORCE_QWENVL_VIDEO_READER"],
        },
        "fixtures": {name: _sha256(path) for name, path in fixtures.items()},
        "structural_handles": sorted(facade.model_adapter.structural_handles(model)),
        "cases": {},
    }
    for name, sample in cases.items():
        batch = facade.processor_adapter.prepare_inputs(processor, sample, phase="eval")
        state = facade.state_builder.build(batch, phase="eval")
        result = facade.runner.generate(
            runtime,
            state,
            processed_batch=batch,
            max_new_tokens=16,
            do_sample=False,
        )
        assert result.text is not None and result.text.text[0].strip()
        assert result.speech is None
        assert facade.native_validator is not None
        assert facade.native_validator.validate_state(runtime, state, batch) == []
        report["cases"][name] = {
            "processor_contract": batch.metadata["processor_output_contract"],
            "media_loader_return_arity": batch.metadata["media_loader_return_arity"],
            "spans": [
                {"modality": span.modality, "start": span.start, "end": span.end}
                for span in state.modality_token_spans
            ],
            "generated_token_shape": _shape(result.text.token_ids),
            "text": result.text.text,
        }

    trace_batch = facade.processor_adapter.prepare_inputs(
        processor, cases["text"], phase="eval"
    )
    trace_state = facade.state_builder.build(trace_batch, phase="eval")
    trace_result = facade.runner.generate(
        runtime,
        trace_state,
        processed_batch=trace_batch,
        trace_request=TraceRequest(
            text_logits=True,
            text_hidden_states=True,
            text_kv_cache=True,
        ),
        max_new_tokens=2,
        do_sample=False,
    )
    assert trace_result.traces is not None
    assert trace_result.traces.text_logits is not None
    assert trace_result.traces.text_hidden_states is not None
    assert trace_result.traces.text_kv_cache is not None
    report["trace"] = {
        "logits_shape": _shape(trace_result.traces.text_logits),
        "hidden_state_layers": len(trace_result.traces.text_hidden_states),
        "kv_cache_type": type(trace_result.traces.text_kv_cache).__qualname__,
    }

    report_path = Path(
        os.environ.get(
            "QWEN25_VL_SMOKE_REPORT", tmp_path / "qwen25_vl_smoke.json"
        )
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    assert report_path.is_file()
