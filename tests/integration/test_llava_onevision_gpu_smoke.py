"""Opt-in real LLaVA-OneVision text/image baseline smoke on GPU 1."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

import pytest

from mllm_comp.backbones.base import BackboneRuntime
from mllm_comp.backbones.llava_onevision import (
    LlavaOneVisionSample,
    create_llava_onevision_backbone,
)


pytestmark = pytest.mark.gpu


def _enabled() -> bool:
    return os.environ.get("MLLM_COMP_RUN_LLAVA_ONEVISION_GPU_SMOKE") == "1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _shape(value: Any) -> list[int] | None:
    shape = getattr(value, "shape", None)
    return None if shape is None else [int(dim) for dim in shape]


@pytest.mark.skipif(not _enabled(), reason="real LLaVA-OneVision GPU smoke is opt-in")
def test_llava_onevision_text_and_image_baseline(tmp_path: Path) -> None:
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == "1", (
        "Launch with physical GPU 1 only: CUDA_VISIBLE_DEVICES=1."
    )
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    image_module = pytest.importorskip("PIL.Image")

    assert torch.cuda.is_available()
    assert torch.cuda.device_count() == 1
    model_path_value = os.environ.get("LLAVA_ONEVISION_MODEL_PATH")
    if not model_path_value:
        pytest.fail("LLAVA_ONEVISION_MODEL_PATH must point to a local checkpoint.")
    model_path = Path(model_path_value)
    assert model_path.is_dir()

    image_path = tmp_path / "llava_onevision_smoke.png"
    image_module.new("RGB", (384, 256), color=(176, 48, 64)).save(image_path)

    facade = create_llava_onevision_backbone()
    processor = facade.processor_adapter.load_processor(
        str(model_path), local_files_only=True
    )
    model = facade.model_adapter.load_model(
        str(model_path),
        train=False,
        local_files_only=True,
        # The pinned source builder moves the vision tower to float16 when a
        # concrete device_map is used.  Keep pixels/model in the same dtype.
        torch_dtype=torch.float16,
        attn_implementation="sdpa",
        device_map="cuda:0",
    )
    runtime = BackboneRuntime(
        model=model,
        processor=processor,
        model_path_or_id=str(model_path),
        device=torch.device("cuda:0"),
        dtype=torch.float16,
        metadata={"enable_talker": False, "speech_sample_rate": None},
    )
    cases = {
        "text": LlavaOneVisionSample(
            sample_id="text",
            prompt="Reply with one short greeting.",
        ),
        "image": LlavaOneVisionSample(
            sample_id="image",
            prompt="What is the dominant color?",
            images=(str(image_path),),
        ),
    }
    report: dict[str, Any] = {
        "environment": {
            "transformers": transformers.__version__,
            "torch": torch.__version__,
            "cuda_visible_devices": os.environ["CUDA_VISIBLE_DEVICES"],
            "logical_device": "cuda:0",
            "gpu_name": torch.cuda.get_device_name(0),
            "dtype": "float16",
            "model_path": str(model_path),
            "config_sha256": _sha256(model_path / "config.json"),
        },
        "fixture_sha256": _sha256(image_path),
        "structural_handles": sorted(facade.model_adapter.structural_handles(model)),
        "cases": {},
    }
    for name, sample in cases.items():
        batch = facade.processor_adapter.prepare_inputs(processor, sample, phase="eval")
        state = facade.state_builder.build(batch, phase="eval")
        assert facade.native_validator is not None
        assert facade.native_validator.validate_state(runtime, state, batch) == []
        result = facade.runner.generate(
            runtime,
            state,
            processed_batch=batch,
            max_new_tokens=16,
            do_sample=False,
        )
        assert result.text is not None and result.text.text[0].strip()
        assert result.speech is None
        report["cases"][name] = {
            "modalities": batch.metadata["modalities"],
            "image_sizes": batch.metadata["image_sizes"],
            "anyres": batch.metadata["anyres"],
            "spans": [
                {"modality": span.modality, "start": span.start, "end": span.end}
                for span in state.modality_token_spans
            ],
            "generated_token_shape": _shape(result.text.token_ids),
            "text": result.text.text,
        }

    report_path = Path(
        os.environ.get(
            "LLAVA_ONEVISION_SMOKE_REPORT",
            tmp_path / "llava_onevision_smoke.json",
        )
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    assert report_path.is_file()
