"""Opt-in real Qwen3-VL smoke using the diagnostics execution entry point.

Set ``MLLM_COMP_RUN_QWEN3_VL_GPU_SMOKE=1``, ``QWEN3_VL_MODEL_PATH``, and expose
one assigned GPU through ``CUDA_VISIBLE_DEVICES`` before launching Python.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from scripts.diagnostics.validate_qwen3_vl_gpu_smoke import main


pytestmark = pytest.mark.gpu


@pytest.mark.skipif(
    os.environ.get("MLLM_COMP_RUN_QWEN3_VL_GPU_SMOKE") != "1",
    reason="real Qwen3-VL GPU smoke is opt-in",
)
def test_qwen3_vl_all_native_modalities_and_text_traces(tmp_path: Path) -> None:
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    pytest.importorskip("qwen_vl_utils")
    pytest.importorskip("PIL.Image")
    model_path = os.environ.get("QWEN3_VL_MODEL_PATH")
    if not model_path:
        pytest.fail("QWEN3_VL_MODEL_PATH must point to an existing local checkpoint.")
    report_path = Path(
        os.environ.get("QWEN3_VL_SMOKE_REPORT", tmp_path / "qwen3_vl_smoke.json")
    )
    assert main([
        "--model_path", model_path,
        "--report_path", str(report_path),
        "--video_reader", os.environ.get("FORCE_QWENVL_VIDEO_READER", "torchcodec"),
    ]) == 0

    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["status"] == "passed"
    assert set(report["cases"]) == {"text", "image", "video"}
    for case in report["cases"].values():
        assert case["validation_errors"] == []
        assert case["text"][0].strip()
    assert report["trace"]["logits_shape"] is not None
    assert report["trace"]["hidden_state_layers"] > 0
    assert report["trace"]["kv_cache_type"] != "NoneType"
