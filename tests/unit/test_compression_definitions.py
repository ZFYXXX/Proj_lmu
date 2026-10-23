from __future__ import annotations

from pathlib import Path

import pytest

from mllm_comp.config.loader import load_config
from mllm_comp.config.validators import validate_runnable_config
from mllm_comp.core.vocabulary import CompressionMode
from mllm_comp.pipeline.definition import resolve_compression_definition


_ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize(
    "path",
    sorted([
        *_ROOT.joinpath("configs").rglob("*.yaml"),
        *_ROOT.joinpath("examples", "compression").glob("*.yaml"),
    ]),
    ids=lambda path: str(path.relative_to(_ROOT)),
)
def test_repository_recipes_and_compression_examples(
    path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("QWEN25_OMNI_MODEL_PATH", "/models/qwen25-omni")
    monkeypatch.setenv("FASTSLM_HTA_CHECKPOINT_PATH", "/models/fastslm-hta.pt")
    monkeypatch.setenv("QWEN25_VL_MODEL_PATH", "/models/qwen25-vl")
    monkeypatch.setenv("QWEN3_VL_MODEL_PATH", "/models/qwen3-vl")
    monkeypatch.setenv("LLAVA_ONEVISION_MODEL_PATH", "/models/llava-onevision")
    monkeypatch.setenv("MLLM_COMP_TRAIN_DATASET", "/data/train.jsonl")
    monkeypatch.setenv("MLLM_COMP_TRAIN_RUN_DIR", "/runs/train")
    monkeypatch.setenv("MLLM_COMP_EVAL_RUN_DIR", "/runs/eval")
    monkeypatch.setenv("MLLM_COMP_RUN_DIR", "/runs/compression")
    monkeypatch.setenv("MLLM_COMP_IMAGELLM_PROBE_ROOT", "/runs/imagellm-probe")
    monkeypatch.setenv("SEGMENTWISE_KEEP_RATIO", "0.25")
    monkeypatch.setenv("SEGMENTWISE_RUN_TAG", "r25")
    monkeypatch.setenv("AUDIOMARATHON_TASK_DIR", "/data/audiomarathon_tasks")

    config = load_config(path)
    if path.is_relative_to(_ROOT / "configs"):
        validate_runnable_config(config)
    # The schema examples deliberately omit datasets and other recipe inputs.
    if not any(stage.enabled for stage in config.pipeline.stages):
        return
    interface = (
        "eval"
        if config.eval.compression_mode is CompressionMode.COMPRESSED
        else "train"
    )

    definition = resolve_compression_definition(
        config,
        compression_mode="compressed",
        interface=interface,
        runtime_batch_size=(
            config.eval.runtime_batch_size if interface == "eval" else None
        ),
    )

    assert definition.stages
