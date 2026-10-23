"""Contract checks for the public runnable training examples."""

from __future__ import annotations

from pathlib import Path

import pytest

from mllm_comp.config.loader import load_config
from mllm_comp.config.validators import validate_runnable_config
from mllm_comp.integrations.ms_swift import MSSwiftTrainAdapter


def _value_after(argv: list[str], name: str) -> str:
    return argv[argv.index(f"--{name}") + 1]


@pytest.mark.parametrize(
    ("config_path", "backbone_name", "model_env", "model_type", "environment"),
    [
        (
            "examples/train/qwen25_omni_sft.yaml",
            "qwen25_omni",
            "QWEN25_OMNI_MODEL_PATH",
            "qwen2_5_omni",
            {"ENABLE_AUDIO_OUTPUT": "0"},
        ),
        (
            "examples/train/qwen3_vl_sft.yaml",
            "qwen3_vl",
            "QWEN3_VL_MODEL_PATH",
            "qwen3_vl",
            {},
        ),
    ],
)
def test_runnable_train_example_builds_native_swift_plan(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    config_path: str,
    backbone_name: str,
    model_env: str,
    model_type: str,
    environment: dict[str, str],
) -> None:
    model_path = tmp_path / backbone_name
    dataset_path = Path("examples/train/data/text_sft_smoke.jsonl").resolve()
    output_path = tmp_path / f"{backbone_name}-train"
    monkeypatch.setenv(model_env, str(model_path))
    monkeypatch.setenv("MLLM_COMP_TRAIN_DATASET", str(dataset_path))
    monkeypatch.setenv("MLLM_COMP_TRAIN_RUN_DIR", str(output_path))

    config = load_config(config_path)
    validate_runnable_config(config)
    plan = MSSwiftTrainAdapter().build_launch_plan(config)

    assert config.backbone.name == backbone_name
    assert plan.argv[:2] == ["swift", "sft"]
    assert _value_after(plan.argv, "model") == str(model_path)
    assert _value_after(plan.argv, "model_type") == model_type
    assert _value_after(plan.argv, "template") == model_type
    assert _value_after(plan.argv, "dataset") == str(dataset_path)
    assert _value_after(plan.argv, "max_steps") == "1"
    assert _value_after(plan.argv, "output_dir") == str(output_path.resolve())
    assert plan.environment == environment
