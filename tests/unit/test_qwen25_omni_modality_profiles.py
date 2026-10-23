"""Reference-profile contracts for the eight Qwen2.5-Omni train entries."""

from __future__ import annotations

from pathlib import Path

import pytest

from mllm_comp.config.loader import load_config
from mllm_comp.config.validators import validate_runnable_config
from mllm_comp.integrations.ms_swift import MSSwiftTrainAdapter


_MEDIA_ENVIRONMENT_NAMES = {
    "MAX_PIXELS",
    "USE_AUDIO_IN_VIDEO",
    "SAMPLING_RATE",
    "VIDEO_MIN_PIXELS",
    "VIDEO_MAX_PIXELS",
    "FPS",
    "FPS_MIN_FRAMES",
    "FPS_MAX_FRAMES",
    "FRAME_FACTOR",
}


def _value_after(argv: list[str], name: str) -> str:
    return argv[argv.index(f"--{name}") + 1]


@pytest.mark.parametrize(
    (
        "profile",
        "tuner_type",
        "max_length",
        "learning_rate",
        "per_device_batch",
        "per_device_eval_batch",
        "gradient_accumulation",
        "media_environment",
        "max_pixels",
    ),
    [
        (
            "img_sft",
            "full",
            4096,
            1e-5,
            16,
            1,
            4,
            {"MAX_PIXELS": "451584", "USE_AUDIO_IN_VIDEO": "0"},
            451584,
        ),
        (
            "img_lora",
            "lora",
            4096,
            1e-4,
            16,
            16,
            4,
            {"MAX_PIXELS": "1003520", "USE_AUDIO_IN_VIDEO": "0"},
            1003520,
        ),
        (
            "aud_sft",
            "full",
            4096,
            1e-5,
            16,
            1,
            4,
            {"USE_AUDIO_IN_VIDEO": "0", "SAMPLING_RATE": "16000"},
            None,
        ),
        (
            "aud_lora",
            "lora",
            4096,
            1e-4,
            16,
            16,
            4,
            {"USE_AUDIO_IN_VIDEO": "0", "SAMPLING_RATE": "16000"},
            None,
        ),
        (
            "vid_sft",
            "full",
            4096,
            1e-5,
            1,
            1,
            64,
            {
                "USE_AUDIO_IN_VIDEO": "0",
                "VIDEO_MAX_PIXELS": "100352",
                "FPS": "2.0",
                "FPS_MIN_FRAMES": "4",
                "FPS_MAX_FRAMES": "12",
                "FRAME_FACTOR": "2",
            },
            None,
        ),
        (
            "vid_lora",
            "lora",
            4096,
            1e-4,
            16,
            16,
            4,
            {
                "USE_AUDIO_IN_VIDEO": "0",
                "VIDEO_MAX_PIXELS": "100352",
                "FPS": "2.0",
                "FPS_MIN_FRAMES": "4",
                "FPS_MAX_FRAMES": "12",
                "FRAME_FACTOR": "2",
            },
            None,
        ),
        (
            "av_sft",
            "full",
            2048,
            1e-5,
            1,
            1,
            64,
            {
                "USE_AUDIO_IN_VIDEO": "1",
                "SAMPLING_RATE": "16000",
                "VIDEO_MIN_PIXELS": "50176",
                "VIDEO_MAX_PIXELS": "50176",
                "FPS": "2.0",
                "FPS_MIN_FRAMES": "4",
                "FPS_MAX_FRAMES": "12",
                "FRAME_FACTOR": "2",
            },
            None,
        ),
        (
            "av_lora",
            "lora",
            2048,
            1e-4,
            16,
            16,
            4,
            {
                "USE_AUDIO_IN_VIDEO": "1",
                "SAMPLING_RATE": "16000",
                "VIDEO_MIN_PIXELS": "50176",
                "VIDEO_MAX_PIXELS": "50176",
                "FPS": "2.0",
                "FPS_MIN_FRAMES": "4",
                "FPS_MAX_FRAMES": "12",
                "FRAME_FACTOR": "2",
            },
            None,
        ),
    ],
)
def test_qwen25_omni_profile_builds_expected_native_plan(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    profile: str,
    tuner_type: str,
    max_length: int,
    learning_rate: float,
    per_device_batch: int,
    per_device_eval_batch: int,
    gradient_accumulation: int,
    media_environment: dict[str, str],
    max_pixels: int | None,
) -> None:
    model_path = tmp_path / "qwen25-omni"
    dataset_path = tmp_path / "train.jsonl"
    output_path = tmp_path / profile
    monkeypatch.setenv("QWEN25_OMNI_MODEL_PATH", str(model_path))
    monkeypatch.setenv("MLLM_COMP_TRAIN_DATASET", str(dataset_path))
    monkeypatch.setenv("MLLM_COMP_TRAIN_RUN_DIR", str(output_path))

    config = load_config(f"configs/train/qwen2_5_omni/train_{profile}.yaml")
    validate_runnable_config(config)
    plan = MSSwiftTrainAdapter().build_launch_plan(config)

    assert plan.argv[:2] == ["swift", "sft"]
    assert _value_after(plan.argv, "model_type") == "qwen2_5_omni"
    assert _value_after(plan.argv, "template") == "qwen2_5_omni"
    assert _value_after(plan.argv, "tuner_type") == tuner_type
    assert int(_value_after(plan.argv, "max_length")) == max_length
    assert float(_value_after(plan.argv, "learning_rate")) == learning_rate
    assert int(_value_after(plan.argv, "per_device_train_batch_size")) == per_device_batch
    assert int(_value_after(plan.argv, "per_device_eval_batch_size")) == per_device_eval_batch
    assert int(_value_after(plan.argv, "gradient_accumulation_steps")) == gradient_accumulation
    assert plan.environment == {"ENABLE_AUDIO_OUTPUT": "0", **media_environment}
    assert set(plan.unset_environment) == _MEDIA_ENVIRONMENT_NAMES
    assert "--max_pixels" not in plan.argv
    if max_pixels is not None:
        assert int(plan.environment["MAX_PIXELS"]) == max_pixels
