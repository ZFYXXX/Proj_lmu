"""Regression checks for lmms-eval task-owned dataset metadata."""

from __future__ import annotations

from pathlib import Path

import pytest

from mllm_comp.config.loader import load_config
from mllm_comp.config.schema import experiment_config_from_dict
from mllm_comp.config.validators import validate_runnable_config
from mllm_comp.core.manifest import BackendRecord
from mllm_comp.integrations.lmms_eval import LMMSEvalAdapter
from mllm_comp.run_manifest import build_run_manifest
from mllm_comp.experiments import run_experiment


@pytest.mark.parametrize(
    ("config_path", "model_environment", "backbone_name", "expected_task"),
    [
        (
            "configs/eval/lmms_eval/qwen25_omni_img.yaml",
            "QWEN25_OMNI_MODEL_PATH",
            "qwen2_5_omni",
            "mme",
        ),
        (
            "configs/eval/lmms_eval/qwen25_omni_aud.yaml",
            "QWEN25_OMNI_MODEL_PATH",
            "qwen2_5_omni",
            "mmau_test_mini",
        ),
        (
            "configs/eval/lmms_eval/qwen3_vl.yaml",
            "QWEN3_VL_MODEL_PATH",
            "mllm_comp_qwen3_vl",
            "mmmu_val",
        ),
        (
            "configs/eval/lmms_eval/qwen25_vl_14bench.yaml",
            "QWEN25_VL_MODEL_PATH",
            "mllm_comp_qwen25_vl",
            "gqa,vqav2_val,ok_vqa_val2014,textvqa_val,docvqa_val,ocrbench,"
            "scienceqa_img,ai2d,mmstar,refcoco_bbox_rec_testA,"
            "refcoco_bbox_rec_testB,refcoco+_bbox_rec_testA,"
            "refcoco+_bbox_rec_testB,refcocog_bbox_rec_test",
        ),
        (
            "configs/eval/lmms_eval/qwen3_vl_14bench.yaml",
            "QWEN3_VL_MODEL_PATH",
            "mllm_comp_qwen3_vl",
            "gqa,vqav2_val,ok_vqa_val2014,textvqa_val,docvqa_val,ocrbench,"
            "scienceqa_img,ai2d,mmstar,refcoco_bbox_rec_testA,"
            "refcoco_bbox_rec_testB,refcoco+_bbox_rec_testA,"
            "refcoco+_bbox_rec_testB,refcocog_bbox_rec_test",
        ),
        (
            "configs/eval/lmms_eval/llava_onevision_14bench.yaml",
            "LLAVA_ONEVISION_MODEL_PATH",
            "mllm_comp_llava_onevision",
            "gqa,vqav2_val,ok_vqa_val2014,textvqa_val,docvqa_val,ocrbench,"
            "scienceqa_img,ai2d,mmstar,refcoco_bbox_rec_testA,"
            "refcoco_bbox_rec_testB,refcoco+_bbox_rec_testA,"
            "refcoco+_bbox_rec_testB,refcocog_bbox_rec_test",
        ),
    ],
)
def test_project_eval_config_builds_native_plan(
    monkeypatch: pytest.MonkeyPatch,
    config_path: str,
    model_environment: str,
    backbone_name: str,
    expected_task: str,
) -> None:
    monkeypatch.setenv(model_environment, "/models/fixture")
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)

    config = load_config(config_path)
    validate_runnable_config(config)
    result = run_experiment(config, task="eval", plan_only=True)
    assert result.status == "planned", result.error
    plan = result.output["plan"]

    assert plan.model_name == backbone_name
    assert plan.argv[plan.argv.index("--tasks") + 1] == expected_task


def test_qwen25_omni_audio_eval_disables_post_decode_stop_truncation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("QWEN25_OMNI_MODEL_PATH", "/models/fixture")

    config = load_config("configs/eval/lmms_eval/qwen25_omni_aud.yaml")

    assert config.eval.generation["until"] == []
    assert "max_new_tokens" not in config.eval.generation
    assert "do_sample" not in config.eval.generation


@pytest.mark.parametrize(
    ("config_path", "model_environment"),
    [
        ("examples/eval/qwen25_omni_3b_mme.yaml", "QWEN25_OMNI_MODEL_PATH"),
        ("examples/eval/qwen3_vl_mme.yaml", "QWEN3_VL_MODEL_PATH"),
    ],
)
def test_eval_example_delegates_dataset_ownership_to_lmms_task(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    config_path: str,
    model_environment: str,
) -> None:
    monkeypatch.setenv(model_environment, "/models/fixture")
    monkeypatch.setenv("MLLM_COMP_RUN_DIR", str(tmp_path / "eval"))

    config = load_config(config_path)
    validate_runnable_config(config)
    plan = LMMSEvalAdapter().build_launch_plan(config)

    assert config.data.dataset_id is None
    assert config.data.sample_split is None
    assert plan.argv[plan.argv.index("--tasks") + 1] == "mme"
    assert all("dataset_id" not in value for value in plan.argv)
    assert all("dataset_path" not in value for value in plan.argv)

    manifest = build_run_manifest(
        config,
        eval_backend=BackendRecord(
            name="lmms_eval",
            version="0.7.2",
            args={"argv": plan.argv},
        ),
    )
    assert manifest.dataset_id is None
    assert manifest.dataset_commit is None
    assert manifest.eval_tasks == ["mme"]
    assert manifest.eval_backend is not None
    assert manifest.eval_backend.args["argv"] == plan.argv


def _explicit_dataset_config(*, command: str) -> object:
    return experiment_config_from_dict(
        {
            "experiment": {"name": f"{command}-dataset"},
            "backbone": {"name": "qwen25_omni", "model_id": "fixture"},
            "analysis": {"tasks": ["fixture"]} if command == "analysis" else {},
            "train": {"backend": "ms_swift"} if command == "train" else {},
            "eval": {},
            "data": {
                "dataset_id": "org/explicit-dataset",
                "dataset_commit": "revision-1",
                "sample_split": "train",
            },
            "output": {
                "run_dir": f"runs/{command}",
                "manifest_path": f"runs/{command}/manifest.json",
            },
        },
        path=f"{command}.yaml",
    )


@pytest.mark.parametrize("command", ["train", "analysis"])
def test_explicit_train_and_analysis_dataset_identity_is_preserved(
    command: str,
) -> None:
    config = _explicit_dataset_config(command=command)
    if command == "train":
        config.train.args["datasets"] = ["org/explicit-dataset"]
    train_backend = (
        BackendRecord(name="ms_swift", version="4.4.0") if command == "train" else None
    )

    manifest = build_run_manifest(config, train_backend=train_backend)

    assert config.data.dataset_id == "org/explicit-dataset"
    assert manifest.dataset_id == "org/explicit-dataset"
    assert manifest.dataset_commit == "revision-1"


@pytest.mark.parametrize("dataset_id,dataset_commit,expected", [
    ("${UNUSED_DATASET}", "revision-1", (None, None)),
    ("org/explicit-dataset", "${UNUSED_REVISION}", ("org/explicit-dataset", None)),
])
def test_unused_training_data_placeholders_are_not_manifest_identity(
    dataset_id, dataset_commit, expected,
):
    config = _explicit_dataset_config(command="train")
    config.train.args["datasets"] = ["org/explicit-dataset"]
    config.data.dataset_id = dataset_id
    config.data.dataset_commit = dataset_commit
    validate_runnable_config(config, task="train")

    manifest = build_run_manifest(
        config, train_backend=BackendRecord(
            name="ms_swift", version="4.4.0",
            args={"datasets": ["org/explicit-dataset"]},
        ),
    )

    assert (manifest.dataset_id, manifest.dataset_commit) == expected
    assert manifest.effective_config["data"]["dataset_id"] == dataset_id
    assert manifest.effective_config["data"]["dataset_commit"] == dataset_commit
    assert manifest.train_backend.args["datasets"] == ["org/explicit-dataset"]


def test_lmms_eval_manifest_does_not_trust_platform_dataset_hint() -> None:
    config = experiment_config_from_dict(
        {
            "experiment": {"name": "eval-dataset-hint"},
            "backbone": {"name": "qwen25_omni", "model_id": "fixture"},
            "eval": {"tasks": ["mme"]},
            "data": {
                "dataset_id": "lmms-lab/MME",
                "dataset_commit": "unverified",
            },
            "output": {
                "run_dir": "runs/eval",
                "manifest_path": "runs/eval/manifest.json",
            },
        },
        path="eval.yaml",
    )

    manifest = build_run_manifest(
        config,
        eval_backend=BackendRecord(name="lmms_eval", version="0.7.2"),
    )

    assert manifest.dataset_id is None
    assert manifest.dataset_commit is None
