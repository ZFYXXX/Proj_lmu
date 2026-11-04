"""Create the path-local profile/config for the frozen balanced LMU training arm."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha256(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--checkpoint-root", type=Path, required=True)
    parser.add_argument("--identity", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--profile-output", type=Path, required=True)
    parser.add_argument("--training-output", type=Path, required=True)
    parser.add_argument("--run-id", default="qwen25_lmunext_balanced2560_r1_v1")
    args = parser.parse_args()

    repo, bundle = args.repo.resolve(), args.bundle.resolve()
    manifest = bundle / "balanced_manifest.json"
    parquet = bundle / "balanced_qa.parquet"
    for path in (repo / "scripts/workflow.py", manifest, parquet,
                 args.model / "config.json", args.checkpoint_root / "checkpoints/old_mlp/projector.safetensors",
                 args.checkpoint_root / "checkpoints/image_context_selector/selector.safetensors",
                 args.identity):
        if not path.is_file():
            raise FileNotFoundError(path)
    payload = json.loads(manifest.read_text())
    if len(payload.get("entries", [])) != 2560 or payload.get("selection_reads_answers") is not False:
        raise ValueError("Expected the frozen answer-independent 2,560-image cohort")
    for entry in payload["entries"]:
        path = Path(entry["path"])
        if not path.is_absolute() or not path.is_file():
            raise ValueError(f"Relocate the bundle before preparing LMU training: {path}")
    counts = payload.get("source_counts", {})
    expected_counts = {
        "ocrvqa_next_train": 320, "chartqa_next_train": 320, "docvqa_next_train": 256,
        "gqa_next_train": 384, "llava_next_coco_train": 256, "refcoco_train": 512,
        "scienceqa_train": 256, "ai2d_next_train": 256,
    }
    if counts != expected_counts:
        raise ValueError(f"Unexpected frozen source quotas: {counts}")

    profile = {
        "schema_version": 1,
        "python": "/home/hk-project-p0022189/tum_yvc3016/miniconda3/envs/zfy-mllm-comp/bin/python",
        "data_root": "/hkfs/work/workspace/scratch/tum_yvc3016-comp/mllm_dataset/lmms_lab_dataset",
        "model": str(args.model.resolve()),
        "checkpoint_root": str(args.checkpoint_root.resolve()),
        "identity": str(args.identity.resolve()),
        "training_identity": str(args.identity.resolve()),
        "training_manifest": str(manifest),
        "training_output_root": str(args.output_root.resolve()),
        "output_root": str(args.output_root.resolve()),
        "cache_dir": str(args.cache_dir.resolve()),
        "gpus": ["0"],
        "result_bundle": str((repo / "artifacts/eval/qwen25_p6000_reuse_20261103.json.gz").resolve()),
        "seed_from": [
            "/hkfs/work/workspace/scratch/tum_yvc3016-comp/workspace/outputs/qwen25_remaining_dynamic_testdev_20261101",
            "/hkfs/work/workspace/scratch/tum_yvc3016-comp/workspace/outputs/qwen25_remaining_benchmarks_20261031",
        ],
        "environment": {"MLLM_MMSTAR_PARQUET": "/hkfs/work/workspace/scratch/tum_yvc3016-comp/mllm_dataset/lmms_lab_dataset/MMStar/mmstar.parquet"},
    }
    digest = sha256(manifest)
    training = {
        "schema_version": 1,
        "run_id": args.run_id,
        "training_manifest_sha256": digest,
        "manifest_mode": "balanced_arm_only",
        "seed": 42,
        "steps": 320,
        "global_batch": 8,
        "learning_rate": 0.0001,
        "weight_decay": 0.01,
        "warmup_steps": 16,
        "clip_grad_norm": 1.0,
        "teacher_kl_weight": 0.5,
        "save_every": 16,
        "smoke_steps": 2,
        "gpus": ["0"],
        "defer_evaluation": True,
        "arms": {
            "R1_balanced_data_scratch": {
                "method": "importance_fps",
                "pooling": "mean",
                "train_pool": False,
                "training_manifest": str(manifest),
                "training_manifest_sha256": digest,
                "source_counts": expected_counts,
            }
        },
        "evaluation_tasks": json.loads((repo / "configs/workflow/experiments/r1_balanced_full_eval.json").read_text())["method_tasks"]["importance_fps"],
        "checkpoint_choice": "fixed_step320",
        "initialization": "same_seed_zero_residual_mlp",
        "initial_parameter_sha256": "90f25294fa8fcd306393e8dd7997c813db67eaec8fcf2b1e504dd8572099e1ed",
    }
    args.profile_output.parent.mkdir(parents=True, exist_ok=True)
    args.training_output.parent.mkdir(parents=True, exist_ok=True)
    args.profile_output.write_text(json.dumps(profile, indent=2) + "\n")
    args.training_output.write_text(json.dumps(training, indent=2) + "\n")
    print(json.dumps({"profile": str(args.profile_output), "training_config": str(args.training_output),
                      "manifest_sha256": digest, "examples": 2560, "training_gpus": 1,
                      "evaluation_gpus": 4}, indent=2))


if __name__ == "__main__":
    main()
