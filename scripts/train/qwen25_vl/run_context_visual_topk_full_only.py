"""Run the frozen context selector experiment on AI2D full only.

The smoke is complete. This runner skips the non-independent 500-example
development pass, preserves its partial attempts, and evaluates all rows.
Model/plugin source is imported from the hash-pinned worktree recorded in the
original protocol; this orchestration file is separately SHA-256 recorded.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import sys
import time
import traceback


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--gpus", nargs="+", type=int, default=[0, 1])
    args = parser.parse_args()
    root = args.root.resolve()
    protocol = json.loads((root / "protocol.json").read_text())
    source_root = Path(protocol["source"]["source_root"]).resolve()
    sys.path.insert(0, str(source_root))
    from scripts.train.qwen25_vl import run_context_visual_topk as runner
    from mllm_comp.analysis.common.learned_topk_experiment import available_gpus
    from mllm_comp.core.serialization import atomic_write_json, atomic_write_text
    import yaml

    if runner.source_identity(source_root) != protocol["source"]:
        raise ValueError("Frozen method source differs from the original protocol")
    if runner.dependency_identity() != protocol["dependencies"]:
        raise ValueError("Evaluator dependency snapshot changed")
    if len(set(args.gpus)) != len(args.gpus) or available_gpus(args.gpus) != args.gpus:
        raise RuntimeError("Requested GPU is busy; no full evaluation launched")
    smoke = json.loads((root / "eval_smoke_complete.json").read_text())
    if smoke.get("status") != "complete" or set(smoke["methods"]) != set(runner.METHODS):
        raise ValueError("All five methods must pass the existing smoke first")

    orchestration = Path(__file__).resolve()
    orchestration_sha = hashlib.sha256(orchestration.read_bytes()).hexdigest()
    atomic_write_json(root / "eval_dev_skipped.json", {
        "status": "skipped_by_user_preference_after_partial_start",
        "reason": "500-example development evaluation overlaps full evaluation and is not an independent confirmation; smoke passed.",
        "completed_methods": [m for m in runner.METHODS if (root / "eval/dev" / m / "complete.json").exists()],
        "partial_attempts_preserved": True,
        "orchestration_sha256": orchestration_sha,
        "time": runner.native.now(),
    })
    manifest = json.loads((root / "manifest.json").read_text())
    atomic_write_json(root / "manifest_before_full_only.json", manifest)
    manifest.pop("error", None)
    manifest.update(status="running", stage="eval:full", skipped_phase="dev", updated_utc=runner.native.now())
    atomic_write_json(root / "manifest.json", manifest)

    def worker(gpu, methods):
        for method in methods:
            pointer = root / "eval/full_ai2d" / method / "complete.json"
            if pointer.exists():
                saved = json.loads(pointer.read_text())
                if saved["source"] != protocol["source"] or runner.file_sha(saved["config"]) != saved["config_sha256"]:
                    raise ValueError("Existing full result has changed provenance")
                runner.native.check_eval(Path(saved["run"]), method, "full", tasks=("ai2d",),
                    counts={"ai2d": 3088}, development_count=500, prefix="context_topk")
                continue
            run = pointer.parent / f"attempt_{time.time_ns()}"
            cfg = runner.eval_config(root, protocol, method, "full", run)
            cfg["eval"]["tasks"] = ["context_topk_ai2d"]
            config_path = root / "configs" / f"full_ai2d_{method}_{time.time_ns()}.yaml"
            config_path.parent.mkdir(parents=True, exist_ok=True)
            atomic_write_text(config_path, yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False))
            runner.native.execute(
                [sys.executable, "-m", "mllm_comp.cli.main", "eval", "--config", str(config_path)],
                pointer.parent / f"launch_{time.time_ns()}.log", [gpu],
                {"MLLM_TOPK_COHORT": str(root / "cohort.json"),
                 "MLLM_TOPK_PERFORMANCE": str(run / "request_performance.jsonl"),
                 "HF_HOME": str(root / "cache/huggingface"),
                 "HF_DATASETS_CACHE": str(root / "cache/datasets"),
                 "XDG_CACHE_HOME": str(root / "cache/xdg")})
            checks = runner.native.check_eval(run, method, "full", tasks=("ai2d",),
                counts={"ai2d": 3088}, development_count=500, prefix="context_topk")
            if method.endswith("_context"):
                for line in (run / "compression_runtime.requests.jsonl").read_text().splitlines():
                    row = json.loads(line)
                    delta = row["runtime"]["activations"][0]["selection"]
                    expected_mode = method.removesuffix("_context")
                    if delta["context_mode"] != expected_mode:
                        raise ValueError("Wrong context arm evaluated")
                    if method == "question_context" and (not delta["reads_question"] or
                            not delta["question_tokens"] or not delta["question_ids_sha256"]):
                        raise ValueError("Missing prompt-only question audit")
            atomic_write_json(pointer, {"run": str(run), "source": protocol["source"],
                "config": str(config_path), "config_sha256": runner.file_sha(config_path),
                "requests": checks["requests"], "completed_utc": runner.native.now(),
                "orchestration_sha256": orchestration_sha})

    try:
        with ThreadPoolExecutor(max_workers=len(args.gpus)) as pool:
            futures = [pool.submit(worker, gpu, list(runner.METHODS[i::len(args.gpus)]))
                       for i, gpu in enumerate(args.gpus)]
            for future in futures:
                future.result()
        atomic_write_json(root / "eval_full_ai2d_complete.json", {
            "status": "complete", "datasets": ["ai2d"], "methods": list(runner.METHODS),
            "orchestration_sha256": orchestration_sha, "time": runner.native.now()})
        summary = {}
        metric = "exact_match"
        for method in runner.METHODS:
            run = Path(json.loads((root / "eval/full_ai2d" / method / "complete.json").read_text())["run"])
            runner.native.check_eval(run, method, "full", tasks=("ai2d",),
                counts={"ai2d": 3088}, development_count=500, prefix="context_topk")
            samples, = run.glob("**/*_samples_context_topk_ai2d.jsonl")
            rows = [json.loads(line) for line in samples.read_text().splitlines()]
            summary[method] = {"exact_match": sum(float(row[metric]) for row in rows) / len(rows),
                "requests": len(rows), "run": str(run)}
        atomic_write_json(root / "summary_full_ai2d.json", {
            "dataset": "ai2d", "questions": 3088, "metric": metric,
            "scores": summary, "methods": list(runner.METHODS), "time": runner.native.now()})
        manifest.update(status="complete", stage="complete", finished_utc=runner.native.now(),
                        datasets_evaluated=["ai2d"], development_evaluation="skipped_by_user_preference")
        atomic_write_json(root / "manifest.json", manifest)
    except BaseException as exc:
        manifest.update(status="failed", stage="eval:full", error=str(exc), updated_utc=runner.native.now())
        atomic_write_json(root / "manifest.json", manifest)
        atomic_write_json(root / f"full_eval_failure_{time.time_ns()}.json",
            {"error": str(exc), "traceback": traceback.format_exc()})
        raise


if __name__ == "__main__":
    main()
