"""Serial Qwen2.5 evaluation using full local datasets and resumable methods.

Pass --limit for a small end-to-end check of the very same full-source path.
Without --limit every selected source shard is evaluated.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from mllm_comp.analysis.common.anchor7b_runtime import identify_checkpoint, validate_identity
from scripts.eval.lmms_eval.p6000_dataset_smoke import specifications

METHODS = ("dense", "old_mlp", "visual_selector", "image_context_selector", "question_context_selector",
           "protected_selector", "eadp", "prunesid", "mmtok")
TRAINED = {"old_mlp", "visual_selector", "image_context_selector", "question_context_selector", "protected_selector"}
DEFAULT_METHODS = ("dense", "eadp", "prunesid", "mmtok", "old_mlp", "visual_selector",
                   "image_context_selector", "question_context_selector", "protected_selector")
DEFAULT_TASKS = ("gqa", "vqav2_val", "scienceqa_img", "mmmu", "hallusionbench", "textvqa",
                 "refcoco_bbox_rec_testA", "refcoco_bbox_rec_testB", "refcocog_bbox_rec_test")
PRIMARY = {"chartqa": "relaxed_overall", "ocrbench": "ocrbench_accuracy", "ai2d": "leading_option_accuracy",
           "mmstar": "average", "pope": "pope_accuracy", "gqa": "exact_match", "vqav2_val": "exact_match",
           "okvqa": "exact_match", "scienceqa_img": "exact_match", "mmmu": "mmmu_acc",
           "textvqa": "exact_match", "docvqa": "anls", "hallusionbench": "aAcc"}


def dump(path, value):
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    os.replace(temp, path)


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_identity(reference, model, output):
    old = json.loads(reference.read_text())
    if Path(old["checkpoint"]).resolve() == model.resolve():
        validate_identity(old)
        identity = old
    else:
        print("Checking model file hashes against trained checkpoints; this reads the full model once.", flush=True)
        identity = identify_checkpoint(str(model))
        if identity["identity_sha256"] != old["identity_sha256"]:
            raise ValueError("Target Qwen2.5 model content differs from the training base; refusing to use the trained heads")
    path = output / "model_identity.json"
    if path.exists() and json.loads(path.read_text()) != identity:
        raise ValueError("Model identity differs from existing run")
    dump(path, identity)
    return path


def score_name(task):
    if task == "hallusionbench":
        return "generation_nonempty_smoke"
    return "refcoco_ACC@0.5" if task.startswith("refcoco") else PRIMARY[task]


def collect(attempt, tasks, limit):
    manifest = json.loads((attempt / "manifest.json").read_text())
    if manifest.get("status") != "complete" or manifest.get("eval_exit_code") != 0:
        raise RuntimeError(f"Evaluation incomplete: {attempt / 'manifest.json'}")
    scores = {}
    for task in tasks:
        record = manifest["tasks"][task]
        if record.get("status") != "generation_and_scoring_passed":
            raise ValueError(f"Task incomplete: {task}")
        expected = record["eval_rows"] if limit is None else min(record["eval_rows"], limit)
        files = list((attempt / "results").rglob(f"*_samples_{record['task']}.jsonl"))
        if len(files) != 1:
            raise ValueError(f"Expected one sample log for {task}, found {len(files)}")
        with files[0].open() as stream:
            rows = [json.loads(line) for line in stream if line.strip()]
        ids = [row["doc_id"] for row in rows]
        if len(rows) != expected or len(set(ids)) != expected:
            raise ValueError(f"{task}: expected {expected} unique predictions, found {len(rows)}")
        metric_base = score_name(task)
        metric = metric_base + (",flexible-extract" if task == "ai2d" else ",none")
        if metric not in record["metrics"]:
            raise ValueError(f"Missing primary metric {metric} for {task}")
        ordered = sorted(rows, key=lambda row: row["doc_id"])
        if task == "hallusionbench" and any(not isinstance(r.get("hallusion_record"), dict) or
                                             not str(r["hallusion_record"].get("model_prediction", "")).strip()
                                             for r in ordered):
            raise ValueError("HallusionBench has missing records or empty predictions")
        if [row["doc_id"] for row in ordered] != list(range(expected)):
            raise ValueError(f"Non-contiguous local document IDs for {task}")
        scores[task] = {"score": record["metrics"][metric], "metric": metric, "samples": expected,
                        "samples_path": str(files[0]), "source_stats": record["source_stats"],
                        "source_rows": record["source_rows"], "eval_rows": record["eval_rows"],
                        "chunk": record.get("chunk"),
                        "doc_hashes_sha256": hashlib.sha256(json.dumps([r.get('doc_hash') for r in ordered]).encode()).hexdigest()}
    return scores


def report(output, methods, tasks):
    cells = {}
    for method in methods:
        cells[method] = {}
        for task in tasks:
            marker = output / "cells" / method / task / "complete.json"
            if marker.exists():
                cells[method][task] = json.loads(marker.read_text())["score"]
    lines = ["# Qwen2.5 evaluation", "", "Normalized score = method score / same-task dense score × 100.",
             "A missing or zero baseline is shown as —. Limited checks are not benchmark results.", "",
             "| Method | " + " | ".join(tasks) + " |", "|---|" + "---:|" * len(tasks)]
    if cells.get("dense", {}).get("mmmu", {}).get("subset") == "single_image_only":
        lines.insert(4, "MMMU is the single-image subset only; it is not the official full-validation score.")
    normalized = {}
    for method in methods:
        normalized[method] = {}
        values = []
        for task in tasks:
            row, base = cells.get(method, {}).get(task), cells.get("dense", {}).get(task)
            value = 100 * row["score"] / base["score"] if row and base and row["score"] is not None and base["score"] else None
            normalized[method][task] = value
            values.append(f"{value:.2f}%" if value is not None else "—")
        lines.append("| " + method + " | " + " | ".join(values) + " |")
    (output / "report.md").write_text("\n".join(lines) + "\n")
    dump(output / "summary.json", {"raw_scores": cells, "normalized_percent": normalized})


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, help="Shared writable dataset cache; defaults under output")
    parser.add_argument("--reference-identity", type=Path)
    parser.add_argument("--checkpoint", action="append", default=[], metavar="METHOD=DIR")
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=list(DEFAULT_METHODS))
    parser.add_argument("--tasks", nargs="+", choices=[row[0] for row in specifications()], default=list(DEFAULT_TASKS))
    parser.add_argument("--limit", type=int, help="Samples per full-source task; omit for full evaluation")
    parser.add_argument("--chunk-size", type=int, default=512, help="Maximum processed requests per resumable chunk")
    parser.add_argument("--mmmu-single-image-only", dest="mmmu_single_image_only", action="store_true", default=True,
                        help="Evaluate MMMU single-image subset only (default)")
    parser.add_argument("--mmmu-full", dest="mmmu_single_image_only", action="store_false",
                        help="Request the full MMMU set; unsupported bindings will be refused")
    parser.add_argument("--first-shard-only", action="store_true", help="Use the first source shard for a limited hardware check")
    parser.add_argument("--gpu", default=os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0],
                        help="CUDA device ID (defaults to the first device assigned to the job)")
    parser.add_argument("--hallusion-image", type=Path)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")
    if args.chunk_size < 1:
        parser.error("--chunk-size must be positive")
    if args.first_shard_only and args.limit is None:
        parser.error("--first-shard-only requires --limit")
    if len(args.methods) != len(set(args.methods)) or len(args.tasks) != len(set(args.tasks)):
        parser.error("Methods and tasks must not contain duplicates")
    if "dense" not in args.methods:
        parser.error("Include dense as the reference method")
    args.methods = ["dense"] + [m for m in args.methods if m != "dense"]
    if "mmmu" in args.tasks and args.limit is None and not args.mmmu_single_image_only and set(args.methods) - {"dense", "prunesid"}:
        import pyarrow.parquet as pq
        files = sorted((args.data_root / "MMMU/data").glob("validation-*.parquet"))
        if not files:
            parser.error("MMMU validation parquet missing")
        # MMMU's native visual loader scans the constructed prompt, including
        # answer options; scanning question alone misses images in the choices.
        counts = [len(set(re.findall(r"<image \d+>", (row["question"] or "") + " " + (row["options"] or ""))))
                  for path in files for row in pq.read_table(path, columns=["question", "options"]).to_pylist()]
        unsupported = sum(count != 1 for count in counts)
        if unsupported:
            parser.error(f"MMMU has {unsupported} zero/multi-image questions; not all requested compressed bindings "
                         "have validated support. Full MMMU compression remains blocked.")
    checkpoints = {}
    for item in args.checkpoint:
        method, sep, directory = item.partition("=")
        if not sep or method not in TRAINED or method in checkpoints:
            parser.error(f"Invalid checkpoint assignment: {item}")
        checkpoints[method] = Path(directory).resolve()
    missing = (set(args.methods) & TRAINED) - checkpoints.keys()
    if missing or (set(args.methods) & TRAINED and not args.reference_identity):
        parser.error(f"Trained methods require --reference-identity and --checkpoint METHOD=DIR; missing {sorted(missing)}")
    if not args.data_root.is_dir() or not (args.model / "config.json").is_file():
        parser.error("Dataset root or local model not found")
    args.output = args.output.resolve()
    if args.output.exists() and not args.resume:
        parser.error("Output exists; select a new directory or use --resume")
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "cells").mkdir(exist_ok=True)
    hashes = {}
    for method, directory in checkpoints.items():
        if not directory.is_dir():
            parser.error(f"Checkpoint missing: {directory}")
        files = sorted(p for p in directory.iterdir() if p.is_file() and p.suffix in (".json", ".safetensors"))
        if not files:
            parser.error(f"No checkpoint config/weights in {directory}")
        hashes[method] = {p.name: sha(p) for p in files}
    identity = resolve_identity(args.reference_identity.resolve(), args.model, args.output) if args.reference_identity else None
    plan = {"data_root": str(args.data_root.resolve()), "model": str(args.model.resolve()),
            "model_identity_sha256": json.loads(identity.read_text())["identity_sha256"] if identity else None,
            "methods": args.methods, "tasks": args.tasks, "limit": args.limit, "chunk_size": args.chunk_size,
            "mmmu_single_image_only": args.mmmu_single_image_only,
            "first_shard_only": args.first_shard_only,
            "cache_dir": str(args.cache_dir.resolve()) if args.cache_dir else str(args.output / "cache"),
            "checkpoints": {k: str(v) for k, v in checkpoints.items()}, "checkpoint_hashes": hashes,
            "source_hashes": {str(p.relative_to(ROOT)): sha(p) for p in (Path(__file__).resolve(), ROOT / "scripts/eval/lmms_eval/p6000_dataset_smoke.py")},
            "hallusion_image": str(args.hallusion_image.resolve()) if args.hallusion_image else None}
    plan_path = args.output / "plan.json"
    if plan_path.exists() and json.loads(plan_path.read_text()) != plan:
        raise ValueError("Run plan changed; choose a new output rather than resuming")
    dump(plan_path, plan)
    dump(args.output / "status.json", {"status": "running", "mode": "limited" if args.limit else "full"})
    os.environ.setdefault("OPENAI_API_KEY", "unused-local-smoke-no-judge")
    from lmms_eval.api.registry import get_aggregation
    from lmms_eval.utils import load_yaml_config
    for method in args.methods:
        for task in args.tasks:
            cell = args.output / "cells" / method / task
            cell.mkdir(parents=True, exist_ok=True)
            marker = cell / "complete.json"
            if marker.exists():
                print(f"SKIP completed {method}/{task}", flush=True)
                continue
            chunk_scores = []
            start = 0
            while True:
                for relative, expected_hash in plan["source_hashes"].items():
                    if sha(ROOT / relative) != expected_hash:
                        raise RuntimeError(f"Source changed during execution: {relative}; restart with a new output")
                chunk = cell / f"chunk_{start:09d}"
                chunk.mkdir(exist_ok=True)
                chunk_marker = chunk / "complete.json"
                if chunk_marker.exists():
                    result = json.loads(chunk_marker.read_text())
                    print(f"SKIP completed {method}/{task}/{start}", flush=True)
                else:
                    attempt = chunk / f"attempt_{time.time_ns()}"
                    attempt.mkdir()
                    size = min(args.chunk_size, args.limit - start) if args.limit is not None else args.chunk_size
                    cmd = [sys.executable, "-u", str(ROOT / "scripts/eval/lmms_eval/p6000_dataset_smoke.py"),
                           "--input-mode", "full", "--run", "--output", str(attempt), "--shared-cache", plan["cache_dir"],
                           "--data-root", str(args.data_root.resolve()), "--model", str(args.model.resolve()), "--gpu", args.gpu,
                           "--method", method, "--tasks", task, "--chunk-start", str(start), "--chunk-size", str(size)]
                    if args.first_shard_only: cmd += ["--first-shard-only", "--eval-limit", str(args.limit)]
                    if task == "mmmu" and args.mmmu_single_image_only: cmd += ["--mmmu-single-image-only"]
                    if method in TRAINED: cmd += ["--identity-path", str(identity), "--method-checkpoint", str(checkpoints[method])]
                    if args.hallusion_image: cmd += ["--hallusion-image", str(args.hallusion_image.resolve())]
                    if task == "hallusionbench": cmd += ["--hallusion-scoring", "deferred"]
                    print(f"START {method}/{task}/{start}; log: {attempt / 'worker.log'}", flush=True)
                    with (attempt / "worker.log").open("x") as log:
                        completed = subprocess.run(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
                    try:
                        if completed.returncode:
                            raise RuntimeError(f"Evaluator exited {completed.returncode}; see {attempt / 'worker.log'}")
                        result = collect(attempt, [task], None)[task]
                        if result["chunk"]["start"] != start:
                            raise ValueError(f"Incorrect chunk start for {method}/{task}")
                        if method != "dense":
                            baseline_path = args.output / "cells" / "dense" / task / f"chunk_{start:09d}" / "complete.json"
                            baseline = json.loads(baseline_path.read_text())
                            for field in ("source_stats", "source_rows", "eval_rows", "samples", "chunk", "doc_hashes_sha256"):
                                if result[field] != baseline[field]:
                                    raise ValueError(f"Dataset differs from dense for {method}/{task}/{start}: {field}")
                        result["attempt"] = str(attempt)
                        dump(chunk_marker, result)
                        print(f"PASS {method}/{task}/{start}", flush=True)
                    except Exception as exc:
                        dump(args.output / "status.json", {"status": "failed", "method": method, "task": task,
                                                            "chunk_start": start, "error": str(exc), "attempt": str(attempt)})
                        raise
                chunk_scores.append(result)
                start = result["chunk"]["stop"]
                end = result["chunk"]["total"]
                if args.limit is not None:
                    end = min(end, args.limit)
                if start >= end:
                    break
            metric_name = score_name(task)
            if task == "hallusionbench":
                aggregate = None
            elif len(chunk_scores) == 1:
                aggregate = chunk_scores[0]["score"]
            else:
                template = Path(chunk_scores[0]["attempt"]) / "tasks" / f"{task}.yaml"
                task_config = load_yaml_config(str(template))
                metric_config = next(item for item in task_config["metric_list"] if item["metric"] == metric_name)
                aggregation = metric_config["aggregation"]
                if isinstance(aggregation, str):
                    aggregation = get_aggregation(aggregation)
                if not callable(aggregation):
                    raise ValueError(f"Missing native aggregation for {task}/{metric_name}")
                payloads = []
                for part in chunk_scores:
                    with Path(part["samples_path"]).open() as stream:
                        payloads.extend(row[metric_name] for row in map(json.loads, stream) if metric_name in row)
                aggregate = float(aggregation(payloads))
            result = {"score": aggregate, "metric": metric_name, "samples": sum(p["samples"] for p in chunk_scores),
                      "chunks": len(chunk_scores), "source_rows": chunk_scores[0]["source_rows"],
                      "subset": "single_image_only" if task == "mmmu" and args.mmmu_single_image_only else "full",
                      "processed_total": chunk_scores[0]["chunk"]["total"],
                      "excluded": chunk_scores[0]["chunk"].get("excluded", 0)}
            if task == "hallusionbench":
                result.update(metric="aAcc", scoring_status="pending_official_judge")
            dump(marker, {"method": method, "task": task, "score": result})
            report(args.output, args.methods, args.tasks)
            print(f"PASS {method}/{task}: {result['samples']} requests", flush=True)
    dump(args.output / "status.json", {"status": "complete", "mode": "limited" if args.limit else "full"})
    print(f"Completed; report: {args.output / 'report.md'}", flush=True)


if __name__ == "__main__":
    main()
