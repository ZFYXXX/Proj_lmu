"""Resumable Qwen2.5 evaluation with a shared queue of request chunks.

Pass --limit for a small end-to-end check of the very same full-source path.
Without --limit every selected source shard is evaluated.
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import socket
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
DEFAULT_TASKS = ("gqa", "scienceqa_img", "mmmu", "hallusionbench", "textvqa",
                 "refcoco_bbox_rec_testA", "refcoco_bbox_rec_testB", "refcocog_bbox_rec_test", "vqav2_testdev")
PRIMARY = {"chartqa": "relaxed_overall", "ocrbench": "ocrbench_accuracy", "ai2d": "leading_option_accuracy",
           "mmstar": "average", "pope": "pope_accuracy", "gqa": "exact_match", "vqav2_val": "exact_match",
           "okvqa": "exact_match", "scienceqa_img": "exact_match", "mmmu": "mmmu_acc",
           "textvqa": "exact_match", "docvqa": "anls", "hallusionbench": "aAcc"}


def dump(path, value):
    temp = path.with_name(f"{path.name}.{os.getpid()}.{time.time_ns()}.tmp")
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
    if task == "vqav2_testdev":
        return "submission"
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
        expected_status = "generation_and_submission_passed" if task == "vqav2_testdev" else "generation_and_scoring_passed"
        if record.get("status") != expected_status:
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
        if task == "vqav2_testdev":
            vqav2_submissions(ordered)
        if [row["doc_id"] for row in ordered] != list(range(expected)):
            raise ValueError(f"Non-contiguous local document IDs for {task}")
        scores[task] = {"score": None if task == "vqav2_testdev" else record["metrics"][metric],
                        "metric": metric, "samples": expected,
                        "samples_path": str(files[0]), "source_stats": record["source_stats"],
                        "source_rows": record["source_rows"], "eval_rows": record["eval_rows"],
                        "chunk": record.get("chunk"),
                        "doc_hashes_sha256": hashlib.sha256(json.dumps([r.get('doc_hash') for r in ordered]).encode()).hexdigest()}
    return scores


def vqav2_submissions(rows):
    """Validate the native submission payload; never fabricate a test-dev score."""
    submissions, seen = [], set()
    for row in rows:
        item = row.get("submission")
        if not isinstance(item, dict) or set(item) != {"question_id", "answer"}:
            raise ValueError("VQAv2 test-dev sample lacks a native submission record")
        question_id = item["question_id"]
        if not isinstance(question_id, int) or isinstance(question_id, bool) or question_id in seen:
            raise ValueError(f"Duplicate or invalid VQAv2 test-dev question ID: {question_id}")
        if not isinstance(item["answer"], str) or not item["answer"].strip():
            raise ValueError(f"Empty VQAv2 test-dev prediction: {question_id}")
        doc_id = row.get("doc", {}).get("question_id")
        if doc_id is not None and doc_id != question_id:
            raise ValueError("VQAv2 test-dev submission question ID differs from its sample")
        seen.add(question_id)
        submissions.append({"question_id": question_id, "answer": item["answer"]})
    return submissions


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
    if "vqav2_testdev" in tasks:
        lines.insert(4, "VQAv2 test-dev has no public gold labels; scores are pending official evaluation. Per-method submission JSON files are retained.")
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
    temporary = output / f"report.md.{os.getpid()}.{time.time_ns()}.tmp"
    temporary.write_text("\n".join(lines) + "\n")
    os.replace(temporary, output / "report.md")
    dump(output / "summary.json", {"raw_scores": cells, "normalized_percent": normalized})


def cell_lock(cell: Path, worker_index: int):
    """Claim work non-blockingly; the OS releases the lock when a worker exits."""
    lock = (cell / ".claim.lock").open("a+")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        lock.close()
        return None
    lock.seek(0)
    lock.truncate()
    json.dump({"pid": os.getpid(), "host": socket.gethostname(), "worker_index": worker_index,
               "slurm_job_id": os.environ.get("SLURM_JOB_ID"), "claimed_at": time.time()}, lock)
    lock.flush()
    return lock


def claim_next_work(output, methods, tasks, chunk_size, limit, worker_index):
    """Share chunks, including long test-dev runs, rather than pinning datasets to GPUs."""
    for task in tasks:
        for method in methods:
            cell = output / "cells" / method / task
            cell.mkdir(parents=True, exist_ok=True)
            if (cell / "complete.json").is_file():
                continue
            if (cell / "failed.json").is_file():
                raise RuntimeError(f"Failed cell requires inspection: {method}/{task}")
            dense = output / "cells" / "dense" / task
            reference_path = dense / "chunk_000000000" / "complete.json"
            if reference_path.is_file():
                total = json.loads(reference_path.read_text())["chunk"]["total"]
                stop = min(total, limit) if limit is not None else total
                starts = range(0, stop, chunk_size)
            elif method == "dense":
                starts = [0]  # The first processed chunk discovers the native task length.
            else:
                continue
            all_chunks_done = True
            for start in starts:
                chunk = cell / f"chunk_{start:09d}"
                if (chunk / "complete.json").is_file():
                    continue
                all_chunks_done = False
                if method != "dense" and not (dense / chunk.name / "complete.json").is_file():
                    continue
                chunk.mkdir(exist_ok=True)
                lock = cell_lock(chunk, worker_index)
                if lock is not None:
                    # Recheck after obtaining the lock: another worker may have
                    # finished between the initial marker check and this claim.
                    if (chunk / "complete.json").exists():
                        lock.close()
                        continue
                    return method, task, cell, lock, start
            if all_chunks_done and reference_path.is_file():
                lock = cell_lock(cell, worker_index)
                if lock is not None:
                    return method, task, cell, lock, None  # Final aggregation, no GPU work.
    return None


def seed_completed_cells(source: Path, output: Path, methods, tasks) -> None:
    """Import only completion records from a previous run; prediction files stay put."""
    current = json.loads((output / "plan.json").read_text())
    checked = set()
    for method in methods:
        for task in tasks:
            target = output / "cells" / method / task
            if (target / "complete.json").exists():
                continue
            candidates = sorted(source.glob(f"worker_*/cells/{method}/{task}"))
            if not candidates:
                continue
            target.mkdir(parents=True, exist_ok=True)
            for source_cell in candidates:
                source_plan = source_cell.parents[2] / "plan.json"
                if source_plan not in checked:
                    previous = json.loads(source_plan.read_text())
                    for field in ("data_root", "model_identity_sha256", "limit", "chunk_size",
                                  "mmmu_single_image_only", "first_shard_only", "checkpoint_hashes"):
                        if previous.get(field) != current.get(field):
                            raise ValueError(f"Cannot reuse prior run with different {field}: {source_plan}")
                    checked.add(source_plan)
                marker = source_cell / "complete.json"
                if marker.is_file() and not (target / "complete.json").exists():
                    shutil.copy2(marker, target / "complete.json")
                for chunk_marker in source_cell.glob("chunk_*/complete.json"):
                    target_chunk = target / chunk_marker.parent.name
                    target_chunk.mkdir(exist_ok=True)
                    if not (target_chunk / "complete.json").exists():
                        shutil.copy2(chunk_marker, target_chunk / "complete.json")
            if (target / "complete.json").exists():
                print(f"SEEDED completed {method}/{task} from {source}", flush=True)


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
    parser.add_argument("--shared-workers", type=int, default=1,
                        help="Number of workers dynamically claiming request chunks in this shared output")
    parser.add_argument("--worker-index", type=int, default=0,
                        help="Worker identifier for shared queue logs")
    parser.add_argument("--seed-from", type=Path,
                        help="Import only completed/chunk completion markers from an earlier worker_*/ run")
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")
    if args.chunk_size < 1:
        parser.error("--chunk-size must be positive")
    if args.first_shard_only and args.limit is None:
        parser.error("--first-shard-only requires --limit")
    if len(args.methods) != len(set(args.methods)) or len(args.tasks) != len(set(args.tasks)):
        parser.error("Methods and tasks must not contain duplicates")
    if args.shared_workers < 1 or not 0 <= args.worker_index < args.shared_workers:
        parser.error("worker-index must be in [0, shared-workers)")
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
    if args.seed_from:
        seed_done = args.output / "seed_complete.json"
        if args.worker_index == 0:
            if not seed_done.exists():
                seed_completed_cells(args.seed_from.resolve(), args.output, args.methods, args.tasks)
                dump(seed_done, {"source": str(args.seed_from.resolve()), "status": "complete"})
        else:
            deadline = time.monotonic() + 600
            while not seed_done.exists():
                if time.monotonic() >= deadline:
                    raise TimeoutError("Worker 0 did not finish importing prior completion markers")
                time.sleep(0.5)
    dump(args.output / "status.json", {"status": "running", "mode": "limited" if args.limit else "full"})
    os.environ.setdefault("OPENAI_API_KEY", "unused-local-smoke-no-judge")
    from lmms_eval.api.registry import get_aggregation
    from lmms_eval.utils import load_yaml_config
    while True:
        claimed = claim_next_work(args.output, args.methods, args.tasks, args.chunk_size,
                                  args.limit, args.worker_index)
        if claimed is None:
            all_done = all((args.output / "cells" / method / task / "complete.json").is_file()
                           for method in args.methods for task in args.tasks)
            if all_done:
                break
            if args.shared_workers == 1:
                raise RuntimeError("Incomplete cells remain but none could be claimed")
            time.sleep(3)
            continue
        method, task, cell, lock, start = claimed
        try:
            cell = args.output / "cells" / method / task
            cell.mkdir(parents=True, exist_ok=True)
            marker = cell / "complete.json"
            if marker.exists():
                print(f"SKIP completed {method}/{task}", flush=True)
                continue
            chunk_scores = []
            while start is not None:
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
                        failure = {"status": "failed", "method": method, "task": task,
                                   "chunk_start": start, "error": str(exc), "attempt": str(attempt),
                                   "worker_index": args.worker_index}
                        dump(cell / "failed.json", failure)
                        dump(args.output / f"worker_status_{args.worker_index}.json", failure)
                        raise
                chunk_scores.append(result)
                break
            if start is not None:
                continue
            first = json.loads((cell / "chunk_000000000" / "complete.json").read_text())
            total = first["chunk"]["total"]
            stop = min(total, args.limit) if args.limit is not None else total
            chunk_scores = [json.loads((cell / f"chunk_{offset:09d}" / "complete.json").read_text())
                            for offset in range(0, stop, args.chunk_size)]
            for offset, part in zip(range(0, stop, args.chunk_size), chunk_scores):
                if (part["chunk"]["start"] != offset or part["chunk"]["stop"] != min(stop, offset + args.chunk_size)
                        or part["chunk"]["total"] != total or part["samples"] != part["chunk"]["stop"] - offset):
                    raise ValueError(f"Non-contiguous or inconsistent chunk coverage for {method}/{task}")
            metric_name = score_name(task)
            if task in {"hallusionbench", "vqav2_testdev"}:
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
            if task == "vqav2_testdev":
                def submission_rows():
                    for part in chunk_scores:
                        with Path(part["samples_path"]).open() as stream:
                            for line in stream:
                                if line.strip():
                                    yield json.loads(line)
                submissions = vqav2_submissions(submission_rows())
                if len(submissions) != result["samples"]:
                    raise ValueError("VQAv2 test-dev merged submission count differs from processed requests")
                submission_path = cell / "vqav2_testdev_submission.json"
                dump(submission_path, submissions)
                result.update(scoring_status="pending_official_server", submission_path=str(submission_path),
                              source_split="testdev")
            dump(marker, {"method": method, "task": task, "score": result})
            report(args.output, args.methods, args.tasks)
            print(f"PASS {method}/{task}: {result['samples']} requests", flush=True)
        except Exception as exc:
            if not (cell / "failed.json").exists():
                failure = {"status": "failed", "method": method, "task": task, "chunk_start": start,
                           "error": str(exc), "worker_index": args.worker_index}
                dump(cell / "failed.json", failure)
                dump(args.output / f"worker_status_{args.worker_index}.json", failure)
            raise
        finally:
            lock.close()
    dump(args.output / f"worker_status_{args.worker_index}.json", {"status": "complete"})
    all_done = all((args.output / "cells" / method / task / "complete.json").is_file()
                   for method in args.methods for task in args.tasks)
    if all_done:
        report(args.output, args.methods, args.tasks)
        dump(args.output / "status.json", {"status": "complete", "mode": "limited" if args.limit else "full"})
    print(f"Completed; report: {args.output / 'report.md'}", flush=True)


if __name__ == "__main__":
    main()
