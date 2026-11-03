"""Resumable Qwen2.5 evaluation with a shared queue of request chunks.

Pass --limit for a small end-to-end check of the very same full-source path.
Without --limit every selected source shard is evaluated.
"""
from __future__ import annotations

import argparse
import csv
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
from scripts.eval.lmms_eval.portable_results import sample_signature

METHODS = ("dense", "old_mlp", "visual_selector", "image_context_selector", "question_context_selector",
           "protected_selector", "eadp", "prunesid", "mmtok", "importance_fps", "importance_fps_pool")
TRAINED = {"old_mlp", "visual_selector", "image_context_selector", "question_context_selector", "protected_selector",
           "importance_fps", "importance_fps_pool"}
DEFAULT_METHODS = ("dense", "eadp", "prunesid", "mmtok", "old_mlp", "visual_selector",
                   "image_context_selector", "question_context_selector", "protected_selector", "importance_fps", "importance_fps_pool")
DEFAULT_TASKS = ("gqa", "okvqa", "scienceqa_img", "ai2d", "mmstar", "mmmu", "pope", "hallusionbench",
                 "textvqa", "docvqa", "ocrbench", "chartqa", "refcoco_bbox_rec_testA", "refcoco_bbox_rec_testB",
                 "refcoco+_bbox_rec_testA", "refcoco+_bbox_rec_testB", "refcocog_bbox_rec_test")
CATEGORIES = {"General": ("gqa", "okvqa"), "Reasoning": ("scienceqa_img", "ai2d", "mmstar", "mmmu"),
              "Hallusion": ("pope", "hallusionbench"), "OCR": ("textvqa", "docvqa", "ocrbench", "chartqa"),
              "Grounding": ("refcoco_bbox_rec_testA", "refcoco_bbox_rec_testB", "refcoco+_bbox_rec_testA",
                            "refcoco+_bbox_rec_testB", "refcocog_bbox_rec_test")}
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
        scores[task]["sample_signatures_sha256"] = hashlib.sha256(json.dumps([sample_signature(r) for r in ordered]).encode()).hexdigest()
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
    lines = ["# Qwen2.5 evaluation", "", "Raw task scores and dense-relative scores are both retained.",
             "Normalized score = method score / same-task dense score × 100.",
             "Raw OCRBench is /1000; other raw task scores are percentages (DocVQA: ANLS × 100).",
             "Missing or pending scores are —, not zero. Limited checks are not benchmark estimates.",
             "VQAv2 is excluded from the default suite. Category means equally weight the listed tasks; incomplete categories have no mean."]
    if cells.get("dense", {}).get("mmmu", {}).get("subset") == "single_image_only":
        lines.insert(4, "MMMU is the single-image subset only; it is not the official full-validation score.")
    if "vqav2_testdev" in tasks:
        lines.insert(4, "VQAv2 test-dev has no public gold labels; scores are pending official evaluation. Per-method submission JSON files are retained.")
    normalized, raw_display, rows = {}, {}, []
    for method in methods:
        normalized[method] = {}
        raw_display[method] = {}
        for task in tasks:
            row, base = cells.get(method, {}).get(task), cells.get("dense", {}).get(task)
            value = 100 * row["score"] / base["score"] if row and base and row["score"] is not None and base["score"] else None
            normalized[method][task] = value
            raw = row.get("score") if row else None
            display = None if raw is None else raw * (1000 if task == "ocrbench" else 100)
            raw_display[method][task] = display
            rows.append({"method": method, "task": task, "metric": row.get("metric") if row else None,
                "samples": row.get("samples") if row else None, "raw_score": raw, "raw_display": display,
                "raw_unit": "/1000" if task == "ocrbench" else "percent",
                "normalized_percent": value, "status": row.get("scoring_status", "complete") if row else "pending",
                "reused_from": row.get("reused_from") if row else None})
    for heading, values, relative in (("Raw task scores", raw_display, False), ("Dense-relative scores (%)", normalized, True)):
        lines.extend(["", f"## {heading}", "", "| Method | " + " | ".join(tasks) + " |", "|---|" + "---:|" * len(tasks)])
        for method in methods:
            fields = []
            for task in tasks:
                value = values[method][task]
                fields.append("—" if value is None else (f"{value:.0f}" if task == "ocrbench" and not relative else f"{value:.2f}%"))
            lines.append("| " + method + " | " + " | ".join(fields) + " |")
    categories = {}
    for method in methods:
        categories[method] = {}
        for category, members in CATEGORIES.items():
            selected = [task for task in members if task in tasks]
            values = [normalized[method][task] for task in selected]
            categories[method][category] = {"tasks": selected, "completed": sum(v is not None for v in values),
                "expected": len(selected), "normalized_mean": sum(values) / len(values)
                    if values and all(v is not None for v in values) else None}
    lines.extend(["", "## Category means (dense-relative %)", "", "| Method | " + " | ".join(CATEGORIES) + " |",
                  "|---|" + "---:|" * len(CATEGORIES)])
    for method in methods:
        values = [categories[method][category]["normalized_mean"] for category in CATEGORIES]
        lines.append("| " + method + " | " + " | ".join("—" if v is None else f"{v:.2f}%" for v in values) + " |")
    temporary = output / f"report.md.{os.getpid()}.{time.time_ns()}.tmp"
    temporary.write_text("\n".join(lines) + "\n")
    os.replace(temporary, output / "report.md")
    dump(output / "summary.json", {"raw_scores": cells, "raw_display": raw_display,
        "normalized_percent": normalized, "category_means": categories, "category_tasks": CATEGORIES})
    temporary = output / f"scores.csv.{os.getpid()}.{time.time_ns()}.tmp"
    with temporary.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]) if rows else ["method", "task"])
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, output / "scores.csv")


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


def claim_next_work(output, methods, tasks, chunk_size, limit, worker_index, preferred_method=None, reference_only=False):
    """Share chunks, including long test-dev runs, rather than pinning datasets to GPUs."""
    ordered_methods = ([preferred_method] if preferred_method in methods else []) + [m for m in methods if m != preferred_method]
    for method in ordered_methods:
        for task in tasks:
            cell = output / "cells" / method / task
            cell.mkdir(parents=True, exist_ok=True)
            if (cell / "complete.json").is_file():
                continue
            if (cell / "failed.json").is_file():
                raise RuntimeError(f"Failed cell requires inspection: {method}/{task}")
            dense = output / "cells" / "dense" / task
            reference_path = dense / "chunk_000000000" / "complete.json"
            if (dense / "complete.json").is_file():
                reference = json.loads((dense / "complete.json").read_text())["score"]
                total = reference.get("processed_total", reference["samples"])
                stop = min(total, limit) if limit is not None else total
                starts = range(0, stop, chunk_size)
            elif reference_path.is_file():
                total = json.loads(reference_path.read_text())["chunk"]["total"]
                stop = min(total, limit) if limit is not None else total
                starts = range(0, stop, chunk_size)
            elif method == "dense":
                starts = [0]  # The first processed chunk discovers the native task length.
            elif reference_only:
                prior = next((output / "cells" / m / task / "chunk_000000000" / "complete.json"
                              for m in methods if (output / "cells" / m / task / "chunk_000000000" / "complete.json").is_file()), None)
                if prior is not None:
                    total = json.loads(prior.read_text())["chunk"]["total"]
                    stop = min(total, limit) if limit is not None else total
                    starts = range(0, stop, chunk_size)
                else:
                    starts = [0]
            else:
                continue
            all_chunks_done = True
            for start in starts:
                chunk = cell / f"chunk_{start:09d}"
                if (chunk / "complete.json").is_file():
                    continue
                all_chunks_done = False
                if method != "dense" and not reference_only and not (dense / "complete.json").is_file() and not (dense / chunk.name / "complete.json").is_file():
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
            if all_chunks_done and (cell / "chunk_000000000" / "complete.json").is_file():
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
            if (source / "cells" / method / task).is_dir():
                candidates.insert(0, source / "cells" / method / task)
            if not candidates:
                continue
            target.mkdir(parents=True, exist_ok=True)
            for source_cell in candidates:
                source_plan = source_cell.parents[2] / "plan.json"
                if source_plan not in checked:
                    previous = json.loads(source_plan.read_text())
                    for field in ("data_root", "model_identity_sha256", "limit",
                                  "mmmu_single_image_only", "first_shard_only"):
                        if previous.get(field) != current.get(field):
                            raise ValueError(f"Cannot reuse prior run with different {field}: {source_plan}")
                    checked.add(source_plan)
                previous = json.loads(source_plan.read_text())
                if method in TRAINED and previous.get("checkpoint_hashes", {}).get(method) != current["checkpoint_hashes"].get(method):
                    raise ValueError(f"Checkpoint changed for reused {method}: {source_plan}")
                if method in {"importance_fps", "importance_fps_pool"} and previous.get("scorer_hashes") != current.get("scorer_hashes"):
                    raise ValueError("Scorer changed for reused importance-FPS results")
                marker = source_cell / "complete.json"
                if marker.is_file() and not (target / "complete.json").exists():
                    payload = json.loads(marker.read_text())
                    payload["score"]["reused_from"] = str(source_cell.resolve())
                    dump(target / "complete.json", payload)
                # Full-cell reuse is independent of chunk size. Partial chunks
                # can only be resumed under their original boundaries.
                for chunk_marker in source_cell.glob("chunk_*/complete.json") if previous["chunk_size"] == current["chunk_size"] else ():
                    target_chunk = target / chunk_marker.parent.name
                    target_chunk.mkdir(exist_ok=True)
                    if not (target_chunk / "complete.json").exists():
                        shutil.copy2(chunk_marker, target_chunk / "complete.json")
            if (target / "complete.json").exists():
                print(f"SEEDED completed {method}/{task} from {source}", flush=True)


def baseline_hashes(cell):
    visited = set()
    while (cell / "complete.json").is_file():
        if cell.resolve() in visited:
            raise ValueError("Cycle in reused result provenance")
        visited.add(cell.resolve())
        marker = cell / "complete.json"
        record = json.loads(marker.read_text())["score"]
        if record.get("evidence_path"):
            return json.loads(Path(record["evidence_path"]).read_text())["sample_signatures"]
        if not record.get("reused_from"):
            break
        cell = Path(record["reused_from"])
    hashes = {}
    for path in sorted(cell.glob("chunk_*/complete.json")):
        chunk = json.loads(path.read_text())
        start = chunk["chunk"]["start"]
        with Path(chunk["samples_path"]).open() as stream:
            for row in map(json.loads, stream):
                hashes[str(start + row["doc_id"])] = sample_signature(row)
    return hashes


def validate_dense_chunk(output, task, result, reference_only=False):
    cell = output / "cells" / "dense" / task
    start, stop = result["chunk"]["start"], result["chunk"]["stop"]
    path = cell / f"chunk_{start:09d}" / "complete.json"
    if path.is_file():
        baseline = json.loads(path.read_text())
        if baseline["chunk"] == result["chunk"] and "sample_signatures_sha256" in baseline:
            for field in ("source_stats", "source_rows", "eval_rows", "samples", "chunk", "doc_hashes_sha256", "sample_signatures_sha256"):
                if result[field] != baseline[field]:
                    raise ValueError(f"Dataset differs from dense for {task}/{start}: {field}")
            return
    hashes = baseline_hashes(cell)
    if reference_only and not all(str(index) in hashes for index in range(start, stop)):
        for candidate in sorted((output / "cells").glob(f"*/{task}")):
            if candidate.parent.name == "dense":
                continue
            alternative = baseline_hashes(candidate)
            if all(str(index) in alternative for index in range(start, stop)):
                hashes = alternative
                break
        else:
            result["reference_validation"] = "pending_matching_reference"
            return
    expected = [hashes[str(index)] for index in range(start, stop)]
    digest = hashlib.sha256(json.dumps(expected).encode()).hexdigest()
    if any(value is None for value in expected) or digest != result["sample_signatures_sha256"]:
        raise ValueError(f"Dataset differs from reused dense for {task}/{start}")


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, help="Shared writable dataset cache; defaults under output")
    parser.add_argument("--reference-identity", type=Path)
    parser.add_argument("--checkpoint", action="append", default=[], metavar="METHOD=DIR")
    parser.add_argument("--scorer-checkpoint", type=Path, help="Frozen image-context scorer used by both R1 and T2")
    parser.add_argument("--resident-model", action="store_true", help="Reuse a platform model across chunks using the native evaluator")
    parser.add_argument("--reference-only", action="store_true", help="Reuse dense results for reporting; never schedule dense generation")
    parser.add_argument("--result-bundle", type=Path, help="Portable full P6000 results with dataset and protocol checksums")
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=list(DEFAULT_METHODS))
    parser.add_argument("--tasks", nargs="+", choices=[row[0] for row in specifications()], default=list(DEFAULT_TASKS))
    parser.add_argument("--limit", type=int, help="Samples per full-source task; omit for full evaluation")
    parser.add_argument("--chunk-size", type=int, default=1024, help="Requests per checkpoint; resident models are not reloaded at boundaries")
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
    parser.add_argument("--seed-from", type=Path, action="append", default=[],
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
    if "dense" not in args.methods and not args.reference_only:
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
    if set(args.methods) & {"importance_fps", "importance_fps_pool"} and not args.scorer_checkpoint:
        parser.error("R1/T2 require --scorer-checkpoint (existing image-context selector)")
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
    scorer_hashes = ({p.name: sha(p) for p in args.scorer_checkpoint.iterdir()
                      if p.is_file() and p.suffix in {".json", ".safetensors"}} if args.scorer_checkpoint else {})
    plan = {"data_root": str(args.data_root.resolve()), "model": str(args.model.resolve()),
            "model_identity_sha256": json.loads(identity.read_text())["identity_sha256"] if identity else None,
            "methods": args.methods, "tasks": args.tasks, "limit": args.limit, "chunk_size": args.chunk_size,
            "mmmu_single_image_only": args.mmmu_single_image_only,
            "first_shard_only": args.first_shard_only,
            "cache_dir": str(args.cache_dir.resolve()) if args.cache_dir else str(args.output / "cache"),
            "checkpoints": {k: str(v) for k, v in checkpoints.items()}, "checkpoint_hashes": hashes,
            "scorer_hashes": scorer_hashes, "scorer_checkpoint": str(args.scorer_checkpoint.resolve()) if args.scorer_checkpoint else None,
            "resident_model": args.resident_model, "reference_only": args.reference_only,
            "result_bundle_sha256": sha(args.result_bundle) if args.result_bundle else None,
            "source_hashes": {name: sha(ROOT / name) for name in (
                "scripts/eval/lmms_eval/qwen25_portable_queue.py", "scripts/eval/lmms_eval/p6000_dataset_smoke.py",
                "scripts/eval/lmms_eval/resident_lmms.py", "scripts/eval/lmms_eval/portable_results.py",
                "mllm_comp/compressors/comparison/importance_anchor_adapter.py",
                "mllm_comp/compressors/image/spatial_anchor_ablation.py")},
            "hallusion_image": str(args.hallusion_image.resolve()) if args.hallusion_image else None}
    plan_path = args.output / "plan.json"
    if plan_path.exists() and json.loads(plan_path.read_text()) != plan:
        raise ValueError("Run plan changed; choose a new output rather than resuming")
    dump(plan_path, plan)
    if args.seed_from or args.result_bundle:
        seed_done = args.output / "seed_complete.json"
        if args.worker_index == 0:
            if not seed_done.exists():
                for source in args.seed_from:
                    seed_completed_cells(source.resolve(), args.output, args.methods, args.tasks)
                if args.result_bundle:
                    from scripts.eval.lmms_eval.portable_results import import_bundle
                    import_bundle(args.result_bundle, args.output, args.methods, args.tasks)
                dump(seed_done, {"sources": [str(s.resolve()) for s in args.seed_from], "status": "complete"})
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
    scheduled_methods = [m for m in args.methods if not (args.reference_only and m == "dense")]
    resident = None
    preferred_method = None
    if args.resident_model:
        from scripts.eval.lmms_eval.resident_lmms import ResidentEvaluator
        resident = ResidentEvaluator(args.output / f"resident_worker_{args.worker_index}.json")
        import atexit
        atexit.register(resident.close)
    while True:
        claimed = claim_next_work(args.output, scheduled_methods, args.tasks, args.chunk_size,
                                  args.limit, args.worker_index, preferred_method=preferred_method,
                                  reference_only=args.reference_only)
        if claimed is None:
            all_done = all((args.output / "cells" / method / task / "complete.json").is_file()
                           for method in scheduled_methods for task in args.tasks)
            if all_done:
                break
            if args.shared_workers == 1:
                raise RuntimeError("Incomplete cells remain but none could be claimed")
            time.sleep(3)
            continue
        method, task, cell, lock, start = claimed
        preferred_method = method
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
                    if method in {"importance_fps", "importance_fps_pool"}: cmd += ["--scorer-checkpoint", str(args.scorer_checkpoint.resolve())]
                    if args.hallusion_image: cmd += ["--hallusion-image", str(args.hallusion_image.resolve())]
                    if task == "hallusionbench": cmd += ["--hallusion-scoring", "deferred"]
                    print(f"START {method}/{task}/{start}; log: {attempt / 'worker.log'}", flush=True)
                    try:
                        if resident is None:
                            with (attempt / "worker.log").open("x") as log:
                                completed = subprocess.run(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
                            if completed.returncode:
                                raise RuntimeError(f"Evaluator exited {completed.returncode}; see {attempt / 'worker.log'}")
                        else:
                            import argparse as _argparse
                            from scripts.eval.lmms_eval.p6000_dataset_smoke import prepare
                            preparation = _argparse.Namespace(data_root=args.data_root, model=str(args.model), output=attempt,
                                samples=2, input_mode="full", eval_limit=args.limit if args.first_shard_only else None,
                                first_shard_only=args.first_shard_only, shared_cache=Path(plan["cache_dir"]),
                                chunk_start=start, chunk_size=size, mmmu_single_image_only=args.mmmu_single_image_only,
                                hallusion_scoring="deferred" if task == "hallusionbench" else "smoke",
                                gpu=args.gpu, method=method, identity_path=str(identity) if identity else None,
                                method_checkpoint=str(checkpoints[method]) if method in TRAINED else None,
                                scorer_checkpoint=str(args.scorer_checkpoint) if args.scorer_checkpoint else None,
                                hallusion_image=args.hallusion_image, tasks=[task])
                            manifest = prepare(preparation)
                            if manifest["status"] != "prepared":
                                raise RuntimeError(f"Dataset preparation failed: {attempt / 'manifest.json'}")
                            resident.evaluate(attempt, manifest, attempt / "worker.log")
                        result = collect(attempt, [task], None)[task]
                        if result["chunk"]["start"] != start:
                            raise ValueError(f"Incorrect chunk start for {method}/{task}")
                        if method != "dense":
                            validate_dense_chunk(args.output, task, result, reference_only=args.reference_only)
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
            if args.limit is not None:
                result["evaluation_scope"] = "limited_smoke"
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
                   for method in scheduled_methods for task in args.tasks)
    if all_done:
        report(args.output, args.methods, args.tasks)
        dump(args.output / "status.json", {"status": "complete", "mode": "limited" if args.limit else "full"})
    print(f"Completed; report: {args.output / 'report.md'}", flush=True)
    if resident is not None:
        resident.close()


if __name__ == "__main__":
    main()
