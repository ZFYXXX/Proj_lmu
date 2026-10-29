"""Qwen2.5-VL-7B: full POPE followed by RefCOCO+ testA/testB.

The five requested projector-level conditions share the native lmms-eval entry,
prompts, dynamic image resolution and generation settings. Grounding is scored
with the repository's versioned qwen-grounding-v2 pixel-coordinate protocol.
"""
from __future__ import annotations

import argparse
import ast
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import copy
from datetime import datetime, timezone
import fcntl
import hashlib
from io import BytesIO
import json
import math
import os
from pathlib import Path
from queue import Empty, Queue
import subprocess
import sys
from threading import Event
import time
import traceback

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from configs.eval.lmms_eval.tasks.projector_extra import extra_utils
from mllm_comp.analysis.common.anchor7b_runtime import file_sha, validate_identity
from mllm_comp.analysis.common.learned_topk_experiment import (
    dependency_identity,
    gpu_free_mib,
    owned_output,
    reject_overlap,
    source_identity,
)
from mllm_comp.core.serialization import atomic_write_json as write_json
from mllm_comp.core.serialization import atomic_write_text as write_text
from mllm_comp.integrations.lmms_eval.grounding import PROTOCOL, box_scores
from scripts.eval.lmms_eval import run_projector_extra_benchmarks as base


METHODS = ("dense", "visual_selector", "old_mlp", "image_context_selector", "eadp")
TASKS = ("pope", "refcoco_plus_testA", "refcoco_plus_testB")
TASK_NAMES = {
    "pope": "qwen25_pope_grounding_pope",
    "refcoco_plus_testA": "refcoco+_bbox_rec_testA_local",
    "refcoco_plus_testB": "refcoco+_bbox_rec_testB_local",
}
COUNTS = {"pope": 9000, "refcoco_plus_testA": 5726, "refcoco_plus_testB": 4889}
SPLITS = {"pope": "test", "refcoco_plus_testA": "testA", "refcoco_plus_testB": "testB"}
DATA = {
    "pope": [Path("/home/YinPeiran/datasets/lmms_eval_datasets/POPE/Full") / f"{name}-00000-of-00001.parquet"
             for name in ("adversarial", "popular", "random")],
    "refcoco_plus_testA": [Path("/home/YinPeiran/datasets/lmms_eval_datasets/RefCOCOplus/data/testA-00000-of-00001.parquet")],
    "refcoco_plus_testB": [Path("/home/YinPeiran/datasets/lmms_eval_datasets/RefCOCOplus/data/testB-00000-of-00001.parquet")],
}
TASK_DIR = ROOT / "configs/eval/lmms_eval/tasks/qwen25_pope_grounding"
Q25 = ROOT / "outputs/train/qwen25_vl/context_visual_topk_20261024_v2"


def read(path):
    return json.loads(Path(path).read_text())


def now():
    return datetime.now(timezone.utc).isoformat()


def _raw_image(doc):
    value = doc["image"]
    if isinstance(value, dict):
        return value.get("bytes"), value.get("path")
    return value, None


def _image_key(doc):
    raw, path = _raw_image(doc)
    return hashlib.sha256(raw).hexdigest() if isinstance(raw, bytes) else str(path or raw)


def _image_size(doc):
    from PIL import Image
    raw, path = _raw_image(doc)
    source = BytesIO(raw) if isinstance(raw, bytes) else path or raw
    with Image.open(source) as image:
        return image.size


def templates():
    all_configs, provenance = base.template_configs()
    return {method: all_configs["qwen25"][method] for method in METHODS}, provenance


def prepare(root):
    import pyarrow.parquet as pq
    from mllm_comp.analysis.common.visual_token_phenomena import media_fingerprint

    if root.exists():
        raise ValueError("Use a fresh output directory; existing results are never overwritten")
    root.mkdir(parents=True)
    write_json(root / "manifest.json", {"status": "preparing", "started_utc": now()})
    try:
        training_manifest = read(Q25 / "protocol.json")["training_manifest"]
        training_entries = read(training_manifest)["entries"]
        cohort, image_cache, size_cache = {}, {}, {}
        pins = {training_manifest: file_sha(training_manifest)}
        for task, paths in DATA.items():
            rows = []
            for path in paths:
                pins[str(path)] = file_sha(path)
                for batch in pq.ParquetFile(path).iter_batches(batch_size=16):
                    for doc in batch.to_pylist():
                        key = _image_key(doc)
                        if key not in image_cache:
                            image_cache[key] = media_fingerprint([extra_utils.image_from_doc(doc)])
                        fingerprint = image_cache[key]
                        if task == "pope":
                            rows.append({
                                "row": len(rows), "image_fingerprint": fingerprint,
                                "category": doc["category"], "target": doc["answer"],
                                "sample_id": str(doc.get("index", doc.get("question_id", doc.get("id")))),
                                "prompt": extra_utils.pope_doc_to_text(
                                    doc, {"post_prompt": "\nAnswer the question using a single word or phrase."}),
                            })
                        else:
                            if key not in size_cache:
                                size_cache[key] = _image_size(doc)
                            width, height = size_cache[key]
                            x, y, w, h = doc["bbox"]
                            target = [x / width, y / height, (x + w) / width, (y + h) / height]
                            for answer_index, answer in enumerate(doc["answer"]):
                                rows.append({
                                    "row": len(rows), "image_fingerprint": fingerprint,
                                    "category": SPLITS[task], "target": target,
                                    "sample_id": f'{doc["question_id"]}:{answer_index}',
                                    "prompt": f"Locate {answer} in this image and output the bbox coordinates in JSON format.",
                                })
            if len(rows) != COUNTS[task]:
                raise ValueError(f"Unexpected {task} size: {len(rows)}")
            overlap = sorted({entry["image_fingerprint"] for entry in training_entries}
                             & {row["image_fingerprint"] for row in rows})
            write_json(root / f"{task}_training_overlap.json", {"overlap_images": len(overlap), "overlap": overlap})
            reject_overlap(training_entries, rows)
            smoke = extra_utils.smoke_indices([row["category"] for row in rows]) if task == "pope" else list(range(8))
            cohort[task] = {
                "rows": rows, "paths": [str(path) for path in paths], "smoke_indices": smoke,
                "independent_images": len({row["image_fingerprint"] for row in rows}),
            }
            print(f"{task}: {len(rows)} requests / {cohort[task]['independent_images']} image fingerprints", flush=True)
        if Counter(row["category"] for row in cohort["pope"]["rows"]) != {
                "adversarial": 3000, "popular": 3000, "random": 3000}:
            raise ValueError("POPE does not contain all three complete strategies")
        write_json(root / "cohort.json", cohort)
        pins[str(root / "cohort.json")] = file_sha(root / "cohort.json")
        configs, provenance = templates()
        pins.update(provenance)
        for path in TASK_DIR.glob("*.py"):
            pins[str(path)] = file_sha(path)
        for path in TASK_DIR.glob("*.yaml"):
            pins[str(path)] = file_sha(path)
        for cfg in configs.values():
            for stage in cfg["pipeline"]["stages"]:
                options = stage["config"]
                for key in ("checkpoint_path", "adapter_path"):
                    location = options.get(key)
                    if location:
                        for path in Path(location).iterdir():
                            if path.suffix in (".safetensors", ".json"):
                                pins[str(path)] = file_sha(path)
                if options.get("identity_path"):
                    pins[options["identity_path"]] = file_sha(options["identity_path"])
                    validate_identity(read(options["identity_path"]))
        protocol = {
            "created_utc": now(), "source": source_identity(ROOT), "dependencies": dependency_identity(),
            "templates": configs, "input_hashes": pins, "methods": list(METHODS), "tasks": list(TASKS),
            "counts": COUNTS, "no_training": True, "backbone": "Qwen2.5-VL-7B-Instruct",
            "budget": "compressed methods use min(128,N) post-merger tokens at native dynamic resolution",
            "grounding_protocol": PROTOCOL,
            "generation": {"greedy": True, "repetition_penalty": 1.05,
                           "pope_max_new_tokens": 16, "grounding_max_new_tokens": 64},
            "primary_metrics": {"pope": "accuracy", "grounding": "ACC@0.5"},
            "limits": [
                "EADP is the current Qwen projector-level adaptation, not a certified original-paper reproduction.",
                "RefCOCO+ expressions sharing an image are clustered by image for confidence intervals.",
                "Latency is end-to-end model request time under shared-GPU conditions.",
            ],
        }
        write_json(root / "protocol.json", protocol)
        write_json(root / "manifest.json", {
            "status": "prepared", "protocol_sha256": file_sha(root / "protocol.json"),
            "total_full_cells": len(METHODS) * len(TASKS),
            "full_requests": len(METHODS) * sum(COUNTS.values()), "updated_utc": now(),
        })
        report(root, protocol)
    except BaseException as error:
        write_json(root / "manifest.json", {"status": "failed_preparation", "error": str(error), "updated_utc": now()})
        raise


def validate(root):
    protocol = read(root / "protocol.json")
    if read(root / "manifest.json")["protocol_sha256"] != file_sha(root / "protocol.json"):
        raise ValueError("Protocol changed")
    if protocol["source"] != source_identity(ROOT) or protocol["dependencies"] != dependency_identity():
        raise ValueError("Source or evaluator changed; create a new run")
    for path, checksum in protocol["input_hashes"].items():
        if file_sha(path) != checksum:
            raise ValueError(f"Pinned input changed: {path}")
    return protocol


def make_config(protocol, method, phase, task, run):
    cfg = copy.deepcopy(protocol["templates"][method])
    cfg["experiment"]["name"] = f"qwen25_pope_refcocoplus_{phase}_{method}_{task}"
    cfg["eval"].update(tasks=[TASK_NAMES[task]], sample_limit=None)
    cfg["eval"]["args"]["include_path"] = str(TASK_DIR)
    cfg["data"] = {"dataset_id": "local_pope_refcocoplus", "sample_split": SPLITS[task]}
    cfg["output"] = {"run_dir": str(run), "manifest_path": str(run / "manifest.json")}
    return cfg


def _close_target(actual, expected):
    # lmms-eval serializes list-valued doc_to_target fields as their Python
    # representation in sample JSONL. Parse only this trusted local artifact.
    if isinstance(actual, str) and isinstance(expected, list):
        try:
            actual = ast.literal_eval(actual)
        except (ValueError, SyntaxError):
            return False
    if isinstance(actual, list) and isinstance(expected, list) and len(actual) == len(expected):
        return all(abs(float(a) - float(b)) < 1e-6 for a, b in zip(actual, expected))
    return actual == expected


def check(root, method, phase, task, run):
    cohort = read(root / "cohort.json")[task]
    expected = ([cohort["rows"][i] for i in cohort["smoke_indices"]]
                if phase == "smoke" else cohort["rows"])
    name, count = TASK_NAMES[task], len(expected)
    samples = list(run.glob(f"**/*_samples_{name}.jsonl"))
    results = list(run.glob("**/*_results.json"))
    if len(samples) != 1 or len(results) != 1:
        raise ValueError("Missing or duplicate native evaluator artifacts")
    rows = [json.loads(line) for line in samples[0].read_text().splitlines()]
    if len(rows) != count or {row["doc_id"] for row in rows} != set(range(count)):
        raise ValueError("Missing or duplicate native predictions")
    rows.sort(key=lambda row: row["doc_id"])
    for row, item in zip(rows, expected):
        if row["input"] != item["prompt"] or not _close_target(row["target"], item["target"]):
            raise ValueError("Prompt/target alignment changed")
    native = read(results[0])["results"][name]
    if task == "pope":
        score = sum(float(row["pope_accuracy"]) for row in rows) / count
        primary = "pope_accuracy"
        metrics = {"accuracy": score}
        for category in sorted({item["category"] for item in expected}):
            payloads = [row["pope_precision"] for row, item in zip(rows, expected) if item["category"] == category]
            metrics[category] = extra_utils.pope_statistics(payloads)
    else:
        metric_values = {metric: [] for metric in ("IoU", "ACC@0.1", "ACC@0.3", "ACC@0.5", "ACC@0.7", "ACC@0.9", "Center_ACC")}
        valid = 0
        for row in rows:
            payload = row["refcoco_ACC@0.5"]
            if payload.get("grounding_protocol") != PROTOCOL or payload.get("coordinate_space") != "processed_pixels":
                raise ValueError("Grounding scorer did not use qwen-grounding-v2 processed-pixel coordinates")
            if not payload.get("input_size") or min(payload["input_size"]) <= 0:
                raise ValueError("Missing verified Qwen2.5 processor image size")
            if not all(math.isfinite(float(value)) for value in payload["pred"]):
                raise ValueError("Non-finite grounding prediction")
            valid += payload["parse_status"] == "valid"
            scores = box_scores(payload["pred"], payload["bbox"])
            for metric, value in scores.items():
                metric_values[metric].append(value)
        metrics = {metric: sum(values) / count for metric, values in metric_values.items()}
        metrics["parse_valid_fraction"] = valid / count
        score, primary = metrics["ACC@0.5"], "refcoco_ACC@0.5"
    matching = [key for key in native if key.split(",")[0] == primary]
    if len(matching) != 1 or not math.isfinite(score) or abs(float(native[matching[0]]) - score) > 1e-10:
        raise ValueError("Native aggregate and per-request scores disagree")
    performance = [json.loads(line) for line in (run / "request_performance.jsonl").read_text().splitlines()]
    def doc_id(sample_id):
        fields = sample_id.split(":")
        if fields[0] != name or fields[1] != SPLITS[task]:
            raise ValueError(f"Unexpected request identity: {sample_id}")
        return int(fields[2])
    perf = {doc_id(row["sample_id"]): row for row in performance}
    if len(performance) != count or set(perf) != set(range(count)):
        raise ValueError("Missing or duplicate performance rows")
    if any(row["status"] != "complete" or row["original_visual_tokens"] <= 0
           or not math.isfinite(row["generation_s"]) for row in performance):
        raise ValueError("Failed or non-finite request")
    retained = [perf[index]["original_visual_tokens"] for index in range(count)]
    if method != "dense":
        traces = [json.loads(line) for line in (run / "compression_runtime.requests.jsonl").read_text().splitlines()]
        indexed = {doc_id(row["request"]["sample_id"]): row for row in traces}
        if len(traces) != count or set(indexed) != set(range(count)):
            raise ValueError("Missing or duplicate compression traces")
        for index in range(count):
            activation, = indexed[index]["runtime"]["activations"]
            retained[index] = base.check_trace(activation, perf[index]["original_visual_tokens"], 28)
    checked = {
        "requests": count, "score": score, "metrics": metrics, "native_metrics": native,
        "samples": str(samples[0]), "samples_sha256": file_sha(samples[0]),
        "mean_source_tokens": sum(row["original_visual_tokens"] for row in performance) / count,
        "mean_retained_tokens": sum(retained) / count,
        "mean_retained_fraction": sum(retained[i] / perf[i]["original_visual_tokens"] for i in range(count)) / count,
        "generation_seconds": sum(row["generation_s"] for row in performance),
        "peak_allocated_bytes": max(row["peak_allocated_bytes"] or 0 for row in performance),
    }
    write_json(run / "protocol_checks.json", checked)
    return checked


def child_env(root, gpu, phase, run):
    env = base.child_env(root, "qwen25", gpu, phase, run)
    env["MLLM_EXTRA_SMOKE"] = "1" if phase == "smoke" else "0"
    return env


def evaluate(root, protocol, method, phase, task, gpu):
    import yaml
    directory = root / "eval" / phase / method / task
    marker = directory / "complete.json"
    if marker.exists():
        saved = read(marker)
        if file_sha(saved["config"]) != saved["config_sha256"] or file_sha(saved["samples"]) != saved["samples_sha256"]:
            raise ValueError("Completed result provenance changed")
        return
    stamp = time.time_ns()
    run = directory / f"attempt_{stamp}"
    config_path = root / "configs" / f"{phase}_{method}_{task}_{stamp}.yaml"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    write_text(config_path, yaml.safe_dump(make_config(protocol, method, phase, task, run), sort_keys=False))
    while gpu_free_mib().get(gpu, 0) < 46000:
        write_json(root / f"worker_gpu{gpu}.json", {"status": "waiting_for_memory", "updated_utc": now()})
        time.sleep(30)
    directory.mkdir(parents=True, exist_ok=True)
    log = directory / f"launch_{stamp}.log"
    command = [sys.executable, "-m", "mllm_comp.cli.main", "eval", "--config", str(config_path)]
    with log.open("x") as stream:
        child = subprocess.Popen(command, cwd=ROOT, env=child_env(root, gpu, phase, run), stdout=stream, stderr=subprocess.STDOUT)
        write_json(root / f"worker_gpu{gpu}.json", {
            "status": "running", "method": method, "task": task, "phase": phase,
            "pid": child.pid, "log": str(log), "updated_utc": now(),
        })
        return_code = child.wait()
    if return_code:
        raise RuntimeError(f"Native eval failed ({return_code}); preserved {log}")
    checked = check(root, method, phase, task, run)
    write_json(marker, {**checked, "run": str(run), "config": str(config_path),
                       "config_sha256": file_sha(config_path), "finished_utc": now()})


def report(root, protocol):
    records = []
    lines = ["# Qwen2.5-VL-7B: POPE and RefCOCO+ K128", "",
             "| Method | POPE accuracy | RefCOCO+ testA ACC@0.5 | RefCOCO+ testB ACC@0.5 |",
             "|---|---:|---:|---:|"]
    for method in METHODS:
        values = []
        for task in TASKS:
            marker = root / "eval/full" / method / task / "complete.json"
            if marker.exists():
                item = read(marker)
                records.append({"method": method, "task": task, **item})
                values.append(f'{item["score"] * 100:.2f}%')
            else:
                values.append("pending")
        lines.append("| " + " | ".join((method, *values)) + " |")
    write_json(root / "summary.json", {"complete_cells": len(records), "total_cells": 15, "results": records})
    write_text(root / "report.md", "\n".join(lines + ["", *protocol["limits"]]) + "\n")


def run_jobs(root, protocol, gpus, phase, tasks):
    queue, stop = Queue(), Event()
    for task in tasks:
        for method in METHODS:
            queue.put((method, task))
    def worker(gpu):
        try:
            while not stop.is_set():
                try:
                    method, task = queue.get_nowait()
                except Empty:
                    return
                evaluate(root, protocol, method, phase, task, gpu)
                report(root, protocol)
        except BaseException as error:
            stop.set()
            write_json(root / f"worker_gpu{gpu}.json", {
                "status": "failed", "error": str(error), "traceback": traceback.format_exc(), "updated_utc": now()})
            raise
    with ThreadPoolExecutor(max_workers=len(gpus)) as pool:
        jobs = [pool.submit(worker, gpu) for gpu in gpus]
        for job in jobs:
            job.result()


def final_statistics(root):
    cohort, comparisons = read(root / "cohort.json"), []
    for task in TASKS:
        def rows(method):
            marker = read(root / "eval/full" / method / task / "complete.json")
            return sorted((json.loads(line) for line in Path(marker["samples"]).read_text().splitlines()),
                          key=lambda row: row["doc_id"])
        baseline = rows("dense")
        def value(row):
            if task == "pope":
                return float(row["pope_accuracy"])
            payload = row["refcoco_ACC@0.5"]
            return box_scores(payload["pred"], payload["bbox"])["ACC@0.5"]
        for method in METHODS[1:]:
            actual = rows(method)
            comparisons.append({
                "task": task, "method": method,
                **base.bootstrap([value(a) - value(b) for a, b in zip(actual, baseline)],
                                 [row["image_fingerprint"] for row in cohort[task]["rows"]]),
            })
    write_json(root / "paired_bootstrap.json", {"comparisons": comparisons})


def pipeline(root, gpus):
    protocol = validate(root)
    with (root / "pipeline.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        state = read(root / "manifest.json")
        def stage(status, **extra):
            state.update(status=status, pid=os.getpid(), gpus=gpus, updated_utc=now(), **extra)
            write_json(root / "manifest.json", state)
        try:
            stage("smoke")
            run_jobs(root, protocol, gpus, "smoke", TASKS)
            # User requested POPE first. Complete its five full conditions before grounding.
            stage("full_pope")
            run_jobs(root, protocol, gpus, "full", ("pope",))
            stage("full_refcocoplus_testA")
            run_jobs(root, protocol, gpus, "full", ("refcoco_plus_testA",))
            stage("full_refcocoplus_testB")
            run_jobs(root, protocol, gpus, "full", ("refcoco_plus_testB",))
            report(root, protocol)
            if read(root / "summary.json")["complete_cells"] != 15:
                raise ValueError("Full matrix incomplete")
            stage("final_statistics")
            final_statistics(root)
            stage("complete")
        except BaseException as error:
            stage("failed", error=str(error), traceback=traceback.format_exc())
            raise


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("prepare", "pipeline", "report"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--gpus", type=int, nargs="+", default=[0, 1])
    args = parser.parse_args()
    root = owned_output(args.root)
    if not args.gpus or len(set(args.gpus)) != len(args.gpus) or set(args.gpus) - {0, 1}:
        raise ValueError("Only distinct GPU0/GPU1 workers are supported")
    if args.command == "prepare":
        prepare(root)
    elif args.command == "pipeline":
        pipeline(root, args.gpus)
    else:
        report(root, read(root / "protocol.json"))


if __name__ == "__main__":
    main()
