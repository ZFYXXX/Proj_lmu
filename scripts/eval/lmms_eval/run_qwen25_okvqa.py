"""Queue the five frozen Qwen2.5 projector conditions on full OK-VQA val."""
from __future__ import annotations

import argparse
import ast
from concurrent.futures import ThreadPoolExecutor
import copy
from datetime import datetime, timezone
import fcntl
import hashlib
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

from mllm_comp.analysis.common.anchor7b_runtime import file_sha, validate_identity
from mllm_comp.analysis.common.learned_topk_experiment import (
    dependency_identity, gpu_free_mib, owned_output, reject_overlap, source_identity,
)
from mllm_comp.core.serialization import atomic_write_json as write_json
from mllm_comp.core.serialization import atomic_write_text as write_text
from scripts.eval.lmms_eval import run_projector_extra_benchmarks as base
from scripts.eval.lmms_eval import run_qwen25_pope_refcocoplus as prior

METHODS = prior.METHODS
COUNT = 5046
TASK_NAME = "qwen25_ok_vqa_val2014"
SPLIT = "validation"
DATA = tuple(sorted(Path("/home/YinPeiran/datasets/lmms_eval_datasets/OK-VQA/data").glob("val2014-*.parquet")))
TASK_DIR = ROOT / "configs/eval/lmms_eval/tasks/qwen25_okvqa"
Q25 = ROOT / "outputs/train/qwen25_vl/context_visual_topk_20261024_v2"
PREREQUISITE = ROOT / "outputs/eval/qwen25_vl/pope_refcocoplus_k128_20261027_v2/manifest.json"


def read(path):
    return json.loads(Path(path).read_text())


def now():
    return datetime.now(timezone.utc).isoformat()


def _pid_alive(pid):
    """Return whether a recorded supervisor PID is still running."""
    if not isinstance(pid, int) or pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except (ProcessLookupError, PermissionError, OSError):
        return False
    return True


def prepare(root):
    import pyarrow.parquet as pq
    from mllm_comp.analysis.common.visual_token_phenomena import media_fingerprint
    from configs.eval.lmms_eval.tasks.projector_extra.extra_utils import image_from_doc

    if root.exists():
        raise ValueError("Use a fresh output directory")
    root.mkdir(parents=True)
    write_json(root / "manifest.json", {"status": "preparing", "started_utc": now()})
    try:
        training_manifest = read(Q25 / "protocol.json")["training_manifest"]
        training = read(training_manifest)["entries"]
        pins = {training_manifest: file_sha(training_manifest)}
        rows, fingerprints = [], {}
        for path in DATA:
            pins[str(path)] = file_sha(path)
            for batch in pq.ParquetFile(path).iter_batches(batch_size=16):
                for doc in batch.to_pylist():
                    value = doc["image"]
                    raw = value.get("bytes") if isinstance(value, dict) else value
                    key = hashlib.sha256(raw).hexdigest() if isinstance(raw, bytes) else str(value)
                    if key not in fingerprints:
                        fingerprints[key] = media_fingerprint([image_from_doc(doc)])
                    rows.append({
                        "row": len(rows), "question_id": str(doc["question_id"]),
                        "image_fingerprint": fingerprints[key], "answers": doc["answers"],
                        "question_type": doc["question_type"], "answer_type": doc["answer_type"],
                        "prompt": doc["question"] + "\nWhen the provided information is insufficient, respond with 'Unanswerable'.\nAnswer the question using a single word or phrase.",
                    })
        if len(rows) != COUNT or len({row["question_id"] for row in rows}) != COUNT:
            raise ValueError("Wrong OK-VQA row count or duplicate question IDs")
        overlap = sorted({entry["image_fingerprint"] for entry in training}
                         & {row["image_fingerprint"] for row in rows})
        write_json(root / "training_overlap.json", {"overlap_images": len(overlap), "overlap": overlap})
        reject_overlap(training, rows)
        cohort = {"rows": rows, "smoke_indices": list(range(8)),
                  "independent_images": len({row["image_fingerprint"] for row in rows})}
        write_json(root / "cohort.json", cohort)
        pins[str(root / "cohort.json")] = file_sha(root / "cohort.json")
        templates, provenance = prior.templates()
        pins.update(provenance)
        for path in [*TASK_DIR.glob("*.py"), *TASK_DIR.glob("*.yaml")]:
            pins[str(path)] = file_sha(path)
        for cfg in templates.values():
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
            "templates": templates, "input_hashes": pins, "methods": list(METHODS),
            "count": COUNT, "task": TASK_NAME, "prerequisite": str(PREREQUISITE),
            "backbone": "Qwen2.5-VL-7B-Instruct", "no_training": True,
            "budget": "compressed methods use min(128,N), native dynamic resolution",
            "metric": "native OK-VQA VQA soft exact_match",
            "generation": {"greedy": True, "max_new_tokens": 16, "repetition_penalty": 1.05},
            "limits": [
                "This is full OK-VQA val2014, not VQAv2-lite or a development subset.",
                "EADP is the current Qwen adaptation, not a certified original-paper reproduction.",
                "Confidence intervals cluster repeated questions by image.",
            ],
        }
        write_json(root / "protocol.json", protocol)
        write_json(root / "manifest.json", {
            "status": "prepared", "protocol_sha256": file_sha(root / "protocol.json"),
            "total_full_cells": len(METHODS), "full_requests": COUNT * len(METHODS),
            "updated_utc": now(),
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
        raise ValueError("Source or evaluator changed")
    for path, checksum in protocol["input_hashes"].items():
        if file_sha(path) != checksum:
            raise ValueError(f"Pinned input changed: {path}")
    return protocol


def make_config(protocol, method, phase, run):
    cfg = copy.deepcopy(protocol["templates"][method])
    cfg["experiment"]["name"] = f"qwen25_okvqa_{phase}_{method}"
    cfg["eval"].update(tasks=[TASK_NAME], sample_limit=None)
    cfg["eval"]["args"]["include_path"] = str(TASK_DIR)
    cfg["data"] = {"dataset_id": "local_okvqa_val2014", "sample_split": SPLIT}
    cfg["output"] = {"run_dir": str(run), "manifest_path": str(run / "manifest.json")}
    return cfg


def _answers(value):
    if isinstance(value, str):
        try:
            value = ast.literal_eval(value)
        except (ValueError, SyntaxError):
            return None
    return value if isinstance(value, list) else None


def answers_match_logged_target(value, expected):
    """lmms-eval's native scorer normalizes doc['answers'] in place.

    The sample logger may see either the raw list or that exact normalized
    list. Preserve order and repetitions, and accept only those two forms.
    """
    from lmms_eval.tasks._task_utils.vqa_eval_metric import EvalAIAnswerProcessor
    actual = _answers(value)
    if actual is None or not all(isinstance(answer, str) for answer in actual):
        return False
    processor = EvalAIAnswerProcessor()
    return actual == expected or actual == [processor(answer) for answer in expected]


def check(root, method, phase, run):
    cohort = read(root / "cohort.json")
    expected = ([cohort["rows"][i] for i in cohort["smoke_indices"]]
                if phase == "smoke" else cohort["rows"])
    count = len(expected)
    samples = list(run.glob(f"**/*_samples_{TASK_NAME}.jsonl"))
    result_files = list(run.glob("**/*_results.json"))
    if len(samples) != 1 or len(result_files) != 1:
        raise ValueError("Missing or duplicate evaluator artifacts")
    rows = sorted((json.loads(line) for line in samples[0].read_text().splitlines()), key=lambda row: row["doc_id"])
    if len(rows) != count or {row["doc_id"] for row in rows} != set(range(count)):
        raise ValueError("Missing or duplicate predictions")
    for row, item in zip(rows, expected):
        if row["input"] != item["prompt"] or not answers_match_logged_target(row["target"], item["answers"]):
            raise ValueError("Question/answer alignment changed")
    score = sum(float(row["exact_match"]) for row in rows) / count
    native = read(result_files[0])["results"][TASK_NAME]
    keys = [key for key in native if key.split(",")[0] == "exact_match"]
    if len(keys) != 1 or not math.isfinite(score) or abs(float(native[keys[0]]) - score) > 1e-10:
        raise ValueError("Native aggregate mismatch")
    performance = [json.loads(line) for line in (run / "request_performance.jsonl").read_text().splitlines()]
    def doc_id(sample_id):
        fields = sample_id.split(":")
        if fields[0] != TASK_NAME or fields[1] != SPLIT:
            raise ValueError("Unexpected request identity")
        return int(fields[2])
    perf = {doc_id(row["sample_id"]): row for row in performance}
    if len(performance) != count or set(perf) != set(range(count)):
        raise ValueError("Missing performance rows")
    retained = [perf[i]["original_visual_tokens"] for i in range(count)]
    if method != "dense":
        traces = [json.loads(line) for line in (run / "compression_runtime.requests.jsonl").read_text().splitlines()]
        indexed = {doc_id(row["request"]["sample_id"]): row for row in traces}
        if len(traces) != count or set(indexed) != set(range(count)):
            raise ValueError("Missing compression traces")
        for index in range(count):
            activation, = indexed[index]["runtime"]["activations"]
            retained[index] = base.check_trace(activation, perf[index]["original_visual_tokens"], 28)
    checked = {
        "requests": count, "score": score, "samples": str(samples[0]), "samples_sha256": file_sha(samples[0]),
        "native_metrics": native,
        "mean_source_tokens": sum(row["original_visual_tokens"] for row in performance) / count,
        "mean_retained_tokens": sum(retained) / count,
        "generation_seconds": sum(row["generation_s"] for row in performance),
        "peak_allocated_bytes": max(row["peak_allocated_bytes"] or 0 for row in performance),
    }
    write_json(run / "protocol_checks.json", checked)
    return checked


def evaluate(root, protocol, method, phase, gpu):
    import yaml
    directory = root / "eval" / phase / method
    marker = directory / "complete.json"
    if marker.exists():
        saved = read(marker)
        if file_sha(saved["config"]) != saved["config_sha256"] or file_sha(saved["samples"]) != saved["samples_sha256"]:
            raise ValueError("Completed provenance changed")
        return
    stamp = time.time_ns()
    run = directory / f"attempt_{stamp}"
    config_path = root / "configs" / f"{phase}_{method}_{stamp}.yaml"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    write_text(config_path, yaml.safe_dump(make_config(protocol, method, phase, run), sort_keys=False))
    while gpu_free_mib().get(gpu, 0) < 46000:
        write_json(root / f"worker_gpu{gpu}.json", {"status": "waiting_for_memory", "updated_utc": now()})
        time.sleep(30)
    directory.mkdir(parents=True, exist_ok=True)
    log = directory / f"launch_{stamp}.log"
    env = base.child_env(root, "qwen25", gpu, phase, run)
    with log.open("x") as stream:
        child = subprocess.Popen([sys.executable, "-m", "mllm_comp.cli.main", "eval", "--config", str(config_path)],
                                 cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT)
        write_json(root / f"worker_gpu{gpu}.json", {
            "status": "running", "method": method, "phase": phase, "pid": child.pid,
            "log": str(log), "updated_utc": now(),
        })
        code = child.wait()
    if code:
        raise RuntimeError(f"Native eval failed ({code}); preserved {log}")
    checked = check(root, method, phase, run)
    write_json(marker, {**checked, "run": str(run), "config": str(config_path),
                       "config_sha256": file_sha(config_path), "finished_utc": now()})


def report(root, protocol):
    records = []
    lines = ["# Qwen2.5-VL-7B OK-VQA val2014 K128", "", "| Method | Soft accuracy |", "|---|---:|"]
    for method in METHODS:
        marker = root / "eval/full" / method / "complete.json"
        if marker.exists():
            item = read(marker)
            records.append({"method": method, **item})
            value = f'{item["score"] * 100:.2f}%'
        else:
            value = "pending"
        lines.append(f"| {method} | {value} |")
    write_json(root / "summary.json", {"complete_cells": len(records), "total_cells": len(METHODS), "results": records})
    write_text(root / "report.md", "\n".join(lines + ["", *protocol["limits"]]) + "\n")


def run_phase(root, protocol, gpus, phase):
    queue, stop = Queue(), Event()
    for method in METHODS:
        queue.put(method)
    def worker(gpu):
        try:
            while not stop.is_set():
                try:
                    method = queue.get_nowait()
                except Empty:
                    return
                evaluate(root, protocol, method, phase, gpu)
                report(root, protocol)
        except BaseException as error:
            stop.set()
            write_json(root / f"worker_gpu{gpu}.json", {"status": "failed", "error": str(error),
                       "traceback": traceback.format_exc(), "updated_utc": now()})
            raise
    with ThreadPoolExecutor(max_workers=len(gpus)) as pool:
        jobs = [pool.submit(worker, gpu) for gpu in gpus]
        for job in jobs:
            job.result()


def final_statistics(root):
    cohort = read(root / "cohort.json")
    def rows(method):
        marker = read(root / "eval/full" / method / "complete.json")
        return sorted((json.loads(line) for line in Path(marker["samples"]).read_text().splitlines()),
                      key=lambda row: row["doc_id"])
    baseline, comparisons = rows("dense"), []
    for method in METHODS[1:]:
        actual = rows(method)
        comparisons.append({"method": method, **base.bootstrap(
            [float(a["exact_match"]) - float(b["exact_match"]) for a, b in zip(actual, baseline)],
            [row["image_fingerprint"] for row in cohort["rows"]])})
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
            while True:
                prerequisite = read(protocol["prerequisite"])
                if prerequisite["status"] == "complete":
                    break
                if prerequisite["status"] == "failed":
                    raise RuntimeError("Prerequisite POPE/RefCOCO+ queue failed")
                if not _pid_alive(prerequisite.get("pid")):
                    raise RuntimeError(
                        "Prerequisite POPE/RefCOCO+ queue is stale: non-terminal status "
                        f"{prerequisite.get('status')!r} with dead pid "
                        f"{prerequisite.get('pid')!r}; recover it before waiting"
                    )
                stage("queued", prerequisite_status=prerequisite["status"])
                time.sleep(30)
            stage("smoke")
            run_phase(root, protocol, gpus, "smoke")
            stage("full")
            run_phase(root, protocol, gpus, "full")
            report(root, protocol)
            if read(root / "summary.json")["complete_cells"] != len(METHODS):
                raise ValueError("Full OK-VQA matrix incomplete")
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
