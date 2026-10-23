"""Durable isolated 3B geometry ablation, queued behind the pinned 7B run."""
import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from mllm_comp.analysis.common.anchor_distance_ablation import specifications
from mllm_comp.analysis.common.evidence_interaction import bootstrap
from mllm_comp.analysis.common.learned_hybrid_runtime import CHECKPOINT, REVISION
from mllm_comp.analysis.probes.anchor_distance_ablation import AnchorDistanceAblation, Session, implementation_hash
from mllm_comp.core.serialization import atomic_write_json


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def select_cohort_entries(entries):
    """250 images per task, selected without answers or model outputs.

    Keep the existing 100 development images per task and hash-select 150
    additional images per task from the old held-out pool. The latter pool
    was evaluated in earlier experiments, so it is replication, not a fresh
    independent discovery confirmation.
    """
    selected = []
    tasks = sorted({e["task"] for e in entries})
    if len(tasks) != 2:
        raise ValueError("Expected ChartQA and TextVQA")
    for task in tasks:
        development = [e for e in entries if e["task"] == task and e["split"] == "development"]
        candidates = [e for e in entries if e["task"] == task and e["split"] == "confirmation"]
        if len(development) != 100 or len(candidates) < 150:
            raise ValueError("Need 100 development and at least 150 additional images per task")
        candidates.sort(key=lambda e:(hashlib.sha256(("anchor-geometry-n500-seed42:"+e["image_fingerprint"]).encode()).hexdigest(), e["sample_id"]))
        selected.extend(development+candidates[:150])
    if len(selected) != 500 or len({e["image_fingerprint"] for e in selected}) != 500:
        raise ValueError("Expected exactly 500 globally unique images")
    return selected


def expected_entries(root, stage):
    entries = json.loads((root/"cohort.json").read_text())["entries"]
    if stage == "all":
        return entries
    if stage == "smoke":
        return [e for e in entries if e.get("smoke", False)]
    return [e for e in entries if e["split"] == stage]


def prepare(root, source):
    import yaml
    root.mkdir(parents=True, exist_ok=True)
    if (root/"protocol.json").exists():
        protocol = json.loads((root/"protocol.json").read_text())
        if protocol["implementation_hash"] != implementation_hash() or protocol["source_7b_run"] != str(source):
            raise ValueError("Do not reuse a run directory for a different implementation")
        return
    cohort = json.loads((source/"cohort.json").read_text())
    selected = select_cohort_entries(cohort["entries"])
    cohort["entries"] = selected
    cohort["counts"] = {"total_images": 500, "per_task": 250, "development": 200, "additional_replication": 300}
    cohort["scope_note"] = "200 previous development images plus 300 answer-blind hash-selected previous held-out images. Exploratory/replication, not new independent confirmation."
    cohort["geometry_selection"] = {"seed": 42, "reads_answers_or_predictions": False, "rule": "100 existing development + 150 sha256-ranked held-out images per task"}
    atomic_write_json(root/"cohort.json", cohort)
    configs = root/"configs"; configs.mkdir(exist_ok=True)
    for stage in ["smoke", "development", "confirmation"]:
        for shard in [0, 1]:
            raw = yaml.safe_load((source/"configs"/f"{stage}_{shard}.yaml").read_text())
            raw["experiment"]["name"] = f"anchor_distance_3b_{stage}_{shard}"
            raw["backbone"]["model_path"] = CHECKPOINT
            raw["backbone"]["model_revision"] = REVISION
            raw["analysis"]["sampling"]["manifest_path"] = str(root/"cohort.json")
            # Direct Session API avoids modifying the registry frozen by 7B.
            raw["analysis"]["probes"] = []
            (configs/f"{stage}_{shard}.yaml").write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
    protocol = {"checkpoint": CHECKPOINT, "revision": REVISION, "source_7b_run": str(source),
        "implementation_hash": implementation_hash(), "budget": 128, "uses_mlp": False, "training": False,
        "model_dtype": "bfloat16", "geometry_dtype": "float32", "attention": "eager",
        "n_images": 500, "images_per_task": 250, "development_images": 200, "additional_replication_images": 300,
        "random_seeds": [11, 23, 47],
        "conditions": ["dense"]+[asdict(s) for s in specifications()],
        "aggregation": "FP32 means of ORIGINAL unnormalized features, BF16 outputs in all arms",
        "positions": "same rounded 3-axis cluster centroid rule; text unchanged",
        "first_anchor": "source token 0 for all arms, including conditional kmeans++ initialization",
        "kmeans_definition": "ordinary Euclidean Lloyd on unit input features; means are NOT renormalized; max 10 updates; deterministic empty-cluster split",
        "generation": {"do_sample": False, "repetition_penalty": 1., "chartqa_max_new": 16, "textvqa_max_new": 32},
        "inference_inputs": "Gold-free prompt only; gold answer appended solely for separate NLL pass",
        "statistics": "10,000 image-level paired bootstrap; average three initialization seeds within each image, not 3x the sample count",
        "interpretation": "Exploratory/replication on previously evaluated images; not new independent confirmation. Report 500 pooled and 200/300 split results. Reconstruction error is explanatory, native generation score is primary. No novelty claim about Euclidean distance.",
        "timing_caveat": "Eager diagnostic implementation with statistics, no deployment speed claim; concurrent jobs can contend for GPU.",
        "created_utc": datetime.now(timezone.utc).isoformat()}
    protocol["artifact_hashes"] = {str(p.relative_to(root)): sha(p) for p in [root/"cohort.json"]+sorted(configs.glob("*.yaml"))}
    atomic_write_json(root/"protocol.json", protocol)


def validate(root):
    protocol = json.loads((root/"protocol.json").read_text())
    if protocol["implementation_hash"] != implementation_hash():
        raise ValueError("Frozen source changed; use a new run directory")
    if any(sha(root/name) != value for name, value in protocol["artifact_hashes"].items()):
        raise ValueError("Frozen cohort/config changed")
    return protocol


def worker(root, stage, device):
    import torch
    from mllm_comp.analysis.common.learned_hybrid_runtime import load_runtime
    from mllm_comp.config.loader import load_config
    from mllm_comp.registry.integrations import get_integration
    validate(root)
    torch.set_num_threads(4)
    torch.cuda.set_device(device)
    torch.manual_seed(42)
    torch.backends.cuda.matmul.allow_tf32 = False
    bundle = load_runtime(device)
    expected = expected_entries(root, stage)
    completed = []
    try:
        adapter = get_integration("lmms_eval")()
        for shard in [0, 1]:
            config = load_config(root/"configs"/f"{stage}_{shard}.yaml")
            session = Session(AnchorDistanceAblation(), bundle["facade"], bundle["runtime"],
                {"artifact_root": str(root), "cohort_path": str(root/"cohort.json"), "stage": stage},
                "pre_compression", str(root/"lifecycle"/f"{stage}_{shard}"))
            for sample in adapter.iter_analysis_samples(config=config, backbone=bundle["facade"],
                    runtime=bundle["runtime"], sample_limit=None, requires_teacher_forcing=True):
                with torch.inference_mode():
                    session.process(sample)
            session.finish()
            completed.extend(session.completed)
        if {e["sample_id"] for e in expected} != {e["sample_id"] for e in completed} or len(expected) != len(completed):
            raise ValueError("Missing/duplicate images in geometry ablation")
        atomic_write_json(root/f"{stage}_worker.json", {"status": "complete", "n_images": len(completed),
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(), "implementation_hash": implementation_hash()})
    finally:
        bundle["runtime"].close()


def report(root, stage):
    import statistics
    expected = expected_entries(root, stage)
    stages = ["development", "confirmation"] if stage == "all" else [stage]
    summaries = [json.loads(p.read_text()) for split in stages for p in (root/"samples"/split).glob("*/summary.json")]
    if {s["sample_id"] for s in summaries} != {e["sample_id"] for e in expected} or len(summaries) != len(expected):
        raise ValueError("Incomplete geometry report")
    names = ["dense"]+[s.name for s in specifications()]
    grouped = {}
    for summary in summaries:
        rows = [json.loads(line) for line in Path(summary["condition_path"]).read_text().splitlines()]
        by_name = {r["variant"]: r for r in rows}
        if set(by_name) != set(names) or len(rows) != len(names):
            raise ValueError("Incomplete condition set")
        if any(r["gold_answer_in_decode_context"] or (r["variant"] != "dense" and r["retained_visual_tokens"] != 128) for r in rows):
            raise ValueError("Gold leakage or budget mismatch")
        grouped.setdefault(summary["task"], []).append(by_name)
    tables = {}
    for task, images in grouped.items():
        table = {}
        for name in names:
            values = [i[name] for i in images]
            table[name] = {"n_images": len(values), "mean_score": statistics.mean(v["primary_task_score"] for v in values),
                "mean_nll": statistics.mean(v["answer_nll"] for v in values),
                "median_decoder_prefill_gpu_ms": statistics.median(v["decoder_prefill_gpu_ms"] for v in values),
                "mean_visual_tokens": statistics.mean(v["retained_visual_tokens"] for v in values),
                "score_minus_cosine": bootstrap([i[name]["primary_task_score"]-i["cosine_farthest"]["primary_task_score"] for i in images])}
            if name != "dense":
                table[name].update({"median_diagnostic_compression_ms": statistics.median(v["diagnostic_compression_ms"] for v in values),
                    "raw_relative_sse": statistics.mean(v["geometry"]["raw_relative_sse"] for v in values),
                    "unit_sse_per_token": statistics.mean(v["geometry"]["unit_sse_per_token"] for v in values),
                    "anchor_overlap_with_cosine": statistics.mean(v["comparison_to_cosine"]["anchor_overlap_fraction"] for v in values),
                    "partition_identical_fraction": statistics.mean(v["comparison_to_cosine"]["partition_identical"] for v in values)})
        for prefix in ["unit_kmeanspp", "unit_kmeans10"]:
            seed_names = [f"{prefix}_s{s}" for s in (11, 23, 47)]
            per_image = [statistics.mean(i[n]["primary_task_score"] for n in seed_names) for i in images]
            table[prefix+"_seed_mean"] = {"n_images": len(images), "mean_score": statistics.mean(per_image),
                "seed_score_sd": statistics.stdev(table[n]["mean_score"] for n in seed_names),
                "score_minus_cosine": bootstrap([v-i["cosine_farthest"]["primary_task_score"] for v, i in zip(per_image, images)])}
        tables[task] = table
    result = {"status": "complete", "stage": stage, "n_images": len(summaries), "tables": tables,
        "median_image_elapsed_s": statistics.median(s["elapsed_s"] for s in summaries),
        "total_image_elapsed_s": sum(s["elapsed_s"] for s in summaries),
        "interpretation": "Exploratory/replication paired image-level CIs on previously evaluated images, not a fresh independent confirmation. Seeds are not independent images. Diagnostic timing is not deployment latency."}
    atomic_write_json(root/f"{stage}_report.json", result)
    lines = [f"# 3B anchor geometry ablation: {stage}", "", result["interpretation"], "",
        "| Task | Variant | Native score | Delta vs cosine (pp), 95% CI |", "|---|---|---:|---:|"]
    for task, table in tables.items():
        for name, stats in table.items():
            effect = stats["score_minus_cosine"]
            lines.append(f"| {task} | {name} | {100*stats['mean_score']:.2f}% | {100*effect['mean']:+.2f} [{100*effect['ci95'][0]:+.2f}, {100*effect['ci95'][1]:+.2f}] |")
    (root/f"{stage}_report.md").write_text("\n".join(lines)+"\n", encoding="utf-8")
    return result


def run(root, source, device):
    with (root/"runner.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        validate(root)
        state = {"status": "queued", "waiting_for": str(source/"run_manifest.json"), "pid": os.getpid(),
            "checkpoint": CHECKPOINT, "revision": REVISION, "implementation_hash": implementation_hash(),
            "stages": {}, "started_utc": datetime.now(timezone.utc).isoformat()}
        process = None
        atomic_write_json(root/"run_manifest.json", state)
        try:
            while True:
                parent = json.loads((source/"run_manifest.json").read_text())
                if parent["status"] == "complete":
                    break
                if parent["status"] == "failed":
                    state.update(status="blocked", error="7B queue failed; not competing with or modifying that run")
                    atomic_write_json(root/"run_manifest.json", state)
                    return
                time.sleep(10)
            validate(root)
            state.update(status="running", waiting_for=None)
            for stage in ["smoke", "development", "confirmation"]:
                log = root/f"{stage}.log"
                with log.open("a") as stream:
                    command = [sys.executable, str(Path(__file__)), "--root", str(root), "--source-run", str(source),
                        "--device", device, "--worker", stage]
                    process = subprocess.Popen(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True,
                        env={**os.environ, "PYTHONUNBUFFERED": "1", "OMP_NUM_THREADS": "4", "TOKENIZERS_PARALLELISM": "false"})
                    state["active_stage"] = stage
                    state["stages"][stage] = {"status": "running", "pid": process.pid, "log": str(log)}
                    atomic_write_json(root/"run_manifest.json", state)
                    if process.wait() != 0:
                        raise RuntimeError(f"{stage} failed; inspect {log}")
                validate(root)
                result = report(root, stage)
                state["stages"][stage]["status"] = "complete"
                if stage == "smoke":
                    state["estimated_500_image_minutes_from_smoke"] = result["total_image_elapsed_s"]/result["n_images"]*500/60
                atomic_write_json(root/"run_manifest.json", state)
            report(root, "all")
            state.update(status="complete", completed_utc=datetime.now(timezone.utc).isoformat())
        except BaseException as exc:
            if process and process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
                process.wait()
            state.update(status="failed", error=str(exc), traceback=traceback.format_exc())
            raise
        finally:
            atomic_write_json(root/"run_manifest.json", state)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--source-run", type=Path, required=True)
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--worker", choices=["smoke", "development", "confirmation"])
    parser.add_argument("--launch", action="store_true")
    args = parser.parse_args()
    root, source = args.root.resolve(), args.source_run.resolve()
    if args.worker:
        worker(root, args.worker, args.device)
    else:
        prepare(root, source)
        if args.launch:
            with (root/"supervisor.log").open("a") as stream:
                process = subprocess.Popen([sys.executable, str(Path(__file__)), "--root", str(root),
                    "--source-run", str(source), "--device", args.device], cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT,
                    start_new_session=True, env={**os.environ, "PYTHONUNBUFFERED": "1"})
            print(json.dumps({"pid": process.pid, "root": str(root), "status": "queued_after_7b"}))
        else:
            run(root, source, args.device)
