"""Pinned 7B full ChartQA -> AI2D evaluation, without retraining.

GPU 0 can begin immediately; GPU 1 waits for the existing 500-image geometry
queue. All questions stay in the full result, with train-image-clean results
reported separately and paired uncertainty clustered by image content.
"""
import argparse
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from datetime import datetime, timezone
import fcntl
import hashlib
from io import BytesIO
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time
import traceback

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from mllm_comp.analysis.common.anchor7b_runtime import (
    implementation_hash as runtime_hash, file_sha, validate_identity, load_runtime,
    load_adapter, anchor_base, compact_anchor, explicit_generate,
)
from mllm_comp.analysis.common.compression_training_check import evaluate_state
from mllm_comp.analysis.common.condition_ledger import append_json, records
from mllm_comp.analysis.common.evidence_interaction import digest
from mllm_comp.analysis.common.visual_token_phenomena import media_fingerprint
from mllm_comp.core.serialization import atomic_write_json
from scripts.train.qwen25_vl.train_learned_hybrid import prepare_example
from scripts.eval.lmms_eval.run_ai2d_compression_check import upstream_score

METHODS = ["dense", "feature_anchor_adapter"]
DATA_ROOT = Path("/home/YinPeiran/datasets/lmms_eval_datasets")
DATASETS = {"chartqa": (2500, "ChartQA/data/test-*.parquet"), "ai2d": (3088, "ai2d/data/test-*.parquet")}


def code_hash():
    import inspect
    from lmms_eval.tasks.chartqa import utils as chart
    from lmms_eval.tasks.textvqa import utils as text
    from lmms_eval.tasks.ai2d import utils as ai2d
    from mllm_comp.integrations.lmms_eval.adapter import _analysis_target
    paths = [Path(__file__), Path(inspect.getsourcefile(prepare_example)),
             Path(inspect.getsourcefile(upstream_score)), Path(inspect.getsourcefile(_analysis_target)),
             Path(inspect.getsourcefile(chart)), Path(inspect.getsourcefile(text)), Path(inspect.getsourcefile(ai2d)),
             ROOT/"mllm_comp/backbones/qwen25_vl/provider_bridge.py"]
    h = hashlib.sha256(runtime_hash().encode())
    for p in paths:
        h.update(p.name.encode()); h.update(p.read_bytes())
    return h.hexdigest()


def prompt_target(dataset, doc):
    from lmms_eval.tasks.chartqa import utils as chart
    from lmms_eval.tasks.textvqa import utils as text
    from lmms_eval.tasks.ai2d import utils as ai2d
    from mllm_comp.integrations.lmms_eval.adapter import _analysis_target
    if dataset == "chartqa":
        prompt = chart.chartqa_doc_to_text(doc, {"pre_prompt": "", "post_prompt": "\nAnswer the question with a single word."})
        answer = doc["answer"]
        target = str(answer[0] if isinstance(answer, list) else answer)
    elif dataset == "textvqa":
        prompt = text.textvqa_doc_to_text(doc, {"pre_prompt": "", "post_prompt": "\nAnswer the question using a single word or phrase."})
        target, _ = _analysis_target(None, doc, task_name="textvqa")
        if target is None or len(doc["answers"]) != 10:
            raise ValueError("TextVQA needs ten native references and a deterministic NLL target")
    elif dataset == "ai2d":
        prompt = ai2d.ai2d_doc_to_text(doc, {"pre_prompt": "", "post_prompt": "\nAnswer with the option's letter from the given choices directly.", "prompt_format": "mcq"})
        target = chr(65+int(doc["answer"]))
    else:
        raise ValueError(dataset)
    return prompt, target


def score_prediction(dataset, prediction, doc):
    from lmms_eval.tasks.chartqa import utils as chart
    from lmms_eval.tasks.textvqa import utils as text
    if dataset == "chartqa":
        metrics = chart.chartqa_process_results(deepcopy(doc), [prediction])
        answers = doc["answer"] if isinstance(doc["answer"], list) else [doc["answer"]]
        strict = float(any(prediction.strip().casefold() == str(a).strip().casefold() for a in answers))
        return float(metrics["relaxed_overall"]), metrics, strict, prediction
    if dataset == "textvqa":
        metrics = text.textvqa_process_results(deepcopy(doc), [prediction])
        return float(metrics["exact_match"]), metrics, None, prediction
    score, filtered = upstream_score(prediction, doc)
    strict = float(prediction.strip().upper() == chr(65+int(doc["answer"])))
    return score, {"exact_match": score}, strict, filtered


def clustered_delta(groups, method, reference="dense", repeats=10000):
    """Question-weighted paired effect, bootstrap resamples entire images."""
    import numpy as np
    counts = np.asarray([len(questions) for questions in groups], dtype=float)
    differences = np.asarray([sum(q[method]["primary_task_score"]-q[reference]["primary_task_score"] for q in questions)
                              for questions in groups], dtype=float)
    if not len(counts):
        return None
    rng = np.random.default_rng(42)
    draws = []
    for start in range(0, repeats, 200):
        indices = rng.integers(0, len(counts), size=(min(200, repeats-start), len(counts)))
        draws.extend((differences[indices].sum(1)/counts[indices].sum(1)).tolist())
    return {"mean": float(differences.sum()/counts.sum()), "ci95": [float(x) for x in np.quantile(draws, [.025, .975])],
            "n_images": len(counts), "n_questions": int(counts.sum()), "estimand": "question-weighted score difference; image-cluster bootstrap"}


def compatible_reuse(protocol, old):
    for field in ["checkpoint", "revision", "base_identity_sha256", "adapter_sha256", "adapter_config_sha256",
                  "training_manifest_sha256", "budget", "dtype", "attention", "seed", "processor"]:
        if protocol[field] != old[field]:
            raise ValueError(f"Cannot reuse predictions with changed {field}")
    if any(old["generation"].get(k) != v for k,v in protocol["generation"].items()):
        raise ValueError("Cannot reuse predictions with changed generation")


def reuse_conditions(root, protocol, dataset, stage, entry, config_hash, ledger, saved):
    """Copy immutable completed predictions, never rerun or relabel old scores."""
    reuse = protocol.get("reuse_run")
    if not reuse:
        return 0
    source = Path(reuse)/dataset/"samples"/stage/entry["sample_id"].replace(":", "_")/"conditions.jsonl"
    if not source.exists():
        return 0
    old = json.loads((Path(reuse)/"protocol.json").read_text())
    compatible_reuse(protocol, old)
    old_hash = digest(json.dumps(old, sort_keys=True)+dataset+stage)
    raw = source.read_bytes()
    added, observed = 0, set()
    for line in raw.splitlines(keepends=True):
        if not line.endswith(b"\n"):
            break  # stopped write: preserve source, do not import a partial row
        row = json.loads(line)
        if row["variant"] not in METHODS:
            continue
        key = (row["sample_id"], row["variant"])
        if key in observed:
            raise ValueError("Duplicate source prediction")
        observed.add(key)
        if row["sample_id"] != entry["sample_id"] or row["image_fingerprint"] != entry["image_fingerprint"]:
            raise ValueError("Reused prediction image mismatch")
        if row["config_hash"] != old_hash or row["status"] != "completed" or row["gold_answer_in_decode_context"]:
            raise ValueError("Invalid source result provenance")
        if row["base_identity_sha256"] != protocol["base_identity_sha256"] or row["repetition_penalty"] != 1.:
            raise ValueError("Reused checkpoint/generation mismatch")
        if row["variant"] == "feature_anchor_adapter" and row["adapter_sha256"] != protocol["adapter_sha256"]:
            raise ValueError("Reused MLP checkpoint mismatch")
        if row["variant"] != "dense" and row["retained_visual_tokens"] != 128:
            raise ValueError("Reused result has wrong budget")
        if key not in saved:
            copied = {**row, "config_hash": config_hash, "reuse_provenance": {"source": str(source),
                "source_sha256": hashlib.sha256(raw).hexdigest(), "original_config_hash": old_hash,
                "reason": "Same predictions retained after removing unused arms/datasets; no model or generation change"}}
            append_json(ledger, copied); saved[key] = copied; added += 1
    return added


def prepare(root, source, geometry, reuse=None):
    import pyarrow.parquet as pq
    from PIL import Image
    root.mkdir(parents=True, exist_ok=True)
    if (root/"protocol.json").exists():
        return validate(root)
    identity = json.loads((source/"model_identity.json").read_text())
    validate_identity(identity)
    source_status = json.loads((source/"run_manifest.json").read_text())
    if source_status["status"] != "complete":
        raise ValueError("Require completed 7B training/evaluation before full expansion")
    adapter = source/"model/checkpoints/step_000320"
    train_hashes = {e["image_fingerprint"] for e in json.loads((source/"training_samples.json").read_text())["entries"]}
    datasets = {}
    for dataset, (expected, pattern) in DATASETS.items():
        folder = root/dataset; folder.mkdir(exist_ok=True)
        files = sorted(DATA_ROOT.glob(pattern))
        if not files:
            raise ValueError(f"No local {dataset} files")
        entries, first = [], {}
        if reuse:
            prior = json.loads((reuse/"protocol.json").read_text())["datasets"][dataset]
            if set(prior["data_files"]) != {str(p) for p in files} or file_sha(reuse/dataset/"samples.json") != prior["samples_sha256"]:
                raise ValueError("Source full-dataset manifest is incompatible")
            if any(file_sha(p) != prior["data_files"][str(p)] for p in files):
                raise ValueError("Dataset changed since original full manifest")
            entries = json.loads((reuse/dataset/"samples.json").read_text())["entries"]
            for e in entries:
                first.setdefault(e["image_fingerprint"], e["doc_id"])
                if e["training_image_overlap"] != (e["image_fingerprint"] in train_hashes):
                    raise ValueError("Reused training-overlap flags differ")
        else:
            for path in files:
                index = 0
                for batch in pq.ParquetFile(path).iter_batches(batch_size=16):
                    for doc in batch.to_pylist():
                        image = doc["image"]
                        with Image.open(BytesIO(image["bytes"]) if image.get("bytes") else image["path"]) as loaded:
                            rgb = loaded.convert("RGB")
                            fingerprint = media_fingerprint([rgb])
                            rgb.close()
                        first.setdefault(fingerprint, len(entries))
                        entries.append({"sample_id": f"{dataset}:{len(entries)}", "doc_id": len(entries), "path": str(path), "row": index,
                            "image_fingerprint": fingerprint, "shard": int(digest(fingerprint)[:8], 16)%2,
                            "training_image_overlap": fingerprint in train_hashes})
                        index += 1
        if len(entries) != expected:
            raise ValueError(f"{dataset}: expected {expected}, found {len(entries)}")
        smoke = set(sorted(first, key=lambda f:digest(dataset+":7b-full-smoke:"+f))[:16])
        for e in entries:
            e["smoke"] = e["image_fingerprint"] in smoke and e["doc_id"] == first[e["image_fingerprint"]]
        atomic_write_json(folder/"samples.json", {"entries": entries, "sampling": "All questions, no answer/performance filtering"})
        datasets[dataset] = {"n_questions": len(entries), "n_images": len(first),
            "training_overlap_questions": sum(e["training_image_overlap"] for e in entries),
            "training_overlap_images": len(set(first)&train_hashes),
            "samples_sha256": file_sha(folder/"samples.json"), "data_files": {str(p): file_sha(p) for p in files}}
        print(json.dumps({"prepared": dataset, **{k:v for k,v in datasets[dataset].items() if k != "data_files"}}), flush=True)
    atomic_write_json(root/"model_identity.json", identity)
    protocol = {"checkpoint": identity["checkpoint"], "revision": identity["revision"], "base_identity_sha256": identity["identity_sha256"],
        "source_training_run": str(source), "geometry_queue": str(geometry), "adapter_path": str(adapter),
        "adapter_sha256": file_sha(adapter/"projector.safetensors"), "adapter_config_sha256": file_sha(adapter/"projector_config.json"),
        "training_manifest_sha256": file_sha(source/"training_samples.json"), "implementation_hash": code_hash(),
        "identity_file_sha256": file_sha(root/"model_identity.json"), "methods": METHODS, "budget": 128,
        "training": False, "dtype": "bfloat16", "attention": "eager", "seed": 42,
        "processor": {"use_fast": False, "min_pixels": 200704, "max_pixels": 1605632},
        "generation": {"do_sample": False, "repetition_penalty": 1., "chartqa_max_new": 16, "ai2d_max_new": 16},
        "datasets": datasets, "reuse_run": str(reuse) if reuse else None,
        "comparison_scope": "Only dense and trained MLP K128; no untrained anchor evaluation, TextVQA or PruMerge+ in this full run",
        "primary_subset": "Same train-image-content-overlap-free questions for every method; also report complete splits",
        "uncertainty": "10,000 paired image-cluster bootstrap, retains question-weighted estimand",
        "timing_caveat": "Diagnostic teacher-NLL plus cached generation, not optimized deployment speed benchmark",
        "created_utc": datetime.now(timezone.utc).isoformat()}
    if reuse:
        previous = json.loads((reuse/"protocol.json").read_text())
        compatible_reuse(protocol, previous)
        protocol["reuse_protocol_sha256"] = file_sha(reuse/"protocol.json")
    atomic_write_json(root/"protocol.json", protocol)
    return protocol


def validate(root):
    protocol = json.loads((root/"protocol.json").read_text())
    if protocol["implementation_hash"] != code_hash():
        raise ValueError("Frozen full-evaluation source changed")
    if protocol["identity_file_sha256"] != file_sha(root/"model_identity.json"):
        raise ValueError("Base identity file changed")
    validate_identity(json.loads((root/"model_identity.json").read_text()))
    path = Path(protocol["adapter_path"])
    if file_sha(path/"projector.safetensors") != protocol["adapter_sha256"] or file_sha(path/"projector_config.json") != protocol["adapter_config_sha256"]:
        raise ValueError("Trained adapter changed")
    for dataset, spec in protocol["datasets"].items():
        if file_sha(root/dataset/"samples.json") != spec["samples_sha256"]:
            raise ValueError("Frozen full sample manifest changed")
    if protocol.get("reuse_run") and file_sha(Path(protocol["reuse_run"])/"protocol.json") != protocol["reuse_protocol_sha256"]:
        raise ValueError("Source reuse protocol changed")
    return protocol


def worker(root, dataset, stage, shard, device):
    import torch
    import pyarrow.parquet as pq
    torch.cuda.set_device(device)
    torch.set_num_threads(4)
    torch.manual_seed(42)
    protocol = validate(root)
    identity = json.loads((root/"model_identity.json").read_text())
    folder = root/dataset
    entries = [e for e in json.loads((folder/"samples.json").read_text())["entries"]
               if e["shard"] == shard and (stage == "full" or e["smoke"])]
    by_path = defaultdict(dict)
    for e in entries:
        by_path[e["path"]][e["row"]] = e
    config_hash = digest(json.dumps(protocol, sort_keys=True)+dataset+stage)
    status_path = folder/f"{stage}_{shard}_manifest.json"
    status = {"status": "loading", "dataset": dataset, "stage": stage, "device": device, "shard": shard,
        "expected_questions": len(entries), "completed_questions": 0, "reused_conditions": 0, "config_hash": config_hash,
        "started_utc": datetime.now(timezone.utc).isoformat()}
    atomic_write_json(status_path, status)
    bundle = None
    try:
        bundle = load_runtime(identity, device)
        module = load_adapter(protocol["adapter_path"], identity, device).eval()
        status["status"] = "running"
        atomic_write_json(status_path, status)
        for path, wanted in by_path.items():
            if file_sha(path) != protocol["datasets"][dataset]["data_files"][path]:
                raise ValueError("Local dataset bytes changed")
            index = 0
            for batch in pq.ParquetFile(path).iter_batches(batch_size=8):
                for doc in batch.to_pylist():
                    entry = wanted.get(index); index += 1
                    if entry is None:
                        continue
                    sample_dir = folder/"samples"/stage/entry["sample_id"].replace(":", "_")
                    sample_dir.mkdir(parents=True, exist_ok=True)
                    ledger = sample_dir/"conditions.jsonl"
                    saved = records(ledger, config_hash)
                    status["reused_conditions"] += reuse_conditions(root, protocol, dataset, stage, entry, config_hash, ledger, saved)
                    if len(saved) != len(METHODS):
                        question, target = prompt_target(dataset, doc)
                        with torch.inference_mode():
                            state = prepare_example(bundle, {**entry, "image": doc["image"], "question": question, "target": target})
                        with torch.inference_mode():
                            base = anchor_base(state)
                            for name in METHODS:
                                key = (entry["sample_id"], name)
                                if key in saved:
                                    continue
                                started = time.perf_counter()
                                output, compact, trace = None, None, None
                                if name == "feature_anchor_adapter":
                                    with torch.autocast("cuda", dtype=torch.bfloat16):
                                        output = module(base)
                                    compact = compact_anchor(state, output)
                                    trace = {k:output[k] for k in ["indices", "members", "position_policy", "partition_hash"]}
                                max_new = 16
                                result = evaluate_state(bundle, state, compact, max_new=max_new, native_check=False)
                                if stage == "smoke" and name == "dense":
                                    if explicit_generate(bundle, state, max_new=max_new) != result["generated_token_ids"]:
                                        raise ValueError("7B native/cached dense generation mismatch")
                                    result["native_generation_identity_passed"] = True
                                selected = state if compact is None else compact
                                expected_decode = int(selected["positions"][:, :, :selected["prefix"]].max())+1
                                if result["first_appended_token_position"] != expected_decode:
                                    raise ValueError("Decode M-RoPE position mismatch")
                                if name != "dense" and (len(output["indices"]) != 128 or result["retained_visual_tokens"] != 128):
                                    raise ValueError("Compressed visual-token budget is not exactly 128")
                                prediction = result["prediction"]
                                score, metrics, strict, filtered = score_prediction(dataset, prediction, doc)
                                if not 0 <= score <= 1:
                                    raise ValueError("Native score out of range")
                                row = {**result, **entry, "dataset": dataset, "variant": name, "config_hash": config_hash,
                                    "status": "completed", "checkpoint": identity["checkpoint"], "revision": identity["revision"],
                                    "base_identity_sha256": identity["identity_sha256"], "repetition_penalty": 1.,
                                    "adapter_sha256": protocol["adapter_sha256"] if name == "feature_anchor_adapter" else None,
                                    "question": question, "target": target, "references": doc.get("answers", doc.get("answer")),
                                    "primary_task_score": score, "native_task_metrics": metrics, "strict_score": strict,
                                    "filtered_prediction": filtered, "compression_trace": trace, "elapsed_s": time.perf_counter()-started}
                                append_json(ledger, row); saved[key] = row
                        del state, base, output, compact
                    if {k[1] for k in saved} != set(METHODS) or len(saved) != len(METHODS):
                        raise ValueError("Incomplete or duplicate method conditions")
                    atomic_write_json(sample_dir/"summary.json", {**entry, "status": "completed", "condition_count": len(saved), "config_hash": config_hash})
                    status["completed_questions"] += 1
                    atomic_write_json(status_path, status)
                    print(json.dumps({"dataset": dataset, "stage": stage, "shard": shard,
                        "completed": status["completed_questions"], "expected": len(entries)}), flush=True)
        if status["completed_questions"] != len(entries):
            raise ValueError("Some full evaluation questions were not found")
        status.update(status="complete", peak_allocated_bytes=torch.cuda.max_memory_allocated())
    except BaseException as exc:
        status.update(status="failed", error=str(exc), traceback=traceback.format_exc())
        raise
    finally:
        atomic_write_json(status_path, status)
        if bundle:
            bundle["runtime"].close()


def report(root, dataset, stage):
    protocol = validate(root)
    folder = root/dataset
    entries = [e for e in json.loads((folder/"samples.json").read_text())["entries"] if stage == "full" or e["smoke"]]
    groups = {"all": defaultdict(list), "train_overlap_free": defaultdict(list)}
    for e in entries:
        sample_dir = folder/"samples"/stage/e["sample_id"].replace(":", "_")
        summary = json.loads((sample_dir/"summary.json").read_text())
        raw = [json.loads(s) for s in (sample_dir/"conditions.jsonl").read_text().splitlines()]
        rows = {r["variant"]:r for r in raw}
        if summary["condition_count"] != len(METHODS) or len(raw) != len(METHODS) or set(rows) != set(METHODS):
            raise ValueError("Incomplete question cannot enter final report")
        for method, row in rows.items():
            if row["image_fingerprint"] != e["image_fingerprint"] or row["gold_answer_in_decode_context"] or row["base_identity_sha256"] != protocol["base_identity_sha256"]:
                raise ValueError("Mixed identity, image or gold leakage")
            if method != "dense" and row["retained_visual_tokens"] != 128:
                raise ValueError("Unequal compressed budget")
        if stage == "smoke" and not rows["dense"].get("native_generation_identity_passed"):
            raise ValueError("Native generation smoke gate missing")
        groups["all"][e["image_fingerprint"]].append(rows)
        if not e["training_image_overlap"]:
            groups["train_overlap_free"][e["image_fingerprint"]].append(rows)
    tables = {}
    for subset, images in groups.items():
        if not images:
            tables[subset] = {"status": "no_eligible_images"}; continue
        table = {}
        for name in METHODS:
            rows = [q[name] for questions in images.values() for q in questions]
            table[name] = {"n_questions": len(rows), "n_images": len(images),
                "accuracy": sum(r["primary_task_score"] for r in rows)/len(rows),
                "mean_visual_tokens": sum(r["retained_visual_tokens"] for r in rows)/len(rows),
                "mean_nll": sum(r["answer_nll"] for r in rows)/len(rows),
                "score_delta_vs_dense": clustered_delta(list(images.values()), name, "dense")}
        tables[subset] = table
    result = {"status": "complete", "dataset": dataset, "stage": stage, "n_questions": len(entries), "tables": tables,
        "checkpoint": protocol["checkpoint"], "adapter_sha256": protocol["adapter_sha256"],
        "interpretation": "Fixed checkpoint, no retraining. Full splits and train-image-clean subsets; question-weighted scores with image-cluster paired confidence intervals."}
    atomic_write_json(folder/f"{stage}_report.json", result)
    lines = [f"# Qwen2.5-VL-7B {dataset}: {stage}", "", result["interpretation"], "",
        "| Subset | Method | Questions / images | Native score | Visual tokens |", "|---|---|---:|---:|---:|"]
    for subset, table in tables.items():
        if "status" in table:
            continue
        for name in METHODS:
            row = table[name]
            lines.append(f"| {subset} | {name} | {row['n_questions']} / {row['n_images']} | {100*row['accuracy']:.2f}% | {row['mean_visual_tokens']:.1f} |")
    lines += ["", protocol["timing_caveat"], "", protocol["comparison_scope"]]
    (folder/f"{stage}_report.md").write_text("\n".join(lines)+"\n", encoding="utf-8")
    return result


def run(root, source, geometry, reuse=None):
    root.mkdir(parents=True, exist_ok=True)
    with (root/"runner.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        state = {"status": "preparing", "pid": os.getpid(), "started_utc": datetime.now(timezone.utc).isoformat(),
            "source_training_run": str(source), "gpu1_waits_for": str(geometry/"run_manifest.json"), "stages": {}}
        atomic_write_json(root/"run_manifest.json", state)
        logs = root/"logs"; logs.mkdir(exist_ok=True)
        mutex, stop = threading.Lock(), threading.Event()
        children = []
        def execute(dataset, stage, shard, device):
            validate(root)
            if stop.is_set():
                raise RuntimeError("Sibling failed; not starting additional jobs")
            name = f"{dataset}_{stage}_{shard}"
            log = logs/f"{name}_{int(time.time())}.log"
            with log.open("w") as stream:
                command = [sys.executable, str(Path(__file__)), "--root", str(root), "--worker", dataset,
                    "--stage", stage, "--shard", str(shard), "--device", device]
                child = subprocess.Popen(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True,
                    env={**os.environ, "PYTHONUNBUFFERED": "1", "OMP_NUM_THREADS": "4", "TOKENIZERS_PARALLELISM": "false"})
                with mutex:
                    children.append(child)
                    state["stages"][name] = {"status": "running", "pid": child.pid, "device": device, "log": str(log)}
                    atomic_write_json(root/"run_manifest.json", state)
                while child.poll() is None:
                    if stop.wait(2):
                        os.killpg(child.pid, signal.SIGTERM); child.wait()
                        raise RuntimeError("Stopped owned child after sibling failure")
                with mutex:
                    state["stages"][name]["status"] = "complete" if child.returncode == 0 else "failed"
                    atomic_write_json(root/"run_manifest.json", state)
                if child.returncode:
                    stop.set()
                    raise RuntimeError(f"{name} failed: {log}")
        def wait_geometry():
            while not stop.is_set():
                status = json.loads((geometry/"run_manifest.json").read_text())["status"]
                if status == "complete":
                    return
                if status in {"failed", "blocked", "superseded"}:
                    raise RuntimeError("GPU1 geometry queue did not complete; not interfering with it")
                stop.wait(5)
            raise RuntimeError("Stopped waiting for GPU1")
        def lane1(dataset):
            try:
                wait_geometry()
                execute(dataset, "full", 1, "cuda:1")
            except BaseException:
                stop.set(); raise
        try:
            protocol = prepare(root, source, geometry, reuse)
            state.update(status="running", checkpoint=protocol["checkpoint"], implementation_hash=protocol["implementation_hash"])
            atomic_write_json(root/"run_manifest.json", state)
            for dataset in DATASETS:
                state["active_dataset"] = dataset
                # Smoke uses GPU0 only, so the 500-image run keeps exclusive use
                # of our GPU1 lane. Full shards share no image content.
                for shard in [0, 1]:
                    execute(dataset, "smoke", shard, "cuda:0")
                report(root, dataset, "smoke")
                with ThreadPoolExecutor(max_workers=2) as pool:
                    a = pool.submit(execute, dataset, "full", 0, "cuda:0")
                    b = pool.submit(lane1, dataset)
                    try:
                        a.result(); b.result()
                    except BaseException:
                        stop.set(); raise
                report(root, dataset, "full")
            state.update(status="complete", completed_utc=datetime.now(timezone.utc).isoformat())
        except BaseException as exc:
            stop.set()
            for child in children:
                if child.poll() is None:
                    os.killpg(child.pid, signal.SIGTERM); child.wait()
            state.update(status="failed", error=str(exc), traceback=traceback.format_exc())
            raise
        finally:
            atomic_write_json(root/"run_manifest.json", state)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--source-run", type=Path)
    parser.add_argument("--geometry-run", type=Path)
    parser.add_argument("--reuse-run", type=Path)
    parser.add_argument("--launch", action="store_true")
    parser.add_argument("--worker", choices=list(DATASETS))
    parser.add_argument("--stage", choices=["smoke", "full"], default="smoke")
    parser.add_argument("--shard", type=int, choices=[0, 1], default=0)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    root = args.root.resolve()
    if args.worker:
        worker(root, args.worker, args.stage, args.shard, args.device)
    else:
        if args.source_run is None or args.geometry_run is None:
            parser.error("Supervisor requires --source-run and --geometry-run")
        source, geometry = args.source_run.resolve(), args.geometry_run.resolve()
        reuse = args.reuse_run.resolve() if args.reuse_run else None
        root.mkdir(parents=True, exist_ok=True)
        if args.launch:
            with (root/"supervisor.log").open("a") as stream:
                command = [sys.executable, str(Path(__file__)), "--root", str(root),
                    "--source-run", str(source), "--geometry-run", str(geometry)]
                if reuse:
                    command += ["--reuse-run", str(reuse)]
                child = subprocess.Popen(command, cwd=ROOT,
                    stdout=stream, stderr=subprocess.STDOUT, start_new_session=True,
                    env={**os.environ, "PYTHONUNBUFFERED": "1"})
            print(json.dumps({"pid": child.pid, "root": str(root)}))
        else:
            run(root, source, geometry, reuse)
