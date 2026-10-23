"""Matched full-split projector comparisons, preserving historical predictions.

Run CPU tests first. The supervisor gates full execution on native-generation
parity on all smoke images, including direct-kernel/native-binding parity.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import fcntl
import hashlib
import inspect
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time
import traceback
import urllib.request

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from mllm_comp.analysis.common.anchor7b_runtime import (
    load_runtime, file_sha, validate_identity, explicit_generate,
)
from mllm_comp.analysis.common.condition_ledger import append_json, records
from mllm_comp.analysis.common.evidence_interaction import digest
from mllm_comp.analysis.common.projector_comparison import (
    METHODS, SOURCES, prepare_prompt_state, select_indices, decode_prompt,
)
from mllm_comp.core.serialization import atomic_write_json
from scripts.eval.lmms_eval import run_anchor7b_full as general
from scripts.eval.lmms_eval import run_anchor7b_ocr_doc_full as ocrdoc

EVAL_ROOT = ROOT / "outputs/eval/qwen25_vl"
SOURCE_RUNS = {
    "chartqa": EVAL_ROOT / "anchor7b_chartqa_ai2d_mlp_only_20261021_v3",
    "ai2d": EVAL_ROOT / "anchor7b_chartqa_ai2d_mlp_only_20261021_v3",
    "ocrbench": EVAL_ROOT / "anchor7b_ocrbench_docvqa_full_20261021_v1",
    "docvqa": EVAL_ROOT / "anchor7b_ocrbench_docvqa_full_20261021_v1",
}
DATASETS = ("ocrbench", "chartqa", "ai2d", "docvqa")
EXPECTED = {"chartqa": 2500, "ai2d": 3088, "ocrbench": 1000, "docvqa": 5349}
MAX_NEW = {"chartqa": 16, "ai2d": 16, "ocrbench": 128, "docvqa": 32}
ALL_METHODS = ("dense", "feature_anchor_adapter", *METHODS)


def now():
    return datetime.now(timezone.utc).isoformat()


def implementation_files():
    from mllm_comp.analysis.common import projector_comparison
    from mllm_comp.compressors.comparison import cdpruner, eadp, hiprune, qwen_projector, operations
    from mllm_comp.analysis.probes.visual_evidence_interaction import cached_decode
    modules = (projector_comparison, cdpruner, eadp, hiprune, qwen_projector, operations)
    paths = [Path(__file__), Path(inspect.getsourcefile(cached_decode))]
    paths.extend(Path(m.__file__) for m in modules)
    return {str(p.relative_to(ROOT)): file_sha(p) for p in paths}


def prepare(root):
    root.mkdir(parents=True, exist_ok=True)
    if (root / "protocol.json").exists():
        return validate(root)
    datasets, source_protocols = {}, {}
    first = None
    for dataset in DATASETS:
        source = SOURCE_RUNS[dataset]
        protocol = (general if dataset in {"chartqa", "ai2d"} else ocrdoc).validate(source)
        if not (source / dataset / "full_report.json").exists():
            raise ValueError(f"Historical full evaluation is not complete: {dataset}")
        if first is None:
            first = protocol
        for key in ("checkpoint", "revision", "base_identity_sha256", "adapter_sha256", "processor", "dtype", "attention", "budget"):
            if protocol[key] != first[key]:
                raise ValueError(f"Historical evaluations disagree on {key}")
        if protocol["generation"]["repetition_penalty"] != 1.0:
            raise ValueError("Historical repetition penalty changed")
        source_protocols[str(source)] = file_sha(source / "protocol.json")
        samples = json.loads((source / dataset / "samples.json").read_text())
        if len(samples["entries"]) != EXPECTED[dataset]:
            raise ValueError("Wrong full-split question count")
        folder = root / dataset
        folder.mkdir(exist_ok=True)
        atomic_write_json(folder / "samples.json", samples)
        datasets[dataset] = {**protocol["datasets"][dataset], "source_run": str(source),
            "source_config_hash": digest(json.dumps(protocol, sort_keys=True) + dataset + "full"),
            "samples_sha256": file_sha(folder / "samples.json"), "max_new_tokens": MAX_NEW[dataset]}
    identity = json.loads((SOURCE_RUNS["ocrbench"] / "model_identity.json").read_text())
    atomic_write_json(root / "model_identity.json", identity)
    provenance = {"status": "not_checked", "official_repository": "Qwen/Qwen2.5-VL-7B-Instruct"}
    try:
        remote = json.load(urllib.request.urlopen(
            "https://huggingface.co/api/models/Qwen/Qwen2.5-VL-7B-Instruct?blobs=true", timeout=25))
        matches = {}
        for entry in remote["siblings"]:
            name = entry["rfilename"]
            if name.startswith("model-") and name in identity["files"] and entry.get("lfs"):
                matches[name] = entry["lfs"]["sha256"] == identity["files"][name]["sha256"]
        if len(matches) != 5 or not all(matches.values()):
            raise ValueError("Local weight shards differ from official HF metadata")
        provenance.update(status="all_five_weight_shards_match", revision=remote["sha"], matches=matches)
    except (OSError, TimeoutError) as exc:
        provenance.update(error=str(exc))
    atomic_write_json(root / "official_weight_provenance.json", provenance)
    protocol = {"created_utc": now(), "checkpoint": first["checkpoint"], "revision": first["revision"],
        "base_identity_sha256": first["base_identity_sha256"], "identity_sha256": file_sha(root / "model_identity.json"),
        "official_weight_provenance_sha256": file_sha(root / "official_weight_provenance.json"),
        "adapter_sha256": first["adapter_sha256"], "budget": 128, "seed": 42,
        "dtype": "bfloat16", "attention": "eager", "processor": first["processor"],
        "methods": ALL_METHODS, "new_methods": METHODS, "sources": SOURCES,
        "method_scope": "Framework Qwen2.5 adaptations, not original-paper benchmark reproduction claims",
        "training": {"new_methods": False, "feature_anchor_adapter": "Previously trained supervised MLP; no retraining in this run"},
        "generation": {"do_sample": False, "repetition_penalty": 1.0, "eos_policy": "pinned_checkpoint_generation_config"},
        "datasets": datasets, "source_protocols": source_protocols,
        "implementation_files": implementation_files(),
        "primary_estimand": "Question-weighted native score; 10000 paired bootstraps clustered by independent image content",
        "reuse": "Dense/MLP predictions reused only with identical checkpoint, inputs, positions, generation and native scorer",
        "fairness": "Equal K128; training and query dependence disclosed. No training-free superiority claim from a trained arm alone.",
        "external_baseline_status": "Absolute Bifocal/OCRBench baseline protocol discrepancy remains under audit",
        "timing": "New rows include vision/scores/selection/decode components; shared-GPU diagnostic timing, not a deployment speed claim"}
    atomic_write_json(root / "protocol.json", protocol)
    return protocol


def validate(root):
    protocol = json.loads((root / "protocol.json").read_text())
    if implementation_files() != protocol["implementation_files"]:
        raise ValueError("Frozen comparison implementation changed")
    if file_sha(root / "model_identity.json") != protocol["identity_sha256"]:
        raise ValueError("Model identity changed")
    validate_identity(json.loads((root / "model_identity.json").read_text()))
    for path, sha in protocol["source_protocols"].items():
        if file_sha(Path(path) / "protocol.json") != sha:
            raise ValueError("Historical source protocol changed")
    for dataset, spec in protocol["datasets"].items():
        if file_sha(root / dataset / "samples.json") != spec["samples_sha256"]:
            raise ValueError("Sample list changed")
    return protocol


def task(dataset, doc, prediction=None):
    if dataset in {"chartqa", "ai2d"}:
        prompt, target = general.prompt_target(dataset, doc)
        refs = doc.get("answers", doc.get("answer"))
        scored = general.score_prediction(dataset, prediction, doc) if prediction is not None else None
    else:
        prompt, target, refs = ocrdoc.prompt_target(dataset, doc)
        scored = ocrdoc.score_prediction(dataset, prediction, doc) if prediction is not None else None
    return prompt, target, refs, scored


def iter_docs(entries, data_files):
    import pyarrow.parquet as pq
    wanted = defaultdict(dict)
    for entry in entries:
        wanted[entry["path"]][entry["row"]] = entry
    for path, rows in wanted.items():
        if file_sha(path) != data_files[path]:
            raise ValueError("Dataset parquet changed")
        index = 0
        for batch in pq.ParquetFile(path).iter_batches(batch_size=8):
            for doc in batch.to_pylist():
                entry = rows.get(index)
                index += 1
                if entry is not None:
                    yield entry, doc
            if index > max(rows):
                break


def source_rows(spec, entry):
    folder = Path(spec["source_run"]) / entry["sample_id"].split(":")[0] / "samples/full" / entry["sample_id"].replace(":", "_")
    path = folder / "conditions.jsonl"
    if not (folder / "summary.json").exists():
        raise ValueError(f"Historical sample incomplete: {entry['sample_id']}")
    rows = records(path, spec["source_config_hash"])
    if {key[1] for key in rows} != {"dense", "feature_anchor_adapter"} or len(rows) != 2:
        raise ValueError("Historical result has missing or duplicate arms")
    return path, {key[1]: value for key, value in rows.items()}


def native_binding_generate(bundle, state, method, max_new):
    from transformers import GenerationConfig
    from mllm_comp.compressors.comparison.qwen_projector import QwenProjectorComparison
    model = bundle["model"]
    session = QwenProjectorComparison(model, method, {"tokens_per_crop": 128}, "qwen25_vl").install()
    decode_positions = []
    def capture_decode(module, args, kwargs):
        past = kwargs.get("past_key_values")
        if past is not None and past.get_seq_length() > 0:
            decode_positions.append(int(kwargs["position_ids"][-3:].max()))
    hook = bundle["llm"].register_forward_pre_hook(capture_decode, with_kwargs=True)
    try:
        config = GenerationConfig(max_new_tokens=max_new, do_sample=False, use_cache=True,
            repetition_penalty=1.0, eos_token_id=model.generation_config.eos_token_id,
            pad_token_id=bundle["processor"].tokenizer.pad_token_id)
        ids = model.generate(**state["prompt_inputs"], generation_config=config)[0, state["prefix"]:].tolist()
        selection = session.runtime_metadata()["selection"]
        expected = int(state["positions"][:, :, :state["prefix"]].max()) + 1
        if decode_positions != list(range(expected, expected + len(decode_positions))):
            raise ValueError(f"{method}: native generate decode positions shifted")
        selection["native_decode_positions_checked"] = len(decode_positions)
        return ids, selection
    finally:
        hook.remove()
        session.uninstall()


def audit_native(bundle, state, old, dataset, doc, max_new):
    """Implementation audit, not a small-sample task-quality conclusion."""
    from transformers import GenerationConfig
    matched = explicit_generate(bundle, state, max_new=max_new)
    direct = decode_prompt(bundle, state, max_new=max_new)
    if matched != direct["generated_token_ids"] or matched != old["generated_token_ids"]:
        raise ValueError("Historical dense, native model.generate and cached decode disagree")
    # Standard lmms-eval wrapper inherits checkpoint repetition penalty 1.05
    # and passes tokenizer EOS. This arm never changes the primary protocol.
    native = bundle["model"].generate(**state["prompt_inputs"],
        generation_config=GenerationConfig(max_new_tokens=max_new, do_sample=False,
            repetition_penalty=1.05, eos_token_id=bundle["processor"].tokenizer.eos_token_id,
            pad_token_id=bundle["processor"].tokenizer.pad_token_id, use_cache=True))
    ids = native[0, state["prefix"]:].tolist()
    text = bundle["processor"].tokenizer.decode(ids, skip_special_tokens=True).strip()
    return {"native_cached_historical_parity": True,
        "standard_generation_prediction": text, "standard_generated_token_ids": ids,
        "standard_primary_task_score": float(task(dataset, doc, text)[3][0]),
        "matched_primary_task_score": old["primary_task_score"],
        "changed_under_standard_generation_settings": ids != matched,
        "purpose": "Generation implementation/protocol audit; not inference about benchmark accuracy"}


def worker(root, stage, shard, device):
    import torch
    torch.set_num_threads(4)
    torch.cuda.set_device(device)
    torch.manual_seed(42)
    protocol = validate(root)
    identity = json.loads((root / "model_identity.json").read_text())
    status_path = root / f"{stage}_{shard}_manifest.json"
    status = {"status": "loading", "stage": stage, "shard": shard, "device": device,
        "started_utc": now(), "completed_questions": 0, "completed_new_conditions": 0}
    atomic_write_json(status_path, status)
    bundle = None
    try:
        bundle = load_runtime(identity, device)
        for dataset in DATASETS:
            spec = protocol["datasets"][dataset]
            entries = [e for e in json.loads((root / dataset / "samples.json").read_text())["entries"]
                       if e["shard"] == shard and (stage == "full" or e["smoke"])]
            config_hash = digest(json.dumps(protocol, sort_keys=True) + dataset + stage)
            status.update(status="running", dataset=dataset, dataset_expected=len(entries), dataset_completed=0)
            atomic_write_json(status_path, status)
            for entry, doc in iter_docs(entries, spec["data_files"]):
                folder = root / dataset / "samples" / stage / entry["sample_id"].replace(":", "_")
                folder.mkdir(parents=True, exist_ok=True)
                ledger = folder / "conditions.jsonl"
                saved = records(ledger, config_hash)
                old_path, old_rows = source_rows(spec, entry)
                prompt, target, refs, _ = task(dataset, doc)
                for method in ("dense", "feature_anchor_adapter"):
                    old = old_rows[method]
                    if (old["base_identity_sha256"] != identity["identity_sha256"] or old["question"] != prompt or
                            old["image_fingerprint"] != entry["image_fingerprint"] or old["gold_answer_in_decode_context"] or
                            old["max_new_tokens"] != MAX_NEW[dataset] or
                            float(task(dataset, doc, old["prediction"])[3][0]) != old["primary_task_score"]):
                        raise ValueError("Historical prediction/input/scorer identity failed")
                    if method == "feature_anchor_adapter" and old["adapter_sha256"] != protocol["adapter_sha256"]:
                        raise ValueError("Historical adapter mismatch")
                    key = (entry["sample_id"], method)
                    if key not in saved:
                        copied = {**old, "config_hash": config_hash, "reused_from": str(old_path),
                            "source_config_hash": old["config_hash"], "source_ledger_sha256": file_sha(old_path)}
                        append_json(ledger, copied)
                        saved[key] = copied
                missing = [m for m in METHODS if (entry["sample_id"], m) not in saved]
                if missing:
                    started = time.perf_counter()
                    with torch.inference_mode():
                        state, captured = prepare_prompt_state(bundle, entry, doc["image"], prompt)
                        torch.cuda.synchronize(device)
                        shared_vision_prepare_s = time.perf_counter() - started
                        if stage == "smoke":
                            audit = audit_native(bundle, state, old_rows["dense"], dataset, doc, MAX_NEW[dataset])
                            atomic_write_json(folder / "native_audit.json", audit)
                        for method in missing:
                            torch.cuda.synchronize(device)
                            begin = time.perf_counter()
                            selected, trace = select_indices(bundle, state, method,
                                attention=captured["attention"], capture_config=captured["config"])
                            torch.cuda.synchronize(device)
                            selection_s = time.perf_counter() - begin
                            begin = time.perf_counter()
                            result = decode_prompt(bundle, state, max_new=MAX_NEW[dataset], selected=selected)
                            torch.cuda.synchronize(device)
                            decode_s = time.perf_counter() - begin
                            if stage == "smoke":
                                native, native_trace = native_binding_generate(bundle, state, method, MAX_NEW[dataset])
                                if native != result["generated_token_ids"] or native_trace["selected_indices"] != selected.tolist():
                                    raise ValueError(f"{method}: direct kernel/native generate binding parity failed")
                                result["native_binding_generation_identity_passed"] = True
                            score, metrics, *rest = task(dataset, doc, result["prediction"])[3]
                            if not math.isfinite(score) or not 0 <= score <= 1 or result["retained_visual_tokens"] != 128:
                                raise ValueError("Nonfinite score, wrong K128 budget or invalid task score")
                            row = {**result, **entry, "dataset": dataset, "variant": method,
                                "config_hash": config_hash, "status": "completed", "checkpoint": identity["checkpoint"],
                                "revision": identity["revision"], "base_identity_sha256": identity["identity_sha256"],
                                "question": prompt, "references": refs, "primary_task_score": float(score),
                                "native_task_metrics": metrics, "compression_trace": trace,
                                "ocrbench_category": doc.get("question_type") if dataset == "ocrbench" else None,
                                "trained": False, "repetition_penalty": 1.0,
                                "shared_vision_prepare_s": shared_vision_prepare_s,
                                "selection_s": selection_s, "decode_s": decode_s,
                                "elapsed_s": shared_vision_prepare_s + selection_s + decode_s}
                            append_json(ledger, row)
                            saved[(entry["sample_id"], method)] = row
                            status["completed_new_conditions"] += 1
                    del state, captured, result, selected
                if {key[1] for key in saved} != set(ALL_METHODS):
                    raise ValueError("Missing comparison arm")
                if stage == "smoke":
                    if not (folder / "native_audit.json").exists() or any(
                            not saved[(entry["sample_id"], m)].get("native_binding_generation_identity_passed") for m in METHODS):
                        raise ValueError("Smoke parity evidence missing")
                atomic_write_json(folder / "summary.json", {**entry, "status": "complete",
                    "condition_count": len(ALL_METHODS), "config_hash": config_hash})
                status["completed_questions"] += 1
                status["dataset_completed"] += 1
                atomic_write_json(status_path, status)
                print(json.dumps({"stage": stage, "dataset": dataset, "shard": shard,
                    "completed": status["dataset_completed"], "expected": len(entries)}), flush=True)
            if status["dataset_completed"] != len(entries):
                raise ValueError(f"Missing questions in {dataset} shard {shard}")
        status.update(status="complete", completed_utc=now(), peak_allocated_bytes=torch.cuda.max_memory_allocated())
    except BaseException as exc:
        status.update(status="failed", error=str(exc), traceback=traceback.format_exc())
        atomic_write_json(root / f"failure_{stage}_{shard}.json", status)
        raise
    finally:
        atomic_write_json(status_path, status)
        if bundle:
            bundle["runtime"].close()


def report(root, stage):
    protocol = validate(root)
    lines = ["# Qwen2.5-VL-7B projector comparisons", "",
        "Fixed K128 and historical protocol. New methods are framework Qwen adaptations.", "",
        "Dense and the supervised MLP are reused from identical full-split runs. Other methods receive no training.", "",
        "| Dataset | Method | Questions | Native score | Delta vs dense (pp) | Visual tokens |",
        "|---|---|---:|---:|---:|---:|"]
    results = {}
    for dataset in DATASETS:
        entries = [e for e in json.loads((root / dataset / "samples.json").read_text())["entries"] if stage == "full" or e["smoke"]]
        config_hash = digest(json.dumps(protocol, sort_keys=True) + dataset + stage)
        images = defaultdict(list)
        for e in entries:
            folder = root / dataset / "samples" / stage / e["sample_id"].replace(":", "_")
            if not (folder / "summary.json").exists():
                raise ValueError("Incomplete sample cannot enter report")
            saved = records(folder / "conditions.jsonl", config_hash)
            rows = {k[1]: r for k, r in saved.items()}
            if set(rows) != set(ALL_METHODS):
                raise ValueError("Incomplete method set")
            for method, row in rows.items():
                if row["image_fingerprint"] != e["image_fingerprint"] or row["gold_answer_in_decode_context"]:
                    raise ValueError("Mixed image or answer-leakage flag")
                if method != "dense" and row["retained_visual_tokens"] != 128:
                    raise ValueError("Wrong compressed token count")
            images[e["image_fingerprint"]].append(rows)
        table = {}
        for method in ALL_METHODS:
            rows = [q[method] for questions in images.values() for q in questions]
            delta = general.clustered_delta(list(images.values()), method)
            categories = defaultdict(lambda: {"count": 0, "score_sum": 0.0})
            bins = defaultdict(lambda: {"count": 0, "score_sum": 0.0})
            for row in rows:
                if row.get("ocrbench_category"):
                    c = categories[row["ocrbench_category"]]
                    c["count"] += 1; c["score_sum"] += row["primary_task_score"]
                n = row["source_visual_tokens"]
                key = "128-511" if n < 512 else "512-1023" if n < 1024 else "1024-2048" if n <= 2048 else ">2048"
                b = bins[key]; b["count"] += 1; b["score_sum"] += row["primary_task_score"]
            table[method] = {"n_questions": len(rows), "n_images": len(images),
                "score": sum(r["primary_task_score"] for r in rows) / len(rows),
                "mean_visual_tokens": sum(r["retained_visual_tokens"] for r in rows) / len(rows),
                "delta_vs_dense": delta,
                "delta_vs_mlp": general.clustered_delta(list(images.values()), method, "feature_anchor_adapter"),
                "categories": dict(categories), "source_token_bins": dict(bins),
                "paired_wins_ties_losses_vs_mlp": {
                    "wins": sum(q[method]["primary_task_score"] > q["feature_anchor_adapter"]["primary_task_score"] for qs in images.values() for q in qs),
                    "ties": sum(q[method]["primary_task_score"] == q["feature_anchor_adapter"]["primary_task_score"] for qs in images.values() for q in qs),
                    "losses": sum(q[method]["primary_task_score"] < q["feature_anchor_adapter"]["primary_task_score"] for qs in images.values() for q in qs)}}
            value = table[method]
            label = f"{value['score'] * len(rows):.0f}/{len(rows)}" if dataset == "ocrbench" else f"{value['score'] * 100:.2f}%"
            lines.append(f"| {dataset} | {method} | {len(rows)} | {label} | {100 * delta['mean']:+.2f} | {value['mean_visual_tokens']:.1f} |")
        results[dataset] = table
    result = {"status": "complete", "stage": stage, "tables": results, "protocol_sha256": file_sha(root / "protocol.json")}
    atomic_write_json(root / f"{stage}_report.json", result)
    lines += ["", "Image-cluster paired 95% intervals, OCR categories and token-count buckets are in the JSON report.",
        "", protocol["fairness"], "", protocol["timing"], "", protocol["external_baseline_status"]]
    # atomic_write_json is used for machine data; apply_patch owns source edits.
    from mllm_comp.core.serialization import atomic_write_text
    atomic_write_text(root / f"{stage}_report.md", "\n".join(lines) + "\n")
    return result


def run(root):
    root.mkdir(parents=True, exist_ok=True)
    with (root / "runner.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        prepare(root)
        manifest_path = root / "run_manifest.json"
        state = {"status": "running", "started_utc": now(), "pid": os.getpid(), "stages": {}}
        atomic_write_json(manifest_path, state)
        stop, mutex, children = threading.Event(), threading.Lock(), []
        logs = root / "logs"; logs.mkdir(exist_ok=True)

        def execute(stage, shard):
            while True:
                if stop.is_set():
                    raise RuntimeError("Sibling worker failed")
                raw = subprocess.check_output(["nvidia-smi", f"--id={shard}",
                    "--query-gpu=memory.free", "--format=csv,noheader,nounits"], text=True)
                if int(raw.strip()) >= 40000:
                    break
                with mutex:
                    state["stages"][f"{stage}_{shard}"] = {"status": "waiting_for_gpu_memory", "free_mib": int(raw.strip())}
                    atomic_write_json(manifest_path, state)
                stop.wait(15)
            log = logs / f"{stage}_{shard}_{int(time.time())}.log"
            with log.open("w") as stream:
                child = subprocess.Popen([sys.executable, str(Path(__file__)), "--root", str(root),
                    "--worker", stage, "--shard", str(shard), "--device", f"cuda:{shard}"],
                    cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True,
                    env={**os.environ, "PYTHONUNBUFFERED": "1", "OMP_NUM_THREADS": "4", "TOKENIZERS_PARALLELISM": "false"})
                with mutex:
                    children.append(child)
                    state["stages"][f"{stage}_{shard}"] = {"status": "running", "pid": child.pid, "log": str(log)}
                    atomic_write_json(manifest_path, state)
                while child.poll() is None:
                    if stop.wait(2):
                        os.killpg(child.pid, signal.SIGTERM)
                        child.wait()
                        raise RuntimeError("Stopped owned comparison worker after sibling failure")
                with mutex:
                    state["stages"][f"{stage}_{shard}"]["status"] = "complete" if child.returncode == 0 else "failed"
                    atomic_write_json(manifest_path, state)
                if child.returncode:
                    stop.set()
                    raise RuntimeError(f"Worker failed; see {log}")
        try:
            for stage in ("smoke", "full"):
                state["active_stage"] = stage
                atomic_write_json(manifest_path, state)
                with ThreadPoolExecutor(max_workers=2) as pool:
                    for future in as_completed([pool.submit(execute, stage, shard) for shard in (0, 1)]):
                        future.result()
                report(root, stage)
            state.update(status="complete", completed_utc=now())
        except BaseException as exc:
            stop.set()
            for child in children:
                if child.poll() is None:
                    os.killpg(child.pid, signal.SIGTERM)
                    child.wait()
            state.update(status="failed", error=str(exc), traceback=traceback.format_exc())
            raise
        finally:
            atomic_write_json(manifest_path, state)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--worker", choices=("smoke", "full"))
    parser.add_argument("--shard", type=int, choices=(0, 1), default=0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    root = args.root.resolve()
    if args.worker:
        worker(root, args.worker, args.shard, args.device)
    elif args.prepare_only:
        prepare(root)
    else:
        run(root)
