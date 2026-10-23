"""Persistent two-GPU queue for fitting audits and a zero-init strong control."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import threading
import time
import traceback

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from mllm_comp.analysis.common.compression_training_check import implementation_hash
from mllm_comp.analysis.common.evidence_interaction import bootstrap
from mllm_comp.analysis.common.learned_hybrid_runtime import code_hash
from mllm_comp.core.serialization import atomic_write_json


def prepare(root, source):
    import yaml
    root.mkdir(parents=True, exist_ok=True)
    if (root / "protocol.json").exists():
        raise ValueError("Existing run: use --launch or resume, not --prepare")
    (root / "configs").mkdir(exist_ok=True)
    shutil.copyfile(source / "cohort.json", root / "cohort.json")
    atomic_write_json(root / "protocol.json", {
        "source": str(source), "seed": 42, "checkpoint_family": "Qwen2.5-VL-3B-Instruct",
        "audit": {"n_train_images": 256, "textvqa": 192, "scienceqa": 64,
                  "methods": "dense, feature-anchor and all four hybrid initial/final checkpoints"},
        "capacity": {"n_train_images": 64, "textvqa": 48, "scienceqa": 16,
                     "initialization": "summary_residual step320", "loss": "CE only", "steps": 320, "accumulation": 4,
                     "scope": "Optimization/capacity diagnostic, NOT generalization evidence"},
        "strong_control": {"initialization": "exact original feature_anchor_mean K128 with rounded centroid M-RoPE",
            "adapter": "zero-initialized 256-width residual MLP", "frozen": "vision, merger, LLM",
            "loss": "CE + 0.5 teacher KL", "steps": 320, "accumulation": 8, "train_images": 2560,
            "evaluations": "16 smoke, 200 development, 800 previously-used held-out images; no new independent confirmation"},
        "selection": "fixed image hash, no outcome filtering", "chartqa_training": False,
        "lr": .0002, "weight_decay": .01, "gradient_clip": 1.,
        "decision": "Do not claim more data will solve failure. Distinguish failure to fit, task transfer, interface damage, and objective mismatch."
    })
    for stage in ["smoke", "development", "confirmation"]:
        for shard in [0, 1]:
            template = yaml.safe_load((source / "configs" / f"{stage}_{shard}.yaml").read_text())
            template["experiment"]["name"] = f"anchor_training_check_{stage}_{shard}"
            template["analysis"]["sampling"]["manifest_path"] = str(root / "cohort.json")
            template["analysis"]["probes"] = [{"name": "anchor_training_evaluation", "enabled": True,
                "artifact_root": str(root), "cohort_path": str(root / "cohort.json"), "stage": stage,
                "checkpoint_path": str(root / "anchor/checkpoints/step_000320"), "save_raw": False}]
            (root / "configs" / f"{stage}_{shard}.yaml").write_text(yaml.safe_dump(template, sort_keys=False), encoding="utf-8")


def hashes(root):
    paths = [Path(__file__), Path(__file__).with_name("check_compression_training.py"),
             ROOT / "mllm_comp/analysis/probes/anchor_training_evaluation.py", ROOT / "mllm_comp/registry/catalog.py",
             root / "cohort.json", root / "protocol.json"] + sorted((root / "configs").glob("*.yaml"))
    return {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}


def report(root, stage):
    cohort = json.loads((root / "cohort.json").read_text())
    entries = [e for e in cohort["entries"] if (e.get("smoke") if stage == "smoke" else e["split"] == stage)]
    tasks = {}
    historical = Path(json.loads((root / "protocol.json").read_text())["source"])
    matched = 0
    for entry in entries:
        folder = root / "samples" / stage / entry["sample_id"].replace(":", "_")
        summary = json.loads((folder / "summary.json").read_text())
        rows = {r["variant"]: r for r in map(json.loads, (folder / "conditions.jsonl").read_text().splitlines())}
        if summary["implementation_hash"] != implementation_hash() or set(rows) != {"dense", "feature_anchor_untrained", "feature_anchor_adapter"}:
            raise ValueError("Incomplete/mixed evaluation")
        for name, row in rows.items():
            if row["image_fingerprint"] != entry["image_fingerprint"] or row["gold_answer_in_decode_context"]:
                raise ValueError("Gold leak or image mismatch")
            if name != "dense" and row["retained_visual_tokens"] != 128:
                raise ValueError("Wrong visual budget")
        if stage == "smoke" and not rows["dense"].get("native_generation_identity_passed"):
            raise ValueError("Missing dense/native generation parity")
        prior_path = historical / "samples" / stage / entry["sample_id"].replace(":", "_") / "conditions.jsonl"
        prior = {r["variant"]: r for r in map(json.loads, prior_path.read_text().splitlines())}
        if prior["dense"]["prediction"] != rows["dense"]["prediction"] or prior["dense"]["source_visual_tokens"] != rows["dense"]["source_visual_tokens"]:
            raise ValueError("Historical dense prediction/count differs; cannot attribute adapter effect")
        matched += 1
        tasks.setdefault(entry["task"], []).append(rows)
    tables = {}
    for task, images in tasks.items():
        tables[task] = {name: {"n_images": len(images),
            "mean_score": sum(i[name]["primary_task_score"] for i in images)/len(images),
            "mean_nll": sum(i[name]["answer_nll"] for i in images)/len(images)} for name in images[0]}
        tables[task]["adapter_minus_untrained"] = bootstrap([
            i["feature_anchor_adapter"]["primary_task_score"]-i["feature_anchor_untrained"]["primary_task_score"] for i in images])
    result = {"status": "complete", "stage": stage, "n_images": len(entries), "historical_dense_matches": matched,
              "tables": tables, "interpretation": "One training seed; previously-used images held out from training, not new independent confirmation."}
    atomic_write_json(root / f"{stage}_report.json", result)
    return result


def run(root):
    root = root.resolve()
    source = Path(json.loads((root / "protocol.json").read_text())["source"])
    (root / "logs").mkdir(exist_ok=True)
    with (root / "runner.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        state = {"status": "running", "pid": os.getpid(), "started_utc": datetime.now(timezone.utc).isoformat(),
                 "implementation_hash": implementation_hash(), "old_checkpoint_implementation_hash": code_hash(),
                 "frozen_hashes": hashes(root), "stages": {}}
        mutex, stop = threading.Lock(), threading.Event()
        def execute(name, cmd):
            if stop.is_set():
                raise RuntimeError("Another owned worker failed; stopping queue")
            if hashes(root) != state["frozen_hashes"] or implementation_hash() != state["implementation_hash"]:
                raise RuntimeError("Frozen code/config changed")
            path = root / "logs" / f"{name}_{int(time.time())}.log"
            with path.open("w") as stream:
                child = subprocess.Popen(cmd, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT,
                    env={**os.environ, "PYTHONUNBUFFERED": "1", "OMP_NUM_THREADS": "4", "TOKENIZERS_PARALLELISM": "false"})
                with mutex:
                    state["stages"][name] = {"status": "running", "pid": child.pid, "log": str(path)}
                    atomic_write_json(root / "run_manifest.json", state)
                while child.poll() is None:
                    if stop.wait(2):
                        child.terminate()  # Only this runner's own child.
                        child.wait()
                        raise RuntimeError("Stopped after sibling failure")
                with mutex:
                    state["stages"][name]["status"] = "complete" if child.returncode == 0 else "failed"
                    atomic_write_json(root / "run_manifest.json", state)
                if child.returncode:
                    stop.set()
                    raise RuntimeError(f"{name} failed; see {path}")
        def command(mode, gpu, smoke=False):
            folder = root / (("smoke_" if smoke else "") + mode)
            return [sys.executable, str(Path(__file__).with_name("check_compression_training.py")),
                "--source", str(source), "--output", str(folder), "--mode", mode, "--device", f"cuda:{gpu}",
                "--steps", "2" if smoke else "320", "--accumulation", "2" if smoke else "4" if mode == "overfit" else "8",
                "--resume"] + (["--smoke"] if smoke else [])
        def branch0():
            execute("audit_smoke", command("audit", 0, True))
            execute("audit_train256", command("audit", 0))
            execute("overfit_smoke", command("overfit", 0, True))
            execute("overfit_train64", command("overfit", 0))
        def branch1():
            execute("anchor_smoke", command("anchor", 1, True))
            smoke = json.loads((root / "smoke_anchor/before.summary.json").read_text())
            for table in smoke["tables"].values():
                if table["feature_anchor_untrained"] != table["feature_anchor_adapter"]:
                    raise ValueError("Zero-init adapter changed native feature-anchor outputs")
            execute("anchor_train2560", command("anchor", 1))
            for stage in ["smoke", "development", "confirmation"]:
                for shard in [0, 1]:
                    execute(f"eval_{stage}_{shard}", [str(Path(sys.executable).parent / "mllm-comp"), "analyze", "--config",
                        str(root / "configs" / f"{stage}_{shard}.yaml"), "--device", "cuda:1", "--output_dir",
                        str(root / "lifecycle" / f"{stage}_{shard}_{int(time.time())}")])
                report(root, stage)
        def guarded(fn):
            try:
                fn()
            except BaseException:
                stop.set()
                raise
        try:
            with ThreadPoolExecutor(max_workers=2) as pool:
                futures = [pool.submit(guarded, fn) for fn in [branch0, branch1]]
                for future in futures:
                    future.result()
            components = {name: json.loads((root / path).read_text()) for name, path in {
                "train256_initial_final_audit": "audit/predictions.summary.json",
                "capacity64_before": "overfit/before.summary.json", "capacity64_after": "overfit/after.summary.json",
                "anchor_train_before": "anchor/before.summary.json", "anchor_train_after": "anchor/after.summary.json",
                "heldout800": "confirmation_report.json"}.items()}
            atomic_write_json(root / "final_report.json", {"status": "complete", "components": components,
                "warning": "Capacity-check data are reused training images. Only the 800-image table measures held-out quality; it reuses a previously evaluated cohort."})
            lines = ["# Compression training checks", "", "Train-only fitting diagnostics and held-out quality are separate.", "",
                     "| Task | Dense | Feature-anchor K128 | Trained zero-init adapter | Adapter minus untrained, 95% CI |",
                     "|---|---:|---:|---:|---|"]
            for task, table in components["heldout800"]["tables"].items():
                effect = table["adapter_minus_untrained"]
                values = [100*table[n]["mean_score"] for n in ["dense", "feature_anchor_untrained", "feature_anchor_adapter"]]
                lines.append(f"| {task} | {values[0]:.2f}% | {values[1]:.2f}% | {values[2]:.2f}% | "
                    f"{100*effect['mean']:.2f} pp [{100*effect['ci95'][0]:.2f}, {100*effect['ci95'][1]:.2f}] |")
            lines += ["", "Training-set generation/NLL summaries: see final_report.json. Small repeated-data checks cannot establish generalization."]
            (root / "final_report.md").write_text("\n".join(lines)+"\n", encoding="utf-8")
            state["status"] = "complete"
        except BaseException as exc:
            state.update({"status": "failed", "error": str(exc), "traceback": traceback.format_exc()})
            raise
        finally:
            state["updated_utc"] = datetime.now(timezone.utc).isoformat()
            atomic_write_json(root / "run_manifest.json", state)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--source", type=Path)
    p.add_argument("--prepare", action="store_true")
    p.add_argument("--launch", action="store_true")
    args = p.parse_args()
    root = args.root.resolve()
    if args.prepare:
        prepare(root, args.source.resolve())
    if args.launch:
        with (root / "supervisor.log").open("a") as stream:
            child = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), "--root", str(root)],
                cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
        print(json.dumps({"supervisor_pid": child.pid, "root": str(root)}))
    elif not args.prepare:
        run(root)
