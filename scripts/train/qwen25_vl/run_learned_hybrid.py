"""Durable two-GPU training and evaluation queue; fail only our own children."""

import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts/analysis"))
from mllm_comp.analysis.common.learned_hybrid_runtime import code_hash
from mllm_comp.compressors.image.learned_hybrid import VARIANTS
from mllm_comp.core.serialization import atomic_write_json
from report_learned_hybrid import report


def frozen_hashes(root):
    paths = [Path(__file__), Path(__file__).with_name("train_learned_hybrid.py"), Path(__file__).with_name("prepare_learned_hybrid.py"),
             ROOT / "scripts/analysis/report_learned_hybrid.py", ROOT / "mllm_comp/analysis/probes/learned_hybrid_evaluation.py",
             ROOT / "mllm_comp/registry/catalog.py", ROOT / "mllm_comp/analysis/probes/__init__.py",
             root / "preregistration.json", root / "training_samples.json", root / "cohort.json"]
    paths.extend(sorted((root / "configs").glob("*.yaml")))
    paths.extend(sorted((root / "preflight/configs").glob("*.yaml")))
    return {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}


def run(root):
    root = root.resolve()
    logs = root / "logs"
    logs.mkdir(exist_ok=True)
    with (root / "runner.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        manifest = {"status": "running", "pid": os.getpid(), "started_utc": datetime.now(timezone.utc).isoformat(),
                    "implementation_hash": code_hash(), "frozen_hashes": frozen_hashes(root), "stages": {},
                    "software": {name: importlib.metadata.version(name) for name in ["torch", "transformers", "datasets", "numpy"]},
                    "gpu": subprocess.check_output(["nvidia-smi", "--query-gpu=index,name,uuid,memory.total,driver_version", "--format=csv"], text=True)}
        children = []

        def execute(stage, commands):
            nonlocal children
            if manifest["implementation_hash"] != code_hash() or manifest["frozen_hashes"] != frozen_hashes(root):
                raise RuntimeError("Code, data manifest or configuration changed during the frozen suite.")
            children = []
            manifest["active_stage"] = stage
            attempt = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
            for gpu, command in enumerate(commands):
                path = logs / f"{stage}_{gpu}_{attempt}.log"
                stream = path.open("w")
                process = subprocess.Popen(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT,
                    env={**os.environ, "PYTHONUNBUFFERED": "1", "OMP_NUM_THREADS": "4", "TOKENIZERS_PARALLELISM": "false"})
                children.append((process, stream, str(path)))
            manifest["stages"][stage] = {"status": "running", "children": [{"pid": p.pid, "log": path} for p, _, path in children]}
            atomic_write_json(root / "run_manifest.json", manifest, ensure_ascii=False)
            while any(p.poll() is None for p, _, _ in children):
                if any(p.poll() not in {None, 0} for p, _, _ in children):
                    raise RuntimeError(f"{stage} worker failed; see logs.")
                time.sleep(5)
            if any(p.returncode for p, _, _ in children):
                raise RuntimeError(f"{stage} worker exited unsuccessfully.")
            for _, stream, _ in children:
                stream.close()
            if manifest["implementation_hash"] != code_hash() or manifest["frozen_hashes"] != frozen_hashes(root):
                raise RuntimeError("Suite code/configuration changed during a stage.")
            manifest["stages"][stage]["status"] = "complete"
            atomic_write_json(root / "run_manifest.json", manifest, ensure_ascii=False)

        try:
            waves = [("pool_adapter", "spectral_residual"), ("summary_random", "summary_residual")]
            for mode in ["smoke", "train"]:
                for number, variants in enumerate(waves):
                    commands = []
                    for gpu, variant in enumerate(variants):
                        folder = root / ("smoke_models" if mode == "smoke" else "models") / variant
                        command = [sys.executable, str(Path(__file__).with_name("train_learned_hybrid.py")), "--root", str(root),
                                   "--output", str(folder), "--variant", variant, "--device", f"cuda:{gpu}", "--resume"]
                        if mode == "smoke":
                            command += ["--steps", "2", "--accumulation", "2", "--smoke"]
                        commands.append(command)
                    execute(f"{mode}_{number}", commands)
                    for variant in variants:
                        folder = root / ("smoke_models" if mode == "smoke" else "models") / variant
                        state = json.loads((folder / "manifest.json").read_text())
                        if state["status"] != "complete" or not state["checkpoint_roundtrip_passed"]:
                            raise RuntimeError(f"Training or checkpoint restore gate failed: {variant}.")
                        if mode == "smoke" and not state.get("native_generation_identity_passed"):
                            raise RuntimeError("Native training-runtime generation parity failed.")
                if mode == "smoke":
                    preview = root / "preflight"
                    commands = [[str(Path(sys.executable).parent / "mllm-comp"), "analyze", "--config", str(preview / "configs" / f"smoke_{gpu}.yaml"),
                        "--device", f"cuda:{gpu}", "--output_dir", str(preview / "lifecycle" / f"smoke_{gpu}_{int(time.time())}")] for gpu in [0, 1]]
                    execute("preflight_evaluation", commands)
                    if report(preview, "smoke")["status"] != "complete":
                        raise RuntimeError("Saved-checkpoint held-out generation preflight failed.")
                    atomic_write_json(root / "frozen.json", {"implementation_hash": code_hash(), "frozen_hashes": frozen_hashes(root),
                        "frozen_utc": datetime.now(timezone.utc).isoformat(), "smoke_variants_passed": list(VARIANTS)}, ensure_ascii=False)
            for stage in ["smoke", "development", "confirmation"]:
                commands = [[str(Path(sys.executable).parent / "mllm-comp"), "analyze", "--config", str(root / "configs" / f"{stage}_{gpu}.yaml"),
                             "--device", f"cuda:{gpu}", "--output_dir", str(root / "lifecycle" / f"{stage}_{gpu}_{int(time.time())}")] for gpu in [0, 1]]
                execute(f"eval_{stage}", commands)
                result = report(root, stage)
                if result["status"] != "complete":
                    raise RuntimeError("Incomplete held-out evaluation.")
            (root / "final_report.md").write_text((root / "confirmation_report.md").read_text(), encoding="utf-8")
            manifest["status"] = "complete"
        except BaseException as exc:
            manifest.update({"status": "failed", "error": str(exc), "traceback": traceback.format_exc()})
            for process, _, _ in children:
                if process.poll() is None:
                    process.terminate()
            for process, _, _ in children:
                process.wait()
            raise
        finally:
            for _, stream, _ in children:
                stream.close()
            manifest["updated_utc"] = datetime.now(timezone.utc).isoformat()
            atomic_write_json(root / "run_manifest.json", manifest, ensure_ascii=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--launch", action="store_true")
    args = parser.parse_args()
    if args.launch:
        with (args.root.resolve() / "supervisor.log").open("a") as stream:
            child = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), "--root", str(args.root.resolve())],
                cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
        print(json.dumps({"supervisor_pid": child.pid, "root": str(args.root.resolve())}))
    else:
        run(args.root)
