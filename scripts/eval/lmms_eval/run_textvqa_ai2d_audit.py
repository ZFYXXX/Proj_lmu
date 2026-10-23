"""Background recovery of held-out eval plus an independently gated AI2D run."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
import traceback

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from mllm_comp.core.serialization import atomic_write_json
from scripts.train.qwen25_vl.run_compression_training_check import report as report_anchor


def prepare_recovery(root, old):
    import yaml
    if (root / "protocol.json").exists():
        return
    root.mkdir(parents=True, exist_ok=True)
    (root / "configs").mkdir(exist_ok=True)
    source = json.loads((old / "protocol.json").read_text())["source"]
    atomic_write_json(root / "protocol.json", {"source": source, "training_run": str(old), "training_rerun": False,
        "recovery_reason": "Installed analysis CLI could not import scripts namespace. Ledger helpers moved to package namespace; model/positions/generation unchanged.",
        "scope": "New evaluation artifacts only; preserve failed old run and completed training checkpoints"})
    atomic_write_json(root / "cohort.json", json.loads((old / "cohort.json").read_text()))
    for stage in ["smoke", "development", "confirmation"]:
        for shard in [0,1]:
            config = yaml.safe_load((old / "configs" / f"{stage}_{shard}.yaml").read_text())
            config["analysis"]["sampling"]["manifest_path"] = str(root / "cohort.json")
            probe = config["analysis"]["probes"][0]
            probe["artifact_root"] = str(root)
            probe["cohort_path"] = str(root / "cohort.json")
            (root / "configs" / f"{stage}_{shard}.yaml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")


def run(root, ai2d, old):
    prepare_recovery(root / "recovered_anchor_eval", old)
    recovery = root / "recovered_anchor_eval"
    logs = root / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    with (root / "runner.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        freeze_paths = [Path(__file__), ROOT / "scripts/eval/lmms_eval/run_ai2d_compression_check.py",
            ROOT / "mllm_comp/analysis/probes/anchor_training_evaluation.py", ROOT / "mllm_comp/analysis/common/condition_ledger.py",
            ai2d / "protocol.json", ai2d / "samples.json"] + sorted((recovery / "configs").glob("*.yaml"))
        def hashes():
            return {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in freeze_paths}
        manifest = {"status": "running", "pid": os.getpid(), "started_utc": datetime.now(timezone.utc).isoformat(),
                    "stages": {}, "frozen_hashes": hashes(), "ai2d_root": str(ai2d), "recovery_root": str(recovery)}
        mutex, stop, smoke_passed = threading.Lock(), threading.Event(), threading.Event()
        def execute(name, command):
            if stop.is_set() or hashes() != manifest["frozen_hashes"]:
                raise RuntimeError("Stopped or frozen code/config changed")
            path = logs / f"{name}_{int(time.time())}.log"
            with path.open("w") as stream:
                child = subprocess.Popen(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT,
                    env={**os.environ, "PYTHONUNBUFFERED": "1", "OMP_NUM_THREADS": "4", "TOKENIZERS_PARALLELISM": "false"})
                with mutex:
                    manifest["stages"][name] = {"status": "running", "pid": child.pid, "log": str(path)}
                    atomic_write_json(root / "run_manifest.json", manifest)
                while child.poll() is None:
                    if stop.wait(2):
                        child.terminate()
                        child.wait()
                        raise RuntimeError("Stopped owned child after sibling failure")
                with mutex:
                    manifest["stages"][name]["status"] = "complete" if child.returncode == 0 else "failed"
                    atomic_write_json(root / "run_manifest.json", manifest)
                if child.returncode:
                    raise RuntimeError(f"{name} failed: {path}")
        def ai_command(stage, shard, gpu):
            return [sys.executable, str(ROOT / "scripts/eval/lmms_eval/run_ai2d_compression_check.py"),
                    "--root", str(ai2d), "--stage", stage, "--shard", str(shard), "--device", f"cuda:{gpu}"]
        def lane0():
            for stage in ["smoke", "development", "confirmation"]:
                for shard in [0,1]:
                    execute(f"recover_{stage}_{shard}", [str(Path(sys.executable).parent / "mllm-comp"), "analyze",
                        "--config", str(recovery / "configs" / f"{stage}_{shard}.yaml"), "--device", "cuda:0",
                        "--output_dir", str(recovery / "lifecycle" / f"{stage}_{shard}_{int(time.time())}")])
                report_anchor(recovery, stage)
            while not smoke_passed.wait(2):
                if stop.is_set():
                    raise RuntimeError("AI2D smoke failed")
            execute("ai2d_full_0", ai_command("full", 0, 0))
        def lane1():
            for shard in [0,1]:
                execute(f"ai2d_smoke_{shard}", ai_command("smoke", shard, 1))
            execute("ai2d_smoke_report", [sys.executable, str(ROOT / "scripts/eval/lmms_eval/run_ai2d_compression_check.py"),
                "--root", str(ai2d), "--stage", "smoke", "--report"])
            smoke_passed.set()
            execute("ai2d_full_1", ai_command("full", 1, 1))
        def guarded(fn):
            try:
                fn()
            except BaseException:
                stop.set()
                raise
        try:
            with ThreadPoolExecutor(max_workers=2) as pool:
                futures = [pool.submit(guarded, fn) for fn in [lane0, lane1]]
                for future in futures:
                    future.result()
            execute("ai2d_full_report", [sys.executable, str(ROOT / "scripts/eval/lmms_eval/run_ai2d_compression_check.py"),
                "--root", str(ai2d), "--stage", "full", "--report"])
            manifest["status"] = "complete"
        except BaseException as exc:
            stop.set()
            manifest.update({"status": "failed", "error": str(exc), "traceback": traceback.format_exc()})
            raise
        finally:
            manifest["updated_utc"] = datetime.now(timezone.utc).isoformat()
            atomic_write_json(root / "run_manifest.json", manifest)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--ai2d", type=Path, required=True)
    p.add_argument("--old", type=Path, required=True)
    p.add_argument("--launch", action="store_true")
    args = p.parse_args()
    root, ai2d, old = args.root.resolve(), args.ai2d.resolve(), args.old.resolve()
    root.mkdir(parents=True, exist_ok=True)
    if args.launch:
        with (root / "supervisor.log").open("a") as stream:
            child = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), "--root", str(root),
                "--ai2d", str(ai2d), "--old", str(old)], cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
        print(json.dumps({"supervisor_pid": child.pid, "root": str(root)}))
    else:
        run(root, ai2d, old)
