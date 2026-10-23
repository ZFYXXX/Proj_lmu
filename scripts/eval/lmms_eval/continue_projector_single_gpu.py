"""Resource-only handoff: preserve the active CLI worker, serialize the queue."""
import argparse
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from scripts.eval.lmms_eval.run_projector_ratio_platform import (
    count_and_check, now, run_name, validate,
)
from mllm_comp.analysis.common.anchor7b_runtime import file_sha
from mllm_comp.core.serialization import atomic_write_json


def alive(pid):
    try:
        return Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[0] != "Z"
    except FileNotFoundError:
        return False


def verify_worker(pid, config):
    if alive(pid):
        cmd = Path(f"/proc/{pid}/cmdline").read_bytes().replace(b"\0", b" ").decode()
        if str(config) not in cmd or "mllm_comp.cli.main" not in cmd:
            raise ValueError(f"Adopted process {pid} no longer matches its pinned configuration")


def run(root, gpu, interrupted_job):
    with (root / "runner.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        protocol = validate(root)
        state_path = root / "run_manifest.json"
        state = json.loads(state_path.read_text())
        prior_pid = state["pid"]
        if alive(prior_pid):
            raise ValueError("Stop the previous two-GPU supervisor before handing off")
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        old = state["jobs"][interrupted_job]
        if alive(old["pid"]):
            raise ValueError("Interrupted worker must have exited before archiving")
        archive = root / "interrupted" / stamp / interrupted_job
        archive.mkdir(parents=True, exist_ok=False)
        source = root / "full" / interrupted_job.removeprefix("full_")
        if source.exists():
            source.rename(archive / "output")
        old_log = Path(old["log"])
        if old_log.exists():
            old_log.rename(archive / "worker.log")
        state["jobs"][interrupted_job] = {
            "status": "queued", "gpu": gpu,
            "interrupted_attempt": {**old, "status": "interrupted_by_user_resource_request", "archive": str(archive)},
        }
        override = {"changed_utc": now(), "reason": "User requested freeing one RTX PRO GPU",
            "allowed_gpus": [gpu], "reserved_from_this_experiment": [1 - gpu],
            "previous_supervisor_pid": prior_pid, "supervisor_pid": os.getpid(),
            "scheduler_sha256": file_sha(Path(__file__)),
            "evaluation_protocol_changed": False,
            "interrupted_job": interrupted_job, "preserved_attempt": str(archive),
            "restart_policy": "Interrupted native eval is rerun on the remaining GPU; no partial scores relabeled as full"}
        atomic_write_json(root / "resource_override.json", override)
        state.update(pid=os.getpid(), status="running", allowed_gpus=[gpu], resource_handoff_utc=now())
        atomic_write_json(state_path, state)
        child = None
        try:
            for method in protocol["methods"]:
                name = run_name(method, protocol["target_count"])
                job = f"full_{name}"
                previous = state["jobs"].get(job, {})
                if previous.get("status") == "complete":
                    continue
                config = root / "configs" / f"{job}.yaml"
                validate(root)
                if previous.get("status") == "running":
                    if previous["gpu"] != gpu:
                        raise ValueError("A non-interrupted worker occupies the reserved GPU")
                    pid = previous["pid"]
                    verify_worker(pid, config)
                    while alive(pid):
                        verify_worker(pid, config)
                        time.sleep(10)
                else:
                    while True:
                        free = int(subprocess.check_output(["nvidia-smi", f"--id={gpu}",
                            "--query-gpu=memory.free", "--format=csv,noheader,nounits"], text=True).strip())
                        if free >= 40000:
                            break
                        time.sleep(15)
                    command = [sys.executable, "-m", "mllm_comp.cli.main", "eval", "--config", str(config)]
                    log = root / "logs" / f"{job}.log"
                    with log.open("w") as stream:
                        child = subprocess.Popen(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT,
                            start_new_session=True, env={**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu),
                                "PYTHONUNBUFFERED": "1", "OMP_NUM_THREADS": "4", "TOKENIZERS_PARALLELISM": "false"})
                        state["jobs"][job] = {**previous, "status": "running", "pid": child.pid, "gpu": gpu,
                            "command": command, "config": str(config), "log": str(log), "started_utc": now()}
                        atomic_write_json(state_path, state)
                        if child.wait():
                            raise RuntimeError(f"{job} failed; see {log}")
                checks = count_and_check(root / "full" / name, method, "full", protocol["target_count"])
                state["jobs"][job].update(status="complete", checks=checks, completed_utc=now())
                atomic_write_json(state_path, state)
            state.update(status="complete", completed_utc=now())
        except BaseException as exc:
            state.update(status="failed", error=str(exc), traceback=traceback.format_exc())
            raise
        finally:
            atomic_write_json(state_path, state)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--gpu", type=int, choices=(0, 1), required=True)
    parser.add_argument("--interrupted-job", required=True)
    args = parser.parse_args()
    run(args.root.resolve(), args.gpu, args.interrupted_job)
