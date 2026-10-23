"""Detached two-GPU smoke/development/confirmation execution with frozen gates."""

import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).parent))
from mllm_comp.analysis.probes.visual_reconstruction_utility import implementation_hash
from mllm_comp.core.serialization import atomic_write_json
from report_reconstruction_utility import report


def analysis_hashes(root):
    paths = [Path(__file__).resolve(), Path(__file__).with_name("report_reconstruction_utility.py"),
             Path(__file__).with_name("prepare_reconstruction_utility.py"), root / "preregistration.json",
             ROOT / "mllm_comp/registry/catalog.py", ROOT / "mllm_comp/analysis/probes/__init__.py"]
    paths.extend(sorted((root / "configs").glob("*.yaml")))
    return {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}


def run(root, stages):
    root = root.resolve()
    logs = root / "logs"
    logs.mkdir(exist_ok=True)
    with (root / "runner.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        manifest = {"status": "running", "pid": os.getpid(), "stages": stages,
                    "started_utc": datetime.now(timezone.utc).isoformat(), "platform": platform.platform(), "python": sys.version,
                    "implementation_hash": implementation_hash(), "analysis_hashes": analysis_hashes(root),
                    "cohort_hash": hashlib.sha256((root / "cohort.json").read_bytes()).hexdigest(),
                    "software": {key: importlib.metadata.version(key) for key in ["torch", "transformers", "datasets", "numpy"]},
                    "gpu": subprocess.check_output(["nvidia-smi", "--query-gpu=index,name,uuid,memory.total,driver_version", "--format=csv"], text=True),
                    "stage_status": {}}
        children = []
        try:
            for stage in stages:
                manifest["active_stage"] = stage
                if stage == "confirmation":
                    frozen = json.loads((root / "frozen.json").read_text())
                    if any(frozen[key] != manifest[key] for key in ["implementation_hash", "analysis_hashes", "cohort_hash"]):
                        raise RuntimeError("Frozen implementation/statistics/cohort changed.")
                children = []
                attempt = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
                for gpu in [0, 1]:
                    path = logs / f"{stage}_{gpu}_{attempt}.log"
                    stream = path.open("w")
                    command = [str(Path(sys.executable).parent / "mllm-comp"), "analyze", "--config", str(root / "configs" / f"{stage}_{gpu}.yaml"),
                               "--device", f"cuda:{gpu}", "--output_dir", str(root / "lifecycle" / f"{stage}_{gpu}_{attempt}")]
                    child = subprocess.Popen(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT,
                        env={**os.environ, "PYTHONUNBUFFERED": "1", "OMP_NUM_THREADS": "4", "TOKENIZERS_PARALLELISM": "false"})
                    children.append((child, stream, str(path)))
                manifest["stage_status"][stage] = {"status": "running", "children": [{"pid": p.pid, "log": log} for p, _, log in children]}
                atomic_write_json(root / "run_manifest.json", manifest, ensure_ascii=False)
                while any(p.poll() is None for p, _, _ in children):
                    if any(p.poll() not in {None, 0} for p, _, _ in children):
                        raise RuntimeError(f"{stage} failed; inspect worker logs.")
                    time.sleep(5)
                if any(p.returncode != 0 for p, _, _ in children):
                    raise RuntimeError(f"{stage} worker failed.")
                for _, stream, _ in children:
                    stream.close()
                result = report(root, stage)
                if result["status"] != "complete" or result["implementation_hash"] != implementation_hash():
                    raise RuntimeError("Incomplete stage or changed implementation.")
                if manifest["analysis_hashes"] != analysis_hashes(root):
                    raise RuntimeError("Statistical script, config, or preregistration changed during execution.")
                if stage == "smoke":
                    summaries = [json.loads(path.read_text()) for path in (root / "samples/smoke").glob("*/summary.json")]
                    if len(summaries) != 16 or any(s["native_generation_identity_passed"] is not True or not (s["cache_free_identity_passed"] is True or s["cache_parity_fp32_reference_passed"] is True) for s in summaries):
                        raise RuntimeError("Real Qwen native/cache smoke parity failed.")
                if stage == "development":
                    freeze = {key: manifest[key] for key in ["implementation_hash", "analysis_hashes", "cohort_hash"]}
                    freeze.update({"development_images": result["completed_images"], "frozen_utc": datetime.now(timezone.utc).isoformat(),
                                   "estimated_confirmation_wall_hours": result["elapsed_s_sum"] * 4 / 2 / 3600})
                    atomic_write_json(root / "frozen.json", freeze, ensure_ascii=False)
                manifest["stage_status"][stage].update({"status": "complete", "completed_images": result["completed_images"]})
                atomic_write_json(root / "run_manifest.json", manifest, ensure_ascii=False)
            manifest["status"] = "complete"
            if (root / "confirmation_report.md").exists():
                (root / "final_report.md").write_text((root / "confirmation_report.md").read_text(), encoding="utf-8")
        except BaseException as exc:
            manifest.update({"status": "failed", "error": str(exc), "traceback": traceback.format_exc()})
            for child, _, _ in children:
                if child.poll() is None:
                    child.terminate()
            for child, _, _ in children:
                child.wait()
            if manifest.get("active_stage") in manifest["stage_status"]:
                manifest["stage_status"][manifest["active_stage"]]["status"] = "failed"
            raise
        finally:
            for _, stream, _ in children:
                stream.close()
            manifest["updated_utc"] = datetime.now(timezone.utc).isoformat()
            atomic_write_json(root / "run_manifest.json", manifest, ensure_ascii=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--stages", nargs="+", default=["smoke", "development", "confirmation"])
    parser.add_argument("--launch", action="store_true")
    args = parser.parse_args()
    if args.launch:
        with (args.root.resolve() / "supervisor.log").open("a") as log:
            child = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), "--root", str(args.root.resolve()), "--stages", *args.stages],
                cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        print(json.dumps({"supervisor_pid": child.pid, "root": str(args.root.resolve())}))
    else:
        run(args.root, args.stages)
