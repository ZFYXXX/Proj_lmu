"""Plan, smoke, run, prepare a manual job and inspect the portable queue."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import signal
from string import Template
import subprocess
import sys
import time

from .catalog import DEFAULT_CATALOG, ROOT, load_catalog, load_tasks


def read(path):
    return json.loads(Path(path).read_text())


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, data):
    path = Path(path)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n")
    os.replace(temporary, path)


def configuration(profile_path, experiment_path, *, catalog_path=None, methods=None, tasks=None):
    profile, experiment = read(profile_path), read(experiment_path)
    if profile.get("schema_version") != 1 or experiment.get("schema_version") != 1:
        raise ValueError("Profile and experiment require schema_version=1")
    catalog_path = Path(catalog_path or DEFAULT_CATALOG).resolve()
    catalog, known_tasks = load_catalog(catalog_path), load_tasks()
    mapping = experiment.get("method_tasks", {})
    if not isinstance(mapping, dict) or not mapping:
        raise ValueError("Experiment needs a nonempty method_tasks mapping")
    for method, members in mapping.items():
        if method not in catalog or not isinstance(members, list) or not members:
            raise ValueError(f"Invalid method/tasks: {method}")
        if any(not isinstance(task, str) for task in members) or len(set(members)) != len(members) or set(members) - known_tasks.keys():
            raise ValueError(f"Unknown or duplicate tasks: {method}")
    if methods and set(methods) - mapping.keys():
        raise ValueError("Method filter contains a method absent from experiment")
    if tasks and set(tasks) - {task for members in mapping.values() for task in members}:
        raise ValueError("Task filter contains a task absent from experiment")
    cells = [[method, task] for method, members in mapping.items() for task in members
             if (not methods or method in methods) and (not tasks or task in tasks)]
    if not cells:
        raise ValueError("Selection contains no experiment cells")
    if not isinstance(experiment.get("reference_only", True), bool):
        raise ValueError("reference_only must be a boolean")
    reference_only = experiment.get("reference_only", True)
    if reference_only and any(method == "dense" for method, _ in cells):
        raise ValueError("Dense is scheduled in a reference-only experiment")
    if not reference_only:
        dense_tasks = {task for method, task in cells if method == "dense"}
        if {task for _, task in cells} - dense_tasks:
            raise ValueError("Each task requires a dense cell, or set reference_only=true")
    run_id = experiment.get("run_id", Path(experiment_path).stem)
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", run_id) or run_id in {".", ".."}:
        raise ValueError("run_id must be a single directory name")
    chunk_size = experiment.get("chunk_size", 1024)
    if type(chunk_size) is not int or chunk_size < 1:
        raise ValueError("chunk_size must be a positive integer")
    def expand(value):
        if isinstance(value, dict):
            return {key: expand(item) for key, item in value.items()}
        if isinstance(value, list):
            return [expand(item) for item in value]
        return Template(value).substitute(REPO=str(ROOT)) if isinstance(value, str) else value
    profile = expand(profile)
    for key in ("python", "data_root", "model", "checkpoint_root", "output_root", "cache_dir"):
        if not isinstance(profile.get(key), str) or not Path(profile[key]).is_absolute():
            raise ValueError(f"Machine profile needs an absolute {key}")
    roles = dict(profile.get("checkpoints", {}))
    checkpoint_root = Path(profile["checkpoint_root"])
    selected = list(dict.fromkeys(method for method, _ in cells))
    checkpoints = {method: roles.get(catalog[method]["checkpoint"], str(checkpoint_root / "checkpoints" / catalog[method]["checkpoint"]))
                   for method in selected if catalog[method].get("checkpoint")}
    scorer_roles = {catalog[method]["scorer"] for method in selected if catalog[method].get("scorer")}
    if len(scorer_roles) > 1:
        raise ValueError("This queue supports one scorer checkpoint per experiment")
    scorer = next(iter(scorer_roles), None)
    return {"profile": profile, "run_id": run_id, "requested_cells": cells, "methods": selected,
            "tasks": list(dict.fromkeys(task for _, task in cells)), "checkpoints": checkpoints,
            "scorer_checkpoint": roles.get(scorer, str(checkpoint_root / "checkpoints" / scorer)) if scorer else None,
            "identity": profile.get("identity", str(checkpoint_root / "model_identity.json")),
            "catalog_path": str(catalog_path), "catalog_sha256": digest(catalog_path),
            "reference_only": reference_only, "chunk_size": chunk_size,
            "experiment": {"schema_version": 1, "method_tasks": {
                method: [task for m, task in cells if m == method] for method in selected}}}


def check_paths(config, *, include_reuse=True):
    profile = config["profile"]
    required = [Path(profile["python"]), Path(profile["data_root"]), Path(profile["model"]) / "config.json"]
    if config["checkpoints"]:
        required.append(Path(config["identity"]))
    for directory in [*config["checkpoints"].values(), config["scorer_checkpoint"]]:
        if directory:
            folder = Path(directory)
            required.append(folder)
            if not list(folder.glob("*.safetensors")) or not list(folder.glob("*_config.json")):
                raise ValueError(f"Missing checkpoint config/weights: {directory}")
    if include_reuse:
        required.extend(Path(path) for path in profile.get("seed_from", []))
    if include_reuse and profile.get("result_bundle"):
        required.append(Path(profile["result_bundle"]))
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise ValueError("Missing paths:\n" + "\n".join(missing))
    task_catalog = load_tasks()
    data_root = Path(profile["data_root"])
    for task in config["tasks"]:
        sources = list(data_root.glob(task_catalog[task]["pattern"]))
        if task == "mmstar" and not sources:
            for pattern in ("MMStar/data/validation-*.parquet", "MMStar/data/val-*.parquet", "MMStar/data/test-*.parquet"):
                sources.extend(data_root.glob(pattern))
            extra = profile.get("environment", {}).get("MLLM_MMSTAR_PARQUET")
            if not sources and extra:
                sources.append(Path(extra))
        if task == "hallusionbench" and not sources and profile.get("hallusion_image"):
            sources.append(Path(profile["hallusion_image"]))
        if not sources or any(not path.is_file() for path in sources):
            raise ValueError(f"Dataset sources missing for {task}: {task_catalog[task]['pattern']}")


def runtime_files(root=None):
    root = root or ROOT
    for directory in ("mllm_comp", "scripts", "configs", "requirements", "constraints"):
        for path in sorted((root / directory).rglob("*")):
            if path.is_file() and not any(part in {"__pycache__", ".pytest_cache"} for part in path.parts) and path.suffix != ".pyc":
                yield path.relative_to(root)
    yield Path("pyproject.toml")


def freeze(config, output, *, limit=None, resume=False):
    """Record dirty source by content and execute only the copied source tree."""
    output = Path(output).resolve()
    metadata = output / "workflow.json"
    hashes = {str(path): digest(ROOT / path) for path in runtime_files()}
    input_hashes = {}
    for path in config.get('extra_input_files', []):
        input_hashes[str(path)] = digest(path)
    for directory in {*config["checkpoints"].values(), config["scorer_checkpoint"]} - {None}:
        for path in sorted(Path(directory).glob("*")):
            if path.is_file() and (path.suffix == ".safetensors" or path.name.endswith("_config.json")):
                input_hashes[str(path)] = digest(path)
    if config["checkpoints"]:
        input_hashes[config["identity"]] = digest(config["identity"])
    bundle = config["profile"].get("result_bundle") if limit is None else None
    if bundle:
        input_hashes[bundle] = digest(bundle)
    signature = hashlib.sha256(json.dumps({"config": config, "source": hashes, "inputs": input_hashes, "limit": limit}, sort_keys=True).encode()).hexdigest()
    if output.exists():
        if not resume or not metadata.is_file() or read(metadata)["signature"] != signature:
            raise ValueError("Output exists or source/config changed; use --resume with the same version or choose a new run_id")
        saved = read(metadata)
        for name, expected in saved["source_hashes"].items():
            if digest(output / "source" / name) != expected:
                raise ValueError(f"Frozen source was modified: {name}")
        return saved
    output.mkdir(parents=True)
    (output / "logs").mkdir()
    snapshot = output / "source"
    for name in hashes:
        target = snapshot / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / name, target)
    # Custom catalogs outside the repository are pinned into this snapshot too.
    shutil.copy2(config["catalog_path"], snapshot / "configs/workflow/selected_methods.json")
    write(snapshot / "configs/workflow/selected_experiment.json", config["experiment"])
    if bundle:
        (snapshot / "artifacts/eval").mkdir(parents=True)
        shutil.copy2(bundle, snapshot / "artifacts/eval/reuse.json.gz")
    git = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True)
    dirty = subprocess.run(["git", "status", "--short"], cwd=ROOT, capture_output=True, text=True)
    saved = {"schema_version": 1, "signature": signature, "config": config, "source_hashes": hashes,
             "git_commit": git.stdout.strip(), "git_status": dirty.stdout.splitlines(), "limit": limit,
             "input_hashes": input_hashes, "result_bundle": str(snapshot / "artifacts/eval/reuse.json.gz") if bundle else None,
             "created_utc": datetime.now(timezone.utc).isoformat(), "source_root": str(snapshot),
             "python": config["profile"]["python"], "status": "prepared"}
    write(metadata, saved)
    return saved


def queue_command(saved, output, index, workers):
    config, source = saved["config"], Path(saved["source_root"])
    profile = config["profile"]
    command = [profile["python"], "-u", str(source / "scripts/eval/lmms_eval/qwen25_portable_queue.py"),
               "--data-root", profile["data_root"], "--model", profile["model"], "--output", str(output),
               "--cache-dir", profile["cache_dir"], "--experiment", str(source / "configs/workflow/selected_experiment.json"),
               "--method-catalog", str(source / "configs/workflow/selected_methods.json"), "--resident-model", "--resume",
               "--gpu", "0", "--shared-workers", str(workers), "--worker-index", str(index),
               "--chunk-size", str(config["chunk_size"]), "--mmmu-single-image-only"]
    if config["checkpoints"]:
        command += ["--reference-identity", str(Path(output) / "shared/model_identity.json")]
    for method, checkpoint in config["checkpoints"].items():
        command += ["--checkpoint", f"{method}={checkpoint}"]
    if config["scorer_checkpoint"]:
        command += ["--scorer-checkpoint", config["scorer_checkpoint"]]
    if profile.get("hallusion_image"):
        command += ["--hallusion-image", profile["hallusion_image"]]
    if config["reference_only"]:
        command += ["--reference-only"]
    if saved["limit"] is not None:
        command += ["--limit", str(saved["limit"])]
    else:
        if saved.get("result_bundle"):
            command += ["--result-bundle", saved["result_bundle"]]
        for prior in profile.get("seed_from", []):
            command += ["--seed-from", prior]
    return command


def slurm_script(saved, output):
    profile, root = saved["config"]["profile"], Path(saved["source_root"])
    spec = profile.get("slurm", {})
    if not re.fullmatch(r"28-[A-Za-z0-9_-]+", spec.get("job_name", "")):
        raise ValueError("LMU job_name must start with 28-")
    for key in ("account", "partition", "time"):
        if not re.fullmatch(r"[A-Za-z0-9_:-]+", str(spec.get(key, ""))):
            raise ValueError(f"Invalid Slurm field: {key}")
    parts = [int(value) for value in spec["time"].split(":")]
    if len(parts) != 3 or not 0 < parts[0] * 3600 + parts[1] * 60 + parts[2] < 86400:
        raise ValueError("Use a Slurm time below 24 hours, HH:MM:SS")
    if spec.get("gpus") != 4 or type(spec.get("cpus")) is not int or spec["cpus"] < 1:
        raise ValueError("LMU profile requires four GPUs and positive CPUs")
    return "\n".join(["#!/usr/bin/env bash", f"#SBATCH --job-name={spec['job_name']}",
        f"#SBATCH --account={spec['account']}", f"#SBATCH --partition={spec['partition']}",
        "#SBATCH --nodes=1", "#SBATCH --ntasks=1", "#SBATCH --gres=gpu:4",
        f"#SBATCH --cpus-per-task={spec['cpus']}", f"#SBATCH --time={spec['time']}",
        "#SBATCH --signal=B:TERM@120", '#SBATCH --mail-type="NONE"',
        f"#SBATCH --output={shlex.quote(str(Path(output) / 'logs/slurm_%j_output.log'))}",
        f"#SBATCH --error={shlex.quote(str(Path(output) / 'logs/slurm_%j_error.log'))}",
        "set -euo pipefail", "export WANDB_MODE=offline HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1",
        f"cd {shlex.quote(str(root))}", shlex.join([profile["python"], str(root / "scripts/workflow.py"),
            "execute", "--run-dir", str(output)]), ""])


def execute(output):
    output = Path(output).resolve()
    saved = read(output / "workflow.json")
    config, source = saved["config"], Path(saved["source_root"])
    for name, expected in saved["source_hashes"].items():
        if digest(source / name) != expected:
            raise ValueError(f"Frozen source changed: {name}")
    if digest(source / "configs/workflow/selected_methods.json") != config["catalog_sha256"] or read(source / "configs/workflow/selected_experiment.json") != config["experiment"]:
        raise ValueError("Frozen method/experiment config changed")
    for path, expected in saved.get("input_hashes", {}).items():
        # The portable evidence bundle has its own immutable copied file.
        if path == config["profile"].get("result_bundle"):
            path = saved["result_bundle"]
        if digest(path) != expected:
            raise ValueError(f"Checkpoint or result bundle changed: {path}")
    profile = config["profile"]
    devices = (os.environ["CUDA_VISIBLE_DEVICES"].split(",") if os.environ.get("SLURM_JOB_ID")
               and os.environ.get("CUDA_VISIBLE_DEVICES") else profile.get("gpus", ["0"]))
    if not devices or len(set(devices)) != len(devices):
        raise ValueError("Expected distinct CUDA devices")
    if os.environ.get("SLURM_JOB_ID") and len(devices) != 4:
        raise ValueError("LMU execution requires four allocated CUDA devices")
    if not os.environ.get("SLURM_JOB_ID"):
        selected = subprocess.check_output(["nvidia-smi", "--id=" + ",".join(devices),
            "--query-gpu=uuid", "--format=csv,noheader"], text=True).splitlines()
        processes = subprocess.check_output(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid",
                                             "--format=csv,noheader"], text=True).splitlines()
        if any(row.split(",")[0].strip() in selected for row in processes):
            raise ValueError("Selected local GPU has active compute processes; choose a free GPU in the profile")
    environment = {**os.environ, **profile.get("environment", {}), "PYTHONPATH": str(source),
                   "HF_HUB_OFFLINE": "1", "HF_DATASETS_OFFLINE": "1", "WANDB_MODE": "offline",
                   "OMP_NUM_THREADS": "2", "MKL_NUM_THREADS": "2", "TOKENIZERS_PARALLELISM": "false"}
    if config["checkpoints"]:
        (output / "shared").mkdir(exist_ok=True)
        script = "from pathlib import Path; from scripts.eval.lmms_eval.qwen25_portable_queue import resolve_identity; import sys; resolve_identity(*map(Path,sys.argv[1:]))"
        subprocess.run([profile["python"], "-c", script, config["identity"], profile["model"],
                        str(output / "shared")], cwd=source, env=environment, check=True)
    workers, streams, interrupted = [], [], []
    def stop(number, _frame):
        interrupted.append(number)
    previous = {number: signal.signal(number, stop) for number in (signal.SIGTERM, signal.SIGINT)}
    monitor = None
    started = time.monotonic()
    try:
        gpu_log = (output / "logs/gpu_usage.log").open("a")
        streams.append(gpu_log)
        if shutil.which("nvidia-smi"):
            monitor = subprocess.Popen(["nvidia-smi", "--query-gpu=timestamp,index,uuid,memory.used,utilization.gpu",
                "--format=csv,noheader", "-l", "60"], stdout=gpu_log, stderr=subprocess.STDOUT, start_new_session=True)
        for index, device in enumerate(devices):
            stream = (output / f"logs/worker_{index}.log").open("a")
            streams.append(stream)
            workers.append(subprocess.Popen(queue_command(saved, output, index, len(devices)), cwd=source,
                env={**environment, "CUDA_VISIBLE_DEVICES": device}, stdout=stream, stderr=subprocess.STDOUT,
                start_new_session=True))
        while any(worker.poll() is None for worker in workers):
            if interrupted or any(worker.poll() not in (None, 0) for worker in workers):
                break
            time.sleep(1)
    finally:
        for process in [*workers, monitor]:
            if process and process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
        for process in [*workers, monitor]:
            if process:
                try:
                    process.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
        for stream in streams:
            stream.close()
        for number, handler in previous.items():
            signal.signal(number, handler)
    state = inspect_run(output)
    state.update(worker_exit_codes=[worker.returncode for worker in workers], elapsed_seconds=time.monotonic() - started,
                 slurm_job_id=os.environ.get("SLURM_JOB_ID"))
    state["status"] = ("interrupted" if interrupted else "complete" if state["completed"] == state["expected"]
                       and all(worker.returncode == 0 for worker in workers) else "failed")
    write(output / "workflow_status.json", state)
    print(json.dumps(state, ensure_ascii=False, indent=2))
    return 128 + interrupted[0] if interrupted else int(state["status"] != "complete")


def inspect_run(output):
    output = Path(output)
    metadata = read(output / "workflow.json") if (output / "workflow.json").exists() else None
    if metadata and "training" in metadata["config"]:
        state = read(output / "training_status.json") if (output / "training_status.json").exists() else {"status": "prepared"}
        arms = {}
        for name in metadata["config"]["training"]["arms"]:
            path = output / "arms" / name / "train/manifest.json"
            arms[name] = read(path) if path.exists() else {"status": "pending", "step": 0}
            arms[name]["evaluation_complete"] = (output / "arms" / name / "complete.json").exists()
        return {"run_dir": str(output), **state, "arms": arms}
    cells = metadata["config"]["requested_cells"] if metadata else read(output / "plan.json")["requested_cells"]
    done = [cell for cell in cells if (output / "cells" / cell[0] / cell[1] / "complete.json").is_file()]
    failures = [str(path) for path in output.glob("cells/*/*/failed.json")]
    return {"run_dir": str(output), "completed": len(done), "expected": len(cells), "completed_cells": done,
            "pending_cells": [cell for cell in cells if cell not in done], "failures": failures,
            "chunks_completed": len(list(output.glob("cells/*/*/chunk_*/complete.json"))),
            "report": str(output / "report.md"), "status_file": str(output / "workflow_status.json")}


def main(argv=None):
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("action", choices=("list", "plan", "check", "smoke", "run", "prepare", "status", "report", "execute", "train", "train-plan"))
    parser.add_argument("--profile", type=Path, default=ROOT / "configs/workflow/profiles/p6000.json")
    parser.add_argument("--experiment", type=Path, default=ROOT / "configs/workflow/experiments/r1_t2_general.json")
    parser.add_argument("--method-catalog", type=Path)
    parser.add_argument("--methods", nargs="+")
    parser.add_argument("--tasks", nargs="+")
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--run-id")
    parser.add_argument("--limit", type=int, default=2, help="Smoke samples per task; full run has no limit")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--training-config", type=Path, default=ROOT / "configs/workflow/training/anchor_adaptation.json")
    parser.add_argument("--launch", action="store_true", help="Launch a persistent local training supervisor")
    parser.add_argument("--wait-for-gpu", action="store_true", help="Queue training until a selected local GPU is free")
    args = parser.parse_args(argv)
    try:
        if args.action in {"train", "train-plan"}:
            from .anchor_training import start
            return start(args)
        if args.action == "list":
            print(json.dumps({"methods": load_catalog(args.method_catalog), "tasks": load_tasks()}, ensure_ascii=False, indent=2))
            return 0
        if args.action in {"execute", "status", "report"}:
            if not args.run_dir:
                raise ValueError("This action requires --run-dir")
            if args.action == "execute":
                return execute(args.run_dir)
            if args.action == "status":
                print(json.dumps(inspect_run(args.run_dir), ensure_ascii=False, indent=2))
            else:
                print((args.run_dir / "report.md").read_text())
            return 0
        config = configuration(args.profile, args.experiment, catalog_path=args.method_catalog, methods=args.methods, tasks=args.tasks)
        if args.run_id:
            if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", args.run_id):
                raise ValueError("Invalid run_id")
            config["run_id"] = args.run_id
        output = (args.run_dir or Path(config["profile"]["output_root"]) / config["run_id"]).resolve()
        if args.action == "plan":
            print(json.dumps({"output": str(output), **config}, ensure_ascii=False, indent=2))
            return 0
        check_paths(config, include_reuse=args.action != "smoke")
        if args.action == "check":
            print(f"Paths and selection passed: {len(config['requested_cells'])} cells. No GPU inference started.")
            return 0
        if args.action == "prepare":
            # Only write a job script; the user submits it manually on LMU.
            slurm_script({"config": config, "source_root": "preview"}, output)
        limit = args.limit if args.action == "smoke" else None
        if limit is not None and limit < 1:
            raise ValueError("Smoke limit must be positive")
        if args.action == "smoke" and not args.run_dir:
            output = output.with_name(output.name + "_smoke_" + datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S"))
        saved = freeze(config, output, limit=limit, resume=args.resume)
        if args.action == "prepare":
            job = output / "job.sbatch"
            job.write_text(slurm_script(saved, output))
            subprocess.run(["bash", "-n", str(job)], check=True)
            write(output / "prepared_job.json", {"job_script": str(job), "run_dir": str(output),
                                                 "signature": saved["signature"], "submitted": False})
            print(f"Prepared job script: {job}\nSubmit manually on LMU: {shlex.join(['sbatch', str(job)])}")
            return 0
        # Explicit interpreter avoids activating/changing the user's shared env.
        return subprocess.call([saved["python"], str(Path(saved["source_root"]) / "scripts/workflow.py"),
                                "execute", "--run-dir", str(output)])
    except (ValueError, OSError, KeyError, subprocess.CalledProcessError) as exc:
        if args.action == "execute" and args.run_dir and args.run_dir.is_dir():
            write(args.run_dir / "workflow_status.json", {"status": "failed", "error": str(exc)})
        print(f"workflow: {exc}", file=sys.stderr)
        return 2
