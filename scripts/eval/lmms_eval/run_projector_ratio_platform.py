"""Schedule the unified mllm-comp eval CLI; this script has no data/model loop."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import fcntl
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
from mllm_comp.analysis.common.anchor7b_runtime import file_sha, validate_identity
from mllm_comp.core.serialization import atomic_write_json, atomic_write_text

TASK_DIR = ROOT / "configs/eval/lmms_eval/tasks/qwen25_7b_projector_ratio"
TASKS = {"qwen25_7b_ratio_chartqa": 2500, "qwen25_7b_ratio_ai2d": 3088,
         "qwen25_7b_ratio_ocrbench": 1000, "qwen25_7b_ratio_docvqa_val": 5349}
SOURCE = ROOT / "outputs/eval/qwen25_vl/anchor7b_ocrbench_docvqa_full_20261021_v1"
TRAINED = ROOT / "outputs/train/qwen25_vl/anchor_mlp_7b_k128_ddp_20261021_v1/model/checkpoints/step_000320"
METHODS = ("dense", "feature_anchor_adapter", "cdpruner", "dppruner", "eadp", "hiprune", "bifocal_stage1")


def now():
    return datetime.now(timezone.utc).isoformat()


def code_files():
    paths = [Path(__file__), ROOT / "mllm_comp/compressors/comparison/qwen_projector.py",
        ROOT / "mllm_comp/compressors/comparison/anchor_adapter.py",
        ROOT / "mllm_comp/compressors/comparison/registry.py",
        ROOT / "mllm_comp/integrations/lmms_eval/model_bridge.py", *TASK_DIR.glob("*.py"), *TASK_DIR.glob("*.yaml")]
    return {str(p.relative_to(ROOT)): file_sha(p) for p in paths}


def run_name(method, target_count=None, ratio=.25):
    if method == "dense":
        return method
    return f"{method}_k{target_count}" if target_count is not None else f"{method}_r{int(round(ratio * 100)):03d}"


def config(root, method, stage, ratio=.25, target_count=None):
    checkpoint = json.loads((SOURCE / "protocol.json").read_text())["checkpoint"]
    name = run_name(method, target_count, ratio)
    output = root / stage / name
    pipeline = []
    if method != "dense":
        algorithm = "bifocal" if method == "bifocal_stage1" else method
        cfg = {"tokens_per_crop": target_count} if target_count is not None else {"retention_ratio": ratio}
        if method == "feature_anchor_adapter":
            cfg.update(adapter_path=str(TRAINED), identity_path=str(SOURCE / "model_identity.json"))
        if method == "bifocal_stage1":
            if target_count is not None:
                raise ValueError("Bifocal Stage 1 has no exact fixed-count binding; exclude it from this matrix")
            cfg = {"s1_frac": ratio, "block": "auto", "coarse_frac": .29,
                "k_frac": .75, "clsr": "1", "respool": True, "respool_pos": False,
                "metric": "vit_attn", "metric_agg": "mass", "vit_block": 31}
        pipeline = [{"name": name, "enabled": True, "plugin": algorithm, "algorithm": algorithm,
            "variant": f"{algorithm}.image", "binding": f"{algorithm}.image.qwen25_vl",
            "placement": "projector_level", "method": "token_compression", "object_family": "tensor_slot",
            "execution_phase": "generate", "modalities": ["image"], "target_modalities": ["image"], "config": cfg}]
    return {"experiment": {"name": name, "seed": 42},
        "backbone": {"name": "qwen25_vl", "model_path": checkpoint,
            "model_revision": "cc594898137f460bfe9f0759e9844b3ce807cfb5", "trust_remote_code": False,
            "config": {"torch_dtype": "bfloat16", "attn_implementation": "eager",
                "model_load_kwargs": {"local_files_only": True},
                "processor_load_kwargs": {"local_files_only": True, "use_fast": False,
                    "min_pixels": 200704, "max_pixels": 1605632}}},
        "pipeline": {"stages": pipeline}, "analysis": {}, "train": {"compression_mode": "none"},
        "eval": {"backend": "lmms_eval", "compression_mode": "none" if method == "dense" else "compressed",
            "target": "baseline" if method == "dense" else "compressed", "checkpoint_path": checkpoint,
            "tasks": list(TASKS), "runtime_batch_size": 1, "num_processes": 1,
            "sample_limit": 8 if stage == "smoke" else None, "generation": {},
            "args": {"force_simple": True, "log_samples": True, "verbosity": "INFO", "include_path": str(TASK_DIR)}},
        "data": {"dataset_id": "local-parquet-four-full-splits", "sample_split": "task_defined"},
        "output": {"run_dir": str(output), "manifest_path": str(output / "manifest.json")}}


def prepare(root, target_count=None):
    import yaml
    root.mkdir(parents=True, exist_ok=True)
    if target_count is not None and (isinstance(target_count, bool) or target_count <= 0):
        raise ValueError("target_count must be a positive integer")
    if (root / "protocol.json").exists():
        protocol = validate(root)
        if protocol.get("target_count") != target_count:
            raise ValueError("Cannot change the frozen budget in an existing run")
        return protocol
    identity = json.loads((SOURCE / "model_identity.json").read_text())
    validate_identity(identity)
    configs = root / "configs"; configs.mkdir()
    files = {}
    methods = tuple(m for m in METHODS if target_count is None or m != "bifocal_stage1")
    for ratio in ((.25,) if target_count is not None else (.25, .50, .15)):
        for method in methods:
            if method == "dense" and ratio != .25:
                continue
            for stage in ("smoke", "full"):
                name = run_name(method, target_count, ratio)
                path = configs / f"{stage}_{name}.yaml"
                atomic_write_text(path, yaml.safe_dump(config(root, method, stage, ratio, target_count), allow_unicode=True, sort_keys=False))
                files[str(path)] = file_sha(path)
    protocol = {"created_utc": now(), "entry": "python -m mllm_comp.cli.main eval --config <yaml>",
        "target_count": target_count,
        "primary_retention_ratio": .25 if target_count is None else None,
        "prepared_additional_ratios": [.50, .15] if target_count is None else [],
        "launched_ratios": [.25] if target_count is None else [], "methods": methods, "tasks": TASKS,
        "base_identity_sha256": identity["identity_sha256"], "identity_path": str(SOURCE / "model_identity.json"),
        "adapter_sha256": file_sha(TRAINED / "projector.safetensors"),
        "budget_definition": "Per-image sum of visual tokens across decoder layers / (original post-merger tokens * decoder depth)",
        "rounding": "K_i=max(1,min(N_i,round(rho*N_i))); report actual per-image counts and ratios",
        "generation": "Task YAML plus native platform Qwen wrapper; checkpoint repetition_penalty=1.05 inherited, tokenizer EOS",
        "no_old_prediction_reuse": "Generation and budgets changed; rerun every method, including dense and trained MLP",
        "bifocal_scope": "Projector Stage 1 only, explicitly not the complete two-stage Bifocal paper result",
        "mlp_scope": "Existing K128-trained MLP evaluated at varying K; no retraining; disclose budget generalization",
        "ai2d_metrics": "Native exact_match plus separately labeled post-hoc leading-option sensitivity metric",
        "implementation_files": code_files(), "configs": files}
    atomic_write_json(root / "protocol.json", protocol)
    if target_count is not None:
        protocol.update(
            budget_definition="Fixed post-merger output budget; native image-dependent grid and resolution remain unchanged. No Bifocal layer-average budget target.",
            rounding=f"K_i=min(N_i,{target_count}); never duplicate tokens when N_i is smaller",
            bifocal_scope="Excluded: platform Stage 1 binding cannot express the exact fixed-count policy",
            mlp_scope=f"Existing K128-trained MLP; evaluated at K{target_count}; no retraining",
            dynamic_resolution="Infer the actual rectangular post-merger grid per image; repair placeholders, mask, text/visual M-RoPE and decode cache for its compacted sequence")
        atomic_write_json(root / "protocol.json", protocol)
    return protocol


def validate(root):
    protocol = json.loads((root / "protocol.json").read_text())
    if protocol["implementation_files"] != code_files():
        raise ValueError("Frozen platform evaluation code changed")
    identity = json.loads(Path(protocol["identity_path"]).read_text())
    validate_identity(identity)
    if file_sha(TRAINED / "projector.safetensors") != protocol["adapter_sha256"]:
        raise ValueError("Trained MLP changed")
    for path, sha in protocol["configs"].items():
        if file_sha(path) != sha:
            raise ValueError("Frozen platform configuration changed")
    return protocol


def count_and_check(run, method, stage, target_count=None):
    expected = {task: min(count, 8) if stage == "smoke" else count for task, count in TASKS.items()}
    result_paths = list(run.glob("**/*_results.json"))
    if len(result_paths) != 1:
        raise ValueError(f"Expected one native result file in {run}")
    native = json.loads(result_paths[0].read_text())
    if set(native["results"]) != set(TASKS):
        raise ValueError("Missing native task metrics")
    counts = {}
    for task, n in expected.items():
        paths = list(run.glob(f"**/*_samples_{task}.jsonl"))
        if len(paths) != 1:
            raise ValueError(f"Missing native sample log: {task}")
        with paths[0].open() as stream:
            counts[task] = sum(1 for _ in stream)
        if counts[task] != n:
            raise ValueError(f"{task}: expected {n}, found {counts[task]}")
    ratios, source_counts, result_counts = [], [], []
    if method != "dense":
        traces = list(run.glob("compression_runtime*.requests.jsonl"))
        if len(traces) != 1:
            raise ValueError("Compressed method has no native request trace")
        with traces[0].open() as stream:
            for line in stream:
                row = json.loads(line)
                activations = row["runtime"]["activations"]
                if len(activations) != 1:
                    raise ValueError("Unexpected compression stage count")
                a = activations[0]
                selection = a.get("selection") or a.get("latest_layout_delta")
                if selection is None:
                    raise ValueError("Missing actual per-image selection trace")
                n, k = selection["source_count"], selection["result_count"]
                expected_k = min(n, target_count) if target_count is not None else max(1, min(n, round(n * .25)))
                if k != expected_k:
                    raise ValueError("Actual per-image token count differs from requested budget")
                observed = selection.get("observed_visual_tokens_per_decoder_layer")
                if observed is not None and (len(observed) != 28 or any(v != k for v in observed)):
                    raise ValueError("Observed decoder-layer token counts disagree")
                source_counts.append(n); result_counts.append(k); ratios.append(k / n)
        if len(ratios) != sum(expected.values()):
            raise ValueError("Incomplete request trace count")
    checked = {"status": "complete", "method": method, "stage": stage, "counts": counts,
        "native_result": str(result_paths[0]), "native_metrics": native["results"],
        "trace_requests": len(ratios), "mean_actual_retention_ratio": sum(ratios) / len(ratios) if ratios else 1.0,
        "mean_source_tokens": sum(source_counts) / len(source_counts) if source_counts else None,
        "mean_retained_tokens": sum(result_counts) / len(result_counts) if result_counts else None}
    atomic_write_json(run / "protocol_checks.json", checked)
    return checked


def run(root, target_count=None):
    root.mkdir(parents=True, exist_ok=True)
    with (root / "runner.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        protocol = prepare(root, target_count)
        methods = protocol["methods"]
        logs = root / "logs"; logs.mkdir(exist_ok=True)
        state = {"status": "running", "pid": os.getpid(), "started_utc": now(), "jobs": {}}
        state_path = root / "run_manifest.json"
        atomic_write_json(state_path, state)
        stop, mutex, children = threading.Event(), threading.Lock(), []

        def execute(stage, method, gpu):
            validate(root)
            name = run_name(method, target_count)
            job = f"{stage}_{name}"
            while not stop.is_set():
                free = int(subprocess.check_output(["nvidia-smi", f"--id={gpu}", "--query-gpu=memory.free",
                    "--format=csv,noheader,nounits"], text=True).strip())
                if free >= 40000:
                    break
                stop.wait(15)
            if stop.is_set():
                raise RuntimeError("Sibling job failed")
            path = root / "configs" / f"{stage}_{name}.yaml"
            command = [sys.executable, "-m", "mllm_comp.cli.main", "eval", "--config", str(path)]
            log = logs / f"{job}.log"
            with log.open("w") as stream:
                child = subprocess.Popen(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT,
                    start_new_session=True, env={**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu),
                        "PYTHONUNBUFFERED": "1", "OMP_NUM_THREADS": "4", "TOKENIZERS_PARALLELISM": "false"})
                with mutex:
                    children.append(child)
                    state["jobs"][job] = {"status": "running", "pid": child.pid, "gpu": gpu,
                        "config": str(path), "command": command, "log": str(log)}
                    atomic_write_json(state_path, state)
                while child.poll() is None:
                    if stop.wait(2):
                        os.killpg(child.pid, signal.SIGTERM)
                        child.wait()
                        raise RuntimeError("Stopped owned platform job after sibling failure")
                if child.returncode:
                    stop.set()
                    raise RuntimeError(f"{job} failed; see {log}")
            checked = count_and_check(root / stage / name, method, stage, target_count)
            with mutex:
                state["jobs"][job].update(status="complete", checks=checked, completed_utc=now())
                atomic_write_json(state_path, state)

        try:
            for stage in ("smoke", "full"):
                state["active_stage"] = stage
                atomic_write_json(state_path, state)
                # Fixed waves keep one owned model process per physical GPU.
                for start in range(0, len(methods), 2):
                    with ThreadPoolExecutor(max_workers=2) as pool:
                        futures = [pool.submit(execute, stage, method, offset)
                                   for offset, method in enumerate(methods[start:start + 2])]
                        for future in as_completed(futures):
                            future.result()
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
            atomic_write_json(state_path, state)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--target-count", type=int, help="Fixed output budget; independent of native dynamic resolution")
    args = parser.parse_args()
    if args.prepare_only:
        prepare(args.root.resolve(), args.target_count)
    else:
        run(args.root.resolve(), args.target_count)
