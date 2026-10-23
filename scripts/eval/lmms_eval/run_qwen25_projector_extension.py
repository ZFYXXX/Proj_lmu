"""Sequential native-platform evaluations, gated by dense identity and smoke."""
import argparse
from datetime import datetime, timezone
import fcntl
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from scripts.eval.lmms_eval import run_projector_ratio_platform as base
from mllm_comp.analysis.common.anchor7b_runtime import file_sha, validate_identity
from mllm_comp.core.serialization import atomic_write_json, atomic_write_text

PRIOR = ROOT / "outputs/eval/qwen25_vl/projector_dynamic_k128_platform_20261021_v1"
METHODS = {
    "divprune_k128": ("divprune", {"target_count": 128}, "Max-min diversity selection; Qwen post-merger adaptation"),
    "visionzip_k128": ("visionzip", {"budget": 128, "attention_mode": "mean_received", "dominant_weight": 27, "contextual_weight": 5}, "108 dominant + 20 contextual tokens; Qwen post-merger adaptation, not original pre-projector placement"),
    "aim_input_merge_k128": ("aim", {"target_count": 128}, "Only AIM's input bipartite merging; no decoder-stage AIM pruning"),
    "gap_attention_k128": ("gap", {"target_count": 128}, "Framework received-vision-attention Top-K; not an independently certified full paper reproduction"),
    "hiprune_qwen_l16_k128": ("hiprune", {"tokens_per_crop": 128, "object_layer": 16, "vision_layer": -1, "alpha": .1}, "Official Qwen object-layer setting, adapted to the fixed K128/native-resolution protocol"),
    "eadp_qwen_beta2_k128": ("eadp", {"tokens_per_crop": 128, "alpha": .5, "beta": 2.}, "Official Qwen script beta=2; native rather than official fixed-square resolution"),
}
IDENTITY_METHODS = tuple(name for name, (algorithm, _, _) in METHODS.items() if algorithm in {"divprune", "visionzip", "aim", "gap"})
SCORES = {"qwen25_7b_ratio_chartqa": "relaxed_overall", "qwen25_7b_ratio_ai2d": "exact_match",
          "qwen25_7b_ratio_ocrbench": "ocrbench_accuracy", "qwen25_7b_ratio_docvqa_val": "anls"}


def config(root, name, phase):
    algorithm, params, _ = METHODS[name]
    cfg = base.config(root, algorithm, "full" if phase == "full" else "smoke", target_count=128)
    cfg["experiment"]["name"] = f"{phase}_{name}"
    stage = cfg["pipeline"]["stages"][0]
    stage.update(name=name, config=dict(params))
    if phase == "identity":
        if algorithm == "visionzip":
            stage["config"]["budget"] = 2048  # Native processor upper bound; no compression.
        else:
            stage["config"] = {"retain_ratio": 1.}
    path = root / phase / name
    cfg["output"] = {"run_dir": str(path), "manifest_path": str(path / "manifest.json")}
    return cfg


def implementation_files():
    paths = {Path(__file__), *[ROOT / p for p in base.code_files()],
        ROOT / "scripts/eval/lmms_eval/run_projector_ratio_platform.py",
        ROOT / "mllm_comp/backbones/qwen25_vl/provider_bridge.py",
        ROOT / "mllm_comp/backbones/qwen25_vl/image_compression.py",
        ROOT / "mllm_comp/compressors/multimodal_registry.py"}
    for filename in ("divprune", "visionzip", "aim", "gap"):
        paths.add(ROOT / f"mllm_comp/compressors/image/{filename}.py")
    for filename in ("operations", "cdpruner", "hiprune", "eadp"):
        paths.add(ROOT / f"mllm_comp/compressors/comparison/{filename}.py")
    return {str(p.relative_to(ROOT)): file_sha(p) for p in sorted(paths)}


def prepare(root):
    import yaml
    root.mkdir(parents=True, exist_ok=True)
    if (root / "protocol.json").exists():
        return validate(root)
    old = json.loads((PRIOR / "run_manifest.json").read_text())
    if old["status"] != "complete":
        raise ValueError("Prior matrix must be complete; do not overlap its GPU queue")
    identity = json.loads((base.SOURCE / "model_identity.json").read_text())
    validate_identity(identity)
    (root / "configs").mkdir()
    hashes = {}
    for phase in ("identity", "smoke", "full"):
        for name in (IDENTITY_METHODS if phase == "identity" else METHODS):
            path = root / "configs" / f"{phase}_{name}.yaml"
            atomic_write_text(path, yaml.safe_dump(config(root, name, phase), allow_unicode=True, sort_keys=False))
            hashes[str(path)] = file_sha(path)
    protocol = {"created_utc": base.now(), "allowed_gpus": [0], "reserved_gpu": 1,
        "entry": "python -m mllm_comp.cli.main eval --config ...", "target_count": 128,
        "dynamic_resolution": "Native per-image grid, unchanged processor; K_i=min(128,N_i), no proportional or layer-average target",
        "methods": {n: {"algorithm": a, "config": c, "scope": s} for n, (a, c, s) in METHODS.items()},
        "tasks": base.TASKS, "identity_methods": IDENTITY_METHODS,
        "gate": "All four real-7B identity runs must match prior dense logged responses, and all six 32-request smokes must pass before full splits",
        "source_references": {"divprune": "https://github.com/vbdi/divprune",
            "visionzip": "https://github.com/JIA-Lab-research/VisionZip",
            "hiprune": "https://github.com/Danielement321/HiPrune/tree/82781005a7e72a6be9ede58fd77473efa72b5e4f",
            "eadp": "https://github.com/SJTU-DeepVisionLab/EADP/tree/e1a08801461c181871b9ff1cb0803b8e9966f083"},
        "paper_equivalence": "Not claimed. Identity/cache checks validate execution, not official selection equivalence or published quality.",
        "prior_run": str(PRIOR), "prior_protocol_sha256": file_sha(PRIOR / "protocol.json"),
        "prior_manifest_sha256": file_sha(PRIOR / "run_manifest.json"),
        "identity_path": str(base.SOURCE / "model_identity.json"), "base_identity_sha256": identity["identity_sha256"],
        "implementation_files": implementation_files(), "configs": hashes,
        "training": "None; existing trained MLP is an explicitly trained reference, not an equal-training-budget control",
        "ai2d": "Keep native primary exact_match and separately labeled leading-option sensitivity score",
        "timing": "Shared GPU and differing execution periods; elapsed times are not matched speed claims"}
    atomic_write_json(root / "protocol.json", protocol)
    return protocol


def validate(root):
    p = json.loads((root / "protocol.json").read_text())
    if p["implementation_files"] != implementation_files():
        raise ValueError("Frozen extension implementation changed")
    for path, sha in p["configs"].items():
        if file_sha(path) != sha:
            raise ValueError("Frozen configuration changed")
    if file_sha(PRIOR / "protocol.json") != p["prior_protocol_sha256"] or file_sha(PRIOR / "run_manifest.json") != p["prior_manifest_sha256"]:
        raise ValueError("Prior reference provenance changed")
    validate_identity(json.loads(Path(p["identity_path"]).read_text()))
    return p


def sample_rows(run, task):
    paths = list(run.glob(f"**/*_samples_{task}.jsonl"))
    if len(paths) != 1:
        raise ValueError(f"Expected one native sample ledger for {task} in {run}")
    with paths[0].open() as f:
        rows = [json.loads(line) for line in f]
    indexed = {int(x["doc_id"]): x for x in rows}
    if len(indexed) != len(rows):
        raise ValueError("Duplicate native sample IDs")
    return indexed


def request_key(sample_id):
    """Native request ordinal changes with length sorting/limit; doc ID does not."""
    parts = str(sample_id).split(":")
    if len(parts) != 4:
        raise ValueError("Malformed native sample ID")
    task, split, doc_id, ordinal = parts
    if task not in base.TASKS or split != ("validation" if task.endswith("docvqa_val") else "test"):
        raise ValueError("Unknown native task or split")
    if not doc_id.isdigit() or not ordinal.isdigit() or int(doc_id) >= base.TASKS[task]:
        raise ValueError("Invalid native sample index")
    return task, split, int(doc_id)


def source_counts():
    counts = {}
    with (PRIOR / "full/hiprune_k128/compression_runtime.requests.jsonl").open() as f:
        for line in f:
            x = json.loads(line); key = request_key(x["request"]["sample_id"])
            if key in counts:
                raise ValueError("Duplicate source-reference document")
            counts[key] = x["runtime"]["activations"][0]["selection"]["source_count"]
    if len(counts) != sum(base.TASKS.values()):
        raise ValueError("Incomplete prior source-token reference")
    return counts


def check(run, name, phase, sources):
    results = list(run.glob("**/*_results.json"))
    if len(results) != 1:
        raise ValueError("Missing or duplicate native result file")
    native = json.loads(results[0].read_text())["results"]
    if set(native) != set(base.TASKS):
        raise ValueError("Missing dataset scores")
    for task, total in base.TASKS.items():
        rows = sample_rows(run, task)
        if set(rows) != set(range(total if phase == "full" else 8)):
            raise ValueError(f"Incomplete native task: {task}")
        reference = sample_rows(PRIOR / "full/dense", task)
        for i, x in rows.items():
            if any(x[k] != reference[i][k] for k in ("input", "target")):
                raise ValueError("Question, sample order or gold reference changed")
            if not math.isfinite(float(x[SCORES[task]])):
                raise ValueError("Nonfinite native score")
            if phase == "identity" and x["filtered_resps"] != reference[i]["filtered_resps"]:
                raise ValueError(f"Dense identity response mismatch: {name}/{task}/{i}")
    traces = list(run.glob("compression_runtime*.requests.jsonl"))
    if len(traces) != 1:
        raise ValueError("Missing request trace")
    seen, original, retained, verified = set(), [], [], 0
    with traces[0].open() as f:
        for line in f:
            x = json.loads(line); sid = request_key(x["request"]["sample_id"])
            if sid in seen or sid not in sources:
                raise ValueError("Duplicate or unknown request trace")
            seen.add(sid); a, = x["runtime"]["activations"]
            delta = a.get("selection") or a.get("latest_layout_delta")
            n, k = delta["source_count"], delta["result_count"]
            if n != sources[sid] or k != (n if phase == "identity" else min(n, 128)):
                raise ValueError("Source grid or actual retained token count changed")
            original.append(n); retained.append(k)
            if name in IDENTITY_METHODS:
                g = a["projector_generate"]["last"]
                if g["prefill_calls"] != 1 or g["prompt_removed"] != n - k:
                    raise ValueError("Invalid compacted prefill lifecycle")
                if g["decode_steps"]:
                    if g["decode_position_verified"] is not True:
                        raise ValueError("First decode M-RoPE position not verified")
                    verified += 1
            elif name.startswith("hiprune") and a["config"]["object_layer"] != 16:
                raise ValueError("HiPrune layer setting drifted")
            elif name.startswith("eadp") and a["config"]["beta"] != 2.:
                raise ValueError("EADP beta setting drifted")
    if len(seen) != (sum(base.TASKS.values()) if phase == "full" else 32):
        raise ValueError("Incomplete compression traces")
    out = {"status": "complete", "name": name, "phase": phase, "requests": len(seen),
        "mean_source_tokens": sum(original) / len(original), "mean_output_tokens": sum(retained) / len(retained),
        "first_decode_position_verified_requests": verified, "native_metrics": native, "native_result": str(results[0]),
        "scope": METHODS[name][2]}
    atomic_write_json(run / "protocol_checks.json", out)
    return out


def report(root, state):
    old = json.loads((PRIOR / "run_manifest.json").read_text())
    rows = {"baseline": old["jobs"]["full_dense"]["checks"]["native_metrics"],
        "trained_anchor_mlp_k128": old["jobs"]["full_feature_anchor_adapter_k128"]["checks"]["native_metrics"]}
    rows.update({name: state["jobs"][f"full_{name}"]["checks"]["native_metrics"] for name in METHODS})
    atomic_write_json(root / "summary.json", {"status": "complete", "metrics": rows, "scopes": {n: s for n, (_, _, s) in METHODS.items()}})
    lines = ["# Native dynamic-resolution K128 extension", "", "All scores below use the same native platform protocol. These are explicitly scoped Qwen adaptations, not certified paper reproductions.", "", "| Method | ChartQA % | AI2D primary % | OCRBench /1000 | DocVQA ANLS /100 |", "|---|---:|---:|---:|---:|"]
    for name, metrics in rows.items():
        values = [next(float(v) for k, v in metrics[t].items() if k.split(",")[0] == SCORES[t]) for t in SCORES]
        lines.append(f"| {name} | {100*values[0]:.2f} | {100*values[1]:.2f} | {1000*values[2]:.0f} | {100*values[3]:.2f} |")
    lines += ["", "AI2D primary is format-sensitive; see the separately retained leading_option_accuracy metric. The MLP reference is trained; the added methods are training-free. Timing is not a controlled speed benchmark."]
    atomic_write_text(root / "report.md", "\n".join(lines) + "\n")


def run(root):
    prepare(root)
    with (root / "runner.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        source = source_counts()
        path = root / "run_manifest.json"
        state = json.loads(path.read_text()) if path.exists() else {"jobs": {}, "started_utc": base.now()}
        if any(j["status"] == "running" for j in state["jobs"].values()):
            raise ValueError("Do not duplicate a running or interrupted job; inspect and preserve its attempt first")
        state.update(status="running", pid=os.getpid(), allowed_gpus=[0])
        atomic_write_json(path, state)
        (root / "logs").mkdir(exist_ok=True)
        try:
            for phase in ("identity", "smoke", "full"):
                state["active_stage"] = phase; atomic_write_json(path, state)
                for name in (IDENTITY_METHODS if phase == "identity" else METHODS):
                    job = f"{phase}_{name}"
                    if state["jobs"].get(job, {}).get("status") == "complete":
                        check(root / phase / name, name, phase, source)
                        continue
                    validate(root)
                    while int(subprocess.check_output(["nvidia-smi", "--id=0", "--query-gpu=memory.free", "--format=csv,noheader,nounits"], text=True).strip()) < 40000:
                        time.sleep(15)
                    cfg = root / "configs" / f"{job}.yaml"
                    command = [sys.executable, "-m", "mllm_comp.cli.main", "eval", "--config", str(cfg)]
                    log = root / "logs" / f"{job}.log"
                    with log.open("w") as stream:
                        child = subprocess.Popen(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT,
                            start_new_session=True, env={**os.environ, "CUDA_VISIBLE_DEVICES": "0", "PYTHONUNBUFFERED": "1", "OMP_NUM_THREADS": "4", "TOKENIZERS_PARALLELISM": "false"})
                        state["jobs"][job] = {"status": "running", "pid": child.pid, "gpu": 0, "command": command, "log": str(log), "started_utc": base.now()}
                        atomic_write_json(path, state)
                        if child.wait():
                            raise RuntimeError(f"{job} failed; see {log}")
                    checked = check(root / phase / name, name, phase, source)
                    state["jobs"][job].update(status="complete", checks=checked, completed_utc=base.now())
                    atomic_write_json(path, state)
            report(root, state)
            state.update(status="complete", completed_utc=base.now())
        except BaseException as exc:
            if "job" in locals() and state["jobs"].get(job, {}).get("status") == "running":
                state["jobs"][job].update(status="failed", error=str(exc))
            state.update(status="failed", error=str(exc), traceback=traceback.format_exc())
            raise
        finally:
            atomic_write_json(path, state)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    prepare(args.root.resolve()) if args.prepare_only else run(args.root.resolve())
