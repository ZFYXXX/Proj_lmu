"""Two budget-neutral recovery arms; no retraining, no historical full reruns."""
import argparse
from dataclasses import asdict
import fcntl
from functools import lru_cache
import json
import math
from pathlib import Path
import sys
import time
import traceback

import yaml

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from mllm_comp.analysis.common.anchor7b_runtime import file_sha
from mllm_comp.analysis.common.learned_topk_experiment import (
    available_gpus, dependency_identity, owned_output, shared_policy, source_identity,
)
from mllm_comp.compressors.image.visual_token_recovery import RecoverySpec
from mllm_comp.core.serialization import atomic_write_json as write_json, atomic_write_text as write_text
from scripts.train.qwen25_vl import run_protected_visual_topk as protected

word = protected.word
TASKS = protected.TASKS
ARMS = ("neighbor_mean", "query_gated_residual")
SMOKE_ARMS = ("protected_parity", "zero_recovery_parity", *ARMS)


def read(path):
    return json.loads(Path(path).read_text())


@lru_cache(maxsize=12)
def historical_traces(reference, task):
    marker = read(Path(reference)/f"eval/full/learned_word/{task}/complete.json")
    return protected.trace_rows(Path(marker["run"]), task)


def prepare(root, reference, noise_reference):
    if root.exists():
        raise ValueError("Choose a fresh recovery directory; preserve old artifacts")
    prior = read(reference/"protocol.json")
    if (read(reference/"manifest.json")["status"] != "complete" or
            read(reference/"full_complete.json")["requests"] != 13176):
        raise ValueError("Protected reference must be fully complete")
    if prior["dependencies"] != dependency_identity():
        raise ValueError("Historical evaluator dependencies changed")
    for filename, digest in prior["input_hashes"].items():
        if file_sha(filename) != digest:
            raise ValueError(f"Historical input changed: {filename}")
    source = Path(prior["source"]["source_root"])
    for name in ("protected_visual_topk.py", "word_visual_topk.py", "learned_visual_topk.py"):
        relative = Path("mllm_comp/compressors/image")/name
        if file_sha(ROOT/relative) != file_sha(source/relative):
            raise ValueError("Old selection/checkpoint implementation changed")
    pins = dict(prior["input_hashes"])
    controls = {task: {} for task in TASKS}
    def pin(filename):
        pins[str(filename)] = file_sha(filename)
    for filename in (reference/"protocol.json", reference/"manifest.json", reference/"full_complete.json"):
        pin(filename)
    old_word = read(Path(prior["reference"])/"protocol.json")
    if read(noise_reference/"manifest.json")["status"] != "complete":
        raise ValueError("Pinned noise control incomplete")
    pin(noise_reference/"protocol.json")
    pin(noise_reference/"manifest.json")
    for task in TASKS:
        for name, parent, arm in (("protected_no_recovery", reference, "learned_word"),
                                  ("word_unified", Path(prior["reference"]), "trained"),
                                  ("vp_noise", noise_reference, "vp_noise")):
            marker = parent/f"eval/full/{arm}/{task}/complete.json"
            saved = read(marker)
            if saved["requests"] != word.context.COUNTS[task] or file_sha(saved["config"]) != saved["config_sha256"]:
                raise ValueError("Incomplete or edited historical evaluation")
            run = Path(saved["run"])
            samples, _ = word.sample_rows(run, f"context_topk_{task}", word.context.COUNTS[task], word.context.METRICS[task])
            for filename in (marker, Path(saved["config"]), samples, run/"compression_runtime.requests.jsonl"):
                pin(filename)
            controls[task][name] = {"run":str(run), "task_name":f"context_topk_{task}"}
        controls[task].update(old_word["controls"][task])
        cfg = protected.config(reference, prior, "learned_word", "smoke", task, root/"unused")
        relative = f"configs/eval/lmms_eval/tasks/context_visual_topk/{task}.yaml"
        if word.task_contract(source/relative, cfg) != word.task_contract(ROOT/relative, cfg):
            raise ValueError("Native task contract changed")
    limits = [
        "No retraining; frozen word step320 and visual prior. No test-based tuning of the .25 residual cap.",
        "Both new arms have EXACTLY the historical learned-protected indices and min(128,N) budget.",
        "Protected representations remain bitwise unchanged; only unprotected context features absorb discarded sources.",
        "Mean arm routes all discarded sources to nearest context and averages including the representative.",
        "Residual arm weights by positive squared cosine and sigmoid(query residual / visual-score std), blend capped at .25.",
        "Query weights are uncalibrated proxies, NOT causal information values; no lossless or first-moment conservation claim.",
        "Original representative M-RoPE is kept; displaced source positions are not independently retained.",
        "Inspired by AdaptMerge nearest-neighbor merging; NOT official AdaptMerge/TRIM/SpecFlow reproduction or novelty proof.",
        "All native prompts/generation/scoring unchanged. AI2D strict primary plus existing leading-option auxiliary.",
        "Historical full quality controls reused, including best OCR noise arm; old timings not matched speed measurements.",
        "Single-seed exploratory runs. Image-cluster paired bootstrap10000; no500-image dev pass.",
    ]
    protocol = {"source":source_identity(ROOT), "dependencies":prior["dependencies"],
        "identity_path":prior["identity_path"], "resource_policy":shared_policy(), "input_hashes":pins,
        "reference":str(reference), "noise_reference":str(noise_reference), "controls":controls,
        "arms":{a:asdict(RecoverySpec(mode=a)) for a in ARMS}, "training":False,
        "full_requests":13176, "smoke_requests":96, "counts":word.context.COUNTS,
        "limits":limits, "created_utc":word.context.native.now()}
    root.mkdir(parents=True)
    write_json(root/"protocol.json", protocol)
    write_json(root/"manifest.json", {"status":"prepared", "stage":"prepared", "source":protocol["source"],
        "completed_cells":[], "protocol_sha256":file_sha(root/"protocol.json"),
        "created_utc":word.context.native.now(), "training":False})
    print(json.dumps({"status":"prepared", "full_requests":13176, "smoke_requests":96}), flush=True)


def validate(root):
    protocol = word.context.native.validate(root)
    if read(root/"manifest.json")["protocol_sha256"] != file_sha(root/"protocol.json"):
        raise ValueError("Recovery protocol changed")
    return protocol


def config(root, protocol, arm, phase, task, run):
    reference = Path(protocol["reference"])
    cfg = protected.config(reference, read(reference/"protocol.json"), "learned_word", phase, task, run)
    if arm != "protected_parity":
        stage, = cfg["pipeline"]["stages"]
        algorithm = "recovered_visual_topk"
        stage.update(name=f"recovery_{arm}", algorithm=algorithm, plugin=algorithm,
            variant=f"{algorithm}.image", binding=f"{algorithm}.image.qwen25_vl")
        stage["config"]["recovery"] = ({"mode":"query_gated_residual", "maximum_blend":0.}
            if arm == "zero_recovery_parity" else protocol["arms"][arm])
    cfg["experiment"]["name"] = f"recovery_{phase}_{arm}_{task}"
    return cfg


def check(root, protocol, run, arm, phase, task):
    reference = Path(protocol["reference"])
    checks = protected.check(reference, read(reference/"protocol.json"), run, "learned_word", phase, task)
    traces = protected.trace_rows(run, task)
    old = historical_traces(str(reference), task)
    for i, trace in traces.items():
        if any(trace[key] != old[i][key] for key in ("selected_indices", "protected_indices", "source_count")):
            raise ValueError("Recovery changed historical source tokens, selection, or protection")
        if arm == "protected_parity":
            continue
        n, k = trace["source_count"], trace["result_count"]
        dropped = trace["recovery_dropped_indices"]
        if sorted(dropped + trace["selected_indices"]) != list(range(n)):
            raise ValueError("Recovery source population is duplicated or missing")
        if (not trace["recovery_protected_unchanged"] or not trace["recovered_features_injected"]
                or trace["recovery_extra_tokens"] != 0 or trace["recovery_selection_changed"]):
            raise ValueError("Invalid recovery injection/protection/budget trace")
        values = trace["recovery_source_weights"] + trace["recovery_blend"] + [trace["recovery_mean_relative_change"]]
        if any(not math.isfinite(v) for v in values):
            raise ValueError("Nonfinite recovery statistics")
        targets = trace["recovery_target_indices"]
        context = set(trace["selected_indices"]) - set(trace["protected_indices"])
        if trace["recovery_applied"]:
            if len(targets) != n-k or len(trace["recovery_source_weights"]) != n-k or not set(targets) <= context:
                raise ValueError("Discarded source was assigned to a protected/invalid target")
            if sum(trace["recovery_assignment_counts"]) != n-k:
                raise ValueError("Recovery population accounting failed")
        elif n != k and context:
            raise ValueError("Applicable recovery silently skipped")
        if trace["recovery_mode"] == "query_gated_residual" and any(
                b > trace["recovery_maximum_blend"]+1e-6 or b < 0 for b in trace["recovery_blend"]):
            raise ValueError("Residual blend outside frozen bound")
        if arm == "zero_recovery_parity" and trace["recovery_changed_tokens"] != 0:
            raise ValueError("Zero-blend recovery changed token contents")
    if arm in ("protected_parity", "zero_recovery_parity"):
        old_run = Path(read(reference/f"eval/full/learned_word/{task}/complete.json")["run"])
        _, previous = word.sample_rows(old_run, f"context_topk_{task}", word.context.COUNTS[task], word.context.METRICS[task])
        _, current = word.sample_rows(run, f"context_topk_{task}", 8, word.context.METRICS[task])
        if any(a["filtered_resps"] != b["filtered_resps"] for a,b in zip(current, previous)):
            raise ValueError("Historical protected generation parity failed")
    return checks


def report(root, protocol):
    protected_protocol = read(Path(protocol["reference"])/"protocol.json")
    word_protocol = read(Path(protected_protocol["reference"])/"protocol.json")
    cohort = read(Path(word_protocol["reference"])/"cohort.json")
    records = []
    for arm in ARMS:
        for task in TASKS:
            marker = root/f"eval/full/{arm}/{task}/complete.json"
            if not marker.exists():
                continue
            run = Path(read(marker)["run"])
            _, rows = word.sample_rows(run, f"context_topk_{task}", word.context.COUNTS[task], word.context.METRICS[task])
            scores = [float(r[word.context.METRICS[task]]) for r in rows]
            images = [r["image_fingerprint"] for r in cohort[task]["rows"]]
            controls = {}
            for name, ref in protocol["controls"][task].items():
                _, previous = word.sample_rows(Path(ref["run"]), ref["task_name"], len(rows), word.context.METRICS[task])
                if any((a["input"],a["target"]) != (b["input"],b["target"]) for a,b in zip(rows,previous)):
                    raise ValueError("Report prompt/answer pairing changed")
                values = [float(r[word.context.METRICS[task]]) for r in previous]
                controls[name] = {"score":sum(values)/len(values),
                    "paired_difference":word.context.clustered_difference(scores, values, images)}
            native = read(next(run.glob("**/*_results.json")))["results"][f"context_topk_{task}"]
            traces = list(protected.trace_rows(run, task).values())
            performance = [json.loads(s) for s in (run/"request_performance.jsonl").read_text().splitlines()]
            records.append({"arm":arm, "task":task, "questions":len(rows), "independent_images":len(set(images)),
                "score":sum(scores)/len(scores), "native_metrics":native, "controls":controls,
                "mean_changed_context_tokens":sum(t["recovery_changed_tokens"] for t in traces)/len(traces),
                "mean_relative_feature_change":sum(t["recovery_mean_relative_change"] for t in traces)/len(traces),
                "mean_retained_tokens":sum(t["result_count"] for t in traces)/len(traces),
                "total_generation_seconds":sum(p["generation_s"] for p in performance),
                "run":str(run)})
    write_json(root/"summary.json", {"completed_cells":len(records), "total_cells":6,
        "results":records, "limits":protocol["limits"]})
    lines = ["# Fixed-selection K128 context recovery", "", "| Arm | Task | Native score | No recovery | Old MLP | Dense | Noise |",
        "|---|---|---:|---:|---:|---:|---:|"]
    for row in records:
        c = row["controls"]
        values = [row["score"], *[c[k]["score"] for k in ("protected_no_recovery", "old_mlp", "dense", "vp_noise")]]
        display = [f"{1000*v:.0f}" if row["task"] == "ocrbench" else f"{100*v:.2f}%" for v in values]
        lines.append("| "+" | ".join([row["arm"], row["task"], *display])+" |")
    write_text(root/"report.md", "\n".join(lines+["", "OCRBench uses native points /1000; AI2D strict is primary, leading-option is auxiliary in summary.json.", "", *protocol["limits"]])+"\n")


def pipeline(root, gpu):
    protocol = validate(root)
    manifest = read(root/"manifest.json")
    def stage(name, status="running"):
        manifest.update(stage=name, status=status, gpu=gpu, updated_utc=word.context.native.now())
        write_json(root/"manifest.json", manifest)
    def evaluate(arm, phase, task):
        marker = root/f"eval/{phase}/{arm}/{task}/complete.json"
        if marker.exists():
            saved = read(marker)
            if saved["source"] != protocol["source"] or file_sha(saved["config"]) != saved["config_sha256"]:
                raise ValueError("Completed recovery cell provenance changed")
            check(root, protocol, Path(saved["run"]), arm, phase, task)
            return
        while available_gpus([gpu]) != [gpu]:
            stage("waiting_for_gpu", "queued")
            time.sleep(20)
        validate(root)
        stage(f"{phase}:{arm}:{task}")
        folder = marker.parent
        run = folder/f"attempt_{time.time_ns()}"
        filename = root/"configs"/f"{phase}_{arm}_{task}_{time.time_ns()}.yaml"
        write_text(filename, yaml.safe_dump(config(root, protocol, arm, phase, task, run), sort_keys=False))
        word.context.native.execute([sys.executable, "-m", "mllm_comp.cli.main", "eval", "--config", str(filename)],
            folder/f"launch_{time.time_ns()}.log", [gpu], {"MLLM_TOPK_PERFORMANCE":str(run/"request_performance.jsonl"),
                "HF_HOME":str(root/"cache/huggingface"), "HF_DATASETS_CACHE":str(root/"cache/datasets"),
                "XDG_CACHE_HOME":str(root/"cache/xdg")})
        checks = check(root, protocol, run, arm, phase, task)
        write_json(marker, {"source":protocol["source"], "run":str(run), "config":str(filename),
            "config_sha256":file_sha(filename), "requests":checks["requests"], "completed_utc":word.context.native.now()})
        if phase == "full":
            manifest["completed_cells"] = sorted(set(manifest["completed_cells"]) | {f"{arm}/{task}"})
            write_json(root/"manifest.json", manifest)
            report(root, protocol)
    try:
        for arm in SMOKE_ARMS:
            for task in TASKS:
                evaluate(arm, "smoke", task)
        write_json(root/"smoke_complete.json", {"status":"complete", "requests":96, "time":word.context.native.now()})
        for arm in ARMS:
            for task in TASKS:
                evaluate(arm, "full", task)
        stage("complete", "complete")
        manifest["finished_utc"] = word.context.native.now()
        write_json(root/"manifest.json", manifest)
        write_json(root/"full_complete.json", {"status":"complete", "requests":13176, "time":word.context.native.now()})
    except BaseException as error:
        stage("failed", "failed")
        write_json(root/f"failure_{time.time_ns()}.json", {"error":str(error), "traceback":traceback.format_exc()})
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("prepare", "pipeline", "report"))
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--noise-reference", type=Path)
    parser.add_argument("--gpu", type=int, default=1)
    args = parser.parse_args()
    root = owned_output(args.root)
    if args.action == "prepare":
        if args.reference is None or args.noise_reference is None:
            parser.error("prepare requires protected and noise reference roots")
        prepare(root, owned_output(args.reference), owned_output(args.noise_reference))
    else:
        with (root/"runner.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            if args.action == "report":
                report(root, validate(root))
            else:
                pipeline(root, args.gpu)
