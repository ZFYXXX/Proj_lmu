"""Fixed K128 diagnostic: protected query evidence plus complementary context.

No new training, no official AnchorPrune reproduction, no old full reruns.
All native smokes must pass before either arm enters full evaluation.
"""
import argparse
from dataclasses import asdict
import fcntl
import json
from pathlib import Path
import sys
import time
import traceback

import yaml

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from mllm_comp.analysis.common.anchor7b_runtime import file_sha, validate_identity
from mllm_comp.analysis.common.learned_topk_experiment import (
    available_gpus, dependency_identity, owned_output, shared_policy, source_identity,
)
from mllm_comp.compressors.image.protected_visual_topk import ProtectedTopKSpec
from mllm_comp.core.serialization import atomic_write_json as write_json, atomic_write_text as write_text
from scripts.train.qwen25_vl import run_word_visual_topk as word
from scripts.diagnostics.run_selector_condition_audit import trace_rows

TASKS = word.TASKS
ARMS = ("learned_word", "raw_cosine")


def read(path):
    return json.loads(Path(path).read_text())


def prepare(root, reference):
    if root.exists():
        raise ValueError("Preserve old results; choose a fresh run root")
    prior = read(reference/"protocol.json")
    if read(reference/"full_complete.json")["requests"] != 6588 or read(reference/"manifest.json")["status"] != "complete":
        raise ValueError("Word-level reference must be fully complete")
    if prior["dependencies"] != dependency_identity():
        raise ValueError("Evaluator dependencies changed")
    for filename, digest in prior["input_hashes"].items():
        if file_sha(filename) != digest:
            raise ValueError(f"Reference input changed: {filename}")
    identity = read(prior["identity_path"])
    validate_identity(identity)
    old_source = Path(prior["source"]["source_root"])
    for name in ("word_visual_topk.py", "learned_visual_topk.py"):
        relative = Path("mllm_comp/compressors/image")/name
        if file_sha(ROOT/relative) != file_sha(old_source/relative):
            raise ValueError("Existing checkpoint implementation changed")
    checkpoint = word.context.native.latest(reference/"training")
    pins = dict(prior["input_hashes"])
    for filename in (reference/"protocol.json", reference/"manifest.json", reference/"full_complete.json",
                     reference/"summary.json", checkpoint/"selector.safetensors", checkpoint/"selector_config.json",
                     Path(prior["identity_path"])):
        pins[str(filename)] = file_sha(filename)
    for task in TASKS:
        marker = reference/f"eval/full/trained/{task}/complete.json"
        completed = read(marker)
        if completed["requests"] != word.context.COUNTS[task]:
            raise ValueError("Incomplete historical cell")
        samples, _ = word.sample_rows(Path(completed["run"]), f"context_topk_{task}",
                                      word.context.COUNTS[task], word.context.METRICS[task])
        for filename in (marker, samples, Path(completed["run"])/"compression_runtime.requests.jsonl"):
            pins[str(filename)] = file_sha(filename)
        cfg = word.config(reference, prior, "trained", "smoke", task, root/"unused")
        for candidate in (old_source, ROOT):
            name = candidate/f"configs/eval/lmms_eval/tasks/context_visual_topk/{task}.yaml"
            contract = word.task_contract(name, cfg)
            if candidate == old_source:
                old_contract = contract
            elif contract != old_contract:
                raise ValueError("Native task or generation contract changed")
    limits = [
        "Fixed min(128,N); original visual content/order/M-RoPE and original LLM question.",
        "No new training, mediator, recovery, dynamic K, noise or OT.",
        "Both arms reuse the exact completed word head and frozen learned visual prior.",
        "AnchorPrune-inspired selection policy, NOT official AnchorPrune: context weights are our learned visual scores.",
        "Two exploratory policy arms, not a clean attribution of protection versus novelty individually.",
        "Wrong/generic questions alter only scorer audits; not the LLM generation question.",
        "No verified region annotations; selection change does not prove semantic evidence coverage.",
        "Historical quality controls reused; old timings are not matched speed measurements.",
    ]
    protocol = {"source":source_identity(ROOT), "dependencies":dependency_identity(),
        "identity_path":prior["identity_path"], "resource_policy":shared_policy(),
        "input_hashes":pins, "reference":str(reference), "checkpoint":str(checkpoint),
        "arms":{arm:asdict(ProtectedTopKSpec(relevance=arm)) for arm in ARMS},
        "training":False, "full_requests":13176, "smoke_requests":72,
        "counts":word.context.COUNTS, "limits":limits, "created_utc":word.context.native.now()}
    root.mkdir(parents=True)
    write_json(root/"protocol.json", protocol)
    write_json(root/"manifest.json", {"status":"prepared", "stage":"prepared", "source":protocol["source"],
        "completed_cells":[], "protocol_sha256":file_sha(root/"protocol.json"),
        "created_utc":word.context.native.now(), "training":False})
    print(json.dumps({"status":"prepared", "arms":list(ARMS), "full_requests":13176, "new_training":False}), flush=True)


def validate(root):
    protocol = word.context.native.validate(root)
    if read(root/"manifest.json")["protocol_sha256"] != file_sha(root/"protocol.json"):
        raise ValueError("Protected diagnostic protocol changed")
    return protocol


def config(root, protocol, arm, phase, task, run):
    reference = Path(protocol["reference"])
    prior = read(reference/"protocol.json")
    cfg = word.config(reference, prior, "trained", phase, task, run)
    if arm != "unified_parity":
        stage, = cfg["pipeline"]["stages"]
        stage.update(name=f"protected_{arm}", algorithm="protected_visual_topk", plugin="protected_visual_topk",
                     variant="protected_visual_topk.image", binding="protected_visual_topk.image.qwen25_vl")
        stage["config"]["protection"] = protocol["arms"][arm]
    cfg["experiment"]["name"] = f"protected_{phase}_{arm}_{task}"
    return cfg


def check(root, protocol, run, arm, phase, task):
    prior = read(Path(protocol["reference"])/"protocol.json")
    checks = word.check(Path(protocol["reference"]), prior, run, "trained", phase, task)
    traces = trace_rows(run, task)
    if arm == "unified_parity":
        marker = read(Path(protocol["reference"])/f"eval/full/trained/{task}/complete.json")
        old_run = Path(marker["run"])
        old_traces = trace_rows(old_run, task)
        _, old = word.sample_rows(old_run, f"context_topk_{task}", word.context.COUNTS[task], word.context.METRICS[task])
        _, new = word.sample_rows(run, f"context_topk_{task}", 8, word.context.METRICS[task])
        for i, trace in traces.items():
            if trace["selected_indices"] != old_traces[i]["selected_indices"] or new[i]["filtered_resps"] != old[i]["filtered_resps"]:
                raise ValueError("Historical word selection/prediction parity failed")
    else:
        for trace in traces.values():
            if (trace.get("selection_policy") != "protected_context" or trace.get("policy_relevance") != arm or
                    trace.get("official_anchorprune_reproduction") is not False or trace.get("new_training") is not False):
                raise ValueError("Wrong protected arm or misleading provenance")
            anchor, selected = trace["protected_indices"], trace["selected_indices"]
            if (not set(anchor) <= set(selected) or len(anchor) != len(set(anchor)) or
                    trace["protected_count"] != len(anchor) or trace["context_count"] + len(anchor) != len(selected)):
                raise ValueError("Protected/context accounting failed")
            if trace["source_count"] > 128 and not 20 <= len(anchor) <= 64:
                raise ValueError("Adaptive anchor outside frozen bounds")
    return checks


def report(root, protocol):
    prior = read(Path(protocol["reference"])/"protocol.json")
    cohort = read(Path(prior["reference"])/"cohort.json")
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
            comparisons = {}
            refs = dict(prior["controls"][task])
            word_run = read(Path(protocol["reference"])/f"eval/full/trained/{task}/complete.json")["run"]
            refs["word_unified"] = {"run":word_run, "task_name":f"context_topk_{task}"}
            for name, ref in refs.items():
                _, previous = word.sample_rows(Path(ref["run"]), ref["task_name"], len(rows), word.context.METRICS[task])
                if any((a["input"],a["target"]) != (b["input"],b["target"]) for a,b in zip(rows, previous)):
                    raise ValueError("Report prompt/target pairing changed")
                values = [float(r[word.context.METRICS[task]]) for r in previous]
                comparisons[name] = {"score":sum(values)/len(values),
                    "paired_difference":word.context.clustered_difference(scores, values, images)}
            traces = list(trace_rows(run, task).values())
            records.append({"arm":arm, "task":task, "questions":len(rows), "independent_images":len(set(images)),
                "score":sum(scores)/len(scores), "controls":comparisons,
                "mean_anchor_tokens":sum(t["protected_count"] for t in traces)/len(traces),
                "mean_source_tokens":sum(t["source_count"] for t in traces)/len(traces),
                "condition_audits":{name:{"informative_questions":sum(t["question_selection_audit"][name]["question_changed_after_cleaning"] for t in traces),
                    "mean_exchanged_tokens_all":sum(t["question_selection_audit"][name]["exchanged_tokens"] for t in traces)/len(traces)}
                    for name in ("shuffled","generic")}})
    write_json(root/"summary.json", {"completed_cells":len(records), "total_cells":6, "results":records, "limits":protocol["limits"]})
    lines = ["# Fixed-K128 protected evidence diagnostic", "", "| Arm | Dataset | Score | Word unified | Difference | Mean anchor |",
             "|---|---|---:|---:|---:|---:|"]
    for row in records:
        previous = row["controls"]["word_unified"]["score"]
        lines.append(f"| {row['arm']} | {row['task']} | {100*row['score']:.2f}% | {100*previous:.2f}% | {100*(row['score']-previous):+.2f} pp | {row['mean_anchor_tokens']:.1f} |")
    write_text(root/"report.md", "\n".join(lines+["", "OCRBench percentage = native points / 1000. AI2D uses strict matching.", "", *protocol["limits"]])+"\n")


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
                raise ValueError("Completed result provenance changed")
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
            report(root, protocol)
            write_json(root/"manifest.json", manifest)
    try:
        for arm in ("unified_parity", *ARMS):
            for task in TASKS:
                evaluate(arm, "smoke", task)
        write_json(root/"smoke_complete.json", {"status":"complete", "requests":72, "time":word.context.native.now()})
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
    parser.add_argument("--gpu", type=int, default=1)
    args = parser.parse_args()
    root = owned_output(args.root)
    if args.action == "prepare":
        if args.reference is None:
            parser.error("prepare requires --reference")
        prepare(root, owned_output(args.reference))
    else:
        with (root/"runner.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            if args.action == "report":
                report(root, validate(root))
            else:
                pipeline(root, args.gpu)
