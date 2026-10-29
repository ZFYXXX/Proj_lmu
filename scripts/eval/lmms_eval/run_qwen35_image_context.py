"""Train the missing image-global context arm and queue three native full evaluations."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
from queue import Empty, Queue
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from mllm_comp.analysis.common.anchor7b_runtime import file_sha
from mllm_comp.analysis.common.learned_topk_experiment import owned_output, source_identity, reject_overlap, validate_training_entries
from mllm_comp.core.serialization import atomic_write_json as write_json, atomic_write_text as write_text
from mllm_comp.compressors.comparison.qwen35_image_context import METHOD
from scripts.eval.lmms_eval import run_qwen35_transfer as shared

PREVIOUS = Path("/home/ZhouFengyu/fastv-repro/MLLM-Comp-Platform/MLLM-Comp-Platform-main/outputs/eval/qwen35_vl/transfer_k128_20261026_v1")


def read(path):
    return json.loads(Path(path).read_text())


def now():
    return datetime.now(timezone.utc).isoformat()


def prepare(root, previous):
    if root.exists():
        raise ValueError("Use a fresh run identity")
    old = read(previous / "protocol.json")
    entries = read(old["training_manifest"])["entries"]
    validate_training_entries(entries)
    cohort = read(old["cohort_path"])
    for task in shared.COUNTS:
        reject_overlap(entries, cohort[task]["rows"])
    for name, checksum in old["input_hashes"].items():
        if file_sha(name) != checksum:
            raise ValueError(f"Changed original training/evaluation input: {name}")
    prior_dir = previous / "training/visual"
    pointer = read(prior_dir / "latest.json")
    if pointer["step"] != 320 or read(prior_dir / "manifest.json")["status"] != "complete":
        raise ValueError("The same-base visual prior is incomplete")
    prior = prior_dir / pointer["checkpoint"]
    root.mkdir(parents=True)
    write_json(root / "model_identity.json", read(old["identity_path"]))
    protocol = {**old, "created_utc": now(), "source": source_identity(ROOT),
        "identity_path": str(root / "model_identity.json"), "previous": str(previous),
        "visual_prior_path": str(prior), "methods": [METHOD],
        "training": {**old["training"], "trained_head_kinds": ["image"],
                     "context_mode": "image", "warm_start": "completed_qwen35_visual_step320"},
        "limits": ["Image-global context only; no question tokens, recovery, or adaptive budget.",
                   "Warm-start the same completed visual prior as the question-context arm; train one additional 320-step pass.",
                   "Native dynamic resolution and fixed post-merger K128; all scoring uses the MLLM-Comp lmms-eval entry.",
                   "Reuse completed dense and other-method full results; one training seed, no test-based checkpoint selection."]}
    for name in ("head.safetensors", "head_config.json"):
        protocol["input_hashes"][str(prior / name)] = file_sha(prior / name)
    write_json(root / "protocol.json", protocol)
    write_json(root / "manifest.json", {"status": "prepared", "created_utc": now(),
        "protocol_sha256": file_sha(root / "protocol.json"), "source": protocol["source"]})


def config(root, p, phase, task, run):
    cfg = shared.config(root, p, "eadp", phase, task, run)
    folder = root / "training/image"
    pointer = read(folder / "latest.json")
    if pointer["step"] != 320 or read(folder / "manifest.json")["status"] != "complete":
        raise ValueError("Only a completed fixed step320 image context head can be evaluated")
    name = "qwen35_image_context_projector"
    stage, = cfg["pipeline"]["stages"]
    stage.update(name=METHOD, algorithm=name, plugin=name, variant=f"{name}.image",
                 binding=f"{name}.image.qwen3_5", conditioning_modalities=[])
    stage["config"] = {"method": METHOD, "target_count": 128, "identity_path": p["identity_path"],
                       "checkpoint_path": str(folder / pointer["checkpoint"])}
    cfg["experiment"]["name"] = f"qwen35_image_context_{phase}_{task}"
    return cfg


def check(run, phase, task):
    result = shared.check(run, METHOD, phase, task)
    for line in (run / "compression_runtime.requests.jsonl").open():
        record = json.loads(line)["runtime"]["activations"][0]["selection"]
        if record.get("method") != METHOD or record.get("context_mode") != "image" or record.get("reads_question") is not False:
            raise ValueError("Image-only conditioning trace is incorrect")
    return result


def evaluate(root, p, gpu, phase, task):
    import yaml
    directory = root / "eval" / phase / METHOD / task
    marker = directory / "complete.json"
    if marker.exists():
        completed = read(marker)
        if file_sha(completed["config"]) != completed["config_sha256"]:
            raise ValueError("Completed evaluation config changed")
        check(Path(completed["run"]), phase, task)
        return
    stamp = time.time_ns()
    run = directory / f"attempt_{stamp}"
    filename = root / "configs" / f"{phase}_{task}_{stamp}.yaml"
    filename.parent.mkdir(parents=True, exist_ok=True)
    write_text(filename, yaml.safe_dump(config(root, p, phase, task, run), sort_keys=False))
    shared.execute(root, gpu, [sys.executable, "-m", "mllm_comp.cli.main", "eval", "--config", str(filename)],
        directory / f"launch_{stamp}.log", extra={"MLLM_TOPK_PERFORMANCE": str(run / "request_performance.jsonl")})
    write_json(marker, {"finished_utc": now(), "run": str(run), "config": str(filename),
        "config_sha256": file_sha(filename), **check(run, phase, task)})


def gpu_gate(root, output):
    import torch
    from mllm_comp.analysis.common.qwen35_transfer import load_bundle, load_head, prepare_training_example
    from mllm_comp.compressors.comparison.qwen35_image_context import create_image_head, Qwen35ImageContextProjector
    from scripts.train.qwen25_vl.train_learned_hybrid import load_examples
    p = shared.validate(root)
    identity = read(p["identity_path"])
    bundle = load_bundle(identity, "cuda:0")
    model = bundle["model"]
    visual = load_head(p["visual_prior_path"], identity=identity, device="cpu")
    module = create_image_head(identity["hidden_size"], visual=visual).cuda()
    state = prepare_training_example(bundle, load_examples(Path(p["training_manifest"]))[0])
    with torch.no_grad():
        initial = module(state["features"])
        plain = visual.cuda()(state["features"])
    if not torch.equal(initial["scores"], plain["scores"]) or not torch.equal(initial["indices"], plain["indices"]):
        raise ValueError("Zero-initialized context changed the visual prior")
    class NoQuestionTokenizer:
        def encode(self, *args, **kwargs):
            raise AssertionError("Image context must not tokenize the question")
    call = dict(max_new_tokens=3, min_new_tokens=3, do_sample=False, repetition_penalty=1.05,
                return_dict_in_generate=True, output_scores=True)
    with torch.no_grad():
        dense = model.generate(**state["prompt"], **call)
    traces = []
    for target in (128, len(state["features"])):
        session = Qwen35ImageContextProjector(model, {"method": METHOD, "target_count": target},
            module=module, tokenizer=NoQuestionTokenizer()).install()
        try:
            with torch.no_grad():
                actual = model.generate(**state["prompt"], **call)
            trace = session.runtime_metadata()["selection"]
            if trace["result_count"] != min(target, len(state["features"])) or not trace["decode_position_verified"]:
                raise ValueError("Real image context generation/count/cache gate failed")
            if target == len(state["features"]) and (not torch.equal(actual.sequences, dense.sequences) or any(
                    not torch.equal(a, b) for a, b in zip(actual.scores, dense.scores))):
                raise ValueError("Dense identity generation scores differ")
            traces.append(trace)
        finally:
            session.uninstall()
    write_json(output / "complete.json", {"status": "complete", "finished_utc": now(),
        "zero_context_equals_visual_prior": True, "dense_identity_scores_equal": True,
        "question_tokenizer_never_read": True, "records": traces})


def report(root, p):
    records = []
    for task in shared.COUNTS:
        marker = root / "eval/full" / METHOD / task / "complete.json"
        if marker.exists():
            c = read(marker)
            if task == "ai2d":
                rows = [json.loads(s) for s in Path(c["samples"]).read_text().splitlines() if s.strip()]
                c["leading_option_score"] = sum(r["leading_option_accuracy"] for r in rows) / len(rows)
            records.append({"task": task, **c})
    write_json(root / "summary.json", {"completed_cells": len(records), "total_cells": 3, "results": records})
    lines = ["# Qwen3.5 image-global context", "", "| Method | ChartQA | OCRBench | AI2D leading option |", "|---|---:|---:|---:|"]
    for method in ("dense", "visual_selector", "question_selector", METHOD):
        scores = []
        base = root if method == METHOD else Path(p["previous"])
        for task in shared.COUNTS:
            marker = base / "eval/full" / method / task / "complete.json"
            if not marker.exists():
                scores.append("pending")
                continue
            c = read(marker)
            score = c["score"]
            if task == "ai2d":
                rows = [json.loads(s) for s in Path(c["samples"]).read_text().splitlines() if s.strip()]
                score = sum(r["leading_option_accuracy"] for r in rows) / len(rows)
            scores.append(f"{score*1000:.0f}" if task == "ocrbench" else f"{score*100:.2f}%")
        lines.append("| " + " | ".join((method, *scores)) + " |")
    write_text(root / "report.md", "\n".join(lines + ["", *p["limits"]]) + "\n")


def pipeline(root, gpus):
    p = shared.validate(root)
    with (root / "pipeline.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        state = read(root / "manifest.json")
        def stage(name, **extra):
            state.update(status="running", stage=name, updated_utc=now(), pid=os.getpid(), gpus=gpus, **extra)
            write_json(root / "manifest.json", state)
        def train():
            for smoke, output in [(True, root / "gates/training_smoke"), (False, root / "training/image")]:
                if (output / "manifest.json").exists() and read(output / "manifest.json")["status"] == "complete":
                    continue
                command = [sys.executable, str(ROOT / "scripts/train/train_qwen35_transfer.py"), "--root", str(root),
                           "--kind", "image", "--output", str(output)]
                if smoke:
                    command.append("--smoke")
                shared.execute(root, gpus[0], command, output / f"launch_{time.time_ns()}.log", training=True)
        def gate():
            output = root / "gates/generation"
            if not (output / "complete.json").exists():
                shared.execute(root, gpus[-1], [sys.executable, str(Path(__file__).resolve()), "gpu-gate", "--root", str(root),
                    "--output", str(output)], output / f"launch_{time.time_ns()}.log")
        try:
            stage("training_smoke_then_step320_and_generation_gate")
            if len(gpus) == 1:
                gate()
                train()
            else:
                with ThreadPoolExecutor(max_workers=2) as pool:
                    jobs = [pool.submit(train), pool.submit(gate)]
                    for job in jobs:
                        job.result()
            stage("full_evaluation_queue")
            queue = Queue()
            for task in shared.COUNTS:
                queue.put(task)
            def worker(gpu):
                while True:
                    try:
                        task = queue.get_nowait()
                    except Empty:
                        return
                    for phase in ("smoke", "full"):
                        write_json(root / f"worker_gpu{gpu}.json", {"status": "running", "stage": f"{phase}:{task}", "gpu": gpu, "updated_utc": now()})
                        evaluate(root, p, gpu, phase, task)
                    report(root, p)
                    queue.task_done()
            with ThreadPoolExecutor(max_workers=len(gpus)) as pool:
                jobs = [pool.submit(worker, gpu) for gpu in gpus]
                for job in jobs:
                    job.result()
            report(root, p)
            if read(root / "summary.json")["completed_cells"] != 3:
                raise ValueError("Incomplete full evaluation")
            state.update(status="complete", stage="complete", finished_utc=now())
        except BaseException as error:
            state.update(status="failed", error=str(error), traceback=traceback.format_exc())
            raise
        finally:
            report(root, p)
            write_json(root / "manifest.json", state)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("prepare", "launch", "pipeline", "gpu-gate", "report"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--previous", type=Path, default=PREVIOUS)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--gpus", type=int, nargs="+", default=[0, 1])
    args = parser.parse_args()
    root = owned_output(args.root)
    if args.command == "prepare":
        prepare(root, args.previous.resolve())
    elif args.command == "pipeline":
        pipeline(root, args.gpus)
    elif args.command == "gpu-gate":
        gpu_gate(root, owned_output(args.output))
    elif args.command == "report":
        report(root, shared.validate(root))
    else:
        shared.validate(root)
        command = [sys.executable, str(Path(__file__).resolve()), "pipeline", "--root", str(root), "--gpus", *map(str, args.gpus)]
        log = root / f"coordinator_{time.time_ns()}.log"
        with log.open("x") as stream:
            child = subprocess.Popen(command, cwd=ROOT, env=os.environ, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
        record = {"pid": child.pid, "command": command, "log": str(log), "started_utc": now()}
        write_json(root / "coordinator_process.json", record)
        print(json.dumps(record), flush=True)


if __name__ == "__main__":
    main()
