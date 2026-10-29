"""Resume-safe external-baseline corrections; reuse unchanged full evaluations."""
import argparse
import ast
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import fcntl
import glob
import importlib.util
import json
import os
from pathlib import Path
from queue import Empty, Queue
import subprocess
import sys
from threading import Event
import time
import traceback

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from mllm_comp.analysis.common.anchor7b_runtime import file_sha
from mllm_comp.analysis.common.learned_topk_experiment import owned_output, source_identity
from mllm_comp.core.serialization import atomic_write_json as write_json, atomic_write_text as write_text
from scripts.eval.lmms_eval import run_qwen35_transfer as previous_runner

PREVIOUS = Path("/home/ZhouFengyu/fastv-repro/MLLM-Comp-Platform/MLLM-Comp-Platform-main/outputs/eval/qwen35_vl/transfer_k128_20261026_v1")
CHANGED = ("mmtok", "bifocal_stage1", "eadp")
SOURCES = {
    "eadp": Path("/home/ZhouFengyu/fastv-repro/research_sources/EADP/Qwen_vl/model/pruner.py"),
    "prunesid": Path("/home/ZhouFengyu/fastv-repro/research_sources/PruneSID/prunesid/prunesid_qwen/modeling_qwen2_vl.py"),
    "mmtok": Path("/home/ZhouFengyu/fastv-repro/MMTok/mmtok/core/semantic_selector.py"),
    "bifocal_query": Path("/home/ZhouFengyu/fastv-repro/research_sources/Bifocal-audit-20261027/bifocal/query.py"),
}


def read(path):
    return json.loads(Path(path).read_text())


def now():
    return datetime.now(timezone.utc).isoformat()


def prepare(root, previous):
    if root.exists():
        raise ValueError("Use a fresh result directory")
    old = read(previous / "protocol.json")
    if read(previous / "manifest.json")["status"] != "complete":
        raise ValueError("Reference matrix is incomplete")
    hashes = {name: value for name, value in old["input_hashes"].items()
              if name == old["cohort_path"] or any(folder in name for folder in ("/ChartQA/", "/OCRBench/", "/ai2d/"))}
    for name, checksum in hashes.items():
        if file_sha(name) != checksum:
            raise ValueError(f"Changed evaluation input: {name}")
    root.mkdir(parents=True)
    write_json(root / "model_identity.json", read(old["identity_path"]))
    protocol = {
        "created_utc": now(), "previous": str(previous), "source": source_identity(ROOT),
        "identity_path": str(root / "model_identity.json"), "cohort_path": old["cohort_path"],
        "input_hashes": hashes, "dependencies": old["dependencies"], "eval": old["eval"],
        "counts": old["counts"], "changed_methods": list(CHANGED), "reused_methods": ["dense", "prunesid", "old_mlp", "visual_selector", "question_selector", "protected_selector"],
        "official_sources": {name: {"path": str(path), "sha256": file_sha(path)} for name, path in SOURCES.items()},
        "source_policy": "isolated source snapshot; no new Git commit; previous trained heads stay frozen",
        "limits": ["Released Qwen selection/text recipes with native dynamic resolution and fixed post-merger K128.",
                   "Qwen3.5 remains a controlled adaptation; this does not certify original paper protocols.",
                   "Bifocal Stage1 component only; hybrid-decoder Stage2 is outside this correction.",
                   "Unchanged dense/own-method/PruneSID full predictions are referenced, never counted as reruns.",
                   "No training or checkpoint selection; shared-GPU timings are descriptive."]}
    write_json(root / "protocol.json", protocol)
    write_json(root / "manifest.json", {"status": "prepared", "created_utc": now(),
        "protocol_sha256": file_sha(root / "protocol.json"), "source": protocol["source"]})
    return protocol


def validate(root):
    p = read(root / "protocol.json")
    if file_sha(root / "protocol.json") != read(root / "manifest.json")["protocol_sha256"]:
        raise ValueError("Protocol changed")
    if source_identity(ROOT) != p["source"]:
        raise ValueError("Frozen execution source changed")
    for name, checksum in p["input_hashes"].items():
        if file_sha(name) != checksum:
            raise ValueError(f"Changed evaluation input: {name}")
    for value in p["official_sources"].values():
        if file_sha(value["path"]) != value["sha256"]:
            raise ValueError("Official source snapshot changed")
    import transformers
    if transformers.__version__ != p["dependencies"]["transformers_import"]["version"]:
        raise ValueError("Wrong native Transformers runtime")
    return p


def config(root, protocol, method, phase, task, run):
    result = previous_runner.config(root, protocol, method, phase, task, run)
    stage, = result["pipeline"]["stages"]
    name = "qwen35_reference_projector"
    stage.update(algorithm=name, plugin=name, variant=f"{name}.image", binding=f"{name}.image.qwen3_5")
    stage["config"] = {"method": method, "target_count": 128}
    result["experiment"]["name"] = f"qwen35_reference_v2_{phase}_{method}_{task}"
    return result


def check(run, method, phase, task):
    result = previous_runner.check(run, method, phase, task)
    for line in (run / "compression_runtime.requests.jsonl").open():
        record = json.loads(line)["runtime"]["activations"][0]["selection"]
        if record.get("adapter_version") != "qwen_native_reference_v2":
            raise ValueError("Wrong baseline implementation")
        if method == "mmtok" and (record["text_temperature"] != .01 or record["question_recipe"] != "mmtok_qwen_Question_prefix_split_and_retokenize"):
            raise ValueError("MMTok Qwen recipe regression")
        if method == "bifocal_stage1" and record["question_recipe"] != "bifocal_query_without_task_suffix_lowercase_words_mean_subwords":
            raise ValueError("Bifocal word-input regression")
    return result


def evaluate(root, p, gpu, method, phase, task):
    import yaml
    directory = root / "eval" / phase / method / task
    marker = directory / "complete.json"
    if marker.exists():
        completed = read(marker)
        if file_sha(completed["config"]) != completed["config_sha256"]:
            raise ValueError("Completed config changed")
        check(Path(completed["run"]), method, phase, task)
        return
    stamp = time.time_ns()
    run = directory / f"attempt_{stamp}"
    filename = root / "configs" / f"{phase}_{method}_{task}_{stamp}.yaml"
    filename.parent.mkdir(parents=True, exist_ok=True)
    write_text(filename, yaml.safe_dump(config(root, p, method, phase, task, run), sort_keys=False))
    previous_runner.execute(root, gpu, [sys.executable, "-m", "mllm_comp.cli.main", "eval", "--config", str(filename)],
        directory / f"launch_{stamp}.log", extra={"MLLM_TOPK_PERFORMANCE": str(run / "request_performance.jsonl"),
                                                "MLLM_TOPK_COHORT": p["cohort_path"]})
    write_json(marker, {"finished_utc": now(), "run": str(run), "config": str(filename),
        "config_sha256": file_sha(filename), **check(run, method, phase, task)})


def load_source(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def official_prunesid():
    import math
    import numpy as np
    import torch
    tree = ast.parse(SOURCES["prunesid"].read_text())
    funcs = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in {"pca_group", "nms"}]
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "Qwen2VisionTransformerPretrainedModel_prunesid")
    fwd = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "forward")
    begin = next(i for i, n in enumerate(fwd.body) if isinstance(n, ast.Assign) and any(isinstance(t, ast.Tuple) for t in n.targets))
    fn = ast.FunctionDef(name="reference", args=ast.arguments(posonlyargs=[], args=[ast.arg(arg="hidden_states"), ast.arg(arg="need_token_num")],
        kwonlyargs=[], kw_defaults=[], defaults=[]), body=fwd.body[begin:], decorator_list=[])
    scope = {"torch": torch, "F": torch.nn.functional, "np": np, "math": math}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[*funcs, fn], type_ignores=[])), str(SOURCES["prunesid"]), "exec"), scope)
    return scope["reference"]


def gpu_gate(root, output):
    import torch
    import pyarrow.parquet as pq
    from mllm_comp.analysis.common.qwen35_transfer import load_bundle
    from mllm_comp.analysis.common.learned_hybrid_runtime import move_inputs
    from mllm_comp.backbones.qwen3_5 import Qwen3_5Sample
    from mllm_comp.backbones.qwen3_5.vision_attention import capture_qwen3_5_premerger_features, pool_premerger_features_to_merger_tokens
    from mllm_comp.compressors.comparison.qwen35_reference import (
        METHODS, Qwen35ReferenceProjector, reference_question_embeddings, reference_selection,
    )
    from mllm_comp.compressors.comparison.qwen35_topk import Qwen35TransferredProjector
    from configs.eval.lmms_eval.tasks.qwen25_7b_projector_ratio.utils import decoded
    p = validate(root)
    bundle = load_bundle(read(p["identity_path"]), "cuda:0")
    model, tokenizer = bundle["model"], bundle["processor"].tokenizer
    official_e = load_source("reference_eadp", SOURCES["eadp"])
    official_m = load_source("reference_mmtok", SOURCES["mmtok"])
    official_b = load_source("reference_bifocal_query", SOURCES["bifocal_query"])
    official_p = official_prunesid()
    cohort = read(p["cohort_path"])
    records = []
    output.mkdir(parents=True, exist_ok=True)
    for task in previous_runner.COUNTS:
        previous = read(Path(p["previous"]) / "eval/full/dense" / task / "complete.json")
        first = json.loads(Path(previous["samples"]).open().readline())
        files = cohort[task].get("paths", [cohort[task].get("path")])
        tables = [pq.read_table(path) for path in files]
        import pyarrow as pa
        table = pa.concat_tables(tables)
        document = table.slice(cohort[task]["rows"][0]["row"], 1).to_pylist()[0]
        image = decoded(document)["image"]
        text = first["input"]
        sample = Qwen3_5Sample(f"reference_gate/{task}", [
            {"role": "system", "content": [{"type": "text", "text": "You are a helpful assistant."}]},
            {"role": "user", "content": [{"type": "image", "image": image, "min_pixels": 200704, "max_pixels": 1605632},
                                         {"type": "text", "text": text}]}], enable_thinking=False)
        batch = bundle["facade"].processor_adapter.prepare_inputs(bundle["processor"], sample, phase="eval")
        inputs = move_inputs(batch.model_inputs, bundle["device"])
        with torch.no_grad(), capture_qwen3_5_premerger_features(model.model.visual) as captured:
            features, = model.get_image_features(inputs["pixel_values"], inputs["image_grid_thw"], return_dict=True).pooler_output
        vision, = pool_premerger_features_to_merger_tokens(captured[0], inputs["image_grid_thw"], spatial_merge_size=2)
        h, w = (inputs["image_grid_thw"][0, 1:] // 2).tolist()
        for method in ("eadp", "mmtok", "prunesid"):
            q, _ = reference_question_embeddings(model, tokenizer, inputs, method)
            if method != "prunesid":
                if method == "mmtok":
                    tokens = tokenizer(f"Question: {text}".split(), is_split_into_words=True, return_tensors="pt", padding=True, truncation=True)["input_ids"]
                    if tokens.shape[1] > 1 and int(tokens[0, 0]) == tokenizer.bos_token_id:
                        tokens = tokens[:, 1:]
                else:
                    tokens = tokenizer(text.strip(), add_special_tokens=True, return_tensors="pt", padding=True, truncation=True)["input_ids"]
                with torch.no_grad():
                    expected_text = model.get_input_embeddings()(tokens.cuda())[0]
                if not torch.equal(q, expected_text):
                    raise ValueError(f"Question recipe differs from independent reference: {method}/{task}")
            out, _ = reference_selection(features, method=method, question=q, grid=(h, w), raw_features=vision)
            if method == "eadp":
                pruner = official_e.VisualTokenPruner(visual_token_num=128, alpha=.5, beta=2.)
                importance = pruner.compute_importance(features[None], q.mean(0, keepdim=True), q[None], h, w)
                expected, _ = official_e._greed_select_impl(importance, official_e._sim_visual_impl(features[None]), 128)
                expected = expected[0].sort().values
            elif method == "mmtok":
                expected, _ = official_m.SemanticTokenSelector(128).mm_coverage_selection(q, features, vision, tv_temp=.01, vv_temp=.2)
                expected = torch.tensor(expected, device=features.device)
            else:
                with torch.random.fork_rng(devices=[0]):
                    torch.manual_seed(0)
                    _, expected = official_p(features, 128)
            if not torch.equal(out["indices"], expected):
                raise ValueError(f"Released selection/order mismatch: {method}/{task}")
            records.append({"task": task, "method": method, "official_same_order": True, "N": len(features)})
        q, _ = reference_question_embeddings(model, tokenizer, inputs, "bifocal_stage1")
        from mllm_comp.compressors.image.bifocal import bifocal_extract_query
        query = bifocal_extract_query(text)
        expected = official_b.word_embeddings(tokenizer, model.get_input_embeddings(), model.device, query)
        if not torch.equal(q, expected):
            raise ValueError("Released Bifocal word embedding mismatch")
        call = dict(max_new_tokens=3, min_new_tokens=3, do_sample=False, repetition_penalty=1.05,
                    return_dict_in_generate=True, output_scores=True)
        with torch.no_grad():
            dense = model.generate(**inputs, **call)
        for method in METHODS:
            session = Qwen35ReferenceProjector(model, {"method": method}, tokenizer=tokenizer).install()
            try:
                with torch.no_grad():
                    compressed = model.generate(**inputs, **call)
                trace = session.runtime_metadata()["selection"]
                if trace["result_count"] != 128 or not trace["decode_position_verified"]:
                    raise ValueError("Real-model count/cache verification failed")
                records.append({"task": task, "method": method, "count_cache_passed": True, "N": len(features)})
            finally:
                session.uninstall()
            if method == "prunesid":
                old = Qwen35TransferredProjector(model, {"method": method, "target_count": 128}, tokenizer=tokenizer).install()
                try:
                    with torch.no_grad():
                        old_output = model.generate(**inputs, **call)
                    if not torch.equal(old_output.sequences, compressed.sequences) or any(
                            not torch.equal(a, b) for a, b in zip(old_output.scores, compressed.scores)):
                        raise ValueError("Unchanged PruneSID cannot reuse old results")
                finally:
                    old.uninstall()
            identity = Qwen35ReferenceProjector(model, {"method": method, "target_count": len(features)}, tokenizer=tokenizer).install()
            try:
                with torch.no_grad():
                    actual = model.generate(**inputs, **call)
                if not torch.equal(actual.sequences, dense.sequences) or any(
                        not torch.equal(a, b) for a, b in zip(actual.scores, dense.scores)):
                    raise ValueError(f"Dense identity logits/generation differ: {method}/{task}")
            finally:
                identity.uninstall()
        write_json(output / "progress.json", {"task_completed": task, "records": records})
        del table, tables
    write_json(output / "complete.json", {"status": "complete", "finished_utc": now(), "records": records,
        "dense_identity_scores_equal": True, "prunesid_reuse_generation_scores_equal": True,
        "peak_allocated_bytes": torch.cuda.max_memory_allocated()})


def report(root, p):
    completed = []
    for method in CHANGED:
        for task in previous_runner.COUNTS:
            marker = root / "eval/full" / method / task / "complete.json"
            if marker.exists():
                completed.append({"method": method, "task": task, **read(marker)})
    write_json(root / "summary.json", {"completed_cells": len(completed), "total_cells": 9, "new_results": completed,
        "reused_from": p["previous"], "reused_methods": p["reused_methods"]})
    lines = ["# Qwen3.5 native-reference corrections", "", "| Method | ChartQA | OCRBench | AI2D strict |", "|---|---:|---:|---:|"]
    for method in ("dense", "eadp", "prunesid", "mmtok", "bifocal_stage1", "old_mlp", "visual_selector", "question_selector", "protected_selector"):
        scores = []
        for task in previous_runner.COUNTS:
            base = root if method in CHANGED else Path(p["previous"])
            marker = base / "eval/full" / method / task / "complete.json"
            if marker.exists():
                score = read(marker)["score"]
                scores.append(f"{score*1000:.0f}" if task == "ocrbench" else f"{score*100:.2f}%")
            else:
                scores.append("pending")
        lines.append("| " + " | ".join((method, *scores)) + " |")
    write_text(root / "report.md", "\n".join(lines + ["", *p["limits"]]) + "\n")


def pipeline(root, gpus):
    p = validate(root)
    with (root / "pipeline.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        state = read(root / "manifest.json")
        state.update(status="running", pid=os.getpid(), gpus=gpus, started_utc=now())
        write_json(root / "manifest.json", state)
        queue = Queue()
        for task in previous_runner.COUNTS:
            for method in CHANGED:
                queue.put((method, task))
        stop = Event()

        def worker(gpu):
            marker = root / f"worker_gpu{gpu}.json"
            def stage(name, status="running", **extra):
                write_json(marker, {"status": status, "stage": name, "gpu": gpu, "updated_utc": now(), **extra})
            try:
                stage("real_9b_reference_gate")
                gate = root / "gates" / f"gpu{gpu}"
                if not (gate / "complete.json").exists():
                    previous_runner.execute(root, gpu, [sys.executable, str(ROOT / "scripts/eval/lmms_eval/run_qwen35_reference.py"),
                        "gpu-gate", "--root", str(root), "--output", str(gate)], gate / f"launch_{time.time_ns()}.log")
                while not stop.is_set():
                    try:
                        method, task = queue.get_nowait()
                    except Empty:
                        break
                    for phase in ("smoke", "full"):
                        stage(f"{phase}:{method}:{task}")
                        evaluate(root, p, gpu, method, phase, task)
                    report(root, p)
                    queue.task_done()
                stage("complete", "complete")
            except BaseException as error:
                stop.set()
                stage("failed", "failed", error=str(error), traceback=traceback.format_exc())
                raise
        try:
            with ThreadPoolExecutor(max_workers=len(gpus)) as pool:
                jobs = [pool.submit(worker, gpu) for gpu in gpus]
                for job in jobs:
                    job.result()
            report(root, p)
            if read(root / "summary.json")["completed_cells"] != 9:
                raise ValueError("Incomplete correction matrix")
            state.update(status="complete", finished_utc=now())
        except BaseException as error:
            state.update(status="failed", error=str(error), traceback=traceback.format_exc())
            raise
        finally:
            report(root, p)
            write_json(root / "manifest.json", state)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("prepare", "pipeline", "gpu-gate", "report", "launch"))
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
        report(root, validate(root))
    else:
        validate(root)
        log = root / f"coordinator_{time.time_ns()}.log"
        command = [sys.executable, str(Path(__file__).resolve()), "pipeline", "--root", str(root), "--gpus", *map(str, args.gpus)]
        with log.open("x") as stream:
            child = subprocess.Popen(command, cwd=ROOT, env=os.environ, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
        process = {"pid": child.pid, "command": command, "log": str(log), "started_utc": now()}
        write_json(root / "coordinator_process.json", process)
        print(json.dumps(process), flush=True)


if __name__ == "__main__":
    main()
