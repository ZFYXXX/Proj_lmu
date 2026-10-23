"""Two-GPU, fixed-definition L1/correlation/Mahalanobis/response metric study."""
import argparse
from collections import defaultdict
from datetime import datetime, timezone
import fcntl
import importlib.metadata
from io import BytesIO
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from mllm_comp.analysis.common.token_metric_ablation import METHODS, NEW_METHODS, unit, regularized_diagonal
from mllm_comp.analysis.common.evidence_interaction import bootstrap, holm, digest
from mllm_comp.analysis.common.learned_hybrid_runtime import CHECKPOINT, REVISION, load_runtime, prepare_state
from mllm_comp.analysis.common.visual_token_phenomena import media_fingerprint
from mllm_comp.analysis.probes.token_metric_ablation import TokenMetricAblation, Session, implementation_hash, file_sha
from mllm_comp.analysis.probes.anchor_distance_ablation import implementation_hash as legacy_hash
from mllm_comp.core.serialization import atomic_write_json
from scripts.analysis.qwen25_vl.run_anchor_distance_ablation import expected_entries


def calibration_selection(entries, excluded):
    selected = []
    for source in ["textvqa_train", "scienceqa_img_train"]:
        eligible = [e for e in entries if e["source"] == source and e["image_fingerprint"] not in excluded]
        # Some historical manifests use scienceqa_train rather than _img_train.
        if source == "scienceqa_img_train" and not eligible:
            eligible = [e for e in entries if "scienceqa" in e["source"] and e["image_fingerprint"] not in excluded]
        eligible.sort(key=lambda e:(digest("token-metric-calibration-seed42:"+e["image_fingerprint"]), e["sample_id"]))
        if len(eligible) < 32:
            raise ValueError("Need 32 disjoint calibration images per training source")
        selected.extend(eligible[:32])
    if len({e["image_fingerprint"] for e in selected}) != 64:
        raise ValueError("Calibration images must be unique")
    return selected


def prepare(root, source, training):
    import yaml
    root.mkdir(parents=True, exist_ok=True)
    if (root/"protocol.json").exists():
        return validate(root, calibration=False)
    prior = json.loads((source/"protocol.json").read_text())
    if json.loads((source/"run_manifest.json").read_text())["status"] != "complete" or prior["implementation_hash"] != legacy_hash():
        raise ValueError("Require complete, unchanged previous 500-image experiment")
    cohort = json.loads((source/"cohort.json").read_text())
    entries = cohort["entries"]
    if len(entries) != 500 or len({e["image_fingerprint"] for e in entries}) != 500:
        raise ValueError("Must reuse exactly the same 500 independent images")
    training_path = training/"training_samples.json"
    selected = calibration_selection(json.loads(training_path.read_text())["entries"], {e["image_fingerprint"] for e in entries})
    (root/"cohort.json").write_bytes((source/"cohort.json").read_bytes())
    atomic_write_json(root/"calibration_manifest.json", {"entries": selected, "n_images": 64,
        "overlap_with_evaluation": 0, "selection_reads_answers_or_predictions": False,
        "feature_extraction_reads_only_parquet_image_column": True,
        "training_manifest": str(training_path), "training_manifest_sha256": file_sha(training_path)})
    configs = root/"configs"; configs.mkdir(exist_ok=True)
    for stage in ["smoke", "development", "confirmation"]:
        for shard in [0, 1]:
            raw = yaml.safe_load((source/"configs"/f"{stage}_{shard}.yaml").read_text())
            raw["experiment"]["name"] = f"token_metric_3b_{stage}_{shard}"
            raw["analysis"]["sampling"]["manifest_path"] = str(root/"cohort.json")
            (configs/f"{stage}_{shard}.yaml").write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
    protocol = {"checkpoint": CHECKPOINT, "revision": REVISION, "implementation_hash": implementation_hash(),
        "source_geometry_run": str(source), "source_protocol_sha256": file_sha(source/"protocol.json"),
        "source_cohort_sha256": file_sha(source/"cohort.json"), "methods": METHODS, "new_methods": NEW_METHODS,
        "n_images": 500, "images_per_task": 250, "budget": 128, "uses_mlp": False, "training": False,
        "dtype": "bfloat16", "metric_dtype": "float32", "attention": "eager", "seed": 42,
        "first_anchor": 0, "tie_break": "lowest source index", "assignment": "nearest metric anchor; anchors pinned to own clusters",
        "aggregation": "Unchanged original-feature FP32 mean -> BF16, rounded native 3-axis centroid M-RoPE, original text positions",
        "definitions": {"unit_l1": "L1 on per-token L2 normalized features",
            "correlation": "Subtract each token's channel mean, L2 normalize, negative dot dissimilarity",
            "diag_mahalanobis": "Euclidean on unit features / sqrt(frozen regularized channel variance); no re-normalization",
            "query_response_kv": "Euclidean on concat(prompt-only native post-MRoPE QK logits / sqrt(HQ), native V / sqrt(Dv)); lambda=1",
            "query_response_js": "sqrt(JS divergence) on softmax(logits) over flattened question-position x head axes; epsilon=1e-12; not visual-key softmax"},
        "calibration": {"n_train_images": 64, "shrinkage": .1, "image_weighted": True,
            "features": "L2 normalized post-merger native 3B features; no question/answer column read", "variance_floor_relative": 1e-6},
        "generation": {"do_sample": False, "repetition_penalty": 1., "chartqa_max_new": 16, "textvqa_max_new": 32},
        "processor": {"use_fast": False, "min_pixels": 200704, "max_pixels": 1605632},
        "statistics": "10,000 paired image bootstrap, 10 prespecified comparisons (5 new methods x 2 tasks) with Holm adjustment; report development/replication/pooled separately",
        "interpretation": "Exploratory reused images, not an independent confirmation. Task generation score is primary; geometry/NLL are explanatory. No claim of metric novelty or deployable speed.",
        "timing": "Record selection, mean+statistics, response extraction, decoder prefill separately. Legacy baseline latency is historical, not contemporaneous. Calibration cost separate.",
        "software": {n:importlib.metadata.version(n) for n in ["torch", "transformers", "numpy", "safetensors"]},
        "gpu": subprocess.check_output(["nvidia-smi", "--query-gpu=index,name,uuid,memory.total,driver_version", "--format=csv"], text=True),
        "checkpoint_file_stats": {p.name:{"bytes":p.stat().st_size, "mtime_ns":p.stat().st_mtime_ns} for p in
            [Path(CHECKPOINT)/"config.json", Path(CHECKPOINT)/"model.safetensors.index.json"]+sorted(Path(CHECKPOINT).glob("*.safetensors"))},
        "created_utc": datetime.now(timezone.utc).isoformat()}
    protocol["artifact_hashes"] = {str(p.relative_to(root)): file_sha(p) for p in
        [root/"cohort.json", root/"calibration_manifest.json"]+sorted(configs.glob("*.yaml"))}
    atomic_write_json(root/"protocol.json", protocol)
    return protocol


def validate(root, calibration=True):
    protocol = json.loads((root/"protocol.json").read_text())
    if protocol["implementation_hash"] != implementation_hash():
        raise ValueError("Metric source changed after freezing")
    for name, expected in protocol["checkpoint_file_stats"].items():
        actual = (Path(CHECKPOINT)/name).stat()
        if actual.st_size != expected["bytes"] or actual.st_mtime_ns != expected["mtime_ns"]:
            raise ValueError("Pinned original 3B model files changed")
    if any(file_sha(root/name) != sha for name, sha in protocol["artifact_hashes"].items()):
        raise ValueError("Frozen config/cohort/calibration selection changed")
    source = Path(protocol["source_geometry_run"])
    if file_sha(source/"protocol.json") != protocol["source_protocol_sha256"] or file_sha(root/"cohort.json") != protocol["source_cohort_sha256"]:
        raise ValueError("Previous experiment/cohort identity changed")
    if calibration:
        info = json.loads((root/"calibration.json").read_text())
        if info["implementation_hash"] != implementation_hash() or info["sha256"] != file_sha(root/"calibration.safetensors") or info["n_images"] != 64:
            raise ValueError("Frozen calibration invalid")
    return protocol


def calibrate(root, device):
    import torch
    import pyarrow.parquet as pq
    from PIL import Image
    from safetensors.torch import save_file
    from mllm_comp.backbones.qwen25_vl import Qwen25VLSample, DEFAULT_SYSTEM_PROMPT
    protocol = validate(root, calibration=False)
    torch.set_num_threads(4); torch.cuda.set_device(device); torch.manual_seed(42)
    torch.backends.cuda.matmul.allow_tf32 = False
    bundle = load_runtime(device)
    manifest = json.loads((root/"calibration_manifest.json").read_text())
    if file_sha(manifest["training_manifest"]) != manifest["training_manifest_sha256"]:
        raise ValueError("Calibration training manifest changed")
    grouped = defaultdict(dict)
    for e in manifest["entries"]:
        grouped[e["path"]][e["row"]] = e
    count, token_count, mean_sum, moment_sum = 0, 0, None, None
    started = time.perf_counter()
    try:
        for path, wanted in grouped.items():
            index = 0
            # Physically do not read question/answer columns.
            for batch in pq.ParquetFile(path).iter_batches(batch_size=8, columns=["image"]):
                for doc in batch.to_pylist():
                    entry = wanted.get(index); index += 1
                    if entry is None:
                        continue
                    data = doc["image"]
                    with Image.open(BytesIO(data["bytes"]) if data.get("bytes") else data["path"]) as loaded:
                        image = loaded.convert("RGB")
                    if media_fingerprint([image]) != entry["image_fingerprint"]:
                        raise ValueError("Calibration image content mismatch")
                    sample = Qwen25VLSample(entry["sample_id"], [
                        {"role": "system", "content": [{"type": "text", "text": DEFAULT_SYSTEM_PROMPT}]},
                        {"role": "user", "content": [{"type": "image", "image": image, "min_pixels": 200704, "max_pixels": 1605632},
                            {"type": "text", "text": "Describe the image."}]}])
                    prompt = bundle["facade"].processor_adapter.prepare_inputs(bundle["processor"], sample, phase="eval")
                    with torch.inference_mode():
                        state = prepare_state(bundle, prompt.model_inputs)
                        values = unit(state["features"]).cpu().double()
                    mean, moment = values.mean(0), values.square().mean(0)
                    mean_sum = mean if mean_sum is None else mean_sum+mean
                    moment_sum = moment if moment_sum is None else moment_sum+moment
                    count += 1; token_count += len(values)
                    del state, values, prompt
                    image.close()
                    print(json.dumps({"calibration_images": count, "expected": 64}), flush=True)
                if index > max(wanted):
                    break
        if count != 64:
            raise ValueError("Incomplete calibration")
        diagonal = regularized_diagonal(mean_sum/count, moment_sum/count, protocol["calibration"]["shrinkage"])
        save_file({"diagonal": diagonal, "feature_mean": (mean_sum/count).float()}, str(root/"calibration.safetensors"))
        atomic_write_json(root/"calibration.json", {"status": "complete", "n_images": count, "n_tokens": token_count,
            "implementation_hash": implementation_hash(), "sha256": file_sha(root/"calibration.safetensors"),
            "calibration_manifest_sha256": file_sha(root/"calibration_manifest.json"), "elapsed_s": time.perf_counter()-started,
            "diagonal_min": float(diagonal.min()), "diagonal_max": float(diagonal.max()), "eval_image_overlap": 0})
    finally:
        bundle["runtime"].close()


def worker(root, stage, shard, device):
    import torch
    from mllm_comp.config.loader import load_config
    from mllm_comp.registry.integrations import get_integration
    validate(root)
    torch.set_num_threads(4); torch.cuda.set_device(device); torch.manual_seed(42)
    torch.backends.cuda.matmul.allow_tf32 = False
    expected = [e for e in expected_entries(root, stage) if e["shard"] == shard]
    status = {"status": "loading", "stage": stage, "shard": shard, "device": device, "completed_images": 0, "expected_images": len(expected)}
    status_path = root/f"{stage}_{shard}_manifest.json"
    atomic_write_json(status_path, status)
    bundle = None
    try:
        bundle = load_runtime(device)
        config = load_config(root/"configs"/f"{stage}_{shard}.yaml")
        session = Session(TokenMetricAblation(), bundle["facade"], bundle["runtime"],
            {"artifact_root": str(root), "cohort_path": str(root/"cohort.json"), "stage": stage},
            "pre_compression", str(root/"lifecycle"/f"{stage}_{shard}"))
        adapter = get_integration("lmms_eval")()
        status["status"] = "running"
        atomic_write_json(status_path, status)
        for sample in adapter.iter_analysis_samples(config=config, backbone=bundle["facade"], runtime=bundle["runtime"],
                sample_limit=None, requires_teacher_forcing=True):
            with torch.inference_mode():
                session.process(sample)
            status["completed_images"] += 1
            atomic_write_json(status_path, status)
        if {s["sample_id"] for s in session.completed} != {e["sample_id"] for e in expected} or len(session.completed) != len(expected):
            raise ValueError("Missing/duplicate metric evaluation images")
        session.finish()
        status.update(status="complete", peak_allocated_bytes=torch.cuda.max_memory_allocated())
    except BaseException as exc:
        status.update(status="failed", error=str(exc), traceback=traceback.format_exc())
        raise
    finally:
        atomic_write_json(status_path, status)
        if bundle:
            bundle["runtime"].close()


def report(root, stage):
    import statistics
    validate(root)
    stages = ["development", "confirmation"] if stage == "all" else [stage]
    expected = expected_entries(root, stage)
    summaries = [json.loads(p.read_text()) for split in stages for p in (root/"samples"/split).glob("*/summary.json")]
    if {s["sample_id"] for s in summaries} != {e["sample_id"] for e in expected} or len(summaries) != len(expected):
        raise ValueError("Incomplete metric report")
    grouped = defaultdict(list)
    for summary in summaries:
        raw = [json.loads(s) for s in Path(summary["condition_path"]).read_text().splitlines()]
        rows = {r["variant"]:r for r in raw}
        if len(raw) != len(METHODS) or set(rows) != set(METHODS):
            raise ValueError("Incomplete condition set")
        if any(r["gold_answer_in_decode_context"] or (name != "dense" and r["retained_visual_tokens"] != 128) for name,r in rows.items()):
            raise ValueError("Gold leakage or token budget error")
        grouped[summary["task"]].append(rows)
    tables, tests = {}, []
    for task, images in sorted(grouped.items()):
        table = {}
        for name in METHODS:
            table[name] = {"n_images": len(images), "mean_score": statistics.mean(i[name]["primary_task_score"] for i in images),
                "mean_nll": statistics.mean(i[name]["answer_nll"] for i in images),
                "score_minus_cosine": bootstrap([i[name]["primary_task_score"]-i["cosine_farthest"]["primary_task_score"] for i in images])}
            if name in NEW_METHODS:
                table[name].update({"median_selection_ms": statistics.median(i[name]["selection_ms"] for i in images),
                    "median_aggregation_and_statistics_ms": statistics.median(i[name]["aggregation_and_statistics_ms"] for i in images),
                    "median_response_extraction_ms": statistics.median(i[name]["response_extraction_ms"] for i in images),
                    "median_decoder_prefill_gpu_ms": statistics.median(i[name]["decoder_prefill_gpu_ms"] for i in images),
                    "anchor_overlap_with_cosine": statistics.mean(i[name]["comparison_to_cosine"]["anchor_overlap_fraction"] for i in images),
                    "raw_relative_sse": statistics.mean(i[name]["geometry"]["raw_relative_sse"] for i in images)})
                tests.append(table[name]["score_minus_cosine"])
        tables[task] = table
    if stage in {"confirmation", "all"}:
        if len(tests) != 10:
            raise ValueError("Expected exactly 5 methods x 2 task comparisons")
        for test, adjusted in zip(tests, holm([v["one_sided_p"] for v in tests])):
            test["holm_adjusted_p"] = adjusted
    result = {"status": "complete", "stage": stage, "n_images": len(summaries), "tables": tables,
        "interpretation": "Exploratory paired-image study on the same previous images, not fresh independent confirmation. Historical baseline timings are not comparable to current timings."}
    atomic_write_json(root/f"{stage}_report.json", result)
    lines = [f"# Token metric study: {stage}", "", result["interpretation"], "",
        "| Task | Metric | Native score | Delta vs cosine (pp), 95% CI |", "|---|---|---:|---:|"]
    for task, table in tables.items():
        for name, row in table.items():
            d = row["score_minus_cosine"]
            lines.append(f"| {task} | {name} | {100*row['mean_score']:.2f}% | {100*d['mean']:+.2f} [{100*d['ci95'][0]:+.2f}, {100*d['ci95'][1]:+.2f}] |")
    (root/f"{stage}_report.md").write_text("\n".join(lines)+"\n", encoding="utf-8")
    return result


def run(root):
    logs = root/"logs"; logs.mkdir(exist_ok=True)
    with (root/"runner.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        validate(root, calibration=False)
        state = {"status": "running", "pid": os.getpid(), "started_utc": datetime.now(timezone.utc).isoformat(),
            "implementation_hash": implementation_hash(), "stages": {}, "n_images": 500, "new_methods": NEW_METHODS}
        children = []
        def execute(stage, commands):
            nonlocal children
            validate(root, calibration=stage != "calibration")
            children = []
            for index, command in enumerate(commands):
                path = logs/f"{stage}_{index}_{int(time.time())}.log"
                stream = path.open("w")
                process = subprocess.Popen(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True,
                    env={**os.environ, "PYTHONUNBUFFERED": "1", "OMP_NUM_THREADS": "4", "TOKENIZERS_PARALLELISM": "false"})
                children.append((process, stream, path))
            state["active_stage"] = stage
            state["stages"][stage] = {"status": "running", "children": [{"pid":p.pid, "log":str(path)} for p,_,path in children]}
            atomic_write_json(root/"run_manifest.json", state)
            while any(p.poll() is None for p,_,_ in children):
                if any(p.poll() not in {None,0} for p,_,_ in children):
                    raise RuntimeError(f"{stage} worker failed; inspect owned logs")
                time.sleep(2)
            if any(p.returncode for p,_,_ in children):
                raise RuntimeError(f"{stage} failed")
            for _, stream, _ in children:
                stream.close()
            state["stages"][stage]["status"] = "complete"
            atomic_write_json(root/"run_manifest.json", state)
        try:
            if not (root/"calibration.json").exists():
                execute("calibration", [[sys.executable, str(Path(__file__)), "--root", str(root), "--calibrate", "--device", "cuda:1"]])
            for stage in ["smoke", "development", "confirmation"]:
                commands = [[sys.executable, str(Path(__file__)), "--root", str(root), "--worker", stage,
                    "--shard", str(shard), "--device", f"cuda:{shard}"] for shard in [0,1]]
                execute(stage, commands)
                report(root, stage)
            report(root, "all")
            state.update(status="complete", completed_utc=datetime.now(timezone.utc).isoformat())
        except BaseException as exc:
            for p,_,_ in children:
                if p.poll() is None:
                    os.killpg(p.pid, signal.SIGTERM); p.wait()
            state.update(status="failed", error=str(exc), traceback=traceback.format_exc())
            raise
        finally:
            atomic_write_json(root/"run_manifest.json", state)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--source-run", type=Path)
    parser.add_argument("--training-run", type=Path)
    parser.add_argument("--launch", action="store_true")
    parser.add_argument("--calibrate", action="store_true")
    parser.add_argument("--worker", choices=["smoke", "development", "confirmation"])
    parser.add_argument("--shard", type=int, choices=[0,1], default=0)
    parser.add_argument("--device", default="cuda:1")
    args = parser.parse_args(); root = args.root.resolve()
    if args.calibrate:
        calibrate(root, args.device)
    elif args.worker:
        worker(root, args.worker, args.shard, args.device)
    else:
        if args.source_run and args.training_run:
            prepare(root, args.source_run.resolve(), args.training_run.resolve())
        elif not (root/"protocol.json").exists():
            parser.error("A fresh run requires --source-run and --training-run")
        if args.launch:
            with (root/"supervisor.log").open("a") as stream:
                p = subprocess.Popen([sys.executable, str(Path(__file__)), "--root", str(root)], cwd=ROOT, stdout=stream,
                    stderr=subprocess.STDOUT, start_new_session=True, env={**os.environ, "PYTHONUNBUFFERED":"1"})
            print(json.dumps({"pid":p.pid, "root":str(root)}))
        else:
            run(root)
