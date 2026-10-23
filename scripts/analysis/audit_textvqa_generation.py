"""Isolate inherited repetition penalty on matched and mismatched saved cases.

This is a targeted implementation diagnostic, not a random quality estimate.
The optional dependency keeps one owned evaluation process per GPU.
"""
import argparse
from collections import Counter
import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from mllm_comp.analysis.common.condition_ledger import append_json, records
from mllm_comp.analysis.common.evidence_interaction import digest
from mllm_comp.analysis.common.learned_hybrid_runtime import load_runtime
from mllm_comp.analysis.common.compression_training_check import evaluate_state
from mllm_comp.analysis.common.visual_token_phenomena import media_fingerprint
from mllm_comp.core.serialization import atomic_write_json
from scripts.train.qwen25_vl.train_learned_hybrid import prepare_example


def compare_saved(root):
    base = ROOT / "outputs/eval/qwen25_vl/textvqa_val_full_k64_20261018_v3/baseline/models__Qwen2.5-VL-3B-Instruct/20261018_130200_samples_qwen25_projector_quality_textvqa_full.jsonl"
    native = {r["doc_id"]: r for r in map(json.loads, base.read_text().splitlines())}
    source = ROOT / "outputs/train/qwen25_vl/learned_hybrid_k128_20261021_v1"
    compared = []
    for stage in ["development", "confirmation"]:
        for path in sorted((source / "samples" / stage).glob("*textvqa*/conditions.jsonl")):
            row = next(r for r in map(json.loads, path.read_text().splitlines()) if r["variant"] == "dense")
            index = int(row["sample_id"].rsplit(":",1)[1])
            prediction = native[index]["filtered_resps"]
            if isinstance(prediction, list):
                prediction, = prediction
            compared.append({"doc_id": index, "sample_id": row["sample_id"], "image_fingerprint": row["image_fingerprint"],
                "historical_native": prediction.strip(), "historical_analysis": row["prediction"].strip(),
                "native_score": native[index]["exact_match"], "analysis_score": row["primary_task_score"],
                "prediction_disagrees": prediction.strip() != row["prediction"].strip()})
    if len(compared) != 500 or len({e["doc_id"] for e in compared}) != 500:
        raise ValueError("Expected 500 matched independently selected images")
    result = {"n_images": len(compared), "n_prediction_disagreements": sum(e["prediction_disagrees"] for e in compared),
        "historical_native_score": sum(e["native_score"] for e in compared)/len(compared),
        "historical_analysis_score": sum(e["analysis_score"] for e in compared)/len(compared),
        "checkpoint_default_repetition_penalty": 1.05, "diagnostic_repetition_penalty": 1.,
        "interpretation": "Generation protocol discrepancy confirmed in code. Attribution of prediction differences requires same-input intervention below.",
        "comparisons": compared}
    root.mkdir(parents=True, exist_ok=True)
    atomic_write_json(root / "generation_path_comparison.json", result)
    return compared


def run(root, device, after):
    compared = compare_saved(root)
    status = {"status": "waiting_for_gpu_dependency" if after else "running", "dependency": str(after) if after else None,
              "device": device, "scope": "All 25 discrepant cases plus 25 hash-selected concordant cases; not an accuracy estimate"}
    atomic_write_json(root / "generation_audit_manifest.json", status)
    try:
        while after:
            if after.exists():
                dependency = json.loads(after.read_text())
                if dependency["status"] == "failed":
                    raise RuntimeError("GPU dependency failed; do not start additional evaluation")
                if dependency["status"] == "complete":
                    break
            time.sleep(10)
        status["status"] = "running"
        atomic_write_json(root / "generation_audit_manifest.json", status)
        import torch
        import pyarrow.parquet as pq
        from io import BytesIO
        from PIL import Image
        selected = [e for e in compared if e["prediction_disagrees"]] + sorted(
            [e for e in compared if not e["prediction_disagrees"]], key=lambda e:digest("generation-control:"+str(e["doc_id"])))[:25]
        wanted = {e["doc_id"]: e for e in selected}
        docs, offset = {}, 0
        for path in sorted(Path("/home/YinPeiran/datasets/lmms_eval_datasets/textvqa/data").glob("validation-*.parquet")):
            for batch in pq.ParquetFile(path).iter_batches(batch_size=16):
                for doc in batch.to_pylist():
                    if offset in wanted:
                        docs[offset] = doc
                    offset += 1
        bundle = load_runtime(device)
        config_hash = digest(json.dumps(selected, sort_keys=True)+"native-penalty-audit-v1")
        path = root / "generation_conditions.jsonl"
        previous = records(path, config_hash)
        for entry in selected:
            key = (entry["sample_id"], "penalty_intervention")
            if key in previous:
                continue
            doc = docs[entry["doc_id"]]
            image = doc["image"]
            fingerprint = media_fingerprint([Image.open(BytesIO(image["bytes"]) if image.get("bytes") else image["path"]).convert("RGB")])
            if fingerprint != entry["image_fingerprint"]:
                raise ValueError("Generation audit image mismatch")
            state = prepare_example(bundle, {**entry, "image": image, "question": doc["question"].capitalize()+
                "\nAnswer the question using a single word or phrase.", "target": Counter(doc["answers"]).most_common(1)[0][0]})
            with torch.inference_mode():
                manual = evaluate_state(bundle, state)
                outputs = {}
                for penalty in [1., 1.05]:
                    config = copy.deepcopy(bundle["model"].generation_config)
                    config.do_sample, config.temperature, config.top_p, config.top_k = False, None, None, None
                    config.max_new_tokens, config.repetition_penalty, config.use_cache = 32, penalty, True
                    ids = bundle["model"].generate(**state["prompt_inputs"], generation_config=config)[0,state["prefix"]:].tolist()
                    outputs[str(penalty)] = {"prediction": bundle["processor"].tokenizer.decode(ids, skip_special_tokens=True).strip(),
                                            "generated_token_ids": ids}
            row = {**entry, "variant": "penalty_intervention", "config_hash": config_hash, "status": "completed",
                   "manual_penalty_1": manual["prediction"], "native_penalties": outputs,
                   "native_1_matches_manual_tokens": outputs["1.0"]["generated_token_ids"] == manual["generated_token_ids"],
                   "native_105_matches_historical_native": outputs["1.05"]["prediction"] == entry["historical_native"],
                   "native_1_matches_historical_analysis": outputs["1.0"]["prediction"] == entry["historical_analysis"],
                   "selection_uses_answers": False, "gold_answer_in_decode_context": False}
            append_json(path, row)
            previous[key] = row
            print(json.dumps({"completed":len(previous), "expected":len(selected)}), flush=True)
        if len(previous) != len(selected):
            raise ValueError("Incomplete targeted generation audit")
        summary = {}
        for group, rows in [("discrepant", [r for r in previous.values() if r["prediction_disagrees"]]),
                            ("concordant_controls", [r for r in previous.values() if not r["prediction_disagrees"]])]:
            summary[group] = {"n":len(rows), **{field:sum(r[field] for r in rows) for field in [
                "native_1_matches_manual_tokens", "native_105_matches_historical_native", "native_1_matches_historical_analysis"]}}
        atomic_write_json(root / "generation_audit_summary.json", {"status":"complete", "groups":summary,
            "scope":status["scope"], "warning":"If matched inputs with explicit penalties do not reproduce both histories, investigate preprocessing or numerical/runtime differences; do not blame penalty alone."})
        status["status"] = "complete"
    except BaseException as exc:
        status.update({"status":"failed", "error":str(exc), "traceback":traceback.format_exc()})
        raise
    finally:
        atomic_write_json(root / "generation_audit_manifest.json", status)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--root",type=Path,required=True)
    p.add_argument("--device",default="cuda:1")
    p.add_argument("--after",type=Path)
    p.add_argument("--launch",action="store_true")
    args = p.parse_args()
    root = args.root.resolve()
    if args.launch:
        compare_saved(root)
        with (root/"generation_supervisor.log").open("a") as stream:
            command = [sys.executable,str(Path(__file__).resolve()),"--root",str(root),"--device",args.device]
            if args.after:
                command += ["--after",str(args.after.resolve())]
            child = subprocess.Popen(command,cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True,
                env={**os.environ,"PYTHONUNBUFFERED":"1","OMP_NUM_THREADS":"4"})
        print(json.dumps({"pid":child.pid,"root":str(root)}))
    else:
        run(root,args.device,args.after.resolve() if args.after else None)
