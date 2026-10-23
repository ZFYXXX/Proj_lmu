"""Full local AI2D cross-check with upstream prompts/filter/scoring and fixed K128.

Reuses the native-Qwen state, compaction, checkpoint, and cache diagnostics. No
training is performed. All questions are retained; image-level uncertainty and
train-overlap-free scores are reported separately.
"""
import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
from io import BytesIO
import json
from pathlib import Path
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from mllm_comp.analysis.common.compression_training_check import (
    anchor_base, compact_anchor, load_anchor, evaluate_state, implementation_hash,
)
from mllm_comp.analysis.common.condition_ledger import append_json, records
from mllm_comp.analysis.common.evidence_interaction import digest, bootstrap
from mllm_comp.analysis.common.learned_hybrid_runtime import CHECKPOINT, REVISION, load_runtime, load_module, compact_state
from mllm_comp.analysis.common.visual_token_phenomena import media_fingerprint
from mllm_comp.analysis.hooks import HookSession
from mllm_comp.compressors.image.learned_hybrid import VARIANTS
from mllm_comp.core.serialization import atomic_write_json
from scripts.train.qwen25_vl.train_learned_hybrid import prepare_example

METHODS = ["dense", "feature_anchor_mean_k128", "grid_mean_k128", "prumerge_plus_k128", *VARIANTS, "feature_anchor_adapter"]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def prepare(root, hybrid, anchor):
    import pyarrow.parquet as pq
    from PIL import Image
    if (root / "protocol.json").exists():
        raise ValueError("Existing run must be resumed, not overwritten")
    train = json.loads((hybrid / "training_samples.json").read_text())
    train_hashes = {e["image_fingerprint"] for e in train["entries"]}
    files = sorted(Path("/home/YinPeiran/datasets/lmms_eval_datasets/ai2d/data").glob("test-*.parquet"))
    entries, unique = [], {}
    for path in files:
        index = 0
        for batch in pq.ParquetFile(path).iter_batches(batch_size=16):
            for doc in batch.to_pylist():
                data = doc["image"]
                image = Image.open(BytesIO(data["bytes"]) if data.get("bytes") else data["path"]).convert("RGB")
                fingerprint = media_fingerprint([image])
                unique.setdefault(fingerprint, len(entries))
                entries.append({"sample_id": f"ai2d:{len(entries)}", "doc_id": len(entries), "path": str(path), "row": index,
                    "image_fingerprint": fingerprint, "shard": int(digest(fingerprint)[:8],16)%2,
                    "training_image_overlap": fingerprint in train_hashes})
                index += 1
    if len(entries) != 3088:
        raise ValueError("Expected the full 3,088-question test split")
    smoke_fingerprints = set(sorted(unique, key=lambda f: digest("ai2d-smoke:"+f))[:16])
    for e in entries:
        e["smoke"] = e["image_fingerprint"] in smoke_fingerprints and e["doc_id"] == unique[e["image_fingerprint"]]
    modules = {name: hybrid / "models" / name / "checkpoints/step_000320" for name in VARIANTS}
    modules["feature_anchor_adapter"] = anchor / "checkpoints/step_000320"
    root.mkdir(parents=True, exist_ok=True)
    protocol = {"dataset": "AI2D full test", "n_questions": len(entries), "n_unique_content_images": len(unique),
        "training_overlap_questions": sum(e["training_image_overlap"] for e in entries),
        "training_overlap_images": len(set(unique)&train_hashes), "checkpoint": CHECKPOINT, "revision": REVISION,
        "dtype": "bfloat16", "attention": "eager", "seed": 42, "max_new_tokens": 16,
        "processor": {"use_fast": False, "min_pixels": 200704, "max_pixels": 1605632},
        "methods": METHODS, "budget": 128, "training": False,
        "metric": "Upstream AI2D MCQ prompt, MultiChoiceRegexFilter and exact_match_hf_evaluate",
        "primary_quality_subset": "Exclude exact image-content overlap with compressor training, for ALL methods equally",
        "uncertainty": "Paired bootstrap resamples independent image groups, not questions independently",
        "source_hybrid": str(hybrid), "source_anchor": str(anchor),
        "module_paths": {k: str(v) for k,v in modules.items()},
        "module_sha256": {k: sha(v / "projector.safetensors") for k,v in modules.items()},
        "implementation_hash": implementation_hash(), "script_sha256": sha(__file__),
        "data_files": {str(p): sha(p) for p in files}, "training_manifest_sha256": sha(hybrid / "training_samples.json"),
        "scope": "Cross-dataset check on fixed existing checkpoints; not a faithful reproduction of source papers"}
    atomic_write_json(root / "protocol.json", protocol)
    atomic_write_json(root / "samples.json", {"entries": entries, "sampling": "Complete test split; no answer/performance filtering"})
    print(json.dumps({k: protocol[k] for k in ["n_questions", "n_unique_content_images", "training_overlap_questions", "training_overlap_images"]}), flush=True)


def upstream_score(prediction, doc):
    from lmms_eval.tasks.ai2d.utils import MultiChoiceRegexFilter
    from lmms_eval.api.metrics import exact_match_hf_evaluate
    filter = MultiChoiceRegexFilter(group_select=0, ignore_case=True, ignore_punctuation=True, regex_pattern="([A-Z])\\.")
    filtered, = filter.apply([[prediction]], [doc])
    target = chr(65+int(doc["answer"]))
    value = exact_match_hf_evaluate(references=[target], predictions=[filtered], ignore_case=True, ignore_punctuation=True)
    return float(value["exact_match"]), filtered


def vision_metrics(bundle, captured, state):
    from mllm_comp.backbones.qwen25_vl.provider_bridge import _merged_vision_metrics, _apply_rotary_pos_emb_vision_4573
    vision = bundle["vision"]
    block = vision.blocks[-1].attn
    raw = block.qkv(captured["hidden"]).reshape(-1, 3, block.num_heads, block.head_dim)
    q, k = _apply_rotary_pos_emb_vision_4573(raw[:, 0], raw[:, 1], *captured["positions"])
    window, _ = vision.get_window_index(state["prompt_inputs"]["image_grid_thw"])
    attn, keys = _merged_vision_metrics(q, k, patch_counts=[len(raw)], feature_counts=[len(state["features"])],
        window_index=window, spatial_merge_unit=vision.spatial_merge_size**2, scaling=float(block.scaling))
    return attn[0], keys[0]


def worker(root, stage, shard, device):
    import torch
    import pyarrow.parquet as pq
    from lmms_eval.tasks.ai2d.utils import ai2d_doc_to_text
    from mllm_comp.compressors.image.fixed_visual_merge import fixed_visual_merge, FixedVisualMergeSpec
    from mllm_comp.compressors.image.prumerge import prumerge_image_tokens, PruMergeSpec
    torch.manual_seed(42)
    protocol = json.loads((root / "protocol.json").read_text())
    if protocol["implementation_hash"] != implementation_hash() or protocol["script_sha256"] != sha(__file__):
        raise ValueError("Frozen evaluation implementation changed")
    entries = [e for e in json.loads((root / "samples.json").read_text())["entries"]
               if e["shard"] == shard and (stage == "full" or e["smoke"])]
    docs = {}
    wanted = defaultdict(dict)
    for entry in entries:
        wanted[entry["path"]][entry["row"]] = entry
    for path, rows in wanted.items():
        index = 0
        if sha(path) != protocol["data_files"][path]:
            raise ValueError("Dataset bytes changed")
        for batch in pq.ParquetFile(path).iter_batches(batch_size=16):
            for doc in batch.to_pylist():
                if index in rows:
                    docs[rows[index]["sample_id"]] = doc
                index += 1
    bundle = load_runtime(device)
    modules = {}
    for name, path in protocol["module_paths"].items():
        if sha(Path(path) / "projector.safetensors") != protocol["module_sha256"][name]:
            raise ValueError("Saved checkpoint bytes changed")
        modules[name] = (load_anchor if name == "feature_anchor_adapter" else load_module)(path, device).eval()
    config_hash = digest(json.dumps(protocol, sort_keys=True)+sha(root / "samples.json")+stage)
    status_path = root / f"{stage}_{shard}_manifest.json"
    status = {"status": "running", "expected_questions": len(entries), "completed_questions": 0,
              "stage": stage, "shard": shard, "device": device, "config_hash": config_hash,
              "started_utc": datetime.now(timezone.utc).isoformat()}
    atomic_write_json(status_path, status)
    try:
        for entry in entries:
            folder = root / "samples" / stage / entry["sample_id"].replace(":", "_")
            folder.mkdir(parents=True, exist_ok=True)
            ledger = folder / "conditions.jsonl"
            recorded = records(ledger, config_hash)
            if len(recorded) != len(METHODS):
                doc = docs[entry["sample_id"]]
                question = ai2d_doc_to_text(doc, {"pre_prompt": "", "post_prompt": "\nAnswer with the option's letter from the given choices directly.", "prompt_format": "mcq"})
                target = chr(65+int(doc["answer"]))
                captured = {}
                def capture(module, args, kwargs):
                    captured["hidden"] = (args[0] if args else kwargs["hidden_states"]).detach()
                    captured["positions"] = kwargs["position_embeddings"]
                with torch.inference_mode(), HookSession() as hooks:
                    hooks.register_forward_pre_hook(bundle["vision"].blocks[-1].attn, capture, with_kwargs=True)
                    state = prepare_example(bundle, {**entry, "image": doc["image"], "question": question, "target": target})
                with torch.inference_mode():
                    attention, keys = vision_metrics(bundle, captured, state)
                    base = anchor_base(state)
                    grid = fixed_visual_merge(state["features"], grid_height=state["height"], grid_width=state["width"],
                        spec=FixedVisualMergeSpec(partition="grid", target_count=128))
                    pru = prumerge_image_tokens(state["features"], attention, keys,
                        spec=PruMergeSpec(variant="plus", adaptive=False, target_count=128, cluster_size=32))
                    del captured, attention, keys
                    for name in METHODS:
                        key = (entry["sample_id"], name)
                        if key in recorded:
                            continue
                        started = time.perf_counter()
                        output, compact, trace = None, None, None
                        if name in {"feature_anchor_mean_k128", "feature_anchor_adapter"}:
                            with torch.autocast("cuda", dtype=torch.bfloat16):
                                output = base if name == "feature_anchor_mean_k128" else modules[name](base)
                            compact = compact_anchor(state, output)
                        elif name == "grid_mean_k128":
                            output = {"indices": grid.indices.tolist(), "tokens": grid.tokens, "members": grid.metadata["cluster_members"],
                                      "position_policy": "rounded_native_mrope_cluster_centroid"}
                            compact = compact_anchor(state, output)
                        elif name == "prumerge_plus_k128":
                            output = {"indices": pru.indices.tolist(), "tokens": pru.tokens,
                                      "members": pru.metadata.get("cluster_members", []), "position_policy": "native_source_representative"}
                            compact = compact_state(state, output)
                            trace = dict(pru.metadata)
                        elif name != "dense":
                            with torch.autocast("cuda", dtype=torch.bfloat16):
                                output = modules[name](state["features"], height=state["height"], width=state["width"],
                                    random_seed=int(digest(entry["image_fingerprint"]+":42")[:8],16))
                            compact = compact_state(state, output)
                        result = evaluate_state(bundle, state, compact, max_new=16, native_check=stage == "smoke" and name == "dense")
                        score, filtered = upstream_score(result["prediction"], doc)
                        if output is not None and (len(output["indices"]) != 128 or result["retained_visual_tokens"] != 128):
                            raise ValueError("Unfair visual token budget")
                        row = {**result, "sample_id": entry["sample_id"], "variant": name, "config_hash": config_hash,
                            "image_fingerprint": entry["image_fingerprint"], "training_image_overlap": entry["training_image_overlap"],
                            "checkpoint": CHECKPOINT, "revision": REVISION, "status": "completed", "question": question,
                            "target": target, "filtered_prediction": filtered, "primary_task_score": score,
                            "strict_letter_score": float(result["prediction"].strip().upper() == target),
                            "compression_trace": trace if trace is not None else (None if output is None else
                                {k: output[k] for k in ["indices", "members", "position_policy"]}),
                            "elapsed_s": time.perf_counter()-started}
                        append_json(ledger, row)
                        recorded[key] = row
                del state, base, grid, pru, output, compact
            if {k[1] for k in recorded} != set(METHODS) or len(recorded) != len(METHODS):
                raise ValueError("Incomplete or duplicate sample conditions")
            atomic_write_json(folder / "summary.json", {**entry, "config_hash": config_hash, "condition_count": len(recorded), "status": "completed"})
            status["completed_questions"] += 1
            atomic_write_json(status_path, status)
            print(json.dumps({"stage": stage, "shard": shard, "completed": status["completed_questions"], "expected": len(entries)}), flush=True)
        status["status"] = "complete"
    except BaseException as exc:
        status.update({"status": "failed", "error": str(exc), "traceback": traceback.format_exc()})
        raise
    finally:
        atomic_write_json(status_path, status)


def report(root, stage):
    entries = [e for e in json.loads((root / "samples.json").read_text())["entries"] if stage == "full" or e["smoke"]]
    groups = {"all": defaultdict(list), "train_overlap_free": defaultdict(list)}
    for entry in entries:
        folder = root / "samples" / stage / entry["sample_id"].replace(":", "_")
        if not (folder / "summary.json").exists():
            raise ValueError("Cannot report incomplete evaluation")
        rows = {r["variant"]: r for r in map(json.loads, (folder / "conditions.jsonl").read_text().splitlines())}
        if set(rows) != set(METHODS):
            raise ValueError("Missing evaluation methods")
        if stage == "smoke" and not rows["dense"].get("native_generation_identity_passed"):
            raise ValueError("Native/dense identity gate missing")
        groups["all"][entry["image_fingerprint"]].append(rows)
        if not entry["training_image_overlap"]:
            groups["train_overlap_free"][entry["image_fingerprint"]].append(rows)
    result = {"status": "complete", "stage": stage, "n_questions": len(entries), "tables": {},
              "scope": "Fixed-checkpoint cross-dataset evaluation; image-cluster bootstrap; one training seed"}
    for subset, images in groups.items():
        n = sum(len(q) for q in images.values())
        table = {}
        for method in METHODS:
            rows = [q[method] for questions in images.values() for q in questions]
            table[method] = {"n_questions": n, "n_images": len(images),
                "accuracy": sum(r["primary_task_score"] for r in rows)/n,
                "strict_letter_accuracy": sum(r["strict_letter_score"] for r in rows)/n,
                "mean_visual_tokens": sum(r["retained_visual_tokens"] for r in rows)/n,
                "mean_nll": sum(r["answer_nll"] for r in rows)/n,
                "generation_reached_limit_without_eos": sum(not r["stopped_on_eos"] for r in rows)}
        # Image-weighted effect is distinguished from question-weighted accuracy.
        for method in METHODS:
            table[method]["image_mean_delta_vs_feature_anchor"] = bootstrap([
                sum(q[method]["primary_task_score"]-q["feature_anchor_mean_k128"]["primary_task_score"] for q in questions)/len(questions)
                for questions in images.values()])
        result["tables"][subset] = table
    atomic_write_json(root / f"{stage}_report.json", result)
    lines = [f"# AI2D {stage}: fixed K128 cross-check", "", f"Questions: {len(entries)}", "",
        "| Subset | Method | Accuracy | Strict letter | Mean visual tokens |", "|---|---|---:|---:|---:|"]
    for subset, table in result["tables"].items():
        for method, values in table.items():
            lines.append(f"| {subset} | {method} | {100*values['accuracy']:.2f}% | {100*values['strict_letter_accuracy']:.2f}% | {values['mean_visual_tokens']:.1f} |")
    lines += ["", "All questions are included. Primary clean comparison excludes train-image overlap for every method equally. Bootstrap groups questions by image; image-weighted effects and question-weighted accuracies are separate estimands. Timing includes diagnostic scoring and is not a deployment speed benchmark."]
    (root / f"{stage}_report.md").write_text("\n".join(lines)+"\n", encoding="utf-8")
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--hybrid", type=Path)
    p.add_argument("--anchor", type=Path)
    p.add_argument("--prepare", action="store_true")
    p.add_argument("--report", action="store_true")
    p.add_argument("--stage", choices=["smoke", "full"], default="smoke")
    p.add_argument("--shard", type=int, choices=[0,1], default=0)
    p.add_argument("--device", default="cuda:0")
    args = p.parse_args()
    if args.prepare:
        prepare(args.root.resolve(), args.hybrid.resolve(), args.anchor.resolve())
    elif args.report:
        print(json.dumps(report(args.root.resolve(), args.stage)))
    else:
        worker(args.root.resolve(), args.stage, args.shard, args.device)
