"""Train-only generation audit, CE-only capacity check, and strong-anchor control.

No old architecture/checkpoint is changed. Small repeated-data checks are explicitly
optimization diagnostics, not independent evidence of downstream generalization.
"""
import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from scripts.train.qwen25_vl.train_learned_hybrid import load_examples, prepare_example
from mllm_comp.analysis.common.compression_training_check import (
    AnchorAdapterSpec, create_anchor_adapter, anchor_base, compact_anchor, implementation_hash,
    save_anchor, load_anchor, evaluate_state,
)
from mllm_comp.analysis.common.learned_hybrid_runtime import (
    CHECKPOINT, REVISION, answer_logits, compact_state, load_runtime, load_module, save_module,
)
from mllm_comp.analysis.common.evidence_interaction import digest
from mllm_comp.compressors.image.learned_hybrid import VARIANTS, LearnedHybridSpec, create_learned_hybrid
from mllm_comp.core.serialization import atomic_write_json, to_jsonable


def selected_examples(source, text_count, science_count):
    all_examples = load_examples(source / "training_samples.json")
    selected = []
    for name, count in [("textvqa_train", text_count), ("scienceqa_train", science_count)]:
        rows = sorted((e for e in all_examples if e["source"] == name),
                      key=lambda e: digest("training_check_v1:" + e["image_fingerprint"]))[:count]
        if len(rows) != count:
            raise ValueError(f"Insufficient train-only examples: {name}")
        selected.extend(rows)
    return sorted(selected, key=lambda e: digest("check_order:" + e["image_fingerprint"]))


def add_answers(examples):
    """Recover all human answers for native TextVQA scoring, without changing sampling."""
    import pyarrow.parquet as pq
    from lmms_eval.tasks.textvqa.utils import textvqa_process_results
    grouped = defaultdict(dict)
    for e in examples:
        grouped[e["path"]][e["row"]] = e
    for path, wanted in grouped.items():
        index = 0
        for batch in pq.ParquetFile(path).iter_batches(batch_size=32):
            for doc in batch.to_pylist():
                if index in wanted:
                    e = wanted[index]
                    e["scoring_doc"] = {"answers": list(doc["answers"]), "question_id": doc["question_id"]} if e["source"] == "textvqa_train" else None
                index += 1
            if index > max(wanted):
                break
    for e in examples:
        if "scoring_doc" not in e:
            raise ValueError("Missing scoring document")
    return textvqa_process_results


def append_json(path, row):
    import os
    encoded = json.dumps(to_jsonable(row), ensure_ascii=False, allow_nan=False)
    with path.open("a") as f:
        f.write(encoded + "\n")
        f.flush()
        os.fsync(f.fileno())


def records(path, config_hash):
    result = {}
    if not path.exists():
        return result
    with path.open("rb+") as stream:
        while True:
            offset = stream.tell()
            line = stream.readline()
            if not line:
                break
            if not line.endswith(b"\n"):
                stream.truncate(offset)
                break
            row = json.loads(line)
            key = (row["sample_id"], row["variant"])
            if row["config_hash"] != config_hash or key in result:
                raise ValueError("Mixed or duplicate train-check records")
            result[key] = row
    return result


def evaluate(bundle, examples, modules, path, config_hash, native_scorer, *, smoke=False):
    import torch
    rows = records(path, config_hash)
    for number, e in enumerate(examples):
        if all((e["sample_id"], name) in rows for name in modules):
            continue
        state = prepare_example(bundle, e)
        base = anchor_base(state)
        for name, module in modules.items():
            key = (e["sample_id"], name)
            if key in rows:
                continue
            output, compact = None, None
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                if name != "dense":
                    if name.startswith("feature_anchor"):
                        output = base if module is None else module(base)
                        compact = compact_anchor(state, output)
                    else:
                        output = module(state["features"], height=state["height"], width=state["width"],
                            random_seed=int(digest(e["image_fingerprint"] + ":42")[:8], 16))
                        compact = compact_state(state, output)
                result = evaluate_state(bundle, state, compact, max_new=32 if e["source"] == "textvqa_train" else 16,
                                        native_check=smoke and name == "dense")
            prediction = result["prediction"]
            score = native_scorer(e["scoring_doc"], [prediction])["exact_match"] if e["source"] == "textvqa_train" else float(prediction.strip().upper() == e["target"])
            row = {"sample_id": e["sample_id"], "image_fingerprint": e["image_fingerprint"], "source": e["source"],
                   "variant": name, "config_hash": config_hash, "status": "completed", "target": e["target"],
                   "primary_task_score": score, "target_exact_match": float(prediction.casefold() == e["target"].casefold()),
                   "scope": "training_set_diagnostic_not_generalization", "checkpoint": CHECKPOINT, "revision": REVISION,
                   "compression_trace": None if output is None else {k: output[k] for k in ["indices", "members", "position_policy"]}, **result}
            append_json(path, row)
            rows[key] = row
        print(json.dumps({"evaluation": str(path), "completed_images": number+1, "expected_images": len(examples)}), flush=True)
    expected = {(e["sample_id"], name) for e in examples for name in modules}
    if set(rows) != expected:
        raise ValueError("Incomplete training diagnostic")
    aggregate = {}
    for source in sorted({e["source"] for e in examples}):
        aggregate[source] = {}
        for name in modules:
            subset = [r for r in rows.values() if r["source"] == source and r["variant"] == name]
            aggregate[source][name] = {"n_images": len(subset), **{field: sum(r[field] for r in subset)/len(subset)
                for field in ["primary_task_score", "target_exact_match", "answer_nll"]}}
    atomic_write_json(path.with_suffix(".summary.json"), {"status": "complete", "config_hash": config_hash,
        "scope": "training_set_diagnostic_not_generalization", "tables": aggregate})
    return aggregate


def run(args):
    import torch
    import torch.nn.functional as F
    folder = args.output.resolve()
    folder.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    examples = selected_examples(args.source, 192, 64) if args.mode == "audit" else (
        selected_examples(args.source, 48, 16) if args.mode == "overfit" else load_examples(args.source / "training_samples.json"))
    if args.smoke:
        # Both sources, without performance-based sample selection.
        examples = [next(e for e in examples if e["source"] == s) for s in ["textvqa_train", "scienceqa_train"]]
    native_scorer = add_answers(examples if args.mode != "anchor" else selected_examples(args.source, 48, 16))
    diagnostic = examples if args.mode != "anchor" else selected_examples(args.source, 48, 16)
    if args.mode == "anchor":
        add_answers(diagnostic)
        if args.smoke:
            diagnostic = [next(e for e in diagnostic if e["source"] == s) for s in ["textvqa_train", "scienceqa_train"]]
    config = {"mode": args.mode, "steps": args.steps, "accumulation": args.accumulation, "smoke": args.smoke,
              "implementation_hash": implementation_hash(), "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "source_manifest_sha256": hashlib.sha256((args.source / "training_samples.json").read_bytes()).hexdigest(),
              "source": str(args.source.resolve()), "train_ids": [e["sample_id"] for e in examples],
              "diagnostic_ids": [e["sample_id"] for e in diagnostic], "seed": 42,
              "loss": "CE only" if args.mode == "overfit" else "CE + 0.5 teacher KL", "learning_rate": .0002}
    config["source_checkpoint_sha256"] = {v: hashlib.sha256((args.source / "models" / v /
        "checkpoints/step_000320/projector.safetensors").read_bytes()).hexdigest() for v in VARIANTS}
    config_hash = digest(json.dumps(config, sort_keys=True))
    manifest = {**config, "config_hash": config_hash, "status": "running", "checkpoint": CHECKPOINT, "revision": REVISION,
                "started_utc": datetime.now(timezone.utc).isoformat(), "steps_completed": 0,
                "interpretation": "Train-only optimization diagnostics; anchor generalization evaluated separately on reused held-out cohort."}
    atomic_write_json(folder / "manifest.json", manifest)
    try:
        bundle = load_runtime(args.device)
        if args.mode == "audit":
            modules = {"dense": None, "feature_anchor_untrained": None}
            for variant in VARIANTS:
                torch.manual_seed(42)
                modules[variant + "_step0"] = create_learned_hybrid(LearnedHybridSpec(bundle["llm"].config.hidden_size, variant)).to(args.device).eval()
                path = args.source / "models" / variant / "checkpoints/step_000320"
                modules[variant + "_step320"] = load_module(path, args.device).eval()
            evaluate(bundle, examples, modules, folder / "predictions.jsonl", config_hash, native_scorer, smoke=args.smoke)
        else:
            anchor = args.mode == "anchor"
            if anchor:
                module = create_anchor_adapter(AnchorAdapterSpec(bundle["llm"].config.hidden_size)).to(args.device)
                save_fn, load_fn, arm = save_anchor, load_anchor, "feature_anchor_adapter"
            else:
                module = load_module(args.source / "models/summary_residual/checkpoints/step_000320", args.device)
                save_fn, load_fn, arm = save_module, load_module, "summary_residual_ce_overfit"
            optimizer = torch.optim.AdamW(module.parameters(), lr=.0002, weight_decay=.01)
            warmup = min(16, max(1, args.steps // 10))
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda s: (s+1)/warmup if s < warmup else
                .5*(1+math.cos(math.pi*min(1., (s-warmup)/max(1, args.steps-warmup)))))
            step, micro = 0, 0
            if args.resume and (folder / "latest.json").exists():
                pointer = json.loads((folder / "latest.json").read_text())
                checkpoint = folder / pointer["checkpoint"]
                saved = torch.load(checkpoint / "training_state.pt", map_location=args.device, weights_only=False)
                if saved["config_hash"] != config_hash:
                    raise ValueError("Resume config mismatch")
                module = load_fn(checkpoint, args.device)
                optimizer = torch.optim.AdamW(module.parameters(), lr=.0002, weight_decay=.01)
                scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda s: (s+1)/warmup if s < warmup else
                    .5*(1+math.cos(math.pi*min(1., (s-warmup)/max(1, args.steps-warmup)))))
                optimizer.load_state_dict(saved["optimizer"])
                scheduler.load_state_dict(saved["scheduler"])
                step, micro = saved["step"], saved["micro"]
                torch.set_rng_state(saved["torch_rng"].cpu())
                torch.cuda.set_rng_state_all([r.cpu() for r in saved["cuda_rng"]])
            if step == 0:
                evaluate(bundle, diagnostic, {"dense": None, "feature_anchor_untrained": None, arm: module.eval()},
                         folder / "before.jsonl", config_hash, native_scorer, smoke=args.smoke)
                if anchor:
                    before = records(folder / "before.jsonl", config_hash)
                    for e in diagnostic:
                        original, adapted = (before[(e["sample_id"], name)] for name in ["feature_anchor_untrained", arm])
                        if any(original[k] != adapted[k] for k in ["prediction", "generated_token_ids", "answer_nll", "first_appended_token_position"]):
                            raise ValueError("Zero-init adapter changed feature-anchor forward/generation")
            llm = bundle["llm"]
            llm.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
            optimizer.zero_grad(set_to_none=True)
            accum = defaultdict(float)
            started = time.perf_counter()
            while step < args.steps:
                e = examples[micro % len(examples)]
                state = prepare_example(bundle, e)
                base = anchor_base(state) if anchor else None
                teacher = None
                llm.eval()
                if anchor:
                    with torch.no_grad():
                        teacher, _ = answer_logits(bundle, state["hidden"], state["positions"], state["teacher_inputs"]["labels"])
                llm.train()
                module.train()
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    output = module(base) if anchor else module(state["features"], height=state["height"], width=state["width"],
                        random_seed=int(digest(e["image_fingerprint"] + ":42")[:8], 16))
                    compact = compact_anchor(state, output) if anchor else compact_state(state, output)
                    logits, targets = answer_logits(bundle, compact["hidden"], compact["positions"], compact["labels"])
                    ce = F.cross_entropy(logits, targets)
                    kl = F.kl_div(F.log_softmax(logits, -1), F.softmax(teacher, -1), reduction="batchmean").clamp_min(0) if anchor else ce.new_zeros(())
                    loss = ce + .5*kl
                if not bool(torch.isfinite(loss)):
                    raise ValueError("Nonfinite training loss")
                (loss / args.accumulation).backward()
                for name, value in [("loss", loss), ("answer_ce", ce), ("teacher_kl", kl)]:
                    accum[name] += float(value.detach()) / args.accumulation
                micro += 1
                if micro % args.accumulation:
                    continue
                if any(p.grad is not None for p in bundle["model"].parameters()):
                    raise ValueError("Backbone is not frozen")
                norm = torch.nn.utils.clip_grad_norm_(module.parameters(), 1.)
                if not bool(torch.isfinite(norm)) or float(norm) == 0:
                    raise ValueError("Invalid compressor gradients")
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                step += 1
                row = {"step": step, "micro": micro, **accum, "gradient_norm": float(norm),
                       "elapsed_s": time.perf_counter()-started, "learning_rate": scheduler.get_last_lr()[0]}
                append_json(folder / "training.jsonl", row)
                print(json.dumps(row), flush=True)
                accum.clear()
                if step == 1 or step % 16 == 0 or step == args.steps:
                    path = folder / "checkpoints" / f"step_{step:06d}"
                    save_fn(module, path)
                    torch.save({"config_hash": config_hash, "step": step, "micro": micro,
                        "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
                        "torch_rng": torch.get_rng_state(), "cuda_rng": torch.cuda.get_rng_state_all()}, path / "training_state.pt")
                    restored = load_fn(path, args.device)
                    if not all(torch.equal(a, b) for a, b in zip(module.state_dict().values(), restored.state_dict().values())):
                        raise ValueError("Saved/reloaded checkpoint mismatch")
                    del restored
                    atomic_write_json(folder / "latest.json", {"checkpoint": str(path.relative_to(folder)), "step": step,
                        "sha256": hashlib.sha256((path / "projector.safetensors").read_bytes()).hexdigest()})
                    manifest.update({"steps_completed": step, "checkpoint_roundtrip_passed": True})
                    atomic_write_json(folder / "manifest.json", manifest)
                del state, compact, output, logits, loss, teacher
            evaluate(bundle, diagnostic, {"dense": None, "feature_anchor_untrained": None, arm: module.eval()},
                     folder / "after.jsonl", config_hash, native_scorer, smoke=args.smoke)
            manifest["steps_completed"] = step
        manifest.update({"status": "complete", "completed_utc": datetime.now(timezone.utc).isoformat(),
                         "peak_allocated_bytes": torch.cuda.max_memory_allocated(args.device)})
    except BaseException as exc:
        manifest.update({"status": "failed", "error": str(exc), "traceback": traceback.format_exc()})
        raise
    finally:
        atomic_write_json(folder / "manifest.json", manifest)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mode", choices=["audit", "anchor", "overfit"], required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--steps", type=int, default=320)
    parser.add_argument("--accumulation", type=int, default=8)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    run(parser.parse_args())
