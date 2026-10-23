"""Frozen-Qwen CE/KL/reconstruction training with exact resume checks."""

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
from io import BytesIO
import json
import math
from pathlib import Path
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from mllm_comp.analysis.common.evidence_interaction import digest
from mllm_comp.analysis.common.learned_hybrid_runtime import (
    CHECKPOINT, REVISION, answer_logits, code_hash, compact_state, load_module, load_runtime, prepare_state, save_module,
)
from mllm_comp.analysis.common.visual_token_phenomena import media_fingerprint
from mllm_comp.analysis.probes.visual_evidence_interaction import cached_decode
from mllm_comp.compressors.image.learned_hybrid import LearnedHybridSpec, VARIANTS, create_learned_hybrid
from mllm_comp.core.serialization import atomic_write_json


def load_examples(path):
    import pyarrow.parquet as pq
    manifest = json.loads(path.read_text())
    grouped = defaultdict(dict)
    for entry in manifest["entries"]:
        grouped[entry["path"]][entry["row"]] = entry
    examples = {}
    for filename, rows in grouped.items():
        index = 0
        for batch in pq.ParquetFile(filename).iter_batches(batch_size=32):
            for doc in batch.to_pylist():
                if index in rows:
                    entry = rows[index]
                    if entry["source"] == "textvqa_train":
                        answers = [str(x).strip() for x in doc["answers"] if str(x).strip()]
                        if not answers:
                            raise ValueError("Selected train example has no answer; do not silently skip.")
                        target = Counter(answers).most_common(1)[0][0]
                        question = doc["question"].capitalize() + "\nAnswer the question using a single word or phrase."
                    else:
                        choices = "\n".join(f"{chr(65+i)}. {choice}" for i, choice in enumerate(doc["choices"]))
                        hint = str(doc.get("hint") or "").strip()
                        question = (f"Context: {hint}\n" if hint else "") + f"Question: {doc['question']}\nChoices:\n{choices}\nAnswer with the option's letter from the given choices directly."
                        target = chr(65 + int(doc["answer"]))
                    examples[entry["sample_id"]] = {**entry, "image": doc["image"], "question": question, "target": target}
                index += 1
            if index > max(rows):
                break
    if len(examples) != len(manifest["entries"]):
        raise ValueError("Missing selected training examples.")
    return [examples[entry["sample_id"]] for entry in manifest["entries"]]


def prepare_example(bundle, example):
    from PIL import Image
    from mllm_comp.backbones.qwen25_vl import DEFAULT_SYSTEM_PROMPT, Qwen25VLSample
    data = example["image"]
    image = Image.open(BytesIO(data["bytes"]) if data.get("bytes") else data["path"]).convert("RGB")
    if media_fingerprint([image]) != example["image_fingerprint"]:
        raise ValueError("Training image fingerprint changed.")
    sample = Qwen25VLSample(example["sample_id"], [
        {"role": "system", "content": [{"type": "text", "text": DEFAULT_SYSTEM_PROMPT}]},
        {"role": "user", "content": [{"type": "image", "image": image, "min_pixels": 200704, "max_pixels": 1605632},
                                      {"type": "text", "text": example["question"]}]}])
    adapter, processor = bundle["facade"].processor_adapter, bundle["processor"]
    prompt = adapter.prepare_inputs(processor, sample, phase="train")
    teacher = adapter.prepare_teacher_forced_inputs(processor, sample, target_text=example["target"], prompt_batch=prompt, phase="train")
    return prepare_state(bundle, prompt.model_inputs, teacher.model_inputs)


def main(args):
    import torch
    import torch.nn.functional as F
    from transformers import GenerationConfig
    root, folder = args.root.resolve(), args.output.resolve()
    folder.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    examples = load_examples(root / "training_samples.json")
    bundle = load_runtime(args.device)
    llm = bundle["llm"]
    llm.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    spec = LearnedHybridSpec(hidden_size=llm.config.hidden_size, variant=args.variant)
    torch.manual_seed(42)
    module = create_learned_hybrid(spec).to(args.device)
    optimizer = torch.optim.AdamW(module.parameters(), lr=.0002, weight_decay=.01)
    warmup = min(16, max(1, args.steps // 10))

    def schedule(step):
        if step < warmup:
            return (step + 1) / warmup
        return .5 * (1 + math.cos(math.pi * min(1., (step - warmup) / max(1, args.steps - warmup))))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)
    config = {"variant": args.variant, "steps": args.steps, "accumulation": args.accumulation,
              "training_manifest_sha256": hashlib.sha256((root / "training_samples.json").read_bytes()).hexdigest(),
              "implementation_hash": code_hash(), "seed": 42}
    config_hash = digest(json.dumps(config, sort_keys=True))
    step, micro = 0, 0
    restored = False
    previous_manifest = json.loads((folder / "manifest.json").read_text()) if (folder / "manifest.json").exists() else {}
    if args.resume and (folder / "latest.json").exists():
        pointer = json.loads((folder / "latest.json").read_text())
        checkpoint = folder / pointer["checkpoint"]
        state = torch.load(checkpoint / "training_state.pt", map_location=args.device, weights_only=False)
        if state["config_hash"] != config_hash:
            raise ValueError("Resume configuration/data/implementation mismatch.")
        module = load_module(checkpoint, args.device)
        optimizer = torch.optim.AdamW(module.parameters(), lr=.0002, weight_decay=.01)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        step, micro = state["step"], state["micro"]
        torch.set_rng_state(state["torch_rng"].cpu())
        torch.cuda.set_rng_state_all([s.cpu() for s in state["cuda_rng"]])
        restored = state["roundtrip_checked"]
    manifest = {**config, "config_hash": config_hash, "status": "running", "checkpoint": CHECKPOINT, "revision": REVISION,
                "started_utc": datetime.now(timezone.utc).isoformat(), "trainable_parameters": sum(p.numel() for p in module.parameters()),
                "train_images": len(examples), "frozen_parameters": sum(p.numel() for p in bundle["model"].parameters()),
                "parameter_dtype": "float32", "model_dtype": "bfloat16", "steps_completed": step,
                "train_eval_image_overlap": 0, "loss_weights": {"ce": 1., "teacher_kl": .5, "reconstruction": .1 if args.variant != "pool_adapter" else 0.}}
    if args.resume and previous_manifest.get("config_hash") == config_hash:
        manifest["native_generation_identity_passed"] = previous_manifest.get("native_generation_identity_passed", False)
    atomic_write_json(folder / "manifest.json", manifest, ensure_ascii=False)
    optimizer.zero_grad(set_to_none=True)
    elapsed_start = time.perf_counter()
    accum = defaultdict(float)
    try:
        while step < args.steps:
            example = examples[micro % len(examples)]
            state = prepare_example(bundle, example)
            random_seed = int(digest(example["image_fingerprint"] + ":42")[:8], 16)
            llm.eval()
            with torch.no_grad():
                teacher_logits, targets = answer_logits(bundle, state["hidden"], state["positions"], state["teacher_inputs"]["labels"])
            if args.smoke and micro == 0:
                with torch.inference_mode():
                    positions = state["positions"][:, :, :state["prefix"]]
                    hidden = state["hidden"][:, :state["prefix"]]
                    eos = bundle["model"].generation_config.eos_token_id
                    eos = [eos] if isinstance(eos, int) else list(eos)
                    diagnostic = cached_decode(llm=llm, head=bundle["model"].lm_head, embedding=bundle["model"].get_input_embeddings(),
                        hidden=hidden, positions=positions, eos_ids=eos, max_new=16)
                    native = bundle["model"].generate(**state["prompt_inputs"], generation_config=GenerationConfig(max_new_tokens=16,
                        do_sample=False, use_cache=True, eos_token_id=eos, pad_token_id=bundle["processor"].tokenizer.pad_token_id))[0, state["prefix"]:].tolist()
                    if native != diagnostic["generated_token_ids"]:
                        raise ValueError("Native/dense training-runtime generation mismatch.")
                    manifest["native_generation_identity_passed"] = True
            module.train()
            llm.train()
            with torch.autocast("cuda", dtype=torch.bfloat16):
                output = module(state["features"], height=state["height"], width=state["width"], random_seed=random_seed)
                compact = compact_state(state, output)
                student_logits, compact_targets = answer_logits(bundle, compact["hidden"], compact["positions"], compact["labels"])
                if not torch.equal(targets, compact_targets):
                    raise ValueError("Teacher/student answer mismatch.")
                ce = F.cross_entropy(student_logits, targets)
                log_teacher = F.log_softmax(teacher_logits, -1)
                kl = F.kl_div(F.log_softmax(student_logits, -1), log_teacher.exp(), reduction="batchmean").clamp_min(0)
                rec = output["reconstruction_loss"]
                loss = ce + .5 * kl + .1 * rec
            if not bool(torch.isfinite(loss)):
                raise ValueError("Nonfinite training loss.")
            (loss / args.accumulation).backward()
            for name, value in [("loss", loss), ("answer_ce", ce), ("teacher_kl", kl), ("reconstruction_loss", rec)]:
                accum[name] += float(value.detach()) / args.accumulation
            micro += 1
            if micro % args.accumulation:
                continue
            if any(p.grad is not None for p in bundle["model"].parameters()):
                raise ValueError("Frozen backbone unexpectedly accumulated gradients.")
            norm = torch.nn.utils.clip_grad_norm_(module.parameters(), 1.)
            if not bool(torch.isfinite(norm)) or float(norm) == 0:
                raise ValueError("Nonfinite or zero compressor gradients.")
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            step += 1
            record = {"step": step, "micro": micro, **accum, "gradient_norm": float(norm),
                      "learning_rate": scheduler.get_last_lr()[0], "elapsed_s": time.perf_counter() - elapsed_start,
                      "sample_id": example["sample_id"], "visual_tokens": len(state["features"]), "output_tokens": len(output["indices"]),
                      "score_mean": output["score_mean"], "score_std": output["score_std"]}
            with (folder / "training.jsonl").open("a") as log:
                log.write(json.dumps(record, allow_nan=False) + "\n")
                log.flush()
            print(json.dumps(record), flush=True)
            accum.clear()
            if step == 1 or step % 16 == 0 or step == args.steps:
                path = folder / "checkpoints" / f"step_{step:06d}"
                save_module(module, path)
                torch.save({"config_hash": config_hash, "step": step, "micro": micro,
                            "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
                            "torch_rng": torch.get_rng_state(), "cuda_rng": torch.cuda.get_rng_state_all(),
                            "roundtrip_checked": restored}, path / "training_state.pt")
                if not restored:
                    restored_module = load_module(path, args.device).eval()
                    module.eval()
                    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                        a = module(state["features"], height=state["height"], width=state["width"], random_seed=random_seed)
                        b = restored_module(state["features"], height=state["height"], width=state["width"], random_seed=random_seed)
                    if a["indices"] != b["indices"] or not torch.equal(a["tokens"], b["tokens"]):
                        raise ValueError("Saved/restored compressor output mismatch.")
                    saved = torch.load(path / "training_state.pt", map_location=args.device, weights_only=False)
                    module = restored_module
                    optimizer = torch.optim.AdamW(module.parameters(), lr=.0002, weight_decay=.01)
                    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)
                    optimizer.load_state_dict(saved["optimizer"])
                    scheduler.load_state_dict(saved["scheduler"])
                    torch.set_rng_state(saved["torch_rng"].cpu())
                    torch.cuda.set_rng_state_all([s.cpu() for s in saved["cuda_rng"]])
                    restored = True
                    saved["roundtrip_checked"] = True
                    torch.save(saved, path / "training_state.pt")
                atomic_write_json(folder / "latest.json", {"checkpoint": str(path.relative_to(folder)), "step": step,
                    "sha256": hashlib.sha256((path / "projector.safetensors").read_bytes()).hexdigest()}, ensure_ascii=False)
                manifest.update({"steps_completed": step, "micro_steps_completed": micro, "checkpoint_roundtrip_passed": restored})
                atomic_write_json(folder / "manifest.json", manifest, ensure_ascii=False)
            del state, output, compact, teacher_logits, student_logits, loss
        manifest.update({"status": "complete", "steps_completed": step, "checkpoint_roundtrip_passed": restored,
                         "elapsed_s": time.perf_counter() - elapsed_start, "completed_utc": datetime.now(timezone.utc).isoformat(),
                         "peak_allocated_bytes": torch.cuda.max_memory_allocated(args.device)})
        atomic_write_json(folder / "manifest.json", manifest, ensure_ascii=False)
    except BaseException as exc:
        manifest.update({"status": "failed", "error": str(exc), "traceback": traceback.format_exc(), "steps_completed": step})
        atomic_write_json(folder / "manifest.json", manifest, ensure_ascii=False)
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--variant", choices=VARIANTS, required=True)
    parser.add_argument("--steps", type=int, default=320)
    parser.add_argument("--accumulation", type=int, default=8)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    main(parser.parse_args())
