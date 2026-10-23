"""Two-GPU frozen-7B teacher/student training of a zero-init anchor adapter."""
import argparse
from contextlib import nullcontext
from datetime import datetime, timezone, timedelta
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from mllm_comp.analysis.common.anchor7b_runtime import (
    AnchorAdapterSpec, create_anchor_adapter, anchor_base, compact_anchor, answer_logits,
    load_runtime, save_adapter, load_adapter, implementation_hash, validate_identity,
    file_sha, global_batch_indices, explicit_generate,
)
from mllm_comp.analysis.common.condition_ledger import append_json
from mllm_comp.analysis.probes.visual_evidence_interaction import cached_decode
from mllm_comp.core.serialization import atomic_write_json
from scripts.train.qwen25_vl.train_learned_hybrid import load_examples, prepare_example


def parameter_hash(module):
    h = hashlib.sha256()
    for name, value in module.state_dict().items():
        h.update(name.encode())
        h.update(value.detach().float().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def recover_log(path, step):
    """Preserve an interrupted attempt, then retain only checkpointed log rows."""
    if not path.exists():
        return None
    raw=path.read_bytes()
    valid=[]
    for line in raw.splitlines(keepends=True):
        if not line.endswith(b"\n"):
            break
        row=json.loads(line)
        if row["step"]<=step:
            valid.append(line)
    recovered=b"".join(valid)
    if recovered==raw:
        return None
    archive=path.with_name(path.name+f".interrupted_{time.time_ns()}")
    path.rename(archive)
    path.write_bytes(recovered)
    return str(archive)


def run(args):
    import torch
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel as DDP
    import torch.nn.functional as F
    rank, world, local = int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"]), int(os.environ["LOCAL_RANK"])
    if world != 2:
        raise ValueError("This protocol requires exactly two DDP workers")
    torch.cuda.set_device(local)
    dist.init_process_group("nccl", timeout=timedelta(minutes=10))
    device = f"cuda:{local}"
    root, folder = args.root.resolve(), args.output.resolve()
    folder.mkdir(parents=True, exist_ok=True)
    identity = json.loads((root / "model_identity.json").read_text())
    config = {"base_identity_sha256": identity["identity_sha256"], "implementation_hash": implementation_hash(),
              "trainer_sha256": file_sha(__file__), "training_manifest_sha256": file_sha(root / "training_samples.json"),
              "seed": 42, "world_size": world, "accumulation_per_rank": args.accumulation,
              "global_batch": world*args.accumulation, "steps": args.steps, "smoke": args.smoke,
              "loss": {"answer_ce": 1., "teacher_kl": .5}, "learning_rate": .0002,
              "weight_decay": .01, "clip_grad_norm": 1., "adapter": {"hidden_size": 3584, "width": 256},
              "budget": 128, "positions": "rounded_native_mrope_cluster_centroid; text unchanged"}
    config_hash = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
    status = {**config, "config_hash": config_hash, "rank": rank, "status": "starting",
              "checkpoint": identity["checkpoint"], "revision": identity["revision"],
              "started_utc": datetime.now(timezone.utc).isoformat(), "steps_completed": 0}
    status_path = folder / f"rank_{rank}_manifest.json"
    def persist():
        atomic_write_json(status_path, status)
        if rank == 0:
            atomic_write_json(folder / "manifest.json", {k:v for k,v in status.items() if k != "rank"})
    persist()
    try:
        torch.manual_seed(42)
        torch.cuda.manual_seed_all(42)
        examples = load_examples(root / "training_samples.json")
        if args.smoke:
            examples = examples[:args.steps*world*args.accumulation]
        if len(examples) != args.steps*world*args.accumulation:
            raise ValueError("One-pass training requires exactly steps × global batch distinct images")
        if len({e["image_fingerprint"] for e in examples}) != len(examples):
            raise ValueError("Duplicate training image")
        # Avoid simultaneously staging two 16.6 GB checkpoints in shared host RAM.
        bundle = None
        for owner in range(world):
            if rank == owner:
                bundle = load_runtime(identity, device)
            dist.barrier()
        llm = bundle["llm"]
        llm.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        torch.manual_seed(42)
        module = create_anchor_adapter(AnchorAdapterSpec(hidden_size=3584, width=256)).to(device)
        optimizer = torch.optim.AdamW(module.parameters(), lr=.0002, weight_decay=.01)
        warmup = min(16, max(1, args.steps//10))
        def schedule(s):
            return (s+1)/warmup if s < warmup else .5*(1+math.cos(math.pi*min(1., (s-warmup)/max(1,args.steps-warmup))))
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)
        step = 0
        if args.resume and (folder / "latest.json").exists():
            pointer = json.loads((folder / "latest.json").read_text())
            checkpoint = folder / pointer["checkpoint"]
            if file_sha(checkpoint / "projector.safetensors") != pointer["sha256"]:
                raise ValueError("Checkpoint checksum differs from latest pointer")
            saved = torch.load(checkpoint / "training_state.pt", map_location=device, weights_only=False)
            if saved["config_hash"] != config_hash:
                raise ValueError("Cannot resume changed data/configuration/implementation")
            restored = load_adapter(checkpoint, identity, device)
            module.load_state_dict(restored.state_dict())
            del restored
            optimizer.load_state_dict(saved["optimizer"])
            scheduler.load_state_dict(saved["scheduler"])
            step = saved["step"]
            rng = saved["rank_rng"][rank]
            torch.set_rng_state(rng["torch_rng"].cpu())
            torch.cuda.set_rng_state(rng["cuda_rng"].cpu(), device)
            status["resumed_from_step"] = step
            archived=[]
            for log in [folder/f"rank_{rank}_data_trace.jsonl"]+([folder/"training.jsonl"] if rank==0 else []):
                archive=recover_log(log,step)
                if archive:
                    archived.append(archive)
            status["interrupted_log_archives"]=archived
            if (folder / f"rank_{rank}_gates.json").exists():
                status.update(json.loads((folder / f"rank_{rank}_gates.json").read_text()))
        status.update({"status": "running", "steps_completed": step, "train_images": len(examples),
                       "trainable_parameters": sum(p.numel() for p in module.parameters()),
                       "frozen_parameters": sum(p.numel() for p in bundle["model"].parameters()),
                       "trainable_dtype": "float32", "backbone_dtype": "bfloat16"})
        persist()
        if args.smoke and step == 0:
            gates = []
            for index in global_batch_indices(0, world_size=world, accumulation=args.accumulation, rank=rank):
                example = examples[index]
                state = prepare_example(bundle, example)
                base = anchor_base(state)
                module.eval()
                with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                    output = module(base)
                if not torch.equal(output["tokens"], base["tokens"]):
                    raise ValueError("Zero-init adapter changed feature-anchor values")
                original, corrected = compact_anchor(state, base), compact_anchor(state, output)
                if not torch.equal(original["hidden"], corrected["hidden"]) or not torch.equal(original["positions"], corrected["positions"]):
                    raise ValueError("Zero-init layout/positions changed")
                max_new = 32 if example["source"] == "textvqa_train" else 16
                eos = bundle["model"].generation_config.eos_token_id
                eos = [eos] if isinstance(eos,int) else list(eos)
                llm.eval()
                with torch.inference_mode():
                    dense = cached_decode(llm=llm, head=bundle["model"].lm_head, embedding=bundle["model"].get_input_embeddings(),
                        hidden=state["hidden"][:, :state["prefix"]], positions=state["positions"][:, :, :state["prefix"]], eos_ids=eos, max_new=max_new)
                    native = explicit_generate(bundle, state, max_new=max_new)
                    if dense["generated_token_ids"] != native:
                        raise ValueError("7B native/manual dense greedy generation differs")
                    compressed = cached_decode(llm=llm, head=bundle["model"].lm_head, embedding=bundle["model"].get_input_embeddings(),
                        hidden=corrected["hidden"][:, :corrected["prefix"]], positions=corrected["positions"][:, :, :corrected["prefix"]], eos_ids=eos, max_new=max_new)
                if compressed["first_appended_token_position"] != int(corrected["positions"][:, :, :corrected["prefix"]].max())+1:
                    raise ValueError("Compressed first decode position mismatch")
                gates.append({"sample_id": example["sample_id"], "source_tokens": len(state["features"]), "result_tokens":128,
                              "zero_init_values_and_positions_identical":True, "native_generation_identical":True,
                              "compressed_cached_decode":compressed, "repetition_penalty":1.})
            status.update({"native_generation_identity_passed":True, "zero_init_identity_passed":True,
                           "compressed_cache_smoke_passed":True, "smoke_images_per_rank":len(gates)})
            atomic_write_json(folder / f"rank_{rank}_gates.json", {k:status[k] for k in ["native_generation_identity_passed",
                "zero_init_identity_passed", "compressed_cache_smoke_passed", "smoke_images_per_rank"]})
            atomic_write_json(folder / f"rank_{rank}_generation_smoke.json", gates)
            persist()
        ddp = DDP(module, device_ids=[local], output_device=local, broadcast_buffers=False, find_unused_parameters=False)
        started = time.perf_counter()
        while step < args.steps and (args.stop_after is None or step < args.stop_after):
            optimizer.zero_grad(set_to_none=True)
            totals = torch.zeros(3, device=device, dtype=torch.float64)
            indices = global_batch_indices(step, world_size=world, accumulation=args.accumulation, rank=rank)
            for micro, index in enumerate(indices):
                example = examples[index]
                state = prepare_example(bundle, example)
                base = anchor_base(state)
                llm.eval()
                with torch.no_grad():
                    teacher, teacher_targets = answer_logits(bundle, state["hidden"], state["positions"], state["teacher_inputs"]["labels"])
                llm.train()
                ddp.train()
                sync = nullcontext() if micro+1 == args.accumulation else ddp.no_sync()
                with sync:
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        output = ddp(base)
                        compact = compact_anchor(state, output)
                        student, targets = answer_logits(bundle, compact["hidden"], compact["positions"], compact["labels"])
                        if not torch.equal(teacher_targets, targets):
                            raise ValueError("Teacher/student answer labels differ")
                        ce = F.cross_entropy(student, targets)
                        kl = F.kl_div(F.log_softmax(student,-1), F.softmax(teacher,-1), reduction="batchmean").clamp_min(0)
                        loss = ce+.5*kl
                    if not bool(torch.isfinite(loss)):
                        raise ValueError("Nonfinite training loss")
                    (loss/args.accumulation).backward()
                totals += torch.stack([loss.detach(),ce.detach(),kl.detach()]).double()/args.accumulation
                del state, base, output, compact, student, teacher, loss
            if any(p.grad is not None for p in bundle["model"].parameters()):
                raise ValueError("Frozen native 7B backbone accumulated gradients")
            norm = torch.nn.utils.clip_grad_norm_(module.parameters(),1.)
            if not bool(torch.isfinite(norm)) or float(norm)==0:
                raise ValueError("Invalid MLP gradients")
            optimizer.step()
            scheduler.step()
            step += 1
            dist.all_reduce(totals,op=dist.ReduceOp.SUM)
            totals /= world
            append_json(folder / f"rank_{rank}_data_trace.jsonl", {"step":step,"rank":rank,"indices":indices,
                "sample_ids":[examples[i]["sample_id"] for i in indices]})
            if rank==0:
                record={"step":step,"global_examples_seen":step*world*args.accumulation,"loss":float(totals[0]),
                    "answer_ce":float(totals[1]),"teacher_kl":float(totals[2]),"gradient_norm":float(norm),
                    "learning_rate":scheduler.get_last_lr()[0],"elapsed_s":time.perf_counter()-started}
                append_json(folder / "training.jsonl",record)
                print(json.dumps(record),flush=True)
            if step==1 or step%16==0 or step==args.steps or step==args.stop_after:
                hashes=[None]*world
                dist.all_gather_object(hashes,parameter_hash(module))
                if len(set(hashes))!=1:
                    raise ValueError("DDP workers have different MLP parameters")
                rank_rng=[None]*world
                dist.all_gather_object(rank_rng,{"torch_rng":torch.get_rng_state(),"cuda_rng":torch.cuda.get_rng_state(device)})
                if rank==0:
                    checkpoint=folder / "checkpoints" / f"step_{step:06d}"
                    save_adapter(module,checkpoint,identity)
                    torch.save({"config_hash":config_hash,"step":step,"optimizer":optimizer.state_dict(),
                        "scheduler":scheduler.state_dict(),"rank_rng":rank_rng},checkpoint / "training_state.pt")
                    restored=load_adapter(checkpoint,identity,device)
                    if parameter_hash(restored)!=hashes[0]:
                        raise ValueError("Saved/restored MLP parameters differ")
                    del restored
                    # Reload checks must not change the resumed stochastic state.
                    torch.set_rng_state(rank_rng[rank]["torch_rng"])
                    torch.cuda.set_rng_state(rank_rng[rank]["cuda_rng"],device)
                    atomic_write_json(folder / "latest.json",{"checkpoint":str(checkpoint.relative_to(folder)),"step":step,
                        "sha256":file_sha(checkpoint / "projector.safetensors"),"parameter_sha256":hashes[0]})
                dist.barrier()
                status.update({"steps_completed":step,"ddp_parameter_identity_passed":True,
                    "checkpoint_roundtrip_passed":True,"global_examples_seen":step*world*args.accumulation})
                persist()
        validate_identity(identity)
        status.update({"status":"complete" if step==args.steps else "checkpointed", "steps_completed":step,
            "peak_allocated_bytes":torch.cuda.max_memory_allocated(device),"completed_utc":datetime.now(timezone.utc).isoformat()})
        persist()
        dist.barrier()
    except BaseException as exc:
        status.update({"status":"failed","error":str(exc),"traceback":traceback.format_exc()})
        persist()
        raise
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--root",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--steps",type=int,default=320)
    p.add_argument("--accumulation",type=int,default=4)
    p.add_argument("--stop-after",type=int)
    p.add_argument("--smoke",action="store_true")
    p.add_argument("--resume",action="store_true")
    run(p.parse_args())
