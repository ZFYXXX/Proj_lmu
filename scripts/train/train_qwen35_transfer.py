"""Same-data frozen Qwen3.5-9B small-head training; standalone restartable job."""
import argparse
import json
import os
from pathlib import Path
import random
import sys
import time
import traceback

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from mllm_comp.analysis.common.qwen35_transfer import (
    KINDS,answer_logits,create_head,load_bundle,load_head,prepare_training_example,
    question_embeddings,save_head,select_tokens,
)
from mllm_comp.analysis.common.learned_topk_experiment import (
    batch_indices,digest_json,lr_multiplier,rng_state,restore_rng,training_loss,
)
from mllm_comp.analysis.common.topk_training_modes import parameter_digest,training_selection
from mllm_comp.core.serialization import atomic_write_json as write_json
from scripts.train.qwen25_vl.train_learned_hybrid import load_examples


def read(path):return json.loads(Path(path).read_text())


def train(root,kind,output,*,smoke=False):
    import numpy as np
    import torch
    import torch.nn.functional as F
    from scripts.eval.lmms_eval.run_qwen35_transfer import validate
    p=validate(root)
    identity=read(p["identity_path"])
    if kind not in KINDS:raise ValueError("Unknown head; noise is excluded")
    for seed_fn in (random.seed,np.random.seed,torch.manual_seed):seed_fn(42)
    torch.cuda.set_device(0);torch.cuda.manual_seed(42)
    torch.backends.cuda.matmul.allow_tf32=False
    device=torch.device("cuda:0")
    bundle=load_bundle(identity,device)
    examples=load_examples(Path(p["training_manifest"]))
    if kind in ("question","word"):
        pointer=read(root/"training/visual/latest.json")
        if pointer["step"]!=320:raise ValueError("Complete the visual prior before conditioned heads")
        visual=load_head(root/"training/visual"/pointer["checkpoint"],identity=identity,device="cpu")
    else:visual=None
    from mllm_comp.compressors.image.word_visual_topk import template_patterns
    module=create_head(kind,identity["hidden_size"],visual=visual,patterns=template_patterns(bundle["processor"].tokenizer)).to(device)
    optimizer=torch.optim.AdamW([x for x in module.parameters() if x.requires_grad],lr=2e-4,weight_decay=.01)
    scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer,lr_multiplier)
    contract={"protocol_sha256":p["protocol_sha256"],"kind":kind,"source":p["source"],
              "training_manifest_sha256":p["input_hashes"][p["training_manifest"]],"base_identity":identity["identity_sha256"],
              "seed":42,"global_batch":8,"steps":320,"smoke":smoke}
    output.mkdir(parents=True,exist_ok=True)
    manifest={"status":"running","kind":kind,"contract":contract,"steps_completed":0,"training_images":2560,
              "backbone_frozen":True,"head_trainable_parameters":sum(x.numel() for x in module.parameters() if x.requires_grad),
              "initial_parameter_sha256":parameter_digest(module)}
    step=0
    if (output/"latest.json").exists():
        pointer=read(output/"latest.json");checkpoint=output/pointer["checkpoint"]
        state=torch.load(checkpoint/"training_state.pt",map_location="cpu",weights_only=False)
        from mllm_comp.analysis.common.anchor7b_runtime import file_sha
        if state["contract"]!=contract or file_sha(checkpoint/"training_state.pt")!=pointer["state_sha256"]:
            raise ValueError("Resume contract or state hash changed")
        restored=load_head(checkpoint,identity=identity,device=device)
        module.load_state_dict(restored.state_dict());del restored
        optimizer.load_state_dict(state["optimizer"]);scheduler.load_state_dict(state["scheduler"])
        restore_rng(state["rng"],device);step=state["step"]
    write_json(output/"manifest.json",manifest)
    llm=bundle["llm"]
    llm.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant":False})
    def do_step(index):
        optimizer.zero_grad(set_to_none=True)
        totals={}
        for i in batch_indices(index):
            free,_=torch.cuda.mem_get_info(device)
            if free<8*2**30:raise RuntimeError("GPU headroom below8GiB; preserving last checkpoint")
            state=prepare_training_example(bundle,examples[i])
            llm.eval()
            with torch.no_grad():teacher,targets=answer_logits(bundle,state["hidden"],state["positions"],state["teacher"]["labels"])
            llm.train();module.train()
            if kind=="mlp":
                # Same anchor means + residual MLP and centroid positions as2.5.
                output_tokens,_=select_tokens(state["features"],method="old_mlp",module=module,grid=state["grid"])
                visual=state["visual"];local=output_tokens["indices"]
                keep=torch.ones(state["hidden"].shape[1],device=device,dtype=torch.bool);keep[visual]=False;keep[visual[local]]=True
                gather=keep.nonzero(as_tuple=True)[0]
                hidden=state["hidden"].index_select(1,gather).clone()
                positions=state["positions"][:,:,gather].clone()
                destinations=(state["teacher"]["input_ids"][:,gather][0]==bundle["model"].config.image_token_id).nonzero(as_tuple=True)[0]
                hidden[0,destinations]=output_tokens["tokens"]
                for j,group in enumerate(output_tokens["members"]):
                    positions[-3:,0,destinations[j]]=state["positions"][-3:,0,visual[list(group)]].float().mean(-1).round().to(positions.dtype)
                positions[0,0]=torch.arange(len(gather),device=device)
                labels=state["teacher"]["labels"][:,gather]
                selection=None
            else:
                kwargs={}
                if kind in ("question","word"):
                    q,_=question_embeddings(bundle["model"],bundle["processor"].tokenizer,state["prompt"],wordwise=kind=="word")
                    kwargs["question"]=q
                selection=training_selection(module,state["features"],"soft",**kwargs)
                hidden=state["hidden"].index_copy(1,state["visual"],selection["tokens"][None])
                positions=state["positions"];labels=state["teacher"]["labels"]
            with torch.autocast("cuda",dtype=torch.bfloat16):student,actual_targets=answer_logits(bundle,hidden,positions,labels)
            if not torch.equal(targets,actual_targets):raise ValueError("Teacher/student targets changed")
            if selection is None:
                ce=F.cross_entropy(student.float(),targets)
                kl=F.kl_div(F.log_softmax(student.float(),-1),F.softmax(teacher.float(),-1),reduction="batchmean").clamp_min(0)
                loss=ce+.5*kl
                metrics={"loss":float(loss.detach()),"answer_ce":float(ce.detach()),"teacher_kl":float(kl.detach())}
            else:loss,metrics=training_loss(student,teacher,targets,selection,index)
            if not loss.isfinite():raise ValueError("Nonfinite loss")
            (loss/8).backward()
            for name,value in metrics.items():totals[name]=totals.get(name,0)+value/8
            del state,student,teacher,loss,hidden
        if any(x.grad is not None for x in bundle["model"].parameters()):raise ValueError("Frozen backbone acquired gradients")
        norm=torch.nn.utils.clip_grad_norm_([x for x in module.parameters() if x.requires_grad],1.)
        if not norm.isfinite() or float(norm)==0:raise ValueError("Zero/nonfinite head gradient")
        optimizer.step();scheduler.step()
        return {**totals,"gradient_norm":float(norm),"parameter_sha256":parameter_digest(module)}
    def save(step):
        from mllm_comp.analysis.common.anchor7b_runtime import file_sha
        checkpoint=output/"checkpoints"/f"step_{step:06d}_{time.time_ns()}"
        save_head(module,checkpoint,kind=kind,identity=identity,provenance={"step":step,"contract":contract})
        torch.save({"step":step,"contract":contract,"optimizer":optimizer.state_dict(),"scheduler":scheduler.state_dict(),
                    "rng":rng_state(device)},checkpoint/"training_state.pt")
        write_json(output/"latest.json",{"step":step,"checkpoint":str(checkpoint.relative_to(output)),
            "state_sha256":file_sha(checkpoint/"training_state.pt")})
    try:
        limit=2 if smoke else 320
        while step<limit:
            started=time.perf_counter()
            result=do_step(step);step+=1
            if smoke and step==2:
                expected=parameter_digest(module)
                # Read the actual step1 checkpoint, not an in-memory copy.
                pointer=read(output/"latest.json");checkpoint=output/pointer["checkpoint"]
                replay=torch.load(checkpoint/"training_state.pt",map_location="cpu",weights_only=False)
                if replay["step"]!=1 or replay["contract"]!=contract:raise ValueError("Wrong replay checkpoint")
                restored=load_head(checkpoint,identity=identity,device=device)
                module.load_state_dict(restored.state_dict());del restored
                optimizer.load_state_dict(replay["optimizer"])
                scheduler.load_state_dict(replay["scheduler"]);restore_rng(replay["rng"],device)
                again=do_step(1)
                if parameter_digest(module)!=expected:raise ValueError("Save/restore replay parameter mismatch")
                result["resume_replay_exact"]=True
            if step==1 or step%16==0 or step==limit:save(step)
            record={"step":step,"seen_images":step*8,"seconds":time.perf_counter()-started,**result}
            with (output/"training.jsonl").open("a") as stream:stream.write(json.dumps(record,allow_nan=False)+"\n")
            print(json.dumps(record),flush=True)
            manifest.update(steps_completed=step,peak_allocated_bytes=torch.cuda.max_memory_allocated(),status="running")
            write_json(output/"manifest.json",manifest)
        manifest.update(status="complete",resume_replay_exact=bool(smoke))
        write_json(output/"manifest.json",manifest)
    except BaseException as error:
        manifest.update(status="failed",error=str(error),traceback=traceback.format_exc(),steps_completed=step)
        write_json(output/"manifest.json",manifest)
        raise


if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("--root",required=True,type=Path)
    parser.add_argument("--output",required=True,type=Path);parser.add_argument("--kind",required=True,choices=KINDS)
    parser.add_argument("--smoke",action="store_true");args=parser.parse_args()
    train(args.root.resolve(),args.kind,args.output.resolve(),smoke=args.smoke)
