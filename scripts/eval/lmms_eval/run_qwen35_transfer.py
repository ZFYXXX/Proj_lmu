"""Frozen-source Qwen3.5-9B transfer, two independent GPU workers, no noise.

All task generation/scoring goes through the MLLM-Comp native eval entry.
Worker0 trains frozen-base small heads; worker1 evaluates dense/external arms.
No Qwen2.5 head weights, predictions or Qwen3.8 results are reused.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import fcntl
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT))
from mllm_comp.analysis.common.anchor7b_runtime import file_sha,validate_identity
from mllm_comp.analysis.common.learned_topk_experiment import (
    dependency_identity,digest_json,gpu_free_mib,owned_output,reject_overlap,
    source_identity,validate_training_entries,
)
from mllm_comp.analysis.common.qwen35_transfer import CHECKPOINT,EXTERNAL,METHOD_KIND,identify
from mllm_comp.core.serialization import atomic_write_json as write_json,atomic_write_text as write_text

COUNTS={"chartqa":2500,"ocrbench":1000,"ai2d":3088}
METRICS={"chartqa":"relaxed_overall","ocrbench":"ocrbench_accuracy","ai2d":"exact_match"}
METHODS=("dense",*EXTERNAL,"old_mlp","visual_selector","question_selector","protected_selector")
TRAIN_MANIFEST=ROOT/"outputs/train/qwen25_vl/anchor_mlp_7b_k128_ddp_20261021_v1/training_samples.json"
REFERENCE=ROOT/"outputs/train/qwen25_vl/context_visual_topk_20261024_v2"


def read(path):return json.loads(Path(path).read_text())
def now():return datetime.now(timezone.utc).isoformat()


def prepare(root,training_manifest=TRAIN_MANIFEST,reference=REFERENCE):
    root=owned_output(root)
    if root.exists():raise ValueError("Use a fresh run; never overwrite previous artifacts")
    cohort=read(reference/"cohort.json")
    entries=read(training_manifest)["entries"]
    validate_training_entries(entries)
    for task,count in COUNTS.items():
        if len(cohort[task]["rows"])!=count:raise ValueError("Wrong full cohort")
        reject_overlap(entries,cohort[task]["rows"])
    # Pin raw inputs, not historical scores or checkpoints.
    old=read(reference/"protocol.json")
    pins={str(training_manifest):file_sha(training_manifest),str(reference/"cohort.json"):file_sha(reference/"cohort.json")}
    for name,checksum in old["input_hashes"].items():
        if name.endswith(".parquet"):
            if file_sha(name)!=checksum:raise ValueError(f"Shared raw data changed: {name}")
            pins[name]=checksum
    for entry in entries:
        if entry["path"] not in pins:pins[entry["path"]]=file_sha(entry["path"])
    identity=identify(CHECKPOINT)
    root.mkdir(parents=True)
    write_json(root/"model_identity.json",identity)
    # Dependency source plus actual Transformers import, not distribution metadata alone.
    import transformers
    deps=dependency_identity()
    deps["transformers_import"]={"version":transformers.__version__,"file":str(Path(transformers.__file__).resolve())}
    for rel in ("models/qwen3_5/modeling_qwen3_5.py","generation/utils.py","cache_utils.py"):
        filename=Path(transformers.__file__).parent/rel
        deps[rel]=file_sha(filename)
    protocol={"created_utc":now(),"source":source_identity(ROOT),"dependencies":deps,
        "identity_path":str(root/"model_identity.json"),"training_manifest":str(training_manifest),
        "cohort_path":str(reference/"cohort.json"),"input_hashes":pins,"methods":list(METHODS),"counts":COUNTS,
        "training":{"seed":42,"steps":320,"global_batch":8,"lr":2e-4,"weight_decay":.01,
            "warmup_steps":16,"loss":"CE + .5 teacher KL + annealed soft/hard BCE for selectors",
            "base_frozen":True,"training_images":2560,"trained_head_kinds":["visual","mlp","question","word"]},
        "eval":{"target_tokens":128,"seed":42,"greedy":True,"repetition_penalty":1.05,
            "enable_thinking":False,"max_new_tokens":{"chartqa":16,"ocrbench":128,"ai2d":16},
            "min_pixels":200704,"max_pixels":1605632,"smoke_per_task":8,"no_development_pass":True},
        "limits":["Qwen3.5-9B versus Qwen2.5-7B also changes scale; not an architecture-only attribution.",
            "External methods are controlled Qwen3.5 adaptations, not official paper protocol reproductions.",
            "Bifocal Stage1 only; Stage1+2 requires a separate hybrid-decoder adaptation.",
            "No noise gate, OT, dynamic budget, or new recovery variant.",
            "Shared-GPU timings are descriptive, not matched speed benchmarks."]}
    write_json(root/"protocol.json",protocol)
    write_json(root/"manifest.json",{"status":"prepared","stage":"prepared","created_utc":now(),
        "protocol_sha256":file_sha(root/"protocol.json"),"source":protocol["source"],"checkpoint":CHECKPOINT})
    return protocol


def validate(root):
    p=read(root/"protocol.json")
    checksum=file_sha(root/"protocol.json")
    if read(root/"manifest.json")["protocol_sha256"]!=checksum:raise ValueError("Protocol changed")
    if p["source"]!=source_identity(ROOT):raise ValueError("Execution source changed; create a new run")
    for name,value in p["input_hashes"].items():
        if file_sha(name)!=value:raise ValueError(f"Input changed: {name}")
    validate_identity(read(p["identity_path"]))
    import transformers
    if transformers.__version__!=p["dependencies"]["transformers_import"]["version"]:
        raise ValueError("Wrong Transformers runtime")
    p["protocol_sha256"]=checksum
    return p


def child_env(root,gpu):
    env={**os.environ,"CUDA_VISIBLE_DEVICES":str(gpu),"OMP_NUM_THREADS":"2","MKL_NUM_THREADS":"2",
        "TOKENIZERS_PARALLELISM":"false","CUBLAS_WORKSPACE_CONFIG":":4096:8",
        "PYTHONPATH":f"/home/ZhouFengyu/fastv-repro/qwen38-triton38:/home/ZhouFengyu/fastv-repro/qwen38-transformers-519:{ROOT}",
        "MLLM_TOPK_HASH_LOCAL_SOURCE":"1","MLLM_TOPK_NORMAL_GPU":"1",
        "HF_HOME":str(root/"cache/huggingface"),"HF_DATASETS_CACHE":str(root/"cache/datasets"),
        "XDG_CACHE_HOME":str(root/"cache/xdg"),"HF_HUB_OFFLINE":"1","HF_DATASETS_OFFLINE":"1",
        "WANDB_MODE":"disabled"}
    for key in ("HF_HOME","HF_DATASETS_CACHE","XDG_CACHE_HOME"):Path(env[key]).mkdir(parents=True,exist_ok=True)
    return env


def execute(root,gpu,command,log,*,training=False,extra=None):
    log.parent.mkdir(parents=True,exist_ok=True)
    required=65536 if training else 40960
    while gpu_free_mib().get(gpu,0)<required:
        write_json(log.with_suffix(".waiting.json"),{"status":"waiting_for_memory","gpu":gpu,
            "minimum_free_mib":required,"updated_utc":now()})
        time.sleep(30)
    env=child_env(root,gpu);env.update(extra or {})
    with log.open("x") as stream:
        child=subprocess.Popen(command,cwd=ROOT,env=env,stdout=stream,stderr=subprocess.STDOUT)
        write_json(log.with_suffix(".process.json"),{"pid":child.pid,"gpu":gpu,"command":command,
            "log":str(log),"started_utc":now(),"status":"running"})
        code=child.wait()
    write_json(log.with_suffix(".process.json"),{"pid":child.pid,"gpu":gpu,"command":command,
        "log":str(log),"finished_utc":now(),"status":"complete" if code==0 else "failed","returncode":code})
    if code:raise RuntimeError(f"Child failed ({code}): {log}")


def config(root,p,method,phase,task,run):
    identity=read(p["identity_path"])
    stages=[]
    if method!="dense":
        cfg={"method":method,"target_count":128,"identity_path":p["identity_path"]}
        if method in METHOD_KIND:
            directory=root/"training"/METHOD_KIND[method]
            pointer=read(directory/"latest.json")
            if pointer["step"]!=320 or read(directory/"manifest.json")["status"]!="complete":raise ValueError("Head incomplete")
            cfg["checkpoint_path"]=str(directory/pointer["checkpoint"])
        name="qwen35_transferred_projector"
        stages=[{"name":method,"algorithm":name,"plugin":name,"variant":f"{name}.image",
            "binding":f"{name}.image.qwen3_5","placement":"projector_level","method":"token_compression",
            "object_family":"tensor_slot","execution_phase":"generate","modalities":["image"],
            "target_modalities":["image"],"conditioning_modalities":["text"],"config":cfg}]
    return {"experiment":{"name":f"qwen35_{phase}_{method}_{task}","seed":42},
        "backbone":{"name":"qwen3_5","model_path":identity["checkpoint"],"model_revision":identity["revision"],
            "trust_remote_code":False,"config":{"torch_dtype":"bfloat16","attn_implementation":"eager",
                "model_load_kwargs":{"local_files_only":True},"processor_load_kwargs":{"local_files_only":True,
                    "use_fast":False,"min_pixels":200704,"max_pixels":1605632}}},
        "pipeline":{"stages":stages},"analysis":{},"train":{"compression_mode":"none"},
        "eval":{"backend":"lmms_eval","compression_mode":"compressed" if stages else "none",
            "target":"compressed" if stages else "baseline","checkpoint_path":identity["checkpoint"],
            "tasks":[f"context_topk_{task}"],"runtime_batch_size":1,"num_processes":1,
            "sample_limit":8 if phase=="smoke" else None,"generation":{"do_sample":False,"repetition_penalty":1.05},
            "args":{"force_simple":True,"log_samples":True,"verbosity":"INFO",
                "include_path":str(ROOT/"configs/eval/lmms_eval/tasks/context_visual_topk")}},
        "data":{"dataset_id":"context_topk_native_tasks","sample_split":"test"},
        "output":{"run_dir":str(run),"manifest_path":str(run/"manifest.json")}}


def check(run,method,phase,task):
    n=8 if phase=="smoke" else COUNTS[task]
    files=list(run.glob(f"**/*_samples_context_topk_{task}.jsonl"))
    if len(files)!=1:raise ValueError("Missing native predictions")
    rows=[json.loads(s) for s in files[0].read_text().splitlines()]
    if len(rows)!=n or {r["doc_id"] for r in rows}!=set(range(n)):raise ValueError("Missing/duplicate predictions")
    if any(not math.isfinite(float(r[METRICS[task]])) for r in rows):raise ValueError("Nonfinite native score")
    performance=[json.loads(s) for s in (run/"request_performance.jsonl").read_text().splitlines()]
    if len(performance)!=n or any(r["status"]!="complete" for r in performance):raise ValueError("Incomplete generation")
    source={r["sample_id"]:r["original_visual_tokens"] for r in performance}
    if len(source)!=n:raise ValueError("Duplicate generation trace")
    if method!="dense":
        traces=[json.loads(s) for s in (run/"compression_runtime.requests.jsonl").read_text().splitlines()]
        if len(traces)!=n:raise ValueError("Missing compression trace")
        seen=set()
        for row in traces:
            sid=row["request"]["sample_id"]
            if sid in seen or sid not in source:raise ValueError("Invalid trace pairing")
            seen.add(sid)
            stages=row.get("stages",[])
            # Runtime records are inspected recursively; avoid assuming the 2.5 layout wrapper.
            def selections(value):
                if isinstance(value,dict):
                    if "expected_first_decode_position" in value and "selected_indices" in value:yield value
                    else:
                        for v in value.values():yield from selections(v)
                elif isinstance(value,list):
                    for v in value:yield from selections(v)
            records=list(selections(row))
            if not records:raise ValueError("No Qwen3.5 position/budget trace")
            for t in records:
                k=min(128,source[sid])
                if t["source_count"]!=source[sid] or t["result_count"]!=k or len(set(t["selected_indices"]))!=k:
                    raise ValueError("Wrong per-image budget")
                if t["decode_position_verified"] is False:raise ValueError("Decode position failed")
                if t["observed_visual_tokens_per_decoder_layer"]!=[k]*32:raise ValueError("Hybrid prefill token mismatch")
    return {"requests":n,"score":sum(float(r[METRICS[task]]) for r in rows)/n,"samples":str(files[0])}


def evaluate(root,p,gpu,method,phase,task):
    import yaml
    directory=root/f"eval/{phase}/{method}/{task}"
    marker=directory/"complete.json"
    if marker.exists():
        saved=read(marker)
        if file_sha(saved["config"])!=saved["config_sha256"]:raise ValueError("Completed config changed")
        check(Path(saved["run"]),method,phase,task)
        return
    stamp=time.time_ns();run=directory/f"attempt_{stamp}"
    filename=root/"configs"/f"{phase}_{method}_{task}_{stamp}.yaml"
    filename.parent.mkdir(parents=True,exist_ok=True)
    write_text(filename,yaml.safe_dump(config(root,p,method,phase,task,run),sort_keys=False))
    execute(root,gpu,[sys.executable,"-m","mllm_comp.cli.main","eval","--config",str(filename)],
        directory/f"launch_{stamp}.log",extra={"MLLM_TOPK_PERFORMANCE":str(run/"request_performance.jsonl")})
    checks=check(run,method,phase,task)
    write_json(marker,{"finished_utc":now(),"run":str(run),"config":str(filename),
        "config_sha256":file_sha(filename),**checks})
    report(root,p)


def report(root,p):
    records=[]
    for method in METHODS:
        for task in COUNTS:
            marker=root/f"eval/full/{method}/{task}/complete.json"
            if marker.exists():records.append({"method":method,"task":task,**read(marker)})
    write_json(root/"summary.json",{"completed_cells":len(records),"total_cells":len(METHODS)*3,"results":records})
    lines=["# Qwen3.5-9B K128 transfer","","| Method | ChartQA | OCRBench | AI2D strict |","|---|---:|---:|---:|"]
    for method in METHODS:
        values=[]
        for task in COUNTS:
            match=[r for r in records if r["method"]==method and r["task"]==task]
            values.append((f"{match[0]['score']*1000:.0f}" if task=="ocrbench" else f"{match[0]['score']*100:.2f}%") if match else "pending")
        lines.append("| "+" | ".join((method,*values))+" |")
    write_text(root/"report.md","\n".join(lines+["",*p["limits"]])+"\n")


def worker(root,gpu,lane):
    p=validate(root)
    marker=root/f"worker_{lane}.json"
    lock=root/f"worker_{lane}.lock"
    with lock.open("a") as stream:
        fcntl.flock(stream,fcntl.LOCK_EX|fcntl.LOCK_NB)
        def stage(name,status="running",**kwargs):
            write_json(marker,{"status":status,"stage":name,"gpu":gpu,"pid":os.getpid(),"updated_utc":now(),**kwargs})
        try:
            stage("real_9b_gate")
            gate=root/f"gates/gpu{gpu}_{lane}"
            if not (gate/"complete.json").exists():
                execute(root,gpu,[sys.executable,str(ROOT/"scripts/eval/lmms_eval/run_qwen35_transfer.py"),
                    "gpu-gate","--root",str(root),"--output",str(gate)],gate/f"launch_{time.time_ns()}.log")
            if lane=="external":
                methods=("dense",*EXTERNAL)
                for method in methods:
                    for task in COUNTS:
                        stage(f"smoke:{method}:{task}");evaluate(root,p,gpu,method,"smoke",task)
                    for task in COUNTS:
                        stage(f"full:{method}:{task}");evaluate(root,p,gpu,method,"full",task)
            else:
                for kind in ("visual","mlp","question","word"):
                    smoke=root/"gates"/f"train_{kind}"
                    if not (smoke/"manifest.json").exists() or read(smoke/"manifest.json")["status"]!="complete":
                        stage(f"training_smoke:{kind}")
                        execute(root,gpu,[sys.executable,str(ROOT/"scripts/train/train_qwen35_transfer.py"),"--root",str(root),
                            "--kind",kind,"--output",str(smoke),"--smoke"],smoke/f"launch_{time.time_ns()}.log",training=True)
                    output=root/"training"/kind
                    if not (output/"manifest.json").exists() or read(output/"manifest.json")["status"]!="complete":
                        stage(f"training:{kind}")
                        execute(root,gpu,[sys.executable,str(ROOT/"scripts/train/train_qwen35_transfer.py"),"--root",str(root),
                            "--kind",kind,"--output",str(output)],output/f"launch_{time.time_ns()}.log",training=True)
                    method=next(m for m,k in METHOD_KIND.items() if k==kind)
                    for task in COUNTS:
                        stage(f"smoke:{method}:{task}");evaluate(root,p,gpu,method,"smoke",task)
                    for task in COUNTS:
                        stage(f"full:{method}:{task}");evaluate(root,p,gpu,method,"full",task)
            stage("complete","complete")
        except BaseException as error:
            stage("failed","failed",error=str(error),traceback=traceback.format_exc())
            raise


def gpu_gate(root,output):
    import torch
    from mllm_comp.analysis.common.qwen35_transfer import load_bundle,create_head,prepare_training_example
    from mllm_comp.compressors.comparison.qwen35_topk import Qwen35TransferredProjector
    from scripts.train.qwen25_vl.train_learned_hybrid import load_examples
    p=validate(root)
    bundle=load_bundle(read(p["identity_path"]),"cuda:0")
    example=load_examples(Path(p["training_manifest"]))[0]
    state=prepare_training_example(bundle,example)
    model=bundle["model"];tokenizer=bundle["processor"].tokenizer
    with torch.no_grad():dense=model.generate(**state["prompt"],max_new_tokens=3,min_new_tokens=3,do_sample=False,repetition_penalty=1.05)
    records=[]
    for method in (*METHOD_KIND,*EXTERNAL):
        visual=create_head("visual",4096)
        head=create_head(METHOD_KIND[method],4096,visual=visual) if method in METHOD_KIND else None
        if head is not None:head=head.cuda()
        session=Qwen35TransferredProjector(model,{"method":method,"target_count":128},module=head,tokenizer=tokenizer).install()
        try:
            with torch.no_grad():model.generate(**state["prompt"],max_new_tokens=3,min_new_tokens=3,do_sample=False,repetition_penalty=1.05)
            trace=session.runtime_metadata()["selection"]
            if trace["result_count"]!=128 or not trace["decode_position_verified"]:raise ValueError("Real 9B generation gate failed")
            records.append({"method":method,**trace})
        finally:session.uninstall()
        del session,head,visual
    session=Qwen35TransferredProjector(model,{"method":"old_mlp","target_count":len(state["features"])},
        module=create_head("mlp",4096).cuda(),tokenizer=tokenizer).install()
    try:
        with torch.no_grad():identity=model.generate(**state["prompt"],max_new_tokens=3,min_new_tokens=3,do_sample=False,repetition_penalty=1.05)
        if not torch.equal(identity,dense):raise ValueError("Real 9B identity differs from dense")
    finally:session.uninstall()
    write_json(output/"complete.json",{"status":"complete","finished_utc":now(),"checkpoint":CHECKPOINT,
        "dense_identity_equal":True,"all_methods_first_decode_verified":True,"records":records,
        "peak_allocated_bytes":torch.cuda.max_memory_allocated()})


def pipeline(root,gpus):
    p=validate(root)
    manifest=read(root/"manifest.json")
    with (root/"pipeline.lock").open("a") as stream:
        fcntl.flock(stream,fcntl.LOCK_EX|fcntl.LOCK_NB)
        manifest.update(status="running",stage="two_gpu_workers",pid=os.getpid(),gpus=gpus,started_utc=now())
        write_json(root/"manifest.json",manifest)
        try:
            with ThreadPoolExecutor(max_workers=2) as pool:
                jobs=[pool.submit(worker,root,gpus[0],"training"),pool.submit(worker,root,gpus[1],"external")]
                for job in jobs:job.result()
            report(root,p)
            if read(root/"summary.json")["completed_cells"]!=len(METHODS)*3:raise ValueError("Incomplete full matrix")
            manifest.update(status="complete",stage="complete",finished_utc=now())
        except BaseException as error:
            manifest.update(status="failed",error=str(error),traceback=traceback.format_exc())
            raise
        finally:write_json(root/"manifest.json",manifest)


def main():
    parser=argparse.ArgumentParser();parser.add_argument("command",choices=("prepare","pipeline","gpu-gate","report"))
    parser.add_argument("--root",required=True,type=Path);parser.add_argument("--output",type=Path)
    parser.add_argument("--training-manifest",type=Path,default=TRAIN_MANIFEST)
    parser.add_argument("--reference",type=Path,default=REFERENCE)
    parser.add_argument("--gpus",type=int,nargs=2,default=[0,1]);args=parser.parse_args()
    root=owned_output(args.root)
    if args.command=="prepare":prepare(root,args.training_manifest.resolve(),args.reference.resolve())
    elif args.command=="pipeline":pipeline(root,args.gpus)
    elif args.command=="gpu-gate":gpu_gate(root,owned_output(args.output))
    else:report(root,validate(root))


if __name__=="__main__":main()
