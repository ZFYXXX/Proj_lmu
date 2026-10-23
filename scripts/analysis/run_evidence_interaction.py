"""Durable two-GPU stage runner with fail-fast gates and frozen confirmation."""

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
import traceback

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(Path(__file__).parent))
from mllm_comp.analysis.probes.visual_evidence_interaction import implementation_hash
from mllm_comp.core.serialization import atomic_write_json
from report_evidence_interaction import report


def analysis_hashes(root):
    paths=[Path(__file__).resolve(),Path(__file__).with_name("report_evidence_interaction.py"),root/"preregistration.json"]
    paths.extend(sorted((root/"configs").glob("*.yaml")))
    return {str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}


def run(root,stages):
    root=Path(root).resolve(); logs=root/"logs";logs.mkdir(exist_ok=True)
    lock=root/"runner.lock"
    import fcntl
    with lock.open("a") as handle:
        fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
        manifest={"status":"running","pid":os.getpid(),"stages":stages,"started_utc":datetime.now(timezone.utc).isoformat(),
                  "platform":platform.platform(),"python":sys.version,"implementation_hash":implementation_hash(),
                  "software":{n:importlib.metadata.version(n) for n in ["torch","transformers","datasets","numpy"]},
                  "gpu":subprocess.check_output(["nvidia-smi","--query-gpu=index,name,uuid,memory.total,driver_version","--format=csv"],text=True),
                  "cohort_hash":hashlib.sha256((root/"cohort.json").read_bytes()).hexdigest(),
                  "analysis_hashes":analysis_hashes(root),"stage_status":{}}
        target=root/"run_manifest.json"
        try:
            for stage in stages:
                manifest["active_stage"]=stage
                if stage in {"confirmation","question_review","budget_review"}:
                    freeze=json.loads((root/"frozen.json").read_text())
                    if (freeze["implementation_hash"]!=implementation_hash()
                            or freeze["analysis_hashes"]!=analysis_hashes(root)
                            or freeze["cohort_hash"]!=manifest["cohort_hash"]):
                        raise RuntimeError("Implementation, statistical code, preregistration or configuration changed after freeze.")
                children=[]
                attempt=datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
                for gpu in [0,1]:
                    out=logs/f"{stage}_{gpu}_{attempt}.log"
                    command=[str(Path(sys.executable).parent/"mllm-comp"),"analyze","--config",str(root/"configs"/f"{stage}_{gpu}.yaml"),
                             "--device",f"cuda:{gpu}","--output_dir",str(root/"lifecycle"/f"{stage}_{gpu}_{attempt}")]
                    stream=out.open("w")
                    env={**os.environ,"PYTHONUNBUFFERED":"1","TOKENIZERS_PARALLELISM":"false","OMP_NUM_THREADS":"4"}
                    child=subprocess.Popen(command,cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT,env=env)
                    children.append((child,stream,str(out)))
                manifest["stage_status"][stage]={"status":"running","children":[{"pid":p.pid,"log":log} for p,_,log in children]}
                atomic_write_json(target,manifest,ensure_ascii=False)
                while any(p.poll() is None for p,_,_ in children):
                    failed=[p for p,_,_ in children if p.poll() not in {None,0}]
                    if failed:
                        for p,_,_ in children:
                            if p.poll() is None:p.terminate()
                        for p,_,_ in children:p.wait()
                        raise RuntimeError(f"Stage {stage} failed; see shard logs. Other evidence shard stopped.")
                    time.sleep(5)
                for p,stream,_ in children:
                    stream.close()
                    if p.returncode:raise RuntimeError(f"{stage} worker exited {p.returncode}.")
                result=report(root,stage)
                if result["status"]!="complete":raise RuntimeError(f"Incomplete {stage}: {result['completed_images']}/{result['expected_images']}")
                if result["implementation_hash"]!=implementation_hash():raise RuntimeError("Implementation changed while the stage was running.")
                if manifest["analysis_hashes"]!=analysis_hashes(root):raise RuntimeError("Statistical code or configuration changed during execution.")
                if stage=="smoke":
                    summaries=[json.loads(p.read_text()) for p in (root/"samples/smoke").glob("*/summary.json")]
                    if len(summaries)!=16 or any(s["native_generation_identity_passed"] is not True for s in summaries):
                        raise RuntimeError("Smoke native-generation parity gate failed.")
                if stage=="development":
                    freeze={"implementation_hash":implementation_hash(),"cohort_hash":manifest["cohort_hash"],
                            "analysis_hashes":analysis_hashes(root),
                            "preregistration_hash":hashlib.sha256((root/"preregistration.json").read_bytes()).hexdigest(),
                            "development_images":200,"frozen_utc":datetime.now(timezone.utc).isoformat(),
                            "estimated_confirmation_worker_hours":result["elapsed_s_sum"]*4/3600/2}
                    atomic_write_json(root/"frozen.json",freeze,ensure_ascii=False)
                manifest["stage_status"][stage].update({"status":"complete","completed_images":result["completed_images"]})
                atomic_write_json(target,manifest,ensure_ascii=False)
            manifest["status"]="complete"
            if all((root/f"{s}_report.md").exists() for s in ["confirmation","question_review","budget_review"]):
                sections=["# Visual evidence interaction: final report\n\nThe primary cohort contains 1000 independent images: 200 development and 800 confirmation. Secondary questions do not add independent images.\n"]
                sections.extend((root/f"{s}_report.md").read_text() for s in ["confirmation","question_review","budget_review"])
                (root/"final_report.md").write_text("\n\n".join(sections),encoding="utf-8")
        except BaseException as exc:
            manifest["status"]="failed";manifest["error"]=str(exc);manifest["traceback"]=traceback.format_exc()
            if manifest.get("active_stage") in manifest["stage_status"]:
                manifest["stage_status"][manifest["active_stage"]]["status"]="failed"
            raise
        finally:
            manifest["updated_utc"]=datetime.now(timezone.utc).isoformat()
            atomic_write_json(target,manifest,ensure_ascii=False)


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--root",type=Path,required=True)
    p.add_argument("--stages",nargs="+",default=["smoke","development","confirmation","question_review","budget_review"])
    p.add_argument("--launch",action="store_true");a=p.parse_args()
    if a.launch:
        root=a.root.resolve();log=root/"supervisor.log"
        with log.open("a") as stream:
            child=subprocess.Popen([sys.executable,str(Path(__file__).resolve()),"--root",str(root),"--stages",*a.stages],
                                   cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
        print(json.dumps({"supervisor_pid":child.pid,"log":str(log)}))
    else:run(a.root,a.stages)
