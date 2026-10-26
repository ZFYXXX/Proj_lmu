"""Frozen visual prior + learned word matching; isolated, resumable experiment."""
import argparse
from collections import defaultdict
from dataclasses import asdict
import fcntl
import itertools
import json
from pathlib import Path
import sys
import time
import traceback

import yaml

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT))
from mllm_comp.analysis.common.anchor7b_runtime import file_sha,validate_identity
from mllm_comp.analysis.common.learned_topk_experiment import (
    available_gpus,dependency_identity,owned_output,shared_policy,source_identity,reject_overlap,validate_training_entries,
)
from mllm_comp.analysis.common.topk_training_modes import parameter_digest
from mllm_comp.core.serialization import atomic_write_json as write_json,atomic_write_text as write_text
from scripts.train.qwen25_vl import run_context_visual_topk as context
from scripts.train.qwen25_vl.noise_ot_reporting import resolve_completed_plan
from scripts.train.qwen25_vl.run_context_visual_topk_selective_eval import sample_rows,task_contract
from scripts.diagnostics.run_selector_condition_audit import trace_rows

TASKS=("chartqa","ocrbench","ai2d")
CONTROLS=("dense","old_mlp","frozen_visual","question_context")


def read(path):return json.loads(Path(path).read_text())


def reference_traces(cell):
    """Historical controls may share a run and use a different native prefix."""
    result={}
    for line in (Path(cell["run"])/"compression_runtime.requests.jsonl").read_text().splitlines():
        row=json.loads(line)
        name,split,index,_=row["request"]["sample_id"].split(":")
        if name!=cell["task_name"]:continue
        if split!="test" or not index.isdigit() or int(index) in result:raise ValueError("Invalid reference trace ID")
        result[int(index)]=row["runtime"]["activations"][0]["selection"]
    if not result:raise ValueError("Missing reference task trace")
    return result


def control_artifacts(cell):
    pins=dict(cell.get("artifacts_sha256",{}))
    if "completion_marker" in cell:
        pins[cell["completion_marker"]]=cell["completion_marker_sha256"]
    if not pins:raise ValueError("Control lacks pinned completion/artifact provenance")
    for path,digest in pins.items():
        if file_sha(path)!=digest:raise ValueError(f"Historical control changed: {path}")
    return pins


def prepare(root,reference,assignment,evidence=None):
    from transformers import AutoTokenizer
    from mllm_comp.compressors.image.learned_visual_topk import load_selector as load_visual
    from mllm_comp.compressors.image.word_visual_topk import WordVisualTopK,WordVisualTopKSpec,save_selector,template_patterns
    if root.exists():raise ValueError("Choose a new root; preserve prior artifacts")
    old=read(reference/"arms/question/protocol.json")
    if read(reference/"manifest.json")["status"]!="complete" or old["dependencies"]!=dependency_identity():
        raise ValueError("Incomplete reference or software changed")
    for filename,digest in old["input_hashes"].items():
        if file_sha(filename)!=digest:raise ValueError(f"Reference input changed: {filename}")
    identity=read(old["identity_path"]);validate_identity(identity)
    entries=read(old["training_manifest"])["entries"];validate_training_entries(entries)
    cohort=read(reference/"cohort.json")
    for task in TASKS:
        reject_overlap(entries,cohort[task]["rows"])
        cfg=context.eval_config(reference,read(reference/"protocol.json"),"question_context","full",root/"unused")
        prior=Path(old["source"]["source_root"])/f"configs/eval/lmms_eval/tasks/context_visual_topk/{task}.yaml"
        current=ROOT/f"configs/eval/lmms_eval/tasks/context_visual_topk/{task}.yaml"
        if task_contract(prior,cfg)!=task_contract(current,cfg):raise ValueError("Native task contract changed")
    assignment=assignment.resolve()
    assignment_doc=read(assignment)
    if assignment_doc["base_identity_sha256"]!=identity["identity_sha256"]:raise ValueError("Question assignment uses wrong base")
    visual=load_visual(old["warmstart_checkpoint"],base_identity=identity["identity_sha256"],device="cpu")
    tokenizer=AutoTokenizer.from_pretrained(identity["checkpoint"],local_files_only=True,use_fast=False)
    spec=WordVisualTopKSpec(identity["hidden_size"] if "hidden_size" in identity else visual.spec.hidden_size,
        template_token_ids=template_patterns(tokenizer))
    module=WordVisualTopK(spec,visual=visual)
    evidence_doc=read(evidence) if evidence else {"format":"word_verified_evidence_v1","rows":{},"status":"no_verified_annotations"}
    if evidence_doc.get("format")!="word_verified_evidence_v1" or set(evidence_doc["rows"])-{e["sample_id"] for e in entries}:
        raise ValueError("Wrong/unknown evidence training manifest")
    # Never replace missing independent annotations with attention pseudo-labels.
    for row in evidence_doc["rows"].values():
        if row.get("status")!="verified" or row.get("source") not in ("human","dataset_ground_truth") or not row.get("reviewer"):
            raise ValueError("Unverified evidence supervision")
    root.mkdir(parents=True)
    write_json(root/"evidence_training.json",evidence_doc)
    save_selector(module,root/"initial",base_identity=identity["identity_sha256"],
        provenance={"status":"untrained_word_head","visual_prior_checkpoint":old["warmstart_checkpoint"]})
    pins={**old["input_hashes"],str(reference/"protocol.json"):file_sha(reference/"protocol.json"),
        str(reference/"evaluation_plan.json"):file_sha(reference/"evaluation_plan.json"),
        str(reference/"cohort.json"):file_sha(reference/"cohort.json"),str(assignment):file_sha(assignment),
        str(root/"evidence_training.json"):file_sha(root/"evidence_training.json")}
    for f in (root/"initial/selector.safetensors",root/"initial/selector_config.json"):pins[str(f)]=file_sha(f)
    plan=resolve_completed_plan(reference,read(reference/"evaluation_plan.json"),file_sha=file_sha,
        expected_source=read(reference/"protocol.json")["source"])
    controls={}
    for task in TASKS:
        controls[task]={}
        for method in CONTROLS:
            cell=next(c for c in plan["cells"] if (c["method"],c["dataset"])==(method,task))
            samples,_=sample_rows(Path(cell["run"]),cell["task_name"],context.COUNTS[task],context.METRICS[task])
            controls[task][method]=cell
            pins.update(control_artifacts(cell))
            pins[str(samples)]=file_sha(samples)
            if method in ("frozen_visual","question_context"):
                f=Path(cell["run"])/"compression_runtime.requests.jsonl";pins[str(f)]=file_sha(f)
    training={**old["training"],"training_mask_mode":"soft","frozen_visual_prior":True,"evidence_weight":1.,
        "initialization":"same fixed visual step320; new learned word projections; positive mix=1",
        "contextual_question_encoder":False,"verified_evidence_images":len(evidence_doc["rows"])}
    training.pop("context_mode",None)
    p={**old,"source":source_identity(ROOT),"created_utc":context.native.now(),"resource_policy":shared_policy(),
        "input_hashes":pins,"selector_type":"word_visual_topk","word_spec":asdict(spec),
        "initial_parameter_sha256":parameter_digest(module),"reference":str(reference),"controls":controls,
        "audit_assignment":str(assignment),"evidence_training_manifest":str(root/"evidence_training.json"),
        "training":training,"methods":["word_visual_topk"],"allowed_gpus":[1],
        "claims":"Exploratory learned word-matching residual, not a semantic-grounding or novelty proof",
        "limits":["Frozen Qwen2.5-VL-7B and frozen visual prior; only the word-matching residual is trained.",
            "Same 2560 independent training images, seed42, batch8, 320 steps; fixed step320, no test-based selection.",
            "No verified evidence annotations are fabricated. Evidence loss is inactive when the manifest is empty.",
            "Raw word embeddings are not contextual; relation/negation comprehension is not established.",
            "No OT, noise, token recovery, dynamic K, Qwen3.5, random-pruning reruns or 500-image dev pass.",
            "Three native full evaluations only for the NEW method; reuse pinned old quality controls, not timing comparisons.",
            "Wrong/generic selector audits are scorer-only; the LLM always receives its true question.",
            "Native same-image question variation is unverified for distinct evidence; larger selection change alone is not success.",
            "Semantic evidence and paraphrase validation remain pending independent review. Full scores are exploratory."]}
    write_json(root/"protocol.json",p)
    write_json(root/"manifest.json",{"status":"prepared","source":p["source"],"completed_cells":[],
        "protocol_sha256":file_sha(root/"protocol.json"),"created_utc":context.native.now(),
        "semantic_evidence_status":"pending_review","verified_evidence_training_images":len(evidence_doc["rows"])})
    print(json.dumps({"train_images":2560,"steps":320,"word_trainable_parameters":sum(x.numel() for x in module.parameters() if x.requires_grad),
        "frozen_prior_parameters":sum(x.numel() for x in module.visual.parameters()),"full_eval_requests":6588,
        "verified_evidence_training_images":len(evidence_doc["rows"])}),flush=True)


def validate(root):
    p=context.native.validate(root)
    if read(root/"manifest.json")["protocol_sha256"]!=file_sha(root/"protocol.json"):raise ValueError("Word protocol changed")
    return p


def config(root,p,method,phase,task,run):
    reference=Path(p["reference"])
    cfg=context.eval_config(reference,read(reference/"protocol.json"),"question_context",phase,run)
    trained=method=="trained"
    if trained and (read(root/"training/manifest.json")["status"]!="complete" or read(root/"training/latest.json")["step"]!=320):
        raise ValueError("Only evaluate the completed fixed step320")
    checkpoint=context.native.latest(root/"training") if trained else root/"initial"
    stage,=cfg["pipeline"]["stages"]
    stage.update(name=f"word_{method}",algorithm="word_visual_topk",plugin="word_visual_topk",
        variant="word_visual_topk.image",binding="word_visual_topk.image.qwen25_vl",conditioning_modalities=["text"])
    stage["config"]={"target_count":128,"width":256,"match_width":128,"seed":42,
        "checkpoint_path":str(checkpoint),"identity_path":p["identity_path"],"visual_only":method=="visual_only",
        "audit_conditions":p["audit_assignment"],"audit_conditions_sha256":p["input_hashes"][p["audit_assignment"]]}
    cfg["experiment"]["name"]=f"word_topk_{phase}_{method}_{task}"
    cfg["eval"].update(tasks=[f"context_topk_{task}"],sample_limit=8 if phase=="smoke" else None)
    return cfg


def check(root,p,run,method,phase,task):
    checks=context.native.check_eval(run,method,phase,tasks=(task,),counts=context.COUNTS,prefix="context_topk")
    n=8 if phase=="smoke" else context.COUNTS[task]
    _,rows=sample_rows(run,f"context_topk_{task}",n,context.METRICS[task])
    ref=p["controls"][task]["frozen_visual"]
    _,old=sample_rows(Path(ref["run"]),ref["task_name"],context.COUNTS[task],context.METRICS[task])
    old_traces=reference_traces(ref) if method=="visual_only" else None
    for i,t in trace_rows(run,task).items():
        if not t["word_matching"] or not t["visual_prior_frozen"] or t["semantic_evidence_verified"]:
            raise ValueError("Missing word/prior/evidence honesty trace")
        if not t["question_tokens"] or not t["raw_question_ids_sha256"] or set(t["question_selection_audit"])!={"shuffled","generic"}:
            raise ValueError("Missing question selection diagnostics")
        if (rows[i]["input"],rows[i]["target"])!=(old[i]["input"],old[i]["target"]):raise ValueError("Original LLM prompt/target changed")
        if method=="visual_only" and (t["selected_indices"]!=old_traces[i]["selected_indices"] or rows[i]["filtered_resps"]!=old[i]["filtered_resps"]):
            raise ValueError("Frozen visual-prior control fails historical parity")
    return checks


def report(root,p):
    cohort=read(Path(p["reference"])/"cohort.json")
    records=[];pair_records=[]
    for task in TASKS:
        marker=root/f"eval/full/trained/{task}/complete.json"
        if not marker.exists():continue
        run=Path(read(marker)["run"]);metric=context.METRICS[task]
        _,rows=sample_rows(run,f"context_topk_{task}",context.COUNTS[task],metric)
        scores=[float(r[metric]) for r in rows];images=[r["image_fingerprint"] for r in cohort[task]["rows"]]
        traces=trace_rows(run,task)
        record={"task":task,"questions":len(rows),"independent_images":len(set(images)),
            "score":sum(scores)/len(scores),"controls":{},"semantic_evidence_status":"pending_review",
            "selection_audits":{}}
        for method,ref in p["controls"][task].items():
            _,old=sample_rows(Path(ref["run"]),ref["task_name"],len(rows),metric)
            if any((a["input"],a["target"])!=(b["input"],b["target"]) for a,b in zip(rows,old)):raise ValueError("Report prompt/target mismatch")
            values=[float(r[metric]) for r in old]
            record["controls"][method]={"score":sum(values)/len(values),
                "paired_difference":context.clustered_difference(scores,values,images)}
        for mode in ("shuffled","generic"):
            audits=[t["question_selection_audit"][mode] for t in traces.values()]
            informative=[a for a in audits if a["question_changed_after_cleaning"]]
            record["selection_audits"][mode]={"questions_changed_after_cleaning":len(informative),
                "mean_exchanged_tokens_all":sum(a["exchanged_tokens"] for a in audits)/len(audits),
                "mean_exchanged_tokens_changed_questions":sum(a["exchanged_tokens"] for a in informative)/len(informative) if informative else None}
        groups=defaultdict(list)
        for i,image in enumerate(images):groups[image].append(i)
        for image,ids in groups.items():
            for a,b in itertools.combinations(ids,2):
                if traces[a]["question_ids_sha256"]==traces[b]["question_ids_sha256"]:continue
                sa,sb=set(traces[a]["selected_indices"]),set(traces[b]["selected_indices"])
                pair_records.append({"task":task,"image_fingerprint":image,"doc_ids":[a,b],
                    "questions":[rows[a]["input"],rows[b]["input"]],"selection_overlap":len(sa&sb)/min(len(sa),len(sb)),
                    "distinct_evidence_verified":False,"paraphrase_verified":False,"status":"pending_semantic_review"})
        records.append(record)
    write_json(root/"summary.json",{"completed_cells":len(records),"total_cells":3,"results":records,"limits":p["limits"]})
    write_text(root/"native_same_image_pairs.jsonl","".join(json.dumps(r,ensure_ascii=False)+"\n" for r in pair_records))
    lines=["# Word matching residual: fixed K128, exploratory","","| Task | New word head | Frozen visual | Old global question | Old aggregation MLP | Dense |","|---|---:|---:|---:|---:|---:|"]
    for r in records:
        c=r["controls"]
        lines.append(f"| {r['task']} | {100*r['score']:.2f}% | {100*c['frozen_visual']['score']:.2f}% | {100*c['question_context']['score']:.2f}% | {100*c['old_mlp']['score']:.2f}% | {100*c['dense']['score']:.2f}% |")
    write_text(root/"report.md","\n".join(lines+["","OCRBench percentage is native points / 1000. AI2D is strict matching.","",*p["limits"]])+"\n")


def pipeline(root,gpu):
    p=validate(root);manifest=read(root/"manifest.json")
    def stage(name,status="running"):
        manifest.update(stage=name,status=status,gpu=gpu,updated_utc=context.native.now());write_json(root/"manifest.json",manifest)
    def idle():
        while available_gpus([gpu])!=[gpu]:stage("waiting_for_gpu","queued");time.sleep(20)
    def evaluate(method,phase,task):
        folder=root/f"eval/{phase}/{method}/{task}";marker=folder/"complete.json"
        if marker.exists():
            saved=read(marker)
            if saved["source"]!=p["source"] or file_sha(saved["config"])!=saved["config_sha256"]:raise ValueError("Evaluation provenance changed")
            check(root,p,Path(saved["run"]),method,phase,task);return
        idle();validate(root);stage(f"{phase}:{method}:{task}")
        run=folder/f"attempt_{time.time_ns()}";filename=root/"configs"/f"{phase}_{method}_{task}_{time.time_ns()}.yaml"
        write_text(filename,yaml.safe_dump(config(root,p,method,phase,task,run),sort_keys=False))
        context.native.execute([sys.executable,"-m","mllm_comp.cli.main","eval","--config",str(filename)],
            folder/f"launch_{time.time_ns()}.log",[gpu],{"MLLM_TOPK_PERFORMANCE":str(run/"request_performance.jsonl"),
                "HF_HOME":str(root/"cache/huggingface"),"HF_DATASETS_CACHE":str(root/"cache/datasets"),"XDG_CACHE_HOME":str(root/"cache/xdg")})
        checks=check(root,p,run,method,phase,task)
        write_json(marker,{"source":p["source"],"run":str(run),"config":str(filename),"config_sha256":file_sha(filename),
            "requests":checks["requests"],"completed_utc":context.native.now()})
        if phase=="full":
            manifest["completed_cells"]=sorted(set(manifest["completed_cells"])|{task});report(root,p);write_json(root/"manifest.json",manifest)
    try:
        for method in ("visual_only","initial"):
            for task in TASKS:evaluate(method,"smoke",task)
        idle();stage("real7b_training_restore_gates")
        if not (root/"gates_passed.json").exists():context.native.gates(root,[gpu])
        idle();stage("training_320_steps")
        if not (root/"training/manifest.json").exists() or read(root/"training/manifest.json")["status"]!="complete":
            context.native.execute(context.native.training_command(root,root/"training",[gpu],
                resume=(root/"training/latest.json").exists()),root/f"train_launch_{time.time_ns()}.log",[gpu])
        for task in TASKS:evaluate("trained","smoke",task)
        write_json(root/"smoke_complete.json",{"status":"complete","requests":72,"time":context.native.now()})
        for task in TASKS:evaluate("trained","full",task)
        stage("complete");manifest.update(status="complete",finished_utc=context.native.now());write_json(root/"manifest.json",manifest)
        write_json(root/"full_complete.json",{"status":"complete","requests":6588,"semantic_evidence_status":"pending_review","time":context.native.now()})
    except BaseException as exc:
        stage("failed","failed");manifest["error"]=str(exc);write_json(root/"manifest.json",manifest)
        write_json(root/f"failure_{time.time_ns()}.json",{"traceback":traceback.format_exc()});raise


if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("command",choices=("prepare","pipeline"))
    parser.add_argument("--root",type=Path,required=True);parser.add_argument("--reference",type=Path)
    parser.add_argument("--assignment",type=Path);parser.add_argument("--evidence",type=Path)
    parser.add_argument("--gpu",type=int,choices=(1,),default=1)
    args=parser.parse_args();root=owned_output(args.root)
    if args.command=="prepare":
        if args.reference is None or args.assignment is None:parser.error("--reference and --assignment are required")
        existed=root.exists()
        try:
            prepare(root,args.reference.resolve(),args.assignment,args.evidence)
        except BaseException:
            if not existed and root.exists():
                write_json(root/"preparation_failure.json",{"status":"failed","stage":"preparation",
                    "time":context.native.now(),"traceback":traceback.format_exc()})
            raise
    else:
        with (root/"runner.lock").open("a") as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);pipeline(root,args.gpu)
