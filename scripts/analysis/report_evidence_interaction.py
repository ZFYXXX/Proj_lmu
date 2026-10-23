"""Report complete image-level evidence; never test partial confirmation data."""

import argparse
from collections import defaultdict
import json
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from mllm_comp.analysis.common.evidence_interaction import bootstrap, condition_id, holm
from mllm_comp.core.serialization import atomic_write_json


def report(root: Path, stage: str):
    cohort=json.loads((root/"cohort.json").read_text())
    summaries=[json.loads(p.read_text()) for p in sorted((root/"samples"/stage).glob("*/summary.json"))]
    expected=16 if stage=="smoke" else 200 if stage=="development" else 800 if stage=="confirmation" else sum(r.get(stage,False) for r in cohort["entries"])
    result={"stage":stage,"completed_images":len(summaries),"expected_images":expected,
            "status":"complete" if len(summaries)==expected else "incomplete",
            "inference_permitted":stage=="confirmation" and len(summaries)==800}
    if len(summaries)!=expected:
        atomic_write_json(root/f"{stage}_report.json",result,ensure_ascii=False)
        return result
    keys=[s["image_fingerprint"] for s in summaries]
    if len(set(keys))!=len(keys):raise ValueError("Duplicate independent image in stage.")
    expected_ids=set()
    for entry in cohort["entries"]:
        include=entry.get("smoke",False) if stage=="smoke" else entry["split"]==stage if stage in {"development","confirmation"} else entry.get(stage,False)
        if include:expected_ids.add(f"{entry['task']}:{entry['second_doc_id']}" if stage=="question_review" else entry["sample_id"])
    if {s["sample_id"] for s in summaries}!=expected_ids:
        raise ValueError("Completed images do not exactly match the frozen stage manifest.")
    code={s["implementation_hash"] for s in summaries}
    if len(code)!=1:raise ValueError("Mixed implementations; confirmation invalid.")
    by_task=defaultdict(list)
    references=defaultdict(lambda:defaultdict(list))
    strict_references=defaultdict(lambda:defaultdict(list))
    for s in summaries:
        pairs=json.loads(Path(s["interactions_path"]).read_text())
        conditions=[json.loads(line) for line in Path(s["condition_path"]).read_text().splitlines()]
        condition_lookup={r["request_id"]:r for r in conditions}
        if "chartqa" in s["task"]:
            for pair in pairs:
                values=[condition_lookup[f"{pair['budget']}/{pair['seed']}/{condition_id(slots)}"]["strict_answer_match"]
                        for slots in [(),(pair["a"],),(pair["b"],),(pair["a"],pair["b"])]]
                pair["strict_score_interaction"]=values[3]-values[1]-values[2]+values[0]
        if len(conditions)!=s["condition_count"]:raise ValueError("Incomplete image ledger.")
        ref={r["request_id"]:r for r in conditions if r["request_id"].startswith("reference/")}
        for name,row in ref.items():
            references[s["task"]][name].append(row["primary_task_score"])
            if row["strict_answer_match"] is not None:strict_references[s["task"]][name].append(row["strict_answer_match"])
        for budget in sorted({p["budget"] for p in pairs}):
            group={kind:[p for p in pairs if p["budget"]==budget and p["kind"]==kind] for kind in ["question","random","mixed"]}
            if any(len(v)!=3*({"question":6,"random":6,"mixed":16}[k]) for k,v in group.items()):
                raise ValueError("Missing seeds/pairs; image is not a complete experimental unit.")
            image={"sample_id":s["sample_id"],"image_fingerprint":s["image_fingerprint"],"budget":budget,
                   "baseline_score":ref["reference/baseline"]["primary_task_score"],
                   "baseline_score_group":"zero" if ref["reference/baseline"]["primary_task_score"]==0 else "full" if ref["reference/baseline"]["primary_task_score"]==1 else "partial"}
            if "reference/feature_anchor_mean_k128" in ref:
                image["mean_compression_maintains_score"]=ref["reference/feature_anchor_mean_k128"]["primary_task_score"]>=image["baseline_score"]
            for kind,values in group.items():
                for metric in ["score_interaction","logprob_interaction","joint_only_improvement"]:
                    image[f"{kind}_{metric}"]=statistics.mean(float(p[metric]) for p in values)
                if "chartqa" in s["task"]:
                    image[f"{kind}_strict_score_interaction"]=statistics.mean(p["strict_score_interaction"] for p in values)
            image["question_minus_random"]=image["question_score_interaction"]-image["random_score_interaction"]
            by_task[(s["task"],budget)].append(image)
    tests=[]
    tables=[]
    for (task,budget),images in sorted(by_task.items()):
        table={"task":task,"budget":budget,"n_images":len(images),"metrics":{},"subgroups":{}}
        for name in ["question_score_interaction","random_score_interaction","mixed_score_interaction","question_minus_random",
                     "question_joint_only_improvement","random_joint_only_improvement","question_logprob_interaction","random_logprob_interaction"]:
            table["metrics"][name]=bootstrap([r[name] for r in images])
        for key in ["mean_compression_maintains_score"]:
            for flag in [False,True]:
                subset=[r for r in images if r.get(key) is flag]
                if subset:table["subgroups"][f"{key}={flag}"]={"n_images":len(subset),"question_minus_random":bootstrap([r["question_minus_random"] for r in subset])}
        for label in ["zero","partial","full"]:
            subset=[r for r in images if r["baseline_score_group"]==label]
            if subset:table["subgroups"][f"baseline_score={label}"]={"n_images":len(subset),"question_minus_random":bootstrap([r["question_minus_random"] for r in subset])}
        if "chartqa" in task:
            for kind in ["question","random","mixed"]:
                key=f"{kind}_strict_score_interaction"
                table["metrics"][key]=bootstrap([r[key] for r in images])
        if result["inference_permitted"] and budget==128:
            for name in ["question_score_interaction","question_minus_random"]:
                tests.append({"task":task,"endpoint":name,**table["metrics"][name]})
        tables.append(table)
    if tests:
        if len(tests)!=4:raise ValueError("Exactly four preregistered confirmation tests required.")
        for test,p in zip(tests,holm([t["one_sided_p"] for t in tests])):
            test["holm_adjusted_p"]=p
            test["supported"]=p<.05 and test["ci95"][0]>0
    result.update({"tables":tables,"confirmatory_tests":tests,
                   "references":{task:{name:{"mean_score":statistics.mean(v),"n_images":len(v),
                                              "paired_delta_vs_baseline":bootstrap([a-b for a,b in zip(v,arms["reference/baseline"])]),
                                              "strict_answer_match":statistics.mean(strict_references[task][name]) if strict_references[task][name] else None}
                                         for name,v in arms.items()} for task,arms in references.items()},
                   "elapsed_s_median":statistics.median(s["elapsed_s"] for s in summaries),
                   "elapsed_s_sum":sum(s["elapsed_s"] for s in summaries),
                   "interpretation":"Region replacement interactions are conditional on the selected background and controls, not proof of semantic binding.",
                   "implementation_hash":next(iter(code))})
    if tests:
        supported={task:all(t["supported"] for t in tests if t["task"]==task) for task in cohort["counts"]}
        result["decision"]="supported_across_both_tasks" if all(supported.values()) else "task_specific" if any(supported.values()) else "insufficient_confirmatory_evidence"
    if stage in {"question_review","budget_review"}:
        if stage=="question_review":
            for s in summaries:
                original_folder=root/"samples"/"confirmation"/s["primary_sample_id"].replace(":","_")
                review_folder=Path(s["condition_path"]).parent
                for seed in [11,23,47]:
                    before=json.loads((original_folder/f"plan_128_{seed}.json").read_text())
                    after=json.loads((review_folder/f"plan_128_{seed}.json").read_text())
                    if before!=after:raise ValueError("Question review changed regions, controls, or background.")
        paired=[]
        for task in cohort["counts"]:
            for budget in ([128] if stage=="question_review" else [64,256]):
                effects=[]
                for s in summaries:
                    if s["task"]!=task:continue
                    primary=root/"samples"/"confirmation"/s["primary_sample_id"].replace(":","_")/"interactions.json"
                    original=json.loads(primary.read_text())
                    review=json.loads(Path(s["interactions_path"]).read_text())
                    a=[p["score_interaction"] for p in original if p["kind"]=="question" and p["budget"]==128]
                    b=[p["score_interaction"] for p in review if p["kind"]=="question" and p["budget"]==budget]
                    if a and b:effects.append(statistics.mean(b)-statistics.mean(a))
                if effects:paired.append({"task":task,"budget":budget,"question_interaction_delta_vs_primary":bootstrap(effects)})
        result["paired_secondary_reviews"]=paired
        result["not_applicable"]=[{"sample_id":s["sample_id"],**x} for s in summaries for x in s["not_applicable"]]
    atomic_write_json(root/f"{stage}_report.json",result,ensure_ascii=False)
    lines=[f"# Evidence interaction: {stage}","",f"Independent images: {len(summaries)}/{expected}",
           f"Formal confirmation inference: {result['inference_permitted']}","",
           "| Task | K | Images | Question interaction | Random interaction | Difference, 95% CI |","|---|---:|---:|---:|---:|---|"]
    for t in tables:
        m=t["metrics"]; d=m["question_minus_random"]
        lines.append(f"| {t['task']} | {t['budget']} | {t['n_images']} | {m['question_score_interaction']['mean']:.5f} | {m['random_score_interaction']['mean']:.5f} | {d['mean']:.5f} [{d['ci95'][0]:.5f}, {d['ci95'][1]:.5f}] |")
    lines.extend(["",f"Decision: {result.get('decision','Exploratory / secondary results only')}","",result["interpretation"],
                  "", "The selector uses a dense Layer-0 prompt pass; this diagnostic is not a deployable speedup claim."])
    lines.extend(["","## Compression references","","| Task | Method | Native score | Strict match | Delta vs dense, 95% CI |","|---|---|---:|---:|---|"])
    for task,arms in result["references"].items():
        for method,values in arms.items():
            d=values["paired_delta_vs_baseline"]
            strict="—" if values["strict_answer_match"] is None else f"{values['strict_answer_match']:.4f}"
            lines.append(f"| {task} | {method} | {values['mean_score']:.4f} | {strict} | {d['mean']:.4f} [{d['ci95'][0]:.4f}, {d['ci95'][1]:.4f}] |")
    (root/f"{stage}_report.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    atomic_write_json(root/f"{stage}_image_metrics.json",{f"{task}:{budget}":rows for (task,budget),rows in by_task.items()},ensure_ascii=False)
    case_index=[]
    for summary in summaries:
        folder=Path(summary["condition_path"]).parent
        pairs=json.loads(Path(summary["interactions_path"]).read_text())
        case_index.append({"sample_id":summary["sample_id"],"image_fingerprint":summary["image_fingerprint"],
                           "question":summary["question"],"regions_png":str(folder/"regions.png"),
                           "condition_path":summary["condition_path"],"interactions_path":summary["interactions_path"],
                           "largest_observed_interactions":sorted(pairs,key=lambda p:-p["score_interaction"])[:3],
                           "case_ranking_uses_outcomes":True,"case_ranking_is_explanatory_only":True})
    atomic_write_json(root/f"{stage}_case_index.json",case_index,ensure_ascii=False)
    return result


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--root",type=Path,required=True);p.add_argument("--stage",required=True)
    a=p.parse_args();r=report(a.root.resolve(),a.stage)
    print(json.dumps({k:v for k,v in r.items() if k in {"stage","status","completed_images","expected_images","decision"}}))
