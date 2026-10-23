"""CPU reconstruction of six intrinsic-property experiments and intervention reports."""
from __future__ import annotations
import argparse
from collections import defaultdict
import csv
import json
import importlib.metadata
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_limits
from .data import load_run, write_json
from .probes import task_specs,evaluate

PROSODY_SKILLS = {"Dissonant Emotion Interpretation","Phonemic Stress Pattern Analysis",
                  "Emotion State summarisation","Emotion Flip Detection"}

def write_csv(path,rows):
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with Path(path).open("w",newline="",encoding="utf-8") as handle:
        writer = csv.DictWriter(handle,fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

def aggregate_metrics(rows,keys,metrics):
    groups = defaultdict(list)
    for row in rows:
        groups[tuple(row[k] for k in keys)].append(row)
    result = []
    for key,members in groups.items():
        item = {**dict(zip(keys,key)),"n_clips":len(members)}
        for metric in metrics:
            values = [r[metric] for r in members if r.get(metric) is not None]
            item[metric] = float(np.mean(values)) if values else None
            item[metric+"_n"] = len(values)
        result.append(item)
    return result

def paired_interval(values,seed):
    values = np.asarray(values)
    if not len(values):
        return [None,None]
    rng = np.random.default_rng(seed)
    # Fixed paired clips, not separate resampling of clean/intervened populations.
    estimates = [values[rng.integers(0,len(values),len(values))].mean() for _ in range(1000)]
    return np.quantile(estimates,[.025,.975]).tolist()

def qa_report(rows,seed):
    groups = defaultdict(list)
    for row in rows:
        clean = row["accuracy"]
        for variant in row["variants"]:
            key = (row["task"],variant["name"],"all")
            groups[key].append((row,variant,clean))
            if row["task"]=="speech" and row["sub-category"] in PROSODY_SKILLS:
                groups[(row["task"],variant["name"],"prosody_skills")].append((row,variant,clean))
            if variant["name"]=="pitch_flatten":
                control = next((v for v in row["variants"] if v["name"]=="world_resynthesis"),None)
                if control is not None:
                    groups[(row["task"],variant["name"],"vs_world_resynthesis")].append(
                        (row,variant,control["accuracy"]))
    result,drift,answers = [],[],[]
    for (domain,name,subset),pairs in groups.items():
        delta = [v["accuracy"]-base for _,v,base in pairs]
        lo,hi = paired_interval(delta,seed)
        result.append({"domain":domain,"variant":name,"subset":subset,"n":len(pairs),
                       "reference_accuracy":float(np.mean([base for _,_,base in pairs])),
                       "intervened_original_reference_accuracy":float(np.mean([v["accuracy"] for _,v,_ in pairs])),
                       "delta_accuracy":float(np.mean(delta)),"ci95_low":lo,"ci95_high":hi,
                       "interpretation":"original reference reused; answer invariance is not verified"})
    for row in rows:
        answers.append({"id":row["id"],"domain":row["task"],"variant":"clean",
                        "prediction":row["prediction"],"accuracy":row["accuracy"]})
        for v in row["variants"]:
            answers.append({"id":row["id"],"domain":row["task"],"variant":v["name"],
                            "prediction":v["prediction"],"accuracy":v["accuracy"]})
            for stage,values in v["drift"].items():
                drift.append({"id":row["id"],"domain":row["task"],"variant":v["name"],
                              "stage":stage,**values})
    return result,drift,answers

def load_stage(rows,stage):
    records = []
    for row in rows:
        directory = Path(row["_directory"])
        with np.load(directory/"clean.npz",allow_pickle=False) as features:
            required = [f"{stage}__{suffix}" for suffix in ("x","t","clip")]
            if any(k not in features for k in required):
                raise ValueError(f"Missing stage {stage}: {row['id']}")
            x,t,clip = (features[k] for k in required)
            if x.ndim!=2 or len(x)!=len(t) or len(t)<1 or not np.isfinite(x).all():
                raise ValueError("Invalid native token features")
            if np.any(np.diff(t)<=0):
                raise ValueError("Token time coordinates must increase")
        with np.load(directory/"targets.npz",allow_pickle=False) as targets:
            targets = dict(targets)
        variants = {}
        for v in row["variants"]:
            with np.load(directory/f"{v['name']}.npz",allow_pickle=False) as f:
                variants[v["name"]] = f[stage]
        records.append({"row":row,"x":x.astype(np.float32),"t":t,"clip":clip,
                        "targets":targets,"labels":json.loads((directory/"labels.json").read_text()),
                        "variant_clips":variants})
    return records

def representation_periodicity(records,stage):
    """Centered embedding autocorrelation at weak beat periods, with physical lags."""
    from .observer import interpolate
    result = []
    for record in records:
        bpm = record["labels"].get("bpm")
        if bpm is None or bpm<=0:
            continue
        t = record["t"]
        x = record["x"]-record["x"].mean(0)
        for multiple in (.5,1.0,2.0):
            lag = multiple*60/bpm
            valid = t+lag<=t[-1]
            if valid.sum()<2:
                continue
            a = x[valid]
            b = interpolate(x,t,t[valid]+lag)
            denominator = float(np.sqrt(np.sum(a*a)*np.sum(b*b)))
            result.append({"id":record["row"]["id"],"stage":stage,"weak_bpm":bpm,
                           "beat_period_multiple":multiple,"lag_seconds":lag,
                           "embedding_autocorrelation":float(np.sum(a*b)/denominator)
                           if denominator>1e-12 else None})
    return result

def plot_results(output,summary,stages,qa):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    primary = {"S1_event":"accuracy","S2_onset":"event_f1","M1_chroma":"chroma_correlation",
               "M1_pitch_class":"accuracy","M1_f0":"r2","M1_key":"accuracy","M1_chroma_clip":"chroma_correlation",
               "M2_beat":"event_f1","M2_bpm":"tempo_accuracy_4pct","M2_periodicity":"r2",
               "SP1_phoneme":"per","SP2_f0":"r2","SP2_energy":"r2","SP2_rate":"r2","SP2_stress":"accuracy"}
    for experiment in summary["experiments"]:
        specs = [s for s in summary["specs"] if s["experiment"]==experiment]
        fig,axes = plt.subplots(len(specs),1,figsize=(11,max(3.5,len(specs)*3.1)),squeeze=False,layout="constrained")
        for ax,spec in zip(axes[:,0],specs):
            name = spec["name"]
            entries = summary["probes"].get(name,{})
            found = False
            for head in ("linear","mlp"):
                for ratio in summary["probe_config"]["keep_ratios"]:
                    values = []
                    for stage in stages:
                        metric = entries.get(stage,{}).get(head,{}).get("ratios",{}).get(str(ratio),{})
                        values.append(metric.get(primary[name]))
                    if any(v is not None for v in values):
                        ax.plot(range(len(stages)),[np.nan if v is None else v for v in values],
                                label=f"{head}, keep={ratio:g}")
                        found = True
            ax.set_title(name+" (weak targets)")
            ax.set_ylabel(primary[name])
            positions = sorted(set([0,len(stages)-1,*range(0,len(stages),max(1,len(stages)//10))]))
            ax.set_xticks(positions,[stages[i] for i in positions],rotation=35,ha="right")
            ax.grid(axis="y",alpha=.2)
            if found:
                ax.legend(fontsize=8)
            else:
                ax.text(.5,.5,"Insufficient valid labels / clips",ha="center",transform=ax.transAxes)
        fig.savefig(output/f"{experiment}_layers.png",dpi=160)
        fig.savefig(output/f"{experiment}_layers.pdf")
        plt.close(fig)
    if qa:
        selected = [q for q in qa if q["subset"]=="all"]
        fig,ax = plt.subplots(figsize=(10,max(4,.35*len(selected))),layout="constrained")
        values = np.array([q["delta_accuracy"]*100 for q in selected])
        lo = np.array([q["ci95_low"]*100 for q in selected])
        hi = np.array([q["ci95_high"]*100 for q in selected])
        ax.errorbar(values,np.arange(len(values)),xerr=np.maximum(0,np.stack([values-lo,hi-values])),fmt="o")
        ax.set_yticks(range(len(values)),[q["domain"]+": "+q["variant"] for q in selected])
        ax.axvline(0,color="grey",linewidth=1)
        ax.set_xlabel("Paired original-reference QA delta (percentage points); labels not re-annotated")
        fig.savefig(output/"qa_interventions.png",dpi=160)
        fig.savefig(output/"qa_interventions.pdf")
        plt.close(fig)

def build_report(run_dir):
    run = Path(run_dir)
    plan,rows = load_run(run)
    (run/"complete.json").unlink(missing_ok=True)
    output = run/"report"
    output.mkdir(exist_ok=True)
    figures = output/"figures"
    figures.mkdir(exist_ok=True)
    specs = task_specs(plan["experiments"])
    summary = {"schema":plan["schema"],"experiments":plan["experiments"],"specs":specs,
               "probe_config":plan["probe_config"],"n":len(rows),"probes":{},
               "versions":{name:importlib.metadata.version(name) for name in
                           ("numpy","scipy","scikit-learn","matplotlib")},
               "interpretation":plan["interpretation"]}
    metrics,interventions,folds,periodicity = [],[],{},[]
    with (output/"probe_predictions.jsonl").open("w") as predictions, threadpool_limits(limits=1):
        for domain in ("sound","music","speech"):
            selected = [r for r in rows if r["task"]==domain]
            if not selected:
                continue
            current = [s for s in specs if s["domain"]==domain]
            for stage in plan["stages"]:
                records = load_stage(selected,stage)
                if domain=="music" and "M2" in plan["experiments"]:
                    periodicity.extend(representation_periodicity(records,stage))
                for spec in current:
                    for head in (("linear","mlp") if spec["name"]=="S1_event" else ("linear",)):
                        print(f"Probe {spec['name']} / {stage} / {head}",flush=True)
                        result,pred,changes,split = evaluate(spec,records,stage,plan["probe_config"],head)
                        summary["probes"].setdefault(spec["name"],{}).setdefault(stage,{})[head] = result
                        folds[f"{spec['name']}/{stage}/{head}"] = split
                        for record in pred:
                            predictions.write(json.dumps({"probe":spec["name"],"stage":stage,"head":head,
                                                          **record},ensure_ascii=False,allow_nan=False)+"\n")
                        for record in changes:
                            interventions.append({"probe":spec["name"],"stage":stage,"head":head,**record})
                        for ratio,values in result.get("ratios",{}).items():
                            for metric,value in values.items():
                                if isinstance(value,(int,float)) or value is None:
                                    metrics.append({"experiment":spec["experiment"],"probe":spec["name"],
                                        "stage":stage,"head":head,"keep_ratio":ratio,"metric":metric,"value":value,
                                        "delta_from_clean":values["delta_from_clean"].get(metric),
                                        "n_clips":values["n_clips"]})
                        predictions.flush()
                del records
    qa,drift,answers = qa_report(rows,plan["seed"])
    intervention_summary = aggregate_metrics(interventions,("probe","stage","head","variant"),
        ("consistent","accuracy","chroma_equivariance_correlation","chroma_invariance_correlation"))
    drift_summary = aggregate_metrics(drift,("domain","variant","stage"),
        ("clip_cosine_distance","time_aligned_cosine_distance"))
    coverage = []
    for row in rows:
        labels = json.loads((Path(row["_directory"])/"labels.json").read_text())
        coverage.append({"id":row["id"],"domain":row["task"],"duration":row["duration"],
                         "event_annotation":labels.get("event_annotation"),
                         "aligned_phones":len(labels.get("speech_alignment",{}).get("phones",[])),
                         "skipped_interventions":row["skipped_interventions"]})
    summary.update(label_coverage=coverage,qa_interventions=qa,probe_interventions=intervention_summary)
    write_json(output/"summary.json",summary)
    write_json(output/"folds.json",folds)
    write_csv(output/"probe_metrics.csv",metrics)
    write_csv(output/"probe_interventions.csv",interventions)
    write_csv(output/"probe_intervention_summary.csv",intervention_summary)
    write_csv(output/"qa_interventions.csv",qa)
    write_csv(output/"representation_drift.csv",drift)
    write_csv(output/"representation_drift_summary.csv",drift_summary)
    write_csv(output/"music_periodicity.csv",periodicity)
    write_json(output/"qa_answers.json",answers)
    plot_results(figures,summary,plan["stages"],qa)
    statuses = {}
    for spec in specs:
        records = summary["probes"].get(spec["name"],{})
        complete = sum(v["status"]=="complete" for heads in records.values() for v in heads.values())
        statuses[spec["name"]] = "complete" if complete else "insufficient_data"
    write_json(output/"status.json",statuses)
    (output/"README.zh.md").write_text(
        "# MMAU 本质属性分析\n\n"
        "标签为 QA 词典、信号算法或 ASR/MFA 弱标签。缺失和排除见 summary.json。\n"
        "probe_metrics.csv 为音频组折外指标；probe_predictions.jsonl 保存逐帧/逐片段预测。\n"
        "keep_ratio<1 为离线原生 token 保留/池化，用 clean-trained probe 在原时间轴评估；不是压缩模型 QA。\n"
        "QA 干预沿用原答案，音高/时序等问题的答案可能改变；delta 不等价于推理能力损失。\n"
        "SP1 PER 为弱音素参考与逐帧分类折叠序列的编辑距离，不是 ASR WER。\n"
        "未完成协议见 status.json；小规模 smoke 不能作为正式实验结论。\n",
        encoding="utf-8")
    write_json(run/"complete.json",{"samples":len(rows),"report":str(output),"protocol_status":statuses,
                                   "all_probes_evaluable":all(v=="complete" for v in statuses.values())})
    return summary

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir",required=True)
    args = parser.parse_args()
    build_report(args.run_dir)

if __name__ == "__main__":
    main()
