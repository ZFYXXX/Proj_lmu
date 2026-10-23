"""S1/S2/M1/M2/SP1/SP2 on MMAU-mini: GPU collection then CPU probing."""
from __future__ import annotations
import argparse
from collections import Counter
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
from .data import ROOT, TASK, DOMAINS, inventory, load_records, read_annotations, stage_names, write_json, validate_speech
from .labels import event_label

def build_parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint",default=os.environ.get("CHECKPOINT_PATH"))
    p.add_argument("--output-dir",default=os.environ.get("ANALYSIS_RUN_DIR"))
    p.add_argument("--mini-json",default=os.environ.get("MMAU_MINI_JSON",str(
        ROOT.parents[1]/"AudioMCQevaluator/mmau-test-mini.json")))
    p.add_argument("--audio-root",default=os.environ.get("MMAU_MINI_AUDIO_ROOT"))
    p.add_argument("--gpus",default=os.environ.get("CUDA_VISIBLE_DEVICES","0"))
    p.add_argument("--experiments",default="S1,S2,M1,M2,SP1,SP2")
    p.add_argument("--event-labels",help="Reviewed or automatically generated S1 annotation JSONL")
    p.add_argument("--speech-labels",help="Phone/word interval JSONL from prepare_speech")
    p.add_argument("--layers",default="all")
    p.add_argument("--attention-backend",choices=("eager","flash_attention_2"),default="flash_attention_2")
    p.add_argument("--interventions",choices=("all","none"),default="all")
    p.add_argument("--keep-ratios",default="1,0.5,0.25",
                   help="Offline token retention; does not change native QA runtime")
    p.add_argument("--compression-method",choices=("uniform_drop","mean_pool"),default="uniform_drop")
    p.add_argument("--folds",type=int,default=5)
    p.add_argument("--alpha",type=float,default=1.0,help="Fixed ridge regularization; no test-based tuning")
    p.add_argument("--max-train-frames",type=int,default=256,
                   help="Evenly spaced valid frames per training clip; evaluation uses all valid frames")
    p.add_argument("--mlp-hidden",type=int,default=128,help="S1 MLP fixed hidden width at every layer")
    p.add_argument("--seed",type=int,default=42)
    p.add_argument("--limit",type=int,help="First N selected records; smoke only")
    p.add_argument("--dry-run",action="store_true")
    return p

def prepare_plan(args):
    manifest = Path(args.mini_json).expanduser().resolve()
    audio_root = Path(args.audio_root).expanduser().resolve() if args.audio_root else manifest.parent
    records = load_records(manifest,audio_root)
    experiments = list(dict.fromkeys(x.strip().upper() for x in args.experiments.split(",")))
    if not experiments or any(x not in DOMAINS for x in experiments):
        raise ValueError(f"--experiments must select from {list(DOMAINS)}")
    selected = [r for r in records if r["task"] in {DOMAINS[x] for x in experiments}]
    if args.limit is not None:
        if not 1 <= args.limit <= len(selected):
            raise ValueError("Invalid --limit")
        selected = selected[:args.limit]
    gpus = [v.strip() for v in args.gpus.split(",") if v.strip()]
    if not gpus or len(set(gpus)) != len(gpus) or len(gpus)>len(selected):
        raise ValueError("Distinct GPUs and at least one sample per GPU required")
    ratios = sorted(set(float(x) for x in args.keep_ratios.split(",")),reverse=True)
    if not ratios or ratios[0] != 1 or any(not math.isfinite(x) or not 0<x<=1 for x in ratios):
        raise ValueError("--keep-ratios must include 1 and lie in (0, 1]")
    if args.folds<2 or args.max_train_frames<1 or args.mlp_hidden<1 or not math.isfinite(args.alpha) or args.alpha<=0:
        raise ValueError("Invalid probe configuration")
    event_annotations, speech_annotations = read_annotations(args.event_labels), read_annotations(args.speech_labels)
    known = {r["id"]:r["task"] for r in records}
    for collection,domain in ((event_annotations,"sound"),(speech_annotations,"speech")):
        if any(known.get(key)!=domain for key in collection):
            raise ValueError(f"Annotation ID is not an MMAU-mini {domain} record")
    for row in selected:
        if row["task"]=="sound":
            row["event_annotation"] = event_label(row,event_annotations.get(row["id"]))
        if row["task"]=="speech":
            annotation = speech_annotations.get(row["id"],{})
            validate_speech({"id":row["id"],**annotation})
            row["speech_annotation"] = annotation
    speech = [r for r in selected if r["task"]=="speech"]
    coverage = {"event_labels":dict(Counter(r["event_annotation"]["status"] for r in selected if r["task"]=="sound")),
                "speech_clips":len(speech),"aligned_speech_clips":sum(bool(r["speech_annotation"].get("phones")) for r in speech)}
    if "SP1" in experiments and speech and not coverage["aligned_speech_clips"] and not args.dry_run:
        raise ValueError("SP1 needs --speech-labels. Run prepare_speech, or explicitly select experiments without SP1.")
    checkpoint,depth,layers = None,None,None
    if args.checkpoint:
        checkpoint = Path(args.checkpoint).expanduser().resolve()
        cfg = json.loads((checkpoint/"config.json").read_text())
        depth = int(cfg["thinker_config"]["audio_config"]["encoder_layers"])
        layers = list(range(depth)) if args.layers=="all" else sorted(set(map(int,args.layers.split(","))))
        if not layers or not set(layers).issubset(range(depth)):
            raise ValueError("Invalid audio encoder layers")
    elif not args.dry_run:
        raise ValueError("--checkpoint is required")
    if not args.output_dir and not args.dry_run:
        raise ValueError("--output-dir must name a new directory")
    output = str(Path(args.output_dir).expanduser().resolve()) if args.output_dir else None
    plan = {"schema":"intrinsic-audio-properties-v1","task":TASK,"mini_json":str(manifest),
            "audio_root":str(audio_root),"checkpoint":str(checkpoint) if checkpoint else None,
            "output_dir":output,"gpus":gpus,"shard_count":len(gpus),"seed":args.seed,
            "layers":layers,"encoder_layers":depth,"stages":stage_names(layers) if layers is not None else None,
            "attention_backend":args.attention_backend,"sample_count":len(selected),
            "full_test_mini":len(selected)==1000,"inventory":inventory(selected),
            "experiments":experiments,"interventions":args.interventions=="all","label_coverage":coverage,
            "probe_config":{"folds":args.folds,"alpha":args.alpha,"max_train_frames":args.max_train_frames,
                            "mlp_hidden":args.mlp_hidden,"seed":args.seed,"keep_ratios":ratios,
                            "compression_method":args.compression_method},
            "interpretation":{"labels":"weak teachers, not new human ground truth",
              "validation":"within-mini audio-group CV; not an external benchmark",
              "compression":"offline native-token reduction with a frozen clean-trained probe; no compressed QA or speedup claim",
              "qa_delta":"paired original-reference accuracy; changed questions may require new answers",
              "SP1":"requires real phoneme intervals; missing clips explicitly excluded"}}
    return plan,selected

def run_workers(plan):
    output = Path(plan["output_dir"])
    workers = []
    try:
        for rank,gpu in enumerate(plan["gpus"]):
            shard = output/f"shard-{rank:03d}"
            shard.mkdir()
            env = {**os.environ,"CUDA_VISIBLE_DEVICES":gpu,"PYTHONUNBUFFERED":"1"}
            env["PYTHONPATH"] = os.pathsep.join(filter(None,[str(ROOT),str(
                ROOT/"third_party/frameworks/lmms-eval"),env.get("PYTHONPATH")]))
            with (shard/"worker.log").open("w") as log:
                workers.append(subprocess.Popen([sys.executable,"-m","scripts.analysis.audio_category.worker",
                    "--run-dir",str(output),"--rank",str(rank)],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT))
            print(f"GPU {gpu}: {shard/'worker.log'}",flush=True)
        while any(w.poll() is None for w in workers):
            if any(w.poll() not in (None,0) for w in workers):
                raise RuntimeError("A GPU worker failed; inspect worker.log")
            time.sleep(1)
        if any(w.returncode!=0 for w in workers):
            raise RuntimeError("A GPU worker failed; inspect worker.log")
    finally:
        for w in workers:
            if w.poll() is None:
                w.terminate()
        for w in workers:
            try:
                w.wait(timeout=10)
            except subprocess.TimeoutExpired:
                w.kill()
                w.wait()

def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        plan,selected = prepare_plan(args)
    except (OSError,KeyError,ValueError) as exc:
        parser.error(str(exc))
    print(json.dumps(plan,ensure_ascii=False,indent=2),flush=True)
    if args.dry_run:
        return 0
    from .report import build_report
    import librosa, pyworld, soundfile, torch, transformers, pydantic
    if not torch.cuda.is_available():
        parser.error("Collection requires CUDA; labels and report can run on CPU")
    if transformers.__version__!="4.57.3":
        parser.error("Native time maps require Transformers 4.57.3")
    output = Path(plan["output_dir"])
    output.mkdir(parents=True,exist_ok=False)
    write_json(output/"plan.json",plan)
    write_json(output/"selected.json",selected)
    run_workers(plan)
    build_report(output)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
