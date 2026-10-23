"""GPU collection of native MMAU QA, token features and waveform interventions."""
from __future__ import annotations
import argparse
from copy import deepcopy
import importlib.metadata
import inspect
import json
import math
from pathlib import Path
import random
from .data import ROOT, TASK, write_json, stage_names

def load_task(plan):
    from lmms_eval.api.task import ConfigurableMessagesTask
    from lmms_eval.utils import load_yaml_config
    task_root = ROOT/"third_party/frameworks/lmms-eval/lmms_eval/tasks"
    raw = load_yaml_config(str(task_root/"mmaunew3"/f"{TASK}.yaml"))
    raw["dataset_path"] = str(task_root/"mmaunew2/mmaunew2_test_mini_dataset.py")
    raw["dataset_kwargs"]["data_files"] = {"test":plan["mini_json"]}
    raw["lmms_eval_specific_kwargs"]["default"]["audio_root"] = plan["audio_root"]
    return ConfigurableMessagesTask(config=raw, model_name="qwen2_5_omni")

def runtime_config(plan):
    from mllm_comp.config.schema import experiment_config_from_dict
    return experiment_config_from_dict({
        "experiment":{"name":"mmau_intrinsic_audio_properties","seed":plan["seed"]},
        "backbone":{"name":"qwen25_omni","model_path":plan["checkpoint"],"trust_remote_code":False,
                    "config":{"torch_dtype":"bfloat16","enable_talker":False,
                              "attn_implementation":plan["attention_backend"]}},
        "pipeline":{"stages":[]},
        "eval":{"backend":"lmms_eval","tasks":[TASK],"compression_mode":"none","runtime_batch_size":1},
        "output":{"run_dir":plan["output_dir"]}})

def feature_drift(clean, changed, stages, time_scale=1.0):
    import numpy as np
    from .observer import interpolate
    result = {}
    for stage in stages:
        a, b = clean[f"{stage}__clip"], changed[f"{stage}__clip"]
        cosine = float(np.dot(a,b)/max(np.linalg.norm(a)*np.linalg.norm(b), 1e-12))
        x = clean[f"{stage}__x"].astype(np.float32)
        z = interpolate(changed[f"{stage}__x"].astype(np.float32),
                        changed[f"{stage}__t"]*time_scale, clean[f"{stage}__t"])
        denom = np.maximum(np.linalg.norm(x,axis=1)*np.linalg.norm(z,axis=1),1e-12)
        result[stage] = {"clip_cosine_distance":float(np.clip(1-cosine,0,2)),
                         "time_aligned_cosine_distance":float(np.mean(np.clip(1-(x*z).sum(1)/denom,0,2)))}
    return result

def ensure_audio_capacity(extractor,duration):
    """Do not silently truncate the 66 mini clips >30 s, or stretched variants."""
    needed = math.ceil(duration*extractor.sampling_rate/extractor.hop_length)*extractor.hop_length
    if needed>extractor.n_samples:
        extractor.n_samples = needed
        extractor.chunk_length = needed/extractor.sampling_rate
        extractor.nb_max_frames = needed//extractor.hop_length

def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--rank", type=int, required=True)
    args = parser.parse_args(argv)
    run = Path(args.run_dir)
    plan = json.loads((run/"plan.json").read_text())
    selected = json.loads((run/"selected.json").read_text())
    assigned = {r["doc_index"]:r for r in selected[args.rank::plan["shard_count"]]}
    import numpy as np
    import soundfile as sf
    import torch
    import transformers
    from mllm_comp.runtime import load_backbone_runtime
    from mllm_comp.integrations.lmms_eval.model_bridge import LMMSEvalRequestBridge
    from .observer import CategoryObserver
    from .signal import read_audio, weak_targets, intervene, variant_specs, SAMPLE_RATE
    if transformers.__version__ != "4.57.3":
        raise RuntimeError("Native time maps require Transformers 4.57.3")
    torch.cuda.set_device(0)
    random.seed(plan["seed"])
    np.random.seed(plan["seed"])
    torch.manual_seed(plan["seed"])
    task = load_task(plan)
    facade, runtime = load_backbone_runtime(runtime_config(plan), device="cuda:0", train=False,
                                           compression_mode="none", enable_talker=False)
    runtime.model.eval().requires_grad_(False)
    tower = runtime.model.thinker.audio_tower
    if len(tower.layers) != plan["encoder_layers"]:
        raise ValueError("Checkpoint architecture differs from plan")
    extractor = runtime.processor.feature_extractor
    observer = CategoryObserver(tower, plan["layers"], sample_rate=extractor.sampling_rate,
                                hop_length=extractor.hop_length)
    bridge = LMMSEvalRequestBridge(facade=facade, runtime=runtime)
    shard = run/f"shard-{args.rank:03d}"
    write_json(shard/"runtime.json", {
        "versions":{name:importlib.metadata.version(name) for name in
                    ("torch","transformers","datasets","numpy","librosa","pyworld","qwen-omni-utils")},
        "gpu":torch.cuda.get_device_name(0), "audio_source":inspect.getfile(type(tower)),
        "sample_rate":extractor.sampling_rate, "hop_length":extractor.hop_length,
        "initial_audio_capacity_samples":extractor.n_samples,
        "audio_length_policy":"expand extractor capacity to cover each full waveform, including interventions",
        "generation":task.get_config("generation_kwargs"),
        "features":"native token vectors float16; clip mean float32; exact nominal convolution/pool centers",
        "qa_compression":"none; token-pooling sensitivity is measured offline, not a deployed model speedup"})

    def generate(doc, row, path, variant, duration):
        ensure_audio_capacity(extractor,duration)
        messages = deepcopy(task.doc_to_messages(doc))
        sample = bridge.build_chat_sample(raw_messages=messages, sample_id=f"{TASK}:{row['id']}:{variant}")
        audio = [item for msg in sample.conversation for item in msg.get("content",[])
                 if item.get("type")=="audio"]
        if len(audio)!=1 or str(Path(str(audio[0]["audio"])).resolve())!=row["audio_path"]:
            raise ValueError("Native prompt does not match selected audio")
        audio[0]["audio"] = str(path)
        observer.begin()
        with torch.inference_mode():
            prediction = bridge._generate_sample(
                sample, request_generation=task.get_config("generation_kwargs"),
                context=task.doc_to_text(doc), task=TASK, split=task.get_config("test_split"),
                doc_id=row["doc_index"], request_mode="chat", cache_hook=None)
        features = observer.finish()
        captured_seconds = float(features["C__t"][-1]+2*extractor.hop_length/extractor.sampling_rate)
        if captured_seconds<duration-.05:
            raise ValueError(f"Audio was truncated: captured {captured_seconds}s of {duration}s")
        score = task.process_results(doc,[prediction])["mmaunew3_accuracy"]
        if not score["has_ground_truth"]:
            raise ValueError("Missing native QA reference")
        return features, {"prediction":prediction,"accuracy":float(score["score"]),"score_details":score,
                          "input_duration_seconds":duration,"captured_duration_seconds":captured_seconds}

    completed = 0
    observer.install()
    try:
        with (shard/"samples.jsonl").open("x",encoding="utf-8") as handle:
            for index,doc in task.doc_iterator(rank=0,world_size=1,limit=1000):
                if index not in assigned:
                    continue
                row = assigned[index]
                if row["id"]!=doc["id"]:
                    raise ValueError("Native task ordering differs from manifest")
                directory = shard/f"{index:04d}"
                directory.mkdir()
                y = read_audio(row["audio_path"])
                targets, labels = weak_targets(y,row["task"],row.get("speech_annotation"))
                labels["event_annotation"] = row.get("event_annotation")
                np.savez_compressed(directory/"targets.npz",**targets)
                write_json(directory/"labels.json",labels)
                print(f"rank={args.rank} sample={index} task={row['task']} clean",flush=True)
                clean, qa = generate(doc,row,row["audio_path"],"clean",len(y)/SAMPLE_RATE)
                np.savez_compressed(directory/"clean.npz",**clean)
                variants, skipped = [], []
                specs = variant_specs(row["task"],plan["experiments"]) if plan["interventions"] else []
                for number,name in enumerate(specs):
                    changed, info = intervene(y,name,targets,plan["seed"]+index*101+number)
                    if changed is None:
                        skipped.append(info)
                        continue
                    path = directory/"intervention.wav"
                    # FLOAT preserves gain > 1 without clipping/PCM saturation.
                    sf.write(path,changed,SAMPLE_RATE,subtype="FLOAT")
                    print(f"rank={args.rank} sample={index} {name}",flush=True)
                    features, answer = generate(doc,row,path,name,len(changed)/SAMPLE_RATE)
                    drift = feature_drift(clean,features,plan["stages"],info["time_scale"])
                    np.savez_compressed(directory/f"{name}.npz",
                                        **{s:features[f"{s}__clip"] for s in plan["stages"]})
                    variants.append({**info,**answer,"drift":drift})
                    del features
                    path.unlink()
                result = {**row,**qa,"directory":directory.name,"variants":variants,
                          "skipped_interventions":skipped,"duration":labels["duration"]}
                handle.write(json.dumps(result,ensure_ascii=False,allow_nan=False)+"\n")
                handle.flush()
                completed += 1
    finally:
        observer.close()
    if completed != len(assigned):
        raise RuntimeError("Incomplete worker coverage")
    write_json(shard/"complete.json",{"samples":completed})

if __name__ == "__main__":
    main()
