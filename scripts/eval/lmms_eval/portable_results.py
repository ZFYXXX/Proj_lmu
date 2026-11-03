"""Portable evidence for reusing full results without moving model/data files."""
import gzip
import hashlib
import inspect
import json
import os
from pathlib import Path


def sample_signature(row):
    value = [row.get("doc_hash"), row.get("input"), row.get("target")]
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def normalize(value):
    if callable(value):
        try:
            source = inspect.getsource(value)
        except (OSError, TypeError):
            source = value.__name__
        return {"function": value.__name__, "source_sha256": hashlib.sha256(source.encode()).hexdigest()}
    if isinstance(value, dict):
        return {key: normalize(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [normalize(item) for item in value]
    return value


def contract(config, template):
    from lmms_eval.utils import load_yaml_config
    from scripts.eval.lmms_eval.qwen25_portable_queue import sha
    task = load_yaml_config(str(template))
    fields = ("doc_to_text", "doc_to_target", "process_results", "filter_list", "metric_list", "lmms_eval_specific_kwargs")
    stages = []
    for stage in config["pipeline"]["stages"]:
        options = dict(stage["config"])
        for key in ("adapter_path", "checkpoint_path", "scorer_path"):
            if key in options:
                folder = Path(options[key])
                options[key] = {p.name: sha(p) for p in sorted(folder.iterdir())
                               if p.is_file() and (p.suffix == ".safetensors" or p.name.endswith("_config.json"))}
        options.pop("identity_path", None)
        if stage["algorithm"] == "importance_anchor_adapter":
            options.setdefault("spatial_beta", 0.)
            options.setdefault("pooling", "mean")
        stages.append({"algorithm": stage["algorithm"], "config": options})
    processor = config["backbone"]["config"].get("processor_load_kwargs", {})
    defaults=json.loads((Path(config["backbone"]["model_path"]) / "generation_config.json").read_text())
    return {"processor": {key: processor.get(key, False if key == "use_fast" else None)
                          for key in ("min_pixels", "max_pixels", "use_fast")},
            "attention": config["backbone"]["config"].get("attn_implementation"),
            "dtype": config["backbone"]["config"].get("torch_dtype"),
            "generation": {"repetition_penalty": defaults.get("repetition_penalty", 1.),
                           **task.get("generation_kwargs", {}), **config["eval"].get("generation", {})},
            "task": normalize({key: task[key] for key in fields if key in task}), "stages": stages}


def import_bundle(bundle_path, output, methods, tasks):
    from scripts.eval.lmms_eval.qwen25_portable_queue import dump, sha
    from scripts.eval.lmms_eval.p6000_dataset_smoke import specifications
    from types import SimpleNamespace
    import yaml
    plan = json.loads((output / "plan.json").read_text())
    if plan["limit"] is not None:
        raise ValueError("Full-result bundles cannot be imported into a limited smoke run")
    opener = gzip.open if str(bundle_path).endswith(".gz") else open
    with opener(bundle_path, "rt") as stream:
        bundle = json.load(stream)
    if bundle["model_identity_sha256"] != plan["model_identity_sha256"]:
        raise ValueError("Reuse bundle belongs to another base model")
    templates = {name: template for name, template, _ in specifications()}
    hashes, audit = {}, []
    for entry in bundle["cells"]:
        method, task = entry["method"], entry["task"]
        if method not in methods or task not in tasks:
            continue
        cell = output / "cells" / method / task
        if (cell / "complete.json").is_file():
            continue
        # Generate just the standard model settings, without instantiating data
        # or loading a model. Export and import compare this same contract.
        from scripts.eval.lmms_eval.p6000_dataset_smoke import model_config
        args = SimpleNamespace(model=plan["model"], method=method,
            identity_path=str(output / "model_identity.json"), method_checkpoint=plan["checkpoints"].get(method),
            scorer_checkpoint=plan.get("scorer_checkpoint"))
        current = contract(model_config(args), templates[task])
        if entry["contract"] != current:
            audit.append({"method": method, "task": task, "status": "not_reused", "reason": "protocol_mismatch"})
            continue
        valid_sources = True
        for source in entry["dataset_files"]:
            path = Path(plan["data_root"]) / source["relative_path"]
            if not path.is_file() and task == "mmstar":
                choices = [Path(os.environ["MLLM_MMSTAR_PARQUET"])] if os.environ.get("MLLM_MMSTAR_PARQUET") else []
                for pattern in ("MMStar/data/validation-*.parquet", "MMStar/data/val-*.parquet", "MMStar/data/test-*.parquet"):
                    choices.extend(sorted(Path(plan["data_root"]).glob(pattern)))
                if len(choices) == 1:
                    path = choices[0]
            if not path.is_file():
                valid_sources = False
                break
            if str(path) not in hashes:
                hashes[str(path)] = sha(path)
            if path.stat().st_size != source["bytes"] or hashes[str(path)] != source["sha256"]:
                valid_sources = False
                break
        if not valid_sources:
            audit.append({"method": method, "task": task, "status": "not_reused", "reason": "dataset_content_or_layout_differs"})
            continue
        evidence = output / "reused_evidence" / f"{method}__{task}.json"
        evidence.parent.mkdir(exist_ok=True)
        dump(evidence, {"sample_signatures": entry["sample_signatures"],
                        "predictions": entry.get("predictions"), "source": entry["source"]})
        cell.mkdir(parents=True, exist_ok=True)
        score = {"score": entry["score"], "samples": entry["samples"], "metric": entry["metric"],
            "processed_total": entry["samples"], "subset": "full", "reused_from": str(bundle_path.resolve()),
            "evidence_path": str(evidence.resolve()), "source": entry["source"]}
        dump(cell / "complete.json", {"method": method, "task": task, "score": score})
        audit.append({"method": method, "task": task, "status": "reused", "score": entry["score"]})
        print(f"SEEDED portable {method}/{task}: {entry['samples']} requests", flush=True)
    dump(output / "bundle_import.json", {"bundle_sha256": sha(bundle_path), "cells": audit})
