"""Freeze local train-only examples, excluding every evaluation image by RGB hash."""

import argparse
from collections import defaultdict
import hashlib
from io import BytesIO
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from mllm_comp.analysis.common.evidence_interaction import digest
from mllm_comp.analysis.common.visual_token_phenomena import media_fingerprint
from mllm_comp.compressors.image.learned_hybrid import VARIANTS
from mllm_comp.core.serialization import atomic_write_json


def prepare_preflight(root):
    import yaml
    preview = root / "preflight"
    preview.mkdir(exist_ok=True)
    for name in ["cohort.json", "training_samples.json"]:
        atomic_write_json(preview / name, json.loads((root / name).read_text()), ensure_ascii=False)
    for shard in [0, 1]:
        config = yaml.safe_load((root / "configs" / f"smoke_{shard}.yaml").read_text())
        probe = config["analysis"]["probes"][0]
        probe["artifact_root"] = str(preview)
        probe["allow_smoke_checkpoints"] = True
        probe["checkpoint_paths"] = {variant: str(root / "smoke_models" / variant) for variant in VARIANTS}
        target = preview / "configs" / f"smoke_{shard}.yaml"
        target.parent.mkdir(exist_ok=True)
        target.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")


def prepare(root, evaluation):
    import pyarrow.parquet as pq
    from PIL import Image
    import yaml
    root = root.resolve()
    if (root / "training_samples.json").exists():
        raise ValueError("Training cohort exists; resume it rather than overwrite.")
    root.mkdir(parents=True, exist_ok=True)
    cohort = json.loads(evaluation.read_text())
    excluded = set()
    for source in cohort["sources"]:
        for batch in pq.ParquetFile(source["path"]).iter_batches(batch_size=32, columns=["image"]):
            for row in batch.to_pylist():
                image = row["image"]
                value = Image.open(BytesIO(image["bytes"]) if image.get("bytes") else image["path"]).convert("RGB")
                excluded.add(media_fingerprint([value]))
    print(f"Evaluation exclusion set: {len(excluded)} unique images", flush=True)
    samples, source_records = [], []
    datasets = {
        "textvqa_train": ([Path(f"/home/YinPeiran/datasets/lmms_eval_datasets/textvqa/data/train-{i:05d}-of-00020.parquet") for i in range(4)], 2048),
        "scienceqa_train": ([Path("/home/YinPeiran/datasets/lmms_eval_datasets/ScienceQA/ScienceQA-IMG/train-00000-of-00001.parquet")], 512),
    }
    for name, (paths, amount) in datasets.items():
        groups = defaultdict(list)
        for path in paths:
            checksum = hashlib.sha256()
            with path.open("rb") as handle:
                for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                    checksum.update(chunk)
            source_records.append({"path": str(path), "sha256": checksum.hexdigest(), "split": "train"})
            index = 0
            for batch in pq.ParquetFile(path).iter_batches(batch_size=32, columns=["image"]):
                for row in batch.to_pylist():
                    image = row["image"]
                    value = Image.open(BytesIO(image["bytes"]) if image.get("bytes") else image["path"]).convert("RGB")
                    fingerprint = media_fingerprint([value])
                    if fingerprint not in excluded:
                        groups[fingerprint].append({"path": str(path), "row": index})
                    index += 1
        chosen = sorted(groups, key=lambda key: (digest(f"20261021:{name}:{key}"), key))[:amount]
        if len(chosen) != amount:
            raise ValueError(f"Only {len(chosen)}/{amount} independent training images for {name}.")
        for key in chosen:
            row = min(groups[key], key=lambda item: digest(f"20261021:{item['path']}:{item['row']}"))
            samples.append({**row, "source": name, "image_fingerprint": key, "sample_id": f"{name}:{Path(row['path']).name}:{row['row']}"})
            excluded.add(key)
        print(f"Selected {name}: {amount}", flush=True)
    samples.sort(key=lambda row: digest(f"42:order:{row['sample_id']}"))
    atomic_write_json(root / "training_samples.json", {"seed": 42, "entries": samples, "sources": source_records,
        "selection_reads_answers": False, "excluded_all_evaluation_source_images": True,
        "evaluation_cohort": str(evaluation.resolve()), "sampling": "RGB hash, one question per image, fixed hash order, train splits only"}, ensure_ascii=False)
    atomic_write_json(root / "cohort.json", cohort, ensure_ascii=False)
    registration = {"seed": 42, "variants": list(VARIANTS), "target_tokens": 128, "summary_tokens": 64, "width": 256,
        "training_images": 2560, "optimizer_steps": 320, "gradient_accumulation": 8, "epochs": 1,
        "learning_rate": .0002, "weight_decay": .01, "loss": {"answer_ce": 1., "teacher_kl": .5, "reconstruction_infonce": .1},
        "frozen": ["vision_encoder", "native_merger", "llm"], "trainable": "new compression module only, FP32 optimizer",
        "controls": "Identical train examples/order/steps/lr across four arms; three summary arms have identical parameters; simpler pool arm reports its smaller parameter count",
        "selection_score": "normalized orthogonal residual fraction, differs from ReConV raw residual norm",
        "position_policy": "Original representative positions for summaries and original detail positions; text M-RoPE unchanged",
        "evaluation": "200 fixed development images + 800 fixed held-out evaluation images, no training on either; benchmark images have been evaluated previously",
        "scope": "Single-seed small-data native Qwen2.5-VL-3B adaptation, not official ReConV/Braco reproduction or a new SOTA claim",
        "chartqa_training": "No local ChartQA train split found; ChartQA is transfer evaluation",
        "checkpoint_selection": "Final fixed step 320 only; no test-based checkpoint selection",
        "primary_contrasts": ["summary_residual-summary_random", "spectral_residual-summary_residual", "spectral_residual-pool_adapter"],
        "bootstrap": "10,000 paired image resamples per task, six contrasts Holm-corrected for superiority; task scores primary",
        "frozen_after_smoke": True}
    atomic_write_json(root / "preregistration.json", registration, ensure_ascii=False)
    prior_root = evaluation.parent
    for stage in ["smoke", "development", "confirmation"]:
        for shard in [0, 1]:
            template = yaml.safe_load((prior_root / "configs" / f"{stage}_{shard}.yaml").read_text())
            template["experiment"]["name"] = f"learned_hybrid_{stage}_{shard}"
            template["analysis"]["sampling"] = {"manifest_path": str(root / "cohort.json"), "manifest_stage": stage, "manifest_shard": shard}
            template["analysis"]["probes"] = [{"name": "learned_hybrid_evaluation", "enabled": True, "artifact_root": str(root),
                "cohort_path": str(root / "cohort.json"), "stage": stage,
                "checkpoint_paths": {variant: str(root / "models" / variant) for variant in VARIANTS}, "save_raw": False}]
            config = root / "configs" / f"{stage}_{shard}.yaml"
            config.parent.mkdir(exist_ok=True)
            config.write_text(yaml.safe_dump(template, sort_keys=False), encoding="utf-8")
    prepare_preflight(root)
    print(json.dumps({"root": str(root), "training_images": len(samples), "train_eval_image_overlap": 0}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--evaluation", type=Path, required=True)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    if args.preflight_only:
        prepare_preflight(args.root.resolve())
    else:
        prepare(args.root, args.evaluation)
