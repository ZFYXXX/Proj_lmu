"""Prepare a new 1,000-image cohort without consulting answers or outcomes."""

import argparse
import hashlib
from io import BytesIO
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).parent))
from prepare_evidence_interaction import TASKS
from mllm_comp.analysis.common.evidence_interaction import digest
from mllm_comp.analysis.common.visual_token_phenomena import media_fingerprint
from mllm_comp.core.serialization import atomic_write_json


def prepare(output, previous, seed=20261021):
    import pyarrow.parquet as pq
    from PIL import Image
    import yaml
    if (output / "cohort.json").exists():
        raise ValueError("Cohort already exists; do not overwrite it.")
    output.mkdir(parents=True, exist_ok=True)
    prior = json.loads(previous.read_text())
    excluded = {row["image_fingerprint"] for row in prior["entries"]}
    original_excluded = set(excluded)
    entries, counts, sources = [], {}, []
    for task, files in TASKS.items():
        grouped, doc_id = {}, 0
        for filename in files:
            checksum = hashlib.sha256()
            with Path(filename).open("rb") as handle:
                for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                    checksum.update(chunk)
            sources.append({"path": filename, "sha256": checksum.hexdigest()})
            for batch in pq.ParquetFile(filename).iter_batches(batch_size=32, columns=["image"]):
                for row in batch.to_pylist():
                    value = row["image"]
                    image = Image.open(BytesIO(value["bytes"]) if value.get("bytes") else value["path"]).convert("RGB")
                    fingerprint = media_fingerprint([image])
                    grouped.setdefault(fingerprint, []).append(doc_id)
                    doc_id += 1
        available = sorted((key for key in grouped if key not in excluded), key=lambda key: (digest(f"{seed}:image:{key}"), key))
        if len(available) < 500:
            raise ValueError(f"Insufficient new independent images: {task} {len(available)}.")
        for index, fingerprint in enumerate(available[:500]):
            docs = sorted(grouped[fingerprint], key=lambda i: (digest(f"{seed}:question:{fingerprint}:{i}"), i))
            entries.append({"task": task, "sample_id": f"{task}:{docs[0]}", "doc_id": docs[0],
                            "image_fingerprint": fingerprint, "split": "development" if index < 100 else "confirmation",
                            "smoke": index < 8, "shard": index % 2})
            excluded.add(fingerprint)
        counts[task] = {"documents": doc_id, "unique_images": len(grouped), "new_available_images": len(available),
                        "selected_images": 500, "development": 100, "confirmation": 400}
    if original_excluded.intersection(r["image_fingerprint"] for r in entries) or len({r["image_fingerprint"] for r in entries}) != 1000:
        raise ValueError("Old/new or cross-task image overlap.")
    cohort = {"schema_version": 1, "seed": seed, "entries": entries, "counts": counts, "sources": sources,
              "selection_reads_answers": False, "excluded_cohort": str(previous),
              "excluded_cohort_sha256": hashlib.sha256(previous.read_bytes()).hexdigest(),
              "excludes_all_previous_1000_images": True,
              "sampling": "RGB content hash; fixed hash order; one hashed question; no answer/outcome access"}
    atomic_write_json(output / "cohort.json", cohort, ensure_ascii=False)
    registration = {
        "schema_version": 1, "purpose": "reconstruction error versus task utility; discovery, no training",
        "cohort_sha256": hashlib.sha256((output / "cohort.json").read_bytes()).hexdigest(),
        "main_budget": 128, "summary_tokens": 96, "original_detail_tokens": 32,
        "summary": "deterministic rectangular grid cell means, native representative M-RoPE",
        "residual": "squared FP32 error relative to own cell mean; excludes summary representatives",
        "pca_control": "centered feature-channel PCA, exact FP32 SVD, rank 8; distinct from spatial DCT",
        "random_seeds": [11, 23, 47], "conditions_per_image": 43,
        "smoke_cache_parity": "Dense native generation must match exactly; cached/cache-free greedy disagreements require same-prefix BF16 KL<=.01 and top1 regret<=.25 plus FP32 KL<=1e-6 and max logit difference<=.002; record every discrepancy. BF16-FP32-BF16 weights restored exactly. Defined during smoke, before development/confirmation.",
        "confirmatory_tests": [f"{task}:{test}" for task in TASKS for test in ["high_residual_minus_random", "high_residual_minus_spatial_random"]],
        "primary_endpoint": "native full-generation task score; ChartQA relaxed + secondary strict; TextVQA full-answer soft score",
        "bootstrap": "10,000 image-level paired resamples, average three random seeds within image, Holm four tests",
        "confirmation_gate": "Only after all 800 confirmation images; positive CI95 lower and Holm p<.05",
        "minimum_practical_gain": .02, "practical_gain_rule": "Separate descriptive gate: mean score advantage at least .02 on both contrasts",
        "frequency": {"dc_unchanged": True, "band_fraction": .25, "requested_perturbation_ac_energy_fractions": [.05, .10],
                      "dose": "common across low/high/three random bands; maximum capped at .5 times smallest band energy, both doses scaled together to preserve 1:2 ratio",
                      "matched_quantity": "squared norm of FP32 delta, not total feature energy removed",
                      "inference": "secondary mechanistic evidence; no general high-frequency claim from task accuracy alone"},
        "tile_interventions": "16 exhaustive rectangular nonoverlapping tiles; within-tile mean replacement, length/positions fixed",
        "tile_analysis": "within-image Spearman of residual/attention/PCA/norm versus NLL increase and score decrease; bootstrap images, no pooling tiles as images",
        "runtime_seed": 42, "generation_max_new_tokens": {"chartqa": 16, "textvqa": 32},
        "interpretation": "Proxy is untrained grid reconstruction, not a reproduction or rejection of learned ReConV; failure motivates testing the proposed trained hybrid separately",
        "fallback": "If task-score advantage not supported, retain negative result and evaluate a minimal summary+reconstruction-residual hybrid with training-matched controls in a new run; do not silently change this run",
    }
    atomic_write_json(output / "preregistration.json", registration, ensure_ascii=False)
    template = json.loads((previous.parent / "run_manifest.json").read_text())
    if template["status"] != "complete":
        raise ValueError("Previous frozen experiment must finish before launching next run.")
    config_base = yaml.safe_load((previous.parent / "configs/development_0.yaml").read_text())
    for stage in ["smoke", "development", "confirmation"]:
        for shard in [0, 1]:
            config = json.loads(json.dumps(config_base))
            config["experiment"]["name"] = f"reconstruction_utility_{stage}_{shard}"
            config["analysis"]["sampling"] = {"manifest_path": str(output / "cohort.json"), "manifest_stage": stage, "manifest_shard": shard}
            config["analysis"]["probes"] = [{"name": "visual_reconstruction_utility", "enabled": True,
                "artifact_root": str(output), "cohort_path": str(output / "cohort.json"), "stage": stage,
                "budget": 128, "base_tokens": 96, "save_raw": False}]
            atomic_config = output / "configs" / f"{stage}_{shard}.yaml"
            atomic_config.parent.mkdir(exist_ok=True)
            # Generated configuration is an output artifact, not a source edit.
            atomic_config.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    print(json.dumps({"output": str(output), "counts": counts, "old_cohort_overlap": 0}, ensure_ascii=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--previous", required=True, type=Path)
    args = parser.parse_args()
    prepare(args.output.resolve(), args.previous.resolve())
