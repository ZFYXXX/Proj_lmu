"""Freeze the answer-blind 1,000-image cohort and executable analysis configs."""

import argparse
import hashlib
from io import BytesIO
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from mllm_comp.analysis.common.evidence_interaction import digest
from mllm_comp.analysis.common.visual_token_phenomena import media_fingerprint
from mllm_comp.core.serialization import atomic_write_json

TASKS = {
    "qwen25_vl_chartqa_full": ["/home/YinPeiran/datasets/lmms_eval_datasets/ChartQA/data/test-00000-of-00001.parquet"],
    "qwen25_projector_quality_textvqa_full": [f"/home/YinPeiran/datasets/lmms_eval_datasets/textvqa/data/validation-0000{i}-of-00003.parquet" for i in range(3)],
}


def prepare(output: Path, seed: int = 20261020) -> None:
    import pyarrow.parquet as pq
    from PIL import Image
    import yaml
    if (output / "cohort.json").exists():
        raise ValueError("Cohort already exists; use the existing manifest, never overwrite it.")
    output.mkdir(parents=True, exist_ok=True)
    entries, sources, excluded, counts = [], [], set(), {}
    for task, files in TASKS.items():
        grouped = {}
        doc_id = 0
        for filename in files:
            path = Path(filename)
            checksum = hashlib.sha256()
            with path.open("rb") as source:
                for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
                    checksum.update(chunk)
            sources.append({"path": filename, "sha256": checksum.hexdigest()})
            for batch in pq.ParquetFile(path).iter_batches(batch_size=32, columns=["image"]):
                for row in batch.to_pylist():
                    data = row["image"]
                    image = Image.open(BytesIO(data["bytes"]) if data.get("bytes") else data["path"]).convert("RGB")
                    key = media_fingerprint([image])
                    grouped.setdefault(key, []).append(doc_id)
                    doc_id += 1
        ordered = sorted((key for key in grouped if key not in excluded), key=lambda key: (digest(f"{seed}:image:{key}"), key))
        if len(ordered) < 500:
            raise ValueError(f"Only {len(ordered)} independent images in {task}.")
        chosen = ordered[:500]
        excluded.update(chosen)
        task_entries = []
        for index, fingerprint in enumerate(chosen):
            docs = sorted(grouped[fingerprint], key=lambda d: (digest(f"{seed}:question:{fingerprint}:{d}"), d))
            task_entries.append({"task": task, "doc_id": docs[0], "sample_id": f"{task}:{docs[0]}",
                                 "image_fingerprint": fingerprint, "split": "development" if index < 100 else "confirmation",
                                 "smoke": index < 8, "shard": index % 2, "second_doc_id": docs[1] if len(docs) > 1 else None})
        confirmation = task_entries[100:]
        for row in confirmation[:100]:
            row["budget_review"] = True
        question_rows = [row for row in confirmation if row["second_doc_id"] is not None][:50]
        for row in question_rows:
            row["question_review"] = True
        entries.extend(task_entries)
        counts[task] = {"documents": doc_id, "unique_images": len(grouped), "selected_images": 500,
                        "development": 100, "confirmation": 400, "question_review": len(question_rows)}
    cohort = {"schema_version": 1, "seed": seed, "selection_reads_answers": False, "counts": counts,
              "entries": entries, "sources": sources, "sampling": "bottom-k RGB content SHA256, one hashed question per image; ChartQA first for cross-dataset de-duplication"}
    atomic_write_json(output / "cohort.json", cohort, ensure_ascii=False)
    registration = {"schema_version": 1, "cohort_sha256": hashlib.sha256((output / "cohort.json").read_bytes()).hexdigest(),
                    "main_budget": 128, "seeds": [11,23,47], "conditions_per_seed": 37,
                    "confirmatory_tests": [f"{task}:{test}" for task in TASKS for test in ["question_interaction_gt_zero", "question_minus_random_gt_zero"]],
                    "bootstrap_unit": "image", "bootstrap_repeats": 10000, "multiplicity": "Holm four one-sided tests, alpha=.05",
                    "inference": "Only after all 800 confirmation images complete; development excluded",
                    "scope": "Region replacement interactions, not a claim of semantic relation preservation or deployable selection",
                    "runtime_seed": 42, "generation_max_new_tokens": {"chartqa":16,"textvqa":32},
                    "budget_review": [64,256], "budget256_min_source_tokens": 288}
    atomic_write_json(output / "preregistration.json", registration, ensure_ascii=False)
    for stage in ["smoke", "development", "confirmation", "question_review", "budget_review"]:
        for shard in [0,1]:
            config = {"experiment": {"name": f"evidence_interaction_{stage}_{shard}", "seed": 42},
                      "backbone": {"name":"qwen25_vl", "model_path":"/home/ZhouFengyu/fastv-repro/models/Qwen2.5-VL-3B-Instruct",
                                   "model_revision":"1b5a7674d40a0ac9acef541a46fed12d710bd36b", "trust_remote_code":False,
                                   "config":{"torch_dtype":"bfloat16", "attn_implementation":"eager", "model_load_kwargs":{"local_files_only":True},
                                             "processor_load_kwargs":{"local_files_only":True,"use_fast":False,"min_pixels":200704,"max_pixels":1605632}}},
                      "pipeline":{"stages":[]},
                      "analysis":{"backend":"lmms_eval", "compression_mode":"none", "phase":"pre_compression", "runtime_batch_size":1,
                                  "tasks":list(TASKS), "task_include_path":[str(ROOT / "configs/eval/lmms_eval/tasks/qwen25_vl_chartqa_full"),str(ROOT / "configs/eval/lmms_eval/tasks/qwen25_vl_projector_quality")],
                                  "streaming_datasets":True, "sampling":{"manifest_path":str(output / "cohort.json"),"manifest_stage":stage,"manifest_shard":shard},
                                  "probes":[{"name":"visual_evidence_interaction","enabled":True,"cohort_path":str(output / "cohort.json"),
                                             "artifact_root":str(output),"stage":stage,"budget":128,"random_seeds":[11,23,47],"save_raw":False}]},
                      "train":{},"eval":{},"data":{},"output":{}}
            path = output / "configs" / f"{stage}_{shard}.yaml"
            path.parent.mkdir(exist_ok=True)
            path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    print(json.dumps({"output":str(output),"counts":counts}, ensure_ascii=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    prepare(args.output.resolve())
