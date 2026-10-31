"""Exercise local benchmark data through the platform's native eval entrypoint.

Small parquet slices avoid rebuilding huge full-dataset caches. Source files are
read-only. Override --data-root/--model when moving to another machine.
"""
from __future__ import annotations

import argparse
from functools import lru_cache
from io import BytesIO
from datetime import datetime, timezone
from importlib.metadata import version
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import traceback

import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq
import yaml

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
TASKS = ROOT / "configs/eval/lmms_eval/tasks"
UPSTREAM = Path(__import__("lmms_eval").__file__).parent / "tasks"


class Function(str):
    """Write native lmms-eval function tags without altering source YAMLs."""


yaml.SafeDumper.add_representer(Function, lambda dumper, value: dumper.represent_scalar("!function", str(value)))


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str) + "\n")


def specifications():
    # Each entry retains the existing task's prompt, target and native scorer.
    rows = [
        ("chartqa", TASKS / "qwen25_7b_projector_ratio/chartqa.yaml", "ChartQA/data/test-*.parquet"),
        ("gqa", TASKS / "qwen25_vl_projector_quality/gqa.yaml", "GQA/testdev_balanced_instructions/testdev-*.parquet"),
        ("vqav2_val", UPSTREAM / "vqav2/vqav2_val.yaml", "VQAv2/data/validation-*.parquet"),
        ("okvqa", TASKS / "qwen25_okvqa_checked/okvqa.yaml", "OK-VQA/data/val2014-*.parquet"),
        ("scienceqa_img", TASKS / "qwen25_vl_scienceqa_full/scienceqa_img_full.yaml", "ScienceQA/ScienceQA-IMG/test-*.parquet"),
        ("ai2d", TASKS / "context_visual_topk/ai2d.yaml", "ai2d/data/test-*.parquet"),
        ("mmstar", TASKS / "projector_extra/mmstar.yaml", "MMStar/mmstar.parquet"),
        ("mmmu", UPSTREAM / "mmmu/mmmu_val.yaml", "MMMU/data/validation-*.parquet"),
        ("pope", TASKS / "projector_extra/pope.yaml", "POPE/Full/*.parquet"),
        ("hallusionbench", UPSTREAM / "hallusion_bench/hallusion_bench_image.yaml", "HallusionBench/data/image-*.parquet"),
        ("textvqa", TASKS / "qwen25_vl_projector_quality/textvqa_full.yaml", "textvqa/data/validation-*.parquet"),
        ("docvqa", TASKS / "qwen25_7b_projector_ratio/docvqa.yaml", "DocVQA/DocVQA/validation-*.parquet"),
        ("ocrbench", TASKS / "qwen25_7b_projector_ratio/ocrbench.yaml", "OCRBench/data/test-*.parquet"),
    ]
    for dataset, prefix, splits in [
        ("RefCOCO", "refcoco", ["testA", "testB"]),
        ("RefCOCOplus", "refcoco+", ["testA", "testB"]),
        ("RefCOCOg", "refcocog", ["test"]),
    ]:
        for split in splits:
            rows.append((f"{prefix}_bbox_rec_{split}",
                         TASKS / "qwen25_pope_grounding/refcoco_plus_testA.yaml",
                         f"{dataset}/data/{split}-*.parquet"))
    return rows


def check_shards(paths):
    """Refuse incomplete Hugging Face parquet shard sets."""
    groups = {}
    for path in paths:
        match = re.match(r"^(.*)-(\d+)-of-(\d+)\.parquet$", path.name)
        if match:
            prefix, part, total = match.groups()
            groups.setdefault((prefix, int(total)), set()).add(int(part))
    for (prefix, total), parts in groups.items():
        if parts != set(range(total)):
            raise ValueError(f"Incomplete parquet split {prefix}: {len(parts)}/{total} shards")


@lru_cache(maxsize=1)
def gqa_index():
    image_dir = Path(os.environ["MLLM_DATA_ROOT"]) / "GQA/testdev_all_images"
    paths = sorted(image_dir.glob("*.parquet"))
    if not paths:
        raise FileNotFoundError(f"Missing GQA image parquet in {image_dir}")
    check_shards(paths)
    index = {}
    for path in paths:
        parquet = pq.ParquetFile(path)
        for group in range(parquet.num_row_groups):
            for offset, image_id in enumerate(parquet.read_row_group(group, columns=["id"]).column("id").to_pylist()):
                if image_id in index:
                    raise ValueError(f"Duplicate GQA image ID: {image_id}")
                index[image_id] = (str(path), group, offset)
    return index


@lru_cache(maxsize=2)
def gqa_group(path, group):
    return pq.ParquetFile(path).read_row_group(group, columns=["image"]).column("image").to_pylist()


def gqa_full_visual(doc):
    from PIL import Image
    path, group, offset = gqa_index()[doc["imageId"]]
    value = gqa_group(path, group)[offset]
    with Image.open(BytesIO(value["bytes"]) if value.get("bytes") else value["path"]) as image:
        return [image.convert("RGB")]


def hb_deferred_result(doc, results):
    """Keep the upstream judge payload while explicitly deferring scoring."""
    record = {key: value for key, value in doc.items() if key != "image"}
    record["model_prediction"] = results[0]
    return {"hallusion_record": record, "generation_nonempty_smoke": float(bool(str(results[0]).strip()))}


def hb_record_count(records):
    return len(records)


def slice_process_docs(dataset):
    """Apply the native task preprocessing before selecting evaluation rows."""
    from lmms_eval.utils import load_yaml_config
    template = os.environ["MLLM_CHUNK_TEMPLATE"]
    original = load_yaml_config(template).get("process_docs")
    if original is not None:
        dataset = original(dataset)
    original_total = len(dataset)
    if os.environ.get("MLLM_MMMU_SINGLE_IMAGE_ONLY") == "1":
        # Match MMMU's visual loader, which also reads image references in options.
        keep = [i for i, (question, options, kind) in enumerate(zip(dataset["question"], dataset["options"], dataset["question_type"]))
                if len(set(re.findall(r"<image \d+>", (question or "") + " " +
                                      ((options or "") if kind == "multiple-choice" else "")))) == 1]
        dataset = dataset.select(keep)
    total = len(dataset)
    start = int(os.environ["MLLM_CHUNK_START"])
    stop = min(total, start + int(os.environ["MLLM_CHUNK_SIZE"]))
    if start >= total:
        raise ValueError(f"Chunk start {start} is outside {total} processed rows")
    Path(os.environ["MLLM_CHUNK_INFO"]).write_text(json.dumps({"total": total, "start": start, "stop": stop,
        "original_total": original_total, "excluded": original_total - total}))
    return dataset.select(range(start, stop))


def prepare(args):
    from huggingface_hub import hf_hub_download
    from lmms_eval.utils import load_yaml_config
    from lmms_eval.api.task import ConfigurableTask

    # Standard MMMU scoring is deterministic. Its utility module nevertheless
    # constructs an unused OpenAI client at import time; no judge call is made.
    os.environ.setdefault("OPENAI_API_KEY", "unused-local-smoke-no-judge")
    out = args.output.resolve()
    for sub in ("tasks", "data", "results", "cache"):
        (out / sub).mkdir(parents=True, exist_ok=True)
    selected_specs = [row for row in specifications() if row[0] in args.tasks]
    manifest = {"status": "preparing", "created_utc": datetime.now(timezone.utc).isoformat(),
                "model": args.model, "data_root": str(args.data_root), "method": args.method,
                "identity_path": args.identity_path, "method_checkpoint": args.method_checkpoint,
                "input_mode": args.input_mode, "samples_per_split": args.samples,
                "eval_limit": args.eval_limit, "first_shard_only": args.first_shard_only, "gpu": args.gpu, "tasks": {},
                "selected_tasks": args.tasks, "python_executable": sys.executable,
                "transformers_version": __import__("transformers").__version__,
                "lmms_eval_source": __import__("lmms_eval").__file__,
                "versions": {p: version(p) for p in ("torch", "transformers", "datasets", "pyarrow", "accelerate", "qwen-vl-utils")},
                "purpose": ("Full dataset evaluation" if args.input_mode == "full" and args.eval_limit is None and not args.first_shard_only
                            else "Limited flow validation; scores are not benchmark estimates.")}
    save(out / "manifest.json", manifest)
    names = []
    for name, template, pattern in selected_specs:
        try:
            sources = sorted(args.data_root.glob(pattern))
            if not sources and name == "mmstar":
                for alternative in ("MMStar/data/validation-*.parquet", "MMStar/data/val-*.parquet",
                                    "MMStar/data/test-*.parquet"):
                    sources = sorted(args.data_root.glob(alternative))
                    if sources:
                        break
                if not sources:
                    sources = [ROOT.parents[1] / "datasets/MMStar/mmstar.parquet"]
            if not sources and name == "hallusionbench":
                if args.input_mode == "full" and not args.hallusion_image:
                    raise FileNotFoundError("Provide a local HallusionBench image dataset for full evaluation")
                sources = ([args.hallusion_image] if args.hallusion_image else
                           [Path(hf_hub_download("lmms-lab/HallusionBench", "data/image-00000-of-00001.parquet",
                                                 repo_type="dataset", cache_dir=str(out / "cache/hub")))])
            if not sources or not all(p.is_file() for p in sources):
                raise FileNotFoundError(f"No source matches {args.data_root / pattern}")
            check_shards(sources)
            table = None
            if args.input_mode == "sample":
                selected = sources if name == "pope" else sources[:1]
                tables = []
                for source in selected:
                    batch = next(pq.ParquetFile(source).iter_batches(batch_size=args.samples))
                    tables.append(pa.Table.from_batches([batch]))
                table = pa.concat_tables(tables)
            if name == "gqa" and table is not None:
                ids = table.column("imageId").to_pylist()
                images = ds.dataset(str(args.data_root / "GQA/testdev_all_images"), format="parquet").to_table(
                    filter=ds.field("id").isin(ids), columns=["id", "image"])
                by_id = {r["id"]: r["image"] for r in images.to_pylist()}
                table = table.append_column("image", pa.array([by_id[i] for i in ids], type=images.schema.field("image").type))
            inputs = [str(p) for p in (sources[:1] if args.first_shard_only else sources)]
            if table is not None:
                sample = out / "data" / f"{name}.parquet"
                pq.write_table(table, sample)
                inputs = [str(sample)]
            tag = "projector_full" if args.input_mode == "full" else "p6000_smoke"
            task_name = f"{name}_{tag}" if name.startswith("refcoco") else f"{tag}_{name}"
            task = {"include": str(template), "task": task_name, "dataset_path": "parquet", "dataset_name": None,
                    "dataset_kwargs": {"data_files": {"smoke": inputs}, "cache_dir": str(args.shared_cache or out / "cache/datasets")},
                    "test_split": "smoke", "training_split": None, "validation_split": None, "fewshot_split": None}
            if args.chunk_size is not None:
                os.environ.update(MLLM_CHUNK_TEMPLATE=str(template), MLLM_CHUNK_START=str(args.chunk_start),
                                  MLLM_CHUNK_SIZE=str(args.chunk_size), MLLM_CHUNK_INFO=str(out / "data/chunk_info.json"))
                os.environ["MLLM_MMMU_SINGLE_IMAGE_ONLY"] = "1" if name == "mmmu" and args.mmmu_single_image_only else "0"
                task["process_docs"] = Function("scripts.eval.lmms_eval.p6000_dataset_smoke.slice_process_docs")
            if name == "gqa":
                task["doc_to_visual"] = Function("configs.eval.lmms_eval.tasks.projector_extra.extra_utils.visual" if table is not None
                                                 else "scripts.eval.lmms_eval.p6000_dataset_smoke.gqa_full_visual")
            if name == "hallusionbench" and args.hallusion_scoring == "smoke":
                if args.input_mode == "full" and args.eval_limit is None:
                    raise ValueError("Full HallusionBench requires official scoring, not generation-only smoke")
                # The original aAcc/qAcc/fAcc aggregators invoke a paid GPT judge.
                # Validate generation locally without mislabelling a substitute as official scoring.
                task["process_results"] = Function("scripts.eval.lmms_eval.p6000_dataset_smoke.hb_smoke_result")
                task["metric_list"] = [{"metric": "generation_nonempty_smoke", "aggregation": "mean", "higher_is_better": True}]
            if name == "hallusionbench" and args.hallusion_scoring == "deferred":
                task["process_results"] = Function("scripts.eval.lmms_eval.p6000_dataset_smoke.hb_deferred_result")
                task["metric_list"] = [
                    {"metric": "generation_nonempty_smoke", "aggregation": "mean", "higher_is_better": True},
                    {"metric": "hallusion_record", "aggregation": Function("scripts.eval.lmms_eval.p6000_dataset_smoke.hb_record_count"), "higher_is_better": True},
                ]
            text = yaml.safe_dump(task, sort_keys=False, allow_unicode=True)
            config_path = out / "tasks" / f"{name}.yaml"
            config_path.write_text(text)
            config = load_yaml_config(str(config_path))
            loaded = ConfigurableTask(config=config)
            docs = loaded.eval_docs
            os.environ["MLLM_DATA_ROOT"] = str(args.data_root.resolve())
            doc = docs[0]
            visual = loaded.doc_to_visual(doc)
            prompt = loaded.doc_to_text(doc)
            if (not visual and name != "mmmu") or not str(prompt).strip():
                raise ValueError("Empty visual input or prompt")
            entry = {"status": "data_passed", "task": task_name, "sources": [str(p) for p in sources],
                     "source_stats": [{"path": str(p), "bytes": p.stat().st_size, "mtime_ns": p.stat().st_mtime_ns} for p in sources],
                     "source_rows": sum(pq.ParquetFile(p).metadata.num_rows for p in sources),
                     "sampled_source_rows": table.num_rows if table is not None else None, "eval_rows": len(docs),
                     "image_sizes": [list(image.size) for image in visual], "template": str(template)}
            if args.chunk_size is not None:
                entry["chunk"] = json.loads((out / "data/chunk_info.json").read_text())
            if name == "hallusionbench" and args.hallusion_scoring == "smoke":
                entry["scoring_note"] = "Generation smoke only; official aAcc/qAcc/fAcc needs external judge, not exercised."
            manifest["tasks"][name] = entry
            names.append(task_name)
            print(f"DATA PASS {name}: {len(docs)} eval rows, images={entry['image_sizes']}", flush=True)
        except Exception:
            manifest["tasks"][name] = {"status": "data_failed", "error": traceback.format_exc()}
            print(f"DATA FAILED {name}\n{traceback.format_exc()}", flush=True)
        save(out / "manifest.json", manifest)
    base = yaml.safe_load((ROOT / "configs/eval/lmms_eval/qwen25_vl_14bench.yaml").read_text())
    base["backbone"]["model_path"] = args.model
    base["backbone"]["config"]["attn_implementation"] = "eager"
    base["backbone"]["config"]["processor_load_kwargs"]["use_fast"] = False
    if args.method != "dense":
        algorithms = {"old_mlp": "feature_anchor_adapter", "visual_selector": "learned_visual_topk",
                      "image_context_selector": "context_visual_topk", "question_context_selector": "context_visual_topk",
                      "protected_selector": "protected_visual_topk", "eadp": "eadp", "prunesid": "prunesid", "mmtok": "mmtok"}
        algorithm = algorithms[args.method]
        if args.method == "eadp":
            options = {"alpha": 0.5, "beta": 2.0, "tokens_per_crop": 128}
        elif args.method == "prunesid":
            options = {"tokens_per_crop": 128}
        elif args.method == "mmtok":
            options = {"target_count": 128}
        else:
            if not args.identity_path or not args.method_checkpoint:
                raise ValueError("A trained method requires --identity-path and --method-checkpoint")
            options = {"identity_path": args.identity_path}
            if args.method == "old_mlp":
                options.update(adapter_path=args.method_checkpoint, tokens_per_crop=128)
            else:
                options.update(checkpoint_path=args.method_checkpoint, seed=42, target_count=128, width=256)
                if args.method in {"image_context_selector", "question_context_selector"}:
                    options["context_mode"] = "image" if args.method == "image_context_selector" else "question"
                if args.method == "protected_selector":
                    options.update(match_width=128, visual_only=False,
                        protection={"maximum_anchor_ratio": 0.5, "minimum_anchor": 20,
                                    "novelty_patience": 3, "novelty_threshold": 0.2, "relevance": "learned_word"})
        base["pipeline"]["stages"] = [{"algorithm": algorithm, "plugin": algorithm,
            "binding": f"{algorithm}.image.qwen25_vl", "variant": f"{algorithm}.image",
            "config": options, "execution_phase": "generate", "method": "token_compression",
            "modalities": ["image"], "target_modalities": ["image"], "name": args.method,
            "object_family": "tensor_slot", "placement": "projector_level"}]
        if args.method == "protected_selector":
            base["pipeline"]["stages"][0]["conditioning_modalities"] = ["text"]
        base["eval"].update(compression_mode="compressed", target="compressed")
    base["eval"].update(checkpoint_path=args.model, tasks=names, sample_limit=args.eval_limit,
                         generation={"do_sample": False, "temperature": 0., "repetition_penalty": 1.05})
    base["eval"]["args"]["include_path"] = str(out / "tasks")
    base["output"] = {"run_dir": str(out / "results"), "manifest_path": str(out / "results/manifest.json")}
    (out / "eval.yaml").write_text(yaml.safe_dump(base, sort_keys=False))
    manifest["status"] = "prepared" if len(names) == len(selected_specs) else "prepared_with_data_failures"
    save(out / "manifest.json", manifest)
    return manifest


def hb_smoke_result(doc, results):
    return {"generation_nonempty_smoke": float(bool(results and str(results[0]).strip()))}


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=2)
    parser.add_argument("--input-mode", choices=["sample", "full"], default="sample")
    parser.add_argument("--eval-limit", type=int)
    parser.add_argument("--first-shard-only", action="store_true", help="Limit source data as well as requests for P6000 checks")
    parser.add_argument("--shared-cache", type=Path)
    parser.add_argument("--chunk-start", type=int, default=0)
    parser.add_argument("--chunk-size", type=int)
    parser.add_argument("--mmmu-single-image-only", action="store_true")
    parser.add_argument("--hallusion-scoring", choices=["smoke", "official", "deferred"], default="smoke")
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--method", choices=["dense", "old_mlp", "visual_selector", "image_context_selector",
                                              "question_context_selector", "protected_selector", "eadp", "prunesid", "mmtok"], default="dense")
    parser.add_argument("--identity-path")
    parser.add_argument("--method-checkpoint")
    parser.add_argument("--hallusion-image", type=Path)
    parser.add_argument("--tasks", nargs="+", choices=[row[0] for row in specifications()],
                        default=["gqa", "vqav2_val", "scienceqa_img", "mmmu", "hallusionbench",
                                 "refcoco_bbox_rec_testA", "refcoco_bbox_rec_testB", "refcocog_bbox_rec_test"])
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()
    if args.samples < 1:
        parser.error("--samples must be positive")
    if args.eval_limit is not None and args.eval_limit < 1:
        parser.error("--eval-limit must be positive")
    if args.first_shard_only and (args.input_mode != "full" or args.eval_limit is None):
        parser.error("--first-shard-only requires full-source mode and a finite --eval-limit")
    if args.chunk_size is not None and (args.chunk_size < 1 or args.chunk_start < 0 or len(args.tasks) != 1):
        parser.error("Chunk mode requires one task, a nonnegative start and a positive chunk size")
    if args.hallusion_scoring == "official" and "hallusionbench" in args.tasks and not os.environ.get("OPENAI_API_KEY"):
        parser.error("Official HallusionBench scorer needs configured external judge credentials")
    manifest = prepare(args)
    if args.run:
        if manifest["status"] != "prepared":
            raise RuntimeError("Dataset preparation failed; see manifest.json")
        manifest["status"] = "running"
        save(args.output / "manifest.json", manifest)
        env = {**os.environ, "CUDA_VISIBLE_DEVICES": args.gpu, "PYTHONPATH": str(ROOT),
               "HF_HUB_OFFLINE": "1", "HF_DATASETS_OFFLINE": "1", "TOKENIZERS_PARALLELISM": "false",
               "OMP_NUM_THREADS": "2", "MKL_NUM_THREADS": "2", "MLLM_DATA_ROOT": str(args.data_root.resolve())}
        command = [sys.executable, "-m", "mllm_comp.cli.main", "eval", "--config", str(args.output / "eval.yaml"),
                   "--output_dir", str(args.output / "results"), "--manifest", str(args.output / "results/manifest.json")]
        code = subprocess.call(command, cwd=ROOT, env=env)
        scored = {}
        for path in (args.output / "results").rglob("*_results.json"):
            scored.update(json.loads(path.read_text()).get("results", {}))
        for record in manifest["tasks"].values():
            task = record.get("task")
            if task in scored:
                metrics = scored[task]
                if any(isinstance(v, (int, float)) and not math.isfinite(v) for v in metrics.values()):
                    record.update(status="eval_failed", error="Nonfinite metric")
                else:
                    record.update(status="generation_and_scoring_passed", metrics=metrics)
        complete = code == 0 and all(r["status"] == "generation_and_scoring_passed" for r in manifest["tasks"].values())
        manifest.update(status="complete" if complete else "incomplete", eval_exit_code=code)
        save(args.output / "manifest.json", manifest)
        sys.exit(0 if complete else 1)


if __name__ == "__main__":
    main()
