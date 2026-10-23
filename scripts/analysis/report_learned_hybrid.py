"""Paired held-out task metrics for a fixed-step, single-seed training pilot."""

import argparse
from collections import defaultdict
import json
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from mllm_comp.analysis.common.evidence_interaction import bootstrap, holm
from mllm_comp.analysis.common.learned_hybrid_runtime import code_hash
from mllm_comp.compressors.image.learned_hybrid import VARIANTS
from mllm_comp.core.serialization import atomic_write_json


def report(root, stage):
    cohort = json.loads((root / "cohort.json").read_text())
    entries = [e for e in cohort["entries"] if (e.get("smoke", False) if stage == "smoke" else e["split"] == stage)]
    expected = {e["sample_id"]: e["image_fingerprint"] for e in entries}
    summaries = [json.loads(p.read_text()) for p in sorted((root / "samples" / stage).glob("*/summary.json"))]
    result = {"stage": stage, "expected_images": len(entries), "completed_images": len(summaries), "status": "incomplete"}
    if len(summaries) != len(entries):
        atomic_write_json(root / f"{stage}_report.json", result, ensure_ascii=False)
        return result
    if {s["sample_id"] for s in summaries} != set(expected) or len({s["image_fingerprint"] for s in summaries}) != len(summaries):
        raise ValueError("Missing, extra, or duplicate held-out images.")
    tasks = defaultdict(list)
    old = Path(json.loads((root / "training_samples.json").read_text())["evaluation_cohort"]).parent
    matched_previous, previous_predictions_differ = {}, []
    for summary in summaries:
        if expected[summary["sample_id"]] != summary["image_fingerprint"] or summary["implementation_hash"] != code_hash():
            raise ValueError("Wrong held-out image or mixed implementation.")
        rows = [json.loads(line) for line in Path(summary["condition_path"]).read_text().splitlines()]
        lookup = {r["variant"]: r for r in rows}
        if set(lookup) != {"dense", *VARIANTS} or len(rows) != 5:
            raise ValueError("Incomplete/duplicate image methods.")
        for row in rows:
            if row["variant"] != "dense" and row["retained_visual_tokens"] != 128:
                raise ValueError("Unfair output token budget.")
            if row["gold_answer_in_decode_context"] or not row["text_positions_preserved"] or not row["placeholder_mask_labels_alignment_passed"]:
                raise ValueError("Gold/position/label alignment error.")
        if stage == "smoke" and lookup["dense"].get("native_generation_identity_passed") is not True:
            raise ValueError("Native/dense generation parity missing.")
        tasks[summary["task"]].append(lookup)
        previous = old / "samples" / stage / summary["sample_id"].replace(":", "_") / "conditions.jsonl"
        if previous.exists():
            prior = {r["request_id"]: r for r in map(json.loads, previous.read_text().splitlines())}
            if prior["dense"]["prediction"] == lookup["dense"]["prediction"] and prior["dense"]["source_visual_tokens"] == lookup["dense"]["source_visual_tokens"]:
                matched_previous[summary["sample_id"]] = prior
            else:
                previous_predictions_differ.append(summary["sample_id"])
    tables, tests = [], []
    contrasts = [("summary_residual", "summary_random"), ("spectral_residual", "summary_residual"), ("spectral_residual", "pool_adapter")]
    for task, images in sorted(tasks.items()):
        arms = {}
        for variant in ["dense", *VARIANTS]:
            strict = [image[variant]["strict_score"] for image in images]
            arms[variant] = {"n_images": len(images), "mean_score": statistics.mean(image[variant]["primary_task_score"] for image in images),
                "mean_nll": statistics.mean(image[variant]["answer_nll"] for image in images),
                "mean_tokens": statistics.mean(image[variant]["retained_visual_tokens"] for image in images),
                "strict_score": statistics.mean(strict) if all(v is not None for v in strict) else None,
                "paired_delta_vs_dense": bootstrap([image[variant]["primary_task_score"] - image["dense"]["primary_task_score"] for image in images])}
        effects = []
        for a, b in contrasts:
            effect = {"task": task, "contrast": f"{a}-{b}", **bootstrap([image[a]["primary_task_score"] - image[b]["primary_task_score"] for image in images])}
            effects.append(effect)
            if stage == "confirmation":
                tests.append(effect)
        tables.append({"task": task, "n_images": len(images), "methods": arms, "contrasts": effects})
    if tests:
        if len(tests) != 6 or len(summaries) != 800:
            raise ValueError("Complete final evaluation requires 800 images and six prespecified contrasts.")
        for test, adjusted in zip(tests, holm([t["one_sided_p"] for t in tests])):
            test.update({"holm_adjusted_p": adjusted, "supported": adjusted < .05 and test["ci95"][0] > 0})
    old_references = {}
    if len(matched_previous) == len(summaries):
        for task in tasks:
            subset = [matched_previous[s["sample_id"]] for s in summaries if s["task"] == task]
            old_references[task] = {name: statistics.mean(row[name]["primary_task_score"] for row in subset)
                for name in ["reference/feature_mean_centroid128", "reference/feature_representative128", "reference/prumerge_plus128"]}
    result.update({"status": "complete", "implementation_hash": code_hash(), "tables": tables, "paired_tests": tests,
                   "previous_baseline_predictions_differ": previous_predictions_differ, "matched_historical_references": old_references,
                   "interpretation": "Single seed, small train-only sample, frozen native Qwen2.5-VL adaptation. Reused evaluation images are held out from training, not a new independent discovery confirmation.",
                   "elapsed_s_sum": sum(s["elapsed_s"] for s in summaries)})
    atomic_write_json(root / f"{stage}_report.json", result, ensure_ascii=False)
    lines = [f"# Learned hybrid pilot: {stage}", "", f"Completed images: {len(summaries)}/{len(entries)}", "", result["interpretation"], "",
             "| Task | Method | Native score | Strict score | Visual tokens |", "|---|---|---:|---:|---:|"]
    for table in tables:
        for variant, metrics in table["methods"].items():
            strict = "—" if metrics["strict_score"] is None else f"{metrics['strict_score']:.4f}"
            lines.append(f"| {table['task']} | {variant} | {metrics['mean_score']:.4f} | {strict} | {metrics['mean_tokens']:.1f} |")
    lines += ["", "## Paired task-score effects", "", "| Task | Contrast | Mean | Image bootstrap 95% CI | Holm p |", "|---|---|---:|---|---:|"]
    for table in tables:
        for effect in table["contrasts"]:
            lines.append(f"| {table['task']} | {effect['contrast']} | {effect['mean']:.5f} | [{effect['ci95'][0]:.5f}, {effect['ci95'][1]:.5f}] | {effect.get('holm_adjusted_p', 'development only')} |")
    lines += ["", "Historical feature-anchor/PruMerge+ references are included in JSON only if every same-image dense prediction and source-token count matched. Image bootstrap intervals do not measure training-seed variability."]
    (root / f"{stage}_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--stage", required=True)
    args = parser.parse_args()
    print(json.dumps({k: v for k, v in report(args.root.resolve(), args.stage).items() if k in ["stage", "status", "completed_images"]}))
