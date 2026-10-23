"""Image-level task utility, residual correlations, and matched-frequency effects."""

import argparse
from collections import defaultdict
import json
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from mllm_comp.analysis.common.evidence_interaction import SEEDS, bootstrap, holm
from mllm_comp.analysis.common.visual_token_phenomena import spearman_correlation
from mllm_comp.core.serialization import atomic_write_json


def stage_entries(cohort, stage):
    return [row for row in cohort["entries"] if (row.get("smoke", False) if stage == "smoke" else row["split"] == stage)]


def image_metrics(summary):
    rows = [json.loads(line) for line in Path(summary["condition_path"]).read_text().splitlines()]
    lookup = {row["request_id"]: row for row in rows}
    if len(rows) != 43 or len(lookup) != len(rows) or summary["condition_count"] != len(rows):
        raise ValueError("Incomplete/duplicate reconstruction condition ledger.")
    if any(row["config_hash"] != summary["config_hash"] or row["sample_id"] != summary["sample_id"] for row in rows):
        raise ValueError("Mixed image or configuration in ledger.")
    for row in rows:
        if not row["placeholder_mask_labels_alignment_passed"] or not row["text_positions_preserved"] or row["gold_answer_in_decode_context"]:
            raise ValueError("Failed position/alignment/gold-free generation gate.")
        count = row["retained_visual_tokens"]
        if row["request_id"].startswith(("hybrid/", "reference/")) and count != 128:
            raise ValueError("Unfair visual token budget.")
        if row["request_id"].startswith(("spectral/", "tile/")) and count != summary["source_visual_tokens"]:
            raise ValueError("Mechanistic intervention changed sequence length.")
    score = lambda name: lookup[name]["primary_task_score"]
    high = score("hybrid/high_residual")
    result = {"sample_id": summary["sample_id"], "image_fingerprint": summary["image_fingerprint"], "task": summary["task"],
              "source_visual_tokens": summary["source_visual_tokens"], "dense_score": score("dense"),
              "high_residual_score": high, "effects": {}, "correlations": {}, "spectral": {}}
    for kind in ["random", "spatial_random"]:
        result["effects"][f"high_residual_minus_{kind}"] = high - statistics.mean(score(f"hybrid/{kind}_{seed}") for seed in SEEDS)
        result["effects"][f"nll_gain_vs_{kind}"] = statistics.mean(lookup[f"hybrid/{kind}_{seed}"]["candidate_answer_nll"] for seed in SEEDS) - lookup["hybrid/high_residual"]["candidate_answer_nll"]
    for control in ["low_residual", "attention", "pca_residual", "diversity"]:
        result["effects"][f"high_residual_minus_{control}"] = high - score(f"hybrid/{control}")
    result["effects"]["high_residual_minus_dense"] = high - score("dense")
    result["effects"]["mean_minus_representative_same_positions"] = score("reference/feature_mean_native128") - score("reference/feature_representative128")
    result["effects"]["centroid_minus_native_same_values"] = score("reference/feature_mean_centroid128") - score("reference/feature_mean_native128")
    for fraction in [.05, .10]:
        prefix = f"spectral/{fraction:g}/"
        low, high_frequency = score(prefix + "low"), score(prefix + "high")
        random_score = statistics.mean(score(prefix + f"random_{seed}") for seed in SEEDS)
        doses = [lookup[prefix + name]["frequency_trace"]["actual_ac_fraction"] for name in ["low", "high", "random_11", "random_23", "random_47"]]
        if max(doses) - min(doses) > 1e-8:
            raise ValueError("Requested spectral conditions used different intervention energy.")
        result["spectral"][str(fraction)] = {"high_minus_low_damage": low - high_frequency,
            "high_minus_random_damage": random_score - high_frequency, "actual_ac_fraction": doses[0],
            "high_damage": score("dense") - high_frequency, "low_damage": score("dense") - low,
            "requested_dose_capped": doses[0] < fraction - 1e-8}
    tiles = [lookup[f"tile/{i}"] for i in range(16)]
    for score_name in ["tile_residual_score", "tile_pca_score", "tile_attention_score", "tile_feature_norm"]:
        scores = [r[score_name] for r in tiles]
        result["correlations"][score_name] = {
            "nll_damage": spearman_correlation(scores, [r["candidate_answer_nll"] - lookup["dense"]["candidate_answer_nll"] for r in tiles]),
            "score_damage": spearman_correlation(scores, [score("dense") - r["primary_task_score"] for r in tiles])}
    return result, lookup


def report(root, stage):
    cohort = json.loads((root / "cohort.json").read_text())
    entries = stage_entries(cohort, stage)
    summaries = [json.loads(path.read_text()) for path in sorted((root / "samples" / stage).glob("*/summary.json"))]
    expected = {row["sample_id"] for row in entries}
    completed = {s["sample_id"] for s in summaries}
    result = {"stage": stage, "expected_images": len(expected), "completed_images": len(summaries),
              "status": "complete" if completed == expected and len(completed) == len(summaries) else "incomplete",
              "inference_permitted": False}
    if result["status"] != "complete":
        atomic_write_json(root / f"{stage}_report.json", result, ensure_ascii=False)
        return result
    if len({s["image_fingerprint"] for s in summaries}) != len(summaries) or len({s["implementation_hash"] for s in summaries}) != 1:
        raise ValueError("Duplicate images or mixed implementations in report.")
    fingerprints = {row["sample_id"]: row["image_fingerprint"] for row in entries}
    if any(s["image_fingerprint"] != fingerprints[s["sample_id"]] for s in summaries):
        raise ValueError("Cohort fingerprint mismatch.")
    grouped, conditions = defaultdict(list), defaultdict(list)
    for summary in summaries:
        metrics, lookup = image_metrics(summary)
        grouped[summary["task"]].append(metrics)
        conditions[summary["task"]].append(lookup)
    tables, tests = [], []
    for task, images in sorted(grouped.items()):
        effects = {key: bootstrap([row["effects"][key] for row in images]) for key in images[0]["effects"]}
        correlations = {}
        for scorer in images[0]["correlations"]:
            correlations[scorer] = {}
            for damage in ["nll_damage", "score_damage"]:
                valid = [r["correlations"][scorer][damage] for r in images if r["correlations"][scorer][damage] is not None]
                correlations[scorer][damage] = {"defined_images": len(valid), "undefined_constant_images": len(images) - len(valid),
                    "mean_rho": bootstrap(valid) if valid else None}
        frequencies = {str(fraction): {key: bootstrap([row["spectral"][str(fraction)][key] for row in images])
            for key in ["high_minus_low_damage", "high_minus_random_damage", "high_damage", "low_damage", "actual_ac_fraction", "requested_dose_capped"]}
            for fraction in [.05, .10]}
        refs = {}
        for name in sorted(conditions[task][0]):
            values = [lookup[name]["primary_task_score"] for lookup in conditions[task]]
            strict = [lookup[name]["strict_answer_match"] for lookup in conditions[task]]
            refs[name] = {"mean_score": statistics.mean(values), "n_images": len(values),
                          "paired_delta_vs_dense": bootstrap([v - image["dense_score"] for v, image in zip(values, images)]),
                          "mean_nll": statistics.mean(lookup[name]["candidate_answer_nll"] for lookup in conditions[task]),
                          "mean_tokens": statistics.mean(lookup[name]["retained_visual_tokens"] for lookup in conditions[task]),
                          "strict_score": statistics.mean(strict) if all(v is not None for v in strict) else None}
        subgroups = {}
        for label, predicate in [("dense_zero", lambda r: r["dense_score"] == 0), ("dense_partial", lambda r: 0 < r["dense_score"] < 1),
                                 ("dense_full", lambda r: r["dense_score"] == 1), ("hybrid_loses_score", lambda r: r["high_residual_score"] < r["dense_score"])]:
            subset = [row for row in images if predicate(row)]
            if subset:
                subgroups[label] = {"n_images": len(subset), "high_minus_spatial_random": bootstrap([r["effects"]["high_residual_minus_spatial_random"] for r in subset])}
        tables.append({"task": task, "n_images": len(images), "effects": effects, "spectral": frequencies,
                       "within_image_correlations": correlations, "conditions": refs, "subgroups": subgroups})
        if stage == "confirmation":
            for key in ["high_residual_minus_random", "high_residual_minus_spatial_random"]:
                tests.append({"task": task, "endpoint": key, **effects[key]})
    if tests:
        if len(tests) != 4 or len(summaries) != 800:
            raise ValueError("Frozen confirmation requires 800 images and four tests.")
        result["inference_permitted"] = True
        for test, adjusted in zip(tests, holm([t["one_sided_p"] for t in tests])):
            test["holm_adjusted_p"] = adjusted
            test["supported"] = adjusted < .05 and test["ci95"][0] > 0
            test["practical_gain_met"] = test["mean"] >= .02
        by_task = {task: all(t["supported"] and t["practical_gain_met"] for t in tests if t["task"] == task) for task in grouped}
        result["decision"] = "task_utility_supported_both_tasks" if all(by_task.values()) else "task_specific" if any(by_task.values()) else "no_stable_practical_advantage_established"
        result["fallback_recommendation"] = "test_minimal_trained_hybrid_with_matched_training_controls" if not any(by_task.values()) else "study_supported_task_before_expanding_architecture"
    result.update({"tables": tables, "confirmatory_tests": tests, "implementation_hash": summaries[0]["implementation_hash"],
                   "elapsed_s_sum": sum(s["elapsed_s"] for s in summaries), "elapsed_s_median": statistics.median(s["elapsed_s"] for s in summaries),
                   "interpretation": "Untrained grid reconstruction proxy; no claim of rejecting learned reconstruction compressors. Frequency effects are secondary and length preserving."})
    atomic_write_json(root / f"{stage}_report.json", result, ensure_ascii=False)
    atomic_write_json(root / f"{stage}_image_metrics.json", dict(grouped), ensure_ascii=False)
    lines = [f"# Reconstruction utility: {stage}", "", f"Independent images: {len(summaries)}/{len(expected)}", "",
             f"Decision: {result.get('decision', 'development / implementation evidence only')}", "", result["interpretation"], "",
             "| Task | Method | Native score | Strict score | Visual tokens |", "|---|---|---:|---:|---:|"]
    for table in tables:
        for name, metrics in table["conditions"].items():
            if name.startswith(("spectral/", "tile/")):
                continue
            strict = "—" if metrics["strict_score"] is None else f"{metrics['strict_score']:.4f}"
            lines.append(f"| {table['task']} | {name} | {metrics['mean_score']:.4f} | {strict} | {metrics['mean_tokens']:.1f} |")
    lines.extend(["", "## Paired task-score effects", "", "| Task | Contrast | Mean | Image bootstrap 95% CI |", "|---|---|---:|---|"])
    for table in tables:
        for key, metrics in table["effects"].items():
            lines.append(f"| {table['task']} | {key} | {metrics['mean']:.5f} | [{metrics['ci95'][0]:.5f}, {metrics['ci95'][1]:.5f}] |")
    lines.extend(["", "## Confirmation tests", "", *[json.dumps(test, ensure_ascii=False) for test in tests], "",
                  "All random seeds are averaged within image. Tile correlations use 16 interventions per image and report undefined constant cases. Complete spectra/doses/correlations are in the JSON report.",
                  "Dense Layer-0 attention and exact SVD are diagnostic controls; this run does not establish deployable inference speedup."])
    (root / f"{stage}_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--stage", required=True)
    args = parser.parse_args()
    result = report(args.root.resolve(), args.stage)
    print(json.dumps({k: result.get(k) for k in ["stage", "status", "completed_images", "expected_images", "decision"]}))
