#!/usr/bin/env python3
"""Summarize the frozen-model pair geometry discovery pilot."""

from __future__ import annotations

import argparse
import json
import random
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Sequence


CONDITIONS = (
    "feature_mean",
    "feature_normmatched_mean",
    "position_centroid",
    "feature_normmatched_mean_plus_position_centroid",
)


def bootstrap_mean(values: Sequence[float], *, seed: int = 20261020, draws: int = 10000) -> tuple[float, float, float]:
    if not values:
        raise ValueError("Cannot bootstrap an empty set.")
    rng = random.Random(seed)
    means = sorted(
        sum(values[rng.randrange(len(values))] for _ in values) / len(values)
        for _ in range(draws)
    )
    return statistics.mean(values), means[int(0.025 * draws)], means[int(0.975 * draws)]


def condition_rows(records: Sequence[dict], condition: str) -> list[dict]:
    return [
        next(row for row in record["pair_geometry_interventions"]["conditions"] if row["condition"] == condition)
        for record in records
    ]


def fmt_ci(values: Sequence[float]) -> str:
    mean, low, high = bootstrap_mean(values)
    return f"{mean:.5f} [{low:.5f}, {high:.5f}]"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("samples", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    records = [json.loads(line) for line in args.samples.read_text(encoding="utf-8").splitlines() if line.strip()]
    records = [r for r in records if r.get("status") == "ok" and r.get("pair_geometry_interventions", {}).get("status") == "ok"]
    if not records:
        raise SystemExit("No successful pair geometry records found.")
    tasks = defaultdict(list)
    for record in records:
        tasks[record["task_name"]].append(record)

    lines = [
        "# Pair geometry discovery pilot",
        "",
        f"Samples: **{len(records)}**; tasks: {', '.join(f'{name}={len(rows)}' for name, rows in sorted(tasks.items()))}.",
        "",
        "Frozen Qwen2.5-VL-3B, one image/question per sample. A deterministic, answer-blind pair was chosen in one of four feature-similarity × grid-distance quadrants. Each intervention altered only those two visual positions; sequence length, mask, and text M-RoPE positions were held fixed. These are teacher-forced answer-token diagnostics, not end-task generation results.",
        "",
        "## Overall paired effects",
        "",
        "Mean answer NLL delta with image-bootstrap 95% interval. Positive values mean the intervention made the gold answer less likely. KL is teacher-to-candidate KL on answer predictors; top-1 changes compare candidate logits with the dense teacher.",
        "",
        "| Intervention | NLL delta, mean [95% CI] | Teacher KL, mean | Top-1 changes | Answer flips |",
        "|---|---:|---:|---:|---:|",
    ]
    for condition in CONDITIONS:
        rows = condition_rows(records, condition)
        lines.append(
            f"| `{condition}` | {fmt_ci([float(r['answer_nll_delta']) for r in rows])} | "
            f"{statistics.mean(float(r['teacher_to_candidate_answer_kl']) for r in rows):.6f} | "
            f"{sum(float(r['teacher_top1_agreement']) < 1 for r in rows)}/{len(rows)} | "
            f"{sum(bool(r['answer_flip']) for r in rows)}/{len(rows)} |"
        )

    lines += ["", "## Task-level NLL deltas", "", "| Task | n | Feature mean | Norm-matched mean | Position centroid | Combined |", "|---|---:|---:|---:|---:|---:|"]
    for task, task_records in sorted(tasks.items()):
        vals = []
        for condition in CONDITIONS:
            rows = condition_rows(task_records, condition)
            vals.append(f"{statistics.mean(float(r['answer_nll_delta']) for r in rows):.5f}")
        lines.append(f"| {task} | {len(task_records)} | " + " | ".join(vals) + " |")

    lines += ["", "## Sampling balance and counterfactual checks", ""]
    category_counts = Counter(r["pair_geometry_interventions"]["pair_category"] for r in records)
    lines.append(f"Pair categories: {dict(sorted(category_counts.items()))}.")
    lines.append(
        "All successful records report unchanged sequence length and text positions. "
        f"Observed answer predictor counts: {dict(sorted(Counter(r['baseline']['answer_token_count'] for r in records).items()))}."
    )

    nll = {c: [float(r["answer_nll_delta"]) for r in condition_rows(records, c)] for c in CONDITIONS}
    norm_vs_raw = [a - b for a, b in zip(nll["feature_normmatched_mean"], nll["feature_mean"])]
    position_vs_feature = [a - b for a, b in zip(nll["position_centroid"], nll["feature_normmatched_mean"])]
    interaction = [
        combo - feature - position
        for combo, feature, position in zip(
            nll["feature_normmatched_mean_plus_position_centroid"],
            nll["feature_normmatched_mean"],
            nll["position_centroid"],
        )
    ]
    lines += [
        "",
        "## Paired contrasts",
        "",
        "Bootstrap intervals below resample the same images jointly; contrasts are descriptive at this pilot size.",
        "",
        f"- Norm matching minus raw feature mean: `{fmt_ci(norm_vs_raw)}` NLL.",
        f"- Position centroid minus norm-matched feature mean: `{fmt_ci(position_vs_feature)}` NLL.",
        f"- Combined minus the sum of separate feature/position effects (interaction): `{fmt_ci(interaction)}` NLL.",
        "",
        "## Interpretation and limits",
        "",
        "This screen does **not** establish that feature direction or M-RoPE position is irrelevant: changing only two of hundreds of visual tokens is a deliberately local perturbation, and most examples have a single scored answer token. Near-zero effects therefore mean this intervention was too weak to distinguish mechanisms, not that broad compression is harmless. The small number of top-1 changes also makes accuracy conclusions inappropriate.",
        "",
        "The useful next experiment is a stronger but still length-preserving factorial perturbation: apply matched feature-direction collapse and content↔position swaps to spatially coherent groups of 8–16 tokens, compare against random groups matched on norm/cosine/span, and measure both answer-token NLL/KL and free-generation changes. Keep the current pilot as the control showing that isolated pair perturbations are largely silent.",
        "",
        "Pair choice and group construction must remain answer-blind; all confidence intervals are exploratory and should not be presented as confirmatory evidence.",
        "",
    ]
    rendered = "\n".join(lines)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
