#!/usr/bin/env python3
"""Summarize 16-token spatial-block vs matched-random interventions."""

from __future__ import annotations

import argparse
import json
import random
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Sequence


GROUPS = ("spatial_block", "matched_random")
INTERVENTIONS = (
    "direction_collapse",
    "position_permutation",
    "direction_collapse_plus_position_permutation",
)


def ci(values: Sequence[float], *, seed: int = 20261021, draws: int = 10000) -> tuple[float, float, float]:
    rng = random.Random(seed)
    means = sorted(sum(values[rng.randrange(len(values))] for _ in values) / len(values) for _ in range(draws))
    return statistics.mean(values), means[int(0.025 * draws)], means[int(0.975 * draws)]


def metric(record: dict, group: str, intervention: str, key: str) -> float:
    row = next(
        item for item in record["group_geometry_interventions"]["conditions"]
        if item["group"] == group and item["intervention"] == intervention
    )
    return float(row[key])


def fmt(values: Sequence[float]) -> str:
    mean, low, high = ci(values)
    return f"{mean:.5f} [{low:.5f}, {high:.5f}]"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("samples", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    records = [json.loads(line) for line in args.samples.read_text(encoding="utf-8").splitlines() if line.strip()]
    records = [r for r in records if r.get("status") == "ok" and r.get("group_geometry_interventions", {}).get("status") == "ok"]
    if not records:
        raise SystemExit("No successful group geometry records found.")
    tasks = defaultdict(list)
    for record in records:
        tasks[record["task_name"]].append(record)

    lines = [
        "# Group geometry discovery pilot",
        "",
        f"Samples: **{len(records)}**; tasks: {', '.join(f'{task}={len(rows)}' for task, rows in sorted(tasks.items()))}.",
        "",
        "Frozen Qwen2.5-VL-3B. Per image, one 16-token contiguous spatial block and one 16-token random group from the same 20-token canvas were selected without answers. The random group was chosen to match feature-norm mean/std, mean pairwise cosine, and bounding-box span. Each group received norm-preserving feature-direction collapse, content↔M-RoPE position permutation, and both together. Sequence length, mask, and text positions were unchanged.",
        "",
        "## Answer-NLL effects",
        "",
        "Entries are mean gold-answer NLL delta with image-bootstrap 95% CI; positive means the gold answer became less likely.",
        "",
        "| Group | Intervention | NLL delta, mean [95% CI] | Teacher KL, mean [95% CI] | Top-1 changes | Answer flips |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for group in GROUPS:
        for intervention in INTERVENTIONS:
            vals = [metric(r, group, intervention, "answer_nll_delta") for r in records]
            kl = [metric(r, group, intervention, "teacher_to_candidate_answer_kl") for r in records]
            changed = sum(metric(r, group, intervention, "teacher_top1_agreement") < 1 for r in records)
            flips = sum(bool(next(
                item for item in r["group_geometry_interventions"]["conditions"]
                if item["group"] == group and item["intervention"] == intervention
            )["answer_flip"]) for r in records)
            lines.append(f"| {group} | `{intervention}` | {fmt(vals)} | {fmt(kl)} | {changed}/{len(records)} | {flips}/{len(records)} |")

    lines += ["", "## Paired spatial-block minus matched-random contrasts", ""]
    for intervention in INTERVENTIONS:
        delta = [
            metric(r, "spatial_block", intervention, "answer_nll_delta")
            - metric(r, "matched_random", intervention, "answer_nll_delta")
            for r in records
        ]
        lines.append(f"- `{intervention}` NLL difference: **{fmt(delta)}**.")
    for intervention in INTERVENTIONS:
        delta = [
            metric(r, "spatial_block", intervention, "teacher_to_candidate_answer_kl")
            - metric(r, "matched_random", intervention, "teacher_to_candidate_answer_kl")
            for r in records
        ]
        lines.append(f"- `{intervention}` teacher-KL difference: **{fmt(delta)}**.")

    lines += ["", "## Task-level mean NLL deltas", "", "| Task | Block: direction | Random: direction | Block: position | Random: position | Block: combined | Random: combined |", "|---|---:|---:|---:|---:|---:|---:|"]
    for task, rows in sorted(tasks.items()):
        vals = [statistics.mean(metric(r, group, intervention, "answer_nll_delta") for r in rows)
                for group, intervention in (("spatial_block", "direction_collapse"), ("matched_random", "direction_collapse"),
                                            ("spatial_block", "position_permutation"), ("matched_random", "position_permutation"),
                                            ("spatial_block", "direction_collapse_plus_position_permutation"),
                                            ("matched_random", "direction_collapse_plus_position_permutation"))]
        lines.append(f"| {task} | " + " | ".join(f"{v:.5f}" for v in vals) + " |")

    match_scores = [float(r["group_geometry_interventions"]["matched_random_selection_score"]) for r in records]
    match_diffs = {}
    for key in ("norm_mean", "norm_std", "mean_pair_cosine", "bbox_height", "bbox_width"):
        match_diffs[key] = statistics.mean(
            abs(float(r["group_geometry_interventions"]["matched_random_statistics"][key])
                - float(r["group_geometry_interventions"]["spatial_block_statistics"][key]))
            for r in records
        )
    lines += [
        "",
        "## Interpretation",
        "",
        f"Mean random-control matching score was {statistics.mean(match_scores):.3f} (lower is a closer match). Average absolute block/control differences were: norm mean {match_diffs['norm_mean']:.3f}, norm std {match_diffs['norm_std']:.3f}, pairwise cosine {match_diffs['mean_pair_cosine']:.4f}, bbox height {match_diffs['bbox_height']:.2f} cells, bbox width {match_diffs['bbox_width']:.2f} cells. This is only approximate matching, especially for spatial span and feature norms, so block-vs-random contrasts remain confounded and exploratory. The effects are heterogeneous across tasks; paired confidence intervals matter more than separate means. NLL improvements (negative deltas) can occur when the model is already wrong and do not imply general benefit.",
        "",
        "This remains a mechanism pilot, not a compression or accuracy comparison. It tests changes at layer-0 inputs and does not demonstrate that a deployable compressor can predict which regions to preserve. A consistent block advantage would motivate spatially structured forgetting/retrieval; a consistent direction-collapse effect would motivate testing feature-direction redundancy. If intervals overlap zero and task signs disagree, the next step is larger held-out sampling and free-generation confirmation, not method design.",
        "",
    ]
    rendered = "\n".join(lines)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
