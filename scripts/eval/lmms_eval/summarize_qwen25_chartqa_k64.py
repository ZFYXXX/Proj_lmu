#!/usr/bin/env python3
"""Summarize the full Qwen2.5-VL-3B ChartQA fixed-budget comparison."""

from __future__ import annotations

import argparse
import glob
import json
import re
import statistics
from pathlib import Path
from typing import Any

import numpy as np


METHODS = ("baseline", "feature_anchor_mean_k64", "grid_mean_k64", "prumerge_plus_k64", "bifocal")
K64_METHODS = METHODS[1:4]
N = 2500


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def sample_file(run_dir: Path) -> Path:
    files = sorted(Path(item) for item in glob.glob(str(run_dir / "models__Qwen2.5-VL-3B-Instruct" / "*_samples_*.jsonl")))
    if len(files) != 1:
        raise RuntimeError(f"Expected one sample JSONL under {run_dir}, found {files}")
    return files[0]


def elapsed_seconds(path: Path) -> float | None:
    if not path.exists():
        return None
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        match = re.search(r"Elapsed \(wall clock\) time .*?\):\s*(\S+)", line)
        if not match:
            continue
        fields = [float(value) for value in match.group(1).split(":")]
        if len(fields) == 2:
            return 60 * fields[0] + fields[1]
        if len(fields) == 3:
            return 3600 * fields[0] + 60 * fields[1] + fields[2]
        return fields[0]
    return None


def peak_memory_mb(path: Path) -> float | None:
    if not path.exists():
        return None
    values = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        fields = line.split(",")
        if len(fields) > 1:
            try:
                values.append(float(fields[1].strip()))
            except ValueError:
                pass
    return max(values) if values else None


def bootstrap_ci(baseline: np.ndarray, candidate: np.ndarray, seed: int) -> list[float]:
    rng = np.random.default_rng(seed)
    diffs = candidate - baseline
    indices = rng.integers(0, len(diffs), size=(10_000, len(diffs)))
    draws = diffs[indices].mean(axis=1) * 100
    return [float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))]


def summarize(root: Path) -> dict[str, Any]:
    samples: dict[str, list[dict[str, Any]]] = {}
    traces: dict[str, list[dict[str, Any]]] = {}
    results: dict[str, Any] = {}
    for method in METHODS:
        run_dir = root / method
        result_path = run_dir / "result.json"
        result = json.loads(result_path.read_text(encoding="utf-8"))
        if result.get("status") != "succeeded":
            raise RuntimeError(f"Run {method} did not succeed: {result.get('error')}")
        metric_rows = result["output"]["metrics"]
        task_metrics = next(row["metrics"] for row in metric_rows if row.get("task") == "qwen25_vl_chartqa_full")
        if int(next(row["num_samples"] for row in metric_rows if row.get("task") == "qwen25_vl_chartqa_full")) != N:
            raise RuntimeError(f"Run {method} is not a full {N}-sample ChartQA evaluation")
        rows = read_jsonl(sample_file(run_dir))
        if len(rows) != N:
            raise RuntimeError(f"Run {method} has {len(rows)} sample rows, expected {N}")
        samples[method] = rows
        time_file = root / "logs" / f"{method}.time.txt"
        gpu_file = root / "logs" / f"{method}.gpu.csv"
        elapsed = elapsed_seconds(time_file)
        results[method] = {
            "relaxed_overall": float(task_metrics["relaxed_overall,none"]),
            "relaxed_human_split": task_metrics.get("relaxed_human_split,none"),
            "relaxed_augmented_split": task_metrics.get("relaxed_augmented_split,none"),
            "num_samples": N,
            "elapsed_seconds": elapsed,
            "samples_per_second": None if elapsed is None else N / elapsed,
            "peak_gpu_memory_mb_sampled": peak_memory_mb(gpu_file),
        }
        trace_path = run_dir / "compression_runtime.requests.jsonl"
        traces[method] = read_jsonl(trace_path) if trace_path.exists() else []

    base_rows = samples["baseline"]
    base_keys = [(row["doc_id"], row["doc_hash"]) for row in base_rows]
    base_scores = np.asarray([float(row["relaxed_overall"]) for row in base_rows])
    scores = {"baseline": base_scores}
    for index, method in enumerate(METHODS[1:], start=1):
        rows = samples[method]
        if [(row["doc_id"], row["doc_hash"]) for row in rows] != base_keys:
            raise RuntimeError(f"Sample order/content differs between baseline and {method}")
        values = np.asarray([float(row["relaxed_overall"]) for row in rows])
        scores[method] = values
        wins = int(np.sum(values > base_scores))
        ties = int(np.sum(values == base_scores))
        losses = N - wins - ties
        results[method].update({
            "delta_vs_baseline_pp": float((values.mean() - base_scores.mean()) * 100),
            "paired_bootstrap_95ci_pp": bootstrap_ci(base_scores, values, 42 + index),
            "paired_wins": wins,
            "paired_ties": ties,
            "paired_losses": losses,
        })

    source_counts: dict[str, list[int]] = {}
    for method in K64_METHODS:
        rows = traces[method]
        if len(rows) != N:
            raise RuntimeError(f"Missing per-request compression trace for {method}")
        counts, source, decoded = [], [], []
        for row in rows:
            activation = row["runtime"]["activations"][0]
            layout = activation["latest_layout_delta"]
            counts.append(int(layout["result_count"]))
            source.append(int(layout["source_count"]))
            decoded.append(activation.get("projector_generate", {}).get("last", {}).get("decode_position_verified"))
            if method == "prumerge_plus_k64":
                record = activation.get("layout_delta_records", [{}])[-1]
                if not {"prumerge_center_indices", "prumerge_neighbour_indices", "prumerge_global_count"}.issubset(record):
                    raise RuntimeError("PruMerge+ evidence missing from a ChartQA trace")
        if set(counts) != {64} or decoded.count(True) != N:
            raise RuntimeError(f"K64 token count or decode-position verification failed for {method}")
        source_counts[method] = source
        results[method].update({
            "visual_tokens_min": min(counts), "visual_tokens_max": max(counts), "visual_tokens_mean": statistics.mean(counts),
            "source_visual_tokens_mean": statistics.mean(source), "decode_position_verified": decoded.count(True),
        })
    if any(counts != source_counts[K64_METHODS[0]] for counts in source_counts.values()):
        raise RuntimeError("K64 methods did not receive the same original visual-token counts")

    bifocal_trace = traces["bifocal"]
    if len(bifocal_trace) != N:
        raise RuntimeError("Missing Bifocal per-request trace")
    bifocal_counts = [int(row["runtime"]["activations"][0]["latest_layout_delta"]["result_count"]) for row in bifocal_trace]
    results["bifocal"].update({
        "visual_tokens_min": min(bifocal_counts), "visual_tokens_max": max(bifocal_counts),
        "visual_tokens_mean": statistics.mean(bifocal_counts),
        "source_visual_tokens_mean": statistics.mean(source_counts[K64_METHODS[0]]),
    })
    results["baseline"].update({
        "visual_tokens_mean": statistics.mean(source_counts[K64_METHODS[0]]),
        "source_visual_tokens_mean": statistics.mean(source_counts[K64_METHODS[0]]),
    })

    paired: dict[str, Any] = {}
    for left in METHODS[1:]:
        paired[left] = {}
        for right in METHODS[1:]:
            paired[left][right] = {
                "left_wins": int(np.sum(scores[left] > scores[right])),
                "ties": int(np.sum(scores[left] == scores[right])),
                "right_wins": int(np.sum(scores[left] < scores[right])),
            }

    bins = ((0, 256), (257, 320), (321, 384), (385, 10_000))
    binned = {}
    for lower, upper in bins:
        indices = [i for i, count in enumerate(source_counts[K64_METHODS[0]]) if lower <= count <= upper]
        binned[f"{lower}-{upper}"] = {
            "num_samples": len(indices),
            **{method: float(scores[method][indices].mean()) if indices else None for method in METHODS},
        }

    return {
        "dataset": "ChartQA test", "num_samples": N,
        "checkpoint": "/home/ZhouFengyu/fastv-repro/models/Qwen2.5-VL-3B-Instruct",
        "revision": "1b5a7674d40a0ac9acef541a46fed12d710bd36b", "seed": 42,
        "primary_metric": "relaxed_overall", "methods": results,
        "source_visual_token_bins": binned, "paired_comparisons": paired,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    args = parser.parse_args()
    report = summarize(args.run_root)
    (args.run_root / "summary.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    lines = [
        "# Qwen2.5-VL-3B ChartQA full comparison", "",
        f"Samples: {N}  ", f"Primary metric: `{report['primary_metric']}`  ",
        f"Checkpoint: `{report['checkpoint']}`  ", f"Revision: `{report['revision']}`  ", f"Seed: {report['seed']}", "",
        "| Method | Relaxed overall | Δ vs baseline | Paired 95% CI | Mean visual tokens | Wall time | Samples/s | Sampled peak GPU memory |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for method in METHODS:
        row = report["methods"][method]
        delta, ci = row.get("delta_vs_baseline_pp"), row.get("paired_bootstrap_95ci_pp")
        elapsed, speed, memory = row["elapsed_seconds"], row["samples_per_second"], row["peak_gpu_memory_mb_sampled"]
        lines.append("| {} | {:.2%} | {} | {} | {} | {} | {} | {} |".format(
            method, row["relaxed_overall"], "—" if delta is None else f"{delta:+.2f} pp",
            "—" if ci is None else f"[{ci[0]:+.2f}, {ci[1]:+.2f}] pp",
            "—" if "visual_tokens_mean" not in row else f"{row['visual_tokens_mean']:.0f}",
            "—" if elapsed is None else f"{elapsed / 60:.1f} min",
            "—" if speed is None else f"{speed:.2f}",
            "—" if memory is None else f"{memory / 1024:.1f} GiB sampled",
        ))
    lines.extend([
        "",
        "| Method | Human split | Augmented split |",
        "|---|---:|---:|",
    ])
    for method in METHODS:
        row = report["methods"][method]
        human, augmented = row["relaxed_human_split"], row["relaxed_augmented_split"]
        lines.append(f"| {method} | {'—' if human is None else f'{human:.2%}'} | {'—' if augmented is None else f'{augmented:.2%}'} |")
    lines.extend(["", "Bifocal has variable token counts and is not part of the fixed-K=64 ranking.", "Feature-anchor is the untrained Stage 1 proxy, not a trained Projector.", ""])
    (args.run_root / "report.md").write_text("\n".join(lines), encoding="utf-8")
    print(args.run_root / "report.md")


if __name__ == "__main__":
    main()
