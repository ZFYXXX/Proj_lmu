#!/usr/bin/env python3
"""Compare ChartQA quality across fixed visual-token budgets."""

from __future__ import annotations

import argparse
import glob
import json
import re
import statistics
from pathlib import Path
from typing import Any

import numpy as np


N = 2500
METHODS = (
    "baseline",
    "feature_anchor_mean_k64",
    "feature_anchor_mean_k128",
    "feature_anchor_mean_k256",
    "prumerge_plus_k64",
    "prumerge_plus_k128",
    "prumerge_plus_k256",
)
OLD_METHODS = {"baseline", "feature_anchor_mean_k64", "prumerge_plus_k64"}


def sample_file(run_dir: Path) -> Path:
    files = sorted(Path(path) for path in glob.glob(str(run_dir / "models__Qwen2.5-VL-3B-Instruct" / "*_samples_*.jsonl")))
    if len(files) != 1:
        raise RuntimeError(f"Expected one sample JSONL in {run_dir}, found {files}")
    return files[0]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def seconds(path: Path) -> float | None:
    if not path.exists():
        return None
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        match = re.search(r"Elapsed \(wall clock\) time .*?\):\s*(\S+)", line)
        if match:
            values = [float(x) for x in match.group(1).split(":")]
            if len(values) == 2:
                return values[0] * 60 + values[1]
            if len(values) == 3:
                return values[0] * 3600 + values[1] * 60 + values[2]
            return values[0]
    return None


def paired_ci(base: np.ndarray, current: np.ndarray, seed: int) -> list[float]:
    rng = np.random.default_rng(seed)
    diff = current - base
    draws = diff[rng.integers(0, len(diff), size=(10_000, len(diff)))].mean(axis=1) * 100
    return [float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))]


def summarize(root: Path, previous_root: Path) -> dict[str, Any]:
    rows_by_method: dict[str, list[dict[str, Any]]] = {}
    scores: dict[str, np.ndarray] = {}
    traces: dict[str, list[dict[str, Any]]] = {}
    result: dict[str, Any] = {}
    for index, method in enumerate(METHODS):
        source_root = previous_root if method in OLD_METHODS else root
        run_dir = source_root / method
        payload = json.loads((run_dir / "result.json").read_text(encoding="utf-8"))
        if payload.get("status") != "succeeded":
            raise RuntimeError(f"Run {method} did not succeed")
        metric_row = next(row for row in payload["output"]["metrics"] if row.get("task") == "qwen25_vl_chartqa_full")
        if int(metric_row["num_samples"]) != N:
            raise RuntimeError(f"{method} has {metric_row['num_samples']} samples, expected {N}")
        metric = metric_row["metrics"]
        path = sample_file(run_dir)
        samples = read_jsonl(path)
        if len(samples) != N:
            raise RuntimeError(f"{method} sample JSONL has {len(samples)} rows, expected {N}")
        rows_by_method[method] = samples
        scores[method] = np.asarray([float(row["relaxed_overall"]) for row in samples])
        traces_path = run_dir / "compression_runtime.requests.jsonl"
        traces[method] = read_jsonl(traces_path) if traces_path.exists() else []
        duration = seconds(source_root / "logs" / f"{method}.time.txt")
        result[method] = {
            "relaxed_overall": float(metric["relaxed_overall,none"]),
            "relaxed_human_split": metric.get("relaxed_human_split,none"),
            "relaxed_augmented_split": metric.get("relaxed_augmented_split,none"),
            "elapsed_seconds": duration,
            "samples_per_second": None if duration is None else N / duration,
            "num_samples": N,
        }

    expected_keys = [(row["doc_id"], row["doc_hash"]) for row in rows_by_method["baseline"]]
    for method in METHODS[1:]:
        if [(row["doc_id"], row["doc_hash"]) for row in rows_by_method[method]] != expected_keys:
            raise RuntimeError(f"Sample IDs/hashes differ for {method}")

    source_lists = {}
    for method in METHODS[1:]:
        rows = traces[method]
        if len(rows) != N:
            raise RuntimeError(f"Missing per-sample compression traces for {method}")
        target = int(method.rsplit("k", 1)[1])
        counts, sources, decoded = [], [], []
        for row in rows:
            activation = row["runtime"]["activations"][0]
            layout = activation["latest_layout_delta"]
            counts.append(int(layout["result_count"]))
            sources.append(int(layout["source_count"]))
            decoded.append(activation.get("projector_generate", {}).get("last", {}).get("decode_position_verified"))
            if method.startswith("prumerge"):
                record = activation.get("layout_delta_records", [{}])[-1]
                needed = {"prumerge_center_indices", "prumerge_neighbour_indices", "prumerge_global_count"}
                if not needed.issubset(record):
                    raise RuntimeError(f"Incomplete PruMerge+ evidence in {method}")
        if set(counts) != {target} or decoded.count(True) != N:
            raise RuntimeError(f"Unexpected visual budget or decode-position failure in {method}")
        source_lists[method] = sources
        result[method].update({
            "visual_tokens_min": min(counts), "visual_tokens_max": max(counts),
            "visual_tokens_mean": statistics.mean(counts),
            "source_visual_tokens_mean": statistics.mean(sources),
            "decode_position_verified": decoded.count(True),
        })
    reference_sources = source_lists["feature_anchor_mean_k64"]
    if any(value != reference_sources for value in source_lists.values()):
        raise RuntimeError("Raw visual-token counts differ across budget sweep runs")
    result["baseline"].update({"visual_tokens_mean": statistics.mean(reference_sources), "source_visual_tokens_mean": statistics.mean(reference_sources)})

    baseline = scores["baseline"]
    for index, method in enumerate(METHODS[1:], start=1):
        values = scores[method]
        result[method].update({
            "delta_vs_baseline_pp": float((values.mean() - baseline.mean()) * 100),
            "paired_bootstrap_95ci_pp": paired_ci(baseline, values, 42 + index),
            "paired_wins": int(np.sum(values > baseline)),
            "paired_ties": int(np.sum(values == baseline)),
            "paired_losses": int(np.sum(values < baseline)),
        })

    paired = {}
    for left in METHODS[1:]:
        paired[left] = {}
        for right in METHODS[1:]:
            paired[left][right] = {
                "left_wins": int(np.sum(scores[left] > scores[right])),
                "ties": int(np.sum(scores[left] == scores[right])),
                "right_wins": int(np.sum(scores[left] < scores[right])),
            }
    bins = ((257, 320), (321, 384), (385, 10_000))
    binned = {}
    for lower, upper in bins:
        idx = [i for i, count in enumerate(reference_sources) if lower <= count <= upper]
        binned[f"{lower}-{upper}"] = {
            "num_samples": len(idx),
            **{method: float(scores[method][idx].mean()) if idx else None for method in METHODS},
        }
    return {
        "dataset": "ChartQA test", "num_samples": N, "primary_metric": "relaxed_overall",
        "checkpoint": "/home/ZhouFengyu/fastv-repro/models/Qwen2.5-VL-3B-Instruct",
        "revision": "1b5a7674d40a0ac9acef541a46fed12d710bd36b", "seed": 42,
        "methods": result, "paired_comparisons": paired,
        "source_visual_token_bins": binned,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--previous-root", type=Path, required=True)
    args = parser.parse_args()
    report = summarize(args.run_root, args.previous_root)
    (args.run_root / "summary.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    lines = [
        "# Qwen2.5-VL-3B ChartQA fixed-budget sweep", "",
        f"Samples: {N}; primary metric: `{report['primary_metric']}`  ",
        f"Checkpoint: `{report['checkpoint']}`; revision: `{report['revision']}`; seed: {report['seed']}", "",
        "| Method | Relaxed overall | Δ vs baseline | Paired 95% CI | Mean visual tokens | Samples/s |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for method in METHODS:
        row = report["methods"][method]
        ci, delta = row.get("paired_bootstrap_95ci_pp"), row.get("delta_vs_baseline_pp")
        lines.append("| {} | {:.2%} | {} | {} | {} | {} |".format(
            method, row["relaxed_overall"], "—" if delta is None else f"{delta:+.2f} pp",
            "—" if ci is None else f"[{ci[0]:+.2f}, {ci[1]:+.2f}] pp",
            "—" if "visual_tokens_mean" not in row else f"{row['visual_tokens_mean']:.0f}",
            "—" if row["samples_per_second"] is None else f"{row['samples_per_second']:.2f}",
        ))
    lines.extend(["", "Untrained Stage 1 proxy; this sweep isolates budget within a compressor. PruMerge+ remains a comparison method.", ""])
    (args.run_root / "report.md").write_text("\n".join(lines), encoding="utf-8")
    print(args.run_root / "report.md")


if __name__ == "__main__":
    main()
