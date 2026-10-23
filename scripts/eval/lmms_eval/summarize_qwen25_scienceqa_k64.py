#!/usr/bin/env python3
"""Summarize the fixed-budget Qwen2.5 ScienceQA comparison."""

from __future__ import annotations

import argparse
import glob
import json
import random
import re
import statistics
import time
from pathlib import Path
from typing import Any


METHODS = (
    "baseline",
    "feature_anchor_mean_k64",
    "grid_mean_k64",
    "prumerge_plus_k64",
    "bifocal",
)
K64_METHODS = METHODS[1:4]


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _sample_file(directory: Path) -> Path:
    files = sorted(
        Path(path)
        for path in glob.glob(
            str(directory / "models__Qwen2.5-VL-3B-Instruct" / "*_samples_*.jsonl")
        )
    )
    if len(files) != 1:
        raise RuntimeError(f"Expected one sample JSONL under {directory}, found {files}")
    return files[0]


def _paired_bootstrap(
    baseline: list[float], compressed: list[float], *, seed: int, draws: int = 10_000
) -> tuple[float, float]:
    rng = random.Random(seed)
    count = len(baseline)
    differences = []
    for _ in range(draws):
        indices = [rng.randrange(count) for _ in range(count)]
        differences.append(
            100.0
            * (
                sum(compressed[index] - baseline[index] for index in indices)
                / count
            )
        )
    differences.sort()
    return differences[int(0.025 * draws)], differences[int(0.975 * draws)]


def _elapsed_seconds(path: Path) -> float | None:
    if not path.exists():
        return None
    text = path.read_text(encoding="utf-8", errors="replace")
    for line in text.splitlines():
        match = re.search(r"Elapsed \(wall clock\) time .*?\):\s*(\S+)", line)
        if match:
            raw = match.group(1)
            fields = [float(value) for value in raw.split(":")]
            if len(fields) == 1:
                return fields[0]
            if len(fields) == 2:  # GNU time's m:ss format
                return 60.0 * fields[0] + fields[1]
            if len(fields) == 3:  # GNU time's h:mm:ss format
                return 3600.0 * fields[0] + 60.0 * fields[1] + fields[2]
            raise ValueError(f"Unexpected GNU time duration: {raw}")
    return None


def _peak_gpu_memory_mb(path: Path) -> float | None:
    if not path.exists():
        return None
    values = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        fields = line.split(",")
        if len(fields) >= 2:
            try:
                values.append(float(fields[1].strip()))
            except ValueError:
                continue
    return max(values) if values else None


def summarize(root: Path) -> dict[str, Any]:
    results: dict[str, Any] = {}
    samples: dict[str, list[dict[str, Any]]] = {}
    traces: dict[str, list[dict[str, Any]]] = {}
    for method in METHODS:
        run_dir = root / method
        result_path = run_dir / "result.json"
        if not result_path.exists():
            raise RuntimeError(f"Missing result for {method}: {result_path}")
        result = json.loads(result_path.read_text(encoding="utf-8"))
        if result.get("status") != "succeeded":
            raise RuntimeError(f"Run {method} did not succeed: {result.get('error')}")
        metric = result["output"]["metrics"][0]
        rows = _read_jsonl(_sample_file(run_dir))
        if len(rows) != 2017 or int(metric["num_samples"]) != 2017:
            raise RuntimeError(f"Run {method} has {len(rows)} rows, expected 2017")
        samples[method] = rows
        results[method] = {
            "accuracy": float(metric["metrics"]["exact_match,none"]),
            "num_samples": len(rows),
            "elapsed_seconds": _elapsed_seconds(root / "logs" / f"{method}.time.txt"),
            "gpu_samples_path": str(root / "logs" / f"{method}.gpu.csv"),
            "peak_gpu_memory_mb": _peak_gpu_memory_mb(root / "logs" / f"{method}.gpu.csv"),
        }
        trace_path = run_dir / "compression_runtime.requests.jsonl"
        traces[method] = _read_jsonl(trace_path) if trace_path.exists() else []

    baseline = samples["baseline"]
    baseline_keys = [(row["doc_id"], row["doc_hash"]) for row in baseline]
    baseline_correct = [float(row["exact_match"]) for row in baseline]
    for method in METHODS[1:]:
        rows = samples[method]
        keys = [(row["doc_id"], row["doc_hash"]) for row in rows]
        if keys != baseline_keys:
            raise RuntimeError(f"Sample order/content differs between baseline and {method}")
        correct = [float(row["exact_match"]) for row in rows]
        lo, hi = _paired_bootstrap(
            baseline_correct, correct, seed=42 + METHODS.index(method)
        )
        wins = sum(c > b for b, c in zip(baseline_correct, correct, strict=True))
        ties = sum(c == b for b, c in zip(baseline_correct, correct, strict=True))
        losses = len(rows) - wins - ties
        results[method].update(
            {
                "delta_vs_baseline_pp": 100.0 * (statistics.mean(correct) - statistics.mean(baseline_correct)),
                "paired_bootstrap_95ci_pp": [lo, hi],
                "paired_wins": wins,
                "paired_ties": ties,
                "paired_losses": losses,
            }
        )

    for method in K64_METHODS:
        rows = traces.get(method, [])
        if len(rows) != 2017:
            raise RuntimeError(f"Missing per-request compression trace for {method}")
        counts = []
        decoded = []
        source_counts = []
        for row in rows:
            activation = row["runtime"]["activations"][0]
            layout = activation["latest_layout_delta"]
            source_counts.append(int(layout["source_count"]))
            counts.append(int(layout["result_count"]))
            state = activation.get("projector_generate", {}).get("last", {})
            decoded.append(state.get("decode_position_verified"))
            if method == "prumerge_plus_k64":
                record = activation.get("layout_delta_records", [{}])[-1]
                required = {
                    "prumerge_center_indices",
                    "prumerge_neighbour_indices",
                    "prumerge_global_count",
                }
                if not required.issubset(record):
                    raise RuntimeError("PruMerge+ trace is missing center/neighbour/global-token evidence")
        if set(counts) != {64}:
            raise RuntimeError(f"{method} did not emit exactly K=64 for every sample")
        if decoded.count(True) != 2017:
            raise RuntimeError(f"{method} decode M-RoPE check failed or was absent")
        results[method].update(
            {
                "visual_tokens_min": min(counts),
                "visual_tokens_max": max(counts),
                "visual_tokens_mean": statistics.mean(counts),
                "source_visual_tokens_min": min(source_counts),
                "source_visual_tokens_max": max(source_counts),
                "source_visual_tokens_mean": statistics.mean(source_counts),
                "decode_position_verified": decoded.count(True),
            }
        )

    # The raw token distribution is method-independent; check that all three
    # compressors saw exactly the same pre-compression visual sequence.
    source_by_method: dict[str, list[int]] = {}
    for method in K64_METHODS:
        source_by_method[method] = [
            int(row["runtime"]["activations"][0]["latest_layout_delta"]["source_count"])
            for row in traces[method]
        ]
    reference_source_counts = source_by_method["feature_anchor_mean_k64"]
    for method, counts in source_by_method.items():
        if counts != reference_source_counts:
            raise RuntimeError(f"Raw visual-token counts differ between feature-anchor and {method}")
    results["baseline"].update(
        {
            "source_visual_tokens_min": min(reference_source_counts),
            "source_visual_tokens_max": max(reference_source_counts),
            "source_visual_tokens_mean": statistics.mean(reference_source_counts),
            "visual_tokens_min": min(reference_source_counts),
            "visual_tokens_max": max(reference_source_counts),
            "visual_tokens_mean": statistics.mean(reference_source_counts),
        }
    )
    bifocal_trace = traces.get("bifocal", [])
    if len(bifocal_trace) != 2017:
        raise RuntimeError("Missing per-request Bifocal compression trace")
    bifocal_visual_counts = [
        int(row["runtime"]["activations"][0]["latest_layout_delta"]["result_count"])
        for row in bifocal_trace
    ]
    results["bifocal"].update(
        {
            "visual_tokens_min": min(bifocal_visual_counts),
            "visual_tokens_max": max(bifocal_visual_counts),
            "visual_tokens_mean": statistics.mean(bifocal_visual_counts),
            "source_visual_tokens_min": min(reference_source_counts),
            "source_visual_tokens_max": max(reference_source_counts),
            "source_visual_tokens_mean": statistics.mean(reference_source_counts),
        }
    )
    for method, row in results.items():
        elapsed = row["elapsed_seconds"]
        row["samples_per_second"] = None if elapsed is None else row["num_samples"] / elapsed

    bins = ((0, 256), (257, 320), (321, 384), (385, 10_000))
    binned: dict[str, Any] = {}
    for lower, upper in bins:
        label = f"{lower}-{upper}"
        indices = [
            index
            for index, trace in enumerate(traces["feature_anchor_mean_k64"])
            if lower <= int(trace["runtime"]["activations"][0]["latest_layout_delta"]["source_count"]) <= upper
        ]
        binned[label] = {"num_samples": len(indices)}
        for method in METHODS:
            values = [float(samples[method][index]["exact_match"]) for index in indices]
            binned[label][method] = sum(values) / len(values) if values else None

    comparative: dict[str, Any] = {}
    for left in METHODS[1:]:
        left_ok = [float(row["exact_match"]) for row in samples[left]]
        comparative[left] = {}
        for right in METHODS[1:]:
            right_ok = [float(row["exact_match"]) for row in samples[right]]
            comparative[left][right] = {
                "left_wins": sum(a > b for a, b in zip(left_ok, right_ok, strict=True)),
                "ties": sum(a == b for a, b in zip(left_ok, right_ok, strict=True)),
                "right_wins": sum(a < b for a, b in zip(left_ok, right_ok, strict=True)),
            }

    return {
        "dataset": "ScienceQA-IMG test",
        "num_samples": 2017,
        "checkpoint": "/home/ZhouFengyu/fastv-repro/models/Qwen2.5-VL-3B-Instruct",
        "revision": "1b5a7674d40a0ac9acef541a46fed12d710bd36b",
        "seed": 42,
        "methods": results,
        "source_visual_token_bins": binned,
        "paired_comparisons": comparative,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    args = parser.parse_args()
    report = summarize(args.run_root)
    (args.run_root / "summary.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    lines = [
        "# Qwen2.5-VL-3B ScienceQA-IMG full comparison",
        "",
        f"Samples: {report['num_samples']}  ",
        f"Checkpoint: `{report['checkpoint']}`  ",
        f"Revision: `{report['revision']}`  ",
        f"Seed: {report['seed']}",
        "",
        "| Method | Exact match | Δ vs baseline | Paired 95% CI | Mean visual tokens | Wall time | Samples/s | Peak GPU memory |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for method in METHODS:
        row = report["methods"][method]
        delta = row.get("delta_vs_baseline_pp")
        ci = row.get("paired_bootstrap_95ci_pp")
        lines.append(
            "| {method} | {accuracy:.2%} | {delta} | {ci} | {tokens} | {elapsed} | {throughput} | {memory} |".format(
                method=method,
                accuracy=row["accuracy"],
                delta=("—" if delta is None else f"{delta:+.2f} pp"),
                ci=("—" if ci is None else f"[{ci[0]:+.2f}, {ci[1]:+.2f}] pp"),
                tokens=("—" if "visual_tokens_mean" not in row else f"{row['visual_tokens_mean']:.0f}"),
                elapsed=("—" if row["elapsed_seconds"] is None else f"{row['elapsed_seconds'] / 60:.1f} min"),
                throughput=("—" if row["samples_per_second"] is None else f"{row['samples_per_second']:.2f}"),
                memory=("—" if row["peak_gpu_memory_mb"] is None else f"{row['peak_gpu_memory_mb'] / 1024:.1f} GiB sampled"),
            )
        )
    lines.extend(["", "## Accuracy by original visual-token count", ""])
    lines.append("| Source-token bin | N | Baseline | Feature-anchor | Grid | PruMerge+ | Bifocal |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for label, row in report["source_visual_token_bins"].items():
        fields = [label, str(row["num_samples"])]
        fields.extend("—" if row[method] is None else f"{row[method]:.2%}" for method in METHODS)
        lines.append("| " + " | ".join(fields) + " |")
    lines.extend([
        "",
        "## Interpretation",
        "",
        "Feature-anchor is the untrained Stage 1 mean-pooling proxy, not a trained Projector.",
        "Bifocal uses a variable visual-token count and is reported separately from the fixed K=64 comparison.",
        "",
    ])
    (args.run_root / "report.md").write_text("\n".join(lines), encoding="utf-8")
    print(args.run_root / "report.md")


if __name__ == "__main__":
    main()
