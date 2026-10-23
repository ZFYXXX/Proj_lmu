"""Build stages 2 and 4 reports from completed silent-redundancy runs."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from . import ACOUSTIC_FEATURES, CAUSAL_SELECTORS
from .metrics import jsonable

HERE = Path(__file__).resolve().parent
COLORS = {
    "attention": "#0072B2",
    "intensity": "#D55E00",
    "variation": "#009E73",
    "temporal_change": "#CC79A7",
    "random": "#777777",
}


def load_records(directory: Path) -> list[dict]:
    rows = []
    for path in sorted(directory.glob("samples.rank*.jsonl")):
        with path.open(encoding="utf-8") as stream:
            rows.extend(json.loads(line) for line in stream if line.strip())
    return sorted(rows, key=lambda row: row["doc_index"])


def write_csv(path: Path, rows: list[dict], fieldnames=None):
    if fieldnames is None:
        if not rows:
            return
        fieldnames = []
        for row in rows:
            for key in row:
                if key not in fieldnames:
                    fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def bootstrap_mean(values, *, seed=42, repeats=1000):
    values = np.asarray(values, dtype=float)
    if values.ndim == 1:
        values = values[:, None]
    mean = np.nanmean(values, axis=0)
    if len(values) < 2:
        return mean, np.full_like(mean, np.nan), np.full_like(mean, np.nan)
    rng = np.random.default_rng(seed)
    estimates = []
    for start in range(0, repeats, 50):
        count = min(50, repeats - start)
        indices = rng.integers(0, len(values), size=(count, len(values)))
        estimates.append(np.nanmean(values[indices], axis=1))
    estimates = np.concatenate(estimates)
    low, high = np.nanquantile(estimates, [0.025, 0.975], axis=0)
    return mean, low, high


def _save_figure(fig, path: Path):
    fig.savefig(path.with_suffix(".png"), dpi=180, facecolor="white")
    fig.savefig(path.with_suffix(".pdf"), facecolor="white")
    import matplotlib.pyplot as plt
    plt.close(fig)


def observational_report(run_dir: Path, output: Path) -> dict:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    observe = run_dir / "observe"
    rows = load_records(observe)
    if not rows:
        raise ValueError(f"No observation records in {observe}")
    output.mkdir(parents=True, exist_ok=True)
    figures = output / "figures"
    figures.mkdir(exist_ok=True)
    layers = [entry["layer"] for entry in rows[0]["layers"]]
    correlation_rows = []
    result = {
        "samples": len(rows),
        "accuracy": float(np.mean([row["accuracy"] for row in rows])),
        "layers": {},
    }
    for layer_position, layer in enumerate(layers):
        layer_result = {"features": {}}
        fig, axes = plt.subplots(len(ACOUSTIC_FEATURES), 2, figsize=(10, 10),
                                 layout="constrained")
        for feature_position, feature in enumerate(ACOUSTIC_FEATURES):
            values = [row["layers"][layer_position]["features"][feature] for row in rows]
            correlations = np.asarray([
                [value["pearson"] if value["pearson"] is not None else np.nan,
                 value["spearman"] if value["spearman"] is not None else np.nan]
                for value in values
            ])
            mean, low, high = bootstrap_mean(correlations)
            quintiles = []
            for bin_index in range(5):
                bin_rows = [value["quintiles"][bin_index] for value in values]
                count = sum(item["n"] for item in bin_rows)
                entry = {
                    "bin": bin_index,
                    "tokens": count,
                    "attention_mean": sum(item["attention_sum"] for item in bin_rows) / max(1, count),
                }
                for ratio in (75, 50, 25):
                    entry[f"retention_{ratio}"] = sum(item[f"kept_{ratio}"] for item in bin_rows) / max(1, count)
                quintiles.append(entry)
            histogram = np.sum([
                np.asarray(value["rank_histogram_10x10"], dtype=float) for value in values
            ], axis=0)
            histogram = histogram / np.maximum(histogram.sum(axis=1, keepdims=True), 1)
            layer_result["features"][feature] = {
                "pearson": {"mean": mean[0], "ci95": [low[0], high[0]]},
                "spearman": {"mean": mean[1], "ci95": [low[1], high[1]]},
                "quintiles": quintiles,
                "conditional_attention_rank_histogram": histogram,
            }
            correlation_rows.extend([
                {"layer": layer, "feature": feature, "metric": metric,
                 "mean": mean[index], "ci_low": low[index], "ci_high": high[index]}
                for index, metric in enumerate(("pearson", "spearman"))
            ])
            axis = axes[feature_position, 0]
            image = axis.imshow(histogram.T, origin="lower", aspect="auto", vmin=0, vmax=0.35,
                                cmap="viridis", extent=(0, 1, 0, 1))
            axis.set(title=f"{feature}: P(attention rank | acoustic rank)",
                     xlabel="Acoustic within-chunk rank", ylabel="Attention within-chunk rank")
            fig.colorbar(image, ax=axis, label="Conditional fraction")
            axis = axes[feature_position, 1]
            x = np.arange(1, 6)
            for ratio, color in ((75, "#56B4E9"), (50, "#0072B2"), (25, "#D55E00")):
                axis.plot(x, [entry[f"retention_{ratio}"] for entry in quintiles], marker="o",
                          color=color, label=f"keep {ratio}%")
            axis.set(title=f"{feature}: attention-TopK retention", xlabel="Acoustic quintile",
                     ylabel="Fraction retained", xticks=x, ylim=(0, 1))
            axis.legend(fontsize=8)
        quadrants = {
            key: sum(row["layers"][layer_position]["temporal_change_attention_quadrants"][key]
                     for row in rows)
            for key in rows[0]["layers"][layer_position]["temporal_change_attention_quadrants"]
        }
        layer_result["temporal_change_attention_quadrants"] = quadrants
        result["layers"][str(layer)] = layer_result
        fig.suptitle(f"MMSU-Final observational analysis | encoder layer {layer} | n={len(rows)}")
        _save_figure(fig, figures / f"layer{layer}_aggregate")

    write_csv(output / "correlations.csv", correlation_rows)
    for row in rows:
        if not row.get("example_file"):
            continue
        data = np.load(observe / row["example_file"], allow_pickle=False)
        time = data["time_s"]
        mel = data["mel"]
        mel_step = float(data["mel_step_s"])
        bounds = data["boundaries"]
        fig, axes = plt.subplots(5, 1, figsize=(12, 11), layout="constrained", sharex=True)
        axes[0].imshow(mel, origin="lower", aspect="auto", cmap="magma",
                       extent=(0, mel.shape[1] * mel_step, 0, mel.shape[0]))
        axes[0].set_ylabel("Log-Mel bin")
        for feature, color in zip(ACOUSTIC_FEATURES, ("#D55E00", "#009E73", "#CC79A7")):
            values = data[f"acoustic_{feature}"]
            normalized = (values - np.nanmin(values)) / max(float(np.nanmax(values) - np.nanmin(values)), 1e-12)
            axes[1].plot(time, normalized, lw=0.9, label=feature, color=color)
        axes[1].set_ylabel("Normalized\nacoustics")
        axes[1].legend(ncols=3, fontsize=8)
        for layer, color in zip(layers, ("#56B4E9", "#0072B2", "#D55E00")):
            axes[2].plot(time, data[f"attention_l{layer}"], lw=0.9, label=f"layer {layer}", color=color)
        axes[2].axhline(1, color="0.5", ls="--", lw=0.7)
        axes[2].set_ylabel("Received\nattention")
        axes[2].legend(ncols=len(layers), fontsize=8)
        for y, layer in enumerate(layers):
            mask = data[f"keep_l{layer}_r50"].astype(bool)
            axes[3].scatter(time[mask], np.full(mask.sum(), y), marker="|", s=50,
                            label=f"layer {layer}")
        axes[3].set(yticks=np.arange(len(layers)), yticklabels=[f"L{layer}" for layer in layers],
                    ylabel="Virtual\nkeep 50%", ylim=(-0.6, len(layers) - 0.4))
        low_change = data["acoustic_temporal_change"] <= np.quantile(data["acoustic_temporal_change"], 0.25)
        top_attention = data[f"attention_l{layers[-1]}"] >= np.quantile(
            data[f"attention_l{layers[-1]}"], 0.75)
        axes[4].fill_between(time, 0, 1, where=low_change, color="#999999", alpha=0.35,
                             label="low temporal change")
        axes[4].scatter(time[low_change & top_attention], np.full((low_change & top_attention).sum(), 0.5),
                        color="#D55E00", s=18, label="low-change / high-attention")
        axes[4].set(yticks=[], ylabel="Candidate", xlabel="Audio source time (s)", ylim=(0, 1))
        axes[4].legend(fontsize=8, loc="upper right")
        for axis in axes:
            for boundary in bounds[1:-1]:
                if boundary < len(time):
                    axis.axvline(time[boundary], color="white" if axis is axes[0] else "0.6",
                                 lw=0.6, ls=":")
            axis.spines[["top", "right"]].set_visible(False)
        fig.suptitle(
            f"Doc {row['doc_index']} | {row['task_name']} | accuracy={row['accuracy']}\n"
            "Vertical selections are time tokens spanning all Mel frequencies"
        )
        _save_figure(fig, figures / f"doc{row['doc_index']}_five_panel")
        data.close()
    (output / "summary.json").write_text(
        json.dumps(jsonable(result), ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    return result


def _taxonomy_map():
    raw = json.loads((HERE / "task_taxonomy.json").read_text(encoding="utf-8"))
    mapping = {}
    for major, domains in raw.items():
        for domain, tasks in domains.items():
            for task in tasks:
                if task in mapping:
                    raise ValueError(f"Duplicate taxonomy task {task}")
                mapping[task] = f"{major}/{domain}"
    if len(mapping) != 47:
        raise ValueError("MMSU taxonomy must contain 47 tasks")
    return mapping


def comparison_report(run_dir: Path, output: Path, observation: dict) -> dict:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    baseline = load_records(run_dir / "observe")
    baseline_by_id = {row["id"]: row for row in baseline}
    taxonomy = _taxonomy_map()
    condition_dirs = []
    conditions_root = run_dir / "conditions"
    if conditions_root.exists():
        condition_dirs = sorted(path.parent for path in conditions_root.glob("*/analysis.json"))
    condition_rows = []
    group_rows = []
    paired_rows = []
    results = {}
    baseline_accuracy = np.asarray([row["accuracy"] for row in baseline], dtype=float)
    groups = ["All", *sorted(set(taxonomy.values()))]
    for directory in condition_dirs:
        settings = json.loads((directory / "analysis.json").read_text())
        rows = load_records(directory)
        if len(rows) != len(baseline) or {row["id"] for row in rows} != set(baseline_by_id):
            raise ValueError(f"Condition coverage differs from baseline: {directory}")
        rows = sorted(rows, key=lambda row: baseline_by_id[row["id"]]["doc_index"])
        accuracy = np.asarray([row["accuracy"] for row in rows], dtype=float)
        delta = accuracy - baseline_accuracy
        mean, low, high = bootstrap_mean(delta)
        condition_id = directory.name
        realized = float(np.mean([row["pruning"]["realized_encoder_ratio"] for row in rows]))
        transitions = {
            "correct_to_wrong": int(np.sum((baseline_accuracy == 1) & (accuracy == 0))),
            "wrong_to_correct": int(np.sum((baseline_accuracy == 0) & (accuracy == 1))),
        }
        summary = {
            "selector": settings["selector"],
            "target_ratio": settings["target_ratio"],
            "keep_ratio_per_stage": settings["keep_ratio_per_stage"],
            "layers": settings["layers"],
            "accuracy": float(accuracy.mean()),
            "baseline_accuracy": float(baseline_accuracy.mean()),
            "delta_pp": float(100 * mean[0]),
            "delta_ci95_pp": [float(100 * low[0]), float(100 * high[0])],
            "realized_encoder_ratio": realized,
            **transitions,
        }
        if settings["selector"] in CAUSAL_SELECTORS:
            causal_chunks = [chunk
                             for row in rows
                             for stage in row["pruning"]["stages"]
                             for chunk in stage["chunks"]
                             if chunk.get("dropped_tokens", 0) > 0]
            dropped_total = sum(chunk["dropped_tokens"] for chunk in causal_chunks)
            summary["dropped_attention_mean"] = (
                sum(chunk["dropped_attention_mean"] * chunk["dropped_tokens"]
                    for chunk in causal_chunks) / dropped_total if dropped_total else None
            )
            summary["dropped_temporal_change_mean"] = (
                sum(chunk["dropped_temporal_change_mean"] * chunk["dropped_tokens"]
                    for chunk in causal_chunks) / dropped_total if dropped_total else None
            )
        results[condition_id] = summary
        condition_rows.append({"condition": condition_id, **summary,
                               "layers": ",".join(map(str, settings["layers"])),
                               "delta_ci95_pp": json.dumps(summary["delta_ci95_pp"])})
        for group in groups:
            selected = np.ones(len(rows), dtype=bool) if group == "All" else np.asarray([
                taxonomy[row["task_name"]] == group for row in rows
            ])
            if not selected.any():
                group_rows.append({
                    "condition": condition_id,
                    "group": group,
                    "n": 0,
                    "accuracy": None,
                    "baseline_accuracy": None,
                    "delta_pp": None,
                    "ci_low_pp": None,
                    "ci_high_pp": None,
                })
                continue
            group_delta = delta[selected]
            group_mean, group_low, group_high = bootstrap_mean(group_delta)
            group_rows.append({
                "condition": condition_id,
                "group": group,
                "n": int(selected.sum()),
                "accuracy": float(accuracy[selected].mean()),
                "baseline_accuracy": float(baseline_accuracy[selected].mean()),
                "delta_pp": float(100 * group_mean[0]),
                "ci_low_pp": float(100 * group_low[0]),
                "ci_high_pp": float(100 * group_high[0]),
            })
        paired_rows.extend({
            "condition": condition_id,
            "id": row["id"],
            "doc_index": row["doc_index"],
            "baseline_accuracy": int(base["accuracy"]),
            "condition_accuracy": int(row["accuracy"]),
            "delta": int(row["accuracy"] - base["accuracy"]),
        } for row, base in zip(rows, baseline))

    output.mkdir(parents=True, exist_ok=True)
    write_csv(output / "condition_accuracy.csv", condition_rows)
    write_csv(output / "six_group_accuracy.csv", group_rows)
    write_csv(output / "paired_outcomes.csv", paired_rows)
    standard = [item for item in results.values() if item["selector"] not in CAUSAL_SELECTORS]
    if standard:
        fig, axis = plt.subplots(figsize=(8, 5), layout="constrained")
        for selector in COLORS:
            selected = sorted((item for item in standard if item["selector"] == selector),
                              key=lambda item: item["realized_encoder_ratio"])
            if selected:
                axis.plot([100 * item["realized_encoder_ratio"] for item in selected],
                          [100 * item["accuracy"] for item in selected], marker="o",
                          color=COLORS[selector], label=selector)
        axis.axhline(100 * baseline_accuracy.mean(), color="black", ls="--", lw=1, label="baseline")
        axis.set(xlabel="Realized encoder-token retention (%)", ylabel="MMSU-Final accuracy (%)",
                 title="Equal-budget pruning comparison")
        axis.legend(ncols=2, fontsize=8)
        axis.spines[["top", "right"]].set_visible(False)
        _save_figure(fig, output / "accuracy_retention_curve")
    causal = [item for item in results.values() if item["selector"] in CAUSAL_SELECTORS]
    if causal:
        fig, axis = plt.subplots(figsize=(9, 5), layout="constrained")
        labels = [item["selector"].replace("causal_", "") for item in causal]
        damage = [-item["delta_pp"] for item in causal]
        axis.bar(np.arange(len(causal)), damage, color=["#D55E00", "#56B4E9", "#009E73", "#999999"])
        axis.axhline(0, color="black", lw=0.8)
        axis.set(xticks=np.arange(len(causal)), xticklabels=labels, ylabel="Accuracy damage (pp)",
                 title="Matched 10% causal deletion at encoder layer 9")
        axis.tick_params(axis="x", rotation=18)
        axis.spines[["top", "right"]].set_visible(False)
        _save_figure(fig, output / "causal_deletion_damage")

    decision = {}
    at_50 = {item["selector"]: item for item in standard
             if abs(item["target_ratio"] - 0.50) < 1e-6}
    if "attention" in at_50:
        acoustic = [at_50[name]["accuracy"] for name in ACOUSTIC_FEATURES if name in at_50]
        decision["attention_minus_best_acoustic_pp_at_50"] = (
            100 * (at_50["attention"]["accuracy"] - max(acoustic)) if acoustic else None)
    causal_by_name = {item["selector"]: item for item in causal}
    high_name = "causal_low_change_high_attention"
    low_name = "causal_low_change_low_attention"
    if high_name in causal_by_name and low_name in causal_by_name:
        decision["extra_damage_low_change_high_vs_low_attention_pp"] = (
            causal_by_name[low_name]["delta_pp"] - causal_by_name[high_name]["delta_pp"]
        )
    final = {
        "baseline_accuracy": float(baseline_accuracy.mean()),
        "conditions": results,
        "observational": observation,
        "decision_evidence": decision,
        "interpretation_rule": (
            "Only call low-change tokens task-informative when the observational high-attention "
            "quadrant exists and matched deletion of those tokens damages MMSU more than controls. "
            "Only call low-change/low-attention tokens redundant when their deletion is harmless."
        ),
        "bootstrap": {"unit": "MMSU audio/question record", "repeats": 1000, "seed": 42},
    }
    (output / "summary.json").write_text(
        json.dumps(jsonable(final), ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    return final


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--observational-only", action="store_true")
    args = parser.parse_args()
    run_dir = Path(args.run_dir).expanduser().resolve()
    observation = observational_report(run_dir, run_dir / "report" / "observational")
    if not args.observational_only:
        comparison_report(run_dir, run_dir / "report" / "final", observation)
    print(f"Report written to {run_dir / 'report'}")


if __name__ == "__main__":
    main()
