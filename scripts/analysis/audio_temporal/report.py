"""Create tables and figures for one completed temporal-audio MMSU run."""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np

from .metrics import REGION_NAMES, jsonable


PROVENANCE_METRICS = (
    "tokens",
    "nominal_span_ms_mean",
    "nominal_span_ms_max",
    "effective_coverage_ms_mean",
    "max_gap_ms_mean",
    "max_gap_ms_max",
    "fragments_mean",
    "fragmented_tokens",
    "hard_chunk_crossings",
    "input_boundary_crossings",
)


def load_jsonl(paths):
    rows = []
    for path in paths:
        with path.open(encoding="utf-8") as source:
            rows.extend(json.loads(line) for line in source if line.strip())
    return rows


def _mean_ci(values):
    values = np.asarray([value for value in values if value is not None], dtype=float)
    if not len(values):
        return {"n": 0, "mean": None, "ci95_low": None, "ci95_high": None}
    mean = float(values.mean())
    half = 1.96 * float(values.std(ddof=1)) / math.sqrt(len(values)) if len(values) > 1 else 0.0
    return {"n": len(values), "mean": mean, "ci95_low": mean - half, "ci95_high": mean + half}


def _stage_order(name):
    if name == "C":
        return (0, 0)
    if name.startswith("E_l"):
        return (1, int(name[3:]))
    if name == "E_L":
        return (2, 0)
    if name == "P_in":
        return (3, 0)
    if name == "P_out":
        return (4, 0)
    return (5, name)


def observational_tables(records):
    """Equal-weight unique audio files for representation-level measurements."""
    lag_values, region_values, prediction_values, tokens = {}, {}, {}, {}
    for record in records:
        audio = record.get("audio_path") or record.get("id") or record["doc_index"]
        for stage in record["stages"]:
            name = stage["name"]
            tokens.setdefault((audio, name), []).append(stage["tokens"])
            for row in stage["lag_similarity"]:
                key = (audio, name, float(row["lag_ms"]))
                lag_values.setdefault(key, {"raw": [], "demean": [], "pairs": []})
                lag_values[key]["raw"].append(row["raw_cosine"])
                lag_values[key]["demean"].append(row["demean_cosine"])
                lag_values[key]["pairs"].append(row["pairs"])
            for region, row in stage["region_similarity"].items():
                key = (audio, name, region)
                region_values.setdefault(key, {"raw": [], "demean": [], "pairs": []})
                region_values[key]["raw"].append(row["raw_cosine"])
                region_values[key]["demean"].append(row["demean_cosine"])
                region_values[key]["pairs"].append(row["pairs"])
            for row in stage.get("predictability", []):
                key = (audio, name, float(row["block_ms"]), row["predictor"])
                prediction_values.setdefault(key, {"l2": [], "cosine": [], "tokens": []})
                prediction_values[key]["l2"].append(row["prediction_relative_l2"])
                prediction_values[key]["cosine"].append(row["prediction_cosine_mean"])
                prediction_values[key]["tokens"].append(row["predicted_tokens"])

    per_audio_lag = []
    for (audio, stage, lag), values in lag_values.items():
        per_audio_lag.append({
            "audio": audio, "stage": stage, "lag_ms": lag,
            "raw_cosine": _mean_ci(values["raw"])["mean"],
            "demean_cosine": _mean_ci(values["demean"])["mean"],
            "pairs": int(sum(values["pairs"])),
        })
    lag_summary = []
    for stage in sorted({row["stage"] for row in per_audio_lag}, key=_stage_order):
        for lag in sorted({row["lag_ms"] for row in per_audio_lag if row["stage"] == stage}):
            subset = [row for row in per_audio_lag if row["stage"] == stage and row["lag_ms"] == lag]
            raw, demean = _mean_ci([row["raw_cosine"] for row in subset]), _mean_ci(
                [row["demean_cosine"] for row in subset])
            lag_summary.append({
                "stage": stage, "lag_ms": lag, "unique_audio": raw["n"],
                "raw_mean": raw["mean"], "raw_ci95_low": raw["ci95_low"],
                "raw_ci95_high": raw["ci95_high"], "demean_mean": demean["mean"],
                "demean_ci95_low": demean["ci95_low"],
                "demean_ci95_high": demean["ci95_high"],
            })

    per_audio_region = []
    for (audio, stage, region), values in region_values.items():
        per_audio_region.append({
            "audio": audio, "stage": stage, "region": region,
            "raw_cosine": _mean_ci(values["raw"])["mean"],
            "demean_cosine": _mean_ci(values["demean"])["mean"],
            "pairs": int(sum(values["pairs"])),
        })
    region_summary = []
    for stage in sorted({row["stage"] for row in per_audio_region}, key=_stage_order):
        for region in REGION_NAMES:
            subset = [row for row in per_audio_region
                      if row["stage"] == stage and row["region"] == region and row["pairs"] > 0]
            raw, demean = _mean_ci([row["raw_cosine"] for row in subset]), _mean_ci(
                [row["demean_cosine"] for row in subset])
            region_summary.append({
                "stage": stage, "region": region, "unique_audio": raw["n"],
                "raw_mean": raw["mean"], "raw_ci95_low": raw["ci95_low"],
                "raw_ci95_high": raw["ci95_high"], "demean_mean": demean["mean"],
                "demean_ci95_low": demean["ci95_low"],
                "demean_ci95_high": demean["ci95_high"],
            })
    token_summary = []
    for stage in sorted({stage for _, stage in tokens}, key=_stage_order):
        per_audio = [float(np.mean(value)) for (audio, name), value in tokens.items() if name == stage]
        token_summary.append({"stage": stage, **_mean_ci(per_audio)})
    per_audio_prediction = []
    for (audio, stage, block_ms, predictor), values in prediction_values.items():
        per_audio_prediction.append({
            "audio": audio, "stage": stage, "block_ms": block_ms, "predictor": predictor,
            "prediction_relative_l2": _mean_ci(values["l2"])["mean"],
            "prediction_cosine_mean": _mean_ci(values["cosine"])["mean"],
            "predicted_tokens": _mean_ci(values["tokens"])["mean"],
        })
    prediction_summary = []
    for stage in sorted({row["stage"] for row in per_audio_prediction}, key=_stage_order):
        predictors = sorted({row["predictor"] for row in per_audio_prediction
                             if row["stage"] == stage})
        for predictor in predictors:
            blocks = sorted({row["block_ms"] for row in per_audio_prediction
                             if row["stage"] == stage and row["predictor"] == predictor})
            for block_ms in blocks:
                subset = [row for row in per_audio_prediction if row["stage"] == stage
                          and row["predictor"] == predictor and row["block_ms"] == block_ms]
                l2 = _mean_ci([row["prediction_relative_l2"] for row in subset])
                cosine = _mean_ci([row["prediction_cosine_mean"] for row in subset])
                prediction_summary.append({
                    "stage": stage, "predictor": predictor, "block_ms": block_ms,
                    "unique_audio": l2["n"], "relative_l2_mean": l2["mean"],
                    "relative_l2_ci95_low": l2["ci95_low"],
                    "relative_l2_ci95_high": l2["ci95_high"],
                    "cosine_mean": cosine["mean"],
                })
    return lag_summary, region_summary, prediction_summary, token_summary


def baseline_provenance_table(records):
    """Summarize native P_in coverage with each unique audio weighted once."""
    by_audio = {}
    for record in records:
        audio = record.get("audio_path") or record.get("id") or record["doc_index"]
        values = by_audio.setdefault(audio, {metric: [] for metric in PROVENANCE_METRICS})
        provenance = record["projector_provenance"]
        for metric in PROVENANCE_METRICS:
            values[metric].append(provenance.get(metric))
    rows = []
    for metric in PROVENANCE_METRICS:
        per_audio = [_mean_ci(values[metric])["mean"] for values in by_audio.values()]
        estimate = _mean_ci(per_audio)
        rows.append({
            "stage": "P_in",
            "metric": metric,
            "unique_audio": estimate["n"],
            "mean": estimate["mean"],
            "ci95_low": estimate["ci95_low"],
            "ci95_high": estimate["ci95_high"],
        })
    return rows


def ltbm_provenance_report(records, run_dir, report_dir):
    """Equal-weight audio summaries and actual merged-token coverage figures."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    by_audio = {}
    for record in records:
        audio = record.get("audio_path") or record.get("id") or record["doc_index"]
        by_audio.setdefault(audio, record)
    unique = list(by_audio.values())
    audio_rows = []
    for record in unique:
        audio_rows.append({
            "audio_path": record.get("audio_path"),
            "doc_index": record["doc_index"],
            "original_projector_tokens": record["original_projector_tokens"],
            "kept_projector_tokens": record["projector_tokens"],
            "achieved_keep_ratio": record["ltbm_achieved_keep_ratio"],
            **record["ltbm_provenance"],
        })
    _write_csv(report_dir / "ltbm_per_audio.csv", audio_rows)
    metric_rows = []
    for metric in PROVENANCE_METRICS:
        estimate = _mean_ci([record["ltbm_provenance"].get(metric) for record in unique])
        metric_rows.append({"stage": "LTBM_P_in", "metric": metric,
                            "unique_audio": estimate["n"], "mean": estimate["mean"],
                            "ci95_low": estimate["ci95_low"],
                            "ci95_high": estimate["ci95_high"]})
    _write_csv(report_dir / "ltbm_provenance.csv", metric_rows)

    token_rows = [trace for record in unique for trace in record["ltbm_token_provenance"]]
    figures = []
    figure, axes = plt.subplots(1, 3, figsize=(12, 3.8))
    for axis, field, label in zip(
            axes, ("nominal_span_ms", "max_gap_ms", "fragments"),
            ("Nominal span (ms)", "Maximum gap (ms)", "Fragments")):
        values = [item[field] for item in token_rows]
        bins = (np.arange(0.5, max(values, default=1) + 1.5, 1)
                if field == "fragments" else "auto")
        axis.hist(values, bins=bins)
        axis.set_xlabel(label)
        axis.set_ylabel("Merged token count")
        axis.grid(axis="y", alpha=0.2)
    figure.suptitle("Native LTBM output-token temporal support (unique audio)")
    figures.extend(_save_figure(figure, report_dir, "ltbm_provenance_distributions"))
    plt.close(figure)

    example = next((record for record in records if record.get("example_file")), None)
    if example is not None:
        with np.load(run_dir / example["example_file"], allow_pickle=False) as data:
            mel, mel_time = data["mel"], data["mel_time_ms"]
            starts = data["projector_interval_start_ms"]
            ends = data["projector_interval_end_ms"]
            figure, axes = plt.subplots(3, 1, figsize=(11, 7), sharex=True,
                                        gridspec_kw={"height_ratios": [4, 1, 2]})
            axes[0].imshow(mel, aspect="auto", origin="lower",
                           extent=[mel_time[0], mel_time[-1], 0, mel.shape[0]], cmap="magma")
            axes[0].set_ylabel("Mel bin")
            for index, (start, end) in enumerate(zip(starts, ends)):
                axes[1].plot([start, end], [index, index], linewidth=0.7)
            axes[1].set_ylabel("Native P_in")
            for index, trace in enumerate(example["ltbm_token_provenance"]):
                for start, end in trace["intervals_ms"]:
                    axes[2].plot([start, end], [index, index], linewidth=0.9)
            axes[2].set_ylabel("LTBM P_in")
            axes[2].set_xlabel("Time (ms)")
            figure.suptitle("Spectrogram and actual LTBM token provenance")
            figures.extend(_save_figure(figure, report_dir, "ltbm_token_coverage_example"))
            plt.close(figure)
    return metric_rows, figures


def condition_table(rows):
    summaries = []
    for condition_id in sorted({row["condition_id"] for row in rows}):
        subset = [row for row in rows if row["condition_id"] == condition_id]
        condition = subset[0]["condition"]
        accuracy = _mean_ci([row["accuracy"] for row in subset])
        delta = _mean_ci([row["accuracy_delta"] for row in subset])
        total_seconds = _mean_ci([row["diagnostic_total_seconds"] for row in subset])
        generation_seconds = _mean_ci([row["diagnostic_generation_seconds"] for row in subset])
        peak_memory = _mean_ci([row.get("peak_gpu_memory_bytes") for row in subset])
        interventions = [row["intervention"] for row in subset]
        summary = {
            "condition_id": condition_id,
            "family": condition["family"],
            **{key: value for key, value in condition.items() if key not in {"id", "family"}},
            "samples": len(subset),
            "accuracy": accuracy["mean"],
            "accuracy_ci95_low": accuracy["ci95_low"],
            "accuracy_ci95_high": accuracy["ci95_high"],
            "paired_accuracy_delta": delta["mean"],
            "delta_ci95_low": delta["ci95_low"],
            "delta_ci95_high": delta["ci95_high"],
            "total_seconds_mean": total_seconds["mean"],
            "generation_seconds_mean": generation_seconds["mean"],
            "peak_gpu_memory_bytes_mean": peak_memory["mean"],
        }
        for field in (
            "prediction_relative_l2", "prediction_cosine_mean",
            "representation_relative_l2", "representation_cosine_mean",
            "pair_similarity_mean", "pair_distance_ms_mean", "merged_pair_similarity_mean",
            "merged_pair_distance_ms_mean", "similarity_match_mae",
            "distance_ms_q25", "distance_ms_q50", "distance_ms_q75", "distance_ms_q95",
            "realized_keep_ratio", "projector_tokens", "original_projector_tokens",
            "pooled_representation_relative_l2"):
            summary[field] = _mean_ci([item.get(field) for item in interventions])["mean"]
        for field in PROVENANCE_METRICS:
            summary[f"provenance_{field}"] = _mean_ci([
                item.get("provenance", {}).get(field) for item in interventions])["mean"]
        summary["exact_budget_rate"] = _mean_ci([
            float(item["exact_budget"]) for item in interventions
            if item.get("exact_budget") is not None])["mean"]
        for field in ("predicted_tokens", "replaced_tokens", "realized_merges",
                      "similarity_matched_common_budget"):
            summary[field] = _mean_ci([item.get(field) for item in interventions])["mean"]
        summaries.append(summary)
    return summaries


def _write_csv(path, rows):
    if not rows:
        return
    fields = []
    for row in rows:
        fields.extend(key for key in row if key not in fields)
    with path.open("w", newline="", encoding="utf-8") as destination:
        writer = csv.DictWriter(destination, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _save_figure(figure, output, name):
    paths = []
    for extension in ("png", "pdf"):
        path = output / f"{name}.{extension}"
        figure.savefig(path, dpi=220, bbox_inches="tight")
        paths.append(path.name)
    return paths


def _heatmap(plt, rows, *, value, columns, title, ylabel, output, name):
    stages = sorted({row["stage"] for row in rows}, key=_stage_order)
    x_values = columns(rows)
    matrix = np.full((len(stages), len(x_values)), np.nan)
    for row in rows:
        matrix[stages.index(row["stage"]), x_values.index(
            row["lag_ms"] if "lag_ms" in row else row["region"])] = (
                np.nan if row[value] is None else row[value])
    figure, axis = plt.subplots(figsize=(max(6, len(x_values) * 0.9), max(3, len(stages) * 0.55)))
    finite = matrix[np.isfinite(matrix)]
    limits = ({"vmin": float(finite.min()), "vmax": float(finite.max())}
              if len(finite) else {})
    image = axis.imshow(matrix, aspect="auto", cmap="viridis", **limits)
    axis.set_xticks(range(len(x_values)), x_values, rotation=30, ha="right")
    axis.set_yticks(range(len(stages)), stages)
    axis.set_xlabel("Lag (ms)" if "lag_ms" in rows[0] else "Input-derived region proxy")
    axis.set_ylabel(ylabel)
    axis.set_title(title)
    figure.colorbar(image, ax=axis, label="Cosine similarity")
    return figure, _save_figure(figure, output, name)


def create_figures(run_dir, report_dir, settings, records, condition_rows,
                   lag_rows, region_rows, prediction_rows, token_rows,
                   condition_rows_summary):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    paths = []
    if lag_rows:
        lags = sorted({row["lag_ms"] for row in lag_rows})
        for value, label in (("raw_mean", "Raw"), ("demean_mean", "De-meaned")):
            figure, saved = _heatmap(
                plt, lag_rows, value=value, columns=lambda rows: lags,
                title=f"{label} lag cosine across representation stages",
                ylabel="Representation stage", output=report_dir,
                name=f"lag_heatmap_{value.replace('_mean', '')}")
            paths.extend(saved)
            plt.close(figure)
        figure, axis = plt.subplots(figsize=(7, 4.5))
        for stage in sorted({row["stage"] for row in lag_rows}, key=_stage_order):
            subset = sorted([row for row in lag_rows if row["stage"] == stage],
                            key=lambda row: row["lag_ms"])
            axis.plot([row["lag_ms"] for row in subset], [row["demean_mean"] for row in subset],
                      marker="o", label=stage)
        axis.set_xscale("log", base=2)
        axis.set_xlabel("Lag (ms)")
        axis.set_ylabel("De-meaned cosine similarity")
        axis.set_title("Temporal similarity decay")
        axis.grid(alpha=0.25)
        axis.legend(ncol=2, fontsize=8)
        paths.extend(_save_figure(figure, report_dir, "similarity_decay"))
        plt.close(figure)
    if region_rows:
        regions = list(REGION_NAMES)
        figure, saved = _heatmap(
            plt, region_rows, value="demean_mean", columns=lambda rows: regions,
            title="Region-specific temporal redundancy", ylabel="Representation stage",
            output=report_dir, name="region_redundancy_heatmap")
        paths.extend(saved)
        plt.close(figure)
    if token_rows:
        figure, axis = plt.subplots(figsize=(7, 4))
        axis.bar([row["stage"] for row in token_rows], [row["mean"] for row in token_rows])
        axis.set_ylabel("Mean token count per unique audio")
        axis.set_title("Representation length by stage")
        axis.tick_params(axis="x", rotation=30)
        paths.extend(_save_figure(figure, report_dir, "stage_token_counts"))
        plt.close(figure)
    if prediction_rows:
        stages = sorted({row["stage"] for row in prediction_rows}, key=_stage_order)
        blocks = sorted({row["block_ms"] for row in prediction_rows})
        for predictor in sorted({row["predictor"] for row in prediction_rows}):
            matrix = np.full((len(stages), len(blocks)), np.nan)
            for row in prediction_rows:
                if row["predictor"] == predictor and row["relative_l2_mean"] is not None:
                    matrix[stages.index(row["stage"]), blocks.index(row["block_ms"])] = row[
                        "relative_l2_mean"]
            figure, axis = plt.subplots(figsize=(max(6, len(blocks) * 1.1),
                                                 max(3, len(stages) * 0.55)))
            image = axis.imshow(matrix, aspect="auto", cmap="magma")
            axis.set_xticks(range(len(blocks)), blocks)
            axis.set_yticks(range(len(stages)), stages)
            axis.set_xlabel("Missing block duration (ms)")
            axis.set_ylabel("Representation stage")
            axis.set_title(f"Layer-wise predictability: {predictor}")
            figure.colorbar(image, ax=axis, label="Prediction relative L2")
            paths.extend(_save_figure(figure, report_dir, f"predictability_layers_{predictor}"))
            plt.close(figure)

    family = condition_rows_summary[0]["family"] if condition_rows_summary else None
    if family:
        baseline_accuracy = float(np.mean([row["accuracy"] for row in records]))
        figure, axis = plt.subplots(figsize=(8, 4.5))
        if family == "locality":
            for scope in ("local", "global"):
                subset = sorted([row for row in condition_rows_summary if row["scope"] == scope],
                                key=lambda row: row["merge_fraction"])
                axis.plot([row["merge_fraction"] for row in subset], [row["accuracy"] for row in subset],
                          marker="o", label=scope)
            axis.set_xlabel("Merged-token fraction")
            axis.set_ylabel("MMSU-final accuracy")
            axis.set_title("Locality at similarity-matched merge pairs")
            axis.legend()
            axis.axhline(baseline_accuracy, color="black", linestyle="--", linewidth=1)
        elif family == "predictability":
            predictors = sorted({row["predictor"] for row in condition_rows_summary})
            for predictor in predictors:
                subset = sorted([row for row in condition_rows_summary if row["predictor"] == predictor],
                                key=lambda row: row["block_ms"])
                axis.plot([row["block_ms"] for row in subset],
                          [row["prediction_relative_l2"] for row in subset],
                          marker="o", label=predictor)
            axis.set_xlabel("Missing block duration (ms)")
            axis.set_ylabel("Prediction relative L2")
            axis.set_title("Continuous-block temporal predictability")
            axis.legend(fontsize=8)
        else:
            labels = [row.get("region") or row.get("strategy") or row["condition_id"]
                      for row in condition_rows_summary]
            values = [row["paired_accuracy_delta"] for row in condition_rows_summary]
            errors = [[value - row["delta_ci95_low"] for value, row in zip(values, condition_rows_summary)],
                      [row["delta_ci95_high"] - value for value, row in zip(values, condition_rows_summary)]]
            axis.bar(labels, values, yerr=errors, capsize=3)
            axis.axhline(0, color="black", linewidth=0.8)
            axis.set_ylabel("Paired MMSU-final accuracy delta")
            axis.set_title("Length-preserving temporal intervention")
            axis.tick_params(axis="x", rotation=30)
        axis.grid(axis="y", alpha=0.2)
        primary_name = ("predictability_reconstruction"
                        if family == "predictability" else f"{family}_task_effect")
        paths.extend(_save_figure(figure, report_dir, primary_name))
        plt.close(figure)

        if family == "predictability":
            figure, axis = plt.subplots(figsize=(7, 4.5))
            for predictor in sorted({row["predictor"] for row in condition_rows_summary}):
                subset = sorted([row for row in condition_rows_summary
                                 if row["predictor"] == predictor], key=lambda row: row["block_ms"])
                axis.plot([row["block_ms"] for row in subset],
                          [row["accuracy"] for row in subset], marker="o", label=predictor)
            axis.axhline(baseline_accuracy, color="black", linestyle="--", linewidth=1,
                         label="baseline")
            axis.set_xlabel("Missing block duration (ms)")
            axis.set_ylabel("MMSU-final accuracy")
            axis.set_title("Task effect of continuous-block prediction")
            axis.grid(alpha=0.2)
            axis.legend(fontsize=8)
            paths.extend(_save_figure(figure, report_dir, "predictability_task_effect"))
            plt.close(figure)

        if family == "locality":
            figure, axes = plt.subplots(1, 2, figsize=(10, 4.2))
            for scope in ("local", "global"):
                subset = sorted([row for row in condition_rows_summary if row["scope"] == scope],
                                key=lambda row: row["merge_fraction"])
                axes[0].plot([row["merge_fraction"] for row in subset],
                             [row["pair_similarity_mean"] for row in subset],
                             marker="o", label=scope)
            global_rows = sorted(
                [row for row in condition_rows_summary if row["scope"] == "global"],
                key=lambda row: row["merge_fraction"])
            axes[1].plot([row["merge_fraction"] for row in global_rows],
                         [row["similarity_match_mae"] for row in global_rows], marker="o")
            axes[0].set_xlabel("Merged-token fraction")
            axes[0].set_ylabel("Mean matched-pair cosine")
            axes[0].set_title("Matched cosine check")
            axes[0].legend()
            axes[1].set_xlabel("Merged-token fraction")
            axes[1].set_ylabel("Global-to-local cosine match MAE")
            axes[1].set_title("Residual matching error")
            for current in axes:
                current.grid(alpha=0.2)
            paths.extend(_save_figure(figure, report_dir, "locality_similarity_match"))
            plt.close(figure)

        if family == "locality":
            figure, axis = plt.subplots(figsize=(8, 4.5))
            distance_rows = [row for row in condition_rows_summary
                             if row["distance_ms_q50"] is not None]
            labels = [row.get("method") or row.get("scope") for row in distance_rows]
            labels = [f"{label}\n{row.get('keep_ratio', row.get('merge_fraction')):g}"
                      for label, row in zip(labels, distance_rows)]
            medians = [row["distance_ms_q50"] for row in distance_rows]
            upper = [row["distance_ms_q95"] - row["distance_ms_q50"]
                     if row["distance_ms_q95"] is not None else 0 for row in distance_rows]
            axis.errorbar(range(len(labels)), medians, yerr=[np.zeros(len(labels)), upper],
                          fmt="o", capsize=3)
            axis.set_xticks(range(len(labels)), labels, rotation=30, ha="right")
            axis.set_ylabel("Merge distance (ms): median to 95th percentile")
            axis.set_title("Temporal merge-distance distribution summary")
            axis.grid(axis="y", alpha=0.2)
            paths.extend(_save_figure(figure, report_dir, f"{family}_merge_distance"))
            plt.close(figure)

            provenance_axes = (
                ("provenance_nominal_span_ms_mean", "Mean nominal span (ms)"),
                ("provenance_max_gap_ms_mean", "Mean maximum gap (ms)"),
                ("provenance_fragments_mean", "Mean fragments per token"),
            )
            figure, axes = plt.subplots(1, 3, figsize=(13, 4.2), sharey=True)
            for axis, (field, xlabel) in zip(axes, provenance_axes):
                for row in condition_rows_summary:
                    value = row.get(field)
                    delta = row.get("paired_accuracy_delta")
                    if value is None or delta is None:
                        continue
                    label = row.get("method") or row.get("scope")
                    axis.scatter(value, -delta, s=35)
                    axis.annotate(label, (value, -delta), xytext=(3, 2),
                                  textcoords="offset points", fontsize=7)
                axis.set_xlabel(xlabel)
                axis.grid(alpha=0.2)
            axes[0].set_ylabel("MMSU-final accuracy damage (baseline - condition)")
            figure.suptitle("Temporal support fragmentation vs task degradation")
            paths.extend(_save_figure(figure, report_dir,
                                      f"{family}_provenance_vs_damage"))
            plt.close(figure)

        if family == "substitution":
            figure, axis = plt.subplots(figsize=(6, 4.5))
            for row in condition_rows_summary:
                if row["pair_similarity_mean"] is None or row["paired_accuracy_delta"] is None:
                    continue
                axis.scatter(row["pair_similarity_mean"], row["paired_accuracy_delta"], s=45)
                axis.annotate(row["strategy"],
                              (row["pair_similarity_mean"], row["paired_accuracy_delta"]),
                              xytext=(4, 3), textcoords="offset points", fontsize=8)
            axis.axhline(0, color="black", linewidth=0.8)
            axis.set_xlabel("Mean donor-receiver cosine")
            axis.set_ylabel("Paired MMSU-final accuracy delta")
            axis.set_title("Geometric similarity vs causal substitutability")
            axis.grid(alpha=0.2)
            paths.extend(_save_figure(figure, report_dir, "substitution_similarity_vs_damage"))
            plt.close(figure)

    example_files = sorted(run_dir.glob("example.rank*.doc*.npz"))
    if example_files:
        with np.load(example_files[0], allow_pickle=False) as data:
            if "mel" in data and "projector_interval_start_ms" in data:
                example_name = example_files[0].stem
                doc_index = int(example_name.split("doc")[-1])
                trace_row = next((row for row in condition_rows
                                  if row["doc_index"] == doc_index
                                  and row["intervention"].get("length_changed")
                                  and row["intervention"].get("token_provenance")), None)
                row_count = 3 if trace_row else 2
                ratios = [4, 1, 1] if trace_row else [4, 1]
                figure, axes = plt.subplots(row_count, 1, figsize=(10, 6 if trace_row else 5),
                                            sharex=True, gridspec_kw={"height_ratios": ratios})
                mel, mel_time = data["mel"], data["mel_time_ms"]
                axes[0].imshow(mel, aspect="auto", origin="lower",
                               extent=[mel_time[0], mel_time[-1], 0, mel.shape[0]], cmap="magma")
                axes[0].set_ylabel("Mel bin")
                axes[0].set_title("Spectrogram and native P_in temporal coverage")
                starts, ends = data["projector_interval_start_ms"], data["projector_interval_end_ms"]
                for index, (start, end) in enumerate(zip(starts, ends)):
                    axes[1].plot([start, end], [index % 2, index % 2], linewidth=2)
                axes[1].set_yticks([])
                axes[1].set_ylabel("P_in")
                if trace_row:
                    for token_index, trace in enumerate(
                            trace_row["intervention"]["token_provenance"]):
                        for start, end in trace["intervals_ms"]:
                            axes[2].plot([start, end], [token_index, token_index], linewidth=1.5)
                    axes[2].set_yticks([])
                    axes[2].set_ylabel("Merged")
                    axes[2].set_title(trace_row["condition_id"], fontsize=9)
                axes[-1].set_xlabel("Time (ms)")
                paths.extend(_save_figure(figure, report_dir, "token_coverage_example"))
                plt.close(figure)

                if trace_row:
                    trace_items = trace_row["intervention"]["token_provenance"]
                    trace_title = trace_row["condition_id"]
                else:
                    trace_items = [
                        {"nominal_span_ms": float(end - start), "max_gap_ms": 0.0,
                         "fragments": 1}
                        for start, end in zip(starts, ends)]
                    trace_title = "native P_in"
                figure, axes = plt.subplots(1, 3, figsize=(12, 3.8))
                for axis, field, label in zip(
                        axes,
                        ("nominal_span_ms", "max_gap_ms", "fragments"),
                        ("Nominal span (ms)", "Maximum gap (ms)", "Fragments")):
                    values = [item[field] for item in trace_items]
                    bins = (np.arange(0.5, max(values, default=1) + 1.5, 1)
                            if field == "fragments" else "auto")
                    axis.hist(values, bins=bins)
                    axis.set_xlabel(label)
                    axis.set_ylabel("Token count")
                    axis.grid(axis="y", alpha=0.2)
                figure.suptitle(f"Example token provenance distribution: {trace_title}")
                paths.extend(_save_figure(
                    figure, report_dir, "token_provenance_distributions_example"))
                plt.close(figure)
    return paths


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    args = parser.parse_args()
    run_dir = Path(args.run_dir).expanduser().resolve()
    settings = json.loads((run_dir / "analysis.json").read_text())
    coverage = json.loads((run_dir / "coverage.json").read_text())
    if not coverage["complete"]:
        raise RuntimeError("Refuse to summarize an incomplete run")
    records = load_jsonl(sorted(run_dir.glob("samples.rank*.jsonl")))
    conditions = load_jsonl(sorted(run_dir.glob("conditions.rank*.jsonl")))
    report_dir = run_dir / "report"
    report_dir.mkdir(exist_ok=True)
    lag_rows, region_rows, prediction_rows, token_rows = observational_tables(records)
    baseline_provenance_rows = baseline_provenance_table(records)
    condition_rows_summary = condition_table(conditions)
    _write_csv(report_dir / "lag_similarity.csv", lag_rows)
    _write_csv(report_dir / "region_similarity.csv", region_rows)
    _write_csv(report_dir / "layer_predictability.csv", prediction_rows)
    _write_csv(report_dir / "stage_token_counts.csv", token_rows)
    _write_csv(report_dir / "baseline_provenance.csv", baseline_provenance_rows)
    _write_csv(report_dir / "condition_summary.csv", condition_rows_summary)
    figures = create_figures(
        run_dir, report_dir, settings, records, conditions,
        lag_rows, region_rows, prediction_rows, token_rows, condition_rows_summary)
    ltbm_rows = []
    if settings["experiment"] == "provenance":
        ltbm_rows, ltbm_figures = ltbm_provenance_report(records, run_dir, report_dir)
        figures.extend(ltbm_figures)
    baseline = _mean_ci([row["accuracy"] for row in records])
    summary = {
        "experiment": settings["experiment"],
        "dataset": "mmsufinal",
        "baseline_accuracy": (None if settings["experiment"] == "provenance" else baseline),
        "ltbm_accuracy": (baseline if settings["experiment"] == "provenance" else None),
        "question_weighting": "one MMSU-final question, matching the task scorer",
        "representation_weighting": "one unique audio_path after averaging repeated questions",
        "lag_similarity": lag_rows,
        "region_similarity": region_rows,
        "layer_predictability": prediction_rows,
        "stage_token_counts": token_rows,
        "baseline_provenance": baseline_provenance_rows,
        "ltbm_provenance": ltbm_rows,
        "conditions": condition_rows_summary,
        "figures": figures,
        "claim_rule": (
            "Native LTBM token provenance is descriptive. This run has no paired baseline, "
            "so its accuracy alone does not measure compression damage."
            if settings["experiment"] == "provenance" else
            "High similarity alone is descriptive. Temporal redundancy is supported only when "
            "a similarity-based substitution/merge also causes little paired task degradation."),
        "region_warning": "Regions are input log-Mel proxies, not ground-truth VAD/event labels.",
    }
    (report_dir / "summary.json").write_text(
        json.dumps(jsonable(summary), ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {len(figures)} figure files to {report_dir}")


if __name__ == "__main__":
    main()
