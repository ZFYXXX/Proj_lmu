"""Paired full-benchmark statistics, coverage checks and diagnostic figures."""
import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path

import numpy as np

from scripts.analysis.audio_temporal.metrics import jsonable
from .launch import coverage, iter_rows, read_rows
from .attributes import ATTRIBUTE_FIELDS, prepare_attributes
from .plan import EXPERIMENTS


def paired_interval(values, clusters, *, repeats=1000, seed=42):
    """Question-weighted mean; resample independent audio clusters, paired across conditions."""
    values = np.asarray(values, dtype=float)
    units, inverse = np.unique(np.asarray(clusters, dtype=str), return_inverse=True)
    sums = np.bincount(inverse, weights=values)
    counts = np.bincount(inverse)
    rng = np.random.default_rng(seed)
    estimates = []
    for start in range(0, repeats, 100):
        draws = rng.integers(len(units), size=(min(100, repeats - start), len(units)))
        estimates.extend((sums[draws].sum(1) / counts[draws].sum(1)).tolist())
    return dict(mean=float(values.mean()), ci95=np.quantile(estimates, [0.025, 0.975]).tolist(),
                upper95=float(np.quantile(estimates, 0.95)), audio_clusters=len(units), questions=len(values))


def interaction_damage(baseline, removed_a, removed_b, removed_ab):
    baseline = np.asarray(baseline)
    return (baseline - removed_ab) - (baseline - removed_a) - (baseline - removed_b)


def _rank(values):
    values = np.asarray(values)
    ranks = np.zeros(len(values), dtype=float)
    for value in np.unique(values):
        positions = np.flatnonzero(np.sort(values) == value)
        ranks[values == value] = positions.mean()
    return ranks


def correlation(a, b):
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    valid = np.isfinite(a) & np.isfinite(b)
    a, b = a[valid], b[valid]
    if len(a) < 3 or np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return dict(n=len(a), pearson=None, spearman=None)
    return dict(n=len(a), pearson=float(np.corrcoef(a, b)[0, 1]),
                spearman=float(np.corrcoef(_rank(a), _rank(b))[0, 1]))


def mean_finite(values):
    values = [value for value in values if value is not None and np.isfinite(value)]
    return float(np.mean(values)) if values else None


def summarize_stages(rows):
    grouped = defaultdict(list)
    for row in rows:
        for stage in row["stages"]:
            grouped[stage["name"]].append(stage)
    result = {}
    for name, stages in grouped.items():
        result[name] = dict(tokens_mean=float(np.mean([stage["tokens"] for stage in stages])))
        spectrum_rows = [stage["temporal_spectrum"] for stage in stages if stage.get("temporal_spectrum")]
        if spectrum_rows:
            result[name]["temporal_spectrum"] = dict(
                nyquist_fraction_edges=spectrum_rows[0]["nyquist_fraction_edges"],
                nominal_nyquist_hz=spectrum_rows[0]["nominal_nyquist_hz"])
            for key in ("raw", "demean"):
                valid = [row[key] for row in spectrum_rows if row[key] is not None]
                result[name]["temporal_spectrum"][key] = np.mean(valid, axis=0).tolist() if valid else None
        drift = [stage["drift"] for stage in stages if "drift" in stage]
        if drift:
            result[name]["drift"] = {key: mean_finite([r[key] for r in drift]) for key in ("relative_l2", "mean_cosine")}
    return result


def build_summary(directory, *, repeats=1000, tolerance_pp=1.0, experiment=None, checked=None):
    settings = json.loads((directory / "analysis.json").read_text())
    checked = checked or coverage(directory, settings["expected_samples"], settings["conditions"])
    if not checked["complete"]:
        raise ValueError("Incomplete/duplicate/mismatched paired coverage; inspect coverage before reporting")
    baseline = sorted(read_rows(sorted(directory.glob("samples.rank*.jsonl"))), key=lambda row: row["doc_index"])
    if settings["experiment"] == "all":
        if experiment not in settings["experiment_conditions"]:
            raise ValueError("Select an experiment report from the shared all run")
        members = settings["experiment_conditions"][experiment]
        aliases = {row["source_condition_id"]: row["id"] for row in members}
        settings = dict(settings, experiment=experiment,
                        conditions=[{k: v for k, v in row.items() if k != "source_condition_id"} for row in members])
    else:
        if experiment and experiment != settings["experiment"]:
            raise ValueError("Requested report differs from the evaluated experiment")
        aliases = {row["id"]: row["id"] for row in settings["conditions"]}
    by_condition = defaultdict(list)
    if aliases:
        for row in iter_rows(sorted(directory.glob("conditions.rank*.jsonl"))):
            if row["condition_id"] in aliases:
                local_id = aliases[row["condition_id"]]
                by_condition[local_id].append(dict(row, condition_id=local_id))
    for rows in by_condition.values():
        rows.sort(key=lambda row: row["doc_index"])
    scores = np.array([row["accuracy"] for row in baseline], dtype=float)
    clusters = [row.get("audio_path") or f"doc:{row['doc_index']}" for row in baseline]
    result = dict(experiment=settings["experiment"], full_benchmark=settings["full_benchmark"],
        questions=len(baseline), accuracy=float(scores.mean()), coverage=checked,
        accuracy_units="fraction in JSON/CSV; percentage points on damage plots; native lmms-eval accuracy uses percent",
        bootstrap=dict(repeats=repeats, unit="audio_path cluster, question-weighted mean", seed=42),
        tolerance_pp=tolerance_pp, baseline_stages=summarize_stages(baseline), conditions={}, interactions=[],
        scope="MMSU task-conditional evidence; low energy alone is not redundancy; timings include analysis overhead")
    fractions = np.array([row["spectrum"]["power_fraction"] for row in baseline if row["spectrum"]["has_energy"]])
    result["frequency_energy"] = dict(hz=baseline[0]["spectrum"]["hz"],
        nonzero_audio=len(fractions), band_edges_hz=baseline[0]["spectrum"]["band_edges_hz"],
        mean_band_fraction=np.mean([row["spectrum"]["band_fraction"] for row in baseline if row["spectrum"]["has_energy"]], axis=0).tolist()
        if len(fractions) else [0.0] * 8,
        mean_power_fraction=fractions.mean(0).tolist() if len(fractions) else [],
        percentile_power_fraction=np.quantile(fractions, [0.1, 0.5, 0.9], axis=0).tolist() if len(fractions) else [],
        rolloff_hz={q: {"mean": mean_finite([row["spectrum"]["rolloff_hz"][q] for row in baseline])}
                    for q in ("0.85", "0.95", "0.99")})
    proxy_values = defaultdict(list)
    for row in baseline:
        proxy_values["physical_rms"].append(row["spectrum"]["rms"])
        proxy = row.get("acoustic_proxy_summary", {})
        for name in ("median_f0_hz", "voiced_fraction", "proxy_pause_fraction", "proxy_pause_mean_s", "proxy_pause_rate_hz"):
            value = proxy.get(name)
            if value is not None:
                proxy_values[name].append(value)
    result["acoustic_proxy_distributions"] = {}
    for name, values in proxy_values.items():
        values = np.asarray(values, dtype=float)
        values = values[np.isfinite(values)]
        if len(values):
            counts, edges = np.histogram(values, bins=30)
            result["acoustic_proxy_distributions"][name] = dict(
                n=len(values), quantiles_10_50_90=np.quantile(values, [0.1, 0.5, 0.9]).tolist(),
                histogram_counts=counts.tolist(), histogram_edges=edges.tolist())
    categories = sorted({str(row.get("question_type")) for row in baseline})
    for condition in settings["conditions"]:
        rows = by_condition[condition["id"]]
        changed = np.array([row["accuracy"] for row in rows], dtype=float)
        # Verify the row's recorded reference, rather than trusting only its ID.
        if any(float(row["baseline_accuracy"]) != score for row, score in zip(rows, scores)):
            raise ValueError("A condition row's baseline score differs from its paired native answer")
        damage = paired_interval(scores - changed, clusters, repeats=repeats)
        summary = dict(condition=condition, accuracy=float(changed.mean()), damage=damage,
            within_tolerance=bool(damage["upper95"] * 100 <= tolerance_pp) if settings["full_benchmark"] else None,
            stage_metrics=summarize_stages(rows), categories={},
            answer_change_rate=float(np.mean([a["prediction"] != b["prediction"] for a, b in zip(baseline, rows)])),
            mean_output_tokens=float(np.mean([row["projector_tokens"] for row in rows])),
            mean_input_tokens=float(np.mean([row["projector_tokens"] for row in baseline])))
        filtered = [row["waveform_filter"] for row in rows if row.get("waveform_filter")]
        if filtered:
            summary["removed_power_fraction"] = mean_finite([row["removed_power_fraction"] for row in filtered])
            summary["rms_gain_mean"] = mean_finite([row["rms_gain"] for row in filtered])
            summary["degenerate_energy_match_questions"] = sum(row.get("energy_match_degenerate", False) for row in filtered)
        for category in categories:
            mask = np.array([str(row.get("question_type")) == category for row in baseline])
            summary["categories"][category] = dict(n=int(mask.sum()), baseline=float(scores[mask].mean()),
                accuracy=float(changed[mask].mean()), damage=float((scores - changed)[mask].mean()))
        summary["breakdowns"] = {}
        for field in ("sub_category", "task_name", "dataset_name"):
            summary["breakdowns"][field] = {}
            for group in sorted({str(row.get(field)) for row in baseline}):
                mask = np.array([str(row.get(field)) == group for row in baseline])
                summary["breakdowns"][field][group] = dict(n=int(mask.sum()),
                    baseline=float(scores[mask].mean()), accuracy=float(changed[mask].mean()),
                    damage=float((scores - changed)[mask].mean()))
        result["conditions"][condition["id"]] = summary
    for condition in settings["conditions"]:
        bands = condition.get("bands", [])
        if settings["experiment"] != "complementarity" or len(bands) != 2:
            continue
        a, b = bands
        value = interaction_damage(scores,
            np.array([r["accuracy"] for r in by_condition[f"remove_{a}"]]),
            np.array([r["accuracy"] for r in by_condition[f"remove_{b}"]]),
            np.array([r["accuracy"] for r in by_condition[condition["id"]]]))
        result["interactions"].append(dict(bands=bands, **paired_interval(value, clusters, repeats=repeats),
            interpretation="Positive deletion interaction is consistent with backup redundancy; not proof of synergy"))
    band_conditions = [result["conditions"].get(f"band{index}_raw") for index in range(8)]
    if all(band_conditions):
        result["energy_importance_correlation"] = correlation(result["frequency_energy"]["mean_band_fraction"],
            [row["damage"]["mean"] for row in band_conditions])
        result["energy_importance_correlation"]["unit"] = "eight band means, descriptive only"
    identities = {name: row for name, row in result["conditions"].items()
                  if name in {"wave_identity", "time_lowpass1", "mel128_mean"}}
    result["identity_controls"] = {name: dict(answer_change_rate=row["answer_change_rate"],
        max_relative_l2=max(stage["drift"]["relative_l2"] for record in by_condition[name]
                            for stage in record["stages"])) for name, row in identities.items()}
    result["identity_controls_passed"] = (all(row["answer_change_rate"] == 0 and row["max_relative_l2"] <= 1e-5
                                             for row in result["identity_controls"].values()) if identities else None)
    if result["identity_controls_passed"] is False:
        for row in result["conditions"].values():
            row["within_tolerance"] = None
    if settings["experiment"] in ATTRIBUTE_FIELDS:
        result["attribute_analysis"] = summarize_attributes(settings["experiment"], baseline, repeats)
    if settings["experiment"] == "envelope":
        result["envelope_examples"] = []
        for row in baseline:
            if row.get("example_file"):
                with np.load(directory / row["example_file"], allow_pickle=False) as arrays:
                    result["envelope_examples"].append(dict(doc_index=row["doc_index"],
                        time_ms=arrays["target_time_ms"].tolist(), log_rms=arrays["target_log_rms"].tolist()))
    result["inference_scope"] = "Pointwise intervals, not corrected for choosing among many conditions; confirm selected settings separately"
    return result


def summarize_attributes(experiment, baseline, repeats):
    analysis = prepare_attributes(experiment, baseline)
    original = np.array([row["accuracy"] for row in baseline])
    clusters = np.array([row.get("audio_path") or f"doc:{row['doc_index']}" for row in baseline])
    controls = {name: np.array(row["labels"]) for name, row in analysis["control_bins"].items()}
    controls["task_name"] = np.array([str(row.get("task_name")) for row in baseline])
    controls["sub_category"] = np.array([str(row.get("sub_category")) for row in baseline])
    for attribute in analysis["attributes"].values():
        labels = np.array(attribute["labels"])
        attribute["baseline_groups"] = {}
        for group in attribute["groups"]:
            mask = labels == group
            count = int(mask.sum())
            interval = paired_interval(original[mask], clusters[mask], repeats=repeats) if count else None
            metrics = dict(n=count, accuracy=interval["mean"] if interval else None,
                           ci95=interval["ci95"] if interval else None,
                           audio_clusters=interval["audio_clusters"] if interval else 0, controls={})
            for field, control_labels in controls.items():
                metrics["controls"][field] = {}
                for value in sorted(set(control_labels[mask])):
                    selected = mask & (control_labels == value)
                    metrics["controls"][field][value] = dict(
                        n=int(selected.sum()), accuracy=float(original[selected].mean()))
            attribute["baseline_groups"][group] = metrics
    return analysis


def make_figures(result, directory):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    prefix = f"{result['experiment']} | N={result['questions']}" + ("" if result["full_benchmark"] else " | SMOKE ONLY")

    def save(fig, name):
        fig.suptitle(prefix, fontsize=11)
        fig.tight_layout()
        for extension in ("png", "pdf"):
            fig.savefig(directory / f"{name}.{extension}", dpi=160, bbox_inches="tight")
        plt.close(fig)

    energy = result["frequency_energy"]
    if result["acoustic_proxy_distributions"]:
        fig, axes = plt.subplots(2, 3, figsize=(10, 6))
        for axis, (name, distribution) in zip(axes.flat, result["acoustic_proxy_distributions"].items()):
            edges = np.asarray(distribution["histogram_edges"])
            axis.bar(edges[:-1], distribution["histogram_counts"], width=np.diff(edges), align="edge")
            axis.set(xlabel=name, ylabel="Questions", title=f"Valid N={distribution['n']}")
        for axis in list(axes.flat)[len(result["acoustic_proxy_distributions"]):]:
            axis.set_visible(False)
        save(fig, "acoustic_proxy_distributions")
    if energy["nonzero_audio"]:
        fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
        hz = energy["hz"]
        q10, median, q90 = np.array(energy["percentile_power_fraction"])
        axes[0].fill_between(hz, q10, q90, alpha=0.25, label="10–90% clips")
        axes[0].plot(hz, energy["mean_power_fraction"], label="mean normalized power")
        axes[0].set(xlabel="Acoustic frequency (Hz)", ylabel="Power fraction per FFT bin", yscale="log")
        axes[0].legend(fontsize=7)
        axes[1].plot(hz, np.cumsum(energy["mean_power_fraction"]))
        axes[1].set(xlabel="Acoustic frequency (Hz)", ylabel="Mean cumulative energy", ylim=(0, 1.02))
        save(fig, "frequency_energy")
    if result["conditions"]:
        names = list(result["conditions"])
        rows = list(result["conditions"].values())
        means = np.array([row["damage"]["mean"] * 100 for row in rows])
        lower = np.array([row["damage"]["ci95"][0] * 100 for row in rows])
        upper = np.array([row["damage"]["ci95"][1] * 100 for row in rows])
        fig, axis = plt.subplots(figsize=(9, max(3.5, 0.27 * len(names))))
        axis.hlines(range(len(names)), lower, upper, color="tab:blue")
        axis.plot(means, range(len(names)), "o", ms=4)
        axis.set(yticks=range(len(names)), yticklabels=names, xlabel="Accuracy damage (percentage points), paired 95% CI")
        axis.axvline(0, color="black", lw=0.7)
        axis.axvline(result["tolerance_pp"], color="tab:red", ls="--", lw=0.8)
        axis.invert_yaxis()
        save(fig, "paired_damage")
        stages = list(result["baseline_stages"])
        heat = [[row["stage_metrics"][stage].get("drift", {}).get("relative_l2", np.nan) for stage in stages] for row in rows]
        fig, axis = plt.subplots(figsize=(7, max(3.5, 0.25 * len(names))))
        display = axis.imshow(heat, aspect="auto", vmin=0)
        axis.set(xticks=range(len(stages)), xticklabels=stages, yticks=range(len(names)), yticklabels=names)
        fig.colorbar(display, ax=axis, label="Representation relative L2 on native grid")
        save(fig, "condition_layer_drift")
        categories = list(rows[0]["categories"])
        fig, axis = plt.subplots(figsize=(max(6, len(categories)), max(3.5, 0.25 * len(names))))
        values = [[row["categories"][c]["damage"] * 100 for c in categories] for row in rows]
        display = axis.imshow(values, aspect="auto", cmap="RdBu_r")
        axis.set(xticks=range(len(categories)), xticklabels=categories, yticks=range(len(names)), yticklabels=names)
        axis.tick_params(axis="x", rotation=30)
        fig.colorbar(display, ax=axis, label="Accuracy damage (pp), descriptive by native question_type")
        save(fig, "condition_category_damage")
    if "energy_importance_correlation" in result:
        fig, axis = plt.subplots(figsize=(5, 4))
        for index, energy_fraction in enumerate(energy["mean_band_fraction"]):
            damage = result["conditions"][f"band{index}_raw"]["damage"]["mean"] * 100
            axis.scatter(energy_fraction * 100, damage)
            axis.annotate(f"{index}–{index + 1} kHz", (energy_fraction * 100, damage), fontsize=8)
        axis.set(xlabel="Mean band energy (%)", ylabel="Accuracy damage (pp)")
        save(fig, "band_energy_importance")
    if result["interactions"]:
        values = np.full((4, 4), np.nan)
        for item in result["interactions"]:
            a, b = item["bands"]
            values[a, b] = values[b, a] = item["mean"] * 100
        fig, axis = plt.subplots(figsize=(5, 4))
        display = axis.imshow(values, cmap="RdBu_r")
        axis.set(xlabel="2 kHz band index", ylabel="2 kHz band index")
        fig.colorbar(display, ax=axis, label="Joint minus summed deletion damage (pp)")
        save(fig, "frequency_interaction")
    stage_names = list(result["baseline_stages"])
    spectra = [result["baseline_stages"][name].get("temporal_spectrum", {}).get("demean") for name in stage_names]
    if all(item is not None for item in spectra):
        fig, axis = plt.subplots(figsize=(8, 3.5))
        display = axis.imshow(spectra, aspect="auto", extent=(0, 1, len(stage_names) - 0.5, -0.5))
        axis.set(yticks=range(len(stage_names)), yticklabels=stage_names,
                 xlabel="Temporal modulation / stage Nyquist (not acoustic Hz)")
        fig.colorbar(display, ax=axis, label="Demeaned power fraction")
        save(fig, "representation_temporal_spectrum")
    for name, attribute in result.get("attribute_analysis", {}).get("attributes", {}).items():
        groups = attribute["groups"]
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        values = [v for v in attribute["values"] if v is not None]
        if values:
            axes[0].hist(values, bins=30)
        for cutoff in attribute["cutoffs"]:
            axes[0].axvline(cutoff, color="tab:red", ls="--")
        axes[0].set(xlabel=f"{name} ({attribute['unit']})", ylabel="Questions",
                    title=f"Valid {len(values)}/{len(attribute['values'])}; input-only cutoffs")
        metrics = [attribute["baseline_groups"][group] for group in groups]
        for index, row in enumerate(metrics):
            if row["n"]:
                axes[1].plot(index, row["accuracy"] * 100, "o", color="tab:blue")
                axes[1].vlines(index, row["ci95"][0] * 100, row["ci95"][1] * 100, color="tab:blue")
        axes[1].set(xticks=range(len(groups)),
                    xticklabels=[f"{group}\nN={row['n']}" for group, row in zip(groups, metrics)],
                    ylabel="Native MMSU accuracy (%), 95% CI", ylim=(-2, 102),
                    title="Original input only; descriptive association")
        save(fig, f"attribute_{name}")
    examples = result.get("envelope_examples", [])
    if examples:
        fig, axes = plt.subplots(len(examples), 1, figsize=(10, 2.5 * len(examples)), squeeze=False)
        for axis, example in zip(axes.flat, examples):
            axis.plot(np.asarray(example["time_ms"]) / 1000, example["log_rms"], lw=0.8)
            axis.set(xlabel="Time (s)", ylabel="Original log-RMS", title=f"doc {example['doc_index']}")
        save(fig, "envelope_examples")


def write_report(result, target):
    target.mkdir(parents=True, exist_ok=True)
    (target / "summary.json").write_text(json.dumps(jsonable(result), ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    with (target / "conditions.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["condition", "accuracy", "damage_pp", "ci95_low_pp", "ci95_high_pp", "upper95_pp", "within_tolerance", "output_tokens_mean"])
        for name, row in result["conditions"].items():
            damage = row["damage"]
            writer.writerow([name, row["accuracy"], 100 * damage["mean"], *[100 * v for v in damage["ci95"]],
                             100 * damage["upper95"], row["within_tolerance"], row["mean_output_tokens"]])
    make_figures(result, target)
    with (target / "category_damage.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["condition", "grouping", "group", "n", "baseline_accuracy", "accuracy", "damage_pp"])
        for name, row in result["conditions"].items():
            for field, groups in {"question_type": row["categories"], **row["breakdowns"]}.items():
                for group, metrics in groups.items():
                    writer.writerow([name, field, group, metrics["n"], metrics["baseline"], metrics["accuracy"], 100 * metrics["damage"]])
    if "attribute_analysis" in result:
        write_attribute_tables(result["attribute_analysis"], target)
    print(f"Wrote {target}; full benchmark={result['full_benchmark']}; N={result['questions']}")


def write_attribute_tables(analysis, target):
    with (target / "attribute_assignments.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        fields = ["doc_index", "audio_path", "task_name", "sub_category", "duration_s", "native_tokens", "valid_f0_frames"]
        writer.writerow(fields + ["attribute", "unit", "value", "group", "duration_group", "native_tokens_group"])
        for name, attribute in analysis["attributes"].items():
            for i, record in enumerate(analysis["records"]):
                writer.writerow([record.get(key) for key in fields] +
                    [name, attribute["unit"], attribute["values"][i], attribute["labels"][i],
                     analysis["control_bins"]["duration_s"]["labels"][i],
                     analysis["control_bins"]["native_tokens"]["labels"][i]])
    with (target / "attribute_baseline.csv").open("w", encoding="utf-8", newline="") as handle, \
            (target / "attribute_controls.csv").open("w", encoding="utf-8", newline="") as controls:
        writer, control_writer = csv.writer(handle), csv.writer(controls)
        writer.writerow(["attribute", "unit", "cutoff_low", "cutoff_high", "group", "n", "audio_clusters",
                         "accuracy", "ci95_low", "ci95_high"])
        control_writer.writerow(["attribute", "group", "control", "control_group", "n", "accuracy"])
        for name, attribute in analysis["attributes"].items():
            for group, metrics in attribute["baseline_groups"].items():
                writer.writerow([name, attribute["unit"], *(attribute["cutoffs"] or [None, None]), group,
                                 metrics["n"], metrics["audio_clusters"], metrics["accuracy"],
                                 *(metrics["ci95"] or [None, None])])
                for field, breakdown in metrics["controls"].items():
                    for value, row in breakdown.items():
                        control_writer.writerow([name, group, field, value, row["n"], row["accuracy"]])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--experiment", choices=EXPERIMENTS, help="Rebuild one view of a shared all run without inference")
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--tolerance-pp", type=float, default=1.0,
                        help="Prespecified allowed accuracy loss; do not choose it after viewing results")
    args = parser.parse_args()
    if args.bootstrap < 1 or args.tolerance_pp < 0:
        parser.error("Bootstrap repeats must be positive and tolerance nonnegative")
    directory = Path(args.run_dir).expanduser().resolve()
    settings = json.loads((directory / "analysis.json").read_text())
    checked = coverage(directory, settings["expected_samples"], settings["conditions"])
    if not checked["complete"]:
        raise ValueError("Incomplete paired run; no reports were generated")
    shared = settings["experiment"] == "all"
    experiments = [args.experiment] if args.experiment else list(EXPERIMENTS) if shared else [settings["experiment"]]
    for experiment in experiments:
        result = build_summary(directory, repeats=args.bootstrap, tolerance_pp=args.tolerance_pp,
                               experiment=experiment, checked=checked)
        target = directory / "report" / experiment if shared else directory / "report"
        write_report(result, target)
    if shared and not args.experiment:
        index = dict(experiments={name: f"{name}/summary.json" for name in experiments},
                     coverage=checked, expected_generation_calls=settings["expected_generation_calls"],
                     independent_generation_calls=settings["independent_generation_calls"],
                     generation_calls_saved=settings["generation_calls_saved"],
                     scope=f"One shared baseline and each distinct intervention on every question; {len(experiments)} report views; native token counts preserved")
        (directory / "report" / "summary.json").write_text(json.dumps(index, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
