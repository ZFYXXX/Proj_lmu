"""Coverage-checked paired reports, task breakdowns and scientific diagnostics."""
import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path

import numpy as np

from .common import coverage, rows, write_json


def paired_interval(values, audio_paths, repeats=1000, seed=42):
    values = np.asarray(values, dtype=float)
    units, inverse = np.unique(np.asarray(audio_paths, dtype=str), return_inverse=True)
    sums, counts = np.bincount(inverse, weights=values), np.bincount(inverse)
    rng = np.random.default_rng(seed)
    estimates = []
    for start in range(0, repeats, 100):
        draws = rng.integers(len(units), size=(min(100, repeats - start), len(units)))
        estimates.extend((sums[draws].sum(1) / counts[draws].sum(1)).tolist())
    return dict(mean=float(values.mean()), ci95=np.quantile(estimates, [.025, .975]).tolist(),
                lower95=float(np.quantile(estimates, .05)), independent_audios=len(units))


def csv_write(path, records):
    records = list(records)
    if not records:
        return
    fields = list(dict.fromkeys(key for row in records for key in row))
    with Path(path).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)


def summarize(directory):
    root = Path(directory)
    settings = json.loads((root / "analysis.json").read_text())
    evaluation = root / "evaluation"
    checked = coverage(evaluation, settings["expected_samples"], settings["conditions"])
    write_json(evaluation / "coverage.json", checked)
    if not checked["complete"]:
        raise ValueError("Cannot report incomplete/duplicate/misaligned paired coverage")
    baseline = sorted(rows(evaluation.glob("samples.rank*.jsonl")), key=lambda r: r["doc_index"])
    n = len(baseline)
    scores = np.array([r["accuracy"] for r in baseline])
    audio = np.array([r["audio_path"] for r in baseline])
    groups = {"all": np.arange(n)}
    for field in ("question_type", "sub_category", "task_name"):
        labels = np.array([str(r.get(field, "Unknown")) for r in baseline])
        for label in sorted(set(labels)):
            groups[f"{field}:{label}"] = np.flatnonzero(labels == label)
    # Dense numeric arrays avoid keeping hundreds of thousands of JSON dicts.
    metrics = {c["id"]: {key: np.full(n, np.nan) for key in (
        "accuracy", "retention", "tokens_per_second", "retained_k", "mse", "preparation_seconds", "generation_seconds")}
        for c in settings["conditions"]}
    mismatched_identity = []
    unmatched_smoothing = defaultdict(int)
    clamped_budgets = defaultdict(int)
    for row in rows(evaluation.glob("conditions.rank*.jsonl")):
        index, name = row["doc_index"], row["condition_id"]
        if (row["id"] != baseline[index]["id"] or row["audio_path"] != baseline[index]["audio_path"]
                or row["baseline_accuracy"] != scores[index]):
            raise ValueError(f"Condition {name} has mismatched baseline metadata at {index}")
        data = metrics[name]
        data["accuracy"][index] = row["accuracy"]
        details = row["intervention"]
        for key in ("retention", "tokens_per_second", "retained_k", "mse"):
            value = details.get(key)
            if value is not None:
                data[key][index] = value
        data["preparation_seconds"][index] = row.get("analysis_preparation_seconds", np.nan)
        data["generation_seconds"][index] = row.get("diagnostic_generation_seconds", np.nan)
        if row["condition"]["method"] == "identity" and row["prediction"] != baseline[index]["prediction"]:
            mismatched_identity.append(dict(condition=name, doc_index=index))
        if details.get("matched") is False:
            unmatched_smoothing[name] += 1
        clamped_budgets[name] += int(details.get("budget_clamped_by_hard_chunks", False))
    summaries, task_rows, comparisons = {}, [], []

    def finite_mean(x):
        x = x[np.isfinite(x)]
        return float(x.mean()) if len(x) else None

    for condition in settings["conditions"]:
        name = condition["id"]
        data = metrics[name]
        for label, indices in groups.items():
            delta = data["accuracy"][indices] - scores[indices]
            interval = paired_interval(delta, audio[indices], settings["bootstrap_repeats"])
            row = dict(condition_id=name, group=label, questions=len(indices),
                baseline_accuracy=float(scores[indices].mean()), accuracy=float(data["accuracy"][indices].mean()),
                delta_pp=100 * interval["mean"], ci95_low_pp=100 * interval["ci95"][0],
                ci95_high_pp=100 * interval["ci95"][1], one_sided_lower95_pp=100 * interval["lower95"],
                within_tolerance=100 * interval["lower95"] >= -settings["tolerance_pp"],
                correct_to_wrong=int(np.sum((scores[indices] == 1) & (data["accuracy"][indices] == 0))),
                wrong_to_correct=int(np.sum((scores[indices] == 0) & (data["accuracy"][indices] == 1))),
                retention=finite_mean(data["retention"][indices]))
            task_rows.append(row)
            if label == "all":
                summaries[name] = dict(**row, condition=condition,
                    means={key: finite_mean(value) for key, value in data.items()},
                    unmatched_smoothing_samples=unmatched_smoothing[name],
                    budget_clamped_samples=clamped_budgets[name])
        if condition["family"] == "segment" and condition["method"] not in {"uniform"} and not condition.get("broadcast"):
            reference = f"uniform_r{condition['ratio']:g}"
        elif condition["family"] == "runs" and condition["method"] == "collapse":
            reference = f"E_matched_uniform_K{condition['codebook_size']}"
        else:
            continue
        if reference in metrics:
            equal = np.allclose(data["retention"], metrics[reference]["retention"], atol=0, rtol=0)
            if not equal:
                raise ValueError(f"Matched-token control differs: {name} vs {reference}")
            interval = paired_interval(data["accuracy"] - metrics[reference]["accuracy"], audio, settings["bootstrap_repeats"])
            comparisons.append(dict(condition_id=name, reference=reference, matched_budget=True,
                accuracy_advantage_pp=100 * interval["mean"], ci95_low_pp=100 * interval["ci95"][0],
                ci95_high_pp=100 * interval["ci95"][1]))
    summary = dict(experiments=settings["experiments"], questions=n,
        full_mmsufinal=settings["full_mmsufinal"], baseline_accuracy=float(scores.mean()),
        tolerance_pp=settings["tolerance_pp"], conditions=summaries, matched_comparisons=comparisons,
        identity_prediction_mismatches=mismatched_identity, identity_checks_passed=not mismatched_identity,
        interpretation_valid=not mismatched_identity,
        units="accuracy fractions; delta and intervals percentage points; entropy bits",
        inference="paired bootstrap by audio; descriptive intervals, no multiple-comparison correction",
        limitations=["Task-conditional evidence on MMSU speech understanding.",
            "A uses per-audio PCA, with rank capped by observed token count.",
            "B is cross-fitted Markov uncertainty of bidirectionally contextualized hidden units.",
            "B/C/E shrink P_in; encoder compute is unchanged and duration is not an extra model input.",
            "H residual squared norm is not waveform energy; inspect unmatched smoothing counts.",
            "Timings are diagnostic with shared reference/SVD/unit preparation; no standalone acceleration claim."])
    return settings, baseline, summary, task_rows


def stage_tables(baseline):
    rank_rows, unit_rows = [], []
    for sample in baseline:
        for stage in sample["stages"]:
            base = dict(doc_index=sample["doc_index"], audio_path=sample["audio_path"], stage=stage["name"])
            if "rank" in stage:
                for name in ("raw", "centered"):
                    rank_rows.append(dict(**base, scope=name, **stage["rank"][name]))
                for width, values in stage["rank"]["local"].items():
                    rank_rows.append(dict(**base, scope=f"local_{width}ms", **values))
            for k, values in stage.get("discrete", {}).items():
                unit_rows.append(dict(**base, **{key: value for key, value in values.items()
                                                if not isinstance(value, list)}))
    return rank_rows, unit_rows


def figures(root, settings, baseline, summary, rank_rows, unit_rows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    report_dir = root / "report"
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                         "pdf.fonttype": 42, "ps.fonttype": 42})

    def save(fig, name):
        fig.savefig(report_dir / f"{name}.png", dpi=180, bbox_inches="tight")
        fig.savefig(report_dir / f"{name}.pdf", bbox_inches="tight")
        plt.close(fig)

    def mean(values):
        valid = [x for x in values if x is not None and np.isfinite(x)]
        return float(np.mean(valid)) if valid else np.nan

    def heatmap(matrix, xlabels, ylabels, title, filename):
        fig, ax = plt.subplots(figsize=(max(5, len(xlabels) * 1.1), max(3, len(ylabels) * .42)))
        im = ax.imshow(np.ma.masked_invalid(matrix), aspect="auto", cmap="viridis")
        ax.set(xticks=range(len(xlabels)), xticklabels=xlabels, yticks=range(len(ylabels)), yticklabels=ylabels, title=title)
        plt.setp(ax.get_xticklabels(), rotation=25, ha="right")
        fig.colorbar(im, ax=ax)
        fig.tight_layout()
        save(fig, filename)

    stages = settings["stages"]
    if rank_rows:
        scopes = ["raw", "centered"] + [f"local_{width}ms" for width in settings["windows_ms"]]
        matrix = np.array([[mean([r.get("normalized_effective_rank") for r in rank_rows
                                 if r["stage"] == stage and r["scope"] == scope]) for scope in scopes] for stage in stages])
        heatmap(matrix, scopes, stages, "A: effective rank / available rank", "A_rank_heatmap")
    if unit_rows:
        for metric, title in (("transition_entropy_bits", "E: empirical transition entropy (bits)"),
                              ("predictive_entropy_bits", "B: held-out predictive entropy (bits)"),
                              ("mean_run_length", "E: mean run length (native tokens)")):
            matrix = np.array([[mean([r[metric] for r in unit_rows if r["stage"] == stage and r["codebook_size"] == k])
                                for k in settings["codebook_sizes"]] for stage in stages])
            heatmap(matrix, settings["codebook_sizes"], stages, title, metric)

    for experiment in settings["experiments"]:
        fig, ax = plt.subplots(figsize=(7, 4))
        curves = defaultdict(list)
        for name in settings["experiment_conditions"][experiment]:
            row = summary["conditions"][name]
            condition = row["condition"]
            if condition["method"] == "identity":
                continue
            if experiment == "A":
                label = f"{condition['stage']} {condition['method']}"
                x = condition["k"]
            elif experiment == "H":
                label, x = condition["method"], condition["k"]
            else:
                label = condition["method"] + (f" K={condition['codebook_size']}" if "codebook_size" in condition else "")
                if condition.get("broadcast"):
                    label += " broadcast"
                x = 100 * row["retention"]
            curves[label].append((x, row))
        for label, points in curves.items():
            points.sort(key=lambda item: item[0])
            x = [p[0] for p in points]
            y = np.array([p[1]["delta_pp"] for p in points])
            low = np.array([p[1]["ci95_low_pp"] for p in points])
            high = np.array([p[1]["ci95_high_pp"] for p in points])
            ax.errorbar(x, y, yerr=np.maximum(0, [y - low, high - y]), marker="o", capsize=2, label=label)
        ax.axhline(0, color="black", lw=.7)
        ax.axhline(-settings["tolerance_pp"], color="grey", ls="--", lw=.7)
        ax.set(xlabel="Requested retained dimensions" if experiment == "A" else "Residual rank" if experiment == "H"
               else "Realized retained tokens (%)", ylabel="Accuracy change (percentage points)", title=f"{experiment}: paired MMSU-final")
        ax.legend(fontsize=7, loc="best")
        fig.tight_layout()
        save(fig, f"{experiment}_performance")

    for sample in baseline:
        if "example_file" not in sample:
            continue
        index = sample["doc_index"]
        with np.load(root / "evaluation" / sample["example_file"], allow_pickle=False) as cache:
            example = {key: cache[key] for key in cache.files}
        mel = example["mel"]
        extent = [0, sample["duration_seconds"], 0, mel.shape[0]]

        def mel_axis(ax):
            ax.imshow(mel, origin="lower", aspect="auto", extent=extent, cmap="magma")
            ax.set(ylabel="Mel bin")

        if "A" in settings["experiments"]:
            fig, ax = plt.subplots(figsize=(6, 4))
            for stage in stages:
                singular = example[f"singular_{stage}"]
                if singular.sum() > 0:
                    ax.semilogy(np.arange(1, len(singular) + 1), np.maximum(singular / singular.sum(), 1e-12), label=stage)
            ax.set(xlabel="Singular component", ylabel="Normalized singular value", title=f"A: sample {index}")
            ax.legend(fontsize=7)
            save(fig, f"A_spectrum_doc{index}")
            fig, axes = plt.subplots(2, 1, sharex=True, figsize=(9, 5))
            mel_axis(axes[0])
            for width in settings["windows_ms"]:
                values = example[f"local_P_in_{width:g}"]
                if len(values):
                    axes[1].plot(values[:, 0] / 1000, values[:, 2], ".-", label=f"{width:g} ms")
            axes[1].set(xlabel="Time (s)", ylabel="Normalized effective rank")
            axes[1].legend()
            save(fig, f"A_local_rank_doc{index}")
        if set(settings["experiments"]) & {"B", "E"}:
            k = settings["codebook_sizes"][0]
            time = example["time_P_in"] / 1000
            fig, axes = plt.subplots(3, 1, sharex=True, figsize=(10, 6), gridspec_kw={"height_ratios": [2, 1, .45]})
            mel_axis(axes[0])
            axes[1].plot(time, example[f"entropy_P_in_K{k}"], label="Predictive entropy")
            axes[1].plot(time, example[f"surprisal_P_in_K{k}"], alpha=.65, label="Surprisal")
            axes[1].set(ylabel="Bits")
            axes[1].legend()
            axes[2].imshow(example[f"codes_P_in_K{k}"][None, :], aspect="auto", interpolation="nearest",
                           extent=[time[0], time[-1], 0, 1], cmap="tab20")
            axes[2].set(xlabel="Time (s)", ylabel="Unit ID", yticks=[])
            for condition in settings["conditions"]:
                if condition["family"] == "segment" and condition["method"] == "entropy" and condition.get("codebook_size") == k:
                    filename = root / "evaluation" / f"example.doc{index}.{condition['id']}.npz"
                    with np.load(filename) as retained:
                        positions = retained["retained_time_ms"] / 1000
                    axes[0].plot(positions, np.full(len(positions), mel.shape[0] * .95), ".", color="cyan", ms=2)
                    axes[0].set_title(f"B: retained positions, requested ratio {condition['ratio']:g}")
                    break
            save(fig, f"BE_units_entropy_doc{index}")
        if "C" in settings["experiments"]:
            chosen = []
            available_ratios = sorted({c["ratio"] for c in settings["conditions"] if c["family"] == "segment"})
            ratio = min(available_ratios, key=lambda value: abs(value - .25))
            for method in ("uniform", "cosine", "l2"):
                chosen.extend(c for c in settings["conditions"] if c["family"] == "segment"
                              and c["method"] == method and c["ratio"] == ratio and not c.get("broadcast"))
            fig, axes = plt.subplots(1 + len(chosen), 1, sharex=True, figsize=(10, 3 + len(chosen)))
            mel_axis(axes[0])
            native_times = example["time_P_in"] / 1000
            for ax, condition in zip(axes[1:], chosen):
                with np.load(root / "evaluation" / f"example.doc{index}.{condition['id']}.npz") as segments:
                    for a, b in zip(segments["segment_start"], segments["segment_end"]):
                        left = max(0, native_times[a] - .02)
                        right = min(sample["duration_seconds"], native_times[b - 1] + .02)
                        ax.axvspan(left, right, ymin=0, ymax=1, color=plt.cm.viridis(1 / (b - a)), ec="white", lw=.4)
                    ax.plot(segments["retained_time_ms"] / 1000, np.full(len(segments["retained_time_ms"]), .5), "k.", ms=2)
                ax.set(ylabel=condition["method"], yticks=[])
            axes[-1].set_xlabel("Time (s)")
            save(fig, f"C_segments_doc{index}")
        if "E" in settings["experiments"]:
            stage = next(s for s in sample["stages"] if s["name"] == "P_in")
            fig, ax = plt.subplots(figsize=(7, 3))
            for k, values in stage["discrete"].items():
                ax.plot(values["run_length_bin_labels"], values["run_length_histogram"], ".-", label=f"K={k}")
            ax.set(xlabel="Run length (native tokens)", ylabel="Number of runs", title=f"E: sample {index}")
            ax.legend()
            save(fig, f"E_run_lengths_doc{index}")
        if "H" in settings["experiments"]:
            chosen = [c for c in settings["conditions"] if c["family"] == "mel" and c["method"] == "svd"][:4]
            fig, axes = plt.subplots(1 + len(chosen), 1, sharex=True, figsize=(9, 2 * (1 + len(chosen))))
            axes[0].imshow(mel, origin="lower", aspect="auto", extent=extent, cmap="magma", vmin=mel.min(), vmax=mel.max())
            axes[0].set_ylabel("Original")
            for ax, condition in zip(axes[1:], chosen):
                with np.load(root / "evaluation" / f"example.doc{index}.{condition['id']}.npz") as modified:
                    ax.imshow(modified["mel"], origin="lower", aspect="auto", extent=extent, cmap="magma", vmin=mel.min(), vmax=mel.max())
                ax.set_ylabel(f"Rank {condition['k']}")
            axes[-1].set_xlabel("Time (s)")
            save(fig, f"H_mel_doc{index}")
            fig, ax = plt.subplots(figsize=(6, 3))
            singular = example["singular_Mel"]
            ax.semilogy(np.arange(1, len(singular) + 1), np.maximum(singular, 1e-12))
            ax.set(xlabel="Residual singular component", ylabel="Singular value", title=f"H: sample {index}")
            save(fig, f"H_spectrum_doc{index}")


def report(directory):
    root = Path(directory)
    settings, baseline, summary, task_rows = summarize(root)
    output = root / "report"
    output.mkdir(exist_ok=True)
    rank_rows, unit_rows = stage_tables(baseline)
    write_json(output / "summary.json", summary)
    csv_write(output / "paired_task_results.csv", task_rows)
    csv_write(output / "matched_budget_comparisons.csv", summary["matched_comparisons"])
    csv_write(output / "rank_by_audio.csv", rank_rows)
    csv_write(output / "units_by_audio.csv", unit_rows)
    figures(root, settings, baseline, summary, rank_rows, unit_rows)
    experiment_names = "/".join(settings["experiments"])
    lines = [f"# MMSU-final {experiment_names} 报告", "", f"题数：{len(baseline)}；原始准确率：{100 * summary['baseline_accuracy']:.2f}%。",
             f"恒等干预预测完全一致：{summary['identity_checks_passed']}。完整 5000 题：{settings['full_mmsufinal']}。", "",
             "差值为干预 − 原始准确率（百分点），CI 按音频配对 bootstrap。统计区间未做多重比较校正。",
             "“满足容忍度”要求单侧 95% 下界不低于预设负容忍度，不等同于证明完全无损。", "",
             "| 条件 | 保留率 | ΔAcc (pp) | 95% CI | 满足容忍度 |", "|---|---:|---:|---|---|"]
    for name, row in summary["conditions"].items():
        lines.append(f"| {name} | {row['retention']:.3f} | {row['delta_pp']:+.2f} | "
                     f"[{row['ci95_low_pp']:+.2f}, {row['ci95_high_pp']:+.2f}] | {row['within_tolerance']} |")
    lines.extend(["", "逐题结果在 evaluation/*.jsonl；分题型结果在 paired_task_results.csv；匹配预算比较另表保存。",
                  "恒等干预不一致时，先检查数值后端／接线，不能据此解释冗余结论。", "",
                  "运行耗时含分析、参考表征和缓存准备，不是独立部署速度。P_in 压缩不减少音频 encoder FLOPs。"])
    (output / "README.zh.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Report: {output}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    report(parser.parse_args().run_dir)
