"""Stream saved probe artifacts into task/group tables and diagnostic figures."""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import json
import math
from pathlib import Path
import statistics
from typing import Any, Iterator

import yaml

from mllm_comp.analysis.common.attention_flow import AttentionFlowSampleAccumulator, attention_flow_metrics
from mllm_comp.analysis.common.distribution import DISTRIBUTION_METRICS
from mllm_comp.analysis.common.prediction import PREDICTION_METRICS, PredictionSampleAccumulator
from mllm_comp.analysis.common.representation import RepresentationSampleAccumulator
from mllm_comp.analysis.common.residual import RESIDUAL_METRICS, ResidualSampleAccumulator
from mllm_comp.core.serialization import atomic_write_json


GROUP_FIELDS = ("correctness", "audio_type", "category", "sub_category", "difficulty", "source_task")


def iter_raw_samples(path: Path) -> Iterator[dict]:
    """Read the one-object-per-line array written by write_analysis_sample_artifact.

    Do not json.load a multi-gigabyte attention artifact. This deliberately follows
    the platform writer's format rather than implementing another JSON parser.
    """
    inside = False
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not inside:
                if stripped.startswith('"samples": ['):
                    if stripped != '"samples": [':
                        raise ValueError(f"Unexpected empty/inline samples array: {path}")
                    inside = True
                continue
            if stripped == "],":
                return
            if stripped:
                yield json.loads(stripped.removesuffix(","))
    raise ValueError(f"Incomplete platform per-sample artifact: {path}")


def make_accumulator(spec: dict):
    name = spec["name"]
    if name == "representation_dynamics":
        return RepresentationSampleAccumulator()
    if name == "prediction_dynamics":
        return PredictionSampleAccumulator(metrics=spec.get("metrics", PREDICTION_METRICS))
    if name == "attention_flow":
        return AttentionFlowSampleAccumulator(metrics=spec.get(
            "metrics", attention_flow_metrics(spec["token_groups"])))
    if name == "residual_stream_decomposition":
        metrics = spec.get("metrics", (*RESIDUAL_METRICS, *DISTRIBUTION_METRICS))
        return ResidualSampleAccumulator(
            residual_metrics=[metric for metric in RESIDUAL_METRICS if metric in metrics],
            distribution_metrics=[metric for metric in DISTRIBUTION_METRICS if metric in metrics],
            token_groups=spec["token_groups"])
    raise ValueError(name)


def groups_for(row: dict, medians: dict) -> list[tuple[str, str, str]]:
    task = row["task_name"]
    result = [(task, "all", "all")]
    result.extend((task, field, str(row[field])) for field in GROUP_FIELDS
                  if row.get(field) not in (None, "", "unknown"))
    length = "short" if row["duration_seconds"] < medians[task] else "long"
    result.append((task, "length", length))
    return result


def verify_sample(row: dict, spec: dict, head_count: int) -> None:
    expected = {str(index) for index in spec["layers"]}
    name = spec["name"]
    actual = row.get("residual_stats" if name == "residual_stream_decomposition" else "layers", {})
    if set(actual) != expected:
        raise ValueError(f"{name}: incomplete layers for {row['sample_id']}")
    if row.get("token_counts", {}).get("audio", 0) <= 0:
        raise ValueError(f"{name}: no audio tokens for {row['sample_id']}")
    if name == "attention_flow":
        expected_heads = {str(index) for index in range(head_count)}
        if any(set(layer.get("heads", {})) != expected_heads for layer in actual.values()):
            raise ValueError(f"Attention head coverage is incomplete: {row['sample_id']}")
    if name == "residual_stream_decomposition":
        causal = row.get("causal", {})
        if set(causal) != {str(index) for index in spec["causal_layers"]}:
            raise ValueError(f"Causal layer coverage is incomplete: {row['sample_id']}")
        operations = {f"remove_{operation}_update" for operation in spec["causal_operations"]}
        for layer in causal.values():
            if set(layer) != operations or any(value.get("status") != "finite" for value in layer.values()):
                raise ValueError(f"Causal operation coverage is incomplete: {row['sample_id']}")


def read_metadata(run_dir: Path, plan: dict) -> tuple[dict, dict]:
    metadata, inventories = {}, []
    for rank in range(plan["shard_count"]):
        shard = run_dir / f"shard-{rank:03d}"
        manifest = json.loads((shard / "manifest.json").read_text())
        if manifest.get("invariants_passed") is not True:
            raise ValueError(f"Incomplete or failed shard: {shard}")
        inventory = json.loads((shard / "dataset_inventory.json").read_text())
        inventories.append(inventory)
        counts = Counter()
        with (shard / "samples.jsonl").open(encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                identity = row["sample_id"]
                if identity in metadata:
                    raise ValueError(f"Duplicate sample across shards: {identity}")
                if row["doc_id"] % plan["shard_count"] != rank:
                    raise ValueError(f"Incorrect shard assignment: {identity}")
                if row["task_name"] not in plan["leaf_tasks"]:
                    raise ValueError(f"Unexpected task: {row['task_name']}")
                metadata[identity] = row
                counts[row["task_name"]] += 1
        if any(counts[name] != item["assigned_rows"] for name, item in inventory.items()):
            raise ValueError(f"Missing baseline/scoring records in {shard}")
    coverage = {}
    for task in plan["leaf_tasks"]:
        rows = [row for row in metadata.values() if row["task_name"] == task]
        selected = inventories[0][task]["selected_rows"]
        total = inventories[0][task]["dataset_rows"]
        if any((item[task]["selected_rows"], item[task]["dataset_rows"]) != (selected, total)
               for item in inventories):
            raise ValueError(f"Dataset inventory differs across shards: {task}")
        if {row["doc_id"] for row in rows} != set(range(selected)):
            raise ValueError(f"Incomplete document coverage for {task}")
        scores = [row["accuracy"] for row in rows if row["accuracy"] is not None]
        coverage[task] = {
            "dataset_rows": total, "selected_rows": selected, "completed_rows": len(rows),
            "full_dataset": selected == total, "labeled_rows": len(scores),
            "unlabeled_rows": len(rows) - len(scores),
            "accuracy": statistics.mean(scores) if scores else None,
        }
    return metadata, coverage


def numeric_fields(value: Any, prefix: str = "") -> Iterator[tuple[str, float]]:
    if isinstance(value, dict):
        for key, item in value.items():
            if key in {"mass_conservation", "sample_count", "token_count", "status"}:
                continue
            yield from numeric_fields(item, f"{prefix}.{key}" if prefix else key)
    elif isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value):
        yield prefix, value


def prediction_depth(row: dict) -> dict:
    layers = sorted(row["layers"], key=int)
    stable = None
    for layer in reversed(layers):
        if row["layers"][layer]["metrics"]["top1_matches_final"] != 1.0:
            break
        stable = int(layer)
    final = row["layers"][layers[-1]]["final_top1"]
    return {"sample_id": row["sample_id"], "task_name": row["task_name"],
            "stable_from_selected_layer": stable, "final_top1_token_id": final["token_id"],
            "final_top1_token": final["token"]}


def at(value: dict, *path: str) -> float:
    for key in path:
        value = value.get(key, {}) if isinstance(value, dict) else None
    return float(value) if isinstance(value, (float, int)) else math.nan


def save_figure(fig: Any, target: Path) -> None:
    import matplotlib.pyplot as plt
    target.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(target.with_suffix(".png"), dpi=160, bbox_inches="tight")
    fig.savefig(target.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def plot_main(probe: str, data: dict, target: Path, title: str) -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    if probe == "representation_dynamics":
        layers = sorted(data["layers"], key=int)
        comparisons = ("attn_res", "attn_pure", "ffn_res", "ffn_pure", "full_layer_res")
        groups = ("system", "text", "audio")
        fig, axes = plt.subplots(1, 5, figsize=(15, 7), constrained_layout=True)
        for ax, comparison in zip(axes, comparisons):
            values = [[1 - at(data["layers"][layer], comparison, group, "cosine")
                       for group in groups] for layer in layers]
            img = ax.imshow(values, aspect="auto", vmin=0, vmax=2, cmap="viridis")
            ax.set(title=comparison, xticks=range(3), xticklabels=groups,
                   yticks=range(len(layers)), yticklabels=layers, ylabel="LLM layer")
        fig.colorbar(img, ax=axes, label="1 - cosine (pure: update alignment)", shrink=0.6)
    elif probe == "prediction_dynamics":
        layers = sorted(data["layers"], key=int)
        fig, axes = plt.subplots(1, 3, figsize=(13, 3.8), constrained_layout=True)
        for ax, metric in zip(axes, ("js", "rank_of_final_top1", "top1_matches_final")):
            ax.plot(list(map(int, layers)), [at(data["layers"][layer], metric) for layer in layers], marker=".")
            ax.set(xlabel="LLM layer", ylabel=metric)
            if metric == "rank_of_final_top1":
                ax.set_yscale("log")
            ax.grid(alpha=0.2)
    elif probe == "attention_flow":
        layers = sorted(data["layers"], key=int)
        heads = sorted(data["layers"][layers[0]]["heads"], key=int)
        metrics = ("text_to_audio_mass", "decision_to_audio_mass", "audio_to_audio_mass",
                   "audio_to_text_mass", "text_to_audio_entropy_norm", "decision_to_audio_entropy_norm")
        fig, axes = plt.subplots(2, 3, figsize=(15, 10), constrained_layout=True)
        for ax, metric in zip(axes.flat, metrics):
            values = [[at(data["layers"][layer], "heads", head, metric) for head in heads] for layer in layers]
            img = ax.imshow(values, aspect="auto", vmin=0, vmax=1, cmap="viridis")
            ax.set(title=metric, xlabel="Query head", ylabel="LLM layer",
                   yticks=range(len(layers)), yticklabels=layers)
            ax.set_xticks(range(0, len(heads), 2), heads[::2])
        fig.colorbar(img, ax=axes.ravel().tolist(), shrink=0.7)
    else:
        layers = sorted(data["residual_stats"], key=int)
        causal_layers = sorted(data["causal"], key=int)
        fig, axes = plt.subplots(2, 3, figsize=(15, 8), constrained_layout=True)
        for row, operation in enumerate(("attn", "ffn")):
            for group in ("audio", "text", "system"):
                axes[row, 0].plot(list(map(int, layers)), [at(data["residual_stats"][layer], operation, group, "residual_ratio")
                                                       for layer in layers], label=group)
            axes[row, 0].set(title=f"{operation}: relative write size", xlabel="LLM layer", ylabel="update / input norm")
            axes[row, 0].legend()
            effect = [at(data["causal"][layer], f"remove_{operation}_update", "decision", "js") for layer in causal_layers]
            axes[row, 1].plot(list(map(int, causal_layers)), effect, marker=".")
            axes[row, 1].set(title=f"{operation}: deletion effect", xlabel="LLM layer", ylabel="Decision JS")
            paired = [layer for layer in causal_layers if layer in data["residual_stats"]]
            for group in ("audio", "text"):
                norms = [at(data["residual_stats"][layer], operation, group, "update_norm") for layer in paired]
                shifts = [at(data["causal"][layer], f"remove_{operation}_update", "decision", "js") for layer in paired]
                axes[row, 2].scatter(norms, shifts, label=group, s=22, alpha=0.75)
                for i in range(0, len(paired), max(1, len(paired) // 8)):
                    if np.isfinite(norms[i]) and np.isfinite(shifts[i]):
                        axes[row, 2].annotate(f"L{paired[i]}", (norms[i], shifts[i]), fontsize=7)
            axes[row, 2].set(title="Write size vs all-token deletion", xlabel="Mean update norm", ylabel="Decision JS")
            axes[row, 2].legend()
    fig.suptitle(title, fontsize=11)
    save_figure(fig, target)


def comparison_series(probe: str, data: dict) -> tuple[list[int], list[tuple[str, list[float]]]]:
    root = data["causal"] if probe == "residual_stream_decomposition" else data["layers"]
    keys = sorted(root, key=int)
    if probe == "representation_dynamics":
        series = [(group, [1 - at(root[layer], "full_layer_res", group, "cosine") for layer in keys])
                  for group in ("audio", "text")]
    elif probe == "prediction_dynamics":
        series = [(metric, [at(root[layer], metric) for layer in keys])
                  for metric in ("js", "top1_matches_final")]
    elif probe == "attention_flow":
        series = [(metric, [at(root[layer], "mean", metric) for layer in keys])
                  for metric in ("text_to_audio_mass", "decision_to_audio_mass")]
    else:
        series = [(operation, [at(root[layer], f"remove_{operation}_update", "decision", "js") for layer in keys])
                  for operation in ("attn", "ffn")]
    return list(map(int, keys)), series


def plot_groups(probe: str, groups: list[tuple[str, dict]], target: Path, title: str) -> None:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)
    for label, data in groups:
        layers, series = comparison_series(probe, data)
        for ax, (metric, values) in zip(axes, series):
            ax.plot(layers, values, marker=".", label=label)
            ax.set(xlabel="LLM layer", ylabel=metric)
            ax.grid(alpha=0.2)
    axes[-1].legend(fontsize=7)
    fig.suptitle(title, fontsize=10)
    save_figure(fig, target)


def build_report(run_dir: Path, *, plots: bool = True) -> dict:
    run_dir = Path(run_dir).resolve()
    plan = json.loads((run_dir / "plan.json").read_text())
    config = yaml.safe_load((run_dir / "analysis.resolved.yaml").read_text())
    metadata, coverage = read_metadata(run_dir, plan)
    medians = {task: statistics.median(row["duration_seconds"] for row in metadata.values()
                                      if row["task_name"] == task) for task in plan["leaf_tasks"]}
    report_dir = run_dir / "report"
    report_dir.mkdir(exist_ok=True)
    group_ids: dict[tuple[str, str, str], list[str]] = {}
    for identity, row in metadata.items():
        for key in groups_for(row, medians):
            group_ids.setdefault(key, []).append(identity)
    group_index = {}
    for index, key in enumerate(sorted(group_ids)):
        scores = [metadata[identity]["accuracy"] for identity in group_ids[key]
                  if metadata[identity]["accuracy"] is not None]
        group_index[key] = {
            "id": f"group-{index:04d}", "task_name": key[0], "dimension": key[1], "value": key[2],
            "sample_count": len(group_ids[key]), "labeled_count": len(scores),
            "accuracy": statistics.mean(scores) if scores else None,
        }
    atomic_write_json(report_dir / "groups.json", list(group_index.values()))
    if plots:
        import matplotlib
        matplotlib.use("Agg")
    first_tokens = {task: Counter() for task in plan["leaf_tasks"]}
    with (report_dir / "group_metrics.csv").open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["probe", "task", "dimension", "group", "metric_path", "value"])
        for spec in config["analysis"]["probes"]:
            probe = spec["name"]
            accumulators, seen = {}, set()
            depths = []
            for rank in range(plan["shard_count"]):
                raw = run_dir / f"shard-{rank:03d}" / "analysis" / probe / f"{probe}.raw.json"
                for row in iter_raw_samples(raw):
                    identity = row["sample_id"]
                    if identity not in metadata or identity in seen:
                        raise ValueError(f"Unknown or duplicated {probe} row: {identity}")
                    verify_sample(row, spec, plan["query_heads"])
                    seen.add(identity)
                    for key in groups_for(metadata[identity], medians):
                        if key not in accumulators:
                            accumulators[key] = make_accumulator(spec)
                        accumulators[key].add(row)
                    if probe == "prediction_dynamics":
                        depth = prediction_depth(row)
                        depths.append(depth)
                        first_tokens[row["task_name"]][depth["final_top1_token"]] += 1
            if seen != set(metadata):
                raise ValueError(f"{probe} is missing {len(set(metadata) - seen)} selected samples")
            for task in coverage:
                coverage[task].setdefault("probe_rows", {})[probe] = sum(
                    metadata[identity]["task_name"] == task for identity in seen)
            if depths:
                with (report_dir / "prediction_depths.csv").open("w", newline="", encoding="utf-8") as handle:
                    depth_writer = csv.DictWriter(handle, fieldnames=list(depths[0]))
                    depth_writer.writeheader()
                    depth_writer.writerows(depths)
            small_groups = {}
            for key in sorted(accumulators):
                data = accumulators[key].result()
                descriptor = group_index[key]
                target = report_dir / "aggregates" / probe / f"{descriptor['id']}.json"
                target.parent.mkdir(parents=True, exist_ok=True)
                atomic_write_json(target, {"group": descriptor, "payload": data})
                for metric, value in numeric_fields(data):
                    writer.writerow([probe, *key, metric, value])
                if plots and key[1] == "all":
                    plot_main(probe, data, report_dir / "figures" / key[0] / probe,
                              f"{key[0]} | {probe} | n={descriptor['sample_count']}")
                if key[1] != "all" and key[1] != "source_task":
                    small_groups.setdefault((key[0], key[1]), []).append((key[2], data))
            if plots:
                for (task, dimension), values in small_groups.items():
                    # High-cardinality subcategories remain in complete JSON/CSV tables.
                    if 1 < len(values) <= 10:
                        plot_groups(probe, values, report_dir / "figures" / task / f"{probe}__{dimension}",
                                    f"{task} | {probe} | grouped by {dimension}")
            print(f"Reported {probe}: {len(seen)} samples, {len(accumulators)} groups", flush=True)
    summary = {
        "coverage_passed": True, "tasks": coverage,
        "length_median_seconds_by_task": medians,
        "length_rule": "short < per-task median; long >= median; no answer labels used",
        "first_output_token_counts": {task: dict(counter.most_common()) for task, counter in first_tokens.items()},
        "input_protocol": plan["input_protocol"], "prediction_position": plan["prediction_position"],
        "causal_scope": plan["causal_scope"],
    }
    atomic_write_json(report_dir / "summary.json", summary)
    lines = ["# 四项 Thinker 分析报告", "", "覆盖检查通过；下表区分全库运行与显式 sample-limit 运行。", "",
             "| Task | 数据集行数 | 完成行数 | 有标签行数 | 原始模型准确率 | 全库 |",
             "| --- | ---: | ---: | ---: | ---: | --- |"]
    for task, item in coverage.items():
        accuracy = "不可用" if item["accuracy"] is None else f"{item['accuracy']:.4%}"
        lines.append(f"| {task} | {item['dataset_rows']} | {item['completed_rows']} | {item['labeled_rows']} | {accuracy} | {item['full_dataset']} |")
    lines.extend(["", "- 当前实验任务为 mmsufinal；无标签记录不计入准确率分母。",
                  "- 分组：正确性、音频类型（原数据存在时）、类别、子类别、难度、来源任务和音频长短。",
                  "- 长短以各 task 所选样本的音频时长中位数划分；阈值在 summary.json。",
                  "- prediction 是首输出 Token 的 logit lens；保存首 Token 频数以检查其是否为答案字母。",
                  "- prediction_depths.csv 的稳定层只针对已选择层；完整全层配置才覆盖全部后续层。",
                  "- residual cosine_to_input 是更新前后 residual 的 cosine。删除覆盖全部非 padding Token。",
                  "- 删除的 JS/top-1 变化不是准确率变化；此报告准确率来自同一对话输入的一次原始生成。",
                  "- CSV/JSON 保留全部分组。图为诊断均值，无置信区间；高于十类的分组保留在表格。",
                  "- 本版本不自动加入错配、加噪、时间打乱或纯文本对照。", ""])
    for task, counter in first_tokens.items():
        if counter:
            lines.extend([f"## {task} 的最终首输出 Token", "", "```json",
                          json.dumps(dict(counter.most_common(15)), ensure_ascii=False, indent=2), "```", ""])
    (report_dir / "README.zh.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"Report: {report_dir / 'README.zh.md'}", flush=True)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()
    build_report(Path(args.run_dir), plots=not args.no_plots)


if __name__ == "__main__":
    main()
