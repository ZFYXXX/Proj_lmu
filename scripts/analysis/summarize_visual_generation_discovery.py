"""Audit image-disjoint generation diagnostics and report paired task outcomes."""

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import random
import statistics

from mllm_comp.analysis.common.discovery_sampling import load_excluded_images
from mllm_comp.analysis.common.visual_token_phenomena import spearman_correlation


def interval(values):
    rng = random.Random(42)
    draws = sorted(statistics.mean(rng.choices(values, k=len(values))) for _ in range(2000))
    return {"mean": statistics.mean(values), "ci95_low": draws[49], "ci95_high": draws[1949], "images": len(values)}


def summarize(run_dir):
    config = json.loads((run_dir / "effective_config.json").read_text())
    probe = next(v for v in config["analysis"]["probes"] if v["name"] == "visual_token_phenomena")
    source = run_dir / "analysis/visual_token_phenomena/visual_token_phenomena.samples.jsonl"
    raw = source.read_bytes()
    records = [json.loads(line) for line in raw.splitlines() if line.strip()]
    expected = config["analysis"]["sampling"]["unique_images_per_task"]
    excluded, _ = load_excluded_images(config["analysis"]["sampling"].get("exclude_samples", []))
    if len(records) != expected * len(config["analysis"]["tasks"]) or any(v["status"] != "ok" for v in records):
        raise ValueError("Generation cohort is incomplete or contains sample failures.")
    if any(v["image_fingerprint"] in excluded for v in records):
        raise ValueError("Discovery images leaked into the image-disjoint cohort.")
    tasks = {}
    for task in config["analysis"]["tasks"]:
        rows = [v for v in records if v["task_name"] == task]
        if len(rows) != expected or len({v["image_fingerprint"] for v in rows}) != expected:
            raise ValueError("Expected one question per independent image in each task.")
        table = defaultdict(list)
        correlations = defaultdict(list)
        examples = []
        strata = {"baseline_full_score": 0, "baseline_partial_score": 0, "baseline_zero_score": 0}
        truncations = 0
        for row in rows:
            outputs = row["free_generation"]
            if len(outputs) != 1 + 6 * len(probe["joint_deletion_fractions"]):
                raise ValueError("Missing free-generation conditions.")
            baseline = next(v for v in outputs if v["ranking"] == "baseline")
            if not baseline.get("native_generation_identity_passed"):
                raise ValueError("Dense custom decoding must exactly match native model.generate.")
            score = baseline["primary_task_score"]
            strata["baseline_full_score" if score == 1 else "baseline_zero_score" if score == 0 else "baseline_partial_score"] += 1
            budget_by_fraction = {}
            grouped = defaultdict(list)
            loss = []
            metrics = defaultdict(list)
            for value in outputs:
                if value["gold_answer_in_decode_context"] or value["use_cache"]:
                    raise ValueError("Unexpected decoding protocol or gold-context leak.")
                if value["first_appended_token_position"] != baseline["first_appended_token_position"]:
                    raise ValueError("Text decode positions changed after visual deletion.")
                if value["selection_uses_answer"] != value["ranking"].startswith("singleton_oracle"):
                    raise ValueError("Incorrect oracle versus prompt-only selection provenance.")
                truncations += not value["stopped_on_eos"]
                key = (value["ranking"], value["fraction_units"])
                grouped[key].append(value["primary_task_score"])
                if value["ranking"] == "baseline":
                    continue
                previous = budget_by_fraction.setdefault(value["fraction_units"], value["retained_visual_tokens"])
                if value["retained_visual_tokens"] != previous:
                    raise ValueError("Compared generation conditions have different actual budgets.")
                loss.append(score - value["primary_task_score"])
                metrics["abs_answer_nll_delta"].append(abs(value["answer_nll_delta"]))
                metrics["positive_answer_nll_delta"].append(max(0, value["answer_nll_delta"]))
                metrics["first_answer_token_kl"].append(value["first_answer_token_kl"])
                metrics["teacher_to_candidate_answer_kl"].append(value["teacher_to_candidate_answer_kl"])
                if value["answer_nll_delta"] <= 0 and score > value["primary_task_score"]:
                    examples.append({"sample_id": row["sample_id"], "ranking": value["ranking"], "fraction_units": value["fraction_units"],
                                     "baseline_prediction": baseline["prediction"], "candidate_prediction": value["prediction"],
                                     "score_delta": value["primary_task_score_delta"], "answer_nll_delta": value["answer_nll_delta"],
                                     "teacher_kl": value["teacher_to_candidate_answer_kl"]})
            for key, values in grouped.items():
                table[key].append((statistics.mean(values), score))
            for name, values in metrics.items():
                correlation = spearman_correlation(values, loss)
                if correlation is not None:
                    correlations[name].append(correlation)
        tasks[task] = {
            "independent_images": expected, "baseline_score_strata": strata,
            "conditions": [{"ranking": key[0], "fraction_units": key[1],
                            "mean_task_score": statistics.mean(value[0] for value in values),
                            "paired_score_delta": interval([value[0] - value[1] for value in values])}
                           for key, values in sorted(table.items())],
            "within_image_diagnostic_correlation": {name: interval(values) for name, values in correlations.items()},
            "nll_nonincrease_but_generated_score_decrease": examples,
            "generation_conditions_ending_at_length_cap": truncations,
        }
    return {"run_dir": str(run_dir.resolve()), "source_sha256": hashlib.sha256(raw).hexdigest(),
            "images_by_task": expected, "questions": len(records),
            "unique_images_across_tasks": len({r["image_fingerprint"] for r in records}),
            "excluded_discovery_images": len(excluded), "tasks": tasks,
            "limitations": ["Small follow-up cohort; all intervals exploratory.",
                            "Gold-answer NLL and singleton oracle are diagnostics, not deployable selection inputs.",
                            "All conditions use greedy cache-free decoding with repetition_penalty=1 and a fixed length cap.",
                            "Cached visual embeddings and full decoder suffix recomputation are not an end-to-end speed benchmark.",
                            "Undefined within-image correlations are excluded and their eligible image counts reported."]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run_dir", type=Path, required=True)
    args = parser.parse_args()
    report = args.run_dir / "generation_discovery_report.md"
    output = args.run_dir / "generation_discovery_report.json"
    if output.exists() or report.exists():
        raise FileExistsError("Do not overwrite earlier analysis artifacts.")
    payload = summarize(args.run_dir)
    lines = ["# 独立图像上的诊断与真实生成", "",
             f"完成 {payload['questions']} 个问题；排除 {payload['excluded_discovery_images']} 张发现阶段图像。各任务每图一个问题。跨任务共有 {payload['unique_images_across_tasks']} 张不同图像。", "",
             "采样由固定候选池内的图像/问题哈希决定，不读取答案表现。所有 dense 生成都逐 Token 验证与原生 generate 一致。Oracle 使用正确答案构造排序，其他选择只读取图像和问题。此表验证测量与现象，不是新方法结果。", "",
             "| 任务 | 排名 | 删除区域比例 | 平均官方任务得分 % | 相对 baseline 得分变化 pp，图像配对95% CI |", "|---|---|---:|---:|---:|"]
    for task, values in payload["tasks"].items():
        for row in values["conditions"]:
            ci = row["paired_score_delta"]
            lines.append(f"| {task} | {row['ranking']} | {row['fraction_units']:.0%} | {row['mean_task_score']*100:.2f} | {ci['mean']*100:+.2f} [{ci['ci95_low']*100:+.2f}, {ci['ci95_high']*100:+.2f}] |")
    lines += ["", "随机三个种子先按图像平均。删除比例指64个干预区域，实际视觉 Token 预算在同一比例各组间完全相同；它不代表保留 K64。TextVQA 得分使用全部人类参考的原生软评分。", "", "## 指标与生成不一致的例子", ""]
    for task, values in payload["tasks"].items():
        examples = values["nll_nonincrease_but_generated_score_decrease"]
        lines.append(f"- {task}: NLL 未上升但生成得分下降的条件记录 {len(examples)} 条；来自 {len({v['sample_id'] for v in examples})} 张图。条件之间不独立，不能把条数作为独立样本。")
    lines += ["", "## 边界", ""] + [f"- {text}" for text in payload["limitations"]]
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    report.write_text("\n".join(lines) + "\n")
    print(json.dumps({"questions": payload["questions"], "report": str(report)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
