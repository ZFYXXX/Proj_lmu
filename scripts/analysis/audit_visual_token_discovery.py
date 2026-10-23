"""Reassess existing phenomenon records without model execution or training.

The audit preserves source records and distinguishes logged completion from
scientific validity. Confidence intervals resample images, not questions or
dependent question pairs. Small-effect thresholds are descriptive scales and
must not be mistaken for a measured numerical noise floor.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
from itertools import combinations
import json
import math
from pathlib import Path
import random
import statistics
from typing import Any

from mllm_comp.analysis.common.visual_token_phenomena import (
    jaccard,
    spearman_correlation,
    top_fraction_indices,
)


def bootstrap(values: list[float], *, seed: int = 42) -> dict[str, Any] | None:
    if not values:
        return None
    rng = random.Random(seed)
    draws = sorted(
        statistics.mean(rng.choices(values, k=len(values))) for _ in range(2000)
    )
    return {
        "mean": statistics.mean(values),
        "ci95_low": draws[49],
        "ci95_high": draws[1949],
        "independent_image_count": len(values),
    }


def random_set_jaccard(units: int, selected: int) -> float:
    """Exact E[J] for two independent uniform selected-subsets of units."""
    denominator = math.comb(units, selected)
    lower = max(0, 2 * selected - units)
    probabilities = [
        (i, math.comb(selected, i) * math.comb(units - selected, selected - i) / denominator)
        for i in range(lower, selected + 1)
    ]
    assert math.isclose(sum(p for _, p in probabilities), 1.0, abs_tol=1e-12)
    return sum(p * i / (2 * selected - i) for i, p in probabilities)


def effects(record: dict[str, Any], kind: str) -> list[float]:
    return [float(row["answer_nll_delta"]) for row in record["interventions"][kind]]


def mean_by_image(records: list[dict[str, Any]], metric: Any) -> list[float]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for record in records:
        value = metric(record)
        if value is not None and math.isfinite(value):
            grouped[record["image_fingerprint"]].append(value)
    return [statistics.mean(grouped[key]) for key in sorted(grouped)]


def audit_task(records: list[dict[str, Any]]) -> dict[str, Any]:
    task = records[0]["task_name"]
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        groups[record["image_fingerprint"]].append(record)
    units = {int(record["target_units"]) for record in records}
    if len(units) != 1:
        raise ValueError(f"Mixed intervention granularities for {task}.")
    unit_count = next(iter(units))
    selected_count = len(top_fraction_indices(list(range(unit_count)), 0.25))
    null = random_set_jaccard(unit_count, selected_count)
    targets = Counter(str(record["target_text"]) for record in records)
    invalid_textvqa_targets = sum(
        task.startswith("textvqa") and record["target_text"] == "answer"
        for record in records
    )
    result: dict[str, Any] = {
        "records": len(records),
        "unique_images": len(groups),
        "multiquestion_images": sum(len(group) > 1 for group in groups.values()),
        "source_tokens_median": statistics.median(r["source_tokens"] for r in records),
        "invalid_literal_textvqa_targets": invalid_textvqa_targets,
        "answer_target_validity": "invalid" if invalid_textvqa_targets else "content_mask_audit_required",
        "top_target_frequencies": targets.most_common(5),
        "noncontiguous_supervised_positions": sum(
            any(b != a + 1 for a, b in zip(
                record["objective"]["answer_token_positions_by_sample"][0],
                record["objective"]["answer_token_positions_by_sample"][0][1:],
            )) for record in records
        ),
        "uniform_random_top_quartile_expected_jaccard": null,
        "query_overlap": {},
        "score_vs_random": {},
        "intervention_effects": {},
    }
    kinds = sorted(records[0]["interventions"])
    for kind in kinds:
        image_means = []
        image_ranks = []
        pair_count = 0
        duplicate_prompt_pairs = 0
        partition_mismatch_pairs = 0
        for fingerprint in sorted(groups):
            overlaps = []
            correlations = []
            for left, right in combinations(groups[fingerprint], 2):
                if left["prompt"] == right["prompt"]:
                    duplicate_prompt_pairs += 1
                    continue
                if left["partition_hash"] != right["partition_hash"]:
                    partition_mismatch_pairs += 1
                    continue
                x, y = effects(left, kind), effects(right, kind)
                overlaps.append(jaccard(top_fraction_indices(x, 0.25), top_fraction_indices(y, 0.25)))
                rho = spearman_correlation(x, y)
                if rho is not None:
                    correlations.append(rho)
            if overlaps:
                image_means.append(statistics.mean(overlaps))
                pair_count += len(overlaps)
            if correlations:
                image_ranks.append(statistics.mean(correlations))
        result["query_overlap"][kind] = {
            "image_macro_jaccard": bootstrap(image_means),
            "image_macro_spearman": bootstrap(image_ranks),
            "question_pairs_descriptive_only": pair_count,
            "duplicate_prompt_pairs_excluded": duplicate_prompt_pairs,
            "partition_mismatch_pairs_excluded": partition_mismatch_pairs,
            "jaccard_above_uniform_null": bootstrap([v - null for v in image_means]),
            "interpretation": "Requires same-query reliability and answer-context controls; low overlap alone does not establish query dependence.",
        }
        values = [value for record in records for value in effects(record, kind)]
        if any(not math.isfinite(value) for value in values):
            raise ValueError(f"Non-finite NLL effect in {task}/{kind}.")
        result["intervention_effects"][kind] = {
            "unit_interventions": len(values),
            "mean_delta": statistics.mean(values),
            "median_abs_delta": statistics.median(abs(v) for v in values),
            "negative_delta_fraction": statistics.mean(v < 0 for v in values),
            "abs_delta_at_most": {
                str(threshold): statistics.mean(abs(v) <= threshold for v in values)
                for threshold in (0.0001, 0.001, 0.01)
            },
        }
        scores = records[0]["score_causal_spearman"][kind]
        for score in sorted(name for name in scores if not name.startswith("random_seed_")):
            def difference(record: dict[str, Any], score: str = score) -> float | None:
                row = record["score_causal_spearman"][kind]
                random_values = [v for name, v in row.items() if name.startswith("random_seed_") and v is not None]
                if row.get(score) is None or not random_values:
                    return None
                return float(row[score]) - statistics.mean(random_values)
            result["score_vs_random"][f"{kind}:{score}"] = {
                "image_macro_rho": bootstrap(mean_by_image(records, lambda r: r["score_causal_spearman"][kind].get(score))),
                "paired_rho_minus_three_random_mean": bootstrap(mean_by_image(records, difference)),
            }
    if {"zero", "global_mean"}.issubset(kinds):
        result["cross_intervention_rank_agreement"] = bootstrap(mean_by_image(
            records, lambda r: spearman_correlation(effects(r, "zero"), effects(r, "global_mean"))
        ))
        result["cross_intervention_top_quartile_jaccard"] = bootstrap(mean_by_image(
            records, lambda r: jaccard(top_fraction_indices(effects(r, "zero"), 0.25), top_fraction_indices(effects(r, "global_mean"), 0.25))
        ))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", action="append", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.report.exists():
        parser.error("Audit outputs must be new paths; original artifacts are preserved.")
    rows: list[dict[str, Any]] = []
    sources = []
    for path in args.samples:
        raw = path.read_bytes()
        rows.extend(json.loads(line) for line in raw.splitlines() if line.strip())
        sources.append({"path": str(path.resolve()), "sha256": hashlib.sha256(raw).hexdigest()})
    identities = [(r["task_name"], r["sample_id"]) for r in rows]
    if len(set(identities)) != len(identities):
        raise ValueError("Duplicate task/sample identities in audit inputs.")
    successful = [r for r in rows if r["status"] == "ok"]
    tasks = {
        task: audit_task([r for r in successful if r["task_name"] == task])
        for task in sorted({r["task_name"] for r in successful})
    }
    payload = {
        "artifact_kind": "measurement_audit",
        "sources": sources,
        "record_count": len(rows),
        "logged_successful_records": len(successful),
        "methodological_limits": [
            "answer_flip is teacher-forced top-1 token change, not generated-answer change.",
            "Query overlap requires same-query reliability and answer-context controls before causal interpretation.",
            "Prompt-only and gold-answer-context scores are separate when present; only the former is prompt-available.",
            "Replacement interventions preserve sequence length; physical_delete records are actual suffix reruns with removed tokens.",
            "No repeated identical-condition measurements exist to estimate the noise floor.",
            "Single-unit effects do not measure joint-deletion interactions.",
            "Score confidence intervals are exploratory; maxima across scores are selection-biased.",
        ],
        "tasks": tasks,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    lines = [
        "# 视觉 Token 现象记录：测量审计",
        "",
        "此文件由现有 JSONL 重分析产生，不运行模型。",
        "",
        "执行成功不等于科学解释成立。先核对目标字段和 assistant 正文标签，再把单元干预关联解释为排序信号。",
        "",
        "下表 overlap 先在每张图内平均不同问题对，再按图平均。95% CI 以图像为重采样单位。数值是未经同问题稳定性校准的描述统计。",
        "",
        "| 任务 | 记录/独立图像 | 错误字面目标 | global_mean Jaccard，95% CI | zero Jaccard，95% CI |",
        "|---|---:|---:|---:|---:|",
    ]
    for task, data in tasks.items():
        def formatted(kind: str) -> str:
            stat = data["query_overlap"][kind]["image_macro_jaccard"]
            return "N/A" if stat is None else f"{stat['mean']:.3f} [{stat['ci95_low']:.3f}, {stat['ci95_high']:.3f}]"
        lines.append(f"| {task} | {data['records']}/{data['unique_images']} | {data['invalid_literal_textvqa_targets']} | {formatted('global_mean')} | {formatted('zero')} |")
    null = next(iter(tasks.values()))["uniform_random_top_quartile_expected_jaccard"]
    lines.extend([
        "", f"64 个单元随机选 16 个时，独立随机集合 Jaccard 的精确期望为 **{null:.6f}**。这不是所有空间/面积约束下的随机基线；正式干预还需要面积匹配随机集合。",
        "", "原始实验的 Jaccard 固定阈值 0.4/0.7 尚未经过测量可靠性校准，不再用作 query-aware / image-only 的决策线。",
        "", "## 干预机制之间的稳定性", "",
        "| 任务 | zero 对 global_mean 的图像平均 Spearman，95% CI | Top-25% Jaccard，95% CI |",
        "|---|---:|---:|",
    ])
    for task, data in tasks.items():
        def stat_text(key: str) -> str:
            value = data.get(key)
            return "N/A" if value is None else f"{value['mean']:.3f} [{value['ci95_low']:.3f}, {value['ci95_high']:.3f}]"
        lines.append(f"| {task} | {stat_text('cross_intervention_rank_agreement')} | {stat_text('cross_intervention_top_quartile_jaccard')} |")
    lines.extend(["", "这衡量干预机制的一致性，不等于相同条件重复测量可靠性。TextVQA 行仅用于记录错误输入下的行为，不用于正确答案推断。", "", "## 测量边界", ""])
    lines.extend(f"- {item}" for item in payload["methodological_limits"])
    lines.extend(["", "## 可复现输入", ""])
    lines.extend(f"- `{source['path']}`，SHA256 `{source['sha256']}`。" for source in sources)
    args.report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"records": len(rows), "output": str(args.output), "report": str(args.report), "invalid_textvqa_targets": sum(t["invalid_literal_textvqa_targets"] for t in tasks.values())}, ensure_ascii=False))


if __name__ == "__main__":
    main()
