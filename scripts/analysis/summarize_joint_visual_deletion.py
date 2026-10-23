"""Validate and report a discovery pilot; use images as the averaging unit."""

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import statistics


def summarize(run_dir: Path) -> dict:
    source = run_dir / "analysis/visual_token_phenomena/visual_token_phenomena.samples.jsonl"
    raw = source.read_bytes()
    rows = [json.loads(line) for line in raw.splitlines() if line.strip()]
    config = json.loads((run_dir / "effective_config.json").read_text())
    probe = next(p for p in config["analysis"]["probes"] if p["name"] == "visual_token_phenomena")
    active_tasks = set(config["analysis"]["tasks"])
    expected = {value for value in probe.get("sample_ids", ()) if value.rsplit(":", 1)[0] in active_tasks}
    if any(row["status"] != "ok" for row in rows):
        raise ValueError("Pilot contains failed samples; do not report it as complete.")
    if len({row["sample_id"] for row in rows}) != len(rows):
        raise ValueError("Duplicate sample IDs.")
    if expected and {row["sample_id"] for row in rows} != expected:
        raise ValueError("Completed samples differ from the explicit allowlist.")
    sample_summaries = []
    aggregate = defaultdict(lambda: defaultdict(list))
    for row in rows:
        singletons = row["interventions"]["physical_delete"]
        if len(singletons) != row["target_units"] or not row["uncompressed_suffix_identity"]["passed"]:
            raise ValueError("Missing singleton interventions or failed identity gate.")
        sizes = {value["unit"]: len(value["members"]) for value in singletons}
        effects = {value["unit"]: value["answer_nll_delta"] for value in singletons}
        coverage = [index for value in singletons for index in value["members"]]
        if sorted(coverage) != list(range(row["source_tokens"])):
            raise ValueError("Singleton units do not exactly cover the image tokens.")
        if "textvqa" in row["task_name"] and row["target_text"] not in row.get("target_references", ()):
            raise ValueError("TextVQA target must come from the recorded human references.")
        positions = row["objective"]["answer_token_positions_by_sample"][0]
        if any(b != a + 1 for a, b in zip(positions, positions[1:])):
            raise ValueError("Noncontiguous assistant content labels require investigation.")
        conditions = row["joint_deletion"]
        expected_conditions = len(probe["joint_deletion_fractions"]) * 4 * (1 + len(probe["random_seeds"]))
        if len(conditions) != expected_conditions:
            raise ValueError("Missing joint deletion conditions.")
        by_identity = {(value["ranking"], value["fraction_units"]): value for value in conditions if value["ranking"] != "size_matched_random"}
        low_oracle = "singleton_oracle_abs_low" if any(key[0] == "singleton_oracle_abs_low" for key in by_identity) else "singleton_oracle_low"
        fraction_budgets = {}
        for value in conditions:
            units = value["units"]
            if len(set(units)) != len(units):
                raise ValueError("Duplicate unit in joint deletion.")
            deleted = sum(sizes[index] for index in units)
            if deleted != value["deleted_visual_tokens"] or deleted + value["retained_visual_tokens"] != row["source_tokens"]:
                raise ValueError("Deleted or retained token count mismatch.")
            if "size_quota" in value:
                histogram = Counter(sizes[i] for i in units)
                quota = Counter({int(k): int(v) for k, v in value["size_quota"].items() if v})
                if histogram != quota:
                    raise ValueError("Ranked units violate the common area quota.")
                previous = fraction_budgets.setdefault(value["fraction_units"], histogram)
                if histogram != previous:
                    raise ValueError("Different rankings have different actual token budgets.")
            if not math.isclose(value["sum_singleton_nll_deltas"], sum(effects[index] for index in units), abs_tol=1e-7):
                raise ValueError("Invalid additive singleton reference.")
            if not all(math.isfinite(value[key]) for key in ("answer_nll_delta", "joint_minus_sum_singletons", "first_answer_token_kl")):
                raise ValueError("Nonfinite joint metrics.")
            if value["ranking"] == "size_matched_random":
                target = by_identity[value["matched_to"], value["fraction_units"]]
                if Counter(sizes[i] for i in units) != Counter(sizes[i] for i in target["units"]):
                    raise ValueError("Random control does not exactly match the unit-size histogram.")
            key = (row["task_name"], value["ranking"], value["fraction_units"], value["matched_to"])
            aggregate[key][row["image_fingerprint"]].append(value["answer_nll_delta"])
        baseline_repeats = row["stability"]["baseline_repeats"]
        unit_repeats = row["stability"]["single_unit_repeats"]
        if len(baseline_repeats) != probe["stability_repeats"] or not unit_repeats:
            raise ValueError("Missing repeated-condition measurements.")
        largest = max(singletons, key=lambda value: value["answer_nll_delta"])
        sample_summaries.append({
            "sample_id": row["sample_id"], "task": row["task_name"], "image": row["image_fingerprint"],
            "baseline_content_nll": row["baseline"]["teacher_answer_nll"],
            "baseline_target_token_argmax_exact": row["baseline"]["teacher_answer_exact"],
            "largest_unit": largest["unit"], "largest_delta": largest["answer_nll_delta"],
            "largest_deleted_tokens": largest["deleted_visual_tokens"],
            "baseline_repeat_max_logit_error": max(v["max_abs_logit_error"] for v in baseline_repeats),
            "singleton_repeat_max_logit_error": max(v["max_abs_logit_error_from_repeat1"] for v in unit_repeats),
            "singleton_repeat_max_nll_delta_difference": max(abs(v["nll_delta_difference_from_initial"]) for v in unit_repeats),
            "low_oracle_ranking": low_oracle,
            "oracle_low_curves": [{**value, "matched_random_mean_nll_delta": statistics.mean(v["answer_nll_delta"] for v in conditions if v["ranking"] == "size_matched_random" and v["matched_to"] == low_oracle and v["fraction_units"] == value["fraction_units"])} for value in conditions if value["ranking"] == low_oracle],
            "region_map": row.get("region_map"),
        })
    macro = [
        {"task": key[0], "ranking": key[1], "fraction_units": key[2], "matched_to": key[3],
         "image_macro_nll_delta": statistics.mean(statistics.mean(values) for values in images.values()),
         "independent_images": len(images)}
        for key, images in sorted(aggregate.items(), key=lambda item: str(item[0]))
    ]
    return {"run_dir": str(run_dir.resolve()), "source_sha256": hashlib.sha256(raw).hexdigest(),
            "successful_samples": len(rows), "unique_images": len({(row["task_name"], row["image_fingerprint"]) for row in rows}),
            "selection_note": probe.get("selection_note"), "samples": sample_summaries, "image_macro": macro}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run_dir", type=Path, required=True)
    args = parser.parse_args()
    payload = summarize(args.run_dir)
    output = args.run_dir / "joint_deletion_report.json"
    report = args.run_dir / "joint_deletion_report.md"
    if output.exists() or report.exists():
        raise FileExistsError("Summary outputs already exist; preserve the earlier report.")
    lines = ["# 联合删除与重复性 pilot", "",
             f"完成 {payload['successful_samples']} 个问题，{payload['unique_images']} 张独立图像。", "",
             f"抽样：{payload['selection_note']}", "",
             "按初轮现象选择的探索集，不做显著性结论。答案内容 NLL 和 teacher-forced token 变化不等于真实生成准确率。重复 BF16 一致也不证明 FP32 精度或问题改写稳定性。", "",
             "## 同条件重复与高敏感区域", "",
             "| 样本 | 原始 NLL | 最敏感单元 ΔNLL / Token数 | 重复 baseline 最大 logit差 | 重复单元最大 logit差 |", "|---|---:|---:|---:|---:|"]
    for row in payload["samples"]:
        lines.append(f"| {row['sample_id']} | {row['baseline_content_nll']:.4f} | {row['largest_delta']:.4f} / {row['largest_deleted_tokens']} | {row['baseline_repeat_max_logit_error']:.6g} | {row['singleton_repeat_max_logit_error']:.6g} |")
    lines += ["", "## 单区排序与联合删除", "",
              "`singleton_oracle_abs_low` 按单区 |ΔNLL| 从小到大选择；旧 smoke 的 `singleton_oracle_low` 使用有符号 ΔNLL，二者不能混合解释。该排序读取正确答案损失，是诊断 Oracle。带 size_quota 的正式条件在所有排序及随机对照间统一区域大小配额和真实删除 Token 数。", "",
              "| 样本 | 删除区域比例 | 删除 Token 数 | 单独 ΔNLL 之和 | 真实联合 ΔNLL | 联合减单独之和 | 匹配随机平均 ΔNLL |", "|---|---:|---:|---:|---:|---:|---:|"]
    for row in payload["samples"]:
        for value in row["oracle_low_curves"]:
            lines.append(f"| {row['sample_id']} | {value['fraction_units']:.0%} | {value['deleted_visual_tokens']} | {value['sum_singleton_nll_deltas']:.4f} | {value['answer_nll_delta']:.4f} | {value['joint_minus_sum_singletons']:.4f} | {value['matched_random_mean_nll_delta']:.4f} |")
    lines += ["", "非加性可能来自 softmax、网络非线性、位置/序列变化及表示交互；其符号不能直接命名为语义冗余或互补。", "", "## 区域图", ""]
    for row in payload["samples"]:
        if row["region_map"]:
            path = row["region_map"]["path"]
            lines.append(f"- [{row['sample_id']}]({path})")
    lines += ["", f"源记录 SHA256：`{payload['source_sha256']}`。所有随机种子先在图像内平均；宏平均使用独立图像。"]
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    report.write_text("\n".join(lines) + "\n")
    print(json.dumps({"samples": payload["successful_samples"], "images": payload["unique_images"], "report": str(report)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
