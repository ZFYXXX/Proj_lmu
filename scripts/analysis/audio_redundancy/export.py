"""Create a small, analysis-ready copy of a completed run for transfer."""
import argparse
import csv
import gzip
import json
from pathlib import Path
import shutil


BASELINE_FIELDS = (
    "doc_index", "id", "audio_path", "question_type", "sub_category", "task_name",
    "accuracy", "prediction", "encoder_tokens", "projector_tokens", "duration_seconds",
    "held_out_fold",
)
CONDITION_FIELDS = (
    "doc_index", "condition_id", "accuracy", "baseline_accuracy", "prediction",
    "retention", "tokens_per_second", "retained_k", "mse", "matched",
    "budget_clamped_by_hard_chunks", "analysis_preparation_seconds",
    "diagnostic_generation_seconds",
)


def export_rows(paths, destination, fields, transform):
    count = 0
    with gzip.open(destination, "wt", encoding="utf-8", newline="", compresslevel=6) as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for path in sorted(paths):
            with path.open(encoding="utf-8") as source:
                for line in source:
                    if line.strip():
                        writer.writerow(transform(json.loads(line)))
                        count += 1
    return count


def export_portable(directory):
    root = Path(directory)
    settings = json.loads((root / "analysis.json").read_text(encoding="utf-8"))
    report = root / "report"
    summary = json.loads((report / "summary.json").read_text(encoding="utf-8"))
    evaluation = root / "evaluation"
    expected = settings["expected_samples"]
    if summary["questions"] != expected or len(summary["conditions"]) != len(settings["conditions"]):
        raise ValueError("Portable export requires a complete report")

    portable = root / "portable"
    portable.mkdir(exist_ok=True)
    figures = portable / "figures"
    figures.mkdir(exist_ok=True)
    for name in ("summary.json", "matched_budget_comparisons.csv", "README.zh.md"):
        source = report / name
        if source.is_file():
            shutil.copy2(source, portable / name)
    plan = {key: value for key, value in settings.items()
            if key not in {"document_ids", "audio_paths", "folds"}}
    (portable / "plan.json").write_text(json.dumps(plan, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    for name in ("paired_task_results.csv", "rank_by_audio.csv", "units_by_audio.csv"):
        source = report / name
        if source.is_file():
            with source.open("rb") as original, gzip.open(portable / f"{name}.gz", "wb", compresslevel=6) as target:
                shutil.copyfileobj(original, target)
    for phase in ("calibration", "evaluation"):
        source = root / phase / "coverage.json"
        if source.is_file():
            shutil.copy2(source, portable / f"{phase}_coverage.json")
        for name in ("recovered_exit.json", "recovered_launcher_exit.json"):
            source = root / phase / name
            if source.is_file():
                shutil.copy2(source, portable / f"{phase}_{name}")
    metadata = root / "models" / "metadata.json"
    if metadata.is_file():
        shutil.copy2(metadata, portable / "model_metadata.json")
    environment = root / "models" / "fit_environment.json"
    if environment.is_file():
        shutil.copy2(environment, portable / "fit_environment.json")

    baseline_count = export_rows(evaluation.glob("samples.rank*.jsonl"), portable / "baseline.csv.gz",
                                 BASELINE_FIELDS, lambda row: {key: row.get(key) for key in BASELINE_FIELDS})

    def condition_values(row):
        details = row["intervention"]
        return {key: row.get(key) if key in row else details.get(key) for key in CONDITION_FIELDS}

    condition_count = export_rows(evaluation.glob("conditions.rank*.jsonl"),
                                  portable / "condition_scores.csv.gz", CONDITION_FIELDS, condition_values)
    if baseline_count != expected or condition_count != expected * len(settings["conditions"]):
        raise ValueError("Portable export row count differs from the complete evaluation")

    example_ids = settings.get("example_doc_ids", [])
    selected_example = next((index for index in example_ids
                             if any(report.glob(f"*doc{index}.png"))), None)
    for source in report.glob("*.png"):
        if "_doc" not in source.stem or source.stem.endswith(f"_doc{selected_example}"):
            shutil.copy2(source, figures / source.name)

    instructions = (
        "# 可迁移分析包\n\n"
        "本目录可单独从远端复制。`summary.json` 和 `README.zh.md` 是条件汇总；"
        "`baseline.csv.gz` 与 `condition_scores.csv.gz` 是逐题配对数值和预测文本；"
        "其余 CSV.gz 是分题型、逐音频有效秩及离散状态统计。`figures/` 只保留 PNG 总图"
        "和一个示例音频。准确率与保留率在逐题表中均为 0–1 比例。\n\n"
        "这里不包含校准 NPZ、码本权重、原生 lmms-eval 副本和全部示例数组；"
        "这些仍保留在远端原始运行目录，若需要重拟合或重绘任意示例须使用原目录。\n"
    )
    (portable / "PORTABLE_README.zh.md").write_text(instructions, encoding="utf-8")
    files = sorted(path for path in portable.rglob("*") if path.is_file() and path.name != "manifest.json")
    manifest = dict(experiments=summary["experiments"], questions=expected,
                    conditions=len(settings["conditions"]), baseline_rows=baseline_count,
                    condition_rows=condition_count, example_doc_id=selected_example,
                    total_bytes=sum(path.stat().st_size for path in files),
                    files=[str(path.relative_to(portable)) for path in files])
    (portable / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
                                              encoding="utf-8")
    print(f"Portable analysis export: {portable} ({manifest['total_bytes'] / 1048576:.1f} MiB)", flush=True)
    return portable


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    export_portable(parser.parse_args().run_dir)
