"""Merge non-overlapping four-GPU queue records without recomputing metrics."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import socket


def atomic_json(path: Path, value) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--workers", type=int, required=True)
    args = parser.parse_args()
    root = args.run_dir.resolve()
    workers, tasks, methods = [], [], None
    raw_scores, normalized = {}, {}
    seen_tasks = set()
    for index in range(args.workers):
        folder = root / f"worker_{index}"
        status_path, plan_path, summary_path = folder / "status.json", folder / "plan.json", folder / "summary.json"
        status = json.loads(status_path.read_text()) if status_path.is_file() else {"status": "missing"}
        record = {"worker": index, "path": str(folder), "status": status}
        if plan_path.is_file():
            plan = json.loads(plan_path.read_text())
            record.update(gpu=index, tasks=plan["tasks"], methods=plan["methods"])
            overlap = seen_tasks.intersection(plan["tasks"])
            if overlap:
                raise ValueError(f"Tasks assigned to multiple workers: {sorted(overlap)}")
            seen_tasks.update(plan["tasks"])
            tasks.extend(plan["tasks"])
            if methods is None:
                methods = plan["methods"]
            elif methods != plan["methods"]:
                raise ValueError("Workers use different method orders")
        if summary_path.is_file():
            summary = json.loads(summary_path.read_text())
            for method, values in summary["raw_scores"].items():
                raw_scores.setdefault(method, {}).update(values)
            for method, values in summary["normalized_percent"].items():
                normalized.setdefault(method, {}).update(values)
        workers.append(record)
    complete = all(item["status"].get("status") == "complete" for item in workers)
    combined = {"status": "complete" if complete else "incomplete", "worker_count": args.workers,
                "methods": methods or [], "tasks": tasks, "workers": workers,
                "raw_scores": raw_scores, "normalized_percent": normalized,
                "slurm": {"job_id": os.environ.get("SLURM_JOB_ID"),
                           "job_name": os.environ.get("SLURM_JOB_NAME"),
                           "node_list": os.environ.get("SLURM_JOB_NODELIST"),
                           "hostname": socket.gethostname(),
                           "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES")},
                "gpu_usage_log": str(root / "logs/gpu_usage.log")}
    atomic_json(root / "parallel_summary.json", combined)
    atomic_json(root / "parallel_status.json", {key: combined[key] for key in ("status", "worker_count", "workers")})

    lines = ["# Qwen2.5 four-GPU evaluation", "",
             f"Overall status: **{combined['status']}**. Each GPU owns disjoint datasets; methods are serial within a GPU.", "",
             "| GPU | Tasks | Status |", "|---:|---|---|"]
    for item in workers:
        lines.append(f"| {item['worker']} | {', '.join(item.get('tasks', [])) or '—'} | {item['status'].get('status', 'missing')} |")
    if methods and tasks:
        lines += ["", "Normalized score = method score / same-task dense score × 100. HallusionBench remains pending judge.", "",
                  "| Method | " + " | ".join(tasks) + " |", "|---|" + "---:|" * len(tasks)]
        for method in methods:
            values = []
            for task in tasks:
                value = normalized.get(method, {}).get(task)
                values.append(f"{value:.2f}%" if value is not None else "—")
            lines.append("| " + method + " | " + " | ".join(values) + " |")
    (root / "parallel_report.md").write_text("\n".join(lines) + "\n")
    print(f"Wrote {root / 'parallel_report.md'} ({combined['status']})")


if __name__ == "__main__":
    main()
