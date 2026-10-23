"""Run all four silent-redundancy analysis stages with resumable condition directories."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

from . import CAUSAL_SELECTORS, SELECTORS, TARGET_RATIOS


PER_STAGE_KEEP = {0.75: 0.915, 0.50: 0.80, 0.25: 0.64}


def _complete(path: Path) -> bool:
    coverage = path / "coverage.json"
    if not coverage.exists():
        return False
    return bool(json.loads(coverage.read_text()).get("complete"))


def _run(command: list[str], *, environment: dict[str, str]):
    print("RUN", " ".join(command), flush=True)
    subprocess.run(command, check=True, env=environment)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default=os.environ.get("CHECKPOINT_PATH"))
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--num-processes", type=int, required=True)
    parser.add_argument("--port-base", type=int, default=29720)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--example-doc-ids", default="0,833,1666,2499,3332,4165")
    parser.add_argument("--phase", choices=("all", "observe", "compare", "causal", "report"),
                        default="all")
    parser.add_argument("--mmsu-jsonl", default=os.environ.get("MMSU_JSONL"))
    args = parser.parse_args()
    if not args.checkpoint:
        parser.error("Set CHECKPOINT_PATH or pass --checkpoint")
    checkpoint = str(Path(args.checkpoint).expanduser().resolve())
    run_dir = Path(args.run_dir).expanduser().resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    environment = dict(os.environ)
    environment["CHECKPOINT_PATH"] = checkpoint
    if args.mmsu_jsonl:
        environment["MMSU_JSONL"] = str(Path(args.mmsu_jsonl).expanduser().resolve())
    common = [
        "--checkpoint", checkpoint,
        "--num-processes", str(args.num_processes),
        "--example-doc-ids", args.example_doc_ids,
    ]
    if args.limit is not None:
        common += ["--limit", str(args.limit)]
    plan_path = run_dir / "suite_plan.json"
    previous = json.loads(plan_path.read_text()) if plan_path.exists() else {}
    if previous and previous.get("checkpoint") != checkpoint:
        raise ValueError(
            f"RUN_DIR was created for checkpoint {previous.get('checkpoint')}, not {checkpoint}"
        )
    if previous and previous.get("limit") != args.limit:
        raise ValueError(
            f"RUN_DIR was created with limit={previous.get('limit')}, not limit={args.limit}"
        )
    plan = {
        "checkpoint": checkpoint,
        "num_processes": args.num_processes,
        "limit": args.limit,
        "observation": "observe",
        "comparison": previous.get("comparison", []),
        "causal": previous.get("causal", []),
        "generation_calls_per_sample": 1 + len(SELECTORS) * len(TARGET_RATIOS) + len(CAUSAL_SELECTORS),
    }
    index = 0
    observe = run_dir / "observe"
    if args.phase in ("all", "observe"):
        if not _complete(observe):
            _run([
                sys.executable, "-m", "scripts.analysis.silent_redundancy_mmsu.launch", "observe",
                "--output-dir", str(observe),
                "--main-process-port", str(args.port_base + index),
                *common,
            ], environment=environment)
        else:
            print(f"SKIP complete {observe}", flush=True)
        index += 1
        _run([
            sys.executable, "-m", "scripts.analysis.silent_redundancy_mmsu.report",
            "--run-dir", str(run_dir), "--observational-only",
        ], environment=environment)

    conditions_root = run_dir / "conditions"
    conditions_root.mkdir(exist_ok=True)
    if args.phase in ("all", "compare"):
        plan["comparison"] = []
        for selector in SELECTORS:
            for target in TARGET_RATIOS:
                name = f"{selector}_keep{int(target * 100)}"
                destination = conditions_root / name
                plan["comparison"].append({
                    "name": name,
                    "selector": selector,
                    "target_ratio": target,
                    "keep_ratio_per_stage": PER_STAGE_KEEP[target],
                    "layers": [3, 6, 9],
                })
                if not _complete(destination):
                    _run([
                        sys.executable, "-m", "scripts.analysis.silent_redundancy_mmsu.launch", "prune",
                        "--output-dir", str(destination),
                        "--main-process-port", str(args.port_base + index),
                        "--selector", selector,
                        "--target-ratio", str(target),
                        "--keep-ratio-per-stage", str(PER_STAGE_KEEP[target]),
                        "--layers", "3,6,9",
                        *common,
                    ], environment=environment)
                else:
                    print(f"SKIP complete {destination}", flush=True)
                index += 1

    if args.phase in ("all", "causal"):
        plan["causal"] = []
        for selector in CAUSAL_SELECTORS:
            name = selector
            destination = conditions_root / name
            plan["causal"].append({
                "name": name,
                "selector": selector,
                "target_ratio": 0.90,
                "keep_ratio_per_stage": 0.90,
                "layers": [9],
            })
            if not _complete(destination):
                _run([
                    sys.executable, "-m", "scripts.analysis.silent_redundancy_mmsu.launch", "prune",
                    "--output-dir", str(destination),
                    "--main-process-port", str(args.port_base + index),
                    "--selector", selector,
                    "--target-ratio", "0.90",
                    "--keep-ratio-per-stage", "0.90",
                    "--layers", "9",
                    *common,
                ], environment=environment)
            else:
                print(f"SKIP complete {destination}", flush=True)
            index += 1

    plan_path.write_text(
        json.dumps(plan, ensure_ascii=False, indent=2), encoding="utf-8")
    if args.phase in ("all", "report"):
        if not _complete(observe):
            raise RuntimeError("Final report requires a complete observation baseline")
        _run([
            sys.executable, "-m", "scripts.analysis.silent_redundancy_mmsu.report",
            "--run-dir", str(run_dir),
        ], environment=environment)


if __name__ == "__main__":
    main()
