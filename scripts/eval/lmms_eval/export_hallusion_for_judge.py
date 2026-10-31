"""Export completed HallusionBench predictions for later external judging.

This command never calls a judge or sends data to a remote service.
"""
import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--allow-limited", action="store_true", help="Permit a smoke run, never a benchmark result")
    args = parser.parse_args()
    run = args.run_dir.resolve()
    plan = json.loads((run / "plan.json").read_text())
    if "hallusionbench" not in plan["tasks"]:
        parser.error("Run does not contain HallusionBench")
    if plan["limit"] is not None and not args.allow_limited:
        parser.error("Limited smoke results require --allow-limited")
    out = run / "hallusion_judge_input"
    out.mkdir(exist_ok=False)
    for method in plan["methods"]:
        cell = run / "cells" / method / "hallusionbench"
        complete = json.loads((cell / "complete.json").read_text())["score"]
        if complete.get("scoring_status") != "pending_official_judge":
            raise ValueError(f"{method}: no completed deferred generation")
        records, expected_start = [], 0
        for path in sorted(cell.glob("chunk_*/complete.json")):
            part = json.loads(path.read_text())
            chunk = part["chunk"]
            if chunk["start"] != expected_start:
                raise ValueError(f"{method}: missing or overlapping chunk at {expected_start}")
            with Path(part["samples_path"]).open() as stream:
                rows = sorted((json.loads(line) for line in stream if line.strip()), key=lambda row: row["doc_id"])
            for row in rows:
                payload = row.get("hallusion_record")
                if not isinstance(payload, dict) or not payload.get("model_prediction"):
                    raise ValueError(f"{method}: missing judge payload")
                records.append({"source_index": chunk["start"] + row["doc_id"], "record": payload})
            expected_start = chunk["stop"]
        if len(records) != complete["samples"] or expected_start != complete["samples"]:
            raise ValueError(f"{method}: sample count mismatch")
        with (out / f"{method}.jsonl").open("x") as stream:
            for item in records:
                stream.write(json.dumps(item, ensure_ascii=False) + "\n")
    (out / "manifest.json").write_text(json.dumps({"status": "prepared_not_scored", "source_run": str(run),
        "methods": plan["methods"], "limited": plan["limit"] is not None,
        "note": "Each JSONL line wraps the native judge payload under record; no external request was made."}, indent=2) + "\n")
    print(f"Prepared judge inputs (not scores): {out}")


if __name__ == "__main__":
    main()
