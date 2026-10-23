"""Post-hoc output-format sensitivity analysis; never changes native scores."""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from mllm_comp.core.serialization import atomic_write_json, atomic_write_text
from scripts.eval.lmms_eval.run_anchor7b_full import clustered_delta


def audit(source, output):
    entries = json.loads((source / "ai2d/samples.json").read_text())["entries"]
    expected = {e["sample_id"] for e in entries}
    if len(entries) != 3088:
        raise ValueError("This audit expects the entire 3088-question AI2D split")
    native, alternate = defaultdict(list), defaultdict(list)
    methods, cases, matched = None, [], set()
    for entry in entries:
        folder = source / "ai2d/samples/full" / entry["sample_id"].replace(":", "_")
        if not (folder / "summary.json").exists():
            raise ValueError("AI2D full evaluation is not complete")
        raw = [json.loads(s) for s in (folder / "conditions.jsonl").read_text().splitlines()]
        rows = {r["variant"]: r for r in raw}
        if len(raw) != len(rows) or not {"dense", "feature_anchor_adapter"}.issubset(rows):
            raise ValueError("Duplicate or missing method")
        methods = set(rows) if methods is None else methods
        if set(rows) != methods:
            raise ValueError("Incomplete comparison arms")
        target = rows["dense"]["target"]
        changed = {}
        for method, row in rows.items():
            if row["sample_id"] != entry["sample_id"] or row["image_fingerprint"] != entry["image_fingerprint"]:
                raise ValueError("Wrong sample or image provenance")
            match = re.match(r"^\s*([A-D])(?:[.\s)]|$)", row["prediction"])
            parsed = match.group(1) if match else None
            score = float(parsed == target)
            changed[method] = {**row, "primary_task_score": score}
            if score != row["primary_task_score"]:
                cases.append({"sample_id": entry["sample_id"], "method": method,
                    "prediction": row["prediction"], "target": target,
                    "native_score": row["primary_task_score"], "first_letter_score": score})
        native[entry["image_fingerprint"]].append(rows)
        alternate[entry["image_fingerprint"]].append(changed)
        matched.add(entry["sample_id"])
    if matched != expected:
        raise ValueError("Missing or unexpected sample IDs")
    table = {}
    for method in sorted(methods):
        original = [q[method] for questions in native.values() for q in questions]
        other = [q[method] for questions in alternate.values() for q in questions]
        table[method] = {"n_questions": len(original), "n_images": len(native),
            "native_score": sum(r["primary_task_score"] for r in original) / len(original),
            "first_letter_score": sum(r["primary_task_score"] for r in other) / len(other),
            "native_delta_vs_dense": clustered_delta(list(native.values()), method),
            "first_letter_delta_vs_dense": clustered_delta(list(alternate.values()), method),
            "format_sensitive_cases": sum(c["method"] == method for c in cases)}
    result = {"status": "complete", "source_run": str(source),
        "analysis": "Post-hoc output-format sensitivity; the native upstream metric is retained",
        "rule": "Take only an unambiguous leading A-D followed by punctuation/whitespace/end; otherwise count incorrect",
        "limitation": "A leading letter can disagree with later explanation. This is an alternate parser, not semantic adjudication.",
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "tables": table,
        "case_count": len(cases)}
    atomic_write_json(output.with_suffix(".json"), result)
    atomic_write_json(output.with_name(output.stem + "_cases.json"), {"cases": cases})
    lines = ["# AI2D answer-format sensitivity", "", result["analysis"], "",
        "| Method | Native score | Leading-option score | Delta vs dense under leading-option score (pp) |",
        "|---|---:|---:|---:|"]
    for method, row in table.items():
        lines.append(f"| {method} | {100 * row['native_score']:.2f}% | {100 * row['first_letter_score']:.2f}% | {100 * row['first_letter_delta_vs_dense']['mean']:+.2f} |")
    lines += ["", result["limitation"], "", "Paired image-cluster bootstrap intervals and changed cases are saved separately."]
    atomic_write_text(output.with_suffix(".md"), "\n".join(lines) + "\n")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.source_run.resolve(), args.output.resolve())
    print(json.dumps(result["tables"], indent=2))
