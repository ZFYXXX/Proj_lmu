"""Small JSONL/coverage helpers shared by launch and reports."""
import json
from collections import Counter
from pathlib import Path

from scripts.analysis.audio_temporal.metrics import jsonable


def write_json(path, value):
    Path(path).write_text(json.dumps(jsonable(value), ensure_ascii=False,
                                    indent=2, allow_nan=False), encoding="utf-8")


def write_row(handle, row):
    handle.write(json.dumps(jsonable(row), ensure_ascii=False, allow_nan=False) + "\n")
    handle.flush()


def rows(paths):
    for path in sorted(paths):
        with Path(path).open(encoding="utf-8") as source:
            for line in source:
                if line.strip():
                    yield json.loads(line)


def coverage(directory, expected, conditions):
    directory = Path(directory)
    targets = set(range(expected))
    baseline = Counter(r["doc_index"] for r in rows(directory.glob("samples.rank*.jsonl")))
    counts = {c["id"]: Counter() for c in conditions}
    unexpected = set()
    for row in rows(directory.glob("conditions.rank*.jsonl")):
        if row["condition_id"] not in counts:
            unexpected.add(row["condition_id"])
        else:
            counts[row["condition_id"]][row["doc_index"]] += 1

    def check(counter):
        return dict(rows=sum(counter.values()), unique=len(counter), expected=expected,
                    complete=set(counter) == targets and all(v == 1 for v in counter.values()))

    result = dict(baseline=check(baseline), conditions={k: check(v) for k, v in counts.items()},
                  unexpected_conditions=sorted(unexpected))
    result["complete"] = (result["baseline"]["complete"] and not unexpected
                          and all(r["complete"] for r in result["conditions"].values()))
    return result
