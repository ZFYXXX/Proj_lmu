"""MMAU-mini records, auditable weak labels, and completed-run coverage."""
from __future__ import annotations
from collections import Counter
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
TASK = "mmaunew3_test_mini"
DOMAINS = {"S1": "sound", "S2": "sound", "M1": "music", "M2": "music",
           "SP1": "speech", "SP2": "speech"}

def stage_names(layers):
    return ["C", *(f"E_{i:03d}" for i in layers), "P_in", "P_out"]

def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                          encoding="utf-8")

def load_records(manifest, audio_root):
    records = json.loads(Path(manifest).read_text())
    if not isinstance(records, list) or len(records) != 1000:
        raise ValueError("Expected the 1,000-row MMAU test-mini manifest")
    seen = set()
    for index, row in enumerate(records):
        if row.get("split") != "test-mini" or row.get("task") not in set(DOMAINS.values()):
            raise ValueError(f"Not a test-mini record: {index}")
        if not row.get("id") or row["id"] in seen or row.get("answer") not in row["choices"]:
            raise ValueError(f"Invalid ID/reference answer: {index}")
        seen.add(row["id"])
        audio = (Path(audio_root) / row["audio_id"]).resolve()
        if not audio.is_file():
            raise FileNotFoundError(audio)
        row.update(doc_index=index, audio_path=str(audio), group_id=str(audio))
    return records

def inventory(rows):
    return {"n": len(rows), "task_counts": dict(Counter(r["task"] for r in rows)),
            "unique_audio_paths": len({r["group_id"] for r in rows})}

def read_annotations(path):
    if not path:
        return {}
    result = {}
    for line in Path(path).read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row["id"] in result:
            raise ValueError(f"Duplicate annotation: {row['id']}")
        result[row["id"]] = row
    return result

def validate_speech(row, duration=None):
    for field in ("phones", "words"):
        previous = 0.0
        for item in row.get(field, []):
            start, end = float(item["start"]), float(item["end"])
            if not (0 <= start < end) or start < previous - 1e-5:
                raise ValueError(f"Invalid/overlapping {field} intervals: {row['id']}")
            if duration is not None and end > duration + 0.1:
                raise ValueError(f"Alignment exceeds audio duration: {row['id']}")
            if not isinstance(item["label"], str) or not item["label"].strip():
                raise ValueError("Empty alignment label")
            previous = end

def load_run(run_dir):
    run = Path(run_dir)
    plan = json.loads((run / "plan.json").read_text())
    selected = json.loads((run / "selected.json").read_text())
    expected = {r["id"]: r for r in selected}
    found = {}
    for rank in range(plan["shard_count"]):
        shard = run / f"shard-{rank:03d}"
        marker = shard / "complete.json"
        if not marker.is_file():
            raise ValueError(f"Worker {rank} did not complete")
        lines = [json.loads(x) for x in (shard / "samples.jsonl").read_text().splitlines() if x]
        if len(lines) != json.loads(marker.read_text())["samples"]:
            raise ValueError("Worker completion count mismatch")
        for row in lines:
            if row["id"] not in expected or row["id"] in found:
                raise ValueError("Unexpected/duplicate sample")
            for key in ("doc_index", "task", "group_id"):
                if row[key] != expected[row["id"]][key]:
                    raise ValueError(f"Sample mismatch: {key}")
            row["_directory"] = str(shard / row["directory"])
            for name in ("clean.npz", "targets.npz", "labels.json"):
                if not (Path(row["_directory"]) / name).is_file():
                    raise ValueError(f"Missing {name}: {row['id']}")
            for variant in row["variants"]:
                if not (Path(row["_directory"]) / f"{variant['name']}.npz").is_file():
                    raise ValueError("Missing intervention features")
            found[row["id"]] = row
    if set(found) != set(expected):
        raise ValueError("Incomplete selected-sample coverage")
    return plan, [found[r["id"]] for r in selected]

