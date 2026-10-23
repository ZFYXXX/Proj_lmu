"""Resolve frozen cohort entries before image processing."""

import hashlib
import json
from pathlib import Path


def select_manifest_entries(path, *, task_name, stage, shard):
    if stage not in {"smoke","development","confirmation","question_review","budget_review"} or shard not in {0,1}:
        raise ValueError("Explicit evidence cohort stage and shard (0/1) required.")
    raw = Path(path).read_bytes()
    manifest = json.loads(raw)
    selected = {}
    for row in manifest["entries"]:
        if row["task"] != task_name or row["shard"] != shard:
            continue
        include = row.get("smoke",False) if stage == "smoke" else row["split"] == stage if stage in {"development","confirmation"} else row.get(stage,False)
        if not include:
            continue
        entry = {**row,"stage":stage,"primary_sample_id":row["sample_id"]}
        if stage == "question_review":
            entry["doc_id"] = row["second_doc_id"]
            entry["sample_id"] = f"{task_name}:{row['second_doc_id']}"
        if entry["doc_id"] in selected:
            raise ValueError("Duplicate document in frozen cohort.")
        selected[entry["doc_id"]] = entry
    return selected, {"manifest_sha256":hashlib.sha256(raw).hexdigest(),"manifest_stage":stage,
                      "manifest_shard":shard,"selected_documents":len(selected),"selection_reads_answers":False}
