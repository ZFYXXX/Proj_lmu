"""Recover interrupted Qwen2.5 evaluation queues without overwriting results.

This helper amends only the provenance state of an existing run after the
source snapshot changed additively.  It preserves the old protocol/manifest,
records the amendment, and leaves completed cell markers untouched.  The
normal queue scripts then rerun only cells that do not have a valid
``complete.json`` marker.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from mllm_comp.analysis.common.anchor7b_runtime import file_sha
from mllm_comp.analysis.common.learned_topk_experiment import dependency_identity, owned_output, source_identity
from mllm_comp.core.serialization import atomic_write_json as write_json


def now():
    return datetime.now(timezone.utc).isoformat()


def read(path):
    return json.loads(Path(path).read_text())


def amend(root: Path, reason: str):
    root = owned_output(root)
    protocol_path = root / "protocol.json"
    manifest_path = root / "manifest.json"
    if not protocol_path.exists() or not manifest_path.exists():
        raise FileNotFoundError(f"not a prepared run: {root}")
    protocol = read(protocol_path)
    manifest = read(manifest_path)
    recovery = root / "recovery"
    recovery.mkdir(exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + f"_{time.time_ns()}"
    previous_status = manifest.get("status")
    old_protocol = recovery / f"protocol.before_{stamp}.json"
    old_manifest = recovery / f"manifest.before_{stamp}.json"
    shutil.copy2(protocol_path, old_protocol)
    shutil.copy2(manifest_path, old_manifest)

    # Verify that immutable dataset/checkpoint/task pins still match before
    # changing only the source provenance.  A changed input is a hard stop.
    for path, checksum in protocol.get("input_hashes", {}).items():
        if file_sha(path) != checksum:
            raise ValueError(f"Pinned input changed; refusing recovery: {path}")

    old_source = protocol.get("source")
    old_dependencies = protocol.get("dependencies")
    new_source = source_identity(ROOT)
    new_dependencies = dependency_identity()
    if old_dependencies != new_dependencies:
        raise ValueError("Evaluator dependencies changed; audit in a fresh experiment before recovery")
    if old_source == new_source and old_dependencies == new_dependencies:
        raise ValueError("run is already compatible with the current source; no amendment needed")

    protocol["source"] = new_source
    protocol["dependencies"] = new_dependencies
    protocol.setdefault("recovery_amendments", []).append({
        "created_utc": now(),
        "reason": reason,
        "previous_protocol_sha256": file_sha(old_protocol),
        "previous_source": old_source,
        "previous_dependencies": old_dependencies,
        "preserved_complete_markers": True,
        "rerun_policy": "rerun only cells without a valid complete.json; preserve prior attempts",
    })
    write_json(protocol_path, protocol)
    manifest["status"] = "prepared"
    manifest.pop("error", None)
    manifest.pop("traceback", None)
    manifest.pop("prerequisite_status", None)
    manifest["protocol_sha256"] = file_sha(protocol_path)
    manifest["recovered_utc"] = now()
    manifest["recovery_reason"] = reason
    manifest["preserved_previous_manifest"] = str(old_manifest)
    write_json(manifest_path, manifest)
    write_json(recovery / f"amendment_{stamp}.json", {
        "run": str(root), "protocol": str(protocol_path), "manifest": str(manifest_path),
        "previous_protocol": str(old_protocol), "previous_manifest": str(old_manifest),
        "new_protocol_sha256": file_sha(protocol_path), "created_utc": now(),
    })
    print(json.dumps({"root": str(root), "protocol_sha256": file_sha(protocol_path),
                      "old_status": previous_status, "preserved": True}, ensure_ascii=False))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", action="append", type=Path, required=True)
    parser.add_argument("--reason", required=True)
    args = parser.parse_args()
    for root in args.root:
        amend(root, args.reason)


if __name__ == "__main__":
    main()
