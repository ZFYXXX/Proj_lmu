#!/usr/bin/env python
"""Join lmms-eval quality metrics with audio-compression efficiency summaries."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from mllm_comp.analysis.audio_comparison import compare_runs


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path}: expected a JSON object")
    return payload


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Compare common lmms-eval quality metrics and per-run efficiency summaries."
        )
    )
    parser.add_argument("--baseline_result", required=True)
    parser.add_argument("--compressed_result", required=True)
    parser.add_argument("--baseline_efficiency", required=True)
    parser.add_argument("--compressed_efficiency", required=True)
    parser.add_argument("--baseline_manifest", required=True)
    parser.add_argument("--compressed_manifest", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)

    paths = {
        "baseline_result": Path(args.baseline_result),
        "compressed_result": Path(args.compressed_result),
        "baseline_efficiency": Path(args.baseline_efficiency),
        "compressed_efficiency": Path(args.compressed_efficiency),
        "baseline_manifest": Path(args.baseline_manifest),
        "compressed_manifest": Path(args.compressed_manifest),
    }
    comparison = compare_runs(
        baseline_result=_load_json(paths["baseline_result"]),
        compressed_result=_load_json(paths["compressed_result"]),
        baseline_efficiency=_load_json(paths["baseline_efficiency"]),
        compressed_efficiency=_load_json(paths["compressed_efficiency"]),
        baseline_manifest=_load_json(paths["baseline_manifest"]),
        compressed_manifest=_load_json(paths["compressed_manifest"]),
    )
    comparison["sources"] = {
        name: {"path": str(path.resolve()), "sha256": _sha256(path)}
        for name, path in paths.items()
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(comparison, ensure_ascii=False, indent=2)
    output.write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
