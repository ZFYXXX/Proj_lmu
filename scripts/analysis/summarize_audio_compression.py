#!/usr/bin/env python
"""Summarize Qwen2.5-Omni audio-compression diagnostic JSONL files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from mllm_comp.analysis.audio_summary import _read_records, _resolve_paths, summarize_records


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Summarize per-sample audio compression token and latency traces."
    )
    parser.add_argument(
        "inputs",
        nargs="+",
        help="Diagnostic JSONL files, glob patterns, or run directories.",
    )
    parser.add_argument("--output", help="Optional JSON summary path.")
    args = parser.parse_args(argv)

    paths = _resolve_paths(args.inputs)
    if not paths:
        parser.error("No compression diagnostic JSONL files were found.")
    summary = summarize_records(_read_records(paths))
    summary["source_files"] = [str(path.resolve()) for path in paths]
    text = json.dumps(summary, ensure_ascii=False, indent=2)
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
