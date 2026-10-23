"""Compare experiment records from two run directories."""

from __future__ import annotations

import argparse
import json
import sys

from mllm_comp.core.errors import MLLMCompError
from mllm_comp.core.serialization import atomic_write_json
from mllm_comp.experiment_tools import compare_runs


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="mllm-comp compare")
    parser.add_argument("left_run", help="First run directory.")
    parser.add_argument("right_run", help="Second run directory.")
    parser.add_argument("--output", help="Also write the comparison to this JSON file.")
    args = parser.parse_args(argv)
    try:
        result = compare_runs(args.left_run, args.right_run)
        if args.output is not None:
            atomic_write_json(args.output, result, ensure_ascii=False)
    except (MLLMCompError, OSError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        print(json.dumps({
            "status": "failed",
            "error": {"type": type(exc).__name__, "message": str(exc)},
        }, ensure_ascii=False, indent=2))
        return 2
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
