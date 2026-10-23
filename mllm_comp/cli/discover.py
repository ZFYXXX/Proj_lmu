"""Discover declared capabilities and their verification evidence."""

from __future__ import annotations

import argparse
import json
import sys

from mllm_comp.core.errors import MLLMCompError
from mllm_comp.core.vocabulary import (
    CompressionMethod,
    ExecutionPhase,
    Modality,
    enum_values,
)
from mllm_comp.experiment_tools import discover_capabilities
from mllm_comp.registry.catalog import CapabilityKind


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="mllm-comp discover")
    parser.add_argument(
        "--kind", choices=enum_values(CapabilityKind), default="compressor",
        help="Capability kind to inspect. Defaults to compressor.",
    )
    parser.add_argument("--name", help="Capability name; algorithm name for compressors.")
    parser.add_argument("--method", choices=enum_values(CompressionMethod))
    parser.add_argument("--variant", help="Compression variant name.")
    parser.add_argument("--binding", help="Compression binding name.")
    parser.add_argument("--backbone", help="Backbone name.")
    parser.add_argument("--modality", choices=enum_values(Modality))
    parser.add_argument("--phase", choices=enum_values(ExecutionPhase))
    parser.add_argument(
        "--extension", action="append", default=[],
        help="Call an explicit module:register compression extension; repeat for multiple extensions.",
    )
    args = parser.parse_args(argv)
    try:
        result = discover_capabilities(
            kind=args.kind,
            name=args.name,
            method=args.method,
            variant=args.variant,
            binding=args.binding,
            backbone=args.backbone,
            modality=args.modality,
            phase=args.phase,
            extensions=tuple(args.extension),
        )
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
