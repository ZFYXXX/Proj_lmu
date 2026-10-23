"""Print exact ImageLLM support units without loading a model."""

from __future__ import annotations

import argparse
import json

from mllm_comp.compressors.support_report import (
    builtin_imagellm_rows,
    render_support_matrix_markdown,
)
from mllm_comp.core.vocabulary import ExecutionPhase, enum_values


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Report exact algorithm/variant/backbone/modality/phase support "
            "units; no Cartesian support is inferred."
        )
    )
    parser.add_argument(
        "--phase",
        choices=enum_values(ExecutionPhase),
        default=ExecutionPhase.GENERATE.value,
    )
    parser.add_argument("--format", choices=("markdown", "json"), default="markdown")
    parser.add_argument(
        "--require_status",
        help="Exit non-zero unless every reported unit has this exact status.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    rows = builtin_imagellm_rows(phase=args.phase)
    if args.format == "json":
        print(
            json.dumps(
                [row.to_dict() for row in rows],
                ensure_ascii=False,
                indent=2,
            )
        )
    else:
        print(render_support_matrix_markdown(rows), end="")
    if args.require_status is not None and (
        not rows or any(row.status != args.require_status for row in rows)
    ):
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
