"""Top-level CLI dispatcher."""

from __future__ import annotations

import argparse
import sys

from . import analyze, compare, discover, eval, train


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="mllm-comp")
    subparsers = parser.add_subparsers(dest="command", required=True)
    for name in ["eval", "analyze", "train", "discover", "compare"]:
        subparsers.add_parser(name)
    return parser


def main(argv: list[str] | None = None) -> int:
    if argv is None:
        argv = sys.argv[1:]
    args = build_parser().parse_args(argv[:1])
    rest = argv[1:]
    if args.command == "analyze":
        return analyze.main(rest)
    if args.command == "train":
        return train.main(rest)
    if args.command == "eval":
        return eval.main(rest)
    if args.command == "discover":
        return discover.main(rest)
    if args.command == "compare":
        return compare.main(rest)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
