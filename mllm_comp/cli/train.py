"""Train through an active external framework backend."""

from __future__ import annotations

import argparse

from mllm_comp.registry.catalog import CapabilityKind, capability_names

from ._common import (
    add_compression_mode_arg,
    add_config_args,
    add_output_args,
    add_runtime_mode_args,
    common_config_overrides,
    configure_logging,
    run_command,
    provided_overrides,
)


def main(argv: list[str] | None = None) -> int:
    parser = add_config_args(argparse.ArgumentParser(prog="mllm-comp train"))
    add_compression_mode_arg(parser)
    add_output_args(parser)
    parser.add_argument(
        "--backend",
        choices=capability_names(CapabilityKind.INTEGRATION, interface="train"),
        help="Training framework backend. Defaults to train.backend or ms_swift.",
    )
    parser.add_argument(
        "--checkpoint",
        help="Initial model path, or native Swift checkpoint to resume when using --from_run.",
    )
    parser.add_argument("--resume", help="Native backend resume state or checkpoint path.")
    parser.add_argument("--max_steps", type=int, help="Smoke-run step limit override.")
    parser.add_argument(
        "--artifact_format",
        choices=["transformers"],
        help="Reloadable compressed-checkpoint format passed to ms-swift.",
    )
    add_runtime_mode_args(parser)
    parser.add_argument(
        "--plan_only",
        action="store_true",
        help="Print the resolved ms-swift argv/environment without starting training.",
    )
    args = parser.parse_args(argv)
    if args.from_run is not None and args.resume is not None:
        parser.error("--from_run selects native resume state; use --checkpoint to select another saved checkpoint.")
    configure_logging(args.log_level)
    overrides = common_config_overrides(parser, args)
    overrides["train"] = provided_overrides(
        backend=args.backend,
        compression_mode=args.compression_mode,
        checkpoint_in=overrides["backbone"].get("model_path"),
        resume=args.resume,
        max_steps=args.max_steps,
        runtime_mode=args.runtime_mode,
        args=provided_overrides(artifact_format=args.artifact_format),
    )
    return run_command(
        args.config,
        task="train",
        from_run=args.from_run,
        overrides=overrides,
        plan_only=args.plan_only,
    )
