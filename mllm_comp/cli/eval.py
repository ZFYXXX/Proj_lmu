"""Evaluate a platform-owned runtime through lmms-eval."""

from __future__ import annotations

import argparse

from mllm_comp.core.vocabulary import (
    DistributedLauncher,
    EvalTarget,
    enum_values,
)
from mllm_comp.registry.catalog import CapabilityKind, capability_names

from ._common import (
    add_compression_mode_arg,
    add_config_args,
    add_output_args,
    add_runtime_args,
    add_runtime_mode_args,
    add_sample_args,
    common_config_overrides,
    configure_logging,
    run_command,
    provided_overrides,
)


def main(argv: list[str] | None = None) -> int:
    parser = add_config_args(argparse.ArgumentParser(prog="mllm-comp eval"))
    add_compression_mode_arg(parser)
    add_output_args(parser)
    add_runtime_args(parser)
    add_sample_args(parser)
    parser.add_argument(
        "--backend",
        choices=capability_names(CapabilityKind.INTEGRATION, interface="eval"),
        help="Evaluation framework backend. Defaults to eval.backend or lmms_eval.",
    )
    parser.add_argument("--checkpoint", help="Model path override, or saved model/HTA artifact when using --from_run.")
    parser.add_argument(
        "--target",
        choices=enum_values(EvalTarget),
        help="Metric target written to the manifest. Defaults should derive from compression mode.",
    )
    parser.add_argument(
        "--generation_max_new_tokens",
        type=int,
        help="Override generation.max_new_tokens in eval args.",
    )
    parser.add_argument(
        "--use_cache",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Explicitly enable or disable generation KV cache for protocol matching.",
    )
    parser.add_argument("--temperature", type=float, help="Generation temperature override.")
    parser.add_argument("--top_p", type=float, help="Generation top-p override.")
    parser.add_argument(
        "--num_processes",
        type=int,
        help=(
            "Number of local lmms-eval worker processes. Values above one use "
            "one process per visible GPU."
        ),
    )
    parser.add_argument(
        "--distributed_launcher",
        choices=enum_values(DistributedLauncher),
        help="Launcher used when --num_processes is greater than one.",
    )
    parser.add_argument(
        "--main_process_port",
        type=int,
        help="Optional local rendezvous port for Accelerate or Torchrun.",
    )
    add_runtime_mode_args(parser)
    parser.add_argument(
        "--plan_only",
        "--dry_run",
        action="store_true",
        help="Print native lmms-eval argv/environment/cwd/versions/outputs without launching.",
    )
    args = parser.parse_args(argv)
    configure_logging(args.log_level)
    overrides = common_config_overrides(parser, args)
    overrides["eval"] = provided_overrides(
        backend=args.backend,
        compression_mode=args.compression_mode,
        checkpoint_path=overrides["backbone"].get("model_path"),
        target=args.target,
        tasks=(
            [value.strip() for value in args.tasks.split(",") if value.strip()]
            if args.tasks is not None else None
        ),
        sample_limit=args.sample_limit,
        runtime_batch_size=args.runtime_batch_size,
        num_processes=args.num_processes,
        distributed_launcher=args.distributed_launcher,
        main_process_port=args.main_process_port,
        runtime_mode=args.runtime_mode,
        generation=provided_overrides(
            max_new_tokens=args.generation_max_new_tokens,
            use_cache=args.use_cache,
            temperature=args.temperature,
            top_p=args.top_p,
        ),
    )
    return run_command(
        args.config,
        task="eval",
        from_run=args.from_run,
        overrides=overrides,
        plan_only=args.plan_only,
        device=args.device,
    )
