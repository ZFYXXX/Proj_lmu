"""Run configured mechanism-analysis Probes over one lmms-eval sample loop."""

from __future__ import annotations

import argparse

from mllm_comp.analysis.definition import AnalysisOverrides
from mllm_comp.core.vocabulary import (
    AnalysisPhase,
    enum_values,
)

from ._common import (
    add_analysis_probe_override_args,
    add_compression_mode_arg,
    add_config_args,
    add_output_args,
    add_runtime_args,
    add_sample_args,
    common_config_overrides,
    configure_logging,
    run_command,
    provided_overrides,
)


def main(argv: list[str] | None = None) -> int:
    parser = add_config_args(argparse.ArgumentParser(prog="mllm-comp analyze"))
    add_compression_mode_arg(parser)
    add_output_args(parser)
    add_runtime_args(parser)
    add_sample_args(parser)
    add_analysis_probe_override_args(parser)
    parser.add_argument(
        "--checkpoint",
        help="Local model path, or saved model/HTA artifact when using --from_run.",
    )
    parser.add_argument(
        "--phase",
        choices=enum_values(AnalysisPhase),
        help="Analysis phase recorded in AnalysisRecord.",
    )
    parser.add_argument(
        "--probe",
        action="append",
        help="Analysis probe name. Can be repeated to override analysis.probes.",
    )
    parser.add_argument(
        "--plan_only", action="store_true",
        help="Print the effective analysis definition without loading a model or writing artifacts.",
    )
    args = parser.parse_args(argv)
    configure_logging(args.log_level)
    overrides = common_config_overrides(parser, args)
    overrides["analysis"] = provided_overrides(
        compression_mode=args.compression_mode,
        phase=args.phase,
        tasks=(
            [value.strip() for value in args.tasks.split(",") if value.strip()]
            if args.tasks is not None else None
        ),
        sample_limit=args.sample_limit,
        runtime_batch_size=args.runtime_batch_size,
    )
    return run_command(
        args.config,
        task="analyze",
        from_run=args.from_run,
        overrides=overrides,
        device=args.device,
        plan_only=args.plan_only,
        analysis_overrides=AnalysisOverrides(
            probes=args.probe,
            layers=args.layers,
            metrics=args.metrics,
            token_groups=args.token_groups,
            collect_points=args.collect_points,
            operations=args.operations,
            target_token_types=args.target_token_types,
            attn_edges=args.attn_edges,
            save_raw=args.save_raw,
        ),
    )
