"""Shared CLI helpers."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from mllm_comp.config import config_to_dict
from mllm_comp.config.loader import _apply_overrides, load_config
from mllm_comp.config.schema import experiment_config_from_dict
from mllm_comp.analysis.definition import AnalysisOverrides
from mllm_comp.core.errors import ConfigError
from mllm_comp.core.vocabulary import CompressionMode, RuntimeMode, enum_values
from mllm_comp.experiments import ExperimentResult, ExperimentTask, restore_config, run_experiment


def add_config_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--config", help="Path to experiment config.")
    source.add_argument("--from_run", help="Restore a saved run's checkpoint into an independent new experiment.")
    parser.add_argument(
        "--log_level",
        type=str.upper,
        choices=("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"),
        default="INFO",
        help="Python logging level for this command process.",
    )
    return parser


def add_output_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument("--output_dir", help="Exact new or empty run directory; generated when omitted from config.")
    parser.add_argument("--manifest", help="Exact manifest path; defaults to <run_dir>/manifest.json.")
    parser.add_argument("--run_id", help="Override experiment.run_id in the config.")
    return parser


def add_runtime_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument("--device", help="Runtime device override, for example cuda:0 or cpu.")
    parser.add_argument(
        "--runtime_batch_size",
        type=int,
        help=(
            "Override task runtime batch size; limits depend on the command and "
            "backbone."
        ),
    )
    return parser


def add_runtime_mode_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--runtime_mode", choices=enum_values(RuntimeMode),
        help="Runtime instrumentation mode; default disables profile/debug hooks.",
    )
    mode.add_argument(
        "--profile", dest="runtime_mode", action="store_const",
        const=RuntimeMode.PROFILE.value,
        help="Shortcut for --runtime_mode profile; enable latency/memory profiling.",
    )
    mode.add_argument(
        "--debug", dest="runtime_mode", action="store_const",
        const=RuntimeMode.DEBUG.value,
        help="Shortcut for --runtime_mode debug; enable per-stage/per-layer token-count debugging.",
    )
    return parser


def add_sample_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument(
        "--tasks",
        help="Comma-separated task names overriding the configured task list.",
    )
    parser.add_argument("--sample_limit", type=int, help="Limit task samples for smoke runs.")
    return parser


def add_compression_mode_arg(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument(
        "--compression_mode",
        choices=enum_values(CompressionMode),
        help="Whether this task runs without compression or with configured compression stages.",
    )
    return parser


def add_analysis_probe_override_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument(
        "--layers",
        help="Temporary analysis layer override, for example all or 0,8,16,24,31.",
    )
    parser.add_argument(
        "--metrics",
        help="Temporary analysis metric override, for example cosine or js,top1_changed.",
    )
    parser.add_argument(
        "--token_groups",
        help="Temporary token-group override, for example system,text,vision.",
    )
    parser.add_argument(
        "--collect_points",
        help="Representation-dynamics collection points, for example layer_input,after_attn,after_mlp.",
    )
    parser.add_argument(
        "--operations",
        help="Layer-importance operations, for example layer,layer_strict,attn,ffn.",
    )
    parser.add_argument(
        "--target_token_types",
        help="Layer-importance target token types, for example vision,text,system.",
    )
    parser.add_argument(
        "--attn_edges",
        help="Layer-importance target2source attention edges, for example t2v,v2v.",
    )
    parser.add_argument(
        "--save_raw",
        action="store_true",
        help="Request raw per-sample analysis artifacts when supported by the probe.",
    )
    return parser


def configure_logging(level: str) -> None:
    """Configure CLI logging without replacing existing application handlers."""

    logging.basicConfig(level=level)
    logging.getLogger().setLevel(level)


def provided_overrides(**values: Any) -> dict[str, Any]:
    """Keep explicitly supplied CLI values, including zero and False."""

    return {name: value for name, value in values.items() if value is not None}


def common_config_overrides(
    parser: argparse.ArgumentParser,
    args: argparse.Namespace,
) -> dict[str, Any]:
    checkpoint = args.checkpoint
    if checkpoint is not None:
        checkpoint = checkpoint.strip()
        if not checkpoint:
            parser.error("--checkpoint must be a non-empty path.")
    return {
        "experiment": provided_overrides(run_id=args.run_id),
        "backbone": provided_overrides(
            model_path=checkpoint,
            config=provided_overrides(device=getattr(args, "device", None)),
        ),
        "output": provided_overrides(
            run_dir=args.output_dir,
            manifest_path=args.manifest,
        ),
    }


def run_command(
    path: str | Path | None,
    *,
    task: ExperimentTask,
    overrides: Mapping[str, Any] | None = None,
    from_run: str | Path | None = None,
    device: str | None = None,
    analysis_overrides: AnalysisOverrides | None = None,
    plan_only: bool = False,
) -> int:
    """Parse command overrides once, then print the common experiment outcome."""

    try:
        if from_run is None:
            config = load_config(path, overrides=overrides)
        else:
            checkpoint = (overrides or {}).get("backbone", {}).get("model_path")
            config = restore_config(from_run, task=task, checkpoint=checkpoint)
            restored_overrides = {name: dict(values) for name, values in (overrides or {}).items()}
            # --checkpoint selects the restore artifact. restore_config assigns
            # it to model weights, HTA weights or Swift resume as appropriate.
            for section, field in (("backbone", "model_path"), ("eval", "checkpoint_path"), ("train", "checkpoint_in")):
                restored_overrides.get(section, {}).pop(field, None)
            config = experiment_config_from_dict(
                _apply_overrides(config_to_dict(config), restored_overrides), path=config.path,
            )
    except (ConfigError, ValueError, OSError) as exc:
        result = ExperimentResult(task=task)
        result.record_failure(exc, stage="configuration")
    else:
        result = run_experiment(
            config, task=task, device=device,
            analysis_overrides=analysis_overrides, plan_only=plan_only,
        )
    if result.error is not None:
        print(result.error.message, file=sys.stderr)
    print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))
    return result.returncode if result.returncode is not None else 0
