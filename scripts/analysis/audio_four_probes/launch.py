"""Launch independent, disjoint data shards; each GPU owns one model replica."""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

import yaml

from mllm_comp.analysis.definition import AnalysisOverrides, resolve_analysis_definition
from mllm_comp.config.schema import experiment_config_from_dict
from mllm_comp.core.serialization import atomic_write_json


SUITE_DIR = Path(__file__).resolve().parent
ROOT = SUITE_DIR.parents[2]
PROBES = (
    "representation_dynamics", "prediction_dynamics", "attention_flow",
    "residual_stream_decomposition",
)
TAYLOR_PROBE = "hierarchical_taylor"
DATA_ENV = {
    "mmsu_jsonl": "MMSU_JSONL",
    "mmsu_dataset_dir": "MMSUFINAL_DATASET_DIR",
}


def layer_selection(value: str, count: int) -> list[int]:
    layers = list(range(count)) if value == "all" else list(dict.fromkeys(
        int(item.strip()) for item in value.split(",") if item.strip()))
    if not layers or any(index < 0 or index >= count for index in layers):
        raise ValueError(f"Layer selection {value!r} is outside 0..{count - 1}")
    return layers


def prepare_config(raw: dict, *, checkpoint: str, output_dir: Path,
                   layers: str | None = None, causal_layers: str | None = None,
                   sample_limit: int | None = None, attention_layer_batch_size: int | None = None,
                   taylor_layer_batch_size: int | None = None,
                   probes: str | None = None) -> tuple[dict, dict]:
    """Resolve the real architecture and both observation/causal layer selections."""
    raw = deepcopy(raw)
    checkpoint_path = Path(checkpoint).expanduser().resolve()
    architecture = json.loads((checkpoint_path / "config.json").read_text())
    text_config = architecture["thinker_config"]["text_config"]
    count = int(text_config["num_hidden_layers"])
    requested = ([spec["name"] for spec in raw["analysis"]["probes"] if spec.get("enabled", True)]
                 if probes is None else probes.split(","))
    supported = (*PROBES, TAYLOR_PROBE)
    if not requested or len(set(requested)) != len(requested) or set(requested) - set(supported):
        raise ValueError(f"--probes must select distinct names from {supported}")
    is_taylor = TAYLOR_PROBE in requested
    if is_taylor and len(requested) != 1:
        raise ValueError("Run hierarchical_taylor separately from the four forward-only probes")
    tasks = raw["analysis"]["tasks"]
    if tasks != ["mmsufinal"]:
        raise ValueError("This analysis suite only supports tasks: [mmsufinal]")
    if sample_limit is not None and sample_limit < 1:
        raise ValueError("--sample-limit must be positive; omit it for the full dataset")
    if attention_layer_batch_size is not None and attention_layer_batch_size < 1:
        raise ValueError("--attention-layer-batch-size must be positive")
    if taylor_layer_batch_size is not None and (taylor_layer_batch_size < 1 or not is_taylor):
        raise ValueError("--taylor-layer-batch-size must be positive and used with hierarchical_taylor")
    raw["backbone"]["model_path"] = str(checkpoint_path)
    raw["output"] = {
        "run_dir": str(output_dir), "manifest_path": str(output_dir / "manifest.json")}
    if sample_limit is not None:
        raw["analysis"]["sample_limit"] = sample_limit
    selected = []
    for spec in raw["analysis"]["probes"]:
        name = spec["name"]
        if name not in requested:
            continue
        spec["enabled"] = True
        if name != TAYLOR_PROBE:
            spec["save_raw"] = True
        value = layers or spec.get("layers", "all")
        spec["layers"] = layer_selection(value, count) if isinstance(value, str) else value
        if name == "residual_stream_decomposition":
            # A smoke --layers override also bounds deletion unless explicitly overridden.
            value = causal_layers or layers or spec.get("causal_layers", "all")
            spec["causal_layers"] = layer_selection(value, count) if isinstance(value, str) else value
        elif name == "attention_flow":
            if attention_layer_batch_size is not None:
                spec["layer_batch_size"] = attention_layer_batch_size
        elif name == TAYLOR_PROBE and taylor_layer_batch_size is not None:
            spec["layer_batch_size"] = taylor_layer_batch_size
        selected.append(spec)
    if {item["name"] for item in selected} != set(requested):
        raise ValueError("The YAML does not contain every requested probe")
    raw["analysis"]["probes"] = selected
    config = experiment_config_from_dict(raw)
    if (config.backbone.name != "qwen25_omni" or config.pipeline.stages
            or str(config.analysis.compression_mode) != "none"
            or config.backbone.config.get("attn_implementation") != "eager"
            or config.backbone.config.get("enable_talker") is not False):
        raise ValueError("The suite requires uncompressed Qwen2.5-Omni, eager attention, Talker disabled")
    resolve_analysis_definition(config, AnalysisOverrides())
    passes = 0
    for spec in selected:
        if spec["name"] == "residual_stream_decomposition":
            passes += 1 + len(spec["causal_layers"]) * len(spec["causal_operations"])
        elif spec["name"] in ("attention_flow", TAYLOR_PROBE):
            passes += math.ceil(len(spec["layers"]) / spec.get("layer_batch_size", 1))
        else:
            passes += 1
    plan = {
        "benchmark": tasks[0], "leaf_tasks": list(tasks),
        "decoder_layers": count, "query_heads": int(text_config["num_attention_heads"]),
        "analysis_forwards_per_sample": passes, "baseline_generations_per_sample": 1,
        "probes": requested, "input_protocol": "lmms_eval_chat",
        "sample_limit_per_leaf_task": raw["analysis"].get("sample_limit"),
    }
    if is_taylor:
        validation = selected[0].get("causal_validation", {}).get("enabled", False)
        plan.update(
            analysis_backwards_per_sample=passes,
            objective="teacher_forced_answer_cross_entropy_v1",
            target_source="lmms_eval.doc_to_target", ground_truth_required=True,
            causal_validation_enabled=validation,
            result_format="native_taylor_task_scores_and_ranks")
        if validation:
            # Candidate-dependent validation adds forwards to the capture passes.
            plan["analysis_forwards_per_sample"] = None
            plan["taylor_capture_forwards_per_sample"] = passes
    else:
        plan.update(prediction_position="first_output_token_unconditioned_on_answer",
                    causal_scope="all_nonpadding_tokens")
    return raw, plan


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--gpus", default=os.environ.get("CUDA_VISIBLE_DEVICES", "0"))
    parser.add_argument("--sample-limit", type=int)
    parser.add_argument("--layers", help="all or comma-separated zero-based LLM layer indices")
    parser.add_argument("--causal-layers", help="defaults to --layers or the YAML causal_layers")
    parser.add_argument("--attention-layer-batch-size", type=int,
                        help="override the YAML attention layer_batch_size")
    parser.add_argument("--taylor-layer-batch-size", type=int,
                        help="override the YAML Taylor capture layer_batch_size")
    parser.add_argument("--probes", help="comma-separated subset; default: enabled probes in YAML")
    parser.add_argument("--dry-run", action="store_true", help="print plan without loading weights or starting workers")
    for field, variable in DATA_ENV.items():
        parser.add_argument("--" + field.replace("_", "-"), default=os.environ.get(variable))
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    output_dir = Path(args.output_dir).expanduser().resolve()
    try:
        raw, plan = prepare_config(
            yaml.safe_load(Path(args.config).read_text()), checkpoint=args.checkpoint,
            output_dir=output_dir, layers=args.layers, causal_layers=args.causal_layers,
            sample_limit=args.sample_limit, attention_layer_batch_size=args.attention_layer_batch_size,
            taylor_layer_batch_size=args.taylor_layer_batch_size,
            probes=args.probes)
        devices = [item.strip() for item in args.gpus.split(",") if item.strip()]
        if not devices or len(set(devices)) != len(devices):
            raise ValueError("--gpus must list distinct visible GPU IDs")
        if TAYLOR_PROBE in plan["probes"] and len(devices) != 1:
            raise ValueError("hierarchical_taylor uses one GPU per benchmark for native task-wide "
                             "score aggregation; select one --gpus ID or set TAYLOR_GPU")
        if args.sample_limit is not None and args.sample_limit < len(devices):
            raise ValueError("Use at least one sample per GPU, or fewer --gpus for a smoke run")
        paths = {field: str(Path(getattr(args, field)).expanduser().resolve())
                 for field in DATA_ENV if getattr(args, field)}
        for field, value in paths.items():
            if not Path(value).exists():
                raise ValueError(f"{field} does not exist: {value}")
        plan.update(gpus=devices, shard_count=len(devices), output_dir=str(output_dir), data_paths=paths)
        print(json.dumps(plan, ensure_ascii=False, indent=2), flush=True)
        if args.dry_run:
            return 0
        # The output directory must be new; failed and completed runs remain inspectable.
        output_dir.mkdir(parents=True, exist_ok=False)
        (output_dir / "analysis.resolved.yaml").write_text(yaml.safe_dump(raw, sort_keys=False))
        atomic_write_json(output_dir / "plan.json", plan)
    except (OSError, KeyError, ValueError) as exc:
        parser.error(str(exc))
    workers = []
    try:
        for rank, gpu in enumerate(devices):
            shard_dir = output_dir / f"shard-{rank:03d}"
            shard_dir.mkdir()
            log = (shard_dir / "worker.log").open("w")
            env = {**os.environ, "CUDA_VISIBLE_DEVICES": gpu, "PYTHONUNBUFFERED": "1"}
            env["PYTHONPATH"] = os.pathsep.join(filter(None, [
                str(ROOT), str(ROOT / "third_party/frameworks/lmms-eval"), env.get("PYTHONPATH")]))
            command = [sys.executable, "-m", "scripts.analysis.audio_four_probes.worker",
                       "--run-dir", str(output_dir), "--rank", str(rank)]
            try:
                process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
            finally:
                log.close()
            workers.append(process)
            print(f"GPU {gpu}: {shard_dir / 'worker.log'}", flush=True)
        while any(process.poll() is None for process in workers):
            failed = [rank for rank, process in enumerate(workers) if process.poll() not in (None, 0)]
            if failed:
                raise RuntimeError(f"Analysis workers failed: {failed}; inspect their worker.log files")
            time.sleep(1)
        if any(process.returncode != 0 for process in workers):
            raise RuntimeError("An analysis worker failed; inspect shard worker.log files")
        if TAYLOR_PROBE in plan["probes"]:
            print(f"Taylor scores and ranks: {output_dir / 'shard-000/analysis/hierarchical_taylor'}",
                  flush=True)
        else:
            from .report import build_report
            build_report(output_dir)
    finally:
        for process in workers:
            if process.poll() is None:
                process.terminate()
        for process in workers:
            if process.poll() is None:
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
