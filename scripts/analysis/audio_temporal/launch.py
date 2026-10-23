"""Launch one standalone temporal-audio experiment on the MMSU-final task."""
from __future__ import annotations

import argparse
import copy
from dataclasses import replace
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]


def _csv(text, cast):
    return [cast(value.strip()) for value in text.split(",") if value.strip()]


def _default_layers(count):
    points = min(5, count)
    return sorted({round(index * (count - 1) / max(1, points - 1)) for index in range(points)})


def worker_plan(plan, settings_path):
    argv = list(plan.argv)
    matches = [index for index in range(1, len(argv))
               if argv[index - 1] == "-m" and argv[index] == "lmms_eval"]
    if len(matches) != 1:
        raise ValueError("Expected exactly one lmms_eval worker module")
    argv[matches[0]] = "scripts.analysis.audio_temporal.worker"
    environment = dict(plan.environment)
    environment["AUDIO_TEMPORAL_CONFIG"] = str(settings_path)
    existing = environment.get("PYTHONPATH", os.environ.get("PYTHONPATH", ""))
    environment["PYTHONPATH"] = os.pathsep.join(filter(None, [
        str(ROOT), str(ROOT / "third_party/frameworks/lmms-eval"), existing]))
    return replace(plan, argv=argv, environment=environment)


def _load_jsonl(paths):
    rows = []
    for path in paths:
        with path.open(encoding="utf-8") as source:
            rows.extend(json.loads(line) for line in source if line.strip())
    return rows


def _coverage(output, expected, conditions):
    baseline = _load_jsonl(sorted(output.glob("samples.rank*.jsonl")))
    condition_rows = _load_jsonl(sorted(output.glob("conditions.rank*.jsonl")))
    baseline_ids = {int(row["doc_index"]) for row in baseline}
    condition_report = {}
    for condition in conditions:
        ids = {int(row["doc_index"]) for row in condition_rows
               if row["condition_id"] == condition["id"]}
        total = sum(row["condition_id"] == condition["id"] for row in condition_rows)
        condition_report[condition["id"]] = {
            "rows": total, "unique_doc_indices": len(ids), "expected": expected,
            "complete": total == expected and len(ids) == expected}
    complete = len(baseline) == expected and len(baseline_ids) == expected and all(
        item["complete"] for item in condition_report.values())
    return {
        "baseline": {"rows": len(baseline), "unique_doc_indices": len(baseline_ids),
                     "expected": expected,
                     "complete": len(baseline) == expected and len(baseline_ids) == expected},
        "conditions": condition_report,
        "complete": complete,
    }


def main():
    from .plan import EXPERIMENTS, build_conditions

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", required=True, choices=EXPERIMENTS)
    parser.add_argument("--config", default=str(
        ROOT / "scripts/analysis/audio_temporal/eval_mmsufinal.yaml"))
    parser.add_argument("--checkpoint", default=(
        os.environ.get("CHECKPOINT_PATH") or os.environ.get("QWEN25_OMNI_MODEL_PATH")))
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--num-processes", type=int, default=4)
    parser.add_argument("--main-process-port", type=int, default=29659)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--layers", help="Comma-separated encoder layers; default five fixed depths")
    parser.add_argument("--lags-ms", default="40,80,160,320,640,1280,2560")
    parser.add_argument("--region-lag-ms", type=float, default=80.0)
    parser.add_argument("--example-doc-ids", default="0,1,2")
    parser.add_argument("--block-ms", default="40,80,160,320")
    parser.add_argument("--predictors", default="copy_left,linear,local_mean")
    parser.add_argument("--replace-ratios", default="0.1")
    parser.add_argument("--ltbm-keep-ratio", type=float, default=0.5)
    parser.add_argument("--ltbm-window-size", type=int, default=8)
    parser.add_argument("--locality-fractions", default="0.05,0.1,0.2")
    parser.add_argument("--region-types", default="silent,stable,transient,boundary")
    parser.add_argument("--window-ms", type=float, default=320.0)
    parser.add_argument("--far-ms", type=float, default=2000.0)
    args = parser.parse_args()
    if not args.checkpoint:
        parser.error("Provide --checkpoint or CHECKPOINT_PATH")
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")
    if args.num_processes < 1 or args.region_lag_ms <= 0 or args.window_ms <= 0 or args.far_ms <= 0:
        parser.error("Processes and physical time parameters must be positive")

    import yaml
    from mllm_comp.config.loader import load_config
    from mllm_comp.integrations.lmms_eval.adapter import LMMSEvalAdapter
    from mllm_comp.integrations.native_cli import run_native_cli
    from .metrics import jsonable

    checkpoint = Path(args.checkpoint).expanduser().resolve()
    checkpoint_config = json.loads((checkpoint / "config.json").read_text())
    audio_config = checkpoint_config["thinker_config"]["audio_config"]
    encoder_layers = int(audio_config["encoder_layers"])
    layers = (_default_layers(encoder_layers) if args.layers is None
              else sorted(set(_csv(args.layers, int))))
    if not layers or not set(layers).issubset(range(encoder_layers)):
        parser.error("--layers contains an index outside the checkpoint audio encoder")
    lags_ms = _csv(args.lags_ms, float)
    block_ms = _csv(args.block_ms, float)
    replace_ratios = _csv(args.replace_ratios, float)
    locality_fractions = _csv(args.locality_fractions, float)
    if (not lags_ms or not block_ms or not replace_ratios
            or not locality_fractions or any(value <= 0 for value in lags_ms + block_ms)
            or any(not 0 < value < 1 for value in replace_ratios + locality_fractions)):
        parser.error("Invalid lag, block or ratio list")
    if not 0 < args.ltbm_keep_ratio <= 1 or args.ltbm_window_size < 1:
        parser.error("LTBM keep ratio must be in (0, 1] and window size must be positive")
    predictors = _csv(args.predictors, str)
    region_types = _csv(args.region_types, str)
    if not predictors or not set(predictors) <= {"copy_left", "linear", "local_mean"}:
        parser.error("--predictors contains an unknown predictor")
    if not region_types or not set(region_types) <= {"silent", "stable", "transient", "boundary"}:
        parser.error("--region-types contains an unknown input-region proxy")
    if args.far_ms <= args.window_ms:
        parser.error("--far-ms must be greater than --window-ms for a separated locality control")
    example_doc_ids = _csv(args.example_doc_ids, int)
    if any(index < 0 for index in example_doc_ids):
        parser.error("--example-doc-ids must be nonnegative")
    conditions = build_conditions(
        args.experiment,
        block_ms=block_ms,
        predictors=predictors,
        replace_ratios=replace_ratios,
        locality_fractions=locality_fractions,
        region_types=region_types,
        window_ms=args.window_ms,
        far_ms=args.far_ms)

    raw = copy.deepcopy(load_config(args.config).raw)
    if raw["backbone"]["name"] != "qwen25_omni":
        parser.error("Temporal analysis currently targets qwen25_omni only")
    if raw.get("pipeline", {}).get("stages"):
        parser.error("Use a baseline config with pipeline.stages: []")
    if raw["eval"]["tasks"] != ["mmsufinal"]:
        parser.error("This analysis requires eval.tasks: [mmsufinal]")
    output = Path(args.output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=False)
    raw["backbone"]["model_path"] = str(checkpoint)
    raw["backbone"].pop("model_id", None)
    raw["backbone"]["config"]["enable_talker"] = False
    if args.experiment == "provenance":
        stage = yaml.safe_load((Path(__file__).parent / "ltbm_stage.yaml").read_text())
        stage["criterion"]["window_size"] = args.ltbm_window_size
        stage["schedule"]["keep_ratio"] = args.ltbm_keep_ratio
        raw["pipeline"] = {"stages": [stage]}
    else:
        raw["pipeline"] = {"stages": []}
    raw["analysis"] = {}
    raw["eval"].update(
        compression_mode=("compressed" if args.experiment == "provenance" else "none"),
        target=("compressed" if args.experiment == "provenance" else "baseline"),
        runtime_mode="default",
        checkpoint_path=str(checkpoint), runtime_batch_size=1,
        num_processes=args.num_processes, main_process_port=args.main_process_port,
        sample_limit=args.limit)
    eval_args = raw["eval"].setdefault("args", {})
    for field in ("use_cache", "cache_requests", "forward_only_probe"):
        eval_args.pop(field, None)
    if any("cache" in str(value) for value in eval_args.get("native_argv", [])):
        parser.error("Remove cache overrides from eval.args.native_argv")
    eval_args["log_samples"] = True
    raw["output"] = {
        "run_dir": str(output), "manifest_path": str(output / "manifest.json")}
    config_path = output / "eval.resolved.yaml"
    config_path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
    expected = min(args.limit, 5000) if args.limit is not None else 5000
    settings = {
        "experiment": args.experiment,
        "output_dir": str(output),
        "checkpoint": str(checkpoint),
        "config": str(config_path),
        "encoder_layers": encoder_layers,
        "layers": layers,
        "lags_ms": lags_ms,
        "region_lag_ms": args.region_lag_ms,
        "example_doc_ids": example_doc_ids,
        "seed": int(raw["experiment"].get("seed", 42)),
        "expected_samples": expected,
        "conditions": conditions,
        "ltbm": ({"keep_ratio": args.ltbm_keep_ratio,
                  "window_size": args.ltbm_window_size}
                 if args.experiment == "provenance" else None),
        "design": {
            "dataset": "mmsufinal",
            "condition_assignment": "every condition on every baseline document",
            "condition_execution_order": "cyclic_rotation_by_doc_index",
            "scorer": "lmms_eval.tasks.mmsufinal.utils.mmsufinal_process_results",
            "experiment_2_removed": True,
            "no_cross_file_merge": True,
            "no_cross_hard_chunk_merge": args.experiment != "provenance",
            "provenance_source": ("native_ltbm_merge_rounds"
                                  if args.experiment == "provenance" else "native_baseline"),
        },
    }
    settings_path = output / "analysis.json"
    settings_path.write_text(json.dumps(settings, ensure_ascii=False, indent=2), encoding="utf-8")

    def command_runner(plan):
        temporal = worker_plan(plan, settings_path)
        (output / "launch.json").write_text(json.dumps({
            "argv": temporal.argv,
            "cwd": temporal.cwd,
            "experiment": args.experiment,
            "compression_mode": raw["eval"]["compression_mode"],
            "condition_count": len(conditions),
            "expected_generation_calls": expected * (1 + len(conditions)),
        }, indent=2), encoding="utf-8")
        return run_native_cli(temporal)

    adapter = LMMSEvalAdapter(command_runner=command_runner)
    result = adapter.evaluate(config=load_config(config_path), checkpoint_path=str(checkpoint))
    coverage = _coverage(output, expected, conditions)
    (output / "coverage.json").write_text(json.dumps(coverage, indent=2), encoding="utf-8")
    (output / "manifest.json").write_text(json.dumps(jsonable({
        "analysis": settings,
        "coverage": coverage,
        "native_result_path": result.raw_output_path,
        "native_sample_paths": result.sample_output_paths,
        "interpretation": (
            "Native LTBM provenance describes actual token sources but is not a paired "
            "compression-damage comparison. Input regions are log-Mel proxies."
            if args.experiment == "provenance" else
            "Observational statistics are descriptive; only paired substitutions/merges "
            "support causal replaceability claims. Input regions are log-Mel proxies."),
    }), ensure_ascii=False, indent=2), encoding="utf-8")
    if not coverage["complete"]:
        raise RuntimeError(f"Incomplete MMSU temporal run; see {output / 'coverage.json'}")
    print(f"Completed {args.experiment}: {expected} references and "
          f"{expected * len(conditions)} paired conditions.")
    print(f"Plot: {sys.executable} -m scripts.analysis.audio_temporal.report "
          f"--run-dir {output}")


if __name__ == "__main__":
    main()
