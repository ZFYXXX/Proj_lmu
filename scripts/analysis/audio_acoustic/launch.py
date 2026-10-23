"""Launch one independently selected acoustic experiment on native MMSU-final."""
import argparse
import copy
from dataclasses import replace
import json
import os
from pathlib import Path
import sys

from .plan import EXPERIMENTS, build_conditions, build_suite_conditions

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent


def csv_values(value, cast=float):
    return [cast(item.strip()) for item in value.split(",") if item.strip()]


def worker_plan(plan, settings):
    argv = list(plan.argv)
    matches = [i for i in range(1, len(argv)) if argv[i - 1:i + 1] == ["-m", "lmms_eval"]]
    if len(matches) != 1:
        raise ValueError("Expected one native lmms_eval worker module")
    argv[matches[0]] = "scripts.analysis.audio_acoustic.worker"
    environment = dict(plan.environment)
    environment["AUDIO_ACOUSTIC_CONFIG"] = str(settings)
    environment["PYTHONPATH"] = os.pathsep.join(filter(None, [str(ROOT),
        str(ROOT / "third_party/frameworks/lmms-eval"),
        environment.get("PYTHONPATH", os.environ.get("PYTHONPATH", ""))]))
    return replace(plan, argv=argv, environment=environment)


def iter_rows(paths):
    for path in paths:
        with path.open(encoding="utf-8") as source:
            for line in source:
                if line.strip():
                    yield json.loads(line)


def read_rows(paths):
    return list(iter_rows(paths))


def coverage(output, expected, conditions):
    baseline_ids = [row["doc_index"] for row in iter_rows(sorted(output.glob("samples.rank*.jsonl")))]
    baseline_set = set(baseline_ids)
    complete = len(baseline_ids) == len(baseline_set) == expected and baseline_set == set(range(expected))
    report = dict(baseline=dict(rows=len(baseline_ids), unique=len(baseline_set), expected=expected, complete=complete),
                  conditions={}, complete=complete)
    known = {c["id"] for c in conditions}
    ids_by_condition = {name: [] for name in known}
    unexpected = set()
    for row in iter_rows(sorted(output.glob("conditions.rank*.jsonl"))):
        if row["condition_id"] in known:
            ids_by_condition[row["condition_id"]].append(row["doc_index"])
        else:
            unexpected.add(row["condition_id"])
    report["unexpected_conditions"] = sorted(unexpected)
    for condition in conditions:
        ids = ids_by_condition[condition["id"]]
        okay = len(ids) == len(set(ids)) == expected and set(ids) == baseline_set
        report["conditions"][condition["id"]] = dict(rows=len(ids), complete=okay)
    report["complete"] &= not unexpected and all(v["complete"] for v in report["conditions"].values())
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", required=True, choices=(*EXPERIMENTS, "all"))
    parser.add_argument("--checkpoint", default=os.environ.get("CHECKPOINT_PATH"))
    parser.add_argument("--config", default=str(HERE / "eval_mmsufinal.yaml"))
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--num-processes", type=int, default=4)
    parser.add_argument("--main-process-port", type=int, default=29669)
    parser.add_argument("--limit", type=int, help="Smoke test only; omit for all 5,000 MMSU questions")
    parser.add_argument("--layers", help="Comma-separated audio encoder layers, or all; default five fixed depths")
    parser.add_argument("--cutoffs-hz", default="6000,4000,3000,2000,1000")
    parser.add_argument("--transition-hz", type=float, default=200.0,
                        help="Smooth filter transition width; rerun 100/400 Hz to diagnose filtering artifacts")
    parser.add_argument("--example-doc-ids", default="0,1,2")
    parser.add_argument("--mmsu-jsonl", default=os.environ.get("MMSU_JSONL"))
    parser.add_argument("--plan-only", action="store_true", help="Print the plan without loading weights or creating output")
    args = parser.parse_args()
    if not args.checkpoint:
        parser.error("Set CHECKPOINT_PATH or --checkpoint to a full Qwen2.5-Omni checkpoint")
    if args.num_processes < 1 or (args.limit is not None and args.limit < args.num_processes):
        parser.error("Use positive process count and limit >= num-processes")
    checkpoint = Path(args.checkpoint).expanduser().resolve()
    audio_config = json.loads((checkpoint / "config.json").read_text())["thinker_config"]["audio_config"]
    depth = int(audio_config["encoder_layers"])
    layers = (list(range(depth)) if args.layers == "all" else csv_values(args.layers, int) if args.layers
              else sorted({round(i * (depth - 1) / max(1, min(5, depth) - 1)) for i in range(min(5, depth))}))
    if not layers or not set(layers) <= set(range(depth)):
        parser.error("Selected layers are outside this checkpoint's audio encoder")
    cutoffs = csv_values(args.cutoffs_hz)
    if not cutoffs:
        parser.error("Cutoff list must be nonempty")
    examples = csv_values(args.example_doc_ids, int)
    if any(value < 0 for value in examples):
        parser.error("Example document IDs must be nonnegative")
    options = dict(cutoffs=cutoffs)
    if args.experiment == "all":
        conditions, experiments = build_suite_conditions(**options)
    else:
        conditions = build_conditions(args.experiment, **options)
        experiments = {}
    if not 0 < args.transition_hz <= 500:
        parser.error("--transition-hz must be in (0, 500]")
    for condition in conditions:
        if condition["family"] == "wave":
            condition["transition_hz"] = args.transition_hz
    for members in experiments.values():
        for condition in members:
            if condition["family"] == "wave":
                condition["transition_hz"] = args.transition_hz
    output = Path(args.output_dir).expanduser().resolve()
    settings = dict(experiment=args.experiment, output_dir=str(output), checkpoint=str(checkpoint),
                    encoder_layers=depth, layers=sorted(set(layers)),
                    example_doc_ids=examples, conditions=conditions, experiment_conditions=experiments,
                    expected_samples=min(args.limit, 5000) if args.limit is not None else 5000,
                    full_benchmark=args.limit is None or args.limit >= 5000,
                    mmsu_jsonl=str(Path(args.mmsu_jsonl).expanduser().resolve()) if args.mmsu_jsonl else None,
                    seed=42, sampling_rate=16000,
                    design=dict(condition_assignment="every condition on every document",
                                baseline="native lmms-eval prompt/generation/scorer",
                                interaction="joint deletion damage minus individual damages; positive can mean backup redundancy",
                                spectral_axes="W: acoustic Hz; hidden: time modulation only",
                                acoustic_attributes="original-waveform ACF/RMS/low-activity proxies; no trained readout",
                                attribute_groups="input-only unique-audio terciles; ties stay together; unavailable separate",
                                token_layout="all interventions preserve native encoder and projector token counts",
                                timing="diagnostic only; hooks and repeated decoding are included"))
    settings["expected_generation_calls"] = settings["expected_samples"] * (len(conditions) + 1)
    if experiments:
        settings["independent_generation_calls"] = settings["expected_samples"] * sum(
            len(rows) + 1 for rows in experiments.values())
        settings["generation_calls_saved"] = settings["independent_generation_calls"] - settings["expected_generation_calls"]
    if args.plan_only:
        print(json.dumps(settings, ensure_ascii=False, indent=2))
        return
    if settings["mmsu_jsonl"] and not Path(settings["mmsu_jsonl"]).is_file():
        parser.error("--mmsu-jsonl / MMSU_JSONL does not point to a file")

    import yaml
    from mllm_comp.config.loader import load_config
    from mllm_comp.integrations.lmms_eval.adapter import LMMSEvalAdapter
    from mllm_comp.integrations.native_cli import run_native_cli
    from scripts.analysis.audio_temporal.metrics import jsonable

    raw = copy.deepcopy(load_config(args.config).raw)
    if raw["backbone"]["name"] != "qwen25_omni" or raw["eval"]["tasks"] != ["mmsufinal"]:
        parser.error("This suite requires qwen25_omni and only mmsufinal")
    if raw.get("pipeline", {}).get("stages") or raw.get("analysis"):
        parser.error("Use an uncompressed baseline with pipeline.stages: [] and analysis: {}")
    raw["backbone"]["model_path"] = str(checkpoint)
    raw["backbone"].pop("model_id", None)
    raw["backbone"]["config"]["enable_talker"] = False
    if int(raw["backbone"]["config"].get("sampling_rate", 16000)) != 16000:
        parser.error("The frozen acoustic frequency grid requires 16 kHz input")
    raw["eval"].update(compression_mode="none", target="baseline", runtime_mode="default",
                       checkpoint_path=str(checkpoint), runtime_batch_size=1,
                       num_processes=args.num_processes, main_process_port=args.main_process_port,
                       sample_limit=args.limit)
    native_args = raw["eval"].setdefault("args", {})
    for field in ("use_cache", "cache_requests", "forward_only_probe"):
        native_args.pop(field, None)
    if native_args.get("native_argv"):
        parser.error("Remove eval.args.native_argv; it can override the paired coverage contract")
    native_args["log_samples"] = True
    raw["output"] = dict(run_dir=str(output), manifest_path=str(output / "manifest.json"))
    settings["seed"] = int(raw["experiment"].get("seed", 42))
    output.mkdir(parents=True, exist_ok=False)
    config_path = output / "eval.resolved.yaml"
    config_path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
    settings_path = output / "analysis.json"
    settings_path.write_text(json.dumps(settings, ensure_ascii=False, indent=2), encoding="utf-8")

    def command_runner(plan):
        wrapped = worker_plan(plan, settings_path)
        (output / "launch.json").write_text(json.dumps(dict(argv=wrapped.argv, cwd=wrapped.cwd,
            expected_generation_calls=settings["expected_generation_calls"]), indent=2), encoding="utf-8")
        return run_native_cli(wrapped)

    result = LMMSEvalAdapter(command_runner=command_runner).evaluate(
        config=load_config(config_path), checkpoint_path=str(checkpoint))
    checked = coverage(output, settings["expected_samples"], conditions)
    (output / "coverage.json").write_text(json.dumps(checked, indent=2), encoding="utf-8")
    (output / "manifest.json").write_text(json.dumps(jsonable(dict(analysis=settings, coverage=checked,
        native_result_path=result.raw_output_path, native_sample_paths=result.sample_output_paths)),
        ensure_ascii=False, indent=2), encoding="utf-8")
    if not checked["complete"]:
        raise RuntimeError(f"Incomplete paired MMSU run: {output / 'coverage.json'}")
    print(f"Completed {args.experiment}: {settings['expected_samples']} questions, {len(conditions)} conditions.")
    print(f"Report: {sys.executable} -m scripts.analysis.audio_acoustic.report --run-dir {output}")


if __name__ == "__main__":
    main()
