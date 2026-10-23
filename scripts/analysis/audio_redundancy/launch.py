"""Launch cross-fitting and A/B/C/E/H evaluation without changing framework files."""
import argparse
from dataclasses import replace
import json
import os
from pathlib import Path
import signal
import subprocess
import sys

from .common import coverage, write_json
from .plan import EXPERIMENTS, build_conditions

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent


def phase_complete(directory, expected, conditions, *, calibration=False):
    """Trust a finished native pass only when every paired row and result exists."""
    directory = Path(directory)
    if not directory.is_dir():
        return False
    checked = coverage(directory, expected, conditions)
    write_json(directory / "coverage.json", checked)
    if not checked["complete"]:
        return False
    if calibration and any(not (directory / "features" / f"doc{i}.npz").is_file()
                           for i in range(expected)):
        return False
    for path in directory.rglob("*_results.json"):
        try:
            result = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if "mmsufinal" in result.get("results", {}):
            return True
    return False


def models_complete(directory, settings):
    directory = Path(directory)
    metadata = directory / "metadata.json"
    if not metadata.is_file():
        return False
    try:
        records = json.loads(metadata.read_text(encoding="utf-8"))["models"]
    except (OSError, ValueError, KeyError, TypeError):
        return False
    expected = {(fold, stage, size) for fold in (0, 1)
                for stage in settings["stages"] for size in settings["codebook_sizes"]}
    actual = {(row["held_out_fold"], row["stage"], row["codebook_size"]) for row in records}
    return actual == expected and all((directory / row["file"]).is_file() for row in records)


def run_stages(settings, arguments):
    """Keep GPU/native libraries, CPU fitting and reporting in separate processes."""
    phases = (["calibration", "fit"] if settings["needs_calibration"] else []) + ["evaluation", "report"]
    root = Path(settings["output_dir"])
    environment = {**os.environ, "PYTHONFAULTHANDLER": "1", "PYTHONUNBUFFERED": "1"}
    for phase in phases:
        print(f"{','.join(settings['experiments'])}: starting {phase} phase", flush=True)
        command = [sys.executable, "-X", "faulthandler", "-m", "scripts.analysis.audio_redundancy.launch",
                   *arguments, "--phase", phase, "--resume"]
        result = subprocess.run(command, env=environment)
        if result.returncode == 0:
            continue
        # A launcher SIGSEGV cannot be caught inside that launcher. Validate its
        # saved outputs here, in the supervisor that never imports the GPU adapter.
        if result.returncode in (-signal.SIGSEGV, 128 + signal.SIGSEGV) and phase in {"calibration", "evaluation"}:
            conditions = [] if phase == "calibration" else settings["conditions"]
            if phase_complete(root / phase, settings["expected_samples"], conditions,
                              calibration=phase == "calibration"):
                write_json(root / phase / "recovered_launcher_exit.json", dict(
                    phase=phase, launcher_returncode=result.returncode,
                    reason="Native result and all per-document outputs complete before launcher SIGSEGV"))
                print(f"{phase}: launcher exited {result.returncode}; complete saved outputs validated", flush=True)
                continue
        raise RuntimeError(f"{phase} process failed (return code {result.returncode}); "
                           f"run is incomplete: {root}")


def csv_values(value, cast=str):
    return list(dict.fromkeys(cast(v.strip()) for v in value.split(",") if v.strip()))


def worker_plan(plan, settings_path, phase):
    argv = list(plan.argv)
    matches = [i for i in range(1, len(argv)) if argv[i - 1:i + 1] == ["-m", "lmms_eval"]]
    if len(matches) != 1:
        raise ValueError("Expected one native lmms_eval worker")
    argv[matches[0]] = "scripts.analysis.audio_redundancy.worker"
    env = dict(plan.environment)
    env.update(AUDIO_REDUNDANCY_CONFIG=str(settings_path), AUDIO_REDUNDANCY_PHASE=phase,
               PYTHONFAULTHANDLER="1", PYTHONUNBUFFERED="1")
    env["PYTHONPATH"] = os.pathsep.join(filter(None, [str(ROOT), str(ROOT / "third_party/frameworks/lmms-eval"),
        env.get("PYTHONPATH", os.environ.get("PYTHONPATH", ""))]))
    return replace(plan, argv=argv, environment=env)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiments", default="all", help="all, or comma-separated A,B,C,E,H")
    parser.add_argument("--checkpoint", default=os.environ.get("CHECKPOINT_PATH"))
    parser.add_argument("--mmsu-jsonl", default=os.environ.get("MMSU_JSONL"))
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--num-processes", type=int, default=4)
    parser.add_argument("--main-process-port", type=int, default=29679)
    parser.add_argument("--limit", type=int, help="Smoke subset only; omitted = complete source JSONL")
    parser.add_argument("--layers", help="Comma-separated indices, or all; default five fixed encoder depths")
    parser.add_argument("--intervention-stages", default="P_in", help="A: captured E_lN/E_L/P_in/P_out")
    parser.add_argument("--ranks", default="16,32,64,128,256")
    parser.add_argument("--windows-ms", default="500,1000,2000")
    parser.add_argument("--keep-ratios", default="0.75,0.5,0.25,0.1")
    parser.add_argument("--codebook-sizes", default="256")
    parser.add_argument("--codebook-pca-dim", type=int, default=32)
    parser.add_argument("--calibration-tokens", type=int, default=32)
    parser.add_argument("--fit-max-tokens", type=int, default=8192)
    parser.add_argument("--markov-alpha", type=float, default=.1)
    parser.add_argument("--fit-threads", type=int, default=4)
    parser.add_argument("--worker-threads", type=int, default=1)
    parser.add_argument("--mel-ranks", default="4,8,16,32,64")
    parser.add_argument("--example-doc-ids", default="0,1,2")
    parser.add_argument("--attention-backend", choices=("flash_attention_2", "eager", "sdpa"), default="flash_attention_2")
    parser.add_argument("--bootstrap-repeats", type=int, default=1000)
    parser.add_argument("--tolerance-pp", type=float, default=1.)
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--resume", action="store_true", help="Continue an existing run after a completed phase")
    parser.add_argument("--phase", "--codebook-phase", "--e-phase", dest="phase",
                        choices=("calibration", "fit", "evaluation", "report"),
                        help="Run one stage; by default all required stages run in separate processes")
    args = parser.parse_args()
    if not args.checkpoint:
        parser.error("Provide CHECKPOINT_PATH or --checkpoint")
    checkpoint = Path(args.checkpoint).expanduser().resolve()
    configuration = json.loads((checkpoint / "config.json").read_text())
    if "thinker_config" not in configuration:
        parser.error("A complete full/merged Qwen2.5-Omni checkpoint with thinker_config is required")
    audio = configuration["thinker_config"]["audio_config"]
    depth = int(audio["encoder_layers"])
    experiments = list(EXPERIMENTS) if args.experiments == "all" else csv_values(args.experiments)
    if not experiments or not set(experiments) <= set(EXPERIMENTS):
        parser.error("--experiments must select A,B,C,E,H")
    if args.phase in {"calibration", "fit"} and not set(experiments) & {"B", "E"}:
        parser.error("Calibration and fitting require experiment B or E")
    layers = (list(range(depth)) if args.layers == "all" else csv_values(args.layers, int) if args.layers
              else sorted({round(i * (depth - 1) / max(1, min(5, depth) - 1)) for i in range(min(5, depth))}))
    if not layers or not set(layers) <= set(range(depth)):
        parser.error("Layer indices are outside the audio encoder")
    layers.sort()
    stages = ["E_L" if i == depth - 1 else f"E_l{i}" for i in layers] + ["P_in", "P_out"]
    intervention_stages = csv_values(args.intervention_stages)
    if not intervention_stages or not set(intervention_stages) <= set(stages):
        parser.error("A intervention stage must be among captured stages: " + ",".join(stages))
    ranks, sizes, mel_ranks = (csv_values(value, int) for value in (args.ranks, args.codebook_sizes, args.mel_ranks))
    ratios, windows = csv_values(args.keep_ratios, float), csv_values(args.windows_ms, float)
    if (not all((ranks, sizes, mel_ranks, ratios, windows)) or min(ranks + sizes + mel_ranks) < 1
            or max(mel_ranks) > 128 or any(not 0 < r <= 1 for r in ratios) or min(windows) <= 0):
        parser.error("Invalid rank/codebook/window/budget list")
    if (min(args.num_processes, args.fit_threads, args.worker_threads, args.codebook_pca_dim,
            args.calibration_tokens, args.fit_max_tokens, args.bootstrap_repeats) < 1
            or args.markov_alpha <= 0 or args.tolerance_pp < 0
            or (args.limit is not None and args.limit < args.num_processes)):
        parser.error("Invalid process/calibration/report settings; limit must cover every worker")
    import yaml
    if args.mmsu_jsonl:
        source = Path(args.mmsu_jsonl).expanduser().resolve()
    else:
        # Read just the data path; leave YAML !function tags to native lmms-eval.
        class TaskLoader(yaml.SafeLoader):
            pass
        TaskLoader.add_constructor("!function", lambda loader, node: loader.construct_scalar(node))
        task = yaml.load((ROOT / "third_party/frameworks/lmms-eval/lmms_eval/tasks/mmsufinal/mmsufinal.yaml").read_text(), Loader=TaskLoader)
        source = Path(task["dataset_kwargs"]["data_files"]["test"]).expanduser().resolve()
    documents = [json.loads(line) for line in source.read_text(encoding="utf-8").splitlines() if line.strip()]
    source_count = len(documents)
    documents = documents[:args.limit] if args.limit else documents
    if len(documents) < args.num_processes:
        parser.error("Dataset has fewer rows than workers")
    if len({r["id"] for r in documents}) != len(documents):
        parser.error("Duplicate MMSU IDs")
    needs_calibration = bool(set(experiments) & {"B", "E"})
    from .calibration import assign_folds
    paths = [r["audio_path"] for r in documents]
    folds = assign_folds(paths) if needs_calibration else None
    conditions, membership = build_conditions(experiments, ranks=ranks, stages=intervention_stages,
        ratios=ratios, codebook_sizes=sizes, mel_ranks=mel_ranks)
    output = Path(args.output_dir).expanduser().resolve()
    settings = dict(experiments=experiments, output_dir=str(output), checkpoint=str(checkpoint),
        mmsu_jsonl=str(source), source_samples=source_count, expected_samples=len(documents),
        full_source=len(documents) == source_count, full_mmsufinal=len(documents) == source_count == 5000,
        document_ids=[r["id"] for r in documents], audio_paths=paths, folds=folds,
        encoder_layers=depth, layers=layers, stages=stages, intervention_stages=intervention_stages,
        needs_calibration=needs_calibration, codebook_sizes=sizes, windows_ms=windows,
        codebook_pca_dim=args.codebook_pca_dim, calibration_tokens=args.calibration_tokens,
        fit_max_tokens=args.fit_max_tokens, markov_alpha=args.markov_alpha, fit_threads=args.fit_threads,
        worker_threads=args.worker_threads, conditions=conditions, experiment_conditions=membership,
        example_doc_ids=csv_values(args.example_doc_ids, int), seed=42,
        bootstrap_repeats=args.bootstrap_repeats, tolerance_pp=args.tolerance_pp,
        generation_calls=len(documents) * (1 + len(conditions) + int(needs_calibration)),
        design=dict(task="mmsufinal", scorer="native mmsufinal_process_results",
                    calibration="two-fold audio-disjoint unsupervised cross-fitting",
                    A="per-audio PCA reconstruction, not a shared channel bottleneck",
                    B="first-order predictive entropy/surprisal of contextual hidden units, not causal signal entropy",
                    CE="contiguous P_in compression, native hard boundaries retained, compact positions",
                    H="mean plus low-rank residual in native log-Mel domain; same shape",
                    timing="diagnostic only; includes analysis overhead and reference preparation; not standalone speedup"))
    brief = {key: settings[key] for key in ("experiments", "expected_samples", "full_mmsufinal", "layers",
             "stages", "needs_calibration", "generation_calls", "experiment_conditions")}
    brief.update(num_processes=args.num_processes, unique_conditions=len(conditions), output_dir=str(output))
    if args.phase:
        brief["phase"] = args.phase
    if args.plan_only:
        print(json.dumps(brief, ensure_ascii=False, indent=2))
        return
    settings_path = output / "analysis.json"
    if args.resume:
        if not settings_path.is_file():
            parser.error(f"Cannot resume without {settings_path}")
        if json.loads(settings_path.read_text(encoding="utf-8")) != settings:
            parser.error("Resume arguments differ from the frozen analysis.json plan")
    else:
        output.mkdir(parents=True, exist_ok=False)
        write_json(settings_path, settings)
    print(json.dumps(brief, ensure_ascii=False, indent=2), flush=True)
    if args.phase is None:
        run_stages(settings, sys.argv[1:])
        return

    def run_phase(phase, port):
        directory = output / phase
        phase_conditions = [] if phase == "calibration" else conditions
        completed = phase_complete(directory, len(documents), phase_conditions,
                                   calibration=phase == "calibration")
        if completed:
            print(f"Skipping complete {phase} phase: {directory}", flush=True)
            return
        if directory.exists():
            raise RuntimeError(f"Existing {phase} phase is incomplete: {directory / 'coverage.json'}. "
                               "Use a fresh output directory for this phase.")
        from mllm_comp.config.loader import load_config
        from mllm_comp.integrations.lmms_eval.adapter import LMMSEvalAdapter
        from mllm_comp.integrations.native_cli import NativeCLIProcessError, run_native_cli

        directory.mkdir()
        if phase == "calibration":
            (directory / "features").mkdir()
        raw = yaml.safe_load((HERE / "eval_mmsufinal.yaml").read_text())
        raw["backbone"]["model_path"] = str(checkpoint)
        raw["backbone"]["config"]["attn_implementation"] = args.attention_backend
        raw["eval"].update(checkpoint_path=str(checkpoint), num_processes=args.num_processes,
                           main_process_port=port, sample_limit=args.limit)
        raw["output"] = dict(run_dir=str(directory), manifest_path=str(directory / "manifest.json"))
        config_path = directory / "eval.resolved.yaml"
        config_path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")

        def runner(plan):
            active = worker_plan(plan, settings_path, phase)
            write_json(directory / "launch.json", dict(argv=active.argv, cwd=active.cwd, phase=phase))
            return run_native_cli(active)

        try:
            result = LMMSEvalAdapter(command_runner=runner).evaluate(
                config=load_config(config_path), checkpoint_path=str(checkpoint))
        except NativeCLIProcessError as exc:
            if not phase_complete(directory, len(documents), phase_conditions,
                                  calibration=phase == "calibration"):
                raise
            write_json(directory / "recovered_exit.json", dict(
                phase=phase, worker_returncode=exc.returncode,
                reason="Native scorer and all per-document outputs complete before worker exit"))
            print(f"Worker exited {exc.returncode} after complete {phase} outputs; continuing.", flush=True)
            return
        if not phase_complete(directory, len(documents), phase_conditions,
                              calibration=phase == "calibration"):
            raise RuntimeError(f"Incomplete or duplicate coverage/native result: {directory / 'coverage.json'}")
        return result

    if args.phase == "calibration":
        run_phase("calibration", args.main_process_port)
        return
    if needs_calibration:
        if not phase_complete(output / "calibration", len(documents), [], calibration=True):
            raise RuntimeError("B/E calibration is incomplete; fitting/evaluation cannot start")
        from .calibration import fit_models
        model_dir = output / "models"
        if models_complete(model_dir, settings):
            print(f"Skipping complete models: {model_dir}", flush=True)
        elif args.phase != "fit":
            raise RuntimeError(f"B/E codebooks are incomplete: {model_dir}")
        else:
            print(f"Fitting codebooks: {model_dir}", flush=True)
            fit_models(settings, resume_incomplete=True)
        if args.phase == "fit":
            return
    if args.phase == "evaluation":
        run_phase("evaluation", args.main_process_port + int(needs_calibration))
        return
    if not phase_complete(output / "evaluation", len(documents), conditions):
        raise RuntimeError("Evaluation is incomplete; report cannot start")
    from .report import report
    report(output)
    from .export import export_portable
    export_portable(output)


if __name__ == "__main__":
    main()
