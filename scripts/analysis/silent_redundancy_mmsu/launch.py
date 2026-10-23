"""Launch one MMSU-Final observation or real encoder-pruning condition."""
from __future__ import annotations

import argparse
import copy
from dataclasses import replace
import json
import os
from pathlib import Path

from . import ACOUSTIC_FEATURES, CAUSAL_SELECTORS, DEFAULT_LAYERS
from .metrics import jsonable

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent


def _worker_plan(plan, settings_path: Path, mode: str):
    argv = list(plan.argv)
    matches = [index for index in range(1, len(argv))
               if argv[index - 1] == "-m" and argv[index] == "lmms_eval"]
    if len(matches) != 1:
        raise ValueError("Expected exactly one lmms_eval worker module")
    module = ("scripts.analysis.silent_redundancy_mmsu.observe_worker"
              if mode == "observe"
              else "scripts.analysis.silent_redundancy_mmsu.prune_worker")
    argv[matches[0]] = module
    environment = dict(plan.environment)
    environment["SILENT_REDUNDANCY_CONFIG"] = str(settings_path)
    existing = environment.get("PYTHONPATH", os.environ.get("PYTHONPATH", ""))
    environment["PYTHONPATH"] = os.pathsep.join(filter(None, [
        str(ROOT), str(ROOT / "third_party/frameworks/lmms-eval"), existing,
    ]))
    return replace(plan, argv=argv, environment=environment)


def _stage(keep_ratio: float, layers: list[int], minimum_keep_tokens: int):
    return {
        "name": "silent_redundancy_encoder_topk",
        "placement": "encoder_level",
        "execution_phase": "generate",
        "plugin": "encoder_attention_topk_audio",
        "algorithm": "encoder_attention_topk_audio",
        "variant": "encoder_attention_topk_audio.audio",
        "binding": "encoder_attention_topk_audio.audio.qwen25_omni",
        "method": "token_compression",
        "object_family": "tensor_slot",
        "object_subtype": "audio_encoder_layer_post_mhsa_pre_ffn",
        "modalities": ["audio"],
        "target_modalities": ["audio"],
        "conditioning_modalities": [],
        "preserved_modalities": ["text", "image", "video"],
        "coupling_policy": "independent",
        "media_alignment": "temporal",
        "token_roles": ["observation"],
        "operation": {"type": "keep", "target_site": "audio_encoder_layer_post_mhsa_pre_ffn"},
        "criterion": {"score_source": "mean_received_self_attention"},
        "schedule": {"keep_ratio": keep_ratio, "minimum_keep_tokens": minimum_keep_tokens},
        "activation_scope": {"layers": layers},
        "enabled": True,
    }


def _load_records(output: Path):
    rows = []
    for path in sorted(output.glob("samples.rank*.jsonl")):
        with path.open(encoding="utf-8") as stream:
            rows.extend(json.loads(line) for line in stream if line.strip())
    return sorted(rows, key=lambda row: row["doc_index"])


def _coverage(rows, expected: int):
    ids = [row["id"] for row in rows]
    docs = [row["doc_index"] for row in rows]
    complete = (len(rows) == expected and len(set(ids)) == expected
                and set(docs) == set(range(expected)))
    return {
        "expected_samples": expected,
        "observed_samples": len(rows),
        "unique_ids": len(set(ids)),
        "missing_doc_indices": sorted(set(range(expected)) - set(docs)),
        "complete": complete,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="mode", required=True)
    for name in ("observe", "prune"):
        command = sub.add_parser(name)
        command.add_argument("--checkpoint", default=os.environ.get("CHECKPOINT_PATH"))
        command.add_argument("--output-dir", required=True)
        command.add_argument("--config", default=str(HERE / "eval_mmsufinal.yaml"))
        command.add_argument("--num-processes", type=int, required=True)
        command.add_argument("--main-process-port", type=int, required=True)
        command.add_argument("--limit", type=int)
        command.add_argument("--layers", default=",".join(map(str, DEFAULT_LAYERS)))
        command.add_argument("--example-doc-ids", default="0,833,1666,2499,3332,4165")
        command.add_argument("--mmsu-jsonl", default=os.environ.get("MMSU_JSONL"))
        command.add_argument("--query-block", type=int, default=64)
    prune = sub.choices["prune"]
    prune.add_argument("--selector", required=True,
                       choices=["attention", "random", *ACOUSTIC_FEATURES, *CAUSAL_SELECTORS])
    prune.add_argument("--target-ratio", type=float, required=True)
    prune.add_argument("--keep-ratio-per-stage", type=float)
    prune.add_argument("--minimum-keep-tokens", type=int, default=2)
    args = parser.parse_args()

    if not args.checkpoint:
        parser.error("Set CHECKPOINT_PATH or pass --checkpoint")
    if args.limit is not None and args.limit < args.num_processes:
        parser.error("--limit must be at least --num-processes for distributed smoke tests")
    if args.query_block < 1:
        parser.error("--query-block must be positive")
    layers = [int(value) for value in args.layers.split(",") if value.strip()]
    examples = [int(value) for value in args.example_doc_ids.split(",") if value.strip()]
    if len(set(layers)) != len(layers) or any(layer < 0 for layer in layers):
        parser.error("--layers must contain unique non-negative indices")

    import yaml
    from mllm_comp.config.loader import load_config
    from mllm_comp.integrations.lmms_eval.adapter import LMMSEvalAdapter
    from mllm_comp.integrations.native_cli import run_native_cli

    raw = copy.deepcopy(load_config(args.config).raw)
    if raw["backbone"]["name"] != "qwen25_omni" or raw["eval"]["tasks"] != ["mmsufinal"]:
        parser.error("This analysis requires qwen25_omni and eval.tasks: [mmsufinal]")
    output = Path(args.output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=False)
    checkpoint = str(Path(args.checkpoint).expanduser().resolve())
    raw["backbone"]["model_path"] = checkpoint
    raw["backbone"].pop("model_id", None)
    raw["backbone"]["config"]["enable_talker"] = False
    raw["eval"].update(
        checkpoint_path=checkpoint,
        runtime_batch_size=1,
        num_processes=args.num_processes,
        main_process_port=args.main_process_port,
        sample_limit=args.limit,
    )
    native_args = raw["eval"].setdefault("args", {})
    for field in ("use_cache", "cache_requests", "forward_only_probe"):
        native_args.pop(field, None)
    native_args["log_samples"] = True
    raw["output"] = {"run_dir": str(output), "manifest_path": str(output / "manifest.json")}
    settings = {
        "mode": args.mode,
        "output_dir": str(output),
        "checkpoint": checkpoint,
        "layers": layers,
        "example_doc_ids": examples,
        "query_block": args.query_block,
        "seed": int(raw["experiment"].get("seed", 42)),
        "expected_samples": min(args.limit, 5000) if args.limit is not None else 5000,
        "mmsu_jsonl": args.mmsu_jsonl,
    }
    if args.mode == "observe":
        raw["pipeline"] = {"stages": []}
        raw["eval"].update(compression_mode="none", target="baseline")
    else:
        keep_ratio = args.keep_ratio_per_stage
        if keep_ratio is None:
            keep_ratio = args.target_ratio ** (1 / len(layers))
        if not 0 < args.target_ratio <= 1 or not 0 < keep_ratio <= 1:
            parser.error("Ratios must be in (0, 1]")
        if args.selector in CAUSAL_SELECTORS and len(layers) != 1:
            parser.error("Causal deletion uses exactly one intervention layer")
        raw["pipeline"] = {"stages": [_stage(keep_ratio, layers, args.minimum_keep_tokens)]}
        raw["eval"].update(compression_mode="compressed", target="compressed")
        settings.update(
            selector=args.selector,
            target_ratio=args.target_ratio,
            keep_ratio_per_stage=keep_ratio,
            minimum_keep_tokens=args.minimum_keep_tokens,
        )
    config_path = output / "eval.resolved.yaml"
    config_path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
    settings_path = output / "analysis.json"
    settings_path.write_text(json.dumps(settings, ensure_ascii=False, indent=2), encoding="utf-8")

    def command_runner(plan):
        wrapped = _worker_plan(plan, settings_path, args.mode)
        (output / "launch.json").write_text(json.dumps({
            "argv": wrapped.argv,
            "cwd": wrapped.cwd,
            "mode": args.mode,
            "selector": settings.get("selector"),
        }, ensure_ascii=False, indent=2), encoding="utf-8")
        return run_native_cli(wrapped)

    result = LMMSEvalAdapter(command_runner=command_runner).evaluate(
        config=load_config(config_path), checkpoint_path=checkpoint)
    rows = _load_records(output)
    coverage = _coverage(rows, settings["expected_samples"])
    (output / "coverage.json").write_text(json.dumps(coverage, indent=2), encoding="utf-8")
    manifest = {
        "analysis": settings,
        "coverage": coverage,
        "native_result_path": result.raw_output_path,
        "native_sample_paths": result.sample_output_paths,
    }
    (output / "manifest.json").write_text(
        json.dumps(jsonable(manifest), ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    if not coverage["complete"]:
        raise RuntimeError(f"Incomplete MMSU-Final run; see {output / 'coverage.json'}")
    realized = None
    if args.mode == "prune":
        realized = sum(row["pruning"]["final_tokens"] for row in rows) / sum(
            row["pruning"]["initial_tokens"] for row in rows)
    print(json.dumps({"completed": args.mode, "samples": len(rows), "realized_ratio": realized}))


if __name__ == "__main__":
    main()
