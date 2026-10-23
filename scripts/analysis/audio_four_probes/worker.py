"""Run the selected probes on one disjoint native-task data shard."""
from __future__ import annotations

import argparse
from collections.abc import Mapping
from copy import deepcopy
import importlib.metadata
import json
import os
from pathlib import Path
from time import perf_counter
from typing import Any

import yaml

from mllm_comp.analysis.definition import AnalysisOverrides, resolve_analysis_definition
from mllm_comp.analysis.records import AnalysisSample
from mllm_comp.analysis.runtime import execute_analysis
from mllm_comp.cli.analyze import _analysis_invariants_passed
from mllm_comp.config.schema import experiment_config_from_dict
from mllm_comp.core.serialization import atomic_write_json, to_jsonable
from mllm_comp.core.vocabulary import RuntimeMode
from mllm_comp.run_manifest import build_run_manifest

from .launch import ROOT, TAYLOR_PROBE


def task_config_overrides(name: str, raw: dict, paths: dict) -> dict:
    """Only relocate dataset paths; native prompts, generation and scorers stay intact."""
    raw = deepcopy(raw)
    kwargs = raw.setdefault("dataset_kwargs", {})
    if name != "mmsufinal":
        raise ValueError(f"Unsupported suite task: {name}")
    if paths.get("mmsu_jsonl"):
        kwargs["data_files"] = {"test": paths["mmsu_jsonl"]}
    return raw


def load_tasks(plan: dict) -> tuple[dict, dict]:
    from lmms_eval.api.task import ConfigurableMessagesTask
    from lmms_eval.utils import load_yaml_config

    paths = plan["data_paths"]
    if paths.get("mmsu_dataset_dir"):
        os.environ["MMSUFINAL_DATASET_DIR"] = paths["mmsu_dataset_dir"]
    task_root = ROOT / "third_party/frameworks/lmms-eval/lmms_eval/tasks"
    tasks, configs = {}, {}
    for name in plan["leaf_tasks"]:
        source = task_root / name / f"{name}.yaml"
        raw = task_config_overrides(name, load_yaml_config(str(source)), paths)
        configs[name] = describe_config(raw)
        tasks[name] = ConfigurableMessagesTask(config=raw, model_name="qwen2_5_omni")
    return tasks, configs


def describe_config(value: Any) -> Any:
    if callable(value):
        return f"{value.__module__}.{value.__qualname__}"
    if isinstance(value, dict):
        return {key: describe_config(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [describe_config(item) for item in value]
    return value


def sample_score(task: Any, doc: dict, prediction: str) -> tuple[float, dict]:
    result = task.process_results(doc, [prediction])
    details = result["accuracy"]
    return float(details["score"]), {"accuracy": details}


def sample_metadata(task_name: str, doc_id: int, doc: dict, *, score: float | None,
                    audio_paths: list[str], duration_seconds: float) -> dict:
    def text(value: Any) -> str:
        if isinstance(value, list):
            return " / ".join(map(str, value))
        return str(value) if value is not None else "unknown"

    # MMSU's Linguistics/Paralinguistics labels are not sound/music/speech labels.
    return {
        "sample_id": f"{task_name}:{doc_id}", "task_name": task_name,
        "doc_id": doc_id, "raw_id": text(doc.get("id")),
        "correctness": "unlabeled" if score is None else ("correct" if score == 1.0 else "incorrect"),
        "accuracy": score, "audio_paths": audio_paths, "duration_seconds": duration_seconds,
        "audio_type": "unknown", "difficulty": text(doc.get("difficulty")),
        "category": text(doc.get("question_type", doc.get("category"))),
        "sub_category": text(doc.get("sub_category", doc.get("sub-category"))),
        "source_task": text(doc.get("task_name", doc.get("task"))),
    }


def audio_information(sample: Any) -> tuple[list[str], float]:
    import soundfile as sf

    paths = [str(item["audio"]) for message in sample.conversation
             for item in message.get("content", []) if item.get("type") == "audio"]
    if not paths:
        raise ValueError(f"No audio in native sample {sample.sample_id}")
    duration = sum(float(sf.info(path).duration) for path in paths)
    return paths, duration


class BenchmarkAdapter:
    """Reuse native chat samples for both full-answer scoring and every analysis."""

    def __init__(self, tasks: dict, *, plan: dict, rank: int, output_dir: Path):
        self.tasks, self.plan, self.rank, self.output_dir = tasks, plan, rank, output_dir
        limit = plan["sample_limit_per_leaf_task"]
        self.inventory = {}
        for name, task in tasks.items():
            total = len(task.eval_docs)
            selected = min(total, limit) if limit is not None else total
            assigned = len(range(rank, selected, plan["shard_count"]))
            self.inventory[name] = {"dataset_rows": total, "selected_rows": selected, "assigned_rows": assigned}
        if not sum(row["assigned_rows"] for row in self.inventory.values()):
            raise ValueError("This worker has no rows; use fewer GPUs or a larger sample limit")

    def dependency_report(self) -> dict:
        return {"versions": {"lmms_eval": "vendored-native-tasks"}}

    def iter_analysis_samples(self, *, config: Any, backbone: Any, runtime: Any,
                              sample_limit: int | None):
        from tqdm import tqdm
        from mllm_comp.integrations.lmms_eval.model_bridge import LMMSEvalRequestBridge

        bridge = LMMSEvalRequestBridge(facade=backbone, runtime=runtime)
        is_taylor = TAYLOR_PROBE in self.plan.get("probes", [])
        total = sum(row["assigned_rows"] for row in self.inventory.values())
        with (self.output_dir / "samples.jsonl").open("x", encoding="utf-8") as output:
            with tqdm(total=total, desc=f"analysis rank {self.rank}", mininterval=10) as progress:
                for task_name, task in self.tasks.items():
                    for doc_id, doc in task.doc_iterator(
                            rank=self.rank, world_size=self.plan["shard_count"], limit=sample_limit):
                        started = perf_counter()
                        sample_id = f"{task_name}:{doc_id}"
                        target_text = None
                        if is_taylor:
                            from mllm_comp.integrations.lmms_eval.adapter import _analysis_target

                            target_text, reason = _analysis_target(task, doc, task_name=task_name)
                            if target_text is None:
                                raise ValueError(f"{sample_id}: hierarchical_taylor requires a reference answer. {reason}")
                        context = task.doc_to_text(doc)
                        sample = bridge.build_chat_sample(
                            raw_messages=task.doc_to_messages(doc), sample_id=sample_id)
                        paths, duration = audio_information(sample)
                        prediction = bridge._generate_sample(
                            sample, request_generation=task.get_config("generation_kwargs"),
                            context=context, task=task_name, split=task.get_config("test_split"),
                            doc_id=doc_id, request_mode="chat", cache_hook=None)
                        score, score_details = sample_score(task, doc, prediction)
                        processed = backbone.processor_adapter.prepare_inputs(
                            runtime.processor, sample, phase="eval")
                        record = AnalysisSample(
                            sample_id=sample_id, task_name=task_name,
                            processor_outputs=dict(processed.model_inputs), raw_doc=doc,
                            metadata={"prompt": context, "prompt_lengths": list(processed.prompt_lengths),
                                      **dict(processed.metadata)})
                        if is_taylor:
                            teacher_forced = backbone.processor_adapter.prepare_teacher_forced_inputs(
                                runtime.processor, sample, target_text=target_text,
                                prompt_batch=processed, phase="eval",
                                runtime_mode=str(runtime.metadata.get("runtime_mode", RuntimeMode.DEFAULT.value)))
                            objective = teacher_forced.metadata.get("teacher_forcing")
                            if not isinstance(objective, Mapping):
                                raise ValueError("Teacher-forced preprocessing did not return objective metadata")
                            record.update(
                                target_text=target_text, target_source="lmms_eval.doc_to_target",
                                teacher_forced_processor_outputs=dict(teacher_forced.model_inputs),
                                objective=dict(objective))
                        # The shared runtime invokes all selected sessions before resuming here.
                        yield record
                        row = sample_metadata(task_name, doc_id, doc, score=score,
                                              audio_paths=paths, duration_seconds=duration)
                        row.update(prediction=prediction, score_details=score_details,
                                   prompt=context, input_protocol="lmms_eval_chat",
                                   system_messages=[message for message in sample.conversation
                                                    if message["role"] == "system"],
                                   elapsed_seconds=perf_counter() - started, rank=self.rank)
                        if is_taylor:
                            row.update(target_text=target_text, target_source=record["target_source"],
                                       objective=record["objective"])
                        output.write(json.dumps(to_jsonable(row), ensure_ascii=False, allow_nan=False) + "\n")
                        output.flush()
                        progress.update(1)
                        progress.set_postfix(task=task_name, doc=doc_id, refresh=False)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--rank", type=int, required=True)
    args = parser.parse_args(argv)
    run_dir = Path(args.run_dir).resolve()
    plan = json.loads((run_dir / "plan.json").read_text())
    if not 0 <= args.rank < plan["shard_count"]:
        parser.error("--rank is outside the planned shard count")
    output_dir = run_dir / f"shard-{args.rank:03d}"
    tasks, task_configs = load_tasks(plan)
    adapter = BenchmarkAdapter(tasks, plan=plan, rank=args.rank, output_dir=output_dir)
    atomic_write_json(output_dir / "tasks.resolved.json", task_configs)
    atomic_write_json(output_dir / "dataset_inventory.json", adapter.inventory)
    raw = yaml.safe_load((run_dir / "analysis.resolved.yaml").read_text())
    raw["output"] = {"run_dir": str(output_dir), "manifest_path": str(output_dir / "manifest.json")}
    config = experiment_config_from_dict(raw)
    definition = resolve_analysis_definition(config, AnalysisOverrides())
    import torch
    import numpy as np
    import random

    random.seed(config.experiment.seed)
    np.random.seed(config.experiment.seed)
    torch.manual_seed(config.experiment.seed)
    torch.cuda.set_device(0)
    versions = {}
    for name in ("torch", "transformers", "datasets", "qwen-omni-utils", "soundfile"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    atomic_write_json(output_dir / "runtime.json", {
        "rank": args.rank, "physical_gpu": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "gpu_name": torch.cuda.get_device_name(0), "versions": versions,
        "decoder_layers": plan["decoder_layers"], "query_heads": plan["query_heads"],
        "input_protocol": "lmms_eval_chat", "config": str(run_dir / "analysis.resolved.yaml"),
    })
    result = execute_analysis(config, definition, device="cuda:0", adapter=adapter)
    expected = sum(row["assigned_rows"] for row in adapter.inventory.values())
    passed = _analysis_invariants_passed(result) and result.sample_count == expected
    manifest = build_run_manifest(
        config, eval_backend=result.backend, analysis_records=result.records,
        checkpoint_path=config.backbone.model_path, invariants_passed=passed,
        invariants_run=["AnalysisSemanticBinding", "AnalysisTokenGroupLayout", "AnalysisHookCleanup"],
        compression_runtime_metadata=result.compression_runtime_metadata)
    atomic_write_json(config.output.manifest_path, manifest)
    if not passed:
        raise RuntimeError("This shard is incomplete or probes skipped samples. Inspect its manifest and analysis artifacts.")
    print(f"Completed rank {args.rank}: {result.sample_count} samples", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
