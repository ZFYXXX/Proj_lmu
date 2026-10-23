"""Render one exact ImageLLM eval or compression-aware training config."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import yaml

from mllm_comp.compressors.multimodal_registry import (
    builtin_multimodal_compression_contracts,
)
from mllm_comp.core.serialization import to_jsonable
from mllm_comp.core.support import SupportStatus
from mllm_comp.core.vocabulary import ExecutionPhase
from mllm_comp.pipeline.stage import CompressionStageConfig


BACKBONES = ("qwen25_vl", "qwen3_vl", "llava_onevision")


def _phase(algorithm: str, backbone: str, *, mode: str) -> ExecutionPhase:
    registry = builtin_multimodal_compression_contracts()
    units = [
        unit
        for unit in registry.support.find(algorithm=algorithm, backbone=backbone)
        if unit.status is SupportStatus.ACTIVE
        and (mode != "train" or unit.train_step_verified)
    ]
    if not units:
        if algorithm == "d2pruner":
            raise ValueError(
                "D2Pruner has no verified recipe in this renderer; provide a "
                "calibrated prior in a research config. Exact support units "
                "remain declared."
            )
        raise ValueError(f"{algorithm}/{backbone} has no active {mode} support unit.")
    if mode == "train":
        # Historical teacher-forcing evidence may use generate; Swift runs forward.
        binding = registry.binding(f"{algorithm}.image.{backbone}")
        if ExecutionPhase.FORWARD not in binding.supported_phases:
            raise ValueError(
                f"{algorithm}/{backbone} has no forward binding for ms-swift training."
            )
        return ExecutionPhase.FORWARD
    phases = {unit.identity.phase for unit in units}
    return (
        ExecutionPhase.FORWARD
        if ExecutionPhase.FORWARD in phases
        else next(iter(phases))
    )


def _stage(
    algorithm: str,
    backbone: str,
    *,
    mode: str,
    model_path: str,
) -> dict[str, Any]:
    registry = builtin_multimodal_compression_contracts()
    binding = registry.binding(f"{algorithm}.image.{backbone}")
    variant = registry.variant(binding.variant)
    descriptor = registry.algorithm(algorithm)
    target = binding.ordered_targets()[0]
    config = dict(variant.config_defaults)
    if algorithm == "tome":
        config.update({"r": 1, "output_r": 1})
    if algorithm == "sgl":
        config.update({"guide_model_path": model_path, "guide_device": "cuda:0"})
    stage = CompressionStageConfig(
        name=f"{algorithm}_image_{mode}",
        placement=target.placement,
        execution_phase=_phase(algorithm, backbone, mode=mode),
        plugin=algorithm,
        algorithm=algorithm,
        variant=variant.name,
        binding=binding.name,
        method=descriptor.method,
        object_family="tensor_slot",
        object_subtype=target.name,
        modalities=set(variant.target_modalities),
        conditioning_modalities=set(variant.conditioning_modalities),
        preserved_modalities=set(variant.preserved_modalities),
        coupling_policy=variant.coupling_policy,
        media_alignment=variant.media_alignment,
        config=config,
    )
    return to_jsonable(stage)


def build_config(
    *,
    algorithm: str,
    backbone: str,
    mode: str,
    model_path: str,
    dataset: str,
    output_dir: str,
    eval_task: str = "mme",
) -> dict[str, Any]:
    stage = _stage(algorithm, backbone, mode=mode, model_path=model_path)
    common: dict[str, Any] = {
        "experiment": {
            "name": f"{backbone}_{algorithm}_{mode}",
            "seed": 42,
        },
        "backbone": {
            "name": backbone,
            "model_path": model_path,
            "trust_remote_code": False,
        },
        "pipeline": {"stages": [stage]},
        "analysis": {},
        "data": {
            "dataset_id": dataset,
            "dataset_commit": None,
            "sample_split": "train" if mode == "train" else "test",
        },
        "output": {
            "run_dir": output_dir,
            "manifest_path": f"{output_dir}/manifest.json",
        },
    }
    if mode == "train":
        common.update(
            {
                "train": {
                    "backend": "ms_swift",
                    "compression_mode": "compressed",
                    "max_steps": 1,
                    "args": {
                        "datasets": [dataset],
                        "trainable_branch": "language_model",
                        "tuner_type": "lora",
                        "target_modules": ["all-linear"],
                        "lora_rank": 4,
                        "lora_alpha": 8,
                        "torch_dtype": "bfloat16",
                        "attn_impl": "eager",
                        "per_device_train_batch_size": 1,
                        "gradient_accumulation_steps": 1,
                        "learning_rate": 0.0001,
                        "gradient_checkpointing": True,
                        "max_length": 1024,
                        "freeze_vit": False,
                        "freeze_aligner": False,
                        "split_dataset_ratio": 0,
                        "load_from_cache_file": False,
                        "dataset_num_proc": 1,
                        "dataloader_num_workers": 0,
                        "logging_steps": 1,
                        "save_steps": 1,
                        "save_total_limit": 1,
                        "report_to": ["none"],
                    },
                },
                "eval": {},
            }
        )
        if backbone == "llava_onevision":
            common["train"]["args"].update(
                {
                    "local_repo_path": "third_party/backbones/LLaVA-NeXT",
                    "vit_gradient_checkpointing": False,
                }
            )
    else:
        common.update(
            {
                "train": {"compression_mode": "none"},
                "eval": {
                    "backend": "lmms_eval",
                    "compression_mode": "compressed",
                    "target": "compressed",
                    "checkpoint_path": model_path,
                    "tasks": [eval_task],
                    "runtime_batch_size": 1,
                    "num_processes": 1,
                    "sample_limit": 2,
                    "generation": {},
                    "args": {
                        "force_simple": True,
                        "forward_only_probe": (
                            stage["execution_phase"] == ExecutionPhase.FORWARD.value
                        ),
                        "log_samples": True,
                    },
                },
            }
        )
    return common


def main(argv: list[str] | None = None) -> int:
    registry = builtin_multimodal_compression_contracts()
    algorithms = tuple(sorted(item.name for item in registry.algorithms()))
    parser = argparse.ArgumentParser()
    parser.add_argument("--algorithm", choices=algorithms, required=True)
    parser.add_argument("--backbone", choices=BACKBONES, required=True)
    parser.add_argument("--mode", choices=("eval", "train"), required=True)
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--eval_task", default="mme")
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    config = build_config(
        algorithm=args.algorithm,
        backbone=args.backbone,
        mode=args.mode,
        model_path=args.model_path,
        dataset=args.dataset,
        output_dir=args.output_dir,
        eval_task=args.eval_task,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        yaml.safe_dump(config, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
