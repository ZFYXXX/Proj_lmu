#!/usr/bin/env python3
"""Single-seed Stage-2 training for the Qwen2.5-VL feature-anchor projector.

This is intentionally a small, resumable research pilot.  The base checkpoint,
vision tower and LLM stay frozen; only the registered compression projector is
optimized with answer-token cross entropy.  No teacher logits, hidden states or
Oracle records enter the loss.
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict, dataclass
from io import BytesIO
import json
import math
from pathlib import Path
import random
import sys
import time
from types import SimpleNamespace
from typing import Any, Iterable

from PIL import Image

from mllm_comp.backbones.base import BackboneRuntime
from mllm_comp.backbones.qwen25_vl import (
    DEFAULT_SYSTEM_PROMPT,
    Qwen25VLSample,
    create_qwen25_vl_backbone,
)
from mllm_comp.compressors.image.stage2_projector import (
    load_stage2_projector,
    save_stage2_projector,
)
from mllm_comp.compressors.multimodal_registry import (
    builtin_multimodal_compression_contracts,
)
from mllm_comp.core.vocabulary import ExecutionPhase, Modality


@dataclass(frozen=True)
class TrainingExample:
    sample_id: str
    source: str
    image_bytes: bytes
    image_name: str
    question: str
    target: str


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument(
        "--model-revision",
        default="1b5a7674d40a0ac9acef541a46fed12d710bd36b",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--variant",
        choices=("mean_only", "meta", "dispersion", "meta_residual"),
        required=True,
    )
    parser.add_argument("--target-tokens", type=int, default=64)
    parser.add_argument("--bottleneck-dim", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--samples-per-source", type=int, default=256)
    parser.add_argument("--gradient-accumulation", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--max-optimizer-steps", type=int, default=None)
    parser.add_argument("--save-steps", type=int, default=16)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--min-pixels", type=int, default=200704)
    parser.add_argument("--max-pixels", type=int, default=1605632)
    parser.add_argument(
        "--scienceqa",
        default="/home/YinPeiran/datasets/lmms_eval_datasets/ScienceQA/ScienceQA-IMG/train-00000-of-00001.parquet",
    )
    parser.add_argument(
        "--textvqa",
        default="/home/YinPeiran/datasets/lmms_eval_datasets/textvqa/data/train-00000-of-00020.parquet",
    )
    return parser.parse_args()


def _rows(path: str, limit: int) -> Iterable[dict[str, Any]]:
    import pyarrow.parquet as pq

    # ScienceQA-IMG stores encoded images in a large parquet file.  Stream only
    # the requested pilot prefix instead of materialising the full split.
    remaining = limit
    parquet = pq.ParquetFile(path)
    for batch in parquet.iter_batches(batch_size=min(64, max(1, remaining))):
        rows = batch.slice(0, remaining).to_pylist()
        yield from rows
        remaining -= len(rows)
        if remaining <= 0:
            break


def _image_bytes(value: Any) -> tuple[bytes, str]:
    if not isinstance(value, dict) or not value.get("bytes"):
        raise ValueError("pilot rows must carry embedded image bytes")
    return bytes(value["bytes"]), str(value.get("path") or "image")


def _scienceqa_examples(path: str, limit: int) -> list[TrainingExample]:
    result = []
    for index, row in enumerate(_rows(path, limit)):
        image, name = _image_bytes(row["image"])
        choices = [str(item) for item in row["choices"]]
        letters = [chr(ord("A") + choice_index) for choice_index in range(len(choices))]
        choice_text = "\n".join(
            f"{letter}. {choice}" for letter, choice in zip(letters, choices)
        )
        hint = str(row.get("hint") or "").strip()
        question = str(row["question"]).strip()
        prompt = (
            (f"Context: {hint}\n" if hint else "")
            + f"Question: {question}\nChoices:\n{choice_text}"
            + "\nAnswer with the option's letter from the given choices directly."
        )
        answer = int(row["answer"])
        result.append(
            TrainingExample(
                sample_id=f"scienceqa:{index}",
                source="scienceqa",
                image_bytes=image,
                image_name=name,
                question=prompt,
                target=letters[answer],
            )
        )
    return result


def _textvqa_examples(path: str, limit: int) -> list[TrainingExample]:
    result = []
    for index, row in enumerate(_rows(path, limit)):
        image, name = _image_bytes(row["image"])
        answers = [str(item).strip() for item in row.get("answers", ()) if str(item).strip()]
        if not answers:
            continue
        target = Counter(answers).most_common(1)[0][0]
        result.append(
            TrainingExample(
                sample_id=f"textvqa:{row.get('question_id', index)}",
                source="textvqa",
                image_bytes=image,
                image_name=name,
                question=(
                    str(row["question"]).strip()
                    + "\nAnswer the question with a single word or short phrase."
                ),
                target=target,
            )
        )
    return result[:limit]


def _sample(example: TrainingExample) -> Qwen25VLSample:
    image = Image.open(BytesIO(example.image_bytes)).convert("RGB")
    conversation = [
        {
            "role": "system",
            "content": [{"type": "text", "text": DEFAULT_SYSTEM_PROMPT}],
        },
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": example.question},
            ],
        },
    ]
    return Qwen25VLSample(example.sample_id, conversation)


def _move_inputs(values: dict[str, Any], *, device: Any, torch: Any) -> dict[str, Any]:
    result = {}
    for name, value in values.items():
        if not hasattr(value, "to"):
            result[name] = value
            continue
        value = value.to(device)
        if value.is_floating_point():
            value = value.to(dtype=torch.bfloat16)
        result[name] = value
    result["use_cache"] = False
    return result


def _activation(model: Any, runtime: Any, args: argparse.Namespace) -> Any:
    registry = builtin_multimodal_compression_contracts()
    binding = registry.binding("feature_anchor_mean.image.qwen25_vl")
    stage = SimpleNamespace(
        name=f"stage2-feature-anchor-{args.variant}",
        algorithm="feature_anchor_mean",
        variant="feature_anchor_mean.image",
        binding="feature_anchor_mean.image.qwen25_vl",
        target_modalities={Modality.IMAGE},
        operation={"type": "merge", "target_site": "post_vision_merger_pre_llm_scatter"},
        criterion={"score_source": "normalized_visual_feature_cosine"},
        schedule={"granularity": "per_image", "selector": {"modality": "image"}},
        config={
            "target_count": args.target_tokens,
            "stage2_variant": args.variant,
            "bottleneck_dim": args.bottleneck_dim,
            "trainable_projector": True,
        },
    )
    return binding.executor_factory(
        model=model,
        runtime=runtime,
        facade=None,
        stage=stage,
        binding=binding,
        binding_config=SimpleNamespace(config={}),
        execution_phase=ExecutionPhase.FORWARD,
    )


def _save(
    output_dir: Path,
    *,
    module: Any,
    optimizer: Any,
    scheduler: Any,
    optimizer_step: int,
    micro_step: int,
    torch: Any,
) -> None:
    save_stage2_projector(module, output_dir / "projector.safetensors")
    torch.save(
        {
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "optimizer_step": optimizer_step,
            "micro_step": micro_step,
            "torch_rng_state": torch.get_rng_state(),
            "cuda_rng_state": torch.cuda.get_rng_state_all(),
        },
        output_dir / "training_state.pt",
    )


def main() -> int:
    args = _arguments()
    import torch

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    examples = _scienceqa_examples(args.scienceqa, args.samples_per_source)
    examples += _textvqa_examples(args.textvqa, args.samples_per_source)
    random.Random(args.seed).shuffle(examples)
    sample_manifest = [
        {"sample_id": item.sample_id, "source": item.source, "target": item.target}
        for item in examples
    ]
    (output_dir / "samples.json").write_text(
        json.dumps(sample_manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    facade = create_qwen25_vl_backbone()
    processor = facade.processor_adapter.load_processor(
        args.model,
        revision=args.model_revision,
        local_files_only=True,
        use_fast=False,
        min_pixels=args.min_pixels,
        max_pixels=args.max_pixels,
    )
    model = facade.model_adapter.load_model(
        args.model,
        train=True,
        revision=args.model_revision,
        local_files_only=True,
        torch_dtype=torch.bfloat16,
        attn_implementation="eager",
    ).to(args.device)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    runtime = BackboneRuntime(
        model=model,
        processor=processor,
        model_path_or_id=args.model,
        device=torch.device(args.device),
        dtype=torch.bfloat16,
        metadata={"runtime_mode": "default", "compression_mode": "compressed"},
    )
    activation = _activation(model, runtime, args)
    activation.install()
    module = model.mllm_comp_feature_anchor_projector
    trainable = [parameter for parameter in module.parameters() if parameter.requires_grad]
    if not trainable:
        raise RuntimeError("Stage-2 projector exposed no trainable parameters")
    optimizer = torch.optim.AdamW(
        trainable, lr=args.learning_rate, weight_decay=args.weight_decay
    )
    natural_steps = math.ceil(len(examples) / args.gradient_accumulation)
    total_steps = args.max_optimizer_steps or natural_steps
    warmup_steps = max(1, round(total_steps * args.warmup_ratio))

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return float(step + 1) / float(warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    optimizer_step = 0
    micro_step = 0
    state_path = output_dir / "training_state.pt"
    if args.resume and state_path.is_file():
        load_stage2_projector(module, output_dir / "projector.safetensors")
        state = torch.load(state_path, map_location="cpu", weights_only=False)
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        optimizer_step = int(state["optimizer_step"])
        micro_step = int(state["micro_step"])
        torch.set_rng_state(state["torch_rng_state"])
        torch.cuda.set_rng_state_all(state["cuda_rng_state"])
    manifest = {
        "schema_version": 1,
        "status": "running",
        "model": str(Path(args.model).resolve()),
        "model_revision": args.model_revision,
        "variant": args.variant,
        "target_tokens": args.target_tokens,
        "bottleneck_dim": args.bottleneck_dim,
        "seed": args.seed,
        "num_examples": len(examples),
        "gradient_accumulation": args.gradient_accumulation,
        "planned_optimizer_steps": total_steps,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "loss": "answer_token_cross_entropy_only",
        "teacher": False,
        "processor": {
            "use_fast": False,
            "min_pixels": args.min_pixels,
            "max_pixels": args.max_pixels,
        },
        "data": {
            "scienceqa": str(Path(args.scienceqa).resolve()),
            "textvqa": str(Path(args.textvqa).resolve()),
            "samples_per_source": args.samples_per_source,
            "selection": "deterministic_split_prefix_then_seeded_shuffle",
        },
        "software": {
            "torch": torch.__version__,
            "transformers": __import__("transformers").__version__,
        },
        "gpu": (
            torch.cuda.get_device_name(torch.device(args.device))
            if torch.cuda.is_available()
            else None
        ),
        "frozen": ["vision_encoder", "native_merger", "llm"],
        "trainable_parameters": sum(parameter.numel() for parameter in trainable),
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    log_path = output_dir / "training.jsonl"
    optimizer.zero_grad(set_to_none=True)
    started = time.monotonic()
    accumulated_loss = 0.0
    try:
        while optimizer_step < total_steps:
            index = micro_step % len(examples)
            example = examples[index]
            sample = _sample(example)
            prompt = facade.processor_adapter.prepare_inputs(
                processor, sample, phase="train"
            )
            batch = facade.processor_adapter.prepare_teacher_forced_inputs(
                processor,
                sample,
                target_text=example.target,
                prompt_batch=prompt,
                phase="train",
            )
            inputs = _move_inputs(batch.model_inputs, device=args.device, torch=torch)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                output = model(**inputs)
                loss = output.loss / args.gradient_accumulation
            loss.backward()
            accumulated_loss += float(
                loss.detach().float().item() * args.gradient_accumulation
            )
            micro_step += 1
            if micro_step % args.gradient_accumulation:
                continue
            gradient_norm = torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            optimizer_step += 1
            record = {
                "optimizer_step": optimizer_step,
                "micro_step": micro_step,
                "sample_id": example.sample_id,
                "loss": accumulated_loss / args.gradient_accumulation,
                "gradient_norm": float(gradient_norm.detach().float().item()),
                "learning_rate": float(scheduler.get_last_lr()[0]),
                "elapsed_seconds": time.monotonic() - started,
            }
            with log_path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(record, ensure_ascii=False) + "\n")
            print(json.dumps(record), flush=True)
            accumulated_loss = 0.0
            if optimizer_step % args.save_steps == 0 or optimizer_step == total_steps:
                _save(
                    output_dir,
                    module=module,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    optimizer_step=optimizer_step,
                    micro_step=micro_step,
                    torch=torch,
                )
        manifest.update(
            {
                "status": "succeeded",
                "completed_optimizer_steps": optimizer_step,
                "completed_micro_steps": micro_step,
                "elapsed_seconds": time.monotonic() - started,
                "completed_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            }
        )
        (output_dir / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    except BaseException as exc:
        manifest.update(
            {
                "status": "failed",
                "error": f"{type(exc).__name__}: {exc}",
                "completed_optimizer_steps": optimizer_step,
                "completed_micro_steps": micro_step,
            }
        )
        (output_dir / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        raise
    finally:
        activation.uninstall()
    return 0


if __name__ == "__main__":
    sys.exit(main())

