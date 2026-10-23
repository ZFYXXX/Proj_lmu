"""Run reproducible one-step compression-aware ImageLLM training evidence.

This is an internal evidence harness, not a public train-gate bypass.  It loads
one real backbone checkpoint, reuses it across methods, restores the selected
trainable parameters after every case, and writes the exact effective stage
configuration and runtime evidence needed to promote SupportUnits later.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import time
from types import SimpleNamespace
from typing import Any, Mapping, Sequence

from mllm_comp.backbones.base import BackboneRuntime
from mllm_comp.compressors.multimodal_registry import (
    builtin_multimodal_compression_contracts,
)
from mllm_comp.core.serialization import atomic_write_text, to_jsonable
from mllm_comp.core.support import SupportStatus
from mllm_comp.core.vocabulary import CompressionMode, ExecutionPhase
from mllm_comp.pipeline.runtime_session import RuntimeCompressionDispatcher
from mllm_comp.pipeline.stage import CompressionStageConfig
from mllm_comp.registry.backbones import get_backbone


ALGORITHMS = (
    "aim",
    "btp",
    "dart",
    "divprune",
    "fastv",
    "feather",
    "finepruner",
    "gap",
    "halfv",
    "ivc_prune",
    "mmtok",
    "nuwa",
    "pact",
    "prumerge",
    "pyramid_drop",
    "restore",
    "saint",
    "sgl",
    "sparsevlm",
    "tome",
    "transprune",
    "visionzip",
    "vispruner",
    "zoo_prune",
)
BACKBONES = ("qwen25_vl", "qwen3_vl", "llava_onevision")
VOLATILE_EVIDENCE_KEYS = {
    "activation_id",
    "event_index",
    "invocation_id",
    "invocation_index",
    "name",
    "request_id",
    "status",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_smoke_record(dataset_path: Path) -> tuple[dict[str, Any], str]:
    with dataset_path.open("r", encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                payload = json.loads(line)
                if not isinstance(payload, dict):
                    raise ValueError("The first non-empty dataset row is not a mapping.")
                return payload, hashlib.sha256(line.encode("utf-8")).hexdigest()
    raise ValueError(f"Dataset {dataset_path} has no non-empty record.")


def _conversation_text(item: Mapping[str, Any]) -> str:
    value = item.get("value", item.get("content", ""))
    if isinstance(value, str):
        return value
    if isinstance(value, Sequence):
        return "\n".join(
            str(part.get("text", "")) if isinstance(part, Mapping) else str(part)
            for part in value
        )
    return str(value)


def _sample_payload(
    backbone: str,
    record: Mapping[str, Any],
    *,
    dataset_path: Path,
) -> tuple[Any, str, Path]:
    conversations = record.get("conversations")
    images = record.get("images")
    if not isinstance(conversations, list) or len(conversations) < 2:
        raise ValueError("Smoke record requires user and assistant conversations.")
    if not isinstance(images, list) or len(images) != 1:
        raise ValueError("Smoke record must contain exactly one image.")
    image_path = Path(str(images[0]))
    if not image_path.is_absolute():
        image_path = dataset_path.parent / image_path
    image_path = image_path.resolve()
    if not image_path.is_file():
        raise FileNotFoundError(image_path)
    prompt = _conversation_text(conversations[0]).replace("<image>", "").strip()
    target = _conversation_text(conversations[1]).strip()
    if not prompt or not target:
        raise ValueError("Smoke record prompt and assistant target must be non-empty.")
    sample_id = str(record.get("id", "imagellm-train-step"))
    if backbone == "qwen25_vl":
        from mllm_comp.backbones.qwen25_vl import Qwen25VLSample
        from mllm_comp.backbones.qwen25_vl.implementation import DEFAULT_SYSTEM_PROMPT

        sample = Qwen25VLSample(
            sample_id=sample_id,
            conversation=_qwen_conversation(image_path, prompt, DEFAULT_SYSTEM_PROMPT),
        )
    elif backbone == "qwen3_vl":
        from mllm_comp.backbones.qwen3_vl import Qwen3VLSample
        from mllm_comp.backbones.qwen3_vl.implementation import DEFAULT_SYSTEM_PROMPT

        sample = Qwen3VLSample(
            sample_id=sample_id,
            conversation=_qwen_conversation(image_path, prompt, DEFAULT_SYSTEM_PROMPT),
            enable_thinking=False,
        )
    elif backbone == "llava_onevision":
        from mllm_comp.backbones.llava_onevision import LlavaOneVisionSample

        sample = LlavaOneVisionSample(
            sample_id=sample_id,
            prompt=prompt,
            images=(str(image_path),),
        )
    else:  # pragma: no cover - argparse rejects this
        raise ValueError(f"Unsupported backbone {backbone!r}.")
    return sample, target, image_path


def _qwen_conversation(
    image_path: Path,
    prompt: str,
    system_prompt: str,
) -> list[dict[str, Any]]:
    return [
        {
            "role": "system",
            "content": [{"type": "text", "text": system_prompt}],
        },
        {
            "role": "user",
            "content": [
                {"type": "image", "image": str(image_path)},
                {"type": "text", "text": prompt},
            ],
        },
    ]


def _active_phase(registry: Any, algorithm: str, backbone: str) -> ExecutionPhase:
    active = [
        unit.identity.phase
        for unit in registry.support.find(algorithm=algorithm, backbone=backbone)
        if unit.status is SupportStatus.ACTIVE
    ]
    if not active:
        raise ValueError(f"{algorithm}/{backbone} has no active inference phase.")
    return ExecutionPhase.FORWARD if ExecutionPhase.FORWARD in active else active[0]


def _stage(
    registry: Any,
    algorithm: str,
    backbone: str,
    *,
    model_path: str,
    device: str,
) -> CompressionStageConfig:
    binding = registry.binding(f"{algorithm}.image.{backbone}")
    variant = registry.variant(binding.variant)
    descriptor = registry.algorithm(algorithm)
    config = dict(variant.config_defaults)
    if algorithm == "tome":
        config.setdefault("r", 1)
        config["output_r"] = 1
    if algorithm == "sgl":
        config["guide_model_path"] = model_path
        config["guide_device"] = device
    return CompressionStageConfig(
        name=f"{algorithm}-image-train-step",
        placement=binding.ordered_targets()[0].placement,
        execution_phase=_active_phase(registry, algorithm, backbone),
        plugin=algorithm,
        method=descriptor.method,
        algorithm=algorithm,
        variant=variant.name,
        binding=binding.name,
        target_modalities=set(variant.target_modalities),
        conditioning_modalities=set(variant.conditioning_modalities),
        preserved_modalities=set(variant.preserved_modalities),
        coupling_policy=variant.coupling_policy,
        media_alignment=variant.media_alignment,
        config=config,
    )


def _move(value: Any, *, device: Any, dtype: Any, torch: Any) -> Any:
    if torch.is_tensor(value):
        if value.is_floating_point():
            return value.to(device=device, dtype=dtype)
        return value.to(device=device)
    if isinstance(value, list):
        return [_move(item, device=device, dtype=dtype, torch=torch) for item in value]
    if isinstance(value, tuple):
        return tuple(
            _move(item, device=device, dtype=dtype, torch=torch) for item in value
        )
    if isinstance(value, dict):
        return {
            name: _move(item, device=device, dtype=dtype, torch=torch)
            for name, item in value.items()
        }
    return value


def _select_trainable_parameters(model: Any) -> list[tuple[str, Any]]:
    named = [(name, parameter) for name, parameter in model.named_parameters()]
    visual = [
        item
        for item in named
        if any(token in item[0].lower() for token in ("visual", "vision", "mm_projector"))
        and item[1].is_floating_point()
        and item[1].ndim >= 1
        and item[1].numel() <= 8_000_000
    ]
    language = [
        item
        for item in named
        if "norm" in item[0].lower()
        and item[1].is_floating_point()
        and 1 <= item[1].numel() <= 1_000_000
    ]
    if not visual or not language:
        raise RuntimeError("Could not resolve compact visual and language trainable parameters.")
    selected = [visual[0], language[-1]]
    selected_ids = {id(parameter) for _, parameter in selected}
    for _, parameter in named:
        parameter.requires_grad_(id(parameter) in selected_ids)
    return selected


def _canonical(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {
            key: _canonical(item)
            for key, item in sorted(value.items())
            if key not in VOLATILE_EVIDENCE_KEYS
        }
    if isinstance(value, (list, tuple)):
        return [_canonical(item) for item in value]
    return value


def _layout_records(session: Any) -> list[dict[str, Any]]:
    metadata = session.runtime_metadata()
    summary = metadata.get("layout_delta_summary", {})
    records = summary.get("records", []) if isinstance(summary, Mapping) else []
    return [dict(item) for item in records if isinstance(item, Mapping)]


def _first_difference(expected: Any, actual: Any, *, path: str = "$.") -> str:
    if isinstance(expected, Mapping) and isinstance(actual, Mapping):
        expected_keys = set(expected)
        actual_keys = set(actual)
        if expected_keys != actual_keys:
            return (
                f"{path} keys differ: expected={sorted(expected_keys)!r}, "
                f"actual={sorted(actual_keys)!r}"
            )
        for key in sorted(expected_keys):
            if expected[key] != actual[key]:
                return _first_difference(
                    expected[key],
                    actual[key],
                    path=f"{path}{key}.",
                )
    elif isinstance(expected, list) and isinstance(actual, list):
        if len(expected) != len(actual):
            return f"{path} length differs: expected={len(expected)}, actual={len(actual)}"
        for index, (expected_item, actual_item) in enumerate(zip(expected, actual)):
            if expected_item != actual_item:
                return _first_difference(
                    expected_item,
                    actual_item,
                    path=f"{path}[{index}].",
                )
    return f"{path} expected={expected!r}, actual={actual!r}"


def _image_token_id(batch: Any) -> int:
    value = batch.metadata.get("token_ids", {}).get("image")
    if value is None:
        raise RuntimeError("Teacher-forced batch omitted its image token ID.")
    return int(value)


def _gradient_stats(selected: Sequence[tuple[str, Any]], torch: Any) -> dict[str, Any]:
    records = []
    for name, parameter in selected:
        gradient = parameter.grad
        records.append(
            {
                "name": name,
                "present": gradient is not None,
                "finite": bool(torch.isfinite(gradient).all().item()) if gradient is not None else False,
                "nonzero": bool((gradient != 0).any().item()) if gradient is not None else False,
                "norm": float(gradient.float().norm().item()) if gradient is not None else 0.0,
            }
        )
    if not any(item["finite"] and item["nonzero"] for item in records):
        raise RuntimeError("No selected trainable parameter received a finite non-zero gradient.")
    return {"parameters": records}


def _run_case(
    *,
    algorithm: str,
    backbone: str,
    facade: Any,
    runtime: BackboneRuntime,
    stage: CompressionStageConfig,
    model_inputs: Mapping[str, Any],
    batch: Any,
    selected: Sequence[tuple[str, Any]],
    registry: Any,
    torch: Any,
) -> dict[str, Any]:
    dispatcher = RuntimeCompressionDispatcher(contract_registry=registry)
    config = SimpleNamespace(pipeline=SimpleNamespace(stages=[stage]))
    session = dispatcher.activate(
        config=config,
        facade=facade,
        runtime=runtime,
        compression_mode=CompressionMode.COMPRESSED,
        execution_phase=stage.execution_phase,
    )
    assert session is not None
    optimizer = torch.optim.SGD([parameter for _, parameter in selected], lr=0.01)
    before = {name: parameter.detach().clone() for name, parameter in selected}
    observations: list[dict[str, Any]] = []
    first_record_count: int | None = None

    def forward(dummy: Any) -> Any:
        nonlocal first_record_count
        sgl = runtime.metadata.get("_sgl_controller")
        if sgl is not None:
            sgl.prepare_teacher_forced(
                model_inputs=model_inputs,
                image_token_id=_image_token_id(batch),
            )
        output = runtime.model(**model_inputs)
        loss = getattr(output, "loss", None)
        logits = getattr(output, "logits", None)
        if loss is None or logits is None:
            raise RuntimeError("Compressed training forward omitted loss or logits.")
        if not bool(torch.isfinite(loss).item()):
            raise RuntimeError("Compressed training loss is non-finite.")
        records = _layout_records(session)
        if not records:
            raise RuntimeError("Compressed training forward emitted no LayoutDelta evidence.")
        if first_record_count is None:
            first_record_count = len(records)
            request_records = records
        else:
            request_records = records[-first_record_count:]
        observations.append(
            {
                "grad_enabled": torch.is_grad_enabled(),
                "loss": float(loss.detach().float().item()),
                "logits_shape": [int(value) for value in logits.shape],
                "layout_records": _canonical(request_records),
            }
        )
        return loss + dummy.sum() * 0

    try:
        optimizer.zero_grad(set_to_none=True)
        dummy = torch.ones((), device=runtime.device, requires_grad=True)
        loss = torch.utils.checkpoint.checkpoint(
            forward,
            dummy,
            use_reentrant=True,
            preserve_rng_state=True,
        )
        loss.backward()
        if len(observations) < 2:
            raise RuntimeError("Gradient checkpointing did not replay the compressed forward.")
        if observations[0]["layout_records"] != observations[-1]["layout_records"]:
            mismatch = _first_difference(
                observations[0]["layout_records"],
                observations[-1]["layout_records"],
            )
            raise RuntimeError(
                "Gradient-checkpoint replay changed the compression layout: " + mismatch
            )
        if observations[0]["logits_shape"] != observations[-1]["logits_shape"]:
            raise RuntimeError("Gradient-checkpoint replay changed the logits layout.")
        gradients = _gradient_stats(selected, torch)
        optimizer.step()
        parameter_deltas = {
            name: float((parameter.detach() - before[name]).abs().max().float().item())
            for name, parameter in selected
        }
        if not any(math.isfinite(value) and value > 0 for value in parameter_deltas.values()):
            raise RuntimeError("Optimizer step did not change a selected parameter.")
        records = observations[0]["layout_records"]
        if not any(
            int(item.get("result_count", 0)) < int(item.get("source_count", 0))
            for item in records
        ):
            raise RuntimeError("The active method did not reduce or merge any real tokens.")
        return {
            "status": "passed",
            "algorithm": algorithm,
            "backbone": backbone,
            "execution_phase": stage.execution_phase.value,
            "use_cache": False,
            "teacher_forcing": True,
            "gradient_checkpoint_replay": True,
            "loss": observations[0]["loss"],
            "input_label_tokens": int(model_inputs["labels"].shape[-1]),
            "output_logit_tokens": observations[0]["logits_shape"][1],
            "layout_records": records,
            "gradients": gradients,
            "optimizer": {"name": "SGD", "lr": 0.01, "max_parameter_delta": parameter_deltas},
            "stage": to_jsonable(stage),
        }
    finally:
        try:
            session.uninstall()
        finally:
            for name, parameter in selected:
                parameter.data.copy_(before[name])
                parameter.grad = None
            optimizer.zero_grad(set_to_none=True)
            torch.cuda.empty_cache()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backbone", choices=BACKBONES, required=True)
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--algorithms", nargs="*", choices=ALGORITHMS)
    args = parser.parse_args(argv)

    import torch
    import torch.utils.checkpoint

    if not torch.cuda.is_available():
        raise RuntimeError("ImageLLM train-step evidence requires CUDA.")
    dataset_path = Path(args.dataset).resolve()
    model_path = Path(args.model_path).resolve()
    output_path = Path(args.output).resolve()
    record, dataset_record_sha256 = _read_smoke_record(dataset_path)
    sample, target, image_path = _sample_payload(
        args.backbone,
        record,
        dataset_path=dataset_path,
    )
    facade = get_backbone(args.backbone)
    processor_kwargs = {"local_files_only": True} if args.backbone != "llava_onevision" else {}
    processor = facade.processor_adapter.load_processor(str(model_path), **processor_kwargs)
    load_kwargs: dict[str, Any] = {
        "torch_dtype": torch.bfloat16,
        "device_map": args.device,
        "attn_implementation": "eager",
    }
    if args.backbone != "llava_onevision":
        load_kwargs["local_files_only"] = True
    model = facade.model_adapter.load_model(str(model_path), train=True, **load_kwargs)
    device = torch.device(args.device)
    runtime = BackboneRuntime(
        model=model,
        processor=processor,
        model_path_or_id=str(model_path),
        device=device,
        dtype=torch.bfloat16,
        metadata={"runtime_mode": "default", "compression_mode": "compressed"},
    )
    prompt_batch = facade.processor_adapter.prepare_inputs(processor, sample, phase="train")
    batch = facade.processor_adapter.prepare_teacher_forced_inputs(
        processor,
        sample,
        target_text=target,
        prompt_batch=prompt_batch,
        phase="train",
    )
    state = facade.state_builder.build(batch, raw_batch=sample, phase="train")
    model_inputs = {
        name: _move(value, device=device, dtype=torch.bfloat16, torch=torch)
        for name, value in state.native_model_inputs.items()
    }
    model_inputs.update({"use_cache": False, "return_dict": True})
    if "labels" not in model_inputs:
        raise RuntimeError("Teacher-forced state omitted labels.")
    selected = _select_trainable_parameters(model)
    registry = builtin_multimodal_compression_contracts()
    algorithms = tuple(args.algorithms or ALGORITHMS)
    results: list[dict[str, Any]] = []
    started = time.time()
    for algorithm in algorithms:
        case_started = time.time()
        stage = _stage(
            registry,
            algorithm,
            args.backbone,
            model_path=str(model_path),
            device=args.device,
        )
        try:
            result = _run_case(
                algorithm=algorithm,
                backbone=args.backbone,
                facade=facade,
                runtime=runtime,
                stage=stage,
                model_inputs=model_inputs,
                batch=batch,
                selected=selected,
                registry=registry,
                torch=torch,
            )
        except BaseException as exc:
            result = {
                "status": "failed",
                "algorithm": algorithm,
                "backbone": args.backbone,
                "execution_phase": stage.execution_phase.value,
                "error": f"{type(exc).__name__}: {exc}",
                "stage": to_jsonable(stage),
            }
        result["elapsed_seconds"] = round(time.time() - case_started, 3)
        results.append(result)
        summary = {
            key: result[key]
            for key in (
                "status",
                "algorithm",
                "backbone",
                "execution_phase",
                "loss",
                "error",
                "elapsed_seconds",
            )
            if key in result
        }
        print(json.dumps(to_jsonable(summary), ensure_ascii=False), flush=True)

    payload = {
        "schema_version": 1,
        "kind": "imagellm_train_step_matrix",
        "backbone": args.backbone,
        "model_path": str(model_path),
        "model_config_sha256": _sha256(model_path / "config.json"),
        "dataset_path": str(dataset_path),
        "dataset_record_id": str(record.get("id")),
        "dataset_record_sha256": dataset_record_sha256,
        "image_path": str(image_path),
        "image_sha256": _sha256(image_path),
        "device": args.device,
        "gpu_name": torch.cuda.get_device_name(device),
        "batch_size": 1,
        "gradient_accumulation_steps": 1,
        "optimizer_steps_per_case": 1,
        "selected_trainable_parameters": [name for name, _ in selected],
        "passed": sum(item["status"] == "passed" for item in results),
        "failed": sum(item["status"] != "passed" for item in results),
        "elapsed_seconds": round(time.time() - started, 3),
        "cases": results,
    }
    atomic_write_text(
        output_path,
        json.dumps(to_jsonable(payload), ensure_ascii=False, indent=2) + "\n",
    )
    return 0 if payload["failed"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
