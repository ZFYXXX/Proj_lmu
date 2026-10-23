"""Shared evidence helpers for exact batch-one video training cells."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shlex
import sys
import time
from collections.abc import Mapping, Sequence
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from mllm_comp.backbones.base import BackboneRuntime
from mllm_comp.compressors.multimodal_registry import builtin_multimodal_compression_contracts
from mllm_comp.core.serialization import atomic_write_text, to_jsonable
from mllm_comp.core.vocabulary import ExecutionPhase
from scripts.diagnostics.video_checkpoint_common import (
    FIXED_NEXTQA_DOC_IDS,
    _gpu_snapshot,
    _load_fixed_documents,
    _validate_admission,
    _video_path,
    add_checkpoint_arguments,
)

FIXTURE_DOC_ID = 442
OPTIMIZER_STEPS = 5
VOLATILE_LAYOUT_KEYS = frozenset(
    {
        "activation_id",
        "event_index",
        "invocation_id",
        "invocation_index",
        "name",
        "request_id",
        "status",
    }
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    atomic_write_text(
        path,
        json.dumps(to_jsonable(payload), ensure_ascii=False, indent=2) + "\n",
    )


def append_jsonl(path: Path, payload: Mapping[str, Any]) -> None:
    """Durably append one diagnostic record before a caller evaluates gates."""

    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (
        json.dumps(to_jsonable(payload), ensure_ascii=False, sort_keys=True) + "\n"
    )
    with path.open("a", encoding="utf-8") as stream:
        stream.write(encoded)
        stream.flush()
        os.fsync(stream.fileno())


def canonical(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {
            key: canonical(item)
            for key, item in sorted(value.items())
            if key not in VOLATILE_LAYOUT_KEYS
        }
    if isinstance(value, (list, tuple)):
        return [canonical(item) for item in value]
    return value


def select_final_norm_weight(facade: Any, model: Any) -> list[tuple[str, Any]]:
    """Freeze the model and select only the facade-owned final LLM norm weight."""

    handles = facade.model_adapter.structural_handles(model)
    final_norm = handles.get("llm.final_norm")
    target = getattr(final_norm, "weight", None)
    if target is None or not target.is_floating_point() or target.ndim != 1:
        raise RuntimeError("Backbone llm.final_norm.weight is unavailable or invalid")
    named = list(model.named_parameters())
    matches = [name for name, parameter in named if parameter is target]
    if len(matches) != 1:
        raise RuntimeError(
            "Backbone llm.final_norm.weight must map to exactly one named parameter"
        )
    for _, parameter in named:
        parameter.requires_grad_(parameter is target)
    return [(matches[0], target)]


def parameter_records(
    selected: Sequence[tuple[str, Any]],
    *,
    before: Mapping[str, Any],
    torch: Any,
) -> list[dict[str, Any]]:
    records = []
    for name, parameter in selected:
        gradient = parameter.grad
        present = gradient is not None
        finite = bool(torch.isfinite(gradient).all().item()) if present else False
        nonzero = bool((gradient != 0).any().item()) if present else False
        norm = float(gradient.detach().float().norm().item()) if present else 0.0
        delta = float((parameter.detach() - before[name]).abs().max().float().item())
        records.append(
            {
                "name": name,
                "facade_handle": "llm.final_norm.weight",
                "numel": int(parameter.numel()),
                "dtype": str(parameter.dtype),
                "gradient_present": present,
                "gradient_finite": finite,
                "gradient_nonzero": nonzero,
                "gradient_norm": norm,
                "max_parameter_delta": delta,
            }
        )
    return records


def target_parameter_checks(records: Sequence[Mapping[str, Any]]) -> dict[str, bool]:
    return {
        "exact_single_intended_target": len(records) == 1,
        "target_gradient_present": bool(
            len(records) == 1 and records[0].get("gradient_present")
        ),
        "target_gradient_finite": bool(
            len(records) == 1 and records[0].get("gradient_finite")
        ),
        "target_gradient_nonzero": bool(
            len(records) == 1 and records[0].get("gradient_nonzero")
        ),
        "target_parameter_updated": bool(
            len(records) == 1
            and math.isfinite(float(records[0].get("max_parameter_delta", 0.0)))
            and float(records[0].get("max_parameter_delta", 0.0)) > 0.0
        ),
    }


def training_step_checks(
    *, loss: Any, parameters: Sequence[Mapping[str, Any]], stable_layout: bool, torch: Any,
) -> dict[str, bool]:
    """Training evidence shared by methods whose layout checks are independent."""
    return {
        "finite_loss": bool(loss is not None and torch.isfinite(loss).all().item()),
        "layout_stable_across_steps": stable_layout,
        **target_parameter_checks(parameters),
    }


def validate_step_records(
    records: Sequence[Mapping[str, Any]],
    *,
    optimizer_steps: int = OPTIMIZER_STEPS,
) -> None:
    if optimizer_steps < 1:
        raise ValueError("optimizer_steps must be positive")
    if len(records) != optimizer_steps:
        raise RuntimeError(
            f"Expected exactly {optimizer_steps} optimizer steps, got {len(records)}"
        )
    for expected_step, record in enumerate(records, start=1):
        if int(record.get("step", -1)) != expected_step:
            raise RuntimeError("Training step indices are not contiguous")
        checks = record.get("checks")
        if not isinstance(checks, Mapping) or not checks or not all(checks.values()):
            raise RuntimeError(
                f"Training step {expected_step} failed exact checks: {checks!r}"
            )


def save_lightweight_checkpoint(
    checkpoint_dir: Path,
    *,
    selected: Sequence[tuple[str, Any]],
    scope: Mapping[str, str],
    canonical_config: Mapping[str, Any],
    base_checkpoint: Path,
    code_revision: str,
    optimizer: Mapping[str, Any],
    optimizer_steps: int = OPTIMIZER_STEPS,
    torch: Any,
) -> dict[str, Any]:
    from safetensors.torch import save_file

    required_scope = {"algorithm", "variant", "backbone", "binding"}
    if set(scope) != required_scope or not all(scope.values()):
        raise RuntimeError("Lightweight checkpoint scope must be exact and complete")
    if optimizer_steps < 1:
        raise ValueError("optimizer_steps must be positive")
    checkpoint_dir.mkdir(parents=True, exist_ok=False)
    tensor_path = checkpoint_dir / "adapter_model.safetensors"
    temporary = checkpoint_dir / ".adapter_model.safetensors.tmp"
    tensors = {
        name: parameter.detach().to(device="cpu").contiguous()
        for name, parameter in selected
    }
    if len(tensors) != 1:
        raise RuntimeError("The frozen lightweight route saves exactly one target")
    save_file(tensors, str(temporary))
    temporary.replace(tensor_path)
    config_path = checkpoint_dir / "adapter_config.json"
    state_path = checkpoint_dir / "training_state.json"
    atomic_json(
        config_path,
        {
            "schema_version": 1,
            "format": "mllm_comp_selected_parameter_checkpoint",
            "base_checkpoint": str(base_checkpoint),
            **dict(scope),
            "trainable_parameter_names": list(tensors),
            "trainable_facade_handle": "llm.final_norm.weight",
            "canonical_config": dict(canonical_config),
            "code_revision": code_revision,
        },
    )
    atomic_json(
        state_path,
        {
            "global_step": optimizer_steps,
            "batch_size": 1,
            "fixture_doc_id": FIXTURE_DOC_ID,
            "optimizer": dict(optimizer),
            "torch_version": torch.__version__,
        },
    )
    return {
        "path": str(checkpoint_dir),
        "format": "mllm_comp_selected_parameter_checkpoint",
        "tensor_path": str(tensor_path),
        "tensor_sha256": sha256(tensor_path),
        "tensor_size_bytes": tensor_path.stat().st_size,
        "config_path": str(config_path),
        "config_sha256": sha256(config_path),
        "training_state_path": str(state_path),
        "training_state_sha256": sha256(state_path),
        "parameter_names": list(tensors),
        "parameter_count": sum(int(parameter.numel()) for _, parameter in selected),
    }


def load_lightweight_checkpoint(
    model: Any,
    checkpoint: Mapping[str, Any],
    *,
    torch: Any,
) -> dict[str, Any]:
    from safetensors.torch import load_file

    path = Path(str(checkpoint["tensor_path"]))
    if sha256(path) != checkpoint["tensor_sha256"]:
        raise RuntimeError("Lightweight checkpoint checksum changed before reload")
    tensors = load_file(str(path), device="cpu")
    parameters = dict(model.named_parameters())
    expected = tuple(str(name) for name in checkpoint["parameter_names"])
    if tuple(sorted(tensors)) != tuple(sorted(expected)):
        raise RuntimeError("Lightweight checkpoint parameter names changed")
    records = []
    with torch.no_grad():
        for name in expected:
            parameter = parameters.get(name)
            if parameter is None:
                raise RuntimeError(f"Fresh model omitted checkpoint tensor {name!r}")
            source = tensors[name].to(device=parameter.device, dtype=parameter.dtype)
            if tuple(source.shape) != tuple(parameter.shape):
                raise RuntimeError(f"Checkpoint tensor shape changed for {name!r}")
            parameter.copy_(source)
            exact = bool(torch.equal(parameter.detach().cpu(), tensors[name]))
            if not exact:
                raise RuntimeError(f"Checkpoint tensor {name!r} did not reload exactly")
            records.append(
                {
                    "name": name,
                    "facade_handle": "llm.final_norm.weight",
                    "shape": [int(item) for item in parameter.shape],
                    "dtype": str(parameter.dtype),
                    "exact_reload": exact,
                }
            )
    return {"parameter_records": records, "all_exact": True}


def training_main(
    argv: list[str] | None = None, *, parser: Any, run: Any,
    scope: Mapping[str, str], gpu_snapshot: Any,
) -> int:
    """Own argument validation, result publication and failure evidence."""
    args = parser.parse_args(argv)
    if not math.isfinite(args.learning_rate) or args.learning_rate <= 0:
        raise RuntimeError("learning_rate must be finite and positive")
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    report_path = output_dir / "report.json"
    started = time.time()
    try:
        report = run(args)
    except BaseException as exc:
        progress_path = output_dir / "progress.jsonl"
        failure: dict[str, Any] = {
            "schema_version": 1,
            "status": "failed",
            "scope": {**scope, "execution_phase": "forward"},
            "error_type": type(exc).__name__,
            "error": str(exc),
            "command": shlex.join(sys.argv),
            "code_revision": args.code_revision,
            "physical_gpu": args.physical_gpu,
            "elapsed_seconds": round(time.time() - started, 3),
        }
        if progress_path.is_file():
            failure["progress_path"] = str(progress_path)
            failure["progress_sha256"] = sha256(progress_path)
        try:
            import torch

            if torch.cuda.is_available() and torch.cuda.device_count() == 1:
                failure["observed_peak_allocated_mib"] = round(
                    torch.cuda.max_memory_allocated() / (1024**2), 3
                )
                failure["observed_peak_reserved_mib"] = round(
                    torch.cuda.max_memory_reserved() / (1024**2), 3
                )
        except Exception:
            pass
        try:
            failure["gpu_after"] = gpu_snapshot(args.physical_gpu)
        except Exception:
            pass
        atomic_json(report_path, failure)
        raise
    atomic_json(report_path, report)
    print(json.dumps({"status": report["status"], "report": str(report_path)}))
    return 0


def training_state(
    *,
    facade: Any,
    processor: Any,
    sample: Any,
    target: str,
) -> tuple[Any, Any]:
    prompt_batch = facade.processor_adapter.prepare_inputs(
        processor, sample, phase="train"
    )
    batch = facade.processor_adapter.prepare_teacher_forced_inputs(
        processor,
        sample,
        target_text=target,
        prompt_batch=prompt_batch,
        phase="train",
    )
    state = facade.state_builder.build(batch, raw_batch=sample, phase="train")
    state = replace(
        state,
        native_model_inputs={
            **state.native_model_inputs,
            "use_cache": False,
            "return_dict": True,
        },
    )
    return batch, state


def load_training_runtime(
    checkpoint: Path, *, facade: Any, torch: Any, train: bool, device_map: Any
) -> tuple[Any, Any, Any, BackboneRuntime]:
    processor = facade.processor_adapter.load_processor(
        str(checkpoint), local_files_only=True
    )
    model = facade.model_adapter.load_model(
        str(checkpoint),
        train=train,
        local_files_only=True,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
        device_map=device_map,
    )
    disable_checkpointing = getattr(model, "gradient_checkpointing_disable", None)
    if callable(disable_checkpointing):
        disable_checkpointing()
    runtime = BackboneRuntime(
        model=model,
        processor=processor,
        model_path_or_id=str(checkpoint),
        device=torch.device("cuda:0"),
        dtype=torch.bfloat16,
        metadata={
            "enable_talker": False,
            "speech_sample_rate": None,
            "runtime_mode": "default",
            "compression_mode": "compressed",
        },
    )
    return facade, processor, model, runtime


def training_forward_step(
    *, facade: Any, runtime: Any, state: Any, optimizer: Any,
    selected: Sequence[tuple[str, Any]], torch: Any,
) -> tuple[Any, Any, list[dict[str, Any]]]:
    """Update the selected weights and retain their gradient/update evidence."""

    optimizer.zero_grad(set_to_none=True)
    before = {name: parameter.detach().clone() for name, parameter in selected}
    output = facade.runner.forward_train(runtime, state)
    loss = getattr(output, "loss", None)
    logits = getattr(output, "logits", None)
    if loss is None or logits is None:
        raise RuntimeError("Compressed training forward omitted loss or logits")
    loss.backward()
    optimizer.step()
    torch.cuda.synchronize()
    return loss, logits, parameter_records(selected, before=before, torch=torch)


def final_norm_optimizer(
    facade: Any, model: Any, *, learning_rate: float, torch: Any,
) -> tuple[list[tuple[str, Any]], Any, dict[str, Any]]:
    selected = select_final_norm_weight(facade, model)
    config = {"name": "AdamW", "lr": learning_rate, "weight_decay": 0.0}
    optimizer = torch.optim.AdamW(
        [parameter for _, parameter in selected], lr=learning_rate, weight_decay=0.0,
    )
    return selected, optimizer, config


def training_document(
    parquet_path: Path, video_root: Path,
    expected_doc_ids: tuple[int, ...] = FIXED_NEXTQA_DOC_IDS,
) -> tuple[dict[str, Any], Path]:
    documents = {int(doc["doc_id"]): doc for doc in _load_fixed_documents(parquet_path)}
    if tuple(documents) != expected_doc_ids:
        raise RuntimeError("The NExT-QA fixed document set changed")
    document = documents[FIXTURE_DOC_ID]
    return document, _video_path(video_root, str(document["video"]))


def save_training_checkpoint(
    args: Any, *, records: Sequence[Mapping[str, Any]], selected: Sequence[tuple[str, Any]],
    scope: Mapping[str, str], canonical_config: Mapping[str, Any],
    optimizer: Mapping[str, Any], optimizer_steps: int, torch: Any,
) -> dict[str, Any]:
    """Save only after all requested optimizer steps have passed their own checks."""
    validate_step_records(records, optimizer_steps=optimizer_steps)
    return save_lightweight_checkpoint(
        args.output_dir.resolve() / f"checkpoint-{optimizer_steps}", selected=selected,
        scope=scope, canonical_config=canonical_config, base_checkpoint=args.checkpoint.resolve(),
        code_revision=args.code_revision, optimizer=optimizer, optimizer_steps=optimizer_steps, torch=torch,
    )


def inference_state(*, facade: Any, processor: Any, sample: Any) -> tuple[Any, Any]:
    """Build the existing cache-free, unlabelled post-reload inference input."""
    batch = facade.processor_adapter.prepare_inputs(processor, sample, phase="eval")
    state = facade.state_builder.build(batch, raw_batch=sample, phase="eval")
    return batch, replace(
        state,
        native_model_inputs={**state.native_model_inputs, "use_cache": False, "return_dict": True},
    )


def inference_forward(*, facade: Any, runtime: Any, state: Any, torch: Any) -> Any:
    runtime.model.eval()
    with torch.inference_mode():
        output = facade.runner.forward_train(runtime, state)
    return getattr(output, "logits", None)


def record_training_step(progress_path: Path, records: list, record: Mapping[str, Any]) -> None:
    """Persist a failed step too; do not evaluate its success summary first."""
    append_jsonl(progress_path, record)
    records.append(record)
    checks = record["checks"]
    if not all(checks.values()):
        raise RuntimeError(f"Training step {record['step']} failed: {checks!r}")


def print_training_step(record: Mapping[str, Any], **summary: Any) -> None:
    print(json.dumps({"step": record["step"], "loss": record["loss"], **summary}, ensure_ascii=False), flush=True)


def record_post_reload(progress_path: Path, report: Mapping[str, Any]) -> None:
    append_jsonl(progress_path, {"event": "post_reload_inference", **report})
    checks = report["checks"]
    if not all(checks.values()):
        raise RuntimeError(f"Post-reload compressed inference failed: {checks!r}")


def checkpoint_saved(
    progress_path: Path, checkpoint: Mapping[str, Any], *, optimizer_steps: int
) -> None:
    append_jsonl(progress_path, {
        "event": "checkpoint_saved", "global_step": optimizer_steps, "checkpoint": checkpoint,
    })


def training_environment_report(*, visible: str, torch: Any, transformers: Any) -> dict[str, Any]:
    return {
        "python": sys.version.split()[0], "torch": torch.__version__,
        "transformers": transformers.__version__, "cuda_visible_devices": visible,
        "logical_device": "cuda:0", "logical_device_count": torch.cuda.device_count(),
        "gpu_name": torch.cuda.get_device_name(0), "dtype": "bfloat16",
        "attention_implementation": "sdpa", "offline": True, "gradient_checkpointing": False,
    }


def training_parameter_report(
    optimizer: Mapping[str, Any], selected_names: Sequence[str], steps: Sequence,
    *, optimizer_steps: int,
) -> dict[str, Any]:
    return {
        "batch_size": 1, "optimizer_steps": optimizer_steps, "gradient_accumulation_steps": 1,
        "optimizer": optimizer, "intended_trainable_handle": "llm.final_norm.weight",
        "selected_trainable_parameters": selected_names, "compressor_trainable_parameters": 0,
        "steps": steps,
    }


def training_report(
    args: Any, *, scope: Mapping[str, Any], canonical_config: Mapping[str, Any],
    checkpoint: Mapping[str, Any], fixture_path: Path, started: float,
    provenance: Mapping[str, Any] | None = None,
    boundaries: Mapping[str, bool] | None = None, **sections: Any,
) -> dict[str, Any]:
    """Publish the common training evidence envelope with method-owned sections."""
    progress_path = args.output_dir.resolve() / "progress.jsonl"
    return {
        "schema_version": 1,
        "status": "passed",
        "scope": {
            **scope, "execution_phase": "forward", "canonical_config": canonical_config,
            "quality_claim": "functional_migration_only",
        },
        "provenance": {
            "code_revision": args.code_revision, "command": shlex.join(sys.argv),
            "checkpoint": checkpoint, "nextqa_parquet": str(args.nextqa_parquet.resolve()),
            "fixture_path": str(fixture_path), "fixture_sha256": sha256(fixture_path),
            "progress_path": str(progress_path), "progress_sha256": sha256(progress_path),
            **(provenance or {}),
        },
        **sections,
        "elapsed_seconds": round(time.time() - started, 3),
        "support_activation_claimed": False,
        "boundaries": {
            "test_derived_code_correctness_fixture": True,
            "full_nextqa_benchmark": False,
            "paper_quality_or_efficiency_reproduction": False,
            "convergence_claim": False,
            "batch_greater_than_one": False,
            **(boundaries or {}),
        },
    }


def projector_layout_record(
    records: Sequence, algorithm_records: Sequence, splice: Any,
) -> dict[str, Any]:
    """Use the same canonical evidence for step stability and the saved report."""
    return canonical({
        "compression": records[0] if len(records) == 1 else None,
        "algorithm": algorithm_records[0] if len(algorithm_records) == 1 else None,
        "splice": splice,
    })


def training_parser(*, estimated_peak_mib: int) -> Any:
    parser = argparse.ArgumentParser()
    add_checkpoint_arguments(parser, estimated_peak_mib=estimated_peak_mib)
    parser.add_argument("--learning_rate", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=42)
    return parser


def llava_training_fixture(
    *, document: Mapping[str, Any], video: Path, output_dir: Path
) -> tuple[dict[str, Any], Path]:
    return video_training_fixture(
        document=document, video=video, output_dir=output_dir,
        preprocessing={"decoded_frames": 32, "decode_policy": "uniform_local_decord"},
    )


def video_training_fixture(
    *, document: Mapping[str, Any], video: Path, output_dir: Path,
    preprocessing: Mapping[str, Any],
) -> tuple[dict[str, Any], Path]:
    fixture = {
        "schema_version": 1,
        "kind": "nextqa_mc_test_derived_sft_code_correctness_fixture",
        "quality_or_training_data_claim": False,
        "task": "nextqa_mc_test",
        "split": "test",
        "doc_id": FIXTURE_DOC_ID,
        "video_id": document["video"],
        "video_path": str(video),
        "prompt": document["prompt"],
        "target": document["target"],
        "video_preprocessing": dict(preprocessing),
    }
    path = output_dir / "fixture.json"
    atomic_json(path, fixture)
    return fixture, path


def training_activation(
    *, facade: Any, model: Any, runtime: BackboneRuntime, stage: Any,
    canonical_config: Mapping[str, Any],
) -> Any:
    """Construct the stage's forward executor and verify the admitted method config."""
    binding = builtin_multimodal_compression_contracts().binding(stage.binding)
    activation = binding.executor_factory(
        model=model, runtime=runtime, facade=facade, stage=stage, binding=binding,
        binding_config=SimpleNamespace(config={}), execution_phase=ExecutionPhase.FORWARD,
    )
    if dict(activation.runtime_metadata()["canonical_config"]) != dict(canonical_config):
        raise RuntimeError(f"Resolved {stage.algorithm} training config differs from canonical")
    return activation


def training_cache_exact(
    cache: Mapping[str, Any], *, source_length: int, result_length: int,
    boundary_layer: int | None = None,
) -> bool:
    """A training-forward record must contain no physical cache mutation."""
    return (
        cache.get("policy") == "none_training_forward"
        and int(cache.get("source_length", -1)) == source_length
        and int(cache.get("result_length", -1)) == result_length
        and tuple(cache.get("touched_layers", ())) == ()
        and "cache_type" not in cache
        and "layers_before" not in cache
        and "layers_after" not in cache
        and (
            boundary_layer is None
            or (cache.get("boundary") == "before_layer"
                and int(cache.get("boundary_layer", -1)) == boundary_layer)
        )
    )


def prepare_training_device(
    args: Any, *, framework_label: str
) -> tuple[Any, Any, dict[str, Any], dict[str, Any]]:
    """Admit and seed the pinned native training runtime once per command."""
    gpu_before = _gpu_snapshot(args.physical_gpu)
    admission = _validate_admission(
        gpu_before,
        estimated_peak_mib=args.estimated_peak_mib,
        safety_margin_mib=args.safety_margin_mib,
    )

    import torch
    import transformers

    if transformers.__version__ != "4.57.3":
        raise RuntimeError(
            f"The pinned {framework_label} bridge requires Transformers 4.57.3, got "
            f"{transformers.__version__}"
        )
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError(
            f"Expected one logical CUDA device, got {torch.cuda.device_count()}"
        )
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.cuda.reset_peak_memory_stats()

    return torch, transformers, gpu_before, admission
