"""Shared fixed NExT-QA fixtures, GPU admission and inference evidence I/O."""

from __future__ import annotations

import argparse
import csv
import json
import os
import shlex
import subprocess
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

FIXED_NEXTQA_DOC_IDS = (3849, 442, 28, 205)

FIXED_NEXTQA_EXPECTATIONS: Mapping[int, Mapping[str, Any]] = {
    3849: {"video": "3571539122", "qid": 6, "type": "TC", "answer": 4},
    442: {"video": "3064389881", "qid": 2, "type": "TN", "answer": 3},
    28: {"video": "6097011028", "qid": 8, "type": "CW", "answer": 1},
    205: {"video": "2597214137", "qid": 6, "type": "DL", "answer": 0},
}

NEXTQA_OPTIONS = ("A", "B", "C", "D", "E")

EXPECTED_NEXTQA_MC_ROWS = 8564

CANONICAL_VQTOKEN_CONFIG = {
    "num_clusters": 32,
    "adaptive": False,
    "use_cross_attention": False,
}


def add_checkpoint_arguments(
    parser: argparse.ArgumentParser, *, estimated_peak_mib: int
) -> None:
    """Arguments shared by offline video generation and training diagnostics."""
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--nextqa_parquet", type=Path, required=True)
    parser.add_argument("--video_root", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--code_revision", required=True)
    parser.add_argument("--physical_gpu", type=int, choices=(0, 1), required=True)
    parser.add_argument("--estimated_peak_mib", type=int, default=estimated_peak_mib)
    parser.add_argument("--safety_margin_mib", type=int, default=16384)


def _shape(value: Any) -> list[int] | None:
    shape = getattr(value, "shape", None)
    return None if shape is None else [int(item) for item in shape]


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    temporary.replace(path)


def _atomic_write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        "".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows
        ),
        encoding="utf-8",
    )
    temporary.replace(path)


def _run_checked(command: Sequence[str]) -> str:
    completed = subprocess.run(
        list(command),
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    return completed.stdout.strip()


def _csv_rows(text: str) -> list[list[str]]:
    return [
        [column.strip() for column in row]
        for row in csv.reader(text.splitlines())
        if row
    ]


def _gpu_snapshot(physical_gpu: int) -> dict[str, Any]:
    gpu_rows = _csv_rows(
        _run_checked(
            (
                "nvidia-smi",
                "--query-gpu=index,name,uuid,memory.total,memory.used,memory.free",
                "--format=csv,noheader,nounits",
            )
        )
    )
    selected = next(
        (row for row in gpu_rows if int(row[0]) == physical_gpu),
        None,
    )
    if selected is None or len(selected) != 6:
        raise RuntimeError(
            f"Physical GPU {physical_gpu} was not reported by nvidia-smi"
        )
    uuid = selected[2]
    process_rows = _csv_rows(
        _run_checked(
            (
                "nvidia-smi",
                "--query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory",
                "--format=csv,noheader,nounits",
            )
        )
    )
    processes = [
        {
            "pid": int(row[1]),
            "process_name": row[2],
            "used_memory_mib": int(row[3]),
        }
        for row in process_rows
        if len(row) == 4 and row[0] == uuid
    ]
    return {
        "physical_device": physical_gpu,
        "name": selected[1],
        "uuid": uuid,
        "total_memory_mib": int(selected[3]),
        "used_memory_mib": int(selected[4]),
        "free_memory_mib": int(selected[5]),
        "compute_processes": processes,
    }


def _validate_admission(
    snapshot: Mapping[str, Any],
    *,
    estimated_peak_mib: int,
    safety_margin_mib: int,
) -> dict[str, Any]:
    required_free_mib = estimated_peak_mib + safety_margin_mib
    free_mib = int(snapshot["free_memory_mib"])
    if free_mib <= required_free_mib:
        raise RuntimeError(
            "GPU admission failed: "
            f"free={free_mib} MiB must exceed estimate+margin="
            f"{required_free_mib} MiB"
        )
    processes = list(snapshot.get("compute_processes", ()))
    return {
        "estimated_peak_mib": estimated_peak_mib,
        "safety_margin_mib": safety_margin_mib,
        "required_free_mib": required_free_mib,
        "observed_free_mib": free_mib,
        "decision": (
            "admitted_with_existing_compute_processes"
            if processes
            else "admitted_without_existing_compute_processes"
        ),
        "concurrency": len(processes),
    }


def _nextqa_prompt(document: Mapping[str, Any]) -> str:
    lines = [str(document["question"]).strip()]
    for index, option in enumerate(NEXTQA_OPTIONS):
        lines.append(f"{option}. {str(document[f'a{index}']).strip()}")
    return "\n".join(lines)


def _normalise_fixed_document(
    document: Mapping[str, Any], *, doc_id: int
) -> dict[str, Any]:
    expected = FIXED_NEXTQA_EXPECTATIONS[doc_id]
    normalised = dict(document)
    observed = {
        "video": str(normalised["video"]),
        "qid": int(normalised["qid"]),
        "type": str(normalised["type"]),
        "answer": int(normalised["answer"]),
    }
    if observed != expected:
        raise RuntimeError(
            f"NExT-QA doc {doc_id} identity changed: "
            f"expected={dict(expected)!r}, observed={observed!r}"
        )
    normalised.update(observed)
    normalised["doc_id"] = doc_id
    normalised["target"] = NEXTQA_OPTIONS[observed["answer"]]
    normalised["prompt"] = _nextqa_prompt(normalised)
    return normalised


def _load_fixed_documents(parquet_path: Path) -> list[dict[str, Any]]:
    import pyarrow.parquet as parquet

    columns = ["video", "qid", "type", "question", "answer"] + [
        f"a{index}" for index in range(5)
    ]
    table = parquet.read_table(parquet_path, columns=columns)
    if table.num_rows != EXPECTED_NEXTQA_MC_ROWS:
        raise RuntimeError(
            "NExT-QA MC test row count changed: "
            f"expected={EXPECTED_NEXTQA_MC_ROWS}, observed={table.num_rows}"
        )
    documents = []
    for doc_id in FIXED_NEXTQA_DOC_IDS:
        raw = {name: table.column(name)[doc_id].as_py() for name in columns}
        documents.append(_normalise_fixed_document(raw, doc_id=doc_id))
    return documents


def _video_path(video_root: Path, video_id: str) -> Path:
    direct = video_root / f"{video_id}.mp4"
    if direct.is_file():
        return direct
    matches = tuple(video_root.rglob(f"{video_id}.mp4"))
    if len(matches) != 1:
        raise RuntimeError(
            f"Expected exactly one local video {video_id}.mp4 below {video_root}, "
            f"found {len(matches)}"
        )
    return matches[0]


def _parse_choice(text: str) -> str | None:
    padded = " " + text.strip() + " "
    candidates = []
    for option in NEXTQA_OPTIONS:
        if (
            f"({option})" in padded
            or f" {option} " in padded
            or f" {option}." in padded
        ):
            candidates.append((padded.rfind(option), option))
    return None if not candidates else max(candidates)[1]


def _tensor_rows(value: Any) -> list[list[int]]:
    detach = getattr(value, "detach", None)
    if callable(detach):
        value = detach().cpu().tolist()
    elif hasattr(value, "tolist"):
        value = value.tolist()
    if not isinstance(value, list):
        raise RuntimeError("Generated token IDs are not row-like")
    if value and not isinstance(value[0], list):
        value = [value]
    return [[int(token) for token in row] for row in value]


class _DecoderInputAudit:
    def __init__(self, layer: Any) -> None:
        self.records: list[dict[str, Any]] = []

        def hook(_module: Any, args: tuple[Any, ...], kwargs: Mapping[str, Any]):
            hidden_states = kwargs.get("hidden_states")
            if hidden_states is None and args:
                hidden_states = args[0]
            hidden_shape = _shape(hidden_states)
            self.records.append(
                {
                    "hidden_shape": hidden_shape,
                    "query_length": (
                        None if hidden_shape is None else int(hidden_shape[-2])
                    ),
                    "attention_mask_shape": _shape(kwargs.get("attention_mask")),
                    "position_ids_shape": _shape(kwargs.get("position_ids")),
                    "cache_position_shape": _shape(kwargs.get("cache_position")),
                    "has_past_key_value": kwargs.get("past_key_value") is not None,
                }
            )

        self._handle = layer.register_forward_pre_hook(hook, with_kwargs=True)

    def mark(self) -> int:
        return len(self.records)

    def since(self, mark: int) -> list[dict[str, Any]]:
        return self.records[mark:]

    def close(self) -> None:
        self._handle.remove()


def _decoder_summary(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not records:
        raise RuntimeError("The first language-model layer was never invoked")
    prefill = max(records, key=lambda item: int(item.get("query_length") or 0))
    query_length = int(prefill.get("query_length") or 0)
    if query_length <= 1:
        raise RuntimeError(
            f"No multi-token decoder prefill was observed: {list(records)!r}"
        )
    attention_shape = prefill.get("attention_mask_shape")
    position_shape = prefill.get("position_ids_shape")
    cache_shape = prefill.get("cache_position_shape")
    invariants = {
        "prefill_query_positive": query_length > 1,
        "attention_mask_aligned_or_native_none": (
            attention_shape is None or int(attention_shape[-1]) == query_length
        ),
        "position_ids_aligned_or_native_none": (
            position_shape is None or int(position_shape[-1]) == query_length
        ),
        "cache_position_aligned_or_native_none": (
            cache_shape is None or int(cache_shape[-1]) == query_length
        ),
    }
    if not all(invariants.values()):
        raise RuntimeError(f"Decoder prefill alignment failed: {invariants!r}")
    return {
        "call_count": len(records),
        "prefill": dict(prefill),
        "decode_query_lengths": [
            int(item["query_length"])
            for item in records
            if item.get("query_length") == 1
        ],
        "invariants": invariants,
    }


def _checkpoint_record(checkpoint: Path) -> dict[str, Any]:
    weight_files = sorted(checkpoint.glob("*.safetensors"))
    if not weight_files:
        raise RuntimeError(f"Checkpoint has no safetensors files: {checkpoint}")
    config = checkpoint / "config.json"
    if not config.is_file():
        raise RuntimeError(f"Checkpoint config is missing: {config}")
    return {
        "path": str(checkpoint),
        "config_path": str(config),
        "weight_files": [
            {"name": path.name, "size_bytes": path.stat().st_size}
            for path in weight_files
        ],
        "total_weight_bytes": sum(path.stat().st_size for path in weight_files),
    }


def prepare_checkpoint_environment(physical_gpu: int) -> str:
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible != str(physical_gpu):
        raise RuntimeError(
            "Expose exactly the selected physical GPU before launch: "
            f"expected CUDA_VISIBLE_DEVICES={physical_gpu}, observed={visible!r}"
        )
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["HF_DATASETS_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    return visible


def generation_documents(
    parquet_path: Path,
    video_root: Path,
) -> tuple[list[dict[str, Any]], dict[str, Path]]:
    if not parquet_path.is_file():
        raise RuntimeError(f"NExT-QA Parquet is missing: {parquet_path}")
    if not video_root.is_dir():
        raise RuntimeError(f"NExT-QA video root is missing: {video_root}")
    documents = _load_fixed_documents(parquet_path)
    videos = {
        document["video"]: _video_path(video_root, document["video"])
        for document in documents
    }
    return documents, videos


def admit_generation_device(args: argparse.Namespace, *, bridge: str):
    """Check the physical GPU before importing or allocating with CUDA."""
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
            f"The pinned {bridge} bridge requires Transformers 4.57.3, got "
            f"{transformers.__version__}"
        )
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError(
            f"Expected one logical CUDA device, got {torch.cuda.device_count()}"
        )
    return torch, transformers, gpu_before, admission


def load_generation_runtime(
    checkpoint: Path,
    *,
    facade: Any,
    torch: Any,
    dtype: Any,
    device_map: Any,
    metadata: Mapping[str, Any],
):
    from mllm_comp.backbones.base import BackboneRuntime

    processor = facade.processor_adapter.load_processor(
        str(checkpoint), local_files_only=True
    )
    model = facade.model_adapter.load_model(
        str(checkpoint),
        train=False,
        local_files_only=True,
        torch_dtype=dtype,
        attn_implementation="sdpa",
        device_map=device_map,
    )
    model.eval()
    runtime = BackboneRuntime(
        model=model,
        processor=processor,
        model_path_or_id=str(checkpoint),
        device=torch.device("cuda:0"),
        dtype=dtype,
        metadata=metadata,
    )
    return processor, model, runtime


def generation_activation(*, facade: Any, runtime: Any, binding_name: str, stage: Any):
    from types import SimpleNamespace

    from mllm_comp.compressors.multimodal_registry import (
        builtin_multimodal_compression_contracts,
    )
    from mllm_comp.core.vocabulary import ExecutionPhase

    binding = builtin_multimodal_compression_contracts().binding(binding_name)
    return binding.executor_factory(
        model=runtime.model,
        runtime=runtime,
        facade=facade,
        stage=stage,
        binding=binding,
        binding_config=SimpleNamespace(config={}),
        execution_phase=ExecutionPhase.GENERATE,
    )


def prepare_generation_state(
    *,
    facade: Any,
    runtime: Any,
    processor: Any,
    sample: Any,
    bridge: str,
):
    batch = facade.processor_adapter.prepare_inputs(processor, sample, phase="eval")
    state = facade.state_builder.build(batch, phase="eval")
    if facade.native_validator is None:
        raise RuntimeError(f"{bridge} native validator is missing")
    violations = facade.native_validator.validate_state(runtime, state, batch)
    if violations:
        raise RuntimeError(f"{bridge} native input validation failed: {violations}")
    return batch, state


@contextmanager
def compare_generation(
    *,
    activation: Any,
    generate: Callable[..., dict[str, Any]],
    torch: Any,
    **generation_inputs: Any,
):
    """Compare identical inputs and keep hooks installed while evidence is checked.

    The supplied generation function retains the backbone's decoder audits.
    Uninstall also runs when generation or the caller's evidence checks fail.
    """
    with torch.inference_mode():
        baseline = generate(torch=torch, **generation_inputs)
    before = activation.runtime_metadata()
    try:
        activation.install()
        with torch.inference_mode():
            compressed = generate(torch=torch, **generation_inputs)
        yield baseline, compressed, before, activation.runtime_metadata()
    finally:
        if activation.installed:
            activation.uninstall()


def generation_case(document: Mapping[str, Any], video: Path) -> dict[str, Any]:
    return {
        "task": "nextqa_mc_test",
        "split": "test",
        "doc_id": int(document["doc_id"]),
        "qid": int(document["qid"]),
        "question_type": document["type"],
        "video_id": document["video"],
        "video_path": str(video),
        "video_size_bytes": video.stat().st_size,
        "target": document["target"],
        "prompt": document["prompt"],
    }


def generation_sample(
    document: Mapping[str, Any],
    baseline: Mapping[str, Any],
    compressed: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "task": "nextqa_mc_test",
        "split": "test",
        "doc_id": int(document["doc_id"]),
        "qid": int(document["qid"]),
        "video": document["video"],
        "prompt": document["prompt"],
        "target": document["target"],
        "baseline_response": baseline["text"],
        "baseline_choice": baseline["parsed_choice"],
        "compressed_response": compressed["text"],
        "compressed_choice": compressed["parsed_choice"],
    }


def generation_environment(*, visible: str, torch: Any, transformers: Any, dtype: str):
    return {
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "cuda_visible_devices": visible,
        "logical_device": "cuda:0",
        "logical_device_count": torch.cuda.device_count(),
        "gpu_name": torch.cuda.get_device_name(0),
        "dtype": dtype,
        "offline": True,
    }


def checkpoint_gpu_report(
    *, before: Any, admission: Any, after: Any, torch: Any, **details: Any,
):
    return {
        "before": before,
        "admission": admission,
        "observed_peak_allocated_mib": round(
            torch.cuda.max_memory_allocated(0) / (1024**2), 3
        ),
        "observed_peak_reserved_mib": round(
            torch.cuda.max_memory_reserved(0) / (1024**2), 3
        ),
        "after": after,
        "other_processes_modified": False,
        **details,
    }


def inference_main(
    args: argparse.Namespace,
    run: Callable[[argparse.Namespace], tuple[dict[str, Any], list[dict[str, Any]]]],
) -> int:
    """Persist inference evidence after the command validates its arguments."""
    output_dir = args.output_dir.resolve()
    report_path = output_dir / "report.json"
    samples_path = output_dir / "samples.jsonl"
    if report_path.exists() or samples_path.exists():
        raise RuntimeError(
            f"Refusing to overwrite existing evidence below {output_dir}"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()
    try:
        report, sample_rows = run(args)
    except BaseException as exc:
        failure = {
            "status": "failed",
            "error_type": type(exc).__name__,
            "error": str(exc),
            "command": shlex.join(sys.argv),
            "code_revision": args.code_revision,
            "physical_gpu": args.physical_gpu,
            "elapsed_seconds": round(time.time() - started, 3),
        }
        _atomic_write_json(report_path, failure)
        raise
    _atomic_write_jsonl(samples_path, sample_rows)
    report["artifacts"] = {
        "report": str(report_path),
        "lmms_eval_samples": str(samples_path),
    }
    _atomic_write_json(report_path, report)
    print(json.dumps({
        "status": report["status"], "report": str(report_path),
        "sample_count": len(sample_rows),
    }, ensure_ascii=False))
    return 0
