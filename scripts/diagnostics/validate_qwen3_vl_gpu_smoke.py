"""Run Qwen3-VL text/image/video generation and traces on one assigned GPU."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time
from typing import Any

from mllm_comp.backbones.base import BackboneRuntime, TraceRequest
from mllm_comp.backbones.qwen3_vl import (
    DEFAULT_SYSTEM_PROMPT,
    Qwen3VLSample,
    create_qwen3_vl_backbone,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _shape(value: Any) -> list[int] | None:
    shape = getattr(value, "shape", None)
    return None if shape is None else [int(dim) for dim in shape]


def _conversation(content: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "role": "system",
            "content": [{"type": "text", "text": DEFAULT_SYSTEM_PROMPT}],
        },
        {"role": "user", "content": content},
    ]


def _create_media(root: Path) -> tuple[Path, Path]:
    from PIL import Image

    image_path = root / "qwen3_vl_smoke.png"
    video_path = root / "qwen3_vl_smoke.mp4"
    Image.new("RGB", (256, 256), color=(192, 64, 32)).save(image_path)
    environment_ffmpeg = Path(sys.executable).with_name("ffmpeg")
    ffmpeg = shutil.which("ffmpeg") or (
        str(environment_ffmpeg) if environment_ffmpeg.is_file() else None
    )
    if ffmpeg is None:
        raise RuntimeError("ffmpeg is required for the deterministic video fixture.")
    subprocess.run(
        [
            ffmpeg,
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-f",
            "lavfi",
            "-i",
            "color=c=green:s=256x256:r=4:d=1",
            "-an",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            str(video_path),
        ],
        check=True,
    )
    return image_path, video_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--report_path", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max_new_tokens", type=int, default=16)
    parser.add_argument("--video_reader", default="torchcodec")
    args = parser.parse_args(argv)

    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    os.environ["FORCE_QWENVL_VIDEO_READER"] = args.video_reader

    import torch
    import transformers
    from importlib import metadata

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError(
            "Expose one assigned GPU through CUDA_VISIBLE_DEVICES; "
            f"got {torch.cuda.device_count()} logical CUDA devices."
        )
    model_path = Path(args.model_path).resolve()
    if not model_path.is_dir():
        raise RuntimeError(f"Qwen3-VL checkpoint directory does not exist: {model_path}")

    started_at = time.time()
    facade = create_qwen3_vl_backbone()
    processor = facade.processor_adapter.load_processor(
        str(model_path), local_files_only=True
    )
    model = facade.model_adapter.load_model(
        str(model_path),
        train=False,
        local_files_only=True,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map={"": 0},
    )
    runtime = BackboneRuntime(
        model=model,
        processor=processor,
        model_path_or_id=str(model_path),
        device=torch.device(args.device),
        dtype=torch.bfloat16,
        metadata={"enable_talker": False, "speech_sample_rate": None},
    )
    handles = facade.model_adapter.structural_handles(model)
    required_handles = {"llm", "lm_head", "vision_encoder", "vision_projector"}
    missing_handles = sorted(required_handles.difference(handles))
    if missing_handles:
        raise RuntimeError(f"Missing Qwen3-VL structural handles: {missing_handles}")
    if not any(name.startswith("vision_deepstack_projector[") for name in handles):
        raise RuntimeError("Qwen3-VL DeepStack projectors were not discovered.")
    if facade.native_validator is None:
        raise RuntimeError("Qwen3-VL native validator is missing.")

    with tempfile.TemporaryDirectory(prefix="qwen3-vl-gpu-") as temp_dir:
        image_path, video_path = _create_media(Path(temp_dir))
        cases = {
            "text": Qwen3VLSample(
                "text",
                _conversation(
                    [{"type": "text", "text": "Reply with one short greeting."}]
                ),
                enable_thinking=False,
            ),
            "image": Qwen3VLSample(
                "image",
                _conversation(
                    [
                        {"type": "image", "image": str(image_path)},
                        {"type": "text", "text": "What is the dominant color?"},
                    ]
                ),
                enable_thinking=False,
            ),
            "video": Qwen3VLSample(
                "video",
                _conversation(
                    [
                        {
                            "type": "video",
                            "video": str(video_path),
                            "fps": 4.0,
                        },
                        {
                            "type": "text",
                            "text": "What color dominates the video?",
                        },
                    ]
                ),
                enable_thinking=False,
            ),
        }
        case_reports: dict[str, Any] = {}
        for name, sample in cases.items():
            case_started = time.time()
            batch = facade.processor_adapter.prepare_inputs(
                processor, sample, phase="eval"
            )
            state = facade.state_builder.build(batch, phase="eval")
            result = facade.runner.generate(
                runtime,
                state,
                processed_batch=batch,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
            )
            if result.text is None or not result.text.text[0].strip():
                raise RuntimeError(f"Qwen3-VL {name} generation returned no text.")
            if result.speech is not None:
                raise RuntimeError("Qwen3-VL unexpectedly returned speech output.")
            validation_errors = facade.native_validator.validate_state(
                runtime, state, batch
            )
            if validation_errors:
                raise RuntimeError(
                    f"Qwen3-VL {name} native state is invalid: {validation_errors}"
                )
            case_reports[name] = {
                "validation_errors": validation_errors,
                "processor_contract": batch.metadata["processor_output_contract"],
                "expected_token_counts": batch.metadata[
                    "expected_token_counts_by_sample"
                ],
                "video_metadata_present": batch.metadata["video_metadata_present"],
                "spans": [
                    {
                        "modality": span.modality,
                        "start": span.start,
                        "end": span.end,
                    }
                    for span in state.modality_token_spans
                ],
                "generated_token_shape": _shape(result.text.token_ids),
                "text": result.text.text,
                "elapsed_seconds": round(time.time() - case_started, 3),
            }

        trace_sample = cases["text"]
        trace_batch = facade.processor_adapter.prepare_inputs(
            processor, trace_sample, phase="eval"
        )
        trace_state = facade.state_builder.build(trace_batch, phase="eval")
        trace_result = facade.runner.generate(
            runtime,
            trace_state,
            processed_batch=trace_batch,
            trace_request=TraceRequest(
                text_logits=True,
                text_hidden_states=True,
                text_kv_cache=True,
            ),
            max_new_tokens=2,
            do_sample=False,
        )
        traces = trace_result.traces
        if traces is None:
            raise RuntimeError("Qwen3-VL trace request returned no artifacts.")
        if traces.text_logits is None:
            raise RuntimeError("Qwen3-VL text logits trace is missing.")
        if traces.text_hidden_states is None:
            raise RuntimeError("Qwen3-VL hidden-state trace is missing.")
        if traces.text_kv_cache is None:
            raise RuntimeError("Qwen3-VL KV-cache trace is missing.")

    report = {
        "status": "passed",
        "environment": {
            "transformers": transformers.__version__,
            "qwen_vl_utils": metadata.version("qwen-vl-utils"),
            "torch": torch.__version__,
            "cuda_visible_devices": visible,
            "logical_device_count": torch.cuda.device_count(),
            "logical_device": args.device,
            "gpu_name": torch.cuda.get_device_name(0),
            "dtype": "bfloat16",
            "video_reader": args.video_reader,
            "model_path": str(model_path),
            "config_sha256": _sha256(model_path / "config.json"),
        },
        "structural_handles": {
            "count": len(handles),
            "deepstack_projectors": sorted(
                name
                for name in handles
                if name.startswith("vision_deepstack_projector[")
            ),
        },
        "cases": case_reports,
        "trace": {
            "logits_shape": _shape(traces.text_logits),
            "hidden_state_layers": len(traces.text_hidden_states),
            "kv_cache_type": type(traces.text_kv_cache).__qualname__,
            "capture_semantics": traces.metadata["capture_semantics"],
        },
        "peak_allocated_gib": round(
            torch.cuda.max_memory_allocated(0) / (1024**3), 3
        ),
        "elapsed_seconds": round(time.time() - started_at, 3),
    }
    report_path = Path(args.report_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(report, ensure_ascii=False, indent=2)
    report_path.write_text(payload, encoding="utf-8")
    print(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
