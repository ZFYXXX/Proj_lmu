"""Validate the Qwen3-VL processor contract without loading model weights."""

from __future__ import annotations

import argparse
from importlib import metadata
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from typing import Any

from mllm_comp.backbones.base import BackboneRuntime
from mllm_comp.backbones.qwen3_vl import (
    DEFAULT_SYSTEM_PROMPT,
    Qwen3VLSample,
    create_qwen3_vl_backbone,
)


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

    image_path = root / "qwen3_vl_processor.png"
    video_path = root / "qwen3_vl_processor.mp4"
    Image.new("RGB", (256, 256), color=(192, 64, 32)).save(image_path)
    environment_ffmpeg = Path(sys.executable).with_name("ffmpeg")
    ffmpeg = shutil.which("ffmpeg") or (
        str(environment_ffmpeg) if environment_ffmpeg.is_file() else None
    )
    if ffmpeg is None:
        raise RuntimeError("ffmpeg is required for the Qwen3-VL processor contract.")
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
    parser.add_argument("--report_path")
    parser.add_argument("--video_reader", default="torchcodec")
    args = parser.parse_args(argv)

    os.environ["FORCE_QWENVL_VIDEO_READER"] = args.video_reader
    import transformers
    from transformers import Qwen3VLProcessor  # noqa: F401

    model_path = Path(args.model_path).resolve()
    if not model_path.is_dir():
        raise RuntimeError(f"Qwen3-VL checkpoint directory does not exist: {model_path}")

    facade = create_qwen3_vl_backbone()
    processor = facade.processor_adapter.load_processor(
        str(model_path), local_files_only=True
    )
    runtime = BackboneRuntime(
        model=object(),
        processor=processor,
        model_path_or_id=str(model_path),
        device="cpu",
    )

    with tempfile.TemporaryDirectory(prefix="qwen3-vl-processor-") as temp_dir:
        image_path, video_path = _create_media(Path(temp_dir))
        cases = {
            "text": Qwen3VLSample(
                "text",
                _conversation([{"type": "text", "text": "Say hello."}]),
                enable_thinking=False,
            ),
            "image": Qwen3VLSample(
                "image",
                _conversation(
                    [
                        {"type": "image", "image": str(image_path)},
                        {"type": "text", "text": "Describe the color."},
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
                        {"type": "text", "text": "Describe the color."},
                    ]
                ),
                enable_thinking=False,
            ),
        }
        case_reports: dict[str, Any] = {}
        for name, sample in cases.items():
            batch = facade.processor_adapter.prepare_inputs(
                processor, sample, phase="eval"
            )
            state = facade.state_builder.build(batch, phase="eval")
            violations = facade.native_validator.validate_state(runtime, state, batch)
            if violations:
                details = [
                    {
                        "name": item.name,
                        "field": item.field,
                        "expected": item.expected,
                        "actual": item.actual,
                    }
                    for item in violations
                ]
                raise RuntimeError(
                    f"Qwen3-VL processor contract failed for {name}: {details}"
                )
            case_reports[name] = {
                "processor_keys": sorted(batch.model_inputs),
                "shapes": {
                    key: _shape(value) for key, value in batch.model_inputs.items()
                },
                "expected_token_counts": batch.metadata[
                    "expected_token_counts_by_sample"
                ],
                "spans": [
                    {
                        "modality": span.modality,
                        "start": span.start,
                        "end": span.end,
                    }
                    for span in state.modality_token_spans
                ],
                "image_patch_size": batch.metadata["image_patch_size"],
                "spatial_merge_size": batch.metadata["spatial_merge_size"],
                "video_metadata_present": batch.metadata["video_metadata_present"],
            }

    report = {
        "status": "passed",
        "model_path": str(model_path),
        "transformers": transformers.__version__,
        "qwen_vl_utils": metadata.version("qwen-vl-utils"),
        "video_reader": args.video_reader,
        "cases": case_reports,
    }
    payload = json.dumps(report, ensure_ascii=False, indent=2)
    if args.report_path:
        report_path = Path(args.report_path)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(payload, encoding="utf-8")
    print(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
