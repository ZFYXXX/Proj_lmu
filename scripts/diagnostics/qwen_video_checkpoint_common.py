"""Qwen video generation evidence, training layout checks and native loading.

These checks belong to Qwen's placeholder/M-RoPE contract, independent of the
video algorithm that produced the representatives.
"""

from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Any, Mapping

from mllm_comp.backbones.base import BackboneRuntime, TraceRequest
from mllm_comp.backbones.qwen3_vl import create_qwen3_vl_backbone
from mllm_comp.backbones.qwen3_vl.video_compression import QWEN3_VL_IGNORE_INDEX
from mllm_comp.backbones.qwen25_vl import DEFAULT_SYSTEM_PROMPT
from scripts.diagnostics.video_checkpoint_common import (
    _decoder_summary,
    _DecoderInputAudit,
    _parse_choice,
    _shape,
    _tensor_rows,
)
from scripts.diagnostics.video_training_checkpoint_common import load_training_runtime


def load_qwen3_runtime(checkpoint: Path, *, torch: Any, train: bool):
    return load_training_runtime(
        checkpoint, facade=create_qwen3_vl_backbone(), torch=torch, train=train, device_map={"": 0},
    )


def span_count(value: Any) -> int:
    try:
        spans = tuple(tuple(int(item) for item in span) for span in value)
    except (TypeError, ValueError):
        return -1
    if any(len(span) != 2 or span[0] < 0 or span[1] <= span[0] for span in spans):
        return -1
    return sum(end - start for start, end in spans)


def mrope_gather_exact(splice: Mapping[str, Any]) -> bool:
    source = tuple(
        tuple(int(value) for value in row)
        for row in splice.get("source_visual_mrope", ())
    )
    result = tuple(
        tuple(int(value) for value in row)
        for row in splice.get("result_visual_mrope", ())
    )
    representatives = tuple(
        int(value) for value in splice.get("representative_indices", ())
    )
    return bool(
        source
        and result
        and len(result) == len(representatives)
        and result == tuple(source[index] for index in representatives)
    )


def completion_only_labels_exact(state: Any, splice: Mapping[str, Any]) -> bool:
    labels = state.labels
    if labels is None or tuple(labels.shape[:1]) != (1,):
        return False
    positions = tuple(
        index
        for start, end in (
            tuple(int(item) for item in span)
            for span in splice.get("source_placeholder_spans", ())
        )
        for index in range(start, end)
    )
    if len(positions) != int(splice.get("source_placeholder_count", -1)):
        return False
    return bool(
        positions
        and (labels[0, list(positions)] == QWEN3_VL_IGNORE_INDEX).all().item()
        and (labels != QWEN3_VL_IGNORE_INDEX).any().item()
    )


QWEN25_VIDEO_FPS = 1.0
QWEN25_VIDEO_MAX_PIXELS = 128 * 28 * 28


def _conversation(
    *,
    prompt: str,
    video: Path,
    fps: float = QWEN25_VIDEO_FPS,
    max_pixels: int = QWEN25_VIDEO_MAX_PIXELS,
) -> list[dict[str, Any]]:
    if fps <= 0:
        raise ValueError("Qwen2.5-VL video fps must be positive")
    if max_pixels <= 0:
        raise ValueError("Qwen2.5-VL video max_pixels must be positive")
    return [
        {
            "role": "system",
            "content": [{"type": "text", "text": DEFAULT_SYSTEM_PROMPT}],
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "video",
                    "video": str(video),
                    "fps": float(fps),
                    "max_pixels": int(max_pixels),
                },
                {"type": "text", "text": prompt},
            ],
        },
    ]


def float_values(value: Any) -> list[float]:
    detach = getattr(value, "detach", None)
    if callable(detach):
        value = detach().cpu().reshape(-1).tolist()
    elif hasattr(value, "tolist"):
        value = value.tolist()
    if not isinstance(value, list):
        value = [value]
    flattened: list[float] = []
    for item in value:
        if isinstance(item, list):
            flattened.extend(float(child) for child in item)
        else:
            flattened.append(float(item))
    return flattened


def qwen25_training_grid_record(
    grid: list[int], *, merge_size: int, timing: Any,
) -> dict[str, Any]:
    """Record a method-validated Qwen2.5 raster and its required positive timing."""
    timing = float_values(timing)
    if len(timing) != 1 or not math.isfinite(timing[0]) or timing[0] <= 0:
        raise RuntimeError(f"Expected one positive video timing value, got {timing!r}")
    temporal, height, width = grid
    per_slice = height * width // (merge_size**2)
    return {
        "video_grid_thw": grid,
        "spatial_merge_size": merge_size,
        "temporal_slice_count": temporal,
        "per_temporal_slice_source_count": per_slice,
        "source_grid_token_count": temporal * per_slice,
        "second_per_grid_ts": timing,
    }


def qwen25_video_input_record(state: Any, batch: Any) -> dict[str, Any]:
    """Validate one native video and return its reusable input evidence."""
    spans = [span for span in state.modality_token_spans if span.modality == "video"]
    if len(spans) != 1:
        raise RuntimeError(f"Expected one Qwen2.5-VL video span, got {spans!r}")
    grid_rows = _tensor_rows(state.video_grid_thw)
    if len(grid_rows) != 1 or len(grid_rows[0]) != 3:
        raise RuntimeError(f"Expected one video_grid_thw row, got {grid_rows!r}")
    grid = grid_rows[0]
    merge_size = int(batch.metadata["spatial_merge_size"])
    source_count = grid[0] * grid[1] * grid[2] // (merge_size**2)
    span = spans[0]
    span_count = span.end - span.start
    if span_count != source_count:
        raise RuntimeError(
            "Qwen2.5-VL video span differs from video_grid_thw token count: "
            f"span={span_count}, grid={source_count}"
        )
    timing = float_values(state.video_second_per_grid)
    if len(timing) != 1 or not math.isfinite(timing[0]):
        raise RuntimeError(
            "Expected one finite Qwen2.5-VL second_per_grid_ts value, got "
            f"{timing!r}"
        )
    return {
        "video_grid_thw": grid,
        "second_per_grid_ts": timing,
        "spatial_merge_size": merge_size,
        "source_grid_token_count": source_count,
        "video_span": {"start": span.start, "end": span.end, "count": span_count},
        "pixel_values_videos_shape": _shape(state.video_pixel_values),
        "processor_output_contract": batch.metadata["processor_output_contract"],
        "labels_absent_for_inference": state.labels is None
        and (
            "labels" not in state.native_model_inputs
            or state.native_model_inputs["labels"] is None
        ),
    }


def generation_report(
    *,
    facade: Any,
    runtime: BackboneRuntime,
    state: Any,
    batch: Any,
    audit: _DecoderInputAudit,
    torch: Any,
    max_new_tokens: int,
    min_new_tokens: int = 2,
) -> dict[str, Any]:
    if not 1 <= min_new_tokens <= max_new_tokens:
        raise ValueError("min_new_tokens must be within the generation bound")
    mark = audit.mark()
    started = time.time()
    result = facade.runner.generate(
        runtime,
        state,
        processed_batch=batch,
        trace_request=TraceRequest(text_logits=True),
        max_new_tokens=max_new_tokens,
        min_new_tokens=min_new_tokens,
        do_sample=False,
        use_cache=True,
    )
    torch.cuda.synchronize()
    if result.text is None:
        raise RuntimeError("Qwen2.5-VL generation returned no text branch")
    token_rows = _tensor_rows(result.text.token_ids)
    if len(token_rows) != 1 or not 2 <= len(token_rows[0]) <= max_new_tokens:
        raise RuntimeError(
            "Generated token count is outside the finite request bound: "
            f"rows={token_rows!r}, max_new_tokens={max_new_tokens}"
        )
    traces = result.traces
    logits = None if traces is None else traces.text_logits
    if logits is None or not bool(torch.isfinite(logits).all().item()):
        raise RuntimeError("Generated text logits are missing or non-finite")
    text_rows = list(result.text.text)
    if len(text_rows) != 1:
        raise RuntimeError(f"Expected one decoded result, got {text_rows!r}")
    return {
        "generated_token_ids": token_rows[0],
        "generated_token_count": len(token_rows[0]),
        "text": text_rows[0],
        "parsed_choice": _parse_choice(text_rows[0]),
        "logits_shape": _shape(logits),
        "logits_finite": True,
        "decoder": _decoder_summary(audit.since(mark)),
        "elapsed_seconds": round(time.time() - started, 3),
    }


def decode_record_coherent(
    record: Any,
    *,
    removed_count: int,
    result_sequence_length: int,
) -> bool:
    if not isinstance(record, Mapping):
        return False
    input_attention_length = record.get("input_attention_length")
    if input_attention_length is not None and int(input_attention_length) < int(
        result_sequence_length
    ):
        return False
    if bool(record.get("already_compact")):
        return True
    if int(record.get("removed_count", -1)) != int(removed_count):
        return False
    output_attention_length = record.get("output_attention_length")
    if input_attention_length is not None and output_attention_length is not None:
        if int(input_attention_length) - int(output_attention_length) != int(
            removed_count
        ):
            return False
    return True


def valid_spans(value: Any) -> tuple[tuple[int, int], ...]:
    try:
        spans = tuple(tuple(int(item) for item in span) for span in value)
    except (TypeError, ValueError):
        return ()
    if any(len(span) != 2 or span[0] < 0 or span[1] <= span[0] for span in spans):
        return ()
    return tuple((span[0], span[1]) for span in spans)


def training_timing_preserved(splice: Mapping[str, Any] | None, expected: float) -> bool:
    if splice is None:
        return False
    try:
        observed = float(splice.get("second_per_grid_seconds"))
    except (TypeError, ValueError):
        return False
    return math.isfinite(observed) and math.isclose(
        observed,
        expected,
        rel_tol=0.0,
        abs_tol=1e-9,
    )
