"""Local-parquet ChartQA wrappers for deterministic offline evaluation."""

from __future__ import annotations

from collections.abc import Mapping
from io import BytesIO
from typing import Any

from PIL import Image

from lmms_eval.tasks.chartqa import utils as _chartqa


def _decode(value: Any) -> Any:
    if isinstance(value, Mapping):
        if "bytes" in value and "path" in value:
            raw, path = value.get("bytes"), value.get("path")
            if raw:
                return Image.open(BytesIO(raw)).convert("RGB")
            if path:
                return Image.open(path).convert("RGB")
        return {key: _decode(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_decode(item) for item in value]
    return value


def chartqa_doc_to_visual(doc):
    return _chartqa.chartqa_doc_to_visual(_decode(doc))


def chartqa_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    return _chartqa.chartqa_doc_to_text(_decode(doc), lmms_eval_specific_kwargs)


def chartqa_process_results(doc, results):
    return _chartqa.chartqa_process_results(_decode(doc), results)
