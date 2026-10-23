"""Local-parquet ScienceQA wrappers for the Qwen2.5-VL K=64 comparison."""

from __future__ import annotations

from io import BytesIO
from typing import Any, Mapping

from PIL import Image
from lmms_eval.tasks.scienceqa import utils as _scienceqa


def _decode_image(value: Any) -> Any:
    if isinstance(value, Mapping) and "bytes" in value and "path" in value:
        raw, path = value.get("bytes"), value.get("path")
        if raw:
            return Image.open(BytesIO(raw)).convert("RGB")
        if path:
            return Image.open(path).convert("RGB")
    return value


def sqa_doc_to_visual(doc):
    image = _decode_image(doc.get("image"))
    return [] if image is None else [image.convert("RGB")]


def sqa_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    return _scienceqa.sqa_doc_to_text(doc, lmms_eval_specific_kwargs)


def sqa_doc_to_target(doc):
    return _scienceqa.sqa_doc_to_target(doc)


def sqa_process_results(doc, results):
    return _scienceqa.sqa_process_results(doc, results)
