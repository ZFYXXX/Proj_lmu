"""Local-parquet wrappers for image-only Qwen3.8 smoke tasks.

The source parquet files store image features as {bytes, path} structs rather
than decoded PIL images, so the wrappers decode those structs before handing
documents to the unchanged LMMS-Eval task prompts.
"""

from __future__ import annotations

from io import BytesIO
from typing import Any, Mapping

from PIL import Image

from lmms_eval.tasks.pope import utils as _pope
from lmms_eval.tasks.scienceqa import utils as _scienceqa
from lmms_eval.tasks.textvqa import utils as _textvqa
from lmms_eval.tasks.ai2d import utils as _ai2d
from lmms_eval.tasks.chartqa import utils as _chartqa
from lmms_eval.tasks.docvqa import utils as _docvqa
from lmms_eval.tasks.ocrbench import utils as _ocrbench
from lmms_eval.tasks.ok_vqa import utils as _okvqa
from lmms_eval.tasks.vqav2 import utils as _vqav2
from lmms_eval.tasks.refcoco import utils_rec as _refcoco
from lmms_eval.tasks.mmmu import utils as _mmmu


def _decode(value: Any) -> Any:
    if isinstance(value, Mapping):
        if "bytes" in value and "path" in value:
            raw = value.get("bytes")
            path = value.get("path")
            if raw:
                return Image.open(BytesIO(raw)).convert("RGB")
            if path:
                return Image.open(path).convert("RGB")
        return {key: _decode(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_decode(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_decode(item) for item in value)
    return value


def _visual(fn, doc: Mapping[str, Any], *args: Any, **kwargs: Any):
    return _decode(fn(_decode(doc), *args, **kwargs))


def pope_doc_to_visual(doc): return _visual(_pope.pope_doc_to_visual, doc)
def pope_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    return _pope.pope_doc_to_text(_decode(doc), lmms_eval_specific_kwargs)
def sqa_doc_to_visual(doc): return _visual(_scienceqa.sqa_doc_to_visual, doc)
def sqa_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    return _scienceqa.sqa_doc_to_text(_decode(doc), lmms_eval_specific_kwargs)
def sqa_doc_to_target(doc): return _scienceqa.sqa_doc_to_target(_decode(doc))
def textvqa_doc_to_visual(doc): return _visual(_textvqa.textvqa_doc_to_visual, doc)
def textvqa_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    return _textvqa.textvqa_doc_to_text(_decode(doc), lmms_eval_specific_kwargs)
def ai2d_doc_to_visual(doc): return _visual(_ai2d.ai2d_doc_to_visual, doc)
def ai2d_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    return _ai2d.ai2d_doc_to_text(_decode(doc), lmms_eval_specific_kwargs)
def ai2d_doc_to_target(doc, model_specific_target_kwargs=None):
    return _ai2d.ai2d_doc_to_target(
        _decode(doc), model_specific_target_kwargs or {"default": "mcq"}
    )
def chartqa_doc_to_visual(doc): return _visual(_chartqa.chartqa_doc_to_visual, doc)
def chartqa_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    return _chartqa.chartqa_doc_to_text(_decode(doc), lmms_eval_specific_kwargs)
def docvqa_doc_to_visual(doc): return _visual(_docvqa.docvqa_doc_to_visual, doc)
def docvqa_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    return _docvqa.docvqa_doc_to_text(_decode(doc), lmms_eval_specific_kwargs)
def ocrbench_doc_to_visual(doc): return _visual(_ocrbench.ocrbench_doc_to_visual, doc)
def ocrbench_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    return _ocrbench.ocrbench_doc_to_text(_decode(doc), lmms_eval_specific_kwargs)
def okvqa_doc_to_visual(doc): return _visual(_okvqa.ok_vqa_doc_to_visual, doc)
def okvqa_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    return _okvqa.ok_vqa_doc_to_text(_decode(doc), lmms_eval_specific_kwargs)
def vqav2_doc_to_visual(doc): return _visual(_vqav2.vqav2_doc_to_visual, doc)
def vqav2_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    return _vqav2.vqav2_doc_to_text(_decode(doc), lmms_eval_specific_kwargs)
def refcoco_doc_to_visual(doc): return _visual(_refcoco.refcoco_bbox_rec_doc_to_visual, doc)
def refcoco_doc_to_text(doc):
    normalized = _decode(doc)
    answer = normalized.get("answer", "")
    if isinstance(answer, list):
        normalized["answer"] = answer[0] if answer else ""
    return _refcoco.refcoco_bbox_rec_doc_to_text(normalized)
def mmmu_doc_to_visual(doc): return _visual(_mmmu.mmmu_doc_to_visual, doc)
def mmmu_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    return _mmmu.mmmu_doc_to_text(_decode(doc), lmms_eval_specific_kwargs)


def smoke_process_results(doc, results):
    del doc
    prediction = results[0] if results else ""
    return {"smoke_nonempty": bool(str(prediction).strip())}


def aggregate_smoke_nonempty(results):
    if not results:
        return 0.0
    return 100.0 * sum(bool(value) for value in results) / len(results)
