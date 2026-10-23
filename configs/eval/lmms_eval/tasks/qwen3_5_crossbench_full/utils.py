"""Official-score wrappers for locally cached Qwen3.8 cross-benchmark evals."""

from __future__ import annotations

from io import BytesIO
from typing import Any, Mapping

from datasets import Dataset
from PIL import Image

from lmms_eval.tasks.scienceqa import utils as _scienceqa
from lmms_eval.tasks.ocrbench import utils as _ocrbench
from lmms_eval.tasks.pope import utils as _pope
from lmms_eval.tasks.refcoco import utils_rec as _refcoco
from lmms_eval.tasks.mmbench import en_utils as _mmbench


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


def ocrbench_doc_to_visual(doc):
    return [_decode_image(doc["image"]).convert("RGB")]


def ocrbench_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    return _ocrbench.ocrbench_doc_to_text(doc, lmms_eval_specific_kwargs or {})


def ocrbench_process_results(doc, results):
    return _ocrbench.ocrbench_process_results(doc, results)


def ocrbench_aggregate_accuracy(results, args):
    # lmms-eval's upstream task accumulator is module-global; reset it so this
    # run's score cannot inherit counts if a task is evaluated twice in-process.
    for key in _ocrbench.OCRBench_score:
        _ocrbench.OCRBench_score[key] = 0
    return _ocrbench.ocrbench_aggregate_accuracy(results, args)


def pope_doc_to_visual(doc):
    return [_decode_image(doc["image"]).convert("RGB")]


def pope_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    return _pope.pope_doc_to_text(doc, lmms_eval_specific_kwargs or {})


def pope_process_results(doc, results):
    return _pope.pope_process_results(doc, results)


def pope_aggregate_accuracy(results):
    return _pope.pope_aggregate_accuracy(results)


def pope_aggregate_precision(results):
    return _pope.pope_aggregate_precision(results)


def pope_aggregate_recall(results):
    return _pope.pope_aggregate_recall(results)


def pope_aggregate_f1_score(results):
    return _pope.pope_aggregate_f1_score(results)


def pope_aggregate_yes_ratio(results):
    return _pope.pope_aggregate_yes_ratio(results)


def refcoco_preprocess_dataset(dataset):
    """Decode dimensions, normalize xywh boxes, and expand each GT phrase."""
    rows = []
    for source in dataset:
        row = dict(source)
        image = _decode_image(row["image"])
        width, height = image.size
        x, y, box_w, box_h = row["bbox"]
        row["bbox"] = [x / width, y / height, (x + box_w) / width, (y + box_h) / height]
        answers = row.pop("answer")
        for answer in answers:
            item = dict(row)
            item["answer"] = answer
            rows.append(item)
    return Dataset.from_list(rows)


def refcoco_doc_to_visual(doc):
    return [_decode_image(doc["image"]).convert("RGB")]


def refcoco_doc_to_text(doc):
    return _refcoco.refcoco_bbox_rec_doc_to_text(doc)


def refcoco_process_results(doc, result):
    return _refcoco.refcoco_bbox_rec_process_result(doc, result)


def refcoco_iou(results):
    return _refcoco.refcoco_bbox_rec_iou(results)


def refcoco_acc01(results): return _refcoco.refcoco_bbox_rec_acc01(results)
def refcoco_acc03(results): return _refcoco.refcoco_bbox_rec_acc03(results)
def refcoco_acc05(results): return _refcoco.refcoco_bbox_rec_acc05(results)
def refcoco_acc07(results): return _refcoco.refcoco_bbox_rec_acc07(results)
def refcoco_acc09(results): return _refcoco.refcoco_bbox_rec_acc09(results)
def refcoco_center_acc(results): return _refcoco.refcoco_bbox_rec_center_acc(results)


def mmbench_doc_to_visual(doc):
    doc = dict(doc)
    doc["image"] = _decode_image(doc["image"])
    return _mmbench.mmbench_doc_to_visual(doc)


def mmbench_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    return _mmbench.mmbench_doc_to_text(doc, lmms_eval_specific_kwargs)


def mmbench_process_results(doc, results):
    return _mmbench.mmbench_process_results(doc, results)


def mmbench_aggregate_static(results, args):
    return _mmbench.mmbench_aggregate_dev_results_static(results, args)


def mmbench_aggregate_submission(results, args):
    return _mmbench.mmbench_aggregate_dev_results_submission(results, args)
