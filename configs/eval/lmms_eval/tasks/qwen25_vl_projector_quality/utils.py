"""Offline local-parquet task wrappers with the upstream task metrics."""

from __future__ import annotations

from io import BytesIO
from typing import Any, Mapping

from PIL import Image

from lmms_eval.tasks.chartqa import utils as _chartqa
from lmms_eval.tasks.pope import utils as _pope
from lmms_eval.tasks.textvqa import utils as _textvqa


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


def _visual(fn, doc):
    return _decode(fn(_decode(doc)))


def chartqa_doc_to_visual(doc):
    return _visual(_chartqa.chartqa_doc_to_visual, doc)


def chartqa_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    return _chartqa.chartqa_doc_to_text(_decode(doc), lmms_eval_specific_kwargs)


def chartqa_process_results(doc, results):
    return _chartqa.chartqa_process_results(_decode(doc), results)


def textvqa_doc_to_visual(doc):
    return _visual(_textvqa.textvqa_doc_to_visual, doc)


def textvqa_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    return _textvqa.textvqa_doc_to_text(_decode(doc), lmms_eval_specific_kwargs)


def textvqa_process_results(doc, results):
    return _textvqa.textvqa_process_results(_decode(doc), results)


def textvqa_aggregate_submissions(results):
    return _textvqa.textvqa_aggregate_submissions(results)


def pope_doc_to_visual(doc):
    return _visual(_pope.pope_doc_to_visual, doc)


def pope_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    return _pope.pope_doc_to_text(_decode(doc), lmms_eval_specific_kwargs)


def pope_process_results(doc, results):
    return _pope.pope_process_results(_decode(doc), results)


pope_aggregate_accuracy = _pope.pope_aggregate_accuracy
pope_aggregate_precision = _pope.pope_aggregate_precision
pope_aggregate_recall = _pope.pope_aggregate_recall
pope_aggregate_f1_score = _pope.pope_aggregate_f1_score
pope_aggregate_yes_ratio = _pope.pope_aggregate_yes_ratio


def gqa_doc_to_visual(doc):
    return [_gqa_images_by_id()[doc["imageId"]]]


@__import__("functools").lru_cache(maxsize=1)
def _gqa_images_by_id():
    from datasets import load_dataset

    path = "/home/YinPeiran/datasets/lmms_eval_datasets/GQA/testdev_all_images/testdev-00000-of-00001.parquet"
    rows = load_dataset("parquet", data_files=path, split="train")
    return {row["id"]: row["image"].convert("RGB") for row in rows}


def gqa_doc_to_text(doc, lmms_eval_specific_kwargs):
    return f"{lmms_eval_specific_kwargs['pre_prompt']}{doc['question']}{lmms_eval_specific_kwargs['post_prompt']}"
