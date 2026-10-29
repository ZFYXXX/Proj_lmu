"""Local OK-VQA validation task with native lmms-eval soft accuracy."""

from io import BytesIO
import os

from PIL import Image
from lmms_eval.tasks.ok_vqa import utils as native


def process_docs(dataset):
    return dataset.select(range(min(8, len(dataset)))) if os.environ.get("MLLM_EXTRA_SMOKE") == "1" else dataset


def _image(doc):
    value = doc["image"]
    if isinstance(value, Image.Image):
        return value.convert("RGB")
    if isinstance(value, dict):
        value = BytesIO(value["bytes"]) if value.get("bytes") else value["path"]
    with Image.open(value) as image:
        return image.convert("RGB")


def doc_to_visual(doc):
    return [_image(doc)]


def doc_to_text(doc, lmms_eval_specific_kwargs=None):
    return native.ok_vqa_doc_to_text(doc, lmms_eval_specific_kwargs)


process_results = native.ok_vqa_process_results

