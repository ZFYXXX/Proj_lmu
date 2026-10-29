"""Local RefCOCO+ tasks using Qwen2.5's native grounding prompt.

The platform's qwen-grounding-v2 wrapper owns prediction parsing and processed-
image coordinate normalization. These functions only prepare the canonical
dataset rows and aggregate its corrected payloads.
"""
import os

from lmms_eval.tasks.refcoco import utils_rec as native


def process_docs(dataset):
    rows = native.refcoco_bbox_rec_preprocess_dataset(dataset)
    if os.environ.get("MLLM_EXTRA_SMOKE") == "1":
        return rows.select(range(min(8, len(rows))))
    return rows


def doc_to_visual(doc):
    return native.refcoco_bbox_rec_doc_to_visual(doc)


def doc_to_text(doc):
    # Qwen2.5-VL is trained to return pixel-space boxes for this prompt. The
    # scorer verifies and normalizes against the actual post-resize image grid.
    return f"Locate {doc['answer']} in this image and output the bbox coordinates in JSON format."


def process_results(doc, results):
    # Replaced at runtime by qwen-grounding-v2 for platform Qwen backbones.
    return native.refcoco_bbox_rec_process_result(doc, results)


iou = native.refcoco_bbox_rec_iou
acc01 = native.refcoco_bbox_rec_acc01
acc03 = native.refcoco_bbox_rec_acc03
acc05 = native.refcoco_bbox_rec_acc05
acc07 = native.refcoco_bbox_rec_acc07
acc09 = native.refcoco_bbox_rec_acc09
center_acc = native.refcoco_bbox_rec_center_acc
