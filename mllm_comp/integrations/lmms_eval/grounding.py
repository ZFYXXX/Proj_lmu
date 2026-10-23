"""Versioned RefCOCO scoring, without modifying the shared lmms-eval checkout.

Qwen3-VL emits 0..1000 coordinates; Qwen2.5-VL emits pixel coordinates
in the image it actually receives (after resizing), not the dataset image.
No coordinate-space inference from prediction magnitude is permitted.
"""
from __future__ import annotations

import functools
import json
import math
import re

PROTOCOL = "qwen-grounding-v2"
METRICS = ("IoU", "ACC@0.1", "ACC@0.3", "ACC@0.5", "ACC@0.7", "ACC@0.9", "Center_ACC")
_NUMBER = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"
_QUAD = re.compile(r"[\[(]\s*(" + _NUMBER + r")\s*,\s*(" + _NUMBER + r")\s*,\s*(" + _NUMBER + r")\s*,\s*(" + _NUMBER + r")\s*[\])]")
_PAIR = re.compile(r"\(\s*(" + _NUMBER + r")\s*,\s*(" + _NUMBER + r")\s*\)\s*,?\s*\(\s*(" + _NUMBER + r")\s*,\s*(" + _NUMBER + r")\s*\)")


def parse_box(value):
    """Extract the first explicit bbox, never arbitrary digits from prose."""
    def walk(obj):
        if isinstance(obj, dict):
            for key in ("bbox_2d", "bbox", "box"):
                if key in obj:
                    return walk(obj[key])
        if isinstance(obj, (list, tuple)):
            if len(obj) == 4 and all(type(x) in (int, float) for x in obj):
                return [float(x) for x in obj] if all(math.isfinite(x) for x in obj) else None
            if obj and isinstance(obj[0], (dict, list, tuple)):
                return walk(obj[0])
        return None

    if not isinstance(value, str):
        return walk(value)
    text = re.sub(r"^\s*```(?:json)?\s*|\s*```\s*$", "", value.strip())
    try:
        # A syntactically valid JSON value is authoritative, including invalid boxes.
        return walk(json.loads(text))
    except (ValueError, TypeError):
        pass
    match = _QUAD.search(text) or _PAIR.search(text)
    if match:
        box = [float(x) for x in match.groups()]
        return box if all(math.isfinite(x) for x in box) else None
    return None


def qwen25_input_size(width, height, min_pixels, max_pixels):
    """Qwen2.5 smart_resize geometry, returned as (width, height).

    Matches qwen-vl-utils and the Qwen2VL image processor (factor 28).
    The bridge verifies this against the actual image_grid_thw before inference.
    """
    width, height = int(width), int(height)
    if min(width, height, min_pixels) <= 0 or max_pixels < min_pixels:
        raise ValueError("Invalid image size or processor pixel bounds")
    if max(width, height) / min(width, height) > 200:
        raise ValueError("Unsupported image aspect ratio")
    factor = 28
    w, h = max(factor, round(width / factor) * factor), max(factor, round(height / factor) * factor)
    if w * h > max_pixels:
        beta = math.sqrt(width * height / max_pixels)
        w, h = math.floor(width / beta / factor) * factor, math.floor(height / beta / factor) * factor
    elif w * h < min_pixels:
        beta = math.sqrt(min_pixels / (width * height))
        w, h = math.ceil(width * beta / factor) * factor, math.ceil(height * beta / factor) * factor
    return w, h


def normalize_prediction(raw, backbone, *, input_size=None):
    if backbone == "qwen3_vl":
        width = height = 1000.
    elif backbone == "qwen25_vl":
        if input_size is None or min(input_size) <= 0:
            raise ValueError("Qwen2.5 grounding needs verified processed image dimensions")
        width, height = input_size
    else:
        raise ValueError(f"Unsupported grounding backbone: {backbone}")
    box = parse_box(raw)
    if box is None:
        return [-1.] * 4, "parse_error"
    box = [max(0., min(1., x / scale)) for x, scale in zip(box, (width, height, width, height))]
    if box[2] <= box[0] or box[3] <= box[1]:
        return [-1.] * 4, "invalid_geometry"
    return box, "valid"


def iou(a, b):
    aa = max(0., a[2] - a[0]) * max(0., a[3] - a[1])
    bb = max(0., b[2] - b[0]) * max(0., b[3] - b[1])
    intersection = max(0., min(a[2], b[2]) - max(a[0], b[0])) * max(0., min(a[3], b[3]) - max(a[1], b[1]))
    union = aa + bb - intersection
    return intersection / union if union > 0 else 0.


def box_scores(pred, target):
    overlap = iou(pred, target)
    valid = pred[2] > pred[0] and pred[3] > pred[1]
    cx, cy = (pred[0] + pred[2]) / 2, (pred[1] + pred[3]) / 2
    return {"IoU": overlap,
            **{f"ACC@{t}": float(overlap >= t and valid) for t in (.1, .3, .5, .7, .9)},
            "Center_ACC": float(valid and target[0] <= cx <= target[2] and target[1] <= cy <= target[3])}


def is_refcoco_rec(task):
    return isinstance(task, str) and task.startswith(
        ("refcoco_bbox_rec_", "refcoco+_bbox_rec_", "refcocog_bbox_rec_")
    )


def document_input_size(doc, min_pixels, max_pixels):
    width, height = doc.get("image_width"), doc.get("image_height")
    if width is None or height is None:
        width, height = doc["image"].size
    return qwen25_input_size(width, height, min_pixels, max_pixels)


def corrected_payload(doc, results, backbone, *, input_size=None):
    raw = results
    while isinstance(raw, (list, tuple)):
        raw = raw[0] if raw else ""
    pred, status = normalize_prediction(raw, backbone, input_size=input_size)
    return {"answer": doc["answer"], "pred": pred, "ann_id": doc["question_id"],
            "bbox": doc["bbox"], "grounding_protocol": PROTOCOL,
            "coordinate_space": "normalized_1000" if backbone == "qwen3_vl" else "processed_pixels",
            "input_size": list(input_size) if input_size else None, "parse_status": status}


def rescore_rows(rows, backbone, *, size_for_row=None):
    """Re-score saved predictions, preserving raw text and all non-RefCOCO fields."""
    if not rows or len({row['doc_id'] for row in rows}) != len(rows):
        raise ValueError("Empty samples or duplicate doc_id")
    corrected, invalid = [], 0
    totals = {metric: 0. for metric in METRICS}
    for row in rows:
        old = row['refcoco_IoU']
        doc = {'answer': old['answer'], 'question_id': old['ann_id'], 'bbox': old['bbox']}
        size = size_for_row(row) if size_for_row else None
        payload = corrected_payload(doc, row['filtered_resps'], backbone, input_size=size)
        invalid += payload['parse_status'] != 'valid'
        scores = box_scores(payload['pred'], payload['bbox'])
        for metric in METRICS:
            totals[metric] += scores[metric]
        corrected.append({**row, **{f'refcoco_{m}': dict(payload) for m in METRICS}})
    return corrected, {f'refcoco_{m},none': v / len(rows) for m, v in totals.items()}, invalid


def install_grounding_scorer(backbone, min_pixels, max_pixels):
    """Install once per process; only platform Qwen RefCOCO bbox tasks change.

    ConfigurableTask is patched rather than YAML's dynamically imported modules.
    Reinstalling updates model context without stacking wrappers.
    """
    from lmms_eval.api.task import ConfigurableTask
    current = ConfigurableTask.process_results
    original = getattr(current, "_grounding_original", current)

    @functools.wraps(original)
    def process_results(self, doc, results, *args, **kwargs):
        task = getattr(self.config, "task", "")
        if backbone not in {"qwen3_vl", "qwen25_vl"} or not is_refcoco_rec(task):
            return original(self, doc, results, *args, **kwargs)
        size = document_input_size(doc, min_pixels, max_pixels) if backbone == "qwen25_vl" else None
        payload = corrected_payload(doc, results, backbone, input_size=size)
        return {f"refcoco_{metric}": dict(payload) for metric in METRICS}

    process_results._grounding_original = original
    ConfigurableTask.process_results = process_results
