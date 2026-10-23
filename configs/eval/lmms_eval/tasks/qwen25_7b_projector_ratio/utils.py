"""Offline parquet media decoding with native lmms-eval task semantics."""
from collections.abc import Mapping
from io import BytesIO
import re
from PIL import Image
from lmms_eval.tasks.chartqa import utils as chart
from lmms_eval.tasks.ocrbench import utils as ocr
from lmms_eval.tasks.docvqa import utils as docvqa
from lmms_eval.tasks.ai2d import utils as ai2d
from lmms_eval.api.metrics import anls, exact_match_hf_evaluate


def decoded(doc):
    doc = dict(doc)
    data = doc["image"]
    if isinstance(data, Mapping):
        with Image.open(BytesIO(data["bytes"]) if data.get("bytes") else data["path"]) as image:
            doc["image"] = image.convert("RGB")
    return doc


def visual(doc):
    return [decoded(doc)["image"].convert("RGB")]


chartqa_doc_to_text = chart.chartqa_doc_to_text
chartqa_process_results = chart.chartqa_process_results
ocrbench_doc_to_text = ocr.ocrbench_doc_to_text
docvqa_doc_to_text = docvqa.docvqa_doc_to_text
ai2d_doc_to_text = ai2d.ai2d_doc_to_text
ai2d_doc_to_target = ai2d.ai2d_doc_to_target
MultiChoiceRegexFilter = ai2d.MultiChoiceRegexFilter


def ocrbench_process_results(doc, results):
    # Same per-question official score. Numeric mean equals official sum/1000
    # on the full split without its mutable module-level accumulation.
    native = ocr.ocrbench_process_results(doc, results)
    return {"ocrbench_accuracy": float(native["ocrbench_accuracy"]["score"])}


def docvqa_results(doc, results):
    return anls(doc["answers"], results)


def ai2d_results(doc, results):
    target = chr(65 + int(doc["answer"]))
    primary = exact_match_hf_evaluate(references=[target], predictions=results,
                                     ignore_case=True, ignore_punctuation=True)["exact_match"]
    match = re.match(r"^\s*([A-D])(?:[.\s)]|$)", results[0])
    return {"exact_match": primary,
            "leading_option_accuracy": float(bool(match) and match.group(1) == target)}
