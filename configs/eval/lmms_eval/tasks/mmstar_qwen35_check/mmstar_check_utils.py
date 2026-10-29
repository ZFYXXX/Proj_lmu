"""MMStar Qwen3.5 format validation: no guessing letters from prose."""
import json
import os
import re
from pathlib import Path
from configs.eval.lmms_eval.tasks.projector_extra.extra_utils import image_from_doc

SUFFIX = '\nAnswer with exactly one letter: A, B, C, or D. Do not explain.'


def visual(doc):
    return [image_from_doc(doc)]


def prompt(doc, lmms_eval_specific_kwargs=None):
    return doc['question'].strip() + SUFFIX


def process_docs(dataset):
    path = os.environ.get('MMSTAR_Q35_CHECK_INDICES')
    return dataset.select(json.loads(Path(path).read_text())['indices']) if path else dataset


def extract(response):
    """Accept only a whole-answer letter or a single-field JSON answer."""
    text = response.strip()
    match = re.fullmatch(r'(?:([A-Da-d])|\(([A-Da-d])\))[.!]?', text)
    if match:
        return (match.group(1) or match.group(2)).upper()
    try:
        value = json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return None
    if isinstance(value, dict) and set(value) == {'answer'} and isinstance(value['answer'], str):
        letter = value['answer'].strip().upper()
        return letter if letter in ('A', 'B', 'C', 'D') else None
    return None


def results(doc, output):
    answer = extract(output[0])
    return {'strict_accuracy': float(answer == doc['answer'].strip().upper()),
            'valid_answer': float(answer is not None)}
