"""Local media decoding; native MMStar/POPE scores plus audited POPE diagnostics.

Do not edit the shared evaluator. Its POPE yes_ratio counts ground truth instead
of predictions, and its recall drops invalid outputs. Correct these secondary
metrics here while retaining its exact-answer accuracy unchanged.
"""
from collections import Counter, defaultdict
from collections.abc import Mapping
from io import BytesIO
import os

from PIL import Image
from lmms_eval.tasks.mmstar import utils as mmstar
from lmms_eval.tasks.pope import utils as pope


def image_from_doc(doc):
    data = doc['image']
    if isinstance(data, Image.Image):
        return data.convert('RGB')
    if isinstance(data, Mapping):
        data = BytesIO(data['bytes']) if data.get('bytes') else data['path']
    elif isinstance(data, (bytes, bytearray, memoryview)):
        data = BytesIO(bytes(data))
    with Image.open(data) as image:
        return image.convert('RGB')


def visual(doc):
    return [image_from_doc(doc)]


def smoke_indices(labels, count=8):
    """Round-robin strata in source order, without reading targets or scores."""
    groups = defaultdict(list)
    for index, label in enumerate(labels):
        groups[label].append(index)
    result = []
    offset = 0
    while len(result) < min(count, len(labels)):
        for indices in groups.values():
            if offset < len(indices):
                result.append(indices[offset])
                if len(result) == min(count, len(labels)):
                    break
        offset += 1
    return result


def process_docs(dataset):
    if os.environ.get('MLLM_EXTRA_SMOKE') == '1':
        return dataset.select(smoke_indices(dataset['category']))
    return dataset


mmstar_doc_to_text = mmstar.mmstar_doc_to_text
mmstar_aggregate_results = mmstar.mmstar_aggregate_results
pope_doc_to_text = pope.pope_doc_to_text


def mmstar_results(doc, results):
    native = mmstar.mmstar_process_results(doc, results)
    return {**native, 'mmstar_micro_accuracy': float(native['average']['score'])}


def pope_results(doc, results):
    native = pope.pope_process_results(doc, results)
    record = native['pope_accuracy']
    return {'pope_accuracy': float(record['score']),
            **{name: record for name in ('pope_precision', 'pope_recall',
               'pope_f1_score', 'pope_yes_ratio', 'pope_invalid_ratio')}}


def pope_statistics(results):
    if not results:
        raise ValueError('Empty POPE results')
    if any(r['ground_truth'] not in ('yes', 'no') for r in results):
        raise ValueError('Invalid POPE ground truth')
    tp = sum(r['ground_truth'] == 'yes' and r['prediction'] == 'yes' for r in results)
    predicted_yes = sum(r['prediction'] == 'yes' for r in results)
    actual_yes = sum(r['ground_truth'] == 'yes' for r in results)
    precision = tp / predicted_yes if predicted_yes else 0.
    recall = tp / actual_yes if actual_yes else 0.
    return {'pope_accuracy': sum(r['prediction'] == r['ground_truth'] for r in results) / len(results),
            'pope_precision': precision, 'pope_recall': recall,
            'pope_f1_score': 2 * precision * recall / (precision + recall) if precision + recall else 0.,
            'pope_yes_ratio': predicted_yes / len(results),
            'pope_invalid_ratio': sum(r['prediction'] not in ('yes', 'no') for r in results) / len(results)}


def pope_precision(results):
    return pope_statistics(results)['pope_precision']


def pope_recall(results):
    return pope_statistics(results)['pope_recall']


def pope_f1(results):
    return pope_statistics(results)['pope_f1_score']


def pope_yes_ratio(results):
    return pope_statistics(results)['pope_yes_ratio']


def pope_invalid_ratio(results):
    return pope_statistics(results)['pope_invalid_ratio']


def macro_score(records):
    """The installed MMStar scorer weights its detailed categories equally."""
    totals, counts = Counter(), Counter()
    for record in records:
        category = record['l2_category']
        totals[category] += record['score']
        counts[category] += 1
    if not counts:
        raise ValueError('Empty MMStar results')
    return sum(totals[c] / counts[c] for c in counts) / len(counts)
