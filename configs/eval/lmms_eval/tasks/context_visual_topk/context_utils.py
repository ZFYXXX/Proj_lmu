"""Unmodified native scoring; externally pinned image-hash development splits."""
import json
import os
from pathlib import Path
from configs.eval.lmms_eval.tasks.qwen25_7b_projector_ratio.utils import (
    visual, chartqa_doc_to_text, chartqa_process_results, ocrbench_doc_to_text, ocrbench_process_results,
    ai2d_doc_to_text, ai2d_doc_to_target, ai2d_results, MultiChoiceRegexFilter,
)


def subset(dataset, task):
    cohort = json.loads(Path(os.environ['MLLM_TOPK_COHORT']).read_text())
    return dataset.select(cohort[task]['development_indices'])


def chartqa_development(dataset):
    return subset(dataset, 'chartqa')


def ocrbench_development(dataset):
    return subset(dataset, 'ocrbench')


def ai2d_development(dataset):
    return subset(dataset, 'ai2d')
