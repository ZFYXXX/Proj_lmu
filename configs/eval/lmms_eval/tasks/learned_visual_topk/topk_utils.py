"""Native task semantics; development subset is fixed by an external cohort."""
import json
import os
from pathlib import Path
from configs.eval.lmms_eval.tasks.qwen25_7b_projector_ratio.utils import (
    visual, chartqa_doc_to_text, chartqa_process_results, ocrbench_doc_to_text, ocrbench_process_results,
)


def chartqa_development(dataset):
    cohort = json.loads(Path(os.environ['MLLM_TOPK_COHORT']).read_text())
    return dataset.select(cohort['chartqa']['development_indices'])


def ocrbench_development(dataset):
    cohort = json.loads(Path(os.environ['MLLM_TOPK_COHORT']).read_text())
    return dataset.select(cohort['ocrbench']['development_indices'])
