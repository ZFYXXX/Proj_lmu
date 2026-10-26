"""Precommitted independent-image cohort, native prompts and scorers unchanged."""
import json
import os
from pathlib import Path


def subset(dataset,task):
    data=json.loads(Path(os.environ['MLLM_SELECTOR_MECHANISM_COHORT']).read_text())
    return dataset.select(data['tasks'][task]['source_doc_ids'])


def chartqa_subset(dataset):return subset(dataset,'chartqa')
def ocrbench_subset(dataset):return subset(dataset,'ocrbench')
def ai2d_subset(dataset):return subset(dataset,'ai2d')
