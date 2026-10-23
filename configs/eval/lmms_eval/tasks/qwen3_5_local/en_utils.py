"""Expose upstream MMBench EN utilities to the local parquet task loader."""

from lmms_eval.tasks.mmbench.en_utils import (
    mmbench_aggregate_dev_results_static,
    mmbench_aggregate_dev_results_submission,
    mmbench_doc_to_text,
    mmbench_doc_to_visual,
    mmbench_process_results,
)

__all__ = [
    "mmbench_aggregate_dev_results_static",
    "mmbench_aggregate_dev_results_submission",
    "mmbench_doc_to_text",
    "mmbench_doc_to_visual",
    "mmbench_process_results",
]
