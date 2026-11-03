"""Native OK-VQA scoring; no answer extraction, cleaning or relaxed matching."""
from configs.eval.lmms_eval.tasks.qwen25_okvqa.utils import (
    process_docs, doc_to_visual, doc_to_text, process_results,
)
