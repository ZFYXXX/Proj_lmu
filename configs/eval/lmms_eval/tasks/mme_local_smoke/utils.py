from lmms_eval.tasks.mme.utils import (
    mme_doc_to_text,
    mme_doc_to_visual,
    parse_pred_ans,
)


def mme_smoke_process_results(doc, results):
    predicted = parse_pred_ans(results[0])
    expected = str(doc["answer"]).lower().strip().replace(".", "")
    return {"mme_smoke_accuracy": float(predicted == expected)}


def mme_smoke_aggregate(results):
    if not results:
        return 0.0
    return sum(float(item) for item in results) / len(results)
