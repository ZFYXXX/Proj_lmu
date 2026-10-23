"""Read-only re-scoring of saved TextVQA results against all ten references."""
import argparse
import ast
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from mllm_comp.core.serialization import atomic_write_json


def consensus_score(prediction, answers, normalize):
    """Independent closed form of leave-one-annotator-out TextVQA accuracy."""
    if len(answers) != 10:
        raise ValueError("Exactly ten human references required")
    matches = sum(normalize(str(a)) == normalize(prediction) for a in answers)
    return (matches * min(1., max(0, matches-1)/3) + (10-matches) * min(1., matches/3))/10


def audit(prediction_root, output):
    import pyarrow.parquet as pq
    from lmms_eval.tasks.textvqa.utils import EvalAIAnswerProcessor, textvqa_process_results
    normalize = EvalAIAnswerProcessor()
    paths = sorted(Path("/home/YinPeiran/datasets/lmms_eval_datasets/textvqa/data").glob("validation-*.parquet"))
    docs = []
    for path in paths:
        docs.extend(pq.read_table(path, columns=["question_id", "image_id", "question", "answers"]).to_pylist())
    if len(docs) != 5000 or len({d["question_id"] for d in docs}) != 5000 or any(len(d["answers"]) != 10 for d in docs):
        raise ValueError("Unexpected validation documents/reference counts")
    tables, predictions = {}, {}
    files = sorted(prediction_root.glob("*/models__*/*samples*textvqa*.jsonl"))
    if len(files) != 5:
        raise ValueError(f"Expected five complete historical methods; got {len(files)}")
    for file in files:
        method = file.parents[1].name
        rows = [json.loads(line) for line in file.read_text().splitlines()]
        if len(rows) != 5000 or {r["doc_id"] for r in rows} != set(range(5000)):
            raise ValueError(f"Missing or duplicate document IDs: {method}")
        errors, score_sum = [], 0.
        diagnostic = Counter()
        output_rows = {}
        for row in rows:
            index = row["doc_id"]
            doc = docs[index]
            prediction = row["filtered_resps"]
            if isinstance(prediction, list):
                prediction, = prediction
            if not isinstance(prediction, str):
                raise ValueError("Expected one prediction string")
            if row["submission"]["question_id"] != doc["question_id"]:
                raise ValueError(f"Question/reference misalignment at {method}/{index}")
            expected_prompt = doc["question"].capitalize()+"\nAnswer the question using a single word or phrase."
            if row["input"] != expected_prompt:
                raise ValueError(f"Unexpected/gold-contaminated prompt at {method}/{index}")
            if ast.literal_eval(row["target"]) != doc["answers"]:
                raise ValueError(f"Saved target references differ at {method}/{index}")
            independent = consensus_score(prediction, doc["answers"], normalize)
            upstream = textvqa_process_results({"answers": list(doc["answers"]), "question_id": doc["question_id"]}, [prediction])["exact_match"]
            if max(abs(independent-upstream), abs(independent-row["exact_match"])) > 1e-12:
                errors.append({"doc_id": index, "stored": row["exact_match"], "upstream": upstream, "independent": independent})
            score_sum += independent
            diagnostic["empty_prediction"] += int(not prediction.strip())
            diagnostic["multi_line_prediction"] += int("\n" in prediction.strip())
            diagnostic["more_than_10_words"] += int(len(prediction.split()) > 10)
            # Diagnostic only: never use a substring match to inflate task accuracy.
            p = normalize(prediction)
            diagnostic["zero_score_with_reference_as_word_span"] += int(independent == 0 and any(
                normalize(a) and f" {normalize(a)} " in f" {p} " for a in doc["answers"]))
            output_rows[index] = {"prediction": prediction, "score": independent}
        predictions[method] = output_rows
        tables[method] = {"n_questions": len(rows), "rescored_accuracy": score_sum/len(rows),
            "score_mismatches": len(errors), "mismatch_examples": errors[:20], "format_diagnostics": dict(diagnostic),
            "predictions_sha256": hashlib.sha256(file.read_bytes()).hexdigest()}
    cases = []
    for method in predictions:
        if method == "baseline":
            continue
        lost = [i for i in range(5000) if predictions["baseline"][i]["score"] >= .9 and predictions[method][i]["score"] == 0]
        lost.sort(key=lambda i: hashlib.sha256(f"textvqa-audit:{i}".encode()).hexdigest())
        cases.extend({"method": method, "doc_id": i, **docs[i], "baseline": predictions["baseline"][i],
                      "compressed": predictions[method][i]} for i in lost[:20])
    result = {"status": "complete", "dataset": "TextVQA validation", "n_questions": len(docs),
              "n_image_ids": len({d["image_id"] for d in docs}), "answers_per_question": 10,
              "score_definition": "Official leave-one-annotator-out consensus soft accuracy, NOT single-reference exact match",
              "normalization": "Upstream EvalAIAnswerProcessor; consensus aggregation independently recomputed in closed form",
              "prompt": "Question plus short-answer instruction; OCR strings are NOT provided", "methods": tables,
              "limitations": ["Re-scoring saved predictions does not verify historical image pixel/processor identity or KV-cache behavior.",
                              "Historical logs do not include generated token IDs/EOS, so generation truncation cannot be fully audited offline.",
                              "Answer NLL uses one deterministic target and is not the native ten-reference task score."]}
    output.mkdir(parents=True, exist_ok=True)
    atomic_write_json(output / "scoring_audit.json", result)
    atomic_write_json(output / "failure_cases.json", cases)
    lines = ["# TextVQA saved-prediction scoring audit", "", "5,000 questions; ten references per question. No inference or training rerun.", "",
             "| Method | Re-scored native accuracy | Score mismatches | Empty | >10 words |", "|---|---:|---:|---:|---:|"]
    for name, table in tables.items():
        d = table["format_diagnostics"]
        lines.append(f"| {name} | {100*table['rescored_accuracy']:.3f}% | {table['score_mismatches']} | {d['empty_prediction']} | {d['more_than_10_words']} |")
    lines += ["", "The exact_match field name is misleading here: the actual metric is ten-reference consensus soft accuracy.", "",
              "## Limits", "", *["- "+x for x in result["limitations"]]]
    (output / "report.md").write_text("\n".join(lines)+"\n", encoding="utf-8")
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--prediction-root", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    args = p.parse_args()
    result = audit(args.prediction_root, args.output)
    print(json.dumps(result, ensure_ascii=False, indent=2))
