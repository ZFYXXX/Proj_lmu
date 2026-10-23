from __future__ import annotations

from typing import Any


def doc_to_visual(doc: dict[str, Any], lmms_eval_specific_kwargs: Any = None) -> list[str]:
    del lmms_eval_specific_kwargs
    return [str(doc["video"])]


def doc_to_text(doc: dict[str, Any], lmms_eval_specific_kwargs: Any = None) -> str:
    del lmms_eval_specific_kwargs
    return str(doc["question"])


def doc_to_target(doc: dict[str, Any]) -> str:
    return str(doc["answer"])


def process_results(doc: dict[str, Any], results: list[str]) -> dict[str, Any]:
    del doc
    return {"smoke_nonempty": {"nonempty": bool(results and results[0].strip())}}


def aggregate_nonempty(results: list[dict[str, Any]]) -> float:
    if not results:
        return 0.0
    return 100.0 * sum(bool(result["nonempty"]) for result in results) / len(results)
