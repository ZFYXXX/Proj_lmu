"""Image-level sampling before expensive multimodal preprocessing."""

import hashlib
import json
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

from mllm_comp.analysis.common.visual_token_phenomena import media_fingerprint


def load_excluded_images(paths: list[str]) -> tuple[set[str], list[dict[str, str]]]:
    excluded = set()
    sources = []
    for value in paths:
        path = Path(value)
        raw = path.read_bytes()
        sources.append({"path": str(path.resolve()), "sha256": hashlib.sha256(raw).hexdigest()})
        for line in raw.splitlines():
            row = json.loads(line)
            if row.get("image_fingerprint"):
                excluded.add(str(row["image_fingerprint"]))
    return excluded, sources


def sample_unique_image_docs(
    docs: Iterable[tuple[Any, Any]], *, doc_to_visual: Callable,
    count: int, seed: int, excluded: set[str],
) -> tuple[list[tuple[Any, Any]], dict[str, Any]]:
    """Bottom-k SHA256 image sample; answers and model outcomes are never read.

    A bounded reservoir stores at most count decoded documents. Among repeated
    questions for a selected image, a separate hash chooses one question.
    """
    if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
        raise ValueError("unique_images_per_task must be a positive integer.")
    selected: dict[str, tuple[str, str, Any, Any]] = {}
    seen = set()
    scanned = 0
    for doc_id, doc in docs:
        scanned += 1
        fingerprint = media_fingerprint(doc_to_visual(doc), doc)
        if fingerprint in excluded:
            continue
        seen.add(fingerprint)
        image_key = hashlib.sha256(f"{seed}:image:{fingerprint}".encode()).hexdigest()
        question_key = hashlib.sha256(f"{seed}:question:{fingerprint}:{doc_id}".encode()).hexdigest()
        if fingerprint in selected:
            if question_key < selected[fingerprint][1]:
                selected[fingerprint] = (image_key, question_key, doc_id, doc)
        elif len(selected) < count:
            selected[fingerprint] = (image_key, question_key, doc_id, doc)
        else:
            worst = max(selected, key=lambda key: (selected[key][0], key))
            if (image_key, fingerprint) < (selected[worst][0], worst):
                del selected[worst]
                selected[fingerprint] = (image_key, question_key, doc_id, doc)
    if len(selected) != count:
        raise ValueError(f"Need {count} independent eligible images; found {len(selected)} in {scanned} documents.")
    values = sorted(selected.items(), key=lambda item: (item[1][0], item[0]))
    return [(value[2], value[3]) for _, value in values], {
        "policy": "bottom_k_sha256_image_then_question", "seed": seed,
        "scanned_documents": scanned, "eligible_unique_images": len(seen),
        "selected_images": count, "excluded_image_count": len(excluded),
        "selected_fingerprints": [fingerprint for fingerprint, _ in values],
        "selection_reads_answers": False,
    }
