"""Local-parquet GQA task helpers for the Qwen3.8 experiment."""

from __future__ import annotations

from functools import lru_cache

from datasets import load_dataset


_GQA_IMAGES = (
    "/home/YinPeiran/datasets/lmms_eval_datasets/GQA/"
    "testdev_all_images/testdev-00000-of-00001.parquet"
)


@lru_cache(maxsize=1)
def _gqa_images_by_id():
    """Load the small local testdev image table once, only when a sample is used."""

    images = load_dataset(
        "parquet",
        data_files=_GQA_IMAGES,
        split="train",
    )
    return {row["id"]: row["image"].convert("RGB") for row in images}


def gqa_doc_to_visual(doc):
    try:
        image = _gqa_images_by_id()[doc["imageId"]]
    except KeyError as exc:
        raise ValueError(f"No local GQA image for imageId={doc['imageId']!r}.") from exc
    return [image]


def gqa_doc_to_text(doc, lmms_eval_specific_kwargs):
    pre_prompt = lmms_eval_specific_kwargs["pre_prompt"]
    post_prompt = lmms_eval_specific_kwargs["post_prompt"]
    return f"{pre_prompt}{doc['question']}{post_prompt}"
