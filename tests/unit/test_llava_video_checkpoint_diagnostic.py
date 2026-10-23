from __future__ import annotations

import pytest

from scripts.diagnostics import validate_llava_video_compression_checkpoint as target
from scripts.diagnostics import video_checkpoint_common as _video_checkpoint_common
from tests.fixtures.video_reports import generation_evidence as _generation


def _document(doc_id: int) -> dict[str, object]:
    expected = _video_checkpoint_common.FIXED_NEXTQA_EXPECTATIONS[doc_id]
    return {
        **expected,
        "question": "Which option is visible?",
        "a0": "zero",
        "a1": "one",
        "a2": "two",
        "a3": "three",
        "a4": "four",
    }


def test_fixed_nextqa_document_freezes_identity_prompt_and_target() -> None:
    document = _video_checkpoint_common._normalise_fixed_document(_document(442), doc_id=442)

    assert document["target"] == "D"
    assert document["prompt"] == (
        "Which option is visible?\n" "A. zero\nB. one\nC. two\nD. three\nE. four"
    )


def test_fixed_nextqa_document_rejects_dataset_drift() -> None:
    document = _document(28)
    document["video"] = "different"

    with pytest.raises(RuntimeError, match="identity changed"):
        _video_checkpoint_common._normalise_fixed_document(document, doc_id=28)


@pytest.mark.parametrize(
    ("response", "expected"),
    (("(B)", "B"), ("answer D.", "D"), ("A or C; final C", "C"), ("none", None)),
)
def test_choice_parser_is_deterministic(response: str, expected: str | None) -> None:
    assert _video_checkpoint_common._parse_choice(response) == expected


def test_decoder_summary_requires_and_checks_prefill_alignment() -> None:
    summary = _video_checkpoint_common._decoder_summary(
        [
            {
                "hidden_shape": [1, 41, 896],
                "query_length": 41,
                "attention_mask_shape": [1, 1, 41, 41],
                "position_ids_shape": [1, 41],
                "cache_position_shape": [41],
                "has_past_key_value": False,
            },
            {
                "hidden_shape": [1, 1, 896],
                "query_length": 1,
                "attention_mask_shape": [1, 1, 1, 42],
                "position_ids_shape": [1, 1],
                "cache_position_shape": [1],
                "has_past_key_value": True,
            },
        ]
    )

    assert summary["prefill"]["query_length"] == 41
    assert summary["decode_query_lengths"] == [1]
    assert all(summary["invariants"].values())


def test_vqtoken_sample_evidence_requires_exact_decoder_reduction() -> None:
    checks = target._validate_vqtoken_sample(
        baseline=_generation(140),
        compressed=_generation(108),
        activation_records=[{"source_count": 64, "result_count": 32}],
        latest_splice={
            "compression_applied": True,
            "source_patch_count": 64,
            "result_patch_count": 32,
            "structural_source_positions": [1, 2],
            "structural_result_positions": [1, 2],
        },
        labels_absent=True,
        placeholder_count=1,
    )

    assert all(checks.values())


def test_vqtoken_sample_evidence_rejects_sequence_drift() -> None:
    with pytest.raises(RuntimeError, match="sample evidence failed"):
        target._validate_vqtoken_sample(
            baseline=_generation(140),
            compressed=_generation(109),
            activation_records=[{"source_count": 64, "result_count": 32}],
            latest_splice={
                "compression_applied": True,
                "source_patch_count": 64,
                "result_patch_count": 32,
                "structural_source_positions": [1, 2],
                "structural_result_positions": [1, 2],
            },
            labels_absent=True,
            placeholder_count=1,
        )
