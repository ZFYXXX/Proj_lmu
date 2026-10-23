from __future__ import annotations

from pathlib import Path

import pytest

from scripts.diagnostics import qwen_video_checkpoint_common as _qwen_video_checkpoint_common
from scripts.diagnostics import validate_qwen25_vl_video_compression_checkpoint as target
from tests.fixtures.video_reports import generation_evidence as _generation


def _sample_evidence() -> dict[str, object]:
    representatives = tuple(range(0, 64, 2))
    source_mrope = tuple((index, index + 100, index + 200) for index in range(64))
    return {
        "baseline": _generation(100),
        "compressed": _generation(68),
        "activation_records": [{"source_count": 64, "result_count": 32}],
        "latest_splice": {
            "compression_applied": True,
            "source_placeholder_count": 64,
            "result_placeholder_count": 32,
            "layout_delta": {"source_count": 64, "result_count": 32},
            "source_video_grid_preserved": True,
            "source_video_grid_thw": (1, 16, 16),
            "result_grid_policy": "irregular_sparse_source_anchors",
            "representative_indices": representatives,
            "source_visual_mrope": source_mrope,
            "result_visual_mrope": tuple(
                source_mrope[index] for index in representatives
            ),
        },
        "latest_generation": {
            "prefill_complete": True,
            "source_sequence_length": 100,
            "result_sequence_length": 68,
            "removed_count": 32,
            "decode_repairs": 1,
            "last_decode_record": {
                "input_attention_length": 101,
                "output_attention_length": 69,
                "already_compact": False,
            },
        },
        "source_grid_thw": (1, 16, 16),
        "source_grid_token_count": 64,
        "labels_absent": True,
        "video_span_count": 1,
    }


def test_conversation_freezes_qwen25_native_video_policy() -> None:
    conversation = _qwen_video_checkpoint_common._conversation(
        prompt="Question\nA. one\nB. two",
        video=Path("/dataset/sample.mp4"),
    )

    video = conversation[1]["content"][0]
    assert video == {
        "type": "video",
        "video": str(Path("/dataset/sample.mp4")),
        "fps": 1.0,
        "max_pixels": 128 * 28 * 28,
    }
    assert conversation[1]["content"][1]["text"] == "Question\nA. one\nB. two"


def test_stage_selects_exact_qwen25_generate_binding() -> None:
    stage = target._stage()

    assert stage.algorithm == "vqtoken"
    assert stage.variant == "vqtoken.video"
    assert stage.binding == "vqtoken.video.qwen25_vl"
    assert stage.execution_phase.value == "generate"


def test_vqtoken_qwen25_evidence_requires_grid_mrope_and_decode_repair() -> None:
    checks = target._validate_vqtoken_sample(**_sample_evidence())

    assert all(checks.values())
    assert checks["mrope_sparse_anchor_gather_exact"]
    assert checks["decode_repair_exercised"]


def test_vqtoken_qwen25_evidence_rejects_mrope_drift() -> None:
    evidence = _sample_evidence()
    latest_splice = dict(evidence["latest_splice"])
    result_mrope = list(latest_splice["result_visual_mrope"])
    result_mrope[4] = (999, 999, 999)
    latest_splice["result_visual_mrope"] = result_mrope
    evidence["latest_splice"] = latest_splice

    with pytest.raises(RuntimeError, match="sample evidence failed"):
        target._validate_vqtoken_sample(**evidence)


def test_vqtoken_qwen25_evidence_rejects_missing_decode_repair() -> None:
    evidence = _sample_evidence()
    latest_generation = dict(evidence["latest_generation"])
    latest_generation["decode_repairs"] = 0
    latest_generation["last_decode_record"] = None
    evidence["latest_generation"] = latest_generation

    with pytest.raises(RuntimeError, match="sample evidence failed"):
        target._validate_vqtoken_sample(**evidence)
