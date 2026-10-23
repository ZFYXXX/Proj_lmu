from __future__ import annotations


import pytest

from scripts.diagnostics import validate_qwen3_vl_video_compression_checkpoint as target
from tests.fixtures.video_reports import generation_evidence as _generation


def _sample_evidence() -> dict[str, object]:
    representatives = tuple(range(0, 64, 2))
    source_mrope = tuple((index, index + 100, index + 200) for index in range(64))
    return {
        "baseline": _generation(108),
        "compressed": _generation(76),
        "activation_records": [{"source_count": 64, "result_count": 32}],
        "latest_splice": {
            "compression_applied": True,
            "source_placeholder_count": 64,
            "result_placeholder_count": 32,
            "source_placeholder_spans": ((10, 30), (35, 55), (60, 84)),
            "result_placeholder_spans": ((10, 20), (25, 35), (40, 52)),
            "layout_delta": {"source_count": 64, "result_count": 32},
            "source_video_grid_preserved": True,
            "source_video_grid_thw": (1, 16, 16),
            "result_grid_policy": "irregular_sparse_source_anchors",
            "representative_indices": representatives,
            "source_visual_mrope": source_mrope,
            "result_visual_mrope": tuple(
                source_mrope[index] for index in representatives
            ),
            "visual_pos_masks_compacted": True,
            "deepstack_branch_count": 3,
            "deepstack_result_shapes": ((32, 2048),) * 3,
            "shared_main_deepstack_layout": True,
        },
        "latest_generation": {
            "prefill_complete": True,
            "source_sequence_length": 108,
            "result_sequence_length": 76,
            "removed_count": 32,
            "decode_repairs": 1,
            "last_decode_record": {
                "input_attention_length": 109,
                "output_attention_length": 77,
                "already_compact": False,
            },
        },
        "source_grid_thw": (1, 16, 16),
        "source_grid_token_count": 64,
        "labels_absent": True,
        "source_placeholder_spans": ((10, 30), (35, 55), (60, 84)),
        "native_video_count": 1,
        "expected_deepstack_branches": 3,
    }


def test_stage_selects_exact_qwen3_generate_binding() -> None:
    stage = target._stage()

    assert stage.algorithm == "vqtoken"
    assert stage.variant == "vqtoken.video"
    assert stage.binding == "vqtoken.video.qwen3_vl"
    assert stage.execution_phase.value == "generate"


def test_vqtoken_qwen3_evidence_requires_interleaved_deepstack_contract() -> None:
    checks = target._validate_vqtoken_sample(**_sample_evidence())

    assert all(checks.values())
    assert checks["native_placeholder_spans_preserved"]
    assert checks["mrope_sparse_anchor_gather_exact"]
    assert checks["deepstack_result_counts_exact"]
    assert checks["visual_pos_masks_compacted"]


def test_vqtoken_qwen3_evidence_rejects_deepstack_drift() -> None:
    evidence = _sample_evidence()
    latest_splice = dict(evidence["latest_splice"])
    latest_splice["deepstack_result_shapes"] = ((32, 2048), (31, 2048), (32, 2048))
    evidence["latest_splice"] = latest_splice

    with pytest.raises(RuntimeError, match="sample evidence failed"):
        target._validate_vqtoken_sample(**evidence)


def test_vqtoken_qwen3_evidence_rejects_native_span_drift() -> None:
    evidence = _sample_evidence()
    latest_splice = dict(evidence["latest_splice"])
    latest_splice["source_placeholder_spans"] = ((10, 74),)
    evidence["latest_splice"] = latest_splice

    with pytest.raises(RuntimeError, match="sample evidence failed"):
        target._validate_vqtoken_sample(**evidence)


def test_vqtoken_qwen3_evidence_rejects_mrope_drift() -> None:
    evidence = _sample_evidence()
    latest_splice = dict(evidence["latest_splice"])
    result_mrope = list(latest_splice["result_visual_mrope"])
    result_mrope[4] = (999, 999, 999)
    latest_splice["result_visual_mrope"] = result_mrope
    evidence["latest_splice"] = latest_splice

    with pytest.raises(RuntimeError, match="sample evidence failed"):
        target._validate_vqtoken_sample(**evidence)


def test_vqtoken_qwen3_evidence_rejects_missing_decode_repair() -> None:
    evidence = _sample_evidence()
    latest_generation = dict(evidence["latest_generation"])
    latest_generation["decode_repairs"] = 0
    latest_generation["last_decode_record"] = None
    evidence["latest_generation"] = latest_generation

    with pytest.raises(RuntimeError, match="sample evidence failed"):
        target._validate_vqtoken_sample(**evidence)
