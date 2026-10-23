from __future__ import annotations

import pytest

from mllm_comp.analysis.common.visual_token_phenomena import (
    answer_logit_metrics,
    answer_predictor_positions,
    average_ranks,
    jaccard,
    safe_budget,
    spatial_coverage,
    spearman_correlation,
    top_fraction_indices,
)
from mllm_comp.analysis.probes.visual_token_phenomena import (
    VisualTokenPhenomena,
    _compact_layer_state,
)


def test_answer_predictor_positions_apply_causal_shift() -> None:
    torch = pytest.importorskip("torch")
    labels = torch.tensor([[-100, -100, 7, 8]])

    positions, targets = answer_predictor_positions(labels)

    assert positions == [1, 2]
    assert targets == [7, 8]


def test_answer_logit_metrics_detect_answer_flip_and_nll_increase() -> None:
    torch = pytest.importorskip("torch")
    teacher = torch.zeros(1, 3, 4)
    student = teacher.clone()
    teacher[0, 1, 2] = 5
    student[0, 1, 1] = 5

    report = answer_logit_metrics(
        teacher,
        student,
        predictor_positions=[1],
        targets=[2],
    )

    assert report["answer_flip"] is True
    assert report["answer_nll_delta"] > 0
    assert report["teacher_to_candidate_answer_kl"] > 0


def test_spearman_uses_average_tie_ranks() -> None:
    assert average_ranks([1, 1, 3]) == pytest.approx([1.5, 1.5, 3.0])
    assert spearman_correlation([1, 2, 3], [10, 20, 30]) == pytest.approx(1.0)
    assert spearman_correlation([1, 1, 1], [10, 20, 30]) is None


def test_top_fraction_and_spatial_coverage_are_deterministic() -> None:
    assert top_fraction_indices([1, 4, 4, 2], 0.5) == (1, 2)
    assert jaccard((0, 1), (1, 2)) == pytest.approx(1 / 3)
    report = spatial_coverage(
        [0, 3],
        [[0, 0], [0, 3], [3, 0], [3, 3]],
        height=4,
        width=4,
        bins=2,
    )
    assert report["coarse_bin_coverage"] == pytest.approx(0.5)
    assert report["maximum_centroid_hole_normalized"] <= 1.0


def test_safe_budget_requires_all_three_gates() -> None:
    rows = [
        {
            "budget": 32,
            "answer_nll_relative_delta": 0.01,
            "teacher_to_candidate_answer_kl": 0.01,
            "answer_flip": True,
        },
        {
            "budget": 64,
            "answer_nll_relative_delta": 0.04,
            "teacher_to_candidate_answer_kl": 0.03,
            "answer_flip": False,
        },
    ]
    assert safe_budget(rows) == 64


def test_probe_config_rejects_unknown_intervention() -> None:
    with pytest.raises(ValueError, match="Unsupported causal interventions"):
        VisualTokenPhenomena().validate_config(
            {
                "name": "visual_token_phenomena",
                "layers": [0],
                "interventions": ["delete_everything"],
            }
        )


def test_physical_delete_compactor_identity_keeps_hidden_and_positions() -> None:
    torch = pytest.importorskip("torch")
    hidden = torch.arange(8, dtype=torch.float32).reshape(1, 4, 2)
    positions = torch.arange(12, dtype=torch.long).reshape(3, 4)
    compact = _compact_layer_state(
        hidden=hidden,
        position_ids=positions,
        visual_positions=[1, 2],
        members=((0,), (1,)),
        operator="retain",
        token_scores=torch.ones(2),
        budget=2,
    )

    assert torch.equal(compact["hidden"], hidden)
    assert torch.equal(compact["position_ids"][:, 0], positions)
    assert compact["source_to_compact"] == {0: 0, 1: 1, 2: 2, 3: 3}


def test_physical_delete_preserves_text_and_original_three_axis_positions() -> None:
    torch = pytest.importorskip("torch")
    hidden = torch.arange(8, dtype=torch.float32).reshape(1, 4, 2)
    positions = torch.tensor([[0, 1, 1, 9], [0, 2, 4, 9], [0, 3, 7, 9]])
    compact = _compact_layer_state(
        hidden=hidden, position_ids=positions, visual_positions=[1, 2],
        members=((1,),), operator="retain", token_scores=torch.ones(2), budget=1,
    )
    assert torch.equal(compact["hidden"], hidden[:, [0, 2, 3]])
    assert torch.equal(compact["position_ids"][:, 0], positions[:, [0, 2, 3]])
    assert compact["source_to_compact"] == {0: 0, 2: 1, 3: 2}
    with pytest.raises(ValueError, match="distinct"):
        _compact_layer_state(hidden=hidden, position_ids=positions, visual_positions=[1, 2],
                             members=((1,), (1,)), operator="retain", token_scores=torch.ones(2), budget=2)

