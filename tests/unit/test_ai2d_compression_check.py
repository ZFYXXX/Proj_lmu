import pytest
from scripts.eval.lmms_eval.run_ai2d_compression_check import upstream_score


@pytest.mark.parametrize("prediction,expected", [("B", 1), ("B. blue", 1), ("C", 0), ("", 0), ("The answer is B", 0)])
def test_upstream_filter_does_not_invent_answer_extraction(prediction, expected):
    score, _ = upstream_score(prediction, {"options": ["red", "blue", "green"], "answer": "1"})
    assert score == expected
