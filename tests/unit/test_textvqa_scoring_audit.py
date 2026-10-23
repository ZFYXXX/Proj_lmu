import pytest
from scripts.analysis.audit_textvqa_scoring import consensus_score


@pytest.mark.parametrize("count,expected", [(0,0), (1,.3), (2,.6), (3,.9), (4,1), (10,1)])
def test_independent_consensus(count, expected):
    assert consensus_score("yes", ["yes"]*count+["no"]*(10-count), str.casefold) == pytest.approx(expected)


def test_reject_single_reference_shortcut():
    with pytest.raises(ValueError):
        consensus_score("yes", ["yes"], str.casefold)
