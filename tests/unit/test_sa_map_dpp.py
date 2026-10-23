"""DPP degeneracy regression and an independent determinant oracle."""

import pytest

torch = pytest.importorskip("torch")

from mllm_comp.compressors.projector_level.similarity_attention_merge_prune import (
    _greedy_attention_dpp_indices,
)


@pytest.mark.parametrize("keep", [2, 4, 8, 12])
def test_repeated_features_never_reselect_a_candidate(keep):
    features = torch.ones(16, 8)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        selected = _greedy_attention_dpp_indices(
            features, torch.ones(16), keep_count=keep, attention_power=1.0, jitter=1e-6,
        )
    assert len(selected.unique()) == keep


def test_selection_matches_greedy_principal_determinants():
    features = torch.tensor([[1., 0., 0.], [0., 1., 0.], [0.3, 0.2, 1.], [1., 1., 0.2]])
    importance = torch.tensor([1., 0.9, 1.2, 0.5])
    basis = torch.nn.functional.normalize(features.double(), dim=1) * importance.double()[:, None]
    kernel = basis @ basis.T
    expected = []
    for _ in range(3):
        choices = [i for i in range(4) if i not in expected]
        choice = max(choices, key=lambda i: torch.linalg.det(kernel[expected + [i]][:, expected + [i]]).item())
        expected.append(choice)
    actual = _greedy_attention_dpp_indices(
        features, importance, keep_count=3, attention_power=1., jitter=1e-6,
    )
    assert actual.tolist() == expected
