import torch

from mllm_comp.compressors.comparison.importance_anchor_adapter import importance_pool


def test_pool_uniform_zero_mass_and_singleton_preserve_contents():
    features = torch.tensor([[1., 2.], [3., 8.], [7., 9.], [11., 4.], [13., 6.]])
    result, stats = importance_pool(features, [[0, 1], [2], [3, 4]], torch.tensor([.5, .5, .9, 0., 0.]))
    torch.testing.assert_close(result, torch.stack([features[:2].mean(0), features[2], features[3:].mean(0)]))
    assert [s['uniform_fallback'] for s in stats] == [False, False, True]


def test_pool_weighted_value_and_probability_gradient():
    features = torch.tensor([[1., 2.], [5., 6.]])
    probabilities = torch.tensor([.25, .75], requires_grad=True)
    result, stats = importance_pool(features, [[0, 1]], probabilities)
    torch.testing.assert_close(result, torch.tensor([[4., 5.]]))
    result.sum().backward()
    assert torch.isfinite(probabilities.grad).all() and (probabilities.grad.abs() > 0).all()
    assert abs(stats[0]['weights_sum'] - 1) < 1e-6
