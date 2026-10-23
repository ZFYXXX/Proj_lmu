import ast
from pathlib import Path

import pytest
import torch

from mllm_comp.compressors.image.learned_visual_topk import (
    continuous_topk, hard_topk_indices, LearnedVisualTopK, LearnedVisualTopKSpec,
    save_selector, load_selector, SOURCE,
)


def test_official_operator_values_and_gradients():
    path = Path('/home/ZhouFengyu/fastv-repro/VisionSelector') / SOURCE['file']
    if not path.exists():
        pytest.skip('Optional official source not present')
    import hashlib
    assert hashlib.sha256(path.read_bytes()).hexdigest() == SOURCE['sha256']
    nodes = [n for n in ast.parse(path.read_text()).body
        if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name in {'TopK', '_find_ts'}]
    ns = {'torch': torch, 'Function': torch.autograd.Function, 'sigmoid': torch.sigmoid,
        'sigmoid_grad': torch.vmap(torch.vmap(torch.func.grad(torch.sigmoid)))}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), 'exec'), ns)
    x = torch.linspace(-3, 3, 40).reshape(2, 20).requires_grad_()
    y = x.detach().clone().requires_grad_()
    p, q = continuous_topk(x, 6), ns['TopK'].apply(y, 6)
    torch.testing.assert_close(p, q)
    coeff = torch.arange(40).reshape_as(x).float()
    (p * coeff).sum().backward(); (q * coeff).sum().backward()
    torch.testing.assert_close(x.grad, y.grad, rtol=1e-5, atol=1e-5)


def test_float64_gradcheck():
    torch.manual_seed(7)
    x = torch.randn(2, 11, dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(lambda v: continuous_topk(v, 4), (x,))


@pytest.mark.parametrize('k', [0, 1, 128, 2048])
def test_mass_and_boundary_gradients(k):
    x = torch.linspace(-10, 10, 2048).reshape(1, -1).requires_grad_()
    p = continuous_topk(x, k)
    assert abs(float(p.detach().sum()) - k) < .001
    (p * torch.arange(2048)).sum().backward()
    assert torch.isfinite(x.grad).all()


def test_ranking_unchanged_and_ties_stable():
    x = torch.tensor([[0., 0., 2., 1.]])
    assert hard_topk_indices(x[0], 3).tolist() == [0, 2, 3]
    assert hard_topk_indices(continuous_topk(x, 2)[0], 2).tolist() == [2, 3]


def test_soft_gradient_hard_count_and_no_feature_selection_gradient():
    module = LearnedVisualTopK(LearnedVisualTopKSpec(24, width=8, target_count=3))
    x = torch.randn(7, 24)
    soft = module(x, soft=True)
    assert soft['tokens'].shape == x.shape
    (soft['tokens'].square().mean() + torch.nn.functional.binary_cross_entropy(soft['soft_weights'], soft['hard_mask'])).backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in module.parameters())
    assert sum(float(p.grad.abs().sum()) for p in module.parameters()) > 0
    hard = module(x)
    assert len(hard['indices'].unique()) == len(hard['tokens']) == 3
    torch.testing.assert_close(hard['tokens'], x[hard['indices']])


def test_checkpoint_identity_and_no_overwrite(tmp_path):
    module = LearnedVisualTopK(LearnedVisualTopKSpec(24))
    folder = tmp_path / 'checkpoint'
    save_selector(module, folder, base_identity='base', provenance={'git_commit': 'test'})
    loaded = load_selector(folder, base_identity='base', device='cpu')
    x = torch.randn(130, 24)
    assert torch.equal(module(x)['indices'], loaded(x)['indices'])
    with pytest.raises(ValueError, match='different base'):
        load_selector(folder, base_identity='wrong', device='cpu')
    with pytest.raises(ValueError, match='already exists'):
        save_selector(module, folder, base_identity='base', provenance={})


@pytest.mark.parametrize('x,k', [(torch.tensor([[float('nan')]]), 1), (torch.ones(1,3), 4), (torch.ones(1,3), True)])
def test_invalid_inputs(x, k):
    with pytest.raises(ValueError):
        continuous_topk(x, k)
