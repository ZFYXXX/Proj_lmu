import pytest
import torch

from mllm_comp.compressors.image.spatial_anchor_ablation import spatial_anchor_merge
from mllm_comp.compressors.comparison.importance_anchor_adapter import adapt_clusters


def test_importance_changes_first_anchor_and_covers_duplicate_grid():
    x = torch.ones(240, 8)
    p = torch.linspace(.01, .99, 240)
    a = spatial_anchor_merge(x, 12, 20, 128, importance=p)
    b = spatial_anchor_merge(x, 12, 20, 128, importance=p)
    assert a['anchors'] == b['anchors'] and 239 in a['anchors']
    assert len(a['members']) == 128 and all(a['members'])
    assert sorted(i for g in a['members'] for i in g) == list(range(240))
    assert all(anchor in group for anchor, group in zip(a['anchors'], a['members']))


def test_singleton_bypass_skips_mlp_but_keeps_merged_correction():
    calls = []
    def adapter(base):
        calls.append(len(base['tokens']))
        return {'tokens': base['tokens'] + 7}
    base = {'tokens': torch.arange(12).reshape(3, 4).float(), 'members': [[0], [1, 2], [3]]}
    result = adapt_clusters(adapter, base, True)
    assert calls == [1]
    assert torch.equal(result[[0, 2]], base['tokens'][[0, 2]])
    assert torch.equal(result[1], base['tokens'][1] + 7)
    assert torch.equal(adapt_clusters(adapter, base, False), base['tokens'] + 7)


def test_all_singletons_are_identity_and_invalid_importance_fails():
    x = torch.randn(12, 8)
    base = spatial_anchor_merge(x, 3, 4, 128, importance=torch.ones(12))
    def forbidden(_):
        raise AssertionError('Singleton-only input must not run the MLP')
    result = adapt_clusters(forbidden, base, True)
    assert torch.equal(result, x[base['indices']])
    with pytest.raises(ValueError):
        spatial_anchor_merge(x, 3, 4, importance=torch.zeros(12))
