import pytest
import torch

from mllm_comp.compressors.image.spatial_anchor_ablation import spatial_anchor_merge


def test_only_assignment_changes_not_anchor_selection():
    x = torch.zeros(12, 2)
    x[:, 1] = 1
    x[0] = torch.tensor([1., 0.])
    importance = torch.zeros(12)
    importance[0], importance[11] = 1., .9
    base = spatial_anchor_merge(x, 1, 12, 2, importance=importance)
    local = spatial_anchor_merge(x, 1, 12, 2, importance=importance, assignment_neighbors=1)
    assert set(base['anchors']) == set(local['anchors']) == {0, 11}
    assigned = lambda result, token: next(a for a,g in zip(result['anchors'],result['members']) if token in g)
    assert assigned(base, 1) == 11 and assigned(local, 1) == 0
    assert local['assignment_changed_count'] > 0
    assert local['assigned_anchor_grid_distance_mean'] < local['unrestricted_anchor_grid_distance_mean']


def test_local4_exact_partition_and_nearest_candidate_constraint():
    generator = torch.Generator().manual_seed(42)
    x = torch.randn(240, 12, generator=generator)
    importance = torch.rand(240, generator=generator)
    base = spatial_anchor_merge(x, 12, 20, 36, importance=importance)
    local = spatial_anchor_merge(x, 12, 20, 36, importance=importance, assignment_neighbors=4)
    assert set(base['anchors']) == set(local['anchors'])
    assert len(local['tokens']) == len(local['members']) == 36
    assert sorted(i for group in local['members'] for i in group) == list(range(240))
    anchors = sorted(local['anchors'])
    for anchor, group, value in zip(local['anchors'], local['members'], local['tokens']):
        assert anchor in group
        torch.testing.assert_close(value, x[group].float().mean(0))
        for token in group:
            nearest = sorted(anchors, key=lambda a: ((token//20-a//20)**2+(token%20-a%20)**2, a))[:4]
            assert anchor in nearest


@pytest.mark.parametrize('neighbors', [0, 3, 100])
def test_disabled_or_all_candidates_reproduce_original(neighbors):
    x = torch.randn(12, 5)
    original = spatial_anchor_merge(x, 3, 4, 3)
    other = spatial_anchor_merge(x, 3, 4, 3, assignment_neighbors=neighbors)
    for key in ['members', 'anchors', 'indices', 'partition_hash']:
        assert original[key] == other[key]
    assert torch.equal(original['tokens'], other['tokens'])


@pytest.mark.parametrize('bad', [True, -1, 1.5])
def test_invalid_neighbors_fail(bad):
    with pytest.raises(ValueError, match='assignment_neighbors'):
        spatial_anchor_merge(torch.randn(12, 4), 3, 4, assignment_neighbors=bad)


def test_native_local_assignment_generation_keeps_budget_and_positions():
    from tests.unit.test_projector_comparison_evaluation import tiny_state
    from mllm_comp.analysis.common.compression_training_check import AnchorAdapterSpec, create_anchor_adapter
    from mllm_comp.compressors.image.context_visual_topk import ContextVisualTopK, ContextVisualTopKSpec
    from mllm_comp.compressors.comparison.importance_anchor_adapter import QwenImportanceAnchorAdapter
    bundle,state,_,_ = tiny_state()
    session = QwenImportanceAnchorAdapter(bundle['model'],
        {'retention_ratio': .5, 'importance_gamma': 1., 'assignment_neighbors': 1},
        adapter_module=create_anchor_adapter(AnchorAdapterSpec(24, 4)),
        selector_module=ContextVisualTopK(ContextVisualTopKSpec(24, width=4, context_mode='image'))).install()
    try:
        with torch.no_grad():
            bundle['model'].generate(**state['prompt_inputs'], max_new_tokens=3, min_new_tokens=3,
                                    do_sample=False, pad_token_id=0)
        record = session.runtime_metadata()['selection']
        assert record['source_count'] == 6 and record['result_count'] == 3
        assert record['assignment_neighbors'] == 1 and record['decode_position_verified']
    finally:
        session.uninstall()


def test_regret_fallback_repairs_bad_local_match_without_changing_anchors():
    x = torch.zeros(12, 2)
    x[:, 1] = 1
    x[0] = torch.tensor([1., 0.])
    importance = torch.zeros(12)
    importance[0], importance[11] = 1., .9
    global_result = spatial_anchor_merge(x, 1, 12, 2, importance=importance)
    hard = spatial_anchor_merge(x, 1, 12, 2, importance=importance, assignment_neighbors=1)
    guarded = spatial_anchor_merge(x, 1, 12, 2, importance=importance,
                                   assignment_neighbors=1, assignment_max_regret=.1)
    assert guarded['members'] == global_result['members'] != hard['members']
    assert set(guarded['anchors']) == set(hard['anchors'])
    assert 1 in guarded['global_fallback_indices']
    assert 0 not in guarded['global_fallback_indices'] and 11 not in guarded['global_fallback_indices']
    assert guarded['global_fallback_count'] == len(guarded['global_fallback_indices'])
    torch.testing.assert_close(guarded['tokens'], global_result['tokens'])


def test_large_regret_budget_matches_hard_locality_and_exact_budget():
    torch.manual_seed(42)
    x = torch.randn(240, 12)
    hard = spatial_anchor_merge(x, 12, 20, 36, assignment_neighbors=4)
    guarded = spatial_anchor_merge(x, 12, 20, 36, assignment_neighbors=4, assignment_max_regret=2.)
    assert hard['partition_hash'] == guarded['partition_hash']
    assert guarded['global_fallback_count'] == 0 and len(guarded['tokens']) == 36
    for group, value in zip(guarded['members'], guarded['tokens']):
        torch.testing.assert_close(value, x[group].mean(0))


@pytest.mark.parametrize('bad', [True, -.01, 2.01, float('nan'), float('inf'), '0.1'])
def test_invalid_regret_fails(bad):
    with pytest.raises(ValueError, match='assignment_max_regret'):
        spatial_anchor_merge(torch.randn(12, 4), 3, 4, assignment_neighbors=1, assignment_max_regret=bad)


def test_regret_requires_local_cosine_assignment():
    with pytest.raises(ValueError, match='requires'):
        spatial_anchor_merge(torch.randn(12, 4), 3, 4, assignment_max_regret=.1)
    with pytest.raises(ValueError, match='requires'):
        spatial_anchor_merge(torch.randn(12, 4), 3, 4, beta=.1,
                             assignment_neighbors=1, assignment_max_regret=.1)


def test_native_regret_generation_keeps_decode_positions():
    from tests.unit.test_projector_comparison_evaluation import tiny_state
    from mllm_comp.analysis.common.compression_training_check import AnchorAdapterSpec, create_anchor_adapter
    from mllm_comp.compressors.image.context_visual_topk import ContextVisualTopK, ContextVisualTopKSpec
    from mllm_comp.compressors.comparison.importance_anchor_adapter import QwenImportanceAnchorAdapter
    bundle, state, _, _ = tiny_state()
    session = QwenImportanceAnchorAdapter(bundle['model'],
        {'retention_ratio': .5, 'importance_gamma': 1., 'assignment_neighbors': 1, 'assignment_max_regret': .1},
        adapter_module=create_anchor_adapter(AnchorAdapterSpec(24, 4)),
        selector_module=ContextVisualTopK(ContextVisualTopKSpec(24, width=4, context_mode='image'))).install()
    try:
        with torch.no_grad():
            bundle['model'].generate(**state['prompt_inputs'], max_new_tokens=3, min_new_tokens=3,
                                    do_sample=False, pad_token_id=0)
        trace = session.runtime_metadata()['selection']
        assert trace['result_count'] == 3 and trace['decode_position_verified']
        assert trace['assignment_max_regret'] == .1
    finally:
        session.uninstall()
