import copy

import pytest
import torch

from mllm_comp.analysis.common.learned_topk_experiment import training_loss
from mllm_comp.analysis.common.topk_training_modes import (
    parameter_digest, selection_diagnostics, training_selection,
)
from mllm_comp.compressors.image.learned_visual_topk import LearnedVisualTopK, LearnedVisualTopKSpec
from scripts.train.qwen25_vl.run_topk_mask_ablation import matched_settings, paired_scores


def scorer(k=3):
    return LearnedVisualTopK(LearnedVisualTopKSpec(8, width=4, target_count=k))


def test_soft_path_exactly_preserves_original_and_initialization():
    torch.manual_seed(7)
    features = torch.randn(7, 8)
    a, b = scorer(), scorer()
    assert parameter_digest(a) == parameter_digest(b)
    original, soft = a(features, soft=True), training_selection(b, features, 'soft')
    assert torch.equal(original['tokens'], soft['tokens'])
    assert torch.equal(original['soft_weights'], soft['soft_weights'])
    original['tokens'].square().mean().backward()
    soft['tokens'].square().mean().backward()
    assert all(torch.equal(p.grad, q.grad) for p, q in zip(a.parameters(), b.parameters()))


def test_ste_forward_exact_hard_mask_and_same_surrogate_jacobian():
    torch.manual_seed(3)
    features = torch.randn(7, 8)
    a, b = scorer(), scorer()
    soft = training_selection(a, features, 'soft')
    ste = training_selection(b, features, 'ste')
    assert torch.equal(ste['applied_weights'], ste['hard_mask'])
    assert torch.equal(ste['tokens'], features * ste['hard_mask'][:, None])
    assert ste['tokens'].shape == features.shape
    assert len(ste['indices'].unique()) == 3
    upstream = torch.randn_like(features)
    (soft['tokens'] * upstream).sum().backward()
    (ste['tokens'] * upstream).sum().backward()
    for p, q in zip(a.parameters(), b.parameters()):
        torch.testing.assert_close(p.grad, q.grad, rtol=0, atol=0)
    assert sum(float(p.grad.abs().sum()) for p in b.parameters()) > 0
    metrics = selection_diagnostics(ste)
    assert metrics['applied_nonzero_tokens'] == metrics['target_visual_tokens'] == 3
    assert metrics['physical_visual_tokens'] == metrics['source_visual_tokens'] == 7


@pytest.mark.parametrize('mode', ['soft', 'ste'])
@pytest.mark.parametrize('n', [2, 3, 7])
def test_budget_boundaries_finite_gradients_and_identical_deployment(mode, n):
    module = scorer()
    features = torch.randn(n, 8)
    selection = training_selection(module, features, mode)
    assert float(selection['applied_weights'].detach().sum()) == pytest.approx(min(n, 3), abs=1e-5)
    if n <= 3:
        assert torch.equal(selection['tokens'], features)
    selection['tokens'].sum().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in module.parameters())
    hard = module(features)
    assert torch.equal(hard['tokens'], features[hard['indices']])
    assert len(hard['tokens']) == min(n, 3)


def test_common_bce_uses_soft_probabilities_not_ste_mask():
    features = torch.randn(7, 8)
    soft = training_selection(scorer(), features, 'soft')
    ste = training_selection(scorer(), features, 'ste')
    logits, teacher, labels = torch.randn(2, 5), torch.randn(2, 5), torch.tensor([1, 2])
    a, ma = training_loss(logits, teacher, labels, soft, 100)
    b, mb = training_loss(logits, teacher, labels, ste, 100)
    assert torch.equal(a, b) and ma == mb
    assert ma['mask_bce'] > 0


def test_arm_contract_rejects_nonmask_confounds():
    base = {k: 'same' for k in ('source', 'dependencies', 'identity_path', 'training_manifest',
        'initial_parameter_sha256', 'eval', 'resource_policy')}
    arms = {m: {**base, 'training': {'training_mask_mode': m, 'seed': 42, 'steps': 320}} for m in ('soft', 'ste')}
    assert matched_settings(arms)
    changed = copy.deepcopy(arms)
    changed['ste']['training']['seed'] = 43
    with pytest.raises(ValueError, match='Non-mask'):
        matched_settings(changed)
    changed = copy.deepcopy(arms)
    changed['ste']['initial_parameter_sha256'] = 'other'
    with pytest.raises(ValueError, match='initial_parameter'):
        matched_settings(changed)


def test_paired_development_maps_back_to_independent_images():
    a = {0: {'input': 'q', 'target': 'a', 'relaxed_overall': 1.},
         1: {'input': 'q2', 'target': 'a2', 'relaxed_overall': 0.}}
    b = {i: {**v, 'relaxed_overall': 0.} for i, v in a.items()}
    cohort = {'development_indices': [2, 0], 'rows': [
        {'image_fingerprint': 'A'}, {'image_fingerprint': 'unused'}, {'image_fingerprint': 'B'}]}
    result = paired_scores(a, b, cohort, 'dev', 'chartqa')
    assert result['independent_images'] == 2
    assert result['question_weighted_delta'] == .5
    assert result['question_wins'] == result['question_ties'] == 1
    with pytest.raises(ValueError, match='prompts/targets'):
        paired_scores(a, {**b, 0: {**b[0], 'target': 'changed'}}, cohort, 'dev', 'chartqa')


def test_unknown_mask_mode_fails_explicitly():
    with pytest.raises(ValueError, match='Unknown'):
        training_selection(scorer(), torch.randn(7, 8), 'hard_without_gradient')


@pytest.mark.parametrize('mode', ['soft', 'ste'])
def test_native_qwen_student_backprop_preserves_positions_labels_and_frozen_base(mode):
    from tests.unit.test_projector_comparison_evaluation import tiny_state
    from mllm_comp.analysis.common.learned_hybrid_runtime import answer_logits
    bundle, state, _, _ = tiny_state()
    bundle['model'].requires_grad_(False)
    module = LearnedVisualTopK(LearnedVisualTopKSpec(state['features'].shape[1], width=4, target_count=3))
    labels = torch.full_like(state['prompt_inputs']['input_ids'], -100)
    labels[0, -1] = 3
    before = state['positions'].clone()
    with torch.no_grad():
        teacher, targets = answer_logits(bundle, state['hidden'], state['positions'], labels)
    selection = training_selection(module, state['features'], mode)
    hidden = state['hidden'].index_copy(1, torch.tensor(state['visual']), selection['tokens'][None])
    student, actual_targets = answer_logits(bundle, hidden, state['positions'], labels)
    loss, _ = training_loss(student, teacher, targets, selection, 0)
    loss.backward()
    assert torch.equal(targets, actual_targets)
    assert torch.equal(state['positions'], before)
    assert hidden.shape == state['hidden'].shape
    assert all(p.grad is None for p in bundle['model'].parameters())
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in module.parameters())
    assert sum(float(p.grad.abs().sum()) for p in module.parameters()) > 0


@pytest.mark.parametrize('mode', ['soft', 'ste'])
def test_exact_optimizer_restore_for_each_mask_mode(tmp_path, monkeypatch, mode):
    from mllm_comp.analysis.common import learned_topk_experiment as experiment
    monkeypatch.setattr(experiment, 'OWNER', tmp_path)
    contract = {'base_identity_sha256': 'test', 'training_mask_mode': mode}
    def create():
        module = scorer()
        opt = torch.optim.AdamW(module.parameters(), lr=2e-4)
        sched = torch.optim.lr_scheduler.LambdaLR(opt, experiment.lr_multiplier)
        return module, opt, sched
    def advance(module, opt, sched, start, end):
        for step in range(start, end):
            selection = training_selection(module, torch.randn(7, 8), mode)
            loss, _ = training_loss(selection['tokens'][:2, :5], torch.randn(2, 5),
                torch.tensor([1, 2]), selection, step)
            loss.backward(); opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
    torch.manual_seed(9)
    a, oa, sa = create(); advance(a, oa, sa, 0, 4)
    torch.manual_seed(9)
    b, ob, sb = create(); advance(b, ob, sb, 0, 2)
    path = tmp_path / 'checkpoint'
    experiment.save_training_state(path, selector=b, optimizer=ob, scheduler=sb,
        step=2, contract=contract, random_states=[experiment.rng_state()])
    c, oc, sc = create()
    step = experiment.restore_training_state(path, selector=c, optimizer=oc, scheduler=sc, contract=contract)
    advance(c, oc, sc, step, 4)
    assert parameter_digest(a) == parameter_digest(c)
    assert sa.state_dict() == sc.state_dict()
    for key, values in oa.state_dict()['state'].items():
        for field, value in values.items():
            assert torch.equal(value, oc.state_dict()['state'][key][field])
    with pytest.raises(ValueError, match='contract changed'):
        experiment.restore_training_state(path, selector=c, optimizer=oc, scheduler=sc,
            contract={**contract, 'training_mask_mode': 'ste' if mode == 'soft' else 'soft'})
