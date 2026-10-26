import pytest
import torch

from mllm_comp.analysis.common.visual_noise_gate import noise_gate
from mllm_comp.analysis.common.topk_training_modes import training_selection
from mllm_comp.compressors.image.learned_visual_topk import LearnedVisualTopK, LearnedVisualTopKSpec
from mllm_comp.compressors.image.context_visual_topk import ContextVisualTopK, ContextVisualTopKSpec
from mllm_comp.compressors.image.query_transport import (
    QueryTransportSpec, semi_unbalanced_transport, transport_topk,
)
from mllm_comp.compressors.comparison.context_topk_ot import QwenContextTopKOTComparison


def test_noise_formula_endpoints_and_finite_endpoint_gradients():
    x, epsilon = torch.randn(4, 8), torch.randn(4, 8)
    alpha = torch.tensor([0., .25, .75, 1.], requires_grad=True)
    out = noise_gate(x, alpha, noise=epsilon)
    torch.testing.assert_close(out, alpha[:, None].sqrt() * x + (1-alpha[:, None]).sqrt() * epsilon)
    assert torch.equal(out[0], epsilon[0]) and torch.equal(out[-1], x[-1])
    out.sum().backward()
    assert torch.isfinite(alpha.grad).all()


@pytest.mark.parametrize('contextual', [False, True])
def test_noise_full_length_cardinality_gradient_and_unchanged_deployment(contextual):
    spec = dict(hidden_size=8, width=4, target_count=3)
    selector = ContextVisualTopK(ContextVisualTopKSpec(**spec)) if contextual else LearnedVisualTopK(LearnedVisualTopKSpec(**spec))
    x, q, epsilon = torch.randn(7, 8), torch.randn(3, 8), torch.randn(7, 8)
    kwargs = {'question': q} if contextual else {}
    selected_before = selector(x, **kwargs)['indices'].clone()
    result = training_selection(selector, x, 'vp_noise', noise=epsilon, **kwargs)
    assert result['tokens'].shape == x.shape
    assert float(result['soft_weights'].sum()) == pytest.approx(3, abs=1e-5)
    torch.testing.assert_close(result['tokens'], noise_gate(x, result['soft_weights'], noise=epsilon))
    (result['tokens'] * torch.randn_like(x)).sum().backward()
    assert all(p.grad is not None and p.grad.isfinite().all() for p in selector.parameters())
    assert sum(float(p.grad.abs().sum()) for p in selector.parameters()) > 0
    assert torch.equal(selector(x, **kwargs)['indices'], selected_before)


def test_noise_rng_replay_and_all_kept_identity():
    x = torch.randn(4, 8)
    state = torch.get_rng_state()
    a = noise_gate(x, torch.full((4,), .5))
    torch.set_rng_state(state)
    assert torch.equal(a, noise_gate(x, torch.full((4,), .5)))
    assert torch.equal(x, noise_gate(x, torch.ones(4)))
    with pytest.raises(ValueError, match='weights'):
        noise_gate(x, torch.full((4,), 1.01))
    with pytest.raises(ValueError, match='noise'):
        noise_gate(x, torch.ones(4), noise=torch.randn(3, 8))


@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
def test_ot_fixed_rows_and_relaxed_columns_against_primal_optimum(dtype):
    # Independently optimize row-stochastic primal via scipy; do not compare
    # Sinkhorn against an identical copy of its own update implementation.
    from scipy.optimize import minimize
    import numpy as np
    cost = torch.tensor([[0., .3, .8], [.7, .2, 0.]], dtype=dtype)
    spec = QueryTransportSpec(tolerance=1e-6 if dtype == torch.float32 else 1e-10)
    plan, stats = semi_unbalanced_transport(cost, spec)
    torch.testing.assert_close(plan.sum(1), torch.full((2,), .5, dtype=dtype))
    def objective(z):
        raw = z.reshape(2, 3)
        probabilities = np.exp(raw - raw.max(1, keepdims=True))
        p = .5 * probabilities / probabilities.sum(1, keepdims=True)
        c = p.sum(0)
        return (p * cost.double().numpy()).sum() + spec.epsilon * (p * (np.log(p)-1)).sum() + spec.marginal_penalty * (c * np.log(c / (1/3)) - c + 1/3).sum()
    reference = minimize(objective, np.zeros(6), method='BFGS', options={'gtol': 1e-9})
    assert objective(plan.double().log().numpy().reshape(-1)) == pytest.approx(reference.fun, abs=1e-7)
    assert stats['ot_total_mass'] == pytest.approx(1, abs=1e-6)
    assert not torch.allclose(plan.sum(0), plan.new_full((3,), 1/3), atol=1e-3)


def test_ot_uniform_cost_permutation_equivariance_and_failure():
    spec = QueryTransportSpec(tolerance=1e-7)
    plan, _ = semi_unbalanced_transport(torch.zeros(3, 7, dtype=torch.float64), spec)
    torch.testing.assert_close(plan, torch.full_like(plan, 1/21))
    c = torch.rand(3, 7, dtype=torch.float64)
    a, _ = semi_unbalanced_transport(c, spec)
    perm = torch.tensor([3, 2, 0, 1, 5, 6, 4])
    b, _ = semi_unbalanced_transport(c[:, perm], spec)
    torch.testing.assert_close(b, a[:, perm])
    with pytest.raises(ValueError, match='converge'):
        semi_unbalanced_transport(c, QueryTransportSpec(max_iterations=1, tolerance=1e-12))
    with pytest.raises(ValueError, match='finite'):
        semi_unbalanced_transport(torch.tensor([[float('nan')]]))


def test_ot_exact_budget_determinism_and_weight_zero_original_ranking():
    from mllm_comp.compressors.image.learned_visual_topk import hard_topk_indices
    x, q, scores = torch.randn(19, 8), torch.randn(5, 8), torch.randn(19)
    a, trace = transport_topk(x, q, scores, 7)
    b, _ = transport_topk(x, q, scores, 7)
    assert a.tolist() == sorted(set(a.tolist())) and len(a) == 7 and torch.equal(a, b)
    assert trace['ot_applied'] and not trace['ot_discrete_coverage_guarantee']
    zero, _ = transport_topk(x, q, scores, 7, QueryTransportSpec(weight=0))
    assert torch.equal(zero, hard_topk_indices(scores, 7))
    for kwargs in ({'epsilon': 0}, {'weight': -1}, {'max_iterations': 0}):
        with pytest.raises(ValueError):
            QueryTransportSpec(**kwargs)


@pytest.mark.parametrize('weight', [0., 1.])
def test_native_ot_identity_decode_position_labels_and_answer_exclusion(weight):
    from tests.unit.test_projector_comparison_evaluation import tiny_state
    bundle, state, _, _ = tiny_state()
    model, prompt = bundle['model'], state['prompt_inputs']
    with torch.no_grad():
        dense = model(**prompt, use_cache=False).logits
    session = QwenContextTopKOTComparison(model, {'target_count': 128, 'ot_weight': weight}).install()
    try:
        with torch.no_grad():
            torch.testing.assert_close(model(**prompt, use_cache=False).logits, dense, rtol=1e-5, atol=1e-6)
    finally:
        session.uninstall()
    session = QwenContextTopKOTComparison(model, {'target_count': 3, 'ot_weight': weight}).install()
    captured = []
    compact = session.compact_selection
    def capture(call, hidden, local, stats):
        result = compact(call, hidden, local, stats)
        captured.append(call['position_ids'].detach().clone())
        return result
    session.compact_selection = capture
    try:
        labels = torch.full_like(prompt['input_ids'], -100)
        labels[:, -1] = 3
        with torch.no_grad():
            result = model(**prompt, labels=labels, use_cache=False)
            first = dict(session.runtime_metadata()['selection'])
            assert torch.isfinite(result.loss)
            model.generate(**prompt, max_new_tokens=3, min_new_tokens=3, do_sample=False, pad_token_id=0)
        trace = session.runtime_metadata()['selection']
        assert first['selected_indices'] == trace['selected_indices']
        assert first['question_ids_sha256'] == trace['question_ids_sha256']
        assert trace['result_count'] == 3 and trace['decode_position_verified']
        assert trace['observed_visual_tokens_per_decoder_layer'] == [3] * 4
        # Text rotary coordinates survive prefill compaction unchanged.
        before = state['positions'][:, :, [i for i in range(state['prefix']) if i not in state['visual']]]
        after = captured[0][-3:, :, [0, 1, 2, 3, 4, 8, 9, 10, 11]]
        torch.testing.assert_close(before, after)
    finally:
        session.uninstall()


def test_noise_native_backward_freezes_base():
    from tests.unit.test_projector_comparison_evaluation import tiny_state
    from mllm_comp.analysis.common.learned_hybrid_runtime import answer_logits
    from mllm_comp.analysis.common.learned_topk_experiment import training_loss
    bundle, state, _, _ = tiny_state()
    bundle['model'].requires_grad_(False)
    selector = ContextVisualTopK(ContextVisualTopKSpec(state['features'].shape[1], width=4, target_count=3))
    labels = torch.full_like(state['prompt_inputs']['input_ids'], -100)
    labels[:, -1] = 3
    with torch.no_grad():
        teacher, targets = answer_logits(bundle, state['hidden'], state['positions'], labels)
    selection = training_selection(selector, state['features'], 'vp_noise', question=torch.randn(3, state['features'].shape[1]))
    hidden = state['hidden'].index_copy(1, torch.tensor(state['visual']), selection['tokens'][None])
    student, targets = answer_logits(bundle, hidden, state['positions'], labels)
    loss, _ = training_loss(student, teacher, targets, selection, 0)
    loss.backward()
    assert all(p.grad is None for p in bundle['model'].parameters())
    assert all(p.grad is not None and p.grad.isfinite().all() for p in selector.parameters())


def test_noise_context_optimizer_and_rng_resume_exact(tmp_path, monkeypatch):
    from mllm_comp.analysis.common import learned_topk_experiment as experiment
    from mllm_comp.compressors.image.context_visual_topk import save_selector, load_selector
    monkeypatch.setattr(experiment, 'OWNER', tmp_path)
    contract = {'base_identity_sha256': 'test', 'training_mask_mode': 'vp_noise'}
    def create():
        module = ContextVisualTopK(ContextVisualTopKSpec(8, 4, 3))
        optimizer = torch.optim.AdamW(module.parameters(), lr=2e-4)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, experiment.lr_multiplier)
        return module, optimizer, scheduler
    def advance(module, optimizer, scheduler, start, end):
        for _ in range(start, end):
            selection = training_selection(module, torch.randn(7, 8), 'vp_noise', question=torch.randn(3, 8))
            (selection['tokens'] * torch.randn(7, 8)).sum().backward()
            optimizer.step(); scheduler.step(); optimizer.zero_grad(set_to_none=True)
    torch.manual_seed(9)
    a, oa, sa = create(); advance(a, oa, sa, 0, 4)
    torch.manual_seed(9)
    b, ob, sb = create(); advance(b, ob, sb, 0, 2)
    path = tmp_path / 'checkpoint'
    experiment.save_training_state(path, selector=b, optimizer=ob, scheduler=sb, step=2,
        contract=contract, random_states=[experiment.rng_state()], save_fn=save_selector)
    c, oc, sc = create()
    step = experiment.restore_training_state(path, selector=c, optimizer=oc, scheduler=sc,
        contract=contract, load_fn=load_selector)
    advance(c, oc, sc, step, 4)
    assert all(torch.equal(value, c.state_dict()[key]) for key, value in a.state_dict().items())
    assert sa.state_dict() == sc.state_dict()
    for key, values in oa.state_dict()['state'].items():
        for field, value in values.items():
            assert torch.equal(value, oc.state_dict()['state'][key][field])


def test_native_ot_stage_contract():
    from mllm_comp.pipeline.stage import CompressionStageConfig
    from mllm_comp.pipeline.v2_validation import validate_v2_stage_contract
    from mllm_comp.compressors.multimodal_registry import builtin_multimodal_compression_contracts
    algorithm = 'context_visual_topk_ot'
    stage = CompressionStageConfig.from_dict({'name': 'ot', 'plugin': algorithm, 'algorithm': algorithm,
        'variant': f'{algorithm}.image', 'binding': f'{algorithm}.image.qwen25_vl',
        'placement': 'projector_level', 'method': 'token_compression', 'object_family': 'tensor_slot',
        'execution_phase': 'generate', 'modalities': ['image'], 'target_modalities': ['image'],
        'conditioning_modalities': ['text'], 'config': {}})
    validate_v2_stage_contract(stage, registry=builtin_multimodal_compression_contracts(), backbone_name='qwen25_vl')
