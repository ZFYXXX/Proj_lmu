import copy

import pytest
import torch

from mllm_comp.compressors.image.input_adaptive_topk import (
    InputAdaptiveTopK, InputAdaptiveTopKSpec, save_selector, load_selector,
)
from mllm_comp.compressors.image.word_visual_topk import WordVisualTopK, WordVisualTopKSpec


def model():
    return InputAdaptiveTopK(InputAdaptiveTopKSpec(12, 8, 3, match_width=4, gate_width=5))


def test_initial_gate_matches_fixed_word_head_exactly():
    adaptive = model()
    fixed = WordVisualTopK(WordVisualTopKSpec(12, 8, 3, match_width=4), visual=copy.deepcopy(adaptive.visual))
    x, q = torch.randn(11, 12), torch.randn(5, 12)
    a, b = adaptive(x, question=q, soft=True), fixed(x, question=q, soft=True)
    torch.testing.assert_close(a['scores'], b['scores'], rtol=0, atol=0)
    torch.testing.assert_close(a['soft_weights'], b['soft_weights'], rtol=0, atol=0)
    assert a['word_diagnostics']['input_gate'] == 1
    assert torch.equal(a['indices'], b['indices'])


def test_gate_depends_on_input_after_learning_and_preserves_budget():
    m = model()
    with torch.no_grad():
        m.input_gate[-1].weight.fill_(1.)
    x, q = torch.randn(11, 12), torch.randn(5, 12)
    a, b = m(x, question=q), m(x, question=-q)
    assert a['word_diagnostics']['input_gate'] != b['word_diagnostics']['input_gate']
    for out in (a, b):
        assert 0 < out['word_diagnostics']['input_gate'] < 2
        assert len(out['indices']) == len(out['indices'].unique()) == 3
        assert torch.equal(out['tokens'], x[out['indices']])
        assert torch.equal(out['indices'], out['indices'].sort().values)


def test_soft_gradients_and_gate_initialization_are_well_defined():
    m = model().train()
    x, q = torch.randn(11, 12), torch.randn(5, 12, requires_grad=True)
    opt = torch.optim.AdamW([p for p in m.parameters() if p.requires_grad], lr=.01)
    for step in range(2):
        opt.zero_grad()
        out = m(x, question=q, soft=True)
        torch.testing.assert_close(out['soft_weights'].sum(), torch.tensor(3.), atol=1e-5, rtol=0)
        (out['tokens'] * torch.arange(11)[:, None]).sum().backward()
        assert q.grad is None
        assert all(p.grad is None for p in m.visual.parameters())
        assert m.input_gate[-1].weight.grad.isfinite().all()
        assert m.input_gate[-1].weight.grad.norm() > 0
        if step:
            assert m.input_gate[0].weight.grad.norm() > 0
        opt.step()


def test_repeat_invariance_identity_and_visual_only():
    m = model()
    x, q = torch.randn(11, 12), torch.randn(5, 12)
    torch.testing.assert_close(m(x, question=q)['scores'], m(x, question=q.repeat(2, 1))['scores'])
    torch.testing.assert_close(m(x, question=q, visual_only=True)['scores'], m.visual(x)['scores'])
    assert torch.equal(m(x[:2], question=q)['tokens'], x[:2])
    with pytest.raises(ValueError, match='Nonfinite'):
        m(x * float('nan'), question=q)
    with pytest.raises(ValueError, match='nonempty'):
        m(x, question=q[:0])


def test_checkpoint_integrity_and_independent_format(tmp_path):
    m = model()
    folder = tmp_path / 'checkpoint'
    save_selector(m, folder, base_identity='base', provenance={'unit': True})
    actual = load_selector(folder, base_identity='base', device='cpu')
    assert all(torch.equal(v, actual.state_dict()[k]) for k, v in m.state_dict().items())
    with pytest.raises(ValueError, match='base'):
        load_selector(folder, base_identity='wrong', device='cpu')
    with (folder / 'selector.safetensors').open('ab') as stream:
        stream.write(b'bad')
    with pytest.raises(ValueError, match='integrity'):
        load_selector(folder, base_identity='base', device='cpu')


@pytest.mark.parametrize('soft', [False, True])
def test_native_binding_identity_budget_decode_and_training(soft):
    from tests.unit.test_projector_comparison_evaluation import tiny_state
    from mllm_comp.compressors.comparison.input_adaptive_topk import QwenInputAdaptiveTopKComparison
    bundle, state, _, _ = tiny_state()
    base = bundle['model'].requires_grad_(False)
    prompt = state['prompt_inputs']
    with torch.no_grad():
        expected = base(**prompt, use_cache=False).logits
    identity = QwenInputAdaptiveTopKComparison(base, {'target_count': 128}).install()
    try:
        with torch.no_grad():
            torch.testing.assert_close(base(**prompt, use_cache=False).logits, expected, rtol=1e-5, atol=1e-6)
    finally:
        identity.uninstall()
    session = QwenInputAdaptiveTopKComparison(base,
        {'target_count': 3, 'training_mode': 'soft' if soft else 'hard'},
        phase='forward' if soft else 'generate').install()
    try:
        if soft:
            labels = torch.full_like(prompt['input_ids'], -100)
            labels[:, -1] = 3
            base(**prompt, labels=labels, use_cache=False).loss.backward()
            assert session.selector.input_gate[-1].weight.grad.norm() > 0
            assert all(p.grad is None for p in base.parameters())
        else:
            with torch.no_grad():
                base.generate(**prompt, max_new_tokens=3, min_new_tokens=3, do_sample=False, pad_token_id=0)
            trace = session.runtime_metadata()['selection']
            assert trace['result_count'] == 3 and trace['decode_position_verified']
            assert trace['input_adaptive_gate'] and trace['question_tokens']
            assert trace['observed_visual_tokens_per_decoder_layer'] == [3] * 4
    finally:
        session.uninstall()


def test_exact_training_restore(tmp_path, monkeypatch):
    from mllm_comp.analysis.common import learned_topk_experiment as exp
    monkeypatch.setattr(exp, 'OWNER', tmp_path)
    def create():
        m = model()
        o = torch.optim.AdamW([p for p in m.parameters() if p.requires_grad], lr=.01)
        return m, o, torch.optim.lr_scheduler.LambdaLR(o, exp.lr_multiplier)
    def steps(m, o, s, count):
        for _ in range(count):
            out = m(torch.randn(11, 12), question=torch.randn(5, 12), soft=True)
            (out['tokens'] * torch.randn_like(out['tokens'])).sum().backward()
            o.step(); s.step(); o.zero_grad()
    torch.manual_seed(42)
    a, oa, sa = create(); steps(a, oa, sa, 4)
    torch.manual_seed(42)
    b, ob, sb = create(); steps(b, ob, sb, 2)
    folder = tmp_path / 'saved'
    contract = {'base_identity_sha256': 'base'}
    exp.save_training_state(folder, selector=b, optimizer=ob, scheduler=sb, step=2,
        contract=contract, random_states=[exp.rng_state()], save_fn=save_selector)
    c, oc, sc = create()
    assert exp.restore_training_state(folder, selector=c, optimizer=oc, scheduler=sc,
        contract=contract, load_fn=load_selector) == 2
    steps(c, oc, sc, 2)
    assert all(torch.equal(v, c.state_dict()[k]) for k, v in a.state_dict().items())
    assert sa.state_dict() == sc.state_dict()


def test_mmstar_bootstrap_preserves_native_category_macro():
    from scripts.train.qwen25_vl.run_input_adaptive_topk import paired_mmstar
    def row(category, score):
        return {'average': {'l2_category': category, 'score': score}}
    a = [row('A', 1), row('A', 1), row('B', 0)]
    b = [row('A', 0), row('A', 0), row('B', 0)]
    actual = paired_mmstar(a, b, ['one', 'one', 'two'], resamples=100)
    assert actual['delta'] == .5  # not the question micro difference, 2/3
    assert actual['ci95'] == [.5, .5]
    assert actual['independent_images'] == 2
