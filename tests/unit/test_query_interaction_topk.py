import copy
import json

import pytest
import torch

from mllm_comp.compressors.image.query_interaction_topk import (
    QueryInteractionTopK, QueryInteractionSpec, save_selector, load_selector,
)
from mllm_comp.compressors.image.word_visual_topk import WordVisualTopK, WordVisualTopKSpec
from mllm_comp.analysis.common.task_loss_supervision import (
    candidate_sets, retention_target, teacher_ranking_loss, load_target,
)


def model(mode='contextual'):
    return QueryInteractionTopK(QueryInteractionSpec(12, 8, 3, match_width=4, interaction=mode, heads=2))


def test_word_arm_is_exact_old_method_and_same_trainable_initialization():
    m = model('word')
    old = WordVisualTopK(WordVisualTopKSpec(12, 8, 3, match_width=4), visual=copy.deepcopy(m.visual))
    assert all(torch.equal(t, old.state_dict()[k]) for k, t in m.state_dict().items())
    x, q = torch.randn(9, 12), torch.randn(4, 12)
    for soft in (False, True):
        a, b = m(x, question=q, soft=soft), old(x, question=q, soft=soft)
        torch.testing.assert_close(a['scores'], b['scores'], rtol=0, atol=0)
        assert torch.equal(a['indices'], b['indices'])


def test_contextual_scores_use_question_order_and_token_specific_changes():
    m = model()
    generator = torch.Generator().manual_seed(918)
    x, q = torch.randn(9, 12, generator=generator), torch.randn(5, 12, generator=generator)
    a = m(x, question=q)['scores']
    b = m(x, question=q.flip(0))['scores']
    scale = m.visual(x)['scores'].std().clamp_min(1e-4)
    assert (a-b).abs().max() > scale*1e-4
    assert (a-b).std() > scale*1e-4
    hard = m(x, question=q)
    assert len(hard['indices'].unique()) == 3
    assert torch.equal(hard['tokens'], x[hard['indices']])
    assert torch.equal(m(x[:2], question=q)['tokens'], x[:2])


@pytest.mark.parametrize('mode', ['word', 'contextual'])
def test_training_gradients_teacher_targets_and_roundtrip(mode, tmp_path):
    m = model(mode)
    x, q = torch.randn(9, 12), torch.randn(5, 12, requires_grad=True)
    out = m(x, question=q, soft=True)
    torch.testing.assert_close(out['soft_weights'].sum(), torch.tensor(3.), atol=1e-5, rtol=0)
    target = torch.tensor([1., 1., 1., 0, 0, 0, 0, 0, 0])
    loss = teacher_ranking_loss(out['scores'], target) + out['tokens'].square().mean()
    loss.backward()
    assert m.question_projection.weight.grad.norm() > 0
    assert q.grad is None
    assert all(p.grad is None for p in m.visual.parameters())
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters() if p.requires_grad)
    if mode == 'contextual':
        for prefix in ('text_qkv', 'cross_q', 'cross_k', 'cross_v', 'token_head'):
            assert sum(float(p.grad.norm()) for k, p in m.named_parameters() if k.startswith(prefix)) > 0
    folder = tmp_path / 'checkpoint'
    save_selector(m, folder, base_identity='base', provenance={})
    actual = load_selector(folder, base_identity='base', device='cpu')
    assert all(torch.equal(t, actual.state_dict()[k]) for k, t in m.state_dict().items())
    with pytest.raises(ValueError, match='integrity'):
        load_selector(folder, base_identity='other', device='cpu')


def test_rectangular_candidate_sets_are_unique_fixed_budget_deterministic():
    scores = torch.linspace(-1, 1, 15*23)
    a = candidate_sets(scores, 15, 23)
    assert a == candidate_sets(scores, 15, 23)
    assert len({tuple(s) for s in a}) == 8
    assert all(len(s) == 128 and s == sorted(set(s)) and 0 <= s[0] <= s[-1] < len(scores) for s in a)
    assert candidate_sets(scores[:128], 8, 16) == [list(range(128))]


def test_teacher_mass_ties_and_bad_labels():
    candidates = [[0, 1], [2, 3]]
    tied = retention_target(candidates, [1., 1.], 4)
    torch.testing.assert_close(tied, torch.full((4,), .5))
    preferred = retention_target(candidates, [.1, 3.], 4)
    assert preferred[0] > preferred[3]
    assert preferred.sum() == 2
    for invalid in ([[0, 0], [2, 3]], [[0], [1, 2]], [[-1, 0], [1, 2]]):
        with pytest.raises(ValueError, match='budget/member'):
            retention_target(invalid, [1., 2.], 4)
    with pytest.raises(ValueError, match='Nonfinite'):
        retention_target(candidates, [float('nan'), 1.], 4)


def test_teacher_checks_question_and_image_identity(tmp_path):
    from mllm_comp.analysis.common.anchor7b_runtime import file_sha
    p = tmp_path/'label.json'
    row = {'sample_id': 'x', 'image_fingerprint': 'img', 'question_ids_sha256': 'q',
           'source_count': 4, 'height': 2, 'width': 2, 'candidates': [[0,1],[2,3]], 'losses': [1.,2.]}
    p.write_text(json.dumps(row))
    index = {'rows': {'x': {'path': str(p), 'sha256': file_sha(p)}}, 'temperature': .25}
    example = {'sample_id': 'x', 'image_fingerprint': 'img'}
    state = {'visual': [1,2,3,4], 'height': 2, 'width': 2}
    torch.testing.assert_close(load_target(index, example, state, {'question_ids_sha256': 'q'}, device='cpu').sum(), torch.tensor(2.))
    with pytest.raises(ValueError, match='alignment'):
        load_target(index, example, state, {'question_ids_sha256': 'different'}, device='cpu')


@pytest.mark.parametrize('soft', [False, True])
def test_qwen_binding_dense_identity_hard_cache_soft_gradients(soft):
    from tests.unit.test_projector_comparison_evaluation import tiny_state
    from mllm_comp.compressors.comparison.query_interaction_topk import QwenQueryInteractionTopK
    bundle, state, _, _ = tiny_state()
    base = bundle['model'].requires_grad_(False)
    prompt = state['prompt_inputs']
    with torch.no_grad():
        dense = base(**prompt, use_cache=False).logits
    identity = QwenQueryInteractionTopK(base, {'target_count': 128}).install()
    try:
        with torch.no_grad():
            torch.testing.assert_close(base(**prompt, use_cache=False).logits, dense, rtol=1e-5, atol=1e-6)
    finally:
        identity.uninstall()
    session = QwenQueryInteractionTopK(base, {'target_count': 3, 'training_mode': 'soft' if soft else 'hard'},
                                     phase='forward' if soft else 'generate').install()
    try:
        if soft:
            labels = torch.full_like(prompt['input_ids'], -100)
            labels[:, -1] = 3
            base(**prompt, labels=labels, use_cache=False).loss.backward()
            assert session.selector.cross_v.weight.grad.norm() > 0
            assert all(p.grad is None for p in base.parameters())
        else:
            with torch.no_grad():
                base.generate(**prompt, max_new_tokens=3, min_new_tokens=3, do_sample=False, pad_token_id=0)
            trace = session.runtime_metadata()['selection']
            assert trace['result_count'] == 3 and trace['decode_position_verified']
            assert trace['per_token_cross_attention'] and trace['question_tokens']
            assert trace['observed_visual_tokens_per_decoder_layer'] == [3]*4
    finally:
        session.uninstall()


def test_teacher_hard_compaction_preserves_labels_and_all_text_positions():
    from mllm_comp.analysis.common.learned_hybrid_runtime import compact_state
    n, length, width = 144, 156, 12
    ids = torch.ones(1, length, dtype=torch.long)
    ids[0, 3:3+n] = 10
    labels = torch.full_like(ids, -100)
    labels[0, -2:] = torch.tensor([7, 8])
    hidden = torch.randn(1, length, width)
    pos = torch.arange(length).reshape(1, 1, -1).repeat(3, 1, 1)
    state = {'features': hidden[0, 3:3+n], 'hidden': hidden, 'positions': pos, 'visual': list(range(3, 3+n)),
             'prefix': length-2, 'teacher_inputs': {'input_ids': ids, 'labels': labels, 'attention_mask': torch.ones_like(ids)}}
    members = candidate_sets(torch.randn(n), 12, 12)[2]
    compact = compact_state(state, {'indices': members, 'tokens': state['features'][members]})
    assert compact['hidden'].shape[1] == length - 16
    assert compact['labels'][compact['labels'] != -100].tolist() == [7, 8]
    keep = [i for i in range(length) if i not in set(state['visual']) or i-3 in members]
    assert torch.equal(compact['positions'], pos[:, :, keep])


def test_contextual_optimizer_and_rng_resume_exactly(tmp_path, monkeypatch):
    from mllm_comp.analysis.common import learned_topk_experiment as exp
    monkeypatch.setattr(exp, 'OWNER', tmp_path)
    def create():
        m = model()
        opt = torch.optim.AdamW([p for p in m.parameters() if p.requires_grad], lr=.01)
        return m, opt, torch.optim.lr_scheduler.LambdaLR(opt, exp.lr_multiplier)
    def steps(m, opt, scheduler, count):
        for _ in range(count):
            out = m(torch.randn(9, 12), question=torch.randn(5, 12), soft=True)
            (out['tokens']*torch.randn_like(out['tokens'])).sum().backward()
            opt.step(); scheduler.step(); opt.zero_grad()
    torch.manual_seed(42)
    a, oa, sa = create(); steps(a, oa, sa, 4)
    torch.manual_seed(42)
    b, ob, sb = create(); steps(b, ob, sb, 2)
    folder = tmp_path/'saved'
    contract = {'base_identity_sha256': 'base'}
    exp.save_training_state(folder, selector=b, optimizer=ob, scheduler=sb, step=2,
        contract=contract, random_states=[exp.rng_state()], save_fn=save_selector)
    c, oc, sc = create()
    assert exp.restore_training_state(folder, selector=c, optimizer=oc, scheduler=sc,
        contract=contract, load_fn=load_selector) == 2
    steps(c, oc, sc, 2)
    assert all(torch.equal(t, c.state_dict()[k]) for k, t in a.state_dict().items())
