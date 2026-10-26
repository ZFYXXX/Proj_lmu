import json
import pytest
import torch

from mllm_comp.compressors.image.context_visual_topk import (
    ContextVisualTopK, ContextVisualTopKSpec, load_selector, save_selector, prompt_question,
)
from mllm_comp.compressors.image.learned_visual_topk import LearnedVisualTopK, LearnedVisualTopKSpec
from mllm_comp.compressors.comparison.context_topk import QwenContextTopKComparison
from tests.unit.test_projector_comparison_evaluation import tiny_state


@pytest.mark.parametrize('mode', ['question', 'image'])
def test_zero_initialization_matches_old_scorer_and_gradients(mode):
    old = LearnedVisualTopK(LearnedVisualTopKSpec(12, 8, 3))
    model = ContextVisualTopK(ContextVisualTopKSpec(12, 8, 3, context_mode=mode), visual=old)
    x, q = torch.randn(11, 12), torch.randn(5, 12)
    result = model(x, question=q, soft=True)
    from mllm_comp.analysis.common.topk_training_modes import selection_diagnostics
    assert selection_diagnostics(result)['physical_visual_tokens'] == 11
    torch.testing.assert_close(result['scores'], old(x)['scores'], rtol=0, atol=0)
    torch.testing.assert_close(result['soft_weights'].sum(), torch.tensor(3.), atol=1e-5, rtol=0)
    (result['tokens'] * torch.randn_like(result['tokens'])).sum().backward()
    assert model.context_projection.weight.grad.isfinite().all()
    assert model.context_projection.weight.grad.norm() > 0
    assert result['indices'].tolist() == sorted(set(result['indices'].tolist()))


def test_question_changes_scores_and_image_control_ignores_question():
    x, a, b = torch.randn(10, 12), torch.randn(4, 12), torch.randn(5, 12)
    for mode in ('question', 'image'):
        model = ContextVisualTopK(ContextVisualTopKSpec(12, 8, 3, context_mode=mode))
        torch.nn.init.normal_(model.context_projection.weight, std=.1)
        same = torch.equal(model(x, question=a)['scores'], model(x, question=b)['scores'])
        assert same == (mode == 'image')
    with pytest.raises(ValueError, match='context'):
        ContextVisualTopK(ContextVisualTopKSpec(12))(x)


def test_prompt_excludes_answer_and_handles_last_user():
    bundle, state, _, _ = tiny_state()
    model, prompt = bundle['model'], state['prompt_inputs']
    a, audit = prompt_question(model, prompt)
    teacher = dict(prompt)
    teacher['input_ids'] = torch.cat((prompt['input_ids'], torch.tensor([[151644, 5, 198, 42, 151645]])), -1)
    teacher['attention_mask'] = torch.ones_like(teacher['input_ids'])
    labels = torch.full_like(teacher['input_ids'], -100)
    labels[:, -2] = 42
    teacher['labels'] = labels
    b, new = prompt_question(model, teacher)
    assert torch.equal(a, b) and audit == new
    assert not a.requires_grad
    different = dict(prompt)
    different['input_ids'] = prompt['input_ids'].clone()
    different['input_ids'][0, 3] = 43
    assert prompt_question(model, different)[1]['question_ids_sha256'] != audit['question_ids_sha256']


@pytest.mark.parametrize('mode', ['question', 'image'])
def test_binding_identity_generate_cache_and_request_isolation(mode):
    bundle, state, _, _ = tiny_state()
    model, prompt = bundle['model'], state['prompt_inputs']
    with torch.no_grad():
        dense = model(**prompt, use_cache=False).logits
    session = QwenContextTopKComparison(model, {'target_count': 128, 'context_mode': mode}).install()
    try:
        with torch.no_grad():
            torch.testing.assert_close(model(**prompt, use_cache=False).logits, dense, rtol=1e-5, atol=1e-6)
    finally:
        session.uninstall()
    session = QwenContextTopKComparison(model, {'target_count': 3, 'context_mode': mode}).install()
    try:
        audits = []
        for token in (9, 43):
            current = dict(prompt)
            current['input_ids'] = prompt['input_ids'].clone()
            current['input_ids'][0, 3] = token
            with torch.no_grad():
                model.generate(**current, max_new_tokens=3, min_new_tokens=3, do_sample=False, pad_token_id=0)
            trace = session.runtime_metadata()['selection']
            assert trace['result_count'] == 3 and trace['decode_position_verified']
            assert trace['observed_visual_tokens_per_decoder_layer'] == [3] * 4
            audits.append(trace.get('question_ids_sha256'))
        if mode == 'question':
            assert audits[0] != audits[1]
    finally:
        session.uninstall()


def test_checkpoint_is_independent_and_integrity_checked(tmp_path):
    module = ContextVisualTopK(ContextVisualTopKSpec(12, 8, 3))
    folder = tmp_path / 'checkpoint'
    save_selector(module, folder, base_identity='base', provenance={'warmstart': 'test'})
    actual = load_selector(folder, base_identity='base', device='cpu')
    assert all(torch.equal(a, actual.state_dict()[k]) for k, a in module.state_dict().items())
    with pytest.raises(ValueError, match='base'):
        load_selector(folder, base_identity='wrong', device='cpu')
    with (folder / 'selector.safetensors').open('ab') as stream:
        stream.write(b'corrupt')
    with pytest.raises(ValueError, match='integrity'):
        load_selector(folder, base_identity='base', device='cpu')


@pytest.mark.parametrize('mode', ['question', 'image'])
def test_exact_context_resume(tmp_path, monkeypatch, mode):
    from mllm_comp.analysis.common import learned_topk_experiment as experiment
    monkeypatch.setattr(experiment, 'OWNER', tmp_path)
    def create():
        model = ContextVisualTopK(ContextVisualTopKSpec(8, 4, 3, context_mode=mode))
        opt = torch.optim.AdamW(model.parameters(), lr=2e-4)
        sched = torch.optim.lr_scheduler.LambdaLR(opt, experiment.lr_multiplier)
        return model, opt, sched
    def advance(model, opt, sched, start, end):
        for step in range(start, end):
            selection = model(torch.randn(7, 8), question=torch.randn(4, 8), soft=True)
            loss = (selection['tokens'] * torch.randn_like(selection['tokens'])).sum()
            loss.backward(); opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
    contract = {'base_identity_sha256': 'test', 'context_mode': mode}
    torch.manual_seed(42); a, oa, sa = create(); advance(a, oa, sa, 0, 4)
    torch.manual_seed(42); b, ob, sb = create(); advance(b, ob, sb, 0, 2)
    folder = tmp_path / 'checkpoint'
    experiment.save_training_state(folder, selector=b, optimizer=ob, scheduler=sb,
        step=2, contract=contract, random_states=[experiment.rng_state()], save_fn=save_selector)
    c, oc, sc = create()
    step = experiment.restore_training_state(folder, selector=c, optimizer=oc, scheduler=sc,
        contract=contract, load_fn=load_selector)
    advance(c, oc, sc, step, 4)
    assert all(torch.equal(a.state_dict()[k], c.state_dict()[k]) for k in a.state_dict())
    assert sa.state_dict() == sc.state_dict()
    for key, row in oa.state_dict()['state'].items():
        for field, value in row.items():
            assert torch.equal(value, oc.state_dict()['state'][key][field])


@pytest.mark.parametrize('method', ['dense', 'old_mlp', 'frozen_visual', 'image_context', 'question_context'])
def test_native_context_configs_three_tasks_and_checkpoint_isolation(tmp_path, method):
    from scripts.train.qwen25_vl.run_context_visual_topk import eval_config
    from mllm_comp.config.schema import experiment_config_from_dict
    from mllm_comp.config.validators import validate_runnable_config
    from mllm_comp.run_manifest import build_run_manifest
    identity = tmp_path / 'identity.json'
    identity.write_text(json.dumps({'checkpoint': '/readonly/7B', 'revision': 'local-sha256:test'}))
    for folder in [tmp_path / 'reference/training', *[tmp_path / 'arms' / m / 'training' for m in ('image', 'question')]]:
        folder.mkdir(parents=True)
        (folder / 'manifest.json').write_text(json.dumps({'status': 'complete', 'steps_completed': 320}))
        (folder / 'latest.json').write_text(json.dumps({'step': 320, 'checkpoint': 'checkpoints/step320'}))
    protocol = {'identity_path': str(identity), 'old_adapter': '/readonly/old', 'reference_train': str(tmp_path / 'reference')}
    cfg = eval_config(tmp_path, protocol, method, 'dev', tmp_path / 'eval')
    assert cfg['eval']['tasks'] == ['context_topk_chartqa_dev', 'context_topk_ocrbench_dev', 'context_topk_ai2d_dev']
    assert cfg['backbone']['model_path'] == '/readonly/7B'
    assert cfg['eval']['generation'] == {'do_sample': False, 'repetition_penalty': 1.05}
    parsed = experiment_config_from_dict(cfg)
    validate_runnable_config(parsed, task='eval')
    assert build_run_manifest(parsed)
    if method.endswith('_context'):
        stage, = cfg['pipeline']['stages']
        assert stage['algorithm'] == 'context_visual_topk'
        assert stage['config']['context_mode'] == method.removesuffix('_context')
        assert stage['config']['target_count'] == 128


def test_cluster_bootstrap_uses_question_weighted_estimand():
    from scripts.train.qwen25_vl.run_context_visual_topk import clustered_difference
    result = clustered_difference([1., 1., 0.], [0., 0., 0.], ['A', 'A', 'B'], resamples=100)
    assert result['delta'] == pytest.approx(2 / 3)
    assert result['independent_images'] == 2
    assert result == clustered_difference([1., 1., 0.], [0., 0., 0.], ['A', 'A', 'B'], resamples=100)
    with pytest.raises(ValueError, match='finite'):
        clustered_difference([float('nan')], [0], ['A'])


def test_context_soft_native_binding_freezes_base():
    bundle, state, _, _ = tiny_state()
    model = bundle['model'].requires_grad_(False)
    session = QwenContextTopKComparison(model, {'target_count': 3, 'training_mode': 'soft'}, phase='forward').install()
    try:
        labels = torch.full_like(state['prompt_inputs']['input_ids'], -100)
        labels[:, -1] = 3
        result = model(**state['prompt_inputs'], labels=labels, use_cache=False)
        result.loss.backward()
        assert all(p.grad is None for p in model.parameters())
        assert session.selector.context_projection.weight.grad.norm() > 0
        assert result.logits.shape[1] == state['prefix']
    finally:
        session.uninstall()
