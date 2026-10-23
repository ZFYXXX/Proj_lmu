import json
from pathlib import Path
import random

import numpy as np
import pytest
import torch
from types import SimpleNamespace

from mllm_comp.analysis.common import learned_topk_experiment as experiment
from mllm_comp.compressors.image.learned_visual_topk import LearnedVisualTopK, LearnedVisualTopKSpec
from scripts.eval.lmms_eval.report_learned_visual_topk import paired_image_bootstrap
from scripts.train.qwen25_vl.run_learned_visual_topk import check_eval, eval_config, METHODS


def test_gpu_guard_never_borrows_foreign_or_busy_devices():
    processes = [{'gpu': 0, 'pid': 1, 'uid': 123}, {'gpu': 1, 'pid': 2, 'uid': 456}]
    assert experiment.available_gpus([0, 1], processes=processes, uid=123) == []
    assert experiment.available_gpus([0, 1], processes=processes, uid=123, allow_own=True) == [0]
    assert experiment.available_gpus([0, 1], processes=[], uid=123) == [0, 1]


def test_global_batch_order_is_identical_with_one_or_two_gpus():
    for step in range(320):
        single = experiment.batch_indices(step)
        dual = [i for rank in range(2) for i in experiment.batch_indices(step, world_size=2, rank=rank)]
        assert sorted(dual) == single == list(range(step * 8, (step + 1) * 8))
    with pytest.raises(ValueError):
        experiment.batch_indices(320)


def test_owned_output_does_not_allow_shared_or_broad_directories():
    for path in ('/home/WuHao/test', '/', '/home/ZhouFengyu'):
        with pytest.raises(ValueError):
            experiment.owned_output(path)


def test_development_selection_deduplicates_without_targets():
    rows = [{'row': i, 'image_fingerprint': str(i // 2)} for i in range(600)]
    a = experiment.development_indices(rows)
    assert a == experiment.development_indices(list(reversed(rows)))
    assert len(a) == 250 and len({rows[i]['image_fingerprint'] for i in a}) == 250
    assert all(i % 2 == 0 for i in a)
    assert a == experiment.development_indices(rows)
    with pytest.raises(ValueError, match='overlap'):
        experiment.reject_overlap([rows[0]], rows)


def test_loss_weights_gradients_and_no_teacher_gradient():
    scorer = LearnedVisualTopK(LearnedVisualTopKSpec(8, width=4, target_count=3))
    selection = scorer(torch.randn(7, 8), soft=True)
    teacher = torch.randn(2, 5, requires_grad=True)
    student = selection['tokens'][:2, :5]
    loss, metrics = experiment.training_loss(student, teacher, torch.tensor([1, 2]), selection, 0)
    loss.backward()
    assert metrics['bce_weight'] == .1 and experiment.bce_weight(319) == 3.
    assert teacher.grad is None
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in scorer.parameters())
    assert sum(float(p.grad.abs().sum()) for p in scorer.parameters()) > 0


def test_full_optimizer_rng_schedule_resume_is_exact_and_rejects_mutation(tmp_path, monkeypatch):
    monkeypatch.setattr(experiment, 'OWNER', tmp_path)
    contract = {'base_identity_sha256': 'test', 'source': 'snapshot', 'world_size': 1}
    def create():
        module = LearnedVisualTopK(LearnedVisualTopKSpec(8, width=4, target_count=3))
        opt = torch.optim.AdamW(module.parameters(), lr=2e-4, weight_decay=.01)
        sched = torch.optim.lr_scheduler.LambdaLR(opt, experiment.lr_multiplier)
        return module, opt, sched
    def seed():
        random.seed(42); np.random.seed(42); torch.manual_seed(42)
    def advance(module, opt, sched, start, stop):
        for step in range(start, stop):
            x = torch.randn(7, 8) * (1 + random.random() + np.random.rand())
            selection = module(x, soft=True)
            student, teacher = selection['tokens'][:2, :5], torch.randn(2, 5)
            loss, _ = experiment.training_loss(student, teacher, torch.tensor([1, 2]), selection, step)
            loss.backward(); torch.nn.utils.clip_grad_norm_(module.parameters(), 1.)
            opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
    seed(); a, oa, sa = create(); advance(a, oa, sa, 0, 4)
    seed(); b, ob, sb = create(); advance(b, ob, sb, 0, 2)
    folder = tmp_path / 'checkpoint'
    experiment.save_training_state(folder, selector=b, optimizer=ob, scheduler=sb,
        step=2, contract=contract, random_states=[experiment.rng_state()])
    c, oc, sc = create()
    step = experiment.restore_training_state(folder, selector=c, optimizer=oc, scheduler=sc, contract=contract)
    advance(c, oc, sc, step, 4)
    assert all(torch.equal(a.state_dict()[k], c.state_dict()[k]) for k in a.state_dict())
    assert sa.state_dict() == sc.state_dict()
    for key, state in oa.state_dict()['state'].items():
        for field, value in state.items():
            assert torch.equal(value, oc.state_dict()['state'][key][field])
    with pytest.raises(ValueError, match='contract changed'):
        experiment.restore_training_state(folder, selector=c, optimizer=oc, scheduler=sc,
            contract={**contract, 'world_size': 2})


@pytest.mark.parametrize('method', METHODS)
def test_all_native_eval_configs_use_same_checkpoint_budget_and_tasks(tmp_path, method):
    identity = tmp_path / 'identity.json'
    identity.write_text(json.dumps({'checkpoint': '/readonly/7B', 'revision': 'local-sha256:identity'}))
    trained = tmp_path / 'training'; trained.mkdir()
    (trained / 'manifest.json').write_text(json.dumps({'status': 'complete', 'steps_completed': 320}))
    (trained / 'latest.json').write_text(json.dumps({'checkpoint': 'checkpoints/step320'}))
    config = eval_config(tmp_path, {'identity_path': str(identity), 'old_adapter': '/readonly/old'}, method, 'dev', tmp_path / 'out')
    assert config['backbone']['model_path'] == '/readonly/7B'
    assert config['eval']['generation'] == {'do_sample': False, 'repetition_penalty': 1.05}
    assert config['eval']['tasks'] == ['learned_topk_chartqa_dev', 'learned_topk_ocrbench_dev']
    from mllm_comp.config.schema import experiment_config_from_dict
    from mllm_comp.config.validators import validate_runnable_config
    from mllm_comp.run_manifest import build_run_manifest
    parsed = experiment_config_from_dict(config)
    validate_runnable_config(parsed, task='eval')
    manifest = build_run_manifest(parsed)
    assert manifest
    if method != 'dense':
        stage, = config['pipeline']['stages']
        assert stage['placement'] == 'projector_level' and stage['execution_phase'] == 'generate'
        assert stage['config'].get('target_count', stage['config'].get('tokens_per_crop')) == 128


def test_bootstrap_is_paired_image_level_and_deterministic():
    a = paired_image_bootstrap([.2, -.1, .3, 0.])
    assert a == paired_image_bootstrap([.2, -.1, .3, 0.])
    assert a['independent_images'] == 4
    assert a['ci95'][0] <= a['mean_image_weighted_delta'] <= a['ci95'][1]
    with pytest.raises(ValueError):
        paired_image_bootstrap([float('nan')])


def test_native_request_performance_is_opt_in_and_records_dense_tokens(tmp_path, monkeypatch):
    from mllm_comp.integrations.lmms_eval.request_performance import request_performance
    monkeypatch.setattr(experiment, 'OWNER', tmp_path)
    monkeypatch.delenv('MLLM_TOPK_PERFORMANCE', raising=False)
    runtime = SimpleNamespace(device='cpu', model=SimpleNamespace(config=SimpleNamespace(image_token_id=10)))
    batch = SimpleNamespace(model_inputs={'input_ids': torch.tensor([[1, 10, 10, 2]])})
    with request_performance(runtime, batch, 'sample'):
        pass
    path = tmp_path / 'performance.jsonl'
    assert not path.exists()
    monkeypatch.setenv('MLLM_TOPK_PERFORMANCE', str(path))
    with request_performance(runtime, batch, 'sample'):
        pass
    record = json.loads(path.read_text())
    assert record['original_visual_tokens'] == 2 and record['status'] == 'complete'
    assert record['generation_s'] >= 0
    with pytest.raises(RuntimeError):
        with request_performance(runtime, batch, 'failed'):
            raise RuntimeError('simulated failure')
    assert json.loads(path.read_text().splitlines()[-1])['status'] == 'failed'


def test_resume_rejects_partial_and_corrupted_checkpoints(tmp_path, monkeypatch):
    monkeypatch.setattr(experiment, 'OWNER', tmp_path)
    module = LearnedVisualTopK(LearnedVisualTopKSpec(8, width=4, target_count=3))
    opt = torch.optim.AdamW(module.parameters())
    sched = torch.optim.lr_scheduler.LambdaLR(opt, experiment.lr_multiplier)
    contract = {'base_identity_sha256': 'test'}
    folder = tmp_path / 'partial'; folder.mkdir()
    with pytest.raises(ValueError, match='Incomplete'):
        experiment.restore_training_state(folder, selector=module, optimizer=opt, scheduler=sched, contract=contract)
    saved = tmp_path / 'saved'
    experiment.save_training_state(saved, selector=module, optimizer=opt, scheduler=sched,
        step=0, contract=contract, random_states=[experiment.rng_state()])
    (saved / 'training_state.pt').write_bytes(b'corrupt')
    with pytest.raises(ValueError, match='integrity'):
        experiment.restore_training_state(saved, selector=module, optimizer=opt, scheduler=sched, contract=contract)


@pytest.mark.parametrize('fault', [None, 'budget', 'duplicate_indices', 'decode', 'layer', 'missing_generation'])
def test_native_eval_acceptance_rejects_corrupt_traces(tmp_path, fault):
    names = ['learned_topk_chartqa', 'learned_topk_ocrbench']
    (tmp_path / 'fixture_results.json').write_text(json.dumps({'results': {n: {} for n in names}}))
    traces, performance = [], []
    for name in names:
        key = 'relaxed_overall' if name.endswith('chartqa') else 'ocrbench_accuracy'
        rows = [{'doc_id': i, key: 1.} for i in range(8)]
        (tmp_path / f'fixture_samples_{name}.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in rows))
        for i in range(8):
            sid = f'{name}:test:{i}:{i}'
            delta = {'source_count': 256, 'result_count': 128, 'selected_indices': list(range(128)),
                     'decode_position_verified': True, 'observed_visual_tokens_per_decoder_layer': [128] * 28}
            traces.append({'request': {'sample_id': sid}, 'runtime': {'activations': [{'selection': delta}]}})
            performance.append({'sample_id': sid, 'status': 'complete', 'original_visual_tokens': 256, 'generation_s': .5})
    delta = traces[0]['runtime']['activations'][0]['selection']
    if fault == 'budget':
        delta['result_count'] = 127
    elif fault == 'duplicate_indices':
        delta['selected_indices'][1] = 0
    elif fault == 'decode':
        delta['decode_position_verified'] = False
    elif fault == 'layer':
        delta['observed_visual_tokens_per_decoder_layer'][1] = 256
    elif fault == 'missing_generation':
        performance.pop()
    (tmp_path / 'compression_runtime.requests.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in traces))
    (tmp_path / 'request_performance.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in performance))
    if fault:
        with pytest.raises(ValueError):
            check_eval(tmp_path, 'trained_selector', 'smoke')
    else:
        checked = check_eval(tmp_path, 'trained_selector', 'smoke')
        assert checked['requests'] == checked['first_decode_verified_requests'] == 16
