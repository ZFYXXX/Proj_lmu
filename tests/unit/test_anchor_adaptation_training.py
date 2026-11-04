import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from mllm_comp.workflow import anchor_training as training
from mllm_comp.workflow.runner import write
from mllm_comp.analysis.common.anchor7b_runtime import AnchorAdapterSpec, create_anchor_adapter, save_adapter, load_adapter
from mllm_comp.compressors.image.context_visual_topk import ContextVisualTopK, ContextVisualTopKSpec, save_selector


def toy_state():
    features = torch.randn(256, 16, generator=torch.Generator().manual_seed(9))
    hidden = torch.randn(1, 268, 16, generator=torch.Generator().manual_seed(7))
    hidden[0, 8:264] = features
    labels = torch.full((1, 268), -100)
    labels[0, 266:] = torch.tensor([1, 2])
    ids = torch.ones(1, 268, dtype=torch.long)
    ids[0, 8:264] = 99
    return {'features': features, 'height': 16, 'width': 16, 'hidden': hidden,
            'positions': torch.arange(268)[None, None].expand(3, 1, -1).clone(), 'visual': list(range(8, 264)),
            'prefix': 266, 'teacher_inputs': {'input_ids': ids, 'labels': labels, 'attention_mask': torch.ones_like(ids)}}


def test_fixed_anchors_and_pool_gradients_share_the_deployed_partition():
    state = toy_state()
    anchor = ContextVisualTopK(ContextVisualTopKSpec(16, 8, 128, 42, 'image')).requires_grad_(False)
    pool = copy.deepcopy(anchor).requires_grad_(True)
    arm = {'method': 'importance_fps_pool', 'pooling': 'importance', 'train_pool': True}
    initial = training.parameter_hash(anchor)
    frozen = training.cluster_input(state, arm, anchor)
    joint = training.cluster_input(state, arm, anchor, pool)
    assert frozen['members'] == joint['members'] and frozen['indices'] == joint['indices']
    torch.testing.assert_close(frozen['tokens'], joint['tokens'], rtol=0, atol=0)
    mlp = create_anchor_adapter(AnchorAdapterSpec(16, 8))
    mlp(joint)['tokens'].square().mean().backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in pool.parameters())
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in mlp.parameters())
    assert all(p.grad is None for p in anchor.parameters()) and training.parameter_hash(anchor) == initial


class FrozenToyLLM(nn.Module):
    def gradient_checkpointing_enable(self, **kwargs):
        pass

    def forward(self, inputs_embeds, **kwargs):
        denominator = torch.arange(1, inputs_embeds.shape[1]+1, device=inputs_embeds.device)[None, :, None]
        return SimpleNamespace(last_hidden_state=inputs_embeds.cumsum(1) / denominator)


def test_actual_training_loop_checkpoint_resume_matches_continuous_cpu(tmp_path, monkeypatch):
    torch.set_num_threads(1)
    identity = {'hidden_size': 16, 'identity_sha256': 'toy-base'}
    write(tmp_path / 'identity.json', identity)
    mlp = create_anchor_adapter(AnchorAdapterSpec(16, 8))
    selector = ContextVisualTopK(ContextVisualTopKSpec(16, 8, 128, 42, 'image'))
    save_adapter(mlp, tmp_path / 'old', identity)
    save_selector(selector, tmp_path / 'selector', base_identity='toy-base', provenance={})
    config = {'profile': {}, 'identity': str(tmp_path / 'identity.json'),
        'checkpoints': {'old_mlp': str(tmp_path / 'old')}, 'scorer_checkpoint': str(tmp_path / 'selector'),
        'training': {'arms': {'joint': {'method': 'importance_fps_pool', 'pooling': 'importance', 'train_pool': True}},
            'seed': 42, 'steps': 320, 'smoke_steps': 2, 'global_batch': 8, 'learning_rate': .0001,
            'weight_decay': .01, 'warmup_steps': 16, 'teacher_kl_weight': .5, 'clip_grad_norm': 1., 'save_every': 16}}
    roots = [tmp_path / 'continuous', tmp_path / 'resumed']
    for root in roots:
        root.mkdir(); write(root / 'workflow.json', {'signature': 'same-protocol', 'config': config})
    model = nn.Module(); model.lm_head = nn.Linear(16, 7, bias=False); model.requires_grad_(False)
    bundle = {'device': torch.device('cpu'), 'model': model, 'llm': FrozenToyLLM()}
    examples = [{'sample_id': f'sample_{i}'} for i in range(16)]
    from scripts.train.qwen25_vl import train_learned_hybrid
    monkeypatch.setattr(train_learned_hybrid, 'prepare_example', lambda bundle, example: toy_state())
    first = training.train_arm(bundle, examples, roots[0], 'joint', phase='smoke')
    first_rng = torch.get_rng_state().clone()
    training.train_arm(bundle, examples, roots[1], 'joint', phase='smoke', stop_after=1)
    second = training.train_arm(bundle, examples, roots[1], 'joint', phase='smoke')
    assert torch.equal(first_rng, torch.get_rng_state())
    assert training.parameter_hash(load_adapter(first, identity, 'cpu')) == training.parameter_hash(load_adapter(second, identity, 'cpu'))
    from mllm_comp.compressors.image.context_visual_topk import load_selector
    assert training.parameter_hash(load_selector(first / 'pool_selector', base_identity='toy-base', device='cpu')) == training.parameter_hash(load_selector(second / 'pool_selector', base_identity='toy-base', device='cpu'))
    rows = [json.loads(line) for line in (roots[1] / 'arms/joint/smoke/training.jsonl').read_text().splitlines()]
    assert [row['sample_indices'] for row in rows] == [list(range(8)), list(range(8, 16))]
    assert all(row['pool_gradient_norm'] > 0 for row in rows)
    assert all(p.grad is None for p in model.parameters())


def test_gpu_queue_never_uses_a_device_with_existing_compute_processes(monkeypatch):
    def query(command, **kwargs):
        return '0, GPU-zero\n1, GPU-one\n' if '--query-gpu=index,uuid' in command else 'GPU-zero, 999\n'
    monkeypatch.setattr(training.subprocess, 'check_output', query)
    assert training.idle_devices(['0', '1']) == ['1']
    with pytest.raises(ValueError, match='unavailable'):
        training.idle_devices(['8'])


def test_explicit_shared_gpu_queue_checks_free_memory_and_selected_device(monkeypatch):
    monkeypatch.setattr(training.subprocess, 'check_output',
                        lambda *args, **kwargs: '0, 76000\n1, 46000\n')
    assert training.runnable_devices(['0'], 60) == ['0']
    assert training.runnable_devices(['1'], 60) == []
    with pytest.raises(ValueError, match='positive'):
        training.runnable_devices(['0'], 0)
    with pytest.raises(ValueError, match='unavailable'):
        training.runnable_devices(['8'], 60)


def test_machine_schedule_shares_only_explicitly_authorized_gpu(tmp_path):
    path = tmp_path / 'schedule.json'
    write(path, {'gpus': ['0', '1'], 'share_gpus': ['0'], 'share_min_free_gib': 50})
    assert training.scheduling_arguments({'training_schedule': str(path)}) == [
        '--gpus', '0', '1', '--share-min-free-gib', '50', '--share-gpus', '0']
    assert training.scheduling_arguments({}) == []


def test_scheduler_adopts_existing_worker_without_restarting_it(tmp_path):
    import subprocess
    import sys
    process = subprocess.Popen([sys.executable, '-c', 'import time; print("ready", flush=True); time.sleep(30)',
        'mllm_comp.workflow.anchor_training', 'worker', '--root', str(tmp_path), '--arm', 'R1'],
        stdout=subprocess.PIPE, text=True)
    try:
        assert process.stdout.readline().strip() == 'ready'
        adopted = training.AdoptedWorker(process.pid, tmp_path, 'R1')
        assert adopted.pid == process.pid and adopted.poll() is None
        with pytest.raises(ValueError, match='unrelated'):
            training.AdoptedWorker(process.pid, tmp_path, 'T0')
        process.terminate(); process.wait(timeout=5)
        assert adopted.poll() == 1
        (tmp_path / 'arms/R1').mkdir(parents=True)
        write(tmp_path / 'arms/R1/complete.json', {'complete': True})
        assert adopted.poll() == 0
    finally:
        if process.poll() is None:
            process.kill(); process.wait()
