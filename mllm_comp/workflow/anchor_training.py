"""Config-driven warm-start anchor training, with a durable owner-scoped GPU queue."""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import random
import re
import signal
from string import Template
import subprocess
import sys
import time
import traceback

from .catalog import ROOT, DEFAULT_CATALOG, load_catalog
from .runner import read, write, digest, freeze


def now():
    return datetime.now(timezone.utc).isoformat()


def plan(profile_path, config_path):
    profile = json.loads(Template(Path(profile_path).read_text()).substitute(REPO=str(ROOT)))
    config = read(config_path)
    if config.get('schema_version') != 1 or config.get('steps') != 320 or config.get('global_batch') != 8:
        raise ValueError('First adaptation protocol fixes 320 steps and global batch eight')
    initialization = config.get('initialization')
    if initialization not in {'same_old_step320_mlp_warm_start_reset_optimizer', 'same_seed_zero_residual_mlp'}:
        raise ValueError('Unknown MLP initialization protocol')
    manifest = Path(profile['training_manifest'])
    if digest(manifest) != config['training_manifest_sha256']:
        raise ValueError('Historical training manifest changed')
    data = read(manifest)
    entries = data['entries']
    if len(entries) != 2560 or len({e['image_fingerprint'] for e in entries}) != 2560:
        raise ValueError('Expected exactly 2560 distinct training images')
    manifest_mode = config.get('manifest_mode', 'historical_control')
    if manifest_mode == 'historical_control':
        if Counter(e['source'] for e in entries) != {'textvqa_train': 2048, 'scienceqa_train': 512} or data['selection_reads_answers']:
            raise ValueError('Historical training composition/selection changed')
    elif manifest_mode == 'balanced_arm_only':
        from .training_data import validate_manifest
        validate_manifest(manifest, digest(manifest))
    else:
        raise ValueError(f'Unknown training manifest mode: {manifest_mode}')
    for path in {e['path'] for e in entries}:
        source_path = Path(path)
        if not source_path.is_absolute():
            source_path = manifest.parent / source_path
        if not source_path.is_file():
            raise ValueError(f'Training parquet missing: {source_path}')
    for name, arm in config['arms'].items():
        if name.startswith('.') or '/' in name or arm['method'] not in {'old_mlp', 'importance_fps', 'importance_fps_pool'}:
            raise ValueError('Invalid training arm')
        if arm['train_pool'] and arm['pooling'] != 'importance':
            raise ValueError('Pool scorer training needs importance pooling')
        if arm.get('training_manifest'):
            from .training_data import validate_manifest
            path = Path(Template(arm['training_manifest']).substitute(REPO=str(ROOT)))
            validate_manifest(path, arm['training_manifest_sha256'], arm.get('source_counts'))
            arm['training_manifest'] = str(path)
    heads = Path(profile['checkpoint_root'])
    identity = profile.get('training_identity', profile.get('identity', str(heads / 'model_identity.json')))
    old, scorer = str(heads / 'checkpoints/old_mlp'), str(heads / 'checkpoints/image_context_selector')
    pin = read(identity)
    if Path(pin['checkpoint']).resolve() != Path(profile['model']).resolve():
        raise ValueError('This local trainer needs the pinned original model location')
    extra = [manifest]
    for source in data.get('sources', []):
        source_path = Path(source['path'])
        extra.append(source_path if source_path.is_absolute() else manifest.parent / source_path)
    for arm in config['arms'].values():
        if arm.get('training_manifest'):
            arm_manifest = Path(arm['training_manifest'])
            extra.append(arm_manifest)
            for source in read(arm_manifest).get('sources', []):
                source_path = Path(source['path'])
                extra.append(source_path if source_path.is_absolute() else arm_manifest.parent / source_path)
    return {'profile': profile, 'run_id': config['run_id'], 'training': config,
            'extra_input_files': sorted({str(path) for path in extra}),
            'manifest_sha256': digest(manifest), 'identity': identity,
            'checkpoints': {'old_mlp': old}, 'scorer_checkpoint': scorer,
            'catalog_path': str(DEFAULT_CATALOG), 'catalog_sha256': digest(DEFAULT_CATALOG),
            'requested_cells': [], 'experiment': {'schema_version': 1, 'method_tasks': {
                'importance_fps': config['evaluation_tasks'], 'importance_fps_pool': config['evaluation_tasks']}}}


def cluster_input(state, arm, anchor_selector=None, pool_selector=None):
    """Same kernels as deployment. Hard assignments never receive gradients."""
    import torch
    from mllm_comp.analysis.common.anchor7b_runtime import anchor_base
    from mllm_comp.compressors.image.spatial_anchor_ablation import spatial_anchor_merge
    from mllm_comp.compressors.comparison.importance_anchor_adapter import importance_pool
    if arm['method'] == 'old_mlp':
        return anchor_base(state)
    with torch.no_grad():
        anchor_p = anchor_selector(state['features'], soft=True)['soft_weights'].detach()
        base = spatial_anchor_merge(state['features'], state['height'], state['width'], 128, importance=anchor_p)
    if arm['pooling'] == 'importance':
        weights = pool_selector(state['features'], soft=True)['soft_weights'] if pool_selector is not None else anchor_p
        base['tokens'], _ = importance_pool(state['features'], base['members'], weights)
    return base


def parameter_hash(module):
    value = hashlib.sha256()
    for name, tensor in module.state_dict().items():
        value.update(name.encode())
        value.update(tensor.detach().float().cpu().contiguous().numpy().tobytes())
    return value.hexdigest()


def initialize_adapter(config, checkpoint, identity, device):
    from mllm_comp.analysis.common.anchor7b_runtime import load_adapter, create_anchor_adapter, AnchorAdapterSpec
    if config.get('initialization') == 'same_seed_zero_residual_mlp':
        spec = read(Path(checkpoint) / 'projector_config.json')['spec']
        return create_anchor_adapter(AnchorAdapterSpec(**spec)).to(device).train()
    return load_adapter(checkpoint, identity, device).train()


def train_arm(bundle, examples, root, name, *, phase='train', stop_after=None):
    import numpy as np
    import torch
    import torch.nn.functional as F
    from mllm_comp.analysis.common.anchor7b_runtime import load_adapter, save_adapter, compact_anchor, answer_logits
    from mllm_comp.compressors.image.context_visual_topk import load_selector, save_selector
    from scripts.train.qwen25_vl.train_anchor7b_ddp import recover_log
    from mllm_comp.analysis.common.condition_ledger import append_json

    saved = read(root / 'workflow.json')
    p, config = saved['config'], saved['config']['training']
    arm, device = config['arms'][name], bundle['device']
    cuda = torch.device(device).type == 'cuda'
    folder = root / 'arms' / name / phase
    folder.mkdir(parents=True, exist_ok=True)
    identity = read(p['identity'])
    seed = config['seed']
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if cuda:
        torch.cuda.manual_seed_all(seed)
    mlp = initialize_adapter(config, p['checkpoints']['old_mlp'], identity, device)
    initial_mlp_hash = parameter_hash(mlp)
    if config.get('initial_parameter_sha256') and initial_mlp_hash != config['initial_parameter_sha256']:
        raise ValueError('Arms did not start with the pinned identical MLP parameters')
    anchor = load_selector(p['scorer_checkpoint'], base_identity=identity['identity_sha256'], device=device).eval().requires_grad_(False)
    pool = (load_selector(p['scorer_checkpoint'], base_identity=identity['identity_sha256'], device=device).train()
            if arm['train_pool'] else None)
    parameters = list(mlp.parameters()) + (list(pool.parameters()) if pool is not None else [])
    optimizer = torch.optim.AdamW(parameters, lr=config['learning_rate'], weight_decay=config['weight_decay'])
    def factor(step):
        warmup = config['warmup_steps']
        return (step + 1) / warmup if step < warmup else .5 * (1 + math.cos(math.pi * min(1., (step - warmup) / (config['steps'] - warmup))))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, factor)
    binding_hash = hashlib.sha256(json.dumps({'signature': saved['signature'], 'arm': name, 'phase': phase}, sort_keys=True).encode()).hexdigest()
    target = config['smoke_steps'] if phase == 'smoke' else config['steps']
    step = 0
    initial_anchor_hash = parameter_hash(anchor)
    if (folder / 'latest.json').is_file():
        pointer = read(folder / 'latest.json')
        checkpoint = folder / pointer['checkpoint']
        if digest(checkpoint / 'projector.safetensors') != pointer['mlp_sha256']:
            raise ValueError('Resume MLP checksum mismatch')
        if digest(checkpoint / 'training_state.pt') != pointer['state_sha256']:
            raise ValueError('Resume state checksum mismatch')
        state = torch.load(checkpoint / 'training_state.pt', map_location=device, weights_only=False)
        if state['binding_hash'] != binding_hash:
            raise ValueError('Training source/config/data/initialization changed')
        mlp.load_state_dict(load_adapter(checkpoint, identity, device).state_dict())
        if pool is not None:
            pool.load_state_dict(load_selector(checkpoint / 'pool_selector', base_identity=identity['identity_sha256'], device=device).state_dict())
        optimizer.load_state_dict(state['optimizer']); scheduler.load_state_dict(state['scheduler'])
        step = state['step']
        random.setstate(state['python_rng']); np.random.set_state(state['numpy_rng'])
        torch.set_rng_state(state['torch_rng'].cpu())
        if cuda:
            torch.cuda.set_rng_state(state['cuda_rng'].cpu(), device)
        recover_log(folder / 'training.jsonl', step)
    status = {'status': 'running', 'arm': name, 'phase': phase, 'step': step, 'target_steps': target,
              'trainable_parameters': sum(v.numel() for v in parameters), 'mlp_parameters': sum(v.numel() for v in mlp.parameters()),
              'pool_parameters': sum(v.numel() for v in pool.parameters()) if pool is not None else 0,
              'anchor_scorer_frozen': True, 'seed': seed, 'global_batch': config['global_batch'],
              'optimizer_reset_at_warm_start': True, 'updated_utc': now()}
    status.update(initialization=config.get('initialization', 'warm_start'),
                  initial_mlp_sha256=initial_mlp_hash, anchor_scorer_sha256=initial_anchor_hash)
    write(folder / 'manifest.json', status)
    llm = bundle['llm']
    llm.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
    started = time.monotonic()
    try:
        while step < min(target, stop_after or target):
            optimizer.zero_grad(set_to_none=True)
            totals = [0., 0., 0.]
            for index in range(step * config['global_batch'], (step + 1) * config['global_batch']):
                from scripts.train.qwen25_vl.train_learned_hybrid import prepare_example
                state = prepare_example(bundle, examples[index])
                base = cluster_input(state, arm, anchor, pool)
                llm.eval()
                with torch.no_grad():
                    teacher, labels = answer_logits(bundle, state['hidden'], state['positions'], state['teacher_inputs']['labels'])
                llm.train()
                with torch.autocast(torch.device(device).type, dtype=torch.bfloat16, enabled=cuda):
                    output = mlp(base)
                    compact = compact_anchor(state, output)
                    student, targets = answer_logits(bundle, compact['hidden'], compact['positions'], compact['labels'])
                    if not torch.equal(labels, targets):
                        raise ValueError('Teacher/student answer labels changed')
                    ce = F.cross_entropy(student, targets)
                    kl = F.kl_div(F.log_softmax(student, -1), F.softmax(teacher, -1), reduction='batchmean').clamp_min(0)
                    loss = ce + config['teacher_kl_weight'] * kl
                if not torch.isfinite(loss):
                    raise ValueError('Nonfinite training loss')
                (loss / config['global_batch']).backward()
                for position, item in enumerate((loss, ce, kl)):
                    totals[position] += float(item.detach()) / config['global_batch']
                del state, base, teacher, student, compact, output, loss, ce, kl
            if any(v.grad is not None for v in bundle['model'].parameters()) or any(v.grad is not None for v in anchor.parameters()):
                raise ValueError('Frozen backbone or anchor scorer received gradients')
            norm = torch.nn.utils.clip_grad_norm_(parameters, config['clip_grad_norm'])
            if not torch.isfinite(norm) or float(norm) == 0:
                raise ValueError('Invalid trainable gradients')
            pool_grad = sum(float(v.grad.detach().square().sum()) for v in pool.parameters() if v.grad is not None) ** .5 if pool is not None else None
            optimizer.step(); scheduler.step(); step += 1
            append_json(folder / 'training.jsonl', {'step': step, 'sample_indices': list(range((step-1)*8, step*8)),
                'sample_ids': [e['sample_id'] for e in examples[(step-1)*8:step*8]], 'loss': totals[0],
                'answer_ce': totals[1], 'teacher_kl': totals[2], 'gradient_norm': float(norm), 'pool_gradient_norm': pool_grad,
                'learning_rate': scheduler.get_last_lr()[0], 'elapsed_seconds': time.monotonic() - started})
            if step == 1 or step % config['save_every'] == 0 or step == target or step == stop_after:
                checkpoint = folder / 'checkpoints' / f'step_{step:06d}_{time.time_ns()}'
                save_adapter(mlp, checkpoint, identity)
                if pool is not None:
                    save_selector(pool, checkpoint / 'pool_selector', base_identity=identity['identity_sha256'],
                                  provenance={'training_arm': name, 'step': step, 'anchor_selection': 'fixed original scorer'})
                torch.save({'binding_hash': binding_hash, 'step': step, 'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict(), 'python_rng': random.getstate(), 'numpy_rng': np.random.get_state(),
                    'torch_rng': torch.get_rng_state(), 'cuda_rng': torch.cuda.get_rng_state(device) if cuda else None}, checkpoint / 'training_state.pt')
                with torch.random.fork_rng(devices=[]):
                    restored = load_adapter(checkpoint, identity, device)
                if parameter_hash(restored) != parameter_hash(mlp):
                    raise ValueError('MLP save/load changed parameters')
                del restored
                write(folder / 'latest.json', {'step': step, 'checkpoint': str(checkpoint.relative_to(folder)),
                    'mlp_sha256': digest(checkpoint / 'projector.safetensors'), 'state_sha256': digest(checkpoint / 'training_state.pt')})
            status.update(step=step, updated_utc=now(), last_loss=totals[0])
            write(folder / 'manifest.json', status)
        if parameter_hash(anchor) != initial_anchor_hash:
            raise ValueError('Fixed anchor scorer parameters changed')
        status.update(status='complete' if step == target else 'checkpointed', step=step,
                      anchor_weights_unchanged=True, updated_utc=now())
        write(folder / 'manifest.json', status)
        return folder / read(folder / 'latest.json')['checkpoint']
    except BaseException as exc:
        write(folder / 'manifest.json', {**status, 'status': 'failed', 'error': str(exc), 'traceback': traceback.format_exc()})
        raise


def evaluate(root, name, checkpoint):
    saved = read(root / 'workflow.json')
    p, source = saved['config'], Path(saved['source_root'])
    arm = p['training']['arms'][name]
    folder = root / 'arms' / name / 'evaluation'
    folder.mkdir(exist_ok=True)
    catalog = load_catalog()
    if arm['train_pool']:
        catalog[arm['method']]['options']['pool_scorer_path'] = str(checkpoint / 'pool_selector')
    write(folder / 'methods.json', {'schema_version': 1, 'methods': catalog})
    write(folder / 'experiment.json', {'schema_version': 1, 'method_tasks': {arm['method']: p['training']['evaluation_tasks']}})
    command = [sys.executable, '-u', str(source / 'scripts/eval/lmms_eval/qwen25_portable_queue.py'),
        '--data-root', p['profile']['data_root'], '--model', p['profile']['model'], '--output', str(folder / 'results'),
        '--cache-dir', str(root / 'cache'), '--experiment', str(folder / 'experiment.json'), '--method-catalog', str(folder / 'methods.json'),
        '--reference-identity', p['identity'], '--checkpoint', f"{arm['method']}={checkpoint}",
        '--resident-model', '--reference-only', '--gpu', '0', '--resume', '--chunk-size', '1024']
    if arm['method'] != 'old_mlp':
        command += ['--scorer-checkpoint', p['scorer_checkpoint']]
    if saved.get('result_bundle'):
        command += ['--result-bundle', saved['result_bundle']]
    with (folder / 'worker.log').open('a') as stream:
        subprocess.run(command, cwd=source, stdout=stream, stderr=subprocess.STDOUT, check=True)


def worker(root, name):
    import gc
    import torch
    from mllm_comp.analysis.common.anchor7b_runtime import load_runtime
    from scripts.train.qwen25_vl.train_learned_hybrid import load_examples
    saved = read(root / 'workflow.json')
    p = saved['config']
    for relative, expected in saved['source_hashes'].items():
        if digest(Path(saved['source_root']) / relative) != expected:
            raise ValueError(f'Frozen training source changed: {relative}')
    for path, expected in saved['input_hashes'].items():
        actual = saved['result_bundle'] if path == p['profile'].get('result_bundle') else path
        if digest(actual) != expected:
            raise ValueError(f'Frozen training input changed: {path}')
    if digest(root / 'training_samples.json') != p['manifest_sha256']:
        raise ValueError('Frozen training manifest changed')
    manifest = root / f'training_samples_{name}.json'
    if manifest.exists():
        expected = p['training']['arms'][name]['training_manifest_sha256']
        if digest(manifest) != expected:
            raise ValueError('Arm training manifest changed')
        from .training_data import load_normalized_examples
        examples = load_normalized_examples(manifest)
    else:
        examples = load_examples(root / 'training_samples.json')
    bundle = load_runtime(read(p['identity']), 'cuda:0')
    # Resume a two-step compact-LLM smoke before touching the formal optimizer.
    train_arm(bundle, examples, root, name, phase='smoke', stop_after=1)
    train_arm(bundle, examples, root, name, phase='smoke', stop_after=2)
    checkpoint = train_arm(bundle, examples, root, name)
    if read(root / 'arms' / name / 'train/latest.json')['step'] != 320:
        raise ValueError('Only fixed step320 can enter full evaluation')
    del bundle, examples
    gc.collect(); torch.cuda.empty_cache()
    if not p['training'].get('defer_evaluation', False):
        evaluate(root, name, checkpoint)
    write(root / 'arms' / name / 'complete.json', {
        'arm': name, 'checkpoint': str(checkpoint), 'evaluation_deferred': p['training'].get('defer_evaluation', False),
        'completed_utc': now()})


def idle_devices(selected):
    rows = subprocess.check_output(['nvidia-smi', '--query-gpu=index,uuid', '--format=csv,noheader'], text=True)
    identities = {index.strip(): uuid.strip() for index, uuid in (row.split(',') for row in rows.splitlines())}
    processes = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid', '--format=csv,noheader'], text=True)
    busy = {row.split(',')[0].strip() for row in processes.splitlines()}
    if any(device not in identities for device in selected):
        raise ValueError('Training GPU list contains an unavailable physical index')
    return [device for device in selected if identities.get(device, device) not in busy]


def report_training(root):
    saved = read(root / 'workflow.json')
    config = saved['config']['training']
    tasks = config['evaluation_tasks']
    references = root / 'reference_scores.json'
    if not references.exists():
        values = {}
        if saved.get('result_bundle'):
            import gzip
            with gzip.open(saved['result_bundle'], 'rt') as stream:
                for cell in json.load(stream)['cells']:
                    if cell['task'] in tasks:
                        values.setdefault(cell['method'], {})[cell['task']] = cell['score']
        write(references, values)
    rows = [(f'Frozen {method}', 'reference', values) for method, values in read(references).items()]
    for name, arm in config['arms'].items():
        progress = root / 'arms' / name / 'train/manifest.json'
        step = str(read(progress)['step']) if progress.exists() else 'pending'
        summary = root / 'arms' / name / 'evaluation/results/summary.json'
        values = ({task: item['score'] for task, item in read(summary)['raw_scores'].get(arm['method'], {}).items()}
                  if summary.exists() else {})
        rows.append((name, step, values))
    initialization = config.get('initialization', 'same_old_step320_mlp_warm_start_reset_optimizer')
    lines = ['# K128 anchor adaptation', '', f'Fixed step320; initialization: {initialization}.',
             'Each arm uses its pinned training manifest; optimizer and frozen scorer are shared.', '',
             '| Arm | Step | ' + ' | '.join(tasks) + ' |', '|---|---:|' + '---:|' * len(tasks)]
    for name, step, scores in rows:
        fields = ['—' if scores.get(task) is None else f"{scores[task] * 1000:.0f}" if task == 'ocrbench'
                  else f"{scores[task] * 100:.2f}%" for task in tasks]
        lines.append('| ' + name + ' | ' + step + ' | ' + ' | '.join(fields) + ' |')
    (root / 'report.md').write_text('\n'.join(lines) + '\n')


def runnable_devices(selected, share_min_free_gib=None):
    if share_min_free_gib is None:
        return idle_devices(selected)
    if share_min_free_gib <= 0:
        raise ValueError('Sharing requires a positive free-memory threshold')
    rows = subprocess.check_output(['nvidia-smi', '--query-gpu=index,memory.free',
                                    '--format=csv,noheader,nounits'], text=True)
    free = {index.strip(): float(memory.strip()) for index, memory in
            (row.split(',') for row in rows.splitlines())}
    if any(device not in free for device in selected):
        raise ValueError('Training GPU list contains an unavailable physical index')
    return [device for device in selected if free[device] >= share_min_free_gib * 1024]


class AdoptedWorker:
    """Observe an existing owned worker without restarting its training."""
    def __init__(self, pid, root, name):
        self.pid, self.root, self.name = pid, root, name
        command = Path(f'/proc/{pid}/cmdline').read_bytes().split(b'\0')
        arguments = [item.decode() for item in command if item]
        if ('mllm_comp.workflow.anchor_training' not in arguments or 'worker' not in arguments
                or arguments[arguments.index('--root') + 1] != str(root)
                or arguments[arguments.index('--arm') + 1] != name):
            raise ValueError('Refusing to adopt an unrelated process')
        self.start_time = self._state()[1]
        self.returncode = None

    def _state(self):
        fields = Path(f'/proc/{self.pid}/stat').read_text().rsplit(')', 1)[1].split()
        return fields[0], fields[19]

    def poll(self):
        try:
            state, started = self._state()
            alive = state != 'Z' and started == self.start_time
        except FileNotFoundError:
            alive = False
        if alive:
            return None
        self.returncode = 0 if (self.root / 'arms' / self.name / 'complete.json').exists() else 1
        return self.returncode

    def wait(self, timeout=None):
        deadline = time.monotonic() + timeout if timeout is not None else float('inf')
        while self.poll() is None:
            if time.monotonic() >= deadline:
                raise subprocess.TimeoutExpired(str(self.pid), timeout)
            time.sleep(.1)
        return self.returncode


def supervise(root, wait, selected_gpus=None, share_min_free_gib=None,
              share_gpus=None, adopt_running=False):
    saved = read(root / 'workflow.json')
    config = saved['config']['training']
    selected = selected_gpus or config['gpus']
    shared = set(share_gpus if share_gpus is not None else selected)
    if not shared.issubset(selected):
        raise ValueError('Shared GPUs must belong to the selected GPU list')
    def eligible(devices):
        return [device for device in devices if device in runnable_devices(
            [device], share_min_free_gib if device in shared else None)]
    write(root / 'scheduling.json', {'selected_gpus': selected,
          'share_min_free_gib': share_min_free_gib, 'shared_gpus': sorted(shared), 'updated_utc': now(),
          'training_source_unchanged': True})
    lock = (root / 'supervisor.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    running, streams, leases, assigned, interrupted = {}, {}, {}, {}, []
    signal.signal(signal.SIGTERM, lambda number, frame: interrupted.append(number))
    signal.signal(signal.SIGINT, lambda number, frame: interrupted.append(number))
    try:
        if adopt_running:
            previous = read(root / 'training_status.json')
            for name, item in previous.get('running', {}).items():
                device = item['gpu']
                if device not in selected or name not in config['arms']:
                    raise ValueError('Adopted worker does not belong to the selected schedule')
                process = AdoptedWorker(item['pid'], root, name)
                lock_dir = ROOT / 'outputs/gpu_locks'; lock_dir.mkdir(parents=True, exist_ok=True)
                lease = (lock_dir / f'anchor_train_gpu_{device}.lock').open('a')
                fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
                running[name], leases[name], assigned[name] = process, lease, device
                streams[name] = (root / 'logs' / f'{name}.log').open('a')
        while not interrupted:
            report_training(root)
            for name, process in list(running.items()):
                if process.poll() is not None:
                    streams.pop(name).close(); leases.pop(name).close(); assigned.pop(name); del running[name]
                    if process.returncode:
                        raise RuntimeError(f'{name} failed with {process.returncode}; inspect logs/{name}.log')
            pending = [name for name in config['arms'] if name not in running and not (root / 'arms' / name / 'complete.json').exists()]
            if not pending and not running:
                write(root / 'training_status.json', {'status': 'complete', 'updated_utc': now(), 'arms': list(config['arms'])})
                return 0
            available = [device for device in eligible(selected)
                         if device not in assigned.values()]
            for device in available:
                if not pending:
                    break
                lock_dir = ROOT / 'outputs/gpu_locks'; lock_dir.mkdir(parents=True, exist_ok=True)
                lease = (lock_dir / f'anchor_train_gpu_{device}.lock').open('a')
                try:
                    fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    lease.close(); continue
                if device not in eligible([device]):
                    lease.close(); continue
                name = pending.pop(0)
                stream = (root / 'logs' / f'{name}.log').open('a')
                command = [sys.executable, '-u', '-m', 'mllm_comp.workflow.anchor_training', 'worker', '--root', str(root), '--arm', name]
                environment = {**os.environ, **saved['config']['profile'].get('environment', {}), 'CUDA_VISIBLE_DEVICES': device,
                    'PYTHONPATH': saved['source_root'], 'OMP_NUM_THREADS': '2', 'MKL_NUM_THREADS': '2',
                    'HF_HUB_OFFLINE': '1', 'HF_DATASETS_OFFLINE': '1', 'WANDB_MODE': 'offline'}
                running[name] = subprocess.Popen(command, cwd=saved['source_root'], env=environment,
                    stdin=subprocess.DEVNULL, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
                streams[name], leases[name] = stream, lease
                assigned[name] = device
            write(root / 'training_status.json', {'status': 'running' if running else 'waiting_for_gpu',
                'running': {name: {'pid': process.pid, 'gpu': assigned[name]} for name, process in running.items()},
                'pending': pending, 'updated_utc': now(), 'selected_gpus': selected,
                'share_min_free_gib': share_min_free_gib, 'shared_gpus': sorted(shared)})
            if not running and not wait:
                raise RuntimeError('Selected GPUs are occupied; launch with --wait-for-gpu to queue')
            time.sleep(15)
        write(root / 'training_status.json', {'status': 'interrupted', 'updated_utc': now()})
        return 143
    except BaseException as exc:
        write(root / 'training_status.json', {'status': 'failed', 'error': str(exc), 'traceback': traceback.format_exc(), 'updated_utc': now()})
        raise
    finally:
        for process in running.values():
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
        for process in running.values():
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL); process.wait()
        for stream in streams.values(): stream.close()
        for lease in leases.values(): lease.close()
        lock.close()


def scheduling_arguments(profile):
    if not profile.get('training_schedule'):
        return []
    schedule = read(profile['training_schedule'])
    devices, shared = schedule['gpus'], schedule.get('share_gpus', [])
    threshold = schedule.get('share_min_free_gib')
    if not devices or not set(shared).issubset(devices) or (shared and (threshold is None or threshold <= 0)):
        raise ValueError('Invalid training GPU schedule')
    result = ['--gpus', *devices]
    if shared:
        result += ['--share-min-free-gib', str(threshold), '--share-gpus', *shared]
    return result


def start(args):
    config = plan(args.profile, args.training_config)
    if args.run_id:
        if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*', args.run_id):
            raise ValueError('Invalid training run_id')
        config['run_id'] = args.run_id
    if args.action == 'train-plan':
        print(json.dumps(config, ensure_ascii=False, indent=2)); return 0
    default_output_root = Path(config['profile'].get('training_output_root', ROOT / 'outputs'))
    output = (args.run_dir or default_output_root / 'train' / config['run_id']).resolve()
    allowed_output_root = Path(config['profile'].get('training_output_root', ROOT / 'outputs')).resolve()
    if not output.is_relative_to(allowed_output_root):
        raise ValueError(f'Training output must stay under the configured output root: {allowed_output_root}')
    saved = freeze(config, output, resume=args.resume)
    target = output / 'training_samples.json'
    if not target.exists():
        import shutil
        shutil.copy2(config['profile']['training_manifest'], target)
    import shutil
    for name, arm in config['training']['arms'].items():
        if arm.get('training_manifest'):
            target = output / f'training_samples_{name}.json'
            if not target.exists():
                shutil.copy2(arm['training_manifest'], target)
    command = [saved['python'], '-u', '-m', 'mllm_comp.workflow.anchor_training', 'supervise', '--root', str(output)]
    command += scheduling_arguments(config['profile'])
    if args.wait_for_gpu:
        command.append('--wait-for-gpu')
    environment = {**os.environ, 'PYTHONPATH': saved['source_root'], 'PYTHONUNBUFFERED': '1'}
    if args.launch:
        with (output / 'supervisor.log').open('a') as stream:
            process = subprocess.Popen(command, cwd=saved['source_root'], env=environment,
                stdin=subprocess.DEVNULL, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
        write(output / 'supervisor.json', {'pid': process.pid, 'command': command, 'started_utc': now()})
        print(json.dumps({'pid': process.pid, 'run_dir': str(output), 'log': str(output / 'supervisor.log')}))
        return 0
    return subprocess.call(command, cwd=saved['source_root'], env=environment)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('action', choices=['supervise', 'worker'])
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--arm')
    parser.add_argument('--wait-for-gpu', action='store_true')
    parser.add_argument('--gpus', nargs='+', help='Runtime scheduling override; does not change the training protocol')
    parser.add_argument('--share-min-free-gib', type=float,
                        help='Explicitly allow occupied GPUs only above this free-memory threshold')
    parser.add_argument('--share-gpus', nargs='+', help='Only these GPUs may use the explicit sharing threshold')
    parser.add_argument('--adopt-running', action='store_true', help='Keep verified existing workers alive during scheduler replacement')
    args = parser.parse_args()
    if args.action == 'worker':
        worker(args.root, args.arm)
    else:
        raise SystemExit(supervise(args.root, args.wait_for_gpu, args.gpus, args.share_min_free_gib,
                                  args.share_gpus, args.adopt_running))
