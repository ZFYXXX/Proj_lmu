"""Version/data/resource gates and restartable state for learned visual Top-K.

No CUDA context is created by resource inspection. Outputs stay in the owner's
tree; model/dataset inputs may be read-only shared paths.
"""
from collections import Counter
import hashlib
import json
import math
import os
from pathlib import Path
import random
import subprocess

OWNER = Path('/home/ZhouFengyu').resolve()


def digest_json(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def owned_output(path):
    path = Path(path).resolve()
    if not path.is_relative_to(OWNER) or path == OWNER:
        raise ValueError(f'Output must be below {OWNER}: {path}')
    return path


def source_identity(root):
    root = Path(root).resolve()
    def git(*args):
        return subprocess.check_output(['git', '-C', str(root), *args], text=True).strip()
    if git('status', '--porcelain', '--untracked-files=normal'):
        raise ValueError('Freeze a clean source commit before preparing/running an experiment')
    import mllm_comp
    actual = Path(mllm_comp.__file__).resolve()
    if not actual.is_relative_to(root):
        raise ValueError(f'Imported source outside the execution snapshot: {actual}')
    return {'commit': git('rev-parse', 'HEAD'), 'source_root': str(root),
            'module_path': str(actual), 'tree': git('rev-parse', 'HEAD^{tree}')}


def dependency_identity():
    """Fingerprint the editable evaluator too: it is outside the worktree."""
    from importlib.metadata import version
    from importlib.util import find_spec
    package = Path(find_spec('lmms_eval').origin).resolve().parent
    digest = hashlib.sha256()
    count = 0
    for path in sorted(package.rglob('*')):
        if path.is_file() and path.suffix in ('.py', '.yaml', '.yml'):
            digest.update(str(path.relative_to(package)).encode())
            digest.update(hashlib.sha256(path.read_bytes()).digest())
            count += 1
    return {'versions': {name: version(name) for name in ('torch', 'transformers', 'safetensors', 'numpy', 'lmms_eval')},
            'lmms_eval_root': str(package), 'lmms_eval_source_sha256': digest.hexdigest(), 'lmms_eval_files': count}


def gpu_processes():
    """Fail closed on query errors. Resolve UUIDs, not visible-device numbering."""
    gpu_lines = subprocess.check_output(['nvidia-smi', '--query-gpu=index,uuid', '--format=csv,noheader'], text=True)
    mapping = {parts[1].strip(): int(parts[0]) for line in gpu_lines.splitlines() if (parts := line.split(','))}
    lines = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid', '--format=csv,noheader'], text=True)
    result = []
    for line in lines.splitlines():
        uuid, pid = [s.strip() for s in line.split(',')]
        try:
            uid = Path(f'/proc/{int(pid)}').stat().st_uid
        except FileNotFoundError:
            continue  # Process exited between the driver query and /proc lookup.
        result.append({'gpu': mapping[uuid], 'pid': int(pid), 'uid': uid})
    return result


def available_gpus(requested, *, processes=None, uid=None, allow_own=False):
    uid = os.getuid() if uid is None else uid
    processes = gpu_processes() if processes is None else processes
    return [i for i in requested if not any(p['gpu'] == i and
        (not allow_own or p['uid'] != uid) for p in processes)]


def validate_training_entries(entries):
    if len(entries) != 2560 or Counter(e['source'] for e in entries) != {
            'textvqa_train': 2048, 'scienceqa_train': 512}:
        raise ValueError('Training cohort must remain TextVQA 2048 + ScienceQA 512')
    if len({e['sample_id'] for e in entries}) != len(entries):
        raise ValueError('Duplicate training sample IDs')
    if len({e['image_fingerprint'] for e in entries}) != len(entries):
        raise ValueError('Duplicate independent training images')


def reject_overlap(train_entries, eval_rows):
    overlap = {e['image_fingerprint'] for e in train_entries} & {e['image_fingerprint'] for e in eval_rows}
    if overlap:
        raise ValueError(f'Train/eval image overlap ({len(overlap)}): {sorted(overlap)}; do not silently resample')


def development_indices(rows, count=250, seed=42):
    # First source row per image, chosen without reading target or predictions.
    by_image = {}
    for row in rows:
        key = row['image_fingerprint']
        by_image[key] = min(by_image.get(key, row['row']), row['row'])
    if len(by_image) < count:
        raise ValueError('Not enough independent development images')
    keys = sorted(by_image, key=lambda x: hashlib.sha256(f'{seed}:{x}'.encode()).hexdigest())[:count]
    return [by_image[k] for k in keys]


def batch_indices(step, *, world_size=1, rank=0):
    if world_size not in (1, 2) or not 0 <= rank < world_size or not 0 <= step < 320:
        raise ValueError('Supported: 320 steps, global batch 8, one or two ranks')
    return list(range(step * 8 + rank, (step + 1) * 8, world_size))


def lr_multiplier(step):
    if step < 16:
        return (step + 1) / 16
    return .5 * (1 + math.cos(math.pi * min(1., (step - 16) / (320 - 16))))


def bce_weight(step):
    if not 0 <= step < 320:
        raise ValueError('BCE annealing step outside the frozen 320-step schedule')
    return .1 + 2.9 * step / 319


def training_loss(student, teacher, targets, selection, step):
    import torch
    import torch.nn.functional as F
    if student.shape != teacher.shape or targets.shape != student.shape[:-1]:
        raise ValueError('Teacher/student answer alignment mismatch')
    ce = F.cross_entropy(student.float(), targets)
    kl = F.kl_div(F.log_softmax(student.float(), -1), F.softmax(teacher.detach().float(), -1),
                  reduction='batchmean').clamp_min(0)
    bce = F.binary_cross_entropy(selection['soft_weights'].float(), selection['hard_mask'].detach().float())
    loss = ce + .5 * kl + bce_weight(step) * bce
    if not torch.isfinite(loss):
        raise ValueError('Nonfinite selector training loss')
    return loss, {'loss': float(loss.detach()), 'answer_ce': float(ce.detach()),
                  'teacher_kl': float(kl.detach()), 'mask_bce': float(bce.detach()), 'bce_weight': bce_weight(step)}


def rng_state(device=None):
    import numpy as np
    import torch
    return {'python': random.getstate(), 'numpy': np.random.get_state(), 'torch': torch.get_rng_state(),
            'cuda': torch.cuda.get_rng_state(device) if device is not None else None}


def restore_rng(state, device=None):
    import numpy as np
    import torch
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'].cpu())
    if device is not None:
        torch.cuda.set_rng_state(state['cuda'].cpu(), device)


def save_training_state(folder, *, selector, optimizer, scheduler, step, contract, random_states):
    import torch
    from mllm_comp.compressors.image.learned_visual_topk import save_selector
    from mllm_comp.core.serialization import atomic_write_json
    folder = owned_output(folder)
    save_selector(selector, folder, base_identity=contract['base_identity_sha256'], provenance=contract)
    torch.save({'contract_hash': digest_json(contract), 'step': step, 'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(), 'random_states': random_states}, folder / 'training_state.pt')
    atomic_write_json(folder / 'complete.json', {'step': step, 'contract_hash': digest_json(contract),
        'training_state_sha256': hashlib.sha256((folder / 'training_state.pt').read_bytes()).hexdigest()})


def restore_training_state(folder, *, selector, optimizer, scheduler, contract, rank=0, device=None):
    import torch
    from mllm_comp.compressors.image.learned_visual_topk import load_selector
    folder = Path(folder)
    if not (folder / 'complete.json').exists():
        raise ValueError('Incomplete checkpoint: no atomic completion marker')
    complete = json.loads((folder / 'complete.json').read_text())
    if complete['training_state_sha256'] != hashlib.sha256((folder / 'training_state.pt').read_bytes()).hexdigest():
        raise ValueError('Training state integrity check failed')
    state = torch.load(folder / 'training_state.pt', map_location='cpu', weights_only=False)
    if state['contract_hash'] != digest_json(contract):
        raise ValueError('Resume rejected: code/config/data/base/world-size contract changed')
    saved = load_selector(folder, base_identity=contract['base_identity_sha256'], device='cpu')
    selector.load_state_dict(saved.state_dict(), strict=True)
    optimizer.load_state_dict(state['optimizer'])
    scheduler.load_state_dict(state['scheduler'])
    restore_rng(state['random_states'][rank], device)
    return state['step']
