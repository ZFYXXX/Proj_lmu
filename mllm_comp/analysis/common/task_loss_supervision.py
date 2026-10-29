"""Budget-matched candidate-set teacher; labels are proxies, not causal GT."""
import hashlib
import json
import math
from pathlib import Path

import torch
import torch.nn.functional as F

from .anchor7b_runtime import file_sha


def candidate_sets(scores, height, width, *, k=128, count=8, seed=42):
    if scores.ndim != 1 or len(scores) != height*width or not torch.isfinite(scores).all():
        raise ValueError('Invalid native grid/scores')
    n = len(scores)
    k = min(k, n)
    ranking = torch.argsort(scores.detach().cpu(), descending=True, stable=True).tolist()
    base = sorted(ranking[:k])
    sets = [base]
    if n == k:
        return sets
    generator = torch.Generator().manual_seed(seed)
    tiles = torch.randperm(16, generator=generator).tolist()
    for attempt in range(max(128, 8*count)):
        tile = tiles[attempt % 16]
        row, col = divmod(tile, 4)
        region = {r*width+c for r in range(row*height//4, (row+1)*height//4)
                  for c in range(col*width//4, (col+1)*width//4)}
        forced = [i for i in ranking if i in region][:min(32, k)]
        if attempt >= 16 or not forced:
            forced = torch.randperm(n, generator=generator)[:min(32, k)].tolist()
        order = list(dict.fromkeys(forced + ranking))
        selected = sorted(order[:k])
        if selected not in sets:
            sets.append(selected)
        if len(sets) == count:
            return sets
    raise ValueError('Cannot create requested distinct teacher candidate sets')


def retention_target(candidates, losses, source_count, *, temperature=.25):
    if len(candidates) != len(losses) or not candidates or temperature <= 0:
        raise ValueError('Invalid teacher candidates/losses')
    if any(not math.isfinite(x) for x in losses):
        raise ValueError('Nonfinite teacher loss')
    k = len(candidates[0])
    masks = torch.zeros(len(candidates), source_count)
    for row, members in enumerate(candidates):
        if len(members) != k or members != sorted(set(members)) or not members or members[0] < 0 or members[-1] >= source_count:
            raise ValueError('Teacher set budget/member mismatch')
        masks[row, members] = 1
    weights = (-torch.tensor(losses, dtype=torch.float32)/temperature).softmax(0)
    return weights @ masks


def teacher_ranking_loss(scores, target):
    target = target.to(device=scores.device, dtype=torch.float32)
    if target.shape != scores.shape or not torch.isfinite(target).all() or (target < 0).any() or target.sum() <= 0:
        raise ValueError('Invalid retention target')
    return F.kl_div(scores.float().log_softmax(-1), target/target.sum(), reduction='sum')


def label_key(sample_id):
    return hashlib.sha256(sample_id.encode()).hexdigest()


def load_target(index, example, state, audit, *, device):
    item = index['rows'][example['sample_id']]
    path = Path(item['path'])
    if file_sha(path) != item['sha256']:
        raise ValueError('Teacher label artifact changed')
    row = json.loads(path.read_text())
    if (row['sample_id'] != example['sample_id'] or row['image_fingerprint'] != example['image_fingerprint']
            or row['question_ids_sha256'] != audit['question_ids_sha256']
            or row['source_count'] != len(state['visual'])
            or row['height'] != state['height'] or row['width'] != state['width']):
        raise ValueError('Teacher label/image/question/grid alignment changed')
    return retention_target(row['candidates'], row['losses'], row['source_count'],
                            temperature=index['temperature']).to(device)
