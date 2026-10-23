"""Measurements, without token selection or mutation in the model forward."""
from __future__ import annotations

import math
import numpy as np
import torch
from torch.nn import functional as F


def jsonable(value):
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return jsonable(value.tolist())
    if isinstance(value, np.generic):
        return jsonable(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def adjacent_cosines(x, boundaries, lag=1):
    """All overlapping pairs; cross-chunk pairs are explicitly missing."""
    flat = x.detach().float().reshape(x.shape[0], -1)
    values = torch.full((len(flat),), float('nan'), device=x.device)
    for start, end in zip(boundaries, boundaries[1:]):
        if end - start <= lag:
            continue
        a, b = flat[start:end-lag], flat[start+lag:end]
        cos = F.cosine_similarity(a, b, dim=-1)
        cos[(a.norm(dim=-1) == 0) | (b.norm(dim=-1) == 0)] = float('nan')
        values[start:end-lag] = cos
    return values.cpu().numpy()


def finite_mean(x):
    x = np.asarray(x, dtype=float)
    valid = np.isfinite(x)
    return float(x[valid].mean()) if valid.any() else None


def similarity_summary(x, boundaries):
    values = adjacent_cosines(x, boundaries)
    # Reuse lag=1 instead of recomputing it and transferring it from the GPU.
    lags = {'1': finite_mean(values)}
    lags.update({str(d): finite_mean(adjacent_cosines(x, boundaries, d))
                 for d in (2, 4, 8, 16, 32)})
    # The platform A-ToMe implementation ranks disjoint pairs from each chunk start.
    disjoint = np.concatenate([values[a:b-1:2] for a, b in zip(boundaries, boundaries[1:])])
    return values, {'adjacent_mean': finite_mean(values),
                    'disjoint_mean': finite_mean(disjoint), 'lag_means': lags}


@torch.no_grad()
def attention_summary(q, k, v, native_head_output, boundaries, scaling,
                      query_block=64, check_av=False, save_chunk=None, time_step_ms=None):
    """Full bidirectional packed chunks, including self; no dense global matrix.

    Q/K/V are the actual projections [token, head, dim]. The shadow QK matmul
    uses their dtype, and softmax accumulates in FP32. FA2 need not be bit exact.
    """
    if query_block < 1:
        raise ValueError('query_block must be positive')
    head_count = q.shape[1]
    received = torch.empty((head_count, len(q)), device=q.device, dtype=torch.float32)
    chunk_stats = []
    peak_hist = torch.zeros((head_count, 10), device=q.device)
    uniform_hist = torch.zeros(10, device=q.device)
    error_sq = native_sq = max_error = 0.0
    saved_attention = None
    distance_edges = np.array([0, 20, 40, 80, 160, 320, 640, 1280, 2560, np.inf])
    distance_mass = torch.zeros((head_count, len(distance_edges)-1), device=q.device)
    distance_reference = torch.zeros_like(distance_mass)
    distance_sum = torch.zeros(head_count, device=q.device)
    for chunk_id, (a, b) in enumerate(zip(boundaries, boundaries[1:])):
        n = b - a
        qc, kc, vc = [t[a:b].transpose(0, 1) for t in (q, k, v)]
        incoming = torch.zeros((head_count, n), device=q.device)
        above_uniform = torch.zeros_like(incoming)
        query_entropy = torch.zeros(head_count, device=q.device)
        saved = []
        for i in range(0, n, query_block):
            logits = (qc[:, i:i+query_block] @ kc.transpose(-2, -1)) * scaling
            p = torch.softmax(logits, dim=-1, dtype=torch.float32)
            incoming += p.sum(dim=-2)
            above_uniform += (p > 1.0/n + 1e-7).sum(dim=-2)
            query_entropy += -(p * p.clamp_min(1e-30).log()).sum(dim=(-1, -2))
            if time_step_ms is not None:
                distance = (torch.arange(i, min(i+query_block, n), device=q.device)[:, None]
                            - torch.arange(n, device=q.device)[None, :]).abs() * time_step_ms
                bins = torch.bucketize(distance.contiguous(),
                    torch.as_tensor(distance_edges[1:-1], device=q.device, dtype=distance.dtype), right=True)
                distance_mass.scatter_add_(1, bins.flatten().expand(head_count, -1), p.flatten(1))
                distance_reference.scatter_add_(1, bins.flatten().expand(head_count, -1),
                                                 torch.full_like(p, 1.0/n).flatten(1))
                distance_sum += (p * distance).sum((-1, -2))
            if check_av:
                shadow = (p.to(vc.dtype) @ vc).float()
                native = native_head_output[a+i:min(a+i+query_block, b)].transpose(0, 1).float()
                delta = shadow - native
                error_sq += float(delta.square().sum())
                native_sq += float(native.square().sum())
                max_error = max(max_error, float(delta.abs().max()))
            if chunk_id == save_chunk:
                saved.append(p.cpu().numpy())
        if not torch.isfinite(incoming).all():
            raise ValueError('Non-finite reconstructed attention')
        if not torch.allclose(incoming.sum(-1), torch.full_like(incoming[:, 0], n), atol=2e-3, rtol=2e-4):
            raise ValueError('Incoming attention mass does not sum to the visible chunk length')
        received[:, a:b] = incoming
        prob = incoming / n
        peaks = incoming == incoming.amax(-1, keepdim=True)
        peak_weights = peaks.float() / peaks.sum(-1, keepdim=True)
        # Split exact ties equally; uniform attention must not create a start preference.
        bin_ids = (torch.arange(n, device=q.device) * 10 // n).clamp_max(9)
        peak_hist.scatter_add_(1, bin_ids.expand(head_count, -1), peak_weights)
        uniform_hist.scatter_add_(0, bin_ids, torch.full((n,), 1.0/n, device=q.device))
        ent = -(prob * prob.clamp_min(1e-30).log()).sum(-1)
        if n > 1:
            ent, query_entropy = ent / math.log(n), query_entropy / (n * math.log(n))
        else:
            ent[:] = float('nan')
            query_entropy[:] = float('nan')
        count = max(1, math.ceil(n * 0.1))
        chunk_stats.append({
            'peak_r': incoming.amax(-1).cpu().numpy(),
            'start_r': incoming[:, 0].cpu().numpy(),
            'top10pct_mass': prob.topk(count, dim=-1).values.sum(-1).cpu().numpy(),
            'incoming_entropy': ent.cpu().numpy(),
            'query_entropy': query_entropy.cpu().numpy(),
            'peak_query_above_uniform': ((above_uniform / n) * peak_weights).sum(-1).cpu().numpy(),
            'peak_start_weight': peak_weights[:, 0].cpu().numpy(),
            'uniform_peak_start_weight': np.full(head_count, 1.0/n),
            'uniform_top10pct_mass': np.full(head_count, count/n),
        })
        if saved:
            saved_attention = np.concatenate(saved, axis=1)
    heads = {}
    for name in chunk_stats[0]:
        values = np.stack([c[name] for c in chunk_stats])
        finite = np.isfinite(values)
        heads[name] = np.divide(np.where(finite, values, 0).sum(0), finite.sum(0),
                                out=np.full(head_count, np.nan), where=finite.sum(0) > 0)
    heads['peak_position_hist'] = (peak_hist / len(chunk_stats)).cpu().numpy()
    heads['uniform_peak_position_hist'] = (uniform_hist / len(chunk_stats)).cpu().numpy()
    # Keep the short-tail measurements separate from the equal-chunk means.
    heads['last_chunk_peak_r'] = chunk_stats[-1]['peak_r']
    heads['first_chunk_start_r'] = chunk_stats[0]['start_r']
    heads['later_chunks_start_r'] = (np.mean([c['start_r'] for c in chunk_stats[1:]], axis=0)
                                     if len(chunk_stats) > 1 else np.full(head_count, np.nan))
    if time_step_ms is not None:
        heads['distance_ms_edges'] = distance_edges
        heads['distance_mass'] = (distance_mass / len(q)).cpu().numpy()
        heads['distance_uniform_mass'] = (distance_reference / len(q)).cpu().numpy()
        heads['mean_distance_ms'] = (distance_sum / len(q)).cpu().numpy()
    checks = None if not check_av else {
        'av_relative_l2': math.sqrt(error_sq / max(native_sq, 1e-30)),
        'av_max_abs_error': max_error,
    }
    return received.cpu().numpy(), heads, checks, saved_attention


def virtual_selection(scores, keep_ratio=0.1, segments=10, seed=42, random_repeats=100):
    """Diagnostic indices only, using the project's exact segment budget policy."""
    from mllm_comp.compressors.projector_level.segmentwise_topk import (
        segmentwise_topk_indices, segment_boundaries,
    )
    scores = np.asarray(scores, dtype=float)
    if not len(scores) or not np.isfinite(scores).all():
        raise ValueError('Virtual selection requires finite nonempty scores')
    if not 0 < keep_ratio <= 1 or segments < 1:
        raise ValueError('Invalid virtual budget or segment count')
    k = max(1, int(len(scores) * keep_ratio))
    top = np.sort(np.argsort(-scores, kind='stable')[:k])
    segment = np.asarray(segmentwise_topk_indices(scores.tolist(), keep_ratio=keep_ratio,
                        num_segments=segments, allocation='balanced_exact'), dtype=int)
    bins = segment_boundaries(len(scores), segments)

    def coverage(indices):
        counts = np.asarray([np.sum((indices >= a) & (indices < b)) for a, b in bins])
        gaps = np.diff(np.r_[-1, np.sort(indices), len(scores)]) - 1
        return {'segment_coverage': float(np.mean(counts > 0)),
                'largest_segment_share': float(counts.max() / k),
                'longest_unselected_fraction': float(gaps.max() / len(scores)),
                'adjacent_selected_fraction': float(np.mean(np.diff(indices) == 1)) if k > 1 else None}

    rng = np.random.default_rng(seed)
    controls = [coverage(np.sort(rng.choice(len(scores), k, replace=False))) for _ in range(random_repeats)]
    random_summary = {name: finite_mean([c[name] if c[name] is not None else np.nan for c in controls])
                      for name in controls[0]}
    cutoff = scores[top].min()
    return {'budget': k, 'token_count': len(scores), 'segments': len(bins),
            'cutoff_tie_count': int(np.sum(scores == cutoff)),
            'cutoff_slots': int(k - np.sum(scores > cutoff)),
            'topk': coverage(top), 'segmentwise': coverage(segment), 'random': random_summary,
            'random_repeats': random_repeats}, top, segment
