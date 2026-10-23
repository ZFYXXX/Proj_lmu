"""Request-local attention interventions; no checkpoint or framework edits."""
from __future__ import annotations

from contextlib import contextmanager

import numpy as np
import torch


def conditions(layers, heads, *, families=('range', 'head', 'sink', 'evidence'),
               windows_ms=(80, 320, 1280), sink_heads=None, fractions=(.05, .1, .2),
               block_ms=80, seed=42):
    """Freeze every intervention before reading predictions; scan all heads."""
    sink_heads = range(heads) if sink_heads is None else sink_heads
    result = []
    def add(family, name, **kwargs):
        result.append(dict(id=name, family=family, **kwargs))
    for layer in layers:
        if 'range' in families:
            for window in windows_ms:
                for selector in ('window', 'random'):
                    for placement in ('pre', 'post'):
                        add('range', f'range_l{layer}_{selector}_{window:g}ms_{placement}',
                            layer=layer, window_ms=window, selector=selector, placement=placement)
        if 'head' in families:
            for head in range(heads):
                controls = np.random.default_rng(seed + layer * heads + head).choice(
                    [h for h in range(heads) if h != head], min(3, heads-1), replace=False).tolist()
                for operation in ('zero', 'restore'):
                    add('head', f'head_l{layer}_h{head}_{operation}', layer=layer, head=head,
                        operation=operation, random_control_heads=controls)
        if 'sink' in families:
            for head in sink_heads:
                for selector in ('high', 'low', 'random'):
                    for operation in ('v_zero', 'v_norm', 'key_mask', 'restore_v', 'restore_head'):
                        add('sink', f'sink_l{layer}_h{head}_{selector}_{operation}', layer=layer,
                            head=head, selector=selector, operation=operation, fraction=.1, block_ms=block_ms)
    if 'sink' in families:
        for operation in ('mel_roll', 'mel_mean'):
            add('input', operation, operation=operation, labels_preserved=False)
    if 'evidence' in families:
        for fraction in fractions:
            for selector in ('low', 'high', 'random'):
                for operation in ('mean', 'norm', 'delete', 'restore'):
                    add('evidence', f'evidence_{selector}_{fraction:g}_{operation}',
                        selector=selector, fraction=fraction, operation=operation, block_ms=block_ms)
        for keep in (.1, .25, .5):
            for operation in ('topk', 'refill_low', 'refill_random'):
                add('refill', f'refill_{keep:g}_{operation}', keep_fraction=keep,
                    refill_fraction=.25, operation=operation)
    if len({c['id'] for c in result}) != len(result):
        raise ValueError('Duplicate condition IDs')
    return result


def choose_blocks(scores, boundaries, width, fraction, selector, seed):
    """Same-duration blocks, same audio-specific budget; retain a key per chunk."""
    blocks, capacities = [], []
    for chunk, (a, b) in enumerate(zip(boundaries, boundaries[1:])):
        starts = list(range(a, b-width+1, width))
        capacities.append(min(len(starts), (b-a-1)//width))
        blocks.extend((s, s+width, chunk) for s in starts)
    budget = min(sum(capacities), max(1, int(len(blocks) * fraction))) if blocks else 0
    means = np.array([np.mean(scores[a:b]) for a, b, _ in blocks])
    rng = np.random.default_rng(seed)
    if selector == 'random':
        order = rng.permutation(len(blocks))
    else:
        order = np.argsort(-means if selector == 'high' else means, kind='stable')
    used = np.zeros(len(capacities), dtype=int)
    selected = []
    for index in order:
        if len(selected) == budget:
            break
        a, b, chunk = blocks[index]
        if used[chunk] < capacities[chunk]:
            selected.append(int(index))
            used[chunk] += 1
    indices = sorted(i for j in selected for i in range(blocks[j][0], blocks[j][1]))
    cutoff = float(means[selected[-1]]) if selected and selector != 'random' else None
    return np.asarray(indices, dtype=int), {
        'selected_blocks': [[blocks[j][0], blocks[j][1]] for j in selected],
        'block_tokens': width, 'eligible_blocks': len(blocks), 'selected_block_count': budget,
        'selected_tokens': len(indices), 'active': bool(indices),
        'block_score_min': float(means.min()) if len(means) else None,
        'block_score_max': float(means.max()) if len(means) else None,
        'all_block_scores_tied': bool(len(means) and np.all(means == means[0])),
        'cutoff_score': cutoff,
        'cutoff_tied_blocks': int(np.sum(means == cutoff)) if cutoff is not None else 0,
    }


def range_mask(length, radius, selector, seed):
    pos = np.arange(length)
    local = abs(pos[:, None] - pos[None, :]) <= radius
    if selector == 'window':
        return local
    mask = np.eye(length, dtype=bool)
    rng = np.random.default_rng(seed)
    for query in range(length):
        candidates = pos[pos != query]
        mask[query, rng.choice(candidates, int(local[query].sum())-1, replace=False)] = True
    return mask


def shadow_probabilities(q, k, scaling):
    return torch.softmax((q @ k.transpose(-2, -1)) * scaling, dim=-1, dtype=torch.float32)


def norm_matched(values, indices):
    """Replace directions by an unselected mean, preserving each selected norm."""
    keep = torch.ones(len(values), dtype=torch.bool, device=values.device)
    keep[indices] = False
    donor = values[keep].float().mean(0)
    norm = donor.norm()
    if norm == 0:
        donor = torch.zeros_like(donor)
        donor[0] = 1
        norm = donor.norm()
    return (values[indices].float().norm(dim=-1, keepdim=True) * donor / norm).to(values.dtype)


def intervention_head_output(q, k, v, native, boundaries, scaling, condition, selected, step_ms, seed):
    """Apply a shadow-A delta to native head outputs; unchanged heads remain exact."""
    changed = native.clone()
    target_heads = [condition['head']] if condition['family'] == 'sink' else list(range(q.shape[1]))
    selected_set = set(selected)
    for chunk, (a, b) in enumerate(zip(boundaries, boundaries[1:])):
        local_selected = [j-a for j in selected_set if a <= j < b]
        for head in target_heads:
            qc, kc, vc = q[a:b, head], k[a:b, head], v[a:b, head]
            p = shadow_probabilities(qc, kc, scaling)
            base = p.to(vc.dtype) @ vc
            if condition['family'] == 'range':
                allowed = torch.as_tensor(range_mask(b-a, int(condition['window_ms']//step_ms),
                    condition['selector'], seed+chunk), device=q.device)
                if condition['placement'] == 'pre':
                    logits = (qc @ kc.T) * scaling
                    weights = torch.softmax(logits.masked_fill(~allowed, -torch.inf), -1, dtype=torch.float32)
                else:
                    weights = p * allowed
                altered = weights.to(vc.dtype) @ vc
            else:
                if not local_selected:
                    continue
                operation = condition['operation']
                if operation in ('v_zero', 'v_norm', 'restore_v'):
                    replacement = vc.clone()
                    replacement[local_selected] = (norm_matched(vc, local_selected)
                        if operation == 'v_norm' else 0)
                    if operation == 'restore_v':
                        replacement[local_selected] = vc[local_selected]
                    altered = p.to(vc.dtype) @ replacement
                else:
                    logits = (qc @ kc.T) * scaling
                    logits[:, local_selected] = -torch.inf
                    altered = torch.softmax(logits, -1, dtype=torch.float32).to(vc.dtype) @ vc
            changed[a:b, head] = (native[a:b, head].float() + (altered.float() - base.float())).to(native.dtype)
    if condition.get('operation') == 'restore_head':
        changed.copy_(native)
    return changed


def refill_indices(scores, keep_fraction, operation, refill_fraction, seed):
    n = len(scores)
    budget = min(n, max(1, int(n * keep_fraction)))
    order = np.argsort(-scores, kind='stable')
    top, outside = order[:budget], order[budget:]
    count = min(len(outside), max(1, int(budget * refill_fraction)))
    if operation == 'topk' or count == 0:
        keep = top
    else:
        refill = (outside[::-1][:count] if operation == 'refill_low'
                  else np.random.default_rng(seed).choice(outside, count, replace=False))
        keep = np.r_[top[:budget-count], refill]
    return np.sort(keep), {'kept_tokens': budget, 'refilled_tokens': count if operation != 'topk' else 0}


def correlation(x, y):
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    if len(x) < 2:
        return None
    x, y = x-x.mean(), y-y.mean()
    denom = np.linalg.norm(x) * np.linalg.norm(y)
    return float(x @ y / denom) if denom > 1e-12 else None


class AttentionInterventions:
    """One native reference followed by fixed conditions for the same audio."""
    def __init__(self, tower, *, step_ms, observer):
        self.tower, self.step_ms, self.observer = tower, step_ms, observer
        self.condition = None
        self.handles, self.cache = [], {}
        self.reference_scores = {}
        self.reference_hidden = None
        self.reference_mel = None

    def install(self):
        self.handles.append(self.tower.register_forward_pre_hook(self._tower_pre, with_kwargs=True))
        self.handles.append(self.tower.proj.register_forward_pre_hook(self._projector_pre))
        for layer in self.observer.layers:
            attn = self.tower.layers[layer].self_attn
            self.handles.append(attn.register_forward_pre_hook(self._attn_pre(layer), with_kwargs=True))
            for name in ('q', 'k', 'v'):
                self.handles.append(getattr(attn, name+'_proj').register_forward_hook(self._qkv(layer, name)))
            self.handles.append(attn.out_proj.register_forward_pre_hook(self._head_output(layer)))

    def close(self):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()
        self.cache.clear()
        self.reference_hidden = self.reference_mel = None

    def begin(self, condition=None, *, seed=42):
        self.condition, self.seed = condition, seed
        self.details, self.cache, self.calls = {}, {}, 0
        self.selected = np.array([], dtype=int)
        self.keep_indices = None
        self.last_hidden = None
        if condition is None:
            self.reference_scores = {}
            self.reference_hidden = self.reference_mel = None
        elif condition['family'] == 'sink':
            scores = self.reference_scores[condition['layer']][condition['head']]
            width = max(1, round(condition['block_ms']/self.step_ms))
            self.selected, self.details = choose_blocks(scores, self.bounds, width,
                condition['fraction'], condition['selector'], seed)
            self.details['selected_mean_r'] = float(scores[self.selected].mean()) if len(self.selected) else None
        elif condition['family'] in ('evidence', 'refill'):
            last = self.reference_scores[len(self.tower.layers)-1].mean(0)
            scores = last[:len(last)//2*2].reshape(-1, 2).mean(1)
            if condition['family'] == 'refill':
                self.keep_indices, self.details = refill_indices(scores, condition['keep_fraction'],
                    condition['operation'], condition['refill_fraction'], seed)
            else:
                self.selected, self.details = choose_blocks(scores, [0, len(scores)],
                    max(1, round(condition['block_ms']/(2*self.step_ms))),
                    condition['fraction'], condition['selector'], seed)
                if condition['operation'] == 'delete':
                    self.keep_indices = np.setdiff1d(np.arange(len(scores)), self.selected)
        self.details['intervention_space'] = ('pre_projector_after_ln_post'
            if condition and condition['family'] in ('evidence', 'refill') else 'encoder')

    def finish(self):
        if self.calls != 1 or self.cache or self.last_hidden is None:
            raise RuntimeError('Incomplete or repeated audio intervention forward')
        if self.condition is None:
            self.reference_scores = dict(self.observer.scores)
        if self.condition and self.condition['family'] == 'input':
            following = {}
            valid = self.mapping_valid
            for layer, scores in self.observer.scores.items():
                original = self.reference_scores[layer]
                following[str(layer)] = {
                    'position_correlation': [correlation(a[valid], b[valid]) for a, b in zip(original, scores)],
                    'content_correlation': ([correlation(a[self.mapping][valid], b[valid])
                        for a, b in zip(original, scores)] if self.condition['operation'] == 'mel_roll' else None),
                    'peak_positions_before': [int(a.argmax()) if np.sum(a == a.max()) == 1 else None for a in original],
                    'peak_positions_after': [int(a.argmax()) if np.sum(a == a.max()) == 1 else None for a in scores],
                    'peak_tie_counts_before': (original == original.max(-1, keepdims=True)).sum(-1).tolist(),
                    'peak_tie_counts_after': (scores == scores.max(-1, keepdims=True)).sum(-1).tolist(),
                }
            self.details['following'] = following
            self.details['following_valid_tokens'] = int(valid.sum())
            self.details['labels_preserved'] = False
        return self.details

    def _tower_pre(self, module, args, kwargs):
        self.calls += 1
        features = kwargs.get('input_features', args[0] if args else None)
        lengths = kwargs.get('feature_lens', args[1] if len(args)>1 else None)
        if lengths.numel() != 1:
            raise ValueError('Suite requires one audio per request')
        length = int(lengths[0])
        mel_lengths = [min(2*module.n_window, length-a) for a in range(0, length, 2*module.n_window)]
        bounds = np.r_[0, np.cumsum([(n+1)//2 for n in mel_lengths])].tolist()
        if self.condition is None:
            self.bounds = bounds
            self.reference_mel = features.detach()
        elif bounds != self.bounds:
            raise ValueError('Length-preserving conditions changed encoder chunk layout')
        if self.condition and self.condition['family'] == 'input':
            changed = features.clone()
            self.mapping = np.arange(bounds[-1])
            self.mapping_valid = np.ones(bounds[-1], dtype=bool)
            shifts = []
            offset = 0
            for c, frames in enumerate(mel_lengths):
                a, b = bounds[c:c+2]
                if self.condition['operation'] == 'mel_mean':
                    changed[:, offset:offset+frames] = features[:, offset:offset+frames].mean(-1, keepdim=True)
                else:
                    shift = (frames//4)*2
                    shifts.append(shift)
                    changed[:, offset:offset+frames] = features[:, offset:offset+frames].roll(shift, -1)
                    self.mapping[a:b] = np.roll(np.arange(a, b), shift//2)
                    # Odd tails do not have a bijective stride-2 mapping; exclude from following statistics.
                    if frames % 2:
                        self.mapping_valid[a:b] = False
                    else:
                        for edge in (a, b, a+shift//2):
                            self.mapping_valid[max(a, edge-2):min(b, edge+2)] = False
                offset += frames
            self.details['mel_shift_frames_per_chunk'] = shifts if self.condition['operation'] == 'mel_roll' else None
            if 'input_features' in kwargs:
                return args, {**kwargs, 'input_features': changed}
            return (changed, *args[1:]), kwargs

    def _attn_pre(self, layer):
        def hook(module, args, kwargs):
            if self.condition and self.condition.get('layer') == layer:
                cu = kwargs.get('cu_seqlens', args[1] if len(args)>1 else None)
                if cu.tolist() != self.bounds:
                    raise ValueError('Intervention must preserve native chunk boundaries')
                self.cache[layer] = {'heads': module.num_heads, 'scaling': module.scaling}
        return hook

    def _qkv(self, layer, name):
        def hook(module, args, output):
            if layer in self.cache and self.condition['family'] != 'head':
                self.cache[layer][name] = output.detach().reshape(len(output), -1, module.out_features//self.cache[layer]['heads'])
        return hook

    def _head_output(self, layer):
        def hook(module, args):
            if layer not in self.cache:
                return
            state = self.cache.pop(layer)
            native = args[0].detach().reshape(self.bounds[-1], state['heads'], -1)
            if self.condition['family'] == 'head':
                changed = native.clone()
                changed[:, self.condition['head']] = 0
                if self.condition['operation'] == 'restore':
                    changed[:, self.condition['head']] = native[:, self.condition['head']]
            else:
                changed = intervention_head_output(state['q'], state['k'], state['v'], native,
                    self.bounds, state['scaling'], self.condition, self.selected, self.step_ms, self.seed)
            self.details['attention_computation'] = ('native_head_zero_or_restore'
                if self.condition['family'] == 'head' else 'native_plus_shadow_delta')
            if self.condition['family'] == 'range':
                radius = int(self.condition['window_ms']//self.step_ms)
                allowed = sum(int(range_mask(b-a, radius, 'window', self.seed).sum())
                              for a, b in zip(self.bounds, self.bounds[1:]))
                available = sum((b-a)**2 for a, b in zip(self.bounds, self.bounds[1:]))
                self.details.update(kept_connections_per_head=allowed,
                    available_connections_per_head=available, connection_keep_fraction=allowed/available,
                    active=allowed < available, effective_radius_ms=radius*self.step_ms)
            self.details['head_output_relative_l2'] = float((changed.float()-native.float()).norm()
                                                            / native.float().norm().clamp_min(1e-12))
            return (changed.reshape_as(args[0]), *args[1:])
        return hook

    def _projector_pre(self, module, args):
        hidden = args[0].detach()
        if self.condition is None:
            self.reference_hidden = hidden
            self.last_hidden = hidden
            return
        changed = hidden
        if self.condition['family'] in ('evidence', 'refill'):
            if self.keep_indices is not None:
                changed = hidden[self.keep_indices]
            elif len(self.selected):
                changed = hidden.clone()
                if self.condition['operation'] == 'norm':
                    changed[self.selected] = norm_matched(hidden, self.selected)
                else:
                    unselected = np.setdiff1d(np.arange(len(hidden)), self.selected)
                    changed[self.selected] = hidden[unselected].mean(0)
                if self.condition['operation'] == 'restore':
                    changed[self.selected] = hidden[self.selected]
        reference = self.reference_hidden
        self.details['original_projector_tokens'] = len(reference)
        self.details['projector_tokens'] = len(changed)
        self.details['kept_indices'] = self.keep_indices.tolist() if self.keep_indices is not None else None
        comparable = reference if self.keep_indices is None else reference[self.keep_indices]
        self.details['representation_relative_l2'] = float((changed.float()-comparable.float()).norm()
                                                            / comparable.float().norm().clamp_min(1e-12))
        self.details['pooled_representation_relative_l2'] = float(
            (changed.float().mean(0)-reference.float().mean(0)).norm() / reference.float().mean(0).norm().clamp_min(1e-12))
        self.last_hidden = changed
        return (changed, *args[1:])


@contextmanager
def compact_audio_request(bridge, controller):
    """Reuse the existing legalizer and mirror native compact audio M-RoPE."""
    if controller.keep_indices is None:
        yield
        return
    from mllm_comp.backbones.qwen25_omni.audio_compression import legalize_qwen25_omni_audio_tokens
    from mllm_comp.core.actions import TokenAction, LegalizationPolicy
    tower, thinker, runner = controller.tower, bridge.runtime.model.thinker, bridge.facade.runner
    original_lengths, original_rope, original_generate = tower._get_feat_extract_output_lengths, thinker.get_rope_index, runner.generate
    count = len(controller.keep_indices)
    def lengths(raw):
        cnn, output = original_lengths(raw)
        if output.numel() != 1 or int(output[0]) != len(controller.reference_hidden):
            raise ValueError('Audio output length differs from frozen reference')
        return cnn, output.new_tensor([count])
    def rope(*args, **kwargs):
        args = list(args)
        audio = args[5] if len(args)>5 else kwargs.get('audio_seqlens')
        if audio is not None:
            compact = audio.new_tensor([4*count])
            if len(args)>5:
                args[5] = compact
            else:
                kwargs['audio_seqlens'] = compact
        return original_rope(*args, **kwargs)
    def generate(runtime, state, *, processed_batch, **kwargs):
        spans = [s for s in state.modality_token_spans if s.modality == 'audio']
        if len(spans) != 1:
            raise ValueError('Expected one audio span')
        span = spans[0]
        action = TokenAction(op='keep', target_indices=controller.keep_indices.tolist(), metadata={
            'sample_index': span.sample_index, 'span_start': span.start, 'span_end': span.end,
            'keep_count': count, 'original_count': span.end-span.start, 'audio_unit_index': 0,
            'selector': 'attention_analysis', 'target_site': 'audio_encoder_output_pre_projector'})
        compact = legalize_qwen25_omni_audio_tokens(state, token_actions=[action],
            policy=LegalizationPolicy(position_policy='compact', mask_policy='rebuild_from_tokens',
                                       label_policy='rebuild_ignore_index'), stage_name='attention_analysis')
        return original_generate(runtime, compact, processed_batch=processed_batch, **kwargs)
    tower._get_feat_extract_output_lengths, thinker.get_rope_index, runner.generate = lengths, rope, generate
    try:
        yield
    finally:
        tower._get_feat_extract_output_lengths, thinker.get_rope_index, runner.generate = original_lengths, original_rope, original_generate


class FrozenProbe:
    """Optional fixed linear regression readout; this module never fits on B."""
    def __init__(self, weights_path, targets_path):
        import json
        with np.load(weights_path, allow_pickle=False) as data:
            self.weights, self.bias = data['weight'], data['bias']
            self.names = data['attributes'].tolist()
        with open(targets_path, encoding='utf-8') as source:
            self.targets = json.load(source)
        if self.weights.ndim != 2 or self.weights.shape[1] != len(self.names):
            raise ValueError('Probe weight must be [pre_projector_dim, attributes]')
        if self.bias.shape != (len(self.names),) or not np.isfinite(self.weights).all() or not np.isfinite(self.bias).all():
            raise ValueError('Probe weights/bias must be finite with one bias per attribute')
    def read(self, hidden, sample_id):
        value = hidden.float().mean(0).cpu().numpy() @ self.weights + self.bias
        target = np.array([self.targets[sample_id][name] for name in self.names], dtype=float)
        return dict(values=value.tolist(), squared_error=np.square(value-target).tolist(), attributes=self.names)
