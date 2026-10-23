"""Read-only hooks for the Transformers 4.57.3 Qwen2.5-Omni audio tower."""
from __future__ import annotations

import numpy as np
import torch

from .metrics import attention_summary, similarity_summary, virtual_selection


class AudioEncoderObserver:
    def __init__(self, tower, *, sample_rate, hop_length, layers=None,
                 query_block=64, example_chunk=0, virtual_keep_ratio=0.1, segments=10,
                 retain_scores=False):
        if tower.training:
            raise ValueError('Audio sink observation requires model.eval()')
        if sample_rate <= 0 or hop_length <= 0:
            raise ValueError('Read actual sampling_rate and hop_length from the feature extractor')
        self.tower = tower
        self.step = hop_length / sample_rate
        self.layers = list(range(len(tower.layers))) if layers is None else sorted(set(layers))
        if not self.layers or not set(self.layers).issubset(range(len(tower.layers))):
            raise ValueError('Invalid encoder layer selection')
        self.query_block, self.example_chunk = query_block, example_chunk
        self.keep_ratio, self.segments = virtual_keep_ratio, segments
        self.handles, self.cache = [], {}
        self.active = False
        self.retain_scores = retain_scores

    def install(self):
        if self.handles:
            raise RuntimeError('Observer already installed')
        self.handles.append(self.tower.register_forward_pre_hook(self._tower_pre, with_kwargs=True))
        for index in self.layers:
            layer = self.tower.layers[index]
            attn = layer.self_attn
            self.handles.append(attn.register_forward_pre_hook(self._attention_pre(index), with_kwargs=True))
            for name in ('q', 'k', 'v'):
                self.handles.append(getattr(attn, name + '_proj').register_forward_hook(self._projection(index, name)))
            self.handles.append(attn.out_proj.register_forward_pre_hook(self._attention_output(index)))
            self.handles.append(layer.final_layer_norm.register_forward_pre_hook(self._post_mhsa(index)))
            self.handles.append(layer.register_forward_hook(self._layer_output(index)))

    def close(self):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()
        self.cache.clear()
        self.active = False

    def begin(self, *, example=False, check_av=False, seed=42):
        if self.active:
            raise RuntimeError('Previous audio observation was not finished')
        self.active, self.example, self.check_av = True, example, check_av
        self.seed, self.tower_calls = seed, 0
        self.cache, self.rows, self.arrays = {}, {}, {}
        self.scores = {}

    def finish(self):
        self.active = False
        if self.tower_calls != 1 or set(self.rows) != set(self.layers) or self.cache:
            raise RuntimeError(f'Incomplete observation: audio calls={self.tower_calls}, layers={sorted(self.rows)}')
        return {'mel_length': self.mel_length, 'encoder_tokens': self.bounds[-1],
                'chunk_lengths': np.diff(self.bounds).tolist(), 'time_step_s': 2 * self.step,
                'self_attention_included': True, 'head_aggregation': 'separate_heads',
                'chunk_aggregation': 'equal_chunk_mean_within_audio',
                'similarity_key_space': 'concatenated_actual_head_keys',
                'example_chunk': self.example_chunk if self.example else None,
                'layers': [self.rows[i] for i in self.layers]}, self.arrays

    def _tower_pre(self, module, args, kwargs):
        if not self.active:
            return
        self.tower_calls += 1
        if self.tower_calls != 1:
            raise RuntimeError('Expected one audio tower call per MMSU request')
        features = kwargs.get('input_features', args[0] if args else None)
        lengths = kwargs.get('feature_lens', args[1] if len(args) > 1 else None)
        if lengths is None or lengths.numel() != 1 or features.ndim != 2:
            raise ValueError('This observer supports one packed audio per request')
        self.mel_length = int(lengths[0])
        mel_chunks = [min(2 * module.n_window, self.mel_length - a)
                      for a in range(0, self.mel_length, 2 * module.n_window)]
        enc_lengths = [(n + 1) // 2 for n in mel_chunks]
        self.bounds = np.r_[0, np.cumsum(enc_lengths)].tolist()
        mel_starts = np.r_[0, np.cumsum(mel_chunks)]
        self.arrays['time_s'] = np.concatenate([(mel_starts[c] + 2 * np.arange(n)) * self.step
                                               for c, n in enumerate(enc_lengths)])
        self.arrays['boundaries'] = np.asarray(self.bounds)
        if self.example:
            self.arrays['mel'] = features.detach().float().cpu().numpy()
            self.arrays['mel_step_s'] = np.asarray(self.step)

    def _attention_pre(self, index):
        def hook(module, args, kwargs):
            if not self.active:
                return
            if index in self.rows or index in self.cache:
                raise RuntimeError(f'Encoder attention {index} ran more than once')
            cu = kwargs.get('cu_seqlens', args[1] if len(args) > 1 else None)
            if cu is None or cu.tolist() != self.bounds:
                raise ValueError('Actual cu_seqlens differs from frontend time mapping')
            mask = kwargs.get('attention_mask', args[2] if len(args) > 2 else None)
            if mask is not None:
                # This implementation deliberately supports only the verified block mask.
                for a, b in zip(self.bounds, self.bounds[1:]):
                    if torch.any(mask[..., a:b, a:b] != 0):
                        raise ValueError('Unexpected restriction inside a bidirectional chunk')
                    if a and not torch.all(mask[..., a:b, :a] < -1e4):
                        raise ValueError('Unmasked keys outside the chunk')
                    if b < self.bounds[-1] and not torch.all(mask[..., a:b, b:] < -1e4):
                        raise ValueError('Unmasked keys outside the chunk')
            elif module.config._attn_implementation != 'flash_attention_2':
                raise ValueError('Only FA2 can use cu_seqlens without an explicit block mask')
            self.cache[index] = {'heads': module.num_heads, 'scaling': module.scaling}
        return hook

    def _projection(self, index, name):
        def hook(module, args, output):
            if self.active:
                state = self.cache[index]
                state[name] = output.detach().reshape(len(output), state['heads'], -1)
        return hook

    def _attention_output(self, index):
        def hook(module, args):
            if not self.active:
                return
            state = self.cache[index]
            native = args[0].detach().reshape(self.bounds[-1], state['heads'], -1)
            chunk = self.example_chunk if self.example and self.example_chunk < len(self.bounds)-1 else None
            r, heads, checks, matrix = attention_summary(
                state['q'], state['k'], state['v'], native, self.bounds, state['scaling'],
                query_block=self.query_block, check_av=self.check_av, save_chunk=chunk,
                time_step_ms=2000 * self.step)
            if self.retain_scores:
                self.scores[index] = r
            similarity = {}
            for name in ('k', 'v'):
                values, stats = similarity_summary(state[name], self.bounds)
                similarity[name] = stats
                if self.example:
                    self.arrays[f'{name}_adj_{index}'] = values
            self.rows[index] = {'layer': index, 'heads': heads, 'similarity': similarity, 'shadow_check': checks}
            if self.example:
                self.arrays[f'r_{index}'] = r
                if matrix is not None:
                    self.arrays[f'attention_{index}'] = matrix
                    a, b = self.bounds[chunk:chunk+2]
                    self.arrays[f'keys_{index}'] = state['k'][a:b].float().flatten(1).cpu().numpy()
            if index == len(self.tower.layers)-1:
                # Match the platform's pair-averaged pre-projector scoring coordinates.
                raw = r.mean(axis=0)
                pooled = raw[:len(raw)//2*2].reshape(-1, 2).mean(axis=1)
                result, top, segment = virtual_selection(pooled, self.keep_ratio, self.segments, self.seed)
                self.rows[index]['virtual_selection'] = result
                if self.example:
                    self.arrays['pooled_scores'] = pooled
                    self.arrays['pooled_time_s'] = self.arrays['time_s'][:len(pooled)*2].reshape(-1, 2).mean(axis=1)
                    self.arrays['virtual_topk'] = top
                    self.arrays['virtual_segmentwise'] = segment
            for name in ('q', 'k', 'v'):
                del state[name]
        return hook

    def _post_mhsa(self, index):
        def hook(module, args):
            if not self.active:
                return
            values, summary = similarity_summary(args[0], self.bounds)
            self.rows[index]['similarity']['post_mhsa'] = summary
            if self.example:
                self.arrays[f'h_adj_{index}'] = values
                if self.example_chunk < len(self.bounds)-1:
                    a, b = self.bounds[self.example_chunk:self.example_chunk+2]
                    self.arrays[f'hidden_{index}'] = args[0][a:b].detach().float().cpu().numpy()
        return hook

    def _layer_output(self, index):
        def hook(module, args, output):
            if self.active:
                _, summary = similarity_summary(output[0], self.bounds)
                self.rows[index]['similarity']['post_ffn'] = summary
                del self.cache[index]
        return hook
