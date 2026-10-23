"""Compact branch caches; the model clock remains in original token coordinates."""
from copy import deepcopy

import torch
from transformers.cache_utils import DynamicCache


class CompactCache:
    """One sample, one modality. Compatible with existing replacement kernels."""

    def __init__(self):
        self.keys = {}
        self.values = {}

    def get_seq_length(self, layer_idx=0):
        key = self.keys.get(layer_idx)
        return 0 if key is None else key.shape[-2]

    def update(self, key, value, layer_idx, cache_kwargs=None):
        if layer_idx in self.keys:
            key = torch.cat((self.keys[layer_idx], key), dim=-2)
            value = torch.cat((self.values[layer_idx], value), dim=-2)
        self.keys[layer_idx], self.values[layer_idx] = key, value
        return key, value

    def reorder_cache(self, beam_idx):
        for i in self.keys:
            self.keys[i] = self.keys[i].index_select(0, beam_idx)
            self.values[i] = self.values[i].index_select(0, beam_idx)


class ModalitySplitCache(DynamicCache):
    """Native layers retain HF storage; selected layers own no global KV slots.

    Global length counts input positions (including padding), while compact cache
    lengths count valid tokens only. No zero/placeholder image KV is allocated.
    """

    def __init__(self, config):
        super().__init__(config=config)
        self._split_config = config
        self.global_length = 0
        self.branches = {}

    def reset(self):
        # HF 4.57 DynamicLayer.reset zeros tensors but retains their sequence
        # dimensions. Rebuild native storage so neither zeros nor stale history
        # become prefix tokens of the next independent request.
        DynamicCache.__init__(self, config=self._split_config)
        self.global_length = 0
        self.branches.clear()

    def get_seq_length(self, layer_idx=0):
        return self.global_length

    def get_mask_sizes(self, cache_position, layer_idx=0):
        return self.global_length + cache_position.shape[0], 0

    def branch(self, layer, sample, modality):
        return self.branches.setdefault((layer, sample, modality), CompactCache())

    def modality_lengths(self):
        result = {}
        for (layer, sample, modality), cache in self.branches.items():
            row = result.setdefault(layer, {}).setdefault(sample, {"text": 0, "image": 0})
            row[modality] = cache.get_seq_length(layer)
        return result

    def reorder_cache(self, beam_idx):
        if self.branches:
            # Recurrent owners contain WeakMethod objects; reconstruct them instead
            # of deepcopying owners, and clone duplicate beams independently.
            from mllm_comp.backbones.qwen3_vl.attention_replacement import _store_cache_state
            reordered = {}
            for new_sample, old_sample in enumerate(beam_idx.tolist()):
                for (layer, sample, modality), cache in self.branches.items():
                    if sample != old_sample:
                        continue
                    new = CompactCache()
                    new.keys = {i: x.clone() for i, x in cache.keys.items()}
                    new.values = {i: x.clone() for i, x in cache.values.items()}
                    owner = getattr(cache, "_mllm_comp_attention_states", None)
                    if owner is not None:
                        for i, state in owner.layers.items():
                            _store_cache_state(new, i, deepcopy(state))
                    reordered[layer, new_sample, modality] = new
            self.branches = reordered
        super().reorder_cache(beam_idx)

    def crop(self, max_length):
        raise ValueError("Modality-split caches do not support speculative/cropped generation.")

    def batch_repeat_interleave(self, repeats):
        raise ValueError("Expand inputs before constructing modality-split caches.")

    def batch_select_indices(self, indices):
        self.reorder_cache(indices)
