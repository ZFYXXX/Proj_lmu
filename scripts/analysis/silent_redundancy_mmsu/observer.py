"""Read-only Qwen2.5-Omni attention/acoustic observer for stages 1 and 2."""
from __future__ import annotations

import numpy as np
import torch

from .metrics import aligned_acoustic_features, observational_summary


class SilentRedundancyObserver:
    def __init__(self, tower, *, sample_rate: int, hop_length: int,
                 layers=(3, 6, 9), query_block=64):
        if tower.training:
            raise ValueError("Silent-redundancy observation requires model.eval()")
        self.tower = tower
        self.layers = tuple(sorted(set(int(index) for index in layers)))
        if not self.layers or not set(self.layers).issubset(range(len(tower.layers))):
            raise ValueError("Invalid encoder layers")
        self.sample_rate = int(sample_rate)
        self.hop_length = int(hop_length)
        self.mel_step_s = self.hop_length / self.sample_rate
        self.query_block = int(query_block)
        self.handles = []
        self.active = False

    def install(self):
        if self.handles:
            raise RuntimeError("Observer is already installed")
        self.handles.append(self.tower.register_forward_pre_hook(self._tower_pre, with_kwargs=True))
        for index in self.layers:
            attention = self.tower.layers[index].self_attn
            self.handles.append(attention.register_forward_pre_hook(
                self._attention_pre(index), with_kwargs=True))
            self.handles.append(attention.q_proj.register_forward_hook(self._projection(index, "q")))
            self.handles.append(attention.k_proj.register_forward_hook(self._projection(index, "k")))
            self.handles.append(attention.out_proj.register_forward_pre_hook(self._attention_output(index)))

    def close(self):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()
        self.active = False

    def begin(self, *, example: bool):
        if self.active:
            raise RuntimeError("Previous observation was not finished")
        self.active = True
        self.example = bool(example)
        self.calls = 0
        self.cache = {}
        self.scores = {}
        self.arrays = {}

    def _tower_pre(self, module, args, kwargs):
        if not self.active:
            return
        self.calls += 1
        features = kwargs.get("input_features", args[0] if args else None)
        lengths = kwargs.get("feature_lens", args[1] if len(args) > 1 else None)
        if self.calls != 1 or features is None or lengths is None or lengths.numel() != 1:
            raise ValueError("Observer supports exactly one packed audio per request")
        self.mel_length = int(lengths[0])
        values, chunk_lengths, time_indices, boundaries = aligned_acoustic_features(
            features, self.mel_length, int(module.n_window))
        self.acoustic = {name: value.detach().cpu().numpy() for name, value in values.items()}
        self.chunk_lengths = chunk_lengths
        self.boundaries = boundaries
        self.time_s = time_indices * self.mel_step_s
        if self.example:
            self.arrays.update(
                mel=features[:, :self.mel_length].detach().float().cpu().numpy(),
                mel_step_s=np.asarray(self.mel_step_s),
                time_s=self.time_s,
                boundaries=self.boundaries,
                **{f"acoustic_{name}": value for name, value in self.acoustic.items()},
            )

    def _attention_pre(self, index):
        def hook(module, args, kwargs):
            if not self.active:
                return
            cu = kwargs.get("cu_seqlens", args[1] if len(args) > 1 else None)
            if cu is None or cu.detach().cpu().tolist() != self.boundaries.tolist():
                raise ValueError("Attention chunks differ from the frontend time map")
            self.cache[index] = {"heads": int(module.num_heads), "scaling": float(module.scaling)}
        return hook

    def _projection(self, index, name):
        def hook(module, args, output):
            if self.active:
                heads = self.cache[index]["heads"]
                self.cache[index][name] = output.detach().reshape(len(output), heads, -1)
        return hook

    @torch.no_grad()
    def _received_attention(self, q, k, scaling):
        heads = q.shape[1]
        received = torch.empty((heads, len(q)), dtype=torch.float32, device=q.device)
        for start, end in zip(self.boundaries[:-1], self.boundaries[1:]):
            query = q[start:end].transpose(0, 1)
            key = k[start:end].transpose(0, 1)
            incoming = torch.zeros((heads, end - start), dtype=torch.float32, device=q.device)
            for offset in range(0, end - start, self.query_block):
                weights = torch.softmax(
                    (query[:, offset:offset + self.query_block] @ key.transpose(-2, -1)) * scaling,
                    dim=-1,
                    dtype=torch.float32,
                )
                incoming += weights.sum(dim=-2)
            received[:, start:end] = incoming
        return received.mean(dim=0).cpu().numpy()

    def _attention_output(self, index):
        def hook(module, args):
            if not self.active:
                return
            state = self.cache[index]
            score = self._received_attention(state["q"], state["k"], state["scaling"])
            self.scores[index] = score
            del self.cache[index]
            if self.example:
                self.arrays[f"attention_l{index}"] = score
                for ratio in (0.75, 0.50, 0.25):
                    from .metrics import exact_chunk_topk_mask
                    self.arrays[f"keep_l{index}_r{int(100 * ratio)}"] = exact_chunk_topk_mask(
                        score, self.boundaries, ratio)
        return hook

    def finish(self):
        self.active = False
        if self.calls != 1 or set(self.scores) != set(self.layers) or self.cache:
            raise RuntimeError("Incomplete silent-redundancy observation")
        layers = []
        for index in self.layers:
            layers.append({
                "layer": index,
                **observational_summary(
                    self.scores[index], self.acoustic, self.boundaries,
                ),
            })
        measurements = {
            "mel_length": self.mel_length,
            "encoder_tokens": int(self.boundaries[-1]),
            "chunk_lengths": self.chunk_lengths,
            "mel_step_s": self.mel_step_s,
            "encoder_nominal_step_s": 2 * self.mel_step_s,
            "layers": layers,
        }
        return measurements, self.arrays
