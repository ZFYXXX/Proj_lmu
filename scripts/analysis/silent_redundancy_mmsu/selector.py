"""Process-local scoring adapters for real encoder Top-K pruning."""
from __future__ import annotations

import math

import numpy as np
import torch

from . import ACOUSTIC_FEATURES, CAUSAL_SELECTORS
from .metrics import aligned_acoustic_features, causal_drop_indices


class PruningSelector:
    """Patch only the score source while retaining the platform's real pruning.

    The platform still owns the compressed audio forward, per-chunk budgets,
    chronological reordering, cu_seqlens rebuild, native pooling and projector.
    This object replaces the imported scoring helper inside the worker process
    and tracks original time provenance through the exact selected indices.
    """

    def __init__(self, tower, *, selector: str, layers: list[int], keep_ratio: float,
                 minimum_keep_tokens: int, seed: int, sample_rate: int, hop_length: int):
        allowed = {"attention", "random", *ACOUSTIC_FEATURES, *CAUSAL_SELECTORS}
        if selector not in allowed:
            raise ValueError(f"Unsupported selector {selector!r}")
        self.tower = tower
        self.selector = selector
        self.layers = tuple(int(layer) for layer in layers)
        self.keep_ratio = float(keep_ratio)
        self.minimum_keep_tokens = int(minimum_keep_tokens)
        self.seed = int(seed)
        self.mel_step_s = int(hop_length) / int(sample_rate)
        self.active = False
        self.handle = None
        self._module = None

    def install(self):
        if self._module is not None:
            raise RuntimeError("Selector is already installed")
        from mllm_comp.backbones.qwen25_omni import encoder_compression
        from mllm_comp.compressors.encoder_level import attention_topk

        self._module = encoder_compression
        self._selection_module = attention_topk
        self._original_score = encoder_compression.mean_received_attention_scores
        self._original_select = attention_topk.select_ragged_attention_topk
        encoder_compression.mean_received_attention_scores = self._score
        attention_topk.select_ragged_attention_topk = self._select
        self.handle = self.tower.register_forward_pre_hook(self._tower_pre, with_kwargs=True)

    def close(self):
        if self._module is not None:
            self._module.mean_received_attention_scores = self._original_score
            self._selection_module.select_ragged_attention_topk = self._original_select
            self._module = None
        if self.handle is not None:
            self.handle.remove()
            self.handle = None
        self.active = False

    def begin(self, doc_index: int, *, example: bool):
        if self.active:
            raise RuntimeError("Previous pruning request was not finished")
        self.active = True
        self.doc_index = int(doc_index)
        self.example = bool(example)
        self.calls = 0
        self.score_calls = 0
        self.select_calls = 0
        self.stage_reports = []
        self.example_arrays = {}

    def _tower_pre(self, module, args, kwargs):
        if not self.active:
            return
        self.calls += 1
        features = kwargs.get("input_features", args[0] if args else None)
        lengths = kwargs.get("feature_lens", args[1] if len(args) > 1 else None)
        if self.calls != 1 or features is None or lengths is None or lengths.numel() != 1:
            raise ValueError("Pruning analysis supports one packed audio per request")
        self.mel_length = int(lengths[0])
        values, chunk_lengths, time_indices, boundaries = aligned_acoustic_features(
            features, self.mel_length, int(module.n_window))
        split_values = {
            name: list(value.split(chunk_lengths)) for name, value in values.items()
        }
        provenance = []
        for start, end in zip(boundaries[:-1], boundaries[1:]):
            provenance.append(torch.arange(start, end, device=features.device, dtype=torch.long))
        self.current = [
            {
                **{name: split_values[name][chunk] for name in ACOUSTIC_FEATURES},
                "provenance": provenance[chunk],
            }
            for chunk in range(len(chunk_lengths))
        ]
        self.initial_chunk_lengths = list(chunk_lengths)
        self.initial_tokens = int(boundaries[-1])
        self.boundaries = boundaries
        self.time_s = time_indices * self.mel_step_s
        self._random = []
        generator = torch.Generator(device=features.device)
        generator.manual_seed(self.seed + self.doc_index * 1009)
        for length in chunk_lengths:
            self._random.append(torch.rand(length, generator=generator, device=features.device))
        if self.example:
            self.example_arrays.update(
                mel=features[:, :self.mel_length].detach().float().cpu().numpy(),
                mel_step_s=np.asarray(self.mel_step_s),
                time_s=self.time_s,
                boundaries=boundaries,
                **{f"acoustic_{name}": value.detach().cpu().numpy()
                   for name, value in values.items()},
            )

    def _expected_output(self, length: int) -> int:
        return min(length, max(self.minimum_keep_tokens, int(math.floor(length * self.keep_ratio))))

    def _score(self, attention, hidden_states, chunk_lengths):
        if not self.active:
            return self._original_score(attention, hidden_states, chunk_lengths)
        actual = [len(chunk["provenance"]) for chunk in self.current]
        if list(chunk_lengths) != actual:
            raise RuntimeError(f"Selector provenance lengths {actual} != runtime {chunk_lengths}")
        native = None
        if self.selector == "attention" or self.selector in CAUSAL_SELECTORS:
            native = self._original_score(attention, hidden_states, chunk_lengths)
        self._pending_causal = []
        if self.selector == "attention":
            scores = native
        elif self.selector in ACOUSTIC_FEATURES:
            scores = torch.cat([chunk[self.selector] for chunk in self.current])
        elif self.selector == "random":
            scores = torch.cat(self._random)
        else:
            native_chunks = native.split(chunk_lengths)
            priorities = []
            for chunk_index, (chunk, native_chunk) in enumerate(zip(self.current, native_chunks)):
                length = len(native_chunk)
                drop_count = length - self._expected_output(length)
                dropped = causal_drop_indices(
                    chunk["temporal_change"].detach().cpu().numpy(),
                    native_chunk.detach().cpu().numpy(),
                    drop_count,
                    self.selector,
                    seed=self.seed + self.doc_index * 1009 + self.score_calls * 97 + chunk_index,
                )
                drop_tensor = torch.as_tensor(dropped, device=hidden_states.device)
                priority = torch.ones(length, dtype=torch.float32, device=hidden_states.device)
                priority[drop_tensor] = 0
                priorities.append(priority)
                self._pending_causal.append({
                    "dropped_tokens": int(len(dropped)),
                    "dropped_attention_mean": (
                        float(native_chunk[drop_tensor].float().mean()) if len(dropped) else None
                    ),
                    "dropped_temporal_change_mean": (
                        float(chunk["temporal_change"][drop_tensor].float().mean()) if len(dropped) else None
                    ),
                })
            scores = torch.cat(priorities)
        if not torch.isfinite(scores).all():
            raise ValueError("Non-finite pruning scores")
        self._pending_scores = scores.detach()
        self.score_calls += 1
        return scores

    def _select(self, hidden_states, scores, chunk_lengths, output_lengths, *, collect_details=False):
        output, native_reports = self._original_select(
            hidden_states,
            scores,
            chunk_lengths,
            output_lengths,
            collect_details=collect_details,
        )
        if not self.active:
            return output, native_reports
        if self.select_calls >= len(self.layers):
            raise RuntimeError("More Top-K stages ran than were frozen in the plan")
        score_chunks = scores.split(chunk_lengths)
        next_chunks = []
        next_random = []
        chunks_report = []
        for chunk_index, (state, random_score, score, output_length) in enumerate(
            zip(self.current, self._random, score_chunks, output_lengths)
        ):
            try:
                ranking = torch.argsort(score, descending=True, stable=True)
            except TypeError:  # pragma: no cover
                ranking = torch.argsort(score, descending=True)
            indices = torch.sort(ranking[:output_length])[0].to(dtype=torch.long)
            next_chunks.append({name: state[name].index_select(0, indices)
                                for name in (*ACOUSTIC_FEATURES, "provenance")})
            next_random.append(random_score.index_select(0, indices))
            row = {
                "chunk": chunk_index,
                "input_tokens": int(len(score)),
                "output_tokens": int(output_length),
                "mean_selected_score": float(score.index_select(0, indices).float().mean()),
            }
            if self._pending_causal:
                row.update(self._pending_causal[chunk_index])
            if self.example:
                row["kept_original_indices"] = state["provenance"].index_select(0, indices).cpu().tolist()
            chunks_report.append(row)
        self.current, self._random = next_chunks, next_random
        kept = torch.cat([chunk["provenance"] for chunk in self.current])
        layer = self.layers[self.select_calls]
        self.stage_reports.append({
            "layer": layer,
            "input_tokens": int(sum(chunk_lengths)),
            "output_tokens": int(sum(output_lengths)),
            "chunks": chunks_report,
        })
        if self.example:
            mask = np.zeros(self.initial_tokens, dtype=bool)
            mask[kept.detach().cpu().numpy()] = True
            self.example_arrays[f"kept_after_l{layer}"] = mask
        self.select_calls += 1
        return output, native_reports

    def finish(self):
        self.active = False
        if self.calls != 1 or self.score_calls != len(self.layers) or self.select_calls != len(self.layers):
            raise RuntimeError(
                f"Incomplete pruning trace: audio={self.calls}, score={self.score_calls}, "
                f"select={self.select_calls}, expected={len(self.layers)}"
            )
        final_tokens = sum(len(chunk["provenance"]) for chunk in self.current)
        report = {
            "selector": self.selector,
            "layers": list(self.layers),
            "keep_ratio_per_stage": self.keep_ratio,
            "initial_tokens": self.initial_tokens,
            "final_tokens": final_tokens,
            "realized_encoder_ratio": final_tokens / self.initial_tokens,
            "stages": self.stage_reports,
        }
        return report, self.example_arrays
