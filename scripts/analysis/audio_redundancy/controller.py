"""Request-local interventions; all hook handles are removed on close."""
import numpy as np
import torch

from .algorithms import (change_scores, feasible_budget, matched_smoothing,
                         mel_reconstruction, pool_groups, score_groups, subspace, uniform_groups)


class RedundancyController:
    def __init__(self, tower, capture):
        self.tower, self.capture = tower, capture
        self.handles = []
        self.condition = None
        self.keep_indices = None

    def install(self):
        self.handles.append(self.tower.register_forward_pre_hook(self._mel_hook, with_kwargs=True))
        self.handles.append(self.tower.proj.register_forward_pre_hook(self._pin_hook))
        self.handles.append(self.tower.proj.register_forward_hook(self._pout_hook))
        for index in self.capture.layers:
            name = "E_L" if index == len(self.tower.layers) - 1 else f"E_l{index}"
            self.handles.append(self.tower.layers[index].register_forward_hook(self._layer_hook(name)))

    def close(self):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()

    def reset(self):
        self.condition = None
        self.keep_indices = None
        self.bases = {}
        self.mel_decomposition = None

    @property
    def reference_hidden(self):
        # Contract consumed by the existing compact_audio_request context.
        return self.capture.hidden["P_in"]

    def prepare(self, condition, discrete, *, seed=42, example=False):
        self.condition = condition
        self.keep_indices = None
        self.applications = 0
        self.details = dict(condition_id=condition["id"], family=condition["family"])
        self.arrays = {}
        family = condition["family"]
        if family == "subspace":
            stage = condition["stage"]
            hidden = self.capture.hidden[stage]
            basis = None
            if condition["method"] == "pca":
                if stage not in self.bases:
                    x = hidden.float()
                    self.bases[stage] = torch.linalg.svd(x - x.mean(0), full_matrices=False).Vh.T
                basis = self.bases[stage]
            self.changed, info = subspace(hidden, condition["k"], condition["method"], seed=seed, basis=basis)
            self.details.update(info, stage=stage, requested_k=condition["k"],
                                original_tokens=len(hidden), output_tokens=len(hidden), retention=1.)
            return
        if family == "mel":
            original = self.capture.mel
            if condition["method"] == "identity":
                self.changed = original
                info = dict(mse=0., relative_l2=0.)
            else:
                if self.mel_decomposition is None:
                    x = original.float()
                    self.mel_decomposition = torch.linalg.svd(x - x.mean(0), full_matrices=False)
                reconstructed, info = mel_reconstruction(original, condition["k"], decomposition=self.mel_decomposition)
                if condition["method"] == "svd":
                    self.changed = reconstructed.to(original.dtype)
                else:
                    self.changed, match = matched_smoothing(original, info["mse"],
                        axis=0 if condition["method"] == "time_matched" else 1)
                    info.update(match)
                info["relative_l2"] = float((self.changed.float() - original.float()).norm()
                                             / original.float().norm().clamp_min(1e-12))
            n = len(self.reference_hidden)
            self.details.update(info, domain="native_log_mel", stage="Mel",
                                original_tokens=n, output_tokens=n, retention=1.)
            if example:
                self.arrays["mel"] = self.changed.float().cpu().numpy().T
            return
        hidden = self.reference_hidden
        times, chunks = self.capture.layouts["P_in"]
        n = len(hidden)
        method = condition["method"]
        broadcast = condition.get("broadcast", False) or (family == "runs" and method == "broadcast")
        if family == "runs":
            unit = discrete[("P_in", condition["codebook_size"])]
            groups = unit["groups"]
            requested = budget = len(groups)
            if method == "matched_uniform":
                groups = uniform_groups(chunks, budget)
        else:
            requested, budget = feasible_budget(chunks, condition["ratio"])
            if method in {"uniform", "sample"}:
                groups = uniform_groups(chunks, budget)
            else:
                if method in {"l2", "cosine"}:
                    scores = change_scores(hidden, method)
                elif method == "random":
                    scores = np.random.default_rng(seed).random(n)
                else:
                    unit = discrete[("P_in", condition["codebook_size"])]
                    scores = unit["surprisal"] if method == "surprisal" else unit["entropy"]
                    if method == "low_entropy":
                        scores = -scores
                groups = score_groups(chunks, budget, scores)
        self.changed, error = pool_groups(hidden, groups, broadcast=broadcast, sample=method == "sample")
        anchors = np.array([g[(len(g) - 1) // 2] for g in groups], dtype=int)
        if not broadcast and len(self.changed) != n:
            self.keep_indices = anchors
        lengths = np.array([len(g) for g in groups])
        duration = self.capture.mel_length * self.capture.mel_step_ms / 1000
        self.details.update(stage="P_in", original_tokens=n, output_tokens=len(self.changed),
            requested_budget=requested, feasible_segment_budget=budget,
            budget_clamped_by_hard_chunks=budget != requested, retention=len(self.changed) / n,
            tokens_per_second=len(self.changed) / duration, segment_count=len(groups),
            mean_segment_tokens=float(lengths.mean()), max_segment_tokens=int(lengths.max()),
            reconstruction_relative_l2=error, broadcast=broadcast,
            position_policy="compact", duration_supplied_to_model=False)
        if example:
            self.arrays.update(segment_start=np.array([g[0] for g in groups]),
                               segment_end=np.array([g[-1] + 1 for g in groups]),
                               retained_time_ms=times[anchors], segment_tokens=lengths)
        # Groups always partition the entire native sequence, in order.
        if not np.array_equal(np.concatenate(groups), np.arange(n)):
            raise RuntimeError("Segmentation lost, duplicated or reordered source tokens")

    def finish(self):
        if self.applications != 1:
            raise RuntimeError(f"Expected one intervention application, got {self.applications}")
        return self.details

    def _replace(self, hidden):
        if hidden.shape[1:] != self.changed.shape[1:]:
            raise ValueError("Intervention channel shape changed")
        self.applications += 1
        return self.changed.to(device=hidden.device, dtype=hidden.dtype)

    def _pin_hook(self, module, args):
        if self.condition and (self.condition["family"] in {"segment", "runs"}
                               or self.condition.get("stage") == "P_in"):
            return (self._replace(args[0]), *args[1:])

    def _pout_hook(self, module, args, output):
        if self.condition and self.condition.get("stage") == "P_out":
            return self._replace(output)

    def _layer_hook(self, name):
        def hook(module, args, output):
            if self.condition and self.condition.get("stage") == name:
                return (self._replace(output[0]), *output[1:])
        return hook

    def _mel_hook(self, module, args, kwargs):
        if self.condition and self.condition["family"] == "mel":
            features = kwargs.get("input_features", args[0] if args else None)
            if features.ndim != 2 or features.shape[1] != len(self.capture.mel):
                raise ValueError("Expected single valid packed Mel matching native reference")
            changed = self.changed.T.to(device=features.device, dtype=features.dtype)
            self.applications += 1
            if "input_features" in kwargs:
                return args, {**kwargs, "input_features": changed}
            return (changed, *args[1:]), kwargs
