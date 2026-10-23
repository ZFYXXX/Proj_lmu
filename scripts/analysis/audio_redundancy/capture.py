"""Reuse the existing verified physical-time observer without lag calculations."""
import numpy as np

from scripts.analysis.audio_temporal.observer import TemporalObserver


class RepresentationCapture(TemporalObserver):
    def __init__(self, tower, *, sample_rate, hop_length, layers):
        super().__init__(tower, sample_rate=sample_rate, hop_length=hop_length,
                         layers=layers, lags_ms=(40,))

    def begin(self, *, example=False, seed=42):
        super().begin(example=example, seed=seed)
        self.hidden = {}
        self.layouts = {}
        self.mel = None

    def _tower_pre(self, module, args, kwargs):
        super()._tower_pre(module, args, kwargs)
        if self.active:
            features = kwargs.get("input_features", args[0] if args else None)
            self.mel = features[:, :self.mel_length].detach().T.contiguous()

    def _record(self, name, values, time_ms, chunk_ids, labels, *, layer):
        if name in self.seen or len(values) != len(time_ms):
            raise ValueError(f"Unexpected representation layout at {name}")
        self.hidden[name] = values.detach()
        self.layouts[name] = (np.asarray(time_ms), np.asarray(chunk_ids))
        self.rows.append(dict(name=name, layer=layer, tokens=len(values), channels=values.shape[-1]))
        self.seen.add(name)

    def calibration_arrays(self, tokens, *, seed):
        """One contiguous within-chunk block per stage/audio, without labels."""
        from .algorithms import chunk_ranges
        rng = np.random.default_rng(seed)
        arrays = {}
        for name, hidden in self.hidden.items():
            if name == "C":
                continue
            times, chunks = self.layouts[name]
            ranges = chunk_ranges(chunks)
            candidates = [(a, b) for a, b in ranges if b - a >= tokens]
            if candidates:
                weights = np.array([b - a - tokens + 1 for a, b in candidates], dtype=float)
                a, b = candidates[int(rng.choice(len(candidates), p=weights / weights.sum()))]
                start = int(rng.integers(a, b - tokens + 1))
                indices = np.arange(start, start + tokens)
            else:
                a, b = max(ranges, key=lambda pair: pair[1] - pair[0])
                indices = np.arange(a, b)
            arrays[f"hidden_{name}"] = hidden[indices.tolist()].float().cpu().numpy().astype(np.float16)
            arrays[f"time_{name}"] = times[indices]
            arrays[f"chunks_{name}"] = chunks[indices]
        return arrays
