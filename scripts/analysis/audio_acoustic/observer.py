"""Read-only captures plus optional, length-preserving P_in smoothing."""
import numpy as np

from .algorithms import representation_drift, temporal_spectrum, time_lowpass


class AcousticObserver:
    """Capture the verified Qwen2.5-Omni audio stages with physical time maps."""

    def __init__(self, tower, *, sample_rate, hop_length, layers):
        if tower.training:
            raise ValueError("Acoustic observation requires model.eval()")
        self.tower = tower
        self.mel_step_ms = 1000.0 * int(hop_length) / int(sample_rate)
        self.layers = sorted(set(int(index) for index in layers))
        if not self.layers or not set(self.layers).issubset(range(len(tower.layers))):
            raise ValueError("Invalid audio encoder layer selection")
        self.handles = []
        self.active = False
        self.reference = {}
        self.reference_hidden = None

    def install(self):
        if self.handles:
            raise RuntimeError("Observer already installed")
        self.handles.append(self.tower.register_forward_pre_hook(self._tower_pre, with_kwargs=True))
        self.handles.append(self.tower.layers[0].register_forward_pre_hook(self._capture_c))
        for index in self.layers:
            self.handles.append(self.tower.layers[index].register_forward_hook(self._capture_layer(index)))
        self.handles.append(self.tower.proj.register_forward_pre_hook(self._capture_projector_input))
        self.handles.append(self.tower.proj.register_forward_hook(self._capture_projector_output))

    def close(self):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()
        self.active = False

    def begin(self, *, condition=None, example=False):
        if self.active:
            raise RuntimeError("Previous acoustic observation was not finished")
        self.active = True
        self.example = bool(example)
        self.calls = 0
        self.rows = []
        self.arrays = {}
        self.seen = set()
        self.condition = condition
        self.prepared = None
        self.intervention = {}
        if condition is None:
            self.reference = {}
            self.reference_hidden = None
        elif condition["family"] == "time_lowpass":
            self._prepare_hidden()

    def _prepare_hidden(self):
        condition = self.condition
        if self.reference_hidden is None:
            raise RuntimeError("A native reference must precede each condition")
        hidden = self.reference_hidden
        self.prepared = time_lowpass(hidden, self.reference_chunks, condition["nyquist_ratio"])
        self.intervention.update(original_tokens=len(hidden), output_tokens=len(self.prepared),
                                 space="P_in_after_native_pool_and_ln_post")

    def _tower_pre(self, module, args, kwargs):
        if not self.active:
            return
        self.calls += 1
        if self.calls != 1:
            raise RuntimeError("Expected exactly one packed audio tower call per MMSU request")
        features = kwargs.get("input_features", args[0] if args else None)
        lengths = kwargs.get("feature_lens", args[1] if len(args) > 1 else None)
        if features is None or lengths is None or features.ndim != 2 or lengths.numel() != 1:
            raise ValueError("Acoustic analysis supports one packed audio per request")
        self.mel_length = int(lengths[0])
        mel_chunks = [min(2 * module.n_window, self.mel_length - start)
                      for start in range(0, self.mel_length, 2 * module.n_window)]
        encoder_lengths = [(length + 1) // 2 for length in mel_chunks]
        self.encoder_bounds = np.r_[0, np.cumsum(encoder_lengths)]
        mel_starts = np.r_[0, np.cumsum(mel_chunks[:-1])]
        self.encoder_time_ms = np.concatenate([
            (mel_starts[chunk] + 2 * np.arange(length)) * self.mel_step_ms
            for chunk, length in enumerate(encoder_lengths)])
        self.encoder_chunk_ids = np.concatenate([
            np.full(length, chunk, dtype=int) for chunk, length in enumerate(encoder_lengths)])
        if self.example:
            self.arrays.update(
                mel=features[:, :self.mel_length].detach().float().cpu().numpy(),
                mel_time_ms=np.arange(self.mel_length) * self.mel_step_ms,
                encoder_time_ms=self.encoder_time_ms,
                encoder_chunk_id=self.encoder_chunk_ids,
            )

    def _capture_c(self, module, args):
        if self.active:
            self._record("C", args[0], self.encoder_time_ms, self.encoder_chunk_ids, layer=None)

    def _capture_layer(self, index):
        def hook(module, args, output):
            if self.active:
                name = "E_L" if index == len(self.tower.layers) - 1 else f"E_l{index}"
                self._record(name, output[0], self.encoder_time_ms, self.encoder_chunk_ids, layer=index)
        return hook

    def _projector_layout(self, output_length):
        n = len(self.encoder_time_ms)
        if output_length < 1 or output_length > n:
            raise ValueError("Unexpected pre-projector sequence length")
        if output_length in {n // 2, (n + 1) // 2}:
            groups = [np.arange(2 * index, min(n, 2 * index + 2))
                      for index in range(output_length)]
        else:
            groups = [group for group in np.array_split(np.arange(n), output_length) if len(group)]
        if len(groups) != output_length or any(not len(group) for group in groups):
            raise RuntimeError("Could not map native pooling outputs to encoder time")
        times, chunk_ids, crossings = [], [], 0
        for output_index, group in enumerate(groups):
            times.append(float(np.mean(self.encoder_time_ms[group])))
            chunks = np.unique(self.encoder_chunk_ids[group])
            if len(chunks) == 1:
                chunk_ids.append(int(chunks[0]))
            else:
                # Keep cross-chunk pool outputs out of within-chunk time smoothing.
                chunk_ids.append(-1 - output_index)
                crossings += 1
        return np.asarray(times), np.asarray(chunk_ids), crossings

    def _capture_projector_input(self, module, args):
        if not self.active:
            return
        hidden = args[0]
        times, chunks, crossings = self._projector_layout(len(hidden))
        self.native_pool_crossings = crossings
        if self.condition is None:
            self.reference_hidden = hidden.detach()
            self.reference_chunks = chunks.copy()
        changed = self.prepared if self.prepared is not None else hidden
        self.projector_time_ms, self.projector_chunk_ids = times, chunks
        self._record("P_in", changed, times, chunks, layer=None)
        if self.example:
            self.arrays["projector_time_ms"] = times
            self.arrays["projector_chunk_id"] = chunks
        if self.prepared is not None:
            return (changed,) + args[1:]

    def _capture_projector_output(self, module, args, output):
        if self.active:
            self._record("P_out", output, self.projector_time_ms, self.projector_chunk_ids, layer=None)

    def _record(self, name, values, time_ms, chunk_ids, *, layer):
        if name in self.seen or len(values) != len(time_ms):
            raise RuntimeError(f"Unexpected acoustic capture at {name}")
        row = dict(name=name, layer=layer, tokens=len(values), channels=values.shape[-1])
        row["temporal_spectrum"] = temporal_spectrum(values, chunk_ids)
        row["temporal_spectrum"]["nominal_nyquist_hz"] = 1000 / (
            2 * (4 if name in {"P_in", "P_out"} else 2) * self.mel_step_ms)
        if self.condition is None:
            self.reference[name] = values.detach()
        else:
            row["drift"] = representation_drift(values, self.reference[name])
        self.rows.append(row)
        self.seen.add(name)
        if self.example:
            import torch
            self.arrays[f"hidden_{name}"] = values.detach().to(dtype=torch.float16).cpu().numpy()
            self.arrays[f"time_{name}"] = np.asarray(time_ms)

    def finish(self):
        self.active = False
        expected = {"C", "P_in", "P_out"}
        expected.update("E_L" if index == len(self.tower.layers) - 1 else f"E_l{index}"
                        for index in self.layers)
        if self.calls != 1 or self.seen != expected:
            raise RuntimeError(
                f"Incomplete acoustic capture: calls={self.calls}, "
                f"missing={sorted(expected - self.seen)}, extra={sorted(self.seen - expected)}")
        result = dict(
            mel_length=self.mel_length,
            encoder_tokens=len(self.encoder_time_ms),
            projector_tokens=len(self.projector_time_ms),
            mel_step_ms=self.mel_step_ms,
            encoder_nominal_step_ms=2 * self.mel_step_ms,
            chunk_lengths=np.diff(self.encoder_bounds).tolist(),
            native_pool_crossings=self.native_pool_crossings,
            stages=self.rows,
            intervention=self.intervention,
        )
        return result, self.arrays
