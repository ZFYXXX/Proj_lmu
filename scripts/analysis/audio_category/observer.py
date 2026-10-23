"""Read-only native audio-token capture with convolution/pooling time coordinates."""
from __future__ import annotations
import numpy as np
from .data import stage_names

def pool_tokens(values, times, ratio):
    """Contiguous token mean pooling; return fewer tokens and their physical centers."""
    if not 0 < ratio <= 1:
        raise ValueError("keep ratio must be in (0, 1]")
    if ratio == 1:
        return values, times
    count = max(1, int(np.ceil(len(values) * ratio)))
    groups = np.array_split(np.arange(len(values)), count)
    return (np.stack([values[g].mean(0) for g in groups]),
            np.asarray([times[g].mean() for g in groups]))

def interpolate(values, times, query):
    from scipy.interpolate import interp1d
    if len(times) == 1:
        return np.repeat(values[:1], len(query), axis=0)
    return interp1d(times, values, axis=0, bounds_error=False,
                    fill_value=(values[0], values[-1]), assume_sorted=True)(query).astype(np.float32)

class CategoryObserver:
    def __init__(self, tower, layers, *, sample_rate, hop_length):
        if tower.training:
            raise ValueError("Feature extraction requires eval mode")
        self.tower, self.layers = tower, list(layers)
        if not self.layers or not set(self.layers).issubset(range(len(tower.layers))):
            raise ValueError("Invalid encoder layers")
        self.step = hop_length / sample_rate
        self.handles, self.active = [], False

    def install(self):
        self.handles.append(self.tower.register_forward_pre_hook(self._start, with_kwargs=True))
        self.handles.append(self.tower.layers[0].register_forward_pre_hook(self._input))
        for index in self.layers:
            self.handles.append(self.tower.layers[index].register_forward_hook(self._layer(index)))
        self.handles.append(self.tower.proj.register_forward_pre_hook(self._projector_input))
        self.handles.append(self.tower.proj.register_forward_hook(self._projector_output))

    def close(self):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()
        self.active = False

    def begin(self):
        if self.active:
            raise RuntimeError("Previous capture not finished")
        self.active, self.calls, self.features = True, 0, {}

    def finish(self):
        self.active = False
        expected = {f"{s}__{suffix}" for s in stage_names(self.layers) for suffix in ("x", "t", "clip")}
        if self.calls != 1 or set(self.features) != expected:
            raise RuntimeError("Incomplete audio feature capture")
        return self.features

    def _start(self, module, args, kwargs):
        if not self.active:
            return
        self.calls += 1
        if self.calls != 1:
            raise RuntimeError("Expected one audio tower prefill per request")
        mel = kwargs.get("input_features", args[0] if args else None)
        lengths = kwargs.get("feature_lens", args[1] if len(args) > 1 else None)
        if mel is None or lengths is None or mel.ndim != 2 or lengths.numel() != 1:
            raise ValueError("Expected one packed audio")
        length = int(lengths[0])
        if length != mel.shape[1]:
            raise ValueError("Packed Mel length mismatch")
        chunk = 2 * module.n_window
        self.encoder_times = np.concatenate([
            (start + 2 * np.arange((min(chunk, length-start) + 1)//2)) * self.step
            for start in range(0, length, chunk)])
        # Verified native AvgPool1d has kernel=stride=2, floor output, no padding.
        pool = module.avg_pooler
        if pool.kernel_size != (2,) and pool.kernel_size != 2:
            raise ValueError("Unsupported native audio pooling kernel")
        if pool.stride not in (2, (2,)) or pool.padding not in (0, (0,)) or pool.ceil_mode:
            raise ValueError("Unsupported native audio pooling geometry")
        n = len(self.encoder_times) // 2 * 2
        self.projector_times = self.encoder_times[:n].reshape(-1, 2).mean(1)

    def _record(self, name, values, times):
        if values.ndim != 2 or len(values) != len(times) or f"{name}__x" in self.features:
            raise ValueError(f"Token/time mismatch at {name}")
        array = values.detach().float().cpu().numpy()
        self.features[f"{name}__x"] = array.astype(np.float16)
        self.features[f"{name}__t"] = times.astype(np.float64)
        self.features[f"{name}__clip"] = array.mean(0)

    def _input(self, module, args):
        if self.active:
            self._record("C", args[0], self.encoder_times)

    def _layer(self, index):
        def hook(module, args, output):
            if self.active:
                self._record(f"E_{index:03d}", output[0], self.encoder_times)
        return hook

    def _projector_input(self, module, args):
        if self.active:
            self._record("P_in", args[0], self.projector_times)

    def _projector_output(self, module, args, output):
        if self.active:
            self._record("P_out", output, self.projector_times)

