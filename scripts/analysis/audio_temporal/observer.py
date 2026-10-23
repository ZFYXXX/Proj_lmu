"""Read-only capture of C, E_l, E_L, P_in and P_out with physical time maps."""
from __future__ import annotations

import numpy as np

from .metrics import input_region_labels, lag_similarity, map_labels, region_similarity
from .provenance import initialize_traces, trace_summary


class TemporalObserver:
    """One-request observer for the verified Qwen2.5-Omni audio tower."""

    def __init__(self, tower, *, sample_rate, hop_length, layers, lags_ms,
                 region_lag_ms=80.0, prediction_grid=()):
        if tower.training:
            raise ValueError("Temporal observation requires model.eval()")
        self.tower = tower
        self.sample_rate = int(sample_rate)
        self.hop_length = int(hop_length)
        self.mel_step_ms = 1000.0 * self.hop_length / self.sample_rate
        self.layers = sorted(set(int(index) for index in layers))
        if not self.layers or not set(self.layers).issubset(range(len(tower.layers))):
            raise ValueError("Invalid audio encoder layer selection")
        self.lags_ms = tuple(float(value) for value in lags_ms)
        if not self.lags_ms or any(value <= 0 for value in self.lags_ms):
            raise ValueError("Physical lags must be positive")
        self.region_lag_ms = float(region_lag_ms)
        self.prediction_grid = tuple(dict(item) for item in prediction_grid)
        self.handles = []
        self.active = False

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

    def begin(self, *, example=False, seed=42):
        if self.active:
            raise RuntimeError("Previous temporal observation was not finished")
        self.active = True
        self.example = bool(example)
        self.seed = int(seed)
        self.calls = 0
        self.rows = []
        self.arrays = {}
        self.seen = set()
        self.pool_mapping = None
        self.region_proxy = None

    def finish(self):
        self.active = False
        expected = {"C", "P_in", "P_out"}
        expected.update("E_L" if index == len(self.tower.layers) - 1 else f"E_l{index}"
                        for index in self.layers)
        if self.calls != 1 or self.seen != expected:
            raise RuntimeError(
                f"Incomplete temporal capture: calls={self.calls}, "
                f"missing={sorted(expected - self.seen)}, extra={sorted(self.seen - expected)}")
        region_counts = {name: int(np.sum(self.region_proxy["labels"] == name))
                         for name in np.unique(self.region_proxy["labels"])}
        result = {
            "mel_length": self.mel_length,
            "encoder_tokens": len(self.encoder_time_ms),
            "projector_tokens": len(self.projector_time_ms),
            "mel_step_ms": self.mel_step_ms,
            "encoder_nominal_step_ms": 2 * self.mel_step_ms,
            "chunk_lengths": np.diff(self.encoder_bounds).tolist(),
            "native_pool_crossings": self.native_pool_crossings,
            "region_proxy": {
                "definition": "input_log_mel_activity_and_spectral_flux_quantiles",
                "ground_truth_labels": False,
                "thresholds": self.region_proxy["thresholds"],
                "mel_frame_counts": region_counts,
            },
            "stages": self.rows,
            "projector_provenance": trace_summary(
                self.projector_traces,
                boundary_times_ms=self.region_proxy["hard_boundaries_ms"]),
        }
        return result, self.arrays

    def _tower_pre(self, module, args, kwargs):
        if not self.active:
            return
        self.calls += 1
        if self.calls != 1:
            raise RuntimeError("Expected exactly one packed audio tower call per MMSU request")
        features = kwargs.get("input_features", args[0] if args else None)
        lengths = kwargs.get("feature_lens", args[1] if len(args) > 1 else None)
        if features is None or lengths is None or features.ndim != 2 or lengths.numel() != 1:
            raise ValueError("Temporal analysis supports one packed audio per request")
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
        self.region_proxy = input_region_labels(
            features, self.mel_length, mel_step_ms=self.mel_step_ms,
            chunk_mel_lengths=mel_chunks)
        self.encoder_labels = map_labels(
            self.region_proxy["time_ms"], self.region_proxy["labels"], self.encoder_time_ms)
        if self.example:
            self.arrays.update(
                mel=np.asarray(
                    features[:, :self.mel_length].detach().float().cpu().tolist(), dtype=np.float32),
                mel_time_ms=self.region_proxy["time_ms"],
                mel_region=self.region_proxy["labels"],
                mel_activity=self.region_proxy["activity"],
                mel_spectral_flux=self.region_proxy["spectral_flux"],
                encoder_time_ms=self.encoder_time_ms,
                encoder_chunk_id=self.encoder_chunk_ids,
            )

    def _capture_c(self, module, args):
        del module
        if self.active:
            self._record("C", args[0], self.encoder_time_ms, self.encoder_chunk_ids,
                         self.encoder_labels, layer=None)

    def _capture_layer(self, index):
        def hook(module, args, output):
            del module, args
            if self.active:
                name = "E_L" if index == len(self.tower.layers) - 1 else f"E_l{index}"
                self._record(name, output[0], self.encoder_time_ms, self.encoder_chunk_ids,
                             self.encoder_labels, layer=index)
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
                # A native pooled token that already straddles chunks is isolated from all merges.
                chunk_ids.append(-1 - output_index)
                crossings += 1
        return groups, np.asarray(times), np.asarray(chunk_ids), crossings

    def _capture_projector_input(self, module, args):
        del module
        if not self.active:
            return
        values = args[0]
        groups, times, chunks, crossings = self._projector_layout(len(values))
        self.pool_mapping = groups
        self.projector_time_ms, self.projector_chunk_ids = times, chunks
        self.native_pool_crossings = crossings
        self.projector_labels = map_labels(
            self.region_proxy["time_ms"], self.region_proxy["labels"], times)
        self.projector_traces = initialize_traces(times, chunks)
        self._record("P_in", values, times, chunks, self.projector_labels, layer=None)
        if self.example:
            self.arrays["projector_pool_start"] = np.asarray([group[0] for group in groups])
            self.arrays["projector_pool_end"] = np.asarray([group[-1] + 1 for group in groups])
            self.arrays["projector_time_ms"] = times
            self.arrays["projector_chunk_id"] = chunks
            self.arrays["projector_region"] = self.projector_labels
            self.arrays["projector_interval_start_ms"] = np.asarray(
                [trace.intervals_ms[0][0] for trace in self.projector_traces])
            self.arrays["projector_interval_end_ms"] = np.asarray(
                [trace.intervals_ms[-1][1] for trace in self.projector_traces])

    def _capture_projector_output(self, module, args, output):
        del module, args
        if self.active:
            self._record("P_out", output, self.projector_time_ms, self.projector_chunk_ids,
                         self.projector_labels, layer=None)

    def _record(self, name, values, time_ms, chunk_ids, labels, *, layer):
        if name in self.seen:
            raise RuntimeError(f"Stage {name} ran more than once")
        if len(values) != len(time_ms):
            raise ValueError(f"Stage {name} length does not match its physical time map")
        row = {
            "name": name,
            "layer": layer,
            "tokens": len(values),
            "lag_similarity": lag_similarity(values, time_ms, chunk_ids, self.lags_ms),
            "region_similarity": region_similarity(
                values, time_ms, chunk_ids, labels, lag_ms=self.region_lag_ms),
        }
        if self.prediction_grid:
            from .algorithms import predict_blocks

            traces = initialize_traces(time_ms, chunk_ids)
            row["predictability"] = []
            for condition in self.prediction_grid:
                _, details = predict_blocks(
                    values, traces,
                    block_ms=condition["block_ms"],
                    predictor=condition["predictor"],
                    replace_ratio=condition["replace_ratio"],
                    # Keep the missing blocks identical across predictors for a
                    # given duration, and identical to the paired P_in task
                    # intervention.  Only the predictor is allowed to change.
                    seed=self.seed)
                row["predictability"].append({
                    "block_ms": condition["block_ms"],
                    "predictor": condition["predictor"],
                    "replace_ratio": condition["replace_ratio"],
                    "prediction_relative_l2": details["prediction_relative_l2"],
                    "prediction_cosine_mean": details["prediction_cosine_mean"],
                    "predicted_tokens": details["predicted_tokens"],
                    "effective_block_tokens": details["effective_block_tokens"],
                })
        self.rows.append(row)
        self.seen.add(name)
        if self.example:
            self.arrays[f"hidden_{name}"] = np.asarray(
                values.detach().float().cpu().tolist(), dtype=np.float16)
