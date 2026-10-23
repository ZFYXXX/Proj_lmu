"""Request-local temporal interventions at Qwen2.5-Omni P_in."""
from __future__ import annotations

from contextlib import contextmanager

import numpy as np

from .algorithms import (
    locality_merge,
    predict_blocks,
    substitute_region,
    substitute_tokens,
)
from .metrics import input_region_labels, map_labels, relative_l2
from .provenance import chronological_anchor_indices, initialize_traces, trace_summary


class TemporalController:
    """Apply one frozen condition without altering the platform or checkpoint."""

    def __init__(self, tower, *, sample_rate, hop_length):
        self.tower = tower
        self.mel_step_ms = 1000.0 * float(hop_length) / float(sample_rate)
        self.handles = []
        self.condition = None
        self.reference_hidden = None

    def install(self):
        if self.handles:
            raise RuntimeError("Controller already installed")
        self.handles.append(self.tower.register_forward_pre_hook(self._tower_pre, with_kwargs=True))
        self.handles.append(self.tower.proj.register_forward_pre_hook(self._projector_pre))

    def close(self):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()
        self.reference_hidden = None

    def begin(self, condition=None, *, seed=42, retain_trace=False):
        self.condition = condition
        self.seed = int(seed)
        self.retain_trace = bool(retain_trace)
        self.calls = 0
        self.projector_calls = 0
        self.keep_indices = None
        self.last_hidden = None
        self.output_traces = None
        self.details = {}
        self.current_region_proxy = None
        if condition is None:
            self.reference_hidden = None
            self.reference_traces = None
            self.reference_labels = None
            self.prepared_hidden = None
        else:
            if self.reference_hidden is None:
                raise RuntimeError("Run the native reference before any temporal condition")
            self._prepare_condition()

    def finish(self):
        if self.calls != 1 or self.projector_calls != 1 or self.last_hidden is None:
            raise RuntimeError(
                f"Incomplete temporal intervention: audio_calls={self.calls}, "
                f"projector_calls={self.projector_calls}")
        return self.details

    def _tower_pre(self, module, args, kwargs):
        self.calls += 1
        if self.calls != 1:
            raise RuntimeError("Expected one packed audio call per MMSU request")
        features = kwargs.get("input_features", args[0] if args else None)
        lengths = kwargs.get("feature_lens", args[1] if len(args) > 1 else None)
        if features is None or lengths is None or lengths.numel() != 1:
            raise ValueError("Temporal interventions require one packed audio")
        mel_length = int(lengths[0])
        mel_chunks = [min(2 * module.n_window, mel_length - start)
                      for start in range(0, mel_length, 2 * module.n_window)]
        encoder_lengths = [(length + 1) // 2 for length in mel_chunks]
        starts = np.r_[0, np.cumsum(mel_chunks[:-1])]
        self.current_encoder_time = np.concatenate([
            (starts[chunk] + 2 * np.arange(length)) * self.mel_step_ms
            for chunk, length in enumerate(encoder_lengths)])
        self.current_encoder_chunks = np.concatenate([
            np.full(length, chunk, dtype=int) for chunk, length in enumerate(encoder_lengths)])
        if self.condition is None:
            self.current_region_proxy = input_region_labels(
                features, mel_length, mel_step_ms=self.mel_step_ms,
                chunk_mel_lengths=mel_chunks)

    def _layout(self, output_length):
        n = len(self.current_encoder_time)
        if output_length in {n // 2, (n + 1) // 2}:
            groups = [np.arange(2 * index, min(n, 2 * index + 2))
                      for index in range(output_length)]
        else:
            groups = [group for group in np.array_split(np.arange(n), output_length) if len(group)]
        if len(groups) != output_length:
            raise ValueError("Could not map P_in tokens to encoder time")
        times, chunks = [], []
        crossings = 0
        for output_index, group in enumerate(groups):
            times.append(float(np.mean(self.current_encoder_time[group])))
            source_chunks = np.unique(self.current_encoder_chunks[group])
            if len(source_chunks) == 1:
                chunks.append(int(source_chunks[0]))
            else:
                chunks.append(-1 - output_index)
                crossings += 1
        if self.current_region_proxy is None:
            labels = np.asarray(self.reference_labels)
            if len(labels) != len(times):
                raise ValueError("Condition P_in length differs from baseline region mapping")
        else:
            labels = map_labels(
                self.current_region_proxy["time_ms"], self.current_region_proxy["labels"], times)
        return np.asarray(times), np.asarray(chunks), labels, crossings

    def _projector_pre(self, module, args):
        del module
        self.projector_calls += 1
        if self.projector_calls != 1:
            raise RuntimeError("Expected one P_in tensor per request")
        hidden = args[0]
        times, chunks, labels, crossings = self._layout(len(hidden))
        traces = initialize_traces(times, chunks)
        if self.condition is None:
            self.reference_hidden = hidden.detach()
            self.reference_time_ms = times
            self.reference_chunk_ids = chunks
            self.reference_labels = labels
            self.reference_traces = traces
            self.last_hidden = hidden
            self.output_traces = traces
            self.details = {
                "intervention_space": "P_in_after_native_pool_and_ln_post",
                "original_projector_tokens": len(hidden),
                "projector_tokens": len(hidden),
                "native_pool_crossings": crossings,
            }
            return
        if len(hidden) != len(self.reference_hidden) or not np.array_equal(chunks, self.reference_chunk_ids):
            raise ValueError("Condition input layout differs from its native reference")
        changed = self.prepared_hidden
        self.details.update(
            native_replay_relative_l2=relative_l2(hidden, self.reference_hidden),
            native_pool_crossings=crossings,
        )
        self.last_hidden = changed
        return (changed, *args[1:])

    def _prepare_condition(self):
        """Freeze the mutation before generation so placeholder count is known."""
        condition = self.condition
        family = condition["family"]
        hidden = self.reference_hidden
        traces = self.reference_traces
        labels = self.reference_labels
        if family == "region_substitution":
            changed, details = substitute_region(
                hidden, traces, labels, region=condition["region"],
                replace_ratio=condition["replace_ratio"])
            output_traces = traces
        elif family == "predictability":
            changed, details = predict_blocks(
                hidden, traces, block_ms=condition["block_ms"],
                predictor=condition["predictor"], replace_ratio=condition["replace_ratio"],
                seed=self.seed)
            output_traces = traces
        elif family == "substitution":
            changed, details = substitute_tokens(
                hidden, traces, strategy=condition["strategy"],
                replace_ratio=condition["replace_ratio"], window_ms=condition["window_ms"],
                far_ms=condition["far_ms"], seed=self.seed)
            output_traces = traces
        elif family == "locality":
            changed, output_traces, details = locality_merge(
                hidden, traces, scope=condition["scope"],
                merge_fraction=condition["merge_fraction"],
                window_ms=condition["window_ms"], far_ms=condition["far_ms"])
        else:
            raise ValueError(f"Unsupported temporal condition family {family!r}")

        if not self.retain_trace:
            for verbose_field in ("replacement_pairs", "blocks", "uniform_groups", "merge_pairs"):
                details.pop(verbose_field, None)

        length_changed = len(changed) != len(hidden)
        if length_changed:
            self.keep_indices = chronological_anchor_indices(output_traces)
            if len(np.unique(self.keep_indices)) != len(self.keep_indices):
                raise RuntimeError("Merged output tokens must have unique native anchors")
        reference_mean = hidden.detach().float().mean(0)
        changed_mean = changed.detach().float().mean(0)
        pooled_error = float((changed_mean - reference_mean).norm()
                             / reference_mean.norm().clamp_min(1e-12))
        boundary_times = self.reference_time_ms[self.reference_labels == "boundary"]
        provenance = trace_summary(output_traces, boundary_times_ms=boundary_times)
        self.details = {
            **details,
            "condition_id": condition["id"],
            "family": family,
            "intervention_space": "P_in_after_native_pool_and_ln_post",
            "original_projector_tokens": len(hidden),
            "projector_tokens": len(changed),
            "length_changed": length_changed,
            "pooled_representation_relative_l2": pooled_error,
            "provenance": provenance,
        }
        if self.retain_trace:
            trace_rows = [trace.as_dict() for trace in output_traces]
            for trace_row in trace_rows:
                intervals = trace_row["intervals_ms"]
                trace_row["crosses_input_boundary"] = bool(intervals and any(
                    intervals[0][0] < boundary < intervals[-1][1]
                    for boundary in boundary_times))
            self.details["token_provenance"] = trace_rows
        self.prepared_hidden = changed
        self.output_traces = output_traces


@contextmanager
def compact_audio_request(bridge, controller):
    """Compact audio placeholders through the platform's existing legalizer."""
    if controller.keep_indices is None:
        yield
        return
    from mllm_comp.backbones.qwen25_omni.audio_compression import (
        legalize_qwen25_omni_audio_tokens,
    )
    from mllm_comp.core.actions import LegalizationPolicy, TokenAction

    tower = controller.tower
    thinker = bridge.runtime.model.thinker
    runner = bridge.facade.runner
    original_lengths = tower._get_feat_extract_output_lengths
    original_rope = thinker.get_rope_index
    original_generate = runner.generate
    count = len(controller.keep_indices)

    def lengths(raw):
        after_cnn, output = original_lengths(raw)
        if output.numel() != 1 or int(output[0]) != len(controller.reference_hidden):
            raise ValueError("Native audio output length differs from the frozen reference")
        return after_cnn, output.new_tensor([count])

    def rope(*args, **kwargs):
        positional = list(args)
        audio = positional[5] if len(positional) > 5 else kwargs.get("audio_seqlens")
        if audio is not None:
            compact = audio.new_tensor([4 * count])
            if len(positional) > 5:
                positional[5] = compact
            else:
                kwargs["audio_seqlens"] = compact
        return original_rope(*positional, **kwargs)

    def generate(runtime, state, *, processed_batch, **kwargs):
        spans = [span for span in state.modality_token_spans if span.modality == "audio"]
        if len(spans) != 1:
            raise ValueError("MMSU temporal analysis expects one audio placeholder span")
        span = spans[0]
        action = TokenAction(
            op="keep",
            target_indices=controller.keep_indices.tolist(),
            metadata={
                "sample_index": span.sample_index,
                "span_start": span.start,
                "span_end": span.end,
                "keep_count": count,
                "original_count": span.end - span.start,
                "audio_unit_index": 0,
                "selector": "audio_temporal_analysis",
                "target_site": "audio_encoder_output_pre_projector",
            },
        )
        compact = legalize_qwen25_omni_audio_tokens(
            state,
            token_actions=[action],
            policy=LegalizationPolicy(
                position_policy="compact",
                mask_policy="rebuild_from_tokens",
                label_policy="rebuild_ignore_index"),
            stage_name="audio_temporal_analysis")
        return original_generate(runtime, compact, processed_batch=processed_batch, **kwargs)

    tower._get_feat_extract_output_lengths = lengths
    thinker.get_rope_index = rope
    runner.generate = generate
    try:
        yield
    finally:
        tower._get_feat_extract_output_lengths = original_lengths
        thinker.get_rope_index = original_rope
        runner.generate = original_generate
