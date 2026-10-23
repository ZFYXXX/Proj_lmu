"""Trace native LTBM projector merges back to uncompressed P_in time intervals."""
from __future__ import annotations

import numpy as np

from .observer import TemporalObserver
from .provenance import replay_ltbm_rounds, trace_summary


class LTBMProvenanceObserver(TemporalObserver):
    """Capture the original projector input before the installed LTBM hook."""

    def install(self):
        if self.handles:
            raise RuntimeError("Observer already installed")
        self.handles.append(self.tower.register_forward_pre_hook(
            self._tower_pre, with_kwargs=True))
        self.handles.append(self.tower.proj.register_forward_pre_hook(
            self._capture_projector_input, prepend=True))
        self.handles.append(self.tower.proj.register_forward_hook(self._capture_output))

    def begin(self, *, example=False, seed=42):
        super().begin(example=example, seed=seed)
        self.tower_invocations = 0
        self.compressed_projector_tokens = None

    def _tower_pre(self, module, args, kwargs):
        # The native compressor can make a preflight audio-tower pass before
        # its request-local merge. The final pass is the generated answer's.
        self.tower_invocations += 1
        self.calls = 0
        self.seen.clear()
        self.rows.clear()
        self.arrays.clear()
        super()._tower_pre(module, args, kwargs)

    def _capture_output(self, module, args, output):
        del module, args
        if self.active:
            self.compressed_projector_tokens = len(output)

    def finish(self, report):
        self.active = False
        if self.tower_invocations < 1 or self.seen != {"P_in"}:
            raise RuntimeError("Missing original P_in capture for the LTBM request")
        if not isinstance(report, dict) or report.get("plugin") != "local_temporal_bipartite_merging_audio":
            raise RuntimeError("Native LTBM compression report is missing")
        items = report.get("items", ())
        if len(items) != 1:
            raise RuntimeError("MMSU temporal provenance requires exactly one audio unit")
        item = items[0]
        original, kept = int(item["original_tokens"]), int(item["kept_tokens"])
        if original != len(self.projector_traces) or kept != self.compressed_projector_tokens:
            raise RuntimeError("LTBM report does not match captured projector lengths")
        merge_rounds = item.get("merge_rounds", ())
        output_traces = replay_ltbm_rounds(
            self.projector_traces, merge_rounds, kept_tokens=kept)
        boundaries = self.region_proxy["hard_boundaries_ms"]
        trace_rows = [trace.as_dict() for trace in output_traces]
        for row in trace_rows:
            intervals = row["intervals_ms"]
            row["crosses_input_boundary"] = bool(intervals and any(
                intervals[0][0] < boundary < intervals[-1][1]
                for boundary in boundaries))
        if self.example:
            self.arrays["ltbm_output_center_ms"] = np.asarray(
                [trace.center_ms for trace in output_traces])
        result = {
            "mel_length": self.mel_length,
            "mel_step_ms": self.mel_step_ms,
            "encoder_tokens": len(self.encoder_time_ms),
            "original_projector_tokens": original,
            "projector_tokens": kept,
            "native_pool_crossings": self.native_pool_crossings,
            "tower_invocations": self.tower_invocations,
            "stages": self.rows,
            "projector_provenance": trace_summary(self.projector_traces),
            "ltbm_provenance": trace_summary(output_traces, boundary_times_ms=boundaries),
            "ltbm_token_provenance": trace_rows,
            "ltbm_merge_rounds": merge_rounds,
            "ltbm_requested_keep_ratio": report.get("requested_keep_ratio"),
            "ltbm_achieved_keep_ratio": report.get("achieved_keep_ratio"),
        }
        return result, self.arrays
