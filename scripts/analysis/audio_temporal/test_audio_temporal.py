from __future__ import annotations

from dataclasses import dataclass

import pytest


def test_removed_experiment_two_is_not_addressable():
    from .plan import EXPERIMENTS, build_conditions

    assert "2" not in EXPERIMENTS
    assert "local_far_similarity" not in EXPERIMENTS
    with pytest.raises(ValueError):
        build_conditions("2")


def test_frozen_condition_families_are_distinct():
    from .plan import build_conditions

    assert build_conditions("similarity") == []
    assert {row["family"] for row in build_conditions("regions")} == {"region_substitution"}
    assert {row["family"] for row in build_conditions("predictability")} == {"predictability"}
    assert {row["family"] for row in build_conditions("substitution")} == {"substitution"}
    assert {row["family"] for row in build_conditions("locality")} == {"locality"}
    with pytest.raises(ValueError):
        build_conditions("compression")


def test_provenance_preserves_disconnected_support_and_rejects_chunk_crossing():
    np = pytest.importorskip("numpy")
    from .provenance import initialize_traces, merge_traces

    traces = initialize_traces(np.array([0.0, 10.0, 20.0, 30.0]), np.array([0, 0, 0, 1]))
    merged = merge_traces([traces[0], traces[2]])
    assert merged.as_dict()["fragments"] == 2
    assert merged.as_dict()["max_gap_ms"] > 0
    with pytest.raises(ValueError):
        merge_traces([traces[2], traces[3]])


def test_actual_ltbm_rounds_reconstruct_every_original_token_once():
    torch = pytest.importorskip("torch")
    np = pytest.importorskip("numpy")
    from mllm_comp.compressors.projector_level.local_temporal_bipartite_merging import (
        local_temporal_bipartite_merge)
    from .provenance import initialize_traces, replay_ltbm_rounds, trace_summary

    torch.manual_seed(4)
    values = torch.randn(12, 6)
    traces = initialize_traces(np.arange(12) * 80.0, np.array([0] * 6 + [1] * 6))
    output, details = local_temporal_bipartite_merge(values, keep_count=4, window_size=3)
    output_traces = replay_ltbm_rounds(traces, details["merge_rounds"], kept_tokens=4)
    assert len(output) == len(output_traces) == 4
    assert sorted(index for trace in output_traces for index in trace.source_indices) == list(range(12))
    assert trace_summary(output_traces)["tokens"] == 4
    with pytest.raises(ValueError):
        replay_ltbm_rounds(traces, details["merge_rounds"], kept_tokens=5)


def test_ltbm_round_can_merge_multiple_sources_into_one_destination():
    np = pytest.importorskip("numpy")
    from .provenance import initialize_traces, replay_ltbm_rounds

    traces = initialize_traces(np.arange(4) * 40.0, np.array([0, 0, 1, 1]))
    rounds = [{"round_index": 0, "input_tokens": 4, "output_tokens": 2,
               "merge_count": 2, "merge_pairs": [[0, 3], [2, 3]]}]
    output = replay_ltbm_rounds(traces, rounds, kept_tokens=2)
    assert [trace.source_indices for trace in output] == [(1,), (0, 2, 3)]
    assert output[1].hard_chunks == (0, 1)
    assert output[1].as_dict()["fragments"] >= 2


def test_locality_has_shared_realized_budget():
    torch = pytest.importorskip("torch")
    np = pytest.importorskip("numpy")
    from .algorithms import locality_merge
    from .provenance import initialize_traces

    torch.manual_seed(2)
    values = torch.randn(20, 8)
    traces = initialize_traces(np.arange(20) * 100.0, np.zeros(20, dtype=int))
    local = locality_merge(values, traces, scope="local", merge_fraction=0.2,
                           window_ms=300.0, far_ms=700.0)
    global_result = locality_merge(values, traces, scope="global", merge_fraction=0.2,
                                   window_ms=300.0, far_ms=700.0)
    assert len(local[0]) == len(global_result[0])
    assert local[2]["similarity_matched_common_budget"] == global_result[2][
        "similarity_matched_common_budget"]


def test_substitution_controls_share_actual_replacement_budget():
    torch = pytest.importorskip("torch")
    np = pytest.importorskip("numpy")
    from .algorithms import substitute_tokens
    from .provenance import initialize_traces

    torch.manual_seed(7)
    values = torch.randn(30, 8)
    traces = initialize_traces(np.arange(30) * 100.0, np.zeros(30, dtype=int))
    counts = []
    for strategy in ("local_high", "far_high", "random_local", "local_low"):
        _, details = substitute_tokens(
            values, traces, strategy=strategy, replace_ratio=0.2,
            window_ms=300.0, far_ms=900.0, seed=5)
        counts.append(details["replaced_tokens"])
    assert len(set(counts)) == 1


def test_predictors_use_the_same_missing_blocks_for_a_fixed_seed():
    torch = pytest.importorskip("torch")
    np = pytest.importorskip("numpy")
    from .algorithms import predict_blocks
    from .provenance import initialize_traces

    torch.manual_seed(11)
    values = torch.randn(40, 8)
    traces = initialize_traces(np.arange(40) * 80.0, np.zeros(40, dtype=int))
    selected = []
    for predictor in ("copy_left", "linear", "local_mean"):
        _, details = predict_blocks(
            values, traces, block_ms=160.0, predictor=predictor,
            replace_ratio=0.2, seed=17)
        selected.append(details["blocks"])
    assert selected[0] == selected[1] == selected[2]


def test_condition_execution_order_is_cyclic_and_membership_preserving():
    from .worker import condition_execution_order

    conditions = [{"id": value} for value in ("a", "b", "c")]
    assert [row["id"] for row in condition_execution_order(conditions, 0)] == ["a", "b", "c"]
    assert [row["id"] for row in condition_execution_order(conditions, 1)] == ["b", "c", "a"]
    assert [row["id"] for row in condition_execution_order(conditions, 5)] == ["c", "a", "b"]
    assert [row["id"] for row in conditions] == ["a", "b", "c"]


def test_baseline_provenance_weights_each_audio_once():
    from .report import baseline_provenance_table

    def record(audio, tokens):
        return {
            "audio_path": audio,
            "doc_index": 0,
            "projector_provenance": {"tokens": tokens},
        }

    rows = baseline_provenance_table([
        record("repeated.wav", 10), record("repeated.wav", 10), record("single.wav", 20)])
    tokens = next(row for row in rows if row["metric"] == "tokens")
    assert tokens["unique_audio"] == 2
    assert tokens["mean"] == pytest.approx(15.0)


def test_observer_captures_all_declared_representation_stages():
    torch = pytest.importorskip("torch")
    from .observer import TemporalObserver

    class Layer(torch.nn.Module):
        def forward(self, hidden, **kwargs):
            del kwargs
            return (hidden + 0.1,)

    class Tower(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.n_window = 8
            self.layers = torch.nn.ModuleList([Layer(), Layer()])
            self.proj = torch.nn.Linear(4, 3, bias=False)

        def register_forward_pre_hook(self, hook, with_kwargs=False):
            if with_kwargs:  # Compatibility shim for this test's old CPU PyTorch.
                return super().register_forward_pre_hook(lambda module, args: hook(module, args, {}))
            return super().register_forward_pre_hook(hook)

        def forward(self, input_features, feature_lens=None):
            encoder_tokens = (int(feature_lens[0]) + 1) // 2
            hidden = torch.arange(encoder_tokens * 4, dtype=torch.float32).reshape(
                encoder_tokens, 4)
            for layer in self.layers:
                hidden = layer(hidden)[0]
            pooled = hidden[:encoder_tokens // 2 * 2].reshape(-1, 2, 4).mean(1)
            return self.proj(pooled)

    tower = Tower().eval()
    observer = TemporalObserver(
        tower, sample_rate=16000, hop_length=160, layers=[0, 1], lags_ms=[20, 40])
    observer.install()
    try:
        observer.begin(example=True, seed=1)
        tower(torch.randn(8, 16), torch.tensor([16]))
        measurements, arrays = observer.finish()
    finally:
        observer.close()
    assert {"hidden_C", "hidden_E_l0", "hidden_E_L", "hidden_P_in", "hidden_P_out"} <= set(arrays)
    assert [stage["name"] for stage in measurements["stages"]] == [
        "C", "E_l0", "E_L", "P_in", "P_out"]


def test_ltbm_observer_captures_pre_merge_input_after_preflight():
    torch = pytest.importorskip("torch")
    from mllm_comp.compressors.projector_level.local_temporal_bipartite_merging import (
        local_temporal_bipartite_merge)
    from .ltbm_observer import LTBMProvenanceObserver

    class Projector(torch.nn.Identity):
        def register_forward_pre_hook(self, hook, prepend=False):
            handle = super().register_forward_pre_hook(hook)
            if prepend:  # Emulate modern PyTorch's ordering in this old CPU runtime.
                self._forward_pre_hooks.move_to_end(handle.id, last=False)
            return handle

    class Tower(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.n_window = 8
            self.layers = torch.nn.ModuleList([torch.nn.Identity()])
            self.proj = Projector()
            self.marker = torch.nn.Parameter(torch.zeros(()))

        def register_forward_pre_hook(self, hook, with_kwargs=False):
            if with_kwargs:  # Local CPU test runtime predates this PyTorch hook option.
                return super().register_forward_pre_hook(
                    lambda module, args: hook(module, args, {}))
            return super().register_forward_pre_hook(hook)

        def forward(self, input_features, feature_lens=None):
            encoder_tokens = (int(feature_lens[0]) + 1) // 2
            return self.proj(torch.arange(
                (encoder_tokens // 2) * 4, dtype=torch.float32).reshape(-1, 4))

    tower = Tower().eval()
    active = {"merge": False, "details": None}

    def merge_hook(module, args):
        del module
        if not active["merge"]:
            return None
        output, active["details"] = local_temporal_bipartite_merge(
            args[0], keep_count=2, window_size=2)
        return (output,)

    native_hook = tower.proj.register_forward_pre_hook(merge_hook)
    observer = LTBMProvenanceObserver(
        tower, sample_rate=16000, hop_length=160, layers=[0], lags_ms=[40])
    observer.install()
    features, lengths = torch.randn(8, 16), torch.tensor([16])
    try:
        observer.begin(example=True, seed=1)
        tower(features, lengths)  # Native LTBM preflight
        active["merge"] = True
        merged = tower(features, lengths)
        report = {
            "plugin": "local_temporal_bipartite_merging_audio",
            "items": [{"original_tokens": 4, "kept_tokens": 2,
                       "merge_rounds": active["details"]["merge_rounds"]}],
            "requested_keep_ratio": 0.5,
            "achieved_keep_ratio": 0.5,
        }
        measurements, arrays = observer.finish(report)
    finally:
        observer.close()
        native_hook.remove()
    assert len(merged) == measurements["projector_tokens"] == 2
    assert measurements["original_projector_tokens"] == 4
    assert measurements["tower_invocations"] == 2
    assert sorted(index for item in measurements["ltbm_token_provenance"]
                  for index in item["source_indices"]) == list(range(4))
    assert "mel" in arrays


def test_ltbm_report_weights_repeated_audio_once(tmp_path):
    pytest.importorskip("matplotlib")
    from .report import PROVENANCE_METRICS, ltbm_provenance_report

    def record(audio, tokens):
        return {
            "audio_path": audio, "doc_index": 0,
            "original_projector_tokens": tokens * 2,
            "projector_tokens": tokens,
            "ltbm_achieved_keep_ratio": 0.5,
            "ltbm_provenance": {metric: tokens for metric in PROVENANCE_METRICS},
            "ltbm_token_provenance": [{"nominal_span_ms": 40.0,
                                       "max_gap_ms": 0.0, "fragments": 1}],
        }

    rows, figures = ltbm_provenance_report(
        [record("a.wav", 2), record("a.wav", 2), record("b.wav", 4)],
        tmp_path, tmp_path)
    token_row = next(row for row in rows if row["metric"] == "tokens")
    assert token_row["unique_audio"] == 2
    assert token_row["mean"] == pytest.approx(3.0)
    assert (tmp_path / "ltbm_per_audio.csv").exists()
    assert "ltbm_provenance_distributions.png" in figures


@dataclass
class _Plan:
    argv: list[str]
    environment: dict[str, str]
    cwd: str = "."


def test_worker_plan_replaces_only_lmms_eval(tmp_path):
    from .launch import worker_plan

    original = _Plan(["python", "-m", "lmms_eval", "--tasks", "mmsufinal"], {})
    changed = worker_plan(original, tmp_path / "analysis.json")
    assert changed.argv[2] == "scripts.analysis.audio_temporal.worker"
    assert changed.environment["AUDIO_TEMPORAL_CONFIG"].endswith("analysis.json")
