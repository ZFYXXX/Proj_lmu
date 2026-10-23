import numpy as np
import torch
import json

from .metrics import (
    aligned_acoustic_features,
    average_tie_ranks,
    causal_drop_indices,
    exact_chunk_topk_mask,
    observational_summary,
)


def tiny_audio_encoder():
    from transformers.models.qwen2_5_omni.configuration_qwen2_5_omni import (
        Qwen2_5OmniAudioEncoderConfig,
    )
    from transformers.models.qwen2_5_omni.modeling_qwen2_5_omni import Qwen2_5OmniAudioEncoder

    torch.manual_seed(12)
    config = Qwen2_5OmniAudioEncoderConfig(
        d_model=8,
        encoder_attention_heads=2,
        encoder_layers=3,
        encoder_ffn_dim=16,
        num_mel_bins=4,
        max_source_positions=8,
        n_window=4,
        output_dim=6,
    )
    config._attn_implementation = "eager"
    return Qwen2_5OmniAudioEncoder(config).eval()


def test_aligned_features_match_native_chunk_lengths():
    mel = torch.arange(128 * 205, dtype=torch.float32).reshape(128, 205)
    values, lengths, time_indices, boundaries = aligned_acoustic_features(mel, 205, 100)
    assert lengths == [100, 3]
    assert boundaries.tolist() == [0, 100, 103]
    assert time_indices[[0, 99, 100, 102]].tolist() == [0, 198, 200, 204]
    assert all(value.shape == (103,) for value in values.values())
    assert all(torch.isfinite(value).all() for value in values.values())


def test_average_tie_ranks_and_exact_chunk_budget():
    np.testing.assert_allclose(average_tie_ranks(np.array([1, 1, 3, 2])), [1 / 6, 1 / 6, 1, 2 / 3])
    scores = np.array([5, 4, 3, 2, 1, 1, 2, 3, 4, 5], dtype=float)
    mask = exact_chunk_topk_mask(scores, [0, 5, 10], 0.5, minimum_keep_tokens=2)
    assert np.flatnonzero(mask).tolist() == [0, 1, 8, 9]


def test_observational_summary_counts_tokens_once():
    attention = np.arange(1, 11, dtype=float)
    acoustic = {
        "intensity": attention.copy(),
        "variation": attention[::-1].copy(),
        "temporal_change": np.r_[np.arange(5), np.arange(5)],
    }
    result = observational_summary(attention, acoustic, [0, 5, 10])
    assert result["tokens"] == 10
    assert np.isclose(result["features"]["intensity"]["spearman"], 1)
    assert np.isclose(result["features"]["variation"]["spearman"], -1)
    assert sum(row["n"] for row in result["features"]["intensity"]["quintiles"]) == 10


def test_causal_sets_have_equal_budget_and_expected_pools():
    change = np.arange(20, dtype=float)
    attention = np.linspace(0, 1, 20)
    modes = [
        "causal_low_change_high_attention",
        "causal_low_change_low_attention",
        "causal_high_change_attention_matched",
        "causal_low_change_random",
    ]
    selected = {mode: causal_drop_indices(change, attention, 2, mode, seed=7) for mode in modes}
    assert all(len(indices) == 2 and len(set(indices)) == 2 for indices in selected.values())
    assert set(selected["causal_low_change_high_attention"]).issubset(range(5))
    assert set(selected["causal_low_change_low_attention"]).issubset(range(5))
    assert set(selected["causal_low_change_random"]).issubset(range(5))
    assert set(selected["causal_high_change_attention_matched"]).issubset(range(15, 20))


def test_real_encoder_observer_preserves_output():
    from .observer import SilentRedundancyObserver

    tower = tiny_audio_encoder()
    features = torch.randn(4, 13)
    lengths = torch.tensor([13])
    with torch.no_grad():
        baseline = tower(features, feature_lens=lengths, aftercnn_lens=torch.tensor([7])).last_hidden_state
        observer = SilentRedundancyObserver(
            tower, sample_rate=16000, hop_length=160, layers=(0, 1, 2), query_block=2,
        )
        observer.install()
        observer.begin(example=True)
        observed = tower(features, feature_lens=lengths, aftercnn_lens=torch.tensor([7])).last_hidden_state
        measurements, arrays = observer.finish()
        observer.close()
    assert torch.equal(baseline, observed)
    assert measurements["chunk_lengths"] == [4, 3]
    assert measurements["encoder_tokens"] == 7
    assert len(measurements["layers"]) == 3
    np.testing.assert_allclose(arrays["time_s"], [0, 0.02, 0.04, 0.06, 0.08, 0.10, 0.12])
    assert not observer.handles


def test_selector_patch_tracks_provenance_across_three_stages():
    from .selector import PruningSelector

    tower = tiny_audio_encoder()
    selector = PruningSelector(
        tower,
        selector="temporal_change",
        layers=[0, 1, 2],
        keep_ratio=0.75,
        minimum_keep_tokens=1,
        seed=42,
        sample_rate=16000,
        hop_length=160,
    )
    selector.install()
    try:
        selector.begin(0, example=True)
        features = torch.randn(4, 13)
        selector._tower_pre(tower, (features,), {"feature_lens": torch.tensor([13])})
        hidden = torch.randn(7, 8)
        lengths = [4, 3]
        for layer in tower.layers:
            scores = selector._score(layer.self_attn, hidden, lengths)
            output_lengths = [max(1, int(length * 0.75)) for length in lengths]
            hidden, _ = selector._select(hidden, scores, lengths, output_lengths)
            lengths = output_lengths
        report, arrays = selector.finish()
    finally:
        selector.close()
    assert report["initial_tokens"] == 7
    assert report["final_tokens"] == sum(lengths)
    assert [stage["layer"] for stage in report["stages"]] == [0, 1, 2]
    assert arrays["kept_after_l2"].sum() == report["final_tokens"]


def test_reports_support_small_smoke_with_missing_taxonomy_groups(tmp_path):
    from .metrics import jsonable, observational_summary
    from .report import comparison_report, observational_report

    observe = tmp_path / "observe"
    observe.mkdir()
    attention = np.arange(1, 8, dtype=float)
    acoustic = {
        "intensity": attention,
        "variation": attention[::-1],
        "temporal_change": np.array([0, 1, 2, 3, 0, 1, 2], dtype=float),
    }
    layers = [{"layer": layer, **observational_summary(attention, acoustic, [0, 4, 7])}
              for layer in (3, 6, 9)]
    base_rows = [
        {"id": "a", "doc_index": 0, "audio_path": "a.wav", "question_type": "perception",
         "sub_category": "Linguistics", "task_name": "content_grounding", "prediction": "A",
         "accuracy": 1, "mel_length": 13, "encoder_tokens": 7, "chunk_lengths": [4, 3],
         "mel_step_s": 0.01, "encoder_nominal_step_s": 0.02, "layers": layers,
         "example_file": "example.rank0.doc0.npz"},
        {"id": "b", "doc_index": 1, "audio_path": "b.wav", "question_type": "perception",
         "sub_category": "Linguistics", "task_name": "content_grounding", "prediction": "B",
         "accuracy": 0, "mel_length": 13, "encoder_tokens": 7, "chunk_lengths": [4, 3],
         "mel_step_s": 0.01, "encoder_nominal_step_s": 0.02, "layers": layers},
    ]
    (observe / "samples.rank0.jsonl").write_text(
        "\n".join(json.dumps(jsonable(row)) for row in base_rows), encoding="utf-8")
    arrays = {
        "mel": np.ones((4, 13)), "mel_step_s": np.asarray(0.01),
        "time_s": np.arange(7) * 0.02, "boundaries": np.array([0, 4, 7]),
        **{f"acoustic_{key}": value for key, value in acoustic.items()},
    }
    for layer in (3, 6, 9):
        arrays[f"attention_l{layer}"] = attention
        arrays[f"keep_l{layer}_r50"] = np.array([1, 1, 0, 0, 1, 0, 0], dtype=bool)
    np.savez_compressed(observe / "example.rank0.doc0.npz", **arrays)
    condition = tmp_path / "conditions" / "attention_keep50"
    condition.mkdir(parents=True)
    settings = {"selector": "attention", "target_ratio": 0.5,
                "keep_ratio_per_stage": 0.8, "layers": [3, 6, 9]}
    (condition / "analysis.json").write_text(json.dumps(settings), encoding="utf-8")
    condition_rows = []
    for row in base_rows:
        condition_rows.append({key: row[key] for key in (
            "id", "doc_index", "audio_path", "question_type", "sub_category", "task_name")})
        condition_rows[-1].update(prediction=row["prediction"], accuracy=row["accuracy"], target_ratio=0.5,
            pruning={"realized_encoder_ratio": 0.5, "initial_tokens": 7, "final_tokens": 4,
                     "selector": "attention", "layers": [3, 6, 9], "keep_ratio_per_stage": 0.8,
                     "stages": []})
    (condition / "samples.rank0.jsonl").write_text(
        "\n".join(json.dumps(row) for row in condition_rows), encoding="utf-8")
    observation = observational_report(tmp_path, tmp_path / "report" / "observational")
    final = comparison_report(tmp_path, tmp_path / "report" / "final", observation)
    assert observation["samples"] == 2
    assert final["conditions"]["attention_keep50"]["delta_pp"] == 0
    assert (tmp_path / "report" / "observational" / "figures" / "doc0_five_panel.png").exists()
    assert (tmp_path / "report" / "final" / "condition_accuracy.csv").exists()
