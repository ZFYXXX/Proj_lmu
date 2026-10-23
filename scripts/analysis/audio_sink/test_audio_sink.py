"""CPU checks with analytical attention and the real HF audio encoder class."""
from __future__ import annotations

import json
import sys

import numpy as np
import pytest
import torch

from scripts.analysis.audio_sink.metrics import (
    adjacent_cosines, attention_summary, jsonable, virtual_selection,
)
from scripts.analysis.audio_sink.observer import AudioEncoderObserver
from scripts.analysis.audio_sink.report import bootstrap_mean, coverage_report, compare_baseline


def test_uniform_ragged_attention_and_ties():
    q = torch.zeros(7, 2, 3)
    k = torch.zeros_like(q)
    v = torch.arange(42, dtype=torch.float32).reshape(7, 2, 3)
    bounds = [0, 3, 7]
    native = torch.cat([v[a:b].mean(0, keepdim=True).expand(b-a, -1, -1) for a, b in zip(bounds, bounds[1:])])
    r, stats, checks, matrix = attention_summary(q, k, v, native, bounds, 1, query_block=2, check_av=True, save_chunk=1)
    np.testing.assert_allclose(r, 1, atol=1e-6)
    np.testing.assert_allclose(stats['incoming_entropy'], 1, atol=1e-6)
    np.testing.assert_allclose(stats['query_entropy'], 1, atol=1e-6)
    np.testing.assert_allclose(stats['peak_start_weight'], (1/3+1/4)/2, atol=1e-6)
    np.testing.assert_allclose(stats['peak_position_hist'], np.broadcast_to(stats['uniform_peak_position_hist'], (2, 10)), atol=1e-6)
    np.testing.assert_allclose(stats['top10pct_mass'], stats['uniform_top10pct_mass'], atol=1e-6)
    np.testing.assert_allclose(stats['first_chunk_start_r'], 1, atol=1e-6)
    np.testing.assert_allclose(stats['later_chunks_start_r'], 1, atol=1e-6)
    assert checks['av_relative_l2'] < 1e-6
    np.testing.assert_allclose(matrix, 0.25)


def test_diagonal_is_not_a_column_sink():
    q = torch.eye(5).unsqueeze(1)
    k = q * 100
    v = q.clone()
    r, stats, _, _ = attention_summary(q, k, v, v, [0, 5], 1)
    np.testing.assert_allclose(r, 1, atol=1e-6)
    assert stats['incoming_entropy'][0] > .999
    assert stats['query_entropy'][0] < 1e-5
    q = torch.ones(5, 1, 1)
    k = torch.zeros_like(q)
    k[3] = 100
    r, stats, _, _ = attention_summary(q, k, q, q, [0, 5], 1)
    assert r[0, 3] > 4.999
    assert stats['incoming_entropy'][0] < 1e-5
    assert stats['peak_query_above_uniform'][0] == 1


def test_adjacent_pairs_do_not_cross_chunks():
    x = torch.tensor([[1., 0], [1., 0], [0., 1], [0., 1], [0., 1]])
    values = adjacent_cosines(x, [0, 2, 5])
    np.testing.assert_allclose(values, [1, np.nan, 1, 1, np.nan], equal_nan=True)
    assert np.isnan(adjacent_cosines(x, [0, 2, 5], lag=4)).all()


def test_virtual_candidates_do_not_modify_scores_and_use_equal_budgets():
    scores = np.arange(100, dtype=float)
    before = scores.copy()
    stats, top, segment = virtual_selection(scores, keep_ratio=.1, segments=10)
    np.testing.assert_array_equal(scores, before)
    assert len(top) == len(segment) == stats['budget'] == 10
    assert stats['topk']['segment_coverage'] == .1
    assert stats['segmentwise']['segment_coverage'] == 1
    flat, _, _ = virtual_selection(np.ones(100))
    assert flat['cutoff_tie_count'] == 100
    assert flat['cutoff_slots'] == 10


def tiny_audio_encoder():
    from transformers.models.qwen2_5_omni.configuration_qwen2_5_omni import Qwen2_5OmniAudioEncoderConfig
    from transformers.models.qwen2_5_omni.modeling_qwen2_5_omni import Qwen2_5OmniAudioEncoder
    torch.manual_seed(12)
    config = Qwen2_5OmniAudioEncoderConfig(d_model=8, encoder_attention_heads=2,
        encoder_layers=3, encoder_ffn_dim=16, num_mel_bins=4, max_source_positions=8,
        n_window=4, output_dim=6)
    config._attn_implementation = 'eager'
    return Qwen2_5OmniAudioEncoder(config).eval()


def test_real_hf_encoder_hooks_preserve_output_and_map_time():
    tower = tiny_audio_encoder()
    features = torch.randn(4, 13)
    lengths = torch.tensor([13])
    with torch.no_grad():
        baseline = tower(features, feature_lens=lengths, aftercnn_lens=torch.tensor([7])).last_hidden_state
        observer = AudioEncoderObserver(tower, sample_rate=16000, hop_length=160, query_block=2)
        observer.install()
        observer.begin(example=True, check_av=True)
        observed = tower(features, feature_lens=lengths, aftercnn_lens=torch.tensor([7])).last_hidden_state
        stats, arrays = observer.finish()
        observer.close()
    assert torch.equal(baseline, observed)
    assert stats['chunk_lengths'] == [4, 3]
    assert stats['encoder_tokens'] == 7
    np.testing.assert_allclose(arrays['time_s'], [0, .02, .04, .06, .08, .10, .12])
    assert len(stats['layers']) == 3
    assert max(row['shadow_check']['av_relative_l2'] for row in stats['layers']) < 1e-6
    assert arrays['hidden_2'].shape == (4, 8)
    assert np.isnan(arrays['k_adj_0'][3])
    assert not observer.handles


def test_worker_launch_replaces_only_the_child_module():
    from mllm_comp.integrations.native_cli import NativeCLIPlan
    from scripts.analysis.audio_sink.launch import worker_plan
    argv = ['python', '-m', 'accelerate.commands.launch', '--num_processes', '4', '-m', 'lmms_eval', '--tasks', 'mmsufinal']
    plan = NativeCLIPlan(argv=argv, environment={}, cwd='/tmp', expected_output_paths=[])
    changed = worker_plan(plan, '/tmp/observation.json')
    assert plan.argv == argv
    assert changed.argv[2] == 'accelerate.commands.launch'
    assert changed.argv[6] == 'scripts.analysis.audio_sink.worker'
    assert changed.environment['AUDIO_SINK_OBSERVATION_CONFIG'] == '/tmp/observation.json'


def test_group_bootstrap_and_pairing():
    mean, lo, hi = bootstrap_mean([[1, 2], [3, 4]], ['same', 'same'])
    np.testing.assert_allclose(mean, [2, 3])
    assert np.isnan(lo).all() and np.isnan(hi).all()
    rows = [{'id': str(i), 'doc_index': i, 'audio_path': str(i), 'prediction': 'A', 'accuracy': 1} for i in range(2)]
    assert coverage_report(rows, 2)['complete']
    assert not coverage_report(rows + rows[:1], 3)['complete']
    assert compare_baseline(rows, rows)['outputs_identical']
    changed = [dict(rows[0], accuracy=0, prediction='B'), rows[1]]
    assert compare_baseline(changed, rows)['transitions']['correct_to_wrong'] == 1


def test_report_from_real_small_encoder_measurements(tmp_path, monkeypatch):
    from scripts.analysis.audio_sink.report import main
    tower = tiny_audio_encoder()
    observer = AudioEncoderObserver(tower, sample_rate=16000, hop_length=160)
    observer.install()
    rows = []
    with torch.no_grad():
        for i in range(2):
            observer.begin(example=(i == 0), check_av=True)
            frames = 13 if i == 0 else 7
            tower(torch.randn(4, frames), feature_lens=torch.tensor([frames]),
                  aftercnn_lens=torch.tensor([(frames+1)//2]))
            stats, arrays = observer.finish()
            row = {'id': f'unit-test-{i}', 'doc_index': i, 'audio_path': str(i),
                   'accuracy': 1, 'prediction': 'A', **stats}
            if i == 0:
                row['example_file'] = 'example.npz'
                np.savez_compressed(tmp_path / 'example.npz', **arrays)
            rows.append(row)
    observer.close()
    (tmp_path / 'samples.rank0.jsonl').write_text('\n'.join(json.dumps(jsonable(row)) for row in rows))
    (tmp_path / 'observation.json').write_text(json.dumps({'expected_samples': 2}))
    monkeypatch.setattr(sys, 'argv', ['report', '--run-dir', str(tmp_path), '--bootstrap', '10'])
    main()
    assert (tmp_path / 'figures/overview.png').exists()
    assert (tmp_path / 'figures/sink_diagnostics.png').exists()
    assert (tmp_path / 'figures/doc0_layer2_representation.pdf').exists()
    summary = json.loads((tmp_path / 'summary.json').read_text())
    assert summary['shadow_validation']['max_av_relative_l2'] < 1e-6
    assert summary['boundary_comparison']['later_chunks_start_r']['valid_audio_count'] == [1, 1, 1]
    assert summary['boundary_comparison']['later_chunks_start_r']['ci95_low'] == [None, None, None]
