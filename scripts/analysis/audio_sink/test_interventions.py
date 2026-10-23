"""Analytical interventions, real HF encoder/decoder contracts, and full-B bookkeeping."""
from __future__ import annotations
import json
from types import SimpleNamespace
import numpy as np
import pytest
import torch
from .metrics import attention_summary, jsonable
from .observer import AudioEncoderObserver
from .interventions import (
    conditions, choose_blocks, range_mask, intervention_head_output, AttentionInterventions,
    compact_audio_request, refill_indices, FrozenProbe,
)
from .test_audio_sink import tiny_audio_encoder
from .suite_report import paired_intervals, summarize, condition_coverage, render


@pytest.fixture(autouse=True)
def small_cpu_workloads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def test_distance_mass_in_milliseconds():
    q = torch.zeros(3, 1, 2)
    _, heads, _, _ = attention_summary(q, q, q, q, [0, 3], 1, time_step_ms=20)
    np.testing.assert_allclose(heads['distance_mass'][0, :3], [1/3, 4/9, 2/9], atol=1e-6)
    np.testing.assert_allclose(heads['distance_mass'], heads['distance_uniform_mass'])
    np.testing.assert_allclose(heads['mean_distance_ms'], 160/9, atol=1e-6)


def test_range_presoftmax_and_postsoftmax_are_distinct_and_random_budget_matches():
    for n in (1, 5, 9):
        for radius in (0, 1, 3, 20):
            local, random = range_mask(n, radius, 'window', 42), range_mask(n, radius, 'random', 42)
            np.testing.assert_array_equal(local.sum(1), random.sum(1))
            assert random.diagonal().all()
    q = torch.zeros(4, 2, 1)
    v = torch.arange(8, dtype=torch.float32).reshape(4, 2, 1)
    native = v.mean(0, keepdim=True).expand_as(v).clone()
    spec = dict(family='range', window_ms=0, selector='window')
    pre = intervention_head_output(q, q, v, native, [0, 4], 1, dict(spec, placement='pre'), [], 20, 42)
    post = intervention_head_output(q, q, v, native, [0, 4], 1, dict(spec, placement='post'), [], 20, 42)
    torch.testing.assert_close(pre, v)
    torch.testing.assert_close(post, v/4)


def test_fixed_a_value_path_key_renormalization_and_restoration():
    q = torch.zeros(4, 2, 1)
    v = torch.arange(8, dtype=torch.float32).reshape(4, 2, 1)
    native = v.mean(0, keepdim=True).expand_as(v).clone()
    outputs = {}
    for operation in ('v_zero', 'key_mask', 'v_norm', 'restore_v', 'restore_head'):
        spec = dict(family='sink', head=0, operation=operation)
        outputs[operation] = intervention_head_output(q, q, v, native, [0, 4], 1, spec, [3], 20, 42)
        assert torch.equal(outputs[operation][:, 1], native[:, 1])
    torch.testing.assert_close(outputs['v_zero'][:, 0], native[:, 0]-v[3, 0]/4)
    torch.testing.assert_close(outputs['key_mask'][:, 0], v[:3, 0].mean(0).expand(4, 1))
    assert torch.equal(outputs['restore_v'], native)
    assert torch.equal(outputs['restore_head'], native)


def test_frozen_heads_equal_block_budgets_and_refill_budget():
    plan = conditions([0, 2], 4, families=['head'])
    assert len(plan) == 16
    assert all(c['head'] not in c['random_control_heads'] for c in plan)
    sink_plan = conditions([0], 4, families=['sink'])
    assert {c['head'] for c in sink_plan if c['family'] == 'sink'} == {0, 1, 2, 3}
    scores = np.arange(13, dtype=float)
    chosen = [choose_blocks(scores, [0, 8, 13], 2, .7, selector, 42)[0]
              for selector in ('low', 'high', 'random')]
    assert len(set(map(len, chosen))) == 1
    for indices in chosen:
        assert len(set(indices) & set(range(8))) < 8
        assert len(set(indices) & set(range(8, 13))) < 5
    selected, info = choose_blocks(np.ones(2), [0, 2], 4, .1, 'low', 42)
    assert not len(selected) and not info['active']
    kept = [refill_indices(scores, .5, op, .25, 42)[0] for op in ('topk', 'refill_low', 'refill_random')]
    assert all(len(v)==6 and len(set(v))==6 for v in kept)
    assert not np.array_equal(kept[0], kept[1])


def native_fixture(tower, features):
    from mllm_comp.core.state import ForwardState, TokenSpan
    n = len(features[0])
    output_count = ((n+1)//2)//2
    ids = torch.tensor([[1, 10, *([12]*output_count), 11, 2]])
    state = ForwardState(input_ids=ids, attention_mask=torch.ones_like(ids), sample_ids=['fixture'],
        audio_values=features[None, ...], audio_attention_mask=torch.ones(1, n, dtype=torch.long),
        audio_feature_lengths=torch.tensor([n]),
        native_model_inputs=dict(input_ids=ids, attention_mask=torch.ones_like(ids),
                                input_features=features[None, ...], feature_attention_mask=torch.ones(1, n, dtype=torch.long)),
        modality_token_spans=[TokenSpan(sample_index=0, start=2, end=2+output_count, modality='audio')],
        modality_metadata=dict(token_ids={'audio':12}, pad_token_id=0, padding_side='right',
                               expected_token_counts_by_sample=[{'audio':output_count}]))
    calls = {}
    thinker = SimpleNamespace(audio_tower=tower)
    def rope(*args, **kwargs):
        calls['rope_lengths'] = args[5] if len(args)>5 else kwargs.get('audio_seqlens')
        return None
    thinker.get_rope_index = rope
    def generate(runtime, state, *, processed_batch, **kwargs):
        from mllm_comp.backbones.qwen25_omni.validation import Qwen25OmniNativeValidator
        assert not Qwen25OmniNativeValidator().validate_state(runtime, state, processed_batch)
        cnn, count = tower._get_feat_extract_output_lengths(torch.tensor([n]))
        result = tower(features, feature_lens=torch.tensor([n]), aftercnn_lens=cnn).last_hidden_state
        assert len(result) == int(count[0]) == int((state.input_ids==12).sum())
        thinker.get_rope_index(input_ids=state.input_ids, audio_seqlens=torch.tensor([n]))
        calls['state'] = state
        return result
    bridge = SimpleNamespace(runtime=SimpleNamespace(model=SimpleNamespace(thinker=thinker)),
                             facade=SimpleNamespace(runner=SimpleNamespace(generate=generate)))
    processed = SimpleNamespace(sample_ids=['fixture'], metadata=state.modality_metadata)
    return bridge, state, processed, calls


def test_all_conditions_on_real_encoder_and_native_placeholder_contracts(tmp_path):
    tower = tiny_audio_encoder()
    features = torch.randn(4, 41)
    bridge, state, processed, calls = native_fixture(tower, features)
    observer = AudioEncoderObserver(tower, sample_rate=16000, hop_length=160, retain_scores=True)
    observer.install()
    controller = AttentionInterventions(tower, step_ms=20, observer=observer)
    controller.install()
    cases = conditions([0, 2], 2, windows_ms=[0, 40], fractions=[.25], block_ms=20)
    try:
        with torch.no_grad():
            controller.begin()
            observer.begin(check_av=True)
            original = bridge.facade.runner.generate(bridge.runtime, state, processed_batch=processed)
            measurements, _ = observer.finish()
            controller.finish()
            rng_before = torch.random.get_rng_state().clone()
            for condition in cases:
                controller.begin(condition)
                if condition['family']=='input':
                    observer.begin()
                with compact_audio_request(bridge, controller):
                    output = bridge.facade.runner.generate(bridge.runtime, state, processed_batch=processed)
                if condition['family']=='input':
                    observer.finish()
                details = controller.finish()
                assert torch.isfinite(output).all()
                if condition.get('operation') in ('restore', 'restore_v', 'restore_head'):
                    assert torch.equal(output, original), condition
                if controller.keep_indices is not None:
                    assert len(output)==len(controller.keep_indices)
                    assert calls['rope_lengths'].tolist()==[4*len(output)]
                    assert calls['state'].audio_values is state.audio_values
                else:
                    assert output.shape==original.shape
            assert torch.equal(rng_before, torch.random.get_rng_state())
    finally:
        controller.close()
        observer.close()
    assert not controller.handles and not observer.handles


def test_frozen_probe_never_fits_or_selects_heads(tmp_path):
    weights, targets = tmp_path/'probe.npz', tmp_path/'targets.json'
    np.savez(weights, weight=np.eye(2), bias=np.zeros(2), attributes=np.array(['a', 'b']))
    targets.write_text(json.dumps({'id': {'a':1, 'b':2}}))
    probe = FrozenProbe(weights, targets)
    result = probe.read(torch.tensor([[1.,2.],[1.,2.]]), 'id')
    assert result['squared_error']==[0.,0.]
    assert result['values']==[1.,2.]


def test_kernel_benchmark_does_not_label_cpu_as_sparse_gpu_cost():
    from .kernel_benchmark import benchmark_reference, dense_masked
    q, k, v = (torch.randn(5, 2, 3) for _ in range(3))
    result = benchmark_reference(dict(q=q, k=k, v=v, scaling=1), [0,2,5], 20, [40])
    assert result['status']=='not_measured'
    masks = [torch.eye(2, dtype=torch.bool), torch.eye(3, dtype=torch.bool)]
    torch.testing.assert_close(dense_masked(q,k,v,[0,2,5],1,masks), v)


def test_full_b_pairing_bootstrap_and_smoke_report(tmp_path):
    from .suite_report import main
    cases = conditions([0], 2, windows_ms=[40], fractions=[.25], block_ms=20)
    plan = dict(conditions=cases, observer_layers=[0], probe_weights=None, seed=42)
    (tmp_path/'suite_plan.json').write_text(json.dumps(plan))
    (tmp_path/'observation.json').write_text(json.dumps({'expected_samples':3}))
    reference = []
    for i in range(3):
        reference.append(dict(id=str(i), doc_index=i, audio_path=str(i), prediction='A', accuracy=1,
            question_type='perception' if i<2 else 'reasoning', sub_category='sound', task_name='event',
            layers=[dict(layer=0, heads=dict(distance_mass=np.full((2,9),1/9).tolist(),
                                           distance_uniform_mass=np.full((2,9),1/9).tolist()))]))
    (tmp_path/'samples.rank0.jsonl').write_text('\n'.join(json.dumps(row) for row in reference))
    rows = []
    for condition in cases:
        for i in range(3):
            restore = condition.get('operation') in ('restore', 'restore_v', 'restore_head')
            altered = i==0 and not restore
            if condition['family']=='evidence' and condition['selector']=='high' and condition['operation']=='mean':
                altered = False
            rows.append(dict(id=str(i), doc_index=i, condition_id=condition['id'],
                prediction='B' if altered else 'A', labels_preserved=condition['family']!='input',
                accuracy=None if condition['family']=='input' else int(not altered),
                original_label_agreement=0, diagnostic_generation_seconds=.1,
                intervention=dict(pooled_representation_relative_l2=.1, projector_tokens=10, selected_mean_r=3)))
    path = tmp_path/'conditions.rank0.jsonl'
    path.write_text('\n'.join(json.dumps(row) for row in rows))
    assert all(v['complete'] for v in condition_coverage(tmp_path, reference, plan).values())
    with pytest.raises(ValueError, match='5,000'):
        summarize(tmp_path, repeats=10)
    summary = summarize(tmp_path, repeats=10, allow_smoke=True)
    assert not summary['full_B'] and summary['restore_checks_passed']
    assert summary['conditions'][0]['transitions']['correct_to_wrong']==1
    contrast = summary['paired_mechanism_comparisons']['evidence_low_0.25_mean_minus_high']
    assert contrast['accuracy_delta'] == pytest.approx(-1/3)
    assert 'range_l0_window_40ms_pre_minus_random' in summary['paired_mechanism_comparisons']
    assert all(r['accuracy_delta'] != r['accuracy_delta'] for r in summary['conditions'] if r['family']=='input')
    (tmp_path/'figures').mkdir()
    render(summary, tmp_path/'figures')
    assert (tmp_path/'figures/attention_distance.png').exists()
    assert (tmp_path/'figures/head_attributes_l0.pdf').exists()
    assert (tmp_path/'figures/low_attention_refill.png').exists()
    path.write_text('\n'.join(json.dumps(row) for row in rows[:-1]))
    with pytest.raises(ValueError, match='missing'):
        summarize(tmp_path, repeats=10, allow_smoke=True)
    mean, lo, hi = paired_intervals([[1.,3.,5.]], ['same']*3, repeats=10)
    assert mean[0]==3 and np.isnan(lo[0]) and np.isnan(hi[0])


def test_suite_transport_with_real_hf_thinker_generation(tmp_path, monkeypatch):
    import sys
    from types import ModuleType
    from transformers import Qwen2_5OmniThinkerConfig, Qwen2_5OmniThinkerForConditionalGeneration
    from .suite_runner import SuiteRunner
    config = Qwen2_5OmniThinkerConfig(
        audio_config=dict(d_model=8, encoder_layers=2, encoder_attention_heads=2,
            encoder_ffn_dim=16, num_mel_bins=4, max_source_positions=8, n_window=4, output_dim=16),
        vision_config=dict(depth=1, hidden_size=16, intermediate_size=32, out_hidden_size=16,
            num_heads=2, patch_size=2, spatial_merge_size=1, fullatt_block_indexes=[0]),
        text_config=dict(vocab_size=32, hidden_size=16, intermediate_size=32, num_hidden_layers=1,
            num_attention_heads=2, num_key_value_heads=2,
            rope_scaling=dict(rope_type='default', mrope_section=[2,1,1]),
            pad_token_id=0, bos_token_id=1, eos_token_id=2),
        audio_token_index=12, audio_start_token_id=10, audio_end_token_id=11,
        image_token_index=13, video_token_index=14, vision_start_token_id=15)
    config._attn_implementation = 'eager'
    thinker = Qwen2_5OmniThinkerForConditionalGeneration(config).eval()
    features = torch.randn(4, 41)
    bridge, state, processed, _ = native_fixture(thinker.audio_tower, features)
    bridge.runtime.model.thinker = thinker
    bridge.runtime.processor = SimpleNamespace(feature_extractor=SimpleNamespace(sampling_rate=16000, hop_length=160))
    bridge._generation_kwargs = lambda request: (dict(thinker_do_sample=False), [])
    bridge._document = lambda task, split, index: dict(id=str(index), audio_path=str(index),
        question_type='perception', sub_category='fixture', task_name='fixture')
    def generate(runtime, state, *, processed_batch, **kwargs):
        ids = thinker.generate(**state.native_model_inputs, max_new_tokens=2, do_sample=False,
                               eos_token_id=2, pad_token_id=0)
        return ' '.join(map(str, ids[0, state.input_ids.shape[1]:].tolist()))
    bridge.facade.runner.generate = generate
    # Only the scorer is a fixture; encoder, decoder, generation, and compact positions are real HF.
    scorer = ModuleType('lmms_eval.tasks.mmsufinal.utils')
    scorer.mmsufinal_process_results = lambda doc, answers: {'accuracy': {'score': float(bool(answers[0]))}}
    monkeypatch.setitem(sys.modules, 'lmms_eval.tasks.mmsufinal.utils', scorer)
    cases = conditions([0], 2, windows_ms=[40], fractions=[.25], block_ms=20)
    plan = dict(conditions=cases, observer_layers=[0,1], layers=[0], encoder_layers=2, encoder_heads=2,
                probe_weights=None, kernel_benchmark_samples=0, seed=42)
    (tmp_path/'suite_plan.json').write_text(json.dumps(plan))
    settings = dict(suite_plan=str(tmp_path/'suite_plan.json'), output_dir=str(tmp_path),
        query_block=64, example_chunk=0, example_doc_ids=[], check_av_samples=2, expected_samples=2)
    (tmp_path/'observation.json').write_text(json.dumps(settings))
    suite = SuiteRunner(bridge, settings, 0)
    rows = []
    try:
        with torch.no_grad():
            for i in range(2):
                kwargs = dict(task='mmsufinal', split='test', doc_id=i, request_generation={}, cache_hook=None)
                prediction, row = suite.run(None, kwargs,
                    lambda ignored: bridge.facade.runner.generate(bridge.runtime, state, processed_batch=processed))
                assert prediction
                rows.append(row)
    finally:
        suite.close()
    (tmp_path/'samples.rank0.jsonl').write_text('\n'.join(json.dumps(jsonable(row)) for row in rows))
    summary = summarize(tmp_path, repeats=10, allow_smoke=True)
    assert summary['restore_checks_passed']
    assert all(c['samples']==2 for c in summary['conditions'])
    assert summary['conditions'][0]['family']=='range'
