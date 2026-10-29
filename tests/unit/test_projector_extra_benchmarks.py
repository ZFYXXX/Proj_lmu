from io import BytesIO
import copy

from PIL import Image
import pytest

from configs.eval.lmms_eval.tasks.projector_extra import extra_utils as tasks
from scripts.eval.lmms_eval import run_projector_extra_benchmarks as runner


@pytest.mark.parametrize('kind', ['bytes', 'mapping', 'pil'])
def test_image_decoding_matches_raw_rgb(kind):
    source = Image.new('RGB', (8, 6), (12, 20, 30))
    stream = BytesIO()
    source.save(stream, format='PNG')
    value = {'bytes': stream.getvalue()} if kind == 'mapping' else stream.getvalue() if kind == 'bytes' else source
    result, = tasks.visual({'image': value})
    assert result.mode == 'RGB' and result.size == source.size
    assert result.tobytes() == source.tobytes()


def test_smoke_stratifies_without_reading_answers():
    labels = ['a'] * 20 + ['b'] * 20 + ['c'] * 20
    assert tasks.smoke_indices(labels) == [0, 20, 40, 1, 21, 41, 2, 22]
    assert tasks.smoke_indices(labels) == tasks.smoke_indices(labels)
    assert tasks.smoke_indices(['a']) == [0]
    assert tasks.smoke_indices([]) == []


def test_smoke_environment_does_not_shrink_full_dataset(monkeypatch):
    from datasets import Dataset
    data = Dataset.from_dict({'category': ['a'] * 6 + ['b'] * 6 + ['c'] * 6})
    monkeypatch.setenv('MLLM_EXTRA_SMOKE', '1')
    assert len(tasks.process_docs(data)) == 8
    assert set(tasks.process_docs(data)['category']) == {'a', 'b', 'c'}
    monkeypatch.setenv('MLLM_EXTRA_SMOKE', '0')
    assert tasks.process_docs(data) is data


def test_pope_native_accuracy_unchanged_and_yes_ratio_counts_model_not_target():
    doc = {'question_id': 4, 'answer': 'yes'}
    result = tasks.pope_results(doc, ['no'])
    assert result['pope_accuracy'] == 0
    record = result['pope_precision']
    assert tasks.pope_yes_ratio([record]) == 0
    assert tasks.pope.pope_aggregate_yes_ratio([record]) == 1  # known upstream bug
    assert tasks.pope_results(doc, ['Yes.'])['pope_accuracy'] == 0  # retain native strict semantics


def test_pope_invalid_output_is_not_silently_removed_from_recall():
    records = [{'ground_truth': 'yes', 'prediction': 'yes'},
               {'ground_truth': 'yes', 'prediction': 'unsure'},
               {'ground_truth': 'no', 'prediction': 'yes'},
               {'ground_truth': 'no', 'prediction': 'no'}]
    stats = tasks.pope_statistics(records)
    assert stats['pope_precision'] == .5
    assert stats['pope_recall'] == .5
    assert stats['pope_f1_score'] == .5
    assert stats['pope_yes_ratio'] == .5
    assert stats['pope_invalid_ratio'] == .25


def test_mmstar_preserves_native_macro_and_extraction():
    record = {'index': 1, 'answer': 'B', 'category': 'math', 'l2_category': 'geometry'}
    assert tasks.mmstar_results(record, ['(B)'])['average']['score'] == 1
    assert tasks.mmstar_results(record, ['C'])['mmstar_micro_accuracy'] == 0
    rows = [{'l2_category': 'a', 'score': 1.},
            {'l2_category': 'a', 'score': 1.}, {'l2_category': 'b', 'score': 0.}]
    assert tasks.macro_score(rows) == tasks.mmstar_aggregate_results(rows) == .5


def test_native_task_prompt_does_not_read_answers():
    for fn in (tasks.mmstar_doc_to_text, tasks.pope_doc_to_text):
        a = fn({'question': 'Which object?', 'answer': 'A'}, {'post_prompt': '\nAnswer briefly.'})
        b = fn({'question': 'Which object?', 'answer': 'B'}, {'post_prompt': '\nAnswer briefly.'})
        assert a == b


@pytest.mark.parametrize('task,count', [('mmstar', 1500), ('pope', 9000)])
@pytest.mark.parametrize('phase,limit', [('smoke', 8), ('full', None)])
def test_config_only_changes_task_io_not_model_or_selector(task, count, phase, limit, tmp_path):
    template = {'experiment': {}, 'backbone': {'name': 'pinned'},
                'pipeline': {'stages': []}, 'eval': {'args': {}, 'generation': {'do_sample': False}},
                'data': {}, 'output': {}}
    original = copy.deepcopy(template)
    p = {'templates': {'qwen25': {'dense': template}}}
    cfg = runner.config(p, 'qwen25', 'dense', phase, task, tmp_path / 'run')
    assert template == original
    assert cfg['backbone'] == template['backbone'] and cfg['pipeline'] == template['pipeline']
    assert cfg['eval']['tasks'] == [f'projector_extra_{task}'] and cfg['eval']['sample_limit'] == limit
    assert runner.COUNTS[task] == count


def test_trace_budget_duplicates_and_positions_fail_explicitly():
    record = {'source_count': 256, 'result_count': 128, 'selected_indices': list(range(128)),
              'decode_position_verified': True, 'observed_visual_tokens_per_decoder_layer': [128] * 32}
    assert runner.check_trace({'selection': record}, 256, 32) == 128
    for changed in ({'result_count': 127}, {'selected_indices': [0] * 128},
                    {'decode_position_verified': False}, {'observed_visual_tokens_per_decoder_layer': [256] * 32}):
        with pytest.raises(ValueError):
            runner.check_trace({'selection': {**record, **changed}}, 256, 32)


def test_image_cluster_bootstrap_and_macro_estimand():
    result = runner.bootstrap([1, 1, 0, 0], ['a', 'a', 'b', 'b'], resamples=100)
    assert result['delta'] == .5 and result['independent_images'] == 2
    # A single image contains both strata: no resample can lose a stratum.
    result = runner.bootstrap([1, 1, 0], ['a', 'a', 'a'], ['x', 'x', 'y'], resamples=100)
    assert result['delta'] == .5 and result['ci95'] == [.5, .5]


def test_model_environments_isolate_transformers_and_caches(tmp_path):
    a = runner.child_env(tmp_path, 'qwen25', 0, 'full', tmp_path / 'run')
    b = runner.child_env(tmp_path, 'qwen35', 1, 'smoke', tmp_path / 'run2')
    assert 'transformers-519' not in a['PYTHONPATH'] and 'transformers-519' in b['PYTHONPATH']
    assert a['CUDA_VISIBLE_DEVICES'] == '0' and b['CUDA_VISIBLE_DEVICES'] == '1'
    assert a['MLLM_EXTRA_SMOKE'] == '0' and b['MLLM_EXTRA_SMOKE'] == '1'


def test_mmstar_schedule_does_not_include_pope_or_change_original_protocol():
    from scripts.eval.lmms_eval.run_projector_extra_schedule import task_scope
    p = {'counts': runner.COUNTS, 'models': ['qwen25', 'qwen35'], 'methods': list(runner.METHODS)}
    old = copy.deepcopy(p)
    scope = task_scope(p, ['mmstar'])
    assert scope == {'active_tasks': ['mmstar'], 'deferred_tasks': ['pope'],
                     'active_full_cells': 20, 'active_full_requests': 30000}
    assert p == old
    for invalid in ([], ['mmstar', 'mmstar'], ['unknown']):
        with pytest.raises(ValueError):
            task_scope(p, invalid)
