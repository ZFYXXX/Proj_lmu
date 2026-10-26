import copy
import pytest
import torch

from mllm_comp.analysis.common.selector_conditioning import FORMAT, ids_sha, matched_donors, request_key
from mllm_comp.compressors.comparison.selector_condition import QwenSelectorConditionComparison
from mllm_comp.compressors.comparison.context_topk import QwenContextTopKComparison
from mllm_comp.compressors.image.context_visual_topk import ContextVisualTopK, ContextVisualTopKSpec
from mllm_comp.integrations.lmms_eval.request_context import CURRENT_SAMPLE_ID, sample_request

SAMPLE = 'context_topk_chartqa:test:0:0'


def manifest():
    return {'format': FORMAT, 'rows': {request_key(SAMPLE): {'true_ids': [9, 2], 'donor_ids': [43, 42],
        'generic_ids': [44, 45], 'donor_doc_id': 1, 'donor_image_fingerprint': 'other-image',
        'stratum': 'all', 'informative': True, 'reason': 'different_question'}}}


def test_matched_assignment_is_deterministic_other_image_and_within_task_stratum():
    rows = [{'task': task, 'stratum': 's', 'true_ids': [i+10], 'image_fingerprint': str(i)}
        for task in ('chartqa', 'ocrbench') for i in range(4)]
    before = copy.deepcopy(rows)
    a, b = matched_donors(rows), matched_donors(rows)
    assert a == b and rows == before
    for i, assignment in enumerate(a):
        other = rows[assignment['donor_index']]
        assert rows[i]['task'] == other['task'] and rows[i]['stratum'] == other['stratum']
        assert rows[i]['image_fingerprint'] != other['image_fingerprint']
        assert assignment['informative']
    assert sorted(r['donor_index'] for r in a) == list(range(len(rows)))


def test_identical_questions_and_singleton_are_not_evidence_of_nonuse():
    rows = [{'task': 'ocrbench', 'stratum': 's', 'true_ids': [10], 'image_fingerprint': str(i)} for i in range(4)]
    result = matched_donors(rows)
    assert all(not r['informative'] and r['reason'] == 'identical_question_after_matching' for r in result)
    assert all(r['donor_index'] != i for i, r in enumerate(result))
    assert matched_donors(rows[:1])[0]['reason'] == 'no_other_image_in_stratum'


def test_assignment_maximizes_changed_questions():
    rows = [{'task': 't', 'stratum': 's', 'true_ids': [10+i//2], 'image_fingerprint': str(i)} for i in range(4)]
    assert all(r['informative'] for r in matched_donors(rows))


def test_request_scope_resets_after_nested_exception(monkeypatch):
    from mllm_comp.integrations.lmms_eval.request_performance import request_performance
    monkeypatch.delenv('MLLM_TOPK_PERFORMANCE', raising=False)
    assert CURRENT_SAMPLE_ID.get() is None
    with sample_request('outer'):
        with pytest.raises(RuntimeError):
            with request_performance(None, None, 'inner'):
                assert CURRENT_SAMPLE_ID.get() == 'inner'
                raise RuntimeError('test')
        assert CURRENT_SAMPLE_ID.get() == 'outer'
    assert CURRENT_SAMPLE_ID.get() is None


@pytest.mark.parametrize('mode', ['true', 'shuffled', 'generic'])
def test_native_condition_replaces_selector_only_and_preserves_decode_and_labels(mode):
    from tests.unit.test_projector_comparison_evaluation import tiny_state
    bundle, state, _, _ = tiny_state()
    model, prompt = bundle['model'], state['prompt_inputs']
    selector = ContextVisualTopK(ContextVisualTopKSpec(state['features'].shape[1], width=8, target_count=3))
    torch.nn.init.normal_(selector.context_projection.weight, std=.3)
    with torch.no_grad():
        original = QwenContextTopKComparison(model, {'target_count': 3, 'width': 8}, selector_module=selector).install()
        try:
            baseline = model.generate(**prompt, max_new_tokens=3, min_new_tokens=3, do_sample=False, pad_token_id=0)
            true_indices = original.runtime_metadata()['selection']['selected_indices']
        finally:
            original.uninstall()
    session = QwenSelectorConditionComparison(model, {'target_count': 3, 'width': 8, 'condition_mode': mode},
        selector_module=selector, condition_data=manifest()).install()
    original_prepare = session.prepare_source_inputs
    def prepare(call):
        before = {k: v.clone() for k,v in call.items() if torch.is_tensor(v)}
        original_prepare(call)
        assert all(torch.equal(call[k], v) for k,v in before.items())
    session.prepare_source_inputs = prepare
    before = prompt['input_ids'].clone()
    try:
        with torch.no_grad(), sample_request(SAMPLE):
            actual = model.generate(**prompt, max_new_tokens=3, min_new_tokens=3, do_sample=False, pad_token_id=0)
            trace = session.runtime_metadata()['selection']
            labels = torch.full_like(prompt['input_ids'], -100)
            labels[:, -1] = 3
            result = model(**prompt, labels=labels, use_cache=False)
        assert torch.equal(before, prompt['input_ids']) and torch.isfinite(result.loss)
        assert trace['condition_mode'] == mode and not trace['llm_question_replaced']
        assert trace['true_selected_indices'] == true_indices
        key = {'true':'true_ids', 'shuffled':'donor_ids', 'generic':'generic_ids'}[mode]
        assert trace['condition_ids_sha256'] == ids_sha(manifest()['rows'][request_key(SAMPLE)][key])
        assert trace['decode_position_verified'] and trace['result_count'] == 3
        assert trace['observed_visual_tokens_per_decoder_layer'] == [3] * 4
        if mode == 'true':
            assert torch.equal(actual, baseline) and trace['selection_identical']
    finally:
        session.uninstall()


def test_condition_request_identity_prompt_mismatch_and_answer_fields_fail():
    from tests.unit.test_projector_comparison_evaluation import tiny_state
    bundle, state, _, _ = tiny_state()
    for problem in ('missing', 'different_prompt', 'answer'):
        data = manifest()
        if problem == 'different_prompt':
            data['rows'][request_key(SAMPLE)]['true_ids'] = [44]
        if problem == 'answer':
            data['rows'][request_key(SAMPLE)]['answer'] = 'leak'
        session = QwenSelectorConditionComparison(bundle['model'], {'target_count': 3}, condition_data=data).install()
        try:
            with pytest.raises(ValueError), torch.no_grad():
                if problem == 'missing':
                    bundle['model'](**state['prompt_inputs'])
                else:
                    with sample_request(SAMPLE):
                        bundle['model'](**state['prompt_inputs'])
        finally:
            session.uninstall()


def test_mean_embedding_condition_is_order_invariant_not_a_contextual_encoder():
    selector = ContextVisualTopK(ContextVisualTopKSpec(12, 8, 3))
    torch.nn.init.normal_(selector.context_projection.weight, std=.2)
    x, q = torch.randn(11,12), torch.randn(7,12)
    a, b = selector(x, question=q), selector(x, question=q.flip(0))
    torch.testing.assert_close(a['scores'], b['scores'], rtol=1e-5, atol=1e-8)
    assert torch.equal(a['indices'], b['indices'])


def test_request_lookup_ignores_evaluator_sort_index_but_not_document():
    assert request_key('context_topk_chartqa:test:747:0') == request_key('context_topk_chartqa:test:747:1399')
    assert request_key('context_topk_chartqa:test:747:0') != request_key('context_topk_chartqa:test:748:0')


def test_stage_is_native_projector_only():
    from mllm_comp.pipeline.stage import CompressionStageConfig
    from mllm_comp.pipeline.v2_validation import validate_v2_stage_contract
    from mllm_comp.compressors.multimodal_registry import builtin_multimodal_compression_contracts
    name = 'selector_condition_audit'
    stage = CompressionStageConfig.from_dict({'name':name, 'algorithm':name, 'plugin':name,
        'variant':f'{name}.image', 'binding':f'{name}.image.qwen25_vl', 'placement':'projector_level',
        'method':'token_compression', 'object_family':'tensor_slot', 'execution_phase':'generate',
        'modalities':['image'], 'target_modalities':['image'], 'conditioning_modalities':['text'], 'config':{}})
    validate_v2_stage_contract(stage, registry=builtin_multimodal_compression_contracts(), backbone_name='qwen25_vl')
