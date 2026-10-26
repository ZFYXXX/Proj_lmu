import copy
import json

import pytest
import torch

from mllm_comp.compressors.image.word_visual_topk import (
    WordVisualTopK,WordVisualTopKSpec,clean_ids,prompt_word_question,save_selector,load_selector,
    evidence_masks,evidence_loss,
)
from mllm_comp.analysis.common.topk_training_modes import parameter_digest


def model():
    return WordVisualTopK(WordVisualTopKSpec(12,8,3,match_width=4))


def test_prior_frozen_and_query_gradients_finite_nonzero():
    m=model().train()
    before=parameter_digest(m.visual)
    x,q=torch.randn(11,12,requires_grad=True),torch.randn(5,12,requires_grad=True)
    out=m(x,question=q,soft=True)
    assert len(out['tokens'])==11 and len(out['indices'])==3
    torch.testing.assert_close(out['soft_weights'].sum(),torch.tensor(3.),atol=1e-5,rtol=0)
    (out['tokens']*torch.randn_like(out['tokens'])).sum().backward()
    for name,p in m.named_parameters():
        if name.startswith('visual.'):
            assert not p.requires_grad and p.grad is None
        else:
            assert p.grad is not None and p.grad.isfinite().all() and p.grad.norm()>0
    assert q.grad is None  # frozen embedding source
    assert parameter_digest(m.visual)==before
    assert not m.visual.training


def test_visual_only_exact_prior_and_no_forced_differences_when_k_equals_n():
    m=model();x,q=torch.randn(11,12),torch.randn(5,12)
    out=m(x,question=q,visual_only=True);old=m.visual(x)
    torch.testing.assert_close(out['scores'],old['scores'],rtol=0,atol=0)
    assert torch.equal(out['indices'],old['indices'])
    all_m=WordVisualTopK(WordVisualTopKSpec(12,8,20,match_width=4))
    assert torch.equal(all_m(x,question=q)['tokens'],x)


def test_specific_words_can_change_selection_not_only_common_offset():
    m=WordVisualTopK(WordVisualTopKSpec(2,2,1,match_width=2))
    m.visual.scorer[0]=torch.nn.Identity()
    with torch.no_grad():
        m.visual.scorer[1].weight.copy_(torch.eye(2));m.visual.scorer[1].bias.zero_()
        m.visual.scorer[3].weight.zero_();m.visual.scorer[3].bias.zero_()
        m.visual_projection.weight.copy_(torch.eye(2));m.question_projection.weight.copy_(torch.eye(2))
    x=torch.tensor([[2.,-1.],[-1.,2.]])
    a=m(x,question=torch.tensor([[1.,-1.]]))
    b=m(x,question=torch.tensor([[-1.,1.]]))
    assert a['indices'].tolist()==[0] and b['indices'].tolist()==[1]
    assert (a['scores']-b['scores']).std()>0


def test_word_pooling_is_repeat_invariant_but_not_a_contextual_encoder():
    m=model();x,q=torch.randn(11,12),torch.randn(5,12)
    a=m(x,question=q)
    for other in (q.flip(0),q.repeat(2,1)):
        b=m(x,question=other)
        torch.testing.assert_close(a['scores'],b['scores'],atol=1e-8,rtol=1e-5)
        assert torch.equal(a['indices'],b['indices'])


def test_only_exact_suffix_is_removed_not_words_options_units_or_negation():
    ids=[11,12,13,14,198,80,81]
    assert clean_ids(ids,[(80,81),(198,80,81)])==[11,12,13,14]
    assert clean_ids([80,81,11],[(80,81)])==[80,81,11]
    with pytest.raises(ValueError,match='empty'):clean_ids([80,81],[(80,81)])


def test_word_prompt_is_answer_free_and_suffix_only():
    from tests.unit.test_projector_comparison_evaluation import tiny_state
    bundle,state,_,_=tiny_state();m=bundle['model'];prompt=state['prompt_inputs']
    q,a=prompt_word_question(m,prompt,[(2,)])
    assert a['removed_format_tokens']==1 and a['question_tokens']==1
    teacher=copy.deepcopy(prompt)
    teacher['input_ids']=torch.cat([prompt['input_ids'],torch.tensor([[151644,5,198,42,151645]])],-1)
    teacher['attention_mask']=torch.ones_like(teacher['input_ids'])
    teacher['labels']=torch.full_like(teacher['input_ids'],-100);teacher['labels'][:,-2]=42
    other,b=prompt_word_question(m,teacher,[(2,)])
    assert torch.equal(q,other) and a==b


@pytest.mark.parametrize('soft',[False,True])
def test_native_word_binding_budget_labels_and_cache(soft):
    from tests.unit.test_projector_comparison_evaluation import tiny_state
    from mllm_comp.compressors.comparison.word_topk import QwenWordTopKComparison
    bundle,state,_,_=tiny_state();base=bundle['model'];prompt=state['prompt_inputs']
    session=QwenWordTopKComparison(base,{'target_count':3,'training_mode':'soft' if soft else 'hard'},
        phase='forward' if soft else 'generate').install()
    try:
        if soft:
            labels=torch.full_like(prompt['input_ids'],-100);labels[:,-1]=3
            output=base(**prompt,labels=labels,use_cache=False)
            output.loss.backward()
            assert all(p.grad is None for p in session.selector.visual.parameters())
            assert session.selector.question_projection.weight.grad.norm()>0
        else:
            with torch.no_grad():base.generate(**prompt,max_new_tokens=3,min_new_tokens=3,do_sample=False,pad_token_id=0)
            trace=session.runtime_metadata()['selection']
            assert trace['result_count']==3 and trace['decode_position_verified'] and trace['word_matching']
            assert trace['observed_visual_tokens_per_decoder_layer']==[3]*4
    finally:session.uninstall()


def test_native_visual_only_matches_dense_identity_and_prior_selection():
    from tests.unit.test_projector_comparison_evaluation import tiny_state
    from mllm_comp.compressors.comparison.word_topk import QwenWordTopKComparison
    bundle,state,_,_=tiny_state();base=bundle['model'];prompt=state['prompt_inputs']
    with torch.no_grad():dense=base(**prompt,use_cache=False).logits
    session=QwenWordTopKComparison(base,{'target_count':128,'visual_only':True}).install()
    try:
        with torch.no_grad():torch.testing.assert_close(base(**prompt,use_cache=False).logits,dense,rtol=1e-5,atol=1e-6)
    finally:session.uninstall()


def test_optional_evidence_requires_verified_disjoint_regions():
    assert evidence_masks(None,height=4,width=4,image_fingerprint='image') is None
    ann={'status':'verified','source':'human','reviewer':'reviewer','image_fingerprint':'image',
        'coordinate_space':'normalized_original_image','positive_boxes':[[0,0,.25,.25]],
        'negative_boxes':[[.75,.75,1,1]]}
    masks=evidence_masks(ann,height=4,width=4,image_fingerprint='image')
    scores=torch.randn(16,requires_grad=True);loss=evidence_loss(scores,*masks);loss.backward()
    assert scores.grad[0]<0 and scores.grad[15]>0
    for bad in ({**ann,'source':'attention_pseudo_label'},{**ann,'negative_boxes':ann['positive_boxes']}):
        with pytest.raises(ValueError):evidence_masks(bad,height=4,width=4,image_fingerprint='image')


def test_independent_checkpoint_and_exact_optimizer_restore(tmp_path,monkeypatch):
    from mllm_comp.analysis.common import learned_topk_experiment as ex
    monkeypatch.setattr(ex,'OWNER',tmp_path)
    def create():
        m=model();o=torch.optim.AdamW([p for p in m.parameters() if p.requires_grad],lr=2e-4)
        return m,o,torch.optim.lr_scheduler.LambdaLR(o,ex.lr_multiplier)
    def advance(m,o,s,n):
        for _ in range(n):
            out=m(torch.randn(11,12),question=torch.randn(5,12),soft=True)
            (out['tokens']*torch.randn_like(out['tokens'])).sum().backward()
            o.step();s.step();o.zero_grad(set_to_none=True)
    torch.manual_seed(42);a,oa,sa=create();advance(a,oa,sa,4)
    torch.manual_seed(42);b,ob,sb=create();advance(b,ob,sb,2)
    contract={'base_identity_sha256':'base'}
    ex.save_training_state(tmp_path/'checkpoint',selector=b,optimizer=ob,scheduler=sb,step=2,
        contract=contract,random_states=[ex.rng_state()],save_fn=save_selector)
    c,oc,sc=create()
    assert ex.restore_training_state(tmp_path/'checkpoint',selector=c,optimizer=oc,scheduler=sc,
        contract=contract,load_fn=load_selector)==2
    advance(c,oc,sc,2)
    assert parameter_digest(a)==parameter_digest(c) and sa.state_dict()==sc.state_dict()
    assert not any(p.requires_grad for p in c.visual.parameters())
    with pytest.raises(ValueError):load_selector(tmp_path/'checkpoint',base_identity='other',device='cpu')
    path=tmp_path/'checkpoint/selector_config.json';cfg=json.loads(path.read_text())
    cfg['format']='qwen25_context_visual_topk_v1';path.write_text(json.dumps(cfg))
    with pytest.raises(ValueError):load_selector(tmp_path/'checkpoint',base_identity='base',device='cpu')


def test_word_stage_is_registered_at_projector_input():
    from mllm_comp.compressors.multimodal_registry import builtin_multimodal_compression_contracts
    from mllm_comp.pipeline.stage import CompressionStageConfig
    from mllm_comp.pipeline.v2_validation import validate_v2_stage_contract
    name='word_visual_topk'
    stage=CompressionStageConfig.from_dict({'name':name,'algorithm':name,'plugin':name,'variant':f'{name}.image',
        'binding':f'{name}.image.qwen25_vl','placement':'projector_level','method':'token_compression',
        'object_family':'tensor_slot','execution_phase':'generate','modalities':['image'],'target_modalities':['image'],
        'conditioning_modalities':['text'],'config':{}})
    validate_v2_stage_contract(stage,registry=builtin_multimodal_compression_contracts(),backbone_name='qwen25_vl')


@pytest.mark.parametrize('visual_only',[False,True])
def test_word_condition_audit_replaces_only_scorer_and_visual_control_is_invariant(tmp_path,visual_only):
    from mllm_comp.compressors.comparison.word_topk import QwenWordTopKComparison
    from mllm_comp.analysis.common.anchor7b_runtime import file_sha
    from mllm_comp.core.serialization import atomic_write_json
    from mllm_comp.integrations.lmms_eval.request_context import sample_request
    from tests.unit.test_projector_comparison_evaluation import tiny_state
    from tests.unit.test_selector_condition_audit import manifest,SAMPLE
    path=tmp_path/'assignment.json';atomic_write_json(path,manifest())
    bundle,state,_,_=tiny_state();base=bundle['model'];prompt=state['prompt_inputs']
    before=prompt['input_ids'].clone()
    session=QwenWordTopKComparison(base,{'target_count':3,'visual_only':visual_only,
        'audit_conditions':str(path),'audit_conditions_sha256':file_sha(path)}).install()
    try:
        with torch.no_grad(),sample_request(SAMPLE):
            base.generate(**prompt,max_new_tokens=3,min_new_tokens=3,do_sample=False,pad_token_id=0)
        trace=session.runtime_metadata()['selection']
        assert torch.equal(before,prompt['input_ids'])
        assert trace['decode_position_verified']
        assert set(trace['question_selection_audit'])=={'shuffled','generic'}
        for a in trace['question_selection_audit'].values():
            assert a['question_changed_after_cleaning'] and not a['answers_generated_for_other_question']
            if visual_only:assert a['exchanged_tokens']==0 and a['centered_score_change_rms']==0
    finally:session.uninstall()


def test_historical_control_reader_filters_native_prefixes_and_shared_runs(tmp_path):
    from scripts.train.qwen25_vl.run_word_visual_topk import reference_traces
    from mllm_comp.core.serialization import atomic_write_text
    rows=[{'request':{'sample_id':f'learned_topk_{task}:test:{i}:{5-i}'},
        'runtime':{'activations':[{'selection':{'selected_indices':[i]}}]}}
        for task in ('chartqa','ocrbench') for i in range(2)]
    atomic_write_text(tmp_path/'compression_runtime.requests.jsonl',''.join(json.dumps(r)+'\n' for r in rows))
    result=reference_traces({'run':str(tmp_path),'task_name':'learned_topk_chartqa'})
    assert set(result)=={0,1} and result[1]['selected_indices']==[1]


def test_control_pins_support_reused_artifacts_not_only_new_completion_markers(tmp_path):
    from scripts.train.qwen25_vl.run_word_visual_topk import control_artifacts
    from mllm_comp.analysis.common.anchor7b_runtime import file_sha
    from mllm_comp.core.serialization import atomic_write_json
    path=tmp_path/'old_result.json';atomic_write_json(path,{'status':'complete'})
    digest=file_sha(path)
    assert control_artifacts({'action':'reuse','artifacts_sha256':{str(path):digest}})=={str(path):digest}
    assert control_artifacts({'completion_marker':str(path),'completion_marker_sha256':digest})=={str(path):digest}
    with pytest.raises(ValueError):control_artifacts({'action':'reuse'})
    with pytest.raises(ValueError):control_artifacts({'artifacts_sha256':{str(path):'wrong'}})
