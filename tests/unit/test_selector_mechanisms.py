import copy
import pytest
import torch

from mllm_comp.analysis.common.selector_mechanisms import (
    VARIANTS,ranking_stats,condition_path_stats,plan_swaps,evidence_coverage,spatial_counts,
)
from mllm_comp.analysis.common.selector_mechanism_cohort import choose_images


def test_common_score_shift_is_not_a_ranking_change():
    a=torch.tensor([5.,4.,3.,2.,1.]);b=a+2
    out=ranking_stats(a,b,2)
    assert out['common_shift_energy_fraction']==pytest.approx(1)
    assert out['relative_score_change_rms']==0 and out['swapped_token_count']==0
    assert out['rank_spearman']==pytest.approx(1)
    assert out['score_energy_closure_relative']<1e-12


def test_boundary_crossings_and_zero_change_have_no_nan():
    a=torch.tensor([5.,4.,3.,2.,1.]);b=torch.tensor([5.,2.5,4.,2.,1.])
    out=ranking_stats(a,b,2,boundary_band=1)
    assert out['lost_indices']==[1] and out['gained_indices']==[2]
    assert out['crossings_in_true_boundary_band_fraction']==1
    assert ranking_stats(a,a,5)['true_k_boundary_gap'] is None
    assert ranking_stats(torch.ones(5),torch.ones(5),2)['rank_spearman'] is None


def test_actual_gelu_path_and_constant_linear_condition_control():
    from mllm_comp.compressors.image.context_visual_topk import ContextVisualTopK,ContextVisualTopKSpec
    module=ContextVisualTopK(ContextVisualTopKSpec(12,8,3))
    torch.nn.init.normal_(module.context_projection.weight,std=.2)
    x,q,r=torch.randn(17,12),torch.randn(6,12),torch.randn(6,12)
    a,b=module(x,question=q)['scores'],module(x,question=r)['scores']
    result=condition_path_stats(module,x,q,r,a,b)
    assert result['context_vector_delta_l2']>0 and not result['gelu_linearization_is_exact_identity']
    # With a genuinely linear activation, the shared condition changes only a
    # common logit offset. This is an architectural control, not the real model.
    linear=copy.deepcopy(module);linear.scorer[2]=torch.nn.Identity()
    a,b=linear(x,question=q),linear(x,question=r)
    torch.testing.assert_close((b['scores']-a['scores']).std(),torch.tensor(0.),atol=1e-9,rtol=0)
    assert torch.equal(a['indices'],b['indices'])


def plan():
    x=torch.randn(64,8)
    a=torch.arange(16)
    b=torch.arange(8,24)
    return plan_swaps(a.tolist(),b.tolist(),torch.linspace(1,0,64),torch.linspace(0,1,64),x,height=8,width=8,key='fixed')


def test_crossed_swaps_are_disjoint_exact_budget_and_reconstructable():
    p=plan()
    assert p['eligible'] and not p['candidate_is_verified_evidence']
    assert set(p['states'])==set(VARIANTS)
    assert all(len(v)==len(set(v))==16 for v in p['states'].values())
    assert not set(p['target_lost'])&set(p['control_lost'])
    assert not set(p['target_gained'])&set(p['control_gained'])
    base=set(range(8,24));true=set(range(16))
    for left,l in [('t',p['target_lost']),('c',p['control_lost'])]:
        for right,g in [('t',p['target_gained']),('c',p['control_gained'])]:
            assert set(p['states'][f'restore_{left}{right}'])==(base-set(g))|set(l)
            assert set(p['states'][f'remove_{left}{right}'])==(true-set(l))|set(g)


def test_control_planning_deterministic_and_inapplicability_explicit():
    x=torch.randn(64,8);a=list(range(16));b=list(range(8,24));s=torch.randn(64)
    kwargs=dict(height=8,width=8,key='sample')
    assert plan_swaps(a,b,s,-s,x,**kwargs)==plan_swaps(a,b,s,-s,x,**kwargs)
    no=plan_swaps(a,a,s,s,x,**kwargs)
    assert not no['eligible'] and no['swap_count']==0
    assert all(v==a for v in no['states'].values())


def test_evidence_unknown_not_false_and_verified_boxes_checked():
    kwargs=dict(height=4,width=4,image_fingerprint='image')
    assert evidence_coverage(None,[0],**kwargs)['status']=='pending_review'
    annotation={'status':'verified','source':'human','reviewer':'independent-reviewer','image_fingerprint':'image',
        'coordinate_space':'normalized_original_image','groups':[{'name':'label','boxes':[[0,0,.25,.25]]}]}
    covered=evidence_coverage(annotation,[0],**kwargs)
    assert covered['verified'] and covered['groups'][0]['retained_fraction']==1
    assert not evidence_coverage(annotation,[15],**kwargs)['groups'][0]['any_retained']
    for bad in ({**annotation,'source':'attention_pseudo_label'}, {**annotation,'reviewer':None},
        {**annotation,'image_fingerprint':'other'}, {**annotation,'groups':[{'boxes':[[0,0,2,1]]}]}):
        with pytest.raises(ValueError):evidence_coverage(bad,[0],**kwargs)
    assert sum(spatial_counts([0,1,15],4,4))==3


def test_cohort_one_question_per_image_and_no_outcome_selection():
    rows={f'{t}:{i}':{'task':t,'doc_id':i,'image_fingerprint':f'{t}-{i//2}','true_ids':[i]}
        for t in ('chartqa','ocrbench','ai2d') for i in range(8)}
    a=choose_images(rows,per_task=3)
    assert a==choose_images(rows,per_task=3)
    images=[r['image_fingerprint'] for values in a.values() for r in values]
    assert len(images)==len(set(images))==9


def test_annotation_extension_is_separate_pinned_and_never_overwrites(tmp_path,monkeypatch):
    from scripts.diagnostics import run_selector_mechanisms as runner
    from mllm_comp.core.serialization import atomic_write_json
    from mllm_comp.analysis.common.anchor7b_runtime import file_sha
    root=tmp_path/'run'
    atomic_write_json(root/'protocol.json',{'frozen':True})
    original=file_sha(root/'protocol.json')
    atomic_write_json(root/'cohort.json',{'tasks':{'chartqa':{
        'source_doc_ids':[42,43],'image_fingerprints':['image','other']}}})
    atomic_write_json(root/'eval/full/observe/chartqa/complete.json',{'run':str(tmp_path/'native')})
    annotation={'status':'verified','source':'human','reviewer':'reviewer','image_fingerprint':'image',
        'coordinate_space':'normalized_original_image','groups':[{'name':'label','boxes':[[0,0,.25,.25]]}]}
    export=tmp_path/'annotations.json'
    atomic_write_json(export,{'format':'selector_evidence_annotations_v1','rows':{'chartqa:42':annotation}})
    monkeypatch.setattr(runner,'validate',lambda path: {})
    monkeypatch.setattr(runner.parent,'trace_rows',lambda *args: {
        0:{'native_grid':[4,4],'true_selected_indices':[0],'wrong_question_selected_indices':[15]}})
    runner.review_report(root,export)
    extension=root/'semantic_extensions'/file_sha(export)/'coverage.json'
    result=runner.read(extension)
    assert result['verified_images']==1 and result['unreviewed_images']==1
    assert not result['rows'][0]['semantic_causality_established']
    assert result['rows'][0]['evidence_coverage']['true']['groups'][0]['retained_fraction']==1
    assert file_sha(root/'protocol.json')==original
    with pytest.raises(ValueError,match='Preserve'):runner.review_report(root,export)


@pytest.mark.parametrize('variant',VARIANTS)
def test_native_mechanism_binding_preserves_budget_cache_and_unknown_evidence(variant):
    from tests.unit.test_projector_comparison_evaluation import tiny_state
    from tests.unit.test_selector_condition_audit import manifest,SAMPLE
    from mllm_comp.analysis.common.selector_conditioning import request_key
    from mllm_comp.integrations.lmms_eval.request_context import sample_request
    from mllm_comp.compressors.comparison.selector_mechanism import QwenSelectorMechanismComparison
    data=manifest();data['rows'][request_key(SAMPLE)].update(task='chartqa',doc_id=0,source_doc_id=42,image_fingerprint='image')
    bundle,state,_,_=tiny_state()
    session=QwenSelectorMechanismComparison(bundle['model'],{'target_count':3,'condition_mode':'shuffled',
        'mechanism_variant':variant},condition_data=data).install()
    try:
        with torch.no_grad(),sample_request(SAMPLE):
            bundle['model'].generate(**state['prompt_inputs'],do_sample=False,max_new_tokens=3,min_new_tokens=3,pad_token_id=0)
        t=session.runtime_metadata()['selection']
        assert t['result_count']==3 and t['decode_position_verified']
        assert t['source_doc_id']==42 and t['mechanism_variant']==variant
        assert t['evidence_coverage']['true']['status']=='pending_review' and not t['semantic_mechanism_claimed']
        assert t['score_mechanism']['score_energy_closure_relative']<1e-8
    finally:session.uninstall()
