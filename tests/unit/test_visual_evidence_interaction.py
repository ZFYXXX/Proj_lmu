import json
from itertools import combinations

import pytest

from mllm_comp.analysis.common.evidence_interaction import (
    bootstrap, condition_id, farthest_indices, holm, interactions, materialize_plan, plan_regions,
)
from mllm_comp.analysis.common.evidence_cohort import select_manifest_entries


@pytest.mark.parametrize("height,width",[(16,16),(18,30),(31,19)])
@pytest.mark.parametrize("budget",[64,128])
def test_factorial_exact_budget_original_order_and_only_intended_replacement(height,width,budget):
    import torch
    gen = torch.Generator().manual_seed(4)
    x = torch.randn(height*width,12,generator=gen)
    s = torch.rand(height*width,generator=gen)
    regions = plan_regions(features=x,scores=s,height=height,width=width,sample_id="a",seed=11)
    assert regions == plan_regions(features=x,scores=s,height=height,width=width,sample_id="a",seed=11)
    assert regions != plan_regions(features=x,scores=s,height=height,width=width,sample_id="a",seed=23)
    plan = materialize_plan(regions,features=x,budget=budget)
    assert len(plan["states"]) == 37
    base = set(plan["states"][0]["retained_indices"])
    for state in plan["states"]:
        keep = state["retained_indices"]
        assert keep == sorted(set(keep)) and len(keep) == budget
        added = set(i for slot in state["changed_slots"] for i in regions["candidate_regions"][slot])
        removed = set(i for slot in state["changed_slots"] for i in regions["control_regions"][slot])
        assert set(keep) - base == added
        assert base - set(keep) == removed
    flat = [i for block in regions["candidate_regions"]+regions["control_regions"] for i in block]
    assert len(flat) == len(set(flat)) == 64
    assert not set(flat) & set(plan["core_indices"])
    # Increasing budget must extend the same deterministic background, not
    # change candidates, controls, or the original feature values.
    larger = materialize_plan(regions,features=x,budget=128)
    assert set(plan["core_indices"]) <= set(larger["core_indices"])


def test_invalid_members_and_unsupported_geometry_fail_explicitly():
    import torch
    x,s = torch.ones(256,4),torch.ones(256)
    r = plan_regions(features=x,scores=s,height=16,width=16,sample_id="x",seed=11)
    r["control_regions"][0] = r["candidate_regions"][0]
    with pytest.raises(ValueError,match="Overlapping"):
        materialize_plan(r,features=x,budget=128)
    with pytest.raises(ValueError):
        plan_regions(features=torch.ones(64,4),scores=torch.ones(64),height=2,width=32,sample_id="x",seed=11)
    assert farthest_indices(torch.ones(20,3),4) == [0,1,2,3]


def test_factorial_interactions_and_missing_states():
    rows=[]
    for changed in [()] + [(i,) for i in range(8)] + list(combinations(range(8),2)):
        joint = changed == (0,1)
        rows.append({"condition_id":condition_id(changed),"primary_task_score":float(joint),"candidate_answer_nll":2.-float(joint)})
    values=interactions(rows)
    assert len(values)==28
    assert [sum(v["kind"]==k for v in values) for k in ["question","random","mixed"]] == [6,6,16]
    assert values[0]["score_interaction"] == values[0]["logprob_interaction"] == 1
    assert values[0]["joint_only_improvement"]
    with pytest.raises(ValueError):interactions(rows[:-1])
    with pytest.raises(ValueError):interactions(rows+[rows[0]])


def test_statistics_holm_and_image_bootstrap():
    assert holm([.01,.03,.02,.5]) == pytest.approx([.04,.06,.06,.5])
    positive=bootstrap([.2]*100,repeats=10000)
    zero=bootstrap([0.]*100,repeats=10000)
    assert positive["ci95"] == pytest.approx([.2,.2])
    assert positive["one_sided_p"] < .001
    assert zero["one_sided_p"] == 1


def test_manifest_filter_and_second_question_reuse_primary(tmp_path):
    path=tmp_path/"cohort.json"
    path.write_text(json.dumps({"entries":[{"task":"t","doc_id":5,"sample_id":"t:5","primary_sample_id":"t:5",
                       "split":"confirmation","shard":0,"question_review":True,"second_doc_id":8}]}))
    selected,_=select_manifest_entries(path,task_name="t",stage="question_review",shard=0)
    assert selected[8]["sample_id"]=="t:8"
    assert selected[8]["primary_sample_id"]=="t:5"
    other,_=select_manifest_entries(path,task_name="t",stage="confirmation",shard=1)
    assert other=={}


def test_qwen_tiny_cached_and_cache_free_decode_match():
    import torch
    from transformers import Qwen2_5_VLTextConfig
    from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VLTextModel
    from mllm_comp.analysis.probes.visual_evidence_interaction import cached_decode
    from mllm_comp.analysis.probes.visual_token_phenomena import _run_decoder_suffix
    from mllm_comp.analysis.common.visual_token_generation import greedy_decode_embeddings
    torch.manual_seed(3)
    cfg=Qwen2_5_VLTextConfig(vocab_size=32,hidden_size=24,intermediate_size=48,num_hidden_layers=2,
                          num_attention_heads=4,num_key_value_heads=2,rope_scaling={"type":"mrope","mrope_section":[1,1,1]})
    cfg._attn_implementation="eager"
    model=Qwen2_5_VLTextModel(cfg).eval()
    head=torch.nn.Linear(24,32,bias=False).eval()
    hidden=model.embed_tokens(torch.tensor([[2,3,4]]))
    positions=torch.tensor([[[0,2,5]],[[0,3,5]],[[0,4,5]]])
    cached=cached_decode(llm=model,head=head,embedding=model.embed_tokens,hidden=hidden,positions=positions,eos_ids=[31],max_new=5)
    plain=greedy_decode_embeddings(prompt_hidden=hidden,prompt_position_ids=positions,token_embedding=model.embed_tokens,
          forward_logits=lambda h,p:_run_decoder_suffix(llm=model,lm_head=head,hidden=h,position_ids=p,start_layer=0,last_token_only=True),
          eos_token_ids=[31],max_new_tokens=5)
    assert cached["generated_token_ids"]==plain["generated_token_ids"]
    assert cached["first_appended_token_position"]==6


def test_question_span_handles_bpe_punctuation_newline_boundary():
    import torch
    from transformers import AutoTokenizer
    from mllm_comp.analysis.probes.visual_evidence_interaction import question_positions
    checkpoint="/home/ZhouFengyu/fastv-repro/models/Qwen2.5-VL-3B-Instruct"
    if not __import__("pathlib").Path(checkpoint).exists():pytest.skip("Local tokenizer unavailable")
    tokenizer=AutoTokenizer.from_pretrained(checkpoint,local_files_only=True)
    q="How many colors are used to represent the bar graph?"
    text=tokenizer.apply_chat_template([{"role":"user","content":[{"type":"image"},{"type":"text","text":q+"\nAnswer the question with a single word."}]}],tokenize=False,add_generation_prompt=True)
    ids=tokenizer.encode(text,add_special_tokens=False)
    selected=question_positions(tokenizer,torch.tensor([ids]),q)
    decoded=tokenizer.decode([ids[i] for i in selected],clean_up_tokenization_spaces=False)
    assert q in decoded
    assert "Answer the question" not in decoded
    original="what COLOR is the sign?"
    native=original.capitalize()
    p=tokenizer.apply_chat_template([{"role":"user","content":native+"\nAnswer the question using a single word or phrase."}],tokenize=False,add_generation_prompt=True)
    ids=tokenizer.encode(p,add_special_tokens=False)
    selected=question_positions(tokenizer,torch.tensor([ids]),native)
    assert native in tokenizer.decode([ids[i] for i in selected],clean_up_tokenization_spaces=False)
