"""Run native tests under the isolated Transformers5.19 environment."""
import copy
import json
from types import SimpleNamespace

import pytest
import torch

from mllm_comp.analysis.common.qwen35_transfer import (
    EXTERNAL,METHOD_KIND,create_head,compact_prompt,question_embeddings,save_head,load_head,
)
from mllm_comp.compressors.comparison.qwen35_topk import Qwen35TransferredProjector


class Tokenizer:
    def encode(self,text,**kwargs):
        return [100,12,13] if text=="<|im_start|>user\n" else [90,91,92]
    def convert_tokens_to_ids(self,text):return 101


def tiny():
    transformers=pytest.importorskip("transformers")
    if not hasattr(transformers,"Qwen3_5Config"):pytest.skip("Requires isolated Transformers5.19")
    torch.manual_seed(123)
    text=dict(vocab_size=128,hidden_size=32,intermediate_size=48,num_hidden_layers=4,
        num_attention_heads=2,num_key_value_heads=1,head_dim=16,linear_conv_kernel_dim=4,
        linear_key_head_dim=8,linear_value_head_dim=8,linear_num_key_heads=2,linear_num_value_heads=2,
        max_position_embeddings=128,pad_token_id=0,eos_token_id=2,
        rope_parameters={"rope_type":"default","rope_theta":10000.,"partial_rotary_factor":1.,
            "mrope_section":[2,3,3],"mrope_interleaved":True})
    vision=dict(depth=2,hidden_size=16,intermediate_size=24,num_heads=2,patch_size=2,
        temporal_patch_size=1,spatial_merge_size=2,out_hidden_size=32,num_position_embeddings=16)
    cfg=transformers.Qwen3_5Config(text_config=text,vision_config=vision,image_token_id=10,
        video_token_id=11,vision_start_token_id=7,vision_end_token_id=8)
    cfg._attn_implementation="eager"
    model=transformers.Qwen3_5ForConditionalGeneration(cfg).eval()
    ids=torch.tensor([[100,12,13,7,*([10]*6),8,23,24,101,100,14,13]])
    inputs=dict(input_ids=ids,attention_mask=torch.ones_like(ids),pixel_values=torch.randn(24,12),
        image_grid_thw=torch.tensor([[1,4,6]]),mm_token_type_ids=(ids==10).long())
    return model,inputs


def head(method):
    from mllm_comp.compressors.image.learned_visual_topk import LearnedVisualTopK,LearnedVisualTopKSpec
    from mllm_comp.compressors.image.context_visual_topk import ContextVisualTopK,ContextVisualTopKSpec
    from mllm_comp.compressors.image.word_visual_topk import WordVisualTopK,WordVisualTopKSpec
    if method=="old_mlp":return create_head("mlp",32)
    if method=="visual_selector":return LearnedVisualTopK(LearnedVisualTopKSpec(32,target_count=3))
    if method=="question_selector":return ContextVisualTopK(ContextVisualTopKSpec(32,target_count=3))
    if method=="protected_selector":return WordVisualTopK(WordVisualTopKSpec(32,target_count=3))
    return None


@pytest.mark.parametrize("method",[*METHOD_KIND,*EXTERNAL])
def test_native_generate_compaction_hybrid_cache_and_restore(method):
    model,inputs=tiny()
    before=copy.deepcopy(inputs)
    session=Qwen35TransferredProjector(model,{"method":method,"target_count":3},
        module=head(method),tokenizer=Tokenizer()).install()
    try:
        with torch.no_grad():out=model.generate(**inputs,max_new_tokens=3,min_new_tokens=3,do_sample=False,pad_token_id=0,eos_token_id=2)
        record=session.runtime_metadata()["selection"]
        assert out.shape[1]==inputs["input_ids"].shape[1]+3
        assert torch.equal(out[:,:inputs["input_ids"].shape[1]],inputs["input_ids"])
        assert record["source_count"]==6 and record["result_count"]==3
        assert record["decode_position_verified"] and record["observed_visual_tokens_per_decoder_layer"]==[3]*4
        assert record["text_mrope_preserved"] and record["physical_sequence_compacted"]
        if method in EXTERNAL:assert not record["new_base_training"]
    finally:session.uninstall()
    assert all(torch.equal(before[k],inputs[k]) for k in before)
    assert "generate" not in model.__dict__


@pytest.mark.parametrize("method",["old_mlp","visual_selector","question_selector","protected_selector"])
def test_dense_identity_and_no_extra_vision_encoding(method):
    model,inputs=tiny()
    with torch.no_grad():expected=model.generate(**inputs,max_new_tokens=3,min_new_tokens=3,do_sample=False,pad_token_id=0,eos_token_id=2)
    calls=[]
    hook=model.model.visual.register_forward_hook(lambda *args:calls.append(1))
    session=Qwen35TransferredProjector(model,{"method":method,"target_count":128},
        module=create_head("mlp",32),tokenizer=Tokenizer()).install()
    try:
        with torch.no_grad():actual=model.generate(**inputs,max_new_tokens=3,min_new_tokens=3,do_sample=False,pad_token_id=0,eos_token_id=2)
        assert torch.equal(actual,expected) and calls==[1]
    finally:session.uninstall();hook.remove()


def test_question_extraction_uses_new_tokenizer_and_excludes_teacher_answer():
    model,inputs=tiny()
    q,audit=question_embeddings(model,Tokenizer(),inputs)
    assert q.shape==(2,32)
    other={**inputs,"input_ids":torch.cat((inputs["input_ids"],torch.tensor([[29,30]])),1)}
    labels=torch.full_like(other["input_ids"],-100);labels[:,-2:]=other["input_ids"][:,-2:]
    other["labels"]=labels
    actual,new=question_embeddings(model,Tokenizer(),other)
    assert audit==new and torch.equal(q,actual)


def test_official_prunesid_order_survives_gather():
    model,inputs=tiny()
    with torch.no_grad():features,=model.get_image_features(inputs["pixel_values"],inputs["image_grid_thw"],return_dict=True).pooler_output
    local=torch.tensor([4,1,3])
    prepared,record,gather=compact_prompt(model,inputs,features,{"tokens":features[local],"indices":local,"official_order":True})
    source=(inputs["input_ids"][0]==10).nonzero(as_tuple=True)[0]
    kept=gather[(prepared["input_ids"][0]==10)]
    assert torch.equal(kept,source[local]) and record["selected_indices"]==[4,1,3]


@pytest.mark.parametrize("kind",["mlp","visual","question","word"])
def test_head_roundtrip_and_wrong_base_rejection(tmp_path,kind):
    identity={"hidden_size":32,"identity_sha256":"base35"}
    visual=create_head("visual",32)
    module=create_head(kind,32,visual=visual)
    save_head(module,tmp_path/"head",kind=kind,identity=identity,provenance={"step":320})
    actual=load_head(tmp_path/"head",identity=identity,device="cpu")
    assert all(torch.equal(v,actual.state_dict()[k]) for k,v in module.state_dict().items())
    with pytest.raises(ValueError):load_head(tmp_path/"head",identity={**identity,"identity_sha256":"base25"},device="cpu")


def test_noise_excluded_and_registry_generate_only():
    from mllm_comp.compressors.multimodal_registry import builtin_multimodal_compression_contracts
    assert "noise_selector" not in METHOD_KIND and "vp_noise" not in EXTERNAL
    with pytest.raises(ValueError):create_head("vp_noise",32)
    binding=builtin_multimodal_compression_contracts().binding("qwen35_transferred_projector.image.qwen3_5")
    assert {p.value for p in binding.supported_phases}=={"generate"}
