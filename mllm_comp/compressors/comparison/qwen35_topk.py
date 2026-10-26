"""Native Qwen3.5 hybrid-cache generation for independently retrained heads."""
from contextvars import ContextVar
import json
from pathlib import Path
import types

import torch

from mllm_comp.analysis.common.qwen35_transfer import (
    METHOD_KIND, EXTERNAL, compact_prompt, load_head, question_embeddings, select_tokens,
)


class Qwen35TransferredProjector:
    def __init__(self, model, config, *, runtime=None, module=None, tokenizer=None):
        config=dict(config)
        allowed={"method","checkpoint_path","identity_path","target_count"}
        if set(config)-allowed: raise ValueError("Unknown Qwen3.5 transfer configuration")
        self.method=config["method"]
        if self.method not in (*METHOD_KIND,*EXTERNAL): raise ValueError("Unknown Qwen3.5 transfer method")
        if model.config.model_type!="qwen3_5": raise ValueError("Never bind Qwen3.5 transfer to another backbone")
        self.target=config.get("target_count",128)
        if isinstance(self.target,bool) or not isinstance(self.target,int) or self.target<1: raise ValueError("Invalid budget")
        self.model=model
        self.config=config
        self.tokenizer=tokenizer or getattr(getattr(runtime,"processor",None),"tokenizer",None)
        if self.tokenizer is None: raise ValueError("Native Qwen3.5 tokenizer required")
        if module is None and self.method not in EXTERNAL:
            identity=json.loads(Path(config["identity_path"]).read_text())
            if Path(runtime.model_path_or_id).resolve()!=Path(identity["checkpoint"]).resolve(): raise ValueError("Wrong runtime base")
            checkpoint=json.loads((Path(config["checkpoint_path"])/"head_config.json").read_text())
            if checkpoint["kind"]!=METHOD_KIND[self.method]: raise ValueError("Wrong trained head kind")
            if checkpoint["provenance"].get("step")!=320: raise ValueError("Only fixed step320 head allowed in full evaluation")
            module=load_head(config["checkpoint_path"],identity=identity,device=model.device)
        if module is not None and module.spec.hidden_size!=model.config.text_config.hidden_size: raise ValueError("Head dimension mismatch")
        self.module=module.eval().requires_grad_(False) if module is not None else None
        self.record=ContextVar(f"qwen35_transfer_{id(self)}",default=None)
        self.installed=False

    def install(self):
        if self.installed:return self
        self.original=self.model.generate
        self.had_generate="generate" in self.model.__dict__
        owner=self
        def generate(_model,*args,**kwargs):
            if args or kwargs.get("pixel_values") is None: return owner.original(*args,**kwargs)
            if kwargs.get("num_beams",1)!=1 or kwargs.get("do_sample",False): raise ValueError("Greedy single-beam transfer only")
            original_ids=kwargs["input_ids"]
            old_delta=owner.model.model.rope_deltas
            owner.record.set(None)
            try:
                with torch.no_grad():
                    from contextlib import nullcontext
                    from mllm_comp.backbones.qwen3_5.vision_attention import (
                        capture_qwen3_5_premerger_features,capture_qwen3_5_final_vision_attention,
                        pool_premerger_features_to_merger_tokens,pool_patch_statistics_to_merger_tokens)
                    context=(capture_qwen3_5_premerger_features(owner.model.model.visual) if owner.method=="mmtok" else
                        capture_qwen3_5_final_vision_attention(owner.model.model.visual) if owner.method=="bifocal_stage1" else nullcontext(None))
                    with context as captured:
                        groups=owner.model.get_image_features(kwargs["pixel_values"],kwargs["image_grid_thw"],return_dict=True).pooler_output
                    features,=groups
                    q,audit=(None,{"reads_question":False})
                    if owner.method in {"question_selector","protected_selector","eadp","mmtok","bifocal_stage1"}:
                        q,audit=question_embeddings(owner.model,owner.tokenizer,kwargs,wordwise=owner.method=="protected_selector")
                        audit["reads_question"]=True
                    grid=tuple(int(x)//owner.model.config.vision_config.spatial_merge_size for x in kwargs["image_grid_thw"][0,1:])
                    if owner.method in EXTERNAL:
                        k=min(owner.target,len(features))
                        if owner.method=="eadp":
                            from .eadp import select_eadp_indices,normalize_eadp_config
                            selections,stats=select_eadp_indices(features[None],q,patches_per_side=grid[1],grid_height=grid[0],
                                cfg=normalize_eadp_config({"tokens_per_crop":k,"alpha":.5,"beta":2.}))
                            indices=selections[0];output={"indices":indices,"tokens":features[indices]}
                        elif owner.method=="prunesid":
                            from .prunesid import select_prunesid_indices
                            selections,stats=select_prunesid_indices(features[None],cfg={"tokens_per_crop":k})
                            indices=selections[0];output={"indices":indices,"tokens":features[indices],"official_order":True}
                        elif owner.method=="mmtok":
                            from mllm_comp.compressors.image.mmtok import MMTokSpec,mmtok_image_tokens
                            if len(captured)!=1:raise ValueError("Missing native pre-merger features")
                            raw,=pool_premerger_features_to_merger_tokens(captured[0],kwargs["image_grid_thw"],spatial_merge_size=2)
                            result=mmtok_image_tokens(features,raw,q,spec=MMTokSpec(target_count=k))
                            output={"tokens":result.tokens,"indices":result.indices};stats=dict(result.metadata)
                        else:
                            from mllm_comp.compressors.image.bifocal import BifocalSpec,bifocal_image_tokens
                            if len(captured)!=1:raise ValueError("Missing native vision attention")
                            received,_=pool_patch_statistics_to_merger_tokens(captured[0],merged_token_count=len(features),spatial_merge_size=2)
                            result=bifocal_image_tokens(features,{"mass":received},grid_height=grid[0],grid_width=grid[1],
                                word_embeddings=q,spec=BifocalSpec(target_count=k))
                            output={"tokens":result.tokens,"indices":result.indices};stats=dict(result.metadata)
                        stats.update(trained=False,controlled_adaptation=True,official_paper_reproduction=False)
                    else:
                        output,stats=select_tokens(features,method=owner.method,module=owner.module,question=q,grid=grid,target=owner.target)
                    if len(output["indices"])!=min(owner.target,len(features)):raise ValueError("Transfer output budget failed")
                    prepared,record,_=compact_prompt(owner.model,kwargs,features,output)
                    record.update(stats,**audit,method=owner.method,new_base_training=owner.method not in EXTERNAL,
                        observed_visual_tokens_per_decoder_layer=[None]*len(owner.model.model.language_model.layers))
                    owner.record.set(record)
                prefill=True
                def check_input(_module,_args,call):
                    nonlocal prefill
                    positions=call.get("position_ids")
                    hidden=call.get("inputs_embeds")
                    if prefill:
                        prefill=False
                        if hidden.shape[1]!=record["output_prompt_tokens"]:raise ValueError("Prefill length not physically compacted")
                    elif record["decode_position_verified"] is None:
                        seen=positions[-3:,:,0]
                        if not bool((seen==record["expected_first_decode_position"]).all()):
                            raise ValueError("Qwen3.5 first decode M-RoPE mismatch")
                        record.update(decode_position_verified=True,first_decode_position=seen.flatten().tolist())
                hooks=[owner.model.model.language_model.register_forward_pre_hook(check_input,with_kwargs=True)]
                def layer_hook(i):
                    def observe(_layer,_args,call):
                        # Prefill has the full compact prompt; decode has one row.
                        h=call.get("hidden_states",_args[0] if _args else None)
                        if h is not None and h.shape[1]==record["output_prompt_tokens"]:
                            record["observed_visual_tokens_per_decoder_layer"][i]=record["result_count"]
                    return observe
                hooks += [layer.register_forward_pre_hook(layer_hook(i),with_kwargs=True)
                          for i,layer in enumerate(owner.model.model.language_model.layers)]
                try: raw=owner.original(**prepared)
                finally:
                    for hook in hooks:hook.remove()
                compact_length=prepared["input_ids"].shape[1]
                sequence=getattr(raw,"sequences",raw)
                restored=torch.cat((original_ids,sequence[:,compact_length:]),dim=-1)
                if hasattr(raw,"sequences"):raw.sequences=restored
                else:raw=restored
                if record["observed_visual_tokens_per_decoder_layer"]!=[record["result_count"]]*len(hooks[1:]):
                    raise ValueError("Incomplete hybrid decoder prefill observation")
                return raw
            finally: owner.model.model.rope_deltas=old_delta
        self.model.generate=types.MethodType(generate,self.model)
        self.installed=True
        return self

    def uninstall(self):
        if self.installed:
            if self.had_generate:self.model.generate=self.original
            else:delattr(self.model,"generate")
            self.installed=False

    def clear_diagnostic_records(self):self.record.set(None)

    def runtime_metadata(self):
        return {"backbone":"qwen3_5","method":self.method,"installed":self.installed,
            "selection":self.record.get(),"layout_delta_records":[self.record.get()] if self.record.get() else [],
            "budget_policy":{"fixed_tokens":self.target},"checkpoint_path":self.config.get("checkpoint_path"),
            "training_physical_compression":False,"hybrid_attention_native_cache":True}


def create_executor(*,model,stage,runtime=None,**kwargs):
    return Qwen35TransferredProjector(model,stage.config,runtime=runtime)


def register_contracts(registry):
    from mllm_comp.compressors.contracts import AlgorithmDescriptor,ModalityVariantDescriptor,BackbonePlacementBinding,BindingTarget
    from mllm_comp.core.vocabulary import CompressionMethod,Placement,TokenOperation,Modality,ExecutionPhase
    from mllm_comp.core.support import SupportUnit,SupportUnitId
    name="qwen35_transferred_projector"
    registry.register_algorithm(AlgorithmDescriptor(name=name,version="v1.0.0",method=CompressionMethod.TOKEN_COMPRESSION,
        kernel=select_tokens,supported_placements=frozenset({Placement.PROJECTOR_LEVEL}),
        operations=frozenset({TokenOperation.DROP,TokenOperation.MERGE}),
        metadata={"claim":"Same algorithms, independently retrained small Qwen3.5-9B heads; not 2.5 weight transfer"}))
    registry.register_variant(ModalityVariantDescriptor(name=f"{name}.image",algorithm=name,
        target_modalities=frozenset({Modality.IMAGE}),conditioning_modalities=frozenset({Modality.TEXT})))
    binding=f"{name}.image.qwen3_5"
    registry.register_binding(BackbonePlacementBinding(name=binding,backbone="qwen3_5",variant=f"{name}.image",
        targets=(BindingTarget(name="transferred_projector",placement=Placement.PROJECTOR_LEVEL,
            target_site="post_merger_pre_llm_scatter",target_modalities=frozenset({Modality.IMAGE}),
            required_handles=frozenset({"vision_projector","llm"})),),
        supported_phases=frozenset({ExecutionPhase.GENERATE}),executor_factory=create_executor,
        legalizer="qwen35.native_sparse_mrope_hybrid_cache",exclusive_runtime_slot="image_comparison",
        conflicts=frozenset(b.name for b in registry.bindings() if b.backbone=="qwen3_5")))
    registry.extend_support([SupportUnit(identity=SupportUnitId(algorithm=name,variant=f"{name}.image",
        backbone="qwen3_5",target_modality=Modality.IMAGE,phase=ExecutionPhase.GENERATE),binding=binding)])
