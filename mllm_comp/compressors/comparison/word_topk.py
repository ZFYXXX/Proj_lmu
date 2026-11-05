"""Independent word-level selector, native Qwen placeholder/M-RoPE/cache path."""
import hashlib
import json
from pathlib import Path

from .learned_topk import QwenLearnedTopKComparison
from mllm_comp.compressors.image.word_visual_topk import (
    WordVisualTopK,WordVisualTopKSpec,load_selector,prompt_word_question,clean_ids,
)


class QwenWordTopKComparison(QwenLearnedTopKComparison):
    def __init__(self,model,config,*,runtime=None,selector_module=None,phase="generate"):
        config=dict(config)
        self.visual_only=config.pop("visual_only",False)
        audit_path=config.pop("audit_conditions",None)
        audit_sha=config.pop("audit_conditions_sha256",None)
        match_width=config.pop("match_width",128)
        patterns=config.pop("template_token_ids",())
        if not isinstance(self.visual_only,bool):raise ValueError("visual_only must be boolean")
        self.audit_rows={}
        if audit_path is not None:
            raw=Path(audit_path).read_bytes()
            if hashlib.sha256(raw).hexdigest()!=audit_sha:raise ValueError("Question assignment manifest changed")
            self.audit_rows=json.loads(raw)["rows"]
        device=model.get_input_embeddings().weight.device
        if selector_module is None:
            if config.get("checkpoint_path"):
                if runtime is None or not config.get("identity_path"):raise ValueError("Word checkpoint requires pinned runtime")
                identity=json.loads(Path(config["identity_path"]).read_text())
                if Path(runtime.model_path_or_id).resolve()!=Path(identity["checkpoint"]).resolve():raise ValueError("Wrong word-selector backbone")
                selector_module=load_selector(config["checkpoint_path"],base_identity=identity["identity_sha256"],device=device)
            else:
                selector_module=WordVisualTopK(WordVisualTopKSpec(model.get_input_embeddings().embedding_dim,
                    width=config.get("width",256),target_count=config.get("target_count",128),seed=config.get("seed",42),
                    match_width=match_width,template_token_ids=patterns))
        if not isinstance(selector_module,WordVisualTopK) or selector_module.spec.match_width!=match_width:
            raise ValueError("Wrong word selector/matching width")
        for key,default in (("width",256),("target_count",128),("seed",42)):
            if getattr(selector_module.spec,key)!=config.get(key,default):raise ValueError("Word selector config mismatch")
        super().__init__(model,config,runtime=runtime,selector_module=selector_module,phase=phase)
        self.selector.freeze_prior()  # parent soft-mode flag must not unfreeze old weights
        self.method="word_visual_topk"

    def prepare_source_inputs(self,call):
        super().prepare_source_inputs(call)
        source=self.source.get()
        source["question"],source["question_audit"]=prompt_word_question(self.model,call,self.selector.spec.template_token_ids)

    def prepare_qwen_embeddings(self,call,hidden):
        import torch
        from mllm_comp.integrations.lmms_eval.request_context import CURRENT_SAMPLE_ID
        from mllm_comp.analysis.common.selector_conditioning import ids_sha,request_key
        visual_ids=self.incoming.get().vision[0].nonzero(as_tuple=True)[0]
        features=hidden[0].index_select(0,visual_ids)
        source=self.source.get()
        stats={**source["question_audit"],"trained":bool(self.selector_config.get("checkpoint_path")),
            "checkpoint_path":self.selector_config.get("checkpoint_path"),"reads_question":True,
            "context_mode":"question","word_matching":True,"visual_only_control":self.visual_only,
            "semantic_evidence_verified":False}
        output=self.select_word_tokens(features, source["question"])
        stats.update(output["word_diagnostics"])
        if self.audit_rows:
            key=request_key(CURRENT_SAMPLE_ID.get())
            if key not in self.audit_rows:raise ValueError("Unknown word-audit request")
            row=self.audit_rows[key]
            if ids_sha(row["true_ids"])!=source["question_audit"]["raw_question_ids_sha256"]:
                raise ValueError("Original user prompt changed in word audit")
            if set(row)&{"answer","answers","target","label","labels","prediction","score"}:raise ValueError("Answer in question assignment")
            audits={}
            with torch.no_grad():
                for name,field in (("shuffled","donor_ids"),("generic","generic_ids")):
                    ids=clean_ids(row[field],self.selector.spec.template_token_ids)
                    if any(not 0<=i<self.model.config.vocab_size or i in {151643,151644,151645,
                        self.model.config.image_token_id,self.model.config.vision_start_token_id,self.model.config.vision_end_token_id} for i in ids):
                        raise ValueError("Invalid word-audit token IDs")
                    q=self.model.get_input_embeddings()(torch.tensor(ids,device=features.device,dtype=torch.long)).detach()
                    other=self.select_word_tokens(features, q)
                    a,b=output["indices"],other["indices"]
                    delta=other["scores"]-output["scores"]
                    audits[name]={"question_changed_after_cleaning":ids_sha(ids)!=stats["question_ids_sha256"],
                        "condition_ids_sha256":ids_sha(ids),"selected_indices":b.tolist(),
                        "exchanged_tokens":len(a)-int(torch.isin(a,b).sum()),
                        "selection_overlap":int(torch.isin(a,b).sum())/len(a),
                        "centered_score_change_rms":float((delta-delta.mean()).square().mean().sqrt()),
                        "answers_generated_for_other_question":False}
            stats["question_selection_audit"]=audits
        if self.soft_training:
            hidden=hidden.clone().index_copy(1,visual_ids,output["tokens"][None])
            call.update(inputs_embeds=hidden,input_ids=None)
            self.soft_output=output
            self.last_selection={"source_count":len(features),"result_count":len(features),
                "soft_mass":float(output["soft_weights"].detach().sum()),"selected_indices":output["indices"].tolist(),
                "training_mode":"soft_full_sequence",**stats}
            return hidden
        return self.compact_word_selection(call,hidden,visual_ids,output,stats)

    def select_word_tokens(self, features, question):
        if self.retention_ratio is not None:
            raise ValueError("This word selector has no ratio selection policy; use the protected binding")
        return self.selector(features, question=question, soft=self.soft_training, visual_only=self.visual_only)

    def compact_word_selection(self,call,hidden,visual_ids,output,stats):
        """Extension point for representation recovery; default remains a gather."""
        return self.compact_selection(call,hidden,output["indices"],{**stats,"training_mode":"hard_compact"})


def create_executor(*,model,stage,runtime=None,execution_phase="generate",**kwargs):
    return QwenWordTopKComparison(model,stage.config,runtime=runtime,phase=execution_phase)


def register_contracts(registry):
    from dataclasses import replace
    from mllm_comp.compressors.contracts import AlgorithmDescriptor,ModalityVariantDescriptor
    from mllm_comp.core.vocabulary import CompressionMethod,Placement,TokenOperation,Modality,ExecutionPhase
    from mllm_comp.core.support import SupportUnit,SupportUnitId
    name="word_visual_topk"
    registry.register_algorithm(AlgorithmDescriptor(name=name,version="v1.0.0",method=CompressionMethod.TOKEN_COMPRESSION,
        kernel=WordVisualTopK,supported_placements=frozenset({Placement.PROJECTOR_LEVEL}),
        operations=frozenset({TokenOperation.DROP}),metadata={"claim":"Learned word-level residual on frozen visual prior; no semantic-GT claim"}))
    registry.register_variant(ModalityVariantDescriptor(name=f"{name}.image",algorithm=name,
        target_modalities=frozenset({Modality.IMAGE}),conditioning_modalities=frozenset({Modality.TEXT})))
    source=next(b for b in registry.bindings() if b.name=="context_visual_topk.image.qwen25_vl")
    binding=f"{name}.image.qwen25_vl"
    registry.register_binding(replace(source,name=binding,variant=f"{name}.image",executor_factory=create_executor,
        conflicts=frozenset(b.name for b in registry.bindings() if b.backbone=="qwen25_vl")))
    registry.extend_support(SupportUnit(identity=SupportUnitId(algorithm=name,variant=f"{name}.image",
        backbone="qwen25_vl",target_modality=Modality.IMAGE,phase=phase),binding=binding)
        for phase in (ExecutionPhase.FORWARD,ExecutionPhase.GENERATE))
