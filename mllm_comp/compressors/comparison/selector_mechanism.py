"""Instrument scorer mechanics and execute fixed-budget token mediation."""
import hashlib
import json
from pathlib import Path

from .selector_condition import QwenSelectorConditionComparison
from mllm_comp.analysis.common.selector_mechanisms import (
    VARIANTS, ranking_stats, condition_path_stats, plan_swaps, spatial_counts, evidence_coverage,
)


class QwenSelectorMechanismComparison(QwenSelectorConditionComparison):
    def __init__(self,model,config,**kwargs):
        config=dict(config)
        self.variant=config.pop('mechanism_variant','observe')
        self.max_swap=config.pop('max_swap',8)
        evidence_path=config.pop('evidence_manifest',None)
        evidence_sha=config.pop('evidence_manifest_sha256',None)
        if self.variant not in VARIANTS or isinstance(self.max_swap,bool) or not isinstance(self.max_swap,int) or self.max_swap<1:
            raise ValueError('Invalid mechanism variant/swap budget')
        self.annotations={}
        if evidence_path is not None:
            raw=Path(evidence_path).read_bytes()
            if hashlib.sha256(raw).hexdigest()!=evidence_sha:raise ValueError('Evidence manifest changed')
            document=json.loads(raw)
            if document.get('format')!='selector_evidence_annotations_v1':raise ValueError('Unknown evidence manifest')
            self.annotations=document['rows']
        super().__init__(model,config,**kwargs)
        if self.condition_mode!='shuffled':raise ValueError('Mechanism pilot fixes the matched wrong-question condition')
        self.method='selector_mechanism_audit'

    def prepare_qwen_embeddings(self,call,hidden):
        import torch
        from mllm_comp.integrations.lmms_eval.request_context import CURRENT_SAMPLE_ID
        from mllm_comp.analysis.common.selector_conditioning import request_key
        visual_ids=self.incoming.get().vision[0].nonzero(as_tuple=True)[0]
        features=hidden[0].index_select(0,visual_ids)
        source=self.source.get()
        row=self.condition_data['rows'][request_key(CURRENT_SAMPLE_ID.get())]
        merge=self.model.model.visual.spatial_merge_size
        height,width=(int(v)//merge for v in source['grid'][0,1:])
        with torch.no_grad():
            a=self.selector(features,question=source['true_question'])
            b=self.selector(features,question=source['question'])
            score_stats=ranking_stats(a['scores'],b['scores'],len(a['indices']))
            path_stats=condition_path_stats(self.selector,features,source['true_question'],source['question'],a['scores'],b['scores'])
            plan=plan_swaps(a['indices'].tolist(),b['indices'].tolist(),a['scores'],b['scores'],features,
                height=height,width=width,key=f"{row['task']}:{row.get('source_doc_id',row['doc_id'])}",max_swap=self.max_swap)
        selected=torch.tensor(plan['states'][self.variant],device=features.device,dtype=torch.long)
        intersection=int(torch.isin(selected,a['indices']).sum())
        annotation=self.annotations.get(f"{row['task']}:{row.get('source_doc_id',row['doc_id'])}")
        coverage={name:evidence_coverage(annotation,indices,height=height,width=width,image_fingerprint=row['image_fingerprint'])
            for name,indices in [('true',a['indices'].tolist()),('wrong_question',b['indices'].tolist()),('intervention',selected.tolist())]}
        stats={**source['question_audit'],**source['condition_audit'],
            'trained':bool(self.selector_config.get('checkpoint_path')),'checkpoint_path':self.selector_config.get('checkpoint_path'),
            'reads_question':True,'context_mode':'question','training_mode':'hard_compact',
            'mechanism_variant':self.variant,'source_doc_id':row.get('source_doc_id',row['doc_id']),
            'true_selected_indices':a['indices'].tolist(),'wrong_question_selected_indices':b['indices'].tolist(),
            'selection_identical':bool(torch.equal(selected,a['indices'])),
            'selection_overlap_fraction':intersection/len(selected),
            'selection_jaccard':intersection/(2*len(selected)-intersection),
            'score_mechanism':score_stats,'condition_path':path_stats,
            'swap_plan':{k:v for k,v in plan.items() if k!='states'},
            'swap_plan_sha256':hashlib.sha256(json.dumps(plan,sort_keys=True).encode()).hexdigest(),
            'spatial_bins':{'bins_per_axis':8,**{name:spatial_counts(indices,height,width) for name,indices in
                [('true',a['indices'].tolist()),('wrong_question',b['indices'].tolist()),
                 ('lost',score_stats['lost_indices']),('gained',score_stats['gained_indices'])]}},
            'evidence_coverage':coverage,'semantic_mechanism_claimed':False}
        return self.compact_selection(call,hidden,selected,stats)


def create_executor(*,model,stage,runtime=None,execution_phase='generate',**kwargs):
    return QwenSelectorMechanismComparison(model,stage.config,runtime=runtime,phase=execution_phase)


def register_contracts(registry):
    from dataclasses import replace
    from mllm_comp.compressors.contracts import AlgorithmDescriptor,ModalityVariantDescriptor
    from mllm_comp.core.vocabulary import CompressionMethod,Placement,TokenOperation,Modality,ExecutionPhase
    from mllm_comp.core.support import SupportUnit,SupportUnitId
    name='selector_mechanism_audit'
    registry.register_algorithm(AlgorithmDescriptor(name=name,version='v0.1.0',method=CompressionMethod.TOKEN_COMPRESSION,
        kernel=QwenSelectorMechanismComparison,supported_placements=frozenset({Placement.PROJECTOR_LEVEL}),
        operations=frozenset({TokenOperation.DROP}),metadata={'claim':'Fixed-checkpoint causal candidate-token swaps; not semantic evidence ground truth'}))
    registry.register_variant(ModalityVariantDescriptor(name=f'{name}.image',algorithm=name,
        target_modalities=frozenset({Modality.IMAGE}),conditioning_modalities=frozenset({Modality.TEXT})))
    source=next(b for b in registry.bindings() if b.name=='selector_condition_audit.image.qwen25_vl')
    binding=f'{name}.image.qwen25_vl'
    registry.register_binding(replace(source,name=binding,variant=f'{name}.image',executor_factory=create_executor,
        conflicts=frozenset(b.name for b in registry.bindings() if b.backbone=='qwen25_vl')))
    registry.extend_support(SupportUnit(identity=SupportUnitId(algorithm=name,variant=f'{name}.image',
        backbone='qwen25_vl',target_modality=Modality.IMAGE,phase=phase),binding=binding)
        for phase in (ExecutionPhase.FORWARD,ExecutionPhase.GENERATE))
