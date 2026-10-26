"""OT ranking ablation on the unchanged context-selector checkpoint format."""
from .context_topk import QwenContextTopKComparison
from mllm_comp.compressors.image.query_transport import QueryTransportSpec, transport_topk


class QwenContextTopKOTComparison(QwenContextTopKComparison):
    def __init__(self, model, config, **kwargs):
        config = dict(config)
        ot = {key: config.pop('ot_' + key) for key in QueryTransportSpec.__dataclass_fields__ if 'ot_' + key in config}
        self.ot_spec = QueryTransportSpec(**ot)
        if config.get('training_mode', 'hard') != 'hard':
            raise ValueError('OT ablation is hard inference only; train noise gate separately')
        super().__init__(model, config, **kwargs)
        self.method = 'context_visual_topk_ot'

    def prepare_source_inputs(self, call):
        from mllm_comp.compressors.image.context_visual_topk import prompt_question
        super().prepare_source_inputs(call)
        if self.selector.spec.context_mode == 'image':
            self.source.get()['question'], self.source.get()['question_audit'] = prompt_question(self.model, call)

    def prepare_qwen_embeddings(self, call, hidden):
        import torch
        visual_ids = self.incoming.get().vision[0].nonzero(as_tuple=True)[0]
        features = hidden[0].index_select(0, visual_ids)
        source = self.source.get()
        with torch.no_grad():
            output = self.selector(features, question=source.get('question'))
            selected, stats = transport_topk(features, source['question'], output['scores'],
                self.token_budget(len(features)), self.ot_spec)
        return self.compact_selection(call, hidden, selected, {**stats,
            **source['question_audit'], 'trained': bool(self.selector_config.get('checkpoint_path')),
            'checkpoint_path': self.selector_config.get('checkpoint_path'), 'reads_question': True,
            'context_mode': self.selector.spec.context_mode, 'training_mode': 'hard_compact',
            'ot_ablation': True})

    def runtime_metadata(self):
        from dataclasses import asdict
        return {**super().runtime_metadata(), 'ot_config': asdict(self.ot_spec)}


def create_executor(*, model, stage, runtime=None, execution_phase='generate', **kwargs):
    return QwenContextTopKOTComparison(model, stage.config, runtime=runtime, phase=execution_phase)


def register_contracts(registry):
    from dataclasses import replace
    from mllm_comp.compressors.contracts import AlgorithmDescriptor, ModalityVariantDescriptor
    from mllm_comp.core.vocabulary import CompressionMethod, Placement, TokenOperation, Modality, ExecutionPhase
    from mllm_comp.core.support import SupportUnit, SupportUnitId
    name = 'context_visual_topk_ot'
    registry.register_algorithm(AlgorithmDescriptor(name=name, version='v0.1.0',
        method=CompressionMethod.TOKEN_COMPRESSION, kernel=transport_topk,
        supported_placements=frozenset({Placement.PROJECTOR_LEVEL}), operations=frozenset({TokenOperation.DROP}),
        metadata={'claim': 'Semi-unbalanced transport ranking experiment, not an OTFS reproduction',
            'trained': True, 'fixed_count': True}))
    registry.register_variant(ModalityVariantDescriptor(name=f'{name}.image', algorithm=name,
        target_modalities=frozenset({Modality.IMAGE}), conditioning_modalities=frozenset({Modality.TEXT})))
    source = next(b for b in registry.bindings() if b.name == 'context_visual_topk.image.qwen25_vl')
    binding = f'{name}.image.qwen25_vl'
    registry.register_binding(replace(source, name=binding, variant=f'{name}.image',
        executor_factory=create_executor,
        conflicts=frozenset(b.name for b in registry.bindings() if b.backbone == 'qwen25_vl')))
    registry.extend_support(SupportUnit(identity=SupportUnitId(algorithm=name, variant=f'{name}.image',
        backbone='qwen25_vl', target_modality=Modality.IMAGE, phase=phase), binding=binding)
        for phase in (ExecutionPhase.FORWARD, ExecutionPhase.GENERATE))
