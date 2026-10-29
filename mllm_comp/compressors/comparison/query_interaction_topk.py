"""Contextual selector on the native, tested word-selection lifecycle."""
import json
from pathlib import Path
from .word_topk import QwenWordTopKComparison
from mllm_comp.compressors.image.query_interaction_topk import QueryInteractionTopK, QueryInteractionSpec, load_selector


class QwenQueryInteractionTopK(QwenWordTopKComparison):
    def __init__(self, model, config, *, runtime=None, selector_module=None, phase='generate'):
        config = dict(config)
        interaction = config.pop('interaction', 'contextual')
        heads = config.pop('heads', 4)
        if selector_module is None:
            if config.get('checkpoint_path'):
                if runtime is None or not config.get('identity_path'):
                    raise ValueError('Checkpoint requires pinned backbone identity')
                identity = json.loads(Path(config['identity_path']).read_text())
                if Path(runtime.model_path_or_id).resolve() != Path(identity['checkpoint']).resolve():
                    raise ValueError('Wrong query-interaction backbone')
                selector_module = load_selector(config['checkpoint_path'], base_identity=identity['identity_sha256'],
                    device=model.get_input_embeddings().weight.device)
            else:
                selector_module = QueryInteractionTopK(QueryInteractionSpec(model.get_input_embeddings().embedding_dim,
                    width=config.get('width', 256), target_count=config.get('target_count', 128), seed=config.get('seed', 42),
                    match_width=config.get('match_width', 128), template_token_ids=config.get('template_token_ids', ()),
                    interaction=interaction, heads=heads))
        if not isinstance(selector_module, QueryInteractionTopK) or selector_module.spec.interaction != interaction or selector_module.spec.heads != heads:
            raise ValueError('Wrong query-interaction specification')
        super().__init__(model, config, runtime=runtime, selector_module=selector_module, phase=phase)
        self.method = 'query_interaction_topk'


def create_executor(*, model, stage, runtime=None, execution_phase='generate', **kwargs):
    return QwenQueryInteractionTopK(model, stage.config, runtime=runtime, phase=execution_phase)


def register_contracts(registry):
    from dataclasses import replace
    from mllm_comp.compressors.contracts import AlgorithmDescriptor, ModalityVariantDescriptor
    from mllm_comp.core.vocabulary import CompressionMethod, Placement, TokenOperation, Modality, ExecutionPhase
    from mllm_comp.core.support import SupportUnit, SupportUnitId
    name = 'query_interaction_topk'
    registry.register_algorithm(AlgorithmDescriptor(name=name, version='v1.0.0', method=CompressionMethod.TOKEN_COMPRESSION,
        kernel=QueryInteractionTopK, supported_placements=frozenset({Placement.PROJECTOR_LEVEL}),
        operations=frozenset({TokenOperation.DROP}), metadata={'claim': 'Per-token contextual question interaction, fixed K; exploratory'}))
    registry.register_variant(ModalityVariantDescriptor(name=f'{name}.image', algorithm=name,
        target_modalities=frozenset({Modality.IMAGE}), conditioning_modalities=frozenset({Modality.TEXT})))
    source = next(b for b in registry.bindings() if b.name == 'word_visual_topk.image.qwen25_vl')
    binding = f'{name}.image.qwen25_vl'
    registry.register_binding(replace(source, name=binding, variant=f'{name}.image', executor_factory=create_executor,
        conflicts=frozenset(b.name for b in registry.bindings() if b.backbone == 'qwen25_vl')))
    registry.extend_support(SupportUnit(identity=SupportUnitId(algorithm=name, variant=f'{name}.image',
        backbone='qwen25_vl', target_modality=Modality.IMAGE, phase=phase), binding=binding)
        for phase in (ExecutionPhase.FORWARD, ExecutionPhase.GENERATE))
