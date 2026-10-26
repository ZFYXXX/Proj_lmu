"""Change ONLY the trained selector's condition, never the LLM user question."""
import hashlib
import json
from pathlib import Path

from .context_topk import QwenContextTopKComparison
from mllm_comp.analysis.common.selector_conditioning import FORMAT, ids_sha, request_key


class QwenSelectorConditionComparison(QwenContextTopKComparison):
    def __init__(self, model, config, *, condition_data=None, **kwargs):
        config = dict(config)
        self.condition_mode = config.pop('condition_mode', 'true')
        filename = config.pop('condition_manifest', None)
        checksum = config.pop('condition_manifest_sha256', None)
        if self.condition_mode not in ('true', 'shuffled', 'generic'):
            raise ValueError('Unknown selector condition mode')
        if config.get('training_mode', 'hard') != 'hard' or config.get('context_mode', 'question') != 'question':
            raise ValueError('Condition audit requires hard inference with the question selector')
        if condition_data is None:
            if not filename or not checksum:
                raise ValueError('Condition audit requires a pinned manifest')
            raw = Path(filename).read_bytes()
            if hashlib.sha256(raw).hexdigest() != checksum:
                raise ValueError('Condition manifest checksum mismatch')
            condition_data = json.loads(raw)
        if condition_data.get('format') != FORMAT or not condition_data.get('rows'):
            raise ValueError('Invalid selector condition manifest')
        super().__init__(model, config, **kwargs)
        if config.get('checkpoint_path'):
            weights = Path(config['checkpoint_path']) / 'selector.safetensors'
            if hashlib.sha256(weights.read_bytes()).hexdigest() != condition_data.get('selector_weights_sha256'):
                raise ValueError('Condition manifest belongs to a different selector checkpoint')
            if condition_data.get('base_identity_sha256') != self.base_identity:
                raise ValueError('Condition manifest base identity mismatch')
        self.condition_data = condition_data
        self.condition_manifest_sha256 = checksum
        self.method = 'selector_condition_audit'

    def prepare_source_inputs(self, call):
        import torch
        from mllm_comp.integrations.lmms_eval.request_context import CURRENT_SAMPLE_ID
        super().prepare_source_inputs(call)
        sample_id = CURRENT_SAMPLE_ID.get()
        key = None if sample_id is None else request_key(sample_id)
        if key is None or key not in self.condition_data['rows']:
            raise ValueError('Missing or unknown native evaluation request identity')
        row = self.condition_data['rows'][key]
        if set(row) & {'answer', 'target', 'prediction', 'score', 'label', 'labels'}:
            raise ValueError('Answers/scores must not enter the selector condition manifest')
        source = self.source.get()
        if ids_sha(row['true_ids']) != source['question_audit']['question_ids_sha256']:
            raise ValueError('Actual LLM prompt differs from the frozen true-question IDs')
        key = {'true': 'true_ids', 'shuffled': 'donor_ids', 'generic': 'generic_ids'}[self.condition_mode]
        ids = row[key]
        if not ids or any(isinstance(i, bool) or not isinstance(i, int) or not 0 <= i < self.model.config.vocab_size or
                i in {151643, 151644, 151645, self.model.config.image_token_id,
                    self.model.config.vision_start_token_id, self.model.config.vision_end_token_id} for i in ids):
            raise ValueError('Invalid selector-only condition token IDs')
        source['true_question'] = source['question']
        with torch.no_grad():
            source['question'] = self.model.get_input_embeddings()(torch.tensor(ids,
                device=source['question'].device, dtype=torch.long)).detach()
        source['condition_audit'] = {'condition_mode': self.condition_mode,
            'condition_manifest_sha256': self.condition_manifest_sha256,
            'condition_sample_id': sample_id, 'condition_ids_sha256': ids_sha(ids),
            'condition_tokens': len(ids), 'llm_prompt_ids_sha256': ids_sha(call['input_ids'][0].tolist()),
            'donor_doc_id': row['donor_doc_id'] if self.condition_mode == 'shuffled' else None,
            'donor_image_fingerprint': row['donor_image_fingerprint'] if self.condition_mode == 'shuffled' else None,
            'shuffle_informative': row['informative'], 'shuffle_reason': row['reason'],
            'stratum': row['stratum'], 'llm_question_replaced': False}

    def prepare_qwen_embeddings(self, call, hidden):
        import torch
        visual_ids = self.incoming.get().vision[0].nonzero(as_tuple=True)[0]
        features = hidden[0].index_select(0, visual_ids)
        source = self.source.get()
        with torch.no_grad():
            actual = self.selector(features, question=source['question'])
            true = actual if self.condition_mode == 'true' else self.selector(features, question=source['true_question'])
        selected, original = actual['indices'], true['indices']
        intersection = int(torch.isin(selected, original).sum())
        score_gap = float((actual['scores'] - true['scores']).abs().mean())
        stats = {**source['question_audit'], **source['condition_audit'],
            'reads_question': True, 'context_mode': 'question', 'training_mode': 'hard_compact',
            'trained': bool(self.selector_config.get('checkpoint_path')),
            'checkpoint_path': self.selector_config.get('checkpoint_path'),
            'true_selected_indices': original.tolist(),
            'selection_overlap_fraction': intersection / len(selected),
            'selection_jaccard': intersection / (len(selected) + len(original) - intersection),
            'selection_identical': bool(torch.equal(selected, original)), 'score_mean_abs_change': score_gap}
        return self.compact_selection(call, hidden, selected, stats)


def create_executor(*, model, stage, runtime=None, execution_phase='generate', **kwargs):
    return QwenSelectorConditionComparison(model, stage.config, runtime=runtime, phase=execution_phase)


def register_contracts(registry):
    from dataclasses import replace
    from mllm_comp.compressors.contracts import AlgorithmDescriptor, ModalityVariantDescriptor
    from mllm_comp.core.vocabulary import CompressionMethod, Placement, TokenOperation, Modality, ExecutionPhase
    from mllm_comp.core.support import SupportUnit, SupportUnitId
    name = 'selector_condition_audit'
    registry.register_algorithm(AlgorithmDescriptor(name=name, version='v0.1.0',
        method=CompressionMethod.TOKEN_COMPRESSION, kernel=QwenSelectorConditionComparison,
        supported_placements=frozenset({Placement.PROJECTOR_LEVEL}), operations=frozenset({TokenOperation.DROP}),
        metadata={'claim': 'Inference-only condition intervention on a fixed selector, not a new trained method'}))
    registry.register_variant(ModalityVariantDescriptor(name=f'{name}.image', algorithm=name,
        target_modalities=frozenset({Modality.IMAGE}), conditioning_modalities=frozenset({Modality.TEXT})))
    source = next(b for b in registry.bindings() if b.name == 'context_visual_topk.image.qwen25_vl')
    binding = f'{name}.image.qwen25_vl'
    registry.register_binding(replace(source, name=binding, variant=f'{name}.image', executor_factory=create_executor,
        conflicts=frozenset(b.name for b in registry.bindings() if b.backbone == 'qwen25_vl')))
    registry.extend_support(SupportUnit(identity=SupportUnitId(algorithm=name, variant=f'{name}.image',
        backbone='qwen25_vl', target_modality=Modality.IMAGE, phase=phase), binding=binding)
        for phase in (ExecutionPhase.FORWARD, ExecutionPhase.GENERATE))
