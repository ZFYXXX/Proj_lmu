"""Round-two frozen interventions: importance-guided FPS and singleton bypass."""
from dataclasses import replace
import json
import math
from pathlib import Path

import torch

from .anchor_adapter import QwenAnchorAdapterComparison
from mllm_comp.compressors.image.spatial_anchor_ablation import spatial_anchor_merge, merge_positions


def adapt_clusters(adapter, base, singleton_bypass):
    """Only evaluate the MLP on merged clusters when singleton bypass is enabled."""
    if not singleton_bypass:
        return adapter(base)['tokens']
    tokens = base['tokens'].clone()
    merged = torch.tensor([i for i, g in enumerate(base['members']) if len(g) > 1],
                          device=tokens.device, dtype=torch.long)
    if len(merged):
        corrected = adapter({'tokens': tokens.index_select(0, merged)})['tokens']
        tokens.index_copy_(0, merged, corrected)
    return tokens


def importance_pool(features, members, probabilities):
    """Differentiable FP32 within-cluster pooling; membership stays discrete."""
    if probabilities.shape != (len(features),) or not torch.isfinite(probabilities).all() or (probabilities < 0).any():
        raise ValueError('Invalid pooling probabilities')
    tokens, statistics = [], []
    for group in members:
        ids = torch.as_tensor(group, device=features.device, dtype=torch.long)
        raw = features.index_select(0, ids).float()
        weights = probabilities.index_select(0, ids).float()
        mass = weights.sum()
        fallback = bool(mass == 0)
        alpha = torch.ones_like(weights) / len(group) if fallback else weights / mass
        value = (alpha[:, None] * raw).sum(0)
        if not torch.isfinite(value).all() or not torch.isclose(alpha.sum(), alpha.new_tensor(1.), atol=1e-5):
            raise ValueError('Invalid weighted cluster mean')
        tokens.append(value.to(features.dtype))
        statistics.append({'effective_members': float((1 / alpha.square().sum()).detach()),
            'max_weight': float(alpha.max().detach()), 'probability_mass': float(mass.detach()),
            'uniform_fallback': fallback, 'weights_sum': float(alpha.sum().detach()),
            'mean_shift_norm': float((value - raw.mean(0)).norm().detach())})
    return torch.stack(tokens), statistics


class QwenImportanceAnchorAdapter(QwenAnchorAdapterComparison):
    def __init__(self, model, config, *, runtime=None, adapter_module=None, selector_module=None):
        self.intervention_config = dict(config)
        config = dict(config)
        self.gamma = float(config.pop('importance_gamma', 0.))
        self.singleton_bypass = config.pop('singleton_bypass', False)
        self.scorer_path = config.pop('scorer_path', None)
        self.pool_scorer_path = config.pop('pool_scorer_path', None)
        self.spatial_beta = float(config.pop('spatial_beta', 0.))
        self.pooling = config.pop('pooling', 'mean')
        self.assignment_neighbors = config.pop('assignment_neighbors', 0)
        self.assignment_max_regret = config.pop('assignment_max_regret', None)
        self.grid_anchor_floor = config.pop('grid_anchor_floor', False)
        if not isinstance(self.grid_anchor_floor, bool):
            raise ValueError('grid_anchor_floor must be a boolean')
        if isinstance(self.assignment_neighbors, bool) or not isinstance(self.assignment_neighbors, int) or self.assignment_neighbors < 0:
            raise ValueError('assignment_neighbors must be a nonnegative integer')
        if self.assignment_max_regret is not None:
            if (isinstance(self.assignment_max_regret, bool) or not isinstance(self.assignment_max_regret, (int, float))
                    or not math.isfinite(self.assignment_max_regret) or not 0 <= self.assignment_max_regret <= 2
                    or not self.assignment_neighbors or self.spatial_beta != 0):
                raise ValueError('assignment_max_regret requires a finite gap in [0, 2], local candidates and beta=0')
        if self.gamma not in {0., 1.} or not isinstance(self.singleton_bypass, bool):
            raise ValueError('Round two fixes importance gamma to 0/1 and bypass to a boolean')
        if self.pooling not in {'mean', 'importance'} or not math.isfinite(self.spatial_beta) or self.spatial_beta < 0:
            raise ValueError('Invalid pooling or spatial configuration')
        identity_path = config.get('identity_path')
        super().__init__(model, config, runtime=runtime, adapter_module=adapter_module)
        self.method = 'importance_anchor_adapter'
        self.selector = None
        if self.gamma or self.pooling == 'importance':
            if selector_module is None:
                from mllm_comp.compressors.image.context_visual_topk import load_selector
                identity = json.loads(Path(identity_path).read_text())
                selector_module = load_selector(self.scorer_path, base_identity=identity['identity_sha256'],
                    device=self.decoder.embed_tokens.weight.device)
            spec = selector_module.spec
            if spec.context_mode != 'image' or spec.hidden_size != self.decoder.config.hidden_size or spec.target_count != 128:
                raise ValueError('Expected the existing image-context K128 selector for this backbone')
            self.selector = selector_module.eval().requires_grad_(False)
        self.pool_selector = None
        if self.pool_scorer_path:
            if self.pooling != 'importance':
                raise ValueError('A separate pool scorer requires importance pooling')
            from mllm_comp.compressors.image.context_visual_topk import load_selector
            identity = json.loads(Path(identity_path).read_text())
            self.pool_selector = load_selector(self.pool_scorer_path, base_identity=identity['identity_sha256'],
                device=self.decoder.embed_tokens.weight.device).eval().requires_grad_(False)
            if self.pool_selector.spec != self.selector.spec:
                raise ValueError('Anchor and pool scorer specifications must match')

    def prepare_qwen_embeddings(self, call, hidden):
        visual = self.incoming.get().vision[0].nonzero(as_tuple=True)[0]
        features = hidden[0, visual]
        merge = self.model.model.visual.spatial_merge_size
        height, width = (int(v) // merge for v in self.source.get()['grid'][0, 1:])
        importance, probabilities, score_trace = None, None, {}
        if self.selector is not None:
            with torch.no_grad():
                scored = self.selector(features, soft=True)
            probabilities = scored['soft_weights'].detach().float()
            importance = probabilities.pow(self.gamma) if self.gamma else None
            score_trace = {'selector_scores': scored['scores'].tolist(),
                           'selector_probabilities': scored['soft_weights'].tolist(),
                           'selector_soft_mass': float(scored['soft_weights'].sum())}
        base = spatial_anchor_merge(features, height, width, self.token_budget(len(features)),
                                    beta=self.spatial_beta, importance=importance,
                                    assignment_neighbors=self.assignment_neighbors,
                                    assignment_max_regret=self.assignment_max_regret,
                                    grid_anchor_floor=self.grid_anchor_floor)
        pool_stats = []
        if self.pooling == 'importance':
            if self.pool_selector is not None:
                with torch.no_grad():
                    probabilities = self.pool_selector(features, soft=True)['soft_weights'].detach().float()
                score_trace['pool_selector_probabilities'] = probabilities.tolist()
            base['tokens'], pool_stats = importance_pool(features, base['members'], probabilities)
        with torch.autocast(device_type=hidden.device.type, dtype=torch.bfloat16, enabled=hidden.is_cuda):
            values = adapt_clusters(self.adapter, base, self.singleton_bypass)
        singletons = [i for i, g in enumerate(base['members']) if len(g) == 1]
        if self.singleton_bypass and singletons and not torch.equal(values[singletons], base['tokens'][singletons]):
            raise ValueError('Singleton bypass changed original token contents')
        original = call['position_ids'].clone()
        compact = self.compact_selection(call, hidden, torch.tensor(base['indices'], device=hidden.device),
            {'trained': True, 'new_training': False, 'reads_question': False,
             'adapter_path': self.adapter_path, 'scorer_path': self.scorer_path if self.selector is not None else None,
             'pool_scorer_path': self.pool_scorer_path,
             'importance_gamma': self.gamma, 'singleton_bypass': self.singleton_bypass,
             'singleton_count': len(singletons), 'bypassed_cluster_indices': singletons if self.singleton_bypass else [],
             'cluster_members': base['members'], 'fps_anchor_indices': base['anchors'],
             'cluster_statistics': base['cluster_statistics'], 'partition_hash': base['partition_hash'],
             **{key: base[key] for key in ('assignment_neighbors', 'assignment_effective_neighbors',
                'assignment_changed_count', 'assigned_anchor_grid_distance_mean',
                'unrestricted_anchor_grid_distance_mean', 'fps_selection_order',
                'assignment_max_regret', 'global_fallback_count', 'global_fallback_fraction',
                'global_fallback_indices', 'assignment_cosine_regret_mean',
                'assignment_cosine_regret_max', 'grid_anchor_floor', 'grid_seed_indices',
                'grid_seed_region_ids', 'grid_nonempty_regions', 'grid_coverage_feasible',
                'grid_covered_regions') if key in base},
             'position_policy': 'centroid', 'spatial_beta': self.spatial_beta,
             'pooling': self.pooling, 'pooling_statistics': pool_stats, **score_trace})
        destinations = self.incoming.get().vision[0].nonzero(as_tuple=True)[0]
        compact = compact.clone()
        compact[0, destinations] = values.to(compact.dtype)
        positions = call['position_ids'].clone()
        positions[-3:, 0, destinations] = merge_positions(original[-3:, 0, visual],
                                                        base['members'], base['anchors'], 'centroid')
        if not torch.isfinite(compact).all():
            raise ValueError('Nonfinite importance-anchor output')
        self.last_selection.update(expected_first_decode_position=int(positions[-3:].max()) + 1,
                                   text_positions_preserved=True)
        call.update(inputs_embeds=compact, position_ids=positions)
        return compact

    def runtime_metadata(self):
        # The base borrows the DPPruner layout lifecycle; expose the actual method
        # settings instead of that lifecycle's unused default selector settings.
        return {**super().runtime_metadata(), 'config': self.intervention_config,
                'importance_gamma': self.gamma, 'singleton_bypass': self.singleton_bypass,
                'spatial_beta': self.spatial_beta, 'pooling': self.pooling,
                'assignment_neighbors': self.assignment_neighbors,
                'assignment_max_regret': self.assignment_max_regret,
                'grid_anchor_floor': self.grid_anchor_floor,
                'position_policy': 'centroid', 'new_training': False}


def create_executor(*, model, stage, runtime=None, **kwargs):
    return QwenImportanceAnchorAdapter(model, stage.config, runtime=runtime)


def register_contracts(registry):
    from mllm_comp.compressors.contracts import AlgorithmDescriptor, ModalityVariantDescriptor
    from mllm_comp.core.vocabulary import CompressionMethod, Placement, TokenOperation, Modality, ExecutionPhase
    from mllm_comp.core.support import SupportUnit, SupportUnitId
    name = 'importance_anchor_adapter'
    registry.register_algorithm(AlgorithmDescriptor(name=name, version='frozen-importance-singleton-v1',
        method=CompressionMethod.TOKEN_COMPRESSION, kernel=spatial_anchor_merge,
        supported_placements=frozenset({Placement.PROJECTOR_LEVEL}), operations=frozenset({TokenOperation.MERGE})))
    registry.register_variant(ModalityVariantDescriptor(name=name+'.image', algorithm=name,
        target_modalities=frozenset({Modality.IMAGE})))
    source = next(b for b in registry.bindings() if b.name == 'feature_anchor_adapter.image.qwen25_vl')
    binding = name+'.image.qwen25_vl'
    registry.register_binding(replace(source, name=binding, variant=name+'.image', executor_factory=create_executor,
        legalizer='qwen25_vl.importance_anchor_centroid_compact_cache'))
    registry.extend_support([SupportUnit(identity=SupportUnitId(algorithm=name, variant=name+'.image',
        backbone='qwen25_vl', target_modality=Modality.IMAGE, phase=ExecutionPhase.GENERATE), binding=binding)])
