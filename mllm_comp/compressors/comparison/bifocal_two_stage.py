"""Instance-local Bifocal Stage 1+2 for pinned Transformers 4.57.3 Qwen2.5.

Stage 1 reuses the platform's mixed-granularity kernel. Stage 2 follows the
Qwen reference's last-text-query attention times 3x3 occupancy density. Early
layers intentionally retain their longer KV caches; later layers have shorter
caches. This is a controlled adaptation, not paper-number reproduction.
"""
from mllm_comp.backbones._provider_patching import MethodPatch
from mllm_comp.fixed_mask.runtime import _call


def density_scores(scores, representatives, height, width, weight=1.):
    import torch
    import torch.nn.functional as F
    occupied = torch.zeros(height * width, device=scores.device, dtype=torch.float32)
    occupied[representatives] = 1.
    counts = F.avg_pool2d(occupied.view(1, 1, height, width), 3, stride=1,
        padding=1, count_include_pad=True).flatten().mul(9).round()[representatives]
    return scores.float() * ((1. - weight) + weight * counts / 9.)


def stage1_fraction(layer_count, boundary, average_fraction=.25, final_fraction=32/576):
    if not 0 < boundary < layer_count or not 0 < final_fraction <= average_fraction < 1:
        raise ValueError('Invalid Bifocal two-stage layer/fraction schedule')
    fraction = (average_fraction * layer_count - (layer_count - boundary) * final_fraction) / boundary
    if not final_fraction <= fraction < 1:
        raise ValueError('Requested Bifocal schedule has no valid Stage-1 budget')
    return fraction


class BifocalTwoStage:
    def __init__(self, model, runtime, config, facade=None):
        from mllm_comp.compressors.multimodal_registry import builtin_multimodal_compression_contracts
        from mllm_comp.pipeline.stage import CompressionStageConfig
        from mllm_comp.backbones.qwen25_vl.image_compression import create_qwen25_vl_image_compression_executor
        self.model, self.runtime = model, runtime
        self.decoder = model.model.language_model
        self.layers = self.decoder.layers
        cfg = dict(config)
        self.boundary = cfg.pop('stage2_layer', round(len(self.layers) * 15/28))
        self.final_fraction = float(cfg.pop('stage2_fraction', 32/576))
        self.minimum = cfg.pop('stage2_min_keep', 8)
        self.density_weight = float(cfg.pop('density_weight', 1.))
        average = float(cfg.pop('average_fraction', .25))
        explicit_stage1 = cfg.pop('stage1_config', {})
        if cfg or not isinstance(self.boundary, int) or isinstance(self.boundary, bool):
            raise ValueError(f'Unknown/invalid Bifocal two-stage config: {cfg}')
        if not 0 <= self.density_weight <= 1 or not isinstance(self.minimum, int) or self.minimum < 1:
            raise ValueError('Invalid Bifocal density weight/minimum budget')
        fraction = stage1_fraction(len(self.layers), self.boundary, average, self.final_fraction)
        if {'target_count', 's1_frac'} & set(explicit_stage1):
            raise ValueError('Stage-1 budget is derived from the declared two-stage schedule')
        inner_config = {'s1_frac': fraction, **explicit_stage1}
        stage = CompressionStageConfig.from_dict({'name': 'bifocal_stage1_component',
            'algorithm': 'bifocal', 'plugin': 'bifocal', 'variant': 'bifocal.image',
            'binding': 'bifocal.image.qwen25_vl', 'placement': 'projector_level',
            'method': 'token_compression', 'object_family': 'tensor_slot',
            'execution_phase': 'generate', 'modalities': ['image'], 'target_modalities': ['image'],
            'conditioning_modalities': ['text'], 'preserved_modalities': ['text', 'video', 'audio'],
            'coupling_policy': 'independent', 'media_alignment': 'spatial',
            'config': inner_config})
        binding = builtin_multimodal_compression_contracts().binding('bifocal.image.qwen25_vl')
        self.inner = create_qwen25_vl_image_compression_executor(model=model, runtime=runtime,
            facade=facade, stage=stage, binding=binding, binding_config=None, execution_phase='generate')
        self.config = {'average_fraction': average, 'stage1_fraction': fraction,
            'stage2_fraction': self.final_fraction, 'stage2_layer': self.boundary,
            'stage2_min_keep': self.minimum, 'density_weight': self.density_weight,
            'stage1_config': inner_config}
        self.patch = None
        self.state = None

    def install(self):
        if self.patch is not None:
            return self
        if self.decoder.has_sliding_layers or self.model.config._attn_implementation not in {'eager', 'sdpa'}:
            raise ValueError('Bifocal two-stage requires full-attention eager/sdpa Qwen2.5')
        original = self.decoder.forward
        def forward(_decoder, *args, **kwargs):
            return self.forward(_call(original, args, kwargs))
        # The Stage-1 provider must capture this wrapper as its language forward.
        self.patch = MethodPatch.apply(self.decoder, 'forward', forward, original, label='bifocal_two_stage')
        try:
            self.inner.install()
        except BaseException:
            self.patch.restore()
            self.patch = None
            raise
        return self

    def uninstall(self):
        if self.patch is None:
            return
        self.inner.uninstall()
        self.patch.restore()
        self.patch = None
        self.state = None

    def clear_diagnostic_records(self):
        self.inner.clear_diagnostic_records()
        self.state = None

    def _initialize(self, hidden, positions):
        import torch
        record = self.inner.runtime_metadata()['latest_layout_delta']
        if not record or record.get('bifocal_passthrough'):
            raise ValueError('Two-stage Bifocal requires an applicable single-image Stage-1 result')
        bridge = self.inner._bridge
        dense_mask = bridge._prompt_image_mask.get()
        removal = bridge._prompt_removal_mask.get()
        visual = dense_mask[:, ~removal[0]] if removal is not None else dense_mask
        if visual.shape != hidden.shape[:2] or int(visual.sum()) != record['result_count']:
            raise ValueError('Bifocal Stage-1 visual layout did not reach the decoder')
        grid = record.get('grid') or record.get('native_grid')
        # Kernel grid is also recorded in the LayoutDelta source layout.
        if grid is None:
            grid_thw = bridge._comparison_grid.get()
            if grid_thw is None or len(grid_thw) != 1:
                raise ValueError('Missing native single-image grid')
            grid = (grid_thw[0, 1:] // self.model.model.visual.spatial_merge_size).tolist()
        height, width = map(int, grid)
        representatives = torch.tensor(record['bifocal_representatives'], device=hidden.device)
        if height * width != record['source_count'] or len(representatives) != int(visual.sum()):
            raise ValueError('Bifocal Stage-1 representatives/grid mismatch')
        final = min(len(representatives), max(self.minimum, round(record['source_count'] * self.final_fraction)))
        self.state = {'visual': visual, 'representatives': representatives, 'grid': [height, width],
            'source_count': record['source_count'], 'stage1_count': len(representatives), 'stage2_count': final,
            'prefill_stage1_length': hidden.shape[1], 'decode_steps': 0,
            'expected_first_decode_position': int(positions.max()) + 1,
            'decode_position_verified': None, 'observed_visual_tokens_per_decoder_layer': [],
            'stage2_selected_indices': None, 'prefill_cache_lengths': None}

    def _select(self, hidden, positions):
        import torch
        from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import apply_multimodal_rotary_pos_emb, repeat_kv
        state = self.state
        layer = self.layers[self.boundary]
        attention = layer.self_attn
        normalized = layer.input_layernorm(hidden)
        b, n, _ = normalized.shape
        q = attention.q_proj(normalized).view(b, n, -1, attention.head_dim).transpose(1, 2)
        k = attention.k_proj(normalized).view(b, n, -1, attention.head_dim).transpose(1, 2)
        cos, sin = self.decoder.rotary_emb(hidden, positions)
        q, k = apply_multimodal_rotary_pos_emb(q, k, cos, sin, attention.rope_scaling['mrope_section'])
        k = repeat_kv(k, attention.num_key_value_groups)
        visual = state['visual'][0].nonzero(as_tuple=True)[0]
        nonvisual = (~state['visual'][0]).nonzero(as_tuple=True)[0]
        if not len(nonvisual):
            raise ValueError('Bifocal Stage 2 has no prompt text query')
        logits = (q[0, :, int(nonvisual[-1]):int(nonvisual[-1])+1].float()
            @ k[0, :, visual].float().transpose(-1, -2)) * attention.scaling
        raw = logits.softmax(-1).mean(0).squeeze(0)
        scores = density_scores(raw, state['representatives'], *state['grid'], self.density_weight)
        selected = scores.topk(state['stage2_count']).indices
        keep = torch.cat((nonvisual, visual[selected])).sort().values
        if len(keep.unique()) != len(keep) or not torch.isfinite(scores).all():
            raise ValueError('Invalid Bifocal Stage-2 scores or membership')
        state['stage2_selected_indices'] = state['representatives'][selected].sort().values.tolist()
        state['prefill_stage2_length'] = len(keep)
        return hidden[:, keep], positions[:, :, keep]

    def forward(self, call):
        import torch
        from transformers.cache_utils import DynamicCache
        from transformers.modeling_outputs import BaseModelOutputWithPast
        if self.model.training or call.get('output_attentions') or call.get('output_hidden_states'):
            raise ValueError('Bifocal two-stage is an eval-only generation binding without full-state capture')
        hidden = call.get('inputs_embeds')
        if hidden is None:
            hidden = self.decoder.embed_tokens(call['input_ids'])
        if hidden.shape[0] != 1:
            raise ValueError('Bifocal two-stage supports batch size one')
        mask = call.get('attention_mask')
        if mask is not None and (not torch.is_tensor(mask) or mask.ndim != 2 or not bool((mask == 1).all())):
            raise ValueError('Bifocal two-stage requires an unpadded 2D prompt mask')
        positions = call.get('position_ids')
        if positions is None or positions.ndim != 3 or positions.shape[0] not in (3, 4):
            raise ValueError('Bifocal two-stage needs native M-RoPE positions')
        positions = positions[-3:]
        past = call.get('past_key_values')
        use_cache = call.get('use_cache', True)
        if not use_cache:
            raise ValueError('Bifocal two-stage is generation-only with a KV cache')
        if past is None:
            past = DynamicCache(config=self.decoder.config)
        decode = past.get_seq_length(0) > 0
        if not decode:
            self._initialize(hidden, positions)
        elif self.state is None or hidden.shape[1] != 1:
            raise ValueError('Bifocal cache has no owning request or is not token-wise decode')
        state = self.state
        if decode:
            expected = state['expected_first_decode_position'] + state['decode_steps']
            valid = bool((positions == expected).all())
            if not valid:
                raise ValueError('Bifocal two-stage decode M-RoPE shifted after pruning')
            state['decode_position_verified'] = True
        layer_cache_lengths = []
        for index, layer in enumerate(self.layers):
            if not decode and index == self.boundary:
                hidden, positions = self._select(hidden, positions)
            count = state['stage1_count'] if index < self.boundary else state['stage2_count']
            if not decode:
                observed = hidden.shape[1] - (state['prefill_stage1_length'] - state['stage1_count'])
                if observed != count:
                    raise ValueError('Bifocal decoder visual/text token accounting changed')
                state['observed_visual_tokens_per_decoder_layer'].append(observed)
            previous = past.get_seq_length(index)
            length = hidden.shape[1]
            cache_position = torch.arange(previous, previous + length, device=hidden.device)
            keys = torch.arange(previous + length, device=hidden.device)
            visible = keys[None, :] <= cache_position[:, None]
            causal = torch.zeros((length, previous + length), dtype=hidden.dtype, device=hidden.device)
            causal.masked_fill_(~visible, torch.finfo(hidden.dtype).min)
            embeddings = self.decoder.rotary_emb(hidden, positions)
            result = layer(hidden, attention_mask=causal[None, None], position_ids=None,
                past_key_values=past, use_cache=True, output_attentions=False,
                cache_position=cache_position, position_embeddings=embeddings)
            hidden = result[0]
            actual = past.get_seq_length(index)
            expected = (state['prefill_stage1_length'] if index < self.boundary else state['prefill_stage2_length'])
            expected += state['decode_steps'] + int(decode)
            if actual != expected:
                raise ValueError(f'Bifocal layer {index} cache length {actual} != {expected}')
            layer_cache_lengths.append(actual)
        if decode:
            state['decode_steps'] += 1
        else:
            state['prefill_cache_lengths'] = layer_cache_lengths
        state['last_cache_lengths'] = layer_cache_lengths
        output = BaseModelOutputWithPast(last_hidden_state=self.decoder.norm(hidden), past_key_values=past)
        return output if call.get('return_dict', True) else (output.last_hidden_state, past)

    def runtime_metadata(self):
        state = self.state
        public = None
        if state is not None:
            public = {k: v for k, v in state.items() if k not in {'visual', 'representatives'}}
            counts = state['observed_visual_tokens_per_decoder_layer']
            public['layer_averaged_visual_tokens'] = sum(counts) / len(self.layers)
            public['layer_averaged_retention_ratio'] = sum(counts) / (len(self.layers) * state['source_count'])
        return {**self.inner.runtime_metadata(), 'algorithm': 'bifocal_two_stage', 'stage1_only': False,
            'config': self.config, 'stage2': public,
            'cache_policy': 'early_layers_keep_stage1_later_layers_keep_stage2',
            'classification': 'projector_merge_plus_decoder_pruning', 'paper_number_reproduction': False}


def create_executor(*, model, runtime, stage, facade=None, **kwargs):
    return BifocalTwoStage(model, runtime, stage.config, facade=facade)


def register_contracts(registry):
    from mllm_comp.compressors.contracts import AlgorithmDescriptor, ModalityVariantDescriptor, BackbonePlacementBinding, BindingTarget, TokenUnit
    from mllm_comp.core.vocabulary import CompressionMethod, Placement, TokenOperation, Modality, ExecutionPhase, ModalityCouplingPolicy, MediaAlignment
    from mllm_comp.core.support import SupportUnit, SupportUnitId
    name = 'bifocal_two_stage'
    image = frozenset({Modality.IMAGE})
    binding = f'{name}.image.qwen25_vl'
    registry.register_algorithm(AlgorithmDescriptor(name=name, version='qwen25-stage12-v1',
        method=CompressionMethod.TOKEN_COMPRESSION, kernel=density_scores,
        supported_placements=frozenset({Placement.LLM_LEVEL}),
        operations=frozenset({TokenOperation.MERGE, TokenOperation.DROP}),
        metadata={'source': 'Bifocal paper and read-only Qwen reference', 'projector_only': False}))
    registry.register_variant(ModalityVariantDescriptor(name=f'{name}.image', algorithm=name,
        target_modalities=image, conditioning_modalities=frozenset({Modality.TEXT}),
        preserved_modalities=frozenset({Modality.TEXT, Modality.VIDEO, Modality.AUDIO}),
        coupling_policy=ModalityCouplingPolicy.INDEPENDENT, media_alignment=MediaAlignment.SPATIAL,
        token_unit=TokenUnit.LLM_TOKEN))
    registry.register_binding(BackbonePlacementBinding(name=binding, backbone='qwen25_vl', variant=f'{name}.image',
        targets=(BindingTarget(name='bifocal_two_stage', placement=Placement.LLM_LEVEL,
            target_site='llm', target_modalities=image, required_handles=frozenset({'llm', 'vision_projector'})),),
        supported_phases=frozenset({ExecutionPhase.GENERATE}), executor_factory=create_executor,
        legalizer='qwen25_vl.original_mrope_layer_local_cache', exclusive_runtime_slot='image_comparison',
        conflicts=frozenset(b.name for b in registry.bindings() if b.backbone == 'qwen25_vl'),
        metadata={'single_image_only': True, 'batch_size': 1, 'projector_only': False}))
    registry.extend_support([SupportUnit(identity=SupportUnitId(algorithm=name, variant=f'{name}.image',
        backbone='qwen25_vl', target_modality=Modality.IMAGE, phase=ExecutionPhase.GENERATE), binding=binding)])
