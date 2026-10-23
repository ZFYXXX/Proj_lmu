"""Method-owned image-token selection at existing LLM layer boundaries.

Decoder in the code names denotes the model stack, not incremental decode.
Selection runs during prefill or a direct forward (including training);
generation decode continues from the resulting layout and cache state.

Native providers supply Q/K, requested module observations and sequence repair. These objects
never own a model, cache, module patch, or checkpoint closure.
"""

from dataclasses import dataclass, fields
from typing import Any, Callable

from mllm_comp.core.errors import InvariantError
from mllm_comp.compressors.image.decoder_lifecycle import (
    CacheRepairPolicy,
    layout_merge_groups,
    layout_result_representatives,
)
from mllm_comp.compressors.image._rules import replay_rule_groups
from mllm_comp.compressors.image.pact import (
    PACTSpec, pact_attention_mask, pact_image_tokens, pact_token_scores,
)
from mllm_comp.compressors.image.sparsevlm import (
    SparseVLMSpec,
    remap_sparsevlm_attention_scores,
    select_sparsevlm_text_raters,
    sparsevlm_image_tokens,
)
from mllm_comp.compressors.image.sgl import sgl_image_tokens
from mllm_comp.compressors.image.nuwa import nuwa_stage2_image_tokens
from mllm_comp.compressors.image.fastv import (
    FastVSpec,
    fastv_image_tokens,
    fastv_scorer_query_indices,
)
from mllm_comp.compressors.image.btp import (
    BalancedTokenPruningSpec,
    balanced_token_pruning_image_tokens,
)
from mllm_comp.compressors.image.halfv import (
    HalfVSpec,
    halfv_image_tokens,
    halfv_last_text_attention_scores,
)
from mllm_comp.compressors.image.dart import (
    DARTSpec,
    dart_image_tokens,
    dart_post_image_text_indices,
)
from mllm_comp.compressors.image.d2pruner import (
    D2PrunerSpec,
    d2pruner_image_tokens,
    load_d2pruner_bias_prior,
)
from mllm_comp.compressors.image.feather import (
    FEATHERSpec,
    feather_image_tokens,
    feather_no_rope_scores,
)
from mllm_comp.compressors.image.finepruner import (
    FinePrunerSpec,
    finepruner_arp_image_tokens,
    finepruner_attention_head_slice,
    finepruner_iac_image_tokens,
)
from mllm_comp.compressors.image.ivc_prune import (
    IVCPruneSpec,
    ivc_coordinate_scores,
    ivc_post_image_text_indices,
    ivc_prune_image_tokens,
    project_ivc_value_states,
)
from mllm_comp.compressors.image.transprune import (
    TransPruneSpec,
    select_transprune_ttv_history,
    transprune_image_tokens,
    transprune_instruction_guided_attention,
    transprune_layer_transition_variation,
)
from mllm_comp.compressors.image.pyramid_drop import (
    PyramidDropSchedule,
    PyramidDropSpec,
    pyramid_attention_scores,
    pyramid_drop,
    pyramid_instruction_indices,
)
from mllm_comp.compressors.image.positions import (
    btp_decoder_positions,
    halfv_decoder_positions,
    decoder_token_operation,
    decoder_token_position,
    feather_decoder_positions,
    finepruner_decoder_positions,
    sparsevlm_decoder_positions,
    pact_decoder_position,
    nuwa_decoder_position,
    pyramid_decoder_positions,
    transprune_decoder_positions,
)
from mllm_comp.compressors.layer_positions import LayerPosition
from mllm_comp.compressors.image.token_selection import (
    ImageTokenSelection,
    ImageTokenSelectionResult,
)


@dataclass(frozen=True)
class AttentionCapture:
    """One fixed source, projected with a native decoder layer's Q/K."""

    source_layer: int
    projection_layer: int
    query_indices: Any
    use_rope: bool = True


@dataclass(frozen=True)
class DecoderSelectionRegistration:
    """Explicit native registration; ``apply`` owns only backbone layout repair."""

    method: Any
    apply: Callable[..., Any]
    execution_phase: Any
    activation_id: int


def capture_decoder_attention(call, capture, image_indices, read_attention):
    """Acquire one requested native signal and hand ownership to the method."""
    if capture is None:
        return
    query, image_keys, attention_keys, visibility = read_attention(capture, image_indices)
    call.record_attention(capture, query=query, image_keys=image_keys,
        attention_keys=attention_keys, visibility=visibility, image_indices=image_indices)


def execute_decoder_selection(call, position, hidden_states, *, image_indices,
        protected_indices=(), media_unit="llm_token", source_layout=None, result_layout=None):
    """Run one method and materialize a pure selection at the existing boundary.

    Methods with aggregation or history-specific results retain those results;
    their replay and native repair continue through the same execution path.
    """
    context = dict(image_indices=image_indices, protected_indices=protected_indices,
        media_unit=media_unit, source_layout=source_layout, result_layout=result_layout)
    result = call.select(position, hidden_states, **context)
    if isinstance(result, ImageTokenSelection):
        return ImageTokenSelectionResult.from_selection(result, hidden_states, **context)
    return result


def decoder_layout_replay(delta, *, rules):
    """Capture a selection or mean-merge decision for native checkpoint replay."""
    weighted_rows = delta.result_layout.get("weighted_replay_rows")
    if weighted_rows is not None:
        def replay_weighted(values):
            import torch
            rows = []
            for indices, coefficients in weighted_rows:
                ids = torch.tensor(indices, device=values.device, dtype=torch.long)
                weights = values.new_tensor(coefficients)
                rows.append((values.index_select(1, ids) * weights[None, :, None]).sum(1))
            return torch.stack(rows, dim=1)
        return replay_weighted
    if delta.is_identity:
        return lambda values: values
    if delta.merge_groups:
        groups = layout_merge_groups(delta)
        return lambda values: replay_rule_groups(
            values[0], groups, rules=rules
        ).unsqueeze(0)
    representatives = layout_result_representatives(delta)

    def gather(values):
        import torch

        indices = torch.tensor(representatives, dtype=torch.long, device=values.device)
        return values.index_select(1, indices)

    return gather


class DecoderSelectionCall:
    """Optional native observations for one invocation, without model ownership."""

    output_hidden_layer = None
    # Request record_output_features after the native layer (including DeepStack).
    output_feature_layers = ()
    # Map completed source layers to native Q/K projection layers. Unlike
    # AttentionCapture, this observation provides both pre- and post-RoPE keys.
    output_projection_layers = {}
    transition_layers = ()

    def initialize(self, input_embeddings, image_mask, *, padding_mask, labels):
        """Observe prompt embeddings once, before any application or layer scoring."""
        return None

    def select_before_input(self, layer_index):
        """Whether an inter-layer selection precedes this layer's input capture.

        This does not move post-attention or completed-layer observations. A
        delayed earlier stage must remap pending scores before a later stage
        samples the same entry; a stage sampling its own input captures first.
        """
        return False

    def capture_input(self, layer_index, image_mask, *, padding_mask, labels):
        return None

    def capture_output(self, layer_index, image_mask, *, padding_mask, labels):
        return None

    def prepare_attention_mask(self, base_mask, *, query_length, reference):
        """Apply method-owned attention state before a native layer executes."""
        return base_mask

    def begin_transition(self, layer_index, image_indices):
        """Request normalized inputs and attention/FFN updates for this layer.

        Native providers own the temporary hooks and call record_transition
        after the layer. Selection-only methods need no transition state.
        """
        return False


class NuwaDecoderMethod:
    """Nüwa's second stage; visual aggregation remains in its composite activation."""

    algorithm = "nuwa"
    cache_policy = CacheRepairPolicy.COMPACT_PREFIX
    position_policy = "sparse_source"
    single_image_only = True

    def __init__(self, *, stage, spec, rules, backbone, num_layers=None):
        self.stage, self.spec, self.rules, self.backbone = stage, spec, rules, backbone
        self.score_layers = (
            () if spec.stage2_score_layer is None else (spec.stage2_score_layer,)
        )
        if "stage2_source_scorer" in rules and not self.score_layers:
            raise ValueError(
                "Nüwa stage2_source_scorer requires the attention mode with a fixed QK source."
            )
        self.target_site = decoder_token_operation(stage, "nuwa", backbone=backbone)[
            "target_site"
        ]
        self.resolve_depth(num_layers)

    def resolve_depth(self, num_layers):
        self.positions = (
            nuwa_decoder_position(
                self.stage, self.spec, num_layers, backbone=self.backbone
            ),
        )

    def new_call(self):
        return _NuwaCall(self)


class _NuwaCall(DecoderSelectionCall):
    def __init__(self, method):
        self.method = method
        self.scores = None

    def capture_input(self, layer_index, image_mask, *, padding_mask, labels):
        if layer_index not in self.method.score_layers:
            return None
        queries = fastv_scorer_query_indices(
            image_mask,
            score_source="last_token",
            attention_mask=padding_mask,
            labels=labels,
        )
        return AttentionCapture(layer_index, layer_index, queries)

    def record_attention(
        self, capture, *, query, image_keys, attention_keys, visibility, image_indices
    ):
        self.scores = pyramid_attention_scores(
            rules=self.method.rules,
            instruction_query=query,
            image_keys=image_keys,
            image_count=int(image_indices.numel()),
            device=image_keys.device,
            attention_keys=attention_keys,
            attention_key_mask=visibility,
            image_key_indices=image_indices,
        )

    def select(
        self,
        position,
        hidden_states,
        *,
        source_layout=None,
        result_layout=None,
        **context,
    ):
        method = self.method
        if method.score_layers and self.scores is None:
            raise InvariantError(
                "Nüwa Stage 2 attention scores were not captured before application."
            )
        result = nuwa_stage2_image_tokens(
            hidden_states,
            image_attention_scores=self.scores,
            spec=method.spec,
            rules=method.rules,
            source_layout={**(source_layout or {}), "stage": "stage2"},
            result_layout={
                **(result_layout or {}),
                "score_layer": method.spec.stage2_score_layer,
                "position_policy": (
                    "preserve_stage1_rpme"
                    if method.backbone == "llava_onevision"
                    else "native_sparse_mrope_rpme"
                ),
            },
            **context,
        )
        self.scores = None
        return result


class SGLDecoderMethod:
    """Target selection with scores supplied by the existing guide lifecycle."""

    algorithm = "sgl"
    cache_policy = CacheRepairPolicy.COMPACT_PREFIX
    position_policy = "sparse_source"
    single_image_only = True
    score_layers = ()

    def __init__(self, *, stage, spec, rules, consume_scores, num_layers):
        self.stage, self.spec, self.rules = stage, spec, rules
        self.consume_scores = consume_scores
        self.target_site = decoder_token_operation(stage, "sgl")["target_site"]
        self.resolve_depth(num_layers)

    def resolve_depth(self, num_layers):
        self.positions = (
            decoder_token_position(
                self.stage,
                self.spec.prune_layer(num_layers),
                num_layers,
                algorithm="sgl",
            ),
        )

    def new_call(self):
        return _SGLCall(self)


class _SGLCall(DecoderSelectionCall):
    def __init__(self, method):
        self.method = method

    def select(self, position, hidden_states, *, image_indices, **context):
        method = self.method
        scores = method.consume_scores(
            target_visual_tokens=int(image_indices.numel()), device=hidden_states.device
        )
        return sgl_image_tokens(
            hidden_states,
            image_indices=image_indices,
            image_scores=scores,
            spec=method.spec,
            rules=method.rules,
            **context,
        )


class FastVDecoderMethod:
    legacy_config_keys = frozenset(field.name for field in fields(FastVSpec))
    algorithm = "fastv"
    cache_policy = CacheRepairPolicy.COMPACT_PREFIX
    position_policy = "sparse_source"
    single_image_only = False

    def __init__(self, *, stage, config, rules, backbone):
        self.stage = stage
        self.algorithm = getattr(stage, "algorithm", "fastv")
        self.spec = FastVSpec(**dict(config))
        self.score_layers = () if self.spec.prune_layer == 0 else (self.spec.prune_layer - 1,)
        self.rules = rules
        self.backbone = backbone
        self.target_site = decoder_token_operation(stage)["target_site"]
        self.positions = (decoder_token_position(stage, self.spec.prune_layer),)

    def resolve_depth(self, num_layers):
        self.positions = (decoder_token_position(self.stage, self.spec.prune_layer, num_layers),)

    def new_call(self):
        return _FastVCall(self)

    def select(self, hidden_states, **kwargs):
        return fastv_image_tokens(hidden_states, spec=self.spec, **self.rules, **kwargs)


class DARTDecoderMethod:
    """Fixed scorer-layer keys and completed, normalized hidden states."""

    legacy_config_keys = frozenset(field.name for field in fields(DARTSpec))
    cache_policy = CacheRepairPolicy.COMPACT_PREFIX
    position_policy = "sparse_source"
    single_image_only = True

    def __init__(self, *, stage, config, rules, backbone):
        self.algorithm = getattr(stage, "algorithm", "dart")
        self.stage, self.rules, self.backbone = stage, rules, backbone
        self.spec = DARTSpec(**dict(config))
        self.score_layers = (self.spec.prune_after_layer - 1,)
        self.target_site = decoder_token_operation(stage, "dart", backbone=backbone)["target_site"]
        self.resolve_depth(None)

    def resolve_depth(self, num_layers):
        self.positions = (decoder_token_position(self.stage, self.spec.prune_after_layer,
            num_layers, algorithm="dart", backbone=self.backbone),)

    def new_call(self):
        return _DARTCall(self)

    def select(self, hidden_states, *, normalized_hidden_states, key_states, **kwargs):
        return dart_image_tokens(hidden_states, normalized_hidden_states, key_states,
            spec=self.spec, rules=self.rules, **kwargs)


class _DARTCall(DecoderSelectionCall):
    def __init__(self, method):
        self.method = method
        self.output_hidden_layer = method.score_layers[0]
        self.keys = self.normalized_hidden = self.text_candidates = None

    def capture_input(self, layer_index, image_mask, *, padding_mask, labels):
        if layer_index != self.method.score_layers[0]:
            return None
        self.text_candidates = dart_post_image_text_indices(
            image_mask, attention_mask=padding_mask, labels=labels)
        images = image_mask.nonzero(as_tuple=False).flatten()
        return AttentionCapture(layer_index, layer_index, images[:1])

    def record_attention(self, capture, *, query, image_keys, attention_keys, visibility, image_indices):
        self.keys = attention_keys

    def record_output_hidden(self, layer_index, hidden_states):
        if layer_index != self.output_hidden_layer:
            raise InvariantError("DART normalized hidden states came from the wrong layer.")
        self.normalized_hidden = hidden_states
        self.output_hidden_layer = None

    def select(self, position, hidden_states, **kwargs):
        if position not in self.method.positions:
            raise InvariantError("DART selection ran at an unregistered boundary.")
        if self.keys is None or self.normalized_hidden is None or self.text_candidates is None:
            raise InvariantError("DART fixed pivot state was not captured before its boundary.")
        result = self.method.select(hidden_states, normalized_hidden_states=self.normalized_hidden,
            key_states=self.keys, text_candidate_indices=self.text_candidates, **kwargs)
        self.keys = self.normalized_hidden = self.text_candidates = None
        return result


class D2PrunerDecoderMethod:
    """Own the calibrated prior and per-call last-prompt attention selection."""

    legacy_config_keys = frozenset(field.name for field in fields(D2PrunerSpec))
    cache_policy = CacheRepairPolicy.COMPACT_PREFIX
    position_policy = "sparse_source"
    single_image_only = True

    def __init__(self, *, stage, config, rules, backbone):
        self.algorithm = getattr(stage, "algorithm", "d2pruner")
        self.stage, self.rules, self.backbone = stage, rules, backbone
        self.spec = D2PrunerSpec(**dict(config))
        if self.spec.bias_prior_path is None:
            raise ValueError("bias_prior_path is required for D²Pruner execution.")
        self.score_layers = (self.spec.prune_after_layer - 1,)
        self.target_site = decoder_token_operation(stage, "d2pruner", backbone=backbone)["target_site"]
        self.resolve_depth(None)
        self.prior = None

    def resolve_depth(self, num_layers):
        self.positions = (decoder_token_position(self.stage, self.spec.prune_after_layer,
            num_layers, algorithm="d2pruner", backbone=self.backbone),)

    def prepare(self, runtime):
        """Load once before hook installation, bound to the executing checkpoint."""
        if self.prior is not None:
            return
        model_id = getattr(runtime, "model_path_or_id", None)
        if not isinstance(model_id, str) or not model_id.strip():
            raise ValueError("D²Pruner requires runtime.model_path_or_id to validate its calibrated bias prior.")
        self.prior = load_d2pruner_bias_prior(self.spec.bias_prior_path,
            expected_model_id=model_id, expected_scorer_layer=self.score_layers[0])

    def new_call(self):
        return _D2PrunerCall(self)

    def select(self, hidden_states, **kwargs):
        if self.prior is None:
            raise InvariantError("D²Pruner calibrated bias prior was not prepared before execution.")
        prior, metadata = self.prior
        return d2pruner_image_tokens(hidden_states, bias_prior=prior, bias_metadata=metadata,
            spec=self.spec, rules=self.rules, **kwargs)


class _D2PrunerCall(DecoderSelectionCall):
    def __init__(self, method):
        self.method = method
        self.scores = None

    def capture_input(self, layer_index, image_mask, *, padding_mask, labels):
        if layer_index != self.method.score_layers[0]:
            return None
        queries = fastv_scorer_query_indices(image_mask, score_source="last_token",
            attention_mask=padding_mask, labels=labels)
        return AttentionCapture(layer_index, layer_index, queries)

    def record_attention(self, capture, *, query, image_keys, attention_keys, visibility, image_indices):
        self.scores = pyramid_attention_scores(rules=self.method.rules,
            instruction_query=query, image_keys=image_keys, image_count=int(image_indices.numel()),
            device=image_keys.device, attention_keys=attention_keys,
            attention_key_mask=visibility, image_key_indices=image_indices)

    def select(self, position, hidden_states, **kwargs):
        if position not in self.method.positions:
            raise InvariantError("D²Pruner selection ran at an unregistered boundary.")
        if self.scores is None:
            raise InvariantError("D²Pruner fixed attention scores were not captured before its boundary.")
        result = self.method.select(hidden_states, image_attention_scores=self.scores, **kwargs)
        self.scores = None
        return result


class FEATHERDecoderMethod:
    """Two completed-layer, no-RoPE scores with the original stage budgets."""

    legacy_config_keys = frozenset(field.name for field in fields(FEATHERSpec)) | {
        "single_image_only"
    }
    cache_policy = CacheRepairPolicy.COMPACT_PREFIX
    position_policy = "sparse_source"
    single_image_only = True

    def __init__(self, *, stage, config, rules, backbone):
        self.algorithm = getattr(stage, "algorithm", "feather")
        self.stage, self.rules, self.backbone = stage, rules, backbone
        values = dict(config)
        values.pop("single_image_only", None)
        if "stage_layers" in values:
            values["stage_layers"] = tuple(
                int(value) for value in values["stage_layers"]
            )
        self.spec = FEATHERSpec(**values)
        self.score_layers = tuple(layer - 1 for layer in self.spec.stage_layers)
        self.target_site = decoder_token_operation(stage, "feather", backbone=backbone)[
            "target_site"
        ]
        self.resolve_depth(None)

    def resolve_depth(self, num_layers):
        self.positions = feather_decoder_positions(
            self.stage, self.spec.stage_layers, num_layers, backbone=self.backbone
        )

    def new_call(self):
        return _FEATHERCall(self)

    def select(self, hidden_states, **kwargs):
        return feather_image_tokens(
            hidden_states, spec=self.spec, rules=self.rules, **kwargs
        )


class _FEATHERCall(DecoderSelectionCall):
    def __init__(self, method):
        self.method = method
        self.output_feature_layers = method.score_layers
        self.initial_visual_tokens = None
        self.scores = {}

    def capture_input(self, layer_index, image_mask, *, padding_mask, labels):
        if self.initial_visual_tokens is None:
            self.initial_visual_tokens = int(image_mask.sum().item())
        return None

    def record_output_features(
        self,
        layer_index,
        hidden_states,
        *,
        decoder_layer,
        position_embeddings,
        image_mask,
        padding_mask
    ):
        images = image_mask.nonzero(as_tuple=False).flatten()
        queries = fastv_scorer_query_indices(
            image_mask,
            score_source="last_token",
            attention_mask=padding_mask,
            labels=None,
        )
        self.scores[layer_index] = feather_no_rope_scores(
            hidden_states,
            decoder_layer,
            image_indices=images,
            query_indices=queries,
            rules=self.method.rules,
        )

    def select(self, position, hidden_states, *, image_indices, **kwargs):
        stage_index = self.method.positions.index(position)
        scores = self.scores.pop(self.method.score_layers[stage_index], None)
        if scores is None or self.initial_visual_tokens is None:
            raise InvariantError(
                "FEATHER fixed scoring state is unavailable at application."
            )
        result = self.method.select(
            hidden_states,
            image_indices=image_indices,
            image_attention_scores=scores,
            before_layer=self.method.spec.stage_layers[stage_index],
            initial_visual_tokens=self.initial_visual_tokens,
            **kwargs
        )
        _remap_pending_scores(self.scores, image_indices, result)
        return result


class PACTDecoderMethod:
    """Fixed-source scoring and clustering with subsequent mass-weighted attention."""

    requires_eager_attention = True
    legacy_config_keys = frozenset(field.name for field in fields(PACTSpec))
    cache_policy = CacheRepairPolicy.LAYER_LOCAL
    single_image_only = True

    def __init__(self, *, stage, config, rules, backbone):
        self.algorithm = getattr(stage, "algorithm", "pact")
        self.stage, self.rules = stage, rules
        self.spec = PACTSpec(**config)
        self.score_layers = (self.spec.boundary_layer,)
        self.position_policy = (
            "contiguous" if backbone == "llava_onevision" else "sparse_source"
        )
        self.target_site = decoder_token_operation(stage, "pact")["target_site"]
        self.resolve_depth(None)

    def resolve_depth(self, num_layers):
        self.positions = (pact_decoder_position(self.stage, self.spec, num_layers),)

    def new_call(self):
        return _PACTCall(self)

    def select(self, hidden_states, *, image_indices, protected_indices=(), **kwargs):
        return pact_image_tokens(
            hidden_states,
            image_indices=_unprotected_image_indices(image_indices, protected_indices),
            spec=self.spec,
            rules=self.rules,
            **kwargs
        )


def _unprotected_image_indices(image_indices, protected_indices):
    if not protected_indices:
        return image_indices
    import torch

    protected = torch.tensor(tuple(protected_indices), device=image_indices.device)
    return image_indices[~torch.isin(image_indices, protected)]


class _PACTCall(DecoderSelectionCall):
    def __init__(self, method):
        self.method = method
        self.output_projection_layers = {
            method.spec.boundary_layer: method.spec.reduction_layer
        }
        self.payload = None
        self.token_sizes = None

    def record_output_projection(
        self,
        layer_index,
        hidden_states,
        *,
        features,
        image_indices,
        protected_indices=()
    ):
        query, keys, clustering_keys = features
        images = _unprotected_image_indices(image_indices, protected_indices)
        scores = pact_token_scores(
            query,
            keys,
            images,
            hidden_norms=(
                hidden_states.detach().float().norm(dim=-1)
                if self.method.spec.multiply_by_hidden_norm
                else None
            ),
            rules=self.method.rules,
        )
        self.payload = scores, clustering_keys

    def select(self, position, hidden_states, **kwargs):
        if self.payload is None:
            raise InvariantError(
                "PACT fixed metric sources are unavailable at application."
            )
        scores, clustering_keys = self.payload
        result = self.method.select(
            hidden_states,
            image_attention_scores=scores,
            clustering_keys=clustering_keys,
            **kwargs
        )
        self.payload = None
        if self.method.spec.proportional_attention:
            self.token_sizes = result.token_sizes
        return result

    def prepare_attention_mask(self, base_mask, *, query_length, reference):
        if self.token_sizes is None:
            return base_mask
        return pact_attention_mask(
            base_mask, self.token_sizes, query_length=query_length, reference=reference
        )


class SparseVLMDecoderMethod:
    """Prompt-rater selection and staged attention-guided recycling."""

    legacy_config_keys = frozenset(field.name for field in fields(SparseVLMSpec))
    cache_policy = CacheRepairPolicy.COMPACT_PREFIX
    position_policy = "contiguous"
    single_image_only = True

    def __init__(self, *, stage, config, rules, backbone):
        self.algorithm = getattr(stage, "algorithm", "sparsevlm")
        self.stage, self.rules, self.backbone = stage, rules, backbone
        self.spec = SparseVLMSpec(**config)
        self.score_layers = self.spec.prune_layers
        self.target_site = decoder_token_operation(
            stage, "sparsevlm", backbone=backbone
        )["target_site"]
        self.resolve_depth(None)

    def resolve_depth(self, num_layers):
        self.positions = sparsevlm_decoder_positions(
            self.stage, self.spec, num_layers, backbone=self.backbone
        )

    def new_call(self):
        return _SparseVLMCall(self)


class _SparseVLMCall(DecoderSelectionCall):
    def __init__(self, method):
        self.method = method
        self.raters = None
        self.initial_visual_tokens = None
        self.scores = {}

    def initialize(self, input_embeddings, image_mask, *, padding_mask, labels):
        images = image_mask.nonzero(as_tuple=False).flatten()
        candidates = dart_post_image_text_indices(
            image_mask, attention_mask=padding_mask, labels=labels
        )
        self.raters = select_sparsevlm_text_raters(
            input_embeddings,
            image_indices=images,
            text_candidate_indices=candidates,
            rules=self.method.rules,
        ).rater_indices
        self.initial_visual_tokens = int(images.numel())

    def select_before_input(self, layer_index):
        point = LayerPosition("inter_layer", layer_index)
        return (
            layer_index in self.method.score_layers
            and point in self.method.positions
            and self.method.positions.index(point)
            < self.method.score_layers.index(layer_index)
        )

    def capture_input(self, layer_index, image_mask, *, padding_mask, labels):
        if layer_index in self.method.score_layers:
            return AttentionCapture(layer_index, layer_index, self.raters)
        return None

    def record_attention(
        self, capture, *, query, image_keys, attention_keys, visibility, image_indices
    ):
        self.scores[capture.source_layer] = pyramid_attention_scores(
            rules=self.method.rules,
            instruction_query=query,
            image_keys=image_keys,
            image_count=int(image_indices.numel()),
            device=image_keys.device,
            attention_keys=attention_keys,
            attention_key_mask=visibility,
            image_key_indices=image_indices,
        )

    def select(self, position, hidden_states, *, image_indices, **kwargs):
        import torch

        source = self.method.score_layers[self.method.positions.index(position)]
        scores = self.scores.pop(source, None)
        if scores is None or self.initial_visual_tokens is None:
            raise InvariantError(
                "SparseVLM fixed attention scores are unavailable at application."
            )
        result = sparsevlm_image_tokens(
            hidden_states,
            image_indices=image_indices,
            image_attention_scores=scores,
            layer_index=source,
            initial_visual_tokens=self.initial_visual_tokens,
            spec=self.method.spec,
            rules=self.method.rules,
            **kwargs
        )
        delta = result.layout_delta
        raters = [delta.old_to_new[int(index)] for index in self.raters.tolist()]
        if any(index is None for index in raters) or len(set(raters)) != len(raters):
            raise InvariantError(
                "SparseVLM text-rater identity was not preserved by LayoutDelta."
            )
        self.raters = torch.tensor(
            raters, dtype=torch.long, device=hidden_states.device
        )
        for pending_source, pending in self.scores.items():
            self.scores[pending_source] = remap_sparsevlm_attention_scores(
                pending, delta, image_indices
            )
        return result


class FinePrunerDecoderMethod:
    """Application-feature IAC followed by fixed-source, per-head ARP scores."""

    legacy_config_keys = frozenset(field.name for field in fields(FinePrunerSpec))
    cache_policy = CacheRepairPolicy.LAYER_LOCAL
    position_policy = "contiguous"
    single_image_only = True

    def __init__(self, *, stage, config, rules, backbone):
        self.algorithm = getattr(stage, "algorithm", "finepruner")
        self.stage, self.rules, self.backbone = stage, rules, backbone
        self.spec = FinePrunerSpec(**config)
        self.score_layers = tuple(
            layer - 1 for layer in self.spec.arp_target_after_layers
        )
        self.target_site = decoder_token_operation(
            stage, "finepruner", backbone=backbone
        )["target_site"]
        self.resolve_depth(None)

    def resolve_depth(self, num_layers):
        self.positions = finepruner_decoder_positions(
            self.stage, self.spec, num_layers, backbone=self.backbone
        )

    def new_call(self):
        return _FinePrunerCall(self)


class _FinePrunerCall(DecoderSelectionCall):
    def __init__(self, method):
        self.method = method
        self.initial_visual_tokens = None
        self.attentions = []

    def select_before_input(self, layer_index):
        # An IAC stage at this entry changes the span observed by ARP here.
        return LayerPosition("inter_layer", layer_index) in self.method.positions[:-1]

    def capture_input(self, layer_index, image_mask, *, padding_mask, labels):
        if self.initial_visual_tokens is None:
            self.initial_visual_tokens = int(image_mask.sum().item())
        if layer_index in self.method.score_layers:
            queries = fastv_scorer_query_indices(
                image_mask,
                score_source="last_token",
                attention_mask=padding_mask,
                labels=None,
            )
            return AttentionCapture(layer_index, layer_index, queries)
        return None

    def record_attention(
        self, capture, *, query, image_keys, attention_keys, visibility, image_indices
    ):
        self.attentions.append(
            finepruner_attention_head_slice(
                query=query,
                attention_keys=attention_keys,
                image_key_indices=image_indices,
                attention_key_mask=visibility,
            )
        )

    def select(self, position, hidden_states, *, image_indices, **kwargs):
        import torch

        if self.initial_visual_tokens is None:
            # A legal before-layer-0 IAC boundary precedes the first observation.
            self.initial_visual_tokens = int(image_indices.numel())
        spec = self.method.spec
        stage_index = self.method.positions.index(position)
        common = dict(
            image_indices=image_indices,
            initial_visual_tokens=self.initial_visual_tokens,
            spec=spec,
            rules=self.method.rules,
            **kwargs
        )
        if stage_index < len(spec.iac_after_layers):
            result = finepruner_iac_image_tokens(
                hidden_states,
                completed_layers=spec.iac_after_layers[stage_index],
                **common
            )
            local_keep = (
                torch.isin(image_indices, result.kept_indices.to(image_indices.device))
                .nonzero(as_tuple=False)
                .flatten()
            )
            # Keep each source's original all-key softmax denominator.
            self.attentions = [
                scores.index_select(-1, local_keep.to(scores.device))
                for scores in self.attentions
            ]
            return result
        if len(self.attentions) != len(self.method.score_layers):
            raise InvariantError(
                "FinePruner ARP requires every fixed scoring-layer attention."
            )
        result = finepruner_arp_image_tokens(
            hidden_states,
            self.attentions,
            completed_layers=spec.arp_apply_after_layer,
            **common
        )
        self.attentions.clear()
        return result


class IVCPruneDecoderMethod:
    """Native RoPE coordinates and values from one completed scoring layer."""

    legacy_config_keys = frozenset(field.name for field in fields(IVCPruneSpec)) | {
        "single_image_only"
    }
    cache_policy = CacheRepairPolicy.COMPACT_PREFIX
    position_policy = "sparse_source"
    single_image_only = True

    def __init__(self, *, stage, config, rules, backbone):
        self.algorithm = getattr(stage, "algorithm", "ivc_prune")
        self.stage, self.rules, self.backbone = stage, rules, backbone
        values = dict(config)
        values.pop("single_image_only", None)
        self.spec = IVCPruneSpec(**values)
        self.score_layers = (self.spec.prune_layer - 1,)
        self.target_site = decoder_token_operation(
            stage, "ivc_prune", backbone=backbone
        )["target_site"]
        self.resolve_depth(None)

    def resolve_depth(self, num_layers):
        self.positions = (
            decoder_token_position(
                self.stage,
                self.spec.prune_layer,
                num_layers,
                algorithm="ivc_prune",
                backbone=self.backbone,
            ),
        )

    def new_call(self):
        return _IVCPruneCall(self)

    def select(self, hidden_states, **kwargs):
        return ivc_prune_image_tokens(
            hidden_states, spec=self.spec, rules=self.rules, **kwargs
        )


class _IVCPruneCall(DecoderSelectionCall):
    def __init__(self, method):
        self.method = method
        self.output_feature_layers = method.score_layers
        self.payload = None

    def record_output_features(
        self,
        layer_index,
        hidden_states,
        *,
        decoder_layer,
        position_embeddings,
        image_mask,
        padding_mask
    ):
        images = image_mask.nonzero(as_tuple=False).flatten()
        cosine, sine = ivc_coordinate_scores(
            position_embeddings,
            image_indices=images,
            sequence_length=int(hidden_states.shape[0]),
            rules=self.method.rules,
        )
        values = project_ivc_value_states(hidden_states, decoder_layer)
        queries = ivc_post_image_text_indices(
            image_mask,
            attention_mask=padding_mask,
            skip=self.method.spec.post_image_text_skip,
        )
        self.payload = cosine, sine, values, queries

    def select(self, position, hidden_states, **kwargs):
        if position not in self.method.positions or self.payload is None:
            raise InvariantError(
                "IVC-Prune fixed coordinate/value state is unavailable at application."
            )
        cosine, sine, values, queries = self.payload
        result = self.method.select(
            hidden_states,
            cosine_scores=cosine,
            sine_scores=sine,
            value_states=values,
            text_query_indices=queries,
            before_layer=position.index,
            **kwargs
        )
        self.payload = None
        return result


class PyramidDecoderMethod:
    legacy_config_keys = frozenset(field.name for field in fields(PyramidDropSpec)) | {"layer_indices", "spec"}
    algorithm = "pyramid_drop"
    cache_policy = CacheRepairPolicy.LAYER_LOCAL
    position_policy = "contiguous"
    single_image_only = False

    def __init__(self, *, stage, config, rules, backbone):
        self.stage = stage
        self.algorithm = getattr(stage, "algorithm", "pyramid_drop")
        self.position_policy = "contiguous" if backbone == "llava_onevision" else "sparse_source"
        unknown = set(config) - self.legacy_config_keys
        if unknown:
            raise ValueError(f"Unknown PyramidDrop config keys: {sorted(unknown)!r}.")
        self.config = dict(config)
        self.rules = rules
        self.backbone = backbone
        self.spec = self._spec(config)
        self.target_site = decoder_token_operation(stage, "pyramid_drop", backbone=backbone)["target_site"]
        self.positions = ()
        self.score_layers = ()
        self.schedule = None

    @staticmethod
    def _spec(config):
        fields = {name: config[name] for name in (
            "keep_ratio", "minimum_keep_tokens", "num_stages", "restore_source_order"
        ) if name in config}
        supplied = config.get("spec")
        if supplied is not None and fields:
            raise ValueError("Configure PyramidDrop with either spec or individual spec fields, not both.")
        return supplied if supplied is not None else PyramidDropSpec(**fields)

    def resolve_depth(self, num_layers):
        raw = self.config.get("layer_indices")
        if raw is None:
            schedule = PyramidDropSchedule.uniform(num_layers,
                num_stages=self.spec.num_stages, keep_ratio=self.spec.keep_ratio)
        else:
            if isinstance(raw, (str, bytes)):
                raise ValueError("layer_indices must be a sequence of integers.")
            indices = tuple(raw)
            if not indices or any(isinstance(i, bool) or not isinstance(i, int) for i in indices):
                raise ValueError("layer_indices must contain one or more integers.")
            schedule = PyramidDropSchedule(layer_indices=indices, num_layers=num_layers,
                num_stages=self.config.get("num_stages", len(indices) + 1), keep_ratio=self.spec.keep_ratio)
        indices = schedule.layer_indices
        if tuple(sorted(set(indices))) != indices:
            raise ValueError("PyramidDrop layer_indices must be unique and strictly increasing.")
        if any(i < 0 or i >= num_layers - 1 for i in indices):
            raise ValueError("PyramidDrop boundaries must leave at least one later decoder layer.")
        self.positions = pyramid_decoder_positions(self.stage, indices, num_layers, backbone=self.backbone)
        self.schedule = schedule
        self.score_layers = schedule.layer_indices

    def new_call(self):
        if self.schedule is None:
            raise InvariantError("PyramidDrop decoder depth has not been resolved.")
        return _PyramidCall(self)

    def select(self, hidden_states, **kwargs):
        fields = {name: kwargs.pop(name) for name in (
            "spec", "keep_ratio", "minimum_keep_tokens", "num_stages", "restore_source_order"
        ) if name in kwargs}
        spec = self._spec(fields) if fields else self.spec
        return pyramid_drop(hidden_states, spec=spec, rules=self.rules, **kwargs)


class _FastVCall(DecoderSelectionCall):
    def __init__(self, method):
        self.method = method
        self.payload = (None, None, None) if method.spec.prune_layer == 0 else None

    def capture_input(self, layer_index, image_mask, *, padding_mask, labels):
        if layer_index != self.method.spec.prune_layer - 1:
            return None
        queries = fastv_scorer_query_indices(image_mask,
            score_source=self.method.spec.score_source, attention_mask=padding_mask, labels=labels)
        return AttentionCapture(layer_index, layer_index, queries)

    def capture_output(self, layer_index, image_mask, *, padding_mask, labels):
        return None

    def record_attention(self, capture, *, query, image_keys, attention_keys, visibility, image_indices):
        self.payload = (query, attention_keys, visibility)

    def select(self, position, hidden_states, *, image_indices, **kwargs):
        if position not in self.method.positions:
            raise InvariantError("FastV selection ran at an unregistered boundary.")
        if self.payload is None:
            raise InvariantError("FastV scorer state was not captured before its boundary.")
        query, keys, visibility = self.payload
        self.payload = None
        return self.method.select(hidden_states, image_indices=image_indices,
            scorer_query=query, attention_keys=keys, attention_key_mask=visibility, **kwargs)


class _PyramidCall(DecoderSelectionCall):
    def __init__(self, method):
        self.method = method
        self.scores = {}

    def capture_input(self, layer_index, image_mask, *, padding_mask, labels):
        return None

    def capture_output(self, layer_index, image_mask, *, padding_mask, labels):
        if layer_index not in self.method.schedule.layer_indices or not bool(image_mask.any().item()):
            return None
        queries = pyramid_instruction_indices(image_mask, attention_mask=padding_mask, labels=labels)
        return AttentionCapture(layer_index, layer_index + 1, queries)

    def record_attention(self, capture, *, query, image_keys, attention_keys, visibility, image_indices):
        self.scores[capture.source_layer] = pyramid_attention_scores(
            rules=self.method.rules, instruction_query=query, image_keys=image_keys,
            image_count=int(image_indices.numel()), device=image_keys.device,
            attention_keys=attention_keys, attention_key_mask=visibility, image_key_indices=image_indices)

    def select(self, position, hidden_states, *, image_indices, **kwargs):
        stage_index = self.method.positions.index(position)
        source = self.method.schedule.layer_indices[stage_index]
        scores = self.scores.pop(source, None)
        if scores is None:
            raise InvariantError("PyramidDrop fixed scores were not captured before application.")
        result = self.method.select(hidden_states, image_indices=image_indices,
            image_attention_scores=scores, **kwargs)
        import torch
        local_keep = torch.isin(image_indices, result.kept_indices.to(image_indices.device)).nonzero(as_tuple=False).flatten()
        for source, pending in self.scores.items():
            self.scores[source] = pending.index_select(0, local_keep.to(pending.device))
        return result


class BTPDecoderMethod:
    legacy_config_keys = frozenset(field.name for field in fields(BalancedTokenPruningSpec))
    cache_policy = CacheRepairPolicy.COMPACT_PREFIX
    position_policy = "sparse_source"
    single_image_only = True

    def __init__(self, *, stage, config, rules, backbone):
        self.algorithm = getattr(stage, "algorithm", "btp")
        self.stage, self.rules, self.backbone = stage, rules, backbone
        self.spec = BalancedTokenPruningSpec(**dict(config))
        self.schedule = self.spec.boundary_after_layer_indices
        self.score_layers = self.schedule[:-1]
        self.target_site = decoder_token_operation(stage, "btp", backbone=backbone)["target_site"]
        self.positions = btp_decoder_positions(stage, self.spec, backbone=backbone)

    def resolve_depth(self, num_layers):
        self.positions = btp_decoder_positions(self.stage, self.spec, num_layers, backbone=self.backbone)

    def new_call(self):
        return _BTPCall(self)

    def select(self, hidden_states, *, protected_indices=(), **kwargs):
        # BTP budgets the complete serialized image span, including LLaVA
        # newlines; the native adapter repairs whichever newlines survive.
        del protected_indices
        return balanced_token_pruning_image_tokens(hidden_states, spec=self.spec, rules=self.rules, **kwargs)


class HalfVDecoderMethod:
    legacy_config_keys = frozenset(field.name for field in fields(HalfVSpec))
    cache_policy = CacheRepairPolicy.COMPACT_PREFIX
    position_policy = "sparse_source"
    single_image_only = True

    def __init__(self, *, stage, config, rules, backbone):
        self.algorithm = getattr(stage, "algorithm", "halfv")
        self.stage, self.rules, self.backbone = stage, rules, backbone
        self.spec = HalfVSpec(**dict(config))
        self.schedule = self.spec.boundary_after_layer_indices
        self.score_layers = self.schedule
        self.target_site = decoder_token_operation(stage, "halfv", backbone=backbone)["target_site"]
        self.positions = halfv_decoder_positions(stage, self.spec, backbone=backbone)

    def resolve_depth(self, num_layers):
        self.positions = halfv_decoder_positions(self.stage, self.spec, num_layers, backbone=self.backbone)

    def new_call(self):
        return _HalfVCall(self)

    def select(self, hidden_states, *, protected_indices=(), **kwargs):
        del protected_indices  # HalfV also budgets the complete serialized image span.
        return halfv_image_tokens(hidden_states, spec=self.spec, rules=self.rules, **kwargs)


class _InputScoreCall(DecoderSelectionCall):
    """Fixed input scores which follow survivors across delayed selections."""

    def __init__(self, method):
        self.method = method
        self.scores = {}

    def select_before_input(self, layer_index):
        entry = LayerPosition("inter_layer", layer_index)
        return (layer_index in self.method.score_layers
            and entry in self.method.positions
            and self.method.positions.index(entry) < self.method.score_layers.index(layer_index))

    def capture_input(self, layer_index, image_mask, *, padding_mask, labels):
        if layer_index not in self.method.score_layers or not bool(image_mask.any().item()):
            return None
        queries = fastv_scorer_query_indices(image_mask, score_source="last_token",
            attention_mask=padding_mask, labels=None)
        return AttentionCapture(layer_index, layer_index, queries)


def _remap_pending_scores(scores, image_indices, result):
    import torch

    local_keep = (
        torch.isin(image_indices, result.kept_indices.to(image_indices.device))
        .nonzero(as_tuple=False)
        .flatten()
    )
    for source, pending in scores.items():
        scores[source] = pending.index_select(0, local_keep.to(pending.device))


class _BTPCall(_InputScoreCall):
    def record_attention(self, capture, *, query, image_keys, attention_keys, visibility, image_indices):
        self.scores[capture.source_layer] = pyramid_attention_scores(
            rules=self.method.rules, instruction_query=query, image_keys=image_keys,
            image_count=int(image_indices.numel()), device=image_keys.device,
            attention_keys=attention_keys, attention_key_mask=visibility, image_key_indices=image_indices)

    def select(self, position, hidden_states, *, image_indices, **kwargs):
        stage_index = self.method.positions.index(position)
        source = self.method.schedule[stage_index]
        scores = self.scores.pop(source, None)
        if stage_index < len(self.method.schedule) - 1 and scores is None:
            raise InvariantError("BTP fixed attention scores were not captured before application.")
        result = self.method.select(hidden_states, image_indices=image_indices,
            boundary_after_layer=source, image_attention_scores=scores, **kwargs)
        _remap_pending_scores(self.scores, image_indices, result)
        return result


class _HalfVCall(_InputScoreCall):
    def __init__(self, method):
        super().__init__(method)
        self.initial_visual_tokens = None
        self.selection_hidden = None
        self.output_hidden_layer = method.schedule[0]

    def capture_input(self, layer_index, image_mask, *, padding_mask, labels):
        if self.initial_visual_tokens is None:
            self.initial_visual_tokens = int(image_mask.sum().item())
        capture = super().capture_input(layer_index, image_mask, padding_mask=padding_mask, labels=labels)
        if capture is None:
            return None
        return AttentionCapture(capture.source_layer, capture.projection_layer,
            capture.query_indices, use_rope=layer_index != self.method.schedule[0])

    def record_output_hidden(self, layer_index, hidden_states):
        if layer_index != self.output_hidden_layer:
            raise InvariantError("HalfV normalized hidden states came from the wrong layer.")
        self.selection_hidden = hidden_states
        self.output_hidden_layer = None

    def record_attention(self, capture, *, query, image_keys, attention_keys, visibility, image_indices):
        self.scores[capture.source_layer] = halfv_last_text_attention_scores(query, image_keys, rules=self.method.rules)

    def select(self, position, hidden_states, *, image_indices, **kwargs):
        stage_index = self.method.positions.index(position)
        source = self.method.schedule[stage_index]
        scores = self.scores.pop(source, None)
        if scores is None or self.initial_visual_tokens is None or (stage_index == 0 and self.selection_hidden is None):
            raise InvariantError("HalfV fixed scoring state is unavailable at application.")
        result = self.method.select(hidden_states, image_indices=image_indices,
            boundary_after_layer=source, image_attention_scores=scores,
            initial_visual_tokens=self.initial_visual_tokens,
            selection_hidden_states=self.selection_hidden if stage_index == 0 else None, **kwargs)
        self.selection_hidden = None
        _remap_pending_scores(self.scores, image_indices, result)
        return result


class TransPruneDecoderMethod:
    legacy_config_keys = frozenset(field.name for field in fields(TransPruneSpec))
    cache_policy = CacheRepairPolicy.LAYER_LOCAL
    position_policy = "contiguous"
    single_image_only = True

    def __init__(self, *, stage, config, rules, backbone):
        self.algorithm = getattr(stage, "algorithm", "transprune")
        self.stage, self.rules, self.backbone = stage, rules, backbone
        self.spec = TransPruneSpec(**dict(config))
        self.schedule = self.spec.boundary_after_layer_indices
        self.score_layers = self.schedule
        self.target_site = decoder_token_operation(stage, "transprune", backbone=backbone)["target_site"]
        self.positions = transprune_decoder_positions(stage, self.spec, backbone=backbone)

    def resolve_depth(self, num_layers):
        self.positions = transprune_decoder_positions(self.stage, self.spec, num_layers, backbone=self.backbone)

    def new_call(self):
        return _TransPruneCall(self)

    def select(self, hidden_states, *, protected_indices=(), **kwargs):
        del protected_indices  # Released TransPrune budgets the complete serialized span.
        return transprune_image_tokens(hidden_states, spec=self.spec, rules=self.rules, **kwargs)


class _TransPruneCall(DecoderSelectionCall):
    def __init__(self, method):
        self.method = method
        self.initial_visual_tokens = None
        self.history = []
        self.payloads = {}
        self.transition_layers = tuple(layer for layer in method.spec.accumulating_layer_indices
            if layer <= method.schedule[-1])
        self._transition_layer = None
        self._transition_images = None
        self._transition_keep = None

    def capture_input(self, layer_index, image_mask, *, padding_mask, labels):
        if self.initial_visual_tokens is None:
            self.initial_visual_tokens = int(image_mask.sum().item())
        return None

    def begin_transition(self, layer_index, image_indices):
        if layer_index not in self.transition_layers or not int(image_indices.numel()):
            return False
        self._transition_layer = layer_index
        self._transition_images = image_indices
        self._transition_keep = None
        return True

    def record_transition(self, observation, *, image_indices):
        if self._transition_layer is None:
            raise InvariantError("TransPrune received module observations without a requested source.")
        spec = self.method.spec
        self.history.append(transprune_layer_transition_variation(
            observation.attention_before, observation.attention_after,
            observation.ffn_before, observation.ffn_after,
            image_indices=image_indices, attention_image_indices=self._transition_images,
            selected_visual_indices=self._transition_keep,
            direction_rule=spec.direction_rule, softmax_scope=spec.ttv_softmax_scope,
            rules=self.method.rules))
        self._transition_layer = self._transition_images = self._transition_keep = None

    def capture_output(self, layer_index, image_mask, *, padding_mask, labels):
        if layer_index not in self.method.score_layers or not bool(image_mask.any().item()):
            return None
        queries = dart_post_image_text_indices(image_mask, attention_mask=padding_mask, labels=None)
        if not int(queries.numel()):
            raise InvariantError("TransPrune requires post-image text queries.")
        return AttentionCapture(layer_index, layer_index + 1, queries)

    def record_attention(self, capture, *, query, image_keys, attention_keys, visibility, image_indices):
        scores = transprune_instruction_guided_attention(instruction_queries=query,
            image_keys=image_keys, attention_keys=attention_keys,
            image_indices=image_indices, attention_key_mask=visibility)
        self.payloads[capture.source_layer] = (tuple(self.history), scores)

    def select(self, position, hidden_states, *, image_indices, **kwargs):
        source = self.method.schedule[self.method.positions.index(position)]
        payload = self.payloads.pop(source, None)
        if payload is None or self.initial_visual_tokens is None:
            raise InvariantError("TransPrune fixed TTV/IGA sources are unavailable at application.")
        history, scores = payload
        result = self.method.select(hidden_states, image_indices=image_indices,
            boundary_after_layer=source, ttv_history=history, image_attention_scores=scores,
            initial_visual_tokens=self.initial_visual_tokens, **kwargs)
        keep = result.selected_visual_indices
        self.history = select_transprune_ttv_history(self.history, keep)
        for source, (pending_history, pending_scores) in self.payloads.items():
            self.payloads[source] = (tuple(select_transprune_ttv_history(pending_history, keep)),
                pending_scores.index_select(0, keep.to(pending_scores.device)))
        if position.boundary == "post_attention_pre_ffn" and position.index == self._transition_layer:
            self._transition_keep = keep
        return result
