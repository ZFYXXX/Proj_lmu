"""Physical llava_onevision image callbacks, sites and layout requirements."""

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Mapping


LLAVA_ONEVISION_BRIDGE_API_VERSION = "mllm_comp.llava_onevision.image.v1"
LLAVA_ONEVISION_SOURCE_COMMIT = "ee14dbdde72ebb106ad22a2fd31ea9488f0ab97a"
LLAVA_ONEVISION_SOURCE_ROUTE = (
    "third_party/backbones/LLaVA-NeXT@"
    f"{LLAVA_ONEVISION_SOURCE_COMMIT}:"
    "llava/model/language_model/llava_qwen.py::LlavaQwenForCausalLM+"
    "llava/model/llava_arch.py::LlavaMetaForCausalLM+"
    "llava/model/multimodal_encoder/siglip_encoder.py::SigLipVisionTower"
)

TOME_MERGE_SITE = "vision_block_post_attention_residual_pre_mlp"
TOME_RESTORE_SITE = "vision_encoder_output_pre_anyres_pack"
TOME_PACK_SITE = "post_anyres_pack_pre_llm_splice_tome"
VISIONZIP_SITE = "post_anyres_pack_pre_llm_splice"
PYRAMID_DROP_SITE = "llm_decoder_layer_output_image_tokens"
RESTORE_CALIBRATION_SITE = "llm_decoder_attention_calibration"
FASTV_SITE = "llm_decoder_pre_prune_layer_image_tokens"


@dataclass(frozen=True)
class LlavaCallbackContract:
    """Machine-readable contract for one provider-emitted callback."""

    site: str
    request_type: str
    response_type: str
    source_symbol: str
    timing: str
    required_capabilities: frozenset[str] = frozenset()
    required_repairs: tuple[str, ...] = ()
    requires_packed_metrics: bool = False


@dataclass(frozen=True)
class LlavaImageBindingContract:
    """Physical callback contract for one exact LLaVA binding."""

    binding: str
    variant: str
    callbacks: tuple[LlavaCallbackContract, ...]
    image_only: bool = True
    source_route: str = LLAVA_ONEVISION_SOURCE_ROUTE
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))

    @property
    def sites(self) -> tuple[str, ...]:
        return tuple(item.site for item in self.callbacks)

    @property
    def required_capabilities(self) -> frozenset[str]:
        return frozenset(
            capability
            for callback in self.callbacks
            for capability in callback.required_capabilities
        )


LLAVA_ONEVISION_IMAGE_BINDING_CONTRACTS: Mapping[str, LlavaImageBindingContract] = MappingProxyType({
    contract.binding: contract
    for contract in (
        LlavaImageBindingContract(
            binding="tome.image.llava_onevision",
            variant="tome.image.encoder_dense_boundary",
            callbacks=(
                LlavaCallbackContract(
                    site=TOME_MERGE_SITE,
                    request_type="LlavaEncoderMergeRequest",
                    response_type="LlavaEncoderMergeResponse",
                    source_symbol=(
                        "SigLipEncoderLayer.forward immediately after "
                        "attention residual and before layer_norm2/MLP"
                    ),
                    timing="once per selected vision layer and AnyRes crop batch",
                    required_capabilities=frozenset(
                        {"crop_isolated_encoder_state"}
                    ),
                    required_repairs=(
                        "carry token_sizes into the next vision layer",
                        "add proportional_attention_bias to later attention logits",
                    ),
                ),
                LlavaCallbackContract(
                    site=TOME_RESTORE_SITE,
                    request_type="LlavaEncoderDenseRestoreRequest",
                    response_type="LlavaEncoderDenseRestoreResponse",
                    source_symbol=(
                        "SigLipVisionTower.forward after selected encoder output "
                        "and before LlavaMetaForCausalLM AnyRes packing"
                    ),
                    timing="exactly once per encoder invocation",
                    required_capabilities=frozenset(
                        {"crop_isolated_encoder_state"}
                    ),
                    required_repairs=(
                        "restore every crop to its original dense patch count",
                        "preserve crop order before AnyRes view/unpad operations",
                    ),
                ),
                LlavaCallbackContract(
                    site=TOME_PACK_SITE,
                    request_type="LlavaPackedImageRequest",
                    response_type="LlavaPackedImageResponse",
                    source_symbol=(
                        "LlavaMetaForCausalLM.prepare_inputs_labels_for_multimodal "
                        "after AnyRes packing and before LLM splice"
                    ),
                    timing="once per packed image feature sequence",
                    required_capabilities=frozenset(
                        {
                            "anyres_layout_metadata",
                            "image_newline_indices",
                            "packed_attention_and_keys",
                        }
                    ),
                    required_repairs=(
                        "protect image_newline tokens",
                        "permanently compact the LLM image span",
                        "compact mask positions and labels with representatives",
                    ),
                    requires_packed_metrics=True,
                ),
            ),
            metadata={
                'anyres_crop_isolation': True,
                'permanent_llm_token_reduction': True,
            },
        ),
        LlavaImageBindingContract(
            binding="saint.image.llava_onevision",
            variant="saint.image.packed_feature_prune",
            callbacks=(
                LlavaCallbackContract(
                    site=VISIONZIP_SITE,
                    request_type="LlavaPackedImageRequest",
                    response_type="LlavaPackedImageResponse",
                    source_symbol="post-AnyRes packed image span before LLM splice",
                    timing="once per packed image prompt",
                    required_capabilities=frozenset(
                        {"anyres_layout_metadata", "sequence_layout_repair"}
                    ),
                    required_repairs=(
                        "protect image_newline tokens",
                        "compact packed AnyRes crop boundaries",
                        "rebuild prompt mask and positions",
                    ),
                    requires_packed_metrics=True,
                ),
            ),
            metadata={
                'adaptation': 'post_encoder_feature_redundancy_prune',
                'protect_image_newline': True,
            },
        ),
        LlavaImageBindingContract(
            binding="aim.image.llava_onevision",
            variant="aim.image",
            callbacks=(
                LlavaCallbackContract(
                    site=VISIONZIP_SITE,
                    request_type="LlavaPackedImageRequest",
                    response_type="LlavaPackedImageResponse",
                    source_symbol="post-AnyRes packed image span before LLM splice",
                    timing="once per packed image prompt",
                    required_capabilities=frozenset(
                        {"anyres_layout_metadata", "sequence_layout_repair"}
                    ),
                    required_repairs=(
                        "protect image_newline tokens",
                        "compact packed AnyRes crop boundaries",
                        "preserve differentiable merge values",
                    ),
                ),
            ),
            metadata={
                'protect_image_newline': True,
                'differentiable_merge': True,
            },
        ),
        LlavaImageBindingContract(
            binding="pact.image.llava_onevision",
            variant="pact.image",
            callbacks=(
                LlavaCallbackContract(
                    site=PYRAMID_DROP_SITE,
                    request_type="LlavaPACTRequest",
                    response_type="LlavaDecoderLayerResponse",
                    source_symbol=(
                        "Qwen2 decoder loop after the configured completed layer"
                    ),
                    timing="once per prompt after reduction_layer",
                    required_capabilities=frozenset(
                        {"decoder_loop_control", "sequence_layout_repair"}
                    ),
                    required_repairs=(
                        "protect AnyRes image_newline tokens",
                        "preserve layer-local DynamicCache prefixes",
                        "apply log cluster-mass attention bias",
                    ),
                ),
            ),
            metadata={
                'protect_image_newline': True,
                'cache_policy': 'layer_local',
                'single_image_only': True,
            },
        ),
        LlavaImageBindingContract(
            binding="visionzip.image.llava_onevision",
            variant="visionzip.image",
            callbacks=(
                LlavaCallbackContract(
                    site=VISIONZIP_SITE,
                    request_type="LlavaPackedImageRequest",
                    response_type="LlavaPackedImageResponse",
                    source_symbol=(
                        "LlavaMetaForCausalLM.prepare_inputs_labels_for_multimodal "
                        "after flat/spatial AnyRes packing and immediately before "
                        "IMAGE_TOKEN_INDEX replacement"
                    ),
                    timing="once per packed image feature sequence",
                    required_capabilities=frozenset(
                        {
                            "anyres_layout_metadata",
                            "image_newline_indices",
                            "packed_attention_and_keys",
                        }
                    ),
                    required_repairs=(
                        "protect image_newline tokens as singleton groups",
                        "rewrite packed token count and newline indices",
                        "preserve image_sizes and modalities without reordering",
                    ),
                    requires_packed_metrics=True,
                ),
            ),
            metadata={
                'protect_image_newline': True,
                'per_image_budget': True,
            },
        ),
        LlavaImageBindingContract(
            binding="prumerge.image.llava_onevision",
            variant="prumerge.image",
            callbacks=(
                LlavaCallbackContract(
                    site=VISIONZIP_SITE,
                    request_type="LlavaPackedImageRequest",
                    response_type="LlavaPackedImageResponse",
                    source_symbol="post-AnyRes packed image boundary",
                    timing="once per packed image sequence",
                    required_capabilities=frozenset(
                        {"anyres_layout_metadata", "image_newline_indices", "packed_attention_and_keys"}
                    ),
                    required_repairs=(
                        "protect image_newline tokens",
                        "rewrite AnyRes packed layout",
                        "preserve one-dimensional representative positions",
                    ),
                    requires_packed_metrics=True,
                ),
            ),
            metadata={
                'protect_image_newline': True,
                'position_policy': 'source_representative',
            },
        ),
        LlavaImageBindingContract(
            binding="gap.image.llava_onevision",
            variant="gap.image",
            callbacks=(
                LlavaCallbackContract(
                    site=VISIONZIP_SITE,
                    request_type="LlavaPackedImageRequest",
                    response_type="LlavaPackedImageResponse",
                    source_symbol="post-AnyRes packed image boundary",
                    timing="once per packed image sequence",
                    required_capabilities=frozenset(
                        {"anyres_layout_metadata", "image_newline_indices", "packed_attention_and_keys"}
                    ),
                    required_repairs=(
                        "restore source token order",
                        "preserve original visual and suffix positions",
                        "rewrite packed layout without changing image identity",
                    ),
                    requires_packed_metrics=True,
                ),
            ),
            metadata={
                'protect_image_newline': True,
                'position_policy': 'preserve_original',
            },
        ),
        LlavaImageBindingContract(
            binding="pyramid_drop.image.llava_onevision",
            variant="pyramid_drop.image",
            callbacks=(
                LlavaCallbackContract(
                    site=PYRAMID_DROP_SITE,
                    request_type="LlavaDecoderLayerRequest",
                    response_type="LlavaDecoderLayerResponse",
                    source_symbol=(
                        "Qwen2 decoder loop after a completed decoder layer and "
                        "before the next layer consumes hidden/mask/position/cache"
                    ),
                    timing="after every decoder layer; compression at schedule boundaries",
                    required_capabilities=frozenset(
                        {"decoder_loop_control", "sequence_layout_repair"}
                    ),
                    required_repairs=(
                        "compact attention mask and position ids with kept_indices",
                        "validate the intentional layer-local KV-cache pyramid",
                        "rewrite image token and image_newline indices",
                        "preserve image_sizes and modalities",
                    ),
                ),
            ),
            metadata={
                'requires_decoder_loop_control': True,
                'generate_requires_kv_cache_repair': True,
            },
        ),
        LlavaImageBindingContract(
            binding="feather.image.llava_onevision",
            variant="feather.image",
            callbacks=(
                LlavaCallbackContract(
                    site=FASTV_SITE,
                    request_type="DecoderSelectionRegistration",
                    response_type="LlavaDecoderLayerResponse",
                    source_symbol="Qwen2 decoder loop before FEATHER stage layers",
                    timing="before decoder layers 8 and 16 by default",
                    required_capabilities=frozenset(
                        {"decoder_loop_control", "sequence_layout_repair"}
                    ),
                    required_repairs=(
                        "two-stage sequence compaction",
                        "preserve original position ids",
                        "compact completed DynamicCache prefix",
                    ),
                ),
            ),
            metadata={
                'cache_policy': 'compact_prefix',
                'position_policy': 'preserve_original',
                'single_image_only': True,
            },
        ),
        LlavaImageBindingContract(
            binding="ivc_prune.image.llava_onevision",
            variant="ivc_prune.image",
            callbacks=(
                LlavaCallbackContract(
                    site=FASTV_SITE,
                    request_type="DecoderSelectionRegistration",
                    response_type="LlavaDecoderLayerResponse",
                    source_symbol="Qwen2 decoder loop before IVC-Prune layer",
                    timing="before decoder layer 16 by default",
                    required_capabilities=frozenset(
                        {"decoder_loop_control", "sequence_layout_repair"}
                    ),
                    required_repairs=(
                        "native Qwen2 RoPE coordinate selection",
                        "foreground value selection",
                        "preserve original position ids and compact cache prefix",
                    ),
                ),
            ),
            metadata={
                'coordinate_source': 'native_qwen2_rope',
                'cache_policy': 'compact_prefix',
                'single_image_only': True,
            },
        ),
        LlavaImageBindingContract(
            binding="divprune.image.llava_onevision",
            variant="divprune.image",
            callbacks=(
                LlavaCallbackContract(
                    site=VISIONZIP_SITE,
                    request_type="LlavaPackedImageRequest",
                    response_type="LlavaPackedImageResponse",
                    source_symbol=(
                        "LlavaMetaForCausalLM.prepare_inputs_labels_for_multimodal "
                        "after AnyRes packing and immediately before "
                        "IMAGE_TOKEN_INDEX replacement"
                    ),
                    timing="once per packed image feature sequence",
                    required_capabilities=frozenset(
                        {"anyres_layout_metadata", "image_newline_indices"}
                    ),
                    required_repairs=(
                        "protect image_newline tokens as retained singletons",
                        "rewrite packed token count and newline indices",
                        "compact mask positions and labels with kept indices",
                        "preserve image_sizes and modalities without reordering",
                    ),
                ),
            ),
            metadata={
                'protect_image_newline': True,
                'per_image_selection': True,
                'layer_zero_pre_llm': True,
                'position_policy': 'contiguous_text_positions',
            },
        ),
        LlavaImageBindingContract(
            binding="mmtok.image.llava_onevision",
            variant="mmtok.image",
            callbacks=(
                LlavaCallbackContract(
                    site=VISIONZIP_SITE,
                    request_type="LlavaPackedImageRequest",
                    response_type="LlavaPackedImageResponse",
                    source_symbol=(
                        "LlavaMetaForCausalLM.prepare_inputs_labels_for_multimodal "
                        "after native AnyRes packing and before LLM splice"
                    ),
                    timing="once per packed image feature sequence",
                    required_capabilities=frozenset(
                        {
                            "anyres_layout_metadata",
                            "image_newline_indices",
                            "multisource_embedding_payload",
                        }
                    ),
                    required_repairs=(
                        "select every crop independently",
                        "map selected crop patches through native AnyRes unpadding",
                        "protect image_newline tokens",
                        "compact mask positions and labels with selected indices",
                    ),
                ),
            ),
            metadata={
                'per_crop_selection': True,
                'text_source': 'prompt_input_embeddings',
                'vision_source': 'siglip_pre_projector',
                'protect_image_newline': True,
                'position_policy': 'contiguous_text_positions',
            },
        ),
        LlavaImageBindingContract(
            binding="vispruner.image.llava_onevision",
            variant="vispruner.image",
            callbacks=(
                LlavaCallbackContract(
                    site=VISIONZIP_SITE,
                    request_type="LlavaPackedImageRequest",
                    response_type="LlavaPackedImageResponse",
                    source_symbol=(
                        "SigLip final visual attention and hidden states, then "
                        "native AnyRes pack before LLM splice"
                    ),
                    timing="once per packed image feature sequence",
                    required_capabilities=frozenset(
                        {
                            "anyres_layout_metadata",
                            "image_newline_indices",
                            "visual_attention_crop_payload",
                        }
                    ),
                    required_repairs=(
                        "select every crop independently from visual cues",
                        "map selected crop patches through native AnyRes unpadding",
                        "protect image_newline tokens",
                        "compact mask positions and labels with selected indices",
                    ),
                ),
            ),
            metadata={
                'per_crop_selection': True,
                'attention_source': 'siglip_final_mean_received',
                'selection_feature_source': 'siglip_pre_projector',
                'protect_image_newline': True,
                'position_policy': 'contiguous_text_positions',
            },
        ),
        LlavaImageBindingContract(
            binding="zoo_prune.image.llava_onevision",
            variant="zoo_prune.image",
            callbacks=(
                LlavaCallbackContract(
                    site=VISIONZIP_SITE,
                    request_type="LlavaPackedImageRequest",
                    response_type="LlavaPackedImageResponse",
                    source_symbol=(
                        "native SigLip output and MM projector, then native "
                        "AnyRes pack before LLM splice"
                    ),
                    timing="once per packed image feature sequence",
                    required_capabilities=frozenset(
                        {
                            "anyres_layout_metadata",
                            "image_newline_indices",
                            "projector_input_and_layers",
                        }
                    ),
                    required_repairs=(
                        "score and select every crop independently",
                        "map selected crop patches through native AnyRes unpadding",
                        "protect image_newline tokens",
                        "compact mask positions and labels with selected indices",
                    ),
                ),
            ),
            metadata={
                'per_crop_selection': True,
                'projector_input': 'siglip_pre_projector',
                'projector_proxy': 'native_mm_projector_low_rank_linear_layers',
                'protect_image_newline': True,
                'position_policy': 'contiguous_text_positions',
                'single_image_only': True,
                'training_free': True,
            },
        ),
        LlavaImageBindingContract(
            binding="nuwa.image.llava_onevision",
            variant="nuwa.image",
            callbacks=(
                LlavaCallbackContract(
                    site=VISIONZIP_SITE,
                    request_type="LlavaPackedImageRequest",
                    response_type="LlavaPackedImageResponse",
                    source_symbol=(
                        "SigLip hidden states/final attention followed by native "
                        "AnyRes packing before the LLM splice"
                    ),
                    timing="Stage 1 once per packed image sequence",
                    required_capabilities=frozenset(
                        {
                            "anyres_layout_metadata",
                            "image_newline_indices",
                            "nuwa_visual_source_payload",
                        }
                    ),
                    required_repairs=(
                        "select and aggregate every crop independently",
                        "map benchmark representatives through native AnyRes",
                        "protect image_newline tokens",
                        "build one-dimensional RPME positions",
                    ),
                ),
                LlavaCallbackContract(
                    site=FASTV_SITE,
                    request_type="DecoderSelectionRegistration",
                    response_type="LlavaDecoderLayerResponse",
                    source_symbol=(
                        "Qwen2 decoder loop after Stage-2 scorer completion and "
                        "before the configured layer consumes the compact sequence"
                    ),
                    timing="Stage 2 once before stage2_after_layer",
                    required_capabilities=frozenset(
                        {"decoder_loop_control", "sequence_layout_repair"}
                    ),
                    required_repairs=(
                        "retain Stage-1 RPME positions",
                        "compact every completed DynamicCache prefix layer",
                        "rewrite image token and image_newline indices",
                        "continue decode after the original RPME extent",
                    ),
                ),
            ),
            metadata={
                'composite_binding': True,
                'stage1_per_crop_selection': True,
                'stage1_metric': 'visual_attention_received_times_key_norm',
                'semantic_hidden_range': (16, 25),
                'protect_image_newline': True,
                'stage2_metric': 'last_prompt_attention',
                'cache_policy': 'compact_prefix',
                'cache_transform': 'drop',
                'position_policy': 'one_dimensional_rpme',
                'single_image_only': True,
            },
        ),
        LlavaImageBindingContract(
            binding="restore.image.llava_onevision",
            variant="restore.image",
            callbacks=(
                LlavaCallbackContract(
                    site=VISIONZIP_SITE,
                    request_type="LlavaPackedImageRequest",
                    response_type="LlavaPackedImageResponse",
                    source_symbol=(
                        "configured SigLIP hidden state/attention followed by "
                        "native AnyRes packing before the LLM splice"
                    ),
                    timing="once per packed image sequence",
                    required_capabilities=frozenset(
                        {
                            "anyres_layout_metadata",
                            "image_newline_indices",
                            "visual_attention_crop_payload",
                        }
                    ),
                    required_repairs=(
                        "merge visible patches independently inside each crop",
                        "protect image_newline as singleton groups",
                        "preserve real representative source positions",
                        "publish visual cluster sizes",
                    ),
                ),
                LlavaCallbackContract(
                    site=RESTORE_CALIBRATION_SITE,
                    request_type="decoder_attention_calibration",
                    response_type="decoder_attention_calibration",
                    source_symbol="Qwen2 decoder loop before every attention call",
                    timing="every prefill and decode decoder layer",
                    required_capabilities=frozenset({"decoder_loop_control"}),
                    required_repairs=(
                        "add token-mass and RoPE-distance calibration bias",
                        "keep DynamicCache shape stable",
                    ),
                ),
            ),
            metadata={
                'composite_binding': True,
                'per_crop_visual_merge': True,
                'protect_image_newline': True,
                'representative_policy': 'real_packed_source_median',
                'cache_policy': 'stable',
                'cache_transform': 'none',
                'position_policy': 'one_dimensional_source_representatives',
                'attention_calibration': 'token_mass_plus_rope_distance',
                'single_image_only': True,
            },
        ),
        LlavaImageBindingContract(
            binding="fastv.image.llava_onevision",
            variant="fastv.image",
            callbacks=(
                LlavaCallbackContract(
                    site=FASTV_SITE,
                    request_type="LlavaFastVRequest",
                    response_type="LlavaDecoderLayerResponse",
                    source_symbol=(
                        "Qwen2 decoder loop after the scorer layer and "
                        "before prune_layer consumes the compact sequence"
                    ),
                    timing="once per prompt before the configured prune layer",
                    required_capabilities=frozenset(
                        {"decoder_loop_control", "sequence_layout_repair"}
                    ),
                    required_repairs=(
                        "compact mask and retain sparse source position ids",
                        "compact every completed shallow DynamicCache layer",
                        "rewrite image token and image_newline indices",
                        "preserve image_sizes and modalities",
                    ),
                ),
            ),
            metadata={
                'single_decoder_boundary': True,
                'generate_requires_kv_cache_repair': True,
                'cache_policy': 'compact_prefix',
            },
        ),
        LlavaImageBindingContract(
            binding="sgl.image.llava_onevision",
            variant="sgl.image",
            callbacks=(
                LlavaCallbackContract(
                    site=FASTV_SITE,
                    request_type="LlavaFastVRequest",
                    response_type="LlavaDecoderLayerResponse",
                    source_symbol=(
                        "Qwen2 decoder loop before the fractional SGP prune "
                        "layer consumes the guide-selected compact sequence"
                    ),
                    timing="once per prompt after compact-guide generation",
                    required_capabilities=frozenset(
                        {"decoder_loop_control", "sequence_layout_repair"}
                    ),
                    required_repairs=(
                        "require guide/target visual-token identity",
                        "compact mask and retain sparse source position ids",
                        "compact every completed shallow DynamicCache layer",
                        "rewrite image token and image_newline indices",
                    ),
                ),
            ),
            metadata={
                'single_decoder_boundary': True,
                'generate_requires_kv_cache_repair': True,
                'cache_policy': 'compact_prefix',
                'position_policy': 'preserve_sparse_source',
                'single_image_only': True,
                'training_free': True,
                'dual_model_lifecycle': True,
                'guide_model_path_required': True,
            },
        ),
        LlavaImageBindingContract(
            binding="dart.image.llava_onevision",
            variant="dart.image",
            callbacks=(
                LlavaCallbackContract(
                    site=FASTV_SITE,
                    request_type="DecoderSelectionRegistration",
                    response_type="LlavaDecoderLayerResponse",
                    source_symbol=(
                        "Qwen2 decoder loop after the configured number of "
                        "completed layers and before the next layer"
                    ),
                    timing="once per prompt before prune_after_layer",
                    required_capabilities=frozenset(
                        {"decoder_loop_control", "sequence_layout_repair"}
                    ),
                    required_repairs=(
                        "compact mask and retain sparse source position ids",
                        "compact every completed shallow DynamicCache layer",
                        "rewrite image token and image_newline indices",
                        "preserve image_sizes and modalities",
                    ),
                ),
            ),
            metadata={
                'single_decoder_boundary': True,
                'generate_requires_kv_cache_repair': True,
                'cache_policy': 'compact_prefix',
                'position_policy': 'preserve_sparse_source',
                'single_image_only': True,
            },
        ),
        LlavaImageBindingContract(
            binding="d2pruner.image.llava_onevision",
            variant="d2pruner.image",
            callbacks=(
                LlavaCallbackContract(
                    site=FASTV_SITE,
                    request_type="DecoderSelectionRegistration",
                    response_type="LlavaDecoderLayerResponse",
                    source_symbol=(
                        "Qwen2 decoder loop after the configured scorer layer "
                        "and before the next layer consumes the compact sequence"
                    ),
                    timing="once per prompt before prune_after_layer",
                    required_capabilities=frozenset(
                        {"decoder_loop_control", "sequence_layout_repair"}
                    ),
                    required_repairs=(
                        "compact mask and retain sparse source position ids",
                        "compact every completed shallow DynamicCache layer",
                        "rewrite image token and image_newline indices",
                        "preserve image_sizes and modalities",
                    ),
                ),
            ),
            metadata={
                'single_decoder_boundary': True,
                'generate_requires_kv_cache_repair': True,
                'cache_policy': 'compact_prefix',
                'position_policy': 'preserve_sparse_source',
                'requires_calibrated_bias_prior': True,
                'single_image_only': True,
            },
        ),
        LlavaImageBindingContract(
            binding="sparsevlm.image.llava_onevision",
            variant="sparsevlm.image",
            callbacks=(
                LlavaCallbackContract(
                    site=PYRAMID_DROP_SITE,
                    request_type="DecoderSelectionRegistration",
                    response_type="LlavaDecoderLayerResponse",
                    source_symbol=(
                        "Qwen2 decoder loop after each configured completed "
                        "layer and before the next layer consumes the merged layout"
                    ),
                    timing="after configured layers 2, 6, and 15",
                    required_capabilities=frozenset(
                        {"decoder_loop_control", "sequence_layout_repair"}
                    ),
                    required_repairs=(
                        "mean-merge every completed DynamicCache prefix layer",
                        "rebuild contiguous mask and position ids",
                        "rewrite image token and image_newline indices",
                        "preserve one image identity",
                    ),
                ),
            ),
            metadata={
                'multistage_decoder_boundaries': True,
                'generate_requires_kv_cache_repair': True,
                'cache_policy': 'compact_prefix',
                'cache_transform': 'mean_merge',
                'position_policy': 'contiguous',
                'single_image_only': True,
            },
        ),
        LlavaImageBindingContract(
            binding="finepruner.image.llava_onevision",
            variant="finepruner.image",
            callbacks=(
                LlavaCallbackContract(
                    site=PYRAMID_DROP_SITE,
                    request_type="DecoderSelectionRegistration",
                    response_type="LlavaDecoderLayerResponse",
                    source_symbol=(
                        "Qwen2 decoder loop after completed-layer IAC/ARP "
                        "boundaries with last-text-query attention capture"
                    ),
                    timing=(
                        "IAC after completed layers 7/14; capture ARP at "
                        "20/21/22 and apply after 22"
                    ),
                    required_capabilities=frozenset(
                        {"decoder_loop_control", "sequence_layout_repair"}
                    ),
                    required_repairs=(
                        "preserve completed shallow DynamicCache layer lengths",
                        "rebuild contiguous mask and position ids",
                        "rewrite image token and image_newline indices",
                        "preserve one image identity",
                    ),
                ),
            ),
            metadata={
                'multistage_decoder_boundaries': True,
                'generate_requires_kv_cache_repair': True,
                'cache_policy': 'layer_local',
                'cache_transform': 'none',
                'position_policy': 'contiguous',
                'single_image_only': True,
                'attention_capture': 'last_text_query_per_head',
            },
        ),
        LlavaImageBindingContract(
            binding="btp.image.llava_onevision",
            variant="btp.image",
            callbacks=(
                LlavaCallbackContract(
                    site=PYRAMID_DROP_SITE,
                    request_type="DecoderSelectionRegistration",
                    response_type="LlavaDecoderLayerResponse",
                    source_symbol=(
                        "Qwen2 decoder loop after each released scorer layer "
                        "and before the next layer consumes the compact layout"
                    ),
                    timing=(
                        "after zero-based layers 2/5/14/21, equivalent to "
                        "released prune-before layers 4/7/16/23"
                    ),
                    required_capabilities=frozenset(
                        {"decoder_loop_control", "sequence_layout_repair"}
                    ),
                    required_repairs=(
                        "compact every completed DynamicCache prefix layer",
                        "retain sparse source position ids",
                        "rewrite image token and surviving image_newline indices",
                        "allow the final stage to remove the complete visual span",
                    ),
                ),
            ),
            metadata={
                'multistage_decoder_boundaries': True,
                'generate_requires_kv_cache_repair': True,
                'cache_policy': 'compact_prefix',
                'position_policy': 'preserve_original',
                'single_image_only': True,
                'attention_capture': 'last_text_query_head_mean',
                'final_stage': 'remove_remaining_visual_span',
            },
        ),
        LlavaImageBindingContract(
            binding="halfv.image.llava_onevision",
            variant="halfv.image",
            callbacks=(
                LlavaCallbackContract(
                    site=PYRAMID_DROP_SITE,
                    request_type="DecoderSelectionRegistration",
                    response_type="LlavaDecoderLayerResponse",
                    source_symbol=(
                        "Qwen2 decoder loop after HalfV scorer layers and "
                        "before layers 2/21 consume the compact layout"
                    ),
                    timing="after zero-based completed layers 1 and 20",
                    required_capabilities=frozenset(
                        {"decoder_loop_control", "sequence_layout_repair"}
                    ),
                    required_repairs=(
                        "compact every completed DynamicCache prefix layer",
                        "retain sparse source position ids",
                        "rewrite image token and surviving image_newline indices",
                        "preserve one image identity",
                    ),
                ),
            ),
            metadata={
                'multistage_decoder_boundaries': True,
                'generate_requires_kv_cache_repair': True,
                'cache_policy': 'compact_prefix',
                'cache_transform': 'drop',
                'position_policy': 'preserve_original',
                'single_image_only': True,
                'attention_capture': 'last_text_query_head_mean',
                'score_rope_by_stage': {'ivr': False, 'ssr': True},
                'synchronize_remaining_deepstack': False,
                'paper_architecture_alignment': 'partial_missing_static_visual_compute',
            },
        ),
        LlavaImageBindingContract(
            binding="transprune.image.llava_onevision",
            variant="transprune.image",
            callbacks=(
                LlavaCallbackContract(
                    site=PYRAMID_DROP_SITE,
                    request_type="DecoderSelectionRegistration",
                    response_type="LlavaDecoderLayerResponse",
                    source_symbol=(
                        "native Qwen2 decoder loop with temporary attention/FFN "
                        "capture and next-layer IGA projection"
                    ),
                    timing="after zero-based completed layers 6, 8, and 11",
                    required_capabilities=frozenset(
                        {"decoder_loop_control", "sequence_layout_repair"}
                    ),
                    required_repairs=(
                        "preserve each layer-local DynamicCache prefix",
                        "rebuild contiguous text positions after every stage",
                        "rewrite image token and surviving image_newline indices",
                        "preserve one image identity",
                    ),
                ),
            ),
            metadata={
                'multistage_decoder_boundaries': True,
                'generate_requires_kv_cache_repair': True,
                'cache_policy': 'layer_local',
                'cache_transform': 'none',
                'position_policy': 'contiguous',
                'single_image_only': True,
                'module_capture': ('attention', 'ffn'),
                'iga_source': 'next_layer_post_image_text',
                'synchronize_remaining_deepstack': False,
            },
        ),
    )
})
