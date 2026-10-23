"""Physical image-compression contract for the Qwen3.5 projector boundary."""

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Mapping


QWEN3_5_TOME_BINDING = "tome.image.qwen3_5"
QWEN3_5_TOME_VARIANT = "tome.image.projector_feature"
QWEN3_5_DIVPRUNE_BINDING = "divprune.image.qwen3_5"
QWEN3_5_DIVPRUNE_VARIANT = "divprune.image.projector_feature"
QWEN3_5_VISIONZIP_BINDING = "visionzip.image.qwen3_5"
QWEN3_5_VISIONZIP_VARIANT = "visionzip.image.projector_feature"
QWEN3_5_OTPRUNE_BINDING = "otprune.image.qwen3_5"
QWEN3_5_OTPRUNE_VARIANT = "otprune.image.projector_feature"
QWEN3_5_PRUMERGE_BINDING = "prumerge.image.qwen3_5"
QWEN3_5_PRUMERGE_VARIANT = "prumerge.image"
QWEN3_5_MMTOK_BINDING = "mmtok.image.qwen3_5"
QWEN3_5_MMTOK_VARIANT = "mmtok.image"
QWEN3_5_GAP_BINDING = "gap.image.qwen3_5"
QWEN3_5_GAP_VARIANT = "gap.image"
QWEN3_5_VISPRUNER_BINDING = "vispruner.image.qwen3_5"
QWEN3_5_VISPRUNER_VARIANT = "vispruner.image"
QWEN3_5_AIM_BINDING = "aim.image.qwen3_5"
QWEN3_5_AIM_VARIANT = "aim.image"
QWEN3_5_RESTORE_BINDING = "restore.image.qwen3_5"
QWEN3_5_RESTORE_VARIANT = "restore.image"
QWEN3_5_BIFOCAL_BINDING = "bifocal.image.qwen3_5"
QWEN3_5_BIFOCAL_VARIANT = "bifocal.image"
QWEN3_5_ZOO_PRUNE_BINDING = "zoo_prune.image.qwen3_5"
QWEN3_5_ZOO_PRUNE_VARIANT = "zoo_prune.image"
QWEN3_5_SAINT_BINDING = "saint.image.qwen3_5"
QWEN3_5_SAINT_VARIANT = "saint.image.projector_feature"
QWEN3_5_NUWA_BINDING = "nuwa.image.qwen3_5"
QWEN3_5_NUWA_VARIANT = "nuwa.image.qwen3_5"
QWEN3_5_SPARSEVLM_BINDING = "sparsevlm.image.qwen3_5"
QWEN3_5_SPARSEVLM_VARIANT = "sparsevlm.image.qwen3_5"
QWEN3_5_FINEPRUNER_BINDING = "finepruner.image.qwen3_5"
QWEN3_5_FINEPRUNER_VARIANT = "finepruner.image.qwen3_5"
QWEN3_5_TRANSPRUNE_BINDING = "transprune.image.qwen3_5"
QWEN3_5_TRANSPRUNE_VARIANT = "transprune.image.qwen3_5"
QWEN3_5_FASTV_BINDING = "fastv.image.qwen3_5"
QWEN3_5_PYRAMID_DROP_BINDING = "pyramid_drop.image.qwen3_5"
QWEN3_5_DART_BINDING = "dart.image.qwen3_5"
QWEN3_5_IVC_PRUNE_BINDING = "ivc_prune.image.qwen3_5"
QWEN3_5_FEATHER_BINDING = "feather.image.qwen3_5"
QWEN3_5_BTP_BINDING = "btp.image.qwen3_5"
QWEN3_5_HALFV_BINDING = "halfv.image.qwen3_5"
QWEN3_5_SGL_BINDING = "sgl.image.qwen3_5"
QWEN3_5_D2PRUNER_BINDING = "d2pruner.image.qwen3_5"


@dataclass(frozen=True)
class Qwen3_5ImageBindingContract:
    binding: str
    algorithm: str
    variant: str
    callback: str
    target_sites: tuple[str, ...]
    repairs: tuple[str, ...]
    required_handles: tuple[str, ...]
    image_only: bool = True
    supports_video: bool = False
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.supports_video:
            raise ValueError("The initial Qwen3.5 image binding is image-only.")
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))


QWEN3_5_IMAGE_BINDING_CONTRACTS = MappingProxyType(
    {
        QWEN3_5_TOME_BINDING: Qwen3_5ImageBindingContract(
            binding=QWEN3_5_TOME_BINDING,
            algorithm="tome",
            variant=QWEN3_5_TOME_VARIANT,
            callback="post_merger_pre_llm_scatter",
            target_sites=("post_merger_pre_llm_scatter",),
            repairs=(
                "image_placeholder_compaction",
                "attention_mask_compaction",
                "mm_token_type_compaction",
                "sparse_mrope_positions_and_rope_delta",
            ),
            required_handles=("image_features_pre_llm_scatter", "llm"),
            metadata={
                "position": "post_vision_merger_pre_llm",
                "deepstack": False,
                "cache_policy": "compact_prompt_native_generation_cache",
            },
        ),
        QWEN3_5_DIVPRUNE_BINDING: Qwen3_5ImageBindingContract(
            binding=QWEN3_5_DIVPRUNE_BINDING,
            algorithm="divprune",
            variant=QWEN3_5_DIVPRUNE_VARIANT,
            callback="post_merger_pre_llm_scatter",
            target_sites=("post_merger_pre_llm_scatter",),
            repairs=(
                "image_placeholder_compaction",
                "attention_mask_compaction",
                "mm_token_type_compaction",
                "sparse_mrope_positions_and_rope_delta",
            ),
            required_handles=("image_features_pre_llm_scatter", "llm"),
            metadata={
                "position": "post_vision_merger_pre_llm",
                "deepstack": False,
                "cache_policy": "compact_prompt_native_generation_cache",
                "selection": "greedy_maxmin_cosine",
            },
        ),
        QWEN3_5_VISIONZIP_BINDING: Qwen3_5ImageBindingContract(
            binding=QWEN3_5_VISIONZIP_BINDING,
            algorithm="visionzip",
            variant=QWEN3_5_VISIONZIP_VARIANT,
            callback="post_merger_pre_llm_scatter",
            target_sites=("post_merger_pre_llm_scatter",),
            repairs=(
                "image_placeholder_compaction",
                "attention_mask_compaction",
                "mm_token_type_compaction",
                "sparse_mrope_positions_and_rope_delta",
            ),
            required_handles=(
                "image_features_pre_llm_scatter",
                "vision_encoder",
                "llm",
            ),
            metadata={
                "position": "post_vision_merger_pre_llm",
                "deepstack": False,
                "cache_policy": "compact_prompt_native_generation_cache",
                "attention_mode": "mean_received_no_cls",
                "patch_statistics_mapping": "contiguous_spatial_merger_groups",
                "key_position_encoding": "qwen3_5_rotary",
            },
        ),
        QWEN3_5_OTPRUNE_BINDING: Qwen3_5ImageBindingContract(
            binding=QWEN3_5_OTPRUNE_BINDING,
            algorithm="otprune",
            variant=QWEN3_5_OTPRUNE_VARIANT,
            callback="post_merger_pre_llm_scatter",
            target_sites=("post_merger_pre_llm_scatter",),
            repairs=(
                "image_placeholder_compaction",
                "attention_mask_compaction",
                "mm_token_type_compaction",
                "sparse_mrope_positions_and_rope_delta",
            ),
            required_handles=("image_features_pre_llm_scatter", "llm"),
            metadata={
                "position": "post_vision_merger_pre_llm",
                "deepstack": False,
                "cache_policy": "compact_prompt_native_generation_cache",
                "objective": "ot_distribution_alignment_greedy_logdet",
            },
        ),
        QWEN3_5_PRUMERGE_BINDING: Qwen3_5ImageBindingContract(
            binding=QWEN3_5_PRUMERGE_BINDING,
            algorithm="prumerge",
            variant=QWEN3_5_PRUMERGE_VARIANT,
            callback="post_merger_pre_llm_scatter",
            target_sites=("post_merger_pre_llm_scatter",),
            repairs=(
                "image_placeholder_compaction",
                "attention_mask_compaction",
                "mm_token_type_compaction",
                "sparse_mrope_positions_and_rope_delta",
            ),
            required_handles=(
                "image_features_pre_llm_scatter",
                "vision_encoder",
                "llm",
            ),
            metadata={
                "position": "post_vision_merger_pre_llm",
                "deepstack": False,
                "cache_policy": "compact_prompt_native_generation_cache",
                "variant": "plus",
                "attention_mode": "mean_received_no_cls",
                "key_position_encoding": "qwen3_5_rotary",
                "output_budget": "fixed_total_including_global_residual",
            },
        ),
        QWEN3_5_MMTOK_BINDING: Qwen3_5ImageBindingContract(
            binding=QWEN3_5_MMTOK_BINDING,
            algorithm="mmtok",
            variant=QWEN3_5_MMTOK_VARIANT,
            callback="post_merger_pre_llm_scatter",
            target_sites=("post_merger_pre_llm_scatter",),
            repairs=(
                "image_placeholder_compaction",
                "attention_mask_compaction",
                "mm_token_type_compaction",
                "sparse_mrope_positions_and_rope_delta",
            ),
            required_handles=(
                "image_features_pre_llm_scatter",
                "vision_encoder",
                "llm",
            ),
            metadata={
                "position": "post_vision_merger_pre_llm",
                "deepstack": False,
                "cache_policy": "compact_prompt_native_generation_cache",
                "conditioning": "prompt_input_embeddings",
                "vision_source": "mean_pooled_pre_merger_patch_features",
            },
        ),
        QWEN3_5_GAP_BINDING: Qwen3_5ImageBindingContract(
            binding=QWEN3_5_GAP_BINDING,
            algorithm="gap",
            variant=QWEN3_5_GAP_VARIANT,
            callback="post_merger_pre_llm_scatter",
            target_sites=("post_merger_pre_llm_scatter",),
            repairs=(
                "image_placeholder_compaction",
                "attention_mask_compaction",
                "mm_token_type_compaction",
                "sparse_mrope_positions_and_rope_delta",
            ),
            required_handles=(
                "image_features_pre_llm_scatter",
                "vision_encoder",
                "llm",
            ),
            metadata={
                "position": "post_vision_merger_pre_llm",
                "deepstack": False,
                "cache_policy": "compact_prompt_native_generation_cache",
                "attention_mode": "mean_received_no_cls",
                "selection": "received_attention_topk",
                "position_policy": "preserve_original",
            },
        ),
        QWEN3_5_VISPRUNER_BINDING: Qwen3_5ImageBindingContract(
            binding=QWEN3_5_VISPRUNER_BINDING,
            algorithm="vispruner",
            variant=QWEN3_5_VISPRUNER_VARIANT,
            callback="post_merger_pre_llm_scatter",
            target_sites=("post_merger_pre_llm_scatter",),
            repairs=(
                "image_placeholder_compaction",
                "attention_mask_compaction",
                "mm_token_type_compaction",
                "sparse_mrope_positions_and_rope_delta",
            ),
            required_handles=(
                "image_features_pre_llm_scatter",
                "vision_encoder",
                "llm",
            ),
            metadata={
                "position": "post_vision_merger_pre_llm",
                "deepstack": False,
                "cache_policy": "compact_prompt_native_generation_cache",
                "attention_mode": "mean_received_no_cls",
                "visual_reference": "mean_pooled_pre_merger_patch_features",
                "selection": "important_attention_plus_bipartite_diversity",
            },
        ),
        QWEN3_5_AIM_BINDING: Qwen3_5ImageBindingContract(
            binding=QWEN3_5_AIM_BINDING,
            algorithm="aim",
            variant=QWEN3_5_AIM_VARIANT,
            callback="post_merger_pre_llm_scatter",
            target_sites=("post_merger_pre_llm_scatter",),
            repairs=(
                "image_placeholder_compaction",
                "attention_mask_compaction",
                "mm_token_type_compaction",
                "sparse_mrope_positions_and_rope_delta",
            ),
            required_handles=("image_features_pre_llm_scatter", "llm"),
            metadata={
                "position": "post_vision_merger_pre_llm",
                "deepstack": False,
                "cache_policy": "compact_prompt_native_generation_cache",
                "operation": "bipartite_feature_merge",
                "paper_stage": "projector_merge_adaptation",
            },
        ),
        QWEN3_5_RESTORE_BINDING: Qwen3_5ImageBindingContract(
            binding=QWEN3_5_RESTORE_BINDING,
            algorithm="restore",
            variant=QWEN3_5_RESTORE_VARIANT,
            callback="post_merger_pre_llm_scatter",
            target_sites=("post_merger_pre_llm_scatter",),
            repairs=(
                "image_placeholder_compaction",
                "attention_mask_compaction",
                "mm_token_type_compaction",
                "sparse_mrope_positions_and_rope_delta",
            ),
            required_handles=(
                "image_features_pre_llm_scatter",
                "vision_encoder",
                "llm",
            ),
            metadata={
                "position": "post_vision_merger_pre_llm",
                "deepstack": False,
                "cache_policy": "compact_prompt_native_generation_cache",
                "selection": "vispruner_pristine_plus_anchor_merge",
                "position_strategy": "median",
            },
        ),
        QWEN3_5_BIFOCAL_BINDING: Qwen3_5ImageBindingContract(
            binding=QWEN3_5_BIFOCAL_BINDING,
            algorithm="bifocal",
            variant=QWEN3_5_BIFOCAL_VARIANT,
            callback="post_merger_pre_llm_scatter",
            target_sites=("post_merger_pre_llm_scatter",),
            repairs=(
                "image_placeholder_compaction",
                "attention_mask_compaction",
                "mm_token_type_compaction",
                "sparse_mrope_positions_and_rope_delta",
            ),
            required_handles=("image_features_pre_llm_scatter", "vision_encoder", "llm"),
            metadata={
                "position": "post_vision_merger_pre_llm",
                "deepstack": False,
                "cache_policy": "compact_prompt_native_generation_cache",
                "attention_mode": "mean_received_no_cls",
                "paper_stage": "projector_level_framework_adaptation",
            },
        ),
        QWEN3_5_ZOO_PRUNE_BINDING: Qwen3_5ImageBindingContract(
            binding=QWEN3_5_ZOO_PRUNE_BINDING,
            algorithm="zoo_prune",
            variant=QWEN3_5_ZOO_PRUNE_VARIANT,
            callback="post_merger_pre_llm_scatter",
            target_sites=("post_merger_pre_llm_scatter",),
            repairs=("image_placeholder_compaction", "attention_mask_compaction", "mm_token_type_compaction", "sparse_mrope_positions_and_rope_delta"),
            required_handles=("image_features_pre_llm_scatter", "vision_encoder.merger", "llm"),
            metadata={
                "position": "post_vision_merger_pre_llm",
                "deepstack": False,
                "cache_policy": "compact_prompt_native_generation_cache",
                "sensitivity_source": "qwen3_5_native_patch_merger_proxy",
                "paper_stage": "projector_level_framework_adaptation",
            },
        ),
        QWEN3_5_SAINT_BINDING: Qwen3_5ImageBindingContract(
            binding=QWEN3_5_SAINT_BINDING,
            algorithm="saint",
            variant=QWEN3_5_SAINT_VARIANT,
            callback="post_merger_pre_llm_scatter",
            target_sites=("post_merger_pre_llm_scatter",),
            repairs=("image_placeholder_compaction", "attention_mask_compaction",
                     "mm_token_type_compaction", "sparse_mrope_positions_and_rope_delta"),
            required_handles=("image_features_pre_llm_scatter", "vision_encoder", "llm"),
            metadata={
                "position": "post_vision_merger_pre_llm",
                "deepstack": False,
                "cache_policy": "compact_prompt_native_generation_cache",
                "key_source": "native_final_vision_rotary_keys",
                "paper_stage": "projector_level_framework_adaptation",
            },
        ),
        QWEN3_5_NUWA_BINDING: Qwen3_5ImageBindingContract(
            binding=QWEN3_5_NUWA_BINDING,
            algorithm="nuwa",
            variant=QWEN3_5_NUWA_VARIANT,
            callback="composite_post_merger_and_decoder_stage2",
            target_sites=("post_merger_pre_llm_scatter", "decoder_prune_boundary"),
            repairs=(
                "stage1_weighted_visual_aggregation",
                "image_placeholder_compaction",
                "attention_mask_compaction",
                "mm_token_type_compaction",
                "sparse_mrope_positions_and_rope_delta",
                "stage2_decoder_sequence_compaction",
            ),
            required_handles=("vision_encoder", "image_features_pre_llm_scatter",
                              "language_model.layers", "llm"),
            metadata={
                "position": "post_merger_and_decoder_stage2",
                "vision_attention_source": "mean_received_and_native_rotary_keys",
                "semantic_hidden_range": "configured_native_vision_blocks",
                "stage2_score_layer_must_be_full_attention": True,
                "cache_policy": "no_cache_full_prefix_recompute",
                "paper_stage": "two_stage_framework_adaptation",
            },
        ),
        QWEN3_5_SPARSEVLM_BINDING: Qwen3_5ImageBindingContract(
            binding=QWEN3_5_SPARSEVLM_BINDING,
            algorithm="sparsevlm",
            variant=QWEN3_5_SPARSEVLM_VARIANT,
            callback="decoder_sparsevlm_stage_boundaries",
            target_sites=("decoder_prune_boundary",) * 3,
            repairs=("decoder_hidden_state_compaction", "attention_mask_compaction",
                     "mrope_position_embedding_compaction", "text_and_generated_token_preservation"),
            required_handles=("language_model.layers", "full_attention_qk_stages"),
            metadata={
                "position": "three_decoder_stage_boundaries",
                "score_layers_must_be_full_attention": True,
                "schedule_profile": "qwen3_8_full_attention_aligned",
                "cache_policy": "no_cache_recompute_only",
                "video_support": False,
            },
        ),
        QWEN3_5_FINEPRUNER_BINDING: Qwen3_5ImageBindingContract(
            binding=QWEN3_5_FINEPRUNER_BINDING,
            algorithm="finepruner",
            variant=QWEN3_5_FINEPRUNER_VARIANT,
            callback="decoder_iac_then_arp_boundaries",
            target_sites=("decoder_prune_boundary",) * 3,
            repairs=("decoder_hidden_state_compaction", "attention_mask_compaction",
                     "mrope_position_embedding_compaction", "text_and_generated_token_preservation"),
            required_handles=("language_model.layers", "arp_full_attention_qk"),
            metadata={
                "position": "two_iac_boundaries_then_arp_boundary",
                "arp_score_layers_must_be_full_attention": True,
                "schedule_profile": "qwen3_8_full_attention_aligned",
                "cache_policy": "no_cache_recompute_only",
                "video_support": False,
            },
        ),
        QWEN3_5_TRANSPRUNE_BINDING: Qwen3_5ImageBindingContract(
            binding=QWEN3_5_TRANSPRUNE_BINDING,
            algorithm="transprune",
            variant=QWEN3_5_TRANSPRUNE_VARIANT,
            callback="decoder_ttv_iga_full_attention_boundaries",
            target_sites=("decoder_prune_boundary",) * 3,
            repairs=("decoder_hidden_state_compaction", "attention_mask_compaction",
                     "mrope_position_embedding_compaction", "text_and_generated_token_preservation"),
            required_handles=("language_model.layers", "llm"),
            metadata={
                "position": "three_decoder_stage_boundaries",
                "iga_score_layers_must_be_full_attention": True,
                "schedule_profile": "qwen3_8_full_attention_aligned",
                "cache_policy": "no_cache_recompute_only",
                "video_support": False,
            },
        ),
        QWEN3_5_FASTV_BINDING: Qwen3_5ImageBindingContract(
            binding=QWEN3_5_FASTV_BINDING,
            algorithm="fastv",
            variant="fastv.image",
            callback="decoder_prune_boundary",
            target_sites=("decoder_prune_boundary",),
            repairs=(
                "decoder_hidden_state_compaction",
                "causal_mask_query_key_compaction",
                "mrope_position_embedding_compaction",
                "text_and_generated_token_preservation",
            ),
            required_handles=("language_model.layers", "full_attention_qk", "input_ids"),
            metadata={
                "position": "decoder_layer_boundary",
                "score_layer_must_be_full_attention": True,
                "cache_policy": "no_cache_recompute_only",
                "video_support": False,
            },
        ),
        QWEN3_5_PYRAMID_DROP_BINDING: Qwen3_5ImageBindingContract(
            binding=QWEN3_5_PYRAMID_DROP_BINDING,
            algorithm="pyramid_drop",
            variant="pyramid_drop.image",
            callback="decoder_stage_boundaries",
            target_sites=("decoder_stage_boundaries",),
            repairs=(
                "decoder_hidden_state_compaction",
                "causal_mask_query_key_compaction",
                "mrope_position_embedding_compaction",
                "text_and_generated_token_preservation",
            ),
            required_handles=("language_model.layers", "pyramid_full_attention_qk", "input_ids"),
            metadata={
                "position": "multiple_decoder_layer_boundaries",
                "score_layers_must_be_full_attention": True,
                "cache_policy": "no_cache_recompute_only",
                "video_support": False,
            },
        ),
        QWEN3_5_DART_BINDING: Qwen3_5ImageBindingContract(
            binding=QWEN3_5_DART_BINDING,
            algorithm="dart",
            variant="dart.image",
            callback="decoder_prune_boundary",
            target_sites=("decoder_prune_boundary",),
            repairs=(
                "decoder_hidden_state_compaction",
                "causal_mask_query_key_compaction",
                "mrope_position_embedding_compaction",
                "text_and_generated_token_preservation",
            ),
            required_handles=("language_model.layers", "full_attention_keys", "language_model.norm", "input_ids"),
            metadata={
                "position": "decoder_layer_boundary",
                "score_layer_must_be_full_attention": True,
                "cache_policy": "no_cache_recompute_only",
                "video_support": False,
            },
        ),
        QWEN3_5_IVC_PRUNE_BINDING: Qwen3_5ImageBindingContract(
            binding=QWEN3_5_IVC_PRUNE_BINDING,
            algorithm="ivc_prune",
            variant="ivc_prune.image",
            callback="decoder_prune_boundary",
            target_sites=("decoder_prune_boundary",),
            repairs=(
                "decoder_hidden_state_compaction",
                "causal_mask_query_key_compaction",
                "mrope_position_embedding_compaction",
                "text_and_generated_token_preservation",
            ),
            required_handles=("language_model.layers", "native_mrope_cos_sin", "full_attention_value_projection"),
            metadata={
                "position": "decoder_layer_boundary",
                "score_layer_must_be_full_attention": True,
                "cache_policy": "no_cache_recompute_only",
                "video_support": False,
            },
        ),
        QWEN3_5_FEATHER_BINDING: Qwen3_5ImageBindingContract(
            binding=QWEN3_5_FEATHER_BINDING,
            algorithm="feather",
            variant="feather.image",
            callback="decoder_stage_boundaries",
            target_sites=("decoder_stage_boundaries",),
            repairs=(
                "decoder_hidden_state_compaction",
                "causal_mask_query_key_compaction",
                "mrope_position_embedding_compaction",
                "text_and_generated_token_preservation",
            ),
            required_handles=("language_model.layers", "no_rope_attention_scores", "input_ids"),
            metadata={
                "position": "two_decoder_layer_boundaries",
                "score_layers_must_be_full_attention": True,
                "cache_policy": "no_cache_recompute_only",
                "video_support": False,
            },
        ),
        QWEN3_5_BTP_BINDING: Qwen3_5ImageBindingContract(
            binding=QWEN3_5_BTP_BINDING,
            algorithm="btp",
            variant="btp.image",
            callback="decoder_multistage_boundaries",
            target_sites=("decoder_multistage_boundaries",),
            repairs=("decoder_hidden_state_compaction", "causal_mask_query_key_compaction", "mrope_position_embedding_compaction", "text_and_generated_token_preservation"),
            required_handles=("language_model.layers", "full_attention_qk_stages", "input_ids"),
            metadata={"position": "four_decoder_stage_boundaries", "score_layers_must_be_full_attention": True, "cache_policy": "no_cache_recompute_only", "video_support": False},
        ),
        QWEN3_5_HALFV_BINDING: Qwen3_5ImageBindingContract(
            binding=QWEN3_5_HALFV_BINDING,
            algorithm="halfv",
            variant="halfv.image",
            callback="decoder_ivr_ssr_boundaries",
            target_sites=("decoder_ivr_ssr_boundaries",),
            repairs=("decoder_hidden_state_compaction", "causal_mask_query_key_compaction", "mrope_position_embedding_compaction", "text_and_generated_token_preservation"),
            required_handles=("language_model.layers", "ivr_ssr_attention_qk", "language_model.norm", "input_ids"),
            metadata={"position": "two_decoder_stage_boundaries", "score_layers_must_be_full_attention": True, "cache_policy": "no_cache_recompute_only", "video_support": False},
        ),
        QWEN3_5_SGL_BINDING: Qwen3_5ImageBindingContract(
            binding=QWEN3_5_SGL_BINDING,
            algorithm="sgl",
            variant="sgl.image",
            callback="decoder_guide_prune_boundary",
            target_sites=("decoder_guide_prune_boundary",),
            repairs=("decoder_hidden_state_compaction", "causal_mask_query_key_compaction", "mrope_position_embedding_compaction", "text_and_generated_token_preservation"),
            required_handles=("language_model.layers", "input_ids"),
            metadata={"guide_attention_source": "aligned_qwen3_vl_runtime", "cache_policy": "no_cache_recompute_only", "single_image_only": True, "video_support": False},
        ),
        QWEN3_5_D2PRUNER_BINDING: Qwen3_5ImageBindingContract(
            binding=QWEN3_5_D2PRUNER_BINDING,
            algorithm="d2pruner",
            variant="d2pruner.image",
            callback="decoder_calibrated_spatial_prune_boundary",
            target_sites=("decoder_calibrated_spatial_prune_boundary",),
            repairs=("decoder_hidden_state_compaction", "causal_mask_query_key_compaction", "mrope_position_embedding_compaction", "text_and_generated_token_preservation"),
            required_handles=("language_model.layers", "full_attention_qk", "input_ids"),
            metadata={"calibrated_prior_required": True, "prior_must_match_checkpoint_and_scorer_layer": True, "cache_policy": "no_cache_recompute_only", "video_support": False},
        ),
    }
)
