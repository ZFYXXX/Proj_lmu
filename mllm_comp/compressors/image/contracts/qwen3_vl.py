"""Physical qwen3_vl image callbacks, sites and layout requirements."""

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Mapping


QWEN3_VL_TOME_BINDING = "tome.image.qwen3_vl"
QWEN3_VL_VISIONZIP_BINDING = "visionzip.image.qwen3_vl"
QWEN3_VL_PYRAMID_DROP_BINDING = "pyramid_drop.image.qwen3_vl"
QWEN3_VL_DIVPRUNE_BINDING = "divprune.image.qwen3_vl"
QWEN3_VL_SAINT_BINDING = "saint.image.qwen3_vl"
QWEN3_VL_AIM_BINDING = "aim.image.qwen3_vl"
QWEN3_VL_PACT_BINDING = "pact.image.qwen3_vl"
QWEN3_VL_MMTOK_BINDING = "mmtok.image.qwen3_vl"
QWEN3_VL_VISPRUNER_BINDING = "vispruner.image.qwen3_vl"
QWEN3_VL_ZOO_PRUNE_BINDING = "zoo_prune.image.qwen3_vl"
QWEN3_VL_NUWA_BINDING = "nuwa.image.qwen3_vl"
QWEN3_VL_RESTORE_BINDING = "restore.image.qwen3_vl"
QWEN3_VL_FASTV_BINDING = "fastv.image.qwen3_vl"
QWEN3_VL_SGL_BINDING = "sgl.image.qwen3_vl"
QWEN3_VL_DART_BINDING = "dart.image.qwen3_vl"
QWEN3_VL_D2PRUNER_BINDING = "d2pruner.image.qwen3_vl"
QWEN3_VL_SPARSEVLM_BINDING = "sparsevlm.image.qwen3_vl"
QWEN3_VL_FINEPRUNER_BINDING = "finepruner.image.qwen3_vl"
QWEN3_VL_BTP_BINDING = "btp.image.qwen3_vl"
QWEN3_VL_HALFV_BINDING = "halfv.image.qwen3_vl"
QWEN3_VL_TRANSPRUNE_BINDING = "transprune.image.qwen3_vl"
QWEN3_VL_PRUMERGE_BINDING = "prumerge.image.qwen3_vl"
QWEN3_VL_GAP_BINDING = "gap.image.qwen3_vl"
QWEN3_VL_FEATHER_BINDING = "feather.image.qwen3_vl"
QWEN3_VL_IVC_PRUNE_BINDING = "ivc_prune.image.qwen3_vl"

_TOME_VARIANT = "tome.image.encoder_dense_boundary"
_VISIONZIP_VARIANT = "visionzip.image"
_PYRAMID_DROP_VARIANT = "pyramid_drop.image"
_DIVPRUNE_VARIANT = "divprune.image"
_SAINT_VARIANT = "saint.image.packed_feature_prune"
_AIM_VARIANT = "aim.image"
_PACT_VARIANT = "pact.image"
_MMTOK_VARIANT = "mmtok.image"
_VISPRUNER_VARIANT = "vispruner.image"
_ZOO_PRUNE_VARIANT = "zoo_prune.image"
_NUWA_VARIANT = "nuwa.image"
_RESTORE_VARIANT = "restore.image"
_FASTV_VARIANT = "fastv.image"
_SGL_VARIANT = "sgl.image"
_DART_VARIANT = "dart.image"
_D2PRUNER_VARIANT = "d2pruner.image"
_SPARSEVLM_VARIANT = "sparsevlm.image"
_FINEPRUNER_VARIANT = "finepruner.image"
_BTP_VARIANT = "btp.image"
_HALFV_VARIANT = "halfv.image"
_TRANSPRUNE_VARIANT = "transprune.image"
_PRUMERGE_VARIANT = "prumerge.image"
_GAP_VARIANT = "gap.image"
_FEATHER_VARIANT = "feather.image"
_IVC_PRUNE_VARIANT = "ivc_prune.image"
_CONTRACT_VERSION = "qwen3-vl-image-compression-contract-v1"


@dataclass(frozen=True)
class Qwen3VLImageBindingContract:
    """Machine-checkable physical contract for one Qwen3-VL image binding."""

    binding: str
    algorithm: str
    variant: str
    callback: str
    target_sites: tuple[str, ...]
    repairs: tuple[str, ...]
    required_handles: tuple[str, ...]
    image_only: bool = True
    supports_video: bool = False
    requires_visual_metrics: bool = False
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name in ("binding", "algorithm", "variant", "callback"):
            if not isinstance(getattr(self, name), str) or not getattr(self, name):
                raise ValueError(f"{name} must be a non-empty string.")
        if not self.target_sites:
            raise ValueError("target_sites must not be empty.")
        if self.supports_video:
            raise ValueError(
                "Qwen3-VL image binding contracts cannot implicitly support video."
            )
        if not isinstance(self.requires_visual_metrics, bool):
            raise TypeError("requires_visual_metrics must be a bool.")
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))


QWEN3_VL_IMAGE_BINDING_CONTRACTS: Mapping[str, Qwen3VLImageBindingContract] = MappingProxyType({
    contract.binding: contract
    for contract in (
        Qwen3VLImageBindingContract(
            binding=QWEN3_VL_TOME_BINDING,
            algorithm="tome",
            variant=_TOME_VARIANT,
            callback="encoder_dense_boundary",
            target_sites=(
                "visual_block_post_attention_residual_pre_mlp",
                "visual_pre_deepstack_mergers",
                "visual_output_pre_final_merger",
                "post_visual_mergers_pre_llm_scatter_tome",
            ),
            repairs=(
                "dense_restore_before_every_deepstack_merger",
                "dense_restore_before_final_merger",
                "vision_window_layout_preservation",
                "shared_main_deepstack_permanent_compaction",
                "prompt_mask_position_label_compaction",
            ),
            required_handles=(
                "vision_encoder.blocks",
                "vision_deepstack_projectors",
                "vision_projector",
            ),
            metadata={
                'compact_layout_scope': 'visual_encoder_and_permanent_llm_prompt',
                'structural_restore_is_temporary': True,
                'deepstack_synchronized': True,
                'permanent_llm_token_reduction': True,
            },
        ),
        Qwen3VLImageBindingContract(
            binding=QWEN3_VL_VISIONZIP_BINDING,
            algorithm="visionzip",
            variant=_VISIONZIP_VARIANT,
            callback="post_visual_merger_with_deepstack",
            target_sites=(
                "post_final_merger_pre_llm_scatter",
                "post_deepstack_mergers_pre_deepstack_injection",
            ),
            repairs=(
                "shared_main_deepstack_layout_delta",
                "image_placeholder_compaction",
                "sparse_mrope_rebuild",
            ),
            required_handles=(
                "vision_projector",
                "vision_deepstack_projectors",
                "llm",
            ),
            requires_visual_metrics=True,
            metadata={
                'shared_main_deepstack_delta': True,
            },
        ),
        Qwen3VLImageBindingContract(
            binding=QWEN3_VL_PRUMERGE_BINDING,
            algorithm="prumerge",
            variant=_PRUMERGE_VARIANT,
            callback="post_visual_merger_with_deepstack",
            target_sites=(
                "post_final_merger_pre_llm_scatter",
                "post_deepstack_mergers_pre_deepstack_injection",
            ),
            repairs=(
                "shared_main_deepstack_selection",
                "branch_local_prumerge_values",
                "image_placeholder_compaction",
                "sparse_mrope_rebuild",
            ),
            required_handles=("vision_projector", "vision_deepstack_projectors", "llm"),
            requires_visual_metrics=True,
            metadata={
                'shared_main_deepstack_delta': True,
                'position_policy': 'native_sparse_mrope_representatives',
            },
        ),
        Qwen3VLImageBindingContract(
            binding=QWEN3_VL_GAP_BINDING,
            algorithm="gap",
            variant=_GAP_VARIANT,
            callback="post_visual_merger_with_deepstack",
            target_sites=(
                "post_final_merger_pre_llm_scatter",
                "post_deepstack_mergers_pre_deepstack_injection",
            ),
            repairs=(
                "shared_main_deepstack_source_order_selection",
                "image_placeholder_compaction",
                "original_sparse_mrope_positions",
            ),
            required_handles=("vision_projector", "vision_deepstack_projectors", "llm"),
            requires_visual_metrics=True,
            metadata={
                'shared_main_deepstack_delta': True,
                'position_policy': 'preserve_original',
            },
        ),
        Qwen3VLImageBindingContract(
            binding=QWEN3_VL_PYRAMID_DROP_BINDING,
            algorithm="pyramid_drop",
            variant=_PYRAMID_DROP_VARIANT,
            callback="decoder_layer_loop_after_last_deepstack",
            target_sites=("llm_decoder_layer_output_image_tokens",),
            repairs=(
                "decoder_sequence_compaction",
                "attention_and_position_compaction",
                "generation_kv_cache_repair",
            ),
            required_handles=("llm",),
            metadata={
                'requires_decoder_loop_control': True,
                'generate_requires_kv_cache_repair': True,
            },
        ),
        Qwen3VLImageBindingContract(
            binding=QWEN3_VL_FEATHER_BINDING,
            algorithm="feather",
            variant=_FEATHER_VARIANT,
            callback="decoder_layer_loop_after_last_deepstack",
            target_sites=("llm_decoder_pre_feather_stage_image_tokens",),
            repairs=(
                "two_stage_decoder_sequence_compaction",
                "native_sparse_mrope_position_selection",
                "completed_prefix_dynamic_cache_compaction",
            ),
            required_handles=("llm",),
            metadata={
                'requires_decoder_loop_control': True,
                'cache_policy': 'compact_prefix',
                'position_policy': 'preserve_original_mrope',
                'single_image_only': True,
            },
        ),
        Qwen3VLImageBindingContract(
            binding=QWEN3_VL_IVC_PRUNE_BINDING,
            algorithm="ivc_prune",
            variant=_IVC_PRUNE_VARIANT,
            callback="decoder_layer_loop_after_last_deepstack",
            target_sites=("llm_decoder_pre_ivc_prune_layer_image_tokens",),
            repairs=(
                "native_mrope_coordinate_anchor_selection",
                "foreground_value_selection",
                "original_sparse_mrope_position_selection",
                "completed_prefix_dynamic_cache_compaction",
            ),
            required_handles=("llm",),
            metadata={
                'requires_decoder_loop_control': True,
                'cache_policy': 'compact_prefix',
                'coordinate_source': 'native_three_axis_mrope',
                'single_image_only': True,
            },
        ),
        Qwen3VLImageBindingContract(
            binding=QWEN3_VL_DIVPRUNE_BINDING,
            algorithm="divprune",
            variant=_DIVPRUNE_VARIANT,
            callback="post_visual_merger_with_deepstack",
            target_sites=(
                "post_final_merger_pre_llm_scatter",
                "post_deepstack_mergers_pre_deepstack_injection",
            ),
            repairs=(
                "shared_main_deepstack_layout_delta",
                "per_image_feature_selection",
                "image_placeholder_compaction",
                "sparse_mrope_rebuild",
            ),
            required_handles=(
                "vision_projector",
                "vision_deepstack_projectors",
                "llm",
            ),
            requires_visual_metrics=False,
            metadata={
                'shared_main_deepstack_delta': True,
                'layer_zero_pre_llm': True,
                'per_image_selection': True,
                'position_policy': 'sparse_mrope_rebuild',
            },
        ),
        Qwen3VLImageBindingContract(
            binding=QWEN3_VL_SAINT_BINDING,
            algorithm="saint",
            variant=_SAINT_VARIANT,
            callback="post_visual_merger_with_deepstack",
            target_sites=(
                "post_final_merger_pre_llm_scatter",
                "post_deepstack_mergers_pre_deepstack_injection",
            ),
            repairs=(
                "shared_main_deepstack_layout_delta",
                "per_image_feature_selection",
                "image_placeholder_compaction",
                "sparse_mrope_rebuild",
            ),
            required_handles=(
                "vision_projector",
                "vision_deepstack_projectors",
                "llm",
            ),
            requires_visual_metrics=True,
            metadata={
                'adaptation': 'post_encoder_feature_redundancy_prune',
                'shared_main_deepstack_delta': True,
            },
        ),
        Qwen3VLImageBindingContract(
            binding=QWEN3_VL_AIM_BINDING,
            algorithm="aim",
            variant=_AIM_VARIANT,
            callback="post_visual_merger_with_deepstack",
            target_sites=(
                "post_final_merger_pre_llm_scatter",
                "post_deepstack_mergers_pre_deepstack_injection",
            ),
            repairs=(
                "shared_main_deepstack_layout_delta",
                "differentiable_merge_replay",
                "image_placeholder_compaction",
                "sparse_mrope_rebuild",
            ),
            required_handles=(
                "vision_projector",
                "vision_deepstack_projectors",
                "llm",
            ),
            requires_visual_metrics=False,
            metadata={
                'shared_main_deepstack_delta': True,
                'differentiable_merge': True,
            },
        ),
        Qwen3VLImageBindingContract(
            binding=QWEN3_VL_PACT_BINDING,
            algorithm="pact",
            variant=_PACT_VARIANT,
            callback="decoder_layer_loop_after_last_deepstack",
            target_sites=("llm_decoder_layer_output_image_tokens",),
            repairs=(
                "decoder_sequence_compaction",
                "layer_local_generation_cache_repair",
                "log_cluster_mass_attention_bias",
            ),
            required_handles=("llm",),
            metadata={
                'requires_decoder_loop_control': True,
                'single_image_only': True,
                'cache_policy': 'layer_local',
            },
        ),
        Qwen3VLImageBindingContract(
            binding=QWEN3_VL_MMTOK_BINDING,
            algorithm="mmtok",
            variant=_MMTOK_VARIANT,
            callback="post_visual_merger_with_deepstack",
            target_sites=(
                "post_final_merger_pre_llm_scatter",
                "post_deepstack_mergers_pre_deepstack_injection",
            ),
            repairs=(
                "shared_main_deepstack_layout_delta",
                "per_image_multisource_coverage_selection",
                "image_placeholder_compaction",
                "sparse_mrope_rebuild",
            ),
            required_handles=(
                "vision_projector",
                "vision_deepstack_projectors",
                "llm",
            ),
            requires_visual_metrics=False,
            metadata={
                'shared_main_deepstack_delta': True,
                'text_source': 'prompt_input_embeddings',
                'vision_source': 'pre_merger_visual_features',
                'per_image_selection': True,
                'position_policy': 'sparse_mrope_rebuild',
            },
        ),
        Qwen3VLImageBindingContract(
            binding=QWEN3_VL_VISPRUNER_BINDING,
            algorithm="vispruner",
            variant=_VISPRUNER_VARIANT,
            callback="post_visual_merger_with_deepstack",
            target_sites=(
                "post_final_merger_pre_llm_scatter",
                "post_deepstack_mergers_pre_deepstack_injection",
            ),
            repairs=(
                "shared_main_deepstack_layout_delta",
                "per_image_visual_cue_selection",
                "image_placeholder_compaction",
                "sparse_mrope_rebuild",
            ),
            required_handles=(
                "vision_encoder",
                "vision_projector",
                "vision_deepstack_projectors",
                "llm",
            ),
            requires_visual_metrics=True,
            metadata={
                'shared_main_deepstack_delta': True,
                'attention_source': 'final_visual_mean_received',
                'selection_feature_source': 'pre_merger_visual_features',
                'per_image_selection': True,
                'position_policy': 'sparse_mrope_rebuild',
            },
        ),
        Qwen3VLImageBindingContract(
            binding=QWEN3_VL_ZOO_PRUNE_BINDING,
            algorithm="zoo_prune",
            variant=_ZOO_PRUNE_VARIANT,
            callback="post_visual_merger_with_deepstack",
            target_sites=(
                "post_final_merger_pre_llm_scatter",
                "post_deepstack_mergers_pre_deepstack_injection",
            ),
            repairs=(
                "shared_main_deepstack_layout_delta",
                "per_image_projector_sensitivity_selection",
                "image_placeholder_compaction",
                "sparse_mrope_rebuild",
            ),
            required_handles=(
                "vision_encoder",
                "vision_projector",
                "vision_deepstack_projectors",
                "llm",
            ),
            requires_visual_metrics=False,
            metadata={
                'shared_main_deepstack_delta': True,
                'projector_input': 'native_final_merger_mlp_input',
                'projector_proxy': 'native_merger_mlp_low_rank_linear_layers',
                'training_free': True,
                'single_image_only': True,
                'per_image_selection': True,
                'position_policy': 'sparse_mrope_rebuild',
            },
        ),
        Qwen3VLImageBindingContract(
            binding=QWEN3_VL_NUWA_BINDING,
            algorithm="nuwa",
            variant=_NUWA_VARIANT,
            callback="composite_post_merger_and_decoder_loop",
            target_sites=(
                "post_final_merger_pre_llm_scatter",
                "post_deepstack_mergers_pre_deepstack_injection",
                "llm_decoder_pre_stage2_layer_image_tokens",
            ),
            repairs=(
                "stage1_weighted_visual_aggregation",
                "shared_main_deepstack_layout_delta",
                "image_placeholder_compaction",
                "native_sparse_mrope_rpme_legalization",
                "stage2_decoder_sequence_compaction",
                "completed_prefix_dynamic_cache_compaction",
            ),
            required_handles=(
                "vision_encoder",
                "vision_projector",
                "vision_deepstack_projectors",
                "llm",
            ),
            requires_visual_metrics=True,
            metadata={
                'composite_binding': True,
                'shared_main_deepstack_delta': True,
                'single_image_only': True,
                'cache_policy': 'compact_prefix',
                'position_policy': 'native_sparse_mrope_rpme',
                'stage1_per_image_selection': True,
                'stage1_metric': 'visual_attention_received_times_key_norm',
                'semantic_hidden_range': (16, 25),
                'stage2_metric': 'last_prompt_attention',
                'cache_transform': 'drop',
            },
        ),
        Qwen3VLImageBindingContract(
            binding=QWEN3_VL_RESTORE_BINDING,
            algorithm="restore",
            variant=_RESTORE_VARIANT,
            callback="composite_post_merger_and_decoder_loop",
            target_sites=(
                "post_final_merger_pre_llm_scatter",
                "post_deepstack_mergers_pre_deepstack_injection",
                "llm_decoder_attention_calibration",
            ),
            repairs=(
                "vispruner_pristine_plus_distinctive_anchor_merge",
                "shared_main_deepstack_merge_delta",
                "image_placeholder_compaction",
                "native_sparse_mrope_representative_legalization",
                "token_mass_and_mrope_distance_attention_calibration",
            ),
            required_handles=(
                "vision_encoder",
                "vision_projector",
                "vision_deepstack_projectors",
                "llm",
            ),
            requires_visual_metrics=True,
            metadata={
                'composite_binding': True,
                'shared_main_deepstack_delta': True,
                'single_image_only': True,
                'cache_policy': 'stable',
                'cache_transform': 'none',
                'position_policy': 'native_sparse_mrope_representatives',
                'attention_calibration': 'token_mass_plus_mrope_distance',
                'per_image_visual_merge': True,
                'representative_policy': 'native_sparse_mrope_median',
            },
        ),
        Qwen3VLImageBindingContract(
            binding=QWEN3_VL_FASTV_BINDING,
            algorithm="fastv",
            variant=_FASTV_VARIANT,
            callback="decoder_layer_loop",
            target_sites=("llm_decoder_pre_prune_layer_image_tokens",),
            repairs=(
                "decoder_sequence_compaction",
                "sparse_mrope_position_selection",
                "remaining_deepstack_branch_compaction",
                "shallow_kv_cache_prefix_compaction",
            ),
            required_handles=("llm",),
            metadata={
                'requires_decoder_loop_control': True,
                'cache_policy': 'compact_prefix',
                'remaining_deepstack_synchronized': True,
                'single_decoder_boundary': True,
                'generate_requires_kv_cache_repair': True,
            },
        ),
        Qwen3VLImageBindingContract(
            binding=QWEN3_VL_SGL_BINDING,
            algorithm="sgl",
            variant=_SGL_VARIANT,
            callback="decoder_layer_loop",
            target_sites=("llm_decoder_pre_prune_layer_image_tokens",),
            repairs=(
                "guide_target_visual_identity",
                "decoder_sequence_compaction",
                "sparse_mrope_position_selection",
                "remaining_deepstack_branch_compaction",
                "shallow_kv_cache_prefix_compaction",
            ),
            required_handles=("llm",),
            metadata={
                'requires_decoder_loop_control': True,
                'cache_policy': 'compact_prefix',
                'remaining_deepstack_synchronized': True,
                'single_image_only': True,
                'training_free': True,
                'dual_model_lifecycle': True,
                'single_decoder_boundary': True,
                'generate_requires_kv_cache_repair': True,
                'position_policy': 'preserve_sparse_source',
                'guide_model_path_required': True,
            },
        ),
        Qwen3VLImageBindingContract(
            binding=QWEN3_VL_DART_BINDING,
            algorithm="dart",
            variant=_DART_VARIANT,
            callback="decoder_layer_loop",
            target_sites=("llm_decoder_pre_prune_layer_image_tokens",),
            repairs=(
                "decoder_sequence_compaction",
                "sparse_mrope_position_selection",
                "remaining_deepstack_branch_compaction",
                "shallow_kv_cache_prefix_compaction",
            ),
            required_handles=("llm",),
            metadata={
                'requires_decoder_loop_control': True,
                'cache_policy': 'compact_prefix',
                'remaining_deepstack_synchronized': True,
                'single_image_only': True,
                'single_decoder_boundary': True,
                'generate_requires_kv_cache_repair': True,
                'position_policy': 'preserve_sparse_source',
            },
        ),
        Qwen3VLImageBindingContract(
            binding=QWEN3_VL_D2PRUNER_BINDING,
            algorithm="d2pruner",
            variant=_D2PRUNER_VARIANT,
            callback="decoder_layer_loop",
            target_sites=("llm_decoder_pre_prune_layer_image_tokens",),
            repairs=(
                "decoder_sequence_compaction",
                "sparse_mrope_position_selection",
                "remaining_deepstack_branch_compaction",
                "shallow_kv_cache_prefix_compaction",
            ),
            required_handles=("llm",),
            metadata={
                'requires_decoder_loop_control': True,
                'cache_policy': 'compact_prefix',
                'remaining_deepstack_synchronized': True,
                'requires_calibrated_bias_prior': True,
                'single_image_only': True,
                'single_decoder_boundary': True,
                'generate_requires_kv_cache_repair': True,
                'position_policy': 'preserve_sparse_source',
            },
        ),
        Qwen3VLImageBindingContract(
            binding=QWEN3_VL_SPARSEVLM_BINDING,
            algorithm="sparsevlm",
            variant=_SPARSEVLM_VARIANT,
            callback="decoder_layer_loop_after_deepstack",
            target_sites=("llm_decoder_layer_output_image_tokens",),
            repairs=(
                "decoder_sequence_merge_and_compaction",
                "contiguous_mrope_rebuild",
                "completed_layer_kv_cache_mean_merge",
                "remaining_deepstack_mean_merge",
            ),
            required_handles=("llm",),
            metadata={
                'requires_decoder_loop_control': True,
                'cache_policy': 'compact_prefix',
                'remaining_deepstack_synchronized': True,
                'single_image_only': True,
                'multistage_decoder_boundaries': True,
                'generate_requires_kv_cache_repair': True,
                'cache_transform': 'mean_merge',
                'position_policy': 'contiguous',
            },
        ),
        Qwen3VLImageBindingContract(
            binding=QWEN3_VL_FINEPRUNER_BINDING,
            algorithm="finepruner",
            variant=_FINEPRUNER_VARIANT,
            callback="decoder_layer_loop_after_deepstack",
            target_sites=("llm_decoder_layer_output_image_tokens",),
            repairs=(
                "decoder_sequence_drop",
                "contiguous_mrope_rebuild",
                "layer_local_dynamic_cache",
            ),
            required_handles=("llm",),
            metadata={
                'requires_decoder_loop_control': True,
                'cache_policy': 'layer_local',
                'single_image_only': True,
                'attention_capture': 'last_text_query_per_head',
                'multistage_decoder_boundaries': True,
                'generate_requires_kv_cache_repair': True,
                'cache_transform': 'none',
                'position_policy': 'contiguous',
            },
        ),
        Qwen3VLImageBindingContract(
            binding=QWEN3_VL_BTP_BINDING,
            algorithm="btp",
            variant=_BTP_VARIANT,
            callback="decoder_layer_loop_after_deepstack",
            target_sites=("llm_decoder_layer_output_image_tokens",),
            repairs=(
                "decoder_sequence_drop",
                "sparse_mrope_position_selection",
                "completed_layer_kv_cache_prefix_compaction",
            ),
            required_handles=("llm",),
            metadata={
                'requires_decoder_loop_control': True,
                'cache_policy': 'compact_prefix',
                'single_image_only': True,
                'attention_capture': 'last_text_query_head_mean',
                'final_stage': 'remove_remaining_visual_span',
                'multistage_decoder_boundaries': True,
                'generate_requires_kv_cache_repair': True,
                'position_policy': 'preserve_original',
            },
        ),
        Qwen3VLImageBindingContract(
            binding=QWEN3_VL_HALFV_BINDING,
            algorithm="halfv",
            variant=_HALFV_VARIANT,
            callback="decoder_layer_loop_after_deepstack",
            target_sites=("llm_decoder_layer_output_image_tokens",),
            repairs=(
                "decoder_sequence_drop",
                "sparse_mrope_position_selection",
                "remaining_deepstack_branch_compaction",
                "completed_layer_kv_cache_prefix_compaction",
            ),
            required_handles=("llm",),
            metadata={
                'requires_decoder_loop_control': True,
                'cache_policy': 'compact_prefix',
                'remaining_deepstack_synchronized': True,
                'single_image_only': True,
                'attention_capture': 'last_text_query_head_mean',
                'position_policy': 'preserve_original',
                'score_rope_by_stage': {'ivr': False, 'ssr': True},
                'multistage_decoder_boundaries': True,
                'generate_requires_kv_cache_repair': True,
                'cache_transform': 'drop',
                'synchronize_remaining_deepstack': True,
                'paper_architecture_alignment': 'qwen_token_path_adaptation',
            },
        ),
        Qwen3VLImageBindingContract(
            binding=QWEN3_VL_TRANSPRUNE_BINDING,
            algorithm="transprune",
            variant=_TRANSPRUNE_VARIANT,
            callback="decoder_layer_loop_after_deepstack",
            target_sites=("llm_decoder_layer_output_image_tokens",),
            repairs=(
                "decoder_sequence_drop",
                "contiguous_mrope_rebuild",
                "remaining_deepstack_branch_compaction",
                "layer_local_kv_cache_lifecycle",
            ),
            required_handles=("llm",),
            metadata={
                'requires_decoder_loop_control': True,
                'cache_policy': 'layer_local',
                'single_image_only': True,
                'multi_stage': True,
                'position_policy': 'contiguous',
                'module_capture': ('attention', 'ffn'),
                'iga_source': 'next_layer_post_image_text',
                'synchronize_remaining_deepstack': True,
                'multistage_decoder_boundaries': True,
                'generate_requires_kv_cache_repair': True,
                'cache_transform': 'none',
            },
        ),
    )
})
