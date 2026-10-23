"""Built-in multimodal compression algorithms and exact support matrix."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import MISSING, fields, replace
from types import MappingProxyType
from typing import Any

from mllm_comp.compressors.contracts import (
    AlgorithmDescriptor,
    BackbonePlacementBinding,
    BindingTarget,
    CompressionContractRegistry,
    MediaAlignment,
    ModalityVariantDescriptor,
    TokenUnit,
)
from mllm_comp.compressors.audio_multimodal_registry import (
    qwen25_omni_audio_algorithm_descriptors,
    qwen25_omni_audio_bindings,
    qwen25_omni_audio_support_units,
    qwen25_omni_audio_variant_descriptors,
)
from mllm_comp.compressors.video.registry import (
    video_algorithm_descriptors,
    video_bindings,
    video_support_units,
    video_variant_descriptors,
)
from mllm_comp.compressors.image.contracts.llava_onevision import (
    LLAVA_ONEVISION_IMAGE_BINDING_CONTRACTS,
)
from mllm_comp.compressors.image.contracts.qwen25_vl import (
    QWEN25_VL_IMAGE_BINDING_CONTRACTS,
)
from mllm_comp.compressors.image.contracts.qwen3_vl import (
    QWEN3_VL_IMAGE_BINDING_CONTRACTS,
)
from mllm_comp.compressors.image.contracts.qwen3_5 import (
    QWEN3_5_DIVPRUNE_BINDING,
    QWEN3_5_DIVPRUNE_VARIANT,
    QWEN3_5_IMAGE_BINDING_CONTRACTS,
    QWEN3_5_TOME_BINDING,
    QWEN3_5_TOME_VARIANT,
    QWEN3_5_OTPRUNE_BINDING,
    QWEN3_5_OTPRUNE_VARIANT,
    QWEN3_5_PRUMERGE_BINDING,
    QWEN3_5_MMTOK_BINDING,
    QWEN3_5_GAP_BINDING,
    QWEN3_5_VISPRUNER_BINDING,
    QWEN3_5_AIM_BINDING,
    QWEN3_5_RESTORE_BINDING,
    QWEN3_5_BIFOCAL_BINDING,
    QWEN3_5_ZOO_PRUNE_BINDING,
    QWEN3_5_SAINT_BINDING,
    QWEN3_5_SAINT_VARIANT,
    QWEN3_5_NUWA_BINDING,
    QWEN3_5_NUWA_VARIANT,
    QWEN3_5_SPARSEVLM_BINDING,
    QWEN3_5_SPARSEVLM_VARIANT,
    QWEN3_5_FINEPRUNER_BINDING,
    QWEN3_5_FINEPRUNER_VARIANT,
    QWEN3_5_TRANSPRUNE_BINDING,
    QWEN3_5_TRANSPRUNE_VARIANT,
    QWEN3_5_FASTV_BINDING,
    QWEN3_5_PYRAMID_DROP_BINDING,
    QWEN3_5_DART_BINDING,
    QWEN3_5_IVC_PRUNE_BINDING,
    QWEN3_5_FEATHER_BINDING,
    QWEN3_5_BTP_BINDING,
    QWEN3_5_HALFV_BINDING,
    QWEN3_5_SGL_BINDING,
    QWEN3_5_D2PRUNER_BINDING,
    QWEN3_5_VISIONZIP_BINDING,
    QWEN3_5_VISIONZIP_VARIANT,
)
from mllm_comp.compressors.image.evidence import image_evidence
from mllm_comp.compressors.image.pyramid_drop import pyramid_drop
from mllm_comp.compressors.image.divprune import divprune_image
from mllm_comp.compressors.image.mmtok import mmtok_image_tokens
from mllm_comp.compressors.image.vispruner import vispruner_image_tokens
from mllm_comp.compressors.image.zoo_prune import zoo_prune_image_tokens
from mllm_comp.compressors.image.fastv import FastVSpec, fastv_image_tokens
from mllm_comp.compressors.image.sgl import SGLSpec, sgl_image_tokens
from mllm_comp.compressors.image.dart import DARTSpec, dart_image_tokens
from mllm_comp.compressors.image.d2pruner import D2PrunerSpec, d2pruner_image_tokens
from mllm_comp.compressors.image.sparsevlm import SparseVLMSpec, sparsevlm_image_tokens
from mllm_comp.compressors.image.finepruner import FinePrunerSpec, finepruner_iac_image_tokens
from mllm_comp.compressors.image.btp import BalancedTokenPruningSpec, balanced_token_pruning_image_tokens
from mllm_comp.compressors.image.halfv import HalfVSpec, halfv_image_tokens
from mllm_comp.compressors.image.transprune import TransPruneSpec, transprune_image_tokens
from mllm_comp.compressors.image.nuwa import nuwa_stage1_image_tokens
from mllm_comp.compressors.image.restore import restore_image_tokens
from mllm_comp.compressors.image.tome import tome_merge_image
from mllm_comp.compressors.image.saint import saint_image_tokens
from mllm_comp.compressors.image.aim import aim_image_tokens
from mllm_comp.compressors.image.pact import PACTSpec, pact_image_tokens
from mllm_comp.compressors.image.visionzip import visionzip_image
from mllm_comp.compressors.image.otprune import otprune_image
from mllm_comp.compressors.image.bifocal import bifocal_image_tokens
from mllm_comp.compressors.image.prumerge import prumerge_image_tokens
from mllm_comp.compressors.image.fixed_visual_merge import fixed_visual_merge
from mllm_comp.compressors.image.gap import gap_image_tokens
from mllm_comp.compressors.image.feather import FEATHERSpec, feather_image_tokens
from mllm_comp.compressors.image.ivc_prune import IVCPruneSpec, ivc_prune_image_tokens
from mllm_comp.core.support import (
    EvidenceKind,
    SupportStatus,
    SupportUnit,
    SupportUnitId,
)
from mllm_comp.core.vocabulary import (
    CompressionMethod,
    ExecutionPhase,
    Modality,
    ModalityCouplingPolicy,
    Placement,
    TokenOperation,
)


_IMAGE_BINDING_CONTRACTS = {
    "llava_onevision": LLAVA_ONEVISION_IMAGE_BINDING_CONTRACTS,
    "qwen25_vl": QWEN25_VL_IMAGE_BINDING_CONTRACTS,
    "qwen3_vl": QWEN3_VL_IMAGE_BINDING_CONTRACTS,
    "qwen3_5": QWEN3_5_IMAGE_BINDING_CONTRACTS,
}
_IMAGE_BACKBONES = ("llava_onevision", "qwen25_vl", "qwen3_vl")

# ``AlgorithmDescriptor.supported_placements`` describes the physical hook
# coverage of this implementation.  That is deliberately different from the
# stages where the paper's compression algorithm is defined: a binding may
# contain a structural legalizer, a framework adaptation, or a non-compressing
# calibration hook.  Keep the paper taxonomy explicit so documentation and
# tooling do not have to infer it from runtime targets.
_IMAGE_PAPER_COMPRESSION_STAGES: Mapping[
    str,
    tuple[tuple[Placement, CompressionMethod], ...],
] = MappingProxyType(
    {
        "aim": (
            (Placement.PROJECTOR_LEVEL, CompressionMethod.TOKEN_COMPRESSION),
            (Placement.LLM_LEVEL, CompressionMethod.TOKEN_COMPRESSION),
        ),
        "btp": ((Placement.LLM_LEVEL, CompressionMethod.TOKEN_COMPRESSION),),
        "d2pruner": (
            (Placement.LLM_LEVEL, CompressionMethod.TOKEN_COMPRESSION),
        ),
        "dart": ((Placement.LLM_LEVEL, CompressionMethod.TOKEN_COMPRESSION),),
        "divprune": (
            (Placement.PROJECTOR_LEVEL, CompressionMethod.TOKEN_COMPRESSION),
        ),
        "fastv": ((Placement.LLM_LEVEL, CompressionMethod.TOKEN_COMPRESSION),),
        "feather": ((Placement.LLM_LEVEL, CompressionMethod.TOKEN_COMPRESSION),),
        "finepruner": (
            (Placement.LLM_LEVEL, CompressionMethod.TOKEN_COMPRESSION),
        ),
        "gap": (
            (Placement.PROJECTOR_LEVEL, CompressionMethod.TOKEN_COMPRESSION),
        ),
        "feature_anchor_mean": (
            (Placement.PROJECTOR_LEVEL, CompressionMethod.TOKEN_COMPRESSION),
        ),
        "grid_mean": (
            (Placement.PROJECTOR_LEVEL, CompressionMethod.TOKEN_COMPRESSION),
        ),
        "halfv": (
            (Placement.LLM_LEVEL, CompressionMethod.TOKEN_COMPRESSION),
            (Placement.LLM_LEVEL, CompressionMethod.OPERATION_COMPRESSION),
        ),
        "ivc_prune": (
            (Placement.LLM_LEVEL, CompressionMethod.TOKEN_COMPRESSION),
        ),
        "mmtok": (
            (Placement.PROJECTOR_LEVEL, CompressionMethod.TOKEN_COMPRESSION),
        ),
        "nuwa": (
            (Placement.PROJECTOR_LEVEL, CompressionMethod.TOKEN_COMPRESSION),
            (Placement.LLM_LEVEL, CompressionMethod.TOKEN_COMPRESSION),
        ),
        "otprune": (
            (Placement.PROJECTOR_LEVEL, CompressionMethod.TOKEN_COMPRESSION),
        ),
        "pact": ((Placement.LLM_LEVEL, CompressionMethod.TOKEN_COMPRESSION),),
        "prumerge": (
            (Placement.PROJECTOR_LEVEL, CompressionMethod.TOKEN_COMPRESSION),
        ),
        "pyramid_drop": (
            (Placement.LLM_LEVEL, CompressionMethod.TOKEN_COMPRESSION),
        ),
        "restore": (
            (Placement.PROJECTOR_LEVEL, CompressionMethod.TOKEN_COMPRESSION),
        ),
        "saint": (
            (Placement.ENCODER_LEVEL, CompressionMethod.TOKEN_COMPRESSION),
            (Placement.LLM_LEVEL, CompressionMethod.TOKEN_COMPRESSION),
        ),
        "sgl": ((Placement.LLM_LEVEL, CompressionMethod.TOKEN_COMPRESSION),),
        "sparsevlm": (
            (Placement.LLM_LEVEL, CompressionMethod.TOKEN_COMPRESSION),
        ),
        "tome": (
            (Placement.ENCODER_LEVEL, CompressionMethod.TOKEN_COMPRESSION),
        ),
        "transprune": (
            (Placement.LLM_LEVEL, CompressionMethod.TOKEN_COMPRESSION),
        ),
        "visionzip": (
            (Placement.PROJECTOR_LEVEL, CompressionMethod.TOKEN_COMPRESSION),
        ),
        "bifocal": (
            (Placement.PROJECTOR_LEVEL, CompressionMethod.TOKEN_COMPRESSION),
        ),
        "vispruner": (
            (Placement.PROJECTOR_LEVEL, CompressionMethod.TOKEN_COMPRESSION),
        ),
        "zoo_prune": (
            (Placement.PROJECTOR_LEVEL, CompressionMethod.TOKEN_COMPRESSION),
        ),
    }
)

_IMAGE_PAPER_IMPLEMENTATION_ALIGNMENT: Mapping[
    str, tuple[str, ...]
] = MappingProxyType(
    {
        "aim": (
            "partial",
            "The pre-LLM merge is implemented; progressive LLM pruning is not.",
        ),
        "gap": (
            "platform_composition",
            "GAP defines position-ID preservation as an add-on; the bundled "
            "token selector is a platform baseline.",
        ),
        "halfv": (
            "partial",
            "The released Qwen token-pruning path is shared across backbones; "
            "the LLaVA/Vicuna static visual-computation path is not implemented.",
        ),
        "restore": (
            "aligned_with_auxiliary_hook",
            "The LLM hook calibrates attention and does not itself compress "
            "tokens or operations.",
        ),
        "saint": (
            "partial",
            "Qwen2.5-VL implements the encoder path; LLaVA-OneVision and "
            "Qwen3-VL use pre-LLM adaptations, and LLM-only pruning is absent.",
        ),
        "tome": (
            "aligned_with_framework_adaptation",
            "The paper method merges tokens inside the encoder; post-projector "
            "compaction is a framework adaptation after structural restoration.",
        ),
    }
)


def _resolve_image_position(**kwargs: Any) -> Any:
    from mllm_comp.compressors.image.positions import resolve_image_position

    return resolve_image_position(**kwargs)


def _llava_onevision_executor_factory(**kwargs: Any) -> Any:
    from mllm_comp.backbones.llava_onevision.image_compression import (
        create_llava_onevision_image_compression_executor,
    )

    return create_llava_onevision_image_compression_executor(**kwargs)


def _qwen25_vl_executor_factory(**kwargs: Any) -> Any:
    from mllm_comp.backbones.qwen25_vl.image_compression import (
        create_qwen25_vl_image_compression_executor,
    )

    return create_qwen25_vl_image_compression_executor(**kwargs)


def _qwen3_vl_executor_factory(**kwargs: Any) -> Any:
    from mllm_comp.backbones.qwen3_vl.compression_adapter import (
        create_qwen3_vl_image_compression_executor,
    )

    return create_qwen3_vl_image_compression_executor(**kwargs)


def _qwen3_5_executor_factory(**kwargs: Any) -> Any:
    from mllm_comp.backbones.qwen3_5.compression import (
        create_qwen3_5_image_compression_executor,
    )

    return create_qwen3_5_image_compression_executor(**kwargs)


_IMAGE_EXECUTOR_FACTORIES = {
    "llava_onevision": _llava_onevision_executor_factory,
    "qwen25_vl": _qwen25_vl_executor_factory,
    "qwen3_vl": _qwen3_vl_executor_factory,
    "qwen3_5": _qwen3_5_executor_factory,
}
_IMAGE_METHODS = (
    "tome",
    "saint",
    "aim",
    "pact",
    "visionzip",
    "otprune",
    "pyramid_drop",
    "divprune",
    "mmtok",
    "vispruner",
    "zoo_prune",
    "fastv",
    "sgl",
    "dart",
    "d2pruner",
    "sparsevlm",
    "finepruner",
    "btp",
    "halfv",
    "transprune",
    "nuwa",
    "restore",
    "feature_anchor_mean",
    "grid_mean",
    "prumerge",
    "gap",
    "feather",
    "ivc_prune",
)
_ACTIVE_FORWARD_ALGORITHMS = frozenset(
    {
        "tome",
        "saint",
        "aim",
        "pact",
        "visionzip",
        "pyramid_drop",
        "divprune",
        "mmtok",
        "vispruner",
        "zoo_prune",
        "prumerge",
        "gap",
    }
)
_ACTIVE_GENERATE_ALGORITHMS = frozenset(
    {
        "pyramid_drop",
        "pact",
        "fastv",
        "dart",
        "sparsevlm",
        "finepruner",
        "btp",
        "halfv",
        "transprune",
        "nuwa",
        "restore",
        "sgl",
        "feather",
        "ivc_prune",
    }
)


def _image_method_conflicts(algorithm: str, backbone: str) -> frozenset[str]:
    """Make the one-method-per-model policy explicit and symmetric."""

    return frozenset(
        f"{candidate}.image.{backbone}"
        for candidate in _IMAGE_METHODS
        if candidate != algorithm
    )


def _algorithm_descriptors() -> tuple[AlgorithmDescriptor, ...]:
    descriptors = (
        AlgorithmDescriptor(
            name="tome",
            version="iclr23-reference-v1",
            method=CompressionMethod.TOKEN_COMPRESSION,
            kernel=tome_merge_image,
            supported_placements=frozenset(
                {Placement.ENCODER_LEVEL, Placement.PROJECTOR_LEVEL}
            ),
            operations=frozenset({TokenOperation.MERGE}),
            metadata={
                "paper": "Token Merging: Your ViT But Faster",
                "paper_default": "absolute_r_per_encoder_layer",
            },
        ),
        AlgorithmDescriptor(
            name="saint",
            version="qwen25-supplied-reference-v1",
            method=CompressionMethod.TOKEN_COMPRESSION,
            kernel=saint_image_tokens,
            supported_placements=frozenset(
                {Placement.ENCODER_LEVEL, Placement.PROJECTOR_LEVEL}
            ),
            operations=frozenset({TokenOperation.KEEP, TokenOperation.DROP}),
            metadata={
                "paper": "Similarity-Aware Token Pruning: Your VLM but Faster",
                "reference_default": {
                    "reduction_layers": [16, 24],
                    "keep_ratios": [0.5, 0.4],
                    "prune_level": "token",
                    "similarity_threshold": 0.7,
                    "neighbour_count": 5,
                    "gamma": 10.0,
                },
                "source": "supplied_qwen2_5_vl_implementation",
            },
        ),
        AlgorithmDescriptor(
            name="visionzip",
            version="cvpr25-reference-v1",
            method=CompressionMethod.TOKEN_COMPRESSION,
            kernel=visionzip_image,
            supported_placements=frozenset({Placement.PROJECTOR_LEVEL}),
            operations=frozenset({TokenOperation.KEEP, TokenOperation.MERGE}),
            metadata={
                "paper": (
                    "VisionZip: Longer is Better but Not Necessary in Vision "
                    "Language Models"
                ),
                "reference_default": {
                    "budget": 64,
                    "dominant_tokens": 54,
                    "contextual_tokens": 10,
                },
                "framework_default": {
                    "budget": 32,
                    "dominant_weight": 27,
                    "contextual_weight": 5,
                },
                "released_merge": "target_plus_assigned_mean",
                "projector_scope": "encoder_output_through_llm_input",
            },
        ),
        AlgorithmDescriptor(
            name="otprune",
            version="arxiv26-reference-adapted-v1",
            method=CompressionMethod.TOKEN_COMPRESSION,
            kernel=otprune_image,
            supported_placements=frozenset({Placement.PROJECTOR_LEVEL}),
            operations=frozenset({TokenOperation.KEEP, TokenOperation.DROP}),
            metadata={
                "paper": "OTPrune: Distribution-Aligned Visual Token Pruning via Optimal Transport",
                "reference_default": {"gamma": 0.01, "retain_ratio": 0.098},
                "framework_default": {"gamma": 0.01, "retain_ratio": 0.5},
                "selection": "greedy_logdet_cholesky",
                "projector_scope": "post_vision_merger_pre_llm",
            },
        ),
        AlgorithmDescriptor(
            name="bifocal",
            version="bifocal-qwen25-stage1-v1",
            method=CompressionMethod.TOKEN_COMPRESSION,
            kernel=bifocal_image_tokens,
            supported_placements=frozenset({Placement.PROJECTOR_LEVEL}),
            operations=frozenset({TokenOperation.KEEP, TokenOperation.MERGE}),
            metadata={
                "source": "original_qwen2_5_vl_research_implementation_stage1",
                "scope": "stage1_only",
                "coarse": "block_weighted_pool_centroid_residual_pool",
                "fine": "facility_location_plus_query_relevance(salr+crn)",
                "position": "rpme_native_mrope_slice",
                "vit_metric": "chunked_column_mass(mass|nodiag|vmass)",
            },
        ),
        AlgorithmDescriptor(
            name="pyramid_drop",
            version="cvpr25-reference-v1",
            method=CompressionMethod.TOKEN_COMPRESSION,
            kernel=pyramid_drop,
            supported_placements=frozenset({Placement.LLM_LEVEL}),
            operations=frozenset({TokenOperation.KEEP, TokenOperation.DROP}),
            metadata={
                "paper": "PyramidDrop: Accelerating Your LLVLMs via Pyramid Visual Redundancy Reduction",
                "paper_default": {
                    "num_stages": 4,
                    "keep_ratio": 0.5,
                    "layers_for_32_layer_llm": [7, 15, 23],
                },
            },
        ),
        AlgorithmDescriptor(
            name="feature_anchor_mean",
            version="stage1-proxy-v1",
            method=CompressionMethod.TOKEN_COMPRESSION,
            kernel=fixed_visual_merge,
            supported_placements=frozenset({Placement.PROJECTOR_LEVEL}),
            operations=frozenset({TokenOperation.MERGE}),
            metadata={
                "source": "visual_merge_attribution_stage1_proxy",
                "partition": "deterministic_farthest_first_cosine_hard_assignment",
                "aggregation": "fp32_cluster_mean",
                "reference_default": {"target_count": 64},
                "position_policy": "rounded_native_mrope_cluster_centroid",
            },
        ),
        AlgorithmDescriptor(
            name="grid_mean",
            version="stage1-grid-control-v1",
            method=CompressionMethod.TOKEN_COMPRESSION,
            kernel=fixed_visual_merge,
            supported_placements=frozenset({Placement.PROJECTOR_LEVEL}),
            operations=frozenset({TokenOperation.MERGE}),
            metadata={
                "source": "visual_merge_attribution_stage1_grid_control",
                "partition": "balanced_rectangular_exact_factorization",
                "aggregation": "fp32_cluster_mean",
                "reference_default": {"target_count": 64},
                "position_policy": "rounded_native_mrope_cluster_centroid",
            },
        ),
        AlgorithmDescriptor(
            name="prumerge",
            version="iccv25-fuxian-8989c430-v1",
            method=CompressionMethod.TOKEN_COMPRESSION,
            kernel=prumerge_image_tokens,
            supported_placements=frozenset({Placement.PROJECTOR_LEVEL}),
            operations=frozenset({TokenOperation.KEEP, TokenOperation.MERGE}),
            metadata={
                "paper": "LLaVA-PruMerge: Adaptive Token Reduction for Efficient Large Multimodal Models",
                "reference_commit": "8989c4304db1f8aa22627ae3191a4b61d0f0cd51",
                "reference_default": {
                    "variant": "plus",
                    "adaptive": True,
                    "reduction_ratio": 0.125,
                    "cluster_size": 32,
                },
                "merge_mode": "official_unnormalized_attention_sum",
            },
        ),
        AlgorithmDescriptor(
            name="gap",
            version="arxiv2506-paper-reproduction-v1",
            method=CompressionMethod.TOKEN_COMPRESSION,
            kernel=gap_image_tokens,
            supported_placements=frozenset({Placement.PROJECTOR_LEVEL}),
            operations=frozenset({TokenOperation.KEEP, TokenOperation.DROP}),
            metadata={
                "paper": (
                    "Grounding-Aware Token Pruning: Recovering from Drastic "
                    "Performance Drops in Visual Grounding Caused by Pruning"
                ),
                "source": "paper_reproduction_no_public_author_code_as_of_2026_08_17",
                "reference_default": {"retain_ratio": 0.5},
                "position_policy": "preserve_original",
            },
        ),
        AlgorithmDescriptor(
            name="feather",
            version="arxiv2412-c2a09b27-v1",
            method=CompressionMethod.TOKEN_COMPRESSION,
            kernel=feather_image_tokens,
            supported_placements=frozenset({Placement.LLM_LEVEL}),
            operations=frozenset({TokenOperation.KEEP, TokenOperation.DROP}),
            metadata={
                "paper": "Feather the Throttle",
                "reference_commit": "c2a09b2765967601054c7b1fd513ac3e94ee4fc9",
                "reference_default": {
                    "stage_layers": [8, 16],
                    "reduction_ratio": 0.7,
                    "uniform_stride": 3,
                },
                "score_source": "last_prompt_no_rope",
            },
        ),
        AlgorithmDescriptor(
            name="ivc_prune",
            version="arxiv2602-03afd9ce-v1",
            method=CompressionMethod.TOKEN_COMPRESSION,
            kernel=ivc_prune_image_tokens,
            supported_placements=frozenset({Placement.LLM_LEVEL}),
            operations=frozenset({TokenOperation.KEEP, TokenOperation.DROP}),
            metadata={
                "paper": (
                    "IVC-Prune: Revealing the Implicit Visual Coordinates in "
                    "LVLMs for Vision Token Pruning"
                ),
                "reference_commit": "03afd9cee41f60bb30b51a4105839f2b5d467814",
                "reference_default": {
                    "prune_layer": 16,
                    "total_retain_ratio": 0.5,
                    "axis_ratio_each": 0.1,
                    "semantic_seed_ratio": 0.01,
                },
                "coordinate_source": "backbone_native_rope_cos_sin",
            },
        ),
        AlgorithmDescriptor(
            name="divprune",
            version="cvpr25-fuxian-4ca7149-v1",
            method=CompressionMethod.TOKEN_COMPRESSION,
            kernel=divprune_image,
            supported_placements=frozenset({Placement.PROJECTOR_LEVEL}),
            operations=frozenset({TokenOperation.KEEP, TokenOperation.DROP}),
            metadata={
                "paper": "DivPrune: Diversity-based Visual Token Pruning for Large Multimodal Models",
                "paper_default": {
                    "retain_ratio": 0.098,
                    "prune_layer": 0,
                    "distance": "cosine",
                    "selection": "official_greedy_maxmin",
                },
                "reference_commit": "4ca7149ca478c634c3a1d41376099760ea106b4d",
            },
        ),
        AlgorithmDescriptor(
            name="aim",
            version="iccv25-supplied-qwen25-v1",
            method=CompressionMethod.TOKEN_COMPRESSION,
            kernel=aim_image_tokens,
            supported_placements=frozenset({Placement.PROJECTOR_LEVEL}),
            operations=frozenset({TokenOperation.MERGE}),
            metadata={
                "paper": (
                    "AIM: Adaptive Inference of Multi-Modal LLMs via Token "
                    "Merging and Pruning"
                ),
                "reference_default": {"retain_ratio": 0.2},
                "source": "supplied_qwen2_5_vl_implementation",
                "differentiable_merge": True,
            },
        ),
        AlgorithmDescriptor(
            name="pact",
            version="cvpr25-supplied-qwen25-v1",
            method=CompressionMethod.TOKEN_COMPRESSION,
            kernel=pact_image_tokens,
            supported_placements=frozenset({Placement.LLM_LEVEL}),
            operations=frozenset(
                {TokenOperation.KEEP, TokenOperation.DROP, TokenOperation.MERGE}
            ),
            metadata={
                "paper": "PACT: Pruning And Clustering-based Token reduction",
                "reference_default": {
                    "reduction_layer": 4,
                    "prune_keep_ratio": 1.0,
                    "target_keep_ratio": 0.3,
                    "proportional_attention": True,
                },
                "source": "supplied_qwen2_5_vl_implementation",
                "single_decoder_boundary": True,
            },
        ),
        AlgorithmDescriptor(
            name="mmtok",
            version="iclr26-official-8e26868-qwen25-v2",
            method=CompressionMethod.TOKEN_COMPRESSION,
            kernel=mmtok_image_tokens,
            supported_placements=frozenset({Placement.PROJECTOR_LEVEL}),
            operations=frozenset({TokenOperation.KEEP, TokenOperation.DROP}),
            metadata={
                "paper": (
                    "MMTok: Multimodal Coverage Maximization for Efficient "
                    "Inference of VLMs"
                ),
                "paper_default": {
                    "llava_tokens_per_crop": 32,
                    "qwen_retain_ratio": 0.2,
                    "alpha": 0.5,
                    "text_temperature": 0.02,
                    "vision_temperature": 0.2,
                },
                "reference_commit": "8e2686836ed5bbe05d064f1d1c20e50c73c70b94",
                "official_release_commit": (
                    "8e2686836ed5bbe05d064f1d1c20e50c73c70b94"
                ),
                "qwen25_text_source": "Question-prefix user text; no answers",
                "qwen25_vision_source": "pre_merger_2x2_mean",
                "shared_kernel_entrypoints": ["multimodal_coverage_greedy"],
            },
        ),
        AlgorithmDescriptor(
            name="vispruner",
            version="iccv25-aefa01a-fuxian-4ca7149-native-v1",
            method=CompressionMethod.TOKEN_COMPRESSION,
            kernel=vispruner_image_tokens,
            supported_placements=frozenset({Placement.PROJECTOR_LEVEL}),
            operations=frozenset({TokenOperation.KEEP, TokenOperation.DROP}),
            metadata={
                "paper": (
                    "Beyond Text-Visual Attention: Exploiting Visual Cues for "
                    "Effective Token Pruning in VLMs"
                ),
                "paper_default": {
                    "target_count": 32,
                    "important_ratio": 0.5,
                    "removal_step": 8,
                },
                "reference_commit": "4ca7149ca478c634c3a1d41376099760ea106b4d",
                "official_release_commit": "aefa01a",
                "shared_kernel_entrypoints": [
                    "visual_attention_importance",
                    "iterative_diversity_selection",
                ],
            },
        ),
        AlgorithmDescriptor(
            name="zoo_prune",
            version="cvpr26-c7d05ca-fuxian-4ca7149-native-v1",
            method=CompressionMethod.TOKEN_COMPRESSION,
            kernel=zoo_prune_image_tokens,
            supported_placements=frozenset({Placement.PROJECTOR_LEVEL}),
            operations=frozenset({TokenOperation.KEEP, TokenOperation.DROP}),
            metadata={
                "paper": (
                    "ZOO-Prune: Training-Free Token Pruning via Zeroth-Order "
                    "Gradient Estimation in Vision-Language Models"
                ),
                "paper_default": {
                    "retention_ratio": 0.1112,
                    "num_directions": 64,
                    "step_size": 0.01,
                    "projector_proxy": "low_rank",
                    "low_rank_rank": 128,
                    "low_rank_solver": "exact",
                    "fusion": "multiply",
                },
                "reference_commit": "4ca7149ca478c634c3a1d41376099760ea106b4d",
                "official_release_commit": (
                    "c7d05ca3a12a23f3138efd9c021445e2939d8d4f"
                ),
                "shared_kernel_entrypoints": [
                    "centered_projector_sensitivity",
                    "sensitivity_aware_diverse_selection",
                ],
                "training_free": True,
                "attention_free": True,
            },
        ),
        AlgorithmDescriptor(
            name="fastv",
            version="eccv24-fuxian-4ca7149-v1",
            method=CompressionMethod.TOKEN_COMPRESSION,
            kernel=fastv_image_tokens,
            supported_placements=frozenset({Placement.LLM_LEVEL}),
            operations=frozenset({TokenOperation.KEEP, TokenOperation.DROP}),
            metadata={
                "paper": "An Image is Worth 1/2 Tokens After Layer 2",
                "paper_default": {
                    "prune_layer": 2,
                    "prune_ratio": 0.5,
                    "score_source": "last_token",
                    "min_visual_tokens": 1,
                },
                "reference_commit": "4ca7149ca478c634c3a1d41376099760ea106b4d",
            },
        ),
        AlgorithmDescriptor(
            name="sgl",
            version="cvpr25-fuxian-4ca7149-sgp-v1",
            method=CompressionMethod.TOKEN_COMPRESSION,
            kernel=sgl_image_tokens,
            supported_placements=frozenset({Placement.LLM_LEVEL}),
            operations=frozenset({TokenOperation.KEEP, TokenOperation.DROP}),
            metadata={
                "paper": (
                    "A Stitch in Time Saves Nine: Small VLM is a Precise "
                    "Guidance for Accelerating Large VLMs"
                ),
                "paper_default": {
                    "mode": "sgp",
                    "prune_layer_fraction": 0.05,
                    "keep_ratio": 0.05,
                    "attention_sources": "prompt_and_generated",
                    "attention_aggregation": "sum_layers_heads_queries",
                },
                "reference_commit": "4ca7149ca478c634c3a1d41376099760ea106b4d",
                "official_release_commit": (
                    "15742bb7f428b4097a13591fc2e4d7fed99cf12d"
                ),
                "released_executable_scope": "sgp_only",
                "training_free": True,
                "dual_model_lifecycle": True,
            },
        ),
        AlgorithmDescriptor(
            name="dart",
            version="emnlp25-fuxian-4ca7149-v1",
            method=CompressionMethod.TOKEN_COMPRESSION,
            kernel=dart_image_tokens,
            supported_placements=frozenset({Placement.LLM_LEVEL}),
            operations=frozenset({TokenOperation.KEEP, TokenOperation.DROP}),
            metadata={
                "paper": "Stop Looking for Important Tokens in Multimodal Language Models: Duplication Matters More",
                "paper_default": {
                    "prune_after_layer": 2,
                    "retain_ratio": 1.0 / 9.0,
                    "pivot_image_tokens": 4,
                    "pivot_text_tokens": 4,
                    "allocation": "paper_exact",
                },
                "reference_commit": "4ca7149ca478c634c3a1d41376099760ea106b4d",
            },
        ),
        AlgorithmDescriptor(
            name="d2pruner",
            version="aaai26-fuxian-4ca7149-v1",
            method=CompressionMethod.TOKEN_COMPRESSION,
            kernel=d2pruner_image_tokens,
            supported_placements=frozenset({Placement.LLM_LEVEL}),
            operations=frozenset({TokenOperation.KEEP, TokenOperation.DROP}),
            metadata={
                "paper": (
                    "D2Pruner: Debiased Importance and Structural Diversity for "
                    "MLLM Token Pruning"
                ),
                "paper_default": {
                    "prune_after_layer": 2,
                    "retain_ratio": 0.25,
                    "pivot_ratio": 0.7,
                    "semantic_weight": 0.5,
                    "similarity_threshold": 0.8,
                    "bias_prior": "coco1000_calibrated_attention",
                },
                "reference_commit": "4ca7149ca478c634c3a1d41376099760ea106b4d",
            },
        ),
        AlgorithmDescriptor(
            name="sparsevlm",
            version="fuxian-4ca7149-official-v1",
            method=CompressionMethod.TOKEN_COMPRESSION,
            kernel=sparsevlm_image_tokens,
            supported_placements=frozenset({Placement.LLM_LEVEL}),
            operations=frozenset({TokenOperation.KEEP, TokenOperation.MERGE}),
            metadata={
                "paper": "SparseVLM: Visual Token Sparsification for Efficient VLM Inference",
                "paper_default": {
                    "variant": "official_v1",
                    "prune_layers": [2, 6, 15],
                    "schedule_keep_tokens": [300, 200, 110],
                    "recycle_ratio": 0.3,
                    "cluster_divisor": 10,
                },
                "reference_commit": "4ca7149ca478c634c3a1d41376099760ea106b4d",
            },
        ),
        AlgorithmDescriptor(
            name="finepruner",
            version="fuxian-4ca7149-paper-faithful-v1",
            method=CompressionMethod.TOKEN_COMPRESSION,
            kernel=finepruner_iac_image_tokens,
            supported_placements=frozenset({Placement.LLM_LEVEL}),
            operations=frozenset({TokenOperation.KEEP, TokenOperation.DROP}),
            metadata={
                "paper": (
                    "FinePruner: Unbiased Attention-Head-Level Fine-grained Token "
                    "Reduction for Efficient Inference of Large Vision-Language "
                    "Models"
                ),
                "paper_default": {
                    "iac_after_layers": [7, 14],
                    "iac_hash_bits": [48, 41],
                    "arp_target_after_layers": [20, 21, 22],
                    "arp_retain_ratio": 0.6,
                    "head_ratio": 0.5,
                    "random_seed": 126,
                },
                "reference_commit": "4ca7149ca478c634c3a1d41376099760ea106b4d",
                "shared_kernel_entrypoints": ["iac", "arp"],
            },
        ),
        AlgorithmDescriptor(
            name="btp",
            version="neurips25-fuxian-4ca7149-qwen25-vl-v1",
            method=CompressionMethod.TOKEN_COMPRESSION,
            kernel=balanced_token_pruning_image_tokens,
            supported_placements=frozenset({Placement.LLM_LEVEL}),
            operations=frozenset({TokenOperation.KEEP, TokenOperation.DROP}),
            metadata={
                "paper": (
                    "Balanced Token Pruning: Accelerating Vision Language "
                    "Models Beyond Local Optimization"
                ),
                "paper_default": {
                    "prune_before_layers": [4, 7, 16, 23],
                    "retain_ratios": [0.5, 0.5, 0.5, 0.0],
                    "attention_fractions": [0.1, 0.4, 1.0, 0.0],
                    "attention_oversample": [1.2, 1.0, 1.2, 1.0],
                    "rebalance_stages": [False, False, True, False],
                    "release_variant": "qwen2_5_vl",
                },
                "reference_commit": "4ca7149ca478c634c3a1d41376099760ea106b4d",
                "shared_kernel_entrypoints": [
                    "attention_seeded_maxmin",
                    "rebalanced_attention",
                    "final_wipe",
                ],
            },
        ),
        AlgorithmDescriptor(
            name="halfv",
            version="acl26-fuxian-4ca7149-qwen25-vl-v1",
            method=CompressionMethod.TOKEN_COMPRESSION,
            kernel=halfv_image_tokens,
            supported_placements=frozenset({Placement.LLM_LEVEL}),
            operations=frozenset({TokenOperation.KEEP, TokenOperation.DROP}),
            metadata={
                "paper": (
                    "From Inheritance to Saturation: Disentangling the "
                    "Evolution of Visual Redundancy for Architecture-Aware "
                    "MLLM Inference Acceleration"
                ),
                "paper_default": {
                    "ivr_layer_index": 2,
                    "ivr_retention_ratio": 0.25,
                    "anchor_ratio": 0.1,
                    "ssr_layer_index": 21,
                    "ssr_retention_ratio": 0.05,
                    "release_variant": "qwen2_5_vl",
                },
                "reference_commit": "4ca7149ca478c634c3a1d41376099760ea106b4d",
                "official_qwen25_vl_commit": (
                    "496f50192a8bc92788061ebd1e3138f5c1cda089"
                ),
                "shared_kernel_entrypoints": [
                    "last_text_attention",
                    "anchor_cover",
                    "ssr_attention_topk",
                ],
            },
        ),
        AlgorithmDescriptor(
            name="transprune",
            version="cvpr26-fuxian-4ca7149-native-v1",
            method=CompressionMethod.TOKEN_COMPRESSION,
            kernel=transprune_image_tokens,
            supported_placements=frozenset({Placement.LLM_LEVEL}),
            operations=frozenset({TokenOperation.KEEP, TokenOperation.DROP}),
            metadata={
                "paper": (
                    "TransPrune: Token Transition Pruning for Efficient "
                    "Large Vision-Language Model"
                ),
                "paper_default": {
                    "accumulating_layers": [7, 8, 9, 10, 11, 12],
                    "prune_after_layers": [7, 9, 12],
                    "retain_ratios": [0.875, 0.625, 0.125],
                    "alpha": 0.5,
                    "direction_rule": "release_1_minus_cosine",
                    "ttv_softmax_scope": "release_full_sequence",
                },
                "reference_commit": "4ca7149ca478c634c3a1d41376099760ea106b4d",
                "official_release_commit": (
                    "c9e8284f5808f7692c40590ffc99537e9220a01d"
                ),
                "shared_kernel_entrypoints": [
                    "module_transition_variation",
                    "instruction_guided_attention",
                    "progressive_selection",
                ],
            },
        ),
        AlgorithmDescriptor(
            name="nuwa",
            version="fuxian-4ca7149-released-two-stage-v1",
            method=CompressionMethod.TOKEN_COMPRESSION,
            kernel=nuwa_stage1_image_tokens,
            supported_placements=frozenset(
                {Placement.PROJECTOR_LEVEL, Placement.LLM_LEVEL}
            ),
            operations=frozenset(
                {TokenOperation.KEEP, TokenOperation.DROP, TokenOperation.MERGE}
            ),
            metadata={
                "paper": (
                    "Nüwa: Mending the Spatial Integrity Torn by VLM Token Pruning"
                ),
                "paper_default": {
                    "stage1_total_tokens": 582,
                    "stage2_after_layer": 16,
                    "stage2_retain_ratio": 0.125,
                    "stage2_score": "released_last_text_attention",
                    "region_size": 2,
                    "top_n_per_region": 3,
                    "pillar_quantile": 0.75,
                    "distance_squared_ratio": 0.26,
                    "aggregation_self_mode": "released_add",
                    "semantic_hidden_range": [16, 25],
                    "position_mode": "rpme",
                },
                "reference_commit": "4ca7149ca478c634c3a1d41376099760ea106b4d",
                "shared_kernel_entrypoints": [
                    "regional_benchmark_selection",
                    "semantic_spatial_aggregation",
                    "last_prompt_attention_selection",
                    "relative_position_mapping",
                ],
                "two_stage": True,
            },
        ),
        AlgorithmDescriptor(
            name="restore",
            version="fuxian-4ca7149-released-hybrid-calibration-v1",
            method=CompressionMethod.TOKEN_COMPRESSION,
            kernel=restore_image_tokens,
            supported_placements=frozenset(
                {Placement.PROJECTOR_LEVEL, Placement.LLM_LEVEL}
            ),
            operations=frozenset({TokenOperation.KEEP, TokenOperation.MERGE}),
            metadata={
                "paper": (
                    "Improving Visual Token Reduction via Rectifying Distortions "
                    "for Efficient Multimodal LLM Inference"
                ),
                "paper_default": {
                    "target_count": 64,
                    "pruning_ratio": 0.5,
                    "important_ratio": 0.5,
                    "removal_step": 8,
                    "anchor_mask_mode": "released",
                    "position_strategy": "median",
                    "calibration_constant": 2.0,
                    "first_layer_size": "uniform",
                    "min_visual_tokens": 2,
                },
                "reference_commit": "4ca7149ca478c634c3a1d41376099760ea106b4d",
                "shared_kernel_entrypoints": [
                    "restore_image_tokens",
                    "distinctive_anchor_merge",
                    "restore_attention_bias",
                ],
                "hybrid_visual_merge_and_decoder_calibration": True,
            },
        ),
    )

    normalized: list[AlgorithmDescriptor] = []
    for descriptor in descriptors:
        alignment, *notes = _IMAGE_PAPER_IMPLEMENTATION_ALIGNMENT.get(
            descriptor.name,
            ("aligned",),
        )
        metadata = {
            **descriptor.metadata,
            "paper_compression_stages": tuple(
                {
                    "placement": placement.value,
                    "method": method.value,
                }
                for placement, method in _IMAGE_PAPER_COMPRESSION_STAGES[
                    descriptor.name
                ]
            ),
            "paper_implementation_alignment": alignment,
        }
        if notes:
            metadata["paper_implementation_note"] = notes[0]
        normalized.append(replace(descriptor, metadata=metadata))
    return tuple(normalized)


def _spec_defaults(spec_type: type[Any], *, exclude: tuple[str, ...] = ()) -> dict[str, Any]:
    """Read algorithm defaults without constructing a resource-dependent Spec."""
    return {
        field.name: field.default
        for field in fields(spec_type)
        if field.default is not MISSING and field.name not in exclude
    }


def _variant_descriptors() -> tuple[ModalityVariantDescriptor, ...]:
    preserved = frozenset({Modality.TEXT, Modality.VIDEO, Modality.AUDIO})
    return (
        ModalityVariantDescriptor(
            name="tome.image.encoder_dense_boundary",
            algorithm="tome",
            target_modalities=frozenset({Modality.IMAGE}),
            preserved_modalities=preserved,
            coupling_policy=ModalityCouplingPolicy.INDEPENDENT,
            media_alignment=MediaAlignment.SPATIAL,
            token_unit=TokenUnit.ENCODER_TOKEN,
            supported_rules=frozenset({
                "encoder_aggregator", "encoder_grouper", "encoder_scorer",
                "output_aggregator", "output_grouper", "output_scorer",
            }),
            config_defaults={
                "dense_restore_at_structural_boundaries": True,
                "permanent_output_compaction": True,
                "output_r": 1,
            },
            metadata={
                "scope": "image_only",
                "compute_saving": "remaining_encoder_layers",
                "llm_token_reduction": True,
                "structural_restore_is_temporary": True,
            },
        ),
        ModalityVariantDescriptor(
            name=QWEN3_5_TOME_VARIANT,
            algorithm="tome",
            target_modalities=frozenset({Modality.IMAGE}),
            preserved_modalities=preserved,
            coupling_policy=ModalityCouplingPolicy.INDEPENDENT,
            media_alignment=MediaAlignment.SPATIAL,
            token_unit=TokenUnit.FEATURE_TOKEN,
            config_defaults={"retain_ratio": 0.5},
            metadata={
                "scope": "image_only",
                "placement": "post_merger_pre_llm",
                "mechanism": "ToMe feature merging",
                "initial_integration_smoke": True,
            },
        ),
        ModalityVariantDescriptor(
            name=QWEN3_5_DIVPRUNE_VARIANT,
            algorithm="divprune",
            target_modalities=frozenset({Modality.IMAGE}),
            preserved_modalities=preserved,
            coupling_policy=ModalityCouplingPolicy.INDEPENDENT,
            media_alignment=MediaAlignment.SPATIAL,
            token_unit=TokenUnit.FEATURE_TOKEN,
            supported_rules=frozenset({"scorer", "selector"}),
            config_defaults={
                "retain_ratio": 0.5,
                "target_count": None,
                "prune_layer": 0,
                "distance": "cosine",
                "selection": "official_greedy_maxmin",
                "min_visual_tokens": 1,
                "distance_chunk_size": 512,
            },
            metadata={
                "scope": "image_only",
                "placement": "post_merger_pre_llm",
                "conditioning": "none",
                "mechanism": "greedy_maxmin_cosine_feature_selection",
            },
        ),
        ModalityVariantDescriptor(
            name=QWEN3_5_VISIONZIP_VARIANT,
            algorithm="visionzip",
            target_modalities=frozenset({Modality.IMAGE}),
            preserved_modalities=preserved,
            coupling_policy=ModalityCouplingPolicy.INDEPENDENT,
            media_alignment=MediaAlignment.SPATIAL,
            token_unit=TokenUnit.FEATURE_TOKEN,
            config_defaults={
                "retain_ratio": 0.5,
                "target_count": None,
                "attention_chunk_size": 128,
            },
            metadata={
                "scope": "image_only",
                "placement": "post_merger_pre_llm",
                "attention_mode": "mean_received_no_cls",
                "key_position_encoding": "qwen3_5_rotary",
                "patch_statistics_mapping": "contiguous_spatial_merger_groups",
            },
        ),
        ModalityVariantDescriptor(
            name=QWEN3_5_OTPRUNE_VARIANT,
            algorithm="otprune",
            target_modalities=frozenset({Modality.IMAGE}),
            preserved_modalities=preserved,
            coupling_policy=ModalityCouplingPolicy.INDEPENDENT,
            media_alignment=MediaAlignment.SPATIAL,
            token_unit=TokenUnit.FEATURE_TOKEN,
            config_defaults={
                "retain_ratio": 0.5,
                "target_count": None,
                "gamma": 0.01,
            },
            metadata={
                "scope": "image_only",
                "placement": "post_merger_pre_llm",
                "paper_reference_ratio": 0.098,
                "selection": "greedy_logdet_cholesky",
                "objective": "ot_distribution_alignment",
            },
        ),
        ModalityVariantDescriptor(
            name=QWEN3_5_SAINT_VARIANT,
            algorithm="saint",
            target_modalities=frozenset({Modality.IMAGE}),
            preserved_modalities=preserved,
            coupling_policy=ModalityCouplingPolicy.INDEPENDENT,
            media_alignment=MediaAlignment.SPATIAL,
            token_unit=TokenUnit.FEATURE_TOKEN,
            config_defaults={
                "retain_ratio": 0.5,
                "target_count": None,
                "prune_level": "token",
                "group_size": 4,
                "similarity_threshold": 0.7,
                "neighbour_count": 5,
                "gamma": 10.0,
                "max_iterations": 32,
            },
            metadata={
                "scope": "image_only",
                "placement": "post_merger_pre_llm",
                "key_source": "native_final_vision_rotary_keys",
                "paper_stage": "projector_level_framework_adaptation",
            },
        ),
        ModalityVariantDescriptor(
            name=QWEN3_5_NUWA_VARIANT,
            algorithm="nuwa",
            target_modalities=frozenset({Modality.IMAGE}),
            conditioning_modalities=frozenset({Modality.TEXT}),
            preserved_modalities=preserved,
            coupling_policy=ModalityCouplingPolicy.INDEPENDENT,
            media_alignment=MediaAlignment.SPATIAL,
            token_unit=TokenUnit.FEATURE_TOKEN,
            config_defaults={
                "stage1_total_tokens": 108,
                "stage2_after_layer": 16,
                "stage2_retain_ratio": 0.5,
                "region_size": 2,
                "top_n_per_region": 3,
                "pillar_quantile": 0.75,
                "distance_squared_ratio": 0.26,
                "aggregation_self_mode": "released_add",
                "semantic_hidden_start": 16,
                "semantic_hidden_end": 25,
                "random_stage2": False,
                "random_seed": 44,
                "position_mode": "rpme",
                "single_image_only": True,
            },
            metadata={
                "scope": "image_only",
                "placement": "post_merger_then_decoder_layer_boundary",
                "paper_stage": "two_stage_framework_adaptation",
                "stage2_attention_layer_type": "native_full_attention_required",
            },
        ),
        ModalityVariantDescriptor(
            name=QWEN3_5_SPARSEVLM_VARIANT,
            algorithm="sparsevlm",
            target_modalities=frozenset({Modality.IMAGE}),
            conditioning_modalities=frozenset({Modality.TEXT}),
            preserved_modalities=preserved,
            coupling_policy=ModalityCouplingPolicy.INDEPENDENT,
            media_alignment=MediaAlignment.TOKEN,
            token_unit=TokenUnit.LLM_TOKEN,
            supported_rules=frozenset({
                "aggregator", "cluster_scorer", "cluster_selector", "grouper",
                "rater_scorer", "rater_selector", "recycle_selector", "scorer",
                "selector", "source_scorer",
            }),
            config_defaults={
                "variant": "official_v1",
                "prune_layers": (3, 7, 15),
                "schedule_keep_tokens": (300, 200, 110),
                "schedule_base_visual_tokens": 576,
                "scale_schedule_to_input": True,
                "recycle_ratio": 0.3,
                "cluster_divisor": 10,
                "position_mode": "contiguous",
                "single_image_only": True,
                "random_seed": 0,
            },
            metadata={
                "scope": "image_only",
                "placement": "decoder_layer_boundaries",
                "schedule_profile": "qwen3_8_full_attention_aligned",
                "paper_stage": "official_v1_kernel_with_backbone_layer_schedule",
            },
        ),
        ModalityVariantDescriptor(
            name=QWEN3_5_FINEPRUNER_VARIANT,
            algorithm="finepruner",
            target_modalities=frozenset({Modality.IMAGE}),
            conditioning_modalities=frozenset({Modality.TEXT}),
            preserved_modalities=preserved,
            coupling_policy=ModalityCouplingPolicy.INDEPENDENT,
            media_alignment=MediaAlignment.TOKEN,
            token_unit=TokenUnit.LLM_TOKEN,
            supported_rules=frozenset({
                "grouper", "iac_selector", "scorer", "selector", "head_selector",
            }),
            config_defaults={
                "iac_after_layers": (7, 14),
                "iac_hash_bits": (48, 41),
                "arp_center_after_layer": 20,
                "arp_layer_offsets": (-4, 0, 4),
                "arp_retain_ratio": 0.6,
                "target_visual_tokens": None,
                "head_ratio": 0.5,
                "random_seed": 126,
                "representative": "first",
                "score_query": "last_text_token",
                "position_policy": "contiguous",
                "single_image_only": True,
            },
            metadata={
                "scope": "image_only",
                "placement": "two_iac_boundaries_then_arp_boundary",
                "schedule_profile": "qwen3_8_full_attention_aligned",
                "paper_stage": "kernel_unchanged_backbone_layer_schedule",
            },
        ),
        ModalityVariantDescriptor(
            name=QWEN3_5_TRANSPRUNE_VARIANT,
            algorithm="transprune",
            target_modalities=frozenset({Modality.IMAGE}),
            conditioning_modalities=frozenset({Modality.TEXT}),
            preserved_modalities=preserved,
            coupling_policy=ModalityCouplingPolicy.INDEPENDENT,
            media_alignment=MediaAlignment.TOKEN,
            token_unit=TokenUnit.LLM_TOKEN,
            supported_rules=frozenset({"scorer", "selector", "transition_scorer"}),
            config_defaults={
                "accumulating_layers": (3, 4, 5, 6, 7, 8, 9, 10, 11),
                "prune_after_layers": (3, 7, 11),
                "retain_ratios": (0.875, 0.625, 0.125),
                "alpha": 0.5,
                "direction_rule": "release_1_minus_cosine",
                "ttv_softmax_scope": "release_full_sequence",
                "iga_source": "next_layer_post_image_text",
                "position_policy": "contiguous",
                "score_accumulation": "fp32",
                "single_image_only": True,
            },
            metadata={
                "scope": "image_only",
                "placement": "three_decoder_ttv_iga_boundaries",
                "schedule_profile": "qwen3_8_full_attention_aligned",
                "paper_stage": "kernel_unchanged_backbone_layer_schedule",
            },
        ),
        ModalityVariantDescriptor(
            name="saint.image.encoder_progressive_prune",
            algorithm="saint",
            target_modalities=frozenset({Modality.IMAGE}),
            preserved_modalities=preserved,
            coupling_policy=ModalityCouplingPolicy.INDEPENDENT,
            media_alignment=MediaAlignment.SPATIAL,
            token_unit=TokenUnit.ENCODER_TOKEN,
            supported_rules=frozenset({"scorer", "selector"}),
            config_defaults={
                "layer_indices": (15, 23),
                "keep_ratios": (0.5, 0.4),
                "prune_level": "token",
                "group_size": 4,
                "similarity_threshold": 0.7,
                "neighbour_count": 5,
                "gamma": 10.0,
                "max_iterations": 32,
            },
            metadata={
                "scope": "image_only",
                "permanent_encoder_compaction": True,
                "one_indexed_reference_layers": (16, 24),
                "spatial_merge_alignment": 4,
            },
        ),
        ModalityVariantDescriptor(
            name="saint.image.packed_feature_prune",
            algorithm="saint",
            target_modalities=frozenset({Modality.IMAGE}),
            preserved_modalities=preserved,
            coupling_policy=ModalityCouplingPolicy.INDEPENDENT,
            media_alignment=MediaAlignment.SPATIAL,
            token_unit=TokenUnit.FEATURE_TOKEN,
            supported_rules=frozenset({"scorer", "selector"}),
            config_defaults={
                "keep_ratio": 0.2,
                "prune_level": "token",
                "group_size": 1,
                "similarity_threshold": 0.7,
                "neighbour_count": 5,
                "gamma": 10.0,
                "max_iterations": 32,
            },
            metadata={
                "scope": "image_only",
                "adaptation": "post_encoder_feature_redundancy_prune",
                "reason": "preserve_anyres_or_deepstack_structural_taps",
                "structural_tokens_protected": True,
            },
        ),
        ModalityVariantDescriptor(
            name="visionzip.image",
            algorithm="visionzip",
            target_modalities=frozenset({Modality.IMAGE}),
            preserved_modalities=preserved,
            coupling_policy=ModalityCouplingPolicy.INDEPENDENT,
            media_alignment=MediaAlignment.SPATIAL,
            token_unit=TokenUnit.FEATURE_TOKEN,
            supported_rules=frozenset({
                "aggregator", "contextual_selector", "dominant_selector",
                "grouper", "scorer",
            }),
            config_defaults={
                "budget": 32,
                "dominant_weight": 27,
                "contextual_weight": 5,
            },
            metadata={
                "scope": "image_only",
                "text_agnostic": True,
                "reference_default_budget": 64,
                "framework_default_budget": 32,
                "budget_profile": "cross_backbone_32_token",
            },
        ),
        ModalityVariantDescriptor(
            name="bifocal.image",
            algorithm="bifocal",
            target_modalities=frozenset({Modality.IMAGE}),
            conditioning_modalities=frozenset({Modality.TEXT}),
            preserved_modalities=preserved,
            coupling_policy=ModalityCouplingPolicy.INDEPENDENT,
            media_alignment=MediaAlignment.SPATIAL,
            token_unit=TokenUnit.FEATURE_TOKEN,
            config_defaults={
                "s1_frac": 0.25,
                "block": "auto",
                "coarse_frac": 0.29,
                "k_frac": 0.75,
                "clsr": "1",
                "respool": True,
                "metric": "vit_attn",
                "metric_agg": "mass",
                "vit_block": 31,
                "nrm_sign": 1,
            },
            metadata={
                "scope": "image_only",
                "single_image_only": True,
                "query_source": "prompt_input_ids_last_vision_end_to_im_end",
            },
        ),
        ModalityVariantDescriptor(
            name="pyramid_drop.image",
            algorithm="pyramid_drop",
            target_modalities=frozenset({Modality.IMAGE}),
            conditioning_modalities=frozenset({Modality.TEXT}),
            preserved_modalities=preserved,
            coupling_policy=ModalityCouplingPolicy.INDEPENDENT,
            media_alignment=MediaAlignment.TOKEN,
            token_unit=TokenUnit.LLM_TOKEN,
            supported_rules=frozenset({"scorer", "selector", "source_scorer"}),
            config_defaults={"keep_ratio": 0.5, "num_stages": 4},
            metadata={
                "scope": "image_only",
                "conditioning": "last_instruction_token",
            },
        ),
        ModalityVariantDescriptor(
            name="feature_anchor_mean.image",
            algorithm="feature_anchor_mean",
            target_modalities=frozenset({Modality.IMAGE}),
            preserved_modalities=preserved,
            coupling_policy=ModalityCouplingPolicy.INDEPENDENT,
            media_alignment=MediaAlignment.SPATIAL,
            token_unit=TokenUnit.FEATURE_TOKEN,
            supported_rules=frozenset(),
            config_defaults={"target_count": 64},
            metadata={
                "scope": "image_only",
                "stage1_proxy": True,
                "partition": "feature_anchor",
                "aggregation": "fp32_mean",
                "position_policy": "rounded_native_mrope_cluster_centroid",
            },
        ),
        ModalityVariantDescriptor(
            name="grid_mean.image",
            algorithm="grid_mean",
            target_modalities=frozenset({Modality.IMAGE}),
            preserved_modalities=preserved,
            coupling_policy=ModalityCouplingPolicy.INDEPENDENT,
            media_alignment=MediaAlignment.SPATIAL,
            token_unit=TokenUnit.FEATURE_TOKEN,
            supported_rules=frozenset(),
            config_defaults={"target_count": 64},
            metadata={
                "scope": "image_only",
                "stage1_proxy": True,
                "partition": "grid",
                "aggregation": "fp32_mean",
                "position_policy": "rounded_native_mrope_cluster_centroid",
            },
        ),
        ModalityVariantDescriptor(
            name="prumerge.image",
            algorithm="prumerge",
            target_modalities=frozenset({Modality.IMAGE}),
            preserved_modalities=preserved,
            coupling_policy=ModalityCouplingPolicy.INDEPENDENT,
            media_alignment=MediaAlignment.SPATIAL,
            token_unit=TokenUnit.FEATURE_TOKEN,
            supported_rules=frozenset({"aggregator", "grouper", "scorer", "selector"}),
            config_defaults={
                "variant": "plus",
                "adaptive": True,
                "reduction_ratio": 0.125,
                "target_count": None,
                "iqr_multiplier": 1.5,
                "min_centers": 1,
                "cluster_size": 32,
            },
            metadata={
                "scope": "image_only",
                "importance_source": "mean_received_attention",
                "merge_mode": "official_unnormalized_attention_sum",
                "global_residual_token": True,
            },
        ),
        ModalityVariantDescriptor(
            name="gap.image",
            algorithm="gap",
            target_modalities=frozenset({Modality.IMAGE}),
            preserved_modalities=preserved,
            coupling_policy=ModalityCouplingPolicy.INDEPENDENT,
            media_alignment=MediaAlignment.SPATIAL,
            token_unit=TokenUnit.FEATURE_TOKEN,
            supported_rules=frozenset({"scorer", "selector"}),
            config_defaults={"retain_ratio": 0.5, "target_count": None},
            metadata={
                "scope": "image_only",
                "token_order": "original",
                "position_policy": "preserve_original",
            },
        ),
        ModalityVariantDescriptor(
            name="feather.image",
            algorithm="feather",
            target_modalities=frozenset({Modality.IMAGE}),
            conditioning_modalities=frozenset({Modality.TEXT}),
            preserved_modalities=preserved,
            coupling_policy=ModalityCouplingPolicy.INDEPENDENT,
            media_alignment=MediaAlignment.TOKEN,
            token_unit=TokenUnit.LLM_TOKEN,
            supported_rules=frozenset({"scorer", "selector"}),
            config_defaults={**_spec_defaults(FEATHERSpec), "single_image_only": True},
            metadata={
                "scope": "image_only",
                "score_source": "last_prompt_no_rope",
                "cache_policy": "compact_prefix",
                "position_policy": "preserve_original",
            },
        ),
        ModalityVariantDescriptor(
            name="ivc_prune.image",
            algorithm="ivc_prune",
            target_modalities=frozenset({Modality.IMAGE}),
            conditioning_modalities=frozenset({Modality.TEXT}),
            preserved_modalities=preserved,
            coupling_policy=ModalityCouplingPolicy.INDEPENDENT,
            media_alignment=MediaAlignment.TOKEN,
            token_unit=TokenUnit.LLM_TOKEN,
            supported_rules=frozenset({"coordinate_scorer", "selector", "semantic_scorer"}),
            config_defaults={**_spec_defaults(IVCPruneSpec), "single_image_only": True},
            metadata={
                "scope": "image_only",
                "coordinate_source": "backbone_native_rope_cos_sin",
                "cache_policy": "compact_prefix",
                "position_policy": "preserve_original",
            },
        ),
        ModalityVariantDescriptor(
            name="divprune.image",
            algorithm="divprune",
            target_modalities=frozenset({Modality.IMAGE}),
            preserved_modalities=preserved,
            coupling_policy=ModalityCouplingPolicy.INDEPENDENT,
            media_alignment=MediaAlignment.SPATIAL,
            token_unit=TokenUnit.FEATURE_TOKEN,
            supported_rules=frozenset({"scorer", "selector"}),
            config_defaults={
                "retain_ratio": 0.098,
                "target_count": None,
                "prune_layer": 0,
                "distance": "cosine",
                "selection": "official_greedy_maxmin",
                "min_visual_tokens": 1,
                "distance_chunk_size": 512,
            },
            metadata={
                "scope": "image_only",
                "conditioning": "none",
                "single_media_unit_kernel": True,
            },
        ),
        ModalityVariantDescriptor(
            name="aim.image",
            algorithm="aim",
            target_modalities=frozenset({Modality.IMAGE}),
            preserved_modalities=preserved,
            coupling_policy=ModalityCouplingPolicy.INDEPENDENT,
            media_alignment=MediaAlignment.SPATIAL,
            token_unit=TokenUnit.FEATURE_TOKEN,
            supported_rules=frozenset({"aggregator", "grouper", "scorer"}),
            config_defaults={
                "retain_ratio": 0.2,
                "target_count": None,
                "min_visual_tokens": 1,
            },
            metadata={
                "scope": "image_only",
                "conditioning": "none",
                "differentiable_merge": True,
                "per_image_merge": True,
            },
        ),
        ModalityVariantDescriptor(
            name="pact.image",
            algorithm="pact",
            target_modalities=frozenset({Modality.IMAGE}),
            preserved_modalities=preserved,
            coupling_policy=ModalityCouplingPolicy.INDEPENDENT,
            media_alignment=MediaAlignment.TOKEN,
            token_unit=TokenUnit.LLM_TOKEN,
            supported_rules=frozenset({"aggregator", "grouper", "scorer", "selector", "source_scorer"}),
            config_defaults=_spec_defaults(PACTSpec),
            metadata={
                "scope": "image_only",
                "single_decoder_boundary": True,
                "cache_policy": "layer_local",
                "position_policy": "preserve_original",
                "attention_calibration": "log_cluster_mass",
                "single_image_only": True,
            },
        ),
        ModalityVariantDescriptor(
            name="mmtok.image",
            algorithm="mmtok",
            target_modalities=frozenset({Modality.IMAGE}),
            conditioning_modalities=frozenset({Modality.TEXT}),
            preserved_modalities=preserved,
            coupling_policy=ModalityCouplingPolicy.INDEPENDENT,
            media_alignment=MediaAlignment.SPATIAL,
            token_unit=TokenUnit.FEATURE_TOKEN,
            supported_rules=frozenset({"scorer", "selector"}),
            config_defaults={
                "alpha": 0.5,
                "text_temperature": 0.02,
                "vision_temperature": 0.2,
                "min_visual_tokens": 1,
            },
            metadata={
                "scope": "image_only",
                "conditioning": "prompt_input_embeddings",
                "vision_source": "pre_projector_or_pre_merger_visual_features",
                "per_image_or_crop_selection": True,
                "budget_defaults_by_backbone": {
                    "llava_onevision": {"tokens_per_crop": 32},
                    "qwen25_vl": {"retain_ratio": 0.2, "text_temperature": 0.01},
                    "qwen3_vl": {"retain_ratio": 0.2},
                },
            },
        ),
        ModalityVariantDescriptor(
            name="vispruner.image",
            algorithm="vispruner",
            target_modalities=frozenset({Modality.IMAGE}),
            preserved_modalities=preserved,
            coupling_policy=ModalityCouplingPolicy.INDEPENDENT,
            media_alignment=MediaAlignment.SPATIAL,
            token_unit=TokenUnit.FEATURE_TOKEN,
            supported_rules=frozenset({"scorer", "selector"}),
            config_defaults={
                "target_count": 32,
                "important_ratio": 0.5,
                "removal_step": 8,
                "min_visual_tokens": 1,
            },
            metadata={
                "scope": "image_only",
                "conditioning": "final_visual_encoder_attention",
                "selection_feature_source": "pre_projector_or_pre_merger",
                "per_image_or_crop_selection": True,
                "budget_defaults_by_backbone": {
                    "llava_onevision": {"tokens_per_crop": 32},
                    "qwen25_vl": {"target_count": 32},
                    "qwen3_vl": {"target_count": 32},
                },
            },
        ),
        ModalityVariantDescriptor(
            name="zoo_prune.image",
            algorithm="zoo_prune",
            target_modalities=frozenset({Modality.IMAGE}),
            preserved_modalities=preserved,
            coupling_policy=ModalityCouplingPolicy.INDEPENDENT,
            media_alignment=MediaAlignment.SPATIAL,
            token_unit=TokenUnit.FEATURE_TOKEN,
            supported_rules=frozenset({"scorer", "selector"}),
            config_defaults={
                "num_directions": 64,
                "step_size": 0.01,
                "projector_proxy": "low_rank",
                "low_rank_rank": 128,
                "low_rank_solver": "exact",
                "direction_batch_size": 4,
                "seed": 0,
                "min_visual_tokens": 1,
            },
            metadata={
                "scope": "image_only",
                "conditioning": "none",
                "projector_input": "native_pre_projector_or_merger_mlp_input",
                "budget_default": {"retention_ratio": 0.1112},
                "selection_scope": "per_crop_llava_per_image_qwen",
                "sensitivity": "centered_zeroth_order_projector_response",
                "selection": "multiplicative_sensitivity_diversity",
                "training_free": True,
                "single_image_only": True,
            },
        ),
        ModalityVariantDescriptor(
            name="fastv.image",
            algorithm="fastv",
            target_modalities=frozenset({Modality.IMAGE}),
            conditioning_modalities=frozenset({Modality.TEXT}),
            preserved_modalities=preserved,
            coupling_policy=ModalityCouplingPolicy.INDEPENDENT,
            media_alignment=MediaAlignment.TOKEN,
            token_unit=TokenUnit.LLM_TOKEN,
            supported_rules=frozenset({"scorer", "selector"}),
            config_defaults=_spec_defaults(FastVSpec),
            metadata={
                "scope": "image_only",
                "conditioning": "last_prompt_token_attention",
                "single_decoder_boundary": True,
            },
        ),
        ModalityVariantDescriptor(
            name="sgl.image",
            algorithm="sgl",
            target_modalities=frozenset({Modality.IMAGE}),
            conditioning_modalities=frozenset({Modality.TEXT}),
            preserved_modalities=preserved,
            coupling_policy=ModalityCouplingPolicy.INDEPENDENT,
            media_alignment=MediaAlignment.TOKEN,
            token_unit=TokenUnit.LLM_TOKEN,
            supported_rules=frozenset({"scorer", "selector"}),
            config_defaults=_spec_defaults(SGLSpec),
            metadata={
                "scope": "image_generate_only",
                "conditioning": "small_vlm_prompt_and_generated_attention",
                "single_decoder_boundary": True,
                "single_image_only": True,
                "guide_model_path_required": True,
                "released_executable_scope": "sgp_only",
            },
        ),
        ModalityVariantDescriptor(
            name="dart.image",
            algorithm="dart",
            target_modalities=frozenset({Modality.IMAGE}),
            conditioning_modalities=frozenset({Modality.TEXT}),
            preserved_modalities=preserved,
            coupling_policy=ModalityCouplingPolicy.INDEPENDENT,
            media_alignment=MediaAlignment.TOKEN,
            token_unit=TokenUnit.LLM_TOKEN,
            supported_rules=frozenset({"diversity_scorer", "scorer", "selector"}),
            config_defaults=_spec_defaults(DARTSpec),
            metadata={
                "scope": "image_only",
                "conditioning": "post_image_text_pivots",
                "single_decoder_boundary": True,
                "single_image_only": True,
            },
        ),
        ModalityVariantDescriptor(
            name="d2pruner.image",
            algorithm="d2pruner",
            target_modalities=frozenset({Modality.IMAGE}),
            conditioning_modalities=frozenset({Modality.TEXT}),
            preserved_modalities=preserved,
            coupling_policy=ModalityCouplingPolicy.INDEPENDENT,
            media_alignment=MediaAlignment.TOKEN,
            token_unit=TokenUnit.LLM_TOKEN,
            supported_rules=frozenset({"scorer", "selector", "source_scorer"}),
            config_defaults=_spec_defaults(D2PrunerSpec, exclude=("bias_prior_path",)),
            metadata={
                "scope": "image_only",
                "conditioning": "debiased_last_text_token_attention",
                "single_decoder_boundary": True,
                "requires_calibrated_bias_prior": True,
                "bias_prior_path_required": True,
                "bias_prior_identity": "checkpoint_and_scorer_layer",
                "single_image_only": True,
            },
        ),
        ModalityVariantDescriptor(
            name="sparsevlm.image",
            algorithm="sparsevlm",
            target_modalities=frozenset({Modality.IMAGE}),
            conditioning_modalities=frozenset({Modality.TEXT}),
            preserved_modalities=preserved,
            coupling_policy=ModalityCouplingPolicy.INDEPENDENT,
            media_alignment=MediaAlignment.TOKEN,
            token_unit=TokenUnit.LLM_TOKEN,
            supported_rules=frozenset({
                "aggregator", "cluster_scorer", "cluster_selector",
                "grouper", "rater_scorer", "rater_selector",
                "recycle_selector", "scorer", "selector",
                "source_scorer",
            }),
            config_defaults=_spec_defaults(SparseVLMSpec),
            metadata={
                "scope": "image_only",
                "conditioning": "post_image_text_raters",
                "multistage_decoder_boundaries": True,
                "cache_policy": "compact_prefix",
                "cache_transform": "mean_merge",
                "position_policy": "contiguous",
                "single_image_only": True,
            },
        ),
        ModalityVariantDescriptor(
            name="finepruner.image",
            algorithm="finepruner",
            target_modalities=frozenset({Modality.IMAGE}),
            conditioning_modalities=frozenset({Modality.TEXT}),
            preserved_modalities=preserved,
            coupling_policy=ModalityCouplingPolicy.INDEPENDENT,
            media_alignment=MediaAlignment.TOKEN,
            token_unit=TokenUnit.LLM_TOKEN,
            supported_rules=frozenset({"grouper", "head_selector", "iac_selector", "scorer", "selector"}),
            config_defaults=_spec_defaults(FinePrunerSpec),
            metadata={
                "scope": "image_only",
                "conditioning": "layer_averaged_low_kl_head_attention",
                "multistage_decoder_boundaries": True,
                "cache_policy": "layer_local",
                "cache_transform": "none",
                "position_policy": "contiguous",
                "single_image_only": True,
            },
        ),
        ModalityVariantDescriptor(
            name="btp.image",
            algorithm="btp",
            target_modalities=frozenset({Modality.IMAGE}),
            conditioning_modalities=frozenset({Modality.TEXT}),
            preserved_modalities=preserved,
            coupling_policy=ModalityCouplingPolicy.INDEPENDENT,
            media_alignment=MediaAlignment.TOKEN,
            token_unit=TokenUnit.LLM_TOKEN,
            supported_rules=frozenset({"scorer", "selector", "source_scorer"}),
            config_defaults=_spec_defaults(BalancedTokenPruningSpec),
            metadata={
                "scope": "image_only",
                "conditioning": "last_text_query_head_mean_attention",
                "multistage_decoder_boundaries": True,
                "cache_policy": "compact_prefix",
                "position_policy": "preserve_original",
                "single_image_only": True,
                "final_stage": "remove_remaining_visual_span",
            },
        ),
        ModalityVariantDescriptor(
            name="halfv.image",
            algorithm="halfv",
            target_modalities=frozenset({Modality.IMAGE}),
            conditioning_modalities=frozenset({Modality.TEXT}),
            preserved_modalities=preserved,
            coupling_policy=ModalityCouplingPolicy.INDEPENDENT,
            media_alignment=MediaAlignment.TOKEN,
            token_unit=TokenUnit.LLM_TOKEN,
            supported_rules=frozenset({"scorer", "selector", "source_scorer"}),
            config_defaults=_spec_defaults(HalfVSpec),
            metadata={
                "scope": "image_only",
                "conditioning": "last_text_query_head_mean_attention",
                "multistage_decoder_boundaries": True,
                "cache_policy": "compact_prefix",
                "cache_transform": "drop",
                "position_policy": "preserve_original",
                "single_image_only": True,
                "ivr_score_rope": False,
                "ssr_score_rope": True,
            },
        ),
        ModalityVariantDescriptor(
            name="transprune.image",
            algorithm="transprune",
            target_modalities=frozenset({Modality.IMAGE}),
            conditioning_modalities=frozenset({Modality.TEXT}),
            preserved_modalities=preserved,
            coupling_policy=ModalityCouplingPolicy.INDEPENDENT,
            media_alignment=MediaAlignment.TOKEN,
            token_unit=TokenUnit.LLM_TOKEN,
            supported_rules=frozenset({"scorer", "selector", "transition_scorer"}),
            config_defaults=_spec_defaults(TransPruneSpec),
            metadata={
                "scope": "image_only",
                "conditioning": "accumulated_ttv_plus_next_layer_text_attention",
                "multistage_decoder_boundaries": True,
                "cache_policy": "layer_local",
                "cache_transform": "none",
                "position_policy": "contiguous",
                "single_image_only": True,
                "module_capture": ("attention", "ffn"),
            },
        ),
        ModalityVariantDescriptor(
            name="nuwa.image",
            algorithm="nuwa",
            target_modalities=frozenset({Modality.IMAGE}),
            conditioning_modalities=frozenset({Modality.TEXT}),
            preserved_modalities=preserved,
            coupling_policy=ModalityCouplingPolicy.INDEPENDENT,
            media_alignment=MediaAlignment.TOKEN,
            token_unit=TokenUnit.LLM_TOKEN,
            supported_rules=frozenset({
                "aggregation_weights", "regional_selector", "stage1_scorer",
                "stage1_selector", "stage2_scorer", "stage2_selector",
                "stage2_source_scorer",
            }),
            config_defaults={
                "stage1_total_tokens": 582,
                "stage2_after_layer": 16,
                "stage2_retain_ratio": 0.125,
                "stage2_score": "released_last_text_attention",
                "region_size": 2,
                "top_n_per_region": 3,
                "pillar_quantile": 0.75,
                "distance_squared_ratio": 0.26,
                "aggregation_self_mode": "released_add",
                "semantic_hidden_start": 16,
                "semantic_hidden_end": 25,
                "random_stage2": False,
                "random_seed": 44,
                "position_mode": "rpme",
                "single_image_only": True,
            },
            metadata={
                "scope": "image_only",
                "conditioning": (
                    "visual_attention_key_norm_semantic_hidden_then_"
                    "last_prompt_attention"
                ),
                "composite_boundaries": (
                    "post_visual_projector_or_merger",
                    "decoder_before_layer_16",
                ),
                "cache_policy": "compact_prefix",
                "cache_transform": "drop",
                "position_policy": "rpme_native_sparse_mrope",
                "single_image_only": True,
            },
        ),
        ModalityVariantDescriptor(
            name="restore.image",
            algorithm="restore",
            target_modalities=frozenset({Modality.IMAGE}),
            preserved_modalities=preserved,
            coupling_policy=ModalityCouplingPolicy.INDEPENDENT,
            media_alignment=MediaAlignment.TOKEN,
            token_unit=TokenUnit.LLM_TOKEN,
            supported_rules=frozenset({
                "aggregator", "anchor_scorer", "anchor_selector",
                "calibration_bias", "grouper", "pristine_scorer",
                "pristine_selector",
            }),
            config_defaults={
                "target_count": 64,
                "pruning_ratio": 0.5,
                "important_ratio": 0.5,
                "removal_step": 8,
                "anchor_mask_mode": "released",
                "position_strategy": "median",
                "calibration_constant": 2.0,
                "first_layer_size": "uniform",
                "min_visual_tokens": 2,
            },
            metadata={
                "scope": "image_only",
                "conditioning": "final_visual_attention_then_decoder_positions",
                "composite_boundaries": (
                    "post_visual_projector_or_merger",
                    "every_decoder_attention",
                ),
                "cache_policy": "stable",
                "cache_transform": "none",
                "position_policy": "real_source_representatives",
                "attention_calibration": "token_mass_plus_rope_distance",
                "single_image_only": True,
            },
        ),
    )


def _target(name: str, placement: Placement, order: int, *handles: str,
        **metadata: Any) -> dict[str, Any]:
    """Logical target details; the physical site belongs to its native contract."""
    return dict(name=name, placement=placement, order=order,
        target_modalities=frozenset({Modality.IMAGE}),
        required_handles=frozenset(handles), metadata=metadata)


def _image_binding(name: str, *, targets, supported_phases, legalizer,
        strategy_factory=None, position_resolver=_resolve_image_position) -> BackbonePlacementBinding:
    """Join logical targets to the same physical facts consumed by the executor."""
    backbone = name.rsplit(".", 1)[1]
    contract = _IMAGE_BINDING_CONTRACTS[backbone][name]
    sites = contract.sites if backbone == "llava_onevision" else contract.target_sites
    return BackbonePlacementBinding(
        name=contract.binding, backbone=backbone, variant=contract.variant,
        targets=tuple(BindingTarget(target_site=site, **target)
            for site, target in zip(sites, targets, strict=True)),
        supported_phases=supported_phases,
        position_resolver=position_resolver,
        executor_factory=_IMAGE_EXECUTOR_FACTORIES[backbone],
        strategy_factory=strategy_factory, legalizer=legalizer,
        conflicts=_image_method_conflicts(contract.variant.partition(".")[0], backbone),
        metadata={"image_only": contract.image_only, **contract.metadata},
    )


def _bindings() -> tuple[BackbonePlacementBinding, ...]:
    from mllm_comp.compressors.image.decoder_selection import (
        FastVDecoderMethod, PyramidDecoderMethod, BTPDecoderMethod, HalfVDecoderMethod, TransPruneDecoderMethod, SparseVLMDecoderMethod,
        FEATHERDecoderMethod, IVCPruneDecoderMethod, DARTDecoderMethod, D2PrunerDecoderMethod, FinePrunerDecoderMethod, SparseVLMDecoderMethod, PACTDecoderMethod, SGLDecoderMethod,
    )

    phases = frozenset({ExecutionPhase.FORWARD, ExecutionPhase.GENERATE})
    return (
        _image_binding(
            name="tome.image.llava_onevision",
            targets=(
                _target(
                    "encoder_merge",
                    Placement.ENCODER_LEVEL,
                    0,
                    "vision_encoder.layers",
                ),
                _target(
                    "dense_restore",
                    Placement.ENCODER_LEVEL,
                    1,
                    "vision_encoder",
                    paper_compression_stage=False,
                    component_role="structural_legalizer",
                ),
                _target(
                    "permanent_output_merge",
                    Placement.PROJECTOR_LEVEL,
                    2,
                    "vision_projector",
                    "llm",
                    paper_compression_stage=False,
                    component_role="framework_output_compaction",
                ),
            ),
            supported_phases=phases,
            legalizer="llava_onevision.tome_structural_restore_then_prompt_compact",
        ),
        _image_binding(
            name="tome.image.qwen25_vl",
            targets=(
                _target(
                    "encoder_merge",
                    Placement.ENCODER_LEVEL,
                    0,
                    "vision_encoder.blocks",
                ),
                _target(
                    "dense_restore",
                    Placement.ENCODER_LEVEL,
                    1,
                    "vision_projector",
                    paper_compression_stage=False,
                    component_role="structural_legalizer",
                ),
                _target(
                    "permanent_output_merge",
                    Placement.PROJECTOR_LEVEL,
                    2,
                    "vision_projector",
                    "llm",
                    paper_compression_stage=False,
                    component_role="framework_output_compaction",
                ),
            ),
            supported_phases=phases,
            legalizer="qwen25_vl.tome_structural_restore_then_prompt_compact",
        ),
        _image_binding(
            name="tome.image.qwen3_vl",
            targets=(
                _target(
                    "encoder_merge",
                    Placement.ENCODER_LEVEL,
                    0,
                    "vision_encoder.blocks",
                ),
                _target(
                    "deepstack_dense_restore",
                    Placement.ENCODER_LEVEL,
                    1,
                    "vision_deepstack_projectors",
                    paper_compression_stage=False,
                    component_role="structural_legalizer",
                ),
                _target(
                    "final_dense_restore",
                    Placement.ENCODER_LEVEL,
                    2,
                    "vision_projector",
                    paper_compression_stage=False,
                    component_role="structural_legalizer",
                ),
                _target(
                    "permanent_output_merge",
                    Placement.PROJECTOR_LEVEL,
                    3,
                    "vision_projector",
                    "vision_deepstack_projectors",
                    "llm",
                    paper_compression_stage=False,
                    component_role="framework_output_compaction",
                ),
            ),
            supported_phases=phases,
            legalizer="qwen3_vl.tome_structural_restore_then_prompt_compact",
        ),
        _image_binding(
            name=QWEN3_5_TOME_BINDING,
            targets=(
                _target(
                    "post_merger_feature_merge",
                    Placement.PROJECTOR_LEVEL,
                    0,
                    "image_features_pre_llm_scatter",
                    "llm",
                    framework_adaptation=True,
                ),
            ),
            supported_phases=frozenset({ExecutionPhase.GENERATE}),
            legalizer="qwen3_5.compact_image_placeholder_and_mrope",
            position_resolver=None,
        ),
        _image_binding(
            name=QWEN3_5_DIVPRUNE_BINDING,
            targets=(
                _target(
                    "post_merger_feature_selection",
                    Placement.PROJECTOR_LEVEL,
                    0,
                    "image_features_pre_llm_scatter",
                    "llm",
                    framework_adaptation=True,
                ),
            ),
            supported_phases=frozenset({ExecutionPhase.GENERATE}),
            legalizer="qwen3_5.compact_image_placeholder_and_mrope",
            position_resolver=None,
        ),
        _image_binding(
            name=QWEN3_5_VISIONZIP_BINDING,
            targets=(
                _target(
                    "post_merger_attention_guided_merge",
                    Placement.PROJECTOR_LEVEL,
                    0,
                    "image_features_pre_llm_scatter",
                    "vision_encoder",
                    "llm",
                    framework_adaptation=True,
                ),
            ),
            supported_phases=frozenset({ExecutionPhase.GENERATE}),
            legalizer="qwen3_5.compact_image_placeholder_and_mrope",
            position_resolver=None,
        ),
        _image_binding(
            name=QWEN3_5_OTPRUNE_BINDING,
            targets=(
                _target(
                    "post_merger_ot_distribution_aligned_selection",
                    Placement.PROJECTOR_LEVEL,
                    0,
                    "image_features_pre_llm_scatter",
                    "llm",
                    framework_adaptation=True,
                ),
            ),
            supported_phases=frozenset({ExecutionPhase.GENERATE}),
            legalizer="qwen3_5.compact_image_placeholder_and_mrope",
            position_resolver=None,
        ),
        _image_binding(
            name=QWEN3_5_PRUMERGE_BINDING,
            targets=(
                _target(
                    "post_merger_attention_guided_plus_merge",
                    Placement.PROJECTOR_LEVEL,
                    0,
                    "image_features_pre_llm_scatter",
                    "vision_encoder",
                    "llm",
                    framework_adaptation=True,
                ),
            ),
            supported_phases=frozenset({ExecutionPhase.GENERATE}),
            legalizer="qwen3_5.compact_image_placeholder_and_mrope",
            position_resolver=None,
        ),
        _image_binding(
            name=QWEN3_5_MMTOK_BINDING,
            targets=(
                _target(
                    "post_merger_text_vision_coverage_selection",
                    Placement.PROJECTOR_LEVEL,
                    0,
                    "image_features_pre_llm_scatter",
                    "vision_encoder",
                    "llm",
                    framework_adaptation=True,
                ),
            ),
            supported_phases=frozenset({ExecutionPhase.GENERATE}),
            legalizer="qwen3_5.compact_image_placeholder_and_mrope",
            position_resolver=None,
        ),
        _image_binding(
            name=QWEN3_5_GAP_BINDING,
            targets=(
                _target(
                    "post_merger_received_attention_selection",
                    Placement.PROJECTOR_LEVEL,
                    0,
                    "image_features_pre_llm_scatter",
                    "vision_encoder",
                    "llm",
                    framework_adaptation=True,
                ),
            ),
            supported_phases=frozenset({ExecutionPhase.GENERATE}),
            legalizer="qwen3_5.compact_image_placeholder_and_mrope",
            position_resolver=None,
        ),
        _image_binding(
            name=QWEN3_5_VISPRUNER_BINDING,
            targets=(
                _target(
                    "post_merger_visual_cue_and_diversity_selection",
                    Placement.PROJECTOR_LEVEL,
                    0,
                    "image_features_pre_llm_scatter",
                    "vision_encoder",
                    "llm",
                    framework_adaptation=True,
                ),
            ),
            supported_phases=frozenset({ExecutionPhase.GENERATE}),
            legalizer="qwen3_5.compact_image_placeholder_and_mrope",
            position_resolver=None,
        ),
        _image_binding(
            name=QWEN3_5_AIM_BINDING,
            targets=(
                _target(
                    "post_merger_bipartite_feature_merge",
                    Placement.PROJECTOR_LEVEL,
                    0,
                    "image_features_pre_llm_scatter",
                    "llm",
                    framework_adaptation=True,
                ),
            ),
            supported_phases=frozenset({ExecutionPhase.GENERATE}),
            legalizer="qwen3_5.compact_image_placeholder_and_mrope",
            position_resolver=None,
        ),
        _image_binding(
            name=QWEN3_5_RESTORE_BINDING,
            targets=(
                _target(
                    "post_merger_pristine_selection_and_anchor_merge",
                    Placement.PROJECTOR_LEVEL,
                    0,
                    "image_features_pre_llm_scatter",
                    "vision_encoder",
                    "llm",
                    framework_adaptation=True,
                ),
            ),
            supported_phases=frozenset({ExecutionPhase.GENERATE}),
            legalizer="qwen3_5.compact_image_placeholder_and_mrope",
            position_resolver=None,
        ),
        _image_binding(
            name=QWEN3_5_BIFOCAL_BINDING,
            targets=(
                _target(
                    "post_merger_bifocal_attention_spatial_selection",
                    Placement.PROJECTOR_LEVEL,
                    0,
                    "image_features_pre_llm_scatter",
                    "vision_encoder",
                    "llm",
                    framework_adaptation=True,
                ),
            ),
            supported_phases=frozenset({ExecutionPhase.GENERATE}),
            legalizer="qwen3_5.compact_image_placeholder_and_mrope",
            position_resolver=None,
        ),
        _image_binding(
            name=QWEN3_5_ZOO_PRUNE_BINDING,
            targets=(
                _target(
                    "post_merger_zoo_projector_sensitivity_selection",
                    Placement.PROJECTOR_LEVEL,
                    0,
                    "image_features_pre_llm_scatter",
                    "vision_encoder.merger",
                    "llm",
                    framework_adaptation=True,
                ),
            ),
            supported_phases=frozenset({ExecutionPhase.GENERATE}),
            legalizer="qwen3_5.compact_image_placeholder_and_mrope",
            position_resolver=None,
        ),
        _image_binding(
            name=QWEN3_5_SAINT_BINDING,
            targets=(
                _target(
                    "post_merger_saint_native_key_selection",
                    Placement.PROJECTOR_LEVEL,
                    0,
                    "image_features_pre_llm_scatter",
                    "vision_encoder",
                    "llm",
                    framework_adaptation=True,
                ),
            ),
            supported_phases=frozenset({ExecutionPhase.GENERATE}),
            legalizer="qwen3_5.compact_image_placeholder_and_mrope",
            position_resolver=None,
        ),
        _image_binding(
            name=QWEN3_5_NUWA_BINDING,
            targets=(
                _target(
                    "nuwa_stage1_region_attention_aggregation",
                    Placement.PROJECTOR_LEVEL,
                    0,
                    "vision_encoder",
                    "image_features_pre_llm_scatter",
                    "llm",
                    framework_adaptation=True,
                ),
                _target(
                    "nuwa_stage2_last_text_attention_selection",
                    Placement.LLM_LEVEL,
                    1,
                    "language_model.layers",
                    "llm",
                    framework_adaptation=True,
                ),
            ),
            supported_phases=frozenset({ExecutionPhase.GENERATE}),
            legalizer="qwen3_5.nuwa_sparse_mrope_no_cache",
            position_resolver=None,
        ),
        _image_binding(
            name=QWEN3_5_SPARSEVLM_BINDING,
            targets=tuple(
                _target(
                    f"sparsevlm_rater_attention_stage_{index}",
                    Placement.LLM_LEVEL,
                    layer + 1,
                    "language_model.layers",
                    "llm",
                    framework_adaptation=True,
                    score_layer=layer,
                )
                for index, layer in enumerate((3, 7, 15))
            ),
            supported_phases=frozenset({ExecutionPhase.GENERATE}),
            strategy_factory=SparseVLMDecoderMethod,
            legalizer="qwen3_5.decoder_sparsevlm_multistage_sparse_mrope_no_cache",
            position_resolver=None,
        ),
        _image_binding(
            name=QWEN3_5_FINEPRUNER_BINDING,
            targets=(
                _target("finepruner_iac_boundary_after_7", Placement.LLM_LEVEL, 7,
                        "language_model.layers", "llm", framework_adaptation=True),
                _target("finepruner_iac_boundary_after_14", Placement.LLM_LEVEL, 14,
                        "language_model.layers", "llm", framework_adaptation=True),
                _target("finepruner_arp_boundary_after_24", Placement.LLM_LEVEL, 24,
                        "language_model.layers", "llm", framework_adaptation=True,
                        score_layers=(15, 19, 23)),
            ),
            supported_phases=frozenset({ExecutionPhase.GENERATE}),
            strategy_factory=FinePrunerDecoderMethod,
            legalizer="qwen3_5.decoder_finepruner_multistage_sparse_mrope_no_cache",
            position_resolver=None,
        ),
        _image_binding(
            name=QWEN3_5_TRANSPRUNE_BINDING,
            targets=(
                _target("transprune_ttv_iga_stage_1", Placement.LLM_LEVEL, 4,
                        "language_model.layers",
                        framework_adaptation=True, score_layer=3),
                _target("transprune_ttv_iga_stage_2", Placement.LLM_LEVEL, 8,
                        "language_model.layers",
                        framework_adaptation=True, score_layer=7),
                _target("transprune_ttv_iga_stage_3", Placement.LLM_LEVEL, 12,
                        "language_model.layers",
                        framework_adaptation=True, score_layer=11),
            ),
            supported_phases=frozenset({ExecutionPhase.GENERATE}),
            strategy_factory=TransPruneDecoderMethod,
            legalizer="qwen3_5.decoder_transprune_multistage_sparse_mrope_no_cache",
            position_resolver=None,
        ),
        _image_binding(
            name=QWEN3_5_FASTV_BINDING,
            targets=(
                _target(
                    "decoder_fastv_prune_boundary",
                    Placement.LLM_LEVEL,
                    16,
                    "language_model.layers",
                    "llm",
                    framework_adaptation=True,
                ),
            ),
            supported_phases=frozenset({ExecutionPhase.GENERATE}),
            strategy_factory=FastVDecoderMethod,
            legalizer="qwen3_5.decoder_sparse_mrope_no_cache",
            position_resolver=None,
        ),
        _image_binding(
            name=QWEN3_5_PYRAMID_DROP_BINDING,
            targets=(
                _target(
                    "decoder_pyramid_drop_stages",
                    Placement.LLM_LEVEL,
                    16,
                    "language_model.layers",
                    "llm",
                    framework_adaptation=True,
                ),
            ),
            supported_phases=frozenset({ExecutionPhase.GENERATE}),
            strategy_factory=PyramidDecoderMethod,
            legalizer="qwen3_5.decoder_multistage_sparse_mrope_no_cache",
            position_resolver=None,
        ),
        _image_binding(
            name=QWEN3_5_DART_BINDING,
            targets=(
                _target(
                    "decoder_dart_pivot_merge_boundary",
                    Placement.LLM_LEVEL,
                    16,
                    "language_model.layers",
                    "llm",
                    framework_adaptation=True,
                ),
            ),
            supported_phases=frozenset({ExecutionPhase.GENERATE}),
            strategy_factory=DARTDecoderMethod,
            legalizer="qwen3_5.decoder_dart_sparse_mrope_no_cache",
            position_resolver=None,
        ),
        _image_binding(
            name=QWEN3_5_IVC_PRUNE_BINDING,
            targets=(
                _target(
                    "decoder_ivc_coordinate_semantic_prune_boundary",
                    Placement.LLM_LEVEL,
                    16,
                    "language_model.layers",
                    "llm",
                    framework_adaptation=True,
                ),
            ),
            supported_phases=frozenset({ExecutionPhase.GENERATE}),
            strategy_factory=IVCPruneDecoderMethod,
            legalizer="qwen3_5.decoder_ivc_sparse_mrope_no_cache",
            position_resolver=None,
        ),
        _image_binding(
            name=QWEN3_5_FEATHER_BINDING,
            targets=(
                _target(
                    "decoder_feather_spatial_uniform_stages",
                    Placement.LLM_LEVEL,
                    16,
                    "language_model.layers",
                    "llm",
                    framework_adaptation=True,
                ),
            ),
            supported_phases=frozenset({ExecutionPhase.GENERATE}),
            strategy_factory=FEATHERDecoderMethod,
            legalizer="qwen3_5.decoder_feather_sparse_mrope_no_cache",
            position_resolver=None,
        ),
        _image_binding(
            name=QWEN3_5_BTP_BINDING,
            targets=(_target("decoder_btp_multistage_pruning", Placement.LLM_LEVEL, 16, "language_model.layers", "llm", framework_adaptation=True),),
            supported_phases=frozenset({ExecutionPhase.GENERATE}),
            strategy_factory=BTPDecoderMethod,
            legalizer="qwen3_5.decoder_btp_sparse_mrope_no_cache",
            position_resolver=None,
        ),
        _image_binding(
            name=QWEN3_5_HALFV_BINDING,
            targets=(_target("decoder_halfv_ivr_ssr", Placement.LLM_LEVEL, 16, "language_model.layers", "llm", framework_adaptation=True),),
            supported_phases=frozenset({ExecutionPhase.GENERATE}),
            strategy_factory=HalfVDecoderMethod,
            legalizer="qwen3_5.decoder_halfv_sparse_mrope_no_cache",
            position_resolver=None,
        ),
        _image_binding(
            name=QWEN3_5_SGL_BINDING,
            targets=(_target("decoder_guide_prune_boundary", Placement.LLM_LEVEL, 16, "language_model.layers", "llm", framework_adaptation=True),),
            supported_phases=frozenset({ExecutionPhase.GENERATE}),
            strategy_factory=SGLDecoderMethod,
            legalizer="qwen3_5.sgl_guide_sparse_mrope_no_cache",
            position_resolver=None,
        ),
        _image_binding(
            name=QWEN3_5_D2PRUNER_BINDING,
            targets=(_target("decoder_calibrated_spatial_prune_boundary", Placement.LLM_LEVEL, 16, "language_model.layers", "llm", framework_adaptation=True),),
            supported_phases=frozenset({ExecutionPhase.GENERATE}),
            strategy_factory=D2PrunerDecoderMethod,
            legalizer="qwen3_5.d2pruner_calibrated_sparse_mrope_no_cache",
            position_resolver=None,
        ),
        _image_binding(
            name="saint.image.qwen25_vl",
            targets=(
                _target(
                    "encoder_progressive_prune",
                    Placement.ENCODER_LEVEL,
                    0,
                    "vision_encoder.blocks",
                    "vision_projector",
                    "llm",
                ),
            ),
            supported_phases=frozenset({ExecutionPhase.FORWARD}),
            legalizer="qwen25_vl.saint_progressive_patch_and_sparse_mrope",
        ),
        _image_binding(
            name="saint.image.llava_onevision",
            targets=(
                _target(
                    "packed_image_features",
                    Placement.PROJECTOR_LEVEL,
                    0,
                    "vision_projector",
                    "llm",
                    paper_compression_stage=False,
                    component_role="backbone_pre_llm_adaptation",
                ),
            ),
            supported_phases=frozenset({ExecutionPhase.FORWARD}),
            legalizer="llava_onevision.saint_packed_image_layout",
        ),
        _image_binding(
            name="saint.image.qwen3_vl",
            targets=(
                _target(
                    "main_image_features",
                    Placement.PROJECTOR_LEVEL,
                    0,
                    "vision_projector",
                    "llm",
                    paper_compression_stage=False,
                    component_role="backbone_pre_llm_adaptation",
                ),
                _target(
                    "deepstack_image_features",
                    Placement.PROJECTOR_LEVEL,
                    1,
                    "vision_deepstack_projectors",
                    "llm",
                    paper_compression_stage=False,
                    component_role="backbone_pre_llm_adaptation",
                ),
            ),
            supported_phases=frozenset({ExecutionPhase.FORWARD}),
            legalizer="qwen3_vl.saint_shared_deepstack_sparse_mrope",
        ),
        _image_binding(
            name="aim.image.qwen25_vl",
            targets=(
                _target(
                    "merged_image_features",
                    Placement.PROJECTOR_LEVEL,
                    0,
                    "vision_projector",
                    "llm",
                ),
            ),
            supported_phases=phases,
            legalizer="qwen25_vl.aim_sparse_mrope_prompt",
        ),
        _image_binding(
            name="aim.image.llava_onevision",
            targets=(
                _target(
                    "packed_image_features",
                    Placement.PROJECTOR_LEVEL,
                    0,
                    "vision_projector",
                    "llm",
                ),
            ),
            supported_phases=frozenset({ExecutionPhase.FORWARD}),
            legalizer="llava_onevision.aim_packed_image_layout",
        ),
        _image_binding(
            name="aim.image.qwen3_vl",
            targets=(
                _target(
                    "main_image_features",
                    Placement.PROJECTOR_LEVEL,
                    0,
                    "vision_projector",
                    "llm",
                ),
                _target(
                    "deepstack_image_features",
                    Placement.PROJECTOR_LEVEL,
                    1,
                    "vision_deepstack_projectors",
                    "llm",
                ),
            ),
            supported_phases=frozenset({ExecutionPhase.FORWARD}),
            legalizer="qwen3_vl.aim_shared_deepstack_sparse_mrope",
        ),
        _image_binding(
            name="pact.image.qwen25_vl",
            strategy_factory=PACTDecoderMethod,
            targets=(
                _target(
                    "decoder_pact_boundary",
                    Placement.LLM_LEVEL,
                    0,
                    "llm",
                ),
            ),
            supported_phases=frozenset(
                {ExecutionPhase.FORWARD, ExecutionPhase.GENERATE}
            ),
            legalizer="qwen25_vl.pact_layer_local_cache_proportional_attention",
        ),
        _image_binding(
            name="pact.image.llava_onevision",
            strategy_factory=PACTDecoderMethod,
            targets=(
                _target(
                    "decoder_pact_boundary",
                    Placement.LLM_LEVEL,
                    0,
                    "llm",
                ),
            ),
            supported_phases=frozenset(
                {ExecutionPhase.FORWARD, ExecutionPhase.GENERATE}
            ),
            legalizer="llava_onevision.pact_layer_local_cache_proportional_attention",
        ),
        _image_binding(
            name="pact.image.qwen3_vl",
            strategy_factory=PACTDecoderMethod,
            targets=(
                _target(
                    "decoder_pact_boundary",
                    Placement.LLM_LEVEL,
                    0,
                    "llm",
                ),
            ),
            supported_phases=frozenset(
                {ExecutionPhase.FORWARD, ExecutionPhase.GENERATE}
            ),
            legalizer="qwen3_vl.pact_layer_local_cache",
        ),
        _image_binding(
            name="visionzip.image.llava_onevision",
            targets=(
                _target(
                    "packed_image_features",
                    Placement.PROJECTOR_LEVEL,
                    0,
                    "vision_projector",
                ),
            ),
            supported_phases=phases,
            legalizer="llava_onevision.packed_image_layout",
        ),
        _image_binding(
            name="visionzip.image.qwen25_vl",
            targets=(
                _target(
                    "merged_image_features",
                    Placement.PROJECTOR_LEVEL,
                    0,
                    "vision_projector",
                    "llm",
                ),
            ),
            supported_phases=phases,
            legalizer="qwen25_vl.sparse_mrope_prompt",
        ),
        replace(
            _image_binding(
                name="bifocal.image.qwen25_vl",
                targets=(
                    _target(
                        "merged_image_features",
                        Placement.PROJECTOR_LEVEL,
                        0,
                        "vision_encoder",
                        "vision_projector",
                        "llm",
                    ),
                ),
                supported_phases=phases,
                legalizer="qwen25_vl.sparse_mrope_prompt",
            ),
            position_resolver=None,
        ),
        _image_binding(
            name="visionzip.image.qwen3_vl",
            targets=(
                _target(
                    "main_image_features",
                    Placement.PROJECTOR_LEVEL,
                    0,
                    "vision_projector",
                    "llm",
                ),
                _target(
                    "deepstack_image_features",
                    Placement.PROJECTOR_LEVEL,
                    1,
                    "vision_deepstack_projectors",
                    "llm",
                ),
            ),
            supported_phases=phases,
            legalizer="qwen3_vl.deepstack_sparse_mrope_prompt",
        ),
        *(
            _image_binding(
                name=f"{algorithm}.image.{backbone}",
                targets=(_target(target_name, Placement.LLM_LEVEL, 0, "llm"),),
                supported_phases=(frozenset({ExecutionPhase.GENERATE}) if algorithm == "sgl" else phases),
                strategy_factory=method,
                legalizer=f"{backbone}.{legalizer}",
            )
            for algorithm, method, target_name, legalizer in (
                ("pyramid_drop", PyramidDecoderMethod, "decoder_stage_boundary",
                 "decoder_sparse_image_layout"),
                ("fastv", FastVDecoderMethod, "decoder_prune_boundary",
                 "decoder_sparse_image_layout_compact_cache"),
                ("sgl", None, "decoder_guide_prune_boundary",
                 "decoder_sparse_image_layout_compact_cache"),
                ("dart", DARTDecoderMethod, "decoder_prune_boundary",
                 "decoder_sparse_image_layout_compact_cache"),
                ("d2pruner", D2PrunerDecoderMethod, "decoder_prune_boundary",
                 "decoder_sparse_image_layout_compact_cache"),
                ("sparsevlm", SparseVLMDecoderMethod, "decoder_merge_boundaries",
                 "decoder_contiguous_image_layout_merge_cache"),
                ("finepruner", FinePrunerDecoderMethod, "decoder_iac_arp_boundaries",
                 "decoder_contiguous_image_layout_layer_local_cache"),
                ("btp", BTPDecoderMethod, "decoder_balanced_pruning_boundaries",
                 "decoder_sparse_image_layout_compact_cache"),
                ("halfv", HalfVDecoderMethod, "decoder_ivr_ssr_boundaries",
                 "decoder_sparse_image_layout_compact_cache"),
                ("transprune", TransPruneDecoderMethod, "decoder_ttv_iga_boundaries",
                 "decoder_contiguous_layout_layer_local_cache"),
            )
            for backbone in _IMAGE_BACKBONES
        ),
        _image_binding(
            name="divprune.image.llava_onevision",
            targets=(
                _target(
                    "packed_image_features",
                    Placement.PROJECTOR_LEVEL,
                    0,
                    "vision_projector",
                    "llm",
                ),
            ),
            supported_phases=phases,
            legalizer="llava_onevision.packed_image_layout",
        ),
        _image_binding(
            name="divprune.image.qwen25_vl",
            targets=(
                _target(
                    "merged_image_features",
                    Placement.PROJECTOR_LEVEL,
                    0,
                    "vision_projector",
                    "llm",
                ),
            ),
            supported_phases=phases,
            legalizer="qwen25_vl.sparse_mrope_prompt",
        ),
        _image_binding(
            name="divprune.image.qwen3_vl",
            targets=(
                _target(
                    "main_image_features",
                    Placement.PROJECTOR_LEVEL,
                    0,
                    "vision_projector",
                    "llm",
                ),
                _target(
                    "deepstack_image_features",
                    Placement.PROJECTOR_LEVEL,
                    1,
                    "vision_deepstack_projectors",
                    "llm",
                ),
            ),
            supported_phases=phases,
            legalizer="qwen3_vl.deepstack_sparse_mrope_prompt",
        ),
        _image_binding(
            name="mmtok.image.llava_onevision",
            targets=(
                _target(
                    "multisource_anyres_features",
                    Placement.PROJECTOR_LEVEL,
                    0,
                    "vision_encoder",
                    "vision_projector",
                    "llm",
                ),
            ),
            supported_phases=phases,
            legalizer="llava_onevision.mmtok_crop_to_packed_anyres",
        ),
        _image_binding(
            name="mmtok.image.qwen25_vl",
            targets=(
                _target(
                    "multisource_merged_image_features",
                    Placement.PROJECTOR_LEVEL,
                    0,
                    "vision_encoder",
                    "vision_projector",
                    "llm",
                ),
            ),
            supported_phases=phases,
            legalizer="qwen25_vl.mmtok_sparse_mrope_prompt",
        ),
        _image_binding(
            name="mmtok.image.qwen3_vl",
            targets=(
                _target(
                    "multisource_main_image_features",
                    Placement.PROJECTOR_LEVEL,
                    0,
                    "vision_encoder",
                    "vision_projector",
                    "llm",
                ),
                _target(
                    "deepstack_image_features",
                    Placement.PROJECTOR_LEVEL,
                    1,
                    "vision_deepstack_projectors",
                    "llm",
                ),
            ),
            supported_phases=phases,
            legalizer="qwen3_vl.mmtok_deepstack_sparse_mrope_prompt",
        ),
        _image_binding(
            name="vispruner.image.llava_onevision",
            targets=(
                _target(
                    "visual_attention_crop_features",
                    Placement.PROJECTOR_LEVEL,
                    0,
                    "vision_encoder",
                    "vision_projector",
                    "llm",
                ),
            ),
            supported_phases=phases,
            legalizer="llava_onevision.vispruner_crop_to_packed_anyres",
        ),
        _image_binding(
            name="vispruner.image.qwen25_vl",
            targets=(
                _target(
                    "visual_attention_merged_image_features",
                    Placement.PROJECTOR_LEVEL,
                    0,
                    "vision_encoder",
                    "vision_projector",
                    "llm",
                ),
            ),
            supported_phases=phases,
            legalizer="qwen25_vl.vispruner_sparse_mrope_prompt",
        ),
        _image_binding(
            name="vispruner.image.qwen3_vl",
            targets=(
                _target(
                    "visual_attention_main_image_features",
                    Placement.PROJECTOR_LEVEL,
                    0,
                    "vision_encoder",
                    "vision_projector",
                    "llm",
                ),
                _target(
                    "deepstack_image_features",
                    Placement.PROJECTOR_LEVEL,
                    1,
                    "vision_deepstack_projectors",
                    "llm",
                ),
            ),
            supported_phases=phases,
            legalizer="qwen3_vl.vispruner_deepstack_sparse_mrope_prompt",
        ),
        _image_binding(
            name="zoo_prune.image.llava_onevision",
            targets=(
                _target(
                    "projector_sensitivity_crop_features",
                    Placement.PROJECTOR_LEVEL,
                    0,
                    "vision_encoder",
                    "vision_projector",
                    "llm",
                ),
            ),
            supported_phases=phases,
            legalizer="llava_onevision.zoo_prune_crop_to_packed_anyres",
        ),
        _image_binding(
            name="zoo_prune.image.qwen25_vl",
            targets=(
                _target(
                    "projector_sensitivity_merged_image_features",
                    Placement.PROJECTOR_LEVEL,
                    0,
                    "vision_encoder",
                    "vision_projector",
                    "llm",
                ),
            ),
            supported_phases=phases,
            legalizer="qwen25_vl.zoo_prune_sparse_mrope_prompt",
        ),
        _image_binding(
            name="zoo_prune.image.qwen3_vl",
            targets=(
                _target(
                    "projector_sensitivity_main_image_features",
                    Placement.PROJECTOR_LEVEL,
                    0,
                    "vision_encoder",
                    "vision_projector",
                    "llm",
                ),
                _target(
                    "deepstack_image_features",
                    Placement.PROJECTOR_LEVEL,
                    1,
                    "vision_deepstack_projectors",
                    "llm",
                ),
            ),
            supported_phases=phases,
            legalizer="qwen3_vl.zoo_prune_deepstack_sparse_mrope_prompt",
        ),
        _image_binding(
            name="nuwa.image.llava_onevision",
            targets=(
                _target(
                    "stage1_regional_visual_aggregation",
                    Placement.PROJECTOR_LEVEL,
                    0,
                    "vision_encoder",
                    "vision_projector",
                    "llm",
                ),
                _target(
                    "stage2_last_prompt_attention_selection",
                    Placement.LLM_LEVEL,
                    1,
                    "llm",
                ),
            ),
            supported_phases=phases,
            legalizer="llava_onevision.nuwa_anyres_rpme_compact_cache",
        ),
        _image_binding(
            name="nuwa.image.qwen25_vl",
            targets=(
                _target(
                    "stage1_regional_visual_aggregation",
                    Placement.PROJECTOR_LEVEL,
                    0,
                    "vision_encoder",
                    "vision_projector",
                    "llm",
                ),
                _target(
                    "stage2_last_prompt_attention_selection",
                    Placement.LLM_LEVEL,
                    1,
                    "llm",
                ),
            ),
            supported_phases=phases,
            legalizer="qwen25_vl.nuwa_sparse_mrope_compact_cache",
        ),
        _image_binding(
            name="nuwa.image.qwen3_vl",
            targets=(
                _target(
                    "stage1_regional_visual_aggregation",
                    Placement.PROJECTOR_LEVEL,
                    0,
                    "vision_encoder",
                    "vision_projector",
                    "llm",
                ),
                _target(
                    "stage1_deepstack_aggregation",
                    Placement.PROJECTOR_LEVEL,
                    1,
                    "vision_deepstack_projectors",
                    "llm",
                ),
                _target(
                    "stage2_last_prompt_attention_selection",
                    Placement.LLM_LEVEL,
                    2,
                    "llm",
                ),
            ),
            supported_phases=phases,
            legalizer="qwen3_vl.nuwa_deepstack_sparse_mrope_compact_cache",
        ),
        _image_binding(
            name="restore.image.llava_onevision",
            targets=(
                _target(
                    "vispruner_pristine_plus_distinctive_anchor_merge",
                    Placement.PROJECTOR_LEVEL,
                    0,
                    "vision_encoder",
                    "vision_projector",
                    "llm",
                ),
                _target(
                    "token_mass_and_rope_distance_attention_calibration",
                    Placement.LLM_LEVEL,
                    1,
                    "llm",
                    paper_compression_stage=False,
                    component_role="attention_calibration",
                ),
            ),
            supported_phases=phases,
            legalizer="llava_onevision.restore_anyres_representative_rope_stable_cache",
        ),
        _image_binding(
            name="restore.image.qwen25_vl",
            targets=(
                _target(
                    "vispruner_pristine_plus_distinctive_anchor_merge",
                    Placement.PROJECTOR_LEVEL,
                    0,
                    "vision_encoder",
                    "vision_projector",
                    "llm",
                ),
                _target(
                    "token_mass_and_mrope_distance_attention_calibration",
                    Placement.LLM_LEVEL,
                    1,
                    "llm",
                    paper_compression_stage=False,
                    component_role="attention_calibration",
                ),
            ),
            supported_phases=phases,
            legalizer="qwen25_vl.restore_sparse_mrope_representatives_stable_cache",
        ),
        _image_binding(
            name="restore.image.qwen3_vl",
            targets=(
                _target(
                    "vispruner_pristine_plus_distinctive_anchor_merge",
                    Placement.PROJECTOR_LEVEL,
                    0,
                    "vision_encoder",
                    "vision_projector",
                    "llm",
                ),
                _target(
                    "shared_deepstack_distinctive_anchor_merge",
                    Placement.PROJECTOR_LEVEL,
                    1,
                    "vision_deepstack_projectors",
                    "llm",
                ),
                _target(
                    "token_mass_and_mrope_distance_attention_calibration",
                    Placement.LLM_LEVEL,
                    2,
                    "llm",
                    paper_compression_stage=False,
                    component_role="attention_calibration",
                ),
            ),
            supported_phases=phases,
            legalizer="qwen3_vl.restore_deepstack_sparse_mrope_stable_cache",
        ),
        _image_binding(
            name="feature_anchor_mean.image.qwen25_vl",
            targets=(
                _target(
                    "merged_image_features",
                    Placement.PROJECTOR_LEVEL,
                    0,
                    "vision_projector",
                    "llm",
                ),
            ),
            supported_phases=phases,
            legalizer="qwen25_vl.feature_anchor_mean_centroid_mrope_native_cache",
        ),
        _image_binding(
            name="grid_mean.image.qwen25_vl",
            targets=(
                _target(
                    "merged_image_features",
                    Placement.PROJECTOR_LEVEL,
                    0,
                    "vision_projector",
                    "llm",
                ),
            ),
            supported_phases=phases,
            legalizer="qwen25_vl.grid_mean_centroid_mrope_native_cache",
        ),
        _image_binding(
            name="prumerge.image.llava_onevision",
            targets=(_target("adaptive_merge", Placement.PROJECTOR_LEVEL, 0, "vision_projector", "llm"),),
            supported_phases=frozenset({ExecutionPhase.FORWARD}),
            legalizer="llava_onevision.prumerge_anyres_representative_positions",
        ),
        _image_binding(
            name="prumerge.image.qwen25_vl",
            targets=(_target("adaptive_merge", Placement.PROJECTOR_LEVEL, 0, "vision_encoder", "vision_projector", "llm"),),
            supported_phases=phases,
            legalizer="qwen25_vl.prumerge_sparse_mrope_representatives_native_cache",
        ),
        _image_binding(
            name="prumerge.image.qwen3_vl",
            targets=(
                _target("main_adaptive_merge", Placement.PROJECTOR_LEVEL, 0, "vision_projector", "llm"),
                _target("deepstack_adaptive_merge", Placement.PROJECTOR_LEVEL, 1, "vision_deepstack_projectors", "llm"),
            ),
            supported_phases=frozenset({ExecutionPhase.FORWARD}),
            legalizer="qwen3_vl.prumerge_shared_deepstack_sparse_mrope",
        ),
        _image_binding(
            name="gap.image.llava_onevision",
            targets=(
                _target(
                    "grounding_aware_prune",
                    Placement.PROJECTOR_LEVEL,
                    0,
                    "vision_projector",
                    "llm",
                    component_role="bundled_baseline_pruner_with_gap_positions",
                    paper_defined_selector=False,
                ),
            ),
            supported_phases=frozenset({ExecutionPhase.FORWARD}),
            legalizer="llava_onevision.gap_original_positions",
        ),
        _image_binding(
            name="gap.image.qwen25_vl",
            targets=(
                _target(
                    "grounding_aware_prune",
                    Placement.PROJECTOR_LEVEL,
                    0,
                    "vision_encoder",
                    "vision_projector",
                    "llm",
                    component_role="bundled_baseline_pruner_with_gap_positions",
                    paper_defined_selector=False,
                ),
            ),
            supported_phases=phases,
            legalizer="qwen25_vl.gap_original_sparse_mrope",
        ),
        _image_binding(
            name="gap.image.qwen3_vl",
            targets=(
                _target(
                    "main_grounding_aware_prune",
                    Placement.PROJECTOR_LEVEL,
                    0,
                    "vision_projector",
                    "llm",
                    component_role="bundled_baseline_pruner_with_gap_positions",
                    paper_defined_selector=False,
                ),
                _target(
                    "deepstack_grounding_aware_prune",
                    Placement.PROJECTOR_LEVEL,
                    1,
                    "vision_deepstack_projectors",
                    "llm",
                    component_role="bundled_baseline_pruner_with_gap_positions",
                    paper_defined_selector=False,
                ),
            ),
            supported_phases=frozenset({ExecutionPhase.FORWARD}),
            legalizer="qwen3_vl.gap_shared_deepstack_original_mrope",
        ),
        _image_binding(
            name="feather.image.llava_onevision",
            targets=(_target("two_stage_no_rope_prune", Placement.LLM_LEVEL, 0, "llm"),),
            supported_phases=phases,
            strategy_factory=FEATHERDecoderMethod,
            legalizer="llava_onevision.feather_original_rope_compact_cache",
        ),
        _image_binding(
            name="feather.image.qwen25_vl",
            targets=(_target("two_stage_no_rope_prune", Placement.LLM_LEVEL, 0, "llm"),),
            supported_phases=phases,
            strategy_factory=FEATHERDecoderMethod,
            legalizer="qwen25_vl.feather_original_mrope_compact_cache",
        ),
        _image_binding(
            name="feather.image.qwen3_vl",
            targets=(_target("two_stage_no_rope_prune", Placement.LLM_LEVEL, 0, "llm"),),
            supported_phases=phases,
            strategy_factory=FEATHERDecoderMethod,
            legalizer="qwen3_vl.feather_original_mrope_compact_cache",
        ),
        _image_binding(
            name="ivc_prune.image.llava_onevision",
            targets=(_target("native_rope_coordinate_prune", Placement.LLM_LEVEL, 0, "llm"),),
            supported_phases=phases,
            strategy_factory=IVCPruneDecoderMethod,
            legalizer="llava_onevision.ivc_qwen2_rope_original_positions_compact_cache",
        ),
        _image_binding(
            name="ivc_prune.image.qwen25_vl",
            targets=(_target("native_mrope_coordinate_prune", Placement.LLM_LEVEL, 0, "llm"),),
            supported_phases=phases,
            strategy_factory=IVCPruneDecoderMethod,
            legalizer="qwen25_vl.ivc_native_mrope_original_positions_compact_cache",
        ),
        _image_binding(
            name="ivc_prune.image.qwen3_vl",
            targets=(_target("native_mrope_coordinate_prune", Placement.LLM_LEVEL, 0, "llm"),),
            supported_phases=phases,
            strategy_factory=IVCPruneDecoderMethod,
            legalizer="qwen3_vl.ivc_native_mrope_compact_cache",
        ),
    )


def _declared_support_units(
    bindings: tuple[BackbonePlacementBinding, ...],
) -> tuple[SupportUnit, ...]:
    units: list[SupportUnit] = []
    for binding in bindings:
        algorithm = binding.variant.partition(".")[0]
        phases: list[ExecutionPhase] = []
        if (
            ExecutionPhase.GENERATE in binding.supported_phases
            and (
                algorithm not in _ACTIVE_GENERATE_ALGORITHMS
                or binding.backbone == "qwen3_5"
            )
        ):
            phases.append(ExecutionPhase.GENERATE)
        if (
            ExecutionPhase.FORWARD in binding.supported_phases
            and (
                algorithm not in _ACTIVE_FORWARD_ALGORITHMS
                or binding.backbone == "qwen3_5"
            )
        ):
            phases.append(ExecutionPhase.FORWARD)
        for phase in phases:
            units.append(
                SupportUnit(
                    identity=SupportUnitId(
                        algorithm=algorithm,
                        variant=binding.variant,
                        backbone=binding.backbone,
                        target_modality=Modality.IMAGE,
                        phase=phase,
                    ),
                    binding=binding.name,
                    metadata={
                        "matrix": "fuxian-image-adaptation",
                        "support_promotion": "real_checkpoint_evidence_required",
                    },
                )
            )
    return tuple(units)


def _active_generate_support_units(
    bindings: tuple[BackbonePlacementBinding, ...],
) -> tuple[SupportUnit, ...]:
    """Return image GENERATE units backed by exact cached-lifecycle evidence."""

    units: list[SupportUnit] = []
    for binding in bindings:
        algorithm = binding.variant.partition(".")[0]
        if (
            algorithm not in _ACTIVE_GENERATE_ALGORITHMS
            or binding.backbone == "qwen3_5"
        ):
            continue
        evidence = image_evidence(
            algorithm=algorithm, backbone=binding.backbone,
            binding=binding.name, phase=ExecutionPhase.GENERATE,
        )
        unit_metadata = {
            "matrix": "ImageLLM",
            "inference_active": True,
            "train_step_verified": (
                any(record.kind is EvidenceKind.TRAIN_STEP for record in evidence)
            ),
            "train_verified": False,
            "forward_support_unit": (
                "separate_active_unit"
                if algorithm in {"pyramid_drop", "pact"}
                else "separate_declared_unit"
            ),
            "evidence_artifact": evidence[0].artifact,
            "cache_policy": "compact_prefix",
            "single_image_only": True,
        }
        if algorithm == "pyramid_drop":
            unit_metadata.update(
                {
                    "cache_policy": "layer_local",
                    "cache_transform": "none",
                    "multistage_decoder_boundaries": True,
                    "position_policy": "backbone_native_layer_local",
                }
            )
        elif algorithm == "pact":
            unit_metadata.update(
                {
                    "cache_policy": "layer_local",
                    "cache_transform": "none",
                    "single_decoder_boundary": True,
                    "position_policy": "backbone_native_representative",
                    "attention_calibration": "log_cluster_mass",
                    "flash_attention_2_fallback": "sdpa",
                    "merge_groups_verified": True,
                }
            )
        elif algorithm == "fastv":
            unit_metadata.update(
                {
                    "cache_transform": "drop",
                    "single_decoder_boundary": True,
                    "position_policy": "sparse_source",
                }
            )
        elif algorithm == "sparsevlm":
            unit_metadata.update(
                {
                    "cache_transform": "mean_merge",
                    "multistage_decoder_boundaries": True,
                    "position_policy": "contiguous",
                }
            )
        elif algorithm == "finepruner":
            unit_metadata.update(
                {
                    "cache_policy": "layer_local",
                    "cache_transform": "none",
                    "multistage_decoder_boundaries": True,
                    "position_policy": "contiguous",
                    "rotary_policy": (
                        "native_through_first_iac_boundary_then_contiguous"
                    ),
                }
            )
        elif algorithm == "btp":
            unit_metadata.update(
                {
                    "cache_transform": "drop",
                    "multistage_decoder_boundaries": True,
                    "position_policy": "preserve_original",
                    "final_stage": "final_wipe",
                }
            )
        elif algorithm == "halfv":
            unit_metadata.update(
                {
                    "cache_transform": "drop",
                    "multistage_decoder_boundaries": True,
                    "position_policy": "preserve_original",
                    "stage_sequence": ("ivr", "ssr"),
                }
            )
        elif algorithm == "transprune":
            unit_metadata.update(
                {
                    "cache_policy": "layer_local",
                    "cache_transform": "none",
                    "multistage_decoder_boundaries": True,
                    "position_policy": "contiguous",
                    "rotary_policy": (
                        "native_through_first_boundary_then_contiguous"
                    ),
                }
            )
        elif algorithm == "nuwa":
            unit_metadata.update(
                {
                    "cache_transform": "drop",
                    "composite_binding": True,
                    "stage_sequence": ("stage1", "stage2"),
                    "stage2_after_layer": 16,
                    "position_policy": binding.metadata["position_policy"],
                    "shared_main_deepstack_delta": binding.metadata.get(
                        "shared_main_deepstack_delta",
                        False,
                    ),
                }
            )
        elif algorithm == "restore":
            unit_metadata.update(
                {
                    "cache_policy": "stable",
                    "cache_transform": "none",
                    "composite_binding": True,
                    "position_policy": binding.metadata["position_policy"],
                    "attention_calibration": binding.metadata[
                        "attention_calibration"
                    ],
                    "shared_main_deepstack_delta": binding.metadata.get(
                        "shared_main_deepstack_delta",
                        False,
                    ),
                }
            )
        elif algorithm == "sgl":
            unit_metadata.update(
                {
                    "cache_transform": "drop",
                    "dual_model_lifecycle": True,
                    "guide_model_path_required": True,
                    "released_executable_scope": "sgp_only",
                    "attention_sources": "prompt_and_generated",
                    "position_policy": binding.metadata["position_policy"],
                    "shared_main_deepstack_delta": binding.metadata.get(
                        "shared_main_deepstack_delta",
                        False,
                    ),
                }
            )
        elif algorithm == "feather":
            unit_metadata.update(
                {
                    "cache_transform": "drop",
                    "multistage_decoder_boundaries": True,
                    "stage_layers": (8, 16),
                    "score_source": "last_prompt_no_rope",
                    "position_policy": "preserve_original",
                }
            )
        elif algorithm == "ivc_prune":
            unit_metadata.update(
                {
                    "cache_transform": "drop",
                    "single_decoder_boundary": True,
                    "prune_layer": 16,
                    "coordinate_source": binding.metadata["coordinate_source"],
                    "position_policy": "preserve_original",
                }
            )
        units.append(
            SupportUnit(
                identity=SupportUnitId(
                    algorithm=algorithm,
                    variant=binding.variant,
                    backbone=binding.backbone,
                    target_modality=Modality.IMAGE,
                    phase=ExecutionPhase.GENERATE,
                ),
                binding=binding.name,
                status=SupportStatus.ACTIVE,
                evidence=evidence,
                metadata=unit_metadata,
            )
        )
    return tuple(units)


def _active_forward_support_units(
    bindings: tuple[BackbonePlacementBinding, ...],
) -> tuple[SupportUnit, ...]:
    """Return evidence-bearing image FORWARD support units.

    The evidence artifact records exact implementation hashes because the
    validation was intentionally performed in the mutable Agent lane.  The
    GENERATE units are independent and cannot inherit FORWARD evidence.
    """

    units: list[SupportUnit] = []
    for binding in bindings:
        algorithm = binding.variant.partition(".")[0]
        if (
            algorithm not in _ACTIVE_FORWARD_ALGORITHMS
            or ExecutionPhase.FORWARD not in binding.supported_phases
            or binding.backbone == "qwen3_5"
        ):
            continue
        evidence = image_evidence(
            algorithm=algorithm, backbone=binding.backbone,
            binding=binding.name, phase=ExecutionPhase.FORWARD,
        )
        units.append(
            SupportUnit(
                identity=SupportUnitId(
                    algorithm=algorithm,
                    variant=binding.variant,
                    backbone=binding.backbone,
                    target_modality=Modality.IMAGE,
                    phase=ExecutionPhase.FORWARD,
                ),
                binding=binding.name,
                status=SupportStatus.ACTIVE,
                evidence=evidence,
                metadata={
                    "matrix": "ImageLLM",
                    "inference_active": True,
                    "train_step_verified": (
                        any(record.kind is EvidenceKind.TRAIN_STEP for record in evidence)
                    ),
                    "train_verified": False,
                    "generate_support_unit": (
                        "not_supported_by_binding"
                        if algorithm in {"prumerge", "gap"}
                        else (
                            "separate_active_unit"
                            if algorithm == "pact"
                            else "separate_declared_unit"
                        )
                    ),
                    "evidence_artifact": evidence[0].artifact,
                },
            )
        )
    return tuple(units)


_BUILTIN_MULTIMODAL_REGISTRY: CompressionContractRegistry | None = None


def builtin_multimodal_compression_contracts(*, profile: str | None = None) -> CompressionContractRegistry:
    """Return shared multimodal kernels and phase-exact support units."""

    global _BUILTIN_MULTIMODAL_REGISTRY
    if _BUILTIN_MULTIMODAL_REGISTRY is None:
        registry = CompressionContractRegistry()
        for descriptor in (
            *_algorithm_descriptors(),
            *video_algorithm_descriptors(),
            *qwen25_omni_audio_algorithm_descriptors(),
        ):
            registry.register_algorithm(descriptor)
        for descriptor in (
            *_variant_descriptors(),
            *video_variant_descriptors(),
            *qwen25_omni_audio_variant_descriptors(),
        ):
            registry.register_variant(descriptor)
        image_bindings = _bindings()
        declared_video_bindings = video_bindings()
        audio_bindings = qwen25_omni_audio_bindings()
        bindings = (*image_bindings, *declared_video_bindings, *audio_bindings)
        for binding in bindings:
            registry.register_binding(binding)
        registry.extend_support(_declared_support_units(image_bindings))
        registry.extend_support(_active_forward_support_units(image_bindings))
        registry.extend_support(_active_generate_support_units(image_bindings))
        registry.extend_support(video_support_units())
        registry.extend_support(qwen25_omni_audio_support_units())
        from mllm_comp.compressors.operation_registry import register_qwen3_operation_contracts

        register_qwen3_operation_contracts(registry)
        from mllm_comp.fixed_mask.binding import register_fixed_mask_contracts
        register_fixed_mask_contracts(registry)
        from mllm_comp.compressors.comparison.registry import register_comparison_contracts
        register_comparison_contracts(registry)
        _BUILTIN_MULTIMODAL_REGISTRY = registry
    from mllm_comp.compressors.method_profile import apply_method_profile
    return apply_method_profile(_BUILTIN_MULTIMODAL_REGISTRY, profile)


__all__ = ["builtin_multimodal_compression_contracts"]
