"""Explicit video variants, backbone bindings, and exact support units."""

from __future__ import annotations

from functools import partial
from types import MappingProxyType
from typing import Any, Mapping

from mllm_comp.compressors.contracts import (
    AlgorithmDescriptor,
    BackbonePlacementBinding,
    BindingTarget,
    ModalityVariantDescriptor,
    TokenUnit,
)
from mllm_comp.core.support import (
    EvidenceKind,
    SupportStatus,
    SupportUnit,
    SupportUnitId,
)
from mllm_comp.core.vocabulary import (
    CompressionMethod,
    ExecutionPhase,
    MediaAlignment,
    Modality,
    ModalityCouplingPolicy,
    Placement,
    TokenOperation,
)

from .config import (
    VIDEO_ALGORITHMS,
    VIDEO_BACKBONES,
    VIDEO_REFERENCE_REVISION,
    canonical_video_config_record,
    resolve_canonical_video_config,
    video_required_runtime_signals,
)
from .dycoke import DyCoKeVideoKernel
from .dytok import DyToKVideoKernel
from .fastvid import FastVIDVideoKernel
from .longvu import LongVUVideoKernel
from .llamavid import LLaMAVIDVideoKernel
from .metok import METokVideoKernel
from .prunevid import PruneVidVideoKernel
from .vqtoken import VQTokenVideoKernel
from .visionzip import VisionZipVideoKernel
from .evidence import video_evidence
from .positions import resolve_video_position, video_projector_site


_PRESERVED_MODALITIES = frozenset({Modality.TEXT, Modality.IMAGE, Modality.AUDIO})
_PHASES = frozenset({ExecutionPhase.FORWARD, ExecutionPhase.GENERATE})
_COMPOUND_PROJECTOR_LLM = frozenset({"metok", "prunevid", "dytok", "dycoke"})
_ALGORITHM_PLACEMENTS: Mapping[str, frozenset[Placement]] = MappingProxyType(
    {
        "metok": frozenset({Placement.PROJECTOR_LEVEL, Placement.LLM_LEVEL}),
        "prunevid": frozenset({Placement.PROJECTOR_LEVEL, Placement.LLM_LEVEL}),
        "visionzip": frozenset({Placement.PROJECTOR_LEVEL}),
        "fastvid": frozenset({Placement.PROJECTOR_LEVEL}),
        "vqtoken": frozenset({Placement.PROJECTOR_LEVEL}),
        "dytok": frozenset({Placement.PROJECTOR_LEVEL, Placement.LLM_LEVEL}),
        "longvu": frozenset({Placement.PROJECTOR_LEVEL}),
        "llamavid": frozenset({Placement.PROJECTOR_LEVEL}),
        "dycoke": frozenset({Placement.PROJECTOR_LEVEL, Placement.LLM_LEVEL}),
    }
)


_ALGORITHM_OPERATIONS: Mapping[str, frozenset[TokenOperation]] = MappingProxyType(
    {
        "metok": frozenset({TokenOperation.POOL, TokenOperation.DROP}),
        "prunevid": frozenset({TokenOperation.MERGE, TokenOperation.DROP}),
        "visionzip": frozenset({TokenOperation.KEEP, TokenOperation.MERGE}),
        "fastvid": frozenset(
            {TokenOperation.KEEP, TokenOperation.DROP, TokenOperation.MERGE}
        ),
        "vqtoken": frozenset({TokenOperation.MERGE}),
        "dytok": frozenset({TokenOperation.KEEP, TokenOperation.MERGE}),
        "longvu": frozenset({TokenOperation.DROP, TokenOperation.POOL}),
        "llamavid": frozenset({TokenOperation.POOL}),
        "dycoke": frozenset({TokenOperation.MERGE, TokenOperation.DROP}),
    }
)


_CONDITIONING_MODALITIES: Mapping[str, frozenset[Modality]] = MappingProxyType(
    {
        "metok": frozenset({Modality.TEXT}),
        "prunevid": frozenset({Modality.TEXT}),
        "visionzip": frozenset(),
        "fastvid": frozenset(),
        "vqtoken": frozenset(),
        "dytok": frozenset({Modality.TEXT}),
        "longvu": frozenset(),
        "llamavid": frozenset({Modality.TEXT}),
        "dycoke": frozenset({Modality.TEXT}),
    }
)


_ALGORITHM_STAGES: Mapping[str, tuple[tuple[str, Placement], ...]] = MappingProxyType(
    {
        "metok": (
            ("event_pool", Placement.PROJECTOR_LEVEL),
            ("prefill_attention_prune", Placement.LLM_LEVEL),
        ),
        "prunevid": (
            ("temporal_spatial_reduce", Placement.PROJECTOR_LEVEL),
            ("vtp_prefill_prune", Placement.LLM_LEVEL),
        ),
        "visionzip": (("dominant_contextual_reduce", Placement.PROJECTOR_LEVEL),),
        "fastvid": (("dyseg_stprune_dtm", Placement.PROJECTOR_LEVEL),),
        "vqtoken": (("fixed_k_cluster", Placement.PROJECTOR_LEVEL),),
        "dytok": (
            ("detection_budget_provider", Placement.LLM_LEVEL),
            ("dynamic_visionzip_reduce", Placement.PROJECTOR_LEVEL),
        ),
        "longvu": (("dino_frame_select_sva_fusion", Placement.PROJECTOR_LEVEL),),
        "llamavid": (("qformer_context_content_pool", Placement.PROJECTOR_LEVEL),),
        "dycoke": (
            ("ttm_visual_merge", Placement.PROJECTOR_LEVEL),
            ("decode_kv_view_prune", Placement.LLM_LEVEL),
        ),
    }
)


_PAPERS: Mapping[str, str] = MappingProxyType(
    {
        "metok": "METok",
        "prunevid": "PruneVid",
        "fastvid": "FastVID",
        "vqtoken": "VQToken",
        "dytok": "DyToK",
        "longvu": "LongVU",
        "llamavid": "LLaMA-VID",
        "dycoke": "DyCoKe",
    }
)


_LLM_SITES: Mapping[str, str] = MappingProxyType(
    {
        "llava_onevision": "llm_decoder_video_tokens_prefill_or_decode",
        "qwen25_vl": "llm_decoder_video_tokens_sparse_mrope",
        "qwen3_vl": "llm_decoder_video_tokens_sparse_mrope_visual_pos_masks",
    }
)
_LEGALIZERS: Mapping[str, str] = MappingProxyType(
    {
        "llava_onevision": "llava_onevision.video_projector_llm_splice",
        "qwen25_vl": "qwen25_vl.video_grid_thw_sparse_mrope",
        "qwen3_vl": "qwen3_vl.video_mrope_deepstack_visual_pos_masks",
        "qwen3_5": "qwen3_5.video_sparse_mrope",
    }
)


_PROJECTOR_HANDLES: Mapping[str, frozenset[str]] = MappingProxyType(
    {
        "llava_onevision": frozenset(
            {
                "vision_encoder",
                "vision_projector",
                "anyres_pre_llm_splice",
                "llm",
            }
        ),
        "qwen25_vl": frozenset(
            {"vision_encoder", "vision_projector", "mrope_owner", "llm"}
        ),
        "qwen3_vl": frozenset(
            {
                "vision_encoder",
                "vision_projector",
                "vision_deepstack_projectors",
                "model",
                "llm",
            }
        ),
        "qwen3_5": frozenset(
            {
                "model",
                "vision_encoder",
                "vision_projector",
                "video_features_pre_llm_scatter",
                "mrope_owner",
                "llm",
            }
        ),
    }
)
_LLM_HANDLES: Mapping[str, frozenset[str]] = MappingProxyType(
    {
        "llava_onevision": frozenset({"llm"}),
        "qwen25_vl": frozenset({"llm", "mrope_owner"}),
        "qwen3_vl": frozenset({"llm", "model"}),
        "qwen3_5": frozenset({"llm", "model", "mrope_owner"}),
    }
)


def video_algorithm_descriptors() -> tuple[AlgorithmDescriptor, ...]:
    """Return the eight video-only algorithm identities not already shared.

    ``visionzip`` already exists as the ImageLLM algorithm identity.  Its
    independent ``visionzip.video`` interpretation is added below without
    replacing the verified image kernel.
    """

    kernel_factories = {
        "metok": METokVideoKernel,
        "prunevid": PruneVidVideoKernel,
        "fastvid": FastVIDVideoKernel,
        "vqtoken": VQTokenVideoKernel,
        "dytok": DyToKVideoKernel,
        "longvu": LongVUVideoKernel,
        "llamavid": LLaMAVIDVideoKernel,
        "dycoke": DyCoKeVideoKernel,
    }
    descriptors = []
    for algorithm in VIDEO_ALGORITHMS:
        if algorithm == "visionzip":
            continue
        kernel = kernel_factories[algorithm]()
        descriptors.append(
            AlgorithmDescriptor(
                name=algorithm,
                version=kernel.version,
                method=CompressionMethod.TOKEN_COMPRESSION,
                kernel=kernel,
                stage_validator=kernel.validate_config,
                supported_placements=_ALGORITHM_PLACEMENTS[algorithm],
                operations=_ALGORITHM_OPERATIONS[algorithm],
                metadata={
                    "paper": _PAPERS[algorithm],
                    "capability_status": "target",
                    "interfaces": ("eval", "train"),
                    "implementation_state": "implemented_candidate",
                    "reference_repository": "VideoLLM-Comp",
                    "reference_revision": VIDEO_REFERENCE_REVISION,
                    "canonical_variant": f"{algorithm}.video",
                    "canonical_config": canonical_video_config_record(algorithm),
                    "defining_stages": tuple(
                        {
                            "name": name,
                            "placement": placement.value,
                            "component_role": (
                                "capability_provider"
                                if name == "detection_budget_provider"
                                else "compression"
                            ),
                        }
                        for name, placement in _ALGORITHM_STAGES[algorithm]
                    ),
                    "paper_compression_stages": tuple(
                        {
                            "name": name,
                            "placement": placement.value,
                            "method": CompressionMethod.TOKEN_COMPRESSION.value,
                        }
                        for name, placement in _ALGORITHM_STAGES[algorithm]
                        if name != "detection_budget_provider"
                    ),
                },
            )
        )
    return tuple(descriptors)

_SUPPORTED_RULES: Mapping[str, frozenset[str]] = MappingProxyType(
    {
        "metok": frozenset({
            "event_selector", "frame_grouper", "frame_scorer",
            "keyframe_selector", "pool_transform", "prune_scorer",
            "prune_selector",
        }),
        "prunevid": frozenset({
            "aggregator", "frame_grouper", "static_selector",
            "token_grouper", "vtp_scorer", "vtp_selector",
        }),
        "visionzip": frozenset({
            "aggregator", "context_selector", "dominant_selector",
            "grouper", "scorer",
        }),
        "fastvid": frozenset({
            "aggregator", "center_scorer", "center_selector",
            "context_allocator", "frame_grouper", "grouper",
            "saliency_scorer", "saliency_selector",
        }),
        "vqtoken": frozenset({
            "aggregator", "grouper", "transform",
        }),
        "dytok": frozenset({
            "aggregator", "budget_allocator", "context_selector",
            "dominant_selector", "frame_scorer", "grouper",
            "scorer",
        }),
        "longvu": frozenset({
            "frame_scorer", "frame_selector", "fusion_transform",
        }),
        "llamavid": frozenset({
            "content_aggregator", "context_aggregator", "context_scorer",
            "query_transform",
        }),
        "dycoke": frozenset({
            "decode_decision", "decode_scorer", "decode_selector",
            "ttm_scorer", "ttm_selector",
        }),
    }
)


def video_variant_descriptors() -> tuple[ModalityVariantDescriptor, ...]:
    """Return exactly one independent canonical video variant per method."""

    return tuple(
        ModalityVariantDescriptor(
            name=f"{algorithm}.video",
            algorithm=algorithm,
            target_modalities=frozenset({Modality.VIDEO}),
            conditioning_modalities=_CONDITIONING_MODALITIES[algorithm],
            preserved_modalities=_PRESERVED_MODALITIES,
            coupling_policy=ModalityCouplingPolicy.INDEPENDENT,
            media_alignment=MediaAlignment.SPATIOTEMPORAL,
            token_unit=TokenUnit.FEATURE_TOKEN,
            supported_rules=_SUPPORTED_RULES[algorithm],
            config_defaults=resolve_canonical_video_config(algorithm),
            config_validator=partial(resolve_canonical_video_config, algorithm),
            metadata={
                "scope": "video_only",
                "config_policy": "canonical_only",
                "canonical_profile": "primary_recommended",
                "reference_revision": VIDEO_REFERENCE_REVISION,
                "required_runtime_signals": tuple(
                    sorted(video_required_runtime_signals(algorithm))
                ),
                "single_video_per_sample": True,
                "multi_video_source_groups": "forbidden",
                "quality_claim": "functional_migration_only",
            },
        )
        for algorithm in VIDEO_ALGORITHMS
    )


def _backbones_for_algorithm(algorithm: str) -> tuple[str, ...]:
    """Keep Qwen3.5 video scope limited to the method already ported."""

    if algorithm == "vqtoken":
        return (*VIDEO_BACKBONES, "qwen3_5")
    return VIDEO_BACKBONES


def video_bindings() -> tuple[BackbonePlacementBinding, ...]:
    """Return exact backbone/video bindings with phase-specific evidence."""

    executor_factories = {
        "metok": _metok_executor_factory,
        "prunevid": _prunevid_executor_factory,
        "visionzip": _visionzip_executor_factory,
        "fastvid": _fastvid_executor_factory,
        "vqtoken": _vqtoken_executor_factory,
        "dytok": _dytok_executor_factory,
        "longvu": _longvu_executor_factory,
        "llamavid": _llamavid_executor_factory,
        "dycoke": _dycoke_executor_factory,
    }
    bindings = []
    for algorithm in VIDEO_ALGORITHMS:
        for backbone in _backbones_for_algorithm(algorithm):
            evidence_by_phase = {
                phase.value: video_evidence(algorithm, backbone, phase.value)
                for phase in (ExecutionPhase.FORWARD, ExecutionPhase.GENERATE)
            }
            verified_phases = tuple(phase for phase, records in evidence_by_phase.items() if records)
            targets = tuple(
                _binding_target(
                    algorithm=algorithm,
                    backbone=backbone,
                    name=name,
                    placement=placement,
                    order=order,
                )
                for order, (name, placement) in enumerate(_ALGORITHM_STAGES[algorithm])
            )
            bindings.append(
                BackbonePlacementBinding(
                    name=f"{algorithm}.video.{backbone}",
                    backbone=backbone,
                    variant=f"{algorithm}.video",
                    targets=targets,
                    supported_phases=_PHASES,
                    position_resolver=resolve_video_position,
                    executor_factory=executor_factories[algorithm](backbone),
                    strategy_factory={
                        "vqtoken": VQTokenVideoKernel,
                        "fastvid": FastVIDVideoKernel,
                        "visionzip": VisionZipVideoKernel,
                        "dycoke": DyCoKeVideoKernel,
                    }.get(algorithm),
                    legalizer=_LEGALIZERS[backbone],
                    conflicts=_video_method_conflicts(algorithm, backbone),
                    executor_family="video_compression",
                    metadata={
                        "implementation_state": "implemented_candidate",
                        "required_runtime_context": tuple(
                            sorted(video_required_runtime_signals(algorithm))
                        ),
                        "binding_config_keys": (),
                        "canonical_config": canonical_video_config_record(algorithm),
                        "layout_policy": "per_video_spatiotemporal_layout_delta",
                        "source_anchor_policy": "explicit_per_video",
                        "single_video_per_sample": True,
                        "image_behavior": "preserved",
                        "text_behavior": "preserved",
                        "compound_projector_llm": (
                            algorithm in _COMPOUND_PROJECTOR_LLM
                        ),
                        "qwen25_video_grid_thw": backbone == "qwen25_vl",
                        "qwen3_deepstack_lockstep": backbone == "qwen3_vl",
                        "qwen3_visual_pos_masks": backbone == "qwen3_vl",
                        "decode_policy": (
                            "prefill_and_decode"
                            if algorithm == "dycoke"
                            else (
                                "prefill_decoder_stage"
                                if algorithm in {"metok", "prunevid"}
                                else "prefill_only"
                            )
                        ),
                        "verified_phases": verified_phases,
                        **{
                            f"{phase}_evidence_artifact": evidence_by_phase[phase][0].artifact
                            for phase in verified_phases
                        },
                    },
                )
            )
    return tuple(bindings)


def video_support_units() -> tuple[SupportUnit, ...]:
    """Return exact units, activating only evidence-complete cells."""

    generate_evidence = frozenset(
        {
            EvidenceKind.KERNEL,
            EvidenceKind.CPU_CONTRACT,
            EvidenceKind.REAL_CHECKPOINT,
        }
    )
    forward_evidence = frozenset(
        {
            *generate_evidence,
            EvidenceKind.TRAIN_STEP,
            EvidenceKind.TRAIN_BACKWARD,
            EvidenceKind.CHECKPOINT_RELOAD,
        }
    )
    units = []
    for algorithm in VIDEO_ALGORITHMS:
        for backbone in _backbones_for_algorithm(algorithm):
            for phase in (ExecutionPhase.FORWARD, ExecutionPhase.GENERATE):
                evidence = video_evidence(algorithm, backbone, phase.value)
                evidenced = bool(evidence)
                units.append(
                    SupportUnit(
                        identity=SupportUnitId(
                            algorithm=algorithm,
                            variant=f"{algorithm}.video",
                            backbone=backbone,
                            target_modality=Modality.VIDEO,
                            phase=phase,
                        ),
                        binding=f"{algorithm}.video.{backbone}",
                        status=(
                            SupportStatus.ACTIVE
                            if evidenced
                            else SupportStatus.DECLARED
                        ),
                        evidence=evidence,
                        required_evidence=(
                            forward_evidence
                            if phase is ExecutionPhase.FORWARD
                            else generate_evidence
                        ),
                        metadata={
                            "matrix": (
                                "video-compression-qwen3_5-vqtoken"
                                if backbone == "qwen3_5"
                                else "video-compression-three-backbones"
                            ),
                            "canonical_profile": "primary_recommended",
                            "implementation_state": (
                                "evidence_verified" if evidenced else "binding_declared"
                            ),
                            "support_promotion": (
                                "passed_exact_unit_evidence"
                                if evidenced
                                else "exact_unit_evidence_required"
                            ),
                            "inference_active": evidenced and phase is ExecutionPhase.GENERATE,
                            "forward_active": evidenced and phase is ExecutionPhase.FORWARD,
                            **({"evidence_artifact": evidence[0].artifact} if evidence else {}),
                        },
                    )
                )
    return tuple(units)


def _binding_target(
    *,
    algorithm: str,
    backbone: str,
    name: str,
    placement: Placement,
    order: int,
) -> BindingTarget:
    if name == "detection_budget_provider":
        target_site = "auxiliary_detection_model_attention_budget"
    elif placement is Placement.PROJECTOR_LEVEL:
        target_site = video_projector_site(backbone, "after")
    else:
        target_site = _LLM_SITES[backbone]
    handles = (
        _PROJECTOR_HANDLES[backbone]
        if placement is Placement.PROJECTOR_LEVEL
        else _LLM_HANDLES[backbone]
    )
    return BindingTarget(
        name=name,
        placement=placement,
        target_site=target_site,
        target_modalities=frozenset({Modality.VIDEO}),
        required_handles=handles,
        order=order,
        metadata={
            "algorithm_stage": name,
            "paper_compression_stage": name != "detection_budget_provider",
            "component_role": (
                "capability_provider"
                if name == "detection_budget_provider"
                else "compression"
            ),
            "media_boundary": "one_video",
            **({"application_positions": [{"boundary": "llamavid_pre_output_projectors", "projector_owner": "compression_method"}]}
               if algorithm == "llamavid" else {}),
            **({"application_positions": [
                {"operation": "dino_frame_selection", "boundary": "pre_sva"},
                {"operation": "sva_fusion", "boundary": "longvu_pre_output_projector"}]}
               if algorithm == "longvu" else {}),
        },
    )


def _prunevid_executor_factory(backbone: str) -> Any:
    def factory(**kwargs: Any) -> Any:
        from mllm_comp.backbones.prunevid_video import (
            create_prunevid_video_executor,
        )

        return create_prunevid_video_executor(backbone=backbone, **kwargs)

    return factory


def _metok_executor_factory(backbone: str) -> Any:
    def factory(**kwargs: Any) -> Any:
        from mllm_comp.backbones.metok_video import create_metok_video_executor

        return create_metok_video_executor(backbone=backbone, **kwargs)

    return factory


def _fastvid_executor_factory(backbone: str) -> Any:
    def factory(**kwargs: Any) -> Any:
        from mllm_comp.backbones.fastvid_video import (
            create_fastvid_video_executor,
        )

        return create_fastvid_video_executor(backbone=backbone, **kwargs)

    return factory


def _vqtoken_executor_factory(backbone: str) -> Any:
    def factory(**kwargs: Any) -> Any:
        if backbone == "qwen3_5":
            from mllm_comp.backbones.qwen3_5.video_compression import (
                create_qwen3_5_video_compression_executor,
            )

            return create_qwen3_5_video_compression_executor(**kwargs)
        from mllm_comp.backbones.vqtoken_video import (
            create_vqtoken_video_executor,
        )

        return create_vqtoken_video_executor(backbone=backbone, **kwargs)

    return factory


def _visionzip_executor_factory(backbone: str) -> Any:
    def factory(**kwargs: Any) -> Any:
        from mllm_comp.backbones.visionzip_video import (
            create_visionzip_video_executor,
        )

        return create_visionzip_video_executor(backbone=backbone, **kwargs)

    return factory


def _dytok_executor_factory(backbone: str) -> Any:
    def factory(**kwargs: Any) -> Any:
        from mllm_comp.backbones.dytok_video import create_dytok_video_executor

        return create_dytok_video_executor(backbone=backbone, **kwargs)

    return factory


def _longvu_executor_factory(backbone: str) -> Any:
    def factory(**kwargs: Any) -> Any:
        from mllm_comp.backbones.longvu_video import (
            create_longvu_video_executor,
        )

        return create_longvu_video_executor(backbone=backbone, **kwargs)

    return factory


def _llamavid_executor_factory(backbone: str) -> Any:
    def factory(**kwargs: Any) -> Any:
        from mllm_comp.backbones.llamavid_video import (
            create_llamavid_video_executor,
        )

        return create_llamavid_video_executor(backbone=backbone, **kwargs)

    return factory


def _dycoke_executor_factory(backbone: str) -> Any:
    def factory(**kwargs: Any) -> Any:
        from mllm_comp.backbones.dycoke_video import (
            create_dycoke_video_executor,
        )

        return create_dycoke_video_executor(backbone=backbone, **kwargs)

    return factory


def _video_method_conflicts(algorithm: str, backbone: str) -> frozenset[str]:
    return frozenset(
        f"{candidate}.video.{backbone}"
        for candidate in VIDEO_ALGORITHMS
        if candidate != algorithm
    )


__all__ = [
    "video_algorithm_descriptors",
    "video_bindings",
    "video_support_units",
    "video_variant_descriptors",
]
