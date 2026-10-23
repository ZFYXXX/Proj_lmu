"""Qwen2.5-Omni audio contracts for the multimodal v2 registry.

Each method declares its algorithm, execution requirements and support in one
record. The four public contract layers are built from those declarations.
The Qwen executor factory is imported lazily so constructing the global
contract registry never imports Transformers or model code.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

from mllm_comp.compressors.contracts import (
    AlgorithmDescriptor,
    BackbonePlacementBinding,
    BindingTarget,
    ModalityVariantDescriptor,
    TokenUnit,
)
from mllm_comp.compressors.encoder_level import (
    AdjacentTokenMergingAudio,
    AdjacentTokenMergingSpec,
    EncoderAttentionTopKAudio,
    EncoderAttentionTopKSpec,
    ProgressiveAttentionPruningAudio,
    ProgressiveAttentionPruningSpec,
)
from mllm_comp.compressors.llm_level import (
    AudioPyramid,
    AudioPyramidLLMStrategy,
    AudioPyramidSpec,
    FastAdaSP,
    FastAdaSPLLMStrategy,
    FastAdaSPSpec,
    HeadRouter,
    HeadRouterLLMStrategy,
    HeadRouterSpec,
    StructuredLLMLayerPruning,
    StructuredLLMLayerPruningSpec,
    StructuredLLMLayerPruningStrategy,
)
from mllm_comp.compressors.projector_level import (
    FastSLMHTAAudio,
    FastSLMHTASpec,
    LocalTemporalBipartiteMergingAudio,
    LocalTemporalBipartiteMergingSpec,
    SegmentwiseTopKAudio,
    SegmentwiseTopKSpec,
    SimilarityAttentionMergePruneAudio,
    SimilarityAttentionMergePruneSpec,
    UniformAveragePoolingAudio,
    UniformAveragePoolingSpec,
    UniformSamplingAudio,
    UniformSamplingSpec,
    UnsupervisedSegmentationAudio,
    UnsupervisedSegmentationSpec,
)
from mllm_comp.compressors.projector_level.segmentwise_topk import SegmentwiseTopKProjectorStrategy
from mllm_comp.compressors.projector_level.local_temporal_bipartite_merging import LocalTemporalBipartiteProjectorStrategy
from mllm_comp.compressors.projector_level.uniform_average_pooling import UniformAveragePoolingProjectorStrategy
from mllm_comp.compressors.projector_level.uniform_sampling import UniformSamplingProjectorStrategy
from mllm_comp.compressors.projector_level.unsupervised_segmentation import UnsupervisedSegmentationProjectorStrategy
from mllm_comp.compressors.projector_level.similarity_attention_merge_prune import SimilarityAttentionMergePruneProjectorStrategy
from mllm_comp.core.support import (
    EvidenceKind,
    EvidenceRecord,
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
    OperationKind,
    Placement,
    TokenOperation,
)
from mllm_comp.backbones.qwen25_omni.audio_method_adapters import (
    QwenAdjacentMergingEncoderStrategy,
    QwenAttentionTopKEncoderStrategy,
    QwenFastSLMHTAProjectorStrategy,
    QwenProgressivePruningEncoderStrategy,
)


_DYNAMIC_PHASES = frozenset({ExecutionPhase.FORWARD, ExecutionPhase.GENERATE})
_PREPARE_PHASE = frozenset({ExecutionPhase.PREPARE})
_PRESERVED_MODALITIES = frozenset(
    {Modality.TEXT, Modality.IMAGE, Modality.VIDEO}
)


@dataclass(frozen=True)
class _AudioMethod:
    """One built-in method's algorithm, execution requirements and support claim."""

    plugin_factory: type[Any]
    spec_factory: type[Any]
    strategy_factory: Callable[[Any], Any]
    method: CompressionMethod
    placement: Placement
    operations: frozenset[TokenOperation] | frozenset[OperationKind]
    executor_family: str
    target_site: str
    required_handles: frozenset[str]
    supported_phases: frozenset[ExecutionPhase]
    supported_rules: frozenset[str]
    required_runtime_context: frozenset[str]
    metadata: Mapping[str, Any]
    active: bool
    cardinality: str = "exact"


_AUDIO_METHODS: Mapping[str, _AudioMethod] = MappingProxyType(
    {
        "segmentwise_topk_audio": _AudioMethod(
            plugin_factory=SegmentwiseTopKAudio,
            spec_factory=SegmentwiseTopKSpec,
            strategy_factory=SegmentwiseTopKProjectorStrategy,
            method=CompressionMethod.TOKEN_COMPRESSION,
            placement=Placement.PROJECTOR_LEVEL,
            operations=frozenset({TokenOperation.DROP, TokenOperation.KEEP}),
            executor_family="projector_token",
            target_site="audio_encoder_output_pre_projector",
            required_handles=frozenset({"audio_encoder", "audio_projector", "thinker"}),
            supported_phases=_DYNAMIC_PHASES,
            supported_rules=frozenset({"scorer", "selector"}),
            required_runtime_context=frozenset({"audio_encoder", "audio_projector", "audio_spans"}),
            metadata={"training_free": True},
            active=True,
        ),
        "fastslm_hta_audio": _AudioMethod(
            plugin_factory=FastSLMHTAAudio,
            spec_factory=FastSLMHTASpec,
            strategy_factory=QwenFastSLMHTAProjectorStrategy,
            method=CompressionMethod.TOKEN_COMPRESSION,
            placement=Placement.PROJECTOR_LEVEL,
            operations=frozenset({TokenOperation.POOL}),
            executor_family="projector_token",
            target_site="audio_encoder_output_pre_projector",
            required_handles=frozenset({"audio_encoder", "audio_projector", "thinker"}),
            supported_phases=_DYNAMIC_PHASES,
            supported_rules=frozenset({"transform"}),
            required_runtime_context=frozenset({"audio_encoder", "audio_projector", "audio_spans"}),
            metadata={
                "training_free": False,
                "paper": "arXiv:2601.06199v3",
                "paper_method": "Hierarchical Temporal Abstractor",
                "requires_learned_weights": True,
            },
            active=False,
        ),
        "uniform_average_pooling_audio": _AudioMethod(
            plugin_factory=UniformAveragePoolingAudio,
            spec_factory=UniformAveragePoolingSpec,
            strategy_factory=UniformAveragePoolingProjectorStrategy,
            method=CompressionMethod.TOKEN_COMPRESSION,
            placement=Placement.PROJECTOR_LEVEL,
            operations=frozenset({TokenOperation.POOL}),
            executor_family="projector_token",
            target_site="audio_encoder_output_pre_projector",
            required_handles=frozenset({"audio_encoder", "audio_projector", "thinker"}),
            supported_phases=_DYNAMIC_PHASES,
            supported_rules=frozenset({"aggregator", "grouper"}),
            required_runtime_context=frozenset({"audio_encoder", "audio_projector", "audio_spans"}),
            metadata={
                "training_free": True,
                "paper": "arXiv:2511.20973v1",
                "paper_method": "UniAvg",
            },
            active=False,
        ),
        "uniform_sampling_audio": _AudioMethod(
            plugin_factory=UniformSamplingAudio,
            spec_factory=UniformSamplingSpec,
            strategy_factory=UniformSamplingProjectorStrategy,
            method=CompressionMethod.TOKEN_COMPRESSION,
            placement=Placement.PROJECTOR_LEVEL,
            operations=frozenset({TokenOperation.DROP, TokenOperation.KEEP}),
            executor_family="projector_token",
            target_site="audio_encoder_output_pre_projector",
            required_handles=frozenset({"audio_encoder", "audio_projector", "thinker"}),
            supported_phases=_DYNAMIC_PHASES,
            supported_rules=frozenset({"selector"}),
            required_runtime_context=frozenset({"audio_encoder", "audio_projector", "audio_spans"}),
            metadata={
                "training_free": True,
                "paper": "arXiv:2511.20973v1",
                "paper_method": "UniSamp",
            },
            active=False,
        ),
        "unsupervised_segmentation_audio": _AudioMethod(
            plugin_factory=UnsupervisedSegmentationAudio,
            spec_factory=UnsupervisedSegmentationSpec,
            strategy_factory=UnsupervisedSegmentationProjectorStrategy,
            method=CompressionMethod.TOKEN_COMPRESSION,
            placement=Placement.PROJECTOR_LEVEL,
            operations=frozenset({TokenOperation.MERGE, TokenOperation.POOL}),
            executor_family="projector_token",
            target_site="audio_encoder_output_pre_projector",
            required_handles=frozenset({"audio_encoder", "audio_projector", "thinker"}),
            supported_phases=_DYNAMIC_PHASES,
            supported_rules=frozenset({"aggregator", "grouper", "scorer"}),
            required_runtime_context=frozenset({"audio_encoder", "audio_projector", "audio_spans"}),
            metadata={
                "training_free": True,
                "paper": "arXiv:2511.20973v1",
                "paper_method": "UnSeg",
                "runtime_cardinality": "data_dependent",
            },
            active=False,
            cardinality="runtime",
        ),
        "local_temporal_bipartite_merging_audio": _AudioMethod(
            plugin_factory=LocalTemporalBipartiteMergingAudio,
            spec_factory=LocalTemporalBipartiteMergingSpec,
            strategy_factory=LocalTemporalBipartiteProjectorStrategy,
            method=CompressionMethod.TOKEN_COMPRESSION,
            placement=Placement.PROJECTOR_LEVEL,
            operations=frozenset({TokenOperation.MERGE}),
            executor_family="projector_token",
            target_site="audio_encoder_output_pre_projector",
            required_handles=frozenset({"audio_encoder", "audio_projector", "thinker"}),
            supported_phases=_DYNAMIC_PHASES,
            supported_rules=frozenset({"aggregator", "grouper"}),
            required_runtime_context=frozenset({"audio_encoder", "audio_projector", "audio_spans"}),
            metadata={"training_free": True},
            active=True,
        ),
        "similarity_attention_merge_prune_audio": _AudioMethod(
            plugin_factory=SimilarityAttentionMergePruneAudio,
            spec_factory=SimilarityAttentionMergePruneSpec,
            strategy_factory=SimilarityAttentionMergePruneProjectorStrategy,
            method=CompressionMethod.TOKEN_COMPRESSION,
            placement=Placement.PROJECTOR_LEVEL,
            operations=frozenset({TokenOperation.DROP, TokenOperation.KEEP, TokenOperation.MERGE}),
            executor_family="projector_token",
            target_site="audio_projector_output_post_projector",
            required_handles=frozenset({"audio_encoder", "audio_projector", "thinker"}),
            supported_phases=_DYNAMIC_PHASES,
            supported_rules=frozenset({"aggregator", "grouper", "scorer", "selector"}),
            required_runtime_context=frozenset({
                "audio_encoder_penultimate_attention",
                "audio_projector",
                "audio_spans",
            }),
            metadata={
                "training_free": True,
                "composite_phases": ("g_sam", "attention_dpp_pruning"),
            },
            active=True,
        ),
        "adjacent_token_merging_audio": _AudioMethod(
            plugin_factory=AdjacentTokenMergingAudio,
            spec_factory=AdjacentTokenMergingSpec,
            strategy_factory=QwenAdjacentMergingEncoderStrategy,
            method=CompressionMethod.TOKEN_COMPRESSION,
            placement=Placement.ENCODER_LEVEL,
            operations=frozenset({TokenOperation.MERGE}),
            executor_family="encoder_token",
            target_site="audio_encoder_layer_post_mhsa_pre_ffn",
            required_handles=frozenset({"audio_encoder", "audio_projector", "thinker"}),
            supported_phases=_DYNAMIC_PHASES,
            supported_rules=frozenset({"aggregator", "grouper", "scorer"}),
            required_runtime_context=frozenset({"audio_encoder_layers", "audio_spans"}),
            metadata={
                "training_free": True,
                "paper_defaults": {
                    "merge_ratio_per_layer": 0.1,
                    "first_layer": 2,
                    "layer_interval": 3,
                },
            },
            active=True,
        ),
        "encoder_attention_topk_audio": _AudioMethod(
            plugin_factory=EncoderAttentionTopKAudio,
            spec_factory=EncoderAttentionTopKSpec,
            strategy_factory=QwenAttentionTopKEncoderStrategy,
            method=CompressionMethod.TOKEN_COMPRESSION,
            placement=Placement.ENCODER_LEVEL,
            operations=frozenset({TokenOperation.DROP, TokenOperation.KEEP}),
            executor_family="encoder_token",
            target_site="audio_encoder_layer_post_mhsa_pre_ffn",
            required_handles=frozenset({"audio_encoder", "audio_projector", "thinker"}),
            supported_phases=_DYNAMIC_PHASES,
            supported_rules=frozenset({"scorer", "selector"}),
            required_runtime_context=frozenset({"audio_encoder_layers", "audio_spans"}),
            metadata={
                "training_free": True,
                "paper_default_layers_zero_based": (3, 6, 9),
            },
            active=True,
        ),
        "progressive_attention_pruning_audio": _AudioMethod(
            plugin_factory=ProgressiveAttentionPruningAudio,
            spec_factory=ProgressiveAttentionPruningSpec,
            strategy_factory=QwenProgressivePruningEncoderStrategy,
            method=CompressionMethod.TOKEN_COMPRESSION,
            placement=Placement.ENCODER_LEVEL,
            operations=frozenset({TokenOperation.DROP, TokenOperation.KEEP}),
            executor_family="encoder_token",
            target_site="audio_encoder_layer_output_post_block",
            required_handles=frozenset({"audio_encoder", "audio_projector", "thinker"}),
            supported_phases=_DYNAMIC_PHASES,
            supported_rules=frozenset({"scorer", "selector"}),
            required_runtime_context=frozenset({"audio_encoder_layers", "audio_spans"}),
            metadata={"training_free": True},
            active=True,
        ),
        "audio_pyramid_attention_pruning": _AudioMethod(
            plugin_factory=AudioPyramid,
            spec_factory=AudioPyramidSpec,
            strategy_factory=AudioPyramidLLMStrategy,
            method=CompressionMethod.TOKEN_COMPRESSION,
            placement=Placement.LLM_LEVEL,
            operations=frozenset({TokenOperation.DROP, TokenOperation.KEEP}),
            executor_family="llm_token",
            target_site="llm_decoder_layer_output_audio_tokens",
            required_handles=frozenset({"llm", "thinker", "thinker.lm_head"}),
            supported_phases=_DYNAMIC_PHASES,
            supported_rules=frozenset({"scorer", "selector"}),
            required_runtime_context=frozenset({
                "absolute_mrope_positions",
                "audio_spans",
                "llm_decoder_layers",
                "prefill_decode_state",
                "semantic_instruction_mask",
            }),
            metadata={
                "training_free": True,
                "prefill_compression": True,
                "decode_compression": False,
                "position_policy": "preserve_absolute",
            },
            active=True,
        ),
        "fastadasp_weighted_token_merging": _AudioMethod(
            plugin_factory=FastAdaSP,
            spec_factory=FastAdaSPSpec,
            strategy_factory=FastAdaSPLLMStrategy,
            method=CompressionMethod.TOKEN_COMPRESSION,
            placement=Placement.LLM_LEVEL,
            operations=frozenset({TokenOperation.MERGE, TokenOperation.REWEIGHT}),
            executor_family="llm_token",
            target_site="llm_decoder_layer_post_attention_audio_tokens",
            required_handles=frozenset({"llm", "thinker", "thinker.lm_head"}),
            supported_phases=_DYNAMIC_PHASES,
            supported_rules=frozenset({"aggregator", "grouper", "scorer"}),
            required_runtime_context=frozenset({
                "absolute_mrope_positions",
                "audio_spans",
                "llm_decoder_layers",
                "prefill_decode_state",
            }),
            metadata={
                "training_free": True,
                "paper": "EMNLP 2024 Industry Track, 2024.emnlp-industry.33",
                "paper_method": "FastAdaSP weighted token merge",
                "prefill_compression": True,
                "decode_compression": False,
            },
            active=False,
        ),
        "headrouter_dynamic_audio_pruning": _AudioMethod(
            plugin_factory=HeadRouter,
            spec_factory=HeadRouterSpec,
            strategy_factory=HeadRouterLLMStrategy,
            method=CompressionMethod.TOKEN_COMPRESSION,
            placement=Placement.LLM_LEVEL,
            operations=frozenset({TokenOperation.DROP, TokenOperation.KEEP}),
            executor_family="llm_token",
            target_site="llm_decoder_layer_output_audio_tokens",
            required_handles=frozenset({"llm", "thinker", "thinker.lm_head"}),
            supported_phases=_DYNAMIC_PHASES,
            supported_rules=frozenset({"scorer", "selector"}),
            required_runtime_context=frozenset({
                "absolute_mrope_positions",
                "audio_spans",
                "llm_decoder_layers",
                "llm_input_embeddings",
                "prefill_decode_state",
                "semantic_instruction_mask",
            }),
            metadata={
                "training_free": True,
                "paper": "arXiv:2604.23717v1",
                "paper_method": "HeadRouter",
                "default_profile": "qwen2_5_omni_3b_layer1",
                "prefill_compression": True,
                "decode_compression": False,
            },
            active=False,
        ),
        "structured_llm_layer_pruning": _AudioMethod(
            plugin_factory=StructuredLLMLayerPruning,
            spec_factory=StructuredLLMLayerPruningSpec,
            strategy_factory=StructuredLLMLayerPruningStrategy,
            method=CompressionMethod.OPERATION_COMPRESSION,
            placement=Placement.LLM_LEVEL,
            operations=frozenset({OperationKind.PRUNE}),
            executor_family="llm_operation",
            target_site="llm.layers",
            required_handles=frozenset({"llm"}),
            supported_phases=_PREPARE_PHASE,
            supported_rules=frozenset({"decision"}),
            required_runtime_context=frozenset({
                "checkpoint_export",
                "llm_decoder_layers",
                "pre_optimizer_model_lifecycle",
            }),
            metadata={
                "training_free": False,
                "experiment_profile": "spade_operation_only",
                "requires_checkpoint_export": True,
            },
            active=True,
        ),
    }
)


def _executor_factory(**kwargs: Any) -> Any:
    from mllm_comp.backbones.qwen25_omni.v2_compression import (
        create_qwen25_omni_compression_executor,
    )

    return create_qwen25_omni_compression_executor(**kwargs)


def _resolve_position(**kwargs: Any) -> BackbonePlacementBinding:
    from mllm_comp.backbones.qwen25_omni.v2_compression import (
        resolve_qwen25_omni_position,
    )

    return resolve_qwen25_omni_position(**kwargs)


def qwen25_omni_audio_algorithm_descriptors() -> tuple[AlgorithmDescriptor, ...]:
    """Return the 14 backbone-neutral audio algorithm identities."""

    descriptors = []
    for name, definition in _AUDIO_METHODS.items():
        descriptors.append(
            AlgorithmDescriptor(
                name=name,
                version=str(getattr(definition.plugin_factory, "version", "1")),
                method=definition.method,
                kernel=definition.plugin_factory,
                stage_validator=definition.spec_factory.from_config,
                supported_placements=frozenset({definition.placement}),
                operations=definition.operations,
                metadata={
                    "capability_status": (
                        "active" if definition.active else "target"
                    ),
                    "interfaces": ("analysis", "eval", "train"),
                    "verified_transformers_version": "4.57.3",
                    "spec_factory": definition.spec_factory,
                    **dict(definition.metadata),
                },
            )
        )
    return tuple(descriptors)


def qwen25_omni_audio_variant_descriptors() -> tuple[ModalityVariantDescriptor, ...]:
    """Return one explicit audio interpretation for every algorithm."""

    descriptors = []
    for name, definition in _AUDIO_METHODS.items():
        family = definition.executor_family
        target_site = definition.target_site
        if family == "llm_operation":
            alignment = MediaAlignment.NONE
            token_unit = None
        elif family == "llm_token":
            alignment = MediaAlignment.TOKEN
            token_unit = TokenUnit.LLM_TOKEN
        elif target_site == "audio_projector_output_post_projector":
            alignment = MediaAlignment.TEMPORAL
            token_unit = TokenUnit.FEATURE_TOKEN
        else:
            alignment = MediaAlignment.TEMPORAL
            token_unit = TokenUnit.ENCODER_TOKEN
        conditioning = (
            frozenset({Modality.TEXT}) if family == "llm_token" else frozenset()
        )
        descriptors.append(
            ModalityVariantDescriptor(
                name=f"{name}.audio",
                algorithm=name,
                target_modalities=frozenset({Modality.AUDIO}),
                conditioning_modalities=conditioning,
                preserved_modalities=_PRESERVED_MODALITIES,
                coupling_policy=ModalityCouplingPolicy.INDEPENDENT,
                media_alignment=alignment,
                token_unit=token_unit,
                supported_rules=definition.supported_rules,
                config_defaults=_variant_config_defaults(definition.spec_factory),
                metadata={
                    "scope": "qwen25_omni_thinker_audio",
                    "target_site": target_site,
                    "cardinality": definition.cardinality,
                },
            )
        )
    return tuple(descriptors)


def qwen25_omni_audio_bindings() -> tuple[BackbonePlacementBinding, ...]:
    """Return exact Qwen placements sharing four executor capabilities."""

    family_members: dict[str, set[str]] = {}
    for name, definition in _AUDIO_METHODS.items():
        family_members.setdefault(definition.executor_family, set()).add(name)

    bindings = []
    for name, definition in _AUDIO_METHODS.items():
        family = definition.executor_family
        conflicts = frozenset(
            f"{candidate}.audio.qwen25_omni"
            for candidate in family_members[family]
            if candidate != name
        )
        legalizers = {
            "projector_token": "qwen25_omni.audio_projector_placeholder_mrope",
            "encoder_token": "qwen25_omni.audio_encoder_chunk_pooler_mrope",
            "llm_token": "qwen25_omni.audio_llm_prefill_cache_labels",
            "llm_operation": "qwen25_omni.thinker_structural_operation_export",
        }
        bindings.append(
            BackbonePlacementBinding(
                name=f"{name}.audio.qwen25_omni",
                backbone="qwen25_omni",
                variant=f"{name}.audio",
                targets=(
                    BindingTarget(
                        name=family,
                        placement=definition.placement,
                        target_site=definition.target_site,
                        target_modalities=frozenset({Modality.AUDIO}),
                        required_handles=definition.required_handles,
                    ),
                ),
                supported_phases=definition.supported_phases,
                executor_factory=_executor_factory,
                position_resolver=_resolve_position,
                legalizer=legalizers[family],
                conflicts=conflicts,
                executor_family=family,
                exclusive_runtime_slot=f"qwen25_omni.{family}" if family != "llm_operation" else None,
                prepare_after_slots={
                    "projector_token": ("qwen25_omni.encoder_token",),
                    "llm_token": ("qwen25_omni.encoder_token", "qwen25_omni.projector_token"),
                }.get(family, ()),
                strategy_factory=definition.strategy_factory,
                metadata={
                    "required_runtime_context": tuple(
                        sorted(definition.required_runtime_context)
                    ),
                    "position_policy": _position_policy(family),
                    "layout_policy": "qwen25_omni_audio_layout_delta",
                    "decode_policy": (
                        "prefill_only" if family == "llm_token" else "native"
                    ),
                },
            )
        )
    return tuple(bindings)


def qwen25_omni_audio_support_units() -> tuple[SupportUnit, ...]:
    """Return phase-exact units without inferring any Cartesian product."""

    units = []
    for name, definition in _AUDIO_METHODS.items():
        active = definition.active
        for phase in sorted(definition.supported_phases, key=lambda item: item.value):
            units.append(
                SupportUnit(
                    identity=SupportUnitId(
                        algorithm=name,
                        variant=f"{name}.audio",
                        backbone="qwen25_omni",
                        target_modality=Modality.AUDIO,
                        phase=phase,
                    ),
                    binding=f"{name}.audio.qwen25_omni",
                    status=(SupportStatus.ACTIVE if active else SupportStatus.DECLARED),
                    evidence=(_active_evidence(name, phase) if active else ()),
                    metadata={
                        "matrix": "AudioLLM",
                        "capability_status": "active" if active else "target",
                        "evidence_scope": (
                            "pre_migration_execution_core_reused"
                            if active
                            else "contract_only"
                        ),
                    },
                )
            )
    return tuple(units)


def qwen25_omni_audio_spec_factory(algorithm: str) -> type[Any]:
    """Return the algorithm-owned config normalizer selected by v2 identity."""

    try:
        return _AUDIO_METHODS[algorithm].spec_factory
    except KeyError as exc:
        raise KeyError(f"Unknown Qwen2.5-Omni audio algorithm {algorithm!r}.") from exc


def _variant_config_defaults(spec_factory: type[Any]) -> dict[str, Any]:
    # A Spec also contains schedule/criterion/position values and conditional
    # defaults (for example HeadRouter's file-backed profiles). Flattening those
    # into stage.config can override explicit sections or introduce conflicts.
    # Only these HTA architecture and weight-loading settings live exclusively
    # in stage.config; the other methods retain their native section defaults.
    if spec_factory is FastSLMHTASpec:
        return {
            name: getattr(spec_factory, name)
            for name in (
                "stage_query_tokens", "output_query_tokens", "window_tokens",
                "detail_recovery_depth", "num_attention_heads", "checkpoint_path",
                "checkpoint_strict", "trainable",
            )
        }
    return {}


def _position_policy(family: str) -> str:
    return {
        "projector_token": "compact_rebuild_mrope",
        "encoder_token": "compact_chunk_rebuild_mrope",
        "llm_token": "preserve_absolute_mrope",
        "llm_operation": "reindex_structural_layers",
    }[family]


def _active_evidence(
    algorithm: str,
    phase: ExecutionPhase,
) -> tuple[EvidenceRecord, ...]:
    common = {
        "artifact": "docs/support/qwen25_omni/status.zh.md",
        "source_commit": "ca6d1f23dc4c824e85685c28a9f1ff1171937972",
        "checksum": (
            "e7310423d859aeea4f219ecc64138328ab0aa08a1985111b02ab6b77cacd9b14"
        ),
        "recorded_at": "2026-08-24",
    }
    return (
        EvidenceRecord(
            kind=EvidenceKind.KERNEL,
            passed=True,
            metadata={"algorithm": algorithm, "scope": "algorithm_contract"},
            **common,
        ),
        EvidenceRecord(
            kind=EvidenceKind.CPU_CONTRACT,
            passed=True,
            metadata={
                "algorithm": algorithm,
                "scope": "qwen_executor_contract_reused_by_v2_binding",
            },
            **common,
        ),
        EvidenceRecord(
            kind=EvidenceKind.REAL_CHECKPOINT,
            passed=True,
            metadata={
                "algorithm": algorithm,
                "phase": phase.value,
                "checkpoint": "Qwen2.5-Omni-3B",
                "scope": "unchanged_execution_core_before_v2_dispatch_migration",
            },
            **common,
        ),
        EvidenceRecord(
            kind=EvidenceKind.TRAIN_STEP,
            passed=True,
            metadata={
                "algorithm": algorithm,
                "phase": phase.value,
                "optimizer_step": True,
                "finite_loss": True,
                "nonzero_gradient": True,
                "scope": "pre_migration_backward_baseline",
            },
            **common,
        ),
    )


__all__ = [
    "qwen25_omni_audio_algorithm_descriptors",
    "qwen25_omni_audio_bindings",
    "qwen25_omni_audio_spec_factory",
    "qwen25_omni_audio_support_units",
    "qwen25_omni_audio_variant_descriptors",
]
