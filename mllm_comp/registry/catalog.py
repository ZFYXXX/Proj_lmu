"""Single catalog for declared, target, and active platform capabilities.

The catalog is the only place that assigns a lifecycle status to a built-in
capability. Target methods can execute through their available binding
contracts; exact support records separately describe verification evidence.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable

from mllm_comp.core.errors import ConfigError, RegistryError
from mllm_comp.core.vocabulary import StringEnum


class CapabilityKind(StringEnum):
    BACKBONE = "backbone"
    COMPRESSOR = "compressor"
    ANALYSIS = "analysis"
    INTEGRATION = "integration"


class CapabilityStatus(StringEnum):
    ACTIVE = "active"
    TARGET = "target"
    FUTURE = "future"


_ANALYSIS_BACKBONES = [
    "llava_onevision",
    "qwen25_vl",
    "qwen25_omni",
    "qwen3_vl",
    "qwen3_5",
]


@dataclass(frozen=True)
class CapabilitySpec:
    """Lifecycle and implementation metadata for one built-in capability."""

    name: str
    kind: CapabilityKind
    status: CapabilityStatus
    provider: Any | None = None
    profile: Any | None = None
    activation_phase: str | None = None
    interfaces: frozenset[str] = frozenset()
    override_fields: frozenset[str] = frozenset()
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def is_active(self) -> bool:
        return self.status is CapabilityStatus.ACTIVE

    def validate_config(self, config: dict[str, Any]) -> None:
        """Run a provider's schema validator without declaring it active."""

        if self.provider is None:
            return
        candidate = self.provider() if isinstance(self.provider, type) else self.provider
        validator = getattr(candidate, "validate_config", None)
        if callable(validator):
            try:
                validator(config)
            except ConfigError:
                raise
            except (TypeError, ValueError) as exc:
                raise ConfigError(
                    f"Invalid config for {self.kind.value} capability {self.name!r}: {exc}"
                ) from exc


def _compression_capability_spec(descriptor: Any) -> CapabilitySpec:
    """Project one method descriptor into the generic capability view."""

    return CapabilitySpec(
        name=descriptor.name,
        kind=CapabilityKind.COMPRESSOR,
        status=CapabilityStatus(descriptor.capability_status),
        provider=descriptor.plugin,
        activation_phase=descriptor.activation_phase,
        interfaces=descriptor.interfaces,
        metadata={
            "backbones": sorted(descriptor.backbones),
            "modalities": sorted(item.value for item in descriptor.modalities),
            "placement": descriptor.placement.value,
            "method": descriptor.method.value,
            "target_site": descriptor.target_site,
            "executor_family": descriptor.executor_family,
            "required_context": sorted(descriptor.required_context),
            **dict(descriptor.metadata),
        },
    )


def _multimodal_compression_capability_specs() -> tuple[CapabilitySpec, ...]:
    """Project v2 algorithms without flattening exact support units."""

    from mllm_comp.compressors.multimodal_registry import (
        builtin_multimodal_compression_contracts,
    )

    registry = builtin_multimodal_compression_contracts()
    specs = []
    for algorithm in registry.algorithms():
        units = registry.support.find(algorithm=algorithm.name)
        bindings = tuple(
            binding
            for binding in registry.bindings()
            if registry.variant(binding.variant).algorithm == algorithm.name
        )
        variants = tuple(
            variant
            for variant in registry.variants()
            if variant.algorithm == algorithm.name
        )
        metadata = dict(algorithm.metadata)
        capability_status = CapabilityStatus(
            metadata.pop("capability_status", CapabilityStatus.TARGET.value)
        )
        interfaces = frozenset(
            metadata.pop("interfaces", ("analysis", "eval", "train"))
        )
        target_sites = {
            target.target_site for binding in bindings for target in binding.targets
        }
        executor_families = {
            str(binding.executor_family)
            for binding in bindings
            if binding.executor_family
        }
        required_handles = {
            handle for binding in bindings for handle in binding.required_handles
        }
        specs.append(
            CapabilitySpec(
                name=algorithm.name,
                kind=CapabilityKind.COMPRESSOR,
                status=capability_status,
                provider=algorithm.kernel,
                activation_phase="multimodal_compression_v2",
                interfaces=interfaces,
                metadata={
                    "contract_version": 2,
                    "algorithm_version": algorithm.version,
                    "method": algorithm.method.value,
                    "placements": sorted(
                        placement.value for placement in algorithm.supported_placements
                    ),
                    "operations": sorted(
                        operation.value for operation in algorithm.operations
                    ),
                    "variants": sorted(variant.name for variant in variants),
                    "backbones": sorted({binding.backbone for binding in bindings}),
                    "modalities": sorted({
                        modality.value
                        for variant in variants
                        for modality in variant.target_modalities
                    }),
                    "placement": _single_catalog_value(
                        {
                            target.placement.value
                            for binding in bindings
                            for target in binding.targets
                        }
                    ),
                    "target_site": _single_catalog_value(target_sites),
                    "executor_family": _single_catalog_value(executor_families),
                    "required_context": sorted(required_handles),
                    "support_units": [
                        {
                            "identity": unit.identity.key,
                            "binding": unit.binding,
                            "status": unit.status.value,
                        }
                        for unit in units
                    ],
                    "activation_gate": "implemented_executor_and_valid_contract",
                    **metadata,
                },
            )
        )
    return tuple(specs)


def _single_catalog_value(values: set[str]) -> str | list[str] | None:
    ordered = sorted(values)
    if not ordered:
        return None
    if len(ordered) == 1:
        return ordered[0]
    return ordered


def _builtin_specs() -> tuple[CapabilitySpec, ...]:
    # Imports stay local so implementation modules do not depend on registry
    # construction and external users can import the generic catalog cheaply.
    from mllm_comp.analysis.probes import (
        AttentionFlow,
        HierarchicalTaylor,
        LateInjectionEarlyExit,
        LayerImportance,
        PredictionDynamics,
        RepresentationDynamics,
        ResidualStreamDecomposition,
        VisualMergeAttribution,
        VisualTokenPhenomena,
        VisualEvidenceInteraction,
        VisualReconstructionUtility,
        LearnedHybridEvaluation,
        AnchorTrainingEvaluation,
        Anchor7BEvaluation,
    )
    from mllm_comp.backbones.profiles import (
        QWEN25_OMNI_PROFILE,
        QWEN25_VL_PROFILE,
        QWEN3_VL_PROFILE,
        QWEN3_5_PROFILE,
    )
    from mllm_comp.backbones.qwen25_omni import create_qwen25_omni_backbone
    from mllm_comp.backbones.llava_onevision import (
        LLAVA_ONEVISION_IMAGE_PROFILE,
        create_llava_onevision_backbone,
    )
    from mllm_comp.backbones.qwen25_vl import create_qwen25_vl_backbone
    from mllm_comp.backbones.qwen3_vl import create_qwen3_vl_backbone
    from mllm_comp.backbones.qwen3_5 import create_qwen3_5_backbone
    from mllm_comp.compressors.registry import builtin_compression_methods
    from mllm_comp.compressors.multimodal_registry import (
        builtin_multimodal_compression_contracts,
    )
    from mllm_comp.integrations.lmms_eval import LMMSEvalAdapter
    from mllm_comp.integrations.ms_swift import MSSwiftTrainAdapter
    from mllm_comp.integrations.native_train import NativeSourceTrainAdapter

    return (
        CapabilitySpec(
            name="qwen25_omni",
            kind=CapabilityKind.BACKBONE,
            status=CapabilityStatus.ACTIVE,
            provider=create_qwen25_omni_backbone,
            profile=QWEN25_OMNI_PROFILE,
            activation_phase="phase1a",
            metadata={
                "active_scope": {
                    "input": ["text", "image", "audio", "video"],
                    "output": ["text", "speech"],
                },
                "constraints": {
                    "speech_batch_size": 1,
                    "composite_streaming": False,
                    "minimum_transformers_version": "4.52.3",
                    "verified_transformers_version": "4.57.3",
                },
                "verification": {
                    "date": "2026-07-20",
                    "checkpoint": "Qwen2.5-Omni-3B",
                    "physical_gpu": 1,
                    "cases": [
                        "text",
                        "image",
                        "audio",
                        "silent_video",
                        "audio_in_video",
                        "speech",
                    ],
                },
            },
        ),
        CapabilitySpec(
            name="qwen3_vl",
            kind=CapabilityKind.BACKBONE,
            status=CapabilityStatus.ACTIVE,
            provider=create_qwen3_vl_backbone,
            profile=QWEN3_VL_PROFILE,
            activation_phase="qwen3_vl_adaptation",
            metadata={
                "active_scope": {
                    "input": ["text", "image", "video"],
                    "output": ["text"],
                },
                "constraints": {
                    "audio": False,
                    "talker": False,
                    "minimum_transformers_version": "4.57.0",
                    "verified_transformers_version": "4.57.3",
                    "verified_qwen_vl_utils_version": "0.0.14",
                    "qwen_vl_utils_image_patch_size": 16,
                    "verified_video_reader": "torchcodec",
                },
                "verification": {
                    "date": "2026-08-11",
                    "checkpoint": "Qwen3-VL-8B-Instruct",
                    "physical_gpu": 1,
                    "cases": [
                        "text",
                        "image",
                        "video",
                        "deepstack_handles",
                        "text_logits",
                        "text_hidden_states",
                        "text_kv_cache",
                    ],
                    "report": "outputs/qwen3_vl_smoke_20260811.json",
                    "report_sha256": (
                        "12a8fe720bc113ab90553e86f84613dc99a1b48caa99b5b4"
                        "ef01ef049faea15b"
                    ),
                },
            },
        ),
        CapabilitySpec(
            name="qwen3_5",
            kind=CapabilityKind.BACKBONE,
            status=CapabilityStatus.ACTIVE,
            provider=create_qwen3_5_backbone,
            profile=QWEN3_5_PROFILE,
            activation_phase="qwen3_5_image_video_adaptation",
            metadata={
                "active_scope": {"input": ["text", "image", "video"], "output": ["text"]},
                "constraints": {
                    "audio": False,
                    "verified_transformers_version": "5.19.0",
                    "model_type": "qwen3_5",
                    "video_compression": "vqtoken.video.qwen3_5",
                    "initial_compression": "tome.image.qwen3_5",
                    "image_compression_methods": [
                        "tome.image.qwen3_5",
                        "divprune.image.qwen3_5",
                        "visionzip.image.qwen3_5",
                        "otprune.image.qwen3_5",
                        "prumerge.image.qwen3_5",
                        "mmtok.image.qwen3_5",
                        "gap.image.qwen3_5",
                        "vispruner.image.qwen3_5",
                        "aim.image.qwen3_5",
                        "restore.image.qwen3_5",
                        "bifocal.image.qwen3_5",
                        "zoo_prune.image.qwen3_5",
                        "saint.image.qwen3_5",
                        "nuwa.image.qwen3_5",
                        "sparsevlm.image.qwen3_5",
                        "finepruner.image.qwen3_5",
                        "transprune.image.qwen3_5",
                        "sgl.image.qwen3_5",
                        "d2pruner.image.qwen3_5",
                        "fastv.image.qwen3_5",
                        "pyramid_drop.image.qwen3_5",
                        "dart.image.qwen3_5",
                        "ivc_prune.image.qwen3_5",
                        "feather.image.qwen3_5",
                        "btp.image.qwen3_5",
                        "halfv.image.qwen3_5",
                    ],
                    "visionzip_image_validation": (
                        "registered; CPU layout/generation contract and "
                        "two-sample real-checkpoint smoke passed; "
                        "broader lifecycle validation pending"
                    ),
                    "otprune_image_validation": (
                        "registered; greedy kernel and registry smoke passed; "
                        "two-sample MME Base/OTPrune comparison completed"
                    ),
                    "prumerge_image_validation": (
                        "registered; two-sample smoke and full MME/GQA/MMBench evaluation passed"
                    ),
                    "mmtok_image_validation": (
                        "registered; 10-sample Qwen3.8 MME smoke passed; 216 to 108 tokens"
                    ),
                    "gap_image_validation": (
                        "registered; full MME and 10-sample GQA smoke passed; GQA full run intentionally stopped"
                    ),
                    "vispruner_image_validation": (
                        "registered; 10-sample Qwen3.8 MME smoke passed; 216 to 108 tokens"
                    ),
                    "aim_image_validation": (
                        "registered; 10-sample Qwen3.8 MME smoke passed; 216 to 108 tokens"
                    ),
                    "restore_image_validation": (
                        "registered; 10-sample Qwen3.8 MME smoke passed; anchor merging and attention calibration ran"
                    ),
                    "bifocal_image_validation": (
                        "registered; 10-sample Qwen3.8 MME smoke passed; pooled vision attention plus text embeddings, 216 to 108 tokens"
                    ),
                    "zoo_prune_image_validation": (
                        "registered; 10-sample Qwen3.8 MME smoke passed with 8 finite-difference directions; 216 to 108 tokens"
                    ),
                    "saint_image_validation": (
                        "registered; 10-sample Qwen3.8 MME smoke passed; native final vision-layer rotary keys pooled to merger-token groups; 216 to 108 image features"
                    ),
                    "nuwa_image_validation": (
                        "registered; 10-sample Qwen3.8 MME smoke passed; Stage 1 216 to 108 weighted features, Stage 2 score layer 15 / boundary 16 retains about half the visual tokens; no-cache recompute; cleanup errors none"
                    ),
                    "sparsevlm_image_validation": (
                        "registered; Qwen3.8 full-attention-aligned score layers [3, 7, 15] and unchanged official_v1 kernel; smoke pending"
                    ),
                    "sparsevlm_image_validation": (
                        "registered; 10-sample Qwen3.8 MME smoke passed; official_v1 kernel unchanged; full-attention score layers [3, 7, 15], three-stage token sequence traced"
                    ),
                    "finepruner_image_validation": (
                        "registered; 10-sample Qwen3.8 MME smoke passed; IAC layers [7, 14], ARP score layers [15, 19, 23] align to full attention; kernels unchanged"
                    ),
                    "transprune_image_validation": (
                        "registered; TTV module captures support linear/full mixers; IGA score layers [3, 7, 11] align to full attention; smoke pending"
                    ),
                    "fastv_image_validation": (
                        "registered; 10-sample Qwen3.8 MME smoke passed at score layer 15 / prune boundary 16; no-cache full-prefix recompute; 216 to 108 image tokens"
                    ),
                    "pyramid_drop_image_validation": (
                        "registered; 10-sample Qwen3.8 MME smoke passed across three full-attention boundaries; cache disabled; stages reduce image tokens 216 to 108 to 54 to 27"
                    ),
                    "dart_image_validation": (
                        "registered; 10-sample Qwen3.8 MME smoke passed at full-attention layer 15 / prune boundary 16; 216 to 108 image tokens; no-cache recompute"
                    ),
                    "ivc_prune_image_validation": (
                        "registered; 10-sample Qwen3.8 MME smoke passed at full-attention layer 15 / prune boundary 16; native M-RoPE and value-state path ran; 216 to 108 image tokens"
                    ),
                    "feather_image_validation": (
                        "registered; 10-sample Qwen3.8 MME smoke passed at full-attention outputs after layers 15 and 31; no-RoPE scorer handles gated Q projection"
                    ),
                    "btp_image_validation": (
                        "registered; 10-sample Qwen3.8 MME smoke passed across three attention/diversity stages plus final wipe; image tokens 216 to 108 to 54 to 27 to 0"
                    ),
                    "halfv_image_validation": (
                        "registered; 10-sample Qwen3.8 MME smoke passed; IVR no-RoPE and SSR RoPE stages reduce image tokens 216 to 108 to 54"
                    ),
                },
                "verification": {
                    "status": "image_video_vqtoken_smoke_passed",
                    "checkpoint": "Qwen3.8-27B-BF16",
                    "scope": "single-image plus single-video Base/VQToken batch-size-1 LMMS-Eval smoke",
                    "video_report": "outputs/qwen3_5-eval/video-smoke-vqtoken-final/result.json",
                },
            },
        ),
        CapabilitySpec(
            name="qwen25_vl",
            kind=CapabilityKind.BACKBONE,
            status=CapabilityStatus.ACTIVE,
            provider=create_qwen25_vl_backbone,
            profile=QWEN25_VL_PROFILE,
            activation_phase="phase1c",
            metadata={
                "active_scope": {
                    "input": ["text", "image", "video"],
                    "output": ["text"],
                },
                "constraints": {
                    "audio": False,
                    "minimum_transformers_version": "4.49.0",
                    "verified_transformers_version": "4.57.3",
                    "verified_video_reader": "torchcodec",
                },
                "verification": {
                    "date": "2026-08-11",
                    "checkpoint": "Qwen2.5-VL-3B-Instruct",
                    "physical_gpu": 1,
                    "cases": [
                        "text",
                        "image",
                        "video",
                        "text_logits",
                        "text_hidden_states",
                        "text_kv_cache",
                    ],
                    "report": "outputs/qwen25_vl_smoke_20260811.json",
                    "report_sha256": (
                        "5c26dd88eb6075b8a85a7ae0097cd8afe1857e14db94c503"
                        "7282a807da8685d6"
                    ),
                },
            },
        ),
        CapabilitySpec(
            name="llava_onevision",
            kind=CapabilityKind.BACKBONE,
            status=CapabilityStatus.ACTIVE,
            provider=create_llava_onevision_backbone,
            profile=LLAVA_ONEVISION_IMAGE_PROFILE,
            activation_phase="phase1d",
            metadata={
                "active_scope": {
                    "input": ["text", "image", "video"],
                    "output": ["text"],
                },
                "constraints": {
                    "video": True,
                    "audio": False,
                    "source_route": "vendored LLaVA-NeXT Qwen",
                    "verified_transformers_version": "4.57.3",
                    "verified_dtype": "float16",
                },
                "verification": {
                    "date": "2026-08-11",
                    "checkpoint": "llava-onevision-qwen2-0.5b-ov",
                    "physical_gpu": 1,
                    "cases": ["text", "image", "anyres_metadata"],
                    "report": "outputs/llava_onevision_smoke_20260811.json",
                    "report_sha256": (
                        "c2bfe2fa61be62cc07c2a45a192d61ce2f45b575e2276908"
                        "fa7ca33d26d3c0b5"
                    ),
                },
            },
        ),
        *(
            _compression_capability_spec(descriptor)
            for descriptor in builtin_compression_methods().descriptors()
            if descriptor.name
            not in {
                algorithm.name
                for algorithm in builtin_multimodal_compression_contracts().algorithms()
            }
        ),
        *_multimodal_compression_capability_specs(),
        CapabilitySpec(
            name="representation_dynamics",
            kind=CapabilityKind.ANALYSIS,
            status=CapabilityStatus.ACTIVE,
            provider=RepresentationDynamics,
            activation_phase="analysis_three_backbone_validation",
            override_fields=frozenset({"token_groups", "collect_points"}),
            metadata={"backbones": list(_ANALYSIS_BACKBONES)},
        ),
        CapabilitySpec(
            name="layer_importance",
            kind=CapabilityKind.ANALYSIS,
            status=CapabilityStatus.ACTIVE,
            provider=LayerImportance,
            activation_phase="analysis_three_backbone_validation",
            override_fields=frozenset({"operations", "target_token_types", "attn_edges"}),
            metadata={"backbones": list(_ANALYSIS_BACKBONES)},
        ),
        CapabilitySpec(
            name="residual_stream_decomposition",
            kind=CapabilityKind.ANALYSIS,
            status=CapabilityStatus.ACTIVE,
            provider=ResidualStreamDecomposition,
            activation_phase="analysis_three_backbone_validation",
            override_fields=frozenset({"token_groups", "operations"}),
            metadata={"backbones": list(_ANALYSIS_BACKBONES)},
        ),
        CapabilitySpec(
            name="prediction_dynamics",
            kind=CapabilityKind.ANALYSIS,
            status=CapabilityStatus.ACTIVE,
            provider=PredictionDynamics,
            activation_phase="analysis_three_backbone_validation",
            metadata={"backbones": list(_ANALYSIS_BACKBONES)},
        ),
        CapabilitySpec(
            name="attention_flow",
            kind=CapabilityKind.ANALYSIS,
            status=CapabilityStatus.ACTIVE,
            provider=AttentionFlow,
            activation_phase="analysis_three_backbone_validation",
            override_fields=frozenset({"token_groups"}),
            metadata={"backbones": list(_ANALYSIS_BACKBONES)},
        ),
        CapabilitySpec(
            name="visual_merge_attribution",
            kind=CapabilityKind.ANALYSIS,
            status=CapabilityStatus.ACTIVE,
            provider=VisualMergeAttribution,
            activation_phase="qwen25_vl_stage1_pilot",
            metadata={"backbones": ["qwen25_vl"]},
        ),
        CapabilitySpec(
            name="learned_hybrid_evaluation",
            kind=CapabilityKind.ANALYSIS,
            status=CapabilityStatus.ACTIVE,
            provider=LearnedHybridEvaluation,
            activation_phase="qwen25_vl_learned_hybrid_pilot",
            metadata={"backbones": ["qwen25_vl"]},
        ),
        CapabilitySpec(
            name="anchor_training_evaluation",
            kind=CapabilityKind.ANALYSIS,
            status=CapabilityStatus.ACTIVE,
            provider=AnchorTrainingEvaluation,
            activation_phase="qwen25_vl_training_diagnostic",
            metadata={"backbones": ["qwen25_vl"]},
        ),
        CapabilitySpec(
            name="anchor7b_evaluation",
            kind=CapabilityKind.ANALYSIS,
            status=CapabilityStatus.ACTIVE,
            provider=Anchor7BEvaluation,
            activation_phase="qwen25_vl_7b_anchor_ddp",
            metadata={"backbones": ["qwen25_vl"]},
        ),
        CapabilitySpec(
            name="visual_reconstruction_utility",
            kind=CapabilityKind.ANALYSIS,
            status=CapabilityStatus.ACTIVE,
            provider=VisualReconstructionUtility,
            activation_phase="qwen25_vl_reconstruction_utility_discovery",
            metadata={"backbones": ["qwen25_vl"]},
        ),
        CapabilitySpec(
            name="visual_evidence_interaction",
            kind=CapabilityKind.ANALYSIS,
            status=CapabilityStatus.ACTIVE,
            provider=VisualEvidenceInteraction,
            activation_phase="qwen25_vl_evidence_interaction_discovery",
            metadata={"backbones": ["qwen25_vl"]},
        ),
        CapabilitySpec(
            name="visual_token_phenomena",
            kind=CapabilityKind.ANALYSIS,
            status=CapabilityStatus.ACTIVE,
            provider=VisualTokenPhenomena,
            activation_phase="qwen25_vl_phenomenon_discovery",
            metadata={"backbones": ["qwen25_vl"]},
        ),
        CapabilitySpec(
            name="late_injection_early_exit",
            kind=CapabilityKind.ANALYSIS,
            status=CapabilityStatus.ACTIVE,
            provider=LateInjectionEarlyExit,
            activation_phase="analysis_three_backbone_validation",
            metadata={"backbones": list(_ANALYSIS_BACKBONES)},
        ),
        CapabilitySpec(
            name="hierarchical_taylor",
            kind=CapabilityKind.ANALYSIS,
            status=CapabilityStatus.ACTIVE,
            provider=HierarchicalTaylor,
            activation_phase="analysis_three_backbone_validation",
            metadata={"backbones": list(_ANALYSIS_BACKBONES)},
        ),
        CapabilitySpec(
            name="lmms_eval",
            kind=CapabilityKind.INTEGRATION,
            status=CapabilityStatus.ACTIVE,
            provider=LMMSEvalAdapter,
            activation_phase="phase1a",
            interfaces=frozenset({"analysis", "eval"}),
            metadata={
                "runtime_ownership": "mllm_comp_platform",
                "framework_ownership": ["tasks", "datasets", "metrics"],
                "supported_request_types": ["generate_until"],
                "output": ["text"],
                "constraints": {
                    "runtime_batch_size": 1,
                    "loglikelihood": False,
                    "multi_round": False,
                    "talker_metric_output": False,
                },
            },
        ),
        CapabilitySpec(
            name="native_source",
            kind=CapabilityKind.INTEGRATION,
            status=CapabilityStatus.FUTURE,
            provider=NativeSourceTrainAdapter,
            interfaces=frozenset({"train"}),
        ),
        CapabilitySpec(
            name="ms_swift",
            kind=CapabilityKind.INTEGRATION,
            status=CapabilityStatus.ACTIVE,
            provider=MSSwiftTrainAdapter,
            activation_phase="framework_bridge_phase1a",
            interfaces=frozenset({"train"}),
            metadata={
                "framework_ownership": [
                    "model_lifecycle",
                    "dataset",
                    "trainer",
                    "optimizer",
                    "distributed",
                    "checkpoint",
                ],
                **MSSwiftTrainAdapter.capability_metadata(),
            },
        ),
    )


BUILTIN_CAPABILITIES: tuple[CapabilitySpec, ...] = _builtin_specs()


def _build_capability_index() -> dict[tuple[CapabilityKind, str], CapabilitySpec]:
    index: dict[tuple[CapabilityKind, str], CapabilitySpec] = {}
    for spec in BUILTIN_CAPABILITIES:
        key = (spec.kind, spec.name)
        if key in index:
            raise RegistryError(
                f"Duplicate {spec.kind.value} capability declaration {spec.name!r}."
            )
        profile_name = getattr(spec.profile, "name", spec.name)
        if profile_name != spec.name:
            raise RegistryError(
                f"Capability {spec.name!r} references profile {profile_name!r}."
            )
        if spec.is_active and spec.provider is None:
            raise RegistryError(
                f"Active {spec.kind.value} capability {spec.name!r} has no provider."
            )
        index[key] = spec
    return index


_CAPABILITIES = _build_capability_index()


def _registered_compressor_specs() -> tuple[CapabilitySpec, ...]:
    """Return descriptors registered after the catalog module was imported."""

    from mllm_comp.compressors.registry import builtin_compression_methods

    return tuple(
        _compression_capability_spec(descriptor)
        for descriptor in builtin_compression_methods().descriptors()
        if (CapabilityKind.COMPRESSOR, descriptor.name) not in _CAPABILITIES
    )


def get_capability_spec(kind: CapabilityKind | str, name: str) -> CapabilitySpec:
    """Return one declared capability, regardless of lifecycle status."""

    normalized_kind = CapabilityKind(kind)
    try:
        return _CAPABILITIES[(normalized_kind, name)]
    except KeyError as exc:
        if normalized_kind is CapabilityKind.COMPRESSOR:
            dynamic = {
                spec.name: spec for spec in _registered_compressor_specs()
            }
            if name in dynamic:
                return dynamic[name]
        raise RegistryError(f"Unknown {normalized_kind.value} capability '{name}'.") from exc


def iter_capability_specs(
    *,
    kind: CapabilityKind | str | None = None,
    status: CapabilityStatus | str | None = None,
) -> Iterable[CapabilitySpec]:
    normalized_kind = CapabilityKind(kind) if kind is not None else None
    normalized_status = CapabilityStatus(status) if status is not None else None
    for spec in BUILTIN_CAPABILITIES:
        if normalized_kind is not None and spec.kind is not normalized_kind:
            continue
        if normalized_status is not None and spec.status is not normalized_status:
            continue
        yield spec
    if normalized_kind in {None, CapabilityKind.COMPRESSOR}:
        for spec in _registered_compressor_specs():
            if normalized_status is not None and spec.status is not normalized_status:
                continue
            yield spec


def capability_names(
    kind: CapabilityKind | str,
    *,
    interface: str | None = None,
) -> tuple[str, ...]:
    """List declared names, optionally restricted to one command interface."""

    return tuple(
        spec.name
        for spec in iter_capability_specs(kind=kind)
        if interface is None or interface in spec.interfaces
    )


def active_providers(kind: CapabilityKind | str) -> dict[str, Any]:
    """Build the runtime registry view for one capability kind."""

    return {
        spec.name: spec.provider
        for spec in iter_capability_specs(kind=kind, status=CapabilityStatus.ACTIVE)
        if spec.provider is not None
    }
