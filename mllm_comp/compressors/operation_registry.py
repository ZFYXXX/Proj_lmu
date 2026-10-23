"""Qwen3 module replacement contracts using the shared preparation lifecycle."""

from __future__ import annotations

from mllm_comp.compressors.contracts import (
    AlgorithmDescriptor,
    BackbonePlacementBinding,
    BindingTarget,
    CompressionContractRegistry,
    ModalityVariantDescriptor,
)
from mllm_comp.compressors.llm_level import Qwen3VLAttentionReplacement, Qwen3VLFFNReplacement
from mllm_comp.core.support import SupportUnit, SupportUnitId
from mllm_comp.core.vocabulary import (
    CompressionMethod,
    ExecutionPhase,
    Modality,
    ModalityCouplingPolicy,
    OperationKind,
    Placement,
)


def _qwen3_operation_executor_factory(**kwargs):
    from mllm_comp.backbones.qwen3_vl.operation_binding import create_qwen3_operation_executor

    return create_qwen3_operation_executor(**kwargs)


def _image_operation_executor_factory(**kwargs):
    from mllm_comp.core.errors import BackendIntegrationError

    raise BackendIntegrationError(
        "Image-scoped operations require joint pipeline compilation through "
        "activate_runtime_compression or install_pipeline_split."
    )


def register_qwen3_operation_contracts(registry: CompressionContractRegistry) -> None:
    """Expose existing replacement methods without changing their numerical code.

    The catalog retains its existing active capability declaration. Exact
    preparation support starts as declared: no token-method or checkpoint
    evidence is transferred to these newly identified support units.
    """
    modalities = frozenset({Modality.TEXT, Modality.IMAGE})
    for method, kind, module in (
        (Qwen3VLAttentionReplacement, "attention", "attn"),
        (Qwen3VLFFNReplacement, "ffn", "ffn"),
    ):
        variant_name = f"{method.name}.shared_llm"
        binding_name = f"{variant_name}.qwen3_vl"
        registry.register_algorithm(AlgorithmDescriptor(
            name=method.name, version=method.version,
            method=CompressionMethod.OPERATION_COMPRESSION, kernel=method,
            supported_placements=frozenset({Placement.LLM_LEVEL}),
            operations=frozenset({OperationKind.REPLACE}),
            stage_validator=method().validate_config,
            metadata={
                "capability_status": "active",
                "strategies": sorted(method.strategies),
                "target_kind": kind,
                "shape_effect": "preserve_io_shape",
                "checkpoint_reload": "reapply_config_to_original_checkpoint",
                "legacy_backbones": ["qwen3_vl"],
                "image_scope_backbones": ["qwen25_vl", "qwen3_vl"],
            },
        ))
        registry.register_variant(ModalityVariantDescriptor(
            name=variant_name, algorithm=method.name,
            target_modalities=modalities,
            coupling_policy=ModalityCouplingPolicy.SYNCHRONIZED,
            token_unit=None, supported_rules=frozenset({"factory"}),
        ))
        registry.register_binding(BackbonePlacementBinding(
            name=binding_name, backbone="qwen3_vl", variant=variant_name,
            targets=(BindingTarget(
                name=kind, placement=Placement.LLM_LEVEL,
                target_site=f"llm.layer[*].{module}", target_modalities=modalities,
                required_handles=frozenset({"llm"}),
            ),),
            supported_phases=frozenset({ExecutionPhase.PREPARE}),
            executor_factory=_qwen3_operation_executor_factory,
            strategy_factory=method, executor_family="llm_operation",
            legalizer="qwen3_vl.shape_preserving_module_replacement",
            metadata={"binding_config_keys": ()},
        ))
        registry.extend_support(SupportUnit(
            identity=SupportUnitId(
                algorithm=method.name, variant=variant_name, backbone="qwen3_vl",
                target_modality=modality, phase=ExecutionPhase.PREPARE,
            ),
            binding=binding_name,
            metadata={"scope": "shared_llm_module_replacement"},
        ) for modality in sorted(modalities, key=lambda item: item.value))
        image_variant = f"{method.name}.image_llm"
        registry.register_variant(ModalityVariantDescriptor(
            name=image_variant, algorithm=method.name,
            target_modalities=frozenset({Modality.IMAGE}), token_unit=None,
            metadata={"scope": "physical_modality_split"},
        ))
        for backbone in ("qwen25_vl", "qwen3_vl"):
            image_binding = f"{image_variant}.{backbone}"
            registry.register_binding(BackbonePlacementBinding(
                name=image_binding, backbone=backbone, variant=image_variant,
                targets=(BindingTarget(
                    name=kind, placement=Placement.LLM_LEVEL,
                    target_site=f"llm.layer[*].{module}",
                    target_modalities=frozenset({Modality.IMAGE}),
                    required_handles=frozenset({"llm"}),
                ),),
                supported_phases=frozenset({ExecutionPhase.PREPARE}),
                executor_factory=_image_operation_executor_factory,
                strategy_factory=method, executor_family="llm_operation",
                legalizer="physical_modality_split",
                metadata={"binding_config_keys": ()},
            ))
            registry.extend_support([SupportUnit(
                identity=SupportUnitId(
                    algorithm=method.name, variant=image_variant, backbone=backbone,
                    target_modality=Modality.IMAGE, phase=ExecutionPhase.PREPARE,
                ), binding=image_binding, metadata={"scope": "physical_modality_split"},
            )])
