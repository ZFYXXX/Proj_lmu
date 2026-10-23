"""Fixed masks in the existing prepare-time compression lifecycle."""
from __future__ import annotations

from .spec import normalize_spec


class FixedMaskMethod:
    name = "fixed_mask"
    version = "1.0.0"

    def validate_config(self, config):
        normalize_spec(config.get("config", {}))
        if config.get("operation", {}).get("type", "mask") != "mask":
            raise ValueError("fixed_mask requires operation.type=mask")
        for key, expected in (("apply_mode", "static_mask"), ("shape_effect", "preserve_shape")):
            if config.get("operation", {}).get(key, expected) != expected:
                raise ValueError(f"fixed_mask requires operation.{key}={expected}")


def create_fixed_mask_executor(*, model, facade, contract_projection, **unused):
    from .runtime import FixedMaskSession
    return FixedMaskSession(model, dict(contract_projection.algorithm_config),
                            backbone=facade.capabilities().name)


def register_fixed_mask_contracts(registry):
    from mllm_comp.compressors.contracts import (
        AlgorithmDescriptor, ModalityVariantDescriptor, BackbonePlacementBinding, BindingTarget,
    )
    from mllm_comp.core.vocabulary import (
        CompressionMethod, Placement, OperationKind, Modality, ExecutionPhase, ModalityCouplingPolicy,
    )
    from mllm_comp.core.support import SupportUnit, SupportUnitId

    modalities = frozenset({Modality.IMAGE, Modality.TEXT})
    registry.register_algorithm(AlgorithmDescriptor(
        name="fixed_mask", version="1.0.0", method=CompressionMethod.OPERATION_COMPRESSION,
        kernel=FixedMaskMethod, supported_placements=frozenset({Placement.LLM_LEVEL}),
        operations=frozenset({OperationKind.MASK}), stage_validator=FixedMaskMethod().validate_config,
        metadata={"capability_status": "target", "execution": "dense_mask_reference"},
    ))
    registry.register_variant(ModalityVariantDescriptor(
        name="fixed_mask.image_text", algorithm="fixed_mask", target_modalities=modalities,
        coupling_policy=ModalityCouplingPolicy.SYNCHRONIZED, token_unit=None,
    ))
    for backbone in ("llava_onevision", "qwen25_vl"):
        name = f"fixed_mask.image_text.{backbone}"
        registry.register_binding(BackbonePlacementBinding(
            name=name, backbone=backbone, variant="fixed_mask.image_text",
            targets=(BindingTarget(name="decoder_masks", placement=Placement.LLM_LEVEL,
                target_site="llm", target_modalities=modalities, required_handles=frozenset({"llm"})),),
            supported_phases=frozenset({ExecutionPhase.PREPARE}),
            executor_factory=create_fixed_mask_executor, strategy_factory=FixedMaskMethod,
            executor_family="llm_operation", legalizer="fixed_mask.native_layout",
            exclusive_runtime_slot="fixed_mask", metadata={"binding_config_keys": ()},
        ))
        registry.extend_support(SupportUnit(identity=SupportUnitId(
            algorithm="fixed_mask", variant="fixed_mask.image_text", backbone=backbone,
            target_modality=modality, phase=ExecutionPhase.PREPARE), binding=name,
            metadata={"physical_acceleration": False}) for modality in modalities)


def validate_stages(stages):
    """Bound the supported composition rather than silently stacking wrappers."""
    masks = [stage for stage in stages if stage.algorithm == "fixed_mask"]
    if not masks:
        return
    if len(masks) != 1:
        raise ValueError("Use one fixed_mask stage containing all component selections")
    for stage in stages:
        if stage.algorithm not in {"fixed_mask", "zoo_prune", "pyramid_drop"}:
            raise ValueError("fixed_mask composes with ZOO-Prune/PyramidDrop only")
        if stage.algorithm == "pyramid_drop" and "post_attention" in str(stage.operation):
            raise ValueError("fixed_mask + PyramidDrop requires inter-layer pruning boundaries")
