from __future__ import annotations

from dataclasses import replace

import pytest

from mllm_comp.compressors.contracts import (
    AlgorithmDescriptor,
    BackbonePlacementBinding,
    BindingTarget,
    CompressionContractRegistry,
    CouplingPolicy,
    MediaAlignment,
    ModalityVariantDescriptor,
    TokenUnit,
)
from mllm_comp.core.errors import RegistryError
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
    Modality,
    OperationKind,
    Placement,
    TokenOperation,
)


def _kernel(value: object) -> object:
    return value


def _registry() -> CompressionContractRegistry:
    registry = CompressionContractRegistry()
    registry.register_algorithm(
        AlgorithmDescriptor(
            name="visionzip",
            version="1.0",
            method=CompressionMethod.TOKEN_COMPRESSION,
            kernel=_kernel,
            supported_placements=frozenset({Placement.PROJECTOR_LEVEL}),
            operations=frozenset({TokenOperation.KEEP}),
        )
    )
    registry.register_variant(
        ModalityVariantDescriptor(
            name="visionzip.image",
            algorithm="visionzip",
            target_modalities=frozenset({Modality.IMAGE}),
            conditioning_modalities=frozenset({Modality.TEXT}),
            preserved_modalities=frozenset({Modality.TEXT, Modality.VIDEO}),
            coupling_policy=CouplingPolicy.INDEPENDENT,
            media_alignment=MediaAlignment.SPATIAL,
            token_unit=TokenUnit.FEATURE_TOKEN,
        )
    )
    registry.register_binding(
        BackbonePlacementBinding(
            name="visionzip.image.llava_onevision",
            backbone="llava_onevision",
            variant="visionzip.image",
            targets=(
                BindingTarget(
                    name="packed_image_features",
                    placement=Placement.PROJECTOR_LEVEL,
                    target_site="post_anyres_pack_pre_llm_splice",
                    target_modalities=frozenset({Modality.IMAGE}),
                    required_handles=frozenset({"vision_projector"}),
                ),
            ),
            supported_phases=frozenset(
                {ExecutionPhase.FORWARD, ExecutionPhase.GENERATE}
            ),
            executor_factory=lambda **_: object(),
            legalizer="llava_onevision.image_layout",
        )
    )
    return registry


@pytest.mark.parametrize(
    ("method", "operation", "token_unit"),
    [
        (CompressionMethod.TOKEN_COMPRESSION, TokenOperation.KEEP, TokenUnit.LLM_TOKEN),
        (CompressionMethod.OPERATION_COMPRESSION, OperationKind.PRUNE, None),
    ],
)
def test_method_operations_and_variant_units_match_the_compressed_object(
    method, operation, token_unit
) -> None:
    registry = CompressionContractRegistry()
    algorithm = AlgorithmDescriptor(
        name="method", version="1", method=method, kernel=_kernel,
        supported_placements=frozenset({Placement.LLM_LEVEL}),
        operations=frozenset({operation.value}),
    )
    registry.register_algorithm(algorithm)
    variant = ModalityVariantDescriptor(
        name="method.audio", algorithm="method",
        target_modalities=frozenset({Modality.AUDIO}), token_unit=token_unit,
    )
    registry.register_variant(variant)
    assert next(iter(algorithm.operations)) is operation
    assert registry.variant(variant.name).token_unit is token_unit

    wrong_unit = TokenUnit.LLM_TOKEN if token_unit is None else None
    with pytest.raises(RegistryError, match="token.unit"):
        registry.register_variant(replace(variant, name="wrong", token_unit=wrong_unit))


@pytest.mark.parametrize(
    ("method", "operation"),
    [
        (CompressionMethod.TOKEN_COMPRESSION, OperationKind.PRUNE),
        (CompressionMethod.OPERATION_COMPRESSION, TokenOperation.DROP),
    ],
)
def test_method_rejects_operations_for_a_different_object(method, operation) -> None:
    with pytest.raises(RegistryError, match="operations must use"):
        AlgorithmDescriptor(
            name="wrong", version="1", method=method, kernel=_kernel,
            supported_placements=frozenset({Placement.LLM_LEVEL}),
            operations=frozenset({operation}),
        )


def _evidence() -> tuple[EvidenceRecord, ...]:
    return (
        EvidenceRecord(
            EvidenceKind.KERNEL,
            True,
            artifact="kernel.xml",
            source_commit="abc123",
            checksum="sha256:kernel",
        ),
        EvidenceRecord(
            EvidenceKind.CPU_CONTRACT,
            True,
            artifact="cpu.xml",
            source_commit="abc123",
            checksum="sha256:cpu",
        ),
        EvidenceRecord(
            EvidenceKind.REAL_CHECKPOINT,
            True,
            artifact="checkpoint-manifest.json",
            source_commit="abc123",
            checksum="sha256:checkpoint",
        ),
    )


def test_algorithm_variant_binding_and_support_are_independent_layers() -> None:
    registry = _registry()
    unit = SupportUnit(
        identity=SupportUnitId(
            algorithm="visionzip",
            variant="visionzip.image",
            backbone="llava_onevision",
            target_modality=Modality.IMAGE,
            phase=ExecutionPhase.GENERATE,
        ),
        binding="visionzip.image.llava_onevision",
    ).with_evidence(*_evidence()).advance(SupportStatus.ACTIVE)
    registry.register_support(unit)

    assert registry.algorithm("visionzip").kernel("tokens") == "tokens"
    assert registry.variant("visionzip.image").target_modalities == {Modality.IMAGE}
    assert registry.binding(unit.binding).is_compound is False
    assert registry.support.active() == (unit,)


def test_support_registry_does_not_infer_unverified_cartesian_products() -> None:
    registry = _registry()
    image_generate = SupportUnit(
        identity=SupportUnitId(
            "visionzip",
            "visionzip.image",
            "llava_onevision",
            Modality.IMAGE,
            ExecutionPhase.GENERATE,
        ),
        binding="visionzip.image.llava_onevision",
    ).with_evidence(*_evidence()).advance("active")
    registry.register_support(image_generate)

    assert registry.support.find(target_modality="image") == (image_generate,)
    assert registry.support.find(target_modality="video") == ()
    assert registry.support.find(phase="forward") == ()


def test_contract_registry_clone_has_independent_support_lifecycle() -> None:
    registry = _registry()
    unit = SupportUnit(
        identity=SupportUnitId(
            "visionzip",
            "visionzip.image",
            "llava_onevision",
            Modality.IMAGE,
            ExecutionPhase.GENERATE,
        ),
        binding="visionzip.image.llava_onevision",
    )
    registry.register_support(unit)

    clone = registry.clone()
    advanced = unit.with_evidence(*_evidence()).advance(SupportStatus.ACTIVE)
    clone.support.update(advanced)

    assert registry.support.get(unit.identity).status is SupportStatus.DECLARED
    assert clone.support.get(unit.identity).status is SupportStatus.ACTIVE


def test_activation_requires_explicit_passing_evidence() -> None:
    unit = SupportUnit(
        identity=SupportUnitId(
            "visionzip",
            "visionzip.image",
            "llava_onevision",
            Modality.IMAGE,
            ExecutionPhase.GENERATE,
        ),
        binding="visionzip.image.llava_onevision",
    )
    unit = unit.with_evidence(
        EvidenceRecord(EvidenceKind.KERNEL, True),
        EvidenceRecord(EvidenceKind.CPU_CONTRACT, True),
        EvidenceRecord(EvidenceKind.REAL_CHECKPOINT, False),
    )

    with pytest.raises(RegistryError, match="real_checkpoint"):
        unit.advance(SupportStatus.ACTIVE)


def test_evidence_passed_is_strict_bool() -> None:
    with pytest.raises(RegistryError, match="passed must be a bool"):
        EvidenceRecord(EvidenceKind.KERNEL, "false")  # type: ignore[arg-type]


def test_activation_requires_reproducible_provenance_for_every_passing_kind() -> None:
    unit = SupportUnit(
        identity=SupportUnitId(
            "visionzip",
            "visionzip.image",
            "llava_onevision",
            Modality.IMAGE,
            ExecutionPhase.GENERATE,
        ),
        binding="visionzip.image.llava_onevision",
    ).with_evidence(
        EvidenceRecord(EvidenceKind.KERNEL, True, artifact="kernel.xml"),
        _evidence()[1],
        _evidence()[2],
    )

    with pytest.raises(RegistryError, match="kernel.*source_commit.*checksum"):
        unit.advance(SupportStatus.ACTIVE)


def test_default_active_is_inference_only_and_does_not_force_train_evidence() -> None:
    unit = SupportUnit(
        identity=SupportUnitId(
            "visionzip",
            "visionzip.image",
            "llava_onevision",
            Modality.IMAGE,
            ExecutionPhase.GENERATE,
        ),
        binding="visionzip.image.llava_onevision",
    ).with_evidence(*_evidence()).advance(SupportStatus.ACTIVE)

    assert unit.inference_active is True
    assert unit.train_verified is False


def test_support_lifecycle_cannot_move_backwards() -> None:
    unit = SupportUnit(
        identity=SupportUnitId(
            "visionzip",
            "visionzip.image",
            "llava_onevision",
            Modality.IMAGE,
            ExecutionPhase.GENERATE,
        ),
        binding="visionzip.image.llava_onevision",
    ).with_evidence(
        EvidenceRecord(EvidenceKind.KERNEL, True),
    ).advance(SupportStatus.KERNEL_VERIFIED)

    with pytest.raises(RegistryError, match="backwards"):
        unit.advance(SupportStatus.DECLARED)


def test_training_support_can_require_backward_and_reload_evidence() -> None:
    required = frozenset(
        {
            EvidenceKind.KERNEL,
            EvidenceKind.CPU_CONTRACT,
            EvidenceKind.REAL_CHECKPOINT,
            EvidenceKind.TRAIN_BACKWARD,
            EvidenceKind.CHECKPOINT_RELOAD,
        }
    )
    unit = SupportUnit(
        identity=SupportUnitId(
            "visionzip",
            "visionzip.image",
            "llava_onevision",
            Modality.IMAGE,
            ExecutionPhase.FORWARD,
        ),
        binding="visionzip.image.llava_onevision",
        required_evidence=required,
    ).with_evidence(*_evidence())

    with pytest.raises(RegistryError, match="checkpoint_reload"):
        unit.advance("active")
    unit = unit.with_evidence(
        EvidenceRecord(
            EvidenceKind.TRAIN_BACKWARD,
            True,
            artifact="backward.xml",
            source_commit="abc123",
            checksum="sha256:backward",
        ),
        EvidenceRecord(
            EvidenceKind.CHECKPOINT_RELOAD,
            True,
            artifact="reload.xml",
            source_commit="abc123",
            checksum="sha256:reload",
        ),
    ).advance("train_verified")
    unit = unit.advance("active")
    assert unit.status is SupportStatus.ACTIVE
    assert unit.inference_active is True
    assert unit.train_verified is True


def test_train_step_verification_is_independent_from_full_training_status() -> None:
    unit = SupportUnit(
        identity=SupportUnitId(
            "visionzip",
            "visionzip.image",
            "llava_onevision",
            Modality.IMAGE,
            ExecutionPhase.FORWARD,
        ),
        binding="visionzip.image.llava_onevision",
    ).with_evidence(
        EvidenceRecord(EvidenceKind.TRAIN_STEP, True),
    )

    assert unit.train_step_verified is False
    assert unit.train_verified is False

    unit = unit.with_evidence(
        EvidenceRecord(
            EvidenceKind.TRAIN_STEP,
            True,
            artifact="train-step.json",
            source_commit="abc123",
            checksum="sha256:train-step",
            metadata={
                "optimizer_step": True,
                "finite_loss": True,
                "nonzero_gradient": True,
            },
        )
    )

    assert unit.train_step_verified is True
    assert unit.train_verified is False


def test_compound_binding_orders_multiple_modality_targets() -> None:
    binding = BackbonePlacementBinding(
        name="future_joint_av",
        backbone="future_omni",
        variant="joint.av",
        targets=(
            BindingTarget(
                "video",
                Placement.PROJECTOR_LEVEL,
                "video_features",
                frozenset({Modality.VIDEO}),
                order=1,
            ),
            BindingTarget(
                "audio",
                Placement.PROJECTOR_LEVEL,
                "audio_features",
                frozenset({Modality.AUDIO}),
                order=0,
            ),
        ),
        supported_phases=frozenset({ExecutionPhase.GENERATE}),
        executor_factory=None,
        legalizer="future.av_alignment",
    )

    assert binding.is_compound
    assert [item.name for item in binding.ordered_targets()] == ["audio", "video"]
    assert binding.target_modalities == {Modality.AUDIO, Modality.VIDEO}


def test_variant_rejects_target_preserve_overlap_and_implicit_multi_target() -> None:
    with pytest.raises(RegistryError, match="both target and preserve"):
        ModalityVariantDescriptor(
            name="bad.image",
            algorithm="bad",
            target_modalities=frozenset({Modality.IMAGE}),
            preserved_modalities=frozenset({Modality.IMAGE}),
        )
    with pytest.raises(RegistryError, match="Multi-target"):
        ModalityVariantDescriptor(
            name="bad.av",
            algorithm="bad",
            target_modalities=frozenset({Modality.AUDIO, Modality.VIDEO}),
        )


def test_binding_must_match_variant_modality_and_algorithm_placement() -> None:
    registry = _registry()
    with pytest.raises(RegistryError, match="placements not supported"):
        registry.register_binding(
            BackbonePlacementBinding(
                name="bad.encoder.binding",
                backbone="llava_onevision",
                variant="visionzip.image",
                targets=(
                    BindingTarget(
                        "encoder",
                        Placement.ENCODER_LEVEL,
                        "vision_encoder_layer",
                        frozenset({Modality.IMAGE}),
                    ),
                ),
                supported_phases=frozenset({ExecutionPhase.GENERATE}),
                executor_factory=None,
                legalizer="fixture",
            )
        )


def test_support_unit_must_match_registered_binding_exactly() -> None:
    registry = _registry()
    unit = SupportUnit(
        identity=SupportUnitId(
            "visionzip",
            "visionzip.image",
            "qwen3_vl",
            Modality.IMAGE,
            ExecutionPhase.GENERATE,
        ),
        binding="visionzip.image.llava_onevision",
    )

    with pytest.raises(RegistryError, match="variant/backbone"):
        registry.register_support(unit)
