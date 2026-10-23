from __future__ import annotations

from mllm_comp.compressors.multimodal_registry import (
    builtin_multimodal_compression_contracts,
)
from mllm_comp.core.support import SupportStatus
from mllm_comp.core.vocabulary import (
    CompressionMethod,
    ExecutionPhase,
    Modality,
    Placement,
)


_ALGORITHMS = {
    "aim",
    "btp",
    "d2pruner",
    "dart",
    "divprune",
    "fastv",
    "feather",
    "finepruner",
    "gap",
    "halfv",
    "ivc_prune",
    "mmtok",
    "nuwa",
    "pact",
    "prumerge",
    "pyramid_drop",
    "restore",
    "saint",
    "sgl",
    "sparsevlm",
    "tome",
    "transprune",
    "visionzip",
    "vispruner",
    "zoo_prune",
}
_BACKBONES = {"llava_onevision", "qwen25_vl", "qwen3_vl"}
_TOKEN = CompressionMethod.TOKEN_COMPRESSION
_OPERATION = CompressionMethod.OPERATION_COMPRESSION
_PAPER_COMPRESSION_STAGES = {
    "aim": {
        (Placement.PROJECTOR_LEVEL, _TOKEN),
        (Placement.LLM_LEVEL, _TOKEN),
    },
    "btp": {(Placement.LLM_LEVEL, _TOKEN)},
    "d2pruner": {(Placement.LLM_LEVEL, _TOKEN)},
    "dart": {(Placement.LLM_LEVEL, _TOKEN)},
    "divprune": {(Placement.PROJECTOR_LEVEL, _TOKEN)},
    "fastv": {(Placement.LLM_LEVEL, _TOKEN)},
    "feather": {(Placement.LLM_LEVEL, _TOKEN)},
    "finepruner": {(Placement.LLM_LEVEL, _TOKEN)},
    "gap": {(Placement.PROJECTOR_LEVEL, _TOKEN)},
    "halfv": {
        (Placement.LLM_LEVEL, _TOKEN),
        (Placement.LLM_LEVEL, _OPERATION),
    },
    "ivc_prune": {(Placement.LLM_LEVEL, _TOKEN)},
    "mmtok": {(Placement.PROJECTOR_LEVEL, _TOKEN)},
    "nuwa": {
        (Placement.PROJECTOR_LEVEL, _TOKEN),
        (Placement.LLM_LEVEL, _TOKEN),
    },
    "pact": {(Placement.LLM_LEVEL, _TOKEN)},
    "prumerge": {(Placement.PROJECTOR_LEVEL, _TOKEN)},
    "pyramid_drop": {(Placement.LLM_LEVEL, _TOKEN)},
    "restore": {(Placement.PROJECTOR_LEVEL, _TOKEN)},
    "saint": {
        (Placement.ENCODER_LEVEL, _TOKEN),
        (Placement.LLM_LEVEL, _TOKEN),
    },
    "sgl": {(Placement.LLM_LEVEL, _TOKEN)},
    "sparsevlm": {(Placement.LLM_LEVEL, _TOKEN)},
    "tome": {(Placement.ENCODER_LEVEL, _TOKEN)},
    "transprune": {(Placement.LLM_LEVEL, _TOKEN)},
    "visionzip": {(Placement.PROJECTOR_LEVEL, _TOKEN)},
    "vispruner": {(Placement.PROJECTOR_LEVEL, _TOKEN)},
    "zoo_prune": {(Placement.PROJECTOR_LEVEL, _TOKEN)},
}


def test_registry_preserves_25_image_algorithms_and_phase_exact_cells() -> None:
    registry = builtin_multimodal_compression_contracts()

    assert _ALGORITHMS <= {item.name for item in registry.algorithms()}
    image_variants = [
        variant
        for variant in registry.variants()
        if Modality.IMAGE in variant.target_modalities and variant.algorithm in _ALGORITHMS
    ]
    image_bindings = [
        binding
        for binding in registry.bindings()
        if Modality.IMAGE in binding.target_modalities
        and registry.variant(binding.variant).algorithm in _ALGORITHMS
    ]
    assert len(image_variants) == 26
    assert len(image_bindings) == 75
    units = [unit for unit in registry.support.find(target_modality=Modality.IMAGE)
        if unit.identity.algorithm in _ALGORITHMS]
    assert len(units) == 135
    assert {unit.identity.target_modality for unit in units} == {Modality.IMAGE}
    assert {unit.identity.backbone for unit in units} == _BACKBONES

    generate = [
        unit for unit in units if unit.identity.phase is ExecutionPhase.GENERATE
    ]
    forward = [
        unit for unit in units if unit.identity.phase is ExecutionPhase.FORWARD
    ]
    assert len(generate) == 63
    assert len(forward) == 72
    assert sum(unit.status is SupportStatus.ACTIVE for unit in generate) == 42
    assert sum(unit.status is SupportStatus.ACTIVE for unit in forward) == 36
    train_step_verified = [unit for unit in units if unit.train_step_verified]
    assert len(train_step_verified) == 72
    assert all(unit.status is SupportStatus.ACTIVE for unit in train_step_verified)
    assert {unit.identity.algorithm for unit in train_step_verified} == (
        _ALGORITHMS - {"d2pruner"}
    )
    assert {
        backbone: sum(
            unit.identity.backbone == backbone for unit in train_step_verified
        )
        for backbone in _BACKBONES
    } == {backbone: 24 for backbone in _BACKBONES}
    assert all(not unit.train_verified for unit in units)
    assert all(
        record.has_activation_provenance
        and str(record.artifact).startswith("outputs/imagellm/")
        for unit in units
        if unit.status is SupportStatus.ACTIVE
        for record in unit.evidence
    )


def test_each_algorithm_has_one_kernel_variant_and_three_bindings() -> None:
    registry = builtin_multimodal_compression_contracts()

    for algorithm in _ALGORITHMS:
        variants = [
            item
            for item in registry.variants()
            if item.algorithm == algorithm
            if Modality.IMAGE in item.target_modalities
        ]
        expected_variants = 2 if algorithm == "saint" else 1
        assert len(variants) == expected_variants
        bindings = [
            item
            for item in registry.bindings()
            if item.variant in {variant.name for variant in variants}
        ]
        assert len(bindings) == 3
        assert callable(registry.algorithm(algorithm).kernel)
        assert all(callable(binding.executor_factory) for binding in bindings)


def test_each_algorithm_declares_paper_compression_stages() -> None:
    registry = builtin_multimodal_compression_contracts()

    assert set(_PAPER_COMPRESSION_STAGES) == _ALGORITHMS
    for algorithm, expected in _PAPER_COMPRESSION_STAGES.items():
        descriptor = registry.algorithm(algorithm)
        actual = {
            (
                Placement(stage["placement"]),
                CompressionMethod(stage["method"]),
            )
            for stage in descriptor.metadata["paper_compression_stages"]
        }
        assert actual == expected
        assert descriptor.metadata["paper_implementation_alignment"]


def test_paper_stages_are_not_inferred_from_framework_helper_targets() -> None:
    registry = builtin_multimodal_compression_contracts()

    assert registry.algorithm("tome").supported_placements == {
        Placement.ENCODER_LEVEL,
        Placement.PROJECTOR_LEVEL,
    }
    assert registry.algorithm("tome").metadata[
        "paper_implementation_alignment"
    ] == "aligned_with_framework_adaptation"
    for backbone in _BACKBONES:
        targets = registry.binding(f"tome.image.{backbone}").targets
        output_compaction = next(
            target for target in targets if target.name == "permanent_output_merge"
        )
        assert output_compaction.metadata == {
            "paper_compression_stage": False,
            "component_role": "framework_output_compaction",
        }

    assert registry.algorithm("aim").supported_placements == {
        Placement.PROJECTOR_LEVEL
    }
    assert registry.algorithm("aim").metadata[
        "paper_implementation_alignment"
    ] == "partial"

    assert registry.algorithm("saint").supported_placements == {
        Placement.ENCODER_LEVEL,
        Placement.PROJECTOR_LEVEL,
    }
    assert registry.algorithm("saint").metadata[
        "paper_implementation_alignment"
    ] == "partial"
    assert all(
        target.placement is not Placement.LLM_LEVEL
        for backbone in _BACKBONES
        for target in registry.binding(f"saint.image.{backbone}").targets
    )

    assert registry.algorithm("restore").supported_placements == {
        Placement.PROJECTOR_LEVEL,
        Placement.LLM_LEVEL,
    }
    for backbone in _BACKBONES:
        llm_target = next(
            target
            for target in registry.binding(f"restore.image.{backbone}").targets
            if target.placement is Placement.LLM_LEVEL
        )
        assert llm_target.metadata == {
            "paper_compression_stage": False,
            "component_role": "attention_calibration",
        }

    halfv = registry.algorithm("halfv")
    assert halfv.method is CompressionMethod.TOKEN_COMPRESSION
    assert halfv.metadata["paper_implementation_alignment"] == "partial"
    assert registry.binding("halfv.image.llava_onevision").metadata[
        "paper_architecture_alignment"
    ] == "partial_missing_static_visual_compute"

    for backbone in _BACKBONES:
        for target in registry.binding(f"gap.image.{backbone}").targets:
            assert target.metadata["paper_defined_selector"] is False


def test_d2pruner_remains_declared_without_prior() -> None:
    registry = builtin_multimodal_compression_contracts()

    units = registry.support.find(algorithm="d2pruner")
    assert len(units) == 6
    assert {unit.identity.backbone for unit in units} == _BACKBONES
    assert {unit.identity.phase.value for unit in units} == {"forward", "generate"}
    assert {unit.status for unit in units} == {SupportStatus.DECLARED}
    assert all(unit.evidence == () for unit in units)
    for backbone in _BACKBONES:
        binding = registry.binding(f"d2pruner.image.{backbone}")
        assert binding.metadata["requires_calibrated_bias_prior"] is True


def test_image_method_bindings_are_pairwise_conflicting() -> None:
    registry = builtin_multimodal_compression_contracts()

    for backbone in _BACKBONES:
        for algorithm in _ALGORITHMS:
            binding = registry.binding(f"{algorithm}.image.{backbone}")
            assert binding.conflicts == {
                f"{other}.image.{backbone}"
                for other in _ALGORITHMS
                if other != algorithm
            }


def test_broad_projector_and_qwen3_deepstack_are_explicit() -> None:
    registry = builtin_multimodal_compression_contracts()
    visionzip = registry.algorithm("visionzip")
    qwen3 = registry.binding("visionzip.image.qwen3_vl")

    assert visionzip.metadata["projector_scope"] == "encoder_output_through_llm_input"
    assert qwen3.is_compound
    assert [item.name for item in qwen3.ordered_targets()] == [
        "main_image_features",
        "deepstack_image_features",
    ]
    assert qwen3.metadata["shared_main_deepstack_delta"] is True
