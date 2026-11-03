"""Register implemented comparison methods without promoting unverified bindings."""
from functools import partial


def comparison_kernel(method, *args, **kwargs):
    from importlib import import_module
    module, function = {
        "cdpruner": ("cdpruner", "select_cdpruner_tokens"),
        "dppruner": ("cdpruner", "select_cdpruner_tokens"),
        "eadp": ("eadp", "select_eadp_indices"),
        "hiprune": ("hiprune", "select_hiprune_indices"),
        "prunesid": ("prunesid", "select_prunesid_indices"),
    }[method]
    return getattr(import_module("." + module, __package__), function)(*args, **kwargs)


def create_executor(*, model, stage, binding, execution_phase, **kwargs):
    if binding.backbone != "llava_onevision":
        from .qwen_projector import QwenProjectorComparison
        return QwenProjectorComparison(model, stage.algorithm, stage.config, binding.backbone)
    from .llava import LlavaProjectorComparison
    return LlavaProjectorComparison(model, stage, binding, execution_phase)


def create_operation_executor(*, model, stage, binding, runtime=None, **kwargs):
    if stage.algorithm == "dop" and stage.config.get("token_reducer", "vispruner") == "vispruner" and binding.backbone != "llava_onevision":
        from .qwen_projector import QwenProjectorComparison
        return QwenProjectorComparison(model, "dop", stage.config, binding.backbone)
    from .operations import OperationComparison
    return OperationComparison(model, stage.algorithm, stage.config, binding.backbone,
        checkpoint_path=getattr(runtime, "model_path_or_id", None))


def operation_kernel(*args, **kwargs):
    from .operations import OperationComparison
    return OperationComparison(*args, **kwargs)


def create_decoder_executor(*, stage, binding, contract_projection=None, **kwargs):
    from mllm_comp.compressors.multimodal_registry import builtin_multimodal_compression_contracts
    from mllm_comp.pipeline.v2_validation import validate_v2_stage_contract
    from mllm_comp.backbones.llava_onevision.image_compression import create_llava_onevision_image_compression_executor
    from mllm_comp.backbones.qwen25_vl.image_compression import create_qwen25_vl_image_compression_executor
    from mllm_comp.backbones.qwen3_vl.compression_adapter import create_qwen3_vl_image_compression_executor
    if contract_projection is None:
        contract_projection = validate_v2_stage_contract(stage,
            registry=builtin_multimodal_compression_contracts(), backbone_name=binding.backbone)
    factory = {"llava_onevision": create_llava_onevision_image_compression_executor,
        "qwen25_vl": create_qwen25_vl_image_compression_executor,
        "qwen3_vl": create_qwen3_vl_image_compression_executor}[binding.backbone]
    native = factory(stage=stage, binding=binding,
        contract_projection=contract_projection, **kwargs)
    if stage.algorithm == "core_matching":
        from .native_decoder import CoreMatchingActivation
        model = kwargs["model"]
        decoder = model.get_model() if binding.backbone == "llava_onevision" else model.model.language_model
        return CoreMatchingActivation(native, decoder)
    return native


def vlm_pruner_kernel(*args, **kwargs):
    from .vlm_pruner import build_vlm_pruner_plan
    return build_vlm_pruner_plan(*args, **kwargs)


def register_comparison_contracts(registry):
    from mllm_comp.compressors.contracts import (
        AlgorithmDescriptor, ModalityVariantDescriptor, BackbonePlacementBinding, BindingTarget,
        TokenUnit,
    )
    from mllm_comp.core.vocabulary import (
        CompressionMethod, Placement, TokenOperation, Modality, ExecutionPhase,
        ModalityCouplingPolicy,
    )
    from mllm_comp.core.support import SupportUnit, SupportUnitId

    image = frozenset({Modality.IMAGE})
    for method in ("cdpruner", "dppruner", "eadp", "hiprune"):
        registry.register_algorithm(AlgorithmDescriptor(
            name=method, version="ov-source-port-v1", method=CompressionMethod.TOKEN_COMPRESSION,
            kernel=partial(comparison_kernel, method),
            supported_placements=frozenset({Placement.PROJECTOR_LEVEL}),
            operations=frozenset({TokenOperation.DROP}),
            metadata=({"source": "https://github.com/SJTU-DeepVisionLab/EADP",
                "source_commit": "e1a08801461c181871b9ff1cb0803b8e9966f083",
                "trained": False, "qwen_script_defaults": {"alpha": 0.5, "beta": 2.0},
                "paper_implementation_alignment": "official_qwen_selector_native_rectangular_resolution_adaptation"}
                if method == "eadp" else
                {"source": "OV对比方法.zip", "paper_implementation_alignment": "native_anyres_unpad_adaptation"}),
        ))
        registry.register_variant(ModalityVariantDescriptor(
            name=f"{method}.image", algorithm=method, target_modalities=image,
            conditioning_modalities=frozenset({Modality.TEXT}) if method in {"eadp", "cdpruner"} else frozenset(),
            coupling_policy=ModalityCouplingPolicy.INDEPENDENT, token_unit=TokenUnit.FEATURE_TOKEN,
            config_defaults={},
        ))
        backbone = "llava_onevision"
        name = f"{method}.image.{backbone}"
        registry.register_binding(BackbonePlacementBinding(
            name=name, backbone=backbone, variant=f"{method}.image",
            targets=(BindingTarget(name="projector_selection", placement=Placement.PROJECTOR_LEVEL,
                target_site="post_anyres_pack_pre_llm_splice", target_modalities=image,
                required_handles=frozenset({"vision_projector", "llm"})),),
            supported_phases=frozenset({ExecutionPhase.FORWARD, ExecutionPhase.GENERATE}), executor_factory=create_executor,
            legalizer="llava_onevision.native_anyres_unpad", exclusive_runtime_slot="image_comparison",
            conflicts=frozenset(b.name for b in registry.bindings() if b.backbone == backbone),
            metadata={"image_only": True, "single_image_only": True},
        ))
        registry.extend_support(SupportUnit(identity=SupportUnitId(
            algorithm=method, variant=f"{method}.image", backbone=backbone,
            target_modality=Modality.IMAGE, phase=phase), binding=name)
            for phase in (ExecutionPhase.FORWARD, ExecutionPhase.GENERATE))

    from mllm_comp.core.vocabulary import OperationKind
    for method in ("dop", "shortv", "redundancy_lens", "yopo", "structural_pruning",
                   "short_lvlm", "adallava", "adallava_l", "adallava_h", "skip_vision"):
        registry.register_algorithm(AlgorithmDescriptor(
            name=method, version="ov-source-port-v1", method=CompressionMethod.OPERATION_COMPRESSION,
            kernel=operation_kernel, supported_placements=frozenset({Placement.LLM_LEVEL}),
            operations=frozenset({OperationKind.SKIP, OperationKind.MASK}),
            metadata={"source": "OV对比方法.zip", "physical_acceleration_verified": False},
        ))
        registry.register_variant(ModalityVariantDescriptor(
            name=f"{method}.image", algorithm=method, target_modalities=image,
            coupling_policy=ModalityCouplingPolicy.INDEPENDENT, token_unit=None))
        backbone = "llava_onevision"
        name = f"{method}.image.{backbone}"
        registry.register_binding(BackbonePlacementBinding(
            name=name, backbone=backbone, variant=f"{method}.image",
            targets=(BindingTarget(name="decoder_operations", placement=Placement.LLM_LEVEL,
                target_site="llm", target_modalities=image, required_handles=frozenset({"llm"})),),
            supported_phases=frozenset({ExecutionPhase.PREPARE}),
            executor_factory=create_operation_executor, legalizer="comparison.native_decoder_layout",
            exclusive_runtime_slot="image_comparison", executor_family="llm_operation",
            conflicts=frozenset(b.name for b in registry.bindings() if b.backbone == backbone),
            metadata={"image_only": True, "single_image_only": True},
        ))
        registry.extend_support([SupportUnit(identity=SupportUnitId(
            algorithm=method, variant=f"{method}.image", backbone=backbone,
            target_modality=Modality.IMAGE, phase=ExecutionPhase.PREPARE), binding=name)])

    from .decoder import VLMPrunerMethod, CoreMatchingMethod, TwigVLMMethod
    for strategy in (VLMPrunerMethod, CoreMatchingMethod, TwigVLMMethod):
        method = strategy.algorithm
        registry.register_algorithm(AlgorithmDescriptor(name=method, version="ov-source-port-v1",
            method=CompressionMethod.TOKEN_COMPRESSION, kernel=strategy,
            supported_placements=frozenset({Placement.LLM_LEVEL}),
            operations=frozenset({TokenOperation.DROP, TokenOperation.MERGE})))
        registry.register_variant(ModalityVariantDescriptor(name=f"{method}.image", algorithm=method,
            target_modalities=image, coupling_policy=ModalityCouplingPolicy.INDEPENDENT,
            token_unit=TokenUnit.LLM_TOKEN))
        backbone = "llava_onevision"
        name = f"{method}.image.{backbone}"
        registry.register_binding(BackbonePlacementBinding(name=name, backbone=backbone,
            variant=f"{method}.image", targets=(BindingTarget(name="decoder_selection",
                placement=Placement.LLM_LEVEL, target_site=strategy.target_site,
                target_modalities=image, required_handles=frozenset({"llm"})),),
            supported_phases=frozenset({ExecutionPhase.FORWARD, ExecutionPhase.GENERATE}),
            executor_factory=create_decoder_executor, strategy_factory=strategy,
            legalizer="llava_onevision.shared_decoder_lifecycle", exclusive_runtime_slot="image_comparison",
            conflicts=frozenset(b.name for b in registry.bindings() if b.backbone == backbone)))
        registry.extend_support(SupportUnit(identity=SupportUnitId(algorithm=method,
            variant=f"{method}.image", backbone=backbone, target_modality=Modality.IMAGE,
            phase=phase), binding=name) for phase in (ExecutionPhase.FORWARD, ExecutionPhase.GENERATE))

    from dataclasses import replace
    for source in tuple(registry.bindings()):
        if source.backbone != "llava_onevision" or source.exclusive_runtime_slot != "image_comparison":
            continue
        for backbone in ("qwen25_vl", "qwen3_vl"):
            method = source.variant.split(".")[0]
            name = f"{method}.image.{backbone}"
            targets = source.targets
            phases = source.supported_phases
            if source.executor_factory is create_executor:
                targets = tuple(replace(target, target_site="post_vision_merger_pre_llm_scatter" if backbone == "qwen25_vl" else "post_visual_merger_with_deepstack") for target in targets)
                phases = frozenset({ExecutionPhase.FORWARD, ExecutionPhase.GENERATE})
            registry.register_binding(replace(source, name=name, backbone=backbone, targets=targets,
                supported_phases=phases,
                legalizer=f"{backbone}.native_comparison_layout",
                conflicts=frozenset(b.name for b in registry.bindings() if b.backbone == backbone)))
            registry.extend_support(SupportUnit(identity=SupportUnitId(algorithm=method,
                variant=source.variant, backbone=backbone, target_modality=Modality.IMAGE,
                phase=phase), binding=name) for phase in phases)
    from .anchor_adapter import register_contracts as register_anchor_adapter
    register_anchor_adapter(registry)
    from .importance_anchor_adapter import register_contracts as register_importance_anchor
    register_importance_anchor(registry)
    register_prunesid_contracts(registry)
    from .learned_topk import register_contracts as register_learned_topk
    register_learned_topk(registry)
    from .context_topk import register_contracts as register_context_topk
    register_context_topk(registry)
    from .word_topk import register_contracts as register_word_topk
    register_word_topk(registry)
    from .input_adaptive_topk import register_contracts as register_input_adaptive_topk
    register_input_adaptive_topk(registry)
    from .query_interaction_topk import register_contracts as register_query_interaction_topk
    register_query_interaction_topk(registry)
    from .protected_topk import register_contracts as register_protected_topk
    register_protected_topk(registry)
    from .recovered_topk import register_contracts as register_recovered_topk
    register_recovered_topk(registry)
    from .qwen35_topk import register_contracts as register_qwen35_topk
    register_qwen35_topk(registry)
    from .qwen35_reference import register_contracts as register_qwen35_reference
    register_qwen35_reference(registry)
    from .qwen35_image_context import register_contracts as register_qwen35_image_context
    register_qwen35_image_context(registry)
    from .selector_condition import register_contracts as register_selector_condition
    register_selector_condition(registry)
    from .selector_mechanism import register_contracts as register_selector_mechanism
    register_selector_mechanism(registry)
    from .context_topk_ot import register_contracts as register_context_topk_ot
    register_context_topk_ot(registry)
    from .bifocal_two_stage import register_contracts as register_bifocal_two_stage
    register_bifocal_two_stage(registry)


def register_prunesid_contracts(registry):
    """Only claim the implemented Qwen2.5 image binding, not a Cartesian port."""
    from mllm_comp.compressors.contracts import AlgorithmDescriptor, ModalityVariantDescriptor, BackbonePlacementBinding, BindingTarget
    from mllm_comp.core.vocabulary import CompressionMethod, Placement, TokenOperation, Modality, ExecutionPhase
    from mllm_comp.core.support import SupportUnit, SupportUnitId
    image = frozenset({Modality.IMAGE})
    registry.register_algorithm(AlgorithmDescriptor(name="prunesid", version="official-4ef17d9-qwen25-v1",
        method=CompressionMethod.TOKEN_COMPRESSION, kernel=partial(comparison_kernel, "prunesid"),
        supported_placements=frozenset({Placement.PROJECTOR_LEVEL}), operations=frozenset({TokenOperation.DROP}),
        metadata={"source": "https://github.com/ZhengyaoFang/PruneSID",
            "source_commit": "4ef17d96b390027f31243bc7e1d27bdee77792a5", "trained": False,
            "paper_implementation_alignment": "qwen2_selector_qwen25_native_layout_fixed_budget_adaptation"}))
    registry.register_variant(ModalityVariantDescriptor(name="prunesid.image", algorithm="prunesid", target_modalities=image))
    name = "prunesid.image.qwen25_vl"
    registry.register_binding(BackbonePlacementBinding(name=name, backbone="qwen25_vl", variant="prunesid.image",
        targets=(BindingTarget(name="prunesid_post_merger", placement=Placement.PROJECTOR_LEVEL,
            target_site="post_vision_merger_pre_llm_scatter", target_modalities=image,
            required_handles=frozenset({"vision_projector", "llm"})),),
        supported_phases=frozenset({ExecutionPhase.FORWARD, ExecutionPhase.GENERATE}), executor_factory=create_executor,
        legalizer="qwen25_vl.native_comparison_layout", exclusive_runtime_slot="image_comparison",
        conflicts=frozenset(b.name for b in registry.bindings() if b.backbone == "qwen25_vl"),
        metadata={"single_image_only": True, "trained": False, "real_checkpoint_verified": False}))
    registry.extend_support(SupportUnit(identity=SupportUnitId(algorithm="prunesid", variant="prunesid.image",
        backbone="qwen25_vl", target_modality=Modality.IMAGE, phase=phase), binding=name)
        for phase in (ExecutionPhase.FORWARD, ExecutionPhase.GENERATE))
