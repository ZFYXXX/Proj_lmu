"""Post-merger context recovery with unchanged protected selection and positions."""
from dataclasses import asdict, replace

from .protected_topk import QwenProtectedTopKComparison
from mllm_comp.compressors.image.visual_token_recovery import (
    RecoverySpec, RecoveredProtectedSelector, recover_visual_tokens,
)


class QwenRecoveredTopKComparison(QwenProtectedTopKComparison):
    def __init__(self, model, config, **kwargs):
        config = dict(config)
        recovery = RecoverySpec(**config.pop("recovery", {}))
        super().__init__(model, config, **kwargs)
        self.selector = RecoveredProtectedSelector(self.selector.base, self.selector.policy, recovery)
        self.method = "recovered_visual_topk"
        self.recovery_config = asdict(recovery)

    def compact_word_selection(self, call, hidden, visual_ids, output, stats):
        # The word parent deliberately only gathers in its default hard path.
        # Install the recovered K features BEFORE that same legalizer gathers;
        # otherwise a selector that returns new tokens is silently ignored.
        selected_positions = visual_ids.index_select(0, output["indices"])
        hidden = hidden.clone().index_copy(1, selected_positions, output["tokens"][None])
        return self.compact_selection(call, hidden, output["indices"],
            {**stats, "training_mode": "hard_compact", "recovered_features_injected": True})

    def runtime_metadata(self):
        return {**super().runtime_metadata(), "recovery": self.recovery_config,
                "inference_selection": "unchanged_protected_indices_with_context_recovery"}


def create_executor(*, model, stage, runtime=None, execution_phase="generate", **kwargs):
    return QwenRecoveredTopKComparison(model, stage.config, runtime=runtime, phase=execution_phase)


def register_contracts(registry):
    from mllm_comp.compressors.contracts import AlgorithmDescriptor, ModalityVariantDescriptor
    from mllm_comp.core.vocabulary import CompressionMethod, Placement, TokenOperation, Modality, ExecutionPhase
    from mllm_comp.core.support import SupportUnit, SupportUnitId
    name = "recovered_visual_topk"
    registry.register_algorithm(AlgorithmDescriptor(name=name, version="v1.0.0",
        method=CompressionMethod.TOKEN_COMPRESSION, kernel=recover_visual_tokens,
        supported_placements=frozenset({Placement.PROJECTOR_LEVEL}),
        operations=frozenset({TokenOperation.DROP, TokenOperation.MERGE}),
        metadata={"claim": "Exploratory budget-neutral recovery; not official AdaptMerge/TRIM/SpecFlow",
                  "recovery_trained": False}))
    registry.register_variant(ModalityVariantDescriptor(name=f"{name}.image", algorithm=name,
        target_modalities=frozenset({Modality.IMAGE}), conditioning_modalities=frozenset({Modality.TEXT})))
    source = next(b for b in registry.bindings() if b.name == "protected_visual_topk.image.qwen25_vl")
    binding = f"{name}.image.qwen25_vl"
    registry.register_binding(replace(source, name=binding, variant=f"{name}.image",
        executor_factory=create_executor,
        conflicts=frozenset(b.name for b in registry.bindings() if b.backbone == "qwen25_vl")))
    registry.extend_support(SupportUnit(identity=SupportUnitId(algorithm=name, variant=f"{name}.image",
        backbone="qwen25_vl", target_modality=Modality.IMAGE, phase=phase), binding=binding)
        for phase in (ExecutionPhase.FORWARD, ExecutionPhase.GENERATE))
