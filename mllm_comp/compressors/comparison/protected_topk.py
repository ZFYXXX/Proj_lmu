"""Protected evidence diagnostic using the existing sparse-M-RoPE legalizer."""
from dataclasses import asdict, replace
import json
import math
from pathlib import Path

from .word_topk import QwenWordTopKComparison
from mllm_comp.compressors.image.protected_visual_topk import (
    ProtectedTopKSpec, ProtectedWordSelector, protected_context_indices,
)


class QwenProtectedTopKComparison(QwenWordTopKComparison):
    def __init__(self, model, config, *, runtime=None, selector_module=None, phase="generate"):
        config = dict(config)
        proportional = 'retention_ratio' in config
        ratio = config.pop('retention_ratio', None)
        if proportional:
            if 'target_count' in config or isinstance(ratio, bool) or not isinstance(ratio, (int, float)):
                raise ValueError('Specify a numeric retention_ratio instead of a fixed target_count')
            if not math.isfinite(ratio) or not 0 < ratio <= 1:
                raise ValueError('retention_ratio must be finite and in (0, 1]')
        policy = ProtectedTopKSpec(**config.pop("protection", {}))
        if config.get("training_mode", "hard") != "hard" or config.get("visual_only", False):
            raise ValueError("Protected diagnostic is inference-only; no visual-only bypass")
        super().__init__(model, config, runtime=runtime, selector_module=selector_module, phase=phase)
        # The trained checkpoint keeps its K128 spec. Only the inference policy
        # receives the new per-image budget; no state/spec mutation or retraining.
        if proportional:
            self.retention_ratio = float(ratio)
        self.selector = ProtectedWordSelector(self.selector, policy)
        self.method = "protected_visual_topk"
        self.protection_config = asdict(policy)
        if config.get("identity_path"):
            self.base_identity = json.loads(Path(config["identity_path"]).read_text())["identity_sha256"]

    def select_word_tokens(self, features, question):
        return self.selector(features, question=question,
                             target_count=self.token_budget(len(features)))

    def runtime_metadata(self):
        result = super().runtime_metadata()
        result.update(protection=self.protection_config, new_training=False,
                      official_anchorprune_reproduction=False)
        return result


def create_executor(*, model, stage, runtime=None, execution_phase="generate", **kwargs):
    return QwenProtectedTopKComparison(model, stage.config, runtime=runtime, phase=execution_phase)


def register_contracts(registry):
    from mllm_comp.compressors.contracts import AlgorithmDescriptor, ModalityVariantDescriptor
    from mllm_comp.core.vocabulary import CompressionMethod, Placement, TokenOperation, Modality, ExecutionPhase
    from mllm_comp.core.support import SupportUnit, SupportUnitId
    name = "protected_visual_topk"
    registry.register_algorithm(AlgorithmDescriptor(name=name, version="v1.0.0",
        method=CompressionMethod.TOKEN_COMPRESSION, kernel=protected_context_indices,
        supported_placements=frozenset({Placement.PROJECTOR_LEVEL}),
        operations=frozenset({TokenOperation.DROP}),
        metadata={"claim": "Fixed-budget diagnostic; not official AnchorPrune or a new trained model"}))
    registry.register_variant(ModalityVariantDescriptor(name=f"{name}.image", algorithm=name,
        target_modalities=frozenset({Modality.IMAGE}), conditioning_modalities=frozenset({Modality.TEXT})))
    source = next(b for b in registry.bindings() if b.name == "word_visual_topk.image.qwen25_vl")
    binding = f"{name}.image.qwen25_vl"
    registry.register_binding(replace(source, name=binding, variant=f"{name}.image",
        executor_factory=create_executor,
        conflicts=frozenset(b.name for b in registry.bindings() if b.backbone == "qwen25_vl")))
    registry.extend_support(SupportUnit(identity=SupportUnitId(algorithm=name, variant=f"{name}.image",
        backbone="qwen25_vl", target_modality=Modality.IMAGE, phase=phase), binding=binding)
        for phase in (ExecutionPhase.FORWARD, ExecutionPhase.GENERATE))
