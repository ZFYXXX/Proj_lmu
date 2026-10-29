"""Fixed-K input-conditioned selector using the tested Qwen word lifecycle."""
import json
from pathlib import Path

from .word_topk import QwenWordTopKComparison
from mllm_comp.compressors.image.input_adaptive_topk import (
    InputAdaptiveTopK, InputAdaptiveTopKSpec, load_selector,
)


class QwenInputAdaptiveTopKComparison(QwenWordTopKComparison):
    def __init__(self, model, config, *, runtime=None, selector_module=None, phase="generate"):
        config = dict(config)
        gate_width = config.pop("gate_width", 64)
        if selector_module is None:
            if config.get("checkpoint_path"):
                if runtime is None or not config.get("identity_path"):
                    raise ValueError("Input-adaptive checkpoint requires pinned runtime/base identity")
                identity = json.loads(Path(config["identity_path"]).read_text())
                if Path(runtime.model_path_or_id).resolve() != Path(identity["checkpoint"]).resolve():
                    raise ValueError("Wrong input-adaptive backbone")
                selector_module = load_selector(config["checkpoint_path"], base_identity=identity["identity_sha256"],
                    device=model.get_input_embeddings().weight.device)
            else:
                selector_module = InputAdaptiveTopK(InputAdaptiveTopKSpec(
                    model.get_input_embeddings().embedding_dim, width=config.get("width", 256),
                    target_count=config.get("target_count", 128), seed=config.get("seed", 42),
                    match_width=config.get("match_width", 128), gate_width=gate_width,
                    template_token_ids=config.get("template_token_ids", ())))
        if not isinstance(selector_module, InputAdaptiveTopK) or selector_module.spec.gate_width != gate_width:
            raise ValueError("Wrong input-adaptive gate specification")
        super().__init__(model, config, runtime=runtime, selector_module=selector_module, phase=phase)
        self.method = "input_adaptive_topk"


def create_executor(*, model, stage, runtime=None, execution_phase="generate", **kwargs):
    return QwenInputAdaptiveTopKComparison(model, stage.config, runtime=runtime, phase=execution_phase)


def register_contracts(registry):
    from dataclasses import replace
    from mllm_comp.compressors.contracts import AlgorithmDescriptor, ModalityVariantDescriptor
    from mllm_comp.core.vocabulary import CompressionMethod, Placement, TokenOperation, Modality, ExecutionPhase
    from mllm_comp.core.support import SupportUnit, SupportUnitId
    name = "input_adaptive_topk"
    registry.register_algorithm(AlgorithmDescriptor(name=name, version="v1.0.0",
        method=CompressionMethod.TOKEN_COMPRESSION, kernel=InputAdaptiveTopK,
        supported_placements=frozenset({Placement.PROJECTOR_LEVEL}), operations=frozenset({TokenOperation.DROP}),
        metadata={"claim": "Input-conditioned relevance strength; fixed budget; exploratory trained head"}))
    registry.register_variant(ModalityVariantDescriptor(name=f"{name}.image", algorithm=name,
        target_modalities=frozenset({Modality.IMAGE}), conditioning_modalities=frozenset({Modality.TEXT})))
    source = next(b for b in registry.bindings() if b.name == "word_visual_topk.image.qwen25_vl")
    binding = f"{name}.image.qwen25_vl"
    registry.register_binding(replace(source, name=binding, variant=f"{name}.image", executor_factory=create_executor,
        conflicts=frozenset(b.name for b in registry.bindings() if b.backbone == "qwen25_vl")))
    registry.extend_support(SupportUnit(identity=SupportUnitId(algorithm=name, variant=f"{name}.image",
        backbone="qwen25_vl", target_modality=Modality.IMAGE, phase=phase), binding=binding)
        for phase in (ExecutionPhase.FORWARD, ExecutionPhase.GENERATE))
