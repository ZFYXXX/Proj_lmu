"""Question/image-context arms share the native sparse-MRoPE cache legalizer."""
import json
from pathlib import Path

from .learned_topk import QwenLearnedTopKComparison


class QwenContextTopKComparison(QwenLearnedTopKComparison):
    def __init__(self, model, config, *, runtime=None, selector_module=None, phase="generate"):
        from mllm_comp.compressors.image.context_visual_topk import ContextVisualTopK, ContextVisualTopKSpec, load_selector
        config = dict(config)
        mode = config.pop("context_mode", "question")
        device = model.get_input_embeddings().weight.device
        if selector_module is None:
            if config.get("checkpoint_path"):
                if not config.get("identity_path") or runtime is None:
                    raise ValueError("Context checkpoint requires pinned runtime and base identity")
                identity = json.loads(Path(config["identity_path"]).read_text())
                if Path(runtime.model_path_or_id).resolve() != Path(identity["checkpoint"]).resolve():
                    raise ValueError("Context selector/runtime checkpoint mismatch")
                selector_module = load_selector(config["checkpoint_path"], base_identity=identity["identity_sha256"], device=device)
            else:
                selector_module = ContextVisualTopK(ContextVisualTopKSpec(model.get_input_embeddings().embedding_dim,
                    config.get("width", 256), config.get("target_count", 128), config.get("seed", 42), mode))
        if not isinstance(selector_module, ContextVisualTopK) or selector_module.spec.context_mode != mode:
            raise ValueError("Context selector specification mismatch")
        for key, default in (("width", 256), ("target_count", 128), ("seed", 42)):
            if getattr(selector_module.spec, key) != config.get(key, default):
                raise ValueError(f"Context checkpoint/config {key} mismatch")
        super().__init__(model, config, runtime=runtime, selector_module=selector_module, phase=phase)
        self.method = "context_visual_topk"
        self.selector_config["context_mode"] = mode
        if config.get("identity_path"):
            self.base_identity = json.loads(Path(config["identity_path"]).read_text())["identity_sha256"]

    def prepare_source_inputs(self, call):
        from mllm_comp.compressors.image.context_visual_topk import prompt_question
        super().prepare_source_inputs(call)
        # Source is reset on every prefill by the parent lifecycle. Never read
        # newly generated assistant tokens or cache context across requests.
        if self.selector.spec.context_mode == "question":
            self.source.get()["question"], self.source.get()["question_audit"] = prompt_question(self.model, call)

    def prepare_qwen_embeddings(self, call, hidden):
        import torch
        visual_ids = self.incoming.get().vision[0].nonzero(as_tuple=True)[0]
        features = hidden[0].index_select(0, visual_ids)
        source = self.source.get()
        stats = {"trained": bool(self.selector_config.get("checkpoint_path")),
            "checkpoint_path": self.selector_config.get("checkpoint_path"),
            "reads_question": self.selector.spec.context_mode == "question",
            "context_mode": self.selector.spec.context_mode, **source.get("question_audit", {})}
        if self.soft_training:
            output = self.selector(features, question=source.get("question"), soft=True)
            hidden = hidden.clone().index_copy(1, visual_ids, output["tokens"][None])
            call.update(inputs_embeds=hidden, input_ids=None)
            self.soft_output = output
            self.last_selection = {"source_count": len(features), "result_count": len(features),
                "soft_mass": float(output["soft_weights"].detach().sum()),
                "selected_indices": output["indices"].tolist(), "training_mode": "soft_full_sequence", **stats}
            return hidden
        with torch.no_grad():
            output = self.selector(features, question=source.get("question"))
        return self.compact_selection(call, hidden, output["indices"], {**stats, "training_mode": "hard_compact"})


def create_executor(*, model, stage, runtime=None, execution_phase="generate", **kwargs):
    return QwenContextTopKComparison(model, stage.config, runtime=runtime, phase=execution_phase)


def kernel(*args, **kwargs):
    from mllm_comp.compressors.image.context_visual_topk import ContextVisualTopK
    return ContextVisualTopK(*args, **kwargs)


def register_contracts(registry):
    from mllm_comp.compressors.contracts import AlgorithmDescriptor, ModalityVariantDescriptor, BackbonePlacementBinding, BindingTarget
    from mllm_comp.core.vocabulary import CompressionMethod, Placement, TokenOperation, Modality, ExecutionPhase
    from mllm_comp.core.support import SupportUnit, SupportUnitId
    name = "context_visual_topk"
    image = frozenset({Modality.IMAGE})
    binding = f"{name}.image.qwen25_vl"
    registry.register_algorithm(AlgorithmDescriptor(name=name, version="v1.0.0", method=CompressionMethod.TOKEN_COMPRESSION,
        kernel=kernel, supported_placements=frozenset({Placement.PROJECTOR_LEVEL}), operations=frozenset({TokenOperation.DROP}),
        metadata={"trained": True, "fixed_count": True,
            "claim": "Prompt-input-embedding conditioned scorer, not a paper reproduction or novelty claim"}))
    registry.register_variant(ModalityVariantDescriptor(name=f"{name}.image", algorithm=name, target_modalities=image,
        conditioning_modalities=frozenset({Modality.TEXT})))
    registry.register_binding(BackbonePlacementBinding(name=binding, backbone="qwen25_vl", variant=f"{name}.image",
        targets=(BindingTarget(name="context_selection", placement=Placement.PROJECTOR_LEVEL,
            target_site="post_vision_merger_pre_llm_scatter", target_modalities=image,
            required_handles=frozenset({"vision_projector", "llm"})),),
        supported_phases=frozenset({ExecutionPhase.FORWARD, ExecutionPhase.GENERATE}), executor_factory=create_executor,
        legalizer="qwen25_vl.original_sparse_mrope_compact_cache", exclusive_runtime_slot="image_comparison",
        conflicts=frozenset(b.name for b in registry.bindings() if b.backbone == "qwen25_vl"),
        metadata={"single_image_only": True, "real_checkpoint_verified": False}))
    registry.extend_support(SupportUnit(identity=SupportUnitId(algorithm=name, variant=f"{name}.image",
        backbone="qwen25_vl", target_modality=Modality.IMAGE, phase=phase), binding=binding)
        for phase in (ExecutionPhase.FORWARD, ExecutionPhase.GENERATE))
