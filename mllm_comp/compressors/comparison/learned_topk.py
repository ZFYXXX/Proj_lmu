"""Native Qwen2.5 input-only bindings for learned and random selection."""
from pathlib import Path
import hashlib
import json

from .qwen_projector import QwenProjectorComparison


class QwenLearnedTopKComparison(QwenProjectorComparison):
    def __init__(self, model, config, *, runtime=None, selector_module=None, random=False, phase="generate"):
        import torch
        from mllm_comp.compressors.image.learned_visual_topk import LearnedVisualTopK, LearnedVisualTopKSpec, load_selector
        config = dict(config)
        allowed = {"target_count", "width", "seed", "checkpoint_path", "identity_path", "training_mode"}
        if set(config) - allowed:
            raise ValueError(f"Unknown learned TopK keys: {sorted(set(config) - allowed)}")
        for key, default in (("target_count", 128), ("width", 256), ("seed", 42)):
            config.setdefault(key, default)
            if isinstance(config[key], bool) or not isinstance(config[key], int) or config[key] < (0 if key == "seed" else 1):
                raise ValueError(f"Invalid selector {key}")
        super().__init__(model, "dppruner", {"tokens_per_crop": config["target_count"]}, "qwen25_vl")
        self.method = "random_visual_pruning" if random else "learned_visual_topk"
        self.random_selection = random
        self.selector_config = config
        self.soft_training = config.get("training_mode", "hard") == "soft"
        if config.get("training_mode", "hard") not in {"soft", "hard"} or (self.soft_training and str(phase) != "forward"):
            raise ValueError("Soft training is forward-only; generation must physically compact")
        self.base_identity = None
        if random:
            self.selector = None
            return
        device = self.decoder.embed_tokens.weight.device
        if selector_module is not None:
            self.selector = selector_module.to(device)
        elif config.get("checkpoint_path"):
            if not config.get("identity_path") or runtime is None:
                raise ValueError("Trained selector needs pinned identity and runtime")
            identity = json.loads(Path(config["identity_path"]).read_text())
            if Path(runtime.model_path_or_id).resolve() != Path(identity["checkpoint"]).resolve():
                raise ValueError("Selector/runtime checkpoint mismatch")
            self.base_identity = identity["identity_sha256"]
            self.selector = load_selector(config["checkpoint_path"], base_identity=self.base_identity, device=device)
        else:
            self.selector = LearnedVisualTopK(LearnedVisualTopKSpec(self.decoder.config.hidden_size,
                config["width"], config["target_count"], config["seed"])).to(device)
        if self.selector.spec.hidden_size != self.decoder.config.hidden_size:
            raise ValueError("Selector and decoder hidden dimensions disagree")
        self.selector.train(self.soft_training).requires_grad_(self.soft_training)

    def prepare_qwen_embeddings(self, call, hidden):
        import torch
        from mllm_comp.compressors.image.learned_visual_topk import hard_topk_indices
        visual_ids = self.incoming.get().vision[0].nonzero(as_tuple=True)[0]
        features = hidden[0].index_select(0, visual_ids)
        n, k = len(features), self.token_budget(len(features))
        stats = {"trained": bool(self.selector_config.get("checkpoint_path")),
            "checkpoint_path": self.selector_config.get("checkpoint_path"),
            "seed": self.selector_config["seed"], "reads_question": False}
        if self.soft_training:
            output = self.selector(features, soft=True)
            if output["target_count"] != k:
                raise ValueError("Soft training budget differs from selector spec")
            hidden = hidden.clone().index_copy(1, visual_ids, output["tokens"][None])
            call["inputs_embeds"] = hidden
            call["input_ids"] = None
            self.soft_output = output
            self.last_selection = {"source_count": n, "result_count": n, "soft_mass": float(output["soft_weights"].detach().sum()),
                "selected_indices": output["indices"].tolist(), "training_mode": "soft_full_sequence", **stats}
            return hidden
        if k == n:
            selected = torch.arange(n, device=hidden.device)
        elif self.random_selection:
            # Deterministic per-image seed independent of request order/cache.
            fingerprint = hashlib.sha256(features[:, :min(32, features.shape[1])].detach().float().cpu().contiguous().numpy().tobytes()).hexdigest()
            generator = torch.Generator(device=hidden.device).manual_seed((int(fingerprint[:16], 16) + self.selector_config["seed"]) % (2**63 - 1))
            selected = torch.randperm(n, generator=generator, device=hidden.device)[:k].sort().values
            stats["image_selection_fingerprint"] = fingerprint
        else:
            with torch.no_grad():
                output = self.selector(features)
            selected = hard_topk_indices(output["scores"], k)
        return self.compact_selection(call, hidden, selected, {**stats, "training_mode": "hard_compact"})

    def runtime_metadata(self):
        result = super().runtime_metadata()
        result.update(selector_config=self.selector_config, base_identity_sha256=self.base_identity,
            inference_selection="source_order_hard_topk", training_physical_compression=False)
        return result


def create_executor(*, model, stage, runtime=None, execution_phase="generate", **kwargs):
    return QwenLearnedTopKComparison(model, stage.config, runtime=runtime,
        random=stage.algorithm == "random_visual_pruning", phase=execution_phase)


def kernel(*args, **kwargs):
    from mllm_comp.compressors.image.learned_visual_topk import LearnedVisualTopK
    return LearnedVisualTopK(*args, **kwargs)


def register_contracts(registry):
    from mllm_comp.compressors.contracts import AlgorithmDescriptor, ModalityVariantDescriptor, BackbonePlacementBinding, BindingTarget
    from mllm_comp.core.vocabulary import CompressionMethod, Placement, TokenOperation, Modality, ExecutionPhase
    from mllm_comp.core.support import SupportUnit, SupportUnitId
    image = frozenset({Modality.IMAGE})
    for name in ("learned_visual_topk", "random_visual_pruning"):
        binding_name = f"{name}.image.qwen25_vl"
        registry.register_algorithm(AlgorithmDescriptor(name=name, version="v1.0.0", method=CompressionMethod.TOKEN_COMPRESSION,
            kernel=kernel, supported_placements=frozenset({Placement.PROJECTOR_LEVEL}), operations=frozenset({TokenOperation.DROP}),
            metadata={"trained": name == "learned_visual_topk", "visual_only": True, "soft_training_dense": True,
                "claim": "New visual-only MLP selector; not a full VisionSelector reproduction"}))
        registry.register_variant(ModalityVariantDescriptor(name=f"{name}.image", algorithm=name, target_modalities=image))
        registry.register_binding(BackbonePlacementBinding(name=binding_name, backbone="qwen25_vl", variant=f"{name}.image",
            targets=(BindingTarget(name="visual_selection", placement=Placement.PROJECTOR_LEVEL,
                target_site="post_vision_merger_pre_llm_scatter", target_modalities=image,
                required_handles=frozenset({"vision_projector", "llm"})),),
            supported_phases=frozenset({ExecutionPhase.FORWARD, ExecutionPhase.GENERATE}), executor_factory=create_executor,
            legalizer="qwen25_vl.original_sparse_mrope_compact_cache", exclusive_runtime_slot="image_comparison",
            conflicts=frozenset(b.name for b in registry.bindings() if b.backbone == "qwen25_vl"),
            metadata={"single_image_only": True, "real_checkpoint_verified": False}))
        registry.extend_support(SupportUnit(identity=SupportUnitId(algorithm=name, variant=f"{name}.image",
            backbone="qwen25_vl", target_modality=Modality.IMAGE, phase=phase), binding=binding_name)
            for phase in (ExecutionPhase.FORWARD, ExecutionPhase.GENERATE))
