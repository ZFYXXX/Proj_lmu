"""Expose the previously trained anchor MLP through the native eval lifecycle."""
import json
from pathlib import Path

from .qwen_projector import QwenProjectorComparison


class QwenAnchorAdapterComparison(QwenProjectorComparison):
    def __init__(self, model, config, *, runtime=None, adapter_module=None):
        config = dict(config)
        allowed = {"adapter_path", "identity_path", "retention_ratio", "tokens_per_crop"}
        if set(config) - allowed:
            raise ValueError(f"Unknown anchor adapter keys: {sorted(set(config) - allowed)}")
        budget_config = {k: v for k, v in config.items() if k in {"retention_ratio", "tokens_per_crop"}}
        super().__init__(model, "dppruner", budget_config, "qwen25_vl")
        self.method = "feature_anchor_adapter"
        self.adapter_path = config.get("adapter_path")
        self.base_identity = None
        if adapter_module is None:
            from mllm_comp.analysis.common.anchor7b_runtime import load_adapter
            if not self.adapter_path or not config.get("identity_path"):
                raise ValueError("A trained adapter path and base identity file are required")
            identity = json.loads(Path(config["identity_path"]).read_text())
            if runtime is None or Path(runtime.model_path_or_id).resolve() != Path(identity["checkpoint"]).resolve():
                raise ValueError("MLP adapter identity does not match the runtime checkpoint")
            self.base_identity = identity["identity_sha256"]
            adapter_module = load_adapter(self.adapter_path, identity, self.decoder.embed_tokens.weight.device)
        if adapter_module.spec.hidden_size != self.decoder.config.hidden_size:
            raise ValueError("MLP and decoder hidden dimensions differ")
        self.adapter = adapter_module.eval().requires_grad_(False)

    def prepare_qwen_embeddings(self, call, hidden):
        import torch
        from mllm_comp.compressors.image.fixed_visual_merge import FixedVisualMergeSpec, fixed_visual_merge
        source = self.source.get()
        visual_ids = self.incoming.get().vision[0].nonzero(as_tuple=True)[0]
        features = hidden[0].index_select(0, visual_ids)
        merge = self.model.model.visual.spatial_merge_size
        height, width = (int(x) // merge for x in source["grid"][0, 1:])
        result = fixed_visual_merge(features, grid_height=height, grid_width=width,
            spec=FixedVisualMergeSpec(partition="feature_anchor", target_count=self.token_budget(len(features))))
        base = {"tokens": result.tokens, "indices": result.indices.tolist(),
            "members": result.metadata["cluster_members"]}
        with torch.autocast(device_type=hidden.device.type, dtype=torch.bfloat16, enabled=hidden.device.type == "cuda"):
            output = self.adapter(base)
        original_positions = call["position_ids"].clone()
        compact = self.compact_selection(call, hidden, result.indices,
            {"trained": True, "adapter_path": self.adapter_path,
             "cluster_members": base["members"], "partition_hash": result.metadata["partition_hash"]})
        destinations = self.incoming.get().vision[0].nonzero(as_tuple=True)[0]
        compact = compact.clone()
        compact[0, destinations] = output["tokens"].to(compact.dtype)
        positions = call["position_ids"].clone()
        for row, members in enumerate(base["members"]):
            indices = visual_ids[torch.tensor(members, device=visual_ids.device)]
            positions[-3:, 0, destinations[row]] = original_positions[-3:, 0].index_select(1, indices).float().mean(1).round().to(positions.dtype)
        if not bool(torch.isfinite(compact).all()):
            raise ValueError("Nonfinite MLP-compressed embeddings")
        call["inputs_embeds"] = compact
        call["position_ids"] = positions
        return compact

    def runtime_metadata(self):
        return {**super().runtime_metadata(), "position_policy": "rounded_cluster_centroid_mrope_text_unchanged",
            "trained": True, "base_identity_sha256": self.base_identity,
            "adapter_path": self.adapter_path,
            "budget_generalization": ("Adapter was trained at K128; ratio budgets are evaluation-time generalization"
                if self.retention_ratio is not None else
                f"Adapter was trained at K128; fixed evaluation budget K{self.config['tokens_per_crop']}")}


def create_executor(*, model, stage, runtime=None, **kwargs):
    return QwenAnchorAdapterComparison(model, stage.config, runtime=runtime)


def register_contracts(registry):
    from mllm_comp.compressors.contracts import (
        AlgorithmDescriptor, ModalityVariantDescriptor, BackbonePlacementBinding, BindingTarget, TokenUnit,
    )
    from mllm_comp.compressors.image.fixed_visual_merge import fixed_visual_merge
    from mllm_comp.core.vocabulary import CompressionMethod, Placement, TokenOperation, Modality, ExecutionPhase
    from mllm_comp.core.support import SupportUnit, SupportUnitId
    image = frozenset({Modality.IMAGE})
    name = "feature_anchor_adapter"
    binding = f"{name}.image.qwen25_vl"
    registry.register_algorithm(AlgorithmDescriptor(name=name, version="trained-anchor-mlp-v1",
        method=CompressionMethod.TOKEN_COMPRESSION, kernel=fixed_visual_merge,
        supported_placements=frozenset({Placement.PROJECTOR_LEVEL}), operations=frozenset({TokenOperation.MERGE}),
        metadata={"trained": True, "source": "existing_supervised_anchor_mlp_checkpoint"}))
    registry.register_variant(ModalityVariantDescriptor(name=f"{name}.image", algorithm=name,
        target_modalities=image, token_unit=TokenUnit.FEATURE_TOKEN))
    registry.register_binding(BackbonePlacementBinding(name=binding, backbone="qwen25_vl", variant=f"{name}.image",
        targets=(BindingTarget(name="anchor_mlp", placement=Placement.PROJECTOR_LEVEL,
            target_site="post_vision_merger_pre_llm_scatter", target_modalities=image,
            required_handles=frozenset({"vision_projector", "llm"})),),
        supported_phases=frozenset({ExecutionPhase.GENERATE}), executor_factory=create_executor,
        legalizer="qwen25_vl.native_centroid_mrope_compact_cache", exclusive_runtime_slot="image_comparison",
        conflicts=frozenset(b.name for b in registry.bindings() if b.backbone == "qwen25_vl"),
        metadata={"single_image_only": True, "trained": True}))
    registry.extend_support([SupportUnit(identity=SupportUnitId(algorithm=name, variant=f"{name}.image",
        backbone="qwen25_vl", target_modality=Modality.IMAGE, phase=ExecutionPhase.GENERATE), binding=binding)])
