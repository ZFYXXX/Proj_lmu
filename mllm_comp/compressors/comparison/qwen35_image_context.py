"""Image-global context selector, isolated from the original Qwen3.5 head format."""
from dataclasses import asdict
import hashlib
import json
from pathlib import Path

from mllm_comp.analysis.common.anchor7b_runtime import file_sha
from mllm_comp.analysis.common.qwen35_transfer import implementation_hash as legacy_hash
from mllm_comp.compressors.image.context_visual_topk import ContextVisualTopK, ContextVisualTopKSpec
from .qwen35_topk import Qwen35TransferredProjector

METHOD = "image_context_selector"
FORMAT = "qwen35_image_context_v1"


def implementation_hash():
    return hashlib.sha256(Path(__file__).read_bytes() + legacy_hash().encode()).hexdigest()


def create_image_head(dimension, *, visual):
    if visual is None or visual.spec.hidden_size != dimension:
        raise ValueError("Image context requires the completed same-base visual prior")
    return ContextVisualTopK(ContextVisualTopKSpec(dimension, context_mode="image"), visual=visual)


def save_image_head(module, folder, *, kind="image", identity, provenance):
    from safetensors.torch import save_file
    from mllm_comp.core.serialization import atomic_write_json
    folder = Path(folder)
    if kind != "image" or module.spec.context_mode != "image" or folder.exists():
        raise ValueError("Invalid image head or checkpoint overwrite")
    folder.mkdir(parents=True)
    save_file({k: v.detach().cpu().contiguous() for k, v in module.state_dict().items()}, str(folder / "head.safetensors"))
    atomic_write_json(folder / "head_config.json", {"format": FORMAT, "kind": kind,
        "spec": asdict(module.spec), "base_identity_sha256": identity["identity_sha256"],
        "implementation_hash": implementation_hash(), "weights_sha256": file_sha(folder / "head.safetensors"),
        "provenance": provenance})


def load_image_head(folder, *, identity, device):
    from safetensors.torch import load_file
    folder = Path(folder)
    cfg = json.loads((folder / "head_config.json").read_text())
    if (cfg.get("format") != FORMAT or cfg.get("kind") != "image"
            or cfg["spec"].get("context_mode") != "image"
            or cfg["spec"]["hidden_size"] != identity["hidden_size"]
            or cfg["base_identity_sha256"] != identity["identity_sha256"]
            or cfg["implementation_hash"] != implementation_hash()
            or cfg["weights_sha256"] != file_sha(folder / "head.safetensors")):
        raise ValueError("Wrong image context format/base/source/spec or corrupted weights")
    module = ContextVisualTopK(ContextVisualTopKSpec(**cfg["spec"]))
    module.load_state_dict(load_file(str(folder / "head.safetensors")), strict=True)
    return module.to(device)


class Qwen35ImageContextProjector(Qwen35TransferredProjector):
    """Reuse the tested native compaction without reading question embeddings."""
    def __init__(self, model, config, *, runtime=None, module=None, tokenizer=None):
        if config.get("method") != METHOD:
            raise ValueError("Wrong image context method")
        if module is None:
            identity = json.loads(Path(config["identity_path"]).read_text())
            if Path(runtime.model_path_or_id).resolve() != Path(identity["checkpoint"]).resolve():
                raise ValueError("Wrong Qwen3.5 base")
            cfg = json.loads((Path(config["checkpoint_path"]) / "head_config.json").read_text())
            if cfg["provenance"].get("step") != 320:
                raise ValueError("Evaluation requires the fixed step320 checkpoint")
            module = load_image_head(config["checkpoint_path"], identity=identity, device=model.device)
        if module.spec.context_mode != "image":
            raise ValueError("A question selector cannot enter the image-only binding")
        super().__init__(model, {**config, "method": "visual_selector"}, runtime=runtime, module=module, tokenizer=tokenizer)
        self.method, self.config = METHOD, dict(config)

    def runtime_metadata(self):
        result = super().runtime_metadata()
        record = self.record.get()
        if record is not None:
            record.update(context_mode="image", reads_question=False,
                          conditioning_source="mean_of_native_post_merger_visual_features")
        result.update(context_mode="image", reads_question=False)
        return result


def create_executor(*, model, stage, runtime=None, **kwargs):
    return Qwen35ImageContextProjector(model, stage.config, runtime=runtime)


def register_contracts(registry):
    from mllm_comp.compressors.contracts import AlgorithmDescriptor, ModalityVariantDescriptor, BackbonePlacementBinding, BindingTarget
    from mllm_comp.core.vocabulary import CompressionMethod, Placement, TokenOperation, Modality, ExecutionPhase
    from mllm_comp.core.support import SupportUnit, SupportUnitId
    name = "qwen35_image_context_projector"
    registry.register_algorithm(AlgorithmDescriptor(name=name, version="v1.0.0", method=CompressionMethod.TOKEN_COMPRESSION,
        kernel=create_image_head, supported_placements=frozenset({Placement.PROJECTOR_LEVEL}),
        operations=frozenset({TokenOperation.DROP}), metadata={"context": "image-global", "training": "continuous_topk"}))
    registry.register_variant(ModalityVariantDescriptor(name=f"{name}.image", algorithm=name,
        target_modalities=frozenset({Modality.IMAGE}), conditioning_modalities=frozenset()))
    binding = f"{name}.image.qwen3_5"
    registry.register_binding(BackbonePlacementBinding(name=binding, backbone="qwen3_5", variant=f"{name}.image",
        targets=(BindingTarget(name="image_context", placement=Placement.PROJECTOR_LEVEL,
            target_site="post_merger_pre_llm_scatter", target_modalities=frozenset({Modality.IMAGE}),
            required_handles=frozenset({"vision_projector", "llm"})),),
        supported_phases=frozenset({ExecutionPhase.GENERATE}), executor_factory=create_executor,
        legalizer="qwen35.native_sparse_mrope_hybrid_cache", exclusive_runtime_slot="image_comparison",
        conflicts=frozenset(b.name for b in registry.bindings() if b.backbone == "qwen3_5")))
    registry.extend_support([SupportUnit(identity=SupportUnitId(algorithm=name, variant=f"{name}.image",
        backbone="qwen3_5", target_modality=Modality.IMAGE, phase=ExecutionPhase.GENERATE), binding=binding)])
