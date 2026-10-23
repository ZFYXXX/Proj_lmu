"""Shared native-Qwen state construction for trainable hybrid pilots."""

from dataclasses import asdict
import hashlib
from pathlib import Path

from mllm_comp.analysis.common.visual_token_phenomena import answer_predictor_positions

CHECKPOINT = "/home/ZhouFengyu/fastv-repro/models/Qwen2.5-VL-3B-Instruct"
REVISION = "1b5a7674d40a0ac9acef541a46fed12d710bd36b"


def code_hash():
    root = Path(__file__).parents[2]
    paths = [root / "compressors/image/learned_hybrid.py", Path(__file__),
             root / "analysis/common/spectral_residual.py", root / "analysis/probes/visual_evidence_interaction.py",
             root / "analysis/probes/visual_token_phenomena.py", root / "backbones/qwen25_vl/implementation.py"]
    h = hashlib.sha256()
    for path in paths:
        h.update(str(path.relative_to(root)).encode())
        h.update(path.read_bytes())
    return h.hexdigest()


def load_runtime(device):
    import torch
    from mllm_comp.backbones.base import BackboneRuntime
    from mllm_comp.backbones.qwen25_vl import create_qwen25_vl_backbone
    facade = create_qwen25_vl_backbone()
    processor = facade.processor_adapter.load_processor(CHECKPOINT, revision=REVISION, local_files_only=True,
        use_fast=False, min_pixels=200704, max_pixels=1605632)
    model = facade.model_adapter.load_model(CHECKPOINT, train=False, revision=REVISION, local_files_only=True,
        torch_dtype=torch.bfloat16, attn_implementation="eager").to(device)
    model.requires_grad_(False)
    handles = facade.model_adapter.structural_handles(model)
    llm = handles["llm"]
    if float(getattr(llm.config, "attention_dropout", 0)) != 0:
        raise ValueError("Dropout-free frozen decoder required for paired diagnostics.")
    runtime = BackboneRuntime(model=model, processor=processor, model_path_or_id=CHECKPOINT,
        device=torch.device(device), dtype=torch.bfloat16, revision=REVISION, metadata={})
    return {"facade": facade, "runtime": runtime, "model": model, "processor": processor,
            "llm": llm, "vision": handles["vision_encoder"], "device": torch.device(device)}


def move_inputs(values, device):
    import torch
    result = {}
    for key, value in values.items():
        if isinstance(value, torch.Tensor):
            value = value.to(device)
            if value.is_floating_point():
                value = value.to(torch.bfloat16)
        result[key] = value
    return result


def prepare_state(bundle, prompt_inputs, teacher_inputs=None):
    import torch
    model, vision = bundle["model"], bundle["vision"]
    prompt = move_inputs(prompt_inputs, bundle["device"])
    teacher = move_inputs(teacher_inputs, bundle["device"]) if teacher_inputs is not None else prompt
    prefix = prompt["input_ids"].shape[1]
    grid = prompt["image_grid_thw"]
    if len(grid) != 1 or int(grid[0, 0]) != 1 or prompt["input_ids"].shape[0] != 1:
        raise ValueError("Single still-image, batch one only.")
    if not torch.equal(prompt["input_ids"], teacher["input_ids"][:, :prefix]):
        raise ValueError("Teacher prefix changed the gold-free prompt.")
    if not bool((teacher["attention_mask"] == 1).all()):
        raise ValueError("Unpadded batch one required.")
    height, width = int(grid[0, 1]) // vision.spatial_merge_size, int(grid[0, 2]) // vision.spatial_merge_size
    visual = torch.nonzero(teacher["input_ids"][0] == model.config.image_token_id).flatten().tolist()
    if len(visual) != height * width or len(visual) < 128 or not all(i < prefix for i in visual):
        raise ValueError("Invalid visual grid/placeholder span.")
    with torch.no_grad():
        features, = model.model.get_image_features(prompt["pixel_values"], grid)
        hidden = model.get_input_embeddings()(teacher["input_ids"]).clone()
        hidden[0, visual] = features
        positions, _ = model.model.get_rope_index(teacher["input_ids"], grid, attention_mask=teacher["attention_mask"])
        if positions.shape[0] == 4:
            positions = positions[1:]
    if not bool(torch.isfinite(features).all()):
        raise ValueError("Nonfinite native vision features.")
    return {"features": features.detach(), "hidden": hidden.detach(), "positions": positions,
            "visual": visual, "height": height, "width": width, "prefix": prefix,
            "prompt_inputs": prompt, "teacher_inputs": teacher}


def compact_state(state, output):
    import torch
    from mllm_comp.analysis.probes.visual_token_phenomena import _compact_layer_state
    indices = output["indices"]
    if len(indices) != 128 or indices != sorted(set(indices)):
        raise ValueError("Hybrid must physically emit exactly 128 unique tokens in source order.")
    compact = _compact_layer_state(hidden=state["hidden"], position_ids=state["positions"][:, 0], visual_positions=state["visual"],
        members=tuple((i,) for i in indices), operator="retain", token_scores=None, budget=128)
    mapping = compact["source_to_compact"]
    gather = torch.tensor(sorted(mapping, key=mapping.get), device=state["hidden"].device)
    destinations = torch.tensor([mapping[state["visual"][i]] for i in indices], device=gather.device)
    hidden = compact["hidden"].clone()
    hidden[0, destinations] = output["tokens"]
    inputs = state["teacher_inputs"]
    labels = inputs.get("labels")
    labels = labels.index_select(1, gather) if labels is not None else None
    tokens = inputs["input_ids"].index_select(1, gather)
    mask = inputs["attention_mask"].index_select(1, gather)
    if hidden.shape[1] != tokens.shape[1] or mask.shape != tokens.shape or not bool((mask == 1).all()):
        raise ValueError("Mask/placeholder/embedding alignment failed.")
    if int((tokens == inputs["input_ids"][0, state["visual"][0]]).sum()) != 128:
        raise ValueError("Compressed placeholder count mismatch.")
    original_text = [i for i in mapping if i not in set(state["visual"])]
    if not torch.equal(compact["position_ids"][:, 0, [mapping[i] for i in original_text]], state["positions"][:, 0, original_text]):
        raise ValueError("Text M-RoPE positions changed.")
    if labels is not None:
        before, targets = answer_predictor_positions(inputs["labels"])
        after, compact_targets = answer_predictor_positions(labels)
        if targets != compact_targets or after != [mapping[p] for p in before]:
            raise ValueError("Answer labels/predictors changed under compaction.")
    return {"hidden": hidden, "positions": compact["position_ids"], "labels": labels,
            "prefix": state["prefix"] - len(state["visual"]) + 128}


def answer_logits(bundle, hidden, positions, labels):
    predictors, targets = answer_predictor_positions(labels)
    output = bundle["llm"](inputs_embeds=hidden, position_ids=positions,
        attention_mask=__import__("torch").ones(hidden.shape[:2], dtype=__import__("torch").long, device=hidden.device),
        use_cache=False, output_attentions=False, return_dict=True)
    logits = bundle["model"].lm_head(output.last_hidden_state[:, predictors])[0].float()
    return logits, __import__("torch").tensor(targets, device=logits.device)


def save_module(module, folder):
    import json
    from safetensors.torch import save_file
    from mllm_comp.core.serialization import atomic_write_json
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    save_file({key: value.detach().cpu().contiguous() for key, value in module.state_dict().items()}, str(folder / "projector.safetensors"))
    atomic_write_json(folder / "projector_config.json", {"spec": asdict(module.spec), "implementation_hash": code_hash()}, ensure_ascii=False)


def load_module(folder, device):
    import json
    from safetensors.torch import load_file
    from mllm_comp.compressors.image.learned_hybrid import LearnedHybridSpec, create_learned_hybrid
    folder = Path(folder)
    config = json.loads((folder / "projector_config.json").read_text())
    if config["implementation_hash"] != code_hash():
        raise ValueError("Checkpoint implementation hash does not match current code.")
    module = create_learned_hybrid(LearnedHybridSpec(**config["spec"]))
    module.load_state_dict(load_file(str(folder / "projector.safetensors")), strict=True)
    return module.to(device=device)
