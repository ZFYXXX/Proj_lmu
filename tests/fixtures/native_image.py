"""Small native image models, model inputs, and public activation factories."""

from types import SimpleNamespace
from typing import Any

from mllm_comp.compressors.multimodal_registry import builtin_multimodal_compression_contracts
from mllm_comp.core.vocabulary import Modality


def stage(algorithm, variant, config):
    return SimpleNamespace(
        name=f"{algorithm}-real-forward", algorithm=algorithm, variant=variant,
        target_modalities={Modality.IMAGE}, operation={}, criterion={}, schedule={}, config=config,
    )


def conversation(content, *, system_prompt):
    return [
        {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
        {"role": "user", "content": content},
    ]


def image_conversation(image_path, *, system_prompt):
    return conversation([
        {"type": "image", "image": str(image_path)},
        {"type": "text", "text": "What is the dominant color?"},
    ], system_prompt=system_prompt)


def move_to_gpu(value, torch):
    to = getattr(value, "to", None)
    if not callable(to):
        return value
    value = value.to(torch.device("cuda:0"))
    return value.to(dtype=torch.bfloat16) if value.is_floating_point() else value


def activation(model, stage, binding_config=None):
    registry = builtin_multimodal_compression_contracts()
    binding = registry.binding(stage.binding)
    # Public factories accept the original registry binding too.
    return binding.executor_factory(
        model=model, runtime=SimpleNamespace(metadata={}, model_path_or_id="/fixtures/native-tiny-image"),
        facade=None, stage=stage, binding=binding, binding_config=binding_config, execution_phase=stage.execution_phase,
    )


def qwen_model(backbone):
    from transformers import Qwen3VLConfig, Qwen3VLForConditionalGeneration, Qwen2_5_VLConfig, Qwen2_5_VLForConditionalGeneration
    text = dict(vocab_size=64, hidden_size=24, intermediate_size=32, num_hidden_layers=4,
        num_attention_heads=3, num_key_value_heads=1, head_dim=8, max_position_embeddings=64,
        rope_scaling={"rope_type": "default", "mrope_section": [1, 1, 2]}, pad_token_id=0)
    vision = dict(depth=3, hidden_size=16, intermediate_size=24, num_heads=2,
        patch_size=2, temporal_patch_size=1, spatial_merge_size=2, out_hidden_size=24,
        num_position_embeddings=16, deepstack_visual_indexes=[0, 1, 2], window_size=8,
        fullatt_block_indexes=[1])
    config_cls, model_cls = (Qwen3VLConfig, Qwen3VLForConditionalGeneration) if backbone == "qwen3_vl" else (Qwen2_5_VLConfig, Qwen2_5_VLForConditionalGeneration)
    config = config_cls(text_config=text, vision_config=vision, image_token_id=10,
        video_token_id=11, vision_start_token_id=7, vision_end_token_id=8)
    config._attn_implementation = "eager"
    return model_cls(config)


def llava_model(num_decoder_layers=3):
    import torch
    from llava.model.language_model.llava_qwen import LlavaQwenConfig, LlavaQwenForCausalLM
    from llava.model.multimodal_encoder.siglip_encoder import SigLipVisionConfig, SigLipVisionModel, SigLipVisionTower
    config = LlavaQwenConfig(vocab_size=64, hidden_size=16, intermediate_size=24,
        num_hidden_layers=num_decoder_layers, num_attention_heads=2, num_key_value_heads=1,
        max_position_embeddings=2048, mm_patch_merge_type="flat", image_aspect_ratio="square",
        tokenizer_padding_side="right", tokenizer_model_max_length=2048, pad_token_id=0)
    config._attn_implementation = "eager"
    model = LlavaQwenForCausalLM(config)
    # The native tower asserts 729 patches. Keep that input contract and reduce
    # channels/depth only; construct weights locally without from_pretrained.
    tower = SigLipVisionTower("native-tiny-siglip", SimpleNamespace(), delay_load=True)
    tower.config = SigLipVisionConfig(hidden_size=16, intermediate_size=24,
        num_hidden_layers=3, num_attention_heads=2, image_size=54, patch_size=2)
    tower.vision_tower = SigLipVisionModel(tower.config)
    tower.vision_tower.vision_model.head = torch.nn.Identity()
    tower.is_loaded = True
    model.model.vision_tower = tower
    model.model.mm_projector = torch.nn.Linear(16, 16)
    return model


def model(backbone, *, num_decoder_layers=3):
    return llava_model(num_decoder_layers=num_decoder_layers) if backbone == "llava_onevision" else qwen_model(backbone)


def decoder(model, backbone):
    return model.model if backbone == "llava_onevision" else model.model.language_model


def inputs(torch, backbone):
    if backbone == "llava_onevision":
        ids = torch.tensor([[1, -200, 2, 3]])
        return dict(input_ids=ids, attention_mask=torch.ones_like(ids), images=torch.randn(1, 3, 54, 54, requires_grad=True),
            image_sizes=[(54, 54)], modalities=["image"])
    ids = torch.tensor([[1, 7, 10, 10, 10, 10, 8, 2]])
    return dict(input_ids=ids, attention_mask=torch.ones_like(ids),
        pixel_values=torch.randn(16, 12, requires_grad=True), image_grid_thw=torch.tensor([[1, 4, 4]]))


def training_backward(
    model: Any,
    model_inputs: dict[str, Any],
    torch: Any,
) -> dict[str, Any]:
    """Run one label-aligned training step with a small gradient surface."""

    input_ids = model_inputs["input_ids"]
    labels = torch.full_like(input_ids, -100)
    attention_mask = model_inputs.get("attention_mask")
    if attention_mask is None:
        target_index = int(input_ids.shape[1]) - 1
    else:
        target_index = int(attention_mask[0].nonzero(as_tuple=False)[-1].item())
    target_id = int(input_ids[0, target_index].item())
    assert target_id >= 0
    labels[0, target_index] = target_id

    for parameter in model.parameters():
        parameter.requires_grad_(False)
    training_name, training_parameter = next(
        (name, parameter)
        for name, parameter in model.named_parameters()
        if "patch_embed" in name and parameter.is_floating_point()
    )
    training_parameter.requires_grad_(True)
    model.train()
    model.zero_grad(set_to_none=True)
    output = model(**{**model_inputs, "labels": labels})
    loss = getattr(output, "loss", None)
    logits = getattr(output, "logits", None)
    assert loss is not None and bool(torch.isfinite(loss).item())
    assert logits is not None and int(logits.shape[1]) >= 2
    loss.backward()
    gradient = training_parameter.grad
    assert gradient is not None
    gradient_norm_tensor = gradient.norm().float()
    assert bool(torch.isfinite(gradient_norm_tensor).item())
    gradient_norm = float(gradient_norm_tensor.item())
    assert gradient_norm > 0.0
    model.zero_grad(set_to_none=True)
    model.eval()
    return {
        "loss": float(loss.detach().float().item()),
        "output_logits_shape": [int(item) for item in logits.shape],
        "training_parameter": training_name,
        "visual_gradient_norm": gradient_norm,
        "model_training_mode": True,
    }
