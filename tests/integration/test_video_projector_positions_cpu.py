"""Public video configuration drives the value boundary across device hooks."""

from contextlib import nullcontext

import pytest

from mllm_comp.pipeline.stage import CompressionStageConfig
from tests.fixtures import native_image as image_fixture


@pytest.mark.parametrize("backbone", ["qwen3_vl", "qwen25_vl", "llava_onevision"])
def test_vqtoken_configured_input_with_vision_and_projector_on_different_gpus(
    backbone, algorithm="vqtoken", model_factory=None, inputs_factory=None, activation_factory=image_fixture.activation,
    request_context_factory=None,
):
    torch = pytest.importorskip("torch")
    if torch.cuda.device_count() < 2:
        pytest.skip("Requires the authorized two-GPU validation job.")
    from accelerate import dispatch_model

    torch.manual_seed(443)
    if model_factory is not None:
        model = model_factory().train()
    elif backbone == "llava_onevision":
        from tests.integration.test_video_vqtoken_llava_onevision_positions_cpu import _video_model, _inputs
        model = _video_model().train()
    elif backbone == "qwen3_vl":
        from tests.integration.test_video_vqtoken_qwen3_vl_positions_cpu import _inputs
        model = image_fixture.qwen_model(backbone).train()
    else:
        from tests.integration.test_video_vqtoken_qwen25_vl_positions_cpu import _inputs
        model = image_fixture.qwen_model(backbone).train()
    device_map = {name: "cuda:1" for name, _ in model.named_children() if name != "model"}
    if backbone == "llava_onevision":
        device_map.update({f"model.{name}": "cuda:0" if name == "vision_tower" else "cuda:1"
                          for name, _ in model.model.named_children()})
        device_map["model.image_newline"] = "cuda:1"
        projectors = [model.model.mm_projector]
    else:
        device_map.update({f"model.{name}": "cuda:1" for name, _ in model.model.named_children() if name != "visual"})
        device_map.update({f"model.visual.{name}": "cuda:1" if name in {"merger", "deepstack_merger_list"} else "cuda:0"
                          for name, _ in model.model.visual.named_children()})
        projectors = [model.model.visual.merger, *getattr(model.model.visual, "deepstack_merger_list", ())]
    dispatch_model(model, device_map=device_map)
    target = {"vqtoken": "fixed_k_cluster", "visionzip": "dominant_contextual_reduce", "fastvid": "dyseg_stprune_dtm",
              "dytok": "dynamic_visionzip_reduce", "dycoke": "ttm_visual_merge"}[algorithm]
    stage = CompressionStageConfig.from_dict({
        "name": "video-vq-position", "placement": "projector_level", "execution_phase": "forward",
        "plugin": algorithm, "algorithm": algorithm, "variant": f"{algorithm}.video",
        "binding": f"{algorithm}.video.{backbone}", "target_modalities": ["video"],
        "operation": {"targets": {target: {"target_site": "projector_input"}}},
    })
    activation = activation_factory(model, stage)
    activation.install()
    args = (inputs_factory or _inputs)(torch)
    for name, value in args.items():
        if torch.is_tensor(value):
            args[name] = value.detach().to("cuda:1" if backbone == "llava_onevision" else "cuda:0")
    if backbone == "llava_onevision":
        pixels = args["images"][0].detach().to("cuda:0").requires_grad_(True)
        args["images"] = [pixels]
    else:
        pixels = args["pixel_values_videos"].requires_grad_(True)
    labels = torch.full_like(args["input_ids"], -100)
    labels[0, -1] = args["input_ids"][0, -1]
    args.update(labels=labels, use_cache=False)
    try:
        with (request_context_factory(activation) if request_context_factory else nullcontext()):
            output = model(**args)
        output.loss.backward()
        assert torch.isfinite(output.loss)
        assert pixels.grad is not None and torch.isfinite(pixels.grad).all() and pixels.grad.abs().sum() > 0
        for projector in projectors:
            assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in projector.parameters())
            assert any(p.grad.abs().sum() > 0 for p in projector.parameters())
        metadata = activation.runtime_metadata()
        assert metadata["projector_position"] == "before"
        assert metadata["projector_invocations"] == 1
        assert metadata["projector_layout_deltas"][0]["projector_position"] == "before"
        assert metadata["canonical_evidence_applicable"] is False
    finally:
        activation.uninstall()
