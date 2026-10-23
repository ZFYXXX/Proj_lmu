import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("llava")
from tests.fixtures import native_image
from mllm_comp.pipeline.stage import CompressionStageConfig


@pytest.mark.parametrize("method,config", [
    ("vlm_pruner", {"prune_after_layer": 2, "retain_ratio": .25}),
    ("core_matching", {"prune_after_layer": 2, "maximum_visual_retain_ratio": .25}),
    ("twigvlm", {"twig_start_layer": 1, "twig_layer_count": 1, "average_retention_ratio": .5}),
])
def test_vlm_pruner_native_training_and_checkpoint_replay(method, config):
    torch.manual_seed(14)
    model = native_image.llava_model(num_decoder_layers=4).train()
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": True})
    stage = CompressionStageConfig.from_dict({
        "name": method, "algorithm": method, "plugin": method,
        "variant": f"{method}.image", "binding": f"{method}.image.llava_onevision",
        "placement": "llm_level", "execution_phase": "forward", "target_modalities": ["image"],
        "config": config,
    })
    activation = native_image.activation(model, stage)
    inputs = native_image.inputs(torch, "llava_onevision")
    labels = inputs["input_ids"].clone()
    labels[:, :2] = -100
    activation.install()
    try:
        result = model(**inputs, labels=labels, use_cache=False)
        assert result.logits.shape[1] <= 3 + 183
        assert torch.isfinite(result.loss)
        result.loss.backward()
        assert model.model.mm_projector.weight.grad.abs().sum() > 0
        activation.uninstall()
        from dataclasses import replace
        from mllm_comp.core.vocabulary import ExecutionPhase
        activation = native_image.activation(model, replace(stage, execution_phase=ExecutionPhase.GENERATE))
        activation.install()
        model.eval()
        with torch.no_grad():
            inputs["inputs"] = inputs.pop("input_ids")
            result = model.generate(**inputs, max_new_tokens=3, min_new_tokens=3, do_sample=False)
            assert result.shape[-1] >= 3
    finally:
        activation.uninstall()


def test_weighted_layout_replay_preserves_fusion_and_gradients():
    from types import SimpleNamespace
    from mllm_comp.compressors.comparison.decoder import VLMPrunerMethod
    from mllm_comp.compressors.image.decoder_selection import decoder_layout_replay
    method = VLMPrunerMethod(stage=None, config={"retain_ratio": .5}, rules={}, backbone="llava_onevision")
    call = method.new_call()
    call.set_visual_layout([{"start": 0, "end": 8, "grid_width": 4}])
    hidden = torch.randn(11, 16, requires_grad=True)
    call.keys = torch.randn(1, 11, 8)
    result = call.select(method.positions[0], hidden, image_indices=torch.arange(1, 9))
    replay = decoder_layout_replay(result.layout_delta, rules={})(hidden[None])[0]
    torch.testing.assert_close(replay, result.hidden_states)
    expected = torch.autograd.grad(result.hidden_states.square().sum(), hidden, retain_graph=True)[0]
    actual = torch.autograd.grad(replay.square().sum(), hidden)[0]
    torch.testing.assert_close(actual, expected)
