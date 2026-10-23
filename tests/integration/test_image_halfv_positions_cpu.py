"""HalfV's two fixed sources and the shared multi-stage application lifecycle."""

import importlib

import pytest

from mllm_comp.pipeline.stage import CompressionStageConfig
from mllm_comp.compressors.image import decoder_selection as execution
from tests.fixtures import native_image as image_fixture
from tests.integration import test_image_btp_positions_cpu as shared


def _stage(backbone, boundaries=(1, 2), *, sources=(1, 2), mid=False, phase="forward"):
    return CompressionStageConfig.from_dict({
        "name": "halfv-positions", "placement": "llm_level", "execution_phase": phase,
        "plugin": "halfv", "algorithm": "halfv", "variant": "halfv.image",
        "binding": f"halfv.image.{backbone}", "target_modalities": ["image"],
        "operation": {"targets": {"decoder_ivr_ssr_boundaries": {
            "target_site": "llm_decoder_layer_post_attention_image_tokens" if mid else "llm_decoder_layer_output_image_tokens",
            "layers": list(boundaries), **({} if mid else {"layer_edge": "before"})}}},
        "config": {"ivr_layer_index": sources[0], "ssr_layer_index": sources[1],
            "ivr_retention_ratio": 0.5, "ssr_retention_ratio": 0.25, "anchor_ratio": 0.25},
    })


@pytest.mark.parametrize("backbone", ["qwen3_vl", "qwen25_vl", "llava_onevision"])
@pytest.mark.parametrize("boundaries,mid", [((1, 2), False), ((2, 4), False), ((1, 2), True), ((2, 3), True)])
def test_halfv_native_stages_checkpoint_gradients_and_generate(backbone, boundaries, mid):
    count = 729 if backbone == "llava_onevision" else 4
    shared._run(backbone, boundaries, (1, 2), mid, stage_factory=_stage,
        visual_counts=(int(count * 0.5), int(count * 0.25)))


@pytest.mark.parametrize("backbone", ["qwen3_vl", "qwen25_vl", "llava_onevision"])
@pytest.mark.parametrize("mid", [False, True])
def test_halfv_two_device_sources(backbone, mid):
    count = 729 if backbone == "llava_onevision" else 4
    shared._run(backbone, (2, 3), (1, 2), mid, two_devices=True, stage_factory=_stage,
        visual_counts=(int(count * 0.5), int(count * 0.25)))


@pytest.mark.parametrize("backbone", ["qwen3_vl", "qwen25_vl", "llava_onevision"])
@pytest.mark.parametrize("checkpoint", [False, True])
def test_halfv_empty_ivr_skips_ssr_inside_checkpoint(backbone, checkpoint):
    torch = pytest.importorskip("torch")
    model = image_fixture.model(backbone, num_decoder_layers=4).train()
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": checkpoint})
    stage = _stage(backbone, (1, 3), mid=True)
    stage.config.update(ivr_retention_ratio=0.001, ssr_retention_ratio=0.0001)
    activation = image_fixture.activation(model, stage)
    activation.install()
    if backbone == "llava_onevision":
        activation._bridge.prepare_training_forward()
    try:
        args = image_fixture.inputs(torch, backbone)
        labels = torch.full_like(args["input_ids"], -100)
        labels[0, -1] = args["input_ids"][0, -1]
        output = model(**args, labels=labels, use_cache=False)
        output.loss.backward()
        assert output.logits.shape[1] == (3 if backbone == "llava_onevision" else 4)
        assert len(activation.runtime_metadata()["layout_delta_records"]) == 1
        assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    finally:
        activation.uninstall()


@pytest.mark.parametrize("backbone", ["qwen3_vl", "qwen25_vl", "llava_onevision"])
@pytest.mark.parametrize("boundaries,mid", [((1, 2), False), ((2, 4), False), ((2, 3), True)])
def test_halfv_fixed_native_sources_and_pending_score_mapping(backbone, boundaries, mid, monkeypatch):
    torch = pytest.importorskip("torch")
    torch.manual_seed(332)
    model = image_fixture.model(backbone, num_decoder_layers=4).eval()
    decoder = image_fixture.decoder(model, backbone)
    provider = importlib.import_module(f"mllm_comp.backbones.{backbone}.provider_bridge")
    project, kernel = provider._decoder_query_keys, execution.halfv_image_tokens
    pending, native_inputs, normalized, stages = {}, {}, [], []
    sampling = False

    def observe(index):
        def hook(module, args):
            if not sampling:
                native_inputs[index] = args[0].detach().clone()
        return hook

    def capture(layer, hidden, **kwargs):
        nonlocal sampling
        sampling = True
        try:
            query, image_keys, keys = project(layer, hidden, **kwargs)
        finally:
            sampling = False
        index = list(decoder.layers).index(layer)
        assert index in (0, 1) and kwargs["use_rope"] == bool(index)
        # Independently recompute the original image-only, unscaled per-head softmax.
        pending[index] = torch.einsum("qhd,khd->hqk", query.float(), image_keys.float()).softmax(-1).mean((0, 1))
        return query, image_keys, keys

    def apply(hidden, **kwargs):
        stage = kwargs["spec"].stage_index(kwargs["boundary_after_layer"])
        torch.testing.assert_close(kwargs["image_attention_scores"], pending.pop(stage))
        if stage == 0:
            torch.testing.assert_close(kwargs["selection_hidden_states"], normalized[0][0])
        result = kernel(hidden, **kwargs)
        for key, scores in pending.items():
            pending[key] = scores.index_select(0, result.selected_visual_indices.to(scores.device))
        torch.testing.assert_close(result.hidden_states, hidden.index_select(0, result.kept_indices))
        stages.append(stage)
        return result

    handles = [decoder.layers[index].self_attn.q_proj.register_forward_pre_hook(observe(index)) for index in (0, 1)]
    handles.append(decoder.norm.register_forward_hook(lambda module, args, output: normalized.append(output.detach().clone())))
    sampled_inputs = {}
    def track(layer, hidden, **kwargs):
        sampled_inputs[list(decoder.layers).index(layer)] = layer.input_layernorm(hidden).detach().clone()
        return capture(layer, hidden, **kwargs)
    monkeypatch.setattr(provider, "_decoder_query_keys", track)
    monkeypatch.setattr(execution, "halfv_image_tokens", apply)
    activation = image_fixture.activation(model, _stage(backbone, boundaries, mid=mid))
    activation.install()
    if backbone == "llava_onevision":
        activation._bridge.prepare_training_forward()
    try:
        args = image_fixture.inputs(torch, backbone)
        labels = torch.full_like(args["input_ids"], -100)
        labels[0, -1] = args["input_ids"][0, -1]
        model(**args, labels=labels, use_cache=False)
        assert stages == [0, 1] and not pending
        for index in (0, 1):
            torch.testing.assert_close(sampled_inputs[index], native_inputs[index])
    finally:
        activation.uninstall()
        for handle in handles:
            handle.remove()
