"""PACT keeps its metric source while applying cluster means at a chosen boundary."""

import pytest

from mllm_comp.pipeline.stage import CompressionStageConfig
from tests.fixtures import native_image as image_fixture
from tests.integration import test_image_btp_positions_cpu as shared


def _stage(backbone, boundaries=(1,), *, sources=(0,), mid=False, phase="forward"):
    return CompressionStageConfig.from_dict({
        "name": "pact-positions", "placement": "llm_level", "execution_phase": phase,
        "plugin": "pact", "algorithm": "pact", "variant": "pact.image",
        "binding": f"pact.image.{backbone}", "target_modalities": ["image"],
        "config": {"reduction_layer": sources[0] + 1, "prune_keep_ratio": 0.75,
            "target_keep_ratio": 0.5, "cutoff": 0.5},
        "operation": {"targets": {"decoder_pact_boundary": {
            "target_site": "llm_decoder_layer_post_attention_image_tokens" if mid else "llm_decoder_layer_output_image_tokens",
            "layers": list(boundaries), **({} if mid else {"layer_edge": "before"})}}},
    })


@pytest.mark.parametrize("backbone", ["qwen3_vl", "qwen25_vl", "llava_onevision"])
@pytest.mark.parametrize("point,mid", [(1, False), (4, False), (1, True)])
def test_pact_native_application_and_mean_replay(backbone, point, mid):
    shared._run(backbone, (point,), (0,), mid, stage_factory=_stage,
        visual_counts="recorded", layer_local=True, checkpoint_modes=(None, False, True) if mid else (None,))


@pytest.mark.parametrize("backbone", ["qwen3_vl", "qwen25_vl", "llava_onevision"])
def test_pact_default_aliases(backbone):
    shared._default_aliases(backbone, _stage, "decoder_pact_boundary", (0,))


@pytest.mark.parametrize("backbone", ["qwen3_vl", "qwen25_vl", "llava_onevision"])
def test_pact_two_device_source_and_application(backbone):
    shared._run(backbone, (2,), (1,), True, stage_factory=_stage,
        visual_counts="recorded", layer_local=True, checkpoint_modes=(None,), two_devices=True)


@pytest.mark.parametrize("backbone", ["qwen3_vl", "qwen25_vl", "llava_onevision"])
@pytest.mark.parametrize("proportional", [False, True])
def test_pact_native_source_euti_cluster_means_and_bias(backbone, proportional, monkeypatch):
    import importlib
    import math

    torch = pytest.importorskip("torch")
    torch.manual_seed(367)
    model = image_fixture.model(backbone, num_decoder_layers=4).train()
    decoder = image_fixture.decoder(model, backbone)
    provider = importlib.import_module(f"mllm_comp.backbones.{backbone}.provider_bridge")
    adapter = importlib.import_module("mllm_comp.compressors.image.decoder_selection")
    source, native_inputs, applications, bias_calls = {}, {}, [], []
    project, kernel, bias = provider._decoder_projection_features, adapter.pact_image_tokens, adapter.pact_attention_mask

    def capture_project(layer, hidden, **kwargs):
        assert layer is decoder.layers[1]
        assert 1 not in native_inputs  # Full source 0, before the metric layer executes.
        metrics = project(layer, hidden, **kwargs)
        source.update(hidden=hidden.detach().clone(), metrics=metrics)
        return metrics

    def check_kernel(hidden, **kwargs):
        torch.testing.assert_close(source["hidden"], native_inputs[1], rtol=0, atol=0)
        q, k, clustering = source["metrics"]
        images = kwargs["image_indices"]
        logits = (q[:, images].mean(1, keepdim=True).float() / math.sqrt(q.shape[-1])) @ k[:, images].float().transpose(-1, -2)
        scores = logits.softmax(-1).mean(0).squeeze(0) * source["hidden"][0, images].float().norm(dim=-1)
        torch.testing.assert_close(kwargs["image_attention_scores"], scores)
        torch.testing.assert_close(kwargs["clustering_keys"], clustering, rtol=0, atol=0)
        assert not torch.allclose(hidden.detach(), source["hidden"][0])
        result = kernel(hidden, **kwargs)
        groups = [[i for i, target in result.layout_delta.old_to_new.items() if target == j] for j in range(result.layout_delta.result_count)]
        torch.testing.assert_close(result.hidden_states, torch.stack([hidden[group].mean(0) for group in groups]))
        torch.testing.assert_close(result.token_sizes, result.token_sizes.new_tensor([len(group) for group in groups]))
        assert result.layout_delta.result_layout["application_position"] == {"boundary": "post_attention_pre_ffn", "index": 2}
        applications.append(result.layout_delta)
        return result

    def check_bias(base, sizes, **kwargs):
        result = bias(base, sizes, **kwargs)
        assert applications and sizes.numel() == applications[0].result_count
        visible = (torch.ones(result.shape[-2:], dtype=torch.bool, device=result.device).tril().view(1, 1, *result.shape[-2:]).expand_as(result)
                   if base is None else base == 0)
        expected = sizes.float().log().to(result.device).view(1, 1, 1, -1).expand_as(result)
        torch.testing.assert_close((result if base is None else result - base)[visible], expected[visible])
        bias_calls.append(True)
        return result

    monkeypatch.setattr(provider, "_decoder_projection_features", capture_project)
    monkeypatch.setattr(adapter, "pact_image_tokens", check_kernel)
    monkeypatch.setattr(adapter, "pact_attention_mask", check_bias)
    handles = [layer.register_forward_pre_hook(lambda module, args, i=i: native_inputs.__setitem__(i, args[0].detach().clone())) for i, layer in enumerate(decoder.layers)]
    stage = _stage(backbone, (2,), mid=True)
    stage.config["proportional_attention"] = proportional
    activation = image_fixture.activation(model, stage)
    activation.install()
    if backbone == "llava_onevision":
        activation._bridge.prepare_training_forward()
    try:
        raw = image_fixture.inputs(torch, backbone)
        labels = torch.full_like(raw["input_ids"], -100)
        labels[0, -1] = raw["input_ids"][0, -1]
        result = model(**raw, labels=labels, use_cache=False)
        result.loss.backward()
        assert len(applications) == 1
        assert len(bias_calls) == int(proportional)  # Only layer 3; layer 2 Attention already ran.
        assert decoder.layers[2].mlp.gate_proj.weight.grad.abs().sum() > 0
    finally:
        activation.uninstall()
        for handle in handles:
            handle.remove()


@pytest.mark.parametrize("backbone", ["qwen3_vl", "qwen25_vl", "llava_onevision"])
@pytest.mark.parametrize("invalid", ["source", "mid_application"])
def test_pact_public_depth_error_and_install_cleanup(backbone, invalid):
    from mllm_comp.core.errors import BackendIntegrationError

    model = image_fixture.model(backbone, num_decoder_layers=4)
    depth = len(image_fixture.decoder(model, backbone).layers)
    stage = _stage(backbone, (depth,), sources=(depth - 1,) if invalid == "source" else (0,), mid=True)
    if invalid == "source":
        stage.operation = {}
    bridge = None
    if backbone == "llava_onevision":
        from mllm_comp.backbones.llava_onevision.provider_bridge import install_llava_onevision_ee14dbd_provider_bridge
        # Reusing a public bridge must recover even when activation does not own it.
        bridge = install_llava_onevision_ee14dbd_provider_bridge(model)
    original_forward = model.model.forward
    activation = None
    try:
        with pytest.raises(BackendIntegrationError):
            activation = image_fixture.activation(model, stage)
            activation.install()
        assert model.model.forward == original_forward
        if bridge is not None:
            assert not bridge._patches and not bridge._callbacks
    finally:
        if activation is not None:
            activation.uninstall()
        if bridge is not None:
            bridge.detach()


def test_llava_pact_bare_callback_preserves_default_boundary_and_qk():
    import copy
    from dataclasses import replace
    from mllm_comp.backbones.llava_onevision.image_compression import PYRAMID_DROP_SITE
    from mllm_comp.backbones.llava_onevision.provider_bridge import install_llava_onevision_ee14dbd_provider_bridge

    torch = pytest.importorskip("torch")
    torch.manual_seed(368)
    base = image_fixture.model("llava_onevision", num_decoder_layers=4).eval()
    raw = image_fixture.inputs(torch, "llava_onevision")
    expected, calls = None, []
    for bare in (False, True):
        model = copy.deepcopy(base)
        stage = _stage("llava_onevision")
        stage.operation = {}
        activation = image_fixture.activation(model, stage)
        bridge = handle = None
        if bare:
            # Keep native attention numerics identical to the ordinary activation.
            activation._decoder_attention_override.install()
            bridge = install_llava_onevision_ee14dbd_provider_bridge(model)

            def callback(request):
                assert request.layer_index == activation.spec.boundary_layer == 0
                assert request.query_states is not None and request.key_states is not None
                assert request.query_states.shape[-2] == request.hidden_states.shape[0]
                calls.append(request.layer_index)
                # Existing handlers can still compute their scores from the original Q/K.
                return activation.process_decoder_layer(replace(request, image_attention_scores=None))

            callback._mllm_algorithm = "pact"
            callback._mllm_prune_layer = activation.spec.boundary_layer
            callback._mllm_pact_spec = activation.spec
            callback._mllm_execution_phase = "forward"
            handle = bridge.register_callback(PYRAMID_DROP_SITE, callback)
        else:
            activation.install()
        try:
            torch.manual_seed(369)
            with torch.no_grad():
                logits = model(**raw, use_cache=False).logits
            if expected is None:
                expected = logits
            else:
                torch.testing.assert_close(logits, expected, rtol=0, atol=0)
        finally:
            activation.uninstall()
            if handle is not None:
                handle.remove()
            if bridge is not None:
                bridge.detach()
    assert calls == [0]
