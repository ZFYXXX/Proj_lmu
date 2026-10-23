"""Pinned LLaVA bridge contracts against a locally constructed native model.

Algorithm numerics, gradients and decoder replay live in the native image and
projector tests. Here we check lifecycle, callback API and injected failures.
"""

from functools import wraps
import inspect
from types import SimpleNamespace

import pytest

from mllm_comp.backbones.llava_onevision import image_compression as compression
from mllm_comp.backbones.llava_onevision import provider_bridge as provider
from mllm_comp.core.errors import BackendIntegrationError, InvariantError
from mllm_comp.pipeline.stage import CompressionStageConfig
from tests.fixtures import native_image

torch = pytest.importorskip("torch")
pytest.importorskip("llava")


def _activation(model, algorithm, config, *, phase="forward"):
    stage = CompressionStageConfig.from_dict({
        "name": algorithm,
        "placement": "encoder_level" if algorithm == "tome" else "llm_level" if algorithm in {"fastv", "pyramid_drop"} else "projector_level",
        "execution_phase": phase, "plugin": algorithm, "algorithm": algorithm,
        "variant": "tome.image.encoder_dense_boundary" if algorithm == "tome" else f"{algorithm}.image",
        "binding": f"{algorithm}.image.llava_onevision",
        "target_modalities": ["image"], "config": config,
    })
    return native_image.activation(model, stage)


def test_tome_training_preparation_and_instance_patches_are_reversible():
    model = native_image.llava_model().train()
    encoder = model.get_vision_tower().vision_tower.vision_model.encoder
    original_signature = inspect.signature(model.forward)
    original = {module: module.forward.__func__ for module in (model, model.model, encoder, *encoder.layers)}
    activation = _activation(model, "tome", {"r": 1, "layers": [0]})
    activation.install()
    bridge = activation._bridge
    try:
        assert inspect.signature(model.forward) == original_signature
        assert bridge.patched
        assert set(bridge.runtime_metadata()["active_sites"]) == {
            compression.TOME_MERGE_SITE, compression.TOME_RESTORE_SITE, compression.TOME_PACK_SITE,
        }
        encoder.gradient_checkpointing = True
        bridge.prepare_training_forward()
        assert not encoder.gradient_checkpointing
    finally:
        activation.uninstall()
    assert not bridge.patched
    assert "generate" not in vars(model)
    for module, forward in original.items():
        assert "forward" not in vars(module)
        assert module.forward.__func__ is forward
    bridge.detach()
    assert not hasattr(model, "mllm_compression_bridge")


def test_forward_bridge_rejects_cache_video_generate_and_invalid_train_output(monkeypatch):
    model = native_image.llava_model().eval()
    original_forward = model.forward
    calls = []

    @wraps(original_forward)
    def invalid_training_output(*args, **kwargs):
        calls.append(kwargs)
        return torch.zeros(1, 2)

    monkeypatch.setattr(model, "forward", invalid_training_output)
    activation = _activation(model, "tome", {"r": 1, "layers": [0]})
    activation.install()
    args = native_image.inputs(torch, "llava_onevision")
    try:
        with pytest.raises(BackendIntegrationError, match="use_cache=False"):
            model(**args, use_cache=True)
        with pytest.raises(InvariantError, match="rank-3 logits"):
            model(**args, use_cache=False, labels=args["input_ids"].masked_fill(args["input_ids"] == -200, -100))
        assert calls[-1]["labels"] is None  # The bridge owns label repair and loss.
        with pytest.raises(BackendIntegrationError, match="image-only"):
            model(**{**args, "modalities": ["video"]}, use_cache=False)
        with pytest.raises(BackendIntegrationError, match="does not support generate"):
            model.generate(images=args["images"])
    finally:
        activation.uninstall()


@pytest.mark.parametrize("algorithm", ["vispruner", "zoo_prune"])
def test_provider_captures_pinned_siglip_and_projector_sources(algorithm):
    model = native_image.llava_model().eval()
    projector = torch.nn.Sequential(model.model.mm_projector)
    model.model.mm_projector = projector
    tower = model.get_vision_tower()
    assert not hasattr(tower, "select_layer")
    config = {"tokens_per_crop": 2} if algorithm == "vispruner" else {
        "target_count": 2, "num_directions": 2, "direction_batch_size": 1, "projector_proxy": "full",
    }
    activation = _activation(model, algorithm, config)
    activation.install()
    bridge = activation._bridge
    outputs = []
    hook = tower.vision_tower.register_forward_hook(lambda module, args, output: outputs.append(output))
    previous = bridge._vision_source_capture.set(None)
    try:
        projected = model.encode_images(torch.randn(1, 3, 54, 54))
        capture = bridge._vision_source_capture.get()
        assert capture is not None
        torch.testing.assert_close(capture.projected_crops, projected)
        torch.testing.assert_close(capture.vision_crops, outputs[0].hidden_states[-1])
        if algorithm == "vispruner":
            torch.testing.assert_close(capture.attentions, outputs[0].attentions[-1])
        else:
            assert capture.projector_layers == (projector[0],)
    finally:
        hook.remove()
        activation.uninstall()
        bridge._vision_source_capture.reset(previous)


@pytest.mark.parametrize("algorithm,prune_layer", [
    ("fastv", 2), ("fastv", 4), ("pyramid_drop", None),
], ids=["fastv", "fastv-terminal-boundary", "pyramid-drop"])
def test_raw_decoder_callback_keeps_public_qk_requests_and_selection(algorithm, prune_layer):
    torch.manual_seed(527)
    model = native_image.llava_model(num_decoder_layers=4).eval()
    config = {"prune_layer": prune_layer, "prune_ratio": 0.5} if algorithm == "fastv" else {"keep_ratio": 0.5, "num_stages": 4}
    formal = _activation(model, algorithm, config)
    formal.install()
    bridge = formal._bridge
    args = native_image.inputs(torch, "llava_onevision")
    try:
        with torch.no_grad():
            expected = model(**args, use_cache=False).logits
    finally:
        formal.uninstall()
    activation = _activation(model, algorithm, config)
    site = compression.FASTV_SITE if algorithm == "fastv" else compression.PYRAMID_DROP_SITE
    handler = activation._handler_for_site(site)
    requests, positions = [], [[] for _ in model.model.layers]

    def callback(request):
        requests.append(request)
        assert request.attention_keys is not None
        if algorithm == "pyramid_drop":
            assert request.instruction_query is not None
            assert request.image_keys is not None
            assert request.image_attention_scores is None
        else:
            assert request.scorer_query is not None
        return handler(request)

    callback._mllm_algorithm = algorithm
    callback._mllm_execution_phase = "forward"
    if algorithm == "fastv":
        callback._mllm_prune_layer = prune_layer
    hooks = [layer.register_forward_pre_hook(
        lambda module, args, kwargs, i=i: positions[i].append(kwargs["position_ids"].clone()), with_kwargs=True,
    ) for i, layer in enumerate(model.model.layers)]
    handle = bridge.register_callback(site, callback)
    try:
        with torch.no_grad():
            output = model(**args, use_cache=False)
        torch.testing.assert_close(output.logits, expected)
        if algorithm == "pyramid_drop":
            assert [request.layer_index for request in requests] == list(range(4))
            for request in requests:
                assert request.position_ids.tolist() == [list(range(request.hidden_states.shape[0]))]
        else:
            assert [request.prune_layer for request in requests] == [prune_layer]
            if prune_layer == 4:
                assert output.logits.shape[1] < 732
                assert all(values[-1].shape[-1] == 732 for values in positions)
    finally:
        handle.remove()
        for hook in hooks:
            hook.remove()


@pytest.mark.parametrize("boundary,offset", [("inter_layer", 1), ("post_attention_pre_ffn", 0)])
def test_raw_fastv_callback_rejects_unavailable_terminal_boundaries(boundary, offset):
    model = native_image.llava_model()
    bridge = provider.PinnedLlavaOneVisionEe14dbdBridge(model)
    prune_layer = len(model.model.layers) + offset

    def callback(request):
        pytest.fail("An unavailable decoder boundary must be rejected at registration.")

    callback._mllm_algorithm = "fastv"
    callback._mllm_execution_phase = "forward"
    callback._mllm_prune_layer = prune_layer
    callback._mllm_application_position = provider.LayerPosition(boundary, prune_layer)
    with pytest.raises(BackendIntegrationError, match="requires a valid boundary"):
        bridge.register_callback(compression.FASTV_SITE, callback)
    assert not bridge.patched


def test_provider_patches_are_reentrant_across_distinct_algorithms():
    model = native_image.llava_model(num_decoder_layers=4).eval()
    args = native_image.inputs(torch, "llava_onevision")
    for algorithm, config in (("visionzip", {"budget": 5}), ("pyramid_drop", {"keep_ratio": 0.5, "num_stages": 4})):
        activation = _activation(model, algorithm, config)
        activation.install()
        bridge = activation._bridge
        try:
            assert bridge.patched
            assert "prepare_inputs_labels_for_multimodal" in vars(model)
            assert "forward" in vars(model.model)
            with torch.no_grad():
                assert model(**args, use_cache=False).logits.shape[1] < 732
        finally:
            activation.uninstall()
        assert not bridge.patched
        assert "prepare_inputs_labels_for_multimodal" not in vars(model)
        assert "forward" not in vars(model.model)


def test_production_constructor_rejects_non_pinned_class_before_patching():
    native = native_image.llava_model()
    model = SimpleNamespace(model=native.model, config=native.config, forward=native.forward,
        get_model=native.get_model, get_vision_tower=native.get_vision_tower,
        prepare_inputs_labels_for_multimodal=native.prepare_inputs_labels_for_multimodal, generate=native.generate)
    original = model.forward
    with pytest.raises(BackendIntegrationError, match="class guard failed"):
        provider.PinnedLlavaOneVisionEe14dbdBridge(model)
    assert model.forward is original
    assert "forward" not in vars(native.model)


def test_same_site_duplicate_activation_is_rejected_without_lifecycle_damage():
    model = native_image.llava_model()
    bridge = provider.PinnedLlavaOneVisionEe14dbdBridge(model)
    handle = bridge.register_callback(compression.TOME_MERGE_SITE, lambda request: request)
    try:
        with pytest.raises(BackendIntegrationError, match="already has an activation"):
            bridge.register_callback(compression.TOME_MERGE_SITE, lambda request: request)
        assert bridge.patched
    finally:
        handle.remove()
    assert not bridge.patched
