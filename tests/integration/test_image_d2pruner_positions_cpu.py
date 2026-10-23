"""D²Pruner separates its fixed attention/prior source from application."""

import pytest

from tests.fixtures import native_image as image_fixture
from tests.integration import test_image_fastv_positions_cpu as checks


_fastv_stage = checks._stage


@pytest.fixture
def d2_stages(monkeypatch, tmp_path):
    torch = pytest.importorskip("torch")

    def stage(backbone, boundary, *, prune_layer=2, phase="forward", ratio=0.5, mid=False):
        prior = tmp_path / f"prior-{prune_layer}.pt"
        torch.save({"bias": torch.linspace(0.5, 1.5, 16), "model_id": "/fixtures/native-tiny-image",
            "scorer_layer": prune_layer - 1}, prior)
        stage = _fastv_stage(backbone, boundary, prune_layer=prune_layer, phase=phase, mid=mid)
        stage.name = "d2pruner-position"
        stage.plugin = stage.algorithm = "d2pruner"
        stage.variant = "d2pruner.image"
        stage.binding = f"d2pruner.image.{backbone}"
        stage.config = {"prune_after_layer": prune_layer, "bias_prior_path": str(prior),
            "retain_ratio": 1.0 if ratio == 0 else 0.5}
        return stage

    monkeypatch.setattr(checks, "_stage", stage)
    return stage


@pytest.mark.parametrize("backbone", ["qwen3_vl", "qwen25_vl", "llava_onevision"])
@pytest.mark.parametrize("boundary,source,mid", [
    (0, 1, False), (0, 1, True), (1, 2, False), (1, 2, True), (4, 2, False), (4, 2, True)])
def test_d2_native_lengths_gradients_checkpoint_cache_generate(backbone, boundary, source, mid, d2_stages):
    checks.test_native_fastv_application_lengths_gradient_and_checkpoint(backbone, boundary, source, mid)
    checks.test_native_fastv_prefill_decode_cache_and_repeated_generate(backbone, boundary, source, mid)


@pytest.mark.parametrize("backbone", ["qwen3_vl", "qwen25_vl", "llava_onevision"])
def test_d2_default_and_equivalent_boundaries_match(backbone, d2_stages):
    checks.test_fastv_legacy_and_equivalent_positions_keep_outputs_and_gradients(backbone)


@pytest.mark.parametrize("backbone", ["qwen3_vl", "qwen25_vl", "llava_onevision"])
@pytest.mark.parametrize("boundary", [0, 1])
def test_d2_identity_preserves_all_native_gradients_and_hooks(backbone, boundary, d2_stages):
    checks.test_fastv_mid_identity_preserves_native_residual_hooks_and_all_gradients(backbone, boundary)


@pytest.mark.parametrize("backbone", ["qwen3_vl", "qwen25_vl", "llava_onevision"])
@pytest.mark.parametrize("mid", [False, True])
def test_d2_two_device_checkpoint_and_cache(backbone, mid, d2_stages):
    checks.test_fastv_source_at_two_device_layer_transition(backbone, mid)


@pytest.mark.parametrize("backbone", ["qwen3_vl", "qwen25_vl", "llava_onevision"])
@pytest.mark.parametrize("boundary,source,mid", [(0, 1, False), (0, 1, True), (4, 2, False), (4, 2, True)])
def test_d2_native_fixed_attention_and_bias_at_actual_boundary(backbone, boundary, source, mid, d2_stages, monkeypatch):
    import copy
    import importlib
    torch = pytest.importorskip("torch")
    torch.manual_seed(171)
    reference = image_fixture.model(backbone).eval()
    model = copy.deepcopy(reference)
    decoder = image_fixture.decoder(reference, backbone)
    boundary = min(boundary, len(decoder.layers) - int(mid))
    native = {}

    def capture(name):
        def hook(module, args, kwargs):
            native[name] = (args[0] if args else kwargs["hidden_states"]).detach().clone()
        return hook

    application = decoder.layers[boundary].post_attention_layernorm if mid else (
        decoder.layers[boundary] if boundary < len(decoder.layers) else decoder.norm)
    handles = [module.register_forward_pre_hook(capture(name), with_kwargs=True) for name, module in (
        ("source", decoder.layers[source - 1]), ("application", application))]
    args = image_fixture.inputs(torch, backbone)
    try:
        with torch.no_grad():
            reference(**args, use_cache=False)
    finally:
        for handle in handles:
            handle.remove()
    provider = importlib.import_module(f"mllm_comp.backbones.{backbone}.provider_bridge")
    adapter = importlib.import_module("mllm_comp.compressors.image.decoder_selection")
    project, kernel = provider._decoder_query_keys, adapter.d2pruner_image_tokens
    expected_scores = []
    selections = []

    def check_source(layer, hidden, **kwargs):
        assert layer is image_fixture.decoder(model, backbone).layers[source - 1]
        torch.testing.assert_close(hidden, native["source"], atol=2e-6, rtol=2e-5)
        assert kwargs["instruction_indices"].tolist() == [hidden.shape[1] - 1]
        query, image_keys, keys = project(layer, hidden, **kwargs)
        # The last prompt token sees every earlier token in this dense fixture.
        logits = torch.einsum("qhd,thd->qht", query, keys) / query.shape[-1] ** 0.5
        expected_scores.append(logits.float().softmax(-1).mean((0, 1))[kwargs["image_indices"]])
        return query, image_keys, keys

    def check_kernel(hidden, **kwargs):
        torch.testing.assert_close(hidden, native["application"][0], atol=2e-6, rtol=2e-5)
        torch.testing.assert_close(kwargs["image_attention_scores"], expected_scores[0])
        assert kwargs["bias_metadata"]["validated_scorer_layer"] == source - 1
        assert kwargs["bias_metadata"]["validated_model_identity"] == "/fixtures/native-tiny-image"
        result = kernel(hidden, **kwargs)
        prior = torch.nn.functional.interpolate(torch.linspace(0.5, 1.5, 16).view(1, 1, -1),
            size=len(expected_scores[0]), mode="linear", align_corners=False).flatten()
        torch.testing.assert_close(result.relative_scores, expected_scores[0] / (prior + kwargs["spec"].epsilon))
        torch.testing.assert_close(result.hidden_states, hidden[result.kept_indices], atol=0, rtol=0)
        selections.append(result.kept_indices)
        return result

    monkeypatch.setattr(provider, "_decoder_query_keys", check_source)
    monkeypatch.setattr(adapter, "d2pruner_image_tokens", check_kernel)
    activation = image_fixture.activation(model, d2_stages(backbone, boundary, prune_layer=source, mid=mid))
    activation.install()
    try:
        with torch.no_grad():
            model(**args, use_cache=False)
        assert len(expected_scores) == len(selections) == 1
    finally:
        activation.uninstall()
