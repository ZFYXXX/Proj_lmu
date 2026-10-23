"""DART reuses decoder boundaries while keeping its completed-layer context."""

import pytest

from tests.fixtures import native_image as image_fixture
from tests.integration import test_image_fastv_positions_cpu as checks


_fastv_stage = checks._stage


def _stage(backbone, boundary, *, prune_layer=2, phase="forward", ratio=0.5, mid=False):
    stage = _fastv_stage(backbone, boundary, prune_layer=prune_layer, phase=phase, mid=mid)
    stage.name = "dart-position"
    stage.plugin = stage.algorithm = "dart"
    stage.variant = "dart.image"
    stage.binding = f"dart.image.{backbone}"
    stage.config = {"prune_after_layer": prune_layer, "pivot_image_tokens": 1,
        "pivot_text_tokens": 1, "retain_ratio": 1.0 if ratio == 0 else 0.5}
    if ratio != 0:
        stage.config["target_visual_tokens"] = 364 if backbone == "llava_onevision" else 2
    return stage


@pytest.mark.parametrize("backbone", ["qwen3_vl", "qwen25_vl", "llava_onevision"])
@pytest.mark.parametrize("boundary,source,mid", [(1, 1, False), (4, 2, False), (1, 1, True), (4, 2, True)])
def test_dart_native_lengths_gradients_checkpoint_and_generate(backbone, boundary, source, mid, monkeypatch):
    monkeypatch.setattr(checks, "_stage", _stage)
    checks.test_native_fastv_application_lengths_gradient_and_checkpoint(backbone, boundary, source, mid)
    checks.test_native_fastv_prefill_decode_cache_and_repeated_generate(backbone, boundary, source, mid)


@pytest.mark.parametrize("backbone", ["qwen3_vl", "qwen25_vl", "llava_onevision"])
@pytest.mark.parametrize("mid", [False, True])
def test_dart_two_device_source_application_checkpoint_and_cache(backbone, mid, monkeypatch):
    def stage(*args, **kwargs):
        kwargs["prune_layer"] = 2
        return _stage(*args, **kwargs)
    monkeypatch.setattr(checks, "_stage", stage)
    checks.test_fastv_source_at_two_device_layer_transition(backbone, mid)


@pytest.mark.parametrize("backbone", ["qwen3_vl", "qwen25_vl", "llava_onevision"])
def test_dart_default_and_equivalent_boundaries_match(backbone, monkeypatch):
    monkeypatch.setattr(checks, "_stage", _stage)
    checks.test_fastv_legacy_and_equivalent_positions_keep_outputs_and_gradients(backbone)


@pytest.mark.parametrize("backbone", ["qwen3_vl", "qwen25_vl", "llava_onevision"])
def test_dart_mid_identity_preserves_all_native_gradients_and_hooks(backbone, monkeypatch):
    monkeypatch.setattr(checks, "_stage", _stage)
    checks.test_fastv_mid_identity_preserves_native_residual_hooks_and_all_gradients(backbone, 1)


@pytest.mark.parametrize("backbone", ["qwen3_vl", "qwen25_vl", "llava_onevision"])
@pytest.mark.parametrize("boundary,source,mid", [(1, 1, False), (4, 2, False), (1, 1, True), (4, 2, True)])
def test_dart_fixed_source_matches_native_context_and_pivot_selection(backbone, boundary, source, mid, monkeypatch):
    import copy
    import importlib
    torch = pytest.importorskip("torch")
    torch.manual_seed(161)
    reference = image_fixture.model(backbone).eval()
    model = copy.deepcopy(reference)
    decoder = image_fixture.decoder(reference, backbone)
    boundary = min(boundary, len(decoder.layers) - int(mid))
    native = {}

    def capture(name):
        def hook(module, args, kwargs):
            native[name] = (args[0] if args else kwargs["hidden_states"]).detach().clone(), kwargs
        return hook

    application = decoder.layers[boundary].post_attention_layernorm if mid else (
        decoder.layers[boundary] if boundary < len(decoder.layers) else decoder.norm)
    handles = [module.register_forward_pre_hook(capture(name), with_kwargs=True) for name, module in (
        ("source_input", decoder.layers[source - 1]), ("completed_source", decoder.layers[source]),
        ("application", application))]
    args = image_fixture.inputs(torch, backbone)
    try:
        with torch.no_grad():
            reference(**args, use_cache=False)
    finally:
        for handle in handles:
            handle.remove()
    with torch.no_grad():
        normalized = decoder.norm(native["completed_source"][0])[0]
    provider = importlib.import_module(f"mllm_comp.backbones.{backbone}.provider_bridge")
    adapter = importlib.import_module("mllm_comp.compressors.image.decoder_selection")
    project, kernel = provider._decoder_query_keys, adapter.dart_image_tokens
    calls = []

    def check_source(layer, hidden, **kwargs):
        assert layer is image_fixture.decoder(model, backbone).layers[source - 1]
        torch.testing.assert_close(hidden, native["source_input"][0], atol=2e-6, rtol=2e-5)
        calls.append("source")
        return project(layer, hidden, **kwargs)

    def check_kernel(hidden, normalized_hidden, keys, **kwargs):
        torch.testing.assert_close(hidden, native["application"][0][0], atol=2e-6, rtol=2e-5)
        torch.testing.assert_close(normalized_hidden, normalized, atol=2e-6, rtol=2e-5)
        images, texts = kwargs["image_indices"], kwargs["text_candidate_indices"]
        _, _, native_keys = project(decoder.layers[source - 1], native["source_input"][0],
            instruction_indices=images[:1], image_indices=images,
            position_embeddings=native["source_input"][1]["position_embeddings"])
        torch.testing.assert_close(keys, native_keys, atol=2e-6, rtol=2e-5)
        result = kernel(hidden, normalized_hidden, keys, **kwargs)
        norms = native_keys.reshape(len(hidden), -1).abs().sum(-1)
        pivot = images[norms[images].argmax()]
        text_pivot = texts[norms[texts].argmax()]
        torch.testing.assert_close(result.image_pivot_indices, pivot.reshape(1))
        torch.testing.assert_close(result.text_pivot_indices, text_pivot.reshape(1))
        candidates = images[images != pivot]
        unit = torch.nn.functional.normalize(normalized, dim=-1)
        second = candidates[(unit[candidates] @ unit[pivot]).argmin()]
        selected = torch.stack((pivot, second)).sort().values
        torch.testing.assert_close(result.kept_image_indices, selected)
        torch.testing.assert_close(result.hidden_states, hidden[result.kept_indices], atol=0, rtol=0)
        calls.append("apply")
        return result

    monkeypatch.setattr(provider, "_decoder_query_keys", check_source)
    monkeypatch.setattr(adapter, "dart_image_tokens", check_kernel)
    stage = _stage(backbone, boundary, prune_layer=source, mid=mid)
    stage.config["target_visual_tokens"] = 2
    activation = image_fixture.activation(model, stage)
    activation.install()
    try:
        with torch.no_grad():
            model(**args, use_cache=False)
        assert calls == ["source", "apply"]
    finally:
        activation.uninstall()


@pytest.mark.parametrize("backbone", ["qwen3_vl", "qwen25_vl", "llava_onevision"])
def test_dart_terminal_source_and_application(backbone, monkeypatch):
    monkeypatch.setattr(checks, "_stage", _stage)
    depth = 3 if backbone == "llava_onevision" else 4
    checks.test_native_fastv_application_lengths_gradient_and_checkpoint(backbone, depth, depth, mid=False)
    checks.test_native_fastv_prefill_decode_cache_and_repeated_generate(backbone, depth, depth, mid=False)
