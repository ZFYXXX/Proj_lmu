"""IVC keeps completed-layer V/coordinate scores while moving application."""

import pytest

from mllm_comp.compressors.image import decoder_selection

from tests.fixtures import native_image as image_fixture
from tests.integration import test_image_fastv_positions_cpu as checks


_fastv_stage = checks._stage


def _stage(backbone, boundary, *, prune_layer=2, phase="forward", ratio=0.5, mid=False):
    stage = _fastv_stage(backbone, boundary, prune_layer=prune_layer, phase=phase, mid=mid)
    stage.name = "ivc-position"
    stage.plugin = stage.algorithm = "ivc_prune"
    stage.variant = "ivc_prune.image"
    stage.binding = f"ivc_prune.image.{backbone}"
    target = "native_rope_coordinate_prune" if backbone == "llava_onevision" else "native_mrope_coordinate_prune"
    stage.operation["targets"][target] = stage.operation["targets"].pop("decoder_prune_boundary")
    stage.config = {"prune_layer": prune_layer, "total_retain_ratio": 1.0 if ratio == 0 else 0.5,
        "axis_ratio_each": 0.1, "semantic_seed_ratio": 0.01, "post_image_text_skip": 0}
    return stage


@pytest.mark.parametrize("backbone", ["qwen3_vl", "qwen25_vl", "llava_onevision"])
@pytest.mark.parametrize("boundary,source,mid", [(1, 1, False), (4, 2, False), (1, 1, True), (4, 2, True)])
def test_ivc_native_lengths_gradients_checkpoint_and_generate(backbone, boundary, source, mid, monkeypatch):
    monkeypatch.setattr(checks, "_stage", _stage)
    checks.test_native_fastv_application_lengths_gradient_and_checkpoint(backbone, boundary, source, mid)
    checks.test_native_fastv_prefill_decode_cache_and_repeated_generate(backbone, boundary, source, mid)


@pytest.mark.parametrize("backbone", ["qwen3_vl", "qwen25_vl", "llava_onevision"])
@pytest.mark.parametrize("mid", [False, True])
def test_ivc_two_device_source_application_checkpoint_and_cache(backbone, mid, monkeypatch):
    def stage(*args, **kwargs):
        kwargs["prune_layer"] = 2
        return _stage(*args, **kwargs)
    monkeypatch.setattr(checks, "_stage", stage)
    checks.test_fastv_source_at_two_device_layer_transition(backbone, mid)


@pytest.mark.parametrize("backbone", ["qwen3_vl", "qwen25_vl", "llava_onevision"])
def test_ivc_default_and_equivalent_boundaries_match(backbone, monkeypatch):
    monkeypatch.setattr(checks, "_stage", _stage)
    checks.test_fastv_legacy_and_equivalent_positions_keep_outputs_and_gradients(backbone)


@pytest.mark.parametrize("backbone", ["qwen3_vl", "qwen25_vl", "llava_onevision"])
def test_ivc_mid_identity_preserves_all_native_gradients_and_hooks(backbone, monkeypatch):
    monkeypatch.setattr(checks, "_stage", _stage)
    checks.test_fastv_mid_identity_preserves_native_residual_hooks_and_all_gradients(backbone, 1)


@pytest.mark.parametrize("backbone", ["qwen3_vl", "qwen25_vl", "llava_onevision"])
def test_ivc_terminal_source_and_application(backbone, monkeypatch):
    monkeypatch.setattr(checks, "_stage", _stage)
    depth = 3 if backbone == "llava_onevision" else 4
    checks.test_native_fastv_application_lengths_gradient_and_checkpoint(backbone, depth, depth, mid=False)
    checks.test_native_fastv_prefill_decode_cache_and_repeated_generate(backbone, depth, depth, mid=False)


@pytest.mark.parametrize("backbone", ["qwen3_vl", "qwen25_vl", "llava_onevision"])
@pytest.mark.parametrize("boundary,source,mid", [(1, 1, False), (4, 2, False), (1, 1, True), (4, 2, True)])
def test_ivc_fixed_native_source_application_and_foreground_math(backbone, boundary, source, mid, monkeypatch):
    import copy
    torch = pytest.importorskip("torch")
    torch.manual_seed(251)
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
    source_layer = decoder.layers[source - 1]
    with torch.no_grad():
        native_values = source_layer.self_attn.v_proj(source_layer.input_layernorm(native["completed_source"][0]))[0]
    project, kernel = decoder_selection.project_ivc_value_states, decoder_selection.ivc_prune_image_tokens
    calls = []

    def check_source(hidden, layer):
        assert layer is image_fixture.decoder(model, backbone).layers[source - 1]
        torch.testing.assert_close(hidden, native["completed_source"][0][0], atol=2e-6, rtol=2e-5)
        calls.append("source")
        return project(hidden, layer)

    def check_kernel(hidden=None, **kwargs):
        hidden = kwargs.pop("hidden_states", hidden)
        torch.testing.assert_close(hidden, native["application"][0][0], atol=2e-6, rtol=2e-5)
        torch.testing.assert_close(kwargs["value_states"], native_values, atol=2e-6, rtol=2e-5)
        images, texts = kwargs["image_indices"], kwargs["text_query_indices"]
        planes = native["source_input"][1]["position_embeddings"]
        expected_coordinates = []
        for name, plane in zip(("cosine_scores", "sine_scores"), planes):
            axis = [i for i, n in enumerate(plane.shape) if n == len(hidden)][-1]
            selected = plane.float().index_select(axis, images)
            coordinate = selected.mean(tuple(i for i in range(plane.ndim) if i != axis))
            torch.testing.assert_close(kwargs[name], coordinate, atol=2e-6, rtol=2e-5)
            expected_coordinates.append(coordinate)
        spec = kwargs["spec"]
        values = native_values.float()
        visual = values[images]
        seed_scores = (values[texts] @ visual.T / values.shape[-1]**0.5).softmax(-1).mean(0)
        seed = seed_scores.topk(max(1, round(len(images) * spec.semantic_seed_ratio)), sorted=False).indices
        foreground = torch.cat((values[texts], visual[seed]))
        scores = (foreground @ visual.T / values.shape[-1]**0.5).softmax(-1).mean(0)
        axis_count = round(len(images) * spec.axis_ratio_each)
        axes = [c.topk(axis_count, sorted=False).indices for c in expected_coordinates]
        priority = torch.cat((*axes, seed)).unique(sorted=False)
        target = round(len(images) * spec.total_retain_ratio)
        ranked = scores.argsort(descending=True)
        selected = torch.cat((priority, ranked[~torch.isin(ranked, priority)][:target-len(priority)])).unique(sorted=True)
        result = kernel(hidden, **kwargs)
        assert result.metadata["application_position"] == {"boundary": "post_attention_pre_ffn" if mid else "inter_layer", "index": boundary}
        assert "before_layer" not in result.metadata
        torch.testing.assert_close(result.selected_visual_indices, selected)
        torch.testing.assert_close(result.hidden_states, hidden[result.kept_indices], atol=0, rtol=0)
        calls.append("apply")
        return result

    monkeypatch.setattr(decoder_selection, "project_ivc_value_states", check_source)
    monkeypatch.setattr(decoder_selection, "ivc_prune_image_tokens", check_kernel)
    activation = image_fixture.activation(model, _stage(backbone, boundary, prune_layer=source, mid=mid))
    activation.install()
    try:
        with torch.no_grad():
            model(**args, use_cache=False)
        assert calls == ["source", "apply"]
        if backbone == "llava_onevision":
            expected_site = "llm_decoder_layer_post_attention_image_tokens" if mid else "llm_decoder_layer_output_image_tokens"
            assert activation.runtime_metadata()["layout_delta_records"][0]["site"] == expected_site

    finally:
        activation.uninstall()


@pytest.mark.parametrize("backbone", ["qwen3_vl", "qwen25_vl", "llava_onevision"])
@pytest.mark.parametrize("invalid", ["source", "mid_application"])
def test_ivc_public_depth_validation_preserves_backend_error(backbone, invalid):
    from mllm_comp.core.errors import BackendIntegrationError
    model = image_fixture.model(backbone)
    depth = len(image_fixture.decoder(model, backbone).layers)
    stage = _stage(backbone, depth, prune_layer=depth + 1 if invalid == "source" else 1, mid=True)
    if invalid == "source":
        stage.operation = {}
    activation = None
    try:
        with pytest.raises(BackendIntegrationError):
            activation = image_fixture.activation(model, stage)
            activation.install()
    finally:
        if activation is not None:
            activation.uninstall()
