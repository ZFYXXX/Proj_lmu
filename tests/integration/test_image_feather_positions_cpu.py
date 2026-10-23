"""FEATHER keeps two scoring stages while application positions move."""

import copy

import pytest

from mllm_comp.compressors.image import decoder_selection

from mllm_comp.pipeline.stage import CompressionStageConfig
from tests.fixtures import native_image as image_fixture


def _model(backbone):
    return image_fixture.llava_model(num_decoder_layers=4) if backbone == "llava_onevision" else image_fixture.model(backbone)


def _stage(backbone, boundaries=(2, 3), *, sources=(1, 2), mid=False, phase="forward", identity=False):
    return CompressionStageConfig.from_dict({
        "name": "feather-positions", "placement": "llm_level", "execution_phase": phase,
        "plugin": "feather", "algorithm": "feather", "variant": "feather.image",
        "binding": f"feather.image.{backbone}", "target_modalities": ["image"],
        "operation": {"type": "prune", "targets": {"two_stage_no_rope_prune": {
            "target_site": "llm_decoder_layer_post_attention_image_tokens" if mid else "llm_decoder_layer_output_image_tokens",
            "layers": list(boundaries), **({} if mid else {"layer_edge": "before"})}}},
        "config": {"stage_layers": list(sources), "reduction_ratio": 0.5, "uniform_stride": 3,
            "target_final_retain_ratio": 1.0 if identity else 0.5},
    })


def _run(backbone, boundaries, sources, mid, *, two_devices=False, identity=False):
    torch = pytest.importorskip("torch")
    if two_devices and torch.cuda.device_count() < 2:
        pytest.skip("Explicit two-device FEATHER source/application test.")
    torch.manual_seed(272)
    base = _model(backbone).train()
    expected = None
    for checkpoint in (None, False, True):
        model = copy.deepcopy(base)
        decoder = image_fixture.decoder(model, backbone)
        if two_devices:
            from accelerate import dispatch_model
            prefix = "model" if backbone == "llava_onevision" else "model.language_model"
            device_map = {name: "cuda:0" for name, _ in model.named_children() if name != "model"}
            if backbone != "llava_onevision":
                device_map.update({f"model.{name}": "cuda:0" for name, _ in model.model.named_children() if name != "language_model"})
            device_map.update({f"{prefix}.{name}": "cuda:0" for name, _ in decoder.named_children() if name != "layers"})
            device_map.update({f"{prefix}.layers.{i}": "cuda:0" if i < 2 else "cuda:1" for i in range(len(decoder.layers))})
            dispatch_model(model, device_map=device_map)
        if checkpoint is not None:
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": checkpoint})
        activation = image_fixture.activation(model, _stage(backbone, boundaries, sources=sources, mid=mid, identity=identity))
        activation.install()
        if backbone == "llava_onevision":
            activation._bridge.prepare_training_forward()
        torch.manual_seed(273)
        raw = image_fixture.inputs(torch, backbone)
        args = {name: value.to("cuda:0") if two_devices and torch.is_tensor(value) else value for name, value in raw.items()}
        labels = torch.full_like(args["input_ids"], -100)
        labels[0, -1] = args["input_ids"][0, -1]
        attention, mlp = [[] for _ in decoder.layers], [[] for _ in decoder.layers]
        handles = []
        for i, layer in enumerate(decoder.layers):
            handles.append(layer.self_attn.o_proj.register_forward_pre_hook(lambda module, args, i=i: attention[i].append(args[0].shape[1])))
            handles.append(layer.mlp.register_forward_pre_hook(lambda module, args, i=i: mlp[i].append(args[0].shape[1])))
        try:
            result = model(**args, labels=labels, use_cache=False)
            records = activation.runtime_metadata()["layout_delta_records"]
            assert len(records) == 2
            initial = 732 if backbone == "llava_onevision" else 8
            final = initial if identity else (367 if backbone == "llava_onevision" else 6)
            assert result.logits.shape == (1, final, 64)
            sizes = [initial, records[0]["result_count"], final]
            assert [item[0] for item in attention] == [sizes[sum(point + int(mid) <= i for point in boundaries)] for i in range(4)]
            assert [item[0] for item in mlp] == [sizes[sum(point <= i for point in boundaries)] for i in range(4)]
            result.loss.backward()
            grads = {name: parameter.grad.detach().cpu().clone() for name, parameter in model.named_parameters() if parameter.grad is not None}
            assert grads and all(torch.isfinite(grad).all() for grad in grads.values())
            pixel_grad = raw["images" if backbone == "llava_onevision" else "pixel_values"].grad
            assert pixel_grad is not None and torch.isfinite(pixel_grad).all() and pixel_grad.abs().sum() > 0
            observed = result.logits.detach().cpu(), grads, pixel_grad.clone()
            if expected is None:
                expected = observed
            else:
                torch.testing.assert_close(observed, expected, atol=2e-6, rtol=2e-5)
            assert len(activation.runtime_metadata()["layout_delta_records"]) == 2
            assert activation.runtime_metadata()["application_positions"] == [{"boundary": "post_attention_pre_ffn" if mid else "inter_layer", "index": i} for i in boundaries]
        finally:
            for handle in handles:
                handle.remove()
            activation.uninstall()
        if checkpoint is not True:
            continue
        model.eval()
        activation = image_fixture.activation(model, _stage(backbone, boundaries, sources=sources, mid=mid, identity=identity, phase="generate"))
        activation.install()
        if backbone == "llava_onevision":
            args["inputs"] = args.pop("input_ids")
        outputs = []
        try:
            for _ in range(2):
                with torch.no_grad():
                    generated = model.generate(**args, min_new_tokens=3, max_new_tokens=3, do_sample=False,
                        use_cache=True, eos_token_id=None, pad_token_id=0, return_dict_in_generate=True)
                assert [layer.get_seq_length() for layer in generated.past_key_values.layers] == [final + 2] * 4
                outputs.append(generated.sequences)
            torch.testing.assert_close(outputs[0], outputs[1])
        finally:
            activation.uninstall()
    if identity:
        reference = copy.deepcopy(base)
        torch.manual_seed(273)
        args = image_fixture.inputs(torch, backbone)
        labels = torch.full_like(args["input_ids"], -100)
        labels[0, -1] = args["input_ids"][0, -1]
        output = reference(**args, labels=labels, use_cache=False)
        output.loss.backward()
        grads = {name: parameter.grad for name, parameter in reference.named_parameters() if parameter.grad is not None}
        torch.testing.assert_close((output.logits, grads, args["images" if backbone == "llava_onevision" else "pixel_values"].grad), expected, atol=2e-6, rtol=2e-5)


@pytest.mark.parametrize("backbone", ["qwen3_vl", "qwen25_vl", "llava_onevision"])
@pytest.mark.parametrize("boundaries,sources,mid", [((1, 2), (1, 2), False), ((2, 3), (1, 2), False),
    ((2, 4), (1, 4), False), ((1, 2), (1, 2), True), ((2, 3), (1, 2), True)])
def test_feather_two_native_boundaries_all_gradients_checkpoint_and_generate(backbone, boundaries, sources, mid):
    _run(backbone, boundaries, sources, mid)


@pytest.mark.parametrize("backbone", ["qwen3_vl", "qwen25_vl", "llava_onevision"])
@pytest.mark.parametrize("mid", [False, True])
def test_feather_fixed_sources_across_two_devices(backbone, mid):
    _run(backbone, (2, 3), (1, 2), mid, two_devices=True)


@pytest.mark.parametrize("backbone", ["qwen3_vl", "qwen25_vl", "llava_onevision"])
@pytest.mark.parametrize("mid", [False, True])
def test_feather_identity_matches_all_native_gradients(backbone, mid):
    _run(backbone, (1, 2), (1, 2), mid, identity=True)


@pytest.mark.parametrize("backbone", ["qwen3_vl", "qwen25_vl", "llava_onevision"])
@pytest.mark.parametrize("mid", [False, True])
def test_feather_native_fixed_sources_are_mapped_after_first_selection(backbone, mid, monkeypatch):
    import math
    torch = pytest.importorskip("torch")
    torch.manual_seed(280)
    reference = _model(backbone).eval()
    model = copy.deepcopy(reference)
    decoder = image_fixture.decoder(reference, backbone)
    native = {}
    def capture(name):
        def hook(module, args):
            native[name] = args[0].detach().clone()
        return hook
    # Both scoring layers complete before the first application at boundary 2.
    handles = [decoder.layers[i].register_forward_pre_hook(capture(f"source{i}")) for i in (1, 2)]
    application = decoder.layers[2].post_attention_layernorm if mid else decoder.layers[2]
    handles.append(application.register_forward_pre_hook(capture("application")))
    args = image_fixture.inputs(torch, backbone)
    try:
        with torch.no_grad():
            reference(**args, use_cache=False)
    finally:
        for handle in handles:
            handle.remove()
    image_indices = torch.arange(1, 730) if backbone == "llava_onevision" else torch.arange(2, 6)
    source_scores = []
    with torch.no_grad():
        for i in (1, 2):
            layer = decoder.layers[i - 1]
            hidden = layer.input_layernorm(native[f"source{i}"])
            attention = layer.self_attn
            dim = attention.head_dim
            heads = getattr(attention, "num_heads", attention.config.num_attention_heads)
            kv = attention.config.num_key_value_heads
            query = attention.q_proj(hidden).view(1, -1, heads, dim).transpose(1, 2)
            keys = attention.k_proj(hidden).view(1, -1, kv, dim).transpose(1, 2)
            if hasattr(attention, "q_norm"):
                query = attention.q_norm(query)
                keys = attention.k_norm(keys)
            keys = keys.repeat_interleave(heads // kv, dim=1)
            logits = query[:, :, -1:] @ keys[:, :, image_indices].transpose(-1, -2) * attention.scaling
            source_scores.append(logits.float().softmax(-1).mean(1)[0, 0])
    kernel = decoder_selection.feather_image_tokens
    selections = []
    def check_math(hidden=None, **kwargs):
        hidden = kwargs.pop("hidden_states", hidden)
        step = len(selections)
        expected_scores = source_scores[step]
        if step:
            expected_scores = expected_scores[selections[0]]
        else:
            torch.testing.assert_close(hidden, native["application"][0], atol=2e-6, rtol=2e-5)
        torch.testing.assert_close(kwargs["image_attention_scores"], expected_scores, atol=2e-6, rtol=2e-5)
        current = len(expected_scores)
        target = min(current, max(1, round(len(image_indices) * 0.5)))
        ranked = expected_scores.topk(target, sorted=False).indices
        if not step:
            width = math.ceil(math.sqrt(current))
            height = math.ceil(current / width)
            uniform = (torch.arange(0, height, 3)[:, None] * width + torch.arange(0, width, 3)[None]).flatten()
            selected = torch.cat((ranked, uniform[uniform < current])).unique(sorted=True)
        else:
            selected = ranked.sort().values
        result = kernel(hidden, **kwargs)
        torch.testing.assert_close(result.selected_visual_indices, selected)
        torch.testing.assert_close(result.hidden_states, hidden[result.kept_indices], atol=0, rtol=0)
        assert result.metadata["stage"] == step + 1
        assert result.metadata["application_position"] == {"boundary": "post_attention_pre_ffn" if mid else "inter_layer", "index": (2, 3)[step]}
        selections.append(selected)
        return result
    monkeypatch.setattr(decoder_selection, "feather_image_tokens", check_math)
    activation = image_fixture.activation(model, _stage(backbone, mid=mid))
    activation.install()
    try:
        with torch.no_grad():
            model(**args, use_cache=False)
        assert len(selections) == 2
    finally:
        activation.uninstall()


@pytest.mark.parametrize("backbone", ["qwen3_vl", "qwen25_vl", "llava_onevision"])
def test_feather_default_before_and_previous_output_are_exact(backbone):
    torch = pytest.importorskip("torch")
    torch.manual_seed(282)
    base = _model(backbone).train()
    expected = None
    for mode in ("legacy", "before", "after"):
        model = copy.deepcopy(base)
        stage = _stage(backbone, (1, 2))
        if mode == "legacy":
            stage.operation = {}
        elif mode == "after":
            stage.operation["targets"]["two_stage_no_rope_prune"].update(layers=[0, 1], layer_edge="after")
        activation = image_fixture.activation(model, stage)
        activation.install()
        if backbone == "llava_onevision":
            activation._bridge.prepare_training_forward()
        torch.manual_seed(283)
        args = image_fixture.inputs(torch, backbone)
        labels = torch.full_like(args["input_ids"], -100)
        labels[0, -1] = args["input_ids"][0, -1]
        try:
            result = model(**args, labels=labels, use_cache=False)
            result.loss.backward()
            observed = result.logits.detach(), model.lm_head.weight.grad
            if expected is None:
                expected = tuple(value.clone() for value in observed)
            else:
                torch.testing.assert_close(observed, expected, atol=0, rtol=0)
        finally:
            activation.uninstall()


@pytest.mark.parametrize("backbone", ["qwen3_vl", "qwen25_vl", "llava_onevision"])
@pytest.mark.parametrize("invalid", ["source", "mid_application"])
def test_feather_public_depth_errors_keep_backend_contract(backbone, invalid):
    from mllm_comp.core.errors import BackendIntegrationError
    model = _model(backbone)
    stage = _stage(backbone, (2, 4), sources=(1, 5) if invalid == "source" else (1, 2), mid=True)
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


@pytest.mark.parametrize("backbone", ["qwen3_vl", "qwen25_vl", "llava_onevision"])
@pytest.mark.parametrize("mid", [False, True])
def test_feather_released_budget_uses_original_stage_identity(backbone, mid, monkeypatch):
    torch = pytest.importorskip("torch")
    model = _model(backbone).eval()
    stage = _stage(backbone, mid=mid)
    stage.config.pop("target_final_retain_ratio")
    kernel = decoder_selection.feather_image_tokens
    calls = []
    def validate(*args, **kwargs):
        step = len(calls)
        assert kwargs["before_layer"] == (1, 2)[step]
        result = kernel(*args, **kwargs)
        count = len(kwargs["image_indices"])
        if step == 1:
            assert result.selected_visual_indices.numel() == max(1, int(count * 0.5 * 0.5))
            assert result.metadata["target_mode"] == "released_squared_remaining_ratio"
        else:
            assert result.metadata["attention_tokens"] == max(1, int(count * 0.5))
        calls.append(result.metadata["stage"])
        return result
    monkeypatch.setattr(decoder_selection, "feather_image_tokens", validate)
    activation = image_fixture.activation(model, stage)
    activation.install()
    try:
        with torch.no_grad():
            model(**image_fixture.inputs(torch, backbone), use_cache=False)
        assert calls == [1, 2]
    finally:
        activation.uninstall()
