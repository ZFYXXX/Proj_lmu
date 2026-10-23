"""FastV keeps its scoring rule while configuration moves the application."""

import copy

from mllm_comp.compressors.image import decoder_selection

import pytest

from mllm_comp.pipeline.stage import CompressionStageConfig
from tests.fixtures import native_image as image_fixture


def _stage(backbone, boundary, *, prune_layer=2, phase="forward", ratio=0.5, mid=False):
    return CompressionStageConfig.from_dict({
        "name": "fastv-position", "placement": "llm_level", "execution_phase": phase,
        "plugin": "fastv", "algorithm": "fastv", "variant": "fastv.image",
        "binding": f"fastv.image.{backbone}", "target_modalities": ["image"],
        "operation": {"type": "prune", "targets": {"decoder_prune_boundary": {
            "target_site": "llm_decoder_layer_post_attention_image_tokens" if mid else "llm_decoder_layer_output_image_tokens",
            "layers": [boundary], **({} if mid else {"layer_edge": "before"}),
        }}},
        "config": {"prune_layer": prune_layer, "prune_ratio": ratio},
    })


@pytest.mark.parametrize("backbone,boundary,source_rule,mid", [
    (backbone, boundary, source, mid)
    for backbone in ("qwen3_vl", "qwen25_vl", "llava_onevision")
    for mid, positions in (
        (False, ((0, 1), (1, 2), (2, 2), (3, 2), (4, 2), (4, 0))),
        (True, ((0, 1), (1, 2), (4, 2), (4, 0))),
    )
    for boundary, source in positions
    if mid or backbone != "llava_onevision" or boundary != 3
])
def test_native_fastv_application_lengths_gradient_and_checkpoint(backbone, boundary, source_rule, mid):
    torch = pytest.importorskip("torch")
    torch.manual_seed(73)
    base = image_fixture.model(backbone).train()
    depth = len(image_fixture.decoder(base, backbone).layers)
    boundary = min(boundary, depth - int(mid))
    before, after = (732, 367) if backbone == "llava_onevision" else (8, 6)
    expected = None
    for checkpoint in (None, False, True):
        model = copy.deepcopy(base)
        if checkpoint is not None:
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": checkpoint})
        activation = image_fixture.activation(model, _stage(backbone, boundary, prune_layer=source_rule, mid=mid))
        activation.install()
        if backbone == "llava_onevision":
            activation._bridge.prepare_training_forward()
        decoder = image_fixture.decoder(model, backbone)
        torch.manual_seed(74)
        args = image_fixture.inputs(torch, backbone)
        labels = torch.full_like(args["input_ids"], -100)
        labels[0, -1] = args["input_ids"][0, -1]
        counts = [[] for _ in decoder.layers]
        handles = [layer.mlp.register_forward_pre_hook(
            lambda module, args, i=i: counts[i].append(args[0].shape[1]))
            for i, layer in enumerate(decoder.layers)]
        attention_counts = [[] for _ in decoder.layers]
        handles += [layer.self_attn.o_proj.register_forward_pre_hook(
            lambda module, args, i=i: attention_counts[i].append(args[0].shape[1]))
            for i, layer in enumerate(decoder.layers)]
        try:
            output = model(**args, labels=labels, use_cache=False)
            assert output.logits.shape == (1, after, 64)
            assert [c[0] for c in counts] == [before if i < boundary else after for i in range(depth)]
            assert [c[0] for c in attention_counts] == [before if i < boundary + int(mid) else after for i in range(depth)]
            output.loss.backward()
            grad = args["images" if backbone == "llava_onevision" else "pixel_values"].grad
            assert grad is not None and torch.isfinite(grad).all() and grad.abs().sum() > 0
            assert all(layer.mlp.gate_proj.weight.grad is not None for layer in decoder.layers)
            if expected is None:
                expected = output.logits.detach(), grad.clone()
            else:
                torch.testing.assert_close(output.logits, expected[0])
                torch.testing.assert_close(grad, expected[1])
            metadata = activation.runtime_metadata()
            positions = metadata.get("application_positions", [metadata.get("application_position")])
            assert positions == [{"boundary": "post_attention_pre_ffn" if mid else "inter_layer", "index": boundary}]
        finally:
            for handle in handles:
                handle.remove()
            activation.uninstall()


@pytest.mark.parametrize("backbone", ["qwen3_vl", "qwen25_vl", "llava_onevision"])
@pytest.mark.parametrize("boundary,source_rule", [(0, 1), (1, 2), (4, 2), (4, 0)])
@pytest.mark.parametrize("mid", [False, True])
def test_native_fastv_prefill_decode_cache_and_repeated_generate(backbone, boundary, source_rule, mid):
    torch = pytest.importorskip("torch")
    torch.manual_seed(81)
    model = image_fixture.model(backbone).eval()
    depth = len(image_fixture.decoder(model, backbone).layers)
    boundary = min(boundary, depth - int(mid))
    activation = image_fixture.activation(model, _stage(backbone, boundary, prune_layer=source_rule, phase="generate", mid=mid))
    activation.install()
    if backbone == "qwen3_vl":
        from mllm_comp.backbones.qwen3_vl.provider_bridge import install_qwen3_vl_transformers_4573_bridge
        bridge = install_qwen3_vl_transformers_4573_bridge(model)
    else:
        bridge = activation._bridge
    args = image_fixture.inputs(torch, backbone)
    if backbone == "llava_onevision":
        args["inputs"] = args.pop("input_ids")
    try:
        outputs = []
        for _ in range(2):
            with torch.no_grad():
                result = model.generate(**args, use_cache=True, max_new_tokens=3, min_new_tokens=3,
                    do_sample=False, eos_token_id=None, pad_token_id=0, return_dict_in_generate=True)
            outputs.append(result.sequences)
            report = bridge.runtime_metadata()["latest_decoder_lifecycle"]
            if mid:
                assert report["boundaries"][0]["boundary"] == "post_attention_pre_ffn"
            assert report["selection_runs"] == 1
            assert report["decode_forward_calls"] == 2
            assert report["selection_repeated_during_decode"] is False
            assert report["boundaries"][0]["cache"]["touched_layers"] == list(range(boundary + int(mid)))
            assert [layer.get_seq_length() for layer in result.past_key_values.layers] == [369 if backbone == "llava_onevision" else 8] * depth
        torch.testing.assert_close(outputs[0], outputs[1])
    finally:
        activation.uninstall()


@pytest.mark.parametrize("backbone", ["qwen3_vl", "qwen25_vl", "llava_onevision"])
@pytest.mark.parametrize("boundary,source_rule", [(0, 1), (1, 2), (4, 2)])
@pytest.mark.parametrize("score_source", ["last_token", "mean_query"])
@pytest.mark.parametrize("mid", [False, True])
def test_fastv_uses_fixed_native_layer_input_and_original_attention_ranking(backbone, boundary, source_rule, score_source, monkeypatch, mid):
    import importlib
    torch = pytest.importorskip("torch")
    torch.manual_seed(92)
    reference = image_fixture.model(backbone).eval()
    model = copy.deepcopy(reference)
    decoder = image_fixture.decoder(model, backbone)
    boundary = min(boundary, len(decoder.layers) - int(mid))
    native_inputs = []
    handle = image_fixture.decoder(reference, backbone).layers[source_rule - 1].register_forward_pre_hook(
        lambda module, args, kwargs: native_inputs.append((args[0] if args else kwargs["hidden_states"]).detach().clone()), with_kwargs=True)
    residuals = []
    residual_handle = None
    if mid:
        residual_handle = image_fixture.decoder(reference, backbone).layers[boundary].post_attention_layernorm.register_forward_pre_hook(
            lambda module, args: residuals.append(args[0].detach().clone()))
    args = image_fixture.inputs(torch, backbone)
    try:
        with torch.no_grad():
            reference(**args, use_cache=False)
    finally:
        handle.remove()
        if residual_handle is not None:
            residual_handle.remove()
    provider = importlib.import_module(f"mllm_comp.backbones.{backbone}.provider_bridge")
    project = provider._decoder_query_keys
    compress = decoder_selection.fastv_image_tokens
    calls = []

    def capture(layer, hidden, **kwargs):
        assert layer is decoder.layers[source_rule - 1]
        torch.testing.assert_close(hidden, native_inputs[0], rtol=2e-5, atol=2e-6)
        calls.append("source")
        return project(layer, hidden, **kwargs)

    def check_math(hidden, **kwargs):
        if mid:
            torch.testing.assert_close(hidden, residuals[0][0], rtol=2e-5, atol=2e-6)
        result = compress(hidden, **kwargs)
        query, keys = kwargs["scorer_query"], kwargs["attention_keys"]
        logits = torch.einsum("qhd,thd->qht", query, keys) / query.shape[-1] ** 0.5
        mask = kwargs["attention_key_mask"]
        if mask is not None:
            visibility = mask[None, None, :] if mask.ndim == 1 else mask[:, None, :]
            logits = logits.masked_fill(~visibility.bool(), -torch.inf)
        scores = logits.float().softmax(-1).mean((0, 1)).index_select(0, kwargs["image_indices"])
        torch.testing.assert_close(result.image_scores, scores)
        count = round(len(scores) * 0.5)
        selected = torch.argsort(scores, descending=True, stable=True)[:count].sort().values
        torch.testing.assert_close(result.kept_image_indices, kwargs["image_indices"].index_select(0, selected))
        torch.testing.assert_close(result.hidden_states, hidden.index_select(0, result.kept_indices), rtol=0, atol=0)
        calls.append("apply")
        return result

    monkeypatch.setattr(provider, "_decoder_query_keys", capture)
    monkeypatch.setattr(decoder_selection, "fastv_image_tokens", check_math)
    stage = _stage(backbone, boundary, prune_layer=source_rule, mid=mid)
    stage.config["score_source"] = score_source
    activation = image_fixture.activation(model, stage)
    activation.install()
    try:
        with torch.no_grad():
            model(**args, use_cache=False)
        assert calls == ["source", "apply"]
    finally:
        activation.uninstall()


@pytest.mark.parametrize("backbone", ["qwen3_vl", "qwen25_vl", "llava_onevision"])
def test_fastv_legacy_and_equivalent_positions_keep_outputs_and_gradients(backbone, *, model_factory=image_fixture.model):
    torch = pytest.importorskip("torch")
    torch.manual_seed(105)
    base = model_factory(backbone).eval()
    expected = None
    for expression in ("legacy", "before", "after"):
        model = copy.deepcopy(base)
        stage = _stage(backbone, 2)
        if expression == "legacy":
            stage.operation.clear()
        elif expression == "after":
            next(iter(stage.operation["targets"].values())).update(layers=[1], layer_edge="after")
        activation = image_fixture.activation(model, stage)
        activation.install()
        torch.manual_seed(106)
        args = image_fixture.inputs(torch, backbone)
        labels = torch.full_like(args["input_ids"], -100)
        labels[0, -1] = args["input_ids"][0, -1]
        try:
            result = model(**args, labels=labels, use_cache=False)
            result.loss.backward()
            observed = result.logits.detach(), model.lm_head.weight.grad
            if expected is None:
                expected = tuple(item.clone() for item in observed)
            else:
                for actual, reference in zip(observed, expected):
                    torch.testing.assert_close(actual, reference, atol=0, rtol=0)
        finally:
            activation.uninstall()


@pytest.mark.parametrize("backbone", ["qwen3_vl", "qwen25_vl", "llava_onevision"])
def test_fastv_zero_pruning_matches_native_uncompressed_output_and_gradient(backbone):
    torch = pytest.importorskip("torch")
    torch.manual_seed(112)
    base = image_fixture.model(backbone).eval()
    expected = None
    for boundary in (None, 0, len(image_fixture.decoder(base, backbone).layers)):
        model = copy.deepcopy(base)
        activation = None if boundary is None else image_fixture.activation(model, _stage(backbone, boundary, prune_layer=1, ratio=0))
        if activation is not None:
            activation.install()
        torch.manual_seed(113)
        args = image_fixture.inputs(torch, backbone)
        labels = torch.full_like(args["input_ids"], -100)
        labels[0, -1] = args["input_ids"][0, -1]
        try:
            result = model(**args, labels=labels, use_cache=False)
            result.loss.backward()
            observed = result.logits.detach(), model.lm_head.weight.grad
            if expected is None:
                expected = tuple(item.clone() for item in observed)
            else:
                for actual, reference in zip(observed, expected):
                    torch.testing.assert_close(actual, reference, atol=2e-6, rtol=2e-5)
        finally:
            if activation is not None:
                activation.uninstall()


@pytest.mark.parametrize("backbone", ["qwen3_vl", "qwen25_vl", "llava_onevision"])
@pytest.mark.parametrize("mid", [False, True])
def test_fastv_source_at_two_device_layer_transition(backbone, mid):
    torch = pytest.importorskip("torch")
    if torch.cuda.device_count() < 2:
        pytest.skip("Explicit two-device source/application test.")
    from accelerate import dispatch_model

    for checkpoint in (None, False, True):
        torch.manual_seed(121)
        model = image_fixture.model(backbone).train()
        prefix = "model" if backbone == "llava_onevision" else "model.language_model"
        decoder = image_fixture.decoder(model, backbone)
        device_map = {name: "cuda:0" for name, _ in model.named_children() if name != "model"}
        if backbone != "llava_onevision":
            device_map.update({f"model.{name}": "cuda:0" for name, _ in model.model.named_children() if name != "language_model"})
        device_map.update({f"{prefix}.{name}": "cuda:0" for name, _ in decoder.named_children() if name != "layers"})
        device_map.update({f"{prefix}.layers.{i}": "cuda:0" if i < 2 else "cuda:1" for i in range(len(decoder.layers))})
        dispatch_model(model, device_map=device_map)
        if checkpoint is not None:
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": checkpoint})
        stage = _stage(backbone, 2, prune_layer=3, mid=mid)
        activation = image_fixture.activation(model, stage)
        activation.install()
        if backbone == "llava_onevision":
            activation._bridge.prepare_training_forward()
        raw = image_fixture.inputs(torch, backbone)
        args = {name: value.to("cuda:0") if torch.is_tensor(value) else value for name, value in raw.items()}
        labels = torch.full_like(args["input_ids"], -100)
        labels[0, -1] = args["input_ids"][0, -1]
        try:
            result = model(**args, labels=labels, use_cache=False)
            result.loss.backward()
            assert torch.isfinite(decoder.layers[2].mlp.gate_proj.weight.grad).all()
            assert result.logits.shape[1] == (367 if backbone == "llava_onevision" else 6)
        finally:
            activation.uninstall()
        model.eval()
        activation = image_fixture.activation(model, _stage(backbone, 2, prune_layer=3, phase="generate", mid=mid))
        activation.install()
        if backbone == "llava_onevision":
            args["inputs"] = args.pop("input_ids")
        try:
            with torch.no_grad():
                generated = model.generate(**args, min_new_tokens=2, max_new_tokens=2, do_sample=False,
                    use_cache=True, eos_token_id=None, pad_token_id=0, return_dict_in_generate=True)
            assert [layer.get_seq_length() for layer in generated.past_key_values.layers] == [368 if backbone == "llava_onevision" else 7] * len(decoder.layers)
        finally:
            activation.uninstall()


@pytest.mark.parametrize("backbone", ["qwen3_vl", "qwen25_vl", "llava_onevision"])
@pytest.mark.parametrize("checkpoint", [False, True])
def test_fastv_mid_replays_two_outstanding_graphs_after_uninstall(backbone, checkpoint, monkeypatch):
    torch = pytest.importorskip("torch")
    torch.manual_seed(132)
    base = image_fixture.model(backbone).train()
    compress = decoder_selection.fastv_image_tokens
    selections = []

    def counted(*args, **kwargs):
        result = compress(*args, **kwargs)
        selections.append(result.kept_indices.detach().clone())
        return result

    monkeypatch.setattr(decoder_selection, "fastv_image_tokens", counted)
    gradients = []
    for mode in (None, checkpoint):
        model = copy.deepcopy(base)
        if mode is not None:
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": mode})
        activation = image_fixture.activation(model, _stage(backbone, 1, mid=True))
        activation.install()
        if backbone == "llava_onevision":
            activation._bridge.prepare_training_forward()
        losses, inputs = [], []
        try:
            for seed in (133, 134):
                torch.manual_seed(seed)
                args = image_fixture.inputs(torch, backbone)
                labels = torch.full_like(args["input_ids"], -100)
                labels[0, -1] = args["input_ids"][0, -1]
                losses.append(model(**args, labels=labels, use_cache=False).loss)
                inputs.append(args["images" if backbone == "llava_onevision" else "pixel_values"])
        finally:
            activation.uninstall()
        selection_count = len(selections)
        sum(losses).backward()
        assert len(selections) == selection_count
        gradients.append([model.lm_head.weight.grad.clone(), *[item.grad.clone() for item in inputs]])
    for actual, expected in zip(gradients[1], gradients[0]):
        torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize("backbone", ["qwen3_vl", "qwen25_vl", "llava_onevision"])
def test_fastv_mid_failure_restores_native_modules_and_next_request(backbone, monkeypatch):
    torch = pytest.importorskip("torch")
    model = image_fixture.model(backbone).train()
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    compress = decoder_selection.fastv_image_tokens
    activation = image_fixture.activation(model, _stage(backbone, 1, mid=True))
    activation.install()
    if backbone == "llava_onevision":
        activation._bridge.prepare_training_forward()
    layer = image_fixture.decoder(model, backbone).layers[1]
    original_forward = layer.mlp.forward
    original_layer_forward = layer.forward

    def fail(*args, **kwargs):
        raise RuntimeError("selection failure")

    try:
        args = image_fixture.inputs(torch, backbone)
        labels = torch.full_like(args["input_ids"], -100)
        labels[0, -1] = args["input_ids"][0, -1]
        monkeypatch.setattr(decoder_selection, "fastv_image_tokens", fail)
        with pytest.raises(RuntimeError, match="selection failure"):
            model(**args, labels=labels, use_cache=False)
        assert layer.mlp.forward == original_forward
        assert layer.forward == original_layer_forward
        assert layer.gradient_checkpointing is True
        monkeypatch.setattr(decoder_selection, "fastv_image_tokens", compress)
        output = model(**args, labels=labels, use_cache=False)
        output.loss.backward()
        assert torch.isfinite(layer.mlp.gate_proj.weight.grad).all()
    finally:
        activation.uninstall()


@pytest.mark.parametrize("backbone", ["qwen3_vl", "qwen25_vl", "llava_onevision"])
@pytest.mark.parametrize("boundary", [0, 1])
def test_fastv_mid_identity_preserves_native_residual_hooks_and_all_gradients(backbone, boundary):
    from mllm_comp.analysis.bindings import DecoderLayerBinding
    from mllm_comp.analysis.common.residual import ResidualNodeCapture
    from mllm_comp.analysis.hooks import HookSession

    torch = pytest.importorskip("torch")
    torch.manual_seed(142)
    base = image_fixture.model(backbone).train()
    expected = None
    for mode in ("native", None, False, True):
        model = copy.deepcopy(base)
        if mode is not None and mode != "native":
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": mode})
        activation = None if mode == "native" else image_fixture.activation(
            model, _stage(backbone, boundary, prune_layer=1, ratio=0, mid=True))
        if activation is not None:
            activation.install()
            if backbone == "llava_onevision":
                activation._bridge.prepare_training_forward()
        torch.manual_seed(143)
        args = image_fixture.inputs(torch, backbone)
        labels = torch.full_like(args["input_ids"], -100)
        labels[0, -1] = args["input_ids"][0, -1]
        layers = image_fixture.decoder(model, backbone).layers
        capture = ResidualNodeCapture([DecoderLayerBinding(i, layer, layer.self_attn, layer.mlp)
            for i, layer in enumerate(layers)])
        try:
            with HookSession() as hooks:
                capture.install(hooks)
                output = model(**args, labels=labels, use_cache=False)
                _, evidence = capture.finish(atol=2e-6, rtol=2e-5)
                assert all(item["passed"] for item in evidence.values())
            # Analysis captures one forward, not checkpoint replay.
            output.loss.backward()
            observed = {name: None if value.grad is None else value.grad.detach().clone()
                for name, value in model.named_parameters()}
            observed["input"] = args["images" if backbone == "llava_onevision" else "pixel_values"].grad
            observed["logits"] = output.logits.detach()
            if expected is None:
                expected = observed
                if backbone == "qwen3_vl":
                    branches = [value for name, value in observed.items() if "deepstack_merger_list" in name]
                    assert branches and all(value is not None and value.abs().sum() > 0 for value in branches)
            else:
                assert observed.keys() == expected.keys()
                for name, reference in expected.items():
                    if reference is None:
                        assert observed[name] is None, name
                    else:
                        assert observed[name] is not None, name
                        torch.testing.assert_close(observed[name], reference, atol=2e-6, rtol=2e-5, msg=name)
        finally:
            if activation is not None:
                activation.uninstall()
