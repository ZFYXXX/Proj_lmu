"""Native guide scores stay independent of SGL's target application boundary."""

import copy

import pytest

from tests.fixtures import native_image as image_fixture
from tests.integration import test_image_fastv_positions_cpu as checks


def _stage(backbone, boundary, *, mid=False, keep_ratio=0.5):
    stage = checks._stage(backbone, boundary, phase="generate", mid=mid)
    stage.name = "sgl-position"
    stage.plugin = stage.algorithm = "sgl"
    stage.variant = "sgl.image"
    stage.binding = f"sgl.image.{backbone}"
    targets = stage.operation["targets"]
    targets["decoder_guide_prune_boundary"] = targets.pop("decoder_prune_boundary")
    stage.config = {"guide_model_path": "/fixtures/native-tiny-guide", "prune_layer_fraction": 0.5,
        "keep_ratio": keep_ratio}
    return stage


def _native_guide_scores(torch, backbone, model, args):
    from mllm_comp.compressors.image.sgl import aggregate_sgl_visual_attention
    guide = copy.deepcopy(model).eval()
    inputs = dict(args)
    if backbone == "llava_onevision":
        inputs["inputs"] = inputs.pop("input_ids")
    with torch.no_grad():
        output = guide.generate(**inputs, do_sample=False, use_cache=True, min_new_tokens=2,
            max_new_tokens=2, eos_token_id=None, pad_token_id=0, output_attentions=True,
            return_dict_in_generate=True)
    summary = aggregate_sgl_visual_attention(output.attentions, input_ids=args["input_ids"],
        image_token_id=-200 if backbone == "llava_onevision" else 10)
    assert summary.generated_query_tokens > 0 and summary.prompt_query_tokens > 0
    return summary.visual_scores


def _guide(activation, backbone):
    return activation._sgl_guide_value if backbone == "qwen25_vl" else activation.guide


@pytest.mark.parametrize("backbone", ["qwen3_vl", "qwen25_vl", "llava_onevision"])
@pytest.mark.parametrize("boundary,mid", [(0, False), (1, False), (4, False), (0, True), (1, True), (4, True)])
def test_sgl_native_guide_target_lengths_checkpoint_and_cache(backbone, boundary, mid, keep_ratio=0.5):
    torch = pytest.importorskip("torch")
    torch.manual_seed(181)
    base = image_fixture.model(backbone).train()
    depth = len(image_fixture.decoder(base, backbone).layers)
    boundary = min(boundary, depth - int(mid))
    torch.manual_seed(182)
    scores = _native_guide_scores(torch, backbone, base, image_fixture.inputs(torch, backbone))
    before, after = (732, 367) if backbone == "llava_onevision" else (8, 6)
    expected = None
    if keep_ratio == 1:
        after = before
        reference = copy.deepcopy(base)
        torch.manual_seed(182)
        args = image_fixture.inputs(torch, backbone)
        labels = torch.full_like(args["input_ids"], -100)
        labels[0, -1] = args["input_ids"][0, -1]
        output = reference(**args, labels=labels, use_cache=False)
        output.loss.backward()
        expected = {name: None if value.grad is None else value.grad.clone() for name, value in reference.named_parameters()}
        expected.update(logits=output.logits.detach(), input=args["images" if backbone == "llava_onevision" else "pixel_values"].grad)
    for checkpoint in (None, False, True):
        model = copy.deepcopy(base)
        if checkpoint is not None:
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": checkpoint})
        activation = image_fixture.activation(model, _stage(backbone, boundary, mid=mid, keep_ratio=keep_ratio))
        activation.install()
        if backbone == "llava_onevision":
            activation._bridge.prepare_training_forward()
        controller = _guide(activation, backbone)
        torch.manual_seed(182)
        args = image_fixture.inputs(torch, backbone)
        labels = torch.full_like(args["input_ids"], -100)
        labels[0, -1] = args["input_ids"][0, -1]
        decoder = image_fixture.decoder(model, backbone)
        lengths = {"attention": [[] for _ in decoder.layers], "ffn": [[] for _ in decoder.layers]}
        handles = []
        for i, layer in enumerate(decoder.layers):
            for name, module in (("attention", layer.self_attn.o_proj), ("ffn", layer.mlp)):
                handles.append(module.register_forward_pre_hook(
                    lambda module, args, name=name, i=i: lengths[name][i].append(args[0].shape[1])))
        try:
            controller.stage_scores(scores)
            output = model(**args, labels=labels, use_cache=False)
            assert output.logits.shape[1] == after
            assert [values[0] for values in lengths["ffn"]] == [before if i < boundary else after for i in range(depth)]
            assert [values[0] for values in lengths["attention"]] == [before if i < boundary + int(mid) else after for i in range(depth)]
            output.loss.backward()
            controller.finish_target(success=True)
            grad = args["images" if backbone == "llava_onevision" else "pixel_values"].grad
            assert grad is not None and torch.isfinite(grad).all() and grad.abs().sum() > 0
            observed = {name: None if value.grad is None else value.grad.clone() for name, value in model.named_parameters()}
            observed.update(logits=output.logits.detach(), input=grad.clone())
            if expected is None:
                expected = observed
            else:
                for name, value in expected.items():
                    if value is None:
                        assert observed[name] is None, name
                    else:
                        assert observed[name] is not None, name
                        torch.testing.assert_close(observed[name], value, atol=2e-6, rtol=2e-5, msg=name)
            assert activation.runtime_metadata()["prune_layer"] == boundary
            assert activation.runtime_metadata()["application_position"] == {
                "boundary": "post_attention_pre_ffn" if mid else "inter_layer", "index": boundary}
            if checkpoint is True:
                model.eval()
                generation_inputs = dict(args)
                if backbone == "llava_onevision":
                    generation_inputs["inputs"] = generation_inputs.pop("input_ids")
                generated = []
                for _ in range(2):
                    controller.stage_scores(scores)
                    with torch.no_grad():
                        result = model.generate(**generation_inputs, use_cache=True, do_sample=False,
                            min_new_tokens=3, max_new_tokens=3, eos_token_id=None, pad_token_id=0,
                            return_dict_in_generate=True)
                    controller.finish_target(success=True)
                    assert [layer.get_seq_length() for layer in result.past_key_values.layers] == [after + 2] * depth
                    generated.append(result.sequences)
                torch.testing.assert_close(generated[0], generated[1])
        finally:
            for handle in handles:
                handle.remove()
            activation.uninstall()
            assert controller.runtime_metadata()["guide_attached"] is False
            assert controller.runtime_metadata()["guide_scores_pending"] is False


@pytest.mark.parametrize("backbone", ["qwen3_vl", "qwen25_vl", "llava_onevision"])
def test_sgl_identity_preserves_native_output_and_all_gradients(backbone):
    test_sgl_native_guide_target_lengths_checkpoint_and_cache(backbone, 0, True, keep_ratio=1)


@pytest.mark.parametrize("backbone", ["qwen3_vl", "qwen25_vl", "llava_onevision"])
@pytest.mark.parametrize("mid", [False, True])
def test_sgl_two_device_target_with_native_cpu_guide(backbone, mid):
    torch = pytest.importorskip("torch")
    if torch.cuda.device_count() < 2:
        pytest.skip("Explicit native two-device target and independent guide test.")
    from accelerate import dispatch_model
    for checkpoint in (None, False, True):
        torch.manual_seed(191)
        model = image_fixture.model(backbone).train()
        raw = image_fixture.inputs(torch, backbone)
        scores = _native_guide_scores(torch, backbone, model, raw)
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
        activation = image_fixture.activation(model, _stage(backbone, 2, mid=mid))
        activation.install()
        if backbone == "llava_onevision":
            activation._bridge.prepare_training_forward()
        controller = _guide(activation, backbone)
        args = {name: value.to("cuda:0") if torch.is_tensor(value) else value for name, value in raw.items()}
        labels = torch.full_like(args["input_ids"], -100)
        labels[0, -1] = args["input_ids"][0, -1]
        try:
            controller.stage_scores(scores)
            output = model(**args, labels=labels, use_cache=False)
            output.loss.backward()
            controller.finish_target(success=True)
            assert torch.isfinite(decoder.layers[2].mlp.gate_proj.weight.grad).all()
            model.eval()
            if backbone == "llava_onevision":
                args["inputs"] = args.pop("input_ids")
            controller.stage_scores(scores)
            with torch.no_grad():
                output = model.generate(**args, use_cache=True, do_sample=False, min_new_tokens=2,
                    max_new_tokens=2, eos_token_id=None, pad_token_id=0, return_dict_in_generate=True)
            controller.finish_target(success=True)
            assert [layer.get_seq_length() for layer in output.past_key_values.layers] == [368 if backbone == "llava_onevision" else 7] * len(decoder.layers)
        finally:
            activation.uninstall()


@pytest.mark.parametrize("backbone", ["qwen3_vl", "qwen25_vl", "llava_onevision"])
def test_sgl_default_fraction_and_equivalent_boundaries(backbone):
    torch = pytest.importorskip("torch")
    torch.manual_seed(201)
    base = image_fixture.model(backbone).eval()
    boundary = len(image_fixture.decoder(base, backbone).layers) // 2
    torch.manual_seed(202)
    scores = _native_guide_scores(torch, backbone, base, image_fixture.inputs(torch, backbone))
    expected = None
    for expression in ("legacy", "before", "after"):
        model = copy.deepcopy(base)
        stage = _stage(backbone, boundary)
        if expression == "legacy":
            stage.operation.clear()
        elif expression == "after":
            stage.operation["targets"]["decoder_guide_prune_boundary"].update(layers=[boundary - 1], layer_edge="after")
        activation = image_fixture.activation(model, stage)
        activation.install()
        torch.manual_seed(202)
        args = image_fixture.inputs(torch, backbone)
        labels = torch.full_like(args["input_ids"], -100)
        labels[0, -1] = args["input_ids"][0, -1]
        try:
            _guide(activation, backbone).stage_scores(scores)
            output = model(**args, labels=labels, use_cache=False)
            output.loss.backward()
            observed = output.logits.detach(), model.lm_head.weight.grad
            if expected is None:
                expected = tuple(value.clone() for value in observed)
            else:
                for value, reference in zip(observed, expected):
                    torch.testing.assert_close(value, reference, atol=0, rtol=0)
        finally:
            activation.uninstall()


@pytest.mark.parametrize("backbone", ["qwen3_vl", "qwen25_vl", "llava_onevision"])
@pytest.mark.parametrize("boundary,mid", [(0, False), (4, False), (0, True), (4, True)])
def test_sgl_preserves_guide_ranking_and_selects_actual_native_boundary(backbone, boundary, mid, monkeypatch):
    from mllm_comp.compressors.image import decoder_selection as adapter
    torch = pytest.importorskip("torch")
    torch.manual_seed(211)
    reference = image_fixture.model(backbone).eval()
    model = copy.deepcopy(reference)
    args = image_fixture.inputs(torch, backbone)
    scores = _native_guide_scores(torch, backbone, reference, args)
    decoder = image_fixture.decoder(reference, backbone)
    boundary = min(boundary, len(decoder.layers) - int(mid))
    application = decoder.layers[boundary].post_attention_layernorm if mid else (
        decoder.layers[boundary] if boundary < len(decoder.layers) else decoder.norm)
    captured = []
    handle = application.register_forward_pre_hook(lambda module, args, kwargs:
        captured.append((args[0] if args else kwargs["hidden_states"]).detach().clone()), with_kwargs=True)
    try:
        with torch.no_grad():
            reference(**args, use_cache=False)
    finally:
        handle.remove()
    kernel = adapter.sgl_image_tokens
    calls = []

    def check_kernel(hidden, **kwargs):
        torch.testing.assert_close(hidden, captured[0][0], atol=2e-6, rtol=2e-5)
        torch.testing.assert_close(kwargs["image_scores"], scores, atol=0, rtol=0)
        result = kernel(hidden, **kwargs)
        count = kwargs["spec"].keep_count(len(scores))
        indices = torch.argsort(scores, descending=True, stable=True)[:count].sort().values
        torch.testing.assert_close(result.kept_image_indices, kwargs["image_indices"][indices])
        torch.testing.assert_close(result.hidden_states, hidden[result.kept_indices], atol=0, rtol=0)
        calls.append(True)
        return result

    monkeypatch.setattr(adapter, "sgl_image_tokens", check_kernel)
    activation = image_fixture.activation(model, _stage(backbone, boundary, mid=mid))
    activation.install()
    try:
        _guide(activation, backbone).stage_scores(scores)
        generation_inputs = dict(args)
        if backbone == "llava_onevision":
            generation_inputs["inputs"] = generation_inputs.pop("input_ids")
        with torch.no_grad():
            model.generate(**generation_inputs, use_cache=True, do_sample=False, min_new_tokens=2,
                max_new_tokens=2, eos_token_id=None, pad_token_id=0)
        _guide(activation, backbone).finish_target(success=True)
        assert calls == [True]
    finally:
        activation.uninstall()
