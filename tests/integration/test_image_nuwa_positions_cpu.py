"""Nüwa preserves visual aggregation while moving its second-stage selection."""

import copy
import importlib
import math

import pytest

from mllm_comp.pipeline.stage import CompressionStageConfig
from tests.fixtures import native_image as image_fixture
from tests.integration import test_image_fastv_positions_cpu as checks


def _model(backbone):
    model = image_fixture.llava_model(num_decoder_layers=4) if backbone == "llava_onevision" else image_fixture.qwen_model(backbone)
    if backbone == "qwen25_vl":
        model.model.visual.fullatt_block_indexes = [1, 2]
    return model


def _stage(backbone, boundary, *, prune_layer=2, phase="forward", mid=False, mode="attention", ratio=0.5):
    return CompressionStageConfig.from_dict({
        "name": "nuwa-position", "placement": "projector_level", "execution_phase": phase,
        "plugin": "nuwa", "algorithm": "nuwa", "variant": "nuwa.image",
        "binding": f"nuwa.image.{backbone}", "target_modalities": ["image"],
        "operation": {"type": "prune", "targets": {"stage2_last_prompt_attention_selection": {
            "target_site": "llm_decoder_layer_post_attention_image_tokens" if mid else "llm_decoder_layer_output_image_tokens",
            "layers": [boundary], **({} if mid else {"layer_edge": "before"})}}},
        "config": {"stage1_total_tokens": 12 if backbone == "llava_onevision" else 3,
            "stage2_after_layer": prune_layer, "stage2_retain_ratio": ratio,
            "semantic_hidden_start": 0, "semantic_hidden_end": 3,
            "random_stage2": mode == "random",
            "stage2_score": "paper_text_cosine" if mode == "cosine" else "released_last_text_attention"},
    })


def _run(backbone, boundary, source, mid, mode, *, two_devices=False):
    torch = pytest.importorskip("torch")
    if two_devices and torch.cuda.device_count() < 2:
        pytest.skip("Explicit two-device Nüwa source/application test.")
    torch.manual_seed(301)
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
            device_map.update({f"{prefix}.layers.{i}": "cuda:0" if i < 2 else "cuda:1" for i in range(4)})
            dispatch_model(model, device_map=device_map)
        if checkpoint is not None:
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": checkpoint})
        activation = image_fixture.activation(model, _stage(backbone, boundary, prune_layer=source, mid=mid, mode=mode))
        activation.install()
        if backbone == "llava_onevision":
            activation._bridge.prepare_training_forward()
        torch.manual_seed(302)
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
            before, final = (15, 9) if backbone == "llava_onevision" else (7, 6)
            assert result.logits.shape == (1, final, 64)
            assert [item[0] for item in attention] == [before if i < boundary + int(mid) else final for i in range(4)]
            assert [item[0] for item in mlp] == [before if i < boundary else final for i in range(4)]
            records = copy.deepcopy(activation.runtime_metadata()["layout_delta_records"])
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
            metadata = activation.runtime_metadata()
            assert metadata["layout_delta_records"] == records
            assert metadata["application_position"] == {"boundary": "post_attention_pre_ffn" if mid else "inter_layer", "index": boundary}
            assert records[-1]["result_layout"]["target_site"] == metadata["decoder_target_site"]
            if backbone == "qwen25_vl":
                assert metadata["target_sites"] == ["post_vision_merger_pre_llm_scatter", metadata["decoder_target_site"]]
        finally:
            for handle in handles:
                handle.remove()
            activation.uninstall()
        if checkpoint is not True:
            continue
        model.eval()
        activation = image_fixture.activation(model, _stage(backbone, boundary, prune_layer=source, mid=mid, mode=mode, phase="generate"))
        activation.install()
        if backbone == "qwen3_vl":
            from mllm_comp.backbones.qwen3_vl.provider_bridge import install_qwen3_vl_transformers_4573_bridge
            bridge = install_qwen3_vl_transformers_4573_bridge(model)
        else:
            bridge = activation._bridge
        if backbone == "llava_onevision":
            args["inputs"] = args.pop("input_ids")
        outputs = []
        try:
            for _ in range(2):
                with torch.no_grad():
                    generated = model.generate(**args, min_new_tokens=3, max_new_tokens=3, do_sample=False,
                        use_cache=True, eos_token_id=None, pad_token_id=0, return_dict_in_generate=True)
                assert [layer.get_seq_length() for layer in generated.past_key_values.layers] == [final + 2] * 4
                report = bridge.runtime_metadata()["latest_decoder_lifecycle"]
                assert report["selection_runs"] == 1
                assert report["decode_forward_calls"] == 2
                assert report["selection_repeated_during_decode"] is False
                if backbone == "qwen25_vl":
                    assert bridge.runtime_metadata()["latest_decode_position"]["matches_cache_position_plus_rope_delta"] is True
                    assert report["boundaries"][0]["result_layout"]["position_policy"] == "native_sparse_mrope_rpme"
                assert report["boundaries"][0]["cache"]["touched_layers"] == list(range(boundary + int(mid)))
                outputs.append(generated.sequences)
            torch.testing.assert_close(outputs[0], outputs[1])
        finally:
            activation.uninstall()


@pytest.mark.parametrize("backbone", ["qwen3_vl", "qwen25_vl", "llava_onevision"])
@pytest.mark.parametrize("boundary,source,mid,mode", [
    (0, 1, False, "attention"), (0, 1, True, "attention"), (4, 4, False, "attention"),
    (3, 2, True, "attention"), (0, 16, False, "random"), (0, 16, True, "cosine")])
def test_nuwa_native_composite_positions_gradients_checkpoint_and_generate(backbone, boundary, source, mid, mode):
    _run(backbone, boundary, source, mid, mode)


@pytest.mark.parametrize("backbone", ["qwen3_vl", "qwen25_vl", "llava_onevision"])
@pytest.mark.parametrize("mid", [False, True])
def test_nuwa_fixed_source_across_two_devices(backbone, mid):
    _run(backbone, 3, 2, mid, "attention", two_devices=True)


@pytest.mark.parametrize("backbone", ["qwen3_vl", "qwen25_vl", "llava_onevision"])
def test_nuwa_legacy_and_equivalent_boundaries_match(backbone, monkeypatch):
    monkeypatch.setattr(checks, "_stage", _stage)
    checks.test_fastv_legacy_and_equivalent_positions_keep_outputs_and_gradients(backbone, model_factory=_model)


@pytest.mark.parametrize("backbone", ["qwen3_vl", "qwen25_vl", "llava_onevision"])
@pytest.mark.parametrize("invalid", ["source", "application"])
def test_nuwa_public_factory_rejects_unavailable_native_depth(backbone, invalid):
    from mllm_comp.core.errors import BackendIntegrationError
    stage = _stage(backbone, 4, prune_layer=5) if invalid == "source" else _stage(backbone, 4, mid=True)
    with pytest.raises(BackendIntegrationError):
        image_fixture.activation(_model(backbone), stage).install()


@pytest.mark.parametrize("backbone", ["qwen3_vl", "qwen25_vl", "llava_onevision"])
@pytest.mark.parametrize("mid", [False, True])
def test_nuwa_stage2_identity_preserves_stage1_outputs_and_all_gradients(backbone, mid):
    torch = pytest.importorskip("torch")
    torch.manual_seed(315)
    base = _model(backbone).train()
    expected = None
    for checkpoint in ("legacy", None, False, True):
        model = copy.deepcopy(base)
        if checkpoint in (False, True) and checkpoint is not None:
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": checkpoint})
        stage = _stage(backbone, 0 if mid else 4, prune_layer=1, mid=mid, ratio=1.0)
        if checkpoint == "legacy":
            stage.operation.clear()
        activation = image_fixture.activation(model, stage)
        activation.install()
        if backbone == "llava_onevision":
            activation._bridge.prepare_training_forward()
        torch.manual_seed(316)
        args = image_fixture.inputs(torch, backbone)
        labels = torch.full_like(args["input_ids"], -100)
        labels[0, -1] = args["input_ids"][0, -1]
        try:
            result = model(**args, labels=labels, use_cache=False)
            result.loss.backward()
            grads = {name: value.grad.detach().clone() for name, value in model.named_parameters() if value.grad is not None}
            observed = result.logits.detach(), grads, args["images" if backbone == "llava_onevision" else "pixel_values"].grad
            if expected is None:
                expected = observed
            else:
                torch.testing.assert_close(observed, expected, atol=2e-6, rtol=2e-5)
        finally:
            activation.uninstall()


@pytest.mark.parametrize("backbone", ["qwen3_vl", "qwen25_vl", "llava_onevision"])
@pytest.mark.parametrize("mid", [False, True])
@pytest.mark.parametrize("mode", ["attention", "cosine", "random"])
def test_nuwa_fixed_source_and_actual_application_math(backbone, mid, mode, monkeypatch):
    torch = pytest.importorskip("torch")
    torch.manual_seed(310)
    reference = _model(backbone).eval()
    model = copy.deepcopy(reference)
    decoder = image_fixture.decoder(reference, backbone)
    native = {}
    def capture(name):
        def hook(module, args, kwargs):
            native[name] = (args[0] if args else kwargs["hidden_states"]).detach().clone()
        return hook
    application = decoder.layers[3].post_attention_layernorm if mid else decoder.layers[3]
    handles = [module.register_forward_pre_hook(capture(name), with_kwargs=True)
        for name, module in (("source", decoder.layers[1]), ("application", application))]
    # Keep the same real Stage 1; ratio one defers any second-stage removal.
    activation = image_fixture.activation(reference, _stage(backbone, 4, ratio=1.0))
    activation.install()
    args = image_fixture.inputs(torch, backbone)
    try:
        with torch.no_grad():
            reference(**args, use_cache=False)
    finally:
        for handle in handles:
            handle.remove()
        activation.uninstall()
    provider = importlib.import_module(f"mllm_comp.backbones.{backbone}.provider_bridge")
    from mllm_comp.compressors.image import decoder_selection as adapter

    project, kernel = provider._decoder_query_keys, adapter.nuwa_stage2_image_tokens
    scores, calls = [], []
    def check_source(layer, hidden, **kwargs):
        assert mode == "attention"
        assert layer is image_fixture.decoder(model, backbone).layers[1]
        torch.testing.assert_close(hidden, native["source"], atol=2e-6, rtol=2e-5)
        query, image_keys, keys = project(layer, hidden, **kwargs)
        logits = torch.einsum("qhd,thd->qht", query, keys) / query.shape[-1] ** 0.5
        scores.append(logits.float().softmax(-1).mean((0, 1))[kwargs["image_indices"]])
        return query, image_keys, keys
    def check_kernel(hidden, **kwargs):
        torch.testing.assert_close(hidden, native["application"][0], atol=2e-6, rtol=2e-5)
        indices = kwargs["image_indices"]
        if mode == "attention":
            torch.testing.assert_close(kwargs["image_attention_scores"], scores[0])
            expected_scores = scores[0]
        elif mode == "cosine":
            mask = torch.ones(len(hidden), dtype=torch.bool, device=hidden.device)
            mask[indices] = False
            expected_scores = torch.nn.functional.cosine_similarity(hidden[indices].float(), hidden[mask].float().mean(0).unsqueeze(0), dim=-1)
        else:
            generator = torch.Generator(device=hidden.device).manual_seed(kwargs["spec"].random_seed)
            expected_scores = torch.rand(len(indices), generator=generator, device=hidden.device)
        result = kernel(hidden, **kwargs)
        selected = expected_scores.argsort(descending=True, stable=True)[:math.ceil(len(indices) * 0.5)].sort().values
        torch.testing.assert_close(result.selected_visual_indices, selected)
        torch.testing.assert_close(result.hidden_states, hidden[result.kept_indices], atol=0, rtol=0)
        calls.append(result)
        return result
    monkeypatch.setattr(provider, "_decoder_query_keys", check_source)
    monkeypatch.setattr(adapter, "nuwa_stage2_image_tokens", check_kernel)
    activation = image_fixture.activation(model, _stage(backbone, 3, mid=mid, mode=mode))
    activation.install()
    try:
        with torch.no_grad():
            model(**args, use_cache=False)
        assert len(calls) == 1
        assert len(scores) == int(mode == "attention")
    finally:
        activation.uninstall()
