"""Native LLaVA projector inputs, packed metrics, and AnyRes structural values."""

import copy

import pytest

from mllm_comp.pipeline.stage import CompressionStageConfig
from tests.fixtures import native_image as image_fixture
from tests.integration.test_image_tome_positions_cpu import _stage as _tome_stage


def _stage(algorithm, *, before=True):
    config = {"budget": 32} if algorithm == "visionzip" else {"target_count": 32}
    return CompressionStageConfig.from_dict({
        "name": algorithm + "-projector", "placement": "projector_level", "execution_phase": "forward",
        "plugin": algorithm, "algorithm": algorithm, "variant": algorithm + ".image",
        "binding": algorithm + ".image.llava_onevision", "target_modalities": ["image"],
        "operation": {"target_site": "projector_input" if before else "projector_output"}, "config": config,
    })


def _model():
    torch = pytest.importorskip("torch")
    from llava.model.language_model.llava_qwen import LlavaQwenForCausalLM
    source = image_fixture.llava_model()
    config = copy.deepcopy(source.config)
    config.hidden_size, config.intermediate_size = 24, 32
    config.num_attention_heads = 3
    model = LlavaQwenForCausalLM(config)
    model.model.vision_tower = source.model.vision_tower
    model.model.mm_projector = torch.nn.Sequential(torch.nn.Linear(16, 24), torch.nn.GELU(), torch.nn.Linear(24, 24))
    model.model.image_newline = torch.nn.Parameter(torch.randn(24))
    model.config.mm_patch_merge_type = "spatial_unpad"
    model.config.image_aspect_ratio = "anyres"
    model.config.image_grid_pinpoints = [[54, 54]]
    return model


def _packed(values, newline):
    torch = pytest.importorskip("torch")
    tile = values[1].reshape(27, 27, values.shape[-1])
    return torch.cat((values[0], torch.cat((tile, newline.expand(27, 1, -1)), dim=1).flatten(0, 1)))


@pytest.mark.parametrize("algorithm", ["visionzip", "prumerge", "gap"])
def test_pre_projector_anyres_original_metrics_values_and_gradients(algorithm, monkeypatch):
    torch = pytest.importorskip("torch")
    from mllm_comp.backbones.llava_onevision import image_compression as adapter
    from mllm_comp.backbones.llava_onevision import provider_bridge as provider
    torch.manual_seed(360)
    model = _model().train()
    projector = model.model.mm_projector
    activation = image_fixture.activation(model, _stage(algorithm))
    activation.install()
    activation._bridge.prepare_training_forward()
    projection_inputs, decoder_inputs, observed = [], [], {}
    handles = [projector.register_forward_pre_hook(lambda module, args: projection_inputs.append(args[0])),
        model.model.register_forward_pre_hook(lambda module, args, kwargs: decoder_inputs.append(kwargs["inputs_embeds"]), with_kwargs=True)]
    metric = provider._packed_qk_metrics
    kernel_name = {"visionzip": "visionzip_image", "prumerge": "prumerge_image_tokens", "gap": "gap_image_tokens"}[algorithm]
    kernel = getattr(adapter, kernel_name)

    def scores(layer, packed):
        expected = _packed(projector(projection_inputs[0].detach()), model.model.image_newline)
        torch.testing.assert_close(packed, expected)
        # Preserve complete projected-key normalization, including row newlines.
        normalized = layer.input_layernorm(packed.unsqueeze(0))
        attention = layer.self_attn
        q = attention.q_proj(normalized).view(packed.shape[0], -1, attention.head_dim).permute(1, 0, 2)
        k = attention.k_proj(normalized).view(packed.shape[0], -1, attention.head_dim).permute(1, 0, 2)
        k = k.repeat_interleave(attention.num_key_value_groups, dim=0)
        expected_attention = (q.float() @ k.float().transpose(-1, -2) * attention.scaling).softmax(-1)
        result = metric(layer, packed)
        torch.testing.assert_close(result[0], expected_attention)
        observed["scoring_input"] = packed
        return result

    def compress(values, *args, **kwargs):
        raw = projection_inputs[0]
        expected = _packed(raw, raw.new_zeros((raw.shape[-1],)))
        torch.testing.assert_close(values, expected)
        assert len(kwargs["protected_indices"]) == 27
        result = kernel(values, *args, **kwargs)
        observed.update(raw=raw, args=args, kwargs=kwargs, result=result)
        return result

    monkeypatch.setattr(provider, "_packed_qk_metrics", scores)
    monkeypatch.setattr(adapter, kernel_name, compress)
    ids = torch.tensor([[1, -200, 2, 3]])
    pixels = torch.randn(2, 3, 54, 54, requires_grad=True)
    labels = torch.full_like(ids, -100)
    labels[0, -1] = ids[0, -1]
    try:
        output = model(input_ids=ids, attention_mask=torch.ones_like(ids), labels=labels,
            images=[pixels], image_sizes=[(54, 54)], modalities=["image"], use_cache=False)
        result = observed["result"]
        delta = result.layout_delta
        newlines = [delta.old_to_new[i] for i in observed["kwargs"]["protected_indices"]]
        kept = [i for i in range(delta.result_count) if i not in newlines]
        assert projection_inputs[-1].shape == (len(kept), 16)
        assert len(kept) < 1458
        actual = decoder_inputs[0][0, 1:-2]
        oracle = kernel(_packed(observed["raw"], observed["raw"].new_zeros((16,))), *observed["args"], **observed["kwargs"])
        content = projector(oracle.tokens[kept])
        indices = torch.tensor(kept)
        expected = content.new_zeros((delta.result_count, 24)).index_copy(0, indices, content)
        expected = expected.index_copy(0, torch.tensor(newlines), model.model.image_newline.expand(27, -1))
        torch.testing.assert_close(actual, expected)
        torch.testing.assert_close(actual[newlines], model.model.image_newline.expand(27, -1), atol=0, rtol=0)
        upstream = torch.randn_like(actual)
        parameters = (observed["raw"], *tuple(projector.parameters()), model.model.image_newline)
        actual_grads = torch.autograd.grad(actual, parameters, upstream, retain_graph=True)
        expected_grads = torch.autograd.grad(expected, parameters, upstream, retain_graph=True)
        torch.testing.assert_close(actual_grads, expected_grads)
        output.loss.backward()
        assert pixels.grad is not None and torch.isfinite(pixels.grad).all() and pixels.grad.abs().sum() > 0
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in projector.parameters())
        metadata = activation.runtime_metadata()
        assert metadata["projector_target_sites"] == ["pre_vision_projector"]
        assert metadata["layout_delta_records"][0]["site"] == "pre_vision_projector"
        assert metadata["dense_projector_metric_source"] is True
    finally:
        for handle in handles:
            handle.remove()
        activation.uninstall()


def test_pre_divprune_newline_metric_rejects_and_post_remains_supported(monkeypatch):
    torch = pytest.importorskip("torch")
    from mllm_comp.core.errors import BackendIntegrationError
    from mllm_comp.backbones.llava_onevision import provider_bridge as provider
    monkeypatch.setattr(provider, "_packed_qk_metrics", lambda *args: pytest.fail("DivPrune must not construct attention metrics."))
    model = _model().eval()
    ids = torch.tensor([[1, -200, 2, 3]])
    args = dict(input_ids=ids, attention_mask=torch.ones_like(ids), images=[torch.randn(2, 3, 54, 54)],
        image_sizes=[(54, 54)], modalities=["image"], use_cache=False)
    for before in (True, False):
        activation = image_fixture.activation(model, _stage("divprune", before=before))
        activation.install()
        try:
            with torch.no_grad():
                if before:
                    with pytest.raises(BackendIntegrationError, match="protected-centre"):
                        model(**args)
                else:
                    assert model(**args).logits.shape[1] == 35
                    metadata = activation._bridge.runtime_metadata()
                    assert metadata["packed_metric_sites"] == []
                    record = metadata["latest_layout_deltas"][0]
                    assert record["operation"] == "drop"
                    assert record["source_count"] == 1485
                    assert record["result_count"] == 32
                    assert record["protected_indices"] == [729 + row * 28 + 27 for row in range(27)]
        finally:
            activation.uninstall()


@pytest.mark.parametrize("algorithm", ["visionzip", "gap"])
def test_pre_visionzip_keeps_following_tome_output_step(algorithm):
    torch = pytest.importorskip("torch")
    torch.manual_seed(361)
    model = image_fixture.llava_model().train()
    tome = image_fixture.activation(model, _tome_stage("llava_onevision", "full"))
    visionzip = image_fixture.activation(model, _stage(algorithm))
    tome.install()
    visionzip.install()
    visionzip._bridge.prepare_training_forward()
    args = image_fixture.inputs(torch, "llava_onevision")
    labels = torch.full_like(args["input_ids"], -100)
    labels[0, -1] = args["input_ids"][0, -1]
    try:
        output = model(**args, labels=labels, use_cache=False)
        packed = [r for r in tome.runtime_metadata()["layout_delta_records"] if r["site"] == "post_anyres_pack_pre_llm_splice_tome"]
        assert len(packed) == 1 and packed[0]["source_count"] == 32 and packed[0]["result_count"] == 31
        assert output.logits.shape[1] == 34
        output.loss.backward()
        assert args["images"].grad is not None and args["images"].grad.abs().sum() > 0
    finally:
        visionzip.uninstall()
        tome.uninstall()


@pytest.mark.parametrize("layout", ["pool", "resize"])
def test_pre_projector_non_discrete_anyres_rejects_without_changing_post(layout):
    torch = pytest.importorskip("torch")
    from mllm_comp.core.errors import BackendIntegrationError
    model = _model().eval()
    if layout == "pool":
        model.config.mm_patch_merge_type = "spatial_maxpool2x2"
        crops, image_size = 2, (54, 54)
    else:
        model.config.image_aspect_ratio = "anyres_max_1"
        model.config.image_grid_pinpoints = [[108, 108]]
        crops, image_size = 5, (108, 108)
    ids = torch.tensor([[1, -200, 2, 3]])
    args = dict(input_ids=ids, attention_mask=torch.ones_like(ids), images=[torch.randn(crops, 3, 54, 54)],
        image_sizes=[image_size], modalities=["image"], use_cache=False)
    for before in (True, False):
        activation = image_fixture.activation(model, _stage("visionzip", before=before))
        activation.install()
        try:
            with torch.no_grad():
                if before:
                    with pytest.raises(BackendIntegrationError, match="[Pp]re-projector"):
                        model(**args)
                else:
                    assert model(**args).logits.shape[1] == 35
        finally:
            activation.uninstall()
