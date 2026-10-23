"""SAINT keeps projected packed key scoring while pruning pre-projector values."""

import copy

import pytest

from mllm_comp.pipeline.stage import CompressionStageConfig
from tests.fixtures import native_image as image_fixture
from tests.integration.test_projector_llava_onevision_positions_cpu import _model, _packed


def test_saint_pre_anyres_projected_key_source_newline_and_value_gradients(monkeypatch):
    torch = pytest.importorskip("torch")
    from mllm_comp.backbones.llava_onevision import image_compression as adapter
    from mllm_comp.backbones.llava_onevision import provider_bridge as provider
    torch.manual_seed(396)
    model = _model().train()
    projector = model.model.mm_projector
    reference_projector = copy.deepcopy(projector)
    stage = CompressionStageConfig.from_dict({
        "name": "saint-anyres-input", "placement": "projector_level", "execution_phase": "forward",
        "plugin": "saint", "algorithm": "saint", "variant": "saint.image.packed_feature_prune",
        "binding": "saint.image.llava_onevision", "target_modalities": ["image"],
        "operation": {"target_site": "projector_input"}, "config": {"keep_ratio": 0.05, "group_size": 2},
    })
    activation = image_fixture.activation(model, stage)
    activation.install()
    activation._bridge.prepare_training_forward()
    projection_inputs, decoder_inputs, observed = [], [], {}
    handles = [projector.register_forward_pre_hook(lambda module, args: projection_inputs.append(args[0])),
        model.model.register_forward_pre_hook(lambda module, args, kwargs: decoder_inputs.append(kwargs["inputs_embeds"]), with_kwargs=True)]
    metric, kernel = provider._packed_qk_metrics, adapter.saint_image_tokens

    def capture(layer, tokens):
        assert tokens.shape == (1485, 24)
        with torch.no_grad():
            expected = _packed(reference_projector(projection_inputs[0]), model.model.image_newline)
        torch.testing.assert_close(tokens, expected)
        normalized = layer.input_layernorm(tokens.unsqueeze(0))
        attention = layer.self_attn
        q = attention.q_proj(normalized).view(tokens.shape[0], -1, attention.head_dim).permute(1, 0, 2)
        k = attention.k_proj(normalized).view(tokens.shape[0], -1, attention.head_dim).permute(1, 0, 2)
        k = k.repeat_interleave(attention.num_key_value_groups, dim=0)
        result = metric(layer, tokens)
        torch.testing.assert_close(result[1], k)
        torch.testing.assert_close(result[0], (q.float() @ k.float().transpose(-1, -2) * attention.scaling).softmax(-1))
        observed["keys"] = k.permute(1, 0, 2)
        return result

    def compress(values, keys, **kwargs):
        torch.testing.assert_close(values, _packed(projection_inputs[0], projection_inputs[0].new_zeros(16)))
        torch.testing.assert_close(keys, observed["keys"])
        assert kwargs["spec"].keep_ratio == 0.05 and kwargs["spec"].group_size == 2
        assert len(kwargs["protected_indices"]) == 27
        result = kernel(values, keys, **kwargs)
        observed.update(result=result, kwargs=kwargs)
        return result

    monkeypatch.setattr(provider, "_packed_qk_metrics", capture)
    monkeypatch.setattr(adapter, "saint_image_tokens", compress)
    ids = torch.tensor([[1, -200, 2, 3]])
    labels = torch.full_like(ids, -100)
    labels[0, -1] = ids[0, -1]
    pixels = torch.randn(2, 3, 54, 54, requires_grad=True)
    try:
        output = model(input_ids=ids, labels=labels, attention_mask=torch.ones_like(ids),
            images=[pixels], image_sizes=[(54, 54)], modalities=["image"], use_cache=False)
        result = observed["result"]
        delta = result.layout_delta
        newline_targets = [delta.old_to_new[i] for i in observed["kwargs"]["protected_indices"]]
        content_indices = [i for i in range(delta.result_count) if i not in newline_targets]
        assert [tuple(x.shape) for x in projection_inputs] == [(2, 729, 16), (len(content_indices), 16)]
        raw = _packed(projection_inputs[0], projection_inputs[0].new_zeros(16))
        oracle = kernel(raw, observed["keys"], **observed["kwargs"])
        torch.testing.assert_close(oracle.selected_indices, result.selected_indices, atol=0, rtol=0)
        content = projector(oracle.tokens[content_indices])
        expected = content.new_zeros((delta.result_count, 24)).index_copy(0, torch.tensor(content_indices), content)
        expected = expected.index_copy(0, torch.tensor(newline_targets), model.model.image_newline.expand(27, -1))
        actual = decoder_inputs[0][0, 1:-2]
        torch.testing.assert_close(actual, expected)
        torch.testing.assert_close(actual[newline_targets], model.model.image_newline.expand(27, -1), atol=0, rtol=0)
        parameters = (projection_inputs[0], *tuple(projector.parameters()), model.model.image_newline)
        upstream = torch.randn_like(actual)
        torch.testing.assert_close(torch.autograd.grad(actual, parameters, upstream, retain_graph=True),
            torch.autograd.grad(expected, parameters, upstream, retain_graph=True))
        output.loss.backward()
        assert pixels.grad is not None and torch.isfinite(pixels.grad).all() and pixels.grad.abs().sum() > 0
        metadata = activation.runtime_metadata()
        assert metadata["projector_target_sites"] == ["pre_vision_projector"]
        assert metadata["dense_projector_metric_source"] is True
        record = metadata["layout_delta_records"][0]
        assert record["site"] == record["result_layout"]["compression_site"] == "pre_vision_projector"
    finally:
        for handle in handles:
            handle.remove()
        activation.uninstall()
