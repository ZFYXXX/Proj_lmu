"""AIM explicit groups and VisPruner fixed crop sources at LLaVA projector input."""

import copy

import pytest

from tests.fixtures import native_image as image_fixture
from tests.integration.test_image_projector_positions_cpu import _stage
from tests.integration.test_image_tome_positions_cpu import _stage as _tome_stage
from tests.integration.test_projector_llava_onevision_positions_cpu import _model, _packed


def _inputs(torch):
    ids = torch.tensor([[1, -200, 2, 3]])
    labels = torch.full_like(ids, -100)
    labels[0, -1] = ids[0, -1]
    return dict(input_ids=ids, labels=labels, attention_mask=torch.ones_like(ids),
        images=[torch.randn(2, 3, 54, 54, requires_grad=True)], image_sizes=[(54, 54)], modalities=["image"], use_cache=False)


def test_aim_pre_anyres_explicit_groups_nonlinear_projection_and_gradients(monkeypatch):
    torch = pytest.importorskip("torch")
    from mllm_comp.backbones.llava_onevision import image_compression as adapter
    from mllm_comp.backbones.llava_onevision import provider_bridge as provider
    from mllm_comp.core.layout import OrderingPolicy
    torch.manual_seed(386)
    model = _model().train()
    stage = _stage("llava_onevision", "aim")
    stage.config["target_count"] = 32
    activation = image_fixture.activation(model, stage)
    activation.install()
    activation._bridge.prepare_training_forward()
    projected, decoder_inputs, captured = [], [], {}
    projector = model.model.mm_projector
    handles = [projector.register_forward_pre_hook(lambda module, args: projected.append(args[0])),
        model.model.register_forward_pre_hook(lambda module, args, kwargs: decoder_inputs.append(kwargs["inputs_embeds"]), with_kwargs=True)]
    kernel = adapter.aim_image_tokens

    def compress(values, **kwargs):
        torch.testing.assert_close(values, _packed(projected[0], projected[0].new_zeros(16)))
        result = kernel(values, **kwargs)
        captured.update(result=result, kwargs=kwargs)
        return result

    monkeypatch.setattr(adapter, "aim_image_tokens", compress)
    monkeypatch.setattr(provider, "_packed_qk_metrics", lambda *args: pytest.fail("AIM pre computes cosine from its actual values."))
    args = _inputs(torch)
    try:
        output = model(**args)
        result = captured["result"]
        delta = result.layout_delta
        assert delta.ordering_policy is OrderingPolicy.EXPLICIT and delta.merge_groups
        newlines = [delta.old_to_new[i] for i in captured["kwargs"]["protected_indices"]]
        assert len(newlines) == 27
        content_indices = [i for i in range(delta.result_count) if i not in newlines]
        assert [x.shape for x in projected] == [torch.Size([2, 729, 16]), torch.Size([5, 16])]
        oracle = kernel(_packed(projected[0], projected[0].new_zeros(16)), **captured["kwargs"])
        content = projector(oracle.tokens[content_indices])
        expected = content.new_zeros((delta.result_count, 24)).index_copy(0, torch.tensor(content_indices), content)
        expected = expected.index_copy(0, torch.tensor(newlines), model.model.image_newline.expand(27, -1))
        actual = decoder_inputs[0][0, 1:-2]
        torch.testing.assert_close(actual, expected)
        for i, size in enumerate(result.token_sizes.tolist()):
            assert size == sum(target == i for target in delta.old_to_new.values())
        parameters = (projected[0], *tuple(projector.parameters()), model.model.image_newline)
        upstream = torch.randn_like(actual)
        torch.testing.assert_close(torch.autograd.grad(actual, parameters, upstream, retain_graph=True),
            torch.autograd.grad(expected, parameters, upstream, retain_graph=True))
        output.loss.backward()
        assert args["images"][0].grad is not None and args["images"][0].grad.abs().sum() > 0
        record = activation.runtime_metadata()["layout_delta_records"][0]
        assert record["site"] == "pre_vision_projector"
        assert tuple(record["result_layout"]["representative_indices"]) == tuple(result.representative_indices.tolist())
        assert tuple(tuple(pair) for pair in record["result_layout"]["crop_boundaries"]) == ((0, 32),)
    finally:
        for handle in handles:
            handle.remove()
        activation.uninstall()


def test_vispruner_anyres_crop_budget_fixed_siglip_source_and_pre_values(monkeypatch):
    torch = pytest.importorskip("torch")
    from mllm_comp.backbones.llava_onevision import image_compression as adapter
    from mllm_comp.backbones.llava_onevision import provider_bridge as provider
    torch.manual_seed(387)
    base = _model().train()
    expected = None
    kernel = adapter.vispruner_image_tokens
    monkeypatch.setattr(provider, "_packed_qk_metrics", lambda *args: pytest.fail("VisPruner must retain its SigLIP source."))
    for position in ("projector_output", "projector_input"):
        model = copy.deepcopy(base)
        stage = _stage("llava_onevision", "vispruner", position)
        stage.config = {"tokens_per_crop": 4}
        activation = image_fixture.activation(model, stage)
        activation.install()
        activation._bridge.prepare_training_forward()
        tower_outputs, projected, selections = [], [], []
        handles = [model.model.vision_tower.vision_tower.register_forward_hook(lambda module, args, out: tower_outputs.append(out)),
            model.model.mm_projector.register_forward_pre_hook(lambda module, args: projected.append(args[0]))]

        def select(values, features, attention, **kwargs):
            crop = len(selections)
            source = tower_outputs[0]
            torch.testing.assert_close(features, source.hidden_states[-1][crop])
            torch.testing.assert_close(attention, source.attentions[-1][crop])
            assert values.shape == (729, 16 if position == "projector_input" else 24)
            if position == "projector_input":
                torch.testing.assert_close(values, features)
            assert kwargs["target_count"] == 4
            result = kernel(values, features, attention, **kwargs)
            selections.append(result.indices.tolist())
            return result

        monkeypatch.setattr(adapter, "vispruner_image_tokens", select)
        torch.manual_seed(388)
        args = _inputs(torch)
        try:
            output = model(**args)
            assert len(tower_outputs) == 1 and len(selections) == 2
            assert [len(x) for x in selections] == [4, 4]
            assert output.logits.shape[1] == 38  # two crop budgets + 27 structural rows + text
            if position == "projector_input":
                assert projected[-1].shape == (8, 16)
            else:
                assert len(projected) == 1
            output.loss.backward()
            pixels = args["images"][0]
            assert pixels.grad is not None and torch.isfinite(pixels.grad).all() and pixels.grad.abs().sum() > 0
            assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.model.mm_projector.parameters())
            if expected is None:
                expected = selections, output.logits.detach(), pixels.grad.clone()
            else:
                assert selections == expected[0]
                torch.testing.assert_close(output.logits, expected[1], atol=2e-6, rtol=2e-5)
                torch.testing.assert_close(pixels.grad, expected[2], atol=2e-6, rtol=2e-5)
        finally:
            for handle in handles:
                handle.remove()
            activation.uninstall()


@pytest.mark.parametrize("algorithm", ["aim", "vispruner"])
def test_projector_input_preserves_following_tome_output(algorithm):
    torch = pytest.importorskip("torch")
    torch.manual_seed(389)
    model = image_fixture.llava_model().train()
    tome_stage = _tome_stage("llava_onevision", "full", r=0)
    tome_stage.config["output_r"] = 1
    tome = image_fixture.activation(model, tome_stage)
    stage = _stage("llava_onevision", algorithm)
    stage.config = {"target_count": 8}
    activation = image_fixture.activation(model, stage)
    tome.install()
    activation.install()
    activation._bridge.prepare_training_forward()
    args = image_fixture.inputs(torch, "llava_onevision")
    labels = torch.full_like(args["input_ids"], -100)
    labels[0, -1] = args["input_ids"][0, -1]
    try:
        result = model(**args, labels=labels, use_cache=False)
        records = [r for r in tome.runtime_metadata()["layout_delta_records"] if r["site"] == "post_anyres_pack_pre_llm_splice_tome"]
        assert len(records) == 1 and records[0]["source_count"] == 8 and records[0]["result_count"] == 7
        assert result.logits.shape[1] == 10
        result.loss.backward()
        assert args["images"].grad is not None and args["images"].grad.abs().sum() > 0
    finally:
        activation.uninstall()
        tome.uninstall()
