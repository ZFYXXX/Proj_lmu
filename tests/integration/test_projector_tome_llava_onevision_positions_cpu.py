"""ToMe's independent packed merge keeps projected keys and chooses value space."""

import copy

import pytest

from tests.fixtures import native_image as image_fixture
from tests.integration.test_image_tome_positions_cpu import _stage as _encoder_stage
from tests.integration.test_projector_aim_vispruner_llava_onevision_cpu import _inputs
from tests.integration.test_projector_llava_onevision_positions_cpu import _model, _packed


OUTPUT_SITE = "post_anyres_pack_pre_llm_splice_tome"


def test_two_packed_pre_callbacks_reject_without_misreporting_the_second_position():
    torch = pytest.importorskip("torch")
    from mllm_comp.core.errors import BackendIntegrationError
    from tests.integration.test_projector_llava_onevision_positions_cpu import _stage as projector_stage

    model = _model().eval()
    tome = image_fixture.activation(model, _stage("projector_input"))
    visionzip = image_fixture.activation(model, projector_stage("visionzip", before=True))
    tome.install()
    visionzip.install()
    args = image_fixture.inputs(torch, "llava_onevision")
    try:
        with pytest.raises(BackendIntegrationError, match="one packed compression callback"):
            model(**args, use_cache=False)
    finally:
        visionzip.uninstall()
    try:
        with torch.no_grad():
            output = model(**args, use_cache=False)
        assert output.logits.shape[1] > 0
        assert tome.runtime_metadata()["permanent_output_merge_site"] == "pre_vision_projector"
    finally:
        tome.uninstall()


def test_tome_then_visionzip_composes_native_anyres_layout():
    torch = pytest.importorskip("torch")
    from tests.integration.test_projector_llava_onevision_positions_cpu import _stage as projector_stage

    model = _model().eval()
    tome = image_fixture.activation(model, _stage(None))
    visionzip = image_fixture.activation(model, projector_stage("visionzip", before=False))
    tome.install()
    visionzip.install()
    try:
        with torch.no_grad():
            output = model(**_inputs(torch))
        packed = [record for record in tome.runtime_metadata()["layout_delta_records"] if record["site"] == OUTPUT_SITE]
        assert len(packed) == 1
        assert (packed[0]["source_count"], packed[0]["result_count"]) == (1485, 973)
        record = visionzip.runtime_metadata()["layout_delta_records"][0]
        assert (record["source_count"], record["result_count"]) == (973, 32)
        assert len(record["protected_indices"]) == 27
        assert output.logits.shape[1] == 35
    finally:
        visionzip.uninstall()
        tome.uninstall()


def _stage(position="projector_input", *, encoder_position="mid"):
    stage = _encoder_stage("llava_onevision", encoder_position, r=3)
    stage.config["output_r"] = 512
    if position is not None:
        stage.operation["targets"]["permanent_output_merge"] = {"target_site": position}
    return stage


def _groups_from_keys(torch, keys, protected, r):
    # Independent original matching equation, followed by causal source order.
    metric = keys.float().mean(1)
    metric = metric / metric.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    n = metric.shape[0]
    a = [i for i in range(0, n, 2) if i not in protected]
    b = [i for i in range(1, n, 2) if i not in protected]
    best, destinations = (metric[a] @ metric[b].T).max(-1)
    chosen = best.argsort(descending=True, stable=True)[:r].tolist()
    edges = {a[i]: b[destinations[i].item()] for i in chosen}
    groups = {i: [i] for i in range(n) if i not in edges}
    for source, destination in edges.items():
        groups[destination].append(source)
    return tuple(sorted((tuple(sorted(g)) for g in groups.values()), key=lambda g: g[0]))


@pytest.mark.parametrize("encoder_position,checkpoint", [("mid", False), ("full", True)])
def test_tome_anyres_independent_projected_keys_raw_merge_and_gradient(encoder_position, checkpoint, monkeypatch):
    torch = pytest.importorskip("torch")
    from mllm_comp.backbones.llava_onevision import image_compression as adapter
    from mllm_comp.backbones.llava_onevision import provider_bridge as provider
    from mllm_comp.core.layout import OrderingPolicy

    torch.manual_seed(461)
    base = _model().train()
    base.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": checkpoint})
    original_metric, original_kernel = provider._packed_qk_metrics, adapter.tome_merge_image
    reference_groups = None
    reference_values = None
    for before in (False, True):
        model = copy.deepcopy(base)
        projector = model.model.mm_projector
        reference_projector = copy.deepcopy(projector)
        activation = image_fixture.activation(model, _stage("projector_input" if before else "projector_output", encoder_position=encoder_position))
        activation.install()
        activation._bridge.prepare_training_forward()
        projection_inputs, decoder_inputs, observed = [], [], {}
        encoder = model.get_vision_tower().vision_tower.vision_model.encoder
        mlp_counts = [[] for _ in encoder.layers]
        handles = [projector.register_forward_pre_hook(lambda module, args: projection_inputs.append(args[0])),
            model.model.register_forward_pre_hook(lambda module, args, kwargs: decoder_inputs.append(kwargs["inputs_embeds"]), with_kwargs=True)]
        handles += [layer.mlp.register_forward_pre_hook(lambda module, args, i=i: mlp_counts[i].append(args[0].shape[1]))
            for i, layer in enumerate(encoder.layers)]

        def metric(layer, tokens):
            assert tokens.shape == (1485, 24)
            with torch.no_grad():
                torch.testing.assert_close(tokens, _packed(reference_projector(projection_inputs[0]), model.model.image_newline))
                normalized = layer.input_layernorm(tokens.unsqueeze(0))
                attention = layer.self_attn
                q = attention.q_proj(normalized).view(1485, -1, attention.head_dim).permute(1, 0, 2)
                k = attention.k_proj(normalized).view(1485, -1, attention.head_dim).permute(1, 0, 2)
                k = k.repeat_interleave(attention.num_key_value_groups, dim=0)
                result = original_metric(layer, tokens)
                torch.testing.assert_close(result[0], (q.float() @ k.float().transpose(-1, -2) * attention.scaling).softmax(-1))
                torch.testing.assert_close(result[1], k)
            observed["keys"] = k.permute(1, 0, 2)
            return result

        def kernel(values, keys, **kwargs):
            if values.ndim == 3:  # The separate Encoder step retains its original math.
                return original_kernel(values, keys, **kwargs)
            assert kwargs["r"] == 512
            assert values.shape == (1485, 16 if before else 24)
            torch.testing.assert_close(keys, observed["keys"])
            if before:
                torch.testing.assert_close(values, _packed(projection_inputs[0], values.new_zeros(16)))
            result = original_kernel(values, keys, **kwargs)
            observed.update(result=result, protected=tuple(kwargs["protected_indices"]))
            return result

        monkeypatch.setattr(provider, "_packed_qk_metrics", metric)
        monkeypatch.setattr(adapter, "tome_merge_image", kernel)
        torch.manual_seed(462)
        args = _inputs(torch)
        try:
            output = model(**args)
            assert mlp_counts == [[729], [726 if encoder_position == "mid" else 729], [726]]
            raw = projection_inputs[0]
            assert raw.shape == (2, 729, 16)  # The original dense restore remains in place.
            groups = _groups_from_keys(torch, observed["keys"], observed["protected"], 512)
            result = adapter.tome_preserve_source_order(observed["result"])
            delta = result.layout_delta
            assert result.sequence_source_groups == groups
            assert delta.ordering_policy is OrderingPolicy.PRESERVE_SOURCE
            assert delta.result_count == 973
            assert len(observed["protected"]) == 27
            for index in observed["protected"]:
                assert groups[delta.old_to_new[index]] == (index,)
            torch.testing.assert_close(result.token_sizes.flatten(), result.token_sizes.new_tensor([len(g) for g in groups]))
            newlines = [delta.old_to_new[i] for i in observed["protected"]]
            content_indices = [i for i in range(delta.result_count) if i not in newlines]
            expected_projection_shapes = [(2, 729, 16), (946, 16)] if before else [(2, 729, 16)]
            assert [tuple(v.shape) for v in projection_inputs] == expected_projection_shapes
            raw_packed = _packed(raw, raw.new_zeros(16))
            projected_packed = _packed(projector(raw), model.model.image_newline)
            raw_means = torch.stack([raw_packed[list(g)].mean(0) for g in groups])
            projected_means = torch.stack([projected_packed[list(g)].mean(0) for g in groups])
            if before:
                content = projector(raw_means[content_indices])
                expected = content.new_zeros((973, 24)).index_copy(0, torch.tensor(content_indices), content)
                expected = expected.index_copy(0, torch.tensor(newlines), model.model.image_newline.expand(27, -1))
                torch.testing.assert_close(result.hidden_states, raw_means)
                assert not torch.allclose(expected, projected_means, atol=1e-7, rtol=1e-6)
            else:
                expected = projected_means
                torch.testing.assert_close(result.hidden_states, projected_means)
            actual = decoder_inputs[0][0, 1:-2]
            torch.testing.assert_close(actual, expected)
            torch.testing.assert_close(actual[newlines], model.model.image_newline.expand(27, -1), atol=0, rtol=0)
            parameters = (raw, *tuple(projector.parameters()), model.model.image_newline)
            upstream = torch.randn_like(actual)
            torch.testing.assert_close(torch.autograd.grad(actual, parameters, upstream, retain_graph=True),
                torch.autograd.grad(expected, parameters, upstream, retain_graph=True))
            output.loss.backward()
            assert args["images"][0].grad is not None and args["images"][0].grad.abs().sum() > 0
            assert torch.isfinite(args["images"][0].grad).all()
            assert all(p.grad is not None and torch.isfinite(p.grad).all() for layer in encoder.layers for p in layer.mlp.parameters())
            metadata = activation.runtime_metadata()
            site = "pre_vision_projector" if before else OUTPUT_SITE
            assert metadata["permanent_output_merge_site"] == metadata["target_sites"][-1] == site
            assert metadata["dense_projector_metric_source"] is before
            assert metadata["dense_projector_packing_reference"] is before
            assert metadata["layout_delta_records"][-1]["site"] == site
            assert metadata["last_layout"]["compression_site"] == site
            assert metadata["real_checkpoint_verified"] is False
            if reference_groups is None:
                reference_groups, reference_values = groups, actual.detach().clone()
            else:
                assert groups == reference_groups  # Matching remains on the fixed projected source.
                assert not torch.allclose(actual, reference_values, atol=1e-7, rtol=1e-6)
        finally:
            for handle in handles:
                handle.remove()
            activation.uninstall()


def test_tome_output_default_alias_exact_and_repeated_forward_cleanup():
    torch = pytest.importorskip("torch")
    from mllm_comp.core.errors import BackendIntegrationError

    torch.manual_seed(463)
    model = _model().train()
    original_encode = model.encode_images
    expected = None
    for position in (None, OUTPUT_SITE, "projector_output", "projector_input"):
        activation = image_fixture.activation(model, _stage(position))
        activation.install()
        activation._bridge.prepare_training_forward()
        previous = None
        bridge = activation._bridge
        try:
            for _ in range(2):
                torch.manual_seed(464)
                args = _inputs(torch)
                model.zero_grad(set_to_none=True)
                output = model(**args)
                output.loss.backward()
                result = (output.logits.detach().clone(), args["images"][0].grad.clone(),
                    model.model.mm_projector[0].weight.grad.clone(), model.model.image_newline.grad.clone())
                if previous is not None:
                    torch.testing.assert_close(result, previous, atol=0, rtol=0)
                previous = result
                assert not activation._invocations
                assert activation._bridge._vision_source_capture.get() is None
            if position is None:
                expected = previous
            elif position != "projector_input":
                torch.testing.assert_close(previous, expected, atol=0, rtol=0)
            with pytest.raises(BackendIntegrationError, match="use_cache=False"):
                model(**{**args, "use_cache": True})
            with pytest.raises(BackendIntegrationError, match="does not support generate"):
                model.generate(inputs=args["input_ids"], images=args["images"], image_sizes=args["image_sizes"], max_new_tokens=1)
        finally:
            activation.uninstall()
        assert model.encode_images == original_encode
        assert not bridge._callbacks
