"""Native LLaVA VQToken uses the original pooled lattice on either value space."""

import copy

import pytest

from tests.integration.test_projector_llava_onevision_positions_cpu import _model


def _video_model(pool_mode="bilinear"):
    model = _model()
    model.config.mm_spatial_pool_stride = 2
    model.config.mm_spatial_pool_mode = pool_mode
    model.config.mm_newline_position = "grid"
    model.config.add_faster_video = False
    return model


def _inputs(torch):
    ids = torch.tensor([[1, -200, 2, 3]])
    labels = torch.full_like(ids, -100)
    labels[0, -1] = 3
    return dict(input_ids=ids, labels=labels, attention_mask=torch.ones_like(ids),
        images=[torch.randn(2, 3, 54, 54, requires_grad=True)], image_sizes=[(54, 54)],
        modalities=["video"], use_cache=False)


def _pool(torch, values, mode):
    frames = values.view(2, 27, 27, -1).permute(0, 3, 1, 2)
    if mode == "bilinear":
        pooled = torch.nn.functional.interpolate(frames, size=(14, 14), mode="bilinear")
    else:
        pooled = torch.nn.functional.max_pool2d(frames, 2)
    return pooled.permute(0, 2, 3, 1).flatten(0, 2)


@pytest.mark.parametrize("pool_mode", ["bilinear", "max"])
def test_native_pool_values_vq_normalization_newline_gradients_and_cached_generate(pool_mode):
    torch = pytest.importorskip("torch")
    from mllm_comp.backbones.llava_onevision.video_compression import (
        LLAVA_ONEVISION_VIDEO_SPLICE_SITE, install_llava_onevision_video_bridge,
    )
    from mllm_comp.compressors.video import VQTokenVideoKernel
    torch.manual_seed(430)
    base = _video_model(pool_mode).train()
    baseline = None
    for position in (None, "after", "before"):
        model = copy.deepcopy(base)
        projector = model.model.mm_projector
        bridge = install_llava_onevision_video_bridge(model)
        requests, results, projected_inputs, decoder_inputs, encoded = [], [], [], [], []
        kernel = VQTokenVideoKernel()

        def compress(request):
            requests.append(request)
            result = kernel(request.payload, config={})
            results.append(result)
            return result

        handle = bridge.register_callback(LLAVA_ONEVISION_VIDEO_SPLICE_SITE, compress,
            **({} if position is None else {"projector_position": position}))
        hooks = [projector.register_forward_pre_hook(lambda module, args: projected_inputs.append(args[0])),
            model.model.register_forward_pre_hook(lambda module, args, kwargs: decoder_inputs.append(kwargs.get("inputs_embeds")), with_kwargs=True),
            model.model.vision_tower.register_forward_hook(lambda module, args, output: encoded.append(output))]
        original_hook_count = len(projector._forward_pre_hooks)
        torch.manual_seed(431)
        args = _inputs(torch)
        try:
            output = model(**args)
            assert len(requests) == len(results) == len(encoded) == 1
            assert len(projector._forward_pre_hooks) == original_hook_count
            assert [tuple(value.shape) for value in projected_inputs] == (
                [(2, 729, 16), (32, 16)] if position == "before" else [(2, 729, 16)])
            request, result = requests[0], results[0]
            raw = projected_inputs[0]
            assert request.packing.grid_height == (14 if pool_mode == "bilinear" else 13)
            side = request.packing.grid_height
            assert request.payload.token_count == 2 * side * side
            assert result.result_count == 32 and len(result.source_groups[0]) == 32
            values = _pool(torch, raw, pool_mode) if position == "before" else _pool(torch, projector(raw), pool_mode)
            torch.testing.assert_close(request.payload.features, values)
            normalized = values / torch.linalg.vector_norm(values, dim=-1, keepdim=True)
            compressed = torch.stack([normalized[list(group)].mean(0) for group in result.source_groups[0]])
            torch.testing.assert_close(result.features, compressed)
            content = projector(compressed) if position == "before" else compressed
            entries = [(request.packing.patch_positions[source], content[index])
                for index, source in enumerate(result.representative_indices[0])]
            entries += [(source, model.model.image_newline) for source in request.packing.structural_positions]
            entries.sort(key=lambda item: item[0])
            expected = torch.stack([value for _, value in entries])
            actual = decoder_inputs[0][0, 1:-2]
            torch.testing.assert_close(actual, expected)
            assert output.logits.shape[1] == 3 + 32 + 2 * side
            newline_targets = [i for i, (source, _) in enumerate(entries) if source in request.packing.structural_positions]
            torch.testing.assert_close(actual[newline_targets], model.model.image_newline.expand(2 * side, -1), atol=0, rtol=0)
            parameters = (raw, *tuple(projector.parameters()), model.model.image_newline)
            upstream = torch.randn_like(actual)
            torch.testing.assert_close(torch.autograd.grad(actual, parameters, upstream, retain_graph=True),
                torch.autograd.grad(expected, parameters, upstream, retain_graph=True), atol=2e-6, rtol=2e-5)
            output.loss.backward()
            gradient = args["images"][0].grad
            assert gradient is not None and torch.isfinite(gradient).all() and gradient.abs().sum() > 0
            if baseline is None:
                baseline = (output.logits.detach().clone(), gradient.clone(), result.source_groups, result.representative_indices)
            elif position == "after":
                torch.testing.assert_close(output.logits, baseline[0], atol=0, rtol=0)
                torch.testing.assert_close(gradient, baseline[1], atol=0, rtol=0)
                assert (result.source_groups, result.representative_indices) == baseline[2:]
            metadata = bridge.runtime_metadata()
            expected_site = "video_pre_vision_projector_pooled" if position == "before" else LLAVA_ONEVISION_VIDEO_SPLICE_SITE
            assert metadata["target_site"] == metadata["latest_splice"]["target_site"] == expected_site
            assert metadata["dense_projector_packing_reference"] is (position == "before")
            if position != "before":
                continue
            # Each generation prepares the video once; cached continuation must
            # neither recompress it nor retain the temporary source hook.
            model.eval()
            generation = dict(inputs=args["input_ids"], images=[args["images"][0].detach()],
                image_sizes=args["image_sizes"], modalities=["video"], attention_mask=args["attention_mask"],
                max_new_tokens=2, min_new_tokens=2, do_sample=False, use_cache=True,
                return_dict_in_generate=True, eos_token_id=None, pad_token_id=0)
            first = model.generate(**generation)
            second = model.generate(**generation)
            torch.testing.assert_close(first.sequences, second.sequences, atol=0, rtol=0)
            assert len(requests) == len(encoded) == 3
            assert bridge.runtime_metadata()["callback_count"] == 3
            assert len(projector._forward_pre_hooks) == original_hook_count
            assert first.past_key_values.get_seq_length() == output.logits.shape[1] + 1
        finally:
            for hook in hooks:
                hook.remove()
            handle.remove()
            bridge.detach()
        assert "prepare_inputs_labels_for_multimodal" not in vars(model)


def test_native_pre_capture_cleanup_on_pool_failure_and_fixed_grid_projector(monkeypatch):
    torch = pytest.importorskip("torch")
    from mllm_comp.backbones.llava_onevision.video_compression import LLAVA_ONEVISION_VIDEO_SPLICE_SITE, install_llava_onevision_video_bridge
    from mllm_comp.compressors.video import VQTokenVideoKernel
    from mllm_comp.core.errors import BackendIntegrationError
    model = _video_model()
    projector = model.model.mm_projector
    bridge = install_llava_onevision_video_bridge(model)
    handle = bridge.register_callback(LLAVA_ONEVISION_VIDEO_SPLICE_SITE,
        lambda request: VQTokenVideoKernel()(request.payload, config={}), projector_position="before")
    original_count = len(projector._forward_pre_hooks)

    def fail_pool(*args, **kwargs):
        raise RuntimeError("injected native pool failure")

    monkeypatch.setattr(model, "get_2dPool", fail_pool)
    try:
        with pytest.raises(RuntimeError, match="native pool failure"):
            model(**_inputs(torch))
        assert len(projector._forward_pre_hooks) == original_count
    finally:
        handle.remove()
        bridge.detach()
    assert "prepare_inputs_labels_for_multimodal" not in vars(model)
    from llava.model.multimodal_projector.pooler_projector import PoolerProjector
    model.config.mm_hidden_size = 16
    model.model.mm_projector = PoolerProjector(model.config, model.model.vision_tower.config)
    bridge = install_llava_onevision_video_bridge(model)
    try:
        with pytest.raises(BackendIntegrationError, match="complete fixed spatial grid"):
            bridge.register_callback(LLAVA_ONEVISION_VIDEO_SPLICE_SITE, lambda request: request,
                projector_position="before")
        assert not bridge.patched
    finally:
        bridge.detach()
