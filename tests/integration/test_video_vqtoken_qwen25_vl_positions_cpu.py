"""Native Qwen2.5 video VQToken uses the actual projector-side feature space."""

import copy

import pytest
from tests.fixtures import native_image as image_fixture


def _inputs(torch):
    ids = torch.tensor([[1, 7, *([11] * 48), 8, 2]])
    return dict(input_ids=ids, attention_mask=torch.ones_like(ids),
        pixel_values_videos=torch.randn(192, 12), video_grid_thw=torch.tensor([[3, 8, 8]]),
        second_per_grid_ts=torch.tensor([0.5]))


def _install(model, callback, position=None):
    from mllm_comp.backbones.qwen25_vl.video_compression import (
        QWEN25_VL_VIDEO_SPLICE_SITE, install_qwen25_vl_video_bridge,
    )
    bridge = install_qwen25_vl_video_bridge(model)
    handle = bridge.register_callback(QWEN25_VL_VIDEO_SPLICE_SITE, callback,
        **({} if position is None else {"projector_position": position}))
    return bridge, handle


def test_before_native_window_raw_kmeans_normalized_mean_and_merger_gradient(monkeypatch):
    torch = pytest.importorskip("torch")
    from mllm_comp.backbones.base import BackboneRuntime
    from mllm_comp.backbones.qwen25_vl.implementation import Qwen25VLRunner
    from mllm_comp.compressors.video import vqtoken as kernel_module
    from mllm_comp.core.state import ForwardState

    torch.manual_seed(421)
    model = image_fixture.qwen_model("qwen25_vl").train()
    visual = model.model.visual
    native_merger = copy.deepcopy(visual.merger)
    native_forward = visual.merger.forward
    args = _inputs(torch)
    args["pixel_values_videos"].requires_grad_(True)
    window, _ = visual.get_window_index(args["video_grid_thw"])
    assert not torch.equal(window, torch.arange(48))
    merger_inputs, ln_inputs, merger_outputs, captured = [], [], [], {}
    hooks = [visual.merger.register_forward_pre_hook(lambda module, values: merger_inputs.append(values[0])),
        visual.merger.ln_q.register_forward_pre_hook(lambda module, values: ln_inputs.append(values[0])),
        visual.merger.register_forward_hook(lambda module, values, output: merger_outputs.append(output))]
    original_kmeans = kernel_module._stable_cosine_kmeans

    def kmeans(features, **kwargs):
        torch.testing.assert_close(features, captured["normalized"].detach().float())
        captured["kmeans_calls"] = captured.get("kmeans_calls", 0) + 1
        return original_kmeans(features, **kwargs)

    monkeypatch.setattr(kernel_module, "_stable_cosine_kmeans", kmeans)

    def compress(request):
        # The real native visual forward has applied its N-row window inverse.
        raw = merger_inputs[0].reshape(48, 64)[window.argsort()]
        assert request.payload.features.shape == (48, 64)
        torch.testing.assert_close(request.payload.features, raw)
        assert visual.merger.forward == native_forward
        assert not ln_inputs
        normalized = raw / raw.norm(dim=-1, keepdim=True)
        captured["normalized"] = normalized
        result = kernel_module.vqtoken_video(request.payload)
        compact = torch.stack([normalized[list(group)].mean(0) for group in result.source_groups[0]])
        torch.testing.assert_close(result.features, compact)
        captured["compact"] = compact.reshape(128, 16)
        captured["representatives"] = result.representative_indices[0]
        return result

    bridge, handle = _install(model, compress, "before")
    try:
        labels = args["input_ids"].masked_fill(args["input_ids"] == 11, -100)
        output = Qwen25VLRunner().forward_train(
            BackboneRuntime(model=model, processor=None),
            ForwardState(native_model_inputs={**args, "labels": labels}),
        )
        assert captured["kmeans_calls"] == 1
        assert [value.shape[0] for value in merger_inputs] == [192, 128]
        assert len(ln_inputs) == 1
        torch.testing.assert_close(ln_inputs[0], captured["compact"])
        torch.testing.assert_close(merger_outputs[-1], native_merger(captured["compact"]))
        assert output.logits.shape == (1, 36, 64)
        output.loss.backward()
        pixels = args["pixel_values_videos"]
        assert pixels.grad is not None and torch.isfinite(pixels.grad).all() and pixels.grad.abs().sum() > 0
        assert all(parameter.grad is not None and torch.isfinite(parameter.grad).all()
            for parameter in visual.merger.parameters())
        assert any(parameter.grad.abs().sum() > 0 for parameter in visual.merger.parameters())
        metadata = bridge.runtime_metadata()
        assert metadata["target_site"] == "video_pre_vision_merger"
        assert metadata["latest_splice"]["target_site"] == "video_pre_vision_merger"
        assert tuple(metadata["latest_splice"]["representative_indices"]) == captured["representatives"]
        assert visual.merger.forward == native_forward
    finally:
        for hook in hooks:
            hook.remove()
        handle.remove()
        bridge.detach()


def test_default_and_explicit_after_preserve_native_projected_values_and_gradients():
    torch = pytest.importorskip("torch")
    from mllm_comp.compressors.video.vqtoken import vqtoken_video

    torch.manual_seed(422)
    base = image_fixture.qwen_model("qwen25_vl").train()
    raw = _inputs(torch)
    expected = None
    for position in (None, "after"):
        model = copy.deepcopy(base)
        args = {name: value.detach().clone() for name, value in raw.items()}
        args["pixel_values_videos"].requires_grad_(True)
        ln_counts = []
        hook = model.model.visual.merger.ln_q.register_forward_pre_hook(
            lambda module, values: ln_counts.append(values[0].shape[0]))

        def compress(request):
            assert request.payload.features.shape == (48, 24)
            return vqtoken_video(request.payload)

        bridge, handle = _install(model, compress, position)
        try:
            labels = args["input_ids"].masked_fill(args["input_ids"] == 11, -100)
            output = model(**args, labels=labels, use_cache=False)
            output.loss.backward()
            assert ln_counts == [192]
            observed = (output.logits.detach(), args["pixel_values_videos"].grad,
                {name: parameter.grad for name, parameter in model.model.visual.merger.named_parameters()})
            assert observed[1] is not None and observed[1].abs().sum() > 0
            if expected is None:
                expected = observed
            else:
                torch.testing.assert_close(observed, expected, rtol=0, atol=0)
            assert bridge.runtime_metadata()["target_site"] == "video_post_merger_pre_llm_scatter"
        finally:
            hook.remove()
            handle.remove()
            bridge.detach()


@pytest.mark.parametrize("position", ["before", "after"])
def test_native_generate_repeats_requests_with_one_vqtoken_prefill_and_compact_cache(position):
    torch = pytest.importorskip("torch")
    from mllm_comp.compressors.video.vqtoken import vqtoken_video

    torch.manual_seed(423)
    model = image_fixture.qwen_model("qwen25_vl").eval()
    args = _inputs(torch)
    calls = []

    def compress(request):
        calls.append(request.payload.features.shape)
        return vqtoken_video(request.payload)

    bridge, handle = _install(model, compress, position)
    outputs = []
    try:
        for repeat in range(2):
            with torch.no_grad():
                generated = model.generate(**args, min_new_tokens=3, max_new_tokens=3,
                    do_sample=False, use_cache=True, eos_token_id=None, pad_token_id=0,
                    return_dict_in_generate=True)
            assert len(calls) == repeat + 1
            assert calls[-1] == (48, 64 if position == "before" else 24)
            assert [layer.get_seq_length() for layer in generated.past_key_values.layers] == [38] * 4
            lifecycle = bridge.runtime_metadata()["latest_generation"]
            assert lifecycle["prefill_complete"] is True
            assert lifecycle["source_sequence_length"] == 52
            assert lifecycle["result_sequence_length"] == 36
            assert lifecycle["decode_repairs"] == 2
            outputs.append(generated.sequences)
        torch.testing.assert_close(outputs[0], outputs[1])
    finally:
        handle.remove()
        bridge.detach()


def test_before_encoder_failure_restores_native_merger_and_request_state():
    torch = pytest.importorskip("torch")
    from mllm_comp.compressors.video.vqtoken import vqtoken_video

    torch.manual_seed(424)
    model = image_fixture.qwen_model("qwen25_vl").eval()
    native_forward = model.model.visual.merger.forward
    bridge, handle = _install(model, lambda request: vqtoken_video(request.payload), "before")

    def fail(module, values):
        raise RuntimeError("native encoder fixture failure")

    hook = model.model.visual.blocks[-1].register_forward_pre_hook(fail)
    try:
        with pytest.raises(RuntimeError, match="native encoder fixture failure"):
            model(**_inputs(torch), use_cache=False)
        assert model.model.visual.merger.forward == native_forward
        hook.remove()
        with torch.no_grad():
            output = model(**_inputs(torch), use_cache=False)
        assert output.logits.shape == (1, 36, 64)
    finally:
        hook.remove()
        handle.remove()
        bridge.detach()
