"""SAINT preserves native vision-key axes and mixed Qwen3 DeepStack values."""

import pytest

from mllm_comp.pipeline.stage import CompressionStageConfig
from tests.fixtures import native_image as image_fixture


def _stage(main_pre):
    return CompressionStageConfig.from_dict({
        "name": "saint-mixed-projector", "placement": "projector_level", "execution_phase": "forward",
        "plugin": "saint", "algorithm": "saint", "variant": "saint.image.packed_feature_prune",
        "binding": "saint.image.qwen3_vl", "target_modalities": ["image"],
        "config": {"target_count": 1, "group_size": 1},
        "operation": {"targets": {
            "main_image_features": {"target_site": "projector_input" if main_pre else "projector_output"},
            "deepstack_image_features": {"target_site": "projector_output" if main_pre else "projector_input"},
        }},
    })


@pytest.mark.parametrize("main_pre", [True, False])
def test_saint_equal_token_head_counts_keep_fixed_keys_and_mixed_branch_gradients(main_pre, monkeypatch):
    torch = pytest.importorskip("torch")
    from mllm_comp.backbones.qwen3_vl import compression_adapter as adapter
    from mllm_comp.backbones.qwen3_vl import provider_bridge as provider

    torch.manual_seed(479)
    model = image_fixture.qwen_model("qwen3_vl").train()
    visual = model.model.visual
    activation = image_fixture.activation(model, _stage(main_pre))
    activation.install()
    metrics, kernel = provider._visionzip_metrics_from_capture, adapter.saint_image_tokens
    callback = activation.compress_after_visual_mergers
    fixed_keys, results, branch_inputs, branch_outputs = [], [], [], []
    merger_counts = [[] for _ in range(4)]
    qkv_calls = []

    def capture_metrics(capture, **kwargs):
        assert capture.counts == (8,)
        # This image has two original 2x2 groups and two native attention
        # heads. Equal dimension sizes must never select the wrong axis.
        expected = capture.keys[0].reshape(2, 4, 2, -1).mean(1)
        assert not torch.allclose(expected, expected.transpose(0, 1))
        fixed_keys.append(expected)
        return metrics(capture, **kwargs)

    def check_kernel(tokens, keys, **kwargs):
        torch.testing.assert_close(keys, fixed_keys[0])
        assert tokens.shape == (2, 64 if main_pre else 24)
        assert kwargs["spec"].target_count == 1 and kwargs["spec"].group_size == 1
        return kernel(tokens, keys, **kwargs)

    def check_application(main, branches, **kwargs):
        assert len(fixed_keys) == 1
        assert all(value.shape == (2, 24 if main_pre else 64) for value in branches)
        result = callback(main, branches, **kwargs)
        indices = result.kernel_result.selected_indices
        torch.testing.assert_close(result.main_image_features, main[indices])
        for source, compact in zip(branches, result.deepstack_image_features):
            torch.testing.assert_close(compact, source[indices.to(source.device)])
            source.retain_grad()
            compact.retain_grad()
            branch_inputs.append(source)
            branch_outputs.append(compact)
        assert all(delta is result.layout_delta for delta in result.branch_deltas)
        results.append(result)
        return result

    monkeypatch.setattr(provider, "_visionzip_metrics_from_capture", capture_metrics)
    monkeypatch.setattr(adapter, "saint_image_tokens", check_kernel)
    monkeypatch.setattr(activation, "compress_after_visual_mergers", check_application)
    mergers = [visual.merger, *visual.deepstack_merger_list]
    handles = [merger.register_forward_pre_hook(
        lambda module, args, i=i: merger_counts[i].append(args[0].shape[0])) for i, merger in enumerate(mergers)]
    handles.append(visual.blocks[-1].attn.qkv.register_forward_hook(lambda *args: qkv_calls.append(True)))
    try:
        ids = torch.tensor([[1, 7, 10, 10, 8, 2]])
        pixels = torch.randn(8, 12, requires_grad=True)
        labels = torch.full_like(ids, -100)
        labels[0, -1] = ids[0, -1]
        output = model(input_ids=ids, attention_mask=torch.ones_like(ids), pixel_values=pixels,
            image_grid_thw=torch.tensor([[1, 2, 4]]), labels=labels, use_cache=False)
        assert len(results) == len(qkv_calls) == 1
        assert merger_counts == ([[4], [8], [8], [8]] if main_pre else [[8], [4], [4], [4]])
        assert output.logits.shape[1] == 5
        output.loss.backward()
        selected = results[0].kernel_result.selected_indices
        for source, compact in zip(branch_inputs, branch_outputs):
            expected = torch.zeros_like(source)
            expected[selected.to(source.device)] = compact.grad
            torch.testing.assert_close(source.grad, expected)
        assert pixels.grad is not None and pixels.grad.abs().sum() > 0
        assert all(merger.linear_fc2.weight.grad.abs().sum() > 0 for merger in mergers)
        metadata = activation.runtime_metadata()
        assert metadata["projector_target_sites"] == list(activation.projector_target_sites)
        assert metadata["layout_delta_records"][0]["projector_target_sites"] == list(activation.projector_target_sites)
    finally:
        for handle in handles:
            handle.remove()
        activation.uninstall()
