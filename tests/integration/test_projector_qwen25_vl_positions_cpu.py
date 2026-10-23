"""Projector-input compression preserves Qwen2.5 spatial groups and ToMe order."""

import copy

import pytest

from mllm_comp.pipeline.stage import CompressionStageConfig
from tests.fixtures import native_image as image_fixture
from tests.integration.test_image_tome_positions_cpu import _stage as _tome_stage


def _visionzip_stage():
    return CompressionStageConfig.from_dict({
        "name": "visionzip-pre-projector", "placement": "projector_level", "execution_phase": "forward",
        "plugin": "visionzip", "algorithm": "visionzip", "variant": "visionzip.image",
        "binding": "visionzip.image.qwen25_vl", "target_modalities": ["image"],
        "operation": {"target_site": "projector_input"}, "config": {"budget": 5},
    })


@pytest.mark.parametrize("with_tome", [False, True])
def test_nonidentity_windows_raw_groups_native_merger_and_tome(monkeypatch, with_tome):
    torch = pytest.importorskip("torch")
    from mllm_comp.compressors.image.visionzip import visionzip_image

    torch.manual_seed(361)
    model = image_fixture.qwen_model("qwen25_vl").train()
    visual = model.model.visual
    visual.fullatt_block_indexes = [1, 2]
    native_merger = copy.deepcopy(visual.merger)
    grid = torch.tensor([[1, 8, 8]])
    window, _ = visual.get_window_index(grid)
    assert not torch.equal(window, torch.arange(16))
    reverse = window.argsort()
    activation = image_fixture.activation(model, _visionzip_stage())
    tome = image_fixture.activation(model, _tome_stage("qwen25_vl", "mid", r=1)) if with_tome else None
    captured, events = {}, []
    original_callback = activation.post_visual_merger

    def compression(raw, attention, keys, **kwargs):
        events.append("pre_compression")
        expected_raw = captured["final_visual"].reshape(16, 64)[reverse]
        torch.testing.assert_close(raw, expected_raw)
        assert raw.shape == (16, 64)  # 4 raw patches, before the merger LN.
        expected = visionzip_image(expected_raw, attention, keys, budget=5)
        captured["compressed_patches"] = expected.tokens.reshape(20, 16)
        result = original_callback(raw, attention, keys, **kwargs)
        torch.testing.assert_close(result.kernel_result.tokens, expected.tokens)
        return result

    activation.post_visual_merger = compression
    def merger_input(module, args):
        events.append("merger")
        captured["merger_input"] = args[0]

    handles = [visual.merger.register_forward_pre_hook(merger_input)]
    handles.append(visual.merger.register_forward_hook(
        lambda module, args, output: captured.update(merger_output=output)))
    if tome is not None:
        original_output = tome.compact_visual_output

        def tome_output(tokens, keys, **kwargs):
            events.append("tome_output")
            assert tokens.shape[0] == 5
            assert keys.shape[0] == 5
            return original_output(tokens, keys, **kwargs)

        tome.compact_visual_output = tome_output
    try:
        activation.install()
        if tome is not None:
            tome.install()
        bridge = activation._bridge
        original_block = bridge._execute_visual_block

        def block(*args, **kwargs):
            result = original_block(*args, **kwargs)
            if kwargs["layer_index"] == len(visual.blocks) - 1:
                captured["final_visual"] = result[0]
            return result

        monkeypatch.setattr(bridge, "_execute_visual_block", block)
        ids = torch.tensor([[1, 7, *([10] * 16), 8, 2]])
        pixels = torch.randn(64, 12, requires_grad=True)
        result = model(input_ids=ids, attention_mask=torch.ones_like(ids), pixel_values=pixels,
            image_grid_thw=grid, labels=ids.masked_fill(ids == 10, -100), use_cache=False)
        assert events == ["pre_compression", "merger", *(["tome_output"] if with_tome else [])]
        torch.testing.assert_close(captured["merger_input"], captured["compressed_patches"])
        torch.testing.assert_close(captured["merger_output"], native_merger(captured["compressed_patches"]))
        assert result.logits.shape == (1, 8 if with_tome else 9, 64)
        result.loss.backward()
        assert pixels.grad is not None and torch.isfinite(pixels.grad).all() and pixels.grad.abs().sum() > 0
        for parameter in visual.merger.parameters():
            assert parameter.grad is not None and torch.isfinite(parameter.grad).all()
        metadata = activation.runtime_metadata()
        assert metadata["projector_target_sites"] == ["pre_vision_merger"]
        records = metadata["layout_delta_records"]
        assert len(records) == 1
        assert records[0]["source_count"] == 16 and records[0]["result_count"] == 5
        assert records[0]["result_layout"]["target_site"] == "pre_vision_merger"
        if tome is not None:
            record = tome.runtime_metadata()["layout_delta_records"][-1]
            assert record["site"] == "permanent_visual_output"
            assert (record["source_count"], record["result_count"]) == (5, 4)
    finally:
        for handle in handles:
            handle.remove()
        if tome is not None:
            tome.uninstall()
        activation.uninstall()
