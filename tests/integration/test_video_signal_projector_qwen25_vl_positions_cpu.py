"""Qwen2.5 video signal reducers keep native cues and move actual values."""

import copy

import pytest

from mllm_comp.pipeline.stage import CompressionStageConfig
from tests.fixtures import native_image as image_fixture


def _model():
    model = image_fixture.qwen_model("qwen25_vl")
    model.model.visual.fullatt_block_indexes = [1, 2]
    return model


def _inputs(torch, frames=2):
    ids = torch.tensor([[1, 7, *([11] * (frames * 196)), 8, 2]])
    return dict(input_ids=ids, attention_mask=torch.ones_like(ids),
        pixel_values_videos=torch.randn(frames * 784, 12),
        video_grid_thw=torch.tensor([[frames, 28, 28]]),
        second_per_grid_ts=torch.tensor([0.5]))


def _stage(algorithm, position, *, phase="forward"):
    return CompressionStageConfig.from_dict({
        "name": f"{algorithm}-video-projector", "placement": "projector_level", "execution_phase": phase,
        "plugin": algorithm, "algorithm": algorithm, "variant": f"{algorithm}.video",
        "binding": f"{algorithm}.video.qwen25_vl", "target_modalities": ["video"], "config": {},
        "operation": {} if position is None else {"target_site": position},
    })


def _received_attention(torch, block, values, kwargs):
    # Independent native rotary QK calculation, before any group reduction.
    with torch.no_grad():
        hidden = block.norm1(values[0])
        q, k, _ = block.attn.qkv(hidden).reshape(hidden.shape[0], 3, block.attn.num_heads, -1).unbind(1)
        cos, sin = (value.unsqueeze(-2).float() for value in kwargs["position_embeddings"])
        def rotate(value):
            left, right = value.chunk(2, dim=-1)
            return torch.cat((-right, left), dim=-1)
        q, k = q.float(), k.float()
        q, k = q * cos + rotate(q) * sin, k * cos + rotate(k) * sin
        boundaries = kwargs["cu_seqlens"].tolist()
        return torch.cat([(torch.einsum("qhd,khd->hqk", q[start:stop], k[start:stop])
            * block.attn.scaling).softmax(-1).mean((0, 1))
            for start, stop in zip(boundaries, boundaries[1:])])


@pytest.mark.parametrize("algorithm", ["visionzip", "fastvid"])
def test_native_signal_sources_actual_raw_merger_math_gradients_and_default_alias(algorithm):
    torch = pytest.importorskip("torch")
    torch.manual_seed(431)
    base = _model().train()
    source = _inputs(torch)
    expected_default = None
    for position in (None, "projector_output", "projector_input"):
        model = copy.deepcopy(base)
        visual = model.model.visual
        native_merger = copy.deepcopy(visual.merger)
        args = {name: value.detach().clone() for name, value in source.items()}
        args["pixel_values_videos"].requires_grad_(True)
        window, _ = visual.get_window_index(args["video_grid_thw"])
        reverse = window.argsort()
        assert not torch.equal(window, torch.arange(392))
        before = position == "projector_input"
        activation = image_fixture.activation(model, _stage(algorithm, position))
        captured, layer_outputs, layer_inputs, ln_inputs, merger_outputs = {}, {}, {}, [], []
        hooks = []
        for index in (1, 2):
            hooks.append(visual.blocks[index].register_forward_pre_hook(
                lambda module, values, kwargs, index=index: layer_inputs.__setitem__(index, (values, kwargs)), with_kwargs=True))
            hooks.append(visual.blocks[index].register_forward_hook(
                lambda module, values, output, index=index: layer_outputs.__setitem__(index, output)))
        hooks.append(visual.merger.ln_q.register_forward_pre_hook(lambda module, values: ln_inputs.append(values[0])))
        hooks.append(visual.merger.register_forward_hook(lambda module, values, output: merger_outputs.append(output)))
        original_kernel = activation.kernel

        def kernel(payload, **kwargs):
            raw = layer_outputs[2].reshape(392, 64)[reverse]
            with torch.no_grad():
                projected = native_merger(layer_outputs[2])[reverse]
            torch.testing.assert_close(payload.features, raw if before else projected)
            signal_layer = 1 if algorithm == "visionzip" else 2
            attention = _received_attention(torch, visual.blocks[signal_layer], *layer_inputs[signal_layer])
            attention = attention.reshape(392, 4).mean(1)[reverse]
            if algorithm == "visionzip":
                torch.testing.assert_close(payload.signals["vision_encoder_attention"][-2], attention, rtol=2e-5, atol=2e-6)
                torch.testing.assert_close(payload.signals["vision_encoder_hidden_states"][-2],
                    layer_outputs[1].reshape(392, 64)[reverse])
            else:
                torch.testing.assert_close(payload.signals["token_saliency_attention"], attention, rtol=2e-5, atol=2e-6)
                torch.testing.assert_close(payload.signals["frame_global_features"], projected.reshape(2, 196, 24).mean(1))
                assert payload.signals["frame_global_features"].shape == (2, 24)
                expected_raster = raw.reshape(2, 14, 14, 2, 2, 16).permute(0, 1, 3, 2, 4, 5).reshape(1568, 16)
                torch.testing.assert_close(payload.raw_features, expected_raster)
                if before:
                    assert not payload.signals["frame_global_features"].requires_grad
                    assert payload.metadata["dense_projected_reference_tokens"] == 392
            result = original_kernel(payload, **kwargs)
            values = payload.features
            if algorithm == "visionzip":
                targets = result.metadata["merge_target_indices_by_video"][0]
                expected = []
                for group, target in zip(result.source_groups[0], targets):
                    assigned = [index for index in group if index != target]
                    expected.append(values[target] + values[assigned].mean(0) if assigned else values[target])
                for frame, record in enumerate(result.metadata["frame_records_by_video"][0]):
                    scores = payload.signals["vision_encoder_attention"][-2]
                    wanted = set((scores[frame * 196:(frame + 1) * 196].argsort(descending=True, stable=True)[:126] + frame * 196).tolist())
                    assert set(record["dominant_indices"]) == wanted
                assert result.result_count == 294
            else:
                expected = []
                for record in result.metadata["value_records_by_video"][0]:
                    center, assigned, weight = record["center_index"], record["assigned_indices"], record["center_weight"]
                    expected.append(weight * values[center] + (1 - weight) * values[list(assigned)].mean(0)
                        if assigned else values[center])
                assert result.result_count == 38
            expected = torch.stack(expected)
            torch.testing.assert_close(result.features, expected)
            captured["compact"] = expected
            return result

        activation.kernel = kernel
        try:
            activation.install()
            labels = args["input_ids"].masked_fill(args["input_ids"] == 11, -100)
            output = model(**args, labels=labels, use_cache=False)
            count = captured["compact"].shape[0]
            assert output.logits.shape == (1, count + 4, 64)
            if before:
                expected_counts = ([1568] if algorithm == "fastvid" else []) + [count * 4]
                assert [value.shape[0] for value in ln_inputs] == expected_counts
                torch.testing.assert_close(ln_inputs[-1], captured["compact"].reshape(count * 4, 16))
                torch.testing.assert_close(merger_outputs[-1], native_merger(ln_inputs[-1]))
                if algorithm == "fastvid":
                    assert [value.requires_grad for value in merger_outputs] == [True, False, True]
            else:
                assert [value.shape[0] for value in ln_inputs] == [1568]
            output.loss.backward()
            pixel_grad = args["pixel_values_videos"].grad
            gradients = {name: value.grad for name, value in visual.merger.named_parameters()}
            assert pixel_grad is not None and torch.isfinite(pixel_grad).all() and pixel_grad.abs().sum() > 0
            assert all(value is not None and torch.isfinite(value).all() for value in gradients.values())
            assert any(value.abs().sum() > 0 for value in gradients.values())
            if position is None:
                expected_default = output.logits.detach(), pixel_grad, gradients
            elif position == "projector_output":
                torch.testing.assert_close((output.logits.detach(), pixel_grad, gradients), expected_default, rtol=0, atol=0)
            bridge = activation._bridge.runtime_metadata()
            assert bridge["dense_projected_reference_required"] is (before and algorithm == "fastvid")
            assert bridge["latest_splice"]["dense_projected_reference_tokens"] == (392 if before and algorithm == "fastvid" else 0)
        finally:
            for hook in hooks:
                hook.remove()
            activation.uninstall()


@pytest.mark.parametrize("algorithm", ["visionzip", "fastvid"])
def test_signal_projector_before_repeated_native_generate_and_cache(algorithm):
    torch = pytest.importorskip("torch")
    torch.manual_seed(432)
    model = _model().eval()
    activation = image_fixture.activation(model, _stage(algorithm, "projector_input", phase="generate"))
    args = _inputs(torch)
    expected_length = 298 if algorithm == "visionzip" else 42
    outputs = []
    try:
        activation.install()
        for repeat in range(2):
            with torch.no_grad():
                generated = model.generate(**args, min_new_tokens=3, max_new_tokens=3,
                    do_sample=False, use_cache=True, eos_token_id=None, pad_token_id=0,
                    return_dict_in_generate=True)
            bridge = activation._bridge.runtime_metadata()
            assert bridge["callback_count"] == repeat + 1
            assert bridge["latest_generation"]["result_sequence_length"] == expected_length
            assert bridge["latest_generation"]["decode_repairs"] == 2
            assert [layer.get_seq_length() for layer in generated.past_key_values.layers] == [expected_length + 2] * 4
            outputs.append(generated.sequences)
        torch.testing.assert_close(outputs[0], outputs[1])
    finally:
        activation.uninstall()
