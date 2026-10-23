"""Fixed video signals and actual values stay separate across Qwen3 projectors."""

import copy

import pytest

from mllm_comp.pipeline.stage import CompressionStageConfig
from tests.fixtures import native_image as image_fixture


def _stage(algorithm, position=None, phase="forward"):
    target = "dominant_contextual_reduce" if algorithm == "visionzip" else "dyseg_stprune_dtm"
    return CompressionStageConfig.from_dict({
        "name": f"{algorithm}-video-position", "placement": "projector_level", "execution_phase": phase,
        "plugin": algorithm, "algorithm": algorithm, "variant": f"{algorithm}.video",
        "binding": f"{algorithm}.video.qwen3_vl", "target_modalities": ["video"],
        **({} if position is None else {"operation": {"targets": {target: {
            "target_site": "projector_input" if position == "before" else "projector_output",
        }}}}),
    })


def _inputs(torch):
    # Native Qwen3 videos delimit each temporal slice separately.
    ids = torch.tensor([[1, *([7, *([11] * 196), 8] * 2), 2]])
    return dict(input_ids=ids, attention_mask=torch.ones_like(ids),
                pixel_values_videos=torch.randn(1568, 12, requires_grad=True),
                video_grid_thw=torch.tensor([[2, 28, 28]]))


def _independent_values(torch, algorithm, source, result):
    rows = []
    if algorithm == "visionzip":
        for group, target in zip(result.source_groups[0], result.metadata["merge_target_indices_by_video"][0]):
            assigned = [index for index in group if index != target]
            rows.append(source[target] + source[assigned].mean(0) if assigned else source[target])
    else:
        for record in result.metadata["value_records_by_video"][0]:
            assigned = list(record["assigned_indices"])
            center = source[record["center_index"]]
            weight = record["center_weight"]
            rows.append(weight * center + (1 - weight) * source[assigned].mean(0) if assigned else center)
    return torch.stack(rows)


@pytest.mark.parametrize("algorithm", ["visionzip", "fastvid"])
def test_fixed_native_signals_actual_values_deepstack_math_gradient_and_default_alias(algorithm, monkeypatch):
    torch = pytest.importorskip("torch")
    torch.manual_seed(1042)
    base = image_fixture.qwen_model("qwen3_vl").train()
    args = _inputs(torch)
    default_output = None
    for position in (None, "after", "before"):
        model = copy.deepcopy(base)
        visual = model.model.visual
        mergers = [visual.merger, *visual.deepstack_merger_list]
        activation = image_fixture.activation(model, _stage(algorithm, position))
        provider = activation.signal_provider
        provider.install()
        raw_sources = [[] for _ in mergers]
        hooks = [merger.register_forward_pre_hook(
            lambda module, inputs, i=i: raw_sources[i].append(inputs[0])) for i, merger in enumerate(mergers)]
        try:
            with torch.no_grad(), provider.capture_context(grid_thw=args["video_grid_thw"]) as capture:
                dense_main, dense_deep = model.model.get_video_features(
                    args["pixel_values_videos"], args["video_grid_thw"])
        finally:
            for hook in hooks:
                hook.remove()
        source_values = [values[0] for values in raw_sources]
        dense_values = [dense_main[0], *dense_deep]
        layer = -2 if algorithm == "visionzip" else -1
        assert provider._capture_owner() is visual.blocks[layer]
        applied = []
        calls = [[] for _ in mergers]
        callback = activation._projector_callback

        def check(request):
            payload = request.payload
            actual = (payload.features, *request.deepstack_features)
            for i, value in enumerate(actual):
                expected = source_values[i].reshape(392, 64) if position == "before" else dense_values[i]
                torch.testing.assert_close(value, expected)
            if algorithm == "visionzip":
                torch.testing.assert_close(payload.signals["vision_encoder_hidden_states"][-2],
                                           source_values[2].reshape(392, 64))
                torch.testing.assert_close(payload.signals["vision_encoder_attention"][-2],
                                           capture.attention.reshape(392, 4).mean(1))
            else:
                torch.testing.assert_close(payload.signals["frame_global_features"], dense_values[0].reshape(2, 196, 24).mean(1))
                torch.testing.assert_close(payload.signals["token_saliency_attention"], capture.raw_attention.reshape(392, 4).mean(1))
                raw_raster = source_values[0].reshape(2, 14, 14, 2, 2, 16).permute(0, 1, 3, 2, 4, 5).reshape(1568, 16)
                torch.testing.assert_close(payload.raw_features, raw_raster)
                assert payload.signals["frame_global_features"].shape[1] == 24
            response = callback(request)
            result = response.compression_result
            for value, compressed in zip(actual, (result.features, *response.deepstack_features)):
                torch.testing.assert_close(compressed, _independent_values(torch, algorithm, value, result))
            assert response.synchronization_source_groups == result.source_groups
            applied.append(response)
            return response

        monkeypatch.setattr(activation, "_projector_callback", check)
        activation.install()
        hooks = [merger.norm.register_forward_pre_hook(
            lambda module, inputs, i=i: calls[i].append(inputs[0])) for i, merger in enumerate(mergers)]
        pixels = args["pixel_values_videos"].detach().clone().requires_grad_(True)
        labels = torch.full_like(args["input_ids"], -100)
        labels[0, -1] = 2
        try:
            output = model(**{**args, "pixel_values_videos": pixels}, labels=labels, use_cache=False)
            response = applied[0]
            count = response.compression_result.result_count
            if algorithm == "visionzip":
                assert count == 2 * 147
            assert output.logits.shape[1] == 398 - 392 + count
            for i, value in enumerate((response.compression_result.features, *response.deepstack_features)):
                dense_extra = position == "before" and algorithm == "fastvid" and i == 0
                assert len(calls[i]) == 1 + int(dense_extra)
                rows = count if position == "before" else 392
                expected_shape = (rows, 64) if mergers[i].use_postshuffle_norm else (rows * 4, 16)
                assert tuple(calls[i][-1].shape) == expected_shape
                if position == "before":
                    expected_input = value if mergers[i].use_postshuffle_norm else value.reshape(-1, 16)
                    torch.testing.assert_close(calls[i][-1], expected_input)
            output.loss.backward()
            assert pixels.grad is not None and torch.isfinite(pixels.grad).all() and pixels.grad.abs().sum() > 0
            for merger in mergers:
                assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in merger.parameters())
                assert merger.linear_fc2.weight.grad.abs().sum() > 0
            record = activation._bridge.runtime_metadata()["latest_projector"]
            dense_extra = 392 if position == "before" and algorithm == "fastvid" else 0
            projected_rows = count if position == "before" else 392
            assert record["dense_main_scoring_reference_rows"] == dense_extra
            assert record["projector_input_patch_rows"] == (projected_rows + dense_extra) * 4
            assert record["deepstack_projector_input_patch_rows"] == projected_rows * 4
            if position is None:
                default_output = output.logits.detach()
            elif position == "after":
                torch.testing.assert_close(output.logits, default_output, rtol=0, atol=0)
        finally:
            for hook in hooks:
                hook.remove()
            activation.uninstall()


@pytest.mark.parametrize("algorithm", ["visionzip", "fastvid"])
def test_before_fixed_signal_native_repeated_generate_and_compact_cache(algorithm):
    torch = pytest.importorskip("torch")
    torch.manual_seed(1043)
    model = image_fixture.qwen_model("qwen3_vl").eval()
    args = _inputs(torch)
    activation = image_fixture.activation(model, _stage(algorithm, "before", "generate"))
    activation.install()
    runs = []
    try:
        for _ in range(2):
            with torch.no_grad():
                result = model.generate(**args, max_new_tokens=3, do_sample=False, eos_token_id=None,
                                        pad_token_id=0, use_cache=True, return_dict_in_generate=True)
            runs.append(result.sequences)
            runtime = activation._bridge.runtime_metadata()
            count = runtime["latest_projector"]["result_group_count"]
            generation = runtime["latest_generation"]
            assert generation["prefill_complete"]
            assert generation["source_sequence_length"] == 398
            assert generation["result_sequence_length"] == 6 + count
            assert generation["decode_repairs"] >= 1
            assert result.past_key_values.get_seq_length() == 8 + count
        torch.testing.assert_close(runs[0], runs[1], rtol=0, atol=0)
        assert activation.runtime_metadata()["projector_invocations"] == 2
    finally:
        activation.uninstall()
