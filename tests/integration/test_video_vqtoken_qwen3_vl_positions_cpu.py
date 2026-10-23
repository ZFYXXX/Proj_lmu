"""Native Qwen3 video groups retain original merger norms and DeepStack math."""

import copy

import pytest

from mllm_comp.backbones.qwen3_vl.video_compression import (
    QWEN3_VL_VIDEO_PRE_PROJECTOR_SITE,
    QWEN3_VL_VIDEO_SPLICE_SITE,
    install_qwen3_vl_video_bridge,
)
from mllm_comp.backbones.vqtoken_video import VQTokenVideoExecutor
from tests.fixtures import native_image as image_fixture


def _inputs(torch):
    # 36 complete spatial groups: canonical K=32 remains a strict reduction.
    ids = torch.tensor([[1, 7, *([11] * 36), 8, 2]])
    return dict(input_ids=ids, attention_mask=torch.ones_like(ids),
                pixel_values_videos=torch.randn(144, 12, requires_grad=True),
                video_grid_thw=torch.tensor([[1, 12, 12]]))


def test_before_raw_groups_normalized_mean_deepstack_native_mergers_and_gradient():
    torch = pytest.importorskip("torch")
    torch.manual_seed(941)
    model = image_fixture.qwen_model("qwen3_vl").train()
    visual = model.model.visual
    mergers = [visual.merger, *visual.deepstack_merger_list]
    references = [copy.deepcopy(merger) for merger in mergers]
    bridge = install_qwen3_vl_video_bridge(model)
    executor = VQTokenVideoExecutor(backbone="qwen3_vl", model=model, execution_phase="forward")
    inputs, norm_inputs, projected = [[] for _ in mergers], [[] for _ in mergers], [[] for _ in mergers]
    observed = []

    def callback(request):
        values = (request.payload.features, *request.deepstack_features)
        assert request.payload.metadata["value_space"] == "raw_spatial_groups"
        assert request.payload.token_count == 36
        for index, value in enumerate(values):
            assert value.shape == (36, 64)
            torch.testing.assert_close(value, inputs[index][0].reshape(36, 64), rtol=0, atol=0)
            assert norm_inputs[index] == []
        response = executor._projector_callback(request)
        groups = response.compression_result.source_groups[0]
        assert len(groups) == 32
        for source, result in zip(values, (response.compression_result.features, *response.deepstack_features)):
            normalized = source / torch.linalg.vector_norm(source, dim=-1, keepdim=True)
            expected = torch.stack([normalized[list(group)].mean(0) for group in groups])
            torch.testing.assert_close(result, expected)
        observed.append(response)
        return response

    handle = bridge.register_callback(QWEN3_VL_VIDEO_SPLICE_SITE, callback, projector_position="before")
    hooks = []
    for index, merger in enumerate(mergers):
        hooks.append(merger.register_forward_pre_hook(lambda module, args, i=index: inputs[i].append(args[0])))
        hooks.append(merger.norm.register_forward_pre_hook(lambda module, args, i=index: norm_inputs[i].append(args[0])))
        hooks.append(merger.register_forward_hook(lambda module, args, value, i=index: projected[i].append(value)))
    originals = [merger.forward for merger in mergers]
    args = _inputs(torch)
    labels = torch.full_like(args["input_ids"], -100)
    labels[0, -1] = 2
    try:
        output = model(**args, labels=labels, use_cache=False)
        assert output.logits.shape == (1, 36, 64)
        response = observed[0]
        for index, value in enumerate((response.compression_result.features, *response.deepstack_features)):
            assert [tuple(item.shape) for item in inputs[index]] == [(144, 16), (128, 16)]
            torch.testing.assert_close(inputs[index][1], value.reshape(128, 16))
            assert len(norm_inputs[index]) == 1
            assert norm_inputs[index][0].shape == ((32, 64) if mergers[index].use_postshuffle_norm else (128, 16))
            torch.testing.assert_close(projected[index][-1], references[index](value.reshape(128, 16)))
            assert mergers[index].forward == originals[index]
        output.loss.backward()
        assert args["pixel_values_videos"].grad.abs().sum() > 0
        for merger in mergers:
            assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in merger.parameters())
            assert merger.linear_fc2.weight.grad.abs().sum() > 0
        record = bridge.runtime_metadata()["latest_projector"]
        assert record["actual_projector_site"] == QWEN3_VL_VIDEO_PRE_PROJECTOR_SITE
        assert record["projector_input_patch_rows"] == 128
        assert record["projector_output_rows"] == 32
    finally:
        for hook in hooks:
            hook.remove()
        handle.remove()
        bridge.detach()


def test_default_and_explicit_after_are_exact_and_report_dense_projector_work():
    torch = pytest.importorskip("torch")
    torch.manual_seed(942)
    model = image_fixture.qwen_model("qwen3_vl").eval()
    args = _inputs(torch)
    bridge = install_qwen3_vl_video_bridge(model)
    executor = VQTokenVideoExecutor(backbone="qwen3_vl", model=model, execution_phase="forward")
    outputs = []
    try:
        for kwargs in ({}, {"projector_position": "after"}):
            handle = bridge.register_callback(QWEN3_VL_VIDEO_SPLICE_SITE, executor._projector_callback, **kwargs)
            try:
                with torch.no_grad():
                    outputs.append(model(**args, use_cache=False).logits)
                record = bridge.runtime_metadata()["latest_projector"]
                assert record["actual_projector_site"] == QWEN3_VL_VIDEO_SPLICE_SITE
                assert record["value_space"] == "projected_features"
                assert record["projector_input_patch_rows"] == 144
                assert record["projector_output_rows"] == 36
            finally:
                handle.remove()
        torch.testing.assert_close(outputs[0], outputs[1], rtol=0, atol=0)
    finally:
        bridge.detach()


def test_before_repeated_native_generate_keeps_compact_cache_and_restores_mergers():
    torch = pytest.importorskip("torch")
    torch.manual_seed(943)
    model = image_fixture.qwen_model("qwen3_vl").eval()
    args = _inputs(torch)
    bridge = install_qwen3_vl_video_bridge(model)
    executor = VQTokenVideoExecutor(backbone="qwen3_vl", model=model)
    handle = bridge.register_callback(QWEN3_VL_VIDEO_SPLICE_SITE, executor._projector_callback,
                                      projector_position="before")
    mergers = [model.model.visual.merger, *model.model.visual.deepstack_merger_list]
    originals = [merger.forward for merger in mergers]
    runs = []
    try:
        for _ in range(2):
            with torch.no_grad():
                result = model.generate(**args, max_new_tokens=3, do_sample=False,
                                        eos_token_id=None, pad_token_id=0, use_cache=True,
                                        return_dict_in_generate=True)
            runs.append(result.sequences)
            generation = bridge.runtime_metadata()["latest_generation"]
            assert generation["prefill_complete"] is True
            assert generation["source_sequence_length"] == 40
            assert generation["result_sequence_length"] == 36
            assert generation["decode_repairs"] >= 1
            assert result.past_key_values.get_seq_length() == 38
            assert all(merger.forward == original for merger, original in zip(mergers, originals))
        torch.testing.assert_close(runs[0], runs[1], rtol=0, atol=0)
        assert bridge.runtime_metadata()["callback_count"] == 2
    finally:
        handle.remove()
        bridge.detach()


def test_before_encoder_failure_restores_all_mergers_and_next_native_request(monkeypatch):
    torch = pytest.importorskip("torch")
    model = image_fixture.qwen_model("qwen3_vl").eval()
    visual = model.model.visual
    mergers = [visual.merger, *visual.deepstack_merger_list]
    originals = [merger.forward for merger in mergers]
    bridge = install_qwen3_vl_video_bridge(model)
    executor = VQTokenVideoExecutor(backbone="qwen3_vl", model=model, execution_phase="forward")
    handle = bridge.register_callback(QWEN3_VL_VIDEO_SPLICE_SITE, executor._projector_callback,
                                      projector_position="before")
    args = _inputs(torch)
    def fail(*args, **kwargs):
        raise RuntimeError("encoder failure")
    try:
        with monkeypatch.context() as patch:
            patch.setattr(visual.blocks[-1], "forward", fail)
            with pytest.raises(RuntimeError, match="encoder failure"):
                model(**args, use_cache=False)
        assert all(merger.forward == original for merger, original in zip(mergers, originals))
        with torch.no_grad():
            assert model(**args, use_cache=False).logits.shape[1] == 36
            image, deepstack = model.model.get_image_features(torch.randn(16, 12), torch.tensor([[1, 4, 4]]))
        assert image[0].shape == (4, 24)
        assert all(branch.shape == (4, 24) for branch in deepstack)
    finally:
        handle.remove()
        bridge.detach()
