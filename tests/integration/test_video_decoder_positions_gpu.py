"""A fixed METok score source and later mid applications across two GPUs."""

from dataclasses import replace

import pytest

from tests.integration.test_video_metok_positions_cpu import (
    _model, _inputs, _executor, _context, _stage, _bilinear,
)


def test_qwen3_metok_fixed_sources_and_mid_values_across_two_gpus(monkeypatch):
    torch = pytest.importorskip("torch")
    if torch.cuda.device_count() < 2:
        pytest.skip("Requires the authorized two-GPU validation job.")
    from accelerate import dispatch_model
    from mllm_comp.backbones import video_decoder as provider
    import mllm_comp.compressors.video.metok as metok_kernel
    from mllm_comp.compressors.video.metok import metok_video

    torch.manual_seed(471)
    model = _model("qwen3_vl").train()
    visual, decoder = model.model.visual, model.model.language_model
    device_map = {name: "cuda:1" for name, _ in model.named_children() if name != "model"}
    for name, child in model.model.named_children():
        if name == "visual":
            device_map.update({f"model.visual.{part}": "cuda:1" if part in {"merger", "deepstack_merger_list"} else "cuda:0"
                for part, _ in child.named_children()})
        elif name == "language_model":
            for part, _ in child.named_children():
                if part == "layers":
                    device_map.update({f"model.language_model.layers.{i}": "cuda:0" if i < 2 else "cuda:1"
                        for i in range(len(child.layers))})
                else:
                    device_map[f"model.language_model.{part}"] = "cuda:1"
        else:
            device_map[f"model.{name}"] = "cuda:1"
    dispatch_model(model, device_map=device_map)
    stage = _stage("qwen3_vl", "mid")
    stage.operation["targets"]["prefill_attention_prune"]["layers"] = [2, 3, 4]
    assert stage.config["pruning_layers"] == [0, 1, 19]
    activation = _executor(model, "qwen3_vl", stage)
    source_devices, source_scores, applied_devices, planned, requests = [], [], [], [], []
    ffn_devices = []
    original_scores = provider._native_pruning_scores
    original_boundary = provider.decoder_post_attention_boundary
    original_plan = metok_kernel.plan_metok_pruning_stage
    original_callback = activation._projector_callback

    def scores(**kwargs):
        qk_function = kwargs["qk_function"]
        devices = []
        def qk(*args, **values):
            result = qk_function(*args, **values)
            devices.extend((str(result[0].device), str(result[2].device)))
            return result
        result = original_scores(**{**kwargs, "qk_function": qk})
        source_devices.append((kwargs["decoder_layer"].self_attn.layer_idx, *devices, str(result.device)))
        source_scores.append(result.detach().cpu())
        return result

    def boundary(layer, hidden_states, *, apply_compression, **kwargs):
        def apply(residual):
            applied_devices.append((layer.self_attn.layer_idx, str(residual.device)))
            return apply_compression(residual)
        return original_boundary(layer, hidden_states, apply_compression=apply, **kwargs)

    def plan(**kwargs):
        index = kwargs["stage_index"]
        assert kwargs["visual_scores"].device == torch.device("cuda:1")
        if index < 2:
            expected = source_scores[index][list(kwargs["retained_projector_indices"])]
            torch.testing.assert_close(kwargs["visual_scores"].detach().cpu(), expected, atol=0, rtol=0)
        else:
            assert kwargs["visual_scores"].count_nonzero() == 0
        result = original_plan(**kwargs)
        planned.append(result)
        return result

    def callback(request):
        payload = request.payload
        reference = payload.signals["projector_reference_features"]
        assert payload.features.device == torch.device("cuda:0")
        assert reference.device == torch.device("cuda:1")
        assert payload.features.shape == (392, 64) and reference.shape == (392, 24)
        oracle = metok_video(replace(payload, features=reference), config=activation.config)
        response = original_callback(request)
        result = response.compression_result
        assert result.source_groups == oracle.source_groups
        assert result.metadata["metok_segments"] == oracle.metadata["metok_segments"]
        torch.testing.assert_close(result.features, _bilinear(torch, payload.features, oracle))
        for source, branch in zip(request.deepstack_features, response.deepstack_features):
            torch.testing.assert_close(branch, _bilinear(torch, source, oracle))
        requests.append(request)
        return response

    monkeypatch.setattr(provider, "_native_pruning_scores", scores)
    monkeypatch.setattr(provider, "decoder_post_attention_boundary", boundary)
    monkeypatch.setattr(metok_kernel, "plan_metok_pruning_stage", plan)
    activation._projector_callback = callback
    hooks = [decoder.layers[i].mlp.register_forward_pre_hook(
        lambda module, args, index=i: ffn_devices.append((index, str(args[0].device)))) for i in (2, 3, 4)]
    activation.install()
    torch.manual_seed(472)
    args = _inputs(torch, "qwen3_vl")
    args = {name: value.detach().to("cuda:0") if torch.is_tensor(value) else value for name, value in args.items()}
    pixels = args["pixel_values_videos"].requires_grad_(True)
    try:
        with activation._bridge.request_context(_context("qwen3_vl")):
            output = model(**args)
        output.loss.backward()
        assert torch.isfinite(output.loss)
        assert source_devices == [(0, "cuda:0", "cuda:0", "cuda:0"), (1, "cuda:0", "cuda:0", "cuda:0")]
        assert applied_devices == ffn_devices == [(2, "cuda:1"), (3, "cuda:1"), (4, "cuda:1")]
        assert len(requests) == 1 and len(planned) == 3
        assert pixels.grad is not None and torch.isfinite(pixels.grad).all() and pixels.grad.abs().sum() > 0
        for module in (visual.merger, *visual.deepstack_merger_list, *decoder.layers):
            gradients = [parameter.grad for parameter in module.parameters()]
            assert all(gradient is not None and torch.isfinite(gradient).all() for gradient in gradients)
            assert any(gradient.abs().sum() > 0 for gradient in gradients)
        metadata = activation.runtime_metadata()
        expected_positions = [{"boundary": "post_attention_pre_ffn", "index": i} for i in (2, 3, 4)]
        assert metadata["score_layers"] == [0, 1]
        assert metadata["application_positions"] == expected_positions
        assert [record["application_position"] for record in metadata["pruning_layout_deltas"]] == expected_positions
        assert metadata["projector_position"] == "before"
        assert metadata["projector_invocations"] == 1 and metadata["pruning_invocations"] == 3
        assert metadata["canonical_evidence_applicable"] is False
        assert activation._active.get() is None
    finally:
        for hook in hooks:
            hook.remove()
        activation.uninstall()
