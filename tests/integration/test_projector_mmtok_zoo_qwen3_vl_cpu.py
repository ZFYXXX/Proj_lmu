"""Qwen3 MMTok/Zoo keep dense scoring sources while moving mixed values."""

import copy

import pytest

from mllm_comp.pipeline.stage import CompressionStageConfig
from tests.fixtures import native_image as image_fixture


def _stage(algorithm, main_pre):
    target = "multisource_main_image_features" if algorithm == "mmtok" else "projector_sensitivity_main_image_features"
    config = {"target_count": 2}
    if algorithm == "zoo_prune":
        config.update(num_directions=2, low_rank_rank=2, direction_batch_size=1, seed=13)
    return CompressionStageConfig.from_dict({
        "name": f"{algorithm}-mixed-sources", "placement": "projector_level", "execution_phase": "forward",
        "plugin": algorithm, "algorithm": algorithm, "variant": f"{algorithm}.image",
        "binding": f"{algorithm}.image.qwen3_vl", "target_modalities": ["image"], "config": config,
        "operation": {"targets": {
            target: {"target_site": "projector_input" if main_pre else "projector_output"},
            "deepstack_image_features": {"target_site": "projector_output" if main_pre else "projector_input"},
        }},
    })


@pytest.mark.parametrize("algorithm", ["mmtok", "zoo_prune"])
@pytest.mark.parametrize("main_pre", [True, False])
def test_mixed_values_keep_full_projected_scoring_sources_and_native_norm(algorithm, main_pre, monkeypatch):
    torch = pytest.importorskip("torch")
    from mllm_comp.backbones.qwen3_vl import compression_adapter as adapter
    from mllm_comp.backbones.qwen3_vl import provider_bridge as provider

    torch.manual_seed(509)
    model = image_fixture.qwen_model("qwen3_vl").train()
    visual = model.model.visual
    reference = copy.deepcopy(visual.merger)
    activation = image_fixture.activation(model, _stage(algorithm, main_pre))
    activation.install()
    callback = activation.compress_after_visual_mergers
    kernel_name = "mmtok_image_tokens" if algorithm == "mmtok" else "zoo_prune_image_tokens"
    kernel, restore = getattr(adapter, kernel_name), provider._restore_tome_steps
    source, records, visual_outputs = {}, [], []
    branch_values, branch_results = [], []
    mergers = [visual.merger, *visual.deepstack_merger_list]
    counts = [[] for _ in mergers]

    def capture_raw(*args, **kwargs):
        value = restore(*args, **kwargs)
        source["raw"] = value
        return value

    def check_kernel(projected, first_source, second_source, **kwargs):
        raw = source["raw"]
        expected = reference(raw)
        assert projected.shape == (4, 24)
        torch.testing.assert_close(projected, expected)
        assert projected.requires_grad is not main_pre
        if algorithm == "mmtok":
            vision, text = first_source, second_source
            torch.testing.assert_close(vision, raw.reshape(4, 64))
            torch.testing.assert_close(text, source["prompt"])
            normalise = lambda value: torch.nn.functional.normalize(value.float(), dim=-1, eps=1e-8)
            spec = kwargs["spec"]
            text_coverage = (normalise(text) @ normalise(projected).T / spec.text_temperature).softmax(1) / text.shape[0]
            vision_coverage = (normalise(vision) @ normalise(vision).T / spec.vision_temperature).softmax(1) / 4
            coverage = torch.cat((text_coverage, vision_coverage * spec.alpha))
            best, selected = torch.zeros(coverage.shape[0]), []
            for _ in range(2):
                gains = (coverage - best[:, None]).clamp_min(0).sum(0)
                gains[selected] = -torch.inf
                index = int(gains.argmax())
                selected.append(index)
                best = torch.maximum(best, coverage[:, index])
            expected_indices = sorted(selected)
        else:
            inputs, layers = first_source, second_source
            if reference.use_postshuffle_norm:
                expected_inputs = reference.norm(raw.reshape(-1, reference.hidden_size))
            else:
                expected_inputs = reference.norm(raw).reshape(-1, reference.hidden_size)
            torch.testing.assert_close(inputs, expected_inputs)
            assert tuple(layers) == (visual.merger.linear_fc1, visual.merger.act_fn, visual.merger.linear_fc2)
            assert inputs.device == layers[0].weight.device
            assert kwargs["spec"].projector_proxy == "low_rank" and kwargs["spec"].low_rank_rank == 2
        result = kernel(projected, first_source, second_source, **kwargs)
        if algorithm == "zoo_prune":
            sensitivity = result.sensitivity
            scores = (sensitivity - sensitivity.min()) / (sensitivity.max() - sensitivity.min() + 1e-8)
            first = int(scores.argmax())
            features = torch.nn.functional.normalize(projected.float(), dim=-1, eps=1e-8)
            second_scores = scores * (1 - features @ features[first])
            second_scores[first] = -torch.inf
            expected_indices = sorted([first, int(second_scores.argmax())])
            assert result.metadata["projector_proxy"] == "low_rank"
        assert result.indices.tolist() == expected_indices
        return result

    def check_application(main, branches, **kwargs):
        assert main.shape == (4, 64 if main_pre else 24)
        assert all(branch.shape == (4, 24 if main_pre else 64) for branch in branches)
        assert (kwargs.get("projected_reference") is not None) is main_pre
        result = callback(main, branches, **kwargs)
        indices = result.kernel_result.indices
        torch.testing.assert_close(result.main_image_features, main[indices])
        for value, compact in zip(branches, result.deepstack_image_features):
            torch.testing.assert_close(compact, value[indices.to(value.device)])
            value.retain_grad()
            compact.retain_grad()
            branch_values.append(value)
            branch_results.append(compact)
        records.append(result)
        return result

    monkeypatch.setattr(provider, "_restore_tome_steps", capture_raw)
    monkeypatch.setattr(adapter, kernel_name, check_kernel)
    monkeypatch.setattr(activation, "compress_after_visual_mergers", check_application)
    handles = [merger.register_forward_pre_hook(lambda module, args, i=i: counts[i].append(args[0].shape[0]))
               for i, merger in enumerate(mergers)]
    handles.append(visual.register_forward_hook(lambda module, args, output: visual_outputs.append(output)))
    try:
        args = image_fixture.inputs(torch, "qwen3_vl")
        ids = args["input_ids"][0]
        source["prompt"] = model.model.get_input_embeddings()(ids[ids != 10])
        labels = torch.full_like(args["input_ids"], -100)
        labels[0, -1] = args["input_ids"][0, -1]
        output = model(**args, labels=labels, use_cache=False)
        assert len(records) == len(visual_outputs) == 1
        assert counts == ([[16, 8], [16], [16], [16]] if main_pre else [[16], [8], [8], [8]])
        compressed = records[0].main_image_features
        expected_output = reference(compressed.reshape(-1, 16)) if main_pre else compressed
        torch.testing.assert_close(visual_outputs[0][0], expected_output)
        output.loss.backward()
        selected = records[0].kernel_result.indices
        for value, compact in zip(branch_values, branch_results):
            expected_gradient = torch.zeros_like(value)
            expected_gradient[selected.to(value.device)] = compact.grad
            torch.testing.assert_close(value.grad, expected_gradient)
        assert args["pixel_values"].grad.abs().sum() > 0
        assert all(merger.linear_fc2.weight.grad.abs().sum() > 0 for merger in mergers)
        metadata = activation.runtime_metadata()
        assert metadata["requires_full_projected_scoring_reference"] is True
        assert metadata["layout_delta_records"][0]["requires_full_projected_scoring_reference"] is True
        assert metadata["projector_target_sites"] == list(activation.projector_target_sites)
    finally:
        for handle in handles:
            handle.remove()
        activation.uninstall()
