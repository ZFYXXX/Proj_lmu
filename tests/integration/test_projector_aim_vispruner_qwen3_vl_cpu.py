"""AIM's original branch means and VisPruner's fixed vision source at new sites."""

import copy

import pytest

from mllm_comp.core.layout import OrderingPolicy
from mllm_comp.pipeline.stage import CompressionStageConfig
from tests.fixtures import native_image as image_fixture
from tests.integration import test_image_tome_positions_cpu as native


def _stage(algorithm, *, main_pre=True, deep_pre=False, target=2):
    main_target = "visual_attention_main_image_features" if algorithm == "vispruner" else "main_image_features"
    return CompressionStageConfig.from_dict({
        "name": f"{algorithm}-projector", "placement": "projector_level", "execution_phase": "forward",
        "plugin": algorithm, "algorithm": algorithm, "variant": f"{algorithm}.image",
        "binding": f"{algorithm}.image.qwen3_vl", "target_modalities": ["image"],
        "config": {"target_count": target},
        "operation": {"targets": {
            main_target: {"target_site": "projector_input" if main_pre else "projector_output"},
            "deepstack_image_features": {"target_site": "projector_input" if deep_pre else "projector_output"},
        }},
    })


def test_aim_mixed_native_projectors_keep_original_multiround_branch_means(monkeypatch):
    torch = pytest.importorskip("torch")
    torch.manual_seed(459)
    model = image_fixture.qwen_model("qwen3_vl").train()
    visual = model.model.visual
    reference_merger = copy.deepcopy(visual.merger)
    activation = image_fixture.activation(model, _stage("aim", target=1))
    activation.install()
    callback = activation.compress_after_visual_mergers
    projected_branches, merged_branches, visual_outputs, main_inputs = [], [], [], []
    expected_main = []

    def observe(main, branches, **kwargs):
        assert main.shape == (4, 64) and all(branch.shape == (4, 24) for branch in branches)
        assert not main_inputs  # The main LN/projector has not executed yet.
        result = callback(main, branches, **kwargs)
        assert result.layout_delta.ordering_policy is OrderingPolicy.EXPLICIT
        assert result.kernel_result.token_sizes.tolist() == [4.0]
        expected_main.append(reference_merger(result.main_image_features.reshape(-1, 16)))
        for source, merged in zip(branches, result.deepstack_image_features):
            torch.testing.assert_close(merged, source.mean(0, keepdim=True))
            source.retain_grad()
            merged.retain_grad()
            projected_branches.append(source)
            merged_branches.append(merged)
        return result

    monkeypatch.setattr(activation, "compress_after_visual_mergers", observe)
    handles = [visual.merger.register_forward_pre_hook(lambda module, args: main_inputs.append(args[0])),
               visual.register_forward_hook(lambda module, args, output: visual_outputs.append(output))]
    try:
        args = image_fixture.inputs(torch, "qwen3_vl")
        labels = torch.full_like(args["input_ids"], -100)
        labels[0, -1] = args["input_ids"][0, -1]
        result = model(**args, labels=labels, use_cache=False)
        assert len(main_inputs) == 1 and main_inputs[0].shape == (4, 16)
        torch.testing.assert_close(visual_outputs[0][0], expected_main[0])
        result.loss.backward()
        for source, merged in zip(projected_branches, merged_branches):
            torch.testing.assert_close(source.grad, merged.grad.expand_as(source) / 4)
        assert args["pixel_values"].grad.abs().sum() > 0
        assert all(merger.linear_fc2.weight.grad.abs().sum() > 0 for merger in visual.deepstack_merger_list)
    finally:
        for handle in handles:
            handle.remove()
        activation.uninstall()


def test_aim_multiround_main_and_deepstack_retain_their_existing_distinct_weights():
    torch = pytest.importorskip("torch")
    model = image_fixture.qwen_model("qwen3_vl")
    activation = image_fixture.activation(model, _stage("aim", target=1))
    # Both even sources first join destination 1. The second iteration merges
    # current nodes equally; old DeepStack behavior averages all four members.
    main = torch.tensor([[1., 0.], [1., 0.], [1., 0.], [0., 1.]], requires_grad=True)
    branch_device = "cuda:0" if torch.cuda.is_available() else "cpu"
    branches = tuple(torch.arange(12, dtype=torch.float32, device=branch_device).reshape(4, 3).requires_grad_()
                     for _ in model.model.visual.deepstack_merger_list)
    result = activation.compress_after_visual_mergers(main, branches, image_index=0)
    torch.testing.assert_close(result.main_image_features, main.new_tensor([[0.5, 0.5]]))
    for source, merged in zip(branches, result.deepstack_image_features):
        torch.testing.assert_close(merged, source.mean(0, keepdim=True))
        merged.sum().backward()
        torch.testing.assert_close(source.grad, torch.full_like(source, 0.25))
    result.main_image_features.sum().backward()
    torch.testing.assert_close(main.grad, main.new_tensor([[1/6, 1/6], [1/6, 1/6], [1/6, 1/6], [1/2, 1/2]]))
    assert result.kernel_result.token_sizes.tolist() == [4.0]
    assert result.layout_delta.ordering_policy is OrderingPolicy.EXPLICIT


@pytest.mark.parametrize("with_tome", [False, True])
def test_vispruner_preserves_final_vision_reference_and_attention_after_tome(with_tome, monkeypatch):
    torch = pytest.importorskip("torch")
    from mllm_comp.backbones.qwen3_vl import provider_bridge as provider

    torch.manual_seed(460)
    model = image_fixture.qwen_model("qwen3_vl").train()
    visual = model.model.visual
    activation = image_fixture.activation(model, _stage("vispruner", main_pre=not with_tome, deep_pre=not with_tome))
    tome = image_fixture.activation(model, native._stage("qwen3_vl", "mid", r=1)) if with_tome else None
    if tome is not None:
        tome.install()
    activation.install()
    restore, metrics = provider._restore_tome_steps, provider._visionzip_metrics_from_capture
    callback = activation.compress_after_visual_mergers
    source = {}
    consumed = []

    def capture_dense(*args, **kwargs):
        value = restore(*args, **kwargs)
        source["features"] = value.reshape(-1, 64)
        return value

    def capture_metrics(*args, **kwargs):
        attentions, keys = metrics(*args, **kwargs)
        source["attention"] = attentions[0]
        return attentions, keys

    if tome is not None:
        merge = tome.merge_output_tokens
        def capture_tome(*args, **kwargs):
            result = merge(*args, **kwargs)
            groups = result.sequence_source_groups
            source["features"] = torch.stack([source["features"][list(group)].mean(0) for group in groups])
            attention = source["attention"]
            source["attention"] = torch.stack([torch.stack([
                attention[list(row)][:, list(column)].mean() for column in groups]) for row in groups])
            return result
        monkeypatch.setattr(tome, "merge_output_tokens", capture_tome)

    def check_source(main, branches, *, vision_reference, attention, **kwargs):
        torch.testing.assert_close(vision_reference, source["features"])
        torch.testing.assert_close(attention, source["attention"])
        assert main.shape[1] == (24 if with_tome else 64)
        assert main.shape[0] == (3 if with_tome else 4)
        result = callback(main, branches, vision_reference=vision_reference, attention=attention, **kwargs)
        indices = result.kernel_result.indices
        torch.testing.assert_close(result.main_image_features, main[indices])
        for before, after in zip(branches, result.deepstack_image_features):
            torch.testing.assert_close(after, before[indices.to(before.device)])
        consumed.append(result)
        return result

    monkeypatch.setattr(provider, "_restore_tome_steps", capture_dense)
    monkeypatch.setattr(provider, "_visionzip_metrics_from_capture", capture_metrics)
    monkeypatch.setattr(activation, "compress_after_visual_mergers", check_source)
    try:
        args = image_fixture.inputs(torch, "qwen3_vl")
        labels = torch.full_like(args["input_ids"], -100)
        labels[0, -1] = args["input_ids"][0, -1]
        output = model(**args, labels=labels, use_cache=False)
        output.loss.backward()
        assert len(consumed) == 1 and output.logits.shape[1] == 6
        assert args["pixel_values"].grad.abs().sum() > 0
    finally:
        activation.uninstall()
        if tome is not None:
            tome.uninstall()
