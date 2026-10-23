"""Qwen3 projector positions keep native groups and shared DeepStack decisions."""

import copy

import pytest

from mllm_comp.pipeline.stage import CompressionStageConfig
from tests.fixtures import native_image as image_fixture
from tests.integration import test_image_tome_positions_cpu as native


def _stage(algorithm, *, main_pre=True, deep_pre=True):
    targets = {"visionzip": ("main_image_features", "deepstack_image_features"),
        "divprune": ("main_image_features", "deepstack_image_features"),
        "prumerge": ("main_adaptive_merge", "deepstack_adaptive_merge"),
        "gap": ("main_grounding_aware_prune", "deepstack_grounding_aware_prune")}[algorithm]
    return CompressionStageConfig.from_dict({
        "name": f"{algorithm}-mixed-projectors", "placement": "projector_level", "execution_phase": "forward",
        "plugin": algorithm, "algorithm": algorithm, "variant": f"{algorithm}.image",
        "binding": f"{algorithm}.image.qwen3_vl", "target_modalities": ["image"],
        "config": ({"budget": 2} if algorithm == "visionzip" else
            {"target_count": 2, "adaptive": False, "cluster_size": 2} if algorithm == "prumerge" else {"target_count": 2}),
        "operation": {"targets": {
            targets[0]: {"target_site": "projector_input" if main_pre else "projector_output"},
            targets[1]: {"target_site": "projector_input" if deep_pre else "projector_output"},
        }},
    })


@pytest.mark.parametrize("algorithm,main_pre,deep_pre", [
    ("visionzip", True, False), ("prumerge", False, True),
    ("divprune", False, True), ("gap", True, False),
])
def test_mixed_projector_sites_use_raw_groups_and_preserve_deepstack_gradients(algorithm, main_pre, deep_pre, monkeypatch):
    torch = pytest.importorskip("torch")
    from mllm_comp.backbones.qwen3_vl import provider_bridge as provider

    torch.manual_seed(436)
    model = image_fixture.qwen_model("qwen3_vl").train()
    visual = model.model.visual
    mergers = [visual.merger, *visual.deepstack_merger_list]
    references = [copy.deepcopy(merger) for merger in mergers]
    activation = image_fixture.activation(model, _stage(algorithm, main_pre=main_pre, deep_pre=deep_pre))
    activation.install()
    raw_taps, applications, outputs = [], [], []
    merger_inputs = [[] for _ in mergers]
    restore, callback = provider._restore_tome_steps, activation.compress_after_visual_mergers

    def capture_raw(hidden, *args, **kwargs):
        value = restore(hidden, *args, **kwargs)
        if not raw_taps or value is not raw_taps[-1]:
            value.retain_grad()
            raw_taps.append(value)
        return value

    def check_application(main, branches, **kwargs):
        assert len(raw_taps) == 3
        raw = [raw_taps[-1], *raw_taps]
        for index, value in enumerate((main, *branches)):
            before = main_pre if index == 0 else deep_pre
            expected = raw[index].reshape(4, -1) if before else references[index](raw[index])
            torch.testing.assert_close(value, expected)
        result = callback(main, branches, **kwargs)
        assert result.layout_delta.source_count == 4 and result.layout_delta.result_count == 2
        assert all(delta is result.layout_delta for delta in result.branch_deltas)
        applications.append(result)
        return result

    monkeypatch.setattr(provider, "_restore_tome_steps", capture_raw)
    monkeypatch.setattr(activation, "compress_after_visual_mergers", check_application)
    handles = [merger.register_forward_pre_hook(
        lambda module, args, i=i: merger_inputs[i].append(args[0])) for i, merger in enumerate(mergers)]
    handles.append(visual.register_forward_hook(lambda module, args, output: outputs.append(output)))
    try:
        args = image_fixture.inputs(torch, "qwen3_vl")
        labels = torch.full_like(args["input_ids"], -100)
        labels[0, -1] = args["input_ids"][0, -1]
        output = model(**args, labels=labels, use_cache=False)
        assert len(applications) == len(outputs) == 1
        compressed = applications[0]
        actual = (outputs[0][0], *outputs[0][1])
        for index, value in enumerate((compressed.main_image_features, *compressed.deepstack_image_features)):
            before = main_pre if index == 0 else deep_pre
            assert len(merger_inputs[index]) == 1
            assert merger_inputs[index][0].shape == (8 if before else 16, 16)
            if before:
                torch.testing.assert_close(merger_inputs[index][0], value.reshape(-1, 16))
                expected = references[index](value.reshape(-1, 16))
            else:
                expected = value
            torch.testing.assert_close(actual[index], expected)
        output.loss.backward()
        assert args["pixel_values"].grad.abs().sum() > 0
        for merger in mergers:
            assert all(parameter.grad is not None and torch.isfinite(parameter.grad).all()
                       for parameter in merger.parameters())
            assert merger.linear_fc2.weight.grad.abs().sum() > 0
        metadata = activation.runtime_metadata()
        assert metadata["projector_target_sites"] == list(activation.projector_target_sites)
        assert metadata["layout_delta_records"][0]["projector_target_sites"] == list(activation.projector_target_sites)
    finally:
        for handle in handles:
            handle.remove()
        activation.uninstall()


@pytest.mark.parametrize("before", [True, False])
def test_visionzip_still_runs_tome_output_merge(monkeypatch, before):
    torch = pytest.importorskip("torch")
    torch.manual_seed(437)
    model = image_fixture.qwen_model("qwen3_vl").train()
    tome = image_fixture.activation(model, native._stage("qwen3_vl", "mid", r=1))
    visionzip = image_fixture.activation(model, _stage("visionzip", main_pre=before, deep_pre=before))
    tome.install()
    visionzip.install()
    events, captured, visual_outputs = [], {}, []
    visual = model.model.visual
    mergers = [visual.merger, *visual.deepstack_merger_list]
    merger_outputs = {}
    handles = [merger.register_forward_hook(
        lambda module, args, output, i=i: merger_outputs.update({i: output}))
        for i, merger in enumerate(mergers)]
    handles.append(visual.register_forward_hook(
        lambda module, args, output: visual_outputs.append(output)))
    zip_callback, tome_callback = visionzip.compress_after_visual_mergers, tome.merge_output_tokens

    def merged(values):
        return torch.stack([values[list(group)].mean(0)
            for group in captured["tome"].sequence_source_groups])

    def observe_zip(main, branches, **kwargs):
        if not before:
            for index, value in enumerate((main, *branches)):
                torch.testing.assert_close(value, merged(merger_outputs[index]))
        result = zip_callback(main, branches, **kwargs)
        captured["visionzip"] = result
        events.append(("visionzip", result.layout_delta.source_count, result.layout_delta.result_count))
        return result

    def observe_tome(hidden, keys, *args, **kwargs):
        count = 2 if before else 4
        assert hidden.shape == (count, 24) and keys.shape[0] == count
        result = tome_callback(hidden, keys, *args, **kwargs)
        captured["tome"] = result
        torch.testing.assert_close(result.hidden_states, merged(hidden))
        events.append(("tome_output", result.layout_delta.source_count, result.layout_delta.result_count))
        return result

    monkeypatch.setattr(visionzip, "compress_after_visual_mergers", observe_zip)
    monkeypatch.setattr(tome, "merge_output_tokens", observe_tome)
    try:
        args = image_fixture.inputs(torch, "qwen3_vl")
        labels = torch.full_like(args["input_ids"], -100)
        labels[0, -1] = args["input_ids"][0, -1]
        result = model(**args, labels=labels, use_cache=False)
        assert len(visual_outputs) == 1
        actual = (visual_outputs[0][0], *visual_outputs[0][1])
        compressed = captured["visionzip"]
        expected = (tuple(merged(merger_outputs[i]) for i in range(len(mergers))) if before
            else (compressed.main_image_features, *compressed.deepstack_image_features))
        for values, reference in zip(actual, expected):
            torch.testing.assert_close(values, reference)
        result.loss.backward()
        assert events == ([("visionzip", 4, 2), ("tome_output", 2, 1)] if before
            else [("tome_output", 4, 3), ("visionzip", 3, 2)])
        assert result.logits.shape[1] == (5 if before else 6)
        assert args["pixel_values"].grad.abs().sum() > 0
        for merger in model.model.visual.deepstack_merger_list:
            assert all(parameter.grad is not None and torch.isfinite(parameter.grad).all() for parameter in merger.parameters())
    finally:
        for handle in handles:
            handle.remove()
        visionzip.uninstall()
        tome.uninstall()
