"""The independent ToMe output merge selects raw or projected group values."""

import copy

import pytest

from mllm_comp.pipeline.stage import CompressionStageConfig
from tests.fixtures import native_image as image_fixture


def _stage(position=None, *, encoder="mid", backbone="qwen3_vl"):
    prefix = "vision" if backbone == "llava_onevision" else "visual"
    targets = {"encoder_merge": {"layers": [1], "target_site":
        f"{prefix}_block_post_attention_residual_pre_mlp" if encoder == "mid" else f"{prefix}_block_output"}}
    if position is not None:
        targets["permanent_output_merge"] = {"target_site": position}
    return CompressionStageConfig.from_dict({
        "name": "tome-output-position", "placement": "encoder_level", "execution_phase": "forward",
        "plugin": "tome", "algorithm": "tome", "variant": "tome.image.encoder_dense_boundary",
        "binding": f"tome.image.{backbone}", "target_modalities": ["image"],
        "operation": {"targets": targets}, "config": {"r": 1, "output_r": 1},
    })


def _groups(torch, keys):
    metric = torch.nn.functional.normalize(keys.detach().float().mean(1), dim=-1)
    scores = metric[::2] @ metric[1::2].T
    best, destinations = scores.max(-1)
    source_slot = int(best.argmax())
    source, destination = 2 * source_slot, 2 * int(destinations[source_slot]) + 1
    groups = [(i,) for i in range(len(keys)) if i not in (source, destination)]
    groups.append(tuple(sorted((source, destination))))
    return tuple(sorted(groups, key=lambda group: group[0]))


def _mean(torch, values, groups):
    return torch.stack([values[list(group)].mean(0) for group in groups])


@pytest.mark.parametrize("encoder,checkpoint", [("mid", None), ("full", None), ("mid", False), ("mid", True)])
def test_output_position_uses_fixed_keys_actual_values_and_all_branch_gradients(encoder, checkpoint, monkeypatch):
    torch = pytest.importorskip("torch")
    from mllm_comp.backbones.qwen3_vl import provider_bridge as provider

    torch.manual_seed(471)
    base = image_fixture.qwen_model("qwen3_vl").train()
    original_restore = provider._restore_tome_steps
    expected_default = None
    for position in (None, "post_visual_mergers_pre_llm_scatter_tome", "projector_input"):
        model = copy.deepcopy(base)
        if checkpoint is not None:
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": checkpoint})
        visual = model.model.visual
        mergers = [visual.merger, *visual.deepstack_merger_list]
        references = [copy.deepcopy(merger) for merger in mergers]
        activation = image_fixture.activation(model, _stage(position, encoder=encoder))
        restored, applications, visual_outputs = [], [], []
        merger_inputs = [[] for _ in mergers]
        before = position == "projector_input"

        def restore(*args, **kwargs):
            result = original_restore(*args, **kwargs)
            restored.append(result)
            return result

        callback = activation.merge_output_tokens

        def output_merge(values, keys, **kwargs):
            assert len(restored) == 6  # Each of the three taps plus final dense restore per layer.
            raw = restored[-1]
            source = raw.reshape(4, -1) if before else references[0](raw)
            torch.testing.assert_close(values, source)
            result = callback(values, keys, **kwargs)
            groups = _groups(torch, keys)
            assert result.sequence_source_groups == groups
            torch.testing.assert_close(result.hidden_states, _mean(torch, source, groups))
            assert result.layout_delta.source_count == 4 and result.layout_delta.result_count == 3
            applications.append((result, keys.detach().clone()))
            return result

        monkeypatch.setattr(provider, "_restore_tome_steps", restore)
        activation.merge_output_tokens = output_merge
        handles = [merger.register_forward_pre_hook(lambda module, args, i=i: merger_inputs[i].append(args[0]))
            for i, merger in enumerate(mergers)]
        handles.append(visual.register_forward_hook(lambda module, args, result: visual_outputs.append(result)))
        activation.install()
        torch.manual_seed(472)
        args = image_fixture.inputs(torch, "qwen3_vl")
        labels = torch.full_like(args["input_ids"], -100)
        labels[0, -1] = args["input_ids"][0, -1]
        try:
            output = model(**args, labels=labels, use_cache=False)
            assert len(applications) == len(visual_outputs) == 1
            result, keys = applications[0]
            raw = [restored[-1], restored[0], restored[2], restored[4]]
            actual = [visual_outputs[0][0], *visual_outputs[0][1]]
            for index in range(4):
                grouped = raw[index].reshape(4, -1)
                compact = _mean(torch, grouped if before else references[index](raw[index]), result.sequence_source_groups)
                expected = references[index](compact.reshape(-1, 16)) if before else compact
                torch.testing.assert_close(actual[index], expected)
                assert len(merger_inputs[index]) == 1
                assert merger_inputs[index][0].shape == (12 if before else 16, 16)
                if before:
                    torch.testing.assert_close(merger_inputs[index][0], compact.reshape(-1, 16))
            output.loss.backward()
            pixels_grad = args["pixel_values"].grad
            assert pixels_grad is not None and torch.isfinite(pixels_grad).all() and pixels_grad.abs().sum() > 0
            gradients = [parameter.grad.clone() for merger in mergers for parameter in merger.parameters()]
            assert all(torch.isfinite(gradient).all() for gradient in gradients)
            assert all(merger.linear_fc2.weight.grad.abs().sum() > 0 for merger in mergers)
            assert output.logits.shape[1] == 7
            if expected_default is None:
                expected_default = output.logits.detach().clone(), pixels_grad.clone(), gradients, keys, result.sequence_source_groups
            else:
                torch.testing.assert_close(keys, expected_default[3], atol=0, rtol=0)
                assert result.sequence_source_groups == expected_default[4]
                if not before:
                    torch.testing.assert_close((output.logits, pixels_grad, gradients), expected_default[:3], atol=0, rtol=0)
            records = activation.runtime_metadata()["layout_delta_records"]
            assert records[-1]["target_site"] == activation.output_target_site
            assert records[-1]["projector_position"] == ("before" if before else "after")
        finally:
            for handle in handles:
                handle.remove()
            activation.uninstall()
            monkeypatch.setattr(provider, "_restore_tome_steps", original_restore)


@pytest.mark.parametrize("position", ["projector_input", "projector_output"])
def test_output_position_repeated_forward_and_original_cache_rejection(position):
    torch = pytest.importorskip("torch")
    from mllm_comp.core.errors import BackendIntegrationError

    torch.manual_seed(473)
    model = image_fixture.qwen_model("qwen3_vl").eval()
    args = image_fixture.inputs(torch, "qwen3_vl")
    expected = None
    for _ in range(2):
        activation = image_fixture.activation(model, _stage(position))
        activation.install()
        try:
            with torch.no_grad():
                output = model(**args, use_cache=False)
                repeated = model(**args, use_cache=False)
            torch.testing.assert_close(output.logits, repeated.logits, atol=0, rtol=0)
            if expected is not None:
                torch.testing.assert_close(output.logits, expected, atol=0, rtol=0)
            expected = output.logits
            with pytest.raises(BackendIntegrationError):
                model(**args, use_cache=True)
            with pytest.raises(BackendIntegrationError):
                model.generate(**args, max_new_tokens=1)
        finally:
            activation.uninstall()
