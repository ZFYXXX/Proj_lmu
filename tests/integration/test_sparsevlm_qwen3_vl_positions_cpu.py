"""Early SparseVLM means must reach Qwen3-VL's remaining DeepStack branches."""

import pytest

from tests.fixtures import native_image as image_fixture
from tests.integration.test_image_sparsevlm_positions_cpu import _stage


@pytest.mark.parametrize("point,mid,checkpoint", [
    (0, False, None), (1, False, None),
    (0, True, None), (0, True, False), (0, True, True),
])
def test_sparsevlm_remaining_deepstack_means_and_member_gradients(point, mid, checkpoint, monkeypatch):
    torch = pytest.importorskip("torch")
    from mllm_comp.compressors.image import decoder_selection as adapter

    torch.manual_seed(389)
    model = image_fixture.model("qwen3_vl", num_decoder_layers=4).train()
    if checkpoint is not None:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": checkpoint})
    stage = _stage("qwen3_vl", (point,), sources=(0,), mid=mid)
    # Three recycled members share one cluster, making selection-only replay
    # observably wrong even with this four-token visual fixture.
    stage.config.update(schedule_keep_tokens=[1], recycle_ratio=1.0)
    activation = image_fixture.activation(model, stage)
    activation.install()
    decoder = image_fixture.decoder(model, "qwen3_vl")
    native_process, kernel = decoder._deepstack_process, adapter.sparsevlm_image_tokens
    original, injected, groups, applications = {}, {}, [], []

    def observe_branch(index):
        def hook(module, args, output):
            output.retain_grad()
            original[index] = output
        return hook

    def observe_kernel(hidden, **kwargs):
        result = kernel(hidden, **kwargs)
        images = kwargs["image_indices"].tolist()
        mapping = result.layout_delta.old_to_new
        targets = sorted({mapping[source] for source in images if mapping[source] is not None})
        groups.extend([[local for local, source in enumerate(images) if mapping[source] == target] for target in targets])
        assert sorted(map(len, groups)) == [1, 3]
        applications.append(result.layout_delta)
        return result

    def check_injection(hidden, mask, features):
        index = len(injected)
        expected = original[index]
        if index >= point:
            expected = torch.stack([expected[group].mean(0) for group in groups])
        torch.testing.assert_close(features, expected)
        features.retain_grad()
        injected[index] = features
        return native_process(hidden, mask, features)

    handles = [merger.register_forward_hook(observe_branch(index))
               for index, merger in enumerate(model.model.visual.deepstack_merger_list)]
    monkeypatch.setattr(adapter, "sparsevlm_image_tokens", observe_kernel)
    monkeypatch.setattr(decoder, "_deepstack_process", check_injection)
    try:
        args = image_fixture.inputs(torch, "qwen3_vl")
        labels = torch.full_like(args["input_ids"], -100)
        labels[0, -1] = args["input_ids"][0, -1]
        output = model(**args, labels=labels, use_cache=False)
        output.loss.backward()
        assert output.logits.shape[1] == 6
        assert len(applications) == 1 and len(injected) == 3
        for index, branch in original.items():
            assert branch.grad is not None and branch.grad.abs().sum() > 0
            if index >= point:
                expected_gradient = torch.zeros_like(branch)
                for target, group in enumerate(groups):
                    expected_gradient[group] = injected[index].grad[target] / len(group)
                torch.testing.assert_close(branch.grad, expected_gradient)
        assert args["pixel_values"].grad.abs().sum() > 0
    finally:
        for handle in handles:
            handle.remove()
        activation.uninstall()
