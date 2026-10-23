"""AIM retains its ordered means; VisPruner keeps its original visual source."""

import pytest

from mllm_comp.core.layout import OrderingPolicy
from tests.fixtures import native_image as image_fixture
from tests.integration.test_image_projector_positions_cpu import _stage, _model
from tests.integration.test_image_tome_positions_cpu import _stage as _tome_stage


@pytest.mark.parametrize("multi_round", [False, True])
def test_aim_pre_projector_explicit_representatives_and_original_mean_gradient(multi_round):
    torch = pytest.importorskip("torch")
    from mllm_comp.backbones.qwen25_vl.provider_bridge import _compose_layout_mapping
    from mllm_comp.core.layout import LayoutDelta

    model = _model("qwen25_vl").eval()
    activation = image_fixture.activation(model, _stage("qwen25_vl", "aim"))
    pairs = ([[1., 0.], [2., 0.], [3., 0.], [0., 2.], [4., 0.], [-1., 0.]] if multi_round
        else [[1., 0.], [0., 2.], [0., 1.], [2., 0.]])
    raw = torch.tensor(pairs).repeat(1, 32).requires_grad_()
    result = activation.post_visual_merger(raw, None, None,
        placeholder_indices=tuple(range(len(pairs))), source_sequence_length=len(pairs), image_index=0)
    kernel = result.kernel_result
    assert kernel.layout_delta.ordering_policy is OrderingPolicy.EXPLICIT
    expected_representatives = (3, 5) if multi_round else (1, 3)
    assert result.prompt_repair.retained_placeholder_indices == expected_representatives
    assert kernel.token_sizes.tolist() == ([5., 1.] if multi_round else [2., 2.])
    # Each round averages current nodes, without reweighting by the final
    # group sizes. The two-round case deliberately creates an unequal tree.
    reference = raw.detach().clone().requires_grad_()
    expected = (torch.stack(((reference[[0, 1, 2, 4]].mean(0) + reference[3]) / 2, reference[5]))
        if multi_round else torch.stack(((reference[1] + reference[2]) / 2, (reference[0] + reference[3]) / 2)))
    torch.testing.assert_close(kernel.tokens, expected)
    merger = model.model.visual.merger
    actual_projected = merger(kernel.tokens.reshape(-1, 16))
    expected_projected = merger(expected.reshape(-1, 16))
    torch.testing.assert_close(actual_projected, expected_projected)
    actual_grad = torch.autograd.grad(actual_projected.square().sum(), raw)[0]
    expected_grad = torch.autograd.grad(expected_projected.square().sum(), reference)[0]
    torch.testing.assert_close(actual_grad, expected_grad)
    composed = _compose_layout_mapping(LayoutDelta.identity(len(pairs), media_unit="image_feature"), kernel.layout_delta)
    assert composed.ordering_policy is OrderingPolicy.EXPLICIT
    assert tuple(composed.result_layout["representative_indices"]) == expected_representatives
    if not multi_round:
        # Representative order differs from ordering groups by their minima.
        assert composed.old_to_new[0] == 1 and composed.old_to_new[1] == 0
    assert activation.runtime_metadata()["layout_delta_records"][0]["result_layout"]["target_site"] == "pre_vision_merger"


@pytest.mark.parametrize("position", ["projector_input", "projector_output"])
def test_vispruner_fixed_visual_source_stays_aligned_through_tome(monkeypatch, position):
    torch = pytest.importorskip("torch")
    from mllm_comp.backbones.qwen25_vl import image_compression as adapter
    from mllm_comp.backbones.qwen25_vl import provider_bridge as provider

    torch.manual_seed(381)
    model = _model("qwen25_vl").eval()
    activation = image_fixture.activation(model, _stage("qwen25_vl", "vispruner", position))
    tome = image_fixture.activation(model, _tome_stage("qwen25_vl", "mid", r=1))
    captured = {}
    original_kernel = adapter.vispruner_image_tokens
    original_metrics = provider._merged_vision_metrics
    original_output = tome.compact_visual_output

    def metrics(*args, **kwargs):
        result = original_metrics(*args, **kwargs)
        captured["fixed_attention"] = result[0][0]
        return result

    def output(*args, **kwargs):
        result = original_output(*args, **kwargs)
        captured["tome_delta"] = result.layout_delta
        return result

    def select(values, features, attention, **kwargs):
        reference = captured["final_visual"].reshape(4, 64)
        reference_attention = captured["fixed_attention"]
        if position == "projector_output":
            delta = captured["tome_delta"]
            groups = [[source for source, target in delta.old_to_new.items() if target == index]
                for index in range(delta.result_count)]
            reference = torch.stack([reference[group].mean(0) for group in groups])
            reference_attention = torch.stack([torch.stack([
                reference_attention[:, rows][:, :, cols].mean((-2, -1)) for cols in groups
            ], dim=-1) for rows in groups], dim=-2)
        torch.testing.assert_close(features, reference)
        torch.testing.assert_close(attention, reference_attention)
        assert values.shape[-1] == (64 if position == "projector_input" else 24)
        result = original_kernel(values, features, attention, **kwargs)
        torch.testing.assert_close(result.tokens, values[result.indices])
        captured["selection_count"] = int(result.indices.numel())
        return result

    monkeypatch.setattr(adapter, "vispruner_image_tokens", select)
    monkeypatch.setattr(provider, "_merged_vision_metrics", metrics)
    tome.compact_visual_output = output
    try:
        activation.install()
        tome.install()
        bridge = activation._bridge
        original_block = bridge._execute_visual_block

        def block(*args, **kwargs):
            result = original_block(*args, **kwargs)
            if kwargs["layer_index"] == 2:
                captured["final_visual"] = result[0]
            return result

        monkeypatch.setattr(bridge, "_execute_visual_block", block)
        with torch.no_grad():
            result = model(**image_fixture.inputs(torch, "qwen25_vl"), use_cache=False)
        assert captured["selection_count"] == 2
        assert result.logits.shape[1] == (5 if position == "projector_input" else 6)
        assert tome.runtime_metadata()["layout_delta_records"][-1]["site"] == "permanent_visual_output"
    finally:
        tome.uninstall()
        activation.uninstall()
