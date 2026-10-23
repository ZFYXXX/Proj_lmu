"""MMTok/ZooPrune retain dense scoring references while selecting raw spatial groups."""

import copy

import pytest

from tests.fixtures import native_image as image_fixture
from tests.integration.test_image_projector_positions_cpu import _stage, _model


@pytest.mark.parametrize("algorithm", ["mmtok", "zoo_prune"])
def test_pre_projector_nonidentity_windows_keep_original_scoring_sources(monkeypatch, algorithm):
    torch = pytest.importorskip("torch")
    from mllm_comp.compressors.image.zoo_prune import zoo_prune_image_tokens, ZOOProjectorProxyCache

    torch.manual_seed(401)
    model = _model("qwen25_vl").train()
    visual = model.model.visual
    native_merger = copy.deepcopy(visual.merger)
    stage = _stage("qwen25_vl", algorithm)
    stage.config["target_count"] = 3
    if algorithm == "zoo_prune":
        stage.config.update(num_directions=2, projector_proxy="low_rank", low_rank_rank=2,
            low_rank_solver="randomized", direction_batch_size=1)
    activation = image_fixture.activation(model, stage)
    grid = torch.tensor([[1, 8, 8]])
    window, _ = visual.get_window_index(grid)
    reverse = window.argsort()
    assert not torch.equal(window, torch.arange(16))
    ids = torch.tensor([[1, 7, *([10] * 16), 8, 2]])
    expected_text = model.model.get_input_embeddings()(ids[0, ids[0] != 10])
    pixels = torch.randn(64, 12, requires_grad=True)
    captured, merger_inputs, merger_outputs = {}, [], []
    original = activation.post_visual_merger

    def compress(raw, attention, keys, **kwargs):
        dense = captured["final_visual"]
        expected_raw = dense.reshape(16, 64)[reverse]
        expected_projected = native_merger(dense)[reverse]
        torch.testing.assert_close(raw, expected_raw)
        torch.testing.assert_close(kwargs["vision_reference"], expected_raw)
        torch.testing.assert_close(kwargs["projected_reference"], expected_projected)
        assert raw.shape == (16, 64) and kwargs["projected_reference"].shape == (16, 24)
        assert not kwargs["projected_reference"].requires_grad
        if algorithm == "mmtok":
            torch.testing.assert_close(kwargs["text_embeddings"], expected_text)
            spec = activation._mmtok_spec_value
            projected = torch.nn.functional.normalize(expected_projected.float(), dim=-1, eps=1e-8)
            vision = torch.nn.functional.normalize(expected_raw.float(), dim=-1, eps=1e-8)
            text = torch.nn.functional.normalize(expected_text.float(), dim=-1, eps=1e-8)
            coverage = torch.cat(((text @ projected.T / spec.text_temperature).softmax(-1) / text.shape[0],
                (vision @ vision.T / spec.vision_temperature).softmax(-1) / raw.shape[0] * spec.alpha))
            covered = torch.zeros(coverage.shape[0])
            chosen = []
            for _ in range(3):
                marginal = (coverage - covered[:, None]).clamp_min(0).sum(0)
                marginal[chosen] = float("-inf")
                index = int(marginal.argmax())
                chosen.append(index)
                covered = torch.maximum(covered, coverage[:, index])
            expected_indices = torch.tensor(sorted(chosen))
        else:
            expected_proxy_input = native_merger.ln_q(dense).reshape(16, 64)[reverse]
            torch.testing.assert_close(kwargs["projector_inputs"], expected_proxy_input)
            assert tuple(kwargs["projector_layers"]) == tuple(visual.merger.mlp.children())
            oracle = zoo_prune_image_tokens(expected_projected, expected_proxy_input,
                tuple(native_merger.mlp.children()), spec=activation._zoo_prune_spec_value,
                proxy_cache=ZOOProjectorProxyCache())
            expected_indices = oracle.indices
        result = original(raw, attention, keys, **kwargs)
        torch.testing.assert_close(result.kernel_result.indices, expected_indices)
        if algorithm == "zoo_prune":
            torch.testing.assert_close(result.kernel_result.sensitivity, oracle.sensitivity)
            assert result.kernel_result.metadata["projector_proxy"] == "low_rank"
            assert result.kernel_result.metadata["low_rank_solver_used"] == "randomized+randomized"
        torch.testing.assert_close(result.kernel_result.tokens, expected_raw[expected_indices])
        captured["selected_raw"] = expected_raw[expected_indices].reshape(12, 16)
        return result

    activation.post_visual_merger = compress
    handles = [visual.merger.register_forward_pre_hook(lambda module, args: merger_inputs.append(args[0]))]
    handles.append(visual.merger.register_forward_hook(lambda module, args, output: merger_outputs.append(output)))
    try:
        activation.install()
        bridge = activation._bridge
        original_block = bridge._execute_visual_block

        def block(*args, **kwargs):
            result = original_block(*args, **kwargs)
            if kwargs["layer_index"] == len(visual.blocks) - 1:
                captured["final_visual"] = result[0]
            return result

        monkeypatch.setattr(bridge, "_execute_visual_block", block)
        output = model(input_ids=ids, attention_mask=torch.ones_like(ids), pixel_values=pixels,
            image_grid_thw=grid, labels=ids.masked_fill(ids == 10, -100), use_cache=False)
        assert [value.shape[0] for value in merger_inputs] == [64, 12]
        assert [value.requires_grad for value in merger_outputs] == [False, True]
        torch.testing.assert_close(merger_inputs[-1], captured["selected_raw"])
        torch.testing.assert_close(merger_outputs[-1], native_merger(captured["selected_raw"]))
        assert output.logits.shape == (1, 7, 64)
        output.loss.backward()
        assert pixels.grad is not None and torch.isfinite(pixels.grad).all() and pixels.grad.abs().sum() > 0
        assert all(parameter.grad is not None and torch.isfinite(parameter.grad).all()
            for parameter in visual.merger.parameters())
        metadata = activation.runtime_metadata()
        assert metadata["projector_target_sites"] == ["pre_vision_merger"]
        assert metadata["dense_projected_reference_required"] is True
        records = metadata["layout_delta_records"]
        assert len(records) == 1
        assert records[0]["source_layout"]["projected_source"] == "complete_native_merger_output"
        assert records[0]["result_layout"]["target_site"] == "pre_vision_merger"
        assert records[0]["result_layout"]["dense_projected_reference_required"] is True
    finally:
        for handle in handles:
            handle.remove()
        activation.uninstall()
