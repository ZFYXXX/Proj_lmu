"""Compound projector reducers aggregate spatial raw groups with original cues."""

import copy
import math

import pytest

from mllm_comp.pipeline.stage import CompressionStageConfig
from tests.fixtures import native_image as image_fixture
from tests.integration.test_image_projector_positions_cpu import _model


def _stage(algorithm):
    target = {"nuwa": "stage1_regional_visual_aggregation",
        "restore": "vispruner_pristine_plus_distinctive_anchor_merge"}[algorithm]
    config = ({"stage1_total_tokens": 4, "stage2_after_layer": 2,
        "stage2_retain_ratio": 0.5, "semantic_hidden_start": 1, "semantic_hidden_end": 3}
        if algorithm == "nuwa" else {"target_count": 4})
    return CompressionStageConfig.from_dict({
        "name": f"{algorithm}-raw-groups", "placement": "projector_level", "execution_phase": "forward",
        "plugin": algorithm, "algorithm": algorithm, "variant": f"{algorithm}.image",
        "binding": f"{algorithm}.image.qwen25_vl", "target_modalities": ["image"], "config": config,
        "operation": {"targets": {target: {"target_site": "projector_input"}}},
    })


@pytest.mark.parametrize("algorithm", ["nuwa", "restore"])
def test_compound_pre_projector_nonidentity_window_source_math_and_raw_gradient(monkeypatch, algorithm):
    torch = pytest.importorskip("torch")
    from mllm_comp.compressors.image.restore import restore_image_tokens

    torch.manual_seed(411)
    model = _model("qwen25_vl").train()
    visual = model.model.visual
    native_merger = copy.deepcopy(visual.merger)
    activation = image_fixture.activation(model, _stage(algorithm))
    grid = torch.tensor([[1, 8, 8]])
    window, _ = visual.get_window_index(grid)
    reverse = window.argsort()
    assert not torch.equal(window, torch.arange(16))
    ids = torch.tensor([[1, 7, *([10] * 16), 8, 2]])
    pixels = torch.randn(64, 12, requires_grad=True)
    captured, blocks, merger_inputs, merger_outputs = {}, {}, [], []
    original = activation.post_visual_merger

    def compress(raw, attention, keys, **kwargs):
        expected_raw = blocks[2].reshape(16, 64)[reverse]
        torch.testing.assert_close(raw, expected_raw)
        assert kwargs["projected_reference"] is None
        if algorithm == "nuwa":
            # Semantic IDs 1 and 2 denote the first two completed visual blocks.
            semantic = ((blocks[0].float() + blocks[1].float()) / 2).reshape(16, 64)[reverse]
            torch.testing.assert_close(kwargs["vision_reference"], semantic)
            assert not torch.allclose(semantic, expected_raw)
            assert kwargs["feature_grid"] == (4, 4)
            spec = activation._nuwa_spec_value
            norms = keys.detach().float().permute(1, 0, 2).reshape(16, -1).norm(dim=-1)
            salience = attention.detach().float().mean((0, 1)) * norms
            candidates = []
            for row in range(0, 4, spec.region_size):
                for col in range(0, 4, spec.region_size):
                    local = torch.tensor([r * 4 + c
                        for r in range(row, min(row + spec.region_size, 4))
                        for c in range(col, min(col + spec.region_size, 4))])
                    candidates.extend(local[salience[local].argsort(descending=True, stable=True)
                        [:spec.top_n_per_region]].tolist())
            candidates = torch.tensor(sorted(set(candidates)))
            selected = candidates[salience[candidates].argsort(descending=True, stable=True)[:4]].sort().values
            normalized = torch.nn.functional.normalize(semantic.detach().float(), dim=-1, eps=1e-8)
            similarity = (normalized[selected] @ normalized.T).relu()
            coords = torch.tensor([(r, c) for r in range(4) for c in range(4)]).float()
            proximity = (1 - torch.cdist(coords[selected], coords)
                / (math.sqrt(18) * math.sqrt(spec.distance_squared_ratio))).clamp(0, 1)
            weights = similarity * proximity
            weights[norms[selected] >= torch.quantile(norms[selected], spec.pillar_quantile)] = 0
            weights = weights / weights.sum(-1, keepdim=True).clamp_min(1e-8)
            weights[torch.arange(4), selected] = 1
            expected_compact = weights.to(raw.dtype) @ expected_raw
        else:
            torch.testing.assert_close(kwargs["vision_reference"], expected_raw)
            # The original projected-value call must choose the same pristine
            # tokens, anchor partition and representative coordinates.
            oracle = restore_image_tokens(native_merger(blocks[2])[reverse], expected_raw,
                attention, spec=activation._restore_spec_value)
            membership = oracle.membership
            expected_compact = torch.stack([expected_raw[members].mean(0) for members in membership])
        result = original(raw, attention, keys, **kwargs)
        kernel = result.kernel_result
        if algorithm == "nuwa":
            torch.testing.assert_close(kernel.selected_indices, selected)
            torch.testing.assert_close(kernel.salience, salience)
            torch.testing.assert_close(kernel.aggregation_weights, weights)
        else:
            torch.testing.assert_close(kernel.membership, membership)
            torch.testing.assert_close(kernel.representative_indices, oracle.representative_indices)
            torch.testing.assert_close(kernel.token_sizes, membership.sum(1).float())
            assert bool((membership.sum(0) == 1).all())
            assert kernel.token_sizes.sum().item() == 16
            captured["restore_delta"] = result.prompt_repair.layout_delta
        torch.testing.assert_close(kernel.tokens, expected_compact)
        captured["compact"] = expected_compact.reshape(16, 16)
        return result

    activation.post_visual_merger = compress
    handles = [visual.merger.register_forward_pre_hook(lambda module, args: merger_inputs.append(args[0])),
        visual.merger.register_forward_hook(lambda module, args, output: merger_outputs.append(output))]
    try:
        activation.install()
        bridge = activation._bridge
        original_block = bridge._execute_visual_block

        def block(*args, **kwargs):
            result = original_block(*args, **kwargs)
            blocks[kwargs["layer_index"]] = result[0]
            return result

        monkeypatch.setattr(bridge, "_execute_visual_block", block)
        output = model(input_ids=ids, attention_mask=torch.ones_like(ids), pixel_values=pixels,
            image_grid_thw=grid, labels=ids.masked_fill(ids == 10, -100), use_cache=False)
        assert len(merger_inputs) == 1
        torch.testing.assert_close(merger_inputs[0], captured["compact"])
        torch.testing.assert_close(merger_outputs[0], native_merger(captured["compact"]))
        assert output.logits.shape == (1, 6 if algorithm == "nuwa" else 8, 64)
        output.loss.backward()
        assert pixels.grad is not None and torch.isfinite(pixels.grad).all() and pixels.grad.abs().sum() > 0
        assert all(parameter.grad is not None and torch.isfinite(parameter.grad).all()
            for parameter in visual.merger.parameters())
        metadata = activation.runtime_metadata()
        assert metadata["projector_target_sites"] == ["pre_vision_merger"]
        records = metadata["layout_delta_records"]
        assert records[0]["result_layout"]["target_site"] == "pre_vision_merger"
        if algorithm == "nuwa":
            assert metadata["target_sites"] == ["pre_vision_merger", metadata["decoder_target_site"]]
            assert records[-1]["result_layout"]["target_site"] == metadata["decoder_target_site"]
        else:
            assert metadata["target_sites"] == ["pre_vision_merger", "llm_decoder_attention_calibration"]
            calibration = bridge.runtime_metadata()["latest_restore_calibration"]
            assert calibration["visual_source_tokens"] == 16
            assert calibration["layers_calibrated"] == 4
            assert calibration["key_tokens"] == 8
            assert tuple(records[0]["result_layout"]["representative_indices"]) == tuple(
                captured["restore_delta"].result_layout["representative_indices"])
    finally:
        for handle in handles:
            handle.remove()
        activation.uninstall()
