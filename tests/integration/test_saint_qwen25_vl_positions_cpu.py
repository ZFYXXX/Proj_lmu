"""SAINT applies configured key modules to current values at real visual boundaries."""

import copy
from dataclasses import replace

import pytest

from mllm_comp.pipeline.stage import CompressionStageConfig
from tests.fixtures import native_image as image_fixture


def _stage(boundaries=(0, 1), *, mid=False, sources=(1, 2)):
    return CompressionStageConfig.from_dict({
        "name": "saint-encoder-positions", "placement": "encoder_level", "execution_phase": "forward",
        "plugin": "saint", "algorithm": "saint", "variant": "saint.image.encoder_progressive_prune",
        "binding": "saint.image.qwen25_vl", "target_modalities": ["image"],
        "config": {"layer_indices": list(sources), "keep_ratios": [0.5, 0.5], "group_size": 4},
        "operation": {"targets": {"encoder_progressive_prune": {
            "target_site": "visual_block_post_attention_residual_pre_mlp" if mid else "visual_block_output_progressive_patch_prune",
            "layers": list(boundaries), **({} if mid else {"layer_edge": "before"})}}},
    })


def _inputs(torch):
    ids = torch.tensor([[1, 7, *([10] * 16), 8, 2]])
    return dict(input_ids=ids, attention_mask=torch.ones_like(ids),
        pixel_values=torch.randn(64, 12, requires_grad=True), image_grid_thw=torch.tensor([[1, 8, 8]]))


@pytest.mark.parametrize("boundaries,mid,checkpoint", [((0, 1), False, None), ((0, 2), True, None),
    ((0, 2), True, False), ((0, 2), True, True), ((1, 3), False, None)])
def test_saint_actual_boundary_keys_rotary_windows_and_gradients(boundaries, mid, checkpoint):
    torch = pytest.importorskip("torch")
    from mllm_comp.compressors.image.saint import saint_image_tokens

    torch.manual_seed(391)
    model = image_fixture.qwen_model("qwen25_vl").train()
    visual = model.model.visual
    # Source module 2 is a window-attention block, and may score before it runs.
    assert 2 not in visual.fullatt_block_indexes
    if checkpoint is not None:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": checkpoint})
    stage = _stage(boundaries, mid=mid)
    activation = image_fixture.activation(model, stage)
    raw = _inputs(torch)
    rotary = visual.rot_pos_emb(raw["image_grid_thw"])
    angles = torch.cat((rotary, rotary), dim=-1)
    tracked = torch.arange(64)
    cache, attention_counts, mlp_counts = {}, [], []
    steps = []
    handles = []

    def attention_input(module, args, kwargs, layer):
        length = int(args[0].shape[0])
        attention_counts.append(length)
        cos, sin = kwargs["position_embeddings"]
        torch.testing.assert_close(cos, angles.cos().index_select(0, tracked))
        torch.testing.assert_close(sin, angles.sin().index_select(0, tracked))
        expected_cu = [0, length] if layer in visual.fullatt_block_indexes else [*range(0, length, 16), length]
        assert kwargs["cu_seqlens"].tolist() == expected_cu

    def attention_output(module, args, output, layer):
        cache[("attention_residual", layer)] = cache[("layer_input", layer)] + output

    def mlp_output(module, args, output, layer):
        cache[("layer_output", layer)] = cache[("mlp_residual", layer)] + output

    handles.append(visual.patch_embed.register_forward_hook(lambda module, args, output: cache.update(initial=output)))
    for layer_index, block in enumerate(visual.blocks):
        handles.append(block.norm1.register_forward_pre_hook(
            lambda module, args, i=layer_index: cache.update({("layer_input", i): args[0]})))
        handles.append(block.attn.register_forward_pre_hook(
            lambda module, args, kwargs, i=layer_index: attention_input(module, args, kwargs, i), with_kwargs=True))
        handles.append(block.attn.register_forward_hook(
            lambda module, args, output, i=layer_index: attention_output(module, args, output, i)))
        handles.append(block.norm2.register_forward_pre_hook(
            lambda module, args, i=layer_index: cache.update({("mlp_residual", i): args[0]})))
        handles.append(block.mlp.register_forward_pre_hook(lambda module, args: mlp_counts.append(int(args[0].shape[0]))))
        handles.append(block.mlp.register_forward_hook(
            lambda module, args, output, i=layer_index: mlp_output(module, args, output, i)))
    original = activation.encoder_progressive_prune

    def prune(values, keys, *, boundary_layer, image_index):
        nonlocal tracked
        stage_index = len(steps)
        point = activation.application_positions[stage_index]
        expected_values = (cache[("attention_residual", point.index)] if mid else
            cache["initial"] if point.index == 0 else cache[("layer_output", point.index - 1)])
        torch.testing.assert_close(values, expected_values)
        metric_module = visual.blocks[boundary_layer]
        # Independent native norm/linear projection on the values at this
        # boundary. No RoPE or actual Attention history belongs in this score.
        qkv = torch.nn.functional.linear(metric_module.norm1(values), metric_module.attn.qkv.weight,
            metric_module.attn.qkv.bias).reshape(values.shape[0], 3, metric_module.attn.num_heads, -1)
        expected_keys = qkv[:, 1].reshape(values.shape[0], -1)
        torch.testing.assert_close(keys, expected_keys)
        oracle = saint_image_tokens(values, expected_keys,
            spec=replace(activation._saint_spec_value, keep_ratio=activation._saint_keep_ratios_value[stage_index]))
        result = original(values, keys, boundary_layer=boundary_layer, image_index=image_index)
        torch.testing.assert_close(result.tokens, oracle.tokens)
        torch.testing.assert_close(result.selected_indices, oracle.selected_indices)
        tracked = tracked.index_select(0, result.selected_indices)
        steps.append(result)
        return result

    activation.encoder_progressive_prune = prune
    merger_inputs = []
    handles.append(visual.merger.register_forward_pre_hook(lambda module, args: merger_inputs.append(args[0].shape[0])))
    try:
        activation.install()
        labels = raw["input_ids"].masked_fill(raw["input_ids"] == 10, -100)
        output = model(**raw, labels=labels, use_cache=False)
        sizes = [64, 32, 16]
        assert attention_counts == [sizes[sum(point + int(mid) <= layer for point in boundaries)] for layer in range(3)]
        assert mlp_counts == [sizes[sum(point <= layer for point in boundaries)] for layer in range(3)]
        assert merger_inputs == [16]
        assert output.logits.shape == (1, 8, 64)
        output.loss.backward()
        assert raw["pixel_values"].grad is not None and torch.isfinite(raw["pixel_values"].grad).all()
        assert raw["pixel_values"].grad.abs().sum() > 0
        assert all(parameter.grad is not None and torch.isfinite(parameter.grad).all()
            for block in visual.blocks for parameter in block.mlp.parameters())
        metadata = activation.runtime_metadata()
        assert metadata["key_projection_layers"] == [1, 2]
        assert metadata["visual_execution"] == "manual_attention_and_mlp_without_block_checkpoint"
        records = metadata["layout_delta_records"]
        assert len(records) == 2  # Decoder checkpoint options do not rerun visual pruning.
        assert [record["boundary_layer"] for record in records] == [1, 2]
        assert [record["key_projection_layer"] for record in records] == [1, 2]
        assert [record["application_position"] for record in records] == [point.as_dict() for point in activation.application_positions]
        assert all(record["source_layout"].get("after_visual_layer", 0) >= 0 for record in records)
    finally:
        for handle in handles:
            handle.remove()
        activation.uninstall()


def test_saint_default_and_explicit_output_alias_preserve_values_and_gradients():
    torch = pytest.importorskip("torch")
    torch.manual_seed(392)
    base = image_fixture.qwen_model("qwen25_vl").train()
    base.model.visual.fullatt_block_indexes = [1, 2]
    expected = None
    for operation in ({}, {"target_site": "visual_block_output", "layers": [1, 2]},
            {"target_site": "visual_block_output_progressive_patch_prune", "layers": [2, 3], "layer_edge": "before"}):
        model = copy.deepcopy(base)
        stage = _stage()
        stage.operation = operation
        activation = image_fixture.activation(model, stage)
        try:
            activation.install()
            torch.manual_seed(393)
            raw = _inputs(torch)
            result = model(**raw, labels=raw["input_ids"].masked_fill(raw["input_ids"] == 10, -100), use_cache=False)
            result.loss.backward()
            observed = result.logits.detach(), raw["pixel_values"].grad
            if expected is None:
                expected = observed
            else:
                torch.testing.assert_close(observed, expected, rtol=0, atol=0)
            assert [point.index for point in activation.application_positions] == [2, 3]
        finally:
            activation.uninstall()


@pytest.mark.parametrize("invalid", ["key_module", "mid_depth"])
def test_saint_public_factory_rejects_out_of_depth_modules_and_positions(invalid):
    from mllm_comp.core.errors import BackendIntegrationError

    model = image_fixture.qwen_model("qwen25_vl")
    stage = _stage((0, 3), mid=invalid == "mid_depth", sources=(1, 3) if invalid == "key_module" else (1, 2))
    activation = None
    try:
        with pytest.raises(BackendIntegrationError):
            activation = image_fixture.activation(model, stage)
            activation.install()
    finally:
        if activation is not None:
            activation.uninstall()
