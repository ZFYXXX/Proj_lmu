"""Native image providers exercise selected ToMe boundaries without checkpoints."""

import copy
from types import SimpleNamespace

import pytest

from mllm_comp.pipeline.stage import CompressionStageConfig
from tests.fixtures import native_image as image_fixture


def _stage(backbone, position, *, r=1):
    prefix = "vision" if backbone == "llava_onevision" else "visual"
    site = f"{prefix}_block_post_attention_residual_pre_mlp" if position == "mid" else f"{prefix}_block_output"
    return CompressionStageConfig.from_dict({
        "name": "tome-position", "placement": "encoder_level", "execution_phase": "forward",
        "plugin": "tome", "algorithm": "tome", "variant": "tome.image.encoder_dense_boundary",
        "binding": f"tome.image.{backbone}", "target_modalities": ["image"],
        "operation": {"type": "merge", "targets": {"encoder_merge": {
            "target_site": site, "layers": [0] if position == "initial" else [2] if position == "before" else [1],
            **({"layer_edge": "before"} if position in {"before", "initial"} else {}),
        }}},
        "config": {"r": r, "output_r": r},
    })


@pytest.mark.parametrize("backbone,position", [
    ("qwen25_vl", p) for p in ["mid", "full", "before", "initial"]
] + [("qwen3_vl", p) for p in ["mid", "full", "before", "initial"]])
@pytest.mark.parametrize("checkpoint", [None, False, True])
def test_native_qwen_tome_boundary_forward_gradient_and_restore(backbone, position, checkpoint):
    torch = pytest.importorskip("torch")
    torch.manual_seed(24)
    model = image_fixture.qwen_model(backbone).train()
    if checkpoint is not None:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": checkpoint})
    stage = _stage(backbone, position)
    activation = image_fixture.activation(model, stage)
    activation.install()
    visual = model.model.visual
    mlp_counts = [[] for _ in visual.blocks]
    attention_counts = [[] for _ in visual.blocks]
    handles = [block.mlp.register_forward_pre_hook(
        lambda module, args, i=i: mlp_counts[i].append(args[0].shape[0])) for i, block in enumerate(visual.blocks)]
    handles += [block.attn.proj.register_forward_pre_hook(
        lambda module, args, i=i: attention_counts[i].append(args[0].shape[0])) for i, block in enumerate(visual.blocks)]
    pixels = torch.randn(16, 12, requires_grad=True)
    ids = torch.tensor([[1, 7, 10, 10, 10, 10, 8, 2]])
    labels = ids.clone().masked_fill(ids == 10, -100)
    try:
        output = model(input_ids=ids, attention_mask=torch.ones_like(ids), pixel_values=pixels,
            image_grid_thw=torch.tensor([[1, 4, 4]]), labels=labels, use_cache=False)
        assert output.logits.shape == (1, 7, 64)
        assert mlp_counts[0] == [15 if position == "initial" else 16]
        assert mlp_counts[1] == [15 if position == "mid" or (position == "initial" and backbone == "qwen3_vl") else 16]
        assert mlp_counts[2] == [16 if backbone == "qwen25_vl" else 15]
        assert attention_counts[0] == [15 if position == "initial" else 16]
        output.loss.backward()
        assert pixels.grad is not None and torch.isfinite(pixels.grad).all()
        assert pixels.grad.abs().sum() > 0
        for block in visual.blocks:
            assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in block.mlp.parameters())
        assert all(layer.mlp.gate_proj.weight.grad is not None for layer in model.model.language_model.layers)
        assert activation.runtime_metadata()["encoder_merge_position"]["target_site"] == stage.operation["targets"]["encoder_merge"]["target_site"]
    finally:
        for handle in handles:
            handle.remove()
        activation.uninstall()


@pytest.mark.parametrize("backbone", ["qwen25_vl", "qwen3_vl"])
@pytest.mark.parametrize("position", ["full", "initial"])
def test_native_qwen_zero_merge_matches_uncompressed_model(backbone, position):
    torch = pytest.importorskip("torch")
    torch.manual_seed(44)
    reference = image_fixture.qwen_model(backbone).eval()
    model = copy.deepcopy(reference)
    ids = torch.tensor([[1, 7, 10, 10, 10, 10, 8, 2]])
    args = dict(input_ids=ids, attention_mask=torch.ones_like(ids), pixel_values=torch.randn(16, 12),
        image_grid_thw=torch.tensor([[1, 4, 4]]), use_cache=False)
    activation = image_fixture.activation(model, _stage(backbone, position, r=0))
    activation.install()
    try:
        with torch.no_grad():
            torch.testing.assert_close(model(**args).logits, reference(**args).logits, atol=2e-6, rtol=2e-5)
    finally:
        activation.uninstall()


@pytest.mark.parametrize("position", ["mid", "full", "before", "initial"])
@pytest.mark.parametrize("legacy_budget", [None, "layers", "r_by_layer"])
@pytest.mark.parametrize("checkpoint", [None, False, True])
def test_native_llava_tome_boundary_forward_gradient_and_restore(position, legacy_budget, checkpoint):
    torch = pytest.importorskip("torch")
    torch.manual_seed(53)
    model = image_fixture.llava_model().train()
    if checkpoint is not None:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": checkpoint})
    stage = _stage("llava_onevision", position)
    if legacy_budget == "layers":
        stage.config["layers"] = [0]
    elif legacy_budget == "r_by_layer":
        stage.config["r_by_layer"] = {0: 1}
    activation = image_fixture.activation(model, stage)
    activation.install()
    encoder = model.get_vision_tower().vision_tower.vision_model.encoder
    # Use the existing training entry: ToMe disables stateful SigLip replay,
    # while the language model keeps its requested native checkpoint mode.
    activation._bridge.prepare_training_forward()
    counts = [[] for _ in encoder.layers]
    attention_counts = [[] for _ in encoder.layers]
    handles = [layer.mlp.register_forward_pre_hook(
        lambda module, args, i=i: counts[i].append(args[0].shape[1])) for i, layer in enumerate(encoder.layers)]
    handles += [layer.self_attn.out_proj.register_forward_pre_hook(
        lambda module, args, i=i: attention_counts[i].append(args[0].shape[1])) for i, layer in enumerate(encoder.layers)]
    pixels = torch.randn(1, 3, 54, 54, requires_grad=True)
    ids = torch.tensor([[1, -200, 2, 3]])
    try:
        output = model(input_ids=ids, attention_mask=torch.ones_like(ids), images=pixels,
            image_sizes=[(54, 54)], modalities=["image"], labels=ids.masked_fill(ids == -200, -100), use_cache=False)
        assert output.logits.shape == (1, 731, 64)
        assert counts == [[728 if position == "initial" else 729], [728 if position in {"mid", "initial"} else 729], [728]]
        assert attention_counts[0] == [728 if position == "initial" else 729]
        output.loss.backward()
        assert model.lm_head.weight.grad is not None and torch.isfinite(model.lm_head.weight.grad).all()
        assert pixels.grad is not None and pixels.grad.abs().sum() > 0
        assert torch.isfinite(pixels.grad).all()
        for layer in encoder.layers:
            assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in layer.mlp.parameters())
        assert all(layer.mlp.gate_proj.weight.grad is not None for layer in model.model.layers)
    finally:
        for handle in handles:
            handle.remove()
        activation.uninstall()


def test_native_llava_legacy_callback_and_zero_merge_match_default():
    torch = pytest.importorskip("torch")
    from mllm_comp.backbones.llava_onevision.image_compression import TOME_MERGE_SITE
    torch.manual_seed(31)
    reference = image_fixture.llava_model().eval()
    model = copy.deepcopy(reference)
    stage = _stage("llava_onevision", "mid", r=0)
    stage.operation.clear()
    activation = image_fixture.activation(model, stage)
    activation.install()
    # Ordinary callables registered through the public bridge API have no
    # optional position attribute; their default remains every layer's mid.
    callback = activation._bridge._callbacks[TOME_MERGE_SITE].callback
    del callback._mllm_position_stage
    ids = torch.tensor([[1, -200, 2, 3]])
    args = dict(input_ids=ids, attention_mask=torch.ones_like(ids), images=torch.randn(1, 3, 54, 54),
        image_sizes=[(54, 54)], modalities=["image"], use_cache=False)
    try:
        with torch.no_grad():
            torch.testing.assert_close(model(**args).logits, reference(**args).logits, atol=2e-6, rtol=2e-5)
    finally:
        activation.uninstall()


def test_native_llava_multiple_budgets_keep_legacy_layer_order():
    torch = pytest.importorskip("torch")
    torch.manual_seed(76)
    model = image_fixture.llava_model().eval()
    stage = _stage("llava_onevision", "full")
    stage.operation["targets"]["encoder_merge"]["layers"] = [0, 2]
    stage.config.update(layers=[2, 0, 2], r_by_layer={0: 1, 2: 2})
    activation = image_fixture.activation(model, stage)
    activation.install()
    try:
        with torch.no_grad():
            model(input_ids=torch.tensor([[1, -200, 2]]), images=torch.randn(1, 3, 54, 54),
                image_sizes=[(54, 54)], modalities=["image"], use_cache=False)
        records = [r for r in activation.runtime_metadata()["layout_delta_records"] if "layer_index" in r]
        assert [(r["layer_index"], r["source_count"], r["result_count"]) for r in records] == [(0, 729, 728), (2, 728, 726)]
    finally:
        activation.uninstall()


@pytest.mark.parametrize("backbone", ["qwen25_vl", "qwen3_vl"])
def test_initial_preserves_multiple_images_and_native_windows(backbone):
    torch = pytest.importorskip("torch")
    torch.manual_seed(81)
    model = image_fixture.qwen_model(backbone).train()
    if backbone == "qwen25_vl":
        model.model.visual.window_size = 4  # Four patch windows per image.
    activation = image_fixture.activation(model, _stage(backbone, "initial"))
    activation.install()
    counts = []
    handle = model.model.visual.blocks[0].attn.proj.register_forward_pre_hook(lambda module, args: counts.append(args[0].shape[0]))
    pixels = torch.randn(32, 12, requires_grad=True)
    ids = torch.tensor([[1, 7, 10, 10, 10, 10, 8, 2, 7, 10, 10, 10, 10, 8, 3]])
    try:
        output = model(input_ids=ids, attention_mask=torch.ones_like(ids), pixel_values=pixels,
            image_grid_thw=torch.tensor([[1, 4, 4], [1, 4, 4]]), labels=ids.masked_fill(ids == 10, -100), use_cache=False)
        assert output.logits.shape == (1, 13, 64)
        assert counts == [24 if backbone == "qwen25_vl" else 30]
        output.loss.backward()
        assert pixels.grad is not None and torch.isfinite(pixels.grad).all()
    finally:
        handle.remove()
        activation.uninstall()


@pytest.mark.parametrize("with_visionzip", [False, True])
@pytest.mark.parametrize("fixed_source", [False, True])
def test_single_layer_initial_actual_qk_and_context(with_visionzip, fixed_source, monkeypatch):
    torch = pytest.importorskip("torch")
    from mllm_comp.backbones.qwen25_vl import provider_bridge as provider
    torch.manual_seed(62)
    model = image_fixture.qwen_model("qwen25_vl").eval()
    visual = model.model.visual
    visual.blocks = torch.nn.ModuleList([visual.blocks[0]])
    visual.fullatt_block_indexes = [0]
    stage = _stage("qwen25_vl", "initial")
    stage.operation["targets"]["encoder_merge"]["layers"] = [0, 1]
    stage.config.update(token_sizes=list(range(1, 17)), protected_indices=[15], coordinates=list(range(16)))
    if fixed_source:
        stage.criterion["score_layer"] = 0
    activation = image_fixture.activation(model, stage)
    captured = {}
    steps = []
    original_merge = activation.encoder_dense_boundary
    def merge(hidden, keys, **kwargs):
        result = original_merge(hidden, keys, **kwargs)
        steps.append((result, kwargs))
        return result
    activation.encoder_dense_boundary = merge
    original_compact = activation.compact_visual_output
    def compact(hidden, keys, **kwargs):
        captured["output_keys"] = keys.detach().clone()
        return original_compact(hidden, keys, **kwargs)
    activation.compact_visual_output = compact
    qkv = []
    hooks = [visual.blocks[0].attn.qkv.register_forward_hook(lambda module, args, output: qkv.append(output.detach().clone()))]
    hooks += [visual.blocks[0].attn.proj.register_forward_pre_hook(lambda module, args: captured.update(actual_qkv=qkv[-1]))]
    original_metrics = provider._merged_vision_metrics
    def metrics(query, key, **kwargs):
        captured["metrics"] = (query.detach().clone(), key.detach().clone(), kwargs["key_sizes"].detach().clone())
        result = original_metrics(query, key, **kwargs)
        captured["metric_result"] = tuple([v.detach().clone() for v in values] for values in result)
        return result
    monkeypatch.setattr(provider, "_merged_vision_metrics", metrics)
    activation.install()
    visionzip = None
    if with_visionzip:
        from tests.unit.test_qwen25_vl_image_compression_provider_bridge import _activation as extra_activation
        visionzip = extra_activation(model, "visionzip.image.qwen25_vl", config={"budget": 2})
        visionzip.install()
    try:
        ids = torch.tensor([[1, 7, 10, 10, 10, 10, 8, 2]])
        grid = torch.tensor([[1, 4, 4]])
        with torch.no_grad():
            output = model(input_ids=ids, attention_mask=torch.ones_like(ids), pixel_values=torch.randn(16, 12), image_grid_thw=grid, use_cache=False)
        assert output.logits.shape[1] == (6 if with_visionzip else 7)
        assert len(steps) == 2
        first, second = steps[0][0].kernel_result, steps[1][0].kernel_result
        delta = first.layout_delta
        representatives = delta.result_layout["representative_indices"]
        assert any(group[0] != representative for group, representative in zip(first.sequence_source_groups, representatives))
        assert steps[1][1]["protected_indices"] == (delta.old_to_new[15],)
        assert steps[1][1]["coordinates"] == tuple(representatives)
        torch.testing.assert_close(steps[1][1]["token_sizes"], first.token_sizes)
        assert first.token_sizes.sum().item() == 136
        assert second.token_sizes.sum().item() == 136
        # Independent native rotary calculation at the first Attention input.
        raw = captured["actual_qkv"].reshape(15, 3, 2, 8)
        rotary = visual.rot_pos_emb(grid)
        angles = torch.cat((rotary, rotary), dim=-1)[list(representatives)].unsqueeze(1)
        def rotate(value):
            return torch.cat((-value[..., 4:], value[..., :4]), dim=-1)
        mapping = [delta.old_to_new[i] for i in range(16)]
        expected_q = (raw[:, 0] * angles.cos() + rotate(raw[:, 0]) * angles.sin())[mapping]
        expected_k = (raw[:, 1] * angles.cos() + rotate(raw[:, 1]) * angles.sin())[mapping]
        torch.testing.assert_close(captured["output_keys"], expected_k.reshape(4, 4, 2, 8).mean(1))
        if with_visionzip:
            query, key, sizes = captured["metrics"]
            torch.testing.assert_close(query, expected_q)
            torch.testing.assert_close(key, expected_k)
            torch.testing.assert_close(sizes.flatten(), torch.arange(1, 17, dtype=sizes.dtype))
            logits = torch.einsum("qhd,khd->hqk", expected_q, expected_k) * (8 ** -0.5)
            logits += torch.arange(1, 17, dtype=logits.dtype).log().reshape(1, 1, 16)
            expected_attention = logits.softmax(-1).reshape(2, 4, 4, 4, 4).mean(dim=(2, 4))
            torch.testing.assert_close(captured["metric_result"][0][0], expected_attention)
    finally:
        for hook in hooks:
            hook.remove()
        if visionzip is not None:
            visionzip.uninstall()
        activation.uninstall()


def test_native_llava_initial_keeps_anyres_crops_isolated():
    torch = pytest.importorskip("torch")
    torch.manual_seed(79)
    model = image_fixture.llava_model().train()
    model.config.tokenizer_model_max_length = 4096
    activation = image_fixture.activation(model, _stage("llava_onevision", "initial"))
    activation.install()
    pixels = torch.randn(3, 3, 54, 54, requires_grad=True)
    ids = torch.tensor([[1, -200, 2]])
    try:
        output = model(input_ids=ids, attention_mask=torch.ones_like(ids), images=[pixels],
            image_sizes=[(108, 54)], modalities=["image"], labels=ids.masked_fill(ids == -200, -100), use_cache=False)
        assert output.logits.shape == (1, 2188, 64)
        records = [r for r in activation.runtime_metadata()["layout_delta_records"] if "application_position" in r]
        assert [(r["stream_index"], r["source_count"], r["result_count"]) for r in records] == [(i, 729, 728) for i in range(3)]
        assert all(r["application_position"] == {"boundary": "inter_layer", "index": 0} for r in records)
        output.loss.backward()
        assert pixels.grad is not None and torch.isfinite(pixels.grad).all()
    finally:
        activation.uninstall()


@pytest.mark.parametrize("backbone", ["llava_onevision", "qwen25_vl", "qwen3_vl"])
@pytest.mark.parametrize("source,boundary", [(0, "mid"), (1, "mid"), (0, "full"), (1, "full"), (0, "initial"), (1, "entry")])
def test_fixed_encoder_score_source_follows_actual_merge_layout(backbone, source, boundary):
    torch = pytest.importorskip("torch")
    import copy
    torch.manual_seed(90 + source)
    base = image_fixture.llava_model() if backbone == "llava_onevision" else image_fixture.qwen_model(backbone)
    reference_output = reference_pixels = reference_parameter = None
    for checkpoint in (None, False, True):
        model = copy.deepcopy(base).train()
        if checkpoint is not None:
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": checkpoint})
        stage = _stage(backbone, boundary)
        stage.operation["targets"]["encoder_merge"]["layers"] = [source, 2]
        if boundary == "entry":
            stage.operation["targets"]["encoder_merge"].update(layers=[1, 3], layer_edge="before")
        stage.criterion["score_layer"] = source
        activation = image_fixture.activation(model, stage)
        activation.install()
        if backbone == "llava_onevision":
            activation._bridge.prepare_training_forward()
            layers = model.get_vision_tower().vision_tower.vision_model.encoder.layers
            projection = layers[source].self_attn.k_proj
            ids = torch.tensor([[1, -200, 2, 3]])
            pixels = torch.randn(1, 3, 54, 54, generator=torch.Generator().manual_seed(84)).requires_grad_()
            args = dict(input_ids=ids, images=pixels, image_sizes=[(54, 54)], modalities=["image"], labels=ids.masked_fill(ids == -200, -100))
            parameter = layers[2].mlp.fc1.weight
        else:
            visual = model.model.visual
            layers = visual.blocks
            projection = layers[source].attn.qkv
            ids = torch.tensor([[1, 7, 10, 10, 10, 10, 8, 2]])
            pixels = torch.randn(16, 12, generator=torch.Generator().manual_seed(84)).requires_grad_()
            args = dict(input_ids=ids, pixel_values=pixels, image_grid_thw=torch.tensor([[1, 4, 4]]), labels=ids.masked_fill(ids == 10, -100))
            parameter = layers[2].mlp.gate_proj.weight if backbone == "qwen25_vl" else layers[2].mlp.linear_fc1.weight
        raw = []
        hook = projection.register_forward_hook(lambda module, args, output: raw.append(output.detach().clone()))
        expected = None
        mass = None
        calls = []
        def check(keys, run):
            nonlocal expected, mass
            if expected is None:
                if backbone == "llava_onevision":
                    expected = raw[0][0].reshape(729, 2, 8)
                else:
                    raw_keys = raw[0].reshape(16, 3, 2, 8)[:, 1]
                    angles = visual.rot_pos_emb(args["image_grid_thw"])
                    angles = torch.cat((angles, angles), -1).unsqueeze(1)
                    rotated_half = torch.cat((-raw_keys[..., 4:], raw_keys[..., :4]), -1)
                    expected = raw_keys * angles.cos() + rotated_half * angles.sin()
                mass = expected.new_ones((expected.shape[0],))
            torch.testing.assert_close(keys.detach(), expected)
            result = run()
            kernel = result.kernel_result if hasattr(result, "kernel_result") else result
            delta = kernel.layout_deltas[0]
            calls.append(delta)
            if backbone != "qwen25_vl":
                # Independently carry the original key values and cluster masses.
                groups = [[i for i in range(delta.source_count) if delta.old_to_new[i] == j] for j in range(delta.result_count)]
                expected = torch.stack([(expected[g] * mass[g, None, None]).sum(0) / mass[g].sum() for g in groups])
                mass = torch.stack([mass[g].sum() for g in groups])
            return result
        if backbone == "llava_onevision":
            original = activation.merge_encoder_tokens
            # The provider registered a bound callback at install time.
            from mllm_comp.backbones.llava_onevision.image_compression import TOME_MERGE_SITE
            registered = activation._bridge._callbacks[TOME_MERGE_SITE]
            def callback(request):
                return check(request.keys[0], lambda: original(request))
            # Keep the stage metadata used to resolve source/application positions.
            callback._mllm_position_stage = stage
            from dataclasses import replace
            activation._bridge._callbacks[TOME_MERGE_SITE] = replace(registered, callback=callback)
        elif backbone == "qwen25_vl":
            original = activation.encoder_dense_boundary
            def callback(hidden, keys, **kwargs):
                return check(keys, lambda: original(hidden, keys, **kwargs))
            activation.encoder_dense_boundary = callback
        else:
            original = activation.merge_encoder_tokens
            def callback(hidden, keys, **kwargs):
                return check(keys, lambda: original(hidden, keys, **kwargs))
            activation.merge_encoder_tokens = callback
        try:
            output = model(**args, attention_mask=torch.ones_like(ids), use_cache=False)
            output.loss.backward()
            assert len(calls) == 2
            records = [r for r in activation.runtime_metadata()["layout_delta_records"] if "score_layer" in r]
            assert len(records) == 2 and all(r["score_layer"] == source for r in records)
            assert pixels.grad is not None and parameter.grad is not None
            if checkpoint is None:
                reference_output, reference_pixels, reference_parameter = output.logits.detach(), pixels.grad.detach(), parameter.grad.detach()
            else:
                torch.testing.assert_close(output.logits, reference_output)
                torch.testing.assert_close(pixels.grad, reference_pixels)
                torch.testing.assert_close(parameter.grad, reference_parameter)
        finally:
            hook.remove()
            activation.uninstall()


@pytest.mark.parametrize("boundary", ["mid", "full"])
def test_fixed_score_reference_after_another_controller_with_list_masses(boundary):
    torch = pytest.importorskip("torch")
    torch.manual_seed(92)
    model = image_fixture.qwen_model("qwen25_vl").train()
    visual = model.model.visual
    stages = [_stage("qwen25_vl", boundary), _stage("qwen25_vl", boundary)]
    stages[0].config["token_sizes"] = list(range(1, 17))
    stages[1].criterion["score_layer"] = 0
    first, second = [image_fixture.activation(model, stage) for stage in stages]
    seen = {}
    qkv = []
    hook = visual.blocks[0].attn.qkv.register_forward_hook(lambda module, args, result: qkv.append(result.detach().clone()))
    original_first = first.encoder_dense_boundary
    def merge_first(hidden, keys, **kwargs):
        result = original_first(hidden, keys, **kwargs)
        seen["delta"] = result.kernel_result.layout_delta
        return result
    first.encoder_dense_boundary = merge_first
    original_second = second.encoder_dense_boundary
    def merge_second(hidden, keys, **kwargs):
        raw = qkv[0].reshape(16, 3, 2, 8)[:, 1]
        angles = visual.rot_pos_emb(torch.tensor([[1, 4, 4]]))
        angles = torch.cat((angles, angles), -1).unsqueeze(1)
        reference = raw * angles.cos() + torch.cat((-raw[..., 4:], raw[..., :4]), -1) * angles.sin()
        delta = seen["delta"]
        masses = torch.arange(1, 17, dtype=reference.dtype)
        groups = [[i for i in range(16) if delta.old_to_new[i] == j] for j in range(15)]
        expected = torch.stack([(reference[g] * masses[g, None, None]).sum(0) / masses[g].sum() for g in groups])
        torch.testing.assert_close(keys, expected)
        seen["checked"] = True
        return original_second(hidden, keys, **kwargs)
    second.encoder_dense_boundary = merge_second
    first.install()
    second.install()
    try:
        ids = torch.tensor([[1, 7, 10, 10, 10, 10, 8, 2]])
        pixels = torch.randn(16, 12, requires_grad=True)
        output = model(input_ids=ids, attention_mask=torch.ones_like(ids), pixel_values=pixels,
            image_grid_thw=torch.tensor([[1, 4, 4]]), labels=ids.masked_fill(ids == 10, -100), use_cache=False)
        assert seen["checked"] and output.logits.shape == (1, 6, 64)
        output.loss.backward()
        assert pixels.grad is not None and torch.isfinite(pixels.grad).all()
    finally:
        hook.remove()
        second.uninstall()
        first.uninstall()


@pytest.mark.parametrize("budget", ["layers", "r_by_layer"])
def test_fixed_llava_score_source_ignores_legacy_identity_layers(budget):
    torch = pytest.importorskip("torch")
    torch.manual_seed(94)
    model = image_fixture.llava_model().train()
    stage = _stage("llava_onevision", "mid")
    stage.operation.clear()
    stage.criterion["score_layer"] = 1
    stage.config[budget] = [1, 2] if budget == "layers" else {1: 1, 2: 1}
    activation = image_fixture.activation(model, stage)
    activation.install()
    try:
        ids = torch.tensor([[1, -200, 2, 3]])
        pixels = torch.randn(1, 3, 54, 54, requires_grad=True)
        output = model(input_ids=ids, attention_mask=torch.ones_like(ids), images=pixels,
            image_sizes=[(54, 54)], modalities=["image"], labels=ids.masked_fill(ids == -200, -100), use_cache=False)
        records = [r for r in activation.runtime_metadata()["layout_delta_records"] if "score_layer" in r]
        assert [(r["layer_index"], r["source_count"], r["result_count"]) for r in records] == [(0, 729, 729), (1, 729, 728), (2, 728, 727)]
        output.loss.backward()
        assert pixels.grad is not None and torch.isfinite(pixels.grad).all()
    finally:
        activation.uninstall()


def test_next_input_score_at_full_boundary_uses_native_initial_restore_layout():
    torch = pytest.importorskip("torch")
    torch.manual_seed(95)
    model = image_fixture.qwen_model("qwen25_vl").train()
    visual = model.model.visual
    first_stage = _stage("qwen25_vl", "initial")
    first_stage.config.update(token_sizes=list(range(1, 17)), output_r=0)
    second_stage = _stage("qwen25_vl", "before")
    second_stage.operation["targets"]["encoder_merge"]["layers"] = [1]
    second_stage.criterion["score_layer"] = 1
    first, second = image_fixture.activation(model, first_stage), image_fixture.activation(model, second_stage)
    raw = []
    hook = visual.blocks[1].attn.qkv.register_forward_hook(lambda module, args, value: raw.append(value.detach().clone()))
    seen = {}
    original_first = first.encoder_dense_boundary
    def merge_first(hidden, keys, **kwargs):
        result = original_first(hidden, keys, **kwargs)
        seen["delta"] = result.kernel_result.layout_delta
        return result
    first.encoder_dense_boundary = merge_first
    original_second = second.encoder_dense_boundary
    def merge_second(hidden, keys, **kwargs):
        # The next native layer is evaluated on the dense restored layout;
        # map those rotated input keys back to the still-compact current block.
        source = raw[0].reshape(16, 3, 2, 8)[:, 1]
        angles = visual.rot_pos_emb(torch.tensor([[1, 4, 4]]))
        angles = torch.cat((angles, angles), -1).unsqueeze(1)
        source = source * angles.cos() + torch.cat((-source[..., 4:], source[..., :4]), -1) * angles.sin()
        delta = seen["delta"]
        weights = torch.arange(1, 17, dtype=source.dtype)
        groups = [[i for i in range(16) if delta.old_to_new[i] == j] for j in range(15)]
        expected = torch.stack([(source[g] * weights[g, None, None]).sum(0) / weights[g].sum() for g in groups])
        torch.testing.assert_close(keys, expected)
        seen["checked"] = True
        return original_second(hidden, keys, **kwargs)
    second.encoder_dense_boundary = merge_second
    first.install()
    second.install()
    try:
        ids = torch.tensor([[1, 7, 10, 10, 10, 10, 8, 2]])
        pixels = torch.randn(16, 12, requires_grad=True)
        output = model(input_ids=ids, attention_mask=torch.ones_like(ids), pixel_values=pixels,
            image_grid_thw=torch.tensor([[1, 4, 4]]), labels=ids.masked_fill(ids == 10, -100), use_cache=False)
        assert seen["checked"] and output.logits.shape == (1, 7, 64)
        output.loss.backward()
        assert pixels.grad is not None and torch.isfinite(pixels.grad).all()
    finally:
        hook.remove()
        second.uninstall()
        first.uninstall()


@pytest.mark.parametrize("allowed", [False, True])
def test_llava_fixed_source_uses_final_binding_budget(allowed):
    torch = pytest.importorskip("torch")
    torch.manual_seed(96)
    model = image_fixture.llava_model().eval()
    stage = _stage("llava_onevision", "mid", r=1 if allowed else 0)
    stage.operation.clear()
    stage.criterion["score_layer"] = 1
    selection = SimpleNamespace(config={"layers": [1, 2]} if allowed else {"r": 1})
    activation = image_fixture.activation(model, stage, binding_config=selection)
    activation.install()
    try:
        args = dict(input_ids=torch.tensor([[1, -200, 2, 3]]), images=torch.randn(1, 3, 54, 54), image_sizes=[(54, 54)], modalities=["image"], use_cache=False)
        if allowed:
            with torch.no_grad():
                output = model(**args)
            assert output.logits.shape == (1, 731, 64)
            records = [r for r in activation.runtime_metadata()["layout_delta_records"] if "score_layer" in r]
            assert [(r["layer_index"], r["source_count"], r["result_count"]) for r in records] == [(0, 729, 729), (1, 729, 728), (2, 728, 727)]
        else:
            with pytest.raises(ValueError, match="future layer"):
                model(**args)
            assert not activation.runtime_metadata()["layout_delta_records"]
    finally:
        activation.uninstall()
