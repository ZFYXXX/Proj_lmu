"""DyCoKe keeps TTM values and physical decode cache distinct from sparse views."""

import copy
import gc
import weakref
from dataclasses import replace

import pytest

from mllm_comp.pipeline.stage import CompressionStageConfig

from tests.fixtures import native_image as image_fixture
from tests.integration.test_video_vqtoken_llava_onevision_positions_cpu import _video_model


_BACKBONES = ("llava_onevision", "qwen25_vl", "qwen3_vl")
_LEGACY_KV_SITES = {"llava_onevision": "llm_decoder_video_tokens_prefill_or_decode",
    "qwen25_vl": "llm_decoder_video_tokens_sparse_mrope",
    "qwen3_vl": "llm_decoder_video_tokens_sparse_mrope_visual_pos_masks"}


def _model(backbone):
    source = _video_model() if backbone == "llava_onevision" else image_fixture.qwen_model(backbone)
    config = copy.deepcopy(source.config)
    text = config if backbone == "llava_onevision" else config.text_config
    text.num_hidden_layers = 5
    if hasattr(text, "layer_types"):
        text.layer_types = ["full_attention"] * 5
    model = type(source)(config)
    if backbone == "llava_onevision":
        model.model.vision_tower = source.model.vision_tower
        model.model.mm_projector = source.model.mm_projector
        model.model.image_newline = source.model.image_newline
    return model


def _inputs(torch, backbone):
    if backbone == "llava_onevision":
        return dict(input_ids=torch.tensor([[1, -200, 2, 3]]),
            images=[torch.randn(4, 3, 54, 54, requires_grad=True)], image_sizes=[(54, 54)], modalities=["video"])
    tokens = [1, *([7, *([11] * 16), 8] * 4), 2] if backbone == "qwen3_vl" else [1, 7, *([11] * 64), 8, 2]
    return dict(input_ids=torch.tensor([tokens]), attention_mask=torch.ones(1, len(tokens), dtype=torch.long),
        pixel_values_videos=torch.randn(256, 12, requires_grad=True),
        video_grid_thw=torch.tensor([[4, 8, 8]]),
        **({"second_per_grid_ts": torch.tensor([0.5])} if backbone == "qwen25_vl" else {}))


def _stage(backbone, position=None, sparse_layer=None, *, phase="forward", legacy=False):
    targets = {}
    if position is not None:
        targets["ttm_visual_merge"] = {"target_site": position}
    if sparse_layer is not None:
        targets["decode_kv_view_prune"] = {"layers": [sparse_layer], "target_site":
            _LEGACY_KV_SITES[backbone] if legacy else "decode_kv_before_attention"}
    return CompressionStageConfig.from_dict({
        "name": "dycoke-position", "placement": "projector_level", "execution_phase": phase,
        "plugin": "dycoke", "algorithm": "dycoke", "variant": "dycoke.video",
        "binding": f"dycoke.video.{backbone}", "target_modalities": ["video"],
        "operation": {"targets": targets}, "config": {},
    })


def _pool(torch, values):
    value = values.reshape(4, 27, 27, -1).permute(0, 3, 1, 2)
    return torch.nn.functional.interpolate(value, size=(14, 14), mode="bilinear").permute(0, 2, 3, 1).flatten(0, 2)


def _ttm_indices(torch, source):
    frames = source.reshape(4, -1, source.shape[-1])
    count = frames.shape[1]
    keep = max(1, int((1.0 - 0.7) * count))
    selected = list(range(count))
    for index, reference in ((1, 0), (2, 0), (3, 2)):
        scores = torch.nn.functional.cosine_similarity(frames[reference].float(), frames[index].float(), dim=-1)
        selected.extend((scores.argsort(stable=True)[:keep].sort().values + index * count).tolist())
    return selected


@pytest.mark.parametrize("backbone", _BACKBONES)
def test_native_ttm_pre_values_default_alias_gradient_and_branch_replay(backbone):
    torch = pytest.importorskip("torch")
    from mllm_comp.compressors.video.positions import video_projector_site
    torch.manual_seed(461)
    base = _model(backbone).train()
    baseline = None
    legacy_site = video_projector_site(backbone, "after")
    for position in (None, legacy_site, "projector_input"):
        model = copy.deepcopy(base)
        before = position == "projector_input"
        stage = _stage(backbone, position, 3 if position == legacy_site else None, legacy=True)
        activation = image_fixture.activation(model, stage)
        # Canonical historical evidence must only apply to the original sites.
        activation.binding = replace(activation.binding, metadata={**activation.binding.metadata, "verified_phases": ["forward"]})
        observed = []
        callback = activation._projector_callback

        def capture(request):
            response = callback(request)
            observed.append((request, response))
            return response

        activation._projector_callback = capture
        activation.install()
        visual = model.model if backbone == "llava_onevision" else model.model.visual
        mergers = [visual.mm_projector] if backbone == "llava_onevision" else [visual.merger]
        if backbone == "qwen3_vl":
            mergers += list(visual.deepstack_merger_list)
        inputs, outputs = [[] for _ in mergers], [[] for _ in mergers]
        hooks = []
        for index, merger in enumerate(mergers):
            hooks += [merger.register_forward_pre_hook(lambda module, args, i=index: inputs[i].append(args[0])),
                merger.register_forward_hook(lambda module, args, output, i=index: outputs[i].append(output))]
        torch.manual_seed(462)
        args = _inputs(torch, backbone)
        args["attention_mask"] = torch.ones_like(args["input_ids"])
        labels = torch.full_like(args["input_ids"], -100)
        labels[0, -1] = args["input_ids"][0, -1]
        try:
            output = model(**args, labels=labels, use_cache=False)
            assert len(observed) == 1
            request, response = observed[0]
            result = response.compression_result if backbone == "qwen3_vl" else response
            if backbone == "llava_onevision":
                source = _pool(torch, inputs[0][0] if before else outputs[0][0])
                assert len(request.packing.structural_positions) == 56
                splice = activation._bridge.runtime_metadata()["latest_splice"]
                assert len(splice["structural_result_positions"]) == 56
            elif backbone == "qwen25_vl":
                window, _ = visual.get_window_index(args["video_grid_thw"])
                source = (inputs[0][0].reshape(64, 64) if before else outputs[0][0])[window.argsort()]
            else:
                source = inputs[0][0].reshape(64, 64) if before else outputs[0][0]
            torch.testing.assert_close(request.payload.features, source)
            selected = _ttm_indices(torch, source)
            assert result.representative_indices[0] == tuple(selected)
            torch.testing.assert_close(result.features, source[selected], atol=0, rtol=0)
            if backbone == "qwen3_vl":
                for branch, reduced in zip(request.deepstack_features, response.deepstack_features):
                    torch.testing.assert_close(reduced, branch[selected], atol=0, rtol=0)
            if before:
                values = [result.features, *getattr(response, "deepstack_features", ())]
                for index, compact in enumerate(values):
                    torch.testing.assert_close(inputs[index][1], compact if backbone == "llava_onevision" else compact.reshape(-1, 16))
            output.loss.backward()
            pixels = args["images"][0] if backbone == "llava_onevision" else args["pixel_values_videos"]
            assert pixels.grad is not None and torch.isfinite(pixels.grad).all() and pixels.grad.abs().sum() > 0
            if backbone == "llava_onevision":
                newline_gradient = model.model.image_newline.grad
                assert newline_gradient is not None and torch.isfinite(newline_gradient).all() and newline_gradient.abs().sum() > 0
            grads = [{name: value.grad for name, value in merger.named_parameters()} for merger in mergers]
            assert all(value is not None and torch.isfinite(value).all() for item in grads for value in item.values())
            metadata = activation.runtime_metadata()
            assert metadata["projector_position"] == ("before" if before else "after")
            assert metadata["real_checkpoint_verified"] is (not before)
            assert metadata["projector_layout_deltas"][0]["site"] == video_projector_site(backbone, "before" if before else "after")
            if position is None:
                baseline = output.logits.detach().clone(), pixels.grad.clone(), copy.deepcopy(grads), result.representative_indices
            elif not before:
                torch.testing.assert_close((output.logits, pixels.grad, grads), baseline[:3], atol=0, rtol=0)
                assert result.representative_indices == baseline[3]
        finally:
            for hook in hooks:
                hook.remove()
            activation.uninstall()


@pytest.mark.parametrize("backbone", _BACKBONES)
@pytest.mark.parametrize("first_sparse", [1, 3, 4])
def test_native_decode_trigger_full_storage_sparse_views_and_repeat_requests(backbone, first_sparse, monkeypatch):
    torch = pytest.importorskip("torch")
    from mllm_comp.backbones import dycoke_video as provider
    torch.manual_seed(463)
    model = _model(backbone).eval()
    position = "projector_input" if first_sparse == 1 else "projector_output"
    activation = image_fixture.activation(model, _stage(backbone, position, first_sparse, phase="generate"))
    activation.binding = replace(activation.binding, metadata={**activation.binding.metadata, "verified_phases": ["generate"]})
    activation.install()
    updates, queries, results = [], [], []
    native_update = provider.DyCoKeDecodeCacheController._update
    native_query = provider._decoder_query

    def update(controller, original, keys, values, layer_idx, cache_kwargs):
        returned = native_update(controller, original, keys, values, layer_idx, cache_kwargs)
        physical = controller.cache.layers[layer_idx].keys
        if keys.shape[-2] == 1:
            plan = controller.current_plan
            expected = physical[:, :, list(plan.view_indices)] if layer_idx >= first_sparse else physical
            torch.testing.assert_close(returned[0], expected, atol=0, rtol=0)
            updates.append((layer_idx, physical.shape[-2], returned[0].shape[-2]))
        return returned

    def query(backbone_name, layer, hidden, positions):
        queries.append(layer.self_attn.layer_idx)
        return native_query(backbone_name, layer, hidden, positions)

    monkeypatch.setattr(provider.DyCoKeDecodeCacheController, "_update", update)
    monkeypatch.setattr(provider, "_decoder_query", query)
    args = _inputs(torch, backbone)
    args["attention_mask"] = torch.ones_like(args["input_ids"])
    if backbone == "llava_onevision":
        args["inputs"] = args.pop("input_ids")
    try:
        for repeat in range(2):
            with torch.no_grad():
                generated = model.generate(**args, min_new_tokens=3, max_new_tokens=3, do_sample=False,
                    use_cache=True, eos_token_id=None, pad_token_id=0, return_dict_in_generate=True)
            results.append(generated)
            metadata = activation.runtime_metadata()
            assert metadata["projector_invocations"] == repeat + 1
            assert len(metadata["decode_kv_views"]) == 2
            assert metadata["decode_trigger_layer"] == first_sparse - 1
            assert metadata["real_checkpoint_verified"] is (first_sparse == 3)
            assert metadata["application_positions"] == [{"boundary": "decode_kv_before_attention", "index": first_sparse}]
            records = metadata["decode_kv_views"]
            assert records[0]["reused_visual_selection"] is False
            assert all(record["attention_source_layer"] == first_sparse - 1 and record["first_sparse_layer"] == first_sparse for record in records)
            full = records[-1]["sequence_length"]
            assert [layer.get_seq_length() for layer in generated.past_key_values.layers] == [full] * 5
            block = updates[-10:]
            assert len(block) == 10
            assert [item[0] for item in block] == list(range(5)) * 2
            assert all(view == storage if layer < first_sparse else view < storage for layer, storage, view in block)
        assert queries == [first_sparse - 1] * 4
        torch.testing.assert_close(results[0].sequences, results[1].sequences, atol=0, rtol=0)
        released_cache = weakref.ref(results[0].past_key_values)
        del results[0]
        gc.collect()
        assert released_cache() is None
        assert activation.runtime_metadata()["live_cache_controllers"] == 1
        cache = results[0].past_key_values
        controller = cache._mllm_comp_dycoke_cache_controller
        # The caller still owns this cache and can append the next decode step.
        for layer_index, layer in enumerate(cache.layers):
            keys, values = layer.keys[:, :, -1:], layer.values[:, :, -1:]
            if layer_index == first_sparse - 1:
                controller.prepare_decode_query(keys)
            cache.update(keys, values, layer_index)
        assert [layer.get_seq_length() for layer in cache.layers] == [full + 1] * 5
    finally:
        activation.uninstall()
    for result in results:
        assert not hasattr(result.past_key_values, "_mllm_comp_dycoke_cache_controller")
        assert "update" not in vars(result.past_key_values)


def test_new_stride_recent_method_shares_native_training_replay_and_decode_lifecycle(monkeypatch):
    torch = pytest.importorskip("torch")
    from examples.compression_extensions import stride_recent_video as research
    from mllm_comp.backbones import dycoke_video as provider
    from mllm_comp.config.schema import experiment_config_from_dict
    from mllm_comp.pipeline.definition import resolve_compression_definition

    torch.manual_seed(464)
    model = _model("qwen3_vl").train()
    original_forward = model.forward
    args = _inputs(torch, "qwen3_vl")
    selected = tuple(range(1, 64, 4))

    def activation(phase):
        config = experiment_config_from_dict({
            "backbone": {"name": "qwen3_vl", "model_path": "native-tiny-video"},
            "pipeline": {"extensions": ["examples.compression_extensions.stride_recent_video:register"],
                "stages": [{"name": "recent", "algorithm": research.ALGORITHM,
                            "placement": "projector_level", "execution_phase": phase}]},
        })
        definition = resolve_compression_definition(config, compression_mode="compressed",
            interface="train" if phase == "forward" else "eval")
        projection = definition.projections["recent"]
        binding = projection.bindings[0]
        return binding.executor_factory(model=model, stage=config.pipeline.stages[0],
            binding=binding, contract_projection=projection, execution_phase=phase)

    train = activation("forward")
    observed = []
    callback = train._projector_callback
    def capture(request):
        response = callback(request)
        observed.append((request, response))
        return response
    train._projector_callback = capture
    train.install()
    try:
        labels = torch.full_like(args["input_ids"], -100)
        labels[0, -1] = args["input_ids"][0, -1]
        output = model(**args, labels=labels, use_cache=False)
        request, response = observed[0]
        result = response.compression_result
        assert result.representative_indices == (selected,)
        torch.testing.assert_close(result.features, request.payload.features[list(selected)], atol=0, rtol=0)
        for source, branch in zip(request.deepstack_features, response.deepstack_features):
            torch.testing.assert_close(branch, source[list(selected)], atol=0, rtol=0)
            gradient = torch.autograd.grad(branch.sum(), source, retain_graph=True)[0]
            expected = torch.zeros_like(source)
            expected[list(selected)] = 1
            torch.testing.assert_close(gradient, expected, atol=0, rtol=0)
        output.loss.backward()
        assert torch.isfinite(output.loss)
        assert args["pixel_values_videos"].grad.abs().sum() > 0
        assert train.runtime_metadata()["decode_view_invocations"] == 0
    finally:
        train.uninstall()
    assert model.forward == original_forward

    model.eval()
    generate = activation("generate")
    native_update = provider.DyCoKeDecodeCacheController._update
    sparse_returns = []
    def update(controller, original, keys, values, layer_idx, cache_kwargs):
        returned = native_update(controller, original, keys, values, layer_idx, cache_kwargs)
        if keys.shape[-2] == 1 and layer_idx >= controller.prune_layer:
            physical = controller.cache.layers[layer_idx].keys
            expected = physical[:, :, list(controller.current_plan.view_indices)]
            torch.testing.assert_close(returned[0], expected, atol=0, rtol=0)
            sparse_returns.append((physical.shape[-2], returned[0].shape[-2]))
        return returned
    monkeypatch.setattr(provider.DyCoKeDecodeCacheController, "_update", update)
    generate.install()
    results = []
    try:
        for _ in range(2):
            with torch.no_grad():
                result = model.generate(**args, min_new_tokens=3, max_new_tokens=3, do_sample=False,
                    use_cache=True, eos_token_id=None, pad_token_id=0, return_dict_in_generate=True)
            results.append(result)
            metadata = generate.runtime_metadata()
            assert metadata["algorithm"] == research.ALGORITHM
            assert not metadata["real_checkpoint_verified"]
            records = metadata["decode_kv_views"]
            assert len(records) == 2
            for record in records:
                assert record["kept_visual_positions"] == record["visual_positions"][-2:]
                assert record["attention_similarity"] is None
                assert record["reused_visual_selection"] is False
            full_length = records[-1]["sequence_length"]
            assert [layer.get_seq_length() for layer in result.past_key_values.layers] == [full_length] * 5
        policies = [item.past_key_values._mllm_comp_dycoke_cache_controller._decode_policy for item in results]
        assert policies[0] is not policies[1]
        assert len(sparse_returns) == 8 and all(view < full for full, view in sparse_returns)
    finally:
        generate.uninstall()
    assert model.forward == original_forward
    for result in results:
        assert not hasattr(result.past_key_values, "_mllm_comp_dycoke_cache_controller")
        assert "update" not in vars(result.past_key_values)
