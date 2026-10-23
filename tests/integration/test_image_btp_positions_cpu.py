"""BTP preserves stage rules and FinalWipe while moving their application boundaries."""

import copy
import importlib

import pytest

from mllm_comp.pipeline.stage import CompressionStageConfig
from mllm_comp.compressors.image import decoder_selection as execution
from tests.fixtures import native_image as image_fixture


def _stage(backbone, boundaries=(1, 2, 3), *, sources=(2, 3, 4), mid=False, phase="forward"):
    count = len(sources)
    return CompressionStageConfig.from_dict({
        "name": "btp-positions", "placement": "llm_level", "execution_phase": phase,
        "plugin": "btp", "algorithm": "btp", "variant": "btp.image",
        "binding": f"btp.image.{backbone}", "target_modalities": ["image"],
        "operation": {"targets": {"decoder_balanced_pruning_boundaries": {
            "target_site": "llm_decoder_layer_post_attention_image_tokens" if mid else "llm_decoder_layer_output_image_tokens",
            "layers": list(boundaries), **({} if mid else {"layer_edge": "before"})}}},
        "config": {"prune_before_layers": list(sources), "retain_ratios": [0.5] * (count - 1) + [0.0],
            "attention_fractions": [0.4] * (count - 1) + [0.0], "attention_oversample": [1.2] * count,
            "rebalance_stages": [i == count - 2 and count > 2 for i in range(count)]},
    })


def _run(backbone, boundaries, sources, mid, *, two_devices=False, stage_factory=_stage, visual_counts=None, layer_local=False, checkpoint_modes=(None, False, True)):
    torch = pytest.importorskip("torch")
    if two_devices and torch.cuda.device_count() < 2:
        pytest.skip("Explicit two-device BTP source/application test.")
    torch.manual_seed(272)
    base = image_fixture.model(backbone, num_decoder_layers=4).train()
    expected = None
    for checkpoint in checkpoint_modes:
        model = copy.deepcopy(base)
        decoder = image_fixture.decoder(model, backbone)
        if two_devices:
            from accelerate import dispatch_model
            prefix = "model" if backbone == "llava_onevision" else "model.language_model"
            device_map = {name: "cuda:0" for name, _ in model.named_children() if name != "model"}
            if backbone != "llava_onevision":
                device_map.update({f"model.{name}": "cuda:0" for name, _ in model.model.named_children() if name != "language_model"})
            device_map.update({f"{prefix}.{name}": "cuda:0" for name, _ in decoder.named_children() if name != "layers"})
            device_map.update({f"{prefix}.layers.{i}": "cuda:0" if i < 2 else "cuda:1" for i in range(len(decoder.layers))})
            dispatch_model(model, device_map=device_map)
        if checkpoint is not None:
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": checkpoint})
        activation = image_fixture.activation(model, stage_factory(backbone, boundaries, sources=sources, mid=mid))
        activation.install()
        if backbone == "llava_onevision":
            activation._bridge.prepare_training_forward()
        torch.manual_seed(273)
        raw = image_fixture.inputs(torch, backbone)
        args = {name: value.to("cuda:0") if two_devices and torch.is_tensor(value) else value for name, value in raw.items()}
        labels = torch.full_like(args["input_ids"], -100)
        labels[0, -1] = args["input_ids"][0, -1]
        attention, mlp = [[] for _ in decoder.layers], [[] for _ in decoder.layers]
        handles = []
        for i, layer in enumerate(decoder.layers):
            handles.append(layer.self_attn.o_proj.register_forward_pre_hook(lambda module, args, i=i: attention[i].append(args[0].shape[1])))
            handles.append(layer.mlp.register_forward_pre_hook(lambda module, args, i=i: mlp[i].append(args[0].shape[1])))
        try:
            result = model(**args, labels=labels, use_cache=False)
            records = activation.runtime_metadata()["layout_delta_records"]
            assert len(records) == len(boundaries)
            initial = 732 if backbone == "llava_onevision" else 8
            text_count = 3 if backbone == "llava_onevision" else 4
            # Content-dependent rules report cardinality; the same physical
            # Attention/FFN/cache checks still verify when it takes effect.
            counts = [item["result_count"] - text_count for item in records] if visual_counts == "recorded" else visual_counts
            final = text_count + (counts[-1] if counts is not None else 0)
            assert result.logits.shape == (1, final, 64)
            sizes = [initial, *(record["result_count"] for record in records)]
            visual = initial - text_count
            for index, record in enumerate(records):
                visual = counts[index] if counts is not None else (max(1, int(visual * 0.5)) if index < len(records) - 1 else 0)
                assert record["result_count"] == text_count + visual
            assert [item[0] for item in attention] == [sizes[sum(point + int(mid) <= i for point in boundaries)] for i in range(4)]
            assert [item[0] for item in mlp] == [sizes[sum(point <= i for point in boundaries)] for i in range(4)]
            result.loss.backward()
            assert model.lm_head.weight.grad is not None and torch.isfinite(model.lm_head.weight.grad).all()
            grads = {name: parameter.grad.detach().cpu().clone() for name, parameter in model.named_parameters() if parameter.grad is not None}
            assert grads and all(torch.isfinite(grad).all() for grad in grads.values())
            pixel_grad = raw["images" if backbone == "llava_onevision" else "pixel_values"].grad
            assert pixel_grad is not None and torch.isfinite(pixel_grad).all() and pixel_grad.abs().sum() > 0
            observed = result.logits.detach().cpu(), grads, pixel_grad.clone()
            if expected is None:
                expected = observed
            else:
                torch.testing.assert_close(observed, expected, atol=2e-6, rtol=2e-5)
            assert len(activation.runtime_metadata()["layout_delta_records"]) == len(boundaries)
            assert activation.runtime_metadata()["application_positions"] == [{"boundary": "post_attention_pre_ffn" if mid else "inter_layer", "index": i} for i in boundaries]
        finally:
            for handle in handles:
                handle.remove()
            activation.uninstall()
        if checkpoint is not checkpoint_modes[-1]:
            continue
        model.eval()
        activation = image_fixture.activation(model, stage_factory(backbone, boundaries, sources=sources, mid=mid, phase="generate"))
        activation.install()
        if backbone == "qwen3_vl":
            from mllm_comp.backbones.qwen3_vl.provider_bridge import install_qwen3_vl_transformers_4573_bridge
            bridge = install_qwen3_vl_transformers_4573_bridge(model)
        else:
            bridge = activation._bridge
        if backbone == "llava_onevision":
            args["inputs"] = args.pop("input_ids")
        outputs = []
        try:
            for _ in range(2):
                with torch.no_grad():
                    generated = model.generate(**args, min_new_tokens=3, max_new_tokens=3, do_sample=False,
                        use_cache=True, eos_token_id=None, pad_token_id=0, return_dict_in_generate=True)
                expected_cache = [sizes[sum(point + int(mid) <= i for point in boundaries)] + 2 for i in range(4)] if layer_local else [final + 2] * 4
                assert [layer.get_seq_length() for layer in generated.past_key_values.layers] == expected_cache
                report = bridge.runtime_metadata()["latest_decoder_lifecycle"]
                assert report["selection_runs"] == len(boundaries)
                assert report["decode_forward_calls"] == 2
                assert report["selection_repeated_during_decode"] is False
                assert [record["cache"]["touched_layers"] for record in report["boundaries"]] == ([[] for point in boundaries] if layer_local else [list(range(point + int(mid))) for point in boundaries])
                outputs.append(generated.sequences)
            torch.testing.assert_close(outputs[0], outputs[1])
        finally:
            activation.uninstall()

@pytest.mark.parametrize("backbone", ["qwen3_vl", "qwen25_vl", "llava_onevision"])
@pytest.mark.parametrize("boundaries,mid", [((0, 1, 2), False), ((1, 2, 3), False), ((2, 3, 4), False),
    ((0, 1, 2), True), ((1, 2, 3), True)])
def test_btp_three_native_stages_all_gradients_checkpoint_and_generate(backbone, boundaries, mid):
    _run(backbone, boundaries, (2, 3, 4), mid)


@pytest.mark.parametrize("backbone", ["qwen3_vl", "qwen25_vl", "llava_onevision"])
@pytest.mark.parametrize("mid", [False, True])
def test_btp_sources_across_two_devices(backbone, mid):
    _run(backbone, (1, 2, 3), (2, 3, 4), mid, two_devices=True)


@pytest.mark.parametrize("backbone", ["qwen3_vl", "qwen25_vl", "llava_onevision"])
def test_btp_four_stage_native_schedule(backbone):
    _run(backbone, (0, 1, 2, 3), (2, 3, 4, 5), True)


@pytest.mark.parametrize("backbone", ["qwen3_vl", "qwen25_vl", "llava_onevision"])
def test_btp_default_and_equivalent_boundary_expressions_match(backbone):
    _default_aliases(backbone, _stage, "decoder_balanced_pruning_boundaries", (0, 1, 2))


def _default_aliases(backbone, stage_factory, target_name, after_layers):
    torch = pytest.importorskip("torch")
    torch.manual_seed(325)
    base = image_fixture.model(backbone, num_decoder_layers=4).train()
    expected = None
    for mode in ("legacy", "before", "after"):
        model = copy.deepcopy(base)
        stage = stage_factory(backbone)
        if mode == "legacy":
            stage.operation = {}
        elif mode == "after":
            stage.operation["targets"][target_name].update(layers=list(after_layers), layer_edge="after")
        activation = image_fixture.activation(model, stage)
        activation.install()
        if backbone == "llava_onevision":
            activation._bridge.prepare_training_forward()
        torch.manual_seed(326)
        args = image_fixture.inputs(torch, backbone)
        labels = torch.full_like(args["input_ids"], -100)
        labels[0, -1] = args["input_ids"][0, -1]
        try:
            result = model(**args, labels=labels, use_cache=False)
            result.loss.backward()
            observed = result.logits.detach(), model.lm_head.weight.grad
            if expected is None:
                expected = tuple(value.clone() for value in observed)
            else:
                torch.testing.assert_close(observed, expected, atol=0, rtol=0)
        finally:
            activation.uninstall()


@pytest.mark.parametrize("backbone", ["qwen3_vl", "qwen25_vl", "llava_onevision"])
@pytest.mark.parametrize("boundaries,mid", [((1, 2, 3), False), ((2, 3, 4), False), ((0, 1, 2), True)])
def test_btp_fixed_native_sources_and_mapped_scores(backbone, boundaries, mid, monkeypatch):
    torch = pytest.importorskip("torch")
    torch.manual_seed(330)
    model = image_fixture.model(backbone, num_decoder_layers=4).eval()
    decoder = image_fixture.decoder(model, backbone)
    provider = importlib.import_module(f"mllm_comp.backbones.{backbone}.provider_bridge")
    project, kernel = provider._decoder_query_keys, execution.balanced_token_pruning_image_tokens
    pending, sampled, actual_attention_inputs, stages = {}, {}, {}, []
    inside_score = False
    def observe(index):
        def hook(module, args):
            if not inside_score:
                actual_attention_inputs[index] = args[0].detach().clone()
        return hook
    handles = [decoder.layers[index].self_attn.q_proj.register_forward_pre_hook(observe(index)) for index in (0, 1)]
    def capture(layer, hidden, **kwargs):
        nonlocal inside_score
        index = list(decoder.layers).index(layer)
        assert index in (0, 1) and index not in sampled
        sampled[index] = layer.input_layernorm(hidden).detach().clone()
        inside_score = True
        try:
            query, image_keys, keys = project(layer, hidden, **kwargs)
        finally:
            inside_score = False
        assert kwargs["instruction_indices"].tolist() == [hidden.shape[1] - 1]
        logits = torch.einsum("qhd,thd->qht", query, keys) / query.shape[-1] ** 0.5
        pending[index] = logits.float().softmax(-1).mean((0, 1))[kwargs["image_indices"]]
        return query, image_keys, keys
    def apply(hidden, **kwargs):
        index = len(stages)
        assert kwargs["boundary_after_layer"] == index
        if index < 2:
            torch.testing.assert_close(kwargs["image_attention_scores"], pending.pop(index))
        else:
            assert kwargs["image_attention_scores"] is None
        result = kernel(hidden, **kwargs)
        assert result.metadata["stage_index"] == index
        assert result.metadata["retain_ratio"] == (0.5, 0.5, 0.0)[index]
        torch.testing.assert_close(result.hidden_states, hidden[result.kept_indices], atol=0, rtol=0)
        if index == 2:
            assert result.metadata["stage_kind"] == "final_wipe"
            assert result.kept_image_indices.numel() == 0
        for key, value in pending.items():
            pending[key] = value[result.selected_visual_indices]
        stages.append(index)
        return result
    monkeypatch.setattr(provider, "_decoder_query_keys", capture)
    monkeypatch.setattr(execution, "balanced_token_pruning_image_tokens", apply)
    activation = image_fixture.activation(model, _stage(backbone, boundaries, mid=mid))
    activation.install()
    try:
        with torch.no_grad():
            model(**image_fixture.inputs(torch, backbone), use_cache=False)
        assert stages == [0, 1, 2] and not pending
        # These boundaries leave each source's own Attention intact. In the
        # consecutive default, source 1 must see stage 0's compacted entry.
        torch.testing.assert_close(sampled, actual_attention_inputs, atol=2e-6, rtol=2e-5)
    finally:
        for handle in handles:
            handle.remove()
        activation.uninstall()


@pytest.mark.parametrize("backbone", ["qwen3_vl", "qwen25_vl", "llava_onevision"])
@pytest.mark.parametrize("mid", [False, True])
def test_btp_final_wipe_alone_has_no_future_scoring_dependency(backbone, mid):
    torch = pytest.importorskip("torch")
    model = image_fixture.model(backbone, num_decoder_layers=4).eval()
    activation = image_fixture.activation(model, _stage(backbone, (0,), sources=(23,), mid=mid))
    activation.install()
    try:
        with torch.no_grad():
            result = model(**image_fixture.inputs(torch, backbone), use_cache=False)
        assert result.logits.shape[1] == (3 if backbone == "llava_onevision" else 4)
        assert activation.runtime_metadata()["score_layers"] == []
    finally:
        activation.uninstall()
