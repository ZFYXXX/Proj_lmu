"""Configured callable rules reach all nine existing video executors on three bases."""

from dataclasses import replace

import pytest

from mllm_comp.pipeline.stage import CompressionStageConfig

from tests.fixtures import native_image as image_fixture
from tests.integration import test_video_dytok_positions_cpu as dytok
from tests.integration import test_video_metok_positions_cpu as metok
from tests.integration import test_video_prunevid_positions_cpu as prunevid
from tests.integration import test_video_dycoke_positions_cpu as dycoke
from tests.integration import test_video_longvu_positions_cpu as longvu
from tests.integration import test_video_llamavid_positions_cpu as llamavid


BACKBONES = ("llava_onevision", "qwen25_vl", "qwen3_vl")
ALGORITHMS = ("visionzip", "dytok", "vqtoken", "fastvid", "metok", "prunevid", "dycoke", "longvu", "llamavid")
_CALLS = {}
_EXPECTED_TRANSFORMS = []


def _called(name):
    _CALLS[name] = _CALLS.get(name, 0) + 1


def group_mean(request, scale=2.0):
    _called("aggregator")
    return request["values"][list(request["group"])].mean(0) * scale


def fastvid_values(request, scale=2.0):
    _called("aggregator")
    value = request["values"][request["target"]] * scale
    if request["assigned_indices"]:
        value = value + request["values"][list(request["assigned_indices"])].mean(0)
    return value


def scaled_transform(request, scale=2.0):
    _called("transform")
    original = request["default_transform"]()
    _EXPECTED_TRANSFORMS.append(original.detach() * scale)
    return original * scale


def content_pool(request, scale=2.0):
    _called("content_aggregator")
    expected = request["values"].mean(0) * scale
    if request["branch"] == "main":
        _EXPECTED_TRANSFORMS.append(expected.detach())
    return expected


def first_indices(request):
    _called("selector")
    candidates = request.get("candidates", range(len(request["scores"])))
    return tuple(candidates)[:request["keep_count"]]


def source_scores(request):
    _called("scorer")
    values = request["values"]
    positions = request["visual_positions"]
    return values[0, list(positions)].float().square().mean(-1)


def ttm_scores(request):
    import torch
    _called("ttm_scorer")
    return torch.arange(len(request["values"]), device=request["values"].device).float()


def _configured(stage, rules):
    specs = {name: {"callable": f"{__name__}:{function}",
                    "params": params}
             for name, (function, params) in rules.items()}
    return replace(stage, operation={**stage.operation, "rules": specs})


def _setup(algorithm, backbone, tmp_path):
    import torch
    context = metok._context(backbone) if algorithm in {"metok", "dytok"} else {}
    rules = {"aggregator": ("group_mean", {"scale": 2.0})}
    if algorithm == "metok":
        model, stage = metok._model(backbone), metok._stage(backbone)
        args = dytok._inputs(torch, backbone)
        rules = {"pool_transform": ("scaled_transform", {"scale": 2.0}),
                 "prune_scorer": ("source_scores", {}), "prune_selector": ("first_indices", {})}
        stage = _configured(stage, rules)
        activation = metok._executor(model, backbone, stage)
    elif algorithm == "prunevid":
        model, stage = prunevid._model(backbone), prunevid._stage(backbone, "mid", projector="projector_input")
        args = prunevid._training_inputs(torch, backbone)
        rules.update(vtp_scorer=("source_scores", {}), vtp_selector=("first_indices", {}))
        stage = _configured(stage, rules)
        activation = image_fixture.activation(model, stage)
    elif algorithm == "dycoke":
        model, stage = dycoke._model(backbone), dycoke._stage(backbone)
        args = dycoke._inputs(torch, backbone)
        rules = {"ttm_scorer": ("ttm_scores", {}), "ttm_selector": ("first_indices", {})}
        stage = _configured(stage, rules)
        activation = image_fixture.activation(model, stage)
    elif algorithm == "longvu":
        model, stage = dytok._model(backbone), longvu._stage(backbone)
        args = dytok._inputs(torch, backbone)
        rules = {"fusion_transform": ("scaled_transform", {"scale": 2.0})}
        stage = _configured(stage, rules)
        activation = longvu._executor(model, backbone, stage)
    elif algorithm == "llamavid":
        model, stage = llamavid._model(backbone), llamavid._stage(backbone, "llamavid_post_output_projectors")
        args = llamavid._inputs(torch, backbone)
        rules = {"content_aggregator": ("content_pool", {"scale": 2.0})}
        stage = _configured(stage, rules)
        activation = llamavid._activation(model, stage, llamavid._tokenizer(tmp_path))
        context = ({"conditioning_text": "Which action occurs?"} if backbone == "llava_onevision"
                   else {"video_conditioning_texts": ["Which action occurs?"]})
    else:
        model, args = dytok._model(backbone), dytok._inputs(torch, backbone)
        if algorithm == "dytok":
            stage = _configured(dytok._stage(backbone), rules)
            activation = dytok._executor(model, backbone, stage, dytok._FrameWeights())
        else:
            if algorithm == "fastvid":
                rules = {"aggregator": ("fastvid_values", {"scale": 2.0})}
            stage = _configured(CompressionStageConfig.from_dict({
                "name": "native-video-rules", "placement": "projector_level", "execution_phase": "forward",
                "plugin": algorithm, "algorithm": algorithm, "variant": f"{algorithm}.video",
                "binding": f"{algorithm}.video.{backbone}", "target_modalities": ["video"]}), rules)
            activation = image_fixture.activation(model, stage)
            if algorithm == "fastvid" and backbone == "llava_onevision":
                from tests.integration.test_video_signal_projector_llava_onevision_positions_cpu import _pooling_head
                activation.signal_provider._pooling_head = _pooling_head(tmp_path)
    args["attention_mask"] = torch.ones_like(args["input_ids"])
    args["labels"] = torch.full_like(args["input_ids"], -100)
    args["labels"][0,-1] = args["input_ids"][0,-1]
    args["use_cache"] = False
    return model.train(), activation, args, context, stage


def _assert_values(torch, algorithm, request, result, response):
    sources = [request.payload.features, *getattr(request, "deepstack_features", ())]
    outputs = [result.features, *getattr(response, "deepstack_features", ())]
    if algorithm in {"visionzip", "dytok", "prunevid", "vqtoken"}:
        for source, output in zip(sources, outputs):
            if algorithm == "vqtoken":
                source = torch.nn.functional.normalize(source, p=2, dim=-1)
            expected = torch.stack([source[list(group)].mean(0) * 2 for group in result.source_groups[0]])
            torch.testing.assert_close(output, expected)
    elif algorithm == "fastvid":
        for source, output in zip(sources, outputs):
            rows = []
            for record in result.metadata["value_records_by_video"][0]:
                value = source[record["center_index"]]
                if record["kind"] != "salient":
                    value = value * 2
                    if record["assigned_indices"]:
                        value = value + source[list(record["assigned_indices"])].mean(0)
                rows.append(value)
            torch.testing.assert_close(output, torch.stack(rows))
    elif algorithm == "metok":
        for source, output in zip(sources, outputs):
            torch.testing.assert_close(output, metok._bilinear(torch, source, result) * 2)
        assert result.metadata["value_operation"] == "custom"
    elif algorithm == "dycoke":
        per_frame = request.payload.token_count // 4
        keep = max(1, int(per_frame * (1.0 - .7)))
        expected = tuple(range(per_frame)) + tuple(frame * per_frame + i for frame in range(1,4) for i in range(keep))
        for source, output in zip(sources, outputs):
            torch.testing.assert_close(output, source[list(expected)])
    elif algorithm == "longvu":
        torch.testing.assert_close(result.features, _EXPECTED_TRANSFORMS[0].reshape_as(result.features))
    else:
        torch.testing.assert_close(result.features[1::2], torch.stack(_EXPECTED_TRANSFORMS))


@pytest.mark.parametrize("backbone", BACKBONES)
@pytest.mark.parametrize("algorithm", ALGORITHMS)
def test_configured_video_rule_changes_native_values_and_preserves_gradients(algorithm, backbone, tmp_path):
    torch = pytest.importorskip("torch")
    torch.manual_seed(821)
    _CALLS.clear()
    _EXPECTED_TRANSFORMS.clear()
    model, activation, args, context, stage = _setup(algorithm, backbone, tmp_path)
    captured = []
    original = activation._projector_callback
    def callback(request):
        response = original(request)
        result = getattr(response, "compression_result", response)
        streams = 1 + len(getattr(request, "deepstack_features", ()))
        if algorithm == "metok":
            assert _CALLS["transform"] == streams * len(result.metadata["metok_frames"][0])
        elif algorithm == "prunevid":
            assert _CALLS["aggregator"] == streams * result.result_count
        _assert_values(torch, algorithm, request, result, response)
        captured.append(result)
        return response
    activation._projector_callback = callback
    try:
        activation.install()
        with activation._bridge.request_context(context):
            output = model(**args)
        output.loss.backward()
        assert len(captured) == 1 and _CALLS
        if algorithm in {"metok", "prunevid"}:
            assert _CALLS["scorer"] > 0 and _CALLS["selector"] > 0
        pixels = args["images"][0] if backbone == "llava_onevision" else args["pixel_values_videos"]
        assert pixels.grad is not None and torch.isfinite(pixels.grad).all() and pixels.grad.abs().sum() > 0
        if algorithm == "longvu":
            assert activation.fusion_module.vision_query.grad is not None
        if algorithm == "llamavid":
            assert activation.qformer_module.query_tokens.grad is not None
        metadata = activation.runtime_metadata()
        assert metadata["rules"] == stage.operation["rules"]
        assert metadata["rules_changed"] and metadata["development_runtime"]
        assert metadata["canonical_evidence_applicable"] is False
    finally:
        activation.uninstall()


@pytest.mark.parametrize("algorithm", ("metok", "prunevid"))
def test_custom_mid_decisions_are_not_rerun_during_checkpoint_backward(algorithm, tmp_path):
    torch = pytest.importorskip("torch")
    expected = None
    for checkpoint in (None, False, True):
        torch.manual_seed(831)
        _CALLS.clear()
        _EXPECTED_TRANSFORMS.clear()
        model, activation, args, context, _ = _setup(algorithm, "qwen3_vl", tmp_path)
        if checkpoint is not None:
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": checkpoint})
        try:
            activation.install()
            with activation._bridge.request_context(context):
                output = model(**args)
            decisions = (_CALLS["scorer"], _CALLS["selector"])
            # Recompute must retain the first forward's rule decision, even
            # after the request context and installed runtime have ended.
            activation.uninstall()
            output.loss.backward()
            assert decisions == (_CALLS["scorer"], _CALLS["selector"])
            gradients = {name: parameter.grad for name, parameter in model.named_parameters()
                         if parameter.grad is not None}
            assert all(torch.isfinite(value).all() for value in gradients.values())
            result = (output.logits.detach(), args["pixel_values_videos"].grad, gradients)
            if expected is None:
                expected = result
            else:
                torch.testing.assert_close(result, expected, atol=3e-6, rtol=3e-5)
        finally:
            activation.uninstall()


def decode_scores(request):
    _called("decode_scorer")
    return request["default_scores"].flip(0)


def decode_select(request):
    _called("decode_selector")
    return request["visual_positions"][-request["keep_count"]:]


def decode_reuse(request):
    _called("decode_decision")
    return request["previous_scores"] is not None


def test_configured_dycoke_decode_rules_change_only_actual_cache_views(monkeypatch):
    torch = pytest.importorskip("torch")
    from mllm_comp.backbones import dycoke_video as provider
    _CALLS.clear()
    torch.manual_seed(832)
    model = dycoke._model("qwen3_vl").eval()
    stage = _configured(dycoke._stage("qwen3_vl", sparse_layer=1, phase="generate"), {
        "decode_scorer": ("decode_scores", {}), "decode_selector": ("decode_select", {}),
        "decode_decision": ("decode_reuse", {})})
    activation = image_fixture.activation(model, stage)
    # This is the existing native fixture phase setup; no new support claim.
    activation.binding = replace(activation.binding,
        metadata={**activation.binding.metadata, "verified_phases": ["generate"]})
    updates = []
    original = provider.DyCoKeDecodeCacheController._update
    def update(controller, native, keys, values, layer_idx, cache_kwargs):
        returned = original(controller, native, keys, values, layer_idx, cache_kwargs)
        if keys.shape[-2] == 1:
            physical = controller.cache.layers[layer_idx].keys
            plan = controller.current_plan
            expected = physical[:, :, list(plan.view_indices)] if layer_idx >= 1 else physical
            torch.testing.assert_close(returned[0], expected, rtol=0, atol=0)
            updates.append((layer_idx, physical.shape[-2], returned[0].shape[-2]))
        return returned
    monkeypatch.setattr(provider.DyCoKeDecodeCacheController, "_update", update)
    activation.install()
    args = dycoke._inputs(torch, "qwen3_vl")
    try:
        with torch.no_grad():
            generated = model.generate(**args, min_new_tokens=3, max_new_tokens=3, do_sample=False,
                use_cache=True, eos_token_id=None, pad_token_id=0, return_dict_in_generate=True)
        records = activation.runtime_metadata()["decode_kv_views"]
        assert len(records) == 2
        first, second = records
        assert first["kept_visual_positions"] == first["visual_positions"][-first["kept_visual_count"]:]
        assert second["kept_visual_positions"] == first["kept_visual_positions"]
        assert second["reused_visual_selection"] and not first["reused_visual_selection"]
        assert _CALLS == {"decode_scorer": 2, "decode_decision": 2, "decode_selector": 1}
        assert [layer.get_seq_length() for layer in generated.past_key_values.layers] == [second["sequence_length"]] * 5
        assert len(updates) == 10
        assert all(view == full if layer == 0 else view < full for layer, full, view in updates)
    finally:
        activation.uninstall()


def frame_context_pool(request):
    _called("frame_context_pool")
    values = request["values"]
    return (values[list(request["group"])].square().mean(0)
            + values.mean(0) * .5 + request["target"] / len(values))


@pytest.mark.parametrize("algorithm", ("visionzip", "dytok"))
def test_qwen3_video_frame_context_rule_replays_on_each_real_deepstack_branch(algorithm):
    torch = pytest.importorskip("torch")
    torch.manual_seed(834)
    model = dytok._model("qwen3_vl").train()
    stage = dytok._stage("qwen3_vl") if algorithm == "dytok" else CompressionStageConfig.from_dict({
        "name": "native-video-frame-context", "placement": "projector_level", "execution_phase": "forward",
        "plugin": algorithm, "algorithm": algorithm, "variant": f"{algorithm}.video",
        "binding": f"{algorithm}.video.qwen3_vl", "target_modalities": ["video"]})
    stage = _configured(stage, {"aggregator": ("frame_context_pool", {})})
    activation = (dytok._executor(model, "qwen3_vl", stage, dytok._FrameWeights())
                  if algorithm == "dytok" else image_fixture.activation(model, stage))
    original, captured = activation._projector_callback, []
    def callback(request):
        sources = [request.payload.features, *request.deepstack_features]
        for source in sources:
            source.retain_grad()
        response = original(request)
        result = response.compression_result
        targets = result.metadata["merge_target_indices_by_video"][0]
        for source, actual in zip(sources, [result.features, *response.deepstack_features]):
            expected, source_start, result_start = [], 0, 0
            for record in result.metadata["frame_records_by_video"][0]:
                source_stop = source_start + record["source_count"]
                frame = source[source_start:source_stop]
                result_stop = result_start + len(record["representative_indices"])
                for i in range(result_start, result_stop):
                    expected.append(source[list(result.source_groups[0][i])].square().mean(0)
                        + frame.mean(0) * .5 + (targets[i] - source_start) / len(frame))
                source_start, result_start = source_stop, result_stop
            torch.testing.assert_close(actual, torch.stack(expected))
        captured.extend(sources)
        return response
    activation._projector_callback = callback
    args = dytok._inputs(torch, "qwen3_vl")
    args["attention_mask"] = torch.ones_like(args["input_ids"])
    args["labels"] = torch.full_like(args["input_ids"], -100)
    args["labels"][0,-1] = args["input_ids"][0,-1]
    try:
        activation.install()
        with activation._bridge.request_context(metok._context("qwen3_vl") if algorithm == "dytok" else {}):
            output = model(**args)
        output.loss.backward()
        assert len(captured) == 4
        assert all(value.grad is not None and torch.isfinite(value.grad).all() and value.grad.abs().sum() > 0
                   for value in captured)
    finally:
        activation.uninstall()
