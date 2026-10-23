"""PruneVid fixed score sources and actual projector/decoder application sites."""

import copy
from dataclasses import replace

import pytest

from mllm_comp.pipeline.stage import CompressionStageConfig

from tests.fixtures import native_image as image_fixture
from tests.integration.test_video_dycoke_positions_cpu import _model as _five_layer_model, _inputs


BACKBONES = ("llava_onevision", "qwen25_vl", "qwen3_vl")


def _model(backbone, *, canonical=False):
    source = _five_layer_model(backbone)
    if not canonical:
        return source
    config = copy.deepcopy(source.config)
    text = config if backbone == "llava_onevision" else config.text_config
    text.num_hidden_layers = 12
    if hasattr(text, "layer_types"):
        text.layer_types = ["full_attention"] * 12
    model = type(source)(config)
    if backbone == "llava_onevision":
        model.model.vision_tower = source.model.vision_tower
        model.model.mm_projector = source.model.mm_projector
        model.model.image_newline = source.model.image_newline
    return model


def _stage(backbone, position=None, *, projector=None, source=0, phase="forward"):
    targets = {}
    if projector is not None:
        targets["temporal_spatial_reduce"] = {"target_site": projector}
    if position is not None:
        mid = position == "mid"
        targets["vtp_prefill_prune"] = {"target_site": "llm_decoder_layer_post_attention_video_tokens" if mid else "llm_decoder_layer_output_video_tokens",
            "layers": [0 if position in {"initial", "mid"} else 4],
            **({"layer_edge": "before"} if position == "initial" else {})}
    return CompressionStageConfig.from_dict({
        "name": "prunevid-positions", "placement": "projector_level", "execution_phase": phase,
        "plugin": "prunevid", "algorithm": "prunevid", "variant": "prunevid.video",
        "binding": f"prunevid.video.{backbone}", "target_modalities": ["video"],
        "operation": {"targets": targets}, "config": {} if source is None else {"vtp_selected_layer": source},
    })


def _decoder(model, backbone):
    return model.model if backbone == "llava_onevision" else model.model.language_model


def _training_inputs(torch, backbone):
    args = _inputs(torch, backbone)
    args["attention_mask"] = torch.ones_like(args["input_ids"])
    labels = torch.full_like(args["input_ids"], -100)
    labels[0, -1] = args["input_ids"][0, -1]
    return {**args, "labels": labels, "use_cache": False}


@pytest.mark.parametrize("backbone", BACKBONES)
@pytest.mark.parametrize("position", ["initial", "mid", "terminal"])
def test_native_fixed_projected_plan_raw_values_and_decoder_boundaries(backbone, position, monkeypatch):
    torch = pytest.importorskip("torch")
    from mllm_comp.backbones import video_decoder as adapter
    from mllm_comp.compressors.video import PruneVidVideoKernel
    torch.manual_seed(481)
    model = _model(backbone).train()
    before = position != "terminal"
    activation = image_fixture.activation(model, _stage(backbone, position, projector="projector_input" if before else "projector_output"))
    callback = activation._projector_callback
    observed = []
    score_sources = []
    original_scores = adapter._native_pruning_scores

    def capture_scores(**kwargs):
        score_sources.append((kwargs["decoder_layer"].self_attn.layer_idx, kwargs["hidden_states"].shape[1]))
        return original_scores(**kwargs)

    def capture(request):
        response = callback(request)
        result = response.compression_result if backbone == "qwen3_vl" else response
        payload = request.payload
        reference = payload.signals["projector_reference_features"] if before else payload.features
        oracle = PruneVidVideoKernel()(replace(payload, features=reference), config=activation.config)
        assert result.source_groups == oracle.source_groups
        assert result.representative_indices == oracle.representative_indices
        expected = torch.stack([payload.features[list(group)].mean(0) for group in oracle.source_groups[0]])
        torch.testing.assert_close(result.features, expected)
        if before:
            assert payload.features.shape[-1] != reference.shape[-1]
        if backbone == "qwen3_vl":
            for source, branch in zip(request.deepstack_features, response.deepstack_features):
                torch.testing.assert_close(branch, torch.stack([source[list(g)].mean(0) for g in result.source_groups[0]]))
        observed.append((request, result))
        return response

    monkeypatch.setattr(adapter, "_native_pruning_scores", capture_scores)
    activation._projector_callback = capture
    activation.install()
    decoder = _decoder(model, backbone)
    attention_lengths, mlp_lengths = [[] for _ in decoder.layers], [[] for _ in decoder.layers]
    handles = []
    for index, layer in enumerate(decoder.layers):
        handles += [layer.self_attn.register_forward_pre_hook(lambda m, a, kw, i=index: attention_lengths[i].append(kw["hidden_states"].shape[1]), with_kwargs=True),
            layer.mlp.register_forward_pre_hook(lambda m, a, i=index: mlp_lengths[i].append(a[0].shape[1]))]
    injections = []
    if backbone == "qwen3_vl":
        original_injection = decoder._deepstack_process
        def injection(hidden, mask, branch):
            injections.append((mask.sum().item(), branch.shape[0], branch.requires_grad))
            return original_injection(hidden, mask, branch)
        monkeypatch.setattr(decoder, "_deepstack_process", injection)
    args = _training_inputs(torch, backbone)
    try:
        output = model(**args)
        record = activation.runtime_metadata()["vtp_layout_deltas"][0]
        sequence = record["sequence_layout_delta"]
        old, new = sequence["source_count"], sequence["result_count"]
        assert new < old
        assert score_sources == [(0, old)]
        assert attention_lengths == [[new if position == "initial" or (position == "mid" and i > 0) else old] for i in range(5)]
        assert mlp_lengths == [[old if position == "terminal" else new] for _ in range(5)]
        assert output.logits.shape[1] == new
        assert record["application_position"] == {"boundary": "post_attention_pre_ffn" if position == "mid" else "inter_layer", "index": 5 if position == "terminal" else 0}
        assert record["score_layer"] == 0
        output.loss.backward()
        pixels = args["images"][0] if backbone == "llava_onevision" else args["pixel_values_videos"]
        assert pixels.grad is not None and torch.isfinite(pixels.grad).all() and pixels.grad.abs().sum() > 0
        if backbone == "qwen3_vl":
            expected_count = observed[0][1].result_count if position == "terminal" else record["result_count"]
            assert injections == [(expected_count, expected_count, True)] * 3
            assert all(p.grad is not None and torch.isfinite(p.grad).all() for merger in model.model.visual.deepstack_merger_list for p in merger.parameters())
            assert all(any(p.grad.abs().sum() > 0 for p in merger.parameters()) for merger in model.model.visual.deepstack_merger_list)
        elif backbone == "llava_onevision":
            assert model.model.image_newline.grad is not None and model.model.image_newline.grad.abs().sum() > 0
        assert activation._active.get() is None
        assert activation.runtime_metadata()["projector_position"] == ("before" if before else "after")
    finally:
        for handle in handles:
            handle.remove()
        activation.uninstall()


@pytest.mark.parametrize("backbone", BACKBONES)
def test_native_mid_checkpoint_recomputation_matches_values_and_gradients(backbone):
    torch = pytest.importorskip("torch")
    torch.manual_seed(482)
    base = _model(backbone).train()
    expected = None
    for checkpoint in (None, False, True):
        model = copy.deepcopy(base)
        if checkpoint is not None:
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": checkpoint})
        activation = image_fixture.activation(model, _stage(backbone, "mid", projector="projector_input"))
        activation.install()
        torch.manual_seed(483)
        args = _training_inputs(torch, backbone)
        try:
            output = model(**args)
            output.loss.backward()
            pixels = args["images"][0] if backbone == "llava_onevision" else args["pixel_values_videos"]
            result = (output.logits.detach().clone(), pixels.grad.clone(),
                _decoder(model, backbone).layers[0].mlp.down_proj.weight.grad.clone())
            if expected is None:
                expected = result
            else:
                torch.testing.assert_close(result, expected, atol=3e-6, rtol=3e-5)
            assert activation.runtime_metadata()["vtp_invocations"] == 1
            assert activation._active.get() is None
        finally:
            activation.uninstall()


@pytest.mark.parametrize("backbone", BACKBONES)
def test_native_canonical_default_and_old_physical_alias_are_exact(backbone):
    torch = pytest.importorskip("torch")
    from mllm_comp.compressors.video.positions import video_projector_site
    torch.manual_seed(484)
    base = _model(backbone, canonical=True).train()
    expected = None
    for alias in (False, True):
        model = copy.deepcopy(base)
        stage = _stage(backbone, source=None, projector=video_projector_site(backbone, "after") if alias else None)
        if alias:
            native_site = "llm_decoder_video_tokens_prefill_or_decode" if backbone == "llava_onevision" else "llm_decoder_video_tokens_sparse_mrope" if backbone == "qwen25_vl" else "llm_decoder_video_tokens_sparse_mrope_visual_pos_masks"
            stage.operation["targets"]["vtp_prefill_prune"] = {"target_site": native_site}
        activation = image_fixture.activation(model, stage)
        activation.install()
        torch.manual_seed(485)
        args = _training_inputs(torch, backbone)
        try:
            output = model(**args)
            output.loss.backward()
            pixels = args["images"][0] if backbone == "llava_onevision" else args["pixel_values_videos"]
            actual = (output.logits.detach().clone(), pixels.grad.clone())
            if expected is None:
                expected = actual
            else:
                torch.testing.assert_close(actual, expected, atol=0, rtol=0)
            metadata = activation.runtime_metadata()
            assert metadata["canonical_evidence_applicable"] is True
            assert metadata["application_positions"] == [{"boundary": "inter_layer", "index": 11}]
            assert metadata["vtp_layout_deltas"][0]["score_layer"] == 10
        finally:
            activation.uninstall()


@pytest.mark.parametrize("backbone,position", [("llava_onevision", "initial"), ("qwen25_vl", "mid"), ("qwen3_vl", "terminal")])
def test_native_generate_exact_cache_prefix_and_repeat_requests(backbone, position):
    torch = pytest.importorskip("torch")
    torch.manual_seed(486)
    model = _model(backbone).eval()
    activation = image_fixture.activation(model, _stage(backbone, position, projector="projector_input", phase="generate"))
    activation.install()
    args = _inputs(torch, backbone)
    args["attention_mask"] = torch.ones_like(args["input_ids"])
    if backbone == "llava_onevision":
        args["inputs"] = args.pop("input_ids")
    previous = None
    try:
        for repeat in range(2):
            with torch.no_grad():
                output = model.generate(**args, min_new_tokens=2, max_new_tokens=2, use_cache=True,
                    do_sample=False, eos_token_id=None, pad_token_id=0, return_dict_in_generate=True)
            metadata = activation.runtime_metadata()
            assert metadata["projector_invocations"] == metadata["vtp_invocations"] == repeat + 1
            cache = metadata["cache_repairs"][0]
            assert cache["touched_layers"] == list(range(0 if position == "initial" else 1 if position == "mid" else 5))
            assert [layer.get_seq_length() for layer in output.past_key_values.layers] == [cache["result_length"] + 1] * 5
            if previous is not None:
                torch.testing.assert_close(output.sequences, previous, atol=0, rtol=0)
            previous = output.sequences
            assert activation._active.get() is None
    finally:
        activation.uninstall()


def test_native_install_reports_application_depth_error_without_patches():
    pytest.importorskip("torch")
    from mllm_comp.core.errors import BackendIntegrationError
    model = _model("qwen3_vl")
    stage = _stage("qwen3_vl", "terminal")
    stage.operation["targets"]["vtp_prefill_prune"]["layers"] = [5]
    activation = image_fixture.activation(model, stage)
    original_model_forward = model.forward
    original_decoder_forward = model.model.language_model.forward
    with pytest.raises(BackendIntegrationError, match="outside the 5-layer stack") as raised:
        activation.install()
    assert isinstance(raised.value.__cause__, ValueError)
    assert not activation.installed and not activation._patches
    assert model.forward == original_model_forward
    assert model.model.language_model.forward == original_decoder_forward
    assert not hasattr(model, "_mllm_comp_qwen3_vl_video_bridge")
