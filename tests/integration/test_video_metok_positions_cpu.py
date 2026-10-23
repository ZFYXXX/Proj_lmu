"""METok fixed projected plans and historical QK scores at actual boundaries."""

import copy
import math
from dataclasses import replace
from types import SimpleNamespace

import pytest

import mllm_comp.compressors.video.metok as metok_kernel
from mllm_comp.backbones import video_decoder as decoder_execution

import mllm_comp.backbones.metok_video as provider
from mllm_comp.compressors.video.metok import metok_video
from mllm_comp.pipeline.stage import CompressionStageConfig
from mllm_comp.core.errors import BackendIntegrationError
from tests.integration.test_video_dytok_positions_cpu import _model as _tiny_model, _inputs


def _model(backbone, *, canonical=False):
    source = _tiny_model(backbone)
    config = copy.deepcopy(source.config)
    text = config if backbone == "llava_onevision" else config.text_config
    text.num_hidden_layers = 20 if canonical else 5
    if hasattr(text, "layer_types"):
        text.layer_types = ["full_attention"] * text.num_hidden_layers
    model = type(source)(config)
    if backbone == "llava_onevision":
        model.model.vision_tower = source.model.vision_tower
        model.model.mm_projector = source.model.mm_projector
        model.model.image_newline = source.model.image_newline
    return model


def _stage(backbone, mode="mid", phase="forward"):
    targets = {}
    sources = [0, 1, 3] if mode in {"default", "alias"} else [0, 1, 19]
    if mode not in {"default", "canonical", "canonical_alias"}:
        targets["event_pool"] = {"target_site": "projector_output" if mode == "alias" else "projector_input"}
        targets["prefill_attention_prune"] = {
            "target_site": "llm_decoder_layer_post_attention_video_tokens" if mode == "mid" else "llm_decoder_layer_output_video_tokens",
            "layers": [0, 1, 2] if mode == "mid" else [0, 1, 3] if mode == "alias" else [1, 2, 3] if mode == "same" else [2, 3, 5],
            **({"layer_edge": "before"} if mode != "mid" else {}),
        }
    if mode == "canonical_alias":
        old_sites = {
            "llava_onevision": "llm_decoder_video_tokens_prefill_or_decode",
            "qwen25_vl": "llm_decoder_video_tokens_sparse_mrope",
            "qwen3_vl": "llm_decoder_video_tokens_sparse_mrope_visual_pos_masks",
        }
        targets["prefill_attention_prune"] = {"target_site": old_sites[backbone]}
    return CompressionStageConfig.from_dict({
        "name": "metok-video-position", "placement": "projector_level", "execution_phase": phase,
        "plugin": "metok", "algorithm": "metok", "variant": "metok.video",
        "binding": f"metok.video.{backbone}", "target_modalities": ["video"],
        "operation": {"targets": targets}, "config": {} if mode in {"canonical", "canonical_alias"} else {"pruning_layers": sources},
    })


class _Siglip:
    def pixel_frame_similarity(self, text, pixels):
        import torch
        assert text == "Describe the motion."
        return torch.tensor([0.9, 0.7], device=pixels.device, dtype=pixels.dtype)

    def native_frame_similarity(self, text, raw, *, frame_count, tokens_per_frame):
        assert frame_count == 2 and raw.shape[0] == frame_count * tokens_per_frame
        return self.pixel_frame_similarity(text, raw)


def _executor(model, backbone, stage):
    from mllm_comp.compressors.multimodal_registry import builtin_multimodal_compression_contracts
    binding = builtin_multimodal_compression_contracts().binding(stage.binding)
    return binding.executor_factory(model=model, runtime=SimpleNamespace(metadata={}), stage=stage,
        binding=binding, execution_phase=stage.execution_phase,
        signal_provider=provider.METokVideoSignalProvider(backbone=backbone, model=model, siglip_encoder=_Siglip()))


def _context(backbone):
    return {"conditioning_text": "Describe the motion."} if backbone == "llava_onevision" else {
        "video_conditioning_texts": ["Describe the motion."]}


def _bilinear(torch, source, result):
    rows = []
    for frame in result.metadata["metok_frames"][0]:
        h, w = frame["source_shape"]
        offset = frame["frame_index"] * h * w
        values = source[offset:offset + h * w].reshape(h, w, -1).permute(2, 0, 1)[None]
        rows.append(torch.nn.functional.interpolate(values, size=frame["result_shape"], mode="bilinear",
            align_corners=False)[0].permute(1, 2, 0).reshape(-1, source.shape[-1]))
    return torch.cat(rows)


def _check_run(torch, backbone, model, activation, args, monkeypatch):
    decoder = provider._decoder_owner(model, backbone)
    layers = list(decoder.layers)
    capture_scores, plans, requests = [], [], []
    current_ids = []
    original_scores = decoder_execution._native_pruning_scores
    original_plan = metok_kernel.plan_metok_pruning_stage
    original_callback = activation._projector_callback
    original_branches = []
    projector = model.model.mm_projector if backbone == "llava_onevision" else model.model.visual.merger
    native_projector = copy.deepcopy(projector)
    native_inputs = []
    projector_hook = projector.register_forward_pre_hook(lambda module, values: native_inputs.append(values[0]))

    def callback(request):
        payload = request.payload
        reference = payload.signals.get("projector_reference_features", payload.features)
        dense_raw = native_inputs[0]
        with torch.no_grad():
            dense_projected = native_projector(dense_raw)
            if backbone == "llava_onevision":
                from tests.integration.test_video_vqtoken_llava_onevision_positions_cpu import _pool
                expected_projected = _pool(torch, dense_projected, "bilinear")
                expected_raw = _pool(torch, dense_raw, "bilinear")
            else:
                expected_projected, expected_raw = dense_projected, dense_raw.reshape(392, 64)
                if backbone == "qwen25_vl":
                    window, _ = model.model.visual.get_window_index(request.source_video_grid_thw)
                    assert not torch.equal(window, torch.arange(392))
                    expected_projected, expected_raw = expected_projected[window.argsort()], expected_raw[window.argsort()]
            torch.testing.assert_close(reference, expected_projected)
            torch.testing.assert_close(payload.features, expected_raw if activation.projector_position == "before" else expected_projected)
        expected_plan = metok_video(replace(payload, features=reference), config=activation.config)
        response = original_callback(request)
        result = response.compression_result if backbone == "qwen3_vl" else response
        assert result.source_groups == expected_plan.source_groups
        assert result.metadata["metok_segments"] == expected_plan.metadata["metok_segments"]
        torch.testing.assert_close(result.features, _bilinear(torch, payload.features, expected_plan))
        if activation.projector_position == "before":
            assert reference.shape[1] == 24 and payload.features.shape[1] in (16, 64)
            assert result.metadata["projected_plan_reference_tokens"] == payload.token_count
        if backbone == "qwen3_vl":
            for source, branch in zip(request.deepstack_features, response.deepstack_features):
                torch.testing.assert_close(branch, _bilinear(torch, source, expected_plan))
                branch.retain_grad()
                original_branches.append(branch)
        current_ids[:] = range(result.result_count)
        requests.append((request, result))
        return response

    def scores(**kw):
        layer = layers.index(kw["decoder_layer"])
        assert layer == activation.pruning_layers[len(capture_scores)]
        value = original_scores(**kw)
        with torch.no_grad():
            visual = torch.tensor(kw["visual_positions"], device=kw["hidden_states"].device)
            queries = torch.tensor(kw["query_positions"], device=kw["hidden_states"].device)
            q, _, k = kw["qk_function"](kw["decoder_layer"], kw["hidden_states"],
                instruction_indices=queries, image_indices=visual, position_embeddings=kw["position_embeddings"])
            visibility = kw.get(
                "visibility_function",
                decoder_execution.pyramid_attention_key_visibility,
            )(
                kw["attention_mask"],
                instruction_indices=queries,
                key_count=k.shape[0],
                padding_mask=kw["padding_mask"],
            )
            logits = torch.einsum("qhd,khd->hqk", q, k) / math.sqrt(q.shape[-1])
            if visibility is not None:
                logits = logits.masked_fill(~visibility.to(logits.device).bool()[None], float("-inf"))
            expected = logits.softmax(-1, dtype=torch.float32).to(logits.dtype).mean((0, 1))[visual]
            torch.testing.assert_close(value, expected)
        capture_scores.append((value.detach().clone(), tuple(current_ids), kw["hidden_states"].shape[1]))
        return value

    def plan(**kw):
        stage = kw["stage_index"]
        if stage < 2:
            historical, ids, _ = capture_scores[stage]
            expected = historical[[ids.index(index) for index in current_ids]]
            torch.testing.assert_close(kw["visual_scores"], expected, rtol=0, atol=0)
        else:
            assert len(capture_scores) == 2 and kw["visual_scores"].count_nonzero() == 0
        expected_kept, cursor = [], 0
        for event, count in enumerate(kw["event_lengths"]):
            keep = 0 if stage == 2 else math.ceil(0.76 * count) if event in kw["important_event_indices"] else math.ceil(0.76 * count / 2) if stage == 0 else 0
            if keep:
                indices = kw["visual_scores"][cursor:cursor + count].argsort(descending=True, stable=True)[:keep].sort().values
                expected_kept.extend(cursor + int(index) for index in indices)
            cursor += count
        result = original_plan(**kw)
        assert result.kept_visual_indices == tuple(expected_kept)
        current_ids[:] = result.retained_projector_indices
        plans.append(result)
        return result

    activation._projector_callback = callback
    monkeypatch.setattr(decoder_execution, "_native_pruning_scores", scores)
    monkeypatch.setattr(metok_kernel, "plan_metok_pruning_stage", plan)
    mlp_counts = [[] for _ in layers]
    attention_counts = [[] for _ in layers]
    hooks = [layer.mlp.register_forward_pre_hook(lambda module, values, i=i: mlp_counts[i].append(values[0].shape[1])) for i, layer in enumerate(layers)]
    hooks += [layer.self_attn.register_forward_pre_hook(lambda module, values, kw, i=i: attention_counts[i].append(kw["hidden_states"].shape[1]), with_kwargs=True) for i, layer in enumerate(layers)]
    try:
        activation.install()
        with activation._bridge.request_context(_context(backbone)):
            output = model(**args)
        assert len(plans) == 3 and len(capture_scores) == 2
        assert plans[-1].visual_layout_delta.result_count == 0
        for layer_index in range(len(layers)):
            before = [p for p in plans if p.metadata["application_position"]["index"] < layer_index or (
                p.metadata["application_position"]["index"] == layer_index and p.metadata["application_position"]["boundary"] == "inter_layer")]
            after = [p for p in plans if p.metadata["application_position"]["index"] <= layer_index]
            original_length = plans[0].sequence_layout_delta.source_count
            assert attention_counts[layer_index][0] == (before[-1].sequence_layout_delta.result_count if before else original_length)
            assert mlp_counts[layer_index][0] == (after[-1].sequence_layout_delta.result_count if after else original_length)
        output.loss.backward()
        assert len(plans) == 3 and len(capture_scores) == 2
        pixels = args["images"][0] if backbone == "llava_onevision" else args["pixel_values_videos"]
        assert pixels.grad is not None and pixels.grad.abs().sum() > 0 and torch.isfinite(pixels.grad).all()
        assert all(layer.mlp.gate_proj.weight.grad is not None for layer in layers)
        if backbone == "qwen3_vl":
            assert all(branch.grad is not None and torch.isfinite(branch.grad).all() and branch.grad.abs().sum() > 0 for branch in original_branches[:2])
            assert all(torch.isfinite(branch.grad).all() for branch in original_branches if branch.grad is not None)
        metadata = activation.runtime_metadata()
        assert metadata["score_layers"] == list(activation.pruning_layers[:2])
        canonical = activation.projector_position == "after" and activation.pruning_layers[:2] == (3, 10) and tuple(
            (point.boundary, point.index) for point in activation.application_positions
        ) == (("inter_layer", 3), ("inter_layer", 10), ("inter_layer", 19))
        assert metadata["canonical_evidence_applicable"] is canonical
        assert [r["application_position"] for r in metadata["pruning_layout_deltas"]] == [p.as_dict() for p in activation.application_positions]
        assert [r["boundary_layer"] for r in metadata["pruning_layout_deltas"]] == list(activation.pruning_layers)
        if activation.application_positions[0].index == 2:
            assert capture_scores[0][2] == capture_scores[1][2]
        elif activation.application_positions[0].index == 1:
            assert capture_scores[0][2] > capture_scores[1][2]
        return output.logits.detach(), pixels.grad, tuple(p.kept_visual_indices for p in plans), {
            name: parameter.grad for name, parameter in model.named_parameters() if parameter.grad is not None}
    finally:
        for hook in hooks:
            hook.remove()
        projector_hook.remove()
        activation.uninstall()
        monkeypatch.setattr(
            decoder_execution, "_native_pruning_scores", original_scores
        )
        monkeypatch.setattr(metok_kernel, "plan_metok_pruning_stage", original_plan)


@pytest.mark.parametrize("backbone", ["llava_onevision", "qwen25_vl", "qwen3_vl"])
def test_native_fixed_sources_delayed_mapping_projector_bilinear_and_actual_boundaries(backbone, monkeypatch):
    torch = pytest.importorskip("torch")
    torch.manual_seed(471)
    base = _model(backbone).train()
    default = None
    for mode in ("default", "alias", "delayed", "mid", *(("same",) if backbone == "qwen25_vl" else ())):
        model = copy.deepcopy(base)
        torch.manual_seed(472)
        args = _inputs(torch, backbone)
        activation = _executor(model, backbone, _stage(backbone, mode))
        output = _check_run(torch, backbone, model, activation, args, monkeypatch)
        if mode == "default":
            default = output
        elif mode == "alias":
            torch.testing.assert_close(output, default, rtol=0, atol=0)


def test_qwen3_early_mid_preserves_pending_deepstack_gradients_with_both_checkpoints(monkeypatch):
    torch = pytest.importorskip("torch")
    torch.manual_seed(473)
    base = _model("qwen3_vl").train()
    expected = None
    for checkpoint in (None, False, True):
        model = copy.deepcopy(base)
        if checkpoint is not None:
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": checkpoint})
        torch.manual_seed(474)
        args = _inputs(torch, "qwen3_vl")
        activation = _executor(model, "qwen3_vl", _stage("qwen3_vl", "mid"))
        result = _check_run(torch, "qwen3_vl", model, activation, args, monkeypatch)
        if expected is None:
            expected = result
        else:
            torch.testing.assert_close(result, expected, atol=2e-6, rtol=2e-5)


@pytest.mark.parametrize("backbone,mode", [(b, m) for b in ("llava_onevision", "qwen25_vl", "qwen3_vl") for m in ("delayed", "mid")])
def test_native_cache_uses_actual_completed_prefix_and_repeated_generation(backbone, mode):
    torch = pytest.importorskip("torch")
    torch.manual_seed(475)
    model = _model(backbone).eval()
    args = _inputs(torch, backbone)
    activation = _executor(model, backbone, _stage(backbone, mode, phase="generate"))
    try:
        activation.install()
        if backbone == "llava_onevision":
            call = dict(inputs=args["input_ids"], images=[args["images"][0].detach()], image_sizes=args["image_sizes"],
                modalities=["video"], attention_mask=args["attention_mask"])
        else:
            call = {key: value.detach() if torch.is_tensor(value) else value for key, value in args.items() if key not in {"labels", "use_cache"}}
        outputs = []
        for repeat in range(2):
            with torch.no_grad(), activation._bridge.request_context(_context(backbone)):
                output = model.generate(**call, min_new_tokens=2, max_new_tokens=2, do_sample=False,
                    use_cache=True, return_dict_in_generate=True, eos_token_id=None, pad_token_id=0)
            metadata = activation.runtime_metadata()
            assert metadata["projector_invocations"] == repeat + 1
            assert metadata["pruning_invocations"] == 3 * (repeat + 1)
            assert len(metadata["pruning_layout_deltas"]) == 3
            for record, point in zip(metadata["cache_repairs"], activation.application_positions):
                assert record["touched_layers"] == list(range(point.index + (point.boundary == "post_attention_pre_ffn")))
            final_count = metadata["pruning_layout_deltas"][-1]["sequence_layout_delta"]["result_count"]
            for layer in output.past_key_values.layers:
                assert layer.keys.shape[-2] == final_count + 1
            assert activation._active.get() is None
            outputs.append(output.sequences)
        torch.testing.assert_close(outputs[0], outputs[1], rtol=0, atol=0)
    finally:
        activation.uninstall()


@pytest.mark.parametrize("backbone", ["llava_onevision", "qwen25_vl", "qwen3_vl"])
def test_twenty_layer_canonical_defaults_and_old_physical_alias_are_exact(backbone, monkeypatch):
    torch = pytest.importorskip("torch")
    torch.manual_seed(476)
    base = _model(backbone, canonical=True).train()
    expected = None
    for mode in ("canonical", "canonical_alias"):
        model = copy.deepcopy(base)
        torch.manual_seed(477)
        args = _inputs(torch, backbone)
        stage = _stage(backbone, mode)
        assert stage.config == {}
        activation = _executor(model, backbone, stage)
        assert activation.pruning_layers == (3, 10, 19)
        assert len(provider._decoder_owner(model, backbone).layers) == 20
        result = _check_run(torch, backbone, model, activation, args, monkeypatch)
        if expected is None:
            expected = result
        else:
            torch.testing.assert_close(result, expected, rtol=0, atol=0)


def test_actual_decoder_depth_error_preserves_backend_exception_contract():
    torch = pytest.importorskip("torch")
    model = _model("qwen25_vl").eval()
    stage = _stage("qwen25_vl", "mid")
    stage.operation["targets"]["prefill_attention_prune"]["layers"] = [0, 1, 5]
    activation = _executor(model, "qwen25_vl", stage)
    try:
        activation.install()
        with activation._bridge.request_context(_context("qwen25_vl")), pytest.raises(BackendIntegrationError, match="5-layer stack"):
            model(**_inputs(torch, "qwen25_vl"))
        assert activation.runtime_metadata()["pruning_invocations"] == 0
        assert activation._active.get() is None
    finally:
        activation.uninstall()
