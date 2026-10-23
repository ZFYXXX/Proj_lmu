"""LongVU moves DINO frame selection through its original learned SVA/output P."""

import copy
import math
from dataclasses import replace
from types import SimpleNamespace

import pytest

from mllm_comp.backbones.longvu_video import LongVUVideoSignalProvider
from mllm_comp.compressors.video.longvu import longvu_video
from mllm_comp.core.errors import BackendIntegrationError
from mllm_comp.pipeline.stage import CompressionStageConfig
from tests.integration.test_video_dytok_positions_cpu import _model, _inputs
from tests.unit.test_longvu_video_kernel import _payload, _TinyFusion


SITES = ("pre_sva", "pre_output_projector", "post_output_projector")


def _stage(backbone, site=None, phase="forward"):
    return CompressionStageConfig.from_dict({
        "name": "longvu-frame-selection-position", "placement": "projector_level", "execution_phase": phase,
        "plugin": "longvu", "algorithm": "longvu", "variant": "longvu.video",
        "binding": f"longvu.video.{backbone}", "target_modalities": ["video"],
        "operation": {} if site is None else {"targets": {"dino_frame_select_sva_fusion": {
            "frame_selection_site": site,
        }}},
    })


def _fusion(raw_hidden_size=16, output_hidden_size=24):
    from mllm_comp.backbones._longvu_torch import LongVUFusionModule
    return LongVUFusionModule(raw_hidden_size=raw_hidden_size, dino_hidden_size=5,
        output_hidden_size=output_hidden_size, vision_hidden_size=16, connector_depth=3)


class _Dino:
    """Fixed frame-score fixture; the native raw/pixel capture is real."""
    hidden_size = 5

    def __call__(self, pixels):
        import torch
        return torch.ones(pixels.shape[0], 576, 5, device=pixels.device, dtype=pixels.dtype)


def _executor(model, backbone, stage, fusion=None):
    from mllm_comp.compressors.multimodal_registry import builtin_multimodal_compression_contracts
    binding = builtin_multimodal_compression_contracts().binding(stage.binding)
    parameter = next(model.parameters())
    fusion = _fusion().to(device=parameter.device, dtype=parameter.dtype) if fusion is None else fusion
    provider = LongVUVideoSignalProvider(backbone=backbone, model=model, dino_encoder=_Dino())
    return binding.executor_factory(model=model, runtime=SimpleNamespace(metadata={}), stage=stage,
        binding=binding, execution_phase=stage.execution_phase, signal_provider=provider, fusion_module=fusion)


def _legacy_forward(module, raw_features, dino_features):
    """Original forward algebra, with no frame-selection arguments or branches."""
    raw, dino = module.raw_projector(raw_features), module.dino_projector(dino_features)
    frames = raw.shape[0]
    context = raw.mean(1)[:, None, None].expand(-1, 144, 1, -1).reshape(frames * 144, 1, -1)
    def cells(values):
        return values.reshape(frames, 12, 2, 12, 2, -1).permute(0, 1, 3, 2, 4, 5).reshape(frames * 144, 4, -1)
    queries = module.vision_query.view(1, 1, -1).expand(frames * 144, 1, -1)
    for layer in module.layers:
        queries = layer(queries, context, cells(raw), cells(dino))
    return module.output_projector(queries.squeeze(1)).reshape(frames, 144, -1)


def _resample(torch, raw, frames):
    side = math.isqrt(raw.shape[0] // frames)
    rasters = raw.reshape(frames, side, side, -1).permute(0, 3, 1, 2)
    return torch.nn.functional.interpolate(rasters.float(), size=(24, 24), mode="bilinear",
        align_corners=False).to(raw.dtype).permute(0, 2, 3, 1).reshape(frames, 576, -1)


def _selected(torch, dino):
    # Independent original one-window cosine mean, middle forced, threshold.
    rows = dino.flatten(1)
    rows = rows / rows.norm(dim=1, keepdim=True)
    scores = (rows @ rows.T).mean(1)
    scores[len(rows) // 2] = 0
    return torch.where(scores < 0.83)[0], scores


def _frame_hooks(module, observed):
    return [module.raw_projector.register_forward_pre_hook(
                lambda module, args: observed.append(("raw", args[0].shape[0]))),
            module.layers[0].register_forward_pre_hook(
                lambda module, args: observed.append(("sva", args[0].shape[0] // 144))),
            module.output_projector.register_forward_pre_hook(
                lambda module, args: observed.append(("projector", args[0].shape[0] // 144)))]


def test_original_sva_formula_frame_selection_cost_gradients_and_strict_state_reload():
    torch = pytest.importorskip("torch")
    torch.manual_seed(461)
    template = _payload(frame_count=3)
    raw = torch.randn_like(template.raw_features)
    base = _fusion(3, 4)
    saved_state = copy.deepcopy(base.state_dict())
    reference = _fusion(3, 4)
    reference.load_state_dict(saved_state, strict=True)
    reference_raw = raw.clone().requires_grad_(True)
    selected, scores = _selected(torch, template.signals["dinov2_features"])
    assert selected.tolist() == [1]
    expected = _legacy_forward(reference, _resample(torch, reference_raw, 3)[selected],
        template.signals["dinov2_features"][selected]).reshape(144, 4)
    expected.square().sum().backward()
    expected_gradients = {name: parameter.grad for name, parameter in reference.named_parameters()}
    default = None
    for site in (None, *SITES):
        module = _fusion(3, 4)
        loaded = module.load_state_dict(saved_state, strict=True)
        assert not loaded.missing_keys and not loaded.unexpected_keys
        assert tuple(module.state_dict()) == tuple(saved_state)
        values = raw.clone().requires_grad_(True)
        payload = replace(template, raw_features=values)
        observed = []
        hooks = _frame_hooks(module, observed)
        try:
            result = longvu_video(payload, fusion_module=module,
                **({} if site is None else {"frame_selection_site": site}))
        finally:
            for hook in hooks:
                hook.remove()
        sva_frames = 1 if site in (None, "pre_sva") else 3
        projected_frames = 3 if site == "post_output_projector" else 1
        assert observed == [("raw", sva_frames), ("sva", sva_frames), ("projector", projected_frames)]
        torch.testing.assert_close(result.features, expected, atol=2e-6, rtol=2e-5)
        assert result.metadata["selected_frame_indices_by_video"] == ((1,),)
        torch.testing.assert_close(torch.tensor(result.metadata["frame_similarity_scores_by_video"][0][0]), scores)
        assert result.result_count == 144 and len(result.source_groups[0]) == 144
        assert sorted(i for group in result.source_groups[0] for i in group) == list(range(196, 392))
        result.features.square().sum().backward()
        gradients = {name: parameter.grad for name, parameter in module.named_parameters()}
        torch.testing.assert_close(values.grad, reference_raw.grad, atol=2e-6, rtol=2e-5)
        torch.testing.assert_close(gradients, expected_gradients, atol=3e-6, rtol=3e-5)
        assert all(value is not None and torch.isfinite(value).all() for value in gradients.values())
        assert gradients["output_projector.0.weight"].abs().sum() > 0
        assert gradients["vision_query"].abs().sum() > 0
        assert values.grad.reshape(3, -1)[[0, 2]].abs().sum() == 0
        if site is None:
            default = result.features.detach(), values.grad, gradients, result.source_groups
        else:
            assert result.source_groups == default[3]
            if site == "pre_sva":
                torch.testing.assert_close((result.features.detach(), values.grad, gradients), default[:3], rtol=0, atol=0)
        metadata = result.metadata
        assert metadata["frame_selection_site"] == (site or "pre_sva")
        assert metadata["sva_site"] == "pre_output_projector"
        assert metadata["sva_frame_counts_by_video"] == (sva_frames,)
        assert metadata["output_projector_frame_counts_by_video"] == (projected_frames,)
        assert metadata["extra_sva_frames_by_video"] == (sva_frames - 1,)
        assert metadata["extra_output_projector_frames_by_video"] == (projected_frames - 1,)


def test_custom_fusion_keeps_original_default_call_and_rejects_unsupported_movement():
    payload, fusion = _payload(), _TinyFusion()
    assert longvu_video(payload, fusion_module=fusion).result_count == 144
    for site in SITES[1:]:
        with pytest.raises(BackendIntegrationError, match="does not support moved frame selection"):
            longvu_video(payload, fusion_module=fusion, frame_selection_site=site)


@pytest.mark.parametrize("backbone", ["llava_onevision", "qwen25_vl", "qwen3_vl"])
def test_native_longvu_fixed_source_actual_sva_positions_gradients_default_and_repeated_generate(backbone):
    torch = pytest.importorskip("torch")
    torch.manual_seed(462)
    base = _model(backbone).train()
    base_fusion = _fusion()
    saved_state = copy.deepcopy(base_fusion.state_dict())
    baseline = None
    for site in (None, *SITES):
        model = copy.deepcopy(base)
        fusion = copy.deepcopy(base_fusion)
        reference = copy.deepcopy(base_fusion)
        activation = _executor(model, backbone, _stage(backbone, site), fusion)
        activation.load_compression_state_dict(saved_state)
        observed, captures = [], []
        hooks = _frame_hooks(fusion, observed)
        original = activation._projector_callback

        def callback(request):
            payload = request.payload
            selection, _ = _selected(torch, payload.signals["dinov2_features"])
            assert selection.tolist() == [1]
            assert payload.features.shape == (392, 24)
            rasters = _resample(torch, payload.raw_features, 2)
            expected = _legacy_forward(reference, rasters[selection],
                payload.signals["dinov2_features"][selection]).reshape(144, 24)
            returned = original(request)
            main = returned.compression_result if backbone == "qwen3_vl" else returned
            torch.testing.assert_close(main.features, expected, atol=2e-6, rtol=2e-5)
            assert main.result_count == 144 and main.metadata["selected_frame_indices_by_video"] == ((1,),)
            if backbone == "qwen3_vl":
                for source, actual in zip(request.deepstack_features, returned.deepstack_features):
                    expected_branch = torch.stack([source[list(group)].mean(0) for group in main.source_groups[0]])
                    torch.testing.assert_close(actual, expected_branch)
            captures.append((payload, main))
            return returned

        activation._projector_callback = callback
        torch.manual_seed(463)
        args = _inputs(torch, backbone)
        try:
            activation.install()
            assert model.mllm_comp_longvu_fusion is fusion
            assert {id(p) for p in fusion.parameters()} <= {id(p) for p in model.parameters()}
            output = model(**args)
            assert len(captures) == 1
            expected_sva = 1 if site in (None, "pre_sva") else 2
            expected_p = 2 if site == "post_output_projector" else 1
            assert observed == [("raw", expected_sva), ("sva", expected_sva), ("projector", expected_p)]
            output.loss.backward()
            pixels = args["images"][0] if backbone == "llava_onevision" else args["pixel_values_videos"]
            gradients = {name: parameter.grad for name, parameter in fusion.named_parameters()}
            assert pixels.grad is not None and torch.isfinite(pixels.grad).all() and pixels.grad.abs().sum() > 0
            assert all(value is not None and torch.isfinite(value).all() for value in gradients.values())
            assert gradients["output_projector.0.weight"].abs().sum() > 0
            payload, result = captures[0]
            if baseline is None:
                baseline = output.logits.detach(), pixels.grad, gradients, result.source_groups, payload.raw_features.detach()
            else:
                assert result.source_groups == baseline[3]
                torch.testing.assert_close(payload.raw_features, baseline[4], rtol=0, atol=0)
                torch.testing.assert_close((output.logits.detach(), pixels.grad, gradients), baseline[:3],
                    atol=0 if site == "pre_sva" else 3e-6, rtol=0 if site == "pre_sva" else 3e-5)
            metadata = activation.runtime_metadata()
            assert metadata["frame_selection_site"] == (site or "pre_sva")
            assert metadata["sva_site"] == "pre_output_projector"
            assert metadata["algorithm_records"][0]["extra_sva_frames_by_video"] == (expected_sva - 1,)
            assert metadata["projector_layout_deltas"][0]["output_projector_frame_counts_by_video"] == (expected_p,)
            if site in SITES[1:]:
                assert not metadata["canonical_evidence_applicable"] and not metadata["real_checkpoint_verified"]
        finally:
            for hook in hooks:
                hook.remove()
            activation.uninstall()
        assert not hasattr(model, "mllm_comp_longvu_fusion")
        if site not in SITES[1:]:
            continue
        model.eval()
        generation = _executor(model, backbone, _stage(backbone, site, "generate"), fusion)
        outputs = []
        try:
            generation.install()
            if backbone == "llava_onevision":
                call = dict(inputs=args["input_ids"], images=[args["images"][0].detach()],
                    image_sizes=args["image_sizes"], modalities=["video"], attention_mask=args["attention_mask"])
            else:
                call = {key: value.detach() if torch.is_tensor(value) else value
                    for key, value in args.items() if key not in {"labels", "use_cache"}}
            for repeat in range(2):
                with torch.no_grad():
                    generated = model.generate(**call, min_new_tokens=2, max_new_tokens=2, do_sample=False,
                        use_cache=True, return_dict_in_generate=True, eos_token_id=None, pad_token_id=0)
                metadata = generation.runtime_metadata()
                assert metadata["projector_invocations"] == repeat + 1
                assert len(metadata["algorithm_records"]) == 1
                assert metadata["algorithm_records"][0]["frame_selection_site"] == site
                assert generated.past_key_values.get_seq_length() == output.logits.shape[1] + 1
                outputs.append(generated.sequences)
            torch.testing.assert_close(outputs[0], outputs[1], rtol=0, atol=0)
        finally:
            generation.uninstall()
