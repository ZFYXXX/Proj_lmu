"""DyToK relocates actual video values and captures selected detector layers."""

import copy
import math
from types import SimpleNamespace

import pytest

from mllm_comp.pipeline.stage import CompressionStageConfig
from mllm_comp.backbones.dytok_video import DyToKVideoExecutor

from tests.fixtures import native_image as image_fixture
from tests.integration.test_video_vqtoken_llava_onevision_positions_cpu import _video_model, _pool


def _stage(backbone, position=None, *, layers=None, phase="forward"):
    targets = {}
    if position is not None:
        targets["dynamic_visionzip_reduce"] = {"target_site": position}
    if layers is not None:
        targets["detection_budget_provider"] = {"layers": list(layers)}
    return CompressionStageConfig.from_dict({
        "name": "dytok-video-position", "placement": "projector_level", "execution_phase": phase,
        "plugin": "dytok", "algorithm": "dytok", "variant": "dytok.video",
        "binding": f"dytok.video.{backbone}", "target_modalities": ["video"],
        "operation": {"targets": targets},
    })


def _model(backbone):
    if backbone == "llava_onevision":
        return _video_model()
    return image_fixture.qwen_model(backbone)


def _inputs(torch, backbone):
    if backbone == "llava_onevision":
        from tests.integration.test_video_vqtoken_llava_onevision_positions_cpu import _inputs as values
    elif backbone == "qwen25_vl":
        from tests.integration.test_video_signal_projector_qwen25_vl_positions_cpu import _inputs as values
    else:
        from tests.integration.test_video_signal_projector_qwen3_vl_positions_cpu import _inputs as values
    result = values(torch)
    if backbone != "llava_onevision":
        result["pixel_values_videos"].requires_grad_(True)
        result["labels"] = result["input_ids"].masked_fill(result["input_ids"] == 11, -100)
        result["use_cache"] = False
    return result


class _FrameWeights:
    """Bridge fixture only; real detector Attention is exercised below."""
    def __init__(self):
        self.calls = []

    def configure_target_layers(self, target_layers):
        self.target_layers = target_layers

    def frame_importance(self, text, frames):
        import torch
        assert text == "Describe the motion."
        assert frames.shape[0:2] == (2, 3)
        self.calls.append(frames.detach().clone())
        return torch.tensor([2.0, 3.0], device=frames.device)


def _executor(model, backbone, stage, runner):
    from mllm_comp.compressors.multimodal_registry import builtin_multimodal_compression_contracts
    binding = builtin_multimodal_compression_contracts().binding(stage.binding)
    return binding.executor_factory(model=model,
        runtime=SimpleNamespace(metadata={}), stage=stage, binding=binding,
        execution_phase=stage.execution_phase, detection_runner=runner)


def _values(torch, source, result):
    rows = []
    for group, target in zip(result.source_groups[0], result.metadata["merge_target_indices_by_video"][0]):
        assigned = [index for index in group if index != target]
        rows.append(source[target] + source[assigned].mean(0) if assigned else source[target])
    return torch.stack(rows)


@pytest.mark.parametrize("backbone", ["qwen25_vl", "qwen3_vl", "llava_onevision"])
def test_dytok_native_actual_values_fixed_signals_dynamic_math_default_and_generate(backbone):
    torch = pytest.importorskip("torch")
    torch.manual_seed(443)
    base = _model(backbone).train()
    baseline = None
    for position in (None, "projector_output", "projector_input"):
        model = copy.deepcopy(base)
        before = position == "projector_input"
        runner = _FrameWeights()
        # Preserve the old default/alias setup. The new pre path also exercises
        # nondefault auxiliary module IDs; real capture is verified below.
        selected_layers = (0, 2) if before else tuple(range(16, 24))
        stage = _stage(backbone, position, layers=(0, 2) if before else None)
        activation = _executor(model, backbone, stage, runner)
        projectors = ([model.model.mm_projector] if backbone == "llava_onevision" else
            [model.model.visual.merger, *getattr(model.model.visual, "deepstack_merger_list", ())])
        native_projectors = [copy.deepcopy(projector) for projector in projectors]
        raw_inputs = [[] for _ in projectors]
        outputs = [[] for _ in projectors]
        hooks = []
        for index, projector in enumerate(projectors):
            hooks.append(projector.register_forward_pre_hook(
                lambda module, args, index=index: raw_inputs[index].append(args[0])))
            hooks.append(projector.register_forward_hook(
                lambda module, args, output, index=index: outputs[index].append(output)))
        original = activation._projector_callback
        captured = []

        def callback(request):
            actual = [request.payload.features, *getattr(request, "deepstack_features", ())]
            for index, value in enumerate(actual):
                raw = raw_inputs[index][0]
                if backbone == "llava_onevision":
                    expected = _pool(torch, raw if before else native_projectors[index](raw), "bilinear")
                else:
                    expected = raw.reshape(392, 64) if before else native_projectors[index](raw)
                    if backbone == "qwen25_vl":
                        window, _ = model.model.visual.get_window_index(request.source_video_grid_thw)
                        assert not torch.equal(window, torch.arange(392))
                        expected = expected[window.argsort()]
                torch.testing.assert_close(value, expected)
            result = original(request)
            main = result.compression_result if backbone == "qwen3_vl" else result
            branches = [main.features, *getattr(result, "deepstack_features", ())]
            for source, branch in zip(actual, branches):
                torch.testing.assert_close(branch, _values(torch, source, main))
            assert main.result_count == 294
            budgets = main.metadata["total_budgets_by_video"][0]
            assert sum(budgets) == 294 and budgets[0] < budgets[1]
            assert main.metadata["target_layers"] == selected_layers
            assert tuple(main.metadata["canonical_config"]["llm_config"]["target_layers"]) == tuple(range(16, 24))
            torch.testing.assert_close(request.payload.signals["dytok.frame_importance_by_video"][0],
                torch.tensor([0.4, 0.6], device=request.payload.features.device))
            captured.append((request, main, branches))
            return result

        activation._projector_callback = callback
        torch.manual_seed(444)
        args = _inputs(torch, backbone)
        try:
            activation.install()
            context = ({"conditioning_text": "Describe the motion."} if backbone == "llava_onevision"
                       else {"video_conditioning_texts": ["Describe the motion."]})
            with activation._bridge.request_context(context):
                output = model(**args)
            request, main, branches = captured[0]
            if before:
                for index, branch in enumerate(branches):
                    patches = branch if backbone == "llava_onevision" else branch.reshape(-1, 16)
                    torch.testing.assert_close(raw_inputs[index][-1], patches)
                    torch.testing.assert_close(outputs[index][-1], native_projectors[index](patches))
            output.loss.backward()
            pixels = args["images"][0] if backbone == "llava_onevision" else args["pixel_values_videos"]
            assert pixels.grad is not None and pixels.grad.abs().sum() > 0 and torch.isfinite(pixels.grad).all()
            gradients = [parameter.grad for projector in projectors for parameter in projector.parameters()]
            assert all(value is not None and torch.isfinite(value).all() for value in gradients)
            cues = (request.payload.signals["vision_encoder_attention"][-2].detach(),
                request.payload.signals["vision_encoder_hidden_states"][-2].detach(), runner.calls[0])
            if baseline is None:
                baseline = output.logits.detach(), pixels.grad, cues, main.source_groups
            else:
                torch.testing.assert_close(cues, baseline[2], rtol=0, atol=0)
                assert main.source_groups == baseline[3]
                if not before:
                    torch.testing.assert_close((output.logits.detach(), pixels.grad), baseline[:2], rtol=0, atol=0)
            metadata = activation.runtime_metadata()
            assert metadata["detection_layers"] == runner.target_layers == selected_layers
            assert metadata["canonical_evidence_applicable"] is (not before)
            assert metadata["projector_position"] == ("before" if before else "after")
            assert metadata["algorithm_records"][0]["target_layers"] == selected_layers
        finally:
            for hook in hooks:
                hook.remove()
            activation.uninstall()
        if not before:
            continue
        model.eval()
        runner = _FrameWeights()
        generation = _executor(model, backbone, _stage(backbone, position, layers=(0, 2), phase="generate"), runner)
        generated = []
        try:
            generation.install()
            if backbone == "llava_onevision":
                call = dict(inputs=args["input_ids"], images=[args["images"][0].detach()],
                    image_sizes=args["image_sizes"], modalities=["video"], attention_mask=args["attention_mask"])
            else:
                call = {name: value.detach() if torch.is_tensor(value) else value
                    for name, value in args.items() if name not in {"labels", "use_cache"}}
            for repeat in range(2):
                with torch.no_grad(), generation._bridge.request_context(context):
                    result = model.generate(**call, min_new_tokens=2, max_new_tokens=2, do_sample=False,
                        use_cache=True, return_dict_in_generate=True, eos_token_id=None, pad_token_id=0)
                assert len(runner.calls) == repeat + 1
                assert generation.runtime_metadata()["projector_invocations"] == repeat + 1
                assert result.past_key_values.get_seq_length() == output.logits.shape[1] + 1
                generated.append(result.sequences)
            torch.testing.assert_close(generated[0], generated[1])
        finally:
            generation.uninstall()


def test_frozen_detector_scoped_layers_capture_real_native_attention_and_mean(tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")
    from mllm_comp.backbones._dytok_torch import FrozenDyToKDetector
    from mllm_comp.backbones.llava_onevision import provider_bridge as provider
    from mllm_comp.backbones.llava_onevision.video_compression import LlavaOneVisionVideoPacking

    torch.manual_seed(445)
    model = image_fixture.llava_model(num_decoder_layers=4)
    owner = image_fixture.qwen_model("qwen25_vl")
    model.config.mm_spatial_pool_stride = 2
    model.config.mm_spatial_pool_mode = "bilinear"
    loader = SimpleNamespace(load=lambda *args, **kwargs: SimpleNamespace(model=model, tokenizer=None),
        api=lambda: SimpleNamespace(image_token_index=-200))
    detector = FrozenDyToKDetector(tmp_path, siglip_path=tmp_path, source_loader=loader, target_layers=(0,), owner_model=owner)
    detector.prepare()
    assert all(not parameter.requires_grad for parameter in model.parameters())
    packing = LlavaOneVisionVideoPacking.from_native(frame_count=2, grid_height=14, grid_width=14,
        patch_merge_type="flat", newline_position="one_token")
    ids = torch.tensor([[1, -200, 2, 3]])
    embeddings = torch.randn(1, packing.packed_token_count + 3, 16)
    prepared = (None, None, None, None, embeddings, None)
    original_qk = provider._decoder_query_keys
    collected, executed = [], []
    layers = tuple(model.model.layers)

    def qk(layer, hidden_states, **kwargs):
        query, keys, all_keys = original_qk(layer, hidden_states, **kwargs)
        probability = (torch.einsum("qhd,khd->hqk", query.float(), keys.float())
            / math.sqrt(query.shape[-1])).softmax(-1).mean((0, 1))
        frame_values = probability.reshape(2, 196).mean(1)
        collected.append((layers.index(layer), frame_values))
        return query, keys, all_keys

    monkeypatch.setattr(provider, "_decoder_query_keys", qk)
    hooks = [layer.register_forward_pre_hook(lambda module, args, index=index: executed.append(index))
        for index, layer in enumerate(layers)]
    observed = []
    try:
        for selected in ((0,), (1, 3), (0, 2)):
            executor = DyToKVideoExecutor(backbone="qwen25_vl", model=owner,
                stage=_stage("qwen25_vl", layers=selected), detection_runner=detector)
            executor.signal_provider.prepare()
            assert detector.target_layers == selected
            collected.clear()
            executed.clear()
            with torch.no_grad():
                weights = detector._frame_importance_from_prepared(prepared, input_ids=ids, frame_count=2)
            assert tuple(index for index, _ in collected) == selected
            assert executed == list(range(selected[-1]))
            expected = torch.stack([value for _, value in collected]).mean(0)
            expected = expected / expected.sum()
            torch.testing.assert_close(weights, expected)
            assert not weights.requires_grad
            assert detector.runtime_metadata()["target_layers"] == selected
            observed.append(weights)
        assert not torch.allclose(observed[0], observed[1], atol=1e-7, rtol=1e-7)
        with pytest.raises(ValueError, match="depth"):
            detector.configure_target_layers((4,))
        # Invalid reconfiguration leaves the previously valid modules intact.
        assert detector.target_layers == (0, 2)
    finally:
        for hook in hooks:
            hook.remove()


def test_frozen_detector_depth_is_validated_when_real_model_loads(tmp_path):
    pytest.importorskip("torch")
    from mllm_comp.backbones._dytok_torch import FrozenDyToKDetector
    from mllm_comp.core.errors import BackendIntegrationError
    model = image_fixture.llava_model(num_decoder_layers=3)
    loader = SimpleNamespace(load=lambda *args, **kwargs: SimpleNamespace(model=model), api=lambda: None)
    detector = FrozenDyToKDetector(tmp_path, siglip_path=tmp_path, source_loader=loader, target_layers=(3,))
    with pytest.raises(BackendIntegrationError, match="depth"):
        detector.prepare()


def test_injected_signal_provider_cannot_silently_ignore_nondefault_detector_layers():
    from mllm_comp.core.errors import BackendIntegrationError
    provider = SimpleNamespace()
    # Existing default stubs remain accepted without a new provider protocol.
    executor = DyToKVideoExecutor(backbone="qwen25_vl", model=object(),
        stage=_stage("qwen25_vl"), signal_provider=provider)
    assert executor.signal_provider is provider
    with pytest.raises(BackendIntegrationError, match="signal provider.*changing"):
        DyToKVideoExecutor(backbone="qwen25_vl", model=object(),
            stage=_stage("qwen25_vl", layers=(0, 2)), signal_provider=provider)


def test_injected_detector_cannot_silently_ignore_nondefault_capture_layers():
    from mllm_comp.backbones.dytok_video import DyToKVideoSignalProvider
    from mllm_comp.core.errors import BackendIntegrationError
    runner = SimpleNamespace(frame_importance=lambda text, frames: None)
    provider = DyToKVideoSignalProvider(backbone="qwen25_vl", model=object(), detection_runner=runner)
    provider.prepare()
    provider.target_layers = (0, 2)
    with pytest.raises(BackendIntegrationError, match="detector.*changing"):
        provider.prepare()
