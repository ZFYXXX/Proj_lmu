"""A new video identity reaches three native Projector paths through public extensions."""

import pytest

from mllm_comp.backbones.base import BackboneRuntime
from mllm_comp.config.schema import experiment_config_from_dict
from mllm_comp.core.vocabulary import ExecutionPhase, TokenOperation
from mllm_comp.pipeline.definition import resolve_compression_definition
from mllm_comp.pipeline.runtime_session import RuntimeCompressionDispatcher
from mllm_comp.registry.backbones import get_backbone
from tests.fixtures import native_image as image_fixture
from examples.compression_extensions import stride_video as research

from tests.integration import test_video_vqtoken_qwen25_vl_positions_cpu as qwen25
from tests.integration import test_video_vqtoken_qwen3_vl_positions_cpu as qwen3
from tests.integration import test_video_vqtoken_llava_onevision_positions_cpu as llava


CONFIG = {"stride": 4, "offset": 2}


def _config(backbone):
    return experiment_config_from_dict({
        "backbone": {"name": backbone, "model_id": "native-tiny-video"},
        "pipeline": {"extensions": ["examples.compression_extensions.stride_video:register"],
            "stages": [{"name": "sample-video", "algorithm": research.ALGORITHM,
                "execution_phase": "forward", "config": CONFIG}]},
        "train": {"backend": "ms_swift", "compression_mode": "compressed"},
    })


@pytest.mark.parametrize("backbone", research.BACKBONES)
def test_private_video_parameters_and_identity_resolve_without_canonical_profile(backbone):
    config = _config(backbone)
    definition = resolve_compression_definition(config, compression_mode="compressed", interface="train")
    projection = definition.projections["sample-video"]
    assert projection.algorithm.name == research.ALGORITHM
    assert projection.algorithm.kernel is research.stride_video_tokens
    assert projection.algorithm.operations == frozenset({TokenOperation.DROP})
    assert projection.variant.name == research.VARIANT
    assert projection.variant.supported_rules == frozenset()
    assert dict(projection.algorithm_config) == CONFIG
    assert projection.variant.config_defaults == {"stride": 3, "offset": 1}
    binding = projection.bindings[0]
    assert binding.strategy_factory is research.StrideVideoMethod
    assert binding.executor_factory is definition.registry.binding(f"vqtoken.video.{backbone}").executor_factory
    assert binding.name == f"{research.VARIANT}.{backbone}"
    assert binding.ordered_targets()[0].name == research.TARGET
    assert "canonical_config" not in binding.metadata
    assert not binding.metadata.get("verified_phases")


@pytest.mark.parametrize("backbone,source_method", [
    *((backbone, "vqtoken") for backbone in research.BACKBONES),
    ("qwen3_vl", "fastvid"), ("qwen3_vl", "visionzip"),
])
def test_native_video_selection_deepstack_values_gradients_and_cleanup(backbone, source_method, monkeypatch):
    torch = pytest.importorskip("torch")
    torch.manual_seed(911)
    model = (llava._video_model() if backbone == "llava_onevision" else image_fixture.qwen_model(backbone)).train()
    fixture = {"qwen25_vl": qwen25, "qwen3_vl": qwen3, "llava_onevision": llava}[backbone]
    args = fixture._inputs(torch)
    pixels = args["images"][0] if backbone == "llava_onevision" else args["pixel_values_videos"]
    pixels.requires_grad_(True)
    labels = torch.full_like(args["input_ids"], -100)
    labels[0, -1] = args["input_ids"][0, -1]
    args.update(labels=labels, use_cache=False)
    source_count = {"qwen25_vl": 48, "qwen3_vl": 36, "llava_onevision": 392}[backbone]
    selected = list(range(2, source_count, 4))
    decoder = model.model if backbone == "llava_onevision" else model.model.language_model
    projectors = ([model.model.mm_projector] if backbone == "llava_onevision" else
        [model.model.visual.merger, *getattr(model.model.visual, "deepstack_merger_list", ())])
    projected = [[] for _ in projectors]
    decoder_inputs, decisions, replayed, injected = [], [], [], []
    original_forward = model.forward
    original_projectors = [module.forward for module in projectors]
    original_kernel, original_replay = research.stride_video_tokens, research.replay_stride_video_features

    def compress(payload, **kwargs):
        assert kwargs["config"] == CONFIG and not kwargs["rules"]
        assert payload.token_count == source_count
        result = original_kernel(payload, **kwargs)
        assert result.representative_indices == (tuple(selected),)
        assert result.source_groups == (tuple((index,) for index in selected),)
        delta = result.layout_deltas[0]
        assert delta.kept_indices == tuple(selected)
        assert delta.dropped_indices == tuple(index for index in range(source_count) if index not in selected)
        assert result.coordinates == tuple(payload.coordinates[index] for index in selected)
        torch.testing.assert_close(result.features, payload.features[selected], atol=0, rtol=0)
        decisions.append((payload, result))
        return result

    def replay(features, result, **kwargs):
        assert kwargs["config"] == CONFIG and not kwargs["rules"]
        value = original_replay(features, result, **kwargs)
        torch.testing.assert_close(value, features[selected], atol=0, rtol=0)
        replayed.append(value)
        return value

    monkeypatch.setattr(research, "stride_video_tokens", compress)
    monkeypatch.setattr(research, "replay_stride_video_features", replay)
    register_method = research.register_method

    def register_with_signal_source(registry, **kwargs):
        kwargs["reuse_bindings"] = [f"{source_method}.video.{name}" for name in research.BACKBONES]
        return register_method(registry, **kwargs)

    monkeypatch.setattr(research, "register_method", register_with_signal_source)
    config = _config(backbone)
    definition = resolve_compression_definition(config, compression_mode="compressed", interface="train")
    runtime = BackboneRuntime(model=model, processor=None, model_path_or_id="native-tiny-video",
        metadata={"phase": "train", "runtime_mode": "default"})
    session = RuntimeCompressionDispatcher(contract_registry=definition.registry).activate(
        config=config, facade=get_backbone(backbone), runtime=runtime,
        compression_mode="compressed", execution_phase=ExecutionPhase.FORWARD, definition=definition,
    )
    activation = session.active_records[0].activation
    assert isinstance(activation.kernel, research.StrideVideoMethod)
    bridge = activation._bridge
    hooks = [projector.register_forward_hook(lambda module, values, output, i=i: projected[i].append(output))
        for i, projector in enumerate(projectors)]
    hooks.append(decoder.layers[0].register_forward_pre_hook(
        lambda module, values: decoder_inputs.append(values[0])))
    if backbone == "qwen3_vl":
        deepstack = decoder._deepstack_process
        def observe_deepstack(hidden, mask, features):
            injected.append(features)
            return deepstack(hidden, mask, features)
        monkeypatch.setattr(decoder, "_deepstack_process", observe_deepstack)
    try:
        output = model(**args)
        assert len(decisions) == len(decoder_inputs) == 1
        assert all(len(values) == 1 for values in projected)
        payload, _ = decisions[0]
        dense = projected[0][0]
        if backbone == "qwen25_vl":
            window, _ = model.model.visual.get_window_index(args["video_grid_thw"])
            source = dense[window.argsort()]
        elif backbone == "llava_onevision":
            source = llava._pool(torch, dense, "bilinear")
        else:
            source = dense
        torch.testing.assert_close(payload.features, source)
        if backbone == "llava_onevision":
            # Two 14x14 pooled frames; each of 28 rows ends with one native newline.
            patch_positions = [frame * 210 + row * 15 + column
                for frame in range(2) for row in range(14) for column in range(14)]
            newline_positions = [frame * 210 + row * 15 + 14 for frame in range(2) for row in range(14)]
            entries = [(patch_positions[index], source[index]) for index in selected]
            entries += [(index, model.model.image_newline) for index in newline_positions]
            entries.sort(key=lambda item: item[0])
            expected = torch.stack([value for _, value in entries])
            actual = decoder_inputs[0][0, 1:-2]
            expected_length = 3 + len(selected) + 28
        else:
            expected, actual = source[selected], decoder_inputs[0][0, 2:-2]
            expected_length = 4 + len(selected)
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
        assert output.logits.shape == (1, expected_length, 64)
        upstream = torch.randn_like(actual)
        torch.testing.assert_close(
            torch.autograd.grad(actual, dense, upstream, retain_graph=True)[0],
            torch.autograd.grad(expected, dense, upstream, retain_graph=True)[0], atol=0, rtol=0,
        )
        if backbone == "qwen3_vl":
            assert len(replayed) == len(injected) == len(projectors) - 1 == 3
            for index, branch in enumerate(injected):
                native_branch = projected[index + 1][0]
                torch.testing.assert_close(branch, native_branch[selected], atol=0, rtol=0)
                torch.testing.assert_close(branch, replayed[index], atol=0, rtol=0)
                branch_grad = torch.autograd.grad(branch.sum(), native_branch, retain_graph=True)[0]
                expected_grad = torch.zeros_like(native_branch)
                expected_grad[selected] = 1
                torch.testing.assert_close(branch_grad, expected_grad, atol=0, rtol=0)
        else:
            assert not replayed
        output.loss.backward()
        assert len(decisions) == 1
        assert torch.isfinite(output.loss)
        assert pixels.grad is not None and torch.isfinite(pixels.grad).all() and pixels.grad.abs().sum() > 0
        for projector in projectors:
            assert all(parameter.grad is not None and torch.isfinite(parameter.grad).all()
                for parameter in projector.parameters() if parameter.requires_grad)
            assert any(parameter.grad.abs().sum() > 0 for parameter in projector.parameters())
        if backbone == "llava_onevision":
            assert model.model.image_newline.grad is not None and model.model.image_newline.grad.abs().sum() > 0
        metadata = activation.runtime_metadata()
        assert metadata["algorithm"] == research.ALGORITHM and metadata["variant"] == research.VARIANT
        assert metadata["binding"] == f"{research.VARIANT}.{backbone}"
        assert metadata["projector_invocations"] == 1
        assert metadata["projector_layout_deltas"][0]["stage"] == research.TARGET
        assert not {"canonical_config", "global_fixed_k", "fixed_budget_per_temporal_slice", "stages"} & metadata.keys()
        assert session.runtime_metadata()["activations"][0]["support_status"] != "active"
    finally:
        for hook in hooks:
            hook.remove()
        runtime.close()
    assert not activation.installed and not bridge.patched
    assert model.forward == original_forward
    assert all(module.forward == original for module, original in zip(projectors, original_projectors))
    with torch.no_grad():
        dense_output = model.eval()(**args)
    assert len(decisions) == 1  # The closed session leaves the next native request untouched.
    assert dense_output.logits.shape[1] == source_count + (31 if backbone == "llava_onevision" else 4)
