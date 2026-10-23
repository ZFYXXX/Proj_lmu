"""Compound visual positions retain their decoder stages and default recipes."""

import copy

import pytest

from tests.fixtures import native_image as image_fixture
from tests.integration import test_image_nuwa_positions_cpu as nuwa_checks
from tests.integration.test_image_projector_positions_cpu import _projector
from mllm_comp.pipeline.stage import CompressionStageConfig


def _model(backbone):
    model = nuwa_checks._model(backbone)
    if backbone == "llava_onevision":
        torch = pytest.importorskip("torch")
        layer = model.model.mm_projector
        model.model.mm_projector = torch.nn.Sequential(layer, torch.nn.GELU(),
            torch.nn.Linear(layer.out_features, layer.out_features))
    return model


def _stage(backbone, algorithm, position=None, phase="forward"):
    if algorithm == "nuwa":
        stage = nuwa_checks._stage(backbone, 2, phase=phase)
        stage.operation = {} if position is None else {"target_site": position}
        return stage
    return CompressionStageConfig.from_dict({
        "name": "restore-projector-position", "placement": "projector_level", "execution_phase": phase,
        "plugin": "restore", "algorithm": "restore", "variant": "restore.image",
        "binding": f"restore.image.{backbone}", "target_modalities": ["image"],
        "config": {"target_count": 2}, "operation": {} if position is None else {"target_site": position},
    })


@pytest.mark.parametrize("backbone", ["qwen3_vl", "qwen25_vl", "llava_onevision"])
@pytest.mark.parametrize("algorithm", ["nuwa", "restore"])
def test_compound_native_values_default_alias_and_gradients(backbone, algorithm):
    torch = pytest.importorskip("torch")
    torch.manual_seed(601)
    base = _model(backbone).train()
    old_site = {"qwen3_vl": "post_final_merger_pre_llm_scatter", "qwen25_vl": "post_vision_merger_pre_llm_scatter",
        "llava_onevision": "post_anyres_pack_pre_llm_splice"}[backbone]
    expected = None
    for position in (None, old_site, "projector_input", "projector_output"):
        model = copy.deepcopy(base)
        stage = _stage(backbone, algorithm, position)
        activation = image_fixture.activation(model, stage)
        counts = []
        handle = _projector(model, backbone).register_forward_pre_hook(lambda module, args: counts.append(args[0].numel() // args[0].shape[-1]))
        activation.install()
        if backbone == "llava_onevision":
            activation._bridge.prepare_training_forward()
        torch.manual_seed(602)
        raw = image_fixture.inputs(torch, backbone)
        pixels = raw["images" if backbone == "llava_onevision" else "pixel_values"]
        labels = torch.full_like(raw["input_ids"], -100)
        labels[0, -1] = raw["input_ids"][0, -1]
        try:
            output = model(**raw, labels=labels, use_cache=False)
            output.loss.backward()
            assert pixels.grad is not None and torch.isfinite(pixels.grad).all() and pixels.grad.abs().sum() > 0
            gradients = {name: parameter.grad.clone() for name, parameter in _projector(model, backbone).named_parameters()}
            assert all(torch.isfinite(gradient).all() for gradient in gradients.values())
            if expected is None:
                expected = output.logits.detach(), pixels.grad.clone(), gradients, counts[:]
            elif position == old_site or (position == "projector_output" and backbone != "llava_onevision"):
                torch.testing.assert_close(output.logits, expected[0], rtol=0, atol=0)
                torch.testing.assert_close(pixels.grad, expected[1], rtol=0, atol=0)
                for name, gradient in gradients.items():
                    torch.testing.assert_close(gradient, expected[2][name], rtol=0, atol=0)
                assert counts == expected[3]
            elif backbone != "llava_onevision":
                assert min(counts) < max(expected[3])
            elif position == "projector_output":
                assert len(counts) == 1
            metadata = activation.runtime_metadata()
            assert metadata["projector_target_sites"] == list(activation.projector_target_sites)
        finally:
            handle.remove()
            activation.uninstall()


@pytest.mark.parametrize("backbone", ["qwen3_vl", "qwen25_vl", "llava_onevision"])
@pytest.mark.parametrize("algorithm", ["nuwa", "restore"])
def test_compound_projector_positions_keep_repeated_generation_cache(backbone, algorithm):
    torch = pytest.importorskip("torch")
    torch.manual_seed(603)
    model = _model(backbone).eval()
    # LLaVA's new post-value route differs from its legacy raw aggregation;
    # Qwen's new input route changes the values while preserving decode.
    position = "projector_output" if backbone == "llava_onevision" else "projector_input"
    activation = image_fixture.activation(model, _stage(backbone, algorithm, position, "generate"))
    activation.install()
    raw = image_fixture.inputs(torch, backbone)
    if backbone == "llava_onevision":
        raw["inputs"] = raw.pop("input_ids")
    try:
        with torch.no_grad():
            first = model.generate(**raw, max_new_tokens=2, min_new_tokens=2, do_sample=False, use_cache=True, return_dict_in_generate=True)
            second = model.generate(**raw, max_new_tokens=2, min_new_tokens=2, do_sample=False, use_cache=True, return_dict_in_generate=True)
        assert torch.equal(first.sequences, second.sequences)
        lengths = [layer.get_seq_length() for layer in second.past_key_values.layers]
        assert lengths and min(lengths) > 0 and len(set(lengths)) == 1
        bridge = activation._bridge if backbone != "qwen3_vl" else model._mllm_comp_qwen3_vl_provider_bridge
        metadata = bridge.runtime_metadata()
        lifecycle = metadata["latest_decoder_lifecycle"]
        assert lifecycle["decode_forward_calls"] == 1
        assert lifecycle["selection_repeated_during_decode"] is False
        if algorithm == "restore":
            calibration = metadata["latest_restore_calibration"]
            assert calibration["layers_calibrated"] == len(image_fixture.decoder(model, backbone).layers)
            assert calibration["query_tokens"] == 1
        else:
            assert lifecycle["selection_runs"] == 1
    finally:
        activation.uninstall()


@pytest.mark.parametrize("backbone", ["qwen3_vl", "qwen25_vl", "llava_onevision"])
def test_compound_projector_source_and_values_across_gpus(backbone):
    from tests.integration.test_image_projector_positions_cpu import test_projector_source_and_values_on_different_gpus
    test_projector_source_and_values_on_different_gpus(backbone, algorithms=("nuwa", "restore"),
        stage_factory=lambda base, algorithm: _stage(base, algorithm, "projector_input"), model_factory=_model)
