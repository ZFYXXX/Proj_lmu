"""A new algorithm identity uses all three native image decoder mechanisms."""

import copy
import importlib

import pytest

from mllm_comp.backbones.base import BackboneRuntime
from mllm_comp.core.vocabulary import ExecutionPhase
from mllm_comp.compressors.image.token_selection import ImageTokenSelection
from mllm_comp.compressors.image.decoder_selection import execute_decoder_selection
from mllm_comp.pipeline.definition import resolve_compression_definition
from mllm_comp.pipeline.runtime_session import RuntimeCompressionDispatcher
from mllm_comp.registry.backbones import get_backbone
from tests.fixtures import native_image as image_fixture
from examples.compression_extensions import image_selection as research
from tests.integration.test_projector_llava_onevision_positions_cpu import _model as _anyres_model


@pytest.mark.parametrize("backbone", research.BACKBONES)
def test_new_identity_resolves_through_public_extensions(backbone):
    config = research.experiment_config(backbone)
    definition = resolve_compression_definition(config, compression_mode="compressed", interface="train")
    projection = definition.projections["alternate-content"]
    assert projection.algorithm.name == research.ALGORITHM
    assert projection.algorithm.kernel is research.alternating_image_tokens
    assert projection.variant.name == research.VARIANT
    assert projection.variant.conditioning_modalities == frozenset()
    assert projection.bindings[0].name == f"{research.VARIANT}.{backbone}"
    assert projection.bindings[0].strategy_factory is research.AlternatingImageMethod
    assert dict(projection.algorithm_config) == {"stride": 2, "offset": 1, "apply_before_layer": 2}


def _exercise(torch, model, backbone, monkeypatch, *, checkpoint=False):
    if checkpoint:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    config = research.experiment_config(backbone)
    definition = resolve_compression_definition(config, compression_mode="compressed", interface="train")
    runtime = BackboneRuntime(model=model, processor=None, model_path_or_id="native-tiny-image",
        metadata={"phase": "train", "runtime_mode": "default"})
    session = RuntimeCompressionDispatcher(contract_registry=definition.registry).activate(
        config=config, facade=get_backbone(backbone), runtime=runtime,
        compression_mode="compressed", execution_phase=ExecutionPhase.FORWARD, definition=definition,
    )
    activation = session.active_records[0].activation
    if backbone == "llava_onevision":
        activation._bridge.prepare_training_forward()
    provider = importlib.import_module(f"mllm_comp.backbones.{backbone}.provider_bridge")

    def unexpected_attention(*args, **kwargs):
        pytest.fail("A content-position method must not request attention Q/K signals.")

    monkeypatch.setattr(provider, "_decoder_query_keys", unexpected_attention)
    torch.manual_seed(602)
    raw = image_fixture.inputs(torch, backbone)
    if backbone == "llava_onevision":
        pixels = torch.randn(2, 3, 54, 54, requires_grad=True)
        raw["images"] = [pixels]
        # Native AnyRes produces 729 base patches + 27 rows of 27 patches
        # and one newline. All indices below include the leading text token.
        newlines = [757 + 28 * row for row in range(27)]
        selected = list(range(2, 730, 2)) + [
            730 + 28 * row + column for row in range(27)
            for column in range(row % 2, 27, 2)
        ]
        expected_kept = sorted([0, *selected, *newlines, 1486, 1487])
        before, after = 1488, 759
    else:
        pixels = raw["pixel_values"]
        newlines, expected_kept = [], [0, 1, 3, 5, 6, 7]
        before, after = 8, 6
    labels = torch.full_like(raw["input_ids"], -100)
    labels[0, -1] = raw["input_ids"][0, -1]
    decisions = []
    original = research.alternating_image_tokens

    def observe(hidden, **kwargs):
        decision = original(hidden, **kwargs)
        assert isinstance(decision, ImageTokenSelection)
        assert decision.metadata["algorithm"] == research.ALGORITHM
        assert decision.metadata["variant"] == research.VARIANT
        assert tuple(kwargs.get("protected_indices", ())) == tuple(newlines)
        # The algorithm returns content positions only. Execution preserves
        # structural tokens and derives values, complementary indices and layout.
        assert set(decision.kept_image_indices).isdisjoint(newlines)
        decisions.append(decision)
        return decision

    def observe_execution(call, point, hidden, **kwargs):
        result = execute_decoder_selection(call, point, hidden, **kwargs)
        assert result.kept_indices.tolist() == expected_kept
        assert result.layout_delta.kept_indices == tuple(expected_kept)
        assert all(result.layout_delta.old_to_new[index] is not None for index in newlines)
        torch.testing.assert_close(result.hidden_states, hidden[expected_kept], rtol=0, atol=0)
        return result

    decoder = image_fixture.decoder(model, backbone)
    counts = [[] for _ in decoder.layers]
    handles = [layer.mlp.register_forward_pre_hook(
        lambda module, args, i=i: counts[i].append(args[0].shape[1]))
        for i, layer in enumerate(decoder.layers)]
    with monkeypatch.context() as patch:
        patch.setattr(research, "alternating_image_tokens", observe)
        adapter = importlib.import_module(
            f"mllm_comp.backbones.{backbone}."
            + ("compression_adapter" if backbone == "qwen3_vl" else "image_compression"))
        patch.setattr(adapter, "execute_decoder_selection", observe_execution)
        try:
            output = model(**raw, labels=labels, use_cache=False)
            assert output.logits.shape == (1, after, 64)
            assert [values[0] for values in counts] == [before if i < 2 else after for i in range(len(counts))]
            assert len(decisions) == 1
            output.loss.backward()
            assert len(decisions) == 1  # Checkpoint replay reuses the original decision.
            assert torch.isfinite(output.loss)
            assert pixels.grad is not None and torch.isfinite(pixels.grad).all() and pixels.grad.abs().sum() > 0
            gradients = {name: parameter.grad.detach().clone() for name, parameter in model.named_parameters()
                if parameter.grad is not None}
            assert all(torch.isfinite(gradient).all() for gradient in gradients.values())
            assert all(layer.mlp.gate_proj.weight.grad is not None for layer in decoder.layers)
            if backbone == "qwen3_vl":
                for merger in model.model.visual.deepstack_merger_list:
                    assert all(parameter.grad is not None for parameter in merger.parameters() if parameter.requires_grad)
            record = session.runtime_metadata()["activations"][0]
            assert record["algorithm"] == research.ALGORITHM
            assert record["variant"] == research.VARIANT
            assert record["binding"] == f"{research.VARIANT}.{backbone}"
            assert record["support_status"] != "active"
            return output.logits.detach().clone(), pixels.grad.detach().clone(), gradients
        finally:
            for handle in handles:
                handle.remove()
            runtime.close()


@pytest.mark.parametrize("backbone", research.BACKBONES)
def test_native_new_algorithm_selection_layout_gradient_and_checkpoint(backbone, monkeypatch):
    torch = pytest.importorskip("torch")
    torch.manual_seed(601)
    base = (_anyres_model() if backbone == "llava_onevision" else image_fixture.model(backbone)).train()
    expected = _exercise(torch, copy.deepcopy(base), backbone, monkeypatch)
    if backbone == "qwen3_vl":
        actual = _exercise(torch, copy.deepcopy(base), backbone, monkeypatch, checkpoint=True)
        torch.testing.assert_close(actual[0], expected[0])
        torch.testing.assert_close(actual[1], expected[1])
        assert actual[2].keys() == expected[2].keys()
        for name in actual[2]:
            torch.testing.assert_close(actual[2][name], expected[2][name])
