"""Native image paths retain custom decisions at the existing physical sites."""

import importlib

import pytest

from tests.fixtures import native_image as image_fixture
from tests.integration.test_image_projector_positions_cpu import _model, _mmtok_zoo_model, _stage


@pytest.mark.parametrize("backbone", ["qwen25_vl", "qwen3_vl", "llava_onevision"])
@pytest.mark.parametrize("algorithm", ["gap", "divprune", "vispruner", "mmtok", "zoo_prune", "saint"])
def test_native_selection_rules_values_positions_and_gradient(backbone, algorithm, monkeypatch):
    torch = pytest.importorskip("torch")
    module = importlib.import_module("mllm_comp.backbones." + backbone + "." +
        ("compression_adapter" if backbone == "qwen3_vl" else "image_compression"))
    kernel_name = "divprune_image" if algorithm == "divprune" else algorithm + "_image_tokens"
    original = getattr(module, kernel_name)
    results = []

    def checked(*args, **kwargs):
        assert set(kwargs["rules"]) == {"selector"}
        result = original(*args, **kwargs)
        protected = tuple(kwargs.get("protected_indices", ()))
        eligible = [i for i in range(args[0].shape[0]) if i not in protected]
        count = result.tokens.shape[0] - len(protected)
        indices = torch.tensor(sorted([*protected, *(eligible[-count:] if count else [])]), device=args[0].device)
        torch.testing.assert_close(result.tokens, args[0].index_select(0, indices))
        assert result.metadata["rules_changed"] is True
        results.append(result)
        return result

    monkeypatch.setattr(module, kernel_name, checked)
    for position in ("projector_input", "projector_output"):
        torch.manual_seed(907)
        model = (_mmtok_zoo_model(backbone) if algorithm in {"mmtok", "zoo_prune"} else _model(backbone)).train()
        if algorithm == "saint" and backbone == "qwen25_vl":
            from tests.integration.test_saint_qwen25_vl_positions_cpu import _stage as saint_stage, _inputs
            stage = saint_stage((0, 2), mid=position == "projector_input")
            raw = _inputs(torch)
        else:
            stage = _stage(backbone, algorithm, position)
            raw = image_fixture.inputs(torch, backbone)
        stage.operation["rules"] = {"selector": {
            "callable": "tests.unit.test_image_selection_rules:tail_selector"}}
        activation = image_fixture.activation(model, stage)
        results.clear()
        activation.install()
        try:
            labels = torch.full_like(raw["input_ids"], -100)
            labels[0, -1] = raw["input_ids"][0, -1]
            output = model(**raw, labels=labels, use_cache=False)
            assert results
            # Q3's branches consume the main plan, rather than invoking the user rule again.
            if backbone == "qwen3_vl":
                assert len(results) == 1
            output.loss.backward()
            pixels = raw["images" if backbone == "llava_onevision" else "pixel_values"]
            assert pixels.grad is not None and torch.isfinite(pixels.grad).all() and pixels.grad.abs().sum() > 0
            if backbone == "qwen3_vl":
                for merger in model.model.visual.deepstack_merger_list:
                    grads = [p.grad for p in merger.parameters() if p.requires_grad]
                    assert grads and all(g is not None and torch.isfinite(g).all() for g in grads)
            metadata = activation.runtime_metadata()
            assert metadata["rules"] == stage.operation["rules"] or metadata["rules"] == {
                "selector": {**stage.operation["rules"]["selector"], "params": {}}}
            assert metadata["canonical_evidence_applicable"] is False
        finally:
            activation.uninstall()
