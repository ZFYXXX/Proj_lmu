"""Configured image merge rules on the existing native value/replay paths."""
import copy
import importlib

import pytest

from tests.fixtures import native_image as image_fixture
from tests.integration import test_image_projector_positions_cpu as simple
from tests.integration import test_image_compound_projector_positions_cpu as compound
from tests.integration.test_projector_tome_qwen3_vl_positions_cpu import _stage as tome_stage
from tests.unit.test_image_aim_tome_custom_rules import (
    _aim_independent, _tome_independent, destination_weighted,
)

MODULE = "tests.integration.test_image_merge_rules_cpu"
HOOKS = {
    "visionzip": ("scorer", "dominant_selector", "contextual_selector", "grouper", "aggregator"),
    "prumerge": ("scorer", "selector", "grouper", "aggregator"),
    "aim": ("scorer", "grouper", "aggregator"),
    "tome": tuple(scope + "_" + hook for scope in ("encoder", "output") for hook in ("scorer", "grouper", "aggregator")),
    "nuwa": ("stage1_scorer", "regional_selector", "stage1_selector", "aggregation_weights", "stage2_source_scorer", "stage2_scorer", "stage2_selector"),
    "restore": ("pristine_scorer", "pristine_selector", "anchor_scorer", "anchor_selector", "grouper", "aggregator", "calibration_bias"),
}
KERNELS = {"visionzip": "visionzip_image", "prumerge": "prumerge_image_tokens",
    "aim": "aim_image_tokens", "tome": "tome_merge_image", "nuwa": "nuwa_stage1_image_tokens", "restore": "restore_image_tokens"}


def custom_rule(request, *, algorithm, rule):
    """Small deterministic rules whose math is independently checked below."""
    import torch
    if rule.endswith("aggregator"):
        if algorithm in {"aim", "tome"}:
            return destination_weighted(request)
        return request["values"].square().mean(0)
    if rule == "calibration_bias":
        return request["default_bias"] * .75
    if rule == "aggregation_weights":
        original = request["default_weights"]
        return torch.ones_like(original) / original.shape[1] * 1.25
    if rule.endswith("scorer"):
        if algorithm in {"aim", "tome"}:
            original = request["scores"]
            return torch.arange(original.numel(), device=original.device, dtype=original.dtype).reshape(original.shape)
        if rule == "stage2_source_scorer":
            values = request["image_keys"]
        elif rule == "stage2_scorer":
            values = request["selectable_indices"]
        else:
            values = request.get("tokens", request.get("selection_features", request.get("values")))
        return torch.arange(1, len(values) + 1, device=values.device, dtype=torch.float32)
    if rule.endswith("selector"):
        count = request["keep_count"]
        candidates = request.get("candidate_indices", request.get("candidates"))
        if candidates is None:
            candidates = range(len(request["scores"]))
        candidates = torch.as_tensor(candidates, device=request["scores"].device, dtype=torch.long)
        return candidates[-count:] if count else candidates[:0]
    if algorithm in {"aim", "tome"}:
        return [(int(source), int(request["destination_indices"][0]))
            for source in request["source_indices"][-request["merge_count"]:]]
    if algorithm == "visionzip":
        return torch.zeros(len(request["source_indices"]), device=request["keys"].device, dtype=torch.long)
    if algorithm == "prumerge":
        candidates = list(request["candidate_indices"])
        return torch.tensor([[i for i in reversed(candidates) if i != int(center)][:request["neighbour_count"]]
            for center in request["center_indices"]], device=request["tokens"].device)
    if algorithm == "restore":
        anchors = request["anchors"].tolist()
        groups = [[i] for i in anchors]
        for i in range(len(request["source_indices"])):
            if i not in anchors:
                groups[i % len(groups)].append(i)
        return groups
    raise AssertionError((algorithm, rule))


def _stage(backbone, algorithm, position):
    if algorithm == "tome":
        stage = tome_stage(position, backbone=backbone)
    elif algorithm in {"nuwa", "restore"}:
        stage = compound._stage(backbone, algorithm, position)
    else:
        stage = simple._stage(backbone, algorithm, position)
    if algorithm == "nuwa":
        # Explicit original attention source enables its independent source scorer.
        stage.config["stage2_after_layer"] = 2
    stage.operation["rules"] = {name: {"callable": MODULE + ":custom_rule",
        "params": {"algorithm": algorithm, "rule": name}} for name in HOOKS[algorithm]}
    return stage


def _model(backbone, algorithm):
    return (compound._model if algorithm in {"nuwa", "restore"} else simple._model)(backbone)


def _expected(torch, values, result, algorithm):
    if algorithm == "aim":
        return _aim_independent(values, result, weighted=True)
    if algorithm == "tome":
        return _tome_independent(values, result, weighted=True)
    if algorithm == "nuwa":
        return values.mean(0, keepdim=True).expand(result.tokens.shape[0], -1) * 1.25
    if algorithm == "restore":
        pristine = set(result.pristine_indices.tolist())
        return torch.stack([values[group[0]] if len(group) == 1 and group[0] in pristine else values[list(group)].square().mean(0)
            for group in result.metadata["source_groups"]])
    if algorithm == "prumerge":
        return torch.stack([values[sources[0]] if kind == "protected" else values[list(sources)].square().mean(0)
            for kind, sources, _ in result.aggregation_plan])
    contextual = set(result.metadata["contextual_indices"])
    return torch.stack([values[list(group)].square().mean(0) if int(representative) in contextual else values[int(representative)]
        for group, representative in zip(result.source_groups, result.indices)])


def _install_observers(monkeypatch, backbone, algorithm):
    import torch
    calls, kernels = [], []
    original_rule = custom_rule
    def observed(request, **kwargs):
        calls.append(kwargs["rule"])
        return original_rule(request, **kwargs)
    monkeypatch.setattr(importlib.import_module(MODULE), "custom_rule", observed)
    for suffix in ("provider_bridge", "compression_adapter" if backbone == "qwen3_vl" else "image_compression"):
        module = importlib.import_module("mllm_comp.backbones." + backbone + "." + suffix)
        name = KERNELS[algorithm]
        original = getattr(module, name, None)
        if original is None:
            continue
        def check(values, *args, _original=original, **kwargs):
            result = _original(values, *args, **kwargs)
            assert kwargs.get("rules")
            if algorithm == "visionzip":
                assert result.metadata["score_source"] == "custom"
                assert result.metadata["contextual_target_sampling"] == "custom"
            elif algorithm == "prumerge":
                assert result.metadata["uniform_policy"] == "custom"
            with torch.no_grad():
                expected = _expected(torch, values, result, algorithm)
                torch.testing.assert_close(result.hidden_states if algorithm == "tome" else result.tokens, expected)
            kernels.append(result)
            return result
        monkeypatch.setattr(module, name, check)
    return calls, kernels


def _run(backbone, algorithm, position, checkpoint, monkeypatch):
    torch = pytest.importorskip("torch")
    torch.manual_seed(953)
    model = _model(backbone, algorithm).train()
    if checkpoint is not None:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": checkpoint})
    stage = _stage(backbone, algorithm, position)
    calls, kernels = _install_observers(monkeypatch, backbone, algorithm)
    activation = image_fixture.activation(model, stage)
    branch_checks = []
    if backbone == "qwen3_vl" and algorithm != "tome":
        original_callback = activation.compress_after_visual_mergers
        def check_branches(main, branches, **kwargs):
            result = original_callback(main, branches, **kwargs)
            with torch.no_grad():
                for values, actual in zip(branches, result.deepstack_image_features):
                    torch.testing.assert_close(actual, _expected(torch, values, result.kernel_result, algorithm))
                    branch_checks.append(tuple(values.shape))
            return result
        monkeypatch.setattr(activation, "compress_after_visual_mergers", check_branches)
    activation.install()
    if backbone == "llava_onevision":
        activation._bridge.prepare_training_forward()
    raw = image_fixture.inputs(torch, backbone)
    labels = torch.full_like(raw["input_ids"], -100)
    labels[0, -1] = raw["input_ids"][0, -1]
    try:
        output = model(**raw, labels=labels, use_cache=False)
        assert kernels
        if backbone == "qwen3_vl" and algorithm != "tome":
            assert len(branch_checks) == len(model.model.visual.deepstack_merger_list)
        assert set(calls) == set(HOOKS[algorithm]), (algorithm, calls)
        decisions = [name for name in calls if not name.endswith("aggregator") and name != "calibration_bias"]
        output.loss.backward()
        assert [name for name in calls if not name.endswith("aggregator") and name != "calibration_bias"] == decisions
        pixel = raw["images" if backbone == "llava_onevision" else "pixel_values"]
        assert pixel.grad is not None and torch.isfinite(pixel.grad).all() and pixel.grad.abs().sum() > 0
        projectors = [simple._projector(model, backbone)]
        if backbone == "qwen3_vl":
            projectors.extend(model.model.visual.deepstack_merger_list)
        for projector in projectors:
            grads = [p.grad for p in projector.parameters() if p.requires_grad]
            assert grads and all(g is not None and torch.isfinite(g).all() for g in grads)
            assert any(g.abs().sum() > 0 for g in grads)
        metadata = activation.runtime_metadata()
        assert metadata["rules"] == stage.operation["rules"]
        assert metadata["rules_changed"] is True and metadata["canonical_evidence_applicable"] is False
        return output.logits.detach(), pixel.grad, tuple(g.detach().clone() for g in grads)
    finally:
        activation.uninstall()


@pytest.mark.parametrize("backbone", ["llava_onevision", "qwen25_vl", "qwen3_vl"])
@pytest.mark.parametrize("algorithm", list(HOOKS))
def test_merge_rules_native_all_hooks_actual_values_and_branch_gradients(backbone, algorithm, monkeypatch):
    for position in ("projector_input", "projector_output"):
        with monkeypatch.context() as patch:
            _run(backbone, algorithm, position, None, patch)


@pytest.mark.parametrize("algorithm", ["aim", "tome", "nuwa", "restore"])
def test_qwen3_merge_rules_checkpoint_preserves_decisions_and_gradients(algorithm, monkeypatch):
    torch = pytest.importorskip("torch")
    results = []
    for checkpoint in (None, False, True):
        with monkeypatch.context() as patch:
            results.append(_run("qwen3_vl", algorithm, "projector_input", checkpoint, patch))
    for result in results[1:]:
        torch.testing.assert_close(result[0], results[0][0])
        torch.testing.assert_close(result[1], results[0][1])
        for actual, expected in zip(result[2], results[0][2]):
            torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize("backbone", ["llava_onevision", "qwen25_vl", "qwen3_vl"])
def test_nuwa_random_mode_rejects_a_fixed_attention_source_rule(backbone):
    pytest.importorskip("torch")
    from mllm_comp.core.errors import BackendIntegrationError
    model = _model(backbone, "nuwa")
    stage = _stage(backbone, "nuwa", "projector_input")
    stage.config["random_stage2"] = True
    with pytest.raises(BackendIntegrationError, match="stage2_source_scorer"):
        image_fixture.activation(model, stage)
