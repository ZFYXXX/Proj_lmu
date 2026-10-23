"""Real image providers run Python rules at existing FastV boundaries."""

import copy
import importlib

from mllm_comp.compressors.image import decoder_selection

import pytest

from tests.fixtures import native_image as image_fixture
from tests.integration.test_image_fastv_positions_cpu import _stage
from tests.unit.test_fastv_custom_rules import feature_norm, low_score_select


_CALLS = []


def observed_norm(request, power=2.0):
    _CALLS.append(("scorer", request["scorer_query"] is None))
    return feature_norm(request, power=power)


def observed_selector(request):
    _CALLS.append(("selector", request["scorer_query"] is None))
    return low_score_select(request)


def _custom_stage(backbone, boundary, *, mode="both", mid=False, phase="forward", source=2):
    stage = _stage(backbone, boundary, prune_layer=source, mid=mid, phase=phase)
    if mode in {"scorer", "both", "initial"}:
        stage.criterion["scorer"] = {"callable": f"{__name__}:observed_norm", "params": {"power": 2.0}}
    if mode in {"selector", "both"}:
        stage.operation["selector"] = {"callable": f"{__name__}:observed_selector"}
    return stage


def _adapter(backbone):
    module = "compression_adapter" if backbone == "qwen3_vl" else "image_compression"
    return importlib.import_module(f"mllm_comp.backbones.{backbone}.{module}")


def _labels(torch, args):
    labels = torch.full_like(args["input_ids"], -100)
    labels[0, -1] = args["input_ids"][0, -1]
    return labels


@pytest.mark.parametrize("backbone", ["qwen25_vl", "qwen3_vl", "llava_onevision"])
@pytest.mark.parametrize("mode", ["scorer", "selector", "both", "initial"])
def test_native_custom_rules_use_actual_values_and_preserve_layout_gradient(backbone, mode, monkeypatch):
    torch = pytest.importorskip("torch")
    torch.manual_seed(192)
    model = image_fixture.model(backbone).train()
    depth = len(image_fixture.decoder(model, backbone).layers)
    boundary = depth if mode == "scorer" else 0 if mode == "initial" else 1
    mid = mode in {"selector", "both"}
    stage = _custom_stage(backbone, boundary, mode=mode, mid=mid, source=0 if mode == "initial" else 2)
    activation = image_fixture.activation(model, stage)
    original = decoder_selection.fastv_image_tokens
    decisions = []

    def check(hidden, **kwargs):
        result = original(hidden, **kwargs)
        images = torch.as_tensor(kwargs["image_indices"], device=hidden.device)
        if mode != "selector":
            expected_scores = hidden.index_select(0, images).float().square().sum(-1)
        else:
            query, keys = kwargs["scorer_query"], kwargs["attention_keys"]
            logits = torch.einsum("qhd,thd->qht", query, keys) / query.shape[-1] ** 0.5
            mask = kwargs["attention_key_mask"]
            if mask is not None:
                visible = mask[None, None, :] if mask.ndim == 1 else mask[:, None, :]
                logits = logits.masked_fill(~visible, -torch.inf)
            expected_scores = logits.float().softmax(-1).mean((0, 1)).index_select(0, images)
        torch.testing.assert_close(result.image_scores, expected_scores)
        image_list = images.tolist()
        protected = [image_list.index(i) for i in kwargs.get("protected_indices", ()) if i in image_list]
        count = max(round(len(image_list) * 0.5), len(protected), 1)
        ranking = torch.argsort(expected_scores, descending=mode in {"scorer", "initial"}, stable=True).tolist()
        expected = sorted((protected + [i for i in ranking if i not in protected])[:count])
        torch.testing.assert_close(result.kept_image_indices, images[expected], rtol=0, atol=0)
        torch.testing.assert_close(result.hidden_states, hidden[result.kept_indices], rtol=0, atol=0)
        assert all(i in result.kept_indices.tolist() for i in range(len(hidden)) if i not in image_list)
        assert result.metadata["ranking"].startswith("custom")
        assert (kwargs["scorer_query"] is None) == (mode == "initial")
        decisions.append(result.kept_indices)
        return result

    monkeypatch.setattr(decoder_selection, "fastv_image_tokens", check)
    counts = [[] for _ in range(depth)]
    handles = [layer.mlp.register_forward_pre_hook(
        lambda module, args, i=i: counts[i].append(args[0].shape[1]))
        for i, layer in enumerate(image_fixture.decoder(model, backbone).layers)]
    _CALLS.clear()
    activation.install()
    if backbone == "llava_onevision":
        activation._bridge.prepare_training_forward()
    try:
        args = image_fixture.inputs(torch, backbone)
        output = model(**args, labels=_labels(torch, args), use_cache=False)
        output.loss.backward()
        before, after = (732, 367) if backbone == "llava_onevision" else (8, 6)
        assert output.logits.shape == (1, after, 64)
        assert [values[0] for values in counts] == [before if i < boundary else after for i in range(depth)]
        gradient = args["images" if backbone == "llava_onevision" else "pixel_values"].grad
        assert gradient is not None and torch.isfinite(gradient).all() and gradient.abs().sum() > 0
        assert len(decisions) == 1
        assert len(_CALLS) == (2 if mode == "both" else 1)
        metadata = activation.runtime_metadata()
        assert metadata["rules_changed"] is True and metadata["canonical_evidence_applicable"] is False
        assert metadata["development_runtime"] is True
        assert set(metadata["rules"]) == ({"scorer", "selector"} if mode == "both" else {"scorer" if mode == "initial" else mode})
    finally:
        for handle in handles:
            handle.remove()
        activation.uninstall()


@pytest.mark.parametrize("backbone", ["qwen25_vl", "qwen3_vl", "llava_onevision"])
def test_custom_mid_rules_replay_after_uninstall_with_both_checkpoint_modes(backbone, monkeypatch):
    torch = pytest.importorskip("torch")
    from mllm_comp.compressors import rules
    torch.manual_seed(193)
    base = image_fixture.model(backbone).train()
    loader = rules.load_rule_specs

    loads = []
    def counted_loader(specs):
        loads.append(specs)
        return loader(specs)

    monkeypatch.setattr(rules, "load_rule_specs", counted_loader)
    if backbone != "qwen3_vl":
        monkeypatch.setattr(_adapter(backbone), "load_rule_specs", counted_loader)
    expected = None
    for checkpoint in (None, False, True):
        model = copy.deepcopy(base)
        if checkpoint is not None:
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": checkpoint})
        _CALLS.clear()
        activation = image_fixture.activation(model, _custom_stage(backbone, 1, mid=True))
        load_count = len(loads)
        activation.install()
        if backbone == "llava_onevision":
            activation._bridge.prepare_training_forward()
        torch.manual_seed(194)
        args = image_fixture.inputs(torch, backbone)
        try:
            output = model(**args, labels=_labels(torch, args), use_cache=False)
        finally:
            activation.uninstall()
        assert len(_CALLS) == 2
        output.loss.backward()
        assert len(_CALLS) == 2 and len(loads) == load_count
        gradient = args["images" if backbone == "llava_onevision" else "pixel_values"].grad
        assert gradient is not None and torch.isfinite(gradient).all() and gradient.abs().sum() > 0
        actual = (output.logits.detach(), gradient, image_fixture.decoder(model, backbone).layers[1].mlp.gate_proj.weight.grad)
        if expected is None:
            expected = tuple(value.clone() for value in actual)
        else:
            for value, reference in zip(actual, expected):
                torch.testing.assert_close(value, reference)
    assert len(loads) == 3


@pytest.mark.parametrize("backbone", ["qwen25_vl", "qwen3_vl", "llava_onevision"])
def test_custom_rules_generate_once_per_prefill_with_cache_and_repeat(backbone):
    torch = pytest.importorskip("torch")
    torch.manual_seed(195)
    model = image_fixture.model(backbone).eval()
    activation = image_fixture.activation(model, _custom_stage(backbone, 1, mid=True, phase="generate"))
    activation.install()
    if backbone == "qwen3_vl":
        from mllm_comp.backbones.qwen3_vl.provider_bridge import install_qwen3_vl_transformers_4573_bridge
        bridge = install_qwen3_vl_transformers_4573_bridge(model)
    else:
        bridge = activation._bridge
    args = image_fixture.inputs(torch, backbone)
    if backbone == "llava_onevision":
        args["inputs"] = args.pop("input_ids")
    outputs = []
    try:
        for _ in range(2):
            _CALLS.clear()
            with torch.no_grad():
                result = model.generate(**args, use_cache=True, max_new_tokens=2, min_new_tokens=2,
                    do_sample=False, eos_token_id=None, pad_token_id=0, return_dict_in_generate=True)
            outputs.append(result.sequences)
            assert len(_CALLS) == 2
            report = bridge.runtime_metadata()["latest_decoder_lifecycle"]
            assert report["selection_runs"] == 1 and report["decode_forward_calls"] == 1
            assert report["selection_repeated_during_decode"] is False
            assert report["boundaries"][0]["cache"]["touched_layers"] == [0, 1]
            assert all(layer.get_seq_length() == (368 if backbone == "llava_onevision" else 7)
                for layer in result.past_key_values.layers)
        torch.testing.assert_close(outputs[0], outputs[1])
    finally:
        activation.uninstall()


@pytest.mark.parametrize("backbone", ["qwen25_vl", "qwen3_vl", "llava_onevision"])
def test_direct_factory_rejects_rules_for_unsupported_image_method(backbone):
    from mllm_comp.core.errors import BackendIntegrationError
    from tests.integration.test_image_tome_positions_cpu import _stage as tome_stage
    stage = tome_stage(backbone, "mid")
    stage.criterion["scorer"] = {"callable": f"{__name__}:observed_norm"}
    with pytest.raises(BackendIntegrationError, match="does not support"):
        image_fixture.activation(object(), stage)


@pytest.mark.parametrize("backbone", ["qwen25_vl", "qwen3_vl", "llava_onevision"])
def test_direct_factory_reports_missing_rule_module_as_backend_error(backbone):
    from mllm_comp.core.errors import BackendIntegrationError, ConfigError
    stage = _custom_stage(backbone, 1)
    stage.criterion["scorer"]["callable"] = "missing_fastv_rule_module_for_test:score"
    with pytest.raises(BackendIntegrationError, match="Cannot load scorer rule") as error:
        image_fixture.activation(object(), stage)
    assert isinstance(error.value.__cause__, ConfigError)


@pytest.mark.parametrize("backbone", ["qwen25_vl", "qwen3_vl", "llava_onevision"])
def test_rule_metadata_keeps_activation_snapshot_when_stage_is_reused(backbone):
    stage = _custom_stage(backbone, 1, mode="scorer")
    expected = copy.deepcopy(stage.criterion["scorer"])
    activation = image_fixture.activation(image_fixture.model(backbone), stage)
    stage.criterion["scorer"]["callable"] = f"{__name__}:observed_selector"
    stage.criterion["scorer"]["params"]["power"] = 3.0
    assert activation.runtime_metadata()["rules"] == {"scorer": expected}
    assert activation._fastv_rules["scorer"].func is observed_norm
    assert activation._fastv_rules["scorer"].keywords == {"power": 2.0}
