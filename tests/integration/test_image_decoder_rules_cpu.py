"""Image decoder rules execute at their native source/application boundaries."""

import importlib

import pytest
from tests.fixtures import native_image as image_fixture


CALLS = []


def source_score(request):
    import torch
    if "image_indices" in request:
        count = len(request["image_indices"])
    elif "image_keys" in request:
        count = request["image_keys"].shape[0]
    else:
        count = request["visual_end"] - request["visual_start"]
    result = torch.arange(1, count + 1, dtype=torch.float32)
    CALLS.append(("source", request, result))
    return result


def priority_score(request, kind):
    import torch
    count = len(request["hidden_states"]) if kind == "dart" else len(request["image_indices"])
    result = torch.arange(1, count + 1, dtype=torch.float32)
    if kind == "finepruner":
        attention = request["scorer_attentions"][0]
        heads = attention.shape[0] if attention.ndim == 2 else attention.shape[1]
        result = result.expand(heads, -1).clone()
    CALLS.append(("priority", request, result))
    return result


def diversity_score(request):
    import torch
    result = -torch.arange(len(request["candidate_indices"]), dtype=torch.float32)
    CALLS.append(("diversity", request, result))
    return result


def coordinate_score(request):
    import torch
    row = torch.arange(1, len(request["image_indices"]) + 1, dtype=torch.float32)
    result = torch.stack((row, row.flip(0)))
    CALLS.append(("coordinate", request, result))
    return result


def bucket_labels(request):
    import torch
    result = (torch.arange(len(request["image_indices"])) - 1).clamp_min(0)
    CALLS.append(("buckets", request, result))
    return result


def last_representatives(request):
    import torch
    labels = request["labels"]
    result = torch.tensor([int((labels == label).nonzero()[-1]) for label in labels.unique()], device=labels.device)
    CALLS.append(("iac_selector", request, result))
    return result


def tail_heads(request):
    import torch
    result = torch.arange(request["scores"].shape[0] - request["keep_count"], request["scores"].shape[0])
    CALLS.append(("heads", request, result))
    return result


def tail_selector(request):
    import torch
    count = len(request["image_indices"])
    scores = request.get("scores")
    if scores is not None:
        # Every configured score in this fixture produces increasing, positive
        # priorities, distinct from native normalized attention probabilities.
        assert scores.ndim == 1 and bool((scores[1:] > scores[:-1]).all())
        assert float(scores.max()) >= scores.numel()
    protected = tuple(request.get("protected_indices", ()))
    eligible = [i for i in range(count) if i not in protected]
    room = request["keep_count"] - len(protected)
    result = torch.tensor([*protected, *(eligible[-room:] if room else [])], dtype=torch.long)
    CALLS.append(("selector", request, result))
    return result


def _rule(name, **params):
    return {"callable": f"{__name__}:{name}", "params": params}


def _rules(algorithm):
    result = {"selector": _rule("tail_selector")}
    if algorithm in {"pyramid_drop", "d2pruner", "btp", "halfv"}:
        result.update(source_scorer=_rule("source_score"), scorer=_rule("priority_score", kind=algorithm))
    elif algorithm == "dart":
        result.update(scorer=_rule("priority_score", kind=algorithm), diversity_scorer=_rule("diversity_score"))
    elif algorithm in {"feather", "sgl"}:
        result["scorer"] = _rule("source_score")
    elif algorithm == "ivc_prune":
        result.update(coordinate_scorer=_rule("coordinate_score"), semantic_scorer=_rule("priority_score", kind=algorithm))
    elif algorithm == "transprune":
        result.update(transition_scorer=_rule("source_score"), scorer=_rule("priority_score", kind=algorithm))
    else:
        result.update(grouper=_rule("bucket_labels"), iac_selector=_rule("last_representatives"),
            scorer=_rule("priority_score", kind=algorithm), head_selector=_rule("tail_heads"))
    return result


def _stage(backbone, algorithm, tmp_path, *, mid=False, phase="forward"):
    import torch
    short = {"pyramid_drop": "pyramid", "ivc_prune": "ivc"}.get(algorithm, algorithm)
    if algorithm == "d2pruner":
        from tests.integration.test_image_fastv_positions_cpu import _stage as fastv_stage
        stage = fastv_stage(backbone, 1, prune_layer=2, mid=mid, phase=phase)
        path = tmp_path / "native-d2-prior.pt"
        torch.save({"bias": torch.ones(16), "model_id": "/fixtures/native-tiny-image", "scorer_layer": 1}, path)
        stage.algorithm = stage.plugin = "d2pruner"
        stage.variant = "d2pruner.image"
        stage.binding = f"d2pruner.image.{backbone}"
        stage.config = {"prune_after_layer": 2, "bias_prior_path": str(path), "retain_ratio": 0.5}
    else:
        module = importlib.import_module(f"tests.integration.test_image_{short}_positions_cpu")
        if algorithm in {"dart", "ivc_prune"}:
            stage = module._stage(backbone, 2, prune_layer=2, mid=mid, phase=phase)
        elif algorithm == "sgl":
            stage = module._stage(backbone, 1, mid=mid)
        elif algorithm == "finepruner":
            stage = module._stage(backbone, (0, 1, 3), mid=mid, phase=phase)
        else:
            stage = module._stage(backbone, mid=mid, phase=phase)
    stage.operation["rules"] = _rules(algorithm)
    if algorithm == "dart":
        stage.config.update(pivot_image_tokens=1, pivot_text_tokens=1)
    return stage


def _stage_guide_scores(torch, backbone, model, raw, activation):
    import copy
    from mllm_comp.compressors.image.sgl import aggregate_sgl_visual_attention
    from tests.integration.test_image_sgl_positions_cpu import _guide
    guide = copy.deepcopy(model).eval()
    # Copy before installing the target bridge; this helper is called with an uninstalled target.
    inputs = dict(raw)
    if backbone == "llava_onevision":
        inputs["inputs"] = inputs.pop("input_ids")
    with torch.no_grad():
        output = guide.generate(**inputs, do_sample=False, use_cache=True, min_new_tokens=2,
            max_new_tokens=2, eos_token_id=None, pad_token_id=0, output_attentions=True,
            return_dict_in_generate=True)
    summary = aggregate_sgl_visual_attention(output.attentions, input_ids=raw["input_ids"],
        image_token_id=-200 if backbone == "llava_onevision" else 10, rules=activation._rules)
    return summary.visual_scores, _guide(activation, backbone)


@pytest.mark.parametrize("backbone", ["qwen25_vl", "qwen3_vl", "llava_onevision"])
@pytest.mark.parametrize("algorithm", ["pyramid_drop", "dart", "d2pruner", "feather", "ivc_prune", "btp", "halfv", "finepruner", "transprune", "sgl"])
def test_native_decoder_custom_decisions_replay_gradient_and_generation(backbone, algorithm, tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")
    from mllm_comp.compressors.image import decoder_selection as adapter
    names = {"pyramid_drop": ("pyramid_drop",), "btp": ("balanced_token_pruning_image_tokens",),
        "finepruner": ("finepruner_iac_image_tokens", "finepruner_arp_image_tokens",)}.get(algorithm, (algorithm + "_image_tokens",))
    checked_results = []

    def check_kernel(original):
        def checked(*args, **kwargs):
            actual = kwargs.get("hidden_states", args[0] if args else None)
            start = len(CALLS)
            result = original(*args, **kwargs)
            decisions = [item for item in CALLS[start:] if item[0] in {"selector", "iac_selector"}]
            if decisions:
                _, request, selected = decisions[-1]
                images = request["image_indices"].to(actual.device)
                selected_images = set(images.index_select(0, selected.to(images.device)).tolist())
                protected = set(kwargs.get("protected_indices", ()))
                image_set = set(images.tolist())
                keep = torch.tensor([i for i in range(actual.shape[0])
                    if i not in image_set or i in selected_images or i in protected], device=actual.device)
                torch.testing.assert_close(result.hidden_states, actual.index_select(0, keep))
            checked_results.append(result)
            return result
        return checked

    for name in names:
        monkeypatch.setattr(adapter, name, check_kernel(getattr(adapter, name)))
    for checkpoint in (None, False if backbone != "llava_onevision" else True):
        torch.manual_seed(911)
        model = image_fixture.model(backbone, num_decoder_layers=4).train()
        if checkpoint is not None:
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": checkpoint})
        stage = _stage(backbone, algorithm, tmp_path, mid=checkpoint is not None)
        activation = image_fixture.activation(model, stage)
        raw = image_fixture.inputs(torch, backbone)
        CALLS.clear()
        guide_scores, guide = _stage_guide_scores(torch, backbone, model, raw, activation) if algorithm == "sgl" else (None, None)
        activation.install()
        if backbone == "llava_onevision":
            activation._bridge.prepare_training_forward()
        try:
            if guide is not None:
                guide.stage_scores(guide_scores)
            labels = torch.full_like(raw["input_ids"], -100)
            labels[0, -1] = raw["input_ids"][0, -1]
            checked_results.clear()
            output = model(**raw, labels=labels, use_cache=False)
            count = len(CALLS)
            assert count and checked_results
            expected = ({"source", "priority", "selector"} if algorithm in {"pyramid_drop", "d2pruner", "btp", "halfv", "transprune"}
                else {"priority", "diversity", "selector"} if algorithm == "dart"
                else {"source", "selector"} if algorithm in {"feather", "sgl"}
                else {"coordinate", "priority", "selector"} if algorithm == "ivc_prune"
                else {"buckets", "iac_selector", "priority", "heads", "selector"})
            assert expected.issubset({item[0] for item in CALLS}), (algorithm, expected, [item[0] for item in CALLS])
            output.loss.backward()
            assert len(CALLS) == count, "checkpoint must replay decisions without invoking Python rules"
            pixels = raw["images" if backbone == "llava_onevision" else "pixel_values"]
            assert pixels.grad is not None and torch.isfinite(pixels.grad).all() and pixels.grad.abs().sum() > 0
            assert activation.runtime_metadata()["canonical_evidence_applicable"] is False
            if guide is not None:
                guide.finish_target(success=True)
        finally:
            activation.uninstall()
        if checkpoint is None:
            continue
        model.eval()
        activation = image_fixture.activation(model, _stage(backbone, algorithm, tmp_path, mid=True, phase="generate"))
        if algorithm == "sgl":
            from tests.integration.test_image_sgl_positions_cpu import _guide
            guide = _guide(activation, backbone)
        activation.install()
        outputs, counts = [], []
        try:
            for _ in range(2):
                CALLS.clear()
                if guide is not None:
                    guide.stage_scores(guide_scores)
                inputs = dict(raw)
                if backbone == "llava_onevision":
                    inputs["inputs"] = inputs.pop("input_ids")
                with torch.no_grad():
                    output = model.generate(**inputs, use_cache=True, max_new_tokens=2, min_new_tokens=2,
                        do_sample=False, eos_token_id=None, pad_token_id=0, return_dict_in_generate=True)
                outputs.append(output.sequences)
                counts.append(len(CALLS))
                assert output.past_key_values is not None
                if guide is not None:
                    guide.finish_target(success=True)
            assert counts[0] == counts[1] and counts[0] > 0
            torch.testing.assert_close(outputs[0], outputs[1])
        finally:
            activation.uninstall()
