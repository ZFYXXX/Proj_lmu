"""Custom PACT/SparseVLM decisions survive native application and replay."""

import importlib
import pytest
from tests.fixtures import native_image as image_fixture
from tests.integration import test_image_pact_positions_cpu as pact_checks
from tests.integration import test_image_sparsevlm_positions_cpu as sparse_checks

CALLS = []


def source(request):
    import torch
    CALLS.append("source_scorer")
    count = request["image_indices"].numel() if "image_indices" in request else request["image_keys"].shape[0]
    tensor = request.get("query_states", request.get("instruction_query"))
    return torch.arange(count, device=tensor.device, dtype=torch.float32) + 1


def scores(request):
    import torch
    CALLS.append("scorer")
    return torch.arange(request["image_indices"].numel(), device=request["hidden_states"].device, dtype=torch.float32) + 1


def select(request):
    import torch
    CALLS.append("selector")
    protected = list(request.get("protected_indices", ()))
    available = [i for i in range(request["image_indices"].numel()) if i not in protected]
    needed = request["keep_count"] - len(protected)
    return torch.tensor([*protected, *(available[-needed:] if needed else [])], device=request["scores"].device, dtype=torch.long)


def pact_groups(request):
    import torch
    CALLS.append("grouper")
    count = request["center_count"]
    centers = torch.arange(count, device=request["kept_keys"].device)
    labels = torch.zeros_like(request["default_labels"])
    labels[:count] = centers
    return {"centers": centers, "labels": labels}


def raters(request):
    import torch
    CALLS.append("rater_scorer")
    return torch.arange(request["candidate_indices"].numel(), device=request["input_embeddings"].device, dtype=torch.float32)


def rater_select(request):
    import torch
    CALLS.append("rater_selector")
    return torch.arange(request["keep_count"], device=request["scores"].device)


def recycle(request):
    CALLS.append("recycle_selector")
    return request["candidate_indices"][:request["keep_count"]]


def cluster_scores(request):
    import torch
    CALLS.append("cluster_scorer")
    return torch.arange(request["tokens"].shape[0], device=request["tokens"].device, dtype=torch.float32)


def cluster_select(request):
    import torch
    CALLS.append("cluster_selector")
    return torch.arange(request["keep_count"], device=request["tokens"].device)


def sparse_groups(request):
    import torch
    CALLS.append("grouper")
    labels = torch.arange(request["tokens"].shape[0], device=request["tokens"].device) % request["group_count"]
    labels[request["centers"]] = torch.arange(request["group_count"], device=labels.device)
    return labels


def weighted(request):
    import torch
    values = request["values"]
    weights = torch.arange(1, values.shape[0] + 1, device=values.device, dtype=values.dtype)
    return (values * weights[:, None]).sum(0) / weights.sum()


def _rules(algorithm):
    functions = {"source_scorer": "source", "scorer": "scores", "selector": "select", "aggregator": "weighted",
        "grouper": "pact_groups" if algorithm == "pact" else "sparse_groups"}
    if algorithm == "sparsevlm":
        functions.update(rater_scorer="raters", rater_selector="rater_select", recycle_selector="recycle",
            cluster_scorer="cluster_scores", cluster_selector="cluster_select")
    return {name: {"callable": f"{__name__}:{function}"} for name, function in functions.items()}


def _stage(backbone, algorithm, *, mid=False, phase="forward"):
    stage = (pact_checks._stage(backbone, (1,) if mid else (2,), mid=mid, phase=phase) if algorithm == "pact"
        else sparse_checks._stage(backbone, (0, 2) if mid else (2, 3), mid=mid, phase=phase))
    stage.operation["rules"] = _rules(algorithm)
    return stage


@pytest.mark.parametrize("backbone", ["qwen25_vl", "qwen3_vl", "llava_onevision"])
@pytest.mark.parametrize("algorithm", ["pact", "sparsevlm"])
def test_custom_cluster_plans_native_values_checkpoint_and_cache(backbone, algorithm, monkeypatch):
    torch = pytest.importorskip("torch")
    adapter = importlib.import_module("mllm_comp.compressors.image.decoder_selection")
    name = algorithm + "_image_tokens"
    kernel = getattr(adapter, name)
    results = []

    def checked(*args, **kwargs):
        actual = kwargs.get("hidden_states", args[0] if args else None)
        result = kernel(*args, **kwargs)
        groups = [[i for i, target in result.layout_delta.old_to_new.items() if target == j]
            for j in range(result.layout_delta.result_count)]
        expected = []
        for group in groups:
            values = actual[group]
            weights = torch.arange(1, len(group) + 1, device=values.device, dtype=values.dtype)
            expected.append((values * weights[:, None]).sum(0) / weights.sum())
        torch.testing.assert_close(result.hidden_states, torch.stack(expected))
        if algorithm == "pact":
            torch.testing.assert_close(result.token_sizes, result.token_sizes.new_tensor([len(g) for g in groups]))
        results.append(result)
        return result

    monkeypatch.setattr(adapter, name, checked)
    for checkpoint in (None, False, True):
        torch.manual_seed(941)
        model = image_fixture.model(backbone, num_decoder_layers=4).train()
        if checkpoint is not None:
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": checkpoint})
        activation = image_fixture.activation(model, _stage(backbone, algorithm, mid=checkpoint is not None))
        activation.install()
        if backbone == "llava_onevision":
            activation._bridge.prepare_training_forward()
        CALLS.clear()
        results.clear()
        raw = image_fixture.inputs(torch, backbone)
        try:
            labels = torch.full_like(raw["input_ids"], -100)
            labels[0, -1] = raw["input_ids"][0, -1]
            output = model(**raw, labels=labels, use_cache=False)
            assert results
            assert set(_rules(algorithm)) - {"aggregator"} <= set(CALLS)
            count = len(CALLS)
            output.loss.backward()
            assert len(CALLS) == count
            pixels = raw["images" if backbone == "llava_onevision" else "pixel_values"]
            assert pixels.grad is not None and torch.isfinite(pixels.grad).all() and pixels.grad.abs().sum() > 0
            if backbone == "qwen3_vl":
                for merger in model.model.visual.deepstack_merger_list:
                    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in merger.parameters())
            assert activation.runtime_metadata()["canonical_evidence_applicable"] is False
        finally:
            activation.uninstall()
    model.eval()
    activation = image_fixture.activation(model, _stage(backbone, algorithm, mid=True, phase="generate"))
    activation.install()
    sequences, counts = [], []
    try:
        for _ in range(2):
            CALLS.clear()
            inputs = dict(raw)
            if backbone == "llava_onevision":
                inputs["inputs"] = inputs.pop("input_ids")
            with torch.no_grad():
                output = model.generate(**inputs, use_cache=True, max_new_tokens=2, min_new_tokens=2,
                    do_sample=False, eos_token_id=None, pad_token_id=0, return_dict_in_generate=True)
            sequences.append(output.sequences)
            counts.append(len(CALLS))
            assert output.past_key_values is not None
        assert counts[0] == counts[1] > 0
        assert torch.equal(*sequences)
    finally:
        activation.uninstall()
