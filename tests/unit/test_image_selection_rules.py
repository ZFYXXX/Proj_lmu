"""Custom rules change the original index planners, retaining each outer contract."""

import pytest


def tail_selector(request):
    import torch
    feature = next(request[name] for name in (
        "tokens", "selection_features", "projected_tokens", "normalized_keys") if name in request)
    protected = tuple(request.get("protected_indices", ()))
    eligible = [i for i in range(len(feature)) if i not in protected]
    count = request["keep_count"] - len(protected)
    return torch.tensor([*protected, *(eligible[-count:] if count else [])], device=feature.device)


def _kernel(algorithm, *, rules=None, identity=False):
    import torch
    tokens = torch.arange(48, dtype=torch.float32).reshape(8, 6).requires_grad_()
    attention = torch.eye(8)
    keys = torch.randn(8, 6)
    target = 8 if identity else 3
    if algorithm == "gap":
        from mllm_comp.compressors.image.gap import gap_image_tokens, GAPSpec
        result = gap_image_tokens(tokens, attention, spec=GAPSpec(target_count=target), rules=rules)
    elif algorithm == "divprune":
        from mllm_comp.compressors.image.divprune import divprune_image, DivPruneSpec
        result = divprune_image(tokens, spec=DivPruneSpec(target_count=target), rules=rules)
    elif algorithm == "vispruner":
        from mllm_comp.compressors.image.vispruner import vispruner_image_tokens, VisPrunerSpec
        result = vispruner_image_tokens(tokens, keys, attention,
            spec=VisPrunerSpec(target_count=target), rules=rules)
    elif algorithm == "mmtok":
        from mllm_comp.compressors.image.mmtok import mmtok_image_tokens, MMTokSpec
        result = mmtok_image_tokens(tokens, keys, torch.randn(2, 6),
            spec=MMTokSpec(target_count=target), rules=rules)
    elif algorithm == "zoo_prune":
        from mllm_comp.compressors.image.zoo_prune import zoo_prune_image_tokens, ZOOPruneSpec
        result = zoo_prune_image_tokens(tokens, tokens, (torch.nn.Linear(6, 6),),
            spec=ZOOPruneSpec(target_count=target, num_directions=2,
                projector_proxy="full", low_rank_rank=2), rules=rules)
    else:
        from mllm_comp.compressors.image.saint import saint_image_tokens, SAINTSpec
        result = saint_image_tokens(tokens, keys,
            spec=SAINTSpec(target_count=target, group_size=1), rules=rules)
    return tokens, result


@pytest.mark.parametrize("algorithm", ["gap", "divprune", "vispruner", "mmtok", "zoo_prune", "saint"])
def test_selection_values_gradients_and_identity(algorithm):
    torch = pytest.importorskip("torch")
    torch.manual_seed(904)
    _, default = _kernel(algorithm)
    torch.manual_seed(904)
    _, empty = _kernel(algorithm, rules={})
    torch.testing.assert_close(default.tokens, empty.tokens, rtol=0, atol=0)
    assert default.layout_delta.old_to_new == empty.layout_delta.old_to_new
    tokens, result = _kernel(algorithm, rules={"selector": tail_selector})
    torch.testing.assert_close(result.tokens, tokens[-3:])
    assert result.layout_delta.result_count == 3
    assert result.metadata["rules_changed"]
    result.tokens.sum().backward()
    torch.testing.assert_close(tokens.grad[:5], torch.zeros_like(tokens[:5]))
    torch.testing.assert_close(tokens.grad[5:], torch.ones_like(tokens[5:]))

    def forbidden(request):
        raise AssertionError("An identity operation must bypass custom decisions")

    tokens, result = _kernel(algorithm, rules={"scorer": forbidden, "selector": forbidden}, identity=True)
    torch.testing.assert_close(result.tokens, tokens)


@pytest.mark.parametrize("algorithm", ["gap", "divprune", "vispruner", "mmtok", "zoo_prune", "saint"])
def test_invalid_score_and_selection_result(algorithm):
    torch = pytest.importorskip("torch")
    with pytest.raises(ValueError, match="shape"):
        _kernel(algorithm, rules={"scorer": lambda request: torch.zeros(1)})
    with pytest.raises(ValueError, match="unique indices"):
        _kernel(algorithm, rules={"selector": lambda request: torch.zeros(request["keep_count"], dtype=torch.long)})


@pytest.mark.parametrize("algorithm", ["gap", "divprune", "vispruner", "mmtok", "zoo_prune", "saint"])
def test_scorer_is_consumed_by_selector_once_per_decision(algorithm):
    torch = pytest.importorskip("torch")
    calls = []

    def scorer(request):
        calls.append("score")
        if algorithm == "mmtok":
            shape = (request["text_embeddings"].shape[0] + 8, 8)
        elif algorithm == "divprune":
            shape = (8, 8)
        elif algorithm == "saint":
            shape = ((request["normalized_keys"].shape[0] + 1) // 2,)
        else:
            shape = (8,)
        return torch.full(shape, 0.75)

    def selector(request):
        calls.append("select")
        assert not request["scores"].requires_grad
        torch.testing.assert_close(request["scores"], torch.full_like(request["scores"], 0.75))
        return tail_selector(request)

    _kernel(algorithm, rules={"scorer": scorer, "selector": selector})
    assert calls == ["score", "select"] * (len(calls) // 2)
    assert calls


def test_gap_custom_selector_skips_default_topk_and_retains_protected_tokens(monkeypatch):
    torch = pytest.importorskip("torch")
    from mllm_comp.compressors.image.gap import GAPSpec, gap_image_tokens

    tokens = torch.arange(12, dtype=torch.float32).reshape(6, 2)
    attention = torch.eye(6)

    def forbidden(*args, **kwargs):
        raise AssertionError("Custom selection must replace the default Top-K")

    def selector(request):
        torch.testing.assert_close(request["scores"], torch.full((6,), 1.0 / 6))
        assert request["protected_indices"] == (0,)
        assert request["keep_count"] == 2
        return torch.tensor([4, 0])

    monkeypatch.setattr(torch.Tensor, "topk", forbidden)
    result = gap_image_tokens(
        tokens, attention, spec=GAPSpec(target_count=2), protected_indices=(0,),
        rules={"selector": selector},
    )

    assert result.indices.tolist() == [0, 4]
    torch.testing.assert_close(result.tokens, tokens[[0, 4]])
    assert result.layout_delta.protected_indices == (0,)
