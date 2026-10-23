"""Replace FastV decisions while retaining its budget and layout contract."""

import pytest

from mllm_comp.compressors.image.fastv import FastVSpec, fastv_image_tokens

torch = pytest.importorskip("torch")


def feature_norm(request, power=2.0):
    values = request["hidden_states"].index_select(0, request["image_indices"])
    return values.float().abs().pow(power).sum(-1)


def low_score_select(request):
    protected = list(request["protected_image_positions"])
    ranked = torch.argsort(request["scores"], stable=True).tolist()
    selected = protected + [i for i in ranked if i not in protected]
    return list(reversed(selected[:request["keep_count"]]))


def _inputs():
    hidden = torch.arange(24, dtype=torch.float32).reshape(8, 3).requires_grad_()
    keys = torch.arange(16, dtype=torch.float32).reshape(8, 1, 2) / 10
    return dict(hidden_states=hidden, image_indices=[2, 3, 4, 5],
        scorer_query=torch.ones(1, 2), attention_keys=keys,
        spec=FastVSpec(prune_ratio=0.5), protected_indices=[2])


def test_custom_score_and_selection_keep_budget_protected_source_order_and_gather_gradient():
    values = _inputs()
    original = fastv_image_tokens(**values)
    result = fastv_image_tokens(**values, scorer=feature_norm, selector=low_score_select)
    assert result.kept_image_indices.tolist() == [2, 3]
    assert result.kept_image_indices.tolist() != original.kept_image_indices.tolist()
    assert result.kept_indices.tolist() == [0, 1, 2, 3, 6, 7]
    assert result.layout_delta.protected_indices == (0, 1, 2, 6, 7)
    torch.testing.assert_close(result.image_scores, values["hidden_states"][2:6].square().sum(-1))
    assert result.metadata["ranking"] == "custom_scorer_custom_selector"
    assert result.metadata["score_source"] == "custom"
    result.hidden_states.sum().backward()
    expected = torch.ones_like(values["hidden_states"])
    expected[4:6] = 0
    torch.testing.assert_close(values["hidden_states"].grad, expected)


def test_layer_zero_custom_scorer_uses_actual_values_without_attention():
    values = _inputs()
    values.update(spec=FastVSpec(prune_layer=0), scorer_query=None, attention_keys=None)
    calls = []
    def score(request):
        assert request["scorer_query"] is request["attention_keys"] is None
        assert not torch.is_grad_enabled()
        calls.append(request["hidden_states"])
        return feature_norm(request)
    result = fastv_image_tokens(**values, scorer=score)
    assert calls == [values["hidden_states"]]
    assert result.kept_image_indices.tolist() == [2, 5]
    assert result.metadata["ranking"] == "custom_scorer"
    assert fastv_image_tokens(**values).metadata["ranking"] == "seeded_random"


def test_default_equivalent_python_rules_preserve_output_scores_and_layout():
    values = _inputs()
    def score(request):
        query = request["scorer_query"].unsqueeze(0)
        keys = request["attention_keys"]
        logits = torch.einsum("qhd,thd->qht", query, keys) / query.shape[-1] ** 0.5
        return logits.float().softmax(-1).mean((0, 1))[request["image_indices"]]
    def select(request):
        protected = list(request["protected_image_positions"])
        ranked = torch.argsort(request["scores"], descending=True, stable=True).tolist()
        return (protected + [i for i in ranked if i not in protected])[:request["keep_count"]]
    baseline = fastv_image_tokens(**values)
    for rules in ({"scorer": score}, {"selector": select}, {"scorer": score, "selector": select}):
        result = fastv_image_tokens(**values, **rules)
        torch.testing.assert_close(result.hidden_states, baseline.hidden_states, rtol=0, atol=0)
        torch.testing.assert_close(result.image_scores, baseline.image_scores)
        assert result.layout_delta == baseline.layout_delta


def test_identity_does_not_call_custom_rules():
    def forbidden(request):
        raise AssertionError("identity must not execute custom rules")
    values = _inputs()
    values["spec"] = FastVSpec(prune_ratio=0)
    result = fastv_image_tokens(**values, scorer=forbidden, selector=forbidden)
    assert result.metadata["ranking"] == "identity"
    assert result.layout_delta.is_identity


@pytest.mark.parametrize("scores", [torch.zeros(3), torch.zeros(4, 1), torch.ones(4, dtype=torch.long),
    torch.tensor([0., 1., float("nan"), 2.]), torch.tensor([0., 1., float("inf"), 2.])])
def test_custom_scorer_rejects_invalid_scores(scores):
    with pytest.raises(ValueError, match="finite floating scores"):
        fastv_image_tokens(**_inputs(), scorer=lambda request: scores)


@pytest.mark.parametrize("indices", [[0], [0, 0], [0, 4], [1, 2], [0.0, 1.0], [True, False], [[0, 1]]])
def test_custom_selector_rejects_invalid_budget_indices_or_protection(indices):
    with pytest.raises(ValueError, match="FastV selector"):
        fastv_image_tokens(**_inputs(), selector=lambda request: indices)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for cross-device rule output")
def test_custom_score_output_moves_to_actual_value_device():
    values = _inputs()
    values["hidden_states"] = values["hidden_states"].detach().cuda().requires_grad_()
    result = fastv_image_tokens(**values, scorer=lambda request: torch.arange(4, dtype=torch.float32))
    assert result.image_scores.device == values["hidden_states"].device
    assert result.kept_indices.device == values["hidden_states"].device
    result.hidden_states.sum().backward()
    assert values["hidden_states"].grad is not None
