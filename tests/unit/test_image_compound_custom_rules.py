"""Compound rules preserve actual weighted values, partition mass and replay."""

import pytest

torch = pytest.importorskip("torch")

from mllm_comp.compressors.image.nuwa import NuwaSpec, nuwa_stage1_image_tokens, nuwa_stage2_image_tokens
from mllm_comp.compressors.image.restore import RestoreSpec, restore_image_tokens, replay_restore_values, restore_attention_bias


def _nuwa(rules=None):
    values = torch.arange(18, dtype=torch.float32).reshape(9, 2).requires_grad_()
    return values, nuwa_stage1_image_tokens(values, torch.eye(9), torch.eye(9), torch.ones(1, 9, 2),
        grid_height=3, grid_width=3,
        spec=NuwaSpec(stage1_total_tokens=2, region_size=2, top_n_per_region=1), rules=rules)


def test_nuwa_rules_select_regions_and_freeze_overlapping_weight_collectors():
    called = set()
    def record(name, fn):
        def rule(r):
            called.add(name)
            return fn(r)
        return rule
    def weights(r):
        result = torch.ones_like(r["default_weights"]) / 9
        result[1] *= 2
        return result
    rules = {
        "stage1_scorer": record("stage1_scorer", lambda r: torch.arange(9).float()),
        "regional_selector": record("regional_selector", lambda r: r["candidates"][:r["keep_count"]]),
        "stage1_selector": record("stage1_selector", lambda r: r["candidates"][:r["keep_count"]]),
        "aggregation_weights": record("aggregation_weights", weights),
    }
    values, result = _nuwa(rules)
    assert result.selected_indices.tolist() == [0, 2]
    expected = torch.stack((values.mean(0), values.mean(0) * 2))
    torch.testing.assert_close(result.tokens, expected)
    assert called == set(rules)
    assert result.metadata["rules_changed"] and result.metadata["pillar_tokens"] is None
    branch = (values.detach() + 3).requires_grad_()
    replay = result.aggregation_weights @ branch
    torch.testing.assert_close(replay, torch.stack((branch.mean(0), branch.mean(0) * 2)))
    (result.tokens.sum() + replay.sum()).backward()
    torch.testing.assert_close(values.grad, torch.full_like(values, 1 / 3))
    torch.testing.assert_close(branch.grad, torch.full_like(branch, 1 / 3))


def test_nuwa_stage2_local_selection_preserves_protected_and_nonvisual_tokens():
    values = torch.arange(24, dtype=torch.float32).reshape(6, 4).requires_grad_()
    def select(r):
        torch.testing.assert_close(r["scores"], torch.tensor([3., 2., 1.]))
        assert r["selectable_indices"].tolist() == [1, 2, 3]
        return torch.tensor([0, 2])
    result = nuwa_stage2_image_tokens(values, image_indices=[1, 2, 3, 4], protected_indices=[4],
        image_attention_scores=torch.tensor([0., 4., 2., 1.]),
        spec=NuwaSpec(stage2_retain_ratio=.5), rules={
            "stage2_scorer": lambda r: torch.tensor([3., 2., 1.]), "stage2_selector": select})
    assert result.kept_indices.tolist() == [0, 1, 3, 4, 5]
    torch.testing.assert_close(result.hidden_states, values[[0, 1, 3, 4, 5]])
    result.hidden_states.sum().backward()
    assert values.grad[2].sum() == 0


def _restore(rules=None):
    values = torch.arange(32, dtype=torch.float32).reshape(8, 4).requires_grad_()
    return values, restore_image_tokens(values, torch.eye(8), torch.eye(8),
        spec=RestoreSpec(target_count=4), rules=rules)


def test_restore_rules_keep_partition_mass_and_replay_the_same_nonlinear_aggregation():
    seen = set()
    def wrap(name, fn):
        def rule(r):
            seen.add(name)
            return fn(r)
        return rule
    rules = {
        "pristine_scorer": wrap("pristine_scorer", lambda r: torch.arange(8).float()),
        "pristine_selector": wrap("pristine_selector", lambda r: torch.tensor([0, 1])),
        "anchor_scorer": wrap("anchor_scorer", lambda r: torch.arange(6).float()),
        "anchor_selector": wrap("anchor_selector", lambda r: torch.tensor([0, 5])),
        "grouper": wrap("grouper", lambda r: [(0, 1, 2), (3, 4, 5)]),
        "aggregator": wrap("aggregator", lambda r: r["values"].square().mean(0)),
    }
    values, result = _restore(rules)
    expected = torch.stack((values[0], values[1], values[2:5].square().mean(0), values[5:8].square().mean(0)))
    torch.testing.assert_close(result.tokens, expected)
    assert result.representative_indices.tolist() == [0, 1, 3, 6]
    assert result.token_sizes.tolist() == [1, 1, 3, 3]
    assert result.membership.sum(0).tolist() == [1] * 8
    assert seen == set(rules)
    branch = (values.detach() + 1).requires_grad_()
    seen.clear()
    replay = replay_restore_values(branch, result, rules=rules)
    assert seen == {"aggregator"}
    torch.testing.assert_close(replay, torch.stack((branch[0], branch[1], branch[2:5].square().mean(0), branch[5:8].square().mean(0))))
    (result.tokens.sum() + replay.sum()).backward()
    torch.testing.assert_close(values.grad[:2], torch.ones_like(values[:2]))
    torch.testing.assert_close(values.grad[2:], values.detach()[2:] * (2 / 3))
    torch.testing.assert_close(branch.grad[2:], branch.detach()[2:] * (2 / 3))


def test_restore_calibration_rule_uses_real_positions_and_sizes():
    queries, keys, sizes = torch.tensor([[8, 9]]), torch.tensor([[0, 3, 8]]), torch.tensor([[1., 2., 4.]])
    def bias(r):
        assert torch.equal(r["query_positions"], queries)
        return r["key_sizes"].log()[:, None, None].expand(1, 1, 2, 3)
    result = restore_attention_bias(queries, keys, sizes, torch.tensor([.1, .2]), rules={"calibration_bias": bias})
    torch.testing.assert_close(result, sizes.log()[:, None, None].expand(1, 1, 2, 3))


@pytest.mark.parametrize("method", ["nuwa", "restore"])
def test_empty_rules_preserve_exact_default_compound_output(method):
    kernel = _nuwa if method == "nuwa" else _restore
    _, default = kernel()
    _, explicit = kernel({})
    torch.testing.assert_close(default.tokens, explicit.tokens, rtol=0, atol=0)
    assert default.layout_delta == explicit.layout_delta


@pytest.mark.parametrize("rules", [
    {"aggregation_weights": lambda r: torch.zeros(1)},
    {"regional_selector": lambda r: torch.tensor([8])},
    {"stage1_selector": lambda r: torch.tensor([0, 0])},
])
def test_invalid_nuwa_rule_output_is_rejected(rules):
    with pytest.raises(ValueError):
        _nuwa(rules)


@pytest.mark.parametrize("rules", [
    {"grouper": lambda r: [(0, 1, 2), (2, 3, 4)]},
    {"aggregator": lambda r: r["values"].detach().mean(0)},
    {"anchor_scorer": lambda r: torch.zeros(1)},
])
def test_invalid_restore_rule_output_is_rejected(rules):
    with pytest.raises(ValueError):
        _restore(rules)
