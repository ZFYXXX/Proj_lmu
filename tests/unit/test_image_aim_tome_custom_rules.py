"""AIM/ToMe custom matching and aggregation retain original replay semantics."""
import pytest

from mllm_comp.compressors.image.aim import AIMSpec, aim_image_tokens, replay_aim_image_tokens
from mllm_comp.compressors.image.tome import (
    tome_merge_image, tome_preserve_source_order, tome_restore_dense_boundary,
    tome_merge_reference, replay_tome_image,
)


def ranked_scores(request):
    import torch
    scores = request["scores"]
    return torch.arange(scores.numel(), device=scores.device, dtype=scores.dtype).reshape_as(scores)


def last_sources_first_destination(request):
    return [(source, request["destination_indices"][0])
            for source in request["source_indices"][-request["merge_count"]:]]


def destination_weighted(request, *, weight=3.0):
    values = request["values"]
    return (weight * values[0] + values[1:].sum(0)) / (weight + len(values) - 1)


def _rules():
    return {"scorer": ranked_scores, "grouper": last_sources_first_destination,
            "aggregator": destination_weighted}


def _fail(_):
    raise AssertionError("saved replay or identity must not call a decision rule")


def _aim_independent(values, result, *, weighted):
    import torch
    current = values[list(result.eligible_indices)].float()
    for pairs in result.merge_rounds:
        selected = {source for source, _ in pairs}
        rows = []
        for destination in range(len(current)):
            if destination in selected:
                continue
            members = [destination, *(source for source, target in pairs if target == destination)]
            group = current[members]
            rows.append(destination_weighted({"values": group}) if weighted and len(members) > 1 else group.mean(0))
        current = torch.stack(rows)
    current = current.to(values.dtype)
    if result.layout_delta.protected_indices:
        current = torch.cat((current, values[list(result.layout_delta.protected_indices)]))
    return current[list(result.output_order)]


@pytest.mark.parametrize("protected", [(), (1, 8)])
def test_aim_custom_rounds_preserve_exact_groups_protection_and_replay_gradients(protected):
    torch = pytest.importorskip("torch")
    torch.manual_seed(119)
    x = torch.randn(11, 4, requires_grad=True)
    result = aim_image_tokens(x, spec=AIMSpec(target_count=4), protected_indices=protected, rules=_rules())
    assert result.layout_delta.result_count == 4
    assert len(result.merge_rounds) >= 2
    expected_input = x.detach().clone().requires_grad_()
    expected = _aim_independent(expected_input, result, weighted=True)
    torch.testing.assert_close(result.tokens, expected, rtol=0, atol=0)
    result.tokens.square().sum().backward()
    expected.square().sum().backward()
    torch.testing.assert_close(x.grad, expected_input.grad, rtol=0, atol=0)
    replay_input = x.detach().clone().requires_grad_()
    replay = replay_aim_image_tokens(replay_input, result,
        rules={"scorer": _fail, "grouper": _fail, "aggregator": destination_weighted})
    torch.testing.assert_close(replay, result.tokens, rtol=0, atol=0)
    replay.square().sum().backward()
    torch.testing.assert_close(replay_input.grad, x.grad, rtol=0, atol=0)
    for source in protected:
        index = result.layout_delta.old_to_new[source]
        torch.testing.assert_close(result.tokens[index], x[source], rtol=0, atol=0)
        assert result.token_sizes[index].item() == 1
    assert sum(result.token_sizes.tolist()) == len(x)
    assert result.metadata["rules_changed"] is True


def test_aim_default_remains_current_node_mean_not_final_member_mass_mean():
    torch = pytest.importorskip("torch")
    x = torch.tensor([[2.], [11.], [4.], [7.], [9.], [3.], [14.]])
    # A fixed matcher produces an uneven multi-round lineage.
    result = aim_image_tokens(x, spec=AIMSpec(target_count=2),
        rules={"grouper": last_sources_first_destination})
    expected = _aim_independent(x, result, weighted=False)
    torch.testing.assert_close(result.tokens, expected, rtol=0, atol=0)
    final_groups = [[source for source, target in result.layout_delta.old_to_new.items() if target == i]
                    for i in range(2)]
    member_mean = torch.stack([x[group].mean(0) for group in final_groups])
    assert not torch.allclose(result.tokens, member_mean)
    default = aim_image_tokens(x, spec=AIMSpec(target_count=2))
    explicit = aim_image_tokens(x, spec=AIMSpec(target_count=2), rules={})
    torch.testing.assert_close(default.tokens, explicit.tokens, rtol=0, atol=0)
    assert default.layout_delta.old_to_new == explicit.layout_delta.old_to_new
    torch.testing.assert_close(replay_aim_image_tokens(x, default), default.tokens, rtol=0, atol=0)
    assert "rules_changed" not in default.metadata


def _tome_independent(values, result, *, weighted):
    import torch
    rows = []
    sizes = result.input_token_sizes
    batched = values if result.is_batched else values.unsqueeze(0)
    for batch_index, (pairs, output_groups) in enumerate(zip(result.merge_pairs, result.source_groups)):
        destinations = set(destination for _, destination in pairs)
        outputs = []
        for group in output_groups:
            destination = next((i for i in group if i in destinations), None)
            if destination is None:
                outputs.append(batched[batch_index, group[0]])
                continue
            indices = [destination, *(source for source, target in pairs if target == destination)]
            members = batched[batch_index, indices]
            if weighted:
                outputs.append(destination_weighted({"values": members}))
            else:
                mass = sizes[batch_index, indices]
                outputs.append((members * mass).sum(0) / mass.sum())
        rows.append(torch.stack(outputs))
    output = torch.stack(rows)
    return output if result.is_batched else output[0]


@pytest.mark.parametrize("batched", [False, True])
def test_tome_custom_edges_mass_and_value_replay_include_legalized_order(batched):
    torch = pytest.importorskip("torch")
    torch.manual_seed(120)
    x = torch.randn((2, 9, 4) if batched else (9, 4), requires_grad=True)
    keys = torch.randn((2, 9, 2, 3) if batched else (9, 2, 3))
    sizes = torch.arange(1., 10.).expand(2, -1).clone() if batched else torch.arange(1., 10.)
    result = tome_merge_image(x, keys, r=3, token_sizes=sizes, protected_indices=(0, 1), rules=_rules())
    if not batched:
        result = tome_preserve_source_order(result)
    assert result.merge_pairs[0] == ((4, 3), (6, 3), (8, 3))
    expected_input = x.detach().clone().requires_grad_()
    expected = _tome_independent(expected_input, result, weighted=True)
    torch.testing.assert_close(result.hidden_states, expected, rtol=0, atol=0)
    result.hidden_states.square().sum().backward()
    expected.square().sum().backward()
    torch.testing.assert_close(x.grad, expected_input.grad, rtol=0, atol=0)
    replay_input = x.detach().clone().requires_grad_()
    replay = replay_tome_image(replay_input, result,
        rules={"scorer": _fail, "grouper": _fail, "aggregator": destination_weighted})
    torch.testing.assert_close(replay, result.hidden_states, rtol=0, atol=0)
    replay.square().sum().backward()
    torch.testing.assert_close(replay_input.grad, x.grad, rtol=0, atol=0)
    expected_total = sizes.sum(-1)
    torch.testing.assert_close(result.token_sizes.squeeze(-1).sum(-1), expected_total)
    torch.testing.assert_close(result.proportional_attention_bias.squeeze(), result.token_sizes.squeeze().log())
    dense = tome_restore_dense_boundary(replay, result.layout_deltas)
    assert dense.shape == x.shape
    for batch_index, delta in enumerate(result.layout_deltas):
        for source in (0, 1):
            actual = result.hidden_states[batch_index, delta.old_to_new[source]] if batched else result.hidden_states[delta.old_to_new[source]]
            original = x[batch_index, source] if batched else x[source]
            torch.testing.assert_close(actual, original, rtol=0, atol=0)
    assert result.metadata["merge_value"] == "custom"


def test_tome_default_weighted_math_and_fixed_key_reference_do_not_use_custom_aggregator():
    torch = pytest.importorskip("torch")
    x = torch.arange(28.).reshape(7, 4).requires_grad_()
    keys = torch.arange(42.).reshape(7, 2, 3) + 1
    sizes = torch.arange(1., 8.)
    default = tome_merge_image(x, keys, r=2, token_sizes=sizes)
    explicit = tome_merge_image(x, keys, r=2, token_sizes=sizes, rules={})
    torch.testing.assert_close(default.hidden_states, explicit.hidden_states, rtol=0, atol=0)
    torch.testing.assert_close(default.hidden_states, _tome_independent(x, default, weighted=False))
    torch.testing.assert_close(replay_tome_image(x, default), default.hidden_states, rtol=0, atol=0)
    custom = tome_merge_image(x, keys, r=2, token_sizes=sizes,
        rules={"grouper": last_sources_first_destination, "aggregator": destination_weighted})
    reference = tome_merge_reference(keys, custom.layout_delta, token_sizes=sizes)
    expected = torch.stack([(keys[list(group)] * sizes[list(group), None, None]).sum(0) / sizes[list(group)].sum()
                            for group in custom.sequence_source_groups])
    torch.testing.assert_close(reference, expected, rtol=0, atol=0)
    assert default.metadata["merge_value"] == "token_size_weighted_mean"
    assert "rules_changed" not in default.metadata


@pytest.mark.parametrize("method", ["aim", "tome"])
def test_identity_does_not_invoke_any_rule(method):
    torch = pytest.importorskip("torch")
    x = torch.randn(5, 3, requires_grad=True)
    rules = {name: _fail for name in ("scorer", "grouper", "aggregator")}
    if method == "aim":
        result = aim_image_tokens(x, spec=AIMSpec(target_count=5), rules=rules)
        torch.testing.assert_close(result.tokens, x, rtol=0, atol=0)
    else:
        result = tome_merge_image(x, x, r=0, rules=rules)
        torch.testing.assert_close(result.hidden_states, x, rtol=0, atol=0)


@pytest.mark.parametrize("method", ["aim", "tome"])
@pytest.mark.parametrize("bad", ["budget", "partition", "repeat", "score", "detach"])
def test_invalid_rule_outputs_are_rejected_at_the_actual_kernel(method, bad):
    torch = pytest.importorskip("torch")
    x = torch.randn(6, 3, requires_grad=True)
    rules = {
        "budget": {"grouper": lambda r: []},
        "partition": {"grouper": lambda r: [(1, 2), (4, 3)]},
        "repeat": {"grouper": lambda r: [(0, 1), (0, 3)]},
        "score": {"scorer": lambda r: torch.zeros(1)},
        "detach": {"aggregator": lambda r: r["values"].mean(0).detach()},
    }[bad]
    with pytest.raises(ValueError):
        if method == "aim":
            aim_image_tokens(x, spec=AIMSpec(target_count=4), rules=rules)
        else:
            tome_merge_image(x, x, r=2, rules=rules)


def test_tome_protected_groups_cannot_be_custom_merge_endpoints():
    torch = pytest.importorskip("torch")
    x = torch.randn(6, 3)
    with pytest.raises(ValueError, match="protected"):
        tome_merge_image(x, x, r=1, protected_indices=(0,), rules={"grouper": lambda r: [(0, 1)]})
