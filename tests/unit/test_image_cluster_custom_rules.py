"""Fixed membership is separate from scores and gradient-carrying values."""

import pytest


def test_pact_custom_prune_cluster_mass_and_replay():
    torch = pytest.importorskip("torch")
    from mllm_comp.compressors.image.pact import PACTSpec, pact_image_tokens
    from mllm_comp.compressors.image._rules import replay_rule_groups
    from mllm_comp.compressors.image.decoder_lifecycle import layout_merge_groups
    values = torch.arange(24, dtype=torch.float32).reshape(8, 3).requires_grad_()
    images = torch.arange(1, 7)
    calls = []

    def group(request):
        calls.append("group")
        count = request["center_count"]
        centers = torch.arange(count)
        labels = torch.zeros_like(request["default_labels"])
        labels[:count] = centers
        return {"centers": centers, "labels": labels}

    def aggregate(request):
        return request["values"].square().mean(0)

    rules = {"scorer": lambda request: torch.arange(6, dtype=torch.float32),
        "selector": lambda request: torch.tensor([2, 3, 4, 5]), "grouper": group, "aggregator": aggregate}
    result = pact_image_tokens(values, image_indices=images, clustering_keys=values.detach(),
        image_attention_scores=torch.ones(6), rules=rules,
        spec=PACTSpec(prune_keep_ratio=.75, target_keep_ratio=.3, cutoff=2.0))
    groups = layout_merge_groups(result.layout_delta)
    expected = torch.stack([values[list(g)].square().mean(0) if len(g)>1 else values[g[0]] for g in groups])
    torch.testing.assert_close(result.hidden_states, expected)
    torch.testing.assert_close(result.token_sizes, torch.tensor([len(g) for g in groups], dtype=torch.float32))
    replay = replay_rule_groups(values + 2, groups, rules=rules)
    torch.testing.assert_close(replay, torch.stack([(values[list(g)] + 2).square().mean(0) if len(g)>1 else values[g[0]]+2 for g in groups]))
    assert calls == ["group"]
    replay.sum().backward()
    assert values.grad is not None and torch.isfinite(values.grad).all()
    with pytest.raises(ValueError, match="assign each center"):
        pact_image_tokens(values, image_indices=images, clustering_keys=values.detach(), image_attention_scores=torch.ones(6),
            rules={**rules, "grouper": lambda request: {"centers": request["default_centers"], "labels": torch.full_like(request["default_labels"], -1)}},
            spec=PACTSpec(prune_keep_ratio=.75, cutoff=2.0))


def test_sparse_custom_selection_recycle_membership_and_replay():
    torch = pytest.importorskip("torch")
    from mllm_comp.compressors.image.sparsevlm import SparseVLMSpec, sparsevlm_image_tokens
    from mllm_comp.compressors.image._rules import replay_rule_groups
    from mllm_comp.compressors.image.decoder_lifecycle import layout_merge_groups
    values = torch.arange(30, dtype=torch.float32).reshape(10, 3).requires_grad_()
    images = torch.arange(1, 9)
    calls = []

    def groups(request):
        calls.append("group")
        labels = torch.arange(6) % 2
        return labels

    rules = {"scorer": lambda request: torch.arange(8, dtype=torch.float32),
        "selector": lambda request: torch.tensor([6, 7]),
        "recycle_selector": lambda request: torch.arange(6),
        "cluster_scorer": lambda request: torch.arange(6, dtype=torch.float32),
        "cluster_selector": lambda request: torch.tensor([0, 1]),
        "grouper": groups, "aggregator": lambda request: request["values"].square().mean(0)}
    result = sparsevlm_image_tokens(values, image_indices=images, image_attention_scores=torch.ones(8),
        layer_index=0, initial_visual_tokens=8, rules=rules,
        spec=SparseVLMSpec(prune_layers=(0,), schedule_keep_tokens=(2,), scale_schedule_to_input=False, recycle_ratio=1., cluster_divisor=4))
    expected = torch.stack([values[0], values[7], values[8], values[[1,3,5]].square().mean(0), values[[2,4,6]].square().mean(0), values[9]])
    torch.testing.assert_close(result.hidden_states, expected)
    replay = replay_rule_groups(values, layout_merge_groups(result.layout_delta), rules=rules)
    torch.testing.assert_close(replay, expected)
    assert calls == ["group"]
    replay.sum().backward()
    assert values.grad is not None and values.grad.abs().sum() > 0
    with pytest.raises(ValueError, match="own groups"):
        sparsevlm_image_tokens(values, image_indices=images, image_attention_scores=torch.ones(8),
            layer_index=0, initial_visual_tokens=8, rules={**rules, "grouper": lambda request: torch.ones(6, dtype=torch.long)},
            spec=SparseVLMSpec(prune_layers=(0,), schedule_keep_tokens=(2,), scale_schedule_to_input=False, recycle_ratio=1., cluster_divisor=4))


def test_pact_grouper_passthrough_kept_rejection_with_pruned_reassignment():
    torch = pytest.importorskip("torch")
    from mllm_comp.compressors.image.pact import PACTSpec, pact_image_tokens

    values = torch.arange(9, dtype=torch.float32).reshape(3, 3)
    kwargs = dict(image_indices=torch.arange(3), clustering_keys=torch.eye(3),
        image_attention_scores=torch.tensor([3., 2., 1.]),
        spec=PACTSpec(prune_keep_ratio=.75, cutoff=2.,
            pruned_cutoff_coefficient=.1, include_pruned=True))
    default = pact_image_tokens(values, **kwargs)
    assert bool((default.cluster_assignment.labels[:2] < 0).any())
    observed = []

    def passthrough(request):
        observed.append(True)
        return {"centers": request["default_centers"], "labels": request["default_labels"]}

    custom = pact_image_tokens(values, **kwargs, rules={"grouper": passthrough})
    assert observed == [True]
    for name in ("hidden_states", "kept_indices", "result_image_indices", "token_sizes"):
        torch.testing.assert_close(getattr(custom, name), getattr(default, name), rtol=0, atol=0)
    torch.testing.assert_close(custom.cluster_assignment.centers, default.cluster_assignment.centers)
    torch.testing.assert_close(custom.cluster_assignment.labels, default.cluster_assignment.labels)
    assert custom.layout_delta == default.layout_delta
