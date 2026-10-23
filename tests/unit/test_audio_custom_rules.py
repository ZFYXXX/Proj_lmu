"""Audio custom rules change native arithmetic while retaining saved layouts."""
from types import SimpleNamespace
import pytest

from mllm_comp.core.errors import InvariantError

from tests.fixtures.researcher_audio_rules import (
    NAMES, stage_for, specifications, increasing_scores, last_indices,
    latest_pairs, chronological_groups, first_weighted, scaled_hta, first_layers,
)


@pytest.mark.parametrize("algorithm", list(NAMES))
def test_formal_configuration_loads_exact_applicable_rules_once(algorithm, monkeypatch):
    from mllm_comp.backbones.qwen25_omni.v2_compression import resolve_qwen25_omni_stage_execution
    import mllm_comp.compressors.rules as loading
    calls = []
    original = loading.load_rule_specs

    def load(specs):
        calls.append(specs)
        return original(specs)

    monkeypatch.setattr(loading, "load_rule_specs", load)
    stage = stage_for(algorithm)
    execution = resolve_qwen25_omni_stage_execution(stage)
    strategy = execution.create_strategy(execution.create_spec(stage.plugin_config()), stage=stage)
    assert set(strategy.rules) == set(NAMES[algorithm])
    assert strategy.rule_specs == specifications(algorithm)
    assert calls == [specifications(algorithm)]
    assert execution.binding.metadata["rules_changed"] is True


def test_selection_rules_change_indices_and_keep_segment_and_chunk_budgets():
    torch = pytest.importorskip("torch")
    from mllm_comp.compressors.projector_level.segmentwise_topk import SegmentwiseTopKProjectorStrategy
    from mllm_comp.compressors.projector_level.segmentwise_topk import SegmentwiseTopKSpec
    from mllm_comp.compressors.projector_level.uniform_sampling import uniform_sample
    from mllm_comp.compressors.encoder_level.attention_topk import select_ragged_attention_topk
    x = torch.arange(24.).reshape(8, 3).requires_grad_()
    rule = SegmentwiseTopKProjectorStrategy(SegmentwiseTopKSpec.from_config({"criterion": {"score_source": "uniform", "num_segments": 2}}))
    rule.rules = {"scorer": increasing_scores, "selector": last_indices}
    output, indices, _ = rule.compress(x, scores=None, keep_count=4, item_index=0)
    assert indices.tolist() == [2, 3, 6, 7]
    torch.testing.assert_close(output, x[[2, 3, 6, 7]])
    sampled, details = uniform_sample(x, keep_count=4, compression_factor=2, selector=last_indices)
    assert details["selected_indices"].tolist() == [4, 5, 6, 7]
    decisions = []
    pruned, _ = select_ragged_attention_topk(x, torch.arange(8.).flip(0), [4, 4], [2, 2],
        scorer=increasing_scores, selector=last_indices, replay_decisions=decisions)
    assert [i.tolist() for i in decisions] == [[2, 3], [2, 3]]
    torch.testing.assert_close(pruned, output)
    (output.sum() + sampled.sum() + pruned.sum()).backward()
    assert x.grad.count_nonzero()


@pytest.mark.parametrize("algorithm", ["uniform_average_pooling_audio", "unsupervised_segmentation_audio", "similarity_attention_merge_prune_audio"])
def test_saved_merge_groups_replay_custom_aggregation_and_reference_gradients(algorithm):
    torch = pytest.importorskip("torch")
    from mllm_comp.backbones.qwen25_omni.v2_compression import resolve_qwen25_omni_stage_execution
    stage = stage_for(algorithm)
    execution = resolve_qwen25_omni_stage_execution(stage)
    rule = execution.create_strategy(execution.create_spec(stage.plugin_config()), stage=stage)
    x = torch.arange(24.).reshape(8, 3).requires_grad_()
    scores = {key: torch.ones(8) for key in ("merge", "prune")} if "similarity" in algorithm else None
    output, indices, details = rule.compress(x, scores=scores, keep_count=4, item_index=0)
    plan = rule.replay_plan(4, details, indices, scores=scores)
    def fail(_):
        raise AssertionError("replay may not reselect or regroup")
    rule.rules.update({name: fail for name in rule.rules if name != "aggregator"})
    y = x.detach().clone().requires_grad_()
    replay = rule.replay(y, plan)
    reference = rule.map_reference(y, indices=indices, details=details, scores=scores, keep_count=4)
    torch.testing.assert_close(replay, output, rtol=0, atol=0)
    torch.testing.assert_close(reference, output, rtol=0, atol=0)
    output.square().sum().backward()
    replay.square().sum().backward()
    torch.testing.assert_close(x.grad, y.grad, rtol=0, atol=0)
    groups = details.get("groups") or [tuple(range(a, b)) for a, b in details.get("pool_windows", details.get("segments"))]
    expected = torch.stack([first_weighted({"values": x[list(group)]}) for group in groups])
    if "selected_group_indices" in details:
        expected = expected[details["selected_group_indices"]]
    torch.testing.assert_close(output, expected, rtol=0, atol=0)


def test_adjacent_and_fastadasp_freeze_pairs_weights_and_replay_gradients():
    torch = pytest.importorskip("torch")
    from mllm_comp.compressors.encoder_level.adjacent_token_merging import merge_adjacent_hidden_states, replay_adjacent_pairs
    from mllm_comp.compressors.llm_level.fastadasp import weighted_adjacent_token_merge, _apply_weighted_merge_plan
    x = torch.arange(24.).reshape(8, 3).requires_grad_()
    merged, starts, _ = merge_adjacent_hidden_states(x, torch.ones_like(x), merge_ratio=0.25,
        scorer=increasing_scores, grouper=latest_pairs, aggregator=first_weighted)
    assert starts.tolist() == [4, 6]
    torch.testing.assert_close(merged, torch.cat((x[:4], torch.stack([first_weighted({"values": x[i:i+2]}) for i in [4, 6]]))))
    replay = replay_adjacent_pairs(x, starts, aggregator=first_weighted)
    torch.testing.assert_close(merged, replay, rtol=0, atol=0)
    merged2, reps, details = weighted_adjacent_token_merge(x, torch.zeros(7), torch.ones(8), output_length=4,
        scorer=increasing_scores, grouper=chronological_groups, aggregator=first_weighted)
    assert reps.tolist() == [0, 5, 6, 7]
    plan = details["_checkpoint_replay_plan"]
    torch.testing.assert_close(plan.normalized_weights[:5], torch.arange(1., 6.) / 15)
    torch.testing.assert_close(merged2, _apply_weighted_merge_plan(x, plan, aggregator=first_weighted), rtol=0, atol=0)
    (merged.sum() + merged2.sum()).backward()
    assert (x.grad > 0).all()


def test_rule_contracts_reject_budget_partition_and_detached_value_errors():
    torch = pytest.importorskip("torch")
    from mllm_comp.compressors.audio_rule_utils import token_indices, contiguous_groups, aggregate_groups, token_scores
    x = torch.ones(6, 2, requires_grad=True)
    with pytest.raises(InvariantError):
        token_indices([0, 0], length=6, count=2, device=x.device)
    with pytest.raises(InvariantError):
        contiguous_groups([(0, 2), (1, 3, 4, 5)], length=6)
    with pytest.raises(InvariantError):
        aggregate_groups(x, [(0, 1)], aggregator=lambda r: r["values"].mean(0).detach())
    with pytest.raises(InvariantError):
        token_scores(torch.tensor([float("nan")]), x, length=1)


def test_hta_transform_uses_owned_real_module_and_retains_its_gradients():
    torch = pytest.importorskip("torch")
    from mllm_comp.compressors.projector_level.fastslm_hta import build_fastslm_hta, fastslm_hierarchical_temporal_abstract
    module = build_fastslm_hta(hidden_size=8, num_attention_heads=2, stage_query_tokens=2,
        output_query_tokens=4, window_tokens=8, detail_recovery_depth=1)
    x = torch.randn(8, 8, requires_grad=True)
    baseline, _ = fastslm_hierarchical_temporal_abstract(x, keep_count=4, abstractor=module)
    custom, details = fastslm_hierarchical_temporal_abstract(x, keep_count=4, abstractor=module, transform=scaled_hta)
    torch.testing.assert_close(custom, baseline * 1.5)
    custom.square().sum().backward()
    assert x.grad.count_nonzero()
    assert any(p.grad is not None and p.grad.count_nonzero() for p in module.parameters())
    assert details["implementation"] == "custom_transform"


def test_spade_decision_changes_actual_operation_indices_without_a_new_applier():
    from mllm_comp.compressors.llm_level.structured_layer_pruning import StructuredLLMLayerPruningStrategy, StructuredLLMLayerPruningSpec
    stage = stage_for("structured_llm_layer_pruning")
    rule = StructuredLLMLayerPruningStrategy(StructuredLLMLayerPruningSpec.from_config(stage.plugin_config()))
    rule.rules = {"decision": first_layers}
    actions = rule.plan_operations(model=object(), structural_handles={"llm": SimpleNamespace(layers=[object()] * 4)})
    assert [a.target_indices for a in actions] == [[0], [1]]
    assert rule.runtime_metadata()["selection_source"] == "custom_decision"
