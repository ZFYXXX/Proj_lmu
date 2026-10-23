"""Injected LTBM rules retain the native budget, layout and replay path."""
import pytest

from mllm_comp.compressors.projector_level.local_temporal_bipartite_merging import LocalTemporalBipartiteProjectorStrategy
from mllm_comp.compressors.projector_level.local_temporal_bipartite_merging import (
    LocalTemporalBipartiteMergingSpec, local_temporal_bipartite_merge,
)
from mllm_comp.core.errors import InvariantError


def reverse_sources(request):
    """Choose the latest sources first, with their nearest legal destination."""
    sources = request["source_indices"]
    destinations = request["destination_indices"]
    return [(source, min(destinations, key=lambda d: abs(source // 2 - d // 2)))
            for source in reversed(sources[-request["merge_count"]:])]


def destination_weighted(request, *, destination_weight=3.0):
    values = request["values"]
    return (destination_weight * values[0] + values[1:].sum(0)) / (destination_weight + len(values) - 1)


def _specs():
    module = "tests.unit.test_ltbm_custom_rules"
    return {"grouper": {"callable": module + ":reverse_sources", "params": {}},
            "aggregator": {"callable": module + ":destination_weighted", "params": {"destination_weight": 3.0}}}


def _independent_rounds(values, plan):
    positions = list(range(len(values)))
    for pairs in plan:
        sources = {source for source, _ in pairs}
        outputs, kept = [], []
        for destination in range(len(values)):
            if destination in sources:
                continue
            assigned = [source for source, target in pairs if target == destination]
            output = values[destination]
            if assigned:
                output = (3.0 * output + sum(values[source] for source in assigned)) / (3.0 + len(assigned))
            outputs.append(output)
            kept.append(positions[destination])
        import torch
        values = torch.stack(outputs)
        positions = kept
    return values, positions


def test_custom_groups_aggregation_and_saved_replay_preserve_values_positions_and_gradients():
    torch = pytest.importorskip("torch")
    original = torch.tensor([[2., 1.], [10., 2.], [4., 3.], [6., 4.], [-2., 5.], [-8., 6.]], requires_grad=True)
    rule = LocalTemporalBipartiteProjectorStrategy(LocalTemporalBipartiteMergingSpec(),
        rules={"grouper": reverse_sources, "aggregator": destination_weighted}, rule_specs=_specs())
    output, _, details = rule.compress(original, scores=None, keep_count=2, item_index=0)
    plan = rule.replay_plan(2, details)
    expected_input = original.detach().clone().requires_grad_()
    expected, representatives = _independent_rounds(expected_input, plan)
    torch.testing.assert_close(output, expected, rtol=0, atol=0)
    output.square().sum().backward()
    expected.square().sum().backward()
    torch.testing.assert_close(original.grad, expected_input.grad, rtol=0, atol=0)
    replay_input = original.detach().clone().requires_grad_()
    replayed = rule.replay(replay_input, plan)
    replayed.square().sum().backward()
    torch.testing.assert_close(replayed, output, rtol=0, atol=0)
    torch.testing.assert_close(replay_input.grad, original.grad, rtol=0, atol=0)
    assert rule.representative_indices(len(original), plan, original.device).tolist() == representatives
    torch.testing.assert_close(rule.map_reference(original, indices=None, details=details, scores=None, keep_count=2), output)
    default, _ = local_temporal_bipartite_merge(original.detach(), keep_count=2, window_size=8)
    assert not torch.allclose(default, output)
    assert rule.runtime_metadata()["rules"] == _specs()
    assert rule.runtime_metadata()["rules_changed"] is True
    assert rule.runtime_metadata()["merge_value"] == "custom"
    assert rule.runtime_metadata()["score_source"] == "custom_grouping_rule"


def test_grouper_changes_actual_selected_pairs_and_default_is_exact():
    torch = pytest.importorskip("torch")
    values = torch.ones(6, 2)
    default, details = local_temporal_bipartite_merge(values, keep_count=5, window_size=8)
    explicit, explicit_details = local_temporal_bipartite_merge(values, keep_count=5, window_size=8, grouper=None, aggregator=None)
    torch.testing.assert_close(default, explicit, rtol=0, atol=0)
    assert details == explicit_details
    assert details["merge_pairs"] == [(0, 1)]
    _, custom = local_temporal_bipartite_merge(values, keep_count=5, window_size=8, grouper=reverse_sources)
    assert custom["merge_pairs"] == [(4, 5)]
    assert "grouping_rule" not in details


def test_identity_does_not_call_injected_rules():
    torch = pytest.importorskip("torch")
    def fail(request):
        raise AssertionError("identity must bypass rules")
    values = torch.randn(4, 3, requires_grad=True)
    output, _ = local_temporal_bipartite_merge(values, keep_count=4, window_size=1, grouper=fail, aggregator=fail)
    assert output is values


@pytest.mark.parametrize("pairs,error", [
    ([(0, 1)], "exact"),
    ([(0, 1), (0, 3)], "unique"),
    ([(1, 3), (2, 3)], "even"),
    ([(0, 2), (2, 3)], "even"),
    ([(6, 1), (2, 3)], "locality"),
    ([(True, 1), (2, 3)], "integer"),
])
def test_invalid_custom_pairs_are_rejected_before_aggregation(pairs, error):
    torch = pytest.importorskip("torch")
    def fail(request):
        raise AssertionError("invalid groups must fail before aggregation")
    with pytest.raises(InvariantError, match=error):
        local_temporal_bipartite_merge(torch.randn(8, 3), keep_count=6, window_size=1,
            grouper=lambda request: pairs, aggregator=fail)


@pytest.mark.parametrize("kind,error", [("shape", "vector"), ("dtype", "vector"),
    ("nonfinite", "non-finite"), ("detached", "detached")])
def test_invalid_aggregator_output_is_rejected(kind, error):
    torch = pytest.importorskip("torch")
    def aggregate(request):
        values = request["values"]
        return {"shape": lambda: values, "dtype": lambda: values.mean(0).double(),
                "nonfinite": lambda: values.mean(0) * float("nan"),
                "detached": lambda: values.mean(0).detach()}[kind]()
    with pytest.raises(InvariantError, match=error):
        local_temporal_bipartite_merge(torch.randn(6, 3, requires_grad=True), keep_count=4,
            window_size=1, aggregator=aggregate)


def test_direct_controller_rejects_rules_for_an_unsupported_audio_method():
    from mllm_comp.backbones.qwen25_omni.v2_compression import resolve_qwen25_omni_stage_execution
    from mllm_comp.core.errors import BackendIntegrationError, ConfigError
    from mllm_comp.compressors.rules import builtin_rule_specs
    from tests.fixtures.qwen25_omni_audio import projector_stage
    stage = projector_stage("uniform_sampling_audio")
    stage.operation["aggregator"] = _specs()["aggregator"]
    with pytest.raises(ConfigError, match="does not support"):
        builtin_rule_specs(stage)
    with pytest.raises(BackendIntegrationError, match="does not support") as error:
        resolve_qwen25_omni_stage_execution(stage)
    assert isinstance(error.value.__cause__, ConfigError)


def test_aggregator_only_changes_values_without_changing_single_round_groups():
    torch = pytest.importorskip("torch")
    values = torch.tensor([[2.], [10.], [4.], [6.], [-2.], [-8.]])
    default, original = local_temporal_bipartite_merge(values, keep_count=4, window_size=8)
    custom, changed = local_temporal_bipartite_merge(values, keep_count=4, window_size=8,
        aggregator=destination_weighted)
    assert original["merge_rounds"] == changed["merge_rounds"]
    assert not torch.allclose(default, custom)
    expected, _ = _independent_rounds(values, [changed["merge_pairs"]])
    torch.testing.assert_close(custom, expected, rtol=0, atol=0)


def test_direct_strategy_load_wraps_missing_rule_module_as_backend_error():
    from mllm_comp.backbones.qwen25_omni.v2_compression import resolve_qwen25_omni_stage_execution
    from mllm_comp.core.errors import BackendIntegrationError, ConfigError
    from mllm_comp.compressors.rules import load_rules
    from tests.fixtures.qwen25_omni_audio import projector_stage
    stage = projector_stage("local_temporal_bipartite_merging_audio")
    stage.operation["grouper"] = {"callable": "mllm_comp_missing_ltbm_rule:group"}
    execution = resolve_qwen25_omni_stage_execution(stage)
    spec = execution.create_spec(stage.plugin_config())
    with pytest.raises(ConfigError, match="Cannot load"):
        load_rules(stage, supported_rules=execution.variant.supported_rules)
    with pytest.raises(BackendIntegrationError, match="Cannot load") as error:
        execution.create_strategy(spec, stage=stage)
    assert isinstance(error.value.__cause__, ConfigError)
