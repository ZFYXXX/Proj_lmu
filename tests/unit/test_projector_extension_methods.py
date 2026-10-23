"""Contracts for the two additional Projector-Level audio methods."""

from __future__ import annotations

import pytest

from mllm_comp.compressors.multimodal_registry import (
    builtin_multimodal_compression_contracts,
)
from mllm_comp.compressors.base import CompressionContext
from mllm_comp.compressors.projector_level import (
    LocalTemporalBipartiteMergingAudio,
    LocalTemporalBipartiteMergingSpec,
    SimilarityAttentionMergePruneAudio,
    SimilarityAttentionMergePruneSpec,
    local_temporal_bipartite_merge,
    similarity_attention_merge_prune,
)
from mllm_comp.core.state import ForwardState, TokenSpan
from mllm_comp.core.vocabulary import ExecutionPhase, Placement


def _state() -> ForwardState:
    return ForwardState(
        modality_token_spans=[
            TokenSpan(sample_index=0, start=2, end=18, modality="audio"),
            TokenSpan(sample_index=1, start=3, end=11, modality="audio"),
        ]
    )


def test_ltbm_spec_and_plan_fix_exact_per_audio_budgets() -> None:
    config = {
        "_stage_name": "ltbm",
        "_execution_phase": "forward",
        "operation": {
            "type": "merge",
            "target_site": "audio_encoder_output_pre_projector",
            "merge_value": "mean",
        },
        "criterion": {
            "score_source": "projector_token_cosine",
            "pairing": "alternating_bipartite_local",
            "window_size": 4,
        },
        "schedule": {"keep_ratio": 0.5},
    }
    spec = LocalTemporalBipartiteMergingSpec.from_config(config)
    plan = LocalTemporalBipartiteMergingAudio().plan(
        state=_state(),
        context=CompressionContext(phase="train", model=None),
        config=config,
    )

    assert spec.window_size == 4
    assert spec.merge_value == "mean"
    assert plan.placement is Placement.PROJECTOR_LEVEL
    assert plan.execution_phase is ExecutionPhase.FORWARD
    assert [action.metadata["keep_count"] for action in plan.token_actions] == [8, 4]
    assert plan.object_subtype == "audio_encoder_output_pre_projector"


def test_sa_map_is_one_post_projector_composite_plan() -> None:
    config = {
        "_stage_name": "sa_map",
        "operation": {
            "type": "merge_prune",
            "target_site": "audio_projector_output_post_projector",
            "merge_value": "attention_weighted_mean",
        },
        "criterion": {
            "score_source": "encoder_penultimate_layer_attention",
            "similarity_threshold": 0.75,
            "pruning": "greedy_attention_dpp",
        },
        "schedule": {"keep_ratio": 0.25},
    }
    spec = SimilarityAttentionMergePruneSpec.from_config(config)
    plan = SimilarityAttentionMergePruneAudio().plan(
        state=_state(),
        context=CompressionContext(phase="eval", model=None),
        config=config,
    )

    assert spec.similarity_threshold == 0.75
    assert len(plan.token_actions) == 2
    assert [action.metadata["keep_count"] for action in plan.token_actions] == [4, 2]
    assert plan.object_subtype == "audio_projector_output_post_projector"
    binding = builtin_multimodal_compression_contracts().binding(
        "similarity_attention_merge_prune_audio.audio.qwen25_omni"
    )
    assert binding.targets[0].target_site == plan.object_subtype


def test_ltbm_tensor_path_is_exact_ordered_and_differentiable() -> None:
    torch = pytest.importorskip("torch")
    hidden = torch.randn(12, 8, requires_grad=True)

    output, report = local_temporal_bipartite_merge(
        hidden,
        keep_count=5,
        window_size=3,
    )
    output.sum().backward()

    assert output.shape == (5, 8)
    assert report["merge_count"] == 7
    assert report["window_coordinate"] == "source_destination_partition_order"
    assert report["merge_value"] == "round_equal_mean"
    assert report["token_order"] == "chronological"
    assert hidden.grad is not None
    assert bool(torch.isfinite(hidden.grad).all())


def test_ltbm_uses_equal_mean_again_after_repartitioning() -> None:
    torch = pytest.importorskip("torch")
    hidden = torch.tensor([[2.0], [10.0], [4.0], [6.0], [-2.0], [-8.0]])

    output, report = local_temporal_bipartite_merge(
        hidden,
        keep_count=2,
        window_size=8,
    )

    # Round one produces mean(10, 2, 4) and mean(-8, -2). Round two treats
    # those current tokens equally, rather than recovering original masses.
    assert torch.allclose(output[:, 0], torch.tensor([17.0 / 3.0, -5.0]))
    assert [item["merge_count"] for item in report["merge_rounds"]] == [3, 1]


def test_sa_map_tensor_path_merges_then_prunes_with_gradient() -> None:
    torch = pytest.importorskip("torch")
    hidden = torch.randn(14, 8, requires_grad=True)
    merge_importance = torch.linspace(0.1, 1.4, 14)
    prune_importance = torch.linspace(1.4, 0.1, 14)

    output, report = similarity_attention_merge_prune(
        hidden,
        merge_importance,
        prune_importance,
        keep_count=5,
        similarity_threshold=0.95,
    )
    output.square().sum().backward()

    assert output.shape == (5, 8)
    assert report["post_merge_tokens"] >= 5
    assert report["token_order"] == "chronological"
    assert hidden.grad is not None
    assert bool(torch.isfinite(hidden.grad).all())


def test_sa_map_uses_distinct_merge_and_prune_importance() -> None:
    torch = pytest.importorskip("torch")
    hidden = torch.tensor([[1.0, 0.0], [0.0, 1.0]])

    output, report = similarity_attention_merge_prune(
        hidden,
        torch.tensor([10.0, 1.0]),
        torch.tensor([1.0, 4.0]),
        keep_count=1,
        similarity_threshold=0.99,
    )

    # No grouping happens. ADPruner must therefore select token 1 from the
    # prune importance, irrespective of token 0's larger merge importance.
    assert torch.equal(output, hidden[1:2])
    assert report["selected_group_indices"].tolist() == [1]


@pytest.mark.parametrize("identity", [False, True])
def test_sa_map_weighted_groups_checkpoint_replays_original_value_graph(identity):
    from mllm_comp.compressors.projector_level.similarity_attention_merge_prune import SimilarityAttentionMergePruneProjectorStrategy
    from mllm_comp.compressors.projector_level.similarity_attention_merge_prune import SimilarityAttentionMergePruneSpec

    torch = pytest.importorskip("torch")
    from torch.utils.checkpoint import checkpoint

    values = torch.tensor([[1., 0.], [0.99, 0.01], [-1., 0.], [-0.99, 0.01],
                           [0., 1.], [0.01, 0.99], [0., -1.], [0.01, -0.99]])
    scores = {"merge": torch.tensor([1., 3., 2., 1., 1., 2., 1., 4.]),
              "prune": torch.tensor([4., 1., 2., 1., 1., 2., 3., 1.])}
    keep_count = 8 if identity else 2
    rule = SimilarityAttentionMergePruneProjectorStrategy(SimilarityAttentionMergePruneSpec(similarity_threshold=0.95))
    results = []
    for mode in (None, False, True):
        plan = [None]
        calls = []
        def forward(hidden):
            if plan[0] is not None:
                return rule.replay(hidden, plan[0])
            output, indices, details = rule.compress(hidden, scores=scores, keep_count=keep_count, item_index=0)
            plan[0] = rule.replay_plan(keep_count, details, indices, scores=scores)
            calls.append(details)
            return output
        inputs = values.clone().requires_grad_()
        output = forward(inputs) if mode is None else checkpoint(forward, inputs, use_reentrant=mode)
        output.square().sum().backward()
        assert len(calls) == 1
        expected = []
        anchors = []
        for index in calls[0]["selected_group_indices"].tolist():
            group = calls[0]["groups"][index]
            weights = scores["merge"][group] / scores["merge"][group].sum()
            expected.append((values[group] * weights.unsqueeze(-1)).sum(0))
            anchors.append(group[0])
        torch.testing.assert_close(output.detach(), torch.stack(expected), rtol=0, atol=0)
        assert rule.representative_indices(8, plan[0], inputs.device).tolist() == anchors
        results.append((output.detach(), inputs.grad))
    for actual in results[1:]:
        for got, expected in zip(actual, results[0]):
            torch.testing.assert_close(got, expected, rtol=0, atol=0)


def test_ltbm_saved_rounds_preserve_value_graph_and_destination_positions():
    from mllm_comp.compressors.projector_level.local_temporal_bipartite_merging import LocalTemporalBipartiteProjectorStrategy
    from mllm_comp.compressors.projector_level.local_temporal_bipartite_merging import LocalTemporalBipartiteMergingSpec

    torch = pytest.importorskip("torch")
    rule = LocalTemporalBipartiteProjectorStrategy(LocalTemporalBipartiteMergingSpec.from_config({"schedule": {"keep_ratio": 0.25}}))
    original = torch.tensor([[2.0], [10.0], [4.0], [6.0], [-2.0], [-8.0]], requires_grad=True)
    result, _, details = rule.compress(original, scores=None, keep_count=1, item_index=0)
    plan = rule.replay_plan(1, details)
    assert len(plan) >= 2
    result.square().sum().backward()
    # Independent round-by-round reference keeps each destination's lineage.
    expected = original.detach()
    positions = list(range(6))
    for pairs in plan:
        sources = {source for source, _ in pairs}
        groups = {i: [i] + [source for source, destination in pairs if destination == i]
                  for i in range(expected.shape[0]) if i not in sources}
        expected = torch.stack([expected[group].mean(0) for group in groups.values()])
        positions = [positions[i] for i in groups]
    torch.testing.assert_close(result.detach(), expected, rtol=0, atol=0)
    assert rule.representative_indices(6, plan, original.device).tolist() == positions
    replay_input = original.detach().clone().requires_grad_()
    replayed = rule.replay(replay_input, plan)
    replayed.square().sum().backward()
    torch.testing.assert_close(replayed, result, rtol=0, atol=0)
    torch.testing.assert_close(replay_input.grad, original.grad, rtol=0, atol=0)
    assert rule.merge_position_policy == "preserve_absolute_destination"



@pytest.mark.parametrize("algorithm", ["uniform_sampling_audio", "uniform_average_pooling_audio", "segmentwise_topk_audio",
    "local_temporal_bipartite_merging_audio", "unsupervised_segmentation_audio", "similarity_attention_merge_prune_audio"])
def test_projector_reference_mapping_follows_actual_value_transform(algorithm):
    from mllm_comp.compressors.projector_level.unsupervised_segmentation import unsupervised_segmentation_boundaries

    torch = pytest.importorskip("torch")
    registry = builtin_multimodal_compression_contracts()
    descriptor = registry.algorithm(algorithm)
    binding = registry.binding(f"{algorithm}.audio.qwen25_omni")
    schedule = ({"compression_factor": 2} if algorithm.startswith("uniform_") else
                {} if algorithm == "unsupervised_segmentation_audio" else {"keep_ratio": 0.25})
    config = {"schedule": schedule, "criterion": {"similarity_threshold": 0.95}}
    spec = descriptor.metadata["spec_factory"].from_config(config)
    rule = binding.strategy_factory(spec)
    values = torch.tensor([[1., 0.], [0.99, 0.01], [-1., 0.], [-0.99, 0.01],
                           [0., 1.], [0.01, 0.99], [0., -1.], [0.01, -0.99]])
    scores = torch.tensor([1., 3., 2., 1., 1., 2., 1., 4.])
    if algorithm == "similarity_attention_merge_prune_audio":
        scores = {"merge": scores, "prune": scores.flip(0)}
    count = 4 if algorithm.startswith("uniform_") else 2
    if algorithm == "unsupervised_segmentation_audio":
        count = len(unsupervised_segmentation_boundaries(values)) + 1
    with torch.no_grad():
        output, indices, details = rule.compress(values, scores=scores, keep_count=count, item_index=0)
        mapped = rule.map_reference(values, indices=indices, details=details, scores=scores, keep_count=count)
    torch.testing.assert_close(mapped, output, rtol=0, atol=0)
