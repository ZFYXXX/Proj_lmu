from __future__ import annotations

import pytest

from mllm_comp.analysis.common.visual_merge_attribution import (
    apply_mrope,
    attention_counterfactuals,
    balanced_grid_shape,
    feature_anchor_partition,
    grid_partition,
    partition_statistics,
)
from mllm_comp.analysis.probes.visual_merge_attribution import _all_finite


def test_rectangular_grid_partition_is_exact_and_nonempty() -> None:
    partition = grid_partition(12, 16, 64)

    assert balanced_grid_shape(12, 16, 64) == (8, 8)
    assert len(partition.members) == 64
    assert all(partition.members)
    assert sorted(index for cluster in partition.members for index in cluster) == list(
        range(192)
    )
    assert len(partition.partition_hash) == 64


def test_grid_partition_rejects_impossible_exact_factorization() -> None:
    with pytest.raises(ValueError, match="Cannot form"):
        grid_partition(5, 13, 64)


def test_feature_anchor_partition_is_deterministic_with_duplicate_features() -> None:
    torch = pytest.importorskip("torch")
    features = torch.zeros(80, 4)
    features[:, 0] = 1

    first = feature_anchor_partition(features, height=8, width=10, target_tokens=64)
    second = feature_anchor_partition(features, height=8, width=10, target_tokens=64)

    assert first.members == second.members
    assert first.partition_hash == second.partition_hash
    assert all(first.members)
    assert sorted(index for cluster in first.members for index in cluster) == list(range(80))


def test_partition_statistics_report_centroid_span_and_dispersion() -> None:
    torch = pytest.importorskip("torch")
    features = torch.arange(16, dtype=torch.float32).reshape(4, 4)
    partition = grid_partition(2, 2, 2)

    stats = partition_statistics(partition, features)

    assert stats["clusters"][0]["size"] == 2
    assert stats["clusters"][0]["centroid"] == pytest.approx([0.5, 0.0])
    assert stats["clusters"][0]["span"] == pytest.approx([1.0, 0.0])
    assert stats["clusters"][0]["channel_std_l2"] > 0


def test_mrope_is_identity_for_zero_angles() -> None:
    torch = pytest.importorskip("torch")
    states = torch.randn(2, 3, 8)
    cos = torch.ones(1, 3, 8)
    sin = torch.zeros_like(cos)

    actual = apply_mrope(states, cos, sin, [1, 1, 2])

    assert torch.equal(actual, states)


def test_singleton_clusters_make_all_partition_oracles_exact() -> None:
    torch = pytest.importorskip("torch")
    torch.manual_seed(3)
    heads, queries, dim = 2, 3, 4
    visual = 3
    text = 2
    q = torch.randn(heads, queries, dim)
    tk = torch.randn(heads, text, dim)
    tv = torch.randn(heads, text, dim)
    vk = torch.randn(heads, visual, dim)
    vv = torch.randn(heads, visual, dim)
    members = ((0,), (1,), (2,))

    report = attention_counterfactuals(
        queries=q,
        text_keys=tk,
        text_values=tv,
        visual_keys=vk,
        visual_values=vv,
        representative_keys=vk,
        representative_values=vv,
        members=members,
    )

    for variant in ("A", "B", "C", "D", "O-Z", "O-M"):
        assert report[variant]["attention_output_relative_error"]["mean"] < 1e-6
        assert report[variant]["cluster_log_partition_abs_error"]["mean"] < 1e-6
    assert report["diagnostics"]["higher_order_abs_residual"]["mean"] < 1e-6
    assert report["diagnostics"]["value_realization_relative_error"]["mean"] < 1e-6


def test_message_oracle_exactly_recovers_clustered_attention() -> None:
    torch = pytest.importorskip("torch")
    torch.manual_seed(7)
    heads, queries, dim = 2, 4, 6
    visual = 6
    q = torch.randn(heads, queries, dim)
    tk = torch.randn(heads, 3, dim)
    tv = torch.randn(heads, 3, dim)
    vk = torch.randn(heads, visual, dim)
    vv = torch.randn(heads, visual, dim)
    rk = torch.randn(heads, 2, dim)
    rv = torch.randn(heads, 2, dim)

    report = attention_counterfactuals(
        queries=q,
        text_keys=tk,
        text_values=tv,
        visual_keys=vk,
        visual_values=vv,
        representative_keys=rk,
        representative_values=rv,
        members=((0, 1, 2), (3, 4, 5)),
    )

    assert report["O-Z"]["visual_attention_mass_abs_error"]["mean"] < 1e-6
    assert report["O-Z"]["text_attention_l1_error"]["mean"] < 1e-6
    assert report["O-M"]["attention_output_relative_error"]["mean"] < 1e-6
    assert report["O-Z"]["attention_output_relative_error"]["mean"] > 0


def test_query_value_residual_reduces_first_order_value_realization_error() -> None:
    torch = pytest.importorskip("torch")
    q = torch.tensor([[[0.1, 0.0]]])
    text_keys = torch.tensor([[[0.0, 0.0]]])
    text_values = torch.tensor([[[0.0, 0.0]]])
    visual_keys = torch.tensor([[[-2.0, 0.0], [-1.0, 0.0], [1.0, 0.0], [2.0, 0.0]]])
    visual_values = torch.tensor([[[0.0, -2.0], [0.0, -1.0], [0.0, 1.0], [0.0, 2.0]]])
    report = attention_counterfactuals(
        queries=q,
        text_keys=text_keys,
        text_values=text_values,
        visual_keys=visual_keys,
        visual_values=visual_values,
        representative_keys=torch.zeros(1, 1, 2),
        representative_values=torch.zeros(1, 1, 2),
        members=((0, 1, 2, 3),),
        variants=("O-Z", "O-M", "O-Z-VR1"),
    )

    assert (
        report["O-Z-VR1"]["attention_output_relative_error"]["mean"]
        < report["O-Z"]["attention_output_relative_error"]["mean"]
    )
    assert report["O-M"]["attention_output_relative_error"]["mean"] < 1e-6


def test_query_value_residual_is_exact_for_singleton_clusters() -> None:
    torch = pytest.importorskip("torch")
    torch.manual_seed(13)
    q = torch.randn(2, 2, 4)
    tk = torch.randn(2, 1, 4)
    tv = torch.randn(2, 1, 4)
    vk = torch.randn(2, 2, 4)
    vv = torch.randn(2, 2, 4)
    report = attention_counterfactuals(
        queries=q,
        text_keys=tk,
        text_values=tv,
        visual_keys=vk,
        visual_values=vv,
        representative_keys=vk,
        representative_values=vv,
        members=((0,), (1,)),
        variants=("O-Z", "O-M", "O-Z-VR1"),
    )
    assert report["O-Z-VR1"]["attention_output_relative_error"]["mean"] < 1e-6


def test_report_finite_gate_rejects_nested_nan_and_infinity() -> None:
    assert _all_finite({"sample": {"metrics": [0.0, 1.0]}})
    assert not _all_finite({"sample": {"metrics": [0.0, float("nan")]}})
    assert not _all_finite({"sample": {"metrics": {"max": float("inf")}}})

