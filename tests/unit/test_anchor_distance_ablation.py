import pytest
import torch

from mllm_comp.analysis.common.anchor_distance_ablation import GeometrySpec, cluster, compare, specifications
from mllm_comp.analysis.common.visual_merge_attribution import feature_anchor_partition


def test_cosine_matches_existing_and_normalized_l2():
    torch.manual_seed(5)
    x = torch.randn(160, 24)
    a = cluster(x, height=10, width=16, count=64)
    b = cluster(x, height=10, width=16, count=64, spec=GeometrySpec("unit", metric="unit_l2"))
    old = feature_anchor_partition(x, height=10, width=16, target_tokens=64)
    assert {tuple(g) for g in a["members"]} == set(old.members)
    assert compare(a, b)["partition_identical"]
    assert compare(a, b)["anchor_sequence_identical"]


@pytest.mark.parametrize("spec", [GeometrySpec("raw", metric="raw_l2"),
    GeometrySpec("pp", metric="unit_l2", initialization="kmeanspp"),
    GeometrySpec("lloyd", metric="unit_l2", initialization="kmeanspp", lloyd_steps=10)])
def test_duplicate_degenerate_data_has_exact_k_and_determinism(spec):
    x = torch.ones(160, 16)
    a = cluster(x, height=10, width=16, count=128, spec=spec)
    b = cluster(x, height=10, width=16, count=128, spec=spec)
    assert a["tokens"].shape == (128, 16)
    assert a["members"] == b["members"]
    assert sorted(i for g in a["members"] for i in g) == list(range(160))
    assert all(a["cluster_sizes"])
    assert torch.equal(a["tokens"], torch.ones(128, 16))
    assert len(set(a["indices"])) == 128


def test_raw_norm_changes_selection():
    x = torch.tensor([[1., 0.], [100., 0.], [0., 1.], [-1., 0.]])
    a = cluster(x, height=2, width=2, count=2)
    b = cluster(x, height=2, width=2, count=2, spec=GeometrySpec("raw", metric="raw_l2"))
    assert a["anchors"] == [0, 3]
    assert b["anchors"] == [0, 1]


def test_lloyd_reduces_own_objective_but_outputs_original_feature_means():
    torch.manual_seed(42)
    x = torch.randn(200, 12)*torch.linspace(1., 5., 200)[:, None]
    spec = GeometrySpec("lloyd", metric="unit_l2", initialization="kmeanspp", lloyd_steps=10)
    output = cluster(x, height=10, width=20, count=16, spec=spec)
    history = output["geometry"]["lloyd_sse_history"]
    assert all(b <= a+1e-4 for a, b in zip(history, history[1:]))
    for values, members in zip(output["tokens"], output["members"]):
        assert torch.equal(values, x[members].mean(0))


def test_small_or_nonfinite_fail():
    with pytest.raises(ValueError):
        cluster(torch.ones(16, 8), height=4, width=4, count=128)
    with pytest.raises(ValueError):
        cluster(torch.full((160, 8), float("nan")), height=10, width=16)


@pytest.mark.parametrize("spec", specifications())
def test_all_conditions_compact_labels_text_and_centroid_positions(spec):
    from mllm_comp.analysis.common.compression_training_check import compact_anchor
    torch.manual_seed(42)
    n, start, length = 160, 3, 169
    ids = torch.full((1, length), 3)
    ids[:, start:start+n] = 2
    labels = torch.full_like(ids, -100)
    labels[0, -1] = 4
    positions = torch.arange(length)[None, None].expand(3, 1, length).clone()
    positions[1, 0, start:start+n] = torch.arange(n)//16+start
    positions[2, 0, start:start+n] = torch.arange(n)%16+start
    state = {"features": torch.randn(n, 24).bfloat16(), "height": 10, "width": 16,
        "hidden": torch.randn(1, length, 24).bfloat16(), "positions": positions,
        "visual": list(range(start, start+n)), "prefix": length-1,
        "teacher_inputs": {"input_ids": ids, "labels": labels, "attention_mask": torch.ones_like(ids)}}
    output = cluster(state["features"], height=10, width=16, spec=spec)
    compact = compact_anchor(state, output)
    assert compact["hidden"].shape == (1, 137, 24)
    assert compact["prefix"] == 136
    assert compact["labels"][0, -1] == 4
    assert torch.equal(compact["positions"][:, :, -6:], positions[:, :, -6:])
    for i, members in enumerate(output["members"]):
        expected = positions[:, 0, [start+j for j in members]].float().mean(1).round().long()
        assert torch.equal(compact["positions"][:, 0, start+i], expected)


def test_500_image_selection_per_task_is_deterministic_answer_blind_and_unique():
    import hashlib
    from collections import Counter
    from scripts.analysis.qwen25_vl.run_anchor_distance_ablation import select_cohort_entries
    entries = [{"task": task, "sample_id": f"{task}:{i}", "doc_id": i,
        "image_fingerprint": hashlib.sha256(f"{task}:{i}".encode()).hexdigest(),
        "split": "development" if i < 100 else "confirmation", "shard": i%2,
        "smoke": i < 8, "answer": "irrelevant", "score": 0.}
        for task in ["chartqa", "textvqa"] for i in range(500)]
    chosen = select_cohort_entries(entries)
    altered = [{**e, "answer": "changed", "score": 1.} for e in reversed(entries)]
    other = select_cohort_entries(altered)
    assert {e["sample_id"] for e in chosen} == {e["sample_id"] for e in other}
    assert len({e["image_fingerprint"] for e in chosen}) == 500
    assert Counter(e["task"] for e in chosen) == {"chartqa": 250, "textvqa": 250}
    assert Counter(e["split"] for e in chosen) == {"development": 200, "confirmation": 300}
    assert sum(e["smoke"] for e in chosen) == 16


def test_selection_rejects_cross_task_duplicate_images():
    from scripts.analysis.qwen25_vl.run_anchor_distance_ablation import select_cohort_entries
    entries = [{"task": task, "sample_id": f"{task}:{i}", "image_fingerprint": f"shared-{i}",
        "split": "development" if i < 100 else "confirmation"}
        for task in ["chartqa", "textvqa"] for i in range(500)]
    with pytest.raises(ValueError, match="globally unique"):
        select_cohort_entries(entries)
