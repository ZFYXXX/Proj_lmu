import math
from types import SimpleNamespace

import pytest
import torch

from mllm_comp.analysis.common.token_metric_ablation import (
    NEW_METHODS, unit, regularized_diagonal, metric_space, pair_distances,
    farthest_partition, select_and_merge, response_embedding, extract_responses,
)
from mllm_comp.analysis.common.anchor_distance_ablation import cluster


def test_regularized_diagonal_and_mahalanobis_formula():
    diagonal = regularized_diagonal(torch.zeros(2), torch.tensor([1., 100.]))
    assert torch.allclose(diagonal, torch.tensor([5.95, 95.05]))
    x = torch.tensor([[1., 2.], [3., 1.]])
    values, metric = metric_space(x, "diag_mahalanobis", diagonal)
    expected = ((unit(x)[0]-unit(x)[1]).square()/diagonal).sum()
    assert pair_distances(values[:1], values[1:], metric)[0, 0].square() == pytest.approx(float(expected))


def test_correlation_is_centered_direction_not_original_cosine():
    x = torch.tensor([[1., 3., 7.], [8., 14., 26.]])
    values, metric = metric_space(x, "correlation")
    assert pair_distances(values[:1], values[1:], metric)[0, 0] == pytest.approx(-1., abs=1e-6)
    assert float(unit(x)[0] @ unit(x)[1]) < .999


def test_response_distance_matches_fixed_query_and_value_objective():
    torch.manual_seed(42)
    q, k, v = torch.randn(4, 3, 6), torch.randn(2, 8, 6), torch.randn(2, 8, 6)
    embedded, probabilities, scores, values = response_embedding(q, k, v)
    expected = (scores[0]-scores[1]).square().mean()+(values[0]-values[1]).square().mean()
    assert (embedded[0]-embedded[1]).square().sum() == pytest.approx(float(expected), rel=1e-6)
    assert torch.allclose(probabilities.sum(-1), torch.ones(8))
    js = pair_distances(probabilities, probabilities, "js")
    assert torch.isfinite(js).all()
    assert torch.allclose(js, js.T, atol=1e-5)
    assert js.diagonal().abs().max() < 1e-3


@pytest.mark.parametrize("method", NEW_METHODS)
def test_all_methods_exact_budget_full_coverage_original_means_and_determinism(method):
    torch.manual_seed(5)
    x = torch.randn(160, 24).bfloat16()
    responses = {"embedding": torch.randn(160, 32), "probabilities": torch.randn(160, 16).softmax(-1)}
    kwargs = dict(height=10, width=16, method=method, diagonal=torch.linspace(.1, 2., 24), responses=responses)
    a, b = select_and_merge(x, **kwargs), select_and_merge(x, **kwargs)
    assert a["members"] == b["members"]
    assert len(a["indices"]) == len(set(a["indices"])) == 128
    assert sorted(i for g in a["members"] for i in g) == list(range(160))
    assert all(a["cluster_sizes"])
    for value, members in zip(a["tokens"], a["members"]):
        assert torch.equal(value, x[members].float().mean(0).bfloat16())
    from mllm_comp.analysis.common.compression_training_check import compact_anchor
    length, start = 169, 3
    ids = torch.full((1,length), 3); ids[:,start:start+160] = 2
    labels = torch.full_like(ids, -100); labels[0,-1] = 4
    positions = torch.arange(length)[None,None].expand(3,1,length).clone()
    state = {"features":x, "hidden":torch.randn(1,length,24).bfloat16(), "positions":positions,
        "visual":list(range(start,start+160)), "prefix":length-1,
        "teacher_inputs":{"input_ids":ids, "labels":labels, "attention_mask":torch.ones_like(ids)}}
    compact = compact_anchor(state, a)
    assert compact["hidden"].shape == (1,137,24)
    assert compact["labels"][0,-1] == 4
    assert torch.equal(compact["positions"][:,:,-6:], positions[:,:,-6:])


def test_cosine_control_identical_to_previous_implementation():
    torch.manual_seed(42)
    x = torch.randn(180, 24).bfloat16()
    old, new = cluster(x, height=10, width=18), select_and_merge(x, height=10, width=18, method="cosine_farthest")
    assert old["anchors"] == new["anchors"]
    assert old["members"] == new["members"]
    assert torch.equal(old["tokens"], new["tokens"])


@pytest.mark.parametrize("metric", ["cosine", "l1", "l2", "js"])
def test_duplicate_features_and_ties_still_give_nonempty_clusters(metric):
    values = torch.ones(160, 8)
    if metric == "js":
        values /= 8
    anchors, groups = farthest_partition(values, 128, metric)
    assert anchors == list(range(128))
    assert all(groups)
    assert sorted(i for g in groups for i in g) == list(range(160))


def test_response_extraction_ignores_teacher_answer_and_matches_native_forward():
    from transformers.models.qwen2_5_vl.configuration_qwen2_5_vl import Qwen2_5_VLTextConfig
    from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VLTextModel
    from mllm_comp.analysis.probes.token_metric_ablation import native_response_gate
    config = Qwen2_5_VLTextConfig(vocab_size=32, hidden_size=24, intermediate_size=48, num_hidden_layers=1,
        num_attention_heads=4, num_key_value_heads=2, rope_scaling={"rope_type":"default", "mrope_section":[1,1,1]},
        max_position_embeddings=128, attention_dropout=0.)
    config._attn_implementation = "eager"
    model = Qwen2_5_VLTextModel(config).eval()
    state = {"hidden":torch.randn(1, 13, 24), "positions":torch.arange(13)[None,None].expand(3,1,13).clone(),
        "prefix":10, "visual":list(range(2,6))}
    bundle = {"llm":model, "device":torch.device("cpu")}
    a = extract_responses(bundle, state, [7,8])
    changed = {**state, "hidden":state["hidden"].clone()}
    changed["hidden"][:,10:] = float("nan")
    b = extract_responses(bundle, changed, [7,8])
    assert torch.equal(a["embedding"], b["embedding"])
    assert native_response_gate(bundle, state, [7,8], a)
    with pytest.raises(ValueError, match="exclude the answer"):
        extract_responses(bundle, state, [10])


def test_calibration_selection_is_disjoint_deterministic_and_answer_blind():
    from scripts.analysis.qwen25_vl.run_token_metric_ablation import calibration_selection
    entries = [{"source":s, "sample_id":f"{s}:{i}", "image_fingerprint":f"{s}:{i}", "answer":"a"}
        for s in ["textvqa_train", "scienceqa_train"] for i in range(50)]
    excluded = {f"{s}:0" for s in ["textvqa_train", "scienceqa_train"]}
    a = calibration_selection(entries, excluded)
    b = calibration_selection([{**e,"answer":"changed"} for e in reversed(entries)], excluded)
    assert [e["sample_id"] for e in a] == [e["sample_id"] for e in b]
    assert len(a) == 64
    assert not ({e["image_fingerprint"] for e in a} & excluded)
