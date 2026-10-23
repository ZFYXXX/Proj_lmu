from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
from mllm_comp.compressors.comparison.prunesid import normalize_prunesid_config, select_prunesid_indices
from mllm_comp.compressors.comparison.qwen_projector import QwenProjectorComparison
from tests.fixtures import native_image


@pytest.mark.parametrize("n,k", [(4, 2), (16, 8), (79, 32), (240, 64), (240, 128), (12, 128)])
def test_prunesid_exact_unique_deterministic_and_rng_isolation(n, k):
    torch.manual_seed(37)
    x = torch.randn(1, n, 48)
    before = torch.random.get_rng_state().clone()
    cfg = normalize_prunesid_config({"tokens_per_crop": k})
    indices, stats = select_prunesid_indices(x, cfg=cfg)
    assert torch.equal(before, torch.random.get_rng_state())
    again, _ = select_prunesid_indices(x, cfg=cfg)
    assert torch.equal(indices[0], again[0])
    assert len(indices[0]) == len(indices[0].unique()) == min(k, n)
    assert indices[0].tolist() == sorted(indices[0].tolist())
    assert not stats["trained"]


@pytest.mark.parametrize("value", [0, True, 1.5])
def test_invalid_budget(value):
    with pytest.raises(ValueError):
        normalize_prunesid_config({"tokens_per_crop": value})


def test_nonfinite_rejected():
    with pytest.raises(ValueError, match="finite"):
        select_prunesid_indices(torch.full((1, 8, 4), float("nan")), cfg={})


@pytest.mark.parametrize("constant", [0., 1., -1.])
def test_degenerate_features_never_duplicate(constant):
    selected, _ = select_prunesid_indices(torch.full((1, 80, 24), constant), cfg={"tokens_per_crop": 32})
    assert len(selected[0].unique()) == 32


@pytest.mark.parametrize("method", ["prunesid", "eadp"])
def test_tiny_qwen_forward_cache_decode_and_uninstall(method):
    torch.manual_seed(100)
    model = native_image.qwen_model("qwen25_vl").eval()
    model.resize_token_embeddings(151650, mean_resizing=False)
    inputs = native_image.inputs(torch, "qwen25_vl")
    ids = torch.tensor([[151644, 872, 198, 7, 10, 10, 10, 10, 8, 2, 151645, 3]])
    inputs.update(input_ids=ids, attention_mask=torch.ones_like(ids))
    original = model.forward.__func__
    activation = QwenProjectorComparison(model, method, {"tokens_per_crop": 2}, "qwen25_vl")
    activation.install()
    try:
        with torch.no_grad():
            dense = model(**inputs, use_cache=False)
            assert dense.logits.shape[1] == ids.shape[1] - 2
            assert torch.isfinite(dense.logits).all()
            output = model.generate(**inputs, max_new_tokens=3, min_new_tokens=3, do_sample=False,
                pad_token_id=0, eos_token_id=None)
            record = activation.runtime_metadata()["selection"]
            assert record["source_count"] == 4 and record["result_count"] == 2
            assert len(output[0]) == len(ids[0]) + 3
            assert len(set(record["selected_indices"])) == 2
            assert record["decode_position_verified"] is True
            assert record["observed_visual_tokens_per_decoder_layer"] == [2] * 4
    finally:
        activation.uninstall()
    assert model.forward.__func__ is original


@pytest.mark.parametrize("method", ["prunesid", "eadp"])
def test_identity_budget_preserves_dense_logits(method):
    torch.manual_seed(101)
    model = native_image.qwen_model("qwen25_vl").eval()
    model.resize_token_embeddings(151650, mean_resizing=False)
    inputs = native_image.inputs(torch, "qwen25_vl")
    ids = torch.tensor([[151644, 872, 198, 7, 10, 10, 10, 10, 8, 2, 151645, 3]])
    inputs.update(input_ids=ids, attention_mask=torch.ones_like(ids))
    with torch.no_grad():
        expected = model(**inputs, use_cache=False).logits
    activation = QwenProjectorComparison(model, method, {"tokens_per_crop": 4}, "qwen25_vl")
    activation.install()
    try:
        with torch.no_grad():
            actual = model(**inputs, use_cache=False).logits
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
    finally:
        activation.uninstall()
