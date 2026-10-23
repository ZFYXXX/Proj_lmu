import math

import pytest
import torch

from mllm_comp.analysis.common.spectral_residual import dct_matrix, inverse_dct, spatial_dct, spectral_plan, summary_plan, tile_plan


@pytest.mark.parametrize("height,width", [(16, 16), (18, 30), (31, 19)])
def test_rectangular_dct_round_trip_energy_and_dc(height, width):
    x = torch.randn(height * width, 12, generator=torch.Generator().manual_seed(4))
    coeff, left, right = spatial_dct(x, height, width)
    assert torch.allclose(left @ left.T, torch.eye(height), atol=3e-6)
    restored = inverse_dct(coeff, left, right)
    assert torch.allclose(x, restored, atol=2e-5)
    assert float(coeff.square().sum()) == pytest.approx(float(x.square().sum()), rel=2e-6)
    assert torch.allclose(coeff[0, 0] / math.sqrt(height * width), x.mean(0), atol=2e-6)
    states = spectral_plan(x, height, width, "image")
    assert len(states) == 10
    for fraction in [.05, .10]:
        band_states = [s for s in states if s["trace"]["requested_ac_fraction"] == fraction]
        energies = [float((x - s["values"]).square().sum()) for s in band_states]
        assert max(energies) == pytest.approx(min(energies), rel=2e-5)
        for state in band_states:
            assert state["values"].shape == x.shape
            assert torch.allclose(state["values"].mean(0), x.mean(0), atol=2e-6)
    assert all(torch.equal(a["values"], b["values"]) for a, b in zip(states, spectral_plan(x, height, width, "image")))


@pytest.mark.parametrize("height,width", [(16, 16), (18, 30), (31, 19)])
def test_hybrid_exact_budget_same_summary_and_spatial_match(height, width):
    x = torch.randn(height * width, 12, generator=torch.Generator().manual_seed(4))
    attention = torch.linspace(0, 1, len(x))
    plan = summary_plan(x, height, width, attention, "image")
    repeated = summary_plan(x, height, width, attention, "image")
    assert repeated["selections"] == plan["selections"]
    assert len(plan["base_indices"]) == 96 and len(plan["selections"]) == 11
    assert len({i for group in plan["members"] for i in group}) == len(x)
    for selected in plan["selections"].values():
        assert len(selected) == len(set(selected)) == 32
        assert len(set(selected + plan["base_indices"])) == 128
    for seed in [11, 23, 47]:
        high = plan["selections"]["high_residual"]
        random = plan["selections"][f"spatial_random_{seed}"]
        for group in plan["members"]:
            assert len(set(high).intersection(group)) == len(set(random).intersection(group))
    for cell, group in enumerate(plan["members"]):
        assert torch.allclose(plan["base_values"][cell], x[group].mean(0), atol=1e-6)
    tiles = tile_plan(height, width)
    flat = [i for tile in tiles for i in tile]
    assert len(tiles) == 16 and sorted(flat) == list(range(len(x)))


def test_invalid_grid_and_degenerate_frequency_fail_explicitly():
    with pytest.raises(ValueError):
        spatial_dct(torch.ones(12, 3), 3, 5)
    with pytest.raises(ValueError):
        summary_plan(torch.ones(64, 3), 8, 8, torch.ones(64), "x")
    with pytest.raises(ValueError):
        spectral_plan(torch.zeros(256, 3), 16, 16, "x")


def test_implementation_fingerprint_covers_new_probe_and_dependencies():
    from mllm_comp.analysis.probes.visual_reconstruction_utility import implementation_hash
    assert len(implementation_hash()) == 64


def test_cache_reference_restores_bf16_weights_and_fp32_rope_buffers():
    from transformers import Qwen2_5_VLTextConfig
    from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VLTextModel
    from mllm_comp.analysis.common.cache_parity import verify_bf16_boundary
    config = Qwen2_5_VLTextConfig(vocab_size=32, hidden_size=24, intermediate_size=48, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=2, rope_scaling={"type": "mrope", "mrope_section": [1, 1, 1]})
    config._attn_implementation = "eager"
    torch.manual_seed(8)
    llm = Qwen2_5_VLTextModel(config).eval().to(torch.bfloat16)
    llm.rotary_emb.inv_freq = llm.rotary_emb.inv_freq.float()
    head = torch.nn.Linear(24, 32, bias=False).eval().to(torch.bfloat16)
    weights = {name: value.clone() for name, value in llm.named_parameters()}
    original_inv_freq = llm.rotary_emb.inv_freq.clone()
    with torch.inference_mode():
        hidden = llm.embed_tokens(torch.tensor([[2, 3, 4]]))
        positions = torch.tensor([[[0, 2, 5]], [[0, 3, 5]], [[0, 4, 5]]])
        result = verify_bf16_boundary(llm=llm, head=head, embedding=llm.embed_tokens, hidden=hidden,
            positions=positions, eos_ids=[31], max_new=4)
    assert result["passed"]
    assert all(torch.equal(weights[name], value) for name, value in llm.named_parameters())
    assert torch.equal(original_inv_freq, llm.rotary_emb.inv_freq)
    assert llm.rotary_emb.inv_freq.dtype == torch.float32
