import pytest
import torch

from mllm_comp.compressors.image.learned_hybrid import LearnedHybridSpec, VARIANTS, create_learned_hybrid, spatial_groups
from mllm_comp.analysis.common.learned_hybrid_runtime import compact_state, save_module, load_module


@pytest.mark.parametrize("shape", [(16, 16), (18, 30), (31, 19)])
@pytest.mark.parametrize("variant", VARIANTS)
def test_exact_native_budget_rectangular_grid_gradients_and_determinism(shape, variant):
    height, width = shape
    torch.manual_seed(4)
    model = create_learned_hybrid(LearnedHybridSpec(hidden_size=24, variant=variant, width=16, heads=4))
    x = torch.randn(height * width, 24, requires_grad=True)
    output = model(x, height=height, width=width, random_seed=5)
    repeated = model(x, height=height, width=width, random_seed=5)
    assert output["tokens"].shape == (128, 24)
    assert output["indices"] == sorted(set(output["indices"]))
    assert output["indices"] == repeated["indices"]
    assert torch.equal(output["tokens"], repeated["tokens"])
    assert set(output["summary_indices"]).isdisjoint(output["detail_indices"])
    assert len(output["detail_indices"]) == (0 if variant == "pool_adapter" else 64)
    (output["tokens"].square().mean() + .1 * output["reconstruction_loss"]).backward()
    assert x.grad is None
    assert any(p.grad is not None and bool((p.grad != 0).any()) for p in model.parameters())
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
    if variant == "pool_adapter":
        groups, _ = spatial_groups(height, width, 128)
        expected = torch.stack([x.detach()[g].mean(0) for g in groups])
        assert torch.allclose(output["tokens"], expected)


def test_random_and_residual_controls_have_identical_parameters():
    counts = []
    keys = []
    for variant in VARIANTS[1:]:
        torch.manual_seed(42)
        module = create_learned_hybrid(LearnedHybridSpec(hidden_size=24, variant=variant, width=16, heads=4))
        counts.append(sum(p.numel() for p in module.parameters()))
        keys.append({key: value.clone() for key, value in module.state_dict().items()})
    assert len(set(counts)) == 1
    assert all(torch.equal(keys[0][key], item[key]) for item in keys[1:] for key in keys[0])


def test_constant_field_spectral_spatial_initializations_match():
    x = torch.randn(1, 24).expand(540, 24)
    outputs = []
    for variant in ["summary_residual", "spectral_residual"]:
        torch.manual_seed(42)
        module = create_learned_hybrid(LearnedHybridSpec(hidden_size=24, variant=variant, width=16, heads=4))
        outputs.append(module(x, height=18, width=30)["tokens"])
    assert torch.allclose(outputs[0], outputs[1], atol=2e-5)


def test_compaction_preserves_text_labels_positions_and_backpropagates():
    x = torch.randn(256, 24)
    module = create_learned_hybrid(LearnedHybridSpec(hidden_size=24, variant="spectral_residual", width=16, heads=4))
    output = module(x, height=16, width=16)
    tokens = torch.full((1, 265), 3)
    tokens[0, 3:259] = 2
    tokens[0, 264] = 4
    labels = torch.full_like(tokens, -100)
    labels[0, 264] = 4
    state = {"hidden": torch.randn(1, 265, 24), "positions": torch.arange(265)[None, None].expand(3, 1, 265),
             "visual": list(range(3, 259)), "prefix": 264,
             "teacher_inputs": {"input_ids": tokens, "attention_mask": torch.ones_like(tokens), "labels": labels}}
    compact = compact_state(state, output)
    assert compact["hidden"].shape == (1, 137, 24)
    assert compact["prefix"] == 136
    assert compact["labels"][0, -1] == 4
    assert torch.equal(compact["positions"][:, :, -6:], state["positions"][:, :, -6:])
    compact["hidden"].square().mean().backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in module.parameters())


def test_checkpoint_restores_values_and_next_optimizer_update(tmp_path):
    torch.manual_seed(5)
    module = create_learned_hybrid(LearnedHybridSpec(hidden_size=24, variant="summary_residual", width=16, heads=4))
    x = torch.randn(256, 24)
    optimizer = torch.optim.AdamW(module.parameters(), lr=1e-3)
    def update(model, opt):
        out = model(x, height=16, width=16, random_seed=42)
        loss = out["tokens"].square().mean() + .1 * out["reconstruction_loss"]
        loss.backward()
        opt.step()
        opt.zero_grad(set_to_none=True)
    update(module, optimizer)
    save_module(module, tmp_path)
    restored = load_module(tmp_path, "cpu")
    import copy
    second = torch.optim.AdamW(restored.parameters(), lr=1e-3)
    second.load_state_dict(copy.deepcopy(optimizer.state_dict()))
    assert torch.equal(module(x, height=16, width=16)["tokens"], restored(x, height=16, width=16)["tokens"])
    update(module, optimizer)
    update(restored, second)
    assert all(torch.equal(a, b) for a, b in zip(module.parameters(), restored.parameters()))
