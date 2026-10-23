import copy
import torch
import pytest

from mllm_comp.analysis.common.compression_training_check import (
    AnchorAdapterSpec, create_anchor_adapter, anchor_base, compact_anchor, save_anchor, load_anchor,
)


def make_state():
    torch.manual_seed(8)
    n, d, start = 18 * 30, 24, 3
    length = n + 9
    ids = torch.full((1, length), 3)
    ids[:, start:start+n] = 2
    labels = torch.full_like(ids, -100)
    labels[0, -1] = 4
    pos = torch.arange(length)[None, None].expand(3, 1, length).clone()
    pos[1, 0, start:start+n] = torch.arange(n) // 30 + start
    pos[2, 0, start:start+n] = torch.arange(n) % 30 + start
    return {"features": torch.randn(n, d).bfloat16(), "height": 18, "width": 30,
            "hidden": torch.randn(1, length, d).bfloat16(), "positions": pos,
            "visual": list(range(start, start+n)), "prefix": length-1,
            "teacher_inputs": {"input_ids": ids, "labels": labels, "attention_mask": torch.ones_like(ids)}}


def test_zero_initialization_positions_labels_and_gradient():
    state = make_state()
    base = anchor_base(state)
    model = create_anchor_adapter(AnchorAdapterSpec(24, 16))
    output = model(base)
    assert torch.equal(output["tokens"], base["tokens"])
    compact = compact_anchor(state, output)
    assert compact["hidden"].shape == (1, 137, 24)
    assert compact["labels"][0, -1] == 4
    assert torch.equal(compact["positions"][:, :, -6:], state["positions"][:, :, -6:])
    for i, members in enumerate(base["members"]):
        expected = state["positions"][:, 0, [3+j for j in members]].float().mean(1).round().long()
        assert torch.equal(compact["positions"][:, 0, 3+i], expected)
    compact["hidden"].float().square().mean().backward()
    assert model.adapter[-1].weight.grad.abs().sum() > 0
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
    bad = dict(output, members=list(output["members"][:-1]) + [output["members"][0]])
    with pytest.raises(ValueError, match="membership"):
        compact_anchor(state, bad)


def test_save_restore_next_adam_update(tmp_path):
    base = anchor_base(make_state())
    model = create_anchor_adapter(AnchorAdapterSpec(24, 16))
    opt = torch.optim.AdamW(model.parameters(), lr=2e-4)
    def step(m, o):
        m(base)["tokens"].float().square().mean().backward()
        o.step()
        o.zero_grad(set_to_none=True)
    step(model, opt)
    save_anchor(model, tmp_path)
    second = load_anchor(tmp_path, "cpu")
    other = torch.optim.AdamW(second.parameters(), lr=2e-4)
    other.load_state_dict(copy.deepcopy(opt.state_dict()))
    assert torch.equal(model(base)["tokens"], second(base)["tokens"])
    step(model, opt)
    step(second, other)
    assert all(torch.equal(x, y) for x, y in zip(model.parameters(), second.parameters()))
