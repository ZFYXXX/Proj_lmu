"""Native decoder observation lifetime and gradient isolation."""

import pytest

torch = pytest.importorskip("torch")

from mllm_comp.backbones._decoder_observations import capture_decoder_module_transitions  # noqa: E402


class _Attention(torch.nn.Module):
    def forward(self, hidden_states):
        return (hidden_states * 2.0,)


class _Layer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.input_layernorm = torch.nn.Identity()
        self.self_attn = _Attention()
        self.post_attention_layernorm = torch.nn.Identity()
        self.mlp = torch.nn.Identity()

    def forward(self, hidden_states):
        attention = self.self_attn(self.input_layernorm(hidden_states))[0]
        return self.mlp(self.post_attention_layernorm(attention))


def test_temporary_capture_observes_exact_module_boundaries_and_cleans_up():
    layer = _Layer()
    hidden = torch.tensor([[[1., 0.], [0., 1.], [1., 1.]]], requires_grad=True)
    with capture_decoder_module_transitions(layer) as capture:
        output = layer(hidden)

    torch.testing.assert_close(output, hidden * 2.)
    torch.testing.assert_close(capture.attention_before, hidden)
    for value in (capture.attention_after, capture.ffn_before, capture.ffn_after):
        torch.testing.assert_close(value, hidden * 2.)
    assert all(not value.requires_grad for value in vars(capture).values())
    assert all(not module._forward_hooks for module in layer.modules())
    output.sum().backward()
    torch.testing.assert_close(hidden.grad, torch.full_like(hidden, 2.))


@pytest.mark.parametrize("failure", ["forward", "incomplete", "install"])
def test_temporary_capture_removes_hooks_on_failed_native_scope(failure, monkeypatch):
    layer = _Layer()
    hidden = torch.ones(1, 3, 2)
    def fail(*args, **kwargs):
        raise RuntimeError("native failure")
    if failure == "forward":
        monkeypatch.setattr(layer.mlp, "forward", fail)
    elif failure == "install":
        monkeypatch.setattr(layer.self_attn, "register_forward_hook", fail)
    with pytest.raises(RuntimeError, match="incomplete" if failure == "incomplete" else "native failure"):
        with capture_decoder_module_transitions(layer):
            if failure == "incomplete":
                layer.input_layernorm(hidden)
            else:
                layer(hidden)
    assert all(not module._forward_hooks for module in layer.modules())
