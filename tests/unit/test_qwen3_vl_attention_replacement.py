"""Contracts for Qwen3-VL linear-attention/CNN replacement stages."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from mllm_comp.backbones.qwen3_vl import create_qwen3_vl_backbone
from mllm_comp.config.schema import experiment_config_from_dict
from mllm_comp.pipeline.runtime_compression import activate_runtime_compression
from mllm_comp.compressors.base import CompressionContext
from mllm_comp.compressors.llm_level import Qwen3VLAttentionReplacement
from mllm_comp.compressors.llm_level import Qwen3VLFFNReplacement
from mllm_comp.core.state import ForwardState
from mllm_comp.core.vocabulary import ExecutionPhase, Placement


def _plugin_config(strategy: str, **private: Any) -> dict[str, Any]:
    return {
        "operation": {"type": "replace"},
        "criterion": {"score_source": "manual"},
        "schedule": {"targets": [{"site": "llm.layer[0].attn"}]},
        "config": {"strategy": strategy, **private},
    }


def _install_operation(
    *,
    facade: Any,
    model: Any,
    plugin: str,
    name: str,
    config: dict[str, Any],
) -> Any:
    experiment = experiment_config_from_dict({
        "backbone": {"name": "qwen3_vl"},
        "pipeline": {"stages": [{"name": name, "algorithm": plugin, **config}]},
    })
    session = activate_runtime_compression(
        config=experiment, facade=facade,
        runtime=SimpleNamespace(model=model, metadata={"runtime_mode": "default"}),
        compression_mode="compressed", execution_phase=ExecutionPhase.GENERATE,
    )
    return session.active_records[0]



def test_replacement_planner_emits_attention_actions() -> None:
    plugin = Qwen3VLAttentionReplacement()
    for strategy in (
        "attn_linear_attention",
        "attn_cnn",
        "attn_identity",
        "attn_lolcats_linear_window",
        "attn_nha",
        "attn_sparse_window",
    ):
        plan = plugin.plan(
            state=ForwardState(),
            context=CompressionContext(
                phase="eval",
                model=None,
                extra={"stage_name": f"replace-{strategy}"},
            ),
            config=_plugin_config(strategy),
        )
        assert plan.stage_name == f"replace-{strategy}"
        assert plan.method == "operation_compression"
        assert len(plan.operation_actions) == 1
        action = plan.operation_actions[0]
        assert action.op == "replace"
        assert action.module_path == "llm.layer[0].attn"
        assert action.strategy_name == strategy
        assert action.requires_checkpoint_export is False


def test_lolcats_planner_preserves_layer_specific_configuration() -> None:
    config = _plugin_config(
        "attn_lolcats_linear_window",
        window_size=32,
        decode_window_size=16,
        feature_dim=24,
        window_weight_init=0.25,
        affine_attention_factors=True,
    )
    plan = Qwen3VLAttentionReplacement().plan(
        state=ForwardState(),
        context=CompressionContext(
            phase="eval",
            model=None,
            extra={"stage_name": "replace-lolcats"},
        ),
        config=config,
    )
    action = plan.operation_actions[0]
    assert action.strategy_name == "attn_lolcats_linear_window"
    assert action.metadata["window_size"] == 32
    assert action.metadata["decode_window_size"] == 16
    assert action.metadata["feature_dim"] == 24
    assert action.metadata["window_weight_init"] == 0.25
    assert action.metadata["affine_attention_factors"] is True


def test_nha_planner_preserves_configuration() -> None:
    plan = Qwen3VLAttentionReplacement().plan(
        state=ForwardState(),
        context=CompressionContext(phase="eval", model=None),
        config=_plugin_config(
            "attn_nha",
            num_slots=8,
            window_size=24,
            gate_logit_normalizer=4.0,
            eps=1.0e-5,
        ),
    )
    metadata = plan.operation_actions[0].metadata
    assert metadata["num_slots"] == 8
    assert metadata["window_size"] == 24
    assert metadata["gate_logit_normalizer"] == 4.0
    assert metadata["eps"] == 1.0e-5


def test_ffn_low_rank_planner_emits_replace_action() -> None:
    config = {
        "operation": {"type": "replace"},
        "criterion": {"score_source": "manual"},
        "schedule": {"targets": [{"site": "llm.layer[0].ffn"}]},
        "config": {
            "strategy": "ffn_low_rank",
            "rank": 8,
            "initialization": "column_sampling",
        },
    }
    plan = Qwen3VLFFNReplacement().plan(
        state=ForwardState(),
        context=CompressionContext(
            phase="eval",
            model=None,
            extra={"stage_name": "replace-ffn"},
        ),
        config=config,
    )
    action = plan.operation_actions[0]
    assert plan.stage_name == "replace-ffn"
    assert plan.object_subtype == "ffn"
    assert action.target_kind == "ffn"
    assert action.module_path == "llm.layer[0].ffn"
    assert action.strategy_name == "ffn_low_rank"
    assert action.metadata["rank"] == 8


def test_ffn_two_layer_mlp_planner_preserves_configurable_bottleneck() -> None:
    config = {
        "operation": {"type": "replace"},
        "criterion": {"score_source": "manual"},
        "schedule": {"targets": [{"site": "llm.layer[2].ffn"}]},
        "config": {
            "strategy": "ffn_two_layer_mlp",
            "bottleneck_dim": 12,
            "bias": True,
        },
    }
    plan = Qwen3VLFFNReplacement().plan(
        state=ForwardState(),
        context=CompressionContext(
            phase="eval",
            model=None,
            extra={"stage_name": "replace-two-layer-ffn"},
        ),
        config=config,
    )
    action = plan.operation_actions[0]
    assert action.strategy_name == "ffn_two_layer_mlp"
    assert action.module_path == "llm.layer[2].ffn"
    assert action.metadata["bottleneck_dim"] == 12
    assert action.metadata["bias"] is True


@pytest.mark.parametrize(
    "private",
    [
        {"bottleneck_dim": 0},
        {"bottleneck_dim": True},
        {"bottleneck_dim": 8, "bias": "false"},
    ],
)
def test_ffn_two_layer_mlp_planner_rejects_invalid_config(
    private: dict[str, Any],
) -> None:
    config = {
        "operation": {"type": "replace"},
        "criterion": {"score_source": "manual"},
        "schedule": {"targets": [{"site": "llm.layer[0].ffn"}]},
        "config": {"strategy": "ffn_two_layer_mlp", **private},
    }
    with pytest.raises(ValueError, match="two-layer FFN"):
        Qwen3VLFFNReplacement().validate_config(config)


@pytest.mark.parametrize(
    ("strategy", "private", "message"),
    [
        ("softmax", {}, "strategy"),
        ("attn_cnn", {"kernel_size": 2}, "kernel_size"),
        ("attn_linear_attention", {"eps": 0}, "eps"),
        ("attn_lolcats_linear_window", {"window_size": 0}, "window_size"),
        ("attn_lolcats_linear_window", {"feature_dim": 0}, "feature_dim"),
        ("attn_nha", {"num_slots": 0}, "num_slots"),
        ("attn_nha", {"window_size": 0}, "window_size"),
        ("attn_nha", {"gate_logit_normalizer": 0}, "gate_logit_normalizer"),
        (
            "attn_lolcats_linear_window",
            {"window_weight_init": 1.0},
            "window_weight_init",
        ),
        ("attn_sparse_window", {"local_left": -1}, "local_left"),
        ("attn_sparse_window", {"causal": True, "local_right": 1}, "local_right"),
    ],
)
def test_replacement_planner_rejects_invalid_configuration(
    strategy: str,
    private: dict[str, Any],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        Qwen3VLAttentionReplacement().validate_config(
            _plugin_config(strategy, **private)
        )


def _torch_fixture() -> tuple[Any, Any]:
    torch = pytest.importorskip("torch")
    nn = torch.nn

    class FakeAttention(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.config = SimpleNamespace(
                num_attention_heads=4,
                num_key_value_heads=2,
            )
            self.layer_idx = 0
            self.head_dim = 4
            self.q_proj = nn.Linear(16, 16, bias=False)
            self.k_proj = nn.Linear(16, 8, bias=False)
            self.v_proj = nn.Linear(16, 8, bias=False)
            self.o_proj = nn.Linear(16, 16, bias=False)
            self.q_norm = nn.Identity()
            self.k_norm = nn.Identity()

    return torch, FakeAttention


def _fake_mlp(torch: Any) -> Any:
    nn = torch.nn

    class FakeMLP(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.gate_proj = nn.Linear(16, 32, bias=False)
            self.up_proj = nn.Linear(16, 32, bias=False)
            self.down_proj = nn.Linear(32, 16, bias=False)
            self.act_fn = torch.nn.functional.silu

        def forward(self, hidden_states: Any) -> Any:
            return self.down_proj(
                self.act_fn(self.gate_proj(hidden_states)) * self.up_proj(hidden_states)
            )

    return FakeMLP()


@pytest.mark.parametrize(
    "strategy",
    [
        "attn_linear_attention",
        "attn_cnn",
        "attn_identity",
        "attn_lolcats_linear_window",
        "attn_nha",
        "attn_sparse_window",
    ],
)
def test_replacements_run_forward_and_backward(strategy: str) -> None:
    torch, fake_attention_type = _torch_fixture()
    from mllm_comp.backbones.qwen3_vl.attention_replacement import (
        Qwen3VLCausalConvAttention,
        Qwen3VLIdentityAttention,
        Qwen3VLLinearAttention,
        Qwen3VLLoLCATsLinearWindowAttention,
        Qwen3VLNativeHybridAttention,
        Qwen3VLSparseWindowAttention,
    )

    original = fake_attention_type()
    replacements = {
        "attn_linear_attention": lambda: Qwen3VLLinearAttention(original, debug=True),
        "attn_cnn": lambda: Qwen3VLCausalConvAttention(
            original, kernel_size=3, debug=True
        ),
        "attn_identity": lambda: Qwen3VLIdentityAttention(original, debug=True),
        "attn_lolcats_linear_window": lambda: Qwen3VLLoLCATsLinearWindowAttention(
            original,
            window_size=3,
            feature_dim=4,
            debug=True,
        ),
        "attn_nha": lambda: Qwen3VLNativeHybridAttention(
            original,
            num_slots=2,
            window_size=3,
            debug=True,
        ),
        "attn_sparse_window": lambda: Qwen3VLSparseWindowAttention(
            original,
            local_left=2,
            prefix_size=1,
            debug=True,
        ),
    }
    replacement = replacements[strategy]()
    hidden_states = torch.randn(2, 5, 16, requires_grad=True)
    output, weights = replacement(
        hidden_states,
        position_embeddings=None,
        attention_mask=torch.ones(2, 5),
    )
    assert output.shape == hidden_states.shape
    assert weights is None
    assert replacement.forward_calls == 1
    assert replacement.last_debug["strategy"] == strategy
    assert replacement.last_debug["all_finite"] is True
    output.square().mean().backward()
    assert hidden_states.grad is not None


def test_lolcats_matches_explicit_linear_plus_window_formula() -> None:
    torch, fake_attention_type = _torch_fixture()
    from mllm_comp.backbones.qwen3_vl.attention_replacement import (
        Qwen3VLLoLCATsLinearWindowAttention,
    )

    replacement = Qwen3VLLoLCATsLinearWindowAttention(
        fake_attention_type(),
        window_size=2,
        feature_dim=3,
        window_weight_init=0.2,
        affine_attention_factors=False,
        debug=True,
    )
    hidden_states = torch.randn(1, 5, 16)
    actual, _ = replacement(
        hidden_states,
        position_embeddings=None,
        attention_mask=torch.ones(1, 5),
    )

    batch, sequence_length, _ = hidden_states.shape
    query = replacement.q_proj(hidden_states).view(batch, sequence_length, 4, 4)
    query = query.transpose(1, 2)
    key = replacement.k_proj(hidden_states).view(batch, sequence_length, 2, 4)
    value = replacement.v_proj(hidden_states).view(batch, sequence_length, 2, 4)
    key = key.transpose(1, 2)[:, :, None, :, :].expand(-1, -1, 2, -1, -1)
    value = value.transpose(1, 2)[:, :, None, :, :].expand(-1, -1, 2, -1, -1)
    key = key.reshape(batch, 4, sequence_length, 4)
    value = value.reshape(batch, 4, sequence_length, 4)

    def feature(tensor: Any, weight: Any) -> Any:
        projected = torch.einsum("bhld,hdf->bhlf", tensor.float(), weight.float())
        return torch.cat(
            (torch.softmax(projected, -1), torch.softmax(-projected, -1)),
            dim=-1,
        )

    query_feature = feature(query, replacement.q_feature_weight)
    key_feature = feature(key, replacement.k_feature_weight)
    beta = torch.sigmoid(replacement.window_factor_logits.float())[:, :, 0, :]
    expected_heads = []
    for position in range(sequence_length):
        local_start = max(0, position - replacement.window_size + 1)
        local_scores = (
            torch.einsum(
                "bhd,bhkd->bhk",
                query[:, :, position, :].float(),
                key[:, :, local_start : position + 1, :].float(),
            )
            * replacement.scaling
        )
        local_weights = beta * torch.exp(
            local_scores - local_scores.amax(dim=-1, keepdim=True)
        )
        local_numerator = torch.einsum(
            "bhk,bhkd->bhd",
            local_weights,
            value[:, :, local_start : position + 1, :].float(),
        )
        local_denominator = local_weights.sum(dim=-1, keepdim=True)
        old_feature = key_feature[:, :, :local_start, :]
        old_value = value[:, :, :local_start, :].float()
        linear_weights = torch.einsum(
            "bhf,bhkf->bhk",
            query_feature[:, :, position, :],
            old_feature,
        )
        linear_numerator = torch.einsum(
            "bhk,bhkd->bhd",
            linear_weights,
            old_value,
        )
        linear_denominator = linear_weights.sum(dim=-1, keepdim=True)
        expected_heads.append(
            (local_numerator + linear_numerator)
            / (local_denominator + linear_denominator).clamp_min(replacement.eps)
        )
        assert local_start + (position - local_start + 1) == position + 1
    expected_heads = torch.stack(expected_heads, dim=2).to(hidden_states.dtype)
    expected = replacement.o_proj(
        expected_heads.transpose(1, 2).reshape(batch, sequence_length, 16)
    )
    torch.testing.assert_close(actual, expected, rtol=1.0e-5, atol=1.0e-5)


def test_lolcats_recurrent_decode_matches_full_history() -> None:
    torch, fake_attention_type = _torch_fixture()
    from mllm_comp.backbones.qwen3_vl.attention_replacement import (
        Qwen3VLLoLCATsLinearWindowAttention,
    )

    replacement = Qwen3VLLoLCATsLinearWindowAttention(
        fake_attention_type(),
        window_size=3,
        feature_dim=4,
    )
    query = torch.randn(1, 4, 6, 4)
    key = torch.randn(1, 4, 6, 4)
    value = torch.randn(1, 4, 6, 4)
    keep = torch.ones(1, 6, dtype=torch.bool)
    expected, _ = replacement._hybrid_from_full_history(
        query[:, :, -1:, :],
        key,
        value,
        keep,
        past_length=5,
        window_size=3,
    )
    state = replacement._make_runtime_state(
        key[:, :, :-1, :],
        value[:, :, :-1, :],
        keep[:, :-1],
        window_size=3,
    )
    actual, _ = replacement._decode_from_runtime(
        state,
        query[:, :, -1:, :],
        key[:, :, -1:, :],
        value[:, :, -1:, :],
        keep[:, None, -1:, None],
        window_size=3,
    )
    torch.testing.assert_close(actual, expected, rtol=1.0e-5, atol=1.0e-5)
    assert state.seen_tokens == 6
    assert state.window_keys.shape[2] == 3


def test_nha_matches_explicit_gated_slots_plus_window_formula() -> None:
    torch, fake_attention_type = _torch_fixture()
    from mllm_comp.backbones.qwen3_vl.attention_replacement import (
        Qwen3VLNativeHybridAttention,
    )

    replacement = Qwen3VLNativeHybridAttention(
        fake_attention_type(),
        num_slots=2,
        window_size=2,
        gate_logit_normalizer=8.0,
        debug=True,
    )
    sequence_length = 10
    hidden = torch.randn(1, sequence_length, 16)
    actual, _ = replacement(
        hidden,
        position_embeddings=None,
        attention_mask=torch.ones(1, sequence_length),
    )
    query = replacement.q_proj(hidden).view(1, sequence_length, 4, 4).transpose(1, 2)
    key = replacement.k_proj(hidden).view(1, sequence_length, 2, 4).transpose(1, 2)
    value = replacement.v_proj(hidden).view(1, sequence_length, 2, 4).transpose(1, 2)
    key = key[:, :, None].expand(-1, -1, 2, -1, -1).reshape(1, 4, sequence_length, 4)
    value = (
        value[:, :, None].expand(-1, -1, 2, -1, -1).reshape(1, 4, sequence_length, 4)
    )
    gate = replacement.g_proj(hidden).view(1, sequence_length, 2, 2).transpose(1, 2)
    gate = gate[:, :, None].expand(-1, -1, 2, -1, -1).reshape(1, 4, sequence_length, 2)
    retention = torch.exp(torch.nn.functional.logsigmoid(gate.float()) / 8.0)
    long_key = torch.zeros(1, 4, 2, 4)
    long_value = torch.zeros_like(long_key)
    expected_heads = []
    old_end = 0
    for position in range(sequence_length):
        next_old_end = max(0, position - 1)
        for moved in range(old_end, next_old_end):
            write = 1.0 - retention[:, :, moved, :]
            long_key = (
                retention[:, :, moved, :, None] * long_key
                + write[..., None] * key[:, :, moved, None, :]
            )
            long_value = (
                retention[:, :, moved, :, None] * long_value
                + write[..., None] * value[:, :, moved, None, :]
            )
        old_end = next_old_end
        logits = (
            torch.cat(
                (
                    torch.einsum("bhd,bhmd->bhm", query[:, :, position], long_key),
                    torch.einsum(
                        "bhd,bhwd->bhw",
                        query[:, :, position],
                        key[:, :, old_end : position + 1],
                    ),
                ),
                dim=-1,
            )
            * replacement.scaling
        )
        probability = torch.softmax(logits.float(), dim=-1)
        expected_heads.append(
            torch.einsum("bhm,bhmd->bhd", probability[:, :, :2], long_value)
            + torch.einsum(
                "bhw,bhwd->bhd",
                probability[:, :, 2:],
                value[:, :, old_end : position + 1],
            )
        )
    expected_heads = torch.stack(expected_heads, dim=2).to(hidden.dtype)
    expected = replacement.o_proj(
        expected_heads.transpose(1, 2).reshape(1, sequence_length, 16)
    )
    torch.testing.assert_close(actual, expected, rtol=1.0e-5, atol=1.0e-5)
    assert (
        replacement.last_debug["nha_score_pairs"]
        < replacement.last_debug["full_attention_score_pairs"]
    )


def test_nha_gate_initialization_and_recurrent_decode_are_exact() -> None:
    torch, fake_attention_type = _torch_fixture()
    from mllm_comp.backbones.qwen3_vl.attention_replacement import (
        Qwen3VLNativeHybridAttention,
    )

    replacement = Qwen3VLNativeHybridAttention(
        fake_attention_type(), num_slots=2, window_size=3
    )
    expected_gate = replacement.k_proj.weight.view(2, 2, 2, 16).mean(dim=2)
    torch.testing.assert_close(
        replacement.g_proj.weight.view(2, 2, 16), expected_gate, rtol=0, atol=0
    )
    query = torch.randn(1, 4, 6, 4)
    key = torch.randn(1, 4, 6, 4)
    value = torch.randn(1, 4, 6, 4)
    retention = torch.sigmoid(torch.randn(1, 4, 6, 2))
    keep = torch.ones(1, 6, dtype=torch.bool)
    expected, _, _ = replacement._prefill(
        query, query, key, key, value, retention, keep
    )
    _, _, initial_state = replacement._prefill(
        query[:, :, :5],
        query[:, :, :5],
        key[:, :, :5],
        key[:, :, :5],
        value[:, :, :5],
        retention[:, :, :5],
        keep[:, :5],
    )
    state = replacement._make_runtime_state(initial_state, 5)
    actual, _ = replacement._decode(
        state,
        query[:, :, 5:],
        query[:, :, 5:],
        key[:, :, 5:],
        key[:, :, 5:],
        value[:, :, 5:],
        retention[:, :, 5:],
        keep[:, 5:],
    )
    torch.testing.assert_close(actual, expected[:, :, 5:], rtol=1.0e-6, atol=1.0e-6)
    assert state.seen_tokens == 6
    assert state.window_keys.shape[2] == 3


@pytest.mark.parametrize(
    "strategy",
    [
        "attn_linear_attention",
        "attn_cnn",
        "attn_identity",
        "attn_lolcats_linear_window",
        "attn_nha",
        "attn_sparse_window",
    ],
)
def test_pipeline_replaces_real_parent_attribute(strategy: str) -> None:
    _torch, fake_attention_type = _torch_fixture()
    facade = create_qwen3_vl_backbone()
    layer = SimpleNamespace(self_attn=fake_attention_type(), mlp=object())
    model = SimpleNamespace(
        model=SimpleNamespace(
            language_model=SimpleNamespace(layers=[layer]),
            visual=None,
        ),
        lm_head=None,
    )
    original = layer.self_attn
    record = _install_operation(
        facade=facade,
        model=model,
        plugin="qwen3_vl_attention_replacement",
        name=f"replace-{strategy}",
        config=_plugin_config(
            strategy,
            **({"num_slots": 2, "window_size": 3} if strategy == "attn_nha" else {}),
        ),
    )
    assert layer.self_attn.strategy_name == strategy
    assert layer.self_attn.mllm_comp_replacement_strategy == strategy
    assert record.runtime_metadata()["installed_handles"] == ["llm.layer[0].attn"]
    record.uninstall()
    assert layer.self_attn is original


def test_sparse_window_combines_and_deduplicates_configured_regions() -> None:
    torch, fake_attention_type = _torch_fixture()
    from mllm_comp.backbones.qwen3_vl.attention_replacement import (
        Qwen3VLSparseWindowAttention,
    )

    replacement = Qwen3VLSparseWindowAttention(
        fake_attention_type(),
        causal=True,
        include_self=True,
        local_left=2,
        prefix_size=2,
        prompt_suffix_size=3,
        fixed_positions=[5, 15],
        stride=4,
    )
    indices = replacement._selected_key_indices(
        query_position=10,
        total_length=20,
        prompt_length=20,
        device=torch.device("cpu"),
    )
    assert indices.tolist() == [0, 1, 4, 5, 8, 9, 10]


def test_low_rank_ffn_runs_backward_and_reduces_parameters() -> None:
    torch = pytest.importorskip("torch")
    from mllm_comp.backbones.qwen3_vl.ffn_replacement import Qwen3VLLowRankMLP

    original = _fake_mlp(torch)
    original_parameters = sum(parameter.numel() for parameter in original.parameters())
    replacement = Qwen3VLLowRankMLP(original, rank=4, debug=True)
    replacement_parameters = sum(
        parameter.numel() for parameter in replacement.parameters()
    )
    hidden_states = torch.randn(2, 5, 16, requires_grad=True)
    output = replacement(hidden_states)

    assert output.shape == hidden_states.shape
    assert replacement_parameters < original_parameters
    assert replacement.last_debug["all_finite"] is True
    assert replacement.last_debug["parameter_compression_ratio"] < 1
    output.square().mean().backward()
    assert hidden_states.grad is not None


@pytest.mark.parametrize("bottleneck_dim", [2, 4, 8])
def test_two_layer_mlp_matches_explicit_reference_and_runs_backward(
    bottleneck_dim: int,
) -> None:
    torch = pytest.importorskip("torch")
    from mllm_comp.backbones.qwen3_vl.ffn_replacement import Qwen3VLTwoLayerMLP

    original = _fake_mlp(torch)
    original_parameters = sum(parameter.numel() for parameter in original.parameters())
    replacement = Qwen3VLTwoLayerMLP(
        original,
        bottleneck_dim=bottleneck_dim,
        bias=True,
        debug=True,
    )
    hidden_states = torch.randn(2, 5, 16, requires_grad=True)
    actual = replacement(hidden_states)
    reduced = torch.nn.functional.linear(
        hidden_states,
        replacement.reduce_proj.weight,
        replacement.reduce_proj.bias,
    )
    expected = torch.nn.functional.linear(
        torch.nn.functional.silu(reduced),
        replacement.expand_proj.weight,
        replacement.expand_proj.bias,
    )

    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert actual.shape == hidden_states.shape
    assert replacement.bottleneck_dim == bottleneck_dim
    assert replacement.last_debug["reduced_shape"] == (2, 5, bottleneck_dim)
    assert replacement.last_debug["all_finite"] is True
    assert (
        sum(parameter.numel() for parameter in replacement.parameters())
        < original_parameters
    )
    actual.square().mean().backward()
    assert hidden_states.grad is not None


def test_pipeline_replaces_ffn_parent_attribute() -> None:
    torch = pytest.importorskip("torch")
    facade = create_qwen3_vl_backbone()
    layer = SimpleNamespace(self_attn=object(), mlp=_fake_mlp(torch))
    model = SimpleNamespace(
        model=SimpleNamespace(
            language_model=SimpleNamespace(layers=[layer]),
            visual=None,
        ),
        lm_head=None,
    )
    config = {
        "operation": {"type": "replace"},
        "criterion": {"score_source": "manual"},
        "schedule": {"targets": [{"site": "llm.layer[0].ffn"}]},
        "config": {
            "strategy": "ffn_low_rank",
            "rank": 4,
            "initialization": "column_sampling",
        },
    }
    original = layer.mlp
    record = _install_operation(
        facade=facade,
        model=model,
        plugin="qwen3_vl_ffn_replacement",
        name="replace-ffn",
        config=config,
    )

    assert layer.mlp.strategy_name == "ffn_low_rank"
    assert layer.mlp.rank == 4
    assert record.runtime_metadata()["installed_handles"] == ["llm.layer[0].ffn"]
    record.uninstall()
    assert layer.mlp is original


def test_pipeline_installs_configurable_two_layer_mlp() -> None:
    torch = pytest.importorskip("torch")
    facade = create_qwen3_vl_backbone()
    layer = SimpleNamespace(self_attn=object(), mlp=_fake_mlp(torch))
    model = SimpleNamespace(
        model=SimpleNamespace(
            language_model=SimpleNamespace(layers=[layer]),
            visual=None,
        ),
        lm_head=None,
    )
    config = {
        "operation": {"type": "replace"},
        "criterion": {"score_source": "manual"},
        "schedule": {"targets": [{"site": "llm.layer[0].ffn"}]},
        "config": {
            "strategy": "ffn_two_layer_mlp",
            "bottleneck_dim": 6,
            "bias": False,
            "debug": True,
        },
    }
    original = layer.mlp
    record = _install_operation(
        facade=facade,
        model=model,
        plugin="qwen3_vl_ffn_replacement",
        name="replace-ffn-two-layer-mlp",
        config=config,
    )

    assert layer.mlp.strategy_name == "ffn_two_layer_mlp"
    assert layer.mlp.bottleneck_dim == 6
    assert layer.mlp.bias is False
    assert record.runtime_metadata()["installed_handles"] == ["llm.layer[0].ffn"]
    record.uninstall()
    assert layer.mlp is original


def test_ffn_identity_runs_forward_and_pipeline_installs_it() -> None:
    torch = pytest.importorskip("torch")
    facade = create_qwen3_vl_backbone()
    layer = SimpleNamespace(self_attn=object(), mlp=_fake_mlp(torch))
    model = SimpleNamespace(
        model=SimpleNamespace(
            language_model=SimpleNamespace(layers=[layer]),
            visual=None,
        ),
        lm_head=None,
    )
    config = {
        "operation": {"type": "replace"},
        "criterion": {"score_source": "manual"},
        "schedule": {"targets": [{"site": "llm.layer[0].ffn"}]},
        "config": {"strategy": "ffn_identity", "debug": True},
    }
    original = layer.mlp
    record = _install_operation(
        facade=facade,
        model=model,
        plugin="qwen3_vl_ffn_replacement",
        name="replace-ffn-identity",
        config=config,
    )
    hidden_states = torch.randn(2, 5, 16, requires_grad=True)
    output = layer.mlp(hidden_states)
    assert layer.mlp.strategy_name == "ffn_identity"
    assert torch.equal(output, hidden_states)
    record.uninstall()
    assert layer.mlp is original
    output.square().mean().backward()
    assert hidden_states.grad is not None


@pytest.mark.parametrize("kind", ["ffn", "attention"])
def test_repeated_replacement_reuses_only_the_same_effective_configuration(kind):
    torch, fake_attention_type = _torch_fixture()
    from mllm_comp.core.errors import BackendIntegrationError

    layer = SimpleNamespace(self_attn=fake_attention_type(), mlp=_fake_mlp(torch))
    model = SimpleNamespace(model=SimpleNamespace(
        language_model=SimpleNamespace(layers=[layer]), visual=None), lm_head=None)
    facade = create_qwen3_vl_backbone()
    attribute = "mlp" if kind == "ffn" else "self_attn"
    original = getattr(layer, attribute)
    private = ({"strategy": "ffn_low_rank", "rank": 4} if kind == "ffn" else
               {"strategy": "attn_nha", "num_slots": 2, "window_size": 3})
    config = {
        "operation": {"type": "replace"},
        "schedule": {"targets": [{"site": "llm.layer[0]." + ("ffn" if kind == "ffn" else "attn")}]},
        "config": private,
    }
    plugin = f"qwen3_vl_{kind}_replacement"
    first = _install_operation(facade=facade, model=model, plugin=plugin, name="first", config=config)
    installed = getattr(layer, attribute)
    try:
        # Spelling out a default must still mean the same replacement.
        explicit = {**private, **({"initialization": "column_sampling"} if kind == "ffn" else {"eps": 1e-6})}
        repeated = _install_operation(facade=facade, model=model, plugin=plugin, name="repeated",
                                      config={**config, "config": explicit})
        assert getattr(layer, attribute) is installed
        repeated.uninstall()
        assert getattr(layer, attribute) is installed
        changed = {**private, **({"rank": 8} if kind == "ffn" else {"window_size": 4})}
        with pytest.raises(BackendIntegrationError, match="different.*configuration"):
            _install_operation(facade=facade, model=model, plugin=plugin, name="changed",
                               config={**config, "config": changed})
        assert getattr(layer, attribute) is installed
    finally:
        first.uninstall()
    assert getattr(layer, attribute) is original


@pytest.mark.parametrize("strategy", ["lolcats", "nha"])
def test_recurrent_attention_follows_native_cache_beam_order_and_ownership(strategy):
    torch, fake_attention_type = _torch_fixture()
    pytest.importorskip("transformers")
    from transformers.cache_utils import DynamicCache
    from mllm_comp.backbones.qwen3_vl.attention_replacement import (
        Qwen3VLLoLCATsLinearWindowAttention, Qwen3VLNativeHybridAttention,
    )
    import weakref

    torch.manual_seed(883)
    replacement = (Qwen3VLLoLCATsLinearWindowAttention(
        fake_attention_type(), window_size=3, decode_window_size=3, feature_dim=4)
        if strategy == "lolcats" else Qwen3VLNativeHybridAttention(
            fake_attention_type(), num_slots=2, window_size=3)).eval()
    history = torch.randn(2, 5, 16)
    cache = DynamicCache()
    with torch.no_grad():
        replacement(history, None, torch.ones(2, 5), past_key_values=cache)
        # Interleave another cache of the same length on the same module.
        other_cache = DynamicCache()
        replacement(torch.randn_like(history), None, torch.ones(2, 5), past_key_values=other_cache)
        for beam_idx in (torch.tensor([1, 0, 1]), torch.tensor([2, 0, 1])):
            cache.reorder_cache(beam_idx)
            history = history.index_select(0, beam_idx)
            next_hidden = torch.randn(3, 1, 16)
            history = torch.cat((history, next_hidden), dim=1)
            mask = torch.ones(3, history.shape[1])
            actual, _ = replacement(next_hidden, None, mask, past_key_values=cache)
            expected, _ = replacement(history, None, mask)
            torch.testing.assert_close(actual, expected[:, -1:], rtol=1e-5, atol=1e-6)
    # Holding the model does not retain a finished request's cache or history.
    cache_ref = weakref.ref(cache)
    state_ref = weakref.ref(cache._mllm_comp_attention_states.layers[0])
    del cache
    assert cache_ref() is None
    assert state_ref() is None
