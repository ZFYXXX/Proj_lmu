import math

import pytest


torch = pytest.importorskip("torch")

from mllm_comp.compressors.image.restore import (  # noqa: E402
    RestoreAttentionState,
    RestoreSpec,
    distinctive_anchor_merge,
    merge_restore_executor_config,
    restore_attention_bias,
    restore_image_tokens,
)


class _Config:
    def __init__(self, config):
        self.config = config
        self.operation = {}
        self.criterion = {}
        self.schedule = {}


def test_restore_config_is_strict_and_validates_released_defaults() -> None:
    spec = RestoreSpec()
    assert spec.target_count == 64
    assert spec.pruning_ratio == 0.5
    assert spec.anchor_mask_mode == "released"
    assert spec.calibration_constant == 2.0
    with pytest.raises(ValueError, match="greater than one"):
        RestoreSpec(calibration_constant=1.0)
    with pytest.raises(ValueError, match="Unknown RESTORE"):
        merge_restore_executor_config(
            stage=_Config({"retain_rato": 0.5}),
            binding_config=None,
        )


@pytest.mark.parametrize("mask_mode", ("released", "paper"))
def test_distinctive_anchor_merge_partitions_every_residual(mask_mode: str) -> None:
    features = torch.tensor(
        [
            [1.0, 0.0],
            [0.9, 0.1],
            [0.0, 1.0],
            [0.1, 0.9],
            [-1.0, 0.0],
            [-0.9, 0.1],
        ]
    )
    sources = torch.tensor([1, 2, 4, 5, 7, 8])
    result = distinctive_anchor_merge(
        features,
        sources,
        anchor_count=3,
        anchor_mask_mode=mask_mode,
        position_strategy="median",
        source_count=9,
    )
    assert result.tokens.shape == (3, 2)
    assert result.representative_indices.tolist() == sorted(
        result.representative_indices.tolist()
    )
    assert result.token_sizes.sum().item() == 6
    assert result.membership.sum().item() == 6
    assert torch.all(result.membership.sum(dim=0) <= 1)


def test_restore_hybrid_budget_mass_mapping_and_source_positions_are_exact() -> None:
    features = torch.tensor(
        [
            [1.0, 0.0],
            [0.9, 0.1],
            [0.0, 1.0],
            [0.1, 0.9],
            [-1.0, 0.0],
            [-0.9, 0.1],
            [0.5, 0.5],
            [-0.5, 0.5],
        ]
    )
    attention = torch.tensor([8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0])
    output = restore_image_tokens(
        features,
        features,
        attention,
        spec=RestoreSpec(
            target_count=4,
            pruning_ratio=0.5,
            anchor_mask_mode="paper",
        ),
    )
    assert output.tokens.shape == (4, 2)
    assert output.token_sizes.sum().item() == 8
    assert output.membership.sum(dim=0).tolist() == [1] * 8
    assert output.representative_indices.tolist() == sorted(
        output.representative_indices.tolist()
    )
    assert output.layout_delta.source_count == 8
    assert output.layout_delta.result_count == 4
    assert output.layout_delta.dropped_indices == ()
    assert tuple(output.layout_delta.result_layout["representative_indices"]) == tuple(
        output.representative_indices.tolist()
    )
    for source_index in range(8):
        result_index = output.layout_delta.old_to_new[source_index]
        assert result_index is not None
        assert output.membership[result_index, source_index]


def test_restore_attention_bias_matches_equation_four_for_one_dimensional_rope() -> None:
    bias = restore_attention_bias(
        torch.tensor([[0]]),
        torch.tensor([[0, 1]]),
        torch.tensor([[1.0, 2.0]]),
        torch.tensor([1.0]),
        calibration_constant=2.0,
        dtype=torch.float32,
    )
    assert bias.shape == (1, 1, 1, 2)
    assert bias[0, 0, 0, 0].item() == pytest.approx(0.0)
    expected = math.log(2.0) + math.log(2.0 - math.cos(1.0))
    assert bias[0, 0, 0, 1].item() == pytest.approx(expected)


def test_restore_attention_state_preserves_mass_masks_and_decode_positions() -> None:
    positions = torch.tensor([[0, 2, 4, 6]])
    hidden = torch.zeros((1, 4, 2))
    state = RestoreAttentionState.prefill(
        positions,
        image_indices=torch.tensor([1, 2]),
        visual_sizes=torch.tensor([3.0, 1.0]),
        source_visual_count=4,
        first_layer_size="uniform",
        reference=hidden,
    )
    assert state.key_sizes.tolist() == [[1.0, 3.0, 1.0, 1.0]]
    assert state.first_key_sizes.tolist() == [[1.0, 2.0, 2.0, 1.0]]
    # Zero frequency isolates token mass from the separately tested RoPE equation.
    mask = state.attention_mask(
        None,
        positions,
        torch.zeros(1),
        layer_index=0,
        reference=hidden,
        calibration_constant=2.0,
    )
    assert mask[0, 0, 0, 1].item() == torch.finfo(hidden.dtype).min
    torch.testing.assert_close(
        mask[0, 0, -1], torch.tensor([0.0, math.log(2), math.log(2), 0.0])
    )

    decode_positions = torch.tensor([[7]])
    decode_hidden = hidden[:, :1]
    decoded = state.append(decode_positions, reference=decode_hidden)
    assert state.key_positions.tolist() == [[0, 2, 4, 6]]
    assert decoded.key_positions.tolist() == [[0, 2, 4, 6, 7]]
    assert decoded.key_sizes.tolist() == [[1.0, 3.0, 1.0, 1.0, 1.0]]
    assert decoded.first_key_sizes.tolist() == [[1.0, 2.0, 2.0, 1.0, 1.0]]
    padding = torch.tensor([[[[float("-inf"), 0.0, 0.0, 0.0, 0.0]]]])
    mask = decoded.attention_mask(
        padding,
        decode_positions,
        torch.zeros(1),
        layer_index=1,
        reference=decode_hidden,
        calibration_constant=2.0,
    )
    torch.testing.assert_close(
        mask[0, 0, 0], torch.tensor([float("-inf"), math.log(3), 0.0, 0.0, 0.0])
    )

    cluster = RestoreAttentionState.prefill(
        positions,
        image_indices=torch.tensor([1, 2]),
        visual_sizes=torch.tensor([3.0, 1.0]),
        source_visual_count=4,
        first_layer_size="cluster",
        reference=hidden,
    )
    visible = torch.ones((1, 1, 4, 4), dtype=torch.bool).tril()
    mask = cluster.attention_mask(
        visible,
        positions,
        torch.zeros(1),
        layer_index=0,
        reference=hidden,
        calibration_constant=2.0,
    )
    assert mask[0, 0, 0, 1].item() == torch.finfo(hidden.dtype).min
    torch.testing.assert_close(
        mask[0, 0, -1], torch.tensor([0.0, math.log(3), 0.0, 0.0])
    )


def test_restore_attention_bias_weights_native_mrope_axes_by_sections() -> None:
    query = torch.zeros((3, 1, 1), dtype=torch.long)
    keys = torch.tensor(
        [
            [[0, 1]],
            [[0, 2]],
            [[0, 3]],
        ]
    )
    bias = restore_attention_bias(
        query,
        keys,
        torch.tensor([[1.0, 2.0]]),
        torch.tensor([1.0, 2.0, 3.0]),
        mrope_section=(1, 1, 1),
        calibration_constant=2.0,
        dtype=torch.float32,
    )
    decay = (math.cos(1.0) + math.cos(4.0) + math.cos(9.0)) / 3.0
    expected = math.log(2.0) + math.log(2.0 - decay)
    assert bias.shape == (1, 1, 1, 2)
    assert bias[0, 0, 0, 0].item() == pytest.approx(0.0)
    assert bias[0, 0, 0, 1].item() == pytest.approx(expected)
