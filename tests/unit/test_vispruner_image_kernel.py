"""Exact tests for the shared VisPruner visual-cue selector."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from mllm_comp.compressors.image.vispruner import (
    VisPrunerSpec,
    merge_vispruner_executor_config,
    vispruner_image_tokens,
)


torch = pytest.importorskip("torch")


def test_vispruner_matches_released_important_and_iterative_diverse_order() -> None:
    features = torch.tensor(
        [
            [1.0, 0.0],
            [0.99, 0.10],
            [0.0, 1.0],
            [0.0, 0.9],
            [-1.0, 0.0],
            [0.0, -1.0],
        ]
    )
    received = torch.tensor([0.9, 0.7, 0.8, 0.6, 0.5, 0.4])
    attention = received.view(1, 1, 6).expand(2, 6, 6)
    output = torch.arange(18, dtype=torch.float32).reshape(6, 3)

    result = vispruner_image_tokens(
        output,
        features,
        attention,
        spec=VisPrunerSpec(target_count=4, removal_step=1),
    )

    # Attention first chooses 0,2. Two released odd/even pruning rounds leave
    # residual selection order 5,3; output serialization restores source order.
    assert result.selection_order.tolist() == [0, 2, 5, 3]
    assert result.indices.tolist() == [0, 2, 3, 5]
    assert torch.equal(result.tokens, output[[0, 2, 3, 5]])
    assert result.layout_delta.dropped_indices == (1, 4)
    assert result.metadata["important_count"] == 2
    assert result.metadata["diverse_count"] == 2


def test_vispruner_identity_keeps_exact_layout_and_validates_attention() -> None:
    features = torch.eye(3)
    result = vispruner_image_tokens(
        features,
        features,
        torch.eye(3).unsqueeze(0).unsqueeze(0),
        spec=VisPrunerSpec(target_count=8, important_ratio=1.0),
    )
    assert result.indices.tolist() == [0, 1, 2]
    assert result.layout_delta.is_identity
    assert result.metadata["compression_applied"] is False

    with pytest.raises(ValueError, match="token dimensions"):
        vispruner_image_tokens(
            features,
            features,
            torch.eye(2),
            spec=VisPrunerSpec(target_count=2),
        )


def test_vispruner_config_is_strict_before_executor_install() -> None:
    stage = SimpleNamespace(
        operation={},
        criterion={},
        schedule={},
        config={"target_count": 4, "important_ratio": 0.75},
    )
    assert merge_vispruner_executor_config(
        stage=stage,
        binding_config=None,
    ) == {"target_count": 4, "important_ratio": 0.75}

    stage.config = {"important_rato": 0.5}
    with pytest.raises(ValueError, match="important_rato"):
        merge_vispruner_executor_config(stage=stage, binding_config=None)
