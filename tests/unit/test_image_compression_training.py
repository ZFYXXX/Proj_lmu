"""Unit tests for sequence-mutating image-compression training invariants."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from mllm_comp.backbones.image_training import (
    attach_shifted_causal_loss,
    compact_ignored_labels,
)
from mllm_comp.core.errors import InvariantError

torch = pytest.importorskip("torch")


def test_compact_ignored_labels_preserves_text_targets_in_order() -> None:
    labels = torch.tensor([[-100, -100, -100, -100, 17]])

    compact = compact_ignored_labels(
        labels,
        torch.tensor([0, 2, 4]),
        source_length=5,
        operation="fixture",
    )

    assert compact.tolist() == [[-100, -100, 17]]


def test_compact_ignored_labels_rejects_supervised_token_removal() -> None:
    labels = torch.tensor([[-100, 11, -100]])

    with pytest.raises(InvariantError, match="supervised text label"):
        compact_ignored_labels(
            labels,
            (0, 2),
            source_length=3,
            operation="fixture",
        )


def test_attach_shifted_causal_loss_is_differentiable() -> None:
    logits = torch.randn(1, 4, 32, requires_grad=True)
    output = SimpleNamespace(logits=logits, loss=None)
    labels = torch.tensor([[-100, -100, -100, 7]])

    repaired = attach_shifted_causal_loss(output, labels, operation="fixture")

    assert repaired is output
    assert torch.isfinite(output.loss)
    output.loss.backward()
    assert logits.grad is not None
