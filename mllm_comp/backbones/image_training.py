"""Training-time label repair for sequence-mutating image compression.

The image compression bridges operate on batch-one provider paths and may
remove only visual positions.  Those positions must carry the causal-LM ignore
label; silently discarding a supervised text target would corrupt training.
This module centralises that invariant and the standard shifted causal loss so
the three supported backbones use identical training semantics.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from mllm_comp.core.errors import InvariantError


IGNORE_INDEX = -100


def compact_ignored_labels(
    labels: Any,
    kept_indices: Any | Iterable[int],
    *,
    source_length: int,
    operation: str,
) -> Any:
    """Select one batch-one label sequence after proving drops are ignored."""

    import torch

    if not torch.is_tensor(labels) or labels.ndim != 2 or int(labels.shape[0]) != 1:
        raise InvariantError(
            f"{operation} training requires labels with shape [1, tokens]."
        )
    if int(labels.shape[1]) != int(source_length):
        raise InvariantError(
            f"{operation} labels have length {int(labels.shape[1])}, expected "
            f"{int(source_length)}."
        )
    if torch.is_tensor(kept_indices):
        indices = kept_indices.to(device=labels.device, dtype=torch.long)
    else:
        indices = torch.tensor(
            tuple(int(index) for index in kept_indices),
            dtype=torch.long,
            device=labels.device,
        )
    if indices.ndim != 1 or int(indices.numel()) < 1:
        raise InvariantError(f"{operation} must retain at least one label position.")
    if int(indices.min().item()) < 0 or int(indices.max().item()) >= source_length:
        raise InvariantError(f"{operation} kept label index is outside the sequence.")
    if int(torch.unique(indices).numel()) != int(indices.numel()):
        raise InvariantError(f"{operation} kept label indices must be unique.")

    keep_mask = torch.zeros(source_length, dtype=torch.bool, device=labels.device)
    keep_mask[indices] = True
    removed = labels[0, ~keep_mask]
    if int(removed.numel()) and bool((removed != IGNORE_INDEX).any().item()):
        raise InvariantError(
            f"{operation} attempted to remove a supervised text label; only "
            f"labels={IGNORE_INDEX} visual positions may be compressed."
        )
    return labels.index_select(1, indices)


def repair_labels_for_layout_delta(
    labels: Any,
    delta: Any,
    *,
    operation: str,
) -> Any:
    """Compact labels with one representative for every result layout token."""

    from mllm_comp.compressors.image.decoder_lifecycle import (
        layout_result_representatives,
    )
    from mllm_comp.core.layout import LayoutDelta

    if not isinstance(delta, LayoutDelta):
        raise InvariantError(f"{operation} did not produce a LayoutDelta.")
    return compact_ignored_labels(
        labels,
        layout_result_representatives(delta),
        source_length=delta.source_count,
        operation=operation,
    )


def attach_shifted_causal_loss(output: Any, labels: Any, *, operation: str) -> Any:
    """Attach the standard next-token cross-entropy loss to a model output."""

    logits = getattr(output, "logits", None)
    if logits is None and isinstance(output, dict):
        logits = output.get("logits")
    if logits is None or getattr(logits, "ndim", None) != 3:
        raise InvariantError(
            f"{operation} training requires a return-dict output with rank-3 logits."
        )
    if getattr(labels, "ndim", None) != 2:
        raise InvariantError(f"{operation} training labels must be rank 2.")

    import torch
    import torch.nn.functional as functional

    if not torch.is_tensor(logits):
        raise InvariantError(
            f"{operation} training requires logits to be a tensor."
        )
    if not torch.is_tensor(labels):
        raise InvariantError(f"{operation} training labels must be a tensor.")
    if tuple(labels.shape) != tuple(logits.shape[:2]):
        raise InvariantError(
            f"{operation} repaired labels {tuple(labels.shape)} do not match logits "
            f"{tuple(logits.shape[:2])}."
        )
    if int(logits.shape[1]) < 2:
        raise InvariantError(f"{operation} training requires at least two tokens.")

    shift_logits = logits[:, :-1, :].contiguous().float()
    shift_labels = labels[:, 1:].contiguous().to(logits.device)
    if not bool((shift_labels != IGNORE_INDEX).any().item()):
        raise InvariantError(
            f"{operation} training batch has no supervised next-token target."
        )
    loss = functional.cross_entropy(
        shift_logits.view(-1, int(shift_logits.shape[-1])),
        shift_labels.view(-1),
        ignore_index=IGNORE_INDEX,
    )
    if isinstance(output, dict):
        output["loss"] = loss
    try:
        output.loss = loss
    except (AttributeError, TypeError) as exc:
        raise InvariantError(
            f"{operation} output does not expose a mutable loss field."
        ) from exc
    return output


__all__ = [
    "IGNORE_INDEX",
    "attach_shifted_causal_loss",
    "compact_ignored_labels",
    "repair_labels_for_layout_delta",
]
