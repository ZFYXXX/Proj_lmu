"""Pure image-token decisions and their shared sequence materialization."""

from dataclasses import dataclass, field
from typing import Any, Mapping

from mllm_comp.core.layout import LayoutDelta


@dataclass(frozen=True)
class ImageTokenSelection:
    """Keep these image positions in the current serialized sequence.

    Positions may be supplied in any order. The executor retains source order,
    all non-image tokens and protected positions (for example row newlines).
    Methods may attach scores or diagnostics; they need not construct hidden
    states, layout maps or the complementary dropped positions.
    """

    kept_image_indices: Any
    image_scores: Any = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ImageTokenSelectionResult:
    """A selection applied to values and the complete serialized layout."""

    hidden_states: Any
    kept_indices: Any
    dropped_indices: Any
    kept_image_indices: Any
    image_scores: Any
    layout_delta: LayoutDelta
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def from_selection(cls, selection, hidden_states, *, image_indices,
            protected_indices=(), media_unit="llm_token", source_layout=None,
            result_layout=None):
        """Derive the differentiable gather and its layout from one decision."""
        import torch

        if not isinstance(hidden_states, torch.Tensor) or hidden_states.ndim != 2:
            raise ValueError("hidden_states must be a rank-2 torch.Tensor.")
        if not hidden_states.is_floating_point():
            raise TypeError("hidden_states must use a floating dtype.")
        sequence_length = int(hidden_states.shape[0])

        def positions(value, name):
            tensor = torch.as_tensor(value)
            if tensor.ndim != 1 or (tensor.numel() and tensor.dtype not in {
                    torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8}):
                raise ValueError(f"{name} must contain one-dimensional integer positions.")
            indices = tensor.tolist()
            if len(indices) != len(set(indices)) or any(
                    index < 0 or index >= sequence_length for index in indices):
                raise ValueError(f"{name} must contain unique in-range positions.")
            return set(indices)

        images = positions(image_indices, "image_indices")
        selected = positions(selection.kept_image_indices, "kept_image_indices")
        protected = positions(protected_indices, "protected_indices")
        if not selected.issubset(images):
            raise ValueError("kept_image_indices must select only image positions.")
        selected.update(images & protected)
        dropped = tuple(sorted(images - selected))
        non_image = set(range(sequence_length)) - images
        delta = LayoutDelta.drop(sequence_length, dropped, media_unit=media_unit,
            protected_indices=tuple(sorted(non_image | protected)),
            source_layout=source_layout, result_layout=result_layout)
        kept = torch.tensor(delta.kept_indices, dtype=torch.long, device=hidden_states.device)
        kept_images = torch.tensor(sorted(selected), dtype=torch.long, device=hidden_states.device)
        return cls(
            hidden_states=hidden_states.index_select(0, kept), kept_indices=kept,
            dropped_indices=torch.tensor(dropped, dtype=torch.long, device=hidden_states.device),
            kept_image_indices=kept_images, image_scores=selection.image_scores, layout_delta=delta,
            metadata={**selection.metadata,
                "source_tokens": sequence_length, "result_tokens": delta.result_count,
                "source_image_tokens": len(images), "result_image_tokens": len(selected),
                "protected_image_tokens": len(images & protected),
                "selected_indices": delta.kept_indices, "compression_applied": bool(dropped)},
        )


__all__ = ["ImageTokenSelection", "ImageTokenSelectionResult"]
