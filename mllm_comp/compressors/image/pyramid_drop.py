"""Backbone-neutral PyramidDrop ranking kernel for ImageLLM tokens.

The paper reuses the query of the last instruction token and the keys of image
tokens at the end of each LLM stage.  This module implements that mathematical
decision only; backbone bindings remain responsible for obtaining Q/K and for
repairing masks, positions, caches, and any side branches.
"""

from __future__ import annotations

from ._rules import rule_metadata, score_rule, selection_rule

import math
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping

from mllm_comp.core.layout import LayoutDelta


@dataclass(frozen=True)
class PyramidDropSpec:
    """Paper defaults and safety bounds for one rank-and-drop decision."""

    keep_ratio: float = 0.5
    minimum_keep_tokens: int = 1
    num_stages: int = 4
    restore_source_order: bool = True

    def __post_init__(self) -> None:
        if not 0.0 < float(self.keep_ratio) <= 1.0:
            raise ValueError("keep_ratio must be in (0, 1].")
        if isinstance(self.minimum_keep_tokens, bool) or not isinstance(
            self.minimum_keep_tokens, int
        ):
            raise TypeError("minimum_keep_tokens must be an integer.")
        if self.minimum_keep_tokens < 0:
            raise ValueError("minimum_keep_tokens must be non-negative.")
        if isinstance(self.num_stages, bool) or not isinstance(self.num_stages, int):
            raise TypeError("num_stages must be an integer.")
        if self.num_stages < 2:
            raise ValueError("num_stages must be at least two.")


@dataclass(frozen=True)
class PyramidDropSchedule:
    """Stage boundaries, expressed as zero-based completed decoder layers."""

    layer_indices: tuple[int, ...]
    keep_ratio: float
    num_layers: int
    num_stages: int

    @classmethod
    def uniform(
        cls,
        num_layers: int,
        *,
        num_stages: int = 4,
        keep_ratio: float = 0.5,
    ) -> "PyramidDropSchedule":
        if isinstance(num_layers, bool) or not isinstance(num_layers, int):
            raise TypeError("num_layers must be an integer.")
        if num_layers < num_stages:
            raise ValueError("num_layers must be at least num_stages.")
        spec = PyramidDropSpec(keep_ratio=keep_ratio, num_stages=num_stages)
        boundaries = tuple(
            (num_layers * stage_index) // num_stages - 1
            for stage_index in range(1, num_stages)
        )
        if len(boundaries) != len(set(boundaries)) or boundaries[0] < 0:
            raise ValueError("uniform stage boundaries are not distinct.")
        return cls(
            layer_indices=boundaries,
            keep_ratio=spec.keep_ratio,
            num_layers=num_layers,
            num_stages=num_stages,
        )

    def validate_after_layer(self, minimum_completed_layer: int) -> None:
        """Reject schedules that drop before a required side branch is consumed."""

        if self.layer_indices[0] <= minimum_completed_layer:
            raise ValueError(
                "first PyramidDrop boundary must be after required layer "
                f"{minimum_completed_layer}; got {self.layer_indices[0]}."
            )


@dataclass
class PyramidDropResult:
    """One stage result plus a layout delta suitable for legalization."""

    hidden_states: Any
    kept_indices: Any
    dropped_indices: Any
    kept_image_indices: Any
    image_scores: Any
    layout_delta: LayoutDelta
    metadata: dict[str, Any] = field(default_factory=dict)


def pyramid_instruction_indices(
    image_mask: Any,
    *,
    attention_mask: Any | None = None,
    labels: Any | None = None,
    ignore_index: int = -100,
) -> Any:
    """Locate the paper's instruction query positions for one prompt.

    During training, every first supervised answer token contributes the valid
    non-image token immediately before it.  During evaluation (or when no
    supervised boundary exists), the final valid non-image prompt token is
    used.  The returned rank-one tensor is always in source order.
    """

    torch = _torch()
    if not isinstance(image_mask, torch.Tensor) or image_mask.ndim != 1:
        raise ValueError("image_mask must be a rank-one torch tensor.")
    image_mask = image_mask.bool()
    token_count = int(image_mask.numel())
    valid = torch.ones(token_count, dtype=torch.bool, device=image_mask.device)
    if attention_mask is not None:
        if not isinstance(attention_mask, torch.Tensor):
            raise TypeError("attention_mask must be a torch tensor.")
        padding = attention_mask
        if padding.ndim == 2:
            if int(padding.shape[0]) != 1:
                raise ValueError("attention_mask must have one batch row.")
            padding = padding[0]
        if padding.ndim != 1 or int(padding.numel()) != token_count:
            raise ValueError("attention_mask does not match image_mask.")
        valid = padding.to(device=image_mask.device).bool()

    if labels is not None:
        if not isinstance(labels, torch.Tensor):
            raise TypeError("labels must be a torch tensor.")
        label_row = labels
        if label_row.ndim == 2:
            if int(label_row.shape[0]) != 1:
                raise ValueError("labels must have one batch row.")
            label_row = label_row[0]
        if label_row.ndim != 1 or int(label_row.numel()) != token_count:
            raise ValueError("labels do not match image_mask.")
        supervised = label_row.to(image_mask.device).ne(ignore_index) & valid
        previous = torch.zeros_like(supervised)
        previous[1:] = supervised[:-1]
        starts = (supervised & ~previous).nonzero(as_tuple=False).flatten()
        candidates = starts - 1
        candidates = candidates[candidates >= 0]
        if int(candidates.numel()):
            candidates = candidates[
                valid.index_select(0, candidates)
                & ~image_mask.index_select(0, candidates)
            ]
        if int(candidates.numel()):
            return candidates

    candidates = (valid & ~image_mask).nonzero(as_tuple=False).flatten()
    if int(candidates.numel()) == 0:
        raise ValueError("PyramidDrop requires a valid non-image instruction token.")
    return candidates[-1:]


def pyramid_attention_key_visibility(
    attention_mask: Any | None,
    *,
    instruction_indices: Any,
    key_count: int | None = None,
    padding_mask: Any | None = None,
) -> Any | None:
    """Return visible keys for each selected instruction query."""

    torch = _torch()
    if attention_mask is not None and not isinstance(attention_mask, torch.Tensor):
        raise TypeError("attention_mask must be a torch tensor.")
    device = (
        attention_mask.device
        if attention_mask is not None
        else getattr(instruction_indices, "device", None)
    )
    indices = _index_tensor(
        instruction_indices,
        device=device,
        name="instruction_indices",
    )
    if attention_mask is None or attention_mask.ndim == 2:
        if attention_mask is not None:
            if int(attention_mask.shape[0]) != 1:
                raise ValueError("rank-2 attention_mask must have one batch row.")
            inferred_count = int(attention_mask.shape[-1])
            visible_padding = attention_mask[0].bool()
        else:
            if key_count is None:
                return None
            inferred_count = key_count
            visible_padding = torch.ones(
                inferred_count,
                dtype=torch.bool,
                device=indices.device,
            )
        if key_count is not None and key_count != inferred_count:
            raise ValueError("key_count does not match attention_mask.")
        if padding_mask is not None:
            padding = torch.as_tensor(padding_mask, device=indices.device)
            if padding.ndim == 2 and int(padding.shape[0]) == 1:
                padding = padding[0]
            if padding.ndim != 1 or int(padding.numel()) != inferred_count:
                raise ValueError("padding_mask does not match key_count.")
            visible_padding = visible_padding & padding.bool()
        keys = torch.arange(inferred_count, device=indices.device)
        return (keys.unsqueeze(0) <= indices.unsqueeze(1)) & visible_padding
    if attention_mask.ndim != 4 or int(attention_mask.shape[0]) != 1:
        raise ValueError("attention_mask must have shape [1, tokens] or [1, 1, Q, K].")
    if int(attention_mask.shape[1]) != 1:
        raise ValueError("rank-4 attention_mask must have one attention-mask head.")
    rows = attention_mask[0, 0].index_select(0, indices)
    if rows.dtype == torch.bool:
        return rows
    if not rows.is_floating_point():
        return rows.bool()
    return torch.isfinite(rows) & (rows > torch.finfo(rows.dtype).min / 2)


def pyramid_drop(
    hidden_states: Any,
    *,
    image_indices: Any,
    instruction_query: Any = None,
    image_keys: Any = None,
    attention_keys: Any | None = None,
    attention_key_mask: Any | None = None,
    image_attention_scores: Any | None = None,
    spec: PyramidDropSpec | None = None,
    protected_indices: Iterable[int] = (),
    media_unit: str = "llm_token",
    source_layout: Mapping[str, Any] | None = None,
    result_layout: Mapping[str, Any] | None = None,
    rules: Mapping[str, Any] | None = None,
) -> PyramidDropResult:
    """Rank image tokens with last-instruction Q × image K and drop the tail.

    ``hidden_states`` is a single serialized sequence ``[tokens, channels]``.
    ``image_indices`` indexes its image-token positions.  ``instruction_query``
    may be ``[head_dim]``, ``[heads, head_dim]``, or
    ``[instruction_queries, heads, head_dim]``.  The last form matches the
    released training path, which averages scores from every answer-prefix
    instruction boundary.  ``image_keys`` accepts ``[image_tokens, head_dim]``,
    ``[image_tokens, heads, head_dim]``, or ``[heads, image_tokens, head_dim]``.

    Paper-faithful providers also pass ``attention_keys`` for the complete
    serialized sequence. Scores are then normalized over every visible key
    independently per attention head before image positions are selected and
    heads are averaged, matching the authors' released implementation.

    All non-image tokens are preserved.  Ranking is deterministic for tied
    scores and selected image tokens are restored to source order before the
    compact sequence is returned.
    """

    torch = _torch()
    active_spec = spec or PyramidDropSpec()
    if not isinstance(hidden_states, torch.Tensor) or hidden_states.ndim != 2:
        raise ValueError("hidden_states must be a rank-2 torch.Tensor.")
    device = hidden_states.device
    image_indices = _index_tensor(image_indices, device=device, name="image_indices")
    sequence_length = int(hidden_states.shape[0])
    _validate_unique_range(image_indices, sequence_length, "image_indices")

    protected = tuple(sorted(set(int(index) for index in protected_indices)))
    if any(index < 0 or index >= sequence_length for index in protected):
        raise ValueError("protected_indices contains an index outside the sequence.")

    image_count = int(image_indices.numel())
    if image_count == 0:
        delta = LayoutDelta.identity(
            sequence_length,
            media_unit=media_unit,
            protected_indices=tuple(range(sequence_length)),
            layout=source_layout or {},
        )
        empty_scores = hidden_states.new_empty((0,), dtype=torch.float32)
        return PyramidDropResult(
            hidden_states=hidden_states,
            kept_indices=torch.arange(sequence_length, device=device),
            dropped_indices=torch.empty(0, dtype=torch.long, device=device),
            kept_image_indices=image_indices,
            image_scores=empty_scores,
            layout_delta=delta,
            metadata={
                "algorithm": "pyramid_drop",
                "source_image_tokens": 0,
                "result_image_tokens": 0,
                "keep_ratio": active_spec.keep_ratio,
            },
        )

    if image_attention_scores is None:
        scores = pyramid_attention_scores(
            rules=rules,
            instruction_query=instruction_query,
            image_keys=image_keys,
            image_count=image_count,
            device=device,
            attention_keys=attention_keys,
            attention_key_mask=attention_key_mask,
            image_key_indices=(image_indices if attention_keys is not None else None),
        )
    else:
        if (not isinstance(image_attention_scores, torch.Tensor)
                or image_attention_scores.ndim != 1 or image_attention_scores.numel() != image_count
                or not image_attention_scores.is_floating_point() or not torch.isfinite(image_attention_scores).all()):
            raise ValueError("PyramidDrop image_attention_scores must match the current image span.")
        scores = image_attention_scores.to(device=device, dtype=torch.float32)

    protected_set = set(protected)
    protected_image_positions = torch.tensor(
        [
            position
            for position, sequence_index in enumerate(image_indices.tolist())
            if sequence_index in protected_set
        ],
        dtype=torch.long,
        device=device,
    )
    # The paper removes ceil((1-lambda)*V), hence retains floor(lambda*V).
    paper_keep = image_count - math.ceil((1.0 - active_spec.keep_ratio) * image_count)
    keep_count = min(
        image_count,
        max(
            active_spec.minimum_keep_tokens,
            paper_keep,
            int(protected_image_positions.numel()),
        ),
    )

    request = {"hidden_states": hidden_states, "image_indices": image_indices,
        "instruction_query": instruction_query, "image_keys": image_keys,
        "attention_keys": attention_keys, "attention_key_mask": attention_key_mask,
        "image_attention_scores": image_attention_scores, "keep_count": keep_count,
        "protected_indices": tuple(protected_image_positions.tolist()), "spec": active_spec}
    if keep_count < image_count:
        scores = score_rule(rules, "scorer", request, scores)
    ranked_positions = _stable_descending_indices(scores)
    if protected_image_positions.numel():
        protected_position_set = set(protected_image_positions.tolist())
        ranked_unprotected = torch.tensor(
            [
                value
                for value in ranked_positions.tolist()
                if value not in protected_position_set
            ],
            dtype=torch.long,
            device=device,
        )
        chosen_positions = torch.cat(
            (
                protected_image_positions,
                ranked_unprotected[: keep_count - int(protected_image_positions.numel())],
            )
        )
    else:
        chosen_positions = ranked_positions[:keep_count]
    if keep_count < image_count:
        custom = selection_rule(rules, "selector", {**request, "scores": scores},
            count=image_count, keep_count=keep_count, device=device, protected=protected_image_positions)
        if custom is not None:
            chosen_positions = custom
    if active_spec.restore_source_order:
        chosen_positions = chosen_positions.sort().values
    kept_image_indices = image_indices.index_select(0, chosen_positions)

    image_index_set = set(image_indices.tolist())
    kept_image_set = set(kept_image_indices.tolist())
    dropped = tuple(
        index
        for index in range(sequence_length)
        if index in image_index_set and index not in kept_image_set
    )
    kept = torch.tensor(
        [index for index in range(sequence_length) if index not in set(dropped)],
        dtype=torch.long,
        device=device,
    )
    dropped_tensor = torch.tensor(dropped, dtype=torch.long, device=device)
    non_image = tuple(
        index for index in range(sequence_length) if index not in image_index_set
    )
    layout_protected = tuple(sorted(set(non_image) | protected_set))
    delta = LayoutDelta.drop(
        sequence_length,
        dropped,
        media_unit=media_unit,
        protected_indices=layout_protected,
        source_layout=source_layout,
        result_layout=result_layout,
    )
    output = hidden_states.index_select(0, kept)
    return PyramidDropResult(
        hidden_states=output,
        kept_indices=kept,
        dropped_indices=dropped_tensor,
        kept_image_indices=kept_image_indices,
        image_scores=scores,
        layout_delta=delta,
        metadata={
            "algorithm": "pyramid_drop",
            "ranking": "custom" if rules else "last_instruction_received_attention_probability",
            **rule_metadata(rules),
            "source_tokens": sequence_length,
            "result_tokens": int(output.shape[0]),
            "source_image_tokens": image_count,
            "result_image_tokens": int(kept_image_indices.numel()),
            "keep_ratio": active_spec.keep_ratio,
            "minimum_keep_tokens": active_spec.minimum_keep_tokens,
            "protected_image_tokens": int(protected_image_positions.numel()),
            "restore_source_order": active_spec.restore_source_order,
        },
    )


def pyramid_attention_scores(
    *,
    instruction_query: Any,
    image_keys: Any,
    image_count: int | None = None,
    device: Any | None = None,
    attention_keys: Any | None = None,
    attention_key_mask: Any | None = None,
    image_key_indices: Any | None = None,
    rules: Mapping[str, Any] | None = None,
) -> Any:
    """Compute PyramidDrop's head-averaged attention probabilities.

    The released code applies scaled QK, a causal/padding mask, and a float32
    softmax over the complete key sequence for every head. It then selects
    image positions and averages heads. ``attention_keys`` and
    ``image_key_indices`` expose that exact normalization to provider bridges;
    callers with only image keys retain a self-contained image-stream path.
    """

    torch = _torch()
    if not isinstance(instruction_query, torch.Tensor) or not isinstance(
        image_keys, torch.Tensor
    ):
        raise TypeError("instruction_query and image_keys must be torch tensors.")
    query = instruction_query.to(device=device)
    image_keys = _key_tensor(
        image_keys,
        image_count=image_count,
        device=device,
        name="image_keys",
    )
    keys = (
        image_keys
        if attention_keys is None
        else _key_tensor(
            attention_keys,
            image_count=None,
            device=device,
            name="attention_keys",
        )
    )
    if query.ndim == 1:
        query = query.reshape(1, 1, -1)
    elif query.ndim == 2:
        query = query.unsqueeze(0)
    elif query.ndim != 3:
        raise ValueError(
            "instruction_query must have shape [D], [H, D], or [Q, H, D]."
        )
    if int(query.shape[0]) == 0 or int(query.shape[1]) == 0:
        raise ValueError("instruction_query must contain at least one query and head.")
    if keys.shape[-1] != query.shape[-1]:
        raise ValueError("instruction query and image key head dimensions differ.")
    if query.shape[1] == 1 and keys.shape[1] > 1:
        query = query.expand(-1, keys.shape[1], -1)
    if keys.shape[1] == 1 and query.shape[1] > 1:
        keys = keys.expand(-1, query.shape[1], -1)
    if query.shape[1] != keys.shape[1]:
        raise ValueError("instruction query and image keys have incompatible heads.")
    scale = float(query.shape[-1]) ** -0.5
    logits = torch.einsum("qhd,thd->tqh", query, keys) * scale
    if attention_key_mask is not None:
        mask = torch.as_tensor(attention_key_mask, device=keys.device).bool()
        if mask.ndim == 1:
            if int(mask.numel()) != int(keys.shape[0]):
                raise ValueError("attention_key_mask does not match attention_keys.")
            expanded_mask = mask[:, None, None]
        elif mask.ndim == 2:
            expected = (int(query.shape[0]), int(keys.shape[0]))
            if tuple(int(item) for item in mask.shape) != expected:
                raise ValueError(
                    "rank-2 attention_key_mask must have shape "
                    "[instruction_queries, attention_keys]."
                )
            expanded_mask = mask.transpose(0, 1).unsqueeze(-1)
        else:
            raise ValueError("attention_key_mask must be rank one or rank two.")
        if not bool(mask.any().item()):
            raise ValueError("attention_key_mask must expose at least one key.")
        if bool((~expanded_mask.any(dim=0)).any().item()):
            raise ValueError(
                "attention_key_mask must expose at least one key per query."
            )
        logits = logits.masked_fill(~expanded_mask, float("-inf"))
    probabilities = torch.softmax(logits, dim=0, dtype=torch.float32).to(
        dtype=logits.dtype
    )
    if attention_keys is None:
        selected = probabilities
    else:
        if image_key_indices is None:
            raise ValueError(
                "image_key_indices is required when attention_keys is provided."
            )
        indices = _index_tensor(
            image_key_indices,
            device=keys.device,
            name="image_key_indices",
        )
        _validate_unique_range(indices, int(keys.shape[0]), "image_key_indices")
        if image_count is not None and int(indices.numel()) != image_count:
            raise ValueError("image_key_indices does not match image token count.")
        selected = probabilities.index_select(0, indices)
        if (
            int(selected.shape[0]) != int(image_keys.shape[0])
            or int(selected.shape[-1]) != int(image_keys.shape[1])
        ):
            raise ValueError(
                "selected attention keys do not match the provided image_keys shape."
            )
    source_rule = "stage2_source_scorer" if rules and "stage2_source_scorer" in rules else "source_scorer"
    return score_rule(rules, source_rule,
        {"instruction_query": instruction_query, "image_keys": image_keys,
         "attention_keys": attention_keys, "attention_key_mask": attention_key_mask,
         "image_key_indices": image_key_indices, "logits": logits}, selected.mean(dim=(-1, -2)))


def _key_tensor(
    value: Any,
    *,
    image_count: int | None,
    device: Any | None,
    name: str,
) -> Any:
    torch = _torch()
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a torch tensor.")
    keys = value.to(device=device)
    if keys.ndim == 2:
        keys = keys.unsqueeze(1)
    elif keys.ndim == 3:
        if image_count is not None and keys.shape[0] != image_count:
            if keys.shape[1] == image_count:
                keys = keys.transpose(0, 1)
            else:
                raise ValueError(f"{name} does not match image token count.")
    else:
        raise ValueError(
            f"{name} must have shape [V,D], [V,H,D], or [H,V,D]."
        )
    if image_count is not None and keys.shape[0] != image_count:
        raise ValueError(f"{name} does not match image token count.")
    return keys


def _stable_descending_indices(scores: Any) -> Any:
    torch = _torch()
    return torch.argsort(scores, descending=True, stable=True)


def _index_tensor(value: Any, *, device: Any, name: str) -> Any:
    torch = _torch()
    if isinstance(value, torch.Tensor):
        result = value.to(device=device, dtype=torch.long)
    else:
        result = torch.tensor(tuple(value), dtype=torch.long, device=device)
    if result.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional.")
    return result


def _validate_unique_range(value: Any, upper: int, name: str) -> None:
    items = value.tolist()
    if len(items) != len(set(items)):
        raise ValueError(f"{name} must not contain duplicates.")
    if any(index < 0 or index >= upper for index in items):
        raise ValueError(f"{name} contains an index outside the sequence.")
    if items != sorted(items):
        raise ValueError(f"{name} must be in source order.")


def _torch() -> Any:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - exercised in minimal installs
        raise RuntimeError("PyramidDrop execution requires PyTorch.") from exc
    return torch


__all__ = [
    "PyramidDropResult",
    "PyramidDropSchedule",
    "PyramidDropSpec",
    "pyramid_attention_key_visibility",
    "pyramid_attention_scores",
    "pyramid_drop",
    "pyramid_instruction_indices",
]
