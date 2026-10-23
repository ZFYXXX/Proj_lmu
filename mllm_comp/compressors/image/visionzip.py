"""Backbone-neutral VisionZip kernel for image-token compression.

This module implements the selection and merging algorithm described by
VisionZip without knowing where a particular VLM exposes its vision features.
Backbone adapters are responsible for capturing one image sequence, its final
vision attention, and the corresponding key states.  The kernel deliberately
uses a *single-sequence* contract: batching, padding, and media-unit splitting
belong to the adapter because different VLMs serialize those boundaries in
different ways.

The implementation is lazy with respect to PyTorch.  Importing this module does
not import ``torch``; the dependency is needed only when :func:`visionzip_image`
is executed.
"""

from __future__ import annotations

from ._rules import aggregate_rule, label_rule, rule_metadata, score_rule, selection_rule

from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Iterable, Literal, Mapping

from mllm_comp.core.layout import LayoutDelta, LayoutTokenUnit


AttentionMode = Literal["auto", "cls", "mean_received"]


@dataclass(frozen=True)
class VisionZipOutput:
    """Result of compressing one image-token sequence.

    ``indices`` contains one source representative per output token.  For a
    contextual group this is the uniformly selected contextual target, while
    ``source_groups`` records the target and every source token assigned to it.
    The released implementation computes ``target + mean(assigned)`` rather
    than averaging the complete group.  Groups and outputs are ordered by their
    first source index so that spatial serialization order remains stable.
    """

    tokens: Any
    indices: Any
    source_groups: tuple[tuple[int, ...], ...]
    layout_delta: LayoutDelta
    metadata: Mapping[str, Any]

    @property
    def selected_indices(self) -> Any:
        """Alias used by selection-oriented runtime adapters."""

        return self.indices

    @property
    def groups(self) -> tuple[tuple[int, ...], ...]:
        """Short alias for audit and visualization callers."""

        return self.source_groups

    @property
    def delta(self) -> LayoutDelta:
        """Short alias for layout legalization callers."""

        return self.layout_delta


def scaled_visionzip_allocation(
    budget: int,
    *,
    dominant_weight: int = 27,
    contextual_weight: int = 5,
) -> tuple[int, int]:
    """Scale VisionZip's default 27:5 split to an arbitrary output budget.

    Half values are rounded upward, rather than with Python's banker rounding.
    When compression is requested both branches receive at least one token;
    consequently ``budget`` must be at least two.  Protected special tokens are
    excluded before callers invoke this helper.
    """

    _require_positive_integer("budget", budget)
    _require_positive_integer("dominant_weight", dominant_weight)
    _require_positive_integer("contextual_weight", contextual_weight)
    if budget < 2:
        raise ValueError(
            "VisionZip compression needs at least one dominant and one "
            "contextual token"
        )

    total_weight = dominant_weight + contextual_weight
    dominant_count = (budget * dominant_weight + total_weight // 2) // total_weight
    dominant_count = min(budget - 1, max(1, dominant_count))
    return dominant_count, budget - dominant_count


def visionzip_image(
    tokens: Any,
    attention: Any,
    keys: Any,
    *,
    budget: int = 32,
    attention_mode: AttentionMode = "auto",
    cls_index: int | None = None,
    protected_indices: Iterable[int] = (),
    dominant_weight: int = 27,
    contextual_weight: int = 5,
    coordinates: Iterable[Any] = (),
    media_unit: str | LayoutTokenUnit = LayoutTokenUnit.IMAGE_PATCH,
    source_layout: Mapping[str, Any] | None = None,
    result_layout: Mapping[str, Any] | None = None,
    rules: Mapping[str, Any] | None = None,
) -> VisionZipOutput:
    """Compress one image-token sequence using VisionZip.

    Args:
        tokens: Floating tensor with shape ``[tokens, hidden]``.
        attention: Final vision attention with shape ``[tokens, tokens]`` or
            ``[heads, tokens, tokens]``.  A leading singleton batch dimension is
            also accepted.  CLIP-like towers use the CLS query row; SigLIP-like
            towers use mean attention received by every token.
        keys: Key states with shape ``[tokens, key_dim]`` or
            ``[heads, tokens, head_dim]``.  A leading singleton batch dimension
            is also accepted.  Heads are averaged first and the resulting keys
            are L2-normalised before the dot product, matching the authors'
            released cosine-similarity implementation.
        budget: Total output tokens, including protected special tokens.
        attention_mode: ``"cls"`` for CLIP, ``"mean_received"`` for SigLIP,
            or ``"auto"`` to infer the former when ``cls_index`` is provided.
        cls_index: CLS source index.  It is automatically protected when present.
        protected_indices: Other special tokens retained as unmerged singletons.

    Every non-special source token remains represented: dominant tokens are
    copied, contextual targets receive all remaining tokens by maximum key-dot
    similarity.  Each contextual output is the target token plus the mean of
    its assigned non-target tokens, matching the released implementation.
    Selection is discrete, but the resulting values remain differentiable with
    respect to ``tokens``.
    """

    import torch

    if not isinstance(tokens, torch.Tensor):
        raise TypeError("tokens must be a torch.Tensor")
    if tokens.ndim != 2:
        raise ValueError(
            "VisionZip uses a single-sequence contract; tokens must have shape "
            "[tokens, hidden]"
        )
    if not tokens.is_floating_point():
        raise TypeError("tokens must use a floating dtype")

    source_count = int(tokens.shape[0])
    if source_count == 0:
        raise ValueError("VisionZip cannot compress an empty token sequence")
    _require_positive_integer("budget", budget)
    _require_positive_integer("dominant_weight", dominant_weight)
    _require_positive_integer("contextual_weight", contextual_weight)

    protected = _normalise_indices(
        protected_indices,
        source_count=source_count,
        name="protected_indices",
    )
    if cls_index is not None:
        cls_index = _normalise_index(cls_index, source_count, "cls_index")
        protected = tuple(sorted({*protected, cls_index}))

    resolved_mode = _resolve_attention_mode(attention_mode, cls_index)
    importance = _attention_importance(
        attention,
        token_count=source_count,
        mode=resolved_mode,
        cls_index=cls_index,
        torch=torch,
    )
    canonical_keys = _canonical_keys(keys, token_count=source_count, torch=torch)

    effective_budget = min(budget, source_count)
    if effective_budget < len(protected):
        raise ValueError(
            f"budget={budget} cannot retain {len(protected)} protected tokens"
        )

    candidate_indices = tuple(
        index for index in range(source_count) if index not in set(protected)
    )
    compression_requested = effective_budget < source_count
    regular_budget = effective_budget - len(protected)
    if compression_requested:
        if regular_budget < 2:
            raise ValueError(
                "VisionZip compression needs room for at least one dominant and "
                "one contextual token after protected tokens"
            )
        dominant_count, contextual_count = scaled_visionzip_allocation(
            regular_budget,
            dominant_weight=dominant_weight,
            contextual_weight=contextual_weight,
        )
    else:
        dominant_count = len(candidate_indices)
        contextual_count = 0

    if not compression_requested:
        groups = tuple((index,) for index in range(source_count))
        representatives = tuple(range(source_count))
        dominant_ranked = candidate_indices
        contextual_indices: tuple[int, ...] = ()
        output_tokens = tokens
    else:
        request = {"tokens": tokens, "attention": attention, "keys": keys,
            "candidate_indices": candidate_indices, "protected_indices": protected,
            "keep_count": dominant_count, "attention_mode": resolved_mode}
        importance = score_rule(rules, "scorer", request, importance)
        dominant_ranked = _stable_topk_indices(
            importance,
            candidate_indices,
            dominant_count,
            torch=torch,
        )
        custom = selection_rule(rules, "dominant_selector", {**request, "scores": importance},
            count=source_count, keep_count=dominant_count, device=tokens.device)
        if custom is not None:
            if not set(custom.tolist()).issubset(candidate_indices):
                raise ValueError("dominant_selector must select eligible candidates.")
            dominant_ranked = tuple(custom.tolist())
        dominant_set = set(dominant_ranked)
        non_dominant = tuple(
            index for index in candidate_indices if index not in dominant_set
        )
        contextual_indices = _uniform_sample(non_dominant, contextual_count)
        custom = selection_rule(rules, "contextual_selector",
            {**request, "candidate_indices": non_dominant, "dominant_indices": dominant_ranked,
             "keep_count": contextual_count, "scores": importance},
            count=source_count, keep_count=contextual_count, device=tokens.device)
        if custom is not None:
            if not set(custom.tolist()).issubset(non_dominant):
                raise ValueError("contextual_selector must select remaining candidates.")
            contextual_indices = tuple(custom.tolist())
        contextual_set = set(contextual_indices)
        unassigned = tuple(
            index for index in non_dominant if index not in contextual_set
        )
        contextual_groups = _assign_contextual_groups(
            canonical_keys,
            contextual_indices=contextual_indices,
            source_indices=unassigned,
            torch=torch,
            rules=rules,
        )

        records: list[tuple[int, tuple[int, ...]]] = [
            (index, (index,)) for index in (*protected, *dominant_ranked)
        ]
        records.extend(
            (target, tuple(sorted(group)))
            for target, group in zip(contextual_indices, contextual_groups)
        )
        records.sort(key=lambda record: record[1][0])
        representatives = tuple(record[0] for record in records)
        groups = tuple(record[1] for record in records)
        output_tokens = visionzip_merge_source_groups(
            tokens,
            groups,
            representatives=representatives,
            contextual_indices=contextual_indices,
            rules=rules,
        )

    old_to_new: dict[int, int] = {}
    for result_index, group in enumerate(groups):
        for source_index in group:
            old_to_new[source_index] = result_index
    if set(old_to_new) != set(range(source_count)):
        raise RuntimeError("VisionZip internal error: source groups are incomplete")

    coordinates_tuple = tuple(coordinates)
    layout_delta = LayoutDelta.from_mapping(
        source_count=source_count,
        old_to_new=old_to_new,
        result_count=len(groups),
        media_unit=media_unit,
        coordinates=coordinates_tuple,
        protected_indices=protected,
        source_layout=source_layout,
        result_layout=result_layout,
    )
    index_tensor = torch.tensor(
        representatives,
        dtype=torch.long,
        device=tokens.device,
    )
    metadata = MappingProxyType(
        {
            "algorithm": "visionzip",
            "variant": "visionzip.image",
            **rule_metadata(rules),
            "sequence_contract": "single_image_sequence",
            "attention_mode": resolved_mode,
            "score_source": (
                "custom" if rules and "scorer" in rules else
                "cls_attention" if resolved_mode == "cls" else "mean_received_attention"
            ),
            "similarity": "mean_head_l2_normalized_key_dot_product",
            "merge_value": "custom" if rules and "aggregator" in rules else "target_plus_assigned_mean",
            "requested_budget": budget,
            "effective_budget": effective_budget,
            "source_count": source_count,
            "result_count": len(groups),
            "dominant_contextual_weights": (
                dominant_weight,
                contextual_weight,
            ),
            "dominant_count": len(dominant_ranked),
            "contextual_count": len(contextual_indices),
            "protected_count": len(protected),
            "dominant_indices": tuple(dominant_ranked),
            "contextual_indices": tuple(contextual_indices),
            "contextual_target_sampling": "custom" if rules and "contextual_selector" in rules else "released_prefix_stride",
            "protected_indices": protected,
            "token_order": "source_group_first_index",
            "compression_applied": compression_requested,
        }
    )
    return VisionZipOutput(
        tokens=output_tokens,
        indices=index_tensor,
        source_groups=groups,
        layout_delta=layout_delta,
        metadata=metadata,
    )


def _resolve_attention_mode(
    mode: AttentionMode,
    cls_index: int | None,
) -> Literal["cls", "mean_received"]:
    if mode not in {"auto", "cls", "mean_received"}:
        raise ValueError(
            "attention_mode must be 'auto', 'cls', or 'mean_received'"
        )
    if mode == "auto":
        return "cls" if cls_index is not None else "mean_received"
    if mode == "cls" and cls_index is None:
        raise ValueError("attention_mode='cls' requires cls_index")
    return mode


def _attention_importance(
    attention: Any,
    *,
    token_count: int,
    mode: Literal["cls", "mean_received"],
    cls_index: int | None,
    torch: Any,
) -> Any:
    if not isinstance(attention, torch.Tensor):
        raise TypeError("attention must be a torch.Tensor")
    value = attention
    if value.ndim == 1:
        if mode != "mean_received":
            raise ValueError("precomputed attention scores are supported only in mean_received mode")
        if int(value.shape[0]) != token_count:
            raise ValueError("precomputed attention score count must match tokens")
        value = value.detach().float()
        if not bool(torch.isfinite(value).all().item()):
            raise ValueError("attention must contain only finite values")
        return value
    if value.ndim == 4:
        if int(value.shape[0]) != 1:
            raise ValueError(
                "VisionZip's single-sequence contract accepts only a singleton "
                "attention batch dimension"
            )
        value = value[0]
    if value.ndim == 3:
        if tuple(value.shape[-2:]) != (token_count, token_count):
            raise ValueError("attention token dimensions must match tokens")
        value = value.detach().float().mean(dim=0)
    elif value.ndim == 2:
        if tuple(value.shape) != (token_count, token_count):
            raise ValueError("attention token dimensions must match tokens")
        value = value.detach().float()
    else:
        raise ValueError(
            "attention must have shape [tokens], [tokens, tokens], [heads, tokens, tokens], "
            "or [1, heads, tokens, tokens]"
        )

    if mode == "cls":
        assert cls_index is not None
        return value[cls_index]
    # Rows are queries and columns are keys, so averaging over rows measures
    # how much attention each token receives from all other tokens.
    return value.mean(dim=-2)


def _canonical_keys(keys: Any, *, token_count: int, torch: Any) -> Any:
    if not isinstance(keys, torch.Tensor):
        raise TypeError("keys must be a torch.Tensor")
    value = keys
    if value.ndim == 4:
        if int(value.shape[0]) != 1:
            raise ValueError(
                "VisionZip's single-sequence contract accepts only a singleton "
                "key batch dimension"
            )
        value = value[0]
    if value.ndim == 2:
        if int(value.shape[0]) != token_count:
            raise ValueError("keys token dimension must match tokens")
    elif value.ndim == 3:
        if int(value.shape[1]) != token_count:
            raise ValueError("keys token dimension must match tokens")
        value = value.mean(dim=0)
    else:
        raise ValueError(
            "keys must have shape [tokens, key_dim], [heads, tokens, head_dim], "
            "or [1, heads, tokens, head_dim]"
        )
    value = value.detach().float()
    if not bool(torch.isfinite(value).all().item()):
        raise ValueError("keys must contain only finite values")
    return torch.nn.functional.normalize(value, p=2.0, dim=-1)


def _stable_topk_indices(
    importance: Any,
    candidates: tuple[int, ...],
    count: int,
    *,
    torch: Any,
) -> tuple[int, ...]:
    if not 0 < count <= len(candidates):
        raise ValueError("dominant token count is outside the candidate set")
    candidate_tensor = torch.tensor(
        candidates,
        dtype=torch.long,
        device=importance.device,
    )
    scores = importance.index_select(0, candidate_tensor)
    if not bool(torch.isfinite(scores).all().item()):
        raise ValueError("attention contains non-finite candidate scores")
    order = torch.argsort(scores, descending=True, stable=True)
    return tuple(
        int(index)
        for index in candidate_tensor.index_select(0, order[:count]).cpu().tolist()
    )


def _uniform_sample(
    candidates: tuple[int, ...],
    count: int,
) -> tuple[int, ...]:
    if not 0 < count <= len(candidates):
        raise ValueError("contextual token count is outside the non-dominant set")
    length = len(candidates)
    # The released implementation starts at the first remaining token and uses
    # a fixed integer stride rather than bin centres.
    step = max(1, length // count)
    positions = tuple(index * step for index in range(count))
    if len(set(positions)) != count:
        raise RuntimeError("VisionZip internal error: uniform targets are not unique")
    return tuple(candidates[position] for position in positions)


def _assign_contextual_groups(
    keys: Any,
    *,
    contextual_indices: tuple[int, ...],
    source_indices: tuple[int, ...],
    torch: Any,
    rules: Mapping[str, Any] | None = None,
) -> tuple[tuple[int, ...], ...]:
    groups: list[list[int]] = [[index] for index in contextual_indices]
    if not source_indices:
        return tuple(tuple(group) for group in groups)

    source = torch.tensor(source_indices, dtype=torch.long, device=keys.device)
    targets = torch.tensor(
        contextual_indices,
        dtype=torch.long,
        device=keys.device,
    )
    similarity = keys.index_select(0, source) @ keys.index_select(
        0, targets
    ).transpose(0, 1)
    if not bool(torch.isfinite(similarity).all().item()):
        raise ValueError("keys produce non-finite contextual similarities")
    labels = label_rule(rules, "grouper", {"keys": keys, "source_indices": source_indices,
        "contextual_indices": contextual_indices, "similarities": similarity}, torch.argmax(similarity, dim=-1))
    if bool(((labels < 0) | (labels >= len(contextual_indices))).any()):
        raise ValueError("grouper labels must index contextual targets.")
    assignments = labels.cpu().tolist()
    for source_index, group_index in zip(source_indices, assignments):
        groups[int(group_index)].append(source_index)
    return tuple(tuple(sorted(group)) for group in groups)


def visionzip_merge_source_groups(
    tokens: Any,
    groups: tuple[tuple[int, ...], ...],
    *,
    representatives: tuple[int, ...],
    contextual_indices: tuple[int, ...],
    rules: Mapping[str, Any] | None = None,
) -> Any:
    """Apply VisionZip's released merge values to precomputed source groups.

    This shared primitive is also used when a backbone must replay one
    ``LayoutDelta`` on synchronized feature branches.  Selection, layout
    construction, and backbone-specific validation remain outside this helper.
    """

    import torch

    if len(groups) != len(representatives):
        raise RuntimeError("VisionZip internal error: group/representative mismatch")
    contextual = set(contextual_indices)
    values = []
    for representative, group in zip(representatives, groups):
        if representative not in contextual:
            if group != (representative,):
                raise RuntimeError(
                    "VisionZip internal error: non-contextual output is not a singleton"
                )
            values.append(tokens[representative])
            continue
        assigned = tuple(index for index in group if index != representative)
        value = tokens[representative]
        if assigned:
            indices = torch.tensor(
                assigned,
                dtype=torch.long,
                device=tokens.device,
            )
            value = value + tokens.index_select(0, indices).mean(dim=0)
        members = torch.tensor(group, dtype=torch.long, device=tokens.device)
        value = aggregate_rule(rules, "aggregator",
            {"values": tokens.index_select(0, members), "source_indices": group,
             "representative": representative, "assigned_indices": assigned}, value)
        values.append(value)
    return torch.stack(values, dim=0)


def _normalise_indices(
    values: Iterable[int],
    *,
    source_count: int,
    name: str,
) -> tuple[int, ...]:
    if isinstance(values, (str, bytes)):
        raise TypeError(f"{name} must be an iterable of integers")
    result = tuple(
        _normalise_index(index, source_count, name) for index in values
    )
    if len(result) != len(set(result)):
        raise ValueError(f"{name} must not contain duplicates")
    return tuple(sorted(result))


def _normalise_index(value: int, source_count: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} entries must be integers")
    if not 0 <= value < source_count:
        raise ValueError(f"{name} index {value} is outside the token sequence")
    return value


def _require_positive_integer(name: str, value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if value < 1:
        raise ValueError(f"{name} must be positive")


__all__ = [
    "AttentionMode",
    "VisionZipOutput",
    "scaled_visionzip_allocation",
    "visionzip_image",
]
