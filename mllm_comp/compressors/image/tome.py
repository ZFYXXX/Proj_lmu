"""Token Merging (ToMe) reference kernel for image encoders.

This module implements the backbone-neutral operation described by ToMe.  It
uses the mean key across attention heads as the matching metric, L2-normalises
that metric, and performs one alternating A/B bipartite soft-matching step.
The requested ``r`` is an *absolute number of tokens removed in this layer*,
not a keep ratio.

For a non-identity merge, the paper implementation's ordering is retained:
unmerged source-A tokens first, followed by every destination-B token.  Both
parts retain ascending indices from the input sequence.  Consequently this is
an explicit ordering rather than a source-order compaction; every result
contains a :class:`~mllm_comp.core.layout.LayoutDelta` and source groups so a
backbone adapter can legalise coordinates, masks, and positions correctly.

PyTorch is imported lazily inside :func:`tome_merge_image`.  Importing this
module therefore remains possible in the platform's dependency-light control
plane.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Iterable, Mapping

from mllm_comp.core.layout import LayoutDelta, LayoutTokenUnit, OrderingPolicy
from ._rules import score_rule, aggregate_rule, rule_metadata


_PAPER_ORDERING = "source_a_unmerged_then_destination_b"
_IDENTITY_ORDERING = "source_order_identity"


@dataclass(frozen=True)
class ToMeImageResult:
    """Output of one ToMe image-encoder layer.

    ``hidden_states`` mirrors the input rank: ``[T, C]`` for one sequence or
    ``[B, T, C]`` for a batch.  ``token_sizes`` has a trailing singleton
    dimension (``[T, 1]`` or ``[B, T, 1]``).  The proportional-attention bias
    is already broadcastable to attention logits: ``[1, 1, T]`` for one
    sequence and ``[B, 1, 1, T]`` for a batch.

    Layout information is always batch-normalised.  ``layout_deltas[i]`` and
    ``source_groups[i]`` describe sequence ``i``; this remains true for a
    single-sequence input, where both tuples have length one.
    """

    hidden_states: Any
    token_sizes: Any
    proportional_attention_bias: Any
    layout_deltas: tuple[LayoutDelta, ...]
    source_groups: tuple[tuple[tuple[int, ...], ...], ...]
    metadata: Mapping[str, Any]
    is_batched: bool
    merge_pairs: tuple[tuple[tuple[int, int], ...], ...] = ()
    input_token_sizes: Any = None

    @property
    def layout_delta(self) -> LayoutDelta:
        """Return the sole layout delta for a single-sequence result."""

        if self.is_batched or len(self.layout_deltas) != 1:
            raise ValueError(
                "layout_delta is only available for a single-sequence input; "
                "use layout_deltas for batched inputs"
            )
        return self.layout_deltas[0]

    @property
    def sequence_source_groups(self) -> tuple[tuple[int, ...], ...]:
        """Return source groups for a single-sequence result."""

        if self.is_batched or len(self.source_groups) != 1:
            raise ValueError(
                "sequence_source_groups is only available for a "
                "single-sequence input; use source_groups for batched inputs"
            )
        return self.source_groups[0]


# Short compatibility name for callers that do not need to encode modality in
# the result type.  The concrete type remains image-specific by design.
ToMeResult = ToMeImageResult


def tome_preserve_source_order(result: ToMeImageResult) -> ToMeImageResult:
    """Legalise one paper-ordered result for an order-sensitive LLM prompt.

    ToMe's native A/B output order is correct inside a bidirectional vision
    encoder.  A causal LLM prompt and Qwen's placeholder scatter are ordered
    by source position, however.  This adapter-only legalizer reorders whole
    merge groups by their first source index without changing group values,
    masses, gradients, or membership, and publishes the corresponding remap.
    """

    import torch

    if not isinstance(result, ToMeImageResult):
        raise TypeError("result must be a ToMeImageResult")
    if result.is_batched:
        raise ValueError(
            "source-order legalization currently requires one image sequence"
        )
    groups = result.sequence_source_groups
    order = tuple(sorted(range(len(groups)), key=lambda index: groups[index][0]))
    index = torch.tensor(
        order,
        dtype=torch.long,
        device=result.hidden_states.device,
    )
    reordered_groups = tuple(groups[item] for item in order)
    mapping = {
        source_index: result_index
        for result_index, group in enumerate(reordered_groups)
        for source_index in group
    }
    source_delta = result.layout_delta
    source_order_delta = LayoutDelta.from_mapping(
        source_count=source_delta.source_count,
        old_to_new=mapping,
        result_count=source_delta.result_count,
        media_unit=source_delta.media_unit,
        coordinates=source_delta.coordinates,
        protected_indices=source_delta.protected_indices,
        source_layout=source_delta.source_layout,
        result_layout={
            **dict(source_delta.result_layout),
            "upstream_ordering": source_delta.result_layout.get("ordering"),
            "ordering": "source_group_first_index",
            "source_groups": reordered_groups,
            "representative_indices": tuple(group[0] for group in reordered_groups),
            "representative_policy": "first_source_in_group",
            "result_coordinates": (
                tuple(source_delta.coordinates[group[0]] for group in reordered_groups)
                if source_delta.coordinates
                else ()
            ),
        },
        ordering_policy=OrderingPolicy.PRESERVE_SOURCE,
    )
    return ToMeImageResult(
        hidden_states=result.hidden_states.index_select(0, index),
        token_sizes=result.token_sizes.index_select(0, index),
        proportional_attention_bias=result.proportional_attention_bias.index_select(
            -1, index
        ),
        layout_deltas=(source_order_delta,),
        source_groups=(reordered_groups,),
        metadata=MappingProxyType(
            {
                **dict(result.metadata),
                "upstream_ordering": result.metadata.get("ordering"),
                "ordering": "source_group_first_index",
                "output_legalizer": "causal_llm_source_order",
            }
        ),
        is_batched=False,
        merge_pairs=result.merge_pairs, input_token_sizes=result.input_token_sizes,
    )


def tome_restore_dense_boundary(
    hidden_states: Any,
    layout_delta: LayoutDelta | Iterable[LayoutDelta],
) -> Any:
    """Broadcast compact ToMe features back to the source token layout.

    The first ImageLLM support slice deliberately restores a dense sequence
    before backbone-specific AnyRes packing, spatial mergers, or DeepStack
    consumers.  This helper performs that boundary operation without trying
    to invert a merge: every source token receives the compact feature that
    represents its ``old_to_new`` group.

    ``hidden_states`` accepts ``[T', C]`` with one :class:`LayoutDelta` or
    ``[B, T', C]`` with one delta per batch row.  Drop deltas are rejected
    because a dense boundary cannot synthesize a source token that has no
    surviving representative.
    """

    import torch

    if not torch.is_tensor(hidden_states):
        raise TypeError("hidden_states must be a torch.Tensor")
    if hidden_states.ndim not in (2, 3):
        raise ValueError("hidden_states must have shape [T, C] or [B, T, C]")

    if isinstance(layout_delta, LayoutDelta):
        deltas = (layout_delta,)
    else:
        deltas = tuple(layout_delta)
    expected_batch = 1 if hidden_states.ndim == 2 else int(hidden_states.shape[0])
    if len(deltas) != expected_batch:
        raise ValueError("dense restore requires exactly one LayoutDelta per sequence")
    if expected_batch == 0:
        raise ValueError("dense restore requires at least one sequence")

    compact_count = int(hidden_states.shape[-2])
    source_counts = {delta.source_count for delta in deltas}
    if len(source_counts) != 1:
        raise ValueError("batched dense restore requires a uniform source length")
    for delta in deltas:
        if delta.result_count != compact_count:
            raise ValueError(
                "LayoutDelta result_count does not match compact hidden states"
            )
        if delta.dropped_indices or any(
            delta.old_to_new[index] is None for index in range(delta.source_count)
        ):
            raise ValueError("dense restore requires a merge-only LayoutDelta")

    gather_rows = torch.tensor(
        [
            [int(delta.old_to_new[index]) for index in range(delta.source_count)]
            for delta in deltas
        ],
        dtype=torch.long,
        device=hidden_states.device,
    )
    if hidden_states.ndim == 2:
        return hidden_states.index_select(0, gather_rows[0])
    gather_index = gather_rows.unsqueeze(-1).expand(
        -1,
        -1,
        int(hidden_states.shape[-1]),
    )
    return hidden_states.gather(1, gather_index)


def tome_merge_image(
    hidden_states: Any,
    keys: Any,
    *,
    r: int,
    token_sizes: Any | None = None,
    protected_indices: Iterable[int] = (),
    coordinates: Iterable[Any] = (),
    media_unit: str | LayoutTokenUnit = LayoutTokenUnit.IMAGE_PATCH,
    eps: float = 1e-6,
    clamp_r: bool = True,
    rules: Mapping[str, Any] | None = None,
) -> ToMeImageResult:
    """Apply one paper-consistent ToMe merge step.

    Parameters
    ----------
    hidden_states:
        A single token sequence ``[T, C]`` or a batch ``[B, T, C]``.
    keys:
        Attention keys in one of ``[T, D]``, ``[T, H, D]``, ``[B, T, D]``,
        or ``[B, T, H, D]``.  A present ``H`` dimension is averaged before
        L2/cosine matching.  Backbone adapters are responsible for moving a
        provider-native head dimension into this documented position.
    r:
        Absolute number of tokens requested for removal in this layer.  ToMe
        cannot remove more than half of the unprotected tokens in one A/B
        matching step.  The paper-compatible default clamps an excessive
        request and reports both requested and effective values in metadata;
        set ``clamp_r=False`` to require an exact feasible value.
    token_sizes:
        Optional positive token masses.  Missing masses start at one.  Merged
        hidden states use a size-weighted mean and output masses use a sum.
    protected_indices:
        Input indices that may be neither a merge source nor destination.
    coordinates:
        Optional shared source coordinates, one per token.  Batched inputs use
        the same source coordinate layout for every sequence.
    media_unit:
        Physical unit represented by one input token.  Encoder integrations
        normally use ``image_patch``; a layout-aware post-projector legalizer
        may use ``image_feature`` when the native backbone must temporarily
        restore a regular grid before its spatial merger.
    eps:
        Positive L2-normalisation epsilon.
    clamp_r:
        Whether to clamp ``r`` to the paper's one-step feasible maximum.

    Notes
    -----
    Matching indices are discrete and intentionally selected without gradient
    tracking.  Once selected, size-weighted values and masses remain fully on
    the autograd path, matching ToMe's training-time behaviour.
    """

    # Keeping this import inside the call is part of the public control-plane
    # contract: registry/catalog inspection must not require PyTorch.
    import torch

    _validate_tensor_inputs(torch, hidden_states, keys)
    is_batched = hidden_states.ndim == 3
    values = hidden_states if is_batched else hidden_states.unsqueeze(0)
    batch_size, token_count, _ = (int(size) for size in values.shape)
    if token_count < 1:
        raise ValueError("ToMe requires at least one token")

    requested_r = _strict_non_negative_int(r, "r")
    if not isinstance(clamp_r, bool):
        raise TypeError("clamp_r must be a bool")
    if isinstance(eps, bool) or not isinstance(eps, (int, float)) or eps <= 0:
        raise ValueError("eps must be a positive number")

    protected = _normalise_protected(protected_indices, token_count)
    source_coordinates = tuple(coordinates)
    if source_coordinates and len(source_coordinates) != token_count:
        raise ValueError(
            "coordinates must be empty or contain one entry per source token"
        )
    media_unit_value = (
        media_unit.value if isinstance(media_unit, LayoutTokenUnit) else media_unit
    )

    metric = _canonical_key_metric(
        torch,
        keys,
        is_batched=is_batched,
        batch_size=batch_size,
        token_count=token_count,
        eps=float(eps),
    )
    sizes = _canonical_token_sizes(
        torch,
        token_sizes,
        values=values,
        is_batched=is_batched,
    )

    partition_a = tuple(range(0, token_count, 2))
    partition_b = tuple(range(1, token_count, 2))
    protected_set = set(protected)
    eligible_a = tuple(index for index in partition_a if index not in protected_set)
    eligible_b = tuple(index for index in partition_b if index not in protected_set)

    # The official implementation bounds a one-layer operation by half of the
    # non-protected sequence.  General protected indices also require at least
    # one legal B destination and enough legal A sources.
    paper_cap = (token_count - len(protected)) // 2
    feasible_r = min(paper_cap, len(eligible_a)) if eligible_b else 0
    if requested_r > feasible_r and not clamp_r:
        raise ValueError(
            f"r={requested_r} is not feasible for this layout; maximum is {feasible_r}"
        )
    effective_r = min(requested_r, feasible_r)

    merge_pairs: tuple[tuple[tuple[int, int], ...], ...] = ()
    if effective_r == 0:
        output_values = values
        output_sizes = sizes
        groups_by_batch = tuple(
            tuple((index,) for index in range(token_count)) for _ in range(batch_size)
        )
        layout_deltas = tuple(
            LayoutDelta.identity(
                token_count,
                media_unit=media_unit,
                coordinates=source_coordinates,
                protected_indices=protected,
                layout={
                    "algorithm": "tome",
                    "media_unit": media_unit_value,
                    "ordering": _IDENTITY_ORDERING,
                    "r": 0,
                },
            )
            for _ in range(batch_size)
        )
        ordering = _IDENTITY_ORDERING
    else:
        selected_sources, selected_destinations = _select_bipartite_edges(
            torch,
            metric,
            eligible_a=eligible_a,
            eligible_b=eligible_b,
            r=effective_r, rules=rules, values=values, sizes=sizes,
        )
        output_values, output_sizes = _merge_bipartite_edges(
            torch,
            values,
            sizes,
            partition_a=partition_a,
            partition_b=partition_b,
            selected_sources=selected_sources,
            selected_destinations=selected_destinations, rules=rules,
        )
        groups_accumulator: list[tuple[tuple[int, ...], ...]] = []
        deltas: list[LayoutDelta] = []
        # LayoutDelta is a pure-Python audit contract.  Copy the compact edge
        # list once, after the tensor merge has completed entirely through
        # batched gather/scatter operations.
        selected_sources_cpu = selected_sources.detach().cpu().tolist()
        selected_destinations_cpu = selected_destinations.detach().cpu().tolist()
        for batch_index, (source_row, destination_row) in enumerate(
            zip(selected_sources_cpu, selected_destinations_cpu)
        ):
            selected = dict(zip(source_row, destination_row))
            groups, representative_indices = _paper_ordered_source_groups(
                partition_a=partition_a,
                partition_b=partition_b,
                selected_destinations=selected,
            )
            mapping = {
                source_index: result_index
                for result_index, group in enumerate(groups)
                for source_index in group
            }
            delta = LayoutDelta.from_mapping(
                source_count=token_count,
                old_to_new=mapping,
                result_count=token_count - effective_r,
                media_unit=media_unit,
                coordinates=source_coordinates,
                protected_indices=protected,
                source_layout={
                    "algorithm": "tome",
                    "media_unit": media_unit_value,
                    "partition": "alternating_even_a_odd_b",
                    "partition_a_indices": partition_a,
                    "partition_b_indices": partition_b,
                },
                result_layout={
                    "algorithm": "tome",
                    "media_unit": media_unit_value,
                    "ordering": _PAPER_ORDERING,
                    "source_a_order": "ascending_source_index",
                    "destination_b_order": "ascending_source_index",
                    "source_groups": groups,
                    "representative_indices": representative_indices,
                    "representative_policy": (
                        "unmerged_a_self_destination_b_for_merged_groups"
                    ),
                    "result_coordinates": (
                        tuple(
                            source_coordinates[index]
                            for index in representative_indices
                        )
                        if source_coordinates
                        else ()
                    ),
                    "r": effective_r,
                },
                ordering_policy=OrderingPolicy.EXPLICIT,
            )
            groups_accumulator.append(groups)
            deltas.append(delta)

        merge_pairs = tuple(tuple(zip(sources, destinations))
            for sources, destinations in zip(selected_sources_cpu, selected_destinations_cpu))
        groups_by_batch = tuple(groups_accumulator)
        layout_deltas = tuple(deltas)
        ordering = _PAPER_ORDERING

    # Attention logits convention is [..., heads, query, key].  These shapes
    # broadcast without a backbone-specific reshape.
    batched_attention_bias = output_sizes.squeeze(-1).log()[:, None, None, :]
    result_values = output_values if is_batched else output_values[0]
    result_sizes = output_sizes if is_batched else output_sizes[0]
    attention_bias = batched_attention_bias if is_batched else batched_attention_bias[0]
    metadata = MappingProxyType(
        {
            "algorithm": "tome",
            "variant": "encoder_tome_image",
            "score_metric": "custom" if rules and "scorer" in rules else "mean_head_key_l2_cosine",
            "key_head_reduction": "mean",
            "partition": "alternating_even_a_odd_b",
            "r_requested": requested_r,
            "r_effective": effective_r,
            "r_feasible_max": feasible_r,
            "r_was_clamped": requested_r != effective_r,
            "protected_indices": protected,
            "ordering": ordering,
            "source_a_order": "ascending_source_index",
            "destination_b_order": "ascending_source_index",
            "merge_value": "custom" if rules and "aggregator" in rules else "token_size_weighted_mean",
            "token_size_update": "sum",
            "proportional_attention_bias": "log_token_size",
            "matching_gradient": "detached_discrete_selection",
            "merge_gradient": "autograd_preserved",
            "tensor_implementation": "custom_group_aggregation_with_scatter_sizes" if rules and "aggregator" in rules else "batched_gather_scatter_add",
            "layout_audit_transfer": "one_compact_edge_list_per_result",
            "batch_size": batch_size,
            "is_batched": is_batched,
            "media_unit": media_unit_value,
            **rule_metadata(rules),
        }
    )
    return ToMeImageResult(
        hidden_states=result_values,
        token_sizes=result_sizes,
        proportional_attention_bias=attention_bias,
        layout_deltas=layout_deltas,
        source_groups=groups_by_batch,
        metadata=metadata,
        is_batched=is_batched,
        merge_pairs=merge_pairs, input_token_sizes=sizes,
    )


def _validate_tensor_inputs(torch: Any, hidden_states: Any, keys: Any) -> None:
    if not torch.is_tensor(hidden_states):
        raise TypeError("hidden_states must be a torch.Tensor")
    if hidden_states.ndim not in (2, 3):
        raise ValueError("hidden_states must have shape [T, C] or [B, T, C]")
    if hidden_states.ndim == 3 and int(hidden_states.shape[0]) < 1:
        raise ValueError("batched hidden_states must contain at least one sequence")
    if int(hidden_states.shape[-1]) < 1:
        raise ValueError("hidden_states must have a non-empty feature dimension")
    if not torch.is_floating_point(hidden_states):
        raise TypeError("hidden_states must have a floating-point dtype")
    if not torch.is_tensor(keys):
        raise TypeError("keys must be a torch.Tensor")
    if keys.device != hidden_states.device:
        raise ValueError("keys and hidden_states must be on the same device")
    if not torch.is_floating_point(keys):
        raise TypeError("keys must have a floating-point dtype")


def _canonical_key_metric(
    torch: Any,
    keys: Any,
    *,
    is_batched: bool,
    batch_size: int,
    token_count: int,
    eps: float,
) -> Any:
    if is_batched:
        if keys.ndim == 4:
            if tuple(int(size) for size in keys.shape[:2]) != (
                batch_size,
                token_count,
            ):
                raise ValueError("batched multi-head keys must have shape [B, T, H, D]")
            if int(keys.shape[2]) < 1 or int(keys.shape[3]) < 1:
                raise ValueError("keys must have non-empty head and feature dimensions")
            metric = keys.detach().float().mean(dim=2)
        elif keys.ndim == 3:
            if tuple(int(size) for size in keys.shape[:2]) != (
                batch_size,
                token_count,
            ):
                raise ValueError("batched keys must have shape [B, T, D]")
            if int(keys.shape[2]) < 1:
                raise ValueError("keys must have a non-empty feature dimension")
            metric = keys.detach().float()
        else:
            raise ValueError(
                "keys for batched hidden states must have shape [B, T, D] "
                "or [B, T, H, D]"
            )
    else:
        if keys.ndim == 3:
            if int(keys.shape[0]) != token_count:
                raise ValueError(
                    "single-sequence multi-head keys must have shape [T, H, D]"
                )
            if int(keys.shape[1]) < 1 or int(keys.shape[2]) < 1:
                raise ValueError("keys must have non-empty head and feature dimensions")
            metric = keys.detach().float().mean(dim=1).unsqueeze(0)
        elif keys.ndim == 2:
            if int(keys.shape[0]) != token_count:
                raise ValueError("single-sequence keys must have shape [T, D]")
            if int(keys.shape[1]) < 1:
                raise ValueError("keys must have a non-empty feature dimension")
            metric = keys.detach().float().unsqueeze(0)
        else:
            raise ValueError(
                "keys for single hidden states must have shape [T, D] or [T, H, D]"
            )
    if not bool(torch.isfinite(metric).all().item()):
        raise ValueError("keys must contain only finite values")
    return torch.nn.functional.normalize(metric, p=2.0, dim=-1, eps=eps)


def _canonical_token_sizes(
    torch: Any,
    token_sizes: Any | None,
    *,
    values: Any,
    is_batched: bool,
) -> Any:
    batch_size, token_count = int(values.shape[0]), int(values.shape[1])
    if token_sizes is None:
        sizes = torch.ones(
            (batch_size, token_count, 1),
            dtype=values.dtype,
            device=values.device,
        )
    else:
        if not torch.is_tensor(token_sizes):
            token_sizes = torch.as_tensor(
                token_sizes,
                dtype=values.dtype,
                device=values.device,
            )
        elif token_sizes.device != values.device:
            raise ValueError("token_sizes and hidden_states must be on the same device")
        if is_batched:
            if token_sizes.ndim == 2 and tuple(token_sizes.shape) == (
                batch_size,
                token_count,
            ):
                sizes = token_sizes.unsqueeze(-1)
            elif token_sizes.ndim == 3 and tuple(token_sizes.shape) == (
                batch_size,
                token_count,
                1,
            ):
                sizes = token_sizes
            else:
                raise ValueError(
                    "token_sizes for a batch must have shape [B, T] or [B, T, 1]"
                )
        else:
            if token_sizes.ndim == 1 and int(token_sizes.shape[0]) == token_count:
                sizes = token_sizes.reshape(1, token_count, 1)
            elif token_sizes.ndim == 2 and tuple(token_sizes.shape) == (
                token_count,
                1,
            ):
                sizes = token_sizes.unsqueeze(0)
            else:
                raise ValueError(
                    "token_sizes for one sequence must have shape [T] or [T, 1]"
                )
        if not torch.is_floating_point(sizes):
            sizes = sizes.to(dtype=values.dtype)
        elif sizes.dtype != values.dtype:
            sizes = sizes.to(dtype=values.dtype)
    if not bool(torch.isfinite(sizes.detach()).all().item()):
        raise ValueError("token_sizes must contain only finite values")
    if bool((sizes.detach() <= 0).any().item()):
        raise ValueError("token_sizes must be strictly positive")
    return sizes


def _select_bipartite_edges(
    torch: Any,
    metric: Any,
    *,
    eligible_a: tuple[int, ...],
    eligible_b: tuple[int, ...],
    r: int,
    rules: Mapping[str, Any] | None = None, values: Any = None, sizes: Any = None,
) -> tuple[Any, Any]:
    """Return selected source and destination indices as ``[B, r]`` tensors."""

    with torch.no_grad():
        a_indices = torch.tensor(eligible_a, dtype=torch.long, device=metric.device)
        b_indices = torch.tensor(eligible_b, dtype=torch.long, device=metric.device)
        source_metric = metric.index_select(1, a_indices)
        destination_metric = metric.index_select(1, b_indices)
        scores = source_metric @ destination_metric.transpose(-1, -2)
        if rules and "scorer" in rules:
            scores = torch.stack([score_rule(rules, "scorer", {
                "values": values[i].detach(), "metric": metric[i], "token_sizes": sizes[i].detach(),
                "source_indices": eligible_a, "destination_indices": eligible_b,
                "merge_count": r, "batch_index": i, "scores": scores[i]}, scores[i])
                for i in range(int(metric.shape[0]))])
        best_scores, best_destination_slots = scores.max(dim=-1)
        ranked_source_slots = torch.argsort(
            best_scores,
            dim=-1,
            descending=True,
            stable=True,
        )

        selected_source_slots = ranked_source_slots[:, :r]
        selected_destination_slots = best_destination_slots.gather(
            1,
            selected_source_slots,
        )
        selected_sources = a_indices[selected_source_slots]
        selected_destinations = b_indices[selected_destination_slots]
        if rules and "grouper" in rules:
            rows = [_validate_pairs(rules["grouper"]({"values": values[i].detach(),
                "metric": metric[i], "token_sizes": sizes[i].detach(), "scores": scores[i],
                "source_indices": eligible_a, "destination_indices": eligible_b,
                "merge_count": r, "batch_index": i}), eligible_a=eligible_a, eligible_b=eligible_b, r=r)
                for i in range(int(metric.shape[0]))]
            selected_sources = torch.tensor([[source for source, _ in row] for row in rows], device=metric.device)
            selected_destinations = torch.tensor([[destination for _, destination in row] for row in rows], device=metric.device)
    return selected_sources, selected_destinations


def _paper_ordered_source_groups(
    *,
    partition_a: tuple[int, ...],
    partition_b: tuple[int, ...],
    selected_destinations: Mapping[int, int],
) -> tuple[tuple[tuple[int, ...], ...], tuple[int, ...]]:
    selected_sources = set(selected_destinations)
    unmerged_a = tuple(
        (index,) for index in partition_a if index not in selected_sources
    )
    destination_groups = []
    for destination in partition_b:
        sources = [
            source
            for source, matched_destination in selected_destinations.items()
            if matched_destination == destination
        ]
        destination_groups.append(tuple(sorted((destination, *sources))))
    groups = (*unmerged_a, *destination_groups)
    representatives = (
        *(group[0] for group in unmerged_a),
        *partition_b,
    )
    return groups, representatives


def _merge_bipartite_edges(
    torch: Any,
    values: Any,
    sizes: Any,
    *,
    partition_a: tuple[int, ...],
    partition_b: tuple[int, ...],
    selected_sources: Any,
    selected_destinations: Any,
    rules: Mapping[str, Any] | None = None,
) -> tuple[Any, Any]:
    """Apply all selected edges with batched gather and scatter-add."""

    batch_size, _, channels = (int(size) for size in values.shape)
    merge_count = int(selected_sources.shape[1])
    a_indices = torch.tensor(partition_a, dtype=torch.long, device=values.device)
    b_indices = torch.tensor(partition_b, dtype=torch.long, device=values.device)
    a_values = values.index_select(1, a_indices)
    b_values = values.index_select(1, b_indices)
    a_sizes = sizes.index_select(1, a_indices)
    b_sizes = sizes.index_select(1, b_indices)

    # Original A/B indices are alternating, so division maps them to their
    # slots within the respective compact partitions.
    selected_a_slots = selected_sources // 2
    selected_b_slots = selected_destinations // 2
    selected_a_values = _batch_gather(a_values, selected_a_slots)
    selected_a_sizes = _batch_gather(a_sizes, selected_a_slots)

    destination_weighted = b_values * b_sizes
    selected_weighted = selected_a_values * selected_a_sizes
    weighted_scatter_indices = selected_b_slots.unsqueeze(-1).expand(
        batch_size,
        merge_count,
        channels,
    )
    size_scatter_indices = selected_b_slots.unsqueeze(-1)
    merged_destination_weighted = destination_weighted.scatter_add(
        1,
        weighted_scatter_indices,
        selected_weighted,
    )
    merged_destination_sizes = b_sizes.scatter_add(
        1,
        size_scatter_indices,
        selected_a_sizes,
    )
    merged_destination_values = merged_destination_weighted / merged_destination_sizes
    if rules and "aggregator" in rules:
        batch_outputs = []
        for batch_index in range(batch_size):
            pairs = tuple(zip(selected_sources[batch_index].tolist(), selected_destinations[batch_index].tolist()))
            rows = []
            for destination_slot, destination in enumerate(partition_b):
                assigned = tuple(source for source, target in pairs if target == destination)
                if not assigned:
                    rows.append(values[batch_index, destination])
                    continue
                indices = (destination, *assigned)
                group_values = values[batch_index, list(indices)]
                group_sizes = sizes[batch_index, list(indices)]
                output = aggregate_rule(rules, "aggregator", {"values": group_values,
                    "indices": indices, "destination": destination, "token_sizes": group_sizes,
                    "batch_index": batch_index, "default_values": merged_destination_values[batch_index, destination_slot]},
                    merged_destination_values[batch_index, destination_slot])
                if torch.is_grad_enabled() and group_values.requires_grad and not output.requires_grad:
                    raise ValueError("ToMe aggregator detached the value gradient.")
                rows.append(output)
            batch_outputs.append(torch.stack(rows))
        merged_destination_values = torch.stack(batch_outputs)

    selected_mask = torch.zeros(
        (batch_size, len(partition_a)),
        dtype=torch.bool,
        device=values.device,
    )
    selected_mask.scatter_(1, selected_a_slots, True)
    all_a_slots = torch.arange(
        len(partition_a),
        dtype=torch.long,
        device=values.device,
    ).expand(batch_size, -1)
    unmerged_a_slots = all_a_slots.masked_select(~selected_mask).reshape(
        batch_size,
        len(partition_a) - merge_count,
    )
    unmerged_a_values = _batch_gather(a_values, unmerged_a_slots)
    unmerged_a_sizes = _batch_gather(a_sizes, unmerged_a_slots)
    return (
        torch.cat((unmerged_a_values, merged_destination_values), dim=1),
        torch.cat((unmerged_a_sizes, merged_destination_sizes), dim=1),
    )


def _validate_pairs(value: Any, *, eligible_a: tuple[int, ...], eligible_b: tuple[int, ...], r: int) -> tuple[tuple[int, int], ...]:
    import torch
    if isinstance(value, torch.Tensor):
        if value.dtype not in (torch.int32, torch.int64) or value.shape != (r, 2):
            raise ValueError("ToMe grouper must return integer [merge_count, 2] pairs.")
        value = value.detach().cpu().tolist()
    if not isinstance(value, (list, tuple)) or len(value) != r:
        raise ValueError("ToMe grouper must preserve the effective merge budget.")
    pairs = []
    for pair in value:
        if (not isinstance(pair, (list, tuple)) or len(pair) != 2
                or any(isinstance(i, bool) or not isinstance(i, int) for i in pair)):
            raise ValueError("ToMe grouper must return integer source/destination pairs.")
        source, destination = pair
        if source not in eligible_a or destination not in eligible_b:
            raise ValueError("ToMe grouper must use eligible A/B members and exclude protected tokens.")
        pairs.append((source, destination))
    if len({source for source, _ in pairs}) != r:
        raise ValueError("ToMe grouper may select each source only once.")
    return tuple(pairs)


def replay_tome_image(hidden_states: Any, result: ToMeImageResult, *, token_sizes: Any | None = None,
                      rules: Mapping[str, Any] | None = None) -> Any:
    """Replay one saved matching on actual values, including legalized ordering.

    Fixed attention-key references retain their separate original mass mapping;
    this helper is for value/DeepStack replay and never calls scorer or grouper.
    """
    import torch
    is_batched = hidden_states.ndim == 3
    values = hidden_states if is_batched else hidden_states.unsqueeze(0)
    if (hidden_states.ndim not in (2, 3) or is_batched != result.is_batched
            or len(result.layout_deltas) != int(values.shape[0])
            or any(delta.source_count != int(values.shape[1]) for delta in result.layout_deltas)):
        raise ValueError("ToMe replay values must retain the saved source batch and token layout.")
    if all(delta.is_identity for delta in result.layout_deltas):
        return hidden_states
    if not result.merge_pairs or result.input_token_sizes is None:
        raise ValueError("ToMe replay requires saved matching edges and input masses.")
    if token_sizes is None:
        sizes = result.input_token_sizes.to(device=values.device, dtype=values.dtype)
    else:
        sizes = _canonical_token_sizes(torch, token_sizes, values=values, is_batched=is_batched)
    sources = torch.tensor([[source for source, _ in row] for row in result.merge_pairs], device=values.device)
    destinations = torch.tensor([[destination for _, destination in row] for row in result.merge_pairs], device=values.device)
    partition_a, partition_b = tuple(range(0, int(values.shape[1]), 2)), tuple(range(1, int(values.shape[1]), 2))
    output, _ = _merge_bipartite_edges(torch, values, sizes, partition_a=partition_a, partition_b=partition_b,
        selected_sources=sources, selected_destinations=destinations, rules=rules)
    rows = []
    for batch_index, pairs in enumerate(result.merge_pairs):
        paper_groups, _ = _paper_ordered_source_groups(partition_a=partition_a, partition_b=partition_b,
            selected_destinations=dict(pairs))
        positions = {group: i for i, group in enumerate(paper_groups)}
        order = torch.tensor([positions[group] for group in result.source_groups[batch_index]], device=values.device)
        rows.append(output[batch_index].index_select(0, order))
    merged = torch.stack(rows)
    return merged if is_batched else merged[0]


def _batch_gather(values: Any, indices: Any) -> Any:
    gather_indices = indices.unsqueeze(-1).expand(
        int(values.shape[0]),
        int(indices.shape[1]),
        int(values.shape[2]),
    )
    return values.gather(1, gather_indices)


def _normalise_protected(
    protected_indices: Iterable[int],
    token_count: int,
) -> tuple[int, ...]:
    if isinstance(protected_indices, (str, bytes)):
        raise TypeError("protected_indices must be an iterable of integers")
    raw = tuple(protected_indices)
    if any(isinstance(index, bool) or not isinstance(index, int) for index in raw):
        raise TypeError("protected_indices entries must be integers")
    if len(raw) != len(set(raw)):
        raise ValueError("protected_indices must not contain duplicates")
    if any(index < 0 or index >= token_count for index in raw):
        raise ValueError("protected_indices contains an index outside the sequence")
    return tuple(sorted(raw))


def _strict_non_negative_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if value < 0:
        raise ValueError(f"{name} must be non-negative")
    return value


# Concise name for algorithm descriptors and researcher code.
def tome_merge_reference(
    keys: Any,
    delta: LayoutDelta,
    *,
    token_sizes: Any | None,
) -> Any:
    """Carry a fixed key reference through a mass-weighted merge layout."""

    import torch
    if int(keys.shape[0]) != delta.source_count:
        raise ValueError("ToMe reference and layout counts differ.")
    weights = (
        torch.ones(
            delta.source_count,
            dtype=keys.dtype,
            device=keys.device,
        )
        if token_sizes is None
        else torch.as_tensor(token_sizes, dtype=keys.dtype, device=keys.device).reshape(delta.source_count)
    )
    outputs = []
    for result_index in range(delta.result_count):
        group = tuple(source for source, target in delta.old_to_new.items() if target == result_index)
        indices = torch.tensor(group, dtype=torch.long, device=keys.device)
        group_weights = weights.index_select(0, indices)
        group_keys = keys.index_select(0, indices)
        outputs.append(
            (group_keys * group_weights[:, None, None]).sum(dim=0) / group_weights.sum()
        )
    return torch.stack(outputs, dim=0)



tome_merge = tome_merge_image


__all__ = [
    "ToMeImageResult",
    "ToMeResult",
    "tome_preserve_source_order",
    "tome_restore_dense_boundary",
    "tome_merge",
    "tome_merge_image",
    "tome_merge_reference",
    "replay_tome_image",
]
