"""Pure mathematics for fixed-budget visual-token merge attribution.

The functions in this module deliberately do not know about the analysis
runtime.  This keeps clustering and the single-layer counterfactuals directly
testable without loading a checkpoint.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from typing import Any, Iterable, Mapping, Sequence


ATTRIBUTION_VARIANTS = ("A", "B", "C", "D", "O-Z", "O-M")
VALUE_RESIDUAL_RANKS = (1, 4, 8)
VALUE_RESIDUAL_VARIANTS = tuple(
    f"O-Z-VR{rank}" for rank in VALUE_RESIDUAL_RANKS
)
ALL_ATTRIBUTION_VARIANTS = ATTRIBUTION_VARIANTS + VALUE_RESIDUAL_VARIANTS
PARTITION_METHODS = ("grid", "feature_anchor")


@dataclass(frozen=True)
class VisualPartition:
    """One deterministic, exhaustive hard partition of a rectangular grid."""

    method: str
    height: int
    width: int
    target_tokens: int
    members: tuple[tuple[int, ...], ...]
    partition_hash: str


def balanced_grid_shape(height: int, width: int, target_tokens: int) -> tuple[int, int]:
    """Choose an exact factorization of ``target_tokens`` closest to the grid ratio."""

    _validate_grid_size(height, width, target_tokens)
    candidates: list[tuple[float, int, int]] = []
    for rows in range(1, target_tokens + 1):
        if target_tokens % rows:
            continue
        cols = target_tokens // rows
        if rows > height or cols > width:
            continue
        ratio_error = abs(math.log((rows / cols) / (height / width)))
        candidates.append((ratio_error, rows, cols))
    if not candidates:
        raise ValueError(
            f"Cannot form {target_tokens} non-empty rectangular cells from "
            f"a {height}x{width} grid."
        )
    _, rows, cols = min(candidates, key=lambda item: (item[0], item[1], item[2]))
    return rows, cols


def grid_partition(height: int, width: int, target_tokens: int) -> VisualPartition:
    """Split both axes into balanced contiguous bins with exact full coverage."""

    rows, cols = balanced_grid_shape(height, width, target_tokens)
    row_bins = _balanced_axis_bins(height, rows)
    col_bins = _balanced_axis_bins(width, cols)
    members = tuple(
        tuple(r * width + c for r in row_bin for c in col_bin)
        for row_bin in row_bins
        for col_bin in col_bins
    )
    return make_partition("grid", height, width, target_tokens, members)


def feature_anchor_partition(
    features: Any,
    *,
    height: int,
    width: int,
    target_tokens: int,
    eps: float = 1e-12,
) -> VisualPartition:
    """Deterministic farthest-first cosine anchors followed by hard assignment.

    Anchor zero is the first token in row-major order.  Every later anchor is
    the lowest-index token whose maximum cosine similarity to existing anchors
    is minimal.  Explicitly pinning anchors to their own clusters guarantees
    exactly ``target_tokens`` non-empty clusters, including duplicate features.
    """

    import torch

    _validate_grid_size(height, width, target_tokens)
    if getattr(features, "ndim", None) != 2 or int(features.shape[0]) != height * width:
        raise ValueError("Feature-anchor partition expects [height*width, dim] features.")
    values = features.detach().to(dtype=torch.float32)
    norms = values.norm(dim=-1, keepdim=True).clamp_min(float(eps))
    values = values / norms
    anchors = [0]
    selected = torch.zeros(len(values), dtype=torch.bool, device=values.device)
    selected[0] = True
    best_similarity = values @ values[0]
    while len(anchors) < target_tokens:
        candidates = best_similarity.masked_fill(selected, float("inf"))
        next_anchor = int(torch.argmin(candidates).item())
        anchors.append(next_anchor)
        selected[next_anchor] = True
        best_similarity = torch.maximum(best_similarity, values @ values[next_anchor])

    anchor_tensor = torch.tensor(anchors, device=values.device, dtype=torch.long)
    assignments = torch.argmax(values @ values.index_select(0, anchor_tensor).T, dim=-1)
    for cluster, anchor in enumerate(anchors):
        assignments[anchor] = cluster
    members = tuple(
        tuple(int(index) for index in torch.nonzero(assignments == cluster, as_tuple=False).flatten().tolist())
        for cluster in range(target_tokens)
    )
    return make_partition(
        "feature_anchor", height, width, target_tokens, members
    )


def make_partition(
    method: str,
    height: int,
    width: int,
    target_tokens: int,
    members: Iterable[Iterable[int]],
) -> VisualPartition:
    if method not in PARTITION_METHODS:
        raise ValueError(f"Unsupported visual partition method {method!r}.")
    normalized = tuple(tuple(int(value) for value in cluster) for cluster in members)
    validate_partition(normalized, source_tokens=height * width, target_tokens=target_tokens)
    payload = json.dumps(
        {"method": method, "height": height, "width": width, "members": normalized},
        separators=(",", ":"),
    ).encode("utf-8")
    return VisualPartition(
        method=method,
        height=height,
        width=width,
        target_tokens=target_tokens,
        members=normalized,
        partition_hash=hashlib.sha256(payload).hexdigest(),
    )


def validate_partition(
    members: Sequence[Sequence[int]], *, source_tokens: int, target_tokens: int
) -> None:
    if len(members) != target_tokens:
        raise ValueError(
            f"Partition has {len(members)} clusters; expected {target_tokens}."
        )
    if any(not cluster for cluster in members):
        raise ValueError("Visual partition contains an empty cluster.")
    flat = [int(index) for cluster in members for index in cluster]
    if len(flat) != source_tokens:
        raise ValueError(
            f"Partition covers {len(flat)} assignments; expected {source_tokens}."
        )
    if sorted(flat) != list(range(source_tokens)):
        raise ValueError("Visual partition must cover every source token exactly once.")


def partition_statistics(partition: VisualPartition, features: Any) -> dict[str, Any]:
    """Return cluster size, centroid, span and channel dispersion evidence."""

    import torch

    if getattr(features, "ndim", None) != 2 or int(features.shape[0]) != partition.height * partition.width:
        raise ValueError("Partition statistics received a feature/grid size mismatch.")
    result = []
    for cluster in partition.members:
        indices = torch.tensor(cluster, device=features.device, dtype=torch.long)
        rows = indices.div(partition.width, rounding_mode="floor").to(torch.float32)
        cols = indices.remainder(partition.width).to(torch.float32)
        selected = features.index_select(0, indices).to(torch.float32)
        result.append(
            {
                "size": len(cluster),
                "centroid": [float(rows.mean().item()), float(cols.mean().item())],
                "span": [
                    float((rows.max() - rows.min()).item()),
                    float((cols.max() - cols.min()).item()),
                ],
                "channel_std_l2": float(selected.std(dim=0, correction=0).norm().item()),
                "max_abs_residual": float(
                    (selected - selected.mean(dim=0, keepdim=True)).abs().max().item()
                ),
            }
        )
    return {"clusters": result}


def cluster_mean(values: Any, members: Sequence[Sequence[int]]) -> Any:
    import torch

    return torch.stack(
        [values[list(cluster)].to(torch.float32).mean(dim=0) for cluster in members],
        dim=0,
    )


def apply_mrope(states: Any, cos: Any, sin: Any, mrope_section: Sequence[int]) -> Any:
    """Apply Qwen2.5-VL three-axis RoPE to ``[heads, tokens, dim]`` states."""

    import torch

    if states.ndim != 3:
        raise ValueError("M-RoPE states must have shape [heads, tokens, dim].")
    sections = list(mrope_section) * 2
    if cos.ndim == 4:
        if int(cos.shape[0]) != 3 or int(cos.shape[1]) != 1:
            raise ValueError(
                "Native M-RoPE cos/sin must have shape [3, 1, tokens, dim]."
            )
        cos_parts = cos.split(sections, dim=-1)
        sin_parts = sin.split(sections, dim=-1)
        cos = torch.cat(
            [part[index % 3] for index, part in enumerate(cos_parts)], dim=-1
        )
        sin = torch.cat(
            [part[index % 3] for index, part in enumerate(sin_parts)], dim=-1
        )
    if cos.ndim == 3:
        if int(cos.shape[0]) != 1:
            raise ValueError("M-RoPE helper only supports batch size one.")
        cos = cos[0]
        sin = sin[0]
    if cos.ndim != 2 or cos.shape != sin.shape:
        raise ValueError("M-RoPE cos/sin must have matching [tokens, dim] shapes.")
    half = states.shape[-1] // 2
    rotated = torch.cat((-states[..., half:], states[..., :half]), dim=-1)
    return states * cos.unsqueeze(0) + rotated * sin.unsqueeze(0)


def repeat_kv(values: Any, query_heads: int) -> Any:
    """Expand KV heads to query heads using Qwen GQA ordering."""

    if values.ndim != 3:
        raise ValueError("KV tensor must have shape [kv_heads, tokens, dim].")
    kv_heads = int(values.shape[0])
    if query_heads % kv_heads:
        raise ValueError("Query head count must be divisible by KV head count.")
    return values.repeat_interleave(query_heads // kv_heads, dim=0)


def prepare_value_residual_factors(
    visual_keys: Any,
    visual_values: Any,
    members: Sequence[Sequence[int]],
    ranks: Sequence[int] = VALUE_RESIDUAL_RANKS,
) -> dict[int, tuple[tuple[Any, Any], ...]]:
    """Build deterministic low-rank factors for Cov(K, V) within each cluster.

    The first-order softmax expansion gives a query-conditioned cluster value
    ``mean(V) + Q Cov(K,V) / sqrt(head_dim)``. Truncated SVD factors make the
    correction rank-limited. Keys here are post-M-RoPE keys; values are the
    corresponding native Layer-0 values.
    """

    import torch

    requested = tuple(sorted({int(rank) for rank in ranks}))
    if not requested or any(rank < 1 for rank in requested):
        raise ValueError("Value residual ranks must be positive integers.")
    if visual_keys.ndim != 3 or visual_values.ndim != 3:
        raise ValueError("Visual keys and values must have shape [heads, tokens, dim].")
    if visual_keys.shape != visual_values.shape:
        raise ValueError("Visual keys and values must have matching shapes.")
    max_rank = min(max(requested), int(visual_keys.shape[-1]))
    factors_by_rank: dict[int, list[tuple[Any, Any]]] = {
        rank: [] for rank in requested
    }
    keys_fp32 = visual_keys.float()
    values_fp32 = visual_values.float()
    for cluster in members:
        indices = torch.tensor(cluster, device=visual_keys.device, dtype=torch.long)
        keys = keys_fp32.index_select(1, indices)
        values = values_fp32.index_select(1, indices)
        centered_keys = keys - keys.mean(dim=1, keepdim=True)
        centered_values = values - values.mean(dim=1, keepdim=True)
        covariance = torch.einsum(
            "hmd,hme->hde", centered_keys, centered_values
        ) / float(max(1, len(cluster)))
        left, singular, right = torch.linalg.svd(covariance, full_matrices=False)
        for rank in requested:
            effective_rank = min(rank, max_rank, max(1, len(cluster) - 1))
            # A singleton has zero covariance; retaining a zero rank-1 factor
            # keeps the returned tensor shapes uniform and makes it exact.
            factors_by_rank[rank].append(
                (
                    left[:, :, :effective_rank]
                    * singular[:, None, :effective_rank],
                    right[:, :effective_rank, :],
                )
            )
    return {rank: tuple(items) for rank, items in factors_by_rank.items()}


def attention_counterfactuals(
    *,
    queries: Any,
    text_keys: Any,
    text_values: Any,
    visual_keys: Any,
    visual_values: Any,
    representative_keys: Any,
    representative_values: Any,
    members: Sequence[Sequence[int]],
    variants: Sequence[str] = ATTRIBUTION_VARIANTS,
    value_residual_factors: Mapping[int, Sequence[tuple[Any, Any]]] | None = None,
    eps: float = 1e-12,
    return_outputs: bool = False,
) -> dict[str, Any]:
    """Compute A/B/C/D and exact partition/message Oracles in FP32.

    All tensors use ``[heads, tokens, dim]`` except queries, whose token axis is
    the eligible text-query set.  Text keys/values are the causally visible,
    unchanged prefix.  The caller is responsible for selecting only queries
    with the same visible key set (the normal post-image prompt suffix).
    """

    import torch

    selected_variants = tuple(variants)
    unknown = sorted(set(selected_variants) - set(ALL_ATTRIBUTION_VARIANTS))
    if unknown:
        raise ValueError(f"Unsupported attribution variants: {unknown}.")
    tensors = [
        queries,
        text_keys,
        text_values,
        visual_keys,
        visual_values,
        representative_keys,
        representative_values,
    ]
    if any(getattr(value, "ndim", None) != 3 for value in tensors):
        raise ValueError("Attention counterfactual tensors must be rank three.")
    heads, query_count, dim = (int(value) for value in queries.shape)
    if query_count < 1:
        raise ValueError("Attention counterfactual requires at least one text query.")
    if any(int(value.shape[0]) != heads or int(value.shape[-1]) != dim for value in tensors[1:]):
        raise ValueError("Attention tensors disagree on head count or dimension.")
    validate_partition(
        members,
        source_tokens=int(visual_keys.shape[1]),
        target_tokens=int(representative_keys.shape[1]),
    )

    q = queries.to(torch.float32)
    tk = text_keys.to(torch.float32)
    tv = text_values.to(torch.float32)
    vk = visual_keys.to(torch.float32)
    vv = visual_values.to(torch.float32)
    rk = representative_keys.to(torch.float32)
    rv = representative_values.to(torch.float32)
    beta = dim**-0.5
    text_logits = torch.einsum("hqd,htd->hqt", q, tk) * beta
    visual_logits = torch.einsum("hqd,hnd->hqn", q, vk) * beta
    representative_logits = torch.einsum("hqd,hkd->hqk", q, rk) * beta
    full_logits = torch.cat((text_logits, visual_logits), dim=-1)
    full_weights = torch.softmax(full_logits, dim=-1)
    full_output = torch.einsum(
        "hqs,hsd->hqd", full_weights, torch.cat((tv, vv), dim=1)
    )
    full_visual_mass = full_weights[..., int(tk.shape[1]) :].sum(dim=-1)
    full_text_weights = full_weights[..., : int(tk.shape[1])]

    cluster_logits = []
    cluster_mu_keys = []
    cluster_weighted_values = []
    cluster_log_z = []
    cluster_variances = []
    sizes = []
    for cluster in members:
        index = torch.tensor(cluster, device=vk.device, dtype=torch.long)
        logits = visual_logits.index_select(-1, index)
        keys = vk.index_select(1, index)
        values = vv.index_select(1, index)
        log_z = torch.logsumexp(logits, dim=-1)
        conditional = torch.softmax(logits, dim=-1)
        message = torch.einsum("hqm,hmd->hqd", conditional, values)
        mu = keys.mean(dim=1)
        centered = keys - mu[:, None, :]
        projection = torch.einsum("hqd,hmd->hqm", q, centered)
        directional_variance = projection.square().mean(dim=-1)
        cluster_logits.append(logits)
        cluster_mu_keys.append(mu)
        cluster_weighted_values.append(message)
        cluster_log_z.append(log_z)
        cluster_variances.append(directional_variance)
        sizes.append(len(cluster))
    mu_keys = torch.stack(cluster_mu_keys, dim=1)
    log_z = torch.stack(cluster_log_z, dim=-1)
    oracle_messages = torch.stack(cluster_weighted_values, dim=2)
    directional_variance = torch.stack(cluster_variances, dim=-1)
    size_bias = torch.tensor(sizes, device=q.device, dtype=torch.float32).log()[None, None, :]
    realization = torch.einsum("hqd,hkd->hqk", q, mu_keys - rk) * beta
    second_order = 0.5 * (beta**2) * directional_variance
    biases = {
        "A": torch.zeros_like(representative_logits),
        "B": size_bias.expand_as(representative_logits),
        "C": size_bias + realization,
        "D": size_bias + realization + second_order,
        "O-Z": log_z - representative_logits,
        "O-M": log_z - representative_logits,
    }
    value_ranks = {
        int(variant.removeprefix("O-Z-VR"))
        for variant in selected_variants
        if variant.startswith("O-Z-VR")
    }
    if value_ranks:
        if value_residual_factors is None:
            value_residual_factors = prepare_value_residual_factors(
                vk, vv, members, sorted(value_ranks)
            )
        missing_ranks = value_ranks - set(value_residual_factors)
        if missing_ranks:
            raise ValueError(
                f"Missing precomputed value residual factors for ranks {sorted(missing_ranks)}."
            )
        for rank in value_ranks:
            if len(value_residual_factors[rank]) != len(members):
                raise ValueError("Value residual factor count does not match cluster count.")
            biases[f"O-Z-VR{rank}"] = biases["O-Z"]
    reports: dict[str, Any] = {}
    outputs: dict[str, Any] = {}
    for variant in selected_variants:
        compressed_logits = torch.cat(
            (text_logits, representative_logits + biases[variant]), dim=-1
        )
        weights = torch.softmax(compressed_logits, dim=-1)
        text_count = int(tk.shape[1])
        text_weight = weights[..., :text_count]
        visual_weight = weights[..., text_count:]
        if variant == "O-M":
            visual_message = torch.einsum(
                "hqk,hqkd->hqd", visual_weight, oracle_messages
            )
        elif variant.startswith("O-Z-VR"):
            rank = int(variant.removeprefix("O-Z-VR"))
            messages = []
            for cluster_index, (left, right) in enumerate(
                value_residual_factors[rank]
            ):
                residual = torch.einsum(
                    "hqd,hdr,hre->hqe", q, left, right
                ) / math.sqrt(dim)
                center = vv.index_select(
                    1,
                    torch.tensor(
                        members[cluster_index], device=vv.device, dtype=torch.long
                    ),
                ).mean(dim=1)
                messages.append(center[:, None, :] + residual)
            query_messages = torch.stack(messages, dim=2)
            visual_message = torch.einsum(
                "hqk,hqkd->hqd", visual_weight, query_messages
            )
        else:
            visual_message = torch.einsum("hqk,hkd->hqd", visual_weight, rv)
        output = torch.einsum("hqt,htd->hqd", text_weight, tv) + visual_message
        outputs[variant] = output
        output_error = (output - full_output).norm(dim=-1) / full_output.norm(dim=-1).clamp_min(eps)
        text_error = (text_weight - full_text_weights).abs().sum(dim=-1)
        reports[variant] = {
            "attention_output_relative_error": _head_query_means(output_error),
            "visual_attention_mass_abs_error": _head_query_means(
                (visual_weight.sum(dim=-1) - full_visual_mass).abs()
            ),
            "text_attention_l1_error": _head_query_means(text_error),
            "cluster_log_partition_abs_error": _head_query_means(
                ((representative_logits + biases[variant]) - log_z).abs().mean(dim=-1)
            ),
        }
    reports["diagnostics"] = {
        "higher_order_abs_residual": _head_query_means(
            (biases["O-Z"] - biases["D"]).abs().mean(dim=-1)
        ),
        "value_realization_relative_error": _head_query_means(
            (outputs["O-Z"] - outputs["O-M"]).norm(dim=-1)
            / outputs["O-M"].norm(dim=-1).clamp_min(eps)
        ),
        "query_count": query_count,
    }
    if return_outputs:
        reports["attention_outputs"] = outputs
    return reports


def _head_query_means(values: Any) -> dict[str, Any]:
    per_head = values.mean(dim=1)
    return {
        "mean": float(per_head.mean().item()),
        "heads": {str(index): float(value.item()) for index, value in enumerate(per_head)},
    }


def _balanced_axis_bins(length: int, bins: int) -> tuple[tuple[int, ...], ...]:
    quotient, remainder = divmod(length, bins)
    result = []
    start = 0
    for index in range(bins):
        size = quotient + (1 if index < remainder else 0)
        result.append(tuple(range(start, start + size)))
        start += size
    return tuple(result)


def _validate_grid_size(height: int, width: int, target_tokens: int) -> None:
    if any(isinstance(value, bool) or not isinstance(value, int) or value < 1 for value in (height, width, target_tokens)):
        raise ValueError("Grid dimensions and target token count must be positive integers.")
    if height * width < target_tokens:
        raise ValueError(
            f"Source grid has {height * width} tokens, below target {target_tokens}."
        )


__all__ = [
    "ALL_ATTRIBUTION_VARIANTS",
    "ATTRIBUTION_VARIANTS",
    "PARTITION_METHODS",
    "VALUE_RESIDUAL_RANKS",
    "VALUE_RESIDUAL_VARIANTS",
    "VisualPartition",
    "apply_mrope",
    "attention_counterfactuals",
    "balanced_grid_shape",
    "cluster_mean",
    "feature_anchor_partition",
    "grid_partition",
    "make_partition",
    "partition_statistics",
    "prepare_value_residual_factors",
    "repeat_kv",
    "validate_partition",
]
