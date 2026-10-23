"""Deterministic temporal substitutions, predictions, and locality merges."""
from __future__ import annotations

import numpy as np

from .metrics import cosine_mean, relative_l2
from .provenance import merge_traces


def _ordered_chunks(traces):
    return list(dict.fromkeys(trace.hard_chunks[0] for trace in traces))


def _distance_distribution(values):
    values = np.asarray(values, dtype=float)
    if not len(values):
        return {
            "distance_ms_q25": None, "distance_ms_q50": None,
            "distance_ms_q75": None, "distance_ms_q95": None,
            "distance_histogram_edges_ms": [0, 80, 160, 320, 640, 1280, 2560, 1e12],
            "distance_histogram_counts": [0] * 7,
        }
    edges = np.asarray([0, 80, 160, 320, 640, 1280, 2560, 1e12], dtype=float)
    counts, _ = np.histogram(values, bins=edges)
    quantiles = np.quantile(values, [0.25, 0.5, 0.75, 0.95])
    return {
        "distance_ms_q25": float(quantiles[0]),
        "distance_ms_q50": float(quantiles[1]),
        "distance_ms_q75": float(quantiles[2]),
        "distance_ms_q95": float(quantiles[3]),
        "distance_histogram_edges_ms": edges.tolist(),
        "distance_histogram_counts": counts.tolist(),
    }


def _weighted_group(values, traces, indices):
    import torch

    index = torch.as_tensor(indices, dtype=torch.long, device=values.device)
    weights = torch.as_tensor([traces[i].mass for i in indices], device=values.device,
                              dtype=torch.float32)
    shape = (len(weights),) + (1,) * (values.ndim - 1)
    merged = (values[index].float() * weights.reshape(shape)).sum(0) / weights.sum()
    return merged.to(values.dtype), merge_traces([traces[i] for i in indices])


def _candidate_table(values, centers, *, mode, window_ms=None, far_ms=None):
    """Return candidate i/j/similarity arrays for one hard chunk."""
    import torch

    n = len(values)
    if n < 2:
        return np.empty(0, int), np.empty(0, int), np.empty(0, float)
    if mode == "adjacent":
        left = torch.arange(n - 1, device=values.device)
        right = left + 1
    else:
        pair = torch.triu_indices(n, n, offset=1, device=values.device)
        left, right = pair[0], pair[1]
        distances = torch.as_tensor(centers, device=values.device, dtype=torch.float32)
        distances = (distances[right] - distances[left]).abs()
        if mode == "local":
            keep = distances <= float(window_ms)
        elif mode == "far":
            keep = distances >= float(far_ms)
        elif mode == "global":
            keep = torch.ones_like(distances, dtype=torch.bool)
        else:
            raise ValueError(f"Unknown candidate mode {mode!r}")
        left, right = left[keep], right[keep]
    if not len(left):
        return np.empty(0, int), np.empty(0, int), np.empty(0, float)
    flat = values.detach().float().reshape(n, -1)
    flat = flat / flat.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    similarities = (flat[left] * flat[right]).sum(-1)
    return (
        np.asarray(left.cpu().tolist(), dtype=int),
        np.asarray(right.cpu().tolist(), dtype=int),
        np.asarray(similarities.cpu().tolist(), dtype=float),
    )


def _greedy_pairs(left, right, scores, count, *, descending=True, seed=None):
    if not len(scores) or count <= 0:
        return []
    if seed is not None:
        order = np.random.default_rng(seed).permutation(len(scores))
    else:
        order = np.argsort(-scores if descending else scores, kind="stable")
    used, selected = set(), []
    for candidate in order:
        a, b = int(left[candidate]), int(right[candidate])
        if a in used or b in used:
            continue
        selected.append((a, b, float(scores[candidate])))
        used.update((a, b))
        if len(selected) == count:
            break
    return selected


def _similarity_matched_pairs(left, right, scores, targets):
    """Greedily match a target cosine distribution without token reuse."""
    if not len(scores) or not len(targets):
        return []
    order = np.argsort(scores, kind="stable")
    ordered_scores = scores[order]
    used, selected = set(), []
    for target in sorted(np.asarray(targets, dtype=float), reverse=True):
        center = int(np.searchsorted(ordered_scores, target))
        radius = 0
        chosen = None
        while center - radius - 1 >= 0 or center + radius < len(order):
            probes = []
            if center - radius - 1 >= 0:
                probes.append(center - radius - 1)
            if center + radius < len(order):
                probes.append(center + radius)
            probes.sort(key=lambda position: abs(ordered_scores[position] - target))
            for position in probes:
                candidate = int(order[position])
                a, b = int(left[candidate]), int(right[candidate])
                if a not in used and b not in used:
                    chosen = (a, b, float(scores[candidate]), float(target))
                    break
            if chosen is not None:
                break
            radius += 1
        if chosen is None:
            break
        used.update(chosen[:2])
        selected.append(chosen)
    return selected


def _apply_pairs(values, traces, pairs):
    selected = {a: (a, b) for a, b, *_ in pairs}
    removed = {b for _, b, *_ in pairs}
    outputs, output_traces = [], []
    for index in range(len(values)):
        if index in removed:
            continue
        if index in selected:
            merged, trace = _weighted_group(values, traces, selected[index])
            outputs.append(merged)
            output_traces.append(trace)
        else:
            outputs.append(values[index])
            output_traces.append(traces[index])
    order = np.argsort([trace.center_ms for trace in output_traces], kind="stable")
    import torch
    return torch.stack([outputs[i] for i in order]), [output_traces[i] for i in order]


def substitute_tokens(values, traces, *, strategy, replace_ratio, window_ms, far_ms, seed):
    """Length-preserving donor substitution with a fixed receiver budget."""
    changed = values.clone()
    budget = max(1, int(round(len(values) * float(replace_ratio))))
    local_left, local_right, local_scores = [], [], []
    far_left, far_right, far_scores = [], [], []
    for chunk in _ordered_chunks(traces):
        indices = [i for i, trace in enumerate(traces) if trace.hard_chunks == (chunk,)]
        if len(indices) < 2:
            continue
        chunk_values = values[indices]
        centers = [traces[i].center_ms for i in indices]
        local_table = _candidate_table(chunk_values, centers, mode="local", window_ms=window_ms)
        far_table = _candidate_table(chunk_values, centers, mode="far", far_ms=far_ms)
        for destination, table in (((local_left, local_right, local_scores), local_table),
                                   ((far_left, far_right, far_scores), far_table)):
            destination[0].extend(indices[int(index)] for index in table[0])
            destination[1].extend(indices[int(index)] for index in table[1])
            destination[2].extend(float(score) for score in table[2])
    local_arrays = (np.asarray(local_left), np.asarray(local_right), np.asarray(local_scores))
    high = _greedy_pairs(*local_arrays, count=budget, descending=True)
    low = _greedy_pairs(*local_arrays, count=budget, descending=False)
    random_pairs = _greedy_pairs(*local_arrays, count=budget, seed=seed)
    far = _similarity_matched_pairs(
        np.asarray(far_left), np.asarray(far_right), np.asarray(far_scores),
        [pair[2] for pair in high])
    common_budget = min(len(high), len(low), len(random_pairs), len(far))
    if strategy == "local_high":
        selected = [(*pair, None) for pair in high[:common_budget]]
    elif strategy == "local_low":
        selected = [(*pair, None) for pair in low[:common_budget]]
    elif strategy == "random_local":
        selected = [(*pair, None) for pair in random_pairs[:common_budget]]
    elif strategy == "far_high":
        selected = far[:common_budget]
    else:
        raise ValueError(f"Unknown substitution strategy {strategy!r}")
    centers = np.asarray([trace.center_ms for trace in traces])
    pairs = [(a, b, similarity, abs(centers[b] - centers[a]), target)
             for a, b, similarity, target in selected]
    for donor, receiver, *_ in pairs:
        changed[receiver] = values[donor]
    target_similarity = [pair[4] for pair in pairs if pair[4] is not None]
    return changed, {
        "strategy": strategy,
        "requested_replacements": budget,
        "similarity_matched_common_budget": common_budget,
        "replaced_tokens": len(pairs),
        "replace_ratio_realized": len(pairs) / len(values),
        "pair_similarity_mean": float(np.mean([pair[2] for pair in pairs])) if pairs else None,
        "pair_distance_ms_mean": float(np.mean([pair[3] for pair in pairs])) if pairs else None,
        "similarity_match_mae": (float(np.mean([abs(pair[2] - pair[4]) for pair in pairs
                                                if pair[4] is not None])) if target_similarity else None),
        "replacement_pairs": [[int(a), int(b)] for a, b, *_ in pairs],
        "representation_relative_l2": relative_l2(changed, values),
        "representation_cosine_mean": cosine_mean(changed, values),
    }


def substitute_region(values, traces, labels, *, region, replace_ratio):
    """Replace a fixed budget of one region by its best immediate neighbour."""
    import torch

    labels = np.asarray(labels)
    candidates = np.flatnonzero(labels == region)
    budget = min(len(candidates), max(1, int(round(len(values) * float(replace_ratio)))))
    flat = values.detach().float().reshape(len(values), -1)
    pairs = []
    for receiver in candidates:
        chunk = traces[int(receiver)].hard_chunks
        donors = [index for index in (int(receiver) - 1, int(receiver) + 1)
                  if 0 <= index < len(values) and traces[index].hard_chunks == chunk]
        if not donors:
            continue
        similarities = [float(torch.nn.functional.cosine_similarity(
            flat[receiver:receiver + 1], flat[donor:donor + 1], dim=-1)[0]) for donor in donors]
        best = int(np.argmax(similarities))
        pairs.append((donors[best], int(receiver), similarities[best]))
    # Keep the region budget fixed and deterministic; high similarity is not used to choose the region.
    pairs = sorted(pairs, key=lambda pair: pair[1])
    if budget and len(pairs) > budget:
        selected_positions = np.linspace(0, len(pairs) - 1, budget).round().astype(int)
        pairs = [pairs[int(position)] for position in selected_positions]
    changed = values.clone()
    for donor, receiver, _ in pairs:
        changed[receiver] = values[donor]
    return changed, {
        "region": region,
        "eligible_tokens": int(len(candidates)),
        "requested_replacements": budget,
        "replaced_tokens": len(pairs),
        "pair_similarity_mean": float(np.mean([pair[2] for pair in pairs])) if pairs else None,
        "replacement_pairs": [[a, b] for a, b, _ in pairs],
        "representation_relative_l2": relative_l2(changed, values),
        "representation_cosine_mean": cosine_mean(changed, values),
    }


def predict_blocks(values, traces, *, block_ms, predictor, replace_ratio, seed):
    """Replace non-overlapping continuous blocks by predictions from outside them."""
    if predictor not in {"copy_left", "linear", "local_mean"}:
        raise ValueError(f"Unknown predictor {predictor!r}")
    changed = values.clone()
    available = []
    effective_widths = []
    for chunk in _ordered_chunks(traces):
        indices = [i for i, trace in enumerate(traces) if trace.hard_chunks == (chunk,)]
        if len(indices) < 3:
            continue
        centers = np.asarray([traces[i].center_ms for i in indices])
        step = float(np.median(np.diff(centers)))
        width = max(1, int(round(float(block_ms) / max(step, 1e-6))))
        effective_widths.append(width)
        for start in range(1, len(indices) - width, width):
            available.append(indices[start:start + width])
    token_budget = max(1, int(round(len(values) * float(replace_ratio))))
    order = np.random.default_rng(seed).permutation(len(available)) if available else []
    blocks, count = [], 0
    for candidate in order:
        block = available[int(candidate)]
        if count and count + len(block) > token_budget:
            continue
        blocks.append(block)
        count += len(block)
        if count >= token_budget:
            break
    selected = []
    for block in blocks:
        left, right = block[0] - 1, block[-1] + 1
        if predictor == "copy_left":
            predictions = values[left].unsqueeze(0).expand(len(block), *values.shape[1:])
        elif predictor == "local_mean":
            mean = (values[left].float() + values[right].float()) / 2
            predictions = mean.to(values.dtype).unsqueeze(0).expand(len(block), *values.shape[1:])
        else:
            fractions = np.arange(1, len(block) + 1, dtype=float) / (len(block) + 1)
            outputs = []
            for fraction in fractions:
                outputs.append(((1 - fraction) * values[left].float()
                                + fraction * values[right].float()).to(values.dtype))
            import torch
            predictions = torch.stack(outputs)
        changed[block] = predictions
        selected.extend(block)
    if selected:
        actual, reference = changed[selected], values[selected]
        rel_l2, cosine = relative_l2(actual, reference), cosine_mean(actual, reference)
    else:
        rel_l2 = cosine = None
    return changed, {
        "predictor": predictor,
        "requested_block_ms": float(block_ms),
        "effective_block_tokens": int(round(np.mean(effective_widths))) if effective_widths else None,
        "requested_tokens": token_budget,
        "predicted_tokens": len(selected),
        "blocks": [[int(block[0]), int(block[-1] + 1)] for block in blocks],
        "prediction_relative_l2": rel_l2,
        "prediction_cosine_mean": cosine,
        "representation_relative_l2": relative_l2(changed, values),
        "representation_cosine_mean": cosine_mean(changed, values),
    }


def locality_merge(values, traces, *, scope, merge_fraction, window_ms, far_ms):
    """One-shot local/global merge with global pair cosines matched to local pairs."""
    if scope not in {"local", "global"}:
        raise ValueError("scope must be local or global")
    requested = max(1, int(round(len(values) * float(merge_fraction))))
    local_left, local_right, local_scores = [], [], []
    far_left, far_right, far_scores = [], [], []
    for chunk in _ordered_chunks(traces):
        indices = [i for i, trace in enumerate(traces) if trace.hard_chunks == (chunk,)]
        if len(indices) < 2:
            continue
        centers = [traces[i].center_ms for i in indices]
        local_table = _candidate_table(values[indices], centers, mode="local", window_ms=window_ms)
        far_table = _candidate_table(values[indices], centers, mode="far", far_ms=far_ms)
        for destination, table in (((local_left, local_right, local_scores), local_table),
                                   ((far_left, far_right, far_scores), far_table)):
            destination[0].extend(indices[int(index)] for index in table[0])
            destination[1].extend(indices[int(index)] for index in table[1])
            destination[2].extend(float(score) for score in table[2])
    local_pairs = _greedy_pairs(
        np.asarray(local_left), np.asarray(local_right), np.asarray(local_scores),
        requested, descending=True)
    targets = [pair[2] for pair in local_pairs]
    matched_pairs = _similarity_matched_pairs(
        np.asarray(far_left), np.asarray(far_right), np.asarray(far_scores), targets)
    common_count = len(matched_pairs)
    if scope == "local":
        chosen = [(*pair, pair[2]) for pair in local_pairs[:common_count]]
    else:
        chosen = matched_pairs
    centers = np.asarray([trace.center_ms for trace in traces])
    selected_global = [(a, b, similarity, target, abs(centers[b] - centers[a]))
                       for a, b, similarity, target in chosen]
    pair_shape = [(a, b, similarity) for a, b, similarity, _, _ in selected_global]
    merged, output_traces = _apply_pairs(values, traces, pair_shape)
    return merged, output_traces, {
        "scope": scope,
        "requested_merges": requested,
        "similarity_matched_common_budget": common_count,
        "realized_merges": len(selected_global),
        "output_tokens": len(merged),
        "pair_similarity_mean": (float(np.mean([pair[2] for pair in selected_global]))
                                 if selected_global else None),
        "pair_distance_ms_mean": (float(np.mean([pair[4] for pair in selected_global]))
                                  if selected_global else None),
        **_distance_distribution([pair[4] for pair in selected_global]),
        "similarity_match_mae": (float(np.mean([abs(pair[2] - pair[3]) for pair in selected_global]))
                                 if selected_global else None),
        "merge_pairs": [[int(pair[0]), int(pair[1])] for pair in selected_global],
    }
