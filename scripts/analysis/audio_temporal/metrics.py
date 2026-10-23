"""Numerical measurements shared by the temporal-audio experiments."""
from __future__ import annotations

import math
from collections import defaultdict

import numpy as np


REGION_NAMES = ("silent", "stable", "transient", "boundary", "other")


def jsonable(value):
    """Convert NumPy/PyTorch-adjacent values into strict JSON values."""
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return jsonable(value.tolist())
    if isinstance(value, np.generic):
        return jsonable(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def finite_mean(values):
    values = np.asarray(values, dtype=float)
    valid = np.isfinite(values)
    return float(values[valid].mean()) if valid.any() else None


def _pair_indices(time_ms, chunk_ids, lag_ms):
    """Nearest forward pair for a physical lag, never crossing a hard chunk."""
    time_ms = np.asarray(time_ms, dtype=float)
    chunk_ids = np.asarray(chunk_ids)
    left, right = [], []
    for chunk in np.unique(chunk_ids):
        indices = np.flatnonzero(chunk_ids == chunk)
        if len(indices) < 2:
            continue
        times = time_ms[indices]
        steps = np.diff(times)
        positive = steps[steps > 0]
        tolerance = max(1e-6, 0.51 * float(np.median(positive))) if len(positive) else 1e-6
        for local_index, source in enumerate(indices[:-1]):
            target = times[local_index] + lag_ms
            insertion = int(np.searchsorted(times, target, side="left"))
            candidates = [candidate for candidate in (insertion - 1, insertion)
                          if candidate > local_index and candidate < len(indices)]
            if not candidates:
                continue
            best = min(candidates, key=lambda candidate: abs(times[candidate] - target))
            if abs(times[best] - target) <= tolerance:
                left.append(int(source))
                right.append(int(indices[best]))
    return np.asarray(left, dtype=int), np.asarray(right, dtype=int)


def _cosine_pairs(values, left, right):
    import torch

    if not len(left):
        return np.empty(0, dtype=float)
    flattened = values.detach().float().reshape(len(values), -1)
    left_index = torch.tensor(left.tolist(), dtype=torch.long, device=flattened.device)
    right_index = torch.tensor(right.tolist(), dtype=torch.long, device=flattened.device)
    a, b = flattened[left_index], flattened[right_index]
    numerator = (a * b).sum(-1)
    denominator = a.norm(dim=-1) * b.norm(dim=-1)
    result = numerator / denominator.clamp_min(1e-12)
    result[denominator == 0] = float("nan")
    return np.asarray(result.cpu().tolist(), dtype=float)


def lag_similarity(values, time_ms, chunk_ids, lags_ms):
    """Raw and token-axis-de-meaned lag cosine at physical millisecond lags."""
    import torch

    flattened = values.detach().float().reshape(len(values), -1)
    centered = flattened - flattened.mean(dim=0, keepdim=True)
    time_ms = np.asarray(time_ms, dtype=float)
    rows = []
    for requested in lags_ms:
        left, right = _pair_indices(time_ms, chunk_ids, float(requested))
        raw = _cosine_pairs(flattened, left, right)
        demeaned = _cosine_pairs(centered, left, right)
        distances = time_ms[right] - time_ms[left] if len(left) else np.empty(0)
        rows.append({
            "lag_ms": float(requested),
            "pairs": int(len(left)),
            "effective_lag_ms_mean": finite_mean(distances),
            "effective_lag_ms_std": (float(np.std(distances)) if len(distances) else None),
            "raw_cosine": finite_mean(raw),
            "demean_cosine": finite_mean(demeaned),
        })
    return rows


def input_region_labels(input_features, mel_length, *, mel_step_ms, chunk_mel_lengths):
    """Create non-circular region proxies from log-Mel activity and spectral flux.

    The MMSU task has no frame-level VAD/event labels.  These labels therefore use
    only the input log-Mel signal and are deliberately called proxies in reports.
    They must not be interpreted as ground-truth acoustic event annotations.
    """
    features = np.asarray(input_features.detach().float().cpu().tolist(), dtype=float)
    if features.ndim != 2:
        raise ValueError("Expected packed log-Mel features with rank 2")
    features = features[:, :int(mel_length)]
    activity = features.mean(axis=0)
    flux = np.r_[0.0, np.mean(np.abs(np.diff(features, axis=1)), axis=0)]
    if len(activity) >= 5:
        kernel = np.ones(5, dtype=float) / 5
        activity = np.convolve(activity, kernel, mode="same")
        flux = np.convolve(flux, kernel, mode="same")
    activity_cut = float(np.quantile(activity, 0.2))
    stable_cut = float(np.quantile(flux, 0.35))
    transient_cut = float(np.quantile(flux, 0.8))
    boundary_cut = float(np.quantile(flux, 0.95))
    labels = np.full(len(activity), "other", dtype="U10")
    labels[(activity <= activity_cut) & (flux < boundary_cut)] = "silent"
    labels[(activity > activity_cut) & (flux <= stable_cut)] = "stable"
    labels[(activity > activity_cut) & (flux > transient_cut)] = "transient"
    labels[flux > boundary_cut] = "boundary"

    offset = 0
    hard_boundaries = []
    for chunk_length in chunk_mel_lengths[:-1]:
        offset += int(chunk_length)
        hard_boundaries.append(offset)
        labels[max(0, offset - 2):min(len(labels), offset + 2)] = "boundary"
    return {
        "time_ms": np.arange(len(activity), dtype=float) * float(mel_step_ms),
        "labels": labels,
        "activity": activity,
        "spectral_flux": flux,
        "thresholds": {
            "activity_q20": activity_cut,
            "flux_q35": stable_cut,
            "flux_q80": transient_cut,
            "flux_q95": boundary_cut,
        },
        "hard_boundaries_ms": [float(index * mel_step_ms) for index in hard_boundaries],
    }


def map_labels(source_time_ms, source_labels, target_time_ms):
    """Nearest-time projection of input-derived labels onto a representation stage."""
    source_time_ms = np.asarray(source_time_ms, dtype=float)
    target_time_ms = np.asarray(target_time_ms, dtype=float)
    if not len(source_time_ms):
        return np.full(len(target_time_ms), "other", dtype="U10")
    positions = np.searchsorted(source_time_ms, target_time_ms, side="left")
    positions = np.clip(positions, 0, len(source_time_ms) - 1)
    previous = np.clip(positions - 1, 0, len(source_time_ms) - 1)
    use_previous = abs(source_time_ms[previous] - target_time_ms) < abs(
        source_time_ms[positions] - target_time_ms)
    positions[use_previous] = previous[use_previous]
    return np.asarray(source_labels)[positions]


def region_similarity(values, time_ms, chunk_ids, labels, *, lag_ms):
    """Adjacent physical-lag cosine grouped by the receiver's region label."""
    left, right = _pair_indices(time_ms, chunk_ids, float(lag_ms))
    raw = _cosine_pairs(values, left, right)
    flattened = values.detach().float().reshape(len(values), -1)
    centered = flattened - flattened.mean(0, keepdim=True)
    demeaned = _cosine_pairs(centered, left, right)
    labels = np.asarray(labels)
    grouped = defaultdict(lambda: {"raw": [], "demean": []})
    for index, source in enumerate(left):
        if labels[source] != labels[right[index]]:
            continue
        region = str(labels[source])
        grouped[region]["raw"].append(raw[index])
        grouped[region]["demean"].append(demeaned[index])
    return {
        region: {
            "pairs": len(grouped[region]["raw"]),
            "raw_cosine": finite_mean(grouped[region]["raw"]),
            "demean_cosine": finite_mean(grouped[region]["demean"]),
        }
        for region in REGION_NAMES
    }


def relative_l2(actual, reference):
    import torch

    actual = actual.detach().float()
    reference = reference.detach().float()
    return float((actual - reference).norm() / reference.norm().clamp_min(1e-12))


def cosine_mean(actual, reference):
    import torch

    actual = actual.detach().float().reshape(len(actual), -1)
    reference = reference.detach().float().reshape(len(reference), -1)
    return float(torch.nn.functional.cosine_similarity(actual, reference, dim=-1).mean())
