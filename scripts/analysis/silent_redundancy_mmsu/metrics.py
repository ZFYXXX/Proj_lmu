"""Acoustic alignment and compact per-audio observational statistics."""
from __future__ import annotations

import math
from typing import Iterable

import numpy as np
import torch
from torch.nn import functional as F

from . import ACOUSTIC_FEATURES


def jsonable(value):
    if isinstance(value, dict):
        return {str(key): jsonable(child) for key, child in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(child) for child in value]
    if isinstance(value, np.ndarray):
        return jsonable(value.tolist())
    if isinstance(value, np.generic):
        return jsonable(value.item())
    if isinstance(value, torch.Tensor):
        return jsonable(value.detach().cpu().numpy())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def encoder_layout(feature_length: int, n_window: int) -> tuple[list[int], list[int]]:
    """Return packed Mel chunk lengths and post-CNN token lengths."""
    if feature_length < 1 or n_window < 1:
        raise ValueError("feature_length and n_window must be positive")
    mel = [min(2 * n_window, feature_length - start)
           for start in range(0, feature_length, 2 * n_window)]
    return mel, [(length + 1) // 2 for length in mel]


@torch.no_grad()
def aligned_acoustic_features(
    input_features: torch.Tensor,
    feature_length: int,
    n_window: int,
) -> tuple[dict[str, torch.Tensor], list[int], np.ndarray, np.ndarray]:
    """Map exact model log-Mel input to Qwen encoder-token source windows.

    Qwen's two kernel-3 convolutions give each stride-2 encoder position a
    nominal five-Mel-frame receptive field. Windows are clipped per native
    attention chunk by replicate padding, so no statistic crosses a chunk.
    """
    if input_features.ndim != 2 or input_features.shape[1] < feature_length:
        raise ValueError("Expected [mel_bins, packed_time] input features")
    mel_lengths, token_lengths = encoder_layout(feature_length, n_window)
    chunks = {name: [] for name in ACOUSTIC_FEATURES}
    time_indices: list[torch.Tensor] = []
    mel_offset = 0
    token_offset = 0
    boundaries = [0]
    for mel_length, token_length in zip(mel_lengths, token_lengths):
        mel = input_features[:, mel_offset:mel_offset + mel_length].float()
        padded = F.pad(mel, (2, 2), mode="replicate")
        windows = padded.unfold(-1, 5, 2)
        if windows.shape[1] != token_length:
            raise RuntimeError("Five-frame acoustic windows do not match CNN tokens")
        chunks["intensity"].append(windows.mean(dim=(0, 2)))
        chunks["variation"].append(windows.std(dim=(0, 2), unbiased=False))
        chunks["temporal_change"].append(
            (windows[..., 1:] - windows[..., :-1]).abs().mean(dim=(0, 2))
        )
        time_indices.append(mel_offset + 2 * torch.arange(token_length, device=mel.device))
        mel_offset += mel_length
        token_offset += token_length
        boundaries.append(token_offset)
    values = {name: torch.cat(parts) for name, parts in chunks.items()}
    return values, token_lengths, torch.cat(time_indices).cpu().numpy(), np.asarray(boundaries)


def average_tie_ranks(values: np.ndarray) -> np.ndarray:
    """Zero-to-one average ranks, with exact ties sharing a rank."""
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all():
        raise ValueError("Ranks require a finite nonempty vector")
    order = np.argsort(values, kind="stable")
    ranks = np.empty(len(values), dtype=float)
    sorted_values = values[order]
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and sorted_values[end] == sorted_values[start]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + end - 1)
        start = end
    if len(values) == 1:
        return np.full(1, 0.5)
    return ranks / (len(values) - 1)


def chunkwise_ranks(values: np.ndarray, boundaries: Iterable[int]) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    bounds = np.asarray(list(boundaries), dtype=int)
    if bounds[0] != 0 or bounds[-1] != len(values) or np.any(np.diff(bounds) <= 0):
        raise ValueError("Invalid packed chunk boundaries")
    result = np.empty(len(values), dtype=float)
    for start, end in zip(bounds[:-1], bounds[1:]):
        result[start:end] = average_tie_ranks(values[start:end])
    return result


def correlation(x: np.ndarray, y: np.ndarray) -> float | None:
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    if len(x) != len(y) or len(x) < 2:
        return None
    x, y = x - x.mean(), y - y.mean()
    denominator = np.linalg.norm(x) * np.linalg.norm(y)
    return float(x @ y / denominator) if denominator > 1e-12 else None


def exact_chunk_topk_mask(
    scores: np.ndarray,
    boundaries: Iterable[int],
    keep_ratio: float,
    *,
    minimum_keep_tokens: int = 2,
) -> np.ndarray:
    """The platform's floor budget and stable descending order, per chunk."""
    if not 0 < keep_ratio <= 1:
        raise ValueError("keep_ratio must be in (0, 1]")
    scores = np.asarray(scores, dtype=float)
    bounds = np.asarray(list(boundaries), dtype=int)
    mask = np.zeros(len(scores), dtype=bool)
    for start, end in zip(bounds[:-1], bounds[1:]):
        length = end - start
        keep = min(length, max(minimum_keep_tokens, int(math.floor(length * keep_ratio))))
        indices = np.argsort(-scores[start:end], kind="stable")[:keep] + start
        mask[indices] = True
    return mask


def observational_summary(
    attention: np.ndarray,
    acoustic: dict[str, np.ndarray],
    boundaries: Iterable[int],
    *,
    keep_ratios: tuple[float, ...] = (0.75, 0.50, 0.25),
) -> dict:
    """Sufficient statistics for full-dataset plots without saving all tokens."""
    attention = np.asarray(attention, dtype=float)
    bounds = np.asarray(list(boundaries), dtype=int)
    attention_rank = chunkwise_ranks(attention, bounds)
    keep_masks = {ratio: exact_chunk_topk_mask(attention, bounds, ratio)
                  for ratio in keep_ratios}
    result = {"tokens": len(attention), "features": {}}
    for name in ACOUSTIC_FEATURES:
        values = np.asarray(acoustic[name], dtype=float)
        feature_rank = chunkwise_ranks(values, bounds)
        quantile = np.minimum((feature_rank * 5).astype(int), 4)
        quintiles = []
        for index in range(5):
            selected = quantile == index
            row = {
                "bin": index,
                "n": int(selected.sum()),
                "attention_sum": float(attention[selected].sum()),
            }
            for ratio, mask in keep_masks.items():
                row[f"kept_{int(round(100 * ratio))}"] = int((selected & mask).sum())
            quintiles.append(row)
        hist, _, _ = np.histogram2d(
            feature_rank,
            attention_rank,
            bins=(np.linspace(0, 1, 11), np.linspace(0, 1, 11)),
        )
        result["features"][name] = {
            "pearson": correlation(values, attention),
            "spearman": correlation(feature_rank, attention_rank),
            "quintiles": quintiles,
            "rank_histogram_10x10": hist.astype(int),
        }
    low = chunkwise_ranks(np.asarray(acoustic["temporal_change"], dtype=float), bounds) <= 0.25
    high = chunkwise_ranks(np.asarray(acoustic["temporal_change"], dtype=float), bounds) >= 0.75
    high_attention = attention_rank >= 0.75
    low_attention = attention_rank <= 0.25
    result["temporal_change_attention_quadrants"] = {
        "low_change_high_attention": int((low & high_attention).sum()),
        "low_change_low_attention": int((low & low_attention).sum()),
        "high_change_high_attention": int((high & high_attention).sum()),
        "high_change_low_attention": int((high & low_attention).sum()),
        "low_change_total": int(low.sum()),
        "high_change_total": int(high.sum()),
    }
    return result


def causal_drop_indices(
    temporal_change: np.ndarray,
    attention: np.ndarray,
    drop_count: int,
    mode: str,
    *,
    seed: int,
) -> np.ndarray:
    """Choose an exact, matched causal deletion set inside one chunk."""
    temporal_change = np.asarray(temporal_change, dtype=float)
    attention = np.asarray(attention, dtype=float)
    n = len(attention)
    if len(temporal_change) != n or not 0 <= drop_count <= n:
        raise ValueError("Invalid causal deletion inputs")
    if drop_count == 0:
        return np.empty(0, dtype=int)
    pool_size = max(drop_count, int(math.ceil(0.25 * n)))
    change_order = np.argsort(temporal_change, kind="stable")
    low_pool = change_order[:pool_size]
    high_pool = change_order[-pool_size:]
    if mode == "causal_low_change_high_attention":
        chosen = low_pool[np.argsort(-attention[low_pool], kind="stable")[:drop_count]]
    elif mode == "causal_low_change_low_attention":
        chosen = low_pool[np.argsort(attention[low_pool], kind="stable")[:drop_count]]
    elif mode == "causal_low_change_random":
        chosen = np.random.default_rng(seed).choice(low_pool, drop_count, replace=False)
    elif mode == "causal_high_change_attention_matched":
        targets = low_pool[np.argsort(-attention[low_pool], kind="stable")[:drop_count]]
        remaining = list(map(int, high_pool))
        matched = []
        for target in targets:
            best = min(remaining, key=lambda index: (abs(attention[index] - attention[target]), index))
            matched.append(best)
            remaining.remove(best)
        chosen = np.asarray(matched, dtype=int)
    else:
        raise ValueError(f"Unknown causal selector: {mode}")
    return np.sort(np.asarray(chosen, dtype=int))
