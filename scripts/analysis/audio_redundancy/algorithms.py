"""Numerical experiments on valid native tensors, with explicit time budgets."""
from __future__ import annotations

import numpy as np
import torch


def spectrum_stats(singular, rows, columns, *, centered):
    s = np.asarray(singular, dtype=float)
    capacity = min(max(0, rows - int(centered)), columns)
    energy = float(s @ s)
    result = dict(tokens=rows, channels=columns, rank_capacity=capacity,
                  has_variance=energy > 1e-20, effective_rank=None,
                  normalized_effective_rank=None, participation_ratio=None,
                  k90=None, k95=None)
    if energy <= 1e-20 or capacity == 0:
        return result
    p = s[s > 0] / s.sum()
    effective = float(np.exp(-(p * np.log(p)).sum()))
    cumulative = np.cumsum(s * s) / energy
    result.update(effective_rank=effective, normalized_effective_rank=effective / capacity,
                  participation_ratio=energy ** 2 / float(np.sum(s ** 4)),
                  k90=int(np.searchsorted(cumulative, .90) + 1),
                  k95=int(np.searchsorted(cumulative, .95) + 1))
    return result


def rank_measurement(hidden, *, centered=True):
    x = hidden.detach().float()
    if centered:
        x = x - x.mean(0, keepdim=True)
    s = torch.linalg.svdvals(x).cpu().numpy()
    return spectrum_stats(s, *x.shape, centered=centered), s


def local_ranks(hidden, time_ms, chunks, windows_ms):
    """Nonoverlapping physical windows; do not treat encoder chunk resets as changes."""
    result = []
    time_ms, chunks = np.asarray(time_ms), np.asarray(chunks)
    for width in windows_ms:
        for chunk in np.unique(chunks):
            indices = np.flatnonzero(chunks == chunk)
            bins = np.floor((time_ms[indices] - time_ms[indices[0]]) / width).astype(int)
            for window in np.unique(bins):
                group = indices[bins == window]
                if len(group) < 2:
                    continue
                stats, _ = rank_measurement(hidden[group.tolist()])
                result.append(dict(window_ms=width, center_ms=float(time_ms[group].mean()),
                                   chunk=int(chunk), **stats))
    return result


def subspace(hidden, k, method, *, seed=42, basis=None):
    x = hidden.detach().float()
    if method == "identity":
        return hidden, dict(requested_k=None, retained_k=min(x.shape), relative_l2=0.)
    k = min(int(k), max(0, len(x) - 1), x.shape[1])
    mean = x.mean(0, keepdim=True)
    centered = x - mean
    if k == 0:
        out = mean.expand_as(x)
    else:
        if method == "pca":
            if basis is None:
                basis = torch.linalg.svd(centered, full_matrices=False).Vh.T
            q = basis[:, :k]
        elif method == "random":
            generator = torch.Generator(device="cpu").manual_seed(seed)
            q = torch.randn(x.shape[1], k, generator=generator).to(x.device)
            q = torch.linalg.qr(q, mode="reduced").Q
        else:
            raise ValueError(method)
        out = mean + (centered @ q) @ q.T
    return out.to(hidden.dtype), dict(retained_k=k,
        relative_l2=float((out - x).norm() / x.norm().clamp_min(1e-12)),
        pca_basis_scope="current_audio" if method == "pca" else "random_subspace")


def chunk_ranges(chunks):
    chunks = np.asarray(chunks)
    starts = np.r_[0, np.flatnonzero(chunks[1:] != chunks[:-1]) + 1]
    return list(zip(starts.tolist(), np.r_[starts[1:], len(chunks)].tolist()))


def feasible_budget(chunks, ratio):
    requested = max(1, int(np.floor(len(chunks) * ratio + .5)))
    realized = max(requested, len(chunk_ranges(chunks)))
    return requested, realized


def uniform_groups(chunks, budget):
    """Allocate exactly B bins by longest current interval per chunk."""
    ranges = chunk_ranges(chunks)
    lengths = np.array([b - a for a, b in ranges])
    if not len(ranges) <= budget <= sum(lengths):
        raise ValueError("Budget must fit all hard chunks and native tokens")
    counts = np.ones(len(ranges), dtype=int)
    while counts.sum() < budget:
        priorities = np.where(counts < lengths, lengths / counts, -np.inf)
        counts[int(np.argmax(priorities))] += 1
    return [group for (start, end), count in zip(ranges, counts)
            for group in np.array_split(np.arange(start, end), int(count))]


def score_groups(chunks, budget, scores):
    """Mandatory hard boundaries + highest-scoring B-C internal boundaries."""
    ranges = chunk_ranges(chunks)
    if not len(ranges) <= budget <= len(chunks):
        raise ValueError("Invalid segment budget")
    candidates = np.array([i for start, end in ranges for i in range(start + 1, end)], dtype=int)
    scores = np.asarray(scores, dtype=float)
    order = np.argsort(-scores[candidates], kind="stable")
    selected = candidates[order[:budget - len(ranges)]]
    starts = sorted({a for a, _ in ranges} | set(selected.tolist()))
    return [np.arange(a, b) for a, b in zip(starts, starts[1:] + [len(chunks)])]


def change_scores(hidden, method):
    x = hidden.detach().float()
    if method == "l2":
        values = (x[1:] - x[:-1]).norm(dim=1)
    elif method == "cosine":
        # Zero vectors compare consistently: identical zeros have no change.
        norms = x.norm(dim=1)
        values = 1 - torch.nn.functional.cosine_similarity(x[1:], x[:-1], dim=1)
        values[(norms[1:] == 0) & (norms[:-1] == 0)] = 0
    else:
        raise ValueError(method)
    return np.r_[0., values.cpu().numpy()]


def run_groups(codes, chunks):
    starts = np.r_[0, np.flatnonzero((codes[1:] != codes[:-1]) | (chunks[1:] != chunks[:-1])) + 1]
    return [np.arange(a, b) for a, b in zip(starts, np.r_[starts[1:], len(codes)])]


def pool_groups(hidden, groups, *, broadcast=False, sample=False):
    x = hidden.detach().float()
    pooled = torch.stack([x[int(g[(len(g) - 1) // 2])] if sample else x[g.tolist()].mean(0)
                          for g in groups])
    restored = torch.empty_like(x)
    for row, group in zip(pooled, groups):
        restored[group.tolist()] = row
    error = float((restored - x).norm() / x.norm().clamp_min(1e-12))
    return (restored if broadcast else pooled).to(hidden.dtype), error


def discrete_measurement(hidden, chunks, model, duration_seconds, *, seed=42):
    """Held-out codes, Markov entropy and realized surprisal (bits)."""
    x = hidden.detach().float().cpu().numpy()
    normalized = x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)
    z = (normalized - model["mean"]) @ model["components"].T
    centers = model["centers"]
    # Bounded temporary distance matrices for long speech.
    code_parts = []
    for start in range(0, len(z), 2048):
        batch = z[start:start + 2048]
        distances = (batch * batch).sum(1, keepdims=True) + (centers * centers).sum(1) - 2 * batch @ centers.T
        code_parts.append(distances.argmin(1))
    codes = np.concatenate(code_parts)
    starts = np.r_[True, chunks[1:] != chunks[:-1]]
    previous = np.r_[codes[0], codes[:-1]]
    probabilities = model["transition"][previous].copy()
    probabilities[starts] = model["prior"]
    entropy = -(probabilities * np.log2(probabilities)).sum(1)
    surprisal = -np.log2(probabilities[np.arange(len(codes)), codes])
    groups = run_groups(codes, chunks)
    lengths = np.array([len(g) for g in groups])
    valid = ~starts
    k = len(centers)
    joint = np.bincount(previous[valid] * k + codes[valid], minlength=k * k).reshape(k, k)
    marginal = joint.sum(1)
    nonzero = joint > 0
    conditional = joint / np.maximum(marginal[:, None], 1)
    empirical_entropy = (float(-(joint[nonzero] / joint.sum() * np.log2(conditional[nonzero])).sum())
                         if joint.sum() else None)
    rng = np.random.default_rng(seed)
    shuffled = codes.copy()
    for start, end in chunk_ranges(chunks):
        shuffled[start:end] = rng.permutation(shuffled[start:end])
    shuffled_runs = len(run_groups(shuffled, chunks))
    edges = [1, 2, 3, 5, 9, 17, 33, 65, 129, np.inf]
    return dict(codes=codes, entropy=entropy, surprisal=surprisal, groups=groups,
                stats=dict(codebook_size=k, pca_dimensions=z.shape[1],
                    quantization_mse=float(np.mean((z - centers[codes]) ** 2)),
                    used_clusters=int(len(np.unique(codes))), unique_clusters_per_second=len(np.unique(codes)) / duration_seconds,
                    mean_run_length=float(lengths.mean()), median_run_length=float(np.median(lengths)),
                    run_count=len(groups), runs_per_second=len(groups) / duration_seconds,
                    collapse_retention=len(groups) / len(codes),
                    transition_entropy_bits=empirical_entropy,
                    predictive_entropy_bits=float(entropy.mean()),
                    surprisal_bits=float(surprisal.mean()), entropy_bits_per_second=float(entropy.sum() / duration_seconds),
                    next_code_accuracy=float(np.mean(probabilities.argmax(1)[valid] == codes[valid])) if valid.any() else None,
                    shuffled_mean_run_length=len(codes) / shuffled_runs,
                    run_length_bin_labels=["1", "2", "3-4", "5-8", "9-16", "17-32", "33-64", "65-128", "129+"],
                    run_length_histogram=np.histogram(lengths, bins=edges)[0].tolist()))


def mel_reconstruction(mel, k, *, decomposition=None):
    """mel is time x frequency in the model's native log-Mel domain."""
    x = mel.detach().float()
    mean = x.mean(0, keepdim=True)
    centered = x - mean
    u, s, vh = decomposition or torch.linalg.svd(centered, full_matrices=False)
    retained = min(k, len(s))
    out = mean + (u[:, :retained] * s[:retained]) @ vh[:retained]
    total = float(s.square().sum())
    return out, dict(retained_residual_rank=retained,
                     residual_squared_norm_retention=float(s[:retained].square().sum()) / total if total else None,
                     mse=float((out - x).square().mean()))


def matched_smoothing(mel, target_mse, axis):
    # Reflective Gaussian smoothing via an orthonormal DCT. Parseval makes
    # sigma search cheap even for long audio: one DCT and one inverse, rather
    # than repeatedly applying a very wide time-domain convolution kernel.
    from scipy.fft import dct, idct
    x = mel.detach().float().cpu().numpy()
    if target_mse <= 1e-16:
        return mel.clone(), dict(sigma=0., mse=0., mse_match_error=-target_mse, matched=True)
    coefficients = dct(x, type=2, axis=axis, norm="ortho")
    energy = np.square(coefficients.astype(np.float64)).sum(axis=1 - axis) / x.size
    frequencies = np.pi * np.arange(x.shape[axis], dtype=float) / x.shape[axis]
    maximum = float(energy[1:].sum())

    def weights(sigma):
        return np.exp(-.5 * np.square(frequencies * sigma))

    low, high = 0., float(2 * x.shape[axis])
    if maximum <= target_mse:
        gain = np.zeros(x.shape[axis])
        gain[0] = 1.
        sigma = None  # Infinite-smoothing limit, not a finite fitted sigma.
    else:
        for _ in range(24):
            sigma = (low + high) / 2
            if float(np.sum(energy * np.square(1 - weights(sigma)))) < target_mse:
                low = sigma
            else:
                high = sigma
        sigma = (low + high) / 2
        gain = weights(sigma)
    shape = [1, 1]
    shape[axis] = len(gain)
    out = idct(coefficients * gain.reshape(shape), type=2, axis=axis, norm="ortho").astype(np.float32)
    mse = float(np.mean((out - x) ** 2))
    return torch.as_tensor(out, device=mel.device, dtype=mel.dtype), dict(
        sigma=sigma, kernel="reflective_DCT_Gaussian", mse=mse, target_mse=target_mse, mse_match_error=mse - target_mse,
        matched=abs(mse - target_mse) <= max(1e-7, target_mse * .01))
