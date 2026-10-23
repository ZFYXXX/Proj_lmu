"""Deterministic spatial-frequency interventions and summary-relative selectors.

Spatial DCT acts on the actual H x W token lattice, not the channel axis.
No quantity in this module reads the question answer or model predictions.
"""

from __future__ import annotations

import math
import random

from mllm_comp.analysis.common.evidence_interaction import SEEDS, digest, farthest_indices
from mllm_comp.analysis.common.visual_merge_attribution import grid_partition


def dct_matrix(length, *, device=None):
    import torch
    if length < 1:
        raise ValueError("DCT length must be positive.")
    k = torch.arange(length, device=device, dtype=torch.float32)[:, None]
    x = torch.arange(length, device=device, dtype=torch.float32)[None, :]
    matrix = torch.cos(math.pi / length * (x + .5) * k) * math.sqrt(2 / length)
    matrix[0] /= math.sqrt(2)
    return matrix


def spatial_dct(features, height, width):
    import torch
    if features.ndim != 2 or len(features) != height * width or not torch.isfinite(features).all():
        raise ValueError("Finite rectangular feature grid required.")
    left, right = dct_matrix(height, device=features.device), dct_matrix(width, device=features.device)
    grid = features.float().reshape(height, width, -1)
    coefficients = torch.einsum("ah,hwd,bw->abd", left, grid, right)
    return coefficients, left, right


def inverse_dct(coefficients, left, right):
    import torch
    return torch.einsum("ah,abd,bw->hwd", left, coefficients, right).flatten(0, 1)


def spectral_plan(features, height, width, fingerprint, fractions=(.05, .10)):
    """Attenuate low/high/random AC bands by identical per-image energy.

    DC is never changed. E = ||delta X||_F^2 is matched, not the change in
    ||X||_F^2; cross terms make these distinct quantities. Every band contains
    the same number of coefficients. A common cap prevents amplification.
    """
    import torch
    coefficients, left, right = spatial_dct(features, height, width)
    order = sorted(range(1, height * width), key=lambda i: ((i // width / height) ** 2 + (i % width / width) ** 2, i))
    count = max(1, len(order) // 4)
    bands = {"low": order[:count], "high": order[-count:]}
    for seed in SEEDS:
        bands[f"random_{seed}"] = sorted(random.Random(f"{seed}:{fingerprint}:spectral-v1").sample(order, count))
    flat = coefficients.flatten(0, 1)
    energies = {name: float(flat[indices].square().sum()) for name, indices in bands.items()}
    ac_energy = float(flat[1:].square().sum())
    if ac_energy <= 0 or min(energies.values()) <= 0:
        raise ValueError("Degenerate AC/band energy; cannot perform matched frequency intervention.")
    states = []
    for fraction in fractions:
        if not 0 < fraction < 1:
            raise ValueError("Frequency dose must be between zero and one.")
        # Scale both doses together when the weakest band caps the maximum,
        # preserving a genuine 1:2 dose ratio rather than duplicating states.
        max_fraction = max(fractions)
        energy = fraction / max_fraction * min(max_fraction * ac_energy, .5 * min(energies.values()))
        for name, indices in bands.items():
            scale = math.sqrt(energy / energies[name])
            sparse = torch.zeros_like(flat)
            sparse[indices] = flat[indices] * scale
            delta = inverse_dct(sparse.reshape_as(coefficients), left, right)
            measured = float(delta.square().sum())
            if not math.isclose(measured, energy, rel_tol=2e-4, abs_tol=1e-5):
                raise ValueError("DCT intervention energy did not close.")
            states.append({"name": f"spectral/{fraction:g}/{name}", "values": features.float() - delta,
                           "trace": {"band": name, "requested_ac_fraction": fraction,
                                     "actual_ac_fraction": energy / ac_energy, "perturbation_energy": measured,
                                     "band_energy": energies[name], "attenuation": scale,
                                     "frequency_indices": indices, "dc_preserved": True}})
    return states


def rank_indices(scores, count, eligible, *, descending=True):
    pool = list(eligible)
    if count > len(pool) or len(set(pool)) != len(pool) or count < 0:
        raise ValueError("Invalid selector pool/budget.")
    return sorted(pool, key=lambda i: ((-1 if descending else 1) * float(scores[i]), i))[:count]


def summary_plan(features, height, width, attention, fingerprint, *, base_count=96, extra_count=32):
    """Common grid summary + exact-count original tokens under native positions.

    All hybrid conditions share values, positions, and 96 summary slots. Only
    their 32 extra original tokens differ. Native representative positions
    are used for both mean and representative ablations.
    """
    import torch
    from mllm_comp.compressors.image.fixed_visual_merge import fixed_visual_merge, FixedVisualMergeSpec
    if base_count + extra_count > len(features):
        raise ValueError("Source has fewer tokens than requested budget.")
    merged = fixed_visual_merge(features, grid_height=height, grid_width=width,
                                spec=FixedVisualMergeSpec(partition="grid", target_count=base_count))
    members = [list(group) for group in merged.metadata["cluster_members"]]
    representatives = merged.indices.tolist()
    reconstruction = torch.empty_like(features, dtype=torch.float32)
    cells = [None] * len(features)
    for cell, group in enumerate(members):
        reconstruction[group] = merged.tokens[cell].float()
        for i in group:
            cells[i] = cell
    local_residual = (features.float() - reconstruction).square().sum(-1)
    centered = features.float() - features.float().mean(0, keepdim=True)
    # Exact reduced SVD in FP32: deterministic, no randomized PCA seed.
    _, _, vectors = torch.linalg.svd(centered, full_matrices=False)
    basis = vectors[:min(8, vectors.shape[0])]
    pca_residual = (centered - (centered @ basis.T) @ basis).square().sum(-1).clamp_min(0)
    excluded = set(representatives)
    eligible = [i for i in range(len(features)) if i not in excluded]
    high = rank_indices(local_residual, extra_count, eligible)
    selections = {"high_residual": high,
                  "low_residual": rank_indices(local_residual, extra_count, eligible, descending=False),
                  "attention": rank_indices(attention, extra_count, eligible),
                  "pca_residual": rank_indices(pca_residual, extra_count, eligible),
                  "diversity": farthest_indices(features, extra_count, eligible)}
    for seed in SEEDS:
        rng = random.Random(f"{seed}:{fingerprint}:summary-extra-v1")
        selections[f"random_{seed}"] = rng.sample(eligible, extra_count)
        # Exact per-cell count match controls local spatial allocation.
        selected = []
        for cell, group in enumerate(members):
            amount = sum(cells[i] == cell for i in high)
            selected.extend(rng.sample([i for i in group if i not in excluded], amount))
        selections[f"spatial_random_{seed}"] = selected
    for name, selected in selections.items():
        if len(selected) != extra_count or len(set(selected)) != extra_count or excluded.intersection(selected):
            raise ValueError(f"Invalid hybrid membership: {name}.")
    return {"base_indices": representatives, "base_values": merged.tokens,
            "members": members, "reconstruction": reconstruction,
            "residual_scores": local_residual, "pca_scores": pca_residual,
            "selections": selections, "cells": cells,
            "plan_hash": digest(str((height, width, members, selections)))}


def tile_plan(height, width):
    return [list(group) for group in grid_partition(height, width, 16).members]
