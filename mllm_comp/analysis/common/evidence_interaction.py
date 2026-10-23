"""Answer-blind, equal-budget factorial interventions and image-level statistics."""

from __future__ import annotations

from itertools import combinations
import hashlib
import math
import random
from typing import Any, Mapping, Sequence


SEEDS = (11, 23, 47)


def digest(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def farthest_indices(features: Any, count: int, eligible: Sequence[int] | None = None) -> list[int]:
    import torch
    pool = sorted(range(len(features)) if eligible is None else eligible)
    if count < 0 or count > len(pool) or len(pool) != len(set(pool)):
        raise ValueError("Invalid farthest-point budget or duplicate eligible indices.")
    if count == 0:
        return []
    values = torch.nn.functional.normalize(features[pool].detach().float(), dim=-1, eps=1e-12)
    chosen = [0]
    best = values @ values[0]
    used = torch.zeros(len(pool), device=values.device, dtype=torch.bool)
    used[0] = True
    while len(chosen) < count:
        index = int(best.masked_fill(used, float("inf")).argmin().item())
        chosen.append(index)
        used[index] = True
        best = torch.maximum(best, values @ values[index])
    return [pool[i] for i in chosen]


def plan_regions(*, features: Any, scores: Any, height: int, width: int,
                 sample_id: str, seed: int) -> dict[str, Any]:
    """Eight candidates and eight matched controls, each exactly four tokens."""
    import torch
    if len(features) != height * width or scores.shape != (height * width,):
        raise ValueError("Features/scores do not match the actual post-merger grid.")
    if not bool(torch.isfinite(features).all()) or not bool(torch.isfinite(scores).all()):
        raise ValueError("Nonfinite selector input.")
    # A block must be entirely within its quadrant; odd edge rows/columns
    # remain eligible background tokens rather than being duplicated.
    blocks, quadrants = [], [[] for _ in range(4)]
    for r in range(0, height - 1, 2):
        for c in range(0, width - 1, 2):
            qr = int(r >= height // 2)
            qc = int(c >= width // 2)
            if r < height // 2 <= r + 1 or c < width // 2 <= c + 1:
                continue
            block = [r * width + c, r * width + c + 1,
                     (r + 1) * width + c, (r + 1) * width + c + 1]
            quadrants[2 * qr + qc].append(len(blocks))
            blocks.append(block)
    if any(len(pool) < 4 for pool in quadrants):
        raise ValueError("Grid cannot supply four disjoint 2x2 blocks per quadrant.")
    norms = features.detach().float().norm(dim=-1)
    block_scores = [float(scores[b].float().mean()) for b in blocks]
    stats = torch.stack([torch.stack((norms[b].mean(), norms[b].std(unbiased=False))) for b in blocks])
    scale = stats.std(dim=0, unbiased=False).clamp_min(1e-6)
    rng = random.Random(f"{seed}:{sample_id}:evidence-regions-v1")
    question = [min(pool, key=lambda i: (-block_scores[i], i)) for pool in quadrants]
    randoms = [rng.choice([i for i in pool if i != question[q]]) for q, pool in enumerate(quadrants)]
    candidates = question + randoms
    occupied = set(candidates)
    controls, distances = [], []
    for slot, candidate in enumerate(candidates):
        pool = [i for i in quadrants[slot % 4] if i not in occupied]
        ranked = sorted(pool, key=lambda i: (float(((stats[i] - stats[candidate]) / scale).square().sum()), i))
        selected = rng.choice(ranked[:8])
        occupied.add(selected)
        controls.append(selected)
        distances.append(float(((stats[selected] - stats[candidate]) / scale).square().sum().sqrt()))
    return {"seed": seed, "height": height, "width": width,
            "candidate_regions": [blocks[i] for i in candidates],
            "control_regions": [blocks[i] for i in controls],
            "candidate_kinds": ["question"] * 4 + ["random"] * 4,
            "candidate_scores": [block_scores[i] for i in candidates],
            "control_match_distances": distances,
            "selection_uses_answer": False, "question_score_source": "prompt_question_layer0_attention"}


def materialize_plan(regions: Mapping[str, Any], *, features: Any, budget: int) -> dict[str, Any]:
    candidates = regions["candidate_regions"]
    controls = regions["control_regions"]
    all_regions = candidates + controls
    flat = [int(i) for b in all_regions for i in b]
    if len(candidates) != 8 or len(controls) != 8 or any(len(b) != 4 for b in all_regions):
        raise ValueError("Expected eight four-token candidates and controls.")
    if len(flat) != len(set(flat)) or any(i < 0 or i >= len(features) for i in flat):
        raise ValueError("Overlapping or out-of-range region members.")
    reserved = set(flat)
    core = farthest_indices(features, budget - 32, [i for i in range(len(features)) if i not in reserved])
    states = []
    for changed in [()] + [(i,) for i in range(8)] + list(combinations(range(8), 2)):
        members = core + [i for slot in range(8) for i in (candidates[slot] if slot in changed else controls[slot])]
        if len(members) != budget or len(set(members)) != budget:
            raise ValueError("Factorial state violated the exact token budget.")
        states.append({"condition_id": condition_id(changed), "changed_slots": list(changed),
                       "retained_indices": sorted(members)})
    return {**dict(regions), "budget": budget, "core_indices": sorted(core), "states": states}


def condition_id(changed: Sequence[int]) -> str:
    return "00" if not changed else "+".join(str(i) for i in sorted(changed))


def interactions(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    states = {row["condition_id"]: row for row in rows}
    if set(states) != {condition_id(c) for c in [()] + [(i,) for i in range(8)] + list(combinations(range(8), 2))}:
        raise ValueError("Incomplete or duplicate factorial conditions.")
    if len(rows) != 37:
        raise ValueError("Exactly 37 unique states required.")
    result = []
    for a, b in combinations(range(8), 2):
        zero, one, two, both = (states[condition_id(c)] for c in [(), (a,), (b,), (a, b)])
        score = lambda row: float(row["primary_task_score"])
        value = score(both) - score(one) - score(two) + score(zero)
        nll = -both["candidate_answer_nll"] + one["candidate_answer_nll"] + two["candidate_answer_nll"] - zero["candidate_answer_nll"]
        result.append({"a": a, "b": b,
                       "kind": "question" if b < 4 else "random" if a >= 4 else "mixed",
                       "score_interaction": value, "logprob_interaction": nll,
                       "joint_only_improvement": score(one) <= score(zero) and score(two) <= score(zero) and score(both) > score(zero),
                       "joint_score_gain": score(both) - score(zero)})
    return result


def bootstrap(values: Sequence[float], *, seed: int = 42, repeats: int = 10000) -> dict[str, Any]:
    import numpy as np
    x = np.asarray(values, dtype=float)
    if not len(x) or not np.isfinite(x).all():
        raise ValueError("Bootstrap requires nonempty finite image-level values.")
    rng = np.random.default_rng(seed)
    means = np.concatenate([x[rng.integers(0, len(x), size=(min(200, repeats - i), len(x)))].mean(1)
                            for i in range(0, repeats, 200)])
    mean = float(x.mean())
    # Null-centered image bootstrap: one-sided H0 E[X] <= 0.
    p = float((1 + ((means - mean) >= mean).sum()) / (repeats + 1))
    sd = float(x.std(ddof=1)) if len(x) > 1 else 0.0
    return {"n_images": len(x), "mean": mean,
            "ci95": [float(v) for v in np.quantile(means, [.025, .975])],
            "one_sided_p": p, "mde80_normal_approx": 2.80 * sd / math.sqrt(len(x))}


def holm(pvalues: Sequence[float]) -> list[float]:
    order = sorted(range(len(pvalues)), key=lambda i: (pvalues[i], i))
    adjusted = [0.0] * len(order)
    previous = 0.0
    for rank, index in enumerate(order):
        previous = max(previous, min(1.0, (len(order) - rank) * pvalues[index]))
        adjusted[index] = previous
    return adjusted
