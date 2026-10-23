"""Controlled joint-deletion plans and spatial displays for discovery runs."""

from collections import Counter, defaultdict
from pathlib import Path
import math
import random
from typing import Any, Mapping, Sequence


def size_matched_random_units(
    members: Sequence[Sequence[int]], selected: Sequence[int], *, seed: str,
) -> tuple[int, ...]:
    """Match the entire unit-size histogram, hence exactly match token budget."""
    selected = tuple(int(value) for value in selected)
    if len(set(selected)) != len(selected) or any(i < 0 or i >= len(members) for i in selected):
        raise ValueError("Selected units must be unique valid indices.")
    pools: dict[int, list[int]] = defaultdict(list)
    for index, group in enumerate(members):
        if not group:
            raise ValueError("Intervention units cannot be empty.")
        pools[len(group)].append(index)
    counts = Counter(len(members[index]) for index in selected)
    rng = random.Random(seed)
    return tuple(sorted(index for size in sorted(counts) for index in rng.sample(pools[size], counts[size])))


def joint_deletion_plan(
    *, effects: Sequence[float], members: Sequence[Sequence[int]],
    scores: Mapping[str, Sequence[float]], fractions: Sequence[float],
    seeds: Sequence[int], sample_id: str,
    prompt_attention_key: str = "text_visual_attention_layer_0_prompt",
) -> list[dict[str, Any]]:
    if len(effects) != len(members) or not effects or any(not math.isfinite(v) for v in effects):
        raise ValueError("Finite singleton effects must match the intervention partition.")
    rankings = {
        "singleton_oracle_abs_low": ([abs(value) for value in effects], False),
        "singleton_oracle_high": (effects, True),
        "feature_diversity_low": (scores["feature_diversity"], False),
        "prompt_attention_low": (scores[prompt_attention_key], False),
    }
    pools: dict[int, list[int]] = defaultdict(list)
    for index, group in enumerate(members):
        if not group:
            raise ValueError("Intervention units cannot be empty.")
        pools[len(group)].append(index)
    plan = []
    for fraction in fractions:
        if not 0 < fraction < 1:
            raise ValueError("Joint deletion fractions must lie strictly between zero and one.")
        count = max(1, min(len(members) - 1, int(round(len(members) * fraction))))
        # A common, proportional size quota makes all ranked and random
        # conditions at this fraction use exactly the same actual token budget.
        quotas = {size: count * len(pool) // len(members) for size, pool in pools.items()}
        remainders = sorted(pools, key=lambda size: (-(count * len(pools[size]) % len(members)), size))
        for size in remainders[:count - sum(quotas.values())]:
            quotas[size] += 1
        for name, (values, descending) in rankings.items():
            if len(values) != len(members) or any(not math.isfinite(v) for v in values):
                raise ValueError("Rankings must contain one finite score per intervention unit.")
            selected = tuple(sorted(
                index for size, pool in pools.items()
                for index in sorted(pool, key=lambda i: ((-values[i] if descending else values[i]), i))[:quotas[size]]
            ))
            plan.append({"ranking": name, "fraction_units": fraction, "units": selected, "random_seed": None, "matched_to": None, "size_quota": quotas})
            for seed in seeds:
                matched = size_matched_random_units(members, selected, seed=f"{sample_id}:{name}:{fraction}:{seed}")
                plan.append({"ranking": "size_matched_random", "fraction_units": fraction, "units": matched, "random_seed": seed, "matched_to": name, "size_quota": quotas})
    return plan


def save_region_map(
    *, media: Any, members: Sequence[Sequence[int]], effects: Sequence[float],
    height: int, width: int, path: str, title: str,
) -> dict[str, Any]:
    """Map post-merger row-major cells onto the original image after resize."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    import numpy as np
    from PIL import Image

    values = media if isinstance(media, (tuple, list)) else [media]
    if len(values) != 1 or not isinstance(values[0], Image.Image):
        raise ValueError("Region maps require exactly one decoded PIL image.")
    image = values[0]
    heatmap = np.zeros((height, width), dtype=np.float32)
    coverage = np.zeros((height, width), dtype=np.int64)
    boxes = []
    for unit, group in enumerate(members):
        rows, cols = zip(*(divmod(index, width) for index in group))
        for row, col in zip(rows, cols):
            heatmap[row, col] = effects[unit]
            coverage[row, col] += 1
        boxes.append([min(cols) / width, min(rows) / height, (max(cols) + 1) / width, (max(rows) + 1) / height])
    if not bool((coverage == 1).all()):
        raise ValueError("Region map partition must cover every grid cell exactly once.")
    limit = max(1e-8, float(np.abs(heatmap).max()))
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    for axis in axes:
        axis.imshow(image, extent=(0, 1, 1, 0))
        axis.set_xlim(0, 1)
        axis.set_ylim(1, 0)
        axis.set_aspect(image.height / image.width)
        axis.set_axis_off()
    axes[0].set_title("Original image")
    artist = axes[1].imshow(heatmap, cmap="coolwarm", vmin=-limit, vmax=limit, alpha=.65, interpolation="nearest", extent=(0, 1, 1, 0))
    top = max(range(len(effects)), key=lambda index: effects[index])
    x0, y0, x1, y1 = boxes[top]
    axes[1].add_patch(Rectangle((x0, y0), x1-x0, y1-y0, fill=False, edgecolor="yellow", linewidth=2))
    axes[1].set_title(f"Single-unit deletion: max unit {top}, delta NLL={effects[top]:.3f}")
    fig.suptitle(title[:180], wrap=True)
    fig.colorbar(artist, ax=axes[1], label="Answer content NLL change (nats/token)", shrink=.7)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return {"path": path, "unit_boxes_xyxy_normalized": boxes, "largest_positive_unit": top, "geometry": "row-major post-merger grid, native resize without crop"}
