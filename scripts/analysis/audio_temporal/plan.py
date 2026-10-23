"""Freeze experiment conditions before reading any MMSU prediction."""
from __future__ import annotations


EXPERIMENTS = (
    "capture",
    "provenance",
    "similarity",
    "regions",
    "predictability",
    "substitution",
    "locality",
)


def _condition(family: str, name: str, **fields):
    return {"id": name, "family": family, **fields}


def build_conditions(
    experiment: str,
    *,
    block_ms=(40.0, 80.0, 160.0, 320.0),
    predictors=("copy_left", "linear", "local_mean"),
    replace_ratios=(0.1,),
    locality_fractions=(0.05, 0.1, 0.2),
    region_types=("silent", "stable", "transient", "boundary"),
    window_ms=320.0,
    far_ms=2000.0,
):
    """Return the exact paired conditions for one independently run experiment."""
    if experiment not in EXPERIMENTS:
        raise ValueError(f"Unknown experiment {experiment!r}")
    conditions = []
    if experiment in {"capture", "provenance", "similarity"}:
        return conditions
    if experiment == "regions":
        for ratio in replace_ratios:
            for region in region_types:
                conditions.append(_condition(
                    "region_substitution",
                    f"region_{region}_replace{ratio:g}",
                    region=region,
                    replace_ratio=float(ratio),
                ))
    elif experiment == "predictability":
        for duration in block_ms:
            for predictor in predictors:
                conditions.append(_condition(
                    "predictability",
                    f"predict_{predictor}_{duration:g}ms",
                    predictor=predictor,
                    block_ms=float(duration),
                    replace_ratio=float(replace_ratios[0]),
                ))
    elif experiment == "substitution":
        for ratio in replace_ratios:
            for strategy in ("local_high", "far_high", "random_local", "local_low"):
                conditions.append(_condition(
                    "substitution",
                    f"substitute_{strategy}_{ratio:g}",
                    strategy=strategy,
                    replace_ratio=float(ratio),
                    window_ms=float(window_ms),
                    far_ms=float(far_ms),
                ))
    elif experiment == "locality":
        for fraction in locality_fractions:
            for scope in ("local", "global"):
                conditions.append(_condition(
                    "locality",
                    f"locality_{scope}_merge{fraction:g}",
                    scope=scope,
                    merge_fraction=float(fraction),
                    window_ms=float(window_ms),
                    far_ms=float(far_ms),
                ))
    identifiers = [condition["id"] for condition in conditions]
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("Duplicate temporal-analysis condition IDs")
    return conditions
