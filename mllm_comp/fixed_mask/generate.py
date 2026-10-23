"""Turn framework hierarchical Taylor artifacts into portable fixed selections."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import random
import re

from mllm_comp.analysis.common.rank import aggregate_task_percentiles
from mllm_comp.core.serialization import atomic_write_json
from .spec import KINDS, normalize_spec


def _component(identifier, head_group):
    match = re.fullmatch(r"layer:(\d+)/path:(v2v|t2v|t2t)/head:(\d+)", identifier)
    if match:
        layer, kind, index = match.groups()
        return kind, int(layer), int(index)
    match = re.fullmatch(r"layer:(\d+)/(head|ffn_neuron):(\d+)/group:(all|vision|image|text)", identifier)
    if match:
        layer, unit, index, group = match.groups()
        group = "vision" if group == "image" else group
        if unit == "head" and group != head_group:
            return None
        return "heads" if unit == "head" else f"ffn_{group}", int(layer), int(index)
    return None


def generate_spec(artifact, ratios, *, allocation="layer", selection="taylor", seed=42,
                  family="normalized_taylor", head_group="all", tasks=None):
    """Disable the lowest mean task-percentile components, or a random control.

    Ranking pools are separate for each component kind and budget pool. All
    selected tasks must measure the same candidates; missing scores are errors.
    """
    if allocation not in {"layer", "global"} or selection not in {"taylor", "random"}:
        raise ValueError("Invalid allocation or selection")
    if head_group not in {"all", "vision", "text"}:
        raise ValueError("head_group must be all, vision or text")
    if not ratios or any(k not in KINDS or isinstance(v, bool) or not math.isfinite(v)
                         or not 0 <= v <= 1 for k, v in ratios.items()):
        raise ValueError("ratios must map component kinds to fractions in [0, 1]")
    payload = artifact.get("payload", artifact)
    scalar = payload.get("scores_by_family", {}).get(family, {})
    vectors = payload.get("score_vectors_by_family", {}).get(family, {})
    task_names = sorted(tasks if tasks is not None else set(scalar) | set(vectors))
    if not task_names:
        raise ValueError(f"No task scores for {family}")
    scores = {kind: {} for kind in ratios}
    for task in task_names:
        entries = dict(scalar.get(task, {}))
        for identifier, values in vectors.get(task, {}).items():
            for index, value in enumerate(values):
                key = identifier.replace("/ffn_neuron/", f"/ffn_neuron:{index}/")
                if key in entries and entries[key] != value:
                    raise ValueError(f"Conflicting scalar/vector score: {key}")
                entries[key] = value
        for identifier, value in entries.items():
            component = _component(identifier, head_group)
            if component is None or component[0] not in ratios:
                continue
            kind, layer, index = component
            if not math.isfinite(float(value)):
                raise ValueError(f"Non-finite score: {task}/{identifier}")
            scores[kind].setdefault(task, {})[f"{layer}:{index}"] = float(value)
    result, budgets = {}, {}
    for kind in sorted(ratios):
        task_scores = scores[kind]
        candidates = set(task_scores.get(task_names[0], {}))
        if not candidates or any(set(task_scores.get(task, {})) != candidates for task in task_names):
            raise ValueError(f"{kind} requires identical, nonempty candidate coverage across tasks")
        pools = {}
        for key in sorted(candidates, key=lambda key: tuple(map(int, key.split(":")))):
            layer = int(key.split(":")[0])
            pools.setdefault(layer if allocation == "layer" else "global", []).append(key)
        selected = []
        rng = random.Random(f"{seed}:{kind}")
        budgets[kind] = {}
        for pool, keys in pools.items():
            count = math.floor(len(keys) * ratios[kind])
            if selection == "random":
                chosen = rng.sample(keys, count)
            else:
                ranked = aggregate_task_percentiles({task: {key: task_scores[task][key] for key in keys}
                                                    for task in task_names})["components"]
                chosen = sorted(keys, key=lambda key: (ranked[key]["mean_percentile"],
                                *map(int, key.split(":"))))[:count]
            selected.extend(chosen)
            budgets[kind][str(pool)] = {"candidates": len(keys), "disabled": count}
        result[kind] = {}
        for key in selected:
            layer, index = key.split(":")
            result[kind].setdefault(layer, []).append(int(index))
    result["provenance"] = {"selection": selection, "allocation": allocation, "seed": seed,
        "score_family": family, "head_group": head_group, "tasks": task_names,
        "ratios": dict(ratios), "budgets": budgets,
        "aggregation": "mean_task_percentile_average_ties_v1",
        "source": {key: artifact[key] for key in ("backbone", "checkpoint", "checkpoint_revision") if key in artifact}}
    return normalize_spec(result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scores", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--ratios", required=True, help="e.g. v2v=0.4,t2v=0.6,ffn_vision=0.4")
    parser.add_argument("--allocation", choices=["layer", "global"], default="layer")
    parser.add_argument("--selection", choices=["taylor", "random"], default="taylor")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--family", choices=["normalized_taylor", "direct_removal_taylor"], default="normalized_taylor")
    parser.add_argument("--head-group", choices=["all", "vision", "text"], default="all")
    parser.add_argument("--tasks", nargs="+")
    args = parser.parse_args()
    ratios = {key: float(value) for key, value in (item.split("=") for item in args.ratios.split(","))}
    spec = generate_spec(json.loads(args.scores.read_text(encoding="utf-8")), ratios,
        allocation=args.allocation, selection=args.selection, seed=args.seed,
        family=args.family, head_group=args.head_group, tasks=args.tasks)
    atomic_write_json(args.output, spec, ensure_ascii=False)


if __name__ == "__main__":
    main()
