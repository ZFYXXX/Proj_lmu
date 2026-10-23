"""Isolated, training-free anchor geometry ablations (no production changes)."""
from dataclasses import dataclass
import hashlib
import json


@dataclass(frozen=True)
class GeometrySpec:
    name: str
    metric: str = "cosine"
    initialization: str = "farthest"
    seed: int = 11
    lloyd_steps: int = 0
    assignment_metric: str | None = None


def specifications():
    result = [GeometrySpec("cosine_farthest"),
              GeometrySpec("unit_l2_farthest", metric="unit_l2"),
              GeometrySpec("raw_l2_farthest", metric="raw_l2"),
              GeometrySpec("raw_l2_select_cosine_assign", metric="raw_l2", assignment_metric="cosine")]
    for seed in (11, 23, 47):
        result += [GeometrySpec(f"unit_kmeanspp_s{seed}", metric="unit_l2", initialization="kmeanspp", seed=seed),
                   GeometrySpec(f"unit_kmeans10_s{seed}", metric="unit_l2", initialization="kmeanspp", seed=seed, lloyd_steps=10)]
    return result


def _distance(values, center, metric):
    # Actual squared Euclidean, not renamed cosine. FP32 arithmetic throughout.
    # Negative cosine preserves the original FP32 tie behavior exactly;
    # subtracting from one could round distinct similarities to the same value.
    return -(values @ center) if metric == "cosine" else (values-center).square().sum(-1)


def _assign(values, centers, metric):
    if metric == "cosine":
        return (values @ centers.T).argmax(-1)
    import torch
    return torch.cdist(values, centers, p=2, compute_mode="donot_use_mm_for_euclid_dist").argmin(-1)


def _repair_empty(labels, values, centers, count):
    """Split a populated cluster at its largest residual; lowest index breaks ties."""
    import torch
    labels = labels.clone()
    sizes = torch.bincount(labels, minlength=count)
    repairs = 0
    for empty in torch.nonzero(sizes == 0).flatten().tolist():
        eligible = sizes[labels] > 1
        errors = (values-centers[labels]).square().sum(-1).masked_fill(~eligible, -float("inf"))
        index = int(errors.argmax())
        old = int(labels[index])
        labels[index] = empty
        sizes[old] -= 1
        sizes[empty] += 1
        repairs += 1
    if bool((sizes == 0).any()):
        raise ValueError("Empty-cluster repair failed")
    return labels, repairs


def cluster(features, *, height, width, count=128, spec=GeometrySpec("cosine_farthest")):
    import torch
    if features.ndim != 2 or height*width != len(features) or not 1 <= count <= len(features):
        raise ValueError("Invalid geometry/budget")
    if spec.metric not in {"cosine", "unit_l2", "raw_l2"} or spec.initialization not in {"farthest", "kmeanspp"}:
        raise ValueError("Unknown geometry")
    if spec.lloyd_steps and (spec.metric != "unit_l2" or spec.assignment_metric):
        raise ValueError("This ablation uses ordinary Euclidean Lloyd on fixed unit inputs")
    raw = features.detach().float()
    if not bool(torch.isfinite(raw).all()):
        raise ValueError("Nonfinite source features")
    norms = raw.norm(dim=-1)
    unit = raw/norms[:, None].clamp_min(1e-12)
    values = raw if spec.metric == "raw_l2" else unit
    anchors = [0]  # shared first anchor isolates the later selection rule
    used = torch.zeros(len(raw), device=raw.device, dtype=torch.bool)
    used[0] = True
    best = _distance(values, values[0], spec.metric)
    generator = torch.Generator(device="cpu").manual_seed(spec.seed)
    while len(anchors) < count:
        if spec.initialization == "farthest":
            index = int(best.masked_fill(used, -float("inf")).argmax())
        else:
            # D^2 sampling; _distance already returns squared Euclidean.
            weights = best.clamp_min(0).masked_fill(used, 0).double().cpu()
            index = int(torch.multinomial(weights, 1, generator=generator)) if float(weights.sum()) > 0 else int(torch.nonzero(~used)[0])
        anchors.append(index)
        used[index] = True
        best = torch.minimum(best, _distance(values, values[index], spec.metric))
    ids = torch.tensor(anchors, device=raw.device)
    assignment_metric = spec.assignment_metric or spec.metric
    assigned_values = unit if assignment_metric == "cosine" else values
    labels = _assign(assigned_values, assigned_values[ids], assignment_metric)
    labels[ids] = torch.arange(count, device=raw.device)  # exactly K, even duplicate features
    previous = None
    repairs, iterations, objective = 0, 0, []
    for _ in range(spec.lloyd_steps):
        centers = torch.stack([values[labels == c].mean(0) for c in range(count)])
        proposed = _assign(values, centers, "unit_l2")
        proposed, repaired = _repair_empty(proposed, values, centers, count)
        repairs += repaired
        labels = proposed
        updated = torch.stack([values[labels == c].mean(0) for c in range(count)])
        objective.append(float((values-updated[labels]).square().sum()))
        iterations += 1
        if previous is not None and torch.equal(previous, labels):
            break
        previous = labels.clone()
    members = [tuple(torch.nonzero(labels == c).flatten().tolist()) for c in range(count)]
    records = []
    raw_sse, unit_sse, radii = 0., 0., []
    for group in members:
        group_ids = torch.tensor(group, device=raw.device)
        rows, cols = group_ids//width, group_ids % width
        cr, cc = int(rows.float().mean().round()), int(cols.float().mean().round())
        spatial = (rows-cr).square()+(cols-cc).square()
        representative = int(group_ids[spatial == spatial.min()].min())
        raw_mean, unit_mean = raw[group_ids].mean(0), unit[group_ids].mean(0)
        raw_sse += float((raw[group_ids]-raw_mean).square().sum())
        squared = (unit[group_ids]-unit_mean).square().sum(-1)
        unit_sse += float(squared.sum())
        radii.append(squared)
        records.append((representative, group, raw_mean.to(features.dtype), [cr, cc]))
    records.sort(key=lambda r:r[0])
    final_members = [list(r[1]) for r in records]
    flat = [i for group in final_members for i in group]
    if sorted(flat) != list(range(len(raw))) or len(records) != count or any(not g for g in final_members):
        raise ValueError("Clusters must be a full, exact-K nonempty partition")
    h = hashlib.sha256(json.dumps(final_members, separators=(",", ":")).encode()).hexdigest()
    radius = torch.cat(radii)
    return {"tokens": torch.stack([r[2] for r in records]), "indices": [r[0] for r in records],
        "members": final_members, "partition_hash": h, "position_policy": "rounded_native_mrope_cluster_centroid",
        "anchors": anchors, "centroids": [r[3] for r in records], "cluster_sizes": [len(r[1]) for r in records],
        "geometry": {"raw_relative_sse": raw_sse/max(float(raw.square().sum()), 1e-30),
            "unit_sse_per_token": unit_sse/len(raw), "unit_max_cluster_radius_squared": float(radius.max()),
            "unit_p95_cluster_radius_squared": float(torch.quantile(radius, .95)),
            "source_norm_mean": float(norms.mean()), "source_norm_std": float(norms.std(unbiased=False)),
            "anchor_norm_mean": float(norms[ids].mean()), "anchor_norm_max": float(norms[ids].max()),
            "zero_norm_count": int((norms < 1e-12).sum()), "unit_norm_max_error": float((unit.norm(dim=-1)-1).abs().max()),
            "lloyd_iterations": iterations, "empty_cluster_repairs": repairs, "lloyd_sse_history": objective}}


def compare(reference, candidate):
    anchors_a, anchors_b = set(reference["anchors"]), set(candidate["anchors"])
    mapping_a = {i:c for c,g in enumerate(reference["members"]) for i in g}
    # Label-invariant pair membership agreement with O(N*K), not O(N^2).
    shared = sum(max((sum(mapping_a[i] == label for i in group) for label in set(mapping_a[i] for i in group)), default=0)
                 for group in candidate["members"])
    return {"anchor_overlap_fraction": len(anchors_a & anchors_b)/len(anchors_a),
        "anchor_sequence_identical": reference["anchors"] == candidate["anchors"],
        "partition_identical": reference["members"] == candidate["members"],
        "candidate_cluster_purity_to_reference": shared/len(mapping_a)}
