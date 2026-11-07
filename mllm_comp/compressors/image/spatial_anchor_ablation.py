"""Frozen-MLP ablations: independent spatial partition and native position policy."""
import hashlib
import json
import math

import torch


def spatial_anchor_merge(features, height, width, count=128, beta=0., importance=None, *,
                         assignment_neighbors=0, assignment_max_regret=None, grid_anchor_floor=False):
    if features.ndim != 2 or height * width != len(features) or not len(features):
        raise ValueError("Expected one nonempty rectangular feature grid")
    if not math.isfinite(beta) or beta < 0 or count < 1 or not torch.isfinite(features).all():
        raise ValueError("Invalid spatial partition input")
    if isinstance(assignment_neighbors, bool) or not isinstance(assignment_neighbors, int) or assignment_neighbors < 0:
        raise ValueError('assignment_neighbors must be a nonnegative integer; zero disables locality')
    if not isinstance(grid_anchor_floor, bool):
        raise ValueError('grid_anchor_floor must be a boolean')
    if assignment_max_regret is not None:
        if (isinstance(assignment_max_regret, bool) or not isinstance(assignment_max_regret, (int, float))
                or not math.isfinite(assignment_max_regret) or not 0 <= assignment_max_regret <= 2):
            raise ValueError('assignment_max_regret must be a finite cosine gap in [0, 2]')
        if not assignment_neighbors or beta != 0:
            raise ValueError('assignment_max_regret requires local assignment and spatial beta=0')
    count = min(count, len(features))
    if importance is not None:
        importance = importance.detach().to(device=features.device, dtype=torch.float32)
        if importance.shape != (len(features),) or not torch.isfinite(importance).all() or (importance < 0).any() or not (importance > 0).any():
            raise ValueError("Importance must contain finite nonnegative weights for every token")
    values = features.detach().float()
    values = values / values.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    ids = torch.arange(len(features), device=features.device)
    pixels = torch.stack((ids // width, ids % width), -1).float()
    coords = pixels / pixels.new_tensor([max(1, height - 1), max(1, width - 1)])
    anchors = [0 if importance is None else int(importance.argmax())]
    grid_trace = {}
    if grid_anchor_floor:
        # Keep the original first anchor, then seed uncovered nonempty 2x2
        # regions at their spatial centers, using the SAME total anchor budget.
        regions = ((ids // width) * 2 // height) * 2 + ((ids % width) * 2 // width)
        cell_ids = regions.unique(sorted=True).tolist()
        candidates = []
        for cell in cell_ids:
            members = (regions == cell).nonzero(as_tuple=True)[0]
            center = pixels[members].mean(0)
            cost = (pixels[members] - center).square().sum(-1)
            candidates.append(int(members[cost == cost.min()].min()))
        while len(anchors) < min(count, len(cell_ids)):
            covered = {int(regions[a]) for a in anchors}
            remaining = [a for a in candidates if int(regions[a]) not in covered]
            # If K cannot cover all cells, choose farthest uncovered cells
            # deterministically rather than silently exceeding the budget.
            next_id = max(remaining, key=lambda a: (
                min(float((pixels[a] - pixels[b]).square().sum()) for b in anchors), -a))
            anchors.append(next_id)
        grid_trace = {'grid_anchor_floor': True, 'grid_seed_indices': anchors.copy(),
            'grid_seed_region_ids': [int(regions[a]) for a in anchors],
            'grid_nonempty_regions': len(cell_ids), 'grid_coverage_feasible': count >= len(cell_ids)}
    selected = torch.zeros(len(features), dtype=torch.bool, device=features.device)
    selected[anchors] = True
    # beta=0 uses the historical arithmetic, including its floating-point ties.
    similarity = values @ values[anchors[0]]
    distance = 1 - similarity + beta * (coords - coords[anchors[0]]).square().sum(-1)
    for seed in anchors[1:]:
        next_similarity = values @ values[seed]
        similarity = torch.maximum(similarity, next_similarity)
        distance = torch.minimum(distance, 1 - next_similarity + beta * (coords - coords[seed]).square().sum(-1))
    while len(anchors) < count:
        if importance is not None:
            next_id = int((distance.clamp_min(0) * importance).masked_fill(selected, -float('inf')).argmax())
        else:
            next_id = int(torch.argmin(similarity.masked_fill(selected, float('inf')))
                          if beta == 0 else torch.argmax(distance.masked_fill(selected, -float('inf'))))
        anchors.append(next_id)
        selected[next_id] = True
        next_similarity = values @ values[next_id]
        similarity = torch.maximum(similarity, next_similarity)
        distance = torch.minimum(distance, 1 - next_similarity + beta * (coords - coords[next_id]).square().sum(-1))
    anchor_ids = torch.tensor(anchors, device=features.device)
    similarities = values @ values[anchor_ids].T
    if beta == 0:
        assignment = similarities.argmax(-1)
    else:
        distances = 1 - similarities + beta * (coords[:, None] - coords[anchor_ids][None]).square().sum(-1)
        assignment = distances.argmin(-1)
    locality = {}
    if assignment_neighbors:
        unrestricted = assignment.clone()
        unrestricted[anchor_ids] = torch.arange(count, device=features.device)
        # Only membership changes. FPS above never reads this option. Distance
        # is in native post-merger grid units; equal distances use source order.
        spatial = (pixels[:, None] - pixels[anchor_ids][None]).square().sum(-1)
        neighbor_count = min(assignment_neighbors, count)
        if neighbor_count < count:
            source_order = anchor_ids.argsort()
            nearest = spatial[:, source_order].argsort(dim=-1, stable=True)[:, :neighbor_count]
            candidates = source_order[nearest]
            allowed = torch.zeros_like(similarities, dtype=torch.bool).scatter_(1, candidates, True)
            if beta == 0:
                assignment = similarities.masked_fill(~allowed, -float('inf')).argmax(-1)
            else:
                assignment = distances.masked_fill(~allowed, float('inf')).argmin(-1)
        assignment[anchor_ids] = torch.arange(count, device=features.device)
        if assignment_max_regret is not None:
            # Compare matching quality, not an assumed absolute semantic cosine
            # threshold. Only membership changes; FPS and the budget stay fixed.
            regret = (similarities[ids, unrestricted] - similarities[ids, assignment]).clamp_min(0)
            fallback = (regret > assignment_max_regret) & ~selected
            assignment = torch.where(fallback, unrestricted, assignment)
            eligible = int((~selected).sum())
            locality.update(assignment_max_regret=float(assignment_max_regret),
                global_fallback_count=int(fallback.sum()),
                global_fallback_fraction=float(fallback.sum()) / max(1, eligible),
                global_fallback_indices=fallback.nonzero(as_tuple=True)[0].tolist(),
                assignment_cosine_regret_mean=float(regret.mean()),
                assignment_cosine_regret_max=float(regret.max()))
        locality.update({'assignment_neighbors': assignment_neighbors,
            'assignment_effective_neighbors': neighbor_count,
            'assignment_changed_count': int((assignment != unrestricted).sum()),
            'assigned_anchor_grid_distance_mean': float(spatial[ids, assignment].sqrt().mean()),
            'unrestricted_anchor_grid_distance_mean': float(spatial[ids, unrestricted].sqrt().mean()),
            'fps_selection_order': anchors.copy()})
    assignment[anchor_ids] = torch.arange(count, device=features.device)
    if grid_anchor_floor:
        grid_trace['grid_covered_regions'] = len({int(regions[a]) for a in anchors})
    records = []
    for cluster, anchor in enumerate(anchors):
        members = (assignment == cluster).nonzero(as_tuple=True)[0]
        center = pixels[members].mean(0)
        rounded = center.round()
        # Keep baseline source serialization even when only M-RoPE is changed.
        distances = (pixels[members] - rounded).square().sum(-1)
        representative = int(members[distances == distances.min()].min())
        raw = features[members].float()
        mean = raw.mean(0)
        span = pixels[members].amax(0) - pixels[members].amin(0)
        stats = {"size": len(members), "span_hw": span.tolist(),
                 "spatial_radius_squared": float((coords[members] - coords[members].mean(0)).square().sum(-1).mean()),
                 "feature_variance": float((raw - mean).square().mean()),
                 "anchor_centroid_distance": float((pixels[anchor] - center).norm())}
        records.append((representative, members.tolist(), anchor, mean.to(features.dtype), stats))
    records.sort(key=lambda r: r[0])
    groups = [r[1] for r in records]
    if sorted(i for g in groups for i in g) != list(range(len(features))):
        raise ValueError("Spatial partition lost or duplicated a source token")
    return {"tokens": torch.stack([r[3] for r in records]),
            "indices": [r[0] for r in records], "members": groups,
            "anchors": [r[2] for r in records], "cluster_statistics": [r[4] for r in records],
            "partition_hash": hashlib.sha256(json.dumps(groups, separators=(',', ':')).encode()).hexdigest(),
            **locality, **grid_trace}


def merge_positions(source, members, anchors, policy):
    if policy == 'anchor':
        return source[:, anchors]
    if policy != 'centroid':
        raise ValueError('Unknown visual position policy')
    return torch.stack([source[:, group].float().mean(-1).round() for group in members], -1).to(source.dtype)
