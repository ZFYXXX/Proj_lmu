"""Frozen-MLP ablations: independent spatial partition and native position policy."""
import hashlib
import json
import math

import torch


def spatial_anchor_merge(features, height, width, count=128, beta=0., importance=None):
    if features.ndim != 2 or height * width != len(features) or not len(features):
        raise ValueError("Expected one nonempty rectangular feature grid")
    if not math.isfinite(beta) or beta < 0 or count < 1 or not torch.isfinite(features).all():
        raise ValueError("Invalid spatial partition input")
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
    selected = torch.zeros(len(features), dtype=torch.bool, device=features.device)
    selected[anchors[0]] = True
    # beta=0 uses the historical arithmetic, including its floating-point ties.
    similarity = values @ values[anchors[0]]
    distance = 1 - similarity + beta * (coords - coords[anchors[0]]).square().sum(-1)
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
    assignment[anchor_ids] = torch.arange(count, device=features.device)
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
            "partition_hash": hashlib.sha256(json.dumps(groups, separators=(',', ':')).encode()).hexdigest()}


def merge_positions(source, members, anchors, policy):
    if policy == 'anchor':
        return source[:, anchors]
    if policy != 'centroid':
        raise ValueError('Unknown visual position policy')
    return torch.stack([source[:, group].float().mean(-1).round() for group in members], -1).to(source.dtype)
