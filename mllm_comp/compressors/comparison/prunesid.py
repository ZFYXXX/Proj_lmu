"""PruneSID's released Qwen PSCA/NMS selector, with explicit budget adapters.

Source: ZhengyaoFang/PruneSID @ 4ef17d96b390027f31243bc7e1d27bdee77792a5,
prunesid/prunesid_qwen/modeling_qwen2_vl.py (Apache-2.0 per upstream README).
The Qwen2.5 binding preserves native source order/M-RoPE, bounds PCA rank,
uses request-local seeded randomized PCA, and never duplicates source tokens.
Fixed K is an evaluation adaptation, not the release's need_token_num/576 rule.
"""
import math

import torch
import torch.nn.functional as F


PRUNESID_DEFAULTS = {
    "method": "prunesid", "tokens_per_crop": 128, "pca_seed": 0,
    "log_first_n": 1,
}
SOURCE_COMMIT = "4ef17d96b390027f31243bc7e1d27bdee77792a5"


def normalize_prunesid_config(raw=None):
    cfg = {**PRUNESID_DEFAULTS, **(raw or {})}
    unknown = set(cfg) - set(PRUNESID_DEFAULTS)
    if unknown:
        raise ValueError(f"Unknown PruneSID keys: {sorted(unknown)}")
    for key in ("tokens_per_crop", "pca_seed", "log_first_n"):
        if isinstance(cfg[key], bool) or not isinstance(cfg[key], int):
            raise ValueError(f"PruneSID {key} must be an integer")
    if cfg["tokens_per_crop"] < 1 or min(cfg["pca_seed"], cfg["log_first_n"]) < 0:
        raise ValueError("PruneSID needs a positive budget and nonnegative seed/log count")
    cfg["method"] = "prunesid"
    return cfg


@torch.no_grad()
def select_prunesid_indices(features, *, cfg):
    cfg = normalize_prunesid_config(cfg)
    if features.ndim != 3 or features.shape[0] != 1 or min(features.shape) < 1:
        raise ValueError("PruneSID requires one nonempty [1,N,D] image")
    if not features.is_floating_point() or not torch.isfinite(features).all():
        raise ValueError("PruneSID features must be finite floating point")
    x = features[0]
    n, dim = x.shape
    k = min(n, cfg["tokens_per_crop"])
    metadata = {"token_reducer": "prunesid", "source_commit": SOURCE_COMMIT,
        "trained": False, "pca_seed": cfg["pca_seed"],
        "budget_adapter": "fixed_count_or_explicit_retention_ratio",
        "official_budget_rule": "max(round(N*need_token_num/576),16)",
        "position_adapter": "source_order_original_sparse_mrope"}
    if k == n:
        return [torch.arange(n, device=x.device)], {**metadata, "identity": True}
    rank = min(max(k // 4, 4), n, dim, k)
    devices = [x.device.index] if x.is_cuda else []
    # Do not reset the model's global RNG as a side effect of compression.
    with torch.random.fork_rng(devices=devices):
        torch.random.default_generator.manual_seed(cfg["pca_seed"])
        if x.is_cuda:
            with torch.cuda.device(x.device):
                torch.cuda.manual_seed(cfg["pca_seed"])
        _, _, vectors = torch.pca_lowrank(torch.sigmoid(x.float()).T, q=rank)
    weights = vectors.abs()
    groups = weights.argmax(dim=1)
    weights = weights * F.one_hot(groups, rank)
    scores = weights.clone()
    # Upstream computes this cosine in the input dtype, then casts to FP32.
    normalized = F.normalize(x, dim=-1)
    similarity = (normalized @ normalized.T).float()
    threshold = max(k / (n / 18), 1) * similarity.triu().mean()
    counts, nms_counts, members = [], [], []
    for group in range(rank):
        ids = (groups == group).nonzero(as_tuple=True)[0]
        members.append(ids)
        counts.append(len(ids))
        active = weights[ids, group].clone()
        local_similarity = similarity[ids][:, ids]
        kept = []
        while bool(active.sum() > 0):
            index = int(active.argmax())
            kept.append(index)
            active[index] = 0
            active[local_similarity[index] > threshold] = 0
        nms_counts.append(len(kept))
        if kept:
            scores[ids[torch.tensor(kept, device=x.device)], group] = torch.arange(
                n, n - len(kept), -1, device=x.device, dtype=scores.dtype)
    counts = torch.tensor(counts, device=x.device)
    nms_counts = torch.tensor(nms_counts, device=x.device)
    lower = counts.clamp(max=1)
    upper = torch.minimum(counts, nms_counts.clamp(max=5 * math.ceil(k / 64)))
    while int(upper.sum()) < k:
        upper = torch.minimum(upper + 1, counts)
    remaining = k - int(lower.sum())
    if remaining < 0:
        raise ValueError("PruneSID semantic groups exceed token budget")
    cumulative = (nms_counts.float() / nms_counts.sum().clamp_min(1)).cumsum(0)
    allocations = (cumulative * remaining).round().long()
    quotas = torch.minimum(torch.diff(allocations, prepend=allocations.new_zeros(1)) + lower, upper)
    for group in nms_counts.argsort(descending=True).tolist():
        extra = min(int(upper[group] - quotas[group]), k - int(quotas.sum()))
        quotas[group] += extra
    chosen = []
    for group, ids in enumerate(members):
        # Restrict fallback to actual members, unlike upstream global zero-score
        # sorting. This makes degenerate inputs safe and forbids duplicate IDs.
        order = scores[ids, group].argsort(descending=True)
        chosen.append(ids[order[:int(quotas[group])]])
    indices = torch.cat(chosen).sort().values
    if len(indices) != k or len(indices.unique()) != k:
        raise ValueError("PruneSID allocation lost or duplicated tokens")
    return [indices], {**metadata, "pca_rank": rank,
        "group_counts": counts.tolist(), "nms_counts": nms_counts.tolist(),
        "group_quotas": quotas.tolist(), "nms_threshold": float(threshold)}
