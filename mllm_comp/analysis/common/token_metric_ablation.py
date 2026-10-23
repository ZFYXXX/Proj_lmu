"""Fixed, answer-blind geometries for the second 500-image metric experiment."""
import hashlib
import json
import math
import time

NEW_METHODS = ["unit_l1", "correlation", "diag_mahalanobis", "query_response_kv", "query_response_js"]
METHODS = ["dense", "cosine_farthest"]+NEW_METHODS


def unit(values):
    values = values.detach().float()
    return values/values.norm(dim=-1, keepdim=True).clamp_min(1e-12)


def regularized_diagonal(mean, second_moment, shrinkage=.1):
    variance = (second_moment.double()-mean.double().square()).clamp_min(0)
    target = variance.mean().clamp_min(1e-12)
    result = ((1-shrinkage)*variance+shrinkage*target).clamp_min(target*1e-6).float()
    if not bool(__import__("torch").isfinite(result).all()):
        raise ValueError("Invalid calibrated diagonal")
    return result


def response_embedding(q, k, v):
    """Q: [heads, question, d], K/V: [kv_heads, image_tokens, d]."""
    import torch
    if q.ndim != 3 or k.shape != v.shape or q.shape[-1] != k.shape[-1] or q.shape[0] % k.shape[0]:
        raise ValueError("Invalid Qwen grouped-query response shapes")
    if not q.shape[1]:
        raise ValueError("No question Query")
    key = k.float().repeat_interleave(q.shape[0]//k.shape[0], dim=0)
    scores = torch.einsum("hqd,hnd->nhq", q.float(), key)/math.sqrt(q.shape[-1])
    scores = scores.flatten(1)
    values = v.float().permute(1, 0, 2).flatten(1)
    embedding = torch.cat((scores/math.sqrt(scores.shape[1]), values/math.sqrt(values.shape[1])), dim=-1)
    probabilities = scores.softmax(-1).clamp_min(1e-12)
    probabilities = probabilities/probabilities.sum(-1, keepdim=True)
    if not bool(torch.isfinite(embedding).all()) or not bool(torch.isfinite(probabilities).all()):
        raise ValueError("Nonfinite native response geometry")
    return embedding, probabilities, scores, values


def extract_responses(bundle, state, question_indices):
    """Only prompt states; real Layer0 RMSNorm, bias-bearing Q/K/V and M-RoPE."""
    import torch
    from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import apply_multimodal_rotary_pos_emb
    prefix = state["prefix"]
    if not question_indices or any(i <= state["visual"][-1] or i >= prefix for i in question_indices):
        raise ValueError("Question queries must follow all image tokens and exclude the answer")
    llm = bundle["llm"]
    layer = llm.layers[0]
    attn = layer.self_attn
    hidden = state["hidden"][:, :prefix]
    positions = state["positions"][:, :, :prefix]
    with torch.inference_mode():
        normalized = layer.input_layernorm(hidden)
        q = attn.q_proj(normalized).view(1, prefix, attn.num_heads, attn.head_dim).transpose(1, 2)
        k = attn.k_proj(normalized).view(1, prefix, attn.num_key_value_heads, attn.head_dim).transpose(1, 2)
        v = attn.v_proj(normalized).view(1, prefix, attn.num_key_value_heads, attn.head_dim).transpose(1, 2)
        cos, sin = llm.rotary_emb(hidden, positions)
        q, k = apply_multimodal_rotary_pos_emb(q, k, cos, sin, attn.rope_scaling["mrope_section"])
        question_ids = torch.tensor(question_indices, device=hidden.device)
        visual_ids = torch.tensor(state["visual"], device=hidden.device)
        q = q[0].index_select(1, question_ids)
        k, v = k[0].index_select(1, visual_ids), v[0].index_select(1, visual_ids)
        embedded, probabilities, scores, values = response_embedding(q, k, v)
    return {"embedding": embedded, "probabilities": probabilities, "scores": scores, "values": values,
        "question_indices": list(question_indices), "query_heads": attn.num_heads,
        "key_value_heads": attn.num_key_value_heads, "question_tokens": len(question_indices),
        "response_ms": None, "answer_used": False, "positions": "native_post_mrope"}


def metric_space(features, method, diagonal=None, responses=None):
    import torch
    normalized = unit(features)
    if method == "cosine_farthest":
        return normalized, "cosine"
    if method == "unit_l1":
        return normalized, "l1"
    if method == "correlation":
        raw = features.detach().float()
        return unit(raw-raw.mean(-1, keepdim=True)), "cosine"
    if method == "diag_mahalanobis":
        if diagonal is None or diagonal.shape != (features.shape[1],) or not bool((diagonal > 0).all()):
            raise ValueError("Positive frozen calibration diagonal required")
        return normalized/diagonal.to(features.device).sqrt(), "l2"
    if method == "query_response_kv" and responses is not None:
        return responses["embedding"], "l2"
    if method == "query_response_js" and responses is not None:
        return responses["probabilities"], "js"
    raise ValueError(f"Missing or unsupported metric inputs for {method}")


def pair_distances(x, y, metric):
    import torch
    if metric == "cosine":
        return -(x @ y[0])[:, None] if len(y) == 1 else -(x @ y.T)
    if metric in {"l1", "l2"}:
        return torch.cdist(x, y, p=1 if metric == "l1" else 2, compute_mode="donot_use_mm_for_euclid_dist")
    if metric == "js":
        # Chunking bounds the [source, anchors, query-response-dim] temporary.
        results = []
        hx = -(x*x.clamp_min(1e-30).log()).sum(-1)
        for begin in range(0, len(y), 8):
            yy = y[begin:begin+8]
            hy = -(yy*yy.clamp_min(1e-30).log()).sum(-1)
            middle = (x[:, None]+yy[None])/2
            hm = -(middle*middle.clamp_min(1e-30).log()).sum(-1)
            results.append((hm-(hx[:, None]+hy[None])/2).clamp_min(0).sqrt())
        return torch.cat(results, dim=1)
    raise ValueError(metric)


def farthest_partition(values, count, metric):
    import torch
    if values.ndim != 2 or not 1 <= count <= len(values) or not bool(torch.isfinite(values).all()):
        raise ValueError("Invalid metric features/count")
    anchors = [0]
    used = torch.zeros(len(values), dtype=torch.bool, device=values.device)
    used[0] = True
    best = pair_distances(values, values[:1], metric)[:, 0]
    while len(anchors) < count:
        index = int(best.masked_fill(used, -float("inf")).argmax())
        anchors.append(index); used[index] = True
        best = torch.minimum(best, pair_distances(values, values[index:index+1], metric)[:, 0])
    ids = torch.tensor(anchors, device=values.device)
    labels = pair_distances(values, values[ids], metric).argmin(-1)
    labels[ids] = torch.arange(count, device=values.device)
    members = [torch.nonzero(labels == i).flatten().tolist() for i in range(count)]
    return anchors, members


def merge_partition(features, members, height, width, anchors):
    """Identical original-feature FP32 mean/representative/centroid convention."""
    import torch
    if height*width != len(features) or sorted(i for g in members for i in g) != list(range(len(features))) or any(not g for g in members):
        raise ValueError("Partition must cover every source exactly once, without empty clusters")
    raw, normalized = features.detach().float(), unit(features)
    records, raw_sse, unit_sse = [], 0., 0.
    for group in members:
        ids = torch.tensor(group, device=features.device)
        rr, cc = ids//width, ids % width
        r, c = int(rr.float().mean().round()), int(cc.float().mean().round())
        spatial = (rr-r).square()+(cc-c).square()
        representative = int(ids[spatial == spatial.min()].min())
        mean = raw[ids].mean(0)
        raw_sse += float((raw[ids]-mean).square().sum())
        unit_sse += float((normalized[ids]-normalized[ids].mean(0)).square().sum())
        records.append((representative, list(group), mean.to(features.dtype), [r, c]))
    records.sort(key=lambda r:r[0])
    groups = [r[1] for r in records]
    return {"tokens": torch.stack([r[2] for r in records]), "indices": [r[0] for r in records], "members": groups,
        "anchors": anchors, "centroids": [r[3] for r in records], "cluster_sizes": [len(g) for g in groups],
        "partition_hash": hashlib.sha256(json.dumps(groups, separators=(",", ":")).encode()).hexdigest(),
        "position_policy": "rounded_native_mrope_cluster_centroid",
        "geometry": {"raw_relative_sse": raw_sse/max(float(raw.square().sum()), 1e-30), "unit_sse_per_token": unit_sse/len(raw)}}


def select_and_merge(features, *, height, width, method, diagonal=None, responses=None, count=128):
    import torch
    def sync():
        if features.device.type == "cuda":
            torch.cuda.synchronize(features.device)
    sync(); start = time.perf_counter()
    values, metric = metric_space(features, method, diagonal, responses)
    anchors, groups = farthest_partition(values, count, metric)
    sync(); selected = time.perf_counter()
    result = merge_partition(features, groups, height, width, anchors)
    sync(); done = time.perf_counter()
    result["timing"] = {"selection_ms": 1000*(selected-start), "aggregation_and_statistics_ms": 1000*(done-selected)}
    return result
