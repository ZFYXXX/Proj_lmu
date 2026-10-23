from __future__ import annotations

from dataclasses import dataclass, field, fields
import math
import re
from types import MappingProxyType
from typing import Any, Mapping

from mllm_comp.core.layout import LayoutDelta, LayoutTokenUnit, OrderingPolicy


def _torch() -> tuple[Any, Any]:
    import torch
    import torch.nn.functional as F

    return torch, F


@dataclass(frozen=True)
class BifocalSpec:
    s1_frac: float = 0.25
    block: int | str = "auto"
    coarse_frac: float = 0.29
    k_frac: float = 0.75
    clsr: str = "1"
    no_query: bool = False
    respool: bool = True
    respool_pos: bool = False
    metric: str = "vit_attn"
    metric_agg: str = "mass"
    vit_block: int = 31
    rank_map: str = ""
    rank_mix: float = 0.0
    nrm_sign: int = 1
    sink_clamp: float = 0.0
    s1_merge: str = ""
    s1_mu: float = 1.0
    word_space: int = 0

    def __post_init__(self) -> None:
        block = self.block
        if isinstance(block, str):
            b = block.strip().lower()
            if b != "auto":
                try:
                    block = int(b)
                except ValueError as exc:
                    raise ValueError(f"bifocal block={self.block!r} must be int >= 2 or 'auto'") from exc
            else:
                block = "auto"
        elif isinstance(block, bool) or not isinstance(block, int):
            raise ValueError(f"bifocal block={self.block!r} must be int >= 2 or 'auto'")
        object.__setattr__(self, "block", block)
        for name in ("clsr", "metric", "metric_agg", "rank_map", "s1_merge"):
            object.__setattr__(self, name, str(getattr(self, name)).strip().lower())
        object.__setattr__(self, "s1_frac", float(self.s1_frac))
        object.__setattr__(self, "coarse_frac", float(self.coarse_frac))
        object.__setattr__(self, "k_frac", float(self.k_frac))
        object.__setattr__(self, "rank_mix", float(self.rank_mix))
        object.__setattr__(self, "sink_clamp", float(self.sink_clamp))
        object.__setattr__(self, "s1_mu", float(self.s1_mu))
        object.__setattr__(self, "nrm_sign", int(self.nrm_sign))
        object.__setattr__(self, "vit_block", int(self.vit_block))
        object.__setattr__(self, "word_space", int(self.word_space))
        object.__setattr__(self, "no_query", bool(self.no_query))
        object.__setattr__(self, "respool", bool(self.respool))
        object.__setattr__(self, "respool_pos", bool(self.respool_pos))

        if not 0 < self.s1_frac < 1:
            raise ValueError(f"bifocal s1_frac={self.s1_frac} must be in (0,1)")
        if self.metric not in ("vit_attn", "norm", "uniform"):
            raise ValueError(f"bifocal metric={self.metric!r}: vit_attn/norm/uniform")
        if self.nrm_sign not in (-1, 0, 1):
            raise ValueError(f"bifocal nrm_sign={self.nrm_sign}: -1/0/1")
        if self.block != "auto" and self.block < 2:
            raise ValueError(f"bifocal block={self.block} must be >= 2 or auto")
        if not 0 <= self.k_frac < 1:
            raise ValueError(f"bifocal k_frac={self.k_frac} must be in [0,1) (keep >= 1 rel seat)")
        if self.clsr not in ("1", "0", "nrm"):
            raise ValueError(f"bifocal clsr={self.clsr!r}: 1/0/nrm")
        if self.metric_agg not in ("mass", "vmass", "nodiag"):
            raise ValueError(f"bifocal metric_agg={self.metric_agg!r}: mass/vmass/nodiag")
        if self.metric_agg != "mass" and self.metric != "vit_attn":
            raise ValueError("bifocal metric_agg != mass requires metric=vit_attn")
        if self.rank_map not in ("", "mass", "vmass", "nodiag", "norm"):
            raise ValueError(f"bifocal rank_map={self.rank_map!r}: ''/mass/vmass/nodiag/norm")
        if self.rank_map in ("mass", "vmass", "nodiag") and self.metric != "vit_attn":
            raise ValueError(f"bifocal rank_map={self.rank_map} requires metric=vit_attn")
        if not 0 <= self.rank_mix <= 1:
            raise ValueError(f"bifocal rank_mix={self.rank_mix} must be in [0,1]")
        if self.rank_mix > 0 and self.rank_map == "norm":
            raise ValueError("bifocal rank_mix and rank_map=norm are redundant; give one")
        if (self.rank_map == "norm" or self.rank_mix >= 0.999) and self.clsr == "1":
            raise ValueError("bifocal rank_map=norm needs clsr=nrm or 0 (else clsr=OLS(norm,norm)=0)")
        if self.s1_merge not in ("", "near"):
            raise ValueError(f"bifocal s1_merge={self.s1_merge!r}: ''/near")
        if self.s1_merge and not self.respool:
            raise ValueError("bifocal s1_merge requires respool")
        if self.respool_pos and not self.respool:
            raise ValueError("bifocal respool_pos requires respool")
        if not 0 <= self.s1_mu <= 1:
            raise ValueError(f"bifocal s1_mu={self.s1_mu} must be in [0,1]")
        if not 0 < self.coarse_frac < 1:
            raise ValueError(f"bifocal coarse_frac={self.coarse_frac} must be in (0,1)")
        if not 0 <= self.sink_clamp < 1:
            raise ValueError(f"bifocal sink_clamp={self.sink_clamp} must be in [0,1)")
        if self.word_space not in (0, 1, 2):
            raise ValueError(f"bifocal word_space={self.word_space}: 0/1/2")
        if self.vit_block < 0:
            raise ValueError(f"bifocal vit_block={self.vit_block} must be >= 0")

    @property
    def needs_vit_maps(self) -> bool:
        return self.metric == "vit_attn"

    def block_for(self, H: int, W: int, NT: int) -> int:
        if self.block != "auto":
            return int(self.block)
        return min(
            range(2, max(H, W) + 1),
            key=lambda b: (abs(math.ceil(H / b) * math.ceil(W / b) / NT - self.coarse_frac), b),
        )

    def budget(self, N: int, H: int, W: int) -> tuple[int, int, int, int] | None:
        NT = int(round(N * self.s1_frac))
        if NT < 1:
            return None
        blk = self.block_for(H, W, NT)
        C = math.ceil(H / blk) * math.ceil(W / blk)
        n_fine = NT - C
        if n_fine < 2 or NT >= N:
            return None
        k = min(int(round(n_fine * self.k_frac)), n_fine - 1)
        return NT, C, n_fine, k


BIFOCAL_CONFIG_KEYS = frozenset(item.name for item in fields(BifocalSpec))


def bifocal_spec_from_config(config: Mapping[str, Any]) -> BifocalSpec:
    unknown = set(config) - BIFOCAL_CONFIG_KEYS
    if unknown:
        raise ValueError(f"unknown Bifocal config keys: {sorted(unknown)}")
    return BifocalSpec(**dict(config))


_REC_RE = re.compile(
    r"bounding box coordinate of the region this sentence describes\s*:\s*(.+)",
    re.IGNORECASE | re.DOTALL,
)
_SUFFIXES = (
    "\nanswer the question using a single word or phrase.",
    "\nyour response must be concise.",
    "\nprovide a short and direct response.",
    "\nanswer with the option's letter from the given choices directly.",
    "\nanswer with the option's letter from the given choices directly",
    "\nanswer the question with a single word.",
    "\nwhen the provided information is insufficient, respond with 'unanswerable'.",
)


def bifocal_extract_query(text: Any) -> str | None:
    if not isinstance(text, str):
        return None
    q = text.strip()
    m = _REC_RE.search(q)
    if m:
        return m.group(1).strip().strip('"').strip()
    low = q.lower()
    for suf in _SUFFIXES:
        i = low.rfind(suf)
        if i >= 0:
            q = q[:i]
            low = q.lower()
    return q.strip()


def bifocal_query_text_from_ids(ids: Any, tokenizer: Any, vision_end_id: int, im_end_id: int) -> str | None:
    ids = ids.tolist() if hasattr(ids, "tolist") else list(ids)
    try:
        st = len(ids) - 1 - ids[::-1].index(vision_end_id)
    except ValueError:
        return None
    try:
        ed = ids.index(im_end_id, st)
    except ValueError:
        ed = len(ids)
    return tokenizer.decode(ids[st + 1:ed], skip_special_tokens=True)


def bifocal_split_words(txt: Any) -> list[str]:
    return [
        w
        for w in (raw.strip(".,!?;:\"'()[]{}<>`") for raw in str(txt).lower().split())
        if w
    ]


def bifocal_word_token_ids(tokenizer: Any, w: str, mode: int = 0) -> list[int]:
    bare = tokenizer(w, add_special_tokens=False).input_ids
    if mode == 0:
        return bare
    sp = tokenizer(" " + w, add_special_tokens=False).input_ids
    if mode == 1:
        return sp or bare
    return bare + sp if mode == 2 else sp


def bifocal_word_embeddings(tokenizer: Any, embed_tokens: Any, device: Any, txt: str, mode: int = 0) -> Any:
    torch, _ = _torch()
    words = bifocal_split_words(txt)
    if not words:
        return None
    vecs = []
    with torch.no_grad():
        for w in words:
            ids = bifocal_word_token_ids(tokenizer, w, mode)
            if not ids:
                continue
            vecs.append(
                embed_tokens(torch.as_tensor(ids, device=device).unsqueeze(0))[0].mean(dim=0)
            )
    return torch.stack(vecs, dim=0) if vecs else None


def _bifocal_block_of(flat_idx: Any, W: int, block: int) -> Any:
    Wb = (W + block - 1) // block
    return (flat_idx // W // block) * Wb + (flat_idx % W // block)


def _bifocal_block_weights(w: Any, H: int, W: int, block: int) -> tuple[Any, Any, int]:
    torch, _ = _torch()
    N = H * W
    pos = torch.arange(N, device=w.device)
    blk = _bifocal_block_of(pos, W, block)
    C = ((H + block - 1) // block) * ((W + block - 1) // block)
    wsum = torch.zeros(C, device=w.device, dtype=w.dtype).index_add_(0, blk, w)
    return w / (wsum[blk] + 1e-8), blk, C


def _bifocal_block_pool(embeds: Any, wn: Any, blk: Any, C: int) -> Any:
    torch, _ = _torch()
    tok = torch.zeros(C, embeds.shape[1], device=embeds.device, dtype=torch.float32)
    return tok.index_add(0, blk, embeds.float() * wn.unsqueeze(-1))


def _bifocal_coarse_plan(metric: Any, N: int, H: int, W: int, block: int, dev: Any) -> tuple[Any, Any, int, Any, Any]:
    torch, _ = _torch()
    w = metric.float().clamp_min(0)
    wn, blk, C = _bifocal_block_weights(w, H, W, block)
    pos = torch.arange(N, device=dev)
    r, c = (pos // W).float(), (pos % W).float()
    rr = torch.floor(torch.zeros(C, device=dev).index_add_(0, blk, wn * r) + 0.5).long().clamp(0, H - 1)
    cc = torch.floor(torch.zeros(C, device=dev).index_add_(0, blk, wn * c) + 0.5).long().clamp(0, W - 1)
    idx = rr * W + cc
    order = idx.argsort()
    return wn, blk, C, order, idx[order]


def _bifocal_fl_greedy(feats: Any, sal: Any, blocked: Any, seeds: Any, k: int) -> Any:
    torch, _ = _torch()
    x = feats.float()
    x = x / x.norm(dim=1, keepdim=True).clamp_min(1e-8)
    C = x @ x.t()
    s = sal.float()
    s = s / s.max().clamp_min(1e-8)
    cov = C[seeds].max(dim=0).values.clamp_min(0.0) if seeds.numel() else torch.zeros_like(s)
    free = ~blocked
    out = torch.empty(k, dtype=torch.long, device=feats.device)
    for i in range(k):
        g = (torch.relu(C - cov.unsqueeze(0)) * s.unsqueeze(0)).sum(dim=1)
        t = int(torch.argmax(torch.where(free, g, torch.full_like(g, -1.0))))
        out[i] = t
        free[t] = False
        cov = torch.maximum(cov, C[t])
    return out


def _bifocal_select_fine(rel: Any, sal: Any, coarse_idx: Any, n_fine: int, k: int, embeds: Any) -> Any:
    torch, _ = _torch()
    prio = rel.float().clone().view(-1)
    prio[coarse_idx] = float("-inf")
    if k <= 0:
        return torch.topk(prio, n_fine).indices
    n_text = n_fine - k
    seeds = torch.cat([coarse_idx, torch.topk(prio, n_text).indices])
    blocked = torch.zeros(prio.numel(), dtype=torch.bool, device=prio.device)
    blocked[seeds] = True
    attn_idx = _bifocal_fl_greedy(embeds, sal, blocked, seeds, k)
    prio[attn_idx] = float("-inf")
    return torch.cat([attn_idx, torch.topk(prio, n_text).indices])


def _bifocal_near_merge_plan(
    embeds: Any, metric: Any, coarse_idx: Any, fine_idx: Any, H: int, W: int, block: int, mu: float
) -> tuple[Any, Any, Any | None]:
    torch, F = _torch()
    N, D = embeds.shape
    dev = embeds.device
    x = embeds.float()
    pos = torch.arange(N, device=dev)
    taken = torch.zeros(N, dtype=torch.bool, device=dev)
    taken[coarse_idx] = True
    taken[fine_idx] = True
    d_idx = pos[~taken]
    merged = torch.zeros(N, dtype=torch.bool, device=dev)
    owner_full = torch.full((N,), -1, dtype=torch.long, device=dev)
    if d_idx.numel() == 0 or fine_idx.numel() == 0:
        return merged, owner_full, None
    rc = torch.stack([pos // W, pos % W], dim=1).float()
    dist = torch.cdist(rc[d_idx], rc[fine_idx], p=1)
    dmin, owner = dist.min(dim=1)
    xn = F.normalize(x, dim=-1)
    cos = (xn[d_idx] * xn[fine_idx][owner]).sum(dim=-1)
    ok = (dmin <= float(block)) & (cos > 0)
    if not bool(ok.any()):
        return merged, owner_full, None
    d_ok, own, w = d_idx[ok], owner[ok], (
        metric.float().clamp_min(0)[d_idx][ok] / metric.float().clamp_min(0).max().clamp_min(1e-8)
        * cos[ok] * (1.0 - dmin[ok] / (block + 1.0))) * mu
    merged[d_ok] = True
    owner_full[d_ok] = own
    return merged, owner_full, (d_ok, own, w)


def _bifocal_near_merge_apply(embeds: Any, fine_idx: Any, plan: Any | None) -> Any:
    torch, _ = _torch()
    x = embeds.float()
    out = x[fine_idx]
    if plan is None:
        return out
    d_ok, own, w = plan
    acc = torch.zeros_like(out).index_add(0, own, x[d_ok] * w.unsqueeze(-1))
    wsum = torch.zeros(out.shape[0], device=embeds.device).index_add_(0, own, w)
    return (out + acc) / (1.0 + wsum).unsqueeze(-1)


def _bifocal_residual_plan(
    metric: Any, coarse_idx: Any, fine_idx: Any, N: int,
    H: int, W: int, block: int, dev: Any, merged: Any = None, move_pos: bool = False,
) -> tuple[Any, Any, int, Any, Any, int, Any]:
    torch, _ = _torch()
    w = metric.float().clamp_min(0).clone()
    w[fine_idx] = 0.0
    if merged is not None:
        w[merged] = 0.0
    wn, blk, C = _bifocal_block_weights(w, H, W, block)
    m = torch.ones(N, device=dev)
    m[fine_idx] = 0.0
    if merged is not None:
        m[merged] = 0.0
    cnt = torch.zeros(C, device=dev).index_add_(0, blk, m)
    bid = _bifocal_block_of(coarse_idx, W, block)
    fb = cnt[bid] == 0
    n_fb = int(fb.sum())
    if not move_pos:
        return wn, blk, C, bid, fb, n_fb, coarse_idx
    pos = torch.arange(N, device=dev)
    r, c = (pos // W).float(), (pos % W).float()
    rr = torch.zeros(C, device=dev).index_add_(0, blk, wn * r)
    cc = torch.zeros(C, device=dev).index_add_(0, blk, wn * c)
    res = m > 0
    d2 = torch.where(res, (r - rr[blk]).pow(2) + (c - cc[blk]).pow(2), torch.full_like(r, 1e8))
    order = (blk.double() * 1e9 + d2.double()).argsort()
    bs = blk[order]
    first = torch.ones_like(bs, dtype=torch.bool)
    first[1:] = bs[1:] != bs[:-1]
    winner = torch.full((C,), -1, dtype=torch.long, device=dev)
    winner[bs[first]] = order[first]
    cand = winner[bid]
    ok = (cand >= 0) & res[cand.clamp_min(0)] & ~fb
    return wn, blk, C, bid, fb, n_fb, torch.where(ok, cand, coarse_idx)


def _bifocal_residual_apply(embeds: Any, coarse_tok: Any, plan: Any) -> Any:
    torch, _ = _torch()
    wn, blk, C, bid, fb, n_fb = plan[:6]
    new_tok = _bifocal_block_pool(embeds, wn, blk, C)[bid]
    if n_fb:
        new_tok = torch.where(fb.unsqueeze(-1), coarse_tok, new_tok)
    return new_tok


def _bifocal_smooth(maps: Any, H: int, W: int) -> Any:
    _, F = _torch()
    n = maps.shape[0]
    return F.avg_pool2d(
        maps.view(n, 1, H, W), 3, stride=1, padding=1, count_include_pad=True
    ).view(n, H * W)


def _bifocal_ols_resid(y: Any, x: Any) -> Any:
    torch, _ = _torch()
    xc = x - x.mean(dim=1, keepdim=True)
    yc = y - y.mean(dim=1, keepdim=True)
    d = xc.pow(2).sum(dim=1, keepdim=True)
    b = torch.where(d > 1e-12, (xc * yc).sum(dim=1, keepdim=True) / d, torch.zeros_like(d))
    return yc - b * xc


def _bifocal_relevance(embeds: Any, metric: Any, word_embs: Any, H: int, W: int, spec: BifocalSpec) -> tuple[Any, str, dict]:
    torch, F = _torch()
    diag: dict[str, Any] = {}
    if spec.no_query or word_embs is None or word_embs.shape[0] < 1:
        return metric.clone(), "attn", diag
    try:
        pf = embeds.float()
        nrm = _bifocal_smooth(pf.norm(dim=-1).view(1, -1), H, W)
        sal_s = _bifocal_smooth(metric.float().view(1, -1), H, W)
        if spec.clsr == "nrm":
            clsr = (spec.nrm_sign or 1) * nrm
        elif spec.clsr == "0":
            clsr = sal_s
        else:
            clsr = _bifocal_ols_resid(sal_s, nrm)
        pn = F.normalize(pf, dim=-1)
        wn = F.normalize(word_embs.to(device=embeds.device, dtype=torch.float32), dim=-1)
        sims = _bifocal_smooth(wn @ pn.t(), H, W)
        lo = sims.min(dim=1, keepdim=True).values
        hi = sims.max(dim=1, keepdim=True).values
        ac = clsr - clsr.mean(dim=1, keepdim=True)
        sc = sims - sims.mean(dim=1, keepdim=True)
        r = (sc * ac).sum(dim=1) / (sc.norm(dim=1) * ac.norm(dim=1) + 1e-12)
        wgt = r.clamp_min(0.0) + 1e-6
        wgt = wgt / wgt.sum()
        sn = (sims - lo) / (hi - lo + 1e-8)
        rel = torch.exp((wgt.view(-1, 1) * torch.log(sn + 1e-6)).sum(dim=0)).view(1, -1)
        parts = [rel] + ([clsr] if spec.clsr == "1" else [])
        if spec.nrm_sign:
            parts.append(spec.nrm_sign * nrm)
        s = torch.cat(parts, dim=0)
        lo = s.min(dim=1, keepdim=True).values
        hi = s.max(dim=1, keepdim=True).values
        sn = (s - lo) / (hi - lo + 1e-8)
        out = torch.exp(torch.log(sn + 1e-6).mean(dim=0))
        diag["word_weights"] = [float(v) for v in wgt.detach().cpu().tolist()]
        return out, f"salr+crn({word_embs.shape[0]}w)", diag
    except Exception as exc:
        diag["relevance_error"] = repr(exc)
        return metric.clone(), "attn", diag


@dataclass(frozen=True)
class BifocalStage1Result:
    tokens: Any
    indices: Any
    coarse_indices: Any
    fine_indices: Any
    layout_delta: LayoutDelta
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @property
    def selected_indices(self) -> Any:
        return self.indices


def _pick_metric(vision_maps: Mapping[str, Any] | None, name: str, N: int) -> Any:
    if vision_maps is None or vision_maps.get(name) is None:
        raise ValueError(f"Bifocal needs ViT map {name!r} but the adapter did not supply it")
    value = vision_maps[name]
    if int(value.numel()) != N:
        raise ValueError(f"Bifocal ViT map {name!r} has {int(value.numel())} entries, expected {N}")
    return value.float().view(-1)


def bifocal_image_tokens(
    tokens: Any,
    vision_maps: Mapping[str, Any] | None,
    *,
    grid_height: int,
    grid_width: int,
    word_embeddings: Any | None = None,
    spec: BifocalSpec | None = None,
    media_unit: str | LayoutTokenUnit = "image_feature",
    source_layout: Mapping[str, Any] | None = None,
    result_layout: Mapping[str, Any] | None = None,
) -> BifocalStage1Result:
    torch, F = _torch()
    spec = spec or BifocalSpec()
    embeds = tokens
    H, W = int(grid_height), int(grid_width)
    N, D = embeds.shape
    if N != H * W:
        raise ValueError(f"Bifocal embeds {N} != grid {H}x{W}={H * W}")
    base_meta: dict[str, Any] = {
        "algorithm": "bifocal",
        "stage": "stage1",
        "grid": (H, W),
        "source_tokens": int(N),
    }
    b = spec.budget(N, H, W)
    if b is None:
        delta = LayoutDelta.identity(N, media_unit=media_unit)
        return BifocalStage1Result(
            tokens=embeds,
            indices=torch.arange(N, device=embeds.device),
            coarse_indices=torch.arange(0, device=embeds.device),
            fine_indices=torch.arange(N, device=embeds.device),
            layout_delta=delta,
            metadata=MappingProxyType(
                {**base_meta, "passthrough": True, "requested_tokens": int(N), "result_tokens": int(N)}
            ),
        )
    NT, C, n_fine, k = b
    blk = spec.block_for(H, W, NT)

    with torch.no_grad():
        if spec.metric == "uniform":
            m = torch.ones(N, device=embeds.device)
        elif spec.metric == "norm":
            m = embeds.float().norm(dim=-1)
        else:
            m = _pick_metric(vision_maps, spec.metric_agg, N).to(embeds.device)
        n_cap = 0
        if spec.sink_clamp:
            cap = torch.quantile(m, spec.sink_clamp)
            n_cap = int((m > cap).sum())
            if n_cap:
                m = m.clamp(max=cap)

        if not spec.rank_map:
            m_rank = m
        elif spec.rank_map == "norm":
            m_rank = embeds.float().norm(dim=-1)
        else:
            m_rank = _pick_metric(vision_maps, spec.rank_map, N).to(embeds.device)
        if spec.rank_mix > 0:
            def _mm(t: Any) -> Any:
                lo, hi = t.min(), t.max()
                return (t - lo) / (hi - lo).clamp_min(1e-8)

            m_rank = (1.0 - spec.rank_mix) * _mm(m_rank.float()) + spec.rank_mix * _mm(
                embeds.float().norm(dim=-1)
            )

        c_wn, c_blk, c_count, c_order, coarse_idx = _bifocal_coarse_plan(m, N, H, W, blk, embeds.device)
        rel, used, rel_diag = _bifocal_relevance(embeds, m_rank, word_embeddings, H, W, spec)
        k_eff = k if used != "attn" else -1
        fine_idx = _bifocal_select_fine(rel, m_rank, coarse_idx, n_fine, k_eff, embeds)
        n_mg = 0
        owner = None
        merge_plan = None
        if spec.s1_merge == "near":
            merged, owner, merge_plan = _bifocal_near_merge_plan(
                embeds, m, coarse_idx, fine_idx, H, W, blk, spec.s1_mu
            )
            n_mg = int(merged.sum())
        else:
            merged = None
        residual_plan = None
        if spec.respool:
            residual_plan = _bifocal_residual_plan(
                m, coarse_idx, fine_idx, N, H, W, blk, embeds.device,
                merged=merged, move_pos=spec.respool_pos,
            )
            n_fb = residual_plan[5]
            coarse_idx_final = residual_plan[6]
        else:
            n_fb = 0
            coarse_idx_final = coarse_idx

        idx = torch.cat([coarse_idx_final, fine_idx])
        order = idx.argsort()
        idx = idx[order]
        if idx.numel() != NT or idx.unique().numel() != NT:
            raise RuntimeError(
                f"[bifocal] selected {idx.numel()} (unique {idx.unique().numel()}) != NT={NT}: "
                "coarse/fine overlap or shortfall, budget would be misreported"
            )

    coarse_tok = _bifocal_block_pool(embeds, c_wn, c_blk, c_count)[c_order]
    if spec.s1_merge == "near":
        fine_tok = _bifocal_near_merge_apply(embeds, fine_idx, merge_plan)
    else:
        fine_tok = embeds[fine_idx].float()
    if residual_plan is not None:
        coarse_tok = _bifocal_residual_apply(embeds, coarse_tok, residual_plan)
    tok = torch.cat([coarse_tok, fine_tok])[order]
    coarse_idx = coarse_idx_final

    idx_list = [int(v) for v in idx.tolist()]
    result_of = {src: r for r, src in enumerate(idx_list)}
    fine_list = [int(v) for v in fine_idx.tolist()]
    coarse_list = [int(v) for v in coarse_idx.tolist()]
    fine_set = set(fine_list)
    old_to_new: list[int | None] = [None] * N
    for f in fine_list:
        old_to_new[f] = result_of[f]
    if owner is not None and merged is not None:
        owner_list = owner.tolist()
        for cell in torch.nonzero(merged).view(-1).tolist():
            old_to_new[int(cell)] = result_of[fine_list[int(owner_list[int(cell)])]]
    coarse_block = {
        int(bb): result_of[c]
        for bb, c in zip(_bifocal_block_of(coarse_idx, W, blk).tolist(), coarse_list)
    }
    cell_block = _bifocal_block_of(torch.arange(N), W, blk).tolist()
    for cell in range(N):
        if old_to_new[cell] is None and cell not in fine_set:
            old_to_new[cell] = coarse_block[int(cell_block[cell])]
    if any(value is None for value in old_to_new):
        raise RuntimeError("[bifocal] layout mapping left a source cell unrepresented")
    delta = LayoutDelta.from_mapping(
        source_count=N,
        old_to_new=old_to_new,
        result_count=NT,
        media_unit=media_unit,
        source_layout=dict(source_layout or {}),
        result_layout={
            **dict(result_layout or {}),
            "representative_indices": tuple(idx_list),
            "coarse_representatives": tuple(sorted(coarse_list)),
            "position_policy": "bifocal_rpme_native_mrope_slice",
        },
        ordering_policy=OrderingPolicy.EXPLICIT,
    )
    metadata = {
        **base_meta,
        "passthrough": False,
        "requested_tokens": int(NT),
        "effective_target_count": int(NT),
        "result_tokens": int(NT),
        "coarse_tokens": int(C),
        "fine_tokens": int(n_fine),
        "block": int(blk),
        "k_seats": int(k_eff),
        "chain": used,
        "respool_fallback": int(n_fb),
        "near_merged": int(n_mg),
        "sink_clamped": int(n_cap),
        **rel_diag,
    }
    return BifocalStage1Result(
        tokens=tok.to(embeds.dtype),
        indices=idx,
        coarse_indices=coarse_idx.sort().values,
        fine_indices=fine_idx.sort().values,
        layout_delta=delta,
        metadata=MappingProxyType(metadata),
    )


__all__ = [
    "BIFOCAL_CONFIG_KEYS",
    "BifocalSpec",
    "BifocalStage1Result",
    "bifocal_extract_query",
    "bifocal_image_tokens",
    "bifocal_query_text_from_ids",
    "bifocal_spec_from_config",
    "bifocal_split_words",
    "bifocal_word_embeddings",
    "bifocal_word_token_ids",
]
