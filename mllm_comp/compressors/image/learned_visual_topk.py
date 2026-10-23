"""Visual-only scorer and VisionSelector's continuous cardinality operator.

Source: VisionSelector @ 0336cd58ddc29c443c7ad8d2f45f2911b37ab0e1,
qwen-vl-finetune/compression_method/selector_model.py, TopK / _find_ts.
Only the operator is reused, NOT VisionSelector's TransformerScorer/model loop.
"""
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path

import torch
from torch import nn

SOURCE = {
    "repository": "https://github.com/JulietChoo/VisionSelector",
    "commit": "0336cd58ddc29c443c7ad8d2f45f2911b37ab0e1",
    "file": "qwen-vl-finetune/compression_method/selector_model.py",
    "sha256": "613a5c31962af13634c81147509040c382b097cc270326f6fe9c69dfa532d820",
    "scope": "sigmoid threshold search and implicit-gradient TopK only",
}
FORMAT = "qwen25_learned_visual_topk_v1"


class _ContinuousTopK(torch.autograd.Function):
    @staticmethod
    def forward(ctx, scores, k):
        # FP64 is retained for gradcheck; production BF16/FP16/FP32 -> FP32.
        work = scores.double() if scores.dtype == torch.float64 else scores.float()
        lo = -work.max(1, keepdim=True).values - 10
        hi = -work.min(1, keepdim=True).values + 10
        for _ in range(64):
            mid = (lo + hi) / 2
            lower = torch.sigmoid(work + mid).sum(1, keepdim=True) < k
            lo = torch.where(lower, mid, lo)
            hi = torch.where(lower, hi, mid)
        p = torch.sigmoid(work + (lo + hi) / 2)
        if not torch.allclose(p.sum(1), p.new_full((len(p),), k), rtol=0, atol=1e-3):
            raise ValueError("Continuous TopK failed its cardinality constraint")
        ctx.save_for_backward(p)
        ctx.input_dtype = scores.dtype
        return p

    @staticmethod
    def backward(ctx, grad_output):
        (p,) = ctx.saved_tensors
        # Same Jacobian as upstream vmap(grad(sigmoid)), without importing its
        # model or depending on functional-transform support in checkpointing.
        v = p * (1 - p)
        total = v.sum(1, keepdim=True)
        if not torch.isfinite(total).all() or bool((total <= 1e-12).any()):
            raise ValueError("Continuous TopK gradient saturated; refusing NaN/zero denominator")
        uv = grad_output.to(p.dtype) * v
        return (uv - uv.sum(1, keepdim=True) * v / total).to(ctx.input_dtype), None


def continuous_topk(scores, k):
    if scores.ndim != 2 or min(scores.shape) < 1 or not scores.is_floating_point():
        raise ValueError("TopK scores must be a nonempty floating [batch,tokens] tensor")
    if not torch.isfinite(scores).all():
        raise ValueError("TopK scores must be finite")
    if isinstance(k, bool) or not isinstance(k, int) or not 0 <= k <= scores.shape[1]:
        raise ValueError("TopK k must be an integer in [0,N]")
    if k in (0, scores.shape[1]):
        return scores.float() * 0 + int(k > 0)
    return _ContinuousTopK.apply(scores, k)


def hard_topk_indices(scores, k):
    if scores.ndim != 1 or not torch.isfinite(scores).all():
        raise ValueError("Hard TopK needs finite one-dimensional scores")
    if isinstance(k, bool) or not isinstance(k, int) or not 1 <= k <= len(scores):
        raise ValueError("Hard TopK needs 1 <= K <= N")
    # Stable source-index tie break, then source-order sequence serialization.
    return scores.detach().argsort(descending=True, stable=True)[:k].sort().values


@dataclass(frozen=True)
class LearnedVisualTopKSpec:
    hidden_size: int
    width: int = 256
    target_count: int = 128
    seed: int = 42

    def __post_init__(self):
        for key, value in asdict(self).items():
            if isinstance(value, bool) or not isinstance(value, int) or value < (0 if key == "seed" else 1):
                raise ValueError(f"Invalid selector {key}")


class LearnedVisualTopK(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.spec = spec
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(spec.seed)
            self.scorer = nn.Sequential(nn.LayerNorm(spec.hidden_size), nn.Linear(spec.hidden_size, spec.width),
                nn.GELU(), nn.Linear(spec.width, 1))
            nn.init.normal_(self.scorer[-1].weight, std=1e-4)
            nn.init.zeros_(self.scorer[-1].bias)

    def forward(self, features, *, soft=False):
        if features.ndim != 2 or features.shape[1] != self.spec.hidden_size or len(features) < 1:
            raise ValueError("Selector features must have shape [N,hidden_size]")
        if not torch.isfinite(features).all():
            raise ValueError("Selector features must be finite")
        with torch.autocast(device_type=features.device.type, enabled=False):
            scores = self.scorer(features.detach().float()).squeeze(-1)
            k = min(len(features), self.spec.target_count)
            indices = hard_topk_indices(scores, k)
            weights = continuous_topk(scores[None], k)[0] if soft else None
        target = torch.zeros_like(scores).scatter(0, indices, 1)
        tokens = (features * weights[:, None]).to(features.dtype) if soft else features.index_select(0, indices)
        return {"tokens": tokens, "indices": indices, "scores": scores,
            "soft_weights": weights, "hard_mask": target.detach(),
            "source_count": len(features), "result_count": len(tokens), "target_count": k,
            "mode": "soft_full_sequence" if soft else "hard_compact"}


def implementation_hash():
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def save_selector(module, folder, *, base_identity, provenance):
    from safetensors.torch import save_file
    from mllm_comp.core.serialization import atomic_write_json
    folder = Path(folder)
    if folder.exists():
        raise ValueError(f"Selector checkpoint already exists: {folder}")
    folder.mkdir(parents=True)
    save_file({k: v.detach().cpu().contiguous() for k, v in module.state_dict().items()}, str(folder / "selector.safetensors"))
    atomic_write_json(folder / "selector_config.json", {"format": FORMAT, "spec": asdict(module.spec),
        "implementation_hash": implementation_hash(), "base_identity_sha256": base_identity,
        "weights_sha256": hashlib.sha256((folder / 'selector.safetensors').read_bytes()).hexdigest(),
        "source": SOURCE, "provenance": provenance})


def load_selector(folder, *, base_identity, device):
    from safetensors.torch import load_file
    folder = Path(folder)
    config = json.loads((folder / "selector_config.json").read_text())
    if config["format"] != FORMAT or config["implementation_hash"] != implementation_hash():
        raise ValueError("Selector format or implementation hash mismatch")
    if config["base_identity_sha256"] != base_identity:
        raise ValueError("Selector belongs to a different base checkpoint")
    if config['source'] != SOURCE or config.get('weights_sha256') != hashlib.sha256((folder / 'selector.safetensors').read_bytes()).hexdigest():
        raise ValueError('Selector source/weights integrity check failed')
    module = LearnedVisualTopK(LearnedVisualTopKSpec(**config["spec"]))
    module.load_state_dict(load_file(str(folder / "selector.safetensors")), strict=True)
    return module.to(device)
