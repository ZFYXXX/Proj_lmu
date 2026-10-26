"""Budget-neutral post-selection recovery, NOT an official paper reproduction.

Selection and protected contents are immutable. Only context representatives
absorb discarded features. Original representative M-RoPE is retained by the
binding; representing several positions at one coordinate is an approximation.
"""
from dataclasses import dataclass
import math

import torch
import torch.nn.functional as F

from .protected_visual_topk import ProtectedWordSelector


@dataclass(frozen=True)
class RecoverySpec:
    mode: str = "query_gated_residual"
    maximum_blend: float = .25

    def __post_init__(self):
        if self.mode not in {"neighbor_mean", "query_gated_residual"}:
            raise ValueError("Unknown recovery mode")
        if (isinstance(self.maximum_blend, bool) or not math.isfinite(self.maximum_blend)
                or not 0 <= self.maximum_blend <= 1):
            raise ValueError("Recovery maximum_blend must be in [0,1]")


@torch.no_grad()
def recover_visual_tokens(features, selected, protected, relevance, visual_scores, spec):
    """Nearest-context pooling or conservative relevance-gated residual.

    Route each discarded token i to its nearest CONTEXT representative j by
    cosine, with source-order tie breaking. The mean arm averages x_j and all
    its assigned sources. The residual arm uses

      w_i = max(cos(x_i,x_j),0)^2 * sigmoid((r_i-mean(r))/std(visual_scores))
      y_j = x_j + max_blend/count_j * sum_i w_i*(x_i-x_j).

    Thus the effective convex blend cannot exceed max_blend, even for a large
    source population. Do not amplify weak query residuals by dividing by their
    own tiny standard deviation. These weights are proxies, NOT probabilities
    of causal usefulness. Protected tokens never absorb any source.
    """
    if features.ndim != 2 or not len(features) or not features.shape[1] or not features.is_floating_point():
        raise ValueError("Expected floating nonempty [N,D] features")
    n = len(features)
    for name, ids in (("selected", selected), ("protected", protected)):
        if (ids.ndim != 1 or ids.dtype != torch.long or ids.device != features.device
                or len(ids.unique()) != len(ids) or (ids < 0).any() or (ids >= n).any()
                or not torch.equal(ids, ids.sort().values)):
            raise ValueError(f"Invalid source-ordered {name} indices")
    if not len(selected) or not torch.isin(protected, selected).all():
        raise ValueError("Protected indices must belong to the nonempty selected set")
    if any(t.shape != (n,) or t.device != features.device for t in (relevance, visual_scores)):
        raise ValueError("Recovery scores must align with source tokens")
    if any(not torch.isfinite(t).all() for t in (features, relevance, visual_scores)):
        raise ValueError("Nonfinite recovery input")
    selected_mask = torch.zeros(n, dtype=torch.bool, device=features.device)
    selected_mask[selected] = True
    dropped = (~selected_mask).nonzero(as_tuple=True)[0]
    context_local = (~torch.isin(selected, protected)).nonzero(as_tuple=True)[0]
    context = selected[context_local]
    result = features.index_select(0, selected).clone()
    stats = {"recovery_mode": spec.mode, "recovery_applied": False,
        "recovery_training": False, "recovery_selection_changed": False,
        "recovery_protected_unchanged": True, "recovery_position_policy": "original_representative_mrope",
        "recovery_causal_importance_verified": False, "recovery_lossless": False,
        "recovery_extra_tokens": 0, "recovery_maximum_blend": spec.maximum_blend,
        "recovery_dropped_indices": dropped.tolist(), "recovery_target_indices": [],
        "recovery_source_weights": [], "recovery_context_indices": context.tolist(),
        "recovery_assignment_counts": [0]*len(context), "recovery_blend": [0.]*len(context),
        "recovery_changed_tokens": 0, "recovery_mean_relative_change": 0.,
        "recovery_mean_assignment_cosine": None, "recovery_nonpositive_affinity_sources": 0,
        "recovery_effective_sources": 0}
    if not len(dropped) or not len(context):
        stats["recovery_skip_reason"] = "identity" if not len(dropped) else "no_unprotected_context"
        return result, stats
    with torch.autocast(device_type=features.device.type, enabled=False):
        x = features.float()
        affinity = F.normalize(x[dropped], dim=-1) @ F.normalize(x[context], dim=-1).T
        cosine, targets = affinity.max(-1)
        # A small M x C matrix, not an N x N graph. GEMM avoids nondeterministic
        # CUDA atomic index_add reductions in many-to-one pooling.
        membership = F.one_hot(targets, num_classes=len(context)).float()
        counts = membership.sum(0)
        original = x[context]
        if spec.mode == "neighbor_mean":
            weights = torch.ones_like(cosine)
            recovered = (original + membership.T @ x[dropped]) / (counts[:, None] + 1)
            blend = counts / (counts + 1)
        else:
            scale = visual_scores.float().std(unbiased=False).clamp_min(1e-4)
            query_gate = torch.sigmoid((relevance.float()-relevance.float().mean()) / scale)
            weights = cosine.clamp(0, 1).square() * query_gate[dropped]
            mass = membership * weights[:, None]
            sums = mass.sum(0)
            blend = spec.maximum_blend * sums / counts.clamp_min(1)
            recovered = original + spec.maximum_blend * (
                mass.T @ x[dropped] - sums[:, None]*original) / counts.clamp_min(1)[:, None]
        result[context_local] = recovered.to(features.dtype)
        relative = (result.float()-x[selected]).norm(dim=-1) / x[selected].norm(dim=-1).clamp_min(1e-8)
        if not torch.isfinite(result).all():
            raise ValueError("Nonfinite recovered representation")
        protected_local = torch.isin(selected, protected)
        if not torch.equal(result[protected_local], features[selected[protected_local]]):
            raise RuntimeError("Recovery modified a protected token")
        if spec.mode == "query_gated_residual" and (blend > spec.maximum_blend+1e-6).any():
            raise RuntimeError("Recovery blend exceeded frozen bound")
    stats.update(recovery_applied=True, recovery_target_indices=context[targets].tolist(),
        recovery_source_weights=weights.tolist(), recovery_assignment_counts=counts.long().tolist(),
        recovery_blend=blend.tolist(), recovery_changed_tokens=int((result != features[selected]).any(-1).sum()),
        recovery_mean_relative_change=float(relative.mean()), recovery_mean_assignment_cosine=float(cosine.mean()),
        recovery_nonpositive_affinity_sources=int((cosine <= 0).sum()),
        recovery_effective_sources=int((weights > 0).sum()))
    return result, stats


class RecoveredProtectedSelector(ProtectedWordSelector):
    def __init__(self, base, policy, recovery):
        super().__init__(base, policy)
        self.recovery = recovery

    @torch.no_grad()
    def forward(self, features, *, question, soft=False, visual_only=False):
        output = super().forward(features, question=question, soft=soft, visual_only=visual_only)
        protected = torch.tensor(output["word_diagnostics"]["protected_indices"], device=features.device)
        tokens, stats = recover_visual_tokens(features, output["indices"], protected,
            output["scores"], self.base.visual(features)["scores"], self.recovery)
        output["tokens"] = tokens
        output["word_diagnostics"].update(stats)
        return output
