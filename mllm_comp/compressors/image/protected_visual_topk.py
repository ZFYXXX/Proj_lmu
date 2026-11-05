"""Fixed-budget evidence protection and complementary context, without training.

Inspired by AnchorPrune's ordered selection, not an official reproduction:
the global prior here is our frozen learned visual scorer, not vision attention.
"""
from dataclasses import dataclass
import math

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class ProtectedTopKSpec:
    relevance: str = "learned_word"
    minimum_anchor: int = 20
    maximum_anchor_ratio: float = .5
    novelty_threshold: float = .2
    novelty_patience: int = 3

    def __post_init__(self):
        if self.relevance not in {"learned_word", "raw_cosine"}:
            raise ValueError("Unknown protected relevance source")
        for name in ("minimum_anchor", "novelty_patience"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"Invalid {name}")
        if (isinstance(self.maximum_anchor_ratio, bool) or
                not math.isfinite(self.maximum_anchor_ratio) or not 0 < self.maximum_anchor_ratio <= 1):
            raise ValueError("Invalid maximum_anchor_ratio")
        if (isinstance(self.novelty_threshold, bool) or
                not math.isfinite(self.novelty_threshold) or not 0 <= self.novelty_threshold <= 2):
            raise ValueError("Invalid novelty_threshold")


@torch.no_grad()
def protected_context_indices(features, relevance, visual_scores, k, spec):
    """Return K unique source-ordered original indices, plus the protected set.

    Anchor growth compares all candidates to the SAME initial anchor and counts
    novelty events cumulatively. Context is selected by softmax(visual_scores)
    times novelty relative to the current retained set. Ties use source order.
    """
    if features.ndim != 2 or not len(features) or not features.shape[1]:
        raise ValueError("Expected nonempty visual [N,D]")
    n = len(features)
    if relevance.shape != (n,) or visual_scores.shape != (n,):
        raise ValueError("Scores and visual tokens must align")
    if any(not torch.isfinite(t).all() for t in (features, relevance, visual_scores)):
        raise ValueError("Nonfinite protected-selection input")
    if isinstance(k, bool) or not isinstance(k, int) or k < 1:
        raise ValueError("Expected positive integer K")
    if features.device != relevance.device or features.device != visual_scores.device:
        raise ValueError("Protected-selection devices must match")
    k = min(k, n)
    if k == n:
        identity = torch.arange(n, device=features.device)
        return identity, identity
    x = F.normalize(features.detach().float(), dim=-1)
    ranked = torch.argsort(relevance.float(), descending=True, stable=True)
    minimum = min(spec.minimum_anchor, k)
    cap = min(k, max(minimum, int(spec.maximum_anchor_ratio * k)))
    initial = ranked[:minimum]
    anchor_count = cap
    if cap > minimum:
        candidates = ranked[minimum:cap]
        novelty = (1 - x[candidates] @ x[initial].T).clamp(0, 2).min(-1).values
        events = (novelty > spec.novelty_threshold).cumsum(0)
        stopping = (events >= spec.novelty_patience).nonzero(as_tuple=True)[0]
        if len(stopping):
            anchor_count = minimum + int(stopping[0]) + 1
    anchor = ranked[:anchor_count]
    importance = torch.softmax(visual_scores.float(), dim=0)
    selected = torch.zeros(n, dtype=torch.bool, device=features.device)
    selected[anchor] = True
    novelty = (1 - x @ x[anchor].T).clamp(0, 2).min(-1).values
    for _ in range(k - anchor_count):
        utility = (importance * novelty).masked_fill(selected, -torch.inf)
        index = utility.argmax()
        selected[index] = True
        novelty = torch.minimum(novelty, (1 - x @ x[index]).clamp(0, 2))
    indices = selected.nonzero(as_tuple=True)[0]
    if len(indices) != k or not selected[anchor].all():
        raise RuntimeError("Protected set lost or budget violated")
    return indices, anchor.sort().values


class ProtectedWordSelector(torch.nn.Module):
    """Reuse an immutable word checkpoint; only its hard selection rule changes."""
    def __init__(self, base, policy):
        super().__init__()
        self.base = base
        self.policy = policy
        self.base.eval().requires_grad_(False)

    @property
    def spec(self):
        return self.base.spec

    def freeze_prior(self):
        self.base.freeze_prior()

    @torch.no_grad()
    def forward(self, features, *, question, soft=False, visual_only=False, target_count=None):
        if soft or visual_only:
            raise ValueError("Protected selection is hard diagnostic inference only")
        output = self.base(features, question=question)
        prior = self.base.visual(features)["scores"]
        if self.policy.relevance == "raw_cosine":
            with torch.autocast(device_type=features.device.type, enabled=False):
                relevance = (F.normalize(features.float(), dim=-1) @
                             F.normalize(question.float(), dim=-1).T).max(-1).values
        else:
            relevance = output["scores"] - prior
        k = self.spec.target_count if target_count is None else target_count
        indices, anchor = protected_context_indices(features, relevance, prior, k, self.policy)
        output.update(tokens=features.index_select(0, indices), indices=indices,
                      scores=relevance, result_count=len(indices), target_count=min(k, len(features)),
                      hard_mask=torch.zeros_like(relevance).scatter(0, indices, 1))
        output["word_diagnostics"].update(
            selection_policy="protected_context", policy_relevance=self.policy.relevance,
            protected_indices=anchor.tolist(), protected_count=len(anchor),
            context_count=len(indices)-len(anchor), all_protected_retained=True,
            context_prior="softmax_frozen_visual_score", context_novelty="min_cosine_distance",
            relevance_std=float(relevance.std(unbiased=False)),
            learned_query_residual_used=self.policy.relevance == "learned_word",
            score_audit_meaning="query_priority_not_final_greedy_utility",
            official_anchorprune_reproduction=False, new_training=False)
        return output
