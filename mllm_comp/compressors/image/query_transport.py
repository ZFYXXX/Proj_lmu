"""Semi-unbalanced query-to-visual OT, independent inference experiment.

Fixed uniform query marginal; KL-relaxed uniform visual marginal. Log-domain
Sinkhorn, no learned alignment or labels. Column-mass ranking is a heuristic
subset extraction, NOT a guarantee of discrete multi-object coverage.
"""
from dataclasses import dataclass
import math

import torch
import torch.nn.functional as F

from .learned_visual_topk import hard_topk_indices


@dataclass(frozen=True)
class QueryTransportSpec:
    epsilon: float = .05
    marginal_penalty: float = .1
    weight: float = 1.
    max_iterations: int = 200
    tolerance: float = 1e-5

    def __post_init__(self):
        for key in ('epsilon', 'marginal_penalty', 'tolerance'):
            value = getattr(self, key)
            if isinstance(value, bool) or not math.isfinite(value) or value <= 0:
                raise ValueError(f'Invalid OT {key}')
        if isinstance(self.weight, bool) or not math.isfinite(self.weight) or self.weight < 0:
            raise ValueError('Invalid OT weight')
        if isinstance(self.max_iterations, bool) or not isinstance(self.max_iterations, int) or self.max_iterations < 1:
            raise ValueError('Invalid OT max_iterations')


def semi_unbalanced_transport(cost, spec=QueryTransportSpec()):
    """min <P,C> + eps*sum P(log P-1) + rho*KL(P.T 1 || b).

    Subject to P 1 = a, a=uniform(T), b=uniform(N). The fixed point uses the
    standard relaxed-column exponent rho/(rho+eps). Fail closed if unconverged.
    """
    if cost.ndim != 2 or min(cost.shape) < 1 or not cost.is_floating_point() or not torch.isfinite(cost).all():
        raise ValueError('OT cost must be finite nonempty floating [T,N]')
    work = cost.double() if cost.dtype == torch.float64 else cost.float()
    t, n = work.shape
    log_a, log_b = -math.log(t), -math.log(n)
    kernel = -work / spec.epsilon
    exponent = spec.marginal_penalty / (spec.marginal_penalty + spec.epsilon)
    v = work.new_zeros(n)
    error = math.inf
    for iteration in range(1, spec.max_iterations + 1):
        u = log_a - torch.logsumexp(kernel + v[None], dim=1)
        updated = exponent * (log_b - torch.logsumexp(kernel + u[:, None], dim=0))
        error = float((updated - v).abs().max())
        v = updated
        if error <= spec.tolerance:
            break
    if error > spec.tolerance:
        raise ValueError(f'OT did not converge after {iteration} iterations: {error}')
    u = log_a - torch.logsumexp(kernel + v[None], dim=1)
    plan = (kernel + u[:, None] + v[None]).exp()
    row_error = float((plan.sum(1) - 1 / t).abs().max())
    if not torch.isfinite(plan).all() or row_error > max(1e-6, spec.tolerance):
        raise ValueError('OT plan failed finite/fixed-row-marginal checks')
    return plan, {'ot_iterations': iteration, 'ot_fixed_point_error': error,
        'ot_row_marginal_error': row_error, 'ot_total_mass': float(plan.sum()),
        'ot_query_tokens': t, 'ot_visual_tokens': n}


def transport_topk(features, question, scores, k, spec=QueryTransportSpec()):
    if features.ndim != 2 or question.ndim != 2 or min(features.shape) < 1 or min(question.shape) < 1:
        raise ValueError('OT expects nonempty [N,D] visuals and [T,D] query')
    if features.shape[1] != question.shape[1] or scores.shape != features.shape[:1]:
        raise ValueError('OT feature/query/score dimensions disagree')
    if any(not x.is_floating_point() or not torch.isfinite(x).all() for x in (features, question, scores)):
        raise ValueError('OT inputs must be finite floating tensors')
    if len({x.device for x in (features, question, scores)}) != 1:
        raise ValueError('OT inputs must share a device')
    # An exact original-ranking control, including source-index tie-break.
    if spec.weight == 0 or k == len(features):
        return hard_topk_indices(scores, k), {'ot_applied': False, 'ot_weight': spec.weight}
    with torch.autocast(device_type=features.device.type, enabled=False):
        visual = F.normalize(features.float(), dim=-1)
        query = F.normalize(question.float(), dim=-1)
        cost = 1 - query @ visual.T
        plan, stats = semi_unbalanced_transport(cost, spec)
        mass = plan.sum(0)
        if bool((mass <= 0).any()):
            raise ValueError('OT column mass underflow')
        def standardize(value):
            return (value - value.mean()) / value.std(unbiased=False).clamp_min(1e-6)
        combined = standardize(scores.float()) + spec.weight * standardize(mass.log())
        selected = hard_topk_indices(combined, k)
    return selected, {**stats, 'ot_applied': True, 'ot_weight': spec.weight,
        'ot_epsilon': spec.epsilon, 'ot_marginal_penalty': spec.marginal_penalty,
        'ot_selected_mass': float(mass[selected].sum()),
        'ot_log_mass_std': float(mass.log().std(unbiased=False)),
        'ot_score_std': float(scores.float().std(unbiased=False)),
        'ot_selection': 'zscore_scorer_plus_weight_zscore_log_column_mass',
        'ot_discrete_coverage_guarantee': False}
