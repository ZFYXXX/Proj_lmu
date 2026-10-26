"""Matched full-length soft versus hard-forward/sigmoid-backward training.

No change to the selector class or its checkpoint/hash contract. STE is a
surrogate, not an exact derivative of discrete selection. Both arms retain all
sequence positions; a zero embedding is NOT equivalent to deleting a token.
"""
import hashlib

import torch

MODES = ('soft', 'ste')
# Keep historical two-arm orchestration unchanged. New runs opt in explicitly.
TRAINING_MODES = (*MODES, 'vp_noise')


def parameter_digest(module):
    digest = hashlib.sha256()
    for name, value in sorted(module.state_dict().items()):
        tensor = value.detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str(tensor.dtype).encode())
        digest.update(str(tuple(tensor.shape)).encode())
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def training_selection(selector, features, mode, *, question=None, noise=None):
    if mode not in TRAINING_MODES:
        raise ValueError(f'Unknown training mask mode: {mode}')
    result = selector(features, soft=True, **({'question': question} if question is not None else {}))
    soft, hard = result['soft_weights'], result['hard_mask']
    # Parentheses ensure the forward correction is exactly zero (not rounded
    # (hard + soft) - soft), while its derivative is the continuous Jacobian.
    applied = soft if mode in ('soft', 'vp_noise') else hard + (soft - soft.detach())
    if mode == 'vp_noise':
        from .visual_noise_gate import noise_gate
        result['tokens'] = noise_gate(features, soft, noise=noise)
        result['noise_gate'] = 'sqrt_alpha_x_plus_sqrt_one_minus_alpha_standard_normal'
    else:
        result['tokens'] = (features * applied[:, None]).to(features.dtype)
    result['applied_weights'] = applied
    result['training_mask_mode'] = mode
    # BCE must always read the genuine continuous weights, never STE 0/1.
    return result


def selection_diagnostics(selection):
    soft = selection['soft_weights'].detach()
    applied = selection['applied_weights'].detach()
    hard = selection['hard_mask']
    if not torch.isfinite(applied).all():
        raise ValueError('Nonfinite training mask')
    return {'source_visual_tokens': selection['source_count'],
        'physical_visual_tokens': selection['result_count'],
        'target_visual_tokens': selection['target_count'],
        'soft_mass': float(soft.sum()), 'applied_mass': float(applied.sum()),
        'soft_hard_mean_abs_gap': float((soft - hard).abs().mean()),
        'applied_nonzero_tokens': int((applied != 0).sum())}
