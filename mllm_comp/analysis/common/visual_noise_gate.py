"""Train-only noise throttling, inspired by DiffPrune (not its reproduction).

The downstream sequence still has N tokens. Noise is not deletion, and raw
post-merger features need not have unit covariance: no variance preservation
or hard-deployment improvement is assumed. No extra denoising block is added.
"""
import torch


def noise_gate(features, weights, *, noise=None):
    if features.ndim != 2 or weights.shape != features.shape[:1]:
        raise ValueError('Noise gate requires features [N,D] and weights [N]')
    if not features.is_floating_point() or not weights.is_floating_point():
        raise ValueError('Noise gate requires floating inputs')
    if features.device != weights.device or not torch.isfinite(features).all():
        raise ValueError('Invalid noise gate feature device/values')
    if not torch.isfinite(weights).all() or bool(((weights < 0) | (weights > 1)).any()):
        raise ValueError('Noise gate weights must be finite in [0,1]')
    work = features.float()
    if noise is None:
        noise = torch.randn_like(work)
    if noise.shape != features.shape or noise.device != features.device or not torch.isfinite(noise).all():
        raise ValueError('Invalid noise gate noise shape/device/values')
    alpha = weights.float()
    # Exact endpoint values, with bounded pathwise derivatives near saturation.
    # This is a numerical stabilization, not the singular endpoint derivative.
    def root(value):
        return torch.where(value > 0, value.clamp_min(1e-6).sqrt(), torch.zeros_like(value))
    tokens = root(alpha)[:, None] * work + root(1 - alpha)[:, None] * noise.float()
    if not torch.isfinite(tokens).all():
        raise ValueError('Nonfinite noised tokens')
    return tokens.to(features.dtype)
