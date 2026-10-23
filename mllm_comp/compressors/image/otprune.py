"""Backbone-neutral OTPrune image-token selection kernel.

This follows the released OTPrune kernel construction and its greedy
Cholesky/log-determinant selection, while leaving prompt and media alignment to
the backbone adapter.  The method is defined at the projector boundary; using
it on Qwen3.8 is an integration adaptation, not a reproduction of the paper's
LLaVA evaluation setup.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

from mllm_comp.core.layout import LayoutDelta, LayoutTokenUnit


@dataclass(frozen=True)
class OTPruneSpec:
    retain_ratio: float = 0.5
    target_count: int | None = None
    gamma: float = 0.01

    def __post_init__(self) -> None:
        if isinstance(self.retain_ratio, bool) or not isinstance(
            self.retain_ratio, (int, float)
        ):
            raise TypeError("retain_ratio must be a real number.")
        if not 0.0 < float(self.retain_ratio) <= 1.0:
            raise ValueError("retain_ratio must be in (0, 1].")
        if self.target_count is not None and (
            isinstance(self.target_count, bool)
            or not isinstance(self.target_count, int)
            or self.target_count < 1
        ):
            raise ValueError("target_count must be a positive integer or None.")
        if isinstance(self.gamma, bool) or not isinstance(self.gamma, (int, float)):
            raise TypeError("gamma must be a real number.")
        if float(self.gamma) < 0.0:
            raise ValueError("gamma must be non-negative.")

    def resolve_target_count(self, source_count: int) -> int:
        if isinstance(source_count, bool) or not isinstance(source_count, int):
            raise TypeError("source_count must be an integer.")
        if source_count < 1:
            raise ValueError("source_count must be positive.")
        requested = (
            self.target_count
            if self.target_count is not None
            else int(round(float(self.retain_ratio) * source_count))
        )
        return min(source_count, max(1, requested))


@dataclass(frozen=True)
class OTPruneOutput:
    tokens: Any
    indices: Any
    selection_order: Any
    layout_delta: LayoutDelta
    metadata: Any

    @property
    def selected_indices(self) -> Any:
        return self.indices


def otprune_image(
    tokens: Any,
    *,
    spec: OTPruneSpec | None = None,
    media_unit: str | LayoutTokenUnit = "image_feature",
) -> OTPruneOutput:
    """Select a fixed-size, source-ordered subset with OTPrune's greedy kernel."""

    torch = _torch()
    _validate_tokens(tokens, torch)
    active = spec or OTPruneSpec()
    source_count = int(tokens.shape[0])
    target_count = active.resolve_target_count(source_count)

    if target_count == source_count:
        order = torch.arange(source_count, device=tokens.device)
        indices = order
    else:
        order = _greedy_logdet_indices(
            tokens,
            target_count,
            gamma=float(active.gamma),
            torch=torch,
        )
        indices = order.sort().values

    output = tokens.index_select(0, indices)
    kept = set(int(i) for i in indices.detach().cpu().tolist())
    dropped = tuple(i for i in range(source_count) if i not in kept)
    delta = LayoutDelta.drop(
        source_count,
        dropped,
        media_unit=media_unit,
    )
    return OTPruneOutput(
        tokens=output,
        indices=indices,
        selection_order=order,
        layout_delta=delta,
        metadata=MappingProxyType(
            {
                "algorithm": "otprune",
                "variant": "otprune.image",
                "source_count": source_count,
                "target_count": target_count,
                "gamma": float(active.gamma),
                "selection": "greedy_logdet_cholesky",
                "kernel": "I+gamma*S*S.T",
                "feature_normalization": "column_l2_across_tokens",
            }
        ),
    )


def _greedy_logdet_indices(tokens: Any, target_count: int, *, gamma: float, torch: Any) -> Any:
    """Greedy monotone log-det maximization using rank-one Cholesky updates."""

    source_count = int(tokens.shape[0])
    # Match the released implementation: normalize each feature dimension over
    # the token axis, form token similarities, then build I + gamma*S*S^T.
    work = tokens.detach().to(dtype=torch.float32)
    if not bool(torch.isfinite(work).all().item()):
        raise ValueError("tokens must contain only finite values.")
    column_norm = torch.linalg.vector_norm(work, dim=0, keepdim=True).clamp_min(1e-12)
    normalized = work / column_norm
    similarity = normalized @ normalized.transpose(0, 1)
    kernel = torch.eye(source_count, dtype=torch.float32, device=tokens.device)
    if gamma:
        kernel = kernel + gamma * (similarity @ similarity.transpose(0, 1))
    kernel = (kernel + kernel.transpose(0, 1)) * 0.5

    # L[:, j] stores the residual Cholesky column for the item chosen at step j.
    # Residual diagonal is the determinant gain for adding each candidate.
    factor = torch.zeros(
        (source_count, target_count), dtype=torch.float32, device=tokens.device
    )
    residual = torch.diagonal(kernel).clone()
    selected = torch.zeros(source_count, dtype=torch.bool, device=tokens.device)
    order = torch.empty(target_count, dtype=torch.long, device=tokens.device)
    eps = torch.finfo(torch.float32).eps
    for step in range(target_count):
        scores = residual.masked_fill(selected, float("-inf"))
        chosen = int(torch.argmax(scores).item())
        order[step] = chosen
        selected[chosen] = True
        pivot = residual[chosen].clamp_min(eps).sqrt()
        if step == 0:
            column = kernel[:, chosen] / pivot
        else:
            correction = factor[:, :step] @ factor[chosen, :step]
            column = (kernel[:, chosen] - correction) / pivot
        factor[:, step] = column
        residual = (residual - column.square()).clamp_min(0.0)
    return order


def _validate_tokens(tokens: Any, torch: Any) -> None:
    if not torch.is_tensor(tokens):
        raise TypeError("tokens must be a torch.Tensor.")
    if tokens.ndim != 2:
        raise ValueError("tokens must have shape [num_tokens, hidden_size].")
    if int(tokens.shape[0]) < 1 or int(tokens.shape[1]) < 1:
        raise ValueError("tokens must have non-empty token and feature dimensions.")
    if not tokens.is_floating_point():
        raise TypeError("tokens must have a floating-point dtype.")


def _torch() -> Any:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - depends on runtime env
        raise RuntimeError("OTPrune requires PyTorch when its kernel is executed.") from exc
    return torch


__all__ = ["OTPruneOutput", "OTPruneSpec", "otprune_image"]
