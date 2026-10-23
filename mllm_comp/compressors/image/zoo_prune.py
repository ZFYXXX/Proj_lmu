"""Backbone-neutral ZOO-Prune projector sensitivity and SADS selection.

The shared implementation owns the released zeroth-order finite-difference
estimator, low-rank projector proxy, and multiplicative sensitivity/diversity
selection.  Backbone bindings provide the exact native projector input/layers,
image or crop partitioning, and layout legalization.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from types import MappingProxyType
from typing import Any, Mapping, Sequence

from mllm_comp.core.layout import LayoutDelta, LayoutTokenUnit
from ._rules import rule_metadata, score_rule, selection_rule


ZOO_PRUNE_CONFIG_KEYS = frozenset(
    {
        "retention_ratio",
        "target_count",
        "num_directions",
        "step_size",
        "projector_proxy",
        "low_rank_rank",
        "low_rank_solver",
        "direction_batch_size",
        "seed",
        "min_visual_tokens",
    }
)


def merge_zoo_prune_executor_config(
    *,
    stage: Any,
    binding_config: Any,
    extra_keys: frozenset[str] = frozenset(),
) -> dict[str, Any]:
    """Merge ZOO-Prune knobs and reject misspelled explicit config."""

    allowed = ZOO_PRUNE_CONFIG_KEYS | extra_keys
    result: dict[str, Any] = {}
    if stage is not None:
        for name in ("operation", "criterion", "schedule"):
            value = getattr(stage, name, None)
            if isinstance(value, Mapping):
                result.update(
                    {key: item for key, item in value.items() if key in allowed}
                )
    for source, value in (
        ("stage.config", getattr(stage, "config", None)),
        ("binding_config.config", getattr(binding_config, "config", None)),
    ):
        if not isinstance(value, Mapping):
            continue
        unknown = sorted(set(value) - allowed)
        if unknown:
            raise ValueError(f"Unknown ZOO-Prune {source} keys: {unknown}.")
        result.update(dict(value))
    if "retention_ratio" in result and "target_count" in result:
        raise ValueError(
            "ZOO-Prune accepts only one explicit budget: retention_ratio or "
            "target_count."
        )
    return result


@dataclass(frozen=True)
class ZOOPruneSpec:
    """Released defaults for one independently selected image or crop."""

    retention_ratio: float = 0.1112
    target_count: int | None = None
    num_directions: int = 64
    step_size: float = 0.01
    projector_proxy: str = "low_rank"
    low_rank_rank: int = 128
    low_rank_solver: str = "exact"
    direction_batch_size: int = 4
    seed: int = 0
    min_visual_tokens: int = 1

    def __post_init__(self) -> None:
        _require_finite_real("retention_ratio", self.retention_ratio)
        if not 0.0 < float(self.retention_ratio) <= 1.0:
            raise ValueError("retention_ratio must be in (0, 1].")
        _require_finite_real("step_size", self.step_size)
        if float(self.step_size) <= 0.0:
            raise ValueError("step_size must be positive.")
        for name in (
            "num_directions",
            "low_rank_rank",
            "direction_batch_size",
            "min_visual_tokens",
        ):
            _require_positive_int(name, getattr(self, name))
        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise TypeError("seed must be an integer.")
        if self.target_count is not None:
            _require_positive_int("target_count", self.target_count)
            if self.target_count < self.min_visual_tokens:
                raise ValueError("target_count must be at least min_visual_tokens.")
        proxy = str(self.projector_proxy).strip().lower()
        solver = str(self.low_rank_solver).strip().lower()
        if proxy not in {"full", "low_rank"}:
            raise ValueError("projector_proxy must be 'full' or 'low_rank'.")
        if solver not in {"exact", "randomized"}:
            raise ValueError("low_rank_solver must be 'exact' or 'randomized'.")
        object.__setattr__(self, "projector_proxy", proxy)
        object.__setattr__(self, "low_rank_solver", solver)

    def resolve_target_count(self, source_count: int) -> int:
        _require_positive_int("source_count", source_count)
        requested = self.target_count
        if requested is None:
            requested = int(float(self.retention_ratio) * source_count)
        return min(source_count, max(self.min_visual_tokens, requested))


@dataclass(frozen=True)
class ZOOPruneOutput:
    """Source-ordered projector tokens and their exact drop mapping."""

    tokens: Any
    indices: Any
    selection_order: Any
    sensitivity: Any
    layout_delta: LayoutDelta
    metadata: Mapping[str, Any]

    @property
    def selected_indices(self) -> Any:
        return self.indices


class _LowRankLinear:
    def __init__(self, left: Any, right: Any, bias: Any | None) -> None:
        self.left = left
        self.right = right
        self.bias = bias

    def __call__(self, inputs: Any) -> Any:
        torch = _torch()
        hidden = torch.nn.functional.linear(inputs, self.right)
        return torch.nn.functional.linear(hidden, self.left, self.bias)


class _ProjectorChain:
    def __init__(self, layers: Sequence[Any], *, solver_used: str) -> None:
        self.layers = tuple(layers)
        self.solver_used = solver_used

    def __call__(self, inputs: Any) -> Any:
        hidden = inputs
        for layer in self.layers:
            hidden = layer(hidden)
        return hidden


class ZOOProjectorProxyCache:
    """Activation-owned lazy proxy cache; never enters a model state dict."""

    def __init__(self) -> None:
        self._key: tuple[Any, ...] | None = None
        self._proxy: _ProjectorChain | None = None

    def resolve(
        self,
        projection_layers: Sequence[Any],
        spec: ZOOPruneSpec,
    ) -> _ProjectorChain:
        layers = tuple(projection_layers)
        if not layers:
            raise ValueError("projection_layers must not be empty.")
        key = _projector_cache_key(layers, spec)
        if key is None or self._key != key or self._proxy is None:
            self._proxy = build_zoo_projector_proxy(layers, spec=spec)
            self._key = key
        return self._proxy

    def clear(self) -> None:
        self._key = None
        self._proxy = None


def sample_zoo_directions(
    count: int,
    dimension: int,
    *,
    device: Any,
    dtype: Any,
    seed: int,
) -> Any:
    """Sample shared unit Gaussian directions without advancing global RNG."""

    _require_positive_int("count", count)
    _require_positive_int("dimension", dimension)
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError("seed must be an integer.")
    torch = _torch()
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    directions = torch.randn(
        count,
        dimension,
        device=device,
        dtype=torch.float32,
        generator=generator,
    )
    return torch.nn.functional.normalize(
        directions, dim=-1, eps=1e-12
    ).to(dtype=dtype)


def build_zoo_projector_proxy(
    projection_layers: Sequence[Any],
    *,
    spec: ZOOPruneSpec,
) -> _ProjectorChain:
    """Build the released proxy while retaining native nonlinear layers."""

    torch = _torch()
    layers = tuple(projection_layers)
    if not layers:
        raise ValueError("projection_layers must not be empty.")
    if spec.projector_proxy == "full":
        if not all(callable(layer) for layer in layers):
            raise TypeError("every projection layer must be callable.")
        return _ProjectorChain(layers, solver_used="full")

    proxy_layers: list[Any] = []
    solver_names: list[str] = []
    for layer in layers:
        if isinstance(layer, torch.nn.Linear):
            left, right, solver_used = _truncated_svd(
                layer.weight,
                spec.low_rank_rank,
                solver=spec.low_rank_solver,
                seed=spec.seed,
            )
            bias = None if layer.bias is None else layer.bias.detach()
            proxy_layers.append(_LowRankLinear(left, right, bias))
            solver_names.append(solver_used)
            continue
        if not callable(layer):
            raise TypeError("every projection layer must be callable.")
        parameters = getattr(layer, "parameters", None)
        if callable(parameters) and any(True for _ in parameters(recurse=True)):
            raise ValueError(
                "ZOO-Prune proxy accepts trainable parameters only in Linear layers."
            )
        proxy_layers.append(layer)
    if not solver_names:
        raise ValueError("ZOO-Prune could not find a Linear projector layer.")
    return _ProjectorChain(proxy_layers, solver_used="+".join(solver_names))


def estimate_zoo_sensitivity(
    projector_inputs: Any,
    proxy: Any,
    *,
    spec: ZOOPruneSpec,
    directions: Any | None = None,
) -> Any:
    """Estimate the released centered finite-difference response per token."""

    torch = _torch()
    _validate_features(projector_inputs, "projector_inputs", torch=torch)
    if not callable(proxy):
        raise TypeError("proxy must be callable.")
    token_count, dimension = (int(value) for value in projector_inputs.shape)
    if directions is None:
        directions = sample_zoo_directions(
            spec.num_directions,
            dimension,
            device=projector_inputs.device,
            dtype=projector_inputs.dtype,
            seed=spec.seed,
        )
    if not isinstance(directions, torch.Tensor) or tuple(directions.shape) != (
        spec.num_directions,
        dimension,
    ):
        raise ValueError(
            "directions must have shape [num_directions, projector_input_dim]."
        )
    total = torch.zeros(
        token_count, device=projector_inputs.device, dtype=torch.float32
    )
    step_size = float(spec.step_size)
    with torch.no_grad():
        inputs = projector_inputs.detach()
        for start in range(0, spec.num_directions, spec.direction_batch_size):
            direction_batch = directions[
                start : start + spec.direction_batch_size
            ]
            offsets = step_size * direction_batch[:, None, :]
            plus = proxy(inputs.unsqueeze(0) + offsets)
            minus = proxy(inputs.unsqueeze(0) - offsets)
            response = (plus - minus).float().norm(dim=-1) / (2.0 * step_size)
            total.add_(response.sum(dim=0))
    sensitivity = total.div(float(spec.num_directions))
    if not bool(torch.isfinite(sensitivity).all().item()):
        raise ValueError("ZOO-Prune sensitivity must be finite.")
    return sensitivity


def sensitivity_aware_diverse_indices(
    projected_tokens: Any,
    sensitivity: Any,
    target_count: int,
) -> tuple[Any, Any]:
    """Return SADS source-order indices and greedy selection order."""

    torch = _torch()
    _validate_features(projected_tokens, "projected_tokens", torch=torch)
    if not isinstance(sensitivity, torch.Tensor) or sensitivity.ndim != 1:
        raise ValueError("sensitivity must have shape [tokens].")
    source_count = int(projected_tokens.shape[0])
    if int(sensitivity.shape[0]) != source_count:
        raise ValueError("sensitivity length must match projected tokens.")
    _require_positive_int("target_count", target_count)
    if target_count > source_count:
        raise ValueError("target_count cannot exceed the source token count.")
    if not bool(torch.isfinite(sensitivity).all().item()):
        raise ValueError("sensitivity must contain only finite values.")

    features = torch.nn.functional.normalize(
        projected_tokens.detach().float(), dim=-1, eps=1e-8
    )
    work = sensitivity.detach().float()
    normalized = (work - work.min()) / (work.max() - work.min() + 1e-8)
    selected = torch.empty(
        target_count, dtype=torch.long, device=projected_tokens.device
    )
    blocked = torch.zeros(
        source_count, dtype=torch.bool, device=projected_tokens.device
    )
    first = torch.argmax(normalized)
    selected[0] = first
    blocked[first] = True
    max_similarity = features @ features[first]
    for step in range(1, target_count):
        score = normalized * (1.0 - max_similarity)
        score = score.masked_fill(blocked, float("-inf"))
        index = torch.argmax(score)
        selected[step] = index
        blocked[index] = True
        max_similarity = torch.maximum(max_similarity, features @ features[index])
    return selected.sort().values, selected


def zoo_prune_image_tokens(
    projected_tokens: Any,
    projector_inputs: Any,
    projection_layers: Sequence[Any],
    *,
    spec: ZOOPruneSpec | None = None,
    proxy_cache: ZOOProjectorProxyCache | None = None,
    media_unit: str | LayoutTokenUnit = "image_feature",
    source_layout: Mapping[str, Any] | None = None,
    result_layout: Mapping[str, Any] | None = None,
    rules: Mapping[str, Any] | None = None,
) -> ZOOPruneOutput:
    """Score and select one native image/crop projector-token sequence."""

    torch = _torch()
    _validate_features(projected_tokens, "projected_tokens", torch=torch)
    _validate_features(projector_inputs, "projector_inputs", torch=torch)
    if int(projected_tokens.shape[0]) != int(projector_inputs.shape[0]):
        raise ValueError("projected token and projector input counts must match.")
    if projected_tokens.device != projector_inputs.device:
        raise ValueError("projected tokens and projector inputs must share a device.")
    active = spec or ZOOPruneSpec()
    source_count = int(projected_tokens.shape[0])
    target_count = active.resolve_target_count(source_count)
    if target_count == source_count:
        indices = torch.arange(source_count, device=projected_tokens.device)
        selection_order = indices
        sensitivity = torch.zeros(
            source_count, device=projected_tokens.device, dtype=torch.float32
        )
        solver_used = "identity"
    else:
        cache = proxy_cache or ZOOProjectorProxyCache()
        proxy = cache.resolve(projection_layers, active)
        sensitivity = estimate_zoo_sensitivity(
            projector_inputs,
            proxy,
            spec=active,
        )
        request = {"projected_tokens": projected_tokens, "projector_inputs": projector_inputs,
                   "proxy": proxy, "keep_count": target_count, "spec": active}
        sensitivity = score_rule(rules, "scorer", request, sensitivity)
        selection_order = selection_rule(rules, "selector", {**request, "scores": sensitivity},
            count=source_count, keep_count=target_count, device=projected_tokens.device)
        if selection_order is None:
            indices, selection_order = sensitivity_aware_diverse_indices(
                projected_tokens, sensitivity, target_count)
        else:
            indices = selection_order.sort().values
        solver_used = proxy.solver_used
    tokens = (
        projected_tokens
        if target_count == source_count
        else projected_tokens.index_select(0, indices)
    )
    kept = tuple(int(value) for value in indices.detach().cpu().tolist())
    kept_set = set(kept)
    dropped = tuple(index for index in range(source_count) if index not in kept_set)
    delta = LayoutDelta.drop(
        source_count,
        dropped,
        media_unit=media_unit,
        source_layout=source_layout,
        result_layout=result_layout,
    )
    order = tuple(
        int(value) for value in selection_order.detach().cpu().tolist()
    )
    metadata = MappingProxyType(
        {
            "algorithm": "zoo_prune",
            **rule_metadata(rules),
            **({"sensitivity_rule": "custom"} if rules and "scorer" in rules else {}),
            **({"selection_rule": "custom"} if rules and "selector" in rules else {}),
            "variant": "zoo_prune.image",
            "retention_ratio": float(active.retention_ratio),
            "target_count": active.target_count,
            "effective_target_count": target_count,
            "num_directions": active.num_directions,
            "step_size": float(active.step_size),
            "projector_proxy": active.projector_proxy,
            "low_rank_rank": active.low_rank_rank,
            "low_rank_solver_requested": active.low_rank_solver,
            "low_rank_solver_used": solver_used,
            "direction_batch_size": active.direction_batch_size,
            "seed": active.seed,
            "source_count": source_count,
            "result_count": target_count,
            "selected_indices": kept,
            "selection_order": order,
            "token_order": "source_index",
            "sensitivity_min": float(sensitivity.min().item()),
            "sensitivity_max": float(sensitivity.max().item()),
            "compression_applied": target_count < source_count,
        }
    )
    return ZOOPruneOutput(
        tokens=tokens,
        indices=indices,
        selection_order=selection_order,
        sensitivity=sensitivity,
        layout_delta=delta,
        metadata=metadata,
    )


def _truncated_svd(
    weight: Any,
    rank: int,
    *,
    solver: str,
    seed: int,
) -> tuple[Any, Any, str]:
    torch = _torch()
    work = weight.detach().float()
    resolved_rank = min(rank, int(work.shape[0]), int(work.shape[1]))
    solver_used = solver
    if solver == "exact":
        try:
            u, singular, vh = torch.linalg.svd(work, full_matrices=False)
            left = u[:, :resolved_rank] * singular[:resolved_rank].unsqueeze(0)
            right = vh[:resolved_rank]
        except RuntimeError:
            solver_used = "randomized_fallback"
            left, right = _randomized_svd(work, resolved_rank, seed=seed)
    else:
        left, right = _randomized_svd(work, resolved_rank, seed=seed)
    return (
        left.to(device=weight.device, dtype=weight.dtype),
        right.to(device=weight.device, dtype=weight.dtype),
        solver_used,
    )


def _randomized_svd(work: Any, rank: int, *, seed: int) -> tuple[Any, Any]:
    torch = _torch()
    if bool(getattr(work, "is_cuda", False)):
        index = work.device.index
        device_index = torch.cuda.current_device() if index is None else int(index)
        with torch.random.fork_rng(devices=[device_index]):
            with torch.cuda.device(device_index):
                torch.cuda.manual_seed(seed)
                u, singular, v = torch.svd_lowrank(work, q=rank, niter=4)
    else:
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(seed)
            u, singular, v = torch.svd_lowrank(work, q=rank, niter=4)
    left = u[:, :rank] * singular[:rank].unsqueeze(0)
    right = v[:, :rank].transpose(0, 1)
    return left, right


def _projector_cache_key(
    layers: tuple[Any, ...],
    spec: ZOOPruneSpec,
) -> tuple[Any, ...] | None:
    torch = _torch()
    layer_keys: list[Any] = []
    for layer in layers:
        if isinstance(layer, torch.nn.Linear):
            weight = layer.weight
            version = None
            if spec.projector_proxy == "low_rank":
                # Inference tensors have no version counter. Rebuild their
                # detached proxy rather than reuse it across unseen updates.
                if torch.is_inference(weight):
                    return None
                version = weight._version
            layer_keys.append(
                (
                    "linear",
                    id(layer),
                    int(weight.data_ptr()),
                    tuple(int(value) for value in weight.shape),
                    str(weight.device),
                    str(weight.dtype),
                    version,
                )
            )
        else:
            layer_keys.append(("exact", id(layer)))
    return (
        spec.projector_proxy,
        spec.low_rank_rank,
        spec.low_rank_solver,
        spec.seed,
        tuple(layer_keys),
    )


def _validate_features(value: Any, name: str, *, torch: Any) -> None:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor.")
    if value.ndim != 2 or min(int(item) for item in value.shape) < 1:
        raise ValueError(f"{name} must have shape [tokens, hidden].")
    if not value.is_floating_point():
        raise TypeError(f"{name} must use a floating dtype.")
    if not bool(torch.isfinite(value).all().item()):
        raise ValueError(f"{name} must contain only finite values.")


def _require_finite_real(name: str, value: Any) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a real number.")
    if not math.isfinite(float(value)):
        raise ValueError(f"{name} must be finite.")


def _require_positive_int(name: str, value: Any) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer.")
    if value < 1:
        raise ValueError(f"{name} must be at least one.")


def _torch() -> Any:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - runtime dependency
        raise RuntimeError("ZOO-Prune execution requires PyTorch.") from exc
    return torch


__all__ = [
    "ZOO_PRUNE_CONFIG_KEYS",
    "ZOOProjectorProxyCache",
    "ZOOPruneOutput",
    "ZOOPruneSpec",
    "build_zoo_projector_proxy",
    "estimate_zoo_sensitivity",
    "merge_zoo_prune_executor_config",
    "sample_zoo_directions",
    "sensitivity_aware_diverse_indices",
    "zoo_prune_image_tokens",
]
