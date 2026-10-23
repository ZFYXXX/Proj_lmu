"""Small return-contract helpers for rules inside existing video kernels."""

from typing import Any

from mllm_comp.core.errors import BackendIntegrationError, ConfigError


def load_video_rules(stage: Any, *, contract_projection: Any = None) -> tuple[dict, dict]:
    from mllm_comp.compressors.rules import builtin_rule_specs, load_rule_specs, rule_specs

    try:
        specs = (
            rule_specs(stage, supported_rules=contract_projection.variant.supported_rules)
            if contract_projection is not None else builtin_rule_specs(stage)
        )
        return load_rule_specs(specs), specs
    except ConfigError as exc:
        raise BackendIntegrationError(f"Invalid video compression rules: {exc}") from exc


def video_rule_metadata(specs: dict) -> dict:
    return ({"rules": specs, "rules_changed": True,
             "canonical_evidence_applicable": False, "development_runtime": True,
             "support_activation_claimed": False, "real_checkpoint_verified": False}
            if specs else {})


def indices(value: Any, *, count: int | None, candidates: Any, name: str) -> tuple[int, ...]:
    import torch

    if isinstance(value, torch.Tensor):
        if value.ndim != 1 or value.dtype not in (torch.int32, torch.int64):
            raise BackendIntegrationError(f"{name} must return one-dimensional integer indices")
        value = value.detach().cpu().tolist()
    try:
        result = tuple(value)
    except TypeError as exc:
        raise BackendIntegrationError(f"{name} must return integer indices") from exc
    if ((count is not None and len(result) != count)
            or any(isinstance(i, bool) or not isinstance(i, int) for i in result)
            or len(set(result)) != len(result)
            or not set(result).issubset(candidates)):
        raise BackendIntegrationError(f"{name} must return {count} distinct candidate indices")
    return result


def scores(rule: Any, request: dict, *, count: int, reference: Any, name: str) -> Any:
    import torch

    with torch.no_grad():
        value = rule(request)
    if (not isinstance(value, torch.Tensor) or not value.is_floating_point()
            or tuple(value.shape) != (count,) or not bool(torch.isfinite(value).all().item())):
        raise BackendIntegrationError(f"{name} must return finite floating scores [{count}]")
    return value.detach().to(reference.device)


def select(rule: Any, request: dict, *, count: int | None, candidates: Any, name: str) -> tuple[int, ...]:
    import torch

    with torch.no_grad():
        value = rule(request)
    return indices(value, count=count, candidates=candidates, name=name)


def groups(rule: Any, request: dict, *, count: int | None, candidates: Any, name: str,
           allow_empty: bool = False) -> tuple[tuple[int, ...], ...]:
    import torch

    with torch.no_grad():
        value = rule(request)
    try:
        result = tuple(tuple(group) for group in value)
    except TypeError as exc:
        raise BackendIntegrationError(f"{name} must return nonempty source groups") from exc
    if (count is not None and len(result) != count) or (not allow_empty and any(not group for group in result)):
        raise BackendIntegrationError(f"{name} must return {count} nonempty groups")
    flat = tuple(i for group in result for i in group)
    indices(flat, count=len(tuple(candidates)), candidates=candidates, name=name)
    return tuple(tuple(sorted(group)) for group in result)


def value(rule: Any, request: dict, *, reference: Any, shape: tuple, name: str) -> Any:
    import torch

    result = rule(request)
    if (not isinstance(result, torch.Tensor) or not result.is_floating_point()
            or tuple(result.shape) != shape or result.dtype != reference.dtype
            or result.device != reference.device or not bool(torch.isfinite(result).all().item())):
        raise BackendIntegrationError(f"{name} must return finite values {shape} on the input dtype/device")
    if torch.is_grad_enabled() and reference.requires_grad and not result.requires_grad:
        raise BackendIntegrationError(f"{name} detached the differentiable values")
    return result
