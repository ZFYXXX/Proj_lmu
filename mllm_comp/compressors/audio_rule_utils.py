"""Shared token strategy state and tensor contracts for audio compression methods."""
from typing import Any
from mllm_comp.core.errors import InvariantError


def audio_rule_metadata(owner: Any) -> dict[str, Any]:
    specs = getattr(owner, "rule_specs", {})
    if not specs:
        return {}
    result = {"rules": specs, "rules_changed": True, "canonical_evidence_applicable": False, "development_runtime": True}
    if "scorer" in specs:
        result["score_source"] = "custom"
    if "grouper" in specs:
        result["grouping_rule"] = "custom"
        if hasattr(getattr(owner, "spec", None), "pairing"):
            result["pairing"] = "custom"
        if hasattr(getattr(owner, "spec", None), "boundary_detector"):
            result["boundary_detector"] = "custom"
    if "aggregator" in specs:
        result["merge_value"] = "custom"
    if "transform" in specs:
        result["transform"] = "custom"
    return result


def token_scores(output: Any, reference: Any, *, length: int, nonnegative: bool = False) -> Any:
    import torch
    if not isinstance(output, torch.Tensor) or output.shape != (length,) or not output.is_floating_point():
        raise InvariantError("Audio scorer must return a floating vector matching its token or pair domain.")
    if not bool(torch.isfinite(output).all()) or (nonnegative and bool((output < 0).any())):
        raise InvariantError("Audio scorer returned invalid values.")
    return output.detach().to(reference.device)


def token_indices(output: Any, *, length: int, count: int, device: Any) -> Any:
    import torch
    if isinstance(output, (list, tuple)):
        if any(isinstance(i, bool) or not isinstance(i, int) for i in output):
            raise InvariantError("Audio selection requires integer indices.")
        output = torch.tensor(output, device=device, dtype=torch.long)
    if (not isinstance(output, torch.Tensor) or output.ndim != 1 or len(output) != count
            or output.dtype not in {torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8}):
        raise InvariantError("Audio selection must return the exact number of integer indices.")
    output = output.detach().to(device=device, dtype=torch.long)
    if bool((output < 0).any()) or bool((output >= length).any()) or output.unique().numel() != count:
        raise InvariantError("Audio selection indices must be unique and in range.")
    return output.sort().values


def contiguous_groups(output: Any, *, length: int, count: int | None = None, allow_tail: bool = False) -> tuple[tuple[int, ...], ...]:
    if not isinstance(output, (list, tuple)) or not output or (count is not None and len(output) != count):
        raise InvariantError("Audio groups must contain the required number of nonempty chronological groups.")
    groups = []
    for group in output:
        if not isinstance(group, (list, tuple)) or not group or any(isinstance(i, bool) or not isinstance(i, int) for i in group):
            raise InvariantError("Audio groups must contain integer token indices.")
        groups.append(tuple(group))
    flat = tuple(i for group in groups for i in group)
    if flat != tuple(range(len(flat))) or len(flat) > length or (not allow_tail and len(flat) != length):
        raise InvariantError("Audio groups must partition the chronological input without overlap.")
    return tuple(groups)


def aggregate_groups(values: Any, groups: Any, *, aggregator: Any = None) -> Any:
    import torch
    outputs = []
    for group_index, group in enumerate(groups):
        indices = torch.tensor(group, device=values.device, dtype=torch.long)
        members = values.index_select(0, indices)
        if aggregator is None:
            output = members.mean(0)
        else:
            output = aggregator({"values": members, "indices": tuple(group), "group_index": group_index, "weights": None})
            validate_values(output, values[group[0]])
        outputs.append(output)
    return torch.stack(outputs)


def validate_values(output: Any, reference: Any) -> None:
    import torch
    if (not isinstance(output, torch.Tensor) or output.shape != reference.shape
            or output.dtype != reference.dtype or output.device != reference.device):
        raise InvariantError("Audio transform must retain the requested shape, dtype and device.")
    if not bool(torch.isfinite(output).all()):
        raise InvariantError("Audio transform returned non-finite values.")
    if torch.is_grad_enabled() and reference.requires_grad and not output.requires_grad:
        raise InvariantError("Audio transform detached the value gradient.")


def _torch() -> Any:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - runtime dependency boundary
        raise InvariantError("Token compression requires PyTorch.") from exc
    return torch



class AudioTokenStrategyBase:
    metadata_fields: tuple[str, ...] = ()
    uses_distinct_merge_prune_importance = False

    def __init__(self, spec: Any) -> None:
        self.spec = spec
        self.rules = {}
        self.rule_specs = {}

    @property
    def score_source(self) -> str:
        return str(self.spec.score_source)

    @property
    def attention_layer_offset(self) -> int | None:
        if self.score_source == "encoder_last_layer_attention":
            return -1
        if self.score_source == "encoder_penultimate_layer_attention":
            return -2
        return None

    def runtime_metadata(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "score_source": self.score_source,
        }
        for name in self.metadata_fields:
            if hasattr(self.spec, name):
                result[name] = getattr(self.spec, name)
        result.update(audio_rule_metadata(self))
        return result

    def map_reference(self, reference: Any, *, indices: Any, details: dict[str, Any], scores: Any, keep_count: int) -> Any:
        if indices is None:
            raise InvariantError(f"Encoder attention reference transfer is not implemented for {type(self).__name__}.")
        return reference.index_select(0, indices.to(reference.device))



class AudioLayerTokenStrategyBase:
    metadata_fields: tuple[str, ...] = ()

    def __init__(self, spec: Any) -> None:
        self.spec = spec
        self.rules = {}
        self.rule_specs = {}

    def active_layer_indices(self, num_layers: int) -> tuple[int, ...]:
        return self.spec.resolve_layer_indices(num_layers)

    def runtime_metadata(self, active_layers: tuple[int, ...]) -> dict[str, Any]:
        result: dict[str, Any] = {
            "target_site": self.spec.target_site,
            "score_source": self.spec.score_source,
            "active_layers": list(active_layers),
        }
        for name in self.metadata_fields:
            if hasattr(self.spec, name):
                result[name] = getattr(self.spec, name)
        result.update(audio_rule_metadata(self))
        return result
