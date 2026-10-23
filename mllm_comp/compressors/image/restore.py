"""Backbone-neutral RESTORE hybrid reduction and attention calibration.

RESTORE wraps a base visual-token reducer with distinctive-anchor merging,
preserves one real source position for every result token, and rectifies the
decoder attention logits with token mass and RoPE-distance terms.  This module
owns those equations, calibrated position/mass state, and mask composition.
Backbone bindings own visual metric capture, AnyRes/M-RoPE legalization,
native causal masks, and request/cache lifetime.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from types import MappingProxyType
from typing import Any, Mapping, Sequence

from mllm_comp.compressors.image.vispruner import select_vispruner_indices
from mllm_comp.core.layout import LayoutDelta, LayoutTokenUnit, OrderingPolicy
from mllm_comp.core.errors import BackendIntegrationError, InvariantError
from ._rules import rule_metadata, score_rule, selection_rule


RESTORE_CONFIG_KEYS = frozenset(
    {
        "target_count",
        "tokens_per_crop",
        "pruning_ratio",
        "important_ratio",
        "removal_step",
        "anchor_mask_mode",
        "position_strategy",
        "calibration_constant",
        "first_layer_size",
        "min_visual_tokens",
    }
)


def merge_restore_executor_config(
    *,
    stage: Any,
    binding_config: Any,
    extra_keys: frozenset[str] = frozenset(),
) -> dict[str, Any]:
    """Merge RESTORE knobs and fail fast on misspelled explicit fields."""

    allowed = RESTORE_CONFIG_KEYS | extra_keys
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
            raise ValueError(f"Unknown RESTORE {source} keys: {unknown}.")
        result.update(dict(value))
    return result


@dataclass(frozen=True)
class RestoreSpec:
    """Released RESTORE defaults for one image or one LLaVA AnyRes crop."""

    target_count: int = 64
    pruning_ratio: float = 0.5
    important_ratio: float = 0.5
    removal_step: int = 8
    anchor_mask_mode: str = "released"
    position_strategy: str = "median"
    calibration_constant: float = 2.0
    first_layer_size: str = "uniform"
    min_visual_tokens: int = 2

    def __post_init__(self) -> None:
        for name in ("target_count", "removal_step", "min_visual_tokens"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be an integer.")
            if value < 1:
                raise ValueError(f"{name} must be at least one.")
        if self.min_visual_tokens < 2:
            raise ValueError("RESTORE min_visual_tokens must be at least two.")
        if self.target_count < self.min_visual_tokens:
            raise ValueError("target_count must be at least min_visual_tokens.")
        for name in (
            "pruning_ratio",
            "important_ratio",
            "calibration_constant",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(f"{name} must be a real number.")
            if not math.isfinite(float(value)):
                raise ValueError(f"{name} must be finite.")
        if not 0.0 < float(self.pruning_ratio) < 1.0:
            raise ValueError("pruning_ratio must be in (0, 1).")
        if not 0.0 <= float(self.important_ratio) <= 1.0:
            raise ValueError("important_ratio must be in [0, 1].")
        if float(self.calibration_constant) <= 1.0:
            raise ValueError("calibration_constant must be greater than one.")
        if self.anchor_mask_mode not in {"released", "paper"}:
            raise ValueError("anchor_mask_mode must be 'released' or 'paper'.")
        if self.position_strategy not in {"first", "median", "mean", "last"}:
            raise ValueError("Unsupported RESTORE position_strategy.")
        if self.first_layer_size not in {"uniform", "cluster"}:
            raise ValueError("first_layer_size must be 'uniform' or 'cluster'.")

    def resolve_target_count(
        self,
        source_count: int,
        *,
        target_count: int | None = None,
    ) -> int:
        _positive_integer(source_count, "source_count")
        requested = self.target_count if target_count is None else target_count
        _positive_integer(requested, "target_count")
        return min(source_count, max(self.min_visual_tokens, requested))


@dataclass(frozen=True)
class RestoreAnchorMergeResult:
    tokens: Any
    representative_indices: Any
    token_sizes: Any
    membership: Any


@dataclass(frozen=True)
class RestoreOutput:
    tokens: Any
    representative_indices: Any
    token_sizes: Any
    membership: Any
    pristine_indices: Any
    layout_delta: LayoutDelta
    metadata: Mapping[str, Any]


@dataclass(frozen=True)
class RestoreAttentionState:
    """Position and mass tensors for one request; native providers own its lifetime."""

    key_positions: Any
    key_sizes: Any
    first_key_sizes: Any

    @classmethod
    def prefill(
        cls,
        position_ids: Any,
        *,
        image_indices: Any,
        visual_sizes: Any,
        source_visual_count: int,
        first_layer_size: str,
        reference: Any,
    ) -> RestoreAttentionState:
        sizes = _embed_restore_sizes(
            int(reference.shape[1]),
            image_indices=image_indices,
            visual_sizes=visual_sizes,
            device=reference.device,
        ).unsqueeze(0)
        first_sizes = sizes
        if first_layer_size == "uniform":
            if source_visual_count < 1:
                raise InvariantError("RESTORE source visual mass must be positive.")
            first_sizes = sizes.clone()
            first_sizes[0, image_indices.to(reference.device)] = float(
                source_visual_count
            ) / int(image_indices.numel())
        return cls(position_ids, sizes, first_sizes)

    def append(self, position_ids: Any, *, reference: Any) -> RestoreAttentionState:
        """Extend calibrated keys with the native decode positions and unit mass."""
        torch = _torch()
        units = torch.ones(
            reference.shape[:2], dtype=torch.float32, device=reference.device
        )
        return type(self)(
            torch.cat(
                (self.key_positions.to(position_ids.device), position_ids), dim=-1
            ),
            torch.cat((self.key_sizes.to(reference.device), units), dim=-1),
            torch.cat((self.first_key_sizes.to(reference.device), units), dim=-1),
        )

    def attention_mask(
        self,
        base_mask: Any,
        query_positions: Any,
        inv_freq: Any,
        *,
        layer_index: int,
        reference: Any,
        calibration_constant: float,
        mrope_section: Sequence[int] | None = None,
        rules: Mapping[str, Any] | None = None,
    ) -> Any:
        """Calibrate the existing native mask without changing physical KV storage."""
        torch = _torch()
        sizes = self.first_key_sizes if layer_index == 0 else self.key_sizes
        bias = restore_attention_bias(
            query_positions,
            self.key_positions,
            sizes,
            inv_freq,
            calibration_constant=calibration_constant,
            mrope_section=mrope_section,
            dtype=reference.dtype,
            rules=rules,
        )
        query_length, key_length = int(reference.shape[1]), int(sizes.shape[-1])
        if tuple(bias.shape[-2:]) != (query_length, key_length):
            raise InvariantError(
                "RESTORE attention bias has the wrong query/key shape."
            )
        if base_mask is None:
            query = torch.arange(query_length, device=reference.device)[:, None]
            key = torch.arange(key_length, device=reference.device)[None, :]
            blocked = key > (key_length - query_length + query)
            base_mask = torch.zeros(
                (1, 1, query_length, key_length),
                dtype=reference.dtype,
                device=reference.device,
            ).masked_fill(
                blocked.view(1, 1, query_length, key_length),
                torch.finfo(reference.dtype).min,
            )
        elif not isinstance(base_mask, torch.Tensor):
            raise BackendIntegrationError(
                "RESTORE requires a tensor causal attention mask."
            )
        elif base_mask.dtype == torch.bool:
            base_mask = torch.where(
                base_mask,
                torch.zeros((), dtype=reference.dtype, device=base_mask.device),
                torch.full(
                    (),
                    torch.finfo(reference.dtype).min,
                    dtype=reference.dtype,
                    device=base_mask.device,
                ),
            )
        if tuple(base_mask.shape[-2:]) != (query_length, key_length):
            raise InvariantError("RESTORE causal mask does not match calibrated keys.")
        return base_mask + bias.to(device=base_mask.device, dtype=base_mask.dtype)


def build_restore_prefill_position_ids(
    sequence_length: int,
    *,
    visual_span: tuple[int, int],
    representative_positions: Any,
    original_visual_length: int,
    device: Any,
) -> Any:
    """Preserve representative visual positions and the dense text offset."""

    torch = _torch()
    _positive_integer(sequence_length, "sequence_length")
    _positive_integer(original_visual_length, "original_visual_length")
    start, end = (int(value) for value in visual_span)
    if not 0 <= start <= end <= sequence_length:
        raise ValueError("Invalid RESTORE visual span.")
    positions = torch.as_tensor(
        representative_positions,
        dtype=torch.long,
        device=device,
    )
    if positions.ndim != 1 or int(positions.numel()) != end - start:
        raise ValueError("RESTORE representative positions do not match visual span.")
    if int(positions.numel()) and (
        int(positions.min().item()) < 0
        or int(positions.max().item()) >= original_visual_length
    ):
        raise ValueError("RESTORE representative position exceeds visual source.")
    prefix = torch.arange(start, device=device, dtype=torch.long)
    visual = positions + start
    suffix = torch.arange(
        start + original_visual_length,
        start + original_visual_length + sequence_length - end,
        device=device,
        dtype=torch.long,
    )
    return torch.cat((prefix, visual, suffix))


def build_restore_prefill_sizes(
    sequence_length: int,
    *,
    visual_span: tuple[int, int],
    visual_sizes: Any,
    device: Any,
) -> Any:
    """Embed visual cluster masses in a full prompt-sized key-size vector."""

    torch = _torch()
    _positive_integer(sequence_length, "sequence_length")
    start, end = (int(value) for value in visual_span)
    if not 0 <= start <= end <= sequence_length:
        raise ValueError("Invalid RESTORE visual span.")
    return _embed_restore_sizes(
        sequence_length,
        image_indices=torch.arange(start, end, device=device),
        visual_sizes=visual_sizes,
        device=device,
    )


def _embed_restore_sizes(sequence_length, *, image_indices, visual_sizes, device):
    torch = _torch()
    sizes = torch.as_tensor(visual_sizes, dtype=torch.float32, device=device)
    if sizes.ndim != 1 or int(sizes.numel()) != int(image_indices.numel()):
        raise ValueError("RESTORE visual sizes do not match image positions.")
    if not bool(torch.isfinite(sizes).all().item()) or bool((sizes <= 0).any().item()):
        raise ValueError("RESTORE visual sizes must be finite and positive.")
    result = torch.ones(sequence_length, dtype=torch.float32, device=device)
    result[image_indices.to(device)] = sizes
    return result


def _representative_position(indices: Any, strategy: str) -> Any:
    ordered = indices.sort().values
    if strategy == "first":
        return ordered[0]
    if strategy == "last":
        return ordered[-1]
    if strategy == "mean":
        mean = ordered.float().mean()
        return ordered[(ordered.float() - mean).abs().argmin()]
    return ordered[(int(ordered.numel()) - 1) // 2]


def distinctive_anchor_merge(
    selection_features: Any,
    source_indices: Any,
    *,
    anchor_count: int,
    anchor_mask_mode: str = "released",
    position_strategy: str = "median",
    output_tokens: Any | None = None,
    source_count: int | None = None,
    rules: Mapping[str, Any] | None = None,
) -> RestoreAnchorMergeResult:
    """Merge every supplied source into one distinctive anchor cluster."""

    torch = _torch()
    _floating_matrix(selection_features, "selection_features", torch=torch)
    values = selection_features if output_tokens is None else output_tokens
    _floating_matrix(values, "output_tokens", torch=torch)
    if int(values.shape[0]) != int(selection_features.shape[0]):
        raise ValueError("RESTORE selection and output token counts must match.")
    sources = torch.as_tensor(
        source_indices,
        dtype=torch.long,
        device=selection_features.device,
    )
    if sources.ndim != 1 or int(sources.numel()) != int(selection_features.shape[0]):
        raise ValueError("RESTORE source indices must match selection features.")
    if int(sources.numel()) < 1 or int(sources.min().item()) < 0:
        raise ValueError("RESTORE source indices must be non-negative and non-empty.")
    if int(torch.unique(sources).numel()) != int(sources.numel()):
        raise ValueError("RESTORE source indices must be unique.")
    _positive_integer(anchor_count, "anchor_count")
    if anchor_count > int(sources.numel()):
        raise ValueError("anchor_count cannot exceed the residual token count.")
    if anchor_mask_mode not in {"released", "paper"}:
        raise ValueError("anchor_mask_mode must be 'released' or 'paper'.")
    if position_strategy not in {"first", "median", "mean", "last"}:
        raise ValueError("Unsupported RESTORE position_strategy.")
    resolved_source_count = (
        int(sources.max().item()) + 1 if source_count is None else source_count
    )
    _positive_integer(resolved_source_count, "source_count")
    if int(sources.max().item()) >= resolved_source_count:
        raise ValueError("RESTORE source index lies outside source_count.")

    normalized = torch.nn.functional.normalize(
        selection_features.detach().float(), dim=-1, eps=1e-8
    )
    similarity = normalized @ normalized.transpose(0, 1)
    representativeness = similarity.sum(dim=-1)
    rows = representativeness[:, None]
    columns = representativeness[None, :]
    superior = columns > rows if anchor_mask_mode == "paper" else columns < rows
    masked = similarity.masked_fill(~superior, -1.0)
    maximum = masked.max(dim=-1).values
    maximum = torch.where(superior.any(dim=-1), maximum, torch.zeros_like(maximum))
    score = representativeness * (1.0 - maximum)
    request = {"selection_features": selection_features, "values": values,
               "source_indices": sources, "similarity": similarity,
               "representativeness": representativeness, "keep_count": anchor_count}
    score = score_rule(rules, "anchor_scorer", request, score)
    anchors = torch.topk(score, k=anchor_count).indices.sort().values
    custom = selection_rule(rules, "anchor_selector", {**request, "scores": score},
        count=int(sources.numel()), keep_count=anchor_count, device=selection_features.device)
    if custom is not None:
        anchors = custom.sort().values
    assignment = similarity.index_select(1, anchors).argmax(dim=-1)
    assignment[anchors] = torch.arange(anchor_count, device=anchors.device)
    if rules and "grouper" in rules:
        with torch.no_grad():
            groups = rules["grouper"]({**request, "scores": score, "anchors": anchors})
        if not isinstance(groups, (list, tuple)) or len(groups) != anchor_count:
            raise ValueError("RESTORE grouper must return anchor_count nonempty groups.")
        normalized_groups = []
        for group in groups:
            indices = torch.as_tensor(group, device=assignment.device)
            if indices.ndim != 1 or not indices.numel() or indices.dtype not in (torch.int32, torch.int64):
                raise ValueError("RESTORE groups must contain integer local indices.")
            normalized_groups.append(indices.to(torch.long))
        flat = torch.cat(normalized_groups)
        if flat.numel() != sources.numel() or not torch.equal(flat.sort().values, torch.arange(sources.numel(), device=flat.device)):
            raise ValueError("RESTORE groups must partition all residual indices exactly once.")
        for index, group in enumerate(normalized_groups):
            if not bool((group == anchors[index]).any()):
                raise ValueError("RESTORE groups must retain their corresponding anchor.")
            assignment[group] = index

    merged: list[Any] = []
    positions: list[Any] = []
    sizes: list[int] = []
    membership = torch.zeros(
        (anchor_count, resolved_source_count),
        dtype=torch.bool,
        device=selection_features.device,
    )
    for anchor_index in range(anchor_count):
        members = (assignment == anchor_index).nonzero(as_tuple=False).flatten()
        if int(members.numel()) < 1:
            raise RuntimeError("RESTORE produced an empty anchor cluster.")
        original = sources.index_select(0, members)
        merged.append(_aggregate_restore_group(values.index_select(0, members), original, rules=rules))
        positions.append(_representative_position(original, position_strategy))
        sizes.append(int(members.numel()))
        membership[anchor_index, original] = True
    position_tensor = torch.stack(positions).to(torch.long)
    order = torch.argsort(position_tensor, stable=True)
    return RestoreAnchorMergeResult(
        tokens=torch.stack(merged).index_select(0, order),
        representative_indices=position_tensor.index_select(0, order),
        token_sizes=torch.tensor(
            sizes,
            dtype=torch.float32,
            device=selection_features.device,
        ).index_select(0, order),
        membership=membership.index_select(0, order),
    )


def restore_image_tokens(
    output_tokens: Any,
    selection_features: Any,
    attention: Any,
    *,
    spec: RestoreSpec | None = None,
    target_count: int | None = None,
    media_unit: str | LayoutTokenUnit = "image_feature",
    source_layout: Mapping[str, Any] | None = None,
    result_layout: Mapping[str, Any] | None = None,
    rules: Mapping[str, Any] | None = None,
) -> RestoreOutput:
    """Apply released VisPruner pristine selection plus anchor merging."""

    torch = _torch()
    _floating_matrix(output_tokens, "output_tokens", torch=torch)
    _floating_matrix(selection_features, "selection_features", torch=torch)
    if int(output_tokens.shape[0]) != int(selection_features.shape[0]):
        raise ValueError("RESTORE output and selection token counts must match.")
    selection_features = selection_features.to(device=output_tokens.device)
    if isinstance(attention, torch.Tensor):
        attention = attention.to(device=output_tokens.device)
    active = spec or RestoreSpec()
    source_count = int(output_tokens.shape[0])
    target = active.resolve_target_count(source_count, target_count=target_count)
    if target == source_count:
        representatives = torch.arange(source_count, device=output_tokens.device)
        sizes = torch.ones(
            source_count, dtype=torch.float32, device=output_tokens.device
        )
        membership = torch.eye(
            source_count, dtype=torch.bool, device=output_tokens.device
        )
        delta = LayoutDelta.identity(
            source_count,
            media_unit=media_unit,
            layout=source_layout or result_layout or {},
        )
        metadata = MappingProxyType(
            {
                "algorithm": "restore",
                "variant": "restore.image",
                "source_count": source_count,
                "result_count": source_count,
                "pristine_count": source_count,
                "anchor_count": 0,
                "representative_indices": tuple(range(source_count)),
                "token_sizes": tuple(1.0 for _ in range(source_count)),
                "mass_conserved": True,
                "compression_applied": False,
            }
        )
        return RestoreOutput(
            output_tokens,
            representatives,
            sizes,
            membership,
            representatives,
            delta,
            metadata,
        )

    pristine_count = max(
        1,
        min(target - 1, int(target * float(active.pruning_ratio))),
    )
    anchor_count = target - pristine_count
    pristine_rules = {name: rules["pristine_" + name] for name in ("scorer", "selector")
                      if rules and "pristine_" + name in rules}
    pristine_order = select_vispruner_indices(
        selection_features,
        attention,
        pristine_count,
        important_ratio=float(active.important_ratio),
        removal_step=active.removal_step,
        **({"rules": pristine_rules} if pristine_rules else {}),
    )
    pristine = pristine_order.sort().values
    residual_mask = torch.ones(
        source_count, dtype=torch.bool, device=output_tokens.device
    )
    residual_mask[pristine] = False
    residual = residual_mask.nonzero(as_tuple=False).flatten()
    anchor = distinctive_anchor_merge(
        selection_features.index_select(0, residual),
        residual,
        anchor_count=anchor_count,
        anchor_mask_mode=active.anchor_mask_mode,
        position_strategy=active.position_strategy,
        output_tokens=output_tokens.index_select(0, residual),
        source_count=source_count,
        rules=rules,
    )

    pristine_membership = torch.zeros(
        (pristine_count, source_count),
        dtype=torch.bool,
        device=output_tokens.device,
    )
    pristine_membership[
        torch.arange(pristine_count, device=output_tokens.device), pristine
    ] = True
    tokens = torch.cat((output_tokens.index_select(0, pristine), anchor.tokens), dim=0)
    representatives = torch.cat((pristine, anchor.representative_indices), dim=0)
    sizes = torch.cat(
        (
            torch.ones(
                pristine_count, dtype=torch.float32, device=output_tokens.device
            ),
            anchor.token_sizes,
        ),
        dim=0,
    )
    membership = torch.cat((pristine_membership, anchor.membership), dim=0)
    order = torch.argsort(representatives, stable=True)
    tokens = tokens.index_select(0, order)
    representatives = representatives.index_select(0, order)
    sizes = sizes.index_select(0, order)
    membership = membership.index_select(0, order)
    if not bool((membership.sum(dim=0) == 1).all().item()):
        raise RuntimeError("RESTORE did not partition every source token exactly once.")
    mapping: dict[int, int] = {}
    for result_index in range(target):
        for source_index in membership[result_index].nonzero(
            as_tuple=False
        ).flatten().tolist():
            mapping[int(source_index)] = result_index
    layout = {
        **dict(result_layout or {}),
        "representative_indices": tuple(
            int(value) for value in representatives.detach().cpu().tolist()
        ),
        "position_strategy": active.position_strategy,
        "token_sizes": tuple(float(value) for value in sizes.detach().cpu().tolist()),
    }
    delta = LayoutDelta.from_mapping(
        source_count=source_count,
        old_to_new=mapping,
        result_count=target,
        media_unit=media_unit,
        source_layout=source_layout,
        result_layout=layout,
        ordering_policy=OrderingPolicy.EXPLICIT,
    )
    representative_values = tuple(
        int(value) for value in representatives.detach().cpu().tolist()
    )
    size_values = tuple(float(value) for value in sizes.detach().cpu().tolist())
    pristine_values = tuple(int(value) for value in pristine.detach().cpu().tolist())
    groups = tuple(
        tuple(
            int(value)
            for value in membership[index].nonzero(
                as_tuple=False
            ).flatten().detach().cpu().tolist()
        )
        for index in range(target)
    )
    metadata = MappingProxyType(
        {
            "algorithm": "restore",
            "variant": "restore.image",
            "base_reducer": "vispruner",
            **rule_metadata(rules),
            "anchor_mask_mode": active.anchor_mask_mode,
            "position_strategy": active.position_strategy,
            "source_count": source_count,
            "result_count": target,
            "pristine_count": pristine_count,
            "anchor_count": anchor_count,
            "pristine_indices": pristine_values,
            "representative_indices": representative_values,
            "token_sizes": size_values,
            "source_groups": groups,
            "mass_conserved": math.isclose(sum(size_values), float(source_count)),
            "compression_applied": True,
        }
    )
    return RestoreOutput(
        tokens=tokens,
        representative_indices=representatives,
        token_sizes=sizes,
        membership=membership,
        pristine_indices=pristine,
        layout_delta=delta,
        metadata=metadata,
    )


def restore_attention_bias(
    query_positions: Any,
    key_positions: Any,
    key_sizes: Any,
    inv_freq: Any,
    *,
    calibration_constant: float = 2.0,
    mrope_section: Sequence[int] | None = None,
    dtype: Any | None = None,
    rules: Mapping[str, Any] | None = None,
) -> Any:
    """Return ``log(s_n) + log(c - D(|m-n|))`` for 1D or M-RoPE."""

    torch = _torch()
    if isinstance(calibration_constant, bool) or not isinstance(
        calibration_constant, (int, float)
    ):
        raise TypeError("calibration_constant must be a real number.")
    if not math.isfinite(float(calibration_constant)) or float(
        calibration_constant
    ) <= 1.0:
        raise ValueError("calibration_constant must be finite and greater than one.")
    if not isinstance(query_positions, torch.Tensor) or not isinstance(
        key_positions, torch.Tensor
    ):
        raise TypeError("RESTORE positions must be tensors.")
    queries = query_positions
    keys = key_positions
    if queries.ndim == 2:
        queries = queries.unsqueeze(0)
    if keys.ndim == 2:
        keys = keys.unsqueeze(0)
    if queries.ndim != 3 or keys.ndim != 3:
        raise ValueError("RESTORE positions must have [batch,tokens] or [axes,batch,tokens].")
    if int(queries.shape[0]) != int(keys.shape[0]) or int(
        queries.shape[1]
    ) != int(keys.shape[1]):
        raise ValueError("RESTORE query/key position axes or batch sizes differ.")
    if not isinstance(key_sizes, torch.Tensor) or key_sizes.ndim != 2:
        raise ValueError("RESTORE key sizes must have shape [batch,key_tokens].")
    if tuple(key_sizes.shape) != (int(keys.shape[1]), int(keys.shape[2])):
        raise ValueError("RESTORE key sizes do not match key positions.")
    if not isinstance(inv_freq, torch.Tensor) or inv_freq.ndim != 1 or int(
        inv_freq.numel()
    ) < 1:
        raise ValueError("RESTORE inv_freq must be a non-empty rank-one tensor.")
    axis_count = int(queries.shape[0])
    if axis_count == 1:
        sections = (int(inv_freq.numel()),)
        if mrope_section is not None:
            raise ValueError("mrope_section is only valid for multi-axis positions.")
    else:
        if mrope_section is None or len(mrope_section) != axis_count:
            raise ValueError("RESTORE M-RoPE requires one section per position axis.")
        sections = tuple(int(value) for value in mrope_section)
        if any(value < 0 for value in sections) or sum(sections) != int(
            inv_freq.numel()
        ):
            raise ValueError("RESTORE mrope_section must partition inv_freq exactly.")
    distance = (
        queries.to(torch.float32).unsqueeze(-1)
        - keys.to(torch.float32).unsqueeze(-2)
    ).abs()
    frequency = inv_freq.to(device=distance.device, dtype=torch.float32)
    decay_sum = torch.zeros_like(distance[0])
    offset = 0
    for axis_index, section in enumerate(sections):
        if section:
            axis_frequency = frequency[offset : offset + section]
            decay_sum = decay_sum + (
                distance[axis_index].unsqueeze(-1)
                * axis_frequency.view(1, 1, 1, -1)
            ).cos().sum(dim=-1)
        offset += section
    decay = decay_sum / float(inv_freq.numel())
    rectifier = (float(calibration_constant) - decay).clamp_min(1e-8)
    bias = key_sizes.to(device=distance.device, dtype=torch.float32).clamp_min(
        1e-8
    ).log().unsqueeze(-2)
    bias = bias + rectifier.log()
    target_dtype = dtype or key_sizes.dtype
    bias = bias.unsqueeze(1).to(dtype=target_dtype)
    return score_rule(rules, "calibration_bias",
        {"query_positions": query_positions, "key_positions": key_positions,
         "key_sizes": key_sizes, "inv_freq": inv_freq,
         "calibration_constant": calibration_constant, "mrope_section": mrope_section,
         "default_bias": bias}, bias)


def _aggregate_restore_group(values: Any, source_indices: Any, *, rules: Mapping[str, Any] | None) -> Any:
    if not rules or "aggregator" not in rules:
        return values.mean(dim=0)
    torch = _torch()
    result = rules["aggregator"]({"values": values, "source_indices": source_indices})
    if (not isinstance(result, torch.Tensor) or result.shape != values.shape[1:]
            or result.dtype != values.dtype or result.device != values.device
            or not bool(torch.isfinite(result).all())):
        raise ValueError("RESTORE aggregator must return finite [width] values on the input dtype/device.")
    if torch.is_grad_enabled() and values.requires_grad and not result.requires_grad:
        raise ValueError("RESTORE aggregator detached the value gradient.")
    return result


def replay_restore_values(features: Any, result: RestoreOutput, *, rules: Mapping[str, Any] | None = None) -> Any:
    """Replay frozen memberships on actual values without revisiting decisions."""
    torch = _torch()
    if features.ndim != 2 or features.shape[0] != result.membership.shape[1]:
        raise ValueError("RESTORE replay features must match the saved source layout.")
    pristine = set(result.pristine_indices.detach().cpu().tolist())
    values = []
    for row in result.membership:
        indices = row.nonzero(as_tuple=False).flatten().to(features.device)
        members = features.index_select(0, indices)
        if indices.numel() == 1 and int(indices[0]) in pristine:
            values.append(members[0])
        else:
            values.append(_aggregate_restore_group(members, indices, rules=rules))
    return torch.stack(values)


def _floating_matrix(value: Any, name: str, *, torch: Any) -> None:
    if (
        not isinstance(value, torch.Tensor)
        or value.ndim != 2
        or not value.is_floating_point()
        or min(int(item) for item in value.shape) < 1
    ):
        raise ValueError(f"{name} must be a non-empty floating rank-two tensor.")
    if not bool(torch.isfinite(value).all().item()):
        raise ValueError(f"{name} must contain only finite values.")


def _positive_integer(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer.")
    if value < 1:
        raise ValueError(f"{name} must be positive.")
    return value


def _torch() -> Any:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - optional runtime
        raise RuntimeError("RESTORE execution requires PyTorch.") from exc
    return torch


__all__ = [
    "RESTORE_CONFIG_KEYS",
    "RestoreAnchorMergeResult",
    "RestoreOutput",
    "RestoreSpec",
    "build_restore_prefill_position_ids",
    "build_restore_prefill_sizes",
    "distinctive_anchor_merge",
    "merge_restore_executor_config",
    "restore_attention_bias",
    "restore_image_tokens",
    "replay_restore_values",
]
