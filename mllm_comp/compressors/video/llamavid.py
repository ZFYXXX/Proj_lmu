"""Canonical LLaMA-VID context/content video compression.

The pinned implementation emits two values per temporal unit: one QFormer
text-conditioned context value and one mean-content value.  Both values depend
on the complete raw frame/tubelet raster, so their numerical lineage overlaps
and cannot be encoded by ``LayoutDelta.old_to_new``.  As with FastVID, the
physical layout uses distinct source-ordered singleton anchors while the exact
weighted value lineage is recorded separately for Qwen3 DeepStack replay.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import math
import inspect
from typing import Any

from mllm_comp.core.errors import BackendIntegrationError, ConfigError, InvariantError

from .config import (
    VIDEO_REFERENCE_REVISION,
    resolve_canonical_video_config,
    validate_video_payload_signals,
)
from .contracts import VideoCompressionResult, VideoTokenCoordinate, VideoTokenPayload
from . import _rules as rule_contract


_ALGORITHM = "llamavid"
_VARIANT = "llamavid.video"


class LLaMAVIDVideoKernel:
    """Registry kernel for the frozen 32-query QFormer/mean recipe."""

    name = _ALGORITHM
    version = (
        f"videollm-comp-{VIDEO_REFERENCE_REVISION[:8]}-" "llamavid-context-content-v1"
    )

    def validate_config(self, plugin_config: Mapping[str, Any]) -> None:
        if not isinstance(plugin_config, Mapping):
            raise ConfigError("video plugin config must be a mapping")
        resolve_canonical_video_config(
            _ALGORITHM,
            plugin_config.get("config", plugin_config),
        )

    def __call__(
        self,
        payload: VideoTokenPayload,
        *,
        qformer_module: Any | None = None,
        config: Mapping[str, Any] | None = None,
        pool_position: str = "before",
        rules: Mapping[str, Any] | None = None,
    ) -> VideoCompressionResult:
        return llamavid_video(
            payload,
            qformer_module=qformer_module,
            config=config,
            pool_position=pool_position,
            rules=rules,
        )


LLaMAVIDKernel = LLaMAVIDVideoKernel


def llamavid_video(
    payload: VideoTokenPayload,
    *,
    qformer_module: Any | None = None,
    config: Mapping[str, Any] | None = None,
    pool_position: str = "before",
    rules: Mapping[str, Any] | None = None,
) -> VideoCompressionResult:
    """Run the defining QFormer context and mean-content stages."""

    torch = _torch()
    if pool_position not in {"before", "after"}:
        raise BackendIntegrationError("LLaMA-VID pool_position must be before or after its output projectors")
    if not isinstance(payload, VideoTokenPayload):
        raise TypeError("payload must be a VideoTokenPayload")
    validate_video_payload_signals(_ALGORITHM, payload)
    resolved = resolve_canonical_video_config(_ALGORITHM, config or {})
    if payload.raw_features is None:
        raise BackendIntegrationError("LLaMA-VID requires native raw features")
    _validate_feature_tensor(payload.features, "projected features", torch=torch)
    _validate_feature_tensor(payload.raw_features, "raw features", torch=torch)
    if payload.token_mask is not None:
        mask = payload.token_mask
        if not isinstance(mask, torch.Tensor) or mask.ndim != 1:
            raise BackendIntegrationError(
                "LLaMA-VID token_mask must be a rank-one tensor"
            )
        if not bool(mask.to(dtype=torch.bool).all().item()):
            raise BackendIntegrationError(
                "Canonical LLaMA-VID does not treat padding as visual tokens"
            )

    module = qformer_module or payload.signals.get("bert_qformer")
    compress = getattr(module, "compress_video", None)
    if not callable(compress):
        raise BackendIntegrationError(
            "LLaMA-VID requires a provider-owned QFormer compression module"
        )
    position_kwargs = {}
    if rules:
        try:
            parameter = inspect.signature(compress).parameters.get("rules")
        except (TypeError, ValueError) as exc:
            raise BackendIntegrationError("Cannot inspect injected QFormer rule support") from exc
        if parameter is None or parameter.kind not in (
            inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY,
        ):
            raise BackendIntegrationError("Custom LLaMA-VID rules require compress_video(..., rules=...); the injected module does not expose its internal pool/query rules")
        position_kwargs["rules"] = rules
    if pool_position == "after":
        # Old injected QFormers keep their exact default call signature.
        # Validate the new keyword before invocation without hiding a custom
        # implementation's own TypeError.
        try:
            parameter = inspect.signature(compress).parameters.get("pool_position")
        except (TypeError, ValueError) as exc:
            raise BackendIntegrationError(
                "LLaMA-VID cannot inspect the injected QFormer's explicit pool_position support."
            ) from exc
        if parameter is None or parameter.kind not in (
            inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY,
        ):
            raise BackendIntegrationError(
                "LLaMA-VID post-projector pooling requires compress_video(..., pool_position='after'); the injected QFormer does not expose this position."
            )
        position_kwargs["pool_position"] = pool_position
    _validate_projector_dimensions(payload, resolved)
    text_contexts = _text_contexts(payload)
    text_tokens_by_video = _optional_per_video_signal(
        payload.signals.get("llamavid.bert_tokens_by_video"),
        payload.video_count,
        name="llamavid.bert_tokens_by_video",
    )

    outputs = []
    groups_by_video = []
    representatives_by_video = []
    exact_budgets = []
    value_records_by_video = []
    temporal_counts = []

    for video_index in range(payload.video_count):
        projected_span = payload.video_span(video_index)
        raw_span = payload.raw_video_span(video_index)
        projected = payload.features[projected_span]
        raw = payload.raw_features[raw_span]
        projected_coordinates = payload.coordinates[projected_span]
        raw_coordinates = payload.raw_coordinates[raw_span]
        projected_units = _dense_temporal_units(
            projected_coordinates,
            name="projected coordinates",
        )
        raw_units = _dense_temporal_units(
            raw_coordinates,
            name="raw coordinates",
        )
        if tuple(unit[0] for unit in projected_units) != tuple(
            unit[0] for unit in raw_units
        ):
            raise BackendIntegrationError(
                "LLaMA-VID projected/raw temporal identities do not match"
            )
        raw_sizes = {len(unit[1]) for unit in raw_units}
        if len(raw_sizes) != 1:
            raise BackendIntegrationError(
                "LLaMA-VID requires one fixed raw raster size per temporal unit"
            )
        raw_frames = torch.stack(
            tuple(
                raw.index_select(
                    0,
                    torch.tensor(indices, dtype=torch.long, device=raw.device),
                )
                for _, indices in raw_units
            ),
            dim=0,
        )
        compressed, raw_context_weights = compress(
            raw_frames,
            text_context=text_contexts[video_index],
            text_tokens=text_tokens_by_video[video_index],
            num_query=int(resolved["num_query"]),
            **position_kwargs,
        )
        _validate_module_output(
            compressed,
            raw_context_weights,
            temporal_count=len(raw_units),
            raw_tokens_per_unit=next(iter(raw_sizes)),
            output_hidden=int(projected.shape[-1]),
            torch=torch,
        )

        groups, anchors, records = _layout_and_value_lineage(
            projected_units=projected_units,
            projected_coordinates=projected_coordinates,
            raw_units=raw_units,
            raw_coordinates=raw_coordinates,
            raw_context_weights=raw_context_weights,
            torch=torch,
        )
        flattened = compressed.reshape(-1, int(compressed.shape[-1]))
        if int(flattened.shape[0]) >= int(projected.shape[0]):
            raise BackendIntegrationError(
                "Canonical LLaMA-VID did not produce a non-identity reduction"
            )
        outputs.append(flattened)
        groups_by_video.append(groups)
        representatives_by_video.append(anchors)
        exact_budgets.append(len(groups))
        value_records_by_video.append(records)
        temporal_counts.append(len(projected_units))

    return VideoCompressionResult.from_source_groups(
        payload,
        torch.cat(outputs, dim=0),
        source_groups=groups_by_video,
        representative_indices=representatives_by_video,
        exact_budgets=exact_budgets,
        metadata={
            "algorithm": _VARIANT,
            "reference_revision": VIDEO_REFERENCE_REVISION,
            "canonical_config": dict(resolved),
            "stages": (
                "bert_qformer_cross_attention",
                "context_attention_pool",
                "mean_content_pool",
            ),
            "tokens_per_temporal_unit": 2,
            "temporal_unit_counts_by_video": tuple(temporal_counts),
            "output_order": "context_then_content_per_temporal_unit",
            "layout_source_groups": "first_two_causal_singletons_per_unit",
            "layout_anchor_policy": "source_ordered_distinct_placeholders",
            "value_lineage_policy": "custom" if rules and ('context_aggregator' in rules or 'content_aggregator' in rules) else "exact_overlapping_weighted_full_rasters",
            "value_records_by_video": tuple(value_records_by_video),
            "qformer_queries_are_internal": True,
            "pool_position": pool_position,
            "pool_projectors": ("vlm_att_val_projector", "content_projector"),
            "context_weight_source": "custom" if rules and ('query_transform' in rules or 'context_scorer' in rules) else "raw_qformer_queries_and_raw_keys",
            "output_projectors_before_pool": pool_position == "after",
            "trainable_compressor": True,
        },
    )


def llamavid_structural_groups(
    coordinates: Sequence[VideoTokenCoordinate | Sequence[int]],
) -> tuple[tuple[tuple[int, ...], ...], tuple[int, ...]]:
    """Return two source-ordered physical anchors for every temporal unit."""

    units = _dense_temporal_units(coordinates, name="projected coordinates")
    groups = []
    anchors = []
    for _, indices in units:
        if len(indices) < 2:
            raise BackendIntegrationError(
                "LLaMA-VID needs two distinct placeholders per temporal unit"
            )
        for anchor in indices[:2]:
            groups.append((anchor,))
            anchors.append(anchor)
    return tuple(groups), tuple(anchors)


def replay_llamavid_value_lineage(
    features: Any,
    result: VideoCompressionResult,
    *,
    video_index: int = 0,
    rules: Mapping[str, Any] | None = None,
) -> Any:
    """Replay context/content weighted lineage on one DeepStack branch."""

    torch = _torch()
    _validate_feature_tensor(features, "synchronized features", torch=torch)
    if not isinstance(result, VideoCompressionResult):
        raise TypeError("result must be a VideoCompressionResult")
    source_span = result.source_video_span(video_index)
    if int(features.shape[0]) != source_span.stop - source_span.start:
        raise InvariantError(
            "synchronized LLaMA-VID branch does not match the source layout"
        )
    records_by_video = result.metadata.get("value_records_by_video")
    if not isinstance(records_by_video, tuple) or len(records_by_video) != len(
        result.media_ids
    ):
        raise InvariantError("LLaMA-VID result is missing exact value lineage")

    values = []
    for record in records_by_video[video_index]:
        if not isinstance(record, Mapping):
            raise InvariantError("LLaMA-VID value-lineage record changed")
        indices = tuple(int(value) for value in record.get("source_indices", ()))
        raw_weights = tuple(float(value) for value in record.get("weights", ()))
        if not indices or len(indices) != len(raw_weights):
            raise InvariantError("LLaMA-VID value-lineage weights changed")
        index = torch.tensor(indices, dtype=torch.long, device=features.device)
        weights = torch.tensor(
            raw_weights,
            dtype=features.dtype,
            device=features.device,
        )
        selected = features.index_select(0, index)
        pooled = (selected * weights[:, None]).sum(dim=0)
        name = f"{record['kind']}_aggregator"
        if rules and name in rules:
            pooled = custom_llamavid_pool(selected, weights, rules[name], kind=record["kind"],
                default_value=pooled, branch="deepstack")
        values.append(pooled)
    return torch.stack(values, dim=0)


def custom_llamavid_pool(values: Any, weights: Any, rule: Any, *, kind: str,
                         default_value: Any, branch: str = "main") -> Any:
    return rule_contract.value(rule, {"values": values, "weights": weights, "kind": kind,
        "branch": branch, "default_transform": lambda: default_value}, reference=values,
        shape=(int(values.shape[-1]),), name=f"{kind}_aggregator")


def _layout_and_value_lineage(
    *,
    projected_units: tuple[tuple[int, tuple[int, ...]], ...],
    projected_coordinates: Sequence[VideoTokenCoordinate],
    raw_units: tuple[tuple[int, tuple[int, ...]], ...],
    raw_coordinates: Sequence[VideoTokenCoordinate],
    raw_context_weights: Any,
    torch: Any,
) -> tuple[
    tuple[tuple[int, ...], ...],
    tuple[int, ...],
    tuple[Mapping[str, Any], ...],
]:
    groups, anchors = llamavid_structural_groups(projected_coordinates)
    records = []
    for unit_index, ((temporal, projected_indices), (_, raw_indices)) in enumerate(
        zip(projected_units, raw_units)
    ):
        context = raw_context_weights[unit_index]
        context = context / context.sum()
        content = torch.full_like(context, 1.0 / int(context.numel()))
        for kind, weights in (("context", context), ("content", content)):
            projected_weights = _map_raw_weights_to_projected(
                projected_indices=projected_indices,
                projected_coordinates=projected_coordinates,
                raw_indices=raw_indices,
                raw_coordinates=raw_coordinates,
                raw_weights=weights,
            )
            source_indices = tuple(projected_indices)
            serial_weights = tuple(
                float(projected_weights[index].detach().cpu().item())
                for index in source_indices
            )
            serial_total = math.fsum(serial_weights)
            if (
                not math.isfinite(serial_total)
                or serial_total <= 0.0
                or any(
                    not math.isfinite(value) or value < 0.0 for value in serial_weights
                )
            ):
                raise InvariantError(
                    "LLaMA-VID projected value-lineage weights are invalid"
                )
            # The runtime feature dtype can be FP16. Normalize after converting
            # metadata to Python floats so exact lineage replay does not inherit
            # the half-precision reduction error.
            serial_weights = tuple(value / serial_total for value in serial_weights)
            records.append(
                {
                    "temporal_index": temporal,
                    "kind": kind,
                    "source_indices": source_indices,
                    "weights": serial_weights,
                }
            )
    return groups, anchors, tuple(records)


def _map_raw_weights_to_projected(
    *,
    projected_indices: tuple[int, ...],
    projected_coordinates: Sequence[VideoTokenCoordinate],
    raw_indices: tuple[int, ...],
    raw_coordinates: Sequence[VideoTokenCoordinate],
    raw_weights: Any,
) -> dict[int, Any]:
    projected = tuple(projected_coordinates[index] for index in projected_indices)
    raw = tuple(raw_coordinates[index] for index in raw_indices)
    projected_height = max(value.h for value in projected) + 1
    projected_width = max(value.w for value in projected) + 1
    raw_height = max(value.h for value in raw) + 1
    raw_width = max(value.w for value in raw) + 1
    lookup = {
        (value.h, value.w): index for index, value in zip(projected_indices, projected)
    }
    if len(lookup) != len(projected_indices):
        raise BackendIntegrationError("LLaMA-VID projected raster is not unique")
    result = {index: raw_weights.new_zeros(()) for index in projected_indices}
    for local_index, coordinate in enumerate(raw):
        target_h = min(
            coordinate.h * projected_height // raw_height,
            projected_height - 1,
        )
        target_w = min(
            coordinate.w * projected_width // raw_width,
            projected_width - 1,
        )
        try:
            target = lookup[(target_h, target_w)]
        except KeyError as exc:
            raise BackendIntegrationError(
                "LLaMA-VID raw/projected raster alignment is incomplete"
            ) from exc
        result[target] = result[target] + raw_weights[local_index]
    total = sum(result.values())
    if not bool((total > 0).item()):
        raise InvariantError("LLaMA-VID projected lineage has zero weight")
    return {index: value / total for index, value in result.items()}


def _dense_temporal_units(
    coordinates: Sequence[VideoTokenCoordinate | Sequence[int]],
    *,
    name: str,
) -> tuple[tuple[int, tuple[int, ...]], ...]:
    units: list[tuple[int, list[int]]] = []
    current_t = None
    for index, value in enumerate(coordinates):
        coordinate = _coordinate(value, name=name)
        if coordinate.t != current_t:
            if any(unit_t == coordinate.t for unit_t, _ in units):
                raise BackendIntegrationError(
                    f"LLaMA-VID {name} repeat a non-contiguous temporal unit"
                )
            units.append((coordinate.t, []))
            current_t = coordinate.t
        units[-1][1].append(index)
    if not units:
        raise BackendIntegrationError(f"LLaMA-VID {name} are empty")
    expected_t = tuple(range(len(units)))
    if tuple(value for value, _ in units) != expected_t:
        raise BackendIntegrationError(
            f"LLaMA-VID {name} must use contiguous temporal identities"
        )
    normalised = []
    for temporal, indices in units:
        values = tuple(_coordinate(coordinates[index], name=name) for index in indices)
        height = max(value.h for value in values) + 1
        width = max(value.w for value in values) + 1
        expected = tuple(
            (row, column) for row in range(height) for column in range(width)
        )
        if tuple((value.h, value.w) for value in values) != expected:
            raise BackendIntegrationError(
                f"LLaMA-VID {name} must be dense source-ordered rasters"
            )
        normalised.append((temporal, tuple(indices)))
    return tuple(normalised)


def _coordinate(
    value: VideoTokenCoordinate | Sequence[int], *, name: str
) -> VideoTokenCoordinate:
    if isinstance(value, VideoTokenCoordinate):
        return value
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        values = tuple(value)
        if len(values) == 3:
            return VideoTokenCoordinate(*values)
    raise TypeError(f"{name} entries must be video coordinates")


def _text_contexts(payload: VideoTokenPayload) -> tuple[str, ...]:
    raw = payload.signals.get("text_context")
    if isinstance(raw, str):
        values = (raw,)
    elif isinstance(raw, Sequence) and not isinstance(raw, (str, bytes)):
        values = tuple(raw)
    else:
        raise BackendIntegrationError("LLaMA-VID text_context must be text")
    if len(values) != payload.video_count or any(
        not isinstance(value, str) or not value.strip() for value in values
    ):
        raise BackendIntegrationError(
            "LLaMA-VID requires one non-empty text context per video"
        )
    return tuple(value.strip() for value in values)


def _optional_per_video_signal(
    value: Any,
    count: int,
    *,
    name: str,
) -> tuple[Any | None, ...]:
    if value is None:
        return (None,) * count
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise BackendIntegrationError(f"{name} must contain one value per video")
    result = tuple(value)
    if len(result) != count:
        raise BackendIntegrationError(f"{name} must contain one value per video")
    return result


def _validate_projector_dimensions(
    payload: VideoTokenPayload,
    config: Mapping[str, Any],
) -> None:
    dimensions = payload.signals.get("projector_dimensions")
    if not isinstance(dimensions, Mapping):
        raise BackendIntegrationError(
            "LLaMA-VID projector_dimensions must be a mapping"
        )
    expected = {
        "raw_hidden_size": int(payload.raw_features.shape[-1]),
        "output_hidden_size": int(payload.features.shape[-1]),
        "num_query": int(config["num_query"]),
    }
    for name, value in expected.items():
        if dimensions.get(name) != value:
            raise BackendIntegrationError(
                f"LLaMA-VID projector dimension {name!r} changed"
            )


def _validate_module_output(
    compressed: Any,
    context_weights: Any,
    *,
    temporal_count: int,
    raw_tokens_per_unit: int,
    output_hidden: int,
    torch: Any,
) -> None:
    if not isinstance(compressed, torch.Tensor) or tuple(compressed.shape) != (
        temporal_count,
        2,
        output_hidden,
    ):
        raise BackendIntegrationError(
            "LLaMA-VID QFormer must emit [temporal, 2, output_hidden]"
        )
    if not isinstance(context_weights, torch.Tensor) or tuple(
        context_weights.shape
    ) != (temporal_count, raw_tokens_per_unit):
        raise BackendIntegrationError(
            "LLaMA-VID QFormer must emit one raw context-weight row per unit"
        )
    if not bool(torch.isfinite(compressed).all().item()) or not bool(
        torch.isfinite(context_weights).all().item()
    ):
        raise BackendIntegrationError("LLaMA-VID QFormer emitted non-finite values")
    if bool((context_weights < 0).any().item()) or bool(
        (context_weights.sum(dim=-1) <= 0).any().item()
    ):
        raise BackendIntegrationError(
            "LLaMA-VID context weights must be non-negative with positive rows"
        )


def _validate_feature_tensor(value: Any, name: str, *, torch: Any) -> None:
    if not isinstance(value, torch.Tensor) or value.ndim != 2:
        raise BackendIntegrationError(f"LLaMA-VID {name} must be rank two")
    if (
        not value.is_floating_point()
        or int(value.shape[0]) < 1
        or int(value.shape[1]) < 1
    ):
        raise BackendIntegrationError(f"LLaMA-VID {name} must be non-empty floats")
    if not bool(torch.isfinite(value).all().item()):
        raise BackendIntegrationError(f"LLaMA-VID {name} must be finite")


def _torch() -> Any:
    try:
        import torch
    except ImportError as exc:
        raise BackendIntegrationError(
            "LLaMA-VID video compression requires the existing PyTorch runtime"
        ) from exc
    return torch


__all__ = [
    "LLaMAVIDKernel",
    "LLaMAVIDVideoKernel",
    "llamavid_structural_groups",
    "llamavid_video",
    "replay_llamavid_value_lineage",
]
