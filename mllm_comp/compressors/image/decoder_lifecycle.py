"""Shared prefill/decode and KV-cache contract for image-token methods.

Decoder in the code names denotes the LLM stack; DECODE is the incremental
generation phase. FORWARD denotes a non-cached direct call, including training.
This module owns the method-neutral parts of LLM-level image compression:
classifying each call, restricting selection to one prompt pass per request
(prefill or non-cached forward, with one or more selection boundaries),
repairing exact Transformers 4.57 ``DynamicCache`` prefixes (including
mean-merged entries), validating layer-local cache layouts, and publishing
cache telemetry.

Backbone providers still own masks, position encodings, serialized image
spans, newline tokens, and side branches.  Compression algorithms still own
only their ranking/selection mathematics.  Keeping those responsibilities out
of this module is what lets one lifecycle serve every decoder-side method
without creating a paper-specific decoder loop.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Mapping

from mllm_comp.core.errors import BackendIntegrationError, InvariantError
from mllm_comp.core.layout import LayoutDelta


class DecoderCallPhase(str, Enum):
    """Distinguish a non-cached direct forward from generation prefill and decode."""

    FORWARD = "forward"
    PREFILL = "prefill"
    DECODE = "decode"


class DecoderBoundary(str, Enum):
    """A physical LLM layer boundary, independent of the call's generation phase."""

    BEFORE_LAYER = "before_layer"
    AFTER_LAYER = "after_layer"
    POST_ATTENTION_PRE_FFN = "post_attention_pre_ffn"


class CacheRepairPolicy(str, Enum):
    """How an already-built cache prefix follows a sequence mutation."""

    COMPACT_PREFIX = "compact_prefix"
    LAYER_LOCAL = "layer_local"


class LayerLocalCacheView:
    """Read-only cache facade that makes one layer authoritative for masks.

    Transformers 4.57 mask builders normally size a single mask from layer 0.
    Layer-local compression intentionally leaves different prefix lengths at
    different layers, so providers rebuild each decode mask through this facade
    while still passing the original DynamicCache to the decoder layer itself.
    """

    def __init__(self, past_key_values: Any, layer_index: int) -> None:
        layers = _require_dynamic_cache(past_key_values)
        _validate_layer_index(layer_index, len(layers))
        self._cache = past_key_values
        self._layer_index = layer_index
        source_sliding = getattr(past_key_values, "is_sliding", None)
        if source_sliding is None:
            self.is_sliding = (bool(getattr(layers[layer_index], "is_sliding", False)),)
        else:
            self.is_sliding = (bool(source_sliding[layer_index]),)

    def get_mask_sizes(self, cache_position: Any, layer_idx: int) -> tuple[int, int]:
        if layer_idx != 0:
            raise InvariantError("LayerLocalCacheView exposes exactly one mask layer.")
        return self._cache.get_mask_sizes(cache_position, self._layer_index)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._cache, name)


@dataclass(frozen=True)
class DecoderCall:
    """Normalized lifecycle facts for one provider decoder invocation."""

    phase: DecoderCallPhase
    query_length: int
    use_cache: bool
    cache_lengths_before: tuple[int, ...]


@dataclass(frozen=True)
class CacheLayerTelemetry:
    """Logical and physical cache lengths for one decoder layer."""

    layer_index: int
    logical_length: int
    physical_length: int
    cache_layer_type: str


@dataclass(frozen=True)
class CacheRepairTelemetry:
    """One cache validation or mutation performed at a compression boundary."""

    policy: CacheRepairPolicy
    boundary_layer: int
    boundary: DecoderBoundary
    source_length: int
    result_length: int
    kept_sequence_indices: tuple[int, ...]
    touched_layers: tuple[int, ...]
    layers_before: tuple[CacheLayerTelemetry, ...]
    layers_after: tuple[CacheLayerTelemetry, ...]
    cache_type: str | None
    represented_sequence_indices: tuple[int, ...] = ()
    merge_groups: tuple[tuple[int, ...], ...] = ()

    def as_dict(self) -> dict[str, Any]:
        return {
            "policy": self.policy.value,
            "boundary_layer": self.boundary_layer,
            "boundary": self.boundary.value,
            "source_length": self.source_length,
            "result_length": self.result_length,
            "kept_sequence_indices": list(self.kept_sequence_indices),
            "represented_sequence_indices": list(self.represented_sequence_indices),
            "merge_groups": [list(group) for group in self.merge_groups],
            "touched_layers": list(self.touched_layers),
            "cache_type": self.cache_type,
            "layers_before": [_cache_layer_dict(item) for item in self.layers_before],
            "layers_after": [_cache_layer_dict(item) for item in self.layers_after],
        }


@dataclass(frozen=True)
class DecoderBoundaryTelemetry:
    """Layout and cache facts for one algorithm decision."""

    layer_index: int
    boundary: DecoderBoundary
    source_count: int
    result_count: int
    kept_indices: tuple[int, ...]
    dropped_indices: tuple[int, ...]
    represented_indices: tuple[int, ...]
    merge_groups: tuple[tuple[int, ...], ...]
    old_to_new: tuple[int | None, ...]
    media_unit: str
    source_layout: Mapping[str, Any]
    result_layout: Mapping[str, Any]
    cache: CacheRepairTelemetry

    def as_dict(self) -> dict[str, Any]:
        return {
            "layer_index": self.layer_index,
            "boundary": self.boundary.value,
            "source_count": self.source_count,
            "result_count": self.result_count,
            "kept_indices": list(self.kept_indices),
            "dropped_indices": list(self.dropped_indices),
            "represented_indices": list(self.represented_indices),
            "merge_groups": [list(group) for group in self.merge_groups],
            "old_to_new": list(self.old_to_new),
            "media_unit": self.media_unit,
            "source_layout": dict(self.source_layout),
            "result_layout": dict(self.result_layout),
            "cache": self.cache.as_dict(),
        }


@dataclass
class ImageDecoderLifecycle:
    """Request-local state shared by decoder-side image compression methods.

    Providers create one instance for a guarded forward or generate request and
    call :meth:`begin_call` / :meth:`finish_call` around every decoder
    invocation.  Algorithms may call :meth:`record_boundary` only during the
    one prefill (or a non-cached forward); decode therefore cannot accidentally
    repeat a discrete selection.
    """

    request_id: str
    algorithm: str
    num_layers: int
    requires_decoder_boundary: bool = True
    on_layout_delta: Callable[[LayoutDelta, int, DecoderBoundary], None] | None = None
    prefill_forward_calls: int = 0
    decode_forward_calls: int = 0
    selection_runs: int = 0
    prefill_complete: bool = False
    source_length: int | None = None
    current_length: int | None = None
    boundaries: list[DecoderBoundaryTelemetry] = field(default_factory=list)
    prefill_cache_layers: tuple[CacheLayerTelemetry, ...] = ()
    latest_cache_layers: tuple[CacheLayerTelemetry, ...] = ()
    layer_local_decode_positions: list[dict[str, Any]] = field(default_factory=list)
    _active_call: DecoderCall | None = field(default=None, init=False, repr=False)
    _decode_selection_checkpoint: int | None = field(
        default=None, init=False, repr=False
    )

    def __post_init__(self) -> None:
        if not self.request_id:
            raise ValueError("request_id must not be empty.")
        if not self.algorithm:
            raise ValueError("algorithm must not be empty.")
        if isinstance(self.num_layers, bool) or not isinstance(self.num_layers, int):
            raise TypeError("num_layers must be an integer.")
        if self.num_layers <= 0:
            raise ValueError("num_layers must be positive.")
        if not isinstance(self.requires_decoder_boundary, bool):
            raise TypeError("requires_decoder_boundary must be a boolean.")

    def begin_call(
        self,
        *,
        query_length: int,
        use_cache: bool,
        past_key_values: Any | None,
    ) -> DecoderCall:
        if self._active_call is not None:
            raise InvariantError("A decoder lifecycle call is already active.")
        call = classify_decoder_call(
            query_length=query_length,
            use_cache=use_cache,
            past_key_values=past_key_values,
        )
        if call.phase in (DecoderCallPhase.FORWARD, DecoderCallPhase.PREFILL):
            if self.prefill_forward_calls:
                raise InvariantError(
                    "Image decoder compression permits exactly one prompt pass per request."
                )
            if self.prefill_complete:
                raise InvariantError(
                    "A completed image prefill cannot be started again."
                )
            self.prefill_forward_calls += 1
            self.source_length = call.query_length
            self.current_length = call.query_length
        else:
            if not self.prefill_complete:
                raise InvariantError(
                    "Decode cannot start before compressed prefill completes."
                )
            if not self.prefill_cache_layers:
                raise InvariantError(
                    "Decode requires a completed cached prefill, not a non-cached forward."
                )
            self.decode_forward_calls += 1
            self._decode_selection_checkpoint = self.selection_runs
        self._active_call = call
        return call

    def record_boundary(
        self,
        *,
        layer_index: int,
        boundary: DecoderBoundary | str,
        layout_delta: LayoutDelta,
        past_key_values: Any | None,
        cache_policy: CacheRepairPolicy | str,
        prefix_layer_count: int | None = None,
    ) -> DecoderBoundaryTelemetry:
        call = self._active_call
        if call is None:
            raise InvariantError("record_boundary() requires an active decoder call.")
        if call.phase is DecoderCallPhase.DECODE:
            raise InvariantError(
                "Image token selection is prefill-only and may not repeat during decode."
            )
        _validate_layer_index(layer_index, self.num_layers)
        boundary = DecoderBoundary(boundary)
        policy = CacheRepairPolicy(cache_policy)
        if not isinstance(layout_delta, LayoutDelta):
            raise TypeError("layout_delta must be a LayoutDelta.")
        if layout_delta.media_unit != "llm_token":
            raise InvariantError(
                "Decoder lifecycle boundaries require llm_token units."
            )
        if self.current_length != layout_delta.source_count:
            raise InvariantError(
                "Compression LayoutDelta source count does not match the tracked "
                f"decoder layout: {layout_delta.source_count} != {self.current_length}."
            )
        representatives = layout_result_representatives(layout_delta)
        if self.on_layout_delta is not None:
            self.on_layout_delta(layout_delta, layer_index, boundary)
        if call.phase is DecoderCallPhase.PREFILL and past_key_values is None:
            raise BackendIntegrationError(
                "Cached image prefill must expose its DynamicCache at every boundary."
            )
        if policy is CacheRepairPolicy.COMPACT_PREFIX:
            if boundary is DecoderBoundary.BEFORE_LAYER:
                expected_prefix = (
                    layer_index if prefix_layer_count is None else prefix_layer_count
                )
            else:
                expected_prefix = layer_index + 1 if past_key_values is not None else 0
                if prefix_layer_count != expected_prefix:
                    raise InvariantError(
                        "After-layer compact-prefix repair must explicitly include "
                        "the completed boundary layer."
                    )
            cache = transform_dynamic_cache_prefix(
                past_key_values,
                layout_delta,
                layer_count=expected_prefix,
                boundary_layer=layer_index,
                boundary=boundary,
            )
        else:
            cache = validate_layer_local_dynamic_cache(
                past_key_values,
                source_length=layout_delta.source_count,
                result_length=layout_delta.result_count,
                kept_sequence_indices=representatives,
                represented_sequence_indices=layout_delta.kept_indices,
                merge_groups=layout_delta.merge_groups,
                boundary_layer=layer_index,
                boundary=boundary,
            )
        dropped = tuple(layout_delta.dropped_indices)
        telemetry = DecoderBoundaryTelemetry(
            layer_index=layer_index,
            boundary=boundary,
            source_count=layout_delta.source_count,
            result_count=layout_delta.result_count,
            kept_indices=representatives,
            dropped_indices=dropped,
            represented_indices=tuple(layout_delta.kept_indices),
            merge_groups=tuple(layout_delta.merge_groups),
            old_to_new=tuple(
                layout_delta.old_to_new[index]
                for index in range(layout_delta.source_count)
            ),
            media_unit=layout_delta.media_unit,
            source_layout=dict(layout_delta.source_layout),
            result_layout=dict(layout_delta.result_layout),
            cache=cache,
        )
        self.boundaries.append(telemetry)
        self.selection_runs += 1
        self.current_length = layout_delta.result_count
        return telemetry

    def finish_call(self, *, past_key_values: Any | None) -> None:
        call = self._active_call
        if call is None:
            raise InvariantError("finish_call() requires an active decoder call.")
        try:
            if call.phase is DecoderCallPhase.FORWARD:
                if past_key_values is not None:
                    raise InvariantError(
                        "Non-cached forward unexpectedly returned a cache."
                    )
                self.prefill_complete = True
                return
            layers = dynamic_cache_layers(past_key_values)
            self.latest_cache_layers = layers
            if call.phase is DecoderCallPhase.PREFILL:
                if self.requires_decoder_boundary and not self.boundaries:
                    raise InvariantError(
                        "Compressed image prefill completed without a selection boundary."
                    )
                self.prefill_cache_layers = layers
                self.prefill_complete = True
                return
            if self._decode_selection_checkpoint != self.selection_runs:
                raise InvariantError("Image token selection repeated during decode.")
            _validate_decode_growth(
                self.latest_cache_layers,
                previous=self.prefill_cache_layers,
                decode_calls=self.decode_forward_calls,
            )
        finally:
            self._active_call = None
            self._decode_selection_checkpoint = None

    def record_layer_local_decode_position(
        self,
        *,
        layer_index: int,
        cache_position: Any,
        text_position_ids: Any,
        rotary_position_ids: Any,
        expected_rotary_position_ids: Any,
        rotary_policy: str,
    ) -> None:
        """Validate one layer's physical cache and rotary continuation."""

        call = self._active_call
        if call is None or call.phase is not DecoderCallPhase.DECODE:
            raise InvariantError(
                "Layer-local decode positions require an active decode call."
            )
        _validate_layer_index(layer_index, self.num_layers)
        cache_values = _flat_integer_values(cache_position, "cache_position")
        text_values = _flat_integer_values(text_position_ids, "text_position_ids")
        rotary_values = _flat_integer_values(
            rotary_position_ids, "rotary_position_ids"
        )
        expected_rotary_values = _flat_integer_values(
            expected_rotary_position_ids,
            "expected_rotary_position_ids",
        )
        if len(cache_values) != call.query_length:
            raise InvariantError(
                "Layer-local cache_position does not match the decode query length."
            )
        if text_values != cache_values:
            raise InvariantError(
                "Layer-local text positions must equal the physical cache position."
            )
        if len(rotary_values) % len(cache_values):
            raise InvariantError(
                "Layer-local rotary planes do not match cache_position."
            )
        plane_count = len(rotary_values) // len(cache_values)
        if plane_count < 1:
            raise InvariantError(
                "Layer-local decode requires at least one rotary position plane."
            )
        if expected_rotary_values != rotary_values:
            raise InvariantError(
                "Layer-local rotary positions do not match their declared policy."
            )
        if rotary_policy not in {"native_mrope", "contiguous_physical"}:
            raise InvariantError(
                "Layer-local rotary policy must be native_mrope or "
                "contiguous_physical."
            )
        self.layer_local_decode_positions.append(
            {
                "decode_call_index": self.decode_forward_calls,
                "layer_index": layer_index,
                "cache_positions": list(cache_values),
                "text_position_values": list(text_values),
                "rotary_position_values": list(rotary_values),
                "expected_rotary_position_values": list(expected_rotary_values),
                "rotary_position_plane_count": plane_count,
                "text_matches_physical_cache": True,
                "rotary_matches_expected": True,
                "rotary_policy": rotary_policy,
            }
        )

    def abort_call(self) -> None:
        """Clear an in-flight marker after a provider exception."""

        self._active_call = None
        self._decode_selection_checkpoint = None

    def telemetry(self) -> dict[str, Any]:
        return {
            "request_id": self.request_id,
            "algorithm": self.algorithm,
            "prefill_forward_calls": self.prefill_forward_calls,
            "decode_forward_calls": self.decode_forward_calls,
            "prefill_complete": self.prefill_complete,
            "requires_decoder_boundary": self.requires_decoder_boundary,
            "selection_runs": self.selection_runs,
            "selection_repeated_during_decode": False,
            "source_length": self.source_length,
            "result_length": self.current_length,
            "boundaries": [item.as_dict() for item in self.boundaries],
            "prefill_cache_layers": [
                _cache_layer_dict(item) for item in self.prefill_cache_layers
            ],
            "latest_cache_layers": [
                _cache_layer_dict(item) for item in self.latest_cache_layers
            ],
            "layer_local_decode_positions": [
                dict(item) for item in self.layer_local_decode_positions
            ],
        }


def classify_decoder_call(
    *,
    query_length: int,
    use_cache: bool,
    past_key_values: Any | None,
) -> DecoderCall:
    """Classify the exact forward/prefill/decode envelopes supported today."""

    if isinstance(query_length, bool) or not isinstance(query_length, int):
        raise TypeError("query_length must be an integer.")
    if query_length <= 0:
        raise ValueError("query_length must be positive.")
    if not isinstance(use_cache, bool):
        raise TypeError("use_cache must be resolved to a bool by the provider.")
    if not use_cache:
        if past_key_values is not None:
            raise BackendIntegrationError(
                "A non-cached image decoder call may not carry past_key_values."
            )
        return DecoderCall(
            phase=DecoderCallPhase.FORWARD,
            query_length=query_length,
            use_cache=False,
            cache_lengths_before=(),
        )
    lengths = tuple(
        item.logical_length for item in dynamic_cache_layers(past_key_values)
    )
    has_cached_tokens = any(length > 0 for length in lengths)
    if not has_cached_tokens and query_length > 1:
        phase = DecoderCallPhase.PREFILL
    elif has_cached_tokens and query_length == 1:
        phase = DecoderCallPhase.DECODE
    else:
        raise BackendIntegrationError(
            "Image generation supports one multi-token prefill followed by "
            "single-token decode calls; speculative or chunked decode is not proven."
        )
    return DecoderCall(
        phase=phase,
        query_length=query_length,
        use_cache=True,
        cache_lengths_before=lengths,
    )


def dynamic_cache_layers(
    past_key_values: Any | None,
) -> tuple[CacheLayerTelemetry, ...]:
    """Inspect an exact Transformers 4.57 DynamicCache without mutating it."""

    if past_key_values is None:
        return ()
    layers = _require_dynamic_cache(past_key_values)
    return tuple(_layer_telemetry(index, layer) for index, layer in enumerate(layers))


def dynamic_cache_layer_length(past_key_values: Any, layer_index: int) -> int:
    """Return one layer's logical prefix length for layer-local decode masks."""

    layers = _require_dynamic_cache(past_key_values)
    _validate_layer_index(layer_index, len(layers))
    return _layer_telemetry(layer_index, layers[layer_index]).logical_length


def compact_dynamic_cache_prefix(
    past_key_values: Any | None,
    kept_sequence_indices: Any,
    *,
    source_length: int,
    result_length: int,
    layer_count: int,
    boundary_layer: int,
    boundary: DecoderBoundary = DecoderBoundary.BEFORE_LAYER,
) -> CacheRepairTelemetry:
    """Transactionally select the same prompt indices in every built prefix layer."""

    kept = _index_tuple(kept_sequence_indices)
    _validate_compaction_indices(kept, source_length, result_length)
    mapping: list[int | None] = [None] * source_length
    for result_index, source_index in enumerate(kept):
        mapping[source_index] = result_index
    delta = LayoutDelta.from_mapping(
        source_count=source_length,
        result_count=result_length,
        old_to_new=mapping,
        media_unit="llm_token",
    )
    return transform_dynamic_cache_prefix(
        past_key_values,
        delta,
        layer_count=layer_count,
        boundary_layer=boundary_layer,
        boundary=boundary,
    )


def transform_dynamic_cache_prefix(
    past_key_values: Any | None,
    layout_delta: LayoutDelta,
    *,
    layer_count: int,
    boundary_layer: int,
    boundary: DecoderBoundary = DecoderBoundary.BEFORE_LAYER,
) -> CacheRepairTelemetry:
    """Apply one exact drop/merge layout to every completed cache layer."""

    if not isinstance(layout_delta, LayoutDelta):
        raise TypeError("layout_delta must be a LayoutDelta.")
    if layout_delta.media_unit != "llm_token":
        raise InvariantError("DynamicCache transforms require llm_token units.")
    representatives = layout_result_representatives(layout_delta)
    source_length = layout_delta.source_count
    result_length = layout_delta.result_count
    if isinstance(layer_count, bool) or not isinstance(layer_count, int):
        raise TypeError("layer_count must be an integer.")
    if layer_count < 0:
        raise ValueError("layer_count must be non-negative.")
    if past_key_values is None:
        if layer_count:
            raise BackendIntegrationError(
                "Prefix cache repair requested layers but no cache was provided."
            )
        return _empty_cache_telemetry(
            policy=CacheRepairPolicy.COMPACT_PREFIX,
            boundary_layer=boundary_layer,
            boundary=boundary,
            source_length=source_length,
            result_length=result_length,
            kept=representatives,
            represented=tuple(layout_delta.kept_indices),
            merge_groups=tuple(layout_delta.merge_groups),
        )
    layers = _require_dynamic_cache(past_key_values)
    if layer_count > len(layers):
        raise BackendIntegrationError(
            f"Cache exposes {len(layers)} layers but repair requested {layer_count}."
        )
    before = tuple(_layer_telemetry(i, layer) for i, layer in enumerate(layers))
    prepared: list[tuple[Any, Any, Any, bool]] = []
    for layer_index in range(layer_count):
        layer = layers[layer_index]
        snapshot = before[layer_index]
        if (
            snapshot.logical_length != source_length
            or snapshot.physical_length != source_length
        ):
            raise BackendIntegrationError(
                "Cannot compact DynamicCache layer "
                f"{layer_index}: logical/physical lengths "
                f"{snapshot.logical_length}/{snapshot.physical_length} do not "
                f"match source length {source_length}."
            )
        keys = getattr(layer, "keys", None)
        values = getattr(layer, "values", None)
        if keys is None or values is None or not hasattr(keys, "index_select"):
            raise BackendIntegrationError(
                f"DynamicCache layer {layer_index} does not expose mutable keys/values."
            )
        selected_keys = _transform_cache_tensor(keys, layout_delta)
        selected_values = _transform_cache_tensor(values, layout_delta)
        prepared.append(
            (
                layer,
                selected_keys,
                selected_values,
                hasattr(layer, "cumulative_length"),
            )
        )
    for layer, selected_keys, selected_values, has_cumulative in prepared:
        layer.keys = selected_keys
        layer.values = selected_values
        if has_cumulative:
            layer.cumulative_length = result_length
    after = tuple(_layer_telemetry(i, layer) for i, layer in enumerate(layers))
    for layer_index in range(layer_count):
        snapshot = after[layer_index]
        if (
            snapshot.logical_length != result_length
            or snapshot.physical_length != result_length
        ):
            raise InvariantError(
                f"DynamicCache layer {layer_index} repair did not reach exact result length."
            )
    return CacheRepairTelemetry(
        policy=CacheRepairPolicy.COMPACT_PREFIX,
        boundary_layer=boundary_layer,
        boundary=boundary,
        source_length=source_length,
        result_length=result_length,
        kept_sequence_indices=representatives,
        touched_layers=tuple(range(layer_count)),
        layers_before=before,
        layers_after=after,
        cache_type=type(past_key_values).__name__,
        represented_sequence_indices=tuple(layout_delta.kept_indices),
        merge_groups=tuple(layout_delta.merge_groups),
    )


def validate_layer_local_dynamic_cache(
    past_key_values: Any | None,
    *,
    source_length: int,
    result_length: int,
    kept_sequence_indices: Any,
    represented_sequence_indices: Any | None = None,
    merge_groups: Any = (),
    boundary_layer: int,
    boundary: DecoderBoundary = DecoderBoundary.AFTER_LAYER,
) -> CacheRepairTelemetry:
    """Validate a layer-local boundary without flattening per-layer caches.

    Completed Attention modules retain their prefix; future Attention modules
    receive the compact hidden layout. This also applies when compression runs
    before a layer or between its Attention residual and FFN. No cache is rewritten;
    the extra arguments only retain the exact many-to-one mapping in runtime
    evidence.
    """

    kept = _index_tuple(kept_sequence_indices)
    _validate_layer_local_representatives(kept, source_length, result_length)
    represented = (
        kept
        if represented_sequence_indices is None
        else _index_tuple(represented_sequence_indices)
    )
    groups = tuple(tuple(int(index) for index in group) for group in merge_groups)
    if past_key_values is None:
        return _empty_cache_telemetry(
            policy=CacheRepairPolicy.LAYER_LOCAL,
            boundary_layer=boundary_layer,
            boundary=boundary,
            source_length=source_length,
            result_length=result_length,
            kept=kept,
            represented=represented,
            merge_groups=groups,
        )
    layers = _require_dynamic_cache(past_key_values)
    boundary = DecoderBoundary(boundary)
    _validate_layer_index(boundary_layer, len(layers) + int(boundary is DecoderBoundary.BEFORE_LAYER))
    completed = boundary_layer + int(boundary is not DecoderBoundary.BEFORE_LAYER)
    snapshots = tuple(_layer_telemetry(i, layer) for i, layer in enumerate(layers))
    if completed:
        current = snapshots[completed - 1]
        # A preceding FFN boundary may already have shortened hidden states
        # while that preceding Attention's layer-local cache remains longer.
        valid = (current.logical_length >= source_length and current.physical_length >= source_length) if boundary is DecoderBoundary.BEFORE_LAYER else (
            current.logical_length == source_length and current.physical_length == source_length)
        if not valid:
            raise BackendIntegrationError(
                "Layer-local cache boundary layer "
                f"{completed - 1} has logical/physical lengths "
                f"{current.logical_length}/{current.physical_length}, expected {source_length}."
            )
    unexpected = [
        item.layer_index
        for item in snapshots[completed:]
        if item.logical_length or item.physical_length
    ]
    if unexpected:
        raise BackendIntegrationError(
            "Layer-local cache contains future initialized layers at a prefill "
            f"boundary: {unexpected}."
        )
    return CacheRepairTelemetry(
        policy=CacheRepairPolicy.LAYER_LOCAL,
        boundary_layer=boundary_layer,
        boundary=boundary,
        source_length=source_length,
        result_length=result_length,
        kept_sequence_indices=kept,
        touched_layers=(),
        layers_before=snapshots,
        layers_after=snapshots,
        cache_type=type(past_key_values).__name__,
        represented_sequence_indices=represented,
        merge_groups=groups,
    )


def _require_dynamic_cache(past_key_values: Any) -> tuple[Any, ...]:
    cache_type = type(past_key_values).__name__
    if cache_type != "DynamicCache":
        raise BackendIntegrationError(
            "Decoder-side image compression supports transformers.DynamicCache "
            f"only; got {cache_type}. StaticCache and compile-time caches fail closed."
        )
    if bool(getattr(past_key_values, "is_compileable", False)):
        raise BackendIntegrationError(
            "Compileable/static cache layouts cannot be mutated by image compression."
        )
    if bool(getattr(past_key_values, "offloading", False)):
        raise BackendIntegrationError(
            "Offloaded DynamicCache mutation is not proven for image compression."
        )
    layers = getattr(past_key_values, "layers", None)
    if not isinstance(layers, (list, tuple)):
        raise BackendIntegrationError(
            "Transformers 4.57 DynamicCache must expose an ordered layers collection."
        )
    return tuple(layers)


def _layer_telemetry(layer_index: int, layer: Any) -> CacheLayerTelemetry:
    layer_type = type(layer).__name__
    logical = 0
    get_seq_length = getattr(layer, "get_seq_length", None)
    if callable(get_seq_length):
        logical = int(get_seq_length())
    keys = getattr(layer, "keys", None)
    physical = 0
    if keys is not None and getattr(keys, "ndim", 0) >= 2:
        physical = int(keys.shape[-2])
    if logical < 0 or physical < 0:
        raise InvariantError("Cache layer lengths must be non-negative.")
    return CacheLayerTelemetry(
        layer_index=layer_index,
        logical_length=logical,
        physical_length=physical,
        cache_layer_type=layer_type,
    )


def _validate_decode_growth(
    current: tuple[CacheLayerTelemetry, ...],
    *,
    previous: tuple[CacheLayerTelemetry, ...],
    decode_calls: int,
) -> None:
    if len(current) != len(previous):
        raise InvariantError("DynamicCache layer count changed during decode.")
    for before, after in zip(previous, current):
        expected = before.logical_length + decode_calls
        if after.logical_length != expected:
            raise InvariantError(
                "Decode cache growth differs from the stable prefill layout at "
                f"layer {after.layer_index}: {after.logical_length} != {expected}."
            )


def layout_merge_groups(delta: LayoutDelta, source_indices: Any = None) -> tuple[tuple[int, ...], ...]:
    """Capture discrete member groups for replay or a synchronized visual branch."""
    sources = range(delta.source_count) if source_indices is None else source_indices.tolist()
    groups: dict[int, list[int]] = {}
    for local, source in enumerate(sources):
        target = delta.old_to_new[int(source)]
        if target is not None:
            groups.setdefault(target, []).append(local)
    return tuple(tuple(groups[target]) for target in sorted(groups))


def replay_mean_groups(values: Any, groups: tuple[tuple[int, ...], ...]) -> Any:
    """Apply the recorded equal-weight means without rerunning clustering."""
    import torch
    return torch.stack([
        values.index_select(0, torch.tensor(group, dtype=torch.long, device=values.device)).mean(0)
        for group in groups
    ])


def layout_result_representatives(delta: LayoutDelta) -> tuple[int, ...]:
    """Return one stable source representative for every result token."""

    if not isinstance(delta, LayoutDelta):
        raise TypeError("delta must be a LayoutDelta.")
    preimages: list[list[int]] = [[] for _ in range(delta.result_count)]
    for source_index in range(delta.source_count):
        result_index = delta.old_to_new[source_index]
        if result_index is not None:
            preimages[result_index].append(source_index)
    if any(not group for group in preimages):
        raise InvariantError("LayoutDelta result token has no source preimage.")
    return tuple(group[0] for group in preimages)


def _transform_cache_tensor(tensor: Any, delta: LayoutDelta) -> Any:
    groups: list[list[int]] = [[] for _ in range(delta.result_count)]
    for source_index in range(delta.source_count):
        result_index = delta.old_to_new[source_index]
        if result_index is not None:
            groups[result_index].append(source_index)
    if all(len(group) == 1 for group in groups):
        indices = _indices_for_tensor(tuple(group[0] for group in groups), tensor)
        return tensor.index_select(-2, indices)
    transformed = []
    for group in groups:
        indices = _indices_for_tensor(tuple(group), tensor)
        selected = tensor.index_select(-2, indices)
        transformed.append(selected.mean(dim=-2))
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - runtime dependency guard
        raise RuntimeError("DynamicCache repair requires PyTorch.") from exc
    return torch.stack(transformed, dim=-2)


def _index_tuple(indices: Any) -> tuple[int, ...]:
    if hasattr(indices, "tolist"):
        raw = indices.tolist()
    else:
        raw = list(indices)
    if not isinstance(raw, list) or any(isinstance(item, list) for item in raw):
        raise ValueError("kept_sequence_indices must be one-dimensional.")
    return tuple(int(item) for item in raw)


def _flat_integer_values(value: Any, name: str) -> tuple[int, ...]:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "reshape") and hasattr(value, "tolist"):
        raw = value.reshape(-1).tolist()
    elif hasattr(value, "tolist"):
        raw = value.tolist()
    else:
        raw = list(value)
    if not isinstance(raw, list) or not raw or any(
        isinstance(item, (list, tuple)) for item in raw
    ):
        raise ValueError(f"{name} must contain one or more integer positions.")
    result: list[int] = []
    for item in raw:
        if isinstance(item, bool):
            raise TypeError(f"{name} must contain integer positions, not booleans.")
        integer = int(item)
        if integer != item:
            raise ValueError(f"{name} must contain only integer positions.")
        result.append(integer)
    return tuple(result)


def _validate_compaction_indices(
    kept: tuple[int, ...], source_length: int, result_length: int
) -> None:
    if len(kept) != result_length:
        raise InvariantError("Kept indices do not match result length.")
    if kept != tuple(sorted(set(kept))):
        raise InvariantError("Kept indices must be unique and in source order.")
    if any(index < 0 or index >= source_length for index in kept):
        raise InvariantError("Kept index lies outside the source sequence.")


def _validate_layer_local_representatives(
    representatives: tuple[int, ...], source_length: int, result_length: int
) -> None:
    """Validate evidence-only representatives without imposing source order.

    Layer-local cache policy does not index-select or rewrite the completed
    cache prefix, so an explicit output order (as emitted by PACT clustering)
    is safe and must be retained verbatim.
    """

    if len(representatives) != result_length:
        raise InvariantError("Representative indices do not match result length.")
    if len(set(representatives)) != len(representatives):
        raise InvariantError("Representative indices must be unique.")
    if any(index < 0 or index >= source_length for index in representatives):
        raise InvariantError("Representative index lies outside the source sequence.")


def _indices_for_tensor(kept: tuple[int, ...], tensor: Any) -> Any:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - runtime dependency guard
        raise RuntimeError("DynamicCache repair requires PyTorch.") from exc
    return torch.tensor(kept, dtype=torch.long, device=tensor.device)


def _validate_layer_index(layer_index: int, num_layers: int) -> None:
    if isinstance(layer_index, bool) or not isinstance(layer_index, int):
        raise TypeError("layer_index must be an integer.")
    if layer_index < 0 or layer_index >= num_layers:
        raise ValueError(f"layer_index {layer_index} is outside {num_layers} layers.")


def _empty_cache_telemetry(
    *,
    policy: CacheRepairPolicy,
    boundary_layer: int,
    boundary: DecoderBoundary,
    source_length: int,
    result_length: int,
    kept: tuple[int, ...],
    represented: tuple[int, ...] | None = None,
    merge_groups: tuple[tuple[int, ...], ...] = (),
) -> CacheRepairTelemetry:
    return CacheRepairTelemetry(
        policy=policy,
        boundary_layer=boundary_layer,
        boundary=boundary,
        source_length=source_length,
        result_length=result_length,
        kept_sequence_indices=kept,
        touched_layers=(),
        layers_before=(),
        layers_after=(),
        cache_type=None,
        represented_sequence_indices=(kept if represented is None else represented),
        merge_groups=merge_groups,
    )


def _cache_layer_dict(item: CacheLayerTelemetry) -> dict[str, Any]:
    return {
        "layer_index": item.layer_index,
        "logical_length": item.logical_length,
        "physical_length": item.physical_length,
        "cache_layer_type": item.cache_layer_type,
    }


__all__ = [
    "CacheLayerTelemetry",
    "CacheRepairPolicy",
    "CacheRepairTelemetry",
    "DecoderBoundary",
    "DecoderBoundaryTelemetry",
    "DecoderCall",
    "DecoderCallPhase",
    "ImageDecoderLifecycle",
    "LayerLocalCacheView",
    "classify_decoder_call",
    "compact_dynamic_cache_prefix",
    "transform_dynamic_cache_prefix",
    "dynamic_cache_layer_length",
    "dynamic_cache_layers",
    "layout_result_representatives",
    "layout_merge_groups",
    "replay_mean_groups",
    "validate_layer_local_dynamic_cache",
]
