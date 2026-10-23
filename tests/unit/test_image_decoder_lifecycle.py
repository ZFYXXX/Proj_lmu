from __future__ import annotations

import pytest

from mllm_comp.backbones.image_training import repair_labels_for_layout_delta
from mllm_comp.compressors.image.decoder_lifecycle import (
    CacheRepairPolicy,
    DecoderBoundary,
    DecoderCallPhase,
    ImageDecoderLifecycle,
    LayerLocalCacheView,
    classify_decoder_call,
    compact_dynamic_cache_prefix,
    dynamic_cache_layer_length,
    transform_dynamic_cache_prefix,
    validate_layer_local_dynamic_cache,
)
from mllm_comp.core.errors import BackendIntegrationError, InvariantError
from mllm_comp.core.layout import LayoutDelta, OrderingPolicy


class _EmptyLayer:
    def get_seq_length(self) -> int:
        return 0


class _LengthLayer:
    def __init__(self, length: int) -> None:
        self.length = length

    def get_seq_length(self) -> int:
        return self.length


class DynamicCache:
    def __init__(self, layers: list[object]) -> None:
        self.layers = layers
        self.is_compileable = False
        self.offloading = False

    def get_mask_sizes(self, cache_position: object, layer_idx: int) -> tuple[int, int]:
        del cache_position
        return self.layers[layer_idx].get_seq_length() + 1, layer_idx


class StaticCache:
    def __init__(self) -> None:
        self.layers = []


def test_call_classifier_requires_one_prefill_then_single_token_decode() -> None:
    forward = classify_decoder_call(
        query_length=7, use_cache=False, past_key_values=None
    )
    prefill = classify_decoder_call(
        query_length=7,
        use_cache=True,
        past_key_values=DynamicCache([_EmptyLayer(), _EmptyLayer()]),
    )

    assert forward.phase is DecoderCallPhase.FORWARD
    assert prefill.phase is DecoderCallPhase.PREFILL

    with pytest.raises(BackendIntegrationError, match="speculative or chunked"):
        classify_decoder_call(
            query_length=2,
            use_cache=True,
            past_key_values=DynamicCache([_LengthLayer(7), _LengthLayer(7)]),
        )


def test_static_cache_fails_closed_before_decoder_execution() -> None:
    with pytest.raises(BackendIntegrationError, match="StaticCache"):
        classify_decoder_call(
            query_length=7,
            use_cache=True,
            past_key_values=StaticCache(),
        )


def test_layer_local_cache_view_routes_mask_size_to_exact_layer() -> None:
    cache = DynamicCache([_LengthLayer(12), _LengthLayer(8), _LengthLayer(5)])

    view = LayerLocalCacheView(cache, 1)

    assert view.get_mask_sizes(object(), 0) == (9, 1)
    with pytest.raises(InvariantError, match="exactly one"):
        view.get_mask_sizes(object(), 1)


try:
    import torch
except ImportError:  # pragma: no cover - minimal local environment
    torch = None


class DynamicLayer:
    def __init__(self, length: int, *, offset: float = 0.0) -> None:
        assert torch is not None
        self.keys = torch.arange(
            offset,
            offset + 2 * length,
            dtype=torch.float32,
        ).reshape(1, 1, length, 2)
        self.values = self.keys + 100

    def get_seq_length(self) -> int:
        return int(self.keys.shape[-2])


def _filled_cache(lengths: list[int]) -> DynamicCache:
    return DynamicCache(
        [
            DynamicLayer(length, offset=1000.0 * index)
            for index, length in enumerate(lengths)
        ]
    )


@pytest.mark.skipif(torch is None, reason="PyTorch is not installed")
def test_fastv_prefix_cache_repair_is_exact_and_transactional() -> None:
    cache = _filled_cache([6, 6, 0])
    original_layer_two = cache.layers[2].keys

    telemetry = compact_dynamic_cache_prefix(
        cache,
        torch.tensor([0, 2, 3, 5]),
        source_length=6,
        result_length=4,
        layer_count=2,
        boundary_layer=2,
    )

    assert telemetry.touched_layers == (0, 1)
    assert [item.logical_length for item in telemetry.layers_after] == [4, 4, 0]
    assert cache.layers[0].keys[0, 0, :, 0].tolist() == [0.0, 4.0, 6.0, 10.0]
    assert cache.layers[2].keys is original_layer_two

    mismatched = _filled_cache([6, 5])
    old_keys = mismatched.layers[0].keys
    with pytest.raises(BackendIntegrationError, match="do not match source"):
        compact_dynamic_cache_prefix(
            mismatched,
            torch.tensor([0, 2, 3, 5]),
            source_length=6,
            result_length=4,
            layer_count=2,
            boundary_layer=2,
        )
    assert mismatched.layers[0].keys is old_keys


@pytest.mark.skipif(torch is None, reason="PyTorch is not installed")
def test_sparsevlm_after_layer_cache_transform_mean_merges_every_prefix() -> None:
    cache = _filled_cache([6, 6, 0])
    delta = LayoutDelta.from_mapping(
        source_count=6,
        result_count=4,
        old_to_new=(0, 1, 1, None, 2, 3),
        media_unit="llm_token",
    )

    telemetry = transform_dynamic_cache_prefix(
        cache,
        delta,
        layer_count=2,
        boundary_layer=1,
        boundary=DecoderBoundary.AFTER_LAYER,
    )

    assert telemetry.touched_layers == (0, 1)
    assert telemetry.kept_sequence_indices == (0, 1, 4, 5)
    assert telemetry.represented_sequence_indices == (0, 1, 2, 4, 5)
    assert telemetry.merge_groups == ((1, 2),)
    assert [item.logical_length for item in telemetry.layers_after] == [4, 4, 0]
    assert cache.layers[0].keys[0, 0, :, 0].tolist() == [0.0, 3.0, 8.0, 10.0]

    lifecycle_cache = _filled_cache([6, 6, 0])
    lifecycle = ImageDecoderLifecycle(
        request_id="request-sparsevlm",
        algorithm="sparsevlm",
        num_layers=3,
    )
    lifecycle.begin_call(
        query_length=6,
        use_cache=True,
        past_key_values=DynamicCache([_EmptyLayer(), _EmptyLayer(), _EmptyLayer()]),
    )
    lifecycle.record_boundary(
        layer_index=1,
        boundary=DecoderBoundary.AFTER_LAYER,
        layout_delta=delta,
        past_key_values=lifecycle_cache,
        cache_policy=CacheRepairPolicy.COMPACT_PREFIX,
        prefix_layer_count=2,
    )
    boundary = lifecycle.telemetry()["boundaries"][0]
    assert boundary["merge_groups"] == [[1, 2]]
    assert boundary["old_to_new"] == [0, 1, 1, None, 2, 3]


@pytest.mark.skipif(torch is None, reason="PyTorch is not installed")
def test_decoder_lifecycle_repairs_labels_with_result_representatives() -> None:
    labels = torch.tensor([[11, -100, -100, -100, -100, 22]])
    repaired: list[object] = []

    def on_layout_delta(delta, layer_index, boundary):
        assert layer_index == 1
        assert boundary is DecoderBoundary.AFTER_LAYER
        repaired.append(
            repair_labels_for_layout_delta(
                labels,
                delta,
                operation="unit decoder merge",
            )
        )

    lifecycle = ImageDecoderLifecycle(
        request_id="request-label-repair",
        algorithm="sparsevlm",
        num_layers=3,
        on_layout_delta=on_layout_delta,
    )
    lifecycle.begin_call(query_length=6, use_cache=False, past_key_values=None)
    lifecycle.record_boundary(
        layer_index=1,
        boundary=DecoderBoundary.AFTER_LAYER,
        layout_delta=LayoutDelta.from_mapping(
            source_count=6,
            result_count=4,
            old_to_new=(0, 1, 1, None, 2, 3),
            media_unit="llm_token",
        ),
        past_key_values=None,
        cache_policy=CacheRepairPolicy.LAYER_LOCAL,
    )

    assert len(repaired) == 1
    assert repaired[0].tolist() == [[11, -100, -100, 22]]


@pytest.mark.skipif(torch is None, reason="PyTorch is not installed")
def test_pyramiddrop_keeps_layer_local_cache_lengths() -> None:
    cache = _filled_cache([12, 8, 0, 0])

    telemetry = validate_layer_local_dynamic_cache(
        cache,
        source_length=8,
        result_length=6,
        kept_sequence_indices=torch.tensor([0, 1, 2, 4, 6, 7]),
        boundary_layer=1,
    )

    assert telemetry.touched_layers == ()
    assert [item.logical_length for item in telemetry.layers_after] == [12, 8, 0, 0]
    assert dynamic_cache_layer_length(cache, 1) == 8


@pytest.mark.skipif(torch is None, reason="PyTorch is not installed")
@pytest.mark.parametrize("boundary,index,lengths", [
    (DecoderBoundary.BEFORE_LAYER, 0, [0, 0, 0]),
    (DecoderBoundary.BEFORE_LAYER, 1, [12, 0, 0]),
    (DecoderBoundary.POST_ATTENTION_PRE_FFN, 1, [12, 8, 0]),
])
def test_layer_local_boundaries_track_completed_attention(boundary, index, lengths):
    cache = _filled_cache(lengths)
    report = validate_layer_local_dynamic_cache(cache, source_length=8, result_length=6,
        kept_sequence_indices=[0, 1, 2, 4, 6, 7], boundary_layer=index, boundary=boundary)
    assert report.touched_layers == ()
    assert [item.logical_length for item in report.layers_after] == lengths
    invalid = list(lengths)
    invalid[index + int(boundary is not DecoderBoundary.BEFORE_LAYER)] = 8
    with pytest.raises(BackendIntegrationError, match="future initialized"):
        validate_layer_local_dynamic_cache(_filled_cache(invalid), source_length=8, result_length=6,
            kept_sequence_indices=[0, 1, 2, 4, 6, 7], boundary_layer=index, boundary=boundary)


@pytest.mark.skipif(torch is None, reason="PyTorch is not installed")
@pytest.mark.parametrize("use_cache", [False, True])
def test_pact_layer_local_boundary_records_merges_without_rewriting_cache(
    use_cache: bool,
) -> None:
    active_cache = _filled_cache([8, 8, 0]) if use_cache else None
    lifecycle = ImageDecoderLifecycle(
        request_id=f"request-pact-{use_cache}", algorithm="pact", num_layers=3
    )
    lifecycle.begin_call(
        query_length=8,
        use_cache=use_cache,
        past_key_values=(
            _filled_cache([0, 0, 0]) if use_cache else None
        ),
    )
    delta = LayoutDelta.from_mapping(
        source_count=8,
        result_count=6,
        old_to_new=(0, 2, 2, 3, 1, 1, 4, 5),
        media_unit="llm_token",
        ordering_policy=OrderingPolicy.EXPLICIT,
    )

    lifecycle.record_boundary(
        layer_index=1,
        boundary=DecoderBoundary.AFTER_LAYER,
        layout_delta=delta,
        past_key_values=active_cache,
        cache_policy=CacheRepairPolicy.LAYER_LOCAL,
    )

    boundary = lifecycle.telemetry()["boundaries"][0]
    assert boundary["kept_indices"] == [0, 4, 1, 3, 6, 7]
    assert boundary["merge_groups"] == [[4, 5], [1, 2]]
    assert boundary["cache"]["merge_groups"] == [[4, 5], [1, 2]]
    assert boundary["cache"]["represented_sequence_indices"] == list(range(8))
    if use_cache:
        assert [
            item["logical_length"] for item in boundary["cache"]["layers_after"]
        ] == [8, 8, 0]


@pytest.mark.skipif(torch is None, reason="PyTorch is not installed")
def test_cached_lifecycle_records_boundary_and_decode_growth_without_reselection() -> (
    None
):
    cache = _filled_cache([0, 0, 0])
    lifecycle = ImageDecoderLifecycle(
        request_id="request-2", algorithm="fastv", num_layers=3
    )
    lifecycle.begin_call(query_length=6, use_cache=True, past_key_values=cache)
    cache.layers[0] = DynamicLayer(6)
    cache.layers[1] = DynamicLayer(6, offset=1000.0)
    delta = LayoutDelta.drop(
        6,
        (1, 4),
        media_unit="llm_token",
        protected_indices=(0, 2, 3, 5),
    )
    lifecycle.record_boundary(
        layer_index=2,
        boundary=DecoderBoundary.BEFORE_LAYER,
        layout_delta=delta,
        past_key_values=cache,
        cache_policy=CacheRepairPolicy.COMPACT_PREFIX,
        prefix_layer_count=2,
    )
    cache.layers[2] = DynamicLayer(4, offset=2000.0)
    lifecycle.finish_call(past_key_values=cache)

    for layer in cache.layers:
        layer.keys = torch.cat([layer.keys, layer.keys[..., -1:, :]], dim=-2)
        layer.values = torch.cat([layer.values, layer.values[..., -1:, :]], dim=-2)
    lifecycle.begin_call(query_length=1, use_cache=True, past_key_values=cache)
    lifecycle.finish_call(past_key_values=cache)

    report = lifecycle.telemetry()
    assert report["prefill_forward_calls"] == 1
    assert report["decode_forward_calls"] == 1
    assert report["selection_runs"] == 1
    assert report["selection_repeated_during_decode"] is False
    assert report["boundaries"][0]["cache"]["touched_layers"] == [0, 1]


@pytest.mark.skipif(torch is None, reason="PyTorch is not installed")
def test_lifecycle_forbids_selection_during_decode() -> None:
    cache = _filled_cache([0, 0])
    lifecycle = ImageDecoderLifecycle(
        request_id="request-1", algorithm="fastv", num_layers=2
    )
    lifecycle.begin_call(query_length=5, use_cache=True, past_key_values=cache)
    cache.layers[0] = DynamicLayer(5)
    delta = LayoutDelta.drop(
        5,
        (2,),
        media_unit="llm_token",
        protected_indices=(0, 1, 3, 4),
    )
    lifecycle.record_boundary(
        layer_index=1,
        boundary=DecoderBoundary.BEFORE_LAYER,
        layout_delta=delta,
        past_key_values=cache,
        cache_policy=CacheRepairPolicy.COMPACT_PREFIX,
    )
    cache.layers[1] = DynamicLayer(4, offset=1000.0)
    lifecycle.finish_call(past_key_values=cache)

    for layer in cache.layers:
        layer.keys = torch.cat([layer.keys, layer.keys[..., -1:, :]], dim=-2)
        layer.values = torch.cat([layer.values, layer.values[..., -1:, :]], dim=-2)
    lifecycle.begin_call(query_length=1, use_cache=True, past_key_values=cache)
    with pytest.raises(InvariantError, match="prefill-only"):
        lifecycle.record_boundary(
            layer_index=1,
            boundary=DecoderBoundary.BEFORE_LAYER,
            layout_delta=delta,
            past_key_values=cache,
            cache_policy=CacheRepairPolicy.COMPACT_PREFIX,
        )
    lifecycle.abort_call()
