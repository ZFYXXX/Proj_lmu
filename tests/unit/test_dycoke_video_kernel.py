from __future__ import annotations

from types import SimpleNamespace

import pytest

from mllm_comp.backbones import dycoke_video as dycoke_provider
from mllm_comp.backbones.dycoke_video import (
    DyCoKeDecodeCacheController,
    DyCoKeVideoExecutor,
)
from mllm_comp.backbones.llava_onevision.video_compression import (
    LlavaOneVisionPackedVideoRequest,
    LlavaOneVisionVideoPacking,
)
from mllm_comp.backbones.qwen3_vl.video_compression import (
    Qwen3VLPackedVideoRequest,
    Qwen3VLVideoCompressionResponse,
    Qwen3VLVideoGrid,
)
from mllm_comp.compressors.video import (
    DyCoKeVideoKernel,
    VideoTokenPayload,
    dycoke_video,
    plan_dycoke_kv_view,
    replay_dycoke_source_groups,
)
from mllm_comp.core.errors import BackendIntegrationError, ConfigError
from mllm_comp.core.vocabulary import ExecutionPhase


def _torch():
    return pytest.importorskip("torch", reason="DyCoKe tensor kernel needs PyTorch")


def _features(*, frame_count: int = 4, requires_grad: bool = False):
    torch = _torch()
    frames = [
        torch.eye(4),
        torch.diag(torch.tensor([1.0, -1.0, 1.0, 1.0])),
        torch.diag(torch.tensor([1.0, 1.0, -1.0, 1.0])),
        torch.diag(torch.tensor([1.0, 1.0, -1.0, -1.0])),
    ]
    values = torch.cat(frames[:frame_count], dim=0).float()
    return values.clone().requires_grad_(requires_grad)


def _coordinates(frame_count: int = 4) -> tuple[tuple[int, int, int], ...]:
    return tuple(
        (frame, row, column)
        for frame in range(frame_count)
        for row in range(2)
        for column in range(2)
    )


def _payload(
    *,
    frame_count: int = 4,
    requires_grad: bool = False,
) -> VideoTokenPayload:
    return VideoTokenPayload(
        features=_features(
            frame_count=frame_count,
            requires_grad=requires_grad,
        ),
        media_ids=("video-dycoke",),
        video_offsets=(0, frame_count * 4),
        coordinates=_coordinates(frame_count),
    )


def test_dycoke_control_plane_freezes_both_defining_stages() -> None:
    kernel = DyCoKeVideoKernel()
    kernel.validate_config({"config": {}})
    kernel.validate_config(
        {
            "modality_contract": {
                "algorithm": "dycoke",
                "variant": "dycoke.video",
            },
            "config": {
                "merging_ratio": 0.7,
                "decoder_prune_layer": 3,
                "kv_drop_ratio": 0.8,
                "attention_similarity_threshold": 0.9,
            },
        }
    )
    with pytest.raises(ConfigError, match="non-canonical"):
        kernel.validate_config({"config": {"kv_drop_ratio": 0.7}})


def test_dycoke_ttm_is_exact_stable_source_ordered_and_differentiable() -> None:
    torch = _torch()
    payload = _payload(requires_grad=True)

    first = dycoke_video(payload)
    second = DyCoKeVideoKernel()(payload)

    expected = (0, 1, 2, 3, 5, 10, 15)
    assert first.source_count == 16
    assert first.result_count == 7
    assert first.representative_indices == (expected,)
    assert first.source_groups == (tuple((index,) for index in expected),)
    assert first.representative_indices == second.representative_indices
    torch.testing.assert_close(
        first.features,
        payload.features.index_select(0, torch.tensor(expected)),
    )
    assert first.metadata["selection_ties"] == "source_index_stable"
    assert first.metadata["selection_output_order"] == "source_order_restored"
    assert tuple(
        record["selection_stage"]
        for record in first.metadata["frame_records_by_video"][0]
    ) == (
        "group_anchor_full",
        "adjacent_unique_selection",
        "stride_zero_two_unique_selection",
        "adjacent_unique_selection",
    )

    first.features.sum().backward()
    assert payload.features.grad is not None
    assert bool((payload.features.grad != 0).any().item())


def test_new_video_method_reuses_selected_executor_and_checks_its_protocol():
    from dataclasses import replace
    from examples.compression_extensions import stride_recent_video as research
    from mllm_comp.config.schema import experiment_config_from_dict
    from mllm_comp.pipeline.definition import resolve_compression_definition

    config = experiment_config_from_dict({
        "backbone": {"name": "qwen3_vl", "model_path": "tiny"},
        "pipeline": {"extensions": ["examples.compression_extensions.stride_recent_video:register"],
            "stages": [{"name": "recent", "algorithm": research.ALGORITHM,
                        "placement": "projector_level", "execution_phase": "generate"}]},
    })
    definition = resolve_compression_definition(config, compression_mode="compressed", interface="eval")
    projection = definition.projections["recent"]
    binding = projection.bindings[0]
    assert binding.executor_factory is definition.registry.binding("dycoke.video.qwen3_vl").executor_factory
    args = dict(backbone="qwen3_vl", model=object(), stage=config.pipeline.stages[0],
                contract_projection=projection)
    executor = DyCoKeVideoExecutor(binding=binding, **args)
    assert isinstance(executor.kernel, research.StrideRecentVideoMethod)
    metadata = executor.runtime_metadata()
    assert metadata["algorithm"] == research.ALGORITHM
    assert metadata["canonical_evidence_applicable"] is False
    assert metadata["real_checkpoint_verified"] is False
    with pytest.raises(BackendIntegrationError, match="new_decode_policy"):
        DyCoKeVideoExecutor(binding=replace(binding, strategy_factory=lambda: lambda *a: None), **args)


def test_stride_recent_changes_both_token_selection_and_decode_policy():
    torch = _torch()
    from examples.compression_extensions.stride_recent_video import StrideRecentVideoMethod, resolve_config
    from mllm_comp.compressors.video.dycoke import DyCoKeDecodePolicy

    method = StrideRecentVideoMethod()
    config = resolve_config({"kv_keep_tokens": 1})
    payload = _payload(requires_grad=True)
    result = method(payload, config=config, rules={})
    assert result.representative_indices == ((1, 5, 9, 13),)
    assert result.representative_indices != dycoke_video(payload).representative_indices
    branch = payload.features * 2
    replayed = method.replay(branch, result, config=config, rules={})
    torch.testing.assert_close(replayed, branch[[1, 5, 9, 13]])
    replayed.sum().backward()
    assert payload.features.grad.count_nonzero() == 4 * payload.features.shape[1]

    query, keys = torch.zeros(1, 2, 1, 2), torch.ones(1, 1, 9, 2)
    kwargs = dict(visual_positions=(2, 3, 4, 5, 6), sequence_length=9, attention_scaling=1.0)
    recent = method.new_decode_policy(config=config, rules={}).plan_view(query, keys, **kwargs)
    original = DyCoKeDecodePolicy().plan_view(query, keys, **kwargs)
    assert recent.kept_visual_positions == (6,)
    assert original.kept_visual_positions == (2,)
    assert recent.view_indices == (0, 1, 6, 7, 8)
    assert recent.attention_similarity is None


def test_dycoke_ttm_keeps_unpaired_tail_and_breaks_ties_by_source_index() -> None:
    torch = _torch()
    trailing = dycoke_video(_payload(frame_count=3))
    assert trailing.result_count == 9
    assert trailing.representative_indices[0] == (0, 1, 2, 3, 5, 8, 9, 10, 11)

    tied = VideoTokenPayload(
        features=torch.eye(4).repeat(4, 1),
        media_ids=("tied",),
        video_offsets=(0, 16),
        coordinates=_coordinates(),
    )
    result = dycoke_video(tied)
    assert result.representative_indices[0] == (0, 1, 2, 3, 4, 8, 12)


def test_dycoke_kv_plan_preserves_nonvisual_and_current_decode_tokens() -> None:
    torch = _torch()
    scores = torch.tensor([0.1, 0.9, 0.2, 0.3, 0.4])

    first = plan_dycoke_kv_view(
        scores,
        visual_positions=(2, 3, 4, 5, 6),
        sequence_length=9,
    )
    second = plan_dycoke_kv_view(
        scores * 0.5,
        visual_positions=(2, 3, 4, 5, 6),
        sequence_length=10,
        previous_attention_scores=scores,
        previous_kept_visual_positions=first.kept_visual_positions,
    )

    assert first.kept_visual_positions == (3,)
    assert first.view_indices == (0, 1, 3, 7, 8)
    assert not first.reused_visual_selection
    assert second.reused_visual_selection
    assert second.kept_visual_positions == (3,)
    assert second.view_indices == (0, 1, 3, 7, 8, 9)
    assert second.view_indices[-1] == 9
    assert second.as_dict()["physical_cache_retained"] is True


def test_dycoke_cache_controller_keeps_full_storage_and_returns_sparse_views() -> None:
    torch = _torch()

    class DynamicLayer:
        def __init__(self) -> None:
            self.keys = None
            self.values = None

        def update(self, keys, values, _cache_kwargs=None):
            if self.keys is None:
                self.keys = keys
                self.values = values
            else:
                self.keys = torch.cat((self.keys, keys), dim=-2)
                self.values = torch.cat((self.values, values), dim=-2)
            return self.keys, self.values

    class DynamicCache:
        def __init__(self) -> None:
            self.layers = [DynamicLayer() for _ in range(4)]
            self.offloading = False

        def update(self, key_states, value_states, layer_idx, cache_kwargs=None):
            return self.layers[layer_idx].update(
                key_states,
                value_states,
                cache_kwargs,
            )

    class _Owner:
        def __init__(self) -> None:
            self.plans = []

        def _record_decode_view(self, plan) -> None:
            self.plans.append(plan)

    cache = DynamicCache()
    owner = _Owner()
    prefill = torch.arange(16, dtype=torch.float32).reshape(1, 1, 8, 2)
    for layer_index in (0, 1):
        cache.update(prefill, prefill + 100, layer_index)
    native_update_function = cache.update.__func__
    controller = DyCoKeDecodeCacheController.attach(
        owner=owner,
        cache=cache,
        visual_positions=(2, 3, 4, 5, 6),
        layer_count=4,
        prune_layer=3,
        drop_ratio=0.8,
        similarity_threshold=0.9,
        attention_scaling=2**-0.5,
    )
    for layer_index in (2, 3):
        keys, _values = cache.update(
            prefill,
            prefill + 100,
            layer_index,
        )
        assert int(keys.shape[-2]) == 8

    token = torch.ones(1, 1, 1, 2)
    for layer_index in (0, 1):
        cache.update(token, token + 100, layer_index)
    controller.prepare_decode_query(torch.ones(1, 2, 1, 2))
    layer2_keys, _ = cache.update(token, token + 100, 2)
    layer3_view, _ = cache.update(token, token + 100, 3)

    assert int(layer2_keys.shape[-2]) == 9
    assert int(layer3_view.shape[-2]) == 5
    assert int(cache.layers[3].keys.shape[-2]) == 9
    assert owner.plans[-1].sequence_length == 9
    assert owner.plans[-1].view_indices[-1] == 8

    for layer_index in (0, 1):
        cache.update(token, token + 100, layer_index)
    controller.prepare_decode_query(torch.ones(1, 2, 1, 2))
    cache.update(token, token + 100, 2)
    second_view, _ = cache.update(token, token + 100, 3)

    assert owner.plans[-1].reused_visual_selection
    assert owner.plans[-1].view_indices[-1] == 9
    assert int(second_view.shape[-2]) == 6
    assert int(cache.layers[3].keys.shape[-2]) == 10

    controller.restore()
    assert not hasattr(cache, "_mllm_comp_dycoke_cache_controller")
    assert cache.update.__func__ is native_update_function


def test_dycoke_cache_controller_rejects_non_dynamic_cache() -> None:
    with pytest.raises(BackendIntegrationError, match="DynamicCache exactly"):
        DyCoKeDecodeCacheController.attach(
            owner=SimpleNamespace(),
            cache=SimpleNamespace(update=lambda *_args, **_kwargs: None),
            visual_positions=(1,),
            layer_count=4,
            prune_layer=3,
            drop_ratio=0.8,
            similarity_threshold=0.9,
            attention_scaling=1.0,
        )


def test_dycoke_llava_visual_positions_exclude_structural_newlines() -> None:
    torch = _torch()
    packing = LlavaOneVisionVideoPacking.from_native(
        frame_count=4,
        grid_height=2,
        grid_width=2,
        patch_merge_type="spatial_unpad",
        newline_position="frame",
    )
    packed = torch.zeros(packing.packed_token_count, 4)
    packed[list(packing.patch_positions)] = _features()
    request = LlavaOneVisionPackedVideoRequest(
        invocation_id="dycoke-llava",
        media_id="video-dycoke",
        payload=packing.payload(packed, media_id="video-dycoke"),
        packing=packing,
        source_visual_span=(2, 2 + packing.packed_token_count),
        prompt_input_ids=torch.tensor([[1, -200, 2]]),
        prepared_attention_mask=torch.ones(1, 3, dtype=torch.long),
        prepared_position_ids=None,
        prepared_labels=None,
    )
    result = dycoke_video(request.payload)

    positions = dycoke_provider._result_visual_positions(
        "llava_onevision",
        request,
        result,
        None,
    )

    assert positions == (2, 3, 4, 5, 7, 9, 11)
    assert set(positions).isdisjoint({6, 8, 10, 12})


def test_dycoke_qwen3_executor_replays_identical_ttm_on_deepstack() -> None:
    torch = _torch()
    features = _features()
    grid = Qwen3VLVideoGrid.from_native(
        (4, 4, 4),
        spatial_merge_size=2,
        second_per_grid_seconds=0.5,
    )
    request = Qwen3VLPackedVideoRequest(
        invocation_id="dycoke-qwen3",
        media_id="video-dycoke",
        payload=grid.payload(features, media_id="video-dycoke"),
        deepstack_features=(features * 2, features * 3),
        grid=grid,
        source_placeholder_positions=tuple(range(1, 17)),
        prompt_input_ids=torch.tensor([[1, *([8] * 16), 2]]),
        source_attention_mask=torch.ones(1, 18, dtype=torch.long),
        source_video_grid_thw=torch.tensor([[4, 4, 4]], dtype=torch.long),
        source_second_per_grid_ts=torch.tensor([0.5]),
    )
    executor = DyCoKeVideoExecutor(
        backbone="qwen3_vl",
        model=object(),
        execution_phase=ExecutionPhase.FORWARD,
    )

    response = executor._projector_callback(request)

    assert isinstance(response, Qwen3VLVideoCompressionResponse)
    assert response.compression_result.result_count == 7
    for source, branch in zip(request.deepstack_features, response.deepstack_features):
        indices = response.compression_result.representative_indices[0]
        expected = source[list(indices)]
        torch.testing.assert_close(branch, expected)
    metadata = executor.runtime_metadata()
    assert metadata["projector_invocations"] == 1
    assert metadata["decode_view_invocations"] == 0
    assert metadata["training_free_compressor"] is True
    assert len(metadata["algorithm_records"]) == 1
    algorithm = metadata["algorithm_records"][0]
    assert algorithm["source_count"] == 16
    assert algorithm["result_count"] == 7
    assert algorithm["exact_budgets"] == (7,)
    assert algorithm["frame_result_counts"] == (4, 1, 1, 1)
    assert algorithm["group_anchor_count"] == 1
    assert algorithm["adjacent_selection_count"] == 2
    assert algorithm["stride_zero_two_selection_count"] == 1
    assert algorithm["singleton_source_groups_exact"] is True
    assert algorithm["representatives_source_ordered"] is True
    assert algorithm["deepstack_branch_count"] == 2
    assert algorithm["deepstack_source_shapes"] == ((16, 4), (16, 4))
    assert algorithm["deepstack_result_shapes"] == ((7, 4), (7, 4))


@pytest.mark.parametrize("backbone", ("llava_onevision", "qwen25_vl", "qwen3_vl"))
@pytest.mark.parametrize("layer_index", (2, 3))
@pytest.mark.parametrize("sequence_length", (1, 5))
@pytest.mark.parametrize("with_native_cache", (False, True))
def test_dycoke_non_video_batched_calls_delegate_unchanged(
    backbone: str,
    layer_index: int,
    sequence_length: int,
    with_native_cache: bool,
) -> None:
    executor = DyCoKeVideoExecutor(backbone=backbone, model=object())
    hidden_states = SimpleNamespace(ndim=3, shape=(2, sequence_length, 8))
    cache = SimpleNamespace() if with_native_cache else None
    mask, positions, result = object(), object(), object()
    calls = []

    def native(hidden_states, *, attention_mask, past_key_values, position_embeddings):
        calls.append(
            (hidden_states, attention_mask, past_key_values, position_embeddings)
        )
        return result

    wrapped = executor._layer_forward_replacement(layer_index, native, layer_count=4)
    assert (
        wrapped(
            object(),
            hidden_states,
            attention_mask=mask,
            past_key_values=cache,
            position_embeddings=positions,
        )
        is result
    )
    assert len(calls) == 1
    assert all(
        actual is expected
        for actual, expected in zip(calls[0], (hidden_states, mask, cache, positions))
    )


@pytest.mark.parametrize("backbone", ("llava_onevision", "qwen25_vl", "qwen3_vl"))
@pytest.mark.parametrize("layer_index", (2, 3))
@pytest.mark.parametrize("video_state", ("pending", "controller"))
def test_dycoke_video_calls_still_require_batch_one(
    backbone: str, layer_index: int, video_state: str
) -> None:
    executor = DyCoKeVideoExecutor(backbone=backbone, model=object())
    cache = SimpleNamespace()
    if video_state == "pending":
        executor._pending.set(SimpleNamespace())
    else:
        controller = object.__new__(DyCoKeDecodeCacheController)
        controller.owner = executor
        setattr(cache, dycoke_provider._CACHE_CONTROLLER_ATTRIBUTE, controller)

    def native(hidden_states, *, past_key_values):
        pytest.fail("Invalid video batch reached the native layer")

    wrapped = executor._layer_forward_replacement(layer_index, native, layer_count=4)
    with pytest.raises(BackendIntegrationError, match="batch-one rank-three"):
        wrapped(
            object(),
            SimpleNamespace(ndim=3, shape=(2, 5, 8)),
            past_key_values=cache,
        )


@pytest.mark.parametrize("backbone", ("llava_onevision", "qwen25_vl", "qwen3_vl"))
def test_dycoke_executor_lifecycle_is_reversible_for_each_backbone(
    backbone: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[tuple[str, str]] = []

    class _Attention:
        head_dim = 2
        scaling = 2**-0.5

    class _Layer:
        def __init__(self) -> None:
            self.self_attn = _Attention()

        def forward(
            self,
            hidden_states,
            *,
            attention_mask=None,
            past_key_values=None,
            position_embeddings=None,
        ):
            return hidden_states

    class _Model:
        def forward(self, *args, **kwargs):
            return args, kwargs

    class _Handle:
        def remove(self) -> None:
            events.append((backbone, "remove"))

    class _Bridge:
        def register_callback(self, site, callback):
            assert site == f"{backbone}:dycoke"
            assert callable(callback)
            events.append((backbone, "register"))
            return _Handle()

        def detach(self) -> None:
            events.append((backbone, "detach"))

        def runtime_metadata(self):
            return {"fake_bridge": backbone}

    model = _Model()
    layers = tuple(_Layer() for _ in range(4))
    decoder = SimpleNamespace(layers=layers)
    bridge = _Bridge()
    monkeypatch.setattr(
        dycoke_provider,
        "_decoder_owner",
        lambda selected_model, selected: (
            decoder if selected_model is model and selected == backbone else None
        ),
    )
    monkeypatch.setattr(
        dycoke_provider,
        "install_video_bridge",
        lambda selected_model, selected: (
            bridge,
            selected_model is model and selected == backbone,
            f'{selected}:dycoke',
        ),
    )
    executor = DyCoKeVideoExecutor(backbone=backbone, model=model)

    executor.install()
    executor.install()
    assert executor.installed
    assert "forward" in vars(model)
    assert all("forward" in vars(layer) for layer in layers[2:])
    native_non_video_batch = SimpleNamespace(ndim=3, shape=(2, 5, 8))
    for layer in layers:
        assert layer.forward(native_non_video_batch) is native_non_video_batch
    executor.uninstall()

    assert events == [
        (backbone, "register"),
        (backbone, "remove"),
        (backbone, "detach"),
    ]
    assert not executor.installed
    assert "forward" not in vars(model)
    assert all("forward" not in vars(layer) for layer in layers)
