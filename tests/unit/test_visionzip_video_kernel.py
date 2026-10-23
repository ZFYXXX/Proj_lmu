from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from mllm_comp.backbones.qwen3_vl.video_compression import (
    Qwen3VLPackedVideoRequest,
    Qwen3VLVideoCompressionResponse,
    Qwen3VLVideoGrid,
)
from mllm_comp.backbones.visionzip_video import (
    VisionZipVideoExecutor,
    VisionZipVideoSignalProvider,
)
from mllm_comp.compressors.video import (
    VideoTokenPayload,
    VisionZipVideoKernel,
    merge_visionzip_source_groups,
    visionzip_video,
)
from mllm_comp.core.errors import BackendIntegrationError, ConfigError


def _torch():
    return pytest.importorskip("torch", reason="VisionZip tensor tests need PyTorch")


def _coordinates(frame_count: int = 1) -> tuple[tuple[int, int, int], ...]:
    return tuple(
        (frame, row, column)
        for frame in range(frame_count)
        for row in range(14)
        for column in range(14)
    )


def _signals(frame_count: int = 1) -> dict[str, dict[int, Any]]:
    torch = _torch()
    attention = torch.arange(196, 0, -1, dtype=torch.float32).repeat(frame_count)
    hidden = torch.zeros(frame_count * 196, 21, dtype=torch.float32)
    contextual_targets = tuple(range(126, 187, 3))
    for frame in range(frame_count):
        offset = frame * 196
        for position, target in enumerate(contextual_targets):
            hidden[offset + target, position] = 1.0
        for source in range(126, 196):
            if source not in contextual_targets:
                hidden[offset + source, 0] = 1.0
        # The source at 127 precedes the next contextual target but is assigned
        # to target 186.  Its value target stays 186 while its layout anchor is
        # causally legalized to 127.
        hidden[offset + 127].zero_()
        hidden[offset + 127, -1] = 1.0
    return {
        "vision_encoder_attention": {-2: attention},
        "vision_encoder_hidden_states": {-2: hidden},
    }


def _features(frame_count: int = 1, *, requires_grad: bool = False):
    torch = _torch()
    count = frame_count * 196
    values = torch.arange(count * 3, dtype=torch.float32).reshape(count, 3)
    return values.clone().requires_grad_(requires_grad)


def _payload(
    frame_count: int = 1,
    *,
    requires_grad: bool = False,
) -> VideoTokenPayload:
    return VideoTokenPayload(
        features=_features(frame_count, requires_grad=requires_grad),
        media_ids=("video-visionzip",),
        video_offsets=(0, frame_count * 196),
        coordinates=_coordinates(frame_count),
        signals=_signals(frame_count),
    )


def test_visionzip_control_plane_freezes_the_released_126_plus_21_recipe() -> None:
    kernel = VisionZipVideoKernel()
    kernel.validate_config({"config": {}})
    kernel.validate_config(
        {
            "modality_contract": {
                "algorithm": "visionzip",
                "variant": "visionzip.video",
            },
            "config": {
                "dominant_num": 126,
                "contextual_num": 21,
                "attn_layer": -2,
                "hidden_state_layer": -2,
            },
        }
    )
    with pytest.raises(ConfigError, match="non-canonical"):
        kernel.validate_config({"config": {"dominant_num": 125}})


def test_visionzip_is_exact_deterministic_causal_and_differentiable() -> None:
    torch = _torch()
    payload = _payload(requires_grad=True)

    first = visionzip_video(payload)
    second = VisionZipVideoKernel()(payload)

    assert first.source_count == 196
    assert first.result_count == 147
    assert first.source_groups == second.source_groups
    assert first.representative_indices == second.representative_indices
    assert first.representative_indices[0] == tuple(
        sorted(first.representative_indices[0])
    )
    targets = first.metadata["merge_target_indices_by_video"][0]
    target_position = targets.index(186)
    assert first.source_groups[0][target_position] == (127, 186)
    assert first.representative_indices[0][target_position] == 127
    assert first.metadata["anchor_legalizations_by_video"] == (1,)
    torch.testing.assert_close(
        first.features[target_position],
        payload.features[186] + payload.features[127],
    )
    assert first.metadata["selection_scope"] == "independent_per_temporal_slice"
    assert first.metadata["merge_value"] == (
        "target_plus_mean_assigned_excluding_target"
    )
    assert first.metadata["image_variant_shared"] is False

    first.features.sum().backward()
    assert payload.features.grad is not None
    assert bool((payload.features.grad != 0).any().item())


def test_visionzip_keeps_frame_and_video_boundaries_with_exact_slice_budgets() -> None:
    torch = _torch()
    source = torch.cat((_features(2), _features(1) + 10000.0), dim=0)
    first_signals = _signals(2)
    second_signals = _signals(1)
    payload = VideoTokenPayload(
        features=source,
        media_ids=("video-a", "video-b"),
        video_offsets=(0, 392, 588),
        coordinates=(*_coordinates(2), *_coordinates(1)),
        signals={
            "vision_encoder_attention": {
                -2: torch.cat(
                    (
                        first_signals["vision_encoder_attention"][-2],
                        second_signals["vision_encoder_attention"][-2],
                    )
                )
            },
            "vision_encoder_hidden_states": {
                -2: torch.cat(
                    (
                        first_signals["vision_encoder_hidden_states"][-2],
                        second_signals["vision_encoder_hidden_states"][-2],
                    )
                )
            },
        },
    )

    result = visionzip_video(payload)

    assert result.source_video_offsets == (0, 392, 588)
    assert result.video_offsets == (0, 294, 441)
    assert tuple(delta.result_count for delta in result.layout_deltas) == (294, 147)
    assert tuple(
        len(records) for records in result.metadata["frame_records_by_video"]
    ) == (
        2,
        1,
    )
    assert all(
        record["result_count"] == 147
        for records in result.metadata["frame_records_by_video"]
        for record in records
    )


def test_visionzip_replays_exact_merge_targets_on_qwen3_deepstack() -> None:
    torch = _torch()
    payload = _payload()
    result = visionzip_video(payload)
    branch = payload.features * 5.0

    merged = merge_visionzip_source_groups(branch, result)
    targets = result.metadata["merge_target_indices_by_video"][0]
    position = targets.index(186)

    assert tuple(merged.shape) == (147, 3)
    torch.testing.assert_close(
        merged[position],
        branch[186] + branch[127],
    )


def test_visionzip_qwen3_executor_synchronizes_all_deepstack_branches() -> None:
    torch = _torch()
    features = _features()
    grid = Qwen3VLVideoGrid.from_native(
        (1, 28, 28),
        spatial_merge_size=2,
        second_per_grid_seconds=0.5,
    )
    base = grid.payload(features, media_id="sample-vz:video-0")
    payload = VideoTokenPayload(
        features=base.features,
        media_ids=base.media_ids,
        video_offsets=base.video_offsets,
        coordinates=base.coordinates,
        signals=_signals(),
    )
    request = Qwen3VLPackedVideoRequest(
        invocation_id="visionzip-video-1",
        media_id="sample-vz:video-0",
        payload=payload,
        deepstack_features=tuple(features * value for value in (2.0, 3.0, 4.0)),
        grid=grid,
        source_placeholder_positions=tuple(range(1, 197)),
        prompt_input_ids=torch.tensor([[1, *([8] * 196), 2]]),
        source_attention_mask=torch.ones(1, 198, dtype=torch.long),
        source_video_grid_thw=torch.tensor([[1, 28, 28]], dtype=torch.long),
        source_second_per_grid_ts=torch.tensor([0.5]),
    )
    executor = VisionZipVideoExecutor(backbone="qwen3_vl", model=object())

    response = executor._projector_callback(request)

    assert isinstance(response, Qwen3VLVideoCompressionResponse)
    assert response.compression_result.result_count == 147
    assert len(response.deepstack_features) == 3
    assert all(
        tuple(branch.shape) == (147, 3) for branch in response.deepstack_features
    )
    assert (
        response.synchronization_source_groups
        == response.compression_result.source_groups
    )
    # Fixed merge targets make the branch value transform linear.
    for factor, branch in zip((2.0, 3.0, 4.0), response.deepstack_features):
        torch.testing.assert_close(branch, response.compression_result.features * factor)
    assert executor.runtime_metadata()["projector_invocations"] == 1
    record = executor.runtime_metadata()["algorithm_records"][0]
    assert record["attention_shape"] == (196,)
    assert record["hidden_state_shape"] == (196, 21)
    assert record["temporal_slice_count"] == 1
    assert record["dominant_count"] == 126
    assert record["contextual_target_count"] == 21
    assert record["assigned_contextual_source_count"] == 49
    assert record["source_partition_exact"] is True
    assert record["merge_targets_inside_groups"] is True
    assert record["representatives_source_ordered"] is True
    assert record["image_variant_shared"] is False
    assert record["deepstack_branch_count"] == 3
    assert record["deepstack_result_shapes"] == ((147, 3),) * 3


def test_llava_signal_provider_is_video_scoped_pooled_and_reversible() -> None:
    torch = _torch()
    nn = torch.nn

    class _Inner(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.calls: list[tuple[bool, bool, bool]] = []

        def forward(
            self,
            pixels: Any,
            output_hidden_states: bool = False,
            output_attentions: bool = False,
            return_dict: bool = False,
        ) -> Any:
            self.calls.append((output_hidden_states, output_attentions, return_dict))
            frames = int(pixels.shape[0])
            hidden = torch.arange(frames * 196 * 4, dtype=torch.float32).reshape(
                frames, 196, 4
            )
            attention = torch.ones(frames, 2, 196, 196)
            return SimpleNamespace(
                hidden_states=(hidden - 2.0, hidden, hidden + 2.0),
                attentions=(attention, attention * 2.0),
            )

    class _Model:
        def __init__(self) -> None:
            self.config = SimpleNamespace(mm_spatial_pool_stride=2)
            self.inner = _Inner()
            self.outer = SimpleNamespace(
                vision_tower=self.inner,
                num_patches_per_side=14,
            )

        def get_vision_tower(self) -> Any:
            return self.outer

        def get_2dPool(self, value: Any, stride: int = 2) -> Any:
            batch, _tokens, hidden = value.shape
            image = value.reshape(batch, 14, 14, hidden).permute(0, 3, 1, 2)
            pooled = torch.nn.functional.avg_pool2d(image, stride)
            return pooled.permute(0, 2, 3, 1).reshape(batch, -1, hidden)

    model = _Model()
    native_forward = model.inner.forward
    provider = VisionZipVideoSignalProvider(
        backbone="llava_onevision",
        model=model,
    )
    provider.install()

    model.inner(torch.zeros(2, 3, 4, 4))
    assert model.inner.calls[-1] == (False, False, False)
    frames = torch.zeros(2, 3, 4, 4)
    with provider.capture_context(frames=frames) as capture:
        model.inner(frames)
    signals = provider.payload_signals(
        capture,
        token_count=98,
        coordinates=tuple(
            (frame, row, column)
            for frame in range(2)
            for row in range(7)
            for column in range(7)
        ),
    )

    assert model.inner.calls[-1] == (True, True, True)
    assert tuple(signals["vision_encoder_attention"][-2].shape) == (98,)
    assert tuple(signals["vision_encoder_hidden_states"][-2].shape) == (98, 4)
    provider.uninstall()
    assert getattr(model, "_mllm_comp_visionzip_video_signal_provider", None) is None
    assert getattr(model.inner.forward, "__func__", None) is getattr(
        native_forward, "__func__", None
    )


@pytest.mark.parametrize(
    ("backbone", "expected_first_group"),
    (("qwen25_vl", 4.0), ("qwen3_vl", 0.0)),
)
def test_qwen_signal_provider_merges_native_groups_and_restores_q25_raster(
    backbone: str,
    expected_first_group: float,
) -> None:
    torch = _torch()
    nn = torch.nn

    class _Attention(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.num_heads = 2
            self.scaling = 2**-0.5
            self.qkv = nn.Linear(4, 12, bias=False)
            with torch.no_grad():
                self.qkv.weight.copy_(torch.eye(4).repeat(3, 1))

    class _Block(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.norm1 = nn.Identity()
            self.attn = _Attention()

        def forward(
            self,
            hidden_states: Any,
            cu_seqlens: Any,
            position_embeddings: Any,
        ) -> Any:
            del cu_seqlens, position_embeddings
            return hidden_states + 1.0

    target = _Block()
    visual = SimpleNamespace(
        blocks=(nn.Identity(), target, nn.Identity()),
        spatial_merge_unit=4,
        get_window_index=lambda _grid: (torch.tensor([1, 0]), [0, 8]),
    )
    model = SimpleNamespace(model=SimpleNamespace(visual=visual))
    provider = VisionZipVideoSignalProvider(backbone=backbone, model=model)
    provider.install()
    hidden = torch.arange(8, dtype=torch.float32).unsqueeze(1).repeat(1, 4)
    boundaries = torch.tensor([0, 8], dtype=torch.int32)
    position_embeddings = (torch.ones(8, 2), torch.zeros(8, 2))

    with provider.capture_context(
        grid_thw=torch.tensor([[1, 2, 4]], dtype=torch.long)
    ) as capture:
        target(hidden, boundaries, position_embeddings)
    signals = provider.payload_signals(
        capture,
        token_count=2,
        coordinates=((0, 0, 0), (0, 0, 1)),
    )
    aligned_hidden = signals["vision_encoder_hidden_states"][-2]

    assert tuple(aligned_hidden.shape) == (2, 16)
    assert float(aligned_hidden[0, 0].item()) == expected_first_group + 1.0
    assert tuple(signals["vision_encoder_attention"][-2].shape) == (2,)
    assert bool(torch.isfinite(signals["vision_encoder_attention"][-2]).all())
    provider.uninstall()


def test_visionzip_fails_closed_on_identity_slices_and_signal_drift() -> None:
    torch = _torch()
    small = VideoTokenPayload(
        features=torch.ones(147, 3),
        media_ids=("small",),
        video_offsets=(0, 147),
        coordinates=tuple((0, index // 21, index % 21) for index in range(147)),
        signals={
            "vision_encoder_attention": {-2: torch.ones(147)},
            "vision_encoder_hidden_states": {-2: torch.ones(147, 4)},
        },
    )
    with pytest.raises(BackendIntegrationError, match="more than 147"):
        visionzip_video(small)

    missing = _payload()
    missing = VideoTokenPayload(
        features=missing.features,
        media_ids=missing.media_ids,
        video_offsets=missing.video_offsets,
        coordinates=missing.coordinates,
        signals={"vision_encoder_attention": {-2: torch.ones(196)}},
    )
    with pytest.raises(BackendIntegrationError, match="hidden_states"):
        visionzip_video(missing)
    with pytest.raises(ConfigError, match="non-canonical"):
        visionzip_video(_payload(), config={"contextual_num": 20})

    source = _payload()
    permuted_coordinates = list(source.coordinates)
    permuted_coordinates[0], permuted_coordinates[1] = (
        permuted_coordinates[1],
        permuted_coordinates[0],
    )
    permuted = VideoTokenPayload(
        features=source.features,
        media_ids=source.media_ids,
        video_offsets=source.video_offsets,
        coordinates=tuple(permuted_coordinates),
        signals=source.signals,
    )
    with pytest.raises(BackendIntegrationError, match="row-major"):
        visionzip_video(permuted)
