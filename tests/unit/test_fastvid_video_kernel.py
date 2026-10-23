from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from mllm_comp.backbones.fastvid_video import (
    FastVIDVideoExecutor,
    FastVIDVideoSignalProvider,
)
from mllm_comp.backbones.qwen3_vl.video_compression import (
    Qwen3VLPackedVideoRequest,
    Qwen3VLVideoCompressionResponse,
    Qwen3VLVideoGrid,
)
from mllm_comp.compressors.video import (
    FastVIDVideoKernel,
    VideoTokenPayload,
    fastvid_video,
    replay_fastvid_value_lineage,
)
from mllm_comp.core.errors import BackendIntegrationError, ConfigError


def _torch():
    return pytest.importorskip("torch", reason="FastVID tensor tests need PyTorch")


def _coordinates(
    frame_count: int,
    *,
    side: int = 10,
) -> tuple[tuple[int, int, int], ...]:
    return tuple(
        (frame, row, column)
        for frame in range(frame_count)
        for row in range(side)
        for column in range(side)
    )


def _payload(
    frame_count: int = 16,
    *,
    requires_grad: bool = False,
) -> VideoTokenPayload:
    torch = _torch()
    token_count = frame_count * 100
    features = torch.arange(
        1,
        token_count * 4 + 1,
        dtype=torch.float32,
    ).reshape(token_count, 4)
    features = features.clone().requires_grad_(requires_grad)
    raw = torch.arange(
        1,
        token_count * 3 + 1,
        dtype=torch.float32,
    ).reshape(token_count, 3)
    return VideoTokenPayload(
        features=features,
        media_ids=("video-fastvid",),
        video_offsets=(0, token_count),
        coordinates=_coordinates(frame_count),
        raw_features=raw,
        raw_video_offsets=(0, token_count),
        raw_coordinates=_coordinates(frame_count),
        signals={
            "frame_global_features": torch.ones(frame_count, 4),
            "token_saliency_attention": torch.arange(
                100,
                0,
                -1,
                dtype=torch.float32,
            ).repeat(frame_count),
        },
    )


def test_fastvid_control_plane_freezes_only_the_released_main_recipe() -> None:
    kernel = FastVIDVideoKernel()
    kernel.validate_config({"config": {}})
    kernel.validate_config(
        {
            "modality_contract": {
                "algorithm": "fastvid",
                "variant": "fastvid.video",
            },
            "config": {
                "retention_ratio": 0.10,
                "dyseg_c": 8,
                "dyseg_tau": 0.9,
                "stprune_d": 0.4,
                "dtm_p": 4,
                "dtm_beta": 0.6,
            },
        }
    )
    with pytest.raises(ConfigError, match="non-canonical"):
        kernel.validate_config({"config": {"retention_ratio": 0.2}})


def test_fastvid_is_deterministic_differentiable_and_records_overlap() -> None:
    torch = _torch()
    payload = _payload(requires_grad=True)

    first = fastvid_video(payload)
    second = FastVIDVideoKernel()(payload)

    assert first.source_count == 1600
    assert first.result_count == 160
    assert first.source_groups == second.source_groups
    assert first.representative_indices == second.representative_indices
    torch.testing.assert_close(first.features, second.features)
    assert first.metadata["segment_sizes_by_video"] == ((1, 1, 1, 1, 1, 1, 1, 9),)
    assert first.metadata["context_allocations_by_video"] == (
        (4, 4, 4, 4, 4, 4, 4, 12, 0, 0, 0, 12, 0, 0, 0, 12),
    )
    assert first.metadata["frame_output_counts_by_video"] == (
        (10, 10, 10, 10, 10, 10, 10, 18, 6, 6, 6, 18, 6, 6, 6, 18),
    )
    records = first.metadata["value_records_by_video"][0]
    multi = tuple(record for record in records if record["kind"] == "multi_frame_dtm")
    assert len(multi) == 36
    assigned_counts: dict[int, int] = {}
    for record in multi:
        for source in record["assigned_indices"]:
            assigned_counts[source] = assigned_counts.get(source, 0) + 1
    assert assigned_counts
    assert set(assigned_counts.values()) == {3}
    assert all(len(group) == 1 for group in first.source_groups[0])
    assert first.metadata["layout_source_groups"] == "sampled_center_singletons"
    assert first.metadata["value_lineage_policy"] == (
        "exact_potentially_overlapping_assignments"
    )

    first.features.sum().backward()
    assert payload.features.grad is not None
    assert bool((payload.features.grad != 0).any().item())


def test_fastvid_keeps_video_boundaries_and_replays_exact_value_lineage() -> None:
    torch = _torch()
    first = _payload(1)
    second = _payload(1)
    payload = VideoTokenPayload(
        features=torch.cat((first.features, second.features + 10000.0), dim=0),
        media_ids=("video-a", "video-b"),
        video_offsets=(0, 100, 200),
        coordinates=(*first.coordinates, *second.coordinates),
        raw_features=torch.cat((first.raw_features, second.raw_features), dim=0),
        raw_video_offsets=(0, 100, 200),
        raw_coordinates=(*first.raw_coordinates, *second.raw_coordinates),
        signals={
            "frame_global_features": torch.ones(2, 4),
            "token_saliency_attention": torch.cat(
                (
                    first.signals["token_saliency_attention"],
                    second.signals["token_saliency_attention"],
                )
            ),
        },
    )

    result = fastvid_video(payload)

    assert result.source_video_offsets == (0, 100, 200)
    assert result.video_offsets == (0, 10, 20)
    assert tuple(delta.result_count for delta in result.layout_deltas) == (10, 10)
    branch = payload.features[:100] * 5.0
    replayed = replay_fastvid_value_lineage(branch, result, video_index=0)
    torch.testing.assert_close(replayed, result.features[:10] * 5.0)


def test_fastvid_qwen3_executor_replays_all_deepstack_value_records() -> None:
    torch = _torch()
    source = _payload(1)
    grid = Qwen3VLVideoGrid.from_native(
        (1, 20, 20),
        spatial_merge_size=2,
        second_per_grid_seconds=0.5,
    )
    payload = VideoTokenPayload(
        features=source.features,
        media_ids=("sample-fastvid:video-0",),
        video_offsets=(0, 100),
        coordinates=grid.coordinates,
        raw_features=source.raw_features,
        raw_video_offsets=(0, 100),
        raw_coordinates=source.raw_coordinates,
        signals=source.signals,
    )
    request = Qwen3VLPackedVideoRequest(
        invocation_id="fastvid-video-1",
        media_id="sample-fastvid:video-0",
        payload=payload,
        deepstack_features=tuple(source.features * scale for scale in (2.0, 3.0)),
        grid=grid,
        source_placeholder_positions=tuple(range(1, 101)),
        prompt_input_ids=torch.tensor([[1, *([8] * 100), 2]]),
        source_attention_mask=torch.ones(1, 102, dtype=torch.long),
        source_video_grid_thw=torch.tensor([[1, 20, 20]], dtype=torch.long),
        source_second_per_grid_ts=torch.tensor([0.5]),
    )
    executor = FastVIDVideoExecutor(backbone="qwen3_vl", model=object())

    response = executor._projector_callback(request)

    assert isinstance(response, Qwen3VLVideoCompressionResponse)
    assert response.compression_result.result_count == 10
    assert len(response.deepstack_features) == 2
    for source_branch, result_branch in zip(
        request.deepstack_features,
        response.deepstack_features,
    ):
        torch.testing.assert_close(
            result_branch,
            replay_fastvid_value_lineage(
                source_branch,
                response.compression_result,
            ),
        )
    metadata = executor.runtime_metadata()
    assert metadata["projector_invocations"] == 1
    assert len(metadata["algorithm_records"]) == 1
    record = metadata["algorithm_records"][0]
    assert record["stages"] == ("dyseg", "stprune", "dtm")
    assert record["source_count"] == 100
    assert record["result_count"] == 10
    assert record["projected_features_shape"] == (100, 4)
    assert record["raw_features_shape"] == (100, 3)
    assert record["frame_global_features_shape"] == (1, 4)
    assert record["token_saliency_attention_shape"] == (100,)
    assert record["segment_sizes"] == (1,)
    assert record["context_allocations"] == (4,)
    assert record["frame_output_counts"] == (10,)
    assert record["salient_count"] == 6
    assert record["single_frame_dtm_count"] == 4
    assert record["multi_frame_dtm_count"] == 0
    assert record["sampled_center_singleton_layout"]
    assert record["representatives_source_ordered"]
    assert record["value_lineage_record_count"] == 10
    assert record["deepstack_branch_count"] == 2


def test_llava_fastvid_provider_is_video_scoped_and_attaches_raw_stream() -> None:
    torch = _torch()
    nn = torch.nn

    class _PoolingHead(nn.Module):
        def forward(self, raw: Any) -> tuple[Any, Any]:
            scores = torch.softmax(raw.float().mean(dim=-1), dim=-1)
            return raw.mean(dim=1), scores.unsqueeze(1)

    class _Inner(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.calls: list[tuple[bool, bool]] = []

        def forward(
            self,
            pixels: Any,
            output_hidden_states: bool = False,
            return_dict: bool = False,
        ) -> Any:
            self.calls.append((output_hidden_states, return_dict))
            frames = int(pixels.shape[0])
            raw = torch.arange(
                1,
                frames * 100 * 4 + 1,
                dtype=torch.float32,
            ).reshape(frames, 100, 4)
            return SimpleNamespace(hidden_states=(raw - 1.0, raw))

    class _Model:
        def __init__(self) -> None:
            self.inner = _Inner()
            self.outer = SimpleNamespace(
                vision_tower=self.inner,
                num_patches_per_side=10,
            )

        def get_vision_tower(self) -> Any:
            return self.outer

    model = _Model()
    native_forward = model.inner.forward
    provider = FastVIDVideoSignalProvider(
        backbone="llava_onevision",
        model=model,
        pooling_head=_PoolingHead(),
    )
    provider.install()

    frames = torch.zeros(2, 3, 4, 4)
    model.inner(frames)
    assert model.inner.calls[-1] == (False, False)
    with provider.capture_context(frames=frames) as capture:
        model.inner(frames)
    provider.payload_signals(
        capture,
        token_count=200,
        coordinates=_coordinates(2),
    )
    base = VideoTokenPayload(
        features=torch.ones(200, 4),
        media_ids=("llava-fastvid",),
        video_offsets=(0, 200),
        coordinates=_coordinates(2),
    )
    enriched = provider.enrich_payload(base, capture)

    assert model.inner.calls[-1] == (True, True)
    assert tuple(enriched.raw_features.shape) == (200, 4)
    assert enriched.raw_video_offsets == (0, 200)
    assert tuple(enriched.signals["frame_global_features"].shape) == (2, 4)
    assert tuple(enriched.signals["token_saliency_attention"].shape) == (200,)
    assert enriched.metadata["fastvid_signal_source"] == ("frozen_siglip_pooling_head")
    provider.uninstall()
    assert getattr(model, "_mllm_comp_fastvid_video_signal_provider", None) is None
    assert getattr(model.inner.forward, "__func__", None) is getattr(
        native_forward,
        "__func__",
        None,
    )


@pytest.mark.parametrize(
    ("backbone", "expected_first_raw"),
    (("qwen25_vl", 397.0), ("qwen3_vl", 1.0)),
)
def test_qwen_fastvid_provider_aligns_attention_and_restores_native_raster(
    backbone: str,
    expected_first_raw: float,
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
    reverse = torch.arange(99, -1, -1)
    visual = SimpleNamespace(
        blocks=(nn.Identity(), target),
        spatial_merge_unit=4,
        spatial_merge_size=2,
        get_window_index=lambda _grid: (reverse, [0, 400]),
    )
    model = SimpleNamespace(model=SimpleNamespace(visual=visual))
    provider = FastVIDVideoSignalProvider(backbone=backbone, model=model)
    provider.install()
    hidden = torch.arange(400, dtype=torch.float32).unsqueeze(1).repeat(1, 4)
    boundaries = torch.tensor([0, 400], dtype=torch.int32)
    position_embeddings = (torch.ones(400, 2), torch.zeros(400, 2))
    grid_thw = torch.tensor([[1, 20, 20]], dtype=torch.long)

    with provider.capture_context(grid_thw=grid_thw) as capture:
        target(hidden, boundaries, position_embeddings)
    provider.payload_signals(
        capture,
        token_count=100,
        coordinates=_coordinates(1),
    )
    base = VideoTokenPayload(
        features=torch.arange(1, 401, dtype=torch.float32).reshape(100, 4),
        media_ids=(f"{backbone}-fastvid",),
        video_offsets=(0, 100),
        coordinates=_coordinates(1),
    )
    enriched = provider.enrich_payload(base, capture)

    assert tuple(enriched.raw_features.shape) == (400, 4)
    assert float(enriched.raw_features[0, 0].item()) == expected_first_raw
    assert tuple(enriched.signals["token_saliency_attention"].shape) == (100,)
    assert bool(torch.isfinite(enriched.signals["token_saliency_attention"]).all())
    assert tuple(enriched.signals["frame_global_features"].shape) == (1, 4)
    provider.uninstall()


def test_fastvid_fails_closed_on_missing_signals_layout_drift_and_identity() -> None:
    torch = _torch()
    source = _payload(1)
    without_raw = VideoTokenPayload(
        features=source.features,
        media_ids=source.media_ids,
        video_offsets=source.video_offsets,
        coordinates=source.coordinates,
        signals=source.signals,
    )
    with pytest.raises(BackendIntegrationError, match="raw features"):
        fastvid_video(without_raw)

    without_attention = VideoTokenPayload(
        features=source.features,
        media_ids=source.media_ids,
        video_offsets=source.video_offsets,
        coordinates=source.coordinates,
        raw_features=source.raw_features,
        raw_video_offsets=source.raw_video_offsets,
        raw_coordinates=source.raw_coordinates,
        signals={"frame_global_features": torch.ones(1, 4)},
    )
    with pytest.raises(BackendIntegrationError, match="token_saliency_attention"):
        fastvid_video(without_attention)
    with pytest.raises(ConfigError, match="non-canonical"):
        fastvid_video(source, config={"dtm_beta": 0.5})

    small = VideoTokenPayload(
        features=torch.ones(9, 4),
        media_ids=("small",),
        video_offsets=(0, 9),
        coordinates=tuple((0, index // 3, index % 3) for index in range(9)),
        raw_features=torch.ones(9, 4),
        raw_video_offsets=(0, 9),
        raw_coordinates=tuple((0, index // 3, index % 3) for index in range(9)),
        signals={
            "frame_global_features": torch.ones(1, 4),
            "token_saliency_attention": torch.ones(9),
        },
    )
    with pytest.raises(BackendIntegrationError, match="non-identity raster"):
        fastvid_video(small)

    masked = VideoTokenPayload(
        features=source.features,
        media_ids=source.media_ids,
        video_offsets=source.video_offsets,
        coordinates=source.coordinates,
        raw_features=source.raw_features,
        raw_video_offsets=source.raw_video_offsets,
        raw_coordinates=source.raw_coordinates,
        token_mask=torch.tensor([True] * 99 + [False]),
        signals=source.signals,
    )
    with pytest.raises(BackendIntegrationError, match="padded"):
        fastvid_video(masked)

    coordinates = list(source.coordinates)
    coordinates[0], coordinates[1] = coordinates[1], coordinates[0]
    permuted = VideoTokenPayload(
        features=source.features,
        media_ids=source.media_ids,
        video_offsets=source.video_offsets,
        coordinates=tuple(coordinates),
        raw_features=source.raw_features,
        raw_video_offsets=source.raw_video_offsets,
        raw_coordinates=source.raw_coordinates,
        signals=source.signals,
    )
    with pytest.raises(BackendIntegrationError, match="row-major"):
        fastvid_video(permuted)
