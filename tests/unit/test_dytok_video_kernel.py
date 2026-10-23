from __future__ import annotations

from contextlib import contextmanager
import json
from types import SimpleNamespace
from typing import Any

import pytest

from mllm_comp.backbones import dytok_video as dytok_provider
from mllm_comp.backbones.dytok_video import (
    DyToKVideoExecutor,
    DyToKVideoSignalProvider,
)
from mllm_comp.backbones.qwen3_vl.video_compression import (
    Qwen3VLPackedVideoRequest,
    Qwen3VLVideoCompressionResponse,
    Qwen3VLVideoGrid,
)
from mllm_comp.compressors.video import (
    DyToKVideoKernel,
    VideoTokenPayload,
    aggregate_dytok_frame_importance,
    allocate_dytok_frame_budgets,
    dytok_frame_attention_from_qk,
    dytok_video,
    merge_visionzip_source_groups,
    visionzip_video,
)
from mllm_comp.core.errors import BackendIntegrationError, ConfigError


def _torch():
    return pytest.importorskip("torch", reason="DyToK tensor tests need PyTorch")


def _coordinates(frame_count: int = 2) -> tuple[tuple[int, int, int], ...]:
    return tuple(
        (frame, row, column)
        for frame in range(frame_count)
        for row in range(14)
        for column in range(14)
    )


def _signals(frame_count: int = 2, weights: Any | None = None) -> dict[str, Any]:
    torch = _torch()
    count = frame_count * 196
    attention = torch.arange(196, 0, -1, dtype=torch.float32).repeat(frame_count)
    hidden = torch.nn.functional.one_hot(
        torch.arange(count) % 16,
        num_classes=16,
    ).float()
    if weights is None:
        weights = torch.ones(frame_count, dtype=torch.float32)
    return {
        "vision_encoder_attention": {-2: attention},
        "vision_encoder_hidden_states": {-2: hidden},
        "dytok.frame_importance_by_video": (weights,),
        "detection_attention": {"target_layers": tuple(range(16, 24))},
        "prompt_layout": {"single_visual_placeholder": True},
    }


def _features(frame_count: int = 2, *, requires_grad: bool = False) -> Any:
    torch = _torch()
    values = torch.arange(frame_count * 196 * 3, dtype=torch.float32).reshape(
        frame_count * 196,
        3,
    )
    return values.clone().requires_grad_(requires_grad)


def _payload(
    frame_count: int = 2,
    *,
    weights: Any | None = None,
    requires_grad: bool = False,
) -> VideoTokenPayload:
    return VideoTokenPayload(
        features=_features(frame_count, requires_grad=requires_grad),
        media_ids=("video-dytok",),
        video_offsets=(0, frame_count * 196),
        coordinates=_coordinates(frame_count),
        signals=_signals(frame_count, weights),
    )


def test_dytok_control_plane_freezes_dynamic_detection_and_visionzip() -> None:
    kernel = DyToKVideoKernel()
    kernel.validate_config({"config": {}})
    kernel.validate_config(
        {
            "modality_contract": {
                "algorithm": "dytok",
                "variant": "dytok.video",
            },
            "config": {
                "mode": "dynamic",
                "base_compressor": "visionzip",
                "base_config": {
                    "dominant_num": 126,
                    "contextual_num": 21,
                    "attn_layer": -2,
                    "hidden_state_layer": -2,
                },
                "llm_config": {"target_layers": list(range(16, 24))},
                "upper_limit": 196,
            },
        }
    )
    with pytest.raises(ConfigError, match="non-canonical"):
        kernel.validate_config({"config": {"mode": "static"}})


def test_dytok_budget_allocation_preserves_total_after_caps_and_ties() -> None:
    torch = _torch()

    assert allocate_dytok_frame_budgets(
        torch.tensor([1.0, 0.0]),
        (196, 196),
    ) == (196, 98)
    assert allocate_dytok_frame_budgets(
        torch.ones(3),
        (196, 196, 196),
    ) == (147, 147, 147)
    capped = allocate_dytok_frame_budgets(
        torch.tensor([0.9, 0.1]),
        (100, 200),
    )
    assert capped == (100, 194)
    assert sum(capped) == 294

    with pytest.raises(ValueError, match="positive total mass"):
        allocate_dytok_frame_budgets(torch.zeros(2), (196, 196))


def test_dytok_detection_attention_uses_last_query_and_aggregates_layers() -> None:
    torch = _torch()
    query = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]])
    keys = torch.tensor(
        [
            [[2.0, 0.0], [0.0, 2.0]],
            [[1.0, 0.0], [0.0, 1.0]],
            [[0.0, 1.0], [1.0, 0.0]],
            [[0.0, 2.0], [2.0, 0.0]],
        ]
    )
    layer = dytok_frame_attention_from_qk(
        query,
        keys,
        frame_token_counts=(2, 2),
    )
    importance = aggregate_dytok_frame_importance((layer, layer * 2.0))

    assert tuple(layer.shape) == (2,)
    assert tuple(importance.shape) == (2,)
    assert bool(torch.isfinite(importance).all())
    torch.testing.assert_close(importance.sum(), torch.tensor(1.0))


def test_dytok_dynamic_visionzip_is_exact_deterministic_and_differentiable() -> None:
    torch = _torch()
    payload = _payload(
        weights=torch.tensor([1.0, 0.0]),
        requires_grad=True,
    )

    first = dytok_video(payload)
    second = DyToKVideoKernel()(payload)

    assert first.source_count == 392
    assert first.result_count == 294
    assert first.source_groups == second.source_groups
    assert first.representative_indices == second.representative_indices
    assert first.metadata["total_budgets_by_video"] == ((196, 98),)
    assert first.metadata["dominant_budgets_by_video"] == ((168, 84),)
    assert first.metadata["contextual_budgets_by_video"] == ((28, 14),)
    assert sum(first.metadata["total_budgets_by_video"][0]) == 294
    assert first.metadata["detection_stage"] == (
        "frozen_llm_last_query_visual_attention"
    )
    assert first.metadata["base_compressor"] == "visionzip.video"
    assert first.metadata["training_free_compressor"] is True
    first.features.sum().backward()
    assert payload.features.grad is not None
    assert bool((payload.features.grad != 0).any().item())


def test_uniform_dytok_budget_replays_the_exact_static_visionzip_reduction() -> None:
    torch = _torch()
    payload = _payload(weights=torch.ones(2))

    dynamic = dytok_video(payload)
    static = visionzip_video(payload)

    assert dynamic.metadata["total_budgets_by_video"] == ((147, 147),)
    assert dynamic.source_groups == static.source_groups
    assert dynamic.representative_indices == static.representative_indices
    torch.testing.assert_close(dynamic.features, static.features)


def test_dytok_replays_dynamic_groups_on_every_qwen3_deepstack_branch() -> None:
    torch = _torch()
    features = _features()
    grid = Qwen3VLVideoGrid.from_native(
        (2, 28, 28),
        spatial_merge_size=2,
        second_per_grid_seconds=0.5,
    )
    base = grid.payload(features, media_id="sample-dytok:video-0")
    payload = VideoTokenPayload(
        features=base.features,
        media_ids=base.media_ids,
        video_offsets=base.video_offsets,
        coordinates=base.coordinates,
        signals=_signals(weights=torch.tensor([1.0, 0.0])),
    )
    request = Qwen3VLPackedVideoRequest(
        invocation_id="dytok-video-1",
        media_id="sample-dytok:video-0",
        payload=payload,
        deepstack_features=tuple(features * value for value in (2.0, 3.0, 4.0)),
        grid=grid,
        source_placeholder_positions=tuple(range(1, 393)),
        prompt_input_ids=torch.tensor([[1, *([8] * 392), 2]]),
        source_attention_mask=torch.ones(1, 394, dtype=torch.long),
        source_video_grid_thw=torch.tensor([[2, 28, 28]], dtype=torch.long),
        source_second_per_grid_ts=torch.tensor([0.5]),
    )
    executor = DyToKVideoExecutor(backbone="qwen3_vl", model=object())

    response = executor._projector_callback(request)

    assert isinstance(response, Qwen3VLVideoCompressionResponse)
    assert response.compression_result.result_count == 294
    assert len(response.deepstack_features) == 3
    assert all(
        tuple(branch.shape) == (294, 3) for branch in response.deepstack_features
    )
    expected = merge_visionzip_source_groups(
        features * 2.0,
        response.compression_result,
    )
    torch.testing.assert_close(response.deepstack_features[0], expected)
    metadata = executor.runtime_metadata()
    assert metadata["projector_invocations"] == 1
    assert len(metadata["algorithm_records"]) == 1
    record = metadata["algorithm_records"][0]
    assert record["source_count"] == 392
    assert record["result_count"] == 294
    assert record["target_layers"] == tuple(range(16, 24))
    assert record["frame_importance"] == (1.0, 0.0)
    assert record["provider_frame_importance"] == (1.0, 0.0)
    assert record["provider_kernel_importance_max_abs_delta"] == 0.0
    assert record["total_budgets"] == (196, 98)
    assert record["dominant_budgets"] == (168, 84)
    assert record["contextual_budgets"] == (28, 14)
    assert record["budgets_vary"] is True
    assert record["importance_budget_order_preserved"] is True
    assert record["dominant_contextual_split_exact"] is True
    assert record["source_partition_or_zero_budget_drop_exact"] is True
    assert record["covered_sources_unique"] is True
    assert record["merge_targets_inside_groups"] is True
    assert record["representatives_source_ordered"] is True
    assert record["deepstack_branch_count"] == 3
    assert record["deepstack_result_shapes"] == ((294, 3),) * 3
    json.dumps(metadata)


def test_dytok_provider_combines_visionzip_and_frozen_detection_signals() -> None:
    torch = _torch()
    events: list[str] = []

    class _VisionZipProvider:
        def install(self) -> None:
            events.append("visionzip_install")

        def uninstall(self) -> None:
            events.append("visionzip_uninstall")

        @contextmanager
        def capture_context(self, **kwargs: Any):
            events.append("visionzip_capture")
            yield SimpleNamespace(**kwargs)

        def payload_signals(self, capture, *, token_count, coordinates):
            del capture, token_count, coordinates
            return {
                key: value
                for key, value in _signals().items()
                if key.startswith("vision_encoder_")
            }

        def runtime_metadata(self) -> dict[str, Any]:
            return {"provider": "fake_visionzip"}

    class _Detector:
        def prepare(self) -> None:
            events.append("detector_prepare")

        def frame_importance(self, text: str, frames: Any) -> Any:
            assert text == "Which event happens?"
            assert tuple(frames.shape) == (2, 3, 384, 384)
            events.append("detector_forward")
            return torch.tensor([0.75, 0.25])

        def runtime_metadata(self) -> dict[str, Any]:
            return {"provider": "fake_detector", "frozen": True}

    model = SimpleNamespace()
    provider = DyToKVideoSignalProvider(
        backbone="llava_onevision",
        model=model,
        detection_runner=_Detector(),
        visionzip_provider=_VisionZipProvider(),
    )
    provider.install()
    frames = torch.zeros(2, 3, 384, 384)
    with provider.capture_context(frames=frames) as capture:
        signals = provider.payload_signals(
            capture,
            token_count=392,
            coordinates=_coordinates(),
        )
    payload = VideoTokenPayload(
        features=_features(),
        media_ids=("provider-video",),
        video_offsets=(0, 392),
        coordinates=_coordinates(),
        signals={**signals, "text_context": "Which event happens?"},
    )

    enriched = provider.enrich_payload(payload, capture)

    torch.testing.assert_close(
        enriched.signals["dytok.frame_importance_by_video"][0],
        torch.tensor([0.75, 0.25]),
    )
    assert enriched.signals["prompt_layout"]["assistant_answer_excluded"] is True
    assert enriched.signals["detection_attention"]["target_layers"] == tuple(
        range(16, 24)
    )
    assert provider.runtime_metadata()["detector"]["frozen"] is True
    provider.uninstall()
    assert events == [
        "detector_prepare",
        "visionzip_install",
        "visionzip_capture",
        "detector_prepare",
        "detector_forward",
        "visionzip_uninstall",
    ]


@pytest.mark.parametrize(
    ("backbone", "source_mean", "source_std"),
    (
        (
            "qwen25_vl",
            (0.48145466, 0.4578275, 0.40821073),
            (0.26862954, 0.26130258, 0.27577711),
        ),
        ("qwen3_vl", (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ),
)
def test_dytok_qwen_patchified_video_inverse_matches_siglip_frames(
    backbone: str,
    source_mean: tuple[float, float, float],
    source_std: tuple[float, float, float],
) -> None:
    torch = _torch()
    temporal = 2
    temporal_patch = 2
    channels = 3
    grid_height = 4
    grid_width = 4
    merge_size = 2
    patch_size = 2
    frame_height = grid_height * patch_size
    frame_width = grid_width * patch_size
    clip_mean = torch.tensor(source_mean)
    clip_std = torch.tensor(source_std)
    raw = torch.linspace(
        0.05, 0.95, temporal * temporal_patch * channels * frame_height * frame_width
    ).reshape(
        temporal,
        temporal_patch,
        channels,
        frame_height,
        frame_width,
    ).requires_grad_()
    normalized = (raw - clip_mean.view(1, 1, channels, 1, 1)) / clip_std.view(
        1, 1, channels, 1, 1
    )
    block_height = grid_height // merge_size
    block_width = grid_width // merge_size
    patchified = (
        normalized.reshape(
            temporal,
            temporal_patch,
            channels,
            block_height,
            merge_size,
            patch_size,
            block_width,
            merge_size,
            patch_size,
        )
        .permute(0, 3, 6, 4, 7, 2, 1, 5, 8)
        .contiguous()
        .reshape(
            temporal * grid_height * grid_width,
            channels * temporal_patch * patch_size * patch_size,
        )
    )
    visual = SimpleNamespace(
        config=SimpleNamespace(
            patch_size=patch_size,
            temporal_patch_size=temporal_patch,
            spatial_merge_size=merge_size,
        ),
        patch_embed=SimpleNamespace(
            patch_size=patch_size,
            temporal_patch_size=temporal_patch,
        ),
        spatial_merge_size=merge_size,
    )
    provider = DyToKVideoSignalProvider(
        backbone=backbone,
        model=SimpleNamespace(model=SimpleNamespace(visual=visual)),
        detection_runner=object(),
    )

    reconstructed = provider._qwen_pixels_to_siglip(
        patchified,
        torch.tensor([[temporal, grid_height, grid_width]]),
    )
    expected = (raw.mean(dim=1) - 0.5) / 0.5
    expected = torch.nn.functional.interpolate(
        expected,
        size=(384, 384),
        mode="bilinear",
        align_corners=False,
    )

    assert tuple(reconstructed.shape) == (temporal, channels, 384, 384)
    torch.testing.assert_close(reconstructed, expected, rtol=1e-5, atol=1e-5)
    weights = torch.linspace(0.2, 1.2, reconstructed.numel()).reshape_as(reconstructed)
    actual_gradient = torch.autograd.grad((reconstructed * weights).sum(), raw)[0]
    expected_gradient = torch.autograd.grad((expected * weights).sum(), raw)[0]
    torch.testing.assert_close(actual_gradient, expected_gradient, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("backbone", ("llava_onevision", "qwen25_vl", "qwen3_vl"))
def test_dytok_executor_lifecycle_installs_provider_before_bridge(
    backbone: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[tuple[str, str]] = []

    class _Provider:
        def install(self) -> None:
            events.append((backbone, "provider_install"))

        def uninstall(self) -> None:
            events.append((backbone, "provider_uninstall"))

        def capture_context(self, **kwargs: Any) -> Any:
            del kwargs
            raise AssertionError("fake lifecycle does not execute a video")

        def payload_signals(self, *args: Any, **kwargs: Any) -> Any:
            del args, kwargs
            raise AssertionError("fake lifecycle does not execute a video")

        def runtime_metadata(self) -> dict[str, Any]:
            return {"fake_provider": backbone}

    class _Handle:
        def remove(self) -> None:
            events.append((backbone, "remove"))

    class _Bridge:
        def register_callback(self, site, callback, *, signal_provider=None):
            assert site == f"{backbone}:dynamic_visionzip_reduce"
            assert callable(callback)
            assert isinstance(signal_provider, _Provider)
            events.append((backbone, "register"))
            return _Handle()

        def detach(self) -> None:
            events.append((backbone, "detach"))

        def runtime_metadata(self) -> dict[str, Any]:
            return {"fake_bridge": backbone}

    bridge = _Bridge()
    monkeypatch.setattr(
        dytok_provider,
        "install_video_bridge",
        lambda _model, selected: (
            bridge,
            selected == backbone,
            f'{selected}:dynamic_visionzip_reduce',
        ),
    )
    executor = DyToKVideoExecutor(
        backbone=backbone,
        model=object(),
        signal_provider=_Provider(),
    )

    executor.install()
    executor.install()
    executor.uninstall()

    assert events == [
        (backbone, "provider_install"),
        (backbone, "register"),
        (backbone, "remove"),
        (backbone, "detach"),
        (backbone, "provider_uninstall"),
    ]


def test_dytok_fails_closed_without_detection_or_aligned_vision_signals() -> None:
    torch = _torch()
    payload = _payload()
    missing = VideoTokenPayload(
        features=payload.features,
        media_ids=payload.media_ids,
        video_offsets=payload.video_offsets,
        coordinates=payload.coordinates,
        signals={
            "vision_encoder_attention": {-2: torch.ones(392)},
            "vision_encoder_hidden_states": {-2: torch.ones(392, 4)},
        },
    )
    with pytest.raises(BackendIntegrationError, match="frame_importance"):
        dytok_video(missing)

    class _UnusedVisionProvider:
        def install(self) -> None:
            raise AssertionError("missing detector must fail before visual patching")

        def uninstall(self) -> None:
            pass

        def runtime_metadata(self) -> dict[str, Any]:
            return {}

    provider = DyToKVideoSignalProvider(
        backbone="llava_onevision",
        model=SimpleNamespace(),
        visionzip_provider=_UnusedVisionProvider(),
    )
    with pytest.raises(BackendIntegrationError, match="local dytok_detection"):
        provider.install()
