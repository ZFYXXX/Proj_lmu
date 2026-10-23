from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from mllm_comp.backbones import longvu_video as longvu_provider
from mllm_comp.backbones.longvu_video import (
    LongVUVideoExecutor,
    LongVUVideoSignalProvider,
)
from mllm_comp.backbones.qwen3_vl.video_compression import (
    Qwen3VLPackedVideoRequest,
    Qwen3VLVideoCompressionResponse,
    Qwen3VLVideoGrid,
)
from mllm_comp.compressors.video import (
    LongVUVideoKernel,
    VideoTokenCoordinate,
    VideoTokenPayload,
    longvu_structural_groups,
    longvu_video,
    pool_longvu_source_groups,
    select_longvu_frames,
)
from mllm_comp.core.errors import BackendIntegrationError, ConfigError


def _torch():
    return pytest.importorskip("torch", reason="LongVU tensor tests need PyTorch")


def _coordinates(
    frame_count: int,
    *,
    side: int,
) -> tuple[VideoTokenCoordinate, ...]:
    return tuple(
        VideoTokenCoordinate(t=frame, h=row, w=column, frame_index=frame)
        for frame in range(frame_count)
        for row in range(side)
        for column in range(side)
    )


def _payload(
    *,
    frame_count: int = 2,
    side: int = 14,
    raw_side: int = 16,
    requires_grad: bool = False,
) -> VideoTokenPayload:
    torch = _torch()
    source_count = frame_count * side * side
    raw_count = frame_count * raw_side * raw_side
    projected = torch.arange(
        1,
        source_count * 4 + 1,
        dtype=torch.float32,
    ).reshape(source_count, 4)
    raw = torch.arange(
        1,
        raw_count * 3 + 1,
        dtype=torch.float32,
    ).reshape(raw_count, 3)
    raw = raw.clone().requires_grad_(requires_grad)
    return VideoTokenPayload(
        features=projected,
        media_ids=("sample-longvu:video-0",),
        video_offsets=(0, source_count),
        coordinates=_coordinates(frame_count, side=side),
        raw_features=raw,
        raw_video_offsets=(0, raw_count),
        raw_coordinates=_coordinates(frame_count, side=raw_side),
        signals={
            "pixels": torch.zeros(frame_count, 3, 8, 8),
            "pixel_video_offsets": (0, frame_count),
            "dinov2_features": torch.ones(frame_count, 576, 5),
            "dinov2_video_offsets": (0, frame_count),
        },
    )


class _TinyFusion:
    def __init__(self) -> None:
        torch = _torch()
        self.raw = torch.nn.Linear(3, 4, bias=False)
        self.dino = torch.nn.Linear(5, 4, bias=False)

    def __call__(self, raw: Any, dino: Any) -> Any:
        raw_value = self.raw(raw.mean(dim=1))
        dino_value = self.dino(dino.mean(dim=1))
        return (raw_value + dino_value)[:, None, :].expand(-1, 144, -1)

    def parameters(self):
        yield from self.raw.parameters()
        yield from self.dino.parameters()

    def state_dict(self):
        return {
            "raw.weight": self.raw.weight,
            "dino.weight": self.dino.weight,
        }

    def load_state_dict(self, state: Any, strict: bool = True):
        assert strict is True
        with _torch().no_grad():
            self.raw.weight.copy_(state["raw.weight"])
            self.dino.weight.copy_(state["dino.weight"])


def test_longvu_control_plane_freezes_recipe_and_structural_partition() -> None:
    kernel = LongVUVideoKernel()
    kernel.validate_config({"config": {}})
    kernel.validate_config(
        {
            "modality_contract": {
                "algorithm": "longvu",
                "variant": "longvu.video",
            },
            "config": {
                "window_size": 16,
                "threshold": 0.83,
                "max_frames": 400,
                "min_frames": 1,
                "dino_interp_size": 576,
                "vision_hidden_size": 1024,
                "query_num": 144,
                "connector_depth": 3,
            },
        }
    )
    with pytest.raises(ConfigError, match="non-canonical"):
        kernel.validate_config({"config": {"threshold": 0.8}})

    groups = longvu_structural_groups(14, 14)
    assert len(groups) == 144
    assert tuple(sorted(index for group in groups for index in group)) == tuple(
        range(196)
    )
    assert tuple(group[0] for group in groups) == tuple(
        sorted(group[0] for group in groups)
    )
    with pytest.raises(BackendIntegrationError, match="at least 12x12"):
        longvu_structural_groups(11, 14)


def test_longvu_selects_middle_frame_deterministically_and_clamps() -> None:
    torch = _torch()
    identical = torch.ones(20, 4, 3)
    selected, scores = select_longvu_frames(
        identical,
        window_size=16,
        threshold=0.83,
        max_frames=400,
        min_frames=1,
    )
    assert selected.tolist() == [8, 18]
    assert len(scores) == 2
    assert scores[0][8] == 0.0
    assert scores[1][2] == 0.0

    unique = torch.eye(5).reshape(5, 1, 5)
    clamped, _ = select_longvu_frames(
        unique,
        window_size=16,
        threshold=0.83,
        max_frames=2,
        min_frames=1,
    )
    assert clamped.tolist() == [0, 2]


def test_longvu_kernel_is_deterministic_reducing_and_trainable() -> None:
    torch = _torch()
    payload = _payload(requires_grad=True)
    fusion = _TinyFusion()
    first = longvu_video(payload, fusion_module=fusion)
    second = LongVUVideoKernel()(payload, fusion_module=fusion)

    assert first.source_count == 392
    assert first.result_count == 144
    assert first.metadata["selected_frame_indices_by_video"] == ((1,),)
    assert first.metadata["layout_source_groups"] == "dense_raster_partition_12x12"
    assert first.source_groups == second.source_groups
    assert first.representative_indices == second.representative_indices
    torch.testing.assert_close(first.features, second.features)
    assert all(index >= 196 for group in first.source_groups[0] for index in group)

    first.features.sum().backward()
    assert payload.raw_features.grad is not None
    assert bool((payload.raw_features.grad != 0).any().item())
    assert all(
        parameter.grad is not None and bool((parameter.grad != 0).any().item())
        for parameter in fusion.parameters()
    )


def test_longvu_qwen3_executor_pools_deepstack_through_exact_groups() -> None:
    torch = _torch()
    payload = _payload(frame_count=2)
    grid = Qwen3VLVideoGrid.from_native(
        (2, 28, 28),
        spatial_merge_size=2,
        second_per_grid_seconds=0.5,
    )
    request = Qwen3VLPackedVideoRequest(
        invocation_id="longvu-video-1",
        media_id="sample-longvu:video-0",
        payload=payload,
        deepstack_features=(payload.features * 2.0, payload.features * 3.0),
        grid=grid,
        source_placeholder_positions=tuple(range(1, 393)),
        prompt_input_ids=torch.tensor([[1, *([8] * 392), 2]]),
        source_attention_mask=torch.ones(1, 394, dtype=torch.long),
        source_video_grid_thw=torch.tensor([[2, 28, 28]], dtype=torch.long),
        source_second_per_grid_ts=torch.tensor([0.5]),
    )
    executor = LongVUVideoExecutor(
        backbone="qwen3_vl",
        model=object(),
        fusion_module=_TinyFusion(),
    )
    response = executor._projector_callback(request)

    assert isinstance(response, Qwen3VLVideoCompressionResponse)
    assert response.compression_result.result_count == 144
    for source, pooled in zip(request.deepstack_features, response.deepstack_features):
        torch.testing.assert_close(
            pooled,
            pool_longvu_source_groups(source, response.compression_result),
        )
    record = executor.runtime_metadata()["algorithm_records"][0]
    assert record["algorithm"] == "longvu"
    assert record["variant"] == "longvu.video"
    assert record["source_count"] == 392
    assert record["result_count"] == 144
    assert record["selected_frame_count"] == 1
    assert record["deepstack_branch_count"] == 2
    assert record["dinov2_features_shape"] == (2, 576, 5)


def test_longvu_reference_sva_has_gradients_and_strict_state_reload() -> None:
    torch = _torch()
    from mllm_comp.backbones._longvu_torch import LongVUFusionModule

    first = LongVUFusionModule(
        raw_hidden_size=3,
        dino_hidden_size=5,
        output_hidden_size=4,
        vision_hidden_size=16,
        dino_interp_size=16,
        query_num=4,
        connector_depth=1,
    )
    raw = torch.randn(2, 16, 3, requires_grad=True)
    dino = torch.randn(2, 16, 5)
    output = first(raw, dino)
    assert tuple(output.shape) == (2, 4, 4)
    output.sum().backward()
    assert raw.grad is not None
    assert any(
        parameter.grad is not None and bool((parameter.grad != 0).any().item())
        for parameter in first.parameters()
    )

    second = LongVUFusionModule(
        raw_hidden_size=3,
        dino_hidden_size=5,
        output_hidden_size=4,
        vision_hidden_size=16,
        dino_interp_size=16,
        query_num=4,
        connector_depth=1,
    )
    second.load_state_dict(first.state_dict(), strict=True)
    torch.testing.assert_close(second(raw.detach(), dino), first(raw.detach(), dino))


def test_longvu_dino_encoder_interpolates_rectangular_qwen_raster() -> None:
    torch = _torch()
    from mllm_comp.backbones._longvu_torch import FrozenDinov2Encoder

    class _RectangularDino(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.anchor = torch.nn.Parameter(torch.zeros(()), requires_grad=False)
            self.config = SimpleNamespace(patch_size=2)

        def forward(self, pixels: Any) -> Any:
            batch, _channels, height, width = pixels.shape
            token_count = (height // 2) * (width // 2)
            patches = torch.arange(
                batch * token_count * 3,
                dtype=pixels.dtype,
                device=pixels.device,
            ).reshape(batch, token_count, 3)
            cls = torch.zeros(batch, 1, 3, dtype=pixels.dtype, device=pixels.device)
            return SimpleNamespace(last_hidden_state=torch.cat((cls, patches), dim=1))

    encoder = object.__new__(FrozenDinov2Encoder)
    encoder.model = _RectangularDino()
    encoder.interp_size = 16
    pixels = torch.zeros(2, 3, 4, 6)

    result = encoder(pixels)

    assert tuple(result.shape) == (2, 16, 3)
    assert bool(torch.isfinite(result).all().item())


@pytest.mark.parametrize("backbone", ("llava_onevision", "qwen25_vl", "qwen3_vl"))
def test_longvu_executor_registers_fusion_before_bridge_and_uninstalls(
    backbone: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pytest.importorskip("torch")
    events: list[tuple[str, str]] = []
    model = SimpleNamespace()
    fusion = _TinyFusion()

    class _Provider:
        def install(self) -> None:
            assert getattr(model, "mllm_comp_longvu_fusion") is fusion
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
            assert callable(callback)
            assert signal_provider is provider
            assert isinstance(site, str)
            events.append((backbone, "register"))
            return _Handle()

        def detach(self) -> None:
            events.append((backbone, "detach"))

        def runtime_metadata(self) -> dict[str, Any]:
            return {"fake_bridge": backbone}

    provider = _Provider()
    monkeypatch.setattr(
        longvu_provider,
        "install_video_bridge",
        lambda _model, selected: (
            _Bridge(),
            selected == backbone,
            f'{selected}:dino_frame_select_sva_fusion',
        ),
    )
    executor = LongVUVideoExecutor(
        backbone=backbone,
        model=model,
        signal_provider=provider,
        fusion_module=fusion,
    )
    executor.install()
    assert executor.runtime_metadata()["fusion_registered_pre_optimizer"] is True
    executor.uninstall()

    assert not hasattr(model, "mllm_comp_longvu_fusion")
    assert events == [
        (backbone, "provider_install"),
        (backbone, "register"),
        (backbone, "remove"),
        (backbone, "detach"),
        (backbone, "provider_uninstall"),
    ]


def test_llava_longvu_provider_attaches_raw_pixels_and_dino_without_image_leak() -> (
    None
):
    torch = _torch()

    class _Dino:
        hidden_size = 5

        def __call__(self, pixels: Any) -> Any:
            assert tuple(pixels.shape) == (2, 3, 8, 8)
            return torch.ones(2, 576, 5, device=pixels.device, dtype=pixels.dtype)

    class _Outer:
        def forward(self, frames: Any) -> Any:
            return torch.arange(
                1,
                int(frames.shape[0]) * 256 * 3 + 1,
                dtype=frames.dtype,
            ).reshape(int(frames.shape[0]), 256, 3)

    class _Model:
        def __init__(self) -> None:
            self.outer = _Outer()

        def get_vision_tower(self) -> Any:
            return self.outer

    model = _Model()
    native = model.outer.forward
    provider = LongVUVideoSignalProvider(
        backbone="llava_onevision",
        model=model,
        dino_encoder=_Dino(),
    )
    provider.install()
    frames = torch.zeros(2, 3, 8, 8)
    model.outer.forward(frames)
    with provider.capture_context(frames=frames) as capture:
        model.outer.forward(frames)
    provider.payload_signals(
        capture,
        token_count=392,
        coordinates=_coordinates(2, side=14),
    )
    base = VideoTokenPayload(
        features=torch.ones(392, 4),
        media_ids=("llava-longvu",),
        video_offsets=(0, 392),
        coordinates=_coordinates(2, side=14),
    )
    enriched = provider.enrich_payload(base, capture)

    assert tuple(enriched.raw_features.shape) == (512, 3)
    assert tuple(enriched.signals["pixels"].shape) == (2, 3, 8, 8)
    assert tuple(enriched.signals["dinov2_features"].shape) == (2, 576, 5)
    assert enriched.metadata["longvu_pixel_source"] == "llava_siglip_frames"
    provider.uninstall()
    assert getattr(model.outer.forward, "__func__", None) is getattr(
        native,
        "__func__",
        None,
    )


@pytest.mark.parametrize(
    ("backbone", "expected_first_raw", "expected_first_pixel"),
    (
        ("qwen25_vl", 49.0, (0.48145466 - 0.485) / 0.229),
        ("qwen3_vl", 1.0, (0.5 - 0.485) / 0.229),
    ),
)
def test_qwen_longvu_provider_restores_raster_and_inverse_patchifies_pixels(
    backbone: str,
    expected_first_raw: float,
    expected_first_pixel: float,
) -> None:
    torch = _torch()

    class _Dino:
        hidden_size = 5

        def __call__(self, pixels: Any) -> Any:
            assert tuple(pixels.shape) == (1, 3, 8, 8)
            return torch.ones(1, 576, 5, dtype=pixels.dtype)

    class _Block:
        def forward(self, hidden_states: Any, **kwargs: Any) -> Any:
            del kwargs
            return hidden_states + 1.0

    block = _Block()
    reverse = torch.arange(3, -1, -1)
    visual = SimpleNamespace(
        blocks=(block,),
        spatial_merge_size=2,
        spatial_merge_unit=4,
        patch_embed=SimpleNamespace(patch_size=2, temporal_patch_size=2),
        get_window_index=lambda _grid: (reverse, torch.tensor([0, 4])),
    )
    model = SimpleNamespace(model=SimpleNamespace(visual=visual))
    provider = LongVUVideoSignalProvider(
        backbone=backbone,
        model=model,
        dino_encoder=_Dino(),
    )
    provider.install()
    grid = torch.tensor([[1, 4, 4]], dtype=torch.long)
    raw = torch.arange(64, dtype=torch.float32).reshape(16, 4)
    with provider.capture_context(grid_thw=grid) as capture:
        block.forward(raw)
    base = VideoTokenPayload(
        features=torch.ones(4, 4),
        media_ids=("qwen-longvu",),
        video_offsets=(0, 4),
        coordinates=_coordinates(1, side=2),
        signals={f"{backbone}.pixel_values_videos": torch.zeros(16, 24)},
    )
    enriched = provider.enrich_payload(base, capture)

    assert tuple(enriched.raw_features.shape) == (16, 4)
    assert float(enriched.raw_features[0, 0]) == expected_first_raw
    assert tuple(enriched.signals["pixels"].shape) == (1, 3, 8, 8)
    assert float(enriched.signals["pixels"][0, 0, 0, 0]) == pytest.approx(
        expected_first_pixel,
        abs=2e-7,
    )
    assert enriched.metadata["longvu_source_normalization"] == (
        "openai_clip" if backbone == "qwen25_vl" else "qwen3_half"
    )
    assert enriched.metadata["longvu_pixel_source"] == ("qwen_patchified_video_inverse")
    provider.uninstall()


def test_longvu_missing_local_dino_fails_before_patching() -> None:
    provider = LongVUVideoSignalProvider(
        backbone="llava_onevision",
        model=object(),
    )
    with pytest.raises(BackendIntegrationError, match="longvu_dino_path"):
        provider.prepare()
