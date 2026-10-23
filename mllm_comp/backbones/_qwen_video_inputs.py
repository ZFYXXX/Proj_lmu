"""Physical Qwen video input conversions shared by auxiliary encoders."""

from __future__ import annotations

from typing import Any

from mllm_comp.compressors.video import VideoTokenCoordinate
from mllm_comp.core.errors import BackendIntegrationError


def restore_qwen_video_raster(
    raw: Any,
    grid: Any,
    visual: Any,
    *,
    backbone: str,
    merge_size: int,
    merge_unit: int,
    method_name: str,
) -> tuple[Any, tuple[VideoTokenCoordinate, ...]]:
    """Undo Qwen2.5 window order, then expand spatial merge groups to raster order."""
    import torch

    temporal, height, width = tuple(int(value) for value in grid[0].tolist())
    raw_count = temporal * height * width
    if (
        merge_unit != merge_size * merge_size
        or height % merge_size
        or width % merge_size
        or getattr(raw, "ndim", None) != 2
        or int(raw.shape[0]) != raw_count
    ):
        raise BackendIntegrationError(
            f"Qwen {method_name} raw features do not match video_grid_thw"
        )
    feature_count = raw_count // merge_unit
    grouped = raw.reshape(feature_count, merge_unit, -1)
    if backbone == "qwen25_vl":
        get_window = getattr(visual, "get_window_index", None)
        if not callable(get_window):
            raise BackendIntegrationError(
                f"Qwen2.5-VL {method_name} requires native window indices"
            )
        window_index, _ = get_window(grid)
        if int(window_index.numel()) != feature_count:
            raise BackendIntegrationError(
                f"Qwen2.5-VL {method_name} window indices changed"
            )
        grouped = grouped.index_select(0, torch.argsort(window_index).to(grouped.device))
    raster = (
        grouped.reshape(
            temporal,
            height // merge_size,
            width // merge_size,
            merge_size,
            merge_size,
            -1,
        )
        .permute(0, 1, 3, 2, 4, 5)
        .reshape(raw_count, int(raw.shape[-1]))
    )
    coordinates = tuple(
        VideoTokenCoordinate(t=t, h=h, w=w, tubelet_index=t)
        for t in range(temporal)
        for h in range(height)
        for w in range(width)
    )
    return raster, coordinates


def qwen_video_pixels_to_siglip(
    values: Any,
    grid: Any,
    *,
    backbone: str,
    patch_size: int,
    temporal_patch: int,
    merge_size: int,
    method_name: str,
) -> Any:
    """Inverse patchify native pixels and average each tubelet in SigLIP space."""
    import torch

    temporal, height, width = tuple(int(value) for value in grid[0].tolist())
    expected_width = 3 * temporal_patch * patch_size * patch_size
    if (
        int(values.shape[0]) != temporal * height * width
        or int(values.shape[1]) != expected_width
        or height % merge_size
        or width % merge_size
    ):
        raise BackendIntegrationError(
            f"Qwen {method_name} patchified pixels do not match video_grid_thw"
        )
    patches = values.reshape(
        temporal,
        height // merge_size,
        width // merge_size,
        merge_size,
        merge_size,
        3,
        temporal_patch,
        patch_size,
        patch_size,
    )
    frames = (
        patches.permute(0, 6, 5, 1, 3, 7, 2, 4, 8)
        .contiguous()
        .reshape(
            temporal, temporal_patch, 3, height * patch_size, width * patch_size
        )
    )
    source_mean, source_std = (
        ((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
        if backbone == "qwen25_vl"
        else ((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    )
    mean = torch.tensor(source_mean, device=frames.device, dtype=frames.dtype).view(
        1, 1, 3, 1, 1
    )
    std = torch.tensor(source_std, device=frames.device, dtype=frames.dtype).view(
        1, 1, 3, 1, 1
    )
    frames_01 = (frames * std + mean).mean(dim=1)
    frames_01 = torch.nn.functional.interpolate(
        frames_01, size=(384, 384), mode="bilinear", align_corners=False
    )
    return (frames_01 - 0.5) / 0.5
